"""
tt_decomposition.py — inference/eval wrapper for Tucker-TT hybrid models.

EXPERIMENTAL (see README.md in this directory).

``TuckerTTDecomposition`` is API-compatible with the subset of
``TuckerDecomposition`` the evaluation stack consumes, so the in-loop semantic
eval (``similarity.evaluate_sample``), SimLex and the dimension-consistency
judge work unchanged:

    factors / vocab / roles / shared_factors
    get_role_index, get_dims, check_vocab, fetch_latents, fetch_single_latent
    batch_excluded_role_vector           (the whole evaluate_sample contract)
    get_top_words_for_dimension, get_top_dimensions_for_word
    get_most_similar_elements, get_expected_element, get_top_combinations
    score_scalar, excluded_role_vector, included_role_vector
    to_cupy, load_from_disk, update_from_path
    core (compat property: materializes the dense R^N core, size-guarded)

Every factor-level method is identical to Tucker's — the factors *are* Tucker
factors, and a latent dimension of a role still means what it meant. Only the
core contractions change: a chain of matrix products instead of one R^N
einsum, which is what makes them O(N ρ² R) instead of O(R^N).

Deliberately standalone (small helpers duplicated rather than refactored into
a shared base class), matching experimental/CP, so nothing in the main package
changes for this experimental feature.
"""
from __future__ import annotations

import json
import os
import pickle
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch

from tensormet.naming import ALL_METHODS, candidate_stems, vocab_filename, vocab_filename_legacy
from tensormet.utils import (
    DATA_DIR,
    _to_np,
    extract_roles_from_vocab,
    make_lazy_cupy_pair,
    np_dispatch,
    np_sim,
    readonly_dispatch,
    resolve_checkpoint_path,
    torch_or_pickle_load,
    voc_index,
)
from tensormet.experimental.TT_hybrid.tt_chain import (
    contract, left_envs, right_envs, site_grad, sites, to_dense_core,
)

cp, cpx_sparse = make_lazy_cupy_pair()

# Guard for the dense-core compatibility property: Π R_k elements above this
# raise instead of silently allocating (an order-5 rank-100 core is 40 GB —
# the whole reason this format exists).
_CORE_MATERIALIZE_MAX_ELEMENTS = 20_000_000  # ~160 MB fp64


def _role_index(role: str, role_names: List[str]) -> int:
    try:
        return role_names.index(role)
    except ValueError as e:
        raise ValueError(f"role must be one of {set(role_names)}") from e


def _voc_list_key(role: str) -> str:
    return f"vocab_{role}"


class TuckerTTTensor:
    """``(tt_cores, factors)`` payload — the hybrid's analogue of tensorly's
    ``TuckerTensor``. Deliberately validation-free and iterable, so every
    ``core, factors = tensor`` site in the training loop and in launch.py works
    unchanged."""

    def __init__(self, tt_tucker_tensor):
        tt_cores, factors = tt_tucker_tensor
        self.tt_cores = list(tt_cores)
        self.factors = list(factors)

    def __iter__(self):
        return iter((self.tt_cores, self.factors))

    def __getitem__(self, index):
        return (self.tt_cores, self.factors)[index]

    def __len__(self):
        return 2

    def __repr__(self):
        return (f"TuckerTTTensor(tt_cores={[tuple(C.shape) for C in self.tt_cores]}, "
                f"factors={[tuple(f.shape) for f in self.factors]})")


class TuckerTTDecomposition:
    """Encapsulates a Tucker-TT hybrid decomposition (TT cores + Tucker factors)
    plus the vocabulary, with the same scoring/inspection interface as
    ``TuckerDecomposition``.

    Model: X̂[i_0…i_{N-1}] = Σ_r G[r] Π_n A_n[i_n, r_n] with the core
    G[r] = C_0[:, r_0, :] · … · C_{N-1}[:, r_{N-1}, :].
    """

    def __init__(self, tt_cores, factors, vocab: dict,
                 shared_factors: set | None = None,
                 roles: Optional[List[str]] = None):
        self.tt_cores = list(tt_cores)
        self.factors = list(factors)
        self.vocab = vocab
        self.shared_factors = shared_factors or set()
        self.roles = roles if roles is not None else extract_roles_from_vocab(self.vocab)
        self.decomp_path = None
        self._core_cache = None

    # --- compat: dense core on demand ------------------------------------
    @property
    def core(self):
        """The dense R^N Tucker core, reconstructed from the chain. Cached
        after first access, refused above a size guard — prefer the TT-native
        methods below, which never need it."""
        if self._core_cache is None:
            shape = tuple(int(C.shape[1]) for C in self.tt_cores)
            n_elements = int(np.prod(shape, dtype=object))
            if n_elements > _CORE_MATERIALIZE_MAX_ELEMENTS:
                raise MemoryError(
                    f"Materializing the dense core would need {n_elements} elements "
                    f"(shape {shape}, > {_CORE_MATERIALIZE_MAX_ELEMENTS}); use the "
                    f"TT-native methods (score_scalar / …_role_vector / "
                    f"get_top_combinations) instead."
                )
            self._core_cache = to_dense_core(self._tt_np(), np)
        return self._core_cache

    def _core_np(self):
        return self.core

    def _tt_np(self) -> List[np.ndarray]:
        return [_to_np(C) for C in self.tt_cores]

    def get_role_index(self, role: str) -> int:
        return _role_index(role, self.roles)

    def get_dims(self):
        """Mode dimensions of the reconstructed tensor, i.e. (N_0, ..., N_{k-1})."""
        return tuple(int(f.shape[0]) for f in self.factors)

    def bond_dims(self):
        """ρ_0..ρ_N — the information bottleneck at each cut of the chain."""
        return tuple([int(self.tt_cores[0].shape[0])]
                     + [int(C.shape[2]) for C in self.tt_cores])

    # --- Construction and loading ---------------------------------------
    @classmethod
    def load_from_disk(cls,
                       dataset: str = "fineweb-en",
                       method: str = "siiSoftPlus",
                       divergence: str = "kl",
                       dims: "int | tuple[int, ...]" = 4000,
                       rank: int = 100,
                       order: int = 3,
                       iterations: int | None = None,
                       shared_factors: bool | set | str = False,
                       map_location: str = "cpu",
                       name: Optional[str] = None,
                       tier1: bool = False,
                       subsample_frac: float = 1.0,
                       max_nnz: Optional[int] = None,
                       ) -> "TuckerTTDecomposition":
        """Load a precomputed Tucker-TT decomposition.

        Mirrors ``TuckerDecomposition.load_from_disk`` but scans the TT filename
        family (``…_TT{order}D_…``, see naming.py) and unpacks
        ``TuckerTTTensor`` payloads.
        """
        if method not in ALL_METHODS:
            raise ValueError(f"method must be one of {set(ALL_METHODS)}")
        base = os.path.join(DATA_DIR, "tensors", dataset)
        base = readonly_dispatch(base, tier1)

        parsed_shared = None
        if shared_factors == "all":
            parsed_shared = {(i, j) for i in range(order) for j in range(i + 1, order)}
        elif shared_factors is True:
            parsed_shared = {(1, 2)}
        elif isinstance(shared_factors, set) and shared_factors:
            for item in shared_factors:
                if not (isinstance(item, tuple) and len(item) == 2):
                    raise TypeError(
                        f"shared_factors must be a set of 2-tuples, got item {item!r}"
                    )
            parsed_shared = shared_factors

        # Vocabulary is decomposition-agnostic: same files as Tucker.
        _vdir = os.path.join(base, "vocabularies")
        vocab_path_new = os.path.join(_vdir, vocab_filename(order, dims, shared_factors=parsed_shared))
        vocab_path_old = os.path.join(_vdir, vocab_filename_legacy(dims, shared_factors=parsed_shared, order=order))
        if os.path.exists(vocab_path_new):
            vocab_path = vocab_path_new
        elif os.path.exists(vocab_path_old):
            vocab_path = vocab_path_old
        else:
            raise FileNotFoundError(f"Missing vocab file. Checked {vocab_path_new} and {vocab_path_old}")

        decomp_dir = os.path.join(base, "decomposition")
        stems = candidate_stems(
            divergence, method, order, dims, rank,
            name=name, shared_factors=parsed_shared, subsample_frac=subsample_frac,
            max_nnz=max_nnz, decomposition="tt",
        )

        def _find_highest_iter(prefix: str) -> int:
            highest = -1
            if os.path.exists(decomp_dir):
                for filename in os.listdir(decomp_dir):
                    if filename.startswith(prefix) and filename.endswith("i.pt"):
                        iter_str = filename[len(prefix):-len("i.pt")]
                        if iter_str.isdigit():
                            highest = max(highest, int(iter_str))
            return highest

        file_prefix = None
        if not iterations:
            for stem in stems:
                highest_iter = _find_highest_iter(stem)
                if highest_iter != -1:
                    file_prefix, iterations = stem, highest_iter
                    break
            if file_prefix is None:
                raise FileNotFoundError(
                    f"Could not find any Tucker-TT decomposition in {decomp_dir} "
                    f"matching {stems}"
                )
        else:
            for stem in stems:
                if os.path.exists(os.path.join(decomp_dir, f"{stem}{iterations}i.pt")):
                    file_prefix = stem
                    break
            if file_prefix is None:
                file_prefix = stems[0]

        decomp_path = os.path.join(decomp_dir, f"{file_prefix}{iterations}i.pt")
        if not os.path.exists(decomp_path):
            raise FileNotFoundError(f"Missing decomposition file: {decomp_path}")

        with open(vocab_path, "rb") as f:
            vocab = pickle.load(f)
        roles = [k[len("vocab_"):] for k in vocab.keys() if k.startswith("vocab_")]

        tt_cores, factors = torch_or_pickle_load(decomp_path, map_location=map_location)

        runs_path = os.path.join(decomp_dir, "runs.jsonl")
        if os.path.exists(runs_path):
            with open(runs_path, "r") as f:
                for line in f:
                    run_info = json.loads(line)
                    if run_info.get("results", {}).get("model_path") == decomp_path:
                        print("Loaded Tucker-TT decomposition with the following parameters:")
                        for key, value in run_info.items():
                            print(f"  {key}: {value}")
                        break

        instance = cls(tt_cores=list(tt_cores), factors=list(factors), vocab=vocab,
                       shared_factors=parsed_shared, roles=roles)
        instance.decomp_path = Path(decomp_path)
        return instance

    def update_from_path(self, path=None):
        resolved = resolve_checkpoint_path(path, self.decomp_path)
        tensor = torch.load(resolved, map_location="cpu", weights_only=False)
        tt_cores, factors = tensor
        self.tt_cores = np_dispatch(list(tt_cores))
        self.factors = np_dispatch(list(factors))
        self._core_cache = None
        self.decomp_path = resolved

    # --- Vocab / latent access (same semantics as TuckerDecomposition) ---
    def check_vocab(self, triple: Tuple[str, ...], return_type=bool) -> bool | tuple:
        in_roles = [triple[i] in self.vocab[voc_index(self.roles[i])] for i in range(len(self.roles))]
        if return_type == tuple:
            return tuple(in_roles)
        return all(in_roles)

    def fetch_latents(self, triple: Tuple[str, ...]) -> Tuple[np.ndarray, ...]:
        return tuple(
            self.fetch_single_latent(triple[i], self.roles[i])
            for i in range(len(self.roles))
        )

    def fetch_single_latent(self, element, role=None) -> np.ndarray:
        if role is None:
            role = self.roles[0]
        el_idx = self.vocab[voc_index(role)][element]
        return _to_np(self.factors[self.get_role_index(role)][el_idx])

    def to_cupy(self):
        """Prepare for inference by moving to GPU."""
        for store in (self.tt_cores, self.factors):
            for i, a in enumerate(store):
                if isinstance(a, torch.Tensor):
                    store[i] = cp.array(_to_np(a))

    # --- Scoring (TT-native: a chain of matrix products, never R^N) ------
    def score_scalar(self, triple: Tuple[str, ...]) -> float:
        """Scalar reconstruction score ⟨G, a∘b∘c…⟩."""
        return float(contract(self._tt_np(), list(self.fetch_latents(triple)), [], np))

    def excluded_role_vector(self, triple: Tuple[str, ...], role: str) -> np.ndarray:
        """Activations predicted for the target role's dimensions from the other
        attested elements — the chain with that one rank leg left open. Same
        semantics (and the same downstream ``factor @ v`` ranking) as Tucker's."""
        target = self.get_role_index(role)
        return contract(self._tt_np(), list(self.fetch_latents(triple)), [target], np)

    def included_role_vector(self, triple: Tuple[str, ...], role: str) -> np.ndarray:
        """Per-dimension contribution of the target role inside the full tuple;
        the excluded vector gated by the role's own latent (exactly Tucker's
        ``einsum(core, all latents -> role letter)``)."""
        target = self.get_role_index(role)
        return self.excluded_role_vector(triple, role) * self.fetch_latents(triple)[target]

    def batch_excluded_role_vector(self,
                                   valid_indices: torch.Tensor,
                                   role_name: str) -> torch.Tensor:
        """Batched excluded-role vectors on GPU — the whole contract
        ``similarity.evaluate_sample`` needs. One left and one right sweep meet
        at the target site; the target's own latent is never gathered."""
        target = self.get_role_index(role_name)
        device = self.factors[0].device
        tt_cores = [C if isinstance(C, torch.Tensor) else torch.as_tensor(_to_np(C))
                    for C in self.tt_cores]
        tt_cores = [C.to(device=device, dtype=self.factors[0].dtype) for C in tt_cores]

        mats = [self.factors[i][valid_indices[:, i]] for i in range(len(self.roles))]
        S = sites(tt_cores, mats, torch, skip=target)
        return site_grad(left_envs(S, torch)[target], tt_cores[target],
                         right_envs(S, torch)[target + 1], torch)

    # --- Inspection methods (factor-only; identical logic to Tucker) -----
    def get_top_words_for_dimension(self, role: str, dim_index: int, top_k: int = 10):
        """Top-k words with highest loading on one latent dimension of a role."""
        factor_idx = self.get_role_index(role)
        dim_values = _to_np(self.factors[factor_idx])[:, dim_index]
        scores, indices = torch.topk(torch.tensor(dim_values), top_k)
        vocab_list = self.vocab[_voc_list_key(role)]
        return [
            (vocab_list[idx.item()], score.item())
            for idx, score in zip(indices, scores)
        ]

    def get_top_dimensions_for_word(self, word: str, role: str, top_k: int = 10):
        latent = torch.tensor(self.fetch_single_latent(word, role))
        scores, dims = torch.topk(latent, top_k)
        return [(int(dim), float(score)) for dim, score in zip(dims, scores)]

    def get_most_similar_elements(self, element, role, top_k=5):
        """Most similar elements by cosine over factor rows; a tuple input uses
        the contextualised (included) vector, a string the factor row."""
        if isinstance(element, tuple):
            latent = self.included_role_vector(element, role=role)
        elif isinstance(element, str):
            latent = self.fetch_single_latent(element, role=role)
        elif isinstance(element, np.ndarray):
            latent = element
        else:
            raise ValueError("Must be tuple, str or ndarray")

        F = _to_np(self.factors[self.get_role_index(role)])
        eps = 1e-12
        F_norm = np.maximum(np.linalg.norm(F, axis=1), eps)
        G_norm = max(np.linalg.norm(latent), eps)
        similarities = (F @ latent) / (F_norm * G_norm)
        top_idx = np.argsort(-similarities)[:top_k]
        r2i = voc_index(role)
        return [next(k for k, v in self.vocab[r2i].items() if v == idx) for idx in top_idx]

    def get_expected_element(self, target_tuple: Tuple[str, ...], role: str,
                             verbose: bool = True, method: str = "excluded",
                             metric: str = "dot", k=5):
        """Rank vocabulary items for a role given the rest of the tuple —
        same semantics as TuckerDecomposition.get_expected_element."""
        index = self.get_role_index(role)
        r2i = voc_index(role)
        latents = self.fetch_latents(target_tuple)
        if method == "excluded":
            G_item = self.excluded_role_vector(target_tuple, role=role)
        elif method == "included":
            G_item = self.included_role_vector(target_tuple, role=role)
        else:
            raise NotImplementedError

        factor = _to_np(self.factors[index])
        if metric == "cosine":
            eps = 1e-12
            factor_norm = np.maximum(np.linalg.norm(factor, axis=1), eps)
            G_item_norm = max(np.linalg.norm(G_item), eps)
            scores = (factor @ G_item) / (factor_norm * G_item_norm)
        elif metric == "dot":
            scores = factor @ G_item
        else:
            raise ValueError("metric must be either 'dot' or 'cosine'")

        top_k_indices = np.argsort(scores)[-k:][::-1]
        results = []
        for idx in top_k_indices:
            role_str = next(key for key, v in self.vocab[r2i].items() if v == idx)
            cos_sim = np_sim(factor[idx, :], latents[index])
            results.append({"token": role_str,
                            "score": float(scores[idx]),
                            "activation_cosine": float(cos_sim)})

        if verbose:
            print(f"Top {k} expected {role}s based on the TT core:")
            for r in results:
                print(f"{role.capitalize()}: {r['token']}, "
                      f"Score ({metric}): {r['score']:.4f}, "
                      f"Cosine sim with target {role} activations: {r['activation_cosine']:.4f}")
            return None
        return results

    def get_top_combinations(
            self,
            fixed_element: str,
            fixed_role: str,
            top_k: int = 10,
            restrict_roles: Optional[dict[str, list[str]]] = None,
            exclude_oov: bool = True,
            oov_token: str = "~",
    ) -> list[tuple[tuple, float]]:
        """Top-scoring completions with one role fixed.

        TT-native: the chain is contracted with the fixed role's latent and the
        two free rank legs left open, giving the same (R_a, R_b) matrix the
        Tucker version gets from its core contraction.
        """
        fixed_idx = self.get_role_index(fixed_role)
        other_idxs = [i for i in range(len(self.roles)) if i != fixed_idx]
        if len(other_idxs) > 2:
            raise NotImplementedError(
                "get_top_combinations currently supports at most 2 free roles "
                f"(found {len(other_idxs)} for order-{len(self.roles)} tensor)."
            )

        latents: list = [None] * len(self.roles)
        latents[fixed_idx] = self.fetch_single_latent(fixed_element, fixed_role)
        G_fixed = contract(self._tt_np(), latents, other_idxs, np)

        role_names_free = [self.roles[i] for i in other_idxs]
        factors_free: list[np.ndarray] = []
        vocab_lists_free: list[list[str]] = []
        for role in role_names_free:
            factor = _to_np(self.factors[self.get_role_index(role)])
            vocab_list = list(self.vocab[_voc_list_key(role)])
            if restrict_roles and role in restrict_roles:
                r2i = self.vocab[voc_index(role)]
                keep_words = [w for w in restrict_roles[role] if w in r2i]
                factor = factor[[r2i[w] for w in keep_words]]
                vocab_list = keep_words
            if exclude_oov and oov_token in vocab_list:
                keep = [i for i, w in enumerate(vocab_list) if w != oov_token]
                factor = factor[keep]
                vocab_list = [vocab_list[i] for i in keep]
            factors_free.append(factor)
            vocab_lists_free.append(vocab_list)

        F_a, F_b = factors_free
        scores = F_a @ G_fixed @ F_b.T

        n_b = scores.shape[1]
        flat = scores.ravel()
        if top_k >= flat.size:
            top_flat = np.argsort(-flat)
        else:
            part = np.argpartition(flat, -top_k)[-top_k:]
            top_flat = part[np.argsort(-flat[part])]

        vocab_a, vocab_b = vocab_lists_free
        results = []
        for flat_idx in top_flat:
            i, j = divmod(int(flat_idx), n_b)
            combo: list = [None] * len(self.roles)
            combo[fixed_idx] = fixed_element
            combo[other_idxs[0]] = vocab_a[i]
            combo[other_idxs[1]] = vocab_b[j]
            results.append((tuple(combo), float(scores[i, j])))
        return results


__all__ = ["TuckerTTDecomposition", "TuckerTTTensor"]
