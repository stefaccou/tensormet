"""
cp_decomposition.py — CPDecomposition: inference/eval wrapper for CP models.

EXPERIMENTAL (see reviews/CP_IMPLEMENTATION_PLAN.md, Phase 3).

API-compatible with the subset of ``TuckerDecomposition`` that the evaluation
stack consumes, so the in-loop semantic eval (``similarity.evaluate_sample``),
SimLex and the dimension-consistency judge work unchanged:

    factors / vocab / roles / shared_factors
    get_role_index, check_vocab, fetch_latents, fetch_single_latent
    batch_excluded_role_vector           (the whole evaluate_sample contract)
    get_top_words_for_dimension, get_top_dimensions_for_word
    get_most_similar_elements, get_expected_element, get_top_combinations
    score_scalar, excluded_role_vector, included_role_vector
    to_cupy, load_from_disk, update_from_path
    core (compat property: materializes diag_N(λ) on demand, size-guarded)

All hot paths are CP-native and O(R) instead of O(R^N) — the role the Tucker
core contraction plays is a single (R,) Hadamard-weighted vector here.

Deliberately standalone (small helpers duplicated from tucker_tensor rather
than refactored into a shared base class) so nothing in the main package
changes for this experimental feature; if CP graduates, extracting the common
base is the natural follow-up (plan Phase 3, first bullet).
"""
from __future__ import annotations

import json
import os
import pickle
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch

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
from tensormet.naming import ALL_METHODS, candidate_stems, vocab_filename, vocab_filename_legacy

cp, cpx_sparse = make_lazy_cupy_pair()

# Guard for the compatibility ``core`` materialization: R^N elements above
# this raise instead of silently allocating (order-4 blowup, plan Phase 3).
_CORE_MATERIALIZE_MAX_ELEMENTS = 20_000_000  # ~160 MB fp64


def _role_index(role: str, role_names: List[str]) -> int:
    try:
        return role_names.index(role)
    except ValueError as e:
        raise ValueError(f"role must be one of {set(role_names)}") from e


def _voc_list_key(role: str) -> str:
    return f"vocab_{role}"


class CPDecomposition:
    """Encapsulates a CP decomposition (weights λ and factors) plus vocabulary,
    providing the same scoring/inspection interface as ``TuckerDecomposition``.

    Model: X̂ = Σ_r λ_r · a_r(1) ∘ … ∘ a_r(N). ``weights`` is the (R,) λ
    vector; ``factors[n]`` is (I_n, R).
    """

    def __init__(self, weights, factors, vocab: dict,
                 shared_factors: set | None = None,
                 roles: Optional[List[str]] = None):
        self.weights = weights
        self.factors = factors
        self.vocab = vocab
        self.shared_factors = shared_factors or set()
        self.roles = roles if roles is not None else extract_roles_from_vocab(self.vocab)
        self.decomp_path = None
        self._core_cache = None

    # --- compat: core as materialized superdiagonal ---------------------
    @property
    def core(self):
        """Dense diag_N(λ) — compatibility fallback for code written against
        the Tucker core (e.g. the judge reads ``core.shape``). O(R^N) memory:
        cached after first access, refused above a size guard."""
        if self._core_cache is None:
            w = _to_np(self.weights)
            R = int(w.shape[0])
            N = len(self.factors)
            if R ** N > _CORE_MATERIALIZE_MAX_ELEMENTS:
                raise MemoryError(
                    f"Materializing the CP diagonal core would need {R}^{N} "
                    f"elements (> {_CORE_MATERIALIZE_MAX_ELEMENTS}); use the "
                    f"CP-native methods (weights/…_role_vector) instead."
                )
            G = np.zeros((R,) * N, dtype=w.dtype)
            G[tuple([np.arange(R)] * N)] = w
            self._core_cache = G
        return self._core_cache

    def _core_np(self):
        return self.core

    def _weights_np(self) -> np.ndarray:
        return _to_np(self.weights)

    def get_role_index(self, role: str) -> int:
        return _role_index(role, self.roles)

    def get_dims(self):
        """Mode dimensions of the reconstructed tensor, i.e. (N_0, ..., N_{k-1})."""
        return tuple(int(f.shape[0]) for f in self.factors)

    def get_rank(self, role=None):
        """CP rank — shared by every factor; the role argument is for API parity."""
        if role is None:
            role = self.roles[0]
        return int(self.factors[_role_index(role, self.roles)].shape[1])

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
                       ) -> "CPDecomposition":
        """Load a precomputed CP decomposition from disk.

        Mirrors ``TuckerDecomposition.load_from_disk`` but scans the CP
        filename family (``…_CP{order}D_…``, see naming.py) and unpacks
        ``tensorly.cp_tensor.CPTensor`` payloads (weights, factors).
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
            max_nnz=max_nnz,
            decomposition="cp",
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
                    file_prefix = stem
                    iterations = highest_iter
                    break
            if file_prefix is None:
                raise FileNotFoundError(
                    f"Could not find any CP decomposition files in {decomp_dir} "
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
            raise FileNotFoundError(f"Missing CP decomposition file: {decomp_path}")

        with open(vocab_path, "rb") as f:
            vocab = pickle.load(f)
        roles = [k[len("vocab_"):] for k in vocab.keys() if k.startswith("vocab_")]

        payload = torch_or_pickle_load(decomp_path, map_location=map_location)
        weights, factors = payload  # CPTensor or plain (weights, factors) tuple

        runs_path = os.path.join(decomp_dir, "runs.jsonl")
        if os.path.exists(runs_path):
            with open(runs_path, "r") as f:
                for line in f:
                    run_info = json.loads(line)
                    if run_info.get("results", {}).get("model_path") == decomp_path:
                        print("Loaded CP decomposition with the following parameters:")
                        for key, value in run_info.items():
                            print(f"  {key}: {value}")
                        break

        instance = cls(weights, list(factors), vocab, shared_factors=parsed_shared, roles=roles)
        instance.decomp_path = Path(decomp_path)
        return instance

    def update_from_path(self, path=None):
        resolved = resolve_checkpoint_path(path, self.decomp_path)
        tensor = torch.load(resolved, map_location="cpu", weights_only=False)
        weights, factors = tensor
        self.weights = np_dispatch(weights)
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

    def fetch_single_latent(self, element, role) -> np.ndarray:
        el_idx = self.vocab[voc_index(role)][element]
        return _to_np(self.factors[self.get_role_index(role)][el_idx])

    def to_cupy(self):
        """Prepare for inference by moving to GPU."""
        if isinstance(self.weights, torch.Tensor):
            self.weights = cp.array(_to_np(self.weights))
        for i, f in enumerate(self.factors):
            if isinstance(f, torch.Tensor):
                self.factors[i] = cp.array(_to_np(f))

    # --- Scoring (CP-native, all O(R)) -----------------------------------
    def score_scalar(self, triple: Tuple[str, ...]) -> float:
        """Scalar reconstruction score Σ_r λ_r Π_n latent_n[r]."""
        prod = self._weights_np().copy()
        for latent in self.fetch_latents(triple):
            prod *= latent
        return float(np.sum(prod))

    def excluded_role_vector(self, triple: Tuple[str, ...], role: str) -> np.ndarray:
        """λ ⊛ ⊛_{m≠target} latent_m — the (R,) 'prediction' vector; downstream
        ``factor @ v`` ranking is identical to the Tucker case."""
        target_idx = self.get_role_index(role)
        out = self._weights_np().copy()
        latents = self.fetch_latents(triple)
        for i in range(len(self.roles)):
            if i != target_idx:
                out *= latents[i]
        return out

    def included_role_vector(self, triple: Tuple[str, ...], role: str) -> np.ndarray:
        """λ ⊛ ⊛_all latent_m — per-dimension contribution of the full tuple.
        (Same vector for every role in CP; the role argument is kept for API
        parity with TuckerDecomposition.)"""
        out = self._weights_np().copy()
        for latent in self.fetch_latents(triple):
            out *= latent
        return out

    def batch_excluded_role_vector(self,
                                   valid_indices: torch.Tensor,
                                   role_name: str) -> torch.Tensor:
        """Batched excluded-role vectors: out[p, r] = λ_r Π_{i≠target} F_i[idx_pi, r].

        This single method is the whole contract ``similarity.evaluate_sample``
        needs, so in-loop semantic eval lights up as soon as it exists
        (plan Phase 3). Pure gathered-row Hadamard — no R^N contraction.
        """
        target_idx = self.get_role_index(role_name)
        device = self.factors[0].device
        out = None
        for i in range(len(self.roles)):
            if i == target_idx:
                continue
            rows = self.factors[i][valid_indices[:, i]]  # (n, R)
            out = rows if out is None else out * rows
        w = self.weights
        if not isinstance(w, torch.Tensor):
            w = torch.as_tensor(_to_np(w))
        w = w.to(device)
        return out * w  # broadcast (n, R) * (R,)

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

    def get_top_dimensions_for_word(self, word: str, role=None, top_k: int = 10,
                                    return_words=False):
        if role is None:
            role = self.roles[0]
        latent = torch.tensor(self.fetch_single_latent(word, role))
        if top_k == "full":
            top_k = len(latent)
        scores, dims = torch.topk(latent, top_k)
        if return_words:
            # variant in which we return the representative word as well
            return [
                (int(dim), float(score), self.get_top_words_for_dimension(role, dim, 1)[0][0])
                for dim, score in zip(dims, scores)
            ]
        return [(int(dim), float(score)) for dim, score in zip(dims, scores)]

    def get_most_similar_elements(self, element, role=None, top_k=5):
        """Most similar elements by cosine over factor rows; a tuple input uses
        the contextualised (included) CP vector, a string the factor row."""
        if role is None:
            role = self.roles[0]
        if isinstance(element, tuple):
            latent = self.included_role_vector(element, role=role)
        elif isinstance(element, str):
            latent = self.fetch_single_latent(element, role=role)
        elif isinstance(element, np.ndarray):
            latent = element
        else:
            raise ValueError("Must be tuple, str or ndarray")

        i = self.get_role_index(role)
        F = _to_np(self.factors[i])

        eps = 1e-12
        F_norm = np.maximum(np.linalg.norm(F, axis=1), eps)
        G_norm = max(np.linalg.norm(latent), eps)
        similarities = (F @ latent) / (F_norm * G_norm)
        if top_k == "full":
            top_k = len(F)
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
            role_act = factor[idx, :]
            cos_sim = np_sim(role_act, latents[index])
            results.append({"token": role_str,
                            "score": float(scores[idx]),
                            "activation_cosine": float(cos_sim)})

        if verbose:
            print(f"Top {k} expected {role}s based on the CP weights:")
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

        CP-native: scores = F_a · diag(λ ⊛ fixed_latent) · F_bᵀ — a single
        (R,)-weighted rank-space product instead of a core contraction.
        """
        fixed_idx = self.get_role_index(fixed_role)
        other_idxs = [i for i in range(len(self.roles)) if i != fixed_idx]
        if len(other_idxs) > 2:
            raise NotImplementedError(
                "get_top_combinations currently supports at most 2 free roles "
                f"(found {len(other_idxs)} for order-{len(self.roles)} tensor)."
            )

        v_latent = self.fetch_single_latent(fixed_element, fixed_role)
        w = self._weights_np() * v_latent  # (R,) — the whole "core" contraction

        role_names_free = [self.roles[i] for i in other_idxs]
        factors_free: list[np.ndarray] = []
        vocab_lists_free: list[list[str]] = []
        for role in role_names_free:
            factor = _to_np(self.factors[self.get_role_index(role)])
            vocab_list = list(self.vocab[_voc_list_key(role)])
            if restrict_roles and role in restrict_roles:
                r2i = self.vocab[voc_index(role)]
                keep_words = [wd for wd in restrict_roles[role] if wd in r2i]
                keep_idxs = [r2i[wd] for wd in keep_words]
                factor = factor[keep_idxs]
                vocab_list = keep_words
            if exclude_oov and oov_token in vocab_list:
                oov_idx = vocab_list.index(oov_token)
                keep_mask = [i for i in range(len(vocab_list)) if i != oov_idx]
                factor = factor[keep_mask]
                vocab_list = [wd for wd in vocab_list if wd != oov_token]
            factors_free.append(factor)
            vocab_lists_free.append(vocab_list)

        F_a, F_b = factors_free
        scores = (F_a * w[None, :]) @ F_b.T

        n_a, n_b = scores.shape
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
            combo: list[str] = [None] * len(self.roles)  # type: ignore[list-item]
            combo[fixed_idx] = fixed_element
            combo[other_idxs[0]] = vocab_a[i]
            combo[other_idxs[1]] = vocab_b[j]
            results.append((tuple(combo), float(scores[i, j])))
        return results
