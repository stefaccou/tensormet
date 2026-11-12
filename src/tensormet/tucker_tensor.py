from __future__ import annotations
import os
import pickle

import torch
import json
import math
import numpy as np
from typing import ClassVar, List, Optional, Union, Tuple
from datetime import datetime, UTC
from pathlib import Path

from tensormet.config import RunConfig, infer_ngram_order
from tensormet.utils import (DATA_DIR,
                            date_run,
                            torch_or_pickle_load,
                            readonly_dispatch,
                            tree_to_device,
                            ThreadBudget,
                            shared_factor_suffix,
                            nontrivial_linked_groups,
                            voc_index,
                            extract_roles_from_vocab,
                            einsum_letters,
                            SparseCOOTensor,
                            make_lazy_cupy_pair,
                            np_dispatch,
                            np_sim,
                            resolve_checkpoint_path,
                            resolve_shared_factors,
                            to_np,
                   )
from tensormet.sparse_ops import (
    CoordCOO,
    block_encoding_fits,
)
from tensormet.naming import (
    ALL_METHODS,
    candidate_stems,
    vocab_filename as _vocab_filename,
    vocab_filename_legacy as _vocab_filename_legacy,
    populated_filename,
    populated_filename_legacy,
)

cp, cpx_sparse = make_lazy_cupy_pair()


# Commit 1252fa3: tied factors moved from a per-mode update + copy (which applies
# the correction once per leg) to the pooled tucker_tied_factor_update. Runs
# started before this are usable, but their shared factors are not comparable to
# later ones, so load_from_disk flags them.
SHARED_FACTOR_REWORK = datetime(2026, 8, 25, 15, 38, 55, tzinfo=UTC)


# Old role index when all we had was VSO
# def _role_index(role: str) -> int:
#     if role == "verb":
#         return 0
#     elif role == "subject":
#         return 1
#     elif role == "object":
#         return 2
#     else:
#         raise ValueError("role must be one of {'verb','subject','object'}")
#
# def voc_index(role: str) -> str:
#     if role == "verb":
#         return "v2i"
#     elif role == "subject":
#         return "s2i"
#     elif role == "object":
#         return "o2i"
#     else:
#         raise ValueError("role must be one of {'verb','subject','object'}")

def _role_index(role: str, role_names: list[str]) -> int:
    try:
        return role_names.index(role)
    except ValueError as e:
        raise ValueError(f"role must be one of {set(role_names)}") from e


def voc_list_key(role: str) -> str:
    return f"vocab_{role}"


def find_vocab_path(base, order, dims, shared_factors) -> str:
    """The vocabulary file of a tensor: the {order}D_ naming first, then the
    legacy (3D-only) one."""
    vocab_dir = os.path.join(base, "vocabularies")
    candidates = [
        os.path.join(vocab_dir, _vocab_filename(order, dims, shared_factors=shared_factors)),
        os.path.join(vocab_dir, _vocab_filename_legacy(dims, shared_factors=shared_factors,
                                                       order=order)),
    ]
    for path in candidates:
        if os.path.exists(path):
            return path
    raise FileNotFoundError("Missing vocab file. Checked:\n  " + "\n  ".join(candidates))

class TuckerDecomposition:
    """Encapsulating the tucker decomposition (core and factors) and the vocabulary,
    providing methods for scoring, slicing, visualisation, etc."""

    # Family this class loads by default; the TT subclass overrides it so
    # load_from_disk works from either class without an explicit argument.
    _DECOMPOSITION: ClassVar[str] = "tucker"

    def __init__(self, core, factors: List[torch.Tensor],
                 vocab: dict, shared_factors: set | None = None,
                 roles: Optional[List[str]] = None,
                 ):
        self.core = core
        self.factors = factors
        self.vocab = vocab
        self.shared_factors = shared_factors or set()
        # Respect explicitly provided roles; otherwise parse them from the vocab keys.
        self.roles = roles if roles is not None else extract_roles_from_vocab(self.vocab)
        self.decomp_path = None

    def get_role_index(self, role: str) -> int:
        """Helper method to wrap the module-level _role_index using instance roles."""
        return _role_index(role, self.roles)
    def get_rank(self, role=None):
        if role is None:
            role=self.roles[0]
        return self.core.shape[_role_index(role, self.roles)]


    def _core_np(self):
        return to_np(self.core)

    # --- Construction and loading ---
    @classmethod
    def load_from_disk(cls,
                       dataset: str="4-gram-raw-bos-eos-fineweb-en_1B",
                       method: str="scSoftPlus",
                       divergence: str="kl",
                       dims: "int | tuple[int, ...]"=10000,
                       rank: int=100,
                       order: Optional[int]=None,
                       iterations: int|None=None,
                       shared_factors: bool|set|str="all",
                       map_location: str="cpu",
                       name: Optional[str]="h100",
                       tier1: bool=False,
                       subsample_frac: float=0.025,
                       max_nnz: Optional[int]=None,
                       solver: str="mu",
                       decomposition: Optional[str]=None,
                       tt_rank: Optional[int]=None,
                          ) -> "TuckerDecomposition":

        """Loads a precomputed tucker decomposition from disk.
            Args:
                dataset (str): name of the dataset
                method (str): method used to compute the decomposition
                    - one of "counting", "sc", "sii"
                dims (int): dimensionality of the original tensor modes (vocab size)
                rank (int): rank of the decomposition
                iterations (int): number of iterations used to compute the decomposition
                map_location (str): device to map the loaded tensors to
                name (str, optional): optional name prefix for the tensor file
                decomposition (str, optional): family to load, "tucker" or "tt".
                    Defaults to the class's own family, so
                    TuckerTTDecomposition.load_from_disk() needs no argument.
                    Selects the filename tag (see naming._order_tag) and, for
                    "tt", returns a TuckerTTDecomposition.
                tt_rank (int, optional): TT bond dimension; part of the TT stem,
                    so it must match the run. Defaults to 100.
            Returns:
                ((core, factors), vocab)
                    core: torch.Tensor
                    factors: list[torch.Tensor]
                    vocab: dict with keys 'vocab_v','vocab_s','vocab_o','v2i','s2i','o2i'
        """
        if method not in ALL_METHODS:
            raise ValueError(f"method must be one of {set(ALL_METHODS)}")

        if order is None:
            order = infer_ngram_order(dataset, name) or 3

        decomposition = (decomposition or cls._DECOMPOSITION).lower()
        if decomposition == "cp":
            raise ValueError(
                "decomposition='cp' loads through "
                "tensormet.experimental.CP.cp_decomposition.CPDecomposition.load_from_disk"
            )
        if decomposition not in ("tucker", "tt", "utt"):
            raise ValueError(f"decomposition must be 'tucker', 'tt' or 'utt'; got {decomposition!r}")
        if decomposition in ("tt", "utt") and tt_rank is None:
            tt_rank = 100  # matches ExperimentConfig.tt_rank
        base = os.path.join(DATA_DIR, "tensors", dataset)
        base = readonly_dispatch(base, tier1)

        suffix = ""
        parsed_shared = resolve_shared_factors(shared_factors, order)

        if parsed_shared:
            linked_nontrivial = nontrivial_linked_groups(parsed_shared, num_factors=order)
            suffix = shared_factor_suffix(linked_nontrivial)

        # Handle the new {order}D_ naming format vs legacy naming.
        # New format (post N-D migration): {order}D_{dims}d{suffix}.pkl
        # Legacy format (3D only):         {dims}{suffix}.pkl
        vocab_path = find_vocab_path(base, order, dims, parsed_shared)

        decomp_path = os.path.join(base, "decomposition")
        # Candidate prefixes in descending priority: canonical naming, then the
        # no-shared-factor and legacy fallbacks where the family has them (the TT
        # and SGD families have no legacy artifacts, see naming.candidate_stems).
        stems = candidate_stems(
            divergence, method, order, dims, rank,
            name=name, shared_factors=parsed_shared, subsample_frac=subsample_frac,
            max_nnz=max_nnz, solver=solver, decomposition=decomposition, tt_rank=tt_rank,
        )
        # Notice printed when a fallback stem is the one that matches.
        notices = [None] * len(stems)
        if suffix and len(stems) > 1:
            notices[1] = "No shared-factor decomposition found; falling back to non-shared naming."
        if decomposition == "tucker" and solver != "sgd":
            notices[-1] = f"No new-style ({order}D) decomposition found; falling back to legacy naming."

        def _iterations_of(stem: str) -> int:
            """The iteration count of ``stem``'s file: the one asked for, else the
            highest on disk; -1 when there is none."""
            if iterations:
                return iterations if os.path.exists(
                    os.path.join(decomp_path, f"{stem}{iterations}i.pt")) else -1
            highest = -1
            if os.path.exists(decomp_path):
                for filename in os.listdir(decomp_path):
                    if filename.startswith(stem) and filename.endswith("i.pt"):
                        iter_str = filename[len(stem):-len("i.pt")]
                        if iter_str.isdigit():
                            highest = max(highest, int(iter_str))
            return highest

        # Probe the stems in priority order.
        for stem, notice in zip(stems, notices):
            found = _iterations_of(stem)
            if found != -1:
                if notice:
                    print(notice)
                file_prefix, iterations = stem, found
                break
        else:
            if not iterations:
                raise FileNotFoundError(
                    f"Could not find any decomposition files in {decomp_path} "
                    f"matching any of {stems}"
                )
            file_prefix = stems[0]  # reported as missing below

        tensor_name = f"{file_prefix}{iterations}i.pt"
        decomp_path = os.path.join(decomp_path, tensor_name)

        if not os.path.exists(decomp_path):
            raise FileNotFoundError(f"Missing decomposition file: {decomp_path}")

        # --- 1. Load Vocab ---
        with open(vocab_path, "rb") as f:
            vocab = pickle.load(f)

        # --- 2. Extract Roles ---
        roles = [k[len("vocab_"):] for k in vocab.keys() if k.startswith("vocab_")]

        # --- 3. Backward Compatibility for Legacy Tensors ---
        if roles == ["v", "s", "o"]:
            roles = ["verb", "subject", "object"]
            legacy_map = {"v": "verb", "s": "subject", "o": "object"}
            new_vocab = {}
            for old_r, new_r in legacy_map.items():
                new_vocab[f"vocab_{new_r}"] = vocab.pop(f"vocab_{old_r}")
                new_vocab[f"{new_r}2i"] = vocab.pop(f"{old_r}2i")
            new_vocab.update(vocab)  # keep any remaining keys
            vocab = new_vocab

        # --- 4. Load Factors & Return ---
        (core, factors) = torch_or_pickle_load(decomp_path, map_location=map_location)
        # if there is a "runs.jsonl" file in the decomposition folder, print the record for this tensor
        runs_path = os.path.join(os.path.dirname(decomp_path), "runs.jsonl")
        if os.path.exists(runs_path):
            with open(runs_path, "r") as f:
                for line in f:
                    run_info = json.loads(line)
                    if run_info.get("results", {}).get("model_path") == decomp_path:
                        _label = {"tt": "Tucker-TT", "utt": "shared-core TT"}.get(decomposition, "Tucker")
                        print(f"Loaded {_label} decomposition with the following parameters:")
                        for key, value in run_info.items():
                            print(f"  {key}: {value}")
                        break

        else:
            print("Warning: file creation predates logging of runs; no run info available.")

        # "shared" is in the filename only when the loaded file really ties modes,
        # so a fallback to non-shared naming above never trips this.
        if "shared" in os.path.basename(decomp_path):
            started = date_run(decomp_path)
            if started is None or started < SHARED_FACTOR_REWORK:
                when = ("could not be dated, so it may predate" if started is None
                        else f"started {started:%Y-%m-%d}, before")
                print(
                    f"WARNING: {os.path.basename(decomp_path)} {when} the "
                    f"{SHARED_FACTOR_REWORK:%Y-%m-%d} tied-factor rework: its shared factors were "
                    "updated per mode and copied instead of pooled. The run is usable, but its "
                    "tied factors are not comparable with those of later runs."
                )

        # For "tt" the payload's first element is the list of TT cores, which is
        # TuckerTTDecomposition's first positional argument — so one call builds
        # either family. Only substitute the class when the caller's own is not
        # already a TT one: GenTuckerTT(TuckerTTDecomposition, TuckerGenMixin) and
        # friends must come back as themselves, not stripped down to the base.
        target = cls
        if decomposition == "tt":
            from tensormet.tt_hybrid.tt_decomposition import TuckerTTDecomposition
            if not issubclass(cls, TuckerTTDecomposition):
                target = TuckerTTDecomposition
        elif decomposition == "utt":
            # Shared-core TT: the payload's first element is [C, α, ω].
            from tensormet.uniform_tt.utt_decomposition import UniformTTDecomposition
            if not issubclass(cls, UniformTTDecomposition):
                target = UniformTTDecomposition

        instance = target(core, factors, vocab, shared_factors=parsed_shared, roles=roles)
        instance.decomp_path = Path(decomp_path)
        return instance

    def update_from_path(self, path=None):
        resolved = resolve_checkpoint_path(path, self.decomp_path)
        tensor = torch.load(resolved, map_location="cpu", weights_only=False)
        # SGD checkpoints are resumable dict payloads ({"core", "factors",
        # "raw_state_dict", "optim_state", ...}, see SGDTrainer.checkpoint_payload)
        # rather than a TuckerTensor, so plain attribute access fails on them.
        if isinstance(tensor, dict):
            core, factors = tensor["core"], tensor["factors"]
        else:
            core, factors = tensor.core, tensor.factors
        self.core = np_dispatch(core)
        self.factors = np_dispatch(factors)
        self.decomp_path = resolved

    # New: load_best method, which we update manually.
    # This just uses load_from_disk to return the best model we currently have/think to have for quick testing.
    # Comes with a bunch of reproducibility issues whenever this is changed, so use with caution.

    # The single place to edit when a new decomposition becomes "the best one";
    # anything not listed here keeps its load_from_disk default.
    BEST_CONFIG: ClassVar[dict] = {
        "dataset": "4-gram-raw-bos-eos-fineweb-en_1B",
        "method": "scSoftPlus",
        "divergence": "kl",
        "dims": 10000,
        "rank": 100,
        "decomposition":"tt",
        "shared_factors": "all",
        "name": "h100",
        "subsample_frac": 0.025,
        "solver": "mu",
    }

    @classmethod
    def load_best(cls, **overrides) -> "TuckerDecomposition":
        """Loads the current best-known decomposition; keyword arguments override BEST_CONFIG."""
        return cls.load_from_disk(**{**cls.BEST_CONFIG, **overrides})

    def check_vocab(self, triple: Tuple[str, ...], return_type=bool) -> bool|tuple:
        """Checks if the given (verb, subject, object) triple is in the vocabulary."""

        in_roles = [triple[i] in self.vocab[voc_index(self.roles[i])] for i in range(len(self.roles))]
        if return_type == tuple:
            return tuple(in_roles)
        return all(in_roles)

        # v_in = triple[0] in self.vocab["v2i"]
        # s_in = triple[1] in self.vocab["s2i"]
        # o_in = triple[2] in self.vocab["o2i"]
        # if return_type == tuple:
        #     return (v_in, s_in, o_in)
        # return v_in and s_in and o_in


    # def fetch_latents(self, triple: Tuple[str, str, str]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    #     """Fetches the latent representations for a given (verb, subject, object) triple."""
    #     v_idx = self.vocab["v2i"][triple[0]]
    #     s_idx = self.vocab["s2i"][triple[1]]
    #     o_idx = self.vocab["o2i"][triple[2]]
    #     V, S, O = [ to_np(F) for F in self.factors]     # shapes (DIMS,R)
    #     v = V[v_idx]                                     # (R,)
    #     s = S[s_idx]                                     # (R,)
    #     o = O[o_idx]                                     # (R,)
    #     return v, s, o
    def get_dims(self):
        """Mode dimensions of the reconstructed tensor, i.e. (N_0, ..., N_{k-1})."""
        return tuple(int(f.shape[0]) for f in self.factors)

    def fetch_latents(self, triple: Tuple[str, ...]) -> Tuple[np.ndarray, ...]:
        """Fetches the latent representations for a given tuple of elements."""
        # Map fetch_single_latent across all elements and their corresponding roles
        return tuple(
            self.fetch_single_latent(triple[i], self.roles[i])
            for i in range(len(self.roles))
        )

    def fetch_single_latent(self, element, role=None) -> np.ndarray:
        """Fetches the latent representation for an element."""
        if role == None:
            role = self.roles[0] # default, useful for brevity in shared factor elements
        el_idx = self.vocab[voc_index(role)][element]
        factor_slice = self.factors[self.get_role_index(role)][el_idx]
        return to_np(factor_slice)



    # -- Sparsity methods ---
    def to_cupy(self):
        """
        Prepare for inference by moving to gpu
        """
        if isinstance(self.core, torch.Tensor):
            self.core = cp.array(to_np(self.core))
        for i, f in enumerate(self.factors):
            if isinstance(f, torch.Tensor):
                new_f = cp.array(to_np(f))
                self.factors[i] = new_f

    # -- Scoring and slicing methods ---

    # def score_scalar_old(self, triple: Tuple[str, str, str]) -> float:
    #     """(1) Scalar reconstruction score ⟨G, a∘b∘c⟩."""
    #     G = self._core_np()                                 # (R,R,R)
    #     v, s, o = self.fetch_latents(triple)
    #     return np.einsum('pqr,p,q,r->', G, v, s, o)

    def score_scalar(self, triple: Tuple[str, ...]) -> float:
        """(1) Scalar reconstruction score ⟨G, a∘b∘c...⟩."""
        G = self._core_np()
        latents = self.fetch_latents(triple)
        modes = einsum_letters(len(self.roles))

        eq = f"{''.join(modes)},{','.join(modes)}->"
        return np.einsum(eq, G, *latents)

    # def contribution_tensor_old(self, triple: Tuple[str, str, str]) -> np.ndarray:
    #     """(2) Contribution tensor: G * (a∘b∘c) ∈ R^{R×R×R}."""
    #     G = self._core_np()                                 # (R,R,R)
    #     v, s, o = self.fetch_latents(triple)
    #     # same as doing np.einsum(p, q, r ->pqr) and then multiplying by G
    #     return np.einsum('p,q,r,pqr->pqr', v, s, o, G)

    def contribution_tensor(self, triple: Tuple[str, ...]) -> np.ndarray:
        """(2) Contribution tensor: G * (a∘b∘c...)"""
        G = self._core_np()
        latents = self.fetch_latents(triple)
        modes = einsum_letters(len(self.roles))
        core_str = "".join(modes)

        eq = f"{','.join(modes)},{core_str}->{core_str}"
        return np.einsum(eq, *latents, G)

    # def outer_product_latent_old(self, triple: Tuple[str, str, str]) -> np.ndarray:
    #     """(3) Pseudo-inverse / HOSVD case: a∘b∘c (rank-1 core-space tensor)."""
    #     v, s, o = self.fetch_latents(triple)
    #     return np.einsum('p,q,r->pqr', v, s, o)

    def outer_product_latent(self, triple: Tuple[str, ...]) -> np.ndarray:
        """(3) Pseudo-inverse / HOSVD case: a∘b∘c... (rank-1 core-space tensor)."""
        latents = self.fetch_latents(triple)
        modes = einsum_letters(len(self.roles))

        eq = f"{','.join(modes)}->{''.join(modes)}"
        return np.einsum(eq, *latents)

    # def excluded_role_vector_old(self, triple: Tuple[str, str, str], role: str) -> np.ndarray:
    #     """
    #     Fetches the latent vector for a given excluded role in the triple.
    #     Can be understood as a "prediction":
    #         Given the two other attested elements,what are the activations in the third element's dimensions?
    #     """
    #     v, s, o = self.fetch_latents(triple)
    #     if role == "verb":
    #         return np.einsum('pqr,q,r->p', self._core_np(), s, o)
    #     elif role == "subject":
    #         return np.einsum('pqr,p,r->q', self._core_np(), v, o)
    #     elif role == "object":
    #         return np.einsum('pqr,p,q->r', self._core_np(), v, s)
    #     else:
    #         raise ValueError("role must be one of {'verb','subject','object'}")

    def excluded_role_vector(self, triple: Tuple[str, ...], role: str) -> np.ndarray:
        """
        Fetches the latent vector for a given excluded role in the tuple.
        Can be understood as a "prediction":
            Given the other attested elements, what are the activations in the target element's dimensions?
        """
        target_idx = self.get_role_index(role)
        all_latents = self.fetch_latents(triple)
        latents = [all_latents[i] for i in range(len(self.roles)) if i != target_idx]

        modes = einsum_letters(len(self.roles))
        core_str = "".join(modes)
        vec_strs = [modes[i] for i in range(len(self.roles)) if i != target_idx]
        out_str = modes[target_idx]

        eq = f"{core_str},{','.join(vec_strs)}->{out_str}"
        return np.einsum(eq, self._core_np(), *latents)

    # def included_role_vector_old(self, triple: Tuple[str, str, str], role: str) -> np.ndarray:
    #     """
    #     Fetches the latent vector for a given included role in the triple.
    #     Can be understood as quantifying "contribution":
    #         How important are the dimensions of X in the final contextualised representation of XYZ?
    #     """
    #     v, s, o = self.fetch_latents(triple)
    #     if role == "verb":
    #         return np.einsum('pqr,p,q,r->p', self._core_np(), v, s, o)
    #     elif role == "subject":
    #         return np.einsum('pqr,p,q,r->q', self._core_np(), v, s, o)
    #     elif role == "object":
    #         return np.einsum('pqr,p,q,r->r', self._core_np(), v, s, o)
    #     else:
    #         raise ValueError("role must be one of {'verb','subject','object'}")

    def included_role_vector(self, triple: Tuple[str, ...], role: str) -> np.ndarray:
        """
        Fetches the latent vector for a given included role in the tuple.
        Can be understood as quantifying "contribution":
            How important are the dimensions of X in the final contextualised representation?
        """
        target_idx = self.get_role_index(role)
        latents = self.fetch_latents(triple)

        modes = einsum_letters(len(self.roles))
        core_str = "".join(modes)
        out_str = modes[target_idx]

        eq = f"{core_str},{','.join(modes)}->{out_str}"
        return np.einsum(eq, self._core_np(), *latents)


    # def predicted_role_vector_old(self, triple: Tuple[str, str, str], role: str) -> np.ndarray:
    #     """
    #     Fetches the latent vector for a given excluded role in the triple, WITHOUT instantiating the element (OOV).
    #     Can be understood as a "prediction":
    #         Given the two other attested elements,what should be the activations in the third element's dimensions?
    #     """
    #     latents = {"verb":None, "subject":None, "object":None}
    #
    #     v = latents["verb"]
    #     s = latents["subject"]
    #     o = latents["object"]
    #
    #     if role == "verb":
    #         return np.einsum('pqr,q,r->p', self._core_np(), s, o)
    #     elif role == "subject":
    #         return np.einsum('pqr,p,r->q', self._core_np(), v, o)
    #
    #     elif role == "object":
    #         return np.einsum('pqr,p,q->r', self._core_np(), v, s)
    #     else:
    #         raise ValueError("role must be one of {'verb','subject','object'}")

    # Slicing has moved to the experimental tucker_viz.py

    # top activations utility
    def retrieve_highest_activations(self,
                                     triple: Tuple[str, ...],
                                     role: str,
                                     method: str = "slice",
                                     top_k: int = 10):
        target_word = triple[self.get_role_index(role)]
        slc = self.get_slice(triple=triple, role=role, method=method)

        if method == "slice":
            word_id = self.vocab[voc_index(role)][target_word]
            slc = slc[word_id]

        # we retrieve the "coordinates" of the top-k highest activations
        flat_indices = np.argpartition(slc.flatten(), -top_k)[-top_k:]
        unraveled_indices = [np.unravel_index(idx, slc.shape) for idx in flat_indices]
        top_activations = [(idx, slc[idx]) for idx in unraveled_indices]
        # sort by activation value
        top_activations = sorted(top_activations, key=lambda x: x[1], reverse=True)
        return top_activations

    def get_top_words_for_dimension(self,
                                    role: str,
                                    dim_index: int,
                                    top_k: int = 10):
        """
        For a given latent dimension of a role, return the top-k words with
        highest loading on that dimension.
        """
        factor_idx = self.get_role_index(role)
        role_factors = self.factors[factor_idx]  # (N, R)
        dim_values = to_np(role_factors)[:, dim_index]

        scores, indices = torch.topk(torch.tensor(dim_values), top_k)
        vocab_list = self.vocab[voc_list_key(role)]

        top_words = [
            (vocab_list[idx.item()], score.item())
            for idx, score in zip(indices, scores)
        ]
        return top_words

    def get_top_dimensions_for_word(self,
                                    word: str,
                                    role = None,
                                    top_k: int = 10,
                                    return_words=False):
        if role is None:
            role = self.roles[0]
        latent = self.fetch_single_latent(word, role)
        latent = torch.tensor(latent)
        if top_k == "full":
            top_k = len(latent)
        scores, dims = torch.topk(latent, top_k)
        if return_words:
            # variant in which we return the representative word as well
            top_scores = [
                (int(dim), float(score), self.get_top_words_for_dimension(role, dim, 1)[0][0]) for dim, score in zip(dims, scores)
            ]
        else:
            top_scores = [
                (int(dim), float(score)) for dim, score in zip(dims, scores)
            ]
        return top_scores



    def get_expected_element(self, target_tuple: Tuple[str, ...], role: str, verbose: bool = True,
                             method: str="excluded",
                             metric: str = "dot",
                             k = 5):
        """
        metric: 'dot' for raw unnormalized dot product (favors frequent/confident words),
                'cosine' for scale-invariant cosine similarity (often surfaces rare words).
        """
        index = self.get_role_index(role)
        r2i = voc_index(role)
        latents = self.fetch_latents(target_tuple)
        if method == "excluded":
            G_item = self.excluded_role_vector(target_tuple, role=role)
        elif method == "included":
            G_item = self.included_role_vector(target_tuple, role=role)
        else:
            raise NotImplementedError

        # Safely get the numpy array
        factor = self.factors[index].cpu().numpy() if hasattr(self.factors[index], "cpu") else self.factors[index]

        if metric == "cosine":
            # Safely calculate norms to prevent division by 0
            eps = 1e-12
            factor_norm = np.linalg.norm(factor, axis=1)
            G_item_norm = np.linalg.norm(G_item)

            factor_norm = np.maximum(factor_norm, eps)
            G_item_norm = max(G_item_norm, eps)

            scores = (factor @ G_item) / (factor_norm * G_item_norm)
        elif metric == "dot":
            # Raw dot product accounts for vector magnitude (word prominence)
            scores = factor @ G_item
        else:
            raise ValueError("metric must be either 'dot' or 'cosine'")

        # we get the top k most similar elements

        top_k_indices = np.argsort(scores)[-k:][::-1]

        results = []
        for idx in top_k_indices:
            role_str = self.vocab[voc_list_key(role)][int(idx)]

            role_act = self.factors[index][idx, :].cpu().numpy() if hasattr(self.factors[index], "cpu") else \
                self.factors[index][idx, :]

            # Keep cosine similarity for the specific target context as an interesting debug metric
            cos_sim = np_sim(role_act, latents[index])

            results.append({"token": role_str,
                            "score": float(scores[idx]),
                            "activation_cosine": float(cos_sim)})

        if verbose:
            print(f"Top {k} expected {role}s based on the integrated core tensor:")
            for r in results:
                print(f"{role.capitalize()}: {r['token']}, "
                      f"Score ({metric}): {r['score']:.4f}, "
                      f"Cosine sim with target {role} activations: {r['activation_cosine']:.4f}"
                      )
            return None

        return results


    def get_most_similar_elements(self,
                                  element,
                                  role=None,
                                  top_k=5
                                  ):
        """
        Find the most similar element.
        If a tuple is passed as "element", the contextualised version is used.
        If a single word is passed, the default factor matrix entry is used.
        Parameters
        ----------
        element
        role

        Returns
        list of most similar words
        -------

        """
        if role is None:
            role = self.roles[0]
        if isinstance(element, tuple):
            latent = self.included_role_vector(element, role=role)
            # print("latent from context")
        elif isinstance(element, str):
            latent = self.fetch_single_latent(element, role=role)
            # print("latent from factor")
        elif isinstance(element, np.ndarray):
            latent = element
        else:
            raise ValueError("Must be tuple, str or ndarray")

        i = self.get_role_index(role)
        F = self.factors[i].cpu().numpy() if hasattr(self.factors[0], "cpu") else self.factors[i]

        # --- defensive norm computation ---
        F_norm = np.linalg.norm(F, axis=1)
        G_norm = np.linalg.norm(latent)

        eps = 1e-12  # safeguard lower bound
        F_norm = np.maximum(F_norm, eps)
        G_norm = max(G_norm, eps)

        # --- safe cosine similarities ---
        similarities = (F @ latent) / (F_norm * G_norm)
        if top_k == "full":
            top_k = len(F)
        top_idx = np.argsort(-similarities)[:top_k]
        r2i = voc_index(role)

        top_sims = []
        for idx in top_idx:
            role_str = self.vocab[voc_list_key(role)][int(idx)]
            top_sims.append(role_str)

        return top_sims

    def _fixed_role_matrix(self, fixed_idx, other_idxs, fixed_latent):
        """The (R_a, R_b) matrix get_top_combinations ranks against: the core
        with the fixed role contracted away. Overridden by the TT subclass,
        which contracts the chain instead of forming the dense core."""
        modes = einsum_letters(len(self.roles))
        other_chars = "".join(modes[i] for i in other_idxs)
        eq = f"{''.join(modes)},{modes[fixed_idx]}->{other_chars}"
        return np.einsum(eq, self._core_np(), fixed_latent)

    def get_top_combinations(
            self,
            fixed_element: str,
            fixed_role: str,
            top_k: int = 10,
            restrict_roles: Optional[dict[str, list[str]]] = None,
            exclude_oov: bool = True,
            oov_token: str = "~",
    ) -> list[tuple[tuple, float]]:
        fixed_idx = self.get_role_index(fixed_role)
        other_idxs = [i for i in range(len(self.roles)) if i != fixed_idx]

        if len(other_idxs) > 2:
            raise NotImplementedError(
                "get_top_combinations currently supports at most 2 free roles "
                f"(found {len(other_idxs)} for order-{len(self.roles)} tensor). "
                "Consider fixing additional roles or filing a feature request."
            )

        v_latent = self.fetch_single_latent(fixed_element, fixed_role)
        G_fixed = self._fixed_role_matrix(fixed_idx, other_idxs, v_latent)

        role_names_free: list[str] = [self.roles[i] for i in other_idxs]
        factors_free: list[np.ndarray] = []
        vocab_lists_free: list[list[str]] = []

        for role in role_names_free:
            factor = to_np(self.factors[self.get_role_index(role)])
            vocab_list = list(self.vocab[voc_list_key(role)])

            if restrict_roles and role in restrict_roles:
                r2i = self.vocab[voc_index(role)]
                keep_words = [w for w in restrict_roles[role] if w in r2i]
                keep_idxs = [r2i[w] for w in keep_words]
                factor = factor[keep_idxs]
                vocab_list = keep_words

            if exclude_oov and oov_token in vocab_list:
                oov_idx = vocab_list.index(oov_token)
                keep_mask = [i for i in range(len(vocab_list)) if i != oov_idx]
                factor = factor[keep_mask]
                vocab_list = [w for w in vocab_list if w != oov_token]

            factors_free.append(factor)
            vocab_lists_free.append(vocab_list)

        F_a, F_b = factors_free
        scores = F_a @ G_fixed @ F_b.T

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
            score = float(scores[i, j])

            combo: list[str] = [None] * len(self.roles)  # type: ignore[list-item]
            combo[fixed_idx] = fixed_element
            combo[other_idxs[0]] = vocab_a[i]
            combo[other_idxs[1]] = vocab_b[j]

            results.append((tuple(combo), score))

        return results

    def batch_excluded_role_vector(self,
                                   valid_indices: torch.Tensor,
                                   role_name: str) -> torch.Tensor:
        """Uses GPU-accelerated einsum for batch contraction."""
        target_idx = self.get_role_index(role_name)
        n_roles = len(self.roles)
        device = self.factors[0].device

        # 1. Gather latents directly on GPU
        latents = []
        for i in range(n_roles):
            if i == target_idx: continue
            # Slicing a torch tensor on GPU is nearly instantaneous
            latents.append(self.factors[i][valid_indices[:, i]])

        # 2. Setup Einstein Summation
        modes = einsum_letters(n_roles)
        core_str = "".join(modes)
        input_strs = [f"n{modes[i]}" for i in range(n_roles) if i != target_idx]
        eq = f"{core_str},{','.join(input_strs)}->n{modes[target_idx]}"

        # 3. Compute on GPU
        # Ensure core is on the same device as factors
        core = self.core.to(device) if hasattr(self.core, 'to') else torch.tensor(self.core, device=device)
        return torch.einsum(eq, core, *latents)






def _torch_numel_fits(shape) -> bool:
    """Whether ``torch.sparse_coo_tensor`` can hold *shape* (numel is int64)."""
    return math.prod(tuple(int(s) for s in shape)) <= np.iinfo(np.int64).max


def _resolve_use_coords(use_coords: Optional[bool], shape) -> bool:
    """Decide the N-D CuPy representation for *shape*.

    ``TENSORMET_COORD_TENSOR`` overrides the caller (``launch.py`` asks for
    coordinates explicitly, so an env var that only applied to ``None`` would be
    a dead kill-switch): ``1`` forces coordinates, ``0`` pins the block encoding
    wherever it fits and degrades to coordinates where it cannot.

    Otherwise ``None`` picks coordinates only when the block encoding does not
    fit, and an explicit ``False`` on a shape it cannot represent raises rather
    than producing wrapped indices.
    """
    fits = block_encoding_fits(shape)

    env = os.environ.get("TENSORMET_COORD_TENSOR", "")
    if env in ("1", "true", "True"):
        return True
    if env in ("0", "false", "False"):
        return not fits

    if use_coords is None:
        return not fits
    if not use_coords and not fits:
        raise ValueError(
            f"shape={shape} has prod(shape)={math.prod(shape)} > int32_max**2, which the "
            f"block-encoded coo_matrix cannot represent (its linear index would overflow). "
            f"This shape requires the coordinate representation; do not force use_coords=False "
            f"here."
        )
    return bool(use_coords)


def cupy_to_torch_sparse(
    cu_mat: cpx_sparse.spmatrix,
    orig_shape: Optional[Tuple[int, ...]] = None,
    device: Union[str, torch.device] = "cpu",
    dtype: Optional[torch.dtype] = None,
) -> torch.Tensor:
    """
    Convert a CuPy sparse matrix (any format) back to a torch sparse COO tensor.

    If orig_shape is None:
        - The torch tensor is 2D and has the same shape as cu_mat.
    If orig_shape is provided and len(orig_shape) == 2:
        - The torch tensor is 2D with that shape.
    If orig_shape is provided and len(orig_shape) > 2:
        - We treat cu_mat.row as the flattened N-D index and unflatten it
          back to N-D using np.unravel_index, assuming the representation
          created by `torch_sparse_to_cupy`.

    Args:
        cu_mat: CuPy sparse matrix (COO/CSR/CSC, will be converted to COO).
        orig_shape: original tensor shape (for N-D tensors).
        device: target torch device.
        dtype: target dtype for values (defaults to inferred from data).

    Returns:
        torch.sparse_coo_tensor on the requested device.
    """
    # Coordinate-backed tensors already hold what the block decoding below would
    # have to reconstruct, and are used precisely when that decoding is impossible
    # (prod(shape) overflows the linear index). Short-circuit straight to torch.
    if isinstance(cu_mat, CoordCOO):
        shape = tuple(orig_shape) if orig_shape is not None else cu_mat.shape
        indices_t = torch.from_numpy(cp.asnumpy(cu_mat.coords)).long()
        values_t = torch.from_numpy(cp.asnumpy(cu_mat.data))
        if dtype is not None:
            values_t = values_t.to(dtype)
        if not _torch_numel_fits(shape):
            # torch.sparse_coo_tensor computes prod(shape) in int64 and overflows;
            # SparseCOOTensor is the load-time container for exactly this case.
            return SparseCOOTensor(indices_t, values_t, shape).to(device)
        return torch.sparse_coo_tensor(indices_t, values_t, size=shape).coalesce().to(device)

    # Ensure COO format

    if not cpx_sparse.isspmatrix_coo(cu_mat):
        cu_mat = cu_mat.tocoo()

    row_cp = cu_mat.row
    col_cp = cu_mat.col
    data_cp = cu_mat.data

    row_np = cp.asnumpy(row_cp)
    col_np = cp.asnumpy(col_cp)
    data_np = cp.asnumpy(data_cp)

    if orig_shape is None:
        shape = cu_mat.shape
        indices_np = np.vstack([row_np, col_np])
    else:
        shape = tuple(orig_shape)
        if len(shape) == 2:
            indices_np = np.vstack([row_np, col_np])
        else:
            # --- decode block encoding ---
            size = math.prod(shape)  # arbitrary-precision; never overflows
            int32_max = np.iinfo(np.int32).max
            block_size = min(size, int32_max)

            flat = row_np.astype(np.int64) + col_np.astype(np.int64) * np.int64(block_size)
            coords = np.unravel_index(flat, shape)
            indices_np = np.vstack(coords)

    indices_t = torch.from_numpy(indices_np).long()
    values_t = torch.from_numpy(data_np)
    if dtype is not None:
        values_t = values_t.to(dtype)

    x = torch.sparse_coo_tensor(indices_t, values_t, size=shape)
    x = x.coalesce().to(device)
    return x


def torch_sparse_to_cupy(
    x: torch.Tensor,
    use_coords: Optional[bool] = None,
) -> Tuple[Union[cpx_sparse.coo_matrix, CoordCOO], Tuple[int, ...]]:
    """See ``_torch_sparse_to_cupy``."""
    # Import CuPy here, before the converter runs. On the SGD path CuPy is first
    # imported on first use, and the import leaves a reference to its caller's frame
    # behind: if that caller is the converter, its locals (the whole GPU copy and
    # host copies of the tensor) outlive the call and the SGD trainer runs out of memory.
    cp.cuda
    return _torch_sparse_to_cupy(x, use_coords)


def _torch_sparse_to_cupy(
    x: torch.Tensor,
    use_coords: Optional[bool] = None,
) -> Tuple[Union[cpx_sparse.coo_matrix, CoordCOO], Tuple[int, ...]]:
    """
    Convert a torch sparse COO tensor to a CuPy sparse representation.

    For 2D tensors, the mapping is straightforward.
    For N-D tensors (N>2) there are two representations:

    ``CoordCOO`` (coordinate encoding)
        Keeps the ``(ndim, nnz)`` int32 coordinates as they are. Consumed
        directly by the NNZ-streaming ``*_largedim`` kernels. No linear index
        exists, so ``prod(shape)`` is unbounded.
    ``cupyx.scipy.sparse.coo_matrix`` (block encoding)
        Flattens the N-D indices with ``np.ravel_multi_index`` and splits the
        result into ``(row, col)`` blocks of ``int32_max``. Required by the
        dense-unfolding kernels and the SVD initialisers, which do real matrix
        algebra. Only representable while ``prod(shape) <= int32_max**2``.

    Parameters
    ----------
    use_coords :
        ``True``/``False`` to force a representation; ``None`` (default) picks
        ``CoordCOO`` when the block encoding cannot represent the shape, and
        otherwise honours ``TENSORMET_COORD_TENSOR`` (see
        ``_resolve_use_coords``).

    The original shape is returned for reconstruction.
    Returns:
        (cupy_coo_matrix | CoordCOO, original_shape)
    """
    if not (isinstance(x, (torch.Tensor, SparseCOOTensor)) and x.is_sparse):
        raise TypeError("torch_sparse_to_cupy expects a torch sparse tensor (COO).")
    x = x.coalesce()
    indices = x.indices()  # (ndim, nnz)
    values = x.values()  # (nnz,)
    shape = tuple(x.shape)

    indices_np = indices.cpu().numpy()
    values_np = values.cpu().numpy()

    ndim, nnz = indices_np.shape

    if ndim == 2:
        # unchanged
        row = indices_np[0]
        col = indices_np[1]
        row_cp = cp.asarray(row)
        col_cp = cp.asarray(col)
        data_cp = cp.asarray(values_np)
        cu_mat = cpx_sparse.coo_matrix((data_cp, (row_cp, col_cp)), shape=shape)
    elif _resolve_use_coords(use_coords, shape):
        # --- COORDINATE ENCODING ---
        # No linear index is formed, so prod(shape) is irrelevant: only each
        # individual dimension has to fit int32. This is the only representation
        # that works once prod(shape) passes int32_max**2 (~4.6e18) — a 5-gram at
        # dim=10000 needs 1e20 — and it saves the largedim kernels the decode
        # they otherwise redo every iteration.
        too_wide = [d for d in shape if d > np.iinfo(np.int32).max]
        if too_wide:
            raise ValueError(
                f"CoordCOO stores coordinates as int32; mode dimension(s) {too_wide} "
                f"exceed int32_max. shape={shape}"
            )
        coords_cp = cp.asarray(indices_np.astype(np.int32))
        cu_mat = CoordCOO(coords_cp, cp.asarray(values_np), shape)
    else:
        # --- NEW BLOCK ENCODING ---
        coords = [indices_np[d] for d in range(ndim)]
        size = math.prod(shape)  # arbitrary-precision; never overflows
        flat = np.ravel_multi_index(coords, shape)  # 0..size-1

        int32_max = np.iinfo(np.int32).max
        block_size = min(size, int32_max)
        # number of blocks on the column axis
        n_blocks = (size + block_size - 1) // block_size

        row = flat % block_size
        col = flat // block_size

        row_cp = cp.asarray(row, dtype=cp.int32)
        col_cp = cp.asarray(col, dtype=cp.int32)
        data_cp = cp.asarray(values_np)

        cu_mat = cpx_sparse.coo_matrix(
            (data_cp, (row_cp, col_cp)),
            shape=(block_size, n_blocks),
        )

    return cu_mat, shape




class SparseTupleTensor:
    """Encapsulating the Sparse TupleTensor (built from vectors extracted from corpus) and the vocabulary,
    providing methods for decomposition, refactoring, etc.."""
    def __init__(self, tensor, device="cpu", sparsity_type=None, shared_factors=None, vocab=None):
        self.tensor = tensor
        self.sparsity_type = sparsity_type
        self.shape = tensor.shape
        self.device = device
        self.shared_factors = shared_factors
        self.vocab = vocab

    def link_vocab(self, vocab):
        """Link a vocabulary to this tensor. Accepts a vocab dict or a path to a .pkl file."""
        if isinstance(vocab, (str, Path)):
            with open(vocab, "rb") as f:
                vocab = pickle.load(f)
        if not isinstance(vocab, dict):
            raise TypeError(f"vocab must be a dict or a path to a .pkl file, got {type(vocab)}")
        self.vocab = vocab

    def _require_vocab(self):
        if self.vocab is None:
            raise RuntimeError(
                "No vocabulary linked. Pass add_vocab=True to load_from_disk, "
                "or call link_vocab() with a vocab dict or a path to a .pkl file."
            )

    @property
    def roles(self):
        self._require_vocab()
        return [k[len("vocab_"):] for k in self.vocab if k.startswith("vocab_")]

    # --- Construction and loading ---
    @classmethod
    def load_from_disk(
            cls,
            dataset: str = "fineweb-en",
            method: str = "siiSoftPlus",
            order: Optional[int] = None,
            dims: "int | tuple[int, ...]" = 1000,
            map_location: str = "cpu",
            tier1: bool = False,
            shared_factors: bool|set|str=False,
            add_vocab: bool = False,
    ) -> "SparseTupleTensor":
        """
        Load a populated sparse tensor from disk.

        Expects population artifacts saved as:
            tensors/{dataset}/populated/{method}_{dims}{suffix}.pt
        where suffix matches the shared-factor naming convention.

        If add_vocab=True, automatically loads the matching vocabulary file
        (derived from order, dims, shared_factors — same as TuckerDecomposition).
        """
        if method not in ALL_METHODS:
            raise ValueError(f"method must be one of {set(ALL_METHODS)}")

        if order is None:
            order = infer_ngram_order(dataset) or 3

        base = os.path.join(DATA_DIR, "tensors", dataset)
        base = readonly_dispatch(base, tier1)

        shared_factors = resolve_shared_factors(shared_factors, order)

        linked_nontrivial = nontrivial_linked_groups(shared_factors, num_factors=order)
        suffix = shared_factor_suffix(linked_nontrivial)
        populated_path = os.path.join(base, "populated", populated_filename(method, order, dims, shared_factors=shared_factors))

        if not os.path.exists(populated_path):
            if order == 3: # legacy naming support
                populated_path = os.path.join(base, "populated", populated_filename_legacy(method, dims, shared_factors=shared_factors))
            else:
                raise FileNotFoundError(f"Missing populated tensor file: {populated_path}")

        tensor = torch_or_pickle_load(populated_path, map_location=map_location)

        vocab = None
        if add_vocab:
            with open(find_vocab_path(base, order, dims, shared_factors), "rb") as f:
                vocab = pickle.load(f)

        return cls(
            tensor,
            device=map_location,
            sparsity_type="torch",
            shared_factors=shared_factors,
            vocab=vocab,
        )



    # -- Sparsity methods ---
    def sparse_representation(self, sparse_type, use_coords: Optional[bool] = None):
        # we return the sparse representation of the tensor
        if sparse_type == self.sparsity_type:
            return self.tensor

        if sparse_type == "torch":
            if not self.sparsity_type or self.sparsity_type == "dense":
               return self.tensor.to_sparse()
            elif self.sparsity_type == "cupy":
                return cupy_to_torch_sparse(self.tensor, orig_shape=self.shape)
            else:
                raise NotImplementedError("sparsity_type must be one of {'dense', None, 'cupy', 'torch'}")

        elif sparse_type == "cupy":
            if not (isinstance(self.tensor, (torch.Tensor, SparseCOOTensor)) and self.tensor.is_sparse):
                raise TypeError("cupy expects self.tensor to be a torch sparse tensor.")
            tensor_cupy, shape = torch_sparse_to_cupy(self.tensor, use_coords=use_coords)
            return tensor_cupy
        else:
            raise NotImplementedError(f"Sparse representation for type {sparse_type} not implemented.")



    def tensor_to_sparse(self, sparse_type="cupy", use_coords: Optional[bool] = None):
        self.tensor = self.sparse_representation(sparse_type, use_coords=use_coords)
        self.sparsity_type = sparse_type
        if sparse_type == "cupy":
            self.device = "cuda"


    def tensor_to_dense(self):
        if isinstance(self.tensor, SparseCOOTensor):
            raise TypeError("tensor_to_dense is not supported for SparseCOOTensor (numel overflow).")
        if not isinstance(self.tensor, torch.Tensor) or not self.tensor.is_sparse:
            raise TypeError("tensor_to_dense expects self.tensor to be a torch sparse tensor.")
        self.tensor = self.tensor.to_dense()
        self.sparsity_type = "dense"

    def to_device(self, device):
        self.tensor = tree_to_device(self.tensor, device)
        self.device = device
        if device == "cpu":
            torch.cuda.empty_cache()

    def inspect(self):
        print("type:", type(self.tensor))
        print("sparsity type:", self.sparsity_type)
        print("shape:", self.shape)
        print("device:", self.device)

        if not self.sparsity_type or self.sparsity_type == "dense":
            memory_size = self.tensor.element_size() * self.tensor.nelement()
        elif self.sparsity_type == "torch":
            nnz = self.tensor._nnz()
            dtype_size = self.tensor.values().element_size()
            memory_size = nnz * (self.tensor.indices().element_size() * self.tensor.indices().shape[0] + dtype_size)

        elif self.sparsity_type == "cupy":
            if isinstance(self.tensor, CoordCOO):
                memory_size = self.tensor.data.nbytes + self.tensor.coords.nbytes
            else:
                memory_size = self.tensor.data.nbytes + self.tensor.row.nbytes + self.tensor.col.nbytes
        else:
            memory_size = self.tensor.nbytes
        print(f"approx. memory size: {memory_size / (1024**2):.2f} MB")

    # --- Vocabulary inspection methods ---

    def vocab_size(self, role=None):
        """Return vocab size per role as a {role: int} dict, or for a single role as int."""
        self._require_vocab()
        if role is not None:
            return len(self.vocab[f"vocab_{role}"])
        return {r: len(self.vocab[f"vocab_{r}"]) for r in self.roles}

    def vocab_for(self, role):
        """Return the word list (index → word) for a role."""
        self._require_vocab()
        return self.vocab[f"vocab_{role}"]

    def word_index(self, word, role):
        """Return the tensor index for a word in a given role."""
        self._require_vocab()
        return self.vocab[f"{role}2i"][word]

    def index_word(self, idx, role):
        """Return the word for a tensor index in a given role."""
        self._require_vocab()
        return self.vocab[f"vocab_{role}"][idx]

    def decode_entry(self, indices):
        """Decode a raw COO index tuple to a word tuple."""
        self._require_vocab()
        roles = self.roles
        return tuple(self.index_word(int(idx), roles[i]) for i, idx in enumerate(indices))

    def _coalesced(self, who):
        """``(indices, values)`` of the coalesced torch sparse tensor; needs a vocab."""
        self._require_vocab()
        if not (isinstance(self.tensor, (torch.Tensor, SparseCOOTensor)) and self.tensor.is_sparse):
            raise TypeError(f"{who} requires a torch sparse tensor.")
        t = self.tensor.coalesce()
        return t.indices(), t.values()

    def top_entries(self, k=10, largest=True):
        """Return the top-k entries by value as (word_tuple, value) pairs."""
        indices, values = self._coalesced("top_entries")
        actual_k = min(k, values.numel())
        if actual_k == 0:
            return []
        top_vals, top_pos = torch.topk(values, actual_k, largest=largest)
        return [
            (self.decode_entry(tuple(indices[:, pos].tolist())), top_vals[i].item())
            for i, pos in enumerate(top_pos)
        ]

    def entries_for(self, word, role, k=None):
        """Return tensor entries involving word in the given role as (word_tuple, value) pairs.

        If k is given, returns the top-k by value instead of all entries.
        """
        indices, values = self._coalesced("entries_for")
        word_idx = self.word_index(word, role)
        role_dim = self.roles.index(role)
        mask = indices[role_dim] == word_idx
        if not mask.any():
            return []
        matched_indices = indices[:, mask]
        matched_values = values[mask]
        if k is not None:
            top_vals, top_pos = torch.topk(matched_values, min(k, matched_values.numel()))
            matched_indices = matched_indices[:, top_pos]
            matched_values = top_vals
        return [
            (self.decode_entry(tuple(matched_indices[:, i].tolist())), matched_values[i].item())
            for i in range(matched_indices.shape[1])
        ]

    def marginal_weight(self, word, role):
        """Return the sum of all tensor values for entries involving word in the given role."""
        indices, values = self._coalesced("marginal_weight")
        word_idx = self.word_index(word, role)
        role_dim = self.roles.index(role)
        mask = indices[role_dim] == word_idx
        return values[mask].sum().item() if mask.any() else 0.0

    def top_words_by_marginal(self, role, k=10):
        """Return top-k words by marginal weight (sum of values across all co-occurrences)."""
        indices, values = self._coalesced("top_words_by_marginal")
        role_dim = self.roles.index(role)
        vocab_list = self.vocab_for(role)
        marginals = torch.zeros(len(vocab_list), dtype=values.dtype)
        marginals.scatter_add_(0, indices[role_dim].to(torch.long), values)
        actual_k = min(k, len(vocab_list))
        top_vals, top_idxs = torch.topk(marginals, actual_k)
        return [(vocab_list[idx.item()], top_vals[i].item()) for i, idx in enumerate(top_idxs)]

    def estimate_training_time(self,
                               rank=100,
                               subsample=1
                               ):

        # dim = self.shape[0]
        _max_dim = max(self.shape)
        if self.sparsity_type == "cupy":
            nnz=self.tensor.nnz
        else:
            nnz=self.tensor._nnz()

        # factor_time = (rank**1.76) * (dim**0.16) * (nnz**0.78) * 1e-9
        # print(factor_time, "estimated time per factor update")
        # core_time = (rank**2.6) * nnz * 1e-10
        # print(core_time, "estimated time per core update")
        # print("total:", factor_time + core_time)
        time = 3.4* (rank**2.72) * (nnz**0.25) * 1e-4
        return time*subsample

    def non_negative_tucker_with_similarity(self, cfg: RunConfig, thread_budget: ThreadBudget,
                                            vocab=None, sample_sentences=None):
        """Run the decomposition described by ``cfg`` on this tensor (the loop lives
        in training.py). Returns a dict: the model (``tensor``), the error curve,
        the semantic scores and how far the run got."""
        from tensormet.training import train
        return train(self, cfg, thread_budget, vocab, sample_sentences)
