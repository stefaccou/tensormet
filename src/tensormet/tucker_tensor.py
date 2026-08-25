from __future__ import annotations
import os
import pickle
import signal

import torch
import json
import math
import numpy as np
import tensorly as tl
from tensorly.tucker_tensor import validate_tucker_rank, tucker_normalize, TuckerTensor
from tensorly.tenalg import mode_dot
from typing import ClassVar, List, Optional, Union, Tuple,  Literal
from collections import defaultdict
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager

from tensormet.config import RunConfig
from pathlib import Path
from tensormet.utils import (DATA_DIR,
                            torch_or_pickle_load,
                            readonly_dispatch,
                            tree_to_device,
                            notify_discord,
                            ThreadBudget,
                            shared_factor_suffix,
                            nontrivial_linked_groups,
                            voc_index,
                            extract_roles_from_vocab,
                            einsum_letters,
                            SparseCOOTensor,
                            guarded_cupy_import,
                            make_lazy_cupy_pair,
                            dim_spec_str,
                            np_dispatch,
                            np_sim,
                            resolve_checkpoint_path,
                            sync_devices,
                            _to_np,
                   )
from tensormet.hpc_helpers import mirror_checkpoint
from tensormet.sparse_ops import (
    initialize_nonnegative_tucker,
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
from tensormet.similarity import evaluate_sample, get_eval_num_threads, load_simlex, evaluate_simlex
from tensormet.routing import get_update_routing_step, get_log_step, UpdateRouting, needs_largedim
from tensormet.distance import (
    null_compute_errors,
    NNZGroupingCache,
    precompute_largedim_batches,
    coords_nnz,
)
from tensormet.stochastic_sparse import CooSubsampler
from tensormet.sharded_sparse import (
            ShardedSparseTensor,
            make_sharded_kl_factor_update,
            make_sharded_fr_factor_update,
            make_sharded_kl_core_update,
            make_sharded_fr_core_update,
            make_sharded_kl_compute_errors,
            make_sharded_fr_compute_errors,
        )
import time

cp, cpx_sparse = make_lazy_cupy_pair()


def _as_host(x):
    """Host-side value for save/eval sites shared by the MU (CuPy) and SGD
    (torch/numpy) solver paths. Never touches the lazy `cp` for non-CuPy input,
    so torch-only environments stay CuPy-free."""
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    if isinstance(x, (np.ndarray, np.generic, float, int)) or x is None:
        return x
    if isinstance(x, (list, tuple)):
        return [_as_host(v) for v in x]
    return cp.asnumpy(x)
# Maps tensor role names to SimLex-999 POS tags (first match per POS wins)
_SIMLEX_POS_MAP = {
    "root": "V", "verb": "V",
    "nsubj": "N", "obj": "N", "subject": "N", "object": "N",
}
_SIMLEX_PATH = DATA_DIR / "corpora" / "SimLex-999.txt"


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


def _voc_list_key(role: str) -> str:
    return f"vocab_{role}"

class TuckerDecomposition:
    """Encapsulating the tucker decomposition (core and factors) and the vocabulary,
    providing methods for scoring, slicing, visualisation, etc."""
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
        return _to_np(self.core)

    # --- Construction and loading ---
    @classmethod
    def load_from_disk(cls,
                       dataset: str="fineweb-en",
                       method: str="siiSoftPlus",
                       divergence: str="kl",
                       dims: "int | tuple[int, ...]"=4000,
                       rank: int=100,
                       order: int=3,
                       iterations: int|None=None,
                       shared_factors: bool|set|str=False,
                       map_location: str="cpu",
                       name: Optional[str]=None,
                       tier1: bool=False,
                       subsample_frac: float=1.0,
                       max_nnz: Optional[int]=None,
                       solver: str="mu",
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
            Returns:
                ((core, factors), vocab)
                    core: torch.Tensor
                    factors: list[torch.Tensor]
                    vocab: dict with keys 'vocab_v','vocab_s','vocab_o','v2i','s2i','o2i'
        """
        if method not in ALL_METHODS:
            raise ValueError(f"method must be one of {set(ALL_METHODS)}")
        base = os.path.join(DATA_DIR, "tensors", dataset)
        base = readonly_dispatch(base, tier1)

        parsed_shared = None
        suffix = ""

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

        if parsed_shared:
            linked_nontrivial = nontrivial_linked_groups(parsed_shared, num_factors=order)
            suffix = shared_factor_suffix(linked_nontrivial)

        # Handle the new {order}D_ naming format vs legacy naming.
        # New format (post N-D migration): {order}D_{dims}d{suffix}.pkl
        # Legacy format (3D only):         {dims}{suffix}.pkl
        _vdir = os.path.join(base, "vocabularies")
        vocab_path_new = os.path.join(_vdir, _vocab_filename(order, dims, shared_factors=parsed_shared))
        vocab_path_old = os.path.join(_vdir, _vocab_filename_legacy(dims, shared_factors=parsed_shared, order=order))

        if os.path.exists(vocab_path_new):
            vocab_path = vocab_path_new
        elif os.path.exists(vocab_path_old):
            vocab_path = vocab_path_old
        else:
            raise FileNotFoundError(f"Missing vocab file. Checked {vocab_path_new} and {vocab_path_old}")

        decomp_path = os.path.join(base, "decomposition")
        # Construct candidate prefixes: new naming first, legacy fallback.
        stems = candidate_stems(
            divergence, method, order, dims, rank,
            name=name, shared_factors=parsed_shared, subsample_frac=subsample_frac,
            max_nnz=max_nnz, solver=solver,
        )
        new_file_prefix      = stems[0]
        new_file_prefix_no_sf = stems[1] if len(stems) > 2 else stems[0]
        legacy_file_prefix   = stems[-1]

        def _find_highest_iter(decomp_dir: str, prefix: str) -> int:
            highest = -1
            if os.path.exists(decomp_dir):
                for filename in os.listdir(decomp_dir):
                    if filename.startswith(prefix) and filename.endswith("i.pt"):
                        iter_str = filename[len(prefix):-len("i.pt")]
                        if iter_str.isdigit():
                            highest = max(highest, int(iter_str))
            return highest

        # Look for the highest iteration option if not specified
        if not iterations:
            highest_iter = _find_highest_iter(decomp_path, new_file_prefix)
            if highest_iter != -1:
                file_prefix = new_file_prefix
            elif suffix:
                highest_iter = _find_highest_iter(decomp_path, new_file_prefix_no_sf)
                if highest_iter != -1:
                    print(f"No shared-factor decomposition found; falling back to non-shared naming.")
                    file_prefix = new_file_prefix_no_sf
            if highest_iter == -1:
                highest_iter = _find_highest_iter(decomp_path, legacy_file_prefix)
                if highest_iter != -1:
                    print(f"No new-style ({order}D) decomposition found; falling back to legacy naming.")
                    file_prefix = legacy_file_prefix
                else:
                    raise FileNotFoundError(
                        f"Could not find any decomposition files in {decomp_path} "
                        f"matching '{new_file_prefix}' or '{legacy_file_prefix}'"
                    )
            iterations = highest_iter
        else:
            # When iterations is given explicitly, prefer new naming, fall back to legacy.
            file_prefix = new_file_prefix
            if not os.path.exists(os.path.join(decomp_path, f"{new_file_prefix}{iterations}i.pt")):
                if suffix and os.path.exists(os.path.join(decomp_path, f"{new_file_prefix_no_sf}{iterations}i.pt")):
                    print(f"No shared-factor decomposition found; falling back to non-shared naming.")
                    file_prefix = new_file_prefix_no_sf
                elif os.path.exists(os.path.join(decomp_path, f"{legacy_file_prefix}{iterations}i.pt")):
                    print(f"No new-style ({order}D) decomposition found; falling back to legacy naming.")
                    file_prefix = legacy_file_prefix

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
                        print("Loaded Tucker decomposition with the following parameters:")
                        for key, value in run_info.items():
                            print(f"  {key}: {value}")
                        break

        else:
            print("Warning: file creation predates logging of runs; no run info available.")

        instance = cls(core, factors, vocab, shared_factors=parsed_shared, roles=roles)
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
        "order": 4,
        "shared_factors": "all",
        "name": "h100_1B_sgd",
        "subsample_frac": 1,
        "solver": "sgd",
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
    #     V, S, O = [ _to_np(F) for F in self.factors]     # shapes (DIMS,R)
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
        return _to_np(factor_slice)



    # -- Sparsity methods ---
    def to_cupy(self):
        """
        Prepare for inference by moving to gpu
        """
        if isinstance(self.core, torch.Tensor):
            self.core = cp.array(_to_np(self.core))
        for i, f in enumerate(self.factors):
            if isinstance(f, torch.Tensor):
                new_f = cp.array(_to_np(f))
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
        dim_values = _to_np(role_factors)[:, dim_index]

        scores, indices = torch.topk(torch.tensor(dim_values), top_k)
        vocab_list = self.vocab[_voc_list_key(role)]

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
            role_str = next(key for key, v in self.vocab[r2i].items() if v == idx)

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
            role_str = next(k for k, v in self.vocab[r2i].items() if v == idx)
            top_sims.append(role_str)

        return top_sims

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

        G = self._core_np()
        modes = einsum_letters(len(self.roles))
        fixed_char = modes[fixed_idx]
        other_chars = [modes[i] for i in other_idxs]
        eq_contract = f"{''.join(modes)},{fixed_char}->{''.join(other_chars)}"
        G_fixed = np.einsum(eq_contract, G, v_latent)

        role_names_free: list[str] = [self.roles[i] for i in other_idxs]
        factors_free: list[np.ndarray] = []
        vocab_lists_free: list[list[str]] = []

        for role in role_names_free:
            factor = _to_np(self.factors[self.get_role_index(role)])
            vocab_list = list(self.vocab[_voc_list_key(role)])

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
            order: int = 3,
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

        base = os.path.join(DATA_DIR, "tensors", dataset)
        base = readonly_dispatch(base, tier1)

        if shared_factors == "all":
            shared_factors = tuple(sorted((i, j) for i in range(order) for j in range(i + 1, order)))

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
            _vdir = os.path.join(base, "vocabularies")
            vocab_path_new = os.path.join(_vdir, _vocab_filename(order, dims, shared_factors=shared_factors))
            vocab_path_old = os.path.join(_vdir, _vocab_filename_legacy(dims, shared_factors=shared_factors, order=order))
            if os.path.exists(vocab_path_new):
                vocab_path = vocab_path_new
            elif os.path.exists(vocab_path_old):
                vocab_path = vocab_path_old
            else:
                raise FileNotFoundError(
                    f"add_vocab=True but no vocab file found. "
                    f"Checked:\n  {vocab_path_new}\n  {vocab_path_old}"
                )
            with open(vocab_path, "rb") as f:
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
        # we check if our tensor is a tensorflow tensor or make it one
        if sparse_type == "tensorflow":
            import tensorflow as tf
            if self.sparsity_type != "torch":
                tensor = self.sparse_representation("torch")
            else:
                tensor = self.tensor
            # we build from torch sparse tensor
            indices = tensor.coalesce().indices().t().numpy()   # shape (nnz, ndim)
            values  = tensor.coalesce().values().numpy()        # shape (nnz,)
            shape   = tuple(self.shape)          # e.g. (d0, d1, ..., d_{n-1})
            sparse_tensor = tf.SparseTensor(indices=indices, values=values, dense_shape=shape)
            # we warn users that tensorflow sparse tensors map directly to GPU.
            # additionally, they directly "allocate" the whole GPU memory to tf to reduce fragmentation later on.
            # this makes nvtop commands etc. not useable anymore

            print("WARNING: TensorFlow sparse tensors are allocated on GPU and may reserve large amounts of GPU memory.")

            return sparse_tensor

        elif sparse_type == "torch":
            if not self.sparsity_type or self.sparsity_type == "dense":
               return self.tensor.to_sparse()
            # can work from any tensor-like object
            elif self.sparsity_type == "cupy":
                return cupy_to_torch_sparse(self.tensor, orig_shape=self.shape)
            elif self.sparsity_type == "tensorflow":
                coords = self.tensor.indices.numpy()       # shape (nnz, ndim)
                data   = self.tensor.values.numpy()        # shape (nnz,)
                shape  = tuple(self.shape)  # e.g. (d0, d1, ..., d_{n-1})
                sparse_tensor = torch.sparse_coo_tensor(torch.tensor(coords).t(), torch.tensor(data), size=shape, device="cpu")
                return sparse_tensor
            elif self.sparsity_type == "sparse":
                coords = self.tensor.coords       # shape (nnz, ndim)
                data   = self.tensor.data        # shape (nnz,)
                shape  = tuple(self.shape)  # e.g. (d0, d1, ..., d_{n-1})
                sparse_tensor = torch.sparse_coo_tensor(torch.tensor(coords), torch.tensor(data), size=shape, device="cpu")
                return sparse_tensor
            else:
                raise NotImplementedError("sparsity_type must be one of {'dense', None, 'cupy', 'tensorflow','torch'}")

        elif sparse_type == "sparse":
            # can only work from a sparse torch tensor (or SparseCOOTensor)
            if not (isinstance(self.tensor, (torch.Tensor, SparseCOOTensor)) and self.tensor.is_sparse):
                raise TypeError("sparse expects self.tensor to be a torch sparse tensor.")
            coords = self.tensor.indices().numpy()       # shape (nnz, ndim)
            data   = self.tensor.values().numpy()        # shape (nnz,)
            shape  = tuple(self.tensor.size())  # e.g. (d0, d1, ..., d_{n-1})
            import sparse  # lazy: pulls in numba; only needed for the "sparse" branch
            sparse_tensor = sparse.COO(coords, data, shape=shape)
            return sparse_tensor

        elif sparse_type == "cupy":
            if not (isinstance(self.tensor, (torch.Tensor, SparseCOOTensor)) and self.tensor.is_sparse):
                raise TypeError("cupy expects self.tensor to be a torch sparse tensor.")
            tensor_cupy, shape = torch_sparse_to_cupy(self.tensor, use_coords=use_coords)
            return tensor_cupy
        else:
            raise NotImplementedError(f"Sparse representation for type {sparse_type} not implemented.")



    def tensor_to_sparse(self, sparse_type="tensorflow", use_coords: Optional[bool] = None):
        self.tensor = self.sparse_representation(sparse_type, use_coords=use_coords)
        self.sparsity_type = sparse_type
        if sparse_type in ["tensorflow", "cupy"]:
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
        elif self.sparsity_type == "sparse":
            memory_size = self.tensor.nbytes
        elif self.sparsity_type == "tensorflow":
            nnz = self.tensor.values.shape[0]
            dtype_size = self.tensor.values.dtype.size
            memory_size = nnz * (self.tensor.indices.dtype.size * self.tensor.indices.shape[1] + dtype_size)

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

    def top_entries(self, k=10, largest=True):
        """Return the top-k entries by value as (word_tuple, value) pairs."""
        self._require_vocab()
        if not (isinstance(self.tensor, (torch.Tensor, SparseCOOTensor)) and self.tensor.is_sparse):
            raise TypeError("top_entries requires a torch sparse tensor.")
        t = self.tensor.coalesce()
        values = t.values()
        indices = t.indices()
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
        self._require_vocab()
        if not (isinstance(self.tensor, (torch.Tensor, SparseCOOTensor)) and self.tensor.is_sparse):
            raise TypeError("entries_for requires a torch sparse tensor.")
        word_idx = self.word_index(word, role)
        role_dim = self.roles.index(role)
        t = self.tensor.coalesce()
        indices = t.indices()
        values = t.values()
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
        self._require_vocab()
        if not (isinstance(self.tensor, (torch.Tensor, SparseCOOTensor)) and self.tensor.is_sparse):
            raise TypeError("marginal_weight requires a torch sparse tensor.")
        word_idx = self.word_index(word, role)
        role_dim = self.roles.index(role)
        t = self.tensor.coalesce()
        indices = t.indices()
        values = t.values()
        mask = indices[role_dim] == word_idx
        return values[mask].sum().item() if mask.any() else 0.0

    def top_words_by_marginal(self, role, k=10):
        """Return top-k words by marginal weight (sum of values across all co-occurrences)."""
        self._require_vocab()
        if not (isinstance(self.tensor, (torch.Tensor, SparseCOOTensor)) and self.tensor.is_sparse):
            raise TypeError("top_words_by_marginal requires a torch sparse tensor.")
        role_dim = self.roles.index(role)
        vocab_list = self.vocab_for(role)
        t = self.tensor.coalesce()
        indices = t.indices()
        values = t.values()
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

    def non_negative_tucker_with_similarity(
            self,
            cfg: RunConfig,
            thread_budget: ThreadBudget,
            vocab=None,
            sample_sentences=None,
    ):
        # unpacking the config
        try:
            # experiment config
            rank = list(cfg.exp.rank)
            divergence = cfg.exp.divergence
            dim = cfg.exp.dim
            random_state = cfg.exp.random_state

            # training config
            n_iter_max = cfg.train.n_iter_max
            tol = cfg.train.tol
            verbose = cfg.train.verbose
            return_errors = cfg.train.return_errors
            patience = cfg.train.patience
            warmup_steps = cfg.train.warmup_steps
            largedim = cfg.train.largedim
            checkpoint_saving = cfg.train.checkpoint_saving_steps

            # experiment config (direct result impact)
            init = cfg.exp.init
            epsilon = cfg.exp.epsilon
            normalize_factors = cfg.exp.normalize_factors
            objective = getattr(cfg.exp, "objective", "full")

            # EXPERIMENTAL CP (reviews/CP_IMPLEMENTATION_PLAN.md); getattr for
            # the same deserialized-config back-compat reason as the SGD block.
            decomposition = getattr(cfg.exp, "decomposition", "tucker")
            cp_inner_iters = getattr(cfg.exp, "cp_inner_iters", 1)
            cp_scooch_kappa = getattr(cfg.exp, "cp_scooch_kappa", 0.0)
            # EXPERIMENTAL Tucker-TT hybrid (experimental/TT_hybrid/README.md).
            tt_rank = getattr(cfg.exp, "tt_rank", 100)
            if decomposition not in ("tucker", "cp", "tt"):
                raise ValueError(
                    f"cfg.exp.decomposition must be 'tucker', 'cp' or 'tt'; got {decomposition!r}"
                )

            # SGD solver (sgd/README.md). getattr for the same deserialized-config
            # back-compat reason as above: pre-SGD records carry none of these
            # fields, and the defaults reproduce the MU pipeline exactly.
            solver = getattr(cfg.exp, "solver", "mu")
            sgd_lr = getattr(cfg.exp, "sgd_lr", 1e-2)
            sgd_batch_size = getattr(cfg.exp, "sgd_batch_size", 4096)
            sgd_optimizer = getattr(cfg.exp, "sgd_optimizer", "adam")
            sgd_parametrization = getattr(cfg.exp, "sgd_parametrization", "softplus")
            sgd_steps_per_iteration = getattr(cfg.exp, "sgd_steps_per_iteration", 100)
            sgd_warm_start = getattr(cfg.exp, "sgd_warm_start", None)
            sgd_batch_scope = getattr(cfg.exp, "sgd_batch_scope", "per_device")
            sgd_sync_every = getattr(cfg.exp, "sgd_sync_every", 1)
            sgd_micro_batch = getattr(cfg.exp, "sgd_micro_batch", None)
            sgd_cuda_graph = getattr(cfg.exp, "sgd_cuda_graph", False)
            sgd_comm_backend = getattr(cfg.exp, "sgd_comm_backend", "auto")
            sgd_eval_sample = getattr(cfg.exp, "sgd_eval_sample", None)
            if solver not in ("mu", "sgd"):
                raise ValueError(
                    f"cfg.exp.solver must be 'mu' or 'sgd'; got {solver!r}"
                )

            rec_check_every = cfg.eval.rec_check_every
            sem_check_every = cfg.eval.sem_check_every
            sem_error_type = cfg.eval.sem_error_type
            sem_softmax_temperature = cfg.eval.sem_softmax_temperature
            # LLM-as-judge dimension-consistency scoring (default off). getattr so
            # configs deserialized from before these fields existed keep working.
            dim_consistency = getattr(cfg.eval, "dim_consistency", False)
            dim_consistency_words = getattr(cfg.eval, "dim_consistency_words", 5)
            dim_consistency_diversity = getattr(cfg.eval, "dim_consistency_diversity", True)
            dim_consistency_model = getattr(cfg.eval, "dim_consistency_model", None)
            dim_consistency_method = getattr(cfg.eval, "dim_consistency_method", "score")
            if dim_consistency_method not in ("score", "similarity", "both"):
                raise ValueError(
                    f"cfg.eval.dim_consistency_method must be one of "
                    f"'score', 'similarity', 'both'; got {dim_consistency_method!r}"
                )
            # logging
            rec_log_every = cfg.eval.rec_log_every
            rec_log_every = rec_log_every or rec_check_every
            time_iteration = cfg.eval.time_iteration
            # GPU pool trim cadence: explicit value, else default to the sem-check cadence.
            pool_trim_every = cfg.eval.pool_trim_every
            if pool_trim_every is None:
                pool_trim_every = sem_check_every
            # saving
            save_intermediate = cfg.eval.save_intermediate


        except Exception as e:
            raise ValueError(f"Check config structure: {e}")

        if not isinstance(self, SparseTupleTensor):
            raise TypeError("sparse_tensor must be a SparseTupleTensor instance.")
        _is_sgd = solver == "sgd"
        if _is_sgd:
            # The SGD trainer is torch-native; the tensor from load_from_disk is
            # already a torch sparse COO, so no CuPy conversion ever happens.
            if not self.sparsity_type == "torch":
                raise ValueError("solver='sgd' needs sparsity_type 'torch' "
                                 f"(got {self.sparsity_type!r}); skip tensor_to_sparse('cupy').")
        elif not self.sparsity_type == "cupy":
            raise ValueError("sparse_tensor must have sparsity_type 'cupy'.")

        # --- SGD guard rails: reject MU-only knobs whose semantics
        # don't carry over to a minibatch solver (reinterpretation is deferred).
        if _is_sgd:
            if decomposition != "tucker":
                raise NotImplementedError(
                    f"solver='sgd' with decomposition={decomposition!r} is not implemented "
                    "(sgd/README.md, deferred). Use decomposition='tucker'."
                )
            if getattr(cfg.exp, "subsample_frac", 1.0) < 1.0:
                raise ValueError(
                    "solver='sgd' is already minibatch; subsample_frac < 1.0 has no "
                    "meaning there. Use --sgd-batch-size instead."
                )
            if getattr(cfg.exp, "max_nnz", None):
                raise ValueError(
                    "solver='sgd' does not support max_nnz (MU per-step NNZ ceiling). "
                    "Use --sgd-batch-size instead."
                )
            if isinstance(init, str) and "svd" in init:
                raise ValueError(
                    "solver='sgd' does not support SVD init (CuPy routine). Use "
                    "--init random or --sgd-warm-start <MU model .pt>."
                )
            if normalize_factors:
                raise ValueError(
                    "solver='sgd' does not support normalize_factors=True (scaling "
                    "lives in the softplus/clamp parametrization)."
                )
            if largedim:
                raise ValueError(
                    "solver='sgd' does not use the largedim kernel family; "
                    "drop --largedim."
                )

        # --- EXPERIMENTAL CP swap points -----------------------------------
        # For decomposition == "cp" the `core` variable holds the CP weight
        # vector λ (R,): the CP kernels take it through the same UpdateRouting
        # seam, so this loop is reused rather than forked. Containers become
        # tensorly CPTensor payloads.
        # For decomposition == "tt" it holds the list of TT cores instead (same
        # seam, see experimental/TT_hybrid/README.md).
        _is_cp = decomposition == "cp"
        _is_tt = decomposition == "tt"
        if _is_cp:
            from tensorly.cp_tensor import CPTensor as TensorModel
        elif _is_tt:
            from tensormet.experimental.TT_hybrid.tt_decomposition import (
                TuckerTTTensor as TensorModel,
            )
        else:
            TensorModel = TuckerTensor

        paths = cfg.artifact_paths()

        # Under HPC mode `paths` resolve to node-local $TMPDIR. mirror_paths is the
        # canonical GPFS destination: periodic checkpoints are mirrored there
        # during the run (via hpc_helpers.mirror_checkpoint) so a walltime kill —
        # which never runs the end-of-job copy-back — loses at most one checkpoint
        # interval and stays resumable.
        mirror_paths = cfg.artifact_paths(staged=False) if cfg.train.hpc else None

        if checkpoint_saving:
            os.makedirs(paths["checkpoint_dir"], exist_ok=True)

        # --- RESUME STATE FETCHING ---
        resume_state = cfg.get_resume_state()
        start_iteration = resume_state.get("start_iteration", 0)
        best_sem_score = resume_state.get("best_sem_score", 0.0)
        rec_errors = resume_state.get("rec_errors", [])
        fitness_scores = resume_state.get("fitness_scores", [])
        checkpoint_tensor = resume_state.get("checkpoint_tensor", None)

        shape = tuple(self.shape)
        if _is_cp:
            # CP uses a single rank R (no per-mode core dimensions). Accept the
            # config's per-mode tuple only when uniform.
            from tensorly.cp_tensor import validate_cp_rank
            if isinstance(rank, (list, tuple)):
                if len(set(rank)) > 1:
                    raise ValueError(
                        f"CP decomposition uses a single rank; got per-mode ranks {tuple(rank)}. "
                        f"Pass a uniform --rank."
                    )
                rank = rank[0]
            rank = validate_cp_rank(shape, rank=int(rank))
            modes = list(range(len(shape)))
        else:
            rank = validate_tucker_rank(shape, rank=rank)
            modes = list(range(len(rank)))
        # NOTE (Task 6): the routing size decision is centralised in
        # needs_largedim(dim, ...) (see _largedim_selected below); no local
        # max(shape) threshold variable is needed here anymore.
        # SGD checkpoints are dict payloads carrying optimizer state alongside
        # the (core, factors) views; they are consumed by the trainer below,
        # never by the CuPy init path. Cross-solver loads are impossible by
        # construction (distinct SGD{order}D stems + the solver resume key),
        # but keep the payload check as a last line of defense.
        _sgd_resume_payload = None
        if _is_sgd and checkpoint_tensor is not None:
            if not (isinstance(checkpoint_tensor, dict)
                    and checkpoint_tensor.get("solver") == "sgd"):
                raise ValueError(
                    "solver='sgd' found a non-SGD checkpoint payload; refusing to "
                    "resume from it (MU checkpoints carry no optimizer state)."
                )
            _sgd_resume_payload = checkpoint_tensor
            checkpoint_tensor = None

        if _is_sgd:
            # Init (random or warm start) lives inside SGDTrainer; `core` and
            # `factors` stay undefined on this path — every downstream consumer
            # goes through _sgd_trainer.materialize() instead.
            core = factors = None
        elif checkpoint_tensor is not None:
            if isinstance(checkpoint_tensor, tuple):
                # if TensorLy TuckerTensor / plain (core|weights, factors) tuple
                ckpt_core, ckpt_factors = checkpoint_tensor
            elif _is_cp or _is_tt:
                # CPTensor / TuckerTTTensor payload: iterable as (core, factors)
                ckpt_core, ckpt_factors = checkpoint_tensor
            else:
                # if our TuckerDecomposition class
                ckpt_core, ckpt_factors = checkpoint_tensor.core, checkpoint_tensor.factors

            core = [cp.asarray(C) for C in ckpt_core] if _is_tt else cp.asarray(ckpt_core)
            factors = [cp.asarray(factor) for factor in ckpt_factors]
            if _is_tt:
                from tensormet.experimental.TT_hybrid.tt_chain import core_shapes
                expected = core_shapes(rank, tt_rank)
                if [tuple(int(d) for d in C.shape) for C in core] != expected:
                    raise ValueError(
                        f"decomposition='tt' resumed a checkpoint whose TT cores have "
                        f"shapes {[tuple(C.shape) for C in core]}; expected {expected} "
                        f"for rank={rank}, tt_rank={tt_rank}. Delete it or match the config."
                    )
            if _is_cp and core.ndim != 1:
                # A CP-named checkpoint whose "λ" is not a vector is a Tucker
                # payload; resuming would treat its core as weights.
                raise ValueError(
                    f"decomposition='cp' resumed a checkpoint whose core has shape "
                    f"{tuple(core.shape)}; expected a 1-D weight vector. This is a "
                    f"Tucker payload under a CP filename — delete it and restart."
                )
        elif _is_cp:
            from tensormet.experimental.CP.cp_ops import initialize_nonnegative_cp
            # `core` = λ weight vector; see the CP swap-points note above.
            core, factors = initialize_nonnegative_cp(
                self.tensor, shape, rank, modes, init, random_state,
                thread_budget=thread_budget, divergence=divergence, epsilon=epsilon,
            )
        elif _is_tt:
            from tensormet.experimental.TT_hybrid.tt_ops import initialize_tucker_tt
            # `core` = list of TT cores; see the swap-points note above.
            core, factors = initialize_tucker_tt(
                self.tensor, shape, rank, modes, init, random_state,
                tt_rank=tt_rank, thread_budget=thread_budget, epsilon=epsilon,
            )
        else:
            core, factors = initialize_nonnegative_tucker(self.tensor, shape, rank, modes, init,
                                                           random_state, thread_budget=thread_budget)

        # --- multi-GPU shard initialisation ---

        _n_gpus = getattr(cfg.train, "n_gpus", 1)
        _subsample_frac = getattr(cfg.exp, "subsample_frac", 1.0)
        _subsample_warmup = getattr(cfg.train, "subsample_warmup", 0)

        # --- max_nnz: hard global ceiling on NNZ per update step ---
        # Resolved here into an effective fraction: every downstream frac<1.0
        # gate, the one-time shuffles, the rotating windows and the 1/frac
        # rescales all key off _subsample_frac, and per-shard sampling of
        # frac_eff sums to ~max_nnz regardless of shard count (overshoot is at
        # most 1 element per shard from rounding). Filenames and resume checks
        # carry the raw max_nnz int, never this derived value.
        _max_nnz = int(getattr(cfg.exp, "max_nnz", None) or 0)
        if _max_nnz < 0:
            raise ValueError(f"cfg.exp.max_nnz must be >= 0, got {_max_nnz}")
        # coords_nnz covers both CuPy forms; the SGD path holds a torch sparse COO.
        _full_nnz = (int(self.tensor._nnz()) if _is_sgd
                     else coords_nnz(self.tensor))
        if _max_nnz and _full_nnz > _max_nnz:
            _subsample_frac = min(_subsample_frac, _max_nnz / _full_nnz)
            print(f"max_nnz={_max_nnz}: effective subsample_frac -> "
                  f"{_subsample_frac:.6g} (nnz={_full_nnz})")

        # Masked / completion objective: fit only observed entries (see RunConfig.exp.objective).
        if objective not in ("full", "masked"):
            raise ValueError(f"cfg.exp.objective must be 'full' or 'masked', got {objective!r}")
        masked = objective == "masked"
        # SVD init computes a zero-filled HOSVD (it treats unobserved entries as 0),
        # which is the same "full" assumption the masked objective is meant to avoid.
        # Masked-aware SVD init is not implemented; warn that this combination is
        # likely to hurt rather than help and that random init is the safer choice.
        if masked and isinstance(init, str) and "svd" in init:
            print(
                f"WARNING: init={init!r} with objective='masked' is not implemented. "
                "SVD init fits a zero-filled HOSVD (unobserved entries treated as 0), "
                "which contradicts the masked/completion objective and will probably "
                "worsen results. Consider --init random."
            )
        # Single-GPU stochastic subsampling rescales NNZ values by 1/frac, but the
        # single-GPU masked kernels don't receive `frac` to rescale their (observed-only)
        # denominators to match, which would bias the MU ratio. The multi-GPU sharded
        # path handles this rescaling internally, so masked+subsample is supported there.
        if masked and _n_gpus == 1 and _subsample_frac < 1.0:
            raise NotImplementedError(
                "objective='masked' with subsampling (subsample_frac < 1.0 or a binding "
                "max_nnz) is only supported on the multi-GPU sharded path (n_gpus > 1). "
                "Use subsample_frac=1.0 and max_nnz=0 on a single GPU."
            )

        # --- CP guard rails: paths the CP kernel family does not implement yet.
        if _is_cp and masked:
            raise NotImplementedError(
                "decomposition='cp' does not support objective='masked' yet "
                "(CP_IMPLEMENTATION_PLAN.md §1.8 / Phase 5). Use objective='full'."
            )

        # --- Tucker-TT guard rails (experimental/TT_hybrid/README.md).
        if _is_tt and masked:
            raise NotImplementedError(
                "decomposition='tt' does not support objective='masked' yet "
                "(TT_hybrid/README.md, 'Not implemented'). Use objective='full'."
            )

        # --- SGD trainer construction ---
        _sgd_trainer = None
        if _is_sgd:
            _sgd_init = init
            if _sgd_resume_payload is None and sgd_warm_start:
                # Warm start (init from an MU model artifact) is distinct from
                # resume: parameters start at the MU solution but the optimizer
                # state and step counter start fresh. The artifact is a plain
                # CPU-numpy TuckerTensor (tuple-unpackable); accept a
                # TuckerDecomposition payload's .core/.factors as a fallback.
                _ws = torch.load(sgd_warm_start, map_location="cpu", weights_only=False)
                try:
                    _ws_core, _ws_factors = _ws
                except (TypeError, ValueError):
                    _ws_core, _ws_factors = _ws.core, _ws.factors
                _sgd_init = (_ws_core, _ws_factors)
                print(f"SGD warm start from {sgd_warm_start}")
            _sgd_kwargs = dict(
                rank=rank, divergence=divergence, objective=objective,
                lr=sgd_lr, batch_size=sgd_batch_size, optimizer=sgd_optimizer,
                parametrization=sgd_parametrization,
                shared_factors=self.shared_factors, init=_sgd_init,
                random_state=random_state,
                steps_per_iteration=sgd_steps_per_iteration, epsilon=epsilon,
                micro_batch=sgd_micro_batch, cuda_graph=sgd_cuda_graph,
                eval_sample=sgd_eval_sample,
                resume_payload=_sgd_resume_payload,
            )
            # Construction is the last silent stretch before the loop announces
            # itself, and on the sharded path it is where the collective gets
            # built — historically the place a run could stall with no output.
            print(f"[{time.strftime('%H:%M:%S')}] constructing SGD trainer "
                  f"(n_gpus={_n_gpus})...", flush=True)
            if _n_gpus > 1:
                from tensormet.sgd.sharded_sgd import ShardedSGDTrainer
                _sgd_trainer = ShardedSGDTrainer(
                    self.tensor, device_ids=list(range(_n_gpus)),
                    # Multi-GPU-only knobs; the single-GPU trainer has no
                    # collective, no batch scope and no sync cadence.
                    batch_scope=sgd_batch_scope, sync_every=sgd_sync_every,
                    comm_backend=sgd_comm_backend, **_sgd_kwargs
                )
            else:
                from tensormet.sgd.sgd_trainer import SGDTrainer
                _sgd_trainer = SGDTrainer(self.tensor, **_sgd_kwargs)
            print(f"[{time.strftime('%H:%M:%S')}] SGD trainer ready", flush=True)

        if _n_gpus > 1 and not _is_sgd:
            _sst = ShardedSparseTensor.from_coo(
                self.tensor, shape, device_ids=list(range(_n_gpus)),
                subsample_frac=_subsample_frac, masked=masked,
                # CHANGED (2026-06-12 review, Task 2): seeds the one-time per-shard
                # NNZ shuffle that backs contiguous-window subsampling.
                subsample_seed=int(cfg.exp.random_state or 0),
            )
        else:
            _sst = None

        # --- stochastic subsampling (single-GPU path) ---
        # CHANGED (2026-06-12 review, Task 2): CooSubsampler shuffles the NNZ once
        # here and serves contiguous rotating windows per iteration, replacing the
        # stateful RNG + per-iteration full permutation of subsample_coo. Samples
        # are now a pure function of (random_state, iteration), so resumed runs
        # draw the same sequence as uninterrupted ones (review finding I-3).
        _iter_sampler = (
            CooSubsampler(self.tensor, shape, _subsample_frac, cfg.exp.random_state)
            if (_subsample_frac < 1.0 and _sst is None) else None
        )

        # --- per-mode NNZ grouping cache (single-GPU largedim path) ---
        # CHANGED (2026-06-12 review, Task 3 — E-1/E-2/E-3): when the single-GPU
        # largedim factor kernels run on a static tensor, precompute each mode's
        # column grouping (sort + unique + segment offsets) once here and reuse it
        # every iteration, replacing the per-iteration decode + cp.unique + per-batch
        # cp.where scan. Mirror routing.get_update_routing_step's factor decision so
        # the cache is built only when the factor function is actually a largedim
        # kernel (the only one that accepts `grouping`). Disabled under subsampling
        # (sampled NNZ change every iteration) and on the multi-GPU path (the SST
        # owns its own per-shard caches).
        # CHANGED (2026-06-12 review, Task 6): the dense vs. largedim decision is
        # now the single needs_largedim() predicate, shared by routing, this cache
        # gate AND the multi-GPU override below — so all of them agree on the
        # selected family (previously this gate used divergence-specific 4000
        # literals that could disagree with routing for FR dims in (3000, 4000]).
        _largedim_selected = needs_largedim(dim, largedim=largedim, masked=masked)
        _factor_is_largedim = _largedim_selected
        # The CP and TT kernels stream NNZ directly (no column grouping, own
        # batch estimator), so this cache and the batch sizes below stay None there.
        _grouping_cache = (
            NNZGroupingCache(self.tensor, shape)
            if (_sst is None and _subsample_frac >= 1.0 and _factor_is_largedim
                and not _is_cp and not _is_tt and not _is_sgd)
            else None
        )

        # --- per-iteration batch sizes (single-GPU largedim KL path) ---
        # CHANGED (2026-06-15): the largedim KL factor/core/error kernels sized
        # their column/NNZ batches by calling _estimate_batch_*() every update,
        # and each estimate flushed the CuPy pool to the driver (cudaFree +
        # re-cudaMalloc), stalling the GPU to idle ~6-7×/iteration. The estimates
        # depend only on core/factor shapes+dtype (fixed for the run), so compute
        # them ONCE here and thread them into the kernels' batch_* kwargs. Gated
        # to the KL single-GPU largedim path (FR/sharded/plain kernels size
        # themselves; the SST owns its own batching). nnz_live reserves the
        # kernels' transient decode arrays so hoisting keeps Task 1's headroom.
        # (_full_nnz is computed once above, alongside the max_nnz resolution.)
        _nnz_live = _iter_sampler.n_sample if _iter_sampler is not None else _full_nnz
        _largedim_batches = (
            precompute_largedim_batches(core, factors, modes, masked=masked, nnz_live=_nnz_live,
                                        coord_backed=isinstance(self.tensor, CoordCOO))
            if (divergence == "kl" and _sst is None and _factor_is_largedim
                and not _is_cp and not _is_tt and not _is_sgd)
            else None
        )

        linked_factors = defaultdict(set)
        if self.shared_factors:
            for a, b in self.shared_factors:
                linked_factors[a].add(b)
                linked_factors[b].add(a)

        no_rec_improve_steps = 0
        # If we resumed, grab the last known error to calculate early stopping diff accurately
        last_err = rec_errors[-1] if rec_errors else None

        sem_no_rec_improve_steps = 0

        # Ensure 'best' variables are initialized safely so returning them at the end doesn't fail.
        # NOTE: on resume, best_core/best_factors start as the *checkpoint* tensors, not
        # necessarily the historical best-scoring model on disk — they are only replaced
        # below if semantics improve past the resumed best_sem_score. If semantics never
        # improve during this run, the "best" tensor returned is just the latest checkpoint,
        # labeled with the resumed score.
        if _is_sgd:
            # Host-numpy snapshots (the SGD path never holds CuPy arrays); the
            # save sites below handle numpy transparently via _as_host.
            best_core, best_factors = _sgd_trainer.materialize()
        else:
            # TT: `core` is a list of cores, so copy element-wise (list.copy is shallow).
            best_core = [C.copy() for C in core] if _is_tt else core.copy()
            best_factors = [f.copy() for f in factors]
        best_sem_iteration = start_iteration if start_iteration > 0 else None

        # Decide once which semantic metric drives patience/diff.
        # cfg.eval.sem_primary_key can override the auto-derived default.
        if cfg.eval.sem_primary_key is not None:
            sem_primary_key = cfg.eval.sem_primary_key
        elif sem_error_type == "all":
            sem_primary_key = "average_rank_score"  # stable default (your dict always includes this)
        elif isinstance(sem_error_type, (list, tuple)):
            if len(sem_error_type) == 0:
                raise ValueError("sem_error_type list/tuple must contain at least one key.")
            sem_primary_key = sem_error_type[0]
        else:
            sem_primary_key = sem_error_type

        simlex_pairs = None
        if _SIMLEX_PATH.exists():
            try:
                simlex_pairs = load_simlex(_SIMLEX_PATH)
            except Exception as _e:
                print(f"Warning: could not load SimLex-999 from {_SIMLEX_PATH}: {_e}")

        # --- dimension-consistency judge (optional, default off) ---
        # The judge model (~1 GB fp16 on GPU for the default 0.5B judge or 3.5GB for the 2B one)
        # is loaded HERE, before the iteration loop sizes any GPU batches — NOT lazily at the
        # first semantic check. The per-iteration batch estimators size against free
        # VRAM (_gpu_free_bytes); a judge added mid-run steals ~1 GB that batches
        # sized beforehand already assumed was free (acute on resumed runs, where
        # the pool is grown to fill the GPU before the first check), OOMing the next
        # factor update. Loading up-front makes every batch estimate account for it.
        # Import is deferred so runs without the flag never touch transformers.
        _dim_judge = None
        if dim_consistency:
            from tensormet.judge import DimConsistencyJudge, DEFAULT_JUDGE_MODEL
            _judge_model_name = dim_consistency_model or DEFAULT_JUDGE_MODEL
            _dim_judge = DimConsistencyJudge(
                model_name=_judge_model_name,
                num_dim_words=dim_consistency_words,
                diversity_aware=dim_consistency_diversity,
            )
            print(
                f"Dimension-consistency scoring enabled (method={dim_consistency_method!r}, "
                f"judge={_judge_model_name!r}, words/dim={dim_consistency_words}, "
                f"diversity_aware={dim_consistency_diversity}). "
                f"Loading the judge now so the decomposition's batch sizing "
                f"accounts for its (model-dependent) GPU footprint from the first iteration."
            )
            # Return CuPy's cached-but-unused blocks to the driver so torch gets
            # real headroom, then load the model before the loop begins. The SGD
            # path never allocates through CuPy; its equivalent is torch's own
            # caching allocator, which the judge shares anyway.
            if _is_sgd:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            elif _sst is not None:
                _sst.trim_pools()
            else:
                cp.get_default_memory_pool().free_all_blocks()
                cp.get_default_pinned_memory_pool().free_all_blocks()
            _dim_judge.ensure_loaded()

        # TT streams NNZ with its own estimator, which depends only on the core/
        # factor shapes and dtype — fixed for the run. Size it once here (after
        # the judge is resident, so its footprint is accounted for) instead of
        # once per factor update, core sweep and error pass.
        _tt_batch_nnz = None
        if _is_tt:
            from tensormet.experimental.TT_hybrid.tt_ops import estimate_batch_nnz_tt
            # _gpu_free_bytes() reads whichever device is current; pin to the one
            # the cores live on so a sharded run never sizes against a shard's GPU.
            with core[0].device:
                _tt_batch_nnz = estimate_batch_nnz_tt(core, factors)

        print(divergence, rank, _subsample_frac)
        # CHANGED (2026-06-12 review, Task 6): announce the routing family chosen by
        # the unified needs_largedim() predicate. Sharding engages iff largedim does
        # (and n_gpus > 1), so the three cases below are mutually exclusive.
        if _is_cp:
            _selected_path = (f"cp (nnz-streaming, inner_iters={cp_inner_iters}"
                              + (f", sharded×{_n_gpus}" if _sst is not None else "")
                              + ")")
        elif _is_tt:
            _selected_path = (f"tucker-tt (nnz-streaming, bonds="
                              f"{[int(C.shape[2]) for C in core[:-1]]}"
                              + (f", sharded×{_n_gpus}" if _sst is not None else "")
                              + ")")
        elif _is_sgd:
            _selected_path = (f"sgd (torch, sharded×{_n_gpus})" if _n_gpus > 1
                              else "sgd (torch)")
        elif _sst is not None and _largedim_selected:
            _selected_path = f"sharded×{_n_gpus}"
        elif _largedim_selected:
            _selected_path = "largedim"
        else:
            _selected_path = "small-dim dense"
        print(f"routing path: {_selected_path} (max_dim={max(dim) if isinstance(dim, (tuple, list)) else dim}, "
              f"LARGEDIM_THRESHOLD reached={_largedim_selected}, n_gpus={_n_gpus})")
        # est_iter_time = self.estimate_training_time(rank=rank[0], subsample=_subsample_frac)
        # print(f"estimated training time: {est_iter_time}*{n_iter_max}={est_iter_time*n_iter_max}")

        # --- graceful-stop handler ---
        # First Ctrl+C (SIGINT): request a resumable save at the end of the current
        # iteration, then restore the default handler so a SECOND Ctrl+C exits
        # immediately (as it did before this handler existed). signal.signal only
        # works from the main thread; if we're on a worker thread, skip gracefully.
        _interrupt_requested = {"flag": False}
        _original_sigint = None
        _sigint_installed = False

        def _handle_interrupt(signum, frame):
            print(
                "\nInterrupt received: will save a resumable checkpoint at the end of "
                "the current iteration. Press Ctrl+C again to exit immediately."
            )
            _interrupt_requested["flag"] = True
            # Restore prior handler so a second Ctrl+C behaves as it used to (raises).
            signal.signal(signal.SIGINT, _original_sigint)

        try:
            _original_sigint = signal.signal(signal.SIGINT, _handle_interrupt)
            _sigint_installed = True
        except ValueError:
            # Not running in the main thread; leave default interrupt behaviour intact.
            _sigint_installed = False

        # Wall-clock of the decomposition loop itself (excludes data loading,
        # sparse conversion and process startup — those are captured by
        # launch.py's runtime_seconds). Persisted so benchmarks can report a
        # decomposition time distinct from total process runtime.
        _decomp_loop_start = time.time()
        # Per-iteration solve time (updates + error kernel), excluding the
        # semantic evaluation that also runs inside the loop. Summed into
        # solve_seconds below so benchmarks can separate "time spent actually
        # decomposing" from eval and data-loading overhead. GPU work is
        # asynchronous, so the timer is bracketed by sync_devices() — without
        # it the per-iteration numbers measure kernel *queueing* and the cost
        # lands on whichever later iteration happens to block.
        # CHANGED (2026-08-04, perf regression fix): the barriers run on log
        # steps only. Syncing every iteration serialized the loop against the
        # devices and was part of the Aug-03 iteration-time regression; the
        # cost of the looser bracketing is only that a non-log iteration's
        # queued tail is charged to the next log step's time=.
        _iter_seconds = []
        _sync_backend = "torch" if _is_sgd else "cupy"
        # Initialized so post-loop bookkeeping (e.g. "iterations": iteration + 1) is well
        # defined even if the loop body never executes (start_iteration >= n_iter_max).
        iteration = start_iteration - 1
        for iteration in range(start_iteration, n_iter_max):
            log_step = get_log_step(iteration, rec_log_every, rec_check_every)
            if log_step:
                sync_devices(_n_gpus, _sync_backend)
            _iter_start = time.time()
            if _is_sgd:
                # --- SGD solver (sgd/README.md) ---
                # One iteration = a block of sgd_steps_per_iteration optimizer
                # steps run inside the trainer (torch-native; Adam moments, raw
                # softplus params and the global step counter live there — state
                # the UpdateRouting seam cannot express, hence this branch).
                # On log steps it returns the exact full-NNZ relative error with
                # the same normalization as the MU error kernels, so everything
                # downstream (patience, logging, checkpointing) is shared.
                rel_err = _sgd_trainer.run_block(iteration, log_step)
            else:
                routing = get_update_routing_step(divergence=divergence, dim=dim, log_step=log_step,
                                                  largedim=largedim, masked=masked,
                                                  decomposition=decomposition,
                                                  cp_inner_iters=cp_inner_iters,
                                                  cp_scooch_kappa=cp_scooch_kappa)
                # --- multi-GPU routing override (largedim variants only) ---
                # CHANGED (2026-06-12 review, Task 6): gate on the same needs_largedim()
                # predicate as routing (via _largedim_selected) instead of re-deriving
                # divergence-specific 4000 literals. Sharding now engages iff the
                # largedim path is selected and a shard set exists (n_gpus > 1).
                # CP and TT each have one kernel family, so _largedim_selected
                # does not apply to them.
                if _is_cp and _sst is not None:
                    from tensormet.experimental.CP.cp_routing import (
                        get_sharded_cp_update_routing_step,
                    )
                    routing = get_sharded_cp_update_routing_step(
                        _sst, divergence=divergence, log_step=log_step,
                        inner_iters=cp_inner_iters, scooch_kappa=cp_scooch_kappa,
                    )
                elif _is_tt and _sst is not None:
                    from tensormet.experimental.TT_hybrid.tt_routing import (
                        get_sharded_tt_update_routing_step,
                    )
                    routing = get_sharded_tt_update_routing_step(
                        _sst, divergence=divergence, log_step=log_step,
                    )
                elif _sst is not None and _largedim_selected:
                    if divergence == "kl":
                        routing = UpdateRouting(
                            factor_update=make_sharded_kl_factor_update(_sst),
                            core_update=make_sharded_kl_core_update(_sst),
                            error_fn=make_sharded_kl_compute_errors(_sst) if log_step else null_compute_errors,
                            core_returns_error=routing.core_returns_error,
                        )
                    elif divergence == "fr":
                        routing = UpdateRouting(
                            factor_update=make_sharded_fr_factor_update(_sst),
                            core_update=make_sharded_fr_core_update(_sst),
                            error_fn=make_sharded_fr_compute_errors(_sst) if log_step else null_compute_errors,
                            core_returns_error=False,  # sharded core update never returns (core, error)
                        )
                # --- stochastic tensor selection ---
                if _sst is not None:
                    _sst.set_iter_seed(iteration)
                _use_subsample = (
                    _subsample_frac < 1.0
                    and iteration >= _subsample_warmup
                    and _sst is None   # multi-GPU handles sampling internally
                )
                _current_tensor = (
                    _iter_sampler.sample(iteration)
                    if _use_subsample else self.tensor
                )
                # --- factors ---
                for mode in modes:
                    _factor_kwargs = dict(
                        vec_tensor=_current_tensor,
                        core=core,
                        factors=factors,
                        mode=mode,
                        shape=shape,
                        thread_budget=thread_budget,
                        epsilon=epsilon,
                        verbose=verbose,
                    )
                    # CHANGED (2026-06-12 review, Task 3): hand the largedim factor
                    # kernel its cached per-mode grouping (built lazily on first use)
                    # so it skips the decode/unique/scan. Only set when the cache is
                    # active (single-GPU largedim, no subsampling); the SST path caches
                    # internally and other kernels never receive this kwarg.
                    if _grouping_cache is not None:
                        _factor_kwargs["grouping"] = _grouping_cache.get(mode)
                    # Precomputed col-batch size (largedim KL factor kernel only);
                    # skips the per-update _estimate_batch_cols_for_Z + pool flush.
                    if _largedim_batches is not None:
                        _factor_kwargs["batch_cols"] = _largedim_batches["batch_cols"][mode]
                    if _tt_batch_nnz is not None:
                        _factor_kwargs["batch_nnz"] = _tt_batch_nnz
                    factors[mode] = routing.factor_update(**_factor_kwargs)

                    # new: factor linking
                    if mode in linked_factors:
                        for other in linked_factors[mode]:
                            factors[other] = factors[mode]

                # --- core + error ---
                if routing.core_returns_error:
                    # FR: combined core update + error in one call.
                    # CHANGED (2026-06-12 review, Task 5 — I-1): feed the FULL tensor,
                    # not the subsampled/rescaled _current_tensor. This call only fires
                    # on log steps (routing sets core_returns_error = True*log_step), so
                    # the fused error portion (norm_X², ⟨X,X̂⟩) is unbiased; the core MU
                    # step it also performs is then the exact full-NNZ update that step.
                    core, rel_err = routing.core_update(
                        vec_tensor=self.tensor,
                        shape=shape,
                        core=core,
                        factors=factors,
                        modes=modes,
                        thread_budget=thread_budget,  # we always pass it, even if not needed, to ensure consistency
                        epsilon=epsilon,
                        verbose=verbose
                    )
                else:
                    # KL: core update, then compute error separately
                    _core_kwargs = dict(
                        vec_tensor=_current_tensor,
                        shape=shape,
                        core=core,
                        factors=factors,
                        modes=modes,
                        thread_budget=thread_budget,
                        epsilon=epsilon,
                        verbose=verbose,
                    )
                    # Precomputed NNZ-batch sizes (largedim KL core kernel only);
                    # skips the per-update _estimate_batch_* + pool flush.
                    if _largedim_batches is not None:
                        _core_kwargs["batch_rhat"] = _largedim_batches["batch_rhat"]
                        _core_kwargs["batch_num"] = _largedim_batches["batch_num"]
                    if _tt_batch_nnz is not None:
                        _core_kwargs["batch_nnz"] = _tt_batch_nnz
                    core = routing.core_update(**_core_kwargs)

                    # CHANGED (2026-06-12 review, Task 5 — I-1): the KL error runs on
                    # the FULL tensor, not the subsampled/rescaled _current_tensor.
                    # x·log(x/r), the sum_R_nz zero-correction, and ‖X‖ are nonlinear
                    # in the rescaled values, so a subsampled error is biased by frac.
                    # error_fn only does real work on log steps (else null_compute_errors),
                    # so full-NNZ evaluation is cheap. The core update above keeps using
                    # _current_tensor (its MU numerator is linear → 1/frac-unbiased).
                    _err_kwargs = dict(
                        vec_tensor=self.tensor,
                        shape=shape,
                        core=core,
                        factors=factors,
                        thread_budget=thread_budget,
                        epsilon=epsilon,
                        verbose=verbose,
                    )
                    # Only the real largedim error fn accepts batch_rhat; on non-log
                    # steps error_fn is null_compute_errors (no such kwarg).
                    if _largedim_batches is not None and log_step:
                        _err_kwargs["batch_rhat"] = _largedim_batches["batch_rhat"]
                    # null_compute_errors takes no batch kwarg on non-log steps.
                    if _tt_batch_nnz is not None and log_step:
                        _err_kwargs["batch_nnz"] = _tt_batch_nnz
                    rel_err = routing.error_fn(**_err_kwargs)
                # CP and TT skip this: their factor updates already keep the
                # columns normalized, with the scale absorbed into λ / the TT
                # core of that mode (tucker_normalize needs a dense core anyway).
                if normalize_factors and not _is_cp and not _is_tt:
                    core, factors = tucker_normalize((core, factors))

            if log_step:
                sync_devices(_n_gpus, _sync_backend)
            _iter_seconds.append(time.time() - _iter_start)

            if log_step:
                rec_errors.append(rel_err)

                # ---- reconstruction + patience ----
                has_prev_err = len(rec_errors) >= 2
                if has_prev_err:
                    delta = rec_errors[-2] - rec_errors[-1]


                    message = f"{iteration}: reconstruction error={rec_errors[-1]} (Δ={delta:+.3e})"
                    if time_iteration:
                        message += f", time={_iter_seconds[-1]}"
                    print(message)

                do_rec_check = (
                        rec_check_every > 0
                        and (iteration + 1) % rec_check_every == 0
                )
                # patience only after warmup and once we have a previous error
                if do_rec_check:
                    if rel_err is None:
                        raise ValueError("error should always be available on error checking steps")

                    if last_err is None:
                        last_err = rel_err
                    elif iteration >= warmup_steps:
                        imp_val = abs(float(last_err - rel_err))
                        if imp_val < tol:
                            no_rec_improve_steps += 1
                            if verbose:
                                print(f"No significant change: {no_rec_improve_steps}/{patience} (Δ={imp_val:.3e})")
                            if no_rec_improve_steps >= patience:
                                if verbose:
                                    notify_discord(
                                        f"Stopped after {no_rec_improve_steps} non-improving steps "
                                        f"(patience={patience}). Converged at iteration {iteration} with final error {rec_errors[-1]}",
                                        job_finished=False,
                                    )
                                break
                        else:
                            if no_rec_improve_steps:
                                print(f"Improved (Δ={imp_val:.3e}); resetting patience counter.")
                            no_rec_improve_steps = 0
                        last_err = rel_err

            # ---- similarity evaluation + semantic patience ----
            do_sem_check = (
                    sample_sentences is not None
                    and vocab is not None
                    and sem_check_every > 0
                    and (iteration + 1) % sem_check_every == 0
            )

            if do_sem_check:
                # The SGD path already runs on the pytorch backend; for MU this
                # flips cupy -> pytorch for the eval stack (restored below).
                tl.set_backend("pytorch")
                if _is_sgd:
                    _sem_core_np, _sem_factors_np = _sgd_trainer.materialize()
                    core_cpu = tl.tensor(_sem_core_np)
                    factors_cpu = [tl.tensor(f) for f in _sem_factors_np]
                elif _is_tt:
                    core_cpu = [tl.tensor(cp.asnumpy(C)) for C in core]
                    factors_cpu = [tl.tensor(cp.asnumpy(f)) for f in factors]
                else:
                    core_cpu = tl.tensor(cp.asnumpy(core))
                    factors_cpu = [tl.tensor(cp.asnumpy(f)) for f in factors]
                roles = extract_roles_from_vocab(vocab)
                if _is_cp:
                    # Same eval contract (batch_excluded_role_vector, judge and
                    # SimLex accessors), so everything downstream is unchanged.
                    from tensormet.experimental.CP.cp_decomposition import CPDecomposition
                    tucker_decomp = CPDecomposition(weights=core_cpu, factors=factors_cpu,
                                                    vocab=vocab, roles=roles)
                elif _is_tt:
                    from tensormet.experimental.TT_hybrid.tt_decomposition import TuckerTTDecomposition
                    tucker_decomp = TuckerTTDecomposition(tt_cores=core_cpu, factors=factors_cpu,
                                                          vocab=vocab, roles=roles)
                else:
                    tucker_decomp = TuckerDecomposition(core=core_cpu, factors=factors_cpu, vocab=vocab, roles=roles)

                sem_out = evaluate_sample(
                    tucker_decomp,
                    sample_sentences,
                    sampled=True,
                    seed=random_state,
                    thread_budget=thread_budget,
                    return_type=sem_error_type,
                    softmax_temperature=sem_softmax_temperature,
                )

                if simlex_pairs is not None:
                    try:
                        vecs_by_pos = {"N": {}, "V": {}, "A": {}}
                        for _ri, _role in enumerate(roles):
                            _pos = _SIMLEX_POS_MAP.get(_role)
                            if _pos is None or vecs_by_pos[_pos]:
                                continue
                            _fmat = tucker_decomp.factors[_ri].detach().cpu().numpy()
                            _norms = np.maximum(np.linalg.norm(_fmat, axis=1, keepdims=True), 1e-12)
                            _fmat = _fmat / _norms
                            _vkey = voc_index(_role)
                            if _vkey in tucker_decomp.vocab:
                                vecs_by_pos[_pos] = {w: _fmat[idx] for w, idx in tucker_decomp.vocab[_vkey].items()}
                        # Fallback for positional/n-gram models: all modes share the same
                        # vocab, so any role can supply vectors for empty POS slots.
                        empty_pos = [p for p, v in vecs_by_pos.items() if not v]
                        if empty_pos:
                            for _ri, _role in enumerate(roles):
                                _vkey = voc_index(_role)
                                if _vkey not in tucker_decomp.vocab:
                                    continue
                                _fmat = tucker_decomp.factors[_ri].detach().cpu().numpy()
                                _norms = np.maximum(np.linalg.norm(_fmat, axis=1, keepdims=True), 1e-12)
                                _fmat = _fmat / _norms
                                _vecs = {w: _fmat[idx] for w, idx in tucker_decomp.vocab[_vkey].items()}
                                for _pos in empty_pos:
                                    vecs_by_pos[_pos] = _vecs
                                break
                        simlex_out = evaluate_simlex(simlex_pairs, vecs_by_pos, verbose=verbose)
                        if isinstance(sem_out, dict) and isinstance(simlex_out, dict):
                            flat = {}
                            all_scores = simlex_out.get("ALL", {})
                            if isinstance(all_scores, dict):
                                if all_scores.get("rho") is not None:
                                    flat["simlex_all_rho"] = all_scores["rho"]
                                if all_scores.get("pval") is not None:
                                    flat["simlex_all_pval"] = all_scores["pval"]
                            sem_out = {**sem_out, **flat}
                    except Exception as _simlex_err:
                        print(f"Warning: SimLex evaluation failed ({_simlex_err}); skipping.")

                if _dim_judge is not None:
                    try:
                        # The judge was loaded up-front (see setup) so its measured
                        # footprint is already reflected in every batch estimate; no
                        # mid-run reload is needed here. But cp.asnumpy() above only
                        # frees the CuPy *arrays* — CuPy's memory pool keeps the
                        # underlying blocks cached rather than returning them to the
                        # driver, so PyTorch's allocator can still fail to cudaMalloc
                        # room for the judge's activations even though CuPy itself
                        # has nothing live. Trim the pools right before scoring so the
                        # freed decomposition memory is actually available to torch.
                        if _is_sgd:
                            if torch.cuda.is_available():
                                torch.cuda.empty_cache()
                        elif _sst is not None:
                            _sst.trim_pools()
                        else:
                            cp.get_default_memory_pool().free_all_blocks()
                            cp.get_default_pinned_memory_pool().free_all_blocks()
                        try:
                            _dim_out = {}
                            if dim_consistency_method in ("score", "both"):
                                _dim_out.update(_dim_judge.score(
                                    tucker_decomp, seed=random_state, verbose=False
                                ))
                            if dim_consistency_method in ("similarity", "both"):
                                _dim_out.update(_dim_judge.score_similarity_consistency(
                                    tucker_decomp, seed=random_state, verbose=False
                                ))
                        finally:
                            # score()'s forward passes leave PyTorch's caching
                            # allocator holding "reserved" activation memory that
                            # isn't returned to the driver on its own. The sharded
                            # core update's batch_num is estimated once and then
                            # cached across iterations (see kl_core_update), so if
                            # that reservation lingers it silently shrinks the
                            # headroom the NEXT core update assumed was free,
                            # OOMing CuPy even though the judge is done with the
                            # memory. Give it back right after scoring.
                            if _dim_judge.device is not None and _dim_judge.device.type == "cuda":
                                torch.cuda.empty_cache()
                        if isinstance(sem_out, dict):
                            sem_out = {**sem_out, **_dim_out}
                    except Exception as _dim_err:
                        print(f"Warning: dimension-consistency scoring failed ({_dim_err}); skipping.")

                fitness_scores.append(sem_out)
                # Primary value used for early stopping / diff
                _sem_value_available = True
                if isinstance(sem_out, dict):
                    if sem_primary_key not in sem_out:
                        if sem_primary_key.startswith("simlex_") and simlex_pairs is not None:
                            print(f"Warning: '{sem_primary_key}' not available (all SimLex pairs OOV?); skipping sem check.")
                            _sem_value_available = False
                        elif (
                            sem_primary_key.startswith(("dim_consistency", "similarity_consistency"))
                            and _dim_judge is not None
                        ):
                            # Judge scoring failed this check (warning printed above), or the
                            # configured dim_consistency_method doesn't produce this key;
                            # skip the sem check instead of aborting the run.
                            print(f"Warning: '{sem_primary_key}' not available (judge scoring failed, "
                                  f"or dim_consistency_method={dim_consistency_method!r} doesn't produce it?); "
                                  f"skipping sem check.")
                            _sem_value_available = False
                        elif sem_primary_key.startswith(("dim_consistency", "similarity_consistency")):
                            raise KeyError(
                                f"Primary semantic key '{sem_primary_key}' requires dimension-consistency "
                                f"judge scoring; enable it with --dim-consistency true and "
                                f"--dim-consistency-method matching this key (score/similarity/both)."
                            )
                        else:
                            raise KeyError(f"Primary semantic key '{sem_primary_key}' missing from returned scores.")

                if _sem_value_available:
                    if isinstance(sem_out, dict):
                        sem_value = float(sem_out[sem_primary_key])
                        sem_all_dump = json.dumps(sem_out)
                    else:
                        sem_value = float(sem_out)
                        sem_all_dump = str(sem_out)

                    _rec_err_log = rec_errors[-1] if rec_errors else None
                    print(
                        f"Iteration {iteration + 1}\t"
                        f"Rec_error: {_rec_err_log}\t"
                        f"Sem({sem_primary_key}): {sem_value}\t"
                        f"Sem_all: {sem_all_dump}"
                    )

                if not _is_sgd:
                    tl.set_backend("cupy")

                if _sem_value_available:
                    # track best semantic model (based on primary key)
                    diff = sem_value - float(best_sem_score)
                    if diff > 0:
                        best_sem_score = sem_value
                        if _is_sgd:
                            # already-materialized host snapshots from this check
                            best_core = _sem_core_np
                            best_factors = list(_sem_factors_np)
                        else:
                            best_core = [C.copy() for C in core] if _is_tt else core.copy()
                            best_factors = [factor.copy() for factor in factors]
                        best_sem_iteration = iteration
                        if verbose:
                            print("New best semantic score; saving current best core and factors.")
                        if save_intermediate:
                            # Save host arrays (like the checkpoint path below):
                            # pickled CuPy arrays can only be loaded where CuPy +
                            # a GPU are available, and this file is what
                            # judge_eval/inspect_tucker later load on CPU.
                            if _is_sgd:
                                # tensorly validates the container against the
                                # ACTIVE backend; on the SGD path that is
                                # "pytorch", whose ndim() rejects numpy — wrap
                                # the host snapshots in CPU torch tensors
                                # (zero-copy, still loads fine on CPU boxes).
                                temp_tensor = TensorModel(
                                    (torch.as_tensor(best_core),
                                     [torch.as_tensor(f) for f in best_factors])
                                )
                            else:
                                temp_tensor = TensorModel(
                                    (_as_host(best_core),
                                     [_as_host(factor) for factor in best_factors])
                                )
                            torch.save(temp_tensor, paths["model"])
                            print("saving temp model to", paths["model"])

                            np.save(paths["errors"], np.array([_as_host(e) for e in rec_errors]))

                            # Save semantic scores more robustly
                            if isinstance(sem_out, dict):
                                # save as JSON alongside the provided fitness path
                                with open(paths["fitness_json"], "w") as f:
                                    json.dump(fitness_scores, f, indent=2)
                            else:
                                np.save(paths["fitness"], np.array(fitness_scores, dtype=float))

                    # semantic patience (uses primary key only)
                    if diff < tol:
                        sem_no_rec_improve_steps += 1
                        if verbose:
                            print(f"\tNo semantic improvement: {sem_no_rec_improve_steps}/{patience} (Δ={diff:.3e})")
                        if sem_no_rec_improve_steps >= patience:
                            if verbose:
                                notify_discord(
                                    f"Stopped after {sem_no_rec_improve_steps} non-improving semantic steps "
                                    f"(patience={patience}). Converged at iteration {iteration}.",
                                    job_finished=False,
                                )
                            break
                    else:
                        if sem_no_rec_improve_steps:
                            print(f"\tSemantic improvement (Δ={diff:.3e}); resetting patience counter.")
                        sem_no_rec_improve_steps = 0

            if checkpoint_saving: # only trigger if this is not 0 -> True
                if (iteration + 1) % cfg.train.checkpoint_saving_steps == 0:
                    print(f"saving model at iteration {iteration}")
                    if _is_sgd:
                        # Dict payload: (core, factors) views for tooling that
                        # peeks, plus raw params + optimizer state so resume
                        # continues the exact trajectory (batches replay from
                        # the step counter; see SGDTrainer).
                        checkpoint_tensor = _sgd_trainer.checkpoint_payload(iteration + 1)
                    else:
                        # _as_host also handles the TT family's list-valued core.
                        checkpoint_tensor = TensorModel((_as_host(core), [cp.asnumpy(factor) for factor in factors]))
                    paths = cfg.artifact_paths()
                    torch.save(checkpoint_tensor, paths["checkpoint_dir"] / f"{iteration + 1}.pt")
                    # Durably mirror this checkpoint to GPFS so a walltime kill stays resumable.
                    mirror_checkpoint(paths, mirror_paths, f"{iteration + 1}.pt")

                    # we collect reconstruction and fitness scores if they exist and dump
                    if fitness_scores:
                        last_sem = fitness_scores[-1]
                        if isinstance(last_sem, dict):
                            fitness_primary = last_sem.get(sem_primary_key, None)
                            fitness_dump = json.dumps(last_sem)
                        else:
                            fitness_primary = last_sem
                            fitness_dump = str(last_sem)
                    else:
                        fitness_primary = None
                        fitness_dump = None

                    rec_error = rec_errors[-1] if rec_errors else None

                    with open(paths["checkpoint_dir"] / "log.txt", "a") as f:
                        f.write(
                            f"Iteration {iteration + 1}\t"
                            f"Rec_error: {rec_error}\t"
                            f"Sem({sem_primary_key}): {fitness_primary}\t"
                            f"Sem_all: {fitness_dump}\n"
                        )

            # ---- proactive GPU pool trim (low cadence; see trim_pools) ----
            # Reclaims transient eval/copy blocks so out-of-pool cuBLAS/cuSPARSE
            # workspaces keep their headroom. Deliberately NOT per iteration.
            if pool_trim_every and (iteration + 1) % pool_trim_every == 0:
                if _is_sgd:
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                elif _sst is not None:
                    _sst.trim_pools()
                else:
                    cp.get_default_memory_pool().free_all_blocks()
                    cp.get_default_pinned_memory_pool().free_all_blocks()

            # ---- graceful stop: save a resumable checkpoint, then break ----
            # Triggered by the first Ctrl+C. We write the same artifacts the normal
            # checkpoint path does (a {iteration+1}.pt model plus errors/fitness),
            # so get_resume_state() can pick this run up exactly as a periodic
            # checkpoint. A second Ctrl+C during the save raises KeyboardInterrupt
            # (default handler restored above), exiting immediately as before.
            if _interrupt_requested["flag"]:
                print(f"Saving resumable checkpoint at iteration {iteration} before stopping...")
                try:
                    os.makedirs(paths["checkpoint_dir"], exist_ok=True)
                    if _is_sgd:
                        checkpoint_tensor = _sgd_trainer.checkpoint_payload(iteration + 1)
                    else:
                        checkpoint_tensor = TensorModel(
                            (_as_host(core), [cp.asnumpy(factor) for factor in factors])
                        )
                    ckpt_path = paths["checkpoint_dir"] / f"{iteration + 1}.pt"
                    torch.save(checkpoint_tensor, ckpt_path)
                    np.save(paths["errors"], np.array([_as_host(e) for e in rec_errors]))
                    if fitness_scores:
                        if isinstance(fitness_scores[-1], dict):
                            with open(paths["fitness_json"], "w") as f:
                                json.dump(fitness_scores, f, indent=2)
                        else:
                            np.save(paths["fitness"], np.array(fitness_scores, dtype=float))
                    # Mirror to GPFS so the interrupt-saved checkpoint survives in HPC mode.
                    mirror_checkpoint(paths, mirror_paths, f"{iteration + 1}.pt")
                    print(f"Resumable checkpoint saved to {ckpt_path}")
                except KeyboardInterrupt:
                    # Second Ctrl+C arrived mid-save: abort immediately.
                    raise
                except Exception as _save_err:
                    print(f"Failed to save resumable checkpoint: {_save_err}")
                break

        decomp_seconds = time.time() - _decomp_loop_start
        # solve_seconds counts only the iteration updates/error kernels;
        # decomp_seconds additionally covers in-loop semantic evaluation and
        # checkpointing.
        solve_seconds = float(sum(_iter_seconds))
        print(
            f"decomposition time: {solve_seconds:.2f}s "
            f"(sum of {len(_iter_seconds)} iteration time(s), device-synced at log steps"
            + (f", mean {solve_seconds / len(_iter_seconds):.2f}s" if _iter_seconds else "")
            + f"); {decomp_seconds:.2f}s for the whole loop including in-loop evaluation"
        )

        # With --sgd-eval-sample the logged curve is a subsampled estimate; pay
        # for one exact pass at the end so the reported final_error is exact.
        # (Like rec_errors[-1], it describes the trainer's current state, which
        # is not necessarily the best-semantics model returned below.)
        _sgd_exact_final_error = None
        if _is_sgd and getattr(_sgd_trainer, "eval_sample", None):
            _sgd_exact_final_error = _sgd_trainer.final_relative_error()
            print(f"Exact final reconstruction error: {_sgd_exact_final_error:.6f} "
                  f"(logged curve used --sgd-eval-sample "
                  f"{_sgd_trainer.eval_sample})")

        # Restore whatever SIGINT handler was in place before this run.
        if _sigint_installed:
            signal.signal(signal.SIGINT, _original_sigint)

        if best_sem_iteration is not None:
            if _is_sgd:
                # numpy snapshots + pytorch backend: see the temp-save note.
                tensor = TensorModel(
                    (torch.as_tensor(best_core),
                     [torch.as_tensor(f) for f in best_factors])
                )
            else:
                tensor = TensorModel((best_core, best_factors))
            iteration = best_sem_iteration
        elif _is_sgd:
            # `core`/`factors` are never bound on the SGD path.
            _fin_core, _fin_factors = _sgd_trainer.materialize()
            tensor = TensorModel(
                (torch.as_tensor(_fin_core),
                 [torch.as_tensor(f) for f in _fin_factors])
            )
        else:
            tensor = TensorModel((core, factors))
        if return_errors == "simple":
            return tensor, rec_errors
        elif return_errors == "full":
            return {
                "tensor": tensor,
                "errors": rec_errors,
                "fitness_scores": fitness_scores,
                "sem_primary_key": sem_primary_key,
                "iterations": iteration + 1,
                "final_error": (
                    _sgd_exact_final_error if _sgd_exact_final_error is not None
                    else (rec_errors[-1] if len(rec_errors) > 0 else None)
                ),
                "decomp_seconds": decomp_seconds,
                "solve_seconds": solve_seconds,
                "iter_seconds": list(_iter_seconds),
            }
        else:
            return tensor

