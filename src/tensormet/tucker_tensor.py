from __future__ import annotations
import os
import pickle

import sparse
import torch
import json
import math
import numpy as np
import matplotlib.pyplot as plt
import tensorly as tl
from tensorly.tucker_tensor import validate_tucker_rank, tucker_normalize, TuckerTensor
from tensorly.tenalg import mode_dot
from typing import List, Optional, Union, Tuple,  Literal
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
                            resolve_checkpoint_path,
                   )
from tensormet.sparse_ops import initialize_nonnegative_tucker
from tensormet.similarity import evaluate_sample, get_eval_num_threads, load_simlex, evaluate_simlex
from tensormet.routing import get_update_routing_step, get_log_step, UpdateRouting
from tensormet.stochastic_sparse import subsample_coo, make_iteration_rng
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
# Maps tensor role names to SimLex-999 POS tags (first match per POS wins)
_SIMLEX_POS_MAP = {
    "root": "V", "verb": "V",
    "nsubj": "N", "obj": "N", "subject": "N", "object": "N",
}
_SIMLEX_PATH = DATA_DIR / "corpora" / "SimLex-999.txt"


def _to_np(x):
    # Accept NumPy arrays or torch tensors; return NumPy view/copy
    if hasattr(x, "detach"):  # torch.Tensor
        return x.detach().cpu().numpy()
    return x

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



def np_sim(a: np.ndarray, b: np.ndarray) -> float:
    """Computes cosine similarity between two numpy vectors."""
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


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
        # If roles aren't provided explicitly, parse them from the vocab keys
        self.roles = extract_roles_from_vocab(self.vocab)
        self.decomp_path = None

    def get_role_index(self, role: str) -> int:
        """Helper method to wrap the module-level _role_index using instance roles."""
        return _role_index(role, self.roles)
    
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
        if method not in {"counting", "sc", "sii",
                          "scSoftPlus", "scShifted", "siiSoftPlus", "siiShifted"}:
            raise ValueError("method must be one of {'counting','sc','sii'}")
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
        _ds = dim_spec_str(dims)
        vocab_path_new = os.path.join(base, f"vocabularies/{order}D_{_ds}d{suffix}.pkl")
        vocab_path_old = os.path.join(base, f"vocabularies/{_ds}{suffix}.pkl")

        if os.path.exists(vocab_path_new):
            vocab_path = vocab_path_new
        elif os.path.exists(vocab_path_old):
            vocab_path = vocab_path_old
        else:
            raise FileNotFoundError(f"Missing vocab file. Checked {vocab_path_new} and {vocab_path_old}")



        decomp_path = os.path.join(base, "decomposition")
        # Construct candidate prefixes: new naming first, legacy fallback.
        # New format: {name}{div}_{method}_{order}D_{dims}d{sf_suffix}_{rank}r_
        # Legacy format (3D only): {name}{div}_{method}_{dims}d_{rank}r_
        name_prefix = f"{name + '_' if name else ''}"
        ss_suffix = f"_{str(subsample_frac).replace('.', 'p')}ss" if subsample_frac != 1.0 else ""

        new_file_prefix      = f"{name_prefix}{divergence}_{method}_{order}D_{_ds}d{suffix}_{rank}r{ss_suffix}_"
        new_file_prefix_no_sf = f"{name_prefix}{divergence}_{method}_{order}D_{_ds}d_{rank}r{ss_suffix}_"
        legacy_file_prefix   = f"{name_prefix}{divergence}_{method}_{_ds}d_{rank}r_"

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
        # if there is a "runs.jsonl" file in the decomposition folder, we print the content relevant to the loaded tensor
        runs_path = os.path.join(decomp_path, "runs.jsonl")
        if os.path.exists(runs_path):
            with open(runs_path, "r") as f:
                for line in f:
                    run_info = json.loads(line)
                    if "output_tensor" in run_info.keys():
                        if run_info["output_tensor"] == tensor_name:
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
        self.core = np_dispatch(tensor.core)
        self.factors = np_dispatch(tensor.factors)
        self.decomp_path = resolved


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

    def fetch_latents(self, triple: Tuple[str, ...]) -> Tuple[np.ndarray, ...]:
        """Fetches the latent representations for a given tuple of elements."""
        # Map fetch_single_latent across all elements and their corresponding roles
        return tuple(
            self.fetch_single_latent(triple[i], self.roles[i])
            for i in range(len(self.roles))
        )

    def fetch_single_latent(self, element, role) -> np.ndarray:
        """Fetches the latent representation for an element."""
        el_idx = self.vocab[voc_index(role)][element]
        factor_slice = self.factors[self.get_role_index(role)][el_idx]
        return _to_np(factor_slice)



    # -- Sparsity methods ---
    def to_cupy(self):
        """
        Prepare for inference by moving to gpu
        """
        if isinstance(self.core, torch.Tensor):
            self.core = self.core.numpy()
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
                                    role: str,
                                    top_k: int = 10):
        latent = self.fetch_single_latent(word, role)
        latent = torch.tensor(latent)
        scores, dims = torch.topk(latent, top_k)
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
                                  role,
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

            flat = row_np + col_np * block_size
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
) -> Tuple[cpx_sparse.coo_matrix, Tuple[int, ...]]:
    """
    Convert a torch sparse COO tensor to a CuPy COO sparse matrix.

    For 2D tensors, the mapping is straightforward.
    For N-D tensors (N>2), we flatten the N-D indices to a single row index
    using np.ravel_multi_index and store them in a (prod(shape), 1) matrix.

    The original shape is returned for reconstruction.
    Returns:
        (cupy_coo_matrix, original_shape)
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
    def __init__(self, tensor, device="cpu", sparsity_type=None, shared_factors=None):
        self.tensor = tensor
        self.sparsity_type = sparsity_type
        self.shape = tensor.shape
        self.device = device
        self.shared_factors = shared_factors

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
            shared_factors: Optional[Union[Tuple[Tuple[int, int], ...], str]] = None,
    ) -> "SparseTupleTensor":
        """
        Load a populated sparse tensor from disk.

        Expects population artifacts saved as:
            tensors/{dataset}/populated/{method}_{dims}{suffix}.pt
        where suffix matches the shared-factor naming convention.
        """
        if method not in {
            "counting", "sc", "sii",
            "siiSoftPlus", "siiShifted",
            "scSoftPlus", "scShifted",
        }:
            raise ValueError(
                "method must be one of "
                "{'counting','sc','sii','siiSoftPlus','siiShifted','scSoftPlus','scShifted'}"
            )

        base = os.path.join(DATA_DIR, "tensors", dataset)
        base = readonly_dispatch(base, tier1)

        if shared_factors == "all":
            shared_factors = tuple(sorted((i, j) for i in range(order) for j in range(i + 1, order)))

        linked_nontrivial = nontrivial_linked_groups(shared_factors, num_factors=order)
        suffix = shared_factor_suffix(linked_nontrivial)
        _ds = dim_spec_str(dims)
        populated_path = os.path.join(base, "populated", f"{method}_{order}D_{_ds}d{suffix}.pt")

        if not os.path.exists(populated_path):
            if order == 3: # legacy naming support
                populated_path = os.path.join(base, "populated", f"{method}_{_ds}{suffix}.pt")
            else:
                raise FileNotFoundError(f"Missing populated tensor file: {populated_path}")

        tensor = torch_or_pickle_load(populated_path, map_location=map_location)

        return cls(
            tensor,
            device=map_location,
            sparsity_type="torch",
            shared_factors=shared_factors,
        )



    # -- Sparsity methods ---
    def sparse_representation(self, sparse_type):
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
            sparse_tensor = sparse.COO(coords, data, shape=shape)
            return sparse_tensor

        elif sparse_type == "cupy":
            if not (isinstance(self.tensor, (torch.Tensor, SparseCOOTensor)) and self.tensor.is_sparse):
                raise TypeError("cupy expects self.tensor to be a torch sparse tensor.")
            tensor_cupy, shape = torch_sparse_to_cupy(self.tensor)
            return tensor_cupy
        else:
            raise NotImplementedError(f"Sparse representation for type {sparse_type} not implemented.")



    def tensor_to_sparse(self, sparse_type="tensorflow"):
        self.tensor = self.sparse_representation(sparse_type)
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
            init = cfg.train.init
            tol = cfg.train.tol
            epsilon = cfg.train.epsilon
            verbose = cfg.train.verbose
            return_errors = cfg.train.return_errors
            normalize_factors = cfg.train.normalize_factors
            patience = cfg.train.patience
            warmup_steps = cfg.train.warmup_steps
            largedim = cfg.train.largedim
            checkpoint_saving = cfg.train.checkpoint_saving_steps

            rec_check_every = cfg.eval.rec_check_every
            sem_check_every = cfg.eval.sem_check_every
            sem_error_type = cfg.eval.sem_error_type
            # logging
            rec_log_every = cfg.eval.rec_log_every
            rec_log_every = rec_log_every or rec_check_every
            time_iteration = cfg.eval.time_iteration
            # saving
            save_intermediate = cfg.eval.save_intermediate
            tier1 = cfg.exp.tier1


        except Exception as e:
            raise ValueError(f"Check config structure: {e}")

        if not isinstance(self, SparseTupleTensor):
            raise TypeError("sparse_tensor must be a SparseTupleTensor instance.")
        if not self.sparsity_type == "cupy":
            raise ValueError("sparse_tensor must have sparsity_type 'cupy'.")

        paths = cfg.artifact_paths()

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
        rank = validate_tucker_rank(shape, rank=rank)
        modes = list(range(len(rank)))
        _max_dim = max(shape)
        if checkpoint_tensor is not None:
            if isinstance(checkpoint_tensor, tuple):
                # if TensorLy TuckerTensor
                ckpt_core, ckpt_factors = checkpoint_tensor
            else:
                # if our TuckerDecomposition class
                ckpt_core, ckpt_factors = checkpoint_tensor.core, checkpoint_tensor.factors

            core = cp.asarray(ckpt_core)
            factors = [cp.asarray(factor) for factor in ckpt_factors]
        else:
            core, factors = initialize_nonnegative_tucker(self.tensor, shape, rank, modes, init,
                                                           random_state, thread_budget=thread_budget)

        # --- multi-GPU shard initialisation ---

        _n_gpus = getattr(cfg.train, "n_gpus", 1)
        _subsample_frac = getattr(cfg.train, "subsample_frac", 1.0)
        _subsample_warmup = getattr(cfg.train, "subsample_warmup", 0)

        if _n_gpus > 1:
            _sst = ShardedSparseTensor.from_coo(
                self.tensor, shape, device_ids=list(range(_n_gpus)),
                subsample_frac=_subsample_frac,
            )
        else:
            _sst = None

        # --- stochastic subsampling RNG (single-GPU path) ---
        _iter_rng = make_iteration_rng(cfg.exp.random_state) if _subsample_frac < 1.0 else None

        linked_factors = defaultdict(set)
        if self.shared_factors:
            for a, b in self.shared_factors:
                linked_factors[a].add(b)
                linked_factors[b].add(a)

        no_rec_improve_steps = 0
        # If we resumed, grab the last known error to calculate early stopping diff accurately
        last_err = rec_errors[-1] if rec_errors else None

        sem_no_rec_improve_steps = 0

        # Ensure 'best' variables are initialized safely so returning them at the end doesn't fail
        best_core = core.copy()
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

        print(divergence, rank, _subsample_frac)
        est_iter_time = self.estimate_training_time(rank=rank[0], subsample=_subsample_frac)
        print(f"estimated training time: {est_iter_time}*{n_iter_max}={est_iter_time*n_iter_max}")
        for iteration in range(start_iteration, n_iter_max):
            if time_iteration:
                start_time = time.time()
            log_step = get_log_step(iteration, rec_log_every, rec_check_every)
            routing = get_update_routing_step(divergence=divergence, dim=dim, log_step=log_step, largedim=largedim)
            # --- multi-GPU routing override (largedim variants only) ---
            if _sst is not None:
                if divergence == "kl" and (_max_dim >= 4000 or largedim):
                    routing = UpdateRouting(
                        factor_update=make_sharded_kl_factor_update(_sst),
                        core_update=make_sharded_kl_core_update(_sst),
                        error_fn=make_sharded_kl_compute_errors(_sst),
                        core_returns_error=routing.core_returns_error,
                    )
                elif divergence == "fr" and (_max_dim > 4000 or largedim):
                    routing = UpdateRouting(
                        factor_update=make_sharded_fr_factor_update(_sst),
                        core_update=make_sharded_fr_core_update(_sst),
                        error_fn=make_sharded_fr_compute_errors(_sst),
                        core_returns_error=routing.core_returns_error,
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
                subsample_coo(self.tensor, shape, _subsample_frac, _iter_rng)
                if _use_subsample else self.tensor
            )
            # --- factors ---
            for mode in modes:
                factors[mode] = routing.factor_update(
                    vec_tensor=_current_tensor,
                    core=core,
                    factors=factors,
                    mode=mode,
                    shape=shape,
                    thread_budget=thread_budget,
                    epsilon=epsilon,
                    verbose=verbose
                )

                # new: factor linking
                if mode in linked_factors:
                    for other in linked_factors[mode]:
                        factors[other] = factors[mode]

            # --- core + error ---
            if routing.core_returns_error:
                # FR: combined core update + error in one call
                core, rel_err = routing.core_update(
                    vec_tensor=_current_tensor,
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
                core = routing.core_update(
                    vec_tensor=_current_tensor,
                    shape=shape,
                    core=core,
                    factors=factors,
                    modes=modes,
                    thread_budget=thread_budget,
                    epsilon=epsilon,
                    verbose=verbose
                )
                rel_err = routing.error_fn(
                    vec_tensor=_current_tensor,
                    shape=shape,
                    core=core,
                    factors=factors,
                    thread_budget=thread_budget,
                    epsilon=epsilon,
                    verbose=verbose
                )
            # Normalize if desired
            if normalize_factors:
                core, factors = tucker_normalize((core, factors))

            if log_step:
                rec_errors.append(rel_err)

                # ---- reconstruction + patience ----
                has_prev_err = len(rec_errors) >= 2
                if has_prev_err:
                    delta = rec_errors[-2] - rec_errors[-1]


                    message = f"{iteration}: reconstruction error={rec_errors[-1]} (Δ={delta:+.3e})"
                    if time_iteration:
                        end_time = time.time()
                        message += f", time={end_time - start_time}"
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
                tl.set_backend("pytorch")
                core_cpu = tl.tensor(cp.asnumpy(core))
                factors_cpu = [tl.tensor(cp.asnumpy(f)) for f in factors]
                roles = extract_roles_from_vocab(vocab)
                tucker_decomp = TuckerDecomposition(core=core_cpu, factors=factors_cpu, vocab=vocab, roles=roles)

                sem_out = evaluate_sample(
                    tucker_decomp,
                    sample_sentences,
                    sampled=True,
                    seed=random_state,
                    thread_budget=thread_budget,
                    return_type=sem_error_type,
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

                fitness_scores.append(sem_out)
                # Primary value used for early stopping / diff
                if isinstance(sem_out, dict):
                    if sem_primary_key not in sem_out:
                        if sem_primary_key.startswith("simlex_") and simlex_pairs is not None:
                            print(f"Warning: '{sem_primary_key}' not available (all SimLex pairs OOV?); skipping sem check.")
                            continue
                        raise KeyError(f"Primary semantic key '{sem_primary_key}' missing from returned scores.")
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

                tl.set_backend("cupy")

                # track best semantic model (based on primary key)
                diff = sem_value - float(best_sem_score)
                if diff > 0:
                    best_sem_score = sem_value
                    best_core = core.copy()
                    best_factors = [factor.copy() for factor in factors]
                    best_sem_iteration = iteration
                    if verbose:
                        print("New best semantic score; saving current best core and factors.")
                    if save_intermediate:

                        temp_tensor = TuckerTensor((best_core, best_factors))
                        torch.save(temp_tensor, paths["model"])
                        print("saving temp model to", paths["model"])

                        np.save(paths["errors"], np.array([cp.asnumpy(e) for e in rec_errors]))

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
                    checkpoint_tensor = TuckerTensor((cp.asnumpy(core), [cp.asnumpy(factor) for factor in factors]))
                    paths = cfg.artifact_paths()
                    torch.save(checkpoint_tensor, paths["checkpoint_dir"] / f"{iteration + 1}.pt")

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


        if best_sem_iteration is not None:
            tensor = TuckerTensor((best_core, best_factors))
            iteration = best_sem_iteration
        else:
            tensor = TuckerTensor((core, factors))
        if return_errors == "simple":
            return tensor, rec_errors
        elif return_errors == "full":
            return {
                "tensor": tensor,
                "errors": rec_errors,
                "fitness_scores": fitness_scores,
                "sem_primary_key": sem_primary_key,
                "iterations": iteration + 1,
                "final_error": rec_errors[-1] if len(rec_errors) > 0 else None,
            }
        else:
            return tensor

