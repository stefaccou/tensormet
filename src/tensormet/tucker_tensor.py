import os
import pickle
import sparse
import torch
import json
import cupy as cp
import numpy as np
import matplotlib.pyplot as plt
import cupyx.scipy.sparse as cpx_sparse
import tensorly as tl
from tensorly.tucker_tensor import validate_tucker_rank, tucker_normalize, TuckerTensor
from tensorly.tenalg import mode_dot
from typing import List, Optional, Union, Tuple, Dict, Literal
from collections import defaultdict
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager

from tensormet.config import RunConfig
from tensormet.utils import (DATA_DIR,
                            torch_or_pickle_load,
                            readonly_dispatch,
                            tree_to_device,
                            notify_discord,
                            ThreadBudget,
                   )
from tensormet.sparse_ops import initialize_nonnegative_tucker
from tensormet.similarity import evaluate_sample, get_eval_num_threads
from tensormet.routing import get_update_routing_step, get_log_step

import time

def _to_np(x):
    # Accept NumPy arrays or torch tensors; return NumPy view/copy
    if hasattr(x, "detach"):  # torch.Tensor
        return x.detach().cpu().numpy()
    return x

def _role_index(role: str) -> int:
    if role == "verb":
        return 0
    elif role == "subject":
        return 1
    elif role == "object":
        return 2
    else:
        raise ValueError("role must be one of {'verb','subject','object'}")

def _voc_index(role: str) -> str:
    if role == "verb":
        return "v2i"
    elif role == "subject":
        return "s2i"
    elif role == "object":
        return "o2i"
    else:
        raise ValueError("role must be one of {'verb','subject','object'}")

def np_sim(a: np.ndarray, b: np.ndarray) -> float:
    """Computes cosine similarity between two numpy vectors."""
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


class TuckerDecomposition:
    """Encapsulating the tucker decomposition (core and factors) and the vocabulary,
    providing methods for scoring, slicing, visualisation, etc."""
    def __init__(self, core, factors: List[torch.Tensor], vocab: dict):
        self.core = core
        self.factors = factors
        self.vocab = vocab

    # --- Construction and loading ---
    @classmethod
    def load_from_disk(cls,
                       dataset: str="karrewiet_vectors_ids",
                       method: str="counting",
                       divergence: str="fr",
                       dims: int=750,
                       rank: int=100,
                       iterations: int=1000,
                       map_location: str="cpu",
                       name: Optional[str]=None,
                       tier1: bool=False,
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

        vocab_path = os.path.join(base, f"vocabularies/{dims}.pkl")
        tensor_name = f"{name+'_' if name else ''}{divergence}_{method}_{dims}d_{rank}r_{iterations}i.pt"
        decomp_path = os.path.join(base,"decomposition", tensor_name)
        if not os.path.exists(vocab_path):
            raise FileNotFoundError(f"Missing vocab file: {vocab_path}")
        if not os.path.exists(decomp_path):
            raise FileNotFoundError(f"Missing decomposition file: {decomp_path}")
        # the vocab is here under f"vocabularies_[dims].pkl"
        # Load with torch (they were saved with torch.save)
        with open(vocab_path, "rb") as f:
            vocab = pickle.load(f)
        (core, factors) = torch_or_pickle_load(decomp_path, map_location=map_location)

        # if there is a "runs.jsonl" file in the decomposition folder, we print the content relevant to the loaded tensor
        runs_path = os.path.join(base, "decomposition", "runs.jsonl")
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

        return cls(core, factors, vocab)




    def check_vocab(self, triple: Tuple[str, str, str]) -> bool:
        """Checks if the given (verb, subject, object) triple is in the vocabulary."""
        v_in = triple[0] in self.vocab["v2i"]
        s_in = triple[1] in self.vocab["s2i"]
        o_in = triple[2] in self.vocab["o2i"]
        return v_in and s_in and o_in


    def fetch_latents(self, triple: Tuple[str, str, str]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Fetches the latent representations for a given (verb, subject, object) triple."""
        v_idx = self.vocab["v2i"][triple[0]]
        s_idx = self.vocab["s2i"][triple[1]]
        o_idx = self.vocab["o2i"][triple[2]]
        V, S, O = [ _to_np(F) for F in self.factors]     # shapes (DIMS,R)
        v = V[v_idx]                                     # (R,)
        s = S[s_idx]                                     # (R,)
        o = O[o_idx]                                     # (R,)
        return v, s, o

    def fetch_single_latent(self, element, role) -> np.ndarray:
        """Fetches the latent representation for an element."""
        el_idx = self.vocab[_voc_index(role)][element]
        factor_slice = self.factors[_role_index(role)][el_idx]
        return _to_np(factor_slice)

    def _core_np(self):
        return _to_np(self.core)

    # -- Sparsity methods ---
    def sparse_representation(self):
        import tensorflow as tf
        # we return the sparse representation of the tensor
        # we check if our tensor is a tensorflow tensor or make it one
        if not isinstance(self.core, tf.Tensor):
            core = tf.convert_to_tensor(self.core)
        else:
            core = self.core
        sparse_core = tf.sparse.from_dense(core)
        # we do the same for the factors
        sparse_factors = []
        for factor in self.factors:
            if not isinstance(factor, tf.Tensor):
                factor = tf.convert_to_tensor(factor)
            sparse_factor = tf.sparse.from_dense(factor)
            sparse_factors.append(sparse_factor)

        return sparse_core, sparse_factors



    def tensor_to_sparse(self):
        self.core, self.factors = self.sparse_representation()

    def tensor_to_dense(self):
        import tensorflow as tf
        # If we have TensorFlow sparse tensors
        if isinstance(self.core, tf.SparseTensor):
            self.core = tf.sparse.to_dense(self.core).numpy()
            self.factors = [
                tf.sparse.to_dense(f).numpy() if isinstance(f, tf.SparseTensor) else _to_np(f)
                for f in self.factors
            ]
        else:
            # If they’re already torch/np dense, just ensure NumPy
            self.core = _to_np(self.core)
            self.factors = [_to_np(f) for f in self.factors]

    # -- Scoring and slicing methods ---

    def score_scalar(self, triple: Tuple[str, str, str]) -> float:
        """(1) Scalar reconstruction score ⟨G, a∘b∘c⟩."""
        G = self._core_np()                                 # (R,R,R)
        v, s, o = self.fetch_latents(triple)
        return np.einsum('pqr,p,q,r->', G, v, s, o)

    def contribution_tensor(self, triple: Tuple[str, str, str]) -> np.ndarray:
        """(2) Contribution tensor: G * (a∘b∘c) ∈ R^{R×R×R}."""
        G = self._core_np()                                 # (R,R,R)
        v, s, o = self.fetch_latents(triple)
        # same as doing np.einsum(p, q, r ->pqr) and then multiplying by G
        return np.einsum('p,q,r,pqr->pqr', v, s, o, G)

    def outer_product_latent(self, triple: Tuple[str, str, str]) -> np.ndarray:
        """(3) Pseudo-inverse / HOSVD case: a∘b∘c (rank-1 core-space tensor)."""
        v, s, o = self.fetch_latents(triple)
        return np.einsum('p,q,r->pqr', v, s, o)

    def excluded_role_vector(self, triple: Tuple[str, str, str], role: str) -> np.ndarray:
        """
        Fetches the latent vector for a given excluded role in the triple.
        Can be understood as a "prediction":
            Given the two other attested elements,what are the activations in the third element's dimensions?
        """
        v, s, o = self.fetch_latents(triple)
        if role == "verb":
            return np.einsum('pqr,q,r->p', self._core_np(), s, o)
        elif role == "subject":
            return np.einsum('pqr,p,r->q', self._core_np(), v, o)
        elif role == "object":
            return np.einsum('pqr,p,q->r', self._core_np(), v, s)
        else:
            raise ValueError("role must be one of {'verb','subject','object'}")

    def included_role_vector(self, triple: Tuple[str, str, str], role: str) -> np.ndarray:
        """
        Fetches the latent vector for a given included role in the triple.
        Can be understood as quantifying "contribution":
            How important are the dimensions of X in the final contextualised representation of XYZ?
        """
        v, s, o = self.fetch_latents(triple)
        if role == "verb":
            return np.einsum('pqr,p,q,r->p', self._core_np(), v, s, o)
        elif role == "subject":
            return np.einsum('pqr,p,q,r->q', self._core_np(), v, s, o)
        elif role == "object":
            return np.einsum('pqr,p,q,r->r', self._core_np(), v, s, o)
        else:
            raise ValueError("role must be one of {'verb','subject','object'}")

    def predicted_role_vector(self, triple: Tuple[str, str, str], role: str) -> np.ndarray:
        """
        Fetches the latent vector for a given excluded role in the triple, WITHOUT instantiating the element (OOV).
        Can be understood as a "prediction":
            Given the two other attested elements,what should be the activations in the third element's dimensions?
        """
        latents = {"verb":None, "subject":None, "object":None}
        for i, element in enumerate(latents.keys()):
            if not element == role:
                latents[element] = self.fetch_single_latent(triple[i], element)
        v = latents["verb"]
        s = latents["subject"]
        o = latents["object"]

        if role == "verb":
            return np.einsum('pqr,q,r->p', self._core_np(), s, o)
        elif role == "subject":
            return np.einsum('pqr,p,r->q', self._core_np(), v, o)

        elif role == "object":
            return np.einsum('pqr,p,q->r', self._core_np(), v, s)
        else:
            raise ValueError("role must be one of {'verb','subject','object'}")

    # Slicing

    def get_role_slice(self, role: str, normalize: bool=False) -> np.ndarray:
        G = self._core_np()
        # if role == "verb":
        #     slc = np.einsum('ip, pqr -> iqr', _to_np(self.factors[0]), G)
        # elif role == "subject":
        #     slc = np.einsum('jp, pqr -> ipr', _to_np(self.factors[1]), G)
        # elif role == "object":
        #     slc = np.einsum('kp, pqr -> ipq', _to_np(self.factors[2]), G)
        if role == "verb":
            # (num_verbs, R) × (R, R, R) -> (num_verbs, R, R)
            slc = np.einsum('ip,pqr->i q r', _to_np(self.factors[0]), G)
        elif role == "subject":
            # (num_subj, R) × (R, R, R) -> (num_subj, R, R)
            slc = np.einsum('jp,pqr->j p r', _to_np(self.factors[1]), G)
        elif role == "object":
            # (num_obj, R) × (R, R, R) -> (num_obj, R, R)
            slc = np.einsum('kp,pqr->k p q', _to_np(self.factors[2]), G)

        else:
            raise ValueError("role must be one of {'verb','subject','object'}")
        if normalize:
            slc = slc / np.linalg.norm(slc, axis=-1, keepdims=True)
        return slc

    def role_slice_from_tuple(self, triple: Tuple[str, str, str], role: str) -> np.ndarray:
        G = self._core_np()
        v, s, o = self.fetch_latents(triple)
        if role == "verb":
            slc = np.einsum('pqr,q,r->qr', G, s, o)
        elif role == "subject":
            slc = np.einsum('pqr,p,r->pr', G, v, o)
        elif role == "object":
            slc = np.einsum('pqr,p,q->pq', G, v, s)
        else:
            raise ValueError("role must be one of {'verb','subject','object'}")
        return slc

    def get_weighted_role_slice_from_tuple(self, triple: Tuple[str, str, str], role: str) -> np.ndarray:
        G = self._core_np()
        v, s, o = self.fetch_latents(triple)
        if role == "verb":
            slc = np.einsum('pqr,p,q,r->qr', G, v, s, o)
        elif role == "subject":
            slc = np.einsum('pqr,p,q,r->pr', G, v, s, o)
        elif role == "object":
            slc = np.einsum('pqr,p,q,r->pq', G, v, s, o)
        else:
            raise ValueError("role must be one of {'verb','subject','object'}")
        return slc

    # we create a wrapper that routes to any of the slicing methods
    def get_slice(self, triple: Tuple[str, str, str], role: str, method: str="slice") -> np.ndarray:
        if method == "slice":
            return self.get_role_slice(role=role)
        elif method == "weighted_tuple":
            return self.get_weighted_role_slice_from_tuple(triple, role=role)
        elif method == "tuple":
            return self.role_slice_from_tuple(triple, role=role)
        else:
            raise ValueError("method must be one of {'slice','weighted_tuple','tuple'}")



    # -- Visualisation and inspection methods ---
    def visualize_slice(self,
                        triple: Tuple[str, str, str],
                        role: str,
                        normalize: bool=False,
                        method: str="slice"):


        target_word = triple[{"verb":0, "subject":1, "object":2}[role]]
        slc = self.get_slice(triple=triple, role=role, method=method)
        if method == "slice":
            word_id = self.vocab[f"{role[0]}2i"][target_word]
            slc = slc[word_id]

        if normalize:
            slc = slc / np.linalg.norm(slc)
        plt.figure(figsize=(10, 8))
        im = plt.imshow(slc, cmap="Greys", aspect="auto")
        plt.colorbar(im)

        plt.title(f"{role.capitalize()}-mode integrated core tensor for '{target_word}'")
        plt.xlabel("Latent dimension 1")
        plt.ylabel("Latent dimension 2")

        plt.tight_layout()
        plt.show()

    # top activations utility
    def retrieve_highest_activations(self,
                                     triple: Tuple[str, str, str],
                                     role: str,
                                     method: str="slice",
                                     top_k: int=10):
        target_word = triple[{"verb":0, "subject":1, "object":2}[role]]
        slc = self.get_slice(triple=triple, role=role, method=method)
        if method == "slice":
            word_id = self.vocab[f"{role[0]}2i"][target_word]
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
        factor_idx = _role_index(role)
        role_factors = self.factors[factor_idx]      # (N, R)
        dim_values = _to_np(role_factors)[:, dim_index]

        scores, indices = torch.topk(torch.tensor(dim_values), top_k)
        vocab_list = self.vocab[f"vocab_{role[0]}"]

        top_words = [
            (vocab_list[idx.item()], score.item())
            for idx, score in zip(indices, scores)
        ]
        return top_words

    # todo: no safeguard against division by 0

    def get_expected_element(self, target_tuple, role, verbose=True):
        index = _role_index(role)
        r2i = _voc_index(role)
        latents = self.fetch_latents(target_tuple)
        G_item = self.excluded_role_vector(target_tuple, role=role)
        factor = self.factors[index].cpu().numpy()
        similarities = factor @ G_item / (np.linalg.norm(factor, axis=1) * np.linalg.norm(G_item))
        # we get the top 5 most similar verbs
        k = 5
        top_k_indices = np.argsort(similarities)[-k:][::-1]

        results = []
        for idx in top_k_indices:
            role_str = next(k for k, v in self.vocab[r2i].items() if v == idx)

            role_act = self.factors[index][idx, :].cpu().numpy()
            cos_sim = np_sim(role_act, latents[index])

            results.append({"token": role_str,
                            "similarity": float(similarities[idx]),
                            "activation_cosine": float(cos_sim)})

        if verbose:
            print(f"Top {k} most similar {role}s to the integrated core tensor:")
            for r in results:
                print(f"Subject: {r['token']}, "
                      f"Similarity: {r['similarity']}, "
                      f"Cosine sim with target {role} activations: {r['activation_cosine']}"
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
        else:
            raise ValueError("Must be tuple or str")

        i = _role_index(role)
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
        r2i = _voc_index(role)

        top_sims = []
        for idx in top_idx:
            role_str = next(k for k, v in self.vocab[r2i].items() if v == idx)
            top_sims.append(role_str)

        return top_sims


class ExtendedTucker(TuckerDecomposition):
    def __init__(self, core, factors: List[torch.Tensor], vocab: dict):
        super().__init__(core, factors, vocab)
        self.is_extended: bool = False
        self.extended_roles: set[str] = set()

        self.extended_tokens: dict[str, set[str]] = {
            "verb": set(),
            "subject": set(),
            "object": set()
        }

        self.extensions: dict[str, dict[str, np.ndarray]] = {
            "verb": {},
            "subject": {},
            "object": {}
        }

        self.extension_counts: dict[str, dict[str, int]] = {
            "verb": {},
            "subject": {},
            "object": {}
        }

        self.extension_lengths: dict[str, int] = {
            "verb": 0,
            "subject": 0,
            "object": 0
        }

    @classmethod
    def from_tucker(cls, t: TuckerDecomposition) -> "ExtendedTucker":
        """
        Create an ExtendedTucker that shares core/factors/vocab references with `t`.
        (No copying; changes to dense tensors in-place will reflect in both.)
        """
        return cls(t.core, t.factors, t.vocab)

    @classmethod
    def extend_tucker(
        cls,
        t: TuckerDecomposition,
        dataset,  # iterable of triples (verb, subject, object)
        roles: List[str],
        normalize: bool = True,
        normalize_mode: Literal["l2", "minmax"] = "l2",
        n_threads: int | None = None,
        thread_budget: ThreadBudget | None = None,
        fraction_threads: float = 0.75,
        min_threads: int = 1,
        min_count: int | None = None,
        top_k: int | None = None,
    ) -> "ExtendedTucker":
        """
        Build an ExtendedTucker from `t` and extend specified roles in-place on the new object.

        Example:
            extended = ExtendedTucker.extend_tucker(tucker, sample, ["verb","object"], min_count=5)
        """

        ext = cls.from_tucker(t)
        for role in roles:

            ext.extend_role(
                role=role,
                sample=dataset,
                normalize=normalize,
                normalize_mode=normalize_mode,
                n_threads=n_threads,
                thread_budget=thread_budget,
                fraction_threads=fraction_threads,
                min_threads=min_threads,
                min_count=min_count,
                top_k=top_k,
            )
        return ext

    # -- Override vocab-based extensions --
    def check_vocab(self, triple: Tuple[str, str, str]) -> bool:
        """True if each element is either in base vocab OR in extensions for that role."""
        v_ok = (triple[0] in self.vocab["v2i"]) or (triple[0] in self.extensions["verb"])
        s_ok = (triple[1] in self.vocab["s2i"]) or (triple[1] in self.extensions["subject"])
        o_ok = (triple[2] in self.vocab["o2i"]) or (triple[2] in self.extensions["object"])
        return v_ok and s_ok and o_ok

    def fetch_single_latent(self, element, role) -> np.ndarray:
        """
        First try base vocab, else fall back to extension dict.
        """
        vocab_key = _voc_index(role)
        if element in self.vocab[vocab_key]:
            el_idx = self.vocab[vocab_key][element]
            factor_slice = self.factors[_role_index(role)][el_idx]
            return _to_np(factor_slice)

        if element in self.extensions[role]:
            return np.asarray(self.extensions[role][element])

        raise KeyError(f"{element!r} not in base vocab and not extended for role {role!r}")

    def fetch_latents(self, triple: Tuple[str, str, str]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Use fetch_single_latent so extended tokens work everywhere.
        """
        v = self.fetch_single_latent(triple[0], "verb")
        s = self.fetch_single_latent(triple[1], "subject")
        o = self.fetch_single_latent(triple[2], "object")
        return v, s, o

    # -- Extension method --
    def extend_role(
            self,
            role: str,
            sample,  # iterable of (v,s,o) triples
            normalize: bool =True,
            normalize_mode: Literal["l2", "minmax"] = "l2",

            n_threads: int | None = None,
            thread_budget: ThreadBudget | None = None,
            fraction_threads: float = 0.75,
            min_threads: int = 1,
            min_count: int | None = None,
            top_k: int | None = None,
    ) -> dict[str, np.ndarray]:
        """
        Extend ONE role, store vectors in self.extensions[role], and return the computed dict.

        Notes:
          - We only build reps for OOV tokens that appear in contexts where the other two are in-vocab.
          - Results are merged into any existing extension (later calls can add more tokens).

        See matrix extension.ipynb for development
        """
        if role not in {"verb", "subject", "object"}:
            raise ValueError(f"role must be one of {{'verb','subject','object'}}, got {role!r}")

        # default threads
        if n_threads is None:
            n_threads = get_eval_num_threads(fraction=fraction_threads, min_threads=min_threads)

        r_idx = _role_index(role)
        other_idxs = [i for i in (0, 1, 2) if i != r_idx]

        this_vocab_key = _voc_index(role)
        other_vocab_keys = [_voc_index("verb"), _voc_index("subject"), _voc_index("object")]
        other_vocab_keys = [other_vocab_keys[i] for i in other_idxs]

        this_vocab = self.vocab[this_vocab_key]
        other_vocabs = [self.vocab[k] for k in other_vocab_keys]


        # if normalize:
        #     F_base = _to_np(self.factors[_role_index(role)])  # (N, R)
        #     base_row_norms = np.linalg.norm(F_base, axis=1)
        #     nz = base_row_norms[np.isfinite(base_row_norms) & (base_row_norms > 0)]
        #     print(nz)
        #     target_norm = float(np.median(nz)) if nz.size else 1.0
        #     print(f"target norm:", target_norm)
        #     eps = 1e-12
        #
        #     def _rescale_to_target(vec: np.ndarray) -> np.ndarray:
        #         n = float(np.linalg.norm(vec))
        #         if (not np.isfinite(n)) or (n < eps):
        #             return np.zeros_like(vec, dtype=np.float64)
        #         return vec * (target_norm / (n + eps))
        # else:
        #
        #     def _rescale_to_target(vec: np.ndarray) -> np.ndarray:
        #            return vec

        eps = 1e-12
        range_q = (1.0, 99.0)
        if normalize and normalize_mode == "l2":
            F_base = _to_np(self.factors[_role_index(role)]).astype(np.float64, copy=False)  # (N, R)
            base_row_norms = np.linalg.norm(F_base, axis=1)
            nz = base_row_norms[np.isfinite(base_row_norms) & (base_row_norms > 0)]
            target_norm = float(np.median(nz)) if nz.size else 1.0

            def _post_normalize(extension: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
                out2 = {}
                for tok, vec in extension.items():
                    vec = np.asarray(vec, dtype=np.float64)
                    n = float(np.linalg.norm(vec))
                    if (not np.isfinite(n)) or (n < eps):
                        out2[tok] = np.zeros_like(vec, dtype=np.float64)
                    else:
                        out2[tok] = vec * (target_norm / (n + eps))
                return out2

        elif normalize and normalize_mode == "minmax":
            F_base = _to_np(self.factors[_role_index(role)]).astype(np.float64, copy=False)  # (N, R)

            lo_q, hi_q = range_q
            base_lo = np.nanpercentile(F_base, lo_q, axis=0)
            base_hi = np.nanpercentile(F_base, hi_q, axis=0)
            base_span = np.maximum(base_hi - base_lo, eps)
            base_mid = (base_lo + base_hi) * 0.5

            def _post_normalize(extension: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
                if not extension:
                    return extension

                toks = list(extension.keys())
                E = np.stack([np.asarray(extension[t], dtype=np.float64) for t in toks], axis=0)  # (M, R)

                ext_lo = np.nanpercentile(E, lo_q, axis=0)
                ext_hi = np.nanpercentile(E, hi_q, axis=0)
                ext_span = ext_hi - ext_lo

                # dims with no spread in the batch -> pin to base midpoint
                flat = ext_span < eps
                safe_span = np.where(flat, 1.0, ext_span)

                E2 = (E - ext_lo) * (base_span / safe_span) + base_lo
                E2[:, flat] = base_mid[flat]

                # clean up NaNs/infs if any sneak in
                bad = ~np.isfinite(E2)
                if np.any(bad):
                    E2[bad] = np.take(base_mid, np.where(bad)[1])

                # if clip_to_base:
                #     E2 = np.clip(E2, base_lo, base_hi)

                return {t: E2[i] for i, t in enumerate(toks)}

        else:
            def _post_normalize(extension: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
                return extension

        # collect OOV contexts
        out = defaultdict(list)
        for triple in tqdm(sample, desc=f"building OOV add list ({role})"):
            tok = triple[r_idx]

            # skip already base-vocab or already-extended
            if (tok in this_vocab) or (tok in self.extensions[role]):
                continue

            if all(triple[o_i] in o_v for o_i, o_v in zip(other_idxs, other_vocabs)):
                out[tok].append(triple)

        if not out:
            return {}

        # filter by min_count / top_k
        if (min_count is not None) or (top_k is not None):
            counts0 = {tok: len(ctxs) for tok, ctxs in out.items()}

            if min_count is not None:
                keep = {tok for tok, c in counts0.items() if c >= min_count}
            else:
                keep = set(counts0.keys())

            if top_k is not None:
                ranked = sorted(keep, key=lambda t: (-counts0[t], t))
                keep = set(ranked[:top_k])

            out = {tok: out[tok] for tok in keep}

        if not out:
            return {}

        # ThreadBudget limiter
        limiter = (thread_budget.limit() if thread_budget is not None else None)
        if limiter is None:
            ctx = contextmanager(lambda: (yield))()  # no-op
        else:
            ctx = limiter

        sums: dict[str, np.ndarray] = {}
        counts: dict[str, int] = {}

        def _one_call(tok, ctx_triple):
            rep = self.predicted_role_vector(ctx_triple, role)
            rep = np.asarray(rep, dtype=np.float64)
            # rep = _rescale_to_target(rep)
            return tok, rep

        jobs = [(tok, ctx_triple) for tok, ctxs in out.items() for ctx_triple in ctxs]
        if not jobs:
            return {}

        with ctx:
            with ThreadPoolExecutor(max_workers=n_threads) as ex:
                futures = [ex.submit(_one_call, tok, ctx_triple) for tok, ctx_triple in jobs]

                for fut in tqdm(as_completed(futures), total=len(futures), desc=f"calculating reps ({role})"):
                    tok, rep = fut.result()
                    if tok not in sums:
                        sums[tok] = rep.copy()
                        counts[tok] = 1
                    else:
                        sums[tok] += rep
                        counts[tok] += 1

        # average per token
        extension = {tok: (sums[tok] / counts[tok]) for tok in sums.keys()}

        # if normalize:
        #     for tok in list(extension.keys()):
        #         extension[tok] = _rescale_to_target(np.asarray(extension[tok], dtype=np.float64))
        # post-normalize (either l2 or per-dimension minmax)
        extension = _post_normalize(extension)

        # store + flags
        for tok, vec in extension.items():
            self.extensions[role][tok] = np.asarray(vec)
            self.extended_tokens[role].add(tok)
            self.extension_counts[role][tok] = int(counts[tok])

        self.extension_lengths[role] = len(self.extensions[role])
        self.is_extended = True
        self.extended_roles.add(role)

        return extension






    def select_top_k(self, role: str, top_k: int):
        """
        Keep only the top_k most common extended tokens for `role`
        (based on self.extension_counts[role][tok]).

        If fewer than top_k tokens exist for that role, raise an error.
        """
        if role not in {"verb", "subject", "object"}:
            raise ValueError(f"role must be one of {{'verb','subject','object'}}, got {role!r}")
        if top_k <= 0:
            raise ValueError(f"top_k must be > 0, got {top_k}")
        n_ext = len(self.extensions[role])
        if n_ext < top_k:
            raise ValueError(
                f"Not enough extended tokens for role {role!r}: "
                f"have {n_ext}, requested top_k={top_k}"
            )

        # rank by count desc, then token for determinism
        counts = self.extension_counts[role]
        ranked = sorted(self.extensions[role].keys(), key=lambda t: (-counts.get(t, 0), t))

        keep = set(ranked[:top_k])

        # drop everything else
        drop = [tok for tok in self.extensions[role].keys() if tok not in keep]
        for tok in drop:
            self.extensions[role].pop(tok, None)
            self.extended_tokens[role].discard(tok)
            self.extension_counts[role].pop(tok, None)

        # update length
        self.extension_lengths[role] = len(self.extensions[role])

        # if we removed everything for a role, also update extended_roles / is_extended
        if self.extension_lengths[role] == 0:
            self.extended_roles.discard(role)
        self.is_extended = any(self.extension_lengths[r] > 0 for r in ("verb", "subject", "object"))

        return ranked[:top_k]


    def integrate_extension(self, top_k: int|None) -> TuckerDecomposition:
        """
        Materialize (append) the extension vectors into the factor matrices + vocab,
        returning a plain TuckerDecomposition.

        - for each role, keep only the most common top_k extended tokens (by extension_counts)
        - if a role has < top_k extended tokens: raise
        """
        roles = ["verb", "subject", "object"]
        top_ks = {}
        # sanity
        if top_k:
            for role in roles:
                n_ext = self.extension_lengths[role]
                if n_ext < top_k:
                    raise ValueError(
                        f"Not enough extended tokens for role {role!r}: "
                        f"have {n_ext}, requested top_k={top_k}"
                    )
                if n_ext > top_k:
                    self.select_top_k(role, top_k)
                top_ks[role] = top_k

        else:
            top_ks = {role:self.extension_lengths[role] for role in roles}
        print(top_ks)
        # build new vocab (shallow copy of dict + copies of the lists/mappings we mutate)
        new_vocab = dict(self.vocab)
        for role in roles:
            list_key = f"vocab_{role[0]}"  # vocab_v / vocab_s / vocab_o
            map_key = _voc_index(role)  # v2i / s2i / o2i

            new_vocab[list_key] = list(new_vocab[list_key])
            new_vocab[map_key] = dict(new_vocab[map_key])


        # build new factors by appending extension rows
        new_factors: List[Union[torch.Tensor, np.ndarray]] = []
        for role in roles:
            f_idx = _role_index(role)
            F = self.factors[f_idx]

            # deterministic order: most common first, then token
            counts = self.extension_counts[role]
            if counts == {}:
                print("No extensions for role", role)
                new_factors.append(F)
                continue
            toks = sorted(self.extensions[role].keys(), key=lambda t: (-counts.get(t, 0), t))
            toks = toks[:top_ks[role]]

            vecs_np = np.stack([np.asarray(self.extensions[role][tok]) for tok in toks], axis=0)

            # append to factor (preserve type/device/dtype where possible)
            if isinstance(F, torch.Tensor):
                add = torch.tensor(vecs_np, dtype=F.dtype, device=F.device)
                F_new = torch.cat([F, add], dim=0)
            else:
                F_np = _to_np(F)
                F_new = np.vstack([F_np, vecs_np])

            new_factors.append(F_new)

            # update vocab list + mapping
            list_key = f"vocab_{role[0]}"
            map_key = _voc_index(role)

            base_n = len(new_vocab[list_key])
            for j, tok in enumerate(toks):
                new_vocab[list_key].append(tok)
                new_vocab[map_key][tok] = base_n + j

        # optional: store a tiny bit of provenance in vocab
        # new_vocab["is_extended"] = True
        # new_vocab["extension_top_k"] = int(top_k)

        return TuckerDecomposition(self.core, new_factors, new_vocab)

    # -- Saving and loading --


    def save_extensions(self,
            path: str,
            *,
            roles: Optional[list[str]] = None,
    ) -> None:
        """
        Save ONLY extension vectors (as factor-row matrices) + metadata needed to restore them.

        File contains:
          - per role: tokens (ordered), counts, matrix (n_ext, R)
          - flags: is_extended, extended_roles, extension_lengths
          - a few sanity fields: rank
        """
        if roles is None:
            roles = ["verb", "subject", "object"]

        R = self.factors[0].shape[1] # or core, but as we're working with factors here its explicit

        payload = {
            "rank": R,
            "is_extended": bool(self.is_extended),
            "extended_roles": sorted(list(self.extended_roles)),
            "extension_lengths": dict(self.extension_lengths),
            "roles": {},
        }

        for role in roles:
            if role not in {"verb", "subject", "object"}:
                raise ValueError(f"Unknown role: {role!r}")

            # deterministic order: most common first, then token
            counts = self.extension_counts.get(role, {})
            toks = sorted(self.extensions[role].keys(), key=lambda t: (-counts.get(t, 0), t))

            if len(toks) == 0:
                payload["roles"][role] = {
                    "tokens": [],
                    "counts": [],
                    "matrix": None,
                    "dtype": None,
                }
                continue

            mat = np.stack([np.asarray(self.extensions[role][tok]) for tok in toks], axis=0)
            if mat.shape[1] != R:
                raise ValueError(
                    f"Extension rank mismatch for role {role}: got {mat.shape[1]}, expected {R}"
                )

            role_counts = [int(counts.get(tok, 0)) for tok in toks]


            payload["roles"][role] = {
                "tokens": toks,
                "counts": role_counts,
                "matrix": torch.from_numpy(mat),
                "dtype": str(mat.dtype),
            }


        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        torch.save(payload, path)


    def load_extensions_inplace(
            self,
            path: str,
            *,
            map_location: Union[str, torch.device] = "cpu",
            strict_rank: bool = True,
            overwrite: bool = False,
    ) -> None:
        """
        Load saved extensions into THIS ExtendedTucker instance.

        - Does NOT touch core/factors/vocab (so this object stays "the same" wrt the base tucker).
        - Populates: extensions, extended_tokens, extension_counts, extension_lengths, is_extended, extended_roles.

        Args:
          strict_rank: if True, require saved rank == current rank.
          overwrite: if True, replace existing extensions for roles; else merge (new tokens added, existing left as-is).
        """
        # ---- load payload ----
        try:
            payload = torch.load(path, map_location=map_location)
        except Exception:
            # allow non-torch pickle fallback
            import pickle
            with open(path, "rb") as f:
                payload = pickle.load(f)

        saved_R = int(payload["rank"])
        cur_R = self.factors[0].shape[1]
        if strict_rank and saved_R != cur_R:
            raise ValueError(f"Rank mismatch: file rank={saved_R}, current rank={cur_R}")

        roles_blob = payload.get("roles", {})
        for role, blob in roles_blob.items():
            if role not in {"verb", "subject", "object"}:
                continue

            toks = blob.get("tokens", []) or []
            counts = blob.get("counts", []) or []
            mat = blob.get("matrix", None)

            if overwrite:
                self.extensions[role].clear()
                self.extended_tokens[role].clear()
                self.extension_counts[role].clear()

            if mat is None or len(toks) == 0:
                self.extension_lengths[role] = len(self.extensions[role])
                continue

            if isinstance(mat, torch.Tensor):
                mat_np = mat.detach().cpu().numpy()
            else:
                mat_np = np.asarray(mat)

            if mat_np.ndim != 2 or mat_np.shape[1] != cur_R:
                raise ValueError(
                    f"Bad matrix shape in file for role {role!r}: {mat_np.shape}, expected (n,{cur_R})"
                )
            if len(counts) != len(toks):
                raise ValueError(f"Counts length != tokens length for role {role!r}")

            for i, tok in enumerate(toks):
                if (not overwrite) and (tok in self.extensions[role]):
                    # keep existing
                    continue
                vec = np.asarray(mat_np[i], dtype=np.float64)  # keep your extension dtype stable
                self.extensions[role][tok] = vec
                self.extended_tokens[role].add(tok)
                self.extension_counts[role][tok] = int(counts[i])

            self.extension_lengths[role] = len(self.extensions[role])

        # flags
        self.extended_roles = {r for r in ("verb", "subject", "object") if self.extension_lengths[r] > 0}
        self.is_extended = len(self.extended_roles) > 0

    @classmethod
    def load_extensions(
            cls,
            t: TuckerDecomposition,
            path: str,
            *,
            map_location: Union[str, torch.device] = "cpu",
            strict_rank: bool = True,
            overwrite: bool = False,
    ) -> "ExtendedTucker":
        """
        Convenience: create an ExtendedTucker that shares references with `t`,
        then load extensions from file.
        """
        ext = cls.from_tucker(t)
        ext.load_extensions_inplace(
            path,
            map_location=map_location,
            strict_rank=strict_rank,
            overwrite=overwrite,
        )
        return ext



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
            size = int(np.prod(shape))
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
    if not x.is_sparse:
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
        size = int(np.prod(shape))  # prod of original N-D shape
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
    def load_from_disk(cls,
                       dataset: str="fineweb-en",
                       method: str="siiSoftPlus",
                       dims: int=1000,
                       map_location: str="cpu",
                       tier1: bool=False,
                       shared_factors: set|None=None,
                          ) -> "SparseTupleTensor":

        """Loads a precomputed tucker decomposition from disk.
            Args:
                dataset (str): name of the dataset
                method (str): method used to compute the decomposition
                    - one of "counting", "sc", "sii"
                dims (int): dimensionality of the original tensor modes (vocab size)
                rank (int): rank of the decomposition
                iterations (int): number of iterations used to compute the decomposition
                map_location (str): device to map the loaded tensors to
            Returns:
                ((core, factors), vocab)
                    core: torch.Tensor
                    factors: list[torch.Tensor]

        """
        if method not in {"counting", "sc", "sii",
                      "siiSoftPlus", "siiShifted", "scSoftPlus", "scShifted"}:
            raise ValueError("method must be one of {'counting','sc','sii'}")
        base = os.path.join(DATA_DIR, "tensors", dataset)
        base = readonly_dispatch(base, tier1)
        is_shared = bool(shared_factors)
        if is_shared:
            print("loading in shared population:", shared_factors)
        suffix = "_shared12" if is_shared else ""

        # vocab_path = os.path.join(base, f"vocabularies/{dims}.pkl")
        populated_path = os.path.join(base,"populated", f"{method}_{dims}{suffix}.pt")
        # if not os.path.exists(vocab_path):
        #     raise FileNotFoundError(f"Missing vocab file: {vocab_path}")
        if not os.path.exists(populated_path):
            raise FileNotFoundError(f"Missing decomposition file: {populated_path}")
        # the vocab is here under f"vocabularies_[dims].pkl"
        # Load with torch (they were saved with torch.save)
        # with open(vocab_path, "rb") as f:
        #     vocab = pickle.load(f)
        tensor = torch_or_pickle_load(populated_path, map_location=map_location)

        return cls(tensor, device=map_location, sparsity_type="torch", shared_factors=shared_factors)



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
            # can only work from a sparse torch tensor
            if not isinstance(self.tensor, torch.Tensor) or not self.tensor.is_sparse:
                raise TypeError("sparse expects self.tensor to be a torch sparse tensor.")
            coords = self.tensor.indices().numpy()       # shape (nnz, ndim)
            data   = self.tensor.values().numpy()        # shape (nnz,)
            shape  = tuple(self.tensor.size())  # e.g. (d0, d1, ..., d_{n-1})
            sparse_tensor = sparse.COO(coords, data, shape=shape)
            return sparse_tensor

        elif sparse_type == "cupy":
            if not isinstance(self.tensor, torch.Tensor) or not self.tensor.is_sparse:
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
                               divergence="fr",
                               rank=100,
                               ):

        dim = self.shape[0]
        nnz = self.tensor._nnz()
        if divergence == "fr" and dim <= 4000:
            print("fast GPU algorithm, estimated time:", dim/1000)
        elif divergence == "kl" and dim < 4000:
            print("fast GPU algorithm, estimated time:", dim/200)
        else:
            factor_time = (rank**1.76) * (dim**0.16) * (nnz**0.78) * 1e-9
            print(factor_time, "estimated time per factor update")
            core_time = (rank**2.6) * nnz * 1e-10
            print(core_time, "estimated time per core update")
            print("total:", factor_time + core_time)

    def non_negative_tucker_with_similarity(
            self,
            cfg: RunConfig,
            thread_budget: ThreadBudget,
            vocab=None,
            sample_sentences=None,
            checkpoint_tensor: TuckerDecomposition=None,
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
            time_iteration = cfg.eval.time_iteration
            # saving
            save_intermediate = cfg.eval.save_intermediate


        except:
            raise ValueError("Check config structure.")

        if not isinstance(self, SparseTupleTensor):
            raise TypeError("sparse_tensor must be a SparseTupleTensor instance.")
        if not self.sparsity_type == "cupy":
            raise ValueError("sparse_tensor must have sparsity_type 'cupy'.")

        paths = cfg.artifact_paths()

        if checkpoint_saving:
            os.makedirs(paths["checkpoint_dir"], exist_ok=True)


        shape = tuple(self.shape)
        rank = validate_tucker_rank(shape, rank=rank)
        modes = list(range(len(rank)))
        if checkpoint_tensor:
            core = cp.asarray(checkpoint_tensor.core)
            factors = [cp.asarray(factor) for factor in checkpoint_tensor.factors]
        else:
            core, factors = initialize_nonnegative_tucker(self.tensor, shape, rank, modes, init, random_state)

        linked_factors = defaultdict(set)
        if self.shared_factors:
            for a, b in self.shared_factors:
                linked_factors[a].add(b)
                linked_factors[b].add(a)


        rec_errors = []
        fitness_scores = []
        no_rec_improve_steps = 0
        last_err = None

        sem_no_rec_improve_steps = 0
        best_sem_score = 0
        best_sem_iteration = None

        # Decide once which semantic metric drives patience/diff
        if sem_error_type == "all":
            sem_primary_key = "average_rank_score"  # stable default (your dict always includes this)
        elif isinstance(sem_error_type, (list, tuple)):
            if len(sem_error_type) == 0:
                raise ValueError("sem_error_type list/tuple must contain at least one key.")
            sem_primary_key = sem_error_type[0]
        else:
            sem_primary_key = sem_error_type

        for iteration in range(n_iter_max):
            if time_iteration:
                start_time = time.time()
            log_step = get_log_step(iteration, rec_log_every, rec_check_every)
            routing = get_update_routing_step(divergence=divergence, dim=dim, log_step=log_step, largedim=largedim)
            # --- factors ---

            for mode in modes:
                factors[mode] = routing.factor_update(
                    vec_tensor=self.tensor,
                    core=core,
                    factors=factors,
                    mode=mode,
                    shape=shape,
                    thread_budget=thread_budget,
                    epsilon=epsilon,
                )

                # new: factor linking
                if mode in linked_factors:
                    for other in linked_factors[mode]:
                        factors[other] = factors[mode]


            # --- core + error ---
            if routing.core_returns_error:
                # FR: combined core update + error in one call
                core, rel_err = routing.core_update(
                    vec_tensor=self.tensor,
                    shape=shape,
                    core=core,
                    factors=factors,
                    modes=modes,
                    thread_budget=thread_budget,  # we always pass it, even if not needed, to ensure consistency
                    epsilon=epsilon,
                )
            else:
                # KL: core update, then compute error separately
                core = routing.core_update(
                    vec_tensor=self.tensor,
                    shape=shape,
                    core=core,
                    factors=factors,
                    modes=modes,
                    thread_budget=thread_budget,
                    epsilon=epsilon,
                )
                rel_err = routing.error_fn(
                    vec_tensor=self.tensor,
                    shape=shape,
                    core=core,
                    factors=factors,
                    thread_budget=thread_budget,
                    epsilon=epsilon,
                )
            # Normalize if desired
            if normalize_factors:
                core, factors = tucker_normalize((core, factors))

            if log_step:
                rec_errors.append(rel_err)

                # ---- reconstruction + patience ----
                has_prev_err = len(rec_errors) >= 2
                if verbose and has_prev_err:
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
                            if verbose and no_rec_improve_steps:
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
                tucker_decomp = TuckerDecomposition(core=core_cpu, factors=factors_cpu, vocab=vocab)

                print(iteration, end=":\t")
                sem_out = evaluate_sample(
                    tucker_decomp,
                    sample_sentences,
                    sampled=True,
                    seed=random_state,
                    thread_budget=thread_budget,
                    return_type=sem_error_type,
                )
                fitness_scores.append(sem_out)

                # Primary value used for early stopping / diff
                if isinstance(sem_out, dict):
                    if sem_primary_key not in sem_out:
                        raise KeyError(f"Primary semantic key '{sem_primary_key}' missing from returned scores.")
                    sem_value = float(sem_out[sem_primary_key])
                else:
                    sem_value = float(sem_out)

                tl.set_backend("cupy")

                # track best semantic model (based on primary key)
                diff = sem_value - float(best_sem_score)
                if diff > 0:
                    best_sem_score = sem_value
                    best_core = core.copy()
                    best_factors = factors.copy()
                    best_sem_iteration = iteration
                    if verbose:
                        print("New best semantic score; saving current best core and factors.")
                    if save_intermediate:
                        import json

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
                    if verbose and sem_no_rec_improve_steps:
                        print(f"\tSemantic improvement (Δ={diff:.3e}); resetting patience counter.")
                    sem_no_rec_improve_steps = 0

            if checkpoint_saving: # only trigger if this is not 0 -> True
                if (iteration + 1) % cfg.train.checkpoint_saving_steps == 0:
                    print(f"saving model at iteration {iteration}")
                    checkpoint_tensor = TuckerTensor((core, factors))
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

