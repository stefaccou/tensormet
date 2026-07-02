"""
Mixin for visualization and inspection methods that can be lifted out of TuckerDecomposition.

HOW TO USE
----------
1. Move the desired methods from TuckerDecomposition into TuckerVizMixin below.
   Keep 'self' as-is — the mixin pattern preserves the full instance API.

2. Add TuckerVizMixin as a base class in tucker_tensor.py:

       class TuckerDecomposition(TuckerVizMixin):
           ...

   Put the mixin first so its methods are easy to override.

3. Remove the moved methods from TuckerDecomposition.

CANDIDATES (marked with # [CANDIDATE] in tucker_tensor.py)
-----------------------------------------------------------
From the "Visualisation and inspection" section (~line 644):
    - visualize_slice              matplotlib heatmap; not part of the core scoring API
    - retrieve_highest_activations diagnostic/debug utility; not called during training

From the TF-sparse section (~line 355) — rarely used, depend on TensorFlow:
    - sparse_representation        TF-based sparse conversion; consider experimental/tucker_sparse.py
    - tensor_to_sparse             thin wrapper over sparse_representation (TF path)
    - tensor_to_dense              thin wrapper; TF path only meaningful here

IMPORTS NEEDED WHEN METHODS ARE MOVED HERE
-------------------------------------------
    import numpy as np
    import matplotlib.pyplot as plt
    from tensormet.utils import voc_index
    # plus anything the specific method uses (see tucker_tensor.py method bodies)
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from tensormet.utils import einsum_letters, voc_index
from tensormet.tucker_tensor import _to_np
from typing import Tuple


class TuckerVizMixin:
    """Visualization and rarely-used inspection methods for TuckerDecomposition."""
    # Paste candidate methods from TuckerDecomposition here.

    # def get_role_slice(self, role: str, normalize: bool=False) -> np.ndarray:
    #     G = self._core_np()
    #
    #     if role == "verb":
    #         # (num_verbs, R) × (R, R, R) -> (num_verbs, R, R)
    #         slc = np.einsum('ip,pqr->i q r', _to_np(self.factors[0]), G)
    #     elif role == "subject":
    #         # (num_subj, R) × (R, R, R) -> (num_subj, R, R)
    #         slc = np.einsum('jp,pqr->j p r', _to_np(self.factors[1]), G)
    #     elif role == "object":
    #         # (num_obj, R) × (R, R, R) -> (num_obj, R, R)
    #         slc = np.einsum('kp,pqr->k p q', _to_np(self.factors[2]), G)
    #
    #     else:
    #         raise ValueError("role must be one of {'verb','subject','object'}")
    #     if normalize:
    #         slc = slc / np.linalg.norm(slc, axis=-1, keepdims=True)
    #     return slc
    #
    # def role_slice_from_tuple(self, triple: Tuple[str, str, str], role: str) -> np.ndarray:
    #     G = self._core_np()
    #     v, s, o = self.fetch_latents(triple)
    #     if role == "verb":
    #         slc = np.einsum('pqr,q,r->qr', G, s, o)
    #     elif role == "subject":
    #         slc = np.einsum('pqr,p,r->pr', G, v, o)
    #     elif role == "object":
    #         slc = np.einsum('pqr,p,q->pq', G, v, s)
    #     else:
    #         raise ValueError("role must be one of {'verb','subject','object'}")
    #     return slc
    #
    # def get_weighted_role_slice_from_tuple(self, triple: Tuple[str, str, str], role: str) -> np.ndarray:
    #     G = self._core_np()
    #     v, s, o = self.fetch_latents(triple)
    #     if role == "verb":
    #         slc = np.einsum('pqr,p,q,r->qr', G, v, s, o)
    #     elif role == "subject":
    #         slc = np.einsum('pqr,p,q,r->pr', G, v, s, o)
    #     elif role == "object":
    #         slc = np.einsum('pqr,p,q,r->pq', G, v, s, o)
    #     else:
    #         raise ValueError("role must be one of {'verb','subject','object'}")
    #     return slc

    def get_role_slice(self, role: str, normalize: bool = False) -> np.ndarray:
        target_idx = self.get_role_index(role)
        G = self._core_np()
        factor = _to_np(self.factors[target_idx])

        modes = einsum_letters(len(self.roles))
        core_str = "".join(modes)
        other_modes = "".join([modes[i] for i in range(len(self.roles)) if i != target_idx])
        v_char = "Z"  # Using 'Z' for the vocab dimension to safely avoid collisions

        eq = f"{v_char}{modes[target_idx]},{core_str}->{v_char}{other_modes}"
        slc = np.einsum(eq, factor, G)

        if normalize:
            slc = slc / np.linalg.norm(slc, axis=-1, keepdims=True)
        return slc

    def role_slice_from_tuple(self, triple: Tuple[str, ...], role: str) -> np.ndarray:
        target_idx = self.get_role_index(role)
        G = self._core_np()
        all_latents = self.fetch_latents(triple)
        latents = [all_latents[i] for i in range(len(self.roles)) if i != target_idx]

        modes = einsum_letters(len(self.roles))
        core_str = "".join(modes)
        other_modes = [modes[i] for i in range(len(self.roles)) if i != target_idx]

        eq = f"{core_str},{','.join(other_modes)}->{''.join(other_modes)}"
        return np.einsum(eq, G, *latents)

    def get_weighted_role_slice_from_tuple(self, triple: Tuple[str, ...], role: str) -> np.ndarray:
        target_idx = self.get_role_index(role)
        G = self._core_np()
        latents = self.fetch_latents(triple)

        modes = einsum_letters(len(self.roles))
        core_str = "".join(modes)
        other_modes = "".join([modes[i] for i in range(len(self.roles)) if i != target_idx])

        eq = f"{core_str},{','.join(modes)}->{other_modes}"
        return np.einsum(eq, G, *latents)

    # we create a wrapper that routes to any of the slicing methods
    def get_slice(self, triple: Tuple[str, ...], role: str, method: str="slice") -> np.ndarray:
        if method == "slice":
            return self.get_role_slice(role=role)
        elif method == "weighted_tuple":
            return self.get_weighted_role_slice_from_tuple(triple, role=role)
        elif method == "tuple":
            return self.role_slice_from_tuple(triple, role=role)
        else:
            raise ValueError("method must be one of {'slice','weighted_tuple','tuple'}")



    # -- Visualisation and inspection methods ---
    # [CANDIDATE → experimental/tucker_viz.py]
    # visualize_slice and retrieve_highest_activations are matplotlib-based
    # diagnostic tools that are not part of the core scoring API.
    # Move them to TuckerVizMixin in experimental/tucker_viz.py when they are
    # no longer needed in the main class.
    def visualize_slice(self,
                        triple: Tuple[str, ...],
                        role: str,
                        normalize: bool = False,
                        method: str = "slice"):

        target_word = triple[self.get_role_index(role)]
        slc = self.get_slice(triple=triple, role=role, method=method)

        if method == "slice":
            word_id = self.vocab[voc_index(role)][target_word]
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

