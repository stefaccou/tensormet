"""
Slicing and visualization methods lifted out of TuckerDecomposition.

Not mixed in yet: to use them, make ``TuckerVizMixin`` a base class of
``TuckerDecomposition`` in tucker_tensor.py.
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from tensormet.utils import einsum_letters, voc_index
from tensormet.utils import to_np
from typing import Tuple


class TuckerVizMixin:
    """Visualization and rarely-used inspection methods for TuckerDecomposition."""
    def get_role_slice(self, role: str, normalize: bool = False) -> np.ndarray:
        target_idx = self.get_role_index(role)
        G = self._core_np()
        factor = to_np(self.factors[target_idx])

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

