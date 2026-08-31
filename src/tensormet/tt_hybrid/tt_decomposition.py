"""
tt_decomposition.py — inference/eval wrapper for Tucker-TT hybrid models.

``TuckerTTDecomposition`` subclasses ``TuckerDecomposition``. The factors are
Tucker factors, so every factor-level method (vocab access, inspection,
nearest neighbours, ``load_from_disk``) is inherited unchanged. Only the core
contractions are overridden: a chain of matrix products instead of one R^N
einsum.
"""
from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np
import torch

from tensormet.tucker_tensor import TuckerDecomposition
from tensormet.utils import (
    _to_np,
    extract_roles_from_vocab,
    make_lazy_cupy_pair,
    np_dispatch,
    resolve_checkpoint_path,
)
from tensormet.tt_hybrid.tt_chain import (
    contract, left_envs, right_envs, site_grad, sites, to_dense_core,
)

cp, cpx_sparse = make_lazy_cupy_pair()

# Guard for the dense-core compatibility property: Π R_k elements above this
# raise instead of silently allocating (an order-5 rank-100 core is 40 GB —
# the whole reason this format exists).
_CORE_MATERIALIZE_MAX_ELEMENTS = 20_000_000  # ~160 MB fp64


class TuckerTTDecomposition(TuckerDecomposition):
    """A Tucker-TT hybrid decomposition: TT cores plus Tucker factors."""

    _DECOMPOSITION = "tt"

    # Not super().__init__: the base assigns self.core, which is a read-only
    # property here.
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
        """The dense Tucker core, reconstructed from the chain. Cached after
        first access, refused above a size guard — prefer the TT-native methods
        below, which never need it."""
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

    def get_rank(self, role=None):
        """Tucker rank of a role's factor — read off the factor, not the core."""
        if role is None:
            role = self.roles[0]
        return int(self.factors[self.get_role_index(role)].shape[1])

    def bond_dims(self):
        """ρ_0..ρ_N — the information bottleneck at each cut of the chain."""
        return tuple([int(self.tt_cores[0].shape[0])]
                     + [int(C.shape[2]) for C in self.tt_cores])

    def update_from_path(self, path=None):
        resolved = resolve_checkpoint_path(path, self.decomp_path)
        tt_cores, factors = torch.load(resolved, map_location="cpu", weights_only=False)
        self.tt_cores = np_dispatch(list(tt_cores))
        self.factors = np_dispatch(list(factors))
        self._core_cache = None
        self.decomp_path = resolved

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

        mats = [None if i == target else self.factors[i][valid_indices[:, i]]
                for i in range(len(self.roles))]
        S = sites(tt_cores, mats, torch, skip=target)
        return site_grad(left_envs(S, torch)[target], tt_cores[target],
                         right_envs(S, torch)[target + 1], torch)

    def _fixed_role_matrix(self, fixed_idx, other_idxs, fixed_latent):
        """The (R_a, R_b) matrix get_top_combinations ranks against — contracted
        off the chain, so it never materializes the dense core."""
        latents: list = [None] * len(self.roles)
        latents[fixed_idx] = fixed_latent
        return contract(self._tt_np(), latents, other_idxs, np)


__all__ = ["TuckerTTDecomposition"]

