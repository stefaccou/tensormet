"""
utt_decomposition.py — inference/eval wrapper for shared-core ("uniform") TT models.

``UniformTTDecomposition`` takes the compact model ``[C, alpha, omega]`` and
expands it to an ordinary TT core list for the model's own order, so every
inherited ``TuckerTTDecomposition`` method (scoring, role vectors, top
combinations, the batched eval contract) runs unchanged.

What only this family can do: score tuples of ANY length with the same
parameters (``score_sequence``, ``role_vector``, ``predict_words``), and expose
the per-dimension operators M_r = C[:, r, :] (``operators``).
"""
from __future__ import annotations

from typing import List, Optional, Sequence

import numpy as np
import torch

from tensormet.tt_hybrid.tt_chain import contract
from tensormet.tt_hybrid.tt_decomposition import TuckerTTDecomposition
from tensormet.tucker_tensor import voc_list_key
from tensormet.utils import make_lazy_cupy_pair, to_np
from tensormet.uniform_tt.utt_chain import expand_uniform

cp, cpx_sparse = make_lazy_cupy_pair()


class UniformTTDecomposition(TuckerTTDecomposition):
    """A shared-core Tucker-TT model: one core C, boundary vectors α, ω, one factor."""

    _DECOMPOSITION = "utt"

    def __init__(self, tt_cores, factors, vocab: dict,
                 shared_factors: set | None = None,
                 roles: Optional[List[str]] = None):
        # ``tt_cores`` is the compact [C, alpha, omega] (the training payload).
        self.compact = list(tt_cores)
        if len(self.compact) != 3 or self.compact[0].ndim != 3:
            raise ValueError("UniformTTDecomposition expects [C (ρ,R,ρ), alpha (ρ,), omega (ρ,)]")
        super().__init__(expand_uniform(self.compact, len(factors)), factors, vocab,
                         shared_factors=shared_factors, roles=roles)

    # --- loading / devices ---------------------------------------------------
    def update_from_path(self, path=None):
        # The parent loads the saved core list, which here is [C, alpha, omega].
        super().update_from_path(path)
        self.compact = self.tt_cores
        self.tt_cores = expand_uniform(self.compact, len(self.factors))

    def to_cupy(self):
        """Prepare for inference by moving to GPU."""
        for store in (self.compact, self.factors):
            for i, a in enumerate(store):
                if isinstance(a, torch.Tensor):
                    store[i] = cp.array(to_np(a))
        self.tt_cores = expand_uniform(self.compact, len(self.factors))
        self._core_cache = None

    # --- inspection ------------------------------------------------------------
    def bond_dim(self) -> int:
        return int(self.compact[0].shape[0])

    def operators(self) -> np.ndarray:
        """C as (R, ρ, ρ): ``operators()[r]`` is M_r, the state transition that
        latent dimension r causes."""
        return np.ascontiguousarray(to_np(self.compact[0]).transpose(1, 0, 2))

    def boundaries(self):
        """(alpha, omega): start and end state weights."""
        return to_np(self.compact[1]), to_np(self.compact[2])

    def word_operator(self, word: str) -> np.ndarray:
        """S(word) = Σ_r A[word, r] M_r, (ρ, ρ)."""
        return np.tensordot(self.fetch_single_latent(word), self.operators(), axes=(0, 0))

    # --- any-length scoring ----------------------------------------------------
    def _compact_np(self):
        return [to_np(x) for x in self.compact]

    def _latents(self, tokens: Sequence[Optional[str]]):
        # Every mode shares one factor and vocabulary, so role 0's serves all positions.
        return [None if t is None else self.fetch_single_latent(t) for t in tokens]

    def score_sequence(self, tokens: Sequence[str]) -> float:
        """x̂ for a tuple of any length: αᵀ S(t_0) ⋯ S(t_{n-1}) ω."""
        tt = expand_uniform(self._compact_np(), len(tokens))
        return float(contract(tt, self._latents(tokens), [], np))

    def role_vector(self, tokens: Sequence[Optional[str]], position: int) -> np.ndarray:
        """Excluded-role vector (R,) at ``position`` of a tuple of any length:
        the chain with that rank leg open. ``tokens[position]`` is ignored (may
        be None). At the last position this is a next-word score vector."""
        tt = expand_uniform(self._compact_np(), len(tokens))
        return contract(tt, self._latents(tokens), [int(position)], np)

    def predict_words(self, tokens: Sequence[Optional[str]], position: int, top_k: int = 10):
        """Top-k words for the slot at ``position`` (scores A · role_vector)."""
        scores = to_np(self.factors[0]) @ self.role_vector(tokens, position)
        top = np.argsort(-scores)[:top_k]
        words = self.vocab[voc_list_key(self.roles[0])]
        return [(words[i], float(scores[i])) for i in top]


__all__ = ["UniformTTDecomposition"]
