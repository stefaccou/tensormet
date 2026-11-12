"""
utt_sgd.py — SGD/Adam model for the shared-core TT (``--decomposition utt``).

Same surface as the other SGD
models (``sgd.sgd_model.SGDModelBase``), so the SGD trainers run it unchanged.
Model:

    x̂(i_0 … i_{N-1}) = αᵀ S(i_0) ⋯ S(i_{N-1}) ω,     S(i) = Σ_r A[i, r] · C[:, r, :]

Parameters: C (ρ, R, ρ), α (ρ,), ω (ρ,) and ONE factor A shared by every mode.
The ``core`` views are the list ``[C, α, ω]``, as in the MU loop. Forward
passes expand it to ordinary TT cores (``utt_chain.expand_uniform``), so the
entry chain is TT's; autograd sums the gradient of C over the N positions and
of A over the N modes, which is the pooled update MU derives by hand.

Gauge (``regauge_``), exact, x̂ unchanged: ℓ1 columns of A (scale into C's R
leg), then the MU gauge (``utt_chain.gauge_scales``): C /= λ (Perron root of
T = Σ_r s_r C[:, r, :]), α *= λᴺ, and α, ω balanced to equal ℓ1 mass.
"""
from __future__ import annotations

import math
from typing import List, Optional, Sequence, Tuple, Union

import numpy as np
import torch

from tensormet.sgd.sgd_model import _EPS, SGDModelBase, raw_from_view, view_from_raw
from tensormet.sgd.sgd_tt import _to_tensor, chain_entry_width, sq_norm
from tensormet.tt_hybrid.tt_chain import chain_values
from tensormet.uniform_tt.utt_chain import (
    chain_total, expand_uniform, gauge_scales, transfer_matrix,
)


def _sum_all(C, alpha, omega, A, n_modes) -> torch.Tensor:
    """Σ_all x̂ = αᵀ Tᴺ ω with T = Σ_r s_r C[:, r, :], s = column sums of A (fp64)."""
    T = transfer_matrix(C.to(torch.float64), A.sum(dim=0, dtype=torch.float64))
    return chain_total(T, alpha.to(torch.float64), omega.to(torch.float64), n_modes)


def _gauge(C, alpha, omega, A, n_modes):
    """Exact gauge reset on fp64 copies; returns new (C, α, ω, A)."""
    tiny = torch.finfo(torch.float64).tiny
    s = A.sum(dim=0).clamp_min(tiny)
    A = A / s
    C = C * s[None, :, None]
    # Perron root on the host: T is only ρ×ρ.
    T = transfer_matrix(C, A.sum(dim=0)).cpu().numpy()
    lam, alpha_mult, omega_mult = gauge_scales(T, alpha.sum(), omega.sum(), n_modes)
    if lam is not None:
        C = C / lam
    return C, alpha * alpha_mult, omega * omega_mult, A


class SGDUTTModel(SGDModelBase):
    """Non-negative shared-core TT. ``core`` views are ``[C, α, ω]``."""

    def __init__(
        self,
        shape: Sequence[int],
        rank: Union[int, Sequence[int]],
        tt_rank: int,
        parametrization: str = "softplus",
        shared_factors=None,
        init: Union[str, Tuple] = "random",
        random_state: int = 0,
        device: Optional[Union[str, torch.device]] = None,
        dtype: torch.dtype = torch.float32,
        eps: float = _EPS,
        sum_x: Optional[float] = None,
    ):
        super().__init__()
        device = self._setup(shape, rank, parametrization, shared_factors, eps, device)
        if self.order < 2:
            raise ValueError("decomposition='utt' needs order >= 2.")
        # All modes tied also means one dim and one rank (checked in _setup).
        if self.owner != [0] * self.order:
            raise ValueError("decomposition='utt' needs every mode to share one factor "
                             "(shared_factors='all').")
        self.tt_rank = int(tt_rank)
        R, rho, n = self.rank[0], self.tt_rank, self.order
        self.compact_shapes = [(rho, R, rho), (rho,), (rho,)]
        expanded = [(1, R, rho)] + [(rho, R, rho)] * (n - 2) + [(rho, R, 1)]
        self.entry_width = chain_entry_width(expanded, self.rank)

        compact0, A0 = self._initial_values(init, random_state, dtype, device, sum_x)
        p = self.parametrization
        self._core_raw = torch.nn.ParameterList(
            [torch.nn.Parameter(raw_from_view(x, p, self.eps)) for x in compact0]
        )
        self._factor_raw = torch.nn.ParameterDict(
            {"0": torch.nn.Parameter(raw_from_view(A0, p, self.eps))}
        )

    # --- init ---------------------------------------------------------------
    def _initial_values(self, init, random_state, dtype, device, sum_x):
        """``random`` uses the same draws as ``utt_ops.initialize_uniform_tt``
        (A from TT's init, then C, α, ω from a fresh RNG with the same seed);
        a ``([C, α, ω], factors)`` tuple is a warm start from an MU utt model."""
        n = self.order
        if isinstance(init, (tuple, list)) and len(init) == 2:
            compact, factors = init
            compact = [_to_tensor(x, dtype, device).abs().clamp_min(self.eps) for x in compact]
            A = _to_tensor(factors[0], dtype, device).abs().clamp_min(self.eps)
            got = [tuple(x.shape) for x in compact]
            if got != self.compact_shapes:
                raise ValueError(f"warm start utt core has shapes {got}; expected "
                                 f"{self.compact_shapes} for rank={self.rank[0]}, "
                                 f"tt_rank={self.tt_rank}.")
            if tuple(A.shape) != (self.shape[0], self.rank[0]):
                raise ValueError(f"warm start factor has shape {tuple(A.shape)}; expected "
                                 f"{(self.shape[0], self.rank[0])}.")
            return compact, A
        if init != "random":
            raise ValueError(
                f"solver='sgd' with decomposition='utt' takes init='random' or a "
                f"([C, α, ω], factors) start; got {init!r}. SVD inits are computed "
                f"beforehand by sgd_init.mu_init_for_sgd."
            )

        R, rho = self.rank[0], self.tt_rank
        rng = np.random.RandomState(int(random_state))
        A = _to_tensor(rng.random_sample((self.shape[0], R)), dtype, device)
        A = A.abs().clamp_min(1e-30)
        A = A / A.sum(dim=0, keepdim=True).clamp_min(self.eps)
        rng = np.random.RandomState(int(random_state))
        C = _to_tensor(rng.random_sample((rho, R, rho)) + 0.01, dtype, device)
        alpha = _to_tensor(rng.uniform(0.9, 1.1, rho), dtype, device)
        omega = _to_tensor(rng.uniform(0.9, 1.1, rho), dtype, device)

        C64, a64, o64, A64 = _gauge(C.double(), alpha.double(), omega.double(), A.double(), n)
        if sum_x is not None:
            total = float(_sum_all(C64, a64, o64, A64, n))
            if not (math.isfinite(total) and total > 0.0):
                raise FloatingPointError(f"UTT init chain sum is {total!r}; cannot rescale. "
                                         f"Check rank/tt_rank/shape.")
            c = math.sqrt(float(sum_x) / total)
            a64, o64 = a64 * c, o64 * c
        return [C64.to(dtype), a64.to(dtype), o64.to(dtype)], A64.to(dtype)

    # --- non-negative views -------------------------------------------------
    def nonneg_views(self) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        p = self.parametrization
        compact = [view_from_raw(r, p, self.eps) for r in self._core_raw]
        A = view_from_raw(self._factor_raw["0"], p, self.eps)
        return compact, [A] * self.order

    # --- forward pieces -----------------------------------------------------
    def predict_from(self, core, factors, indices: torch.Tensor) -> torch.Tensor:
        A = factors[0]
        rows = [A[indices[m]] for m in range(self.order)]
        return chain_values(expand_uniform(core, self.order), rows, torch)

    def total_sum_from(self, core, factors) -> torch.Tensor:
        C, alpha, omega = core
        return _sum_all(C, alpha, omega, factors[0], self.order).to(C.dtype)

    def total_sq_norm_from(self, core, factors) -> torch.Tensor:
        A = factors[0]
        G = (A.T @ A).to(torch.float64)
        cores64 = expand_uniform([x.to(torch.float64) for x in core], self.order)
        return sq_norm(cores64, [G] * self.order, torch).to(A.dtype)

    # --- gauge ----------------------------------------------------------------
    @torch.no_grad()
    def regauge_(self):
        """Exact gauge reset (module docstring). x̂ is unchanged."""
        (C, alpha, omega), factors = self.nonneg_views()
        new = _gauge(C.double(), alpha.double(), omega.double(), factors[0].double(),
                     self.order)
        p = self.parametrization
        for raw, x in zip(self._core_raw, new[:3]):
            raw.copy_(raw_from_view(x.to(raw.dtype), p, self.eps))
        raw = self._factor_raw["0"]
        raw.copy_(raw_from_view(new[3].to(raw.dtype), p, self.eps))

    def materialize(self) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """CPU numpy ``([C, α, ω], factors)`` — the MU utt artifact format."""
        with torch.no_grad():
            compact, factors = self.nonneg_views()
            A = factors[0].detach().cpu().numpy()
            return [x.detach().cpu().numpy() for x in compact], [A.copy() for _ in factors]


__all__ = ["SGDUTTModel"]
