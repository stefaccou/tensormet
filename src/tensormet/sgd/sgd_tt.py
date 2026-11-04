"""
sgd_tt.py — SGD/Adam model for the Tucker-TT hybrid (``--decomposition tt``).

Same surface as ``SGDTuckerModel`` (nonneg_views / predict_from /
total_sum_from / total_sq_norm_from / project_ / regauge_ / materialize /
entry_width), so ``GradStepper``, both trainers, the batcher and the
collectives run on it unchanged. The model is the one in
``tt_hybrid/tt_chain.py``:

    X̂[i_0 … i_{N-1}] = Σ_r C_0[:, r_0, :] ⋯ C_{N-1}[:, r_{N-1}, :] · Π_n A_n[i_n, r_n]

Every forward quantity is a tt_chain contraction run on torch, so autograd
supplies the gradient the MU kernels build by hand. Per entry that costs
about N·R·ρ² flops, against prod(R) for the dense Tucker core.

Zero-entry terms (exact, closed form, fp64):
  KL  Σ_all x̂ : the chain on factor column sums
  FR  ‖X̂‖²    : the chain of X̂ against itself, joined by factor Grams (sq_norm)

Gauge. x̂ does not change when (a) factor column r is scaled and C_k[:, r, :]
divided by the same amount, (b) bond b is scaled up in C_{k-1} and down in C_k,
(c) the cores are scaled by numbers whose product is 1. Plain gradient descent
on log-parameters never moves along these directions; Adam's per-coordinate
scaling does, and the chain product then drifts out of fp32 range.
``regauge_()`` resets the model to one fixed gauge (ℓ1-normalized factor
columns as in MU, balanced bonds, equal core masses) without changing x̂. With
``parametrization="exp"`` that reset is an additive shift of the raw
parameters and the log-space gradient is gauge-invariant, so Adam's moments
stay valid; with softplus they are only approximately right after it.
"""
from __future__ import annotations

from typing import List, Optional, Sequence, Tuple, Union

import numpy as np
import torch

from tensormet.sgd.sgd_tucker import (
    _EPS, PARAMETRIZATIONS, _as_rank_tuple, _parse_shared_factors,
    raw_from_view, view_from_raw,
)
from tensormet.tt_hybrid.tt_chain import chain_values, core_shapes


# ---------------------------------------------------------------------------
# chain helpers (torch; fp64 for the whole-tensor sums)
# ---------------------------------------------------------------------------

def sq_norm(tt_cores, grams, xp):
    """‖X̂‖² over ALL entries, without forming any R^N object.

    E (ρ_k × ρ_k) carries the chain of X̂ against a second copy of itself; at
    each site the two copies meet through the factor Gram G_k = A_kᵀA_k:

        E'[b, d] = Σ E[a, c] · C_k[a, r, b] · G_k[r, s] · C_k[c, s, d]

    Cost per site: 2ρ³R + ρ²R². Two-operand einsums only (numpy/cupy/torch).
    """
    E = xp.ones_like(tt_cores[0][:1, 0, :1])                  # (1, 1)
    for C, G in zip(tt_cores, grams):
        T = xp.einsum("ac,arb->crb", E, C)
        T = xp.einsum("rs,crb->csb", G, T)
        E = xp.einsum("csb,csd->bd", T, C)
    return E[0, 0]


def chain_entry_width(shapes, ranks) -> int:
    """Per-entry elements a batch holds: the site matrices, the gathered factor
    rows and the two environment stacks (tt_ops.estimate_batch_nnz_tt)."""
    return int(sum(a * b for a, _r, b in shapes) + sum(int(r) for r in ranks)
               + 2 * sum(a for a, _r, _b in shapes) + 2)


def _to_tensor(x, dtype, device) -> torch.Tensor:
    if isinstance(x, torch.Tensor):
        return x.detach().to(device=device, dtype=dtype)
    return torch.as_tensor(np.asarray(x), dtype=dtype, device=device)


# ---------------------------------------------------------------------------
# model
# ---------------------------------------------------------------------------

class SGDTTModel(torch.nn.Module):
    """Non-negative Tucker-TT model: Tucker factors, TT cores in place of the
    dense core. ``core`` views are the list of TT cores (ρ_k, R_k, ρ_{k+1})."""

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
        self.shape = tuple(int(s) for s in shape)
        self.order = len(self.shape)
        if self.order < 2:
            raise ValueError("decomposition='tt' needs order >= 2.")
        self.rank = _as_rank_tuple(rank, self.order)
        self.tt_rank = int(tt_rank)
        if parametrization not in PARAMETRIZATIONS:
            raise ValueError(f"parametrization must be one of {PARAMETRIZATIONS}.")
        self.parametrization = parametrization
        self.eps = float(eps)
        self.owner = _parse_shared_factors(shared_factors, self.order)
        for m, o in enumerate(self.owner):
            if (self.shape[m], self.rank[m]) != (self.shape[o], self.rank[o]):
                raise ValueError(
                    f"shared modes {o} and {m} differ in (dim, rank): "
                    f"{(self.shape[o], self.rank[o])} vs {(self.shape[m], self.rank[m])}"
                )
        device = torch.device(device) if device is not None else torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.core_shapes = [tuple(int(d) for d in s)
                            for s in core_shapes(self.rank, self.tt_rank)]
        self.entry_width = chain_entry_width(self.core_shapes, self.rank)

        cores0, factors0 = self._initial_values(init, random_state, dtype, device, sum_x)
        p = self.parametrization
        self._core_raw = torch.nn.ParameterList(
            [torch.nn.Parameter(raw_from_view(C, p, self.eps)) for C in cores0]
        )
        # Only owner modes hold a Parameter; aliased modes read the owner's.
        self._factor_raw = torch.nn.ParameterDict({
            str(m): torch.nn.Parameter(raw_from_view(factors0[m], p, self.eps))
            for m in range(self.order) if self.owner[m] == m
        })

    # --- init ---------------------------------------------------------------
    def _initial_values(self, init, random_state, dtype, device, sum_x):
        """``random`` reproduces ``tt_ops.initialize_tucker_tt`` draw for draw
        (same seed → same start as the MU run); a ``(tt_cores, factors)`` tuple
        is a warm start, e.g. an MU TT model artifact."""
        if isinstance(init, (tuple, list)) and len(init) == 2:
            cores, factors = init
            cores = [_to_tensor(C, dtype, device).abs().clamp_min(self.eps) for C in cores]
            factors = [_to_tensor(f, dtype, device).abs().clamp_min(self.eps) for f in factors]
            got = [tuple(C.shape) for C in cores]
            if got != self.core_shapes:
                raise ValueError(
                    f"warm start TT cores have shapes {got}; expected {self.core_shapes} "
                    f"for rank={self.rank}, tt_rank={self.tt_rank}."
                )
            want_f = [(self.shape[m], self.rank[m]) for m in range(self.order)]
            if [tuple(f.shape) for f in factors] != want_f:
                raise ValueError(
                    f"warm start factors have shapes {[tuple(f.shape) for f in factors]}; "
                    f"expected {want_f}."
                )
            return cores, factors
        if init != "random":
            raise ValueError(
                f"solver='sgd' with decomposition='tt' takes init='random' or a "
                f"(tt_cores, factors) start; got {init!r}. SVD inits are computed "
                f"beforehand by sgd_init.mu_init_for_sgd."
            )

        rng = np.random.RandomState(int(random_state))
        factors = [_to_tensor(rng.random_sample((self.shape[m], self.rank[m])), dtype, device)
                   for m in range(self.order)]
        factors = [f.abs().clamp_min(1e-30) for f in factors]
        factors = [f / f.sum(dim=0, keepdim=True).clamp_min(self.eps) for f in factors]
        # The loop ties shared modes to the group's first mode, which is the owner.
        factors = [factors[self.owner[m]] for m in range(self.order)]
        cores = [_to_tensor(rng.random_sample(s) + 0.01, dtype, device)
                 for s in self.core_shapes]
        if sum_x is not None:
            total = float(self._sum_all(cores, factors))
            if not (np.isfinite(total) and total > 0.0):
                raise FloatingPointError(
                    f"TT init chain sum is {total!r}; cannot rescale. "
                    f"Check rank/tt_rank/shape.")
            # Σ_all x̂ = Σ x, spread evenly over the N cores (as in MU).
            scale = (float(sum_x) / total) ** (1.0 / self.order)
            cores = [C * scale for C in cores]
        return cores, factors

    # --- non-negative views -------------------------------------------------
    @property
    def device(self) -> torch.device:
        return self._core_raw[0].device

    def nonneg_views(self) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """``(tt_cores, factors)``, the parametrization applied once per raw
        parameter; tied modes share one factor tensor object."""
        p = self.parametrization
        cores = [view_from_raw(r, p, self.eps) for r in self._core_raw]
        by_owner = {o: view_from_raw(self._factor_raw[str(o)], p, self.eps)
                    for o in sorted(set(self.owner))}
        return cores, [by_owner[self.owner[m]] for m in range(self.order)]

    # --- forward pieces -----------------------------------------------------
    def predict_from(self, cores, factors, indices: torch.Tensor) -> torch.Tensor:
        """x̂ at ``indices`` (an ``(order, B)`` long tensor)."""
        rows = [factors[m][indices[m]] for m in range(self.order)]
        return chain_values(cores, rows, torch)

    def predict_entries(self, indices: torch.Tensor) -> torch.Tensor:
        return self.predict_from(*self.nonneg_views(), indices)

    @staticmethod
    def _sum_all(cores, factors) -> torch.Tensor:
        sums = [f.sum(dim=0, dtype=torch.float64)[None, :] for f in factors]
        return chain_values([C.to(torch.float64) for C in cores], sums, torch)[0]

    def total_sum_from(self, cores, factors) -> torch.Tensor:
        """Σ(X̂) over ALL entries — the exact KL zero-entry term."""
        return self._sum_all(cores, factors).to(cores[0].dtype)

    def total_sq_norm_from(self, cores, factors) -> torch.Tensor:
        """‖X̂‖² over ALL entries — the exact Frobenius zero-entry term."""
        grams, by_id = [], {}
        for f in factors:
            if id(f) not in by_id:  # tied modes: one Gram
                by_id[id(f)] = (f.T @ f).to(torch.float64)
            grams.append(by_id[id(f)])
        cores64 = [C.to(torch.float64) for C in cores]
        return sq_norm(cores64, grams, torch).to(cores[0].dtype)

    def total_sum(self) -> torch.Tensor:
        return self.total_sum_from(*self.nonneg_views())

    def total_sq_norm(self) -> torch.Tensor:
        return self.total_sq_norm_from(*self.nonneg_views())

    # --- after-step maintenance --------------------------------------------
    def project_(self):
        """'clamp' parametrization only: project onto [eps, ∞) after a step."""
        if self.parametrization == "clamp":
            with torch.no_grad():
                for p in self.parameters():
                    p.clamp_(min=self.eps)

    @torch.no_grad()
    def regauge_(self, bond_sweeps: int = 3):
        """Move to the fixed gauge (see module docstring). x̂ is unchanged."""
        cores, factors = self.nonneg_views()
        cores = [C.to(torch.float64) for C in cores]
        owners = sorted(set(self.owner))
        fac = {o: factors[o].to(torch.float64) for o in owners}
        tiny = torch.finfo(torch.float64).tiny

        # (a) ℓ1 factor columns; each column's scale moves into the R leg of
        # every core whose mode uses that factor.
        for o in owners:
            s = fac[o].sum(dim=0).clamp_min(tiny)
            fac[o] = fac[o] / s
            for m in range(self.order):
                if self.owner[m] == o:
                    cores[m] = cores[m] * s[None, :, None]

        # (b) bonds: equal ℓ1 mass on both sides of every bond index. Each
        # sweep is exact; repeating it brings all bonds close to balanced.
        for _ in range(int(bond_sweeps)):
            for k in range(1, self.order):
                left = cores[k - 1].sum(dim=(0, 1)).clamp_min(tiny)
                right = cores[k].sum(dim=(1, 2)).clamp_min(tiny)
                d = (right / left).sqrt()
                cores[k - 1] = cores[k - 1] * d[None, None, :]
                cores[k] = cores[k] / d[:, None, None]

        # (c) equal total mass per core (the product of the scales is 1).
        log_mass = torch.stack([C.sum().clamp_min(tiny).log() for C in cores])
        shift = log_mass - log_mass.mean()
        cores = [C * torch.exp(-s) for C, s in zip(cores, shift)]

        p = self.parametrization
        for raw, C in zip(self._core_raw, cores):
            raw.copy_(raw_from_view(C.to(raw.dtype), p, self.eps))
        for o in owners:
            raw = self._factor_raw[str(o)]
            raw.copy_(raw_from_view(fac[o].to(raw.dtype), p, self.eps))

    def materialize(self) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """Detached CPU numpy ``(tt_cores, factors)`` — the MU TT artifact format."""
        with torch.no_grad():
            cores, factors = self.nonneg_views()
            return ([C.detach().cpu().numpy() for C in cores],
                    [f.detach().cpu().numpy() for f in factors])


__all__ = ["SGDTTModel", "chain_entry_width", "sq_norm"]
