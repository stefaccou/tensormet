"""
sgd_tucker.py — the SGD/Adam non-negative Tucker model.

Model: X̂ = G ×_1 A^(1) ×_2 … ×_N A^(N), kept non-negative by a
parametrization (``sgd_model.PARAMETRIZATIONS``). The TT and utt models
(``sgd_tt.py``, ``uniform_tt/utt_sgd.py``) have the same surface;
``sgd_trainer.make_sgd_model`` picks one. The batcher, objectives, step and
error shared by all models are in ``sgd_step.py``. See sgd/README.md.
"""
from __future__ import annotations

import math
from typing import List, Optional, Sequence, Tuple, Union

import numpy as np
import torch

from tensormet.sgd.sgd_model import (
    _EPS,
    SGDModelBase,
    raw_from_view,
    view_from_raw,
)
from tensormet.utils import einsum_letters


# ---------------------------------------------------------------------------
# the two-group contraction
# ---------------------------------------------------------------------------

def _contraction_plan(rank: Sequence[int]) -> Tuple[int, bool, int]:
    """``(split, gemm_left, width)`` for the two-group entry contraction.

    The modes are split at ``split`` into a left and a right group; each
    group's gathered rows form a row-wise Khatri-Rao product, and the two meet
    the core reshaped to ``(prod(rank[:split]), prod(rank[split:]))``, one via a
    GEMM (``gemm_left`` says which) and the other elementwise. ``width`` is the
    largest intermediate per entry:

        width(s, gemm_left)  = max(kr_left, prod(rank[s:]))   if gemm_left
                             = max(kr_right, prod(rank[:s]))  otherwise

    minimised over both (~sqrt(prod(rank)) for uniform ranks). Flops are
    ``B · prod(rank)`` under every plan: this only trades memory.
    """
    r = [int(x) for x in rank]
    n = len(r)
    if n < 2:
        # A single mode contracts as one dot product; no intermediate beyond
        # the gathered rows.
        return 0, True, 1
    best = None
    for s in range(1, n):
        left, right = math.prod(r[:s]), math.prod(r[s:])
        kr_left = left if s > 1 else 0
        kr_right = right if n - s > 1 else 0
        for gemm_left, w in ((True, max(kr_left, right)),
                             (False, max(kr_right, left))):
            if best is None or w < best[2]:
                best = (s, gemm_left, w)
    s, gemm_left, w = best
    return s, gemm_left, max(int(w), 1)

def _row_khatri_rao(rows: Sequence[torch.Tensor]) -> torch.Tensor:
    """Row-wise Khatri-Rao of ``(B, R_i)`` matrices → ``(B, prod R_i)``, in the
    C order of ``core.reshape``. A single input is returned untouched."""
    out = rows[0]
    for nxt in rows[1:]:
        out = (out.unsqueeze(2) * nxt.unsqueeze(1)).reshape(out.shape[0], -1)
    return out


# ---------------------------------------------------------------------------
# model
# ---------------------------------------------------------------------------

class SGDTuckerModel(SGDModelBase):
    """Non-negative Tucker model with entry-wise prediction and the two exact
    whole-tensor statistics the full objectives need."""

    def __init__(
        self,
        shape: Sequence[int],
        rank: Union[int, Sequence[int]],
        parametrization: str = "softplus",
        shared_factors=None,
        init: Union[str, Tuple] = "random",
        random_state: int = 0,
        device: Optional[Union[str, torch.device]] = None,
        dtype: torch.dtype = torch.float32,
        eps: float = _EPS,
    ):
        super().__init__()
        device = self._setup(shape, rank, parametrization, shared_factors, eps, device)
        if self.order > 25:
            raise ValueError("einsum-letter scheme supports order <= 25.")

        # --- init: sparse_ops.initialize_nonnegative_tucker's 'random', or a
        # (core, factors) warm start, e.g. from an MU model ---
        if init == "random":
            rng = np.random.default_rng(int(random_state))
            core0 = torch.as_tensor(
                rng.random(self.rank) + 0.01, dtype=dtype, device=device
            )
            factors0 = [
                torch.as_tensor(
                    rng.random((self.shape[m], self.rank[m])) + 0.01,
                    dtype=dtype, device=device,
                )
                for m in range(self.order)
            ]
        else:
            core0, factors0 = init
            core0 = torch.as_tensor(np.asarray(core0), dtype=dtype, device=device).abs().clamp_min(self.eps)
            factors0 = [
                torch.as_tensor(np.asarray(f), dtype=dtype, device=device).abs().clamp_min(self.eps)
                for f in factors0
            ]

        self._core_raw = torch.nn.Parameter(raw_from_view(core0, parametrization, self.eps))
        # Only owner modes hold a Parameter; aliased modes read the owner's.
        self._factor_raw = torch.nn.ParameterDict({
            str(m): torch.nn.Parameter(raw_from_view(factors0[m], parametrization, self.eps))
            for m in range(self.order) if self.owner[m] == m
        })

        # --- contraction plan for predict_from, chosen once ---
        self._split, self._gemm_left, self._width = _contraction_plan(self.rank)
        self._left_width = math.prod(self.rank[:self._split]) if self._split else 0
        self._right_width = math.prod(self.rank[self._split:]) if self._split else 0
        # Largest forward intermediate per entry; sizes micro-batches and eval chunks.
        self.entry_width = self._width

        # --- einsum equations, built once ---
        lo = einsum_letters(self.order)                  # core modes, lowercase
        hi = [c.upper() for c in lo]                     # primed copy for Grams
        core_str = "".join(lo)
        self._eq_sum = f"{core_str}," + ",".join(lo) + "->"
        self._eq_sqnorm = (
            f"{core_str},"
            + ",".join(f"{a}{b}" for a, b in zip(lo, hi))
            + f",{''.join(hi)}->"
        )

    # --- non-negative views -------------------------------------------------
    @property
    def core(self) -> torch.Tensor:
        return view_from_raw(self._core_raw, self.parametrization, self.eps)

    def factor(self, mode: int) -> torch.Tensor:
        raw = self._factor_raw[str(self.owner[mode])]
        return view_from_raw(raw, self.parametrization, self.eps)

    @property
    def factors(self) -> List[torch.Tensor]:
        return self.nonneg_views()[1]

    def nonneg_views(self) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """``(core, factors)`` with the parametrization applied once per raw
        parameter. Tied modes share one factor tensor object. Callers needing
        the views more than once in a step take them from here: each read of
        ``.core`` is a pass over the whole core."""
        core = self.core
        by_owner = {o: self.factor(o) for o in sorted(set(self.owner))}
        return core, [by_owner[self.owner[m]] for m in range(self.order)]

    # --- forward pieces -----------------------------------------------------
    def predict_from(
        self,
        core: torch.Tensor,
        factors: Sequence[torch.Tensor],
        indices: torch.Tensor,
    ) -> torch.Tensor:
        """x̂ at ``indices`` (an ``(order, B)`` long tensor) from explicit
        non-negative views; see ``_contraction_plan``."""
        rows = [factors[m][indices[m]] for m in range(self.order)]  # (B, R_m)
        if self.order == 1:
            return rows[0] @ core
        s = self._split
        left = _row_khatri_rao(rows[:s])                  # (B, prod rank[:s])
        right = _row_khatri_rao(rows[s:])                 # (B, prod rank[s:])
        mat = core.reshape(self._left_width, self._right_width)
        # Both orientations compute sum_ij left[z,i] mat[i,j] right[z,j]; they
        # differ only in which group's width the GEMM output carries.
        if self._gemm_left:
            return ((left @ mat) * right).sum(dim=1)
        return ((right @ mat.T) * left).sum(dim=1)

    def total_sum_from(self, core, factors) -> torch.Tensor:
        """sum(X̂) over ALL entries — the exact KL zero-entry term."""
        col_sums = [factors[m].sum(dim=0) for m in range(self.order)]
        return torch.einsum(self._eq_sum, core, *col_sums)

    def total_sq_norm_from(self, core, factors) -> torch.Tensor:
        """‖X̂‖² over ALL entries — the exact Frobenius zero-entry term."""
        grams = [f.T @ f for f in factors]
        return torch.einsum(self._eq_sqnorm, core, *grams, core)

    def materialize(self) -> Tuple[np.ndarray, List[np.ndarray]]:
        """Detached CPU numpy (core, factors) — TuckerDecomposition-ready."""
        with torch.no_grad():
            core = self.core.detach().cpu().numpy()
            factors = [self.factor(m).detach().cpu().numpy() for m in range(self.order)]
        return core, factors


