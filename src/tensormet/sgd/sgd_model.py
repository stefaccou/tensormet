"""
sgd_model.py — what every SGD model shares: the non-negativity
parametrizations, rank / shared-factor parsing, and ``SGDModelBase``, the
surface the trainers use (Tucker, TT and utt models all subclass it).
"""
from __future__ import annotations

from typing import List, Sequence, Tuple, Union

import torch
import torch.nn.functional as F

from tensormet.utils import resolve_shared_factors

_EPS = 1e-12

# Non-negativity parametrizations: raw parameter θ -> non-negative view.
#   softplus  softplus(θ) + eps
#   exp       exp(θ): a rescale of the view is an additive shift of θ, which is
#             what lets the TT/utt models re-gauge without spoiling Adam's state
PARAMETRIZATIONS = ("softplus", "exp")


def view_from_raw(raw: torch.Tensor, parametrization: str, eps: float) -> torch.Tensor:
    if parametrization == "softplus":
        return F.softplus(raw) + eps
    return torch.exp(raw)


def raw_from_view(view: torch.Tensor, parametrization: str, eps: float) -> torch.Tensor:
    """Inverse of ``view_from_raw``, exact down to the dtype's ``tiny`` (no
    1e-6 clamp: ℓ1-normalized factors have entries around 1/V).
    softplus⁻¹(y) = y + log(-expm1(-y)), the overflow-free form of log(expm1(y))."""
    tiny = torch.finfo(view.dtype).tiny
    if parametrization == "softplus":
        y = (view - eps).clamp_min(tiny)
        return y + torch.log(-torch.expm1(-y))
    return torch.log(view.clamp_min(tiny))


def _as_rank_tuple(rank: Union[int, Sequence[int]], order: int) -> Tuple[int, ...]:
    if isinstance(rank, int):
        return (rank,) * order
    rank = tuple(int(r) for r in rank)
    if len(rank) != order:
        raise ValueError(f"rank has {len(rank)} entries for an order-{order} tensor.")
    return rank


def _parse_shared_factors(shared_factors, order: int) -> List[int]:
    """owner[mode]: the lowest mode each mode's factor aliases to. Accepts the
    main package's specs: None/False, True → {(1, 2)}, "all", or (i, j) pairs."""
    pairs = resolve_shared_factors(shared_factors, order) or ()

    owner = list(range(order))

    def find(i):
        while owner[i] != i:
            owner[i] = owner[owner[i]]
            i = owner[i]
        return i

    for a, b in pairs:
        ra, rb = find(a), find(b)
        lo, hi = min(ra, rb), max(ra, rb)
        owner[hi] = lo
    return [find(i) for i in range(order)]


class SGDModelBase(torch.nn.Module):
    """Common surface of the SGD models.

    Subclasses call ``_setup`` first and implement ``nonneg_views``,
    ``predict_from``, ``total_sum_from``, ``total_sq_norm_from`` and
    ``materialize``; the chain models (TT, utt) also override ``regauge_``.
    """

    def _setup(self, shape, rank, parametrization, shared_factors, eps, device) -> torch.device:
        """Set shape/order/rank/parametrization/eps/owner; return the device."""
        self.shape = tuple(int(s) for s in shape)
        self.order = len(self.shape)
        self.rank = _as_rank_tuple(rank, self.order)
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
        if device is not None:
            return torch.device(device)
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    def predict_entries(self, indices: torch.Tensor) -> torch.Tensor:
        """x̂ at ``indices`` (an ``(order, B)`` long tensor)."""
        return self.predict_from(*self.nonneg_views(), indices)

    def total_sum(self) -> torch.Tensor:
        return self.total_sum_from(*self.nonneg_views())

    def total_sq_norm(self) -> torch.Tensor:
        return self.total_sq_norm_from(*self.nonneg_views())

    def regauge_(self):
        """No-op; only the chain models (TT, utt) have a gauge to reset."""


__all__ = [
    "PARAMETRIZATIONS",
    "SGDModelBase",
    "raw_from_view",
    "view_from_raw",
]
