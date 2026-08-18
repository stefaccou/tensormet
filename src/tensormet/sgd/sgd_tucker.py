"""
sgd_tucker.py — EXPERIMENTAL: SGD/Adam-based non-negative Tucker decomposition.

Stochastic-gradient alternative to the multiplicative-update (MU) loop in
``tucker_tensor.non_negative_tucker_with_similarity``. Pure PyTorch: no CuPy,
no pytensorlab, no largedim/sharded kernel machinery.

Per-step cost: O(batch) for ``objective="masked"``. The default
``objective="full"`` additionally pays the exact zero-entry term every step —
O(Σ_m I_m·R_m) for KL (column sums) and O(Σ_m I_m·R_m²) for FR (Grams) — which
dominates a small batch at production mode dimensions. Both are independent of
nnz; the one O(nnz) piece is the exact-error pass on log steps.

Per-step *memory* is set by ``predict_entries``, which contracts the modes in
two groups rather than one at a time (see ``_contraction_plan``): the gathered
rows of each group are combined into a row-wise Khatri-Rao product and the two
groups meet in a single GEMM against the reshaped core. Flops are unchanged —
``batch × prod(rank)`` is irreducible for a dense core — but the largest
intermediate drops from ``batch × prod(rank)/max(rank)`` to
``batch × ~sqrt(prod(rank))``: 16 GB → 160 MB at order 4 / rank 100 / B=4096,
and the forward becomes three kernels instead of an N-operand einsum
decomposition. ``GradStepper`` can still split a step into micro-batches with
gradients accumulating — exact, since the sampled loss is a sum over entries —
but with the two-group plan that only binds past order 4, and
``resolve_micro_batch`` refuses an over-large explicit setting with a message
naming the knob rather than letting CUDA OOM.

Model
-----
X̂ = G ×_1 A^(1) ×_2 … ×_N A^(N), with non-negativity enforced through a
parametrization (default ``softplus``: A = softplus(θ) + eps, so positivity is
structural and gradients never die at the boundary; ``clamp`` stores the
factors directly and projects onto [eps, ∞) after each optimizer step, which
is closer in spirit to MU's clip but can zero-lock).

Objectives (matching the MU path's divergences and error normalizations)
------------------------------------------------------------------------
The key trick that makes *full* (non-masked) objectives SGD-able without
negative sampling: the zero-entry contribution is analytic for a Tucker model.

  KL  full :  Σ_nz [x·log(x/x̂) − x]                 (sampled, rescaled by 1/p)
              + sum(X̂)                              (EXACT: core ⨯ column sums)
  KL  mask :  Σ_nz [x·log(x/x̂) − x + x̂]            (sampled, rescaled)
  FR  full :  Σ_nz [(x − x̂)² − x̂²]                  (sampled, rescaled)
              + ‖X̂‖²                                (EXACT: core ⨯ factor Grams)
  FR  mask :  Σ_nz (x − x̂)²                          (sampled, rescaled)

Reported errors use the same normalization as ``distance.kl_compute_errors``
(KL / Σx) and ``distance.fr_compute_errors`` (‖X−X̂‖ / ‖X‖) so convergence
curves are directly comparable with the MU baseline.

Sampling & reproducibility
--------------------------
``EntryBatcher`` mirrors ``stochastic_sparse.CooSubsampler``: one host-side
seeded permutation of the NNZ at construction, then contiguous rotating
windows — every batch is a pure function of (seed, step), so runs are
deterministic given a seed and resume replays identical batches. Note that
unlike MU, the backward pass accumulates gradients into factor rows via
scatter-adds that are atomicAdd-nondeterministic on CUDA by default; pass
``deterministic=True`` to trade speed for bitwise reproducibility.

Production integration lives one directory up the import path:
``--solver sgd`` routes ``non_negative_tucker_with_similarity`` through
``sgd_trainer.SGDTrainer`` (single GPU) / ``sharded_sgd.ShardedSGDTrainer``
(``--n_gpus > 1``); see experimental/SGD/README.md for the seams. This module
stays standalone for notebook use via ``sgd_non_negative_tucker``.

Usage (standalone)
------------------
    from tensormet.sgd.sgd_tucker import sgd_non_negative_tucker

    out = sgd_non_negative_tucker(
        sparse_tensor,            # torch sparse COO (or SparseTupleTensor with .tensor)
        rank=100,
        divergence="kl",          # or "fr"
        objective="full",         # or "masked"
        n_steps=20_000,
        batch_size=4096,
        lr=1e-2,
    )
    core, factors = out["tensor"]  # tensorly TuckerTensor payload, CPU numpy
    # → TuckerDecomposition(core, factors, vocab) for the eval stack.
"""
from __future__ import annotations

import math
import time
from typing import List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from tensorly import backend_context
from tensorly.tucker_tensor import TuckerTensor

from tensormet.utils import SparseCOOTensor, einsum_letters

_EPS = 1e-12


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _as_rank_tuple(rank: Union[int, Sequence[int]], order: int) -> Tuple[int, ...]:
    if isinstance(rank, int):
        return (rank,) * order
    rank = tuple(int(r) for r in rank)
    if len(rank) != order:
        raise ValueError(f"rank has {len(rank)} entries for an order-{order} tensor.")
    return rank


def _parse_shared_factors(shared_factors, order: int) -> List[int]:
    """Return owner[mode] — the canonical (lowest) mode each mode's factor
    aliases to. Accepts the same specs as the main package: None/False,
    True → {(1, 2)}, "all", or a set/sequence of (i, j) pairs."""
    if shared_factors == "all":
        pairs = [(i, j) for i in range(order) for j in range(i + 1, order)]
    elif shared_factors is True:
        pairs = [(1, 2)]
    elif not shared_factors:
        pairs = []
    else:
        pairs = list(shared_factors)

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


def _inv_softplus(y: torch.Tensor) -> torch.Tensor:
    # softplus⁻¹(y) = log(expm1(y)); clamp keeps the log finite for tiny inits.
    return torch.log(torch.expm1(y.clamp_min(1e-6)))


def _contraction_plan(rank: Sequence[int]) -> Tuple[int, bool, int]:
    """``(split, gemm_left, width)`` for the two-group entry contraction.

    ``predict_entries`` evaluates ``sum_{r...} G[r...] · A1[i1,r1] · … `` by
    splitting the modes at ``split`` into a left and a right group, forming the
    row-wise Khatri-Rao product of each group's gathered rows, and contracting
    the two against the core reshaped to ``(prod(rank[:split]),
    prod(rank[split:]))``. One group goes through a GEMM against that matrix and
    the other meets the result elementwise; ``gemm_left`` says which. ``width``
    is the largest resulting intermediate *per entry*, so a forward pass costs
    ``B · width`` elements.

    The intermediates are the two Khatri-Rao products — each allocated only when
    its group holds more than one mode, since a one-mode group *is* its gathered
    rows — and the GEMM output, which is the size of whichever group did *not*
    go through the GEMM. Hence

        width(s, gemm_left)  = max(kr_left, prod(rank[s:]))   if gemm_left
                             = max(kr_right, prod(rank[:s]))  otherwise

    minimised over both. For uniform ranks the answer is ``~sqrt(prod(rank))``
    instead of the ``prod(rank)/max(rank)`` that contracting the core against
    one row set at a time forces — 10^4 rather than 10^6 at order 4 / rank 100.

    The split must be contiguous in mode order (a non-contiguous one would need
    the core permuted, and a core-sized permute per step costs more than it
    saves), so the guarantee is: **never worse than contracting one mode at a
    time from either end**, i.e. ``min(prod(rank)/rank[0],
    prod(rank)/rank[-1])`` — those are the ``s=1`` and ``s=n-1`` plans. A rank
    tuple whose largest entry sits strictly in the *middle* (say ``(2, 9, 2)``)
    is the one case where permuting first would beat this; it does not arise
    from a scalar ``--rank``.

    Flops are ``B · prod(rank)`` under every plan (the first contraction that
    touches the core must touch all of it), so this is purely a memory and
    kernel-count choice.
    """
    r = [int(x) for x in rank]
    n = len(r)
    if n < 2:
        # A single mode contracts as one dot product; there is no intermediate
        # beyond the gathered rows, which every plan allocates anyway.
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


def _intermediate_width(rank: Sequence[int]) -> int:
    """Elements of the largest ``predict_entries`` intermediate *per entry*."""
    return _contraction_plan(rank)[2]


def _rank_derived_chunk(rank: Sequence[int], budget: int = 1 << 26) -> int:
    """Largest entry count whose forward intermediate fits ``budget`` elements.

    ``budget`` defaults to 2^26 elements (~256 MB fp32). Under the two-group
    contraction the width is ``~sqrt(prod(rank))``, so this returns the full
    batch for everything up to about order 4 / rank 200 and micro-batching
    becomes a no-op there. The 64 floor still exists for the regimes where the
    width genuinely cannot be brought down (order 5+), where it binds and the
    budget is exceeded on purpose — a chunk of 1 would be unusable, but it means
    very large orders want an explicit ``--sgd-micro-batch``.
    """
    return int(max(64, min(1 << 20, budget // _intermediate_width(rank))))


def _default_eval_chunk(rank: Sequence[int], budget: int = 1 << 26) -> int:
    """Entries per chunk in ``full_relative_error`` (see ``_rank_derived_chunk``)."""
    return _rank_derived_chunk(rank, budget)


def _row_khatri_rao(rows: Sequence[torch.Tensor]) -> torch.Tensor:
    """Row-wise Khatri-Rao of ``(B, R_i)`` matrices → ``(B, prod R_i)``.

    Row ``z`` of the result is the flattened outer product of row ``z`` of each
    input, so it indexes exactly like the corresponding flattened block of core
    rank axes (C order, matching ``core.reshape``). A single-element input is
    returned untouched, which is what makes a one-mode group free.
    """
    out = rows[0]
    for nxt in rows[1:]:
        out = (out.unsqueeze(2) * nxt.unsqueeze(1)).reshape(out.shape[0], -1)
    return out


def resolve_micro_batch(
    rank: Sequence[int],
    batch_size: int,
    micro_batch: Optional[int] = None,
    dtype: torch.dtype = torch.float32,
    budget: int = 1 << 26,
    ceiling_bytes: int = 2 << 30,
) -> int:
    """Entries per forward/backward inside one optimizer step.

    A step's sampled loss is a *sum* over its entries, so splitting the batch
    into micro-batches and letting gradients accumulate is mathematically
    identical to one big backward — purely a memory transformation. Since the
    two-group contraction landed this is rarely needed: at order 4 / rank 100
    the derived value is the whole batch. It still binds at order 5+, and an
    explicit setting remains available for tight-VRAM runs.

    ``micro_batch=None`` derives the value from the rank on the same budget
    ``_default_eval_chunk`` uses. An explicit value is honoured but still
    checked against ``ceiling_bytes``, so an over-large request fails with a
    message naming the knob instead of a bare CUDA OOM.
    """
    batch_size = int(batch_size)
    if batch_size < 1:
        raise ValueError(f"batch_size must be >= 1, got {batch_size}")
    inter = _intermediate_width(rank)
    itemsize = torch.empty((), dtype=dtype).element_size()

    if micro_batch is None:
        return min(batch_size, _rank_derived_chunk(rank, budget))

    mb = int(micro_batch)
    if mb < 1:
        raise ValueError(f"sgd_micro_batch must be >= 1, got {micro_batch}")
    mb = min(mb, batch_size)
    projected = inter * mb * itemsize
    if projected > ceiling_bytes:
        raise ValueError(
            f"An SGD forward pass at micro_batch={mb} and rank={tuple(rank)} would "
            f"allocate a {projected / 2**30:.1f} GiB intermediate "
            f"({mb} entries x prod(rank[:-1])={inter} x {itemsize} B), above the "
            f"{ceiling_bytes / 2**30:.1f} GiB ceiling — and autograd saves it for "
            f"the backward pass on top. Lower --sgd-micro-batch (the rank-derived "
            f"default here is {min(batch_size, _rank_derived_chunk(rank, budget))}) "
            f"or lower --rank."
        )
    return mb


# ---------------------------------------------------------------------------
# deterministic rotating-window batcher (CooSubsampler's contract, torch-side)
# ---------------------------------------------------------------------------

class EntryBatcher:
    """One-time seeded shuffle of the NNZ; ``batch(step)`` returns a contiguous
    rotating window of it (wrapping), so batches tile the data like an epoch
    and are a pure function of (seed, step)."""

    def __init__(self, nnz: int, batch_size: int, seed: int, device: torch.device):
        self.nnz = int(nnz)
        self.batch_size = min(int(batch_size), self.nnz)
        perm_np = np.random.default_rng(int(seed)).permutation(self.nnz)
        self._perm = torch.from_numpy(perm_np).to(device)
        self.device = self._perm.device
        self.dtype = self._perm.dtype

    def batch(self, step: int) -> torch.Tensor:
        start = (step * self.batch_size) % self.nnz
        end = start + self.batch_size
        if end <= self.nnz:
            return self._perm[start:end]
        return torch.cat([self._perm[start:], self._perm[: end - self.nnz]])

    def new_buffer(self) -> torch.Tensor:
        """A ``(batch_size,)`` buffer ``batch_into`` can write into.

        Zeroed, not ``empty``: the buffer is always filled by ``batch_into``
        before use, but reading it beforehand (calling ``GradStepper._body``
        directly, as a test might) would otherwise index the NNZ arrays with
        uninitialized int64 garbage and raise from deep inside the forward.
        Valid-but-wrong indices fail loudly at the assertion instead."""
        return torch.zeros(self.batch_size, dtype=self.dtype, device=self.device)

    def batch_into(self, step: int, out: torch.Tensor) -> torch.Tensor:
        """``batch(step)`` written into a caller-owned buffer.

        Same window, no allocation and no ``torch.cat`` on wrap-around. The
        CUDA-graph path needs the batch indices to live at a *fixed* address
        across steps, and this is how they get there: the window offset is a
        host-side value, so the copy must happen outside any captured region
        while everything downstream of ``out`` stays capturable.
        """
        start = (step * self.batch_size) % self.nnz
        end = start + self.batch_size
        if end <= self.nnz:
            out.copy_(self._perm[start:end])
        else:
            head = self.nnz - start
            out[:head].copy_(self._perm[start:])
            out[head:].copy_(self._perm[: end - self.nnz])
        return out


# ---------------------------------------------------------------------------
# model
# ---------------------------------------------------------------------------

class SGDTuckerModel(torch.nn.Module):
    """Non-negative Tucker model with entry-wise prediction and the two
    analytic whole-tensor statistics the full objectives need."""

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
        self.shape = tuple(int(s) for s in shape)
        self.order = len(self.shape)
        if self.order > 25:
            raise ValueError("einsum-letter scheme supports order <= 25.")
        self.rank = _as_rank_tuple(rank, self.order)
        if parametrization not in ("softplus", "clamp"):
            raise ValueError("parametrization must be 'softplus' or 'clamp'.")
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

        # --- init (mirrors sparse_ops.initialize_nonnegative_tucker 'random',
        # or accepts a (core, factors) warm start, e.g. from an MU checkpoint) ---
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

        to_raw = _inv_softplus if parametrization == "softplus" else (lambda t: t)
        self._core_raw = torch.nn.Parameter(to_raw(core0))
        # Only owner modes hold a Parameter; aliased modes read the owner's.
        self._factor_raw = torch.nn.ParameterDict({
            str(m): torch.nn.Parameter(to_raw(factors0[m]))
            for m in range(self.order) if self.owner[m] == m
        })

        # --- contraction plan for predict_entries, chosen once ---
        self._split, self._gemm_left, self._width = _contraction_plan(self.rank)
        self._left_width = math.prod(self.rank[:self._split]) if self._split else 0
        self._right_width = math.prod(self.rank[self._split:]) if self._split else 0

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
        if self.parametrization == "softplus":
            return F.softplus(self._core_raw) + self.eps
        return self._core_raw

    def factor(self, mode: int) -> torch.Tensor:
        raw = self._factor_raw[str(self.owner[mode])]
        if self.parametrization == "softplus":
            return F.softplus(raw) + self.eps
        return raw

    @property
    def factors(self) -> List[torch.Tensor]:
        return self.nonneg_views()[1]

    def nonneg_views(self) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """``(core, factors)`` with the parametrization applied exactly ONCE
        per distinct raw parameter.

        Every read of ``.core`` runs a softplus over the whole core — 4·10^8
        bytes of traffic at order 4 / rank 100 — so callers that need the views
        more than once in a step (a micro-batch loop, or a sampled term plus a
        zero-entry term) must take them from here and pass them down rather
        than reading the properties repeatedly. Aliased modes under
        ``shared_factors`` share one tensor *object*, so their softplus is
        evaluated once and the gather reads one buffer.
        """
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
        non-negative views — see ``_contraction_plan`` for the scheme."""
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

    def predict_entries(self, indices: torch.Tensor) -> torch.Tensor:
        """x̂ at the given entries. ``indices``: (order, B) long tensor."""
        core, factors = self.nonneg_views()
        return self.predict_from(core, factors, indices)

    def total_sum_from(self, core, factors) -> torch.Tensor:
        """sum(X̂) over ALL entries — the exact KL zero-entry term."""
        col_sums = [factors[m].sum(dim=0) for m in range(self.order)]
        return torch.einsum(self._eq_sum, core, *col_sums)

    def total_sq_norm_from(self, core, factors) -> torch.Tensor:
        """‖X̂‖² over ALL entries — the exact Frobenius zero-entry term."""
        grams = [f.T @ f for f in factors]
        return torch.einsum(self._eq_sqnorm, core, *grams, core)

    def total_sum(self) -> torch.Tensor:
        return self.total_sum_from(*self.nonneg_views())

    def total_sq_norm(self) -> torch.Tensor:
        return self.total_sq_norm_from(*self.nonneg_views())

    def project_(self):
        """'clamp' parametrization only: project onto [eps, ∞) after a step."""
        if self.parametrization == "clamp":
            with torch.no_grad():
                self._core_raw.clamp_(min=self.eps)
                for p in self._factor_raw.values():
                    p.clamp_(min=self.eps)

    def materialize(self) -> Tuple[np.ndarray, List[np.ndarray]]:
        """Detached CPU numpy (core, factors) — TuckerDecomposition-ready."""
        with torch.no_grad():
            core = self.core.detach().cpu().numpy()
            factors = [self.factor(m).detach().cpu().numpy() for m in range(self.order)]
        return core, factors


# ---------------------------------------------------------------------------
# objectives
# ---------------------------------------------------------------------------

def sampled_loss(x, x_hat, scale, divergence, masked, eps=_EPS):
    """The *sampled* (nnz) half of the objective, rescaled to full-tensor scale.

    A pure sum over the given entries, which is what makes micro-batching and
    NNZ sharding exact rather than approximate: splitting the entries and adding
    the partial losses reproduces the whole term. ``scale`` = nnz / batch.
    """
    if divergence == "kl":
        x_safe = x.clamp_min(eps)
        nz = x_safe * torch.log(x_safe / (x_hat + eps)) - x
        if masked:
            return scale * (nz + x_hat).sum()
        return scale * nz.sum()
    if divergence == "fr":
        sq = (x - x_hat) ** 2
        if masked:
            return scale * sq.sum()
        return scale * (sq - x_hat ** 2).sum()
    raise ValueError(f"divergence must be 'kl' or 'fr', got {divergence!r}")


def _distinct_views(core, factors) -> List[torch.Tensor]:
    """``[core] + factors`` in first-occurrence order, aliased modes collapsed.

    ``nonneg_views`` hands back one tensor object per distinct raw parameter, so
    identity is the right key here — this is the list whose gradients map
    one-to-one onto the model's trainable parameters."""
    out = [core]
    seen = {id(core)}
    for f in factors:
        if id(f) not in seen:
            seen.add(id(f))
            out.append(f)
    return out


def zero_entry_term(model, divergence, views=None):
    """The EXACT closed-form zero-entry half — a function of the parameters
    only, so it is added once per step no matter how the entries are split.

    ``views`` is an optional ``(core, factors)`` pair from
    ``SGDTuckerModel.nonneg_views``; passing it avoids re-running the
    parametrization when the caller has already materialized it."""
    core, factors = model.nonneg_views() if views is None else views
    if divergence == "kl":
        return model.total_sum_from(core, factors)
    if divergence == "fr":
        return model.total_sq_norm_from(core, factors)
    raise ValueError(f"divergence must be 'kl' or 'fr', got {divergence!r}")


def _batch_loss(model, x, x_hat, scale, divergence, masked, eps=_EPS, views=None):
    """Unbiased estimate of the total objective. ``scale`` = nnz / batch."""
    loss = sampled_loss(x, x_hat, scale, divergence, masked, eps=eps)
    if masked:
        return loss
    return loss + zero_entry_term(model, divergence, views=views)


# ---------------------------------------------------------------------------
# one step's gradient, shared by the single-GPU and sharded trainers
# ---------------------------------------------------------------------------

class GradStepper:
    """Everything needed to fill one model's gradients for one step, on one
    device — the unit both ``SGDTrainer`` and ``ShardedSGDTrainer`` build on.

    Three things live here rather than in the trainers:

    **A flat gradient buffer.** Every parameter's ``.grad`` is a *view* into one
    contiguous tensor, so the sharded trainer's cross-device reduction is a
    single collective over a single buffer (instead of one allocation per
    parameter per replica per step) and zeroing is one kernel. Nothing copies.
    The wiring means ``zero_grad(set_to_none=True)`` must never be called on
    these models — use :meth:`zero_grad_`.

    **Micro-batching.** The batch is split into chunks whose forward
    intermediate fits a memory budget, with gradients accumulating across them.
    Exact, not approximate (see ``sampled_loss``); the zero-entry term and the
    parametrization are each evaluated once per step, outside the loop, because
    they depend only on the parameters. Since the two-group contraction landed
    the split is usually empty — one chunk up to about order 4 / rank 200 — and
    ``_body`` takes a simpler path when that is the case.

    **Optional CUDA-graph capture.** The step body is fixed-shape, so it
    captures cleanly; the one dynamic input, the batch window offset, is handled
    by writing indices into a static buffer *outside* the capture. This was
    added when the order-3 step measured ~50x off its arithmetic roofline —
    Python dispatch, kernel launch and autograd-node overhead, not math. Two of
    the three causes of that kernel count are now gone (a step is one chunk, not
    62, and the forward is three kernels rather than an N-operand einsum
    decomposition), so re-measure before assuming the capture still pays: the
    number to beat is in ``0_tests/test_sgd_multigpu.ipynb`` §5.
    """

    def __init__(
        self,
        model: SGDTuckerModel,
        indices: torch.Tensor,
        values: torch.Tensor,
        batcher: EntryBatcher,
        *,
        scale: float,
        divergence: str,
        masked: bool,
        norm_const: float,
        include_zero_term: bool = True,
        micro_batch: Optional[int] = None,
        cuda_graph: bool = False,
        eps: float = _EPS,
    ):
        self.model = model
        self.indices = indices
        self.values = values
        self.batcher = batcher
        self.scale = float(scale)
        self.divergence = divergence
        self.masked = bool(masked)
        self.norm_const = float(norm_const)
        # A masked objective has no zero-entry term at all; a sharded trainer
        # additionally suppresses it on all but one device when gradients are
        # summed every step (adding it G times would count it G times).
        self.include_zero_term = bool(include_zero_term) and not self.masked
        self.eps = float(eps)
        self.device = values.device

        self.params = list(model.parameters())
        if not self.params:
            raise ValueError("model has no trainable parameters.")
        dtype = self.params[0].dtype
        numel = sum(p.numel() for p in self.params)
        self.flat_grad = torch.zeros(numel, device=self.device, dtype=dtype)
        offset = 0
        for p in self.params:
            p.grad = self.flat_grad[offset:offset + p.numel()].view_as(p)
            offset += p.numel()

        self.batch_size = int(batcher.batch_size)
        self.micro_batch = resolve_micro_batch(
            model.rank, self.batch_size, micro_batch, dtype=dtype
        )
        self._slices = [
            (lo, min(lo + self.micro_batch, self.batch_size))
            for lo in range(0, self.batch_size, self.micro_batch)
        ]
        self._sel = batcher.new_buffer()

        # Staging for the parametrization hoist (see ``_body``). Only allocated
        # when the step is actually split: these buffers are core-sized, and a
        # single-chunk step gets the same "one softplus per step" guarantee for
        # free by keeping everything on one graph.
        self._view_grads: Optional[List[torch.Tensor]] = None
        if len(self._slices) > 1:
            with torch.no_grad():
                self._view_grads = [
                    torch.zeros_like(t)
                    for t in _distinct_views(*model.nonneg_views())
                ]

        self._graph: Optional["torch.cuda.CUDAGraph"] = None
        if cuda_graph:
            self._capture()

    # --- gradient plumbing ---------------------------------------------------

    def zero_grad_(self) -> None:
        """Zero in place. NOT ``set_to_none``: the grads are views into
        ``flat_grad`` and detaching them would break the collective's buffer."""
        self.flat_grad.zero_()

    @property
    def param_numel(self) -> int:
        return int(self.flat_grad.numel())

    def new_param_buffer(self) -> torch.Tensor:
        """A buffer ``pack_params_`` fits, laid out like ``flat_grad``."""
        return torch.empty_like(self.flat_grad)

    @torch.no_grad()
    def pack_params_(self, out: torch.Tensor) -> torch.Tensor:
        """Parameters -> one contiguous vector (off the hot path: parameter
        averaging under ``sgd_sync_every`` and the replica-drift check)."""
        offset = 0
        for p in self.params:
            n = p.numel()
            out[offset:offset + n].copy_(p.detach().reshape(-1))
            offset += n
        return out

    @torch.no_grad()
    def unpack_params_(self, flat: torch.Tensor) -> None:
        offset = 0
        for p in self.params:
            n = p.numel()
            p.copy_(flat[offset:offset + n].view_as(p))
            offset += n

    # --- the step body -------------------------------------------------------

    def _body(self) -> None:
        """Fill ``flat_grad`` from the indices currently in ``self._sel``.

        The parametrization is evaluated exactly ONCE per step in both branches
        below. That is not a micro-optimization: ``model.core`` is a softplus
        over the whole core (400 MB at order 4 / rank 100), and reading it per
        chunk — which is what calling ``predict_entries`` in the loop does —
        costs more memory traffic than the arithmetic the step exists to do.

        Capturable: fixed shapes, no host syncs, no allocations that depend on
        a host-side value.
        """
        self.flat_grad.zero_()
        model = self.model
        core, factors = model.nonneg_views()

        if self._view_grads is None:
            # One chunk: the sampled term and the zero-entry term share a single
            # graph rooted at the raw parameters, so one backward does it all.
            lo, hi = self._slices[0]
            sel = self._sel[lo:hi]
            x_hat = model.predict_from(core, factors, self.indices[:, sel])
            loss = sampled_loss(self.values[sel], x_hat, self.scale,
                                self.divergence, self.masked, eps=self.eps)
            if self.include_zero_term:
                loss = loss + zero_entry_term(model, self.divergence,
                                              views=(core, factors))
            (loss / self.norm_const).backward()
            return

        # Several chunks. Each chunk's graph has to be freed before the next one
        # runs (that is the whole point of splitting), so the chunks cannot
        # share the parametrization's graph directly. Instead they differentiate
        # against DETACHED views with persistent, pre-zeroed gradient buffers —
        # so a chunk's backward stops at the views and accumulates there — and
        # one chain-rule pass at the end pushes the total back through softplus
        # into the raw parameters. Exact, and one softplus instead of one per
        # chunk.
        outs = _distinct_views(core, factors)
        leaves = []
        for src, gbuf in zip(outs, self._view_grads):
            gbuf.zero_()
            leaf = src.detach().requires_grad_(True)
            leaf.grad = gbuf
            leaves.append(leaf)
        leaf_of = {id(s): l for s, l in zip(outs, leaves)}
        core_l = leaf_of[id(core)]
        factors_l = [leaf_of[id(f)] for f in factors]

        for lo, hi in self._slices:
            sel = self._sel[lo:hi]
            x_hat = model.predict_from(core_l, factors_l, self.indices[:, sel])
            loss = sampled_loss(self.values[sel], x_hat, self.scale,
                                self.divergence, self.masked,
                                eps=self.eps) / self.norm_const
            loss.backward()
        if self.include_zero_term:
            z = zero_entry_term(model, self.divergence,
                                views=(core_l, factors_l)) / self.norm_const
            z.backward()
        torch.autograd.backward(outs, [l.grad for l in leaves])

    def compute_grads(self, step: int) -> None:
        """Gradients of step ``step``'s objective, into ``flat_grad``."""
        self.batcher.batch_into(step, self._sel)
        if self._graph is not None:
            self._graph.replay()
        else:
            self._body()

    # --- CUDA graph capture --------------------------------------------------

    def _capture(self, warmup: int = 3) -> None:
        if self.device.type != "cuda":
            raise ValueError("sgd_cuda_graph requires a CUDA device.")
        self.batcher.batch_into(0, self._sel)
        with torch.cuda.device(self.device):
            # Warm up on a side stream: lazy cubin loads, autograd node
            # allocation and cuBLAS workspace grabs must all have happened
            # before the capture, or they get recorded (or refuse to record).
            side = torch.cuda.Stream(device=self.device)
            side.wait_stream(torch.cuda.current_stream(self.device))
            with torch.cuda.stream(side):
                for _ in range(max(1, int(warmup))):
                    self._body()
            torch.cuda.current_stream(self.device).wait_stream(side)
            torch.cuda.synchronize(self.device)

            graph = torch.cuda.CUDAGraph()
            try:
                # thread_local, not the default global mode: the sharded
                # trainer captures one graph per device and other devices'
                # worker threads must not be able to invalidate the capture.
                with torch.cuda.graph(graph, capture_error_mode="thread_local"):
                    self._body()
            except RuntimeError as exc:
                raise RuntimeError(
                    "CUDA graph capture of the SGD step failed "
                    f"({exc}). Re-run with --sgd-cuda-graph false; the capture "
                    "is an optimization, not a requirement."
                ) from exc
            torch.cuda.synchronize(self.device)
        self.flat_grad.zero_()
        self._graph = graph


@torch.no_grad()
def make_eval_subset(
    indices: torch.Tensor,
    values: torch.Tensor,
    eval_sample: Optional[int],
    random_state: int,
) -> Tuple[torch.Tensor, torch.Tensor, float, Tuple[float, float]]:
    """``(indices, values, sample_scale, totals)`` for ``full_relative_error``.

    ``eval_sample=None`` (or ``>= nnz``) keeps the exact pass and returns the
    inputs untouched. Otherwise a fixed uniform subset of that many entries is
    drawn once — host-side, from a seed derived from ``random_state`` so it is
    reproducible and independent of the batcher's shuffle — and gathered into
    contiguous buffers the eval reuses every call.

    *Fixed* rather than redrawn per eval, deliberately: ``tol``/``patience``
    compare *successive* errors, and a subset that moved between evals would put
    sampling noise directly into that difference. Held fixed, the estimate is a
    consistent proxy whose step-to-step changes track the model's, which is what
    the early-stopping contract actually needs.

    Sampling is with replacement (an O(n) draw rather than an O(nnz)
    permutation); each draw is uniform, so the estimator is unbiased either way
    and duplicates are negligible at ``n << nnz``.

    ``totals`` is ``(Σx, Σx²)`` over the FULL tensor — the error denominators,
    computed once here instead of re-reduced over all nnz on every eval.
    """
    nnz = int(values.shape[0])
    totals = (float(values.sum()), float(values.pow(2).sum()))
    if not eval_sample or int(eval_sample) >= nnz:
        return indices, values, 1.0, totals
    n = max(1, int(eval_sample))
    sel_np = np.random.default_rng([int(random_state), 1]).integers(
        0, nnz, size=n, dtype=np.int64
    )
    sel = torch.from_numpy(sel_np).to(indices.device)
    return (indices[:, sel].contiguous(), values[sel].contiguous(),
            nnz / n, totals)


@torch.no_grad()
def full_relative_error(
    model: SGDTuckerModel,
    indices: torch.Tensor,
    values: torch.Tensor,
    divergence: str,
    masked: bool = False,
    chunk: Optional[int] = None,
    eps: float = _EPS,
    sample_scale: float = 1.0,
    totals: Optional[Tuple[float, float]] = None,
) -> float:
    """Objective over the passed nnz (chunked), normalized like distance.py:
    KL / Σx  (kl_compute_errors) or ‖X−X̂‖/‖X‖ (fr_compute_errors).

    With the defaults this is the exact error over all nnz. The cost is
    ``nnz · prod(rank)`` flops — the same per-entry cost as a training step, so
    an eval is worth ``nnz / (3 · batch_size)`` steps of compute and can easily
    dominate the run (MU gets away with it because one MU iteration already
    touches every nnz; a block of SGD steps does not).

    ``sample_scale`` and ``totals`` are the lever for that: pass a random SUBSET
    of the entries as ``indices``/``values``, ``sample_scale = nnz/len(subset)``
    and ``totals = (Σx, Σx²)`` over the FULL tensor. The nnz half of the
    objective is then rescaled to full-tensor scale and the zero-entry half
    stays exact, giving an unbiased estimate of the KL numerator / the squared
    FR numerator (the FR error itself, being a square root, is consistent
    rather than unbiased). The subset must be *random* — a contiguous slice of
    a coalesced tensor is sorted by index, not a sample.
    """
    if chunk is None:
        chunk = _default_eval_chunk(model.rank)
    core, factors = model.nonneg_views()
    nnz = values.shape[0]
    acc = values.new_zeros(())
    for s in range(0, nnz, chunk):
        idx = indices[:, s:s + chunk]
        x = values[s:s + chunk]
        x_hat = model.predict_from(core, factors, idx)
        if divergence == "kl":
            x_safe = x.clamp_min(eps)
            term = x_safe * torch.log(x_safe / (x_hat + eps)) - x
            if masked:
                term = term + x_hat
            acc = acc + term.sum()
        else:
            term = (x - x_hat) ** 2
            if not masked:
                term = term - x_hat ** 2
            acc = acc + term.sum()
    if sample_scale != 1.0:
        acc = acc * sample_scale

    sum_x, sum_x_sq = totals if totals is not None else (
        float(values.sum()), float(values.pow(2).sum())
    )
    if divergence == "kl":
        total = acc if masked else acc + model.total_sum_from(core, factors)
        return float(total / max(sum_x, eps))
    total = acc if masked else acc + model.total_sq_norm_from(core, factors)
    return float(total.clamp_min(0.0).sqrt() / max(sum_x_sq ** 0.5, eps))


# ---------------------------------------------------------------------------
# standalone training loop (notebook use; production goes through SGDTrainer)
# ---------------------------------------------------------------------------

def sgd_non_negative_tucker(
    sparse_tensor,
    rank: Union[int, Sequence[int]],
    divergence: str = "kl",
    objective: str = "full",
    n_steps: int = 20_000,
    batch_size: int = 4096,
    lr: float = 1e-2,
    optimizer: str = "adam",
    parametrization: str = "softplus",
    shared_factors=None,
    init: Union[str, Tuple] = "random",
    random_state: int = 0,
    device: Optional[Union[str, torch.device]] = None,
    dtype: torch.dtype = torch.float32,
    eval_every: int = 500,
    eval_chunk: Optional[int] = None,
    eval_sample: Optional[int] = None,
    tol: float = 1e-5,
    patience: int = 5,
    warmup_steps: int = 0,
    deterministic: bool = False,
    verbose: bool = True,
    return_model: bool = False,
):
    """Fit a non-negative Tucker model by minibatch SGD/Adam.

    Parameters mirror the MU loop where they overlap (divergence, objective,
    tol/patience early stopping on the exact relative error, random_state,
    shared_factors); the optimizer knobs (lr, batch_size, optimizer,
    parametrization) are the new surface SGD introduces.

    ``eval_chunk`` sizes the error pass; None derives it from the rank so the
    per-chunk intermediate stays bounded (see ``_default_eval_chunk``).
    ``eval_sample`` evaluates on a fixed random subset of that many nnz instead
    of all of them — the error pass costs the same per entry as a training step,
    so on a large tensor it otherwise dominates the run (see
    ``make_eval_subset`` / ``full_relative_error``). ``final_error`` is always
    computed exactly, whatever ``eval_sample`` says.

    ``sparse_tensor``: a torch sparse COO tensor, or anything with a
    ``.tensor`` attribute holding one (e.g. SparseTupleTensor with
    sparsity_type='torch' — BEFORE any CuPy conversion).

    Returns a dict shaped like the MU loop's ``return_errors='full'`` output:
    {"tensor": TuckerTensor((core, factors)) on CPU/numpy, "errors": [...],
     "iterations", "final_error", "decomp_seconds"}. ``final_error`` is always
    populated (computed at the end when no eval step fired). Pass
    ``return_model=True`` to also get the live module under ``"model"`` (pins
    device memory for as long as the caller holds the dict).
    """
    if objective not in ("full", "masked"):
        raise ValueError(f"objective must be 'full' or 'masked', got {objective!r}")
    masked = objective == "masked"

    # torch.use_deterministic_algorithms is process-global; restore on exit so
    # the flag never leaks into whatever else runs in this process.
    _det_prev = torch.are_deterministic_algorithms_enabled()
    if deterministic:
        torch.use_deterministic_algorithms(True)
    try:
        t = getattr(sparse_tensor, "tensor", sparse_tensor)
        if not (isinstance(t, (torch.Tensor, SparseCOOTensor)) and t.is_sparse):
            raise TypeError(
                "sgd_non_negative_tucker expects a torch sparse COO tensor "
                "(or an object with .tensor holding one)."
            )
        t = t.coalesce()

        model = SGDTuckerModel(
            shape=t.shape, rank=rank, parametrization=parametrization,
            shared_factors=shared_factors, init=init, random_state=random_state,
            device=device, dtype=dtype,
        )
        dev = model._core_raw.device

        indices = t.indices().to(dev)
        values = t.values().to(dev, dtype=dtype)
        if divergence == "kl" and bool((values < 0).any()):
            raise ValueError("KL divergence requires non-negative tensor values.")
        nnz = values.shape[0]
        if nnz == 0:
            raise ValueError("sparse tensor has no nonzero entries.")

        batcher = EntryBatcher(nnz, batch_size, seed=random_state, device=dev)
        scale = nnz / batcher.batch_size

        ev_idx, ev_val, ev_scale, ev_totals = make_eval_subset(
            indices, values, eval_sample, random_state
        )

        # Normalize the loss by the data scale so lr defaults transfer across
        # datasets (same constants the relative errors use).
        if divergence == "kl":
            norm_const = float(values.sum().clamp_min(_EPS))
        else:
            norm_const = float(values.pow(2).sum().clamp_min(_EPS))

        if optimizer == "adam":
            opt = torch.optim.Adam(model.parameters(), lr=lr)
        elif optimizer == "sgd":
            opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
        else:
            raise ValueError(f"optimizer must be 'adam' or 'sgd', got {optimizer!r}")

        errors: List[float] = []
        last_err: Optional[float] = None
        no_improve = 0
        start = time.time()
        step = -1

        for step in range(n_steps):
            sel = batcher.batch(step)
            idx = indices[:, sel]
            x = values[sel]

            opt.zero_grad(set_to_none=True)
            # One parametrization pass for the whole step: the sampled term and
            # the zero-entry term share these views instead of each running a
            # softplus over the core.
            views = model.nonneg_views()
            x_hat = model.predict_from(views[0], views[1], idx)
            loss = _batch_loss(model, x, x_hat, scale, divergence, masked,
                               views=views) / norm_const
            loss.backward()
            opt.step()
            model.project_()

            if eval_every > 0 and (step + 1) % eval_every == 0:
                rel = full_relative_error(model, ev_idx, ev_val, divergence,
                                          masked=masked, chunk=eval_chunk,
                                          sample_scale=ev_scale, totals=ev_totals)
                errors.append(rel)
                if verbose:
                    delta = f" (Δ={last_err - rel:+.3e})" if last_err is not None else ""
                    print(f"step {step + 1}: relative error={rel:.6f}{delta}")
                # Same patience contract as the MU loop's reconstruction check.
                if last_err is not None and step >= warmup_steps:
                    if abs(last_err - rel) < tol:
                        no_improve += 1
                        if no_improve >= patience:
                            if verbose:
                                print(
                                    f"Stopped after {no_improve} non-improving checks "
                                    f"(patience={patience}) at step {step + 1}."
                                )
                            break
                    else:
                        no_improve = 0
                last_err = rel

        if errors and ev_scale == 1.0:
            final_error = errors[-1]
        elif step >= 0:
            # Either no eval step fired (n_steps < eval_every) or the tracked
            # errors are subsampled estimates — either way the reported final
            # error is computed exactly, over all nnz, once.
            final_error = full_relative_error(model, indices, values, divergence,
                                              masked=masked, chunk=eval_chunk)
        else:
            final_error = None
        decomp_seconds = time.time() - start
        core, factors = model.materialize()
        # materialize() hands back CPU numpy, but the active tensorly backend is
        # typically "pytorch" here; TuckerTensor validates through the dispatched
        # tl.ndim, which on the torch backend calls .dim() and rejects numpy. Build
        # the container under the numpy backend to keep the CPU-numpy payload.
        with backend_context("numpy"):
            tucker = TuckerTensor((core, factors))
        out = {
            "tensor": tucker,
            "errors": errors,
            "iterations": step + 1,
            "final_error": final_error,
            "decomp_seconds": decomp_seconds,
        }
        if return_model:
            out["model"] = model
        return out
    finally:
        torch.use_deterministic_algorithms(_det_prev)
