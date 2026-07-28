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

Per-step *memory* is set by ``predict_entries``: no contraction order avoids a
``batch × prod(rank[:-1])`` intermediate (every gathered factor row carries the
batch axis), which is 16 GB at order 4 / rank 100 / B=4096. ``GradStepper``
therefore splits each step into micro-batches sized from the rank and lets
gradients accumulate — exact, since the sampled loss is a sum over entries —
and ``resolve_micro_batch`` refuses an over-large explicit setting with a
message naming the knob rather than letting CUDA OOM.

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
    from tensormet.experimental.SGD.sgd_tucker import sgd_non_negative_tucker

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

import time
from typing import List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from tensorly import backend_context
from tensorly.tucker_tensor import TuckerTensor

from tensormet.utils import einsum_letters

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


def _intermediate_width(rank: Sequence[int]) -> int:
    """Elements of the largest ``predict_entries`` intermediate *per entry*.

    ``einsum("abcd,za,zb,zc,zd->z")`` cannot avoid a ``B × prod(rank[:-1])``
    intermediate under any contraction order, because every gathered factor row
    carries the batch axis: contracting the core against the first N-1 row sets
    leaves one rank axis un-contracted alongside ``z``. So the memory of a
    forward pass is linear in the batch with this constant.
    """
    inter = 1
    for r in tuple(rank)[:-1]:
        inter *= int(r)
    return max(inter, 1)


def _rank_derived_chunk(rank: Sequence[int], budget: int = 1 << 26) -> int:
    """Largest entry count whose forward intermediate fits ``budget`` elements.

    ``budget`` defaults to 2^26 elements (~256 MB fp32). The floor is 64, not
    1024: at order 4 / rank 100 the constant is 10^6, so the budget asks for 67
    entries and a 1024 floor would silently produce a 4 GB intermediate —
    i.e. the floor used to defeat the budget it was supposed to enforce. 64 is
    still a floor, so past roughly order 4 / rank 200 it binds and the budget is
    exceeded again; that is deliberate (a chunk of 1 would be unusable), but it
    means very large ranks need an explicit ``--sgd-micro-batch``.
    """
    return int(max(64, min(1 << 20, budget // _intermediate_width(rank))))


def _default_eval_chunk(rank: Sequence[int], budget: int = 1 << 26) -> int:
    """Entries per chunk in ``full_relative_error`` (see ``_rank_derived_chunk``)."""
    return _rank_derived_chunk(rank, budget)


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
    identical to one big backward — purely a memory transformation. It is what
    makes ``--order 4 --rank 100`` runnable at all (see ``_intermediate_width``:
    B=4096 there is a 16 GB intermediate before autograd's saved tensors).

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
        """A ``(batch_size,)`` buffer ``batch_into`` can write into."""
        return torch.empty(self.batch_size, dtype=self.dtype, device=self.device)

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

        # --- einsum equations, built once ---
        lo = einsum_letters(self.order)                  # core modes, lowercase
        hi = [c.upper() for c in lo]                     # primed copy for Grams
        core_str = "".join(lo)
        self._eq_predict = f"{core_str}," + ",".join(f"z{c}" for c in lo) + "->z"
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
        return [self.factor(m) for m in range(self.order)]

    # --- forward pieces -----------------------------------------------------
    def predict_entries(self, indices: torch.Tensor) -> torch.Tensor:
        """x̂ at the given entries. ``indices``: (order, B) long tensor."""
        rows = [self.factor(m)[indices[m]] for m in range(self.order)]  # (B, R_m)
        return torch.einsum(self._eq_predict, self.core, *rows)

    def total_sum(self) -> torch.Tensor:
        """sum(X̂) over ALL entries — the exact KL zero-entry term."""
        col_sums = [self.factor(m).sum(dim=0) for m in range(self.order)]
        return torch.einsum(self._eq_sum, self.core, *col_sums)

    def total_sq_norm(self) -> torch.Tensor:
        """‖X̂‖² over ALL entries — the exact Frobenius zero-entry term."""
        grams = [self.factor(m).T @ self.factor(m) for m in range(self.order)]
        return torch.einsum(self._eq_sqnorm, self.core, *grams, self.core)

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


def zero_entry_term(model, divergence):
    """The EXACT closed-form zero-entry half — a function of the parameters
    only, so it is added once per step no matter how the entries are split."""
    if divergence == "kl":
        return model.total_sum()
    if divergence == "fr":
        return model.total_sq_norm()
    raise ValueError(f"divergence must be 'kl' or 'fr', got {divergence!r}")


def _batch_loss(model, x, x_hat, scale, divergence, masked, eps=_EPS):
    """Unbiased estimate of the total objective. ``scale`` = nnz / batch."""
    loss = sampled_loss(x, x_hat, scale, divergence, masked, eps=eps)
    if masked:
        return loss
    return loss + zero_entry_term(model, divergence)


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
    Exact, not approximate (see ``sampled_loss``); the zero-entry term is added
    once, after the loop, because it depends only on the parameters.

    **Optional CUDA-graph capture.** At production mode dimensions the order-3
    step is ~50x off its arithmetic roofline — it is Python dispatch, kernel
    launch and autograd-node overhead, not math. The step body is fixed-shape,
    so it captures cleanly. The one dynamic input, the batch window offset, is
    handled by writing indices into a static buffer *outside* the capture.
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

        Capturable: fixed shapes, no host syncs, no allocations that depend on
        a host-side value.
        """
        self.flat_grad.zero_()
        for lo, hi in self._slices:
            sel = self._sel[lo:hi]
            idx = self.indices[:, sel]
            x = self.values[sel]
            x_hat = self.model.predict_entries(idx)
            loss = sampled_loss(x, x_hat, self.scale, self.divergence,
                                self.masked, eps=self.eps) / self.norm_const
            loss.backward()
        if self.include_zero_term:
            z = zero_entry_term(self.model, self.divergence) / self.norm_const
            z.backward()

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
def full_relative_error(
    model: SGDTuckerModel,
    indices: torch.Tensor,
    values: torch.Tensor,
    divergence: str,
    masked: bool = False,
    chunk: Optional[int] = None,
    eps: float = _EPS,
) -> float:
    """Exact objective over ALL nnz (chunked), normalized like distance.py:
    KL / Σx  (kl_compute_errors) or ‖X−X̂‖/‖X‖ (fr_compute_errors)."""
    if chunk is None:
        chunk = _default_eval_chunk(model.rank)
    nnz = values.shape[0]
    acc = values.new_zeros(())
    for s in range(0, nnz, chunk):
        idx = indices[:, s:s + chunk]
        x = values[s:s + chunk]
        x_hat = model.predict_entries(idx)
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

    if divergence == "kl":
        total = acc if masked else acc + model.total_sum()
        return float(total / values.sum().clamp_min(eps))
    total = acc if masked else acc + model.total_sq_norm()
    norm_x = values.pow(2).sum().sqrt().clamp_min(eps)
    return float(total.clamp_min(0.0).sqrt() / norm_x)


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

    ``eval_chunk`` sizes the exact-error pass; None derives it from the rank so
    the per-chunk intermediate stays bounded (see ``_default_eval_chunk``).

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
        if not (isinstance(t, torch.Tensor) and t.is_sparse):
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
            x_hat = model.predict_entries(idx)
            loss = _batch_loss(model, x, x_hat, scale, divergence, masked) / norm_const
            loss.backward()
            opt.step()
            model.project_()

            if eval_every > 0 and (step + 1) % eval_every == 0:
                rel = full_relative_error(model, indices, values, divergence,
                                          masked=masked, chunk=eval_chunk)
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

        final_error = errors[-1] if errors else (
            # No eval step fired (n_steps < eval_every): compute once at the end
            # so short smoke runs still report an error.
            full_relative_error(model, indices, values, divergence,
                                masked=masked, chunk=eval_chunk)
            if step >= 0 else None
        )
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
