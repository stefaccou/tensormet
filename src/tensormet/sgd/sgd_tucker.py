"""
sgd_tucker.py — SGD/Adam non-negative Tucker, plus the pieces every SGD model
uses: the batcher, the objectives, the per-device step (``GradStepper``) and
the error. See sgd/README.md for costs, multi-GPU knobs and integration.

Model: X̂ = G ×_1 A^(1) ×_2 … ×_N A^(N), kept non-negative by a
parametrization (``sgd_model.PARAMETRIZATIONS``). The TT and utt models
(``sgd_tt.py``, ``experimental/uniform_tt/utt_sgd.py``) have the same surface;
``sgd_trainer.make_sgd_model`` picks one.

Objectives (the MU path's divergences). The zero-entry part is exact and in
closed form, so no negative sampling is needed:

  KL  full :  Σ_nz [x·log(x/x̂) − x]          (sampled, rescaled by nnz/batch)
              + sum(X̂)                       (exact: core ⨯ column sums)
  KL  mask :  Σ_nz [x·log(x/x̂) − x + x̂]     (sampled, rescaled)
  FR  full :  Σ_nz [(x − x̂)² − x̂²]           (sampled, rescaled)
              + ‖X̂‖²                         (exact: core ⨯ factor Grams)
  FR  mask :  Σ_nz (x − x̂)²                   (sampled, rescaled)

Reported errors are KL / Σx and ‖X−X̂‖ / ‖X‖, as in ``distance.py``.

Batches are a pure function of (seed, step), so resume replays them exactly.
The backward scatter-adds are nondeterministic on CUDA; pass
``deterministic=True`` to ``sgd_non_negative_tucker`` for bitwise reproducibility.
"""
from __future__ import annotations

import math
import time
from typing import List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
from tensorly import backend_context
from tensorly.tucker_tensor import TuckerTensor

# _EPS, PARAMETRIZATIONS and the helpers below moved to sgd_model.py; imported
# here too so older imports from this module keep working.
from tensormet.sgd.sgd_model import (  # noqa: F401
    _EPS,
    PARAMETRIZATIONS,
    SGDModelBase,
    _as_rank_tuple,
    _parse_shared_factors,
    raw_from_view,
    view_from_raw,
)
from tensormet.utils import SparseCOOTensor, einsum_letters


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _inv_softplus(y: torch.Tensor) -> torch.Tensor:
    # softplus⁻¹(y) = log(expm1(y)); clamp keeps the log finite for tiny inits.
    return torch.log(torch.expm1(y.clamp_min(1e-6)))


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


def _rank_derived_chunk(rank: Sequence[int], budget: int = 1 << 26,
                        width: Optional[int] = None) -> int:
    """Largest entry count whose forward intermediate fits ``budget`` elements
    (2^26 ≈ 256 MB fp32), within [64, 2^20].

    ``width`` is the per-entry intermediate (``model.entry_width``); None
    derives Tucker's from the rank. The 64 floor only binds at order 5+, where
    an explicit ``--sgd-micro-batch`` is the better choice.
    """
    if width is None:
        width = _contraction_plan(rank)[2]
    return int(max(64, min(1 << 20, budget // max(1, int(width)))))


def _row_khatri_rao(rows: Sequence[torch.Tensor]) -> torch.Tensor:
    """Row-wise Khatri-Rao of ``(B, R_i)`` matrices → ``(B, prod R_i)``, in the
    C order of ``core.reshape``. A single input is returned untouched."""
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
    *,
    width: Optional[int] = None,
) -> int:
    """Entries per forward/backward inside one optimizer step.

    The sampled loss is a sum over entries, so splitting a batch and
    accumulating the gradients is exact. ``micro_batch=None`` derives the value
    from the per-entry ``width`` (None: Tucker's, from the rank). An explicit
    value is checked against ``ceiling_bytes``, so an over-large request fails
    with a message naming the knob instead of a CUDA OOM.
    """
    batch_size = int(batch_size)
    if batch_size < 1:
        raise ValueError(f"batch_size must be >= 1, got {batch_size}")
    inter = _contraction_plan(rank)[2] if width is None else max(1, int(width))
    itemsize = torch.empty((), dtype=dtype).element_size()

    if micro_batch is None:
        return min(batch_size, _rank_derived_chunk(rank, budget, width=inter))

    mb = int(micro_batch)
    if mb < 1:
        raise ValueError(f"sgd_micro_batch must be >= 1, got {micro_batch}")
    mb = min(mb, batch_size)
    projected = inter * mb * itemsize
    if projected > ceiling_bytes:
        raise ValueError(
            f"An SGD forward pass at micro_batch={mb} and rank={tuple(rank)} would "
            f"allocate a {projected / 2**30:.1f} GiB intermediate "
            f"({mb} entries x {inter} elements per entry x {itemsize} B), above the "
            f"{ceiling_bytes / 2**30:.1f} GiB ceiling — and autograd saves it for "
            f"the backward pass on top. Lower --sgd-micro-batch (the rank-derived "
            f"default here is {min(batch_size, _rank_derived_chunk(rank, budget, width=inter))}) "
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
        """A ``(batch_size,)`` index buffer for ``batch_into``. Zeroed, so a read
        before the first fill still indexes valid entries."""
        return torch.zeros(self.batch_size, dtype=self.dtype, device=self.device)

    def batch_into(self, step: int, out: torch.Tensor) -> torch.Tensor:
        """``batch(step)`` written into ``out`` without allocating. The CUDA-graph
        path needs the indices at a fixed address; this copy runs outside the
        captured region."""
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

        if parametrization == "softplus":
            to_raw = _inv_softplus
        elif parametrization == "exp":
            to_raw = torch.log
        else:
            to_raw = lambda t: t  # noqa: E731
        self._core_raw = torch.nn.Parameter(to_raw(core0))
        # Only owner modes hold a Parameter; aliased modes read the owner's.
        self._factor_raw = torch.nn.ParameterDict({
            str(m): torch.nn.Parameter(to_raw(factors0[m]))
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


# ---------------------------------------------------------------------------
# objectives
# ---------------------------------------------------------------------------

def _entry_terms(x, x_hat, divergence, masked, eps=_EPS):
    """Per-entry objective over the nnz: KL x·log(x/x̂) − x (+ x̂ if masked);
    FR (x − x̂)² (− x̂² for the full objective, whose zero-entry term adds ‖X̂‖²)."""
    if divergence == "kl":
        x_safe = x.clamp_min(eps)
        terms = x_safe * torch.log(x_safe / (x_hat + eps)) - x
        return terms + x_hat if masked else terms
    if divergence == "fr":
        sq = (x - x_hat) ** 2
        return sq if masked else sq - x_hat ** 2
    raise ValueError(f"divergence must be 'kl' or 'fr', got {divergence!r}")


def sampled_loss(x, x_hat, scale, divergence, masked, eps=_EPS):
    """The sampled (nnz) half of the objective, rescaled to full-tensor scale
    (``scale`` = nnz / batch). A plain sum over entries, so splitting them
    (micro-batches, NNZ shards) and adding the parts is exact."""
    return scale * _entry_terms(x, x_hat, divergence, masked, eps).sum()


@torch.no_grad()
def nnz_term_sum(model, core, factors, indices, values, divergence, masked, chunk, eps=_EPS):
    """Σ of the per-entry terms over the given entries, ``chunk`` at a time (0-d)."""
    acc = values.new_zeros(())
    for s in range(0, values.shape[0], chunk):
        x_hat = model.predict_from(core, factors, indices[:, s:s + chunk])
        acc = acc + _entry_terms(values[s:s + chunk], x_hat, divergence, masked, eps).sum()
    return acc


def _distinct_views(core, factors) -> List[torch.Tensor]:
    """``[core] + factors`` in first-occurrence order, aliased modes collapsed:
    one tensor per trainable parameter. ``core`` may be a list (TT cores, or
    utt's [C, α, ω])."""
    out = list(core) if isinstance(core, (list, tuple)) else [core]
    seen = {id(t) for t in out}
    for f in factors:
        if id(f) not in seen:
            seen.add(id(f))
            out.append(f)
    return out


def zero_entry_term(model, divergence, views=None):
    """The exact closed-form zero-entry half: a function of the parameters only,
    so it is added once per step however the entries are split. ``views`` is an
    optional ``(core, factors)`` from ``model.nonneg_views()``."""
    core, factors = model.nonneg_views() if views is None else views
    if divergence == "kl":
        return model.total_sum_from(core, factors)
    if divergence == "fr":
        return model.total_sq_norm_from(core, factors)
    raise ValueError(f"divergence must be 'kl' or 'fr', got {divergence!r}")


def _batch_loss(model, x, x_hat, scale, divergence, masked, eps=_EPS, views=None):
    """Unbiased estimate of the total objective (used by the multi-GPU test notebook)."""
    loss = sampled_loss(x, x_hat, scale, divergence, masked, eps=eps)
    if masked:
        return loss
    return loss + zero_entry_term(model, divergence, views=views)


# ---------------------------------------------------------------------------
# one step's gradient, shared by the single-GPU and sharded trainers
# ---------------------------------------------------------------------------

class GradStepper:
    """Fills one model's gradients for one step on one device; both trainers
    build on it.

    - Flat gradient buffer: every parameter's ``.grad`` is a view into
      ``flat_grad``, so the sharded all-reduce is one collective and zeroing is
      one kernel. Never call ``zero_grad(set_to_none=True)`` on these models.
    - Micro-batching: the batch is split into chunks that fit a memory budget
      and the gradients accumulate (exact, see ``sampled_loss``). Up to about
      order 4 / rank 200 there is a single chunk.
    - Optional CUDA-graph capture of the fixed-shape step body; the batch
      indices are copied into a static buffer outside the capture.
    """

    def __init__(
        self,
        model: SGDModelBase,
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
        # A masked objective has no zero-entry term. The sharded trainer also
        # turns it off on all but one device when gradients are summed.
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
            model.rank, self.batch_size, micro_batch, dtype=dtype,
            width=model.entry_width,
        )
        self._slices = [
            (lo, min(lo + self.micro_batch, self.batch_size))
            for lo in range(0, self.batch_size, self.micro_batch)
        ]
        self._sel = batcher.new_buffer()

        # Gradient buffers for the views; only needed (and only allocated) when
        # the step is split into several chunks.
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
        """Zero in place, never ``set_to_none``: the grads are views into ``flat_grad``."""
        self.flat_grad.zero_()

    def new_param_buffer(self) -> torch.Tensor:
        """A buffer ``pack_params_`` fits, laid out like ``flat_grad``."""
        return torch.empty_like(self.flat_grad)

    @torch.no_grad()
    def pack_params_(self, out: torch.Tensor) -> torch.Tensor:
        """Parameters -> one contiguous vector (parameter averaging, drift check)."""
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
        """Fill ``flat_grad`` from the indices in ``self._sel``. The
        parametrization runs once per step in both branches (it is a pass over
        the whole core). Capturable: fixed shapes, no host syncs."""
        self.flat_grad.zero_()
        model = self.model
        core, factors = model.nonneg_views()

        if self._view_grads is None:
            # One chunk: one graph from the raw parameters, one backward.
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

        # Several chunks: each chunk's graph is freed before the next, so the
        # chunks differentiate against detached views whose gradients collect
        # in persistent buffers; one backward at the end pushes the total
        # through the parametrization into the raw parameters.
        outs = _distinct_views(core, factors)
        leaves = []
        for src, gbuf in zip(outs, self._view_grads):
            gbuf.zero_()
            leaf = src.detach().requires_grad_(True)
            leaf.grad = gbuf
            leaves.append(leaf)
        leaf_of = {id(s): l for s, l in zip(outs, leaves)}
        core_l = ([leaf_of[id(c)] for c in core] if isinstance(core, (list, tuple))
                  else leaf_of[id(core)])
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
            # Warm up on a side stream so lazy kernel loads and workspace
            # allocations happen before the capture.
            side = torch.cuda.Stream(device=self.device)
            side.wait_stream(torch.cuda.current_stream(self.device))
            with torch.cuda.stream(side):
                for _ in range(max(1, int(warmup))):
                    self._body()
            torch.cuda.current_stream(self.device).wait_stream(side)
            torch.cuda.synchronize(self.device)

            graph = torch.cuda.CUDAGraph()
            try:
                # thread_local: the sharded trainer captures one graph per
                # device, and other devices' threads must not invalidate it.
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


# ---------------------------------------------------------------------------
# error
# ---------------------------------------------------------------------------

@torch.no_grad()
def make_eval_subset(
    indices: torch.Tensor,
    values: torch.Tensor,
    eval_sample: Optional[int],
    random_state: int,
) -> Tuple[torch.Tensor, torch.Tensor, float, Tuple[float, float]]:
    """``(indices, values, sample_scale, totals)`` for ``full_relative_error``.

    ``eval_sample=None`` (or ``>= nnz``) returns the inputs: the exact error.
    Otherwise a fixed uniform subset of that many entries (with replacement,
    seeded from ``random_state``), drawn once: tol/patience compare successive
    errors, so the subset must not move between evals. ``totals`` is
    ``(Σx, Σx²)`` over the full tensor, the error denominators.
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
    model: SGDModelBase,
    indices: torch.Tensor,
    values: torch.Tensor,
    divergence: str,
    masked: bool = False,
    chunk: Optional[int] = None,
    eps: float = _EPS,
    sample_scale: float = 1.0,
    totals: Optional[Tuple[float, float]] = None,
) -> float:
    """Relative error over the given nnz, normalized as in distance.py: KL / Σx
    or ‖X−X̂‖ / ‖X‖. The zero-entry part is exact.

    For a random subset of the entries, pass ``sample_scale = nnz / len(subset)``
    and ``totals = (Σx, Σx²)`` of the full tensor (``make_eval_subset``): the
    nnz part is then rescaled, which is unbiased for the KL numerator and for
    the squared FR numerator. Costs as much per entry as a training step.
    """
    if chunk is None:
        chunk = _rank_derived_chunk(model.rank, width=model.entry_width)
    core, factors = model.nonneg_views()
    acc = nnz_term_sum(model, core, factors, indices, values, divergence, masked, chunk, eps)
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
# standalone training loop (notebook use; production drives SGDTrainer)
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
    """Fit a non-negative Tucker model by minibatch SGD/Adam: a step-by-step
    loop over ``SGDTrainer`` with early stopping.

    Parameters mirror the MU loop where they overlap (divergence, objective,
    tol/patience on the logged error, random_state, shared_factors).
    ``eval_sample`` logs the error on a fixed random subset of that many nnz
    (``make_eval_subset``); ``final_error`` is always exact.

    ``sparse_tensor``: a torch sparse COO tensor, or an object whose
    ``.tensor`` holds one (e.g. a SparseTupleTensor before any CuPy conversion).

    Returns ``{"tensor": TuckerTensor (CPU numpy), "errors", "iterations",
    "final_error", "decomp_seconds"}``; ``return_model=True`` adds the live
    ``"model"`` (which keeps its device memory alive).
    """
    from tensormet.sgd.sgd_trainer import SGDTrainer

    # torch.use_deterministic_algorithms is process-global; restore it on exit.
    det_prev = torch.are_deterministic_algorithms_enabled()
    if deterministic:
        torch.use_deterministic_algorithms(True)
    try:
        t = getattr(sparse_tensor, "tensor", sparse_tensor)
        if not (isinstance(t, (torch.Tensor, SparseCOOTensor)) and t.is_sparse):
            raise TypeError(
                "sgd_non_negative_tucker expects a torch sparse COO tensor "
                "(or an object with .tensor holding one)."
            )
        # One optimizer step per block, so the error can be logged at any step.
        trainer = SGDTrainer(
            t, rank=rank, divergence=divergence, objective=objective, lr=lr,
            batch_size=batch_size, optimizer=optimizer,
            parametrization=parametrization, shared_factors=shared_factors,
            init=init, random_state=random_state, steps_per_iteration=1,
            device=device, dtype=dtype, eval_chunk=eval_chunk,
            eval_sample=eval_sample,
        )

        errors: List[float] = []
        last_err: Optional[float] = None
        no_improve = 0
        start = time.time()
        step = -1

        for step in range(n_steps):
            log_step = eval_every > 0 and (step + 1) % eval_every == 0
            rel = trainer.run_block(step, log_step)
            if not log_step:
                continue
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

        if errors and trainer.eval_sample is None:
            final_error = errors[-1]
        elif step >= 0:
            # No eval fired, or the logged errors are subsampled: exact, once.
            final_error = trainer.final_relative_error()
        else:
            final_error = None
        decomp_seconds = time.time() - start
        core, factors = trainer.materialize()
        # TuckerTensor validates through the active tensorly backend (often
        # "pytorch" here, which rejects numpy), so build it under numpy.
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
            out["model"] = trainer.model
        return out
    finally:
        torch.use_deterministic_algorithms(det_prev)
