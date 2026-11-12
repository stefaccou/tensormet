"""
sgd_step.py — what every SGD model shares per step: the entry batcher, the
objectives, the per-device step (``GradStepper``) and the error. The models are
in ``sgd_tucker.py`` / ``sgd_tt.py`` / ``uniform_tt/utt_sgd.py``, the trainer in
``sgd_trainer.py``. See sgd/README.md.

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

from typing import List, Optional, Sequence, Tuple

import numpy as np
import torch

from tensormet.sgd.sgd_model import _EPS, SGDModelBase
from tensormet.sgd.sgd_tucker import _contraction_plan


# ---------------------------------------------------------------------------
# memory sizing
# ---------------------------------------------------------------------------

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

    def batch(self, step: int) -> torch.Tensor:
        start = (step * self.batch_size) % self.nnz
        end = start + self.batch_size
        if end <= self.nnz:
            return self._perm[start:end]
        return torch.cat([self._perm[start:], self._perm[: end - self.nnz]])


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

        # Gradient buffers for the views; only needed (and only allocated) when
        # the step is split into several chunks.
        self._view_grads: Optional[List[torch.Tensor]] = None
        if len(self._slices) > 1:
            with torch.no_grad():
                self._view_grads = [
                    torch.zeros_like(t)
                    for t in _distinct_views(*model.nonneg_views())
                ]

    # --- gradient plumbing ---------------------------------------------------

    def zero_grad_(self) -> None:
        """Zero in place, never ``set_to_none``: the grads are views into ``flat_grad``."""
        self.flat_grad.zero_()

    # --- the step body -------------------------------------------------------

    def compute_grads(self, step: int) -> None:
        """Gradients of step ``step``'s objective, into ``flat_grad``. The
        parametrization runs once per step in both branches (it is a pass over
        the whole core)."""
        batch = self.batcher.batch(step)
        self.flat_grad.zero_()
        model = self.model
        core, factors = model.nonneg_views()

        if self._view_grads is None:
            # One chunk: one graph from the raw parameters, one backward.
            lo, hi = self._slices[0]
            sel = batch[lo:hi]
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
            sel = batch[lo:hi]
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


