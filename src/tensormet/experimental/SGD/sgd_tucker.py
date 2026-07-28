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


def _default_eval_chunk(rank: Sequence[int], budget: int = 1 << 26) -> int:
    """Entries per chunk in ``full_relative_error``.

    ``predict_entries`` contracts the core against B gathered factor rows, whose
    largest intermediate is B × prod(rank[:-1]) elements. A fixed chunk is fine
    for the toy ranks but explodes at production scale (B=2^20, rank=100 → 10^10
    elements ≈ 40 GB), so size the chunk from the rank instead: ``budget`` caps
    the intermediate at 2^26 elements (~256 MB fp32).
    """
    inter = 1
    for r in tuple(rank)[:-1]:
        inter *= int(r)
    return int(max(1024, min(1 << 20, budget // max(inter, 1))))


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

def _batch_loss(model, x, x_hat, scale, divergence, masked, eps=_EPS):
    """Unbiased estimate of the total objective. ``scale`` = nnz / batch."""
    if divergence == "kl":
        x_safe = x.clamp_min(eps)
        nz = x_safe * torch.log(x_safe / (x_hat + eps)) - x
        if masked:
            return scale * (nz + x_hat).sum()
        return scale * nz.sum() + model.total_sum()
    if divergence == "fr":
        sq = (x - x_hat) ** 2
        if masked:
            return scale * sq.sum()
        return scale * (sq - x_hat ** 2).sum() + model.total_sq_norm()
    raise ValueError(f"divergence must be 'kl' or 'fr', got {divergence!r}")


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
