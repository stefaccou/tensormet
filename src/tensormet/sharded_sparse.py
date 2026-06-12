"""
sharded_sparse.py — Multi-GPU NNZ sharding and stochastic subsampling.

Design
------
ShardedSparseTensor wraps a primary-device COO matrix and holds one
sub-matrix (NNZ shard) per GPU device.  When n_shards == 1 the class
is a thin no-op wrapper that delegates immediately to the existing
single-GPU functions from distance.py (zero overhead on the fallback path).

For n_shards > 1 the NNZ-dependent accumulations are parallelised across
devices using Python threads (one thread per GPU).  Results are reduced on
the CPU, then transferred back to the primary device.  No NCCL / NVLINK
required.

Stochastic subsampling (multi-GPU path)
----------------------------------------
When ``subsample_frac < 1.0``, each shard's NNZ arrays are shuffled **once**
at construction (host-side, seeded per shard), and each per-shard function
takes a contiguous rotating window of its local NNZ per iteration, rescaling
values by ``1/subsample_frac``.  A contiguous window of a uniformly shuffled
sequence is a uniform sample without replacement, so the estimator is
unbiased — and successive windows tile the shard like an epoch, with zero
per-iteration index allocation (the old per-iteration ``rng.permutation(nnz)``
allocated and sorted 8·nnz bytes on the GPU every call).  Samples are a pure
function of (construction seed, iteration): deterministic and resume-safe.

Call ``sst.set_iter_seed(iteration)`` once at the top of each iteration so
the SST knows which window to take — wrapper function signatures are unchanged.

Reduction strategies
--------------------
Factor / Core updates:
  GPU_k  ->  partial_Num_k.get()  ->  numpy np.add.reduce  ->  cp.asarray on GPU_0

Error functions:
  GPU_k  ->  (scalar_a, scalar_b).get()  ->  Python sum  ->  cp.asarray on GPU_0

Scope
-----
Only the *largedim* variants are sharded:
  - Factor updates:   KL dim >= 4000, FR dim > 4000 or largedim=True
  - Core updates:     same thresholds
  - Error functions:  same thresholds
"""

from __future__ import annotations

import math
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

# Opt-in worker diagnostics for the multi-GPU masked path. Set TENSORMET_DEBUG_SHARD=1
# (ideally alongside CUDA_LAUNCH_BLOCKING=1) to bounds-check row gathers per batch and
# dump device memory + batch context if a shard raises. Off by default: the per-batch
# r_i.max() bounds check forces a host sync that would otherwise hurt GPU overlap.
_DEBUG_SHARD = os.environ.get("TENSORMET_DEBUG_SHARD", "") not in ("", "0", "false", "False")

from tensormet.distance import (
    # Factor update helpers
    _blocked_coo_to_flat_indices,
    _estimate_batch_cols_for_Z,
    _unravel_flat_indices_C,
    _unravel_cols_for_mode,
    fr_factor_update_largedim,
    kl_factor_update_largedim,
    # Core update helpers
    _accumulate_core_num_outer,
    _core_multilinear_grams,
    _estimate_batch_num_for_outer,
    _estimate_batch_rhat_for_tensordot,
    _rhat_from_factor_rows_sequential,
    _tucker_sum_all_entries,
    fr_core_update_largedim,
    kl_core_update_largedim,
    # Error helpers
    fr_compute_errors_largedim,
    kl_compute_errors_largedim,
    # Denominator helpers
    _tucker_den_row_full,
    _tucker_gram_ZtZ,
)
from tensormet.sparse_ops import compute_Zcols_batch, safe_ravel
from tensormet.utils import make_lazy_cupy_pair

cp, cpx_sparse = make_lazy_cupy_pair()

# ---------------------------------------------------------------------------
# Internal utilities
# ---------------------------------------------------------------------------

def _build_shard(
    coo: cpx_sparse.coo_matrix,
    start: int,
    end: int,
    target_device: int,
    shuffle_seed: Optional[int] = None,
) -> cpx_sparse.coo_matrix:
    """
    Extract NNZ slice [start, end) from *coo* and place it on *target_device*.

    Preserves the same ``(block_size, n_blocks)`` shape so that
    ``_blocked_coo_to_flat_indices`` works identically on shards.
    Routes through CPU because CuPy does not support direct cross-device
    tensor slicing.  One-time cost at initialisation.

    When *shuffle_seed* is given (subsampling enabled), the slice is uniformly
    shuffled on the host before transfer, so that the contiguous windows taken
    by ``_apply_subsample`` are uniform samples without replacement.  COO entry
    order carries no meaning for any downstream accumulation (sums are
    order-invariant; ``cp.unique`` re-sorts its input), so the shuffle is
    content-preserving.
    """
    row_np = cp.asnumpy(coo.row[start:end])
    col_np = cp.asnumpy(coo.col[start:end])
    data_np = cp.asnumpy(coo.data[start:end])

    # CHANGED (2026-06-12 review, Task 2): one-time host-side shuffle replaces the
    # per-iteration cp.random.permutation in _apply_subsample. Done here because the
    # NNZ slice already round-trips through the CPU, so the shuffle costs no GPU
    # memory and no extra transfer.
    if shuffle_seed is not None:
        perm = np.random.default_rng(int(shuffle_seed)).permutation(row_np.size)
        row_np, col_np, data_np = row_np[perm], col_np[perm], data_np[perm]

    with cp.cuda.Device(target_device):
        shard = cpx_sparse.coo_matrix(
            (cp.asarray(data_np), (cp.asarray(row_np), cp.asarray(col_np))),
            shape=coo.shape,
        )
    return shard


def _apply_subsample(
    flat: cp.ndarray,
    vals: cp.ndarray,
    subsample_frac: float,
    iteration: Optional[int],
) -> Tuple[cp.ndarray, cp.ndarray]:
    """
    Take this iteration's contiguous NNZ window from pre-shuffled storage.

    CHANGED (2026-06-12 review, Task 2): previously this drew
    ``cp.sort(rng.permutation(nnz)[:n_sample])`` — an 8·nnz-byte int64
    allocation plus a full device sort, *per shard per call*, on exactly the
    GPUs subsampling is meant to relieve.  The shard's NNZ arrays are now
    shuffled once at construction (``_build_shard(shuffle_seed=...)``), so a
    contiguous window ``[(iteration·n_sample) % nnz : +n_sample)`` (wrapping)
    is a uniform sample without replacement.

    Estimator argument: the construction shuffle makes every length-n_sample
    contiguous window a uniformly distributed size-n_sample subset, so any
    linear accumulation over the rescaled window is unbiased
    (``E[sum_S x/frac] = sum x``).  Successive windows tile the shard like an
    epoch: every entry is visited once per ⌈nnz/n_sample⌉ iterations.

    Memory: slicing is O(1) (CuPy views); only the value rescale and the rare
    wrap-around concatenation allocate, both O(n_sample) — never O(nnz).

    Parameters
    ----------
    flat, vals :
        Flat indices and values from ``_blocked_coo_to_flat_indices`` on a
        shard built with ``shuffle_seed`` set (i.e. already in shuffled order).
    subsample_frac :
        Fraction of NNZ to retain.  Must be < 1.0 (caller is responsible
        for not calling this at 1.0).
    iteration :
        Training-loop iteration number; selects the window.  ``None`` is
        treated as 0 (deterministic given (construction seed, iteration) —
        no RNG state survives between calls, so resumed runs draw the same
        windows as uninterrupted ones).

    Returns
    -------
    flat_s, vals_s : windowed and rescaled arrays.
    """
    nnz = int(flat.size)
    n_sample = max(1, int(round(subsample_frac * nnz)))
    start = (int(iteration or 0) * n_sample) % nnz
    end = start + n_sample
    if end <= nnz:
        flat_s = flat[start:end]
        vals_s = vals[start:end]
    else:  # wrap around the end of the shuffled sequence
        flat_s = cp.concatenate((flat[start:], flat[: end - nnz]))
        vals_s = cp.concatenate((vals[start:], vals[: end - nnz]))
    scale = vals.dtype.type(1.0 / subsample_frac)
    return flat_s, vals_s * scale


# ---------------------------------------------------------------------------
# Factor update — per-shard partial numerator
# ---------------------------------------------------------------------------

def _partial_numerator_for_shard(
    shard: cpx_sparse.coo_matrix,
    core_np: Union[np.ndarray, Any],
    factors_np: List[Union[np.ndarray, Any]],
    mode: int,
    shape: Tuple[int, ...],
    divergence: str,
    epsilon: float,
    batch_cols: Optional[int],
    device_id: int,
    subsample_frac: float = 1.0,
    iteration: Optional[int] = None,
    masked: bool = False,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Compute the partial factor-numerator contribution from a single NNZ shard.

    Runs entirely inside ``cp.cuda.Device(device_id)``.  When
    ``subsample_frac < 1.0`` the shard's NNZ is subsampled locally before
    accumulation — no resharding is needed.  *iteration* selects the
    contiguous window of the pre-shuffled shard (see ``_apply_subsample``).

    ``core_np`` and ``factors_np`` may be NumPy arrays *or* CuPy arrays
    already resident on ``device_id``.  ``cp.asarray`` is a no-op in the
    latter case, so no CPU round-trip occurs for the primary shard.

    When ``masked`` is True, the (observed-only) denominator is also accumulated
    on this shard and returned, since for the masked/completion objective the
    denominator depends on the NNZ pattern and cannot be computed analytically.

    Returns
    -------
    (partial_num, partial_den) : np.ndarray of shape ``(I_mode, R_mode)`` each.
        ``partial_den`` is ``None`` when ``masked`` is False.
    """
    # .use() permanently activates the device for this thread; avoids the
    # cuCtxPushCurrent/Pop cycle of the context manager, which leaves the
    # cuBLAS handle in a stale state after many iterations in a persistent pool.
    cp.cuda.Device(device_id).use()
    core_d = cp.asarray(core_np)
    factors_d = [cp.asarray(f) for f in factors_np]
    A_d = factors_d[mode]

    flat, vals = _blocked_coo_to_flat_indices(shard, shape)

    if flat.size == 0:
        zero = cp.asnumpy(cp.zeros_like(A_d))
        return zero, (zero.copy() if masked else None)

    if subsample_frac < 1.0:
        flat, vals = _apply_subsample(flat, vals, subsample_frac, iteration)

    # Under subsampling the numerator's `vals` are already rescaled by 1/frac
    # (see _apply_subsample). The masked denominator weights (unit / x̂) must be
    # rescaled identically so the MU ratio stays unbiased; the factor cancels in
    # the ratio but only if both sides carry it.
    den_scale = (1.0 / subsample_frac) if subsample_frac < 1.0 else 1.0

    idxs = _unravel_flat_indices_C(flat, shape)
    rows = idxs[mode]

    other_modes = [m for m in range(len(shape)) if m != mode]
    other_shape = tuple(shape[m] for m in other_modes)
    other_coords = [idxs[m] for m in other_modes]
    cols = safe_ravel(tuple(other_coords), other_shape, cp)

    numerator = cp.zeros_like(A_d)
    denominator = cp.zeros_like(A_d) if masked else None
    ucols, inv = cp.unique(cols, return_inverse=True)
    n_ucols = int(ucols.size)

    # Estimate AFTER all NNZ bookkeeping is live on the GPU so the free-memory
    # snapshot reflects the actual headroom available for the einsum temporaries.
    if batch_cols is None:
        batch_cols = int(_estimate_batch_cols_for_Z(core_d, factors_d, mode, masked=masked))

    mempool = cp.get_default_memory_pool()
    # cuBLAS/cuSPARSE allocate their GEMM/SpMM workspaces with a raw cudaMalloc
    # *outside* CuPy's pool. The masked path's extra per-batch allocations
    # (A_d[r_i], den_w, S_den, the second SpMM, the persistent denominator) fill
    # the pool enough that this out-of-pool workspace cudaMalloc can fail — which
    # surfaces as CUBLAS_STATUS_NOT_INITIALIZED from gemmStridedBatchedEx, not as
    # a CuPy OutOfMemoryError (this is why full, with a *larger* einsum but no
    # extra allocations, never crashes). On either failure we return the pool's
    # cached blocks to the driver, halve the batch, and retry. Each batch is
    # committed atomically (accumulators are touched only after every allocation
    # for the batch has succeeded), so a retry never double-counts.
    _retryable = [cp.cuda.memory.OutOfMemoryError]
    try:
        _retryable.append(cp.cuda.cublas.CUBLASError)
    except AttributeError:  # pragma: no cover - cuBLAS error class always present in practice
        pass
    _retryable = tuple(_retryable)

    bc = int(batch_cols)
    batch_start = 0
    while batch_start < n_ucols:
        batch_end = min(batch_start + bc, n_ucols)
        try:
            u = ucols[batch_start:batch_end]
            _, idxs_by_mode = _unravel_cols_for_mode(u, shape, mode)
            Z_u = compute_Zcols_batch(
                core=core_d,
                factors=factors_d,
                mode=mode,
                other_modes=other_modes,
                idxs_by_mode=idxs_by_mode,
                epsilon=epsilon,
            )

            nz_idx = cp.where((inv >= batch_start) & (inv < batch_end))[0]
            if nz_idx.size == 0:
                batch_start = batch_end
                continue

            r_i = rows[nz_idx]
            v_i = vals[nz_idx]
            u_i = inv[nz_idx] - batch_start
            Z_rows = Z_u[u_i]

            if _DEBUG_SHARD:
                # Fancy-index gathers (A_d[r_i], Z_u[u_i]) are NOT bounds-checked
                # by CuPy; an OOB index is an illegal access that poisons the
                # context and only surfaces later (often as a cuBLAS error).
                r_max = int(r_i.max()) if r_i.size else -1
                u_max = int(u_i.max()) if u_i.size else -1
                if r_max >= A_d.shape[0] or u_max >= Z_u.shape[0]:
                    raise RuntimeError(
                        f"[shard-diag] OOB gather: r_max={r_max} (A_d rows={A_d.shape[0]}), "
                        f"u_max={u_max} (Z_u rows={Z_u.shape[0]}), device={device_id}, "
                        f"batch={batch_start}:{batch_end}, nnz_b={int(r_i.size)}"
                    )

            nnz_b = int(r_i.size)
            col_idx_b = cp.arange(nnz_b, dtype=cp.int32)
            row_idx_b = r_i.astype(cp.int32)

            if divergence == "kl":
                A_rows = A_d[r_i]
                R_nz = cp.clip(cp.sum(A_rows * Z_rows, axis=1), a_min=epsilon, a_max=None)
                w = v_i / R_nz
            else:  # "fr"
                w = v_i

            # numerator[row] += w * Z  — cuSPARSE SpMM (no serialised atomics).
            # Build contributions into locals first so a failure below leaves the
            # accumulators untouched and the batch can be retried cleanly.
            S_b = cpx_sparse.csr_matrix(
                (w, (row_idx_b, col_idx_b)),
                shape=(numerator.shape[0], nnz_b),
            )
            num_contrib = S_b @ Z_rows

            den_contrib = None
            if masked:
                # Observed-only denominator weight:
                #   KL -> 1 (sum of Z over observed columns)
                #   FR -> Xhat at the observed entry = <A[row], Z[col]>
                if divergence == "kl":
                    den_w = cp.full(nnz_b, den_scale, dtype=Z_rows.dtype)
                else:  # "fr"
                    den_w = cp.sum(A_d[r_i] * Z_rows, axis=1) * den_scale
                S_den = cpx_sparse.csr_matrix(
                    (den_w, (row_idx_b, col_idx_b)),
                    shape=(numerator.shape[0], nnz_b),
                )
                den_contrib = S_den @ Z_rows

            # Every allocation for this batch succeeded → commit atomically.
            numerator += num_contrib
            if masked:
                denominator += den_contrib

            batch_start = batch_end

        except _retryable as exc:
            # Drop this attempt's large temporaries, return cached blocks to the
            # driver so the out-of-pool cuBLAS/cuSPARSE workspace has room, then
            # retry the *same* columns at half the width.
            Z_u = Z_rows = num_contrib = den_contrib = S_b = None
            mempool.free_all_blocks()
            if bc <= 1:
                try:
                    free_b, total_b = cp.cuda.runtime.memGetInfo()
                except Exception:
                    free_b = total_b = -1
                print(
                    f"[shard-diag] FAILED at bc=1 device={device_id} div={divergence} "
                    f"masked={masked} n_ucols={n_ucols} batch={batch_start}:{batch_end} "
                    f"free={free_b/1e9:.2f}GB/{total_b/1e9:.2f}GB "
                    f"err={type(exc).__name__}: {exc}",
                    file=sys.stderr, flush=True,
                )
                raise
            new_bc = max(1, bc // 2)
            print(
                f"[shard-diag] shrinking batch device={device_id} masked={masked} "
                f"{bc}->{new_bc} at batch_start={batch_start} ({type(exc).__name__})",
                file=sys.stderr, flush=True,
            )
            bc = new_bc

    cp.cuda.Device(device_id).synchronize()
    return cp.asnumpy(numerator), (cp.asnumpy(denominator) if masked else None)


def _sharded_factor_update(
    shards: List[cpx_sparse.coo_matrix],
    device_ids: List[int],
    core: cp.ndarray,
    factors: List[cp.ndarray],
    mode: int,
    shape: Tuple[int, ...],
    divergence: str,
    epsilon: float,
    batch_cols: Optional[int],
    verbose: bool,
    subsample_frac: float = 1.0,
    iter_seed: Optional[int] = None,
    pool: Optional[ThreadPoolExecutor] = None,
    masked: bool = False,
) -> cp.ndarray:
    """
    Orchestrate multi-GPU factor numerator computation and reduce on CPU.

    Full objective: the denominator is computed once on the primary device
    (no NNZ access). Masked/completion objective: the denominator depends on
    the observed entries, so each shard returns a partial denominator that is
    reduced on CPU alongside the partial numerators.
    """
    primary = device_ids[0]

    # Serialize to CPU only when non-primary shards need it; for single-GPU
    # or the primary shard, pass the GPU arrays directly so cp.asarray is a
    # no-op and the GPU→CPU→GPU round-trip is avoided.
    if len(device_ids) > 1:
        core_buf: Any = cp.asnumpy(core)
        factors_buf: Any = [cp.asnumpy(f) for f in factors]
    else:
        core_buf = core
        factors_buf = factors
    A_primary = factors[mode]

    # Denominator — analytical (full objective only); masked accumulates per-shard.
    denominator = None
    if not masked:
        with cp.cuda.Device(primary):
            if divergence == "kl":
                den_row = _tucker_den_row_full(core, factors, mode, epsilon=epsilon)
                denominator = den_row[None, :]
            else:
                Gram = _tucker_gram_ZtZ(core, factors, mode, epsilon=epsilon)
                denominator = A_primary @ Gram
                denominator = cp.clip(denominator, a_min=epsilon, a_max=None)

    # Parallel partial numerators (+ denominators when masked)
    partials: List[Optional[Tuple[np.ndarray, Optional[np.ndarray]]]] = [None] * len(device_ids)
    _own_pool = pool is None
    _pool = ThreadPoolExecutor(max_workers=len(device_ids)) if _own_pool else pool
    try:
        futures: Dict = {
            _pool.submit(
                _partial_numerator_for_shard,
                shard=shards[k],
                core_np=core if k == 0 else core_buf,
                factors_np=factors if k == 0 else factors_buf,
                mode=mode,
                shape=shape,
                divergence=divergence,
                epsilon=epsilon,
                batch_cols=batch_cols,
                device_id=device_ids[k],
                subsample_frac=subsample_frac,
                # CHANGED (Task 2): shards no longer need distinct per-shard seeds
                # (each shard has its own construction-time shuffle); the raw
                # iteration number selects the rotating window on every shard.
                iteration=iter_seed,
                masked=masked,
            ): k
            for k in range(len(device_ids))
        }
        for fut in as_completed(futures):
            partials[futures[fut]] = fut.result()
    finally:
        if _own_pool:
            _pool.shutdown(wait=True)

    # CPU reduce + MU update
    numerator_np = np.add.reduce([p[0] for p in partials])

    with cp.cuda.Device(primary):
        numerator = cp.asarray(numerator_np)
        numerator = cp.clip(numerator, a_min=epsilon, a_max=None)
        if masked:
            denominator_np = np.add.reduce([p[1] for p in partials])
            denominator = cp.clip(cp.asarray(denominator_np), a_min=epsilon, a_max=None)
        A_new = A_primary * (numerator / (denominator + epsilon))
        A_new = cp.clip(A_new, a_min=epsilon, a_max=None)

    return A_new


# ---------------------------------------------------------------------------
# Core update — per-shard partial numerator
# ---------------------------------------------------------------------------

def _partial_core_num_for_shard(
    shard: cpx_sparse.coo_matrix,
    core_np: Union[np.ndarray, Any],
    factors_np: List[Union[np.ndarray, Any]],
    shape: Tuple[int, ...],
    divergence: str,
    epsilon: float,
    batch_rhat: Optional[int],
    batch_num: Optional[int],
    device_id: int,
    subsample_frac: float = 1.0,
    iteration: Optional[int] = None,
    masked: bool = False,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Compute the partial core-numerator contribution from a single NNZ shard.

    KL path: Pass 1 computes ``w = x / r̂``; Pass 2 accumulates outer products.
    FR path: Single pass with ``w = x`` directly.

    When ``subsample_frac < 1.0`` the shard's NNZ is subsampled before both
    passes, so the two-pass KL structure operates on the same sampled subset
    (the window selected by *iteration*; see ``_apply_subsample``).

    ``core_np`` and ``factors_np`` may be NumPy arrays *or* CuPy arrays
    already resident on ``device_id``.  ``cp.asarray`` is a no-op in the
    latter case, so no CPU round-trip occurs for the primary shard.

    When ``masked`` is True the (observed-only) denominator accumulator is also
    returned: KL uses unit weights (sum of outer products over observed entries)
    and FR uses ``x̂`` weights, since the masked-objective denominator cannot be
    computed analytically.

    Returns
    -------
    (partial_num, partial_den) : np.ndarray of shape ``core.shape`` each.
        ``partial_den`` is ``None`` when ``masked`` is False.
    """
    cp.cuda.Device(device_id).use()
    core_d = cp.asarray(core_np)
    factors_d = [cp.asarray(f) for f in factors_np]
    N = len(shape)

    flat, xvals = _blocked_coo_to_flat_indices(shard, shape)
    nnz = int(flat.size)

    if nnz == 0:
        zero = cp.asnumpy(cp.zeros_like(core_d))
        return zero, (zero.copy() if masked else None)

    if subsample_frac < 1.0:
        flat, xvals = _apply_subsample(flat, xvals, subsample_frac, iteration)
        nnz = int(flat.size)

    # Rescale the masked denominator weights to match the 1/frac rescaling that
    # _apply_subsample applied to xvals (the numerator), so the MU ratio is unbiased.
    den_scale = (1.0 / subsample_frac) if subsample_frac < 1.0 else 1.0

    idxs = _unravel_flat_indices_C(flat, shape)
    Num = cp.zeros_like(core_d)
    Den = cp.zeros_like(core_d) if masked else None

    # Estimate AFTER NNZ bookkeeping is live so the free-memory snapshot is accurate.
    if batch_rhat is None:
        batch_rhat = int(_estimate_batch_rhat_for_tensordot(core_d, factors_d))
    if batch_num is None:
        batch_num = int(_estimate_batch_num_for_outer(core_d, factors_d))

    if divergence == "kl":
        # Pass 1: w = x / r̂
        w_all = cp.empty_like(xvals)
        for start in range(0, nnz, batch_rhat):
            end = min(start + batch_rhat, nnz)
            mats = [factors_d[n][idxs[n][start:end]] for n in range(N)]
            r_hat = _rhat_from_factor_rows_sequential(core_d, mats, epsilon=epsilon)
            w_all[start:end] = xvals[start:end] / r_hat
        # Pass 2: accumulate outer products
        for start in range(0, nnz, batch_num):
            end = min(start + batch_num, nnz)
            mats = [factors_d[n][idxs[n][start:end]] for n in range(N)]
            _accumulate_core_num_outer(Num, w_all[start:end], mats)
            if masked:
                # KL masked denominator: unit-weighted outer products over observed entries.
                _accumulate_core_num_outer(Den, cp.full(end - start, den_scale, dtype=core_d.dtype), mats)
    else:  # "fr" — single pass
        for start in range(0, nnz, batch_num):
            end = min(start + batch_num, nnz)
            mats = [factors_d[n][idxs[n][start:end]] for n in range(N)]
            _accumulate_core_num_outer(Num, xvals[start:end], mats)
            if masked:
                # FR masked denominator: x̂-weighted outer products over observed entries.
                xhat_b = _rhat_from_factor_rows_sequential(core_d, mats, epsilon=epsilon)
                _accumulate_core_num_outer(Den, xhat_b * den_scale, mats)

    cp.cuda.Device(device_id).synchronize()
    return cp.asnumpy(Num), (cp.asnumpy(Den) if masked else None)


def _sharded_core_update(
    shards: List[cpx_sparse.coo_matrix],
    device_ids: List[int],
    core: cp.ndarray,
    factors: List[cp.ndarray],
    shape: Tuple[int, ...],
    divergence: str,
    epsilon: float,
    batch_rhat: Optional[int],
    batch_num: Optional[int],
    verbose: bool,
    subsample_frac: float = 1.0,
    iter_seed: Optional[int] = None,
    pool: Optional[ThreadPoolExecutor] = None,
    masked: bool = False,
) -> cp.ndarray:
    """
    Orchestrate multi-GPU core numerator computation and reduce on CPU.

    Full objective: the denominator (KL: column-sum outer product; FR: Gram
    contractions) is computed once on the primary device. Masked/completion
    objective: each shard returns a partial denominator (observed-only) that is
    reduced on CPU alongside the partial numerators.
    """
    primary = device_ids[0]
    N = len(shape)

    # Serialize to CPU only when non-primary shards need it; pass GPU arrays
    # for the primary shard (k=0) so cp.asarray is a no-op there.
    if len(device_ids) > 1:
        core_buf: Any = cp.asnumpy(core)
        factors_buf: Any = [cp.asnumpy(f) for f in factors]
    else:
        core_buf = core
        factors_buf = factors

    partials: List[Optional[Tuple[np.ndarray, Optional[np.ndarray]]]] = [None] * len(device_ids)
    _own_pool = pool is None
    _pool = ThreadPoolExecutor(max_workers=len(device_ids)) if _own_pool else pool
    try:
        futures: Dict = {
            _pool.submit(
                _partial_core_num_for_shard,
                shard=shards[k],
                core_np=core if k == 0 else core_buf,
                factors_np=factors if k == 0 else factors_buf,
                shape=shape,
                divergence=divergence,
                epsilon=epsilon,
                batch_rhat=batch_rhat,
                batch_num=batch_num,
                device_id=device_ids[k],
                subsample_frac=subsample_frac,
                iteration=iter_seed,  # CHANGED (Task 2): window index, not a per-shard seed
                masked=masked,
            ): k
            for k in range(len(device_ids))
        }
        for fut in as_completed(futures):
            partials[futures[fut]] = fut.result()
    finally:
        if _own_pool:
            _pool.shutdown(wait=True)

    Num_np = np.add.reduce([p[0] for p in partials])

    with cp.cuda.Device(primary):
        Num = cp.asarray(Num_np)

        if masked:
            # Masked objective: denominator is the reduced observed-only accumulator
            # (Num/Den both summed over observed entries), for both KL and FR.
            Den_np = np.add.reduce([p[1] for p in partials])
            Num = cp.clip(Num, a_min=epsilon, a_max=None)
            Den = cp.asarray(Den_np)
            core_new = core * (Num / (Den + epsilon))
            core_new = cp.clip(core_new, a_min=epsilon, a_max=None)
        elif divergence == "kl":
            sums = [
                cp.clip(cp.sum(factors[n], axis=0), a_min=epsilon, a_max=None)
                for n in range(N)
            ]
            core_new = core * (Num + epsilon)
            for n in range(N):
                shp = [1] * N
                shp[n] = int(sums[n].shape[0])
                core_new = core_new / sums[n].reshape(tuple(shp))
            core_new = cp.clip(core_new, a_min=epsilon, a_max=None)
        else:  # "fr"
            Num = cp.clip(Num, a_min=epsilon, a_max=None)
            grams = [factors[n].T @ factors[n] for n in range(N)]
            Den = _core_multilinear_grams(core, grams, epsilon=epsilon)
            core_new = core * (Num / (Den + epsilon))

    return core_new


# ---------------------------------------------------------------------------
# Error functions — per-shard partial scalars
# ---------------------------------------------------------------------------

def _partial_kl_error_for_shard(
    shard: cpx_sparse.coo_matrix,
    core_np: np.ndarray,
    factors_np: List[np.ndarray],
    shape: Tuple[int, ...],
    epsilon: float,
    batch_rhat: Optional[int],
    device_id: int,
    subsample_frac: float = 1.0,
    iteration: Optional[int] = None,
) -> Tuple[float, float, float]:
    """
    Compute partial KL error scalars from a single NNZ shard.

    Returns
    -------
    (kl_pos, sum_R_nz, sum_X) from this shard's NNZ contribution.
    """
    cp.cuda.Device(device_id).use()
    core_d = cp.asarray(core_np)
    factors_d = [cp.asarray(f) for f in factors_np]
    N = len(shape)

    if batch_rhat is None:
        batch_rhat = int(_estimate_batch_rhat_for_tensordot(core_d, factors_d))

    flat, x_nz = _blocked_coo_to_flat_indices(shard, shape)
    nnz = int(flat.size)

    if nnz == 0:
        return 0.0, 0.0, 0.0

    if subsample_frac < 1.0:
        flat, x_nz = _apply_subsample(flat, x_nz, subsample_frac, iteration)
        nnz = int(flat.size)

    x_nz = cp.clip(x_nz.astype(core_d.dtype), a_min=epsilon, a_max=None)
    idxs = _unravel_flat_indices_C(flat, shape)

    r_nz = cp.empty_like(x_nz)
    for start in range(0, nnz, batch_rhat):
        end = min(start + batch_rhat, nnz)
        mats = [factors_d[n][idxs[n][start:end]] for n in range(N)]
        r_nz[start:end] = _rhat_from_factor_rows_sequential(core_d, mats, epsilon=epsilon)
    r_nz = cp.clip(r_nz, a_min=epsilon, a_max=None)

    term_pos = x_nz * cp.log(x_nz / r_nz) - x_nz + r_nz
    kl_pos = float(cp.sum(term_pos).get())
    sum_R_nz = float(cp.sum(r_nz).get())
    sum_X = float(cp.sum(x_nz).get())

    cp.cuda.Device(device_id).synchronize()
    return kl_pos, sum_R_nz, sum_X


def _sharded_kl_error(
    shards: List[cpx_sparse.coo_matrix],
    device_ids: List[int],
    core: cp.ndarray,
    factors: List[cp.ndarray],
    shape: Tuple[int, ...],
    epsilon: float,
    batch_rhat: Optional[int],
    subsample_frac: float = 1.0,
    iter_seed: Optional[int] = None,
    pool: Optional[ThreadPoolExecutor] = None,
    masked: bool = False,
) -> cp.ndarray:
    """
    Compute relative KL error with sharded NNZ; returns a scalar CuPy array
    on the primary device matching the return type of ``kl_compute_errors_largedim``.

    When ``masked`` is True the zero-entry contribution (sum_R - sum_R_nz) is
    dropped, so the metric reflects the observed-only / completion objective.
    """
    primary = device_ids[0]
    core_np = cp.asnumpy(core)
    factors_np = [cp.asnumpy(f) for f in factors]

    results: List[Optional[Tuple[float, float, float]]] = [None] * len(device_ids)
    _own_pool = pool is None
    _pool = ThreadPoolExecutor(max_workers=len(device_ids)) if _own_pool else pool
    try:
        futures: Dict = {
            _pool.submit(
                _partial_kl_error_for_shard,
                shard=shards[k],
                core_np=core_np,
                factors_np=factors_np,
                shape=shape,
                epsilon=epsilon,
                batch_rhat=batch_rhat,
                device_id=device_ids[k],
                subsample_frac=subsample_frac,
                iteration=iter_seed,  # CHANGED (Task 2): window index, not a per-shard seed
            ): k
            for k in range(len(device_ids))
        }
        for fut in as_completed(futures):
            results[futures[fut]] = fut.result()
    finally:
        if _own_pool:
            _pool.shutdown(wait=True)

    kl_pos_total = sum(r[0] for r in results)
    sum_R_nz_total = sum(r[1] for r in results)
    sum_X_total = sum(r[2] for r in results)

    if masked:
        # Observed-only objective: drop the implicit-zero contribution.
        kl_total = kl_pos_total
    else:
        with cp.cuda.Device(primary):
            sum_R = float(_tucker_sum_all_entries(core, factors, epsilon=epsilon).get())
        kl_zero = sum_R - sum_R_nz_total
        kl_total = kl_pos_total + kl_zero
    rel_kl = kl_total / max(sum_X_total, float(epsilon))

    with cp.cuda.Device(primary):
        return cp.asarray(rel_kl, dtype=core.dtype)


def _partial_fr_error_for_shard(
    shard: cpx_sparse.coo_matrix,
    core_np: np.ndarray,
    factors_np: List[np.ndarray],
    shape: Tuple[int, ...],
    epsilon: float,
    batch_rhat: Optional[int],
    device_id: int,
    subsample_frac: float = 1.0,
    iteration: Optional[int] = None,
    masked: bool = False,
) -> Tuple[float, float, float]:
    """
    Compute partial Frobenius error scalars from a single NNZ shard.

    Returns
    -------
    (norm_X_sq, inner_prod, residual_sq) from this shard's NNZ contribution.
    ``inner_prod`` is 0 when ``masked`` (the full ‖X̂‖² term is not used);
    ``residual_sq`` (= sum (x - x̂)²) is 0 when not ``masked``.
    """
    cp.cuda.Device(device_id).use()
    core_d = cp.asarray(core_np)
    factors_d = [cp.asarray(f) for f in factors_np]
    N = len(shape)

    if batch_rhat is None:
        batch_rhat = int(_estimate_batch_rhat_for_tensordot(core_d, factors_d))

    flat, x_nz = _blocked_coo_to_flat_indices(shard, shape)
    nnz = int(flat.size)

    if nnz == 0:
        return 0.0, 0.0, 0.0

    if subsample_frac < 1.0:
        flat, x_nz = _apply_subsample(flat, x_nz, subsample_frac, iteration)
        nnz = int(flat.size)

    x_nz = cp.clip(x_nz.astype(core_d.dtype), a_min=0.0, a_max=None)
    idxs = _unravel_flat_indices_C(flat, shape)

    norm_X_sq = float(cp.sum(x_nz * x_nz).get())

    inner_prod_d = cp.asarray(0.0, dtype=core_d.dtype)
    residual_sq_d = cp.asarray(0.0, dtype=core_d.dtype)
    for start in range(0, nnz, batch_rhat):
        end = min(start + batch_rhat, nnz)
        mats = [factors_d[n][idxs[n][start:end]] for n in range(N)]
        xhat_b = _rhat_from_factor_rows_sequential(core_d, mats, epsilon=epsilon)
        if masked:
            residual_sq_d += cp.sum((x_nz[start:end] - xhat_b) ** 2)
        else:
            inner_prod_d += cp.sum(x_nz[start:end] * xhat_b)

    inner_prod = float(inner_prod_d.get())
    residual_sq = float(residual_sq_d.get())
    cp.cuda.Device(device_id).synchronize()
    return norm_X_sq, inner_prod, residual_sq


def _sharded_fr_error(
    shards: List[cpx_sparse.coo_matrix],
    device_ids: List[int],
    core: cp.ndarray,
    factors: List[cp.ndarray],
    shape: Tuple[int, ...],
    epsilon: float,
    batch_rhat: Optional[int],
    subsample_frac: float = 1.0,
    iter_seed: Optional[int] = None,
    pool: Optional[ThreadPoolExecutor] = None,
    masked: bool = False,
) -> cp.ndarray:
    """
    Compute relative Frobenius error with sharded NNZ; returns a scalar CuPy
    array on the primary device.

    Full objective uses ‖X - X̂‖²_F = ‖X‖² + ‖X̂‖² - 2⟨X, X̂⟩, with ‖X̂‖²
    computed analytically on the primary device (no NNZ). The masked/completion
    objective uses the observed-only relative RMSE sqrt(sum_Ω (x - x̂)²) / ‖X‖.
    """
    primary = device_ids[0]
    N = len(factors)
    core_np = cp.asnumpy(core)
    factors_np = [cp.asnumpy(f) for f in factors]

    results: List[Optional[Tuple[float, float, float]]] = [None] * len(device_ids)
    _own_pool = pool is None
    _pool = ThreadPoolExecutor(max_workers=len(device_ids)) if _own_pool else pool
    try:
        futures: Dict = {
            _pool.submit(
                _partial_fr_error_for_shard,
                shard=shards[k],
                core_np=core_np,
                factors_np=factors_np,
                shape=shape,
                epsilon=epsilon,
                batch_rhat=batch_rhat,
                device_id=device_ids[k],
                subsample_frac=subsample_frac,
                iteration=iter_seed,  # CHANGED (Task 2): window index, not a per-shard seed
                masked=masked,
            ): k
            for k in range(len(device_ids))
        }
        for fut in as_completed(futures):
            results[futures[fut]] = fut.result()
    finally:
        if _own_pool:
            _pool.shutdown(wait=True)

    norm_X_sq_total = sum(r[0] for r in results)
    inner_prod_total = sum(r[1] for r in results)
    residual_sq_total = sum(r[2] for r in results)

    norm_X = math.sqrt(max(norm_X_sq_total, float(epsilon)))

    if masked:
        # Observed-only relative RMSE.
        result = math.sqrt(max(residual_sq_total, 0.0)) / norm_X
    else:
        with cp.cuda.Device(primary):
            grams = [factors[n].T @ factors[n] for n in range(N)]
            Den = _core_multilinear_grams(core, grams, epsilon=epsilon)
            norm_Xhat_sq = float(cp.sum(core * Den).get())

        if norm_X_sq_total == 0.0:
            result = math.sqrt(max(norm_Xhat_sq, float(epsilon))) / norm_X
        else:
            residual_sq = max(norm_X_sq_total + norm_Xhat_sq - 2.0 * inner_prod_total, 0.0)
            result = math.sqrt(residual_sq) / norm_X

    with cp.cuda.Device(primary):
        return cp.asarray(result, dtype=core.dtype)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

class ShardedSparseTensor:
    """
    Wraps a CuPy COO sparse tensor and pre-shards its NNZ across CUDA devices.

    When ``n_shards == 1`` every method delegates to the corresponding
    single-GPU function in ``distance.py`` with zero overhead.

    Stochastic subsampling
    ----------------------
    Set ``subsample_frac < 1.0`` at construction time to enable per-iteration
    NNZ sampling on the multi-GPU path.  Each shard's NNZ arrays are shuffled
    once at construction (seeded with ``subsample_seed + shard_k``); each
    iteration then samples by taking a contiguous rotating window — zero
    per-iteration index allocation and deterministic given
    (subsample_seed, iteration).  Call ``set_iter_seed(iteration)`` once at
    the start of each training loop iteration so the window advances.  The
    wrapper functions (``make_sharded_*``) require no changes.

    Usage
    -----
    ::

        sst = ShardedSparseTensor.from_coo(
            coo, orig_shape, device_ids=[0, 1, 2, 3], subsample_frac=0.2
        )

        # In training loop:
        sst.set_iter_seed(iteration)

        # Factor / core / error calls work unchanged:
        A_new   = sst.kl_factor_update(core=core, factors=factors, mode=0, shape=shape)
        core_new = sst.kl_core_update(shape=shape, core=core, factors=factors)
        err      = sst.kl_compute_errors(shape=shape, core=core, factors=factors)

    Attributes
    ----------
    full_tensor : cpx_sparse.coo_matrix  — full COO on device_ids[0]
    orig_shape  : tuple[int, ...]        — original N-D tensor shape
    device_ids  : list[int]              — one per shard; [0] is primary
    shards      : list[coo_matrix]       — shards[k] lives on device_ids[k]
    n_shards    : int
    subsample_frac : float               — NNZ fraction; 1.0 = exact
    """

    def __init__(
        self,
        full_tensor: cpx_sparse.coo_matrix,
        orig_shape: Tuple[int, ...],
        device_ids: List[int],
        shards: List[cpx_sparse.coo_matrix],
        subsample_frac: float = 1.0,
        masked: bool = False,
    ) -> None:
        self.full_tensor = full_tensor
        self.orig_shape = orig_shape
        self.device_ids = device_ids
        self.shards = shards
        self.n_shards = len(device_ids)
        self.subsample_frac = float(subsample_frac)
        # Optimisation objective: when True, fit only observed entries
        # (weighted/completion objective), mirroring cfg.exp.objective="masked".
        self.masked = bool(masked)
        self._iter_seed: Optional[int] = None
        # Persistent pool: threads (and their cuBLAS handles) live for the
        # lifetime of this object.  Re-creating a pool each call causes
        # thread-ID recycling in Python 3.13 which leaves stale cuBLAS
        # handles → CUBLAS_STATUS_NOT_INITIALIZED after ~40 iterations.
        self._pool: Optional[ThreadPoolExecutor] = (
            ThreadPoolExecutor(max_workers=self.n_shards)
            if self.n_shards > 1 else None
        )
        # Load cuBLAS GEMM kernels into every device context up front, on this
        # (single) thread, before any parallel factor update can run.
        self._warm_up_gpus()

    def _warm_up_gpus(self) -> None:
        """Force cuBLAS GEMM kernel modules to load on each device, serially.

        CUDA 12 defaults to lazy module loading: a context does not load the
        cuBLAS GEMM cubins until its first matmul. The *full* objective happens
        to issue that first call single-threaded on the primary device (the
        analytic denominator in ``_sharded_factor_update``); the *masked*
        objective skips that step, so all worker threads issue their first
        cuBLAS call concurrently on their own devices in iteration 1. Those
        simultaneous one-time loads race and intermittently surface as
        ``CUBLAS_STATUS_NOT_INITIALIZED`` from ``gemmStridedBatchedEx`` — even
        with plenty of free VRAM. Touching each device here (both a plain and a
        batched matmul, in the dtypes we use) serialises the load so the
        workers find the kernels already resident.
        """
        for did in self.device_ids:
            with cp.cuda.Device(did):
                for dtype in (cp.float32, cp.float64):
                    a2 = cp.ones((2, 2), dtype=dtype)
                    a3 = cp.ones((2, 2, 2), dtype=dtype)  # -> gemmStridedBatchedEx
                    _ = a2 @ a2
                    _ = a3 @ a3
                cp.cuda.Device(did).synchronize()

    def __del__(self) -> None:
        pool = getattr(self, "_pool", None)
        if pool is not None:
            pool.shutdown(wait=False)
            self._pool = None

    def set_iter_seed(self, iteration: int) -> None:
        """
        Record the current *iteration*, which selects this iteration's
        subsample window on every shard.

        Call this once at the top of each training loop iteration.

        CHANGED (Task 2): previously stored ``iteration * n_shards`` so each
        shard could derive a distinct RNG seed (``+ k``).  Shards now carry
        their own construction-time shuffle, so the raw iteration number is
        all that's needed; per-shard windows are decorrelated by the per-shard
        shuffle seeds, and the stride-by-n_sample window walk guarantees
        epoch-like coverage of each shard (a stride of ``n_shards·n_sample``
        could alias with the shard length and revisit the same window forever,
        e.g. frac=0.5 with 2 shards).
        """
        self._iter_seed = int(iteration)
        for did in self.device_ids:
            with cp.cuda.Device(did):
                cp.get_default_memory_pool().free_all_blocks()

    @classmethod
    def from_coo(
        cls,
        coo: cpx_sparse.coo_matrix,
        orig_shape: Tuple[int, ...],
        device_ids: Optional[List[int]] = None,
        subsample_frac: float = 1.0,
        masked: bool = False,
        subsample_seed: int = 0,
    ) -> "ShardedSparseTensor":
        """
        Build a ``ShardedSparseTensor`` from an existing CuPy COO matrix.

        If *device_ids* is ``None`` or length 1, a single-shard object is
        returned and all calls delegate to single-GPU functions.

        Parameters
        ----------
        coo            : full COO on the primary device
        orig_shape     : original N-D tensor shape
        device_ids     : CUDA ordinals; ``None`` → single-GPU fallback
        subsample_frac : NNZ fraction for stochastic updates (multi-GPU only)
        masked         : fit only observed entries (completion objective)
        subsample_seed : base seed for the one-time per-shard NNZ shuffle that
                         backs contiguous-window subsampling (shard k uses
                         ``subsample_seed + k``); ignored at
                         ``subsample_frac == 1.0``.  Typically
                         ``cfg.exp.random_state``.
        """
        coo_coo = coo.tocoo()

        if device_ids is None or len(device_ids) <= 1:
            primary = int(coo_coo.row.device) if hasattr(coo_coo.row, "device") else 0
            return cls(coo_coo, orig_shape, [primary], [coo_coo],
                       subsample_frac=subsample_frac, masked=masked)

        nnz = int(coo_coo.row.size)
        n = len(device_ids)
        boundaries = [int(round(nnz * k / n)) for k in range(n + 1)]

        # CHANGED (Task 2): shuffle each shard once at build time (host-side, free —
        # the slices round-trip through the CPU anyway) so per-iteration subsampling
        # is a contiguous window instead of a fresh device-side permutation(nnz).
        # Exact runs (frac == 1.0) keep the original NNZ order.
        _shuffle = subsample_frac < 1.0
        shards = [
            _build_shard(
                coo_coo, boundaries[k], boundaries[k + 1], device_ids[k],
                shuffle_seed=(int(subsample_seed) + k) if _shuffle else None,
            )
            for k in range(n)
        ]

        return cls(coo_coo, orig_shape, list(device_ids), shards,
                   subsample_frac=subsample_frac, masked=masked)

    # ------------------------------------------------------------------
    # Factor update methods
    # ------------------------------------------------------------------

    def kl_factor_update(
        self,
        core: cp.ndarray,
        factors: List[cp.ndarray],
        mode: int,
        shape: Tuple[int, ...],
        thread_budget=None,
        epsilon: float = 1e-12,
        batch_cols: Optional[int] = None,
        verbose: bool = False,
    ) -> cp.ndarray:
        """KL factor update; single-shard delegates to ``kl_factor_update_largedim``."""
        if self.n_shards == 1:
            return kl_factor_update_largedim(
                vec_tensor=self.full_tensor,
                core=core, factors=factors, mode=mode, shape=shape,
                thread_budget=thread_budget, epsilon=epsilon,
                batch_cols=batch_cols, verbose=verbose, masked=self.masked,
            )
        return _sharded_factor_update(
            shards=self.shards, device_ids=self.device_ids,
            core=core, factors=factors, mode=mode, shape=shape,
            divergence="kl", epsilon=epsilon, batch_cols=batch_cols, verbose=verbose,
            subsample_frac=self.subsample_frac, iter_seed=self._iter_seed,
            pool=self._pool, masked=self.masked,
        )

    def fr_factor_update(
        self,
        core: cp.ndarray,
        factors: List[cp.ndarray],
        mode: int,
        shape: Tuple[int, ...],
        thread_budget=None,
        epsilon: float = 1e-12,
        batch_cols: Optional[int] = None,
        verbose: bool = False,
    ) -> cp.ndarray:
        """FR factor update; single-shard delegates to ``fr_factor_update_largedim``."""
        if self.n_shards == 1:
            return fr_factor_update_largedim(
                vec_tensor=self.full_tensor,
                core=core, factors=factors, mode=mode, shape=shape,
                thread_budget=thread_budget, epsilon=epsilon,
                batch_cols=batch_cols, verbose=verbose, masked=self.masked,
            )
        return _sharded_factor_update(
            shards=self.shards, device_ids=self.device_ids,
            core=core, factors=factors, mode=mode, shape=shape,
            divergence="fr", epsilon=epsilon, batch_cols=batch_cols, verbose=verbose,
            subsample_frac=self.subsample_frac, iter_seed=self._iter_seed,
            pool=self._pool, masked=self.masked,
        )

    # ------------------------------------------------------------------
    # Core update methods
    # ------------------------------------------------------------------

    def kl_core_update(
        self,
        shape: Tuple[int, ...],
        core: cp.ndarray,
        factors: List[cp.ndarray],
        modes=None,
        thread_budget=None,
        epsilon: float = 1e-12,
        batch_rhat: Optional[int] = None,
        batch_num: Optional[int] = None,
        verbose: bool = False,
    ) -> cp.ndarray:
        """KL core update; single-shard delegates to ``kl_core_update_largedim``."""
        if self.n_shards == 1:
            return kl_core_update_largedim(
                vec_tensor=self.full_tensor,
                shape=shape, core=core, factors=factors, modes=modes,
                thread_budget=thread_budget, epsilon=epsilon,
                batch_rhat=batch_rhat, batch_num=batch_num, verbose=verbose,
                masked=self.masked,
            )
        return _sharded_core_update(
            shards=self.shards, device_ids=self.device_ids,
            core=core, factors=factors, shape=shape,
            divergence="kl", epsilon=epsilon,
            batch_rhat=batch_rhat, batch_num=batch_num, verbose=verbose,
            subsample_frac=self.subsample_frac, iter_seed=self._iter_seed,
            pool=self._pool, masked=self.masked,
        )

    def fr_core_update(
        self,
        shape: Tuple[int, ...],
        core: cp.ndarray,
        factors: List[cp.ndarray],
        modes=None,
        thread_budget=None,
        epsilon: float = 1e-12,
        batch_num: Optional[int] = None,
        verbose: bool = False,
    ) -> cp.ndarray:
        """FR core update; single-shard delegates to ``fr_core_update_largedim``."""
        if self.n_shards == 1:
            return fr_core_update_largedim(
                vec_tensor=self.full_tensor,
                shape=shape, core=core, factors=factors, modes=modes,
                thread_budget=thread_budget, epsilon=epsilon,
                batch_num=batch_num, verbose=verbose, masked=self.masked,
            )
        return _sharded_core_update(
            shards=self.shards, device_ids=self.device_ids,
            core=core, factors=factors, shape=shape,
            divergence="fr", epsilon=epsilon,
            batch_rhat=None, batch_num=batch_num, verbose=verbose,
            subsample_frac=self.subsample_frac, iter_seed=self._iter_seed,
            pool=self._pool, masked=self.masked,
        )

    # ------------------------------------------------------------------
    # Error computation methods
    # ------------------------------------------------------------------

    def kl_compute_errors(
        self,
        shape: Tuple[int, ...],
        core: cp.ndarray,
        factors: List[cp.ndarray],
        thread_budget=None,
        epsilon: float = 1e-12,
        batch_rhat: Optional[int] = None,
        verbose: bool = False,
    ) -> cp.ndarray:
        """KL error; single-shard delegates to ``kl_compute_errors_largedim``."""
        if self.n_shards == 1:
            return kl_compute_errors_largedim(
                vec_tensor=self.full_tensor,
                shape=shape, core=core, factors=factors,
                thread_budget=thread_budget, epsilon=epsilon,
                batch_rhat=batch_rhat, verbose=verbose, masked=self.masked,
            )
        return _sharded_kl_error(
            shards=self.shards, device_ids=self.device_ids,
            core=core, factors=factors, shape=shape,
            epsilon=epsilon, batch_rhat=batch_rhat,
            subsample_frac=self.subsample_frac, iter_seed=self._iter_seed,
            pool=self._pool, masked=self.masked,
        )

    def fr_compute_errors(
        self,
        shape: Tuple[int, ...],
        core: cp.ndarray,
        factors: List[cp.ndarray],
        thread_budget=None,
        epsilon: float = 1e-12,
        batch_rhat: Optional[int] = None,
        verbose: bool = False,
    ) -> cp.ndarray:
        """FR error; single-shard delegates to ``fr_compute_errors_largedim``."""
        if self.n_shards == 1:
            return fr_compute_errors_largedim(
                vec_tensor=self.full_tensor,
                shape=shape, core=core, factors=factors,
                thread_budget=thread_budget, epsilon=epsilon,
                batch_rhat=batch_rhat, verbose=verbose, masked=self.masked,
            )
        return _sharded_fr_error(
            shards=self.shards, device_ids=self.device_ids,
            core=core, factors=factors, shape=shape,
            epsilon=epsilon, batch_rhat=batch_rhat,
            subsample_frac=self.subsample_frac, iter_seed=self._iter_seed,
            pool=self._pool, masked=self.masked,
        )


# ---------------------------------------------------------------------------
# Callable wrappers for routing injection
# ---------------------------------------------------------------------------

def make_sharded_kl_factor_update(sst: ShardedSparseTensor):
    """Callable matching ``kl_factor_update_largedim`` signature; routes through *sst*."""
    def _fn(vec_tensor, core, factors, mode, shape,
            thread_budget=None, epsilon=1e-12, batch_cols=None, verbose=False):
        return sst.kl_factor_update(
            core=core, factors=factors, mode=mode, shape=shape,
            thread_budget=thread_budget, epsilon=epsilon,
            batch_cols=batch_cols, verbose=verbose,
        )
    return _fn


def make_sharded_fr_factor_update(sst: ShardedSparseTensor):
    """Callable matching ``fr_factor_update_largedim`` signature; routes through *sst*."""
    def _fn(vec_tensor, core, factors, mode, shape,
            thread_budget=None, epsilon=1e-12, batch_cols=None, verbose=False):
        return sst.fr_factor_update(
            core=core, factors=factors, mode=mode, shape=shape,
            thread_budget=thread_budget, epsilon=epsilon,
            batch_cols=batch_cols, verbose=verbose,
        )
    return _fn


def make_sharded_kl_core_update(sst: ShardedSparseTensor):
    """Callable matching ``kl_core_update_largedim`` signature; routes through *sst*."""
    def _fn(vec_tensor, shape, core, factors, modes=None,
            thread_budget=None, epsilon=1e-12, batch_rhat=None, batch_num=None, verbose=False):
        return sst.kl_core_update(
            shape=shape, core=core, factors=factors, modes=modes,
            thread_budget=thread_budget, epsilon=epsilon,
            batch_rhat=batch_rhat, batch_num=batch_num, verbose=verbose,
        )
    return _fn


def make_sharded_fr_core_update(sst: ShardedSparseTensor):
    """Callable matching ``fr_core_update_largedim`` signature; routes through *sst*."""
    def _fn(vec_tensor, shape, core, factors, modes=None,
            thread_budget=None, epsilon=1e-12, batch_num=None, verbose=False):
        return sst.fr_core_update(
            shape=shape, core=core, factors=factors, modes=modes,
            thread_budget=thread_budget, epsilon=epsilon,
            batch_num=batch_num, verbose=verbose,
        )
    return _fn


def make_sharded_kl_compute_errors(sst: ShardedSparseTensor):
    """Callable matching ``kl_compute_errors_largedim`` signature; routes through *sst*."""
    def _fn(vec_tensor, shape, core, factors,
            thread_budget=None, epsilon=1e-12, batch_rhat=None, verbose=False):
        return sst.kl_compute_errors(
            shape=shape, core=core, factors=factors,
            thread_budget=thread_budget, epsilon=epsilon,
            batch_rhat=batch_rhat, verbose=verbose,
        )
    return _fn


def make_sharded_fr_compute_errors(sst: ShardedSparseTensor):
    """Callable matching ``fr_compute_errors_largedim`` signature; routes through *sst*."""
    def _fn(vec_tensor, shape, core, factors,
            thread_budget=None, epsilon=1e-12, batch_rhat=None, verbose=False):
        return sst.fr_compute_errors(
            shape=shape, core=core, factors=factors,
            thread_budget=thread_budget, epsilon=epsilon,
            batch_rhat=batch_rhat, verbose=verbose,
        )
    return _fn
