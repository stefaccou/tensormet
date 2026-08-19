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

``cfg.exp.max_nnz`` (hard global NNZ ceiling) is applied upstream in
tucker_tensor.py as an effective fraction, so the ``subsample_frac`` received
here may already embed it; per-shard sampling of that fraction sums to
~max_nnz across shards.

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
    coo_to_coords,
    coords_nnz,
    _build_mode_grouping,
    _estimate_batch_cols_for_Z,
    _unravel_cols_for_mode,
    ModeGrouping,
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
from tensormet.sparse_ops import (
    CoordCOO,
    _array_device_id,
    compute_Zcols_batch,
    safe_ravel,
    use_legacy_factor_batch,
    sampled_row_dots,
    group_batch_by_column,
    build_batch_csr_T,
    same_pattern_csr,
    spmm_T,
)
from tensormet.utils import make_lazy_cupy_pair

cp, cpx_sparse = make_lazy_cupy_pair()

# ---------------------------------------------------------------------------
# Internal utilities
# ---------------------------------------------------------------------------

def _build_coord_shard(
    src: CoordCOO,
    start: int,
    end: int,
    target_device: int,
    shuffle_seed: Optional[int] = None,
) -> CoordCOO:
    """Coordinate-backed counterpart of :func:`_build_shard`.

    Same slicing, shuffling and device-placement semantics; the NNZ payload is
    a ``(ndim, nnz)`` coordinate block instead of ``(row, col)``.
    """
    coords, data = src.coords, src.data
    source_device = _array_device_id(coords)

    if source_device is not None and source_device == target_device:
        with cp.cuda.Device(target_device):
            c, d = coords[:, start:end], data[start:end]
            if shuffle_seed is not None:
                perm = cp.asarray(
                    np.random.default_rng(int(shuffle_seed)).permutation(int(d.size))
                )
                c, d = c[:, perm], d[perm]
            else:
                # Slices are views into the full COO; copy so the parent NNZ can
                # be freed instead of being pinned alive (see _build_shard).
                c, d = c.copy(), d.copy()
            return CoordCOO(c, d, src.shape)

    coords_np = cp.asnumpy(coords[:, start:end])
    data_np = cp.asnumpy(data[start:end])
    if shuffle_seed is not None:
        perm = np.random.default_rng(int(shuffle_seed)).permutation(data_np.size)
        coords_np, data_np = coords_np[:, perm], data_np[perm]

    with cp.cuda.Device(target_device):
        return CoordCOO(cp.asarray(coords_np), cp.asarray(data_np), src.shape)


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

    CHANGED (2026-06-12 review, Task 7 — O-3): when *target_device* is the device
    the source COO already lives on (always the case for shard 0, the primary), the
    shard is built from device-local slices — no GPU→CPU→GPU round-trip. The slices
    are copied into fresh contiguous arrays so the parent COO can be released once
    ``from_coo`` drops its reference; otherwise a *view* into the full COO would pin
    the entire NNZ on the primary device, defeating the point of sharding. Only the
    genuine cross-device shards (k > 0) still round-trip through the host, because
    CuPy does not support direct cross-device tensor slicing. One-time cost at
    initialisation either way.

    When *shuffle_seed* is given (subsampling enabled), the slice is uniformly
    shuffled before the shard is built, so that the contiguous windows taken by
    ``_apply_subsample`` are uniform samples without replacement.  COO entry order
    carries no meaning for any downstream accumulation (sums are order-invariant;
    ``cp.unique`` re-sorts its input), so the shuffle is content-preserving. On the
    device-local path the permutation is still drawn host-side (deterministic, and
    identical in distribution to the cross-device shards), but only the index array
    crosses the bus — the shard data never leaves its device.
    """
    # Device-local short-circuit (shard 0, and any shard whose target matches source).
    source_device = _array_device_id(coo.row)
    if source_device is not None and source_device == target_device:
        with cp.cuda.Device(target_device):
            row, col, data = coo.row[start:end], coo.col[start:end], coo.data[start:end]
            if shuffle_seed is not None:
                perm = cp.asarray(
                    np.random.default_rng(int(shuffle_seed)).permutation(int(row.size))
                )
                # Fancy indexing already returns fresh contiguous arrays.
                row, col, data = row[perm], col[perm], data[perm]
            else:
                # Plain slices are views into the full COO; copy so the parent NNZ
                # arrays can be freed instead of being pinned alive by the views.
                row, col, data = row.copy(), col.copy(), data.copy()
            return cpx_sparse.coo_matrix((data, (row, col)), shape=coo.shape)

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
    idxs: List[cp.ndarray],
    vals: cp.ndarray,
    subsample_frac: float,
    iteration: Optional[int],
    rescale: bool = True,
) -> Tuple[List[cp.ndarray], cp.ndarray]:
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
    idxs, vals :
        Per-mode coordinate arrays and values from ``coo_to_coords`` on a
        shard built with ``shuffle_seed`` set (i.e. already in shuffled order).
        CHANGED: windows the N coordinate arrays rather than one flat index,
        which coordinate-backed shards never form. Same window arithmetic.
    subsample_frac :
        Fraction of NNZ to retain.  Must be < 1.0 (caller is responsible
        for not calling this at 1.0).
    iteration :
        Training-loop iteration number; selects the window.  ``None`` is
        treated as 0 (deterministic given (construction seed, iteration) —
        no RNG state survives between calls, so resumed runs draw the same
        windows as uninterrupted ones).
    rescale :
        When True (default) values are multiplied by ``1/frac`` so that a
        **linear** accumulation over the window is unbiased — correct for the
        MU numerators.  When False the raw windowed values are returned; the
        caller must instead weight the *summed* result by ``nnz/n_sample``.
        This is the unbiased path for **nonlinear** error terms
        (``x·log(x/r)``, ``x²``), where rescaling the values would inject a
        ``1/frac`` bias inside the nonlinearity (review finding I-1).

    Returns
    -------
    idxs_s, vals_s : windowed arrays (values rescaled iff ``rescale``).
    """
    nnz = int(vals.size)
    n_sample = max(1, int(round(subsample_frac * nnz)))
    start = (int(iteration or 0) * n_sample) % nnz
    end = start + n_sample

    def _window(a):
        if end <= nnz:
            return a[start:end]
        # wrap around the end of the shuffled sequence
        return cp.concatenate((a[start:], a[: end - nnz]))

    idxs_s = [_window(a) for a in idxs]
    vals_s = _window(vals)
    if not rescale:
        return idxs_s, vals_s
    scale = vals.dtype.type(1.0 / subsample_frac)
    return idxs_s, vals_s * scale


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
    grouping: Optional[ModeGrouping] = None,
    batch_sink: Optional[list] = None,
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

    ``grouping`` : ModeGrouping, optional
        CHANGED (2026-06-12 review, Task 3 — E-1/E-2/E-3): precomputed per-shard,
        per-mode NNZ grouping (built once on this device by the owning
        ShardedSparseTensor). When supplied, the per-iteration flat-index decode,
        ``cp.unique`` sort, and per-batch ``cp.where`` scan are skipped. Only
        passed on the exact (non-subsampling) path; ``None`` under subsampling.

    ``batch_sink`` : list, optional
        CHANGED (2026-06-12 review, Task 4): a one-element mutable list into which
        the realized ``batch_cols`` (after any OOM-retry halving) is written, so the
        owning ShardedSparseTensor can cache and reuse it next iteration. Left as a
        keyword to preserve the 2-tuple return for direct callers.

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

    other_modes = [m for m in range(len(shape)) if m != mode]
    numerator = cp.zeros_like(A_d)
    denominator = cp.zeros_like(A_d) if masked else None

    if grouping is not None:
        # Cached path (Task 3): NNZ already decoded, sorted and grouped by column
        # for this shard. No subsampling is active when a grouping is supplied.
        ucols = grouping.ucols
        segment_offsets = grouping.segment_offsets
        rows_sorted = grouping.rows_sorted
        vals_sorted = grouping.vals_sorted
        col_index = grouping.col_index
        inv = None
        den_scale = 1.0
        n_ucols = int(ucols.size)
        if n_ucols == 0:
            zero = cp.asnumpy(cp.zeros_like(A_d))
            return zero, (zero.copy() if masked else None)
    else:
        idxs, vals = coo_to_coords(shard, shape)

        if vals.size == 0:
            zero = cp.asnumpy(cp.zeros_like(A_d))
            return zero, (zero.copy() if masked else None)

        if subsample_frac < 1.0:
            idxs, vals = _apply_subsample(idxs, vals, subsample_frac, iteration)

        # Under subsampling the numerator's `vals` are already rescaled by 1/frac
        # (see _apply_subsample). The masked denominator weights (unit / x̂) must be
        # rescaled identically so the MU ratio stays unbiased; the factor cancels in
        # the ratio but only if both sides carry it.
        den_scale = (1.0 / subsample_frac) if subsample_frac < 1.0 else 1.0

        rows = idxs[mode]

        other_shape = tuple(shape[m] for m in other_modes)
        other_coords = [idxs[m] for m in other_modes]
        cols = safe_ravel(tuple(other_coords), other_shape, cp)

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
    _cublas_cls: tuple = ()
    try:
        _cublas_cls = (cp.cuda.cublas.CUBLASError,)
        _retryable.append(cp.cuda.cublas.CUBLASError)
    except AttributeError:  # pragma: no cover - cuBLAS error class always present in practice
        pass
    _retryable = tuple(_retryable)

    bc = int(batch_cols)
    batch_start = 0
    # Track whether (and why) the batch had to shrink during this call so the
    # caller can decide whether to persist the reduced width. A cuBLAS-only shrink
    # reflects transient out-of-pool workspace pressure (recover next iteration); a
    # genuine OutOfMemoryError reflects a stable VRAM ceiling (persist, as before).
    shrank_via_cublas = False
    shrank_via_oom = False
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

            # nnz entries belonging to these unique columns
            if grouping is not None:
                # E-2: contiguous slice of the column-grouped arrays — no scan.
                seg_lo = int(segment_offsets[batch_start])
                seg_hi = int(segment_offsets[batch_end])
                if seg_hi == seg_lo:
                    batch_start = batch_end
                    continue
                r_i = rows_sorted[seg_lo:seg_hi]
                v_i = vals_sorted[seg_lo:seg_hi]
                u_i = col_index[seg_lo:seg_hi] - batch_start
            else:
                nz_idx = cp.where((inv >= batch_start) & (inv < batch_end))[0]
                if nz_idx.size == 0:
                    batch_start = batch_end
                    continue

                r_i = rows[nz_idx]
                v_i = vals[nz_idx]
                u_i = inv[nz_idx] - batch_start

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

            if use_legacy_factor_batch():
                # Legacy body (pre 2026-07-29), kept behind
                # TENSORMET_LEGACY_FACTOR_BATCH=1 for A/B validation.
                Z_rows = Z_u[u_i]
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
            else:
                # CHANGED (2026-07-29): scatter-free batch body — mirrors
                # distance.py::kl_factor_update_largedim. The row-dot is a fused
                # sampled dot (SDDMM) and the SpMM runs against the UNGATHERED
                # Z_u via the (m, I) transposed batch matrix P, so the
                # (nnz_b, R) Z_rows gather is never materialized. With a
                # grouping, P is built sort-free from the segment offsets.
                # Contributions still land in locals first (atomic commit).
                m_b = batch_end - batch_start
                if grouping is not None:
                    indptr_b = (segment_offsets[batch_start:batch_end + 1] - seg_lo).astype(cp.int32)
                else:
                    # Uncached batches must be column-grouped up front so
                    # P.data order equals entry order (see group_batch_by_column).
                    indptr_b, u_i, r_i, v_i = group_batch_by_column(u_i, m_b, r_i, v_i)

                if divergence == "kl":
                    R_nz = cp.clip(sampled_row_dots(A_d, Z_u, r_i, u_i), a_min=epsilon, a_max=None)
                    w = v_i / R_nz
                else:  # "fr"
                    w = v_i

                P = build_batch_csr_T(w, r_i, m_b, numerator.shape[0], indptr_b)
                num_contrib = spmm_T(P, Z_u)

                den_contrib = None
                if masked:
                    # Observed-only denominator weight:
                    #   KL -> 1 (sum of Z over observed columns)
                    #   FR -> Xhat at the observed entry = <A[row], Z[col]>
                    if divergence == "kl":
                        den_w = cp.full(nnz_b, den_scale, dtype=Z_u.dtype)
                    else:  # "fr"
                        den_w = sampled_row_dots(A_d, Z_u, r_i, u_i) * den_scale
                    P_den = same_pattern_csr(P, den_w)
                    den_contrib = spmm_T(P_den, Z_u)

            # Every allocation for this batch succeeded → commit atomically.
            numerator += num_contrib
            if masked:
                denominator += den_contrib

            batch_start = batch_end

        except _retryable as exc:
            # Drop this attempt's large temporaries, return cached blocks to the
            # driver so the out-of-pool cuBLAS/cuSPARSE workspace has room, then
            # retry the *same* columns at half the width.
            Z_u = Z_rows = num_contrib = den_contrib = S_b = P = P_den = None
            mempool.free_all_blocks()
            try:
                free_b, total_b = cp.cuda.runtime.memGetInfo()
            except Exception:
                free_b = total_b = -1
            if bc <= 1:
                print(
                    f"[shard-diag] FAILED at bc=1 device={device_id} div={divergence} "
                    f"masked={masked} n_ucols={n_ucols} batch={batch_start}:{batch_end} "
                    f"free={free_b/1e9:.2f}GB/{total_b/1e9:.2f}GB "
                    f"err={type(exc).__name__}: {exc}",
                    file=sys.stderr, flush=True,
                )
                raise
            if _cublas_cls and isinstance(exc, _cublas_cls):
                shrank_via_cublas = True
            else:
                shrank_via_oom = True
            new_bc = max(1, bc // 2)
            print(
                f"[shard-diag] shrinking batch device={device_id} masked={masked} "
                f"{bc}->{new_bc} at batch_start={batch_start} "
                f"free={free_b/1e9:.2f}GB/{total_b/1e9:.2f}GB ({type(exc).__name__})",
                file=sys.stderr, flush=True,
            )
            bc = new_bc

    # Report the batch width actually used (post-retry) so the caller can cache it,
    # plus why it shrank (if it did) so the caller can skip persisting a transient
    # cuBLAS-only reduction.
    if batch_sink is not None:
        batch_sink[0] = int(bc)
        if len(batch_sink) > 1:
            batch_sink[1] = "oom" if shrank_via_oom else ("cublas" if shrank_via_cublas else None)
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
    groupings: Optional[List[Optional[ModeGrouping]]] = None,
    batch_box: Optional[dict] = None,
) -> cp.ndarray:
    """
    Orchestrate multi-GPU factor numerator computation and reduce on CPU.

    Full objective: the denominator is computed once on the primary device
    (no NNZ access). Masked/completion objective: the denominator depends on
    the observed entries, so each shard returns a partial denominator that is
    reduced on CPU alongside the partial numerators.

    ``groupings`` (Task 3): optional per-shard precomputed :class:`ModeGrouping`
    for this mode (``groupings[k]`` lives on ``device_ids[k]``). Supplied only on
    the exact, non-subsampling path so each worker skips the decode/sort/scan.
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
    # CHANGED (Task 4): per-shard sinks collect the realized batch width so the
    # caller can cache it; one list per shard avoids a key collision when several
    # shards share a device. Slot 1 carries the shrink cause ("oom"/"cublas"/None).
    sinks: List[list] = [[None, None] for _ in device_ids]
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
                grouping=(groupings[k] if groupings is not None else None),
                batch_sink=sinks[k],
            ): k
            for k in range(len(device_ids))
        }
        for fut in as_completed(futures):
            partials[futures[fut]] = fut.result()
    finally:
        if _own_pool:
            _pool.shutdown(wait=True)

    # CHANGED (Task 4): report the smallest batch width any shard used (the safe
    # value to reuse: if one device had to shrink, all should).
    if batch_box is not None:
        realized = [s[0] for s in sinks if s[0] is not None]
        if realized:
            batch_box["batch_cols"] = min(realized)
            # Persist only a genuine OOM shrink; a cuBLAS-only shrink is transient.
            causes = {s[1] for s in sinks if len(s) > 1 and s[1] is not None}
            batch_box["shrink_cause"] = (
                "oom" if "oom" in causes else ("cublas" if "cublas" in causes else None)
            )

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
    batch_sink: Optional[list] = None,
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

    idxs, xvals = coo_to_coords(shard, shape)
    nnz = int(xvals.size)

    if nnz == 0:
        zero = cp.asnumpy(cp.zeros_like(core_d))
        return zero, (zero.copy() if masked else None)

    if subsample_frac < 1.0:
        idxs, xvals = _apply_subsample(idxs, xvals, subsample_frac, iteration)
        nnz = int(xvals.size)

    # Rescale the masked denominator weights to match the 1/frac rescaling that
    # _apply_subsample applied to xvals (the numerator), so the MU ratio is unbiased.
    den_scale = (1.0 / subsample_frac) if subsample_frac < 1.0 else 1.0

    Num = cp.zeros_like(core_d)
    Den = cp.zeros_like(core_d) if masked else None

    # Estimate AFTER NNZ bookkeeping is live so the free-memory snapshot is accurate.
    # CHANGED (Task 4): only the KL two-pass path uses batch_rhat — skip its
    # estimate (and its memGetInfo read) for the single-pass FR path.
    if divergence == "kl" and batch_rhat is None:
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

    # CHANGED (Task 4): report the batch sizes used so the caller can cache them
    # (FR leaves batch_rhat None — it runs a single pass).
    if batch_sink is not None:
        batch_sink[0] = (batch_rhat, int(batch_num))
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
    batch_box: Optional[dict] = None,
) -> cp.ndarray:
    """
    Orchestrate multi-GPU core numerator computation and reduce on CPU.

    Full objective: the denominator (KL: column-sum outer product; FR: Gram
    contractions) is computed once on the primary device. Masked/completion
    objective: each shard returns a partial denominator (observed-only) that is
    reduced on CPU alongside the partial numerators.

    CHANGED (Task 4): ``batch_box`` (if given) receives the realized
    ``batch_rhat``/``batch_num`` so the caller can cache them across iterations.
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
    sinks: List[list] = [[None] for _ in device_ids]  # CHANGED (Task 4): realized batch report
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
                batch_sink=sinks[k],
            ): k
            for k in range(len(device_ids))
        }
        for fut in as_completed(futures):
            partials[futures[fut]] = fut.result()
    finally:
        if _own_pool:
            _pool.shutdown(wait=True)

    # CHANGED (Task 4): cache the smallest realized batch sizes across shards.
    if batch_box is not None:
        reports = [s[0] for s in sinks if s[0] is not None]
        if reports:
            rhats = [r[0] for r in reports if r[0] is not None]
            nums = [r[1] for r in reports if r[1] is not None]
            if rhats:
                batch_box["batch_rhat"] = min(rhats)
            if nums:
                batch_box["batch_num"] = min(nums)

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

    CHANGED (2026-06-16): the error is sampled on the **same** ``subsample_frac``
    window as the MU numerators (so it costs O(frac·nnz), restoring the
    pre-Task-5 speed) but stays **unbiased** the way Task 5/finding I-1 require:
    the window is taken with ``rescale=False`` and each *summed* scalar is
    weighted by ``nnz/n_sample`` (≈ ``1/frac``).  This is the review's option
    (b) — weight the sums, never the values, so the ``1/frac`` factor does not
    enter the nonlinear ``x·log(x/r)`` / ``x²`` terms.  ``sum_R`` (the analytic
    full reconstruction sum) is kept exact in the orchestrator, so weighting
    ``sum_R_nz`` here makes ``kl_zero = sum_R − sum_R_nz`` unbiased too.  At
    ``subsample_frac == 1`` the window is the whole shard and ``weight == 1``,
    so the metric is identical to the exact full-NNZ value.
    """
    cp.cuda.Device(device_id).use()
    core_d = cp.asarray(core_np)
    factors_d = [cp.asarray(f) for f in factors_np]
    N = len(shape)

    if batch_rhat is None:
        batch_rhat = int(_estimate_batch_rhat_for_tensordot(core_d, factors_d))

    idxs, x_nz = coo_to_coords(shard, shape)
    nnz_full = int(x_nz.size)

    if nnz_full == 0:
        return 0.0, 0.0, 0.0

    # Sample the same window as the numerators (cheap), but DON'T rescale the
    # values — weight the summed terms by nnz/n_sample instead (unbiased,
    # nonlinearity-safe). frac == 1 → no sampling, weight == 1.
    weight = 1.0
    if subsample_frac < 1.0:
        idxs, x_nz = _apply_subsample(idxs, x_nz, subsample_frac, iteration, rescale=False)
        weight = nnz_full / int(x_nz.size)

    nnz = int(x_nz.size)
    x_nz = cp.clip(x_nz.astype(core_d.dtype), a_min=epsilon, a_max=None)

    r_nz = cp.empty_like(x_nz)
    for start in range(0, nnz, batch_rhat):
        end = min(start + batch_rhat, nnz)
        mats = [factors_d[n][idxs[n][start:end]] for n in range(N)]
        r_nz[start:end] = _rhat_from_factor_rows_sequential(core_d, mats, epsilon=epsilon)
    r_nz = cp.clip(r_nz, a_min=epsilon, a_max=None)

    term_pos = x_nz * cp.log(x_nz / r_nz) - x_nz + r_nz
    kl_pos = float(cp.sum(term_pos).get()) * weight
    sum_R_nz = float(cp.sum(r_nz).get()) * weight
    sum_X = float(cp.sum(x_nz).get()) * weight

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
    pool: Optional[ThreadPoolExecutor] = None,
    masked: bool = False,
    subsample_frac: float = 1.0,
    iter_seed: Optional[int] = None,
) -> cp.ndarray:
    """
    Compute relative KL error with sharded NNZ; returns a scalar CuPy array
    on the primary device matching the return type of ``kl_compute_errors_largedim``.

    When ``masked`` is True the zero-entry contribution (sum_R - sum_R_nz) is
    dropped, so the metric reflects the observed-only / completion objective.

    CHANGED (2026-06-16): the error is evaluated on the per-iteration
    ``subsample_frac`` window (cost O(frac·nnz)), and each shard weights its
    summed scalars by ``nnz/n_sample`` so the metric stays unbiased — see
    ``_partial_kl_error_for_shard``.  ``sum_R`` below is the exact analytic full
    sum (unweighted); combined with the weighted ``sum_R_nz_total`` it gives an
    unbiased ``kl_zero``.  At ``subsample_frac == 1`` this reduces to the exact
    full-NNZ value.
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
                iteration=iter_seed,
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
    masked: bool = False,
    subsample_frac: float = 1.0,
    iteration: Optional[int] = None,
) -> Tuple[float, float, float]:
    """
    Compute partial Frobenius error scalars from a single NNZ shard.

    Returns
    -------
    (norm_X_sq, inner_prod, residual_sq) from this shard's NNZ contribution.
    ``inner_prod`` is 0 when ``masked`` (the full ‖X̂‖² term is not used);
    ``residual_sq`` (= sum (x - x̂)²) is 0 when not ``masked``.

    CHANGED (2026-06-16): sampled on the same ``subsample_frac`` window as the
    MU numerators (cost O(frac·nnz)) but **unbiased** per finding I-1 — the
    window is taken with ``rescale=False`` and the summed quadratic terms
    (``norm_X_sq``, ``inner_prod``, ``residual_sq``) are weighted by
    ``nnz/n_sample`` (≈ ``1/frac``).  Weighting the *sum* avoids the ``1/frac²``
    bias that rescaling the *values* before squaring would produce.  ``norm_Xhat²``
    is analytic and stays exact in the orchestrator; weighting ``norm_X_sq`` and
    ``inner_prod`` keeps the full residual ``‖X‖²+‖X̂‖²−2⟨X,X̂⟩`` unbiased.  In the
    masked ratio the weight cancels.  frac == 1 → weight == 1 (exact full NNZ).
    """
    cp.cuda.Device(device_id).use()
    core_d = cp.asarray(core_np)
    factors_d = [cp.asarray(f) for f in factors_np]
    N = len(shape)

    if batch_rhat is None:
        batch_rhat = int(_estimate_batch_rhat_for_tensordot(core_d, factors_d))

    idxs, x_nz = coo_to_coords(shard, shape)
    nnz_full = int(x_nz.size)

    if nnz_full == 0:
        return 0.0, 0.0, 0.0

    weight = 1.0
    if subsample_frac < 1.0:
        idxs, x_nz = _apply_subsample(idxs, x_nz, subsample_frac, iteration, rescale=False)
        weight = nnz_full / int(x_nz.size)

    nnz = int(x_nz.size)
    x_nz = cp.clip(x_nz.astype(core_d.dtype), a_min=0.0, a_max=None)

    norm_X_sq = float(cp.sum(x_nz * x_nz).get()) * weight

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

    inner_prod = float(inner_prod_d.get()) * weight
    residual_sq = float(residual_sq_d.get()) * weight
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
    pool: Optional[ThreadPoolExecutor] = None,
    masked: bool = False,
    subsample_frac: float = 1.0,
    iter_seed: Optional[int] = None,
) -> cp.ndarray:
    """
    Compute relative Frobenius error with sharded NNZ; returns a scalar CuPy
    array on the primary device.

    Full objective uses ‖X - X̂‖²_F = ‖X‖² + ‖X̂‖² - 2⟨X, X̂⟩, with ‖X̂‖²
    computed analytically on the primary device (no NNZ). The masked/completion
    objective uses the observed-only relative RMSE sqrt(sum_Ω (x - x̂)²) / ‖X‖.

    CHANGED (2026-06-16): the error is evaluated on the per-iteration
    ``subsample_frac`` window (cost O(frac·nnz)) with the summed terms weighted
    by ``nnz/n_sample`` for unbiasedness — see ``_partial_fr_error_for_shard``.
    ``norm_Xhat²`` below stays analytic/exact.  At ``subsample_frac == 1`` this
    reduces to the exact full-NNZ value.
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
                masked=masked,
                subsample_frac=subsample_frac,
                iteration=iter_seed,
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
    full_tensor : cpx_sparse.coo_matrix | None  — full COO on device_ids[0] for the
                  single-shard delegate path; ``None`` for multi-shard (Task 7 — the
                  primary device holds only its own ~1/n shard, and the caller's own
                  reference, e.g. ``TuckerDecomposition.tensor``, is the canonical copy)
    orig_shape  : tuple[int, ...]        — original N-D tensor shape
    device_ids  : list[int]              — one per shard; [0] is primary
    shards      : list[coo_matrix]       — shards[k] lives on device_ids[k]
    n_shards    : int
    nnz         : int                    — total NNZ across all shards
    subsample_frac : float               — NNZ fraction; 1.0 = exact
    """

    def __init__(
        self,
        full_tensor: Optional[cpx_sparse.coo_matrix],
        orig_shape: Tuple[int, ...],
        device_ids: List[int],
        shards: List[cpx_sparse.coo_matrix],
        subsample_frac: float = 1.0,
        masked: bool = False,
        nnz: Optional[int] = None,
    ) -> None:
        self.full_tensor = full_tensor
        self.orig_shape = orig_shape
        self.device_ids = device_ids
        self.shards = shards
        self.n_shards = len(device_ids)
        # CHANGED (2026-06-12 review, Task 7): NNZ metadata kept explicitly so the
        # multi-shard path no longer needs to retain the device-resident full COO
        # just to know its size.
        if nnz is not None:
            self.nnz = int(nnz)
        elif full_tensor is not None:
            self.nnz = coords_nnz(full_tensor)
        else:
            self.nnz = int(sum(coords_nnz(s) for s in shards))
        self.subsample_frac = float(subsample_frac)
        # Optimisation objective: when True, fit only observed entries
        # (weighted/completion objective), mirroring cfg.exp.objective="masked".
        self.masked = bool(masked)
        self._iter_seed: Optional[int] = None
        # CHANGED (2026-06-12 review, Task 3): per-shard cache of per-mode NNZ
        # groupings (sort + unique columns + segment offsets). Built lazily on
        # each shard's own device the first time a mode is updated, then reused
        # every iteration. Only used on the exact (non-subsampling) path — under
        # subsampling the sampled NNZ pattern changes every iteration, so the
        # grouping cannot be reused.
        self._grouping_caches: List[Dict[int, ModeGrouping]] = [
            {} for _ in range(self.n_shards)
        ]
        # CHANGED (2026-06-12 review, Task 4): cache the per-(kernel, mode) batch
        # sizes so the per-iteration hot path stops calling _estimate_batch_*
        # (each of which reads driver memGetInfo). The estimates depend only on the
        # static core/factor shapes, so the first iteration's value is reused. The
        # factor path's OOM-retry can shrink a batch mid-update; the realized width
        # is fed back here and the cache is kept monotonically non-increasing
        # ("persist the reduced value") so later iterations don't re-trip the retry.
        self._factor_batch_cache: Dict[Tuple[str, int], int] = {}
        self._core_batch_cache: Dict[str, Dict[str, int]] = {}
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

    def trim_pools(self) -> None:
        """Return cached-but-unused pool blocks to the driver on every shard device.

        Called at a low cadence (``pool_trim_every``) from the training loop — NOT
        per iteration. Per-iteration flushing was removed (see ``set_iter_seed``
        and ``_gpu_free_bytes``) because the cudaFree/cudaMalloc churn stalled the
        GPU 6-7×/iteration. At ~once per sem-check the cost is negligible, and it
        reclaims the transient blocks left by the semantic-eval GPU→CPU copies so
        the out-of-pool cuBLAS/cuSPARSE workspaces keep their headroom — the
        device-0 ``CUBLASError`` starvation. ``free_all_blocks`` only frees the
        *current* device's cached blocks, so we iterate the shard devices; the
        pinned host pool is device-agnostic and freed once.
        """
        for did in self.device_ids:
            with cp.cuda.Device(did):
                cp.get_default_memory_pool().free_all_blocks()
        cp.get_default_pinned_memory_pool().free_all_blocks()

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

        CHANGED (2026-06-12 review, Task 4): no longer flushes every device's
        memory pool each iteration. Returning pool blocks to the driver forced a
        device sync and a full ``cudaMalloc`` storm on the next iteration's
        working set. The pool is now left intact (its blocks are reused), and the
        only remaining flush is the on-demand one inside the factor OOM-retry
        handler in ``_partial_numerator_for_shard``.
        """
        self._iter_seed = int(iteration)

    def _record_factor_batch(
        self,
        key: Tuple[str, int],
        realized: Optional[int],
        shrink_cause: Optional[str] = None,
    ) -> None:
        """Cache the realized factor batch width.

        CHANGED (Task 4): once an OOM-retry shrinks a batch, the smaller value is
        persisted so later iterations start there instead of re-tripping the retry.

        CHANGED (cuBLAS robustness): a shrink caused by a *transient* cuBLAS
        workspace failure (``shrink_cause == "cublas"``) is NOT persisted. The
        cache keeps the wider width the call started from, so the next iteration
        retries at full speed — one cuBLAS hiccup costs a single slow iteration
        instead of pinning the whole run at batch=1 via the monotonic floor. A
        genuine ``OutOfMemoryError`` shrink (stable VRAM ceiling) is still
        persisted monotonically.
        """
        if realized is None:
            return
        if shrink_cause == "cublas":
            return
        prev = self._factor_batch_cache.get(key)
        self._factor_batch_cache[key] = realized if prev is None else min(prev, int(realized))

    def _record_core_batch(self, divergence: str, box: dict) -> None:
        """Cache the realized core batch sizes (``batch_rhat``/``batch_num``)."""
        cached = dict(self._core_batch_cache.get(divergence, {}))
        for kk in ("batch_rhat", "batch_num"):
            v = box.get(kk)
            if v is None:
                continue
            prev = cached.get(kk)
            cached[kk] = int(v) if prev is None else min(prev, int(v))
        if cached:
            self._core_batch_cache[divergence] = cached

    def _mode_groupings(self, mode: int) -> Optional[List[ModeGrouping]]:
        """Return one :class:`ModeGrouping` per shard for *mode* (Task 3).

        Returns ``None`` under stochastic subsampling (the sampled NNZ pattern
        changes every iteration, so a grouping cannot be reused). Otherwise each
        shard's grouping is built once, on that shard's own device, and cached;
        subsequent iterations reuse it, skipping the per-iteration decode,
        ``cp.unique`` sort, and per-batch ``cp.where`` scan.
        """
        if self.subsample_frac < 1.0:
            return None
        out: List[ModeGrouping] = []
        for k in range(self.n_shards):
            cache = self._grouping_caches[k]
            g = cache.get(mode)
            if g is None:
                with cp.cuda.Device(self.device_ids[k]):
                    g = _build_mode_grouping(self.shards[k], self.orig_shape, mode)
                cache[mode] = g
            out.append(g)
        return out

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
        # Coordinate-backed tensors carry no linear index and need no .tocoo();
        # everything below is shape-agnostic between the two storage forms.
        is_coord = isinstance(coo, CoordCOO)
        coo_coo = coo if is_coord else coo.tocoo()

        if device_ids is None or len(device_ids) <= 1:
            anchor = coo_coo.coords if is_coord else coo_coo.row
            primary = _array_device_id(anchor) or 0
            return cls(coo_coo, orig_shape, [primary], [coo_coo],
                       subsample_frac=subsample_frac, masked=masked)

        nnz = coords_nnz(coo_coo)
        n = len(device_ids)
        boundaries = [int(round(nnz * k / n)) for k in range(n + 1)]

        # CHANGED (Task 2): shuffle each shard once at build time (host-side, free —
        # the slices round-trip through the CPU anyway) so per-iteration subsampling
        # is a contiguous window instead of a fresh device-side permutation(nnz).
        # Exact runs (frac == 1.0) keep the original NNZ order.
        _shuffle = subsample_frac < 1.0
        _build = _build_coord_shard if is_coord else _build_shard
        shards = [
            _build(
                coo_coo, boundaries[k], boundaries[k + 1], device_ids[k],
                shuffle_seed=(int(subsample_seed) + k) if _shuffle else None,
            )
            for k in range(n)
        ]

        # CHANGED (2026-06-12 review, Task 7 — O-3): do NOT retain `coo_coo` as a
        # device-resident `full_tensor`. It is only ever read on the single-shard
        # delegate path (above), so on the multi-shard path it would be dead weight —
        # a full extra NNZ copy on the most contended device on top of shard 0. The
        # caller's own reference (TuckerDecomposition.tensor) is the canonical full
        # copy; we keep only the nnz metadata. Dropping it here lets the local
        # `coo_coo` be collected once the shards (each holding their own ~1/n copy)
        # are built.
        return cls(None, orig_shape, list(device_ids), shards,
                   subsample_frac=subsample_frac, masked=masked, nnz=nnz)

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
        # CHANGED (Task 4): reuse the cached batch width; capture the realized one.
        key = ("kl", mode)
        bc = batch_cols if batch_cols is not None else self._factor_batch_cache.get(key)
        box: dict = {}
        A_new = _sharded_factor_update(
            shards=self.shards, device_ids=self.device_ids,
            core=core, factors=factors, mode=mode, shape=shape,
            divergence="kl", epsilon=epsilon, batch_cols=bc, verbose=verbose,
            subsample_frac=self.subsample_frac, iter_seed=self._iter_seed,
            pool=self._pool, masked=self.masked,
            groupings=self._mode_groupings(mode),
            batch_box=box,
        )
        self._record_factor_batch(key, box.get("batch_cols"), box.get("shrink_cause"))
        return A_new

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
        # CHANGED (Task 4): reuse the cached batch width; capture the realized one.
        key = ("fr", mode)
        bc = batch_cols if batch_cols is not None else self._factor_batch_cache.get(key)
        box: dict = {}
        A_new = _sharded_factor_update(
            shards=self.shards, device_ids=self.device_ids,
            core=core, factors=factors, mode=mode, shape=shape,
            divergence="fr", epsilon=epsilon, batch_cols=bc, verbose=verbose,
            subsample_frac=self.subsample_frac, iter_seed=self._iter_seed,
            pool=self._pool, masked=self.masked,
            groupings=self._mode_groupings(mode),
            batch_box=box,
        )
        self._record_factor_batch(key, box.get("batch_cols"), box.get("shrink_cause"))
        return A_new

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
        # CHANGED (Task 4): reuse cached batch sizes; capture the realized ones.
        cached = self._core_batch_cache.get("kl", {})
        brhat = batch_rhat if batch_rhat is not None else cached.get("batch_rhat")
        bnum = batch_num if batch_num is not None else cached.get("batch_num")
        box: dict = {}
        core_new = _sharded_core_update(
            shards=self.shards, device_ids=self.device_ids,
            core=core, factors=factors, shape=shape,
            divergence="kl", epsilon=epsilon,
            batch_rhat=brhat, batch_num=bnum, verbose=verbose,
            subsample_frac=self.subsample_frac, iter_seed=self._iter_seed,
            pool=self._pool, masked=self.masked,
            batch_box=box,
        )
        self._record_core_batch("kl", box)
        return core_new

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
        # CHANGED (Task 4): reuse cached batch_num; capture the realized one.
        cached = self._core_batch_cache.get("fr", {})
        bnum = batch_num if batch_num is not None else cached.get("batch_num")
        box: dict = {}
        core_new = _sharded_core_update(
            shards=self.shards, device_ids=self.device_ids,
            core=core, factors=factors, shape=shape,
            divergence="fr", epsilon=epsilon,
            batch_rhat=None, batch_num=bnum, verbose=verbose,
            subsample_frac=self.subsample_frac, iter_seed=self._iter_seed,
            pool=self._pool, masked=self.masked,
            batch_box=box,
        )
        self._record_core_batch("fr", box)
        return core_new

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
        # CHANGED (2026-06-16): error sampled on the same subsample window as the
        # numerators (O(frac·nnz)) but weighted per shard for unbiasedness — the
        # frac/iter_seed are forwarded so the window matches this iteration.
        return _sharded_kl_error(
            shards=self.shards, device_ids=self.device_ids,
            core=core, factors=factors, shape=shape,
            epsilon=epsilon, batch_rhat=batch_rhat,
            pool=self._pool, masked=self.masked,
            subsample_frac=self.subsample_frac, iter_seed=self._iter_seed,
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
        # CHANGED (2026-06-16): error sampled on the same subsample window as the
        # numerators (O(frac·nnz)) but weighted per shard for unbiasedness — the
        # frac/iter_seed are forwarded so the window matches this iteration.
        return _sharded_fr_error(
            shards=self.shards, device_ids=self.device_ids,
            core=core, factors=factors, shape=shape,
            epsilon=epsilon, batch_rhat=batch_rhat,
            pool=self._pool, masked=self.masked,
            subsample_frac=self.subsample_frac, iter_seed=self._iter_seed,
        )

    # ------------------------------------------------------------------
    # EXPERIMENTAL CP family (experimental/CP/README.md)
    # ------------------------------------------------------------------
    # Thin delegates: the CP machinery lives under
    # experimental/CP/ and are imported lazily, so a Tucker run never touches
    # them. Everything else these need — shards, pool, warm-up, iter seed,
    # subsample window — is the machinery above, unchanged. CP has no core
    # update to shard (``cp_weight_update`` is a passthrough) and no masked
    # objective, so ``self.masked`` plays no part here.

    def cp_factor_update(
        self,
        core: cp.ndarray,
        factors: List[cp.ndarray],
        mode: int,
        shape: Tuple[int, ...],
        divergence: str,
        thread_budget=None,
        epsilon: float = 1e-12,
        batch_nnz: Optional[int] = None,
        verbose: bool = False,
        inner_iters: int = 1,
        scooch_kappa: float = 0.0,
    ) -> cp.ndarray:
        """CP factor update; ``core`` is the λ weight vector, updated in place."""
        from tensormet.experimental.CP import cp_ops, cp_sharded

        if self.n_shards == 1:
            fn = (cp_ops.cp_fr_factor_update if divergence == "fr"
                  else cp_ops.cp_kl_factor_update)
            kwargs = dict(
                vec_tensor=self.full_tensor, core=core, factors=factors,
                mode=mode, shape=shape, thread_budget=thread_budget,
                epsilon=epsilon, verbose=verbose, batch_nnz=batch_nnz,
            )
            if divergence == "kl":
                kwargs.update(inner_iters=inner_iters, scooch_kappa=scooch_kappa)
            return fn(**kwargs)

        # No batch cache here: the orchestrator estimates once per call and
        # reuses it across shards and inner iterations, which is what the
        # Tucker cache exists to avoid.
        return cp_sharded._sharded_cp_factor_update(
            shards=self.shards, device_ids=self.device_ids,
            weights=core, factors=factors, mode=mode, shape=shape,
            divergence=divergence, epsilon=epsilon, batch_nnz=batch_nnz,
            verbose=verbose, subsample_frac=self.subsample_frac,
            iter_seed=self._iter_seed, pool=self._pool,
            inner_iters=inner_iters, scooch_kappa=scooch_kappa,
        )

    def cp_compute_errors(
        self,
        shape: Tuple[int, ...],
        core: cp.ndarray,
        factors: List[cp.ndarray],
        divergence: str,
        thread_budget=None,
        epsilon: float = 1e-12,
        batch_nnz: Optional[int] = None,
        verbose: bool = False,
    ) -> cp.ndarray:
        """CP relative error (KL or FR); ``core`` is the λ weight vector."""
        from tensormet.experimental.CP import cp_ops, cp_sharded

        if self.n_shards == 1:
            fn = (cp_ops.cp_fr_compute_errors if divergence == "fr"
                  else cp_ops.cp_kl_compute_errors)
            return fn(
                vec_tensor=self.full_tensor, shape=shape, core=core,
                factors=factors, thread_budget=thread_budget, epsilon=epsilon,
                verbose=verbose, batch_nnz=batch_nnz,
            )

        fn = (cp_sharded._sharded_cp_fr_error if divergence == "fr"
              else cp_sharded._sharded_cp_kl_error)
        return fn(
            shards=self.shards, device_ids=self.device_ids,
            weights=core, factors=factors, shape=shape,
            epsilon=epsilon, batch_nnz=batch_nnz, pool=self._pool,
            subsample_frac=self.subsample_frac, iter_seed=self._iter_seed,
        )

    # ------------------------------------------------------------------
    # EXPERIMENTAL Tucker-TT hybrid (experimental/TT_hybrid/README.md)
    # ------------------------------------------------------------------
    # Same arrangement as the CP delegates above: the machinery lives under
    # experimental/TT_hybrid/ and is imported lazily. ``core`` is the list of TT
    # cores here, not an array. KL only, and no masked objective, so
    # ``self.masked`` plays no part; the batch caches stay unused because the TT
    # orchestrators estimate once per call and reuse it across shards.

    def tt_factor_update(
        self,
        core: List[cp.ndarray],
        factors: List[cp.ndarray],
        mode: int,
        shape: Tuple[int, ...],
        thread_budget=None,
        epsilon: float = 1e-12,
        batch_nnz: Optional[int] = None,
        verbose: bool = False,
    ) -> cp.ndarray:
        """TT-KL factor update; ``core[mode]`` is rescaled in place."""
        from tensormet.experimental.TT_hybrid import tt_ops, tt_sharded

        if self.n_shards == 1:
            return tt_ops.tt_kl_factor_update(
                vec_tensor=self.full_tensor, core=core, factors=factors,
                mode=mode, shape=shape, thread_budget=thread_budget,
                epsilon=epsilon, verbose=verbose, batch_nnz=batch_nnz,
            )

        return tt_sharded._sharded_tt_factor_update(
            shards=self.shards, device_ids=self.device_ids,
            tt_cores=core, factors=factors, mode=mode, shape=shape,
            epsilon=epsilon, batch_nnz=batch_nnz, verbose=verbose,
            subsample_frac=self.subsample_frac, iter_seed=self._iter_seed,
            pool=self._pool,
        )

    def tt_core_update(
        self,
        shape: Tuple[int, ...],
        core: List[cp.ndarray],
        factors: List[cp.ndarray],
        modes=None,
        thread_budget=None,
        epsilon: float = 1e-12,
        batch_nnz: Optional[int] = None,
        verbose: bool = False,
    ) -> List[cp.ndarray]:
        """TT-KL core sweep; one reduce per site (see tt_sharded's docstring)."""
        from tensormet.experimental.TT_hybrid import tt_ops, tt_sharded

        if self.n_shards == 1:
            return tt_ops.tt_kl_core_update(
                vec_tensor=self.full_tensor, shape=shape, core=core,
                factors=factors, modes=modes, thread_budget=thread_budget,
                epsilon=epsilon, verbose=verbose, batch_nnz=batch_nnz,
            )

        return tt_sharded._sharded_tt_core_update(
            shards=self.shards, device_ids=self.device_ids,
            tt_cores=core, factors=factors, shape=shape, modes=modes,
            epsilon=epsilon, batch_nnz=batch_nnz, verbose=verbose,
            subsample_frac=self.subsample_frac, iter_seed=self._iter_seed,
            pool=self._pool,
        )

    def tt_compute_errors(
        self,
        shape: Tuple[int, ...],
        core: List[cp.ndarray],
        factors: List[cp.ndarray],
        thread_budget=None,
        epsilon: float = 1e-12,
        batch_nnz: Optional[int] = None,
        verbose: bool = False,
    ) -> cp.ndarray:
        """TT relative KL error; ``Σ_all x̂`` stays closed-form on the primary."""
        from tensormet.experimental.TT_hybrid import tt_ops, tt_sharded

        if self.n_shards == 1:
            return tt_ops.tt_kl_compute_errors(
                vec_tensor=self.full_tensor, shape=shape, core=core,
                factors=factors, thread_budget=thread_budget, epsilon=epsilon,
                verbose=verbose, batch_nnz=batch_nnz,
            )

        return tt_sharded._sharded_tt_kl_error(
            shards=self.shards, device_ids=self.device_ids,
            tt_cores=core, factors=factors, shape=shape,
            epsilon=epsilon, batch_nnz=batch_nnz, pool=self._pool,
            subsample_frac=self.subsample_frac, iter_seed=self._iter_seed,
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
