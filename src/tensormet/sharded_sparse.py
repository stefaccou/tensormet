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
When ``subsample_frac < 1.0``, each per-shard function samples a random
fraction of its local NNZ and rescales values by ``1/subsample_frac``,
giving an unbiased estimator of the full numerator without any per-iteration
resharding.  Seeds are deterministic: ``iter_seed + shard_k`` per shard.

Call ``sst.set_iter_seed(iteration)`` once at the top of each iteration so
the SST's internal seed advances — wrapper function signatures are unchanged.

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
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional, Tuple

import cupy as cp
import cupyx.scipy.sparse as cpx_sparse
import numpy as np

from tensormet.distance import (
    # Factor update helpers
    _blocked_coo_to_flat_indices,           # distance.py:582
    _estimate_batch_cols_for_Z,             # distance.py:751
    _unravel_flat_indices_C,                # distance.py:448
    _unravel_cols_for_mode,                 # distance.py:397
    fr_factor_update_largedim,
    kl_factor_update_largedim,
    # Core update helpers
    _accumulate_core_num_outer,             # distance.py:495
    _core_multilinear_grams,               # distance.py:670
    _estimate_batch_num_for_outer,          # distance.py:703
    _estimate_batch_rhat_for_tensordot,     # distance.py:730
    _rhat_from_factor_rows_sequential,      # distance.py:469
    _tucker_sum_all_entries,               # distance.py:594
    fr_core_update_largedim,
    kl_core_update_largedim,
    # Error helpers
    fr_compute_errors_largedim,
    kl_compute_errors_largedim,
    # Denominator helpers
    _tucker_den_row_full,                   # distance.py:421
    _tucker_gram_ZtZ,                       # distance.py:622
)
from tensormet.sparse_ops import compute_Zcols_batch, safe_ravel


# ---------------------------------------------------------------------------
# Internal utilities
# ---------------------------------------------------------------------------

def _build_shard(
    coo: cpx_sparse.coo_matrix,
    start: int,
    end: int,
    target_device: int,
) -> cpx_sparse.coo_matrix:
    """
    Extract NNZ slice [start, end) from *coo* and place it on *target_device*.

    Preserves the same ``(block_size, n_blocks)`` shape so that
    ``_blocked_coo_to_flat_indices`` works identically on shards.
    Routes through CPU because CuPy does not support direct cross-device
    tensor slicing.  One-time cost at initialisation.
    """
    row_np = cp.asnumpy(coo.row[start:end])
    col_np = cp.asnumpy(coo.col[start:end])
    data_np = cp.asnumpy(coo.data[start:end])

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
    rng_seed: int,
) -> Tuple[cp.ndarray, cp.ndarray]:
    """
    Randomly subsample NNZ on the current GPU device.

    Samples ``max(1, round(subsample_frac * nnz))`` entries without
    replacement, rescaling values by ``1/subsample_frac`` to preserve
    the expectation of any downstream accumulation.

    Parameters
    ----------
    flat, vals :
        Flat indices and values from ``_blocked_coo_to_flat_indices``.
    subsample_frac :
        Fraction of NNZ to retain.  Must be < 1.0 (caller is responsible
        for not calling this at 1.0).
    rng_seed :
        Integer seed for ``cp.random.RandomState`` — deterministic and
        distinct per shard per iteration.

    Returns
    -------
    flat_s, vals_s : subsampled and rescaled arrays.
    """
    nnz = int(flat.size)
    n_sample = max(1, int(round(subsample_frac * nnz)))
    rng = cp.random.RandomState(rng_seed)
    idx = cp.sort(rng.permutation(nnz)[:n_sample])
    scale = vals.dtype.type(1.0 / subsample_frac)
    return flat[idx], vals[idx] * scale


# ---------------------------------------------------------------------------
# Factor update — per-shard partial numerator
# ---------------------------------------------------------------------------

def _partial_numerator_for_shard(
    shard: cpx_sparse.coo_matrix,
    core_np: np.ndarray,
    factors_np: List[np.ndarray],
    mode: int,
    shape: Tuple[int, ...],
    divergence: str,
    epsilon: float,
    batch_cols: Optional[int],
    device_id: int,
    subsample_frac: float = 1.0,
    rng_seed: Optional[int] = None,
) -> np.ndarray:
    """
    Compute the partial factor-numerator contribution from a single NNZ shard.

    Runs entirely inside ``cp.cuda.Device(device_id)``.  When
    ``subsample_frac < 1.0`` the shard's NNZ is subsampled locally before
    accumulation — no resharding is needed.

    Returns
    -------
    partial_num : np.ndarray of shape ``(I_mode, R_mode)``
    """
    with cp.cuda.Device(device_id):
        core_d = cp.asarray(core_np)
        factors_d = [cp.asarray(f) for f in factors_np]
        A_d = factors_d[mode]

        if batch_cols is None:
            batch_cols = int(_estimate_batch_cols_for_Z(core_d, factors_d, mode))

        flat, vals = _blocked_coo_to_flat_indices(shard, shape)

        if flat.size == 0:
            return cp.asnumpy(cp.zeros_like(A_d))

        if subsample_frac < 1.0:
            flat, vals = _apply_subsample(flat, vals, subsample_frac, rng_seed)

        idxs = _unravel_flat_indices_C(flat, shape)
        rows = idxs[mode]

        other_modes = [m for m in range(len(shape)) if m != mode]
        other_shape = tuple(shape[m] for m in other_modes)
        other_coords = [idxs[m] for m in other_modes]
        cols = safe_ravel(tuple(other_coords), other_shape, cp)

        numerator = cp.zeros_like(A_d)
        ucols, inv = cp.unique(cols, return_inverse=True)
        n_ucols = int(ucols.size)

        for batch_start in range(0, n_ucols, batch_cols):
            batch_end = min(batch_start + batch_cols, n_ucols)
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
                continue

            r_i = rows[nz_idx]
            v_i = vals[nz_idx]
            u_i = inv[nz_idx] - batch_start
            Z_rows = Z_u[u_i]

            if divergence == "kl":
                A_rows = A_d[r_i]
                R_nz = cp.sum(A_rows * Z_rows, axis=1)
                R_nz = cp.clip(R_nz, a_min=epsilon, a_max=None)
                w = v_i / R_nz
            else:  # "fr"
                w = v_i

            # numerator[row] += w * Z  — cuSPARSE SpMM (no serialised atomics)
            nnz_b = int(r_i.size)
            S_b = cpx_sparse.csr_matrix(
                (w, (r_i.astype(cp.int32), cp.arange(nnz_b, dtype=cp.int32))),
                shape=(numerator.shape[0], nnz_b),
            )
            numerator += S_b @ Z_rows

        cp.cuda.Device(device_id).synchronize()
        return cp.asnumpy(numerator)


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
) -> cp.ndarray:
    """
    Orchestrate multi-GPU factor numerator computation and reduce on CPU.

    Denominator is computed once on the primary device (no NNZ access).
    Partial numerators are dispatched in parallel threads, summed on CPU,
    then the MU update is applied on the primary device.
    """
    primary = device_ids[0]

    core_np = cp.asnumpy(core)
    factors_np = [cp.asnumpy(f) for f in factors]
    A_primary = factors[mode]

    # Denominator — analytical, no NNZ
    with cp.cuda.Device(primary):
        if divergence == "kl":
            den_row = _tucker_den_row_full(core, factors, mode, epsilon=epsilon)
            denominator = den_row[None, :]
        else:
            Gram = _tucker_gram_ZtZ(core, factors, mode, epsilon=epsilon)
            denominator = A_primary @ Gram
            denominator = cp.clip(denominator, a_min=epsilon, a_max=None)

    # Parallel partial numerators
    partial_nums: List[Optional[np.ndarray]] = [None] * len(device_ids)
    with ThreadPoolExecutor(max_workers=len(device_ids)) as pool:
        futures: Dict = {
            pool.submit(
                _partial_numerator_for_shard,
                shard=shards[k],
                core_np=core_np,
                factors_np=factors_np,
                mode=mode,
                shape=shape,
                divergence=divergence,
                epsilon=epsilon,
                batch_cols=batch_cols,
                device_id=device_ids[k],
                subsample_frac=subsample_frac,
                rng_seed=(None if iter_seed is None else iter_seed + k),
            ): k
            for k in range(len(device_ids))
        }
        for fut in as_completed(futures):
            partial_nums[futures[fut]] = fut.result()

    # CPU reduce + MU update
    numerator_np = np.add.reduce(partial_nums)

    with cp.cuda.Device(primary):
        numerator = cp.asarray(numerator_np)
        numerator = cp.clip(numerator, a_min=epsilon, a_max=None)
        A_new = A_primary * (numerator / (denominator + epsilon))
        A_new = cp.clip(A_new, a_min=epsilon, a_max=None)

    return A_new


# ---------------------------------------------------------------------------
# Core update — per-shard partial numerator
# ---------------------------------------------------------------------------

def _partial_core_num_for_shard(
    shard: cpx_sparse.coo_matrix,
    core_np: np.ndarray,
    factors_np: List[np.ndarray],
    shape: Tuple[int, ...],
    divergence: str,
    epsilon: float,
    batch_rhat: Optional[int],
    batch_num: Optional[int],
    device_id: int,
    subsample_frac: float = 1.0,
    rng_seed: Optional[int] = None,
) -> np.ndarray:
    """
    Compute the partial core-numerator contribution from a single NNZ shard.

    KL path: Pass 1 computes ``w = x / r̂``; Pass 2 accumulates outer products.
    FR path: Single pass with ``w = x`` directly.

    When ``subsample_frac < 1.0`` the shard's NNZ is subsampled before both
    passes, so the two-pass KL structure operates on the same sampled subset.

    Returns
    -------
    partial_num : np.ndarray of shape ``core.shape``
    """
    with cp.cuda.Device(device_id):
        core_d = cp.asarray(core_np)
        factors_d = [cp.asarray(f) for f in factors_np]
        N = len(shape)

        if batch_rhat is None:
            batch_rhat = int(_estimate_batch_rhat_for_tensordot(core_d, factors_d))
        if batch_num is None:
            batch_num = int(_estimate_batch_num_for_outer(core_d, factors_d))

        flat, xvals = _blocked_coo_to_flat_indices(shard, shape)
        nnz = int(flat.size)

        if nnz == 0:
            return cp.asnumpy(cp.zeros_like(core_d))

        if subsample_frac < 1.0:
            flat, xvals = _apply_subsample(flat, xvals, subsample_frac, rng_seed)
            nnz = int(flat.size)

        idxs = _unravel_flat_indices_C(flat, shape)
        Num = cp.zeros_like(core_d)

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
        else:  # "fr" — single pass
            for start in range(0, nnz, batch_num):
                end = min(start + batch_num, nnz)
                mats = [factors_d[n][idxs[n][start:end]] for n in range(N)]
                _accumulate_core_num_outer(Num, xvals[start:end], mats)

        cp.cuda.Device(device_id).synchronize()
        return cp.asnumpy(Num)


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
) -> cp.ndarray:
    """
    Orchestrate multi-GPU core numerator computation and reduce on CPU.

    Denominator (KL: column-sum outer product; FR: Gram contractions) is
    computed once on the primary device.  Partial ``Num`` arrays are summed
    on CPU, then the MU update is applied on the primary device.
    """
    primary = device_ids[0]
    N = len(shape)

    core_np = cp.asnumpy(core)
    factors_np = [cp.asnumpy(f) for f in factors]

    partial_nums: List[Optional[np.ndarray]] = [None] * len(device_ids)
    with ThreadPoolExecutor(max_workers=len(device_ids)) as pool:
        futures: Dict = {
            pool.submit(
                _partial_core_num_for_shard,
                shard=shards[k],
                core_np=core_np,
                factors_np=factors_np,
                shape=shape,
                divergence=divergence,
                epsilon=epsilon,
                batch_rhat=batch_rhat,
                batch_num=batch_num,
                device_id=device_ids[k],
                subsample_frac=subsample_frac,
                rng_seed=(None if iter_seed is None else iter_seed + k),
            ): k
            for k in range(len(device_ids))
        }
        for fut in as_completed(futures):
            partial_nums[futures[fut]] = fut.result()

    Num_np = np.add.reduce(partial_nums)

    with cp.cuda.Device(primary):
        Num = cp.asarray(Num_np)

        if divergence == "kl":
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
    rng_seed: Optional[int] = None,
) -> Tuple[float, float, float]:
    """
    Compute partial KL error scalars from a single NNZ shard.

    Returns
    -------
    (kl_pos, sum_R_nz, sum_X) from this shard's NNZ contribution.
    """
    with cp.cuda.Device(device_id):
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
            flat, x_nz = _apply_subsample(flat, x_nz, subsample_frac, rng_seed)
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
) -> cp.ndarray:
    """
    Compute relative KL error with sharded NNZ; returns a scalar CuPy array
    on the primary device matching the return type of ``kl_compute_errors_largedim``.
    """
    primary = device_ids[0]
    core_np = cp.asnumpy(core)
    factors_np = [cp.asnumpy(f) for f in factors]

    results: List[Optional[Tuple[float, float, float]]] = [None] * len(device_ids)
    with ThreadPoolExecutor(max_workers=len(device_ids)) as pool:
        futures: Dict = {
            pool.submit(
                _partial_kl_error_for_shard,
                shard=shards[k],
                core_np=core_np,
                factors_np=factors_np,
                shape=shape,
                epsilon=epsilon,
                batch_rhat=batch_rhat,
                device_id=device_ids[k],
                subsample_frac=subsample_frac,
                rng_seed=(None if iter_seed is None else iter_seed + k),
            ): k
            for k in range(len(device_ids))
        }
        for fut in as_completed(futures):
            results[futures[fut]] = fut.result()

    kl_pos_total = sum(r[0] for r in results)
    sum_R_nz_total = sum(r[1] for r in results)
    sum_X_total = sum(r[2] for r in results)

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
    rng_seed: Optional[int] = None,
) -> Tuple[float, float]:
    """
    Compute partial Frobenius error scalars from a single NNZ shard.

    Returns
    -------
    (norm_X_sq, inner_prod) from this shard's NNZ contribution.
    """
    with cp.cuda.Device(device_id):
        core_d = cp.asarray(core_np)
        factors_d = [cp.asarray(f) for f in factors_np]
        N = len(shape)

        if batch_rhat is None:
            batch_rhat = int(_estimate_batch_rhat_for_tensordot(core_d, factors_d))

        flat, x_nz = _blocked_coo_to_flat_indices(shard, shape)
        nnz = int(flat.size)

        if nnz == 0:
            return 0.0, 0.0

        if subsample_frac < 1.0:
            flat, x_nz = _apply_subsample(flat, x_nz, subsample_frac, rng_seed)
            nnz = int(flat.size)

        x_nz = cp.clip(x_nz.astype(core_d.dtype), a_min=0.0, a_max=None)
        idxs = _unravel_flat_indices_C(flat, shape)

        norm_X_sq = float(cp.sum(x_nz * x_nz).get())

        inner_prod_d = cp.asarray(0.0, dtype=core_d.dtype)
        for start in range(0, nnz, batch_rhat):
            end = min(start + batch_rhat, nnz)
            mats = [factors_d[n][idxs[n][start:end]] for n in range(N)]
            xhat_b = _rhat_from_factor_rows_sequential(core_d, mats, epsilon=epsilon)
            inner_prod_d += cp.sum(x_nz[start:end] * xhat_b)

        inner_prod = float(inner_prod_d.get())
        cp.cuda.Device(device_id).synchronize()
        return norm_X_sq, inner_prod


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
) -> cp.ndarray:
    """
    Compute relative Frobenius error with sharded NNZ; returns a scalar CuPy
    array on the primary device.

    Uses ‖X - X̂‖²_F = ‖X‖² + ‖X̂‖² - 2⟨X, X̂⟩.
    ‖X̂‖² is computed analytically on the primary device (no NNZ).
    """
    primary = device_ids[0]
    N = len(factors)
    core_np = cp.asnumpy(core)
    factors_np = [cp.asnumpy(f) for f in factors]

    results: List[Optional[Tuple[float, float]]] = [None] * len(device_ids)
    with ThreadPoolExecutor(max_workers=len(device_ids)) as pool:
        futures: Dict = {
            pool.submit(
                _partial_fr_error_for_shard,
                shard=shards[k],
                core_np=core_np,
                factors_np=factors_np,
                shape=shape,
                epsilon=epsilon,
                batch_rhat=batch_rhat,
                device_id=device_ids[k],
                subsample_frac=subsample_frac,
                rng_seed=(None if iter_seed is None else iter_seed + k),
            ): k
            for k in range(len(device_ids))
        }
        for fut in as_completed(futures):
            results[futures[fut]] = fut.result()

    norm_X_sq_total = sum(r[0] for r in results)
    inner_prod_total = sum(r[1] for r in results)

    with cp.cuda.Device(primary):
        grams = [factors[n].T @ factors[n] for n in range(N)]
        Den = _core_multilinear_grams(core, grams, epsilon=epsilon)
        norm_Xhat_sq = float(cp.sum(core * Den).get())

    norm_X = math.sqrt(max(norm_X_sq_total, float(epsilon)))

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
    NNZ sampling on the multi-GPU path.  Call ``set_iter_seed(iteration)``
    once at the start of each training loop iteration so seeds advance
    deterministically.  The wrapper functions (``make_sharded_*``) require
    no changes.

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
    ) -> None:
        self.full_tensor = full_tensor
        self.orig_shape = orig_shape
        self.device_ids = device_ids
        self.shards = shards
        self.n_shards = len(device_ids)
        self.subsample_frac = float(subsample_frac)
        self._iter_seed: Optional[int] = None

    def set_iter_seed(self, iteration: int) -> None:
        """
        Advance the internal seed to *iteration*.

        Call this once at the top of each training loop iteration.  The seed
        passed to shard k will be ``iteration * n_shards + k``, ensuring
        distinct, reproducible samples per shard per iteration.
        """
        self._iter_seed = int(iteration) * self.n_shards

    @classmethod
    def from_coo(
        cls,
        coo: cpx_sparse.coo_matrix,
        orig_shape: Tuple[int, ...],
        device_ids: Optional[List[int]] = None,
        subsample_frac: float = 1.0,
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
        """
        coo_coo = coo.tocoo()

        if device_ids is None or len(device_ids) <= 1:
            primary = int(coo_coo.row.device) if hasattr(coo_coo.row, "device") else 0
            return cls(coo_coo, orig_shape, [primary], [coo_coo],
                       subsample_frac=subsample_frac)

        nnz = int(coo_coo.row.size)
        n = len(device_ids)
        boundaries = [int(round(nnz * k / n)) for k in range(n + 1)]

        shards = [
            _build_shard(coo_coo, boundaries[k], boundaries[k + 1], device_ids[k])
            for k in range(n)
        ]

        return cls(coo_coo, orig_shape, list(device_ids), shards,
                   subsample_frac=subsample_frac)

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
                batch_cols=batch_cols, verbose=verbose,
            )
        return _sharded_factor_update(
            shards=self.shards, device_ids=self.device_ids,
            core=core, factors=factors, mode=mode, shape=shape,
            divergence="kl", epsilon=epsilon, batch_cols=batch_cols, verbose=verbose,
            subsample_frac=self.subsample_frac, iter_seed=self._iter_seed,
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
                batch_cols=batch_cols, verbose=verbose,
            )
        return _sharded_factor_update(
            shards=self.shards, device_ids=self.device_ids,
            core=core, factors=factors, mode=mode, shape=shape,
            divergence="fr", epsilon=epsilon, batch_cols=batch_cols, verbose=verbose,
            subsample_frac=self.subsample_frac, iter_seed=self._iter_seed,
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
            )
        return _sharded_core_update(
            shards=self.shards, device_ids=self.device_ids,
            core=core, factors=factors, shape=shape,
            divergence="kl", epsilon=epsilon,
            batch_rhat=batch_rhat, batch_num=batch_num, verbose=verbose,
            subsample_frac=self.subsample_frac, iter_seed=self._iter_seed,
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
                batch_num=batch_num, verbose=verbose,
            )
        return _sharded_core_update(
            shards=self.shards, device_ids=self.device_ids,
            core=core, factors=factors, shape=shape,
            divergence="fr", epsilon=epsilon,
            batch_rhat=None, batch_num=batch_num, verbose=verbose,
            subsample_frac=self.subsample_frac, iter_seed=self._iter_seed,
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
                batch_rhat=batch_rhat, verbose=verbose,
            )
        return _sharded_kl_error(
            shards=self.shards, device_ids=self.device_ids,
            core=core, factors=factors, shape=shape,
            epsilon=epsilon, batch_rhat=batch_rhat,
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
                batch_rhat=batch_rhat, verbose=verbose,
            )
        return _sharded_fr_error(
            shards=self.shards, device_ids=self.device_ids,
            core=core, factors=factors, shape=shape,
            epsilon=epsilon, batch_rhat=batch_rhat,
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
