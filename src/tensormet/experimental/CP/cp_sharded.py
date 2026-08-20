"""
cp_sharded.py — Multi-GPU NNZ sharding for the nonnegative CP kernels.

EXPERIMENTAL (see reviews/CP_IMPLEMENTATION_PLAN.md, Phase 5).

Mirrors ``sharded_sparse.py``'s design exactly: NNZ-partitioned shards, one
thread per GPU, partials reduced on the CPU, finalize on the primary device.
Shard construction, the persistent thread pool, the cuBLAS warm-up, the
per-iteration subsample window and ``trim_pools`` are all inherited from
``ShardedSparseTensor`` — this module only supplies the CP per-shard workers
and their orchestrators.

What is sharded, and what is not
--------------------------------
Only the NNZ-dependent accumulations cross devices; the CP identities are all
NNZ-free and stay on the primary:

    sharded   FR numerator  M = MTTKRP(X, {A}, mode)      -> (I_mode, R) reduce
    sharded   KL numerator  Phi                            -> (I_mode, R) reduce
    sharded   FR error      (sum x^2, <X, X_hat>)          -> 2 scalars
    sharded   KL error      (kl_pos, sum_nnz x_hat, sum x) -> 3 scalars
    primary   Gamma = (*) A^T A,  sigma_r = prod 1^T a_r
    primary   sum_all x_hat = sum_r lam_r prod (1^T a_r),  ||X_hat||^2 = lam^T Gamma lam
    primary   the MU divide, epsilon-clip, normalize and the in-place lambda write

CP therefore needs **no sharded core update at all**: ``cp_weight_update`` is a
passthrough because the factor updates already absorbed lambda.

Reduce payloads are (I_mode, R) — no R^N object ever crosses the bus.

inner_iters under sharding
--------------------------
Phi depends on B (through x_hat), so every CP-APR inner iteration needs a fresh
fan-out/fan-in cycle: broadcast B, accumulate per shard, reduce, MU on primary.
The bytes are negligible at production nnz (the reduce is a fixed (I_mode, R)
regardless of nnz), but the per-inner barrier latency is not free. FR is
unaffected: its MTTKRP weights are the tensor's own values, so it is
loop-invariant and runs one cycle per mode.
"""
from __future__ import annotations

import math
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from tensormet.utils import make_lazy_cupy_pair
from tensormet.distance import coo_to_coords
from tensormet.sharded_sparse import _apply_subsample
from tensormet.experimental.CP.cp_ops import (
    _cp_absorb_into_weights,
    _cp_fr_mu_step,
    _cp_kl_mu_step,
    _cp_kl_phi_from_idxs,
    _cp_mttkrp_from_idxs,
    _colsum_products,
    _gathered_hadamard,
    _hadamard_of_grams,
    estimate_batch_nnz_cp,
)

cp, cpx_sparse = make_lazy_cupy_pair()


# ---------------------------------------------------------------------------
# Per-shard workers
# ---------------------------------------------------------------------------

def _cp_partial_numerator_for_shard(
    shard,
    factors_np: List[Union[np.ndarray, Any]],
    mode: int,
    shape: Tuple[int, ...],
    divergence: str,
    epsilon: float,
    batch_nnz: Optional[int],
    device_id: int,
    B_np: Optional[Union[np.ndarray, Any]] = None,
    subsample_frac: float = 1.0,
    iteration: Optional[int] = None,
) -> np.ndarray:
    """Partial CP factor numerator from one NNZ shard, as an (I_mode, R) array.

    ``divergence="fr"`` accumulates the MTTKRP weighted by the tensor's own
    values; ``"kl"`` accumulates Phi, which additionally needs the current
    ``B_np`` to form x_hat at the NNZ.

    ``factors_np`` / ``B_np`` may be NumPy or CuPy already resident on
    ``device_id`` (``cp.asarray`` is then a no-op), so the primary shard never
    round-trips through the host.
    """
    # .use() rather than the context manager: see _partial_numerator_for_shard.
    cp.cuda.Device(device_id).use()
    factors_d = [cp.asarray(f) for f in factors_np]
    out = cp.zeros_like(factors_d[mode])

    idxs, vals = coo_to_coords(shard, shape)
    if vals.size == 0:
        return cp.asnumpy(out)

    if subsample_frac < 1.0:
        # Numerators are LINEAR in the values, so rescaling by 1/frac here is
        # the unbiased choice (contrast the error workers below).
        idxs, vals = _apply_subsample(idxs, vals, subsample_frac, iteration)

    nnz = int(vals.size)
    if batch_nnz is None:
        batch_nnz = estimate_batch_nnz_cp(factors_d)

    if divergence == "fr":
        _cp_mttkrp_from_idxs(out, idxs, nnz, factors_d, mode, vals,
                             batch_nnz=batch_nnz)
    else:
        B_d = cp.asarray(B_np)
        _cp_kl_phi_from_idxs(out, B_d, idxs, vals, nnz, factors_d, mode,
                             batch_nnz=batch_nnz, epsilon=epsilon)

    result = cp.asnumpy(out)
    cp.cuda.Device(device_id).synchronize()
    return result


def _cp_partial_fr_error_for_shard(
    shard,
    weights_np,
    factors_np: List[Union[np.ndarray, Any]],
    shape: Tuple[int, ...],
    epsilon: float,
    batch_nnz: Optional[int],
    device_id: int,
    subsample_frac: float = 1.0,
    iteration: Optional[int] = None,
) -> Tuple[float, float]:
    """Partial FR error scalars ``(sum x^2, <X, X_hat>)`` from one NNZ shard.

    Sampled on the same window as the numerators but with ``rescale=False``,
    weighting the *summed* scalars by ``nnz/n_sample`` instead: ``x^2`` is
    nonlinear, so a 1/frac factor inside the square would bias it.
    """
    cp.cuda.Device(device_id).use()
    weights_d = cp.asarray(weights_np)
    factors_d = [cp.asarray(f) for f in factors_np]

    idxs, vals = coo_to_coords(shard, shape)
    nnz_full = int(vals.size)
    if nnz_full == 0:
        return 0.0, 0.0

    weight = 1.0
    if subsample_frac < 1.0:
        idxs, vals = _apply_subsample(idxs, vals, subsample_frac, iteration,
                                      rescale=False)
        weight = nnz_full / int(vals.size)

    nnz = int(vals.size)
    if batch_nnz is None:
        batch_nnz = estimate_batch_nnz_cp(factors_d)

    x_nz = cp.clip(vals.astype(factors_d[0].dtype), a_min=0.0, a_max=None)
    norm_X_sq = cp.sum(x_nz * x_nz)

    inner_prod = cp.asarray(0.0, dtype=factors_d[0].dtype)
    for start in range(0, nnz, int(batch_nnz)):
        end = min(start + int(batch_nnz), nnz)
        H = _gathered_hadamard(factors_d, idxs, start, end)  # (b, R), all modes
        inner_prod += cp.sum(x_nz[start:end] * (H @ weights_d))

    out = (float(norm_X_sq.get()) * weight, float(inner_prod.get()) * weight)
    cp.cuda.Device(device_id).synchronize()
    return out


def _cp_partial_kl_error_for_shard(
    shard,
    weights_np,
    factors_np: List[Union[np.ndarray, Any]],
    shape: Tuple[int, ...],
    epsilon: float,
    batch_nnz: Optional[int],
    device_id: int,
    subsample_frac: float = 1.0,
    iteration: Optional[int] = None,
) -> Tuple[float, float, float]:
    """Partial KL error scalars ``(kl_pos, sum_nnz x_hat, sum x)`` from one shard.

    Same weight-the-sums treatment as the FR worker; ``x log(x/x_hat)`` is the
    nonlinearity here. ``sum_all x_hat`` is analytic and stays exact in the
    orchestrator, so weighting ``sum_nnz x_hat`` keeps ``kl_zero`` unbiased.
    """
    cp.cuda.Device(device_id).use()
    weights_d = cp.asarray(weights_np)
    factors_d = [cp.asarray(f) for f in factors_np]

    idxs, vals = coo_to_coords(shard, shape)
    nnz_full = int(vals.size)
    if nnz_full == 0:
        return 0.0, 0.0, 0.0

    weight = 1.0
    if subsample_frac < 1.0:
        idxs, vals = _apply_subsample(idxs, vals, subsample_frac, iteration,
                                      rescale=False)
        weight = nnz_full / int(vals.size)

    nnz = int(vals.size)
    if batch_nnz is None:
        batch_nnz = estimate_batch_nnz_cp(factors_d)

    x_nz = cp.clip(cp.asarray(vals), a_min=epsilon, a_max=None)

    kl_pos = cp.asarray(0.0, dtype=factors_d[0].dtype)
    sum_xhat_nz = cp.asarray(0.0, dtype=factors_d[0].dtype)
    for start in range(0, nnz, int(batch_nnz)):
        end = min(start + int(batch_nnz), nnz)
        H = _gathered_hadamard(factors_d, idxs, start, end)  # (b, R)
        xhat_b = cp.clip(H @ weights_d, a_min=epsilon, a_max=None)
        x_b = x_nz[start:end]
        kl_pos += cp.sum(x_b * cp.log(x_b / xhat_b) - x_b + xhat_b)
        sum_xhat_nz += cp.sum(xhat_b)

    out = (float(kl_pos.get()) * weight,
           float(sum_xhat_nz.get()) * weight,
           float(cp.sum(x_nz).get()) * weight)
    cp.cuda.Device(device_id).synchronize()
    return out


# ---------------------------------------------------------------------------
# Orchestrators
# ---------------------------------------------------------------------------

def _reduce_numerator(
    shards,
    device_ids: List[int],
    factors,
    factors_buf,
    mode: int,
    shape: Tuple[int, ...],
    divergence: str,
    epsilon: float,
    batch_nnz: Optional[int],
    subsample_frac: float,
    iter_seed: Optional[int],
    pool: Optional[ThreadPoolExecutor],
    B=None,
    B_buf=None,
) -> np.ndarray:
    """One fan-out/fan-in cycle: partial numerators in parallel, summed on CPU."""
    partials: List[Optional[np.ndarray]] = [None] * len(device_ids)
    _own_pool = pool is None
    _pool = ThreadPoolExecutor(max_workers=len(device_ids)) if _own_pool else pool
    try:
        futures: Dict = {
            _pool.submit(
                _cp_partial_numerator_for_shard,
                shard=shards[k],
                # Shard 0 lives on the primary, so hand it the device arrays
                # directly and skip the host round-trip.
                factors_np=factors if k == 0 else factors_buf,
                mode=mode,
                shape=shape,
                divergence=divergence,
                epsilon=epsilon,
                batch_nnz=batch_nnz,
                device_id=device_ids[k],
                B_np=(B if k == 0 else B_buf),
                subsample_frac=subsample_frac,
                iteration=iter_seed,
            ): k
            for k in range(len(device_ids))
        }
        for fut in as_completed(futures):
            partials[futures[fut]] = fut.result()
    finally:
        if _own_pool:
            _pool.shutdown(wait=True)
    return np.add.reduce(partials)


def _sharded_cp_factor_update(
    shards,
    device_ids: List[int],
    weights,
    factors,
    mode: int,
    shape: Tuple[int, ...],
    divergence: str,
    epsilon: float = 1e-12,
    batch_nnz: Optional[int] = None,
    verbose: bool = False,
    subsample_frac: float = 1.0,
    iter_seed: Optional[int] = None,
    pool: Optional[ThreadPoolExecutor] = None,
    inner_iters: int = 1,
    scooch_kappa: float = 0.0,
):
    """Multi-GPU CP factor update; returns the new A(mode) on the primary device.

    ``weights`` (lambda) is mutated IN PLACE on the primary, exactly as the
    single-GPU kernels do, so the training loop's ``core`` variable stays
    current without a loop change.
    """
    if verbose:
        print(f"  [CP-{divergence.upper()}/sharded] Updating factor {mode}...")
    primary = device_ids[0]
    multi = len(device_ids) > 1

    with cp.cuda.Device(primary):
        A = factors[mode]
        B = A * weights[None, :]
        # Hoisted: one estimate per call rather than one per shard per inner
        # iteration (each costs a driver memGetInfo).
        if batch_nnz is None:
            batch_nnz = estimate_batch_nnz_cp(factors)

    factors_buf = [cp.asnumpy(f) for f in factors] if multi else factors

    if divergence == "fr":
        # The FR numerator is weighted by the tensor's own values, so it does
        # not depend on B: one cycle per mode, no inner loop.
        M_np = _reduce_numerator(
            shards, device_ids, factors, factors_buf, mode, shape, "fr",
            epsilon, batch_nnz, subsample_frac, iter_seed, pool,
        )
        with cp.cuda.Device(primary):
            Gamma = _hadamard_of_grams(factors, skip_mode=mode, epsilon=epsilon)
            B = _cp_fr_mu_step(B, cp.asarray(M_np), Gamma, epsilon=epsilon)
            return _cp_absorb_into_weights(B, weights, norm="l2", epsilon=epsilon)

    with cp.cuda.Device(primary):
        sigma = _colsum_products(factors, skip_mode=mode, epsilon=epsilon)

    for _inner in range(max(1, int(inner_iters))):
        # Phi depends on B, so each inner iteration re-broadcasts B and pays a
        # fresh reduce. Only B changes here; the other factors are constant for
        # the whole inner loop.
        B_buf = cp.asnumpy(B) if multi else B
        Phi_np = _reduce_numerator(
            shards, device_ids, factors, factors_buf, mode, shape, "kl",
            epsilon, batch_nnz, subsample_frac, iter_seed, pool,
            B=B, B_buf=B_buf,
        )
        with cp.cuda.Device(primary):
            B = _cp_kl_mu_step(B, cp.asarray(Phi_np), sigma, epsilon=epsilon,
                               scooch_kappa=scooch_kappa)

    with cp.cuda.Device(primary):
        return _cp_absorb_into_weights(B, weights, norm="l1", epsilon=epsilon)


def _run_error_workers(worker, shards, device_ids, weights_np, factors_np, shape,
                       epsilon, batch_nnz, subsample_frac, iter_seed, pool):
    """Fan out one error worker per shard and return the per-shard tuples."""
    results: List[Optional[tuple]] = [None] * len(device_ids)
    _own_pool = pool is None
    _pool = ThreadPoolExecutor(max_workers=len(device_ids)) if _own_pool else pool
    try:
        futures: Dict = {
            _pool.submit(
                worker,
                shard=shards[k],
                weights_np=weights_np,
                factors_np=factors_np,
                shape=shape,
                epsilon=epsilon,
                batch_nnz=batch_nnz,
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
    return results


def _sharded_cp_fr_error(shards, device_ids, weights, factors, shape,
                         epsilon=1e-12, batch_nnz=None, pool=None,
                         subsample_frac=1.0, iter_seed=None):
    """Relative Frobenius error with sharded NNZ; scalar CuPy array on primary."""
    primary = device_ids[0]
    weights_np = cp.asnumpy(weights)
    factors_np = [cp.asnumpy(f) for f in factors]

    results = _run_error_workers(
        _cp_partial_fr_error_for_shard, shards, device_ids, weights_np,
        factors_np, shape, epsilon, batch_nnz, subsample_frac, iter_seed, pool,
    )
    norm_X_sq = sum(r[0] for r in results)
    inner_prod = sum(r[1] for r in results)

    with cp.cuda.Device(primary):
        # ||X_hat||^2 = lam^T (*) A^T A lam — analytic, exact, never sampled.
        G_all = _hadamard_of_grams(factors, epsilon=epsilon)
        norm_Xhat_sq = float((weights @ (G_all @ weights)).get())

    norm_X = math.sqrt(max(norm_X_sq, float(epsilon)))
    if norm_X_sq == 0.0:
        result = math.sqrt(max(norm_Xhat_sq, float(epsilon))) / norm_X
    else:
        residual_sq = max(norm_X_sq + norm_Xhat_sq - 2.0 * inner_prod, 0.0)
        result = math.sqrt(residual_sq) / norm_X

    with cp.cuda.Device(primary):
        return cp.asarray(result, dtype=weights.dtype)


def _sharded_cp_kl_error(shards, device_ids, weights, factors, shape,
                         epsilon=1e-12, batch_nnz=None, pool=None,
                         subsample_frac=1.0, iter_seed=None):
    """Relative generalized KL divergence with sharded NNZ; scalar on primary."""
    primary = device_ids[0]
    weights_np = cp.asnumpy(weights)
    factors_np = [cp.asnumpy(f) for f in factors]

    results = _run_error_workers(
        _cp_partial_kl_error_for_shard, shards, device_ids, weights_np,
        factors_np, shape, epsilon, batch_nnz, subsample_frac, iter_seed, pool,
    )
    kl_pos = sum(r[0] for r in results)
    sum_xhat_nz = sum(r[1] for r in results)
    sum_X = sum(r[2] for r in results)

    with cp.cuda.Device(primary):
        # Sum over ALL entries of X_hat — closed form, exact, O(sum I_n R).
        sum_all = float(cp.sum(weights * _colsum_products(factors, epsilon=epsilon)).get())

    kl_total = kl_pos + (sum_all - sum_xhat_nz)
    result = kl_total / max(sum_X, float(epsilon))

    with cp.cuda.Device(primary):
        return cp.asarray(result, dtype=weights.dtype)
