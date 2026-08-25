"""
tt_sharded.py — Multi-GPU NNZ sharding for the Tucker-TT hybrid kernels (KL).

EXPERIMENTAL (see README.md in this directory).

Mirrors ``experimental/CP/cp_sharded.py``, which mirrors ``sharded_sparse.py``:
NNZ-partitioned shards, one thread per GPU, partials reduced on the CPU,
finalize on the primary device. Shard construction, the persistent thread pool,
the cuBLAS warm-up, the per-iteration subsample window and ``trim_pools`` are
all inherited from ``ShardedSparseTensor`` — this module supplies only the TT
per-shard workers and their orchestrators.

What is sharded, and what is not
--------------------------------
Only the NNZ-dependent accumulations cross devices. Every TT denominator is a
*closed form* over the factor column sums (the same chain run on
``_colsum_batch``), so unlike Tucker's sharded core update nothing but the
numerators is reduced:

    sharded   factor numerator  Num[i, r]        → (I_mode, R_mode) reduce
    sharded   core numerator    Num_k[a, r, b]   → (ρ_k, R_k, ρ_{k+1}) reduce
    sharded   KL error          (kl_pos, Σ_nnz x̂, Σ x)  → 3 scalars
    primary   both denominators, Σ_all x̂         (closed form, NNZ-free)
    primary   the MU divide, ε-clip, the ℓ1 rescale and the in-place core write

No R^N object ever crosses the bus; the largest payload is one TT core.

The core sweep costs N reduce rounds
------------------------------------
``tt_kl_core_update`` visits sites sequentially so each site update sees the
previous ones — that is what makes the sweep a genuine block MU, and therefore
monotone. Sharded, that becomes one fan-out/fan-in cycle *per site*: N full NNZ
passes and N barriers per core update, against CP's zero (its λ update is a
passthrough) and Tucker's one. Updating all sites from a single pass would cost
one barrier instead of N, but each site would then be updated against stale
neighbours and the monotonicity guarantee — the correctness oracle for these
kernels — would be gone. The sequential sweep is kept.

Only the core just written is re-broadcast between sites (``tt_cores_buf[k]``),
so the host traffic per sweep is Σ_k |C_k| rather than N·Σ_k |C_k|.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from tensormet.distance import coo_to_coords
from tensormet.sharded_sparse import _apply_subsample
from tensormet.utils import make_lazy_cupy_pair
from tensormet.experimental.TT_hybrid.tt_chain import (
    left_envs, right_envs, site_grad, sites,
)
from tensormet.experimental.TT_hybrid.tt_ops import (
    _colsum_batch,
    _run_nnz_batches,
    _scatter_add,
    estimate_batch_nnz_tt,
    tt_sum_all_entries,
)

cp, cpx_sparse = make_lazy_cupy_pair()


# ---------------------------------------------------------------------------
# Per-shard workers
# ---------------------------------------------------------------------------

def _tt_partial_factor_num_for_shard(
    shard,
    tt_cores_np: List[Union[np.ndarray, Any]],
    factors_np: List[Union[np.ndarray, Any]],
    mode: int,
    shape: Tuple[int, ...],
    epsilon: float,
    batch_nnz: Optional[int],
    device_id: int,
    subsample_frac: float = 1.0,
    iteration: Optional[int] = None,
) -> np.ndarray:
    """Partial factor numerator from one NNZ shard, as an (I_mode, R_mode) array.

        Num[i, r] = Σ_{p : i_mode(p) = i} (x_p / x̂_p) · Z_p[r]

    ``tt_cores_np`` / ``factors_np`` may be NumPy or CuPy already resident on
    ``device_id`` (``cp.asarray`` is then a no-op), so the primary shard never
    round-trips through the host.
    """
    # .use() rather than the context manager: see _partial_numerator_for_shard.
    cp.cuda.Device(device_id).use()
    tt_cores_d = [cp.asarray(C) for C in tt_cores_np]
    factors_d = [cp.asarray(f) for f in factors_np]
    out = cp.zeros_like(factors_d[mode])

    idxs, vals = coo_to_coords(shard, shape)
    if vals.size == 0:
        return cp.asnumpy(out)

    if subsample_frac < 1.0:
        # The numerator is LINEAR in the values, so rescaling by 1/frac here is
        # the unbiased choice (contrast the error worker below).
        idxs, vals = _apply_subsample(idxs, vals, subsample_frac, iteration)

    nnz = int(vals.size)
    if batch_nnz is None:
        batch_nnz = estimate_batch_nnz_tt(tt_cores_d, factors_d)
    n_modes = len(shape)

    def _body(start, end):
        mats = [factors_d[n][idxs[n][start:end]] for n in range(n_modes)]
        S = sites(tt_cores_d, mats, cp)
        L, R = left_envs(S, cp), right_envs(S, cp)
        xhat = cp.clip(L[n_modes][:, 0], a_min=epsilon, a_max=None)
        Z = site_grad(L[mode], tt_cores_d[mode], R[mode + 1], cp)
        _scatter_add(out, idxs[mode][start:end], vals[start:end] / xhat, Z)

    _run_nnz_batches(nnz, batch_nnz, _body, desc=f"shard factor {mode}")

    result = cp.asnumpy(out)
    cp.cuda.Device(device_id).synchronize()
    return result


def _tt_partial_tied_factor_num_for_shard(
    shard,
    tt_cores_np: List[Union[np.ndarray, Any]],
    factors_np: List[Union[np.ndarray, Any]],
    group: List[int],
    shape: Tuple[int, ...],
    epsilon: float,
    batch_nnz: Optional[int],
    device_id: int,
    subsample_frac: float = 1.0,
    iteration: Optional[int] = None,
) -> np.ndarray:
    """Pooled numerator for a factor tied across ``group``, as (I, R).

        Num[i, r] = Σ_{n∈group} Σ_{p : i_n(p) = i} (x_p / x̂_p) · Z⁽ⁿ⁾_p[r]

    One NNZ pass for the whole group: the left/right sweeps are built once per
    batch and reused for every site gradient. Linear in the values, so the same
    1/frac rescale as the single-leg worker applies.
    """
    cp.cuda.Device(device_id).use()
    tt_cores_d = [cp.asarray(C) for C in tt_cores_np]
    factors_d = [cp.asarray(f) for f in factors_np]
    group = list(group)
    out = cp.zeros_like(factors_d[group[0]])

    idxs, vals = coo_to_coords(shard, shape)
    if vals.size == 0:
        return cp.asnumpy(out)

    if subsample_frac < 1.0:
        idxs, vals = _apply_subsample(idxs, vals, subsample_frac, iteration)

    nnz = int(vals.size)
    if batch_nnz is None:
        batch_nnz = estimate_batch_nnz_tt(tt_cores_d, factors_d)
    n_modes = len(shape)

    def _body(start, end):
        mats = [factors_d[k][idxs[k][start:end]] for k in range(n_modes)]
        S = sites(tt_cores_d, mats, cp)
        L, R = left_envs(S, cp), right_envs(S, cp)
        w = vals[start:end] / cp.clip(L[n_modes][:, 0], a_min=epsilon, a_max=None)
        for n in group:
            Z = site_grad(L[n], tt_cores_d[n], R[n + 1], cp)
            _scatter_add(out, idxs[n][start:end], w, Z)

    _run_nnz_batches(nnz, batch_nnz, _body, desc=f"shard tied factors {tuple(group)}")

    result = cp.asnumpy(out)
    cp.cuda.Device(device_id).synchronize()
    return result


def _tt_partial_core_num_for_shard(
    shard,
    tt_cores_np: List[Union[np.ndarray, Any]],
    factors_np: List[Union[np.ndarray, Any]],
    site: int,
    shape: Tuple[int, ...],
    epsilon: float,
    batch_nnz: Optional[int],
    device_id: int,
    subsample_frac: float = 1.0,
    iteration: Optional[int] = None,
) -> np.ndarray:
    """Partial numerator for TT core ``site``, as a (ρ_k, R_k, ρ_{k+1}) array.

        Num_k[a, r, b] = Σ_p (x_p / x̂_p) · L_k[p, a] · A_k[i_k(p), r] · R_{k+1}[p, b]

    Also linear in the values, so the same 1/frac rescale applies.
    """
    cp.cuda.Device(device_id).use()
    tt_cores_d = [cp.asarray(C) for C in tt_cores_np]
    factors_d = [cp.asarray(f) for f in factors_np]
    num = cp.zeros_like(tt_cores_d[site])

    idxs, vals = coo_to_coords(shard, shape)
    if vals.size == 0:
        return cp.asnumpy(num)

    if subsample_frac < 1.0:
        idxs, vals = _apply_subsample(idxs, vals, subsample_frac, iteration)

    nnz = int(vals.size)
    if batch_nnz is None:
        batch_nnz = estimate_batch_nnz_tt(tt_cores_d, factors_d)
    n_modes = len(shape)

    def _body(start, end):
        mats = [factors_d[n][idxs[n][start:end]] for n in range(n_modes)]
        S = sites(tt_cores_d, mats, cp)
        L, R = left_envs(S, cp), right_envs(S, cp)
        w = vals[start:end] / cp.clip(L[n_modes][:, 0], a_min=epsilon, a_max=None)
        # Σ_p w_p · outer(L_k[p], A_k row, R_{k+1}[p]) as one GEMM.
        LW = (L[site] * w[:, None])[:, :, None] * mats[site][:, None, :]
        num[...] += (LW.reshape(end - start, -1).T @ R[site + 1]).reshape(num.shape)

    _run_nnz_batches(nnz, batch_nnz, _body, desc=f"shard core site {site}")

    result = cp.asnumpy(num)
    cp.cuda.Device(device_id).synchronize()
    return result


def _tt_partial_kl_error_for_shard(
    shard,
    tt_cores_np: List[Union[np.ndarray, Any]],
    factors_np: List[Union[np.ndarray, Any]],
    shape: Tuple[int, ...],
    epsilon: float,
    batch_nnz: Optional[int],
    device_id: int,
    subsample_frac: float = 1.0,
    iteration: Optional[int] = None,
) -> Tuple[float, float, float]:
    """Partial KL error scalars ``(kl_pos, Σ_nnz x̂, Σ x)`` from one shard.

    Sampled on the same window as the numerators but with ``rescale=False``,
    weighting the *summed* scalars by ``nnz/n_sample`` instead: ``x log(x/x̂)``
    is nonlinear, so a 1/frac factor inside the log would bias it. ``Σ_all x̂``
    is analytic and stays exact in the orchestrator, so weighting ``Σ_nnz x̂``
    keeps the zero-entry term unbiased.
    """
    cp.cuda.Device(device_id).use()
    tt_cores_d = [cp.asarray(C) for C in tt_cores_np]
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
        batch_nnz = estimate_batch_nnz_tt(tt_cores_d, factors_d)
    n_modes = len(shape)

    x_nz = cp.clip(cp.asarray(vals), a_min=epsilon, a_max=None)
    kl_pos = cp.asarray(0.0, dtype=factors_d[0].dtype)
    sum_xhat_nz = cp.asarray(0.0, dtype=factors_d[0].dtype)
    def _body(start, end):
        mats = [factors_d[n][idxs[n][start:end]] for n in range(n_modes)]
        # Only the left sweep is needed for x̂.
        xhat = cp.clip(left_envs(sites(tt_cores_d, mats, cp), cp)[n_modes][:, 0],
                       a_min=epsilon, a_max=None)
        x_b = x_nz[start:end]
        # Both reductions before either accumulation, so a retry cannot
        # double-count the first.
        pos = cp.sum(x_b * cp.log(x_b / xhat) - x_b + xhat)
        tot = cp.sum(xhat)
        kl_pos[...] += pos
        sum_xhat_nz[...] += tot

    _run_nnz_batches(nnz, batch_nnz, _body, desc="shard error x̂ pass")

    out = (float(kl_pos.get()) * weight,
           float(sum_xhat_nz.get()) * weight,
           float(cp.sum(x_nz).get()) * weight)
    cp.cuda.Device(device_id).synchronize()
    return out


# ---------------------------------------------------------------------------
# Orchestrators
# ---------------------------------------------------------------------------

def _fan_out(worker, n_shards: int, pool: Optional[ThreadPoolExecutor], kwargs_for):
    """Run ``worker`` once per shard in parallel; return results in shard order.

    ``kwargs_for(k)`` supplies shard k's kwargs, which lets a caller hand shard 0
    the primary's device arrays and the rest their host buffers.
    """
    results: List[Optional[Any]] = [None] * n_shards
    _own_pool = pool is None
    _pool = ThreadPoolExecutor(max_workers=n_shards) if _own_pool else pool
    try:
        futures: Dict = {
            _pool.submit(worker, **kwargs_for(k)): k for k in range(n_shards)
        }
        for fut in as_completed(futures):
            results[futures[fut]] = fut.result()
    finally:
        if _own_pool:
            _pool.shutdown(wait=True)
    return results


def _sharded_tt_factor_update(
    shards,
    device_ids: List[int],
    tt_cores: List[Any],
    factors: List[Any],
    mode: int,
    shape: Tuple[int, ...],
    epsilon: float = 1e-12,
    batch_nnz: Optional[int] = None,
    verbose: bool = False,
    subsample_frac: float = 1.0,
    iter_seed: Optional[int] = None,
    pool: Optional[ThreadPoolExecutor] = None,
):
    """Multi-GPU KL factor update; returns the new A^(mode) on the primary device.

    ``tt_cores[mode]`` is mutated IN PLACE by the ℓ1 rescale, exactly as the
    single-GPU kernel does, so the training loop's ``core`` variable stays
    current without a loop change.
    """
    if verbose:
        print(f"  [TT-KL/sharded] Updating factor {mode}...")
    primary = device_ids[0]
    n_shards = len(device_ids)
    shape = tuple(int(s) for s in shape)

    with cp.cuda.Device(primary):
        A = factors[mode]
        # Hoisted: one estimate per call rather than one per shard (each costs a
        # driver memGetInfo). Shard devices are assumed homogeneous, as in CP.
        if batch_nnz is None:
            batch_nnz = estimate_batch_nnz_tt(tt_cores, factors)
        # Denominator: the same chain run on the factor column sums, i.e. the
        # sum over ALL entries. NNZ-free, so it never leaves the primary.
        S_sum = sites(tt_cores, _colsum_batch(factors, epsilon), cp, skip=mode)
        den = cp.clip(
            site_grad(left_envs(S_sum, cp)[mode], tt_cores[mode],
                      right_envs(S_sum, cp)[mode + 1], cp),
            a_min=epsilon, a_max=None,
        )  # (1, R_mode)

    tt_cores_buf = [cp.asnumpy(C) for C in tt_cores]
    factors_buf = [cp.asnumpy(f) for f in factors]

    partials = _fan_out(
        _tt_partial_factor_num_for_shard, n_shards, pool,
        lambda k: dict(
            shard=shards[k],
            # Shard 0 lives on the primary, so hand it the device arrays
            # directly and skip the host round-trip.
            tt_cores_np=tt_cores if k == 0 else tt_cores_buf,
            factors_np=factors if k == 0 else factors_buf,
            mode=mode, shape=shape, epsilon=epsilon, batch_nnz=batch_nnz,
            device_id=device_ids[k], subsample_frac=subsample_frac,
            iteration=iter_seed,
        ),
    )
    num_np = np.add.reduce(partials)

    with cp.cuda.Device(primary):
        A_new = cp.clip(A * (cp.asarray(num_np) / den), a_min=epsilon, a_max=None)
        scale = cp.clip(cp.sum(A_new, axis=0), a_min=epsilon, a_max=None)
        tt_cores[mode] *= scale[None, :, None]
        return cp.clip(A_new / scale[None, :], a_min=epsilon, a_max=None)


def _sharded_tt_tied_factor_update(
    shards,
    device_ids: List[int],
    tt_cores: List[Any],
    factors: List[Any],
    group: List[int],
    shape: Tuple[int, ...],
    epsilon: float = 1e-12,
    batch_nnz: Optional[int] = None,
    verbose: bool = False,
    subsample_frac: float = 1.0,
    iter_seed: Optional[int] = None,
    pool: Optional[ThreadPoolExecutor] = None,
):
    """Multi-GPU pooled KL update for a factor tied across ``group``.

    One fan-out/fan-in cycle for the whole group (the pooled numerator is a
    single (I, R) reduce), against |group| cycles for the per-mode updates it
    replaces. Every tied mode's core is rescaled in place on the primary.
    """
    if verbose:
        print(f"  [TT-KL/sharded] Updating tied factor group {tuple(group)}...")
    primary = device_ids[0]
    n_shards = len(device_ids)
    shape = tuple(int(s) for s in shape)
    group = list(group)

    with cp.cuda.Device(primary):
        A = factors[group[0]]
        if batch_nnz is None:
            batch_nnz = estimate_batch_nnz_tt(tt_cores, factors)
        # Pooled denominator: Σ_n over the same closed-form column-sum chain the
        # single-leg kernel uses. NNZ-free, so it never leaves the primary.
        sums = _colsum_batch(factors, epsilon)
        den = cp.zeros((1, int(A.shape[1])), dtype=A.dtype)
        for n in group:
            S_sum = sites(tt_cores, sums, cp, skip=n)
            den += site_grad(left_envs(S_sum, cp)[n], tt_cores[n],
                             right_envs(S_sum, cp)[n + 1], cp)
        den = cp.clip(den, a_min=epsilon, a_max=None)

    tt_cores_buf = [cp.asnumpy(C) for C in tt_cores]
    factors_buf = [cp.asnumpy(f) for f in factors]

    partials = _fan_out(
        _tt_partial_tied_factor_num_for_shard, n_shards, pool,
        lambda k: dict(
            shard=shards[k],
            tt_cores_np=tt_cores if k == 0 else tt_cores_buf,
            factors_np=factors if k == 0 else factors_buf,
            group=group, shape=shape, epsilon=epsilon, batch_nnz=batch_nnz,
            device_id=device_ids[k], subsample_frac=subsample_frac,
            iteration=iter_seed,
        ),
    )
    num_np = np.add.reduce(partials)

    with cp.cuda.Device(primary):
        A_new = cp.clip(A * (cp.asarray(num_np) / den), a_min=epsilon, a_max=None)
        scale = cp.clip(cp.sum(A_new, axis=0), a_min=epsilon, a_max=None)
        for n in group:
            tt_cores[n] *= scale[None, :, None]
        return cp.clip(A_new / scale[None, :], a_min=epsilon, a_max=None)


def _sharded_tt_core_update(
    shards,
    device_ids: List[int],
    tt_cores: List[Any],
    factors: List[Any],
    shape: Tuple[int, ...],
    modes=None,
    epsilon: float = 1e-12,
    batch_nnz: Optional[int] = None,
    verbose: bool = False,
    subsample_frac: float = 1.0,
    iter_seed: Optional[int] = None,
    pool: Optional[ThreadPoolExecutor] = None,
):
    """Multi-GPU KL core sweep: one fan-out/fan-in cycle per site, in order.

    Returns the same ``tt_cores`` list (elements rebound on the primary), so the
    training loop's ``core = routing.core_update(...)`` keeps working.
    """
    if verbose:
        print("  [TT-KL/sharded] Updating TT cores...")
    primary = device_ids[0]
    n_shards = len(device_ids)
    shape = tuple(int(s) for s in shape)
    n_modes = len(shape)
    if modes is not None and list(modes) != list(range(n_modes)):
        raise NotImplementedError("_sharded_tt_core_update assumes modes == range(N).")

    with cp.cuda.Device(primary):
        if batch_nnz is None:
            batch_nnz = estimate_batch_nnz_tt(tt_cores, factors)
        sums = _colsum_batch(factors, epsilon)

    # The factors are constant for the whole sweep; only the core just written
    # is refreshed below, so the host traffic is one copy of the chain in total.
    factors_buf = [cp.asnumpy(f) for f in factors]
    tt_cores_buf = [cp.asnumpy(C) for C in tt_cores]

    for k in range(n_modes):
        with cp.cuda.Device(primary):
            # Denominator: rank-1 in the column-sum environments, NNZ-free.
            S_sum = sites(tt_cores, sums, cp)
            Ls, Rs = left_envs(S_sum, cp), right_envs(S_sum, cp)
            den = cp.clip(
                Ls[k][0][:, None, None] * sums[k][0][None, :, None]
                * Rs[k + 1][0][None, None, :],
                a_min=epsilon, a_max=None,
            )

        partials = _fan_out(
            _tt_partial_core_num_for_shard, n_shards, pool,
            lambda k_shard, _site=k: dict(
                shard=shards[k_shard],
                tt_cores_np=tt_cores if k_shard == 0 else tt_cores_buf,
                factors_np=factors if k_shard == 0 else factors_buf,
                site=_site, shape=shape, epsilon=epsilon, batch_nnz=batch_nnz,
                device_id=device_ids[k_shard], subsample_frac=subsample_frac,
                iteration=iter_seed,
            ),
        )
        num_np = np.add.reduce(partials)

        with cp.cuda.Device(primary):
            tt_cores[k] = cp.clip(tt_cores[k] * (cp.asarray(num_np) / den),
                                  a_min=epsilon, a_max=None)
        # Sequential sweep: site k+1 must see this write, so refresh the buffer
        # the other shards read from (only this one core changed).
        tt_cores_buf[k] = cp.asnumpy(tt_cores[k])

    return tt_cores


def _sharded_tt_kl_error(
    shards,
    device_ids: List[int],
    tt_cores: List[Any],
    factors: List[Any],
    shape: Tuple[int, ...],
    epsilon: float = 1e-12,
    batch_nnz: Optional[int] = None,
    pool: Optional[ThreadPoolExecutor] = None,
    subsample_frac: float = 1.0,
    iter_seed: Optional[int] = None,
):
    """Relative generalized KL divergence with sharded NNZ; scalar on primary.

    Normalized exactly as ``tt_ops.tt_kl_compute_errors`` (and as
    ``distance.kl_compute_errors_largedim``):

        D   = Σ_nnz [x log(x/x̂) − x + x̂] + (Σ_all x̂ − Σ_nnz x̂)
        rel = D / Σ_nnz x
    """
    primary = device_ids[0]
    n_shards = len(device_ids)
    shape = tuple(int(s) for s in shape)
    tt_cores_np = [cp.asnumpy(C) for C in tt_cores]
    factors_np = [cp.asnumpy(f) for f in factors]

    results = _fan_out(
        _tt_partial_kl_error_for_shard, n_shards, pool,
        lambda k: dict(
            shard=shards[k], tt_cores_np=tt_cores_np, factors_np=factors_np,
            shape=shape, epsilon=epsilon, batch_nnz=batch_nnz,
            device_id=device_ids[k], subsample_frac=subsample_frac,
            iteration=iter_seed,
        ),
    )
    kl_pos = sum(r[0] for r in results)
    sum_xhat_nz = sum(r[1] for r in results)
    sum_X = sum(r[2] for r in results)

    with cp.cuda.Device(primary):
        # Σ over ALL entries of X̂ — closed form, exact, never sampled.
        sum_all = float(cp.asnumpy(tt_sum_all_entries(tt_cores, factors, epsilon=epsilon)))

    # rel = D / Σ_nnz x is undefined with no data; nan rather than D/ε.
    result = ((kl_pos + (sum_all - sum_xhat_nz)) / sum_X if sum_X > 0
              else float("nan"))

    with cp.cuda.Device(primary):
        return cp.asarray(result, dtype=factors[0].dtype)


__all__ = [
    "_sharded_tt_factor_update",
    "_sharded_tt_tied_factor_update",
    "_sharded_tt_core_update",
    "_sharded_tt_kl_error",
]
