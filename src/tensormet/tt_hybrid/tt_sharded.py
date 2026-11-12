"""
tt_sharded.py — Multi-GPU NNZ sharding for the Tucker-TT hybrid kernels (KL).

EXPERIMENTAL (see README.md in this directory).

Mirrors ``uniform_tt/utt_sharded.py``: shards, thread pool, warm-up and
subsampling are inherited from ``ShardedSparseTensor``. One worker per shard
runs an NNZ pass from tt_ops on its own GPU and returns a partial numerator; the
primary adds them and applies the step. The orchestrators read ``shards /
device_ids / subsample_frac / _iter_seed / _pool`` off the ShardedSparseTensor
directly, so sharded_sparse.py needs no TT delegates.

What is sharded, and what is not
--------------------------------
Only the numerators; every TT denominator is a closed form over the factor
column sums:

    sharded   factor numerator  Num[i, r]        → (I_mode, R_mode) reduce
    sharded   core numerator    Num_k[a, r, b]   → (ρ_k, R_k, ρ_{k+1}) reduce
    sharded   KL error          (kl_pos, Σ_nnz x̂, Σ x)  → 3 scalars
    primary   both denominators, Σ_all x̂         (closed form, NNZ-free)
    primary   the MU divide, ε-clip, the ℓ1 rescale and the in-place core write

No R^N object ever crosses the bus; the largest payload is one TT core.

The core sweep costs N reduce rounds
------------------------------------
Sites are updated sequentially so each sees the previous ones; that keeps the
sweep a monotone block MU (the correctness oracle), at the cost of N NNZ
passes and N barriers per core update. Only the core just written is
re-broadcast between sites.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional

import numpy as np

from tensormet.distance import coo_to_coords
from tensormet.sharded_sparse import apply_subsample
from tensormet.utils import make_lazy_cupy_pair
from tensormet.tt_hybrid.tt_ops import (
    _colsum_batch,
    _xhat_floor,
    apply_core_step,
    apply_factor_step,
    core_den,
    core_pass,
    error_pass,
    estimate_batch_nnz_tt,
    factor_den,
    factor_pass,
    tt_sum_all_entries,
)

cp, cpx_sparse = make_lazy_cupy_pair()


# ---------------------------------------------------------------------------
# Per-shard worker
# ---------------------------------------------------------------------------

def _shard_worker(shard, cores_np, factors_np, shape, kind, arg, epsilon, batch_nnz,
                  device_id, subsample_frac=1.0, iteration=None):
    """Run one NNZ pass on a shard. ``kind`` is 'factor' (``arg`` = the tied
    group), 'core' (``arg`` = the site) or 'error' (``arg`` unused).

    ``cores_np`` / ``factors_np`` may be NumPy or CuPy already resident on
    ``device_id`` (``cp.asarray`` is then a no-op), so the primary shard never
    round-trips through the host. Numerators are linear in the values, so a
    subsample is rescaled by 1/frac; the error is not (x log x is nonlinear), so
    it is sampled without rescaling and the summed scalars are reweighted by
    nnz/n_sample instead. Σ_all x̂ is analytic and stays exact on the primary, so
    weighting Σ_nnz x̂ keeps the zero-entry term unbiased.
    """
    # .use() rather than the context manager: see _partial_numerator_for_shard.
    cp.cuda.Device(device_id).use()
    cores = [cp.asarray(C) for C in cores_np]
    factors = [cp.asarray(f) for f in factors_np]
    xf = _xhat_floor(cores, factors, shape, epsilon)

    idxs, vals = coo_to_coords(shard, shape)
    nnz_full = int(vals.size)
    weight = 1.0
    if subsample_frac < 1.0 and nnz_full:
        if kind == "error":
            idxs, vals = apply_subsample(idxs, vals, subsample_frac, iteration, rescale=False)
            weight = nnz_full / int(vals.size)
        else:
            idxs, vals = apply_subsample(idxs, vals, subsample_frac, iteration)
    if batch_nnz is None:
        batch_nnz = estimate_batch_nnz_tt(cores, factors)

    if kind == "error":
        if nnz_full == 0:
            return 0.0, 0.0, 0.0
        kl_pos, sum_xhat_nz, sum_x = error_pass(idxs, vals, cores, factors, xf, batch_nnz,
                                                epsilon, desc="shard error x̂ pass")
        out = (float(kl_pos.get()) * weight, float(sum_xhat_nz.get()) * weight,
               float(sum_x.get()) * weight)
        cp.cuda.Device(device_id).synchronize()
        return out
    if kind == "factor":
        out = factor_pass(idxs, vals, cores, factors, arg, xf, batch_nnz,
                          desc=f"shard factors {tuple(arg)}")
    elif kind == "core":
        out = core_pass(idxs, vals, cores, factors, arg, xf, batch_nnz,
                        desc=f"shard core site {arg}")
    else:
        raise ValueError(kind)
    result = cp.asnumpy(out)
    cp.cuda.Device(device_id).synchronize()
    return result


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


def _reduce(sst, cores, factors, shape, kind, arg, epsilon, batch_nnz, bufs=None):
    """Fan the pass out over the shards and return the summed partials (a tuple
    of sums when the worker returns tuples). ``bufs`` = (cores_np, factors_np)
    host copies for shards 1.., built here unless the caller keeps them."""
    cores_buf, factors_buf = bufs or ([cp.asnumpy(C) for C in cores],
                                      [cp.asnumpy(f) for f in factors])
    partials = _fan_out(
        _shard_worker, len(sst.device_ids), sst._pool,
        # Shard 0 lives on the primary: hand it the device arrays directly.
        lambda k: dict(
            shard=sst.shards[k],
            cores_np=cores if k == 0 else cores_buf,
            factors_np=factors if k == 0 else factors_buf,
            shape=shape, kind=kind, arg=arg, epsilon=epsilon, batch_nnz=batch_nnz,
            device_id=sst.device_ids[k], subsample_frac=sst.subsample_frac,
            iteration=sst._iter_seed,
        ),
    )
    if isinstance(partials[0], tuple):
        return tuple(sum(p[i] for p in partials) for i in range(len(partials[0])))
    return np.add.reduce(partials)


# ---------------------------------------------------------------------------
# Orchestrators (same signatures as the single-GPU kernels, plus ``sst``;
# ``vec_tensor`` and ``thread_budget`` are accepted and ignored)
# ---------------------------------------------------------------------------

def sharded_tt_tied_factor_update(sst, core, factors, group, shape, vec_tensor=None,
                                  thread_budget=None, epsilon=1e-12, verbose=False,
                                  batch_nnz=None):
    """Multi-GPU pooled KL update for a factor tied across ``group`` (an untied
    factor is ``group=[mode]``): one fan-out/fan-in cycle for the whole group.
    Every tied mode's core is rescaled in place on the primary."""
    if verbose:
        print(f"  [TT-KL/sharded] Updating factor group {tuple(group)}...")
    shape = tuple(int(s) for s in shape)
    group = list(group)
    with cp.cuda.Device(sst.device_ids[0]):
        A = factors[group[0]]
        # Hoisted: one estimate per call rather than one per shard (each costs a
        # driver memGetInfo). Shard devices are assumed homogeneous, as in CP.
        if batch_nnz is None:
            batch_nnz = estimate_batch_nnz_tt(core, factors)
        den = factor_den(core, factors, group, epsilon)
    num = _reduce(sst, core, factors, shape, "factor", group, epsilon, batch_nnz)
    with cp.cuda.Device(sst.device_ids[0]):
        return apply_factor_step(core, A, cp.asarray(num), den, group, epsilon)


def sharded_tt_core_update(sst, core, factors, shape, modes=None, vec_tensor=None,
                           thread_budget=None, epsilon=1e-12, verbose=False,
                           batch_nnz=None):
    """Multi-GPU KL core sweep: one fan-out/fan-in cycle per site, in order.

    Returns the same ``core`` list (elements rebound on the primary), so the
    training loop's ``core = routing.core_update(...)`` keeps working.
    """
    if verbose:
        print("  [TT-KL/sharded] Updating TT cores...")
    shape = tuple(int(s) for s in shape)
    n_modes = len(shape)
    if modes is not None and list(modes) != list(range(n_modes)):
        raise NotImplementedError("sharded_tt_core_update assumes modes == range(N).")
    primary = sst.device_ids[0]

    with cp.cuda.Device(primary):
        if batch_nnz is None:
            batch_nnz = estimate_batch_nnz_tt(core, factors)
        sums = _colsum_batch(factors, epsilon)

    # The factors are constant for the whole sweep; only the core just written
    # is refreshed below, so the host traffic is one copy of the chain in total.
    bufs = ([cp.asnumpy(C) for C in core], [cp.asnumpy(f) for f in factors])
    for k in range(n_modes):
        with cp.cuda.Device(primary):
            den = core_den(core, sums, k, epsilon)
        num = _reduce(sst, core, factors, shape, "core", k, epsilon, batch_nnz, bufs)
        with cp.cuda.Device(primary):
            apply_core_step(core, k, cp.asarray(num), den, epsilon)
        # Sequential sweep: site k+1 must see this write, so refresh the buffer
        # the other shards read from (only this one core changed).
        bufs[0][k] = cp.asnumpy(core[k])
    return core


def sharded_tt_kl_error(sst, core, factors, shape, vec_tensor=None, thread_budget=None,
                        epsilon=1e-12, verbose=False, batch_nnz=None):
    """Relative generalized KL divergence with sharded NNZ; scalar on the primary.

    Normalized exactly as ``tt_ops.tt_kl_compute_errors``:

        D   = Σ_nnz [x log(x/x̂) − x + x̂] + (Σ_all x̂ − Σ_nnz x̂)
        rel = D / Σ_nnz x
    """
    shape = tuple(int(s) for s in shape)
    kl_pos, sum_xhat_nz, sum_x = _reduce(sst, core, factors, shape, "error", None,
                                         epsilon, batch_nnz)
    with cp.cuda.Device(sst.device_ids[0]):
        # Σ over ALL entries of X̂ — closed form, exact, never sampled.
        sum_all = float(cp.asnumpy(tt_sum_all_entries(core, factors, epsilon=epsilon)))
        # rel = D / Σ_nnz x is undefined with no data; nan rather than D/ε.
        result = ((kl_pos + (sum_all - sum_xhat_nz)) / sum_x if sum_x > 0
                  else float("nan"))
        return cp.asarray(result, dtype=factors[0].dtype)


__all__ = [
    "sharded_tt_core_update",
    "sharded_tt_kl_error",
    "sharded_tt_tied_factor_update",
]
