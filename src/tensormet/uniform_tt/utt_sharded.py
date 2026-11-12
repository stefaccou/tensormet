"""
utt_sharded.py — multi-GPU (NNZ-sharded) uniform-TT kernels.

Same structure as tt_hybrid/tt_sharded.py: one worker per shard
runs the NNZ pass from utt_ops (TT's batch machinery on the expanded chain) on
its own GPU and returns a partial numerator; the primary adds them and applies
the step. All denominators are closed forms and stay on the primary.

    reduced across shards   factor numerator (V, R) | core numerator (ρ, R, ρ)
                            | boundary numerators + Σx (2ρ+1,) | 3 error scalars
    primary                 denominators, x̂ floor, the steps, the gauge

Per iteration: one reduce for the factor, one for the core, one for the
boundaries (vs N + 1 for the per-position TT).

The orchestrators read ``shards / device_ids / subsample_frac / _iter_seed /
_pool`` off the ShardedSparseTensor directly, so sharded_sparse.py needs no
uniform-TT delegates.
"""
from __future__ import annotations

import numpy as np

from tensormet.distance import coo_to_coords
from tensormet.sharded_sparse import apply_subsample
from tensormet.utils import make_lazy_cupy_pair
from tensormet.tt_hybrid.tt_sharded import _fan_out
from tensormet.uniform_tt.utt_ops import (
    _check_core, _xhat_floor, apply_factor_step,
    boundary_pass, core_and_boundary_steps, core_pass, error_pass, estimate_batch_nnz_utt,
    factor_pass, tied_factor, tied_factor_setup, utt_sum_all_entries,
)

cp, cpx_sparse = make_lazy_cupy_pair()


# ---------------------------------------------------------------------------
# Per-shard worker
# ---------------------------------------------------------------------------

def _utt_shard_worker(shard, core_np, A_np, shape, kind, xf, epsilon, batch_nnz,
                      device_id, subsample_frac=1.0, iteration=None, loglik=False):
    """Run one NNZ pass on a shard. ``kind`` is 'factor', 'core', 'boundary' or
    'error'. Numerators are linear in the values, so the subsample is rescaled
    by 1/frac; the error is not (x log x is nonlinear), so it is sampled without
    rescaling and the summed scalars are reweighted by nnz/n_sample instead.
    ``loglik`` (core/boundary) also returns Σ x log x̂, linear in x like the
    numerators, so the rescaled window is right for it too."""
    cp.cuda.Device(device_id).use()
    core_d = [cp.asarray(x) for x in core_np]
    A_d = cp.asarray(A_np)

    idxs, vals = coo_to_coords(shard, shape)
    nnz_full = int(vals.size)
    weight = 1.0
    if subsample_frac < 1.0 and nnz_full:
        if kind == "error":
            idxs, vals = apply_subsample(idxs, vals, subsample_frac, iteration, rescale=False)
            weight = nnz_full / max(1, int(vals.size))
        else:
            idxs, vals = apply_subsample(idxs, vals, subsample_frac, iteration)

    if kind == "factor":
        out = factor_pass(idxs, vals, A_d, core_d, xf, batch_nnz)
    elif kind in ("core", "boundary"):
        fn = core_pass if kind == "core" else boundary_pass
        out = fn(idxs, vals, A_d, core_d, xf, batch_nnz, loglik=loglik)
        if loglik:
            result = (cp.asnumpy(out[0]), float(out[1].get()))
            cp.cuda.Device(device_id).synchronize()
            return result
    elif kind == "error":
        if nnz_full == 0:
            return 0.0, 0.0, 0.0
        kl_pos, sum_xhat, sum_x = error_pass(idxs, vals, A_d, core_d, xf, batch_nnz, epsilon)
        out = (float(kl_pos.get()) * weight, float(sum_xhat.get()) * weight,
               float(sum_x.get()) * weight)
        cp.cuda.Device(device_id).synchronize()
        return out
    else:
        raise ValueError(kind)

    result = cp.asnumpy(out)
    cp.cuda.Device(device_id).synchronize()
    return result


def _reduce(sst, core, A, shape, kind, xf, epsilon, batch_nnz, loglik=False):
    """Fan the pass out over the shards and return the summed partials
    (a tuple of sums when the worker returns tuples)."""
    core_buf = [cp.asnumpy(x) for x in core]
    A_buf = cp.asnumpy(A)
    partials = _fan_out(
        _utt_shard_worker, len(sst.device_ids), sst._pool,
        # Shard 0 lives on the primary: hand it the device arrays directly.
        lambda k: dict(
            shard=sst.shards[k],
            core_np=core if k == 0 else core_buf,
            A_np=A if k == 0 else A_buf,
            shape=shape, kind=kind, xf=xf, epsilon=epsilon, batch_nnz=batch_nnz,
            device_id=sst.device_ids[k], subsample_frac=sst.subsample_frac,
            iteration=sst._iter_seed, loglik=loglik,
        ),
    )
    if isinstance(partials[0], tuple):
        return tuple(sum(p[i] for p in partials) for i in range(len(partials[0])))
    return np.add.reduce(partials)


# ---------------------------------------------------------------------------
# Orchestrators (same signatures as the single-GPU kernels, plus ``sst``)
# ---------------------------------------------------------------------------

def sharded_utt_tied_factor_update(sst, core, factors, group, shape, epsilon=1e-12,
                                   verbose=False, batch_nnz=None, power=None):
    shape = tuple(int(s) for s in shape)
    if verbose:
        print("  [UTT-KL/sharded] Updating shared factor...")
    primary = sst.device_ids[0]
    with cp.cuda.Device(primary):
        A, den, xf, batch_nnz = tied_factor_setup(core, factors, group, shape, epsilon,
                                                  batch_nnz)
    num = _reduce(sst, core, A, shape, "factor", xf, epsilon, batch_nnz)
    with cp.cuda.Device(primary):
        return apply_factor_step(core, A, cp.asarray(num), den, len(shape), epsilon, power)


def sharded_utt_core_update(sst, core, factors, shape, epsilon=1e-12, verbose=False,
                            batch_nnz=None, power=None, boundary_power=None):
    shape = tuple(int(s) for s in shape)
    _check_core(core, len(shape))
    if verbose:
        print("  [UTT-KL/sharded] Updating shared core...")
    primary = sst.device_ids[0]
    # Everything off the shards (denominators, steps, gauge) stays on the
    # primary; the shard workers set their own devices in the pool threads.
    with cp.cuda.Device(primary):
        A = tied_factor(factors)
        if batch_nnz is None:
            batch_nnz = estimate_batch_nnz_utt(core, factors)

        def run_pass(kind, state, xf, loglik):
            out = _reduce(sst, state, A, shape, kind, xf, epsilon, batch_nnz, loglik)
            return (cp.asarray(out[0]), out[1]) if loglik else cp.asarray(out)

        return core_and_boundary_steps(core, A, shape, epsilon, run_pass, power,
                                       boundary_power)


def sharded_utt_kl_error(sst, core, factors, shape, epsilon=1e-12, batch_nnz=None):
    """Relative KL error, normalized as utt_kl_compute_errors. Evaluated on this
    iteration's subsample window and reweighted per shard, as on the other
    sharded paths; Σ_all x̂ is closed form and exact."""
    shape = tuple(int(s) for s in shape)
    _check_core(core, len(shape))
    primary = sst.device_ids[0]
    with cp.cuda.Device(primary):
        A = factors[0]
        if batch_nnz is None:
            batch_nnz = estimate_batch_nnz_utt(core, factors)
        xf = _xhat_floor(core, A, shape, epsilon)
        sum_all = float(utt_sum_all_entries(core, [A], epsilon, n_modes=len(shape)))
    kl_pos, sum_xhat_nz, sum_x = _reduce(sst, core, A, shape, "error", xf, epsilon, batch_nnz)
    result = (kl_pos + (sum_all - sum_xhat_nz)) / sum_x if sum_x > 0 else float("nan")
    with cp.cuda.Device(primary):
        return cp.asarray(result, dtype=A.dtype)


__all__ = [
    "sharded_utt_core_update",
    "sharded_utt_kl_error",
    "sharded_utt_tied_factor_update",
]
