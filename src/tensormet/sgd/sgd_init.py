"""MU-style inits (random or any SVD variant) as a starting point for the SGD solver.

The init routines are CuPy code, while SGD keeps the tensor as a torch COO. So the
init runs on a temporary CuPy copy and returns NumPy ``(core, factors)``, the same
form ``--sgd-warm-start`` passes. The SGD models then start from it as they start
from a saved MU model. A Tucker core is first rescaled so that Σ_all x̂ = Σx
(``_match_data_sum``); the TT/utt inits already do that themselves.
"""
from __future__ import annotations

import gc
import math
import time

import tensorly as tl

from tensormet.utils import make_lazy_cupy_pair

cp, _cpx_sparse = make_lazy_cupy_pair()


def mu_init_for_sgd(sparse_tensor, decomposition, shape, rank, modes, init, random_state,
                    *, tt_rank, thread_budget, epsilon, device_ids, core_init="random"):
    """Run the MU path's init for ``decomposition`` on a CuPy copy of ``sparse_tensor``
    (a torch-backed SparseTupleTensor). Returns NumPy ``(core, factors)``; for tt/utt
    ``core`` is a list. CuPy memory is released on every GPU before returning, so the
    SGD trainer gets the GPUs back.
    """
    # svd_stream reads per-mode coordinates; the other SVD inits need the block form.
    tensor_cp = sparse_tensor.sparse_representation(
        "cupy", use_coords=True if init == "svd_stream" else None)
    print(f"[{time.strftime('%H:%M:%S')}] SGD: running init={init!r} on a CuPy copy",
          flush=True)
    with tl.backend_context("cupy"):
        if decomposition == "utt":
            from tensormet.uniform_tt.utt_ops import initialize_uniform_tt
            core, factors = initialize_uniform_tt(
                tensor_cp, shape, rank, modes, init, random_state,
                tt_rank=tt_rank, thread_budget=thread_budget, epsilon=epsilon,
                core_init=core_init)
        elif decomposition == "tt":
            from tensormet.tt_hybrid.tt_ops import initialize_tucker_tt
            core, factors = initialize_tucker_tt(
                tensor_cp, shape, rank, modes, init, random_state,
                tt_rank=tt_rank, thread_budget=thread_budget, epsilon=epsilon,
                core_init=core_init)
        else:
            from tensormet.sparse_ops import initialize_nonnegative_tucker
            core, factors = initialize_nonnegative_tucker(
                tensor_cp, shape, rank, modes, init, random_state,
                thread_budget=thread_budget)
            core = _match_data_sum(core, factors, tensor_cp)

    core_np = ([cp.asnumpy(C) for C in core] if isinstance(core, (list, tuple))
               else cp.asnumpy(core))
    factors_np = [cp.asnumpy(f) for f in factors]
    del core, factors
    # Empty the copy itself, so its ~11 GiB goes even if something still points at
    # the object. Then collect: an array kept alive only by a reference cycle would
    # otherwise survive free_all_blocks. CuPy keeps freed blocks in its pools; torch
    # can't use that memory until released.
    if hasattr(tensor_cp, "coords"):
        tensor_cp.coords = tensor_cp.data = None
    del tensor_cp
    gc.collect()
    for d in device_ids:
        with cp.cuda.Device(d):
            cp.get_default_memory_pool().free_all_blocks()
    cp.get_default_pinned_memory_pool().free_all_blocks()
    _report_leftover(device_ids)
    return core_np, factors_np


def _match_data_sum(core, factors, tensor_cp):
    """Scale a Tucker init's core so that Σ_all x̂ = Σx, as the TT/utt inits do.

    The SVD inits' core is the projection X ×ₙ Fₙᵀ onto NNDSVD columns of norm
    ≈ √σ, so x̂ starts orders of magnitude too large (M-2 in
    reviews/tensormet-review-2026-10-29.md). MU's first step absorbs that; Adam
    does not. This is the exact KL scale and the right order of magnitude for FR.
    """
    from tensormet.distance import _tucker_sum_all_entries
    sum_x = float(cp.sum(tensor_cp.data, dtype=cp.float64))
    sum_xhat = float(_tucker_sum_all_entries(core, factors))
    if not (math.isfinite(sum_xhat) and sum_xhat > 0.0 and sum_x > 0.0):
        raise FloatingPointError(f"SGD init: cannot rescale the core (Σx̂ = {sum_xhat!r}, "
                                 f"Σx = {sum_x!r}).")
    print(f"[{time.strftime('%H:%M:%S')}] SGD init: Σx̂/Σx = {sum_xhat / sum_x:.4g}; "
          f"core rescaled to Σx̂ = Σx", flush=True)
    return core * core.dtype.type(sum_x / sum_xhat)


def _report_leftover(device_ids):
    """Print free memory and what CuPy still holds per GPU, so a leak shows in the log."""
    for d in device_ids:
        with cp.cuda.Device(d):
            held = cp.get_default_memory_pool().used_bytes()
            free = cp.cuda.runtime.memGetInfo()[0]
        print(f"[{time.strftime('%H:%M:%S')}] SGD init done: GPU {d} has {free / 2**30:.1f} GiB "
              f"free, CuPy still holds {held / 2**30:.2f} GiB", flush=True)
        if held > 2**30:
            print(f"WARNING: a CuPy array from the init is still referenced on GPU {d}; "
                  f"the SGD trainer may run out of memory", flush=True)
