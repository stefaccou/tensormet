"""MU-style inits (random or any SVD variant) as a starting point for the SGD solver.

The init routines are CuPy code, while SGD keeps the tensor as a torch COO. So the
init runs on a temporary CuPy copy and returns NumPy ``(core, factors)``, the same
form ``--sgd-warm-start`` passes. The SGD models then start from it exactly as they
start from a saved MU model.
"""
from __future__ import annotations

import gc
import inspect
import time

import tensorly as tl

from tensormet.utils import make_lazy_cupy_pair

cp, _cpx_sparse = make_lazy_cupy_pair()


def mu_init_for_sgd(sparse_tensor, decomposition, shape, rank, modes, init, random_state,
                    *, tt_rank, thread_budget, epsilon, device_ids):
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
            from tensormet.experimental.uniform_tt.utt_ops import initialize_uniform_tt
            core, factors = initialize_uniform_tt(
                tensor_cp, shape, rank, modes, init, random_state,
                tt_rank=tt_rank, thread_budget=thread_budget, epsilon=epsilon)
        elif decomposition == "tt":
            from tensormet.tt_hybrid.tt_ops import initialize_tucker_tt
            core, factors = initialize_tucker_tt(
                tensor_cp, shape, rank, modes, init, random_state,
                tt_rank=tt_rank, thread_budget=thread_budget, epsilon=epsilon)
        else:
            from tensormet.sparse_ops import initialize_nonnegative_tucker
            core, factors = initialize_nonnegative_tucker(
                tensor_cp, shape, rank, modes, init, random_state,
                thread_budget=thread_budget)

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
            _print_holders(d)


def _print_holders(device_id, min_bytes=2**30):
    """Print what still references the large CuPy arrays on one GPU (debug aid)."""
    big = [o for o in gc.get_objects()
           if isinstance(o, cp.ndarray) and o.device.id == device_id and o.nbytes >= min_bytes]
    if not big:
        print("  no large CuPy array is visible to gc; the holder is outside Python objects")
    for a in big:
        print(f"  live array {a.shape} {a.dtype} ({a.nbytes / 2**30:.2f} GiB), referenced by:")
        for r in gc.get_referrers(a):
            if r is big or (inspect.isframe(r) and r.f_code.co_name == "_print_holders"):
                continue
            print(f"    {_describe(r)}")


def _describe(r):
    if inspect.isframe(r):
        held_by = [type(h).__name__ if not inspect.isframe(h) else f"frame of {h.f_code.co_name}"
                   for h in gc.get_referrers(r) if not inspect.isframe(h) or h.f_back is r][:4]
        return (f"local variable in {r.f_code.co_name} ({r.f_code.co_filename}:{r.f_lineno}); "
                f"that frame is held by {held_by}")
    if isinstance(r, dict):
        for owner in gc.get_referrers(r):
            if inspect.ismodule(owner) and owner.__dict__ is r:
                return f"global in module {owner.__name__}"
            if getattr(owner, "__dict__", None) is r:
                return f"attribute of a {type(owner).__module__}.{type(owner).__name__}"
        return f"dict with keys {list(r)[:8]}"
    if isinstance(r, (list, tuple)):
        return f"{type(r).__name__} of length {len(r)}"
    if isinstance(r, cp.ndarray):
        return f"view {r.shape} {r.dtype} (.base of another array)"
    return f"{type(r).__module__}.{type(r).__name__}"
