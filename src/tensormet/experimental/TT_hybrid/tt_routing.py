"""
tt_routing.py — UpdateRouting construction for the Tucker-TT hybrid.

EXPERIMENTAL. Maps the TT kernels onto the ``UpdateRouting`` contract the
Tucker training loop already consumes:

    factor_update      → TT factor MU kernel
    core_update        → sequential MU sweep over the TT cores
    error_fn           → TT KL error / null_compute_errors
    core_returns_error → always False (there is no fused FR variant here)

The hybrid has a single NNZ-streaming kernel family, so Tucker's
dense-vs-largedim split (``needs_largedim``) does not apply. Imported lazily
from ``tensormet.routing`` only when a run asks for ``decomposition == "tt"``.
"""
from __future__ import annotations

from tensormet.distance import null_compute_errors
from tensormet.routing import Divergence, UpdateRouting
from tensormet.experimental.TT_hybrid.tt_ops import (
    tt_kl_compute_errors, tt_kl_core_update, tt_kl_factor_update,
)


def get_tt_update_routing_step(divergence: Divergence, log_step: bool) -> UpdateRouting:
    """Return the Tucker-TT update functions for one training step."""
    if divergence != "kl":
        raise NotImplementedError(
            f"decomposition='tt' implements the KL/Poisson divergence only; got "
            f"{divergence!r}. The Frobenius denominator needs a doubled (Gram-weighted) "
            f"chain — see README.md, 'Not implemented'."
        )
    return UpdateRouting(
        factor_update=tt_kl_factor_update,
        core_update=tt_kl_core_update,
        error_fn=tt_kl_compute_errors if log_step else null_compute_errors,
        core_returns_error=False,
    )


def get_sharded_tt_update_routing_step(sst, divergence: Divergence,
                                       log_step: bool) -> UpdateRouting:
    """Multi-GPU counterpart of :func:`get_tt_update_routing_step`.

    Same contract, with the NNZ-dependent halves routed through *sst*
    (a ``ShardedSparseTensor``). ``vec_tensor`` is accepted and ignored by every
    callable — the SST owns its shards and its own subsample window.

    Unlike CP, the core slot *does* need sharding: the TT core sweep is a
    per-site MU over the NNZ (see tt_sharded.py). The cores are still mutated on
    the primary, so ``core`` stays the loop's live list either way.
    """
    if divergence != "kl":
        raise NotImplementedError(
            f"decomposition='tt' implements the KL/Poisson divergence only; got "
            f"{divergence!r}. The Frobenius denominator needs a doubled (Gram-weighted) "
            f"chain — see README.md, 'Not implemented'."
        )

    def _factor_update(vec_tensor, core, factors, mode, shape,
                       thread_budget=None, epsilon=1e-12, verbose=False,
                       batch_nnz=None):
        return sst.tt_factor_update(
            core=core, factors=factors, mode=mode, shape=shape,
            thread_budget=thread_budget, epsilon=epsilon, verbose=verbose,
            batch_nnz=batch_nnz,
        )

    def _core_update(vec_tensor, shape, core, factors, modes=None,
                     thread_budget=None, epsilon=1e-12, verbose=False,
                     batch_nnz=None):
        return sst.tt_core_update(
            shape=shape, core=core, factors=factors, modes=modes,
            thread_budget=thread_budget, epsilon=epsilon, verbose=verbose,
            batch_nnz=batch_nnz,
        )

    def _error_fn(vec_tensor, shape, core, factors,
                  thread_budget=None, epsilon=1e-12, verbose=False,
                  batch_nnz=None):
        return sst.tt_compute_errors(
            shape=shape, core=core, factors=factors,
            thread_budget=thread_budget, epsilon=epsilon, verbose=verbose,
            batch_nnz=batch_nnz,
        )

    return UpdateRouting(
        factor_update=_factor_update,
        core_update=_core_update,
        error_fn=_error_fn if log_step else null_compute_errors,
        core_returns_error=False,
    )


__all__ = [
    "get_tt_update_routing_step",
    "get_sharded_tt_update_routing_step",
    "tt_kl_compute_errors",
]
