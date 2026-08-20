"""
cp_routing.py — UpdateRouting construction for the nonnegative CP family.

EXPERIMENTAL (see reviews/CP_IMPLEMENTATION_PLAN.md, §2/§Phase 2).

Maps the CP kernels onto the exact ``UpdateRouting`` contract the Tucker loop
already consumes (``routing.py``):

    factor_update      → CP factor MU kernel (λ updated in place inside it)
    core_update        → λ passthrough (KL), or the fused "λ + error" variant
                         mirroring fr_combined_core_errors' contract (FR)
    error_fn           → CP error kernels / null_compute_errors
    core_returns_error → False for KL; True on FR log steps

CP has no dense-Z formulation worth keeping, so the Tucker dense-vs-largedim
split (``needs_largedim``) is irrelevant here: the single NNZ-streaming
family is simultaneously the memory-safe and the fast path (plan, Phase 1
design note).

This module is imported lazily from ``tensormet.routing`` only when a run
requests ``decomposition == "cp"`` — Tucker runs never touch it.
"""
from __future__ import annotations

from functools import partial

from tensormet.routing import UpdateRouting, Divergence
from tensormet.distance import null_compute_errors
from tensormet.experimental.CP.cp_ops import (
    cp_fr_factor_update,
    cp_kl_factor_update,
    cp_weight_update,
    cp_fr_combined_weights_errors,
    cp_kl_compute_errors,
    cp_fr_compute_errors,
)


def get_cp_update_routing_step(divergence: Divergence, log_step: bool,
                               inner_iters: int = 1,
                               scooch_kappa: float = 0.0) -> UpdateRouting:
    """Return the CP update functions for one training step.

    Parameters mirror ``routing.get_update_routing_step`` where meaningful;
    ``dim``/``largedim`` are absent on purpose (single kernel family, see
    module docstring). ``masked`` is not supported for CP yet — the caller
    (the training loop) raises before routing is ever requested.
    """
    if divergence == "kl":
        factor_fn = cp_kl_factor_update
        if inner_iters != 1 or scooch_kappa != 0.0:
            factor_fn = partial(factor_fn, inner_iters=inner_iters,
                                scooch_kappa=scooch_kappa)
        return UpdateRouting(
            factor_update=factor_fn,
            core_update=cp_weight_update,  # λ passthrough; never returns error
            error_fn=cp_kl_compute_errors if log_step else null_compute_errors,
            core_returns_error=False,
        )

    if divergence == "fr":
        return UpdateRouting(
            factor_update=cp_fr_factor_update,
            # Log steps: fused λ passthrough + exact FR error, same
            # (core, rel_error) contract as fr_combined_core_errors.
            core_update=cp_fr_combined_weights_errors if log_step else cp_weight_update,
            error_fn=None if log_step else null_compute_errors,
            core_returns_error=bool(log_step),
        )

    raise ValueError(f"Unknown divergence: {divergence!r}. Expected 'kl' or 'fr'.")


def get_sharded_cp_update_routing_step(sst, divergence: Divergence,
                                       log_step: bool,
                                       inner_iters: int = 1,
                                       scooch_kappa: float = 0.0) -> UpdateRouting:
    """Multi-GPU counterpart of :func:`get_cp_update_routing_step`.

    Same contract, with the NNZ-dependent halves routed through *sst*
    (a ``ShardedSparseTensor``). ``vec_tensor`` is accepted and ignored by every
    callable — the SST owns its shards, and its own subsample window.

    The λ "core slot" needs no sharding: ``cp_weight_update`` is a passthrough,
    because the factor updates already absorbed λ.
    """
    if divergence not in ("kl", "fr"):
        raise ValueError(f"Unknown divergence: {divergence!r}. Expected 'kl' or 'fr'.")

    def _factor_update(vec_tensor, core, factors, mode, shape,
                       thread_budget=None, epsilon=1e-12, verbose=False):
        return sst.cp_factor_update(
            core=core, factors=factors, mode=mode, shape=shape,
            divergence=divergence, thread_budget=thread_budget,
            epsilon=epsilon, verbose=verbose,
            inner_iters=inner_iters, scooch_kappa=scooch_kappa,
        )

    def _error_fn(vec_tensor, shape, core, factors,
                  thread_budget=None, epsilon=1e-12, verbose=False):
        return sst.cp_compute_errors(
            shape=shape, core=core, factors=factors, divergence=divergence,
            thread_budget=thread_budget, epsilon=epsilon, verbose=verbose,
        )

    if divergence == "kl":
        return UpdateRouting(
            factor_update=_factor_update,
            core_update=cp_weight_update,
            error_fn=_error_fn if log_step else null_compute_errors,
            core_returns_error=False,
        )

    def _combined_weights_errors(vec_tensor, shape, core, factors, modes=None,
                                 thread_budget=None, epsilon=1e-12, verbose=False):
        # λ passthrough (the ε-clip) + the sharded exact FR error, matching
        # cp_fr_combined_weights_errors' (core, rel_error) contract.
        weights = cp_weight_update(vec_tensor, shape, core, factors, modes=modes,
                                   thread_budget=thread_budget, epsilon=epsilon,
                                   verbose=verbose)
        return weights, _error_fn(vec_tensor, shape, weights, factors,
                                  thread_budget=thread_budget, epsilon=epsilon,
                                  verbose=verbose)

    return UpdateRouting(
        factor_update=_factor_update,
        core_update=_combined_weights_errors if log_step else cp_weight_update,
        error_fn=None if log_step else null_compute_errors,
        core_returns_error=bool(log_step),
    )


# Re-exported for tests / callers that want the standalone error kernels.
__all__ = [
    "get_cp_update_routing_step",
    "get_sharded_cp_update_routing_step",
    "cp_kl_compute_errors",
    "cp_fr_compute_errors",
]
