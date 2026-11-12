"""
utt_routing.py — UpdateRouting for the shared-core ("uniform") Tucker-TT.

Maps the uniform-TT kernels onto the contract the Tucker training
loop consumes:

    tied_factor_update  → pooled update of the one shared factor
    core_update         → shared core step, then the (α, ω) step
    error_fn            → KL error / null_compute_errors
    factor_update       → refuses (all modes are tied, so it is never reached)

Imported lazily from ``tensormet.routing`` only for ``decomposition == "utt"``.
"""
from __future__ import annotations

from functools import partial

from tensormet.distance import null_compute_errors
from tensormet.routing import Divergence, UpdateRouting
from tensormet.uniform_tt.utt_ops import (
    utt_kl_compute_errors, utt_kl_core_update, utt_kl_factor_update,
    utt_kl_tied_factor_update,
)


def _require_kl(divergence):
    if divergence != "kl":
        raise NotImplementedError(
            f"decomposition='utt' implements the KL/Poisson divergence only; got {divergence!r}."
        )


def get_utt_update_routing_step(divergence: Divergence, log_step: bool,
                                core_power=None) -> UpdateRouting:
    """``core_power`` is the core MU exponent γ (None = 1, undamped; 1/N is MM-safe)
    or an ``AdaptiveCorePower``, which must be the same object every iteration."""
    _require_kl(divergence)
    return UpdateRouting(
        factor_update=utt_kl_factor_update,
        core_update=partial(utt_kl_core_update, power=core_power),
        error_fn=utt_kl_compute_errors if log_step else null_compute_errors,
        core_returns_error=False,
        tied_factor_update=utt_kl_tied_factor_update,
    )


def get_sharded_utt_update_routing_step(sst, divergence: Divergence, log_step: bool,
                                        core_power=None) -> UpdateRouting:
    """Multi-GPU counterpart: same contract, NNZ passes run on *sst*'s shards.
    ``vec_tensor`` is accepted and ignored — the SST owns its shards and window."""
    _require_kl(divergence)
    from tensormet.uniform_tt.utt_sharded import (
        sharded_utt_core_update, sharded_utt_kl_error, sharded_utt_tied_factor_update,
    )

    def _tied_factor_update(vec_tensor, core, factors, group, shape,
                            thread_budget=None, epsilon=1e-12, verbose=False,
                            batch_nnz=None):
        return sharded_utt_tied_factor_update(
            sst, core=core, factors=factors, group=group, shape=shape,
            epsilon=epsilon, verbose=verbose, batch_nnz=batch_nnz,
        )

    def _core_update(vec_tensor, shape, core, factors, modes=None,
                     thread_budget=None, epsilon=1e-12, verbose=False,
                     batch_nnz=None):
        return sharded_utt_core_update(
            sst, core=core, factors=factors, shape=shape, epsilon=epsilon,
            verbose=verbose, batch_nnz=batch_nnz, power=core_power,
        )

    def _error_fn(vec_tensor, shape, core, factors, thread_budget=None,
                  epsilon=1e-12, verbose=False, batch_nnz=None):
        return sharded_utt_kl_error(sst, core=core, factors=factors, shape=shape,
                                    epsilon=epsilon, batch_nnz=batch_nnz)

    return UpdateRouting(
        factor_update=utt_kl_factor_update,
        core_update=_core_update,
        error_fn=_error_fn if log_step else null_compute_errors,
        core_returns_error=False,
        tied_factor_update=_tied_factor_update,
    )


__all__ = ["get_utt_update_routing_step", "get_sharded_utt_update_routing_step"]
