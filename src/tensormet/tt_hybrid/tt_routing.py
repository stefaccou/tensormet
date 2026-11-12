"""
tt_routing.py — UpdateRouting construction for the Tucker-TT hybrid.

EXPERIMENTAL. Maps the TT kernels onto the ``UpdateRouting`` contract the
Tucker training loop already consumes:

    factor_update       → the tied update with ``group=[mode]``
    tied_factor_update  → pooled TT factor MU kernel
    core_update         → sequential MU sweep over the TT cores
    error_fn            → TT KL error / null_compute_errors
    core_returns_error  → always False (there is no fused FR variant here)

The hybrid has a single NNZ-streaming kernel family, so Tucker's
dense-vs-largedim split (``needs_largedim``) does not apply. Imported lazily
from ``tensormet.routing`` only when a run asks for ``decomposition == "tt"``.
"""
from __future__ import annotations

from functools import partial

from tensormet.distance import null_compute_errors
from tensormet.routing import Divergence, UpdateRouting
from tensormet.tt_hybrid.tt_ops import (
    tt_kl_compute_errors, tt_kl_core_update, tt_kl_factor_update,
    tt_kl_tied_factor_update,
)


def _require_kl(divergence):
    if divergence != "kl":
        raise NotImplementedError(
            f"decomposition='tt' implements the KL/Poisson divergence only; got "
            f"{divergence!r}. The Frobenius denominator needs a doubled (Gram-weighted) "
            f"chain — see README.md, 'Not implemented'."
        )


def get_tt_update_routing_step(divergence: Divergence, log_step: bool) -> UpdateRouting:
    """Return the Tucker-TT update functions for one training step."""
    _require_kl(divergence)
    return UpdateRouting(
        factor_update=tt_kl_factor_update,
        core_update=tt_kl_core_update,
        error_fn=tt_kl_compute_errors if log_step else null_compute_errors,
        core_returns_error=False,
        tied_factor_update=tt_kl_tied_factor_update,
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
    _require_kl(divergence)
    from tensormet.tt_hybrid.tt_sharded import (
        sharded_tt_core_update, sharded_tt_kl_error, sharded_tt_tied_factor_update,
    )

    tied = partial(sharded_tt_tied_factor_update, sst)

    def factor_update(core, factors, mode, shape, **kwargs):
        return tied(core=core, factors=factors, group=[mode], shape=shape, **kwargs)

    return UpdateRouting(
        factor_update=factor_update,
        core_update=partial(sharded_tt_core_update, sst),
        error_fn=partial(sharded_tt_kl_error, sst) if log_step else null_compute_errors,
        core_returns_error=False,
        tied_factor_update=tied,
    )


__all__ = [
    "get_tt_update_routing_step",
    "get_sharded_tt_update_routing_step",
    "tt_kl_compute_errors",
]
