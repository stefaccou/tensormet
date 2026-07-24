from typing import Callable, Optional, Literal
from dataclasses import dataclass
from functools import partial
from tensormet.distance import (kl_factor_update, kl_core_update, kl_compute_errors,
                                kl_factor_update_largedim, kl_core_update_largedim, kl_compute_errors_largedim,
                                fr_factor_update, fr_core_update, fr_combined_core_errors,
                                fr_factor_update_largedim, fr_core_update_largedim, fr_compute_errors_largedim,
                                fr_combined_core_errors_largedim,
                                null_compute_errors
                      )
# -- Routing function --
Divergence = Literal["kl", "fr"]

# CHANGED (2026-06-12 review, Task 6): single source of truth for the dense vs.
# largedim/NNZ-streaming routing decision. Replaces the scattered literal
# `3000`/`4000` comparisons that previously lived (and disagreed: KL factor used
# 4000 while KL core/error used 3000) in both this module and the multi-GPU
# override in TuckerDecomposition.fit. One constant, one predicate, used for all
# three kernel choices AND the sharding override so factor/core/error always
# select the same family and multi-GPU engages iff the largedim path does.
LARGEDIM_THRESHOLD = 3000


def needs_largedim(dim, largedim: bool = False, masked: bool = False) -> bool:
    """Whether the largedim (NNZ-streaming) kernel family must be used.

    Returns True when the caller forces it (``largedim``), when the masked
    objective is requested (only the largedim kernels implement masking), or
    when the largest mode dimension reaches :data:`LARGEDIM_THRESHOLD`.

    This is the *only* place the size threshold is encoded; callers (routing
    below and the multi-GPU override in ``fit``) must funnel through it so the
    dense/largedim/sharded choice stays consistent across factor, core and
    error kernels.
    """
    _max_dim = max(dim) if isinstance(dim, (tuple, list)) else dim
    return bool(largedim or masked or (_max_dim >= LARGEDIM_THRESHOLD))


@dataclass(frozen=True)
class UpdateRouting:
    factor_update: Callable
    core_update: Callable
    error_fn: Optional[Callable]
    core_returns_error: bool  # True for FR combined core+error



def get_update_routing_step(divergence: Divergence, dim, log_step: bool, largedim=False,
                            masked: bool = False, decomposition: str = "tucker",
                            cp_inner_iters: int = 1,
                            cp_scooch_kappa: float = 0.0) -> UpdateRouting:
    """Return the correct update functions for the step if logging is active.

    masked : bool
        When True, use the weighted/completion ("masked") objective — fit only
        observed (nonzero) entries. The masked branches live exclusively in the
        largedim NNZ-streaming kernels (they work for any dimensionality), so we
        always route to those and bind ``masked=True`` via functools.partial.
    decomposition : str
        EXPERIMENTAL: "cp" routes to the nonnegative CP kernel family in
        tensormet.experimental.CP (imported lazily so Tucker runs never touch
        it). CP has a single NNZ-streaming family, so dim/largedim/masked do
        not apply there (masked CP is rejected upstream by the training loop).
        cp_inner_iters / cp_scooch_kappa are CP-APR knobs, ignored for Tucker.
    """
    if decomposition == "cp":
        from tensormet.experimental.CP.cp_routing import get_cp_update_routing_step
        return get_cp_update_routing_step(
            divergence=divergence, log_step=log_step,
            inner_iters=cp_inner_iters, scooch_kappa=cp_scooch_kappa,
        )

    # Single decision for the whole step: factor, core and error all follow the
    # same family. `masked` is folded in by needs_largedim (masked kernels stream
    # over NNZ and are correct at any size, so they force the largedim path).
    force_large = needs_largedim(dim, largedim=largedim, masked=masked)

    if divergence == "kl":
        factor_fn = kl_factor_update_largedim if force_large else kl_factor_update
        core_fn = kl_core_update_largedim if force_large else kl_core_update
        error_fn = kl_compute_errors_largedim if force_large else kl_compute_errors
        if masked:
            factor_fn = partial(factor_fn, masked=True)
            core_fn = partial(core_fn, masked=True)
            error_fn = partial(error_fn, masked=True)
        return UpdateRouting(
            factor_update=factor_fn,
            core_update=core_fn,
            error_fn=error_fn if log_step else null_compute_errors,
            core_returns_error=False, # KL core update never returns error
        )

    if divergence == "fr":
        if not force_large:
            return UpdateRouting(
                    factor_update=fr_factor_update,
                    core_update=fr_combined_core_errors if log_step else fr_core_update,  # returns (core, rel_error)
                    error_fn=None if log_step else null_compute_errors,
                    core_returns_error=True * log_step,
                )
        else:
            factor_fn = fr_factor_update_largedim
            core_fn = fr_combined_core_errors_largedim if log_step else fr_core_update_largedim
            if masked:
                factor_fn = partial(factor_fn, masked=True)
                core_fn = partial(core_fn, masked=True)
            return UpdateRouting(
                    factor_update=factor_fn,
                    core_update=core_fn,
                    error_fn=None if log_step else null_compute_errors,
                    core_returns_error=True * log_step,
                )

    raise ValueError(f"Unknown divergence: {divergence!r}. Expected 'kl' or 'fr'.")

def get_log_step(iteration, rec_log_every, rec_check_every):
    log_step = False
    if rec_log_every and (iteration+1) % rec_log_every == 0:
        log_step = True
    if rec_check_every and (iteration + 1)  % rec_check_every == 0:
        log_step = True
    return log_step