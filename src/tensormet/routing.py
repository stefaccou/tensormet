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

@dataclass(frozen=True)
class UpdateRouting:
    factor_update: Callable
    core_update: Callable
    error_fn: Optional[Callable]
    core_returns_error: bool  # True for FR combined core+error



def get_update_routing_step(divergence: Divergence, dim, log_step: bool, largedim=False,
                            masked: bool = False) -> UpdateRouting:
    """Return the correct update functions for the step if logging is active.

    masked : bool
        When True, use the weighted/completion ("masked") objective — fit only
        observed (nonzero) entries. The masked branches live exclusively in the
        largedim NNZ-streaming kernels (they work for any dimensionality), so we
        always route to those and bind ``masked=True`` via functools.partial.
    """
    _max_dim = max(dim) if isinstance(dim, (tuple, list)) else dim
    # Masked kernels stream over NNZ and are correct at any size, so force the
    # largedim path whenever the masked objective is requested.
    force_large = largedim or masked

    if divergence == "kl":
        factor_fn = kl_factor_update_largedim if (_max_dim >= 4000 or force_large) else kl_factor_update
        core_fn = kl_core_update_largedim if (_max_dim >= 3000 or force_large) else kl_core_update
        error_fn = kl_compute_errors_largedim if (_max_dim >= 3000 or force_large) else kl_compute_errors
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
        if _max_dim <= 4000 and not force_large:
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