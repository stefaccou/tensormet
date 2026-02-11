from typing import Callable, Optional, Literal
from dataclasses import dataclass
from distance import (kl_factor_update, kl_core_update, kl_factor_update_largedim, kl_compute_errors,
                      fr_factor_update, fr_core_update, fr_combined_core_errors, null_compute_errors
                      )
# -- Routing function --
Divergence = Literal["kl", "fr"]

@dataclass(frozen=True)
class UpdateRouting:
    factor_update: Callable
    core_update: Callable
    error_fn: Optional[Callable]
    core_returns_error: bool  # True for FR combined core+error

def get_update_routing(divergence: Divergence, dim: int) -> UpdateRouting:
    """Return the correct update functions for this run."""
    if divergence == "kl":
        factor_fn = kl_factor_update_largedim if dim >= 4000 else kl_factor_update
        return UpdateRouting(
            factor_update=factor_fn,
            core_update=kl_core_update,
            error_fn=kl_compute_errors,
            core_returns_error=False, # KL core update never returns error
        )

    if divergence == "fr":
        return UpdateRouting(
                factor_update=fr_factor_update,
                core_update=fr_combined_core_errors,  # returns (core, rel_error)
                error_fn=None,
                core_returns_error=True,
            )

    raise ValueError(f"Unknown divergence: {divergence!r}. Expected 'kl' or 'fr'.")

def get_update_routing_step(divergence: Divergence, dim: int, log_step:bool) -> UpdateRouting:
    """Return the correct update functions for the step if logging is active."""
    if divergence == "kl":
        factor_fn = kl_factor_update_largedim if dim >= 4000 else kl_factor_update
        return UpdateRouting(
            factor_update=factor_fn,
            core_update=kl_core_update,
            error_fn=kl_compute_errors if log_step else null_compute_errors,
            core_returns_error=False, # KL core update never returns error
        )

    if divergence == "fr":
        return UpdateRouting(
                factor_update=fr_factor_update,
                core_update=fr_combined_core_errors if log_step else fr_core_update,  # returns (core, rel_error)
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