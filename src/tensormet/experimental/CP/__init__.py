"""
Nonnegative CP (CANDECOMP/PARAFAC) decomposition — EXPERIMENTAL.

Implements reviews/CP_IMPLEMENTATION_PLAN.md as a self-contained package so
the Tucker pipeline stays untouched. See README.md in this directory for the
integration seams in the main package and how to revert them.

Deliberately lazy, matching tensormet.experimental's convention: importing
this package (or having the training loop import a specific submodule) must
not drag in torch/tensorly/cupy until a symbol is actually used.
"""

_SUBMODULE_BY_NAME = {
    "cp_values_at_nnz": "cp_ops",
    "cp_weighted_mttkrp": "cp_ops",
    "cp_fr_factor_update": "cp_ops",
    "cp_kl_factor_update": "cp_ops",
    "cp_weight_update": "cp_ops",
    "cp_fr_combined_weights_errors": "cp_ops",
    "cp_fr_compute_errors": "cp_ops",
    "cp_kl_compute_errors": "cp_ops",
    "cp_normalize_absorb": "cp_ops",
    "initialize_nonnegative_cp": "cp_ops",
    "estimate_batch_nnz_cp": "cp_ops",
    "get_cp_update_routing_step": "cp_routing",
    "CPDecomposition": "cp_decomposition",
}

__all__ = list(_SUBMODULE_BY_NAME)


def __getattr__(name):
    submodule = _SUBMODULE_BY_NAME.get(name)
    if submodule is not None:
        import importlib
        mod = importlib.import_module(f"tensormet.experimental.CP.{submodule}")
        return getattr(mod, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
