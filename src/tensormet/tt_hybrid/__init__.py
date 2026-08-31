"""
Tucker-TT hybrid decomposition — EXPERIMENTAL.

Tucker factor matrices, tensor-train core. See README.md for the model, the
integration seams in the main package, and how to revert them.

Lazy by convention (as in tensormet.experimental.CP): importing this package
must not drag in torch/tensorly/cupy until a symbol is actually used.

Note: the directory is ``TT_hybrid`` rather than ``TT-hybrid`` because a
hyphen is not a valid Python identifier.
"""

_SUBMODULE_BY_NAME = {
    "bond_dims": "tt_chain",
    "core_shapes": "tt_chain",
    "contract": "tt_chain",
    "to_dense_core": "tt_chain",
    "initialize_tucker_tt": "tt_ops",
    "estimate_batch_nnz_tt": "tt_ops",
    "tt_sum_all_entries": "tt_ops",
    "tt_kl_factor_update": "tt_ops",
    "tt_kl_core_update": "tt_ops",
    "tt_kl_compute_errors": "tt_ops",
    "get_tt_update_routing_step": "tt_routing",
    "get_sharded_tt_update_routing_step": "tt_routing",
    "TuckerTTDecomposition": "tt_decomposition",
    "TuckerTTTensor": "tt_decomposition",
}

__all__ = list(_SUBMODULE_BY_NAME)


def __getattr__(name):
    submodule = _SUBMODULE_BY_NAME.get(name)
    if submodule is not None:
        import importlib
        mod = importlib.import_module(f"tensormet.experimental.TT_hybrid.{submodule}")
        return getattr(mod, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
