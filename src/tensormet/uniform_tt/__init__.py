"""
Shared-core ("uniform") Tucker-TT.

One TT core reused at every position, plus start/end vectors: the n-gram value
is αᵀ S(i_0) ⋯ S(i_{N-1}) ω with one ρ×ρ operator S(i) per word. See README.md
in this directory for the model, the update rules and the integration seams.

Lazy: importing this package must not pull in
torch/tensorly/cupy until a symbol is used.
"""

_SUBMODULE_BY_NAME = {
    "expand_uniform": "utt_chain",
    "initialize_uniform_tt": "utt_ops",
    "estimate_batch_nnz_utt": "utt_ops",
    "utt_sum_all_entries": "utt_ops",
    "utt_kl_tied_factor_update": "utt_ops",
    "utt_kl_core_update": "utt_ops",
    "utt_kl_compute_errors": "utt_ops",
    "get_utt_update_routing_step": "utt_routing",
    "get_sharded_utt_update_routing_step": "utt_routing",
    "UniformTTDecomposition": "utt_decomposition",
}

__all__ = list(_SUBMODULE_BY_NAME)


def __getattr__(name):
    submodule = _SUBMODULE_BY_NAME.get(name)
    if submodule is not None:
        import importlib
        mod = importlib.import_module(f"tensormet.uniform_tt.{submodule}")
        return getattr(mod, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
