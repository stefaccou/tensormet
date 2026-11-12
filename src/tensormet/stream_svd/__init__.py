"""
Streamed GPU SVD init (``--init svd_stream``).

Matrix-free randomized SVD of the mode unfoldings, one pooled basis per group
of tied modes, spread over the run's GPUs and cached in the decomposition
folder. See stream_svd.py for the algorithm and init_cache.py for the cache.

Lazy: importing this package must not
pull in cupy until a symbol is used.
"""

_SUBMODULE_BY_NAME = {
    "initialize_svd_stream": "stream_svd",
    "set_run_context": "stream_svd",
    "InitCache": "init_cache",
}

__all__ = list(_SUBMODULE_BY_NAME)


def __getattr__(name):
    submodule = _SUBMODULE_BY_NAME.get(name)
    if submodule is not None:
        import importlib
        mod = importlib.import_module(f"tensormet.stream_svd.{submodule}")
        return getattr(mod, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
