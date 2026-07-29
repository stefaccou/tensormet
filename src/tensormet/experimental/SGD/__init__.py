"""EXPERIMENTAL SGD Tucker solver (see README.md for the integration seams).

Deliberately lazy, like tensormet.experimental itself: importing this package
must not drag in torch until a symbol is actually used (submit.py runs from
login nodes without GPU stacks).
"""

_SUBMODULE_BY_NAME = {
    "SGDTuckerModel": "sgd_tucker",
    "EntryBatcher": "sgd_tucker",
    "GradStepper": "sgd_tucker",
    "full_relative_error": "sgd_tucker",
    "make_eval_subset": "sgd_tucker",
    "resolve_micro_batch": "sgd_tucker",
    "sampled_loss": "sgd_tucker",
    "zero_entry_term": "sgd_tucker",
    "sgd_non_negative_tucker": "sgd_tucker",
    "SGDTrainer": "sgd_trainer",
    "ShardedSGDTrainer": "sharded_sgd",
    "make_collective": "collectives",
}

__all__ = list(_SUBMODULE_BY_NAME)


def __getattr__(name):
    submodule = _SUBMODULE_BY_NAME.get(name)
    if submodule is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    import importlib
    mod = importlib.import_module(f"tensormet.experimental.SGD.{submodule}")
    return getattr(mod, name)
