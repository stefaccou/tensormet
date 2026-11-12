"""Torch minibatch SGD solver for Tucker, TT and utt (see README.md).

Lazy, like the root tensormet package: importing this package must not pull in
torch until a symbol is used (submit.py runs on login nodes without GPU stacks).
"""

_SUBMODULE_BY_NAME = {
    "SGDModelBase": "sgd_model",
    "SGDTuckerModel": "sgd_tucker",
    "EntryBatcher": "sgd_tucker",
    "GradStepper": "sgd_tucker",
    "full_relative_error": "sgd_tucker",
    "make_eval_subset": "sgd_tucker",
    "resolve_micro_batch": "sgd_tucker",
    "sampled_loss": "sgd_tucker",
    "zero_entry_term": "sgd_tucker",
    "sgd_non_negative_tucker": "sgd_tucker",
    "SGDTTModel": "sgd_tt",
    "SGDTrainer": "sgd_trainer",
    "make_sgd_model": "sgd_trainer",
    "ShardedSGDTrainer": "sharded_sgd",
    "make_collective": "collectives",
    "mu_init_for_sgd": "sgd_init",
}

__all__ = list(_SUBMODULE_BY_NAME)


def __getattr__(name):
    submodule = _SUBMODULE_BY_NAME.get(name)
    if submodule is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    import importlib
    mod = importlib.import_module(f"tensormet.sgd.{submodule}")
    return getattr(mod, name)
