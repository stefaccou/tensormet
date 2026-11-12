"""Torch minibatch SGD solver for Tucker, TT and utt (see README.md).

Lazy, like the root tensormet package: importing this package must not pull in
torch until a symbol is used (submit.py runs on login nodes without GPU stacks).
"""

_SUBMODULE_BY_NAME = {
    "SGDModelBase": "sgd_model",
    "SGDTuckerModel": "sgd_tucker",
    "SGDTTModel": "sgd_tt",
    "EntryBatcher": "sgd_step",
    "GradStepper": "sgd_step",
    "full_relative_error": "sgd_step",
    "make_eval_subset": "sgd_step",
    "resolve_micro_batch": "sgd_step",
    "sampled_loss": "sgd_step",
    "zero_entry_term": "sgd_step",
    "SGDTrainer": "sgd_trainer",
    "make_sgd_model": "sgd_trainer",
    "sgd_non_negative_tucker": "standalone",
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
