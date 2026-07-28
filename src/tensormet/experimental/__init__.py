"""Deliberately lazy: importing tensormet.experimental (or a specific submodule like
tensormet.experimental.submit) must not drag in torch/cupy via extended_tucker, since
submit.py is meant to run from a login node before those are needed.
"""

__all__ = ["ExtendedTucker", "sgd_non_negative_tucker", "SGDTuckerModel",
           "SGDTrainer", "ShardedSGDTrainer"]


def __getattr__(name):
    if name == "ExtendedTucker":
        from tensormet.experimental.extended_tucker import ExtendedTucker
        return ExtendedTucker
    if name in ("sgd_non_negative_tucker", "SGDTuckerModel", "SGDTrainer",
                "ShardedSGDTrainer"):
        from tensormet.experimental import SGD
        return getattr(SGD, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
