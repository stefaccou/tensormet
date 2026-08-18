"""Deliberately lazy: importing tensormet.experimental (or a specific submodule like
tensormet.experimental.submit) must not drag in torch/cupy via extended_tucker, since
submit.py is meant to run from a login node before those are needed.
"""

__all__ = ["ExtendedTucker"]


def __getattr__(name):
    if name == "ExtendedTucker":
        from tensormet.experimental.extended_tucker import ExtendedTucker
        return ExtendedTucker
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
