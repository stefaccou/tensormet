"""
standalone.py — ``sgd_non_negative_tucker``: a step-by-step training loop over
``SGDTrainer`` with early stopping, for notebooks. Production runs drive the
trainer through ``solvers.SGDSolver`` instead.
"""
from __future__ import annotations

import time
from typing import List, Optional, Sequence, Tuple, Union

import torch
from tensorly import backend_context
from tensorly.tucker_tensor import TuckerTensor

from tensormet.sgd.sgd_trainer import SGDTrainer
from tensormet.utils import SparseCOOTensor


def sgd_non_negative_tucker(
    sparse_tensor,
    rank: Union[int, Sequence[int]],
    divergence: str = "kl",
    objective: str = "full",
    n_steps: int = 20_000,
    batch_size: int = 4096,
    lr: float = 1e-2,
    parametrization: str = "softplus",
    shared_factors=None,
    init: Union[str, Tuple] = "random",
    random_state: int = 0,
    device: Optional[Union[str, torch.device]] = None,
    dtype: torch.dtype = torch.float32,
    eval_every: int = 500,
    eval_chunk: Optional[int] = None,
    eval_sample: Optional[int] = None,
    tol: float = 1e-5,
    patience: int = 5,
    warmup_steps: int = 0,
    deterministic: bool = False,
    verbose: bool = True,
    return_model: bool = False,
):
    """Fit a non-negative Tucker model by minibatch SGD/Adam: a step-by-step
    loop over ``SGDTrainer`` with early stopping.

    Parameters mirror the MU loop where they overlap (divergence, objective,
    tol/patience on the logged error, random_state, shared_factors).
    ``eval_sample`` logs the error on a fixed random subset of that many nnz
    (``make_eval_subset``); ``final_error`` is always exact.

    ``sparse_tensor``: a torch sparse COO tensor, or an object whose
    ``.tensor`` holds one (e.g. a SparseTupleTensor before any CuPy conversion).

    Returns ``{"tensor": TuckerTensor (CPU numpy), "errors", "iterations",
    "final_error", "decomp_seconds"}``; ``return_model=True`` adds the live
    ``"model"`` (which keeps its device memory alive).
    """
    # torch.use_deterministic_algorithms is process-global; restore it on exit.
    det_prev = torch.are_deterministic_algorithms_enabled()
    if deterministic:
        torch.use_deterministic_algorithms(True)
    try:
        t = getattr(sparse_tensor, "tensor", sparse_tensor)
        if not (isinstance(t, (torch.Tensor, SparseCOOTensor)) and t.is_sparse):
            raise TypeError(
                "sgd_non_negative_tucker expects a torch sparse COO tensor "
                "(or an object with .tensor holding one)."
            )
        # One optimizer step per block, so the error can be logged at any step.
        trainer = SGDTrainer(
            t, rank=rank, divergence=divergence, objective=objective, lr=lr,
            batch_size=batch_size, parametrization=parametrization, shared_factors=shared_factors,
            init=init, random_state=random_state, steps_per_iteration=1,
            devices=None if device is None else [device], dtype=dtype, eval_chunk=eval_chunk,
            eval_sample=eval_sample,
        )

        errors: List[float] = []
        last_err: Optional[float] = None
        no_improve = 0
        start = time.time()
        step = -1

        for step in range(n_steps):
            log_step = eval_every > 0 and (step + 1) % eval_every == 0
            rel = trainer.run_block(step, log_step)
            if not log_step:
                continue
            errors.append(rel)
            if verbose:
                delta = f" (Δ={last_err - rel:+.3e})" if last_err is not None else ""
                print(f"step {step + 1}: relative error={rel:.6f}{delta}")
            # Same patience contract as the MU loop's reconstruction check.
            if last_err is not None and step >= warmup_steps:
                if abs(last_err - rel) < tol:
                    no_improve += 1
                    if no_improve >= patience:
                        if verbose:
                            print(
                                f"Stopped after {no_improve} non-improving checks "
                                f"(patience={patience}) at step {step + 1}."
                            )
                        break
                else:
                    no_improve = 0
            last_err = rel

        if errors and trainer.eval_sample is None:
            final_error = errors[-1]
        elif step >= 0:
            # No eval fired, or the logged errors are subsampled: exact, once.
            final_error = trainer.final_relative_error()
        else:
            final_error = None
        decomp_seconds = time.time() - start
        core, factors = trainer.materialize()
        # TuckerTensor validates through the active tensorly backend (often
        # "pytorch" here, which rejects numpy), so build it under numpy.
        with backend_context("numpy"):
            tucker = TuckerTensor((core, factors))
        out = {
            "tensor": tucker,
            "errors": errors,
            "iterations": step + 1,
            "final_error": final_error,
            "decomp_seconds": decomp_seconds,
        }
        if return_model:
            out["model"] = trainer.model
        return out
    finally:
        torch.use_deterministic_algorithms(det_prev)
