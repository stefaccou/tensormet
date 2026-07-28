"""
sgd_trainer.py — EXPERIMENTAL: stateful SGD trainer for the production loop.

``SGDTrainer`` packages ``sgd_tucker``'s model/batcher/objective into the
narrow surface ``non_negative_tucker_with_similarity`` consumes when
``cfg.exp.solver == "sgd"``:

    trainer = SGDTrainer(torch_coo, rank=..., divergence=..., ...)
    rel_err = trainer.run_block(iteration, log_step)   # K optimizer steps
    core, factors = trainer.materialize()              # CPU numpy, lazily
    payload = trainer.checkpoint_payload(iteration)    # resumable dict

One loop "iteration" is a block of ``steps_per_iteration`` optimizer steps;
block *i* runs global steps ``i*K .. (i+1)*K-1``. Batches are a pure function
of (seed, step) (see ``EntryBatcher``), so a resumed run — which restores the
raw parameters and optimizer state from ``checkpoint_payload`` and continues
at ``start_iteration`` — replays exactly the batches an uninterrupted run
would have seen.

Unlike the MU kernels this trainer carries state the UpdateRouting seam cannot
express (Adam moments, raw pre-softplus parameters, the step counter), which
is why the loop branches on the solver instead of routing kernels.
"""
from __future__ import annotations

from typing import List, Optional, Sequence, Tuple, Union

import numpy as np
import torch

from tensormet.experimental.SGD.sgd_tucker import (
    _EPS,
    EntryBatcher,
    SGDTuckerModel,
    _batch_loss,
    full_relative_error,
)


class SGDTrainer:
    """Minibatch SGD/Adam trainer over a torch sparse COO tensor."""

    def __init__(
        self,
        sparse_coo: torch.Tensor,
        *,
        rank: Union[int, Sequence[int]],
        divergence: str = "kl",
        objective: str = "full",
        lr: float = 1e-2,
        batch_size: int = 4096,
        optimizer: str = "adam",
        parametrization: str = "softplus",
        shared_factors=None,
        init: Union[str, Tuple] = "random",
        random_state: int = 0,
        steps_per_iteration: int = 100,
        epsilon: float = _EPS,
        device: Optional[Union[str, torch.device]] = None,
        dtype: torch.dtype = torch.float32,
        eval_chunk: Optional[int] = None,
        resume_payload: Optional[dict] = None,
    ):
        if objective not in ("full", "masked"):
            raise ValueError(f"objective must be 'full' or 'masked', got {objective!r}")
        self.divergence = divergence
        self.masked = objective == "masked"
        self.steps_per_iteration = int(steps_per_iteration)
        if self.steps_per_iteration < 1:
            raise ValueError("steps_per_iteration must be >= 1")
        self.eval_chunk = eval_chunk

        if not (isinstance(sparse_coo, torch.Tensor) and sparse_coo.is_sparse):
            raise TypeError("SGDTrainer expects a torch sparse COO tensor.")
        t = sparse_coo.coalesce()

        self.model = SGDTuckerModel(
            shape=t.shape, rank=rank, parametrization=parametrization,
            shared_factors=shared_factors, init=init, random_state=random_state,
            device=device, dtype=dtype, eps=epsilon,
        )
        self.device = self.model._core_raw.device

        self.indices = t.indices().to(self.device)
        self.values = t.values().to(self.device, dtype=dtype)
        if divergence == "kl" and bool((self.values < 0).any()):
            raise ValueError("KL divergence requires non-negative tensor values.")
        self.nnz = int(self.values.shape[0])
        if self.nnz == 0:
            raise ValueError("sparse tensor has no nonzero entries.")

        self.batcher = EntryBatcher(self.nnz, batch_size, seed=random_state,
                                    device=self.device)
        self.scale = self.nnz / self.batcher.batch_size

        # Loss normalized by the data scale (same constants the relative errors
        # use) so lr defaults transfer across datasets.
        if divergence == "kl":
            self.norm_const = float(self.values.sum().clamp_min(_EPS))
        elif divergence == "fr":
            self.norm_const = float(self.values.pow(2).sum().clamp_min(_EPS))
        else:
            raise ValueError(f"divergence must be 'kl' or 'fr', got {divergence!r}")

        if optimizer == "adam":
            self.opt = torch.optim.Adam(self.model.parameters(), lr=lr)
        elif optimizer == "sgd":
            self.opt = torch.optim.SGD(self.model.parameters(), lr=lr, momentum=0.9)
        else:
            raise ValueError(f"optimizer must be 'adam' or 'sgd', got {optimizer!r}")

        if resume_payload is not None:
            self.load_payload(resume_payload)

    # --- one loop iteration -------------------------------------------------

    def run_block(self, iteration: int, log_step: bool) -> Optional[float]:
        """Run optimizer steps ``iteration*K .. (iteration+1)*K - 1``; return
        the exact relative error over all nnz on log steps, else None."""
        k = self.steps_per_iteration
        for step in range(iteration * k, (iteration + 1) * k):
            sel = self.batcher.batch(step)
            idx = self.indices[:, sel]
            x = self.values[sel]

            self.opt.zero_grad(set_to_none=True)
            x_hat = self.model.predict_entries(idx)
            loss = _batch_loss(self.model, x, x_hat, self.scale,
                               self.divergence, self.masked) / self.norm_const
            loss.backward()
            self.opt.step()
            self.model.project_()

        if not log_step:
            return None
        return full_relative_error(
            self.model, self.indices, self.values, self.divergence,
            masked=self.masked, chunk=self.eval_chunk,
        )

    # --- materialization / checkpointing ------------------------------------

    def materialize(self) -> Tuple[np.ndarray, List[np.ndarray]]:
        """Detached CPU numpy (core, factors) — TuckerDecomposition-ready."""
        return self.model.materialize()

    def checkpoint_payload(self, iteration: int) -> dict:
        """Fully resumable checkpoint. ``core``/``factors`` are the
        non-negative views (host numpy) so CPU-only tooling can peek without
        knowing about parametrizations; ``raw_state_dict``/``optim_state``
        carry the actual trainable state (host tensors)."""
        core, factors = self.materialize()
        raw_state = {k: v.detach().cpu() for k, v in self.model.state_dict().items()}
        optim_state = self.opt.state_dict()
        optim_state["state"] = {
            k: {kk: (vv.detach().cpu() if isinstance(vv, torch.Tensor) else vv)
                for kk, vv in v.items()}
            for k, v in optim_state.get("state", {}).items()
        }
        return {
            "solver": "sgd",
            "iteration": int(iteration),
            "core": core,
            "factors": factors,
            "raw_state_dict": raw_state,
            "optim_state": optim_state,
        }

    def load_payload(self, payload: dict) -> None:
        if payload.get("solver") != "sgd":
            raise ValueError(
                "checkpoint payload is not an SGD checkpoint; refusing to load "
                f"(got solver={payload.get('solver')!r})."
            )
        self.model.load_state_dict(payload["raw_state_dict"])
        # load_state_dict moves optimizer state to each param's device.
        self.opt.load_state_dict(payload["optim_state"])
