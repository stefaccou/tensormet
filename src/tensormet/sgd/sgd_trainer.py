"""
sgd_trainer.py — the SGD trainer the production loop drives
(``cfg.exp.solver == "sgd"``), plus the helpers it shares with
``ShardedSGDTrainer``:

    trainer = SGDTrainer(torch_coo, rank=..., divergence=..., ...)
    rel_err = trainer.run_block(iteration, log_step)   # K optimizer steps
    core, factors = trainer.materialize()              # CPU numpy
    payload = trainer.checkpoint_payload(iteration)    # resumable dict

One loop "iteration" is a block of ``steps_per_iteration`` optimizer steps;
batches are a pure function of (seed, step), so resume replays them exactly.
The step body is ``sgd_tucker.GradStepper``; parameter ``.grad``s are views
into its flat buffer, so never call ``zero_grad(set_to_none=True)``.
"""
from __future__ import annotations

from typing import List, Optional, Sequence, Tuple, Union

import numpy as np
import torch

from tensormet.sgd.sgd_tucker import (
    _EPS,
    EntryBatcher,
    GradStepper,
    SGDTuckerModel,
    full_relative_error,
    make_eval_subset,
)
from tensormet.utils import SparseCOOTensor


def make_sgd_model(decomposition: str = "tucker", *, shape, rank, tt_rank=None,
                   parametrization="softplus", shared_factors=None, init="random",
                   random_state=0, device=None, dtype=torch.float32, eps=_EPS,
                   sum_x=None):
    """The SGD model for ``decomposition``. All three share one surface, so the
    trainers never branch on the model type. ``sum_x`` (Σ of the data) sets the
    TT/utt random-init scale; Tucker ignores it, like ``tt_rank``."""
    common = dict(shape=shape, rank=rank, parametrization=parametrization,
                  shared_factors=shared_factors, init=init, random_state=random_state,
                  device=device, dtype=dtype, eps=eps)
    if decomposition == "tucker":
        return SGDTuckerModel(**common)
    if decomposition not in ("tt", "utt"):
        raise ValueError(f"solver='sgd' supports decomposition 'tucker', 'tt' or 'utt'; "
                         f"got {decomposition!r}")
    if tt_rank is None:
        raise ValueError(f"decomposition={decomposition!r} needs tt_rank.")
    if decomposition == "tt":
        from tensormet.sgd.sgd_tt import SGDTTModel
        return SGDTTModel(tt_rank=tt_rank, sum_x=sum_x, **common)
    from tensormet.uniform_tt.utt_sgd import SGDUTTModel
    return SGDUTTModel(tt_rank=tt_rank, sum_x=sum_x, **common)


# ---------------------------------------------------------------------------
# helpers shared with ShardedSGDTrainer
# ---------------------------------------------------------------------------

def coalesced_input(sparse_coo, who: str):
    """The coalesced input tensor. SparseCOOTensor is load_from_disk's stand-in
    where prod(shape) overflows torch's numel; it has every method used here
    (.coalesce, .indices, .values, .shape)."""
    if not (isinstance(sparse_coo, (torch.Tensor, SparseCOOTensor))
            and sparse_coo.is_sparse):
        raise TypeError(f"{who} expects a torch sparse COO tensor.")
    return sparse_coo.coalesce()


def check_common_args(objective: str, steps_per_iteration: int) -> int:
    if objective not in ("full", "masked"):
        raise ValueError(f"objective must be 'full' or 'masked', got {objective!r}")
    steps = int(steps_per_iteration)
    if steps < 1:
        raise ValueError("steps_per_iteration must be >= 1")
    return steps


def loss_norm_const(divergence: str, sum_x: float, sum_x_sq: float) -> float:
    """The loss is divided by the relative error's own denominator, so lr
    defaults transfer across datasets."""
    if divergence == "kl":
        return max(sum_x, _EPS)
    if divergence == "fr":
        return max(sum_x_sq, _EPS)
    raise ValueError(f"divergence must be 'kl' or 'fr', got {divergence!r}")


def checkpoint_payload(model, opt, iteration: int) -> dict:
    """Resumable checkpoint. ``core``/``factors`` are the non-negative views
    (host numpy) for CPU-only tooling; ``raw_state_dict``/``optim_state`` hold
    the trainable state (host tensors)."""
    core, factors = model.materialize()
    raw_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    optim_state = opt.state_dict()
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


def check_sgd_payload(payload: dict) -> None:
    if payload.get("solver") != "sgd":
        raise ValueError(
            "checkpoint payload is not an SGD checkpoint; refusing to load "
            f"(got solver={payload.get('solver')!r})."
        )


# ---------------------------------------------------------------------------
# single-GPU trainer
# ---------------------------------------------------------------------------

class SGDTrainer:
    """Minibatch SGD/Adam trainer over a torch sparse COO tensor."""

    def __init__(
        self,
        sparse_coo: Union[torch.Tensor, SparseCOOTensor],
        *,
        rank: Union[int, Sequence[int]],
        divergence: str = "kl",
        objective: str = "full",
        lr: float = 1e-2,
        batch_size: int = 4096,
        parametrization: str = "softplus",
        shared_factors=None,
        init: Union[str, Tuple] = "random",
        random_state: int = 0,
        steps_per_iteration: int = 100,
        epsilon: float = _EPS,
        device: Optional[Union[str, torch.device]] = None,
        dtype: torch.dtype = torch.float32,
        eval_chunk: Optional[int] = None,
        eval_sample: Optional[int] = None,
        micro_batch: Optional[int] = None,
        resume_payload: Optional[dict] = None,
        decomposition: str = "tucker",
        tt_rank: Optional[int] = None,
        regauge_every: int = 1,
    ):
        self.steps_per_iteration = check_common_args(objective, steps_per_iteration)
        self.divergence = divergence
        self.masked = objective == "masked"
        self.eval_chunk = eval_chunk
        # Blocks between gauge resets (0 = never); a no-op for Tucker.
        self.regauge_every = max(0, int(regauge_every))
        t = coalesced_input(sparse_coo, "SGDTrainer")

        self.model = make_sgd_model(
            decomposition, shape=t.shape, rank=rank, tt_rank=tt_rank,
            parametrization=parametrization, shared_factors=shared_factors,
            init=init, random_state=random_state, device=device, dtype=dtype,
            eps=epsilon, sum_x=float(t.values().sum()),
        )
        self.device = self.model.device

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

        # The logged error costs a training step per entry; eval_sample swaps
        # the exact pass for a fixed random subset (None keeps it exact).
        (self._eval_indices, self._eval_values,
         self._eval_scale, self._eval_totals) = make_eval_subset(
            self.indices, self.values, eval_sample, random_state
        )
        self.eval_sample = (None if self._eval_scale == 1.0
                            else int(self._eval_values.shape[0]))

        self.norm_const = loss_norm_const(divergence, *self._eval_totals)
        self.opt = torch.optim.Adam(self.model.parameters(), lr=lr)

        self.stepper = GradStepper(
            self.model, self.indices, self.values, self.batcher,
            scale=self.scale, divergence=self.divergence, masked=self.masked,
            norm_const=self.norm_const, include_zero_term=True,
            micro_batch=micro_batch, eps=epsilon,
        )
        self.micro_batch = self.stepper.micro_batch

        if resume_payload is not None:
            self.load_payload(resume_payload)

    # --- one loop iteration -------------------------------------------------

    def run_block(self, iteration: int, log_step: bool) -> Optional[float]:
        """Run optimizer steps ``iteration*K .. (iteration+1)*K - 1``; return
        the relative error on log steps, else None."""
        k = self.steps_per_iteration
        for step in range(iteration * k, (iteration + 1) * k):
            self.stepper.compute_grads(step)
            self.opt.step()
        if self.regauge_every and (iteration + 1) % self.regauge_every == 0:
            self.model.regauge_()

        if not log_step:
            return None
        return self._full_relative_error()

    def sync(self) -> None:
        """Block until the device is idle. ``run_block`` returns with work still
        queued on non-log steps, so timings must bracket it with this."""
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

    def final_relative_error(self) -> float:
        """The exact error over all nnz, whatever ``eval_sample`` is."""
        return self._full_relative_error(exact=True)

    def _full_relative_error(self, exact: bool = False) -> float:
        """The logged error: on the eval subset when ``eval_sample`` is set,
        unless ``exact``."""
        if exact or self._eval_scale == 1.0:
            return full_relative_error(
                self.model, self.indices, self.values, self.divergence,
                masked=self.masked, chunk=self.eval_chunk,
                totals=self._eval_totals,
            )
        return full_relative_error(
            self.model, self._eval_indices, self._eval_values, self.divergence,
            masked=self.masked, chunk=self.eval_chunk,
            sample_scale=self._eval_scale, totals=self._eval_totals,
        )

    # --- materialization / checkpointing ------------------------------------

    def materialize(self) -> Tuple[np.ndarray, List[np.ndarray]]:
        """Detached CPU numpy (core, factors); for TT/utt ``core`` is a list."""
        return self.model.materialize()

    def checkpoint_payload(self, iteration: int) -> dict:
        return checkpoint_payload(self.model, self.opt, iteration)

    def load_payload(self, payload: dict) -> None:
        check_sgd_payload(payload)
        self.model.load_state_dict(payload["raw_state_dict"])
        # load_state_dict moves optimizer state to each param's device.
        self.opt.load_state_dict(payload["optim_state"])
