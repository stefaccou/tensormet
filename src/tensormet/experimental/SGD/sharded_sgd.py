"""
sharded_sgd.py — EXPERIMENTAL: single-process multi-GPU SGD Tucker trainer.

Data-parallel counterpart of ``sgd_trainer.SGDTrainer``, selected by the loop
when ``cfg.exp.solver == "sgd"`` and ``cfg.train.n_gpus > 1``. Philosophy
matches ``sharded_sparse.ShardedSparseTensor``: ONE process, per-device
shards, host-side reduction — no torch.distributed / DDP / spawn, so the tee
logger, SIGINT handler, judge, and checkpoint writer stay unambiguous.

What multi-GPU buys here (be honest when budgeting runs): the model is tiny
(factors + core), so the win is (1) sharding the NNZ payload — indices,
values, and the 8·nnz-byte batcher permutation — across device memories, and
(2) a larger effective batch per step. Per-step compute is small batched
einsums; expect sublinear step-time speedup.

Layout
------
- The coalesced NNZ is split contiguously into ``len(device_ids)`` shards;
  shard *g*'s indices/values live on device *g* with its own ``EntryBatcher``
  seeded ``random_state * 1000 + g`` (same convention as sharded_sparse), so
  every shard's batch remains a pure function of (seed, step) and resume
  replays exactly.
- The master ``SGDTuckerModel`` + optimizer live on device 0; devices 1..G-1
  hold full model replicas whose raw parameters are overwritten from the
  master after every optimizer step.

Per step
--------
1. every device computes the *sampled* loss term on its sub-batch
   (``batch_size // G`` entries, scale = nnz_g / batch_g so the shard sum is
   unbiased for the total); the EXACT zero-entry term (total_sum /
   total_sq_norm) is added on the master only — it depends only on the
   parameters, so adding it once keeps the summed gradient correct;
2. backward per device (kernels launch async per device);
3. sync 1: replica grads are summed onto the master's grads (tiny tensors);
4. ``opt.step()`` + ``project_()`` on the master;
5. sync 2: updated raw params are broadcast back to the replicas.

Checkpoints carry only the master's state — identical payload to the
single-GPU trainer, so runs resume across different ``n_gpus`` values (the
trajectory is not bitwise identical across G, same caveat as MU sharding).
"""
from __future__ import annotations

from typing import List, Optional, Sequence, Tuple, Union

import numpy as np
import torch

from tensormet.experimental.SGD.sgd_tucker import (
    _EPS,
    EntryBatcher,
    SGDTuckerModel,
    _default_eval_chunk,
)


def _sampled_loss(model, x, x_hat, scale, divergence, masked, eps=_EPS):
    """The shard-local (sampled) part of ``sgd_tucker._batch_loss`` — the
    exact zero-entry term is intentionally absent; the master adds it once."""
    if divergence == "kl":
        x_safe = x.clamp_min(eps)
        nz = x_safe * torch.log(x_safe / (x_hat + eps)) - x
        if masked:
            return scale * (nz + x_hat).sum()
        return scale * nz.sum()
    sq = (x - x_hat) ** 2
    if masked:
        return scale * sq.sum()
    return scale * (sq - x_hat ** 2).sum()


class ShardedSGDTrainer:
    """Multi-GPU minibatch SGD/Adam trainer; same public surface as
    ``SGDTrainer`` (run_block / materialize / checkpoint_payload)."""

    def __init__(
        self,
        sparse_coo: torch.Tensor,
        *,
        device_ids: Sequence[int],
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
        dtype: torch.dtype = torch.float32,
        eval_chunk: Optional[int] = None,
        resume_payload: Optional[dict] = None,
    ):
        if objective not in ("full", "masked"):
            raise ValueError(f"objective must be 'full' or 'masked', got {objective!r}")
        if divergence not in ("kl", "fr"):
            raise ValueError(f"divergence must be 'kl' or 'fr', got {divergence!r}")
        self.divergence = divergence
        self.masked = objective == "masked"
        self.steps_per_iteration = int(steps_per_iteration)
        if self.steps_per_iteration < 1:
            raise ValueError("steps_per_iteration must be >= 1")

        self.devices = [torch.device(f"cuda:{d}") for d in device_ids]
        n_shards = len(self.devices)
        if n_shards < 2:
            raise ValueError("ShardedSGDTrainer needs >= 2 devices; use SGDTrainer.")

        if not (isinstance(sparse_coo, torch.Tensor) and sparse_coo.is_sparse):
            raise TypeError("ShardedSGDTrainer expects a torch sparse COO tensor.")
        t = sparse_coo.coalesce()
        indices_all = t.indices()
        values_all = t.values().to(dtype=dtype)
        if divergence == "kl" and bool((values_all < 0).any()):
            raise ValueError("KL divergence requires non-negative tensor values.")
        self.nnz = int(values_all.shape[0])
        if self.nnz == 0:
            raise ValueError("sparse tensor has no nonzero entries.")

        # Master model + optimizer on device 0; full replicas elsewhere. The
        # replicas are constructed with the same init and then overwritten from
        # the master so all devices start (and stay) identical.
        self.master = SGDTuckerModel(
            shape=t.shape, rank=rank, parametrization=parametrization,
            shared_factors=shared_factors, init=init, random_state=random_state,
            device=self.devices[0], dtype=dtype, eps=epsilon,
        )
        self.replicas = [
            SGDTuckerModel(
                shape=t.shape, rank=rank, parametrization=parametrization,
                shared_factors=shared_factors, init=init, random_state=random_state,
                device=dev, dtype=dtype, eps=epsilon,
            )
            for dev in self.devices[1:]
        ]
        self._models = [self.master] + self.replicas

        # Contiguous NNZ shards, one per device, with per-shard batchers seeded
        # like sharded_sparse (base * 1000 + shard) and a per-shard slice of
        # the requested batch. No device holds the full permutation.
        bounds = np.linspace(0, self.nnz, n_shards + 1, dtype=np.int64)
        shard_batch = max(1, int(batch_size) // n_shards)
        self.shard_indices: List[torch.Tensor] = []
        self.shard_values: List[torch.Tensor] = []
        self.batchers: List[EntryBatcher] = []
        self.scales: List[float] = []
        for g, dev in enumerate(self.devices):
            lo, hi = int(bounds[g]), int(bounds[g + 1])
            self.shard_indices.append(indices_all[:, lo:hi].to(dev))
            self.shard_values.append(values_all[lo:hi].to(dev))
            nnz_g = hi - lo
            batcher = EntryBatcher(nnz_g, shard_batch,
                                   seed=int(random_state) * 1000 + g, device=dev)
            self.batchers.append(batcher)
            self.scales.append(nnz_g / batcher.batch_size)

        # Global normalization constant (identical to the single-GPU trainer).
        if divergence == "kl":
            self.norm_const = float(values_all.sum().clamp_min(_EPS))
        else:
            self.norm_const = float(values_all.pow(2).sum().clamp_min(_EPS))

        self.eval_chunk = eval_chunk or _default_eval_chunk(self.master.rank)

        if optimizer == "adam":
            self.opt = torch.optim.Adam(self.master.parameters(), lr=lr)
        elif optimizer == "sgd":
            self.opt = torch.optim.SGD(self.master.parameters(), lr=lr, momentum=0.9)
        else:
            raise ValueError(f"optimizer must be 'adam' or 'sgd', got {optimizer!r}")

        if resume_payload is not None:
            self.load_payload(resume_payload)
        else:
            self._broadcast_params()

    # --- parameter plumbing ---------------------------------------------------

    def _broadcast_params(self) -> None:
        """sync 2: master raw params -> replicas (same state-dict structure)."""
        with torch.no_grad():
            master_state = self.master.state_dict()
            for rep in self.replicas:
                for name, p in rep.state_dict().items():
                    p.copy_(master_state[name], non_blocking=True)

    def _reduce_grads(self) -> None:
        """sync 1: sum replica grads onto the master's (tiny tensors)."""
        master_params = dict(self.master.named_parameters())
        for rep in self.replicas:
            for name, p in rep.named_parameters():
                if p.grad is not None:
                    g = p.grad.to(self.devices[0], non_blocking=True)
                    mp = master_params[name]
                    mp.grad = g if mp.grad is None else mp.grad + g

    # --- one loop iteration ---------------------------------------------------

    def run_block(self, iteration: int, log_step: bool) -> Optional[float]:
        k = self.steps_per_iteration
        for step in range(iteration * k, (iteration + 1) * k):
            losses = []
            for g, (model, dev) in enumerate(zip(self._models, self.devices)):
                sel = self.batchers[g].batch(step)
                idx = self.shard_indices[g][:, sel]
                x = self.shard_values[g][sel]
                x_hat = model.predict_entries(idx)
                loss = _sampled_loss(model, x, x_hat, self.scales[g],
                                     self.divergence, self.masked)
                if g == 0 and not self.masked:
                    # exact zero-entry term, master only (see module docstring)
                    if self.divergence == "kl":
                        loss = loss + model.total_sum()
                    else:
                        loss = loss + model.total_sq_norm()
                losses.append(loss / self.norm_const)

            for model in self._models:
                model.zero_grad(set_to_none=True)
            for loss in losses:
                loss.backward()
            self._reduce_grads()
            self.opt.step()
            self.master.project_()
            self._broadcast_params()

        if not log_step:
            return None
        return self._full_relative_error()

    # --- exact error (sharded) ------------------------------------------------

    @torch.no_grad()
    def _full_relative_error(self) -> float:
        """Per-shard chunked accumulation of the nnz term, host-summed; the
        analytic zero-entry term is added once. Same normalization as
        ``sgd_tucker.full_relative_error`` / the MU error kernels."""
        eps = _EPS
        acc = 0.0
        vals_sum = 0.0
        vals_sq_sum = 0.0
        for g, (model, dev) in enumerate(zip(self._models, self.devices)):
            indices = self.shard_indices[g]
            values = self.shard_values[g]
            shard_acc = values.new_zeros(())
            for s in range(0, values.shape[0], self.eval_chunk):
                idx = indices[:, s:s + self.eval_chunk]
                x = values[s:s + self.eval_chunk]
                x_hat = model.predict_entries(idx)
                if self.divergence == "kl":
                    x_safe = x.clamp_min(eps)
                    term = x_safe * torch.log(x_safe / (x_hat + eps)) - x
                    if self.masked:
                        term = term + x_hat
                else:
                    term = (x - x_hat) ** 2
                    if not self.masked:
                        term = term - x_hat ** 2
                shard_acc = shard_acc + term.sum()
            acc += float(shard_acc)
            vals_sum += float(values.sum())
            vals_sq_sum += float(values.pow(2).sum())

        if self.divergence == "kl":
            total = acc if self.masked else acc + float(self.master.total_sum())
            return float(total / max(vals_sum, eps))
        total = acc if self.masked else acc + float(self.master.total_sq_norm())
        return float(max(total, 0.0) ** 0.5 / max(vals_sq_sum ** 0.5, eps))

    # --- materialization / checkpointing --------------------------------------

    def materialize(self) -> Tuple[np.ndarray, List[np.ndarray]]:
        return self.master.materialize()

    def checkpoint_payload(self, iteration: int) -> dict:
        """Master-only payload, byte-compatible with ``SGDTrainer``'s — resume
        works across different n_gpus values."""
        core, factors = self.materialize()
        raw_state = {k: v.detach().cpu() for k, v in self.master.state_dict().items()}
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
        self.master.load_state_dict(payload["raw_state_dict"])
        self.opt.load_state_dict(payload["optim_state"])
        self._broadcast_params()
