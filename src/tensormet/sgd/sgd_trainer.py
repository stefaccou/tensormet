"""
sgd_trainer.py — the SGD trainer the production loop drives
(``cfg.exp.solver == "sgd"``), for one device or several:

    trainer = SGDTrainer(torch_coo, devices=[...], rank=..., divergence=..., ...)
    rel_err = trainer.run_block(iteration, log_step)   # K optimizer steps
    core, factors = trainer.materialize()              # CPU numpy
    payload = trainer.checkpoint_payload(iteration)    # resumable dict

One loop "iteration" is a block of ``steps_per_iteration`` optimizer steps;
batches are a pure function of (seed, step), so resume replays them exactly.
The step body is ``sgd_step.GradStepper``; parameter ``.grad``s are views into
its flat buffer, so never call ``zero_grad(set_to_none=True)``.

Single process, no torch.distributed, so the logger, SIGINT handler, judge and
checkpoint writer stay unambiguous. Each device holds a contiguous NNZ shard
with its own ``EntryBatcher``, a full model replica, an optimizer and a
``GradStepper``; device 0's is ``self.master`` / ``self.opt`` and is what gets
checkpointed. Per step each device backprops its sampled loss (the exact
zero-entry term only on device 0), one all-reduce sums the gradients, and every
device steps on the same sum, so replicas stay bit-identical. Every device
samples ``batch_size`` entries per step. One device is the same code without the
collective, thread pool and replica check. See sgd/README.md.
"""
from __future__ import annotations

import copy
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Callable, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch

from tensormet.sgd.collectives import make_collective
from tensormet.sgd.sgd_model import _EPS
from tensormet.sgd.sgd_step import (
    EntryBatcher,
    GradStepper,
    _rank_derived_chunk,
    full_relative_error,
    make_eval_subset,
    nnz_term_sum,
)
from tensormet.sgd.sgd_tucker import SGDTuckerModel
from tensormet.utils import SparseCOOTensor


def make_sgd_model(decomposition: str = "tucker", *, shape, rank, tt_rank=None,
                   parametrization="softplus", shared_factors=None, init="random",
                   random_state=0, device=None, dtype=torch.float32, eps=_EPS,
                   sum_x=None):
    """The SGD model for ``decomposition``. All three share one surface, so the
    trainer never branches on the model type. ``sum_x`` (Σ of the data) sets the
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


def _as_devices(devices) -> List[torch.device]:
    """Devices as ``torch.device``s: ints are CUDA indices; None is the default device."""
    if devices is None:
        return [torch.device("cuda" if torch.cuda.is_available() else "cpu")]
    return [torch.device(f"cuda:{d}") if isinstance(d, int) else torch.device(d)
            for d in devices]


class SGDTrainer:
    """Minibatch SGD/Adam trainer over a torch sparse COO tensor, on one or more devices."""

    def __init__(
        self,
        sparse_coo: Union[torch.Tensor, SparseCOOTensor],
        *,
        devices: Optional[Sequence[Union[int, str, torch.device]]] = None,
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
        dtype: torch.dtype = torch.float32,
        eval_chunk: Optional[int] = None,
        eval_sample: Optional[int] = None,
        micro_batch: Optional[int] = None,
        comm_backend: str = "auto",
        resume_payload: Optional[dict] = None,
        decomposition: str = "tucker",
        tt_rank: Optional[int] = None,
        regauge_every: int = 1,
    ):
        self.steps_per_iteration = check_common_args(objective, steps_per_iteration)
        if divergence not in ("kl", "fr"):
            raise ValueError(f"divergence must be 'kl' or 'fr', got {divergence!r}")
        self.divergence = divergence
        self.masked = objective == "masked"
        # Blocks between gauge resets (0 = never); a no-op for Tucker.
        self.regauge_every = max(0, int(regauge_every))

        self.devices = _as_devices(devices)
        n_shards = len(self.devices)
        t = coalesced_input(sparse_coo, "SGDTrainer")
        indices_all = t.indices()
        values_all = t.values().to(dtype=dtype)
        if divergence == "kl" and bool((values_all < 0).any()):
            raise ValueError("KL divergence requires non-negative tensor values.")
        self.nnz = int(values_all.shape[0])
        if self.nnz == 0:
            raise ValueError("sparse tensor has no nonzero entries.")
        host_sum = float(values_all.sum())

        # One full replica + optimizer per device. The optimizer step runs on
        # every device off the same all-reduced gradient, so there is no
        # per-step parameter broadcast.
        self._models = [
            make_sgd_model(
                decomposition, shape=t.shape, rank=rank, tt_rank=tt_rank,
                parametrization=parametrization, shared_factors=shared_factors,
                init=init, random_state=random_state, device=dev, dtype=dtype,
                eps=epsilon, sum_x=host_sum,
            )
            for dev in self.devices
        ]
        self.master = self._models[0]
        self.replicas = self._models[1:]
        if n_shards > 1:
            # Before anything runs a matmul or a backward concurrently.
            self._warm_up_devices(dtype)

        # Contiguous NNZ shards, one per device, with per-shard batchers seeded
        # like sharded_sparse (base * 1000 + shard); one device keeps the seed
        # itself. G devices sample G x batch_size entries per step.
        seeds = [int(random_state) if n_shards == 1 else int(random_state) * 1000 + g
                 for g in range(n_shards)]
        bounds = np.linspace(0, self.nnz, n_shards + 1, dtype=np.int64)
        self.shard_indices: List[torch.Tensor] = []
        self.shard_values: List[torch.Tensor] = []
        self.batchers: List[EntryBatcher] = []
        for g, dev in enumerate(self.devices):
            lo, hi = int(bounds[g]), int(bounds[g + 1])
            self.shard_indices.append(indices_all[:, lo:hi].to(dev))
            self.shard_values.append(values_all[lo:hi].to(dev))
            self.batchers.append(EntryBatcher(hi - lo, int(batch_size), seed=seeds[g],
                                              device=dev))

        # The logged error costs a training step per entry; eval_sample swaps the
        # exact pass for a fixed random subset (None keeps it exact). Per-shard
        # subsets, sized in proportion to the shards so their union is a uniform
        # sample of the whole tensor.
        self.eval_indices: List[torch.Tensor] = []
        self.eval_values: List[torch.Tensor] = []
        self.eval_scales: List[float] = []
        eval_totals = None
        for g in range(n_shards):
            nnz_g = int(self.shard_values[g].shape[0])
            n_g = (None if not eval_sample
                   else max(1, round(int(eval_sample) * nnz_g / self.nnz)))
            e_idx, e_val, e_scale, eval_totals = make_eval_subset(
                self.shard_indices[g], self.shard_values[g], n_g, seeds[g])
            self.eval_indices.append(e_idx)
            self.eval_values.append(e_val)
            self.eval_scales.append(e_scale)
        self.eval_sample = (None if all(s == 1.0 for s in self.eval_scales)
                            else sum(int(v.shape[0]) for v in self.eval_values))
        self.eval_chunk = eval_chunk or _rank_derived_chunk(
            self.master.rank, width=self.master.entry_width)

        # (Σx, Σx²) over the whole tensor: the error denominators. One device sums
        # on the device (the float32 sums differ from the host's by summation
        # order; kept so a one-device run reproduces earlier ones bit for bit).
        self._totals = (eval_totals if n_shards == 1
                        else (host_sum, float(values_all.pow(2).sum())))
        self.norm_const = loss_norm_const(divergence, *self._totals)

        self.scales: List[float] = []
        self.steppers: List[GradStepper] = []
        for g in range(n_shards):
            # Gradients are summed every step: each device carries its own
            # shard's nnz term and exactly one device carries the zero term.
            scale = int(self.shard_values[g].shape[0]) / self.batchers[g].batch_size
            self.scales.append(scale)
            self.steppers.append(GradStepper(
                self._models[g], self.shard_indices[g], self.shard_values[g],
                self.batchers[g], scale=scale, divergence=divergence, masked=self.masked,
                norm_const=self.norm_const, include_zero_term=g == 0,
                micro_batch=micro_batch, eps=epsilon,
            ))
        self.micro_batch = self.steppers[0].micro_batch

        self._opts = [torch.optim.Adam(m.parameters(), lr=lr) for m in self._models]
        self.opt = self._opts[0]

        self.collective = None
        self._pool = None
        if n_shards > 1:
            # Printed, because 'auto' can fall back to HostReduce, which changes
            # what every timing of this run means.
            print(f"[{time.strftime('%H:%M:%S')}] building SGD collective "
                  f"(requested backend={comm_backend!r})...", flush=True)
            self.collective = make_collective(self.devices, backend=comm_backend)
            print(f"[{time.strftime('%H:%M:%S')}] SGD collective: {self.collective!r}",
                  flush=True)
            # Persistent pool: the threads (and their cuBLAS handles) live as long
            # as this object. A pool per step recycles thread IDs in Python 3.13
            # and leaves stale cuBLAS handles (as documented in sharded_sparse).
            self._pool = ThreadPoolExecutor(max_workers=n_shards)

        # (master tensor, replica tensor) pairs for the cold-path broadcast.
        with torch.no_grad():
            master_state = self.master.state_dict()
            self._bcast_pairs: List[List[Tuple[torch.Tensor, torch.Tensor]]] = [
                [(master_state[name], p) for name, p in rep.state_dict().items()]
                for rep in self.replicas
            ]
        if resume_payload is not None:
            self.load_payload(resume_payload)
        else:
            self._broadcast_params()

    # --- the surface a one-device caller expects -------------------------------

    @property
    def model(self):
        return self.master

    @property
    def device(self) -> torch.device:
        return self.devices[0]

    @property
    def stepper(self) -> GradStepper:
        return self.steppers[0]

    # --- device plumbing ------------------------------------------------------

    def _warm_up_devices(self, dtype: torch.dtype) -> None:
        """Load cuBLAS/autograd kernels on every device, one device at a time.
        CUDA loads kernels lazily, and concurrent first loads from the worker
        threads race (seen in MU as CUBLAS_STATUS_NOT_INITIALIZED)."""
        for dev in self.devices:
            with torch.cuda.device(dev):
                a2 = torch.ones((8, 8), device=dev, dtype=dtype)
                a3 = torch.ones((2, 8, 8), device=dev, dtype=dtype)
                _ = a2 @ a2
                _ = a3 @ a3
                w = torch.ones((8, 8), device=dev, dtype=dtype, requires_grad=True)
                (w @ w).sum().backward()
                torch.cuda.synchronize(dev)

    def _fan_out(self, fn: Callable, *args) -> None:
        """Run ``fn(g, *args)`` on every device, concurrently, and wait (one
        device: just call it).

        ``torch.cuda.set_device`` is applied per task: the pool does not
        guarantee shard *g* the same thread twice. Worker exceptions are
        re-raised here.
        """
        if self._pool is None:
            fn(0, *args)
            return

        def _task(g):
            torch.cuda.set_device(self.devices[g])
            return fn(g, *args)

        futures = [self._pool.submit(_task, g) for g in range(len(self.devices))]
        errors = []
        for fut in futures:
            try:
                fut.result()
            except BaseException as exc:  # noqa: BLE001 — re-raised below
                errors.append(exc)
        if errors:
            raise errors[0]

    def sync(self) -> None:
        """Block until every device is idle. ``run_block`` returns with work
        still queued on non-log steps, so timings must bracket it with this."""
        for dev in self.devices:
            if dev.type == "cuda":
                torch.cuda.synchronize(dev)

    def __del__(self) -> None:
        pool = getattr(self, "_pool", None)
        if pool is not None:
            pool.shutdown(wait=False)
            self._pool = None

    # --- parameter plumbing ---------------------------------------------------

    def _broadcast_params(self) -> None:
        """Device 0 raw params -> replicas (construction and resume only)."""
        with torch.no_grad():
            for pairs in self._bcast_pairs:
                for src, dst in pairs:
                    dst.copy_(src, non_blocking=True)

    @torch.no_grad()
    def _check_replica_drift(self) -> None:
        """Raise unless every device holds bit-identical parameters.

        Stepping each replica on the same all-reduced gradient is only valid if
        they stay identical; this makes a violation loud. Exact comparison, at
        the log-step cadence.
        """
        replica_params = [dict(rep.named_parameters()) for rep in self.replicas]
        for name, ref in self.master.named_parameters():
            for g, params in enumerate(replica_params, start=1):
                if not torch.equal(ref, params[name].to(ref.device)):
                    raise RuntimeError(
                        f"SGD replica drift: parameter {name!r} differs between "
                        f"{self.devices[0]} and {self.devices[g]}. "
                        "The replicas must stay bit-identical for the redundant "
                        "optimizer step to be valid — this is a bug in the "
                        "collective or a non-deterministic optimizer kernel, not "
                        "a tuning issue."
                    )

    # --- one loop iteration ---------------------------------------------------

    def _grads_for(self, g: int, step: int) -> None:
        self.steppers[g].compute_grads(step)

    def _apply_step(self, g: int) -> None:
        self._opts[g].step()

    def _regauge(self, g: int) -> None:
        self._models[g].regauge_()

    def run_block(self, iteration: int, log_step: bool) -> Optional[float]:
        """Run optimizer steps ``iteration*K .. (iteration+1)*K - 1``; return the
        relative error on log steps, else None."""
        k = self.steps_per_iteration
        for step in range(iteration * k, (iteration + 1) * k):
            self._fan_out(self._grads_for, step)
            if self.collective is not None:
                self.collective.all_reduce([s.flat_grad for s in self.steppers])
            self._fan_out(self._apply_step)

        # Replicas are identical here and the reset is deterministic, so they
        # stay identical after it.
        if self.regauge_every and (iteration + 1) % self.regauge_every == 0:
            self._fan_out(self._regauge)

        if not log_step:
            return None
        self._check_replica_drift()
        return self._full_relative_error()

    # --- error ----------------------------------------------------------------

    @torch.no_grad()
    def _shard_error_terms(self, g: int, out: list, exact: bool) -> None:
        if exact:
            indices, values, scale = (self.shard_indices[g],
                                      self.shard_values[g], 1.0)
        else:
            indices, values, scale = (self.eval_indices[g],
                                      self.eval_values[g], self.eval_scales[g])
        model = self._models[g]
        core, factors = model.nonneg_views()
        acc = nnz_term_sum(model, core, factors, indices, values, self.divergence,
                           self.masked, self.eval_chunk, _EPS)
        out[g] = float(acc) * scale

    def final_relative_error(self) -> float:
        """The exact error over all nnz, whatever ``eval_sample`` is."""
        return self._full_relative_error(exact=True)

    @torch.no_grad()
    def _full_relative_error(self, exact: bool = False) -> float:
        """The logged error: on the eval subset when ``eval_sample`` is set,
        unless ``exact``. Normalized as ``sgd_step.full_relative_error``.

        One device evaluates in the tensors' own dtype on the device. Several
        sum their shards' nnz terms (rescaled when subsampled) on the host and
        add the exact zero-entry term there."""
        if len(self.devices) == 1:
            if exact or self.eval_scales[0] == 1.0:
                return full_relative_error(
                    self.master, self.shard_indices[0], self.shard_values[0],
                    self.divergence, masked=self.masked, chunk=self.eval_chunk,
                    totals=self._totals)
            return full_relative_error(
                self.master, self.eval_indices[0], self.eval_values[0], self.divergence,
                masked=self.masked, chunk=self.eval_chunk,
                sample_scale=self.eval_scales[0], totals=self._totals)

        eps = _EPS
        sinks: list = [None] * len(self.devices)
        self._fan_out(self._shard_error_terms, sinks, exact)
        acc = sum(sinks)
        vals_sum, vals_sq_sum = self._totals
        if self.divergence == "kl":
            total = acc if self.masked else acc + float(self.master.total_sum())
            return float(total / max(vals_sum, eps))
        total = acc if self.masked else acc + float(self.master.total_sq_norm())
        return float(max(total, 0.0) ** 0.5 / max(vals_sq_sum ** 0.5, eps))

    # --- materialization / checkpointing --------------------------------------

    def materialize(self) -> Tuple[np.ndarray, List[np.ndarray]]:
        """Detached CPU numpy (core, factors); for TT/utt ``core`` is a list."""
        return self.master.materialize()

    def checkpoint_payload(self, iteration: int) -> dict:
        """Device 0's payload: replicas are identical at a block boundary."""
        return checkpoint_payload(self.master, self.opt, iteration)

    def load_payload(self, payload: dict) -> None:
        check_sgd_payload(payload)
        self.master.load_state_dict(payload["raw_state_dict"])
        self._broadcast_params()
        # Every device steps, so every optimizer needs the restored moments.
        # The deepcopy matters: Adam keeps ``step`` on the host, where ``.to()``
        # returns the same object, so without it all optimizers would share one
        # ``step`` tensor and increment it concurrently.
        for opt in self._opts:
            opt.load_state_dict(copy.deepcopy(payload["optim_state"]))
        self._check_optimizer_state_disjoint()

    def _check_optimizer_state_disjoint(self) -> None:
        """No optimizer state tensor may be shared between two replicas (see
        ``load_payload``); checked once, at resume."""
        owner: dict = {}
        for g, opt in enumerate(self._opts):
            for st in opt.state.values():
                for key, v in st.items():
                    if not isinstance(v, torch.Tensor):
                        continue
                    first = owner.setdefault(id(v), g)
                    if first != g:
                        raise RuntimeError(
                            f"optimizer state {key!r} is the same tensor object on "
                            f"replica {first} and replica {g}. Every replica steps "
                            "its own optimizer, so shared state means they no "
                            "longer step identically. Expected the per-optimizer "
                            "deepcopy in load_payload to prevent this."
                        )
