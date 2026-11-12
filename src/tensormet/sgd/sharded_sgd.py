"""
sharded_sgd.py — single-process multi-GPU SGD trainer (Tucker, TT, utt).

Parallel counterpart of ``sgd_trainer.SGDTrainer`` (``--n_gpus > 1``), in one
process like ``sharded_sparse.ShardedSparseTensor``: no torch.distributed, so
the logger, SIGINT handler, judge and checkpoint writer stay unambiguous.

Each device holds a contiguous NNZ shard with its own ``EntryBatcher``, a full
model replica, optimizer and ``GradStepper``; device 0's is ``self.master`` /
``self.opt`` and is what gets checkpointed.

Per step (``sync_every == 1``): each device backprops its sampled loss (the
exact zero-entry term only on device 0), one all-reduce sums the gradients,
and every device steps on the same sum, so replicas stay bit-identical.
With ``sync_every == K > 1``: K local steps (each device adds the full
zero-entry term), then parameter averaging. See sgd/README.md.
"""
from __future__ import annotations

import copy
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Callable, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch

from tensormet.sgd.collectives import make_collective
from tensormet.sgd.sgd_trainer import (
    check_common_args,
    check_sgd_payload,
    checkpoint_payload,
    coalesced_input,
    loss_norm_const,
    make_optimizer,
    make_sgd_model,
)
from tensormet.sgd.sgd_tucker import (
    _EPS,
    EntryBatcher,
    GradStepper,
    _rank_derived_chunk,
    make_eval_subset,
    nnz_term_sum,
)
from tensormet.utils import SparseCOOTensor


class ShardedSGDTrainer:
    """Multi-GPU minibatch SGD/Adam trainer; same public surface as
    ``SGDTrainer`` (run_block / materialize / checkpoint_payload)."""

    def __init__(
        self,
        sparse_coo: Union[torch.Tensor, SparseCOOTensor],
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
        eval_sample: Optional[int] = None,
        batch_scope: str = "per_device",
        sync_every: int = 1,
        micro_batch: Optional[int] = None,
        cuda_graph: bool = False,
        comm_backend: str = "auto",
        resume_payload: Optional[dict] = None,
        decomposition: str = "tucker",
        tt_rank: Optional[int] = None,
        regauge_every: int = 1,
    ):
        self.steps_per_iteration = check_common_args(objective, steps_per_iteration)
        if divergence not in ("kl", "fr"):
            raise ValueError(f"divergence must be 'kl' or 'fr', got {divergence!r}")
        if batch_scope not in ("per_device", "global"):
            raise ValueError(
                "batch_scope must be 'per_device' (batch_size entries on EVERY "
                f"device) or 'global' (batch_size split across them); got {batch_scope!r}"
            )
        self.divergence = divergence
        self.masked = objective == "masked"
        self.batch_scope = batch_scope
        self.sync_every = max(1, int(sync_every))
        # Blocks between gauge resets (0 = never); a no-op for Tucker.
        self.regauge_every = max(0, int(regauge_every))

        self.devices = [torch.device(f"cuda:{d}") for d in device_ids]
        n_shards = len(self.devices)
        if n_shards < 2:
            raise ValueError("ShardedSGDTrainer needs >= 2 devices; use SGDTrainer.")

        t = coalesced_input(sparse_coo, "ShardedSGDTrainer")
        indices_all = t.indices()
        values_all = t.values().to(dtype=dtype)
        if divergence == "kl" and bool((values_all < 0).any()):
            raise ValueError("KL divergence requires non-negative tensor values.")
        self.nnz = int(values_all.shape[0])
        if self.nnz == 0:
            raise ValueError("sparse tensor has no nonzero entries.")
        # (Σx, Σx²) over the whole tensor: the error denominators.
        self._totals = (float(values_all.sum()), float(values_all.pow(2).sum()))
        self.norm_const = loss_norm_const(divergence, *self._totals)

        # One full replica + optimizer per device. The optimizer step runs on
        # every device off the same all-reduced gradient, so there is no
        # per-step parameter broadcast.
        self._models = [
            make_sgd_model(
                decomposition, shape=t.shape, rank=rank, tt_rank=tt_rank,
                parametrization=parametrization, shared_factors=shared_factors,
                init=init, random_state=random_state, device=dev, dtype=dtype,
                eps=epsilon, sum_x=self._totals[0],
            )
            for dev in self.devices
        ]
        self.master = self._models[0]
        self.replicas = self._models[1:]
        # Before anything runs a matmul or a backward concurrently, including
        # the CUDA-graph capture below.
        self._warm_up_devices(dtype)

        # Contiguous NNZ shards, one per device, with per-shard batchers seeded
        # like sharded_sparse (base * 1000 + shard).
        bounds = np.linspace(0, self.nnz, n_shards + 1, dtype=np.int64)
        if batch_scope == "per_device":
            # G devices sample G x batch_size entries per step.
            shard_batch = int(batch_size)
        else:
            shard_batch = max(1, int(batch_size) // n_shards)
        self.shard_batch = shard_batch
        self.shard_indices: List[torch.Tensor] = []
        self.shard_values: List[torch.Tensor] = []
        self.batchers: List[EntryBatcher] = []
        self.scales: List[float] = []
        self.steppers: List[GradStepper] = []
        for g, dev in enumerate(self.devices):
            lo, hi = int(bounds[g]), int(bounds[g + 1])
            self.shard_indices.append(indices_all[:, lo:hi].to(dev))
            self.shard_values.append(values_all[lo:hi].to(dev))
            nnz_g = hi - lo
            batcher = EntryBatcher(nnz_g, shard_batch,
                                   seed=int(random_state) * 1000 + g, device=dev)
            self.batchers.append(batcher)
            if self.sync_every > 1:
                # Local SGD: each device optimizes the FULL objective alone (scaled
                # to the global nnz, with the zero-entry term); parameter
                # averaging combines the devices.
                scale = self.nnz / batcher.batch_size
                include_zero = True
            else:
                # Gradients are summed every step: each device carries its own
                # shard's nnz term and exactly one device carries the zero term.
                scale = nnz_g / batcher.batch_size
                include_zero = g == 0
            self.scales.append(scale)
            self.steppers.append(GradStepper(
                self._models[g], self.shard_indices[g], self.shard_values[g],
                batcher, scale=scale, divergence=divergence, masked=self.masked,
                norm_const=self.norm_const, include_zero_term=include_zero,
                micro_batch=micro_batch, cuda_graph=cuda_graph, eps=epsilon,
            ))
        self.micro_batch = self.steppers[0].micro_batch

        self.eval_chunk = eval_chunk or _rank_derived_chunk(
            self.master.rank, width=self.master.entry_width)
        # Per-shard eval subsets, sized in proportion to the shards so their
        # union is a uniform sample of the whole tensor.
        self.eval_indices: List[torch.Tensor] = []
        self.eval_values: List[torch.Tensor] = []
        self.eval_scales: List[float] = []
        for g in range(n_shards):
            nnz_g = int(self.shard_values[g].shape[0])
            n_g = (None if not eval_sample
                   else max(1, round(int(eval_sample) * nnz_g / self.nnz)))
            e_idx, e_val, e_scale, _ = make_eval_subset(
                self.shard_indices[g], self.shard_values[g], n_g,
                int(random_state) * 1000 + g,
            )
            self.eval_indices.append(e_idx)
            self.eval_values.append(e_val)
            self.eval_scales.append(e_scale)
        self.eval_sample = (None if all(s == 1.0 for s in self.eval_scales)
                            else sum(int(v.shape[0]) for v in self.eval_values))

        self._opts = [make_optimizer(m.parameters(), optimizer, lr) for m in self._models]
        self.opt = self._opts[0]

        # Printed, because 'auto' can fall back to HostReduce, which changes
        # what every timing of this run means.
        print(f"[{time.strftime('%H:%M:%S')}] building SGD collective "
              f"(requested backend={comm_backend!r})...", flush=True)
        self.collective = make_collective(self.devices, backend=comm_backend)
        print(f"[{time.strftime('%H:%M:%S')}] SGD collective: {self.collective!r}",
              flush=True)

        # (master tensor, replica tensor) pairs for the cold-path broadcast.
        with torch.no_grad():
            master_state = self.master.state_dict()
            self._bcast_pairs: List[List[Tuple[torch.Tensor, torch.Tensor]]] = [
                [(master_state[name], p) for name, p in rep.state_dict().items()]
                for rep in self.replicas
            ]

        # Persistent pool: the threads (and their cuBLAS handles) live as long
        # as this object. A pool per step recycles thread IDs in Python 3.13 and
        # leaves stale cuBLAS handles (as documented in sharded_sparse).
        self._pool = ThreadPoolExecutor(max_workers=n_shards)

        # One flat parameter vector per device for the sync_every averaging and
        # the drift check; allocated on first use.
        self._param_bufs: Optional[List[torch.Tensor]] = None

        if resume_payload is not None:
            self.load_payload(resume_payload)
        else:
            self._broadcast_params()

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
        """Run ``fn(g, *args)`` on every device, concurrently, and wait.

        ``torch.cuda.set_device`` is applied per task: the pool does not
        guarantee shard *g* the same thread twice. Worker exceptions are
        re-raised here.
        """
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

    def _ensure_param_bufs(self) -> List[torch.Tensor]:
        if self._param_bufs is None:
            self._param_bufs = [s.new_param_buffer() for s in self.steppers]
        return self._param_bufs

    def _pack_params(self, g: int) -> None:
        self.steppers[g].pack_params_(self._param_bufs[g])

    def _unpack_scaled(self, g: int, factor: float) -> None:
        self._param_bufs[g].mul_(factor)
        self.steppers[g].unpack_params_(self._param_bufs[g])

    def _average_params(self) -> None:
        """all-reduce the parameters and divide by G (local-SGD averaging)."""
        bufs = self._ensure_param_bufs()
        self._fan_out(self._pack_params)
        self.collective.all_reduce(bufs)
        self._fan_out(self._unpack_scaled, 1.0 / len(self.devices))

    def _check_replica_drift(self) -> None:
        """Raise unless every device holds bit-identical parameters.

        Stepping each replica on the same all-reduced gradient is only valid if
        they stay identical; this makes a violation loud. Exact comparison, at
        the log-step cadence.
        """
        bufs = self._ensure_param_bufs()
        self._fan_out(self._pack_params)
        ref = bufs[0]
        scratch = torch.empty_like(ref)
        for g in range(1, len(self.devices)):
            scratch.copy_(bufs[g])
            if not torch.equal(ref, scratch):
                n_bad = int((ref != scratch).sum())
                raise RuntimeError(
                    f"SGD replica drift: {n_bad} of {ref.numel()} parameters "
                    f"differ between {self.devices[0]} and {self.devices[g]}. "
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
        self._models[g].project_()

    def _regauge(self, g: int) -> None:
        self._models[g].regauge_()

    def _local_steps(self, g: int, first: int, n_steps: int) -> None:
        stepper, opt, model = self.steppers[g], self._opts[g], self._models[g]
        for step in range(first, first + n_steps):
            stepper.compute_grads(step)
            opt.step()
            model.project_()

    def run_block(self, iteration: int, log_step: bool) -> Optional[float]:
        k = self.steps_per_iteration
        first = iteration * k
        end = first + k

        if self.sync_every <= 1:
            for step in range(first, end):
                self._fan_out(self._grads_for, step)
                self.collective.all_reduce([s.flat_grad for s in self.steppers])
                self._fan_out(self._apply_step)
        else:
            # The sync cadence is clipped at the block boundary, so parameters
            # are averaged before every eval and checkpoint. Pick
            # steps_per_iteration as a multiple of sync_every.
            step = first
            while step < end:
                n = min(self.sync_every, end - step)
                self._fan_out(self._local_steps, step, n)
                self._average_params()
                step += n

        # Replicas are identical here and the reset is deterministic, so they
        # stay identical after it.
        if self.regauge_every and (iteration + 1) % self.regauge_every == 0:
            self._fan_out(self._regauge)

        if not log_step:
            return None
        self._check_replica_drift()
        return self._full_relative_error()

    # --- error (sharded) ------------------------------------------------------

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
        """Each device sums its shard's nnz term (rescaled when subsampled);
        the host adds them and the exact zero-entry term. Same normalization as
        ``sgd_tucker.full_relative_error``."""
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
        return self.master.materialize()

    def checkpoint_payload(self, iteration: int) -> dict:
        """Device 0's payload, the same format as ``SGDTrainer``'s: replicas are
        identical at a block boundary."""
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
