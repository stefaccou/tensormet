"""
sharded_sgd.py — EXPERIMENTAL: single-process multi-GPU SGD Tucker trainer.

Data-parallel counterpart of ``sgd_trainer.SGDTrainer``, selected by the loop
when ``cfg.exp.solver == "sgd"`` and ``cfg.train.n_gpus > 1``. Philosophy
matches ``sharded_sparse.ShardedSparseTensor``: ONE process, per-device
shards, no torch.distributed / DDP / spawn, so the tee logger, SIGINT handler,
judge, and checkpoint writer stay unambiguous.

Why the shape of this file changed
----------------------------------
The first version was slower on 2 GPUs than on 1 at every batch size measured.
It was not a correctness problem — it was three structural ones:

* **Serialized dispatch.** All G forward graphs were built in one Python loop,
  then the autograd engine ran G times in another. At production mode
  dimensions the order-3 step is ~50x off its arithmetic roofline (step time
  was nearly batch-independent: 128x the batch for 2x the time), so the step is
  dispatch, launch and autograd overhead. Doubling the devices doubled the
  serialized overhead — exactly the observed x0.5. Fixed by the persistent
  thread-per-GPU pool that MU has always used (``sharded_sparse:1284-1332``).
* **Per-parameter, per-replica collectives** plus a full ``state_dict()``
  rebuild and a parameter broadcast every step. Fixed by flattening: parameter
  ``.grad``s are views into one contiguous buffer (``GradStepper``), so a step
  costs *one* all-reduce. And because an all-reduce leaves the sum on every
  device, the optimizer runs redundantly per device and the broadcast is gone.
* **The batch was divided across devices**, so G GPUs did the single-GPU amount
  of sampled work for G times the overhead — while the docs claimed a larger
  effective batch. ``batch_scope="per_device"`` (the default) now means what
  the docs said.

Layout
------
- The coalesced NNZ is split contiguously into ``len(device_ids)`` shards;
  shard *g*'s indices/values live on device *g* with its own ``EntryBatcher``
  seeded ``random_state * 1000 + g`` (same convention as sharded_sparse), so
  every shard's batch remains a pure function of (seed, step) and resume
  replays exactly.
- Every device holds a full model replica, its own optimizer, and its own
  ``GradStepper``. Device 0's is aliased as ``self.master`` / ``self.opt`` so
  checkpointing is unchanged and payloads stay byte-compatible with the
  single-GPU trainer's.

Per step (``sgd_sync_every == 1``)
----------------------------------
1. fan out to the pool: each device computes the *sampled* loss term on its own
   sub-batch (scale = nnz_g / batch_g, so the sum over devices is unbiased for
   the total) and backprops it, micro-batched, into its flat gradient buffer.
   The EXACT zero-entry term is added on device 0 only — it is a function of
   the parameters alone, so adding it once keeps the summed gradient correct;
2. one all-reduce of the G flat gradient buffers;
3. fan out again: every device runs ``opt.step()`` + ``project_()`` on the same
   summed gradient, so all replicas stay bit-identical without any broadcast.

Per ``sgd_sync_every == K > 1``
-------------------------------
Each device takes K local Adam steps, then parameters are averaged
(all-reduce / G) — standard local SGD. This divides the barrier count by K,
which is the highest-leverage knob when the step is dispatch-bound. The loss
changes under local steps: each device must optimize an unbiased estimate of
the *full* objective on its own, so ``scale_g = nnz / batch_g`` (global nnz)
and **every** device adds the zero-entry term in full. That makes ``K > 1`` a
win when the zero-entry term is cheap relative to overhead (KL, order 3) and a
loss when it dominates (FR with a large core). See SGD/README.md.

Checkpoints carry only device 0's state — identical payload to the single-GPU
trainer. Resume across a *different* ``n_gpus`` no longer reproduces the same
trajectory under the new defaults (per-device batching and local steps both
make the trajectory a function of G), so ``n_gpus`` joined the
resume-compatibility key; under ``batch_scope="global"`` with
``sync_every=1`` the old promise still holds.
"""
from __future__ import annotations

import copy
from concurrent.futures import ThreadPoolExecutor
from typing import Callable, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch

from tensormet.experimental.SGD.collectives import make_collective
from tensormet.experimental.SGD.sgd_tucker import (
    _EPS,
    EntryBatcher,
    GradStepper,
    SGDTuckerModel,
    _default_eval_chunk,
    sampled_loss,
)

# Back-compat alias: the pre-Phase-1 module exposed the shard-local objective
# under this name and the multi-GPU notebook imports it from here.
_sampled_loss = sampled_loss


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
        batch_scope: str = "per_device",
        sync_every: int = 1,
        micro_batch: Optional[int] = None,
        cuda_graph: bool = False,
        comm_backend: str = "auto",
        resume_payload: Optional[dict] = None,
    ):
        if objective not in ("full", "masked"):
            raise ValueError(f"objective must be 'full' or 'masked', got {objective!r}")
        if divergence not in ("kl", "fr"):
            raise ValueError(f"divergence must be 'kl' or 'fr', got {divergence!r}")
        if batch_scope not in ("per_device", "global"):
            raise ValueError(
                "batch_scope must be 'per_device' (batch_size entries on EVERY "
                f"device) or 'global' (batch_size split across them); got {batch_scope!r}"
            )
        self.divergence = divergence
        self.masked = objective == "masked"
        self.steps_per_iteration = int(steps_per_iteration)
        if self.steps_per_iteration < 1:
            raise ValueError("steps_per_iteration must be >= 1")
        self.batch_scope = batch_scope
        self.sync_every = max(1, int(sync_every))

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

        # One full replica + optimizer per device. Device 0 is the "master" only
        # in the checkpointing sense; the optimizer step runs redundantly on all
        # of them off the same all-reduced gradient (see module docstring), so
        # there is no per-step parameter broadcast.
        self._models = [
            SGDTuckerModel(
                shape=t.shape, rank=rank, parametrization=parametrization,
                shared_factors=shared_factors, init=init, random_state=random_state,
                device=dev, dtype=dtype, eps=epsilon,
            )
            for dev in self.devices
        ]
        self.master = self._models[0]
        self.replicas = self._models[1:]
        # Before anything runs a matmul or a backward concurrently — including
        # the CUDA-graph capture below, which must not record a lazy cubin load.
        self._warm_up_devices(dtype)

        # Global normalization constant (identical to the single-GPU trainer).
        if divergence == "kl":
            self.norm_const = float(values_all.sum().clamp_min(_EPS))
        else:
            self.norm_const = float(values_all.pow(2).sum().clamp_min(_EPS))

        # Contiguous NNZ shards, one per device, with per-shard batchers seeded
        # like sharded_sparse (base * 1000 + shard). No device holds the full
        # permutation.
        bounds = np.linspace(0, self.nnz, n_shards + 1, dtype=np.int64)
        if batch_scope == "per_device":
            # G devices sample G x batch_size entries per step: the effective
            # batch now actually grows with G, and per-device work is constant.
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
                # Local SGD: each device optimizes the FULL objective alone, so
                # it scales to the global nnz and adds the zero-entry term
                # itself. Parameter averaging, not gradient summing, combines
                # the devices.
                scale = self.nnz / batcher.batch_size
                include_zero = True
            else:
                # Gradients are summed every step, so each device carries only
                # its shard's nnz term and exactly one device carries Z.
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

        self.eval_chunk = eval_chunk or _default_eval_chunk(self.master.rank)

        if optimizer == "adam":
            self._opts = [torch.optim.Adam(m.parameters(), lr=lr) for m in self._models]
        elif optimizer == "sgd":
            self._opts = [torch.optim.SGD(m.parameters(), lr=lr, momentum=0.9)
                          for m in self._models]
        else:
            raise ValueError(f"optimizer must be 'adam' or 'sgd', got {optimizer!r}")
        self.opt = self._opts[0]

        self.collective = make_collective(self.devices, backend=comm_backend)

        # Precomputed (src, dst) tensor pairs for the cold-path broadcast — the
        # old code rebuilt two OrderedDicts per call, and called it every step.
        with torch.no_grad():
            master_state = self.master.state_dict()
            self._bcast_pairs: List[List[Tuple[torch.Tensor, torch.Tensor]]] = [
                [(master_state[name], p) for name, p in rep.state_dict().items()]
                for rep in self.replicas
            ]

        # Persistent pool: threads (and their cuBLAS handles) live for the
        # lifetime of this object. Re-creating a pool per step causes thread-ID
        # recycling in Python 3.13, leaving stale cuBLAS handles — the same
        # failure sharded_sparse documents at its own pool construction.
        self._pool = ThreadPoolExecutor(max_workers=n_shards)

        # Lazily allocated: one flat parameter vector per device, for the
        # sync_every averaging and the drift check. Skipped entirely for
        # sync_every == 1 runs that never hit a log step.
        self._param_bufs: Optional[List[torch.Tensor]] = None

        if resume_payload is not None:
            self.load_payload(resume_payload)
        else:
            self._broadcast_params()

    # --- device plumbing ------------------------------------------------------

    def _warm_up_devices(self, dtype: torch.dtype) -> None:
        """Load cuBLAS/autograd kernels into every device context up front.

        CUDA 12 loads cubins lazily, so without this the G worker threads issue
        their first matmul (and their first backward) on their own devices
        *concurrently* in step 1, and those one-time loads race. MU hit this as
        an intermittent CUBLAS_STATUS_NOT_INITIALIZED with plenty of free VRAM;
        doing it serially here, before the pool goes live, is the same fix.
        """
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

        ``torch.cuda.set_device`` is re-applied per task rather than once per
        worker: ``ThreadPoolExecutor`` gives no guarantee that shard *g* lands
        on the same thread twice, and the call is a thread-local driver write.
        Exceptions surface here via ``future.result()`` instead of being
        swallowed into a hung barrier.
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
        """Block until every device is idle.

        ``run_block`` deliberately returns without synchronizing on non-log
        steps (queued work is the point), so any benchmark timing it must call
        this around the timed region or it will attribute work to the wrong
        block — which is what made the original 1-vs-2-GPU sweep non-monotonic.
        """
        for dev in self.devices:
            torch.cuda.synchronize(dev)

    def __del__(self) -> None:
        pool = getattr(self, "_pool", None)
        if pool is not None:
            pool.shutdown(wait=False)
            self._pool = None

    # --- parameter plumbing ---------------------------------------------------

    def _broadcast_params(self) -> None:
        """device 0 raw params -> replicas. Cold path only (construction and
        resume): the per-step broadcast is gone, replaced by the redundant
        optimizer step on the all-reduced gradient."""
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
        """Assert every device holds bit-identical parameters.

        The redundant optimizer step — the thing that lets the per-step
        parameter broadcast go away — is only valid if the all-reduce returns
        the same value on every rank and the optimizer kernels are
        deterministic. Both should hold (NCCL's ring reduction is
        bit-reproducible across ranks, and the ``foreach`` Adam kernels are
        deterministic), so this exists purely to make a violation loud instead
        of leaving it to surface as a subtly wrong final model.

        The comparison is exact rather than a checksum: a checksum that misses
        cancelling drift would defeat the point. It costs one parameter-sized
        device-0 scratch plus a G-1 transfer, at the caller's log-step cadence,
        not per step.
        """
        if len(self.devices) < 2:
            return
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
            # The sync cadence is clipped at the block boundary so parameters
            # are always averaged before an eval or a checkpoint. Pick
            # steps_per_iteration as a multiple of sync_every to avoid a short
            # final chunk each block.
            step = first
            while step < end:
                n = min(self.sync_every, end - step)
                self._fan_out(self._local_steps, step, n)
                self._average_params()
                step += n

        if not log_step:
            return None
        self._check_replica_drift()
        return self._full_relative_error()

    # --- exact error (sharded) ------------------------------------------------

    @torch.no_grad()
    def _shard_error_terms(self, g: int, out: list) -> None:
        eps = _EPS
        model = self._models[g]
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
        out[g] = (float(shard_acc), float(values.sum()), float(values.pow(2).sum()))

    @torch.no_grad()
    def _full_relative_error(self) -> float:
        """Per-shard chunked accumulation of the nnz term (one worker per
        device), host-summed; the analytic zero-entry term is added once. Same
        normalization as ``sgd_tucker.full_relative_error`` / the MU error
        kernels."""
        eps = _EPS
        sinks: list = [None] * len(self.devices)
        self._fan_out(self._shard_error_terms, sinks)
        acc = sum(s[0] for s in sinks)
        vals_sum = sum(s[1] for s in sinks)
        vals_sq_sum = sum(s[2] for s in sinks)

        if self.divergence == "kl":
            total = acc if self.masked else acc + float(self.master.total_sum())
            return float(total / max(vals_sum, eps))
        total = acc if self.masked else acc + float(self.master.total_sq_norm())
        return float(max(total, 0.0) ** 0.5 / max(vals_sq_sum ** 0.5, eps))

    # --- materialization / checkpointing --------------------------------------

    def materialize(self) -> Tuple[np.ndarray, List[np.ndarray]]:
        return self.master.materialize()

    def checkpoint_payload(self, iteration: int) -> dict:
        """Device-0 payload, byte-compatible with ``SGDTrainer``'s. Every
        replica holds the same parameters at a block boundary (asserted by
        ``_check_replica_drift`` on log steps), so device 0 is the whole
        model, not a shard of it."""
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
        self._broadcast_params()
        # Every device runs its own optimizer, so every device needs the
        # restored moments — resuming with fresh moments on the replicas would
        # make them step differently from device 0 and break the replicas'
        # bit-identity.
        #
        # The deepcopy is load-bearing, not defensive. Optimizer.load_state_dict
        # moves each state tensor with ``value.to(device=param.device)``, which
        # yields a fresh tensor per optimizer for the moments — but Adam's
        # ``step`` counter is deliberately kept on the HOST (it is only moved to
        # the device for the capturable/fused paths), and ``.to()`` on a tensor
        # already on the target device returns *the same object*. Loading one
        # payload into all G optimizers therefore hands them a single shared
        # ``step`` tensor, which every ``opt.step()`` increments in place — from
        # G pool threads at once. The replicas then apply different bias
        # corrections and diverge on the first post-resume step. Copying per
        # optimizer removes the aliasing for ``step`` and anything else that
        # happens to land on the host.
        for opt in self._opts:
            opt.load_state_dict(copy.deepcopy(payload["optim_state"]))
        self._check_optimizer_state_disjoint()

    def _check_optimizer_state_disjoint(self) -> None:
        """No optimizer state tensor may be shared between two replicas.

        Cheap (one dict of ids, at resume only) and pointed: shared state is
        silent until the replicas have stepped, at which point it surfaces as
        drift a whole block away from its cause. See ``load_payload`` for the
        aliasing this guards against.
        """
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
