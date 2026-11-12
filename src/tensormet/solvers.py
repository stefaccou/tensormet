"""
solvers.py — the two ways to run one training iteration, behind one interface.

``MUSolver`` (multiplicative updates, CuPy) and ``SGDSolver`` (torch minibatch
Adam, sgd/README.md) both offer what the loop in training.py needs:

    step(iteration, log_step) -> relative error on log steps, else None
    eval_views()              torch (core, factors) for the semantic evaluation
    snapshot()                a copy of the current state, to keep as "best"
    current()                 the live (core, factors)
    container(state, host)    the object that is saved / returned for a state
    checkpoint(iteration)     what a periodic or interrupt checkpoint file holds
    free_cache()              give cached GPU memory back to the driver
    final_error(rec_errors)   the error reported at the end
    size_batches()            after the judge is loaded (batch sizes depend on free VRAM)
    describe()                the routing path, for the log

Setup (init or resume, tying shared factors, the sharded tensor, the
subsampler, the grouping cache) happens in the constructors; what differs per
decomposition family is in families.py.
"""
from __future__ import annotations

import time
from collections import defaultdict

import numpy as np
import torch
from tensorly.tucker_tensor import tucker_normalize, validate_tucker_rank

from tensormet.distance import (
    NNZGroupingCache, coords_nnz, precompute_largedim_batches,
)
from tensormet.routing import needs_largedim
from tensormet.sharded_sparse import ShardedSparseTensor
from tensormet.sparse_ops import CoordCOO
from tensormet.stochastic_sparse import CooSubsampler
from tensormet.utils import make_lazy_cupy_pair, nontrivial_linked_groups

cp, cpx_sparse = make_lazy_cupy_pair()


def _as_host(x):
    """Host-side value for save/eval sites shared by the MU (CuPy) and SGD
    (torch/numpy) solver paths. Never touches the lazy `cp` for non-CuPy input,
    so torch-only environments stay CuPy-free."""
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    if isinstance(x, (np.ndarray, np.generic, float, int)) or x is None:
        return x
    if isinstance(x, (list, tuple)):
        return [_as_host(v) for v in x]
    return cp.asnumpy(x)


def _torch_views(arrays):
    """Zero-copy torch views of CuPy arrays. A repeated array (tied modes) maps
    to one tensor, so evaluate_sample normalises it only once."""
    views = {}
    for a in arrays:
        if id(a) not in views:
            views[id(a)] = torch.from_dlpack(a)
    return [views[id(a)] for a in arrays]


def resolve_rank(shape, cfg):
    """``(rank, modes)``: the validated per-mode ranks (CP: one rank) and the modes."""
    rank = list(cfg.exp.rank)
    if cfg.exp.decomposition == "cp":
        # CP uses a single rank R (no per-mode core dimensions). Accept the
        # config's per-mode tuple only when uniform.
        from tensorly.cp_tensor import validate_cp_rank
        if len(set(rank)) > 1:
            raise ValueError(
                f"CP decomposition uses a single rank; got per-mode ranks {tuple(rank)}. "
                f"Pass a uniform --rank."
            )
        return validate_cp_rank(shape, rank=int(rank[0])), list(range(len(shape)))
    rank = validate_tucker_rank(shape, rank=rank)
    return rank, list(range(len(rank)))


def effective_subsample_frac(cfg, nnz, announce=False):
    """``subsample_frac``, lowered so that at most ``max_nnz`` entries are used per
    update step. Every frac < 1 gate, the one-time shuffles, the rotating windows
    and the 1/frac rescales key off this value, and per-shard sampling of it sums
    to ~max_nnz whatever the shard count. Filenames and resume checks carry the
    raw ``max_nnz`` int, never this derived value."""
    frac = cfg.exp.subsample_frac
    max_nnz = int(cfg.exp.max_nnz or 0)
    if max_nnz < 0:
        raise ValueError(f"cfg.exp.max_nnz must be >= 0, got {max_nnz}")
    if max_nnz and nnz > max_nnz:
        frac = min(frac, max_nnz / nnz)
        if announce:
            print(f"max_nnz={max_nnz}: effective subsample_frac -> {frac:.6g} (nnz={nnz})")
    return frac


def _set_svd_stream_context(cfg, tensor, modes):
    """Experimental streamed GPU SVD: pooled per tied group, all GPUs, cached."""
    from tensormet.stream_svd import set_run_context
    set_run_context(cfg, tied_groups=nontrivial_linked_groups(tensor.shared_factors,
                                                              num_factors=len(modes)),
                    device_ids=list(range(cfg.train.n_gpus)),
                    power_iter=cfg.exp.svd_power_iter,
                    oversampling=cfg.exp.svd_oversampling)


# ---------------------------------------------------------------------------
# Multiplicative updates
# ---------------------------------------------------------------------------

class MUSolver:
    """KL / Frobenius multiplicative updates on CuPy arrays, single GPU or NNZ-sharded."""

    backend = "cupy"
    sync_backend = "cupy"

    def __init__(self, tensor, cfg, family, rank, modes, thread_budget, resume_state):
        e = cfg.exp
        self.tensor, self.cfg, self.family = tensor, cfg, family
        self.rank, self.modes, self.thread_budget = rank, modes, thread_budget
        self.shape = tuple(tensor.shape)
        self.n_gpus = cfg.train.n_gpus
        self.masked = e.objective == "masked"
        data = tensor.tensor

        # --- init, or the resumed checkpoint ---
        checkpoint = resume_state.get("checkpoint_tensor")
        if e.init == "svd_stream" and checkpoint is None:
            _set_svd_stream_context(cfg, tensor, modes)
        if checkpoint is not None:
            self.core, self.factors = self._from_checkpoint(checkpoint, data)
        else:
            self.core, self.factors = family.initialize(data, self.shape, rank, modes,
                                                        cfg, thread_budget)

        # --- sampling: max_nnz resolves into an effective fraction ---
        full_nnz = coords_nnz(data)
        self.subsample_frac = effective_subsample_frac(cfg, full_nnz, announce=True)
        self.sst = (ShardedSparseTensor.from_coo(
            data, self.shape, device_ids=list(range(self.n_gpus)),
            subsample_frac=self.subsample_frac, masked=self.masked,
            # Seeds the one-time per-shard NNZ shuffle used by subsampling.
            subsample_seed=int(e.random_state or 0),
        ) if self.n_gpus > 1 else None)
        # Single-GPU windows are a pure function of (random_state, iteration): resume-safe.
        self.sampler = (CooSubsampler(data, self.shape, self.subsample_frac, e.random_state)
                        if self.subsample_frac < 1.0 and self.sst is None else None)

        # --- Tucker's largedim kernels: grouping cache and batch sizes, built once ---
        # (CP and TT stream NNZ directly with their own batch estimator.)
        self.largedim_selected = needs_largedim(e.dim, largedim=cfg.train.largedim,
                                                masked=self.masked)
        tucker_largedim = e.decomposition == "tucker" and self.largedim_selected
        self.grouping_cache = (NNZGroupingCache(data, self.shape)
                               if tucker_largedim and self.sst is None
                               and self.subsample_frac >= 1.0 else None)
        nnz_live = self.sampler.n_sample if self.sampler is not None else full_nnz
        self.largedim_batches = (precompute_largedim_batches(
            self.core, self.factors, modes, masked=self.masked, nnz_live=nnz_live,
            coord_backed=isinstance(data, CoordCOO))
            if tucker_largedim and e.divergence == "kl" and self.sst is None else None)

        # One AdaptiveCorePower for the whole run: its exponent carries across iterations.
        self.core_power = e.utt_core_power
        if e.decomposition == "utt" and self.core_power == "auto":
            from tensormet.uniform_tt.utt_ops import AdaptiveCorePower
            self.core_power = AdaptiveCorePower()

        # --- shared factors ---
        self.linked = defaultdict(set)
        for a, b in (tensor.shared_factors or ()):
            self.linked[a].add(b)
            self.linked[b].add(a)
        # Sets of modes that share one matrix: {(0,1),(1,2),(2,3)} is one group
        # of four, not three pairs (``linked`` only has the direct links).
        self.tied_groups = nontrivial_linked_groups(tensor.shared_factors,
                                                    num_factors=len(modes))
        self._tie_factors()

        self.batch_nnz = None
        self._core_hints, self._error_hints = {}, {}
        self._build_hints()

    # --- setup helpers ---

    def _from_checkpoint(self, checkpoint, data):
        if isinstance(checkpoint, tuple) or self.cfg.exp.decomposition == "cp":
            # TensorLy TuckerTensor / CPTensor / plain (core|tt_cores, factors) tuple
            core, factors = checkpoint
        else:
            core, factors = checkpoint.core, checkpoint.factors
        core = ([cp.asarray(C) for C in core] if self.family.list_core
                else cp.asarray(core))
        factors = [cp.asarray(f) for f in factors]
        return self.family.check_resume(core, factors, self.cfg, self.rank, data.data.dtype)

    def _tie_factors(self):
        """Make shared modes actually share from step 0. Init draws one matrix per
        mode, and a resumed legacy checkpoint holds several (the old per-mode +
        copy path never left them equal), so without this the first pooled
        update would swap several different matrices for one in a single jump."""
        for group in self.tied_groups:
            head = self.factors[group[0]]
            if any(self.factors[n].shape != head.shape for n in group):
                raise ValueError(
                    f"shared_factors ties modes {group} but their factors have "
                    f"shapes {[tuple(self.factors[n].shape) for n in group]}; tied "
                    f"modes must have the same dimension and rank."
                )
            if any(not bool(cp.allclose(self.factors[n], head)) for n in group[1:]):
                print(f"shared_factors: modes {group} held different factor "
                      f"matrices; tying them to mode {group[0]}'s.")
            for n in group[1:]:
                self.factors[n] = head

    def _build_hints(self):
        """Batch / grouping hints for the kernels, as keyword dicts. The error
        hints only apply on log steps (the null error function takes none)."""
        if self.largedim_batches is not None:
            self._core_hints = dict(batch_rhat=self.largedim_batches["batch_rhat"],
                                    batch_num=self.largedim_batches["batch_num"])
            self._error_hints = dict(batch_rhat=self.largedim_batches["batch_rhat"])
        if self.batch_nnz is not None:
            self._core_hints = self._error_hints = dict(batch_nnz=self.batch_nnz)

    def size_batches(self):
        """Size the TT streaming batch once. It depends only on the core / factor
        shapes and dtype, but on free VRAM too, so this runs after the judge is
        resident, and on the device the cores live on (a sharded run must never
        size against a shard's GPU)."""
        if self.family.batch_nnz is not None:
            with self.core[0].device:
                self.batch_nnz = self.family.batch_nnz(self.core, self.factors)
            self._build_hints()

    def _routing(self, log_step):
        if self.sst is not None:
            return self.family.sharded_routing(self.sst, self.cfg, log_step, self.core_power)
        return self.family.routing(self.cfg, log_step, self.core_power)

    def _factor_hints(self, mode):
        hints = {}
        if self.grouping_cache is not None:
            hints["grouping"] = self.grouping_cache.get(mode)
        if self.largedim_batches is not None:
            hints["batch_cols"] = self.largedim_batches["batch_cols"][mode]
        if self.batch_nnz is not None:
            hints["batch_nnz"] = self.batch_nnz
        return hints

    def _tied_hints(self, group):
        hints = {}
        # Tucker's pooler calls the per-mode kernel once per tied mode, so it
        # takes the same hints that loop does. Tied modes share a shape, so one
        # batch_cols covers the group.
        if self.largedim_batches is not None:
            hints["batch_cols"] = self.largedim_batches["batch_cols"][group[0]]
        if self.grouping_cache is not None:
            hints["grouping_by_mode"] = {n: self.grouping_cache.get(n) for n in group}
        if self.batch_nnz is not None:
            hints["batch_nnz"] = self.batch_nnz
        return hints

    # --- one iteration ---

    def step(self, iteration, log_step):
        cfg = self.cfg
        common = dict(thread_budget=self.thread_budget, epsilon=cfg.exp.epsilon,
                      verbose=cfg.train.verbose)
        routing = self._routing(log_step)
        if self.sst is not None:
            self.sst.set_iter_seed(iteration)
        # _iter sampler is None on the multi-GPU path (the shards sample internally)
        current = (self.sampler.sample(iteration) if self.sampler is not None
                   else self.tensor.tensor)

        # A shared factor sits at several legs at once, so updating it once per
        # mode and copying applies the correction once per leg. Families with a
        # pooled update (Tucker and TT) do the group in one step; the rest keep
        # the per-mode update + copy.
        pooled = self.tied_groups if routing.tied_factor_update is not None else []
        for group in pooled:
            A = routing.tied_factor_update(
                vec_tensor=current, core=self.core, factors=self.factors, group=group,
                shape=self.shape, **common, **self._tied_hints(group))
            for n in group:
                self.factors[n] = A
        pooled_modes = {m for group in pooled for m in group}
        for mode in self.modes:
            if mode in pooled_modes:
                continue
            self.factors[mode] = routing.factor_update(
                vec_tensor=current, core=self.core, factors=self.factors, mode=mode,
                shape=self.shape, **common, **self._factor_hints(mode))
            for other in self.linked.get(mode, ()):
                self.factors[other] = self.factors[mode]

        if routing.core_returns_error:
            # FR: combined core update + error in one call. Only runs on log
            # steps, and gets the FULL tensor so the error is unbiased.
            self.core, rel_err = routing.core_update(
                vec_tensor=self.tensor.tensor, shape=self.shape, core=self.core,
                factors=self.factors, modes=self.modes, **common)
        else:
            self.core = routing.core_update(
                vec_tensor=current, shape=self.shape, core=self.core,
                factors=self.factors, modes=self.modes, **common, **self._core_hints)
            # The KL error runs on the FULL tensor: it is nonlinear in the
            # rescaled values, so a subsampled error would be biased. It only
            # does real work on log steps. (The core update above is linear, so
            # the subsample is fine there.)
            rel_err = routing.error_fn(
                vec_tensor=self.tensor.tensor, shape=self.shape, core=self.core,
                factors=self.factors, **common,
                **(self._error_hints if log_step else {}))
        # CP and TT skip this: their factor updates already keep the columns
        # normalized, with the scale absorbed into λ / the TT core of that mode
        # (tucker_normalize needs a dense core anyway).
        if cfg.exp.normalize_factors and cfg.exp.decomposition == "tucker":
            self.core, self.factors = tucker_normalize((self.core, self.factors))
        return rel_err

    # --- state ---

    def eval_views(self):
        """Zero-copy torch views of the CuPy arrays; tied modes keep one tensor."""
        core = (_torch_views(self.core) if self.family.list_core
                else torch.from_dlpack(self.core))
        return core, _torch_views(self.factors)

    def current(self):
        return self.core, self.factors

    def snapshot(self):
        core = ([C.copy() for C in self.core] if self.family.list_core
                else self.core.copy())
        return core, [f.copy() for f in self.factors]

    def container(self, state, host=True):
        """The model object for ``state``. Saved files hold host arrays; the model
        a run returns keeps whatever the state holds (launch.py converts)."""
        core, factors = state
        if host:
            core, factors = _as_host(core), [_as_host(f) for f in factors]
        return self.family.container(core, factors)

    def checkpoint(self, iteration):
        return self.container(self.current())

    def free_cache(self):
        if self.sst is not None:
            self.sst.trim_pools()
        else:
            cp.get_default_memory_pool().free_all_blocks()
            cp.get_default_pinned_memory_pool().free_all_blocks()

    def final_error(self, rec_errors):
        return rec_errors[-1] if len(rec_errors) > 0 else None

    def describe(self):
        e = self.cfg.exp
        sharded = f", sharded×{self.n_gpus}" if self.sst is not None else ""
        if e.decomposition == "cp":
            return f"cp (nnz-streaming, inner_iters={e.cp_inner_iters}{sharded})"
        if e.decomposition == "utt":
            return f"shared-core tt (nnz-streaming, bond={int(self.core[0].shape[0])}{sharded})"
        if e.decomposition == "tt":
            return (f"tucker-tt (nnz-streaming, bonds="
                    f"{[int(C.shape[2]) for C in self.core[:-1]]}{sharded})")
        if self.sst is not None and self.largedim_selected:
            return f"sharded×{self.n_gpus}"
        return "largedim" if self.largedim_selected else "small-dim dense"


# ---------------------------------------------------------------------------
# SGD
# ---------------------------------------------------------------------------

class SGDSolver:
    """Torch minibatch Adam (sgd/README.md). One loop iteration is a block of
    ``sgd_steps_per_iteration`` optimizer steps run inside the trainer, which
    owns the Adam moments, the raw parameters and the step counter — state the
    UpdateRouting seam cannot express."""

    backend = "pytorch"
    sync_backend = "torch"

    def __init__(self, tensor, cfg, family, rank, modes, thread_budget, resume_state):
        e = cfg.exp
        self.cfg, self.family, self.n_gpus = cfg, family, cfg.train.n_gpus
        self.subsample_frac = effective_subsample_frac(cfg, int(tensor.tensor._nnz()))
        payload = resume_state.get("checkpoint_tensor")
        if payload is not None and not (isinstance(payload, dict)
                                        and payload.get("solver") == "sgd"):
            raise ValueError(
                "solver='sgd' found a non-SGD checkpoint payload; refusing to "
                "resume from it (MU checkpoints carry no optimizer state)."
            )
        # The init runs only on a fresh start (no resume payload, no warm start).
        runs_init = payload is None and not e.sgd_warm_start
        if e.init == "svd_stream" and runs_init:
            _set_svd_stream_context(cfg, tensor, modes)

        init = e.init
        if payload is None and e.sgd_warm_start:
            # Warm start (init from an MU model artifact) is distinct from resume:
            # parameters start at the MU solution but the optimizer state and
            # step counter start fresh. The artifact is a plain CPU-numpy
            # TuckerTensor (tuple-unpackable), for tt/utt a (cores, factors)
            # tuple; a TuckerDecomposition payload's .core/.factors also works.
            loaded = torch.load(e.sgd_warm_start, map_location="cpu", weights_only=False)
            try:
                core, factors = loaded
            except (TypeError, ValueError):
                core, factors = loaded.core, loaded.factors
            init = (core, factors)
            print(f"SGD warm start from {e.sgd_warm_start}")
        elif payload is not None:
            # Resume: load_payload overwrites the start, so build the model from
            # the checkpoint's own values instead of running --init again.
            init = (payload["core"], payload["factors"])
        elif runs_init and isinstance(e.init, str) and e.init != "random":
            # SVD inits: run as on the MU path, hand over like a warm start.
            from tensormet.sgd.sgd_init import mu_init_for_sgd
            init = mu_init_for_sgd(
                tensor, e.decomposition, tuple(tensor.shape), rank, modes, e.init,
                e.random_state, tt_rank=e.tt_rank, thread_budget=thread_budget,
                epsilon=e.epsilon, core_init=e.tt_core_init,
                device_ids=list(range(self.n_gpus)))

        kwargs = dict(
            rank=rank, divergence=e.divergence, objective=e.objective,
            lr=e.sgd_lr, batch_size=e.sgd_batch_size, parametrization=e.sgd_parametrization,
            shared_factors=tensor.shared_factors, init=init, random_state=e.random_state,
            steps_per_iteration=e.sgd_steps_per_iteration, epsilon=e.epsilon,
            micro_batch=e.sgd_micro_batch, eval_sample=e.sgd_eval_sample,
            resume_payload=payload, decomposition=e.decomposition, tt_rank=e.tt_rank,
            regauge_every=e.sgd_regauge_every,
        )
        # Construction is the last silent stretch before the loop announces
        # itself, and on the sharded path it is where the collective gets
        # built — historically the place a run could stall with no output.
        print(f"[{time.strftime('%H:%M:%S')}] constructing SGD trainer "
              f"(n_gpus={self.n_gpus})...", flush=True)
        from tensormet.sgd.sgd_trainer import SGDTrainer
        # One GPU is the default device, as launch.py selected it.
        self.trainer = SGDTrainer(
            tensor.tensor, devices=list(range(self.n_gpus)) if self.n_gpus > 1 else None,
            comm_backend=e.sgd_comm_backend, **kwargs)
        print(f"[{time.strftime('%H:%M:%S')}] SGD trainer ready", flush=True)

    def size_batches(self):
        """The trainer sizes its own batches."""

    def step(self, iteration, log_step):
        # On log steps the trainer returns the exact full-NNZ relative error, with
        # the same normalization as the MU error kernels.
        return self.trainer.run_block(iteration, log_step)

    def eval_views(self):
        core, factors = self.trainer.materialize()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        core = ([torch.as_tensor(C, device=device) for C in core] if self.family.list_core
                else torch.as_tensor(core, device=device))
        return core, [torch.as_tensor(f, device=device) for f in factors]

    def current(self):
        return self.trainer.materialize()

    snapshot = current

    def container(self, state, host=True):
        """Saved / returned model from host snapshots. TT/utt: a plain (cores,
        factors) tuple of numpy arrays. Tucker: tensorly validates the container
        against the ACTIVE backend ("pytorch" here, whose ndim() rejects numpy),
        so wrap the snapshots in CPU torch tensors (zero-copy, loads fine on CPU)."""
        core, factors = state
        if self.family.list_core:
            return self.family.container([np.asarray(C) for C in core],
                                         [np.asarray(f) for f in factors])
        return self.family.container(torch.as_tensor(core),
                                     [torch.as_tensor(f) for f in factors])

    def checkpoint(self, iteration):
        """Dict payload: (core, factors) views for tooling that peeks, plus raw
        params + optimizer state so resume continues the exact trajectory
        (batches replay from the step counter)."""
        return self.trainer.checkpoint_payload(iteration + 1)

    def free_cache(self):
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def final_error(self, rec_errors):
        """With --sgd-eval-sample the logged curve is a subsampled estimate; pay
        for one exact pass at the end so the reported final_error is exact.
        (Like rec_errors[-1], it describes the trainer's current state, which is
        not necessarily the best-semantics model returned.)"""
        if getattr(self.trainer, "eval_sample", None):
            exact = self.trainer.final_relative_error()
            print(f"Exact final reconstruction error: {exact:.6f} "
                  f"(logged curve used --sgd-eval-sample {self.trainer.eval_sample})")
            return exact
        return rec_errors[-1] if len(rec_errors) > 0 else None

    def describe(self):
        label = "" if self.cfg.exp.decomposition == "tucker" else f", {self.cfg.exp.decomposition}"
        return (f"sgd (torch{label}, sharded×{self.n_gpus})" if self.n_gpus > 1
                else f"sgd (torch{label})")


def make_solver(tensor, cfg, family, rank, modes, thread_budget, resume_state):
    cls = SGDSolver if cfg.exp.solver == "sgd" else MUSolver
    return cls(tensor, cfg, family, rank, modes, thread_budget, resume_state)
