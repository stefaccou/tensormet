# SGD Tucker solver (`--solver sgd`)

Torch-native minibatch SGD/Adam alternative to the multiplicative-update (MU)
pipeline, selectable through the standard argument machinery:

```bash
python3 -m tensormet.scripts.nnt \
    --solver sgd --divergence kl --objective full \
    --sgd-lr 1e-2 --sgd-batch-size 4096 --sgd-steps-per-iteration 100 \
    --rank 100 --dim 1000 --n-iter-max 1000
```

Multi-GPU (single process, up to a whole 4-GPU node):

```bash
python3 -m tensormet.scripts.nnt \
    --solver sgd --divergence kl --objective full --n-gpus 4 \
    --sgd-batch-size 4096 \
    --rank 100 --dim 6000 --order 4 --n-iter-max 1000
```

With `--solver mu` (the default) every existing code path and filename is
unchanged.

## What it is

Same non-negative Tucker model as MU; the optimizer changes. Non-negativity is
a parametrization (`softplus` default, or `exp`); the optimizer is Adam; the objective is a
sampled nnz term rescaled to full-tensor scale plus an **exact closed-form
zero-entry term** (`sum(X̂)` for KL via column sums, `‖X̂‖²` for FR via Grams) —
no negative sampling. Errors are normalized exactly like
`distance.kl/fr_compute_errors`, so curves are directly comparable with MU
runs. Algorithm details: `sgd_step.py` module docstring and
`reviews/sgd-tucker-training-signal-2026-07-27.md`.

**Step mapping**: one loop *iteration* = `sgd_steps_per_iteration` optimizer
steps (default 100). All iteration-based knobs (`n_iter_max`,
`rec/sem_check_every`, `checkpoint_saving_steps`, `tol`/`patience`/
`warmup_steps`, `pool_trim_every`) keep their meaning at block granularity.

## Files here

- `sgd_model.py` — what every model shares: the parametrizations
  (`view_from_raw` / `raw_from_view`), rank and shared-factor parsing, and
  `SGDModelBase`, the surface the trainers use.
- `sgd_tt.py` — `SGDTTModel` (Tucker-TT) and `sq_norm`; see "TT and utt".
- `sgd_tucker.py` — the Tucker model (`SGDTuckerModel`) and the two-group
  contraction it uses.
- `sgd_step.py` — what every model shares per step: the deterministic
  `EntryBatcher` (pure function of (seed, step) → resume replays exact
  batches), the objectives, the exact error, memory sizing
  (`resolve_micro_batch`) and `GradStepper`, the per-device step body (flat
  gradient buffer, micro-batching).
- `sgd_trainer.py` — `SGDTrainer(tensor, devices=[...], ...)`, the loop-facing
  surface, for one device or several: `run_block(iteration, log_step)`,
  `materialize()`, `checkpoint_payload()`, `load_payload()`, `sync()`. Single-process
  data parallelism: contiguous NNZ shards + per-shard batchers (seeded
  `random_state*1000+g`; one device keeps `random_state`), a full model +
  optimizer + `GradStepper` per device, a persistent thread-per-GPU pool, **one**
  flattened gradient all-reduce per step and a redundant optimizer step — so
  there is no parameter broadcast at all. One device is the same code without
  the collective, pool and replica check. Device-0 checkpoints. Owns Adam moments,
  raw params and the step counter — state the `UpdateRouting` seam cannot
  express, which is why the loop has a separate SGD solver instead of routing
  kernels.
- `standalone.py` — `sgd_non_negative_tucker` for notebooks (a loop over `SGDTrainer`).
- `sgd_init.py` — `mu_init_for_sgd`: the MU path's SVD inits on a temporary
  CuPy copy of the tensor.
- `collectives.py` — the `Collective` seam: `NcclSingleProcess`
  (`torch.cuda.nccl.all_reduce`, single-process multi-device — no process
  group, no spawn), `HostReduce` (preallocated pinned staging, the fallback
  when NCCL or peer access is missing). The `auto`
  backend probes NCCL with a real 1-element all-reduce **under a 30s watchdog**
  (`TENSORMET_NCCL_PROBE_TIMEOUT`) — comm init's usual failure mode is to block,
  not to raise, and a `try/except` cannot catch that. A timeout is treated as a
  failed probe and falls back to `HostReduce`. The resolved backend is printed
  at construction, so a run that silently measured the host path is visible in
  the log rather than hiding in the timings.

## Multi-GPU knobs and how to set them

Every GPU samples `--sgd-batch-size` entries per step, so the effective batch
is `n_gpus × batch_size` and per-device work is constant in `n_gpus`. The
gradients are summed every step (one all-reduce). Local SGD
(`--sgd-sync-every`), splitting the batch across GPUs (`--sgd-batch-scope`),
CUDA graphs (`--sgd-cuda-graph`), a non-Adam optimizer and the `clamp`
parametrization were removed on 2026-11-12.

| Knob | Default | Trajectory? | What it does |
|---|---|---|---|
| `--sgd-micro-batch` | rank-derived | no | Entries per forward/backward inside one step; gradients accumulate. Exact, memory-only. Rarely binding since the two-group contraction — see below. |
| `--sgd-comm-backend` | `auto` | no | `auto` / `nccl` / `host`. |
| `--sgd-eval-sample` | unset (exact) | no* | nnz per logged error. See "Eval cost" below. |

\* Not in the resume key, but it changes the logged curve and therefore when
`tol`/`patience` fire — same status as `--rec-check-every`.

## Eval cost

The error pass evaluates `x̂` at every nnz, at the same `prod(rank)` flops per
entry a training step pays. So an eval is worth roughly `nnz / (3 · batch_size)`
steps of compute, and relative to a whole block of training:

```
eval / training  ≈  nnz / (3 · sgd_batch_size · sgd_steps_per_iteration · rec_check_every)
```

At the defaults (4096 × 100 × 20) that is ~4× at nnz = 10⁸ and ~40× at
nnz = 10⁹ — i.e. the run is mostly measuring itself. MU does not have this
problem because one MU iteration already touches every nnz; a block of SGD steps
touches `batch_size × steps_per_iteration`.

`--sgd-eval-sample N` evaluates a **fixed** random subset of N entries instead,
rescaling the nnz half to full-tensor scale while the zero-entry half stays
exact — unbiased for the KL numerator and for the squared FR numerator. Fixed
rather than redrawn per eval on purpose: `tol`/`patience` compare *successive*
errors, and a moving subset would inject sampling noise into exactly that
difference. The final reported `final_error` is always computed exactly, over
all nnz, once at the end of the run.

**The two-group contraction is what makes order 4 runnable.**
`predict_entries` splits the modes into two groups, forms the row-wise
Khatri-Rao product of each group's gathered rows, and contracts the two against
the core reshaped to a matrix with a single GEMM. Flops are unchanged
(`batch × prod(rank)` is irreducible for a dense core) but the largest
intermediate drops from `batch × prod(rank)/max(rank)` to
`batch × ~sqrt(prod(rank))` — 16 GB → 160 MB at `--order 4 --rank 100`,
`batch_size=4096` — and the forward becomes three kernels instead of an
N-operand einsum decomposition. The split is chosen at model construction by
`_contraction_plan`, which minimises the intermediate over contiguous split
points and therefore degenerates to the one-mode-at-a-time plan when the ranks
are skewed enough for that to win.

Consequently **micro-batching is now a no-op up to about order 4 / rank 200**:
the rank-derived `--sgd-micro-batch` returns the whole batch there. It still
binds at order 5+, where the intermediate genuinely cannot be brought down, and
the chunk floor of 64 means you should set the knob yourself in that regime. An
explicit setting that would exceed 2 GiB is still refused by name rather than
becoming a CUDA OOM.

**The parametrization is evaluated once per step.** `model.core` is a softplus
over the whole core — 400 MB of traffic at order 4 / rank 100 — so reading it
per micro-batch chunk cost more than the arithmetic the step exists to do.
`SGDTuckerModel.nonneg_views()` materialises the views once (aliased
`shared_factors` modes share one tensor object) and everything downstream takes
them as arguments. When a step *is* split, the chunks differentiate against
detached views with persistent gradient buffers and one chain-rule pass at the
end pushes the total back through softplus — exact, and verified by gradient
parity across chunk counts in `0_tests/test_sgd_solver.ipynb` §6.

**Not implemented: core sharding** (`sgd_core_shard`). Order 4 / rank 100 is
compute-bound with roughly a 50:1
compute-to-zero-entry ratio and a replicated core fits in 80 GB, so plain data
parallelism is sufficient. Splitting the core along the mode-1 rank axis makes
the model an additive sum of G lower-rank Tucker models sharing modes 2…N; the
KL zero-entry term stays perfectly separable (all-reduce one scalar) but
`‖X̂‖²` does not — the mode-1 Gram mixes all mode-1 rank indices, forcing an
O(|core|) all-gather per step, so for FR it trades a core-sized gradient
all-reduce for a core-sized core all-gather: a wash on traffic, a win on
memory. Revisit when rank or order grows past what replication fits.

The cheap half of that idea is still on the table and needs no new
communication: keep the core replicated but shard the *zero-entry* computation
along mode 1, so each device forms its own row block locally and the step
reduces a single scalar. That would remove the one remaining serialization —
device 0 currently carries the whole zero-entry term while the others idle. Worth doing if the sweep shows that imbalance dominating.

If `torch.cuda.nccl.all_reduce` misbehaves (single-process multi-device NCCL is
the less-travelled API), `--sgd-comm-backend host` is a working fallback and
the sweep numbers stay meaningful — it is the same collective, staged through
pinned host memory. From the benchmark driver, set `BENCH_SGD_COMM_BACKEND`.

Observed in the wild: on the VSC wice H100 nodes, `ncclCommInitAll` blocked
indefinitely while the same code ran fine on dodrio A100s (job 61599897,
2026-07-30 — 2.5h of walltime with zero output). Hopper enables NVLS (NVLink
SHARP / CUDA multicast) by default and Ampere has no such path, so
`NCCL_NVLS_ENABLE=0` is the first thing to try on H100; `NCCL_P2P_DISABLE=1` and
`NCCL_SHM_DISABLE=1` are the next two. `NCCL_DEBUG=INFO` shows where init
blocks. The probe watchdog means this now costs 30s and a slower backend rather
than the whole job.

## Integration points

1. `config.py` — `ExperimentConfig.solver` + `sgd_lr`, `sgd_batch_size`,
   `sgd_parametrization`, `sgd_steps_per_iteration`,
   `sgd_warm_start`, `sgd_micro_batch`, `sgd_comm_backend`,
   `sgd_eval_sample`. The trajectory-affecting ones (plus
   `solver`, plus `n_gpus` for SGD) joined the
   resume-compatibility key in `get_resume_state` (an SGD run must never
   splice an MU checkpoint or an SGD run with different optimizer knobs);
   `model_filename()`/`get_resume_state()` pass `solver=` to naming.
2. `naming.py` — `_order_tag(..., solver=)` emits `SGD{order}D` (so SGD and
   MU artifacts can never collide, and resume scans can't cross solvers);
   threaded through `model_stem`/`model_filename`/`candidate_stems`; no
   legacy fallback for SGD stems.
3. `parsing.py` — `--solver` + the `--sgd-*` flags; exp field tuple.
4. `training.py` / `solvers.py` — `SGDSolver` is the SGD half of the solver
   interface the loop drives (`step` = `trainer.run_block`, `checkpoint` = the dict
   payload, `free_cache` = `torch.cuda.empty_cache()`); `check_run_config` rejects
   `decomposition=cp`, `subsample_frac<1`, `max_nnz`, `normalize_factors`,
   `largedim`. Trainer construction (+ warm start or SVD init via
   `sgd_init.mu_init_for_sgd`) is `SGDSolver.__init__`.
   Sem-eval, SimLex, the judge, both patience tracks, best-model selection,
   HPC staging/mirroring and the SIGINT handler are shared with MU.
   `TuckerDecomposition.load_from_disk(solver=)` loads the result.
5. `launch.py` — skips the CuPy import + `tensor_to_sparse("cupy")` and keeps
   the tensorly backend on `"pytorch"` when `solver == "sgd"`; save path uses
   `_as_host`.

Also: `experimental/submit.py` gained
`TIER2_H100_DUAL` (2 GPUs) and `TIER2_H100_QUAD` (whole node, 4 GPUs) — both
still `tasks_per_node=1`, since the sharding happens inside one process.

## Checkpoint / resume semantics

SGD checkpoints are dicts:
`{"solver": "sgd", "iteration", "core", "factors", "raw_state_dict",
"optim_state"}` — `core`/`factors` are host-numpy non-negative views for
CPU-only tooling; the raw/optimizer state makes resume exact (deterministic
batcher + restored Adam moments). The **final model artifact** stays a plain
CPU-numpy `TuckerTensor`, so `TuckerDecomposition.load_from_disk(...,
solver="sgd")`, `inspect_tucker`, and `judge_eval` work unchanged.

`--sgd-warm-start <MU model .pt>` is init, not resume: parameters start at
the MU solution (pushed through the parametrization's inverse), optimizer state and step counter
start fresh.

**Resume and `n_gpus`.** The effective batch is `n_gpus × sgd_batch_size`, so
the trajectory is a function of the device count and `n_gpus` is always part of
the SGD resume-compatibility key: resume at the same `n_gpus`. MU is untouched.
On resume every device's optimizer is restored from the checkpoint, not just
device 0's, because every device steps.

## TT and utt (`--decomposition tt|utt`)

```bash
python3 -m tensormet.scripts.nnt \
    --solver sgd --decomposition utt --tt-rank 100 --rank 100 --order 4 \
    --divergence kl --sgd-parametrization exp --sgd-lr 1e-2 \
    --sgd-warm-start <MU utt model .pt>
```

Same trainers, batcher, collectives and checkpoint format; only the model
changes (`sgd_trainer.make_sgd_model`):

- `sgd_tt.py` — `SGDTTModel`: Tucker factors + TT cores, forward via
  `tt_hybrid.tt_chain` on torch. About `N·R·ρ²` flops per entry (Tucker:
  `prod(R)`), so a step is much cheaper than a Tucker SGD step at order ≥ 4.
- `uniform_tt/utt_sgd.py` — `SGDUTTModel`: `[C, α, ω]` + one
  factor shared by all modes; forward expands to TT cores (`expand_uniform`).

Zero-entry terms are exact closed forms in fp64: Σ_all x̂ is the chain on factor
column sums; ‖X̂‖² is the chain of X̂ against itself joined by the factor Grams
(`sgd_tt.sq_norm`, `2ρ³R + ρ²R²` per site). So both divergences and both
objectives work, unlike MU-TT (KL, full only).

**Gauge.** A chain model has scale directions that leave x̂ unchanged (factor
column vs. core slice, the two sides of a bond, per-core scalars whose product
is 1; for utt, C vs. α through the Perron root). Adam drifts along them and the
N-fold product leaves fp32 range — the order-6 MU failure mode. Every
`--sgd-regauge-every` iterations (default 1; 0 = off) `model.regauge_()` resets
the gauge exactly: ℓ1 factor columns, balanced bonds, equal core masses (TT);
ℓ1 columns + MU's Perron gauge (utt). Prefer `--sgd-parametrization exp`: the
reset is then an additive shift of the raw parameters and the log-space
gradient is gauge-invariant, so Adam's moments stay valid across it. Under
softplus they are only approximately right after a reset, and core entries
in softplus's linear range take additive rather than relative steps.

**Init.** `random` uses the same draws as the MU init
(`initialize_tucker_tt` / `initialize_uniform_tt`) for the same seed. SVD inits
(`--init svd_stream`, `svd`, ...) run exactly as on the MU path, on a temporary
CuPy copy of the tensor (`sgd_init.mu_init_for_sgd`); the result enters the model
like a warm start and the CuPy memory is freed before training. A Tucker SVD
init's core is first rescaled so that Σ_all x̂ = Σx: the projected core starts
orders of magnitude too large, which MU's first step absorbs but Adam does not
(TT/utt inits already rescale). For utt, random
init is known to stall in a frequency-only state (memory notes, 2026-10-29), so
use an SVD init or a warm start.

**Floor.** The KL log term uses `x̂ + epsilon` (absolute), not MU-TT's floor
relative to the mean entry. Fine up to about order 5; at order 6 the mean
entry approaches 1e-12 and this would need the relative floor.

## Not implemented (rejected with clear errors)

`decomposition=cp` × sgd; `subsample_frac<1`/`max_nnz` (SGD is already
minibatch — use `--sgd-batch-size`); `normalize_factors`; `largedim`;
`--utt-core-power` (an MU knob).

## DDP escape hatch (documented, not built)

`collectives.Collective` is the seam. A `TorchDistributed` implementation
backed by `init_process_group("nccl")` slots in unchanged — the trainer only
ever asks it to sum G buffers in place. What a real DDP move would
*additionally* require, and why it is deferred:

- a rank-aware tee logger (only rank 0 writes), SIGINT checkpoint handler, LLM
  judge invocation, and checkpoint writer — the four things the single-process
  design exists to keep unambiguous;
- Slurm profiles with `tasks_per_node=G` and a `torchrun`/`srun` launcher,
  replacing the in-process `launch_nnt_decomposition` call in `submit.py`;
- `select_gpu`'s pre-torch `CUDA_VISIBLE_DEVICES` remapping (`utils.py`) would
  have to become `LOCAL_RANK`-aware.

Up to 4 GPUs on one node, none of that is worth paying for.

## Validation

`0_tests/test_sgd_solver.ipynb` (units, parity, resume-exactness, e2e smoke)
and `0_tests/test_sgd_multigpu.ipynb` (gradient parity, micro-batch parity,
replica-drift, resume exactness, scaling sweep, order-4 smoke) — run on the
GPU box.

**Timing hygiene comes first.** `run_block` deliberately returns with work
still queued on non-log steps, so any benchmark must bracket the timed region
with `trainer.sync()` (both trainers expose it) and record
`torch.cuda.max_memory_allocated(dev)` per device. Timings taken without it
are not orderable — the pre-Phase-1 sweep reported B=128 as *slower* than
B=1024, which is not a physical ordering, and no scaling claim should be made
against numbers gathered that way.
