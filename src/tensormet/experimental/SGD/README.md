# Experimental SGD Tucker solver (`--solver sgd`)

Torch-native minibatch SGD/Adam alternative to the multiplicative-update (MU)
pipeline, selectable through the standard argument machinery:

```bash
python3 -m tensormet.scripts.nnt \
    --solver sgd --divergence kl --objective full \
    --sgd-lr 1e-2 --sgd-batch-size 4096 --sgd-steps-per-iteration 100 \
    --rank 100 --dim 1000 --n-iter-max 1000
```

With `--solver mu` (the default) every existing code path and filename is
byte-identical. Template for this layout: `experimental/CP/` +
`reviews/CP_IMPLEMENTATION_PLAN.md`.

## What it is

Same non-negative Tucker model as MU; the optimizer changes. Non-negativity is
a parametrization (`softplus` default, `clamp` optional); the objective is a
sampled nnz term rescaled to full-tensor scale plus an **exact closed-form
zero-entry term** (`sum(X̂)` for KL via column sums, `‖X̂‖²` for FR via Grams) —
no negative sampling. Errors are normalized exactly like
`distance.kl/fr_compute_errors`, so curves are directly comparable with MU
runs. Algorithm details: `sgd_tucker.py` module docstring and
`reviews/sgd-tucker-training-signal-2026-07-27.md`.

**Step mapping**: one loop *iteration* = `sgd_steps_per_iteration` optimizer
steps (default 100). All iteration-based knobs (`n_iter_max`,
`rec/sem_check_every`, `checkpoint_saving_steps`, `tol`/`patience`/
`warmup_steps`, `pool_trim_every`) keep their meaning at block granularity.

## Files here

- `sgd_tucker.py` — model (`SGDTuckerModel`), deterministic `EntryBatcher`
  (pure function of (seed, step) → resume replays exact batches), objectives,
  exact error, and the standalone `sgd_non_negative_tucker` for notebooks.
  (The old `experimental/sgd_tucker.py` is a re-export shim.)
- `sgd_trainer.py` — `SGDTrainer`, the loop-facing surface:
  `run_block(iteration, log_step)`, `materialize()`, `checkpoint_payload()`,
  `load_payload()`. Owns Adam moments, raw params, and the step counter —
  state the `UpdateRouting` seam cannot express, which is why the loop
  branches on the solver instead of routing kernels.
- `sharded_sgd.py` — `ShardedSGDTrainer` for `--n_gpus > 1`: single-process
  data parallelism (contiguous NNZ shards + per-shard batchers seeded
  `random_state*1000+g`, model replicas, grad all-reduce onto device 0,
  param broadcast back). Master-only checkpoints → resume works across
  different `n_gpus`. Honest caveat: the model is tiny, so the win is data
  sharding + effective batch, not linear step-time scaling.

## Integration seams (guarded; revert = delete this dir + these hunks)

1. `config.py` — `ExperimentConfig.solver` + `sgd_lr`, `sgd_batch_size`,
   `sgd_optimizer`, `sgd_parametrization`, `sgd_steps_per_iteration`,
   `sgd_warm_start`; all of them (plus `solver`) joined the
   resume-compatibility key in `get_resume_state` (an SGD run must never
   splice an MU checkpoint or an SGD run with different optimizer knobs);
   `model_filename()`/`get_resume_state()` pass `solver=` to naming.
2. `naming.py` — `_order_tag(..., solver=)` emits `SGD{order}D` (so SGD and
   MU artifacts can never collide, and resume scans can't cross solvers);
   threaded through `model_stem`/`model_filename`/`candidate_stems`; no
   legacy fallback for SGD stems.
3. `parsing.py` — `--solver` + the six `--sgd-*` flags; exp field tuple.
4. `tucker_tensor.py` — `_is_sgd` branch in
   `non_negative_tucker_with_similarity`: torch-COO input check, guard rails
   (rejects `decomposition=cp`, `subsample_frac<1`, `max_nnz`, SVD init,
   `normalize_factors`, `largedim`), trainer construction (+ warm start),
   the update+error section swapped for `trainer.run_block`, dict checkpoint
   payloads at the periodic/SIGINT sites, `torch.cuda.empty_cache()` instead
   of CuPy pool trims, solver-gated tensorly backend flips, `_as_host` at the
   shared save/eval sites, `TuckerDecomposition.load_from_disk(solver=)`.
   Sem-eval, SimLex, the judge, both patience tracks, best-model selection,
   HPC staging/mirroring and the SIGINT handler are reused unchanged.
5. `launch.py` — skips the CuPy import + `tensor_to_sparse("cupy")` and keeps
   the tensorly backend on `"pytorch"` when `solver == "sgd"`; save path uses
   `_as_host`.

Also: `experimental/__init__.py` re-exports; `experimental/submit.py` gained
`TIER2_H100_DUAL` for 2-GPU runs.

## Checkpoint / resume semantics

SGD checkpoints are dicts:
`{"solver": "sgd", "iteration", "core", "factors", "raw_state_dict",
"optim_state"}` — `core`/`factors` are host-numpy non-negative views for
CPU-only tooling; the raw/optimizer state makes resume exact (deterministic
batcher + restored Adam moments). The **final model artifact** stays a plain
CPU-numpy `TuckerTensor`, so `TuckerDecomposition.load_from_disk(...,
solver="sgd")`, `inspect_tucker`, and `judge_eval` work unchanged.

`--sgd-warm-start <MU model .pt>` is init, not resume: parameters start at
the MU solution (pushed through softplus⁻¹), optimizer state and step counter
start fresh.

## Not implemented (rejected with clear errors)

`decomposition=cp` × sgd; `subsample_frac<1`/`max_nnz` (SGD is already
minibatch — use `--sgd-batch-size`); `init=svd` (CuPy routine — use
warm start); `normalize_factors`; `largedim`.

## Validation

`0_tests/test_sgd_solver.ipynb` (units, parity, resume-exactness, e2e smoke)
and `0_tests/test_sgd_multigpu.ipynb` (gradient parity, trajectory parity,
step-time benchmark) — run on the GPU box.
