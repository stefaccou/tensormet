# Nonnegative CP decomposition (EXPERIMENTAL)

Implementation of `reviews/CP_IMPLEMENTATION_PLAN.md`: a nonnegative CP
(CANDECOMP/PARAFAC) decomposition as an alternative to the Tucker pipeline,
for both divergences (`fr` = Frobenius MU [Welling & Weber 2001; Lee & Seung
2001], `kl` = generalized KL / Poisson, CP-APR MU [Chi & Kolda 2012]),
reusing the sparse-tensor data pipeline, the training-loop scaffolding and
the evaluation stack.

## Usage

```bash
# existing nnt launcher, one new flag:
... --decomposition cp --divergence kl --rank 100 ...
```

Optional CP knobs: `--cp-inner-iters N` (CP-APR maxinner, default 1),
`--cp-scooch-kappa K` (inadmissible-zero nudge, default 0 = off).

Two things to know about `--cp-inner-iters`, both KL-only (FR's numerator is
loop-invariant, so inner iterations do not apply to it):

* **Under subsampling all inner iterations see the same NNZ window** — the
  window is keyed off the *outer* iteration. So `N=3` at
  `--subsample-frac 0.025` takes three MU steps against the same 2.5% of the
  data, sharpening the fit to that sample rather than to the tensor. Past ~2–3
  the extra steps buy progressively less.
* **On multi-GPU each inner iteration costs a full reduce**, because Φ depends
  on `B`. The bytes are negligible at production nnz (the reduce is a fixed
  `(I_mode, R)`), but the barrier latency is not zero.

Model artifacts are named `..._CP{order}D_...` (vs `..._{order}D_...` for
Tucker) and stored as `tensorly.cp_tensor.CPTensor` payloads
`(weights λ, factors)`. Load with `CPDecomposition.load_from_disk(...)`.

Multi-GPU (`n_gpus > 1`) shards the NNZ across devices; see `cp_sharded.py`.
Not yet supported (explicit `NotImplementedError`): the masked/completion
objective, and `solver=sgd` with `decomposition=cp`.

## Package contents

| file                  | contents |
|-----------------------|----------|
| `cp_ops.py`           | NNZ-streaming kernels: shared primitives (`cp_values_at_nnz`, `cp_weighted_mttkrp`), FR/KL factor MU updates (λ updated in place, cp_normalize semantics), λ "core-slot" callables, closed-form error kernels, initialization. The factor updates are compositions of an NNZ-dependent accumulation (`_cp_mttkrp_from_idxs`, `_cp_kl_phi_from_idxs`) and an NNZ-free remainder (`_cp_*_mu_step`, `_cp_absorb_into_weights`) so the two halves can be driven separately |
| `cp_routing.py`       | `get_cp_update_routing_step` / `get_sharded_cp_update_routing_step` — build the `UpdateRouting` the Tucker loop consumes |
| `cp_sharded.py`       | Multi-GPU: per-shard numerator/error workers + CPU-reduce orchestrators, mirroring `sharded_sparse.py`. Only the NNZ-dependent halves are sharded; Γ, σ, Σx̂ and the MU/normalize tail stay on the primary. No sharded core update exists — λ's "core slot" is a passthrough |
| `cp_decomposition.py` | `CPDecomposition` — inference/eval wrapper, API-compatible with the `TuckerDecomposition` subset used by `evaluate_sample`, SimLex and the dimension-consistency judge |

Design invariant: the CP weight vector λ travels through the training loop in
the `core` variable, so `SparseTupleTensor.non_negative_tucker_with_similarity`
is **reused, not forked**. There is no dense-Z kernel family: streaming is
both the memory-safe and the fast path (transients are O(batch_nnz·R), never
R^N), so Tucker's dense/largedim routing split does not apply.

## Integration seams in the main package (and how to revert)

Every change outside this directory is additive and gated on
`decomposition == "cp"`; with the default (`"tucker"`) all code paths and all
artifact filenames are byte-identical to before. To revert the feature
entirely, delete this directory and undo these guarded hunks:

1. `config.py`
   - `ExperimentConfig.decomposition / cp_inner_iters / cp_scooch_kappa` fields
   - `model_filename()` / `get_resume_state()` pass `decomposition=` to naming
   - `get_resume_state()` `is_compatible` gains the decomposition equality check
     (correctness-critical: prevents a CP run silently resuming a Tucker
     checkpoint of identical dims/rank, and vice versa)
2. `naming.py` — `_order_tag()` helper; `model_stem` / `model_filename` /
   `candidate_stems` accept `decomposition=` (default emits the old names)
3. `routing.py` — `get_update_routing_step` accepts `decomposition=` and
   lazily delegates to `cp_routing` for `"cp"`
4. `parsing.py` — `--decomposition`, `--cp-inner-iters`, `--cp-scooch-kappa`
5. `tucker_tensor.py` (`non_negative_tucker_with_similarity`) — guarded swap
   points: config unpack + validation, `validate_cp_rank` vs
   `validate_tucker_rank`, init dispatch, resumed-checkpoint λ-shape check, CP
   guard rails (masked / SGD), Tucker-only gating of `NNZGroupingCache` /
   `precompute_largedim_batches`, routing kwargs, the sharded-CP routing
   override, `tucker_normalize` skip, `CPDecomposition` at sem checks, the
   routing-path banner, and the container sites use `TensorModel`
   (= `TuckerTensor` unless CP)
6. `launch.py` — container-aware final save + decomposition-aware notify text
7. `judge.py` — reads the rank from `factors[i].shape[1]` rather than
   `core.shape[i]`, identical for Tucker and O(1) for CP
8. `sharded_sparse.py` — `ShardedSparseTensor.cp_factor_update` /
   `cp_compute_errors`, two thin delegates that lazily import `cp_sharded`
9. `scripts/benchmarking.sh` — `--solver cp` and its `BENCH_CP_*` knobs

## Validation

`0_tests/test_cp_kernels.ipynb` (GPU required): primitives vs dense
`np.einsum`, per-sweep objective monotonicity for both divergences (the MM
guarantee is the correctness oracle for MU kernels), dense NumPy CP-APR /
FR-MU reference cross-checks, closed-form Σx̂ vs dense reconstruction,
CPDecomposition contract checks, and CPU-only naming/config assertions.

`0_tests/compare_cp_implementations.ipynb` (GPU required; `pyttb` optional):
same-init trajectory comparison against tensorly `non_negative_parafac` (MU)
and `non_negative_parafac_hals`, pyttb `cp_apr` (canonical CP-APR, KL side),
and a dense NumPy reference of our exact conventions — plus planted-factor
recovery (factor match score) and runtime-per-sweep scaling, all scored with
one shared metric.

End-to-end smoke (plan Phase 4, run on the GPU box):

```bash
... --decomposition cp --divergence kl --dim 1000 --rank 50 --iterations 200
... --decomposition cp --divergence fr --dim 1000 --rank 50 --iterations 200
```

then verify: `CP3D` in artifact names, checkpoints resume (and a Tucker
checkpoint of the same dims/rank is *refused*), `runs.jsonl` records
`"decomposition": "cp"`, and sem-eval / SimLex / judge scores populate.
