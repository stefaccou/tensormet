# Tucker-TT hybrid (EXPERIMENTAL)

Tucker factor matrices, tensor-train core. The point is the order-5 wall: a
rank-100 order-5 Tucker core is `100^5` floats ≈ 40 GB, while its TT form at
bond 100 is ~12 MB. Storage becomes linear in the order instead of exponential,
and the per-NNZ cost drops from `O(R^N)` to `O(N ρ² R)` per pass — a full core
sweep is `N` such passes (one per site, see "Multi-GPU" below), so `O(N² ρ² R)`.

```
G[r_0 … r_{N-1}] = C_0[:, r_0, :] · C_1[:, r_1, :] · … · C_{N-1}[:, r_{N-1}, :]
X̂[i_0 … i_{N-1}] = Σ_r G[r] · Π_n A_n[i_n, r_n]
```

`C_k` has shape `(ρ_k, R_k, ρ_{k+1})` with `ρ_0 = ρ_N = 1`; `A_n` is the same
`(I_n, R_n)` Tucker factor as before. Everything stays nonnegative, cores
included, so each block update is a Lee & Seung KL/Poisson MU step and a sweep
is monotone.

## Usage

```bash
# existing nnt launcher, two flags:
... --decomposition tt --tt-rank 100 --divergence kl --rank 100 ...
```

Artifacts are named `..._TT{tt_rank}b{order}D_...` (vs `..._{order}D_...` for
Tucker) and stored as `TuckerTTTensor` payloads `(tt_cores, factors)`. The `b`
fragment is the bond dimension: it fixes the TT core shapes, so two `--tt-rank`
values are two different models and get two different artifact families. Load
with `TuckerTTDecomposition.load_from_disk(..., tt_rank=...)`.

## What changes for interpretation

Nothing at the factor level: a latent dimension of a role still means what it
meant, and `get_top_words_for_dimension`, `get_most_similar_elements`,
SimLex and the dimension-consistency judge are the same code on the same
matrices. What changes is the core:

- Tucker's `G[p,q,r,…]` is a directly readable table of joint topic
  interactions. The hybrid has no such object — you query it (any single entry
  costs `O(N ρ² R)`) but you cannot scan it.
- A bond index `ρ_k` is not a topic of any role. It is a summary of everything
  left of cut `k`, handed to everything right of it, so the bond dimensions
  (`TuckerTTDecomposition.bond_dims()`) measure how much dependency crosses
  each cut. A small `ρ_k` says the two halves are nearly independent given a
  `ρ_k`-dimensional bottleneck.
- Long-range interaction is mediated through the intervening bonds rather than
  represented directly. Since the populated values are specific interaction
  information (`sii*`), check the singular-value decay of the balanced-cut
  unfolding before trusting a small `tt_rank` — that spectrum *is* the exact TT
  rank at that cut.

The dense core is still available via the `core` property (reconstructed from
the chain, refused above a size guard), so anything written against Tucker's
core keeps working at orders where it fits.

## Package contents

| file                   | contents |
|------------------------|----------|
| `tt_chain.py`          | the chain algebra: site matrices, left/right environments, `site_grad`, `contract`, `to_dense_core`, `bond_dims`. Backend-generic (`xp` = cupy / numpy / torch) — only two-operand einsums, which all three spell alike |
| `tt_ops.py`            | KL MU kernels: factor update, sequential core sweep, error, init |
| `tt_sharded.py`        | multi-GPU (`n_gpus > 1`) counterparts of those kernels: per-shard NNZ workers + their orchestrators |
| `tt_routing.py`        | `get_tt_update_routing_step` / `get_sharded_tt_update_routing_step` — builds the `UpdateRouting` the Tucker loop consumes |
| `tt_decomposition.py`  | `TuckerTTDecomposition` (eval/inspection wrapper) and `TuckerTTTensor` (the payload container) |

Design invariants:

- The TT cores travel through the training loop in the `core` variable, exactly
  as CP's λ does, so `SparseTupleTensor.non_negative_tucker_with_similarity` is
  reused, not forked.
- Both MU denominators are sums over **all** entries and both are closed forms:
  run the same chain with the factor column sums in place of the gathered
  latents. Nothing here ever allocates an `R^N` object — not even the KL error
  (Tucker's needs a dense reconstruction on CPU for the zero-entry term).
- The factor update ℓ1-normalizes its columns and absorbs the scale into that
  mode's TT core. This is an exact reparametrization and it is unconditional
  (independent of `normalize_factors`): an N-fold chain product drifts in scale
  otherwise. The loop's `tucker_normalize` is skipped for this family.
- The core sweep visits sites sequentially, so each site update sees the
  previous ones. That is what makes the sweep a genuine block MU (and therefore
  monotone); the price is one NNZ pass per site — and, sharded, one reduce per
  site (see below).

## Multi-GPU

`--n-gpus > 1` routes through `tt_sharded.py`, which mirrors
`experimental/CP/cp_sharded.py`, which mirrors `sharded_sparse.py`:
NNZ-partitioned shards, one thread per GPU, partials summed on the CPU,
finalize on the primary. Shard construction, the persistent pool, the cuBLAS
warm-up, the subsample window and `trim_pools` all come from
`ShardedSparseTensor` unchanged.

Only the NNZ accumulations cross devices. Every TT denominator is a closed form
over the factor column sums, so — unlike Tucker's sharded core update — nothing
but the numerators is reduced:

| reduced across shards | payload | stays on the primary |
|---|---|---|
| factor numerator | `(I_mode, R_mode)` | both denominators, `Σ_all x̂` |
| core numerator, per site | `(ρ_k, R_k, ρ_{k+1})` | the MU divide, ε-clip, ℓ1 rescale |
| KL error | 3 scalars | the in-place core writes |

The cost that is specific to this family: the core sweep is **N reduce rounds
per iteration**, one per site, because site `k` must see site `k-1`'s write.
CP pays none (its λ update is a passthrough) and Tucker pays one. Updating all
sites from a single pass would cost one barrier instead of N, but each site
would then be updated against stale neighbours and the sweep would stop being
monotone — the MM guarantee is the correctness oracle here, so the sequential
sweep is kept. Between sites only the core just written is re-broadcast, so the
host traffic per sweep is `Σ_k |C_k|`, not `N · Σ_k |C_k|`.

As on the Tucker and CP sharded paths, the logged error is evaluated on this
iteration's subsample window and reweighted per shard, rather than on the full
NNZ as the single-GPU kernel does. `Σ_all x̂` stays analytic and exact either
way.

## Not implemented

Explicit `NotImplementedError` in every case:

- **Frobenius divergence.** The FR factor denominator is `Σ_all x̂ · ∂x̂/∂A`,
  which needs a doubled, Gram-weighted chain (transfer matrices with bond `ρ²`)
  rather than the single chain everything here is built on. KL is what the
  pipeline uses; FR is the obvious next kernel if it is wanted.
- **`objective="masked"`**, matching CP's current status.
- **`solver="sgd"`.**

Known rough edge: with linked (`shared_factors`) modes, the loop's
`factors[other] = factors[mode]` copy does not rescale `tt_cores[other]`, so the
scale absorbed at mode `mode` is only compensated when mode `other` is itself
updated. Same INDSCAL-heuristic approximation the Tucker path already makes,
one step coarser.

## Integration seams in the main package (and how to revert)

Every change outside this directory is additive and gated on
`decomposition == "tt"`; with the default (`"tucker"`) all code paths and all
artifact filenames are unchanged. To revert, delete this directory and undo:

1. `config.py` — `ExperimentConfig.tt_rank`; `tt_rank` in `get_resume_state()`'s
   `is_compatible` (a different bond dimension means different core shapes)
2. `naming.py` — `_DECOMPOSITION_TAG` (`'TT'`) and the `tt_rank` kwarg on
   `_order_tag` / `model_stem` / `model_filename` / `candidate_stems`;
   `_DECOMPOSITION_TAG` also gates `candidate_stems`' legacy fallback
3. `routing.py` — `get_update_routing_step` delegates to `tt_routing` for `"tt"`
4. `parsing.py` — `"tt"` in `--decomposition`, new `--tt-rank`
5. `sparse_ops.py` — `with_core=False` on `initialize_nonnegative_tucker` and
   the two SVD init helpers, so factor init can skip the dense-core step (that
   step alone would OOM at order 5)
6. `tucker_tensor.py` — `_as_host` handles list input; `_is_tt` swap points
   (config unpack, `TensorModel`, resume unpack + core-shape check, init
   dispatch, guard rails, `NNZGroupingCache` / `precompute_largedim_batches`
   gating, `_tt_batch_nnz` hoist, `best_core` deep copy, banner,
   sharded-routing branch, `tucker_normalize` skip, `TuckerTTDecomposition` at
   sem checks, `_as_host(core)` at the two checkpoint sites)
7. `launch.py` — container-aware final save + notify text
8. `sharded_sparse.py` — `ShardedSparseTensor.tt_factor_update` /
   `tt_core_update` / `tt_compute_errors`, three lazy delegates alongside the
   CP ones (nothing else in that module knows about TT)
9. `inspect_tucker.py` — `_TT_STEM_RE`, `_tt_rank_from_stem`, and `"tt"` from
   `_decomp_from_stem`, so the run browser labels TT runs with their bond

## Validation

`0_tests/2026-08-19-test_tt_hybrid.ipynb` (GPU required for the kernel cells): chain
algebra against dense `np.einsum`, MU steps against a dense NumPy reference,
per-sweep KL monotonicity (the MM guarantee is the correctness oracle for MU
kernels), agreement with the Tucker pipeline when the bond dimension is large
enough to be exact, `TuckerTTDecomposition` contract checks, and CPU-only
naming/config assertions.

End-to-end smoke, on the GPU box:

```bash
... --decomposition tt --tt-rank 50 --divergence kl --dim 1000 --rank 50 --iterations 200
```

then verify: `TT50b3D` in artifact names, checkpoints resume (and a Tucker or CP
checkpoint of the same dims/rank, or a TT one at a different `--tt-rank`, is
*refused*), `runs.jsonl` records `"decomposition": "tt"` and `"tt_rank"`, and
sem-eval / SimLex / judge scores populate.

`0_tests/2026-08-20-test_tt_multigpu.ipynb` (needs >= 2 CUDA devices) covers the
sharded path against the single-GPU kernels: with `subsample_frac=1.0` a sharded
iteration is an exact NNZ partition of the same sums, so `n_gpus=2` must
reproduce `n_gpus=1` to floating-point tolerance — per-site core numerators,
factor numerators, the in-place ℓ1 rescale, the error, a 5-sweep trajectory
through the real `UpdateRouting` seam, and the unbiasedness of the subsampled
error. End to end, add `--n-gpus 2` to the smoke command above and confirm the
banner reads `tucker-tt (nnz-streaming, bonds=[...], sharded×2)`.
