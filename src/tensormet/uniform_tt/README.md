# Shared-core ("uniform") Tucker-TT — `--decomposition utt`

The Tucker-TT hybrid (`tt_hybrid/`) with one core shared by every
position instead of one core per position. Factors are shared too (all modes).

```
M_r  = C[:, r, :]              (ρ×ρ)  one operator per latent dimension
S(i) = Σ_r A[i, r] · M_r       (ρ×ρ)  one operator per word, the same at every position
x̂(i_0 … i_{N-1}) = αᵀ · S(i_0) ⋯ S(i_{N-1}) · ω
```

Parameters: `A` (V×R), `C` (ρ×R×ρ), start/end vectors `α`, `ω` (ρ). Nothing depends
on the order N.

## Usage

```bash
... --decomposition utt --tt-rank 50 --rank 50 --divergence kl
    [--utt-core-power 0.25]   # core step exponent; default 1 (undamped), 1/N = monotone
    [--utt-core-power auto]   # exponent chosen per iteration by backtracking (see below)
```

`shared_factors` must tie all modes (the default `"all"`). Artifacts are named
`..._UTT{ρ}b{order}D_...`. The payload is the plain tuple `([C, α, ω], factors)`.
Load with:

```python
UniformTTDecomposition.load_from_disk(..., tt_rank=50)
TuckerDecomposition.load_from_disk(..., decomposition="utt", tt_rank=50)
```

## What the model is

A nonnegative weighted finite automaton (a "uniform MPS"). An HMM is the special case
R = ρ, `C[a,r,b] = T[a,b]·δ(r=b)`. Here each latent dimension has its own transition
matrix, and a word mixes them via its row of A (the "multiplicative / second-order RNN"
form).

- **Not autoregressive** in the strict sense. There is no per-step normalization, and the
  chain reads right-to-left equally well. Next-word scores are the excluded-role vector
  at the last position (`predict_words`).
- **Expressiveness.** At equal ρ it is a restriction of the per-position TT. With bond
  N·ρ it contains it (a "position counter" state space with block-superdiagonal
  `M_r`). Compare at equal ρ, at equal parameter count (ρ ≈ √N·ρ_TT), and at ρ = N·ρ_TT.
- **Any length.** One trained model scores tuples of any length: `score_sequence`,
  `role_vector(tokens, position)`, `predict_words`.
- **Interpretation.** The core has no positions or roles. What you inspect is
  `operators()[r]` (the state transition dimension r causes), `boundaries()` and
  `word_operator(w)`.
- **Padding.** `bos-eos` (n−1 `<s>`) suits an automaton, because the start state loops on
  `<s>`. With `padded` (one `<s>` at position 0 only) the model has to spend states on
  "`<s>` only at the start".
- **Values are association scores**, not counts. The HMM reading (α = stationary state
  distribution, ω = 1, consistent marginals across orders) is therefore a structural
  prior, not an identity.

## Updates (KL MU), per iteration

| block | degree of x̂ in it | step |
|---|---|---|
| A (shared) | N | pooled tied update, undamped (as in TT). The ℓ1 scale goes into C **once** |
| C | N | pooled over the N sites, undamped (exponent γ = `--utt-core-power`, default 1; `auto` = backtracking) |
| (α, ω) | 2 | joint step, undamped, then the exact overall scale Σx / Σ_all x̂ |

All steps are undamped by default, like TT's tied factor update. They have the right
fixed point but no monotonicity guarantee. For a block that x̂ is homogeneous of degree d
in, exponent 1/d is the MM-safe step: it bounds x̂ by weighted AM-GM and the log term by
Jensen. That means 1/N for C (`--utt-core-power`) and 1/2 for (α, ω) (`boundary_power`
in the kernels). Those damped steps were the first default, but at production scale they
barely moved the parameters. The scale step restores `Σ_all x̂ = Σ x` after every
iteration.

### Adaptive core step (`--utt-core-power auto`)

**Why.** The undamped core step can overshoot once the model is close to a minimum. The
first production run (4-gram, ρ = 100, `--subsample-frac 0.025`, svd_stream init) showed
this. The error fell smoothly from 13.76 to 11.30 by iteration 180. After that it kept
spiking to about 13.1 and recovering, with no further progress. SimLex, which depends only
on A, was unaffected. A fixed exponent cannot handle both phases: γ = 1 is needed early,
and 1/N is safe but barely moves. The iteration where trouble starts depends on the data,
so a fixed schedule would have to guess it.

**Rule.** Every iteration, the core step is judged on that iteration's NNZ window:

1. The core pass also returns `Σ x log x̂` at the current state (free: x̂ is already
   built for the numerator).
2. Apply the step with the current γ. The boundary pass, which runs next anyway, returns
   `Σ x log x̂` at the new state.
3. Compare the KL at the best overall scale, in the units of the logged relative error:

   ```
   J = log Σ_all x̂ − (Σ_nnz x log x̂) / Σx        (+ a constant)
   ```

   `Σ_all x̂` is closed form. Using the best scale means a pure scale overshoot is not
   held against the core, because the boundary step applies that exact scale right after.
4. If J did not go up, keep the step. If it was the first try, raise γ for the next
   iteration: γ ← min(1, 1.05·γ).
5. If J went up, undo the step, halve γ and go back to step 2. The core numerator is
   reused, so each retry costs one boundary pass. Once γ reaches the MM-safe 1/N, the
   step is kept whatever J does. For N = 4 there are at most 2 retries (1 → 0.5 → 0.25).
6. The boundary (α, ω) step then runs as usual on the accepted state, using the numerator
   from the last boundary pass.

The comparison has no sampling noise, because before and after are computed on the same
entries. That holds under subsampling and across shards: `Σ x log x̂` is linear in x,
like the numerators, so the rescaled window values are correct for it. Each rejection
prints one line:

```
  [UTT] core step at γ=1 raised the window KL by 3.1e-02; undone, retrying at γ=0.5
```

**Cost.** None when steps are accepted (the two sums come from passes that run anyway),
and one boundary pass per retry.

**Scope and limits.**
- Only the core step is controlled. The A step is the same undamped tied update TT uses
  stably. The (α, ω) step is degree 2 and ends with the exact scale step; checking it
  would need one extra pass every iteration. If spikes persist while γ stays near 1 and
  nothing is rejected, the A step is the next suspect.
- γ lives in one `AdaptiveCorePower` object per run (created in `tucker_tensor.py`), so
  it carries across iterations but not across a resume: a resumed run starts again at
  γ = 1. Its `history` holds (γ tried, ΔJ, kept) for every try.
- The run name does not record the exponent (fixed or `auto`), so runs that differ only in
  `--utt-core-power` share artifact names. Use `--name` to keep them apart.
- Constants (in the class): start 1, grow 1.05, shrink 0.5, tolerance 1e-6 on ΔJ
  (float32 roundoff).

**Same machinery as the per-position TT.** A uniform TT is an ordinary TT whose middle
cores are all C, with αᵀC and C·ω at the ends (`expand_uniform`). Every NNZ pass expands
the model and runs the `tt_hybrid` batch operations on it: `sites`, `left_envs` /
`right_envs`, `site_grad`, `_scatter_add`, the site GEMM, `_run_nnz_batches`, and TT's
batch estimator. So:

- The factor pass is the body of `tt_kl_tied_factor_update`, at the same per-entry cost.
- The core numerator sums TT's site GEMM over the middle sites. The two end sites see C
  through α and ω, which costs two cheap (b×R)·(b×ρ) GEMMs.
- The α/ω numerators are those same two GEMMs, contracted with C after the pass.

An earlier version used a precomputed per-word operator table and a per-word gradient
table instead. That was dropped because at large vocabularies the factor update ran slower
than TT's.

**Gauge (exact reparametrizations).** After every A or C step:
- C is divided by the Perron root λ of `T = Σ_r s_r M_r`, and α is multiplied by λᴺ.
- α and ω are balanced to equal ℓ1 mass.

This keeps the chain in range at any order.

Per iteration: 3 NNZ passes (factor, core, boundaries), plus the error pass on log steps,
plus one boundary pass per rejected core step under `auto`.
The per-position TT needs N + 1. Sharded, that is 3 reduces instead of N + 1. The factor
pass costs the same as TT's, and the core update costs about 2 passes instead of N.
Notebook cell 9 times both.

`--solver sgd` fits the same model with Adam instead of MU (KL or FR, full or masked;
`--utt-core-power` is rejected there). Artifacts are `..._SGDUTT{ρ}b{order}D_...`, same
payload. See `sgd/README.md`, "TT and utt".

## Files

| file | contents |
|---|---|
| `utt_chain.py` | `expand_uniform` (compact → standard TT list), transfer matrix, chain total αᵀTᴺω, column-sum sweeps, the gauge scalars (shared by MU and SGD) |
| `utt_ops.py` | NNZ passes (TT machinery on the expanded chain), closed-form denominators, the steps, gauge, single-GPU kernels, init |
| `utt_sharded.py` | multi-GPU workers/orchestrators (reads the `ShardedSparseTensor` directly) |
| `utt_routing.py` | `UpdateRouting` for single- and multi-GPU |
| `utt_decomposition.py` | `UniformTTDecomposition(TuckerTTDecomposition)`: expands the compact model so every inherited eval method works; adds any-length scoring and operator inspection |
| `utt_sgd.py` | `SGDUTTModel` for `--solver sgd` (torch/autograd; see `sgd/README.md`, "TT and utt") |

## Integration seams in the main package

All are gated on `"utt"`; the defaults are unchanged:

- `routing.py`: lazy `utt` branch and the `utt_core_power` kwarg.
- `naming.py`: `"utt": "UTT"`, stem `UTT{tt_rank}b{order}D`.
- `parsing.py`: `utt` choice and `--utt-core-power` (a float or `auto`).
- `config.py`: `ExperimentConfig.utt_core_power`.
- `tucker_tensor.py` also turns `"auto"` into one `AdaptiveCorePower` for the run.
- `tucker_tensor.py`: `_is_utt`. `_is_tt` now covers `utt`, so the list-core handling
  is shared. There are `utt` branches at the resume shape check, init, the all-modes-tied
  guard, batch sizing, banner, sharded routing, the sem-check wrapper and `load_from_disk`.
- `launch.py`: final save and label.
- `experimental/inspect_tucker.py`: `UTT` stem tag.
- Resume: a checkpoint saved in another dtype (old float64 utt files) is cast to the data dtype, with a one-line notice. `utt_core_power` is part of the resume check.
- `--utt-core-power` accepts only a number in (0, 1] or `auto`.


## Validation

`0_tests/2026-10-28-test_uniform_tt.ipynb` (GPU).
