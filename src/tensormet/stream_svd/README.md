# Streamed GPU SVD init (`--init svd_stream`)

Status: implemented and validated (2026-10-29: full test notebook passes, including the real 4-gram, 34 s on 1 GPU, 20.5 s on 2). 2026-11-06: at the old default sketch `q2o10` different seeds give quite different bases (mean column |cos| 0.56); at `--svd-power-iter 6 --svd-oversampling 100` they agree (0.9997), for 20 s instead of 8 s on the 3-gram with 2 GPUs. On 2026-11-12 the default became `q6o100` (merge into the main package). Still open: whether it helps training.

## 1. Goal

Start the non-negative decomposition from an SVD-based guess (NNDSVD) instead of random factors, on the full n-gram tensors:

- 4-grams at $V = 10\text{k}$ with $3.5\times10^8$ non-zeros, up to $V = 131\text{k}$ and order 6.
- Fast and scalable first, exact second.
- Use every GPU the run already has.
- Save the result so later runs skip it.

## 2. Why the existing SVD inits don't do this

| init | problem at our scale |
|---|---|
| `svd`, `svd_loose`, `randomised_svd` | CPU; reject `CoordCOO` (used by every largedim run); each step forms a dense vector of $V^{N-1}$ entries ($10^{12}$, ~8 TB, at 4-grams/10k), so only feasible up to ~3-grams |
| all of them | with shared factors they compute $N$ SVDs and keep only mode 0's |

## 3. What is computed

Let $A_{(n)} \in \mathbb{R}^{V \times V^{N-1}}$ be the mode-$n$ unfolding of the tensor $\mathcal{X}$. With `shared_factors="all"`, one matrix serves every position, so for a group $T$ of tied modes we take the top-$k$ left singular vectors of all of its unfoldings side by side. These are the top-$k$ eigenvectors of

$$
G \;=\; \sum_{n \in T} A_{(n)} A_{(n)}^\top \;\in\; \mathbb{R}^{V\times V}.
$$

Untied modes get one $G$ each ($T = \{n\}$). The eigenvectors $U$ and singular values $s_i = \sqrt{\lambda_i}$ then go through NNDSVD, unchanged from the other inits. NNDSVD keeps the larger of the positive and negative part of each column of $U$ and rescales it with $s_i$, which gives a non-negative factor $F$.

The other part becomes exact zeros, about half of every column after the first. MU multiplies, so those entries stay at the $10^{-30}$ floor: on the 11-05 sweep 99.6% were still near zero after 50 iterations (`22_initialisation/2026-11-13-nndsvd_zero_support.ipynb`). `--svd-fill c` starts them at $c\,\bar F_{:,j}$ (the column mean) instead; $c = 1$ is NNDSVDa. The core is built from the filled $F$.

## 4. How: never form $G$

**Randomized subspace iteration.** Only products $G\Omega$ with a thin $\Omega \in \mathbb{R}^{V\times p}$, where $p = k + o$ (oversampling $o$, `--svd-oversampling`, default 100), are needed:

$$
Y_0 = G\Omega,\qquad Y_{j+1} = G\,\operatorname{orth}(Y_j)\;\;(j < q),\qquad Q = \operatorname{orth}(Y_q),\qquad B = Q^\top G Q .
$$

The top-$k$ eigenpairs of the small $p\times p$ matrix $B$ give $U = Q V_B$ and $s = \sqrt{\lambda(B)}$. That is $q+2$ products in all; the default $q = 6$ (`--svd-power-iter`) gives 8. Each product with $G = AA^\top$ counts as one SVD power iteration. The relative residual $\lVert G u_j - \lambda_j u_j\rVert / \lambda_j$ of every column comes free from $W = GQ$; it is printed and stored in the cache.

**Contexts.** A *context* of an entry, for mode $n$, is the n-gram with position $n$ left out, i.e. a column $c$ of $A_{(n)}$. Writing $a_c$ for that column,

$$
G\Omega \;=\; \sum_{n\in T}\;\sum_{c} a_c\,\big(a_c^\top \Omega\big).
$$

Sorting the non-zeros by context makes each $a_c$ a contiguous slice. The product is then two sparse–dense products per chunk of contexts: $Z = A_c^\top\Omega$, then $Y \mathrel{+}= A_c Z$. It uses the existing `build_batch_csr_T` and `spmm_T`, with no atomics and no dense $V\times V$ or $V\times V^{N-1}$ object. Cost per product is $\mathcal{O}(|T|\cdot \text{nnz}\cdot p)$.

**Sort key.** The key ravels (context, row) into an int64. When the product of the dimensions exceeds $2^{63}$ (6-grams, or 4-grams at $V = 131$k), the running key is first replaced by its rank among the distinct values. Ranks keep the order, so the result is identical.

## 5. Accuracy vs. speed

Quality is measured as the share of the best possible top-$k$ energy:

$$
\rho \;=\; \frac{\operatorname{tr}(U^\top G U)}{\sum_{i\le k}\lambda_i(G)} \;\le 1,
\qquad\text{and}\qquad \operatorname{tr}(U^\top G U) = \sum_{i\le k} s_i^2 .
$$

The second identity lets us measure $\rho$ on real data without ever having $G$: compare $\sum s_i^2$ between the default and a run with many passes.

Roughly, the error shrinks like $(\lambda_{p+1}/\lambda_k)^{q}$, so a flat spectrum (small gap) needs more passes. Each extra pass costs one more streamed product, $1/(q+2)$ of the total (12% at the default).

**Single-precision floor.** $G$ squares the condition number and the products are summed in fp32, so the residual of column $j$ cannot fall below about $6\times10^{-8}\,\lambda_1/\lambda_j$. A column with $s_j < 0.01\,s_1$ therefore never prints as converged ($< 10^{-3}$), whatever $q$; columns with $s_j < 3\times10^{-4}\,s_1$ are noise.

## 6. Multi-GPU

**Contexts must stay whole on one GPU.** If a context's column is split over two GPUs, $a = a^1 + a^2$, the GPUs only produce

$$
a^1a^{1\top} + a^2a^{2\top} \;\neq\; aa^\top = a^1a^{1\top} + a^2a^{2\top} + a^1a^{2\top} + a^2a^{1\top}.
$$

The training shards split the non-zeros arbitrarily, so they can't be reused.

**Cells.** Each context's coordinates are hashed into a *cell*: one fused GPU kernel, 4 bytes per entry. Every context falls in exactly one cell, so

$$
G\Omega = \sum_{\text{GPU } g}\;\sum_{\text{cells of } g} (\cdots) ,
$$

and each GPU computes its part $Y_g$ independently. Then $Y = \sum_g Y_g$ is summed on GPU 0, and the small QR and eigen-step run there.

| | compute | data movement | memory per GPU |
|---|---|---|---|
| per product | $\approx 1/G$ of single-GPU | $\Omega$ out, $Y_g$ back: $\approx 2GVp\cdot4$ B ($\approx$ 0.5 GB at $G=4$, $V=131$k) | — |
| once per mode | hash + sort | share of non-zeros to each GPU via host | sorted share: 12 B/entry |

## 7. Memory (the out-of-memory fix)

The first version sorted all $3.5\times10^8$ entries of a mode in one go. That took about 11 GB of sort scratch on top of the 7 GB tensor on a ~24 GB GPU, and it failed.

Now each GPU's share is split into enough cells that **one cell's sort fits in 40% of the free memory**, measured at that moment. Then:

- **Resident**: if all sorted modes fit (12 B per entry per mode), they are built once and reused by every product.
- **Streaming**: otherwise each product re-sorts cell by cell and throws the result away.

On one 24 GB GPU the 4-gram case streams (about 6 cells per mode). On two GPUs it should stay resident (about 8.4 GB each).

## 8. Cache

- **Location**: `tensors/{dataset}/decomposition/init_cache/`, always the canonical path, also under HPC staging.
- **Key**: populated tensor (method, order, dims, shared factors), the tied group, the sketch settings ($q$, oversampling) and the random state, which seeds the sketch (e.g. `..._q6o100_rs2.npz`). Each seed gets its own init; a rerun with the same seed reuses it. Rank, decomposition, iterations, subsampling and GPU count are left out on purpose.
- **Stored**: raw $U$ ($V\times K$) and $s$ per group; the Tucker core in a separate file.
- **Smaller ranks are exact.** NNDSVD treats each column on its own, so rank $k \le K$ is just the first $k$ columns. The core

  $$
  \mathcal{C} = \mathcal{X}\times_1 F_1^\top \times_2 \cdots \times_N F_N^\top
  $$

  has entry $(r_1,\dots,r_N)$ depending only on columns $r_1,\dots,r_N$. A smaller core is therefore the leading sub-block of a stored larger one.
- **Safety**: a (nnz, $\sum x$, shape) fingerprint triggers a recompute if the tensor changed. Writes are atomic. The fingerprint does not see a reordered vocabulary: if the word order of a tensor changes, delete its `init_cache` files by hand.
- **Other decompositions**: CP, TT and UTT ask for no core (`with_core=False`), so none is built.

## 9. Test results so far

Notebook: `0_tests/2026-10-28-svd_stream_init.ipynb` (ampere, GPUs 1–2), on a planted $300^3$ tensor with $2\times10^4$ non-zeros.

| check | result |
|---|---|
| streamed $G\Omega$ vs dense, per mode and pooled, 600–1800 chunks | rel. error $\le 2\times10^{-6}$ ✔ |
| compacted sort key vs int64 key | identical ✔ |
| old default sketch ($q=2$), mode 0 (decaying spectrum) | $\rho = 0.978$ |
| same sketch, mode 1 (flat, noise-only spectrum) | $\rho = 0.950$, below the 0.95 test threshold ✘ |
| first real-data run (4-gram, $V=10$k, nnz $3.5\times10^8$) | out of memory in the sort ✘ → fixed by sorting in cells (§7) |

The mode-1 result is the speed/accuracy trade-off on a test tensor whose planted core only decays along mode 0. The test now sweeps $q$, prints the spectral gap, and requires $\rho \ge 0.999$ at $q = 30$, which is the correctness check.

After the fixes (2026-10-29), all cells pass:

| check | result |
|---|---|
| $G\Omega$ with 7 cells per GPU; low-memory route vs resident | rel. error $\le 2.4\times10^{-6}$; same basis (cos = 1.000000) ✔ |
| $\rho$ vs $q$ (modes 0 / 1 / pooled) | $q=2$: 0.978 / 0.950 / 0.950; $q=8$: 0.9996 / 0.997 / 0.999; $q=30$: 1.0000 ✔ |
| 1 vs 2 GPUs | NNZ split 50/50; same basis (cos = 1.000000) ✔ |
| cache: hit, truncation, core slice, seed, CP, hook, changed tensor | all ✔ |
| real 4-gram, all 4 modes pooled, $k=100$, 1 GPU (low-memory route) | **34.3 s**, peak 18.5 GB (incl. 7 GB tensor) |
| same, 2 GPUs (resident, 28 cell groupings) | **20.5 s** (1.7×), peak 21.2 / 16.8 GB |
| real data: $\sum s_i^2$ at $q=2$ vs $q=8$ | **0.9886** of the energy, at 34 s vs 85 s → was the old default $q=2$ |

GPU 0 peaks at 21.2 GB on 2 GPUs: it holds the tensor, its cells and the hash scratch. That is the device to watch on smaller cards.

## 10. Open questions

1. ~~Does the notebook pass after the memory fix?~~ Yes.
2. ~~Share of the $q=8$ energy captured at $q=2$?~~ 0.9886 at the old default; the default is now `q6o100`.
3. ~~Time and peak memory, 1 vs 2 GPUs?~~ 34 s / 20.5 s (§9).
4. Does it help training? Short runs with `random` vs `svd_stream`; compare KL curves and dev scores.
   `svd_loose` can't be a baseline at 4-gram/10k. Largedim runs use `CoordCOO`, which the CPU SVD path rejects. Even block-encoded, each ARPACK step forms `A.T @ x`, a dense vector of $V^{N-1}=10^{12}$ floats (~8 TB). It only works up to about 3-grams, which is where to cross-check `svd_stream` against it.

## Files

- `stream_svd.py`: algorithm, cells, multi-GPU, entry point `initialize_svd_stream`.
- `init_cache.py`: cache.
- Core hooks: one branch in `sparse_ops.initialize_nonnegative_tucker`, and a `set_run_context(...)` call in `solvers.py` just before the init dispatch.
- Test knobs (module globals of `stream_svd.py`): `_MIN_CELLS`, `_FORCE_STREAM`, `_FORCE_COMPACT`.
