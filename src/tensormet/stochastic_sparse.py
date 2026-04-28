"""
stochastic_sparse.py — Per-iteration NNZ subsampling for Tucker updates.

Design
------
Each factor/core update accumulates a numerator that is a sum over all NNZ
entries of the sparse tensor.  By sampling a fraction *p* of those entries
uniformly at random and rescaling their values by 1/p, we obtain an unbiased
estimator of the full numerator:

    E[Num_stoch] = Num_exact    (for any uniform random subset S of size p·nnz)

The denominator is analytical (depends only on core and factors) and is
always kept exact.

Wall-clock time per iteration scales as O(p) on the NNZ-bound operations.
Typical useful range: p = 0.1–0.3 for large NNZ counts.

Usage
-----
In the main decomposition loop (tucker_tensor.py), replace ``self.tensor``
with the output of ``subsample_coo`` when stochastic mode is active:

    _current_tensor = (
        subsample_coo(self.tensor, shape, subsample_frac, _iter_rng)
        if _use_subsample else self.tensor
    )

The returned COO has the same ``(block_size, n_blocks)`` shape and rescaled
values, so all existing update functions work without modification.

For multi-GPU (ShardedSparseTensor), sampling happens inside each per-shard
function using a deterministic seed derived from the iteration number and
shard index — no per-iteration resharding is needed.
"""

from __future__ import annotations

import cupy as cp
import cupyx.scipy.sparse as cpx_sparse


def subsample_coo(
    coo: cpx_sparse.coo_matrix,
    shape: tuple,
    frac: float,
    rng: cp.random.RandomState,
) -> cpx_sparse.coo_matrix:
    """
    Return a rescaled random subsample of *coo* with the same shape.

    NNZ entries are sampled uniformly without replacement.  Their values are
    multiplied by ``1/frac`` so that any downstream accumulation over the
    returned matrix is an unbiased estimator of the same accumulation over
    the full matrix.

    The returned matrix preserves the ``(block_size, n_blocks)`` shape of
    the input so it is a drop-in replacement wherever ``vec_tensor`` is
    expected (all largedim update functions, error functions, etc.).

    Parameters
    ----------
    coo :
        Full COO matrix on the primary CUDA device.
    shape :
        Original N-D tensor shape.  Kept for API symmetry; not used here.
    frac :
        Sampling fraction in (0, 1].  ``frac=1.0`` returns an equivalent
        view of the original matrix (still rescaled by 1.0, so values are
        unchanged).
    rng :
        ``cp.random.RandomState`` instance, advanced in-place by this call.
        Use ``make_iteration_rng`` to create one before the training loop.

    Returns
    -------
    cpx_sparse.coo_matrix
        Subsampled COO with ``⌈frac · nnz⌉`` entries and the same shape.
    """
    coo_c = coo.tocoo()
    nnz = int(coo_c.row.size)

    if nnz == 0:
        return coo_c

    n_sample = max(1, int(round(frac * nnz)))

    # Sample without replacement on GPU, then sort to restore row-major order
    # (better cache locality in downstream index arithmetic).
    idx = rng.permutation(nnz)[:n_sample]
    idx = cp.sort(idx)

    scale = coo_c.data.dtype.type(1.0 / frac)

    return cpx_sparse.coo_matrix(
        (coo_c.data[idx] * scale,
         (coo_c.row[idx], coo_c.col[idx])),
        shape=coo_c.shape,
    )


def make_iteration_rng(base_seed: int) -> cp.random.RandomState:
    """
    Create a ``cp.random.RandomState`` seeded from *base_seed*.

    Call this once before the training loop.  Pass the returned object to
    ``subsample_coo`` on each iteration; the state advances automatically so
    each iteration draws a different sample while the full run is reproducible
    given the same ``base_seed``.

    Parameters
    ----------
    base_seed :
        Integer seed, typically ``cfg.exp.random_state``.
    """
    return cp.random.RandomState(seed=int(base_seed))
