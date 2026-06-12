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

CHANGED (2026-06-12 review, Task 2): sampling no longer draws a fresh
``rng.permutation(nnz)`` per iteration (an 8·nnz-byte device allocation plus a
full sort — at nnz = 10⁸–10⁹ that is 0.8–8 GB per iteration on the very GPU
subsampling is meant to relieve).  Instead, ``CooSubsampler`` fixes one
uniform permutation of the NNZ at construction and takes a contiguous rotating
window of it per iteration:

    window(t) = perm[(t·n_sample) % nnz : +n_sample]    (wrapping)

Estimator argument: a contiguous window of a uniformly shuffled sequence is a
uniform sample without replacement, so linear accumulations over the rescaled
window remain unbiased; successive windows tile the NNZ like an epoch (every
entry visited once per ⌈nnz/n_sample⌉ iterations).  The sample is a pure
function of (base_seed, iteration) — no RNG state advances between calls, so
a resumed run draws exactly the same windows as an uninterrupted one (this
also fixes review finding I-3).

Wall-clock time per iteration scales as O(p) on the NNZ-bound operations.
Typical useful range: p = 0.1–0.3 for large NNZ counts.

Usage
-----
In the main decomposition loop (tucker_tensor.py), build the sampler once
before the loop and draw from it per iteration:

    _iter_sampler = CooSubsampler(self.tensor, shape, subsample_frac,
                                  cfg.exp.random_state)
    ...
    _current_tensor = (
        _iter_sampler.sample(iteration) if _use_subsample else self.tensor
    )

The returned COO has the same ``(block_size, n_blocks)`` shape and rescaled
values, so all existing update functions work without modification.

For multi-GPU (ShardedSparseTensor), the same pattern lives shard-side: each
shard's NNZ arrays are shuffled once at construction and the per-shard
functions take contiguous windows — see ``sharded_sparse._apply_subsample``.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from tensormet.utils import make_lazy_cupy_pair
cp, cpx_sparse = make_lazy_cupy_pair()


class CooSubsampler:
    """
    Owns a one-time shuffled ordering of a COO matrix's NNZ and yields
    per-iteration contiguous-window subsamples.

    CHANGED (Task 2): replaces the former ``subsample_coo(coo, shape, frac,
    rng)`` + ``make_iteration_rng(seed)`` pair, which permuted the full NNZ on
    the GPU every iteration with a stateful RNG (not checkpoint-safe).

    Memory: one persistent int64 index array of 8·nnz bytes on the device
    (built from a host-side ``np.random.default_rng`` permutation, so no GPU
    sort ever runs); per-iteration allocations are O(n_sample) — the gathered
    row/col/data of the window — never O(nnz).

    Parameters
    ----------
    coo :
        Full COO matrix on the primary CUDA device, in the package's blocked
        ``(block_size, n_blocks)`` shape.
    shape :
        Original N-D tensor shape.  Kept for API symmetry; not used here.
    frac :
        Sampling fraction in (0, 1].  ``frac=1.0`` disables sampling:
        ``sample()`` returns the original matrix and no permutation is stored.
    base_seed :
        Integer seed for the one-time shuffle, typically
        ``cfg.exp.random_state``.  Together with the iteration number it fully
        determines every sample (resume-safe).
    """

    def __init__(
        self,
        coo: cpx_sparse.coo_matrix,
        shape: tuple,
        frac: float,
        base_seed: Optional[int] = 0,
    ) -> None:
        self.coo = coo.tocoo()
        self.frac = float(frac)
        self.nnz = int(self.coo.row.size)
        self.n_sample = max(1, int(round(self.frac * self.nnz))) if self.nnz else 0
        if self.nnz > 0 and self.frac < 1.0:
            # Host-side permutation transferred once; cheaper and more
            # deterministic across CuPy versions than a device-side shuffle.
            perm_np = np.random.default_rng(int(base_seed or 0)).permutation(self.nnz)
            self._perm = cp.asarray(perm_np)
        else:
            self._perm = None

    def sample(self, iteration: int) -> cpx_sparse.coo_matrix:
        """
        Return iteration *t*'s rescaled subsample of the wrapped COO.

        The window is ``perm[(t·n_sample) % nnz : +n_sample]`` (wrapping), and
        values are multiplied by ``1/frac`` so any downstream accumulation is
        an unbiased estimator of the same accumulation over the full matrix.
        The returned matrix preserves the input's ``(block_size, n_blocks)``
        shape, so it is a drop-in replacement wherever ``vec_tensor`` is
        expected.
        """
        if self._perm is None:
            return self.coo

        start = (int(iteration) * self.n_sample) % self.nnz
        end = start + self.n_sample
        if end <= self.nnz:
            idx = self._perm[start:end]
        else:  # wrap around the end of the shuffled sequence
            idx = cp.concatenate((self._perm[start:], self._perm[: end - self.nnz]))

        scale = self.coo.data.dtype.type(1.0 / self.frac)
        return cpx_sparse.coo_matrix(
            (self.coo.data[idx] * scale,
             (self.coo.row[idx], self.coo.col[idx])),
            shape=self.coo.shape,
        )
