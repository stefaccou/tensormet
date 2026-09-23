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

``CooSubsampler`` shuffles the NNZ once and takes a contiguous rotating window
per iteration:

    window(t) = perm[(t·n_sample) % nnz : +n_sample]    (wrapping)

A window of a uniform shuffle is a uniform sample without replacement, so the
rescaled accumulations stay unbiased, and successive windows tile the NNZ like
an epoch. The sample is a pure function of (base_seed, iteration), so resumed
runs draw the same windows.

``cfg.exp.max_nnz`` is applied upstream in tucker_tensor.py as an effective
fraction, so *p* may already embed it. The multi-GPU equivalent is
``sharded_sparse.apply_subsample``.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from tensormet.utils import make_lazy_cupy_pair
from tensormet.sparse_ops import CoordCOO
cp, cpx_sparse = make_lazy_cupy_pair()


class CooSubsampler:
    """
    Owns a one-time shuffled ordering of a COO matrix's NNZ and yields
    per-iteration contiguous-window subsamples.

    Memory: one persistent int64 permutation (8·nnz bytes, drawn host-side);
    per-iteration allocations are O(n_sample).

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
        # CoordCOO carries no linear index and needs no .tocoo(); its `take`
        # applies the same window along the NNZ axis.
        self._is_coord = isinstance(coo, CoordCOO)
        self.coo = coo if self._is_coord else coo.tocoo()
        self.frac = float(frac)
        self.nnz = self.coo.nnz if self._is_coord else int(self.coo.row.size)
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

        Values are multiplied by ``1/frac`` (unbiased accumulations); the shape
        is unchanged, so it drops in wherever ``vec_tensor`` is expected.
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
        if self._is_coord:
            sampled = self.coo.take(idx)
            sampled.data = sampled.data * scale
            return sampled
        return cpx_sparse.coo_matrix(
            (self.coo.data[idx] * scale,
             (self.coo.row[idx], self.coo.col[idx])),
            shape=self.coo.shape,
        )
