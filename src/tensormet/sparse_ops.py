from __future__ import annotations
import tensorly as tl
import numpy as np
from typing import List, Tuple, Optional, Union
import math
import os
import threading
from tensormet.utils import einsum_letters, cp_einsum_optimize, make_lazy_cupy_pair, lazy_import

# pytensorlab transitively imports TensorFlow + VTK (~200s); defer it until used.
ptl = lazy_import("pytensorlab")
cp, cpx_sparse = make_lazy_cupy_pair()

# -------------------------------------------------------------------
# Helper functions to strictly enforce int64 bounds, bypassing
# np.ravel_multi_index / cp.ravel_multi_index C-level limits.
# -------------------------------------------------------------------
def safe_unravel(flat_idx, shape, xp):
    """Unravels flat indices safely using pure 64-bit array math."""
    coords = []
    curr = flat_idx
    for dim in reversed(shape):
        coords.append(curr % dim)
        curr = curr // dim
    return tuple(reversed(coords))

def safe_ravel(coords, shape, xp):
    """Ravels coordinates safely using pure 64-bit array math."""
    if not coords:
        return xp.zeros(1, dtype=xp.int64)
    flat = xp.zeros_like(coords[0], dtype=xp.int64)
    stride = xp.int64(1)
    for i in reversed(range(len(shape))):
        flat += coords[i].astype(xp.int64) * stride
        stride *= xp.int64(shape[i])
    return flat
# -------------------------------------------------------------------


# -------------------------------------------------------------------
# Coordinate-backed sparse tensor (replaces the block-encoded linear index
# for the NNZ-streaming kernels).
# -------------------------------------------------------------------
class CoordCOO:
    """GPU-resident coordinate-backed sparse tensor for the NNZ-streaming kernels.

    The alternative, produced by ``torch_sparse_to_cupy``, is a 2-D
    ``coo_matrix`` holding a linearised index split into ``(row, col)`` blocks
    of ``int32_max``. It tops out at ``int32_max**2`` (~4.6e18) elements, and
    the linear index itself at int64 (~9.2e18) — a 5-gram at dim=10000 needs
    1e20, which ``np.ravel_multi_index`` rejects outright.

    Keeping the coordinates removes that ceiling (only each individual
    dimension must fit int32) and the per-iteration decode: the largedim
    kernels' first act was always to undo the block encoding, making
    ``coords -> flat -> (row, col) -> flat -> coords`` pure overhead.

    Memory: ``ndim·4`` bytes/NNZ stored vs the block form's ``2·4``, but cheaper
    at peak — the block form also materialises ``flat`` + N ``idxs`` (int64)
    every iteration. Tell ``precompute_largedim_batches(coord_backed=)`` which
    form it is, or it reserves headroom for a decode that never runs.

    ``shape`` is the N-D tensor shape, not a matrix shape: this is not an
    ``spmatrix``. Only the ``*_largedim`` kernels accept it; the dense-unfolding
    kernels and SVD initialisers raise on it.

    See ``utils.SparseCOOTensor`` for the torch-side analogue used at load time.
    """
    is_sparse = True

    def __init__(self, coords, data, shape: tuple):
        # coords: (ndim, nnz) int32 device array; data: (nnz,) device array
        self.coords = coords
        self.data = data
        self.shape = tuple(int(d) for d in shape)

    @property
    def ndim(self) -> int:
        return len(self.shape)

    @property
    def nnz(self) -> int:
        return int(self.data.size)

    # As on coo_matrix, these describe the VALUES, not the coordinates.
    # `tl.context()` reads .dtype to build the random init (sparse_ops.py:1283).
    @property
    def dtype(self):
        return self.data.dtype

    @property
    def device(self):
        return self.data.device

    def coord_list(self) -> list:
        """Per-mode index arrays ``[(nnz,)] * ndim`` (views — no copy, no decode)."""
        return [self.coords[n] for n in range(self.coords.shape[0])]

    def take(self, idx) -> "CoordCOO":
        """Select NNZ by index array or slice, preserving ``shape``."""
        return CoordCOO(self.coords[:, idx], self.data[idx], self.shape)

    def to_device(self, device_id: int) -> "CoordCOO":
        """Copy the NNZ onto *device_id* (host round-trip for cross-device)."""
        if _array_device_id(self.coords) == device_id:
            with cp.cuda.Device(device_id):
                return CoordCOO(self.coords.copy(), self.data.copy(), self.shape)
        coords_np = cp.asnumpy(self.coords)
        data_np = cp.asnumpy(self.data)
        with cp.cuda.Device(device_id):
            return CoordCOO(cp.asarray(coords_np), cp.asarray(data_np), self.shape)

    def __repr__(self) -> str:
        return (f"CoordCOO(shape={self.shape}, nnz={self.nnz}, "
                f"coord_dtype={self.coords.dtype}, dtype={self.data.dtype})")


def _array_device_id(arr):
    """Best-effort CUDA device ordinal for *arr*; ``None`` if it is not on a GPU.

    CuPy arrays expose ``.device`` as a ``cupy.cuda.Device`` whose ``int()`` is the
    ordinal. NumPy >= 2.0 also exposes ``.device``, but as the string ``'cpu'`` (so
    ``int(...)`` raises) — those arrays are not CUDA-resident and must take the
    host-transfer path, hence ``None``.
    """
    dev = getattr(arr, "device", None)
    if dev is None:
        return None
    try:
        return int(dev)
    except (TypeError, ValueError):
        return None


def block_encoding_fits(shape) -> bool:
    """Whether the ``(row, col)`` block-encoded linear index can hold *shape*.

    The encoding stores ``flat % int32_max`` and ``flat // int32_max`` as int32,
    so the total element count must not exceed ``int32_max**2``. (The linear
    index would also have to fit int64, but that bound is the looser of the
    two.) When this is False the tensor must use :class:`CoordCOO`.
    """
    int32_max = int(np.iinfo(np.int32).max)
    return math.prod(tuple(int(s) for s in shape)) <= int32_max * int32_max


def require_matrix_form(vec_tensor, who: str) -> None:
    """Raise a directed error when a matrix-only kernel is handed a CoordCOO."""
    if isinstance(vec_tensor, CoordCOO):
        raise TypeError(
            f"{who} requires the block-encoded coo_matrix form, but received a "
            f"CoordCOO ({vec_tensor!r}). This tensor is too large for the linear "
            f"index that form needs (prod(shape) > int32_max**2), so only the "
            f"NNZ-streaming '*_largedim' kernels can consume it. Use largedim "
            f"routing, or reduce --dim/--order."
        )


def unfold_from_vectorized_sparse(
    vec_tensor: cpx_sparse.spmatrix,
    orig_shape,
    mode: int,
    to_dense: bool = False,
    backend: str = "cupy",
):
    """
    Unfold a sparse tensor that is stored as a vectorized CuPy sparse matrix.

    Parameters
    ----------
    vec_tensor : cupyx.scipy.sparse.spmatrix
        Sparse matrix of shape (np.prod(orig_shape), 1) created by
        `torch_sparse_to_cupy` for an N-D tensor.
    orig_shape : tuple[int, ...]
        Original N-D tensor shape, e.g. (I0, I1, I2).
    mode : int
        Mode along which to unfold.
    to_dense : bool, default False
        If True, return a dense cupy.ndarray.
        If False, return a cupy sparse COO matrix.
    backend : {"cupy", "scipy"}, default "cupy"
        Sparse backend for the (non-dense) output. cupy/cuSPARSE only supports
        32-bit indices, so a mode whose complementary dimensions multiply to more
        than int32_max columns (e.g. the "thin" time mode of a large tensor)
        cannot be represented on the GPU. Pass "scipy" to build a host
        scipy.sparse.coo_matrix with int64 indices instead. Ignored when
        ``to_dense=True``.

    Returns
    -------
    unfolded : cupy.ndarray or cupyx.scipy.sparse.coo_matrix
        Mode-`mode` unfolding of shape
        (orig_shape[mode], np.prod(orig_shape) // orig_shape[mode]).
    """
    # Make sure we're in COO format
    require_matrix_form(vec_tensor, "unfold_from_vectorized_sparse")

    cu = vec_tensor.tocoo()

    row_cp = cu.row
    col_cp = cu.col
    data_cp = cu.data

    orig_shape = tuple(orig_shape)
    # size = int(np.prod(orig_shape))
    # new: We now use math.prod to avoid np.prod 32-bit overflow
    size = math.prod(orig_shape)
    int32_max = np.iinfo(np.int32).max
    block_size = min(size, int32_max)

    # ---- move to host and use int64 for safe arithmetic ----
    row_np = cp.asnumpy(row_cp).astype(np.int64)
    col_np = cp.asnumpy(col_cp).astype(np.int64)

    flat_np = row_np + col_np * np.int64(block_size)

    # coords = np.unravel_index(flat_np, orig_shape)
    # new: We now use safe unravelling
    coords = safe_unravel(flat_np, orig_shape, np)

    row_unf_np = coords[mode]

    other_coords = coords[:mode] + coords[mode + 1:]
    other_shape = tuple(s for i, s in enumerate(orig_shape) if i != mode)

    # col_unf_np = np.ravel_multi_index(other_coords, other_shape)
    # new: We now use safe ravelling
    col_unf_np = safe_ravel(other_coords, other_shape, np)

    n_cols = int(math.prod(other_shape))
    unfolded_shape = (orig_shape[mode], n_cols)

    if backend == "scipy" and not to_dense:
        # Host path: scipy supports int64 indices, so modes whose complementary
        # dimensions exceed int32_max columns (impossible to hold on the GPU) are
        # handled here. No GPU round-trip — data is already host-bound below.
        import scipy.sparse as sp_sparse
        return sp_sparse.coo_matrix(
            (cp.asnumpy(data_cp), (row_unf_np, col_unf_np)),
            shape=unfolded_shape,
        )

    if not to_dense and n_cols > int32_max:
        raise ValueError(
            f"mode-{mode} unfolding has {n_cols} columns, which exceeds cupy's "
            f"int32 index limit ({int32_max}); use backend='scipy' for a host "
            f"scipy.sparse matrix with int64 indices."
        )

    row_unf_cp = cp.asarray(row_unf_np)
    col_unf_cp = cp.asarray(col_unf_np)

    unfolded = cpx_sparse.coo_matrix(
        (data_cp, (row_unf_cp, col_unf_cp)),
        shape=unfolded_shape,
    )

    if to_dense:
        return unfolded.toarray()
    return unfolded



def left_dense_mul_sparse(
    mat: cp.ndarray,
    sp: cpx_sparse.spmatrix
) -> Union[cp.ndarray, cpx_sparse.coo_matrix]:
    """
    Compute mat @ sp, choosing dense or sparse output based on a simple
    memory heuristic.

    mat: cupy ndarray of shape (R, I_mode)
    sp:  cupy sparse matrix of shape (I_mode, K)
    """
    sp = sp.tocoo()
    R, I_mode = mat.shape
    assert I_mode == sp.shape[0], f"mat shape {mat.shape} not compatible with sparse {sp.shape}"

    # Let CuPy handle dense @ sparse; result is cupy.ndarray
    return mat @ sp

def sparse_mode_dot_vec(
    vec_tensor: cpx_sparse.spmatrix,
    curr_shape: Tuple[int, ...],
    factor: cp.ndarray,
    mode: int,
    transpose_factor: bool = True,
) -> Tuple[cpx_sparse.coo_matrix, Tuple[int, ...]]:
    """
    Perform a mode-`mode` product on a vectorized sparse tensor (prod(curr_shape), 1),
    using a dense factor matrix, and return the new vectorized sparse tensor.

    vec_tensor: sparse COO (prod(curr_shape), 1)
    curr_shape: current tensor shape
    factor:     dense matrix of shape (I_mode, R_mode) (or R_mode, I_mode if transpose_factor=False)
    mode:       mode index in [0, len(curr_shape))
    transpose_factor: if True, use factor.T (for Tucker-style X ×_n W_n^T)

    Returns
    -------
    new_vec:   sparse COO (prod(new_shape), 1)
    new_shape: updated shape, with dimension at `mode` replaced by R_mode
    """
    curr_shape = tuple(curr_shape)
    I_mode = curr_shape[mode]

    # Factor handling
    if transpose_factor:
        # factor is (I_mode, R_mode) => mat is (R_mode, I_mode)
        assert factor.shape[0] == I_mode, f"factor shape {factor.shape} not compatible with dim {I_mode}"
        mat = tl.transpose(factor)  # (R_mode, I_mode)
    else:
        # factor is already (R_mode, I_mode)
        assert factor.shape[1] == I_mode, f"factor shape {factor.shape} not compatible with dim {I_mode}"
        mat = factor

    R_mode = mat.shape[0]

    # 1) Unfold current sparse tensor along this mode (sparse COO)
    unfolded = unfold_from_vectorized_sparse(
        vec_tensor,
        curr_shape,
        mode,
        to_dense=False,
    )  # shape: (I_mode, prod(other_dims))

    # 2) Left-multiply with dense matrix; currently returns dense cp.ndarray
    #    -> shape: (R_mode, prod(other_dims))
    unfolded_new = left_dense_mul_sparse(mat, unfolded)

    # 3) Fold back into a new vectorized sparse tensor with updated shape
    new_vec, new_shape = fold_unfolded_sparse_to_vec(
        unfolded_new,
        old_shape=curr_shape,
        mode=mode,
        new_dim=R_mode,
    )
    return new_vec, new_shape

def sparse_multi_mode_dot_vec(
    vec_tensor: cpx_sparse.spmatrix,
    orig_shape: Tuple[int, ...],
    factors: List[cp.ndarray],
    modes: Optional[List[int]] = None,
    transpose_factors: bool = True,
) -> cp.ndarray:
    """
    multi_mode_dot for a vectorized sparse tensor (prod(orig_shape), 1),
    applying dense factor matrices along the given modes, **staying sparse**
    until the final (small) result, which is densified.

    vec_tensor: sparse COO (prod(orig_shape), 1)
    orig_shape: original tensor shape
    factors:    list of factor matrices, one per mode index
                factor[n] has shape (I_n, R_n)
    modes:      list of modes to apply; if None, uses range(len(factors))
    transpose_factors: if True, uses factors[n].T (Tucker-style)
    """
    if modes is None:
        modes = list(range(len(factors)))

    current_vec = vec_tensor
    current_shape = tuple(orig_shape)

    # Apply each mode in any order (commutes)
    for mode in modes:
        current_vec, current_shape = sparse_mode_dot_vec(
            current_vec,
            current_shape,
            factors[mode],
            mode=mode,
            transpose_factor=transpose_factors,
        )

    # At this point, current_vec is still sparse (prod(core_shape), 1)
    core_shape = current_shape  # typically (50, 50, 50) or similar
    # should not overflow the cupy 32bit index limit if dimensions stay reasonable
    # Finally densify the small core
    coo = current_vec.tocoo()
    flat = coo.row
    data = coo.data

    # Build dense core
    coords = cp.unravel_index(flat, core_shape)
    core_dense = cp.zeros(core_shape, dtype=data.dtype)
    core_dense[coords] = data

    return core_dense


def fold_unfolded_sparse_to_vec(
    unfolded: Union[cpx_sparse.spmatrix, cp.ndarray],
    old_shape: Tuple[int, ...],
    mode: int,
    new_dim: int,
) -> Tuple[cpx_sparse.coo_matrix, Tuple[int, ...]]:
    """
    Fold a mode-`mode` unfolded matrix back to a vectorized sparse tensor.

    unfolded:
        - sparse COO or any cupyx.scipy.sparse.spmatrix of shape (new_dim, prod(other_dims)), or
        - dense cupy.ndarray of the same shape.
    old_shape : original N-D shape BEFORE replacing dimension at `mode`
    mode      : mode index that was unfolded
    new_dim   : new size at `mode` (typically rank[mode])

    Returns
    -------
    vec_sparse : COO of shape (prod(new_shape), 1)
    new_shape  : tuple of ints, updated tensor shape
    """

    old_shape = tuple(old_shape)
    N = len(old_shape)

    new_shape = list(old_shape)
    new_shape[mode] = new_dim
    new_shape = tuple(new_shape)

    other_shape = tuple(s for i, s in enumerate(old_shape) if i != mode)

    if cpx_sparse.isspmatrix(unfolded):
        unfolded = unfolded.tocoo()
        row = unfolded.row
        col = unfolded.col
        data = unfolded.data
    else:
        row, col = cp.nonzero(unfolded)
        data = unfolded[row, col]

    # coords_other = cp.unravel_index(col, other_shape)
    # new: We now use safe unravelling (force col to int64)
    coords_other = safe_unravel(col.astype(cp.int64), other_shape, cp)

    coords_full = []
    idx_other = 0
    for i in range(N):
        if i == mode:
            coords_full.append(row)
        else:
            coords_full.append(coords_other[idx_other])
            idx_other += 1

    coords_full = tuple(coords_full)

    # size = int(np.prod(new_shape))
    # new: We now use math to force correct behaviour in large dimensions
    size = math.prod(new_shape)
    int32_max = np.iinfo(np.int32).max
    block_size = min(size, int32_max)

    # flat = cp.ravel_multi_index(coords_full, new_shape)
    # New: use the safe ravelling function
    flat = safe_ravel(coords_full, new_shape, cp)

    # --- block encoding of flat indices ---
    row_vec = flat % block_size
    col_vec = flat // block_size

    n_blocks = int((size + block_size - 1) // block_size)
    vec_sparse = cpx_sparse.coo_matrix(
        (data, (row_vec, col_vec)),
        shape=(block_size, n_blocks),
    )
    vec_sparse.sum_duplicates()

    return vec_sparse, new_shape


def ptl_tucker_to_tensor(tucker: ptl.TuckerTensor,
                         skip_factor: Optional[int] = None) -> np.ndarray:
    """Reconstruct full tensor from Tucker representation, optionally skipping one factor."""
    factors = tucker.factors
    if skip_factor is not None:
        factors = [f for i, f in enumerate(factors) if i != skip_factor]
    return ptl.tmprod(tucker.core, factors, list(range(tucker.ndim)) if skip_factor is None else
                     [i for i in range(tucker.ndim) if i != skip_factor])

def gather_dense_at_block_nz(dense_nd: np.ndarray,
                             vec_tensor: cpx_sparse.spmatrix,
                             orig_shape) -> cp.ndarray:
    require_matrix_form(vec_tensor, "gather_dense_at_block_nz")
    orig_shape = tuple(orig_shape)
    # new: use math.prod instead of numpy
    size = math.prod(orig_shape)
    int32_max = np.iinfo(np.int32).max
    block_size = min(size, int32_max)

    dense_flat = dense_nd.reshape(size, order="C")
    coo = vec_tensor.tocoo()
    flat = coo.row.astype(cp.int64) + coo.col.astype(cp.int64) * cp.int64(block_size)
    return dense_flat[flat.get()]

# def compute_Zcols_batch(core, factors, mode, other_modes, idxs_by_mode, epsilon=1e-12):
#     """
#     Compute Z columns (as rows) for a batch of unfolding columns, without building full Z.
#
#     Returns Z_u with shape (m, R_mode), where m = batch size.
#     """
#     N = core.ndim
#     letters = einsum_letters(N)
#     core_subs = "".join(letters)
#
#     # factor-row matrices for each other mode: (m, Rk)
#     mats = [factors[k][idxs_by_mode[k]] for k in other_modes]
#
#     # einsum: core[a b c ...], M_b[m b], M_c[m c], ... -> out[m a_mode]
#     in_terms = [core_subs] + [("m" + letters[k]) for k in other_modes]
#     out_term = "m" + letters[mode]
#     eq = ",".join(in_terms) + "->" + out_term
#
#     Z_u = cp.einsum(eq, core, *mats)
#     Z_u = cp.clip(Z_u, a_min=epsilon, a_max=None)
#     return Z_u

# CHANGED (2026-08-04, perf regression fix): the einsum bodies of
# compute_Zcols_batch and distance._rhat_from_factor_rows_sequential are the
# default again. Their 2026-07-30 mode-at-a-time peel rewrites (m-batched
# (1,R)x(R,rest) GEMVs) were identified by the Aug-03/04 bisect as the
# iteration-time regression: the jul29 snapshot (einsum) is fast, the peel is
# ~2x slower, and TENSORMET_LEGACY_FACTOR_BATCH=1 does not recover it. The
# peel is kept behind TENSORMET_PEEL_CONTRACTION=1: it bounds the einsum's
# machine-dependent path choice (a (b, R0..R_{N-1}) intermediate was once
# materialized on an 80 GB node), so switch it on if a rank-space contraction
# ever OOMs.
PEEL_CONTRACTION = os.environ.get("TENSORMET_PEEL_CONTRACTION", "") not in ("", "0", "false", "False")


def use_peel_contraction():
    """Read the peel-contraction flag at call time so it can be toggled
    in-process (``sparse_ops.PEEL_CONTRACTION = True``) for A/B comparison."""
    return PEEL_CONTRACTION


def compute_Zcols_batch(core, factors, mode, other_modes, idxs_by_mode, epsilon=1e-12):
    """
    Compute Z columns (as rows) for a batch of unfolding columns, without building full Z.

    Returns
    -------
    Z_u : (m, R_mode)
        Row t is the reconstructed unfolded column corresponding to the
        coordinates encoded in idxs_by_mode for batch item t.
    """
    if not other_modes:
        m = len(list(idxs_by_mode.values())[0]) if idxs_by_mode else 1
        return cp.clip(cp.tile(core, (m, 1)), a_min=epsilon, a_max=None)

    if use_peel_contraction():
        # Memory-bounded fallback: peel one mode at a time in a fixed order.
        # Peak live array is (m, prod(R_k) for k != k0) — the working set
        # _estimate_batch_cols_for_Z budgets for — but the m-batched tiny
        # GEMVs make it measurably slower than the einsum path.
        R_mode = int(core.shape[mode])
        k0 = other_modes[0]
        M0 = factors[k0][idxs_by_mode[k0]]                   # (m, R_k0)
        m = int(M0.shape[0])
        G = cp.ascontiguousarray(cp.transpose(core, tuple(other_modes) + (mode,)))
        tmp = M0 @ G.reshape(G.shape[0], -1)
        for k in other_modes[1:]:
            M = factors[k][idxs_by_mode[k]]                  # (m, R_k)
            Rk = int(M.shape[1])
            tmp = cp.matmul(M[:, None, :], tmp.reshape(m, Rk, -1))[:, 0, :]
        return cp.clip(tmp.reshape(m, R_mode), a_min=epsilon, a_max=None)

    N = core.ndim
    letters = einsum_letters(N)
    core_sub = "".join(letters)                              # e.g. 'abc' for N=3
    mat_subs = ["i" + letters[k] for k in other_modes]      # batch × rank for each contracted mode
    out_sub  = "i" + letters[mode]                           # keep batch + target-mode rank
    eq = core_sub + "," + ",".join(mat_subs) + "->" + out_sub
    mats = [factors[k][idxs_by_mode[k]] for k in other_modes]
    tmp = cp.einsum(eq, core, *mats, optimize=cp_einsum_optimize(1 + len(other_modes)))
    return cp.clip(tmp, a_min=epsilon, a_max=None)


# ---------------------------------------------------------------------------
# Factor-update batch helpers (2026-07-29: SDDMM-style sampled row-dot and
# scatter-free SpMM for the largedim MU inner loop).
#
# The legacy batch body gathered two (nnz_b, R) dense temporaries per batch
# (A[r_i], Z_u[u_i]) and faked a scatter-add with an (I, nnz_b) CSR whose
# columns were arange(nnz_b). These helpers replace that with:
#   - sampled_row_dots: per-entry <A[r], Z[u]> without materializing either
#     gather (an SDDMM: entries of A @ Z.T at the NNZ pattern);
#   - build_batch_csr_T + spmm_T: one (m, I) CSR against the UNGATHERED Z_u,
#     built sort-free from a ModeGrouping's segment offsets when available.
# Set TENSORMET_LEGACY_FACTOR_BATCH=1 to restore the old batch body (A/B).
# ---------------------------------------------------------------------------

LEGACY_FACTOR_BATCH = os.environ.get("TENSORMET_LEGACY_FACTOR_BATCH", "") not in ("", "0", "false", "False")


def use_legacy_factor_batch():
    """Read the legacy-batch flag at call time, so the validation script can
    toggle ``sparse_ops.LEGACY_FACTOR_BATCH`` in-process for A/B comparison
    (a ``from``-import of the constant would freeze the import-time value)."""
    return LEGACY_FACTOR_BATCH

# Opt-out of the fused kernel (fall back to gather + row-dot) without touching
# the SpMM restructuring: TENSORMET_SAMPLED_DOT=gather.
_SAMPLED_DOT_MODE = os.environ.get("TENSORMET_SAMPLED_DOT", "kernel")

_sampled_row_dot_kernel = None


def _get_sampled_row_dot_kernel():
    global _sampled_row_dot_kernel
    if _sampled_row_dot_kernel is None:
        _sampled_row_dot_kernel = cp.ElementwiseKernel(
            "int64 r, int64 u, raw T A, raw T Z, int64 ncol",
            "T y",
            """
            T acc = (T)0;
            const long long ra = r * ncol;
            const long long za = u * ncol;
            for (long long k = 0; k < ncol; ++k) {
                acc += A[ra + k] * Z[za + k];
            }
            y = acc;
            """,
            "tensormet_sampled_row_dot",
        )
    return _sampled_row_dot_kernel


def sampled_row_dots(A, Z, r_idx, u_idx):
    """Per-entry row dot <A[r_idx[k], :], Z[u_idx[k], :]> -> (nnz_b,).

    Equivalent to ``cp.sum(A[r_idx] * Z[u_idx], axis=1)`` (the entries of the
    dense product ``A @ Z.T`` sampled at ``(r_idx, u_idx)``, i.e. an SDDMM),
    but computed by a fused kernel that never materializes the two (nnz_b, R)
    gathers or the (nnz_b, R) product temporary.

    ``A`` and ``Z`` must have the same number of columns (both are R_mode wide
    in the factor update). Mixed dtypes are promoted to their common dtype.
    """
    if _SAMPLED_DOT_MODE == "gather":
        return cp.sum(A[r_idx] * Z[u_idx], axis=1)
    if A.dtype != Z.dtype:
        common = cp.result_type(A.dtype, Z.dtype)
        A = A.astype(common, copy=False)
        Z = Z.astype(common, copy=False)
    A = cp.ascontiguousarray(A)
    Z = cp.ascontiguousarray(Z)
    kern = _get_sampled_row_dot_kernel()
    return kern(r_idx.astype(cp.int64, copy=False), u_idx.astype(cp.int64, copy=False),
                A, Z, cp.int64(A.shape[1]))


def group_batch_by_column(u_idx, m, *arrays):
    """Sort one batch's per-entry arrays by local column ``u_idx`` and build
    the CSR indptr of the (m, I) transposed batch matrix.

    Uncached-path counterpart of the ModeGrouping segment offsets. Returns
    ``(indptr, u_sorted, *arrays_sorted)``.

    CHANGED (2026-07-29 fix): the uncached path originally built P through the
    COO->CSR constructor, whose internal sort left ``P.data`` in SORTED order
    while per-entry weights handed to ``same_pattern_csr`` (masked FR
    denominator) stayed in ENTRY order — silently misaligning them. Sorting
    the batch up front makes entry order and P.data order identical, exactly
    like the grouping path, so pattern reuse is valid by construction.
    """
    order = cp.argsort(u_idx)
    u_sorted = u_idx[order]
    counts = cp.bincount(u_sorted, minlength=m)
    indptr = cp.concatenate(
        (cp.zeros(1, dtype=counts.dtype), cp.cumsum(counts))
    ).astype(cp.int32)
    return (indptr, u_sorted) + tuple(a[order] for a in arrays)


def build_batch_csr_T(data, r_idx, m, n_rows, indptr):
    """Build the transposed batch matrix P of shape (m, n_rows) for one
    factor-update column batch: row j holds the batch entries of local
    column j, at columns ``r_idx`` with values ``data``.

    ``P.T @ Z_u`` then scatter-adds each entry's weighted Z row into the
    numerator without ever gathering ``Z_u[u_idx]`` (see spmm_T).

    ``data``/``r_idx`` MUST be grouped by local column with ``indptr`` marking
    the runs — straight from the ModeGrouping segment offsets, or from
    group_batch_by_column on the uncached path. Construction is then sort-free
    and ``P.data`` preserves entry order (same_pattern_csr relies on this).
    """
    return cpx_sparse.csr_matrix(
        (data, r_idx.astype(cp.int32, copy=False), indptr),
        shape=(m, n_rows),
    )


def same_pattern_csr(P, data):
    """CSR sharing P's indices/indptr (no copy) with different values."""
    return cpx_sparse.csr_matrix((data, P.indices, P.indptr), shape=P.shape)


# spmm_T backend, resolved on the first call (or explicitly via
# probe_spmm_backends). Candidates, tried in order:
#   "cusparse_f" -> cupyx.cusparse.spmm(P, asfortranarray(B), transa=True)
#                   (several CuPy versions ASSERT the dense operand is
#                   F-contiguous, so this is the variant most likely to work)
#   "cusparse_c" -> same call with a C-contiguous dense operand
#   "operator"   -> P.T @ B (always correct; cupyx may convert internally)
# Set TENSORMET_SPMM_T to any of the three names to pin the backend and skip
# the probe. Rejected candidates and their reasons land in _SPMM_T_PROBE.
_SPMM_T_BACKEND = os.environ.get("TENSORMET_SPMM_T") or None
_SPMM_T_PROBE: dict = {}

# The probe writes a module global that every shard thread reads
# (_sharded_factor_update runs n_shards workers in one process), so it needs a
# lock: without one, two threads can probe concurrently and a third can observe
# _SPMM_T_BACKEND between the assignment and the probe completing.
_SPMM_T_LOCK = threading.Lock()
_SPMM_T_NONCANONICAL = 0        # count of calls that needed canonicalization


def _is_canonical(P) -> bool:
    """Whether P satisfies cuSPARSE's canonical-format precondition.

    CuPy computes ``has_canonical_format`` from the actual indices (sorted
    within each row, no duplicates) rather than tracking it as a construction
    flag, so it is a property of THIS matrix, not of the build path.
    """
    try:
        return bool(P.has_canonical_format)
    except Exception:
        return False


def _canonicalize(P):
    """A canonical copy of P, leaving the caller's P untouched.

    ``sum_duplicates`` sorts indices within each row and adds any duplicate
    (row, col) entries — which is what P.T @ B does with them anyway, so the
    product is unchanged. Copying matters: the caller may still hold P for
    ``same_pattern_csr``, and mutating it in place would leave those reused
    indices sorted while the separately-computed weights stay in entry order,
    reintroducing exactly the misalignment the 2026-07-29 group_batch_by_column
    fix removed.
    """
    Pc = P.copy()
    Pc.sum_duplicates()
    return Pc


def _spmm_call(backend, P, B):
    if backend == "operator":
        return P.T @ B
    import cupyx.cusparse as _cux
    if backend == "cusparse_f":
        return _cux.spmm(P, cp.asfortranarray(B), transa=True)
    if backend == "cusparse_c":
        return _cux.spmm(P, cp.ascontiguousarray(B), transa=True)
    raise ValueError(
        f"unknown spmm_T backend {backend!r} "
        "(valid: 'cusparse_f', 'cusparse_c', 'operator')"
    )


def probe_spmm_backends(P=None, B=None, verbose=False):
    """Validate each cuSPARSE spmm_T candidate against the operator path and
    cache the first one that matches; fall back to "operator" if none does.

    Per-candidate outcomes (exception repr, shape/numeric mismatch, or "ok")
    are recorded in ``_SPMM_T_PROBE`` so a fallback is diagnosable instead of
    silent. Call with no arguments (a tiny built-in example is used) from a
    notebook to inspect availability; ``spmm_T`` calls it lazily otherwise.
    """
    global _SPMM_T_BACKEND
    if P is None:
        rs = np.random.default_rng(0)
        m, n_rows, R = 6, 5, 3
        u = cp.asarray(np.sort(rs.integers(0, m, 20)))
        r = cp.asarray(rs.integers(0, n_rows, 20))
        w = cp.asarray(rs.random(20))
        indptr, u, r, w = group_batch_by_column(u, m, r, w)
        P = build_batch_csr_T(w, r, m, n_rows, indptr)
        B = cp.asarray(rs.random((m, R)))
    ref = P.T @ B
    # Probe the BACKEND, not this particular matrix: feed the cuSPARSE
    # candidates a canonical P so a non-canonical first batch cannot condemn
    # them permanently. spmm_T canonicalizes per call for the same reason.
    P_probe = P if _is_canonical(P) else _canonicalize(P)
    chosen = "operator"
    for backend in ("cusparse_f", "cusparse_c"):
        try:
            out = _spmm_call(backend, P_probe, B)
        except Exception as exc:
            _SPMM_T_PROBE[backend] = f"raised: {exc!r}"
            continue
        if out.shape != ref.shape:
            _SPMM_T_PROBE[backend] = f"wrong shape {out.shape}, expected {ref.shape}"
        elif not bool(cp.allclose(out, ref, rtol=1e-5, atol=1e-8)):
            _SPMM_T_PROBE[backend] = "numeric mismatch vs operator path"
        else:
            _SPMM_T_PROBE[backend] = "ok"
            chosen = backend
            break
    _SPMM_T_BACKEND = chosen
    if verbose:
        for k, v in _SPMM_T_PROBE.items():
            print(f"  {k:12s}: {v}")
        print(f"  -> spmm_T backend: {chosen}")
    return chosen


def spmm_T_backend():
    """The resolved spmm_T backend name, or 'unresolved' before first use."""
    return _SPMM_T_BACKEND or "unresolved"


def spmm_T(P, B):
    """Compute ``P.T @ B`` for CSR P (m, I) and dense B (m, R) -> (I, R).

    Prefers cuSPARSE SpMM with the transposed-A op (no materialized transpose
    or format conversion); the backend is resolved once by probe_spmm_backends
    (or pinned via TENSORMET_SPMM_T) and reused for every call.

    The canonical-format check is per call, deliberately. cupyx.cusparse.spmm
    asserts ``a.has_canonical_format``, and that is a property of the individual
    matrix, not a capability of the backend — so the one-shot probe cannot
    settle it. build_batch_csr_T orders P's columns by the tensor row ids
    ``r_idx`` in entry order, which is ascending-and-unique for some batches and
    not for others; probing on a batch that happened to be canonical and then
    reusing that verdict is what produced the AssertionError at
    cupyx/cusparse.py:1481 on 2026-07-31.
    """
    global _SPMM_T_NONCANONICAL
    if _SPMM_T_BACKEND is None:
        with _SPMM_T_LOCK:
            if _SPMM_T_BACKEND is None:        # re-check: another thread may have probed
                probe_spmm_backends(P, B)
    backend = _SPMM_T_BACKEND
    if backend != "operator" and not _is_canonical(P):
        _SPMM_T_NONCANONICAL += 1
        P = _canonicalize(P)
    return _spmm_call(backend, P, B)


def spmm_T_stats():
    """(backend, calls that needed canonicalization). A large count means the
    sort-free build in build_batch_csr_T is not paying off and the batch would
    be better grouped by (column, row) up front."""
    return spmm_T_backend(), _SPMM_T_NONCANONICAL


def _nndsvd_factors(U: np.ndarray, s: np.ndarray) -> np.ndarray:
    """NNDSVD: convert left-singular vectors U (m × k) and singular values s (k,)
    to a non-negative factor matrix.

    For each column j, selects the positive or negative part of U[:, j] based on
    which has larger norm, then scales so that column norm equals sqrt(s[j]).
    """
    u_pos = np.maximum(U, 0.0)
    u_neg = np.maximum(-U, 0.0)
    mp = np.linalg.norm(u_pos, axis=0)   # (k,)
    mm = np.linalg.norm(u_neg, axis=0)   # (k,)
    use_pos = mp >= mm
    scale = np.where(use_pos, mp, mm)
    direction = np.where(use_pos[np.newaxis, :], u_pos, u_neg)   # (m, k)
    valid = scale > 0.0
    sqrt_scale = np.where(valid, np.sqrt(s / np.where(valid, scale, 1.0)), 0.0)
    return direction * sqrt_scale[np.newaxis, :]


def _nndsvd_factors_gpu(U, s):
    """GPU version of NNDSVD: U (m × k) and s (k,) are CuPy arrays."""
    u_pos = cp.maximum(U, 0.0)
    u_neg = cp.maximum(-U, 0.0)
    mp = cp.linalg.norm(u_pos, axis=0)
    mm = cp.linalg.norm(u_neg, axis=0)
    use_pos = mp >= mm
    scale = cp.where(use_pos, mp, mm)
    direction = cp.where(use_pos[cp.newaxis, :], u_pos, u_neg)
    valid = scale > 0.0
    sqrt_scale = cp.where(valid, cp.sqrt(s / cp.where(valid, scale, 1.0)), 0.0)
    return direction * sqrt_scale[cp.newaxis, :]


# init="svd_loose": ARPACK relative Ritz-value tolerance. The SVD only seeds an
# NNDSVD initialization, so machine precision (tol=0, the eigsh default) buys
# nothing; 1e-4 typically cuts the restart count substantially.
_LOOSE_SVD_TOL = 1e-4

# init="randomised_svd": subspace-iteration knobs shared between the worker and
# the main process (which uses them to precompute the exact tqdm total).
_RSVD_OVERSAMPLING = 10
_RSVD_POWER_ITER = 2
# Cap on the dense (n × chunk) intermediate produced by A.T @ block — matches
# the single n-vector footprint of the eigsh path within a small factor.
_RSVD_BLOCK_BYTES = 4 * 1024**3


def _svd_worker(row_np, col_np, data_np, sp_shape, rank_i, random_state_i, threads_per_mode,
                write_conn, tol=0.0):
    """Top-level worker for ProcessPoolExecutor: truncated SVD on one mode unfolding.

    Uses eigsh on the Gram operator A @ A.T (shape m × m, always small) instead
    of svds. svds recovers right singular vectors via A.T @ U, which for a wide
    unfolding (e.g. 10000 × 100M) produces a 15-billion-element matrix that
    overflows LAPACK's int32 indexing. We only need left singular vectors, so
    the Gram approach is both correct and avoids the overflow entirely.

    Each Gram mat-vec calls A.T @ x (producing an n-vector) then A @ y, and
    sends one increment via write_conn for live tqdm tracking in the main process.
    Pipe connections clean up with os.close() on exit — no IPC, no hang risk.

    tol is ARPACK's relative Ritz-value tolerance (0 = machine precision);
    init="svd_loose" passes _LOOSE_SVD_TOL.
    """
    import scipy.sparse
    import scipy.sparse.linalg
    from contextlib import nullcontext
    from threadpoolctl import threadpool_limits

    sp_cpu = scipy.sparse.coo_matrix(
        (data_np, (row_np, col_np)), shape=sp_shape
    ).tocsr()
    m = sp_shape[0]
    k = min(rank_i, m - 1)
    rng = np.random.RandomState(random_state_i)
    v0 = rng.standard_normal(m)

    def _gram_mv(x):
        write_conn.send(1)
        return sp_cpu @ (sp_cpu.T @ x)   # (m,) → (n,) → (m,)

    gram_op = scipy.sparse.linalg.LinearOperator(
        shape=(m, m), matvec=_gram_mv, dtype=sp_cpu.dtype,
    )

    ctx = threadpool_limits(threads_per_mode) if threads_per_mode is not None else nullcontext()
    with ctx:
        eigvals, eigvecs = scipy.sparse.linalg.eigsh(gram_op, k=k, which='LM', v0=v0, tol=tol)

    desc = np.argsort(eigvals)[::-1]
    U = eigvecs[:, desc]                              # (m, k) left singular vectors
    s = np.sqrt(np.maximum(eigvals[desc], 0.0))       # (k,) singular values
    return _nndsvd_factors(U, s)


def _rsvd_gram_matvec_total(sp_shape, rank_i):
    """Exact number of Gram mat-vecs the randomised SVD worker will perform.

    Computed in the main process too, so the tqdm bar gets a true total:
    (2·q + 2) block applications of the Gram operator — one for the initial
    projection, two per power iteration, one for Rayleigh-Ritz — each costing
    p = k + oversampling column mat-vecs.
    """
    m = sp_shape[0]
    k = min(rank_i, m - 1)
    p = min(k + _RSVD_OVERSAMPLING, m)
    return (2 * _RSVD_POWER_ITER + 2) * p


def _randomised_svd_worker(row_np, col_np, data_np, sp_shape, rank_i, random_state_i,
                           threads_per_mode, write_conn):
    """Top-level worker for ProcessPoolExecutor: randomised truncated SVD on one
    mode unfolding, with a deterministic operation count.

    CPU port of _gpu_top_k_eig applied to the implicit Gram operator A @ A.T:
    randomized subspace iteration + CholeskyQR + Rayleigh-Ritz. Unlike ARPACK
    (init="svd"), the total work is fixed in advance — see
    _rsvd_gram_matvec_total — so the main process can give tqdm a real total.

    The Gram operator is applied to blocks of vectors, chunked so the dense
    (n × chunk) intermediate from A.T @ block stays under _RSVD_BLOCK_BYTES
    (n = product of the other mode dims can reach ~1e8+). One increment is
    sent per column, matching the "mv" unit of the eigsh path.
    """
    import scipy.sparse
    import scipy.linalg
    from contextlib import nullcontext
    from threadpoolctl import threadpool_limits

    sp_cpu = scipy.sparse.coo_matrix(
        (data_np, (row_np, col_np)), shape=sp_shape
    ).tocsr()
    m, n = sp_shape
    k = min(rank_i, m - 1)
    p = min(k + _RSVD_OVERSAMPLING, m)
    dtype = sp_cpu.dtype
    chunk = max(1, min(p, int(_RSVD_BLOCK_BYTES // (n * dtype.itemsize))))

    def _gram_block(X):
        out = np.empty_like(X)
        for j0 in range(0, X.shape[1], chunk):
            j1 = min(j0 + chunk, X.shape[1])
            out[:, j0:j1] = sp_cpu @ (sp_cpu.T @ X[:, j0:j1])
            write_conn.send(j1 - j0)
        return out

    rng = np.random.RandomState(random_state_i)
    ctx = threadpool_limits(threads_per_mode) if threads_per_mode is not None else nullcontext()
    with ctx:
        Y = _gram_block(rng.standard_normal((m, p)).astype(dtype))
        for _ in range(_RSVD_POWER_ITER):
            Y = _gram_block(_gram_block(Y))
            # Normalise columns to prevent overflow/underflow across iterations;
            # the column span — all that matters — is unchanged.
            col_norms = np.sqrt(np.sum(Y.astype(np.float64) ** 2, axis=0, keepdims=True))
            Y = (Y / np.maximum(col_norms, 1e-100)).astype(dtype)

        # CholeskyQR; tiny (p × p) algebra in float64 like the GPU path
        G = (Y.astype(np.float64).T @ Y.astype(np.float64))
        G = (G + G.T) * 0.5
        G += np.eye(p) * (np.abs(np.diag(G)).mean() * 1e-10 + 1e-100)
        L = scipy.linalg.cholesky(G, lower=True)
        L_inv = scipy.linalg.solve_triangular(L, np.eye(p), lower=True)
        # Y^T Y = L L^T  ⇒  Q = Y L^{-T} is orthonormal (Rayleigh-Ritz below
        # relies on this; note _gpu_top_k_eig uses Y @ L^{-1}, which only
        # preserves the span).
        Q = (Y @ L_inv.T.astype(dtype))                    # (m, p) orthonormal

        # Rayleigh-Ritz: one more Gram application, then a tiny eigh
        B = Q.astype(np.float64).T @ _gram_block(Q).astype(np.float64)
        B = (B + B.T) * 0.5
        eigvals, eigvecs = np.linalg.eigh(B)               # ascending

    V = np.ascontiguousarray(eigvecs[:, -k:][:, ::-1])     # top-k, descending
    U = (Q @ V.astype(dtype))                              # (m, k) left singular vectors
    s = np.sqrt(np.maximum(eigvals[-k:][::-1], 0.0)).astype(dtype)
    return _nndsvd_factors(U, s)


def _initialize_svd_tucker_cpu(sparse_tensor, shape, rank, modes, random_state, thread_budget=None,
                               variant="svd", with_core=True):
    """Tucker init via truncated SVD of each mode unfolding (CPU/scipy path).

    Extracts COO data for all mode unfoldings in the main process (sequential,
    fast), then dispatches one worker process per mode via ProcessPoolExecutor.
    Each worker process owns an independent BLAS thread pool sized at
    n_threads // n_modes, so all CPUs are saturated without oversubscription.
    A tqdm bar per mode tracks ARPACK mat-vec iterations via a Queue drained by
    a background thread. Core is computed on GPU after all factors are collected.

    variant :
        "svd"            — ARPACK eigsh at machine precision (tol=0).
        "svd_loose"      — ARPACK eigsh with tol=_LOOSE_SVD_TOL; plenty for an
                           NNDSVD init, converges in fewer restarts.
        "randomised_svd" — randomized subspace iteration with a fixed,
                           precomputable mat-vec count (tqdm bars get a total).
    """
    import multiprocessing
    import threading
    from concurrent.futures import ProcessPoolExecutor, as_completed
    from tqdm import tqdm

    n_modes = len(modes)
    n_threads = thread_budget.n_threads if thread_budget is not None else None
    threads_per_mode = max(1, n_threads // n_modes) if n_threads is not None else None

    # GPU → CPU transfer happens here, sequentially (PCIe, fast relative to SVD)
    int32_max = np.iinfo(np.int32).max
    mode_arrays = []
    for mode in modes:
        # backend="scipy": the thin mode's unfolding can have > int32_max columns,
        # which cupy/cuSPARSE cannot represent (it casts indices to int32 and they
        # wrap negative). scipy holds them with int64 indices on the host, and the
        # SVD worker's Gram approach avoids the wide-matrix int32 overflow.
        unfolded = unfold_from_vectorized_sparse(
            sparse_tensor, shape, mode, to_dense=False, backend="scipy"
        )
        coo = unfolded.tocoo()
        # Keep int32 indices only when both dimensions fit; otherwise int64 to
        # avoid wrapping the (wide) column indices negative.
        idx_dtype = np.int32 if max(coo.shape) <= int32_max else np.int64
        mode_arrays.append((
            np.asarray(coo.row, dtype=idx_dtype),
            np.asarray(coo.col, dtype=idx_dtype),
            np.asarray(coo.data),
            unfolded.shape,
        ))

    def _drain(read_conn, pbar):
        # Exits when the write end of the pipe is fully closed (worker exits +
        # main process closes its copy), which raises EOFError on recv().
        # Workers send the number of mat-vecs per increment (1 for eigsh,
        # chunk width for the randomised worker).
        try:
            while True:
                pbar.update(read_conn.recv())
        except EOFError:
            pass

    # One Pipe per mode. write_conn is passed to the worker (picklable Connection).
    # Cleanup is a plain os.close() — no manager process, no RPC, no hang risk.
    pipe_pairs = [multiprocessing.Pipe(duplex=False) for _ in range(n_modes)]

    randomised = variant == "randomised_svd"
    # ARPACK's iteration count is convergence-dependent (no total); the
    # randomised worker's is fixed in advance.
    totals = [
        _rsvd_gram_matvec_total(mode_arrays[i][3], rank[i]) if randomised else None
        for i in range(n_modes)
    ]
    bars = [
        tqdm(desc=f"SVD mode {i}", position=i, leave=True, unit="mv", dynamic_ncols=True,
             total=totals[i])
        for i in range(n_modes)
    ]
    drain_threads = [
        threading.Thread(target=_drain, args=(r, b), daemon=True)
        for (r, _w), b in zip(pipe_pairs, bars)
    ]
    for t in drain_threads:
        t.start()

    factors = [None] * n_modes
    worker = _randomised_svd_worker if randomised else _svd_worker
    worker_extra = () if randomised else (_LOOSE_SVD_TOL if variant == "svd_loose" else 0.0,)
    with ProcessPoolExecutor(max_workers=n_modes) as pool:
        futures = {
            pool.submit(
                worker,
                row, col, data, sp_shape,
                rank[i], random_state + i, threads_per_mode, pipe_pairs[i][1],
                *worker_extra,
            ): i
            for i, (row, col, data, sp_shape) in enumerate(mode_arrays)
        }
        for fut in as_completed(futures):
            i = futures[fut]
            pipe_pairs[i][1].close()      # close main's write end; worker exit closes
            factors[i] = cp.asarray(fut.result())   # its copy → drain thread gets EOFError

    for t in drain_threads:
        t.join()
    for b in bars:
        b.close()

    core = None
    if with_core:
        core = _compute_tucker_core_batched(sparse_tensor, shape, factors, modes)
        core = cp.clip(cp.abs(core), a_min=1e-30, a_max=None)
    factors = [cp.clip(cp.abs(f), a_min=1e-30, a_max=None) for f in factors]
    return core, factors


def _compute_tucker_core_batched(sparse_tensor, shape, factors, modes):
    """Compute G = X ×_n U_n^T for all modes without materialising any large intermediate.

    Uses the same NNZ-batched outer-product accumulation as kl_core_update_largedim:
      G = Σ_{nz} X[nz] · outer(U0[i0,:], U1[i1,:], ..., UN[iN,:])

    This avoids the O(R × prod_other) dense allocation that sparse_multi_mode_dot_vec
    would require for large dimensions (e.g. 60 GB for dim=10000, order=3, rank=150).
    Batch size is estimated from free GPU memory via _estimate_batch_num_for_outer.
    """
    from tensormet.distance import (
        coo_to_coords,
        _accumulate_core_num_outer,
        _estimate_batch_num_for_outer,
    )
    N = len(modes)
    core_shape = tuple(factors[n].shape[1] for n in range(N))
    core = cp.zeros(core_shape, dtype=factors[0].dtype)

    idxs, xvals = coo_to_coords(sparse_tensor, shape)
    nnz = int(xvals.size)
    if nnz == 0:
        return core

    batch_num = _estimate_batch_num_for_outer(core, factors)
    for start in range(0, nnz, int(batch_num)):
        end = min(start + int(batch_num), nnz)
        mats = [factors[n][idxs[n][start:end]] for n in range(N)]
        _accumulate_core_num_outer(core, xvals[start:end], mats)

    return core


def _gpu_top_k_eig(C, k, n_oversampling=10, n_power_iter=3, seed=None):
    """Top-k eigenpairs of a symmetric PSD matrix C (m × m) using only cuBLAS.

    Algorithm: randomized subspace iteration + CholeskyQR + Rayleigh-Ritz.
    cuSOLVER is never called. The only CPU work is a (k+p)×(k+p) Cholesky
    and eigh — matrices of size ~60×60 for typical ranks, negligible cost.

    Steps:
      1. Y = C @ Ω          random projection           (cuBLAS, float64)
      2. Y = C @ (C @ Y)    power iterations with column normalisation
      3. Q = CholeskyQR(Y)  tiny CPU Cholesky on (p×p) + GPU matmul
      4. B = Q.T @ C @ Q    Rayleigh-Ritz               (cuBLAS, result tiny)
      5. eigh(B)            on CPU, B is (p×p)
      6. U = Q @ V          map eigenvectors back         (cuBLAS)

    Column normalisation after each power iteration is essential: without it,
    the dominant eigenvalue is amplified as λ^(2t+1) per iteration, overflowing
    float32 (and even float64) for large Gram matrices. Normalisation preserves
    the column span — which is all that matters — while keeping values finite.
    All power-iteration arithmetic is done in float64 to prevent cancellation.
    """
    import scipy.linalg

    m = C.shape[0]
    p = k + n_oversampling
    orig_dtype = C.dtype

    # Upcast to float64 for numerical stability throughout
    C64 = C.astype(cp.float64)

    rng = cp.random.RandomState(seed)
    Y = rng.standard_normal((m, p))   # float64

    Y = C64 @ Y
    for _ in range(n_power_iter):
        Y = C64 @ (C64 @ Y)
        # Normalise columns to prevent overflow/underflow across iterations.
        # The column span of Y is unchanged by this scaling.
        col_norms = cp.sqrt(cp.sum(Y ** 2, axis=0, keepdims=True))
        Y /= cp.maximum(col_norms, 1e-100)

    # CholeskyQR: Y.T @ Y is (p×p). After column normalisation the diagonal
    # is ≈ 1 so G is well-conditioned and Cholesky succeeds reliably.
    G_cpu = cp.asnumpy(Y.T @ Y)
    G_cpu = (G_cpu + G_cpu.T) * 0.5                         # symmetrize
    G_cpu += np.eye(p) * (np.abs(np.diag(G_cpu)).mean() * 1e-10 + 1e-100)
    L = scipy.linalg.cholesky(G_cpu, lower=True)
    L_inv = scipy.linalg.solve_triangular(L, np.eye(p), lower=True)
    Q = Y @ cp.asarray(L_inv)                               # (m, p) float64

    # Rayleigh-Ritz: project C onto the p-dim subspace
    B_cpu = cp.asnumpy(Q.T @ C64 @ Q)                      # (p, p) — tiny
    B_cpu = (B_cpu + B_cpu.T) * 0.5
    eigvals_cpu, eigvecs_cpu = np.linalg.eigh(B_cpu)

    # Top-k in descending order; cast back to original dtype
    eigvals = eigvals_cpu[-k:][::-1].copy()
    V = cp.asarray(np.ascontiguousarray(eigvecs_cpu[:, -k:][:, ::-1]))
    U = (Q @ V).astype(orig_dtype)                          # (m, k)
    s = cp.sqrt(cp.maximum(cp.asarray(eigvals), 0.0)).astype(orig_dtype)
    return U, s


def _initialize_svd_tucker_gpu(sparse_tensor, shape, rank, modes, random_state, with_core=True):
    """Tucker init via Gram-matrix eigendecomposition (GPU path).

    Computes A @ A.T for each mode unfolding A via cuSPARSE spgemm, then
    extracts the top-k eigenpairs with _gpu_top_k_eig (randomized subspace
    iteration — cuBLAS only, no cuSOLVER). Core is computed via the same
    NNZ-batched accumulation used by kl_core_update_largedim.
    """
    factors = []
    for i, mode in enumerate(modes):
        unfolded = unfold_from_vectorized_sparse(sparse_tensor, shape, mode, to_dense=False)
        A = unfolded.tocsr()
        del unfolded

        # A @ A.T: (I_mode × I_mode) via cuSPARSE spgemm — never touches prod_other densely
        C = A @ A.T
        del A
        if cpx_sparse.isspmatrix(C):
            C = C.toarray()

        k = rank[i]
        U, s = _gpu_top_k_eig(C, k, seed=random_state + i)
        del C

        factors.append(_nndsvd_factors_gpu(U, s))

    core = None
    if with_core:
        core = _compute_tucker_core_batched(sparse_tensor, shape, factors, modes)
        core = cp.clip(cp.abs(core), a_min=1e-30, a_max=None)
    factors = [cp.clip(cp.abs(f), a_min=1e-30, a_max=None) for f in factors]
    return core, factors


def initialize_nonnegative_tucker(sparse_tensor, shape, rank, modes, init, random_state,
                                   thread_budget=None, with_core=True):
    """with_core=False returns (None, factors) and skips the core computation —
    for decompositions that never form a dense R^N core (experimental/TT_hybrid)."""
    if init == "random":
        rng = tl.check_random_state(random_state)
        core = tl.tensor(
            rng.random_sample([rank[i] for i in range(len(modes))]) + 0.01,
            **tl.context(sparse_tensor),
        ) if with_core else None
        factors = [
            tl.tensor(rng.random_sample((shape[mode], rank[i])), **tl.context(sparse_tensor))
            for i, mode in enumerate(modes)
        ]
    elif init in ("svd", "svd_cpu", "svd_loose", "randomised_svd", "randomized_svd"):
        variant = {
            "svd": "svd", "svd_cpu": "svd", "svd_loose": "svd_loose",
            "randomised_svd": "randomised_svd", "randomized_svd": "randomised_svd",
        }[init]
        return _initialize_svd_tucker_cpu(sparse_tensor, shape, rank, modes, random_state,
                                          thread_budget=thread_budget, variant=variant,
                                          with_core=with_core)
    elif init == "svd_gpu":
        return _initialize_svd_tucker_gpu(sparse_tensor, shape, rank, modes, random_state,
                                          with_core=with_core)
    else:
        core, factors = init

    factors = [tl.clip(tl.abs(f), a_min=1e-30, a_max=None) for f in factors]
    if core is not None:
        core = tl.clip(tl.abs(core), a_min=1e-30, a_max=None)
    return core, factors