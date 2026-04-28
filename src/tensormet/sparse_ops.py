import tensorly as tl
import pytensorlab as ptl
import numpy as np
from typing import List, Tuple, Optional, Union
import math
from tensormet.utils import einsum_letters, make_lazy_cupy_pair
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

def unfold_from_vectorized_sparse(
    vec_tensor: cpx_sparse.spmatrix,
    orig_shape,
    mode: int,
    to_dense: bool = False,
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

    Returns
    -------
    unfolded : cupy.ndarray or cupyx.scipy.sparse.coo_matrix
        Mode-`mode` unfolding of shape
        (orig_shape[mode], np.prod(orig_shape) // orig_shape[mode]).
    """
    # Make sure we're in COO format

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

    row_unf_cp = cp.asarray(row_unf_np)
    col_unf_cp = cp.asarray(col_unf_np)

    unfolded_shape = (orig_shape[mode], int(math.prod(other_shape)))
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
    orig_shape = tuple(orig_shape)
    # new: use math.prod instead of numpy
    size = math.prod(orig_shape)
    int32_max = np.iinfo(np.int32).max
    block_size = min(size, int32_max)

    dense_flat = dense_nd.reshape(size, order="C")
    coo = vec_tensor.tocoo()
    flat = coo.row + coo.col * block_size
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

    # 1. Introduce the batch dimension (m) with the first mode
    k0 = other_modes[0]
    M0 = factors[k0][idxs_by_mode[k0]]  # (m, R_k0)
    m = M0.shape[0]

    # tmp shape: (m, R_0, ..., R_{k0-1}, R_{k0+1}, ...)
    tmp = cp.tensordot(M0, core, axes=(1, k0))

    # Track the remaining core axes in their current order
    remaining_axes = [i for i in range(core.ndim) if i != k0]

    # 2. Contract the rest with broadcast+sum to prevent multiplying the 'm' dimension
    for k in other_modes[1:]:
        M = factors[k][idxs_by_mode[k]]  # (m, R_k)

        # Find where axis k is currently located in tmp (shifted by +1 because 'm' is at index 0)
        axis_idx = 1 + remaining_axes.index(k)

        # Reshape M to broadcast across tmp: (m, 1, ..., R_k, ..., 1)
        bcast_shape = [1] * tmp.ndim
        bcast_shape[0] = m
        bcast_shape[axis_idx] = M.shape[1]

        # Multiply and sum over the rank dimension
        tmp = cp.sum(tmp * M.reshape(bcast_shape), axis=axis_idx)

        # Remove the contracted axis from tracking
        remaining_axes.remove(k)

    if tmp.ndim != 2:
        raise RuntimeError(f"Expected 2D result after contractions, got shape {tmp.shape}")

    return cp.clip(tmp, a_min=epsilon, a_max=None)


def initialize_nonnegative_tucker(sparse_tensor, shape, rank, modes, init, random_state):
    if init == "random":
        rng = tl.check_random_state(random_state)
        core = tl.tensor(
            rng.random_sample([rank[i] for i in range(len(modes))]) + 0.01,
            **tl.context(sparse_tensor),
        )
        factors = [
            tl.tensor(rng.random_sample((shape[mode], rank[i])), **tl.context(sparse_tensor))
            for i, mode in enumerate(modes)
        ]
    else:
        core, factors = init


    factors = [tl.clip(tl.abs(f), a_min=1e-30, a_max=None) for f in factors]
    core = tl.clip(tl.abs(core), a_min=1e-30, a_max=None)
    return core, factors