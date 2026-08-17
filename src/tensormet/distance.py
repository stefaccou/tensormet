from __future__ import annotations
import math
import itertools
from dataclasses import dataclass

from tqdm import tqdm
import numpy as np
# import cupy as cp
# import cupyx.scipy.sparse as cpx_sparse

import tensorly as tl

from tensorly.base import unfold
from tensorly.tucker_tensor import tucker_to_tensor
from tensorly.tenalg import mode_dot

from tensormet.sparse_ops import (
    unfold_from_vectorized_sparse,
    sparse_multi_mode_dot_vec,
    ptl_tucker_to_tensor,
    gather_dense_at_block_nz,
    safe_ravel,
    CoordCOO,
    require_matrix_form,
    compute_Zcols_batch,
    use_legacy_factor_batch,
    use_peel_contraction,
    sampled_row_dots,
    group_batch_by_column,
    build_batch_csr_T,
    same_pattern_csr,
    spmm_T,
)
from tensormet.utils import ThreadBudget, einsum_letters, cp_einsum_optimize, make_lazy_cupy_pair, lazy_import

cp, cpx_sparse = make_lazy_cupy_pair()

# pytensorlab transitively imports TensorFlow + VTK (~200s); defer it until used.
ptl = lazy_import("pytensorlab")

# -- Kullback-Leibler Divergence --

def kl_factor_update(vec_tensor, core, factors, mode, shape, thread_budget=None, epsilon=1e-12, verbose=False):
    """
    One multiplicative KL update for a single factor matrix A_n (for `mode`).

    Parameters
    ----------
    vec_tensor : cupyx.scipy.sparse.coo_matrix
        Vectorized sparse tensor (COO).
    shape : tuple[int, ...]
        Original tensor shape.
    core : cupy.ndarray
        Tucker core on GPU (CuPy).
    factors : list[cupy.ndarray]
        Tucker factors on GPU (CuPy).
    mode : int
        Mode to update.
    epsilon : float
        Small positive constant for numerical stability / nonnegativity.

    Returns
    -------
    A : updated factor
    """

    if verbose:
        print(f"  Updating factor {mode}...")

    # Sparse unfolding for this mode
    X = unfold_from_vectorized_sparse(vec_tensor, shape, mode)

    # Dense reconstruction excluding current factor, unfolded along mode
    Z = tucker_to_tensor((core, factors), skip_factor=mode)
    Z = unfold(Z, mode)  # (R, K) after unfold

    A = factors[mode]  # (I_mode, R)
    rows = X.row
    cols = X.col
    vals = X.data

    # Compute reconstruction only at nonzeros: R_nz = sum_r A[i,r] * Z[r,j]
    # A_rows: (nnz, R)
    A_rows = A[rows, :]
    # Z_cols_T: (nnz, R) because Z[:, cols] is (R, nnz)
    Z_cols_T = tl.transpose(Z[:, cols])
    R_nz = tl.sum(A_rows * Z_cols_T, axis=1)
    R_nz = tl.clip(R_nz, a_min=epsilon, a_max=None)

    # W = X / (A Z) at nonzeros
    W_data = vals / R_nz
    W = cpx_sparse.coo_matrix((W_data, (rows, cols)), shape=X.shape)

    # numerator = W @ Z^T   -> (I_mode, R)
    numerator = W @ tl.transpose(Z)

    # denominator = sum_j Z[r,j] broadcast to (I_mode, R)
    den_row = tl.sum(Z, axis=1)  # (R,)
    denominator = den_row[np.newaxis, :]
    denominator = tl.clip(denominator, a_min=epsilon, a_max=None)

    # Multiplicative update
    A = A * (numerator / (denominator + 1e-12))
    A = tl.clip(A, a_min=epsilon, a_max=None)
    return A


def kl_core_update(vec_tensor, shape, core, factors, modes, thread_budget, epsilon=1e-12, verbose=False):
    """
    One multiplicative KL update for the core tensor.

    Mirrors the sequence:
      - build dense reconstruction on CPU via ptl
      - gather reconstructed values at nonzero blocks
      - form X/R sparse ratio tensor
      - compute X_R = (X/R) ×_n W_n^T
      - compute F from column sums of factors
      - core *= X_R / F
      - normalize

    Returns
    -------
    core : updated core.
    """
    if verbose:
        print("  Updating core...")

    # Build a CPU Tucker object for pytensorlab reconstruction
    core_np = tl.to_numpy(core)
    factors_np = [tl.to_numpy(f) for f in factors]
    tucker = ptl.TuckerTensor(core=core_np, factors=factors_np)

    with thread_budget.limit():
        R = ptl_tucker_to_tensor(tucker)

    # Gather reconstructed values at nonzero coordinates of vec_tensor
    data = gather_dense_at_block_nz(R, vec_tensor, shape)
    data = tl.clip(data, a_min=epsilon, a_max=None)

    # X/R at nonz
    X_R_data = vec_tensor.data / cp.asarray(data)
    X_R = cpx_sparse.coo_matrix(
        (X_R_data, (vec_tensor.row, vec_tensor.col)),
        shape=vec_tensor.shape,
    )

    # (X/R) ×_n W_n^T   -> core-shaped tensor
    X_R = sparse_multi_mode_dot_vec(
        vec_tensor=X_R,
        orig_shape=shape,
        factors=factors,
        modes=modes,
        transpose_factors=True,
    )
    X_R = tl.clip(X_R, a_min=epsilon, a_max=None)

    # F = outer product of column sums of factors, broadcast to core shape
    col_sums = [tl.sum(A_n, axis=0) for A_n in factors]
    F = col_sums[0].reshape((core.shape[0],) + (1,) * (core.ndim - 1))
    for n in range(1, core.ndim):
        shape_n = [1] * n + [core.shape[n]] + [1] * (core.ndim - n - 1)
        F = F * col_sums[n].reshape(tuple(shape_n))
    F = tl.clip(F, a_min=epsilon, a_max=None)

    # Multiplicative core update
    new_core = core * X_R / (F + epsilon)
    return new_core

def kl_compute_errors(
        vec_tensor: cpx_sparse.spmatrix,
        shape,
        core,
        factors,
        thread_budget: ThreadBudget,
        epsilon=1e-12,
        verbose=False,
):

    """Generalised KL divergence C_KL(X || R) for sparse X.

    vec_tensor : vectorised sparse X (block-encoded, same as 'tensor')
    shape      : original N-D shape
    core       : current core G
    factors    : list of factor matrices A^{(n)}
    """

    if verbose:
        print("  Computing KL errors...")

    core_np = tl.to_numpy(core)
    factors_np = [tl.to_numpy(f) for f in factors]

    tucker = ptl.TuckerTensor(core=core_np,
                                  factors=factors_np)
    with thread_budget.limit():
        R = ptl_tucker_to_tensor(tucker)

    shape = tuple(shape)

    # --- 1) Dense reconstruction R = G ×_1 A^{(1)} × ... ×_N A^{(N)} ---
    # This is exactly step 3 in Table 2 of the paper.
    # This breaks with dims over 1000
    # R = tucker_to_tensor((core, factors))      # cp.ndarray, shape=shape
    # R = tl.clip(R, a_min=epsilon, a_max=None)
    # R_flat = R.ravel()                         # length = size

    r_nz = gather_dense_at_block_nz(R, vec_tensor, shape)
    r_nz = tl.clip(r_nz, a_min=epsilon, a_max=None)
    # --- 2) Decode sparse X indices to flat indices ---
    X_coo = vec_tensor.tocoo()
    x_nz = X_coo.data

    # --- 3) X_i and R_i at nonzero entries ---
    # the original data can still contain harmful zeros
    x_nz = tl.clip(x_nz, a_min=epsilon, a_max=None)
    r_nz = cp.asarray(r_nz)

    # --- 4) KL contribution from nonzeros ---
    # sum_{i: X_i>0} [X_i log(X_i/R_i) - X_i + R_i]
    term_pos = x_nz * cp.log(x_nz / r_nz) - x_nz + r_nz
    kl_pos = cp.sum(term_pos)
    # --- 5) KL contribution from zero entries ---
    # For X_i = 0, the KL term tends to R_i (limit X→0).
    # Full sum over all i is:
    #   ∑_i [X_i log(X_i/R_i) - X_i + R_i]
    # = (sum over nonzeros) + (sum over zeros),
    # and sum_{zeros} R_i = sum(R) - sum_{nonzeros} R_i.
    sum_R = cp.sum(R)
    sum_R_nz = cp.sum(r_nz)
    kl_zero = sum_R - sum_R_nz
    kl_total = kl_pos + kl_zero

    # --- 6) Optional normalized "relative" KL error ---
    sum_X = cp.sum(x_nz)  # sum over nonzero X
    rel_kl = kl_total / cp.maximum(sum_X, epsilon)
    # print(f"KL divergence: {kl_total}, relative KL: {rel_kl}")
    return rel_kl


# -- Frobenius Norm --
def fr_factor_update(vec_tensor, core, factors, mode, shape, thread_budget=None, epsilon=1e-12, verbose=False):
    if verbose:
        print(f"  Updating factor {mode}...")

    # This still explodes! The created Z tensor does not always fit in memory
    X = unfold_from_vectorized_sparse(vec_tensor, shape,
                                             mode)  # this is the same as B when using dense tensor!
    Z = tucker_to_tensor((core, factors), skip_factor=mode)
    Z = tl.transpose(unfold(Z, mode))
    numerator = X @ Z  # cupy sparse @ dense
    numerator = tl.clip(numerator, a_min=epsilon, a_max=None)
    A = factors[mode]
    denominator = tl.dot(A, tl.dot(tl.transpose(Z), Z))
    denominator = tl.clip(denominator, a_min=epsilon, a_max=None)
    A *= numerator / denominator
    A = tl.clip(A, a_min=epsilon, a_max=None)
    return A


def fr_core_update(vec_tensor, shape, core, factors, modes, thread_budget=None, epsilon=1e-12, verbose=False):
    """
    One multiplicative update for the core tensor.

    [DESCRIBE]

    Returns
    -------
    core : updated core.
    """

    if verbose:
        print("  Updating core...")

    numerator = sparse_multi_mode_dot_vec(
        vec_tensor=vec_tensor,
        orig_shape=shape,
        factors=factors,
        modes=modes,
        transpose_factors=True,  # X ×_n W_n^T
    )
    # we clip the numerator
    numerator = tl.clip(numerator, a_min=epsilon, a_max=None)
    # these operations can again be done with the dense implementation
    for i, f in enumerate(factors):
        if i:
            denominator = mode_dot(denominator, tl.dot(tl.transpose(f), f), i)
        else:
            denominator = mode_dot(core, tl.dot(tl.transpose(f), f), i)
    denominator = tl.clip(denominator, a_min=epsilon, a_max=None)

    new_core = core * numerator / (denominator + epsilon)
    return new_core

def fr_compute_errors(
        vec_tensor: cpx_sparse.spmatrix,
        shape,
        core,
        factors,
        thread_budget: ThreadBudget,
        epsilon=1e-12,
        verbose=False,
):
    """Relative Frobenius error ||X - X̂||_F / ||X||_F for sparse X.

    vec_tensor : vectorised sparse X (block-encoded, same as 'tensor')
    shape      : original N-D shape
    core       : current core G
    factors    : list of factor matrices A^{(n)}
    """

    require_matrix_form(vec_tensor, "fr_compute_errors")

    if verbose:
        print("  Computing Frobenius errors...")

    shape = tuple(shape)

    # --- ||X||_F ---
    X_coo = vec_tensor.tocoo()
    x_nz = X_coo.data
    x_nz = tl.clip(x_nz, a_min=0.0, a_max=None)  # Frobenius is fine with zeros; keep nonneg pipeline consistent
    norm_X_sq = cp.sum(x_nz * x_nz)
    norm_X = cp.sqrt(cp.maximum(norm_X_sq, epsilon))

    # --- <X, X̂> = sum_{nz} X_i * X̂_i ---
    core_np = tl.to_numpy(core)
    factors_np = [tl.to_numpy(f) for f in factors]
    tucker = ptl.TuckerTensor(core=core_np, factors=factors_np)

    with thread_budget.limit():
        R_cpu = ptl_tucker_to_tensor(tucker)  # dense on CPU

    xhat_nz = gather_dense_at_block_nz(R_cpu, vec_tensor, shape)
    xhat_nz = cp.asarray(tl.clip(xhat_nz, a_min=epsilon, a_max=None))

    inner_prod = cp.sum(x_nz * xhat_nz)

    # --- ||X̂||_F^2 without forming X̂ ---
    # ||X̂||_F^2 = <G, G ×_n (A_n^T A_n)>
    denom = core
    for mode, A in enumerate(factors):
        AtA = tl.dot(tl.transpose(A), A)
        denom = mode_dot(denom, AtA, mode)

    denom = tl.clip(denom, a_min=epsilon, a_max=None)
    norm_Xhat_sq = cp.sum(core * denom)

    # --- ||X - X̂||_F^2 = ||X||_F^2 + ||X̂||_F^2 - 2<X, X̂> ---
    residual_sq = norm_X_sq + norm_Xhat_sq - 2.0 * inner_prod
    residual_sq = cp.maximum(residual_sq, 0.0)
    residual_norm = cp.sqrt(residual_sq)

    relative_error = residual_norm / norm_X
    return relative_error

def fr_combined_core_errors(vec_tensor, shape, core, factors, modes, thread_budget=None, epsilon=1e-12, verbose=False):
    """
        One multiplicative KL update for the core tensor.

        [DESCRIBE]

        Returns
        -------
        core : updated core.
        """

    if verbose:
        print("  Computing combined core/errors...")

    numerator = sparse_multi_mode_dot_vec(
        vec_tensor=vec_tensor,
        orig_shape=shape,
        factors=factors,
        modes=modes,
        transpose_factors=True,  # X ×_n W_n^T
    )
    # we clip the numerator
    numerator = tl.clip(numerator, a_min=epsilon, a_max=None)
    # these operations can again be done with the dense implementation
    for i, f in enumerate(factors):
        if i:
            denominator = mode_dot(denominator, tl.dot(tl.transpose(f), f), i)
        else:
            denominator = mode_dot(core, tl.dot(tl.transpose(f), f), i)
    denominator = tl.clip(denominator, a_min=epsilon, a_max=None)

    new_core = core * numerator / (denominator + epsilon)

    # error_start = time.time()
    tensor_coo = vec_tensor.tocoo()
    norm_tensor = cp.sqrt((cp.abs(tensor_coo.data) ** 2).sum())
    # norm_time = print_elapsed_time(error_start, "norm calculation")
    norm_X_sq = norm_tensor ** 2
    norm_Xhat_sq = tl.sum(new_core * denominator)
    inner_prod = tl.sum(numerator * new_core)
    residual_norm = tl.sqrt(norm_X_sq + norm_Xhat_sq - 2 * inner_prod)
    relative_error = residual_norm / norm_tensor
    # end = print_elapsed_time(norm_time, "full error calculation")
    return new_core, relative_error

def null_compute_errors(vec_tensor: cpx_sparse.spmatrix,
        shape,
        core,
        factors,
        thread_budget: ThreadBudget,
        epsilon=1e-12,
        verbose=False,
) -> None:
    # takes the same input, but returns nothing
    return




def _unravel_cols_for_mode(cols, shape, mode):
    """
    Convert unfolding column indices -> per-mode indices for all modes != `mode`,
    consistent with the 3-way decoding you used (last remaining mode varies fastest).

    Returns
    -------
    other_modes : list[int]
    idxs       : dict[int, cupy.ndarray]  # maps mode -> (len(cols),) indices
    """
    N = len(shape)
    other_modes = [m for m in range(N) if m != mode]
    other_dims = [shape[m] for m in other_modes]

    u = cols
    idxs_rev = []
    # last remaining mode varies fastest => mod/div in reverse order
    for dim in reversed(other_dims):
        idxs_rev.append(u % dim)
        u = u // dim

    idxs = list(reversed(idxs_rev))
    return other_modes, {m: idxs[i] for i, m in enumerate(other_modes)}

def _tucker_den_row_full(core, factors, mode, epsilon=1e-12):
    """
    Exact denominator vector for KL MU update:
        den_row[r_mode] = sum_over_all_unfolding_columns Z[r_mode, col]
    without forming Z, for arbitrary N-way Tucker.

    core:  (R0, R1, ..., R_{N-1})
    factors[k]: (Ik, Rk)
    """
    N = core.ndim
    letters = einsum_letters(N)
    core_subs = "".join(letters)

    # s_k[r_k] = sum_i A^{(k)}[i, r_k]
    sums = [cp.sum(factors[k], axis=0) for k in range(N)]
    # einsum: core[a b c ...], sum_b[b], sum_c[c], ... -> output over mode letter
    in_terms = [core_subs] + [letters[k] for k in range(N) if k != mode]
    out_term = letters[mode]
    eq = ",".join(in_terms) + "->" + out_term

    operands = [core] + [sums[k] for k in range(N) if k != mode]
    den_row = cp.einsum(eq, *operands, optimize=cp_einsum_optimize(len(operands)))
    den_row = cp.clip(den_row, a_min=epsilon, a_max=None)
    return den_row



def _unravel_flat_indices_C(flat, shape):
    """
    flat : (m,) cupy int64
    shape: tuple of dims (I0, I1, ..., I_{N-1})

    Returns
    -------
    idxs : list of cupy arrays, each (m,)
           idxs[n] are indices along mode n.
    """
    shape = tuple(int(s) for s in shape)
    N = len(shape)
    u = flat
    idxs_rev = []
    for dim in reversed(shape):        # last mode fastest
        dim = int(dim)
        idxs_rev.append(u % dim)
        u = u // dim
    return list(reversed(idxs_rev))


def _rhat_from_factor_rows_sequential(core, mats, epsilon=1e-12):
    """
    core: (R0, R1, ..., R_{N-1})
    mats[n]: (b, Rn) factor rows for each mode at the b coordinates

    Returns
    -------
    r_hat : (b,)

    The einsum body is the default: the 2026-08-03/04 bisect measured its
    2026-07-30 peel rewrite (the branch below) as part of a ~2x iteration-time
    regression. The peel is kept behind TENSORMET_PEEL_CONTRACTION=1 because
    it bounds the einsum's machine-dependent path choice: the einsum path
    depends on ``b`` (derived from free VRAM), and once materialized a
    (b, R0, ..., R_{N-1}) intermediate on an 80 GB node at rank 100
    (b ≈ 166k). If this contraction ever OOMs, flip the flag rather than
    shrinking the batch estimate.
    """
    if use_peel_contraction():
        # Memory-bounded fallback: fixed-order peel, largest live array is
        # (b, prod(R[1:])) — what _estimate_batch_rhat_for_tensordot budgets
        # for — at the cost of b-batched (1, Rn) x (Rn, rest) GEMVs.
        N = core.ndim
        b = int(mats[0].shape[0])
        tmp = mats[0] @ core.reshape(core.shape[0], -1)
        for n in range(1, N):
            Rn = int(mats[n].shape[1])
            tmp = cp.matmul(mats[n][:, None, :], tmp.reshape(b, Rn, -1))[:, 0, :]
        return cp.clip(tmp.reshape(b), a_min=epsilon, a_max=None)

    N = core.ndim
    letters = einsum_letters(N)
    core_sub = "".join(letters)                   # e.g. 'abc' for N=3
    mat_subs = ["i" + l for l in letters]         # e.g. ['ia', 'ib', 'ic']
    eq = core_sub + "," + ",".join(mat_subs) + "->i"
    r_hat = cp.einsum(eq, core, *mats, optimize=cp_einsum_optimize(1 + N))
    return cp.clip(r_hat, a_min=epsilon, a_max=None)

def _accumulate_core_num_outer(Num, w, mats):
    """
    Optimized core accumulator using Khatri-Rao products and cuBLAS Matrix Multiplication.
    Replaces the slow, massively expanding outer product loop.
    """
    N = len(mats)
    nnz = w.shape[0]
    if nnz == 0: return

    if N == 1:
        Num += cp.sum(w[:, None] * mats[0], axis=0)
        return
    if N == 2:
        Num += (mats[0] * w[:, None]).T @ mats[1]
        return

    # 1. Split modes into Left, Right, and Loop
    # We want Left and Right KR products to fit well within memory (budget ~400MB)
    budget_elements = 100_000_000

    left_modes = []
    left_size = 1
    for i in range(N):
        if left_size * mats[i].shape[1] * nnz < budget_elements:
            left_modes.append(i)
            left_size *= mats[i].shape[1]
        else:
            break

    right_modes = []
    right_size = 1
    for i in reversed(range(len(left_modes), N)):
        if right_size * mats[i].shape[1] * nnz < budget_elements:
            right_modes.append(i)
            right_size *= mats[i].shape[1]
        else:
            break
    right_modes = right_modes[::-1]

    loop_modes = [i for i in range(N) if i not in left_modes and i not in right_modes]

    # Edge Cases
    if not left_modes:
        left_modes = [0]
        if 0 in loop_modes: loop_modes.remove(0)
        if 0 in right_modes: right_modes.remove(0)
    if not right_modes and len(loop_modes) > 0:
        right_modes = [loop_modes.pop()]
    # When nnz is small the budget can absorb every mode into the left block,
    # leaving right/loop empty. The `if not loop_modes` contraction below needs a
    # non-None KR_R, so peel the highest left mode off into the right block.
    if not right_modes and not loop_modes and len(left_modes) > 1:
        right_modes = [left_modes.pop()]

    # 2. Build Khatri-Rao matrices for Left and Right
    def build_KR(modes):
        if not modes: return None
        res = mats[modes[0]]
        for i in modes[1:]:
            res = (res[:, :, None] * mats[i][:, None, :]).reshape(nnz, -1)
        return res

    KR_L = build_KR(left_modes)
    KR_R = build_KR(right_modes)

    # 3. Contract using matrix multiplication
    if not loop_modes:
        slice_sum = (KR_L * w[:, None]).T @ KR_R
        Num += slice_sum.reshape([mats[i].shape[1] for i in left_modes + right_modes])
        return

    # Absorb loop modes into the left/right KR blocks (ceil/floor split) then do a
    # single matmul.  The multi-operand einsum used previously let CuPy's path
    # optimizer produce gigantic intermediates of shape (nnz, L, R, a) or larger —
    # the same 1.45 TB OOM seen for order-4 tensors with high ranks.
    # After absorption the peak intermediate is (L_ext, R_ext) = core-sized,
    # matching what _estimate_batch_num_for_outer already budgets for.
    n_loop = len(loop_modes)
    n_left_loop  = (n_loop + 1) // 2   # ceil — absorbed into left KR
    n_right_loop = n_loop // 2          # floor — absorbed into right KR
    left_loop_modes  = loop_modes[:n_left_loop]
    right_loop_modes = loop_modes[n_left_loop:]

    KR_Lw = KR_L * w[:, None]
    KR_Lw_ext = KR_Lw
    for lm in left_loop_modes:
        KR_Lw_ext = (KR_Lw_ext[:, :, None] * mats[lm][:, None, :]).reshape(nnz, -1)

    KR_R_ext = KR_R
    for rm in right_loop_modes:
        KR_R_ext = (KR_R_ext[:, :, None] * mats[rm][:, None, :]).reshape(nnz, -1)

    # Matmul sums over nnz; output shape: (L_ext, R_ext)
    result = KR_Lw_ext.T @ KR_R_ext

    # Reshape then permute axes to (left_modes | left_loop | right_loop | right_modes),
    # which is the order expected by target_shape = left_modes + loop_modes + right_modes.
    n_left  = len(left_modes)
    n_right = len(right_modes)
    left_ext_dims  = [mats[i].shape[1] for i in left_modes] + [mats[i].shape[1] for i in left_loop_modes]
    right_ext_dims = [mats[i].shape[1] for i in right_modes] + [mats[i].shape[1] for i in right_loop_modes]
    result = result.reshape(left_ext_dims + right_ext_dims)
    # Current axis order: left_modes | left_loop | right_modes | right_loop
    # Target axis order:  left_modes | left_loop | right_loop  | right_modes
    if n_right_loop:
        perm = (list(range(n_left + n_left_loop)) +
                list(range(n_left + n_left_loop + n_right,
                           n_left + n_left_loop + n_right + n_right_loop)) +
                list(range(n_left + n_left_loop, n_left + n_left_loop + n_right)))
        result = result.transpose(perm)

    target_shape = [mats[i].shape[1] for i in left_modes + loop_modes + right_modes]
    Num += result.reshape(target_shape)

    # -- old Python iteration over loop-mode rank combinations (kept for reference) --
    # loop_ranks = [mats[i].shape[1] for i in loop_modes]
    # for loop_idx in itertools.product(*[range(r) for r in loop_ranks]):
    #     v = w.copy()
    #     for loop_i, r in zip(loop_modes, loop_idx):
    #         v *= mats[loop_i][:, r]
    #     if KR_L is not None and KR_R is not None:
    #         slice_sum = (KR_L * v[:, None]).T @ KR_R
    #         full_slice = [slice(None) if i in left_modes + right_modes else loop_idx[loop_modes.index(i)]
    #                       for i in range(N)]
    #         Num[tuple(full_slice)] += slice_sum.reshape([mats[i].shape[1] for i in left_modes + right_modes])
    #     elif KR_L is not None:
    #         slice_sum = cp.sum(KR_L * v[:, None], axis=0)
    #         full_slice = [slice(None) if i in left_modes else loop_idx[loop_modes.index(i)] for i in range(N)]
    #         Num[tuple(full_slice)] += slice_sum.reshape([mats[i].shape[1] for i in left_modes])


def _blocked_coo_to_flat_indices(vec_tensor, orig_shape):
    orig_shape = tuple(orig_shape)
    size = math.prod(orig_shape)
    int32_max = np.iinfo(np.int32).max
    block_size = min(size, int32_max)

    coo = vec_tensor.tocoo()
    flat = coo.row.astype(cp.int64) + coo.col.astype(cp.int64) * cp.int64(block_size)
    vals = coo.data
    return flat, vals


def coo_to_coords(vec_tensor, orig_shape):
    """Per-mode NNZ coordinates and values, whatever the storage form.

    The single seam every NNZ-streaming (``*_largedim``) kernel goes through,
    replacing the ``_blocked_coo_to_flat_indices`` + ``_unravel_flat_indices_C``
    pair they used to open with.

    Free for a ``CoordCOO`` (coordinates are already stored, and no linear index
    is formed — which is what lets order-5 tensors work at all); an identical
    decode to before for the legacy block-encoded ``coo_matrix``.

    Returns
    -------
    idxs : list of N arrays, each (nnz,) — idxs[n] indexes mode n
    vals : (nnz,) values
    """
    if isinstance(vec_tensor, CoordCOO):
        return vec_tensor.coord_list(), vec_tensor.data
    flat, vals = _blocked_coo_to_flat_indices(vec_tensor, orig_shape)
    return _unravel_flat_indices_C(flat, orig_shape), vals


def coords_nnz(vec_tensor) -> int:
    """NNZ count for either storage form, without decoding anything."""
    if isinstance(vec_tensor, CoordCOO):
        return vec_tensor.nnz
    return int(vec_tensor.tocoo().row.size)


# ---------------------------------------------------------------------------
# Per-mode NNZ grouping cache (2026-06-12 review, Task 3 — findings E-1/E-2/E-3)
# ---------------------------------------------------------------------------
#
# The largedim factor update groups NNZ by unfolding column (`cp.unique`), then
# scans the full inverse-map (`cp.where`) once per column batch. Both are done
# every iteration on a *static* tensor, so they are the single largest avoidable
# cost in the hot loop. SOTA sparse-tensor formats (SPLATT/CSF, HiCOO, ALTO)
# hoist exactly this per-mode grouping out of the loop. ModeGrouping precomputes
# it once per (tensor, mode); the kernels then operate on contiguous slices.

@dataclass
class ModeGrouping:
    """Precomputed per-mode NNZ grouping for a largedim factor update.

    Built once per (tensor, mode) and reused every iteration, so the
    ``cp.unique`` sort (E-1), the flat-index decode (E-3), and the per-batch
    full-``inv`` ``cp.where`` scan (E-2) are all hoisted out of the training
    loop.

    The NNZ are reordered so that all entries sharing an unfolding column are
    contiguous: ``rows_sorted[segment_offsets[j]:segment_offsets[j+1]]`` are the
    mode-``mode`` coordinates of the entries in unique column ``ucols[j]`` (and
    ``vals_sorted`` the corresponding values). A batch of unique columns
    ``[start:end)`` is therefore the single contiguous slice
    ``[segment_offsets[start]:segment_offsets[end])`` — no ``cp.where`` scan of
    the whole array, just two offset lookups. ``col_index`` holds, per sorted
    entry, the index into ``ucols`` of its column, so the per-entry Z-row index
    within a batch is ``col_index[slice] - start`` (a plain slice + subtract; no
    ``cp.repeat``, which CuPy will not take a device array of counts for).

    Depends only on the tensor's NNZ pattern (not on the divergence or the
    masked/full objective), so one grouping serves all of those. NOT valid under
    stochastic subsampling, where the sampled entries change every iteration.
    """
    ucols: "cp.ndarray"            # (n_ucols,) sorted unique unfolding-column ids
    segment_offsets: "cp.ndarray"  # (n_ucols+1,) run boundaries in the sorted order
    rows_sorted: "cp.ndarray"      # (nnz,) mode-`mode` coordinate, column-grouped
    vals_sorted: "cp.ndarray"      # (nnz,) tensor values, same order
    col_index: "cp.ndarray"        # (nnz,) index into `ucols` per sorted entry

    @property
    def n_ucols(self) -> int:
        return int(self.ucols.size)


def _build_mode_grouping(vec_tensor, shape, mode) -> ModeGrouping:
    """Decode + sort + group a static COO once for ``mode`` (see ModeGrouping).

    Must be called on the device that owns ``vec_tensor`` (the sort and gather
    run there). The result is reusable for every subsequent iteration.
    """
    idxs, vals = coo_to_coords(vec_tensor, shape)
    rows = idxs[mode]

    other_modes = [m for m in range(len(shape)) if m != mode]
    other_shape = tuple(shape[m] for m in other_modes)
    other_coords = [idxs[m] for m in other_modes]
    cols = safe_ravel(tuple(other_coords), other_shape, cp)

    # Sort NNZ by unfolding column so equal columns form contiguous runs. This is
    # the one sort that the cache replaces the per-iteration cp.unique with.
    order = cp.argsort(cols, kind="stable")
    ucols, inv, counts = cp.unique(cols, return_inverse=True, return_counts=True)
    inv = inv.reshape(-1)  # guard: NumPy 2.0 briefly returned a 2-D inverse
    # segment_offsets[j]:segment_offsets[j+1] is unique column j's run; the cumsum
    # of run lengths, prefixed with 0, gives the (n_ucols+1,) boundary array.
    segment_offsets = cp.concatenate(
        (cp.zeros(1, dtype=counts.dtype), cp.cumsum(counts))
    )
    return ModeGrouping(
        ucols=ucols,
        segment_offsets=segment_offsets,
        rows_sorted=rows[order],
        vals_sorted=vals[order],
        # inv is in original order; reordering by `order` gives the unique-column
        # index per *sorted* entry, aligned with rows_sorted / vals_sorted.
        col_index=inv[order],
    )


class NNZGroupingCache:
    """Lazily builds and stores one :class:`ModeGrouping` per mode for a static COO.

    Pass an instance' per-mode grouping (``cache.get(mode)``) to
    ``kl_factor_update_largedim`` / ``fr_factor_update_largedim`` via the
    ``grouping=`` argument to skip the per-iteration flat-index decode,
    ``cp.unique`` sort, and per-batch ``cp.where`` scan.

    Construct once *before* the training loop and reuse. Valid only while the
    underlying NNZ is static — do NOT use under stochastic subsampling (the
    sampled values change every iteration); pass ``None`` there.
    """

    def __init__(self, vec_tensor, shape):
        self._vec_tensor = vec_tensor
        self._shape = tuple(shape)
        self._by_mode: dict = {}

    def get(self, mode) -> ModeGrouping:
        g = self._by_mode.get(mode)
        if g is None:
            g = _build_mode_grouping(self._vec_tensor, self._shape, mode)
            self._by_mode[mode] = g
        return g


def _tucker_sum_all_entries(core, factors, epsilon=1e-12):
    """
    Exact sum(R) where R = Tucker(core, factors), without forming R.

    sum_R = sum_{r0..rN-1} core[r0..rN-1] * Π_n s_n[rn]
    where s_n[rn] = sum_i A^{(n)}[i, rn]
    """
    N = core.ndim
    letters = einsum_letters(N)
    core_subs = "".join(letters)

    sums = [cp.sum(factors[n], axis=0) for n in range(N)]
    sums = [cp.clip(s, a_min=epsilon, a_max=None) for s in sums]

    # eq: "abc,a,b,c->" (for N=3), etc.
    eq = core_subs + "," + ",".join(letters) + "->"
    sum_R = cp.einsum(eq, core, *sums, optimize=cp_einsum_optimize(1 + N))
    return cp.clip(sum_R, a_min=epsilon, a_max=None)

# FR- specific helpers
def _core_unfold(core, mode):
    """
    Mode-n unfolding of the Tucker core: (R_mode, prod(other Rk))
    using C-order flattening with remaining modes in increasing order.
    """
    G = cp.moveaxis(core, mode, 0)
    return G.reshape(G.shape[0], -1)

def _tucker_gram_ZtZ(core, factors, mode, epsilon=1e-12):
    """
    Compute Gram = Z^T Z exactly, without forming Z.

    In your dense version:
        Z = transpose(unfold(tucker_to_tensor(skip_factor=mode), mode))  # (J, R_mode)
        Gram = Z^T Z                                                    # (R_mode, R_mode)

    Algebra:
        Z = K @ G_(mode)^T
        with K = kron_{k!=mode}(A_k)  and  G_(mode) is core unfolded along mode.

        K^T K = kron_{k!=mode}(A_k^T A_k)

        Gram = G_(mode) @ (K^T K) @ G_(mode)^T

    We compute this by contracting the core with the per-mode Gram matrices (A_k^T A_k),
    then doing one small matrix multiply in rank-space.
    """
    N = core.ndim
    letters = einsum_letters(2 * N)  # need “primed” letters too
    base = letters[:N]
    prim = letters[N:2 * N]

    core_subs = "".join(base)

    other_modes = [k for k in range(N) if k != mode]
    grams = [factors[k].T @ factors[k] for k in other_modes]

    # Build output subscripts: keep mode letter, replace others by their primed version
    out = list(base)
    for k in other_modes:
        out[k] = prim[k]
    out_subs = "".join(out)

    # core[a b c ...], G_b[b B], G_c[c C], ... -> out[a B C ...] (mode stays unprimed)
    gram_terms = [f"{base[k]}{prim[k]}" for k in other_modes]
    eq = core_subs + "," + ",".join(gram_terms) + "->" + out_subs

    Gp = cp.einsum(eq, core, *grams, optimize=cp_einsum_optimize(1 + len(grams)))  # same shape as core, but other modes live in primed space

    G_unf = _core_unfold(core, mode)   # (R_mode, P)
    Gp_unf = _core_unfold(Gp, mode)    # (R_mode, P)

    Gram = Gp_unf @ G_unf.T            # (R_mode, R_mode) -> sparse
    Gram = cp.clip(Gram, a_min=epsilon, a_max=None)
    return Gram

def _core_multilinear_grams(core, grams, epsilon=1e-12):
    """
    Compute:
        D = core ×_0 grams[0] ×_1 grams[1] × ... ×_{N-1} grams[N-1]
    where grams[n] = A_n^T A_n has shape (R_n, R_n).

    Returns D with the same shape as core, without mode_dot / tl overhead.
    """
    N = core.ndim
    letters = einsum_letters(2 * N)   # first N: input indices, next N: output indices
    base = letters[:N]
    prim = letters[N:2 * N]
    core_sub  = "".join(base)                              # e.g. 'abc'
    gram_subs = [base[n] + prim[n] for n in range(N)]     # e.g. ['aA', 'bB', 'cC']
    out_sub   = "".join(prim)                              # e.g. 'ABC'
    eq = core_sub + "," + ",".join(gram_subs) + "->" + out_sub
    tmp = cp.einsum(eq, core, *grams, optimize=cp_einsum_optimize(1 + N))
    return cp.clip(tmp, a_min=epsilon, a_max=None)

    # -- old sequential tensordot+moveaxis loop (kept for reference) --
    # tmp = core
    # for n in range(N):
    #     G = grams[n]  # (R_n, R_n)
    #     # tensordot over core axis n: (R_n,R_n) x (...,R_n,...) -> (R_n, ..., ...)
    #     tmp = cp.tensordot(G, tmp, axes=(1, n))
    #     # tensordot brings the new R_n axis to the front; move it back to position n
    #     tmp = cp.moveaxis(tmp, 0, n)
    # return cp.clip(tmp, a_min=epsilon, a_max=None)


# batch estimation helpers
def _gpu_free_bytes():
    """
    Conservative 'free bytes now' estimate.

    CHANGED (2026-06-15): no longer calls ``mempool.free_all_blocks()`` first.
    That flush returned every cached block to the CUDA driver via ``cudaFree``
    — a synchronizing, single-threaded driver call — and the next kernel then
    had to ``cudaMalloc`` it all back. Run from the per-iteration batch-size
    estimators (~6-7×/iteration), it drained the GPU to idle while one host
    thread sat in the driver, producing the stop-start iteration stalls (and
    defeating the whole point of the memory pool). The driver's own free figure
    plus the pool's cached-but-reusable bytes give the same headroom estimate
    without any cudaFree/cudaMalloc churn.
    """
    mempool = cp.get_default_memory_pool()
    free_b, total_b = cp.cuda.runtime.memGetInfo()
    # Pool blocks that are cached but currently unused are reusable for the next
    # allocation without hitting the driver, so count them as available too.
    pool_reusable = int(mempool.free_bytes())
    return int(free_b) + pool_reusable

def _estimate_batch_num_for_outer(core, factors, safety=0.70, temp_mult=2.0, reserve_b=0):
    """
    New estimator for the optimized matrix-multiplication accumulator.
    It no longer assumes the materialization of the full core outer product!

    reserve_b :
        Bytes held back from the free-VRAM budget for allocations that are not
        live at estimate time. Used when the estimate is hoisted out of the
        iteration loop (precompute_largedim_batches) to reserve the kernel's
        transient NNZ decode arrays, preserving the in-kernel snapshot headroom.
    """
    N = len(factors)
    itemsize = int(np.dtype(core.dtype).itemsize)
    R = [int(factors[n].shape[1]) for n in range(N)]

    # Max KR product size. We split into Left/Right roughly evenly.
    half_N = (N + 1) // 2
    largest_KR_rank = math.prod(sorted(R, reverse=True)[:half_N])

    # Memory per batch element is dominated by the Khatri-Rao matrices
    bytes_per_b = 2 * largest_KR_rank * itemsize
    bytes_per_b = int(math.ceil(bytes_per_b * temp_mult))

    free_b = max(1, int(_gpu_free_bytes()) - int(reserve_b))
    budget_b = int(free_b * safety)

    b = max(1, budget_b // max(1, bytes_per_b))

    # OLD SAFE CODE: hard_cap = max(1, int(1_000_000_000 // max(1, bytes_per_b)))
    # Hard cap: safety rail proportional to free VRAM so it scales with GPU size.
    # 0.95 > safety (0.80), so this only binds if the budget estimate overshoots.
    hard_cap = max(1, int(free_b * 0.95 // max(1, bytes_per_b)))
    return min(int(b), hard_cap)


def _estimate_batch_rhat_for_tensordot(core, factors, safety=0.7, temp_mult=4.0, reserve_b=0):  # Increased temp_mult; safety raised from 0.60
    # reserve_b: bytes held back for allocations not live at estimate time
    # (see _estimate_batch_num_for_outer); set when hoisted out of the loop.
    N = core.ndim
    R = [int(factors[n].shape[1]) for n in range(N)]
    dtype = core.dtype
    itemsize = int(np.dtype(dtype).itemsize)

    # The bottleneck: (batch, R1, R2...)
    prod_rest = math.prod(R[1:])
    tmp_bytes_per_b = prod_rest * itemsize

    # Total bytes per batch element
    bytes_per_b = int(np.ceil(tmp_bytes_per_b * temp_mult))

    free_b = max(1, int(_gpu_free_bytes()) - int(reserve_b))
    # Ensure we leave a large buffer for the rest of the graph
    budget_b = int(free_b * safety)

    b = budget_b // max(1, bytes_per_b)
    return max(1, int(b))


def _estimate_batch_cols_for_Z(core, factors, mode, safety=0.8, temp_mult=4.0,
                               masked=False, workspace_reserve=512 * 1024**2):
    """
    Estimate safe batch size for compute_Zcols_batch.
    Uses pure Python math to avoid numpy 32-bit overflows and sets a hard cap.
    temp_mult has to be sufficiently high: 2 massively undershot the temp need

    masked : bool
        The masked/completion objective builds a *second* CSR (S_den) and runs a
        second SpMM per batch (denominator += S_den @ Z_rows, see
        _partial_numerator_for_shard), roughly doubling the per-batch sparse
        working set. The default (numerator-only) budget under-counts this, which
        let masked runs overshoot VRAM — the cuBLAS workspace cudaMalloc (done
        outside CuPy's pool) then failed as CUBLAS_STATUS_NOT_INITIALIZED. When
        masked=True the per-batch cost is inflated accordingly.
    workspace_reserve : int
        Bytes held back from the free-memory budget for out-of-pool allocations
        (cuBLAS/cuSPARSE handle workspaces) so they always have room.
    """
    N = core.ndim
    R = [int(factors[n].shape[1]) for n in range(N)]
    itemsize = int(np.dtype(core.dtype).itemsize)

    other_modes = [k for k in range(N) if k != mode]
    if not other_modes:
        return 20000

    k0 = other_modes[0]
    # Pure Python math.prod guarantees no 32-bit wrap-around
    remaining_R_prod = math.prod([R[k] for k in range(N) if k != k0])

    # Element-wise operations allocate full temporary copies, so we need a multiplier of ~3.0
    tmp_bytes_per_b = remaining_R_prod * itemsize
    bytes_per_b = int(math.ceil(tmp_bytes_per_b * temp_mult))
    if masked:
        # Second CSR + SpMM + denominator accumulation ≈ doubles the working set.
        bytes_per_b *= 2

    # Reserve headroom for cuBLAS/cuSPARSE workspaces, which cudaMalloc outside
    # CuPy's memory pool and would otherwise have nothing left to allocate.
    free_b = max(1, int(_gpu_free_bytes()) - int(workspace_reserve))
    budget_b = int(free_b * safety)

    b = max(1, budget_b // max(1, bytes_per_b))

    # OLD SAFE CODE: hard_cap = max(1, int(2_000_000_000 // max(1, tmp_bytes_per_b)))
    # Hard cap: anchor to free VRAM so it scales with GPU size.
    # b = free_b * safety / (tmp * temp_mult) = free_b * 0.40 / tmp;
    # hard_cap = free_b * 0.50 / tmp = b * 1.25, so it is always above b and normally doesn't bind.
    hard_cap = max(1, int(free_b * 0.50 // max(1, tmp_bytes_per_b)))

    return min(int(b), hard_cap)


def precompute_largedim_batches(core, factors, modes, masked=False, nnz_live=0,
                                coord_backed=False):
    """
    Precompute the single-GPU largedim KL per-iteration batch sizes ONCE.

    CHANGED (2026-06-15): the batch-size estimates depend only on core/factor
    shapes and dtype (fixed for a run) plus a free-VRAM snapshot, so calling
    them inside every factor/core/error update just repeated identical
    arithmetic — and, before the ``_gpu_free_bytes`` fix, flushed the memory
    pool 6-7×/iteration. The main loop now calls this once, after all persistent
    device allocations are live, and threads the results into the kernels'
    ``batch_*`` kwargs so the kernels skip their internal estimate entirely.

    nnz_live :
        Per-iteration NNZ count of the tensor the kernels will decode (the
        subsample window size under stochastic subsampling, else the full nnz).
        The kernels allocate ~(N+3)·8·nnz_live bytes of transient decode arrays
        (flat / idxs / cols / ucols / inv) that are not live at precompute time;
        reserving those bytes here reproduces the headroom the in-kernel
        estimate got from snapshotting after that bookkeeping (review Task 1),
        so hoisting the call out does not regress peak-memory safety.
    coord_backed :
        True when the kernels consume a ``CoordCOO``, whose coordinates are
        persistent (already in the snapshot) and passed as views — ``flat`` and
        the N ``idxs`` are never allocated, leaving only cols/ucols/inv
        transient. Reserving the full (N+3) tail would shrink every batch.

    Returns a dict with ``batch_cols`` (mode -> int), ``batch_rhat`` and
    ``batch_num``.
    """
    N = core.ndim
    transient_b = (3 if coord_backed else (N + 3)) * 8 * int(nnz_live)
    batch_cols = {
        m: _estimate_batch_cols_for_Z(
            core, factors, m, masked=masked,
            workspace_reserve=512 * 1024**2 + transient_b,
        )
        for m in modes
    }
    return {
        "batch_cols": batch_cols,
        "batch_rhat": _estimate_batch_rhat_for_tensordot(core, factors, reserve_b=transient_b),
        "batch_num": _estimate_batch_num_for_outer(core, factors, reserve_b=transient_b),
    }

def kl_factor_update_largedim(
    vec_tensor,
    core,
    factors,
    mode,
    shape,
    thread_budget=None,
    epsilon=1e-12,
    batch_cols=None,
    verbose=False,
    masked=False,
    grouping=None,
):
    """
    KL multiplicative update for Tucker factor A^(mode) WITHOUT building dense Z,
    but mathematically equivalent to your dense-Z implementation:

        A <- A * ( (W @ Z.T) / sum_j Z[:, j] )

    where W_ij = X_ij / (A Z)_ij at the nonzeros of X_(mode).

    Works for N-way tensors (N = len(shape) = core.ndim).

    masked : bool
        If False (default), the denominator sums Z over ALL unfolding columns
        (the zero-filled / full-tensor objective). If True, only OBSERVED columns
        contribute: den[i, r] = sum_{j in Omega_i} Z[r, j]. This is the weighted/
        completion objective (treat unobserved entries as missing, not zero).
    grouping : ModeGrouping, optional
        CHANGED (2026-06-12 review, Task 3 — E-1/E-2/E-3): precomputed per-mode
        NNZ grouping for this (static) tensor. When supplied, the per-iteration
        flat-index decode, ``cp.unique`` sort, and per-batch full-``inv``
        ``cp.where`` scan are all skipped — the NNZ are already column-grouped, so
        each column batch is a contiguous slice. Must be ``None`` under stochastic
        subsampling (the sampled values change every iteration).
    """

    # Sparse unfolding X_(mode)
    # X = unfold_from_vectorized_sparse(vec_tensor, shape, mode).tocoo()
    # rows = X.row
    # cols = X.col
    # vals = X.data

    if verbose:
        print(f"  Updating factor {mode}...")

    other_modes = [m for m in range(len(shape)) if m != mode]

    if grouping is not None:
        # Cached path (Task 3): NNZ already decoded, sorted and grouped by
        # unfolding column. No _blocked_coo_to_flat_indices / cp.unique here.
        ucols = grouping.ucols
        segment_offsets = grouping.segment_offsets
        rows_sorted = grouping.rows_sorted
        vals_sorted = grouping.vals_sorted
        col_index = grouping.col_index
        inv = None  # contiguous segments replace the inv-scan
    else:
        # Uncached path: decode + group this iteration (subsampled tensors land
        # here, since their NNZ pattern changes every call). Coordinate-backed
        # tensors have nothing to decode — coo_to_coords hands back its stored
        # coordinates, so only the grouping below costs anything.
        idxs, vals = coo_to_coords(vec_tensor, shape)

        rows = idxs[mode]

        other_coords = [idxs[m] for m in other_modes]
        # build a safe unfolded-column id in int64 only for grouping
        other_shape = tuple(shape[m] for m in other_modes)
        cols = safe_ravel(tuple(other_coords), other_shape, cp)

        # Reuse computations across repeated columns
        ucols, inv = cp.unique(cols, return_inverse=True)

    A = factors[mode]  # (I_mode, R_mode)

    if masked:
        # Masked objective: denominator is accumulated over observed columns only
        # (see batch loop below), so no full-column den_row is needed.
        denominator = None
        denominator_acc = cp.zeros_like(A)
    else:
        # Exact denominator over ALL columns (no approximation)
        den_row = _tucker_den_row_full(core, factors, mode, epsilon=epsilon)
        denominator = den_row[None, :]  # (1, R_mode)

    # Accumulate numerator = W @ Z.T without building full Z
    numerator = cp.zeros_like(A)

    # CHANGED (2026-06-12 review, Task 1): estimate AFTER all NNZ bookkeeping
    # (flat, idxs, cols, ucols, inv) is live on the GPU, so the free-memory
    # snapshot reflects the actual headroom for the per-batch temporaries.
    # Previously estimated at the top of the function, which overestimated
    # free VRAM by ~(N+3)*8*nnz bytes. Ported from the sharded path
    # (sharded_sparse.py::_partial_numerator_for_shard). With a grouping the
    # cached arrays are persistent, so the snapshot is already representative.
    if batch_cols is None:
        batch_cols = _estimate_batch_cols_for_Z(core, factors, mode, masked=masked)

    n_ucols = int(ucols.size)
    col_batches = range(0, n_ucols, int(batch_cols))
    if verbose:
        col_batches = tqdm(col_batches, desc=f"  Factor {mode} col-batches", unit="batch", leave=False)
    for start in col_batches:
        end = min(start + int(batch_cols), n_ucols)
        u = ucols[start:end]

        _, idxs_by_mode = _unravel_cols_for_mode(u, shape, mode)  # dict: other_mode -> (m,)

        # Z_u: (m, R_mode)
        Z_u = compute_Zcols_batch(
            core=core,
            factors=factors,
            mode=mode,
            other_modes=other_modes,
            idxs_by_mode=idxs_by_mode,
            epsilon=epsilon,
        )

        # nnz entries belonging to these unique columns
        if grouping is not None:
            # E-2: contiguous slice of the column-grouped arrays — no full scan.
            seg_lo = int(segment_offsets[start])
            seg_hi = int(segment_offsets[end])
            if seg_hi == seg_lo:
                continue
            r_i = rows_sorted[seg_lo:seg_hi]   # (nnz_b,)
            v_i = vals_sorted[seg_lo:seg_hi]   # (nnz_b,)
            # local Z row per entry: ucols index minus the batch's first column.
            u_i = col_index[seg_lo:seg_hi] - start
        else:
            nz_idx = cp.where((inv >= start) & (inv < end))[0]
            if nz_idx.size == 0:
                continue
            r_i = rows[nz_idx]             # (nnz_b,)
            v_i = vals[nz_idx]             # (nnz_b,)
            u_i = inv[nz_idx] - start      # local [0..m)

        nnz_b = int(r_i.size)

        if use_legacy_factor_batch():
            # Legacy body (pre 2026-07-29): two (nnz_b, R) gathers + an
            # (I, nnz_b) arange-column CSR faking the scatter-add. Kept behind
            # TENSORMET_LEGACY_FACTOR_BATCH=1 for A/B validation.
            A_rows = A[r_i]                # (nnz_b, R_mode)
            Z_rows = Z_u[u_i]              # (nnz_b, R_mode)

            # (A Z)_nz
            R_nz = cp.sum(A_rows * Z_rows, axis=1)
            R_nz = cp.clip(R_nz, a_min=epsilon, a_max=None)

            W_data = v_i / R_nz            # (nnz_b,)

            # numerator[row] += W * Z  — cuSPARSE SpMM (no serialised atomics)
            col_idx_b = cp.arange(nnz_b, dtype=cp.int32)
            row_idx_b = r_i.astype(cp.int32)
            S_b = cpx_sparse.csr_matrix(
                (W_data, (row_idx_b, col_idx_b)),
                shape=(numerator.shape[0], nnz_b),
            )
            numerator += S_b @ Z_rows

            if masked:
                # denominator[i, r] += sum_{k: row=i} Z_rows[k, r]  (weight 1 per observed entry)
                S_one = cpx_sparse.csr_matrix(
                    (cp.ones(nnz_b, dtype=Z_rows.dtype), (row_idx_b, col_idx_b)),
                    shape=(numerator.shape[0], nnz_b),
                )
                denominator_acc += S_one @ Z_rows
            continue

        # CHANGED (2026-07-29): scatter-free batch body. R_nz comes from a
        # fused sampled row-dot (SDDMM) instead of two (nnz_b, R) gathers, and
        # the numerator SpMM runs against the UNGATHERED Z_u via the (m, I)
        # transposed batch matrix P — Z_rows is never materialized. With a
        # grouping, P is built sort-free straight from the segment offsets.
        m_b = end - start                  # rows of P == Z_u.shape[0]
        if grouping is not None:
            indptr_b = (segment_offsets[start:end + 1] - seg_lo).astype(cp.int32)
        else:
            # Uncached batches must be column-grouped up front so P.data order
            # equals entry order (see group_batch_by_column).
            indptr_b, u_i, r_i, v_i = group_batch_by_column(u_i, m_b, r_i, v_i)

        R_nz = sampled_row_dots(A, Z_u, r_i, u_i)
        R_nz = cp.clip(R_nz, a_min=epsilon, a_max=None)
        W_data = v_i / R_nz                # (nnz_b,)

        P = build_batch_csr_T(W_data, r_i, m_b, numerator.shape[0], indptr_b)
        numerator += spmm_T(P, Z_u)

        if masked:
            # denominator[i, r] += sum_{k: row=i} Z_u[u_k, r]  (weight 1 per
            # observed entry) — same pattern as P, data swapped to ones.
            P_one = same_pattern_csr(P, cp.ones(nnz_b, dtype=Z_u.dtype))
            denominator_acc += spmm_T(P_one, Z_u)

    if masked:
        denominator = denominator_acc

    # Multiplicative KL update (matching your dense version structure)
    A_new = A * (numerator / (denominator + 1e-12))
    A_new = cp.clip(A_new, a_min=epsilon, a_max=None)
    return A_new




def kl_core_update_largedim(
    vec_tensor,
    shape,
    core,
    factors,
    modes=None,              # assumes all modes
    thread_budget=None,      # kept for API compatibility
    epsilon=1e-12,
    batch_rhat=None, # tested, quite efficient up to 8K dims
    batch_num=None, # tested, quite efficient up to 8K dims
    verbose=False,
    masked=False,
):
    if verbose:
        print("  Updating core...")

    shape = tuple(int(s) for s in shape)
    N = len(shape)
    if modes is None:
        modes = list(range(N))
    if list(modes) != list(range(N)):
        raise NotImplementedError("This version assumes modes == all modes (0..N-1).")
    idxs, xvals = coo_to_coords(vec_tensor, shape)  # list length N, each (nnz,)
    nnz = int(xvals.size)
    if nnz == 0:
        return core

    # Full objective: denominator is the outer product of factor column sums
    # (= sum over ALL tensor entries). Masked objective: denominator is summed
    # over OBSERVED entries only, accumulated alongside the numerator below.
    if masked:
        Den = cp.zeros_like(core)
    else:
        sums = [cp.clip(cp.sum(factors[n], axis=0), a_min=epsilon, a_max=None) for n in range(N)]
    Num = cp.zeros_like(core)

    # --- Pass 1: compute w = x / r_hat in big batches, stash w (or stream into pass 2)
    # Stashing w costs nnz floats; if that's too big, you can stream (see note below).
    w_all = cp.empty_like(xvals)

    # CHANGED (2026-06-12 review, Task 1): estimate AFTER the NNZ bookkeeping
    # (flat, idxs, Num/Den, w_all) is live so the free-memory snapshot is
    # accurate. Previously estimated at the top of the function. Ported from
    # the sharded path (sharded_sparse.py::_partial_core_num_for_shard).
    if batch_rhat is None:
        batch_rhat = _estimate_batch_rhat_for_tensordot(core, factors)
    if batch_num is None:
        batch_num = _estimate_batch_num_for_outer(core, factors)

    rhat_batches = range(0, nnz, int(batch_rhat))
    if verbose:
        rhat_batches = tqdm(rhat_batches, desc="  Core r_hat pass", unit="batch", leave=False)
    for start in rhat_batches:
        end = min(start + int(batch_rhat), nnz)

        mats = [factors[n][idxs[n][start:end]] for n in range(N)]  # each (b, Rn)
        r_hat = _rhat_from_factor_rows_sequential(core, mats, epsilon=epsilon)  # (b,)
        w_all[start:end] = xvals[start:end] / r_hat

    # --- Pass 2: accumulate numerator in tiny batches (controls peak memory)
    # this takes most time!
    num_batches = range(0, nnz, int(batch_num))
    if verbose:
        num_batches = tqdm(num_batches, desc="  Core numerator pass", unit="batch", leave=False)
    for start in num_batches:
        end = min(start + int(batch_num), nnz)
        mats = [factors[n][idxs[n][start:end]] for n in range(N)]
        w = w_all[start:end]
        _accumulate_core_num_outer(Num, w, mats)
        if masked:
            # Den += sum_{k in batch} outer(factor rows)  (weight 1 per observed entry)
            _accumulate_core_num_outer(Den, cp.ones(end - start, dtype=core.dtype), mats)

    if masked:
        # --- MU update (masked): core *= Num / Den, both over observed entries
        core_new = core * (Num / (Den + epsilon))
        core_new = cp.clip(core_new, a_min=epsilon, a_max=None)
        return core_new

    # --- MU update: core *= Num / (outer product of sums)
    core_new = core * (Num + epsilon)  # keep >0

    # Divide by sums via broadcasting, no F allocation
    for n in range(N):
        shp = [1] * N
        shp[n] = sums[n].shape[0]
        core_new = core_new / sums[n].reshape(tuple(shp))
    core_new = cp.clip(core_new, a_min=epsilon, a_max=None)
    return core_new



# --- no-dense KL error ---
def kl_compute_errors_largedim(
    vec_tensor: cpx_sparse.spmatrix,
    shape,
    core,
    factors,
    thread_budget=None,          # kept for API compatibility; unused
    epsilon=1e-12,
    batch_rhat=None, # tested up to 8K
    verbose=False,
    masked=False,
):
    """
    Relative generalized KL divergence C_KL(X || R) for sparse X,
    WITHOUT forming dense R, staying close to the core-update approach.

    Computes:
      KL = sum_{nz} [x log(x/r) - x + r] + (sum_R - sum_{nz} r)
      rel_KL = KL / sum_{nz} x

    masked : bool
        If True, the divergence is evaluated over observed (nonzero) entries
        only; the zero-entry contribution (sum_R - sum_{nz} r) is dropped, so
        the metric is consistent with the masked/completion objective.
    """
    if verbose:
        print("  Computing KL errors...")

    shape = tuple(int(s) for s in shape)
    N = len(shape)

    idxs, x_nz = coo_to_coords(vec_tensor, shape)  # list of N arrays, each (nnz,)
    nnz = int(x_nz.size)
    if nnz == 0:
        # If X is all-zeros, KL reduces to sum_R. Relative term is ill-defined; mirror your style:
        sum_R = _tucker_sum_all_entries(core, factors, epsilon=epsilon)
        return sum_R / cp.maximum(cp.asarray(0.0, dtype=sum_R.dtype), epsilon)

    x_nz = cp.asarray(x_nz)
    x_nz = cp.clip(x_nz, a_min=epsilon, a_max=None)

    # --- compute r_nz in batches (like your core update r_hat pass) ---
    r_nz = cp.empty_like(x_nz)

    # CHANGED (2026-06-12 review, Task 1): estimate AFTER the NNZ bookkeeping
    # (flat, x_nz, idxs, r_nz) is live so the free-memory snapshot is accurate.
    # Previously estimated at the top of the function.
    if batch_rhat is None:
        batch_rhat = _estimate_batch_rhat_for_tensordot(core, factors)

    rhat_batches = range(0, nnz, int(batch_rhat))
    if verbose:
        rhat_batches = tqdm(rhat_batches, desc="  KL error r_hat pass", unit="batch", leave=False)
    for start in rhat_batches:
        end = min(start + int(batch_rhat), nnz)
        mats = [factors[n][idxs[n][start:end]] for n in range(N)]  # each (b, Rn)
        r_nz[start:end] = _rhat_from_factor_rows_sequential(core, mats, epsilon=epsilon)

    r_nz = cp.clip(r_nz, a_min=epsilon, a_max=None)

    # --- KL contribution from nonzeros ---
    term_pos = x_nz * cp.log(x_nz / r_nz) - x_nz + r_nz
    kl_pos = cp.sum(term_pos)

    if masked:
        # Masked/completion objective: only observed entries contribute.
        kl_total = kl_pos
    else:
        # --- zero contribution: sum_R - sum_{nz} r_nz ---
        sum_R = _tucker_sum_all_entries(core, factors, epsilon=epsilon)
        sum_R_nz = cp.sum(r_nz)
        kl_zero = sum_R - sum_R_nz
        kl_total = kl_pos + kl_zero

    sum_X = cp.sum(x_nz)
    rel_kl = kl_total / cp.maximum(sum_X, epsilon)
    return rel_kl

# Frobenius large dim / streaming
def fr_factor_update_largedim(
    vec_tensor,
    core,
    factors,
    mode,
    shape,
    epsilon=1e-12,
    thread_budget=None, # kept for API compatibility; unused
    batch_cols=None,
    verbose=False,
    masked=False,
    grouping=None,
):
    """
    Frobenius (Euclidean) multiplicative update for Tucker factor A^(mode)
    WITHOUT building dense Z, but equivalent to your dense function 3:

        numerator   = X @ Z
        denominator = A @ (Z^T Z)
        A <- A * numerator / denominator

    where X is the sparse unfolding and Z = transpose(unfold(tucker_to_tensor(skip_factor=mode), mode)).

    masked : bool
        If False (default), the denominator A @ (Z^T Z) sums over ALL unfolding
        columns (zero-filled objective). If True, the denominator is restricted
        to observed entries: den[i, r] = sum_{j in Omega_i} Xhat[i, j] Z[r, j],
        accumulated per batch. This is the weighted/completion objective.
    grouping : ModeGrouping, optional
        CHANGED (2026-06-12 review, Task 3 — E-1/E-2/E-3): precomputed per-mode
        NNZ grouping for this (static) tensor. When supplied, the per-iteration
        decode, ``cp.unique`` sort, and per-batch ``cp.where`` scan are skipped
        (the NNZ are already column-grouped). Must be ``None`` under stochastic
        subsampling.
    """
    # Sparse unfolding X_(mode)
    # X = unfold_from_vectorized_sparse(vec_tensor, shape, mode).tocoo()
    # rows = X.row
    # cols = X.col
    # vals = X.data
    if verbose:
        print(f"  Updating factor {mode}...")

    other_modes = [m for m in range(len(shape)) if m != mode]

    if grouping is not None:
        # Cached path (Task 3): NNZ already decoded, sorted and grouped by column.
        ucols = grouping.ucols
        segment_offsets = grouping.segment_offsets
        rows_sorted = grouping.rows_sorted
        vals_sorted = grouping.vals_sorted
        col_index = grouping.col_index
        inv = None
    else:
        # Uncached path: decode + group this iteration (subsampled tensors).
        idxs, vals = coo_to_coords(vec_tensor, shape)

        rows = idxs[mode]

        other_coords = [idxs[m] for m in other_modes]
        # build a safe unfolded-column id in int64 only for grouping
        other_shape = tuple(shape[m] for m in other_modes)
        cols = safe_ravel(tuple(other_coords), other_shape, cp)

        ucols, inv = cp.unique(cols, return_inverse=True)

    A = factors[mode]  # (I_mode, R_mode)

    if masked:
        # Masked objective: denominator is accumulated over observed entries only
        # (see batch loop below).
        denominator_acc = cp.zeros_like(A)
    else:
        # ---- Denominator part: Gram = Z^T Z exactly, no Z materialization
        Gram = _tucker_gram_ZtZ(core, factors, mode, epsilon=epsilon)  # (R, R)
        denominator = A @ Gram
        denominator = cp.clip(denominator, a_min=epsilon, a_max=None)

    # ---- Numerator part: numerator = X @ Z via batching unique columns, no full Z
    numerator = cp.zeros_like(A)

    # CHANGED (2026-06-12 review, Task 1): estimate AFTER all NNZ bookkeeping
    # (flat, idxs, cols, ucols, inv) is live on the GPU so the free-memory
    # snapshot reflects the actual headroom. Previously estimated at the top
    # of the function. Ported from the sharded path
    # (sharded_sparse.py::_partial_numerator_for_shard).
    if batch_cols is None:
        batch_cols = _estimate_batch_cols_for_Z(core, factors, mode, masked=masked)

    n_ucols = int(ucols.size)
    col_batches = range(0, n_ucols, int(batch_cols))
    if verbose:
        col_batches = tqdm(col_batches, desc=f"  Factor {mode} col-batches", unit="batch", leave=False)
    for start in col_batches:
        end = min(start + int(batch_cols), n_ucols)
        u = ucols[start:end]

        _, idxs_by_mode = _unravel_cols_for_mode(u, shape, mode)

        # Z_u: (m, R_mode)  where row t is Z[column=u[t], :]
        Z_u = compute_Zcols_batch(
            core=core,
            factors=factors,
            mode=mode,
            other_modes=other_modes,
            idxs_by_mode=idxs_by_mode,
            epsilon=epsilon,
        )

        # nnz entries belonging to these unique columns
        if grouping is not None:
            # E-2: contiguous slice of the column-grouped arrays — no full scan.
            seg_lo = int(segment_offsets[start])
            seg_hi = int(segment_offsets[end])
            if seg_hi == seg_lo:
                continue
            r_i = rows_sorted[seg_lo:seg_hi]   # (nnz_b,)
            v_i = vals_sorted[seg_lo:seg_hi]   # (nnz_b,)
            u_i = col_index[seg_lo:seg_hi] - start
        else:
            nz_idx = cp.where((inv >= start) & (inv < end))[0]
            if nz_idx.size == 0:
                continue
            r_i = rows[nz_idx]          # (nnz_b,)
            v_i = vals[nz_idx]          # (nnz_b,)
            u_i = inv[nz_idx] - start   # local index into this batch [0..m)

        nnz_b = int(r_i.size)

        if use_legacy_factor_batch():
            # Legacy body (pre 2026-07-29), kept behind
            # TENSORMET_LEGACY_FACTOR_BATCH=1 for A/B validation.
            Z_rows = Z_u[u_i]           # (nnz_b, R_mode)

            # numerator[row] += X_ij * Z[j,:]  — cuSPARSE SpMM (no serialised atomics)
            col_idx_b = cp.arange(nnz_b, dtype=cp.int32)
            row_idx_b = r_i.astype(cp.int32)
            S_b = cpx_sparse.csr_matrix(
                (v_i, (row_idx_b, col_idx_b)),
                shape=(numerator.shape[0], nnz_b),
            )
            numerator += S_b @ Z_rows

            if masked:
                # Xhat at these observed entries = <A[row], Z[col]>
                xhat_b = cp.sum(A[r_i] * Z_rows, axis=1)  # (nnz_b,)
                S_den = cpx_sparse.csr_matrix(
                    (xhat_b, (row_idx_b, col_idx_b)),
                    shape=(numerator.shape[0], nnz_b),
                )
                denominator_acc += S_den @ Z_rows
            continue

        # CHANGED (2026-07-29): scatter-free batch body — same restructuring as
        # kl_factor_update_largedim (see there). FR numerator weights are the
        # raw values v_i, so no sampled row-dot is needed unless masked.
        m_b = end - start
        if grouping is not None:
            indptr_b = (segment_offsets[start:end + 1] - seg_lo).astype(cp.int32)
        else:
            # Uncached batches must be column-grouped up front so P.data order
            # equals entry order (see group_batch_by_column).
            indptr_b, u_i, r_i, v_i = group_batch_by_column(u_i, m_b, r_i, v_i)

        P = build_batch_csr_T(v_i, r_i, m_b, numerator.shape[0], indptr_b)
        numerator += spmm_T(P, Z_u)

        if masked:
            # Xhat at these observed entries = <A[row], Z[col]>
            xhat_b = sampled_row_dots(A, Z_u, r_i, u_i)  # (nnz_b,)
            P_den = same_pattern_csr(P, xhat_b)
            denominator_acc += spmm_T(P_den, Z_u)

    if masked:
        denominator = cp.clip(denominator_acc, a_min=epsilon, a_max=None)

    # MU update
    A_new = A * (numerator / (denominator + 1e-12))
    A_new = cp.clip(A_new, a_min=epsilon, a_max=None)
    return A_new


def fr_core_update_largedim(
    vec_tensor,
    shape,
    core,
    factors,
    modes=None,              # assumes all modes (same constraint style as your KL v2)
    thread_budget=None,      # kept for API compatibility
    epsilon=1e-12,
    batch_num=None,
    verbose=False,
    masked=False,
):
    """
    Frobenius (Euclidean) multiplicative update for the Tucker core WITHOUT dense recon.

    Equivalent to your original function C:
        numerator   = X ×_n A_n^T
        denominator = core ×_n (A_n^T A_n)
        core *= numerator / denominator

    but numerator is accumulated by streaming NNZ and building only core-sized tensors.

    masked : bool
        If False (default), the denominator core ×_n (A_n^T A_n) sums over ALL
        entries. If True, the denominator is restricted to observed entries:
        Den = sum_{k in Omega} Xhat_k * outer(factor rows), accumulated alongside
        the numerator. This is the weighted/completion objective.
    """
    if verbose:
        print("  Updating core...")

    shape = tuple(int(s) for s in shape)
    N = len(shape)

    if modes is None:
        modes = list(range(N))
    if list(modes) != list(range(N)):
        raise NotImplementedError("This version assumes modes == all modes (0..N-1).")

    # --- decode NNZ coordinates (same approach as KL core v2) ---
    idxs, xvals = coo_to_coords(vec_tensor, shape)  # list length N, each (nnz,)
    nnz = int(xvals.size)
    if nnz == 0:
        return core

    # --- numerator: core-shaped accumulator, streamed over NNZ ---
    Num = cp.zeros_like(core)
    Den = cp.zeros_like(core) if masked else None

    # CHANGED (2026-06-12 review, Task 1): estimate AFTER the NNZ bookkeeping
    # (flat, xvals, idxs, Num/Den) is live so the free-memory snapshot is
    # accurate. Previously estimated at the top of the function. Ported from
    # the sharded path (sharded_sparse.py::_partial_core_num_for_shard).
    if batch_num is None:
        batch_num = _estimate_batch_num_for_outer(core, factors)

    # small batches keep peak memory down (like your pass-2 accumulator)
    num_batches = range(0, nnz, int(batch_num))
    if verbose:
        num_batches = tqdm(num_batches, desc="  Core numerator pass", unit="batch", leave=False)
    for start in num_batches:
        end = min(start + int(batch_num), nnz)
        mats = [factors[n][idxs[n][start:end]] for n in range(N)]  # each (b, Rn)
        w = xvals[start:end]  # Frobenius numerator uses X directly (no X/R like KL)
        _accumulate_core_num_outer(Num, w, mats)
        if masked:
            # Xhat at observed entries, then accumulate the X̂-weighted outer products.
            xhat_b = _rhat_from_factor_rows_sequential(core, mats, epsilon=epsilon)  # (b,)
            _accumulate_core_num_outer(Den, xhat_b, mats)

    Num = cp.clip(Num, a_min=epsilon, a_max=None)

    if not masked:
        # --- denominator: rank-space multilinear product with Gram matrices ---
        grams = [factors[n].T @ factors[n] for n in range(N)]  # each (R_n, R_n)
        Den = _core_multilinear_grams(core, grams, epsilon=epsilon)  # core-shaped

    # --- MU update ---
    core_new = core * (Num / (Den + epsilon))
    return core_new

def fr_combined_core_errors_largedim(
    vec_tensor,
    shape,
    core,
    factors,
    modes=None,
    thread_budget=None,
    epsilon=1e-12,
    batch_num=None,
    verbose=False,
    masked=False,
):
    """FR core update + Frobenius error in one pass, sharing Gram matrices.

    Equivalent to calling ``fr_core_update_largedim`` then
    ``fr_compute_errors_largedim`` back-to-back, but computes
    ``grams = [A_n^T A_n]`` and ``Den`` only once instead of twice.
    Mirrors ``fr_combined_core_errors`` for the small-dim path.

    masked : bool
        If True, the core denominator and the reported error are restricted to
        observed entries (weighted/completion objective): the error is the
        relative RMSE over observed entries, sqrt(sum_Omega (x - xhat)^2) / ||X||.

    Returns
    -------
    (core_new, rel_err)
    """
    if verbose:
        print("  Updating core + computing Frobenius errors...")

    shape = tuple(int(s) for s in shape)
    N = len(shape)

    if modes is None:
        modes = list(range(N))
    if list(modes) != list(range(N)):
        raise NotImplementedError("This version assumes modes == all modes (0..N-1).")

    idxs, xvals = coo_to_coords(vec_tensor, shape)
    nnz = int(xvals.size)
    if nnz == 0:
        return core, cp.asarray(0.0, dtype=core.dtype)

    Num = cp.zeros_like(core)
    Den_masked = cp.zeros_like(core) if masked else None
    # Masked error accumulators (observed-only residual).
    residual_sq = cp.asarray(0.0, dtype=core.dtype)
    norm_X_sq = cp.asarray(0.0, dtype=core.dtype)

    # CHANGED (2026-06-12 review, Task 1): estimate AFTER the NNZ bookkeeping
    # (flat, xvals, idxs, Num/Den_masked) is live so the free-memory snapshot
    # is accurate. Previously estimated at the top of the function.
    if batch_num is None:
        batch_num = _estimate_batch_num_for_outer(core, factors)

    num_batches = range(0, nnz, int(batch_num))
    if verbose:
        num_batches = tqdm(num_batches, desc="  Core numerator pass", unit="batch", leave=False)
    for start in num_batches:
        end = min(start + int(batch_num), nnz)
        mats = [factors[n][idxs[n][start:end]] for n in range(N)]
        x_b = xvals[start:end]
        _accumulate_core_num_outer(Num, x_b, mats)
        if masked:
            xhat_b = _rhat_from_factor_rows_sequential(core, mats, epsilon=epsilon)  # (b,)
            _accumulate_core_num_outer(Den_masked, xhat_b, mats)
            x_b_nn = cp.clip(x_b.astype(core.dtype), a_min=0.0, a_max=None)
            residual_sq += cp.sum((x_b_nn - xhat_b) ** 2)
            norm_X_sq += cp.sum(x_b_nn * x_b_nn)

    Num = cp.clip(Num, a_min=epsilon, a_max=None)

    if masked:
        # Masked objective: denominator and error restricted to observed entries.
        core_new = core * (Num / (Den_masked + epsilon))
        residual_sq = cp.maximum(residual_sq, 0.0)
        rel_err = cp.sqrt(residual_sq) / cp.maximum(cp.sqrt(cp.maximum(norm_X_sq, epsilon)), epsilon)
        return core_new, rel_err

    # Compute Gram matrices once; reuse for both the MU denominator and ||X̂||^2.
    grams = [factors[n].T @ factors[n] for n in range(N)]
    Den = _core_multilinear_grams(core, grams, epsilon=epsilon)

    core_new = core * (Num / (Den + epsilon))

    # Frobenius error terms — consistent with the small-dim fr_combined_core_errors:
    #   ||X̂||^2 uses the old Den (slight approx); <X, X̂> = <Num, core_new>.
    x_nz = cp.clip(xvals.astype(core.dtype), a_min=0.0, a_max=None)
    norm_X_sq = cp.sum(x_nz * x_nz)
    norm_X = cp.sqrt(cp.maximum(norm_X_sq, epsilon))

    norm_Xhat_sq = cp.sum(core_new * Den)
    inner_prod = cp.sum(Num * core_new)

    residual_sq = cp.maximum(norm_X_sq + norm_Xhat_sq - 2.0 * inner_prod, 0.0)
    rel_err = cp.sqrt(residual_sq) / cp.maximum(norm_X, epsilon)

    return core_new, rel_err


def fr_compute_errors_largedim(
    vec_tensor,
    shape,
    core,
    factors,
    thread_budget=None,     # API compatibility; unused
    epsilon=1e-12,
    batch_rhat=1000,        # same role as in KL error
    verbose=False,
    masked=False,
):
    """
    Relative Frobenius error ||X - X̂||_F / ||X||_F for sparse X,
    WITHOUT forming dense X̂.

    !! Still has some rounding differences compared to the original !!

    Uses:
      - ||X||_F^2 = sum_{nz} x^2
      - <X, X̂>   = sum_{nz} x * x̂  where x̂ computed at nz by Tucker contraction
      - ||X̂||_F^2 = <core, core ×_n (A_n^T A_n)> (exact, no dense X̂)

    masked : bool
        If True, compute the relative RMSE over observed entries only,
        sqrt(sum_Omega (x - x̂)^2) / ||X||, ignoring the implicit-zero
        contribution (weighted/completion objective).
    """
    if verbose:
        print("  Computing Frobenius errors...")

    shape = tuple(int(s) for s in shape)
    N = len(shape)

    # --- decode NNZ (same helpers as your KL largedim) ---
    idxs, x_nz = coo_to_coords(vec_tensor, shape)  # list of N arrays, each (nnz,)
    nnz = int(x_nz.size)

    x_nz = cp.asarray(x_nz)
    # Frobenius is fine with zeros, but keep nonneg pipeline consistent
    x_nz = cp.clip(x_nz, a_min=0.0, a_max=None)

    # --- ||X||_F ---
    norm_X_sq = cp.sum(x_nz * x_nz)
    norm_X = cp.sqrt(cp.maximum(norm_X_sq, epsilon))

    # Edge case: X is all zeros (relative error ill-defined); mirror your KL style
    if nnz == 0 or float(norm_X_sq.get()) == 0.0:
        # Return ||X̂||/max(||X||,eps) == ||X̂||/eps
        grams = [factors[n].T @ factors[n] for n in range(N)]
        Den = _core_multilinear_grams(core, grams, epsilon=epsilon)
        norm_Xhat_sq = cp.sum(core * Den)
        norm_Xhat = cp.sqrt(cp.maximum(norm_Xhat_sq, epsilon))
        return norm_Xhat / cp.maximum(norm_X, epsilon)

    # CHANGED (2026-06-12 review, Task 1): estimate AFTER the NNZ bookkeeping
    # (x_nz, idxs) is live so the free-memory snapshot is accurate.
    # Previously estimated at the top of the function (only triggered when the
    # caller passed batch_rhat=None explicitly; the default is 1000).
    if batch_rhat is None:
        batch_rhat = _estimate_batch_rhat_for_tensordot(core, factors)

    # --- compute xhat_nz in batches (same technique as KL error) ---
    inner_prod = cp.asarray(0.0, dtype=core.dtype)
    masked_residual_sq = cp.asarray(0.0, dtype=core.dtype)

    rhat_batches = range(0, nnz, int(batch_rhat))
    if verbose:
        rhat_batches = tqdm(rhat_batches, desc="  Frobenius error r_hat pass", unit="batch", leave=False)
    for start in rhat_batches:
        end = min(start + int(batch_rhat), nnz)
        mats = [factors[n][idxs[n][start:end]] for n in range(N)]  # (b, Rn)
        xhat_b = _rhat_from_factor_rows_sequential(core, mats, epsilon=epsilon)  # (b,)
        if masked:
            # observed-only residual sum of squares
            masked_residual_sq += cp.sum((x_nz[start:end] - xhat_b) ** 2)
        else:
            # <X, X̂> batch contribution
            inner_prod += cp.sum(x_nz[start:end] * xhat_b)

    if masked:
        # Masked/completion objective: relative RMSE over observed entries only.
        residual_norm = cp.sqrt(cp.maximum(masked_residual_sq, 0.0))
        return residual_norm / cp.maximum(norm_X, epsilon)

    # --- ||X̂||_F^2 exactly (no dense X̂, no mode_dot) ---
    grams = [factors[n].T @ factors[n] for n in range(N)]  # (R_n, R_n)
    Den = _core_multilinear_grams(core, grams, epsilon=epsilon)  # core-shaped
    norm_Xhat_sq = cp.sum(core * Den)

    # --- ||X - X̂||_F^2 = ||X||^2 + ||X̂||^2 - 2<X, X̂> ---
    residual_sq = norm_X_sq + norm_Xhat_sq - 2.0 * inner_prod
    residual_sq = cp.maximum(residual_sq, 0.0)
    residual_norm = cp.sqrt(residual_sq)

    return residual_norm / cp.maximum(norm_X, epsilon)