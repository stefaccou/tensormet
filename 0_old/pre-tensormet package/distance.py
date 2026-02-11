import numpy as np
import cupy as cp
import cupyx.scipy.sparse as cpx_sparse

import tensorly as tl
import pytensorlab as ptl

from tensorly.base import unfold
# from tensorly_custom.decomposition._tucker import tucker_to_tensor
from tensorly.tucker_tensor import tucker_to_tensor
from tensorly.tenalg import mode_dot


from sparse_ops import (
    unfold_from_vectorized_sparse,
    sparse_multi_mode_dot_vec,
    ptl_tucker_to_tensor,
    gather_dense_at_block_nz
)
from utils import ThreadBudget, print_elapsed_time

import time
from dataclasses import dataclass
from typing import Callable, Optional, Literal

# -- Kullback-Leibler Divergence --

def kl_factor_update(vec_tensor, core, factors, mode, shape, thread_budget=None, epsilon=1e-12):
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

def kl_factor_update_largedim(vec_tensor, core, factors, mode, shape, thread_budget, epsilon=1e-12):
    """
    One multiplicative KL update for a single factor matrix A_n (for `mode`).
    Use when dimensions become large to fully avoid reconstruction on GPU.

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
    X = unfold_from_vectorized_sparse(vec_tensor, shape, mode)
    core_np = tl.to_numpy(core)
    factors_np = [tl.to_numpy(f) for f in factors]
    # Dense reconstruction excluding current factor, unfolded along mode
    tucker = ptl.TuckerTensor(core=core_np,
                                  factors=factors_np)
    with thread_budget.limit():
        Z = ptl_tucker_to_tensor(tucker, skip_factor=mode)
    Z = ptl.tens2mat(Z, mode)


    A = factors[mode]  # (I_mode, R)
    rows = X.row
    cols = X.col
    vals = X.data

    # Compute reconstruction only at nonzeros: R_nz = sum_r A[i,r] * Z[r,j]
    # A_rows: (nnz, R)
    A_rows = A[rows, :]

    Z_cols_T = tl.transpose(Z[:, cols.get()])
    Z_cols_T = cp.asarray(Z_cols_T)

    R_nz = tl.sum(A_rows * Z_cols_T, axis=1)
    R_nz = tl.clip(R_nz, a_min=epsilon, a_max=None)


    # W = X / (A Z) at nonzeros

    W_data = vals / R_nz
    W = cpx_sparse.coo_matrix((W_data, (rows, cols)), shape=X.shape)

    numerator = W @ tl.transpose(cp.asarray(Z))


    # denominator = sum_j Z[r,j] broadcast to (I_mode, R)
    # den_row = tl.sum(Z, axis=1)  # (R,)

    den_row = tl.sum(Z, axis=1)
    denominator = den_row[np.newaxis, :]
    denominator = tl.clip(denominator, a_min=epsilon, a_max=None)

    # Multiplicative update
    A = A * (numerator / (cp.asarray(denominator + 1e-12)))
    A = tl.clip(A, a_min=epsilon, a_max=None)
    return A


def kl_core_update(vec_tensor, shape, core, factors, modes, thread_budget, epsilon=1e-12):
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
    # Build a CPU Tucker object for pytensorlab reconstruction
    core_np = tl.to_numpy(core)
    factors_np = [tl.to_numpy(f) for f in factors]
    tucker = ptl.TuckerTensor(core=core_np, factors=factors_np)

    with thread_budget.limit():
        R = ptl_tucker_to_tensor(tucker)

    # Gather reconstructed values at nonzero coordinates of vec_tensor
    data = gather_dense_at_block_nz(R, vec_tensor, shape)
    data = tl.clip(data, a_min=epsilon, a_max=None)

    # X/R at nonzeros
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
    core *= X_R / (F + epsilon)
    return core


def kl_compute_errors(
        vec_tensor: cpx_sparse.spmatrix,
        shape,
        core,
        factors,
        thread_budget: ThreadBudget,
        epsilon=1e-12,
):
    """Generalised KL divergence C_KL(X || R) for sparse X.

    vec_tensor : vectorised sparse X (block-encoded, same as 'tensor')
    shape      : original N-D shape
    core       : current core G
    factors    : list of factor matrices A^{(n)}
    """

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
def fr_factor_update(vec_tensor, core, factors, mode, shape, thread_budget=None, epsilon=1e-12):
    # This still explodes! The created B tensor does not always fit in memory

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

def fr_factor_update_largedim(vec_tensor, core, factors, mode, shape, thread_budget=None, epsilon=1e-12):
    # This still explodes! The created B tensor does not always fit in memory

    X = unfold_from_vectorized_sparse(vec_tensor, shape,
                                             mode)  # this is the same as B when using dense tensor!
    core_np = tl.to_numpy(core)
    factors_np = [tl.to_numpy(f) for f in factors]
    # Dense reconstruction excluding current factor, unfolded along mode
    tucker = ptl.TuckerTensor(core=core_np,
                              factors=factors_np)
    Z = ptl_tucker_to_tensor(tucker, skip_factor=mode)
    Z = ptl.tens2mat(Z, mode)

    # todo: batched multiplication!
    raise NotImplementedError("implement sparse x sparse multiplication with batching")
    numerator = X @ Z  # cupy sparse @ dense
    numerator = tl.clip(numerator, a_min=epsilon, a_max=None)
    A = factors[mode]
    denominator = tl.dot(A, tl.dot(tl.transpose(Z), Z))
    denominator = tl.clip(denominator, a_min=epsilon, a_max=None)
    A *= numerator / denominator
    A = tl.clip(A, a_min=epsilon, a_max=None)
    return A

def fr_core_update(vec_tensor, shape, core, factors, modes, thread_budget=None, epsilon=1e-12):
    """
    One multiplicative KL update for the core tensor.

    [DESCRIBE]

    Returns
    -------
    core : updated core.
    """

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

    core *= numerator / (denominator + epsilon)
    return core

def fr_compute_errors(
        vec_tensor: cpx_sparse.spmatrix,
        shape,
        core,
        factors,
        thread_budget: ThreadBudget,
        epsilon=1e-12,
):
    """Relative Frobenius error ||X - X̂||_F / ||X||_F for sparse X.

    vec_tensor : vectorised sparse X (block-encoded, same as 'tensor')
    shape      : original N-D shape
    core       : current core G
    factors    : list of factor matrices A^{(n)}
    """

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

def fr_combined_core_errors(vec_tensor, shape, core, factors, modes, thread_budget=None, epsilon=1e-12):
    """
        One multiplicative KL update for the core tensor.

        [DESCRIBE]

        Returns
        -------
        core : updated core.
        """

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

    core *= numerator / (denominator + epsilon)

    # error_start = time.time()
    tensor_coo = vec_tensor.tocoo()
    norm_tensor = cp.sqrt((cp.abs(tensor_coo.data) ** 2).sum())
    # norm_time = print_elapsed_time(error_start, "norm calculation")
    norm_X_sq = norm_tensor ** 2
    norm_Xhat_sq = tl.sum(core * denominator)
    inner_prod = tl.sum(numerator * core)
    residual_norm = tl.sqrt(norm_X_sq + norm_Xhat_sq - 2 * inner_prod)
    relative_error = residual_norm / norm_tensor
    # end = print_elapsed_time(norm_time, "full error calculation")
    return core, relative_error

def null_compute_errors(vec_tensor: cpx_sparse.spmatrix,
        shape,
        core,
        factors,
        thread_budget: ThreadBudget,
        epsilon=1e-12,) -> None:
    # takes the same input, but returns nothing
    return

