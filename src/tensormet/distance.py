import math
import itertools

from tqdm import tqdm
import numpy as np
# import cupy as cp
# import cupyx.scipy.sparse as cpx_sparse

import tensorly as tl
import pytensorlab as ptl

from tensorly.base import unfold
from tensorly.tucker_tensor import tucker_to_tensor
from tensorly.tenalg import mode_dot

from tensormet.sparse_ops import (
    unfold_from_vectorized_sparse,
    sparse_multi_mode_dot_vec,
    ptl_tucker_to_tensor,
    gather_dense_at_block_nz,
    safe_ravel,
    compute_Zcols_batch
)
from tensormet.utils import ThreadBudget, einsum_letters, guarded_cupy_import

cp, cpx_sparse = guarded_cupy_import()

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
    den_row = cp.einsum(eq, *operands)
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
    """
    N = core.ndim
    b = mats[0].shape[0]

    # Start by contracting mode 0 to introduce batch dimension:
    # tmp[b, R1, R2, ...] = sum_{r0} mats0[b,r0] * core[r0, R1, ...]
    tmp = cp.tensordot(mats[0], core, axes=(1, 0))  # (b, R1, R2, ..., R_{N-1})

    # Then fold in remaining modes with multiply+sum over the next rank axis each time.
    for n in range(1, N):
        # tmp has shape (b, Rn, R_{n+1}, ...)
        # multiply by mats[n] broadcasted onto axis=1, then sum over axis=1
        shp = (b, mats[n].shape[1]) + (1,) * (tmp.ndim - 2)
        tmp = cp.sum(tmp * mats[n].reshape(shp), axis=1)

    r_hat = cp.clip(tmp, a_min=epsilon, a_max=None)  # (b,)
    return r_hat

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

    loop_ranks = [mats[i].shape[1] for i in loop_modes]
    for loop_idx in itertools.product(*[range(r) for r in loop_ranks]):
        v = w.copy()
        for loop_i, r in zip(loop_modes, loop_idx):
            v *= mats[loop_i][:, r]

        if KR_L is not None and KR_R is not None:
            # Massive Matrix Multiplication
            slice_sum = (KR_L * v[:, None]).T @ KR_R

            # Auto-align the dimensions
            full_slice = [slice(None) if i in left_modes + right_modes else loop_idx[loop_modes.index(i)] for i in
                          range(N)]
            Num[tuple(full_slice)] += slice_sum.reshape([mats[i].shape[1] for i in left_modes + right_modes])

        elif KR_L is not None:
            slice_sum = cp.sum(KR_L * v[:, None], axis=0)
            full_slice = [slice(None) if i in left_modes else loop_idx[loop_modes.index(i)] for i in range(N)]
            Num[tuple(full_slice)] += slice_sum.reshape([mats[i].shape[1] for i in left_modes])


def _blocked_coo_to_flat_indices(vec_tensor, orig_shape):
    orig_shape = tuple(orig_shape)
    size = int(np.prod(orig_shape))
    int32_max = np.iinfo(np.int32).max
    block_size = min(size, int32_max)

    coo = vec_tensor.tocoo()
    flat = coo.row.astype(cp.int64) + coo.col.astype(cp.int64) * cp.int64(block_size)
    vals = coo.data
    return flat, vals


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
    sum_R = cp.einsum(eq, core, *sums)
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

    Gp = cp.einsum(eq, core, *grams)  # same shape as core, but other modes live in primed space

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
    tmp = core
    N = core.ndim
    for n in range(N):
        G = grams[n]  # (R_n, R_n)
        # tensordot over core axis n: (R_n,R_n) x (...,R_n,...) -> (R_n, ..., ...)
        tmp = cp.tensordot(G, tmp, axes=(1, n))
        # tensordot brings the new R_n axis to the front; move it back to position n
        tmp = cp.moveaxis(tmp, 0, n)
    tmp = cp.clip(tmp, a_min=epsilon, a_max=None)
    return tmp


# batch estimation helpers
def _gpu_free_bytes():
    """
    Conservative 'free bytes now' estimate.
    Flushes the CuPy memory pool first to get an accurate driver-level reading.
    """
    # 1. Force CuPy to return all cached/unused memory to the CUDA driver
    cp.get_default_memory_pool().free_all_blocks()

    # 2. Now ask the driver how much memory is actually free
    free_b, total_b = cp.cuda.runtime.memGetInfo()
    return int(free_b)

def _estimate_batch_num_for_outer(core, factors, safety=0.80, temp_mult=2.0):
    """
    New estimator for the optimized matrix-multiplication accumulator.
    It no longer assumes the materialization of the full core outer product!
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

    free_b = int(_gpu_free_bytes())
    budget_b = int(free_b * safety)

    b = max(1, budget_b // max(1, bytes_per_b))

    # Hard cap to prevent grid/memory timeout issues
    hard_cap = max(1, int(1_000_000_000 // max(1, bytes_per_b)))
    return min(int(b), hard_cap)


def _estimate_batch_rhat_for_tensordot(core, factors, safety=0.60, temp_mult=4.0):  # Increased temp_mult
    N = core.ndim
    R = [int(factors[n].shape[1]) for n in range(N)]
    dtype = core.dtype
    itemsize = int(np.dtype(dtype).itemsize)

    # The bottleneck: (batch, R1, R2...)
    prod_rest = math.prod(R[1:])
    tmp_bytes_per_b = prod_rest * itemsize

    # Total bytes per batch element
    bytes_per_b = int(np.ceil(tmp_bytes_per_b * temp_mult))

    free_b = _gpu_free_bytes()
    # Ensure we leave a large buffer for the rest of the graph
    budget_b = int(free_b * safety)

    b = budget_b // max(1, bytes_per_b)
    return max(1, int(b))


def _estimate_batch_cols_for_Z(core, factors, mode, safety=0.80, temp_mult=2.0):
    """
    Estimate safe batch size for compute_Zcols_batch.
    Uses pure Python math to avoid numpy 32-bit overflows and sets a hard cap.
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

    free_b = int(_gpu_free_bytes())
    budget_b = int(free_b * safety)

    b = max(1, budget_b // max(1, bytes_per_b))

    # HARD CAP: Prevent CUDA grid index limits from silently corrupting memory
    # Cap peak temporary allocation to ~4GB per batch
    hard_cap = max(1, int(2_000_000_000 // max(1, tmp_bytes_per_b)))

    return min(int(b), hard_cap)

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
):
    """
    KL multiplicative update for Tucker factor A^(mode) WITHOUT building dense Z,
    but mathematically equivalent to your dense-Z implementation:

        A <- A * ( (W @ Z.T) / sum_j Z[:, j] )

    where W_ij = X_ij / (A Z)_ij at the nonzeros of X_(mode).

    Works for N-way tensors (N = len(shape) = core.ndim).
    """

    # Sparse unfolding X_(mode)
    # X = unfold_from_vectorized_sparse(vec_tensor, shape, mode).tocoo()
    # rows = X.row
    # cols = X.col
    # vals = X.data

    if verbose:
        print(f"  Updating factor {mode}...")

    if batch_cols is None:
        batch_cols = _estimate_batch_cols_for_Z(core, factors, mode)
    # print("batch cols:", batch_cols)

    # new: avoid X buildup for large dimensions
    flat, vals = _blocked_coo_to_flat_indices(vec_tensor, shape)
    idxs = _unravel_flat_indices_C(flat, shape)

    rows = idxs[mode]

    other_modes = [m for m in range(len(shape)) if m != mode]
    other_coords = [idxs[m] for m in other_modes]

    # build a safe unfolded-column id in int64 only for grouping
    other_shape = tuple(shape[m] for m in other_modes)
    cols = safe_ravel(tuple(other_coords), other_shape, cp)

    A = factors[mode]  # (I_mode, R_mode)

    # Exact denominator over ALL columns (no approximation)
    den_row = _tucker_den_row_full(core, factors, mode, epsilon=epsilon)
    denominator = den_row[None, :]  # (1, R_mode)

    # Accumulate numerator = W @ Z.T without building full Z
    numerator = cp.zeros_like(A)

    # Reuse computations across repeated columns
    ucols, inv = cp.unique(cols, return_inverse=True)

    # Decode unique columns once per batch (general N-way unravel)
    other_modes = [m for m in range(len(shape)) if m != mode]

    col_batches = range(0, int(ucols.size), int(batch_cols))
    if verbose:
        col_batches = tqdm(col_batches, desc=f"  Factor {mode} col-batches", unit="batch", leave=False)
    for start in col_batches:
        end = min(start + int(batch_cols), int(ucols.size))
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
        nz_idx = cp.where((inv >= start) & (inv < end))[0]
        if nz_idx.size == 0:
            continue

        r_i = rows[nz_idx]             # (nnz_b,)
        v_i = vals[nz_idx]             # (nnz_b,)
        u_i = inv[nz_idx] - start      # local [0..m)

        A_rows = A[r_i]                # (nnz_b, R_mode)
        Z_rows = Z_u[u_i]              # (nnz_b, R_mode)

        # (A Z)_nz
        R_nz = cp.sum(A_rows * Z_rows, axis=1)
        R_nz = cp.clip(R_nz, a_min=epsilon, a_max=None)

        W_data = v_i / R_nz            # (nnz_b,)

        # numerator[row] += W * Z  — cuSPARSE SpMM (no serialised atomics)
        nnz_b = int(r_i.size)
        S_b = cpx_sparse.csr_matrix(
            (W_data, (r_i.astype(cp.int32), cp.arange(nnz_b, dtype=cp.int32))),
            shape=(numerator.shape[0], nnz_b),
        )
        numerator += S_b @ Z_rows

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
):
    if verbose:
        print("  Updating core...")

    if batch_rhat is None:
        batch_rhat = _estimate_batch_rhat_for_tensordot(core, factors)
    if batch_num is None:
        batch_num = _estimate_batch_num_for_outer(core, factors)
    shape = tuple(int(s) for s in shape)
    N = len(shape)
    if modes is None:
        modes = list(range(N))
    if list(modes) != list(range(N)):
        raise NotImplementedError("This version assumes modes == all modes (0..N-1).")
    flat, xvals = _blocked_coo_to_flat_indices(vec_tensor, shape)
    nnz = int(flat.size)
    if nnz == 0:
        return core
    idxs = _unravel_flat_indices_C(flat, shape)  # list length N, each (nnz,)

    # Denominator is outer product of column sums, but don't materialize F.
    sums = [cp.clip(cp.sum(factors[n], axis=0), a_min=epsilon, a_max=None) for n in range(N)]
    Num = cp.zeros_like(core)

    # --- Pass 1: compute w = x / r_hat in big batches, stash w (or stream into pass 2)
    # Stashing w costs nnz floats; if that's too big, you can stream (see note below).
    w_all = cp.empty_like(xvals)

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
):
    """
    Relative generalized KL divergence C_KL(X || R) for sparse X,
    WITHOUT forming dense R, staying close to the core-update approach.

    Computes:
      KL = sum_{nz} [x log(x/r) - x + r] + (sum_R - sum_{nz} r)
      rel_KL = KL / sum_{nz} x
    """
    if verbose:
        print("  Computing KL errors...")

    if batch_rhat is None:
        batch_rhat = _estimate_batch_rhat_for_tensordot(core, factors)

    shape = tuple(int(s) for s in shape)
    N = len(shape)

    flat, x_nz = _blocked_coo_to_flat_indices(vec_tensor, shape)
    nnz = int(flat.size)
    if nnz == 0:
        # If X is all-zeros, KL reduces to sum_R. Relative term is ill-defined; mirror your style:
        sum_R = _tucker_sum_all_entries(core, factors, epsilon=epsilon)
        return sum_R / cp.maximum(cp.asarray(0.0, dtype=sum_R.dtype), epsilon)

    x_nz = cp.asarray(x_nz)
    x_nz = cp.clip(x_nz, a_min=epsilon, a_max=None)

    idxs = _unravel_flat_indices_C(flat, shape)  # list of N arrays, each (nnz,)

    # --- compute r_nz in batches (like your core update r_hat pass) ---
    r_nz = cp.empty_like(x_nz)
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
):
    """
    Frobenius (Euclidean) multiplicative update for Tucker factor A^(mode)
    WITHOUT building dense Z, but equivalent to your dense function 3:

        numerator   = X @ Z
        denominator = A @ (Z^T Z)
        A <- A * numerator / denominator

    where X is the sparse unfolding and Z = transpose(unfold(tucker_to_tensor(skip_factor=mode), mode)).
    """
    # Sparse unfolding X_(mode)
    # X = unfold_from_vectorized_sparse(vec_tensor, shape, mode).tocoo()
    # rows = X.row
    # cols = X.col
    # vals = X.data
    if verbose:
        print(f"  Updating factor {mode}...")

    if batch_cols is None:
        batch_cols = _estimate_batch_cols_for_Z(core, factors, mode)
    # new: avoid X buildup for large dimensions
    flat, vals = _blocked_coo_to_flat_indices(vec_tensor, shape)
    idxs = _unravel_flat_indices_C(flat, shape)

    rows = idxs[mode]

    other_modes = [m for m in range(len(shape)) if m != mode]
    other_coords = [idxs[m] for m in other_modes]

    # build a safe unfolded-column id in int64 only for grouping
    other_shape = tuple(shape[m] for m in other_modes)
    cols = safe_ravel(tuple(other_coords), other_shape, cp)

    A = factors[mode]  # (I_mode, R_mode)

    # ---- Denominator part: Gram = Z^T Z exactly, no Z materialization
    Gram = _tucker_gram_ZtZ(core, factors, mode, epsilon=epsilon)  # (R, R)
    denominator = A @ Gram
    denominator = cp.clip(denominator, a_min=epsilon, a_max=None)

    # ---- Numerator part: numerator = X @ Z via batching unique columns, no full Z
    numerator = cp.zeros_like(A)

    ucols, inv = cp.unique(cols, return_inverse=True)
    other_modes = [m for m in range(len(shape)) if m != mode]

    col_batches = range(0, int(ucols.size), int(batch_cols))
    if verbose:
        col_batches = tqdm(col_batches, desc=f"  Factor {mode} col-batches", unit="batch", leave=False)
    for start in col_batches:
        end = min(start + int(batch_cols), int(ucols.size))
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
        nz_idx = cp.where((inv >= start) & (inv < end))[0]
        if nz_idx.size == 0:
            continue

        r_i = rows[nz_idx]          # (nnz_b,)
        v_i = vals[nz_idx]          # (nnz_b,)
        u_i = inv[nz_idx] - start   # local index into this batch [0..m)

        Z_rows = Z_u[u_i]           # (nnz_b, R_mode)

        # numerator[row] += X_ij * Z[j,:]  — cuSPARSE SpMM (no serialised atomics)
        nnz_b = int(r_i.size)
        S_b = cpx_sparse.csr_matrix(
            (v_i, (r_i.astype(cp.int32), cp.arange(nnz_b, dtype=cp.int32))),
            shape=(numerator.shape[0], nnz_b),
        )
        numerator += S_b @ Z_rows

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
    batch_num=64,
    verbose=False,
):
    """
    Frobenius (Euclidean) multiplicative update for the Tucker core WITHOUT dense recon.

    Equivalent to your original function C:
        numerator   = X ×_n A_n^T
        denominator = core ×_n (A_n^T A_n)
        core *= numerator / denominator

    but numerator is accumulated by streaming NNZ and building only core-sized tensors.
    """
    if verbose:
        print("  Updating core...")

    if batch_num is None:
        batch_num = _estimate_batch_num_for_outer(core, factors)
    shape = tuple(int(s) for s in shape)
    N = len(shape)

    if modes is None:
        modes = list(range(N))
    if list(modes) != list(range(N)):
        raise NotImplementedError("This version assumes modes == all modes (0..N-1).")

    # --- decode NNZ coordinates (same approach as KL core v2) ---
    flat, xvals = _blocked_coo_to_flat_indices(vec_tensor, shape)
    nnz = int(flat.size)
    if nnz == 0:
        return core

    idxs = _unravel_flat_indices_C(flat, shape)  # list length N, each (nnz,)

    # --- numerator: core-shaped accumulator, streamed over NNZ ---
    Num = cp.zeros_like(core)
    # small batches keep peak memory down (like your pass-2 accumulator)
    num_batches = range(0, nnz, int(batch_num))
    if verbose:
        num_batches = tqdm(num_batches, desc="  Core numerator pass", unit="batch", leave=False)
    for start in num_batches:
        end = min(start + int(batch_num), nnz)
        mats = [factors[n][idxs[n][start:end]] for n in range(N)]  # each (b, Rn)
        w = xvals[start:end]  # Frobenius numerator uses X directly (no X/R like KL)
        _accumulate_core_num_outer(Num, w, mats)

    Num = cp.clip(Num, a_min=epsilon, a_max=None)

    # --- denominator: rank-space multilinear product with Gram matrices ---
    grams = [factors[n].T @ factors[n] for n in range(N)]  # each (R_n, R_n)
    Den = _core_multilinear_grams(core, grams, epsilon=epsilon)  # core-shaped

    # --- MU update ---
    core_new = core * (Num / (Den + epsilon))
    return core_new

def fr_compute_errors_largedim(
    vec_tensor,
    shape,
    core,
    factors,
    thread_budget=None,     # API compatibility; unused
    epsilon=1e-12,
    batch_rhat=1000,        # same role as in KL error
    verbose=False,
):
    """
    Relative Frobenius error ||X - X̂||_F / ||X||_F for sparse X,
    WITHOUT forming dense X̂.

    !! Still has some rounding differences compared to the original !!

    Uses:
      - ||X||_F^2 = sum_{nz} x^2
      - <X, X̂>   = sum_{nz} x * x̂  where x̂ computed at nz by Tucker contraction
      - ||X̂||_F^2 = <core, core ×_n (A_n^T A_n)> (exact, no dense X̂)
    """
    if verbose:
        print("  Computing Frobenius errors...")

    if batch_rhat is None:
        batch_rhat = _estimate_batch_rhat_for_tensordot(core, factors)
    shape = tuple(int(s) for s in shape)
    N = len(shape)

    # --- decode NNZ (same helpers as your KL largedim) ---
    flat, x_nz = _blocked_coo_to_flat_indices(vec_tensor, shape)
    nnz = int(flat.size)

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

    idxs = _unravel_flat_indices_C(flat, shape)  # list of N arrays, each (nnz,)

    # --- compute xhat_nz in batches (same technique as KL error) ---
    inner_prod = cp.asarray(0.0, dtype=core.dtype)

    rhat_batches = range(0, nnz, int(batch_rhat))
    if verbose:
        rhat_batches = tqdm(rhat_batches, desc="  Frobenius error r_hat pass", unit="batch", leave=False)
    for start in rhat_batches:
        end = min(start + int(batch_rhat), nnz)
        mats = [factors[n][idxs[n][start:end]] for n in range(N)]  # (b, Rn)
        xhat_b = _rhat_from_factor_rows_sequential(core, mats, epsilon=epsilon)  # (b,)
        # <X, X̂> batch contribution
        inner_prod += cp.sum(x_nz[start:end] * xhat_b)

    # --- ||X̂||_F^2 exactly (no dense X̂, no mode_dot) ---
    grams = [factors[n].T @ factors[n] for n in range(N)]  # (R_n, R_n)
    Den = _core_multilinear_grams(core, grams, epsilon=epsilon)  # core-shaped
    norm_Xhat_sq = cp.sum(core * Den)

    # --- ||X - X̂||_F^2 = ||X||^2 + ||X̂||^2 - 2<X, X̂> ---
    residual_sq = norm_X_sq + norm_Xhat_sq - 2.0 * inner_prod
    residual_sq = cp.maximum(residual_sq, 0.0)
    residual_norm = cp.sqrt(residual_sq)

    return residual_norm / cp.maximum(norm_X, epsilon)



def fr_factor_update_largedim_tier1(
    vec_tensor,
    core,
    factors,
    mode,
    shape,
    epsilon=1e-12,
    thread_budget=None,
    batch_cols=None,
    verbose=False,
):
    if verbose:
        print(f"  Updating factor {mode}...")

    if batch_cols is None:
        batch_cols = _estimate_batch_cols_for_Z(core, factors, mode)

    flat, vals = _blocked_coo_to_flat_indices(vec_tensor, shape)
    idxs = _unravel_flat_indices_C(flat, shape)
    rows = idxs[mode]
    nnz = int(flat.size)

    other_modes = [m for m in range(len(shape)) if m != mode]
    other_coords = [idxs[m] for m in other_modes]
    other_shape = tuple(shape[m] for m in other_modes)
    cols = safe_ravel(tuple(other_coords), other_shape, cp)

    A = factors[mode]
    I_mode, R = int(A.shape[0]), int(A.shape[1])

    # Denominator: exact Gram — unchanged
    Gram = _tucker_gram_ZtZ(core, factors, mode, epsilon=epsilon)
    denominator = cp.clip(A @ Gram, a_min=epsilon, a_max=None)

    # Deduplicate columns for efficient Z computation
    ucols, inv = cp.unique(cols, return_inverse=True)
    n_ucols = int(ucols.size)

    # Pre-allocate Z for all unique columns in one go
    Z_unique = cp.zeros((n_ucols, R), dtype=core.dtype)

    col_batches = range(0, n_ucols, int(batch_cols))
    if verbose:
        col_batches = tqdm(col_batches, desc=f"  Factor {mode} Z-batches", unit="batch", leave=False)

    for start in col_batches:
        end = min(start + int(batch_cols), n_ucols)
        u = ucols[start:end]
        _, idxs_by_mode = _unravel_cols_for_mode(u, shape, mode)
        Z_unique[start:end] = compute_Zcols_batch(
            core=core, factors=factors, mode=mode,
            other_modes=other_modes, idxs_by_mode=idxs_by_mode, epsilon=epsilon,
        )

    # GPU gather: expand unique Z to all NNZ — fully parallel, no scatter
    Z_nz = Z_unique[inv]  # (nnz, R)

    # Build sparse S: shape (I_mode, nnz), S[row_k, k] = val_k
    # Then numerator = S @ Z_nz via cuSPARSE SpMM — replaces cp.add.at
    k_idx = cp.arange(nnz, dtype=cp.int32)
    S = cpx_sparse.csr_matrix(
        (vals, (rows.astype(cp.int32), k_idx)),
        shape=(I_mode, nnz),
    )
    numerator = cp.clip(S @ Z_nz, a_min=epsilon, a_max=None)

    A_new = cp.clip(A * (numerator / (denominator + 1e-12)), a_min=epsilon, a_max=None)
    return A_new


def kl_factor_update_largedim_tier1(
    vec_tensor,
    core,
    factors,
    mode,
    shape,
    thread_budget=None,
    epsilon=1e-12,
    batch_cols=None,
    verbose=False,
):
    if verbose:
        print(f"  Updating factor {mode}...")

    if batch_cols is None:
        batch_cols = _estimate_batch_cols_for_Z(core, factors, mode)

    flat, vals = _blocked_coo_to_flat_indices(vec_tensor, shape)
    idxs = _unravel_flat_indices_C(flat, shape)
    rows = idxs[mode]
    nnz = int(flat.size)

    other_modes = [m for m in range(len(shape)) if m != mode]
    other_coords = [idxs[m] for m in other_modes]
    other_shape = tuple(shape[m] for m in other_modes)
    cols = safe_ravel(tuple(other_coords), other_shape, cp)

    A = factors[mode]
    I_mode, R = int(A.shape[0]), int(A.shape[1])

    # Exact denominator (no approximation)
    den_row = _tucker_den_row_full(core, factors, mode, epsilon=epsilon)
    denominator = den_row[None, :]  # (1, R)

    ucols, inv = cp.unique(cols, return_inverse=True)
    n_ucols = int(ucols.size)

    Z_unique = cp.zeros((n_ucols, R), dtype=core.dtype)

    col_batches = range(0, n_ucols, int(batch_cols))
    if verbose:
        col_batches = tqdm(col_batches, desc=f"  Factor {mode} Z-batches", unit="batch", leave=False)

    for start in col_batches:
        end = min(start + int(batch_cols), n_ucols)
        u = ucols[start:end]
        _, idxs_by_mode = _unravel_cols_for_mode(u, shape, mode)
        Z_unique[start:end] = compute_Zcols_batch(
            core=core, factors=factors, mode=mode,
            other_modes=other_modes, idxs_by_mode=idxs_by_mode, epsilon=epsilon,
        )

    # GPU gather — parallel, zero contention
    Z_nz = Z_unique[inv]  # (nnz, R)

    # KL weights: w_k = x_k / (A[row_k] · Z_nz[k])
    A_rows = A[rows]                                         # (nnz, R)
    R_nz = cp.clip(cp.sum(A_rows * Z_nz, axis=1), a_min=epsilon, a_max=None)  # (nnz,)
    w = vals / R_nz                                          # (nnz,)

    # SpMM: numerator = S_W @ Z_nz  where S_W[row_k, k] = w_k
    k_idx = cp.arange(nnz, dtype=cp.int32)
    S_W = cpx_sparse.csr_matrix(
        (w, (rows.astype(cp.int32), k_idx)),
        shape=(I_mode, nnz),
    )
    numerator = cp.clip(S_W @ Z_nz, a_min=epsilon, a_max=None)

    A_new = cp.clip(A * (numerator / (denominator + 1e-12)), a_min=epsilon, a_max=None)
    return A_new