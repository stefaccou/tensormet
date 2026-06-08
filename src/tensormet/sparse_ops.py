from __future__ import annotations
import tensorly as tl
import pytensorlab as ptl
import numpy as np
from typing import List, Tuple, Optional, Union
import math
from tensormet.utils import einsum_letters, cp_einsum_optimize, make_lazy_cupy_pair
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

    N = core.ndim
    letters = einsum_letters(N)
    core_sub = "".join(letters)                              # e.g. 'abc' for N=3
    mat_subs = ["i" + letters[k] for k in other_modes]      # batch × rank for each contracted mode
    out_sub  = "i" + letters[mode]                           # keep batch + target-mode rank
    eq = core_sub + "," + ",".join(mat_subs) + "->" + out_sub
    mats = [factors[k][idxs_by_mode[k]] for k in other_modes]
    tmp = cp.einsum(eq, core, *mats, optimize=cp_einsum_optimize(1 + len(other_modes)))
    return cp.clip(tmp, a_min=epsilon, a_max=None)

    # -- old tensordot+broadcast-sum loop (kept for reference) --
    # k0 = other_modes[0]
    # M0 = factors[k0][idxs_by_mode[k0]]  # (m, R_k0)
    # m = M0.shape[0]
    # tmp = cp.tensordot(M0, core, axes=(1, k0))
    # remaining_axes = [i for i in range(core.ndim) if i != k0]
    # for k in other_modes[1:]:
    #     M = factors[k][idxs_by_mode[k]]  # (m, R_k)
    #     axis_idx = 1 + remaining_axes.index(k)
    #     bcast_shape = [1] * tmp.ndim
    #     bcast_shape[0] = m
    #     bcast_shape[axis_idx] = M.shape[1]
    #     tmp = cp.sum(tmp * M.reshape(bcast_shape), axis=axis_idx)
    #     remaining_axes.remove(k)
    # if tmp.ndim != 2:
    #     raise RuntimeError(f"Expected 2D result after contractions, got shape {tmp.shape}")
    # return cp.clip(tmp, a_min=epsilon, a_max=None)


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


def _svd_worker(row_np, col_np, data_np, sp_shape, rank_i, random_state_i, threads_per_mode,
                write_conn):
    """Top-level worker for ProcessPoolExecutor: truncated SVD on one mode unfolding.

    Uses eigsh on the Gram operator A @ A.T (shape m × m, always small) instead
    of svds. svds recovers right singular vectors via A.T @ U, which for a wide
    unfolding (e.g. 10000 × 100M) produces a 15-billion-element matrix that
    overflows LAPACK's int32 indexing. We only need left singular vectors, so
    the Gram approach is both correct and avoids the overflow entirely.

    Each Gram mat-vec calls A.T @ x (producing an n-vector) then A @ y, and
    sends one increment via write_conn for live tqdm tracking in the main process.
    Pipe connections clean up with os.close() on exit — no IPC, no hang risk.
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
        eigvals, eigvecs = scipy.sparse.linalg.eigsh(gram_op, k=k, which='LM', v0=v0)

    desc = np.argsort(eigvals)[::-1]
    U = eigvecs[:, desc]                              # (m, k) left singular vectors
    s = np.sqrt(np.maximum(eigvals[desc], 0.0))       # (k,) singular values
    return _nndsvd_factors(U, s)


def _initialize_svd_tucker_cpu(sparse_tensor, shape, rank, modes, random_state, thread_budget=None):
    """Tucker init via truncated SVD of each mode unfolding (CPU/scipy path).

    Extracts COO data for all mode unfoldings in the main process (sequential,
    fast), then dispatches one worker process per mode via ProcessPoolExecutor.
    Each worker process owns an independent BLAS thread pool sized at
    n_threads // n_modes, so all CPUs are saturated without oversubscription.
    A tqdm bar per mode tracks ARPACK mat-vec iterations via a Queue drained by
    a background thread. Core is computed on GPU after all factors are collected.
    """
    import multiprocessing
    import threading
    from concurrent.futures import ProcessPoolExecutor, as_completed
    from tqdm import tqdm

    n_modes = len(modes)
    n_threads = thread_budget.n_threads if thread_budget is not None else None
    threads_per_mode = max(1, n_threads // n_modes) if n_threads is not None else None

    # GPU → CPU transfer happens here, sequentially (PCIe, fast relative to SVD)
    mode_arrays = []
    for mode in modes:
        unfolded = unfold_from_vectorized_sparse(sparse_tensor, shape, mode, to_dense=False)
        coo = unfolded.tocoo()
        mode_arrays.append((
            cp.asnumpy(coo.row).astype(np.int32),
            cp.asnumpy(coo.col).astype(np.int32),
            cp.asnumpy(coo.data),
            unfolded.shape,
        ))

    def _drain(read_conn, pbar):
        # Exits when the write end of the pipe is fully closed (worker exits +
        # main process closes its copy), which raises EOFError on recv().
        try:
            while True:
                read_conn.recv()
                pbar.update(1)
        except EOFError:
            pass

    # One Pipe per mode. write_conn is passed to the worker (picklable Connection).
    # Cleanup is a plain os.close() — no manager process, no RPC, no hang risk.
    pipe_pairs = [multiprocessing.Pipe(duplex=False) for _ in range(n_modes)]

    bars = [
        tqdm(desc=f"SVD mode {i}", position=i, leave=True, unit="mv", dynamic_ncols=True)
        for i in range(n_modes)
    ]
    drain_threads = [
        threading.Thread(target=_drain, args=(r, b), daemon=True)
        for (r, _w), b in zip(pipe_pairs, bars)
    ]
    for t in drain_threads:
        t.start()

    factors = [None] * n_modes
    with ProcessPoolExecutor(max_workers=n_modes) as pool:
        futures = {
            pool.submit(
                _svd_worker,
                row, col, data, sp_shape,
                rank[i], random_state + i, threads_per_mode, pipe_pairs[i][1],
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
        _blocked_coo_to_flat_indices,
        _unravel_flat_indices_C,
        _accumulate_core_num_outer,
        _estimate_batch_num_for_outer,
    )
    N = len(modes)
    core_shape = tuple(factors[n].shape[1] for n in range(N))
    core = cp.zeros(core_shape, dtype=factors[0].dtype)

    flat, xvals = _blocked_coo_to_flat_indices(sparse_tensor, shape)
    nnz = int(flat.size)
    if nnz == 0:
        return core
    idxs = _unravel_flat_indices_C(flat, shape)

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


def _initialize_svd_tucker_gpu(sparse_tensor, shape, rank, modes, random_state):
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

    core = _compute_tucker_core_batched(sparse_tensor, shape, factors, modes)
    core = cp.clip(cp.abs(core), a_min=1e-30, a_max=None)
    factors = [cp.clip(cp.abs(f), a_min=1e-30, a_max=None) for f in factors]
    return core, factors


def initialize_nonnegative_tucker(sparse_tensor, shape, rank, modes, init, random_state,
                                   thread_budget=None):
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
    elif init in ("svd", "svd_cpu"):
        return _initialize_svd_tucker_cpu(sparse_tensor, shape, rank, modes, random_state,
                                          thread_budget=thread_budget)
    elif init == "svd_gpu":
        return _initialize_svd_tucker_gpu(sparse_tensor, shape, rank, modes, random_state)
    else:
        core, factors = init

    factors = [tl.clip(tl.abs(f), a_min=1e-30, a_max=None) for f in factors]
    core = tl.clip(tl.abs(core), a_min=1e-30, a_max=None)
    return core, factors