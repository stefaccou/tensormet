from tensorly_custom.tucker_tensor import validate_tucker_rank, tucker_normalize, TuckerTensor
from tensorly_custom.decomposition._tucker import tucker_to_tensor
from tensorly_custom.base import unfold
from tensorly_custom.tenalg import mode_dot
from datetime import datetime
from threadpoolctl import threadpool_limits  # NEW
import tensorly_custom as tl
import pytensorlab as ptl
import numpy as np
import torch
import os
from typing import List, Tuple, Optional, Union
from contextlib import contextmanager
import cupy as cp
import cupyx.scipy.sparse as cpx_sparse
import time
import json
import random
import pickle
import csv
import multiprocessing
from utils import DATA_DIR, select_gpu, tree_to_device, notify_discord, compute_num_threads, ThreadBudget

from tucker_tensor import SparseTupleTensor, TuckerDecomposition

# ----- cupy sparse methods -----

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
    size = int(np.prod(orig_shape))
    int32_max = np.iinfo(np.int32).max
    block_size = min(size, int32_max)

    # ---- move to host and use int64 for safe arithmetic ----
    row_np = cp.asnumpy(row_cp).astype(np.int64)
    col_np = cp.asnumpy(col_cp).astype(np.int64)

    flat_np = row_np + col_np * np.int64(block_size)

    coords = np.unravel_index(flat_np, orig_shape)

    row_unf_np = coords[mode]

    other_coords = coords[:mode] + coords[mode + 1:]
    other_shape = tuple(s for i, s in enumerate(orig_shape) if i != mode)
    col_unf_np = np.ravel_multi_index(other_coords, other_shape)

    row_unf_cp = cp.asarray(row_unf_np)
    col_unf_cp = cp.asarray(col_unf_np)

    unfolded_shape = (orig_shape[mode], int(np.prod(other_shape)))
    unfolded = cpx_sparse.coo_matrix(
        (data_cp, (row_unf_cp, col_unf_cp)),
        shape=unfolded_shape,
    )

    if to_dense:
        return unfolded.toarray()
    return unfolded

def gather_dense_at_block_nz(dense_nd: np.ndarray,
                             vec_tensor: cpx_sparse.spmatrix,
                             orig_shape) -> cp.ndarray:
    orig_shape = tuple(orig_shape)
    size = int(np.prod(orig_shape))
    int32_max = np.iinfo(np.int32).max
    block_size = min(size, int32_max)

    dense_flat = dense_nd.reshape(size, order="C")
    coo = vec_tensor.tocoo()
    flat = coo.row + coo.col * block_size
    return dense_flat[flat.get()]

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

    coords_other = cp.unravel_index(col, other_shape)

    coords_full = []
    idx_other = 0
    for i in range(N):
        if i == mode:
            coords_full.append(row)
        else:
            coords_full.append(coords_other[idx_other])
            idx_other += 1

    coords_full = tuple(coords_full)

    size = int(np.prod(new_shape))
    int32_max = np.iinfo(np.int32).max
    block_size = min(size, int32_max)

    flat = cp.ravel_multi_index(coords_full, new_shape)

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

def ptl_tucker_to_tensor(tucker: ptl.TuckerTensor,
                         skip_factor: Optional[int] = None) -> np.ndarray:
    """Reconstruct full tensor from Tucker representation, optionally skipping one factor."""
    factors = tucker.factors
    if skip_factor is not None:
        factors = [f for i, f in enumerate(factors) if i != skip_factor]
    return ptl.tmprod(tucker.core, factors, list(range(tucker.ndim)) if skip_factor is None else
                     [i for i in range(tucker.ndim) if i != skip_factor])


# -- Kullback-Leibler specific methods ---
def build_sparse_ratio_vec(
    vec_tensor: cpx_sparse.spmatrix,
    shape,
    core,
    factors,
    epsilon: float = 1e-12,
) -> cpx_sparse.coo_matrix:
    """Build a sparse vectorized tensor Y with the same structure as X,
    but data = X_i / R_i, where R is the current reconstruction.

    vec_tensor : vectorized sparse X (block-encoded)
    shape      : original N-D shape
    core       : current core G
    factors    : list of A^{(n)}
    """
    shape = tuple(shape)
    size = int(np.prod(shape))
    int32_max = np.iinfo(np.int32).max
    block_size = min(size, int32_max)

    # 1) Dense reconstruction R = G ×_1 A^{(1)} × ... ×_N A^{(N)}
    #    (follows the paper literally; for huge tensors this may be too big
    #     and you’d want a custom "R at nonzeros only" kernel instead.)
    R = tucker_to_tensor((core, factors))        # dense cp.ndarray of shape `shape`
    R = tl.clip(R, a_min=epsilon, a_max=None)
    R_flat = R.ravel()                           # length = size

    # 2) Same block encoding as elsewhere
    coo = vec_tensor.tocoo()
    row_cp = coo.row
    col_cp = coo.col
    data_cp = coo.data

    # move indices to host to do int64 arithmetic safely
    row_np = cp.asnumpy(row_cp).astype(np.int64)
    col_np = cp.asnumpy(col_cp).astype(np.int64)

    flat_np = row_np + col_np * np.int64(block_size)  # flat indices into R_flat
    flat_cp = cp.asarray(flat_np)

    # 3) Compute X_i / R_i at the same positions
    R_vals = R_flat[flat_cp]                          # cp.ndarray
    R_vals = tl.clip(R_vals, a_min=epsilon, a_max=None)

    ratio_data = data_cp / R_vals                     # X_i / R_i

    # 4) Build sparse ratio vec with identical row, col, shape
    vec_ratio = cpx_sparse.coo_matrix(
        (ratio_data, (row_cp, col_cp)),
        shape=vec_tensor.shape,
    )
    vec_ratio.sum_duplicates()  # just in case

    return vec_ratio
def kl_core_update(
    tensor,        # vectorized sparse X
    shape,         # original N-D shape
    core,          # current core G
    factors,       # list of factor matrices A^{(n)}
    epsilon=1e-12,
):
    """KL core update following Mørup et al. (2008):

       G <- G ⊙ (D / F),

       D = (X / R) ×_1 A^{(1)T} × ... ×_N A^{(N)T}
       F = E ×_1 A^{(1)T} × ... ×_N A^{(N)T},
           with E all-ones, so F is an outer product of column sums.
    """
    shape = tuple(shape)
    modes = list(range(len(shape)))

    # --- 1) Build sparse ratio Y = X / R with same structure as X ---
    vec_ratio = build_sparse_ratio_vec(
        vec_tensor=tensor,
        shape=shape,
        core=core,
        factors=factors,
        epsilon=epsilon,
    )

    # --- 2) D = (X/R) ×_1 A^{(1)T} × ... ×_N A^{(N)T} ---
    D = sparse_multi_mode_dot_vec(
        vec_tensor=vec_ratio,
        orig_shape=shape,
        factors=factors,
        modes=modes,
        transpose_factors=True,  # uses A^{(n)T}, consistent with Tucker
    )
    D = tl.clip(D, a_min=epsilon, a_max=None)

    # --- 3) F = E ×_1 A^{(1)T} × ... ×_N A^{(N)T} ---
    # Since E ≡ 1, F is the N-way outer product of column sums
    col_sums = [tl.sum(A_n, axis=0) for A_n in factors]  # s^{(n)} ∈ R^{R_n}

    # Start from first mode, shape (R_1, 1, 1, ..., 1)
    F = col_sums[0].reshape((core.shape[0],) + (1,) * (core.ndim - 1))

    for n in range(1, core.ndim):
        # reshape s^{(n)} to broadcast along λ_n
        # e.g. for n=1 in 3D core: (1, R_2, 1)
        shape_n = [1] * n + [core.shape[n]] + [1] * (core.ndim - n - 1)
        F = F * col_sums[n].reshape(tuple(shape_n))

    F = tl.clip(F, a_min=epsilon, a_max=None)

    # --- 4) Multiplicative update G <- G ⊙ D/F (α = 1) ---
    core *= D / F

    return core

def compute_kl_divergence_sparse(
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

    ptl_tucker = ptl.TuckerTensor(core=core_np,
                                  factors=factors_np)
    with thread_budget.limit():
        ptl_R = ptl_tucker_to_tensor(ptl_tucker)


    shape = tuple(shape)

    # --- 1) Dense reconstruction R = G ×_1 A^{(1)} × ... ×_N A^{(N)} ---
    # This is exactly step 3 in Table 2 of the paper.
    # This breaks with dims over 1000
    # R = tucker_to_tensor((core, factors))      # cp.ndarray, shape=shape
    # R = tl.clip(R, a_min=epsilon, a_max=None)
    # R_flat = R.ravel()                         # length = size

    r_nz = gather_dense_at_block_nz(ptl_R, vec_tensor, shape)
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
    sum_R = cp.sum(ptl_R)
    sum_R_nz = cp.sum(r_nz)
    kl_zero = sum_R - sum_R_nz
    kl_total = kl_pos + kl_zero

    # --- 6) Optional normalized "relative" KL error ---
    sum_X = cp.sum(x_nz)  # sum over nonzero X
    rel_kl = kl_total / cp.maximum(sum_X, epsilon)
    # print(f"KL divergence: {kl_total}, relative KL: {rel_kl}")
    return kl_total, rel_kl

# --- similarity calculation functions ---
def get_eval_num_threads(fraction: float = 0.75, min_threads: int = 1) -> int:
    """Return n_threads ≈ fraction * available CPUs (at least min_threads)."""
    try:
        n_cores = multiprocessing.cpu_count()
    except NotImplementedError:
        n_cores = os.cpu_count() or 1

    n_threads = max(min_threads, int(n_cores * fraction))
    return n_threads

def load_og_sentences(vector_path):
    sentences = set()
    with open(vector_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        next(reader)  # skip header
        for row in reader:
            sent_id, vector, sentence = row[0], row[1], row[2]
            # the vectors have form "('verb', 'subject', 'object', 'rest', 'rest')"
            # we interpret the vector as a tuple of (verb, subject, object, *rest)
            vector = eval(vector)
            v, s, o = vector[0], vector[1], vector[2]
            sentences.add((v, s, o))
    return list(sentences)

def evaluate_sample(tensor,
                    sentences,
                    n_samples=100,
                    seed=42,
                    thread_budget: ThreadBudget | None = None):
    total_score = 0
    random.seed(seed)
    sampled_sentences = random.sample(sentences, n_samples)

    limiter = (thread_budget.limit() if thread_budget is not None else None)
    if limiter is None:
        ctx = contextmanager(lambda: (yield))()  # no-op
    else:
        ctx = limiter

    with ctx:
        for tup in sampled_sentences:
            if not tensor.check_vocab(tup):
                continue
            score = 0
            for i, element in enumerate(tup):
                role = ["verb", "subject", "object"][i]
                r2i = {"verb": "v2i", "subject": "s2i", "object": "o2i"}[role]
                G_excluded = tensor.excluded_role_vector(tup, role=role)

                # update to avoid division by 0
                F = tensor.factors[i].cpu().numpy()  # (N,R)

                # --- defensive norm computation ---
                F_norm = np.linalg.norm(F, axis=1)
                G_norm = np.linalg.norm(G_excluded)

                eps = 1e-12  # safeguard lower bound
                F_norm = np.maximum(F_norm, eps)
                G_norm = max(G_norm, eps)

                # --- safe cosine similarities ---
                similarities = (F @ G_excluded) / (F_norm * G_norm)

                idx = tensor.vocab[r2i][element]
                sim_i = similarities[idx]

                # --- rank = number of items >= the true similarity ---
                rank = np.sum(similarities >= sim_i)

                # safeguard rank for extreme cases
                if rank == 0:
                    rank = 1

                score += 1.0 / rank
            total_score += score/len(tup)

    average_score = total_score / n_samples
    print(f"Average expected role vector rank score over {n_samples} samples: {average_score}")
    return average_score

def print_elapsed_time(start_time, message=""):
    """Prints the elapsed time since start_time."""
    now = time.time()
    # if the message starts with indents (tabs), add the same number to the elapsed time print
    tabs = ""
    for char in message:
        if char == "\t":
            tabs += "\t"
        else:
            break
    elapsed_time = now - start_time
    minutes, seconds = divmod(elapsed_time, 60)
    if message:
        print(message)
    print(f"{tabs}Elapsed time: {int(minutes)} minutes and {seconds} seconds.")
    return now

# --- The actual magic ---
def non_negative_tucker_with_similarity(
    sparse_tensor,
    rank,
    thread_budget: ThreadBudget,
    divergence: str="frobenius",
    n_iter_max=10,
    init="random",
    tol=10e-5,
    epsilon = 1e-12,
    random_state=42,
    verbose=False,
    warmup_steps=1,
    return_errors="full",
    normalize_factors=False,
    patience: int=3,
    vocab=None,
    sample_sentences: List[Tuple[str, str, str]]=None,
    sem_check_every: int=10,
    sem_fitness_target: int=10000,


):
    if not isinstance(sparse_tensor, SparseTupleTensor):
        raise TypeError("sparse_tensor must be a SparseTupleTensor instance.")
    if not sparse_tensor.sparsity_type == "cupy":
        raise ValueError("sparse_tensor must have sparsity_type 'cupy'.")
    shape = tuple(sparse_tensor.shape)
    tensor = sparse_tensor.tensor
    rank = validate_tucker_rank(shape, rank=rank)

    modes = list(range(len(rank)))
    non_negative = True

    if init == "random":
        rng = tl.check_random_state(random_state)
        core = tl.tensor(
            rng.random_sample([rank[index] for index in range(len(modes))]) + 0.01,
            **tl.context(tensor),
        )
        factors = [
            tl.tensor(
                rng.random_sample((shape[mode], rank[index])), # we changed this to original shape
                **tl.context(tensor),
            )
            for index, mode in enumerate(modes)
        ]
    else:
        (core, factors) = init

    if non_negative is True:
        factors = [tl.abs(f) for f in factors]
        core = tl.abs(core)
    else:
        raise NotImplementedError("Currently only non-negative=True is supported.")


    rec_errors = []
    fitness_scores = []
    no_improve_steps = 0
    sem_no_improve_steps = 0
    best_sem_score = 0
    conv_iteration = None

    for iteration in range(n_iter_max):
        iter_start = time.time()
        for mode in modes:
            X = unfold_from_vectorized_sparse(tensor, shape, mode)
            Z = tucker_to_tensor((core, factors), skip_factor=mode)
            Z = unfold(Z, mode)  # (50, 1000000)
            A = factors[mode]  # (1000, 50)

            rows = X.row  # (nnz,)
            cols = X.col  # (nnz,)
            vals = X.data  # (nnz,)

            R_nz = cp.zeros_like(vals)  # (nnz,)

            for r in range(rank[0]):  # R = rank[mode], e.g. 50
                A_r = A[:, r]  # (I,)
                Z_r = Z[r, :]  # (K,)

                # Gather A[i,r] for all nonzeros and Z[r,j] for all nonzeros:
                A_r_vals = A_r[rows]  # (nnz,)
                Z_r_vals = Z_r[cols]  # (nnz,)

                R_nz += A_r_vals * Z_r_vals  # elementwise, stays (nnz,)
            R_nz = tl.clip(R_nz, a_min=epsilon, a_max=None)

            W_data = (vals / R_nz)  # (nnz,)
            W = cpx_sparse.coo_matrix(
                (W_data, (rows, cols)),
                shape=X.shape,  # (I_n, K)
            )
            # print(f"W shape: {W.shape}, Z shape {Z.shape}")
            numerator = W @ tl.transpose(Z)  # (I_n, R)
            # Denominator: E Z^T  ==  row-wise broadcast of sum over columns of Z
            den_row = tl.sum(Z, axis=1)  # (R,)
            denominator = den_row[np.newaxis, :]  # (1, R) – will broadcast to (I, R)
            denominator = tl.clip(denominator, a_min=epsilon, a_max=None)
            # Multiplicative update
            A = A * (numerator / (denominator + 1e-12))
            A = tl.clip(A, a_min=epsilon, a_max=None)

            factors[mode] = A
        # factors_time = print_elapsed_time(iter_start, "\tfactors calculated")
        core_np = tl.to_numpy(core)
        factors_np = [tl.to_numpy(f) for f in factors]
        ptl_tucker = ptl.TuckerTensor(core=core_np,
                                      factors=factors_np)

        with thread_budget.limit():
            ptl_R = ptl_tucker_to_tensor(ptl_tucker)

        data = gather_dense_at_block_nz(ptl_R, tensor, shape)
        data = tl.clip(data, a_min=epsilon, a_max=None)
        X_R_data = tensor.data / cp.asarray(data)
        ptl_X_R = cpx_sparse.coo_matrix(
            (X_R_data, (tensor.row, tensor.col)),
            shape=tensor.shape,
        )
        X_R = sparse_multi_mode_dot_vec(
            vec_tensor=ptl_X_R,
            orig_shape=shape,
            factors=factors,
            modes=modes,
            transpose_factors=True,  # X ×_n W_n^T
        )
        X_R = tl.clip(X_R, a_min=epsilon, a_max=None)

        col_sums = [tl.sum(A_n, axis=0) for A_n in factors]
        F = col_sums[0].reshape((core.shape[0],) + (1,) * (core.ndim - 1))
        for n in range(1, core.ndim):
            shape_n = [1] * n + [core.shape[n]] + [1] * (core.ndim - n - 1)
            F = F * col_sums[n].reshape(tuple(shape_n))
        F = tl.clip(F, a_min=epsilon, a_max=None)
        core *= X_R / (F + epsilon)

        core, factors = tucker_normalize((core, factors))
        # core_time = print_elapsed_time(factors_time, "\tcore updated")

        kl_val, rel_kl = compute_kl_divergence_sparse(
            vec_tensor=tensor,
            shape=shape,
            core=core,
            factors=factors,
            thread_budget=thread_budget,
            epsilon=epsilon,
        )
        rec_errors.append(rel_kl)
        # errors_time = print_elapsed_time(core_time, "\terror computed")
        if iteration > 1 and verbose:
            print(
                f"{iteration}: reconstruction error={rec_errors[-1]}, variation={rec_errors[-2] - rec_errors[-1]}."
            )


        # we add a patience criterion
        if iteration > warmup_steps:
            improvement = abs(rec_errors[-2] - rec_errors[-1])
            try:
                imp_val = float(improvement)
            except Exception:
                print("look into why this happens")
                print(type(improvement))
                imp_val = float(improvement.item())

            if imp_val < tol:
                no_improve_steps += 1
                if verbose:
                    print(f"No improvement: step {no_improve_steps}/{patience} (improvement={imp_val:.3e}).")
                if no_improve_steps >= patience:
                    if verbose:
                        notify_discord(
                            f"Stopped after {no_improve_steps} consecutive non-improving steps (patience={patience}). Converged at iteration {iteration}.",
                        job_finished=False)
                    break
            else:
                if no_improve_steps > 0 and verbose:
                    print(f"Improved (improvement={imp_val:.3e}); resetting patience counter.")
                no_improve_steps = 0

        if normalize_factors:
            core, factors = tucker_normalize((core, factors))

        # --- similarity calculations ---
        # we load the tensor as a TuckerDecomposition on CPU for evaluation
        if (sample_sentences is not None
            and vocab is not None
            and (iteration + 1) % sem_check_every == 0):
            # compute semantic fitness
            tl.set_backend("pytorch")
            core_cpu = tl.tensor(cp.asnumpy(core))
            factors_cpu = [tl.tensor(cp.asnumpy(f)) for f in factors]
            tucker_decomp = TuckerDecomposition(core=core_cpu, factors=factors_cpu, vocab=vocab)

            fitness_score = evaluate_sample(
                tucker_decomp,
                sample_sentences,
                n_samples=sem_fitness_target,
                seed=random_state,
                thread_budget=thread_budget,
            )
            # print(fitness_score)
            fitness_scores.append(fitness_score)
            tl.set_backend("cupy")

            # semantic patience
            if len(fitness_scores) >= 2:
                improvement = fitness_scores[-1] - best_sem_score
                if improvement > 0:
                    best_sem_score = fitness_scores[-1]
                    best_core = core.copy()
                    best_factors = factors.copy()
                    conv_iteration = iteration
                    if verbose:
                        print("New best semantic score, saving current best core and factors.")

                if improvement < tol:
                    sem_no_improve_steps += 1
                    if verbose:
                        print(f"\tNo semantic improvement: step {sem_no_improve_steps}/{patience} "
                              f"(difference={improvement:.3e}).")
                    if sem_no_improve_steps >= patience:
                        if verbose:
                            notify_discord(
                                "Stopped after "
                                f"{sem_no_improve_steps} consecutive non-improving semantic steps "
                                f"(patience={patience}). Converged at iteration {iteration}.",
                                job_finished=False,
                            )
                        break
                else:
                    if sem_no_improve_steps > 0 and verbose:
                        print(
                            f"\tSemantic improvement ({improvement:.3e}); resetting patience counter."
                        )
                    sem_no_improve_steps = 0
        # iter_end = print_elapsed_time(iter_start, f"End of iteration {iteration}.")
    if conv_iteration:
        tensor = TuckerTensor((best_core, best_factors))
        iteration = conv_iteration
    else:
        tensor = TuckerTensor((core, factors))
    if return_errors == "simple":
        return tensor, rec_errors
    elif return_errors == "full":
        return {"tensor": tensor, "errors": rec_errors, "fitness_scores": fitness_scores,
                "iterations": iteration + 1, "final_error": rec_errors[-1]}
    else:
        return tensor


print("Preparing environment for cupy sparse non-negative tucker decomposition with similarity checks.")
device = select_gpu()
tl.set_backend("cupy")

# we load the sample sentences only once
vector_path = os.path.join(DATA_DIR, "vectors", "fineweb_dutch_10000000.csv")
sentence_sample = load_og_sentences(vector_path)

# we perform the non negative tucker
print("\nStarting non-negative Tucker decomposition on cupy sparse tensor")
dataset = "fineweb_large"
dims = [3000]
methods = ["sc"]
ranks = [[100, 100, 100]]

iters = 1000
tol = 1e-5
random_state = 1
name = "klNoB"
patience = 5
sem_check_every = 20
max_cpu_frac = 0.5
thread_budget = ThreadBudget(n_threads=compute_num_threads(max_cpu_frac))

for dim in dims:
    for rank in ranks:
        vocab_path = os.path.join(DATA_DIR, "tensors", dataset, f"vocabularies/{dim}.pkl")
        with open(vocab_path, "rb") as f:
            vocab = pickle.load(f)

        for method in methods:
            start_time = time.time()
            print(f"\nMethod: {method}, Dimensionality: {dim}")
            # we save this decomposition to disk
            output_dir = os.path.join(DATA_DIR, "tensors", dataset, "decomposition")
            os.makedirs(output_dir, exist_ok=True)
            output_path = os.path.join(output_dir, f"{name+'_' if name else ''}{method}_{dim}d_{rank[0]}r_{iters}i.pt")
            # we check if it already exists
            if os.path.exists(output_path):
                print(f"Decomposition already exists at {output_path}, skipping...")
                continue
            tl.set_backend("cupy")
            sparse_tensor = SparseTupleTensor.load_from_disk(dataset=dataset,
                                                             method=method,
                                                             dims=dim,
                                                             map_location="cpu")
            sparse_tensor.tensor_to_sparse("cupy")
            # we load in the vocab and sample sentences


            tucker_decomp_info = non_negative_tucker_with_similarity(
                sparse_tensor,
                rank=rank,
                divergence="kl",
                thread_budget=thread_budget,
                n_iter_max=iters,
                init="random",
                tol=tol,
                random_state=random_state,
                verbose=True,
                return_errors="full",
                normalize_factors=False,
                vocab=vocab,
                sample_sentences=sentence_sample,
                sem_check_every=sem_check_every,
                sem_fitness_target = 10000,
                warmup_steps = 1, # if method == "counting" else 100,
                patience = patience,
            )

            # the final output should be a Torch tensor on CPU
            tl.set_backend("pytorch")
            core, factors = tucker_decomp_info["tensor"]
            errors = tucker_decomp_info["errors"]
            core = tl.tensor(cp.asnumpy(core))
            factors = [tl.tensor(cp.asnumpy(f)) for f in factors]
            tucker_decomp_torch = TuckerTensor((core, factors))
            torch.save(tucker_decomp_torch, output_path)
            # we save errors as well
            errors_path = output_path.replace(".pt", "_errors.npy")
            np.save(errors_path, np.array([cp.asnumpy(e) for e in errors]))
            # we also save the fitness scores
            fitness_scores = tucker_decomp_info["fitness_scores"]
            fitness_path = output_path.replace(".pt", "_fitness.npy")
            np.save(fitness_path, np.array([cp.asnumpy(f) for f in fitness_scores]))
            end_time = time.time()

            # Machine-readable logging
            log_path = os.path.join(output_dir, "runs.jsonl")  # NDJSON format

            log_entry = {
                "timestamp": datetime.now().isoformat() + "Z",
                "dataset": "fineweb_dutch",
                "name": name,
                "input_vectors": 10000000,
                "method": method,
                "dimensionality": dim,
                "rank": rank,
                "max_iterations": iters,
                "iterations": tucker_decomp_info["iterations"],
                "final_error": float(tucker_decomp_info["final_error"]),
                "final_fitness": (float(fitness_scores[-1]) if fitness_scores else None),
                "tolerance": tol,
                "random_state": random_state,
                "output_tensor": os.path.basename(output_path),
                "runtime_seconds": round(end_time - start_time, 2),
            }

            with open(log_path, "a") as f:
                f.write(json.dumps(log_entry) + "\n")

            notify_discord(
                f"Saved Tucker decomposition {method} - {dim}/{rank[0]} to {output_path}"
                f" in {end_time - start_time:.2f} seconds.")
