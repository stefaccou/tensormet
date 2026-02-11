from tensorly_custom.tucker_tensor import validate_tucker_rank, tucker_normalize, TuckerTensor
from tensorly_custom.decomposition._tucker import tucker_to_tensor
from tensorly_custom.base import unfold
from tensorly_custom.tenalg import mode_dot
from datetime import datetime
import tensorly_custom as tl
import numpy as np
import torch
import os
from typing import List, Tuple
from utils import DATA_DIR, select_gpu, tree_to_device, notify_discord
from typing import Optional, Union
import cupy as cp
import cupyx.scipy.sparse as cpx_sparse
import time
import json

from tucker_tensor import SparseTupleTensor

device = select_gpu()
tl.set_backend("cupy")

# ----- cupy sparse methods -----

def fro_norm_coo(x):
    # x: cupyx.scipy.sparse.coo_matrix
    data = x.data
    return cp.sqrt((cp.abs(data) ** 2).sum())

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
        # print("Applying mode", mode)
        current_vec, current_shape = sparse_mode_dot_vec(
            current_vec,
            current_shape,
            factors[mode],
            mode=mode,
            transpose_factor=transpose_factors,
        )

    # At this point, current_vec is still sparse (prod(core_shape), 1)
    core_shape = current_shape  # typically your (50, 50, 50) or similar
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
def non_negative_tucker(
    sparse_tensor,
    rank,
    n_iter_max=10,
    init="random",
    tol=10e-5,
    random_state=42,
    verbose=False,
    return_errors="full",
    normalize_factors=False,
    patience: int=3,
):
    if not isinstance(sparse_tensor, SparseTupleTensor):
        raise TypeError("sparse_tensor must be a SparseTupleTensor instance.")
    if not sparse_tensor.sparsity_type == "cupy":
        raise ValueError("sparse_tensor must have sparsity_type 'cupy'.")
    shape = tuple(sparse_tensor.shape)
    tensor = sparse_tensor.tensor
    rank = validate_tucker_rank(shape, rank=rank)
    epsilon = 1e-12
    modes = list(range(len(rank)))
    non_negative = True
    # skip_factor = None
    # transpose_factors = False


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


    tensor_coo = tensor.tocoo()
    norm_tensor = cp.sqrt((cp.abs(tensor_coo.data) ** 2).sum())
    rec_errors = []
    no_improve_steps = 0
    for iteration in range(n_iter_max):
        # iter_time = time.time()
        for mode in modes:
            # the first steps can be done with the dense implementation
            # This still explodes! The created B tensor does not always fit in memory
            B = tucker_to_tensor((core, factors), skip_factor=mode)
            B = tl.transpose(unfold(B, mode))
            unfolded = unfold_from_vectorized_sparse(tensor, shape,
                                                     mode)  # this is the same as B when using dense tensor!
            numerator = unfolded @ B  # cupy sparse @ dense
            numerator = tl.clip(numerator, a_min=epsilon, a_max=None)
            denominator = tl.dot(factors[mode], tl.dot(tl.transpose(B), B))
            denominator = tl.clip(denominator, a_min=epsilon, a_max=None)
            factors[mode] *= numerator / denominator

        numerator = sparse_multi_mode_dot_vec(
            vec_tensor=tensor,
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

        core *= numerator / denominator

        norm_X_sq = norm_tensor ** 2
        norm_Xhat_sq = tl.sum(core * denominator)
        inner_prod = tl.sum(numerator * core)
        residual_norm = tl.sqrt(norm_X_sq + norm_Xhat_sq - 2 * inner_prod)
        relative_error = residual_norm / norm_tensor
        # print(f"Iteration {iteration + 1}: relative error = {relative_error:.6f}")
        rec_errors.append(relative_error)

        if iteration > 1 and verbose:
            print(
                f"{iteration}: reconstruction error={rec_errors[-1]}, variation={rec_errors[-2] - rec_errors[-1]}."
            )

        # if iteration > 1 and tl.abs(rec_errors[-2] - rec_errors[-1]) < tol:
        #     if verbose:
        #         print(f"converged in {iteration} iterations.")
        #     break

        # we add a patience criterion
        if iteration > 1:
            improvement = tl.clip(rec_errors[-2] - rec_errors[-1], a_min=0, a_max=None)
            try:
                imp_val = float(improvement)
            except Exception:
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
        # iter_time_end = print_elapsed_time(iter_time)
        # print("-----------------------------------------")

    tensor = TuckerTensor((core, factors))
    if return_errors == "simple":
        return tensor, rec_errors
    elif return_errors == "full":
        return {"tensor": tensor, "errors": rec_errors, "iterations": iteration + 1, "final_error": rec_errors[-1]}
    else:
        return tensor




# we perform the non negative tucker
print("\nStarting non-negative Tucker decomposition on cupy sparse tensor:")
dims = [1000, 1500, 2000, 2500, 3000]
methods = ["sc", "sii"]
ranks = [[100, 100, 100]]

iters = 2500
tol = 1e-7
random_state = 42
for rank in ranks:
    for dim in dims:
        for method in methods:
            dataset = "fineweb_large"
            start_time = time.time()
            print(f"\nMethod: {method}, Dimensionality: {dim}")
            # we save this decomposition to disk
            output_dir = os.path.join(DATA_DIR, "tensors", dataset, "decomposition")
            os.makedirs(output_dir, exist_ok=True)
            output_path = os.path.join(output_dir, f"{method}_{dim}d_{rank[0]}r_{iters}i.pt")
            # we check if it already exists
            # if os.path.exists(output_path):
            #     print(f"Decomposition already exists at {output_path}, skipping...")
            #     continue
            tl.set_backend("cupy")
            sparse_tensor = SparseTupleTensor.load_from_disk(dataset=dataset,
                                                             method=method,
                                                             dims=dim,
                                                             map_location="cpu")
            print("\nConverting to cupy sparse representation:")
            sparse_tensor.tensor_to_sparse("cupy")
            tucker_decomp_info = non_negative_tucker(
                sparse_tensor,
                rank=rank,
                n_iter_max=iters,
                init="random",
                tol=tol,
                random_state=random_state,
                verbose=True,
                return_errors="full",
                normalize_factors=True,
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
            end_time = time.time()

            # Machine-readable logging
            log_path = os.path.join(output_dir, "runs.jsonl")  # NDJSON format

            log_entry = {
                "timestamp": datetime.now().isoformat() + "Z",
                "dataset": "fineweb_dutch",
                "input_vectors": 10000000,
                "method": method,
                "dimensionality": dim,
                "rank": rank,
                "max_iterations": iters,
                "iterations": tucker_decomp_info["iterations"],
                "final_error": float(tucker_decomp_info["final_error"]),
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

