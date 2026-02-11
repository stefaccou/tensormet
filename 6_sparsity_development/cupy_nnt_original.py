from tensorly_custom.tucker_tensor import validate_tucker_rank, tucker_normalize, TuckerTensor
from tensorly_custom.decomposition._tucker import tucker_to_tensor
from tensorly_custom.base import unfold
from tensorly_custom.tenalg import mode_dot

import tensorly_custom as tl
import numpy as np
import pickle
import torch
import os
from typing import List, Tuple
from utils import DATA_DIR, select_gpu, tree_to_device, notify_discord
from typing import Optional, Union
import cupy as cp
import cupyx.scipy.sparse as cpx_sparse
import sparse
import tensorflow as tf
import time


device = select_gpu()
tl.set_backend("cupy")

def _to_np(x):
    # Accept NumPy arrays or torch tensors; return NumPy view/copy
    if hasattr(x, "detach"):  # torch.Tensor
        return x.detach().cpu().numpy()
    return x


def _torch_or_tucker_load(path, map_location="cpu"):
    """Tries to load a torch-saved file, if fails, tries pickle."""
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except RuntimeError:
        with open(path, "rb") as f:
            return pickle.load(f)

def cupy_to_torch_sparse(
    cu_mat: cpx_sparse.spmatrix,
    orig_shape: Optional[Tuple[int, ...]] = None,
    device: Union[str, torch.device] = "cpu",
    dtype: Optional[torch.dtype] = None,
) -> torch.Tensor:
    """
    Convert a CuPy sparse matrix (any format) back to a torch sparse COO tensor.

    If orig_shape is None:
        - The torch tensor is 2D and has the same shape as cu_mat.
    If orig_shape is provided and len(orig_shape) == 2:
        - The torch tensor is 2D with that shape.
    If orig_shape is provided and len(orig_shape) > 2:
        - We treat cu_mat.row as the flattened N-D index and unflatten it
          back to N-D using np.unravel_index, assuming the representation
          created by `torch_sparse_to_cupy`.

    Args:
        cu_mat: CuPy sparse matrix (COO/CSR/CSC, will be converted to COO).
        orig_shape: original tensor shape (for N-D tensors).
        device: target torch device.
        dtype: target dtype for values (defaults to inferred from data).

    Returns:
        torch.sparse_coo_tensor on the requested device.
    """
    # Ensure COO format
    if not cpx_sparse.isspmatrix_coo(cu_mat):
        cu_mat = cu_mat.tocoo()

    row_cp = cu_mat.row
    col_cp = cu_mat.col
    data_cp = cu_mat.data

    # Bring back to NumPy on host
    row_np = cp.asnumpy(row_cp)
    col_np = cp.asnumpy(col_cp)
    data_np = cp.asnumpy(data_cp)

    if orig_shape is None:
        # Simple 2D case, use cu_mat.shape directly
        shape = cu_mat.shape
        indices_np = np.vstack([row_np, col_np])
    else:
        shape = tuple(orig_shape)
        if len(shape) == 2:
            # 2D round-trip
            indices_np = np.vstack([row_np, col_np])
        else:
            # N-D round-trip: row_np contains flattened indices
            flat = row_np
            coords = np.unravel_index(flat, shape)  # tuple of arrays
            indices_np = np.vstack(coords)          # shape: (ndim, nnz)

    indices_t = torch.from_numpy(indices_np).long()
    values_t = torch.from_numpy(data_np)

    if dtype is not None:
        values_t = values_t.to(dtype)

    # Build sparse tensor
    x = torch.sparse_coo_tensor(indices_t, values_t, size=shape)
    x = x.coalesce()
    x = x.to(device)

    return x


def torch_sparse_to_cupy(
    x: torch.Tensor,
) -> Tuple[cpx_sparse.coo_matrix, Tuple[int, ...]]:
    """
    Convert a torch sparse COO tensor to a CuPy COO sparse matrix.

    For 2D tensors, the mapping is straightforward.
    For N-D tensors (N>2), we flatten the N-D indices to a single row index
    using np.ravel_multi_index and store them in a (prod(shape), 1) matrix.

    The original shape is returned for reconstruction.
    Returns:
        (cupy_coo_matrix, original_shape)
    """
    if not x.is_sparse:
        raise TypeError("torch_sparse_to_cupy expects a torch sparse tensor (COO).")

    x = x.coalesce()  # ensure indices are unique & sorted
    indices = x.indices()          # shape: (ndim, nnz)
    values = x.values()            # shape: (nnz,)
    shape = tuple(x.shape)

    # Move to CPU and NumPy
    indices_np = indices.cpu().numpy()
    values_np = values.cpu().numpy()

    ndim, nnz = indices_np.shape

    if ndim == 2:
        # Direct 2D mapping
        row = indices_np[0]
        col = indices_np[1]
        row_cp = cp.asarray(row)
        col_cp = cp.asarray(col)
        data_cp = cp.asarray(values_np)
        cu_mat = cpx_sparse.coo_matrix((data_cp, (row_cp, col_cp)), shape=shape)
    else:
        # Flatten N-D indices to a single dimension (row index)
        coords = [indices_np[d] for d in range(ndim)]
        flat = np.ravel_multi_index(coords, shape)  # shape: (nnz,)
        flat_cp = cp.asarray(flat)
        data_cp = cp.asarray(values_np)
        # Store as a (prod(shape), 1) matrix: column index always 0
        zero_cp = cp.zeros_like(flat_cp)
        cu_mat = cpx_sparse.coo_matrix(
            (data_cp, (flat_cp, zero_cp)),
            shape=(int(np.prod(shape)), 1),
        )

    return cu_mat, shape


def _role_index(role: str) -> int:
    if role == "verb":
        return 0
    elif role == "subject":
        return 1
    elif role == "object":
        return 2
    else:
        raise ValueError("role must be one of {'verb','subject','object'}")


class SparseTupleTensor:
    """Encapsulating the Sparse TupleTensor (built from vectors extracted from corpus) and the vocabulary,
    providing methods for decomposition, refactoring, etc.."""
    def __init__(self, tensor, device="cpu", sparsity_type=None):
        self.tensor = tensor
        self.sparsity_type = sparsity_type
        self.shape = tensor.shape
        self.device = device

    # --- Construction and loading ---
    @classmethod
    def load_from_disk(cls,
                       dataset: str="karrewiet_sparse",
                       method: str="counting",
                       dims: int=1000,
                       map_location: str="cpu",
                       sparsity_type: Optional[str]=None
                          ) -> "SparseTupleTensor":

        """Loads a precomputed tucker decomposition from disk.
            Args:
                dataset (str): name of the dataset
                method (str): method used to compute the decomposition
                    - one of "counting", "sc", "sii"
                dims (int): dimensionality of the original tensor modes (vocab size)
                rank (int): rank of the decomposition
                iterations (int): number of iterations used to compute the decomposition
                map_location (str): device to map the loaded tensors to
            Returns:
                ((core, factors), vocab)
                    core: torch.Tensor
                    factors: list[torch.Tensor]
                    vocab: dict with keys 'vocab_v','vocab_s','vocab_o','v2i','s2i','o2i'
        """
        if method not in {"counting", "sc", "sii"}:
            raise ValueError("method must be one of {'counting','sc','sii'}")
        base = os.path.join(DATA_DIR, "tensors", dataset)

        vocab_path = os.path.join(base, f"vocabularies/{dims}.pkl")
        populated_path = os.path.join(base,"populated", f"{method}_{dims}.pt")
        if not os.path.exists(vocab_path):
            raise FileNotFoundError(f"Missing vocab file: {vocab_path}")
        if not os.path.exists(populated_path):
            raise FileNotFoundError(f"Missing decomposition file: {populated_path}")
        # the vocab is here under f"vocabularies_[dims].pkl"
        # Load with torch (they were saved with torch.save)
        with open(vocab_path, "rb") as f:
            vocab = pickle.load(f)
        tensor = _torch_or_tucker_load(populated_path, map_location=map_location)

        return cls(tensor, device=map_location, sparsity_type=sparsity_type)



    # -- Sparsity methods ---
    def sparse_representation(self, sparse_type):
        # we return the sparse representation of the tensor
        if sparse_type == self.sparsity_type:
            return self.tensor
        # we check if our tensor is a tensorflow tensor or make it one
        if sparse_type == "tensorflow":
            if self.sparsity_type != "torch":
                tensor = self.sparse_representation("torch")
            else:
                tensor = self.tensor
            # we build from torch sparse tensor
            indices = tensor.coalesce().indices().t().numpy()   # shape (nnz, ndim)
            values  = tensor.coalesce().values().numpy()        # shape (nnz,)
            shape   = tuple(self.shape)          # e.g. (d0, d1, ..., d_{n-1})
            sparse_tensor = tf.SparseTensor(indices=indices, values=values, dense_shape=shape)
            # we warn users that tensorflow sparse tensors map directly to GPU.
            # additionally, they directly "allocate" the whole GPU memory to tf to reduce fragmentation later on.
            # this makes nvtop commands etc. not useable anymore

            print("WARNING: TensorFlow sparse tensors are allocated on GPU and may reserve large amounts of GPU memory.")

            return sparse_tensor

        elif sparse_type == "torch":
            if not self.sparsity_type or self.sparsity_type == "dense":
               return self.tensor.to_sparse()
            # can work from any tensor-like object
            elif self.sparsity_type == "cupy":
                return cupy_to_torch_sparse(self.tensor, orig_shape=self.shape)
            elif self.sparsity_type == "tensorflow":
                coords = self.tensor.indices.numpy()       # shape (nnz, ndim)
                data   = self.tensor.values.numpy()        # shape (nnz,)
                shape  = tuple(self.shape)  # e.g. (d0, d1, ..., d_{n-1})
                sparse_tensor = torch.sparse_coo_tensor(torch.tensor(coords).t(), torch.tensor(data), size=shape, device="cpu")
                return sparse_tensor
            else:
                raise NotImplementedError("sparsity_type must be one of {'dense', None, 'cupy', 'tensorflow','torch'}")

        elif sparse_type == "sparse":
            # can only work from a sparse torch tensor
            if not isinstance(self.tensor, torch.Tensor) or not self.tensor.is_sparse:
                raise TypeError("sparse expects self.tensor to be a torch sparse tensor.")
            coords = self.tensor.indices().numpy()       # shape (nnz, ndim)
            data   = self.tensor.values().numpy()        # shape (nnz,)
            shape  = tuple(self.tensor.size())  # e.g. (d0, d1, ..., d_{n-1})
            sparse_tensor = sparse.COO(coords, data, shape=shape)
            return sparse_tensor

        elif sparse_type == "cupy":
            if not isinstance(self.tensor, torch.Tensor) or not self.tensor.is_sparse:
                raise TypeError("cupy expects self.tensor to be a torch sparse tensor.")
            tensor_cupy, shape = torch_sparse_to_cupy(self.tensor)
            return tensor_cupy
        else:
            raise NotImplementedError(f"Sparse representation for type {sparse_type} not implemented.")



    def tensor_to_sparse(self, sparse_type="tensorflow"):
        self.tensor = self.sparse_representation(sparse_type)
        self.sparsity_type = sparse_type
        if sparse_type in ["tensorflow", "cupy"]:
            self.device = "cuda"


    def tensor_to_dense(self):
        if not isinstance(self.tensor, torch.Tensor) or not self.tensor.is_sparse:
            raise TypeError("tensor_to_dense expects self.tensor to be a torch sparse tensor.")
        self.tensor = self.tensor.to_dense()
        self.sparsity_type = "dense"

    def to_device(self, device):
        self.tensor = tree_to_device(self.tensor, device)
        self.device = device
        if device == "cpu":
            torch.cuda.empty_cache()

    def inspect(self):
        print("type:", type(self.tensor))
        print("sparsity type:", self.sparsity_type)
        print("shape:", self.shape)
        print("device:", self.device)

        if not self.sparsity_type or self.sparsity_type == "dense":
            memory_size = self.tensor.element_size() * self.tensor.nelement()
        elif self.sparsity_type == "torch":
            nnz = self.tensor._nnz()
            dtype_size = self.tensor.values().element_size()
            memory_size = nnz * (self.tensor.indices().element_size() * self.tensor.indices().shape[0] + dtype_size)

        elif self.sparsity_type == "cupy":
            memory_size = self.tensor.data.nbytes + self.tensor.row.nbytes + self.tensor.col.nbytes
        elif self.sparsity_type == "sparse":
            memory_size = self.tensor.nbytes
        elif self.sparsity_type == "tensorflow":
            nnz = self.tensor.values.shape[0]
            dtype_size = self.tensor.values.dtype.size
            memory_size = nnz * (self.tensor.indices.dtype.size * self.tensor.indices.shape[1] + dtype_size)

        else:
            memory_size = self.tensor.nbytes
        print(f"approx. memory size: {memory_size / (1024**2):.2f} MB")

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

    # These are the flattened indices of the original tensor
    flat_cp = cu.row  # shape (nnz,)
    data_cp = cu.data

    # Move flat indices to host for unraveling
    flat_np = cp.asnumpy(flat_cp)
    orig_shape = tuple(orig_shape)

    # Recover N-D coordinates
    coords = np.unravel_index(flat_np, orig_shape)  # tuple of ndarrays
    # coords[d][k] gives index along dim d for k-th nnz

    # Row indices in the unfolded matrix are just the mode coordinate
    row_np = coords[mode]

    # Column indices are the flattened coordinates of all other modes
    other_coords = coords[:mode] + coords[mode + 1 :]
    other_shape = tuple(s for i, s in enumerate(orig_shape) if i != mode)
    col_np = np.ravel_multi_index(other_coords, other_shape)

    # Back to CuPy
    row_cp = cp.asarray(row_np)
    col_cp = cp.asarray(col_np)

    unfolded_shape = (orig_shape[mode], int(np.prod(other_shape)))
    unfolded = cpx_sparse.coo_matrix(
        (data_cp, (row_cp, col_cp)),
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

    # Build new shape
    new_shape = list(old_shape)
    new_shape[mode] = new_dim
    new_shape = tuple(new_shape)

    # Other dimensions (all except mode)
    other_shape = tuple(s for i, s in enumerate(old_shape) if i != mode)

    # --- get row, col, data depending on type ---
    if cpx_sparse.isspmatrix(unfolded):
        # sparse path (original behavior)
        unfolded = unfolded.tocoo()
        row = unfolded.row        # (nnz,)
        col = unfolded.col        # (nnz,)
        data = unfolded.data      # (nnz,)
    else:
        # dense path: assume cupy ndarray of shape (new_dim, prod(other_dims))
        # find non-zeros
        row, col = cp.nonzero(unfolded)    # each (nnz,)
        data = unfolded[row, col]          # (nnz,)

    # Decode column index into coordinates of the "other" modes
    coords_other = cp.unravel_index(col, other_shape)  # tuple of (N-1) arrays

    # Now assemble full coordinates (length N) in the new tensor:
    # at position `mode` we put row, others we take from coords_other
    coords_full = []
    idx_other = 0
    for i in range(N):
        if i == mode:
            coords_full.append(row)
        else:
            coords_full.append(coords_other[idx_other])
            idx_other += 1

    coords_full = tuple(coords_full)

    # Flatten multi-index into a single index for the vectorized representation
    flat = cp.ravel_multi_index(coords_full, new_shape)  # (nnz,)
    zero = cp.zeros_like(flat)  # column index always 0 for (prod(new_shape), 1)
    vec_sparse = cpx_sparse.coo_matrix(
        (data, (flat, zero)),
        shape=(int(np.prod(new_shape)), 1),
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
    # print("Final core shape (sparse):", core_shape)
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
    return_errors=False,
    normalize_factors=False,
):
    if not isinstance(sparse_tensor, SparseTupleTensor):
        raise TypeError("sparse_tensor must be a SparseTupleTensor instance.")
    if not sparse_tensor.sparsity_type == "cupy":
        raise ValueError("sparse_tensor must have sparsity_type 'cupy'.")
    # we import "validate tucker rank"
    shape = tuple(sparse_tensor.shape)
    tensor = sparse_tensor.tensor
    rank = validate_tucker_rank(shape, rank=rank)
    epsilon = 1e-12
    modes = list(range(len(rank)))
    non_negative = True
    skip_factor = None
    transpose_factors = False


    if init == "random":
        rng = tl.check_random_state(random_state)
        core = tl.tensor(
            rng.random_sample([rank[index] for index in range(len(modes))]) + 0.01,
            **tl.context(tensor),
        )  # Check this
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

    for iteration in range(n_iter_max):
        # iter_time = time.time()
        for mode in modes:
            # the first steps can be done with the dense implementation
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

        if iteration > 1 and tl.abs(rec_errors[-2] - rec_errors[-1]) < tol:
            if verbose:
                print(f"converged in {iteration} iterations.")
            break
        if normalize_factors:
            core, factors = tucker_normalize((core, factors))
        # iter_time_end = print_elapsed_time(iter_time)
        # print("-----------------------------------------")

    tensor = TuckerTensor((core, factors))
    if return_errors:
        return tensor, rec_errors
    else:
        return tensor




# we perform the non negative tucker
print("\nStarting non-negative Tucker decomposition on cupy sparse tensor:")
for method in ["counting", "sc", "sii"]:
    for dim in [1500, 2000, 2500]:
        time = time.time()
        print(f"\nMethod: {method}, Dimensionality: {dim}")
        tl.set_backend("cupy")
        sparse_tensor = SparseTupleTensor.load_from_disk(dataset="fineweb_sparse",
                                                         method=method,
                                                         dims=dim,
                                                         map_location="cpu",
                                                         sparsity_type="torch")
        print("\nConverting to cupy sparse representation:")
        sparse_tensor.tensor_to_sparse("cupy")
        tucker_decomp, errors = non_negative_tucker(
            sparse_tensor,
            rank=[100, 100, 100],
            n_iter_max=1000,
            init="random",
            tol=10e-12,
            random_state=42,
            verbose=True,
            return_errors=True,
            normalize_factors=True,
        )

        # we save this decomposition to disk
        output_dir = os.path.join(DATA_DIR, "tensors", "fineweb_sparse", "decomposition")
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, f"{method}_{dim}d_100r_1000i.pt")

        # the final output should be a Torch tensor on CPU
        tl.set_backend("pytorch")
        core, factors = tucker_decomp
        core = tl.tensor(cp.asnumpy(core))
        factors = [tl.tensor(cp.asnumpy(f)) for f in factors]
        tucker_decomp_torch = TuckerTensor((core, factors))
        torch.save(tucker_decomp_torch, output_path)
        notify_discord(f"Saved Tucker decomposition {method} - {dim} to {output_path} in {time.time() - time:.2f} seconds.")

