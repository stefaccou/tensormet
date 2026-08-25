"""
tt_ops.py — nonnegative Tucker-TT hybrid multiplicative-update kernels (KL).

EXPERIMENTAL (see README.md in this directory). The factor matrices and their
meaning are exactly Tucker's; only the R^N core is replaced by a tensor train
(tt_chain.py). Because the model stays multilinear and nonnegative in every
block — each factor, each TT core — the Lee & Seung KL/Poisson MU rules apply
per block unchanged, and a sweep is monotone.

Both MU denominators are sums over ALL tensor entries, and both are closed
forms here: run the same chain with the factor column sums in place of the
gathered latents. Nothing in this module ever allocates an R^N object.

Conventions follow distance.py: CuPy in/out, NNZ streamed through
``coo_to_coords`` (free for CoordCOO), ``thread_budget``/``epsilon``/``verbose``
kwargs, ε-clipping against zero-locking.
"""
from __future__ import annotations

import math
import os

import numpy as np
import tensorly as tl
from tqdm import tqdm

from tensormet.distance import coo_to_coords, _gpu_free_bytes
from tensormet.utils import make_lazy_cupy_pair
from tensormet.experimental.TT_hybrid.tt_chain import (
    bond_dims, core_shapes, left_envs, right_envs, site_grad, sites,
)

cp, cpx_sparse = make_lazy_cupy_pair()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# Ceiling on the largest single allocation a batch may make.
MAX_BATCH_ALLOC_BYTES = int(os.environ.get(
    "TENSORMET_TT_MAX_BATCH_ALLOC_BYTES", 1 << 30))

# Floor for the OOM back-off.
MIN_BATCH_NNZ = 256


def _quantize_batch(n):
    """Round ``n`` down to ~1/32 resolution, so repeated estimates return the
    same value and the pool reuses blocks."""
    n = int(n)
    if n < 32:
        return max(1, n)
    granule = 1 << (n.bit_length() - 5)
    return max(1, (n // granule) * granule)


def estimate_batch_nnz_tt(tt_cores, factors, safety=0.7, temp_mult=3.0,
                          reserve_b=0, max_alloc_bytes=None):
    """Safe NNZ batch size for the streaming TT kernels.

    Per entry a batch holds the N site matrices (Σ ρ_k ρ_{k+1}), the gathered
    factor rows (Σ R_k) and the two environment stacks (2 Σ ρ_k).

    The free-memory division sizes against aggregate headroom, so it is capped
    by ``max_alloc_bytes`` on the batch's largest single temporary — a site
    matrix (ρ_k·ρ_{k+1}) or the core kernel's (b, ρ_k, R_k) temp.
    """
    per_entry = (sum(int(C.shape[0]) * int(C.shape[2]) for C in tt_cores)
                 + sum(int(f.shape[1]) for f in factors)
                 + 2 * sum(int(C.shape[0]) for C in tt_cores) + 2)
    itemsize = int(np.dtype(factors[0].dtype).itemsize)
    bytes_per_entry = max(1, int(math.ceil(per_entry * itemsize * temp_mult)))
    free_b = max(1, int(_gpu_free_bytes()) - int(reserve_b))
    batch = int(free_b * safety) // bytes_per_entry

    peak_per_entry = max(
        max(int(C.shape[0]) * int(C.shape[2]) for C in tt_cores),   # site
        max(int(C.shape[0]) * int(C.shape[1]) for C in tt_cores),   # LW temp
    )
    cap = (int(MAX_BATCH_ALLOC_BYTES if max_alloc_bytes is None else max_alloc_bytes)
           // max(1, peak_per_entry * itemsize))
    return _quantize_batch(max(1, min(batch, cap)))


def _colsum_batch(factors, epsilon):
    """Factor column sums shaped as a one-entry batch, so the chain helpers
    return the exact sum over ALL tensor entries."""
    return [cp.clip(cp.sum(f, axis=0), a_min=epsilon, a_max=None)[None, :] for f in factors]


def _scatter_add(out, rows, weights, dense):
    """out[rows[p], :] += weights[p] * dense[p, :], as one cuSPARSE SpMM
    (same skeleton as the largedim Tucker/CP factor kernels)."""
    b = int(dense.shape[0])
    S = cpx_sparse.csr_matrix(
        (weights, (rows.astype(cp.int32), cp.arange(b, dtype=cp.int32))),
        shape=(int(out.shape[0]), b),
    )
    out += S @ dense


def _run_nnz_batches(nnz, batch_nnz, body, verbose=False, desc=None):
    """Run ``body(start, end)`` over [0, nnz), halving the batch on OOM.

    ``body`` is a function so its locals — the site matrices and the per-batch
    temporaries, the largest arrays either kernel touches — are released when it
    returns, rather than staying bound until the next iteration overwrites them.

    Every kernel here accumulates on ``body``'s last statement, so a batch that
    raises has contributed nothing and can be redone at a smaller size.
    """
    b = max(1, int(batch_nnz))
    start = 0
    bar = tqdm(total=nnz, desc=desc, unit="nnz", leave=False) if verbose else None
    try:
        while start < nnz:
            end = min(start + b, nnz)
            failed = False
            try:
                body(start, end)
            except cp.cuda.memory.OutOfMemoryError:
                if b <= MIN_BATCH_NNZ:
                    raise
                failed = True
            if failed:
                # Outside the handler: the exception's traceback pins the failed
                # frame's arrays, so the flush only reclaims once it is gone.
                b = max(MIN_BATCH_NNZ, b // 2)
                cp.get_default_memory_pool().free_all_blocks()
                print(f"  [TT-KL] OOM in {desc or 'nnz pass'}; batch_nnz -> {b}")
                continue
            if bar is not None:
                bar.update(end - start)
            start = end
    finally:
        if bar is not None:
            bar.close()
    return b


def tt_sum_all_entries(tt_cores, factors, epsilon=1e-12):
    """Σ over ALL entries of X̂, in closed form (TT analogue of
    distance._tucker_sum_all_entries)."""
    sums = _colsum_batch(factors, epsilon)
    return cp.clip(left_envs(sites(tt_cores, sums, cp), cp)[len(tt_cores)][0, 0],
                   a_min=epsilon, a_max=None)


# ---------------------------------------------------------------------------
# KL / Poisson multiplicative updates
# ---------------------------------------------------------------------------

def tt_kl_factor_update(vec_tensor, core, factors, mode, shape,
                        thread_budget=None, epsilon=1e-12, verbose=False,
                        batch_nnz=None):
    """KL multiplicative update for factor A^(mode).

        A ← A ⊛ Num ⊘ den
        Num[i, r] = Σ_{p : i_mode(p) = i} (x_p / x̂_p) · Z_p[r]
        Z_p[r]    = ∂x̂_p / ∂A[i_mode(p), r]   (chain with site `mode` open)
        den[r]    = the same chain run on the factor column sums

    The updated columns are ℓ1-normalized and the scale absorbed into TT core
    `mode`. That is an exact reparametrization, and it is what keeps an N-fold
    chain product from drifting in scale; it also replaces ``tucker_normalize``,
    which needs a dense core (the training loop skips it for this family).

    ``core`` is the list of TT cores; core `mode` is updated IN PLACE by the
    rescale, so the loop's ``core`` variable stays current.
    """
    if verbose:
        print(f"  [TT-KL] Updating factor {mode}...")
    tt_cores = core
    shape = tuple(int(s) for s in shape)
    n_modes = len(shape)
    idxs, vals = coo_to_coords(vec_tensor, shape)
    nnz = int(vals.size)
    A = factors[mode]

    S_sum = sites(tt_cores, _colsum_batch(factors, epsilon), cp, skip=mode)
    den = cp.clip(
        site_grad(left_envs(S_sum, cp)[mode], tt_cores[mode], right_envs(S_sum, cp)[mode + 1], cp),
        a_min=epsilon, a_max=None,
    )  # (1, R_mode)

    num = cp.zeros_like(A)
    if nnz:
        if batch_nnz is None:
            batch_nnz = estimate_batch_nnz_tt(tt_cores, factors)

        def _body(start, end):
            mats = [factors[n][idxs[n][start:end]] for n in range(n_modes)]
            S = sites(tt_cores, mats, cp)
            L, R = left_envs(S, cp), right_envs(S, cp)
            xhat = cp.clip(L[n_modes][:, 0], a_min=epsilon, a_max=None)
            Z = site_grad(L[mode], tt_cores[mode], R[mode + 1], cp)
            _scatter_add(num, idxs[mode][start:end], vals[start:end] / xhat, Z)

        _run_nnz_batches(nnz, batch_nnz, _body, verbose, f"  [TT-KL] factor {mode}")

    A_new = cp.clip(A * (num / den), a_min=epsilon, a_max=None)
    scale = cp.clip(cp.sum(A_new, axis=0), a_min=epsilon, a_max=None)
    tt_cores[mode] *= scale[None, :, None]
    return cp.clip(A_new / scale[None, :], a_min=epsilon, a_max=None)


def tt_kl_tied_factor_update(vec_tensor, core, factors, group, shape,
                             thread_budget=None, epsilon=1e-12, verbose=False,
                             batch_nnz=None):
    """KL multiplicative update for ONE factor tied across the modes in ``group``.

        A ← A ⊛ (Σ_{n∈group} Num_n) ⊘ (Σ_{n∈group} den_n)

    A tied factor appears at |group| legs, so x̂ is degree-|group| in it and the
    single-leg rule is simply the wrong update: applying it once per mode and
    copying the result (the loop's ``factors[other] = factors[mode]``) compounds
    the correction to roughly ratio^|group|. Pooling both sides is the correct
    MU, and it restores the Poisson identity Σ_all x̂ = Σ x that the single-leg
    rule satisfies per leg — which is the cheapest runtime check that this is
    right (see 1_method_development/19_tt_hybrid/).

    Cost is ONE NNZ pass, not |group|: the left/right sweeps are built once per
    batch and reused for every site gradient. So this is ~|group|× cheaper than
    the per-mode updates it replaces, not more expensive.

    The ℓ1 rescale is applied to EVERY tied mode's core, so the reparametrization
    stays exact — absorbing it into one core only (as the single-leg kernel must)
    leaves the other legs uncompensated and multiplies x̂ by s^-(|group|-1).
    """
    if verbose:
        print(f"  [TT-KL] Updating tied factor group {tuple(group)}...")
    tt_cores = core
    shape = tuple(int(s) for s in shape)
    n_modes = len(shape)
    group = list(group)
    idxs, vals = coo_to_coords(vec_tensor, shape)
    nnz = int(vals.size)
    A = factors[group[0]]

    sums = _colsum_batch(factors, epsilon)
    den = cp.zeros((1, int(A.shape[1])), dtype=A.dtype)
    for n in group:
        S_sum = sites(tt_cores, sums, cp, skip=n)
        den += site_grad(left_envs(S_sum, cp)[n], tt_cores[n],
                         right_envs(S_sum, cp)[n + 1], cp)
    den = cp.clip(den, a_min=epsilon, a_max=None)

    num = cp.zeros_like(A)
    if nnz:
        if batch_nnz is None:
            batch_nnz = estimate_batch_nnz_tt(tt_cores, factors)

        def _body(start, end):
            mats = [factors[k][idxs[k][start:end]] for k in range(n_modes)]
            S = sites(tt_cores, mats, cp)
            L, R = left_envs(S, cp), right_envs(S, cp)
            w = vals[start:end] / cp.clip(L[n_modes][:, 0], a_min=epsilon, a_max=None)
            for n in group:
                Z = site_grad(L[n], tt_cores[n], R[n + 1], cp)
                _scatter_add(num, idxs[n][start:end], w, Z)

        _run_nnz_batches(nnz, batch_nnz, _body, verbose,
                         f"  [TT-KL] tied factors {tuple(group)}")

    A_new = cp.clip(A * (num / den), a_min=epsilon, a_max=None)
    scale = cp.clip(cp.sum(A_new, axis=0), a_min=epsilon, a_max=None)
    for n in group:
        tt_cores[n] *= scale[None, :, None]
    return cp.clip(A_new / scale[None, :], a_min=epsilon, a_max=None)


def tt_kl_core_update(vec_tensor, shape, core, factors, modes=None,
                      thread_budget=None, epsilon=1e-12, verbose=False,
                      batch_nnz=None):
    """KL multiplicative update of the TT cores, one site at a time.

        C_k ← C_k ⊛ Num_k ⊘ Den_k
        Num_k[a, r, b] = Σ_p (x_p / x̂_p) · L_k[p, a] · A_k[i_k(p), r] · R_{k+1}[p, b]
        Den_k[a, r, b] = Ls_k[a] · s_k[r] · Rs_{k+1}[b]      (column-sum chain)

    Sites are swept sequentially so each update sees the previous ones — that
    is what makes the sweep a genuine block MU (and therefore monotone). The
    price is one NNZ pass per site; each pass is O(N ρ² R) per entry, against
    Tucker's O(R^N).
    """
    if verbose:
        print("  [TT-KL] Updating TT cores...")
    tt_cores = core
    shape = tuple(int(s) for s in shape)
    n_modes = len(shape)
    if modes is not None and list(modes) != list(range(n_modes)):
        raise NotImplementedError("tt_kl_core_update assumes modes == range(N).")

    idxs, vals = coo_to_coords(vec_tensor, shape)
    nnz = int(vals.size)
    if nnz == 0:
        return tt_cores
    if batch_nnz is None:
        batch_nnz = estimate_batch_nnz_tt(tt_cores, factors)

    sums = _colsum_batch(factors, epsilon)
    for k in range(n_modes):
        S_sum = sites(tt_cores, sums, cp)
        Ls, Rs = left_envs(S_sum, cp), right_envs(S_sum, cp)
        den = cp.clip(
            Ls[k][0][:, None, None] * sums[k][0][None, :, None] * Rs[k + 1][0][None, None, :],
            a_min=epsilon, a_max=None,
        )

        num = cp.zeros_like(tt_cores[k])

        def _body(start, end, k=k):
            mats = [factors[n][idxs[n][start:end]] for n in range(n_modes)]
            S = sites(tt_cores, mats, cp)
            L, R = left_envs(S, cp), right_envs(S, cp)
            w = vals[start:end] / cp.clip(L[n_modes][:, 0], a_min=epsilon, a_max=None)
            # Σ_p w_p · outer(L_k[p], A_k row, R_{k+1}[p]) as one GEMM.
            LW = (L[k] * w[:, None])[:, :, None] * mats[k][:, None, :]   # (b, ρ_k, R_k)
            # [...] so the accumulation stays in place and needs no nonlocal.
            num[...] += (LW.reshape(end - start, -1).T @ R[k + 1]).reshape(num.shape)

        _run_nnz_batches(nnz, batch_nnz, _body, verbose, f"  [TT-KL] core site {k}")

        tt_cores[k] = cp.clip(tt_cores[k] * (num / den), a_min=epsilon, a_max=None)
    return tt_cores


def tt_kl_compute_errors(vec_tensor, shape, core, factors,
                         thread_budget=None, epsilon=1e-12, verbose=False,
                         batch_nnz=None):
    """Relative generalized KL divergence, normalized exactly as
    ``distance.kl_compute_errors_largedim``:

        D   = Σ_nnz [x log(x/x̂) − x + x̂] + (Σ_all x̂ − Σ_nnz x̂)
        rel = D / Σ_nnz x

    Σ_all x̂ is closed form, so — unlike Tucker's KL error — this never builds a
    dense reconstruction and needs no CPU excursion.
    """
    if verbose:
        print("  [TT-KL] Computing KL errors...")
    tt_cores = core
    shape = tuple(int(s) for s in shape)
    n_modes = len(shape)
    idxs, vals = coo_to_coords(vec_tensor, shape)
    nnz = int(vals.size)

    sum_all = tt_sum_all_entries(tt_cores, factors, epsilon=epsilon)
    if nnz == 0:
        # rel = D / Σ_nnz x is undefined with no data; nan rather than D/ε.
        return cp.asarray(float("nan"), dtype=sum_all.dtype)

    x_nz = cp.clip(cp.asarray(vals), a_min=epsilon, a_max=None)
    if batch_nnz is None:
        batch_nnz = estimate_batch_nnz_tt(tt_cores, factors)

    kl_pos = cp.asarray(0.0, dtype=factors[0].dtype)
    sum_xhat_nz = cp.asarray(0.0, dtype=factors[0].dtype)

    def _body(start, end):
        mats = [factors[n][idxs[n][start:end]] for n in range(n_modes)]
        # Only the left sweep is needed for x̂.
        xhat = cp.clip(left_envs(sites(tt_cores, mats, cp), cp)[n_modes][:, 0],
                       a_min=epsilon, a_max=None)
        x_b = x_nz[start:end]
        # Both reductions before either accumulation, so a retry cannot
        # double-count the first.
        pos = cp.sum(x_b * cp.log(x_b / xhat) - x_b + xhat)
        tot = cp.sum(xhat)
        kl_pos[...] += pos
        sum_xhat_nz[...] += tot

    _run_nnz_batches(nnz, batch_nnz, _body, verbose, "  [TT-KL] error x̂ pass")

    kl_total = kl_pos + (sum_all - sum_xhat_nz)
    return kl_total / cp.maximum(cp.sum(x_nz), epsilon)


# ---------------------------------------------------------------------------
# Initialization
# ---------------------------------------------------------------------------

def initialize_tucker_tt(sparse_tensor, shape, rank, modes, init, random_state,
                         tt_rank=100, thread_budget=None, epsilon=1e-12):
    """Initialize ``(tt_cores, factors)`` for the hybrid.

    Factors come from the existing Tucker init (``sparse_ops.initialize_nonnegative_tucker``,
    called with ``with_core=False``): the dense core it would otherwise build is
    exactly the object this format exists to avoid. The TT cores start
    uniform-random and are then rescaled once so that Σ_all x̂ = Σ x, which is
    where the KL MU wants to begin.

    Returns
    -------
    (tt_cores, factors) : (list of (ρ_k, R_k, ρ_{k+1}) CuPy arrays,
                           list of (I_n, R_n) CuPy arrays)
    """
    from tensormet.sparse_ops import initialize_nonnegative_tucker

    rank = [int(r) for r in rank]
    rng = tl.check_random_state(random_state)

    if isinstance(init, (tuple, list)) and len(init) == 2:
        tt_cores, factors = init
        return [cp.asarray(C) for C in tt_cores], [cp.asarray(f) for f in factors]
    if init == "random":
        factors = [tl.tensor(rng.random_sample((shape[m], rank[i])), **tl.context(sparse_tensor))
                   for i, m in enumerate(modes)]
    elif isinstance(init, str) and "svd" in init:
        _core, factors = initialize_nonnegative_tucker(
            sparse_tensor, shape, rank, modes, init, random_state,
            thread_budget=thread_budget, with_core=False,
        )
    else:
        raise ValueError(
            f"TT init must be 'random', an svd variant, or a (tt_cores, factors) "
            f"tuple; got {init!r}"
        )

    factors = [cp.clip(cp.abs(cp.asarray(f)), a_min=1e-30, a_max=None) for f in factors]
    dtype = factors[0].dtype
    tt_cores = [cp.asarray(rng.random_sample(s) + 0.01, dtype=dtype)
                for s in core_shapes(rank, tt_rank)]

    _idxs, vals = coo_to_coords(sparse_tensor, tuple(int(s) for s in shape))
    sum_x = cp.clip(cp.sum(vals), a_min=epsilon, a_max=None)
    tt_cores[0] *= sum_x / tt_sum_all_entries(tt_cores, factors, epsilon=epsilon)
    return tt_cores, factors


__all__ = [
    "bond_dims",
    "core_shapes",
    "estimate_batch_nnz_tt",
    "initialize_tucker_tt",
    "tt_kl_compute_errors",
    "tt_kl_core_update",
    "tt_kl_factor_update",
    "tt_kl_tied_factor_update",
    "tt_sum_all_entries",
]
