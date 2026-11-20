"""
Streamed GPU SVD init (``--init svd_stream``).

For each group T of tied modes (or one untied mode), the factor is NNDSVD of
the top-k eigenvectors of

    G = Σ_{n∈T} A_n A_nᵀ        (V × V; A_n = mode-n unfolding)

found by randomized subspace iteration (``--svd-fill`` optionally fills the
NNDSVD zeros, see ``fill_zeros``). G is never formed: G·Ω streams the
NNZ grouped by *context* (the n-gram with position n left out), two sparse
products per chunk of whole contexts.

Contexts are hashed into cells. A cell holds whole contexts, so the Gram
product is a plain sum over cells, and each cell is sorted on its own:
  - multi-GPU: every GPU owns a share of the cells, so each GPU's partial
    G_k·Ω is exact; the partials are summed on the home GPU. (The training
    shards split contexts, so they cannot be reused: a split column
    a = a¹ + a² gives a¹a¹ᵀ + a²a²ᵀ, not aaᵀ.)
  - memory: a GPU's share is split into enough cells that one cell's sort
    fits in free memory. Groupings stay resident when they fit; otherwise they
    are rebuilt cell by cell in every Gram application (streaming).

Results are cached on disk (init_cache.py).
"""
from __future__ import annotations

import math
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import List, Optional, Sequence

import numpy as np
import scipy.linalg
from tqdm import tqdm

from tensormet.distance import coo_to_coords, _gpu_free_bytes
from tensormet.sparse_ops import (
    _array_device_id,
    _compute_tucker_core_batched,
    _nndsvd_factors_gpu,
    build_batch_csr_T,
    spmm_T,
)
from tensormet.utils import make_lazy_cupy_pair
from tensormet.stream_svd.init_cache import InitCache, fingerprint

cp, cpx_sparse = make_lazy_cupy_pair()

# Sketch settings (defaults; --svd-power-iter / --svd-oversampling override them).
# Each G application is one SVD power iteration (G = A Aᵀ), so the total is
# power_iter + 2 applications (range, powers, Rayleigh-Ritz).
_OVERSAMPLING = 100
_POWER_ITER = 6
# A column counts as converged below this relative residual ‖G u − λ u‖ / λ.
_RESIDUAL_TOL = 1e-3

# Share of free VRAM one chunk of the Gram product may use.
_CHUNK_FRACTION = 0.25
# Share of free VRAM one cell's sort may use.
_SORT_FRACTION = 0.4
# Share of free VRAM resident groupings (+ bookkeeping) may take.
_RESIDENT_FRACTION = 0.7
# Per-chunk NNZ cap; keeps the int32 CSR indptr far from overflow.
_MAX_CHUNK_NNZ = 2 ** 30
_INT64_MAX = 2 ** 63 - 1
_INT32_MAX = 2 ** 31 - 1

# Test knobs: force more cells per GPU / force streaming / force the
# rank-compacted sort key.
_MIN_CELLS = 1
_FORCE_STREAM = False
_FORCE_COMPACT = False


def sketch_tag(power_iter: int, oversampling: int) -> str:
    return f"q{int(power_iter)}o{int(oversampling)}"


def _resolve_sketch(power_iter=None, oversampling=None):
    """(q, o) with None replaced by the module defaults, read at call time."""
    q = _POWER_ITER if power_iter is None else int(power_iter)
    o = _OVERSAMPLING if oversampling is None else int(oversampling)
    if q < 0 or o < 0:
        raise ValueError(f"svd_stream: power_iter and oversampling must be >= 0; got {q}, {o}")
    return q, o


# ---------------------------------------------------------------------------
# Run context (set by solvers.py before the init dispatch)
# ---------------------------------------------------------------------------
_RUN_CONTEXT: Optional[dict] = None


def set_run_context(cfg=None, *, tied_groups=None, device_ids=None, use_cache=True,
                    power_iter=None, oversampling=None, fill=0.0):
    """Hand run-level settings to the next ``initialize_svd_stream`` call.

    The init is reached through ``initialize_nonnegative_tucker`` from every
    decomposition, which knows nothing about the run; this avoids threading
    these through all of them. Consumed (cleared) by the call.
    """
    global _RUN_CONTEXT
    _RUN_CONTEXT = {"cfg": cfg, "tied_groups": tied_groups,
                    "device_ids": device_ids, "use_cache": use_cache,
                    "power_iter": power_iter, "oversampling": oversampling, "fill": fill}


def _pop_run_context() -> dict:
    global _RUN_CONTEXT
    ctx, _RUN_CONTEXT = (_RUN_CONTEXT or {}), None
    return ctx


# ---------------------------------------------------------------------------
# Context grouping
# ---------------------------------------------------------------------------
@dataclass
class ContextGrouping:
    """One cell's NNZ for one mode, sorted by context, then by the mode's own
    index. Context j is ``[offsets[j]:offsets[j+1])``."""
    rows: "cp.ndarray"      # (nnz,) int32 mode-n index
    vals: "cp.ndarray"      # (nnz,)
    offsets: "cp.ndarray"   # (n_ctx + 1,) int32 (int64 past int32)
    chunks: list = None     # [(c0, c1, lo, hi)] host ints, from plan_chunks

    @property
    def nnz(self) -> int:
        return int(self.rows.size)

    @property
    def n_ctx(self) -> int:
        return int(self.offsets.size) - 1


def _sort_key(others, other_dims, rows, V):
    """int64 key ordering entries by (context, row).

    Ravels the coordinates directly while the product fits int64; past that
    (6-grams, or 4-grams at V = 131k), the running key is first replaced by its
    rank among the distinct values. Ranks keep the order, so both routes sort
    identically.
    """
    cols = list(others) + [rows]
    dims = list(other_dims) + [V]
    key = cols[0].astype(cp.int64)
    bound = int(dims[0])
    for c, d in zip(cols[1:], dims[1:]):
        if _FORCE_COMPACT or bound * int(d) > _INT64_MAX:
            key = cp.unique(key, return_inverse=True)[1].reshape(-1).astype(cp.int64)
            bound = int(key.max()) + 1 if key.size else 1
        key = key * int(d) + c.astype(cp.int64)
        bound *= int(d)
    return key


def group_by_context(others, other_dims, rows, vals, V) -> ContextGrouping:
    """Sort one cell's entries into a ContextGrouping (on the current device)."""
    n = int(rows.size)
    off_dtype = cp.int32 if n <= _INT32_MAX else cp.int64
    if n == 0:
        return ContextGrouping(cp.zeros(0, cp.int32), vals[:0], cp.zeros(1, off_dtype))
    key = _sort_key(others, other_dims, rows, V)
    order = cp.argsort(key)
    key = key[order]
    rows_s = rows[order].astype(cp.int32, copy=False)
    vals_s = vals[order]
    del order
    ctx = key // int(V)            # row is the last digit of the key
    del key
    starts = cp.flatnonzero(cp.concatenate((cp.ones(1, bool), ctx[1:] != ctx[:-1])))
    del ctx
    offsets = cp.concatenate((starts, cp.asarray([n]))).astype(off_dtype)
    return ContextGrouping(rows_s, vals_s, offsets)


def plan_chunks(g: ContextGrouping, p: int, budget_bytes: int) -> None:
    """Split g into chunks of whole contexts that fit ``budget_bytes``.

    Per chunk: Z = Pᵀ-side product (contexts × p) plus its F-order copy, and
    the CSR (values + indices). A single context can exceed the NNZ cap by at
    most V entries, which the cap leaves room for.
    """
    item = int(g.vals.dtype.itemsize)
    ctx_cap = max(1, budget_bytes // (3 * p * item + 8))
    nnz_cap = max(1, min(budget_bytes // (item + 4), _MAX_CHUNK_NNZ))
    n_ctx, nnz = g.n_ctx, g.nnz
    if n_ctx == 0:
        g.chunks = []
        return
    by_ctx = cp.arange(0, n_ctx, ctx_cap, dtype=cp.int64)
    by_nnz = cp.searchsorted(g.offsets.astype(cp.int64, copy=False),
                             cp.arange(0, nnz, nnz_cap, dtype=cp.int64), side="right") - 1
    cuts = cp.unique(cp.concatenate((by_ctx, by_nnz.astype(cp.int64),
                                     cp.asarray([n_ctx], dtype=cp.int64))))
    c = cp.asnumpy(cuts).tolist()
    o = cp.asnumpy(g.offsets[cuts]).tolist()
    g.chunks = [(c[i], c[i + 1], o[i], o[i + 1]) for i in range(len(c) - 1) if o[i + 1] > o[i]]


def gram_apply_local(groupings: Sequence[ContextGrouping], omega, V: int, on_chunk=None):
    """Σ_g A_g (A_gᵀ Ω) on the current device, chunk by chunk.
    ``on_chunk(n_chunks)`` is called after each chunk (progress reporting)."""
    omega = cp.asfortranarray(omega)      # cuSPARSE's layout; once, not per chunk
    Y = cp.zeros((V, omega.shape[1]), dtype=omega.dtype)
    for g in groupings:
        for c0, c1, lo, hi in g.chunks:
            indptr = (g.offsets[c0:c1 + 1] - lo).astype(cp.int32)
            P = build_batch_csr_T(g.vals[lo:hi], g.rows[lo:hi], c1 - c0, V, indptr)  # A_chunkᵀ
            Z = P @ omega                     # A_chunkᵀ Ω   (contexts × p)
            Y += spmm_T(P, Z)                 # A_chunk Z    (V × p)
            del P, Z
            if on_chunk is not None:
                on_chunk(len(g.chunks))
    return Y


# ---------------------------------------------------------------------------
# Cells: hash partition of the contexts
# ---------------------------------------------------------------------------
_CELL_KERNELS: dict = {}


def cell_ids(cols, n_cells: int):
    """int32 cell per entry from a hash of its context coordinates ``cols``
    (one array per other mode, same dtype). Same context → same cell, on any
    device. One fused kernel: 4 B/NNZ, no temporaries."""
    n_in = len(cols)
    kern = _CELL_KERNELS.get(n_in)
    if kern is None:
        params = ", ".join(f"T c{j}" for j in range(n_in)) + ", uint64 n_cells"
        mix = "".join(f"h ^= (unsigned long long)c{j}; h *= 0xBF58476D1CE4E5B9ULL; h ^= h >> 31;\n"
                      for j in range(n_in))
        body = ("unsigned long long h = 0x9E3779B97F4A7C15ULL;\n" + mix +
                "h ^= h >> 33; h *= 0xFF51AFD7ED558CCDULL; h ^= h >> 33;\n"
                "cell = (int)(h % n_cells);")
        kern = cp.ElementwiseKernel(params, "int32 cell", body, f"svd_stream_cell{n_in}")
        _CELL_KERNELS[n_in] = kern
    return kern(*cols, np.uint64(n_cells))


def _free_on(did) -> int:
    with cp.cuda.Device(did):
        return _gpu_free_bytes()


def _held_bytes(k, nnz, m, n_modes, itemsize) -> int:
    """Bytes a device holds while its cells are sorted, besides the tensor:
    home keeps the cell ids + one mask over all NNZ (5 B); the others keep
    their shipped share (coords, rows, vals, cell ids)."""
    return 5 * nnz if k == 0 else (4 * (n_modes - 1) + 8 + itemsize) * m


def _n_cells_per_device(nnz, n_modes, itemsize, device_ids) -> int:
    """Cells per GPU so that one cell's sort fits in ``_SORT_FRACTION`` of
    what each device has left."""
    m = math.ceil(nnz / len(device_ids))
    sort_bytes = 4 * n_modes + 80      # gathered coords, idx, key build, argsort
    avail = min(_free_on(d) - _held_bytes(k, nnz, m, n_modes, itemsize)
                for k, d in enumerate(device_ids))
    if avail <= 0:
        raise MemoryError(
            f"svd_stream: no GPU memory left to sort {nnz} NNZ over {len(device_ids)} GPU(s); "
            f"use more GPUs or free memory on the home GPU")
    n = math.ceil(m * sort_bytes / (_SORT_FRACTION * avail))
    n = max(n, math.ceil(m / (_INT32_MAX // 2)), _MIN_CELLS)
    return max(1, n)


def _resident_fits(nnz, n_group, n_modes, itemsize, device_ids) -> bool:
    """Whether all of a group's groupings can stay resident: rows + int32
    offsets + vals per entry and mode, plus what the build holds."""
    if _FORCE_STREAM:
        return False
    m = math.ceil(nnz / len(device_ids))
    for k, did in enumerate(device_ids):
        keep = (8 + itemsize) * m * n_group
        if keep + _held_bytes(k, nnz, m, n_modes, itemsize) > _RESIDENT_FRACTION * _free_on(did):
            return False
    return True


@dataclass
class _Source:
    """One device's entries for one mode, and which cells to build from them."""
    others: list            # per other mode, (n,) coordinates
    rows: "cp.ndarray"
    vals: "cp.ndarray"
    ids: "cp.ndarray"       # cell id per entry
    targets: list           # cell ids this device builds, one grouping each


def _prepare_sources(idxs, vals, shape, mode, device_ids, home):
    """Hash contexts into cells on the home device, ship each other GPU its
    share over the host. Home keeps its share as views of the tensor plus the
    cell ids, so nothing of the tensor is copied there.

    Returns per device either a ready ``_Source`` (home) or host arrays to
    upload on that device (``_upload``), plus the other modes' dims.
    """
    G = len(device_ids)
    others = [m for m in range(len(shape)) if m != mode]
    other_dims = [shape[m] for m in others]
    n_cells = _n_cells_per_device(int(vals.size), len(shape), int(vals.dtype.itemsize), device_ids)
    with cp.cuda.Device(home):
        cell = cell_ids([idxs[m] for m in others], G * n_cells)    # device = cell % G
        sources = [None] * G
        for k in range(1, G):
            sel = cp.flatnonzero((cell % G) == k)
            # one array at a time: home only ever holds sel + one gathered array
            sources[k] = {
                "others": [cp.asnumpy(idxs[m][sel].astype(cp.int32)) for m in others],
                "rows": cp.asnumpy(idxs[mode][sel].astype(cp.int32)),
                "vals": cp.asnumpy(vals[sel]),
                "ids": cp.asnumpy(cell[sel] // G),
                "targets": list(range(n_cells)),
            }
            del sel
        sources[0] = _Source([idxs[m] for m in others], idxs[mode], vals, cell,
                             [b * G for b in range(n_cells)])
    return sources, other_dims, n_cells


def _upload(src) -> _Source:
    """Host share → _Source on the current device (no-op for the home _Source)."""
    if isinstance(src, _Source):
        return src
    return _Source([cp.asarray(a) for a in src["others"]], cp.asarray(src["rows"]),
                   cp.asarray(src["vals"]), cp.asarray(src["ids"]), src["targets"])


def _iter_groupings(src: _Source, other_dims, V):
    """Yield one ContextGrouping per target cell, sorting one cell at a time."""
    for t in src.targets:
        idx = cp.flatnonzero(src.ids == t)
        if idx.size == 0:
            continue
        g = group_by_context([o[idx] for o in src.others], other_dims, src.rows[idx],
                             src.vals[idx], V)
        del idx
        yield g


def _warm_up(device_ids, dtype) -> None:
    """Load cuBLAS/cuSPARSE kernels on each device serially, before threads
    race for them (cf. ShardedSparseTensor._warm_up_gpus)."""
    for did in device_ids:
        with cp.cuda.Device(did):
            a = cp.ones((2, 2), dtype=dtype)
            _ = a @ a
            P = cpx_sparse.csr_matrix(a)
            _ = P @ a
            _ = spmm_T(P, a)
            cp.cuda.Device(did).synchronize()


def _run_on_devices(fn, device_ids, pool):
    return [fn(0)] if len(device_ids) == 1 else list(pool.map(fn, range(len(device_ids))))


class _ModeLayout:
    """One mode's groupings, kept resident: ``groupings[k]`` is the list of
    cell groupings on device k."""

    def __init__(self, idxs, vals, shape, mode, device_ids, home, pool):
        sources, other_dims, self.n_cells = _prepare_sources(idxs, vals, shape, mode, device_ids, home)
        V = int(shape[mode])

        def build(k):
            with cp.cuda.Device(device_ids[k]):
                src = _upload(sources[k])
                sources[k] = None
                gs = list(_iter_groupings(src, other_dims, V))
                del src
                return gs

        self.groupings = _run_on_devices(build, device_ids, pool)

    def plan(self, device_ids, p) -> None:
        for did, gs in zip(device_ids, self.groupings):
            with cp.cuda.Device(did):
                for g in gs:
                    plan_chunks(g, p, int(_CHUNK_FRACTION * _gpu_free_bytes()))


# ---------------------------------------------------------------------------
# Subspace iteration
# ---------------------------------------------------------------------------
def _orthonormalize(Y):
    """Orthonormal basis of span(Y) in fp64: CholeskyQR2, host QR if Cholesky fails."""
    Q = Y.astype(cp.float64)
    p = Q.shape[1]
    for _ in range(2):
        Gm = cp.asnumpy(Q.T @ Q)
        Gm = (Gm + Gm.T) * 0.5
        Gm += np.eye(p) * (np.abs(np.diag(Gm)).mean() * 1e-12 + 1e-300)
        try:
            L = np.linalg.cholesky(Gm)
        except np.linalg.LinAlgError:
            q, _r = np.linalg.qr(cp.asnumpy(Q))
            return cp.asarray(q)
        L_inv = scipy.linalg.solve_triangular(L, np.eye(p), lower=True)
        Q = Q @ cp.asarray(L_inv.T)     # QᵀQ = L Lᵀ  ⇒  Q L⁻ᵀ is orthonormal
    return Q


def top_k_eig(apply, m: int, k: int, dtype, seed: int, power_iter=None, oversampling=None):
    """Top-k eigenpairs of the PSD operator ``apply`` (m × m), on the current device.

    Returns (U (m × k), s (k,), r (k,)) with s = sqrt(eigenvalue), i.e. the
    singular values of the stacked unfoldings, and r the relative residuals
    ‖G u_j − λ_j u_j‖ / λ_j (host array; 0 = an exact eigenvector).
    """
    q, o = _resolve_sketch(power_iter, oversampling)
    k = min(int(k), m)
    p = min(k + o, m)
    rng = cp.random.RandomState(seed)
    Y = apply(rng.standard_normal((m, p)).astype(dtype))
    for _ in range(q):
        Y = apply(_orthonormalize(Y).astype(dtype))
    Q = _orthonormalize(Y)
    W = apply(Q.astype(dtype)).astype(cp.float64)       # Rayleigh-Ritz
    B = cp.asnumpy(Q.T @ W)
    B = (B + B.T) * 0.5
    evals, evecs = np.linalg.eigh(B)                     # ascending
    top = np.argsort(evals)[::-1][:k]
    E = cp.asarray(np.ascontiguousarray(evecs[:, top]))
    U64 = Q @ E
    lam = cp.asarray(np.maximum(evals[top], 0.0))
    # W = G Q, so G U = W E: the residuals need no extra Gram application.
    R = W @ E - U64 * lam[cp.newaxis, :]
    r = cp.asnumpy(cp.linalg.norm(R, axis=0) / cp.maximum(lam, 1e-300))
    U = U64.astype(dtype)
    s = cp.sqrt(lam).astype(dtype)
    return U, s, r


def residual_summary(r, tol=None) -> str:
    """One line on how far the basis is from exact eigenvectors."""
    tol = _RESIDUAL_TOL if tol is None else tol
    r = np.asarray(r, dtype=np.float64)
    if r.size == 0:
        return "no columns"
    worst = int(np.argmax(r))
    return (f"{int((r < tol).sum())}/{r.size} columns converged (residual < {tol:g}), "
            f"median {np.median(r):.1e}, worst {r[worst]:.1e} at column {worst}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def _mode_groups(modes, tied_groups) -> List[List[int]]:
    """Partition ``modes`` into tied groups plus singletons, ordered by first mode."""
    groups, seen = [], set()
    for g in tied_groups or []:
        g = sorted(int(m) for m in g if int(m) in modes)
        if g:
            groups.append(g)
            seen.update(g)
    groups += [[m] for m in modes if m not in seen]
    return sorted(groups, key=lambda g: g[0])


class _Progress:
    """tqdm bar over the Gram applications, advanced in fractions (chunks or
    cells) from the per-device worker threads; snapped to the application
    boundary when each application ends, so it always finishes at 100%."""

    def __init__(self, n_apps, desc, enabled=True):
        self._lock = threading.Lock()
        self._apps_done = 0
        self._bar = tqdm(total=n_apps, desc=desc, mininterval=2.0,
                         bar_format="{desc}: {percentage:3.0f}%|{bar}| {n:.2f}/{total} "
                                    "Gram applications [{elapsed}<{remaining}]") if enabled else None

    def add(self, x):
        if self._bar is None:
            return
        with self._lock:
            # never run ahead into the next application
            x = min(x, self._apps_done + 1 - self._bar.n)
            if x > 0:
                self._bar.update(x)

    def end_application(self):
        if self._bar is None:
            return
        with self._lock:
            self._apps_done += 1
            diff = self._apps_done - self._bar.n
            if diff > 0:
                self._bar.update(diff)
            else:
                self._bar.n = self._apps_done
                self._bar.refresh()

    def close(self):
        if self._bar is not None:
            self._bar.close()


def compute_group_basis(idxs, vals, shape, group, k, device_ids, home, pool,
                        seed, verbose=True, power_iter=None, oversampling=None):
    """(U, s, r) for one group of modes, on the home device
    (r: relative residual per column, see top_k_eig)."""
    q, o = _resolve_sketch(power_iter, oversampling)
    V = int(shape[group[0]])
    dtype = vals.dtype
    G = len(device_ids)
    p = min(min(int(k), V) + o, V)
    t0 = time.perf_counter()

    resident = None
    if _resident_fits(int(vals.size), len(group), len(shape), int(dtype.itemsize), device_ids):
        resident = []
        for mode in group:
            resident.append(_ModeLayout(idxs, vals, shape, mode, device_ids, home, pool))
            _free_pools(device_ids)     # drop the build's cached temporaries
        # Plan once all are in place, so chunk sizes see the memory that is left.
        for lay in resident:
            lay.plan(device_ids, p)
        if verbose:
            n_g = sum(len(gs) for lay in resident for gs in lay.groupings)
            print(f"[svd_stream] modes {group}: {n_g} resident cell groupings built in "
                  f"{time.perf_counter() - t0:.1f}s on {G} GPU(s)")
    elif verbose:
        print(f"[svd_stream] modes {group}: groupings don't fit together; "
              f"re-sorting them cell by cell in every Gram application")

    n_apps = q + 2
    progress = _Progress(n_apps, f"[svd_stream] modes {group} q{q} p{p}", enabled=verbose)
    if resident is not None:
        n_chunks = sum(len(g.chunks) for lay in resident for gs in lay.groupings for g in gs)

        def resident_tick(_n):
            progress.add(1.0 / max(n_chunks, 1))

    def apply(omega):
        omega_np = cp.asnumpy(omega) if G > 1 else None

        def local_omega(did):
            return omega if did == home else cp.asarray(omega_np)

        def finish(did, Yk):
            return Yk if did == home else cp.asnumpy(Yk)

        if resident is not None:
            def work(kd):
                did = device_ids[kd]
                with cp.cuda.Device(did):
                    gs = [g for lay in resident for g in lay.groupings[kd]]
                    return finish(did, gram_apply_local(gs, local_omega(did), V, on_chunk=resident_tick))
            parts = _run_on_devices(work, device_ids, pool)
        else:
            parts = []
            for mode in group:
                sources, other_dims, n_cells = _prepare_sources(idxs, vals, shape, mode, device_ids, home)
                # one cell's share of this application, split over its chunks
                cell_share = 1.0 / (len(group) * G * max(n_cells, 1))

                def cell_tick(n_chunks_in_cell, share=cell_share):
                    progress.add(share / max(n_chunks_in_cell, 1))

                def work(kd):
                    did = device_ids[kd]
                    with cp.cuda.Device(did):
                        src = _upload(sources[kd])
                        sources[kd] = None
                        om = local_omega(did)
                        Yk = cp.zeros((V, om.shape[1]), dtype=om.dtype)
                        for g in _iter_groupings(src, other_dims, V):
                            plan_chunks(g, p, int(_CHUNK_FRACTION * _gpu_free_bytes()))
                            Yk += gram_apply_local([g], om, V, on_chunk=cell_tick)
                            del g
                        del src
                        return finish(did, Yk)

                parts += _run_on_devices(work, device_ids, pool)
                del sources

        Y = cp.zeros((V, omega.shape[1]), dtype=omega.dtype)
        host = []
        for x in parts:
            if isinstance(x, np.ndarray):
                host.append(x)
            else:
                Y += x
        if host:
            Y += cp.asarray(sum(host))
        progress.end_application()
        return Y

    try:
        with cp.cuda.Device(home):
            U, s, r = top_k_eig(apply, V, k, dtype, seed, power_iter=q, oversampling=o)
    finally:
        progress.close()
    if verbose:
        print(f"[svd_stream] modes {group}: basis k={U.shape[1]} in "
              f"{time.perf_counter() - t0:.1f}s ({n_apps} Gram applications, p={p})")
        print(f"[svd_stream] modes {group}: {residual_summary(r)}")
    return U, s, r


def _free_pools(device_ids) -> None:
    for did in device_ids:
        with cp.cuda.Device(did):
            cp.get_default_memory_pool().free_all_blocks()
    cp.get_default_pinned_memory_pool().free_all_blocks()


def _forecast(cache, group, k, V, q, o) -> Optional[str]:
    """Rough time forecast from an earlier cached basis of this group (any setting):
    cost ∝ Gram applications × sketch width. Includes the grouping build, so it errs long."""
    for meta in cache.timings(group):
        old = meta.get("sketch", "")
        try:
            q_old, o_old = (int(x) for x in old.split("_")[0][1:].split("o"))
        except ValueError:
            continue
        if meta.get("seconds", 0) <= 0:
            continue
        p_new, p_old = min(k + o, V), min(int(meta["K"]) + o_old, V)
        est = meta["seconds"] * ((q + 2) * p_new) / ((q_old + 2) * p_old)
        return (f"expected ~{est / 60:.1f} min (from cached {old} run: {meta['seconds']:.0f} s; "
                f"rough, the GPU count may differ)")
    return None


def fill_zeros(F, fill):
    """NNDSVDa-style: the exact zeros of NNDSVD become ``fill`` x their column's
    mean, so MU can still grow them (a multiplicative update cannot grow a zero).
    Per column, so rank truncation stays exact. ``fill`` 0 = plain NNDSVD."""
    if not fill:
        return F
    return cp.where(F == 0, fill * F.mean(axis=0, keepdims=True), F)


def initialize_svd_stream(sparse_tensor, shape, rank, modes, random_state, with_core=True,
                          *, cfg=None, tied_groups=None, device_ids=None, use_cache=None,
                          power_iter=None, oversampling=None, fill=None, rotate=None,
                          verbose=True):
    """Tucker-style init ``(core | None, factors)`` from the streamed SVD.

    Keyword arguments override the run context from ``set_run_context``.
    Without either: per-mode bases on the tensor's GPU, no cache, default sketch,
    no fill. ``rotate`` (default: cfg.exp.svd_rotate) is EXPERIMENTAL, see
    experimental/svd_rotation.py.
    """
    ctx = _pop_run_context()
    cfg = cfg if cfg is not None else ctx.get("cfg")
    tied_groups = tied_groups if tied_groups is not None else ctx.get("tied_groups")
    device_ids = device_ids if device_ids is not None else ctx.get("device_ids")
    if use_cache is None:
        use_cache = ctx.get("use_cache", True)
    q, o = _resolve_sketch(power_iter if power_iter is not None else ctx.get("power_iter"),
                           oversampling if oversampling is not None else ctx.get("oversampling"))
    fill = float(fill if fill is not None else ctx.get("fill", 0.0) or 0.0)
    if fill < 0:
        raise ValueError(f"svd_stream: fill must be >= 0; got {fill}")
    rotation = None
    rotate = rotate or getattr(getattr(cfg, "exp", None), "svd_rotate", None) or "none"
    if rotate != "none":
        from tensormet.experimental import svd_rotation as rotation
        rotation.check(rotate, fill)

    shape = tuple(int(s) for s in shape)
    modes = [int(m) for m in modes]
    rank = [int(r) for r in rank]
    idxs, vals = coo_to_coords(sparse_tensor, shape)
    home = _array_device_id(vals)
    home = 0 if home is None else home
    device_ids = [int(d) for d in (device_ids or [home])]
    # home first: it hashes the cells and does the small algebra
    device_ids = [home] + [d for d in device_ids if d != home]

    groups = _mode_groups(modes, tied_groups)
    for g in groups:
        if len({shape[m] for m in g}) > 1 or len({rank[modes.index(m)] for m in g}) > 1:
            raise ValueError(f"svd_stream: tied modes {g} need one dim and one rank; "
                             f"got dims {[shape[m] for m in g]}, "
                             f"ranks {[rank[modes.index(m)] for m in g]}")

    with cp.cuda.Device(home):
        fp = fingerprint(int(vals.size), float(cp.sum(vals, dtype=cp.float64)), shape)
    # The seed drives the sketch, so each random_state gets its own cache files.
    seed_tag = f"{sketch_tag(q, o)}_rs{int(random_state or 0)}"
    cache = InitCache.from_config(cfg, seed_tag, fill) if (cfg is not None and use_cache) else None
    if rotation is not None and cache is not None:
        cache.core_tag = rotation.core_tag(rotate, rank)

    pool = ThreadPoolExecutor(max_workers=len(device_ids)) if len(device_ids) > 1 else None
    if pool is not None:
        _warm_up(device_ids, vals.dtype)
    factors: List = [None] * len(modes)
    token_by_mode = {}
    try:
        for g in groups:
            k = rank[modes.index(g[0])]
            hit = cache.load_basis(g, k, fp) if cache is not None else None
            with cp.cuda.Device(home):
                if hit is not None:
                    U_np, s_np, token, r = hit
                    U, s = cp.asarray(U_np, dtype=vals.dtype), cp.asarray(s_np, dtype=vals.dtype)
                    if verbose and r is not None:
                        print(f"[svd_stream] modes {g}: {residual_summary(r)}")
                else:
                    if verbose and cache is not None:
                        forecast = _forecast(cache, g, k, int(shape[g[0]]), q, o)
                        if forecast:
                            print(f"[svd_stream] modes {g} at {sketch_tag(q, o)}: {forecast}")
                    t0 = time.perf_counter()
                    U, s, r = compute_group_basis(idxs, vals, shape, g, k, device_ids, home, pool,
                                                  seed=int(random_state or 0) + g[0],
                                                  verbose=verbose, power_iter=q, oversampling=o)
                    token = (cache.save_basis(g, cp.asnumpy(U), cp.asnumpy(s), fp,
                                              time.perf_counter() - t0, residuals=r)
                             if cache is not None else None)
                if rotation is not None:
                    F = rotation.rotated_factor(U, s, rotate, f"modes {g}: ", verbose)
                else:
                    F = fill_zeros(_nndsvd_factors_gpu(U, s), fill)
                del U, s
            for m in g:
                factors[modes.index(m)] = F
                token_by_mode[m] = token
    finally:
        if pool is not None:
            pool.shutdown(wait=True)
        _free_pools(device_ids)

    with cp.cuda.Device(home):
        core = None
        if with_core:
            ranks = [int(f.shape[1]) for f in factors]
            tokens = [token_by_mode[m] for m in modes]
            if cache is not None and None not in tokens:
                hit = cache.load_core(groups, ranks, tokens, fp)
                if hit is not None:
                    core = cp.asarray(hit, dtype=vals.dtype)
            if core is None:
                t0 = time.perf_counter()
                core = _compute_tucker_core_batched(sparse_tensor, shape, factors, modes)
                core = cp.clip(cp.abs(core), a_min=1e-30, a_max=None)
                if cache is not None and None not in tokens:
                    cache.save_core(groups, cp.asnumpy(core), ranks, tokens, fp,
                                    time.perf_counter() - t0)
        # After the core, which is built from the unclipped factors.
        factors = [cp.clip(cp.abs(f), a_min=1e-30, a_max=None) for f in factors]
    return core, factors
