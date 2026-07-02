from math import log, prod
from collections import Counter, defaultdict
from pathlib import Path
import os
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
import multiprocessing
import torch
import numpy as np
from tqdm import tqdm
from tensormet.utils import (
    DATA_DIR, shared_factor_suffix, linked_factor_groups, SparseCOOTensor,
    _INT64_MAX, dim_spec_str, compute_num_threads, resolve_mem_budget_gb,
)
from tensormet.naming import ALL_METHODS, DEFAULT_METHODS
import pickle

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.dataset as ds

from itertools import combinations
from functools import reduce
import hashlib, json
import zipfile

_ALL_TENSORS = ALL_METHODS

# epsilon added after log(count) so that singletons (count == 1) map to a small
# non-zero value instead of log(1) == 0, keeping them in the sparse tensor.
_COUNT_LOG_EPS = 1e-8

# Memory-budgeting constants for the Pass-2 worker count, used to keep worker
# fan-out under an explicit ceiling (--max-mem-gb). Pass 2 is *merge-bound* (the
# parent merges partials single-threaded), so a modest worker count already
# saturates throughput; extra workers only pile partial counters in RAM. All of
# these are rough — override per run with --mem-per-worker-gb if the estimate is
# off for your data, and watch the real peak with `sstat -j <id> --format=MaxRSS`.
#
# Per-worker memory is estimated from ROWS (not compressed parquet bytes, which
# under-counts Python-object expansion by ~20x): each worker holds ~one entry per
# sub-counter per row, so per_worker ≈ rows_per_task × n_subcounters × bytes/entry.
_BYTES_PER_ENTRY = 200.0  # in-memory bytes per Counter entry (tuple key + int)
_COUNTER_BLOWUP = 10.0    # parquet bytes -> in-memory bytes, for the global reserve
_IN_FLIGHT_FACTOR = 2.0   # covers the n_workers + 4 in-flight result buffer
# Pass 2 is merge-bound; capping auto workers avoids piling partials for no speedup.
_PASS2_AUTO_WORKER_CAP = 12
# Minimum joint rows per post-processing chunk (avoid tiny parallel chunks).
_PP_MIN_CHUNK = 200_000


def _torch_save_atomic(obj, path: str) -> None:
    """torch.save into a sibling .tmp, then os.replace into place.

    A save that dies mid-write (scratch quota, Lustre I/O error, walltime kill)
    then never leaves a truncated file at the canonical path, so the resume
    check in populate_tensors_parquet can trust file existence.
    """
    tmp = f"{path}.tmp"
    try:
        torch.save(obj, tmp)
        os.replace(tmp, path)  # atomic same-fs rename
    except BaseException:
        try:
            os.remove(tmp)
        except OSError:
            pass
        raise


def _is_complete_pt(path: str) -> bool:
    """True if ``path`` is a fully written torch.save zip archive.

    The end-of-central-directory record is the last thing torch flushes, so a
    save that died mid-write leaves a file that fails to open as a zip. Opening
    reads only that trailing directory — cheap even for multi-GB tensors.
    Guards resumes against truncated files written before saves were atomic.
    """
    try:
        with zipfile.ZipFile(path):
            return True
    except (OSError, zipfile.BadZipFile):
        return False


# ── new top-level worker (must be picklable → module level) ─────
def _pass2_worker(
    shard_paths: list[str],
    cols_to_build: list[str],
    vocabs_max: dict[str, list],   # col → list of strings (picklable)
    batch_rows: int,
    batch_readahead: int,
    fragment_readahead: int,
) -> dict:
    """
    Process a subset of parquet shards and return partial subset_counters.
    Runs in a child process – no shared state with the parent.
    """
    import pyarrow as pa
    import pyarrow.compute as pc
    import pyarrow.dataset as ds
    from collections import Counter
    from itertools import combinations
    from functools import reduce

    # Parallelism here is across processes (one per shard task); pin each worker to
    # a single Arrow thread so N workers ≈ N cores instead of N × all-core pools
    # oversubscribing the node.
    pa.set_cpu_count(1)
    pa.set_io_thread_count(1)

    # Reconstruct Arrow membership arrays locally (not picklable across processes)
    max_arrs = {col: pa.array(vocabs_max[col]) for col in cols_to_build}

    subset_counters = {
        subset: Counter()
        for r in range(2, len(cols_to_build) + 1)
        for subset in combinations(cols_to_build, r)
    }

    dataset = ds.dataset(shard_paths, format="parquet")
    batches = dataset.to_batches(
        columns=cols_to_build,
        batch_size=batch_rows,
        batch_readahead=batch_readahead,
        fragment_readahead=fragment_readahead,
        use_threads=True,
        cache_metadata=True,
    )

    for batch in batches:
        t = pa.table({
            col: _normalize_str_array(batch.column(i))
            for i, col in enumerate(cols_to_build)
        })
        masks = {col: pc.is_in(t[col], value_set=max_arrs[col]) for col in cols_to_build}

        for r in range(2, len(cols_to_build) + 1):
            for subset in combinations(cols_to_build, r):
                subset_mask = reduce(pc.and_, (masks[col] for col in subset))
                t_subset = t.filter(subset_mask)
                if t_subset.num_rows:
                    g_subset = (
                        t_subset.group_by(list(subset))
                        .aggregate([(subset[0], "count")])
                        .rename_columns(list(subset) + ["count"])
                    )
                    _update_counter_from_grouped(
                        subset_counters[subset], g_subset, list(subset), "count"
                    )

    return subset_counters  # Counter is picklable → returned to parent via pickle


def _soft_knee_compress(
    vals: torch.Tensor,
    knee_quantile: float = 0.80,
    scale: "float | None" = None,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Monotonic upper-tail compressor for flattening heavy right tails.

    Values at or below the knee ``tau`` (a high quantile) pass through
    unchanged; values above it are log-compressed, so only the extreme high
    end is flattened while the bulk -- and the relative order of every entry --
    is preserved. Output stays nonnegative for nonnegative input, and the
    transform is C^1-continuous at the knee (slope 1 on both sides), so the
    sorted distribution has no kink.

        f(x) = x                              if x <= tau
        f(x) = tau + s * log1p((x - tau) / s) if x >  tau

    Args:
        vals: 1-D tensor of (nonnegative) values, e.g. ``softplus(sc)``.
        knee_quantile: quantile of ``vals`` used as the knee ``tau``.
        scale: compression scale ``s``. Smaller ``s`` compresses harder. When
            ``None``, defaults to the standard deviation of the sub-knee bulk
            so the tail bends on the bulk's own scale.
        eps: floor keeping ``scale`` strictly positive.

    Returns:
        Tensor of the same shape/dtype as ``vals``.
    """
    if vals.numel() == 0:
        return vals
    # kthvalue rather than torch.quantile: the latter caps its input at ~16M
    # elements, but nnz here can be 50M+ (4-gram tensors).
    n = vals.numel()
    k = min(max(int(knee_quantile * (n - 1)) + 1, 1), n)
    tau = torch.kthvalue(vals, k).values.to(vals.dtype)
    if scale is None:
        bulk = vals[vals <= tau]
        ref = bulk if bulk.numel() > 1 else vals
        scale = ref.std()
    scale = torch.as_tensor(scale, dtype=vals.dtype, device=vals.device).clamp_min(eps)
    out = vals.clone()
    above = vals > tau
    out[above] = tau + scale * torch.log1p((vals[above] - tau) / scale)
    return out


def _make_sparse_coo(idx: torch.Tensor, values: torch.Tensor, size: tuple):
    """
    Create a sparse COO tensor, falling back to SparseCOOTensor when
    prod(size) would overflow int64 (e.g. 5-gram with top_k >= ~9500).
    """
    numel = 1
    for d in size:
        numel *= d
        if numel > _INT64_MAX:
            return SparseCOOTensor(idx, values, size)
    return torch.sparse_coo_tensor(idx, values, size=size)


# ---------------------------------------------------------------------------
# Post-Pass-2 value computation: COW-safe NumPy arrays + optional fork parallel
# ---------------------------------------------------------------------------
#
# After Pass 2 the per-variant tensor values (counting / probLog / sii / sc / ...)
# are computed from the joint + sub counters. Doing this as a per-row Python loop
# is single-core and dominates wall time at scale. We instead convert the counters
# to NumPy arrays ONCE and compute every channel vectorised; the work is then split
# across a *fork* process pool. Crucially the shared state holds ONLY NumPy arrays
# (and plain scalars) so copy-on-write is not defeated by CPython refcounting —
# which is exactly what would happen if forked workers read the giant Counter/dict
# objects, multiplying memory by the worker count.

# Read-only state inherited by forked workers (set in the parent before the pool
# is created). Holds NumPy arrays only.
_PP_STATE: dict = {}


def _encode_keys(a: np.ndarray, base: int) -> np.ndarray:
    """Horner-encode rows of integer ranks into one int64 key per row.

    Matches the build-time encoding so ``np.searchsorted`` lookups are exact.
    """
    a = a.astype(np.int64, copy=False)
    key = np.zeros(a.shape[0], dtype=np.int64)
    for j in range(a.shape[1]):
        key = key * base + a[:, j]
    return key


def _encoding_fits(base: int, order: int) -> bool:
    """True when composite keys for subsets up to size ``order-1`` fit in int64."""
    if order < 2:
        return False
    return base ** (order - 1) < (1 << 62)


def _sii_range(st: dict, sel: np.ndarray, cnt_f: np.ndarray) -> np.ndarray:
    """Vectorised specific interaction information for a block of joint rows.

    Replicates the term order of the scalar reference (inclusion-exclusion: even
    subset orders in the numerator, odd in the denominator) so results match the
    per-row path; rows with any non-positive factor map to the -1e38 sentinel.
    """
    order = st["order"]
    total_len = st["total_len"]
    base = st["base"]
    m = sel.shape[0]
    num = np.ones(m, dtype=np.float64)
    den = np.ones(m, dtype=np.float64)
    nonpos = np.zeros(m, dtype=bool)
    for r in range(1, order + 1):
        for pos in combinations(range(order), r):
            if r == 1:
                p = st["marg"][pos[0]][sel[:, pos[0]]]
            elif r == order:
                p = cnt_f / total_len
            else:
                key = _encode_keys(sel[:, list(pos)], base)
                ip = np.searchsorted(st["sub_keys"][pos], key)
                p = st["sub_counts"][pos][ip].astype(np.float64) / total_len
            nonpos |= p <= 0
            if r % 2 == 0:
                num = num * p
            else:
                den = den * p
    with np.errstate(divide="ignore", invalid="ignore"):
        res = np.log(num / den)
    res[nonpos] = -1e38
    return res


def _sc_range(st: dict, sel: np.ndarray, cnt_f: np.ndarray) -> np.ndarray:
    """Vectorised specific correlation for a block of joint rows."""
    order = st["order"]
    total_len = st["total_len"]
    joint = cnt_f / total_len
    nonpos = joint <= 0
    prod_marg = np.ones(sel.shape[0], dtype=np.float64)
    for i in range(order):
        p = st["marg"][i][sel[:, i]]
        nonpos |= p <= 0
        prod_marg = prod_marg * p
    with np.errstate(divide="ignore", invalid="ignore"):
        res = np.log(joint / prod_marg)
    res[nonpos] = -1e38
    return res


def _fill_range(st: dict, start: int, end: int, variant) -> tuple:
    """Compute index rows + value channels for joint rows [start:end) of a variant.

    Returns ``(out_idx, vals)`` where out_idx is (m, order) int64 and vals maps
    channel name -> (m,) float32, for the rows that pass the variant's vocab cut.
    """
    idx = st["idx_max"][start:end]
    var = np.asarray(variant, dtype=idx.dtype)
    mask = (idx < var).all(axis=1)
    sel = idx[mask]
    cnt = st["counts"][start:end][mask]
    cnt_f = cnt.astype(np.float64)
    need = st["need"]
    total_len = st["total_len"]

    out_idx = sel.astype(np.int64, copy=False)
    vals: dict = {}
    if need["count"]:
        vals["count"] = cnt.astype(np.float32)
    if need["prob_log"]:
        vals["prob_log"] = np.log(cnt_f / total_len).astype(np.float32)
    if need["count_log"]:
        vals["count_log"] = np.log(cnt_f).astype(np.float32)
    if need["count_log_eps"]:
        vals["count_log_eps"] = (np.log(cnt_f) + _COUNT_LOG_EPS).astype(np.float32)
    if need["sii"]:
        vals["sii"] = _sii_range(st, sel, cnt_f).astype(np.float32)
    if need["sc"]:
        vals["sc"] = _sc_range(st, sel, cnt_f).astype(np.float32)
    return out_idx, vals


def _pp_worker(start: int, end: int, variant) -> tuple:
    """Fork-pool entry point: reads the inherited _PP_STATE, one thread per process."""
    from threadpoolctl import threadpool_limits
    with threadpool_limits(1):
        return _fill_range(_PP_STATE, start, end, variant)


def _build_pp_arrays(subset_counters, single_probs, vocabs_max, ranks,
                     cols, order, total_len, base, need) -> dict:
    """Convert the joint + sub counters into COW-safe NumPy arrays.

    Mutates ``subset_counters`` (deletes each converted Counter, then clears it) to
    reclaim the large dict memory before the heavy compute starts.
    """
    full_cols = tuple(cols)
    full_ctr = subset_counters[full_cols]
    n_rows = len(full_ctr)
    idx_max = np.empty((n_rows, order), dtype=np.int32)
    counts = np.empty(n_rows, dtype=np.int64)
    col_ranks = [ranks[c] for c in cols]
    for row, (ktuple, c) in enumerate(full_ctr.items()):
        for j in range(order):
            idx_max[row, j] = col_ranks[j][ktuple[j]]
        counts[row] = c
    del subset_counters[full_cols]

    # marginal probability per rank, per mode (single_probs already normalised)
    marg = [
        np.array(
            [single_probs[cols[i]].get(tok, 0.0) for tok in vocabs_max[cols[i]]],
            dtype=np.float64,
        )
        for i in range(order)
    ]

    sub_keys: dict = {}
    sub_counts: dict = {}
    if need["sii"]:
        for r in range(2, order):  # subset sizes 2 .. order-1
            for pos in combinations(range(order), r):
                sub_cols = tuple(cols[p] for p in pos)
                ctr = subset_counters[sub_cols]
                k = np.empty((len(ctr), r), dtype=np.int32)
                v = np.empty(len(ctr), dtype=np.int64)
                pr = [ranks[cols[p]] for p in pos]
                for ii, (kt, cc) in enumerate(ctr.items()):
                    for j in range(r):
                        k[ii, j] = pr[j][kt[j]]
                    v[ii] = cc
                enc = _encode_keys(k, base)
                order_idx = np.argsort(enc, kind="stable")
                sub_keys[pos] = enc[order_idx]
                sub_counts[pos] = v[order_idx]
                del subset_counters[sub_cols]

    subset_counters.clear()  # everything needed now lives in NumPy arrays

    return {
        "idx_max": idx_max,
        "counts": counts,
        "marg": marg,
        "sub_keys": sub_keys,
        "sub_counts": sub_counts,
        "total_len": total_len,
        "order": order,
        "base": base,
        "need": need,
        "N": n_rows,
    }


# ---------------------------------------------------------------------------
# Pass-2 checkpoint: persist the post-Pass-2 _PP_STATE arrays so a re-run skips
# the (very expensive, multi-hour) joint-count pass. Keyed on everything that
# determines the arrays' contents: source vectors, columns, per-mode max_ks,
# factor linking, and whether sii sub-counters were materialised.
# ---------------------------------------------------------------------------
def _pp_state_cache_path(path_to_tensors, path_to_vectors, cols_to_build,
                         max_ks, shared_factors, need_sii, remove_hapax,
                         min_mode_ks, ensured_vocab):
    key = json.dumps(
        {
            "src": os.fspath(path_to_vectors),
            "cols": list(cols_to_build),
            "max_ks": list(max_ks),
            "shared_factors": shared_factors,
            "sii": bool(need_sii),
            "remove_hapax": bool(remove_hapax),
            "min_mode_ks": sorted((min_mode_ks or {}).items()),
            "ensured_vocab": sorted(ensured_vocab) if ensured_vocab else None,
        },
        sort_keys=True,
    )
    h = hashlib.sha1(key.encode()).hexdigest()[:12]
    cache_dir = Path(path_to_tensors) / "pp_state"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir / f"pp_state_{'-'.join(cols_to_build)}_{h}.npz"


def _save_pp_state_cache(cache_path: Path, st: dict) -> None:
    """Atomically persist the _PP_STATE NumPy arrays to a single .npz file.

    The big arrays (idx_max, counts, marg_*, subkeys_*/subcounts_*) are stored as
    named arrays; the small structural metadata is pickled into a uint8 array so the
    whole checkpoint is one file written then ``os.replace``'d atomically.
    """
    arrays = {"idx_max": st["idx_max"], "counts": st["counts"]}
    for i, m in enumerate(st["marg"]):
        arrays[f"marg_{i}"] = m
    sub_pos = list(st["sub_keys"].keys())
    for j, pos in enumerate(sub_pos):
        arrays[f"subkeys_{j}"] = st["sub_keys"][pos]
        arrays[f"subcounts_{j}"] = st["sub_counts"][pos]
    meta = {
        "version": 1,
        "total_len": st["total_len"],
        "order": st["order"],
        "base": st["base"],
        "n_marg": len(st["marg"]),
        "sub_pos": [list(p) for p in sub_pos],
        "N": st["N"],
    }
    arrays["_meta"] = np.frombuffer(
        pickle.dumps(meta, protocol=pickle.HIGHEST_PROTOCOL), dtype=np.uint8
    )
    tmp = cache_path.with_name(cache_path.name + ".tmp")
    with open(tmp, "wb") as f:
        np.savez(f, **arrays)
    os.replace(tmp, cache_path)  # atomic


def _load_pp_state_cache(cache_path: Path, need: dict) -> dict:
    """Reconstruct _PP_STATE from a checkpoint written by ``_save_pp_state_cache``.

    ``need`` is the *current* run's channel selection and overrides whatever was
    cached (the cache key already pins the only need-dependent part, sii).
    """
    with np.load(cache_path, allow_pickle=False) as npz:
        meta = pickle.loads(npz["_meta"].tobytes())
        idx_max = npz["idx_max"]
        counts = npz["counts"]
        marg = [npz[f"marg_{i}"] for i in range(meta["n_marg"])]
        sub_keys, sub_counts = {}, {}
        for j, pos in enumerate(meta["sub_pos"]):
            tpos = tuple(pos)
            sub_keys[tpos] = npz[f"subkeys_{j}"]
            sub_counts[tpos] = npz[f"subcounts_{j}"]
    return {
        "idx_max": idx_max,
        "counts": counts,
        "marg": marg,
        "sub_keys": sub_keys,
        "sub_counts": sub_counts,
        "total_len": meta["total_len"],
        "order": meta["order"],
        "base": meta["base"],
        "need": need,
        "N": meta["N"],
    }


def _compute_subset_counters(
    parquet_files, total_rows, cols_to_build, vocabs_max, max_ks,
    batch_rows, batch_readahead, fragment_readahead,
    max_workers, shards_per_task, cpu_frac, max_mem_gb, mem_per_worker_gb,
):
    """PASS 2: restricted joint counts, parallelised across shards.

    Returns ``subset_counters`` ({subset_tuple: Counter} for all subsets of size
    >= 2). Extracted from the populator so a cache hit can bypass it entirely.
    """
    import time as _time

    subset_counters = {
        subset: Counter()
        for r in range(2, len(cols_to_build) + 1)
        for subset in combinations(cols_to_build, r)
    }

    shards_per_task = max(1, shards_per_task)

    # Worker count: an explicit --max-workers wins; otherwise scale to the node's
    # cores (cpu_frac) but cap by an explicit memory ceiling so we never recreate
    # the swapping the old shards//100 heuristic was guarding against. The parent's
    # global accumulator grows regardless of worker count; each extra worker adds
    # its own partial counters + in-flight result buffer, which is what must fit.
    if max_workers and max_workers > 0:
        n_workers = min(max_workers, len(parquet_files))
        print(f"using {n_workers} workers (explicit --max-workers, "
              f"{shards_per_task} shard(s)/task)")
    else:
        cpu_budget = compute_num_threads(cpu_frac)
        total_parquet_bytes = sum(p.stat().st_size for p in parquet_files)

        # Per-worker estimate from ROWS: each worker holds ~one entry per sub-counter
        # per row of its task. (Compressed parquet bytes under-count this ~20x.)
        n_subcounters = max(1, (2 ** len(cols_to_build)) - 1 - len(cols_to_build))
        rows_per_task = (total_rows / max(1, len(parquet_files))) * shards_per_task
        if mem_per_worker_gb and mem_per_worker_gb > 0:
            per_worker_gb = float(mem_per_worker_gb)
        else:
            per_worker_gb = rows_per_task * n_subcounters * _BYTES_PER_ENTRY / 1e9

        mem_budget_gb, mem_src = resolve_mem_budget_gb(max_mem_gb)
        global_reserve_gb = total_parquet_bytes * _COUNTER_BLOWUP / 1e9
        headroom_gb = mem_budget_gb - global_reserve_gb
        denom = max(per_worker_gb * _IN_FLIGHT_FACTOR, 1e-9)
        mem_budget_workers = max(1, int(headroom_gb // denom))

        # Pass 2 is merge-bound: a modest worker count saturates the single-threaded
        # parent merge, so cap auto fan-out (more workers only pile partials in RAM).
        n_workers = min(cpu_budget, mem_budget_workers, _PASS2_AUTO_WORKER_CAP,
                        len(parquet_files))
        print(
            f"using {n_workers} workers "
            f"(cpu_budget={cpu_budget} @ cpu_frac={cpu_frac}, "
            f"mem_budget={mem_budget_workers} [ceiling={mem_budget_gb:.1f}GB src={mem_src}, "
            f"global_reserve~{global_reserve_gb:.1f}GB, per_worker~{per_worker_gb:.2f}GB], "
            f"merge_cap={_PASS2_AUTO_WORKER_CAP}, {shards_per_task} shard(s)/task)"
        )
        if headroom_gb <= 0:
            print(
                f"WARNING: estimated global accumulator (~{global_reserve_gb:.1f}GB) exceeds the "
                f"memory ceiling ({mem_budget_gb:.1f}GB); falling back to 1 worker. Pass 2 itself "
                f"may not fit in RAM — request more memory (--mem) or reduce vocab/top-ks."
            )
    n_workers = max(1, n_workers)
    all_shard_paths = [os.fspath(p) for p in parquet_files]
    task_batches = [
        all_shard_paths[i : i + shards_per_task]
        for i in range(0, len(all_shard_paths), shards_per_task)
    ]
    n_tasks = len(task_batches)

    # vocabs_max values are plain lists → picklable
    vocabs_max_plain = {col: list(vocabs_max[col]) for col in cols_to_build}

    print(f"Pass 2/2: computing joint counts restricted to max_ks={max_ks} vocab "
          f"[{n_workers} workers, {n_tasks} tasks, {len(parquet_files)} shards] ...")

    # Only keep n_workers + small buffer tasks in-flight at once.
    # Submitting all tasks upfront causes completed results to pile up as pickled
    # bytes faster than the main process can merge them, flooding memory.
    max_in_flight = n_workers + 4

    t_pass2_start = _time.perf_counter()

    with ProcessPoolExecutor(max_workers=n_workers) as pool:
        task_iter = iter(task_batches)
        pending: dict = {}

        def _submit_one():
            try:
                batch = next(task_iter)
                fut = pool.submit(
                    _pass2_worker, batch, cols_to_build, vocabs_max_plain,
                    batch_rows, batch_readahead, fragment_readahead,
                )
                pending[fut] = None
            except StopIteration:
                pass

        for _ in range(min(max_in_flight, n_tasks)):
            _submit_one()

        n_done = 0
        with tqdm(total=n_tasks, desc="Pass 2/2", unit="task") as pbar:
            while pending:
                done_futs, _ = wait(list(pending.keys()), return_when=FIRST_COMPLETED)
                for fut in done_futs:
                    partial = fut.result()
                    del pending[fut]
                    for subset, counter in partial.items():
                        subset_counters[subset].update(counter)
                    n_done += 1
                    elapsed = _time.perf_counter() - t_pass2_start
                    rate = n_done / elapsed
                    eta = (n_tasks - n_done) / rate if rate > 0 else float("inf")
                    pbar.set_postfix(
                        rate=f"{rate:.1f} t/s",
                        eta=f"{eta/60:.1f} min",
                        in_flight=len(pending),
                        refresh=False,
                    )
                    pbar.update()
                    _submit_one()

    return subset_counters


# -- parquet helpers --
def _normalize_str_array(arr: pa.Array) -> pa.Array:
    """
    Match original normalization for wrong elements that persist:
      el = el or "~"
    We map NULL or "" -> "~".
    """
    # Ensure it's a string array (Parquet should already be string)
    is_null = pc.is_null(arr)
    is_empty = pc.equal(arr, "")
    mask = pc.or_(is_null, is_empty)
    return pc.if_else(mask, pa.scalar("~"), arr)



def _marginals_cache_path(path_to_tensors, path_to_vectors, cols_to_build):
    key = json.dumps(
        {"src": os.fspath(path_to_vectors), "cols": list(cols_to_build)},
        sort_keys=True,
    )
    h = hashlib.sha1(key.encode()).hexdigest()[:12]
    cache_dir = Path(path_to_tensors) / "marginals"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir / f"marginals_{'-'.join(cols_to_build)}_{h}.pkl"

def _load_marginals_cache(cache_path, cols_to_build):
    with open(cache_path, "rb") as f:
        data = pickle.load(f)
    assert data["version"] == 1
    assert data["cols_to_build"] == list(cols_to_build)
    single_probs = {col: Counter(data["single_probs"][col]) for col in cols_to_build}
    return single_probs, data["total_len"]

def _save_marginals_cache(cache_path, single_probs, total_len,
                          cols_to_build, path_to_vectors):
    tmp = cache_path.with_suffix(cache_path.suffix + ".tmp")
    with open(tmp, "wb") as f:
        pickle.dump({
            "version": 1,
            "single_probs": {col: dict(single_probs[col]) for col in cols_to_build},
            "total_len": total_len,
            "cols_to_build": list(cols_to_build),
            "src": os.fspath(path_to_vectors),
        }, f, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(tmp, cache_path)  # atomic

def _update_counter_from_grouped(counter: Counter, grouped: pa.Table, key_cols: list[str], count_col: str) -> None:
    """
    grouped: columns key_cols + [count_col]
    Updates `counter` with counts.
    """
    # Convert small grouped results to Python once per batch
    cols = [grouped[c].to_pylist() for c in key_cols]
    counts = grouped[count_col].to_pylist()

    if len(key_cols) == 1:
        keys = cols[0]
        for k, c in zip(keys, counts):
            counter[k] += int(c)
    else:
        # Dynamically handle 2, 3, 4, ... N columns
        for row in zip(*cols, counts):
            *keys, c = row
            counter[tuple(keys)] += int(c)

def _hapax_report_and_filter(subset_counters: dict) -> dict:
    """
    Remove all entries with count == 1 from every subset counter.
    Prints a before/after comparison and returns the filtered counters.
    """
    print("\n── Hapax removal ──────────────────────────────────────────")

    rows = []
    filtered = {}
    for subset, counter in subset_counters.items():
        name = "(" + ", ".join(subset) + ")"
        n_types_before = len(counter)
        n_tokens_before = sum(counter.values())
        hapax = {k for k, v in counter.items() if v == 1}
        n_hapax = len(hapax)
        new_counter = Counter({k: v for k, v in counter.items() if v > 1})
        n_types_after = len(new_counter)
        n_tokens_after = sum(new_counter.values())
        filtered[subset] = new_counter
        rows.append({
            "subset": name,
            "types_before": n_types_before,
            "types_after": n_types_after,
            "hapax_removed": n_hapax,
            "pct_types_removed": 100 * n_hapax / n_types_before if n_types_before else 0.0,
            "tokens_before": n_tokens_before,
            "tokens_after": n_tokens_after,
            "tokens_removed": n_tokens_before - n_tokens_after,
            "pct_tokens_removed": 100 * (n_tokens_before - n_tokens_after) / n_tokens_before if n_tokens_before else 0.0,
        })

    col_w = max(len(r["subset"]) for r in rows)
    header = (f"{'Subset':<{col_w}}  {'Types before':>14}  {'Types after':>12}"
              f"  {'Hapax rmvd':>11}  {'% types':>8}"
              f"  {'Tokens before':>14}  {'Tokens after':>13}  {'% tokens':>9}")
    print(header)
    print("-" * len(header))
    for r in rows:
        print(
            f"{r['subset']:<{col_w}}  {r['types_before']:>14,}  {r['types_after']:>12,}"
            f"  {r['hapax_removed']:>11,}  {r['pct_types_removed']:>7.1f}%"
            f"  {r['tokens_before']:>14,}  {r['tokens_after']:>13,}  {r['pct_tokens_removed']:>8.1f}%"
        )

    total_hapax = sum(r["hapax_removed"] for r in rows)
    most_affected = max(rows, key=lambda r: r["pct_types_removed"])
    print(f"\nTotal hapax co-occurrences removed: {total_hapax:,}")
    print(f"Most affected subset: {most_affected['subset']} "
          f"({most_affected['pct_types_removed']:.1f}% of types removed)")
    print("───────────────────────────────────────────────────────────\n")

    return filtered

def _most_common_keys(counter: Counter, k: int) -> list:
    # Counter.most_common is deterministic enough for our use;
    # ties will follow internal ordering—same behavior as original Counter use.
    return [x for (x, _) in counter.most_common(k)]


def _shared_topk_hmean(counters: list[Counter], k: int, eps: float = 0.0,
                       min_ks: list[int] | None = None,
                       ensured_vocab: list[str] | None = None) -> list:
    """
    Build a shared top-k vocabulary across multiple marginals using
    generalized harmonic mean.

    counters:      list of Counter objects containing counts or probabilities
    min_ks:        optional per-counter floor. min_ks[i] items from counters[i]
                   are guaranteed to be included regardless of their cross-counter
                   harmonic mean score. Useful when one mode (e.g. verbs) would
                   otherwise score 0 because it has no coverage in the other modes.
    ensured_vocab: optional list of token strings that are pinned into the shared
                   vocabulary by name, regardless of their harmonic mean score.
                   Tokens that do not appear in any counter are silently skipped.
                   Useful for special tokens like <BOS>/<EOS> that only exist in
                   one mode and would otherwise be zeroed out by the harmonic mean.
    Returns: list of at most k keys.
    """
    if not counters:
        return []

    n = len(counters)
    if min_ks is None:
        min_ks = [0] * n

    # Step 1a: guarantee the top min_ks[i] items from each counter
    guaranteed: set = set()
    for counter, min_k in zip(counters, min_ks):
        if min_k > 0:
            for x, _ in counter.most_common(min_k):
                guaranteed.add(x)

    # Step 1b: pin explicitly named tokens (ensured_vocab)
    if ensured_vocab:
        all_seen: set = set()
        for counter in counters:
            all_seen.update(counter.keys())
        for tok in ensured_vocab:
            if tok in all_seen:
                guaranteed.add(tok)
            else:
                print(f"  [ensured_vocab] WARNING: token {tok!r} not found in any marginal counter — skipping.")

    # Step 2: score all remaining keys by harmonic mean, fill up to k
    all_keys: set = set()
    for counter in counters:
        all_keys.update(counter.keys())

    remaining_keys = all_keys - guaranteed
    remaining_k = k - len(guaranteed)

    scored = []
    for x in remaining_keys:
        vals = [float(counter.get(x, 0.0)) for counter in counters]
        if any(v == 0.0 for v in vals):
            hm = 0.0
        else:
            hm = n / sum((1.0 / (v + eps)) for v in vals)
        scored.append((hm, sum(vals), x))

    scored.sort(reverse=True)
    additional = [x for (_, __, x) in scored[:max(0, remaining_k)]]

    guaranteed_sorted = sorted(
        guaranteed,
        key=lambda x: (-sum(c.get(x, 0) for c in counters), x)
    )
    if len(guaranteed_sorted) > k:
        print(f"  [_shared_topk_hmean] WARNING: min_ks/ensured_vocab guaranteed "
              f"{len(guaranteed_sorted)} keys, exceeding k={k} — truncating to the "
              f"top {k} guaranteed keys by count so the returned vocabulary stays at "
              f"most k, per this function's contract.")
        guaranteed_sorted = guaranteed_sorted[:k]
    return guaranteed_sorted + additional



def populate_tensors_parquet(
    path_to_vectors,
    top_ks,
    save: bool = True,
    path_to_tensors=None,
    cols_to_build : list = ["root", "nsubj", "obj"],
    shared_factors=None,
    batch_rows: int = 256_000,
    batch_readahead: int = 4,
    fragment_readahead: int = 2,
    remove_hapax: bool = False,
    top_ks_asymmetric=None,
    min_mode_ks: dict[int, int] | None = None,
    max_workers: int = 0,
    shards_per_task: int = 1,
    ensured_vocab: list[str] | None = None,
    tensors_to_build: list[str] | None = None,
    cpu_frac: float = 0.5,
    max_mem_gb: float | None = None,
    mem_per_worker_gb: float | None = None,
):
    if tensors_to_build is not None:
        unknown = [t for t in tensors_to_build if t not in _ALL_TENSORS]
        if unknown:
            raise ValueError(f"Unknown tensor names: {unknown}. Valid: {_ALL_TENSORS}")
        want = list(tensors_to_build)
    else:
        want = list(DEFAULT_METHODS)

    need_count     = "counting"    in want
    need_prob_log  = any(t in want for t in ("probLog", "probLogSoftPlus", "probLogShifted"))
    need_count_log = "countingLog" in want          # pure log(count); log(1)=0 kept as-is
    need_count_log_eps = "countingLogEps" in want
    need_sii       = any(t in want for t in ("sii", "siiSoftPlus", "siiShifted"))
    need_sc        = any(t in want for t in ("sc",  "scSoftPlus",  "scShifted", "scSoftPlusFlat"))

    path_to_vectors = os.fspath(path_to_vectors)
    n_modes = len(cols_to_build)

    # Normalise both sources into a deduplicated list of per-mode tuples.
    # Uniform entries (int) become (k,)*n_modes; asymmetric entries must have len == n_modes.
    if not isinstance(top_ks, list):
        top_ks = [top_ks]
    variants: list[tuple] = [(k,) * n_modes for k in top_ks]
    for tup in (top_ks_asymmetric or []):
        tup = tuple(tup)
        if len(tup) != n_modes:
            raise ValueError(
                f"top_ks_asymmetric entry {tup} has {len(tup)} elements but "
                f"cols_to_build has {n_modes}."
            )
        variants.append(tup)
    variants = sorted(set(variants))

    # Per-mode maximum needed across all variants (drives pass-2 vocab size)
    max_ks = tuple(max(v[i] for v in variants) for i in range(n_modes))

    print(f"Populating tensors for variants={variants} from {path_to_vectors}...")

    if not path_to_tensors:
        base = os.path.basename(os.path.normpath(path_to_vectors))
        dataset_name = base.split("_")[0] + "_sparse"
        path_to_tensors = DATA_DIR / f"tensors/{dataset_name}/"

    os.makedirs(path_to_tensors, exist_ok=True)
    os.makedirs(f"{path_to_tensors}/populated", exist_ok=True)
    os.makedirs(f"{path_to_tensors}/vocabularies", exist_ok=True)
    print(f"Tensors will be saved to {path_to_tensors}")

    vector_dir = Path(path_to_vectors)
    parquet_files = sorted(vector_dir.glob("*.parquet"))

    if not parquet_files:
        raise FileNotFoundError(f"No parquet shards found in {vector_dir}")

    dataset = ds.dataset(parquet_files, format="parquet")

    # dataset = ds.dataset(path_to_vectors, format="parquet")
    total_rows = dataset.count_rows(use_threads=True, cache_metadata=True)
    print(f"Total rows: {total_rows:,} | Shards: {len(parquet_files)} | batch_rows={batch_rows:,}")


    # -------------------------
    # PASS 1: marginals only
    # -------------------------

    # single_probs = {column:Counter() for column in cols_to_build}
    #
    # batches1 = dataset.to_batches(
    #     columns=cols_to_build,
    #     batch_size=batch_rows,
    #     batch_readahead=batch_readahead,
    #     fragment_readahead=fragment_readahead,
    #     use_threads=True,
    #     cache_metadata=True,
    # )
    #
    # print("Pass 1/2: computing global marginals (v,s,o) ...")
    # seen_rows = 0
    # with tqdm(total=total_rows, desc="Pass 1/2", unit="rows") as pbar:
    #     for batch in batches1:
    #         pbar.update(batch.num_rows)
    #         seen_rows += batch.num_rows
    #
    #         t = pa.table({col:_normalize_str_array(batch.column(i)) for i, col in enumerate(cols_to_build)})
    #         for col in cols_to_build:
    #             g_col = t.group_by([col]).aggregate([(col, "count")]).rename_columns([col, "count"])
    #             _update_counter_from_grouped(single_probs[col], g_col, [col], "count")
    #
    # if seen_rows == 0:
    #     raise ValueError("No rows found in the parquet dataset.")
    # total_len = seen_rows  # global denominator for probabilities

    cache_path = _marginals_cache_path(path_to_tensors, path_to_vectors, cols_to_build)

    if cache_path.exists():
        print(f"Loading cached marginals from {cache_path}")
        single_probs, total_len = _load_marginals_cache(cache_path, cols_to_build)
    else:
        single_probs = {col: Counter() for col in cols_to_build}
        batches1 = dataset.to_batches(
            columns=cols_to_build,
            batch_size=batch_rows,
            batch_readahead=batch_readahead,
            fragment_readahead=fragment_readahead,
            use_threads=True,
            cache_metadata=True,
        )
        print("Pass 1/2: computing global marginals ...")
        seen_rows = 0
        with tqdm(total=total_rows, desc="Pass 1/2", unit="rows") as pbar:
            for batch in batches1:
                pbar.update(batch.num_rows)
                seen_rows += batch.num_rows
                t = pa.table({col: _normalize_str_array(batch.column(i))
                              for i, col in enumerate(cols_to_build)})
                for col in cols_to_build:
                    g_col = (t.group_by([col])
                             .aggregate([(col, "count")])
                             .rename_columns([col, "count"]))
                    _update_counter_from_grouped(single_probs[col], g_col, [col], "count")
        if seen_rows == 0:
            raise ValueError("No rows found in the parquet dataset.")
        total_len = seen_rows
        _save_marginals_cache(cache_path, single_probs, total_len,
                                  cols_to_build, path_to_vectors)

    # vocab for max_k (per-mode) once
    vocabs_max = {
        col: _most_common_keys(single_probs[col], max_ks[i])
        for i, col in enumerate(cols_to_build)
    }

    # Optional factor linking: linked columns share the same top-k vocabulary.
    # Use the maximum of the linked modes' individual max_ks as the shared pool size.
    linked_groups = linked_factor_groups(len(cols_to_build), shared_factors)

    if shared_factors:
        print(f"Applying factor linking: {shared_factors}")

    for group in linked_groups:
        if len(group) <= 1:
            continue

        group_cols = [cols_to_build[i] for i in group]
        group_max_k = max(max_ks[i] for i in group)
        group_min_ks = [min_mode_ks.get(i, 0) for i in group] if min_mode_ks else None
        shared_vocab = _shared_topk_hmean(
            [single_probs[col] for col in group_cols],
            group_max_k,
            min_ks=group_min_ks,
            ensured_vocab=ensured_vocab,
        )

        for col in group_cols:
            vocabs_max[col] = shared_vocab

    # ranks for fast filtering later
    ranks = {col:{el:i for i, el in enumerate(vocabs_max[col])} for col in cols_to_build}

    max_arrs = {col:pa.array(vocabs_max[col]) for col in cols_to_build}


    # convert marginals to probabilities (global)
    for col, counter in single_probs.items():
        for k in list(counter.keys()):
            counter[k] /= total_len

    # Channel selection + fast-path determination — needed both to build the
    # post-Pass-2 arrays and to key the Pass-2 checkpoint.
    order = len(cols_to_build)
    need = {
        "count": need_count, "prob_log": need_prob_log,
        "count_log": need_count_log, "count_log_eps": need_count_log_eps,
        "sii": need_sii, "sc": need_sc,
    }
    base = max(len(vocabs_max[col]) for col in cols_to_build)
    use_fast = (not need_sii) or _encoding_fits(base, order)

    linked_nontrivial = [group for group in linked_groups if len(group) > 1]
    suffix = shared_factor_suffix(linked_nontrivial)

    # -------------------------
    # PASS 2 (+ post-processing arrays), with an on-disk checkpoint
    # -------------------------
    # The joint-count pass is the dominant cost (hours at 1B rows). When the fast
    # path applies we persist the resulting COW-safe arrays (_PP_STATE) so a re-run
    # — e.g. after a walltime kill mid variant-build — skips Pass 2 entirely.
    subset_counters: dict = {}
    full_subset = tuple(cols_to_build)
    pp_state = None
    pp_cache_path = None
    if use_fast:
        pp_cache_path = _pp_state_cache_path(
            path_to_tensors, path_to_vectors, cols_to_build,
            max_ks, shared_factors, need_sii, remove_hapax,
            min_mode_ks, ensured_vocab,
        )
        load_path = pp_cache_path if pp_cache_path.exists() else None
        if load_path is None and not need_sii:
            # A checkpoint written with sii sub-counters is a strict superset of
            # one without (sii only adds sub_keys/sub_counts arrays, and `need`
            # overrides at load), so a run that dropped the sii variants can
            # still reuse an earlier full-set checkpoint.
            superset_path = _pp_state_cache_path(
                path_to_tensors, path_to_vectors, cols_to_build,
                max_ks, shared_factors, True, remove_hapax,
                min_mode_ks, ensured_vocab,
            )
            if superset_path.exists():
                load_path = superset_path
        if load_path is not None:
            print(f"Loading cached post-Pass-2 arrays from {load_path} "
                  f"(skipping Pass 2)...")
            pp_state = _load_pp_state_cache(load_path, need)

    if pp_state is None:
        subset_counters = _compute_subset_counters(
            parquet_files, total_rows, cols_to_build, vocabs_max, max_ks,
            batch_rows, batch_readahead, fragment_readahead,
            max_workers, shards_per_task, cpu_frac, max_mem_gb, mem_per_worker_gb,
        )

        # Alternative hapax removal: remove 1-count tuples, not 1-count words
        if remove_hapax:
            subset_counters = _hapax_report_and_filter(subset_counters)

        # NOTE: we deliberately do NOT materialise a probability dict here. Dividing
        # counts by total_len on the fly (below) avoids duplicating the very large
        # full joint counter, which doubled peak RAM and caused the post-Pass-2 OOM.
        print("Probabilities computed for vocab-restricted subset marginals.")
        print("Probabilities computed for vocab-restricted joints.")

        if use_fast:
            print("Post-processing: converting counters to COW-safe NumPy arrays...")
            pp_state = _build_pp_arrays(
                subset_counters, single_probs, vocabs_max, ranks,
                list(cols_to_build), order, total_len, base, need,
            )
            try:
                _save_pp_state_cache(pp_cache_path, pp_state)
                print(f"Saved post-Pass-2 checkpoint to {pp_cache_path}")
            except Exception as e:  # checkpoint is best-effort; never fail the run
                print(f"WARNING: could not write post-Pass-2 checkpoint "
                      f"({pp_cache_path}): {e!r}")

    # def specific_interaction_information(v, s, o):
    #     return log(
    #         (p_xy[(v, s)] * p_yz[(s, o)] * p_xz[(v, o)]) /
    #         (p_x[v] * p_y[s] * p_z[o] * p_xyz[(v, s, o)])
    #     )

    # def specific_correlation(v, s, o):
    #     return log(p_xyz[(v, s, o)] / (p_x[v] * p_y[s] * p_z[o]))

    def specific_interaction_information(col_realisations):
        if len(col_realisations) != len(cols_to_build):
            raise ReferenceError("Same number of columns expected.")

        assignment = {
            cols_to_build[i]: col_realisations[i]
            for i in range(len(cols_to_build))
        }

        numerator_terms = []
        denominator_terms = []

        # General inclusion-exclusion form:
        # even-order subset marginals in numerator
        # odd-order subset marginals in denominator
        for r in range(1, len(cols_to_build) + 1):
            for subset in combinations(cols_to_build, r):
                key = assignment[subset[0]] if r == 1 else tuple(assignment[col] for col in subset)

                if r == 1:
                    p = single_probs[subset[0]][key]
                else:
                    p = subset_counters[subset].get(key, 0) / total_len

                if p <= 0:
                    return float("-inf")

                if r % 2 == 0:
                    numerator_terms.append(p)
                else:
                    denominator_terms.append(p)

                # Alternative, swapped version. Works worse on VSO
                # if r % 2 == 0:
                #     denominator_terms.append(p)
                # else:
                #     numerator_terms.append(p)

        return log(prod(numerator_terms) / prod(denominator_terms))


    def specific_correlation(col_realisations):
        if len(col_realisations) != len(cols_to_build):
            raise ReferenceError("Same number of columns expected.")

        joint = subset_counters[full_subset].get(tuple(col_realisations), 0) / total_len
        if joint <= 0:
            return float("-inf")

        marginals = []
        for i, realisation in enumerate(col_realisations):
            p = single_probs[cols_to_build[i]][realisation]
            if p <= 0:
                return float("-inf")
            marginals.append(p)

        return log(joint / prod(marginals))


    # -------------------------
    # Build tensors for each variant WITHOUT rescanning
    # -------------------------
    results = {}

    # Fast path uses the COW-safe NumPy arrays built (or loaded from checkpoint)
    # above; the per-row Python fallback (closures) only runs when composite-key
    # encoding would overflow int64 (very large order × vocab).
    global _PP_STATE
    pp_executor = None
    pp_chunks = 1
    if use_fast:
        _PP_STATE = pp_state
        N_rows = _PP_STATE["N"]
        # Parallelise across row-ranges via fork so the big arrays are COW-shared
        # (each worker's only extra memory is its output slice → ceiling respected).
        can_fork = "fork" in multiprocessing.get_all_start_methods()
        pp_workers = min(compute_num_threads(cpu_frac), max(1, N_rows // _PP_MIN_CHUNK))
        if can_fork and pp_workers > 1 and N_rows > 0:
            pp_executor = ProcessPoolExecutor(
                max_workers=pp_workers, mp_context=multiprocessing.get_context("fork")
            )
            pp_chunks = pp_workers
            print(f"Post-processing: {pp_workers} fork workers over {N_rows:,} joint rows.")
        else:
            print(f"Post-processing: serial vectorised over {N_rows:,} joint rows "
                  f"(fork={can_fork}, workers={pp_workers}).")

    def _compute_variant_values(variant):
        """Return (out_idx (m, order) int64, vals {channel: (m,) float32})."""
        if use_fast:
            n_all = _PP_STATE["N"]
            if pp_executor is None:
                return _fill_range(_PP_STATE, 0, n_all, variant)
            bounds = np.linspace(0, n_all, pp_chunks + 1, dtype=np.int64)
            futs = [
                pp_executor.submit(_pp_worker, int(bounds[i]), int(bounds[i + 1]), variant)
                for i in range(pp_chunks) if bounds[i] < bounds[i + 1]
            ]
            parts = [f.result() for f in futs]
            out_idx = np.concatenate([p[0] for p in parts], axis=0)
            vals = {k: np.concatenate([p[1][k] for p in parts]) for k in parts[0][1]}
            return out_idx, vals

        # ---- slow fallback: per-row Python loop (encoding overflow only) ----
        vocabs_v = {col: vocabs_max[col][:variant[i]] for i, col in enumerate(cols_to_build)}
        col2i_v = {col: {el: i for i, el in enumerate(vocabs_v[col])} for col in cols_to_build}

        def in_k(elements_to_check, _variant=variant):
            return all(
                ranks[cols_to_build[i]].get(el, 10**18) < _variant[i]
                for i, el in enumerate(elements_to_check)
            )

        full_counter = subset_counters[tuple(cols_to_build)]
        ub = len(full_counter)
        idx_arr = np.empty((ub, order), dtype=np.int64)
        buf = {k: (np.empty(ub, dtype=np.float32) if need[k] else None) for k in need}
        n = 0
        for els_to_check, cnt in tqdm(full_counter.items(), desc=f"nnz tuples ({dim_spec_str(variant)})"):
            if not in_k(els_to_check):
                continue
            for i, el in enumerate(els_to_check):
                idx_arr[n, i] = col2i_v[cols_to_build[i]][el]
            if need_count:         buf["count"][n] = cnt
            if need_prob_log:      buf["prob_log"][n] = log(cnt / total_len)
            if need_count_log:     buf["count_log"][n] = log(cnt)
            if need_count_log_eps: buf["count_log_eps"][n] = log(cnt) + _COUNT_LOG_EPS
            if need_sii:
                v = specific_interaction_information(els_to_check)
                buf["sii"][n] = float(v) if v != float("-inf") else -1e38
            if need_sc:
                v = specific_correlation(els_to_check)
                buf["sc"][n] = float(v) if v != float("-inf") else -1e38
            n += 1
        out_idx = idx_arr[:n]
        vals = {k: buf[k][:n] for k in need if need[k]}
        return out_idx, vals

    for variant in variants:
        dim_str = dim_spec_str(variant)
        # Resume: skip variants whose tensors (and vocab) are already on disk, so a
        # re-run after a partial/killed job picks up where it left off.
        if save:
            p_pop = f"{path_to_tensors}/populated"
            expected = [f"{p_pop}/{name}_{order}D_{dim_str}d{suffix}.pt" for name in want]
            vocab_pkl = f"{path_to_tensors}/vocabularies/{order}D_{dim_str}d{suffix}.pkl"
            if os.path.exists(vocab_pkl) and all(_is_complete_pt(fp) for fp in expected):
                print(f"\nVariant {variant} (dim_str={dim_str}) already built "
                      f"({len(want)} tensors present) — skipping.")
                continue
        print(f"\nBuilding tensors for variant={variant} (dim_str={dim_str})...")
        vocabs = {col: vocabs_max[col][:variant[i]] for i, col in enumerate(cols_to_build)}
        col2i = {col: {el: i for i, el in enumerate(vocabs[col])} for col in cols_to_build}

        out_idx, vals = _compute_variant_values(variant)
        m = out_idx.shape[0]
        size = tuple(len(vocabs[col]) for col in cols_to_build)

        if m == 0:
            idx = torch.empty((order, 0), dtype=torch.long)
            empty = torch.empty((0,), dtype=torch.float32)
            if need_count:
                count_tensor = _make_sparse_coo(idx, empty, size).coalesce()
            if need_prob_log:
                prob_log_tensor = _make_sparse_coo(idx, empty, size).coalesce()
            if need_count_log:
                count_log_tensor = _make_sparse_coo(idx, empty, size).coalesce()
            if need_count_log_eps:
                count_log_eps_tensor = _make_sparse_coo(idx, empty, size).coalesce()
            if need_sii:
                sii_tensor = _make_sparse_coo(idx, empty, size).coalesce()
            if need_sc:
                sc_tensor = _make_sparse_coo(idx, empty, size).coalesce()
        else:
            # out_idx is (m, order); transpose to (order, m) COO indices. Each channel
            # is built and coalesced from its own (COW-safe) NumPy buffer.
            idx = torch.from_numpy(np.ascontiguousarray(out_idx.T))
            if need_count:
                count_tensor = _make_sparse_coo(idx, torch.from_numpy(vals["count"]), size).coalesce()
            if need_prob_log:
                prob_log_tensor = _make_sparse_coo(idx, torch.from_numpy(vals["prob_log"]), size).coalesce()
            if need_count_log:
                count_log_tensor = _make_sparse_coo(idx, torch.from_numpy(vals["count_log"]), size).coalesce()
            if need_count_log_eps:
                count_log_eps_tensor = _make_sparse_coo(idx, torch.from_numpy(vals["count_log_eps"]), size).coalesce()
            if need_sii:
                sii_tensor = _make_sparse_coo(idx, torch.from_numpy(vals["sii"]), size).coalesce()
            if need_sc:
                sc_tensor = _make_sparse_coo(idx, torch.from_numpy(vals["sc"]), size).coalesce()
            del out_idx, vals

        # countingLog: drop hapax legomena whose log(count) == 0 (cnt == 1), not stored explicit zeros.
        # the epsilon path avoids this by adding the small nonzero constant
        if need_count_log and count_log_tensor._nnz():
            vvals = count_log_tensor.values()
            nz = vvals != 0
            count_log_tensor = _make_sparse_coo(
                count_log_tensor.indices()[:, nz], vvals[nz], size
            ).coalesce()

        # normalized variants
        eps = 1e-8
        if need_prob_log:
            if prob_log_tensor._nnz():
                vvals = prob_log_tensor.values()
                if "probLogShifted" in want: # This is normalised, raw didn't work for our usecase
                    _v = vvals - vvals.min()
                    prob_log_shifted  = _make_sparse_coo(prob_log_tensor.indices(), _v / (_v.max() + eps), size).coalesce()
                if "probLogSoftPlus" in want:
                    prob_log_softplus = _make_sparse_coo(prob_log_tensor.indices(), torch.nn.functional.softplus(vvals), size).coalesce()
            else:
                if "probLogShifted"  in want: prob_log_shifted  = prob_log_tensor
                if "probLogSoftPlus" in want: prob_log_softplus = prob_log_tensor

        if need_sii:
            if sii_tensor._nnz():
                vvals = sii_tensor.values()
                if "siiShifted" in want:
                    sii_shifted  = _make_sparse_coo(sii_tensor.indices(), vvals - vvals.min() + eps, size).coalesce()
                if "siiSoftPlus" in want:
                    sii_softplus = _make_sparse_coo(sii_tensor.indices(), torch.nn.functional.softplus(vvals), size).coalesce()
            else:
                if "siiShifted"  in want: sii_shifted  = sii_tensor
                if "siiSoftPlus" in want: sii_softplus = sii_tensor

        if need_sc:
            if sc_tensor._nnz():
                vvals = sc_tensor.values()
                if "scShifted" in want:
                    sc_shifted  = _make_sparse_coo(sc_tensor.indices(), vvals - vvals.min() + eps, size).coalesce()
                if "scSoftPlus" in want:
                    sc_softplus = _make_sparse_coo(sc_tensor.indices(), torch.nn.functional.softplus(vvals), size).coalesce()
                if "scSoftPlusFlat" in want:
                    _sp = torch.nn.functional.softplus(vvals)
                    sc_softplus_flat = _make_sparse_coo(sc_tensor.indices(), _soft_knee_compress(_sp), size).coalesce()
            else:
                if "scShifted"  in want: sc_shifted  = sc_tensor
                if "scSoftPlus" in want: sc_softplus = sc_tensor
                if "scSoftPlusFlat" in want: sc_softplus_flat = sc_tensor

        vocab = {}
        for col in cols_to_build:
            vocab[f"vocab_{col}"] = vocabs[col]
            vocab[f"{col}2i"] = col2i[col]

        p = f"{path_to_tensors}/populated"

        # Collect every requested tensor under its canonical name. Shared by the
        # save and in-memory paths so the reported sizes match what is persisted.
        built = {}
        if "counting"        in want: built["counting"]        = count_tensor
        if "countingLog"     in want: built["countingLog"]     = count_log_tensor
        if "countingLogEps"  in want: built["countingLogEps"]  = count_log_eps_tensor
        if "probLog"         in want: built["probLog"]         = prob_log_tensor
        if "probLogShifted"  in want: built["probLogShifted"]  = prob_log_shifted
        if "probLogSoftPlus" in want: built["probLogSoftPlus"] = prob_log_softplus
        if "sii"          in want: built["sii"]          = sii_tensor
        if "siiSoftPlus"  in want: built["siiSoftPlus"]  = sii_softplus
        if "siiShifted"   in want: built["siiShifted"]   = sii_shifted
        if "sc"           in want: built["sc"]           = sc_tensor
        if "scSoftPlus"   in want: built["scSoftPlus"]   = sc_softplus
        if "scShifted"    in want: built["scShifted"]    = sc_shifted
        if "scSoftPlusFlat" in want: built["scSoftPlusFlat"] = sc_softplus_flat

        # Report tensor sizes (nonzero entries) just before saving them.
        for name, tens in built.items():
            print(f"  {name} [{order}D {dim_str}d{suffix}]: {tens._nnz()} nonzero values")

        if save:
            for name, tens in built.items():
                _torch_save_atomic(tens, f"{p}/{name}_{order}D_{dim_str}d{suffix}.pt")
            # Vocab last: with atomic tensor saves it doubles as the variant's
            # completion marker for the resume check above.
            vocab_path = f"{path_to_tensors}/vocabularies/{order}D_{dim_str}d{suffix}.pkl"
            vocab_tmp = f"{vocab_path}.tmp"
            with open(vocab_tmp, "wb") as f:
                pickle.dump(vocab, f)
            os.replace(vocab_tmp, vocab_path)  # atomic

        else:
            results[variant] = (built, vocab)

    if pp_executor is not None:
        pp_executor.shutdown()
    _PP_STATE = {}

    return results


def rebuild_vocabularies(
    path_to_vectors,
    top_ks,
    path_to_tensors,
    cols_to_build,
    shared_factors=None,
    top_ks_asymmetric=None,
    min_mode_ks: dict[int, int] | None = None,
    ensured_vocab: list[str] | None = None,
    overwrite: bool = True,
):
    """Rewrite ONLY the per-variant vocabulary pickles, from cached marginals.

    The vocabulary files are a pure function of the Pass-1 marginals plus the
    vocab-selection config (top-ks, factor linking, min_mode_ks, ensured_vocab) —
    they do not depend on Pass 2 or the built tensors. This lets you repair a
    failed vocabulary save in seconds without rebuilding anything.

    Selection here calls the *same* helpers as ``populate_tensors_parquet``
    (``_most_common_keys`` / ``_shared_topk_hmean`` / ``linked_factor_groups`` /
    ``shared_factor_suffix`` / ``dim_spec_str``), so the output is identical to
    what the full run would have written.
    """
    path_to_vectors = os.fspath(path_to_vectors)
    n_modes = len(cols_to_build)

    # Same variant normalisation as populate_tensors_parquet.
    if not isinstance(top_ks, list):
        top_ks = [top_ks]
    variants: list[tuple] = [(k,) * n_modes for k in top_ks]
    for tup in (top_ks_asymmetric or []):
        tup = tuple(tup)
        if len(tup) != n_modes:
            raise ValueError(
                f"top_ks_asymmetric entry {tup} has {len(tup)} elements but "
                f"cols_to_build has {n_modes}."
            )
        variants.append(tup)
    variants = sorted(set(variants))
    max_ks = tuple(max(v[i] for v in variants) for i in range(n_modes))

    vocab_dir = Path(path_to_tensors) / "vocabularies"
    vocab_dir.mkdir(parents=True, exist_ok=True)

    # Marginals must already be cached (Pass 1 of the original run).
    cache_path = _marginals_cache_path(path_to_tensors, path_to_vectors, cols_to_build)
    if not cache_path.exists():
        raise FileNotFoundError(
            f"No cached marginals at {cache_path}. Vocabulary rebuild needs the "
            f"Pass-1 marginals; run the populator (it caches them) first."
        )
    print(f"Loading cached marginals from {cache_path}")
    single_probs, total_len = _load_marginals_cache(cache_path, cols_to_build)

    # vocab for max_k (per-mode), then optional factor linking — identical to the
    # populate path. (No probability conversion / ranks needed for vocab files.)
    vocabs_max = {
        col: _most_common_keys(single_probs[col], max_ks[i])
        for i, col in enumerate(cols_to_build)
    }
    linked_groups = linked_factor_groups(len(cols_to_build), shared_factors)
    if shared_factors:
        print(f"Applying factor linking: {shared_factors}")
    for group in linked_groups:
        if len(group) <= 1:
            continue
        group_cols = [cols_to_build[i] for i in group]
        group_max_k = max(max_ks[i] for i in group)
        group_min_ks = [min_mode_ks.get(i, 0) for i in group] if min_mode_ks else None
        shared_vocab = _shared_topk_hmean(
            [single_probs[col] for col in group_cols],
            group_max_k,
            min_ks=group_min_ks,
            ensured_vocab=ensured_vocab,
        )
        for col in group_cols:
            vocabs_max[col] = shared_vocab

    linked_nontrivial = [group for group in linked_groups if len(group) > 1]
    suffix = shared_factor_suffix(linked_nontrivial)
    order = len(cols_to_build)

    written = []
    for variant in variants:
        dim_str = dim_spec_str(variant)
        out_path = vocab_dir / f"{order}D_{dim_str}d{suffix}.pkl"
        if out_path.exists() and not overwrite:
            print(f"  exists, skipping: {out_path.name}")
            continue
        vocabs = {col: vocabs_max[col][:variant[i]] for i, col in enumerate(cols_to_build)}
        col2i = {col: {el: i for i, el in enumerate(vocabs[col])} for col in cols_to_build}
        vocab = {}
        for col in cols_to_build:
            vocab[f"vocab_{col}"] = vocabs[col]
            vocab[f"{col}2i"] = col2i[col]
        tmp = out_path.with_name(out_path.name + ".tmp")
        with open(tmp, "wb") as f:
            pickle.dump(vocab, f)
        os.replace(tmp, out_path)  # atomic
        written.append(out_path.name)
        print(f"  wrote {out_path.name} "
              f"(sizes: {tuple(len(vocabs[c]) for c in cols_to_build)})")

    print(f"Rebuilt {len(written)} vocabulary file(s) in {vocab_dir}")
    return written