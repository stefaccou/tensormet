from math import log, prod
from collections import Counter, defaultdict
from pathlib import Path
import os
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
import multiprocessing
import torch
import numpy as np
from tqdm import tqdm
from tensormet.utils import DATA_DIR, shared_factor_suffix, linked_factor_groups, SparseCOOTensor, _INT64_MAX, dim_spec_str
import pickle

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.dataset as ds

from itertools import combinations
from functools import reduce
import hashlib, json

_ALL_TENSORS = [
    "counting", "countingLog", "countingLogEps",
    "probLog", "probLogSoftPlus", "probLogShifted",
    "sii", "siiSoftPlus", "siiShifted",
    "sc",  "scSoftPlus",  "scShifted", "scSoftPlusFlat",
]

# epsilon added after log(count) so that singletons (count == 1) map to a small
# non-zero value instead of log(1) == 0, keeping them in the sparse tensor.
_COUNT_LOG_EPS = 1e-8


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

def _hapax_filter_marginals_report(single_probs: dict, cols: list[str]) -> dict:
    """
    Remove count==1 entries from each marginal counter, before vocab selection.
    Prints a per-column before/after report. Returns a new dict of counters.
    """
    print("\n── Marginal hapax removal ────────────────────────────────")
    rows = []
    filtered = {}
    for col in cols:
        c = single_probs[col]
        types_before = len(c)
        tokens_before = sum(c.values())
        new_c = Counter({k: v for k, v in c.items() if v > 10})
        types_after = len(new_c)
        tokens_after = sum(new_c.values())
        filtered[col] = new_c
        rows.append((col, types_before, types_after, tokens_before, tokens_after))

    col_w = max(len(r[0]) for r in rows)
    header = (f"{'Column':<{col_w}}  {'Types before':>14}  {'Types after':>12}"
              f"  {'% removed':>10}  {'Tokens before':>14}  {'Tokens after':>13}")
    print(header)
    print("-" * len(header))
    for col, tb, ta, kb, ka in rows:
        pct = 100 * (tb - ta) / tb if tb else 0.0
        print(f"{col:<{col_w}}  {tb:>14,}  {ta:>12,}  {pct:>9.1f}%  {kb:>14,}  {ka:>13,}")
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
):
    if tensors_to_build is not None:
        unknown = [t for t in tensors_to_build if t not in _ALL_TENSORS]
        if unknown:
            raise ValueError(f"Unknown tensor names: {unknown}. Valid: {_ALL_TENSORS}")
        want = list(tensors_to_build)
    else:
        want = _ALL_TENSORS

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

    # NEW: marginal hapax removal, before vocab selection
    # if remove_hapax:
    #     single_probs = _hapax_filter_marginals_report(single_probs, cols_to_build)

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

    # -------------------------
    # PASS 2: restricted joint counts — parallelised across shards
    # -------------------------
    subset_counters = {
        subset: Counter()
        for r in range(2, len(cols_to_build) + 1)
        for subset in combinations(cols_to_build, r)
    }

    _cpu = multiprocessing.cpu_count()
    if max_workers and max_workers > 0:
        n_workers = min(max_workers, len(parquet_files))
    else:
        # ~1 worker per 100 shards; floor 4, cap at cpu_count.
        # Small shards (many hundreds) get more workers; large shards (tens) stay at 4.
        n_workers = min(max(4, len(parquet_files) // 100), _cpu, len(parquet_files))
    shards_per_worker = len(parquet_files) / n_workers
    print(f"using {n_workers} workers ({shards_per_worker:.0f} shards/worker, {shards_per_task} shard(s)/task)")

    shards_per_task = max(1, shards_per_task)
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

    import time as _time

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



    # Alternative hapax removal: remove 1-count tuples, not 1-count words
    if remove_hapax:
        subset_counters = _hapax_report_and_filter(subset_counters)

    # NOTE: we deliberately do NOT materialise a probability dict here. Dividing
    # counts by total_len on the fly (below) avoids duplicating the very large full
    # joint counter, which doubled peak RAM and was the cause of the post-Pass-2 OOM.
    full_subset = tuple(cols_to_build)

    print("Probabilities computed for vocab-restricted subset marginals.")
    print("Probabilities computed for vocab-restricted joints.")

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

    linked_nontrivial = [group for group in linked_groups if len(group) > 1]
    suffix = shared_factor_suffix(linked_nontrivial)
    order = len(cols_to_build)

    for variant in variants:
        dim_str = dim_spec_str(variant)
        print(f"\nBuilding tensors for variant={variant} (dim_str={dim_str})...")
        vocabs = {col: vocabs_max[col][:variant[i]] for i, col in enumerate(cols_to_build)}
        col2i = {col: {el: i for i, el in enumerate(vocabs[col])} for col in cols_to_build}

        def in_k(elements_to_check, _variant=variant):
            return all(
                ranks[cols_to_build[i]].get(el, 10**18) < _variant[i]
                for i, el in enumerate(elements_to_check)
            )

        # filter tuples from max counter
        full_counter = subset_counters[tuple(cols_to_build)]
        ub = len(full_counter)  # upper bound on surviving (in-vocab) tuples

        # Pre-sized buffers filled up to running index n, then sliced to [:n]. Avoids
        # the tens-of-GB overhead of a Python list-of-lists plus the torch.tensor()
        # conversion copy at 1B scale.
        idx_arr = np.empty((ub, order), dtype=np.int64)
        count_values     = np.empty(ub, dtype=np.float32) if need_count     else None
        prob_log_values  = np.empty(ub, dtype=np.float32) if need_prob_log  else None
        count_log_values = np.empty(ub, dtype=np.float32) if need_count_log else None
        count_log_eps_values = np.empty(ub, dtype=np.float32) if need_count_log_eps else None
        sii_values       = np.empty(ub, dtype=np.float32) if need_sii       else None
        sc_values        = np.empty(ub, dtype=np.float32) if need_sc        else None

        n = 0
        for els_to_check, cnt in tqdm(full_counter.items(), desc=f"nnz tuples ({dim_str})"):
            if not in_k(els_to_check):
                continue
            for i, el in enumerate(els_to_check):
                idx_arr[n, i] = col2i[cols_to_build[i]][el]
            if need_count:
                count_values[n] = cnt
            if need_prob_log:
                # uses log of PROBABILITY --> Negative
                prob_log_values[n] = log(cnt / total_len)
            if need_count_log:
                # pure log of the raw COUNT --> non-negative; log(1)=0 kept as-is
                count_log_values[n] = log(cnt)
            if need_count_log_eps:
                # uses log of the raw COUNT --> non-negative; eps keeps log(1) > 0
                count_log_eps_values[n] = log(cnt) + _COUNT_LOG_EPS
            # Use finite values instead of -inf to prevent coalesce/sorting issues
            if need_sii:
                sii_val = specific_interaction_information(els_to_check)
                sii_values[n] = float(sii_val) if sii_val != float("-inf") else -1e38
            if need_sc:
                sc_val = specific_correlation(els_to_check)
                sc_values[n] = float(sc_val) if sc_val != float("-inf") else -1e38
            n += 1

        size = tuple(len(vocabs[col]) for col in cols_to_build)

        if n == 0:
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
            # Build from pre-sized NumPy buffers (sliced to the n surviving rows).
            # idx is an independent contiguous copy, so idx_arr is freed immediately.
            # Each value channel is built, coalesced and freed before the next so only
            # one channel's buffer is held at a time (torch.from_numpy shares the buffer
            # until coalesce gives the tensor its own storage).
            idx = torch.from_numpy(idx_arr[:n]).t().contiguous()
            del idx_arr

            if need_count:
                count_tensor = _make_sparse_coo(idx, torch.from_numpy(count_values[:n]), size).coalesce()
                del count_values
            if need_prob_log:
                prob_log_tensor = _make_sparse_coo(idx, torch.from_numpy(prob_log_values[:n]), size).coalesce()
                del prob_log_values
            if need_count_log:
                count_log_tensor = _make_sparse_coo(idx, torch.from_numpy(count_log_values[:n]), size).coalesce()
                del count_log_values
            if need_count_log_eps:
                count_log_eps_tensor = _make_sparse_coo(idx, torch.from_numpy(count_log_eps_values[:n]), size).coalesce()
                del count_log_eps_values
            if need_sii:
                sii_tensor = _make_sparse_coo(idx, torch.from_numpy(sii_values[:n]), size).coalesce()
                del sii_values
            if need_sc:
                sc_tensor = _make_sparse_coo(idx, torch.from_numpy(sc_values[:n]), size).coalesce()
                del sc_values

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
                torch.save(tens, f"{p}/{name}_{order}D_{dim_str}d{suffix}.pt")
            with open(f"{path_to_tensors}/vocabularies/{order}D_{dim_str}d{suffix}.pkl", "wb") as f:
                pickle.dump(vocab, f)
        else:
            results[variant] = (built, vocab)

    return results