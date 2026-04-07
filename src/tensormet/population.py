from math import log
from collections import Counter
from pathlib import Path
import csv, os
import numpy as np
import time
import torch
from tqdm import tqdm
from tensormet.utils import notify_discord, DATA_DIR, select_gpu
import pickle
import ast

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.dataset as ds

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

# def _update_counter_from_grouped(counter: Counter, grouped: pa.Table, key_cols: list[str], count_col: str) -> None:
#     """
#     grouped: columns key_cols + [count_col]
#     Updates `counter` with counts.
#     """
#     # Convert small grouped results to Python once per batch
#     cols = [grouped[c].to_pylist() for c in key_cols]
#     counts = grouped[count_col].to_pylist()
#
#     if len(key_cols) == 1:
#         keys = cols[0]
#         for k, c in zip(keys, counts):
#             counter[k] += int(c)
#     elif len(key_cols) == 2:
#         a, b = cols
#         for x, y, c in zip(a, b, counts):
#             counter[(x, y)] += int(c)
#     elif len(key_cols) == 3:
#         a, b, ccol = cols
#         for x, y, z, c in zip(a, b, ccol, counts):
#             counter[(x, y, z)] += int(c)
#     else:
#         raise ValueError("Unsupported number of key columns.")

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

def _most_common_keys(counter: Counter, k: int) -> list:
    # Counter.most_common is deterministic enough for our use;
    # ties will follow internal ordering—same behavior as original Counter use.
    return [x for (x, _) in counter.most_common(k)]

# def populate_tensors_parquet(
#     path_to_vectors,
#     top_ks,
#     save: bool = True,
#     path_to_tensors=None,
#     v_col: str = "root",
#     s_col: str = "nsubj",
#     o_col: str = "obj",
#     batch_rows: int = 256_000,
#     batch_readahead: int = 32,
#     fragment_readahead: int = 8,
# ):
#     path_to_vectors = os.fspath(path_to_vectors)
#     print(f"Populating tensors for top_k={top_ks} from {path_to_vectors}...")
#
#     if not path_to_tensors:
#         base = os.path.basename(os.path.normpath(path_to_vectors))
#         dataset_name = base.split("_")[0] + "_sparse"
#         path_to_tensors = DATA_DIR / f"tensors/{dataset_name}/"
#
#     os.makedirs(path_to_tensors, exist_ok=True)
#     os.makedirs(f"{path_to_tensors}/populated", exist_ok=True)
#     os.makedirs(f"{path_to_tensors}/vocabularies", exist_ok=True)
#     print(f"Tensors will be saved to {path_to_tensors}")
#
#     if not isinstance(top_ks, list):
#         top_ks = [top_ks]
#     top_ks = sorted(top_ks)
#     max_k = top_ks[-1]
#
#     vector_dir = Path(path_to_vectors)
#     parquet_files = sorted(vector_dir.glob("part-*.parquet"))
#
#     if not parquet_files:
#         raise FileNotFoundError(f"No parquet shards found in {vector_dir}")
#
#     dataset = ds.dataset(parquet_files, format="parquet")
#
#     # dataset = ds.dataset(path_to_vectors, format="parquet")
#     total_rows = dataset.count_rows(use_threads=True, cache_metadata=True)
#     print(f"Total rows: {total_rows:,} | Shards: 201 | batch_rows={batch_rows:,}")
#
#     scan_cols = [v_col, s_col, o_col]
#
#     # -------------------------
#     # PASS 1: marginals only
#     # -------------------------
#     p_x = Counter()
#     p_y = Counter()
#     p_z = Counter()
#
#     batches1 = dataset.to_batches(
#         columns=scan_cols,
#         batch_size=batch_rows,
#         batch_readahead=batch_readahead,
#         fragment_readahead=fragment_readahead,
#         use_threads=True,
#         cache_metadata=True,
#     )
#
#     print("Pass 1/2: computing global marginals (v,s,o) ...")
#     seen_rows = 0
#     with tqdm(total=total_rows, desc="Pass 1/2", unit="rows") as pbar:
#         for batch in batches1:
#             pbar.update(batch.num_rows)
#             seen_rows += batch.num_rows
#
#             v = _normalize_str_array(batch.column(0))
#             s = _normalize_str_array(batch.column(1))
#             o = _normalize_str_array(batch.column(2))
#             t = pa.table({"v": v, "s": s, "o": o})
#
#             gx = t.group_by(["v"]).aggregate([("v", "count")]).rename_columns(["v", "count"])
#             gy = t.group_by(["s"]).aggregate([("s", "count")]).rename_columns(["s", "count"])
#             gz = t.group_by(["o"]).aggregate([("o", "count")]).rename_columns(["o", "count"])
#
#             _update_counter_from_grouped(p_x, gx, ["v"], "count")
#             _update_counter_from_grouped(p_y, gy, ["s"], "count")
#             _update_counter_from_grouped(p_z, gz, ["o"], "count")
#
#     if seen_rows == 0:
#         raise ValueError("No rows found in the parquet dataset.")
#     total_len = seen_rows  # global denominator for probabilities
#
#     # vocab for max_k once
#     vocab_v_max = _most_common_keys(p_x, max_k)
#     vocab_s_max = _most_common_keys(p_y, max_k)
#     vocab_o_max = _most_common_keys(p_z, max_k)
#
#     # ranks for fast filtering later
#     v_rank = {v: i for i, v in enumerate(vocab_v_max)}
#     s_rank = {s: i for i, s in enumerate(vocab_s_max)}
#     o_rank = {o: i for i, o in enumerate(vocab_o_max)}
#
#     v_max_arr = pa.array(vocab_v_max)
#     s_max_arr = pa.array(vocab_s_max)
#     o_max_arr = pa.array(vocab_o_max)
#
#     # convert marginals to probabilities (global)
#     for counter in (p_x, p_y, p_z):
#         for k in list(counter.keys()):
#             counter[k] /= total_len
#
#     # -------------------------
#     # PASS 2: restricted joint counts (only keys in max vocab)
#     # -------------------------
#     c_xy = Counter()
#     c_xz = Counter()
#     c_yz = Counter()
#     c_xyz = Counter()
#
#     print("Defining batches for pass 2")
#     batches2 = dataset.to_batches(
#         columns=scan_cols,
#         batch_size=batch_rows,
#         batch_readahead=batch_readahead,
#         fragment_readahead=fragment_readahead,
#         use_threads=True,
#         cache_metadata=True,
#     )
#
#     print(f"Pass 2/2: computing joint counts restricted to max_k={max_k} vocab ...")
#     with tqdm(total=total_rows, desc="Pass 2/2", unit="rows") as pbar:
#         for batch in batches2:
#             pbar.update(batch.num_rows)
#
#             v = _normalize_str_array(batch.column(0))
#             s = _normalize_str_array(batch.column(1))
#             o = _normalize_str_array(batch.column(2))
#             t = pa.table({"v": v, "s": s, "o": o})
#
#             # masks (note: these are per-pair, not “all three”)
#             mv = pc.is_in(t["v"], value_set=v_max_arr)
#             ms = pc.is_in(t["s"], value_set=s_max_arr)
#             mo = pc.is_in(t["o"], value_set=o_max_arr)
#
#             # (v,s) where v,s in vocab
#             m_vs = pc.and_(mv, ms)
#             tvs = t.filter(m_vs)
#             if tvs.num_rows:
#                 gxy = tvs.group_by(["v", "s"]).aggregate([("v", "count")]).rename_columns(["v", "s", "count"])
#                 _update_counter_from_grouped(c_xy, gxy, ["v", "s"], "count")
#
#             # (v,o)
#             m_vo = pc.and_(mv, mo)
#             tvo = t.filter(m_vo)
#             if tvo.num_rows:
#                 gxz = tvo.group_by(["v", "o"]).aggregate([("v", "count")]).rename_columns(["v", "o", "count"])
#                 _update_counter_from_grouped(c_xz, gxz, ["v", "o"], "count")
#
#             # (s,o)
#             m_so = pc.and_(ms, mo)
#             tso = t.filter(m_so)
#             if tso.num_rows:
#                 gyz = tso.group_by(["s", "o"]).aggregate([("s", "count")]).rename_columns(["s", "o", "count"])
#                 _update_counter_from_grouped(c_yz, gyz, ["s", "o"], "count")
#
#             # (v,s,o) where all three in vocab
#             m_vso = pc.and_(m_vs, mo)
#             tvso = t.filter(m_vso)
#             if tvso.num_rows:
#                 gxyz = tvso.group_by(["v", "s", "o"]).aggregate([("v", "count")]).rename_columns(["v", "s", "o", "count"])
#                 _update_counter_from_grouped(c_xyz, gxyz, ["v", "s", "o"], "count")
#
#     # convert restricted joints to probabilities using global denominator (correct)
#     p_xy = Counter({k: v / total_len for k, v in c_xy.items()})
#     p_xz = Counter({k: v / total_len for k, v in c_xz.items()})
#     p_yz = Counter({k: v / total_len for k, v in c_yz.items()})
#     p_xyz = Counter({k: v / total_len for k, v in c_xyz.items()})
#
#     print("Probabilities computed for vocab-restricted joints.")
#
#     def specific_interaction_information(v, s, o):
#         return log(
#             (p_xy[(v, s)] * p_yz[(s, o)] * p_xz[(v, o)]) /
#             (p_x[v] * p_y[s] * p_z[o] * p_xyz[(v, s, o)])
#         )
#
#     def specific_correlation(v, s, o):
#         return log(p_xyz[(v, s, o)] / (p_x[v] * p_y[s] * p_z[o]))
#
#     # -------------------------
#     # Build tensors for each top_k WITHOUT rescanning
#     # -------------------------
#     results = {}
#
#     for top_k in top_ks:
#         print(f"\nBuilding tensors for top_k={top_k} (no rescan) ...")
#
#         vocab_v = vocab_v_max[:top_k]
#         vocab_s = vocab_s_max[:top_k]
#         vocab_o = vocab_o_max[:top_k]
#
#         v2i = {v: i for i, v in enumerate(vocab_v)}
#         s2i = {s: i for i, s in enumerate(vocab_s)}
#         o2i = {o: i for i, o in enumerate(vocab_o)}
#
#         def in_k(v, s, o):
#             return (v_rank.get(v, 10**18) < top_k and
#                     s_rank.get(s, 10**18) < top_k and
#                     o_rank.get(o, 10**18) < top_k)
#
#         # filter triples from max counter
#         indices, count_values, sii_values, sc_values = [], [], [], []
#         for (v, s, o), cnt in tqdm(c_xyz.items(), desc=f"nnz triples (top_k={top_k})"):
#             if not in_k(v, s, o):
#                 continue
#             indices.append([v2i[v], s2i[s], o2i[o]])
#             count_values.append(float(cnt))
#             sii_values.append(float(specific_interaction_information(v, s, o)))
#             sc_values.append(float(specific_correlation(v, s, o)))
#
#         size = (len(vocab_v), len(vocab_s), len(vocab_o))
#         if len(indices) == 0:
#             idx = torch.empty((3, 0), dtype=torch.long)
#             empty = torch.empty((0,), dtype=torch.float32)
#             count_tensor = torch.sparse_coo_tensor(idx, empty, size=size).coalesce()
#             sii_tensor = torch.sparse_coo_tensor(idx, empty, size=size).coalesce()
#             sc_tensor = torch.sparse_coo_tensor(idx, empty, size=size).coalesce()
#         else:
#             idx = torch.tensor(indices, dtype=torch.long).t()
#             count_tensor = torch.sparse_coo_tensor(idx, torch.tensor(count_values, dtype=torch.float32), size=size).coalesce()
#             sii_tensor = torch.sparse_coo_tensor(idx, torch.tensor(sii_values, dtype=torch.float32), size=size).coalesce()
#             sc_tensor  = torch.sparse_coo_tensor(idx, torch.tensor(sc_values, dtype=torch.float32), size=size).coalesce()
#
#         # normalized variants
#         eps = 1e-8
#         if sii_tensor._nnz():
#             vvals = sii_tensor.values()
#             sii_shifted = torch.sparse_coo_tensor(sii_tensor.indices(), vvals - vvals.min() + eps, size=size).coalesce()
#             sii_softplus = torch.sparse_coo_tensor(sii_tensor.indices(), torch.nn.functional.softplus(vvals), size=size).coalesce()
#         else:
#             sii_shifted = sii_tensor
#             sii_softplus = sii_tensor
#
#         if sc_tensor._nnz():
#             vvals = sc_tensor.values()
#             sc_shifted = torch.sparse_coo_tensor(sc_tensor.indices(), vvals - vvals.min() + eps, size=size).coalesce()
#             sc_softplus = torch.sparse_coo_tensor(sc_tensor.indices(), torch.nn.functional.softplus(vvals), size=size).coalesce()
#         else:
#             sc_shifted = sc_tensor
#             sc_softplus = sc_tensor
#
#         vocab = {
#             "vocab_v": vocab_v,
#             "vocab_s": vocab_s,
#             "vocab_o": vocab_o,
#             "v2i": v2i,
#             "s2i": s2i,
#             "o2i": o2i,
#         }
#
#         if save:
#             torch.save(count_tensor, f"{path_to_tensors}/populated/counting_{top_k}.pt")
#             torch.save(sii_tensor,   f"{path_to_tensors}/populated/sii_{top_k}.pt")
#             torch.save(sc_tensor,    f"{path_to_tensors}/populated/sc_{top_k}.pt")
#             torch.save(sc_softplus,  f"{path_to_tensors}/populated/scSoftPlus_{top_k}.pt")
#             torch.save(sii_softplus, f"{path_to_tensors}/populated/siiSoftPlus_{top_k}.pt")
#             torch.save(sc_shifted,   f"{path_to_tensors}/populated/scShifted_{top_k}.pt")
#             torch.save(sii_shifted,  f"{path_to_tensors}/populated/siiShifted_{top_k}.pt")
#             with open(f"{path_to_tensors}/vocabularies/{top_k}.pkl", "wb") as f:
#                 pickle.dump(vocab, f)
#         else:
#             results[top_k] = (count_tensor, sii_tensor, sc_tensor, vocab)
#
#     notify_discord(f"Finished populating tensors from {path_to_vectors}.")
#     return results


def populate_tensors_parquet(
    path_to_vectors,
    top_ks,
    save: bool = True,
    path_to_tensors=None,
    cols_to_build : list = ["root", "nsubj", "obj"],
    shared_factors=None,
    batch_rows: int = 256_000,
    batch_readahead: int = 32,
    fragment_readahead: int = 8,
):
    path_to_vectors = os.fspath(path_to_vectors)
    print(f"Populating tensors for top_k={top_ks} from {path_to_vectors}...")

    if not path_to_tensors:
        base = os.path.basename(os.path.normpath(path_to_vectors))
        dataset_name = base.split("_")[0] + "_sparse"
        path_to_tensors = DATA_DIR / f"tensors/{dataset_name}/"

    os.makedirs(path_to_tensors, exist_ok=True)
    os.makedirs(f"{path_to_tensors}/populated", exist_ok=True)
    os.makedirs(f"{path_to_tensors}/vocabularies", exist_ok=True)
    print(f"Tensors will be saved to {path_to_tensors}")

    if not isinstance(top_ks, list):
        top_ks = [top_ks]
    top_ks = sorted(top_ks)
    max_k = top_ks[-1]

    vector_dir = Path(path_to_vectors)
    parquet_files = sorted(vector_dir.glob("part-*.parquet"))

    if not parquet_files:
        raise FileNotFoundError(f"No parquet shards found in {vector_dir}")

    dataset = ds.dataset(parquet_files, format="parquet")

    # dataset = ds.dataset(path_to_vectors, format="parquet")
    total_rows = dataset.count_rows(use_threads=True, cache_metadata=True)
    print(f"Total rows: {total_rows:,} | Shards: 201 | batch_rows={batch_rows:,}")


    # -------------------------
    # PASS 1: marginals only
    # -------------------------

    single_probs = {column:Counter() for column in cols_to_build}

    batches1 = dataset.to_batches(
        columns=cols_to_build,
        batch_size=batch_rows,
        batch_readahead=batch_readahead,
        fragment_readahead=fragment_readahead,
        use_threads=True,
        cache_metadata=True,
    )

    print("Pass 1/2: computing global marginals (v,s,o) ...")
    seen_rows = 0
    with tqdm(total=total_rows, desc="Pass 1/2", unit="rows") as pbar:
        for batch in batches1:
            pbar.update(batch.num_rows)
            seen_rows += batch.num_rows

            t = pa.table({col:_normalize_str_array(batch.column(i)) for i, col in enumerate(cols_to_build)})
            for col in cols_to_build:
                g_col = t.group_by([col]).aggregate([(col, "count")]).rename_columns([col, "count"])
                _update_counter_from_grouped(single_probs[col], g_col, [col], "count")

    if seen_rows == 0:
        raise ValueError("No rows found in the parquet dataset.")
    total_len = seen_rows  # global denominator for probabilities

    # vocab for max_k once
    vocabs_max = {col: _most_common_keys(single_probs[col], max_k) for col in cols_to_build}

    # Optional factor linking: linked columns share the same top-k vocabulary
    linked_groups = _linked_factor_groups(len(cols_to_build), shared_factors)


    if shared_factors:
        print(f"Applying factor linking: {shared_factors}")

    for group in linked_groups:
        if len(group) <= 1:
            continue

        group_cols = [cols_to_build[i] for i in group]
        shared_vocab = _shared_topk_hmean(
            [single_probs[col] for col in group_cols],
            max_k
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
    # PASS 2: restricted joint counts (only keys in max vocab)
    # -------------------------

    # All subset counters of size >= 2, including the full joint
    subset_counters = {
        subset: Counter()
        for r in range(2, len(cols_to_build) + 1)
        for subset in combinations(cols_to_build, r)
    }

    print("Defining batches for pass 2")
    batches2 = dataset.to_batches(
        columns=cols_to_build,
        batch_size=batch_rows,
        batch_readahead=batch_readahead,
        fragment_readahead=fragment_readahead,
        use_threads=True,
        cache_metadata=True,
    )

    print(f"Pass 2/2: computing joint counts restricted to max_k={max_k} vocab ...")
    with tqdm(total=total_rows, desc="Pass 2/2", unit="rows") as pbar:
        for batch in batches2:
            pbar.update(batch.num_rows)

            t = pa.table({col:_normalize_str_array(batch.column(i)) for i, col in enumerate(cols_to_build)})


            masks = {col:pc.is_in(t[col], value_set=max_arrs[col]) for col in cols_to_build}


             # Count every subset marginal of size >= 2 restricted to max vocab
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
                            subset_counters[subset],
                            g_subset,
                            list(subset),
                            "count",
                        )





    # convert restricted joints to probabilities using global denominator
    subset_probabilities = {
        subset: Counter({k: v / total_len for k, v in counter.items()})
        for subset, counter in subset_counters.items()
    }

    full_subset = tuple(cols_to_build)
    p_full = subset_probabilities[full_subset]

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
                    p = subset_probabilities[subset][key]

                if p <= 0:
                    return float("-inf")

                if r % 2 == 0:
                    numerator_terms.append(p)
                else:
                    denominator_terms.append(p)

        return log(prod(numerator_terms) / prod(denominator_terms))


    def specific_correlation(col_realisations):
        if len(col_realisations) != len(cols_to_build):
            raise ReferenceError("Same number of columns expected.")

        joint = p_full[tuple(col_realisations)]
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
    # Build tensors for each top_k WITHOUT rescanning
    # -------------------------
    results = {}

    for top_k in top_ks:
        print(f"\nBuilding tensors for top_k={top_k} (no rescan) ...")
        vocabs = {col:vocabs_max[col][:top_k] for col in cols_to_build}
        col2i = {col:{el:i for i, el in enumerate(vocabs[col])} for col in cols_to_build}

        def in_k(elements_to_check):
            return all(ranks[cols_to_build[i]].get(el, 10**18) < top_k for i, el in enumerate(elements_to_check))

        # filter triples from max counter
        indices, count_values, sii_values, sc_values = [], [], [], []

        full_counter = subset_counters[tuple(cols_to_build)]

        for els_to_check, cnt in tqdm(full_counter.items(), desc=f"nnz tuples (top_k={top_k})"):
            if not in_k(els_to_check):
                continue
            indices.append([col2i[cols_to_build[i]][el] for i, el in enumerate(els_to_check)])
            count_values.append(float(cnt))
            sii_values.append(float(specific_interaction_information(els_to_check)))
            sc_values.append(float(specific_correlation(els_to_check)))

        # size = (len(vocab_v), len(vocab_s), len(vocab_o))
        size = tuple(len(vocabs[col]) for col in cols_to_build)
        if len(indices) == 0:
            idx = torch.empty((len(cols_to_build), 0), dtype=torch.long)
            empty = torch.empty((0,), dtype=torch.float32)
            count_tensor = torch.sparse_coo_tensor(idx, empty, size=size).coalesce()
            sii_tensor = torch.sparse_coo_tensor(idx, empty, size=size).coalesce()
            sc_tensor = torch.sparse_coo_tensor(idx, empty, size=size).coalesce()

        else:
            idx = torch.tensor(indices, dtype=torch.long).t()
            count_tensor = torch.sparse_coo_tensor(idx, torch.tensor(count_values, dtype=torch.float32), size=size).coalesce()
            sii_tensor = torch.sparse_coo_tensor(idx, torch.tensor(sii_values, dtype=torch.float32), size=size).coalesce()
            sc_tensor  = torch.sparse_coo_tensor(idx, torch.tensor(sc_values, dtype=torch.float32), size=size).coalesce()

        # normalized variants
        eps = 1e-8
        if sii_tensor._nnz():
            vvals = sii_tensor.values()
            sii_shifted = torch.sparse_coo_tensor(sii_tensor.indices(), vvals - vvals.min() + eps, size=size).coalesce()
            sii_softplus = torch.sparse_coo_tensor(sii_tensor.indices(), torch.nn.functional.softplus(vvals), size=size).coalesce()
        else:
            sii_shifted = sii_tensor
            sii_softplus = sii_tensor

        if sc_tensor._nnz():
            vvals = sc_tensor.values()
            sc_shifted = torch.sparse_coo_tensor(sc_tensor.indices(), vvals - vvals.min() + eps, size=size).coalesce()
            sc_softplus = torch.sparse_coo_tensor(sc_tensor.indices(), torch.nn.functional.softplus(vvals), size=size).coalesce()
        else:
            sc_shifted = sc_tensor
            sc_softplus = sc_tensor

        vocab = {}

        for col in cols_to_build:
            vocab[f"vocab_{col}"] = vocabs[col]
            vocab[f"{col}2i"] = col2i[col]

        linked_nontrivial = [group for group in linked_groups if len(group) > 1]
        if linked_nontrivial:
            suffix_parts = ["shared" + "".join(map(str, group)) for group in linked_nontrivial]
            suffix = "_" + "_".join(suffix_parts)
        else:
            suffix = ""

        if save:
            torch.save(count_tensor, f"{path_to_tensors}/populated/counting_{top_k}{suffix}.pt")
            torch.save(sii_tensor,   f"{path_to_tensors}/populated/sii_{top_k}{suffix}.pt")
            torch.save(sc_tensor,    f"{path_to_tensors}/populated/sc_{top_k}{suffix}.pt")
            torch.save(sc_softplus,  f"{path_to_tensors}/populated/scSoftPlus_{top_k}{suffix}.pt")
            torch.save(sii_softplus, f"{path_to_tensors}/populated/siiSoftPlus_{top_k}{suffix}.pt")
            torch.save(sc_shifted,   f"{path_to_tensors}/populated/scShifted_{top_k}{suffix}.pt")
            torch.save(sii_shifted,  f"{path_to_tensors}/populated/siiShifted_{top_k}{suffix}.pt")
            with open(f"{path_to_tensors}/vocabularies/{top_k}{suffix}.pkl", "wb") as f:
                pickle.dump(vocab, f)
        else:
            results[top_k] = (count_tensor, sii_tensor, sc_tensor, vocab)
    # notify_discord(f"Finished populating tensors from {path_to_vectors}.")
    return results