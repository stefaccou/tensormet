import multiprocessing
import os
import csv
from tensormet.utils import ThreadBudget, DATA_DIR, voc_index
import random
import numpy as np
from pathlib import Path
import pickle
import ast
from typing import List, Tuple
from tqdm import tqdm
import pyarrow.dataset as ds
import pyarrow.parquet as pq
import torch

def _to_np(x):
    # Accept NumPy arrays or torch tensors; return NumPy view/copy
    if hasattr(x, "detach"):  # torch.Tensor
        return x.detach().cpu().numpy()
    return x

def get_eval_num_threads(fraction: float = 0.75, min_threads: int = 1) -> int:
    """Return n_threads ≈ fraction * available CPUs (at least min_threads)."""
    try:
        n_cores = multiprocessing.cpu_count()
    except NotImplementedError:
        n_cores = os.cpu_count() or 1

    n_threads = max(min_threads, int(n_cores * fraction))
    return n_threads

def load_og_sentences(vector_path, save=False, order=3):
    sentences = set()
    # vector path is a csv file
    pickle_path = vector_path[:-3] + ".pkl"
    if os.path.exists(pickle_path):
        print("loading pickled version")
        with open(pickle_path, "rb") as f:
            sentences = pickle.load(f)
        return sentences
    with open(vector_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        next(reader)  # skip header
        for row in reader:
            sent_id, vector, sentence = row[0], row[1], row[2]
            # the vectors have form "('verb', 'subject', 'object', 'rest', 'rest')"
            # we interpret the vector as a tuple of (verb, subject, object, *rest)
            # vector = eval(vector)
            # v, s, o = vector[0], vector[1], vector[2]
            # sentences.add((v, s, o))
            vector = eval(vector)
            sentences.add(tuple(vector[:order]))
    if save:
        with open(vector_path, "wb") as f:
            pickle.dump(sentences, f)

    return list(sentences)

def softmax(x, temperature=1.0):
    x = x / temperature
    x = x - np.max(x)          # numerical stability
    exp_x = np.exp(x)
    return exp_x / np.sum(exp_x)

# def evaluate_sample(tensor,
#                     sentences,
#                     sampled = True,
#                     n_samples=100,
#                     seed=42,
#                     thread_budget: ThreadBudget | None = None,
#                     return_type="average_rank_score",
#                     softmax_temperature=0.1
#                     ):
#     try:
#         roles = tensor.roles
#     except Exception as e:
#         print(e, "defaulting to hardcoded 3-way vso)")
#         roles = ["verb", "subject", "object"]
#     print("roles in sample:", roles)
#     rank_score = 0
#     prob_score = 0
#     OOV = 0
#     tildes = 0
#     mean = 0
#     std = 0
#     excluded_tilde = 0
#     if not sampled:
#         random.seed(seed)
#         sentences = random.sample(sentences, n_samples)
#     n_samples = len(sentences)
#
#     limiter = (thread_budget.limit() if thread_budget is not None else None)
#     if limiter is None:
#         ctx = contextmanager(lambda: (yield))()  # no-op
#     else:
#         ctx = limiter
#
#     factors_are_torch = hasattr(tensor.factors[0], "cpu")
#
#     with (ctx):
#         for tup in tqdm(sentences):
#             if not tensor.check_vocab(tup):
#                 OOV += 1
#                 continue
#             score = 0
#             sim_score = 0
#             tilde_ex_score = 0
#             non_tilde_count = 0
#             for i, element in enumerate(tup):
#                 role = roles[i]
#                 r2i = voc_index(role)
#                 G_excluded = tensor.excluded_role_vector(tup, role=role)
#
#                 # update to avoid division by 0
#                 F = tensor.factors[i].cpu().numpy() if factors_are_torch else tensor.factors[i]
#
#                 # --- defensive norm computation ---
#                 F_norm = np.linalg.norm(F, axis=1)
#                 G_norm = np.linalg.norm(G_excluded)
#
#                 eps = 1e-12  # safeguard lower bound
#                 F_norm = np.maximum(F_norm, eps)
#                 G_norm = max(G_norm, eps)
#
#                 # --- safe cosine similarities ---
#                 similarities = (F @ G_excluded) / (F_norm * G_norm)
#                 assert similarities.max() <= 1
#                 mean += similarities.mean() / len(tup)
#                 std += similarities.std() / len(tup)
#
#                 idx = tensor.vocab[r2i][element]
#                 sim_i = similarities[idx]
#                 # print(sim_i)
#
#                 # --- rank = number of items >= the true similarity ---
#                 rank = np.sum(similarities >= sim_i)
#
#                 # safeguard rank for extreme cases
#                 if rank == 0:
#                     rank = 1
#                 score += 1.0 / rank
#                 probs = softmax(similarities, temperature=softmax_temperature)
#                 p_i = probs[idx]
#                 sim_score += p_i
#                 if not element == "~":
#                     tilde_ex_score += probs[idx]
#                     non_tilde_count += 1
#                 else:
#                     tildes += 1
#
#             rank_score += score/len(tup)
#             prob_score += sim_score/len(tup)
#             excluded_tilde += tilde_ex_score/non_tilde_count
#
#
#
#     scores = {}
#
#     scores["average_rank_score"] = rank_score / n_samples
#     scores["average_prob_score"] = prob_score / n_samples
#     scores["absolute_rank_score"] = rank_score / (n_samples-OOV)
#     scores["absolute_prob_score"] = prob_score / (n_samples-OOV)
#     scores["OOV"] = OOV
#     scores["OOV_rate"] = OOV/n_samples
#     scores["tilde_excluded_prob_score"] = excluded_tilde/(n_samples-OOV)
#     scores["tilde_rate"] = tildes / ((n_samples-OOV)*(len(roles)-1)) # verb never has ~ elements
#     # todo: implement a scale-aware harmonic mean
#     # scores["harmonic_mean"] = 2 / (scores["absolute_rank_score"] + scores["absolute_prob_score"])
#     print(f"Average expected role vector rank score over {n_samples} samples: {scores['average_rank_score']}, "
#           f"Average prob score: {scores['average_prob_score']}")
#     print(f"\tWithout {OOV} OOV: rank {scores['absolute_rank_score']}, prob {scores['absolute_prob_score']} "
#           f"- without {tildes}'~': {scores['tilde_excluded_prob_score']}")
#     # make scores JSON-safe (numpy scalars -> python scalars)
#     for k, v in list(scores.items()):
#         if isinstance(v, np.generic):  # np.float32, np.int64, ...
#             scores[k] = v.item()  # -> python float/int
#
#     if return_type == "all":
#         return scores
#
#     if isinstance(return_type, (list, tuple)):
#         missing = [k for k in return_type if k not in scores]
#         if missing:
#             raise NotImplementedError(f"Unknown score keys: {missing}. Choose from {list(scores.keys())}")
#         return {k: scores[k] for k in return_type}
#
#     if isinstance(return_type, str):
#         if return_type in scores:
#             return scores[return_type]
#         raise NotImplementedError(f"Choose one of {list(scores.keys())} or pass a list/tuple or 'all'.")
#
#     raise TypeError("return_type must be a string key, a list/tuple of keys, or 'all'.")

def evaluate_sample(tensor,
                    sentences,
                    sampled=True,
                    n_samples=100,
                    seed=42,
                    thread_budget: ThreadBudget | None = None,
                    return_type="average_rank_score",
                    softmax_temperature=0.1,
                    batch_size=512,
                    show_progress=True  # Toggle for tqdm
                    ):
    device = tensor.factors[0].device
    try:
        roles = tensor.roles
    except:
        roles = ["verb", "subject", "object"]

    if not sampled:
        random.seed(seed)
        sentences = random.sample(sentences, n_samples)

    n_samples = len(sentences)
    n_roles = len(roles)
    eps = 1e-12

    # 1. Optimized CPU Mapping
    valid_indices = []
    clean_tuples = []
    for tup in sentences:
        if tensor.check_vocab(tup):
            # Map role names to vocab indices
            idx_list = [tensor.vocab[voc_index(roles[i])][tup[i]] for i in range(n_roles)]
            valid_indices.append(idx_list)
            clean_tuples.append(tup)

    num_valid = len(valid_indices)
    if num_valid == 0:
        return {"OOV": n_samples, "OOV_rate": 1.0}

    # Move indices to GPU once
    indices_tensor = torch.tensor(valid_indices, device=device)

    # Accumulators
    rank_score = 0.0
    prob_score = 0.0
    excluded_tilde = 0.0
    tildes = 0

    # 2. Role-wise Processing
    # Main loop over roles (outer progress bar if multiple roles)
    role_pbar = tqdm(roles, desc="Total Progress", disable=not show_progress)

    for i, role_name in enumerate(role_pbar):
        F = tensor.factors[i]
        F_norm = torch.norm(F, p=2, dim=1, keepdim=True).clamp(min=eps)

        # Fine-grained sentence-level progress bar
        sentence_pbar = tqdm(total=num_valid,
                             desc=f"Evaluating {role_name[:10]}",
                             leave=False,
                             disable=not show_progress)

        for start in range(0, num_valid, batch_size):
            end = min(start + batch_size, num_valid)
            b_idx = indices_tensor[start:end]
            curr_batch_len = end - start

            # GPU Prediction (Tucker contraction)
            G_batch = tensor.batch_excluded_role_vector(b_idx, role_name)
            G_norm = torch.norm(G_batch, p=2, dim=1, keepdim=True).clamp(min=eps)

            # Cosine Similarity via Batched Matrix Multiplication
            # [Batch, Rank] @ [Rank, Vocab] -> [Batch, Vocab]
            similarities = torch.mm(G_batch, F.t()) / torch.mm(G_norm, F_norm.t())
            similarities = torch.clamp(similarities, -1.0, 1.0)

            # Target score extraction
            targets = b_idx[:, i]
            sim_target = similarities[torch.arange(curr_batch_len), targets].view(-1, 1)

            # Vectorized Rank Calculation (Parallel count on GPU)
            ranks = torch.sum(similarities >= sim_target, dim=1)
            ranks = torch.clamp(ranks, min=1)
            rank_score += torch.sum(1.0 / ranks.float()).item() / n_roles

            # Softmax Probability
            probs = torch.softmax(similarities / softmax_temperature, dim=1)
            p_target = probs[torch.arange(curr_batch_len), targets]
            prob_score += torch.sum(p_target).item() / n_roles

            # CPU Tilde Check (Minimal data transfer)
            p_target_cpu = p_target.detach().cpu().numpy()
            for j in range(curr_batch_len):
                if clean_tuples[start + j][i] == "~":
                    tildes += 1
                else:
                    excluded_tilde += p_target_cpu[j] / n_roles

            sentence_pbar.update(curr_batch_len)
        sentence_pbar.close()

    # 3. Assemble Final Metrics
    OOV = n_samples - num_valid
    scores = {
        "average_rank_score": rank_score / n_samples,
        "average_prob_score": prob_score / n_samples,
        "absolute_rank_score": rank_score / num_valid,
        "absolute_prob_score": prob_score / num_valid,
        "OOV": OOV,
        "OOV_rate": OOV / n_samples,
        "tilde_excluded_prob_score": excluded_tilde / num_valid,
        "tilde_rate": tildes / (num_valid * n_roles) if n_roles > 0 else 0.0
    }
    # Format numeric types for output
    for k, v in scores.items():
        if isinstance(v, (np.generic, torch.Tensor)):
            scores[k] = v.item()

    if return_type == "all":
        return scores
    if isinstance(return_type, (list, tuple)):
        return {k: scores[k] for k in return_type if k in scores}
    return scores.get(return_type, scores["average_rank_score"])

def softmax_batch(x, temperature=1.0):
    x = x / temperature
    x_max = np.max(x, axis=1, keepdims=True)
    exp_x = np.exp(x - x_max)
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)


def load_eval_sentences_cached(
    vector_path: str | os.PathLike,
    *,
    dataset: str,
    roles: List[str] = ("verb", "subject", "object"),
    cache_dir: str | os.PathLike = DATA_DIR / "vectors" / "cache",
    n_samples: int = 100,
    seed: int = 42,
) -> List[Tuple[str, ...]]:
    """
    Parse unique tuples from `vector_path`, using the number of elements implied by `roles`,
    then sample deterministically and cache the sampled list.

    The CSV is expected to contain a tuple-like object in row[1].
    Only the first `len(roles)` elements are used.
    """
    vector_path = Path(vector_path)
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    tuple_len = len(roles)
    if tuple_len == 0:
        raise ValueError("roles must contain at least one role")

    st = vector_path.stat()
    cache_name = (
        f"eval_sentences__dataset={dataset}"
        f"__roles={','.join(roles)}"
        f"__n={n_samples}"
        f"__seed={seed}"
        f"__csv_mtime_ns={st.st_mtime_ns}"
        f"__csv_bytes={st.st_size}"
        f".pkl"
    )
    cache_file = cache_dir / cache_name

    if cache_file.exists():
        with cache_file.open("rb") as f:
            return pickle.load(f)

    sentences: set[Tuple[str, ...]] = set()
    with vector_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        next(reader, None)
        for row in reader:
            if len(row) < 2:
                continue

            vec = ast.literal_eval(row[1])
            if len(vec) < tuple_len:
                continue

            tup = tuple(vec[:tuple_len])
            sentences.add(tup)

    sentences_list = list(sentences)

    rng = random.Random(seed)
    if n_samples > len(sentences_list):
        raise ValueError(
            f"n_samples={n_samples} > number of unique tuples={len(sentences_list)} "
            f"parsed from {vector_path} for roles={roles}"
        )

    sampled_sentences = rng.sample(sentences_list, n_samples)

    tmp = cache_file.with_suffix(".tmp")
    with tmp.open("wb") as f:
        pickle.dump(sampled_sentences, f, protocol=pickle.HIGHEST_PROTOCOL)
    tmp.replace(cache_file)

    return sampled_sentences

def load_eval_sentences_cached_parquet(
    vector_path: str | os.PathLike,
    *,
    dataset: str,
    roles: List[str] = ("root", "nsubj", "obj"),
    column_map=None,
    cache_dir: str | os.PathLike = DATA_DIR / "vectors" / "cache",
    n_samples: int = 100,
    seed: int = 42,
    oversample: int = 3,   # collect this many × n_samples before final sampling
) -> List[Tuple[str, ...]]:
    """
    Stream unique tuples from a Parquet file or directory, sampling lazily
    so we never read more than needed.
    """
    vector_path = Path(vector_path)
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    # --- cache fingerprint (unchanged) ---
    if vector_path.is_dir():
        files = sorted(vector_path.rglob("*.parquet"))
        if not files:
            raise FileNotFoundError(f"No parquet files found under {vector_path}")
        newest_mtime_ns = max(p.stat().st_mtime_ns for p in files)
        total_bytes = sum(p.stat().st_size for p in files)
        fingerprint = f"dir_mtime_ns={newest_mtime_ns}__bytes={total_bytes}__nfiles={len(files)}"
    else:
        st = vector_path.stat()
        fingerprint = f"file_mtime_ns={st.st_mtime_ns}__bytes={st.st_size}"
        files = [vector_path]

    cache_name = (
        f"eval_sentences__dataset={dataset}"
        f"__roles={','.join(roles)}"
        f"__n={n_samples}"
        f"__seed={seed}"
        f"__{fingerprint}"
        f".pkl"
    )
    cache_file = cache_dir / cache_name

    # --- fast path ---
    if cache_file.exists():
        with cache_file.open("rb") as f:
            return pickle.load(f)

    # --- slow path: stream row groups until we have enough ---
    rng = random.Random(seed)
    roles = list(roles)

    # Enumerate (file, row_group_index) pairs, then shuffle for an unbiased sample.
    row_groups = []
    for fp in files:
        pf = pq.ParquetFile(fp)
        for rg_idx in range(pf.num_row_groups):
            row_groups.append((fp, rg_idx))
    rng.shuffle(row_groups)

    target = n_samples * oversample
    seen: set = set()
    collected: list = []

    # Cache open ParquetFile handles so we don't reopen per row group
    open_files: dict = {}

    for fp, rg_idx in row_groups:
        pf = open_files.get(fp)
        if pf is None:
            pf = pq.ParquetFile(fp)
            open_files[fp] = pf

        batch = pf.read_row_group(rg_idx, columns=roles).drop_null()
        cols = [batch[r].to_pylist() for r in roles]

        for tup in zip(*cols):
            if tup not in seen:
                seen.add(tup)
                collected.append(tup)

        if len(collected) >= target:
            break

    if len(collected) < n_samples:
        raise ValueError(
            f"n_samples={n_samples} > number of unique tuples found={len(collected)} "
            f"after scanning {len(row_groups)} row groups"
        )

    sampled = rng.sample(collected, n_samples)

    tmp = cache_file.with_suffix(".tmp")
    with tmp.open("wb") as f:
        pickle.dump(sampled, f, protocol=pickle.HIGHEST_PROTOCOL)
    tmp.replace(cache_file)

    return sampled

def ensure_vocab(vocab, sample, roles):
    legacy_map = {"verb": "v", "subject": "s", "object": "o"}
    clean_sample = []
    for vector in tqdm(sample):
        cleaned = []
        for i, element in enumerate(vector):
            role = roles[i]
            candidates = [f"vocab_{role}"]
            if role in legacy_map:
                candidates.append(f"vocab_{legacy_map[role]}")

            vocab_key = next((k for k in candidates if k in vocab), None)
            if vocab_key is None:
                raise KeyError(f"No vocab key found for role {role!r}")

            cleaned.append(element if element in vocab[vocab_key] else "~")
        clean_sample.append(tuple(cleaned))
    return clean_sample