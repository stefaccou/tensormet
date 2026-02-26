import multiprocessing
import os
import csv
from tensormet.utils import ThreadBudget, DATA_DIR
import random
from contextlib import contextmanager
import numpy as np
from pathlib import Path
import pickle
import ast
from typing import List, Tuple
from tqdm import tqdm
import pyarrow.dataset as ds

def get_eval_num_threads(fraction: float = 0.75, min_threads: int = 1) -> int:
    """Return n_threads ≈ fraction * available CPUs (at least min_threads)."""
    try:
        n_cores = multiprocessing.cpu_count()
    except NotImplementedError:
        n_cores = os.cpu_count() or 1

    n_threads = max(min_threads, int(n_cores * fraction))
    return n_threads

def load_og_sentences(vector_path, save=False):
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
            vector = eval(vector)
            v, s, o = vector[0], vector[1], vector[2]
            sentences.add((v, s, o))
    if save:
        with open(vector_path, "wb") as f:
            pickle.dump(sentences, f)

    return list(sentences)

def softmax(x, temperature=1.0):
    x = x / temperature
    x = x - np.max(x)          # numerical stability
    exp_x = np.exp(x)
    return exp_x / np.sum(exp_x)

def evaluate_sample(tensor,
                    sentences,
                    sampled = True,
                    n_samples=100,
                    seed=42,
                    thread_budget: ThreadBudget | None = None,
                    return_type="average_rank_score",
                    softmax_temperature=0.1
                    ):
    rank_score = 0
    prob_score = 0
    OOV = 0
    mean = 0
    std = 0
    if not sampled:
        random.seed(seed)
        sentences = random.sample(sentences, n_samples)
    n_samples = len(sentences)

    limiter = (thread_budget.limit() if thread_budget is not None else None)
    if limiter is None:
        ctx = contextmanager(lambda: (yield))()  # no-op
    else:
        ctx = limiter

    with (ctx):
        for tup in sentences:
            if not tensor.check_vocab(tup):
                OOV += 1
                continue
            score = 0
            sim_score = 0

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
                assert similarities.max() <= 1
                mean += similarities.mean() / len(tup)
                std += similarities.std() / len(tup)

                idx = tensor.vocab[r2i][element]
                sim_i = similarities[idx]
                # print(sim_i)

                # --- rank = number of items >= the true similarity ---
                rank = np.sum(similarities >= sim_i)

                # safeguard rank for extreme cases
                if rank == 0:
                    rank = 1
                score += 1.0 / rank
                probs = softmax(similarities, temperature=softmax_temperature)
                p_i = probs[idx]
                sim_score += p_i

            rank_score += score/len(tup)
            prob_score += sim_score/len(tup)



    scores = {}

    scores["average_rank_score"] = rank_score / n_samples
    scores["average_prob_score"] = prob_score / n_samples
    scores["absolute_rank_score"] = rank_score / (n_samples-OOV)
    scores["absolute_prob_score"] = prob_score / (n_samples-OOV)
    scores["OOV"] = OOV
    scores["OOV_rate"] = OOV/n_samples
    # todo: implement a scale-aware harmonic mean
    # scores["harmonic_mean"] = 2 / (scores["absolute_rank_score"] + scores["absolute_prob_score"])
    print(f"Average expected role vector rank score over {n_samples} samples: {scores['average_rank_score']}, "
          f"Average prob score: {scores['average_prob_score']}")
    print(f"\tWithout {OOV} OOV: rank {scores['absolute_rank_score']}, prob {scores['absolute_prob_score']}")
    # print(f"Harmonic mean between absolutes: {scores['harmonic_mean']}")
    # print(f"Average mean {mean / (n_samples-OOV)}, std {std / (n_samples-OOV)}")
    if return_type in scores.keys():
        return scores[return_type]
    else:
        raise NotImplementedError(f"Choose one of {scores.keys()}")

def load_eval_sentences_cached(
    vector_path: str | os.PathLike,
    *,
    dataset: str,
    cache_dir: str | os.PathLike = DATA_DIR / "vectors" / "cache",
    n_samples: int = 100,
    seed: int = 42,
) -> List[Tuple[str, str, str]]:
    """
    Parse (verb, subject, object) triples from `vector_path`, then sample exactly the
    same way as `evaluate_sample` (random.sample with `seed`), and cache the sampled list.

    Cache filename is deterministic + interpretable and also includes file mtime/size
    so it naturally invalidates when the CSV changes.
    """
    vector_path = Path(vector_path)
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    st = vector_path.stat()
    cache_name = (
        f"eval_sentences__dataset={dataset}"
        f"__n={n_samples}"
        f"__seed={seed}"
        f"__csv_mtime_ns={st.st_mtime_ns}"
        f"__csv_bytes={st.st_size}"
        f".pkl"
    )
    cache_file = cache_dir / cache_name

    # Fast path: cache hit
    if cache_file.exists():
        with cache_file.open("rb") as f:
            return pickle.load(f)

    # Slow path: parse uniques
    sentences: set[Tuple[str, str, str]] = set()
    with vector_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        next(reader, None)  # skip header safely
        for row in reader:
            if len(row) < 2:
                continue
            vec = ast.literal_eval(row[1])  # safer than eval
            if len(vec) < 3:
                continue
            v, s, o = vec[0], vec[1], vec[2]
            sentences.add((v, s, o))

    sentences_list = list(sentences)

    # Match evaluate_sample's sampling behavior
    rng = random.Random(seed)
    if n_samples > len(sentences_list):
        raise ValueError(
            f"n_samples={n_samples} > number of unique triples={len(sentences_list)} "
            f"parsed from {vector_path}"
        )
    sampled_sentences = rng.sample(sentences_list, n_samples)

    # Atomic write
    tmp = cache_file.with_suffix(".tmp")
    with tmp.open("wb") as f:
        pickle.dump(sampled_sentences, f, protocol=pickle.HIGHEST_PROTOCOL)
    tmp.replace(cache_file)

    return sampled_sentences

# parquet-based pipeline
def load_eval_sentences_cached_parquet(
    vector_path: str | os.PathLike,
    *,
    dataset: str,
    cache_dir: str | os.PathLike = DATA_DIR / "vectors" / "cache",
    n_samples: int = 100,
    seed: int = 42,
) -> List[Tuple[str, str, str]]:
    """
    Read unique (root, nsubj, obj) triples from a Parquet file or Parquet directory,
    sample deterministically (like evaluate_sample), and cache the sampled list.

    Supports:
      - a single .parquet file
      - a directory containing many parquet parts (recommended)
    """
    vector_path = Path(vector_path)
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Build a cache key that invalidates when the parquet changes
    # - For a directory, use the newest mtime and total size across files.
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

    cache_name = (
        f"eval_sentences__dataset={dataset}"
        f"__n={n_samples}"
        f"__seed={seed}"
        f"__{fingerprint}"
        f".pkl"
    )
    cache_file = cache_dir / cache_name

    # Fast path
    if cache_file.exists():
        with cache_file.open("rb") as f:
            return pickle.load(f)

    # Slow path: scan parquet
    dataset_obj = ds.dataset(str(vector_path), format="parquet")

    # Pull only the 3 columns we care about; this is the big win vs CSV

    # to_table() will materialize these 3 columns. If the parquet is huge and you
    # expect very high cardinality, see the streaming option below.
    table = dataset_obj.to_table(columns=["root", "nsubj", "obj"])

    table = table.drop_null()

    triples = list(zip(table["root"].to_pylist(),
                       table["nsubj"].to_pylist(),
                       table["obj"].to_pylist()))

    if n_samples > len(triples):
        raise ValueError(
            f"n_samples={n_samples} > number of unique triples={len(triples)} "
            f"parsed from {vector_path}"
        )

    rng = random.Random(seed)
    sampled = rng.sample(triples, n_samples)

    tmp = cache_file.with_suffix(".tmp")
    with tmp.open("wb") as f:
        pickle.dump(sampled, f, protocol=pickle.HIGHEST_PROTOCOL)
    tmp.replace(cache_file)

    return sampled


voc_dict = {0:"v", 1:"s", 2:"o"}
def ensure_vocab(vocab, sample):
    clean_sample = []
    for vector in tqdm(sample):
        cleaned = []
        for i, element in enumerate(vector):
            vocab_key = f"vocab_{voc_dict[i]}"
            if element in vocab[vocab_key]:
                cleaned.append(element)
            else:
                cleaned.append("~")
        clean_sample.append(tuple(cleaned))
    return clean_sample
