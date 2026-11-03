# tensormet/vector_creation.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from collections import Counter, defaultdict
from typing import Callable, Dict, Iterable, Iterator, Optional, Tuple
from itertools import islice

import json
import time
import hashlib
import signal
import multiprocessing as mp
from collections import deque
import numpy as np
import string
from tqdm import tqdm

import pyarrow as pa
import pyarrow.parquet as pq
import spacy
from datasets import load_dataset

from tensormet.utils import DATA_DIR, compute_num_threads
from tensormet.config import VectorRunConfig, parse_ngram_order, parse_ngram_orders, parse_raw_ngram_orders

import gc


SCHEMA = pa.schema([
    ("sent_id", pa.int64()),
    ("root", pa.string()),
    ("nsubj", pa.string()),
    ("obj", pa.string()),
    ("obj2", pa.string()),
    ("obl", pa.string()),
    ("adj_nsubj", pa.string()),
    ("adj_obj", pa.string()),
])




# ---------------------------------------------------------------------------
# PARQUET I/O
# ---------------------------------------------------------------------------

def _part_path(output_dir: Path, part_id: int) -> Path:
    return output_dir / f"part-{part_id:06d}.parquet"


def _open_part_writer(output_dir: Path, part_id: int, SCHEMA=SCHEMA) -> pq.ParquetWriter:
    part_path = _part_path(output_dir, part_id)
    # Exclusive create: never overwrite shards by accident
    f = part_path.open("xb")
    return pq.ParquetWriter(f, SCHEMA, compression="zstd")


def flush_parquet(writer: pq.ParquetWriter, buffer_rows: list[dict], SCHEMA=SCHEMA) -> int:
    if not buffer_rows:
        return 0
    table = pa.Table.from_pylist(buffer_rows, schema=SCHEMA)
    writer.write_table(table)
    return table.num_rows


def extract_vectors(
    doc,
    start_sent_id: int,
    DEP_NSUBJ: int,
    DEP_OBJ: int,
    DEP_IOBJ: int,
    DEP_OBL: int,
    DEP_AMOD: int,
    POS_VERB: int,
) -> tuple[list[dict], int, int]:
    rows: list[dict] = []
    sent_id = start_sent_id

    for sent in doc.sents:
        root = sent.root
        if root.pos != POS_VERB:
            sent_id += 1
            continue

        nsubj = obj = obj2 = obl = "~"
        adj_nsubj = adj_obj = "~"
        filled = 0

        for child in root.children:
            d = child.dep

            if d == DEP_NSUBJ and nsubj == "~":
                nsubj = child.lemma_
                for grand_child in child.children:
                    if grand_child.dep == DEP_AMOD:
                        adj_nsubj = grand_child.lemma_
                        break
                filled += 1


            elif d == DEP_OBJ:
                if obj == "~":
                    obj = child.lemma_
                    for grand_child in child.children:
                        if grand_child.dep == DEP_AMOD:
                            adj_obj = grand_child.lemma_
                            break
                    filled += 1
                elif obj2 == "~":
                    obj2 = child.lemma_
                    filled += 1

            elif d == DEP_IOBJ and obj2 == "~":
                obj2 = child.lemma_
                filled += 1

            elif d == DEP_OBL and obl == "~":
                obl = child.lemma_
                filled += 1

            if filled == 4:
                break

        rows.append({
            "sent_id": sent_id,
            "root": root.lemma_,
            "nsubj": nsubj,
            "obj": obj,
            "obj2": obj2,
            "obl": obl,
            "adj_nsubj": adj_nsubj,
            "adj_obj": adj_obj,
        })
        sent_id += 1

    return rows, len(rows), sent_id






def create_vectors_parquet_sharded(
    cfg: VectorRunConfig,
    *,
    overwrite: bool = False,
) -> dict:
    """
    Sharded Parquet output for safe resumes:
      - output_dir/part-xxxxxx.parquet
      - output_dir/_meta.json checkpoint
    Returns a small summary dict.
    """
    output_dir = cfg.output_dir()
    output_dir.mkdir(parents=True, exist_ok=True)
    meta_path = output_dir / "_meta.json"

    if overwrite:
        # Conservative overwrite: only allowed if directory exists and user requested it.
        # Remove meta first; shards remain unless you also delete them explicitly.
        # If you prefer full wipe, replace with shutil.rmtree(output_dir) then mkdir.
        if meta_path.exists():
            meta_path.unlink()

    meta: dict = {}
    if meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))

    vector_count = int(meta.get("vector_count", 0))
    global_sent_id = int(meta.get("sent_id", 0))
    raw_seen = int(meta.get("raw_seen", 0))

    part_id = int(meta.get("part_id", 0))
    part_rows = int(meta.get("part_rows", 0))

    # If meta says we're mid-part but file is missing, start a new part.
    if part_rows > 0 and not _part_path(output_dir, part_id).exists():
        part_id += 1
        part_rows = 0

    # ---- spaCy ----
    nlp = spacy.load(cfg.exp.spacy_model, disable=["ner", "textcat"])
    nlp.max_length = 1_000_000
    n_process = max(1, compute_num_threads(cfg.exp.cpu_frac))

    S = nlp.vocab.strings
    DEP_NSUBJ = S["nsubj"]
    DEP_OBJ = S["dobj"]
    DEP_IOBJ = S["dative"]
    DEP_OBL = S["obl"]
    DEP_AMOD = S["amod"]
    POS_VERB = S["VERB"]


    # ---- dataset stream ----
    ds_kwargs = dict(split=cfg.hf.split, streaming=True)
    ds = (
        load_dataset(cfg.hf.path, cfg.hf.config, **ds_kwargs)
        if cfg.hf.config is not None
        else load_dataset(cfg.hf.path, **ds_kwargs)
    )

    if raw_seen > 0:
        print(
            f"Resuming: vectors={vector_count:,} | sent_id={global_sent_id:,} | "
            f"raw_seen={raw_seen:,} | part={part_id} rows_in_part={part_rows:,}"
        )
        ds = ds.skip(raw_seen)

    def gen_texts() -> Iterator[str]:
        nonlocal raw_seen
        for ex in ds:
            raw_seen += 1
            txt = ex.get(cfg.hf.text_column)
            if not txt:
                continue
            if len(txt) > cfg.exp.max_text_length:
                continue
            yield txt

    def save_meta() -> None:
        meta_path.write_text(
            json.dumps(
                {
                    "vector_count": vector_count,
                    "sent_id": global_sent_id,
                    "raw_seen": raw_seen,
                    "part_id": part_id,
                    "part_rows": part_rows,
                }
            ),
            encoding="utf-8",
        )

    buffer: list[dict] = []
    start_time = time.time()
    last_log_time = start_time

    # writer = _open_part_writer(output_dir, part_id)
    writer = None
    docs = None

    def chunked(iterable, size):
        it = iter(iterable)
        while True:
            batch = list(islice(it, size))
            if not batch:
                return
            yield batch
    try:
        for text_batch in chunked(gen_texts(), cfg.exp.batch_size * 8):
            docs = nlp.pipe(text_batch, batch_size=cfg.exp.batch_size, n_process=n_process)
            for doc in docs:
                rows, count, global_sent_id = extract_vectors(
                    doc,
                    global_sent_id,
                    DEP_NSUBJ,
                    DEP_OBJ,
                    DEP_IOBJ,
                    DEP_OBL,
                    DEP_AMOD,
                    POS_VERB,
                )

                if count:
                    buffer.extend(rows)

                if len(buffer) >= cfg.exp.rows_per_flush:
                    if writer is None:
                        writer = _open_part_writer(output_dir, part_id)
                    wrote = flush_parquet(writer, buffer)
                    buffer.clear()

                    vector_count += wrote
                    part_rows += wrote

                    if part_rows >= cfg.exp.rows_per_part:
                        writer.close()
                        part_id += 1
                        part_rows = 0
                        writer = _open_part_writer(output_dir, part_id)

                    save_meta()

                now = time.time()
                if now - last_log_time >= cfg.exp.log_every_s:
                    elapsed = now - start_time
                    vps = vector_count / elapsed if elapsed > 0 else 0.0
                    remaining = max(cfg.exp.target_vectors - vector_count, 0)
                    eta_seconds = remaining / vps if vps > 0 else float("inf")

                    print(
                        f"[{elapsed:,.1f}s] raw_seen={raw_seen:,} | sentences={global_sent_id:,} | "
                        f"vectors={vector_count:,} | vec/s={vps:,.0f} | "
                        f"ETA={eta_seconds/3600:,.2f}h ({eta_seconds/60:,.1f}m) | "
                        f"buffer={len(buffer):,} | part={part_id} part_rows={part_rows:,}"
                    )
                    last_log_time = now

                if vector_count >= cfg.exp.target_vectors:
                    break

    except KeyboardInterrupt:
        pass
    finally:
        if buffer:
            if writer is None:
                writer = _open_part_writer(output_dir, part_id)
            wrote = flush_parquet(writer, buffer)
            vector_count += wrote
            part_rows += wrote
            buffer.clear()

        if writer is not None:
            writer.close()
        save_meta()
        docs = None
        gc.collect()

    print(f"Vectors written: {vector_count} (dir: {output_dir})")
    print(f"Checkpoint saved: {meta_path}")

    return {
        "output_dir": str(output_dir),
        "meta_path": str(meta_path),
        "vectors_written": int(vector_count),
        "sent_id": int(global_sent_id),
        "raw_seen": int(raw_seen),
        "part_id": int(part_id),
        "part_rows": int(part_rows),
        "hf_path": cfg.hf.path,
        "hf_config": cfg.hf.config,
        "hf_split": cfg.hf.split,
        "hf_text_column": cfg.hf.text_column,
    }





# ---------------------------------------------------------------------------
# MAIN PIPELINE
# ---------------------------------------------------------------------------
def create_frame_vectors_parquet_sharded(
        cfg: VectorRunConfig,
        *,
        overwrite: bool = False,
) -> dict:
    from frame_semantic_transformer import FrameSemanticTransformer  # This now needs a custom environment!
    from nltk.corpus import framenet as fn

    output_dir = cfg.output_dir()
    output_dir.mkdir(parents=True, exist_ok=True)
    meta_path = output_dir / "_meta.json"

    if overwrite and meta_path.exists():
        meta_path.unlink()

    meta: dict = {}
    if meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))

    vector_count = int(meta.get("vector_count", 0))
    global_sent_id = int(meta.get("sent_id", 0))
    raw_seen = int(meta.get("raw_seen", 0))
    part_id = int(meta.get("part_id", 0))
    part_rows = int(meta.get("part_rows", 0))

    if part_rows > 0 and not _part_path(output_dir, part_id).exists():
        part_id += 1
        part_rows = 0

    # ---- NLP Setup ----
    # Disable parser/ner for speed, add sentencizer for fast boundary detection
    nlp = spacy.load(cfg.exp.spacy_model, disable=["ner", "parser", "textcat"])
    nlp.add_pipe("sentencizer")
    nlp.max_length = 1_000_000

    print("Initializing FrameSemanticTransformer...")
    frame_transformer = FrameSemanticTransformer(batch_size=cfg.exp.batch_size)
    frame_transformer.setup()

    canonical_orders = get_canonical_orders_cached(total_slots=6)

    # ---- Dataset Stream ----
    ds_kwargs = dict(split=cfg.hf.split, streaming=True)
    ds = (
        load_dataset(cfg.hf.path, cfg.hf.config, **ds_kwargs)
        if cfg.hf.config is not None
        else load_dataset(cfg.hf.path, **ds_kwargs)
    )

    if raw_seen > 0:
        print(
            f"Resuming: vectors={vector_count:,} | sent_id={global_sent_id:,} | "
            f"raw_seen={raw_seen:,} | part={part_id} rows_in_part={part_rows:,}"
        )
        ds = ds.skip(raw_seen)

    def gen_texts() -> Iterator[str]:
        nonlocal raw_seen
        for ex in ds:
            raw_seen += 1
            txt = ex.get(cfg.hf.text_column)
            if not txt or len(txt) > cfg.exp.max_text_length:
                continue
            yield txt

    def save_meta() -> None:
        meta_path.write_text(
            json.dumps({
                "vector_count": vector_count,
                "sent_id": global_sent_id,
                "raw_seen": raw_seen,
                "part_id": part_id,
                "part_rows": part_rows,
            }),
            encoding="utf-8",
        )

    def chunked(iterable, size):
        it = iter(iterable)
        while True:
            batch = list(islice(it, size))
            if not batch:
                return
            yield batch

    buffer: list[dict] = []
    start_time = time.time()
    last_log_time = start_time
    writer = None

    try:
        # Batch heavily to leverage FST's bulk processing
        for text_batch in chunked(gen_texts(), cfg.exp.batch_size * 4):
            batch_sents = []
            batch_sent_ids = []

            # 1. Chunk documents into sentences
            for doc in nlp.pipe(text_batch):
                for sent in doc.sents:
                    batch_sents.append(sent.text)
                    batch_sent_ids.append(global_sent_id)
                    global_sent_id += 1

            if not batch_sents:
                continue

            # 2. Extract frames in bulk
            fst_results = frame_transformer.detect_frames_bulk(batch_sents)

            # 3. Process outputs & build vectors
            for result, sid in zip(fst_results, batch_sent_ids):
                for frame in result.frames:
                    # Target Extraction
                    target_text = "~"
                    if hasattr(frame, "trigger_location") and frame.trigger_location is not None:
                        start_idx = frame.trigger_location
                        remainder = result.sentence[start_idx:]
                        if remainder:
                            target_text = remainder.split()[0].strip(string.punctuation)

                    target_lemma = extract_core_lemmas(target_text, nlp)

                    # FE Extraction
                    instance_elements = {
                        el.name: extract_core_lemmas(el.text, nlp)
                        for el in frame.frame_elements
                    }

                    ordered_keys = canonical_orders.get(frame.name, [])

                    row = {
                        "sent_id": sid,
                        "frame_name": frame.name,
                        "target": target_lemma,
                        "arg1": "~", "arg2": "~", "arg3": "~", "arg4": "~", "arg5": "~"
                    }

                    # Fill padded arguments
                    for i, key in enumerate(ordered_keys[:5]):
                        if not key.startswith("EMPTY"):
                            row[f"arg{i + 1}"] = instance_elements.get(key, "~")

                    buffer.append(row)

            # 4. Flush to Parquet if buffer full
            if len(buffer) >= cfg.exp.rows_per_flush:
                if writer is None:
                    writer = _open_part_writer(output_dir, part_id, SCHEMA=FRAME_SCHEMA)
                wrote = flush_parquet(writer, buffer, SCHEMA=FRAME_SCHEMA)
                buffer.clear()

                vector_count += wrote
                part_rows += wrote

                if part_rows >= cfg.exp.rows_per_part:
                    writer.close()
                    part_id += 1
                    part_rows = 0
                    writer = _open_part_writer(output_dir, part_id, SCHEMA=FRAME_SCHEMA)

                save_meta()

            # 5. Logging
            now = time.time()
            if now - last_log_time >= cfg.exp.log_every_s:
                elapsed = now - start_time
                vps = vector_count / elapsed if elapsed > 0 else 0.0
                remaining = max(cfg.exp.target_vectors - vector_count, 0)
                eta_seconds = remaining / vps if vps > 0 else float("inf")

                print(
                    f"[{elapsed:,.1f}s] raw_seen={raw_seen:,} | sentences={global_sent_id:,} | "
                    f"vectors={vector_count:,} | vec/s={vps:,.0f} | "
                    f"ETA={eta_seconds / 3600:,.2f}h ({eta_seconds / 60:,.1f}m) | "
                    f"buffer={len(buffer):,} | part={part_id} part_rows={part_rows:,}"
                )
                last_log_time = now

            if vector_count >= cfg.exp.target_vectors:
                break

    except KeyboardInterrupt:
        print("\nGracefully interrupting and flushing buffer...")
    finally:
        if buffer:
            if writer is None:
                writer = _open_part_writer(output_dir, part_id, SCHEMA=FRAME_SCHEMA)
            wrote = flush_parquet(writer, buffer, SCHEMA=FRAME_SCHEMA)
            vector_count += wrote
            part_rows += wrote
            buffer.clear()

        if writer is not None:
            writer.close()
        save_meta()
        gc.collect()

    print(f"Vectors written: {vector_count} (dir: {output_dir})")
    print(f"Checkpoint saved: {meta_path}")

    return {
        "output_dir": str(output_dir),
        "meta_path": str(meta_path),
        "vectors_written": int(vector_count),
        "sent_id": int(global_sent_id),
        "raw_seen": int(raw_seen),
        "part_id": int(part_id),
        "part_rows": int(part_rows),
        "hf_path": cfg.hf.path,
        "hf_config": cfg.hf.config,
        "hf_split": cfg.hf.split,
        "hf_text_column": cfg.hf.text_column,
    }

# ---------------------------------------------------------------------------
# FRAME_SCHEMA DEFINITION
# ---------------------------------------------------------------------------
FRAME_SCHEMA = pa.schema([
    ("sent_id", pa.int64()),
    ("frame_name", pa.string()),
    ("target", pa.string()),
    ("arg1", pa.string()),
    ("arg2", pa.string()),
    ("arg3", pa.string()),
    ("arg4", pa.string()),
    ("arg5", pa.string()),
])


# ---------------------------------------------------------------------------
# FRAMENET BLUEPRINT & CANONICAL ORDER LOGIC
# see 10_scaleup/frames/Frame-based tagging.ipynb for development notes
# ---------------------------------------------------------------------------
def _get_fe_type(fe) -> str:
    for attr in ("coreType", "core_type", "type"):
        if hasattr(fe, attr):
            value = getattr(fe, attr)
            if value is not None:
                return str(value)
    return "Unknown"


def _is_core_fe(fe) -> bool:
    return _get_fe_type(fe).lower().startswith("core")


def _fe_priority(fe_type: str) -> int:
    t = fe_type.lower()
    if t == "core": return 0
    if t == "core-unexpressed": return 1
    if t == "extra-thematic": return 2
    if t == "peripheral": return 3
    return 4


def _fe_frequency_in_examples(frame) -> Counter:
    counts = Counter()
    lex_units = getattr(frame, "lexUnit", {}) or {}
    for _, lu in lex_units.items():
        exemplars = getattr(lu, "exemplars", []) or []
        for ex in exemplars:
            ex_fe = getattr(ex, "FE", {}) or {}
            if isinstance(ex_fe, dict):
                for fe_name, spans in ex_fe.items():
                    if spans:
                        counts[fe_name] += 1
    return counts


def frame_blueprint(frame, total_slots=6, use_example_frequency=True) -> dict:
    fe_dict = getattr(frame, "FE", {}) or {}
    freq = _fe_frequency_in_examples(frame) if use_example_frequency else Counter()

    core_fes, non_core_fes = [], []
    for fe_name, fe in fe_dict.items():
        item = {
            "name": fe_name,
            "type": _get_fe_type(fe),
            "priority": _fe_priority(_get_fe_type(fe)),
            "freq": freq.get(fe_name, 0),
        }
        if _is_core_fe(fe):
            core_fes.append(item)
        else:
            non_core_fes.append(item)

    core_fes = sorted(core_fes, key=lambda x: (x["priority"], -x["freq"], x["name"]))
    non_core_fes = sorted(non_core_fes, key=lambda x: (x["priority"], -x["freq"], x["name"]))

    core_names = [x["name"] for x in core_fes]
    remaining = max(0, total_slots - len(core_names))
    added = [x["name"] for x in non_core_fes[:remaining]]
    selected = core_names + added

    return {
        "frame": frame.name,
        "elements": {fe_name: None for fe_name in selected},
        "meta": {"core": core_names, "added": added, "selected": selected},
    }


def build_canonical_orders(total_slots=6) -> dict[str, list[str]]:
    print("Pre-computing canonical FrameNet blueprints...")
    all_blueprints = {
        frame.name: frame_blueprint(frame, total_slots=total_slots)
        for frame in tqdm(fn.frames(), desc="Analyzing Frames")
    }

    # Extract 5 elements + padding
    blueprints = {
        key: {
            k: item["elements"].get(k, "~")
            for k in (list(item["elements"].keys()) + [f"EMPTY_{i}" for i in range(1, total_slots)])[:total_slots - 1]
        }
        for key, item in all_blueprints.items()
    }

    key_counter = Counter()
    element_indices = defaultdict(list)

    for item in blueprints.values():
        key_counter.update(item.keys())
        for index, key in enumerate(item.keys()):
            element_indices[key].append(index)

    canonical_orders = {}
    for frame_name, frame_dict in blueprints.items():
        elements = list(frame_dict.keys())

        def sort_key(el):
            if el.startswith("EMPTY"):
                return (float('inf'), 0, el)
            idx_counts = Counter(element_indices[el])
            pref_idx = idx_counts.most_common(1)[0][0] if idx_counts else 999
            freq = key_counter[el]
            return (pref_idx, -freq, el)

        canonical_orders[frame_name] = sorted(elements, key=sort_key)

    return canonical_orders

def _framenet_cache_dir() -> Path:
    cache_dir = DATA_DIR / "cache" / "framenet"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


def _canonical_orders_cache_path(total_slots: int = 6) -> Path:
    # Include slot count in file name so future config changes do not collide
    return _framenet_cache_dir() / f"canonical_orders_slots{total_slots}.json"


def get_canonical_orders_cached(
    total_slots: int = 6,
    *,
    force_recompute: bool = False,
) -> dict[str, list[str]]:
    cache_path = _canonical_orders_cache_path(total_slots=total_slots)

    if not force_recompute and cache_path.exists():
        print(f"Loading canonical FrameNet orders from cache: {cache_path}")
        with cache_path.open("r", encoding="utf-8") as f:
            data = json.load(f)

        if not isinstance(data, dict):
            raise ValueError(f"Invalid canonical orders cache format: {cache_path}")

        # Defensive normalization
        return {
            str(frame_name): [str(x) for x in order]
            for frame_name, order in data.items()
        }

    canonical_orders = build_canonical_orders(total_slots=total_slots)

    tmp_path = cache_path.with_suffix(".tmp")
    with tmp_path.open("w", encoding="utf-8") as f:
        json.dump(canonical_orders, f, ensure_ascii=False)
    tmp_path.replace(cache_path)

    print(f"Saved canonical FrameNet orders to cache: {cache_path}")
    return canonical_orders

def extract_core_lemmas(text: str, nlp: spacy.Language) -> str:
    """Lemmatizes text and removes stop words to get the semantic core."""
    if not text or text == "UNKNOWN_TARGET" or text == "~":
        return text

    doc = nlp(text)
    lemmas = [token.lemma_ for token in doc if not token.is_stop and not token.is_punct]

    if not lemmas:
        lemmas = [token.lemma_ for token in doc]

    return " ".join(lemmas) if lemmas else "~"


# ---------------------------------------------------------------------------
# N-GRAM VECTOR CREATION
#
# Supports a comma-separated list of orders, e.g. type="4gram,5gram".
# Lemmatisation (spaCy) happens once; n-gram extraction runs in parallel for
# all requested orders.
#
# Each n writes directly to:
#   exp.output_dir/{n}gram/   ← part-*.parquet + _meta.json
#
# This matches the population convention exactly:
#   PopulationExperimentConfig.vectors_dir() = data_dir/vectors/{dataset}
#   where --dataset is passed as "3gram", "4gram", etc.
#
# For single n, cfg.output_dir() == the parquet dir (logs live alongside).
# For multi n,  cfg.output_dir() is a sibling dir used for logs only;
#               parquets go to exp.output_dir/{n}gram/ for each n.
#
# Resume: each n stores its own full state (vector_count, part_id, part_rows,
# raw_seen, sent_id) in its own _meta.json.  On resume, min(raw_seen) across
# all n is used to skip the HF stream.  Part files are never overwritten: if a
# part file already exists when a writer is about to open it, part_id is bumped.
#
# Schema per n: (sent_id int64, w1 string, ..., wN string)
# ---------------------------------------------------------------------------

# Sentence-boundary padding tokens. Pad styles:
#   "bos-eos": n-1 BOS left, one EOS right (classic causal n-gram LM padding).
#   "padded":  one BOS left, one EOS right (symmetric, encoder-like).
BOS_TOKEN = "<s>"
EOS_TOKEN = "</s>"
PAD_STYLES = ("bos-eos", "padded")


def _ngram_schema(n: int) -> pa.Schema:
    fields = [("sent_id", pa.int64())]
    for i in range(1, n + 1):
        fields.append((f"w{i}", pa.string()))
    return pa.schema(fields)


def _extract_ngrams_from_lemmas(
    lemmas: list[str],
    n: int,
    sent_id: int,
    *,
    pad: bool = False,
    pad_style: str = "bos-eos",
    bos: str = BOS_TOKEN,
    eos: str = EOS_TOKEN,
) -> list[dict]:
    if pad:
        # Skip empty sentences entirely; otherwise we'd emit pure-boundary n-grams.
        if not lemmas:
            return []
        if pad_style not in PAD_STYLES:
            raise ValueError(f"Unknown pad_style {pad_style!r}; expected one of {PAD_STYLES}.")
        n_bos = n - 1 if pad_style == "bos-eos" else 1
        # "padded": sentences shorter than n-2 tokens yield no windows.
        seq = [bos] * n_bos + lemmas + [eos]
    else:
        if len(lemmas) < n:
            return []
        seq = lemmas
    rows = []
    for i in range(len(seq) - n + 1):
        row: dict = {"sent_id": sent_id}
        for j in range(n):
            row[f"w{j + 1}"] = seq[i + j]
        rows.append(row)
    return rows


DEDUP_MODES = ("none", "doc", "sent")


def _hash64(s: str) -> int:
    return int.from_bytes(hashlib.blake2b(s.encode("utf-8"), digest_size=8).digest(), "little")


def _doc_hash(text: str) -> int:
    return _hash64(" ".join(text.split()).lower())


def _sent_hash(tokens: list[str]) -> int:
    return _hash64("\x1f".join(tokens))


class _Deduper:
    """Exact document / sentence dedup via 64-bit hashes (first occurrence kept).

    Hashes are stored in ``{state_dir}/_dedup_hashes.npz`` on close and reloaded on
    resume; ~70 bytes of RAM per distinct hash (about 5 GB per 70M sentences).
    """

    def __init__(self, mode: str, min_tokens: int, state_dir: Path, raw_seen: int):
        if mode not in DEDUP_MODES:
            raise ValueError(f"Unknown dedup {mode!r}; expected one of {DEDUP_MODES}.")
        self.mode = mode
        self.min_tokens = min_tokens
        self.path = state_dir / "_dedup_hashes.npz"
        self.docs: set[int] = set()
        self.sents: set[int] = set()
        self.n_docs_dropped = 0
        self.n_sents_dropped = 0
        if mode != "none" and raw_seen > 0:
            if self.path.exists():
                z = np.load(self.path)
                self.docs = set(z["docs"].tolist())
                self.sents = set(z["sents"].tolist())
                if int(z["raw_seen"]) != raw_seen:
                    print(f"Warning: dedup hashes saved at raw_seen={int(z['raw_seen']):,}, "
                          f"resuming at {raw_seen:,}; duplicates of the gap are not caught.")
            else:
                print("Warning: resuming without saved dedup hashes; earlier text is not deduplicated against.")

    @staticmethod
    def _seen(store: set, h: int) -> bool:
        if h in store:
            return True
        store.add(h)
        return False

    def drop_doc(self, text: str) -> bool:
        if self.mode == "none":
            return False
        if self._seen(self.docs, _doc_hash(text)):
            self.n_docs_dropped += 1
            return True
        return False

    def drop_sent(self, tokens: list[str]) -> bool:
        if self.mode != "sent" or len(tokens) < self.min_tokens:
            return False
        if self._seen(self.sents, _sent_hash(tokens)):
            self.n_sents_dropped += 1
            return True
        return False

    def keep_mask(self, doc_h: np.ndarray, sent_doc: np.ndarray,
                  sent_h: np.ndarray, sent_ok: np.ndarray) -> np.ndarray:
        """Batch version of drop_doc/drop_sent on precomputed hashes, in stream order.

        ``sent_ok`` marks sentences long enough for sentence dedup. Returns the
        per-sentence keep mask.
        """
        if self.mode == "none":
            return np.ones(len(sent_doc), dtype=bool)
        keep_doc = np.ones(len(doc_h), dtype=bool)
        for i, h in enumerate(doc_h.tolist()):
            if self._seen(self.docs, h):
                keep_doc[i] = False
                self.n_docs_dropped += 1
        keep = keep_doc[sent_doc]
        if self.mode == "sent":
            idx = np.flatnonzero(keep & sent_ok)
            for i, h in zip(idx.tolist(), sent_h[idx].tolist()):
                if self._seen(self.sents, h):
                    keep[i] = False
                    self.n_sents_dropped += 1
        return keep

    def summary(self) -> str:
        return f"dedup: docs dropped={self.n_docs_dropped:,} | sents dropped={self.n_sents_dropped:,}"

    def save(self, raw_seen: int) -> None:
        if self.mode == "none":
            return
        np.savez(self.path, docs=np.fromiter(self.docs, np.uint64, len(self.docs)),
                 sents=np.fromiter(self.sents, np.uint64, len(self.sents)), raw_seen=raw_seen)


_RAW_NLP = None
_RAW_OPTS: dict = {}


def _raw_worker_init(opts: dict) -> None:
    global _RAW_NLP, _RAW_OPTS
    # Ctrl-C is handled by the main process, which drains the pending batches.
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    _RAW_NLP = spacy.blank("en")
    _RAW_NLP.add_pipe("sentencizer")
    _RAW_NLP.max_length = 1_000_000
    _RAW_OPTS = opts


def _raw_worker_batch(texts: list[str]) -> dict:
    """Worker: texts -> hashes + n-gram windows as Arrow columns.

    Windows match _extract_ngrams_from_lemmas; ``counts[n][s]`` is the number of
    windows of sentence s, so the main process can drop whole sentences by mask.
    """
    o = _RAW_OPTS
    dedup = o["dedup"] != "none"
    doc_h, sent_doc, sent_h, sent_ok = [], [], [], []
    cols = {n: [[] for _ in range(n)] for n in o["orders"]}
    counts = {n: [] for n in o["orders"]}
    for d, doc in enumerate(_RAW_NLP.pipe(texts, batch_size=o["batch_size"])):
        if dedup:
            doc_h.append(_doc_hash(texts[d]))
        for sent in doc.sents:
            tokens = [
                tok.lower_
                for tok in sent
                if not tok.is_punct and not tok.is_space and tok.text.strip()
                and not (o["content_only"] and tok.is_stop)
            ]
            sent_doc.append(d)
            ok = o["dedup"] == "sent" and len(tokens) >= o["min_tokens"]
            sent_ok.append(ok)
            sent_h.append(_sent_hash(tokens) if ok else 0)
            for n in o["orders"]:
                if o["pad"]:
                    n_bos = n - 1 if o["pad_style"] == "bos-eos" else 1
                    seq = [BOS_TOKEN] * n_bos + tokens + [EOS_TOKEN] if tokens else []
                else:
                    seq = tokens
                L = len(seq) - n + 1
                if L > 0:
                    for j in range(n):
                        cols[n][j].extend(seq[j:j + L])
                counts[n].append(max(L, 0))
    return {
        "doc_h": np.array(doc_h, dtype=np.uint64),
        "sent_doc": np.array(sent_doc, dtype=np.int64),
        "sent_h": np.array(sent_h, dtype=np.uint64),
        "sent_ok": np.array(sent_ok, dtype=bool),
        "counts": {n: np.array(c, dtype=np.int64) for n, c in counts.items()},
        "cols": {n: [pa.array(c, type=pa.string()) for c in cs] for n, cs in cols.items()},
    }


def _safe_open_part_writer(output_dir: Path, part_id: int, schema: pa.Schema):
    """Open a ParquetWriter for the given part, bumping part_id if the file already exists."""
    while _part_path(output_dir, part_id).exists():
        part_id += 1
    writer = pq.ParquetWriter(_part_path(output_dir, part_id), schema, compression="zstd")
    return writer, part_id


def create_ngram_vectors_parquet_sharded(
    cfg: VectorRunConfig,
    *,
    overwrite: bool = False,
) -> dict:
    """
    N-gram vector creation from a HF corpus, using spaCy for lemmatisation.

    Streams HF texts, splits into sentences, lemmatizes each sentence (skipping
    punctuation and whitespace), then emits n-grams for every requested order in
    a single pass.  Each order writes to exp.output_dir/{n}-gram-{dataset}_{config}_{target_vectors}/
    so that the population script can find it via the canonical ngram_dir path.
    """
    ngram_orders = parse_ngram_orders(cfg.exp.type)
    if not ngram_orders:
        raise ValueError(
            f"create_ngram_vectors_parquet_sharded called with non-ngram type '{cfg.exp.type}'"
        )

    # base_dir is used only for logs (runs.jsonl, vector_creation_log.txt).
    # For single n, base_dir == n_dir (same location as parquets).
    base_dir = cfg.output_dir()
    base_dir.mkdir(parents=True, exist_ok=True)

    # ---- per-n state ----
    # n_dir = exp.output_dir/{n}-gram-{dataset}_{config}_{target_vectors}
    # Each n stores complete resume state (including raw_seen, sent_id) in its
    # own _meta.json so there is no shared-state split.
    ns: dict[int, dict] = {}
    for n in ngram_orders:
        n_dir = cfg.ngram_dir(n, padded=cfg.exp.pad_sentences, pad_style=cfg.exp.pad_style)
        n_dir.mkdir(parents=True, exist_ok=True)
        n_meta_path = n_dir / "_meta.json"

        if overwrite and n_meta_path.exists():
            n_meta_path.unlink()

        n_meta: dict = {}
        if n_meta_path.exists():
            n_meta = json.loads(n_meta_path.read_text(encoding="utf-8"))

        part_id = int(n_meta.get("part_id", 0))
        part_rows = int(n_meta.get("part_rows", 0))
        # Bump past any existing (possibly partial) part file from a prior run.
        while _part_path(n_dir, part_id).exists():
            part_id += 1
            part_rows = 0

        ns[n] = {
            "output_dir": n_dir,
            "meta_path": n_meta_path,
            "schema": _ngram_schema(n),
            "vector_count": int(n_meta.get("vector_count", 0)),
            "part_id": part_id,
            "part_rows": part_rows,
            "raw_seen": int(n_meta.get("raw_seen", 0)),
            "sent_id": int(n_meta.get("sent_id", 0)),
            "buffer": [],
            "writer": None,
        }

    # Resume from the most conservative point (lowest raw_seen across all n).
    raw_seen = min(st["raw_seen"] for st in ns.values())
    global_sent_id = min(st["sent_id"] for st in ns.values())

    if raw_seen > 0:
        counts_str = ", ".join(f"{n}gram={ns[n]['vector_count']:,}" for n in ngram_orders)
        print(
            f"Resuming: raw_seen={raw_seen:,} | sent_id={global_sent_id:,} | {counts_str}"
        )

    def save_all_metas() -> None:
        """Save every n's meta atomically (same raw_seen / sent_id snapshot)."""
        for n in ngram_orders:
            st = ns[n]
            st["meta_path"].write_text(
                json.dumps({
                    "vector_count": st["vector_count"],
                    "part_id": st["part_id"],
                    "part_rows": st["part_rows"],
                    "raw_seen": raw_seen,
                    "sent_id": global_sent_id,
                }),
                encoding="utf-8",
            )

    def flush_n(n: int) -> None:
        st = ns[n]
        if not st["buffer"]:
            return
        if st["writer"] is None:
            st["writer"], st["part_id"] = _safe_open_part_writer(
                st["output_dir"], st["part_id"], st["schema"]
            )
        wrote = flush_parquet(st["writer"], st["buffer"], SCHEMA=st["schema"])
        st["buffer"].clear()
        st["vector_count"] += wrote
        st["part_rows"] += wrote
        if st["part_rows"] >= cfg.exp.rows_per_part:
            st["writer"].close()
            st["writer"] = None
            st["part_id"] += 1
            st["part_rows"] = 0

    # ---- spaCy: tokeniser + tagger + lemmatiser + sentenciser, no parser/ner ----
    nlp = spacy.load(cfg.exp.spacy_model, disable=["ner", "parser", "senter", "textcat"])
    nlp.add_pipe("sentencizer")
    nlp.max_length = 1_000_000

    # ---- HF dataset stream ----
    ds_kwargs = dict(split=cfg.hf.split, streaming=True)
    ds = (
        load_dataset(cfg.hf.path, cfg.hf.config, **ds_kwargs)
        if cfg.hf.config is not None
        else load_dataset(cfg.hf.path, **ds_kwargs)
    )
    if raw_seen > 0:
        ds = ds.skip(raw_seen)

    dedup = _Deduper(cfg.exp.dedup, cfg.exp.dedup_min_tokens,
                     ns[ngram_orders[0]]["output_dir"], raw_seen)

    def gen_texts() -> Iterator[str]:
        nonlocal raw_seen
        for ex in ds:
            raw_seen += 1
            txt = ex.get(cfg.hf.text_column)
            if not txt or len(txt) > cfg.exp.max_text_length:
                continue
            if dedup.drop_doc(txt):
                continue
            yield txt

    def chunked(iterable, size):
        it = iter(iterable)
        while True:
            batch = list(islice(it, size))
            if not batch:
                return
            yield batch

    start_time = time.time()
    last_log_time = start_time

    try:
        for text_batch in chunked(gen_texts(), cfg.exp.batch_size * 8):
            for doc in nlp.pipe(text_batch, batch_size=cfg.exp.batch_size):
                for sent in doc.sents:
                    lemmas = [
                        tok.lemma_.lower()
                        for tok in sent
                        if not tok.is_punct and not tok.is_space and tok.lemma_.strip()
                        and not (cfg.exp.content_only and tok.is_stop)
                    ]
                    if dedup.drop_sent(lemmas):
                        continue
                    for n in ngram_orders:
                        rows = _extract_ngrams_from_lemmas(
                            lemmas, n, global_sent_id, pad=cfg.exp.pad_sentences,
                            pad_style=cfg.exp.pad_style,
                        )
                        if rows:
                            ns[n]["buffer"].extend(rows)
                    global_sent_id += 1

            # Flush any n whose buffer is full, then checkpoint together.
            flushed = False
            for n in ngram_orders:
                if len(ns[n]["buffer"]) >= cfg.exp.rows_per_flush:
                    flush_n(n)
                    flushed = True
            if flushed:
                save_all_metas()

            now = time.time()
            if now - last_log_time >= cfg.exp.log_every_s:
                elapsed = now - start_time
                tv = sum(ns[n]["vector_count"] for n in ngram_orders)
                vps = tv / elapsed if elapsed > 0 else 0.0
                target = cfg.exp.target_vectors * len(ngram_orders)
                remaining = max(target - tv, 0)
                eta_seconds = remaining / vps if vps > 0 else float("inf")
                per_n = " | ".join(f"{n}gram={ns[n]['vector_count']:,}" for n in ngram_orders)
                print(
                    f"[{elapsed:,.1f}s] raw_seen={raw_seen:,} | sentences={global_sent_id:,} | "
                    f"{per_n} | vec/s={vps:,.0f} | "
                    f"ETA={eta_seconds / 3600:,.2f}h ({eta_seconds / 60:,.1f}m)"
                )
                last_log_time = now

            if all(ns[n]["vector_count"] >= cfg.exp.target_vectors for n in ngram_orders):
                break

    except KeyboardInterrupt:
        print("\nGracefully interrupting and flushing buffers...")
    finally:
        for n in ngram_orders:
            flush_n(n)
            if ns[n]["writer"]:
                ns[n]["writer"].close()
                ns[n]["writer"] = None
        save_all_metas()
        dedup.save(raw_seen)
        gc.collect()

    if dedup.mode != "none":
        print(dedup.summary())

    for n in ngram_orders:
        print(f"{n}-gram vectors written: {ns[n]['vector_count']:,} (dir: {ns[n]['output_dir']})")

    return {
        "output_dir": str(base_dir),
        "n_gram_orders": ngram_orders,
        "per_n": {
            n: {
                "output_dir": str(ns[n]["output_dir"]),
                "vectors_written": ns[n]["vector_count"],
                "part_id": ns[n]["part_id"],
            }
            for n in ngram_orders
        },
        "total_vectors_written": sum(ns[n]["vector_count"] for n in ngram_orders),
        "sent_id": int(global_sent_id),
        "raw_seen": int(raw_seen),
        "hf_path": cfg.hf.path,
        "hf_config": cfg.hf.config,
        "hf_split": cfg.hf.split,
        "hf_text_column": cfg.hf.text_column,
    }


# ---------------------------------------------------------------------------
# RAW N-GRAM VECTOR CREATION (no lemmatisation)
#
# Identical pipeline to create_ngram_vectors_parquet_sharded but uses raw
# lowercased tokens instead of lemmas.  No tagger or lemmatiser is loaded
# (spacy.blank), so this is faster and requires no language-model weights.
# Output directories follow the same naming convention so the population
# script can consume them transparently.
# ---------------------------------------------------------------------------

def create_raw_ngram_vectors_parquet_sharded(
    cfg: VectorRunConfig,
    *,
    overwrite: bool = False,
) -> dict:
    """
    Stream a HF corpus, split into sentences, tokenise with a blank spaCy
    model, and emit raw (lowercased, non-lemmatised) n-grams for every
    requested order in a single pass.

    Output layout and resume semantics are identical to
    create_ngram_vectors_parquet_sharded.
    """
    ngram_orders = parse_raw_ngram_orders(cfg.exp.type)
    if not ngram_orders:
        raise ValueError(
            f"create_raw_ngram_vectors_parquet_sharded called with non-raw-ngram type '{cfg.exp.type}'"
        )

    base_dir = cfg.output_dir()
    base_dir.mkdir(parents=True, exist_ok=True)

    ns: dict[int, dict] = {}
    for n in ngram_orders:
        n_dir = cfg.ngram_dir(n, raw=True, padded=cfg.exp.pad_sentences, pad_style=cfg.exp.pad_style)
        n_dir.mkdir(parents=True, exist_ok=True)
        n_meta_path = n_dir / "_meta.json"

        if overwrite and n_meta_path.exists():
            n_meta_path.unlink()

        n_meta: dict = {}
        if n_meta_path.exists():
            n_meta = json.loads(n_meta_path.read_text(encoding="utf-8"))

        part_id = int(n_meta.get("part_id", 0))
        part_rows = int(n_meta.get("part_rows", 0))
        while _part_path(n_dir, part_id).exists():
            part_id += 1
            part_rows = 0

        ns[n] = {
            "output_dir": n_dir,
            "meta_path": n_meta_path,
            "schema": _ngram_schema(n),
            "vector_count": int(n_meta.get("vector_count", 0)),
            "part_id": part_id,
            "part_rows": part_rows,
            "raw_seen": int(n_meta.get("raw_seen", 0)),
            "sent_id": int(n_meta.get("sent_id", 0)),
            "buffer": [],
            "writer": None,
        }

    raw_seen = min(st["raw_seen"] for st in ns.values())
    global_sent_id = min(st["sent_id"] for st in ns.values())

    if raw_seen > 0:
        counts_str = ", ".join(f"{n}gram={ns[n]['vector_count']:,}" for n in ngram_orders)
        print(f"Resuming: raw_seen={raw_seen:,} | sent_id={global_sent_id:,} | {counts_str}")

    def save_all_metas() -> None:
        for n in ngram_orders:
            st = ns[n]
            st["meta_path"].write_text(
                json.dumps({
                    "vector_count": st["vector_count"],
                    "part_id": st["part_id"],
                    "part_rows": st["part_rows"],
                    "raw_seen": raw_seen,
                    "sent_id": global_sent_id,
                }),
                encoding="utf-8",
            )

    # Buffers hold Arrow tables (built in the workers), not row dicts.
    for st in ns.values():
        st["buffer_rows"] = 0

    def flush_n(n: int) -> None:
        st = ns[n]
        if not st["buffer"]:
            return
        if st["writer"] is None:
            st["writer"], st["part_id"] = _safe_open_part_writer(
                st["output_dir"], st["part_id"], st["schema"]
            )
        table = pa.concat_tables(st["buffer"])
        st["writer"].write_table(table)
        wrote = table.num_rows
        st["buffer"].clear()
        st["buffer_rows"] = 0
        st["vector_count"] += wrote
        st["part_rows"] += wrote
        if st["part_rows"] >= cfg.exp.rows_per_part:
            st["writer"].close()
            st["writer"] = None
            st["part_id"] += 1
            st["part_rows"] = 0

    if cfg.exp.pad_sentences and cfg.exp.pad_style not in PAD_STYLES:
        raise ValueError(f"Unknown pad_style {cfg.exp.pad_style!r}; expected one of {PAD_STYLES}.")

    ds_kwargs = dict(split=cfg.hf.split, streaming=True)
    ds = (
        load_dataset(cfg.hf.path, cfg.hf.config, **ds_kwargs)
        if cfg.hf.config is not None
        else load_dataset(cfg.hf.path, **ds_kwargs)
    )
    if raw_seen > 0:
        ds = ds.skip(raw_seen)

    dedup = _Deduper(cfg.exp.dedup, cfg.exp.dedup_min_tokens,
                     ns[ngram_orders[0]]["output_dir"], raw_seen)

    # read_seen = docs taken from the stream; raw_seen (saved in _meta.json) only
    # advances once a batch's windows are buffered, so resume never skips docs.
    read_seen = raw_seen

    def gen_batches(size: int) -> Iterator[tuple[list[str], int]]:
        nonlocal read_seen
        batch: list[str] = []
        for ex in ds:
            read_seen += 1
            txt = ex.get(cfg.hf.text_column)
            if not txt or len(txt) > cfg.exp.max_text_length:
                continue
            batch.append(txt)
            if len(batch) >= size:
                yield batch, read_seen
                batch = []
        if batch:
            yield batch, read_seen

    def consume(res: dict, batch_end: int) -> None:
        """Main process: dedup on worker hashes, sent_ids, mask windows — in stream order."""
        nonlocal global_sent_id, raw_seen
        keep = dedup.keep_mask(res["doc_h"], res["sent_doc"], res["sent_h"], res["sent_ok"])
        # Kept sentences are numbered consecutively, as before.
        sent_ids = global_sent_id + np.cumsum(keep) - 1
        global_sent_id += int(keep.sum())
        all_kept = bool(keep.all())
        for n in ngram_orders:
            counts = res["counts"][n]
            sid = np.repeat(sent_ids, counts)
            cols = res["cols"][n]
            if not all_kept:
                wmask = np.repeat(keep, counts)
                sid = sid[wmask]
                mask = pa.array(wmask)
                cols = [c.filter(mask) for c in cols]
            if len(sid) == 0:
                continue
            table = pa.Table.from_arrays([pa.array(sid, type=pa.int64()), *cols], schema=ns[n]["schema"])
            ns[n]["buffer"].append(table)
            ns[n]["buffer_rows"] += table.num_rows
        raw_seen = batch_end

    # Workers tokenise, hash and build the windows as Arrow columns; the main
    # process only does the order-dependent parts (dedup sets, sent_ids, writing).
    n_workers = max(1, compute_num_threads(cfg.exp.cpu_frac) - 1)
    max_pending = 4 * n_workers
    print(f"Tokenising with {n_workers} worker processes")
    worker_opts = dict(
        orders=list(ngram_orders), pad=cfg.exp.pad_sentences, pad_style=cfg.exp.pad_style,
        content_only=cfg.exp.content_only, dedup=cfg.exp.dedup,
        min_tokens=cfg.exp.dedup_min_tokens, batch_size=cfg.exp.batch_size,
    )
    pool = mp.get_context("spawn").Pool(
        n_workers, initializer=_raw_worker_init, initargs=(worker_opts,),
    )
    pending: deque = deque()
    batches = gen_batches(cfg.exp.batch_size * 8)

    start_time = time.time()
    last_log_time = start_time

    try:
        while True:
            while len(pending) < max_pending:
                nxt = next(batches, None)
                if nxt is None:
                    break
                texts, batch_end = nxt
                pending.append((pool.apply_async(_raw_worker_batch, (texts,)), batch_end))
            if not pending:
                break
            res, batch_end = pending.popleft()
            consume(res.get(), batch_end)

            flushed = False
            for n in ngram_orders:
                if ns[n]["buffer_rows"] >= cfg.exp.rows_per_flush:
                    flush_n(n)
                    flushed = True
            if flushed:
                save_all_metas()

            now = time.time()
            if now - last_log_time >= cfg.exp.log_every_s:
                elapsed = now - start_time
                tv = sum(ns[n]["vector_count"] for n in ngram_orders)
                vps = tv / elapsed if elapsed > 0 else 0.0
                target = cfg.exp.target_vectors * len(ngram_orders)
                remaining = max(target - tv, 0)
                eta_seconds = remaining / vps if vps > 0 else float("inf")
                per_n = " | ".join(f"{n}gram={ns[n]['vector_count']:,}" for n in ngram_orders)
                print(
                    f"[{elapsed:,.1f}s] raw_seen={raw_seen:,} | sentences={global_sent_id:,} | "
                    f"{per_n} | vec/s={vps:,.0f} | "
                    f"ETA={eta_seconds / 3600:,.2f}h ({eta_seconds / 60:,.1f}m)"
                    + (f" | {dedup.summary()}" if dedup.mode != "none" else "")
                )
                last_log_time = now

            if all(ns[n]["vector_count"] >= cfg.exp.target_vectors for n in ngram_orders):
                break

    except KeyboardInterrupt:
        print("\nGracefully interrupting and flushing buffers...")
    finally:
        # Write out batches already sent to workers: their docs are in the dedup
        # hashes, so dropping them would lose those docs on resume.
        try:
            while pending:
                res, batch_end = pending.popleft()
                consume(res.get(), batch_end)
        except Exception as e:
            print(f"Warning: could not drain pending batches ({e!r}); resume restarts at raw_seen={raw_seen:,}.")
        pool.terminate()
        pool.join()
        for n in ngram_orders:
            flush_n(n)
            if ns[n]["writer"]:
                ns[n]["writer"].close()
                ns[n]["writer"] = None
        save_all_metas()
        dedup.save(raw_seen)
        gc.collect()

    if dedup.mode != "none":
        print(dedup.summary())

    for n in ngram_orders:
        print(f"{n}-gram (raw) vectors written: {ns[n]['vector_count']:,} (dir: {ns[n]['output_dir']})")

    return {
        "output_dir": str(base_dir),
        "n_gram_orders": ngram_orders,
        "per_n": {
            n: {
                "output_dir": str(ns[n]["output_dir"]),
                "vectors_written": ns[n]["vector_count"],
                "part_id": ns[n]["part_id"],
            }
            for n in ngram_orders
        },
        "total_vectors_written": sum(ns[n]["vector_count"] for n in ngram_orders),
        "sent_id": int(global_sent_id),
        "raw_seen": int(raw_seen),
        "hf_path": cfg.hf.path,
        "hf_config": cfg.hf.config,
        "hf_split": cfg.hf.split,
        "hf_text_column": cfg.hf.text_column,
    }