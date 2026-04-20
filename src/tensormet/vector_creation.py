# tensormet/vector_creation.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from collections import Counter, defaultdict
from typing import Callable, Dict, Iterable, Iterator, Optional, Tuple
from itertools import islice

import json
import time
import string
from tqdm import tqdm

import pyarrow as pa
import pyarrow.parquet as pq
import spacy
from datasets import load_dataset

from tensormet.utils import DATA_DIR, compute_num_threads
from tensormet.config import VectorRunConfig, parse_ngram_order, parse_ngram_orders

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
        wrote = flush_parquet(writer, buffer)
        vector_count += wrote
        part_rows += wrote
        buffer.clear()

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
        wrote = flush_parquet(writer, buffer, SCHEMA=FRAME_SCHEMA)
        vector_count += wrote
        part_rows += wrote
        buffer.clear()

        if writer:
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

def _ngram_schema(n: int) -> pa.Schema:
    fields = [("sent_id", pa.int64())]
    for i in range(1, n + 1):
        fields.append((f"w{i}", pa.string()))
    return pa.schema(fields)


def _extract_ngrams_from_lemmas(lemmas: list[str], n: int, sent_id: int) -> list[dict]:
    if len(lemmas) < n:
        return []
    rows = []
    for i in range(len(lemmas) - n + 1):
        row: dict = {"sent_id": sent_id}
        for j in range(n):
            row[f"w{j + 1}"] = lemmas[i + j]
        rows.append(row)
    return rows


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
        n_dir = cfg.ngram_dir(n)
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

    def gen_texts() -> Iterator[str]:
        nonlocal raw_seen
        for ex in ds:
            raw_seen += 1
            txt = ex.get(cfg.hf.text_column)
            if not txt or len(txt) > cfg.exp.max_text_length:
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
                    ]
                    for n in ngram_orders:
                        rows = _extract_ngrams_from_lemmas(lemmas, n, global_sent_id)
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
        gc.collect()

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