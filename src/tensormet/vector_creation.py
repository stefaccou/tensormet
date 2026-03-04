# tensormet/vector_creation.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Iterable, Iterator, Optional, Tuple

import json
import time

import pyarrow as pa
import pyarrow.parquet as pq
import spacy
from datasets import load_dataset

from tensormet.utils import DATA_DIR, compute_num_threads
from tensormet.config import VectorRunConfig

import gc

SCHEMA = pa.schema([
    ("sent_id", pa.int64()),
    ("root", pa.string()),
    ("nsubj", pa.string()),
    ("obj", pa.string()),
    ("obj2", pa.string()),
    ("obl", pa.string()),
])







def _part_path(output_dir: Path, part_id: int) -> Path:
    return output_dir / f"part-{part_id:06d}.parquet"


def _open_part_writer(output_dir: Path, part_id: int) -> pq.ParquetWriter:
    part_path = _part_path(output_dir, part_id)
    # Exclusive create: never overwrite shards by accident
    f = part_path.open("xb")
    return pq.ParquetWriter(f, SCHEMA, compression="zstd")


def flush_parquet(writer: pq.ParquetWriter, buffer_rows: list[dict]) -> int:
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
        filled = 0

        for child in root.children:
            d = child.dep

            if d == DEP_NSUBJ and nsubj == "~":
                nsubj = child.lemma_
                filled += 1

            elif d == DEP_OBJ:
                if obj == "~":
                    obj = child.lemma_
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

    writer = _open_part_writer(output_dir, part_id)
    docs = None

    try:
        docs = nlp.pipe(gen_texts(), batch_size=cfg.exp.batch_size, n_process=n_process)
        for doc in docs:
            rows, count, global_sent_id = extract_vectors(
                doc,
                global_sent_id,
                DEP_NSUBJ,
                DEP_OBJ,
                DEP_IOBJ,
                DEP_OBL,
                POS_VERB,
            )

            if count:
                buffer.extend(rows)

            if len(buffer) >= cfg.exp.rows_per_flush:
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