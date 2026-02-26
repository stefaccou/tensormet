from tensormet.utils import (select_gpu,
                             ThreadBudget,
                             compute_num_threads,
                             DATA_DIR,
                             write_json,
                             append_jsonl,
                             utc_now_iso,
                             tee_output,
                             notify_discord
                             )
from tensormet.tucker_tensor import SparseTupleTensor
from tensormet.similarity import load_eval_sentences_cached_parquet, ensure_vocab
from tensormet.vector_creation import create_vectors_parquet_sharded
import os
import sys
import pickle
import tensorly as tl
from tensorly.tucker_tensor import TuckerTensor
from pathlib import Path
import numpy as np
import torch
import cupy as cp
import time
from dataclasses import asdict

def launch_vector_creation(cfg, *, overwrite: bool | None = None):
    """
    Run vector creation with the same "launcher" conventions:
    - sets thread budget (CPU)
    - creates output directory
    - logs a run record to output_dir/runs.jsonl
    - optionally notifies discord
    """
    # cfg is expected to be VectorRunConfig (cfg.exp is VectorExperimentConfig)
    output_dir = cfg.output_dir()
    print("output_dir: ", output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Thread budget: vector cfg uses exp.cpu_frac (not max_cpu_frac)
    thread_budget = ThreadBudget(n_threads=compute_num_threads(cfg.exp.cpu_frac))

    # Decide overwrite behavior
    # (vector pipeline itself is resume-safe; overwrite typically means "drop _meta.json" etc.)
    do_overwrite = overwrite if overwrite is not None else False

    # Where we record "runs" like decomposition does
    runs_jsonl = output_dir / "runs.jsonl"

    # Optional: tee stdout/stderr to a log file
    log_path = output_dir / "vector_creation_log.txt"

    # Save a run "header" row before starting (helps if it crashes)
    append_jsonl(
        runs_jsonl,
        {
            "timestamp": utc_now_iso(),
            "run_kind": "vector_creation",
            "cfg": asdict(cfg),
            "output_dir": str(output_dir),
            "overwrite": bool(do_overwrite),
        },
    )

    start_time = time.time()

    # If you want the same stdout capture style as decomposition:
    with tee_output(log_path):
        summary = create_vectors_parquet_sharded(cfg, overwrite=do_overwrite)

    end_time = time.time()

    # Record results row
    append_jsonl(
        runs_jsonl,
        {
            "timestamp": utc_now_iso(),
            "run_kind": "vector_creation",
            "cfg": asdict(cfg),
            "results": {
                "runtime_seconds": round(end_time - start_time, 2),
                **summary,  # output_dir, meta_path, vectors_written, etc.
            },
        },
    )

    notify_discord(
        f"Vector creation finished: vectors={summary.get('vectors_written', '??')} "
        f"dir={summary.get('output_dir', str(output_dir))} "
        f"runtime={end_time - start_time:.2f}s"
    )

    return summary

def launch_nnt_decomposition(cfg):
    device = select_gpu()
    thread_budget = ThreadBudget(n_threads=compute_num_threads(cfg.exp.max_cpu_frac))
    tl.set_backend("cupy")

    # we load the sample sentences only once
    vector_path = os.path.join(DATA_DIR, "vectors", cfg.exp.dataset)
    sentence_sample = load_eval_sentences_cached_parquet(vector_path=vector_path,
                                                         dataset=cfg.exp.dataset,
                                                         seed=cfg.exp.random_state,
                                                         n_samples=cfg.eval.sem_fitness_target,
                                                         )

    vocab_path = os.path.join(DATA_DIR, "tensors", cfg.exp.dataset, f"vocabularies/{cfg.exp.dim}.pkl")
    with open(vocab_path, "rb") as f:
        vocab = pickle.load(f)

    if cfg.eval.remove_OOV:
        start = time.time()
        clean_sample = ensure_vocab(vocab, sentence_sample)
        print("cleaned sample in ", time.time() - start)
    else:
        clean_sample = sentence_sample

    paths = cfg.artifact_paths()
    for p in paths.values():
        if isinstance(p, Path):
            p.parent.mkdir(parents=True, exist_ok=True)

    # If model already exists, skip (optional but recommended)
    if paths["model"].exists() and not cfg.exp.overwrite:
        print(f"Decomposition already exists at {paths['model']}, skipping...")
        sys.exit(0)


    # Save config snapshot (single JSON)
    write_json(paths["config"],
               {"timestamp": utc_now_iso(), "run_id": cfg.run_id(), "cfg": asdict(cfg)})


    start_time = time.time()

    sparse_tensor = SparseTupleTensor.load_from_disk(
        dataset=cfg.exp.dataset,
        method=cfg.exp.method,
        dims=cfg.exp.dim,
        tier1=cfg.exp.tier1,
    )

    with tee_output(paths["log"]):
        sparse_tensor.tensor_to_sparse("cupy")
        tucker_decomp_info = sparse_tensor.non_negative_tucker_with_similarity(
            cfg=cfg,
            thread_budget=thread_budget,
            vocab=vocab,
            sample_sentences=clean_sample,
        )

    end_time = time.time()

    # Save model + metrics using cfg paths
    tl.set_backend("pytorch")
    core, factors = tucker_decomp_info["tensor"]
    errors = tucker_decomp_info["errors"]
    fitness_scores = tucker_decomp_info["fitness_scores"]

    core_t = tl.tensor(cp.asnumpy(core))
    factors_t = [tl.tensor(cp.asnumpy(f)) for f in factors]
    tucker_decomp_torch = TuckerTensor((core_t, factors_t))

    torch.save(tucker_decomp_torch, paths["model"])
    np.save(paths["errors"], np.array([cp.asnumpy(e) for e in errors]))
    np.save(paths["fitness"], np.array([cp.asnumpy(f) for f in fitness_scores]))
    append_jsonl(
        paths["runs_jsonl"],
        {
            "timestamp": utc_now_iso(),
            "run_id": cfg.run_id(),
            "cfg": asdict(cfg),
            "results": {
                "iterations": int(tucker_decomp_info["iterations"]),
                "final_error": float(tucker_decomp_info["final_error"]),
                "final_fitness": (float(fitness_scores[-1]) if fitness_scores else None),
                "runtime_seconds": round(end_time - start_time, 2),
                "model_path": str(paths["model"]),
                "errors_path": str(paths["errors"]),
                "fitness_path": str(paths["fitness"]),
                "config_path": str(paths["config"]),
            },
        },
    )
    notify_discord(
        f"Saved Tucker decomposition {cfg.exp.method} - {cfg.exp.dim}/{cfg.exp.rank[0]} to {paths['model']}"
        f" in {end_time - start_time:.2f} seconds."
    )
    print("model, errors and config saved")

    return tucker_decomp_torch