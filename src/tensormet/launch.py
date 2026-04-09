from tensormet.tucker_tensor import extract_roles_from_vocab
from tensormet.utils import (select_gpu,
                             ThreadBudget,
                             compute_num_threads,
                             DATA_DIR,
                             write_json,
                             append_jsonl,
                             utc_now_iso,
                             tee_output,
                             notify_discord,
                             extract_roles_from_vocab,
                             shared_factor_suffix,
                             linked_factor_groups
                             )
import os
import sys
import pickle
import json
import tensorly as tl
from tensorly.tucker_tensor import TuckerTensor
from pathlib import Path
import numpy as np
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
    from tensormet.vector_creation import create_vectors_parquet_sharded, create_frame_vectors_parquet_sharded

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
    if cfg.exp.type == "frames":
        print('Frame-based vector creation')
        with tee_output(log_path):
            summary = create_frame_vectors_parquet_sharded(cfg, overwrite=do_overwrite)
    else:
        print("Syntactic slot-based vector creation")
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


def launch_nnt_decomposition(cfg, gpu_id=None):
    thread_budget = ThreadBudget(n_threads=compute_num_threads(cfg.exp.max_cpu_frac))

    # load in GPU sensitive modules only AFTER device has been set!
    import torch
    import cupy as cp
    from tensormet.tucker_tensor import SparseTupleTensor
    from tensormet.similarity import load_eval_sentences_cached_parquet, ensure_vocab

    tl.set_backend("cupy")

    # we load the sample sentences only once

    # Calculate suffix using the exact same logic as population.py
    linked_groups = linked_factor_groups(cfg.exp.order, cfg.train.shared_factors)
    linked_nontrivial = [group for group in linked_groups if len(group) > 1]
    suffix = shared_factor_suffix(linked_nontrivial)

    # Note the added {cfg.exp.order}D_ to match what population.py writes
    vocab_path = os.path.join(
        DATA_DIR,
        "tensors",
        cfg.exp.dataset,
        f"vocabularies/{cfg.exp.order}D_{cfg.exp.dim}{suffix}.pkl"
    )
    try:
        with open(vocab_path, "rb") as f:
            vocab = pickle.load(f)
    except FileNotFoundError:
        vocab_path = os.path.join(
            DATA_DIR,
            "tensors",
            cfg.exp.dataset,
            f"vocabularies/{cfg.exp.dim}{suffix}.pkl"
        )
        with open(vocab_path, "rb") as f:
            vocab = pickle.load(f)

    roles = extract_roles_from_vocab(vocab)

    vector_path = os.path.join(DATA_DIR, "vectors", cfg.exp.dataset)
    sentence_sample = load_eval_sentences_cached_parquet(vector_path=vector_path,
                                                         dataset=cfg.exp.dataset,
                                                         roles=roles,
                                                         seed=cfg.exp.random_state,
                                                         n_samples=cfg.eval.sem_fitness_target,
                                                         )
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
    if paths["model"].exists() and not cfg.exp.overwrite and not cfg.train.resume:
        print(f"Decomposition already exists at {paths['model']}, skipping...")
        sys.exit(0)


    # Save config snapshot (single JSON)
    write_json(paths["config"],
               {"timestamp": utc_now_iso(), "run_id": cfg.run_id(), "cfg": asdict(cfg)})


    start_time = time.time()

    sparse_tensor = SparseTupleTensor.load_from_disk(
        dataset=cfg.exp.dataset,
        method=cfg.exp.method,
        order=cfg.exp.order,
        dims=cfg.exp.dim,
        tier1=cfg.exp.tier1,
        shared_factors=cfg.train.shared_factors,
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
    np.save(paths["errors"], np.asarray(errors, dtype=float))

    if fitness_scores:
        last = fitness_scores[-1]
        if isinstance(last, dict):
            with open(paths["fitness_json"], "w") as f:
                json.dump(fitness_scores, f, indent=2)
        else:
            np.save(paths["fitness"], np.array([cp.asnumpy(f) for f in fitness_scores]))

    last_fitness = fitness_scores[-1] if fitness_scores else None
    if isinstance(last_fitness, dict):
        final_fitness = float(last_fitness[tucker_decomp_info["sem_primary_key"]])
        final_fitness_full = last_fitness
    else:
        final_fitness = float(last_fitness) if last_fitness is not None else None
        final_fitness_full = None

    append_jsonl(
        paths["runs_jsonl"],
        {
            "timestamp": utc_now_iso(),
            "run_id": cfg.run_id(),
            "cfg": asdict(cfg),
            "results": {
                "iterations": int(tucker_decomp_info["iterations"]),
                "final_error": float(tucker_decomp_info["final_error"]),
                "final_fitness": final_fitness,
                "final_fitness_full": final_fitness_full,
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



def launch_tensor_population(cfg):
    """
    Run sparse tensor population with standard launcher conventions:
    - creates output directories
    - logs a run record to output_dir/populated/runs.jsonl
    - optionally notifies discord
    """
    # Assuming you rename the script 2_sparse_population...py to tensor_population.py
    from tensormet.population import populate_tensors_parquet

    vectors_dir = cfg.exp.vectors_dir()
    output_dir = cfg.exp.output_dir()

    # Mirroring your original script's logic for directory creation
    populated_dir = output_dir / "populated"
    populated_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "vocabularies").mkdir(parents=True, exist_ok=True)

    # Where we record "runs"
    runs_jsonl = populated_dir / "runs.jsonl"
    log_path = populated_dir / "population_log.txt"

    append_jsonl(
        runs_jsonl,
        {
            "timestamp": utc_now_iso(),
            "run_kind": "tensor_population",
            "cfg": asdict(cfg),
            "vectors_dir": str(vectors_dir),
            "output_dir": str(output_dir),
        },
    )

    start_time = time.time()

    with tee_output(log_path):
        results = populate_tensors_parquet(
            path_to_vectors=vectors_dir,
            top_ks=list(cfg.exp.top_ks),
            shared_factors=cfg.exp.shared_factors,
            save=True,
            path_to_tensors=output_dir,
            cols_to_build=list(cfg.exp.cols_to_build),
            batch_rows=cfg.exp.batch_rows,
            batch_readahead=cfg.exp.batch_readahead,
            fragment_readahead=cfg.exp.fragment_readahead,
        )

    end_time = time.time()

    append_jsonl(
        runs_jsonl,
        {
            "timestamp": utc_now_iso(),
            "run_kind": "tensor_population",
            "cfg": asdict(cfg),
            "results": {
                "runtime_seconds": round(end_time - start_time, 2),
                "top_ks_processed": list(cfg.exp.top_ks),
            },
        },
    )

    # notify_discord(
    #     f"Tensor population finished for {cfg.exp.dataset}. "
    #     f"Top Ks: {list(cfg.exp.top_ks)}. "
    #     f"Runtime: {end_time - start_time:.2f}s."
    # )

    return results