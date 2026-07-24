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
                             linked_factor_groups,
                             dim_spec_str,
                             )
from tensormet.naming import vocab_filename, vocab_filename_legacy
from tensormet.hpc_helpers import stage_artifacts_back
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
    from tensormet.vector_creation import (
        create_vectors_parquet_sharded,
        create_frame_vectors_parquet_sharded,
        create_ngram_vectors_parquet_sharded,
        create_raw_ngram_vectors_parquet_sharded,
    )
    from tensormet.config import parse_ngram_orders, parse_raw_ngram_orders

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

    _orders = parse_ngram_orders(cfg.exp.type)
    _raw_orders = parse_raw_ngram_orders(cfg.exp.type)
    if cfg.exp.type == "frames":
        print('Frame-based vector creation')
        with tee_output(log_path):
            summary = create_frame_vectors_parquet_sharded(cfg, overwrite=do_overwrite)
    elif _raw_orders is not None:
        orders_str = ", ".join(f"{n}-gram (raw)" for n in _raw_orders)
        print(f'Raw n-gram vector creation ({orders_str})')
        with tee_output(log_path):
            summary = create_raw_ngram_vectors_parquet_sharded(cfg, overwrite=do_overwrite)
    elif _orders is not None:
        orders_str = ", ".join(f"{n}-gram" for n in _orders)
        print(f'N-gram vector creation ({orders_str})')
        with tee_output(log_path):
            summary = create_ngram_vectors_parquet_sharded(cfg, overwrite=do_overwrite)
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
        min_mode_ks = None
        if cfg.exp.min_mode_ks:
            min_mode_ks = {i: v for i, v in enumerate(cfg.exp.min_mode_ks) if v > 0}
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
            remove_hapax=cfg.exp.remove_hapax,
            top_ks_asymmetric=list(cfg.exp.top_ks_asymmetric) if cfg.exp.top_ks_asymmetric else None,
            min_mode_ks=min_mode_ks,
            max_workers=cfg.exp.max_workers,
            shards_per_task=cfg.exp.shards_per_task,
            ensured_vocab=list(cfg.exp.ensured_vocab) if cfg.exp.ensured_vocab else None,
            tensors_to_build=list(cfg.exp.tensors_to_build) if cfg.exp.tensors_to_build else None,
            cpu_frac=cfg.exp.cpu_frac,
            max_mem_gb=cfg.exp.max_mem_gb,
            mem_per_worker_gb=cfg.exp.mem_per_worker_gb,
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

def launch_nnt_decomposition(cfg):
    thread_budget = ThreadBudget(n_threads=compute_num_threads(cfg.train.max_cpu_frac))
    _n_gpus = getattr(cfg.train, "n_gpus", 1)
    _gpu_id = getattr(cfg.train, "gpu_id", None)
    select_gpu(gpu_id=_gpu_id, n_gpus=_n_gpus)

    # load in GPU sensitive modules only AFTER device has been set!
    import torch
    print(f"[{time.strftime('%H:%M:%S')}] importing cupy...", flush=True)
    import cupy as cp
    print(f"[{time.strftime('%H:%M:%S')}] importing tucker_tensor...", flush=True)
    from tensormet.tucker_tensor import SparseTupleTensor
    print(f"[{time.strftime('%H:%M:%S')}] importing similarity...", flush=True)
    from tensormet.similarity import load_eval_sentences_cached_parquet, ensure_vocab
    print(f"[{time.strftime('%H:%M:%S')}] all deferred imports done", flush=True)

    tl.set_backend("cupy")

    # we load the sample sentences only once

    # Vocabulary path — try new naming first, fall back to legacy (no order prefix)
    _vdir = os.path.join(DATA_DIR, "tensors", cfg.exp.dataset, "vocabularies")
    vocab_path = os.path.join(
        _vdir,
        vocab_filename(cfg.exp.order, cfg.exp.dim, shared_factors=cfg.exp.shared_factors),
    )
    try:
        with open(vocab_path, "rb") as f:
            vocab = pickle.load(f)
        # print("loaded vocab")
        roles = extract_roles_from_vocab(vocab)
        # print("roles:", roles)
    except FileNotFoundError:
        # print(vocab_path, "not found")
        vocab_path = os.path.join(
            _vdir,
            vocab_filename_legacy(cfg.exp.dim, shared_factors=cfg.exp.shared_factors, order=cfg.exp.order),
        )
        with open(vocab_path, "rb") as f:
            vocab = pickle.load(f)
        # print("legacy role definition")
        roles = None

        # Map vocab role names -> actual parquet column names
    _VOCAB_TO_PARQUET = {
        "verb": "root", "v": "root",
        "subject": "nsubj", "s": "nsubj",
        "object": "obj", "o": "obj",
    }
    parquet_roles = (
        [_VOCAB_TO_PARQUET.get(r, r) for r in roles]
        if roles is not None
        else ["root", "nsubj", "obj"]  # legacy fallback default
    )




    vector_path = os.path.join(DATA_DIR, "vectors", cfg.exp.dataset)
    sentence_sample = load_eval_sentences_cached_parquet(vector_path=vector_path,
                                                         dataset=cfg.exp.dataset,
                                                         roles=parquet_roles,
                                                         seed=cfg.exp.random_state,
                                                         n_samples=cfg.eval.sem_fitness_target,
                                                         )
    if cfg.eval.remove_OOV:
        start = time.time()
        clean_sample = ensure_vocab(vocab, sentence_sample, parquet_roles)
        # print("cleaned sample in ", time.time() - start)
    else:
        clean_sample = sentence_sample

    # print("loaded sentence sample")

    # Working paths follow train.hpc: under HPC mode they live on node-local
    # $TMPDIR so the hot-loop writes never hit shared GPFS. final_paths is the
    # canonical GPFS destination — used for the "already exists" skip, the
    # shared runs.jsonl, and the end-of-job copy-back target.
    paths = cfg.artifact_paths()
    final_paths = cfg.artifact_paths(staged=False)
    if cfg.train.hpc:
        print(f"[hpc] staging artifact writes to node-local {paths['model'].parent}")
    for p in paths.values():
        if isinstance(p, Path):
            p.parent.mkdir(parents=True, exist_ok=True)

    # If model already exists, skip (optional but recommended). Always check the
    # canonical GPFS location — the staged tree is empty at job start.
    if final_paths["model"].exists() and not cfg.train.overwrite and not cfg.train.resume:
        print(f"Decomposition already exists at {final_paths['model']}, skipping...")
        return None


    # Save config snapshot (single JSON). Always write to the canonical GPFS
    # location — this file is the resume anchor (get_resume_state discovers a run
    # by globbing {stem}_config.json there), so it must exist even if an HPC job
    # is killed before the end-of-job copy-back. One tiny per-run write; no churn.
    write_json(final_paths["config"],
               {"timestamp": utc_now_iso(), "run_id": cfg.run_id(), "cfg": asdict(cfg)})


    start_time = time.time()

    sparse_tensor = SparseTupleTensor.load_from_disk(
        dataset=cfg.exp.dataset,
        method=cfg.exp.method,
        order=cfg.exp.order,
        dims=cfg.exp.dim,
        tier1=cfg.train.tier1,
        shared_factors=cfg.exp.shared_factors,
    )



    try:
        with tee_output(paths["log"]):
            sparse_tensor.tensor_to_sparse("cupy")
            tucker_decomp_info = sparse_tensor.non_negative_tucker_with_similarity(
                cfg=cfg,
                thread_budget=thread_budget,
                vocab=vocab,
                sample_sentences=clean_sample,
            )

        end_time = time.time()

        # Save model + metrics using cfg paths.
        # For the EXPERIMENTAL CP family, `tensor` is a tensorly CPTensor whose
        # first element is the λ weight vector — unpacking is identical, only
        # the saved container type differs (load_from_disk dispatches on the
        # CP filename tag).
        tl.set_backend("pytorch")
        core, factors = tucker_decomp_info["tensor"]
        errors = tucker_decomp_info["errors"]
        fitness_scores = tucker_decomp_info["fitness_scores"]

        core_t = tl.tensor(cp.asnumpy(core))
        factors_t = [tl.tensor(cp.asnumpy(f)) for f in factors]
        _decomposition = getattr(cfg.exp, "decomposition", "tucker")
        if _decomposition == "cp":
            from tensorly.cp_tensor import CPTensor
            tucker_decomp_torch = CPTensor((core_t, factors_t))
        else:
            tucker_decomp_torch = TuckerTensor((core_t, factors_t))

        torch.save(tucker_decomp_torch, paths["model"])
        np.save(paths["errors"], np.array([e.get() if hasattr(e, "get") else float(e) for e in errors], dtype=float))

        if fitness_scores:
            last = fitness_scores[-1]
            if isinstance(last, dict):
                with open(paths["fitness_json"], "w") as f:
                    json.dump(fitness_scores, f, indent=2)
            else:
                np.save(paths["fitness"], np.array([cp.asnumpy(f) for f in fitness_scores]))

        # Persist timings next to the other per-run artifacts so downstream
        # tooling (e.g. scripts/benchmarking.sh) can report a decomposition time
        # distinct from total process runtime. decomp_seconds is the loop-only
        # time from non_negative_tucker_with_similarity; runtime_seconds also
        # includes data loading and sparse conversion.
        with open(paths["timing_json"], "w") as f:
            json.dump(
                {
                    "decomp_seconds": tucker_decomp_info.get("decomp_seconds"),
                    "runtime_seconds": round(end_time - start_time, 2),
                },
                f,
                indent=2,
            )

        last_fitness = fitness_scores[-1] if fitness_scores else None
        if isinstance(last_fitness, dict):
            final_fitness = float(last_fitness[tucker_decomp_info["sem_primary_key"]])
            final_fitness_full = last_fitness
        else:
            final_fitness = float(last_fitness) if last_fitness is not None else None
            final_fitness_full = None

        # runs.jsonl is the shared run ledger; keep it on canonical GPFS and
        # record the canonical (post-copy-back) artifact locations.
        append_jsonl(
            final_paths["runs_jsonl"],
            {
                "timestamp": utc_now_iso(),
                "run_id": cfg.run_id(),
                "cfg": asdict(cfg),
                "results": {
                    "iterations": int(tucker_decomp_info["iterations"]),
                    "final_error": float(tucker_decomp_info["final_error"]) if tucker_decomp_info["final_error"] is not None else None,
                    "final_fitness": final_fitness,
                    "final_fitness_full": final_fitness_full,
                    "runtime_seconds": round(end_time - start_time, 2),
                    "model_path": str(final_paths["model"]),
                    "errors_path": str(final_paths["errors"]),
                    "fitness_path": str(final_paths["fitness"]),
                    "config_path": str(final_paths["config"]),
                },
            },
        )
        if not (cfg.exp.name and "bench" in cfg.exp.name):
            notify_discord(
                f"Saved {_decomposition.capitalize()} decomposition {cfg.exp.method} - "
                f"{cfg.exp.dim}/{cfg.exp.rank[0]} to {final_paths['model']}"
                f" in {end_time - start_time:.2f} seconds."
            )
        print("model, errors and config saved")

        return tucker_decomp_torch
    finally:
        # Flush staged artifacts back to shared GPFS once, at job end. This is
        # the single gentle sequential write per task that replaces the storm of
        # concurrent in-loop writes. Runs in `finally` so a crash or early stop
        # still preserves whatever save_intermediate staged on node-local SSD.
        if cfg.train.hpc:
            stage_artifacts_back(paths, final_paths)

