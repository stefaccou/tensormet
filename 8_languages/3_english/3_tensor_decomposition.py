import tensorly as tl
import numpy as np
import torch
import os
import cupy as cp
import time
import pickle
from pathlib import Path
from dataclasses import asdict

from tensormet.config import (ExperimentConfig,
                    TrainingConfig,
                    EvalConfig,
                    RunConfig
                    )

from tensormet.utils import (DATA_DIR, select_gpu,
                   notify_discord,
                   compute_num_threads,
                   ThreadBudget,
                   write_json,
                   append_jsonl,
                   utc_now_iso
                   )
from tensormet.tucker_tensor import SparseTupleTensor
from tensormet.similarity import load_eval_sentences_cached, ensure_vocab
from tensorly.tucker_tensor import TuckerTensor


# --- The actual magic ---



# -- Example run --
print("Preparing environment for cupy sparse non-negative tucker decomposition with similarity checks.")
device = select_gpu()
tl.set_backend("cupy")

# we perform the non negative tucker
print("\nStarting non-negative Tucker decomposition on cupy sparse tensor")

dataset = "fineweb-en"
divergences = ["fr"]
name = "quicktest"
methods = ["siiSoftPlus"]
dims = [1000]
ranks = [(100, 100, 100)]
iters = 500

tol = 1e-5
random_state = 1

patience = 5
rec_check_every = 1
rec_log_every = 1
sem_check_every = 10
# choose one of: "{absolute|average}_{rank|prob}_score"
sem_error_type = "absolute_prob_score"
max_cpu_frac = 0.85
thread_budget = ThreadBudget(n_threads=compute_num_threads(max_cpu_frac))

# we load the sample sentences only once
vector_path = os.path.join(DATA_DIR, "vectors", "fineweb_english_vectors.csv")
sentence_sample = load_eval_sentences_cached(vector_path=vector_path,
                                             dataset=dataset,
                                             seed=random_state,
                                             n_samples=10_000,
                                             )
remove_OOV = False

for dim in dims:
    for divergence in divergences:
        for rank in ranks:
            vocab_path = os.path.join(DATA_DIR, "tensors", dataset, f"vocabularies/{dim}.pkl")
            with open(vocab_path, "rb") as f:
                vocab = pickle.load(f)

            if remove_OOV:
                start = time.time()
                clean_sample = ensure_vocab(vocab, sentence_sample)
                print("cleaned sample in ", time.time() - start)
            else:
                clean_sample = sentence_sample


            for method in methods:
                cfg = RunConfig(
                    exp=ExperimentConfig(
                        dataset=dataset,
                        method=method,
                        divergence=divergence,
                        dim=dim,
                        rank=tuple(rank),
                        name=name,
                        random_state=random_state,
                        max_cpu_frac=max_cpu_frac,
                        data_dir=Path(DATA_DIR),
                    ),
                    train=TrainingConfig(
                        n_iter_max=iters,
                        tol=tol,
                        epsilon=1e-12,
                        warmup_steps=1 if divergence == "kl" else 50,
                        patience=patience,
                        normalize_factors=False,
                        verbose=True,
                        return_errors="full",
                    ),
                    eval=EvalConfig(
                        rec_log_every=rec_log_every,
                        rec_check_every=rec_check_every,
                        sem_check_every=sem_check_every,
                        sem_error_type=sem_error_type,
                        remove_OOV=remove_OOV,
                        time_iteration=True,
                        save_intermediate=True
                    ),
                )
                print(cfg)
                paths = cfg.artifact_paths()
                for p in paths.values():
                    if isinstance(p, Path):
                        p.parent.mkdir(parents=True, exist_ok=True)

                # If model already exists, skip (optional but recommended)
                if paths["model"].exists():
                    print(f"Decomposition already exists at {paths['model']}, skipping...")
                    continue

                # Save config snapshot (single JSON)
                write_json(paths["config"],
                           {"timestamp": utc_now_iso(), "run_id": cfg.run_id(), "cfg": asdict(cfg)})

                tl.set_backend("cupy")
                start_time = time.time()

                sparse_tensor = SparseTupleTensor.load_from_disk(
                    dataset=dataset,
                    method=method,
                    dims=dim,
                    map_location="cpu",
                )

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
                    f"Saved Tucker decomposition {method} - {dim}/{rank[0]} to {paths['model']}"
                    f" in {end_time - start_time:.2f} seconds."
                )
                print("model, errors and config saved")
