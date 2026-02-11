from tensorly_custom.tucker_tensor import validate_tucker_rank, tucker_normalize, TuckerTensor
from tensorly_custom.decomposition._tucker import tucker_to_tensor
from tensorly_custom.base import unfold
from tensorly_custom.tenalg import mode_dot
from datetime import datetime, UTC
import tensorly_custom as tl
import pytensorlab as ptl
import numpy as np
import torch
import os
from typing import List, Tuple
import cupy as cp
import cupyx.scipy.sparse as cpx_sparse
import time
import json
import pickle
from pathlib import Path
from dataclasses import asdict

from utils import (DATA_DIR, select_gpu, tree_to_device,
                   notify_discord, compute_num_threads,
                   ThreadBudget, print_elapsed_time,
                   write_json, append_jsonl,
                   utc_now_iso
                   )
from tucker_tensor import SparseTupleTensor, TuckerDecomposition
from sparse_ops import (unfold_from_vectorized_sparse,
                        sparse_multi_mode_dot_vec,
                        ptl_tucker_to_tensor,
                        gather_dense_at_block_nz,
                        initialize_nonnegative_tucker,
                        )
from similarity import (load_og_sentences,
                        load_eval_sentences_cached,
                        evaluate_sample)

from config import (ExperimentConfig,
                    TrainingConfig,
                    EvalConfig,
                    RunConfig,)

from distance import (kl_factor_update,
                      kl_core_update,
                      kl_compute_errors, kl_factor_update_largedim)


# --- The actual magic ---
def non_negative_tucker_with_similarity(
    sparse_tensor,
    cfg: RunConfig,
    thread_budget: ThreadBudget,
    vocab=None,
    sample_sentences=None,
):
    # unpacking the config
    try:
        rank = list(cfg.exp.rank)
        divergence = cfg.train.divergence
        n_iter_max = cfg.train.n_iter_max
        init = cfg.train.init
        tol = cfg.train.tol
        epsilon = cfg.train.epsilon
        random_state = cfg.exp.random_state
        verbose = cfg.train.verbose
        warmup_steps = cfg.train.warmup_steps
        return_errors = cfg.train.return_errors
        normalize_factors = cfg.train.normalize_factors
        patience = cfg.train.patience
        sem_check_every = cfg.eval.sem_check_every
    except:
        raise ValueError("Check config structure.")

    if not isinstance(sparse_tensor, SparseTupleTensor):
        raise TypeError("sparse_tensor must be a SparseTupleTensor instance.")
    if not sparse_tensor.sparsity_type == "cupy":
        raise ValueError("sparse_tensor must have sparsity_type 'cupy'.")


    shape = tuple(sparse_tensor.shape)
    rank = validate_tucker_rank(shape, rank=rank)
    modes = list(range(len(rank)))
    core, factors = initialize_nonnegative_tucker(sparse_tensor.tensor, shape, rank, modes, init, random_state)


    rec_errors = []
    fitness_scores = []
    no_improve_steps = 0
    sem_no_improve_steps = 0
    best_sem_score = 0
    conv_iteration = None

    for iteration in range(n_iter_max):
        for mode in modes:
            factors[mode] = kl_factor_update_largedim(sparse_tensor.tensor, core, factors, mode, shape, epsilon)

        core = kl_core_update(sparse_tensor.tensor, core, factors, modes, shape, thread_budget, epsilon)
        # Normalize (keeps Tucker scale convention consistent with your training loop)
        if normalize_factors:
            core, factors = tucker_normalize((core, factors))


        # kl_val, rel_kl = kl_compute_errors( -> Changed to only return relative
        rel_kl = kl_compute_errors(
            vec_tensor=sparse_tensor.tensor,
            core=core,
            factors=factors,
            shape=shape,
            thread_budget=thread_budget,
            epsilon=epsilon,
        )
        rec_errors.append(rel_kl)


        # ---- reconstruction + patience ----
        has_prev_err = len(rec_errors) >= 2
        if verbose and has_prev_err:
            delta = rec_errors[-2] - rec_errors[-1]
            print(f"{iteration}: reconstruction error={rec_errors[-1]} (Δ={delta:+.3e})")

        # patience only after warmup and once we have a previous error
        if iteration >= warmup_steps and has_prev_err:
            imp_val = abs(float(rec_errors[-2] - rec_errors[-1]))

            if imp_val < tol:
                no_improve_steps += 1
                if verbose:
                    print(f"No improvement: {no_improve_steps}/{patience} (Δ={imp_val:.3e})")
                if no_improve_steps >= patience:
                    if verbose:
                        notify_discord(
                            f"Stopped after {no_improve_steps} non-improving steps "
                            f"(patience={patience}). Converged at iteration {iteration} with final error {rec_errors[-1]}",
                            job_finished=False,
                        )
                    break
            else:
                if verbose and no_improve_steps:
                    print(f"Improved (Δ={imp_val:.3e}); resetting patience counter.")
                no_improve_steps = 0

        # ---- similarity evaluation + semantic patience ----
        do_sem_check = (
                sample_sentences is not None
                and vocab is not None
                and sem_check_every > 0
                and (iteration + 1) % sem_check_every == 0
        )

        if do_sem_check:
            tl.set_backend("pytorch")
            core_cpu = tl.tensor(cp.asnumpy(core))
            factors_cpu = [tl.tensor(cp.asnumpy(f)) for f in factors]
            tucker_decomp = TuckerDecomposition(core=core_cpu, factors=factors_cpu, vocab=vocab)

            fitness_score = evaluate_sample(
                tucker_decomp,
                sample_sentences,
                sampled=True,
                seed=random_state,
                thread_budget=thread_budget,
            )
            fitness_scores.append(fitness_score)
            tl.set_backend("cupy")

            # track best semantic model
            diff = float(fitness_score - best_sem_score)
            if diff > 0:
                best_sem_score = fitness_score
                best_core = core.copy()
                best_factors = factors.copy()
                conv_iteration = iteration
                if verbose:
                    print("New best semantic score; saving current best core and factors.")

            # semantic patience (uses the same tol/patience)
            if diff < tol:
                sem_no_improve_steps += 1
                if verbose:
                    print(f"\tNo semantic improvement: {sem_no_improve_steps}/{patience} (Δ={diff:.3e})")
                if sem_no_improve_steps >= patience:
                    if verbose:
                        notify_discord(
                            f"Stopped after {sem_no_improve_steps} non-improving semantic steps "
                            f"(patience={patience}). Converged at iteration {iteration}.",
                            job_finished=False,
                        )
                    break
            else:
                if verbose and sem_no_improve_steps:
                    print(f"\tSemantic improvement (Δ={diff:.3e}); resetting patience counter.")
                sem_no_improve_steps = 0

    if conv_iteration:
        tensor = TuckerTensor((best_core, best_factors))
        iteration = conv_iteration
    else:
        tensor = TuckerTensor((core, factors))
    if return_errors == "simple":
        return tensor, rec_errors
    elif return_errors == "full":
        return {"tensor": tensor, "errors": rec_errors, "fitness_scores": fitness_scores,
                "iterations": iteration + 1, "final_error": rec_errors[-1]}
    else:
        return tensor


print("Preparing environment for cupy sparse non-negative tucker decomposition with similarity checks.")
device = select_gpu()
tl.set_backend("cupy")

# we perform the non negative tucker
print("\nStarting non-negative Tucker decomposition on cupy sparse tensor")

dataset = "fineweb_large"
name = "klTest"
methods = ["sc"]
dims = [4000]
ranks = [(100, 100, 100), (150, 150, 150)]
iters = 1000

tol = 1e-5
random_state = 1

patience = 5
sem_check_every = 20
max_cpu_frac = 0.9
thread_budget = ThreadBudget(n_threads=compute_num_threads(max_cpu_frac))

# we load the sample sentences only once
vector_path = os.path.join(DATA_DIR, "vectors", "fineweb_dutch_10000000.csv")
sentence_sample = load_eval_sentences_cached(vector_path=vector_path,
                                             dataset="fineweb_large",
                                             seed=random_state,
                                             n_samples=10_000,
                                             )

for dim in dims:
    for rank in ranks:
        vocab_path = os.path.join(DATA_DIR, "tensors", dataset, f"vocabularies/{dim}.pkl")
        with open(vocab_path, "rb") as f:
            vocab = pickle.load(f)

        for method in methods:
            cfg = RunConfig(
                exp=ExperimentConfig(
                    dataset=dataset,
                    method=method,
                    dim=dim,
                    rank=tuple(rank),
                    name=name,
                    random_state=random_state,
                    max_cpu_frac=max_cpu_frac,
                    data_dir=Path(DATA_DIR),
                ),
                train=TrainingConfig(
                    divergence="kl",
                    n_iter_max=iters,
                    tol=tol,
                    epsilon=1e-12,
                    warmup_steps=1,
                    patience=patience,
                    normalize_factors=False,
                    verbose=True,
                    return_errors="full",
                ),
                eval=EvalConfig(
                    sem_check_every=sem_check_every
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

            tucker_decomp_info = non_negative_tucker_with_similarity(
                sparse_tensor=sparse_tensor,
                cfg=cfg,
                thread_budget=thread_budget,
                vocab=vocab,
                sample_sentences=sentence_sample,
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