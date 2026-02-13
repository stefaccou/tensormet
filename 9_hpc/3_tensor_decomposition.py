from tensorly.tucker_tensor import validate_tucker_rank, tucker_normalize, TuckerTensor
from tensorly.tenalg import mode_dot
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
                   utc_now_iso,
                   readonly_dispatch
                   )
from tensormet.tucker_tensor import SparseTupleTensor, TuckerDecomposition
from tensormet.sparse_ops import initialize_nonnegative_tucker
from tensormet.similarity import load_eval_sentences_cached, evaluate_sample, ensure_vocab
from tensormet.routing import get_update_routing_step, get_log_step
import argparse


# argument parsing
def parse_args():
    parser = argparse.ArgumentParser(
        description="Populate tensors from vector files"
    )
    parser.add_argument(
        "--tier1",
        action="store_true",
        help="Route read-only inputs through /readonly/ (leave outputs unchanged).",
    )
    parser.add_argument(
        "--top-ks",
        nargs="+",
        type=int,
        default=None,
        help="List of top-k values"
    )

    return parser.parse_args()


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
        divergence = cfg.exp.divergence
        dim = cfg.exp.dim
        n_iter_max = cfg.train.n_iter_max
        init = cfg.train.init
        tol = cfg.train.tol
        epsilon = cfg.train.epsilon
        random_state = cfg.exp.random_state
        verbose = cfg.train.verbose
        return_errors = cfg.train.return_errors
        normalize_factors = cfg.train.normalize_factors
        patience = cfg.train.patience
        warmup_steps = cfg.train.warmup_steps
        rec_check_every = cfg.eval.rec_check_every
        sem_check_every = cfg.eval.sem_check_every
        sem_error_type = cfg.eval.sem_error_type
        # logging
        rec_log_every = cfg.eval.rec_log_every
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
    no_rec_improve_steps = 0
    last_err = None

    sem_no_rec_improve_steps = 0
    best_sem_score = 0
    best_sem_iteration = None

    for iteration in range(n_iter_max):
        log_step = get_log_step(iteration, rec_log_every, rec_check_every)
        routing = get_update_routing_step(divergence=divergence, dim=dim, log_step=log_step)
        # --- factors ---
        for mode in modes:
            factors[mode] = routing.factor_update(
                vec_tensor=sparse_tensor.tensor,
                core=core,
                factors=factors,
                mode=mode,
                shape=shape,
                thread_budget=thread_budget,
                epsilon=epsilon,
            )

        # --- core + error ---
        if routing.core_returns_error:
            # FR: combined core update + error in one call
            core, rel_err = routing.core_update(
                vec_tensor=sparse_tensor.tensor,
                shape=shape,
                core=core,
                factors=factors,
                modes=modes,
                thread_budget=thread_budget,  # we always pass it, even if not needed, to ensure consistency
                epsilon=epsilon,
            )
        else:
            # KL: core update, then compute error separately
            core = routing.core_update(
                vec_tensor=sparse_tensor.tensor,
                shape=shape,
                core=core,
                factors=factors,
                modes=modes,
                thread_budget=thread_budget,
                epsilon=epsilon,
            )

            rel_err = routing.error_fn(
                vec_tensor=sparse_tensor.tensor,
                shape=shape,
                core=core,
                factors=factors,
                thread_budget=thread_budget,
                epsilon=epsilon,
            )

        # Normalize if desired
        if normalize_factors:
            core, factors = tucker_normalize((core, factors))

        if log_step:
            rec_errors.append(rel_err)

            # ---- reconstruction + patience ----
            has_prev_err = len(rec_errors) >= 2
            if verbose and has_prev_err:
                delta = rec_errors[-2] - rec_errors[-1]
                print(f"{iteration}: reconstruction error={rec_errors[-1]} (Δ={delta:+.3e})")

            do_rec_check = (
                    rec_check_every > 0
                    and (iteration + 1) % rec_check_every == 0
            )
            # patience only after warmup and once we have a previous error
            if do_rec_check:
                if rel_err is None:
                    raise ValueError("erro should always be available on error checking steps")

                if not last_err:
                    last_err = rel_err
                elif iteration >= warmup_steps:
                    imp_val = abs(float(last_err - rel_err))
                    if imp_val < tol:
                        no_rec_improve_steps += 1
                        if verbose:
                            print(f"No significant change: {no_rec_improve_steps}/{patience} (Δ={imp_val:.3e})")
                        if no_rec_improve_steps >= patience:
                            if verbose:
                                notify_discord(
                                    f"Stopped after {no_rec_improve_steps} non-improving steps "
                                    f"(patience={patience}). Converged at iteration {iteration} with final error {rec_errors[-1]}",
                                    job_finished=False,
                                )
                            break
                    else:
                        if verbose and no_rec_improve_steps:
                            print(f"Improved (Δ={imp_val:.3e}); resetting patience counter.")
                        no_rec_improve_steps = 0
                    last_err = rel_err

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
                return_type=sem_error_type
            )
            fitness_scores.append(fitness_score)
            tl.set_backend("cupy")

            # track best semantic model
            diff = float(fitness_score - best_sem_score)
            if diff > 0:
                best_sem_score = fitness_score
                best_core = core.copy()
                best_factors = factors.copy()
                best_sem_iteration = iteration
                if verbose:
                    print("New best semantic score; saving current best core and factors.")

            # semantic patience (uses the same tol/patience)
            if diff < tol:
                sem_no_rec_improve_steps += 1
                if verbose:
                    print(f"\tNo semantic improvement: {sem_no_rec_improve_steps}/{patience} (Δ={diff:.3e})")
                if sem_no_rec_improve_steps >= patience:
                    if verbose:
                        notify_discord(
                            f"Stopped after {sem_no_rec_improve_steps} non-improving semantic steps "
                            f"(patience={patience}). Converged at iteration {iteration}.",
                            job_finished=False,
                        )
                    break
            else:
                if verbose and sem_no_rec_improve_steps:
                    print(f"\tSemantic improvement (Δ={diff:.3e}); resetting patience counter.")
                sem_no_rec_improve_steps = 0

    if best_sem_iteration:
        tensor = TuckerTensor((best_core, best_factors))
        iteration = best_sem_iteration
    else:
        tensor = TuckerTensor((core, factors))
    if return_errors == "simple":
        return tensor, rec_errors
    elif return_errors == "full":
        return {"tensor": tensor, "errors": rec_errors, "fitness_scores": fitness_scores,
                "iterations": iteration + 1, "final_error": rec_errors[-1] if len(rec_errors) > 0 else None}
    else:
        return tensor


print("parsing args")

args = parse_args()
print("Preparing environment for cupy sparse non-negative tucker decomposition with similarity checks.")
# device = select_gpu() NOT NEEDED ON HPC
tl.set_backend("cupy")

# we perform the non negative tucker
print("\nStarting non-negative Tucker decomposition on cupy sparse tensor")

dataset = "fineweb-en"
divergences = ["fr", "kl"]
name = "largedim"
methods = ["siiSoftPlus"]

dims = args.top_ks if args.top_ks is not None else [2000, 4000, 6000]

ranks = [(150, 150, 150)]
iters = 500

tol = 1e-5
random_state = 1

patience = 5
rec_check_every = 5
rec_log_every = 5
sem_check_every = 10
# choose one of: "{absolute|average}_{rank|prob}_score"
sem_error_type = "absolute_prob_score"
max_cpu_frac = 1
thread_budget = ThreadBudget(n_threads=compute_num_threads(max_cpu_frac))
print(thread_budget, "thread budget computed")

path_to_vectors = readonly_dispatch(DATA_DIR/ "vectors" / "fineweb_english_vectors.csv", args.tier1)
sentence_sample = load_eval_sentences_cached(vector_path=path_to_vectors,
                                             dataset=dataset,
                                             seed=random_state,
                                             n_samples=10_000,
                                             )
print("loaded in sample")
remove_OOV = False



for dim in dims:
    for divergence in divergences:
        for rank in ranks:
            vocab_path = os.path.join(DATA_DIR, "tensors", dataset, f"vocabularies/{dim}.pkl")
            vocab_path = readonly_dispatch(vocab_path, args.tier1)
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
                        data_dir=Path(DATA_DIR), # this should never be dispatched to readonly
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


                # if paths["model"].exists():
                #     print(f"Decomposition already exists at {paths['model']}, skipping...")
                #     continue

                # Save config snapshot (single JSON)
                write_json(paths["config"],
                           {"timestamp": utc_now_iso(), "run_id": cfg.run_id(), "cfg": asdict(cfg)})

                tl.set_backend("cupy")
                start_time = time.time()
                print("loading in the sparse tensor")
                sparse_tensor = SparseTupleTensor.load_from_disk(
                    dataset=dataset,
                    method=method,
                    dims=dim,
                    map_location="cpu",
                    tier1=args.tier1
                )
                sparse_tensor.tensor_to_sparse("cupy")

                print("starting decomposition")
                # tucker_decomp_info = non_negative_tucker_with_similarity(
                #     sparse_tensor=sparse_tensor,
                #     cfg=cfg,
                #     thread_budget=thread_budget,
                #     vocab=vocab,
                #     sample_sentences=clean_sample,
                # )

                try:
                    checkpoint_tucker = TuckerDecomposition.load_from_disk(
                        dataset=dataset,
                        method=method,
                        divergence=divergence,
                        dims=dim,
                        rank=150,
                        name=name,
                        iterations=iters
                        )
                    print("continuing decomposition from checkpoint")
                except:
                    print("starting decomposition from scratch")
                    checkpoint_tucker = None

                tucker_decomp_info = sparse_tensor.non_negative_tucker_with_similarity(
                    cfg=cfg,
                    thread_budget=thread_budget,
                    vocab=vocab,
                    sample_sentences=clean_sample,
                    checkpoint_tensor=checkpoint_tucker,
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
