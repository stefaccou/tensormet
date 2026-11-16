"""
training.py — the decomposition training loop, in reading order:

    check_run_config      every guard rail, before anything expensive runs
    load_judge            the dimension-consistency judge (optional)
    evaluate_semantics    sample evaluation + SimLex + judge on the current model
    semantic_check / reconstruction_check
                          the two patience tracks, best-state bookkeeping
    save_checkpoint, graceful_interrupt
    train, run_loop       build the solver; the loop itself

What differs between MU and SGD is behind ``solvers`` (step / views / snapshot /
checkpoint), what differs between Tucker, TT, utt and CP is in ``families``.
``SparseTupleTensor.non_negative_tucker_with_similarity`` calls ``train``.
"""
from __future__ import annotations

import json
import os
import signal
import time
import types
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, List, Optional

import numpy as np
import tensorly as tl
import torch

from tensormet.distance import coords_nnz
from tensormet.families import get_family
from tensormet.hpc_helpers import mirror_checkpoint
from tensormet.routing import get_log_step, needs_largedim
from tensormet.similarity import (
    SIMLEX_PATH, evaluate_sample, evaluate_simlex, load_simlex, simlex_vectors,
)
from tensormet.solvers import _as_host, effective_subsample_frac, make_solver, resolve_rank
from tensormet.utils import (
    extract_roles_from_vocab, nontrivial_linked_groups, notify_discord,
    replay_sem_patience, sync_devices, write_best_record,
)


# ---------------------------------------------------------------------------
# Guard rails
# ---------------------------------------------------------------------------

def check_run_config(cfg, tensor):
    """Raise on every combination of options that is not implemented, before an
    init, an SVD or a judge is paid for."""
    e, ev = cfg.exp, cfg.eval
    sgd = e.solver == "sgd"
    if e.tt_core_init != "random" and e.decomposition not in ("tt", "utt"):
        raise ValueError(f"--tt-core-init {e.tt_core_init} needs decomposition 'tt' or "
                         f"'utt'; got {e.decomposition!r}")
    if e.decomposition not in ("tucker", "cp", "tt", "utt"):
        raise ValueError(
            f"cfg.exp.decomposition must be 'tucker', 'cp', 'tt' or 'utt'; got {e.decomposition!r}")
    if e.solver not in ("mu", "sgd"):
        raise ValueError(f"cfg.exp.solver must be 'mu' or 'sgd'; got {e.solver!r}")
    if ev.dim_consistency_method not in ("score", "similarity", "both"):
        raise ValueError(f"cfg.eval.dim_consistency_method must be one of "
                         f"'score', 'similarity', 'both'; got {ev.dim_consistency_method!r}")
    if e.binary:
        from tensormet.experimental.binary import check_binary
        check_binary(cfg)
    if e.objective not in ("full", "masked"):
        raise ValueError(f"cfg.exp.objective must be 'full' or 'masked', got {e.objective!r}")
    masked = e.objective == "masked"

    # The SGD trainer is torch-native; the tensor from load_from_disk is already
    # a torch sparse COO, so no CuPy conversion ever happens.
    if sgd and tensor.sparsity_type != "torch":
        raise ValueError("solver='sgd' needs sparsity_type 'torch' "
                         f"(got {tensor.sparsity_type!r}); skip tensor_to_sparse('cupy').")
    if not sgd and tensor.sparsity_type != "cupy":
        raise ValueError("sparse_tensor must have sparsity_type 'cupy'.")

    # SGD rejects MU-only knobs whose semantics don't carry over to a minibatch solver.
    if sgd:
        if e.decomposition not in ("tucker", "tt", "utt"):
            raise NotImplementedError(
                f"solver='sgd' with decomposition={e.decomposition!r} is not implemented "
                "(sgd/README.md). Use decomposition 'tucker', 'tt' or 'utt'.")
        if e.decomposition == "utt" and e.utt_core_power is not None:
            raise ValueError(
                "--utt-core-power sets the exponent of the MU core step; solver='sgd' "
                "has no such step (its step size is --sgd-lr). Drop the flag.")
        if e.tt_core_init != "random" and e.init == "random":
            raise ValueError(
                f"solver='sgd' with init='random' draws its own cores; --tt-core-init "
                f"{e.tt_core_init} only applies to SVD inits (e.g. --init svd_stream).")
        if e.subsample_frac < 1.0:
            raise ValueError("solver='sgd' is already minibatch; subsample_frac < 1.0 has no "
                             "meaning there. Use --sgd-batch-size instead.")
        if e.max_nnz:
            raise ValueError("solver='sgd' does not support max_nnz (MU per-step NNZ ceiling). "
                             "Use --sgd-batch-size instead.")
        if e.normalize_factors:
            raise ValueError("solver='sgd' does not support normalize_factors=True (scaling "
                             "lives in the softplus/exp parametrization).")
        if cfg.train.largedim:
            raise ValueError("solver='sgd' does not use the largedim kernel family; "
                             "drop --largedim.")

    # utt shares one core across all positions, so every mode must share one factor too.
    if e.decomposition == "utt":
        groups = nontrivial_linked_groups(tensor.shared_factors, num_factors=len(e.rank))
        if [sorted(g) for g in groups] != [list(range(len(e.rank)))]:
            raise ValueError(
                f"decomposition='utt' shares one core across all positions, so every mode "
                f"must share one factor too (shared_factors='all'); got groups {groups}.")

    if masked and isinstance(e.init, str) and "svd" in e.init:
        # SVD init is a zero-filled HOSVD (it treats unobserved entries as 0), the
        # same "full" assumption the masked objective is meant to avoid.
        print(f"WARNING: init={e.init!r} with objective='masked' is not implemented. "
              "SVD init fits a zero-filled HOSVD (unobserved entries treated as 0), "
              "which contradicts the masked/completion objective and will probably "
              "worsen results. Consider --init random.")
    if masked and cfg.train.n_gpus == 1:
        # The single-GPU masked kernels don't receive `frac` to rescale their
        # (observed-only) denominators under subsampling; the sharded path does it.
        nnz = int(tensor.tensor._nnz()) if sgd else coords_nnz(tensor.tensor)
        if effective_subsample_frac(cfg, nnz) < 1.0:
            raise NotImplementedError(
                "objective='masked' with subsampling (subsample_frac < 1.0 or a binding "
                "max_nnz) is only supported on the multi-GPU sharded path (n_gpus > 1). "
                "Use subsample_frac=1.0 and max_nnz=0 on a single GPU.")
    if e.decomposition == "cp" and masked:
        raise NotImplementedError("decomposition='cp' does not support objective='masked' yet "
                                  "(CP_IMPLEMENTATION_PLAN.md §1.8 / Phase 5). Use objective='full'.")
    if e.decomposition in ("tt", "utt") and masked and not sgd:
        raise NotImplementedError(
            "decomposition='tt' does not support objective='masked' with the MU "
            "solver (tt_hybrid/README.md, 'Not implemented'). Use objective='full' "
            "or --solver sgd.")


# ---------------------------------------------------------------------------
# The judge, and the semantic evaluation
# ---------------------------------------------------------------------------

def load_judge(cfg, solver):
    """``(judge or None, on_its_own_gpu)``.

    The judge model (~1 GB fp16 for the default small judge, 3.5 GB for the 2B
    one) is loaded BEFORE the loop sizes any GPU batches, not lazily at the
    first semantic check: the batch estimators size against free VRAM, and a
    judge added mid-run steals memory that batches sized beforehand already
    assumed was free (acute on resumed runs, where the pool is grown to fill
    the GPU before the first check). With ``--judge-gpu``, select_gpu() appended
    the judge's GPU after the decomposition's, so it is logical device
    ``n_gpus``; it then needs no pool trimming and no fixed batch cap.
    """
    ev = cfg.eval
    if not ev.dim_consistency:
        return None, False
    from tensormet.judge import DimConsistencyJudge, DEFAULT_JUDGE_MODEL
    n_gpus = cfg.train.n_gpus
    model_name = ev.dim_consistency_model or DEFAULT_JUDGE_MODEL
    dedicated = False
    if ev.judge_gpu is not None:
        if torch.cuda.device_count() > n_gpus:
            dedicated = True
        else:
            print(f"Warning: judge GPU {ev.judge_gpu} is not visible to this process "
                  f"(launched without select_gpu?); sharing the decomposition's GPU.")
    judge = DimConsistencyJudge(
        model_name=model_name, num_dim_words=ev.dim_consistency_words,
        diversity_aware=ev.dim_consistency_diversity,
        **({"device": torch.device("cuda", n_gpus), "chunk": None} if dedicated else {}))
    where = (f"dedicated GPU {ev.judge_gpu} (cuda:{n_gpus}), batch sized to fit" if dedicated
             else f"decomposition's GPU, batch <= {judge.chunk}")
    print(
        f"Dimension-consistency scoring enabled (method={ev.dim_consistency_method!r}, "
        f"judge={model_name!r}, words/dim={ev.dim_consistency_words}, "
        f"diversity_aware={ev.dim_consistency_diversity}, on {where}). "
        f"Loading the judge now so the decomposition's batch sizing "
        f"accounts for its (model-dependent) GPU footprint from the first iteration.")
    # Return cached-but-unused blocks to the driver so torch gets real headroom.
    if not dedicated:
        solver.free_cache()
    judge.ensure_loaded()
    return judge, dedicated


@dataclass
class Run:
    """Everything the loop mutates."""
    cfg: Any
    solver: Any
    thread_budget: Any
    paths: dict
    mirror_paths: Optional[dict]
    sem_primary_key: str
    rec_errors: List = field(default_factory=list)
    fitness_scores: List = field(default_factory=list)
    last_err: Any = None
    no_rec_improve: int = 0
    sem_no_improve: int = 0
    best_state: Any = None            # (core, factors): the best-semantics model so far
    best_iteration: Optional[int] = None   # 0-based: the state after iteration k is k - 1
    best_score: float = 0.0
    best_is_stand_in: bool = False    # the checkpoint stands in for an unrecorded best
    simlex_pairs: Any = None
    judge: Any = None
    judge_dedicated: bool = False


def evaluate_semantics(run, vocab, sample_sentences):
    """Sample evaluation, SimLex and the judge on the model as it is now. Returns
    the score dict (or the single score when ``sem_error_type`` names one)."""
    cfg, solver, ev = run.cfg, run.solver, run.cfg.eval
    # For MU this flips cupy -> pytorch for the eval stack (restored below).
    tl.set_backend("pytorch")
    core, factors = solver.eval_views()
    roles = extract_roles_from_vocab(vocab)
    decomp = solver.family.make_decomposition(core, factors, vocab, roles)

    sem_out = evaluate_sample(
        decomp, sample_sentences, sampled=True, seed=cfg.exp.random_state,
        thread_budget=run.thread_budget, return_type=ev.sem_error_type,
        softmax_temperature=ev.sem_softmax_temperature)

    if run.simlex_pairs is not None:
        try:
            simlex_out = evaluate_simlex(run.simlex_pairs, simlex_vectors(decomp, roles),
                                         verbose=cfg.train.verbose)
            if isinstance(sem_out, dict) and isinstance(simlex_out, dict):
                scores = simlex_out.get("ALL", {})
                flat = {}
                if isinstance(scores, dict):
                    if scores.get("rho") is not None:
                        flat["simlex_all_rho"] = scores["rho"]
                    if scores.get("pval") is not None:
                        flat["simlex_all_pval"] = scores["pval"]
                sem_out = {**sem_out, **flat}
        except Exception as err:
            print(f"Warning: SimLex evaluation failed ({err}); skipping.")

    if run.judge is not None:
        try:
            # CuPy's pool keeps freed blocks cached, so torch could fail to
            # cudaMalloc room for the judge's activations: trim right before
            # scoring. A judge on its own GPU needs none of this.
            if not run.judge_dedicated:
                solver.free_cache()
            try:
                scores = {}
                if ev.dim_consistency_method in ("score", "both"):
                    scores.update(run.judge.score(decomp, seed=cfg.exp.random_state,
                                                  verbose=False))
                if ev.dim_consistency_method in ("similarity", "both"):
                    scores.update(run.judge.score_similarity_consistency(
                        decomp, seed=cfg.exp.random_state, verbose=False))
            finally:
                # score()'s forward passes leave torch's caching allocator holding
                # "reserved" activation memory. The sharded core update's batch
                # sizes are estimated once and cached across iterations, so a
                # lingering reservation would silently shrink the headroom the
                # NEXT core update assumed was free: give it back right away.
                if (not run.judge_dedicated and run.judge.device is not None
                        and run.judge.device.type == "cuda"):
                    torch.cuda.empty_cache()
            if isinstance(sem_out, dict):
                sem_out = {**sem_out, **scores}
        except Exception as err:
            print(f"Warning: dimension-consistency scoring failed ({err}); skipping.")

    # The views would otherwise keep this iteration's arrays alive, and torch's
    # cached eval buffers would eat the VRAM the next update's batch sizing
    # assumes is free.
    del decomp, core, factors
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    tl.set_backend(solver.backend)
    return sem_out


# ---------------------------------------------------------------------------
# The two patience tracks
# ---------------------------------------------------------------------------

def reconstruction_check(run, iteration, rel_err, seconds):
    """Log the reconstruction error; True when its patience ran out."""
    cfg = run.cfg
    tc, ev = cfg.train, cfg.eval
    run.rec_errors.append(rel_err)
    if len(run.rec_errors) >= 2:
        delta = run.rec_errors[-2] - run.rec_errors[-1]
        message = f"{iteration}: reconstruction error={run.rec_errors[-1]} (Δ={delta:+.3e})"
        if ev.time_iteration:
            message += f", time={seconds}"
        print(message)

    if not (ev.rec_check_every > 0 and (iteration + 1) % ev.rec_check_every == 0):
        return False
    if rel_err is None:
        raise ValueError("error should always be available on error checking steps")
    if run.last_err is None:
        run.last_err = rel_err
    elif iteration >= tc.warmup_steps:  # patience only after warmup and a previous error
        improvement = abs(float(run.last_err - rel_err))
        if improvement < tc.tol:
            run.no_rec_improve += 1
            if tc.verbose:
                print(f"No significant change: {run.no_rec_improve}/{tc.patience} (Δ={improvement:.3e})")
            if run.no_rec_improve >= tc.patience:
                if tc.verbose:
                    notify_discord(
                        f"Stopped after {run.no_rec_improve} non-improving steps "
                        f"(patience={tc.patience}). Converged at iteration {iteration} "
                        f"with final error {run.rec_errors[-1]}", job_finished=False)
                return True
        else:
            if run.no_rec_improve:
                print(f"Improved (Δ={improvement:.3e}); resetting patience counter.")
            run.no_rec_improve = 0
        run.last_err = rel_err
    return False


def semantic_check(run, iteration, vocab, sample_sentences):
    """Evaluate, log, keep the best model; True when the semantic patience ran out."""
    cfg, solver = run.cfg, run.solver
    tc, ev, paths, key = cfg.train, cfg.eval, run.paths, run.sem_primary_key
    sem_out = evaluate_semantics(run, vocab, sample_sentences)

    run.fitness_scores.append(sem_out)
    if ev.save_intermediate:  # every check, so the log is complete while the run is going
        if isinstance(sem_out, dict):
            with open(paths["fitness_json"], "w") as f:
                json.dump(run.fitness_scores, f, indent=2)
        else:
            np.save(paths["fitness"], np.array(run.fitness_scores, dtype=float))

    # The primary value drives early stopping and the best model.
    if isinstance(sem_out, dict) and key not in sem_out:
        judge_key = key.startswith(("dim_consistency", "similarity_consistency"))
        if key.startswith("simlex_") and run.simlex_pairs is not None:
            print(f"Warning: '{key}' not available (all SimLex pairs OOV?); skipping sem check.")
            return False
        if judge_key and run.judge is not None:
            # Judge scoring failed this check (warning printed above), or the
            # configured dim_consistency_method doesn't produce this key.
            print(f"Warning: '{key}' not available (judge scoring failed, "
                  f"or dim_consistency_method={ev.dim_consistency_method!r} doesn't produce it?); "
                  f"skipping sem check.")
            return False
        if judge_key:
            raise KeyError(
                f"Primary semantic key '{key}' requires dimension-consistency "
                f"judge scoring; enable it with --dim-consistency true and "
                f"--dim-consistency-method matching this key (score/similarity/both).")
        raise KeyError(f"Primary semantic key '{key}' missing from returned scores.")

    if isinstance(sem_out, dict):
        value, everything = float(sem_out[key]), json.dumps(sem_out)
    else:
        value, everything = float(sem_out), str(sem_out)
    print(f"Iteration {iteration + 1}\tRec_error: {run.rec_errors[-1] if run.rec_errors else None}\t"
          f"Sem({key}): {value}\tSem_all: {everything}")

    diff = value - float(run.best_score)
    if diff > 0:
        run.best_score = value
        run.best_state = solver.snapshot()
        run.best_iteration = iteration
        run.best_is_stand_in = False
        if tc.verbose:
            print("New best semantic score; saving current best core and factors.")
        if ev.save_intermediate:
            # Host arrays (like checkpoints): judge_eval / inspect_tucker later load
            # this file on CPU.
            torch.save(solver.container(run.best_state), paths["model"])
            write_best_record(paths["best_json"], iteration + 1, value, key)
            print("saving temp model to", paths["model"])
            np.save(paths["errors"], np.array([_as_host(e) for e in run.rec_errors]))

    if diff < tc.tol:
        run.sem_no_improve += 1
        if tc.verbose:
            print(f"\tNo semantic improvement: {run.sem_no_improve}/{tc.patience} (Δ={diff:.3e})")
        if run.sem_no_improve >= tc.patience:
            if tc.verbose:
                notify_discord(
                    f"Stopped after {run.sem_no_improve} non-improving semantic steps "
                    f"(patience={tc.patience}). Converged at iteration {iteration}.",
                    job_finished=False)
            return True
    else:
        if run.sem_no_improve:
            print(f"\tSemantic improvement (Δ={diff:.3e}); resetting patience counter.")
        run.sem_no_improve = 0
    return False


# ---------------------------------------------------------------------------
# Checkpoints and Ctrl+C
# ---------------------------------------------------------------------------

def save_checkpoint(run, iteration, before_mirror=None):
    """Write checkpoint ``iteration + 1`` and mirror it (HPC mode: to GPFS, so a
    walltime kill stays resumable). ``before_mirror`` runs in between, for files
    the mirror should pick up as well. Returns the checkpoint path."""
    path = run.paths["checkpoint_dir"] / f"{iteration + 1}.pt"
    torch.save(run.solver.checkpoint(iteration), path)
    if before_mirror is not None:
        before_mirror()
    mirror_checkpoint(run.paths, run.mirror_paths, path.name)
    return path


def log_checkpoint(run, iteration):
    """One line per periodic checkpoint: the last errors and scores."""
    key = run.sem_primary_key
    last = run.fitness_scores[-1] if run.fitness_scores else None
    if isinstance(last, dict):
        primary, dump = last.get(key), json.dumps(last)
    else:
        primary, dump = last, (str(last) if last is not None else None)
    rec_error = run.rec_errors[-1] if run.rec_errors else None
    with open(run.paths["checkpoint_dir"] / "log.txt", "a") as f:
        f.write(f"Iteration {iteration + 1}\tRec_error: {rec_error}\t"
                f"Sem({key}): {primary}\tSem_all: {dump}\n")


def save_interrupt_checkpoint(run, iteration):
    """The first Ctrl+C: write the same artifacts a periodic checkpoint does
    (plus errors / fitness), so get_resume_state() can pick the run up. A second
    Ctrl+C during the save raises KeyboardInterrupt, exiting immediately."""
    print(f"Saving resumable checkpoint at iteration {iteration} before stopping...")
    paths = run.paths

    def errors_and_fitness():
        np.save(paths["errors"], np.array([_as_host(e) for e in run.rec_errors]))
        if run.fitness_scores:
            if isinstance(run.fitness_scores[-1], dict):
                with open(paths["fitness_json"], "w") as f:
                    json.dump(run.fitness_scores, f, indent=2)
            else:
                np.save(paths["fitness"], np.array(run.fitness_scores, dtype=float))

    try:
        os.makedirs(paths["checkpoint_dir"], exist_ok=True)
        path = save_checkpoint(run, iteration, before_mirror=errors_and_fitness)
        print(f"Resumable checkpoint saved to {path}")
    except KeyboardInterrupt:
        raise
    except Exception as err:
        print(f"Failed to save resumable checkpoint: {err}")


@contextmanager
def graceful_interrupt():
    """First Ctrl+C (SIGINT): set ``.requested`` so the loop saves a resumable
    checkpoint at the end of the current iteration, and restore the default
    handler so a SECOND Ctrl+C exits immediately. signal.signal only works from
    the main thread; elsewhere the default behaviour stays."""
    state = types.SimpleNamespace(requested=False)
    original = None
    installed = False

    def handler(signum, frame):
        print("\nInterrupt received: will save a resumable checkpoint at the end of "
              "the current iteration. Press Ctrl+C again to exit immediately.")
        state.requested = True
        signal.signal(signal.SIGINT, original)

    try:
        original = signal.signal(signal.SIGINT, handler)
        installed = True
    except ValueError:
        pass
    try:
        yield state
    finally:
        if installed:
            signal.signal(signal.SIGINT, original)


# ---------------------------------------------------------------------------
# The loop
# ---------------------------------------------------------------------------

def sem_primary_key_of(ev):
    """The semantic metric that drives patience, the best model and logging."""
    if ev.sem_primary_key is not None:
        return ev.sem_primary_key
    kind = ev.sem_error_type
    if kind == "all":
        return "average_rank_score"  # stable default (the dict always includes it)
    if isinstance(kind, (list, tuple)):
        if len(kind) == 0:
            raise ValueError("sem_error_type list/tuple must contain at least one key.")
        return kind[0]
    return kind


def train(tensor, cfg, thread_budget, vocab=None, sample_sentences=None):
    check_run_config(cfg, tensor)
    if cfg.train.checkpoint_saving_steps:
        os.makedirs(cfg.artifact_paths()["checkpoint_dir"], exist_ok=True)
    resume = cfg.get_resume_state()
    rank, modes = resolve_rank(tuple(tensor.shape), cfg)
    solver = make_solver(tensor, cfg, get_family(cfg), rank, modes, thread_budget, resume)
    try:
        return run_loop(solver, cfg, rank, resume, thread_budget, vocab, sample_sentences)
    finally:
        solver.close()  # a tensor parked in host RAM goes back on the GPU


def run_loop(solver, cfg, rank, resume, thread_budget, vocab, sample_sentences):
    e, tc, ev = cfg.exp, cfg.train, cfg.eval
    paths = cfg.artifact_paths()
    # Under HPC mode `paths` resolve to node-local $TMPDIR; periodic checkpoints
    # are mirrored to the canonical GPFS paths so a walltime kill stays resumable.
    mirror_paths = cfg.artifact_paths(staged=False) if tc.hpc else None
    start_iteration = resume.get("start_iteration", 0)

    run = Run(cfg=cfg, solver=solver, thread_budget=thread_budget, paths=paths,
              mirror_paths=mirror_paths, sem_primary_key=sem_primary_key_of(ev),
              rec_errors=resume.get("rec_errors", []),
              fitness_scores=resume.get("fitness_scores", []),
              best_score=resume.get("best_sem_score", 0.0))
    run.last_err = run.rec_errors[-1] if run.rec_errors else None  # for the early-stop diff
    best = resume.get("best_tensor")
    if best is not None:
        # The old run's best-semantic model file (get_resume_state checked that its
        # _best.json matches best_sem_score). Host arrays.
        core, factors = (best.core, best.factors) if hasattr(best, "core") else best
        run.best_state = (_as_host(core), [_as_host(f) for f in factors])
        run.best_iteration = resume["best_iteration"] - 1
    else:
        if start_iteration > 0:
            print("Resume: the old run left no best-state record; the checkpoint stands in as "
                  "the best state until semantics improve.")
        run.best_state = solver.snapshot()
        run.best_iteration = start_iteration - 1 if start_iteration > 0 else None
        run.best_is_stand_in = start_iteration > 0
    if start_iteration > 0 and run.fitness_scores:
        run.sem_no_improve = replay_sem_patience(run.fitness_scores, run.sem_primary_key, tc.tol)
        print(f"Resume: semantic patience restored at {run.sem_no_improve}/{tc.patience}.")

    if SIMLEX_PATH.exists():
        try:
            run.simlex_pairs = load_simlex(SIMLEX_PATH)
        except Exception as err:
            print(f"Warning: could not load SimLex-999 from {SIMLEX_PATH}: {err}")
    run.judge, run.judge_dedicated = load_judge(cfg, solver)
    solver.size_batches()  # after the judge is resident: batch sizes depend on free VRAM

    print(e.divergence, rank, solver.subsample_frac)
    dim = e.dim
    reached = needs_largedim(dim, largedim=tc.largedim, masked=e.objective == "masked")
    print(f"routing path: {solver.describe()} (max_dim={max(dim) if isinstance(dim, (tuple, list)) else dim}, "
          f"LARGEDIM_THRESHOLD reached={reached}, n_gpus={tc.n_gpus})")

    rec_log_every = ev.rec_log_every or ev.rec_check_every
    pool_trim_every = ev.pool_trim_every if ev.pool_trim_every is not None else ev.sem_check_every
    # Per-iteration solve time (updates + error, excluding semantic eval). GPU work
    # is async, so the timer is bracketed by sync_devices() on log steps only:
    # syncing every iteration serializes the loop.
    iter_seconds = []
    loop_start = time.time()
    iteration = start_iteration - 1  # defined even if the loop body never executes
    with graceful_interrupt() as interrupt:
        for iteration in range(start_iteration, tc.n_iter_max):
            log_step = get_log_step(iteration, rec_log_every, ev.rec_check_every)
            if log_step:
                sync_devices(tc.n_gpus, solver.sync_backend)
            started = time.time()
            rel_err = solver.step(iteration, log_step)
            if log_step:
                sync_devices(tc.n_gpus, solver.sync_backend)
            iter_seconds.append(time.time() - started)

            if log_step and reconstruction_check(run, iteration, rel_err, iter_seconds[-1]):
                break
            if (sample_sentences is not None and vocab is not None and ev.sem_check_every > 0
                    and (iteration + 1) % ev.sem_check_every == 0
                    and semantic_check(run, iteration, vocab, sample_sentences)):
                break
            if tc.checkpoint_saving_steps and (iteration + 1) % tc.checkpoint_saving_steps == 0:
                print(f"saving model at iteration {iteration}")
                save_checkpoint(run, iteration)
                log_checkpoint(run, iteration)
            # Reclaim transient eval/copy blocks so out-of-pool cuBLAS/cuSPARSE
            # workspaces keep their headroom. Deliberately NOT per iteration.
            if pool_trim_every and (iteration + 1) % pool_trim_every == 0:
                solver.free_cache()
            if interrupt.requested:
                save_interrupt_checkpoint(run, iteration)
                break

    decomp_seconds = time.time() - loop_start
    solve_seconds = float(sum(iter_seconds))
    print(f"decomposition time: {solve_seconds:.2f}s "
          f"(sum of {len(iter_seconds)} iteration time(s), device-synced at log steps"
          + (f", mean {solve_seconds / len(iter_seconds):.2f}s" if iter_seconds else "")
          + f"); {decomp_seconds:.2f}s for the whole loop including in-loop evaluation")
    final_error = solver.final_error(run.rec_errors)

    last_iteration = iteration + 1  # iterations run (like checkpoint names)
    if run.best_iteration is not None:
        model = solver.container(run.best_state, host=False)
        iteration = run.best_iteration
    else:
        model = solver.container(solver.current(), host=False)
    return {
        "tensor": model,
        "errors": run.rec_errors,
        "fitness_scores": run.fitness_scores,
        "sem_primary_key": run.sem_primary_key,
        "iterations": iteration + 1,  # the returned state's iteration (kept for old readers)
        # explicit: the best-semantic state returned (None: no check improved, the final
        # state is returned) and how far the loop got
        "best_iteration": None if run.best_iteration is None else run.best_iteration + 1,
        "best_sem_score": (None if run.best_iteration is None or run.best_is_stand_in
                           else float(run.best_score)),
        "last_iteration": last_iteration,
        "final_error": final_error,
        "decomp_seconds": decomp_seconds,
        "solve_seconds": solve_seconds,
        "iter_seconds": list(iter_seconds),
    }
