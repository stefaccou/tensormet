"""Judge ablation over several judge models, one judge per GPU, all GPUs at once.

Same results as ``python judge_ablation.py --models ...`` (one JSON
per (model, prompt) in --out-dir, then the summary table), but the judge models are
shared out over the GPUs: each GPU takes the next model from a queue (largest
first), judges every prompt with it, unloads it and takes the next one. Only the
trials stored in the session files are judged, so no decomposition is loaded.

On ampere, inside screen (conda activate ccl):
    cd 5_evaluation/COLING
    python judge_test.py                        # default models, GPUs 0 1 2 3
    python judge_test.py --models Qwen/Qwen3.5-2B microsoft/Phi-4-mini-instruct

Gated models (Llama) need the licence accepted on huggingface.co and
`huggingface-cli login` first; without that they fail and the rest carry on.
Gemma is left out: judge.py loads fp16 only, and Gemma overflows in fp16.

Per-GPU output goes to <out-dir>/gpu<N>.log; the terminal gets progress only.
"""
from __future__ import annotations

import argparse
import multiprocessing as mp
import os
import queue
import re
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path

COLING_DIR = Path(__file__).resolve().parent

MODELS = [
    "meta-llama/Llama-3.2-3B-Instruct",     # gated
    "meta-llama/Llama-3.1-8B-Instruct",     # gated
    "microsoft/Phi-4-mini-instruct",
    "allenai/OLMo-2-1124-7B-Instruct",
    "ibm-granite/granite-3.3-8b-instruct",
    "Qwen/Qwen3.5-2B",
    "Qwen/Qwen3.5-4B",
    "Qwen/Qwen3.5-9B",
]
CHUNK = 256  # rows per forward pass, still capped to free VRAM (judge default: 64)


def _size_b(model_name: str) -> float:
    """Parameter count in billions read off the name ("...-9B"); unknown -> 8."""
    m = re.search(r"(?<![\d.])(\d+(?:\.\d+)?)b(?![a-z])", model_name.lower())
    return float(m.group(1)) if m else 8.0


def _worker(gpu, model_q, result_q, trial_sets, prompts, out_dir, overwrite, chunk):
    """Judges models from `model_q` on one GPU until it gets None."""
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)  # before any CUDA call
    os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
    log = open(Path(out_dir) / f"gpu{gpu}.log", "a", buffering=1, encoding="utf-8")
    sys.stdout = sys.stderr = log
    try:
        import gc

        import torch

        from judge_ablation import judge_model
        while (model_name := model_q.get()) is not None:
            print(f"\n[{datetime.now():%H:%M:%S}] {model_name}")
            result_q.put(("start", gpu, model_name, None))
            t0 = time.perf_counter()
            try:
                info = judge_model(model_name, prompts, trial_sets, Path(out_dir), overwrite=overwrite,
                                   require_gpu=True, device="cuda:0", chunk=chunk)
                result_q.put(("done", gpu, model_name, {**info, "seconds": time.perf_counter() - t0}))
            except Exception as e:
                traceback.print_exc()
                result_q.put(("failed", gpu, model_name, f"{type(e).__name__}: {e}"))
                gc.collect()
                torch.cuda.empty_cache()
    except BaseException:
        traceback.print_exc()
        result_q.put(("crashed", gpu, None, traceback.format_exc()))
        raise
    result_q.put(("exit", gpu, None, None))


def main(argv=None):
    from judge_ablation import PROMPTS, load_trial_sets, print_human_agreement, summarize

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--models", nargs="+", default=MODELS, help="HF repo ids of the judge models")
    ap.add_argument("--prompts", nargs="+", default=list(PROMPTS), choices=list(PROMPTS))
    ap.add_argument("--gpus", nargs="+", default=["0", "1", "2", "3"], help="physical GPU ids")
    ap.add_argument("--sessions-dir", type=Path, default=COLING_DIR / "human_eval_results")
    ap.add_argument("--pattern", default="*.jsonl", help="glob for session files")
    ap.add_argument("--out-dir", type=Path, default=COLING_DIR / "judge_ablation_results")
    ap.add_argument("--chunk", type=int, default=CHUNK)
    ap.add_argument("--overwrite", action="store_true", help="rejudge cached trial sets")
    args = ap.parse_args(argv)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    print("Sessions:")
    trial_sets = load_trial_sets(args.sessions_dir, args.pattern)
    print("Human-human agreement:")
    print_human_agreement(trial_sets)

    # Largest first, so the small models fill in the gaps at the end.
    models = sorted(dict.fromkeys(args.models), key=_size_b, reverse=True)
    print(f"\n{len(models)} judge model(s) x {len(args.prompts)} prompt(s) on GPUs {args.gpus}; "
          f"logs in {args.out_dir}/gpu<N>.log")

    ctx = mp.get_context("spawn")
    model_q, result_q = ctx.Queue(), ctx.Queue()
    for m in models:
        model_q.put(m)
    for _ in args.gpus:
        model_q.put(None)
    procs = {g: ctx.Process(target=_worker, name=f"GPU {g}", daemon=True,
                            args=(g, model_q, result_q, trial_sets, args.prompts,
                                  str(args.out_dir), args.overwrite, args.chunk))
             for g in args.gpus}

    t0 = time.perf_counter()
    running, done, failed = {}, {}, {}
    exited = set()
    try:
        for p in procs.values():
            p.start()
        while len(exited) < len(procs):
            try:
                kind, gpu, model_name, payload = result_q.get(timeout=10)
            except queue.Empty:
                # A process killed from outside (OOM killer, segfault) sends nothing.
                for g, p in procs.items():
                    if g not in exited and p.exitcode is not None:
                        exited.add(g)
                        if g in running:
                            failed[running.pop(g)] = f"GPU {g} process died (exit code {p.exitcode})"
                continue
            stamp = f"[{datetime.now():%H:%M:%S}] GPU {gpu}:"
            if kind == "start":
                running[gpu] = model_name
                print(f"{stamp} {model_name} ...")
            elif kind == "done":
                running.pop(gpu, None)
                done[model_name] = payload
                mem = payload["gpu_memory_gb"]
                print(f"{stamp} {model_name} done in {payload['seconds'] / 60:.1f} min "
                      f"(judged {payload['judged'] or '-'}, cached {payload['cached'] or '-'}"
                      + (f", {mem:.1f} GB" if mem else "")
                      + (f", {payload['n_nan']} NAN TRIALS" if payload["n_nan"] else "") + ")")
            elif kind == "failed":
                running.pop(gpu, None)
                failed[model_name] = payload
                print(f"{stamp} {model_name} FAILED: {payload}")
            elif kind == "crashed":
                exited.add(gpu)
                if gpu in running:
                    failed[running.pop(gpu)] = "worker crashed"
                print(f"{stamp} worker crashed:\n{payload}")
            elif kind == "exit":
                exited.add(gpu)
    except KeyboardInterrupt:
        print("\ninterrupted; finished (model, prompt) files are kept and reused on the next run")
        return 130
    finally:
        for p in procs.values():
            if p.is_alive():
                p.terminate()
        model_q.cancel_join_thread()

    print(f"\n{len(done)}/{len(models)} judge models done in {(time.perf_counter() - t0) / 60:.1f} min")
    for m, why in failed.items():
        print(f"  FAILED {m}: {why}")
    not_run = [m for m in models if m not in done and m not in failed]
    if not_run:
        print(f"  never started: {not_run}")
    print()
    summarize(args.out_dir, trial_sets)
    return 1 if failed or not_run else 0


if __name__ == "__main__":
    sys.exit(main())
