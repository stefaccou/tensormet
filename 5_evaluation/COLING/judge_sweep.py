"""Tea-leaves LLM judge over every model of the evaluation suite, on all GPUs at once.

The judge is the one the ablation in human_evaluation.ipynb (section "Judge as one more annotator",
judge_ablation.py) found closest to the annotators: Qwen/Qwen3.5-9B with the default
prompt. It does the task the annotators did: per dimension the top k = 4 words plus one teaLeaves
intruder, trial seed 1 (tealeaves_human.build_tealeaves_tasks: the trials judge.score poses).

Rows, by group:
    table      the models of eval_utils.MODELS in two series: latest (last checkpoint), best (model file)
    baselines  glove_<w>, glove_nmf_<w>, w2v on our words (eval_utils.load_baselines)
    methods    POLAR, SPINE, SPOWV, NNSE, SINr on our words (method_baselines.py)
    human      the trial sets in human_eval_results/, judged on their stored trials

Report latest. The best state is the one training selected with the 2B judge's dim_consistency on this
same task (k = 5), so best is a check, not the headline.

Speed: one judge per GPU, fed parts of trial sets from a shared queue, while several CPU processes load
the models and send only the trials. A trial set is judged once per (judge, prompt): identical trials
(best = latest, a human set = a table model) and every set of an earlier sweep come from records.jsonl.

Outputs in judge_results/:
    records.jsonl         one record per (model, series, seed) and sweep: every trial and the judge's pick
    judge_<stamp>.json    manifest: settings, models loaded / missing, status
    judge_<stamp>.log     everything printed, by every process
    summary_<stamp>.csv   the report's numbers, one row per (model, series)

On ampere, inside screen (conda activate ccl):
    cd 5_evaluation/COLING
    python judge_sweep.py --dry-run          # load every model, print the plan; no GPU
    python judge_sweep.py                    # one judge per visible GPU (--gpus 0 1 2 3 to choose)
    python judge_sweep.py --report-only      # the newest sweep's report again; loads nothing
"""
from __future__ import annotations

import argparse
import io
import json
import multiprocessing as mp
import os
import platform
import queue
import sys
import time
import traceback
from contextlib import redirect_stdout
from datetime import datetime
from pathlib import Path

COLING_DIR = Path(__file__).resolve().parent
SESSIONS_DIR = COLING_DIR / "human_eval_results"
OUT_DIR = COLING_DIR / "judge_results"
RECORDS_PATH = OUT_DIR / "records.jsonl"

JUDGE = "Qwen/Qwen3.5-9B"  # the judge ablation's closest to the annotators
PROMPT = "default"         # judge_ablation.PROMPTS: tensormet.judge.DEFAULT_JUDGE_MESSAGES
K = 4                      # top words per trial, as in the human sessions
SEEDS = [1]
SERIES = ("latest", "best")
GROUPS = ("table", "baselines", "methods", "human")
LOADERS = 4
CHUNK = 256       # rows per forward pass (the judge's default is 64); still capped to the free VRAM
PART_SIZE = 256   # trials per GPU work unit
PROGRESS_EVERY = 60  # seconds


def _tee(log_path):
    """This process's output to the terminal and the sweep log."""
    from polar_sweep import _Tee
    fh = open(log_path, "a", buffering=1, encoding="utf-8")
    sys.stdout = _Tee(sys.__stdout__, fh)
    sys.stderr = _Tee(sys.__stderr__, fh)


# --- Loading (CPU processes) ------------------------------------------------------
def _trial_sets(emit, name, group, series, decomp, settings, **meta):
    """Sends the trials of `decomp`'s first role, one set per seed."""
    import numpy as np
    from tensormet.experimental.tealeaves_human import build_tealeaves_tasks, tasks_fingerprint
    from tensormet.utils import to_np

    k = settings["k"]
    F = np.asarray(to_np(decomp.factors[decomp.get_role_index(decomp.roles[0])]))
    stats = dict(n_words=int(F.shape[0]),
                 # dimensions whose k-th highest loading is <= 0: fewer than k positive words to show
                 n_short_dims=int((np.partition(-F, k - 1, axis=0)[k - 1] >= 0).sum()),
                 n_bad_dims=int((~np.isfinite(F)).any(0).sum()))
    for seed in settings["seeds"]:
        try:
            tasks = build_tealeaves_tasks(decomp, num_dim_words=k, seed=seed)
        except ValueError as e:  # no teaLeaves intruder for some dimension
            emit(("missing", f"{name} {series} seed {seed}", str(e)))
            continue
        emit(("trial_set", dict(name=name, group=group, series=series, seed=seed, k=k,
                                fingerprint=tasks_fingerprint(tasks), **stats, **meta, tasks=tasks)))


def _load_table(run, settings, emit):
    """One table model, both series from one load: the model file is 'best'; 'latest' is then the latest
    checkpoint loaded over it, else the model file (eval_utils.load_decomposition's rule)."""
    import eval_utils as eu
    missing = {}
    for name, config, tk in eu.iter_decompositions(only=[run], missing=missing, checkpoint="model_file"):
        if "best" in settings["series"]:
            _trial_sets(emit, name, "table", "best", tk, settings, config=config, model_path=str(tk.decomp_path),
                        checkpoint=eu.checkpoint_info(tk.model_file, tk.decomp_path))
        if "latest" in settings["series"]:
            try:
                tk.update_from_path()
            except FileNotFoundError:
                pass
            _trial_sets(emit, name, "table", "latest", tk, settings, config=config, model_path=str(tk.decomp_path),
                        checkpoint=eu.checkpoint_info(tk.model_file, tk.decomp_path))
    for name, why in missing.items():
        emit(("missing", name, why))


def _emit_embeddings(models, missing, group, settings, emit):
    from glove_baseline import _as_baseline
    for name, m in models.items():
        emb = _as_baseline(m["words"], m["E_raw"], ["w"], name)
        _trial_sets(emit, name, group, "-", emb, settings, config=m["config"], model_path=m["path"],
                    checkpoint=None)
    for name, why in missing.items():
        emit(("missing", name, why))


def _load_baselines(names, settings, emit):
    import eval_utils as eu
    models, missing = eu.load_baselines(vocabs=("ours",), only=names)
    _emit_embeddings(models, missing, "baselines", settings, emit)


def _load_methods(names, settings, emit):
    import method_baselines as mb
    models, missing = mb.load_methods(mb.build_parser().parse_args(["--vocabs", "ours", "--only", *names]))
    _emit_embeddings(models, missing, "methods", settings, emit)


def _load_human(_, settings, emit):
    """The annotated trial sets, one per fingerprint, as stored in the session files."""
    from eval_utils import selected
    from tensormet.experimental.tealeaves_human import open_session, tasks_fingerprint
    sets = {}
    for path in sorted(SESSIONS_DIR.glob("*.jsonl")):
        s = open_session(path=path, verbose=False)
        entry = sets.setdefault(tasks_fingerprint(s.tasks), dict(meta=s.meta, tasks=s.tasks, sessions=[]))
        entry["sessions"].append(path.name)
    for fp, e in sets.items():
        name = f"human:{e['meta']['model']}"
        if not selected(name, settings["only"]):
            continue
        emit(("trial_set", dict(name=name, group="human", series="-", seed=e["meta"]["seed"],
                                k=e["meta"]["num_dim_words"], fingerprint=fp,
                                n_words=None, n_short_dims=None, n_bad_dims=None,
                                config=dict(role=e["meta"]["role"], sessions=e["sessions"]),
                                model_path=str(SESSIONS_DIR / e["sessions"][0]), checkpoint=None,
                                tasks=e["tasks"])))


JOBS = {"table": _load_table, "baselines": _load_baselines, "methods": _load_methods, "human": _load_human}


def _loader(job_q, result_q, settings, log_path):
    """Loads models job by job until it gets None; a failed job becomes a missing row."""
    _tee(log_path)
    try:
        while (job := job_q.get()) is not None:
            jid, kind, arg = job
            t0 = time.perf_counter()
            try:
                JOBS[kind](arg, settings, result_q.put)
            except Exception as e:
                traceback.print_exc()
                result_q.put(("missing", f"{kind} {arg}", f"{type(e).__name__}: {e}"))
            result_q.put(("job_done", jid, kind, arg, time.perf_counter() - t0))
    except BaseException:
        result_q.put(("error", "a loader", traceback.format_exc()))
        raise


# --- Judging (one process per GPU) ------------------------------------------------
def _gpu_worker(gpu, judge_name, messages, k, chunk, work_q, result_q, log_path):
    """One judge on one GPU: judges parts of trial sets until it gets None."""
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)  # before any CUDA call: this process sees only its GPU
    os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
    os.environ["HF_HUB_OFFLINE"] = "1"  # main downloaded the judge: read the cache, never the Hub
    _tee(log_path)
    try:
        from transformers.utils import logging as hf_logging

        from tensormet.judge import DimConsistencyJudge
        from judge_ablation import judge_trial_set
        hf_logging.disable_progress_bar()
        t0 = time.perf_counter()
        judge = DimConsistencyJudge(judge_name, num_dim_words=k, chunk=chunk, device="cuda:0", messages=messages)
        judge.ensure_loaded()
        if judge.device.type != "cuda":  # the judge falls back to CPU on OOM: far too slow for this
            raise RuntimeError(f"GPU {gpu}: no room for {judge_name}, the judge fell back to CPU")
        result_q.put(("gpu_ready", gpu, judge.gpu_memory_gb, time.perf_counter() - t0))
        while (item := work_q.get()) is not None:
            fp, part, tasks = item
            t = time.perf_counter()
            out = judge_trial_set(judge, tasks)
            result_q.put(("picks", fp, part, out, gpu, len(tasks), time.perf_counter() - t))
    except BaseException:
        result_q.put(("error", f"GPU {gpu}", traceback.format_exc()))
        raise


# --- Dispatching (main process) ---------------------------------------------------
def read_records(sweep=None):
    """Every record of records.jsonl, or those of one sweep (its stamp)."""
    out = []
    if RECORDS_PATH.exists():
        with open(RECORDS_PATH, encoding="utf-8") as fh:
            for line in fh:
                try:
                    r = json.loads(line)
                except ValueError:  # a line cut off by an interrupt
                    continue
                if sweep is None or r.get("sweep") == sweep:
                    out.append(r)
    return out


def load_cache(judge, messages):
    """fingerprint -> the picks of every trial set this judge and prompt already judged."""
    cache = {}
    for r in read_records():
        if r.get("judge") == judge and r.get("messages") == messages:
            cache[r["fingerprint"]] = dict(picks={t["dim"]: t["pick"] for t in r["tasks"]},
                                           scores={t["dim"]: t["scores"] for t in r["tasks"]},
                                           n_nan_scores=r["n_nan_scores"])
    return cache


class _Dispatcher:
    """Sends each new trial set to the GPUs in parts, reassembles the picks, appends the records."""

    def __init__(self, args, messages, stamp, cache, work_q):
        self.args, self.messages, self.stamp, self.work_q = args, messages, stamp, work_q
        self.done = dict(cache)  # fingerprint -> picks, scores, n_nan_scores
        self.cached = set(cache)
        self.pending = {}        # fingerprint -> parts expected, parts received, rows waiting for it
        self.planned = set()     # dry run: fingerprints seen
        self.runs, self.missing = [], {}
        self.new_sets = self.new_trials = self.parts_total = self.parts_done = self.n_finished = 0
        self.gpu_stats = {}      # gpu -> [trials, busy seconds]

    def trial_set(self, info):
        tasks = info.pop("tasks")
        run = {**info, "n_dims": len(tasks)}
        self.runs.append(run)
        fp = run["fingerprint"]
        if self.args.dry_run:
            state = "cached" if fp in self.done else "same" if fp in self.planned else "new"
            self.planned.add(fp)
            if state == "new":
                self.new_sets += 1
                self.new_trials += len(tasks)
            print(f"  {state:6s} {run['name']:42s} {run['series']:6s} seed {run['seed']}  {len(tasks)} dims")
        elif fp in self.done:
            self.finish(run, tasks, self.done[fp])
        elif fp in self.pending:  # the same trials as a row already on the GPUs
            self.pending[fp]["waiting"].append((run, tasks))
        else:
            size = self.args.part_size
            parts = [tasks[i:i + size] for i in range(0, len(tasks), size)]
            for i, part in enumerate(parts):
                self.work_q.put((fp, i, part))
            self.pending[fp] = dict(n=len(parts), got={}, waiting=[(run, tasks)])
            self.parts_total += len(parts)
            self.new_sets += 1
            self.new_trials += len(tasks)

    def picks(self, fp, part, out, gpu, n_tasks, seconds):
        st = self.gpu_stats.setdefault(gpu, [0, 0.0])
        st[0] += n_tasks
        st[1] += seconds
        self.parts_done += 1
        p = self.pending[fp]
        p["got"][part] = out
        if len(p["got"]) < p["n"]:
            return
        del self.pending[fp]
        merged = dict(picks={}, scores={}, n_nan_scores=0)
        for o in p["got"].values():
            merged["picks"].update(o["picks"])
            merged["scores"].update(o["scores"])
            merged["n_nan_scores"] += o["n_nan_scores"]
        self.done[fp] = merged
        for run, tasks in p["waiting"]:
            self.finish(run, tasks, merged)

    def finish(self, run, tasks, result):
        trials = [{**t, "pick": result["picks"][t["dim"]], "scores": result["scores"].get(t["dim"])}
                  for t in tasks]
        n = len(trials)
        n_correct = sum(t["pick"] == t["random_word"] for t in trials)
        # judge.score's diversity multiplier: distinct top words / (dims x k)
        diversity = len({w for t in trials for w in t["words"]}) / (n * len(trials[0]["words"]))
        rec = dict(time=datetime.now().isoformat(timespec="seconds"), sweep=self.stamp,
                   judge=self.args.judge, prompt=self.args.prompt, messages=self.messages, **run,
                   n_correct=n_correct, accuracy=n_correct / n, diversity=diversity,
                   dim_consistency=n_correct / n * diversity, n_nan_scores=result["n_nan_scores"],
                   cached=run["fingerprint"] in self.cached, tasks=trials)
        with open(RECORDS_PATH, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(rec, default=str) + "\n")
        self.n_finished += 1
        print(f"[{datetime.now():%H:%M:%S}] {run['name']:42s} {run['series']:6s} seed {run['seed']}  "
              f"acc={rec['accuracy']:.3f}  ({n} dims{', cached' if rec['cached'] else ''})")
        if rec["n_nan_scores"]:
            print(f"  WARNING: {rec['n_nan_scores']} trials got nan log-probs (fp16 overflow?)")
        if run.get("n_bad_dims"):
            print(f"  WARNING: {run['n_bad_dims']} dimensions have non-finite loadings")


def plan_jobs(args):
    """Loader jobs (kind, arg), slowest first so the loaders end together; and the table's row order."""
    import eval_utils as eu
    import method_baselines as mb

    def sel(names):
        return [n for n in names if eu.selected(n, args.only)]

    w2v, glove, released, table = [], [], [], []
    if "methods" in args.groups:
        mb_args = mb.build_parser().parse_args(["--vocabs", "ours"])
        for n in sel(f"polar_{s}_k{k}" for s in mb_args.polar_sources for k in mb_args.polar_dims):
            (w2v if "w2v" in n else glove).append(("methods", [n]))
        released = [("methods", [n]) for n in sel(mb.released_files(mb_args))]
    if "baselines" in args.groups:
        by_file = {}
        for n, (source, width, _, _) in eu.baseline_specs(vocabs=("ours",)).items():
            by_file.setdefault((source, width), []).append(n)
        for (source, _), names in by_file.items():
            if sel(names):
                (w2v if source == "w2v" else glove).append(("baselines", sel(names)))
    table_order = []
    if "table" in args.groups:
        table_order = sel(eu.run_name(c) for c in eu.configs(args.ngrams))
        table = [("table", n) for n in table_order]
    human = [("human", None)] if "human" in args.groups else []
    return human + w2v + glove + released + table, table_order


def visible_gpus():
    """The physical ids of the GPUs this shell sees."""
    env = os.environ.get("CUDA_VISIBLE_DEVICES")
    if env is not None:
        return [g.strip() for g in env.split(",") if g.strip()]
    import torch
    return [str(i) for i in range(torch.cuda.device_count())]


def _check_alive(procs):
    for p in procs:
        if p.exitcode not in (None, 0):
            raise RuntimeError(f"{p.name} exited with code {p.exitcode}; see the log")


def build_parser():
    from judge_ablation import PROMPTS
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--judge", default=JUDGE, help="HF repo id of the judge (default: %(default)s)")
    p.add_argument("--prompt", default=PROMPT, choices=list(PROMPTS))
    p.add_argument("--k", type=int, default=K, help="top words per trial (default: %(default)s, as the sessions)")
    p.add_argument("--seeds", type=int, nargs="+", default=SEEDS,
                   help="trial seeds (intruder draws); the report averages over them (default: %(default)s)")
    p.add_argument("--series", nargs="+", default=list(SERIES), choices=SERIES,
                   help="table models: latest = last checkpoint, best = model file")
    p.add_argument("--groups", nargs="+", default=list(GROUPS), choices=GROUPS)
    p.add_argument("--only", nargs="+", default=None, metavar="PATTERN",
                   help="row names to run (shell-style), e.g. 'tt_4g_*' 'nnse_*' 'human:*'")
    p.add_argument("--ngrams", nargs="+", type=int, default=None, help="table columns of these orders (default: all)")
    p.add_argument("--gpus", nargs="+", default=None, help="physical GPU ids, one judge each (default: all visible)")
    p.add_argument("--loaders", type=int, default=LOADERS, help="CPU processes loading models (default: %(default)s)")
    p.add_argument("--chunk", type=int, default=CHUNK, help="rows per forward pass, capped to free VRAM")
    p.add_argument("--part-size", type=int, default=PART_SIZE, help="trials per GPU work unit")
    p.add_argument("--rejudge", action="store_true", help="judge every trial set again, ignoring records.jsonl")
    p.add_argument("--dry-run", action="store_true", help="load every model and print the plan; no GPU")
    p.add_argument("--report-only", action="store_true", help="print the newest sweep's report; loads nothing")
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.report_only:
        report()
        return 0
    from judge_ablation import PROMPTS

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().isoformat(timespec="seconds")
    tag = stamp.replace(":", "")
    manifest_path, log_path = OUT_DIR / f"judge_{tag}.json", OUT_DIR / f"judge_{tag}.log"
    _tee(log_path)
    messages = PROMPTS[args.prompt]
    jobs, table_order = plan_jobs(args)
    gpus = [] if args.dry_run else (args.gpus or visible_gpus())
    if not args.dry_run and not gpus:
        raise SystemExit("No GPU visible; use --dry-run to only load the models.")
    n_loaders = max(1, min(args.loaders, len(jobs)))
    settings = dict(k=args.k, seeds=args.seeds, series=args.series, only=args.only)
    manifest = dict(stamp=stamp, status="loading", argv=sys.argv, host=platform.node(), pid=os.getpid(),
                    judge=args.judge, prompt=args.prompt, messages=messages, k=args.k, seeds=args.seeds,
                    series=args.series, groups=args.groups, only=args.only, ngrams=args.ngrams,
                    gpus=gpus, loaders=n_loaders, chunk=args.chunk, part_size=args.part_size,
                    rejudge=args.rejudge, table_order=table_order, records=str(RECORDS_PATH))

    def save(**kw):
        manifest.update(kw)
        manifest_path.write_text(json.dumps(manifest, indent=2, default=str))

    save()
    print(f"[{stamp}] judge sweep: manifest {manifest_path}")
    print(f"judge {args.judge} | prompt {args.prompt} | k={args.k} | seeds {args.seeds} | "
          f"{len(jobs)} loader jobs on {n_loaders} processes | GPUs {gpus or 'none (dry run)'}")
    cache = {} if args.rejudge else load_cache(args.judge, messages)
    print(f"{len(cache)} trial sets already judged with this judge and prompt in {RECORDS_PATH.name}")
    if gpus:
        # One Hub check / download here: GPU workers resolving it at the same time raced on the cache
        # ("does not appear to have a file named model.safetensors-00003-of-00004"); they load offline.
        from huggingface_hub import snapshot_download
        revision = Path(snapshot_download(args.judge)).name  # the snapshot's commit
        print(f"judge files in the local cache: {args.judge} @ {revision}")
        save(judge_revision=revision)

    ctx = mp.get_context("spawn")
    job_q, work_q, result_q = ctx.Queue(), ctx.Queue(), ctx.Queue()
    for i, (kind, arg) in enumerate(jobs):
        job_q.put((i, kind, arg))
    for _ in range(n_loaders):
        job_q.put(None)
    # GPUs first: loading the judge takes longest, and overlaps with the models loading
    procs = [ctx.Process(target=_gpu_worker, name=f"GPU {g}", daemon=True,
                         args=(g, args.judge, messages, args.k, args.chunk, work_q, result_q, str(log_path)))
             for g in gpus]
    procs += [ctx.Process(target=_loader, name=f"loader {i}", daemon=True,
                          args=(job_q, result_q, settings, str(log_path)))
              for i in range(n_loaders)]
    d = _Dispatcher(args, messages, stamp, cache, work_q)
    t0 = last = time.perf_counter()
    jobs_done = 0
    try:
        for p in procs:
            p.start()
        save(status="running" if gpus else "dry-run")
        while jobs_done < len(jobs) or d.pending:
            try:
                kind, *rest = result_q.get(timeout=5)
            except queue.Empty:
                _check_alive(procs)
                kind = None
            if kind == "trial_set":
                d.trial_set(*rest)
            elif kind == "picks":
                d.picks(*rest)
            elif kind == "missing":
                d.missing[rest[0]] = rest[1]
                print(f"  missing {rest[0]}: {rest[1]}")
            elif kind == "job_done":
                jobs_done += 1
                _, job_kind, arg, secs = rest
                print(f"[{datetime.now():%H:%M:%S}] loaded {job_kind} {arg or ''} in {secs:.0f} s "
                      f"({jobs_done}/{len(jobs)} loader jobs)")
            elif kind == "gpu_ready":
                gpu, mem, secs = rest
                print(f"[{datetime.now():%H:%M:%S}] GPU {gpu}: judge loaded in {secs:.0f} s ({mem:.1f} GB)")
            elif kind == "error":
                raise RuntimeError(f"{rest[0]} failed:\n{rest[1]}")
            if time.perf_counter() - last >= PROGRESS_EVERY:
                last = time.perf_counter()
                print(f"[{datetime.now():%H:%M:%S}] {jobs_done}/{len(jobs)} loader jobs, "
                      f"{d.parts_done}/{d.parts_total} GPU parts, {d.n_finished} rows done, "
                      f"{(last - t0) / 60:.1f} min")
        for _ in gpus:
            work_q.put(None)
        for p in procs:
            p.join(timeout=60)
    except KeyboardInterrupt:
        print(f"\ninterrupted; {d.n_finished} rows finished this sweep are in {RECORDS_PATH.name}")
        save(status="interrupted", minutes=round((time.perf_counter() - t0) / 60, 2))
        return 130
    except BaseException as e:
        save(status="failed", error=repr(e))
        raise
    finally:
        for p in procs:
            if p.is_alive():
                p.terminate()
        for q in (job_q, work_q):  # after an interrupt, unread items must not block the exit
            q.cancel_join_thread()

    minutes = (time.perf_counter() - t0) / 60
    save(status="dry-run" if args.dry_run else "finished", minutes=round(minutes, 2), runs=d.runs,
         missing=d.missing, new_trial_sets=d.new_sets, new_trials=d.new_trials,
         gpu_stats={g: dict(trials=n, busy_seconds=round(s, 1)) for g, (n, s) in d.gpu_stats.items()},
         finished_at=datetime.now().isoformat(timespec="seconds"))
    print(f"\n{len(d.runs)} rows, {d.new_sets} trial sets ({d.new_trials} trials) "
          f"{'to judge' if args.dry_run else 'judged'}, the rest cached or shared; {minutes:.1f} min")
    for g, (n, s) in sorted(d.gpu_stats.items()):
        print(f"  GPU {g}: {n} trials in {s:.0f} s busy ({n / max(s, 1e-9):.1f} trials/s)")
    if d.missing:
        print("missing:")
        for name, why in d.missing.items():
            print(f"  {name:42s} {why}")
    if not args.dry_run:
        report(manifest, write_csv=True)
    return 0


# --- Report -----------------------------------------------------------------------
def load_manifest(path=None):
    """A sweep's manifest (default: the newest finished or interrupted one)."""
    if path is None:
        found = [p for p in sorted(OUT_DIR.glob("judge_*.json"))
                 if json.loads(p.read_text()).get("status") in ("finished", "interrupted")]
        if not found:
            raise FileNotFoundError(f"no finished sweep in {OUT_DIR}; run judge_sweep.py first")
        path = found[-1]
    man = json.loads(Path(path).read_text())
    man["path"] = str(path)
    return man


def _show(title, df):
    """Prints `df`; columns n_* and iter* as integers, the rest to three decimals."""
    import pandas as pd
    df = df.copy()
    for c in df.columns:
        if str(c).startswith(("n_", "iter")):
            df[c] = pd.to_numeric(df[c], errors="coerce").round().astype("Int64")
    print(f"\n{title}")
    with pd.option_context("display.width", 250, "display.max_columns", None, "display.max_rows", None):
        print(df.to_string(float_format=lambda v: f"{v:.3f}"))


def report(manifest=None, write_csv=False):
    """Prints a sweep's tables (default: the newest finished one) and returns them as DataFrames."""
    import pandas as pd
    man = manifest if isinstance(manifest, dict) else load_manifest(manifest)
    recs = read_records(man["stamp"])
    if not recs:
        print(f"no records of sweep {man['stamp']} in {RECORDS_PATH}")
        return {}
    cols = ("group", "name", "series", "seed", "n_dims", "n_correct", "accuracy", "diversity", "dim_consistency",
            "n_short_dims", "n_bad_dims", "n_words", "n_nan_scores", "model_path", "fingerprint")
    per_seed = pd.DataFrame([{**{c: r.get(c) for c in cols},
                              "iteration": (r.get("checkpoint") or {}).get("iteration"),
                              "status": (r.get("checkpoint") or {}).get("status")} for r in recs])
    summary = (per_seed.groupby(["group", "name", "series"], sort=False)
               .agg(accuracy=("accuracy", "mean"), accuracy_sd=("accuracy", "std"), n_seeds=("seed", "nunique"),
                    n_dims=("n_dims", "first"), n_correct=("n_correct", "sum"), diversity=("diversity", "mean"),
                    dim_consistency=("dim_consistency", "mean"), n_short_dims=("n_short_dims", "first"),
                    n_bad_dims=("n_bad_dims", "first"), n_words=("n_words", "first"),
                    n_nan_scores=("n_nan_scores", "sum"), iteration=("iteration", "first"),
                    status=("status", "first"), model_path=("model_path", "first"),
                    fingerprint=("fingerprint", "first"))
               .reset_index())
    multi = summary.n_seeds.max() > 1
    out = {"summary": summary}
    print(f"\n=== Tea leaves, judge {man['judge']} | prompt {man['prompt']} | k={man['k']} | seeds {man['seeds']} "
          f"| sweep {man['stamp']} ===")
    print(f"acc = share of dimensions whose intruder the judge picked (chance {1 / (man['k'] + 1):.2f}); "
          f"dim_cons = acc x diversity (distinct top words / (dims x k)), as training's dim_consistency")

    table = summary[summary.group == "table"]
    if len(table):
        series = [s for s in SERIES if s in set(table.series)]
        by = {s: table[table.series == s].set_index("name") for s in series}
        head = by[series[0]]
        cols = {}
        for s in series:
            cols[f"acc_{s}"] = by[s].accuracy
            if multi:
                cols[f"sd_{s}"] = by[s].accuracy_sd
        for s in series:
            cols[f"iter_{s}"] = by[s].iteration
        cols.update({"status": head.status, "n_dims": head.n_dims, f"diversity_{series[0]}": head.diversity,
                     f"dim_cons_{series[0]}": head.dim_consistency})
        order = [n for n in man.get("table_order", []) if n in head.index]
        out["models"] = pd.DataFrame(cols).reindex(order + [n for n in head.index if n not in order])
        _show("Our models (latest = the headline; best = the state training picked with the 2B judge on this task)",
              out["models"])

    base = summary[summary.group.isin(["baselines", "methods"])]
    if len(base):
        keep = ["group", "accuracy"] + (["accuracy_sd"] if multi else []) + [
            "n_dims", "n_short_dims", "n_words", "diversity", "dim_consistency"]
        out["baselines"] = (base.set_index("name")[keep].sort_values(["group", "accuracy"], ascending=[True, False])
                            .rename(columns={"accuracy": "acc", "accuracy_sd": "sd", "dim_consistency": "dim_cons"}))
        _show("Baselines and interpretable methods, on our words (n_short_dims: dimensions with fewer than k "
              "positive words)", out["baselines"])

    human = [r for r in recs if r["group"] == "human"]
    if human:
        from tensormet.experimental.tealeaves_human import open_session
        rows, annotators = [], set()
        for r in human:
            row = {"set": r["name"].removeprefix("human:"), "n_dims": r["n_dims"], "judge": r["accuracy"]}
            for f in r["config"]["sessions"]:
                s = open_session(path=SESSIONS_DIR / f, verbose=False)
                who, summ = s.meta["annotator"], s.summary()
                annotators.add(who)
                row[who] = summ["human_dim_consistency_raw"] if summ["n_scored"] else float("nan")
                row[f"n_{who}"] = summ["n_scored"]
            row["same trials as"] = ", ".join(sorted(f"{x['name']} ({x['series']})" for x in recs
                                                     if x["fingerprint"] == r["fingerprint"] and x["group"] != "human"))
            rows.append(row)
        who = sorted(annotators)
        out["human"] = pd.DataFrame(rows).set_index("set").reindex(
            columns=["n_dims", "judge", *who, *(f"n_{w}" for w in who), "same trials as"])
        _show("Human trial sets: the judge on the annotated trials next to each annotator's raw accuracy "
              "(n_<annotator> = trials scored)", out["human"])

        # the ablation's row for this judge and prompt, on the trials annotated so far
        from judge_ablation import load_trial_sets, row_for
        try:
            with redirect_stdout(io.StringIO()):
                trial_sets = load_trial_sets(SESSIONS_DIR, "*.jsonl")
        except SystemExit:
            trial_sets = {}
        if trial_sets:
            rec = {"model": man["judge"], "prompt": man["prompt"],
                   "trial_sets": {r["fingerprint"]: {"picks": {t["dim"]: t["pick"] for t in r["tasks"]},
                                                     "n_nan_scores": r["n_nan_scores"]} for r in human}}
            out["judge_vs_humans"] = row_for(rec, trial_sets)
            print("\nJudge as one more annotator (leave-one-out, as the ablation in human_evaluation.ipynb):")
            for key, v in out["judge_vs_humans"].items():
                print(f"  {key:28s} {v:.3f}" if isinstance(v, float) else f"  {key:28s} {v}")

    if write_csv:
        path = OUT_DIR / f"summary_{man['stamp'].replace(':', '')}.csv"
        summary.to_csv(path, index=False)
        print(f"\nwrote {path}")
    return out


if __name__ == "__main__":
    sys.exit(main())
