"""Judge ablation: does the LLM judge behave like one more tea-leaves annotator?

For every (judge base model, prompt) pair, the judge is scored exactly like a
human: leave-one-out, each rater's mean agreement with every *other* human.
Humans disagree among themselves too, so the human value is the ceiling and
`<metric>_ratio` = judge / human says how close the judge gets to it (1.0 = as
good as an extra annotator). Two agreement measures, over the trials both raters
answered (skips excluded), each dimension counted once per pair:

  phi             correlation of the two raters' right/wrong (intruder found)
  pick_agreement  fraction of trials where both picked the same word

Script version of the ablation cell in human_evaluation.ipynb.
Session files carry their own trials, so only those are needed -- no
decompositions, no $DATA. From 5_evaluation/COLING:

    python judge_ablation.py \
        --sessions-dir human_eval_results \
        --models Qwen/Qwen3.5-9B Qwen/Qwen3.5-27B

Writes one JSON per (model, prompt) to --out-dir holding the judge's picks per
trial set. The judge's answers do not depend on who annotated, so a new
annotator on existing trials needs no GPU: drop the session file in and run with
--summary-only. Only an unseen trial set or an edited prompt is judged again.
Every run ends by printing (and saving as summary.csv) the table over
everything in --out-dir, scored against the sessions as they are now.
"""
from __future__ import annotations

import argparse
import itertools
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from tensormet.experimental.tealeaves_human import open_session
from tensormet.judge import DEFAULT_JUDGE_MESSAGES, DEFAULT_JUDGE_MODEL, DimConsistencyJudge

# "{listing}" is filled with the comma-separated candidates.
FEW_SHOT = [
    {"role": "system", "content": "You are a helpful assistant tasked with identifying the outlier in a list of words."},
    {"role": "user", "content": "Which word does not belong? apple, banana, orange, car, grape."},
    {"role": "assistant", "content": "car"},
    {"role": "user", "content": "Which word does not belong? like, love, message, adore, hate."},
    {"role": "assistant", "content": "message"},
    {"role": "user", "content": "Which word does not belong? {listing}. Answer with only the word."},
]
# Stammbach et al. (2023), topic model evaluation.
STAMMBACH = [
    {"role": "system",
     "content": "You are a helpful assistant evaluating the top words of a topic model output "
                "for a given topic. Select which word is the least related to all other words. "
                "If multiple words do not fit, choose the word that is most out of place."},
    {"role": "user", "content": "{listing}"},
]

PROMPTS = {
    "default": DEFAULT_JUDGE_MESSAGES,
    "few-shot": FEW_SHOT,
    "stammbach": STAMMBACH,
}

JUDGE = "judge"  # rater name of the LLM next to the annotators
METRICS = ("phi", "pick_agreement")


# --- Sessions -----------------------------------------------------------------
def load_trial_sets(sessions_dir: Path, pattern: str) -> dict:
    """Answered sessions grouped by trial set (= fingerprint):

        {fingerprint: {"label", "tasks", "outlier": {dim: word},
                       "answers": {annotator: {dim: pick or None}}}}

    Annotators of the same decomposition share its trial set, so any number of
    them line up per dimension without extra work.
    """
    trial_sets = {}
    for path in sorted(Path(sessions_dir).glob(pattern)):
        s = open_session(path=path, verbose=False)
        if not s.n_done:
            continue
        ts = trial_sets.setdefault(s.meta["fingerprint"], {
            "label": s.meta["model"], "tasks": s.tasks,
            "outlier": {t["dim"]: t["random_word"] for t in s.tasks}, "answers": {}})
        who = s.meta["annotator"]
        if who in ts["answers"]:
            raise SystemExit(f"{path.name}: a second session of {who!r} on the same trials.")
        ts["answers"][who] = {d: r.get("pick") for d, r in s.responses.items()}
        print(f"  {path.name}: {s.n_done}/{s.n_tasks} answered")
    if not trial_sets:
        raise SystemExit(f"No answered session matching {pattern!r} in {sessions_dir}.")
    return trial_sets


# --- Agreement ----------------------------------------------------------------
def _mean(xs) -> float:
    xs = [x for x in xs if not math.isnan(x)]
    return sum(xs) / len(xs) if xs else math.nan


def _phi(a: list[bool], b: list[bool]) -> float:
    """Pearson on 0/1 data; nan when either rater is constant (e.g. all right)."""
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    if len(a) < 2 or a.std() == 0 or b.std() == 0:
        return math.nan
    return float(np.corrcoef(a, b)[0, 1])


def agreement(raters: dict, trial_sets: dict, x: str, y: str) -> dict:
    """Agreement of raters `x` and `y`, pooled over every trial set both did.

    `raters` is {fingerprint: {rater: {dim: pick}}}; trials either skipped are left out.
    """
    cx, cy, same = [], [], []
    for fp, by_rater in raters.items():
        if x not in by_rater or y not in by_rater:
            continue
        outlier = trial_sets[fp]["outlier"]
        py_all = by_rater[y]
        for dim, px in by_rater[x].items():
            py = py_all.get(dim)
            if px is None or py is None:
                continue
            cx.append(px == outlier[dim])
            cy.append(py == outlier[dim])
            same.append(px == py)
    return {"phi": _phi(cx, cy),
            "pick_agreement": sum(same) / len(same) if same else math.nan,
            "n": len(same)}


def leave_one_out(raters: dict, trial_sets: dict, humans: list[str]) -> dict:
    """Each rater's mean agreement with every *other* human.

    The judge goes through the same formula as the annotators, so its value is
    directly comparable to theirs; the humans' mean is the achievable ceiling.
    """
    out = {}
    for r in humans + [JUDGE]:
        pairs = [agreement(raters, trial_sets, r, h) for h in humans if h != r]
        pairs = [p for p in pairs if p["n"]]
        out[r] = {m: _mean(p[m] for p in pairs) for m in METRICS}
    return out


def print_human_agreement(trial_sets: dict) -> None:
    raters = {fp: ts["answers"] for fp, ts in trial_sets.items()}
    humans = sorted({h for ts in trial_sets.values() for h in ts["answers"]})
    for x, y in itertools.combinations(humans, 2):
        a = agreement(raters, trial_sets, x, y)
        if a["n"]:
            print(f"  {x} vs {y}: phi={a['phi']:.3f}  "
                  f"pick_agreement={a['pick_agreement']:.3f}  n={a['n']}")
    # The spread the judge's leave-one-out value should fall within.
    for h, v in leave_one_out(raters, trial_sets, humans).items():
        if h != JUDGE:
            print(f"  {h} vs other humans: phi={v['phi']:.3f}  "
                  f"pick_agreement={v['pick_agreement']:.3f}")


# --- Judge --------------------------------------------------------------------
def judge_trial_set(judge: DimConsistencyJudge, tasks: list[dict]) -> dict:
    """The judge's pick (and per-candidate log-probs) for every trial of a set."""
    judge.ensure_loaded()  # _evaluate_tasks, unlike score(), does not load the model
    tasks = [dict(t) for t in tasks]  # _evaluate_tasks writes into them
    judge._evaluate_tasks(tasks, answer_key="random_word")
    return {
        "picks": {t["dim"]: t["predicted"] for t in tasks},
        "scores": {t["dim"]: {k: round(v, 4) for k, v in t["scores"].items()} for t in tasks},
        "n_nan_scores": sum(any(math.isnan(v) for v in t["scores"].values()) for t in tasks),
    }


def result_path(out_dir: Path, model_name: str, prompt_name: str) -> Path:
    return Path(out_dir) / f"{model_name.replace('/', '__')}__{prompt_name}.json"


def judge_model(model_name: str, prompt_names: list[str], trial_sets: dict, out_dir: Path,
                overwrite: bool = False, require_gpu: bool = False, **judge_kwargs) -> dict:
    """Judge every trial set with `model_name` under each prompt: one JSON per prompt in `out_dir`.

    Cached files only get the trial sets they lack. The model is loaded once, and only
    if some prompt still needs it; `require_gpu` raises instead of falling back to CPU.
    """
    info = {"judged": [], "cached": [], "n_nan": 0, "device": None, "gpu_memory_gb": None}
    judge = None
    try:
        for prompt_name in prompt_names:
            messages = PROMPTS[prompt_name]
            path = result_path(out_dir, model_name, prompt_name)
            rec = None if overwrite else load_cache(path)
            if rec is None or rec["messages"] != messages:
                rec = {"model": model_name, "prompt": prompt_name,
                       "messages": messages, "trial_sets": {}}
            missing = [fp for fp in trial_sets if fp not in rec["trial_sets"]]
            if not missing:
                print(f"cached: {model_name} | {prompt_name}")
                info["cached"].append(prompt_name)
                continue
            if judge is None:
                judge = DimConsistencyJudge(model_name, **judge_kwargs)
                judge.ensure_loaded()
                info["device"], info["gpu_memory_gb"] = str(judge.device), judge.gpu_memory_gb
                if require_gpu and judge.device.type != "cuda":
                    raise RuntimeError(f"no GPU room for {model_name}: the judge fell back to CPU")
            judge.messages = messages
            n_nan = 0
            for fp in missing:
                result = judge_trial_set(judge, trial_sets[fp]["tasks"])
                rec["trial_sets"][fp] = {"label": trial_sets[fp]["label"], **result}
                n_nan += result["n_nan_scores"]
            print(f"judged: {model_name} | {prompt_name} ({len(missing)} new trial set(s))")
            if n_nan:
                print(f"  WARNING: {n_nan} trials got nan log-probs "
                      "(fp16 overflow?); this row is not trustworthy.")
            path.write_text(json.dumps(rec, indent=1), encoding="utf-8")
            info["judged"].append(prompt_name)
            info["n_nan"] += n_nan
    finally:
        if judge is not None:
            judge.unload()
    return info


def load_cache(path: Path) -> dict | None:
    """A (model, prompt) result file, or None if absent or in an older format."""
    if not path.exists():
        return None
    rec = json.loads(path.read_text(encoding="utf-8"))
    if "trial_sets" not in rec:
        return None
    for ts in rec["trial_sets"].values():
        ts["picks"] = {int(d): p for d, p in ts["picks"].items()}  # JSON keys are str
    return rec


def row_for(rec: dict, trial_sets: dict) -> dict:
    """One summary row: the judge of `rec` scored as one more annotator."""
    judged = {fp: ts for fp, ts in trial_sets.items() if fp in rec["trial_sets"]}
    raters = {fp: {**ts["answers"], JUDGE: rec["trial_sets"][fp]["picks"]}
              for fp, ts in judged.items()}
    humans = sorted({h for ts in judged.values() for h in ts["answers"]})
    loo = leave_one_out(raters, judged, humans)

    row = {"model": rec["model"], "prompt": rec["prompt"]}
    for m in METRICS:
        judge_v = loo[JUDGE][m]
        human_v = _mean(loo[h][m] for h in humans)
        row[f"judge_{m}"] = judge_v
        row[f"human_{m}"] = human_v
        row[f"{m}_ratio"] = judge_v / human_v if human_v else math.nan
    for fp, ts in judged.items():
        picks = rec["trial_sets"][fp]["picks"]
        row[f"judge_acc {ts['label']}"] = _mean(float(picks[d] == w) for d, w in ts["outlier"].items())
    row["n_humans"] = len(humans)
    row["n_nan_scores"] = sum(rec["trial_sets"][fp]["n_nan_scores"] for fp in judged)
    # Answered trial sets this judge has not seen yet: rerun without --summary-only.
    row["n_unjudged_sets"] = len(trial_sets) - len(judged)
    return row


def summarize(out_dir: Path, trial_sets: dict) -> None:
    records = [rec for p in sorted(out_dir.glob("*.json")) if (rec := load_cache(p))]
    if not records:
        print(f"No judge results in {out_dir} yet.")
        return
    df = pd.DataFrame([row_for(rec, trial_sets) for rec in records])
    df = df.set_index(["model", "prompt"]).sort_values("phi_ratio", ascending=False)
    df.to_csv(out_dir / "summary.csv")
    with pd.option_context("display.width", 250, "display.max_columns", None):
        print(df.to_string(float_format=lambda v: f"{v:.3g}"))


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--sessions-dir", type=Path, default=Path("human_eval_results"))
    ap.add_argument("--pattern", default="*.jsonl", help="glob for session files")
    ap.add_argument("--models", nargs="+", default=[DEFAULT_JUDGE_MODEL],
                    help="HF repo ids of the judge base models")
    ap.add_argument("--prompts", nargs="+", default=list(PROMPTS), choices=list(PROMPTS))
    ap.add_argument("--out-dir", type=Path, default=Path("judge_ablation_results"))
    ap.add_argument("--summary-only", action="store_true",
                    help="judge nothing: rescore the cached picks against the current sessions (no GPU)")
    ap.add_argument("--overwrite", action="store_true", help="rejudge cached trial sets")
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    print("Sessions:")
    trial_sets = load_trial_sets(args.sessions_dir, args.pattern)
    print("Human-human agreement:")
    print_human_agreement(trial_sets)

    for model_name in ([] if args.summary_only else args.models):
        judge_model(model_name, args.prompts, trial_sets, args.out_dir, overwrite=args.overwrite)

    summarize(args.out_dir, trial_sets)


if __name__ == "__main__":
    main()
