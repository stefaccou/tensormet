"""Human counterpart of the LLM-as-judge "tea leaves" task (see tensormet.judge).

Same trials, same intruder draw, same score shape as ``DimConsistencyJudge.score``:
for every latent dimension, its top-k words plus one intruder that is weak in this
dimension but salient in some other one, shuffled -- only the judge is a person.
Human and model accuracy are therefore comparable dimension by dimension (see
``compare_with_judge``).

Two properties the earlier notebook prototype did not have:

* every trial is built up front -- one argsort of the whole (N, R) factor for the
  entire run, as judge.py does -- so the wait between dimensions is a widget
  redraw, not a factor lookup. The annotator never waits on the model.
* every answer is appended to its session file the moment it is given, so a run
  can be interrupted and resumed, split over several sittings, or handed to a
  second annotator (``tasks_path=``) without loading the decomposition again.

Annotate in a notebook::

    from tensormet.experimental.tealeaves_human import open_session, run_widget
    s = open_session(tk_pfu, annotator="stef")
    run_widget(s)

or from a terminal, which needs no kernel and no GPU once the session exists::

    python -m tensormet.experimental.tealeaves_human annotate --session <path>

The task construction below mirrors judge.py's "teaLeaves" branch call for call
(same RNG, same seed, same shuffle order), so a session here poses the identical
trials the judge model was scored on. judge.py is deliberately left untouched;
when changing either, change both.
"""
from __future__ import annotations

import hashlib
import json
import os
import random
import statistics
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Optional

import numpy as np

# Relative to the working directory, not inside the installed package.
DEFAULT_RESULTS_DIR = Path("human_eval_results")


# --- Trial construction -----------------------------------------------------
def build_tealeaves_tasks(decomp,
                          role: Optional[str] = None,
                          num_dim_words: int = 5,
                          seed: int = 1,
                          verbose: bool = False) -> list[dict]:
    """One outlier task per latent dimension of `role` (default: the first role).

    Each task is ``{"dim", "words", "random_word", "candidates"}``: the
    dimension's top `num_dim_words` words, the injected intruder, and the two
    shuffled together (what the annotator is shown).

    The intruder for dimension i is drawn from the words in i's bottom half that
    are also in the top 10% of *some* dimension: salient elsewhere, weak here, so
    it cannot be spotted just by being a rare or odd token.

    Draws come from a local RNG seeded with `seed`, so the global random state is
    untouched and the trials are reproducible -- and identical to those
    ``DimConsistencyJudge.score(intruder_choice_option="teaLeaves", seed=seed)``
    poses, as long as `num_dim_words` matches.
    """
    # Deferred: tensormet.utils needs torch and $DATA, which annotate/score must not.
    from tensormet.utils import to_np

    rng = random.Random(seed)

    role = role if role is not None else decomp.roles[0]
    role_idx = decomp.get_role_index(role)
    # Factor columns, not core.shape: identical for Tucker but O(1) for CP,
    # whose `core` is a materialized diag(lambda).
    rank = int(decomp.factors[role_idx].shape[1])
    vocab_list = decomp.vocab[f"vocab_{role}"]
    n_words = decomp.get_dims()[role_idx]   # vocabulary size of this mode
    n_bottom = n_words // 2                 # first index of the bottom half
    n_top = max(1, n_words // 10)

    # One descending argsort of the whole (N, R) factor, against `rank` separate
    # topk calls that each convert the factor to numpy again.
    factor = to_np(decomp.factors[role_idx])[:, :rank]       # (N, R)
    order = np.argsort(-factor, axis=0, kind="stable")       # (N, R) vocab ids
    top_dim_words = {i: [vocab_list[j] for j in order[:num_dim_words, i]]
                     for i in range(rank)}
    bottom_50_percents = {i: {vocab_list[j] for j in order[n_bottom:, i]}
                          for i in range(rank)}
    top_10_percent_of_some_dimension_words = {vocab_list[j]
                                              for j in order[:n_top, :].ravel()}

    if verbose:
        print(f"{len(top_10_percent_of_some_dimension_words)} candidate "
              f"intruder words to choose from")

    tasks = []
    for i in range(rank):
        words = top_dim_words[i]
        # Intersection, not union: salient elsewhere, weak here. sorted() because
        # set order over strings varies with PYTHONHASHSEED, which would break
        # seed reproducibility across processes.
        pool = sorted(bottom_50_percents[i] & top_10_percent_of_some_dimension_words)
        if not pool:
            raise ValueError(
                f"No teaLeaves intruder available for dimension {i} of role "
                f"{role!r}: no word is both in this dimension's bottom half and "
                f"in the top 10% of another dimension."
            )
        random_word = rng.choice(pool)

        candidates = words + [random_word]
        rng.shuffle(candidates)
        tasks.append({"dim": i, "words": words, "candidates": candidates,
                      "random_word": random_word})
    return tasks


def tasks_fingerprint(tasks: list[dict]) -> str:
    """Identity of a trial set: what was shown, in the order it was shown."""
    payload = json.dumps([[t["dim"], t["candidates"]] for t in tasks],
                         sort_keys=True).encode("utf-8")
    return hashlib.sha1(payload).hexdigest()[:12]


def model_label(decomp, fallback: str = "model") -> str:
    """Filename-safe label for a loaded decomposition (its checkpoint stem)."""
    path = getattr(decomp, "decomp_path", None)
    if not path:
        return fallback
    return Path(path).stem


# --- Session (resumable, append-only) ---------------------------------------
@dataclass
class TeaLeavesSession:
    """One annotator's pass over one trial set, backed by an append-only JSONL.

    Line 1 is a meta record carrying the trials themselves, so the file is
    self-contained: resuming, scoring, or handing it to a second annotator never
    needs the decomposition back in memory. Every later line is one answer.
    Replay is last-write-wins per dimension, with ``undo`` records deleting, so a
    correction never rewrites history.
    """
    path: Path
    meta: dict
    tasks: list[dict]
    responses: dict[int, dict] = field(default_factory=dict)

    def __repr__(self) -> str:
        # The default dataclass repr would dump every trial into the notebook.
        return (f"<TeaLeavesSession {self.meta['model']}/{self.meta['role']} "
                f"{self.meta['annotator']}: {self.n_done}/{self.n_tasks} answered "
                f"-> {self.path.name}>")

    # -- persistence
    def _append(self, record: dict) -> None:
        with open(self.path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(record) + "\n")
            fh.flush()
            os.fsync(fh.fileno())

    @property
    def by_dim(self) -> dict[int, dict]:
        return {t["dim"]: t for t in self.tasks}

    # -- progress
    @property
    def n_tasks(self) -> int:
        return len(self.tasks)

    @property
    def n_done(self) -> int:
        return len(self.responses)

    @property
    def n_skipped(self) -> int:
        return sum(1 for r in self.responses.values() if r.get("pick") is None)

    @property
    def remaining(self) -> list[dict]:
        return [t for t in self.tasks if t["dim"] not in self.responses]

    def next_task(self) -> Optional[dict]:
        rem = self.remaining
        return rem[0] if rem else None

    # -- answering
    def record(self, dim: int, pick: Optional[str], seconds: Optional[float] = None) -> bool:
        """Store one answer (``pick=None`` marks it skipped). Returns correctness."""
        task = self.by_dim[dim]
        if pick is not None and pick not in task["candidates"]:
            raise ValueError(f"{pick!r} is not a candidate of dimension {dim}")
        correct = pick == task["random_word"]
        record = {"type": "response", "dim": dim, "pick": pick, "correct": correct,
                  "seconds": None if seconds is None else round(float(seconds), 3),
                  "ts": datetime.now(UTC).isoformat(timespec="seconds")}
        self.responses[dim] = record
        self._append(record)
        return correct

    def undo(self) -> Optional[int]:
        """Drop the most recent answer so it is posed again. Returns its dim."""
        if not self.responses:
            return None
        dim = list(self.responses)[-1]
        del self.responses[dim]
        self._append({"type": "undo", "dim": dim,
                      "ts": datetime.now(UTC).isoformat(timespec="seconds")})
        return dim

    # -- scoring
    def summary(self) -> dict:
        """Accuracy in the shape ``DimConsistencyJudge.score`` returns it.

        A skipped trial ("not sure") counts as wrong: the judge model always
        answers, so letting an annotator drop the dimensions they found unreadable
        would score the model on its legible half only. `n_scored` is therefore
        every trial responded to and `n_answered` only those with a pick; trials
        not yet reached are in neither. The diversity multiplier is a property of
        the decomposition rather than of the answers, so it is computed over all
        trials and equals the judge's value exactly.
        """
        n = len(self.responses)
        n_picked = sum(1 for r in self.responses.values() if r.get("pick") is not None)
        correct = sum(r["correct"] for r in self.responses.values())
        raw = correct / n if n else 0.0

        all_dim_words = set()
        for t in self.tasks:
            all_dim_words.update(t["words"])
        k = self.meta.get("num_dim_words", len(self.tasks[0]["words"]))
        mult = len(all_dim_words) / (self.n_tasks * k)

        times = [r["seconds"] for r in self.responses.values() if r.get("seconds")]
        return {
            "human_dim_consistency": raw * mult,
            "human_dim_consistency_raw": raw,
            "human_dim_consistency_diversity": mult,
            "n_correct": correct,
            "n_scored": n,
            "n_answered": n_picked,
            "n_skipped": self.n_skipped,
            "n_tasks": self.n_tasks,
            "median_seconds": round(statistics.median(times), 2) if times else None,
        }

    def details(self) -> list[dict]:
        """Per-dimension rows: trial, answer, verdict. Mirrors judge's `details`."""
        rows = []
        for t in self.tasks:
            r = self.responses.get(t["dim"])
            rows.append({"dim": t["dim"], "words": t["words"],
                         "outlier": t["random_word"],
                         "pick": None if r is None else r["pick"],
                         "correct": None if r is None else r["correct"],
                         "seconds": None if r is None else r.get("seconds")})
        return rows


def _read_session_file(path: Path) -> tuple[dict, list[dict], dict[int, dict]]:
    meta, tasks, responses = None, None, {}
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if rec.get("type") == "meta":
                meta, tasks = rec, rec["tasks"]
            elif rec.get("type") == "response":
                responses[rec["dim"]] = rec
            elif rec.get("type") == "undo":
                responses.pop(rec["dim"], None)
    if meta is None or tasks is None:
        raise ValueError(f"{path} has no meta line; it is not a tea-leaves session file.")
    return meta, tasks, responses


def open_session(decomp=None,
                 annotator: str = "anon",
                 role: Optional[str] = None,
                 num_dim_words: int = 5,
                 seed: int = 1,
                 name: Optional[str] = None,
                 path: Optional[Path] = None,
                 tasks_path: Optional[Path] = None,
                 results_dir: Optional[Path] = None,
                 verbose: bool = True) -> TeaLeavesSession:
    """Resume this annotator's session, or start it.

    `decomp` is only needed the first time a trial set is built: pass
    `tasks_path` (another session file) instead to give a second annotator the
    same trials, or nothing at all to reopen an existing session by `path`.

    The default filename encodes everything that changes the trials
    (model / role / k / seed) plus the annotator, so two annotators never share a
    file and a changed setting never resumes into the wrong one. When the file
    exists and `decomp` is given, the stored trials are checked against freshly
    built ones, and a mismatch is an error rather than a silently mixed session.
    """
    role_name = role if role is not None else (decomp.roles[0] if decomp is not None else None)

    if path is None:
        label = name if name is not None else (model_label(decomp) if decomp is not None else None)
        if label is None or role_name is None:
            raise ValueError("Pass `path=` explicitly, or a `decomp` to derive the filename from.")
        results_dir = Path(results_dir) if results_dir is not None else DEFAULT_RESULTS_DIR
        results_dir.mkdir(parents=True, exist_ok=True)
        path = results_dir / f"{label}_{role_name}_k{num_dim_words}_seed{seed}_{annotator}.jsonl"
    path = Path(path)

    if path.exists():
        meta, tasks, responses = _read_session_file(path)
        if decomp is not None:
            fresh = build_tealeaves_tasks(decomp, role=meta["role"],
                                          num_dim_words=meta["num_dim_words"],
                                          seed=meta["seed"])
            if tasks_fingerprint(fresh) != meta["fingerprint"]:
                raise ValueError(
                    f"{path} was built from different trials (stored fingerprint "
                    f"{meta['fingerprint']}, rebuilt {tasks_fingerprint(fresh)}). "
                    "The decomposition or the settings changed; start a new session "
                    "under a different `name=`/`annotator=` rather than mixing them."
                )
        session = TeaLeavesSession(path=path, meta=meta, tasks=tasks, responses=responses)
        if verbose:
            print(f"Resuming {path.name}: {session.n_done}/{session.n_tasks} answered.")
        return session

    if tasks_path is not None:
        src_meta, tasks, _ = _read_session_file(Path(tasks_path))
        role_name = src_meta["role"]
        num_dim_words, seed = src_meta["num_dim_words"], src_meta["seed"]
        label = src_meta.get("model")
    elif decomp is not None:
        tasks = build_tealeaves_tasks(decomp, role=role_name,
                                      num_dim_words=num_dim_words, seed=seed,
                                      verbose=verbose)
        label = name if name is not None else model_label(decomp)
    else:
        raise ValueError("Nothing to build from: pass a `decomp`, a `tasks_path`, "
                         "or the `path` of an existing session.")

    meta = {"type": "meta", "task": "tealeaves", "model": label, "role": role_name,
            "num_dim_words": num_dim_words, "seed": seed, "annotator": annotator,
            "fingerprint": tasks_fingerprint(tasks),
            "created": datetime.now(UTC).isoformat(timespec="seconds"),
            "tasks": tasks}
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(json.dumps(meta) + "\n")
    if verbose:
        print(f"New session {path} with {len(tasks)} dimensions.")
    return TeaLeavesSession(path=path, meta=meta, tasks=tasks, responses={})


# --- Annotation UIs ---------------------------------------------------------
PROMPT = "Click the ONE word that does not belong with the others."


def _summary_line(session: TeaLeavesSession) -> str:
    s = session.summary()
    return (f"{s['n_correct']}/{s['n_scored']} correct "
            f"(raw {s['human_dim_consistency_raw']:.3f}, "
            f"diversity-weighted {s['human_dim_consistency']:.3f})"
            + (f", {s['n_skipped']} skipped and counted wrong" if s["n_skipped"] else ""))


def run_widget(session: TeaLeavesSession, feedback: bool = False, width: str = "150px") -> None:
    """ipywidgets annotation UI: one click per dimension, nothing to wait for.

    The buttons are created once and relabelled per trial -- no widget is rebuilt
    and no output cleared between dimensions, which is what keeps the gap between
    trials at a DOM update. Every click is on disk before the next trial is drawn,
    so closing the tab mid-session loses nothing.

    `feedback` off (the default) hides right/wrong until the end: told each time
    which word was the intruder, an annotator learns the draw's statistical
    signature and later trials stop measuring the model.
    """
    import ipywidgets as widgets
    from IPython.display import display

    n_cand = len(session.tasks[0]["candidates"])
    header = widgets.HTML()
    status = widgets.HTML()
    buttons = [widgets.Button(layout=widgets.Layout(width=width)) for _ in range(n_cand)]
    skip_btn = widgets.Button(description="not sure", button_style="warning",
                              layout=widgets.Layout(width="110px"))
    undo_btn = widgets.Button(description="undo", layout=widgets.Layout(width="110px"))
    state = {"task": None, "t0": None}

    def render():
        task = session.next_task()
        state["task"] = task
        header.value = (f"<b>{session.n_done}/{session.n_tasks}</b> &nbsp; "
                        f"{session.meta['model']} &middot; {session.meta['role']}<br>"
                        f"<span style='color:#888'>{PROMPT}</span>")
        if task is None:
            for b in buttons:
                b.description = ""
                b.disabled = True
            skip_btn.disabled = True
            status.value = f"<b>Done.</b> {_summary_line(session)}"
            return
        for b, word in zip(buttons, task["candidates"]):
            b.description = word
            b.disabled = False
        skip_btn.disabled = False
        state["t0"] = time.perf_counter()

    def answer(pick: Optional[str]):
        task = state["task"]
        if task is None:
            return
        elapsed = None if state["t0"] is None else time.perf_counter() - state["t0"]
        correct = session.record(task["dim"], pick, seconds=elapsed)
        if feedback and pick is not None:
            status.value = ("<span style='color:green'>correct</span>" if correct else
                            f"<span style='color:#c00'>no &mdash; it was "
                            f"<b>{task['random_word']}</b></span>")
        render()

    def answer_index(i: int):
        task = state["task"]
        if task is not None:
            answer(task["candidates"][i])

    for idx, btn in enumerate(buttons):
        btn.on_click(lambda _b, i=idx: answer_index(i))
    skip_btn.on_click(lambda _b: answer(None))

    def on_undo(_b):
        session.undo()
        for b in buttons:
            b.disabled = False
        skip_btn.disabled = False
        status.value = ""
        render()

    undo_btn.on_click(on_undo)

    display(widgets.VBox([header, widgets.HBox(buttons),
                          widgets.HBox([skip_btn, undo_btn]), status]))
    render()


def run_terminal(session: TeaLeavesSession, feedback: bool = False) -> TeaLeavesSession:
    """Keyboard annotation loop: type the number of the odd word out.

    Needs no kernel, no widgets and no GPU -- only the session file -- so it runs
    over a plain ssh connection. `s` skips, `u` undoes, `q` saves and stops.
    """
    print(f"{session.meta['model']} / {session.meta['role']} -- {PROMPT}")
    print("keys: 1-N pick, s skip, u undo, q quit (saved after every answer)\n")
    while True:
        task = session.next_task()
        if task is None:
            print(f"\nAll {session.n_tasks} dimensions done. {_summary_line(session)}")
            return session
        options = "   ".join(f"[{i + 1}] {w}" for i, w in enumerate(task["candidates"]))
        print(f"({session.n_done}/{session.n_tasks})  {options}")
        t0 = time.perf_counter()
        try:
            raw = input("> ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            raw = "q"
        if raw == "q":
            print(f"\nStopped at {session.n_done}/{session.n_tasks}; "
                  f"resume with --session {session.path}")
            return session
        if raw == "u":
            dim = session.undo()
            print(f"undone (dimension {dim})\n" if dim is not None else "nothing to undo\n")
            continue
        if raw == "s":
            session.record(task["dim"], None, seconds=time.perf_counter() - t0)
            print()
            continue
        if not raw.isdigit() or not (1 <= int(raw) <= len(task["candidates"])):
            print(f"enter 1-{len(task['candidates'])}, or s/u/q\n")
            continue
        pick = task["candidates"][int(raw) - 1]
        correct = session.record(task["dim"], pick, seconds=time.perf_counter() - t0)
        if feedback:
            print("correct\n" if correct else f"no -- it was {task['random_word']}\n")
        else:
            print()


# --- Comparison with the model judge ----------------------------------------
def compare_with_judge(session: TeaLeavesSession, judge_out: dict) -> dict:
    """Line up a session with ``DimConsistencyJudge.score(..., return_details=True)``.

    Both must come from the same role, seed and `num_dim_words`, which is what
    makes the trials identical; that is checked against the judge's own
    per-dimension trials rather than trusted.

    Returns the two accuracies over the dimensions the human answered, how often
    they picked the same word, and the per-dimension rows behind that.

    Skipped trials are excluded here, unlike in ``summary()``, where they count as
    wrong: agreement needs two picks to compare. `human_accuracy` is therefore over
    the picked dimensions only and will read higher than the session's score --
    `n_skipped_excluded` says over how many.
    """
    if "details" not in judge_out:
        raise ValueError("Call the judge with return_details=True to compare per dimension.")
    judge_by_dim = {d["dim"]: d for d in judge_out["details"]}
    tasks = session.by_dim

    rows, agree, human_ok, judge_ok, skipped = [], 0, 0, 0, 0
    for dim, resp in sorted(session.responses.items()):
        if resp.get("pick") is None:
            skipped += dim in judge_by_dim
            continue
        if dim not in judge_by_dim:
            continue
        jd = judge_by_dim[dim]
        if jd["outlier"] != tasks[dim]["random_word"]:
            raise ValueError(
                f"Dimension {dim} differs between the two runs (human intruder "
                f"{tasks[dim]['random_word']!r}, judge {jd['outlier']!r}): they were "
                "not built with the same role/seed/num_dim_words."
            )
        same = resp["pick"] == jd["predicted"]
        agree += same
        human_ok += resp["correct"]
        judge_ok += jd["correct"]
        rows.append({"dim": dim, "words": tasks[dim]["words"],
                     "outlier": tasks[dim]["random_word"],
                     "human": resp["pick"], "judge": jd["predicted"],
                     "human_correct": resp["correct"], "judge_correct": jd["correct"],
                     "agree": same})

    n = len(rows)
    if not n:
        raise ValueError("No dimension was answered by both the human and the judge.")
    return {"n_compared": n,
            "n_skipped_excluded": skipped,
            "human_accuracy": human_ok / n,
            "judge_accuracy": judge_ok / n,
            "agreement": agree / n,
            "rows": rows}


# --- CLI --------------------------------------------------------------------
# `build` needs the decomposition (and hence the data dir); `annotate` and
# `score` only need the session file, so annotation can happen anywhere.
def _cli():
    import argparse

    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = p.add_subparsers(dest="cmd", required=True)

    b = sub.add_parser("build", help="load a decomposition and write a session file")
    b.add_argument("--annotator", default="anon")
    b.add_argument("--role", default=None, help="default: the model's first role")
    b.add_argument("--num-dim-words", type=int, default=5)
    b.add_argument("--seed", type=int, default=1)
    b.add_argument("--name", default=None, help="label used in the filename")
    b.add_argument("--results-dir", default=None)
    b.add_argument("--annotate", action="store_true", help="start annotating right away")
    # Decomposition selection: the InspectionConfig fields, same meaning as there.
    b.add_argument("--dataset", default="4-gram-raw-bos-eos-fineweb-en_1B")
    b.add_argument("--model-name", default="h100_1B_mu_postFactorUpdate",
                   help="the run's `name` (InspectionConfig.name)")
    b.add_argument("--method", default="scSoftPlus")
    b.add_argument("--divergence", default="kl")
    b.add_argument("--dim", default="10000")
    b.add_argument("--rank", type=int, default=100)
    b.add_argument("--order", type=int, default=None,
                   help="default: inferred from --dataset / --model-name")
    b.add_argument("--iters", type=int, default=1000)
    b.add_argument("--shared-factors", default="all")
    b.add_argument("--subsample-frac", type=float, default=0.025)
    b.add_argument("--solver", default="mu")
    b.add_argument("--decomposition", default="tucker", choices=["tucker", "cp", "tt"])
    b.add_argument("--tt-rank", type=int, default=None)

    a = sub.add_parser("annotate", help="annotate (or resume) an existing session file")
    a.add_argument("--session", required=True)
    a.add_argument("--annotator", default=None,
                   help="annotate the same trials as a different person "
                        "(writes a new file next to --session)")
    a.add_argument("--feedback", action="store_true", help="reveal right/wrong per trial")

    s = sub.add_parser("score", help="print the score of a session file")
    s.add_argument("--session", required=True)
    s.add_argument("--details", action="store_true")

    args = p.parse_args()

    if args.cmd == "build":
        from tensormet.config import InspectionConfig
        dim = tuple(int(x) for x in args.dim.split("-")) if "-" in args.dim else int(args.dim)
        cfg = InspectionConfig(dim=dim, name=args.model_name, dataset=args.dataset,
                               method=args.method, divergence=args.divergence,
                               order=args.order, iters=args.iters, rank=args.rank,
                               shared_factors=args.shared_factors,
                               subsample_frac=args.subsample_frac, solver=args.solver,
                               decomposition=args.decomposition, tt_rank=args.tt_rank)
        session = open_session(cfg.load_tucker(), annotator=args.annotator, role=args.role,
                               num_dim_words=args.num_dim_words, seed=args.seed,
                               name=args.name, results_dir=args.results_dir)
        print(f"session: {session.path}")
        if args.annotate:
            run_terminal(session)
        return

    if args.cmd == "annotate":
        src = Path(args.session)
        if args.annotator:
            meta, _, _ = _read_session_file(src)
            target = src.with_name(src.name.replace(f"_{meta['annotator']}.jsonl",
                                                    f"_{args.annotator}.jsonl"))
            if target == src:
                target = src.with_name(f"{src.stem}_{args.annotator}.jsonl")
            session = open_session(path=target, tasks_path=src, annotator=args.annotator)
        else:
            session = open_session(path=src)
        run_terminal(session, feedback=args.feedback)
        return

    session = open_session(path=Path(args.session), verbose=False)
    print(json.dumps(session.summary(), indent=2))
    if args.details:
        for row in session.details():
            mark = {True: "ok ", False: "MISS", None: "-  "}[row["correct"]]
            print(f"{mark} dim {row['dim']:>3}  {row['words']} + {row['outlier']!r} "
                  f"-> {row['pick']!r}")


if __name__ == "__main__":
    _cli()
