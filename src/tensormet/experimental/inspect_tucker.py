"""Inspect and compare decomposition runs from their on-disk logs.

Three layers, smallest to largest:

* **loading**  -- :func:`load_metrics` / :func:`load_vocab` parse a run's log
  into (iters, rec_error, sem_dicts), and :func:`load_iter_times` pulls the
  per-iteration decomposition times out of the same log.
  :func:`resume_chain` expands one config
  into the log segments a resumed run is split across (a resumed run's own log
  holds only the tail), and :func:`describe_run` prints what a config resolves
  to when a curve looks wrong.
* **plotting** -- :func:`plot_metrics` (one run) and :func:`compare_metrics`
  (any number of runs overlaid) turn those into matplotlib figures.
* **ranking**  -- :func:`evaluate_runs` scores every discovered run on its
  best-ever value of each metric; :func:`find_best` narrows that by facet and
  metric-threshold criteria and ranks it best-first.
* **UI**       -- :func:`make_run_browser` scans the ``decomposition/`` directory
  of one or more datasets (:func:`discover_datasets` finds them) for
  ``*_config.json`` snapshots and offers dataset checkboxes plus faceted
  drop-downs to pick and compare any two runs — even across datasets —
  interactively (requires ``ipywidgets``). :func:`make_run_ranker` is the same
  faceted picker wired to :func:`find_best`: choose criteria, get the top runs
  on a metric as a table plus an optional overlay plot.

The plotting/comparison functions duck-type on their config argument: they only
touch ``.log_path`` / ``.stem`` / ``.vocab_path``, so both :class:`InspectionConfig`
and the lightweight :class:`RunRef` (reconstructed from a config snapshot, works
for legacy-named runs too) are accepted interchangeably.
"""

from __future__ import annotations

import datetime as _dt
import json
import operator
import pickle
import re
from dataclasses import dataclass, replace
from pathlib import Path
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from tensormet.config import InspectionConfig, infer_ngram_order
from tensormet.utils import DATA_DIR


# === loading ===========================================================

# Iteration line shape:
#   "Iteration 5 Rec_error: 1.23e-01 ... Sem_all: {'average_rank_score': 0.4, ...}"
_METRIC_RE = re.compile(
    r"Iteration\s+(\d+)"
    r"\s+Rec_error:\s+([\d\.eE+-]+)"
    r".*?Sem_all:\s+(\{.*\})"
)


def _values_for_key(key, all_its, all_sem):
    """Pull (iterations, values) for one semantic key, skipping iters that lack it."""
    its = [it for it, d in zip(all_its, all_sem) if key in d]
    vals = [d[key] for d in all_sem if key in d]
    return its, vals


def _read_log(cfg):
    iters, rec, sem = [], [], []
    with open(cfg.log_path, "r") as f:
        for line in f:
            m = _METRIC_RE.search(line)
            if m:
                iters.append(int(m.group(1)))
                rec.append(float(m.group(2)))
                sem.append(json.loads(m.group(3)))
    return iters, rec, sem


# Per-iteration timing line shape (written when eval.time_iteration is on,
# the default):
#   "17: reconstruction error=0.53 (Δ=+1.2e-03), time=4.87"
# The number here is the loop's 0-based ``iteration``, while the metric line for
# that same pass prints ``iteration + 1`` (see tucker_tensor's decomposition
# loop) — _read_times adds the 1 back so both series share one x-axis.
# Unanchored like _METRIC_RE: a tqdm bar's carriage returns leave no newline, so
# a metric line can share a "line" with the bar that preceded it.
_TIME_RE = re.compile(
    r"(?:^|[\s\r])(\d+):\s+reconstruction error=.*?,\s*time=([\d\.eE+-]+)"
)


def _monotonic(iters, rec, sem):
    """Sort a parsed series by iteration and drop duplicate iterations.

    Two things make a raw parse non-monotonic, and a line plot of a
    non-monotonic x doubles back on itself instead of reading left-to-right:

    * ``tee_output`` opens the log in append mode (utils.py), so relaunching an
      identical config writes a second pass into the same file with iteration
      numbers restarting from 1;
    * concatenated resume segments are only ordered if their files were.

    Duplicated iterations keep their **last** occurrence (the most recent pass).
    ``average_runs`` also depends on this: ``np.interp`` silently returns
    nonsense when its ``xp`` is not increasing.
    """
    by_iter = {}
    for it, r, s in zip(iters, rec, sem):
        by_iter[it] = (r, s)
    order = sorted(by_iter)
    return (order,
            [by_iter[i][0] for i in order],
            [by_iter[i][1] for i in order])


def load_metrics(*cfgs):
    """Parse one or more run logs into (iters, rec_error, sem_dicts).

    A single config yields its own three lists; several configs are
    concatenated (useful for resumed runs split across files). The result is
    always sorted by iteration and de-duplicated (see :func:`_monotonic`).
    """
    all_iters, all_rec, all_sem = [], [], []
    for cfg in cfgs:
        iters, rec, sem = _read_log(cfg)
        all_iters.extend(iters)
        all_rec.extend(rec)
        all_sem.extend(sem)
    return _monotonic(all_iters, all_rec, all_sem)


def _read_times(cfg):
    its, secs = [], []
    with open(cfg.log_path, "r") as f:
        for line in f:
            m = _TIME_RE.search(line)
            if m:
                its.append(int(m.group(1)) + 1)
                secs.append(float(m.group(2)))
    return its, secs


def load_iter_times(*cfgs):
    """Parse per-iteration decomposition times (seconds) out of one or more logs.

    Returns ``(iters, seconds)``. Several configs are concatenated (the resume
    segments of one run), then sorted with duplicate iterations keeping their
    **last** occurrence — the same repairs :func:`_monotonic` applies to the
    metric series, and for the same reasons (append-mode relaunches, unordered
    segments).

    These are the device-synced update+error timings the decomposition loop sums
    into ``solve_seconds``; they exclude the in-loop semantic evaluation and
    checkpointing that ``decomp_seconds`` also covers. The result is empty for a
    run launched with ``time_iteration`` off, and skips the first logged
    iteration (the timing line rides along with the Δ report, which needs a
    previous error to exist).
    """
    by_iter = {}
    for cfg in cfgs:
        by_iter.update(zip(*_read_times(cfg)))
    order = sorted(by_iter)
    return order, [by_iter[i] for i in order]


def load_vocab(cfg):
    """Unpickle the vocabulary associated with a run config."""
    with open(cfg.vocab_path, "rb") as f:
        return pickle.load(f)


def average_runs(configs, n_grid=500, stitch=True):
    """Average rec_error and semantic metrics across multiple runs.

    ``configs`` is a list of configs (or a dict whose values are configs); each
    may also be a list/tuple of resume-chain segments as accepted by
    :func:`load_metrics`; a lone config is expanded into its chain unless
    ``stitch`` is off. Runs whose log file does not exist are silently skipped.

    Metrics are interpolated onto a shared integer grid that spans the iteration
    range *common to all runs* (clipped at the shortest run), then averaged
    point-wise. The result is a ``(iters, rec_avg, sem_avg)`` triple — the same
    shape as :func:`load_metrics` output — and can be passed directly as a run
    entry in :func:`compare_metrics`.
    """
    raw_configs = list(configs.values() if isinstance(configs, dict) else configs)
    loaded = []
    for cfg in raw_configs:
        segs = _as_run(cfg)
        if stitch and len(segs) == 1:
            segs = resume_chain(segs[0])
        try:
            loaded.append(load_metrics(*segs))
        except FileNotFoundError:
            continue
    if not loaded:
        raise FileNotFoundError("No log files found for any of the provided configs.")

    all_its = [its for its, _, _ in loaded if its]
    grid_min = max(min(its) for its in all_its)
    grid_max = min(max(its) for its in all_its)
    grid = sorted(set(int(v) for v in np.linspace(grid_min, grid_max, n_grid)))

    rec_avg = np.mean(
        [np.interp(grid, its, rec) for its, rec, _ in loaded], axis=0
    ).tolist()

    all_keys: set[str] = set()
    for _, _, sem in loaded:
        for d in sem:
            all_keys.update(d.keys())

    sem_avg = []
    for g_it in grid:
        d = {}
        for key in all_keys:
            vals = []
            for its, _, sem in loaded:
                its_k, vals_k = _values_for_key(key, its, sem)
                if its_k:
                    vals.append(float(np.interp(g_it, its_k, vals_k)))
            if vals:
                d[key] = sum(vals) / len(vals)
        sem_avg.append(d)

    return grid, rec_avg, sem_avg


# === plotting ==========================================================

# Horizontal step between stacked right-hand y-axes, and where the legend sits
# once one axis is out there. Both are in axes coordinates.
_AXIS_OFFSET = 0.11
_LEGEND_X = 1.12


def _place_legend(ax, lines, loc, n_right):
    """Park the legend clear of the axes.

    ``loc="right"`` (default) keeps it beside the plot, stepped out past any
    stacked right-hand axes — good for a two-run compare. ``loc="below"`` drops
    it under the axes in up-to-3 columns — better when there are many long
    labels, as in the :func:`find_best` ranking overlay.
    """
    labels = [l.get_label() for l in lines]
    if loc == "below":
        ax.legend(lines, labels, loc="upper center", bbox_to_anchor=(0.5, -0.12),
                  frameon=False, ncol=min(3, max(1, len(lines))))
    else:
        ax.legend(lines, labels, loc="center left",
                  bbox_to_anchor=(_LEGEND_X + _AXIS_OFFSET * max(0, n_right - 1), 0.5),
                  frameon=False)


def _smooth(vals, window):
    """Centered rolling mean of ``vals``; a window of ``None``/0/1 returns it as-is.

    ``min_periods=1`` keeps the series length, so a smoothed curve still spans the
    full iteration range — its two ends are just averaged over fewer points.
    """
    if not window or window <= 1 or len(vals) == 0:
        return vals
    return pd.Series(vals).rolling(window, center=True, min_periods=1).mean().tolist()


def _time_axis(ax1, n_right):
    """Twin axis for the iteration-time curve, offset clear of ``n_right`` existing ones."""
    axt = ax1.twinx()
    if n_right:
        axt.spines["right"].set_position(("axes", 1.0 + _AXIS_OFFSET * n_right))
    axt.set_ylabel("Iteration time (s)")
    return axt


def plot_metrics(*cfgs, sem_keys=("average_rank_score",),
                 plot_rec_error=True, plot_iter_time=False, title="", ax=None,
                 stitch=True, smooth=None):
    """Plot reconstruction error, semantic scores and/or iteration time for one run.

    Pass ``ax`` to draw into an existing axis (a twin axis is created
    internally for the score curves). Returns the figure.

    With ``plot_iter_time``, the per-iteration decomposition time from
    :func:`load_iter_times` is drawn in grey on its own right-hand axis (or on
    the primary axis when nothing else is plotted). Seconds share no scale with
    errors or scores, hence the separate axis. Silently skipped when the log
    holds no timing lines.

    With ``stitch`` (the default), a single config is expanded into its resume
    chain via :func:`resume_chain`, so a run that was resumed to a higher
    ``n_iter_max`` plots from iteration 0 rather than from where the resume
    began. Pass ``stitch=False`` to plot exactly the segment(s) given.

    ``smooth`` is an optional rolling-average window (in iterations) applied to
    every curve — rec error, scores and iteration time alike — to damp
    per-iteration jitter. ``None``/1 plots the raw series (see :func:`_smooth`).
    """
    if stitch and len(cfgs) == 1:
        cfgs = resume_chain(cfgs[0])
    all_its, all_rec, all_sem = load_metrics(*cfgs)

    ax1 = ax or plt.subplots()[1]
    fig = ax1.figure
    ax1.set_xlabel("Iteration")
    ax1.grid(True)

    colors = plt.cm.tab10.colors
    all_lines = []

    split_axes = (not plot_rec_error) and (len(sem_keys) == 2)
    n_right = 0  # right-hand axes in use, so the time axis lands beside them

    if split_axes:
        ax2 = ax1.twinx()
        n_right = 1
        its0, vals0 = _values_for_key(sem_keys[0], all_its, all_sem)
        (l0,) = ax1.plot(its0, _smooth(vals0, smooth), label=sem_keys[0], color=colors[0])
        ax1.set_ylabel(sem_keys[0])
        its1, vals1 = _values_for_key(sem_keys[1], all_its, all_sem)
        (l1,) = ax2.plot(its1, _smooth(vals1, smooth), label=sem_keys[1], color=colors[1])
        ax2.set_ylabel(sem_keys[1])
        all_lines = [l0, l1]
    else:
        if plot_rec_error:
            (l,) = ax1.plot(all_its, _smooth(all_rec, smooth), label="Rec error", color="red")
            ax1.set_ylabel("Reconstruction Error")
            all_lines.append(l)
        if sem_keys:
            ax2 = ax1.twinx()
            n_right = 1
            ax2.set_ylabel("Score")
            for i, key in enumerate(sem_keys):
                # Offset so the first score curve isn't solid like rec error.
                ls = _LINESTYLES[(i + (1 if plot_rec_error else 0)) % len(_LINESTYLES)]
                its_k, vals_k = _values_for_key(key, all_its, all_sem)
                (l,) = ax2.plot(its_k, _smooth(vals_k, smooth), label=key,
                                color=colors[i % len(colors)], linestyle=ls)
                all_lines.append(l)

    if plot_iter_time:
        t_its, t_secs = load_iter_times(*cfgs)
        if t_its:
            # With nothing else drawn, the time curve owns the primary axis.
            if n_right or plot_rec_error:
                axt = _time_axis(ax1, n_right)
                n_right += 1
            else:
                axt = ax1
                ax1.set_ylabel("Iteration time (s)")
            (l,) = axt.plot(t_its, _smooth(t_secs, smooth), color="0.35",
                            linestyle=_LINESTYLES[-1], label="Iteration time (s)")
            all_lines.append(l)

    # Park the legend outside the axes so it never sits on top of the curves.
    ax1.legend(all_lines, [l.get_label() for l in all_lines], loc="center left",
               bbox_to_anchor=(_LEGEND_X + _AXIS_OFFSET * max(0, n_right - 1), 0.5),
               frameon=False)
    ax1.set_title(title or cfgs[0].stem)
    return fig


# Line styles cycle over reconstruction error + the semantic keys within a run;
# each run is told apart by color instead, so arbitrarily many runs stay legible.
_LINESTYLES = ("-", "--", ":", "-.")


def _as_run(run):
    """Normalize one run argument into a tuple of segment configs.

    A single config becomes a one-segment run; a list/tuple is taken as the
    ordered segments of a resume chain (concatenated by :func:`load_metrics`).
    """
    return tuple(run) if isinstance(run, (list, tuple)) else (run,)


# Model stems end in "_{n_iter_max}i"; everything before it is the structural
# identity a run shares with its own resume segments (see naming.model_stem).
_ITERS_STEM_RE = re.compile(r"_(\d+)i$")


def _chain_prefix(stem):
    """Stem with its trailing ``_{iters}i`` removed — the resume-chain identity."""
    return _ITERS_STEM_RE.sub("", stem)


def resume_chain(cfg):
    """Expand one config into its resume chain: earlier segments, then ``cfg``.

    A run resumed to a higher ``n_iter_max`` lands in a *new* log file that holds
    only the resumed tail — ``get_resume_state`` restarts the loop at
    ``start_iteration=latest_iter``, so that file's first line is (say) iteration
    251, not 1. Loading it alone therefore plots a curve that begins mid-axis.

    This scans the run's decomposition directory for sibling logs whose stem
    differs only in the iteration count and returns every segment with
    ``iters <= cfg.iters``, ascending — exactly the stitching the interactive
    browser does. Returns ``(cfg,)`` unchanged when nothing else is on disk, so
    it is always safe to call.
    """
    insp = cfg.insp if isinstance(cfg, RunRef) else cfg
    try:
        stem, log_path = cfg.stem, cfg.log_path
    except Exception:                      # duck-typed object without both
        return (cfg,)

    def _seg_insp(n):
        # Segments only need their log path (taken from disk below); the insp is
        # carried for callers that want load_tucker on an earlier checkpoint.
        try:
            return replace(insp, iters=n)
        except TypeError:                  # not a dataclass — keep the original
            return insp

    m = _ITERS_STEM_RE.search(stem)
    if m is None:                          # legacy stem: no iteration token
        return (cfg,)
    prefix, iters = _chain_prefix(stem), int(m.group(1))

    segments = []
    for path in log_path.parent.glob(f"{prefix}_*i_log.txt"):
        seg_stem = path.name[: -len("_log.txt")]
        seg_m = _ITERS_STEM_RE.search(seg_stem)
        # Guard the glob's "*": only stems differing *purely* in the iteration
        # count belong to the chain (e.g. "..._100r_0p1ss_2000mn_500i" shares the
        # prefix but is a different run).
        if seg_m is None or _chain_prefix(seg_stem) != prefix:
            continue
        n = int(seg_m.group(1))
        if n > iters:
            continue
        segments.append((n, RunRef(seg_stem, path, seg_stem, _seg_insp(n))))

    if not segments:
        return (cfg,)
    segments.sort(key=lambda s: s[0])
    # Keep the caller's own object as the final segment so titles/labels that
    # read `run[-1]` keep reporting exactly what was passed in.
    return tuple(ref for n, ref in segments if n < iters) + (cfg,)


def resolve_iters(insp):
    """Return ``insp``, or a copy pointing at whatever iteration count actually exists.

    ``InspectionConfig.iters`` is baked into the stem's trailing ``_{n}i`` token,
    so a config built for ``iters=500`` finds nothing on disk if that run was
    never resumed to exactly 500 (e.g. it only ever ran to 1000, or is still
    sitting at 200). This globs the run's directory for sibling logs that
    differ only in that iteration token and swaps in the best match: the
    highest available count that does not exceed the target if one exists
    (an unfinished or unresumed run), else the lowest available count above it
    (the run finished further than asked). Returns ``None`` if no sibling
    exists at all.
    """
    if insp.log_path.exists():
        return insp
    prefix = _chain_prefix(insp.stem)
    candidates = []
    for path in insp.log_path.parent.glob(f"{prefix}_*i_log.txt"):
        seg_stem = path.name[: -len("_log.txt")]
        m = _ITERS_STEM_RE.search(seg_stem)
        if m is None or _chain_prefix(seg_stem) != prefix:
            continue
        candidates.append(int(m.group(1)))
    if not candidates:
        return None
    at_or_below = [n for n in candidates if n <= insp.iters]
    chosen = max(at_or_below) if at_or_below else min(candidates)
    return replace(insp, iters=chosen)


def describe_run(cfg):
    """Print what a config actually loads — the check for a suspect curve.

    Reports the resume segments found on disk and, per segment, the raw
    iteration span plus the two defects :func:`load_metrics` now repairs:
    a span that does not start near 0 (a resumed tail loaded on its own) and
    repeated iteration numbers (an appended relaunch). Returns the chain.
    """
    chain = resume_chain(cfg)
    print(f"{len(chain)} segment(s) for {cfg.stem}:")
    for seg in chain:
        try:
            its, _rec, _sem = _read_log(seg)
        except FileNotFoundError:
            print(f"  {seg.stem}: MISSING LOG ({seg.log_path})")
            continue
        if not its:
            print(f"  {seg.stem}: no parsable metric lines")
            continue
        dup = len(its) - len(set(its))
        flags = []
        if its[0] > min(50, max(its) * 0.1):
            flags.append(f"starts at {its[0]} (resumed tail)")
        if dup:
            flags.append(f"{dup} duplicate iteration(s) (log appended)")
        note = ("  <- " + "; ".join(flags)) if flags else ""
        print(f"  {seg.stem}: {len(its)} points, iters {min(its)}..{max(its)}{note}")
    return chain


def _is_preloaded(run):
    """Return True if *run* is already a (iters, rec, sem) triple from average_runs."""
    return (isinstance(run, tuple) and len(run) == 3
            and isinstance(run[0], list)
            and (not run[0] or isinstance(run[0][0], int)))


def compare_metrics(configs, labels=None, sem_keys=("average_rank_score",),
                    plot_rec_error=True, plot_iter_time=False,
                    title="Training Metrics Comparison",
                    ax=None, clip_common=False, color_by=None, stitch=True,
                    legend_loc="right", smooth=None):
    """Overlay any number of runs on a shared figure; returns the figure.

    configs        : list of runs (a config, or a list of resume-chain segments),
                     or a ``{label: run}`` dict.
    labels         : legend names (fall back to each run's ``stem``).
    color_by       : callable ``label -> group`` or ``{label: group}``; runs in a
                     group share a color. Default: one color per run.
    plot_iter_time : add per-iteration time on its own right-hand axis.
    clip_common    : cap the x-axis at the shortest run's last iteration.
    stitch         : expand a single config into its resume chain so resumed
                     runs start at 0.
    legend_loc     : ``"right"`` or ``"below"``.
    smooth         : rolling-average window in iterations (``None``/1 = raw).

    Rec error and semantic keys are told apart by line style.
    """
    if isinstance(configs, dict):
        if labels is None:
            labels = list(configs.keys())
        configs = list(configs.values())

    runs = [c if _is_preloaded(c) else _as_run(c) for c in configs]
    if stitch:
        runs = [resume_chain(r[0]) if (not _is_preloaded(r) and len(r) == 1) else r
                for r in runs]
    if labels is None:
        labels = [None] * len(runs)
    labels = [str(lbl) if lbl is not None
              else ("averaged" if _is_preloaded(run) else run[-1].stem)
              for lbl, run in zip(labels, runs)]
    loaded = [run if _is_preloaded(run) else load_metrics(*run)
              for run in runs]  # [(its, rec, sem), ...]

    # Build per-run color from color_by, falling back to run index.
    palette = plt.cm.tab10.colors
    if color_by is not None:
        group_fn = color_by if callable(color_by) else color_by.__getitem__
        groups = [group_fn(lbl) for lbl in labels]
        # Preserve first-seen order so group→color is stable.
        seen: dict = {}
        for g in groups:
            if g not in seen:
                seen[g] = len(seen)
        run_colors = [palette[seen[g] % len(palette)] for g in groups]
    else:
        run_colors = [palette[i % len(palette)] for i in range(len(labels))]

    ax1 = ax or plt.subplots()[1]
    fig = ax1.figure
    ax1.set_xlabel("Iteration")
    ax1.grid(True)

    all_lines = []

    split_axes = (not plot_rec_error) and (len(sem_keys) == 2)
    n_right = 1 if (split_axes or sem_keys) else 0

    if split_axes:
        # One semantic key per axis; runs separated by color, keys by axis and style.
        ax2 = ax1.twinx()
        ax1.set_ylabel(sem_keys[0])
        ax2.set_ylabel(sem_keys[1])
        for (its, _rec, sem), lbl, c in zip(loaded, labels, run_colors):
            for k_i, (key, axis) in enumerate(zip(sem_keys, (ax1, ax2))):
                ls = _LINESTYLES[k_i % len(_LINESTYLES)]
                its_k, vals_k = _values_for_key(key, its, sem)
                (l,) = axis.plot(its_k, _smooth(vals_k, smooth), color=c, linestyle=ls,
                                 label=f"{lbl} · {key}")
                all_lines.append(l)
    else:
        if plot_rec_error:
            ax1.set_ylabel("Reconstruction Error")
        ax2 = ax1.twinx() if sem_keys else None
        if ax2 is not None:
            ax2.set_ylabel("Score")
        for (its, rec, sem), lbl, c in zip(loaded, labels, run_colors):
            if plot_rec_error:
                (l,) = ax1.plot(its, _smooth(rec, smooth), color=c, linestyle=_LINESTYLES[0],
                                label=f"{lbl} · Rec error")
                all_lines.append(l)
            for k_i, key in enumerate(sem_keys):
                # Offset so the first score curve isn't solid like rec error.
                ls = _LINESTYLES[(k_i + (1 if plot_rec_error else 0)) % len(_LINESTYLES)]
                its_k, vals_k = _values_for_key(key, its, sem)
                (l,) = ax2.plot(its_k, _smooth(vals_k, smooth), color=c, linestyle=ls,
                                label=f"{lbl} · {key}")
                all_lines.append(l)

    if plot_iter_time:
        axt = None  # created lazily, so a run set with no timings adds no axis
        for run, lbl, c in zip(runs, labels, run_colors):
            if _is_preloaded(run):
                continue
            t_its, t_secs = load_iter_times(*run)
            if not t_its:
                continue
            if axt is None:
                # With nothing else drawn, the time curves own the primary axis.
                if n_right or plot_rec_error:
                    axt = _time_axis(ax1, n_right)
                    n_right += 1
                else:
                    axt = ax1
                    ax1.set_ylabel("Iteration time (s)")
            (l,) = axt.plot(t_its, _smooth(t_secs, smooth), color=c, linestyle=_LINESTYLES[-1],
                            label=f"{lbl} · iter time")
            all_lines.append(l)

    if clip_common:
        finals = [max(its) for its, _, _ in loaded if its]
        if finals:
            ax1.set_xlim(right=min(finals))

    _place_legend(ax1, all_lines, legend_loc, n_right)
    ax1.set_title(title)
    return fig


# === run discovery =====================================================

@dataclass
class RunRef:
    """Duck-typed stand-in for InspectionConfig accepted by the plot functions.

    ``log_path`` / ``stem`` are read straight off disk, so this works for
    legacy-named runs too. ``insp`` is the reconstructed InspectionConfig for
    richer use (``load_tucker``, ...).
    """
    stem: str
    log_path: Path
    short: str
    insp: InspectionConfig


def _sf_to_set(sf):
    """config.json stores shared_factors as a list of pairs (or null)."""
    return {tuple(p) for p in sf} if sf else set()


# One token per linked group, single digit per mode (see shared_factor_suffix):
# "..._shared12_..." links modes 1 and 2, "..._shared012_..." links 0, 1 and 2.
_SHARED_STEM_RE = re.compile(r"(?:^|_)shared(\d+)(?=_|$)")

# "..._0p25ss_..." → subsample_frac 0.25 (see naming._ss; absent means 1.0).
_SS_STEM_RE = re.compile(r"_(\d+(?:p\d+)?)ss(?=_|$)")

# "..._500000mn_..." → max_nnz 500000 (see naming._mn; absent means off).
_MN_STEM_RE = re.compile(r"_(\d+)mn(?=_|$)")

# "..._CP3D_..." marks the experimental CP family, "..._TT100b3D_..." the
# Tucker-TT hybrid at bond 100 (see naming._order_tag); Tucker stems carry the
# bare "..._3D_..." tag instead.
_DECOMP_STEM_RE = re.compile(r"_CP\d+D(?=_|$)")
_TT_STEM_RE = re.compile(r"_TT(\d+)b\d+D(?=_|$)")

# "..._SGD3D_..." marks the SGD solver (naming._order_tag with
# solver="sgd"); MU stems carry no solver tag.
_SOLVER_STEM_RE = re.compile(r"_SGD(?:CP)?\d+D(?=_|$)")


def _ss_from_stem(stem):
    """Recover subsample_frac from a run's filename stem, or None if absent."""
    m = _SS_STEM_RE.search(stem)
    return float(m.group(1).replace("p", ".")) if m else None


def _mn_from_stem(stem):
    """Recover max_nnz from a run's filename stem, or None if absent."""
    m = _MN_STEM_RE.search(stem)
    return int(m.group(1)) if m else None


def _decomp_from_stem(stem):
    """Recover the decomposition family ("cp", "tt" or "tucker") from a run's stem.

    Config snapshots from before ``decomposition`` was recorded lack the field,
    but the model stem always carries the ``CP{order}D`` / ``TT{tt_rank}b{order}D``
    tag for those runs (see naming._order_tag) — so the stem is the
    authoritative fallback.
    """
    if _DECOMP_STEM_RE.search(stem):
        return "cp"
    return "tt" if _TT_STEM_RE.search(stem) else "tucker"


def _tt_rank_from_stem(stem):
    """Recover the TT bond dimension from a run's stem, or None if not a TT run."""
    m = _TT_STEM_RE.search(stem)
    return int(m.group(1)) if m else None


def _solver_from_stem(stem):
    """Recover the solver ("sgd" or "mu") from a run's stem — the SGD{order}D
    tag (naming._order_tag) is the authoritative fallback for snapshots that
    predate the ``solver`` config field."""
    return "sgd" if _SOLVER_STEM_RE.search(stem) else "mu"


def _sf_from_stem(stem):
    """Recover shared-factor links from a run's filename stem.

    Config snapshots from before shared_factors was recorded lack the field, but
    the model stem always carries the ``_shared..`` suffix — so the stem is the
    authoritative fallback when the snapshot has nothing. Each group token is
    expanded back into the pairwise links load_from_disk expects.
    """
    pairs = set()
    for grp in _SHARED_STEM_RE.findall(stem):
        modes = [int(ch) for ch in grp]
        pairs.update((a, b) for i, a in enumerate(modes) for b in modes[i + 1:])
    return pairs


def discover_datasets(data_dir=DATA_DIR):
    """List dataset dirs under ``tensors/`` that hold decomposition snapshots.

    A dataset qualifies if it has a ``decomposition/`` subdir containing at least
    one ``*_config.json``. Returned sorted by name.
    """
    tensors_dir = Path(data_dir) / "tensors"
    if not tensors_dir.is_dir():
        return []
    datasets = []
    for d in sorted(tensors_dir.iterdir()):
        decomp = d / "decomposition"
        if d.is_dir() and decomp.is_dir() and any(decomp.glob("*_config.json")):
            datasets.append(d.name)
    return datasets


def _discover_one(dataset, data_dir):
    """Yield one record dict per run found in a single dataset's decomposition dir."""
    decomp_dir = Path(data_dir) / "tensors" / dataset / "decomposition"
    if not decomp_dir.is_dir():
        return
    for cfg_path in decomp_dir.glob("*_config.json"):
        try:
            with open(cfg_path) as f:
                cfg = json.load(f).get("cfg", {})
        except Exception:
            continue
        exp, train = cfg.get("exp", {}), cfg.get("train", {})
        if not exp:
            continue

        rank = exp.get("rank", 150)
        rank0 = rank[0] if isinstance(rank, (list, tuple)) and rank else int(rank)
        dim = exp.get("dim")
        dim = tuple(dim) if isinstance(dim, list) else dim
        stem = cfg_path.name.replace("_config.json", "")
        sf = _sf_to_set(exp.get("shared_factors")) or _sf_from_stem(stem)
        name = exp.get("name") or "(unnamed)"
        iters = train.get("n_iter_max", 2000)
        # Old config snapshots stored subsample_frac under "train" (alongside
        # shared_factors/init); the stem's "_0p25ss" token is the last resort.
        ss = float(exp.get("subsample_frac") or train.get("subsample_frac")
                   or _ss_from_stem(stem) or 1.0)
        # max_nnz never lived under "train"; config exp → stem token → off.
        mn = int(exp.get("max_nnz") or _mn_from_stem(stem) or 0)
        # Snapshots predating the CP feature lack "decomposition"; fall back to
        # the "CP{order}D" / "TT{tt_rank}b{order}D" stem tag (naming._order_tag).
        decomposition = exp.get("decomposition") or _decomp_from_stem(stem)
        solver = exp.get("solver") or _solver_from_stem(stem)
        tt_rank = exp.get("tt_rank") or _tt_rank_from_stem(stem)

        insp = InspectionConfig(
            dim=dim, name=exp.get("name"), dataset=exp.get("dataset", dataset),
            method=exp.get("method", "siiSoftPlus"), divergence=exp.get("divergence", "kl"),
            order=exp.get("order") or infer_ngram_order(exp.get("dataset", dataset), exp.get("name")) or 3,
            iters=iters, rank=rank0,
            shared_factors=sf, subsample_frac=ss, max_nnz=(mn or None),
            solver=solver, decomposition=decomposition, tt_rank=tt_rank,
        )
        decomp_tag = "" if decomposition == "tucker" else f"[{decomposition.upper()}] "
        if decomposition == "tt" and tt_rank:
            decomp_tag = f"[TT b={tt_rank}] "
        if solver == "sgd":
            decomp_tag = f"[SGD] {decomp_tag}"
        log_path = decomp_dir / f"{stem}_log.txt"
        yield {
            "stem": stem,
            "dataset": dataset,
            "ref": RunRef(stem, log_path,
                          f"{decomp_tag}{dataset}|{name}|{insp.method}|{dim}d|r{rank0}|{iters}i", insp),
            "name": name, "divergence": insp.divergence, "method": insp.method,
            "order": insp.order, "dim": dim, "rank": rank0,
            "subsample_frac": ss, "max_nnz": mn, "iters": iters, "decomposition": decomposition,
            "has_log": log_path.exists() and log_path.stat().st_size > 0,
            "mtime": cfg_path.stat().st_mtime,
        }


def discover_runs(datasets="fineweb-en", data_dir=DATA_DIR):
    """Scan one or more datasets and return one record dict per run (newest first).

    ``datasets`` may be a single dataset name or an iterable of names; records
    from every dataset are pooled and sorted together. Each record carries its
    source ``dataset``, a ready-to-plot ``ref`` (:class:`RunRef`), and the facet
    fields the browser filters on.
    """
    if isinstance(datasets, str):
        datasets = [datasets]
    records = [rec for ds in datasets for rec in _discover_one(ds, data_dir)]
    records.sort(key=lambda r: r["mtime"], reverse=True)
    return records


# === interactive browser ===============================================

_FACETS = [("Name", "name"), ("Decomposition", "decomposition"),
           ("Divergence", "divergence"), ("Method", "method"),
           ("Dim", "dim"), ("Rank", "rank"), ("Subsample", "subsample_frac"),
           ("MaxNNZ", "max_nnz"), ("Iters", "iters")]


def _sortkey(v):
    return (0, v, "") if isinstance(v, (int, float)) else (1, 0, str(v))


# Ordered (field -> formatter) used to build legend labels for plotted runs.
# `dataset` is first so runs pulled from different directories stay distinguishable.
_LABEL_FIELDS = [
    ("dataset", lambda i: i.dataset),
    ("name", lambda i: i.name or "(unnamed)"),
    ("decomp", lambda i: getattr(i, "decomposition", "tucker")),
    ("method", lambda i: i.method),
    ("div", lambda i: i.divergence),
    ("dim", lambda i: f"{i.dim}d"),
    ("rank", lambda i: f"r{i.rank}"),
    ("ss", lambda i: f"ss{i.subsample_frac}"),
    ("mn", lambda i: f"mn{getattr(i, 'max_nnz', None) or 0}"),
    ("iters", lambda i: f"{i.iters}i"),
]


def _diff_labels(insps):
    """Split run descriptors into shared vs. distinguishing fields.

    Returns ``(shared, labels)`` where ``shared`` is a "|"-joined string of the
    fields identical across *all* runs (for the title) and ``labels`` is one
    "|"-joined string per run containing only the fields that differ — so the
    legend prints only what actually distinguishes the plotted runs.
    """
    shared, per_run = [], [[] for _ in insps]
    for _key, fn in _LABEL_FIELDS:
        vals = [fn(i) for i in insps]
        if len(set(vals)) == 1:
            shared.append(vals[0])
        else:
            for lbl, v in zip(per_run, vals):
                lbl.append(v)
    labels = ["|".join(p) if p else "(identical)" for p in per_run]
    return " | ".join(shared), labels


def _chain_key(rec):
    """Structural identity shared by all segments of one resume chain.

    A resumed run that runs to a higher ``n_iter_max`` lands in a new file whose
    stem differs only in the iteration count, so grouping by everything *except*
    ``iters`` collects a run and its continuations. Because the stem omits
    non-structural fields, two runs with the same key and the same ``iters`` would
    collide on disk — so within a key the ``iters`` values are always distinct and
    sort into clean, contiguous segments.
    """
    insp = rec["ref"].insp
    return (rec["dataset"], rec["name"], rec["decomposition"], rec["divergence"], rec["method"],
            rec["order"], rec["dim"], rec["rank"], rec["subsample_frac"], rec.get("max_nnz", 0),
            frozenset(insp.shared_factors or ()))


def make_run_browser(dataset="fineweb-en", data_dir=DATA_DIR,
                     default_sem_keys=("dim_consistency", "simlex_all_rho")):
    """Interactive faceted browser for picking and comparing two runs.

    Requires ``ipywidgets`` and an interactive matplotlib backend.

    Tick one or more datasets to pool their runs; ``dataset`` sets the initial
    ticks (a name, a list, or ``None`` for all). The *after* box hides runs
    whose snapshot predates a date, but resumed runs are still stitched to
    earlier segments (marked ``⛓×N``; untick *stitch* to see one segment).
    *sem_keys* offers only metrics logged for the picked runs, seeded from
    ``default_sem_keys``.

    Save the figure via the *save as* box or ``browser.get_figure()``.
    Returns the displayed ``VBox``.
    """
    try:
        import ipywidgets as widgets
        from IPython.display import clear_output, display
    except ImportError as e:  # pragma: no cover - UI-only dependency
        raise ImportError(
            "make_run_browser needs ipywidgets (and an IPython/Jupyter kernel). "
            "Install with `pip install ipywidgets`."
        ) from e

    # Discover every dataset with snapshots; `dataset` decides which start ticked.
    all_datasets = discover_datasets(data_dir)
    if dataset is None:
        initial = set(all_datasets)
    else:
        initial = {dataset} if isinstance(dataset, str) else set(dataset)
    # Surface any explicitly-requested dataset even if discovery didn't list it.
    for ds in initial:
        if ds not in all_datasets:
            all_datasets.append(ds)
    if not all_datasets:
        raise FileNotFoundError(
            f"No datasets with decomposition snapshots found under {Path(data_dir) / 'tensors'}."
        )

    plt.close("all")  # drop any figures from a previous run of this cell

    # We render plots into an Output widget rather than embedding fig.canvas, so
    # this works with the default inline backend — no %matplotlib widget / ipympl
    # required. A fresh figure is built on each redraw inside the Output and shown
    # via plt.show(), which (under the inline backend) displays it and closes it,
    # so figures never accumulate (no leak).
    plot_out = widgets.Output()

    # `mute` suppresses observer callbacks while we repopulate options
    # programmatically, so we never fight the traitlets event loop mid-rebuild.
    # `fig` holds the most recently drawn Figure so it can be saved/returned even
    # though the inline backend closes it after plt.show().
    # `sem_checks` maps each currently-offered sem key to its Checkbox widget;
    # rebuilt by _populate_sem_dd whenever the available keys change.
    state = {"records": [], "mute": False, "fig": None, "sem_checks": {}}

    dataset_chk = {
        ds: widgets.Checkbox(value=(ds in initial), description=ds, indent=False,
                             layout=widgets.Layout(width="auto", margin="0 12px 0 0"))
        for ds in all_datasets
    }
    facet_dd = {
        key: widgets.Dropdown(options=["(any)"], value="(any)", description=label,
                              style={"description_width": "75px"},
                              layout=widgets.Layout(width="235px"))
        for label, key in _FACETS
    }
    run_a = widgets.Dropdown(description="Run A", layout=widgets.Layout(width="98%"),
                             style={"description_width": "55px"})
    run_b = widgets.Dropdown(description="Run B", layout=widgets.Layout(width="98%"),
                             style={"description_width": "55px"})
    # A checkbox per sem key, inside a collapsible Accordion so it reads as one
    # "sem_keys" drop-down. Checkboxes are (re)built by _populate_sem_dd from
    # whatever keys actually appear in the logs of the currently-selected
    # run(s) — none exist until the first _refresh() populates sem_panel.
    sem_panel = widgets.VBox([], layout=widgets.Layout(
        max_height="160px", overflow_y="auto", padding="4px 8px"))
    sem_dd = widgets.Accordion(children=[sem_panel], layout=widgets.Layout(width="260px"))
    sem_dd.set_title(0, "sem_keys")
    sem_dd.selected_index = None  # start collapsed
    rec_chk = widgets.Checkbox(value=False, description="plot rec error", indent=False)
    # Only meaningful once rec error is actually being plotted; hidden until then
    # (see _on_rec_toggle) rather than just disabled, so the control row doesn't
    # carry a permanently-irrelevant checkbox.
    log_rec_chk = widgets.Checkbox(value=False, description="log scale (rec error)", indent=False,
                                   layout=widgets.Layout(display="none"))
    time_chk = widgets.Checkbox(value=False, description="plot iter time", indent=False)
    # Off by default: smoothing is a reading aid, so the raw series is what you
    # get unless you ask. The window box only shows while it's on (_on_smooth_toggle).
    smooth_chk = widgets.Checkbox(value=False, description="smooth", indent=False)
    smooth_win = widgets.BoundedIntText(value=5, min=2, max=10000, step=1, description="window",
                                        style={"description_width": "50px"},
                                        layout=widgets.Layout(width="130px", display="none"))
    stitch_chk = widgets.Checkbox(value=True, description="stitch resume chains", indent=False)
    clip_chk = widgets.Checkbox(value=False, description="clip to common iters", indent=False)
    after_box = widgets.Text(value="", description="after", placeholder="YYYY-MM-DD",
                             style={"description_width": "45px"},
                             layout=widgets.Layout(width="190px"))
    refresh_btn = widgets.Button(description="↻ Refresh", button_style="info",
                                 layout=widgets.Layout(width="110px"))
    save_name = widgets.Text(value="run_plot.png", description="save as",
                             style={"description_width": "55px"},
                             layout=widgets.Layout(width="240px"))
    save_btn = widgets.Button(description="💾 Save", button_style="success",
                              layout=widgets.Layout(width="90px"))
    status = widgets.HTML()

    def _label(rec, n_seg=1):
        when = _dt.datetime.fromtimestamp(rec["mtime"]).strftime("%b%d")
        flag = "" if rec["has_log"] else "  ⚠ no log"
        chain = f'  ⛓×{n_seg} (→{rec["iters"]}i)' if n_seg > 1 else ""
        decomp = "" if rec["decomposition"] == "tucker" else f'[{rec["decomposition"].upper()}] '
        mn = f' mn{rec["max_nnz"]}' if rec.get("max_nnz") else ""
        return (f'{decomp}[{rec["dataset"]}] {rec["name"]} | {rec["divergence"]}/{rec["method"]} | '
                f'{rec["dim"]}d r{rec["rank"]} ss{rec["subsample_frac"]}{mn} '
                f'{rec["iters"]}i  [{when}]{flag}{chain}')

    def _selected_datasets():
        return {ds for ds, c in dataset_chk.items() if c.value}

    def _dataset_recs():
        """Records belonging to the currently-checked datasets."""
        sel = _selected_datasets()
        return [r for r in state["records"] if r["dataset"] in sel]

    def _after_ts():
        """Parse the *after* box into a POSIX timestamp (cutoff for run mtime).

        Returns the timestamp (``float``) for a valid date, ``None`` when the box
        is empty (no filter), or ``False`` when the text can't be parsed.
        """
        s = after_box.value.strip()
        if not s:
            return None
        try:
            return pd.to_datetime(s).timestamp()
        except Exception:
            return False

    def _filtered():
        recs = _dataset_recs()
        ts = _after_ts()
        if isinstance(ts, float):
            recs = [r for r in recs if r["mtime"] >= ts]
        for key, dd in facet_dd.items():
            if dd.value != "(any)":
                recs = [r for r in recs if r[key] == dd.value]
        return recs

    def _populate_facets():
        # Facet options reflect only the runs in the checked datasets.
        recs = _dataset_recs()
        for key, dd in facet_dd.items():
            vals = sorted({r[key] for r in recs}, key=_sortkey)
            cur = dd.value
            dd.options = ["(any)"] + vals
            dd.value = cur if cur in dd.options else "(any)"

    def _chain_index():
        """Map each chain key to its segments (ascending by iters = resume order).

        Built over the dataset-filtered records (not the facet-filtered ones) so
        facet filters never truncate a chain, while dataset checkboxes still bound
        it — every member of a key shares its dataset.
        """
        idx = {}
        for r in _dataset_recs():
            idx.setdefault(_chain_key(r), []).append(r)
        for members in idx.values():
            members.sort(key=lambda m: m["iters"])
        return idx

    def _sem_keys_available(a, b):
        """Union of Sem_all keys actually logged for the given A/B run(s).

        Excludes ``_AUTO_METRIC_BLOCKLIST`` (OOV superseded by OOV_rate, and
        the stale tilde_* diagnostics) — same as the metric list in
        :func:`evaluate_runs`/:func:`make_run_ranker`.
        """
        keys = set()
        for run in (a, b):
            if not run:
                continue
            try:
                _, _, sem = load_metrics(*run)
            except FileNotFoundError:
                continue
            for d in sem:
                keys.update(d.keys())
        return sorted(keys - _AUTO_METRIC_BLOCKLIST)

    def _sem_keys_selected():
        return tuple(k for k, c in state["sem_checks"].items() if c.value)

    def _update_sem_title():
        n = len(_sem_keys_selected())
        sem_dd.set_title(0, f"sem_keys ({n} selected)" if n else "sem_keys (none)")

    def _on_sem_check(_change):
        # Runs on every toggle regardless of mute (cheap, keeps the title
        # accurate even mid-rebuild); the redraw itself still respects mute.
        _update_sem_title()
        if not state["mute"]:
            _redraw()

    def _populate_sem_dd():
        # Rebuild the checkbox list from what's actually in Run A / Run B's
        # logs, keeping whatever of the current selection still applies (or
        # seeding from default_sem_keys on the very first call). Never re-adds
        # a deselected key on its own — an empty selection stays empty.
        avail = _sem_keys_available(run_a.value, run_b.value)
        old = state["sem_checks"]
        checked = {k for k, c in old.items() if c.value} if old else set(default_sem_keys)
        new_checks = {}
        for key in avail:
            c = widgets.Checkbox(value=(key in checked), description=key, indent=False,
                                 layout=widgets.Layout(width="auto"))
            c.observe(_on_sem_check, "value")
            new_checks[key] = c
        state["sem_checks"] = new_checks
        sem_panel.children = list(new_checks.values())
        _update_sem_title()

    def _rebuild_ab():
        recs = _filtered()
        chains = _chain_index() if stitch_chk.value else None
        opts = []
        for r in recs:
            if chains is not None:
                # this run plus all earlier segments of its resume chain
                members = [m for m in chains[_chain_key(r)] if m["iters"] <= r["iters"]]
                refs = tuple(m["ref"] for m in members)
            else:
                refs = (r["ref"],)
            opts.append((_label(r, n_seg=len(refs)), refs))
        for dd, extra in ((run_a, []), (run_b, [("(none)", None)])):
            cur = dd.value
            dd.options = extra + opts
            vals = [v for _, v in dd.options]
            dd.value = cur if cur in vals else (dd.options[0][1] if dd.options else None)
        _populate_sem_dd()
        n_sel = len(_selected_datasets())
        ts = _after_ts()
        if ts is False:
            date_note = " &nbsp;|&nbsp; <span style='color:#c00'>unparsable 'after' date — ignored</span>"
        elif ts is not None:
            date_note = f" &nbsp;|&nbsp; after {_dt.datetime.fromtimestamp(ts).strftime('%Y-%m-%d')}"
        else:
            date_note = ""
        status.value = (f"<b>{len(recs)}</b> run(s) match the filters "
                        f"&nbsp;|&nbsp; {len(_dataset_recs())} in {n_sel} dataset(s) "
                        f"&nbsp;|&nbsp; {len(state['records'])} total{date_note}")

    def _redraw():
        with plot_out:
            clear_output(wait=True)
            a = run_a.value          # tuple of RunRef (chain segments), or None
            if not a:
                print("No run selected for A (adjust the filters).")
                return
            keys = _sem_keys_selected()
            b = run_b.value          # tuple of RunRef, or None for "(none)"
            win = smooth_win.value if smooth_chk.value else None
            fig, ax = plt.subplots(figsize=(10, 5), constrained_layout=True)
            try:
                # a[-1] is the representative (latest) segment — used for titles/labels.
                # stitch=False: the A/B options are already resolved to chains (or
                # deliberately to a single segment when the checkbox is unticked),
                # so the plot functions must not re-expand them.
                if b is None:
                    plot_metrics(*a, sem_keys=keys, plot_rec_error=rec_chk.value,
                                 plot_iter_time=time_chk.value, smooth=win,
                                 title=a[-1].stem, ax=ax, stitch=False)
                else:
                    shared, (la, lb) = _diff_labels([a[-1].insp, b[-1].insp])
                    compare_metrics([list(a), list(b)], [la, lb],
                                    sem_keys=keys, plot_rec_error=rec_chk.value,
                                    plot_iter_time=time_chk.value, smooth=win,
                                    title=shared, ax=ax, clip_common=clip_chk.value,
                                    stitch=False)
            except FileNotFoundError as e:
                plt.close(fig)
                state["fig"] = None
                print(f"Log file missing:\n{e}")
                return
            if rec_chk.value and log_rec_chk.value:
                ax.set_yscale("log")
            # Keep a handle on the Figure object before plt.show() closes it (under
            # the inline backend) so it can still be saved/returned afterwards.
            state["fig"] = fig
            plt.show()  # inline backend: renders into the Output and closes the fig

    def _on_filter(_change):
        if state["mute"]:
            return
        state["mute"] = True
        try:
            _rebuild_ab()
        finally:
            state["mute"] = False
        _redraw()

    def _on_dataset(_change):
        # Toggling a dataset changes which facet values exist, so repopulate the
        # facets (clearing any now-invalid selection) before rebuilding A/B.
        if state["mute"]:
            return
        state["mute"] = True
        try:
            _populate_facets()
            _rebuild_ab()
        finally:
            state["mute"] = False
        _redraw()

    def _on_select(_change):
        if not state["mute"]:
            _redraw()

    def _on_run_select(_change):
        # Picking a different Run A/B directly (as opposed to via _rebuild_ab,
        # which already calls _populate_sem_dd itself) changes which sem keys
        # actually exist, so refresh the dropdown's options before redrawing.
        if state["mute"]:
            return
        state["mute"] = True
        try:
            _populate_sem_dd()
        finally:
            state["mute"] = False
        _redraw()

    def _on_rec_toggle(change):
        # Show the log-scale toggle only while it does something; untick it
        # along with hiding it so a stale check doesn't silently apply once
        # rec error is switched back on.
        if change["new"]:
            log_rec_chk.layout.display = ""
        else:
            log_rec_chk.layout.display = "none"
            log_rec_chk.value = False

    def _on_smooth_toggle(change):
        # Same pattern as _on_rec_toggle: the window box is only visible while
        # smoothing is on. Its value is kept (it's a setting, not a filter) — with
        # the checkbox off _redraw passes no window, so nothing stale applies.
        smooth_win.layout.display = "" if change["new"] else "none"

    def _save(_btn=None):
        fig = state["fig"]
        if fig is None:
            status.value = "<b>Nothing to save</b> — no figure drawn yet."
            return
        out = Path(save_name.value).expanduser()
        if not out.suffix:
            out = out.with_suffix(".png")
        try:
            # savefig works on the Figure object even after the inline backend
            # has closed it, since `state["fig"]` keeps it alive.
            fig.savefig(out, dpi=150, bbox_inches="tight")
        except Exception as e:  # pragma: no cover - filesystem/IO errors
            status.value = f"<b>Save failed:</b> {e}"
            return
        status.value = f"Saved to <code>{out.resolve()}</code>"

    def _refresh(_btn=None):
        state["mute"] = True
        try:
            # Pool every checkbox's dataset so toggling never needs a rescan.
            state["records"] = discover_runs(list(dataset_chk), data_dir)
            _populate_facets()
            _rebuild_ab()
        finally:
            state["mute"] = False
        _redraw()

    for c in dataset_chk.values():
        c.observe(_on_dataset, "value")
    for dd in facet_dd.values():
        dd.observe(_on_filter, "value")
    run_a.observe(_on_run_select, "value")
    run_b.observe(_on_run_select, "value")
    # sem key checkboxes wire their own _on_sem_check observer as they're
    # (re)created in _populate_sem_dd — nothing to attach on sem_dd itself.
    rec_chk.observe(_on_select, "value")
    # Second observer on the same trait: toggles log_rec_chk's visibility
    # (and clears it when hidden) independently of the redraw triggered above.
    rec_chk.observe(_on_rec_toggle, "value")
    log_rec_chk.observe(_on_select, "value")
    time_chk.observe(_on_select, "value")
    # Smoothing only changes the drawn y-values — redraw, plus a visibility
    # observer for the window box (cf. rec_chk / log_rec_chk above).
    smooth_chk.observe(_on_select, "value")
    smooth_chk.observe(_on_smooth_toggle, "value")
    smooth_win.observe(_on_select, "value")
    # Clipping only changes the x-axis limit on the existing curves — just redraw.
    clip_chk.observe(_on_select, "value")
    # Toggling stitch changes the A/B option values (chains vs single runs), so it
    # needs a rebuild — _on_filter does exactly that (rebuild + redraw).
    stitch_chk.observe(_on_filter, "value")
    # A date cutoff just narrows the selectable runs — same rebuild path as facets.
    after_box.observe(_on_filter, "value")
    refresh_btn.on_click(_refresh)
    save_btn.on_click(_save)

    _refresh()

    datasets_box = widgets.VBox([
        widgets.HTML("<b>Datasets</b> &nbsp;<small>(tick to include / compare across dirs)</small>"),
        widgets.HBox(list(dataset_chk.values()), layout=widgets.Layout(flex_flow="row wrap")),
    ])
    filters = widgets.HBox(list(facet_dd.values()), layout=widgets.Layout(flex_flow="row wrap"))
    controls = widgets.HBox([sem_dd, rec_chk, log_rec_chk, time_chk, smooth_chk, smooth_win,
                             stitch_chk, clip_chk, after_box,
                             refresh_btn, save_name, save_btn],
                            layout=widgets.Layout(flex_flow="row wrap"))
    ui = widgets.VBox([datasets_box, filters, status, run_a, run_b, controls, plot_out])
    # Expose the live state so callers can grab the current Figure out of the UI,
    # e.g. `fig = browser.get_figure(); fig.savefig(...)` or display it elsewhere.
    ui.get_figure = lambda: state["fig"]
    display(ui)
    return ui


# === best-run ranking =================================================
#
# Two layers, mirroring the rest of the module: pure functions
# (:func:`evaluate_runs` / :func:`find_best`) that any caller drives with
# arguments, and :func:`make_run_ranker`, a widget front-end wired to the exact
# same helpers. "Best" means the best value the metric ever reached over the
# run's logged iterations — a min for error-like keys, a max otherwise — which
# is fair across runs of different lengths and matches how ``get_resume_state``
# tracks ``best_sem_score``.

# A metric whose name looks like an error/loss/distance ranks ascending (lower
# is better); everything else ranks descending. Override per call with
# ``lower_is_better``.
_ERROR_LIKE_RE = re.compile(r"(?:^|_)(rec_error|error|loss|rmse|mae|nll|perplexity|ppl)(?:_|$)")

_NUM_RE = r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?"
_THRESH_RE = re.compile(rf"([A-Za-z0-9_]+)\s*(<=|>=|<|>|==|=)\s*({_NUM_RE})")

_OPS = {"<": operator.lt, "<=": operator.le, ">": operator.gt,
        ">=": operator.ge, "==": operator.eq, "=": operator.eq}

# Facet fields carried on every discovery record that a filter may key on.
_FILTER_FACETS = {"name", "decomposition", "divergence", "method", "dim", "rank",
                  "subsample_frac", "max_nnz", "iters", "order", "dataset"}


def _lower_is_better_fn(override):
    """Build ``key -> bool`` ("does a smaller value rank higher for this metric").

    ``override`` may be a bool (applies to every metric), an iterable of keys
    that are lower-better, or ``None`` to infer it from the key name.
    """
    if override is None:
        return lambda k: bool(_ERROR_LIKE_RE.search(str(k).lower()))
    if isinstance(override, bool):
        return lambda k: override
    keys = set(override)
    return lambda k: k in keys


def _parse_thresholds(thresholds):
    """Normalise threshold criteria to a list of ``(key, op, value)`` triples.

    Accepts a string (``"rec_error < 0.5, simlex_all_rho > 0.3"`` — comma,
    semicolon or ``and`` separated), a ``{key: (op, value)}`` / ``{key: value}``
    dict, or an iterable of ``(key, op, value)``. ``op`` may be ``None``, meaning
    "resolve by direction" (``>=`` for higher-better metrics, ``<=`` for
    error-like ones) — deferred to :func:`_apply_thresholds`.
    """
    if not thresholds:
        return []
    out = []
    if isinstance(thresholds, str):
        for part in re.split(r"[,;]|\band\b", thresholds, flags=re.IGNORECASE):
            part = part.strip()
            if not part:
                continue
            m = _THRESH_RE.fullmatch(part)
            if not m:
                raise ValueError(
                    f"Can't parse threshold {part!r} (expected e.g. 'rec_error < 0.5')")
            out.append((m.group(1), m.group(2), float(m.group(3))))
        return out
    if isinstance(thresholds, dict):
        for key, spec in thresholds.items():
            if isinstance(spec, (tuple, list)) and len(spec) == 2:
                out.append((key, str(spec[0]), float(spec[1])))
            else:
                out.append((key, None, float(spec)))
        return out
    for key, op, val in thresholds:
        out.append((key, op if op is None else str(op), float(val)))
    return out


def _apply_thresholds(df, thresholds, lower_is_better=None):
    """Drop rows of a scored frame that fail any threshold (NaN never passes)."""
    lib = _lower_is_better_fn(lower_is_better)
    for key, op, val in _parse_thresholds(thresholds):
        if key not in df.columns:
            raise KeyError(f"No metric column {key!r} to threshold on "
                           f"(scored: {sorted(df.attrs.get('metrics', []))})")
        if op is None:
            op = "<=" if lib(key) else ">="
        df = df[_OPS[op](df[key], val)]
    return df


def _apply_facet_filters(records, filters):
    """Keep only records matching every ``{facet: value | [values]}`` entry.

    A ``list``/``set`` value allows any of several; anything else (including a
    ``tuple``, so ``dim=(1000, 1000, 1000)`` stays one value) is an exact match.
    """
    if not filters:
        return records
    for key, want in filters.items():
        if key not in _FILTER_FACETS:
            raise KeyError(f"Unknown filter facet {key!r}; "
                           f"choose from {sorted(_FILTER_FACETS)}")
        allowed = list(want) if isinstance(want, (list, set)) else [want]
        records = [r for r in records if r.get(key) in allowed]
    return records


def _facet_filter_df(df, filters):
    """DataFrame counterpart of :func:`_apply_facet_filters` (used by the UI)."""
    for key, want in (filters or {}).items():
        allowed = list(want) if isinstance(want, (list, set)) else [want]
        df = df[df[key].apply(lambda v: v in allowed)]
    return df


def _metric_series(loaded, key):
    """``(iters, values)`` for one metric from a :func:`load_metrics` triple.

    ``key`` is ``"rec_error"`` for the reconstruction-error curve, or any key
    present in the ``Sem_all`` dicts.
    """
    its, rec, sem = loaded
    if key in ("rec_error", "reconstruction_error"):
        return list(its), list(rec)
    return _values_for_key(key, its, sem)


def _collapse_chains(records, stitch=True):
    """Group resume segments; return ``[(representative_record, refs_tuple), ...]``.

    With ``stitch`` the representative is the furthest-reaching segment and
    ``refs`` are every segment of the chain (ascending), ready for
    :func:`load_metrics`. Without it each record is its own singleton.
    """
    if not stitch:
        return [(r, (r["ref"],)) for r in records]
    idx = {}
    for r in records:
        idx.setdefault(_chain_key(r), []).append(r)
    out = []
    for members in idx.values():
        members.sort(key=lambda m: m["iters"])
        out.append((members[-1], tuple(m["ref"] for m in members)))
    return out


def _score_loaded(loaded, metrics, lower_is_better_fn):
    """Best-ever value / iteration / final value for each metric of one run."""
    if loaded is None or not loaded[0]:
        return None
    its_all = loaded[0]
    out = {"last_iter": int(its_all[-1])}
    for key in metrics:
        its, vals = _metric_series(loaded, key)
        if not vals:
            continue
        bi = int(np.argmin(vals) if lower_is_better_fn(key) else np.argmax(vals))
        out[key] = float(vals[bi])
        out[f"{key}_iter"] = int(its[bi])
        out[f"{key}_final"] = float(vals[-1])
    return out


def _combined_score(df, keys, lower_is_better=None):
    """Row-wise mean of several already-scored metric columns.

    Each column is sign-flipped first when it's lower-is-better, so the result
    is always "higher = better" even when mixing e.g. ``rec_error`` with
    ``simlex_all_rho`` — a plain mean across inconsistent directions would
    otherwise reward runs for being *worse* on half the metrics.
    """
    lib = _lower_is_better_fn(lower_is_better)
    parts = [(-df[key].astype(float) if lib(key) else df[key].astype(float)) for key in keys]
    return pd.concat(parts, axis=1).mean(axis=1)


# Identity columns, in display order. The ranking metric's own columns are
# spliced in right after `full` (see _order_columns) so the number you sorted
# on is the first thing next to the run name.
_RANK_FRONT = ("rank_pos", "full", "dataset", "name", "decomposition",
               "divergence", "method", "dim", "rank", "subsample_frac",
               "iters", "when")

# Bookkeeping identity fields that are rarely what you're scanning for — they
# land just before `refs` instead of crowding the front next to the metrics.
_RANK_TAIL = ("max_nnz", "n_segments", "has_log", "last_iter")


def _order_columns(df, metric, component_keys=()):
    """rank_pos, full, the ranking metric's columns, then the rest; refs last.

    ``component_keys`` are the individual metrics a composite ``metric`` (e.g.
    ``combined_mean``) was averaged from — their own ``key``/``key_iter``/
    ``key_final`` columns are spliced in right after the composite's, ahead of
    the identity columns, so the numbers that produced the mean are the first
    thing next to it.
    """
    lead = [c for c in ("rank_pos", "full") if c in df.columns]
    metric_cols = [c for c in (metric, f"{metric}_iter", f"{metric}_final")
                   if c in df.columns]
    for key in component_keys:
        metric_cols += [c for c in (key, f"{key}_iter", f"{key}_final")
                        if c in df.columns and c not in metric_cols]
    ident = [c for c in _RANK_FRONT if c in df.columns and c not in lead and c not in metric_cols]
    tail_ident = [c for c in _RANK_TAIL if c in df.columns]
    drop = set(lead) | set(metric_cols) | set(ident) | set(tail_ident) | {"refs", "mtime"}
    mid = [c for c in df.columns if c not in drop]
    tail = ["refs"] if "refs" in df.columns else []
    return df[lead + metric_cols + ident + mid + tail_ident + tail]


def _rank_df(df, metric, thresholds=None, lower_is_better=None, top=None):
    """Threshold, sort best-first on ``metric`` and (re)number a scored frame.

    Shared by :func:`find_best` and :func:`make_run_ranker`.
    """
    lib = _lower_is_better_fn(lower_is_better)
    df = df[df[metric].notna()].copy()
    df = _apply_thresholds(df, thresholds, lower_is_better=lower_is_better)
    df = df.sort_values(metric, ascending=lib(metric), kind="mergesort")
    df = df.reset_index(drop=True)
    df.insert(0, "rank_pos", df.index + 1)
    if top:
        df = df.head(int(top)).copy()
    return df


# Raw evaluate_sample() keys that "auto" metric discovery skips: OOV is a raw
# count made redundant by OOV_rate (same information, comparable across runs
# of different sample sizes), and the tilde_* pair are stale rewrite-era
# diagnostics. Still readable directly from a log's Sem_all if you actually
# want them — pass them explicitly via ``metrics=``.
_AUTO_METRIC_BLOCKLIST = {"OOV", "tilde_excluded_prob_score", "tilde_rate"}


def evaluate_runs(datasets="fineweb-en", data_dir=DATA_DIR,
                  metrics=("average_rank_score",), filters=None, after=None,
                  lower_is_better=None, stitch=True):
    """Score every discovered run and return one row per run as a DataFrame.

    Scans ``datasets`` with :func:`discover_runs`, collapses resume chains to
    their furthest-reaching segment (loaded whole) when ``stitch``, and reads
    each log once. For every metric ``k`` the row carries ``k`` (its best-ever
    value — min for error-like keys, max otherwise), ``k_iter`` (the iteration
    it occurred at) and ``k_final`` (its last logged value); a run whose log is
    missing or has no line for ``k`` gets NaN there.

    ``metrics`` may be ``"auto"`` (``None``) to score ``rec_error`` plus every
    semantic key found in the scanned logs, minus ``_AUTO_METRIC_BLOCKLIST``.
    ``filters`` is a
    ``{facet: value | [values]}`` dict over the discovery facets; ``after`` is a
    date string dropping older config snapshots. The metric names actually
    scored are also on ``df.attrs["metrics"]``. The ``refs`` column holds each
    row's chain segments, ready for :func:`compare_metrics`.
    """
    auto = metrics is None or (isinstance(metrics, str) and metrics == "auto")
    if not auto:
        metrics = [metrics] if isinstance(metrics, str) else list(metrics)
    lib = _lower_is_better_fn(lower_is_better)

    records = _apply_facet_filters(discover_runs(datasets, data_dir), filters)
    if after is not None:
        cutoff = pd.to_datetime(after).timestamp()
        records = [r for r in records if r["mtime"] >= cutoff]

    loaded_chains = []
    for rep, refs in _collapse_chains(records, stitch=stitch):
        try:
            loaded = load_metrics(*refs)
        except FileNotFoundError:
            loaded = None
        loaded_chains.append((rep, refs, loaded))

    if auto:
        keyset = set()
        for _rep, _refs, loaded in loaded_chains:
            if loaded:
                for d in loaded[2]:
                    keyset.update(d)
        keyset -= _AUTO_METRIC_BLOCKLIST
        metrics = ["rec_error"] + sorted(keyset)

    rows = []
    for rep, refs, loaded in loaded_chains:
        scored = _score_loaded(loaded, metrics, lib) or {}
        row = {
            "full": rep["stem"], "dataset": rep["dataset"], "name": rep["name"],
            "decomposition": rep["decomposition"], "divergence": rep["divergence"],
            "method": rep["method"], "order": rep["order"], "dim": rep["dim"],
            "rank": rep["rank"], "subsample_frac": rep["subsample_frac"],
            "max_nnz": rep["max_nnz"], "iters": rep["iters"],
            "n_segments": len(refs),
            "mtime": rep["mtime"],
            "when": _dt.datetime.fromtimestamp(rep["mtime"]).strftime("%Y-%m-%d"),
            "has_log": bool(loaded and loaded[0]),
            "last_iter": scored.get("last_iter", np.nan),
        }
        for key in metrics:
            row[key] = scored.get(key, np.nan)
            row[f"{key}_iter"] = scored.get(f"{key}_iter", np.nan)
            row[f"{key}_final"] = scored.get(f"{key}_final", np.nan)
        row["refs"] = list(refs)
        rows.append(row)

    df = pd.DataFrame(rows)
    df.attrs["metrics"] = list(metrics)
    return df


def find_best(metric="average_rank_score", datasets="fineweb-en", data_dir=DATA_DIR,
              filters=None, thresholds=None, after=None, top=None,
              extra_metrics=(), lower_is_better=None, stitch=True):
    """Rank decomposition runs by their best-ever value of ``metric``.

    Scores every run in ``datasets`` (see :func:`evaluate_runs`) and returns a
    DataFrame sorted best-first — ascending for error-like metrics, descending
    otherwise (override with ``lower_is_better``: a bool, or an iterable of
    lower-better keys). ``rank_pos`` is the 1-based position.

    Criteria narrow the field before ranking:

    * ``filters`` — ``{facet: value | [values]}`` equality filters over the
      discovery facets (``name``, ``method``, ``decomposition``, ``divergence``,
      ``dim``, ``rank``, ``subsample_frac``, ``max_nnz``, ``iters``, ``order``,
      ``dataset``). A list/set allows several values.
    * ``thresholds`` — numeric cutoffs on any metric: a string
      (``"rec_error < 0.5, simlex_all_rho > 0.3"``), a ``{key: (op, value)}``
      dict, or a list of ``(key, op, value)``. A bare number means "at least"
      for higher-better metrics, "at most" for error-like ones.
    * ``after`` — drop runs whose config snapshot predates this date.

    Every metric named in ``thresholds`` and ``extra_metrics`` is scored and
    kept as a column alongside ``metric``. ``top`` truncates to the N best. The
    ``refs`` column feeds :func:`compare_metrics` / :func:`plot_top` directly.
    """
    parsed = _parse_thresholds(thresholds)
    score_metrics = list(dict.fromkeys(
        [metric, *(k for k, _, _ in parsed), *extra_metrics]))
    df = evaluate_runs(datasets, data_dir, metrics=score_metrics, filters=filters,
                       after=after, lower_is_better=lower_is_better, stitch=stitch)
    if df.empty or metric not in df.columns:
        return df
    ranked = _rank_df(df, metric, thresholds=parsed,
                      lower_is_better=lower_is_better, top=top)
    ranked = _order_columns(ranked, metric)
    ranked.attrs["metrics"] = df.attrs.get("metrics", score_metrics)
    return ranked


def plot_top(ranked, metric="average_rank_score", n=5, ax=None, **compare_kw):
    """Overlay the top ``n`` runs of a :func:`find_best` result on one figure.

    Uses the ``refs`` column (already-resolved resume chains, so ``stitch`` is
    off) and ``compare_metrics``: the ranking metric is drawn as the semantic
    curve, or as the rec-error curve when it *is* ``rec_error``. The run names
    go under the plot (``legend_loc="below"``, override via ``compare_kw``).
    Extra keyword args pass through to :func:`compare_metrics`.

    ``metric`` also accepts an iterable of several metric names (e.g. the set
    behind a ranking by combined mean, see :func:`_combined_score`) — every
    one is overlaid as its own curve, since a composite score itself has no
    per-iteration series to plot.
    """
    top = ranked.head(int(n))
    configs = [list(r) for r in top["refs"]]
    labels = list(top["full"])
    keys = (metric,) if isinstance(metric, str) else tuple(metric)
    is_rec = keys in (("rec_error",), ("reconstruction_error",))
    compare_kw.setdefault("legend_loc", "below")
    title_metric = metric if isinstance(metric, str) else " & ".join(keys)
    return compare_metrics(
        configs, labels,
        sem_keys=() if is_rec else keys,
        plot_rec_error=is_rec,
        title=f"Top {len(top)} by {title_metric}",
        ax=ax, stitch=False, **compare_kw,
    )


# === interactive ranker ===============================================

def make_run_ranker(dataset="fineweb-en", data_dir=DATA_DIR,
                    default_metric="dim_consistency"):
    """Faceted widget front-end for :func:`find_best`.

    Requires ``ipywidgets`` and an IPython kernel. Same filters, thresholds and
    ranking as :func:`find_best`, driven from widgets.

    * **metric** — tick one to rank by it; tick several to rank by the mean of
      their sign-normalized best values (``combined_mean``, see
      :func:`_combined_score`).
    * **thresholds** — e.g. ``rec_error < 0.5, simlex_all_rho > 0.3``.
    * **after** / **top N** — date cutoff / keep the N best (0 = all).

    *Plot top* overlays the first rows via :func:`plot_top`; results are
    available as ``ranker.get_figure()`` / ``ranker.get_ranking()``.
    """
    try:
        import ipywidgets as widgets
        from IPython.display import clear_output, display
    except ImportError as e:  # pragma: no cover - UI-only dependency
        raise ImportError(
            "make_run_ranker needs ipywidgets (and an IPython/Jupyter kernel). "
            "Install with `pip install ipywidgets`."
        ) from e

    all_datasets = discover_datasets(data_dir)
    if dataset is None:
        initial = set(all_datasets)
    else:
        initial = {dataset} if isinstance(dataset, str) else set(dataset)
    for ds in initial:
        if ds not in all_datasets:
            all_datasets.append(ds)
    if not all_datasets:
        raise FileNotFoundError(
            f"No datasets with decomposition snapshots found under "
            f"{Path(data_dir) / 'tensors'}."
        )

    plt.close("all")
    state = {"full": None, "ranked": None, "fig": None, "mute": False, "metric_checks": {}}

    dataset_chk = {
        ds: widgets.Checkbox(value=(ds in initial), description=ds, indent=False,
                             layout=widgets.Layout(width="auto", margin="0 12px 0 0"))
        for ds in all_datasets
    }
    facet_dd = {
        key: widgets.Dropdown(options=["(any)"], value="(any)", description=label,
                              style={"description_width": "75px"},
                              layout=widgets.Layout(width="235px"))
        for label, key in _FACETS
    }
    # One checkbox per metric, inside a collapsible Accordion — the same
    # "check none/one/several" pattern as sem_keys in make_run_browser.
    # Options are (re)built in _recompute from every metric found in the
    # checked datasets. One ticked ranks by that metric directly; several rank
    # by the mean of their sign-normalized best-ever values (see
    # _combined_score) as a synthetic combined_mean column.
    metric_panel = widgets.VBox([], layout=widgets.Layout(
        max_height="160px", overflow_y="auto", padding="4px 8px"))
    metric_dd = widgets.Accordion(children=[metric_panel], layout=widgets.Layout(width="280px"))
    metric_dd.set_title(0, "metric")
    metric_dd.selected_index = None
    thresh_box = widgets.Text(value="", description="thresholds",
                              placeholder="rec_error < 0.5, simlex_all_rho > 0.3",
                              style={"description_width": "75px"},
                              layout=widgets.Layout(width="60%"))
    after_box = widgets.Text(value="", description="after", placeholder="YYYY-MM-DD",
                             style={"description_width": "45px"},
                             layout=widgets.Layout(width="190px"))
    top_box = widgets.BoundedIntText(value=10, min=0, max=9999, description="top N",
                                     style={"description_width": "45px"},
                                     layout=widgets.Layout(width="130px"))
    stitch_chk = widgets.Checkbox(value=True, description="stitch resume chains",
                                  indent=False)
    refresh_btn = widgets.Button(description="↻ Refresh", button_style="info",
                                 layout=widgets.Layout(width="110px"))
    nplot_box = widgets.BoundedIntText(value=5, min=1, max=20, description="plot top",
                                       style={"description_width": "60px"},
                                       layout=widgets.Layout(width="150px"))
    plot_btn = widgets.Button(description="📈 Plot top",
                              layout=widgets.Layout(width="120px"))
    save_name = widgets.Text(value="ranker_top.png", description="save as",
                             style={"description_width": "55px"},
                             layout=widgets.Layout(width="240px"))
    save_btn = widgets.Button(description="💾 Save", button_style="success",
                              layout=widgets.Layout(width="90px"))
    status = widgets.HTML()
    table_out = widgets.Output()
    plot_out = widgets.Output()

    def _selected_datasets():
        return {ds for ds, c in dataset_chk.items() if c.value}

    def _facet_filters():
        return {key: dd.value for key, dd in facet_dd.items() if dd.value != "(any)"}

    def _populate_facets():
        df = state["full"]
        for key, dd in facet_dd.items():
            vals = ([] if df is None or key not in df.columns
                    else sorted(set(df[key].tolist()), key=_sortkey))
            cur = dd.value
            dd.options = ["(any)"] + vals
            dd.value = cur if cur in dd.options else "(any)"

    def _metric_keys_selected():
        return tuple(k for k, c in state["metric_checks"].items() if c.value)

    def _update_metric_title():
        keys = _metric_keys_selected()
        if not keys:
            metric_dd.set_title(0, "metric (none)")
        elif len(keys) == 1:
            metric_dd.set_title(0, f"metric: {keys[0]}")
        else:
            metric_dd.set_title(0, f"metric ({len(keys)} averaged)")

    def _on_metric_check(_change):
        _update_metric_title()
        if not state["mute"]:
            _rerank()

    def _populate_metric_dd(metrics):
        # Mirrors _populate_sem_dd in make_run_browser: rebuild the checklist
        # from the metrics on hand, keeping whatever of the current selection
        # is still valid — seeded with default_metric (or the first metric, if
        # that's not among them) the very first time.
        old = state["metric_checks"]
        if old:
            checked = {k for k, c in old.items() if c.value}
        elif default_metric in metrics:
            checked = {default_metric}
        else:
            checked = {metrics[0]} if metrics else set()
        new_checks = {}
        for key in metrics:
            c = widgets.Checkbox(value=(key in checked), description=key, indent=False,
                                 layout=widgets.Layout(width="auto"))
            c.observe(_on_metric_check, "value")
            new_checks[key] = c
        state["metric_checks"] = new_checks
        metric_panel.children = list(new_checks.values())
        _update_metric_title()

    def _recompute(_evt=None):
        sel = sorted(_selected_datasets())
        if not sel:
            state["full"] = None
            status.value = "<b>No dataset selected.</b>"
            _rerank()
            return
        state["mute"] = True
        try:
            df = evaluate_runs(sel, data_dir, metrics="auto",
                               stitch=stitch_chk.value)
            state["full"] = df
            metrics = df.attrs.get("metrics", [default_metric]) or [default_metric]
            _populate_metric_dd(metrics)
            _populate_facets()
        finally:
            state["mute"] = False
        _rerank()

    def _rerank(_evt=None):
        if state["mute"]:
            return
        df = state["full"]
        if df is None or df.empty:
            state["ranked"] = None
            _render_table("No runs scored.")
            return
        view = _facet_filter_df(df, _facet_filters())
        note = ""
        after = after_box.value.strip()
        if after:
            try:
                view = view[view["mtime"] >= pd.to_datetime(after).timestamp()]
            except Exception:
                note = " &nbsp;|&nbsp; <span style='color:#c00'>unparsable 'after' date</span>"

        keys = _metric_keys_selected()
        if not keys:
            state["ranked"] = None
            status.value = "<b style='color:#c00'>Tick at least one metric.</b>"
            _render_table("Pick a metric above.")
            return
        if len(keys) == 1:
            metric = keys[0]
            desc = metric
        else:
            # combined_mean is sign-normalized (see _combined_score) so it's
            # always "higher = better" — no need for _rank_df's direction guess.
            view = view.copy()
            view["combined_mean"] = _combined_score(view, keys)
            metric = "combined_mean"
            desc = f"mean of {', '.join(keys)}"
        asc = _lower_is_better_fn(None)(metric)

        try:
            # lower_is_better left at its default (per-key regex inference):
            # "combined_mean" doesn't match the error-like pattern, so it's
            # correctly treated as higher-is-better without a global override
            # — which would otherwise also flip the direction of any *other*
            # metric named in thresholds (e.g. "rec_error < 0.5").
            ranked = _rank_df(view, metric, thresholds=thresh_box.value,
                              top=(top_box.value or None))
        except (ValueError, KeyError) as e:
            state["ranked"] = None
            status.value = f"<b style='color:#c00'>{e}</b>"
            _render_table("Fix the criteria above.")
            return
        state["ranked"] = _order_columns(ranked, metric,
                                         component_keys=keys if len(keys) > 1 else ())
        status.value = (
            f"<b>{len(ranked)}</b> run(s) ranked by <code>{desc}</code> "
            f"({'ascending' if asc else 'descending'}) &nbsp;|&nbsp; "
            f"{len(view)} pass the filters &nbsp;|&nbsp; {len(df)} scored{note}")
        _render_table()

    def _break_in_two(s):
        # Split near the midpoint, at the closest "_" so a token doesn't get
        # cut mid-word. CSS alone (max-width + word-break) turned out to fight
        # the notebook's own table-layout rules and wrapped down to a
        # few characters a line, so force exactly one break instead.
        s = str(s)
        if len(s) <= 40:
            return s
        mid = len(s) // 2
        cuts = [i for i, ch in enumerate(s) if ch == "_"]
        brk = min(cuts, key=lambda i: abs(i - mid)) + 1 if cuts else mid
        return s[:brk] + "<br>" + s[brk:]

    def _render_table(empty_msg="No runs to show."):
        with table_out:
            clear_output(wait=True)
            r = state["ranked"]
            if r is None or len(r) == 0:
                print(empty_msg)
                return
            show = r.drop(columns=[c for c in ("refs",) if c in r.columns])
            # No max_colwidth cap: `full` shows the whole run name (up through
            # its iteration count), not an ellipsized stub.
            with pd.option_context("display.max_columns", None, "display.width", 200,
                                   "display.max_colwidth", None):
                if "full" in show.columns:
                    show = show.copy()
                    show["full"] = show["full"].map(_break_in_two)
                styler = show.style
                # *_iter / last_iter are always whole iteration numbers, but a
                # NaN anywhere in the column (a metric a run's log lacks)
                # upcasts it to float64 — format those as ints, not "150.000000".
                iter_cols = [c for c in show.columns if c.endswith("_iter") or c == "last_iter"]
                if iter_cols:
                    styler = styler.format(
                        {c: (lambda v: "" if pd.isna(v) else str(int(v))) for c in iter_cols})
                display(styler)

    def _plot(_btn=None):
        with plot_out:
            clear_output(wait=True)
            r = state["ranked"]
            if r is None or len(r) == 0:
                print("Nothing ranked to plot.")
                return
            fig, ax = plt.subplots(figsize=(10, 5), constrained_layout=True)
            try:
                plot_top(r, _metric_keys_selected(), nplot_box.value, ax=ax)
            except FileNotFoundError as e:
                plt.close(fig)
                state["fig"] = None
                print(f"Log file missing:\n{e}")
                return
            state["fig"] = fig
            plt.show()

    def _save(_btn=None):
        fig = state["fig"]
        if fig is None:
            status.value = "<b>Nothing to save</b> — plot the top runs first."
            return
        out = Path(save_name.value).expanduser()
        if not out.suffix:
            out = out.with_suffix(".png")
        try:
            fig.savefig(out, dpi=150, bbox_inches="tight")
        except Exception as e:  # pragma: no cover - filesystem/IO errors
            status.value = f"<b>Save failed:</b> {e}"
            return
        status.value = f"Saved to <code>{out.resolve()}</code>"

    for c in dataset_chk.values():
        c.observe(_recompute, "value")
    stitch_chk.observe(_recompute, "value")
    for dd in facet_dd.values():
        dd.observe(_rerank, "value")
    # metric checkboxes wire their own _on_metric_check observer as they're
    # (re)created in _populate_metric_dd — nothing to attach on metric_dd itself.
    thresh_box.observe(_rerank, "value")
    after_box.observe(_rerank, "value")
    top_box.observe(_rerank, "value")
    refresh_btn.on_click(_recompute)
    plot_btn.on_click(_plot)
    save_btn.on_click(_save)

    _recompute()

    datasets_box = widgets.VBox([
        widgets.HTML("<b>Datasets</b> &nbsp;<small>(tick to include / rank across dirs)</small>"),
        widgets.HBox(list(dataset_chk.values()), layout=widgets.Layout(flex_flow="row wrap")),
    ])
    filters_box = widgets.HBox(list(facet_dd.values()),
                               layout=widgets.Layout(flex_flow="row wrap"))
    controls = widgets.HBox([metric_dd, top_box, stitch_chk, after_box, refresh_btn],
                            layout=widgets.Layout(flex_flow="row wrap"))
    plot_row = widgets.HBox([nplot_box, plot_btn, save_name, save_btn],
                            layout=widgets.Layout(flex_flow="row wrap"))
    ui = widgets.VBox([datasets_box, filters_box, controls, thresh_box, status,
                       table_out, plot_row, plot_out])
    ui.get_figure = lambda: state["fig"]
    ui.get_ranking = lambda: state["ranked"]
    display(ui)
    return ui


def unbiased_eval(word_list, role="nsubj", name_dict=None):
    if not name_dict:
        raise ValueError("Please provide a name_dict containing your models.")

    # Dynamically extract names and setup options
    names = list(name_dict.keys())
    num_options = len(names)

    # Initialize scores dynamically
    scores = {name: 0 for name in names}
    data_records = []

    print("=== Starting Unbiased Evaluation ===")
    print(f"For each word, type a number from '1' to '{num_options}' to vote for that option.")
    print("Type '0' if it's a tie, or 'q' to quit early and see the scores.\n")

    # Generate a list of valid inputs (e.g., ['1', '2', '3', '0', 'q'])
    valid_choices = [str(i) for i in range(1, num_options + 1)] + ['0', 'q']

    for word in word_list:
        try:
            print(f"\n--- Word: '{word}' (Role: {role}) ---")

            # Fetch results for all methods and store them
            results = []
            raw_outputs = {}
            for name, t in name_dict.items():
                output = t.get_most_similar_elements(word, role=role)
                results.append((name, output))
                raw_outputs[name] = output

            # Shuffle the results to blind the test
            random.shuffle(results)

            # Display the blinded options dynamically
            for i, (name, output) in enumerate(results, start=1):
                print(f"Option {i}:\n{output}\n")

            # Prompt for user choice
            while True:
                choice = input(f"Your choice ({', '.join(valid_choices)}): ").strip().lower()

                if choice in valid_choices:
                    break
                print(f"Invalid input. Please enter one of: {', '.join(valid_choices)}")

            # Process the choice
            if choice == 'q':
                print("\nExiting early...")
                break

            winner = "tie"
            if choice == '0':
                print("Tie recorded (no points awarded).")
            else:
                # Map the user's numeric choice back to the correct winner
                choice_idx = int(choice) - 1
                winner = results[choice_idx][0]
                scores[winner] += 1
                print(f"Vote recorded!")

            # Save the round's data
            record = {
                "word": word,
                "winner": winner
            }
            # Dynamically add the raw outputs for every model tested
            record.update(raw_outputs)
            data_records.append(record)

        except Exception as e:
            print(f"Error evaluating '{word}': {e}")

    # Reveal the final scores and identities
    print("\n" + "=" * 30)
    print("=== Final Reveal & Scores ===")
    print("=" * 30)
    for name, score in scores.items():
         print(f"{name}: {score} points")

    # Determine the winner (handling multi-way ties)
    max_score = max(scores.values())
    winners = [name for name, score in scores.items() if score == max_score]

    if max_score == 0:
        print("\nResult: No points were awarded.")
    elif len(winners) == 1:
        print(f"\nWinner: {winners[0]}")
    else:
        print(f"\nResult: It's a tie between {', '.join(winners)}!")

    # Convert the records into a DataFrame and return it
    df = pd.DataFrame(data_records)
    return df

