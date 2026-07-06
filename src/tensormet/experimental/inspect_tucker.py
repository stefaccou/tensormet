"""Inspect and compare decomposition runs from their on-disk logs.

Three layers, smallest to largest:

* **loading**  -- :func:`load_metrics` / :func:`load_vocab` parse a run's log
  into (iters, rec_error, sem_dicts).
* **plotting** -- :func:`plot_metrics` (one run) and :func:`compare_metrics`
  (any number of runs overlaid) turn those into matplotlib figures.
* **UI**       -- :func:`make_run_browser` scans the ``decomposition/`` directory
  of one or more datasets (:func:`discover_datasets` finds them) for
  ``*_config.json`` snapshots and offers dataset checkboxes plus faceted
  drop-downs to pick and compare any two runs — even across datasets —
  interactively (requires ``ipywidgets``).

The plotting/comparison functions duck-type on their config argument: they only
touch ``.log_path`` / ``.stem`` / ``.vocab_path``, so both :class:`InspectionConfig`
and the lightweight :class:`RunRef` (reconstructed from a config snapshot, works
for legacy-named runs too) are accepted interchangeably.
"""

from __future__ import annotations

import datetime as _dt
import json
import pickle
import re
from dataclasses import dataclass
from pathlib import Path
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from tensormet.config import InspectionConfig
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


def load_metrics(*cfgs):
    """Parse one or more run logs into (iters, rec_error, sem_dicts).

    A single config yields its own three lists; several configs are
    concatenated (useful for resumed runs split across files).
    """
    if len(cfgs) == 1:
        return _read_log(cfgs[0])

    all_iters, all_rec, all_sem = [], [], []
    for cfg in cfgs:
        iters, rec, sem = _read_log(cfg)
        all_iters.extend(iters)
        all_rec.extend(rec)
        all_sem.extend(sem)
    return all_iters, all_rec, all_sem


def load_vocab(cfg):
    """Unpickle the vocabulary associated with a run config."""
    with open(cfg.vocab_path, "rb") as f:
        return pickle.load(f)


def average_runs(configs, n_grid=500):
    """Average rec_error and semantic metrics across multiple runs.

    ``configs`` is a list of configs (or a dict whose values are configs); each
    may also be a list/tuple of resume-chain segments as accepted by
    :func:`load_metrics`. Runs whose log file does not exist are silently skipped.

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

def plot_metrics(*cfgs, sem_keys=("average_rank_score",),
                 plot_rec_error=True, title="", ax=None):
    """Plot reconstruction error and/or semantic scores for one run.

    Pass ``ax`` to draw into an existing axis (a twin axis is created
    internally for the score curves). Returns the figure.
    """
    all_its, all_rec, all_sem = load_metrics(*cfgs)

    ax1 = ax or plt.subplots()[1]
    fig = ax1.figure
    ax1.set_xlabel("Iteration")
    ax1.grid(True)

    colors = plt.cm.tab10.colors
    all_lines = []

    split_axes = (not plot_rec_error) and (len(sem_keys) == 2)

    if split_axes:
        ax2 = ax1.twinx()
        its0, vals0 = _values_for_key(sem_keys[0], all_its, all_sem)
        (l0,) = ax1.plot(its0, vals0, label=sem_keys[0], color=colors[0])
        ax1.set_ylabel(sem_keys[0])
        its1, vals1 = _values_for_key(sem_keys[1], all_its, all_sem)
        (l1,) = ax2.plot(its1, vals1, label=sem_keys[1], color=colors[1])
        ax2.set_ylabel(sem_keys[1])
        all_lines = [l0, l1]
    else:
        if plot_rec_error:
            (l,) = ax1.plot(all_its, all_rec, label="Rec error", color="red")
            ax1.set_ylabel("Reconstruction Error")
            all_lines.append(l)
        if sem_keys:
            ax2 = ax1.twinx()
            ax2.set_ylabel("Score")
            for i, key in enumerate(sem_keys):
                # Offset so the first score curve isn't solid like rec error.
                ls = _LINESTYLES[(i + (1 if plot_rec_error else 0)) % len(_LINESTYLES)]
                its_k, vals_k = _values_for_key(key, all_its, all_sem)
                (l,) = ax2.plot(its_k, vals_k, label=key, color=colors[i % len(colors)],
                                linestyle=ls)
                all_lines.append(l)

    # Park the legend outside the axes so it never sits on top of the curves.
    ax1.legend(all_lines, [l.get_label() for l in all_lines],
               loc="center left", bbox_to_anchor=(1.12, 0.5), frameon=False)
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


def _is_preloaded(run):
    """Return True if *run* is already a (iters, rec, sem) triple from average_runs."""
    return (isinstance(run, tuple) and len(run) == 3
            and isinstance(run[0], list)
            and (not run[0] or isinstance(run[0][0], int)))


def compare_metrics(configs, labels=None, sem_keys=("average_rank_score",),
                    plot_rec_error=True, title="Training Metrics Comparison",
                    ax=None, clip_common=False, color_by=None):
    """Overlay any number of runs on a shared figure.

    ``configs`` is the list of runs to plot and ``labels`` a parallel list of
    legend names. Each *run* is either a single config or a list/tuple of configs
    treated as resume-chain segments (concatenated as in :func:`load_metrics`), so
    to overlay several chains pass a list of lists. A ``dict`` may be given instead
    of ``configs``: its values become the runs and its keys the labels (an explicit
    ``labels`` still overrides the keys). When a label is missing it falls back to
    the run's ``stem``.

    ``color_by`` controls how colors are assigned. When omitted each run gets its
    own tab10 color. Pass either:

    * a **callable** ``label -> group`` — e.g. ``lambda l: l.split(" rs")[0]``
      to colour by the method name extracted from the label; or
    * a **dict** ``{label: group}`` for explicit per-label group assignment.

    Runs that map to the same group share a color; each distinct group gets a
    different tab10 color. This lets you visually separate methods while keeping
    random-state curves the same hue.

    Reconstruction error and the semantic keys are told apart by line style.
    Returns the figure.

    With ``clip_common`` set, the x-axis is capped at the last iteration shared by
    every run (the smallest of their final iterations), so short and long runs are
    compared over their common range instead of being squashed.
    """
    if isinstance(configs, dict):
        if labels is None:
            labels = list(configs.keys())
        configs = list(configs.values())

    runs = [c if _is_preloaded(c) else _as_run(c) for c in configs]
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

    if split_axes:
        # One semantic key per axis; runs separated by color, keys by axis and style.
        ax2 = ax1.twinx()
        ax1.set_ylabel(sem_keys[0])
        ax2.set_ylabel(sem_keys[1])
        for (its, _rec, sem), lbl, c in zip(loaded, labels, run_colors):
            for k_i, (key, axis) in enumerate(zip(sem_keys, (ax1, ax2))):
                ls = _LINESTYLES[k_i % len(_LINESTYLES)]
                its_k, vals_k = _values_for_key(key, its, sem)
                (l,) = axis.plot(its_k, vals_k, color=c, linestyle=ls,
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
                (l,) = ax1.plot(its, rec, color=c, linestyle=_LINESTYLES[0],
                                label=f"{lbl} · Rec error")
                all_lines.append(l)
            for k_i, key in enumerate(sem_keys):
                # Offset so the first score curve isn't solid like rec error.
                ls = _LINESTYLES[(k_i + (1 if plot_rec_error else 0)) % len(_LINESTYLES)]
                its_k, vals_k = _values_for_key(key, its, sem)
                (l,) = ax2.plot(its_k, vals_k, color=c, linestyle=ls,
                                label=f"{lbl} · {key}")
                all_lines.append(l)

    if clip_common:
        finals = [max(its) for its, _, _ in loaded if its]
        if finals:
            ax1.set_xlim(right=min(finals))

    ax1.legend(all_lines, [l.get_label() for l in all_lines],
               loc="center left", bbox_to_anchor=(1.12, 0.5), frameon=False)
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


def _ss_from_stem(stem):
    """Recover subsample_frac from a run's filename stem, or None if absent."""
    m = _SS_STEM_RE.search(stem)
    return float(m.group(1).replace("p", ".")) if m else None


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

        insp = InspectionConfig(
            dim=dim, name=exp.get("name"), dataset=exp.get("dataset", dataset),
            method=exp.get("method", "siiSoftPlus"), divergence=exp.get("divergence", "kl"),
            order=exp.get("order", 3), iters=iters, rank=rank0,
            shared_factors=sf, subsample_frac=ss,
        )
        log_path = decomp_dir / f"{stem}_log.txt"
        yield {
            "stem": stem,
            "dataset": dataset,
            "ref": RunRef(stem, log_path,
                          f"{dataset}|{name}|{insp.method}|{dim}d|r{rank0}|{iters}i", insp),
            "name": name, "divergence": insp.divergence, "method": insp.method,
            "order": insp.order, "dim": dim, "rank": rank0,
            "subsample_frac": ss, "iters": iters,
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

_FACETS = [("Name", "name"), ("Divergence", "divergence"), ("Method", "method"),
           ("Dim", "dim"), ("Rank", "rank"), ("Subsample", "subsample_frac"), ("Iters", "iters")]


def _sortkey(v):
    return (0, v, "") if isinstance(v, (int, float)) else (1, 0, str(v))


# Ordered (field -> formatter) used to build legend labels for plotted runs.
# `dataset` is first so runs pulled from different directories stay distinguishable.
_LABEL_FIELDS = [
    ("dataset", lambda i: i.dataset),
    ("name", lambda i: i.name or "(unnamed)"),
    ("method", lambda i: i.method),
    ("div", lambda i: i.divergence),
    ("dim", lambda i: f"{i.dim}d"),
    ("rank", lambda i: f"r{i.rank}"),
    ("ss", lambda i: f"ss{i.subsample_frac}"),
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
    return (rec["dataset"], rec["name"], rec["divergence"], rec["method"],
            rec["order"], rec["dim"], rec["rank"], rec["subsample_frac"],
            frozenset(insp.shared_factors or ()))


def make_run_browser(dataset="fineweb-en", data_dir=DATA_DIR,
                     default_sem_keys=("average_rank_score", "simlex_all_rho")):
    """Interactive faceted browser for picking and comparing two runs.

    Requires ``ipywidgets`` and an interactive matplotlib backend.

    Every dataset under ``tensors/`` that has decomposition snapshots gets a
    checkbox at the top; tick several to pool their runs and compare runs that
    live in *different* directories. The facet drop-downs and the Run A / Run B
    pickers always reflect only the datasets currently checked, so options never
    include runs you can't actually select. (Run B = ``(none)`` plots one run.)

    ``dataset`` sets which boxes start checked — a single name, a list of names,
    or ``None`` to check every discovered dataset. Hit *Refresh* after new runs
    land.

    The *after* box takes a date (e.g. ``2026-06-01``, or anything pandas can
    parse) and restricts the selectable runs to those whose config snapshot was
    written on or after it; leave it empty for no cutoff. The cutoff only narrows
    the picker — a recent resumed run is still stitched back to its earlier
    segments even if those predate the cutoff.

    Runs that were resumed to a higher ``n_iter_max`` are auto-detected and
    stitched: with *stitch resume chains* ticked (the default), selecting a run
    plots it together with its earlier segments, so the curve starts at 0 rather
    than at the iteration the resume began. Chained entries are marked ``⛓×N``.
    Untick to view a single segment (e.g. just the resumed tail) in isolation.

    When comparing two runs of unequal length, tick *clip to common iters* to cap
    the x-axis at the shorter run's final iteration (e.g. 250 vs 2000 → x stops at
    250), so the shared range is compared head-to-head rather than squashed.

    The current plot can be exported two ways: type a path in the *save as* box
    and hit *Save* to write it to disk, or grab the live Figure object via
    ``browser.get_figure()`` (returns ``None`` until the first plot is drawn).

    Returns the displayed ``VBox`` so callers can keep a reference alive.
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
    state = {"records": [], "mute": False, "fig": None}

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
    sem_box = widgets.Text(value=",".join(default_sem_keys), description="sem_keys",
                           style={"description_width": "75px"}, layout=widgets.Layout(width="55%"))
    rec_chk = widgets.Checkbox(value=False, description="plot rec error", indent=False)
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
        return (f'[{rec["dataset"]}] {rec["name"]} | {rec["divergence"]}/{rec["method"]} | '
                f'{rec["dim"]}d r{rec["rank"]} ss{rec["subsample_frac"]} '
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
            keys = tuple(k.strip() for k in sem_box.value.split(",") if k.strip())
            b = run_b.value          # tuple of RunRef, or None for "(none)"
            fig, ax = plt.subplots(figsize=(10, 5), constrained_layout=True)
            try:
                # a[-1] is the representative (latest) segment — used for titles/labels.
                if b is None:
                    plot_metrics(*a, sem_keys=keys, plot_rec_error=rec_chk.value,
                                 title=a[-1].stem, ax=ax)
                else:
                    shared, (la, lb) = _diff_labels([a[-1].insp, b[-1].insp])
                    compare_metrics([list(a), list(b)], [la, lb],
                                    sem_keys=keys, plot_rec_error=rec_chk.value,
                                    title=shared, ax=ax, clip_common=clip_chk.value)
            except FileNotFoundError as e:
                plt.close(fig)
                state["fig"] = None
                print(f"Log file missing:\n{e}")
                return
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
    run_a.observe(_on_select, "value")
    run_b.observe(_on_select, "value")
    sem_box.observe(_on_select, "value")
    rec_chk.observe(_on_select, "value")
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
    controls = widgets.HBox([sem_box, rec_chk, stitch_chk, clip_chk, after_box, refresh_btn, save_name, save_btn])
    ui = widgets.VBox([datasets_box, filters, status, run_a, run_b, controls, plot_out])
    # Expose the live state so callers can grab the current Figure out of the UI,
    # e.g. `fig = browser.get_figure(); fig.savefig(...)` or display it elsewhere.
    ui.get_figure = lambda: state["fig"]
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

