"""Evaluate saved decompositions with the LLM-as-judge dimension-consistency metric.

Two layers, mirroring :mod:`tensormet.experimental.inspect_tucker`:

* **evaluation** -- :func:`evaluate_run` loads one run's decomposition from disk
  and scores it with a :class:`~tensormet.judge.DimConsistencyJudge`;
  :func:`evaluate_many` batches that over several runs and returns a summary
  DataFrame (headless / scripting use).
* **UI**         -- :func:`make_judge_browser` reuses the run discovery + faceted
  filtering of ``inspect_tucker`` but replaces the log plots with on-demand judge
  evaluation: pick any number of runs, hit *Evaluate*, and browse a ranked summary
  table, a score bar chart, and a per-dimension detail view (top words, injected
  outlier, the judge's pick) for each evaluated model (requires ``ipywidgets``).

Evaluation is deliberately button-triggered, never observer-triggered: each run
means loading a decomposition from disk plus a batched judge sweep over every
dimension. The judge model itself (~1 GB fp16 on GPU for the default 0.5B model)
is loaded lazily on the first *Evaluate* click and reused across clicks; the
decompositions are loaded on CPU, so the judge is the only extra GPU tenant.
Results are cached in-session per (run, judge settings), so re-selecting an
already-scored run is free.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from tensormet.judge import DimConsistencyJudge, DEFAULT_JUDGE_MODEL
from tensormet.experimental.inspect_tucker import (
    discover_datasets,
    discover_runs,
    _FACETS,
    _chain_key,
    _sortkey,
)
from tensormet.experimental.parse_master_metaphor_list import load_concepts as load_mml_concepts
from tensormet.utils import DATA_DIR, select_gpu


# === evaluation =========================================================

def _load_decomposition(insp, map_location="cpu", tier1=False):
    """Load ``insp``'s saved decomposition, dispatching on its decomposition family.

    ``insp`` is an :class:`~tensormet.config.InspectionConfig` (or the same
    duck-typed stand-in discovery attaches a ``decomposition`` attribute to —
    see ``inspect_tucker._discover_one``). ``"cp"`` loads a
    :class:`~tensormet.experimental.CP.cp_decomposition.CPDecomposition`;
    anything else (including plain InspectionConfig instances that predate the
    CP feature and so lack the attribute) falls back to
    :meth:`InspectionConfig.load_tucker`.
    """
    if getattr(insp, "decomposition", "tucker") == "cp":
        from tensormet.experimental.CP.cp_decomposition import CPDecomposition
        return CPDecomposition.load_from_disk(
            dataset=insp.dataset, method=insp.method, divergence=insp.divergence,
            dims=insp.dim, rank=insp.rank, order=insp.order, iterations=insp.iters,
            shared_factors=insp.shared_factors, map_location=map_location,
            name=insp.name, tier1=tier1, subsample_frac=insp.subsample_frac,
            max_nnz=getattr(insp, "max_nnz", None),
        )
    return insp.load_tucker(map_location=map_location, tier1=tier1)


def evaluate_run(ref, judge, *, use_latest_checkpoint=False, seed=1, role=None):
    """Load one run's decomposition from disk and judge-score it.

    ``ref`` is a :class:`~tensormet.experimental.inspect_tucker.RunRef` (as found
    by :func:`discover_runs`) or anything exposing ``.insp`` (an
    :class:`~tensormet.config.InspectionConfig`). The decomposition is loaded on
    CPU — only the judge occupies GPU memory during scoring. Dispatches to
    :class:`CPDecomposition` for CP runs, Tucker otherwise (see
    :func:`_load_decomposition`).

    By default the saved model file is scored (the best-semantic model written
    during training). With ``use_latest_checkpoint`` the latest checkpoint from
    the run's ``*_checkpoints/`` dir is scored instead (the newest state, which
    may differ from the semantic best).

    Returns the judge's score dict including per-dimension ``details``.
    """
    tucker = _load_decomposition(ref.insp)
    if use_latest_checkpoint:
        tucker.update_from_path()
    return judge.score(tucker, seed=seed, role=role, return_details=True)


def evaluate_checkpoints(ref, judge, *, seed=1, role=None, progress=print):
    """Judge-score every checkpoint of one run, in iteration order.

    Loads the run's decomposition once, then swaps in each ``{iteration}.pt``
    from its sibling ``*_checkpoints/`` directory and scores it, so the judge's
    view of training progress can be plotted over iterations.

    Returns a list of score dicts (one per checkpoint, ascending iteration),
    each with an added ``"iteration"`` key and per-dimension ``details``.
    """
    tucker = _load_decomposition(ref.insp)
    ckpt_dir = tucker.decomp_path.parent / f"{tucker.decomp_path.stem}_checkpoints"
    its = sorted(int(p.stem) for p in ckpt_dir.glob("*.pt") if p.stem.isdigit())
    if not its:
        raise FileNotFoundError(f"No checkpoints found in {ckpt_dir}")
    history = []
    for it in its:
        # Pass the full path: update_from_path repoints tucker.decomp_path at the
        # loaded checkpoint, so int-relative resolution would break on the 2nd hop.
        tucker.update_from_path(ckpt_dir / f"{it}.pt")
        out = judge.score(tucker, seed=seed, role=role, return_details=True)
        out["iteration"] = it
        if progress:
            progress(f"    ckpt {it}i: dim_consistency={out['dim_consistency']:.4f}")
        history.append(out)
    return history


def evaluate_chain(refs, judge, *, seed=1, role=None, progress=print):
    """Judge-score the checkpoints of a whole resume chain, concatenated.

    ``refs`` are the chain's segments in resume order (ascending ``n_iter_max``),
    as stitched by the browsers. Checkpoint iterations are absolute across
    segments, so the per-segment histories concatenate into one curve starting
    at the first segment's first checkpoint; any overlap at a resume point keeps
    the earlier segment's entry. Segments without a checkpoint directory are
    skipped with a note rather than aborting the chain.

    Returns the same list-of-score-dicts shape as :func:`evaluate_checkpoints`.
    """
    history = []
    for ref in refs:
        try:
            seg = evaluate_checkpoints(ref, judge, seed=seed, role=role,
                                       progress=progress)
        except FileNotFoundError as e:
            if progress:
                progress(f"    (segment {ref.stem}: {e} — skipped)")
            continue
        if history:
            last = history[-1]["iteration"]
            seg = [h for h in seg if h["iteration"] > last]
        history.extend(seg)
    if not history:
        raise FileNotFoundError("No checkpoints found in any segment of the chain.")
    return history


def evaluate_many(refs, *, labels=None, judge=None,
                  judge_model=DEFAULT_JUDGE_MODEL, num_dim_words=5,
                  diversity_aware=True, seed=1, use_latest_checkpoint=False):
    """Judge-score several runs and return (summary DataFrame, results dict).

    ``refs`` is a list of RunRef/InspectionConfig-bearing objects; ``labels`` an
    optional parallel list of row names (defaults to each ref's ``stem``). A
    pre-built ``judge`` may be passed to reuse an already-loaded model; otherwise
    one is created from the keyword settings.

    Runs that fail to load or score are reported in the ``error`` column instead
    of aborting the sweep. The results dict maps label -> full score dict (with
    ``details``) for the runs that succeeded.
    """
    if judge is None:
        judge = DimConsistencyJudge(model_name=judge_model,
                                    num_dim_words=num_dim_words,
                                    diversity_aware=diversity_aware,
                                    device=select_gpu())
    if labels is None:
        labels = [getattr(r, "stem", str(r)) for r in refs]

    rows, results = [], {}
    for label, ref in zip(labels, refs):
        row = {"run": label}
        try:
            out = evaluate_run(ref, judge, seed=seed,
                               use_latest_checkpoint=use_latest_checkpoint)
        except Exception as e:
            row["error"] = str(e)
            rows.append(row)
            continue
        results[label] = out
        row["dim_consistency"] = out["dim_consistency"]
        row["raw_accuracy"] = out["dim_consistency_raw"]
        if "dim_consistency_diversity" in out:
            row["diversity"] = out["dim_consistency_diversity"]
        row["n_dims"] = len(out.get("details", []))
        rows.append(row)

    df = pd.DataFrame(rows).sort_values(
        "dim_consistency", ascending=False, na_position="last"
    ).reset_index(drop=True)
    return df, results


def evaluate_similarity_run(ref, judge, *, query_words=None, seed=1, role=None, top_k=None):
    """Load one run's decomposition from disk and judge-score the consistency of
    its `get_most_similar_elements` neighbourhoods (:meth:`DimConsistencyJudge.
    score_similarity_consistency`), the nearest-neighbour analogue of
    :func:`evaluate_run`'s dimension-consistency check.

    `query_words` defaults (via the judge itself) to the cached Master Metaphor
    List concepts -- words that matter for downstream metaphor analysis, rather
    than an arbitrary vocab sample.

    Returns the judge's score dict including per-query ``details``.
    """
    tucker = _load_decomposition(ref.insp)
    return judge.score_similarity_consistency(tucker, query_words, seed=seed, role=role,
                                              top_k=top_k, return_details=True)


def evaluate_similarity_many(refs, *, labels=None, judge=None,
                             judge_model=DEFAULT_JUDGE_MODEL, query_words=None,
                             top_k=5, diversity_aware=True, seed=1):
    """Judge-score several runs' neighbourhood consistency and return (summary
    DataFrame, results dict). Mirrors :func:`evaluate_many`, but for
    :meth:`DimConsistencyJudge.score_similarity_consistency` instead of
    :meth:`DimConsistencyJudge.score`.

    ``query_words`` is resolved once (defaulting to the cached Master Metaphor
    List concepts) and reused across every run, so scores stay comparable.
    """
    if judge is None:
        judge = DimConsistencyJudge(model_name=judge_model, num_dim_words=top_k,
                                    diversity_aware=diversity_aware, device=select_gpu())
    if query_words is None:
        query_words = load_mml_concepts()
    if labels is None:
        labels = [getattr(r, "stem", str(r)) for r in refs]

    rows, results = [], {}
    for label, ref in zip(labels, refs):
        row = {"run": label}
        try:
            out = evaluate_similarity_run(ref, judge, query_words=query_words, seed=seed)
        except Exception as e:
            row["error"] = str(e)
            rows.append(row)
            continue
        results[label] = out
        row["similarity_consistency"] = out["similarity_consistency"]
        row["raw_accuracy"] = out["similarity_consistency_raw"]
        if "similarity_consistency_diversity" in out:
            row["diversity"] = out["similarity_consistency_diversity"]
        row["n_queries"] = out["n_queries"]
        rows.append(row)

    df = pd.DataFrame(rows).sort_values(
        "similarity_consistency", ascending=False, na_position="last"
    ).reset_index(drop=True)
    return df, results


# === interactive browser ================================================

# Plottable judge metrics: score-dict key -> (summary-table column, legend
# name, linestyle in the checkpoint line plot, bar color in the summary chart).
# Runs are told apart by color, metrics by linestyle — same convention as
# inspect_tucker.compare_metrics.
_PLOT_METRICS = {
    "dim_consistency":           ("dim_consistency", "consistency", "-", "steelblue"),
    "dim_consistency_raw":       ("raw_accuracy", "raw accuracy", "--", "lightsteelblue"),
    "dim_consistency_diversity": ("diversity", "diversity", ":", "darkseagreen"),
}


def make_judge_browser(dataset="fineweb-en", data_dir=DATA_DIR,
                       judge_model=DEFAULT_JUDGE_MODEL, device=None):
    """Interactive browser for judge-evaluating saved decompositions.

    Requires ``ipywidgets`` (and, for GPU scoring, a CUDA-visible device — the
    judge falls back to CPU with a warning otherwise).

    Layout mirrors :func:`~tensormet.experimental.inspect_tucker.make_run_browser`:
    dataset checkboxes pool runs across ``tensors/*/decomposition`` directories and
    the facet drop-downs narrow them. Instead of Run A/B pickers there is a
    multi-select — ctrl/cmd-click any number of runs and hit **▶ Evaluate** to
    score them with the judge. Results accumulate across clicks (cached per run +
    judge settings, so re-evaluating is free) and are shown three ways:

    * a summary table ranked by ``dim_consistency`` (with raw accuracy and the
      diversity multiplier), carrying the judge settings used per row;
    * a bar chart of the same, for eyeballing model differences;
    * a **detail** drop-down: pick any evaluated run to see one row per latent
      dimension — its top words, the injected outlier, the judge's pick and the
      verdict. Tick *only misses* to page through just the failed dimensions.

    Judge settings (model id, words per dimension, diversity awareness, seed) are
    editable between clicks; changing the model id unloads the old judge before
    loading the new one so at most one judge occupies the GPU. *Unload judge*
    frees it explicitly (~1 GB). Ticking *checkpoints* scores every checkpoint in
    each run's ``*_checkpoints/`` directory instead of just the saved model, and
    the bar chart is replaced by a score-vs-iteration line plot (the summary
    table and detail view then show the last checkpoint).

    The *plot* checkboxes choose which judge metrics appear in both charts:
    the final ``consistency`` score, the ``raw accuracy`` before diversity
    rescaling, and the ``diversity`` multiplier itself. In the checkpoint line
    plot runs are told apart by color and metrics by linestyle (as in
    :func:`~tensormet.experimental.inspect_tucker.compare_metrics`); metrics a
    result doesn't carry (diversity on a non-diversity-aware judge) are skipped.

    Runs resumed to a higher ``n_iter_max`` are auto-detected exactly as in
    :func:`~tensormet.experimental.inspect_tucker.make_run_browser`: with
    *stitch resume chains* ticked (the default), a checkpoint sweep of a run
    also sweeps its earlier segments' checkpoint directories, so the curve
    starts at the chain's first checkpoint rather than where the resume began.
    Chained runs are marked ``⛓×N`` in the picker. Stitching only affects
    checkpoint sweeps — the plain (saved-model) score is that of the selected
    segment either way.

    The summary table can be exported with *Save CSV*; the full result dicts
    (including per-dimension details) are available as ``browser.get_results()``.

    Returns the displayed ``VBox``.
    """
    try:
        import ipywidgets as widgets
        from IPython.display import clear_output, display
    except ImportError as e:  # pragma: no cover - UI-only dependency
        raise ImportError(
            "make_judge_browser needs ipywidgets (and an IPython/Jupyter kernel). "
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
            f"No datasets with decomposition snapshots found under {Path(data_dir) / 'tensors'}."
        )

    plt.close("all")

    # results maps result-key -> {"rec", "label", "settings", "out"}; the key folds
    # in the judge settings so the same run scored under different settings gets
    # separate (comparable) rows instead of overwriting.
    state = {"records": [], "results": {}, "judge": None, "mute": False,
             "device": None, "chains": {}}

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
    runs_sel = widgets.SelectMultiple(description="Runs", rows=8,
                                      layout=widgets.Layout(width="98%"),
                                      style={"description_width": "55px"})

    model_box = widgets.Text(value=judge_model, description="judge",
                             style={"description_width": "55px"},
                             layout=widgets.Layout(width="340px"))
    words_box = widgets.BoundedIntText(value=5, min=2, max=50, description="words/dim",
                                       style={"description_width": "75px"},
                                       layout=widgets.Layout(width="160px"))
    div_chk = widgets.Checkbox(value=True, description="diversity aware", indent=False)
    seed_box = widgets.IntText(value=1, description="seed",
                               style={"description_width": "40px"},
                               layout=widgets.Layout(width="130px"))
    ckpt_chk = widgets.Checkbox(value=False, description="checkpoints", indent=False,
                                tooltip="Score every checkpoint in the run's "
                                        "*_checkpoints/ dir and plot scores over "
                                        "iterations instead of the bar chart")
    stitch_chk = widgets.Checkbox(value=True, description="stitch resume chains",
                                  indent=False,
                                  tooltip="Checkpoint sweeps also cover the earlier "
                                          "segments of a resumed run, so curves "
                                          "start at the chain's first checkpoint")
    # Which judge metrics the charts show; raw accuracy off by default (it only
    # differs from consistency by the diversity multiplier, shown separately).
    metric_chk = {
        key: widgets.Checkbox(value=(key != "dim_consistency_raw"),
                              description=name, indent=False)
        for key, (_col, name, _ls, _bc) in _PLOT_METRICS.items()
    }

    eval_btn = widgets.Button(description="▶ Evaluate", button_style="primary",
                              layout=widgets.Layout(width="120px"))
    unload_btn = widgets.Button(description="Unload judge",
                                layout=widgets.Layout(width="120px"))
    clear_btn = widgets.Button(description="Clear results",
                               layout=widgets.Layout(width="120px"))
    refresh_btn = widgets.Button(description="↻ Refresh", button_style="info",
                                 layout=widgets.Layout(width="110px"))
    save_name = widgets.Text(value="judge_scores.csv", description="save as",
                             style={"description_width": "55px"},
                             layout=widgets.Layout(width="260px"))
    save_btn = widgets.Button(description="💾 Save CSV", button_style="success",
                              layout=widgets.Layout(width="110px"))
    status = widgets.HTML()

    detail_dd = widgets.Dropdown(description="detail", options=[("(none)", None)],
                                 value=None, layout=widgets.Layout(width="70%"),
                                 style={"description_width": "55px"})
    misses_chk = widgets.Checkbox(value=False, description="only misses", indent=False)

    log_out = widgets.Output(layout=widgets.Layout(
        max_height="220px", overflow="auto", border="1px solid #ddd"))
    results_out = widgets.Output()
    detail_out = widgets.Output()

    def _label(rec, n_seg=1):
        chain = f"  ⛓×{n_seg}" if n_seg > 1 else ""
        decomp = "" if rec["decomposition"] == "tucker" else f'[{rec["decomposition"].upper()}] '
        mn = f' mn{rec["max_nnz"]}' if rec.get("max_nnz") else ""
        return (f'{decomp}[{rec["dataset"]}] {rec["name"]} | {rec["divergence"]}/{rec["method"]} | '
                f'{rec["dim"]}d r{rec["rank"]} ss{rec["subsample_frac"]}{mn} {rec["iters"]}i'
                f'{chain}')

    def _selected_datasets():
        return {ds for ds, c in dataset_chk.items() if c.value}

    def _dataset_recs():
        sel = _selected_datasets()
        return [r for r in state["records"] if r["dataset"] in sel]

    def _filtered():
        recs = _dataset_recs()
        for key, dd in facet_dd.items():
            if dd.value != "(any)":
                recs = [r for r in recs if r[key] == dd.value]
        return recs

    def _populate_facets():
        recs = _dataset_recs()
        for key, dd in facet_dd.items():
            vals = sorted({r[key] for r in recs}, key=_sortkey)
            cur = dd.value
            dd.options = ["(any)"] + vals
            dd.value = cur if cur in dd.options else "(any)"

    def _chain_members(rec):
        """Earlier segments of rec's resume chain plus rec itself (resume order)."""
        members = state["chains"].get(_chain_key(rec), [rec])
        return [m for m in members if m["iters"] <= rec["iters"]]

    def _rebuild_runs():
        recs = _filtered()
        # Chains are indexed over the dataset-filtered records (not the facet-
        # filtered ones) so facet filters never truncate a chain — same rule as
        # make_run_browser's _chain_index.
        chains = {}
        for r in _dataset_recs():
            chains.setdefault(_chain_key(r), []).append(r)
        for members in chains.values():
            members.sort(key=lambda m: m["iters"])
        state["chains"] = chains
        cur = set(runs_sel.value)
        n_seg = (lambda r: len(_chain_members(r))) if stitch_chk.value else (lambda r: 1)
        runs_sel.options = [(_label(r, n_seg(r)), r["stem"] + "@" + r["dataset"])
                            for r in recs]
        # SelectMultiple values must be hashable, so options carry an id string and
        # the record itself is looked up on evaluate.
        state["by_id"] = {r["stem"] + "@" + r["dataset"]: r for r in recs}
        valid = {v for _, v in runs_sel.options}
        runs_sel.value = tuple(v for v in cur if v in valid)
        status.value = (f"<b>{len(recs)}</b> run(s) match the filters &nbsp;|&nbsp; "
                        f"{len(state['records'])} total &nbsp;|&nbsp; "
                        f"{len(state['results'])} result(s) cached")

    def _settings():
        return {"judge_model": model_box.value.strip(),
                "words": int(words_box.value),
                "diversity": bool(div_chk.value),
                "seed": int(seed_box.value),
                "checkpoint": bool(ckpt_chk.value),
                # Stitching only changes what a checkpoint sweep covers, so it is
                # normalized to False otherwise — plain results never re-key.
                "stitch": bool(stitch_chk.value and ckpt_chk.value)}

    def _result_key(rec, settings):
        return (rec["dataset"], rec["stem"], settings["judge_model"],
                settings["words"], settings["diversity"], settings["seed"],
                settings["checkpoint"], settings["stitch"])

    def _ensure_judge(settings):
        j = state["judge"]
        if j is not None and j.model_name != settings["judge_model"]:
            if j.loaded:
                print(f"Unloading previous judge {j.model_name!r}...")
                j.unload()
            j = None
        if j is None:
            if state["device"] is None:
                # Pick the least-used GPU before torch initialises CUDA:
                # select_gpu sets CUDA_VISIBLE_DEVICES, so the judge lands on it
                # as logical device 0. Only effective once per process — cache it.
                state["device"] = select_gpu()
            j = DimConsistencyJudge(model_name=settings["judge_model"],
                                    device=state["device"])
            state["judge"] = j
        # words/diversity only steer task construction, not the weights — mutate
        # in place so changing them never triggers a model reload.
        j.num_dim_words = settings["words"]
        j.diversity_aware = settings["diversity"]
        return j

    def _render_results():
        with results_out:
            clear_output(wait=True)
            if not state["results"]:
                print("No results yet — select runs and hit ▶ Evaluate.")
                return
            rows = []
            for res in state["results"].values():
                out, s = res["out"], res["settings"]
                rows.append({
                    "run": res["label"],
                    "dim_consistency": out["dim_consistency"],
                    "raw_accuracy": out["dim_consistency_raw"],
                    "diversity": out.get("dim_consistency_diversity", np.nan),
                    "n_dims": len(out.get("details", [])),
                    "words/dim": s["words"],
                    "seed": s["seed"],
                    "ckpts": len(res["history"]) if res.get("history") else np.nan,
                    "judge": s["judge_model"].rsplit("/", 1)[-1],
                })
            df = pd.DataFrame(rows).sort_values(
                "dim_consistency", ascending=False).reset_index(drop=True)
            state["summary_df"] = df
            with pd.option_context("display.max_rows", None,
                                   "display.max_colwidth", None,
                                   "display.width", None):
                display(df.round(4))

            # Checkpoint-swept results get a score-vs-iteration line plot; plain
            # (single-model) results keep the ranked bar chart. Both can coexist.
            # The metric checkboxes pick which curves/bars appear in either.
            hist_res = [r for r in state["results"].values() if r.get("history")]
            flat = df[df["ckpts"].isna()] if hist_res else df
            sel = [k for k in _PLOT_METRICS if metric_chk[k].value] or ["dim_consistency"]

            if hist_res:
                fig, ax = plt.subplots(figsize=(9, 4.5), constrained_layout=True)
                for res in hist_res:
                    its = [h["iteration"] for h in res["history"]]
                    color = None  # run color: set by its first plotted metric
                    for key in sel:
                        _col, mname, ls, _bc = _PLOT_METRICS[key]
                        vals = np.array([h.get(key, np.nan) for h in res["history"]],
                                        dtype=float)
                        if not np.isfinite(vals).any():
                            continue  # e.g. diversity on a non-diversity-aware judge
                        style = ({"marker": "o"} if color is None
                                 else {"marker": ".", "alpha": 0.6})
                        line, = ax.plot(its, vals, linestyle=ls, color=color,
                                        label=f'{res["label"]} · {mname}', **style)
                        color = line.get_color()
                ax.set_xlabel("iteration")
                ax.set_ylabel("score")
                ax.set_ylim(0, 1)
                ax.grid(True, alpha=0.4)
                ax.legend(fontsize=8, loc="best")
                plt.show()

            if len(flat):
                # Only metrics at least one row actually carries get a bar group.
                bars = [(key, *_PLOT_METRICS[key]) for key in sel
                        if flat[_PLOT_METRICS[key][0]].notna().any()]
                if bars:
                    n = len(bars)
                    bh = 0.8 / n
                    fig, ax = plt.subplots(
                        figsize=(9, 0.3 * n * len(flat) + 1.2), constrained_layout=True)
                    y = np.arange(len(flat))[::-1]  # best on top
                    for j, (_key, col, mname, _ls, bc) in enumerate(bars):
                        off = ((n - 1) / 2 - j) * bh
                        ax.barh(y + off, flat[col], height=bh, color=bc, label=mname)
                    ax.set_yticks(y)
                    ax.set_yticklabels(flat["run"], fontsize=8)
                    ax.set_xlim(0, 1)
                    ax.set_xlabel("score")
                    ax.grid(True, axis="x", alpha=0.4)
                    ax.legend(loc="lower right")
                    plt.show()

    def _rebuild_detail_dd():
        cur = detail_dd.value
        opts = [("(none)", None)] + [
            (f'{res["label"]}  (w{res["settings"]["words"]} s{res["settings"]["seed"]}'
             f'{" last ckpt" if res["settings"]["checkpoint"] else ""})', key)
            for key, res in state["results"].items()
        ]
        detail_dd.options = opts
        vals = [v for _, v in opts]
        detail_dd.value = cur if cur in vals else None

    def _render_detail(_change=None):
        with detail_out:
            clear_output(wait=True)
            key = detail_dd.value
            if key is None:
                print("Pick an evaluated run above to inspect its dimensions.")
                return
            res = state["results"][key]
            details = res["out"].get("details", [])
            if misses_chk.value:
                details = [d for d in details if not d["correct"]]
            if not details:
                print("Nothing to show" + (" — no misses. 🎉" if misses_chk.value else "."))
                return
            df = pd.DataFrame([{
                "dim": d["dim"],
                "verdict": "✓" if d["correct"] else "✗",
                "outlier": d["outlier"],
                "judge picked": d["predicted"],
                "top words": ", ".join(d["words"]),
            } for d in details])
            n_ok = sum(d["correct"] for d in res["out"]["details"])
            n_all = len(res["out"]["details"])
            print(f'{res["label"]} — {n_ok}/{n_all} dimensions correct')
            with pd.option_context("display.max_rows", None,
                                   "display.max_colwidth", None,
                                   "display.width", None):
                display(df)

    def _evaluate(_btn=None):
        ids = list(runs_sel.value)
        if not ids:
            status.value = "<b>Nothing selected</b> — pick one or more runs first."
            return
        settings = _settings()
        eval_btn.disabled = True
        try:
            with log_out:
                judge = _ensure_judge(settings)
                for i, rid in enumerate(ids, start=1):
                    rec = state["by_id"].get(rid)
                    if rec is None:
                        continue
                    key = _result_key(rec, settings)
                    segs = ([m["ref"] for m in _chain_members(rec)]
                            if settings["stitch"] else [rec["ref"]])
                    label = _label(rec, len(segs))
                    if key in state["results"]:
                        print(f"[{i}/{len(ids)}] cached: {label}")
                        continue
                    print(f"[{i}/{len(ids)}] loading {label} ...")
                    try:
                        if settings["checkpoint"]:
                            history = evaluate_chain(
                                segs, judge, seed=settings["seed"])
                            # "out" is the final checkpoint so the summary table
                            # and detail view keep working on checkpoint results.
                            out, hist = history[-1], history
                        else:
                            out, hist = evaluate_run(
                                rec["ref"], judge, seed=settings["seed"]), None
                    except Exception as e:
                        print(f"    FAILED: {e}")
                        continue
                    state["results"][key] = {
                        "rec": rec, "label": label, "settings": settings,
                        "out": out, "history": hist,
                    }
                    print(f"    dim_consistency={out['dim_consistency']:.4f} "
                          f"(raw={out['dim_consistency_raw']:.4f})")
        finally:
            eval_btn.disabled = False
        _render_results()
        _rebuild_detail_dd()
        _rebuild_runs()  # refreshes the cached-results count in the status line

    def _unload(_btn=None):
        j = state["judge"]
        if j is not None and j.loaded:
            j.unload()
            status.value = "Judge model unloaded (GPU memory freed)."
        else:
            status.value = "No judge model loaded."

    def _clear(_btn=None):
        state["results"] = {}
        state["summary_df"] = None
        _render_results()
        _rebuild_detail_dd()
        _rebuild_runs()

    def _save(_btn=None):
        df = state.get("summary_df")
        if df is None or df.empty:
            status.value = "<b>Nothing to save</b> — no results yet."
            return
        out = Path(save_name.value).expanduser()
        if not out.suffix:
            out = out.with_suffix(".csv")
        try:
            df.to_csv(out, index=False)
        except Exception as e:  # pragma: no cover - filesystem/IO errors
            status.value = f"<b>Save failed:</b> {e}"
            return
        status.value = f"Saved to <code>{out.resolve()}</code>"

    def _refresh(_btn=None):
        state["mute"] = True
        try:
            state["records"] = discover_runs(list(dataset_chk), data_dir)
            _populate_facets()
            _rebuild_runs()
        finally:
            state["mute"] = False

    def _on_filter(_change):
        if state["mute"]:
            return
        state["mute"] = True
        try:
            _rebuild_runs()
        finally:
            state["mute"] = False

    def _on_dataset(_change):
        if state["mute"]:
            return
        state["mute"] = True
        try:
            _populate_facets()
            _rebuild_runs()
        finally:
            state["mute"] = False

    for c in dataset_chk.values():
        c.observe(_on_dataset, "value")
    for dd in facet_dd.values():
        dd.observe(_on_filter, "value")
    detail_dd.observe(_render_detail, "value")
    misses_chk.observe(_render_detail, "value")
    # Toggling stitch changes the run labels (⛓ markers) — rebuild the picker.
    # Cached results are untouched: stitch is part of the result key, so a
    # stitched and an unstitched sweep of the same run coexist.
    stitch_chk.observe(_on_filter, "value")
    # Metric checkboxes only re-render the existing results — no evaluation.
    for c in metric_chk.values():
        c.observe(lambda _change: _render_results(), "value")
    eval_btn.on_click(_evaluate)
    unload_btn.on_click(_unload)
    clear_btn.on_click(_clear)
    refresh_btn.on_click(_refresh)
    save_btn.on_click(_save)

    _refresh()
    _render_results()
    _render_detail()

    datasets_box = widgets.VBox([
        widgets.HTML("<b>Datasets</b> &nbsp;<small>(tick to include / compare across dirs)</small>"),
        widgets.HBox(list(dataset_chk.values()), layout=widgets.Layout(flex_flow="row wrap")),
    ])
    filters = widgets.HBox(list(facet_dd.values()), layout=widgets.Layout(flex_flow="row wrap"))
    judge_row = widgets.HBox([model_box, words_box, div_chk, seed_box, ckpt_chk,
                              stitch_chk],
                             layout=widgets.Layout(flex_flow="row wrap"))
    action_row = widgets.HBox([eval_btn, unload_btn, clear_btn, refresh_btn, save_name, save_btn],
                              layout=widgets.Layout(flex_flow="row wrap"))
    plot_row = widgets.HBox([widgets.HTML("<b>plot:</b>&nbsp;"), *metric_chk.values()],
                            layout=widgets.Layout(flex_flow="row wrap"))
    detail_row = widgets.HBox([detail_dd, misses_chk])
    ui = widgets.VBox([datasets_box, filters, status, runs_sel, judge_row, action_row,
                       log_out, plot_row, results_out, detail_row, detail_out])
    # Programmatic access to everything the UI computed: full score dicts (with
    # per-dimension details), the current summary frame, and the live judge.
    ui.get_results = lambda: state["results"]
    ui.get_summary = lambda: state.get("summary_df")
    ui.get_judge = lambda: state["judge"]
    display(ui)
    return ui
