"""Overviews and comparisons of the downstream sweep's results (tensormet_eval.jsonl).

Rows are read through downstream_sweep.finished_records (same checkpoint and seed rules as the sweep):
one row per (model, variant), with config columns, the task scores,
NP folds 0 and 1 (SPINE's two), and coverage. Every task is higher-is-better.

    import compare as pc
    df = pc.load()                                          # newest sweep + newest method_baselines run
    pc.overview(df)                                         # models x tasks, rank per task
    pc.overview(df, ref='glove_100')                         # the same, as differences from GloVe
    pc.top(df)                                              # best k per task
    pc.compare(df, 'glove_100', 'tt_4g_r100_*')              # side by side, differences from the first
    pc.coverage(df, 'glove_100', 'tt_4g_r100_*')             # read before trusting a difference
    pc.wins(df, models=['glove*', '*_r100_scSoftPlus_ss0.025'])   # row beats column on how many tasks
    pc.effect(df, 'rank')                                   # models that differ only in rank
    pc.effect(df, 'rank', detail=True)                      # every such pair

Models are shell-style patterns on the run names. `tasks` is a list, or 'all', or the task set of one paper:
'polar' (Mathew et al., 2020) or 'spine' (Subramanian et al., 2018).
Tasks in `SKIP` are left out of every table, whatever `tasks` says (e.g. pc.SKIP = {'word_analogy'}).
MLP and RandomForest fits are unseeded unless the sweep had --random-state, so a difference of
about 0.01 on a classifier task can be noise: `tol` in wins/effect counts those as ties.
"""
from __future__ import annotations

import fnmatch

import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

import downstream_sweep as ds

TASKS = {
    'all': list(ds.COLUMNS),  # NP once, as the 10-fold mean; folds 0/1 only in 'spine'
    'polar': [c for c in ds.COLUMNS if c != 'wordsim353'],
    # the columns of spine_eval.results_table; NP is SPINE's folds 0 and 1, not the 10-fold mean
    'spine': ['wordsim353', 'news_computer', 'news_religion', 'news_sports', 'TREC', 'np_f0', 'np_f1'],
}
CONFIG = ['family', 'ngram', 'rank', 'method', 'ss_frac', 'dims', 'random_state', 'variant']  # what effect() varies
SKIP = set()  # tasks left out everywhere, set from the notebook
DEFAULT_BASE = {'variant': 'raw', 'family': 'tucker', **ds.BASE}  # effect()'s reference value

# Palette: one-hue blue for scores; red <-> grey <-> blue for differences (worse <-> better)
SEQ = LinearSegmentedColormap.from_list('seq', ['#cde2fb', '#86b6ef', '#3987e5', '#1c5cab'])
DIV = LinearSegmentedColormap.from_list('div', ['#e34948', '#f0efec', '#2a78d6'])


# --- Loading ------------------------------------------------------------------------------
def reached(ck):
    """How far a run got: its last checkpoint, or the model file's iteration when that is later
    (a run stopped between an improvement and the next checkpoint). None for baselines / unknown."""
    ck = ck or {}
    if ck.get('common_iteration') is not None:  # series 'same': the iteration its set was cut to
        return ck['common_iteration']
    return max((v for v in (ck.get('last_checkpoint'), ck.get('best_iteration')) if v is not None), default=None)


def _row(r, ck=None):
    """One result record as a row. A run that did not end at the table's iteration count (walltime
    stop, patience, or only a longer run on disk) is named '<model>_<iterations reached>'."""
    cfg = r.get('config') or {}
    folds = r.get('np_bracketing_folds') or [None, None]
    n = reached(ck or r.get('checkpoint'))
    name = r['model'] if n is None or cfg.get('iters') in (None, n) else f"{r['model']}_{n}"
    row = {
        'model': name, 'run': r['model'], 'reached': n, 'variant': r['variant'],
        'family': 'baseline' if 'baseline' in cfg else cfg.get('decomposition'),
        'baseline': cfg.get('baseline'),
        # baselines only: 'own' = the embedding's whole vocabulary (*_full), 'ours' = the best model's words
        'vocab': cfg.get('vocab') or ('ours' if 'baseline' in cfg else None),
        **{k: None if 'baseline' in cfg else cfg.get(k) for k in ('ngram', 'rank', 'method', 'ss_frac', 'dims')},
        # training seed; records written before it was varied are seed 1
        'random_state': None if 'baseline' in cfg or not cfg else cfg.get('random_state', 1),
        'checkpoint': r['model_path'].rsplit('/', 1)[-1],
        **{c: r['scores'].get(c) for c in ds.COLUMNS},
        'np_f0': folds[0], 'np_f1': folds[1],
        'time': r['time'],
    }
    for task, cov in (r.get('coverage') or {}).items():
        for k, v in cov.items():
            row[f'cov.{task}.{k}'] = v
    return row


def load(manifests=None, prefixes=('sweep', 'methods'), min_iters=None):
    """One row per (model, variant) that has a record, indexed by (model, variant).
    `min_iters`: leave out runs that got fewer iterations than this (see reached()); baselines stay.
    `manifests`: manifest paths or dicts; default per prefix the newest manifest with the full model
    list (a dry run counts; one still loading, failed, or started with --only does not). Pass several
    manifests to combine sweeps."""
    if manifests is None:
        manifests = []
        for prefix in prefixes:
            for path in sorted(ds.RESULTS_DIR.glob(f'{prefix}_*.json'), reverse=True):
                man = ds.load_manifest(path)
                if 'models' in man and not man.get('only'):
                    print(f'{prefix}: {path.name} ({man["status"]})')
                    manifests.append(man)
                    break
    rows = {}
    for man in manifests:
        man = man if isinstance(man, dict) else ds.load_manifest(man)
        if 'models' not in man:
            continue
        paths = {name: m['path'] for name, m in man['models'].items()}
        records = ds.finished_records(paths, man['random_state'], man.get('resume_since'))
        for run in man['runs']:
            if run in records:
                rows[run] = _row(records[run], (man['models'].get(run[0]) or {}).get('checkpoint'))
        for name, why in (man.get('missing') or {}).items():
            if why.startswith('left out'):
                print(f'  {name}: {why}')
    df = pd.DataFrame(list(rows.values()))
    if df.empty:
        raise ValueError('no records for these manifests; has the sweep finished a run?')
    df['reached'] = pd.to_numeric(df['reached']).astype('Int64')
    renamed = df.loc[df['model'] != df['run'], 'model'].unique()
    if len(renamed):
        print('  not at the table\'s iteration count, named <model>_<iterations reached>:', ', '.join(renamed))
    if min_iters is not None:
        low = (df['reached'] < min_iters).fillna(False)
        if low.any():
            print(f'  left out (fewer than {min_iters} iterations):', ', '.join(df.loc[low, 'model'].unique()))
        df = df[~low]
    for k in ('ngram', 'rank', 'dims', 'random_state'):
        df[k] = pd.to_numeric(df[k]).astype('Int64')
    df['ss_frac'] = pd.to_numeric(df['ss_frac'])
    df[TASKS['all']] = df[TASKS['all']].apply(pd.to_numeric).astype(float)
    return df.set_index(['model', 'variant'])


# --- Selection ----------------------------------------------------------------------------
def _tasks(tasks):
    if tasks is None:
        tasks = TASKS['all']
    elif isinstance(tasks, str):
        tasks = TASKS[tasks]
    return [t for t in tasks if t not in SKIP]


def _pick(df, items, variant):
    """(model, variant) keys matching `items` (patterns, 'pattern:variant' overrides `variant`), in item order."""
    keys = []
    for item in [items] if isinstance(items, str) else items:
        pat, _, v = item.partition(':')
        v = v or variant
        found = [k for k in df.index if fnmatch.fnmatchcase(k[0], pat) and (v is None or k[1] == v)]
        if not found:
            raise KeyError(f'no run matches {item!r} (variant {v}); models: {sorted(set(df.index.get_level_values(0)))}')
        keys += [k for k in found if k not in keys]
    return keys


def _with_ref(items, ref):
    items = [items] if isinstance(items, str) else list(items)
    return items if ref is None else [*items, ref]


def _labels(keys):
    """Model names, with ':variant' when the keys mix variants."""
    if len({v for _, v in keys}) == 1:
        return [m for m, _ in keys]
    return [f'{m}:{v}' for m, v in keys]


def _scores(df, keys, tasks):
    data = df.loc[keys, tasks].astype(float)
    data.index = _labels(keys)
    return data.dropna(axis=1, how='all')  # e.g. tasks the 'scaled' variant skips


# --- Styling ------------------------------------------------------------------------------
def _style_scores(data, cols, axis=0):
    """Blue per column (axis=0) or per row (axis=1), best bold."""
    return (data.style.format('{:.3f}', subset=cols, na_rep='–')
            .background_gradient(cmap=SEQ, axis=axis, subset=cols)
            .highlight_max(axis=axis, subset=cols, props='font-weight: bold'))


def _shade_deltas(sty, data, cols, axis=0):
    """Red/blue, symmetric around 0, scaled per column (axis=0) or per row (axis=1)."""
    lines = cols if axis == 0 else data.index
    for line in lines:
        vals = data[line] if axis == 0 else data.loc[line, cols]
        m = float(np.nanmax(np.abs(vals.to_numpy(dtype=float)))) if vals.notna().any() else 0.0
        if m > 0:
            subset = [line] if axis == 0 else pd.IndexSlice[[line], cols]
            sty = sty.background_gradient(cmap=DIV, vmin=-m, vmax=m, axis=None, subset=subset)
    return sty


# --- Overviews ----------------------------------------------------------------------------
def overview(df, variant='raw', tasks=None, models=None, ref=None):
    """Models x tasks, sorted by mean rank over the tasks (1 = best); '#1' counts first places.
    With `ref` (a model), cells are differences from it, red = worse, blue = better."""
    keys = _pick(df, _with_ref(models or ['*'], ref), variant)
    data = _scores(df, keys, _tasks(tasks))
    cols = list(data.columns)
    ranks = data.rank(ascending=False, method='min')
    out = data.copy()
    if ref is not None:
        ref_label = _labels(keys)[keys.index(_pick(df, [ref], variant)[0])]
        out[cols] = data - data.loc[ref_label]
    out['mean rank'] = ranks.mean(1)
    out['#1'] = (ranks == 1).sum(1)
    out = out.sort_values(['mean rank', '#1'], ascending=[True, False])

    what = f'variant {variant}' if variant else 'all variants'
    if ref is None:
        sty = _style_scores(out, cols)
        caption = f'{what}: scores, colour per task, best bold'
    else:
        sty = _shade_deltas(out.style.format('{:+.3f}', subset=cols, na_rep='–'), out, cols)
        caption = f'{what}: difference from {ref_label} (blue = better, red = worse)'
    return sty.format('{:.1f}', subset=['mean rank']).set_caption(caption + '; sorted by mean rank')


def top(df, k=5, variant='raw', tasks=None, models=None):
    """The k best models per task, with their scores."""
    data = _scores(df, _pick(df, models or ['*'], variant), _tasks(tasks))
    cols = {t: [f'{m}  {v:.3f}' for m, v in data[t].dropna().sort_values(ascending=False).head(k).items()]
            for t in data.columns}
    return pd.DataFrame({t: pd.Series(v, index=range(1, len(v) + 1)) for t, v in cols.items()})


# --- Comparisons --------------------------------------------------------------------------
def compare(df, *models, variant='raw', tasks=None, ref=None):
    """Tasks x chosen models, then each model's difference from `ref` (default: the first)."""
    keys = _pick(df, _with_ref(models, ref), variant)
    data = _scores(df, keys, _tasks(tasks)).T
    labels = list(data.columns)
    r = labels[keys.index(_pick(df, [ref], variant)[0])] if ref is not None else labels[0]
    delta = data.drop(columns=r).sub(data[r], axis=0).add_prefix('Δ ')
    out = pd.concat([data, delta], axis=1)

    sty = _style_scores(out, labels, axis=1)
    if len(delta.columns):
        sty = _shade_deltas(sty.format('{:+.3f}', subset=list(delta.columns), na_rep='–'),
                            out, list(delta.columns), axis=1)
    return sty.set_caption(f'best per task bold; Δ = model − {r} (blue = better, red = worse)')


def coverage(df, *models, variant='raw', tasks=None):
    """How much of each task the models' vocabularies cover (per model, not per variant)."""
    keys = _pick(df, models, variant)
    wanted = {t.replace('np_f0', 'np_bracketing').replace('np_f1', 'np_bracketing') for t in _tasks(tasks)}
    cols = [c for c in df.columns if c.startswith('cov.') and c.split('.')[1] in wanted]
    data = df.loc[keys, cols].T
    data.columns = _labels(keys)
    data.index = pd.MultiIndex.from_tuples([tuple(c.split('.', 2)[1:]) for c in cols], names=['task', 'measure'])
    return data.dropna(how='all').style.format('{:.3f}', na_rep='–').set_caption(
        'fractions; texts_without_known_word is the share of texts that became a zero vector')


def wins(df, variant='raw', models=None, tasks=None, tol=0.0):
    """Cell = number of tasks where the row model beats the column model by more than `tol`."""
    data = _scores(df, _pick(df, models or ['*'], variant), _tasks(tasks))
    X = data.to_numpy()
    with np.errstate(invalid='ignore'):
        W = ((X[:, None, :] - X[None, :, :]) > tol).sum(2).astype(float)
    np.fill_diagonal(W, np.nan)
    out = pd.DataFrame(W, index=data.index, columns=data.index)
    out['total'] = out.sum(1)
    order = out['total'].sort_values(ascending=False).index
    out = out.loc[order, [*order, 'total']]
    return (out.style.format('{:.0f}', na_rep='–')
            .background_gradient(cmap=SEQ, axis=None, subset=list(order))
            .set_caption(f'tasks (of {data.shape[1]}) where the row beats the column by more than {tol}; '
                         'row + column < tasks means ties or missing scores'))


def pairs(df, factor, variant='raw', tasks=None, base=None):
    """Pairs of runs equal in every CONFIG key but `factor`: the run with `factor` = `base`
    (default: DEFAULT_BASE, or the smallest value present) against each other value.
    Differences per task, other - base. Baselines have no config and never pair."""
    tasks = _tasks(tasks)
    d = df.reset_index()
    if factor != 'variant' and variant is not None:
        d = d[d['variant'] == variant]
    context = [k for k in CONFIG if k != factor]
    d = d.dropna(subset=[*context, factor])
    rows = []
    for _, g in d.groupby(context, sort=False):
        values = list(dict.fromkeys(g[factor]))
        if len(values) < 2:
            continue
        b = base if base is not None else _default_base(factor, values)
        if b not in values:
            continue
        ref = g[g[factor] == b].iloc[0]
        for _, other in g[g[factor] != b].iterrows():
            rows.append({'from': b, 'to': other[factor],
                         'model': _run_label(other, factor), 'vs': _run_label(ref, factor),
                         **{t: other[t] - ref[t] for t in tasks}})
    return pd.DataFrame(rows, columns=['from', 'to', 'model', 'vs', *tasks])


def _default_base(factor, values):
    pref = DEFAULT_BASE.get(factor)
    if pref in values:
        return pref
    try:
        return sorted(values)[0]
    except TypeError:
        return sorted(values, key=str)[0]


def _run_label(row, factor):
    return f"{row['model']}:{row['variant']}" if factor == 'variant' else row['model']


def effect(df, factor, variant='raw', tasks=None, base=None, tol=0.0, detail=False):
    """What changing `factor` does with everything else equal (see pairs()).
    Summary: per change (from -> to), the mean difference per task, the number of pairs, and how
    many (pair, task) cells got better / worse by more than `tol`. detail=True: every pair."""
    p = pairs(df, factor, variant, tasks, base)
    if p.empty:
        return f'no pairs of runs differ only in {factor!r}' + (f' (variant {variant})' if variant else '')
    cols = [t for t in p.columns[4:] if p[t].notna().any()]
    if detail:
        out = p.set_index(['from', 'to', 'model'])[['vs', *cols]]
    else:
        g = p.groupby(['from', 'to'])
        out = g[cols].mean()
        out.insert(0, 'pairs', g.size())
        cells = p[cols]
        keys = [p['from'], p['to']]
        out['better'] = (cells > tol).groupby(keys).sum().sum(1)
        out['worse'] = (cells < -tol).groupby(keys).sum().sum(1)
    sty = _shade_deltas(out.style.format('{:+.3f}', subset=cols, na_rep='–'), out, cols)
    what = f'variant {variant}' if factor != 'variant' and variant else 'all variants'
    return sty.set_caption(f'{factor}: to − from, everything else equal ({what}); blue = better, red = worse')
