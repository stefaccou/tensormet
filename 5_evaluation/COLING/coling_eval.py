"""The COLING evaluation summary: the POLAR suite and the word-intrusion judge, on our runs and the baselines.

Used by Coling_eval_summary.ipynb. Loading follows ../eval.ipynb (section 0), the matched-set tests its
section 9, extended to every setting. The judge enters as its raw accuracy only.
"""
from __future__ import annotations

import fnmatch
import itertools
import json
import re
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from scipy import stats

try:
    from IPython.display import display
except ImportError:  # outside Jupyter
    display = print

import compare as pc
import eval_utils as eu

COLING_DIR = Path(__file__).resolve().parent
ps = pc.ps
JUDGE_DIR = COLING_DIR / 'judge_results'

OUR_WORDS = 10000    # load_best()'s vocabulary: every baseline was restricted to it
TOL = 0.005          # a difference counts only beyond this (classifier noise, GUIDE.md section 7)
ALPHA = 0.05
TEA = ['judge_acc']  # the judge's score columns: its accuracy on word intrusion (diversity: intrusion_table only)
AVG = 'POLAR_average'  # mean over tasks(); NaN if a task is missing
OURS = ('tt', 'tucker')
IGNORE = ('glove_nmf_*',)  # rows load() discards entirely (not listed in .dropped)
# A matched set holds these equal, except the factor
CONFIG = ['family', 'ngram', 'rank', 'method', 'ss_frac', 'dims', 'tt_rank', 'random_state']
ORDERED = ('ngram', 'rank', 'dims', 'ss_frac', 'tt_rank')
LEVELS = {'family': ['tucker', 'tt'], 'method': ['countingLog', 'countingLogEps', 'scSoftPlus']}
# (from, to) per unordered factor, Δ = to − from; default: every level against the first
CONTRASTS = {'family': [('tucker', 'tt')],
             'method': [('countingLog', 'scSoftPlus'), ('countingLogEps', 'scSoftPlus'),
                        ('countingLog', 'countingLogEps')]}
# Expected direction per factor and suite ('POLAR', 'judge acc'): +1 the later level scores higher, −1 lower;
# the tests of that suite are then one-sided. Unlisted: two-sided. Empty: no direction was fixed before the
# results were seen (eval.ipynb section 9 showed rank first), so every test is two-sided.
EXPECTED = {}
CURVE_KEYS = ('rec_error',)  # main text; the training-time checks are in appendix B (TRAINING_KEYS)
TRAINING_KEYS = ('dim_consistency_raw', 'simlex_all_rho')  # training's judge (Qwen3.5-2B, k = 5) and SimLex

# What each baseline is, and its training data (README.md, "Pretrained embedding baselines"); first match wins
SOURCES = [('glove_nmf_*', 'dense, NMF of GloVe', 'Wiki+Gigaword 2024'),
           ('glove_*', 'dense', 'Wiki+Gigaword 2024'),
           ('w2v*', 'dense', 'Google News'),
           ('polar_glove_*', 'interpretable: POLAR', 'Wiki+Gigaword 2024 (GloVe)'),
           ('polar_w2v_*', 'interpretable: POLAR', 'Google News (word2vec)'),
           ('spine_*', 'interpretable: SPINE', 'released vectors'),
           ('spowv_*', 'interpretable: SPOWV', 'released vectors'),
           ('nnse_*', 'interpretable: NNSE', 'ClueWeb09'),
           ('sinr_*', 'interpretable: SINr', 'BNC, lemmas only')]

# Palette of the dataviz guidelines (as eval.ipynb section 9): surface, ink, hairlines, categorical slots in fixed order
SURFACE, INK, INK2, MUTED, GRID, AXIS = '#fcfcfb', '#0b0b0b', '#52514e', '#898781', '#e1e0d9', '#c3c2b7'
SLOTS = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4', '#008300', '#4a3aa7', '#e34948']
FAMILY_MARKER = {'tt': 'o', 'tucker': 's'}


def tasks():
    """The POLAR columns of every table: all but pc.SKIP (NP once, as its 10-fold mean)."""
    return pc._tasks('all')


def score_cols():
    """The score columns of the tables: the POLAR tasks, their average, the judge's."""
    return [*tasks(), AVG, *TEA]


# --- Loading -------------------------------------------------------------------------------
@dataclass
class Suite:
    polar: pd.DataFrame     # rows kept, on our words, indexed (model, variant): what pc.* takes
    full: pd.DataFrame      # baselines on their own whole vocabulary (*_full)
    loaded: pd.DataFrame    # every row loaded (raw variant), before the cuts
    dropped: pd.DataFrame   # rows left out, with the reason
    judge: pd.DataFrame     # the judge summary, every row
    judge_csv: Path
    series: str
    model_files: dict       # run -> model file
    trials: dict            # judge fingerprint -> intruder found per dimension (checks() only)

    @property
    def df(self):
        """The rows kept, raw variant, indexed by model."""
        return self.polar.xs('raw', level='variant')


def _manifests(prefix):
    """Complete manifests of `prefix`, newest first (a dry run counts; --only runs and unfinished loads do not)."""
    out = []
    for path in sorted(ps.RESULTS_DIR.glob(f'{prefix}_*.json'), reverse=True):
        man = ps.load_manifest(path)
        if 'models' in man and not man.get('only'):
            out.append(man)
    return out


def _subset(man, names):
    return {**man, 'models': {n: man['models'][n] for n in names},
            'runs': [r for r in man['runs'] if r[0] in names], 'missing': {}}


def load(series='latest', min_iters=None, exclude=(), skip=('word_analogy',), judge_csv=None):
    """The POLAR rows of `series` (newest complete sweep + newest methods run) with the judge's columns.

    Rows stopped before `min_iters` or matching a pattern of `exclude` are left out (`.dropped`). A GloVe /
    word2vec baseline the newest sweep lacks (a --no-glove run) comes from the newest sweep that has it.
    Series 'same' (polar_sweep.py --series same): our runs only, each matched set at its common checkpoint
    (named <model>_<iteration> when not the table's); no baselines, and no judge scores (the judge has no such series)."""
    pc.SKIP = set(skip)
    prefix = {'latest': 'sweep', 'best': 'best', 'same': 'same'}[series]
    sweeps, methods = _manifests(prefix), _manifests('methods') if series != 'same' else []
    if not sweeps:
        raise FileNotFoundError(f'no complete {prefix}_*.json in {ps.RESULTS_DIR}')
    newest, mans = sweeps[0], []
    print(f"{prefix}: {Path(newest['path']).name} ({newest.get('status')})")
    for name in eu.baseline_specs() if series != 'same' else ():
        if name not in newest['models']:
            older = next((m for m in sweeps[1:] if name in m['models']), None)
            print(f"  {name}: not in it, " + (f"from {Path(older['path']).name}" if older else 'in no manifest'))
            if older:
                mans.append(_subset(older, [name]))
    mans.append(newest)
    if methods:
        print(f"methods: {Path(methods[0]['path']).name} ({methods[0].get('status')})")
        mans.append(methods[0])
    polar = pc.load(manifests=mans)
    polar = polar[[not any(fnmatch.fnmatchcase(m, p) for p in IGNORE) for m in polar.index.get_level_values('model')]]
    files = {name: m['checkpoint']['model_file'] for man in mans for name, m in man['models'].items()
             if (m.get('checkpoint') or {}).get('model_file')}

    # TT bond dimension: '_tt<b>' in the run name, else BASE; 0 = Tucker
    bond = pd.to_numeric(polar['run'].str.extract(r'_tt(\d+)$', expand=False)).fillna(eu.BASE['tt_rank'])
    polar['tt_rank'] = bond.where(polar['family'].eq('tt'), 0).where(polar['family'].isin(OURS)).astype('Int64')

    polar[AVG] = polar[tasks()].astype(float).mean(axis=1, skipna=False)

    judge_csv = Path(judge_csv) if judge_csv else sorted(JUDGE_DIR.glob('summary_*.csv'))[-1]
    judge = pd.read_csv(judge_csv)
    tl = judge[judge['group'].ne('human') & judge['series'].isin([series, '-'])].set_index('name')
    tl = tl[~tl.index.duplicated(keep='last')]
    for col, src in [('judge_acc', 'accuracy'), ('diversity', 'diversity'), ('n_correct', 'n_correct'),
                     ('n_dims', 'n_dims'), ('fingerprint', 'fingerprint')]:
        polar[col] = polar['run'].map(tl[src])
    loaded = polar.xs('raw', level='variant').copy()

    models = polar.index.get_level_values('model')
    ours = polar['family'].isin(OURS).to_numpy()
    why = np.full(len(polar), '', dtype=object)
    if min_iters is not None:
        why[ours & (polar['reached'] < min_iters).fillna(False).to_numpy(bool)] = f'reached < {min_iters}'
    for pat in exclude:
        why[np.array([fnmatch.fnmatchcase(m, pat) for m in models]) & (why == '')] = f'EXCLUDE {pat!r}'
    # one run per setting: the training-seed replicates are compared apart (seed_report)
    replicate = ours & polar['random_state'].fillna(1).ne(1).to_numpy(bool)
    why[replicate & (why == '')] = 'seed replicate (random_state != 1)'
    cut = why != ''
    dropped = (polar[cut].reset_index(level='variant', drop=True)[['run', 'family', 'reached', 'checkpoint']]
               .assign(reason=why[cut]))
    polar = polar[~cut]
    own = polar['vocab'].eq('own').to_numpy()
    full, polar = polar[own].copy(), polar[~own].copy()

    trials = load_trials(judge_csv, polar['fingerprint'].dropna().unique())
    kept = polar['family'].isin(OURS)
    print(f'kept {kept.sum()} of our runs and {(~kept).sum()} baselines on our words (series {series}); '
          f'{len(dropped)} rows left out, {len(full)} *_full rows apart')
    print(f'judge: {judge_csv.name}; trials of {len(trials)} judged rows')
    return Suite(polar, full, loaded, dropped, judge, judge_csv, series, files, trials)


_FP_RE = re.compile(r'"fingerprint":\s*"([^"]+)"')


def load_trials(judge_csv, fingerprints):
    """fingerprint -> intruder found per dimension, from the records of the judge sweep that wrote `judge_csv`
    (judge_results/records.jsonl); checks() tests them against the summary."""
    path = JUDGE_DIR / 'records.jsonl'
    stamp = Path(judge_csv).stem.removeprefix('summary_')
    wanted, out = set(fingerprints), {}
    if not path.exists():
        print(f'no {path}: the judge summary is not checked against its trials')
        return out
    with open(path, encoding='utf-8') as fh:
        for line in fh:
            m = _FP_RE.search(line)
            if not m or m.group(1) not in wanted:
                continue
            try:
                r = json.loads(line)
            except ValueError:  # a line cut off by an interrupt
                continue
            if str(r.get('sweep', '')).replace(':', '') != stamp:
                continue
            out[r['fingerprint']] = np.array([t['pick'] == t['random_word'] for t in r['tasks']])
    if wanted - set(out):
        print(f'  {len(wanted - set(out))} of {len(wanted)} judged rows have no trials of sweep {stamp} in {path.name}: '
              'not checked against the summary')
    return out


def checks(S):
    """Our runs whose POLAR and judge rows scored different states (empty = fine); warns if the judge's
    trial records do not reproduce its summary."""
    ours = _ours(S.df)
    j = S.judge[S.judge['group'].eq('table') & S.judge['series'].eq(S.series)].set_index('name')
    if S.series == 'latest':  # POLAR keeps the checkpoint's file name, the judge its iteration
        state = j['iteration'].map(lambda i: f'{i:.0f}.pt' if pd.notna(i) else None)
    else:  # both keep the model file
        state = j['model_path'].map(lambda p: Path(p).name if isinstance(p, str) else None)
    bad = [m for m, fp in S.df['fingerprint'].dropna().items()
           if fp in S.trials and S.trials[fp].sum() != S.df.at[m, 'n_correct']]
    if bad:
        print('WARNING: the judge records do not reproduce the summary for', ', '.join(bad))
    out = pd.DataFrame({'run': ours['run'], 'POLAR': ours['checkpoint'], 'judge': ours['run'].map(state)})
    return out[out['POLAR'].ne(out['judge'])]


# --- Rows ----------------------------------------------------------------------------------
def _ours(df):
    return df[df['family'].isin(OURS)]


def _source(name):
    return next(((kind, data) for p, kind, data in SOURCES if fnmatch.fnmatchcase(name, p)), ('?', '?'))


def _latent_rank_from_name(name):
    if name.startswith('w2v'):
        return 300.0
    m = re.search(r'_k?(\d+)(?:_full)?$', name)
    return float(m.group(1)) if m else np.nan


def foundations(S):
    """Per row: what it is, its training data, its latent_rank (our rank, a baseline's number of dimensions), and
    how many of our words it knows."""
    df = S.df
    info = S.judge[S.judge['group'].isin(['baselines', 'methods'])].set_index('name')
    info = info[~info.index.duplicated(keep='last')]
    ours = df['family'].isin(OURS)
    out = pd.DataFrame(index=df.index)
    out['kind'] = [f'ours: {f}, {n}-gram' if o else _source(m)[0] for m, f, n, o in zip(df.index, df['family'], df['ngram'], ours)]
    out['data'] = ['FineWeb 1B' if o else _source(m)[1] for m, o in zip(df.index, ours)]
    named = pd.Series([_latent_rank_from_name(m) for m in df.index], index=df.index)
    out['latent_rank'] = df['rank'].astype(float).where(ours, df['run'].map(info['n_dims']).astype(float).fillna(named))
    out['words'] = df['dims'].astype(float).where(ours, df['run'].map(info['n_words']).astype(float))
    out['of our words'] = (out['words'] / OUR_WORDS).where(~ours)
    return out


def overview(S):
    """Every row on our words: POLAR tasks and the judge's accuracy, sorted by AVG (best first); mean rank over
    the POLAR tasks (1 = best), '+ judge' adds judge_acc to the ranked columns (– without a judge score)."""
    T = tasks()
    data = S.df[score_cols()].astype(float)
    f = foundations(S)
    out = pd.concat([f[['kind', 'latent_rank']], data], axis=1)
    out['mean rank'] = data[T].rank(ascending=False, method='min').mean(1)
    out['mean rank, + judge'] = data[[*T, *TEA]].rank(ascending=False, method='min').mean(1, skipna=False)
    out = out.sort_values([AVG, 'mean rank'], ascending=[False, True])
    return (pc._style_scores(out, score_cols()).format('{:.0f}', subset=['latent_rank'], na_rep='–')
            .format('{:.1f}', subset=['mean rank', 'mean rank, + judge'], na_rep='–')
            .set_caption(f'series {S.series}: scores, colour per column, best bold; sorted by {AVG} '
                         f'(the {len(T)} POLAR tasks), best first'))


def _wilson(c, n, level=0.95):
    """Wilson interval of a proportion c / n: (low, high)."""
    z = stats.norm.ppf(0.5 + level / 2)
    c, n = np.asarray(c, float), np.asarray(n, float)
    p = c / n
    centre = (p + z ** 2 / (2 * n)) / (1 + z ** 2 / n)
    half = z * np.sqrt(p * (1 - p) / n + z ** 2 / (4 * n ** 2)) / (1 + z ** 2 / n)
    return centre - half, centre + half


def intrusion_table(S):
    """Per row the judge scored: dimensions judged, judge_acc with its 95% Wilson interval (the uncertainty of
    one model's score from its trials, not over training seeds), and the diversity of the top words (distinct top
    words / (dimensions x k)), the only place diversity is shown. Sorted by kind, then latent_rank."""
    df = S.df[S.df['n_dims'].notna()]
    f = foundations(S).loc[df.index]
    lo, hi = _wilson(df['n_correct'], df['n_dims'])
    out = pd.DataFrame({'kind': f['kind'], 'latent_rank': f['latent_rank'], 'dims judged': df['n_dims'].astype(float),
                        'judge_acc': df['judge_acc'].astype(float), '95% CI low': lo, '95% CI high': hi,
                        'diversity': df['diversity'].astype(float)}, index=df.index)
    if 'n_short_dims' in S.judge.columns:  # dimensions with fewer than k positive words among ours
        j = S.judge[S.judge['series'].isin([S.series, '-'])].drop_duplicates('name', keep='last')
        out['short dims'] = df['run'].map(j.set_index('name')['n_short_dims']).astype(float)
    out = out.sort_values(['kind', 'latent_rank', 'judge_acc'], ascending=[True, True, False])
    acc = ['judge_acc', '95% CI low', '95% CI high']
    return (out.style.format('{:.3f}', subset=[*acc, 'diversity'], na_rep='–')
            .format('{:.0f}', subset=[c for c in ('latent_rank', 'dims judged', 'short dims') if c in out], na_rep='–')
            .background_gradient(cmap=pc.SEQ, subset=['judge_acc'])
            .set_caption(f'series {S.series}: word intrusion per row; 95% Wilson interval of judge_acc over its '
                         'trials (chance 0.20); diversity = distinct top words / (dimensions x k), for reference'))


# --- Matched sets --------------------------------------------------------------------------
def _order(factor, levels):
    levels = list(dict.fromkeys(levels))
    known = [v for v in LEVELS.get(factor, []) if v in levels]
    return known + sorted((v for v in levels if v not in known), key=lambda v: (isinstance(v, str), v))


def _lv(factor, v):
    """A level as text: r100, 10,000, 4g, ss0.025, b50, scSoftPlus."""
    if factor == 'ss_frac':
        return f'ss{float(v):g}'
    fmt = {'rank': 'r{}', 'dims': '{:,}', 'ngram': '{}g', 'tt_rank': 'b{}', 'random_state': 'rs{}'}.get(factor)
    return fmt.format(int(v)) if fmt else str(v)


def _set_label(ctx, factor):
    """A matched set as a run name with the varied setting as '*'."""
    star = {'family': '*', 'ngram': '*g', 'rank': 'r*', 'method': '*', 'ss_frac': 'ss*', 'dims': 'd*', 'tt_rank': 'tt*',
            'random_state': 'rs*'}

    def part(k):
        if k == factor:
            return star[k]
        v = ctx.get(k)
        if v is None:
            return ''
        if k == 'dims':
            return '' if int(v) == OUR_WORDS else f'd{int(v)}'
        if k == 'tt_rank':
            return '' if int(v) in (0, eu.BASE['tt_rank']) else f'tt{int(v)}'
        if k == 'random_state':
            return '' if int(v) == 1 else f'rs{int(v)}'
        if k == 'ngram':
            return f'{int(v)}g'
        if k == 'rank':
            return f'r{int(v)}'
        if k == 'ss_frac':
            return f'ss{float(v):g}'
        return str(v)

    return '_'.join(p for p in map(part, CONFIG) if p)


def matched_sets(S, factor, all_runs=False, warn=True):
    """{set label: Series level -> model}: our runs equal in every CONFIG setting but `factor`, at two levels
    or more. Tucker vs TT pairs TT at the default bond dimension. all_runs: also the rows the cuts left out."""
    runs = _ours(S.loaded if all_runs else S.df)
    ctx = [k for k in CONFIG if k != factor]
    if factor == 'family':
        runs = runs[runs['tt_rank'].isin([0, eu.BASE['tt_rank']])]
        ctx.remove('tt_rank')
    out = {}
    for key, g in runs.dropna(subset=[*ctx, factor]).groupby(ctx, sort=True):
        if g[factor].nunique() < 2:
            continue
        g = g.sort_values('reached', ascending=False, na_position='last')  # a level run twice: the longer run
        s = pd.Series(g.index.to_numpy(), index=g[factor].to_numpy()).groupby(level=0).first()
        s = s.reindex(_order(factor, s.index))
        label = _set_label(dict(zip(ctx, key)), factor)
        reached = runs.loc[s.to_numpy(), 'reached']
        if warn and reached.nunique() > 1:
            print(f'{factor}, {label}: runs stopped at different iterations: '
                  + ', '.join(f'{m} ({r})' for m, r in reached.items()))
        out[label] = s
    return out


def _design(sets, levels, min_levels):
    """(levels, set labels): the most levels (at least min_levels) that some matched sets all have, then the
    most such sets; None if there are none."""
    for k in range(len(levels), min_levels - 1, -1):
        having, combo = max((([label for label, s in sets.items() if set(c) <= set(s.index)], c)
                             for c in itertools.combinations(levels, k)), key=lambda t: len(t[0]))
        if having:
            return combo, having
    return None


def contrasts(factor, sets):
    """[(from, to, [(set, model at from, model at to)])]: neighbouring levels within each set for an ordered
    factor, else CONTRASTS (default: every level against the first)."""
    if factor in ORDERED:
        out = {}
        for label, s in sets.items():
            items = list(s.items())
            for (a, ma), (b, mb) in zip(items, items[1:]):
                out.setdefault((a, b), []).append((label, ma, mb))
        return [(a, b, p) for (a, b), p in sorted(out.items())]
    levels = _order(factor, [v for s in sets.values() for v in s.index])
    out = []
    for a, b in CONTRASTS.get(factor) or [(levels[0], v) for v in levels[1:]]:
        p = [(label, s[a], s[b]) for label, s in sets.items() if a in s.index and b in s.index]
        if p:
            out.append((a, b, p))
    return out


def level_means(S, sets, levels, labels, cols):
    """cols x levels: each level's scores averaged over the matched sets `labels`."""
    return pd.DataFrame({lv: S.df.loc[[sets[label][lv] for label in labels], cols].astype(float).mean()
                         for lv in levels})


def step_deltas(S, pairs, cols):
    """matched sets x cols: score at `to` − score at `from`."""
    return pd.DataFrame({label: S.df.loc[mb, cols].astype(float) - S.df.loc[ma, cols].astype(float)
                         for label, ma, mb in pairs}).T


# --- Tests ---------------------------------------------------------------------------------
def _wilcoxon(d):
    """Wilcoxon signed-rank test over the tasks' differences (zeros split, Demšar 2006): (statistic,
    matched-pairs rank-biserial r (Kerby 2014), p two-sided, p later level higher, p later level lower)."""
    d = pd.Series(d).dropna().to_numpy(float)
    if len(d) < 2 or not d.any():
        return (np.nan,) * 5
    r = stats.rankdata(np.abs(d))
    zero = r[d == 0].sum() / 2
    plus, minus = r[d > 0].sum() + zero, r[d < 0].sum() + zero
    res = stats.wilcoxon(d, zero_method='zsplit')
    up = stats.wilcoxon(d, zero_method='zsplit', alternative='greater').pvalue
    down = stats.wilcoxon(d, zero_method='zsplit', alternative='less').pvalue
    return res.statistic, (plus - minus) / (plus + minus), res.pvalue, up, down


def _page(M):
    """Page's L test (tasks = blocks, columns = levels ascending); two-sided p = twice the smaller one-sided.
    (L, mean Spearman rho of level and score over the tasks, p two-sided, p increasing, p decreasing)."""
    X = M.dropna().to_numpy(float)
    up, down = stats.page_trend_test(X), stats.page_trend_test(X[:, ::-1])
    level = np.arange(1, X.shape[1] + 1)
    ranks = np.apply_along_axis(stats.rankdata, 1, X)
    rho = np.nanmean([np.corrcoef(level, r)[0, 1] if r.std() > 0 else np.nan for r in ranks])
    return up.statistic, rho, min(1.0, 2 * min(up.pvalue, down.pvalue)), up.pvalue, down.pvalue


def _mantel(strata):
    """Stratified Cochran-Armitage trend test in proportions (Mantel 1963), two-sided; with two levels the
    Cochran-Mantel-Haenszel test. strata: (level scores, correct, trials) per matched set. (z, p)."""
    T = V = 0.0
    for x, y, n in strata:
        x, y, n = (np.asarray(v, float) for v in (x, y, n))
        N, p = n.sum(), y.sum() / n.sum()
        T += (x * (y - n * p)).sum()
        V += p * (1 - p) * ((n * x ** 2).sum() - (n * x).sum() ** 2 / N) * N / (N - 1)
    z = T / np.sqrt(V) if V > 0 else np.nan
    return z, 2 * stats.norm.sf(abs(z)) if np.isfinite(z) else np.nan


def _cmh_general(strata):
    """Generalised Cochran-Mantel-Haenszel test of general association (levels x intruder found or not,
    strata = matched sets): does the accuracy differ between the levels, in any pattern? (Q, df, p)."""
    O = E = V = 0.0
    for y, n in strata:
        y, n = np.asarray(y, float), np.asarray(n, float)
        N, p = n.sum(), y.sum() / n.sum()
        O, E = O + y[:-1], E + n[:-1] * p
        V = V + p * (1 - p) * N / (N - 1) * (np.diag(n[:-1]) - np.outer(n[:-1], n[:-1]) / N)
    d = O - E
    try:
        Q = float(d @ np.linalg.solve(V, d))
    except np.linalg.LinAlgError:
        return np.nan, len(d), np.nan
    return Q, len(d), stats.chi2.sf(Q, len(d))


def _holm(p):
    """Holm-adjusted p-values; NaN stays NaN."""
    p = np.asarray(p, float)
    ok = np.flatnonzero(np.isfinite(p))
    out = np.full(len(p), np.nan)
    running = 0.0
    for i, j in enumerate(ok[np.argsort(p[ok])]):
        running = max(running, min(1.0, (len(ok) - i) * p[j]))
        out[j] = running
    return out


def _polar_row(d, stat, effect, p, p_up, p_down):
    return {'POLAR tasks': int(d.notna().sum()), 'POLAR mean Δ': d.mean(),
            'POLAR tasks better': int((d > TOL).sum()), 'POLAR tasks worse': int((d < -TOL).sum()),
            'POLAR statistic': stat, 'POLAR effect': effect, 'POLAR p': p, 'POLAR p ↑': p_up, 'POLAR p ↓': p_down}


def _judge_counts(S, models):
    """Per matched set x level: intruders found, trials, and the sets the judge scored at every level."""
    c = np.array([[S.df.at[m, 'n_correct'] for m in ms] for ms in models], float)
    n = np.array([[S.df.at[m, 'n_dims'] for m in ms] for ms in models], float)
    return c, n, np.isfinite(c).all(1) & np.isfinite(n).all(1)


def _tea_row(S, models):
    """The judge's tests; models: per matched set its runs at the tested levels, in order.
    ↑ / ↓: one-sided, the later level higher / lower."""
    c, n, ok = _judge_counts(S, models)
    row = {'judge acc sets': int(ok.sum()), 'judge acc trials': int(n[ok].sum()), 'judge acc by level': '–',
           'judge acc Δ': np.nan, 'judge acc z': np.nan,
           'judge acc p': np.nan, 'judge acc p ↑': np.nan, 'judge acc p ↓': np.nan}
    if not ok.any():
        return row
    x = np.arange(c.shape[1])
    z, row['judge acc p'] = _mantel([(x, ci, ni) for ci, ni in zip(c[ok], n[ok])])
    row['judge acc z'] = z
    if np.isfinite(z):
        row['judge acc p ↑'], row['judge acc p ↓'] = stats.norm.sf(z), stats.norm.cdf(z)
    row['judge acc by level'] = ' → '.join(f'{v:.3f}' for v in c[ok].sum(0) / n[ok].sum(0))
    row['judge acc Δ'] = np.mean(c[ok][:, -1] / n[ok][:, -1] - c[ok][:, 0] / n[ok][:, 0])
    return row


def _reading(p, sign, pos, neg):
    """The Holm p at ALPHA in words; '–' when there was no test (no judge score, too few tasks)."""
    if not np.isfinite(p):
        return '–'
    return 'no evidence' if not p < ALPHA else pos if sign > 0 else neg


# The suites, as column prefixes: POLAR/SPINE tasks, the judge's accuracy (judge_acc)
POL, ACC = 'POLAR', 'judge acc'
SUITES = [POL, ACC]
SIGN = {POL: 'POLAR effect', ACC: 'judge acc z'}  # the column whose sign is the direction


def _h1(factor):
    """The alternative per suite: +1 the later level scores higher, −1 lower, None two-sided."""
    return {suite: EXPECTED.get(factor, {}).get(suite) for suite in SUITES}


def _finish(t, factor):
    """Holm per suite over the table's tests (one-sided where EXPECTED fixes the direction), readings, H1."""
    h1 = _h1(factor)
    for suite in SUITES:
        e = h1[suite]
        used = t[f'{suite} p'] if not e else t[f'{suite} p ↑'] if e > 0 else t[f'{suite} p ↓']
        t[f'{suite} p (Holm)'] = _holm(used)
        t[f'{suite} reading'] = [_reading(q, s if not e else e, pos, neg) for q, s, pos, neg
                                 in zip(t[f'{suite} p (Holm)'], t[SIGN[suite]], t['_pos'], t['_neg'])]
    t['H1'] = ' · '.join(f"{s} {'↑' if e > 0 else '↓'}" for s, e in h1.items() if e) or 'two-sided'
    return t.drop(columns=['_pos', '_neg'])


@dataclass
class Factor:
    name: str
    sets: dict           # set label -> Series level -> model
    levels: list
    contrasts: list      # (from, to, [(set, model at from, model at to)])
    trend: tuple | None  # (levels, set labels) of the trend test
    tests: pd.DataFrame  # one row per test: POLAR and judge columns


def analyse(S, factor):
    """Matched sets of `factor`, and per test (trend over the most shared levels, then each contrast):
    POLAR/SPINE with the tasks as units (Page's L / Wilcoxon), the judge's accuracy with its trials as units
    (Cochran-Armitage / CMH). Holm per suite within the factor."""
    sets = matched_sets(S, factor)
    levels = _order(factor, [v for s in sets.values() for v in s.index])
    cons = contrasts(factor, sets)
    trend = _design(sets, levels, 3) if factor in ORDERED and sets else None
    T, rows = tasks(), []
    if trend:
        lv, labels = trend
        M = level_means(S, sets, lv, labels, T)
        rows.append({'test': 'trend', 'levels': ' < '.join(_lv(factor, v) for v in lv), 'POLAR sets': len(labels),
                     **_polar_row(M[lv[-1]] - M[lv[0]], *_page(M)),
                     **_tea_row(S, [[sets[label][v] for v in lv] for label in labels]),
                     '_pos': 'better with more', '_neg': 'worse with more'})
    for a, b, pairs in cons:
        d = step_deltas(S, pairs, T).mean()
        rows.append({'test': 'step' if factor in ORDERED else 'contrast', 'levels': _step_label(factor, a, b),
                     'POLAR sets': len(pairs), **_polar_row(d, *_wilcoxon(d)),
                     **_tea_row(S, [[ma, mb] for _, ma, mb in pairs]),
                     '_pos': f'{_lv(factor, b)} better', '_neg': f'{_lv(factor, a)} better'})
    t = pd.DataFrame(rows)
    if len(t):
        t.insert(0, 'factor', factor)
        t = _finish(t, factor)
    return Factor(factor, sets, levels, cons, trend, t)


def _step_label(factor, a, b):
    return f'{_lv(factor, a)} → {_lv(factor, b)}'


POLAR_COLS = ['POLAR sets', 'POLAR tasks', 'POLAR mean Δ', 'POLAR tasks better', 'POLAR tasks worse',
              'POLAR statistic', 'POLAR effect', 'POLAR p (Holm)', 'POLAR reading', 'H1']
TEA_COLS = ['judge acc sets', 'judge acc trials', 'judge acc by level', 'judge acc Δ', 'judge acc z',
            'judge acc p (Holm)', 'judge acc reading', 'H1']
SUMMARY_COLS = ['POLAR sets', 'POLAR mean Δ', 'POLAR effect', 'POLAR p (Holm)', 'POLAR reading',
                'judge acc Δ', 'judge acc p (Holm)', 'judge acc reading', 'H1']


def style_tests(t, cols, caption=''):
    """A test table: Δ columns red / blue, Holm-adjusted p bold below ALPHA."""
    cols = [c for c in cols if c in t.columns]
    data = t.set_index(['factor', 'test', 'levels'])[cols]
    deltas = [c for c in cols if c.endswith((' Δ', ' effect'))]
    pcols = [c for c in cols if re.search(r' p( |$)', c)]
    num = [c for c in cols if c.endswith((' statistic', ' z'))]
    sty = data.style
    if deltas:
        sty = sty.format('{:+.3f}', subset=deltas, na_rep='–')
    sty = (sty.format('{:.4f}', subset=pcols, na_rep='–')
           .apply(lambda c: ['font-weight: bold' if v < ALPHA else '' for v in c],
                  subset=[c for c in pcols if 'Holm' in c]))
    if num:
        sty = sty.format('{:.2f}', subset=num, na_rep='–')
    return pc._shade_deltas(sty, data, deltas).set_caption(caption)


def summary(results):
    """Every factor's tests in one table."""
    t = pd.concat([R.tests for R in results if len(R.tests)], ignore_index=True)
    return style_tests(t, SUMMARY_COLS, f'Δ = the later level − the earlier; bold = Holm p < {ALPHA} '
                                        '(Holm per suite within a factor)')


def scores(S, R):
    """Every run of the matched sets, level by level; best per set bold."""
    cols = score_cols()
    rows = [{'matched set': label, 'level': _lv(R.name, lv), 'model': m, 'reached': S.df.at[m, 'reached'],
             **S.df.loc[m, cols].astype(float).to_dict()}
            for label, s in R.sets.items() for lv, m in s.items()]
    out = pd.DataFrame(rows).set_index(['matched set', 'level'])

    def best_in_set(col):
        top = col.groupby(level=0).transform('max')
        return ['font-weight: bold' if pd.notna(v) and v == b else '' for v, b in zip(col, top)]

    return (out.style.format('{:.3f}', subset=cols, na_rep='–').background_gradient(cmap=pc.SEQ, subset=cols)
            .apply(best_in_set, subset=cols).set_caption(f'{R.name}: the matched runs; best per set bold'))


def coverage_shift(S, R):
    """Per contrast, the change in each task's main coverage measure (to − from, mean over the sets): token
    coverage for the text tasks and NP, the share of triples / pairs scored for discrim_attr / wordsim353."""
    cov = {t: next((c for c in (f'cov.{t}.token_coverage', f'cov.{t}.triples_scored', f'cov.{t}.pairs_scored')
                    if c in S.df.columns), None) for t in tasks()}
    cov = {t: c for t, c in cov.items() if c}
    out = pd.DataFrame({_step_label(R.name, a, b): step_deltas(S, pairs, list(cov.values())).mean()
                        for a, b, pairs in R.contrasts}).T
    out.columns = list(cov)
    return pc._shade_deltas(out.style.format('{:+.3f}', na_rep='–'), out, list(out.columns)).set_caption(
        f'{R.name}: coverage, to − from, mean over the sets (blue = the later level knows more of the task)')


def report(S, factor, plots=True):
    """analyse(), then the matched runs, both test tables and (plots) the profiles and paired differences."""
    R = analyse(S, factor)
    if not R.sets:
        print(f'{factor}: no matched sets among the runs kept')
        return R
    display(scores(S, R))
    h1 = f'; H1 {R.tests["H1"].iloc[0]} (one-sided there)' if R.tests['H1'].iloc[0] != 'two-sided' else '; two-sided'
    display(style_tests(R.tests, POLAR_COLS, f'POLAR/SPINE, {len(tasks())} tasks as the units: mean Δ = later − '
                                              f'earlier level; effect = ρ (trend) or rank-biserial r{h1}'))
    display(style_tests(R.tests, TEA_COLS, f"judge (series {S.series}): acc = accuracy on word intrusion "
                                           f'(judge_acc), its trials as the units{h1}'))
    if plots:
        plot_profiles(S, R)
        plot_steps(S, R)
    return R


# --- Training seed -------------------------------------------------------------------------
def with_seeds(S):
    """S plus the seed replicates that load() left out only for being replicates (their judge trials loaded for
    checks())."""
    from dataclasses import replace
    reps = S.dropped.index[S.dropped['reason'].str.startswith('seed replicate')]
    rows = S.loaded.loc[reps].assign(variant='raw').set_index('variant', append=True)
    fps = set(rows['fingerprint'].dropna()) - set(S.trials)
    return replace(S, polar=pd.concat([S.polar, rows]),
                   trials={**S.trials, **(load_trials(S.judge_csv, fps) if fps else {})})


def seed_noise(S, R):
    """Per column: SD over the seeds per set (ddof 1), pooled SD, SD over our seed-1 runs on our words, their
    ratio, and band = 2·√2·pooled SD (about the largest difference two seeds of one setting show)."""
    cols = score_cols()
    X = {label: S.df.loc[s.to_numpy(), cols].astype(float) for label, s in R.sets.items()}
    SD = pd.DataFrame({label: x.std(ddof=1) for label, x in X.items()})
    DOF = pd.DataFrame({label: x.notna().sum() - 1 for label, x in X.items()})
    DOF = DOF.where(SD.notna(), 0)  # a set without an SD (fewer than two scores) adds nothing to the pool
    out = SD.add_prefix('SD ')
    out['pooled SD'] = np.sqrt((SD ** 2 * DOF).sum(1) / DOF.sum(1))
    ours = _ours(S.df)
    settings = ours[ours['random_state'].eq(1) & ours['dims'].astype(float).eq(OUR_WORDS)]
    out['SD over settings'] = settings[cols].astype(float).std()
    out['seed / settings'] = out['pooled SD'] / out['SD over settings']
    out['band'] = 2 * np.sqrt(2) * out['pooled SD']
    return out


def seed_tests(S, R):
    """Per replicate set, do the seeds differ by more than the judge's own noise? Generalised CMH on the
    accuracy (χ² of homogeneity over the trials); Holm over the sets."""
    rows = []
    for label, s in R.sets.items():
        models = list(s.to_numpy())
        row = {'factor': 'random_state', 'test': 'seeds', 'levels': label, 'judge acc p': np.nan}
        c, n, ok = _judge_counts(S, [models])
        if ok.all():
            Q, _, row['judge acc p'] = _cmh_general([(c[0], n[0])])
            row.update({'judge acc trials': int(n[0].sum()), 'judge acc statistic': Q,
                        'judge acc by seed': ' · '.join(f'{v:.3f}' for v in c[0] / n[0])})
        rows.append(row)
    t = pd.DataFrame(rows)
    t[f'{ACC} p (Holm)'] = _holm(t[f'{ACC} p'])
    t[f'{ACC} reading'] = np.where(t[f'{ACC} p (Holm)'] < ALPHA, 'beyond judge noise', 'within judge noise')
    return t


def beyond_seed(S, results, band):
    """Per change of section 2 (every contrast of `results`): the share of its matched pairs whose |Δ| exceeds
    the seed band, per column. Near 0: the change is no larger than what a new seed does."""
    cols = score_cols()
    out = {}
    for R in results:
        for a, b, pairs in R.contrasts:
            D = step_deltas(S, pairs, cols).abs()
            out[(R.name, _step_label(R.name, a, b))] = {'pairs': len(pairs), **(D.gt(band[cols]).sum() / D.notna().sum())}
    out = pd.DataFrame(out).T
    out.index.names = ['factor', 'change']
    return out


def seed_report(S, results, plots=True):
    """The training-seed replicates (with_seeds): runs per seed, noise, tests, section 2's changes against the
    seed band, and (plots) the scores per seed and the training curves. Returns (suite with seeds, Factor, noise)."""
    S2 = with_seeds(S)
    sets = matched_sets(S2, 'random_state')
    if not sets:
        print('no seed replicates among the runs kept (check S.dropped and MIN_ITERS)')
        return S2, None, None
    R = Factor('random_state', sets, _order('random_state', [v for s in sets.values() for v in s.index]),
               [], None, pd.DataFrame())
    display(scores(S2, R))
    noise = seed_noise(S2, R)
    display(noise.style.format('{:.4f}', na_rep='–').format('{:.0%}', subset=['seed / settings'], na_rep='–')
            .background_gradient(cmap=pc.SEQ, subset=['seed / settings'], vmin=0, vmax=1)
            .set_caption('seed noise per column; SD over settings: our seed-1 runs on our words; band = 2·√2·pooled SD'))
    beyond = beyond_seed(S, results, noise['band'])
    shares = [c for c in beyond.columns if c != 'pairs']
    display(beyond.style.format('{:.0f}', subset=['pairs']).format('{:.0%}', subset=shares, na_rep='–')
            .background_gradient(cmap=pc.SEQ, subset=shares, vmin=0, vmax=1)
            .set_caption("share of section 2's matched pairs whose |Δ| exceeds the seed band of the column"))
    t = seed_tests(S2, R)
    display(style_tests(t, ['judge acc trials', 'judge acc by seed', 'judge acc statistic', 'judge acc p (Holm)',
                            'judge acc reading'],
                        f"check, H0: the seeds differ only by the judge's trial noise; bold = Holm p < {ALPHA} "
                        '(Holm over the sets)'))
    if plots:
        plot_profiles(S2, R, mean=False)
        plot_curves(S2, curve_groups(S2, R))
    return S2, R, noise


# --- Figures -------------------------------------------------------------------------------
def _style(ax, grid='y'):
    ax.set_facecolor(SURFACE)
    ax.grid(axis=grid, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ('top', 'right'):
        ax.spines[side].set_visible(False)
    for side in ('left', 'bottom'):
        ax.spines[side].set_color(AXIS)
    ax.tick_params(colors=AXIS, labelcolor=INK2, labelsize=8)


def _legend(ax, handles, **kw):
    for t in ax.legend(handles=handles, frameon=False, fontsize=8, **kw).get_texts():
        t.set_color(INK2)


def plot_profiles(S, R, cols=None, ncol=6, mean=True):
    """Per column, each matched set as a grey line (marker = family), in ink (mean) the mean over the sets that
    have the most levels."""
    cols = [*tasks(), *TEA] if cols is None else cols
    x = {lv: i for i, lv in enumerate(R.levels)}
    design = _design(R.sets, R.levels, 2) if mean else None
    nrow = -(-(len(cols) + 1) // ncol)  # one spare panel for the legend
    fig, axes = plt.subplots(nrow, ncol, figsize=(2.3 * ncol, 2.3 * nrow), facecolor=SURFACE, squeeze=False)
    axes = axes.ravel()
    marker = (lambda s: 'o') if R.name == 'family' else (lambda s: FAMILY_MARKER.get(S.df.at[s.iloc[0], 'family'], 'o'))
    for ax, col in zip(axes, cols):
        _style(ax)
        for s in R.sets.values():
            ax.plot([x[lv] for lv in s.index], S.df.loc[s.to_numpy(), col].astype(float).to_numpy(), color=MUTED,
                    linewidth=1, alpha=0.8, marker=marker(s), markersize=5, markeredgecolor=SURFACE, markeredgewidth=1)
        if design:
            m = level_means(S, R.sets, *design, [col]).loc[col]
            ax.plot([x[lv] for lv in design[0]], m.to_numpy(), color=INK, linewidth=2, marker='o', markersize=6,
                    markeredgecolor=SURFACE, markeredgewidth=1.5, zorder=3)
        ax.set_title(col, loc='left', fontsize=9, color=INK)
        ax.set_xticks(range(len(R.levels)))
        ax.set_xticklabels([_lv(R.name, lv) for lv in R.levels], rotation=30 if R.name == 'method' else 0)
        ax.set_xlim(-0.3, len(R.levels) - 0.7)
    for ax in axes[len(cols):]:
        ax.axis('off')
    if R.name == 'family':
        handles = [Line2D([], [], color=MUTED, linewidth=1, marker='o', markersize=5, label='one matched set')]
    else:
        handles = [Line2D([], [], color=MUTED, linewidth=1, marker=FAMILY_MARKER.get(f, 'o'), markersize=5,
                          markeredgecolor=SURFACE, label=f'{f}: one matched set')
                   for f in sorted({S.df.at[s.iloc[0], 'family'] for s in R.sets.values()})]
    if design:
        handles.append(Line2D([], [], color=INK, linewidth=2, marker='o', markersize=6, markeredgecolor=SURFACE,
                              label=f'mean of the {len(design[1])} sets with\n'
                                    + ', '.join(_lv(R.name, v) for v in design[0])))
    _legend(axes[-1], handles, loc='center left')
    fig.suptitle(f'{R.name}: scores per level, matched runs joined', x=0.01, ha='left', fontsize=11, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    plt.show()


def _p_text(R, levels, suite):
    """The Holm p of one test in `suite`, with ↑ / ↓ when that suite's test is one-sided."""
    p = R.tests.loc[R.tests['levels'].eq(levels), f'{suite} p (Holm)']
    e = _h1(R.name)[suite]
    mark = '' if not e else '↑' if e > 0 else '↓'
    return f'{p.iloc[0]:.3f}{mark}' if len(p) and np.isfinite(p.iloc[0]) else '–'


def directions(S, R):
    """Per test of `R` and suite: both one-sided p-values (↑ the later level higher, ↓ lower; Holm per column)
    next to the two-sided one; then, for a factor with a trend test, the order-free tests."""
    if not len(R.tests):
        print(f'{R.name}: no tests')
        return
    t = R.tests.set_index(['factor', 'test', 'levels'])
    out = pd.DataFrame(index=t.index)
    for suite in SUITES:
        out[f'{suite} p ↑'] = _holm(t[f'{suite} p ↑'])
        out[f'{suite} p ↓'] = _holm(t[f'{suite} p ↓'])
        out[f'{suite} p two-sided'] = _holm(t[f'{suite} p'])
    display(out.style.format('{:.4f}', na_rep='–')
            .apply(lambda c: ['font-weight: bold' if v < ALPHA else '' for v in c])
            .set_caption(f'{R.name}: Holm-adjusted p per column; ↑ = the later level scores higher, ↓ = lower; '
                         f'bold < {ALPHA}. Use a one-sided column only for a direction fixed before looking'))
    if R.trend:
        display(order_free(S, R))


def order_free(S, R):
    """Do the trend's levels differ at all, in any pattern (not only monotone)? POLAR: Friedman (tasks as
    blocks, level scores = means over the sets; effect Kendall's W); judge accuracy: generalised CMH."""
    levels, labels = R.trend
    models = [[R.sets[label][v] for v in levels] for label in labels]
    M = level_means(S, R.sets, levels, labels, tasks()).dropna()
    chi2, p = stats.friedmanchisquare(*M.to_numpy(float).T)
    rows = [{'suite': POL, 'test': 'Friedman', 'units': f'{len(M)} tasks', 'statistic': chi2,
             'df': len(levels) - 1, "Kendall's W": chi2 / (len(M) * (len(levels) - 1)), 'p': p}]
    c, n, ok = _judge_counts(S, models)
    if ok.any():
        Q, df, p = _cmh_general(list(zip(c[ok], n[ok])))
        rows.append({'suite': ACC, 'test': 'generalised CMH', 'units': f'{int(n[ok].sum())} trials',
                     'statistic': Q, 'df': df, 'p': p})
    out = pd.DataFrame(rows).set_index(['suite', 'test'])
    return (out.style.format('{:.4f}', subset=['p']).format('{:.3g}', subset=['statistic'])
            .format('{:.3f}', subset=["Kendall's W"], na_rep='–').format('{:.0f}', subset=['df'], na_rep='–')
            .apply(lambda c: ['font-weight: bold' if v < ALPHA else '' for v in c], subset=['p'])
            .set_caption(f"{R.name}: order-free tests over {', '.join(_lv(R.name, v) for v in levels)} "
                         f'({len(labels)} sets); one test per suite, unadjusted'))


def plot_steps(S, R):
    """Per contrast, each matched set's difference (to − from) per column, the ink tick their mean;
    POLAR tasks left, the judge's accuracy right, each on its own scale."""
    if not R.contrasts:
        print(f'{R.name}: no matched pairs')
        return
    T = tasks()
    fig, axes = plt.subplots(1, 3, figsize=(14, 3.4), facecolor=SURFACE,
                             gridspec_kw={'width_ratios': [len(T), len(TEA) + 0.5, 3.8]})
    width = 0.75 / len(R.contrasts)
    for ax, cols in zip(axes, (T, TEA)):
        _style(ax)
        ax.axhline(0, color=AXIS, linewidth=1, zorder=1)
        for i, (a, b, pairs) in enumerate(R.contrasts):
            D = step_deltas(S, pairs, cols)
            off = (i - (len(R.contrasts) - 1) / 2) * width
            for j, col in enumerate(cols):
                v = D[col].dropna()
                ax.scatter(np.full(len(v), j + off), v, s=36, color=SLOTS[i % len(SLOTS)], edgecolors=SURFACE,
                           linewidths=1, zorder=3)
                if len(v):
                    ax.hlines(v.mean(), j + off - 0.45 * width, j + off + 0.45 * width, colors=INK, linewidth=2, zorder=4)
        ax.set_xticks(range(len(cols)))
        ax.set_xticklabels(cols, rotation=30, ha='right')
        ax.set_xlim(-0.6, len(cols) - 0.4)
    axes[0].set_ylabel('Δ score (to − from)', color=INK2, fontsize=9)
    axes[0].set_title('POLAR/SPINE tasks', loc='left', fontsize=9, color=INK)
    axes[1].set_title('word intrusion (judge)', loc='left', fontsize=9, color=INK)
    handles = []
    for i, (a, b, pairs) in enumerate(R.contrasts):
        lv = _step_label(R.name, a, b)
        handles.append(Line2D([], [], linestyle='', marker='o', markersize=7, color=SLOTS[i % len(SLOTS)],
                              markeredgecolor=SURFACE,
                              label=f'{lv}: {len(pairs)} set' + 's' * (len(pairs) > 1) + '\np (Holm): '
                                    f'POLAR {_p_text(R, lv, POL)}, judge acc {_p_text(R, lv, ACC)}'))
    handles.append(Line2D([], [], color=INK, linewidth=2, label='mean over the sets'))
    axes[2].axis('off')
    _legend(axes[2], handles, loc='center left')
    title = f'{R.name}: paired differences'
    if R.trend:
        lv = ' < '.join(_lv(R.name, v) for v in R.trend[0])
        title += f'; trend {lv}, p (Holm): POLAR {_p_text(R, lv, POL)}, judge acc {_p_text(R, lv, ACC)}'
    fig.suptitle(title, x=0.01, ha='left', fontsize=11, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    plt.show()


# --- Training logs (inspect_tucker) --------------------------------------------------------
def _ref(model_file):
    """inspect_tucker's RunRef for a model file: its log, stitched over resumes by resume_chain."""
    from tensormet.experimental.inspect_tucker import RunRef
    p = Path(model_file)
    return RunRef(p.stem, p.with_name(f'{p.stem}_log.txt'), p.stem, None)


def curve_groups(S, R):
    """{matched set: {level: run}} of an analysed factor, for plot_curves."""
    return {label: {_lv(R.name, lv): S.loaded.at[m, 'run'] for lv, m in s.items()} for label, s in R.sets.items()}


def plot_curves(S, groups, keys=CURVE_KEYS, smooth=None):
    """Training curves from the runs' logs (inspect_tucker.load_metrics over the resume chain): one row per
    group {label: run}, one panel per key ('rec_error' or a semantic key of the log)."""
    from tensormet.experimental import inspect_tucker as it
    if not groups:
        print('no runs to plot')
        return
    fig, axes = plt.subplots(len(groups), len(keys), figsize=(3.6 * len(keys), 2.7 * len(groups)),
                             facecolor=SURFACE, squeeze=False)
    for row, (title, runs) in zip(axes, groups.items()):
        for ax, key in zip(row, keys):
            _style(ax)
            ax.set_title(key, loc='left', fontsize=9, color=INK)
        handles = []
        for i, (label, run) in enumerate(runs.items()):
            if run not in S.model_files:
                print(f'{run}: no model file in the manifests')
                continue
            try:
                its, rec, sem = it.load_metrics(*it.resume_chain(_ref(S.model_files[run])))
            except FileNotFoundError as e:
                print(f'{run}: {e}')
                continue
            for ax, key in zip(row, keys):
                x, y = (its, rec) if key == 'rec_error' else it._values_for_key(key, its, sem)
                if len(x):
                    ax.plot(x, it._smooth(y, smooth), color=SLOTS[i % len(SLOTS)], linewidth=2)
            handles.append(Line2D([], [], color=SLOTS[i % len(SLOTS)], linewidth=2, label=label))
        row[0].set_ylabel(title, color=INK2, fontsize=8)
        _legend(row[-1], handles, loc='best')
    for ax in axes[-1]:
        ax.set_xlabel('iteration', color=INK2, fontsize=8)
    fig.tight_layout()
    plt.show()


def iter_times(model_file):
    """The iteration times (s) a run's log records, over its resume chain (inspect_tucker.load_iter_times)."""
    from tensormet.experimental import inspect_tucker as it
    try:
        return it.load_iter_times(*it.resume_chain(_ref(model_file)))[1]
    except FileNotFoundError:
        return []


def cost(S, factor, all_runs=True):
    """Per matched set and level: seconds per iteration (median of the times in the run's log) and core
    parameters. all_runs: also the runs the cuts left out (a time per iteration needs no finished run)."""
    from tensormet.tt_hybrid.tt_chain import core_shapes
    rows = []
    for label, s in matched_sets(S, factor, all_runs=all_runs, warn=False).items():
        for lv, m in s.items():
            r = S.loaded.loc[m]
            secs = iter_times(S.model_files[r['run']]) if r['run'] in S.model_files else []
            ranks = [int(r['rank'])] * int(r['ngram'])
            core = (sum(a * b * c for a, b, c in core_shapes(ranks, int(r['tt_rank']))) if r['family'] == 'tt'
                    else int(r['rank']) ** int(r['ngram']))
            rows.append({'matched set': label, 'level': _lv(factor, lv), 'value': lv, 'model': m,
                         'reached': r['reached'], 's / iteration': np.median(secs) if secs else np.nan,
                         'iterations timed': len(secs), 'core parameters': core})
    return pd.DataFrame(rows).set_index(['matched set', 'level'])


def cost_by_factor(S, factors):
    """cost() of several factors in one table, plus '× first level': the time per iteration relative
    to the first level of the matched set (NaN if that level has no time)."""
    table = pd.concat({factor: cost(S, factor) for factor in factors}, names=['factor'])
    per_set = table.groupby(level=['factor', 'matched set'], sort=False)['s / iteration']
    first_level = per_set.transform(lambda times: times.iloc[0])
    table['× first level'] = table['s / iteration'] / first_level
    return table


def time_overview(S):
    """Seconds per iteration of every run of ours that was loaded, also the runs the cuts left out
    ('left out' gives the reason). Median ('s / iteration'), mean, min and max of the iteration
    times in the run's log; sorted by the median."""
    ours = S.loaded[S.loaded['family'].isin(OURS)]
    rows = {}
    for model, row in ours.iterrows():
        times = iter_times(S.model_files[row['run']]) if row['run'] in S.model_files else []
        rows[model] = {
            **row[CONFIG].to_dict(),
            'reached': row['reached'],
            's / iteration': np.median(times) if times else np.nan,
            'mean': np.mean(times) if times else np.nan,
            'min': np.min(times) if times else np.nan,
            'max': np.max(times) if times else np.nan,
            'iterations timed': len(times),
            'left out': S.dropped['reason'].get(model, ''),
        }
    table = pd.DataFrame.from_dict(rows, orient='index').rename_axis('model')
    return table.sort_values('s / iteration')


def plot_time(table, factor='ss_frac'):
    """Seconds per iteration against the level (linear axes), one line per matched set; returns the
    least-squares fit time = fixed + per_unit x level per set."""
    fig, ax = plt.subplots(figsize=(6, 3.6), facecolor=SURFACE)
    _style(ax)
    handles, fits = [], {}
    for i, (label, g) in enumerate(table.dropna(subset=['s / iteration']).groupby(level=0, sort=False)):
        x, y = g['value'].astype(float).to_numpy(), g['s / iteration'].to_numpy(float)
        ax.plot(x, y, color=SLOTS[i % len(SLOTS)], linewidth=2, marker='o', markersize=7, markeredgecolor=SURFACE)
        handles.append(Line2D([], [], color=SLOTS[i % len(SLOTS)], linewidth=2, marker='o', label=label))
        if len(x) >= 2:
            b, a = np.polyfit(x, y, 1)
            r2 = 1 - ((y - (a + b * x)) ** 2).sum() / ((y - y.mean()) ** 2).sum() if len(x) > 2 else np.nan
            fits[label] = {'levels': len(x), 'fixed (s)': a, 'per unit (s)': b, 'R²': r2}
    ax.set_xlabel(factor, color=INK2, fontsize=9)
    ax.set_ylabel('seconds per iteration', color=INK2, fontsize=9)
    ax.set_ylim(bottom=0)
    _legend(ax, handles, loc='upper left')
    fig.tight_layout()
    plt.show()
    return pd.DataFrame(fits).T


# --- Which run is best ---------------------------------------------------------------------
def ranking(S, cols=None, alpha=ALPHA):
    """Our runs sorted by AVG (best first), with the mean rank over `cols` (default the POLAR tasks; 1 = best, ties
    averaged), Friedman's test over the columns and Nemenyi's critical difference (Demšar 2006). Returns (table, CD)."""
    cols = tasks() if cols is None else cols
    ours = _ours(S.df)
    X = ours[cols].astype(float)
    if X.isna().any(axis=1).any():
        print('left out, a column missing:', ', '.join(X.index[X.isna().any(axis=1)]))
    X = X.dropna()
    R = X.rank(ascending=False)
    k, n = X.shape
    chi2, p = stats.friedmanchisquare(*X.to_numpy())
    cd = stats.studentized_range.ppf(1 - alpha, k, np.inf) / np.sqrt(2) * np.sqrt(k * (k + 1) / (6 * n))
    out = ours.loc[X.index, ['family', 'ngram', 'rank', 'method', 'ss_frac', 'dims', 'tt_rank', 'reached']].copy()
    out['mean rank'] = R.mean(axis=1)
    out['#1'] = (R == 1).sum(axis=1)
    out['mean score'] = X.mean(axis=1)
    out[AVG] = ours.loc[X.index, AVG]
    out[TEA] = ours.loc[X.index, TEA]
    out = out.sort_values([AVG, 'mean rank'], ascending=[False, True])
    out['within CD'] = (out['mean rank'] - out['mean rank'].min() <= cd).map({True: '✓', False: ''})
    print(f'Friedman over {n} columns and {k} runs: chi2 = {chi2:.1f}, p = {p:.2g}; '
          f'Nemenyi CD (alpha {alpha}) = {cd:.1f} ranks')
    return out, cd


def style_ranking(rk):
    scores_ = ['mean score', AVG, *TEA]
    return (rk.style.format('{:.2f}', subset=['mean rank']).format('{:.3f}', subset=scores_, na_rep='–')
            .format('{:g}', subset=['ss_frac']).background_gradient(cmap=pc.SEQ, subset=scores_)
            .set_caption(f'our runs by {AVG}, best first; ✓ = not significantly worse than the best mean rank '
                         '(Nemenyi)'))


def plot_ranking(rk, cd):
    """Mean rank per run (marker = family), the band [best, best + CD]."""
    y = np.arange(len(rk))[::-1]
    fig, ax = plt.subplots(figsize=(7.5, 0.26 * len(rk) + 1.3), facecolor=SURFACE)
    _style(ax, grid='x')
    best = rk['mean rank'].min()
    ax.axvspan(best, best + cd, color=GRID, alpha=0.7, zorder=0)
    handles = [Line2D([], [], color=GRID, linewidth=8, label=f'within CD = {cd:.1f} of the best')]
    for fam, mk in FAMILY_MARKER.items():
        sel = rk['family'].eq(fam).to_numpy()
        if sel.any():
            ax.scatter(rk['mean rank'].to_numpy()[sel], y[sel], marker=mk, s=45, color=INK, edgecolors=SURFACE,
                       linewidths=1, zorder=3)
            handles.append(Line2D([], [], linestyle='', marker=mk, markersize=7, color=INK, label=fam))
    ax.set_yticks(y)
    ax.set_yticklabels(rk.index, fontsize=8)
    ax.set_ylim(-0.7, len(rk) - 0.3)
    ax.set_xlabel('mean rank over the tasks (1 = best)', color=INK2, fontsize=9)
    _legend(ax, handles, loc='lower right')
    fig.tight_layout()
    plt.show()


# --- Against the baselines -----------------------------------------------------------------
def matched_baselines(S, headline, vocab_min=0.9):
    """{latent_rank: ([runs of `headline` at that latent_rank that were kept], [baselines of that latent_rank that
    know >= vocab_min of our words])}. headline: {latent_rank: [runs]}, fixed before looking at the results."""
    f = foundations(S)
    base = f[f['of our words'].ge(vocab_min)]
    out = {}
    for lr, models in headline.items():
        kept = [m for m in models if m in S.df.index]
        for m in models:
            if m not in kept:
                print(f'{m}: not among the rows kept (see S.dropped)')
        out[lr] = (kept, list(base.index[base['latent_rank'].eq(float(lr))]))
    return out


def versus(S, model, baselines, caption=''):
    """`model` against each baseline: scores side by side with Δ = model − baseline, then per baseline the tests
    (Wilcoxon over the POLAR tasks, 2 x 2 chi-square on the judge's accuracy), Holm over the baselines; then
    coverage. With 8 tasks the smallest Wilcoxon p is 2/2^8: when Holm over the baselines puts the smallest
    attainable p at ALPHA or above, the POLAR reading says 'no power'."""
    cols = score_cols()
    data = S.df.loc[[model, *baselines], cols].astype(float).T
    delta = pd.DataFrame({f'Δ vs {b}': data[model] - data[b] for b in baselines})
    out = pd.concat([data, delta], axis=1)
    sty = pc._style_scores(out, list(data.columns), axis=1).format('{:+.3f}', subset=list(delta.columns), na_rep='–')
    display(pc._shade_deltas(sty, out, list(delta.columns), axis=1).set_caption(
        caption or f'{model} against {len(baselines)} baselines; best per row bold, Δ blue = ours better'))
    rows = []
    for b in baselines:
        d = S.df.loc[model, tasks()].astype(float) - S.df.loc[b, tasks()].astype(float)
        rows.append({'factor': model, 'test': 'vs', 'levels': b, **_polar_row(d, *_wilcoxon(d)),
                     **_tea_row(S, [[b, model]]), '_pos': 'ours better', '_neg': f'{b} better'})
    t = _finish(pd.DataFrame(rows), None)
    floor = min(1.0, len(baselines) * 2 / 2 ** len(tasks()))  # smallest attainable Holm p on the POLAR tasks
    if floor >= ALPHA:
        t['POLAR reading'] = f'no power (Holm floor {floor:.2f})'
    display(style_tests(t, ['POLAR tasks', 'POLAR mean Δ', 'POLAR tasks better', 'POLAR tasks worse', 'POLAR effect',
                            'POLAR p', 'POLAR p (Holm)', 'POLAR reading', 'judge acc by level', 'judge acc Δ',
                            'judge acc p (Holm)', 'judge acc reading'],
                        f'Δ = {model} − baseline; POLAR p: two-sided Wilcoxon before Holm (smallest attainable '
                        f'{2 / 2 ** len(tasks()):.4f}); bold = Holm p < {ALPHA} (Holm over the baselines); '
                        f'judge acc by level: baseline → ours; smallest attainable POLAR Holm p here: {floor:.3f}'))
    display(pc.coverage(S.polar, model, *baselines))
    return t


# --- Appendix B: the best series -----------------------------------------------------------
def series_compare(S, S_other):
    """Our runs scored in both suites, joined on the run: POLAR_average and judge_acc side by side, Δ = other − S,
    and the file each scored. The caption gives the mean Δ and a Wilcoxon signed-rank test over the runs."""
    a, b = _ours(S.df).set_index('run'), _ours(S_other.df).set_index('run')
    runs = a.index.intersection(b.index)
    cols = [AVG, *TEA]
    out = pd.DataFrame(index=runs)
    notes = []
    for c in cols:
        out[f'{c} ({S.series})'] = a.loc[runs, c].astype(float)
        out[f'{c} ({S_other.series})'] = b.loc[runs, c].astype(float)
        d = out[f'{c} ({S_other.series})'] - out[f'{c} ({S.series})']
        out[f'Δ {c}'] = d
        d = d.dropna()
        p = stats.wilcoxon(d).pvalue if len(d) > 1 and d.ne(0).any() else np.nan
        notes.append(f'{c}: mean Δ {d.mean():+.3f} over {len(d)} runs, Wilcoxon p = {p:.3g}')
    out[f'file ({S.series})'] = a.loc[runs, 'checkpoint']
    out[f'file ({S_other.series})'] = b.loc[runs, 'checkpoint']
    deltas = [f'Δ {c}' for c in cols]
    levels = [c for c in out.columns if c not in deltas and not c.startswith('file')]
    sty = out.style.format('{:.3f}', subset=levels, na_rep='–').format('{:+.3f}', subset=deltas, na_rep='–')
    return pc._shade_deltas(sty, out, deltas).set_caption(f'Δ = {S_other.series} − {S.series}; ' + '; '.join(notes))
