"""Model loading shared by the evaluation suites (POLAR, SPINE, WiC, ...).

MODELS is the paper's model table: per (decomposition, n-gram) column, BASE plus the
values that each replace one BASE key. Every suite that loads through here sees the same
checkpoints, found the same way:

    import eval_utils as eu
    models, missing = eu.load_models()                   # name -> words + first-role factor rows
    for name, config, tk in eu.iter_decompositions():    # the full decompositions, one at a time
        ...

Two states of a run can be loaded (`checkpoint=`):
    'latest'      the highest {k}.pt in {stem}_checkpoints/: where training stopped (default)
    'model_file'  {stem}.pt itself: the best-semantic state, which training overwrites on every improvement
checkpoint_info() records which one was loaded and how far the run got.
load_same() loads a third: the runs that differ in one setting, each at the checkpoint they all have.
"""
from __future__ import annotations

import fnmatch
import json
import re
from functools import lru_cache
from pathlib import Path

import numpy as np

from tensormet.tucker_tensor import TuckerDecomposition
from tensormet.utils import DATA_DIR, to_np

from glove_baseline import load_glove, load_glove_nmf, load_word2vec, read_glove

# --- The model table ---------------------------------------------------------------
BASE = {"rank": 100,
        "method": "scSoftPlus",
        "ss_frac": 0.025,
        "dims": 10000,
        "name": "h100",
        "iters": 500,
        "tt_rank":100,
        "random_state": 1,  # training seed (init + subsample windows); files of seed n > 1 carry the prefix {name}_rs{n}
        }
# The X cells: (decomposition, n-gram) -> {key: values replacing BASE[key]}. Each column also has BASE itself.
MODELS = {
    ("tucker", 3): {"ss_frac": [0.1, 1], "method": ["countingLog", "countingLogEps"],
                    "rank": [200, 300], "dims": [20000]},
    ("tucker", 4): {"method": ["countingLog", "countingLogEps"], "dims": [20000], "random_state": [2, 3]},
    ("tt", 4): {"ss_frac": [0.1, 1], "method": ["countingLog", "countingLogEps"],
                "rank": [200, 300, 1000], "dims": [20000, 200000], "tt_rank": [50,200,300], "random_state": [2, 3]},
    ("tt", 5): {"ss_frac": [0.1], "method": ["countingLog", "countingLogEps"],
                "rank": [200, 300], "dims": [20000]},
    ("tt", 6): {"ss_frac": [0.1], "method": ["countingLog", "countingLogEps"],
                "rank": [200, 300], "dims": [20000]},
}
# Settings shared by every model (not in the table)
FIXED = dict(divergence="kl", shared_factors="all", solver="mu")
# Pretrained baselines, as published, on two vocabularies (see load_baselines):
#   'ours'  the best model's words:  glove_<w>, glove_nmf_<w> (NMF to rank w), w2v
#   'own'   the embedding's words:   glove_<w>_full, w2v_full
# GloVe 2024 Wiki+Gigaword, unzipped from glove.2024.wikigiga.<w>d.zip; word2vec from gensim-data.
GLOVE_DIR = DATA_DIR / "corpora"
GLOVE_PATH = GLOVE_DIR / "wiki_giga_2024_100_MFT20_vectors_seed_2024_alpha_0.75_eta_0.05.050_combined.txt"
GLOVE_WIDTHS = (100, 300)
W2V_NAME = "word2vec-google-news-300"
BASELINE_VOCABS = ("ours", "own")


def find_glove(width):
    """The GloVe 2024 file of `width`: GLOVE_PATH for 100, else the one wiki_giga_2024_<width>_*.txt."""
    if width == 100:
        return GLOVE_PATH
    found = sorted(GLOVE_DIR.glob(f"wiki_giga_2024_{width}_*.txt"))
    if len(found) != 1:
        raise FileNotFoundError(f"expected one wiki_giga_2024_{width}_*.txt in {GLOVE_DIR}, found {[p.name for p in found]}")
    return found[0]


def baseline_specs(glove_widths=GLOVE_WIDTHS, w2v=True, vocabs=BASELINE_VOCABS):
    """name -> (source, width, vocab, nmf) of each baseline asked for."""
    out = {}
    for w in glove_widths or ():
        if "ours" in vocabs:
            out[f"glove_{w}"] = ("glove", w, "ours", False)
            out[f"glove_nmf_{w}"] = ("glove", w, "ours", True)
        if "own" in vocabs:
            out[f"glove_{w}_full"] = ("glove", w, "own", False)
    if w2v:
        if "ours" in vocabs:
            out["w2v"] = ("w2v", 300, "ours", False)
        if "own" in vocabs:
            out["w2v_full"] = ("w2v", 300, "own", False)
    return out


def load_baselines(glove_widths=GLOVE_WIDTHS, w2v=True, vocabs=BASELINE_VOCABS, only=None):
    """name -> dict(path, config, words, E_raw), plus name -> error for a missing file.
    'ours': the words of TuckerDecomposition.load_best() that the embedding has (word2vec: as is, else
    capitalised). 'own': every word of the embedding in file order (most frequent first), as the
    published evaluations use it. No NMF on 'own': it tests non-negativity on our words only."""
    wanted = {n: s for n, s in baseline_specs(glove_widths, w2v, vocabs).items() if selected(n, only)}
    models, missing = {}, {}
    tk_best = TuckerDecomposition.load_best() if any(s[2] == "ours" for s in wanted.values()) else None
    kv = None
    for name, (source, width, vocab, nmf) in wanted.items():
        if source == "glove":
            try:
                path = find_glove(width)
            except FileNotFoundError as e:
                missing[name] = str(e)
                continue
            if vocab == "own":
                words, E = read_glove(path)
            else:
                emb = load_glove_nmf(path, like=tk_best, rank=width) if nmf else load_glove(path, like=tk_best)
                words, E = embedding_matrix(emb)
        else:
            path = W2V_NAME
            if kv is None:
                import gensim.downloader
                kv = gensim.downloader.load(W2V_NAME)
            if vocab == "own":
                words, E = list(kv.index_to_key), np.asarray(kv.vectors, dtype=np.float32)
            else:
                words, E = embedding_matrix(load_word2vec(like=tk_best, keyed_vectors=kv))
        config = {"baseline": name, "source": source, "width": width, "vocab": vocab}
        if vocab == "ours":
            config["like"] = str(tk_best.decomp_path)
        models[name] = dict(path=str(path), config=config, words=words, E_raw=E)
    return models, missing


def configs(ngrams=None, decompositions=None, table=MODELS, base=BASE):
    """The table's configs, column by column, base first; `ngrams` / `decompositions` pick columns."""
    seen = []
    for (decomp, n), variations in table.items():
        if (ngrams is not None and n not in ngrams) or (decompositions is not None and decomp not in decompositions):
            continue
        col = {"decomposition": decomp, "ngram": n, **base}
        for c in [col] + [{**col, k: v} for k, vals in variations.items() for v in vals]:
            if c not in seen:
                seen.append(c)
                yield c


def run_name(c):
    """Every varied key must appear here, or two configs share a name. dims only when not 10000,
    tt_rank only when not 100 and random_state only when not 1, so the names of results written
    before they were varied stay valid."""
    dims = "" if c["dims"] == 10000 else f"_d{c['dims']}"
    tt = "" if c.get("tt_rank", 100) == 100 else f"_tt{c['tt_rank']}"
    rs = "" if c.get("random_state", 1) == 1 else f"_rs{c['random_state']}"
    return f"{c['decomposition']}_{c['ngram']}g_r{c['rank']}_{c['method']}_ss{c['ss_frac']:g}{dims}{tt}{rs}"


def file_name_prefix(c):
    """The model files' name prefix: `name`, plus _rs<n> for a training seed other than 1 (h100_rs2_kl_...)."""
    rs = c.get("random_state", 1)
    return c["name"] if rs == 1 else f"{c['name']}_rs{rs}"


def selected(name, only):
    """True if `name` matches one of the shell-style patterns in `only` (None: everything)."""
    return only is None or any(fnmatch.fnmatchcase(name, pat) for pat in only)


# --- Loading ---------------------------------------------------------------------
CHECKPOINTS = ('latest', 'model_file')


def load_decomposition(c, checkpoint='latest'):
    """'latest': the run's latest checkpoint, or the model file if the run has no checkpoints
    (iter_decompositions leaves such runs out unless they finished).
    'model_file': the model file (best-semantic state). `tk.model_file` is the model file either way.
    TT with tt_rank 100 tries bond rank = rank, then 100; any other tt_rank is loaded as is.
    If `iters` has no file, the highest iteration on disk is used."""
    assert checkpoint in CHECKPOINTS, checkpoint
    kwargs = dict(FIXED, dataset=f"{c['ngram']}-gram-raw-bos-eos-fineweb-en_1B",
                  order=c["ngram"], method=c["method"], rank=c["rank"], dims=c["dims"],
                  subsample_frac=float(c["ss_frac"]),
                  decomposition=c["decomposition"], name=file_name_prefix(c))
    tt_rank = c.get("tt_rank", 100)
    tt_ranks = [None] if c["decomposition"] != "tt" else [c["rank"], 100] if tt_rank == 100 else [tt_rank]
    err = None
    for iters in (c["iters"], None):
        for tt_rank in dict.fromkeys(tt_ranks):
            try:
                tk = TuckerDecomposition.load_from_disk(**kwargs, iterations=iters, tt_rank=tt_rank)
            except FileNotFoundError as e:
                err = e
                continue
            tk.model_file = tk.decomp_path
            if checkpoint == 'latest':
                try:
                    tk.update_from_path()
                except FileNotFoundError:
                    pass  # no checkpoints: keep the model file (source 'model file' in checkpoint_info)
            return tk
    raise err


@lru_cache(maxsize=None)
def _run_records(decomp_dir):
    """Model file name -> its last record in the directory's runs.jsonl (written when a run finishes)."""
    path = Path(decomp_dir) / 'runs.jsonl'
    records = {}
    if path.exists():
        for line in path.read_text().splitlines():
            try:
                r = json.loads(line)
                records[Path(r['results']['model_path']).name] = r
            except (ValueError, KeyError, TypeError):
                continue
    return records


def _read_json(path):
    try:
        return json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return None


def semantic_history(model_file):
    """(primary key, [(iteration, score or None)]) of every semantic check of a run, from
    {stem}_fitness.json and {stem}_config.json; (None, []) if either is missing. Checks run every
    sem_check_every iterations, so check i (0-based) is iteration (i + 1) * sem_check_every; a resumed
    run can be misaligned by the checks after its last checkpoint."""
    model_file = Path(model_file)
    cfg = (_read_json(model_file.with_name(f'{model_file.stem}_config.json')) or {}).get('cfg') or {}
    scores = _read_json(model_file.with_name(f'{model_file.stem}_fitness.json'))
    ev = cfg.get('eval') or {}
    every = ev.get('sem_check_every')
    if not isinstance(scores, list) or not every:
        return None, []
    key = ev.get('sem_primary_key')  # training's rule (tucker_tensor: "Decide once which semantic metric")
    if key is None:
        err = ev.get('sem_error_type', 'full')
        key = 'average_rank_score' if err == 'all' else err[0] if isinstance(err, (list, tuple)) else err

    def value(s):
        v = s.get(key) if isinstance(s, dict) else s
        try:
            return float(v)
        except (TypeError, ValueError):
            return None

    return key, [((i + 1) * every, value(s)) for i, s in enumerate(scores)]


def best_semantic(model_file):
    """Iteration and score of the state the model file holds, from the fitness log: the first check
    that beat every earlier one and 0 (training's rule: `diff = sem_value - best_sem_score > 0`,
    best_sem_score starting at 0). None if no check ever did (the file then holds the final state)."""
    key, history = semantic_history(model_file)
    best = None
    for it, v in history:
        if v is not None and v > (best[1] if best else 0.0):
            best = (it, v)
    last = next(((it, v) for it, v in reversed(history) if v is not None), None)
    return dict(sem_key=key, best_iteration=best and best[0], best_score=best and best[1],
                last_check=last and last[0], last_score=last and last[1])


def checkpoint_info(model_file, loaded):
    """Which state of a run was loaded, and how far the run got.

    status: 'hit max_iters' (a checkpoint at the iteration cap), 'stopped early' (finished below
    the cap: patience), 'unfinished' (no finish record: still running, or killed), or
    'finished, no checkpoints'. saved_iteration is the iteration the model file holds, from
    runs.jsonl (best-semantic iteration; the last one if the run had no semantic evaluation)."""
    model_file, loaded = Path(model_file), Path(loaded)
    m = re.search(r'_(\d+)i$', model_file.stem)
    max_iters = int(m.group(1)) if m else None
    ckpt_dir = model_file.with_name(f'{model_file.stem}_checkpoints')
    ckpts = sorted(int(p.stem) for p in ckpt_dir.glob('*.pt') if p.stem.isdigit())
    last = ckpts[-1] if ckpts else None
    record = _run_records(str(model_file.parent)).get(model_file.name)
    finished = record is not None or model_file.with_name(f'{model_file.stem}_timing.json').exists()
    results = (record or {}).get('results', {})
    saved = results.get('best_iteration') or results.get('iterations')  # best_iteration: runs since 2026-09-24
    if last is not None and max_iters is not None and last >= max_iters:
        status = 'hit max_iters'
    elif not finished:
        status = 'unfinished'
    elif last is None:
        status = 'finished, no checkpoints'
    else:
        status = 'stopped early'
    sem = best_semantic(model_file)
    # the model file's iteration: the record training writes with it (runs since 2026-09-24), else
    # runs.jsonl when the run finished, else the fitness log
    rec = _read_json(model_file.with_name(f'{model_file.stem}_best.json')) or {}
    if rec.get('iteration') is not None:
        best_it, best_from = rec['iteration'], 'best.json'
    elif saved is not None:
        best_it, best_from = saved, 'runs.jsonl'
    else:
        best_it, best_from = sem['best_iteration'], 'fitness log' if sem['best_iteration'] is not None else None
    from_ckpt = loaded.parent == ckpt_dir
    return dict(source='checkpoint' if from_ckpt else 'model file',
                iteration=int(loaded.stem) if from_ckpt else best_it,
                best_iteration=best_it, best_from=best_from,
                max_iters=max_iters, last_checkpoint=last, saved_iteration=saved,
                n_checkpoints=len(ckpts), finished=finished, status=status, **sem_scores(sem),
                model_file=str(model_file), loaded=str(loaded))


def sem_scores(sem):
    """The fitness-log fields checkpoint_info keeps (the log's own best iteration as fitness_best_iteration)."""
    return dict(sem_key=sem['sem_key'], sem_best=sem['best_score'], sem_last=sem['last_score'],
                last_check=sem['last_check'], fitness_best_iteration=sem['best_iteration'])


def embedding_matrix(tk):
    """Words and first-role factor rows: the rows fetch_single_latent returns, gathered in one indexing call."""
    role = tk.roles[0]
    words = list(tk.vocab[f'vocab_{role}'])
    w2i = tk.vocab[f'{role}2i']
    factor = np.asarray(to_np(tk.factors[tk.get_role_index(role)]))
    return words, np.asarray(factor[[w2i[w] for w in words]], dtype=np.float32)


def iter_decompositions(ngrams=None, decompositions=None, only=None, missing=None, checkpoint='latest'):
    """Yields (name, config, decomposition) for each table model on disk, one loaded at a time.
    Models with no file go into the `missing` dict (name -> error), if one is passed."""
    names = set()
    for c in configs(ngrams, decompositions):
        name = run_name(c)
        assert name not in names, f"run_name collision: {name}"
        names.add(name)
        if not selected(name, only):
            continue
        try:
            tk = load_decomposition(c, checkpoint)
        except (FileNotFoundError, ValueError) as e:
            if missing is not None:
                missing[name] = str(e)
            continue
        info = checkpoint_info(tk.model_file, tk.decomp_path)
        if info['n_checkpoints'] == 0 and not info['finished']:  # stopped by walltime before any checkpoint
            if missing is not None:
                missing[name] = (f"left out: no checkpoint before the walltime stop "
                                 f"(model file at iteration {info['best_iteration'] or '?'})")
            del tk
            continue
        yield name, {**FIXED, **c}, tk


def load_models(ngrams=None, decompositions=None, only=None, glove_widths=GLOVE_WIDTHS, w2v=True,
                baseline_vocabs=BASELINE_VOCABS, checkpoint='latest'):
    """name -> dict(path, config, words, E_raw, checkpoint), plus name -> error for models with no file.
    The baselines (see load_baselines) have no 'checkpoint'; glove_widths=(), w2v=False leave them out."""
    models, missing = {}, {}
    for name, config, tk in iter_decompositions(ngrams, decompositions, only, missing, checkpoint):
        words, E = embedding_matrix(tk)
        models[name] = dict(path=str(tk.decomp_path), config=config, words=words, E_raw=E,
                            checkpoint=checkpoint_info(tk.model_file, tk.decomp_path))
        del tk

    baselines, baselines_missing = load_baselines(glove_widths, w2v, baseline_vocabs, only)
    models.update(baselines)
    missing.update(baselines_missing)
    return models, missing


# --- Series 'same': matched runs at one common checkpoint ---------------------------------
def checkpoint_iterations(model_file):
    """Iterations of the run's {k}.pt checkpoints, ascending ([] if it has none)."""
    model_file = Path(model_file)
    ckpt_dir = model_file.with_name(f'{model_file.stem}_checkpoints')
    return sorted(int(p.stem) for p in ckpt_dir.glob('*.pt') if p.stem.isdigit())


def _drop_state(tk):
    """Frees a decomposition's arrays until update_from_path reloads them (TT: `core` is a read-only property)."""
    if hasattr(tk, 'tt_cores'):
        tk.tt_cores = tk._core_cache = None
    else:
        tk.core = None
    tk.factors = None


def _same_key(c, factor):
    """Every setting but `factor`: the runs sharing it form one set."""
    return tuple(sorted((k, v) for k, v in c.items() if k not in (factor, 'iters', 'name')))


def load_same(factor='ss_frac', ngrams=None, decompositions=None, only=None):
    """The table's runs that differ only in `factor` (a set), each loaded at the highest iteration every run of
    its set has a checkpoint of: normally the shortest run's last checkpoint, so the longer runs are compared at
    the iteration the shortest reached. Returns what load_models does, without baselines; `checkpoint` also holds
    common_iteration and common_set. Runs in no set of two with checkpoints are left out."""
    groups = {}
    for c in configs(ngrams, decompositions):
        if selected(run_name(c), only):
            groups.setdefault(_same_key(c, factor), []).append(run_name(c))
    wanted = [n for names in groups.values() if len(names) > 1 for n in names]
    missing, runs = {}, {}
    for name, config, tk in iter_decompositions(ngrams, decompositions, wanted, missing, 'model_file'):
        _drop_state(tk)  # reloaded at the common iteration below
        runs[name] = (config, tk, set(checkpoint_iterations(tk.model_file)))

    common = {}
    for names in groups.values():
        for n in names:
            if n in runs and not runs[n][2]:
                missing[n] = 'left out: no checkpoints to cut to a common iteration'
        names = [n for n in names if n in runs and runs[n][2]]
        if len(names) == 1:
            missing[names[0]] = f'left out: no other run of its {factor} set with checkpoints'
        if len(names) < 2:
            continue
        shared = set.intersection(*(runs[n][2] for n in names))
        for n in names:
            if shared:
                common[n] = (max(shared), names)
            else:
                missing[n] = f"left out: no checkpoint iteration common to {', '.join(names)}"

    models = {}
    for name, (config, tk, _) in runs.items():
        if name not in common:
            continue
        it, names = common[name]
        tk.update_from_path(it)
        words, E = embedding_matrix(tk)
        models[name] = dict(path=str(tk.decomp_path), config=config, words=words, E_raw=E,
                            checkpoint={**checkpoint_info(tk.model_file, tk.decomp_path),
                                        'common_iteration': it, 'common_set': names})
        _drop_state(tk)
    return models, missing


def describe_models(models, missing):
    """Prints which state of each run was loaded, how far the run got, and what is missing."""
    print("\nloaded (* = requested iters not on disk, the run with the most iterations used; ? = unknown)\n"
          "  iter = iteration of the loaded state; best = iteration the model file holds (best semantic score);\n"
          "  last = last checkpoint; sem best / last = primary semantic score at the best / last check")
    print(f"  {'model':42s} {'source':10s} {'iter':>5s} {'best':>5s} {'last':>5s} {'max':>5s}  "
          f"{'status':24s} {'sem best':>8s} {'last':>8s}  size")
    for name, m in models.items():
        size = f"{len(m['words'])} x {m['E_raw'].shape[1]}"
        ck = m.get('checkpoint')
        if ck is None:
            print(f"  {name:42s} {'baseline':10s} {'':41s}{'':24s} {'':19s}  {size}  {Path(m['path']).name}")
            continue
        flag = "*" if ck['max_iters'] != m['config'].get('iters') else " "
        its = [('?' if v is None else str(v))
               for v in (ck['iteration'], ck.get('best_iteration'), ck['last_checkpoint'], ck['max_iters'])]
        sems = [('?' if v is None else f"{v:.4f}") for v in (ck.get('sem_best'), ck.get('sem_last'))]
        print(f" {flag}{name:42s} {ck['source']:10s} {its[0]:>5s} {its[1]:>5s} {its[2]:>5s} {its[3]:>5s}  "
              f"{ck['status']:24s} {sems[0]:>8s} {sems[1]:>8s}  {size}")
    keys = {ck['sem_key'] for m in models.values() if (ck := m.get('checkpoint')) and ck.get('sem_key')}
    if keys:
        print(f"  semantic score: {', '.join(sorted(keys))}")
    print("missing:")
    for name, e in missing.items():
        print(f"  {name:42s} {e}")
