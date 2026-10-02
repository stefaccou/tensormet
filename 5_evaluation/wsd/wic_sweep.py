"""WiC over every model of the table (eval_utils.MODELS) and the GloVe / word2vec baselines.

The method of wic.ipynb (see wic_eval.py): one cosine per item, one threshold θ fit on train,
accuracy on dev and test. Per model, the settings in MODEL_METHODS:

    static | window=0         the factor row of the target itself
    contextual | rounds=1     ContextualEncoder, n-gram windows around the target
    contextual | rounds=3     context up to 3·(n-1) tokens away
    excluded                  what the other words of each window predict at the target

Baselines (GloVe 2024 at every width of eval_utils.GLOVE_WIDTHS, word2vec-google-news-300), static
windows 0 / 2 / 5 / sentence, in two blocks:

    ours  per model, restricted to that model's words, next to that model's own rows. The
          (common) block is then the pairs every row of that model can score: the like-for-like
          comparison with our settings.
    own   once, on each embedding's whole vocabulary (its own capabilities), independent of any
          model. Compare its (all) numbers with a model's (all) numbers.

NNSE (the released depDoc vectors, NNSE_WIDTHS; Murphy et al. 2012) is scored the same way, but in a batch
of its own, so the rows scored before it was added stay valid. Its (common) numbers are therefore among
the NNSE rows only: compare NNSE by its (all) numbers, as eval.ipynb section 6 does.

Every row is one line of out/sweep.jsonl, with its per-pair results on dev and test (item_fields), which
the paired tests in eval.ipynb section 6 need. Rerunning skips a model whose loaded state (checkpoint or
model file) was already scored with the same settings, so a stopped sweep continues where it stopped; a
model scored before NNSE was added gets only its NNSE rows.

On ampere:
    screen -S wic
    conda activate ccl
    cd ~/pc/5_evaluation/wsd
    python wic_sweep.py --dry-run            # load and list the models, score nothing
    python wic_sweep.py                      # 'latest' series: each run's latest checkpoint
    python wic_sweep.py --series best        # the model file (best-semantic state)
    # only the default runs eval.ipynb section 6 reports:
    python wic_sweep.py --only tucker_3g_r100_scSoftPlus_ss0.025 tucker_4g_r100_scSoftPlus_ss0.025 \\
        tt_4g_r100_scSoftPlus_ss0.025 tt_5g_r100_scSoftPlus_ss0.025 tt_6g_r100_scSoftPlus_ss0.025 \\
        'tt_4g_r100_scSoftPlus_ss0.025_rs*'
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
import time
from datetime import datetime
from pathlib import Path

NB_DIR = Path(__file__).resolve().parent
EVAL_DIR = NB_DIR.parent
for d in (NB_DIR, EVAL_DIR, EVAL_DIR / 'polar'):  # polar/: method_baselines (NNSE reader)
    if str(d) not in sys.path:
        sys.path.insert(0, str(d))

import numpy as np  # noqa: E402
import torch  # noqa: E402

import eval_utils as eu  # noqa: E402
import wic_eval as we  # noqa: E402
from glove_baseline import _as_baseline, load_glove, load_word2vec  # noqa: E402
from tensormet.experimental.contextual import ContextualEncoder, NgramSpec  # noqa: E402
from tensormet.utils import select_gpu  # noqa: E402

OUT = NB_DIR / 'out'
RESULTS_PATH = OUT / 'sweep.jsonl'
DATA = NB_DIR / 'data'

# (label, kind, arguments)
MODEL_METHODS = [
    ('static | window=0', 'static', dict(window=0)),
    ('contextual | rounds=1', 'contextual', dict(rounds=1)),
    ('contextual | rounds=3', 'contextual', dict(rounds=3)),
    ('excluded', 'excluded', dict()),
]
WINDOWS = (0, 2, 5, None)  # None: the whole sentence
# Sentence-boundary and placeholder entries are not words
SPECIAL = {'<s>', '</s>', '<BOS>', '<EOS>', '~', ''}
# Every table model is trained on raw (lowercased) text; this tokenisation reads only `raw`
SPEC = NgramSpec(order=4, raw=True, padded=False)
OWN_RUN = 'baselines (own vocab)'
# Bumped when a row gains fields, so older rows are rescored. 2: per-pair results (item_fields)
ROW_VERSION = 2
# Released NNSE widths, scored in their own batch (method names 'nnse-<w>d | ...'); not in settings_key
NNSE_WIDTHS = (300, 1000)


def settings_key(glove_widths):
    """Changes whenever the scored settings change, so old rows are not reused for new settings."""
    s = json.dumps([MODEL_METHODS, WINDOWS, sorted(glove_widths), ROW_VERSION], sort_keys=True, default=str)
    return hashlib.sha1(s.encode()).hexdigest()[:10]


def _bits(mask):
    return ''.join('1' if v else '0' for v in mask)


def item_fields(row, sims, gold):
    """Per pair of dev and test, as '0'/'1' strings in WiC order: correct_<split> (the prediction with
    the all-pairs θ; an unscored pair gets the majority label) and ok_<split> (the method scored it).
    Paired tests across models (eval.ipynb section 11) need these."""
    out = {}
    for split in ('dev', 'test'):
        scores, ok = sims[row['method']][split]
        pred = we.predict(scores, ok, row['all_theta'], row['all_majority'])
        out[f'correct_{split}'] = _bits(pred == gold[split])
        out[f'ok_{split}'] = _bits(ok)
    return out


def window_label(w):
    return f"window={'sentence' if w is None else w}"


# --- Scoring --------------------------------------------------------------------
def model_sims(tk, enc, slots, words):
    """label -> split -> (scores, ok) for the MODEL_METHODS of one model."""
    table = we.StaticTable(tk, words)
    out = {}
    for label, kind, kw in MODEL_METHODS:
        out[label] = {}
        for split, sl in slots.items():
            if kind == 'static':
                out[label][split] = we.static_similarities(table, sl, kw['window'])
            elif kind == 'contextual':
                out[label][split] = we.contextual_similarities(enc, sl, **kw)
            else:
                out[label][split] = we.excluded_similarities(enc, sl, **kw)
    return out


def baseline_sims(baselines, slots, words=None):
    """label -> split -> (scores, ok) for every baseline and window; `words` limits the lookup."""
    out = {}
    for name, emb in baselines.items():
        table = we.StaticTable(emb, words)
        for w in WINDOWS:
            out[f'{name} | static | {window_label(w)}'] = {
                split: we.static_similarities(table, sl, w) for split, sl in slots.items()}
    return out


def append_rows(rows):
    OUT.mkdir(parents=True, exist_ok=True)
    with RESULTS_PATH.open('a', encoding='utf-8') as fh:
        for row in rows:
            fh.write(json.dumps(row, default=str) + '\n')


def done_keys():
    """(run, loaded path, settings) -> the methods already in the results file."""
    if not RESULTS_PATH.is_file():
        return {}
    keys = {}
    for line in RESULTS_PATH.read_text(encoding='utf-8').splitlines():
        try:
            r = json.loads(line)
            keys.setdefault((r['run'], r['loaded'], r['settings']), set()).add(r['method'])
        except (ValueError, KeyError):
            continue
    return keys


def has_batch(done, key, nnse):
    """Whether `key` already has rows of the NNSE batch (nnse=True) or of the main batch (nnse=False)."""
    return any(m.startswith('nnse-') == nnse for m in done.get(key, ()))


def load_baselines(glove_widths, words):
    """name -> embedding over `words` (every WiC token): the same as loading the whole file.
    word2vec looks each word up as is, then capitalised (glove_baseline.load_word2vec)."""
    import gensim.downloader
    out = {}
    for w in glove_widths:
        path = eu.find_glove(w)
        out[f'glove-{w}d'] = load_glove(path, roles=['w'], vocab=words)
    out['word2vec-300d'] = load_word2vec(roles=['w'], vocab=words,
                                         keyed_vectors=gensim.downloader.load(eu.W2V_NAME))
    return out


def load_nnse(words):
    """name -> the released NNSE vectors over `words`, read as polar/method_baselines reads them
    (DATA_DIR/corpora/nnse/depDocNNSE<w>.tab.zip). All-zero rows are words NNSE does not know."""
    import method_baselines as mb
    out, words = {}, set(words)
    for w in NNSE_WIDTHS:
        names, E = mb.read_txt(mb.NNSE_DIR / mb.NNSE_FILE.format(w))
        keep = [i for i, t in enumerate(names) if t in words and E[i].any()]
        out[f'nnse-{w}d'] = _as_baseline([names[i] for i in keep], E[keep], ['w'], mb.NNSE_FILE.format(w))
    return out


def print_nnse(name, rows):
    """Per NNSE width, the test accuracy of its window best on dev."""
    best = {}
    for r in rows:
        src = r['method'].split(' | ')[0]
        if src not in best or r['all_acc_dev'] > best[src]['all_acc_dev']:
            best[src] = r
    print(f'  {name:45s} NNSE test: ' + '  '.join(f"{r['method']} {r['all_acc_test']:.3f}" for r in best.values()))


# --- Main -------------------------------------------------------------------------
def build_parser():
    p = argparse.ArgumentParser(description='WiC over the model table and the baselines.')
    p.add_argument('--series', default='latest', choices=['latest', 'best'],
                   help="latest: each run's latest checkpoint; best: the model file (best-semantic state)")
    p.add_argument('--only', nargs='+', default=None, metavar='PATTERN',
                   help="model names to run (shell-style patterns), e.g. 'tt_4g_*'")
    p.add_argument('--ngrams', nargs='+', type=int, default=None, help='table columns of these orders')
    p.add_argument('--glove-widths', nargs='+', type=int, default=list(eu.GLOVE_WIDTHS))
    p.add_argument('--gpus', nargs='+', type=int, default=[1, 2])
    p.add_argument('--no-resume', action='store_true', help='rescore models already in the results file')
    p.add_argument('--dry-run', action='store_true', help='load and list the models; score nothing')
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    device = select_gpu(args.gpus)
    checkpoint = {'latest': 'latest', 'best': 'model_file'}[args.series]
    stamp = datetime.now().isoformat(timespec='seconds')
    settings = settings_key(args.glove_widths)
    common = dict(sweep=stamp, series=args.series, settings=settings, host=platform.node())

    root = we.download_wic(DATA)
    data = {s: we.load_wic(root, s) for s in we.SPLITS}
    gold = we.labels(data)
    slots = {s: we.locate_targets(items, SPEC) for s, items in data.items()}
    wic_words = sorted({t for pairs in slots.values() for pair in pairs
                        for slot in pair if slot is not None for t in slot[0].tokens})
    print(f'[{stamp}] WiC: {", ".join(f"{s} {len(v)}" for s, v in data.items())} items, '
          f'{len(wic_words)} word forms; results -> {RESULTS_PATH}')

    done = {} if args.no_resume else done_keys()
    baselines = None if args.dry_run else load_baselines(args.glove_widths, wic_words)
    nnse = None if args.dry_run else load_nnse(wic_words)

    # Own vocabulary: independent of the models, scored once per settings; NNSE in its own batch
    own_key = (OWN_RUN, '-', settings)
    for label, embs, is_nnse in (('baselines', baselines, False), ('NNSE', nnse, True)):
        if has_batch(done, own_key, is_nnse):
            print(f'{OWN_RUN}, {label}: already in {RESULTS_PATH.name}')
        elif not args.dry_run:
            sims = baseline_sims(embs, slots)
            rows = we.run_methods(sims, gold)
            append_rows([{**common, 'run': OWN_RUN, 'block': 'own', 'loaded': '-',
                          'sources': {n: b.source for n, b in embs.items()}, **r,
                          **item_fields(r, sims, gold)} for r in rows])
            print(f'{OWN_RUN}, {label}: {len(rows)} rows')

    missing = {}
    n_done = n_skipped = 0
    t0 = time.perf_counter()
    for name, config, tk in eu.iter_decompositions(args.ngrams, only=args.only, missing=missing,
                                                   checkpoint=checkpoint):
        info = eu.checkpoint_info(tk.model_file, tk.decomp_path)
        loaded = str(tk.decomp_path)
        where = f"{info['source']} it {info['iteration']} ({info['status']})"
        key = (name, loaded, settings)
        todo_main, todo_nnse = not has_batch(done, key, False), not has_batch(done, key, True)
        if not (todo_main or todo_nnse):
            print(f'  skip  {name:45s} {where}: already scored')
            n_skipped += 1
        elif args.dry_run:
            print(f'  todo  {name:45s} {where}: ' + ' + '.join(b for b, t in (('main', todo_main),
                                                                            ('NNSE', todo_nnse)) if t))
        else:
            t = time.perf_counter()
            # The spec by hand: after update_from_path, decomp_path is the checkpoint, not
            # <dataset>/decomposition/, so ContextualEncoder cannot read it from the path.
            # Every table model is on '{n}-gram-raw-bos-eos-...' (eval_utils.load_decomposition).
            spec = NgramSpec(order=config['ngram'], raw=True, padded=True)
            enc = ContextualEncoder(tk, spec=spec, device=device)
            words = set(enc.word_index) - SPECIAL
            meta = dict(common, run=name, loaded=loaded, model_file=str(tk.model_file),
                        config=config, checkpoint=info, n_words=len(words))
            if todo_main:
                ours = model_sims(tk, enc, slots, words)
                ours_labels = set(ours)
                ours.update(baseline_sims(baselines, slots, words))
                rows = we.run_methods(ours, gold)
                append_rows([{**meta, 'block': 'model' if r['method'] in ours_labels else 'ours', **r,
                              **item_fields(r, ours, gold)} for r in rows])
                # per kind (static / contextual / excluded), the setting best on dev; kinds are not chosen between
                best = {}
                for r in rows:
                    kind = r['method'].split(' ')[0]
                    if r['method'] in ours_labels and (kind not in best or r['all_acc_dev'] > best[kind]['all_acc_dev']):
                        best[kind] = r
                print(f'[{datetime.now():%H:%M:%S}] {name:45s} {where}  test: '
                      + '  '.join(f"{r['method']} {r['all_acc_test']:.3f}" for r in best.values())
                      + f'  ({time.perf_counter() - t:.0f} s)')
            if todo_nnse:  # NNSE on this model's words: block 'ours', like the other baselines
                sims = baseline_sims(nnse, slots, words)
                rows = we.run_methods(sims, gold)
                append_rows([{**meta, 'block': 'ours', **r, **item_fields(r, sims, gold)} for r in rows])
                print_nnse(name, rows)
            n_done += 1
            del enc
        del tk
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print(f'\n{n_done} models scored, {n_skipped} already done, in {(time.perf_counter() - t0) / 60:.1f} min')
    if missing:
        print('missing:')
        for name, e in missing.items():
            print(f'  {name:45s} {e}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
