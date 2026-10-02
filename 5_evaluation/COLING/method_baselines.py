"""The POLAR evaluation of polar_sweep.py, run on interpretable embedding methods instead of tensormet models.

    polar_<src>_k<K>   POLAR (Mathew et al., 2020): main.ipynb's transform, ported (antonym sets,
                       closest-antonym selection, random dimension order, pinv projection, unit rows),
                       applied to <src>; K polar dimensions (main.ipynb: dim_size = 500)
    spine_<src>        SPINE (Subramanian et al., 2018): the released SPINE vectors, as downloaded
    spowv_<src>        SPOWV (Faruqui et al., 2015), SPINE's baseline, from the same release
    nnse_<w>           NNSE (Murphy et al., 2012): the released depDoc model (dependency + document
                       co-occurrence, ClueWeb09) of width <w>, as downloaded; 35,362 (300) / 35,529 (1000)
                       lowercase words
    sinr_bnc           SINr (Prouteau et al., 2021): the English model shipped in the SINr repo
                       (notebooks/sinrvec_bnc.pk: BNC, lemmas of nouns/verbs/adjectives only), as is
    ..._full           the same on the method's whole vocabulary (POLAR: every word of <src>, as
                       main.ipynb transforms them; the others: the whole file), not only our words

Memory: polar_w2v_k500_full is 3M x 500 (6 GB); polar_glove_k500_full
1.29M x 500. Leave them out with --only or --vocabs ours when the machine is busy.

<src> is glove (GloVe 2024 100d, the file of polar_sweep's glove_100 rows) or w2v (Google News word2vec,
POLAR's paper input). Without _full, a method keeps only the words of TuckerDecomposition.load_best(),
the vocabulary the glove_<w> / glove_nmf_<w> / w2v rows use; either way it then goes through
polar_sweep's pipeline unchanged: same variants, tasks, grids, selection, results log
(tensormet_eval.jsonl) and resume. The POLAR transform is always fit on the whole source vocabulary
(antonyms outside our vocabulary count); only the rows evaluated differ.

Outputs in polar_results/: methods_<stamp>.json (manifest), methods_<stamp>.log, fit_times_methods_<stamp>.log.
Read them with  man = ps.load_manifest(prefix='methods').

On ampere, inside screen:
    cd 5_evaluation/COLING
    python method_baselines.py --dry-run
    python method_baselines.py
    python method_baselines.py --export-txt      # also write SPINE-format files for spine_eval
"""
from __future__ import annotations

import io
import pickle
import random
import sys
import zipfile
from collections import defaultdict
from contextlib import contextmanager
from pathlib import Path

import numpy as np

import polar_sweep as ps
from eval_utils import GLOVE_PATH, selected
from glove_baseline import _as_baseline, read_glove

from tensormet.tucker_tensor import TuckerDecomposition
from tensormet.utils import DATA_DIR

ANTONYM_DIR = ps.THIRD_PARTY_DIR / 'POLAR/Antonym_sets'
ANTONYM_FILES = ['LenciBenotto.val', 'LenciBenotto.test', 'EVALution.val', 'EVALution.test']  # main.ipynb's order
W2V_NAME = 'word2vec-google-news-300'
POLAR_SOURCES = ['glove', 'w2v']
POLAR_DIMS = [500]  # main.ipynb's dim_size
POLAR_SEED = 42  # main.ipynb seeds `random` with 42 before shuffling the dimension order
# The SPINE release (README: "Word Embeddings", Google Drive folder), downloaded by hand: SPINE and its
# baseline SPOWV (Faruqui et al., 2015), each on GloVe and word2vec. A missing file is reported, not fatal.
SPINE_DIR = DATA_DIR / 'corpora' / 'spine'
RELEASED_METHODS = {'spine': 'SPINE', 'spowv': 'SPOWV'}  # name prefix -> file prefix
SPINE_SOURCES = {'glove': 'glove', 'w2v': 'word2vec'}  # source -> file suffix
SPINE_PATHS = {(m, src): SPINE_DIR / f'{f}_{s}.txt'
               for m, f in RELEASED_METHODS.items() for src, s in SPINE_SOURCES.items()}
# NNSE's released depDoc models (the paper's full model), downloaded by hand and read from the zip
NNSE_DIR = DATA_DIR / 'corpora' / 'nnse'
NNSE_URL = 'https://www.cs.cmu.edu/~bmurphy/NNSE/'
NNSE_FILE = 'depDocNNSE{}.tab.zip'
NNSE_WIDTHS = [300, 1000]  # 300: the width of r300, glove_300, w2v; 1000: the width later papers use
# SINr has no English release besides the example model in its repo (notebooks/sinrvec_en.ipynb there)
SINR_PATH = ps.THIRD_PARTY_DIR / 'sinr/notebooks/sinrvec_bnc.pk'
RELEASED = ['spine', 'spowv', 'nnse', 'sinr']
# --export-txt only: the SPINE harness, which is not part of COLING/
SPINE_SUITE_DIR = ps.COLING_DIR.parent / 'spine'
TXT_DIR = SPINE_SUITE_DIR / 'out' / 'methods'
EXPORT_MAX_WORDS = 200_000  # --export-txt skips larger embeddings (the full-vocabulary POLAR ones)
VOCABS = ('ours', 'own')


# --- POLAR: port of main.ipynb ---------------------------------------------------------
def raw_antonym_pairs(contains):
    """Cell 11: antonym pairs of the four sets whose words are in the embedding, lowercased, deduplicated."""
    pairs = []
    for fname in ANTONYM_FILES:
        with open(ANTONYM_DIR / fname) as fp:
            for line in fp:
                parts = line.split()
                if parts[3] == 'antonym':
                    word1 = parts[0].split('-')[0]
                    word2 = parts[1].split('-')[0]
                    if contains(word1) and contains(word2):
                        pairs.append((word1.strip().lower(), word2.strip().lower()))
    return list(dict.fromkeys(pairs))


def antonym_words():
    """Every word the antonym sets could contribute, as written and lowercased."""
    words = set()
    for fname in ANTONYM_FILES:
        with open(ANTONYM_DIR / fname) as fp:
            for line in fp:
                parts = line.split()
                if parts[3] == 'antonym':
                    for p in parts[:2]:
                        w = p.split('-')[0]
                        words |= {w, w.strip().lower()}
    return words


def closest_antonyms(pairs, vec):
    """Cell 12: per alphabetically first word, the partner with the lowest |cosine|."""
    partners = defaultdict(list)
    for w1, w2 in pairs:
        a, b = (w1, w2) if w1 < w2 else (w2, w1)
        partners[a].append(b)

    def abs_cos(x, y):
        return abs(float(x @ y) / (np.linalg.norm(x) * np.linalg.norm(y)))

    return [(a, min(bs, key=lambda b: abs_cos(vec(a), vec(b)))) for a, bs in partners.items()]


def polar_projection(pairs, vec, k, seed=POLAR_SEED):
    """Cells 15, 21, 34: antonym difference vectors, random order, pinv of the first k.
    Returns the (k', d) matrix mapping a (unit) word vector into polar space; k' = min(k, #pairs)."""
    directions = np.array([vec(w1) - vec(w2) for w1, w2 in pairs])
    order = list(range(len(directions)))
    random.Random(seed).shuffle(order)
    return np.linalg.pinv(directions[order[:k]].T)


def polar_rows(P, E, chunk=200_000):
    """Cells 5 + 30: project the unit-normalised word vectors, then unit-normalise each row.
    A row's scale cancels in the final normalisation, so E is projected as is; chunked float32,
    so the full-vocabulary versions fit in memory. A zero row stays zero."""
    Pt = P.astype(np.float32).T
    out = np.empty((len(E), Pt.shape[1]), dtype=np.float32)
    for i in range(0, len(E), chunk):
        Z = np.asarray(E[i:i + chunk], dtype=np.float32) @ Pt
        norm = np.linalg.norm(Z, axis=1, keepdims=True)
        norm[norm == 0] = 1
        out[i:i + chunk] = Z / norm
    return out


def _unit(v):
    """Cell 5: POLAR normalises the input embedding first."""
    return v / np.linalg.norm(v)


# --- Sources ----------------------------------------------------------------------------
# Each returns dict(path, contains, vec, ours=(words, E), own=(words, E) or None). contains/vec answer
# for the whole embedding; 'ours' is our vocabulary's rows, 'own' (full=True) every row in file order.
def source_glove(vocab, full=False):
    """GloVe 2024 100d. Without `full`, only our words and the antonym-set words are read (all POLAR needs)."""
    words, E = read_glove(GLOVE_PATH, keep=None if full else set(vocab) | antonym_words(), verbose=False)
    index = {w: i for i, w in enumerate(words)}
    ours = [w for w in vocab if w in index]
    print(f"{len(ours)} of our {len(vocab)} words in {GLOVE_PATH.name}" + (f"; {len(words)} in all" if full else ""))
    return dict(path=str(GLOVE_PATH), contains=index.__contains__, vec=lambda w: _unit(E[index[w]]),
                ours=(ours, E[[index[w] for w in ours]]), own=(words, E) if full else None)


def source_w2v(vocab, full=False):
    """Google News word2vec; our lowercase words looked up as is, then capitalised (glove_baseline's rule)."""
    import gensim.downloader
    kv = gensim.downloader.load(W2V_NAME)
    ours, forms = [], []
    for w in vocab:
        form = next((f for f in (w, w.capitalize()) if f in kv), None)
        if form is not None:
            ours.append(w)
            forms.append(form)
    print(f"{len(ours)} of our {len(vocab)} words in {W2V_NAME}")
    return dict(path=W2V_NAME, contains=kv.__contains__, vec=lambda w: _unit(kv[w]),
                ours=(ours, kv[forms]), own=(list(kv.index_to_key), kv.vectors) if full else None)


SOURCES = {'glove': source_glove, 'w2v': source_w2v}


@contextmanager
def _open_text(path):
    """A text file, or the single file in a zip (NNSE ships one .tab per zip), read in place.
    errors='replace': a stray non-UTF-8 word must not stop the run."""
    if Path(path).suffix != '.zip':
        with open(path, encoding='utf-8', errors='replace') as fh:
            yield fh
        return
    with zipfile.ZipFile(path) as zf:
        (member,) = [n for n in zf.namelist() if not n.endswith('/')]
        with io.TextIOWrapper(zf.open(member), encoding='utf-8', errors='replace') as fh:
            yield fh


def read_txt(path):
    """Released vectors as text (SPINE, NNSE), 'word v1 v2 ...' per line as SPINE's scripts split it.
    Header lines before the first vector are skipped: word2vec's 'count dim', or NNSE's
    '#' lines ('#singVals 1 1 ...', '#target d1 d2 ...' in the 300 file; '#labels d1 d2 ...' in the 1000)."""
    words, rows = [], []
    with _open_text(path) as fh:
        for i, line in enumerate(fh):
            parts = line.split()
            if not parts or (not rows and (parts[0].startswith('#')
                                           or (len(parts) == 2 and all(p.isdigit() for p in parts)))):
                continue
            try:
                rows.append(np.array(parts[1:], dtype=np.float32))
            except ValueError as e:
                raise ValueError(f'{path} line {i + 1}: {e}') from None
            words.append(parts[0])
    widths = {len(r) for r in rows}
    if len(widths) != 1:
        raise ValueError(f'{path} mixes vector widths {sorted(widths)}')
    print(f"{len(words)} vectors of dim {widths.pop()} read from {Path(path).name}")
    return words, np.stack(rows)


class _Stub:
    """Stands in for every class of the SINr pickle that _VectorsOnly does not build; keeps nothing."""

    def __init__(self, *args, **kwargs):
        pass

    def __setstate__(self, state):
        pass


class _VectorsOnly(pickle.Unpickler):
    """Builds numpy and scipy objects and plain builtin types only. Anything else (sinrvec_bnc.pk: the
    fitted sklearn NearestNeighbors) becomes a _Stub, so neither the sinr package nor the sklearn it was
    saved with is needed, and nothing outside numpy/scipy is imported or called. Protocol 2 writes the
    Python 2 module names (__builtin__, copy_reg)."""
    MODULES = {'numpy', 'scipy', '_codecs', 'copyreg', 'copy_reg'}
    BUILTINS = {'set', 'frozenset', 'list', 'dict', 'tuple', 'bytes', 'bytearray', 'str', 'unicode',
                'int', 'long', 'float', 'complex', 'bool', 'slice', 'range', 'xrange', 'object'}

    def find_class(self, module, name):
        if module.split('.')[0] in self.MODULES or (module in ('builtins', '__builtin__') and name in self.BUILTINS):
            return super().find_class(module, name)
        return _Stub


def read_sinr(path):
    """A saved SINrVectors (SINrVectors.save pickles its __dict__): its vocabulary and vectors, dense."""
    with open(path, 'rb') as fh:
        d = _VectorsOnly(fh).load()
    words, E = list(d['vocab']), np.asarray(d['vectors'].toarray(), dtype=np.float32)
    print(f"{len(words)} vectors of dim {E.shape[1]} read from {Path(path).name}")
    return words, E


def _name(base, vocab):
    return base + ('_full' if vocab == 'own' else '')


def released_files(args):
    """name -> (method, path, reader, how to get the file) of every requested released embedding."""
    out = {}
    for (m, src), path in SPINE_PATHS.items():
        if m in args.released and src in args.spine_sources:
            out[f'{m}_{src}'] = (RELEASED_METHODS[m], path, read_txt,
                                 'download the released SPINE vectors (third_party/spine/README.md)')
    if 'nnse' in args.released:
        for w in args.nnse_widths:
            f = NNSE_FILE.format(w)
            out[f'nnse_{w}'] = ('NNSE', NNSE_DIR / f, read_txt, f'wget -P {NNSE_DIR} {NNSE_URL}{f}')
    if 'sinr' in args.released:
        out['sinr_bnc'] = ('SINr', SINR_PATH, read_sinr,
                           f'git clone https://github.com/SINr-Embeddings/sinr {SINR_PATH.parents[1]}')
    return out


# --- Models -----------------------------------------------------------------------------
def load_methods(args):
    """name -> dict(path, config, words, E_raw) for every requested method, plus name -> why missing."""
    tk_best = TuckerDecomposition.load_best()
    like = str(tk_best.decomp_path)
    vocab = list(tk_best.vocab[f'vocab_{tk_best.roles[0]}'])
    del tk_best
    models, missing = {}, {}

    def vocab_config(v):
        return {'vocab': v, **({'like': like} if v == 'ours' else {})}

    for src in args.polar_sources:
        names = {(k, v): _name(f'polar_{src}_k{k}', v) for k in args.polar_dims for v in args.vocabs}
        names = {key: n for key, n in names.items() if selected(n, args.only)}
        if not names:
            continue
        s = SOURCES[src](vocab, full=any(v == 'own' for _, v in names))
        pairs = raw_antonym_pairs(s['contains'])
        pairs = [(a, b) for a, b in pairs if s['contains'](a) and s['contains'](b)]  # lowercased forms (w2v is cased)
        pairs = closest_antonyms(pairs, s['vec'])
        print(f"POLAR on {src}: {len(pairs)} antonym pairs (main.ipynb on its 2024 GloVe: 1467)")
        projections = {}  # the transform is fit on the whole embedding, so both vocabularies share it
        for (k, v), name in names.items():
            if k not in projections:
                projections[k] = polar_projection(pairs, s['vec'], k, args.polar_seed)
            P = projections[k]
            words, E = s[v]
            models[name] = dict(
                path=f"{s['path']}#polar_k{P.shape[0]}_seed{args.polar_seed}",
                config={'baseline': name, 'method': 'POLAR', 'source': s['path'], **vocab_config(v),
                        'dims_requested': k, 'dims': int(P.shape[0]), 'antonym_pairs': len(pairs),
                        'order': 'random', 'seed': args.polar_seed},
                words=words, E_raw=polar_rows(P, E))
        del s

    ours_set = set(vocab)
    for base, (method, path, reader, how) in released_files(args).items():
        names = {v: _name(base, v) for v in args.vocabs}
        names = {v: n for v, n in names.items() if selected(n, args.only)}
        if not names:
            continue
        if not Path(path).is_file():
            missing.update({n: f'{path} not found: {how}' for n in names.values()})
            continue
        words_all, E_all = reader(path)
        nnz = (E_all != 0).sum(1)
        zeros = int((nnz == 0).sum())
        if zeros:  # no active dimension (SINr's filtering can leave some): unknown like OOV; cosine would be NaN
            print(f"  {zeros} all-zero rows dropped (treated as unknown words)")
            keep = nnz > 0
            words_all = [w for w, k in zip(words_all, keep) if k]
            E_all, nnz = E_all[keep], nnz[keep]
        for v, name in names.items():
            if v == 'own':
                words, E, n = words_all, E_all, nnz
            else:  # our words, in file order
                rows = [i for i, w in enumerate(words_all) if w in ours_set]
                words, E, n = [words_all[i] for i in rows], E_all[rows], nnz[rows]
                print(f"{len(words)} of our {len(vocab)} words in {Path(path).name}")
            models[name] = dict(path=str(path), words=words, E_raw=E,
                                config={'baseline': name, 'method': method, 'source': str(path),
                                        **vocab_config(v), 'dim': int(E.shape[1]),
                                        'nonzeros_per_word': float(n.mean()), 'zero_rows_dropped': zeros})

    if args.export_txt:
        if str(SPINE_SUITE_DIR) not in sys.path:
            sys.path.insert(0, str(SPINE_SUITE_DIR))
        from spine_export import write_embeddings_txt
        for name, m in models.items():
            if len(m['words']) > EXPORT_MAX_WORDS:
                print(f"not exported: {name} ({len(m['words'])} words; SPINE's scripts load the whole file)")
                continue
            out = TXT_DIR / f'{name}.txt'
            write_embeddings_txt(_as_baseline(m['words'], m['E_raw'], ['w'], name), out)
            print(f"wrote {out}")
    return models, missing


def spec(args):
    return dict(methods='POLAR, SPINE, SPOWV, NNSE, SINr; vocabs: ours = the words of '
                        'TuckerDecomposition.load_best(), own = the whole embedding (*_full)', vocabs=args.vocabs,
                polar_sources=args.polar_sources, polar_dims=args.polar_dims, polar_seed=args.polar_seed,
                spine_sources=args.spine_sources, released=args.released,
                spine_paths={f'{m}_{s}': str(v) for (m, s), v in SPINE_PATHS.items()},
                nnse_widths=args.nnse_widths,
                nnse_paths={f'nnse_{w}': str(NNSE_DIR / NNSE_FILE.format(w)) for w in args.nnse_widths},
                sinr_path=str(SINR_PATH), antonym_files=ANTONYM_FILES)


def build_parser():
    p = ps.build_parser('POLAR downstream evaluation of interpretable embedding methods.', table=False)
    p.add_argument('--polar-sources', nargs='*', default=POLAR_SOURCES, choices=POLAR_SOURCES)
    p.add_argument('--polar-dims', nargs='+', type=int, default=POLAR_DIMS, metavar='K')
    p.add_argument('--polar-seed', type=int, default=POLAR_SEED)
    p.add_argument('--spine-sources', nargs='*', default=list(SPINE_SOURCES), choices=list(SPINE_SOURCES))
    p.add_argument('--released', nargs='*', default=RELEASED, choices=RELEASED,
                   help='released embeddings: spine, spowv (the SPINE release), nnse, sinr')
    p.add_argument('--nnse-widths', nargs='+', type=int, default=NNSE_WIDTHS, choices=[50, 300, 1000, 2500],
                   help='widths of the NNSE depDoc models (default: %(default)s)')
    p.add_argument('--vocabs', nargs='+', default=list(VOCABS), choices=VOCABS,
                   help="ours: the best model's words; own: the method's whole vocabulary (*_full)")
    p.add_argument('--export-txt', action='store_true',
                   help=f'also write each method in SPINE text format to {TXT_DIR}, for spine_eval')
    return p


if __name__ == '__main__':
    sys.exit(ps.main(parser=build_parser(), load=load_methods, spec=spec, prefix='methods'))
