"""GloVe baselines for the tea-leaves task, in the shape the evaluation expects.

``build_tealeaves_tasks`` (tealeaves_human.py) and ``DimConsistencyJudge.score``
(tensormet.judge) both read a decomposition through the same five members --
``roles``, ``get_role_index``, ``factors[i]`` as (N, R), ``vocab["vocab_<role>"]``
and ``get_dims()`` -- and nothing else. A pretrained embedding matrix satisfies
all five, so it can be scored on the identical trials without a decomposition
behind it::

    from glove_baseline import load_glove, load_glove_nmf
    # GloVe 2024 Wiki+Gigaword 100d (glove.2024.wikigiga.100d.zip), as downloaded
    path = DATA_DIR / "corpora" / "wiki_giga_2024_100_MFT20_vectors_seed_2024_alpha_0.75_eta_0.05.050_combined.txt"
    tk_dict["glove"]     = load_glove(path, like=tk_pfu)
    tk_dict["glove_nmf"] = load_glove_nmf(path, like=tk_pfu, rank=100)

Both then flow through ``open_session(...)``, ``run_widget``, ``summary()`` and
the judge unchanged. ``load_word2vec(like=tk_pfu)`` wraps the pretrained Google
News word2vec the same way.

The two are different claims, and only the second is a fair fight:

* ``load_glove`` uses the raw embedding axes. GloVe never optimised for
  axis-alignment and its coordinates are signed, so reading the positive tail of
  an arbitrary basis direction should score near chance. That is a real result --
  it shows the metric discriminates -- but report it as "unrotated embedding
  dimensions", not as "GloVe is uninterpretable".
* ``load_glove_nmf`` factorises the same vectors non-negatively, which is the
  same parts-based, axis-aligned regime the MU solver trains in. This is the
  baseline that isolates "does the extra tensor structure buy interpretability"
  from "does non-negativity buy it".

The fairest baseline of all -- NMF on the word x context marginal of the very
n-gram counts the tensor is built from -- is not here: it needs the data
pipeline, not an embedding file.

``simlex_rho`` / ``simlex_table`` score the same objects on SimLex-999 through
``tensormet.similarity``, which is the second half of the baseline: tea leaves
ask whether the dimensions are readable, SimLex asks whether the space is any
good. A model can win one and lose the other, and that contrast is the point.
"""
from __future__ import annotations

import gzip
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import torch

from tensormet.similarity import evaluate_simlex, load_simlex
from tensormet.tucker_tensor import _SIMLEX_PATH, _SIMLEX_POS_MAP, TuckerDecomposition
from tensormet.utils import to_np


class GloveBaseline(TuckerDecomposition):
    """An embedding matrix wearing the decomposition interface.

    ``core`` is None: every method the evaluation touches
    (``get_top_words_for_dimension``, ``get_top_dimensions_for_word``,
    ``get_most_similar_elements`` on a string) reads only ``factors`` and
    ``vocab``. The contextualised paths -- scoring a tuple,
    ``included_role_vector``, ``get_expected_element`` -- have no core to work
    with and will raise; there is no n-way interaction here to ask about.
    """

    def __init__(self, factors, vocab, roles, source: Optional[str] = None):
        super().__init__(core=None, factors=factors, vocab=vocab, roles=roles)
        self.source = source

    def get_rank(self, role=None):
        # The base class reads core.shape; here the rank is the factor's columns.
        return int(self.factors[0].shape[1])

    def __repr__(self) -> str:
        n, r = self.factors[0].shape
        return f"<{type(self).__name__} {n} words x {r} dims ({self.source})>"


# --- Loading ----------------------------------------------------------------
def _open_text(path: Path):
    return (gzip.open(path, "rt", encoding="utf-8") if path.suffix == ".gz"
            else open(path, "r", encoding="utf-8"))


def read_glove(path, keep: Optional[set] = None,
               verbose: bool = True) -> tuple[list[str], np.ndarray]:
    """Parse a GloVe text file, optionally restricted to `keep`.

    Returns the words in file order (descending corpus frequency) and their
    (N, D) matrix. Plain text or .gz; no gensim, no download step.
    """
    path = Path(path)
    words, vecs = [], []
    with _open_text(path) as fh:
        for line in fh:
            word, _, rest = line.partition(" ")
            if keep is not None and word not in keep:
                continue
            words.append(word)
            vecs.append(np.fromstring(rest, sep=" ", dtype=np.float32))
    if not words:
        raise ValueError(f"No vectors read from {path}"
                         + (" that are in `keep`." if keep else "."))

    widths = {v.size for v in vecs}
    if len(widths) != 1:
        raise ValueError(f"{path} mixes vector widths {sorted(widths)}; "
                         "the file is truncated or not GloVe-formatted.")
    if verbose:
        print(f"{len(words)} vectors of dim {widths.pop()} read from {path.name}"
              + (f" ({len(keep) - len(words)} of the model's words missing)"
                 if keep is not None else ""))
    return words, np.stack(vecs)


def _vocab_of(like, roles: Sequence[str]) -> list[str]:
    return list(like.vocab[f"vocab_{roles[0]}"])


def _as_baseline(words: list[str], matrix: np.ndarray,
                 roles: Sequence[str], source: str) -> GloveBaseline:
    """Wrap an (N, R) matrix as a shared-factor decomposition over `roles`."""
    factor = torch.from_numpy(np.ascontiguousarray(matrix, dtype=np.float32))
    vocab = {}
    for role in roles:
        # One matrix for every mode: a word's embedding does not depend on the
        # position it occupies, which is exactly `shared_factors="all"`.
        vocab[f"vocab_{role}"] = words
        vocab[f"{role}2i"] = {w: i for i, w in enumerate(words)}
    return GloveBaseline(factors=[factor] * len(roles), vocab=vocab,
                         roles=list(roles), source=source)


def load_glove(path, like=None, roles: Optional[Sequence[str]] = None,
               vocab: Optional[Sequence[str]] = None,
               verbose: bool = True) -> GloveBaseline:
    """Raw GloVe axes as latent dimensions.

    `like` is a loaded decomposition whose vocabulary and roles the baseline
    copies, so both models are scored over the same words -- which matters:
    the intruder pool (bottom half, top 10% elsewhere) and the diversity
    multiplier `len(all_dim_words) / (rank * k)` are both functions of the
    vocabulary size. Pass `vocab=` to set the word list by hand, or `roles=`
    alone to load the whole file -- untrimmed GloVe answers far more SimLex
    pairs, but is no longer comparable on tea leaves.

    Words in the model but absent from GloVe are dropped, so N ends up slightly
    below the model's; the count is printed. Pick the file whose width matches
    the model's rank (wiki_giga_2024_100_* for rank=100) to keep the two diversity
    denominators identical.
    """
    if like is None and roles is None:
        raise ValueError("Pass `like=` a loaded decomposition, or `roles=`.")
    roles = list(roles) if roles is not None else list(like.roles)
    if vocab is not None:
        keep = set(vocab)
    elif like is not None:
        keep = set(_vocab_of(like, roles))
    else:
        keep = None   # roles= alone: the whole file, for SimLex without a model

    words, matrix = read_glove(path, keep=keep, verbose=verbose)
    return _as_baseline(words, matrix, roles, source=Path(path).name)


def load_word2vec(like=None, name: str = "word2vec-google-news-300",
                  roles: Optional[Sequence[str]] = None,
                  vocab: Optional[Sequence[str]] = None,
                  keyed_vectors=None, verbose: bool = True) -> GloveBaseline:
    """Pretrained word2vec from gensim-data, as downloaded, on the model's words.

    Google News vectors are case-sensitive and the model's vocabulary is
    lowercase, so each word is looked up as is, then capitalised (the rule of
    the STS-B word2vec baseline in 0_tests). The vector is stored under the
    model's word. Pass `keyed_vectors=` an already loaded KeyedVectors to skip
    the ~1 min load.
    """
    if like is None and vocab is None:
        raise ValueError("Pass `like=` a loaded decomposition, or `vocab=`.")
    roles = list(roles) if roles is not None else list(like.roles)
    if keyed_vectors is None:
        import gensim.downloader
        keyed_vectors = gensim.downloader.load(name)

    wanted = list(vocab) if vocab is not None else _vocab_of(like, roles)
    words, forms = [], []
    for w in wanted:
        form = next((f for f in (w, w.capitalize()) if f in keyed_vectors), None)
        if form is not None:
            words.append(w)
            forms.append(form)
    if not words:
        raise ValueError(f"None of the {len(wanted)} words are in {name}.")
    if verbose:
        print(f"{len(words)} vectors of dim {keyed_vectors.vector_size} read from {name} "
              f"({len(wanted) - len(words)} of the model's words missing)")
    return _as_baseline(words, keyed_vectors[forms], roles, source=name)


# --- Non-negative variant ---------------------------------------------------
def _split_signs(matrix: np.ndarray) -> np.ndarray:
    """(N, D) signed -> (N, 2D) non-negative, losslessly: [max(x,0), max(-x,0)].

    Shifting by the global minimum would be simpler but adds a constant to every
    coordinate, and NMF then spends components reconstructing that offset.
    """
    return np.concatenate([np.maximum(matrix, 0.0), np.maximum(-matrix, 0.0)],
                          axis=1)


def nmf_mu(V: np.ndarray, rank: int, iterations: int = 200, seed: int = 0,
           tol: float = 1e-5, verbose: bool = False) -> np.ndarray:
    """Frobenius NMF by multiplicative updates; returns W of V ~ W H, (N, rank).

    Hand-rolled rather than sklearn's: it is thirty lines of matmuls at this
    size, it keeps the baseline runnable wherever the session files are, and it
    is the same update family the decompositions themselves are trained with.
    """
    rng = np.random.default_rng(seed)
    V = np.ascontiguousarray(V, dtype=np.float64)
    n, d = V.shape
    # Scaled random init: W H then starts at roughly V's magnitude, which MU
    # otherwise spends its first iterations climbing to.
    scale = np.sqrt(V.mean() / rank) if V.mean() > 0 else 1.0
    W = rng.random((n, rank)) * scale + 1e-6
    H = rng.random((rank, d)) * scale + 1e-6

    eps = 1e-10
    prev = None
    for it in range(iterations):
        H *= (W.T @ V) / (W.T @ W @ H + eps)
        W *= (V @ H.T) / (W @ (H @ H.T) + eps)
        if verbose or tol:
            err = float(np.linalg.norm(V - W @ H))
            if verbose and it % 20 == 0:
                print(f"  nmf iter {it:>4}  ||V - WH||_F = {err:.4f}")
            if prev is not None and abs(prev - err) <= tol * max(prev, eps):
                if verbose:
                    print(f"  converged at iter {it} (||V - WH||_F = {err:.4f})")
                break
            prev = err
    return W.astype(np.float32)


def load_glove_nmf(path, like=None, rank: Optional[int] = None,
                   roles: Optional[Sequence[str]] = None,
                   vocab: Optional[Sequence[str]] = None,
                   iterations: int = 100, seed: int = 1,
                   verbose: bool = True) -> GloveBaseline:
    """GloVe refactorised non-negatively: the axis-aligned baseline.

    The signed vectors are split into positive and negative parts (N, 2D) and
    factorised to `rank` components; the (N, rank) loadings become the factor.
    `rank` defaults to the model's, so the diversity multiplier's denominator
    matches trial for trial.
    """
    if rank is None:
        if like is None:
            raise ValueError("Pass `rank=`, or `like=` a decomposition to take it from.")
        rank = int(like.get_rank())

    base = load_glove(path, like=like, roles=roles, vocab=vocab, verbose=verbose)
    words = base.vocab[f"vocab_{base.roles[0]}"]
    signed = base.factors[0].numpy()

    if verbose:
        print(f"NMF: {signed.shape[0]} x {2 * signed.shape[1]} -> rank {rank}, "
              f"{iterations} MU iterations (seed {seed})")
    W = nmf_mu(_split_signs(signed), rank=rank, iterations=iterations,
               seed=seed, verbose=verbose)
    return _as_baseline(words, W, base.roles,
                        source=f"{Path(path).name}+nmf{rank}")


# --- SimLex-999 -------------------------------------------------------------
def simlex_rho(model, path=None, verbose: bool = True) -> dict:
    """SimLex-999 Spearman rho, in the shape ``evaluate_simlex`` returns it.

    Works on a GloveBaseline or on any loaded decomposition, so the baseline and
    the models are scored by the same code that produces `simlex_all_rho` during
    training: roles are mapped to POS tags via ``_SIMLEX_POS_MAP``, and any POS
    left empty is filled from the first role -- which is the whole story for
    positional n-gram models, whose modes share one vocabulary.

    ``evaluate_simlex`` normalises each pair itself, so the raw factor rows go in
    as they are.
    """
    pairs = load_simlex(_SIMLEX_PATH if path is None else path)

    by_role = {}
    for role in model.roles:
        key = f"{role}2i"
        if key not in model.vocab:
            continue
        factor = to_np(model.factors[model.get_role_index(role)])
        by_role[role] = {w: factor[i] for w, i in model.vocab[key].items()}
    if not by_role:
        raise ValueError(f"{model!r} has no `<role>2i` index to look words up by.")

    vecs_by_pos = {"N": {}, "V": {}, "A": {}}
    for role, vecs in by_role.items():
        pos = _SIMLEX_POS_MAP.get(role)
        if pos is not None and not vecs_by_pos[pos]:
            vecs_by_pos[pos] = vecs
    fallback = by_role[next(iter(by_role))]
    for pos, vecs in vecs_by_pos.items():
        if not vecs:
            vecs_by_pos[pos] = fallback

    return evaluate_simlex(pairs, vecs_by_pos, verbose=verbose)


def simlex_table(models: dict, path=None) -> dict:
    """`simlex_rho` over a dict of models -- one rho per row, ALL and per-POS.

    Note what the OOV column is telling you: a baseline restricted to a model's
    vocabulary answers fewer SimLex pairs than full GloVe does, and rho over an
    easier subset is not the same number. Compare rows with equal OOV counts.
    """
    scores = {name: simlex_rho(m, path=path, verbose=False)
              for name, m in models.items()}

    width = max(len(n) for n in scores)
    print(f"{'model':<{width}} {'ALL':>8} {'N':>8} {'V':>8} {'A':>8} {'OOV':>6}")
    print("-" * (width + 42))
    def cell(s, pos):
        rho = s.get(pos, {}).get("rho")
        return "     n/a" if rho is None else f"{rho:>8.3f}"

    for name, s in scores.items():
        cells = " ".join(cell(s, p) for p in ("ALL", "N", "V", "A"))
        print(f"{name:<{width}} {cells} {s.get('ALL', {}).get('oov', 0):>6}")
    return scores


# --- Sanity check -----------------------------------------------------------
def preview(model, dims: Sequence[int] = (0, 1, 2, 3, 4), top_k: int = 8) -> None:
    """Top words of a few dimensions -- the eyeball test before annotating 100."""
    role = model.roles[0]
    for d in dims:
        words = [w for w, _ in model.get_top_words_for_dimension(role, d, top_k)]
        print(f"dim {d:>3}: {', '.join(words)}")
