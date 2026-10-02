"""Word-in-Context (WiC) evaluation for decompositions and embedding baselines.

WiC (Pilehvar & Camacho-Collados, NAACL 2019) asks, for a target word in two
sentences, whether it has the same sense in both. Every method here reduces an
item to one cosine similarity, and a single threshold chosen on **train** turns
that into a label, so the dev and test accuracies are never selected on.

Two ways to get a vector for a target occurrence:

* ``static_similarities`` -- the mean of the type-level vectors within
  ±`window` tokens of the target. The same for the Tucker factor, GloVe and
  word2vec (anything with the ``roles`` / ``factors`` / ``vocab`` interface of
  glove_baseline.py). A static target vector is identical in both sentences,
  so any sense signal comes from the neighbours.
* ``contextual_similarities`` -- the target's own vector from
  ``ContextualEncoder``, which differs per sentence.
* ``excluded_similarities`` -- what the core predicts at the target's slot from
  the other words of each n-gram window (the excluded-role vector). It never
  reads the target, so it also scores OOV targets.

All methods tokenise through ``split_sentences`` with the Tucker model's spec,
so they see the same tokens and the same target position.

    import wic_eval as we
    root = we.download_wic("data")
    data = {s: we.load_wic(root, s) for s in we.SPLITS}
    slots = {s: we.locate_targets(items, enc.spec) for s, items in data.items()}
    sims = {s: we.contextual_similarities(enc, slots[s]) for s in we.SPLITS}
    row = we.evaluate(sims, we.labels(data))
"""
from __future__ import annotations

import io
import json
import urllib.request
import zipfile
from collections import Counter
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from tensormet.experimental.contextual import BOS, EOS, Sentence, cosine, split_sentences
from tensormet.utils import to_np

WIC_URL = "https://pilehvar.github.io/wic/package/WiC_dataset.zip"
SPLITS = ("train", "dev", "test")
EXPECTED_SIZES = {"train": 5428, "dev": 638, "test": 1400}

# (scores, ok): one cosine per item, and whether the method could score it.
Sims = Tuple[np.ndarray, np.ndarray]
# The sentence holding the target, and the target's token index in it.
Slot = Optional[Tuple[Sentence, int]]


# --- Data -------------------------------------------------------------------
@dataclass(frozen=True)
class WicItem:
    lemma: str
    pos: str
    s1: str
    s2: str
    idx1: int  # word index into s.split(" ")
    idx2: int
    label: Optional[bool]  # None when the split has no gold file

    @staticmethod
    def _char(sentence: str, idx: int) -> int:
        words = sentence.split(" ")
        return sum(len(w) + 1 for w in words[:idx])

    @property
    def char1(self) -> int:
        return self._char(self.s1, self.idx1)

    @property
    def char2(self) -> int:
        return self._char(self.s2, self.idx2)

    @property
    def surface1(self) -> str:
        return self.s1.split(" ")[self.idx1]

    @property
    def surface2(self) -> str:
        return self.s2.split(" ")[self.idx2]


def download_wic(dest, url: str = WIC_URL) -> Path:
    """Fetch and unpack the official WiC zip into `dest` once; returns `dest`."""
    dest = Path(dest)
    if list(dest.rglob("train.data.txt")):
        return dest
    dest.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(url) as resp:
        zipfile.ZipFile(io.BytesIO(resp.read())).extractall(dest)
    return dest


def _find(root: Path, name: str) -> Optional[Path]:
    hits = sorted(root.rglob(name))
    return hits[0] if hits else None


def load_wic(root, split: str) -> List[WicItem]:
    """Items of one split. Labels are None if `<split>.gold.txt` is absent."""
    root = Path(root)
    data = _find(root, f"{split}.data.txt")
    if data is None:
        raise FileNotFoundError(f"{split}.data.txt not found under {root}")
    gold = _find(root, f"{split}.gold.txt")
    tags = ([line.strip() == "T" for line in gold.read_text(encoding="utf-8").splitlines()
             if line.strip()] if gold else None)

    items = []
    for line in data.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        lemma, pos, idx, s1, s2 = line.split("\t")
        i1, i2 = (int(i) for i in idx.split("-"))
        items.append(WicItem(lemma, pos, s1, s2, i1, i2, None))
    if tags is not None:
        if len(tags) != len(items):
            raise ValueError(f"{split}: {len(items)} items but {len(tags)} gold labels")
        items = [replace(it, label=t) for it, t in zip(items, tags)]
    return items


def labels(data: Dict[str, List[WicItem]]) -> Dict[str, np.ndarray]:
    """Gold labels per split, for the splits that have them."""
    return {split: np.array([it.label for it in items], dtype=bool)
            for split, items in data.items() if items and items[0].label is not None}


def preflight(root, verbose: bool = True) -> Dict[str, dict]:
    """Item counts, gold presence and label balance per split."""
    out = {}
    for split in SPLITS:
        try:
            items = load_wic(root, split)
        except FileNotFoundError as e:
            out[split] = {"ok": False, "error": str(e)}
            continue
        has_gold = items[0].label is not None
        out[split] = {"ok": True, "items": len(items),
                      "expected": EXPECTED_SIZES[split], "gold": has_gold,
                      "frac_true": (round(float(np.mean([it.label for it in items])), 4)
                                    if has_gold else None)}
    if verbose:
        for split, info in out.items():
            if not info["ok"]:
                print(f"MISSING  {split}: {info['error']}")
                continue
            size = "" if info["items"] == info["expected"] else f" (expected {info['expected']})"
            gold = f"gold, {info['frac_true']:.1%} T" if info["gold"] else "no gold"
            print(f"     ok  {split:<5} {info['items']:>5} items{size}  {gold}")
    return out


# --- Target positions -------------------------------------------------------
def locate(text: str, char: int, spec) -> Slot:
    """(sentence, token index) of the token covering `char`, or None."""
    for sentence in split_sentences(text, spec):
        i = sentence.token_at(char)
        if i is not None:
            return sentence, i
    return None


def locate_targets(items: Sequence[WicItem], spec) -> List[Tuple[Slot, Slot]]:
    """Both target slots of every item, tokenised as the model sees text."""
    return [(locate(it.s1, it.char1, spec), locate(it.s2, it.char2, spec)) for it in items]


def target_mismatches(items: Sequence[WicItem], slots) -> List[tuple]:
    """Items whose located token differs from the WiC surface word.

    A token that is a prefix of the surface word is fine: the tokenizer split
    it (``well-known`` -> ``well``). A missing slot is always reported.
    """
    bad = []
    for it, pair in zip(items, slots):
        for surface, slot in ((it.surface1, pair[0]), (it.surface2, pair[1])):
            tok = None if slot is None else slot[0].tokens[slot[1]]
            if tok is None or not surface.lower().startswith(tok):
                bad.append((it.lemma, surface, tok))
    return bad


# --- Static vectors ---------------------------------------------------------
class StaticTable:
    """Word -> row lookup over one role's factor, optionally limited to `vocab`."""

    def __init__(self, model, vocab: Optional[set] = None, role: Optional[str] = None):
        role = role or model.roles[0]
        self.matrix = to_np(model.factors[model.get_role_index(role)]).astype(np.float64)
        index = model.vocab[f"{role}2i"]
        self.index = {w: i for w, i in index.items() if vocab is None or w in vocab}

    def mean(self, tokens: Sequence[str]) -> Optional[np.ndarray]:
        rows = [self.index[t] for t in tokens if t in self.index]
        return self.matrix[rows].mean(axis=0) if rows else None


def static_similarities(table: StaticTable, slots, window: Optional[int]) -> Sims:
    """Cosine of the mean vectors within ±`window` tokens of each target.

    `window=None` takes the whole sentence; `window=0` the target alone. A side
    with no in-vocabulary token in its window makes the item unscorable.
    """
    n = len(slots)
    scores, ok = np.full(n, np.nan), np.zeros(n, dtype=bool)
    for k, pair in enumerate(slots):
        vecs = []
        for slot in pair:
            if slot is None:
                break
            sentence, i = slot
            lo, hi = (0, len(sentence.tokens)) if window is None else (
                max(0, i - window), i + window + 1)
            v = table.mean(sentence.tokens[lo:hi])
            if v is None:
                break
            vecs.append(v)
        if len(vecs) == 2:
            scores[k], ok[k] = cosine(*vecs), True
    return scores, ok


# --- Contextual vectors -----------------------------------------------------
def contextual_similarities(enc, slots, *, rounds: int = 1,
                            context_strength: float = 1.0, batch: int = 4096) -> Sims:
    """Cosine of the two targets' ContextualEncoder vectors.

    Unscorable when a target lies in no fully in-vocabulary n-gram window.
    """
    flat = [slot for pair in slots for slot in pair]
    present = [k for k, slot in enumerate(flat) if slot is not None]
    vecs = np.zeros((len(flat), enc.rank))
    has = np.zeros(len(flat), dtype=bool)
    for start in range(0, len(present), batch):
        chunk = present[start:start + batch]
        tvs = enc.batch_token_vectors([flat[k][0] for k in chunk], rounds=rounds,
                                      context_strength=context_strength)
        for k, tv in zip(chunk, tvs):
            i = flat[k][1]
            vecs[k], has[k] = tv.vectors[i], tv.mask[i]

    ok = has[0::2] & has[1::2]
    scores = np.full(len(slots), np.nan)
    if ok.any():
        scores[ok] = cosine(vecs[0::2][ok], vecs[1::2][ok])
    return scores, ok


@torch.no_grad()
def excluded_vectors(enc, flat_slots: Sequence[Slot]) -> Tuple[np.ndarray, np.ndarray]:
    """What the context predicts at each target's position: (vectors (N, R), has (N,)).

    The excluded-role vector of the target's slot in every n-gram window covering
    it (the core contracted with the other n-1 factor rows), each scaled to sum
    to 1, averaged. The target's own row is never read, so an OOV target is
    fine; a window only needs its other n-1 tokens in the vocabulary.
    """
    n = enc.spec.order
    pad = n - 1 if enc.spec.padded else 0
    windows, owner, slot_of = [], [], []
    for k, slot in enumerate(flat_slots):
        if slot is None:
            continue
        sentence, i = slot
        seq = [BOS] * pad + list(sentence.tokens) + ([EOS] if enc.spec.padded else [])
        rows = [enc.word_index.get(t, -1) for t in seq]
        p = i + pad
        rows[p] = 0  # placeholder: slot p's own row does not enter its excluded vector
        for s in range(max(0, p - n + 1), min(p, len(seq) - n) + 1):
            if min(rows[s:s + n]) >= 0:
                windows.append(rows[s:s + n])
                owner.append(k)
                slot_of.append(p - s)

    dev = enc.device
    total = torch.zeros(len(flat_slots), enc.rank, dtype=enc.factor.dtype, device=dev)
    count = torch.zeros(len(flat_slots), dtype=enc.factor.dtype, device=dev)
    W = torch.tensor(windows, dtype=torch.long, device=dev).reshape(-1, n)
    O = torch.tensor(owner, dtype=torch.long, device=dev)
    J = torch.tensor(slot_of, dtype=torch.long, device=dev)
    for start in range(0, len(W), enc.chunk):
        w, o, j = W[start:start + enc.chunk], O[start:start + enc.chunk], J[start:start + enc.chunk]
        msgs = torch.stack(enc._messages(enc.factor, w), dim=1)  # (b, n, R)
        m = msgs[torch.arange(len(w), device=dev), j]
        mass = m.sum(dim=1, keepdim=True)
        good = mass[:, 0] > enc.eps
        total.index_add_(0, o[good], m[good] / mass[good])
        count.index_add_(0, o[good], torch.ones_like(o[good], dtype=count.dtype))

    has = count > 0
    vecs = total / count.clamp(min=1)[:, None]
    return vecs.cpu().numpy(), has.cpu().numpy()


def excluded_similarities(enc, slots) -> Sims:
    """Cosine of the two targets' excluded-role vectors (see excluded_vectors).

    Unscorable only when a side has no window whose other n-1 tokens are all
    in the vocabulary.
    """
    flat = [slot for pair in slots for slot in pair]
    vecs, has = excluded_vectors(enc, flat)
    ok = has[0::2] & has[1::2]
    scores = np.full(len(slots), np.nan)
    if ok.any():
        scores[ok] = cosine(vecs[0::2][ok], vecs[1::2][ok])
    return scores, ok


# --- Threshold and accuracy -------------------------------------------------
def fit_threshold(scores: np.ndarray, gold: np.ndarray) -> Tuple[float, float]:
    """θ maximising accuracy of `score >= θ`; returns (θ, accuracy).

    Candidates are the midpoints between distinct sorted scores plus ±inf; the
    first best one wins, so the result is deterministic.
    """
    order = np.argsort(scores, kind="stable")
    s, y = scores[order], gold[order]
    n = len(s)
    # Cut k: items [0, k) predicted F, [k, n) predicted T.
    neg_below = np.concatenate([[0], np.cumsum(~y)])
    pos_above = y.sum() - np.concatenate([[0], np.cumsum(y)])
    acc = (neg_below + pos_above) / n
    valid = np.ones(n + 1, dtype=bool)
    valid[1:n] = s[1:] > s[:-1]  # no cut inside a tie
    k = int(np.flatnonzero(valid)[np.argmax(acc[valid])])
    theta = -np.inf if k == 0 else np.inf if k == n else (s[k - 1] + s[k]) / 2
    return float(theta), float(acc[k])


def predict(scores: np.ndarray, ok: np.ndarray, theta: float, majority: bool) -> np.ndarray:
    """Labels: `score >= θ` where the method scored the pair, else the majority label."""
    pred = np.full(len(scores), majority)
    pred[ok] = scores[ok] >= theta
    return pred


def evaluate(sims: Dict[str, Sims], gold: Dict[str, np.ndarray],
             keep: Optional[Dict[str, np.ndarray]] = None) -> dict:
    """θ on train, then accuracy on every split with gold.

    `keep[split]` picks the pairs to score (default: all). Kept pairs the method
    cannot score get the majority label of the kept train pairs, so a method is
    charged for its coverage rather than scored on an easier subset.
    """
    keep = keep or {s: np.ones(len(g), dtype=bool) for s, g in gold.items()}
    s_tr, ok_tr = sims["train"]
    fit = keep["train"] & ok_tr
    theta, _ = fit_threshold(s_tr[fit], gold["train"][fit])
    majority = bool(gold["train"][keep["train"]].mean() >= 0.5)

    out = {"theta": theta, "majority": majority}
    for split, g in gold.items():
        scores, ok = sims[split]
        k = keep[split]
        pred = predict(scores, ok, theta, majority)
        out[f"acc_{split}"] = float((pred[k] == g[k]).mean()) if k.any() else float("nan")
        out[f"n_{split}"] = int(k.sum())
        out[f"scored_{split}"] = int((k & ok).sum())
    return out


def common_mask(all_sims: Dict[str, Dict[str, Sims]]) -> Dict[str, np.ndarray]:
    """Per split, the pairs every method can score."""
    splits = next(iter(all_sims.values())).keys()
    return {split: np.logical_and.reduce([s[split][1] for s in all_sims.values()])
            for split in splits}


def run_methods(all_sims: Dict[str, Dict[str, Sims]], gold: Dict[str, np.ndarray],
                results_path=None, meta: Optional[dict] = None) -> List[dict]:
    """One row per method: accuracy on the common subset and on all pairs.

    `all_sims` maps a method label to its per-split Sims. Rows are appended to
    `results_path` (JSONL) if given.
    """
    common = common_mask(all_sims)
    rows = []
    for label, sims in all_sims.items():
        row = {"method": label, **(meta or {})}
        row.update({f"common_{k}": v for k, v in evaluate(sims, gold, common).items()})
        row.update({f"all_{k}": v for k, v in evaluate(sims, gold).items()})
        rows.append(row)

    if results_path:
        results_path = Path(results_path)
        results_path.parent.mkdir(parents=True, exist_ok=True)
        with results_path.open("a", encoding="utf-8") as fh:
            for row in rows:
                fh.write(json.dumps(row, default=str) + "\n")
    return rows


def results_frame(rows: Sequence[dict], splits: Sequence[str] = ("dev", "test")):
    """DataFrame: accuracies on the common subset, then on all pairs."""
    import pandas as pd

    records = []
    for row in rows:
        rec = {"method": row["method"]}
        for block in ("common", "all"):
            for split in splits:
                if f"{block}_acc_{split}" in row:
                    rec[f"{split} ({block})"] = row[f"{block}_acc_{split}"]
        for split in splits:
            if f"all_scored_{split}" in row:
                rec[f"{split} scored"] = f"{row[f'all_scored_{split}']}/{row[f'all_n_{split}']}"
        rec["θ (all)"] = row["all_theta"]
        records.append(rec)
    return pd.DataFrame.from_records(records).set_index("method")


# --- Coverage ---------------------------------------------------------------
def coverage(items: Sequence[WicItem], slots, vocab: set,
             contextual_ok: Optional[np.ndarray] = None, top: int = 15) -> dict:
    """What the vocabulary lets each method see, for one split.

    `contextual_ok` is the `ok` mask of a contextual_similarities run: whether
    both targets lie in a fully in-vocabulary window.
    """
    n = len(items)
    found = both_in = 0
    ctx_share, oov = [], Counter()
    for pair in slots:
        if all(slot is not None for slot in pair):
            found += 1
        in_vocab = []
        for slot in pair:
            if slot is None:
                in_vocab.append(False)
                continue
            sentence, i = slot
            target = sentence.tokens[i]
            in_vocab.append(target in vocab)
            if target not in vocab:
                oov[target] += 1
            others = [t for j, t in enumerate(sentence.tokens) if j != i]
            if others:
                ctx_share.append(sum(t in vocab for t in others) / len(others))
        both_in += all(in_vocab)

    out = {"items": n,
           "targets_located": round(found / n, 4),
           "target_in_vocab_both": round(both_in / n, 4),
           "context_token_coverage": round(float(np.mean(ctx_share)), 4) if ctx_share else 0.0,
           "top_oov_targets": oov.most_common(top)}
    if contextual_ok is not None:
        out["contextual_vector_both"] = round(float(contextual_ok.mean()), 4)
    return out


def _side_reason(it: WicItem, slot: Slot, vocab: set, order: Optional[int]) -> str:
    """Why one side of an item gets no vector ("ok" if it does)."""
    if slot is None:
        return "not located"
    sentence, i = slot
    if sentence.tokens[i] not in vocab:
        return ("target OOV, lemma in vocab" if it.lemma.lower() in vocab
                else "target OOV")
    if order is None:
        return "ok"
    # The encoder's padded windows: n-1 <s> in front, one </s> behind.
    seq = ["<s>"] * (order - 1) + list(sentence.tokens) + ["</s>"]
    p = i + order - 1
    starts = range(max(0, p - order + 1), min(p, len(seq) - order) + 1)
    if any(all(t in vocab for t in seq[s:s + order]) for s in starts):
        return "ok"
    return "no in-vocab window"


def _show_context(slot: Slot, vocab: set, span: int) -> str:
    """±`span` tokens around the target: target as *word*, OOV words as [word]."""
    if slot is None:
        return ""
    sentence, i = slot
    out = []
    for j in range(max(0, i - span), min(len(sentence.tokens), i + span + 1)):
        t = sentence.tokens[j]
        t = f"*{t}*" if j == i else t
        out.append(t if sentence.tokens[j] in vocab else f"[{t}]")
    return " ".join(out)


def unscored(items: Sequence[WicItem], slots, ok: np.ndarray, vocab: set,
             order: Optional[int] = None):
    """One row per item a method could not score, with the reason per side.

    `ok` is the method's mask and `vocab` the words it can look up. With
    `order` (the n-gram order) a side also fails when no fully in-vocabulary
    window covers the target, as in ContextualEncoder; the context column then
    shows ±(order-1) tokens, else ±3.
    """
    import pandas as pd

    span = order - 1 if order else 3
    records = []
    for k in np.flatnonzero(~ok):
        it = items[k]
        rec = {"item": int(k), "lemma": it.lemma, "pos": it.pos, "label": it.label}
        for side, slot in (("1", slots[k][0]), ("2", slots[k][1])):
            rec[f"reason {side}"] = _side_reason(it, slot, vocab, order)
            rec[f"context {side}"] = _show_context(slot, vocab, span)
        records.append(rec)
    return pd.DataFrame.from_records(records).set_index("item") if records else pd.DataFrame()
