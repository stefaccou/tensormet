"""Contextualised embeddings from a fitted n-gram Tucker model.

A sentence is read as its overlapping n-gram windows. In each window, the
included-role vector of a token's slot, scaled to sum to 1, says which latent
dimensions that token uses there. A token's vector is the average over the
windows covering it. Each extra round replaces the neighbours' factor rows by
their current vectors, so context reaches rounds·(n-1) tokens each side.

    enc = ContextualEncoder(model, device=select_gpu())
    enc.token_vectors(text)                # local n-gram context
    enc.token_vectors(text, rounds=3)      # context up to 3·(n-1) tokens away
    enc.token_vectors(text, rounds=0)      # no context
    enc.batch_sentence_vectors(texts)      # one row per text
    enc.plot_mixing(text, "bank")          # how one word's vector is built

context_strength (0 to 1) sets how much the neighbours count.
Runs with torch on CPU or GPU; the batch_* methods process many texts in one pass.
Requires fully tied factors (shared_factors="all").
"""
from __future__ import annotations

import re
from dataclasses import dataclass, replace
from functools import lru_cache
from typing import List, Optional, Sequence, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import spacy
import torch
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from tensormet.tt_hybrid import tt_chain
from tensormet.utils import einsum_letters, resolve_shared_factors, voc_index

BOS = "<s>"
EOS = "</s>"

# Plot colours: default palette slots 1-2, ink, and chart chrome.
_OWN, _CONTEXT = "#2a78d6", "#eb6834"
_INK, _INK_2, _MUTED = "#0b0b0b", "#52514e", "#898781"
_SURFACE, _GRID, _AXIS = "#fcfcfb", "#e1e0d9", "#c3c2b7"


# --- text front end (mirrors vector_creation.py) ---------------------------

@dataclass(frozen=True)
class NgramSpec:
    """How to turn text into the n-grams a model was fitted on."""

    order: int
    raw: bool = True
    padded: bool = True
    spacy_model: str = "en_core_web_md"

    @classmethod
    def from_model(cls, model, **overrides) -> "NgramSpec":
        """Read the spec off the model: order from its roles, tokenisation and
        padding from the dataset directory name."""
        fields = {"order": len(model.roles)}
        if "raw" not in overrides or "padded" not in overrides:
            fields["raw"], fields["padded"] = _dataset_flags(model)
        return cls(**{**fields, **overrides})


_SPEC_HINT = "Build the NgramSpec explicitly, e.g. NgramSpec(order=4, raw=True, padded=True)."


def _dataset_flags(model) -> Tuple[bool, bool]:
    """(raw, padded) from a dataset name like '4-gram-raw-bos-eos-fineweb-en_1B'."""
    if model.decomp_path is None:
        raise ValueError(f"Model has no decomp_path, so raw/padded cannot be inferred. {_SPEC_HINT}")
    dataset = model.decomp_path.parents[1].name
    m = re.match(r"\d+-?gram(-raw)?(-bos-eos)?-", dataset)
    if m is None:
        raise ValueError(f"Cannot read raw/padded from dataset name {dataset!r}. {_SPEC_HINT}")
    return m.group(1) is not None, m.group(2) is not None


@dataclass(frozen=True)
class Sentence:
    """One sentence, tokenised as the model sees it. Spans are character
    offsets into `text`."""

    text: str
    tokens: Tuple[str, ...]
    spans: Tuple[Tuple[int, int], ...]

    def token_at(self, char_index: int) -> Optional[int]:
        """Index of the token covering `char_index`, or None."""
        for i, (start, end) in enumerate(self.spans):
            if start <= char_index < end:
                return i
        return None


@lru_cache(maxsize=4)
def _pipeline(raw: bool, spacy_model: str):
    nlp = (spacy.blank("en") if raw
           else spacy.load(spacy_model, disable=["ner", "parser", "senter", "textcat"]))
    nlp.add_pipe("sentencizer")
    nlp.max_length = 1_000_000
    return nlp


def split_sentences(text: str, spec: NgramSpec) -> List[Sentence]:
    """Tokenise exactly as vector_creation does: lowercased tokens (or lemmas),
    punctuation and whitespace dropped."""
    doc = _pipeline(spec.raw, spec.spacy_model)(text)
    out = []
    for sent in doc.sents:
        tokens, spans = [], []
        for tok in sent:
            if tok.is_punct or tok.is_space:
                continue
            form = tok.lower_ if spec.raw else tok.lemma_.lower()
            if not form.strip():
                continue
            tokens.append(form)
            spans.append((tok.idx, tok.idx + len(tok.text)))
        if tokens:
            out.append(Sentence(text, tuple(tokens), tuple(spans)))
    return out


# --- results ---------------------------------------------------------------

@dataclass(frozen=True)
class TokenVectors:
    """Per-token vectors, each summing to 1. `mask` is False (and the row
    zero) for tokens without a vector. `n_kept` of the text's `n_windows`
    n-gram windows are fully in vocabulary."""

    tokens: Tuple[str, ...]
    vectors: np.ndarray  # (L, R)
    mask: np.ndarray  # (L,) bool
    spans: Tuple[Tuple[int, int], ...]
    n_windows: int = 0
    n_kept: int = 0

    def __len__(self) -> int:
        return len(self.tokens)

    @property
    def coverage(self) -> float:
        """Fraction of the text's windows that are fully in vocabulary."""
        return self.n_kept / self.n_windows if self.n_windows else 0.0

    def valid(self) -> "TokenVectors":
        """Drop the tokens that have no vector."""
        keep = np.flatnonzero(self.mask)
        return replace(
            self,
            tokens=tuple(self.tokens[i] for i in keep),
            vectors=self.vectors[keep],
            mask=self.mask[keep],
            spans=tuple(self.spans[i] for i in keep),
        )


def cosine(a: np.ndarray, b: np.ndarray, eps: float = 1e-12) -> Union[float, np.ndarray]:
    """Cosine similarity; accepts vectors or stacks of row vectors."""
    a, b = np.atleast_2d(a), np.atleast_2d(b)
    num = (a * b).sum(axis=1)
    den = np.maximum(np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1), eps)
    sims = num / den
    return float(sims[0]) if sims.size == 1 else sims


@dataclass(frozen=True)
class MixingTrace:
    """How one word's vector is built; see ContextualEncoder.trace. Every
    vector sums to 1, and `context` and `mixed` have one row per window."""

    word: str
    rounds: int
    windows: Tuple[Tuple[str, ...], ...]  # the words of each window
    slots: Tuple[int, ...]  # the word's place in each window
    own: np.ndarray  # (R,) the word's factor row
    context: np.ndarray  # (W, R) what the other words of the window predict
    mixed: np.ndarray  # (W, R) own × context
    vector: np.ndarray  # (R,) mean of `mixed`: the word's vector


# --- encoder ---------------------------------------------------------------

class ContextualEncoder:
    """Turns running text into contextual token and sentence vectors."""

    def __init__(self, model, spec: Optional[NgramSpec] = None, *,
                 device: Union[str, torch.device] = "cpu",
                 eps: float = 1e-12, chunk_bytes: int = 256 << 20):
        _require_tied(model)
        self.model = model
        self.spec = spec or NgramSpec.from_model(model)
        self.device = torch.device(device)
        self.eps = eps
        n = self.spec.order
        if n != len(model.roles):
            raise ValueError(f"spec.order is {n}, but the model has {len(model.roles)} roles.")

        self.word_index = model.vocab[voc_index(model.roles[0])]
        if self.spec.padded and not (BOS in self.word_index and EOS in self.word_index):
            raise ValueError(
                f"The spec is padded, but {BOS} or {EOS} is not in the vocabulary, so every "
                "sentence-boundary window would be dropped. Pass NgramSpec(..., padded=False)."
            )
        self.factor = _to_torch(model.factors[0], self.device)
        self.rank = int(self.factor.shape[1])

        tt_cores = getattr(model, "tt_cores", None)
        if tt_cores is not None:
            self.core = None
            self.tt_cores = [_to_torch(C, self.device).to(self.factor.dtype) for C in tt_cores]
            bond = max(int(d) for C in self.tt_cores for d in (C.shape[0], C.shape[2]))
            # n site matrices, one gradient, and the gathered rows.
            per_window = n * bond ** 2 + self.rank * bond + 4 * n * self.rank
        else:
            self.tt_cores = None
            self.core = _to_torch(model.core, self.device).to(self.factor.dtype)
            per_window = self.rank ** (n - 1)
        # Windows per contraction call, so the largest intermediate stays under chunk_bytes.
        self.chunk = max(1, chunk_bytes // (self.factor.element_size() * per_window))

        # dim_mass: the model contracted with the factor column sums at every other
        # slot, summed over slots (the tied factor's MU denominator).
        colsum = self.factor.sum(dim=0)
        slots = torch.zeros(1, n, dtype=torch.long, device=self.device)
        self.dim_mass = sum(self._messages(colsum[None], slots))[0].clamp(min=eps)

        # SIF frequency estimate: Zipf over the vocab, whose index is the frequency
        # rank once <s> and </s> are skipped.
        self._specials = np.array(sorted(self.word_index[t] for t in (BOS, EOS)
                                         if t in self.word_index), dtype=np.int64)
        n_words = len(self.word_index) - len(self._specials)
        self._zipf = float(np.log(n_words) + 0.5772156649)

    # -- token vectors --

    def token_vectors(self, text: Union[str, Sentence], *, rounds: int = 1,
                      damping: float = 0.0,
                      context_strength: float = 1.0) -> TokenVectors:
        """Vector per token of one text; see batch_token_vectors."""
        return self.batch_token_vectors([text], rounds=rounds, damping=damping,
                                        context_strength=context_strength)[0]

    @torch.no_grad()
    def batch_token_vectors(self, texts: Sequence[Union[str, Sentence]], *,
                            rounds: int = 1, damping: float = 0.0,
                            context_strength: float = 1.0) -> List[TokenVectors]:
        """Vector per token, for many texts in one pass.

        rounds=0 uses no context. rounds=1 uses the n-gram windows containing
        the token; each extra round carries context n-1 tokens further.
        `damping` mixes in the previous round's vectors.
        `context_strength` sets how much the neighbours count: 0 not at all
        (but only tokens in a window get a vector), 1 fully.
        """
        if rounds < 0:
            raise ValueError("rounds must be at least 0")
        if context_strength < 0:
            raise ValueError("context_strength must be at least 0")

        docs = [[t] if isinstance(t, Sentence) else split_sentences(t, self.spec)
                for t in texts]
        sentences = [s for doc in docs for s in doc]
        rows, windows, token_slices, counts = self._layout(sentences)
        vectors, covered = self._propagate(rows, windows, rounds, damping, context_strength)
        vectors, covered = vectors.cpu().numpy(), covered.cpu().numpy()

        out, k = [], 0
        for doc in docs:
            parts = []
            for sentence in doc:
                where = token_slices[k]
                parts.append(TokenVectors(sentence.tokens, vectors[where], covered[where],
                                          sentence.spans, *counts[k]))
                k += 1
            out.append(_concat(parts, self.rank, vectors.dtype))
        return out

    def word_in_context(self, text: str, char_index: int, *, rounds: int = 1,
                        context_strength: float = 1.0) -> Optional[np.ndarray]:
        """Vector for the token covering `char_index`, or None if that token has
        no vector."""
        for sentence in split_sentences(text, self.spec):
            i = sentence.token_at(char_index)
            if i is None:
                continue
            tv = self.token_vectors(sentence, rounds=rounds,
                                    context_strength=context_strength)
            return tv.vectors[i] if tv.mask[i] else None
        return None

    def _layout(self, sentences: Sequence[Sentence]):
        """Lay all sentences end to end as one sequence of positions.

        Returns the vocab row per position (-1 if out of vocabulary), the
        (n_windows, n) positions of every kept window, per sentence the slice of
        positions holding its real tokens, and per sentence (windows, kept windows).
        Windows never cross sentences, and windows containing an out-of-vocabulary
        token are dropped.
        """
        n = self.spec.order
        pad = n - 1 if self.spec.padded else 0
        rows, windows, token_slices, counts = [], [], [], []
        for sentence in sentences:
            seq = list(sentence.tokens)
            if self.spec.padded:
                seq = [BOS] * pad + seq + [EOS]
            offset = len(rows)
            rows.extend(self.word_index.get(t, -1) for t in seq)
            token_slices.append(slice(offset + pad, offset + pad + len(sentence.tokens)))
            starts = range(offset, offset + len(seq) - n + 1)
            kept = [list(range(s, s + n)) for s in starts
                    if all(rows[p] >= 0 for p in range(s, s + n))]
            windows.extend(kept)
            counts.append((len(starts), len(kept)))

        rows = torch.tensor(rows, dtype=torch.long, device=self.device)
        windows = torch.tensor(windows, dtype=torch.long, device=self.device).reshape(-1, n)
        return rows, windows, token_slices, counts

    def _propagate(self, rows: torch.Tensor, windows: torch.Tensor,
                   rounds: int, damping: float, context_strength: float = 1.0):
        """Run the rounds. Returns (vectors (positions, R), covered (positions,))."""
        n = self.spec.order
        known = rows >= 0
        own = torch.zeros(len(rows), self.rank, dtype=self.factor.dtype, device=self.device)
        own[known] = self.factor[rows[known]]
        # Without context, a token's vector is its factor row weighted by dim_mass.
        vectors = _unit_sum(own * self.dim_mass, self.eps)
        covered = known if rounds == 0 else torch.zeros_like(known)

        for _ in range(rounds):
            # Vectors carry dim_mass; divide it out so neighbours enter the model
            # like factor rows.
            latents = _unit_sum(vectors / self.dim_mass, self.eps)
            total = torch.zeros_like(vectors)
            count = torch.zeros(len(rows), dtype=vectors.dtype, device=self.device)
            for start in range(0, len(windows), self.chunk):
                batch = windows[start:start + self.chunk]
                messages = self._with_strength(self._messages(latents, batch), context_strength)
                for j in range(n):
                    included = own[batch[:, j]] * messages[j]
                    mass = included.sum(dim=1, keepdim=True)
                    # Windows the model gives (almost) no mass have no direction.
                    ok = mass[:, 0] > self.eps
                    positions = batch[ok, j]
                    total.index_add_(0, positions, included[ok] / mass[ok])
                    count.index_add_(0, positions, torch.ones_like(positions, dtype=vectors.dtype))
            got = count > 0
            # Positions without a usable window keep their previous vector.
            new = torch.where(got[:, None], total / count.clamp(min=1)[:, None], vectors)
            vectors = new if damping == 0.0 else (1.0 - damping) * new + damping * vectors
            covered = covered | got

        vectors[~covered] = 0.0
        return vectors, covered

    def _messages(self, vectors: torch.Tensor, windows: torch.Tensor) -> List[torch.Tensor]:
        """Excluded-role vector of every slot: what the other slots of each
        window predict there. A list of n tensors, each (n_windows, R)."""
        n = self.spec.order
        latents = [vectors[windows[:, j]] for j in range(n)]

        if self.tt_cores is not None:
            sites = tt_chain.sites(self.tt_cores, latents, torch)
            left, right = tt_chain.left_envs(sites, torch), tt_chain.right_envs(sites, torch)
            return [tt_chain.site_grad(left[j], self.tt_cores[j], right[j + 1], torch)
                    for j in range(n)]

        letters = "".join(einsum_letters(n))
        out = []
        for j in range(n):
            others = [k for k in range(n) if k != j]
            eq = f"{letters}," + ",".join(f"Z{letters[k]}" for k in others) + f"->Z{letters[j]}"
            out.append(torch.einsum(eq, self.core, *(latents[k] for k in others)))
        return out

    def _with_strength(self, messages: List[torch.Tensor],
                       context_strength: float) -> List[torch.Tensor]:
        """dim_mass · (message / dim_mass)^s: s=1 keeps the messages, s=0 ignores
        the neighbours."""
        if context_strength == 1.0:
            return messages
        return [self.dim_mass * (m / self.dim_mass).clamp(min=0) ** context_strength
                for m in messages]

    # -- sentence vectors --

    def sentence_vector(self, text: Union[str, Sentence], *, rounds: int = 1,
                        weighting: str = "sif", a: float = 1e-3,
                        context_strength: float = 1.0) -> np.ndarray:
        """One sentence vector; see batch_sentence_vectors."""
        tv = self.token_vectors(text, rounds=rounds, context_strength=context_strength)
        return self.pool(tv, weighting=weighting, a=a)

    def batch_sentence_vectors(self, texts: Sequence[Union[str, Sentence]], *,
                               rounds: int = 1, weighting: str = "sif",
                               a: float = 1e-3,
                               context_strength: float = 1.0) -> np.ndarray:
        """One L2-normalised vector per text, (len(texts), R), in one pass.

        A text without any token vector gets a NaN row, so its similarities
        come out NaN instead of a made-up value.
        """
        pooled = [self.pool(tv, weighting=weighting, a=a)
                  for tv in self.batch_token_vectors(texts, rounds=rounds,
                                                     context_strength=context_strength)]
        return np.stack(pooled) if pooled else np.empty((0, self.rank))

    def pool(self, tv: TokenVectors, *, weighting: str = "sif",
             a: float = 1e-3) -> np.ndarray:
        """Pool token vectors into one L2-normalised sentence vector, weighting
        each token by `word_weights`."""
        tv = tv.valid()
        if len(tv) == 0:
            return np.full(tv.vectors.shape[1], np.nan)
        pooled = self.word_weights(tv.tokens, weighting=weighting, a=a) @ tv.vectors
        return pooled / max(float(np.linalg.norm(pooled)), self.eps)

    def word_weights(self, tokens: Sequence[str], *, weighting: str = "sif",
                     a: float = 1e-3) -> np.ndarray:
        """Pooling weight per in-vocabulary token.

        "sif": a/(a+p(w)), with p(w) = 1/((rank+1)·ln V) guessed from the word's
        frequency rank. Smaller `a` favours rare words more; large `a` approaches
        "mean", which weights every token 1.
        """
        if weighting == "mean":
            return np.ones(len(tokens))
        if weighting != "sif":
            raise ValueError("weighting must be 'sif' or 'mean'")
        index = np.array([self.word_index[t] for t in tokens], dtype=np.int64)
        ranks = (index - np.searchsorted(self._specials, index)).astype(np.float64)
        return a / (a + 1.0 / ((ranks + 1.0) * self._zipf))

    def gated_token_vectors(self, text: Union[str, Sentence, TokenVectors], *,
                            weighting: str = "sif") -> TokenVectors:
        """Ablation for the sentence-wide vector: local (rounds=1) vectors
        multiplied by the sentence vector instead of message passing, then
        scaled to sum to 1. Pass rounds=1 TokenVectors to reuse them."""
        tv = text if isinstance(text, TokenVectors) else self.token_vectors(text, rounds=1)
        sentence = self.pool(tv, weighting=weighting)
        if np.isnan(sentence).any():
            return tv
        gated = tv.vectors * sentence
        gated = gated / np.maximum(gated.sum(axis=1, keepdims=True), self.eps)
        return replace(tv, vectors=gated)

    # -- visualisation --

    @torch.no_grad()
    def trace(self, text: str, word: str, *, rounds: int = 1,
              context_strength: float = 1.0) -> MixingTrace:
        """The steps behind the vector of `word` (its first occurrence in `text`)
        in the last round; the result equals token_vectors with the same settings."""
        if rounds < 1:
            raise ValueError("rounds must be at least 1")
        found = [s for s in split_sentences(text, self.spec) if word in s.tokens]
        if not found:
            raise ValueError(f"{word!r} is not a token of the text")
        sentence = found[0]

        rows, all_windows, token_slices, _ = self._layout([sentence])
        position = token_slices[0].start + sentence.tokens.index(word)
        windows = all_windows[(all_windows == position).any(dim=1)]
        if len(windows) == 0:
            raise ValueError(f"{word!r} has no window without out-of-vocabulary words")

        # Neighbours enter the last round with their vectors from the round before.
        vectors, _ = self._propagate(rows, all_windows, rounds - 1, 0.0, context_strength)
        latents = _unit_sum(vectors / self.dim_mass, self.eps)
        messages = torch.stack(self._with_strength(self._messages(latents, windows),
                                                   context_strength), dim=1)
        slots = (windows == position).int().argmax(dim=1)
        context = messages[torch.arange(len(windows), device=self.device), slots]
        own = self.factor[rows[position]]
        mixed = own * context
        ok = mixed.sum(dim=1) > self.eps
        if not ok.any():
            raise ValueError(f"the model gives every window around {word!r} (almost) no mass")
        windows, slots, context, mixed = windows[ok], slots[ok], context[ok], mixed[ok]

        mixed = _unit_sum(mixed, self.eps)
        vector = mixed.mean(dim=0, keepdim=True)

        seq = list(sentence.tokens)
        if self.spec.padded:
            seq = [BOS] * (self.spec.order - 1) + seq + [EOS]
        return MixingTrace(
            word=word,
            rounds=rounds,
            windows=tuple(tuple(seq[p] for p in w) for w in windows.tolist()),
            slots=tuple(slots.tolist()),
            own=_unit_sum(own[None], self.eps)[0].cpu().numpy(),
            context=_unit_sum(context, self.eps).cpu().numpy(),
            mixed=mixed.cpu().numpy(),
            vector=vector[0].cpu().numpy(),
        )

    def plot_mixing(self, text: str, word: str, *, rounds: int = 1,
                    context_strength: float = 1.0, top_dims: int = 3):
        """Plot `trace`: one panel per window with the word's own row and the
        context's prediction as bars and their combination as a line, then a
        panel with the word's vector. Returns the figure."""
        tr = self.trace(text, word, rounds=rounds, context_strength=context_strength)
        dims = np.arange(self.rank)
        n_windows = len(tr.windows)
        top = float(max(tr.own.max(), tr.context.max(), tr.mixed.max(), tr.vector.max())) * 1.1

        fig, axes = plt.subplots(
            n_windows + 2, 1, figsize=(12, 0.6 + 2.0 * (n_windows + 1)),
            gridspec_kw={"height_ratios": [0.3] + [4] * (n_windows + 1)},
            constrained_layout=True, facecolor=_SURFACE,
        )
        where = f"round {rounds}, context strength {context_strength:g}"
        fig.suptitle(f"How the vector of '{word}' is built ({where})",
                     x=0.01, ha="left", color=_INK, fontsize=12)
        axes[0].axis("off")
        axes[0].legend(
            handles=[Patch(color=_OWN, label=f"'{word}': its own factor row"),
                     Patch(color=_CONTEXT, label="context: what the other words in the window predict"),
                     Line2D([], [], color=_INK, linewidth=1.5, label="combined: own × context")],
            loc="center left", ncol=3, frameon=False, fontsize=9, labelcolor=_INK_2,
        )

        for i, ax in enumerate(axes[1:-1]):
            ax.bar(dims, tr.context[i], width=0.8, color=_CONTEXT, zorder=2)
            ax.bar(dims, tr.own, width=0.4, color=_OWN, zorder=3)
            ax.plot(dims, tr.mixed[i], color=_INK, linewidth=1.5, zorder=4,
                    solid_joinstyle="round", solid_capstyle="round")
            words = [f"[{w}]" if k == tr.slots[i] else w for k, w in enumerate(tr.windows[i])]
            ax.set_title(f"window {i + 1} of {n_windows}:  " + "  ".join(words),
                         loc="left", color=_INK, fontsize=10)

        ax = axes[-1]
        ax.bar(dims, tr.own, width=0.4, color=_OWN, zorder=3)
        ax.plot(dims, tr.vector, color=_INK, linewidth=1.5, zorder=4,
                solid_joinstyle="round", solid_capstyle="round")
        names = {i: w for w, i in self.word_index.items()}
        largest = np.argsort(tr.vector)[::-1][:top_dims]
        column = self.factor[:, torch.as_tensor(largest.copy(), device=self.device)].clone()
        column[torch.as_tensor(self._specials, device=self.device)] = -1.0
        best = column.argmax(dim=0).tolist()
        labels = ", ".join(f"{d} ({names[b]})" for d, b in zip(largest.tolist(), best))
        how = f"mean of the {n_windows} combined lines"
        ax.set_title(f"'{word}' vector = {how}   ·   largest dimensions (top word): {labels}",
                     loc="left", color=_INK, fontsize=10)
        ax.set_xlabel("dimension", color=_MUTED, fontsize=9)

        for ax in axes[1:]:
            ax.set_facecolor(_SURFACE)
            ax.set_xlim(-1, self.rank)
            ax.set_ylim(0, top)
            ax.set_xticks(range(0, self.rank, 10))
            ax.grid(axis="y", color=_GRID, linewidth=0.75)
            ax.spines[["top", "right"]].set_visible(False)
            ax.spines[["left", "bottom"]].set_color(_AXIS)
            ax.tick_params(colors=_MUTED, labelsize=8)
            ax.set_ylabel("share", color=_MUTED, fontsize=9)
        for ax in axes[1:-1]:
            ax.tick_params(labelbottom=False)
        return fig


def _to_torch(x, device: torch.device) -> torch.Tensor:
    return (x if isinstance(x, torch.Tensor) else torch.as_tensor(x)).to(device)


def _unit_sum(x: torch.Tensor, eps: float) -> torch.Tensor:
    """Scale each row to sum to 1; zero rows stay zero."""
    return x / x.sum(dim=1, keepdim=True).clamp(min=eps)


def _require_tied(model) -> None:
    need = set(resolve_shared_factors("all", len(model.roles)))
    have = {tuple(pair) for pair in (model.shared_factors or ())}
    if not need <= have:
        raise ValueError(
            "ContextualEncoder needs fully tied factors (shared_factors='all'); "
            f"this model ties {sorted(have) or 'nothing'}."
        )


def _concat(parts: List[TokenVectors], rank: int, dtype) -> TokenVectors:
    if not parts:
        return TokenVectors((), np.zeros((0, rank), dtype=dtype),
                            np.zeros(0, dtype=bool), ())
    return TokenVectors(
        tuple(t for p in parts for t in p.tokens),
        np.concatenate([p.vectors for p in parts]),
        np.concatenate([p.mask for p in parts]),
        tuple(s for p in parts for s in p.spans),
        sum(p.n_windows for p in parts),
        sum(p.n_kept for p in parts),
    )


__all__ = [
    "NgramSpec",
    "Sentence",
    "TokenVectors",
    "MixingTrace",
    "ContextualEncoder",
    "split_sentences",
    "cosine",
]
