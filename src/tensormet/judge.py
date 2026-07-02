"""LLM-as-judge "dimension consistency" scoring.

For every latent dimension of a Tucker factor, take the top-k words, inject one
random vocabulary word, and ask a small judge LLM which word does not belong.
The fraction of dimensions where the judge picks the injected word measures how
semantically coherent the learned dimensions are.

Ported from the prototype in 1_method_development/16_LLM_as_judge/outlier_scoring.py.
Heavy imports (transformers) and the model load itself are deferred to first use so
tensormet stays importable — and existing runs stay untouched — when the metric is
disabled (the default).
"""
from __future__ import annotations

import random
from typing import Optional

import torch
import torch.nn.functional as F

DEFAULT_JUDGE_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"


class DimConsistencyJudge:
    """Lazily-loaded judge model that scores dimension consistency of a
    TuckerDecomposition via the outlier-detection task.

    Constructing the judge is free: the model/tokenizer are only loaded on the
    first call to score(). The decomposition GPU is typically near-full with
    CuPy pools at that point, so the caller should trim those pools right
    before the first score() (see non_negative_tucker_with_similarity); on CUDA
    OOM the judge falls back to CPU with a warning instead of failing the run.
    """

    def __init__(self,
                 model_name: str = DEFAULT_JUDGE_MODEL,
                 num_dim_words: int = 5,
                 diversity_aware: bool = True,
                 chunk: int = 64):
        self.model_name = model_name
        self.num_dim_words = num_dim_words
        self.diversity_aware = diversity_aware
        self.chunk = chunk
        self.model = None
        self.tokenizer = None
        self.device = None
        self._pad_id = None

    @property
    def loaded(self) -> bool:
        return self.model is not None

    def _load_on(self, device) -> None:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name, dtype=torch.float16
        ).to(device)
        self.model.eval()
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self._pad_id = (self.tokenizer.pad_token_id
                        if self.tokenizer.pad_token_id is not None
                        else self.tokenizer.eos_token_id)
        self.device = device

    def _ensure_loaded(self) -> None:
        if self.loaded:
            return
        want_cuda = torch.cuda.is_available()
        target = torch.device(0) if want_cuda else torch.device("cpu")
        print(f"Loading dimension-consistency judge {self.model_name!r} onto {target} "
              f"(fp16, ~1 GB for the default 0.5B model)...")
        try:
            self._load_on(target)
        except torch.cuda.OutOfMemoryError:
            print(f"WARNING: not enough GPU memory to load judge {self.model_name!r}; "
                  "falling back to CPU (dimension-consistency scoring will be slower).")
            self.model = None
            torch.cuda.empty_cache()
            self._load_on(torch.device("cpu"))
        print(f"Judge model loaded on {self.device}.")

    def unload(self) -> None:
        """Release the judge model (frees ~1 GB of GPU memory when CUDA-resident)."""
        was_cuda = self.device is not None and self.device.type == "cuda"
        self.model = None
        self.tokenizer = None
        self.device = None
        self._pad_id = None
        if was_cuda:
            torch.cuda.empty_cache()

    def build_prompt(self, candidates: list[str]) -> str:
        """Chat prompt asking the judge model to name the outlier, up to (but not
        including) the assistant's answer."""
        listing = ", ".join(candidates)
        messages = [
            {"role": "user",
             "content": f"Which word does not belong with the others? {listing}. "
                        "Answer with only the word."},
        ]
        return self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

    @torch.no_grad()
    def score_sequences(self, prompts: list[str], completions: list[str]) -> list[float]:
        """Batched, length-normalised log-prob of each completion given its prompt.

        `prompts[i]` is scored against `completions[i]` (parallel lists). Prefix and
        completion lengths may differ from row to row, so the completion span is
        tracked per row rather than assuming a single shared prefix length. This lets
        a single batch mix candidates from *different* dimensions (whose prompts differ).

        Right-padding is safe: with a causal model + attention mask, pad tokens sit to
        the right of every completion and never influence the scored positions.
        """
        prefix_ids = [self.tokenizer(p, add_special_tokens=False).input_ids for p in prompts]
        comp_ids   = [self.tokenizer(c, add_special_tokens=False).input_ids for c in completions]

        out = [0.0] * len(prompts)
        for s in range(0, len(prompts), self.chunk):
            pref = prefix_ids[s:s + self.chunk]
            comp = comp_ids[s:s + self.chunk]
            rows = [p + c for p, c in zip(pref, comp)]
            L = max(len(r) for r in rows)

            ids  = torch.full((len(rows), L), self._pad_id, dtype=torch.long)
            attn = torch.zeros((len(rows), L), dtype=torch.long)
            for i, r in enumerate(rows):
                ids[i, :len(r)]  = torch.tensor(r, dtype=torch.long)
                attn[i, :len(r)] = 1
            ids, attn = ids.to(self.device), attn.to(self.device)

            logits = self.model(ids, attention_mask=attn).logits
            # logits at position t predict token t+1, so shift by one.
            log_probs = F.log_softmax(logits[:, :-1, :].float(), dim=-1)
            tgt = ids[:, 1:]
            token_logps = log_probs.gather(2, tgt.unsqueeze(-1)).squeeze(-1)  # [b, L-1]

            for i, (p, c) in enumerate(zip(pref, comp)):
                start = len(p) - 1          # -1 for the shift above
                end = start + len(c)
                out[s + i] = token_logps[i, start:end].mean().item()
        return out

    def score(self,
              tucker_decomp,
              seed: int = 1,
              role: Optional[str] = None,
              verbose: bool = False) -> dict:
        """Score every dimension of `role` (default: first role) with the outlier task.

        The candidate/outlier draws come from a local RNG seeded with `seed`, so the
        global random state is untouched and the same tasks are posed at every
        semantic check of a run (scores stay comparable across iterations).

        Returns a dict of [0,1] scores:
          dim_consistency           final score (accuracy × diversity multiplier when
                                    diversity_aware; the prototype's raw correct count
                                    is this × rank)
          dim_consistency_raw       plain accuracy before diversity rescaling
          dim_consistency_diversity distinct top words seen / max possible
                                    (only when diversity_aware)
        """
        self._ensure_loaded()
        rng = random.Random(seed)

        role = role if role is not None else tucker_decomp.roles[0]
        role_idx = tucker_decomp.get_role_index(role)
        rank = int(tucker_decomp.core.shape[role_idx])
        vocab_list = tucker_decomp.vocab[f"vocab_{role}"]

        # 1. Build one outlier task per dimension.
        tasks = []
        all_dim_words = set() if self.diversity_aware else None
        for i in range(rank):
            seq = tucker_decomp.get_top_words_for_dimension(role, i, self.num_dim_words)
            words = [w for w, score in seq]
            if self.diversity_aware:
                all_dim_words.update(words)
            # Draw an outlier that is absent from the top words, otherwise the
            # candidate list would contain a duplicate and the scores dict would
            # silently collapse it (making `outlier == random_word` ambiguous).
            pool = [w for w in vocab_list if w not in words]
            random_word = rng.choice(pool)

            candidates = words + [random_word]
            rng.shuffle(candidates)
            tasks.append({"dim": i, "candidates": candidates, "random_word": random_word})

        # 2. Flatten to parallel (prompt, completion) lists, remembering the owning task.
        prompts, completions, owner = [], [], []
        for t_idx, t in enumerate(tasks):
            prompt = self.build_prompt(t["candidates"])
            for c in t["candidates"]:
                prompts.append(prompt)
                completions.append(c)
                owner.append(t_idx)

        # 3. One batched sweep over everything.
        flat_scores = self.score_sequences(prompts, completions)

        # 4. Regroup scores back onto their task.
        for t in tasks:
            t["scores"] = {}
        for t_idx, comp, sc in zip(owner, completions, flat_scores):
            tasks[t_idx]["scores"][comp] = sc

        # 5. Evaluate. The model is *asked* to name the word that doesn't belong,
        # so the candidate it is most likely to emit (highest log-prob) is the
        # predicted outlier -> max, not min.
        correct = 0
        for t in tasks:
            scores = t["scores"]
            outlier = max(scores, key=scores.get)
            if verbose:
                print(f"dim {t['dim']}: {t['random_word']} added to {t['candidates']}")
                print("  scores:", {k: round(v, 3) for k, v in scores.items()})
                print("  outlier:", outlier)
            if outlier == t["random_word"]:
                correct += 1

        raw = correct / rank
        out = {"dim_consistency": raw, "dim_consistency_raw": raw}
        if self.diversity_aware:
            mult = len(all_dim_words) / (rank * self.num_dim_words)
            out["dim_consistency"] = raw * mult
            out["dim_consistency_diversity"] = mult
        if verbose:
            print(f"dimension consistency: {correct}/{rank} correct -> {out}")
        return out
