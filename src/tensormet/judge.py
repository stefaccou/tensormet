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

import math
import random
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F

# DEFAULT_JUDGE_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
DEFAULT_JUDGE_MODEL = "Qwen/Qwen3.5-2B" # heavier model, but needed for good performance

def _gpu_free_bytes(device) -> int:
    """Conservative 'free bytes now' estimate,
    --> torch analogue of the CuPy-side``distance._gpu_free_bytes``:
    the driver's own free figure (``cudaMemGetInfo``
    via ``torch.cuda.mem_get_info``) plus the caching allocator's already-reserved
    but currently-unallocated bytes, which are reusable without a fresh
    cudaMalloc. Deliberately does not call ``torch.cuda.empty_cache()`` first —
    same reasoning as the CuPy version: that flush is a synchronizing
    cudaFree/cudaMalloc round trip, and this is called once per chunk.
    """
    free_b, _total_b = torch.cuda.mem_get_info(device)
    reserved = torch.cuda.memory_reserved(device)
    allocated = torch.cuda.memory_allocated(device)
    pool_reusable = max(0, reserved - allocated)
    return int(free_b) + int(pool_reusable)


def _estimate_chunk_rows(vocab_size: int, seq_len: int, free_b: int,
                          safety: float = 0.7, temp_mult: float = 6.0) -> int:
    """Estimate how many rows of length ``seq_len`` fit in ``free_b`` bytes.

    Mirrors ``distance._estimate_batch_*``: bytes-per-row times a safety
    fraction of currently-free memory. The bottleneck is the logits/log-probs
    tensors of shape ``[rows, seq_len, vocab_size]`` that ``score_sequences``
    materializes in float32 — the fp16 model output, its ``.float()`` cast, and
    ``log_softmax``'s own output tensor all briefly coexist, so temp_mult=6.0
    (~1.5x a single fp32 copy) covers that overlap instead of just one copy.
    """
    bytes_per_row = int(math.ceil(seq_len * vocab_size * 4 * temp_mult))
    budget_b = int(free_b * safety)
    return max(1, budget_b // max(1, bytes_per_row))

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
                 chunk: int = 64,
                 device=None):
        self.model_name = model_name
        self.num_dim_words = num_dim_words
        self.diversity_aware = diversity_aware
        self.chunk = chunk
        # Where to load the model when it's time (e.g. select_gpu()'s pick);
        # None = auto (cuda:0 if available, else CPU). Survives unload().
        self.target_device = torch.device(device) if device is not None else None
        self.model = None
        self.tokenizer = None
        self.device = None
        self._pad_id = None
        # Measured (not guessed) GPU footprint of the loaded weights, set by
        # _load_on(); None until loaded, and None again once loaded on CPU.
        self.gpu_memory_bytes: Optional[int] = None

    @property
    def loaded(self) -> bool:
        return self.model is not None

    @property
    def gpu_memory_gb(self) -> Optional[float]:
        """Measured GPU memory used by the loaded weights (None if not CUDA-resident)."""
        return None if self.gpu_memory_bytes is None else self.gpu_memory_bytes / (1024 ** 3)

    def _load_on(self, device) -> None:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        is_cuda = device.type == "cuda"
        if is_cuda:
            torch.cuda.synchronize(device)
            mem_before = torch.cuda.memory_allocated(device)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name, dtype=torch.float16
        ).to(device)
        self.model.eval()
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self._pad_id = (self.tokenizer.pad_token_id
                        if self.tokenizer.pad_token_id is not None
                        else self.tokenizer.eos_token_id)
        self.device = device
        if is_cuda:
            torch.cuda.synchronize(device)
            self.gpu_memory_bytes = torch.cuda.memory_allocated(device) - mem_before
        else:
            self.gpu_memory_bytes = None

    def _ensure_loaded(self) -> None:
        if self.loaded:
            return
        if self.target_device is not None:
            target = self.target_device
        else:
            target = torch.device(0) if torch.cuda.is_available() else torch.device("cpu")
        print(f"Loading dimension-consistency judge {self.model_name!r} onto {target} (fp16)...")
        try:
            self._load_on(target)
        except torch.cuda.OutOfMemoryError:
            print(f"WARNING: not enough GPU memory to load judge {self.model_name!r}; "
                  "falling back to CPU (dimension-consistency scoring will be slower).")
            self.model = None
            torch.cuda.empty_cache()
            self._load_on(torch.device("cpu"))
        if self.gpu_memory_gb is not None:
            print(f"Judge model loaded on {self.device} using {self.gpu_memory_gb:.2f} GB GPU memory.")
        else:
            print(f"Judge model loaded on {self.device}.")

    def ensure_loaded(self) -> None:
        """Load the model now if it isn't already (idempotent).

        Call this *before* the decomposition starts sizing its per-iteration GPU
        batches, so the judge's GPU footprint (see `gpu_memory_gb` once loaded) is
        already resident and reflected in every free-VRAM estimate. Loading it
        lazily instead lets batches sized beforehand (e.g. on a resumed run) be
        computed against memory the judge later steals, OOMing the next factor
        update.
        """
        self._ensure_loaded()

    def unload(self) -> None:
        """Release the judge model (frees its measured `gpu_memory_gb` when CUDA-resident)."""
        was_cuda = self.device is not None and self.device.type == "cuda"
        self.model = None
        self.tokenizer = None
        self.device = None
        self._pad_id = None
        self.gpu_memory_bytes = None
        if was_cuda:
            torch.cuda.empty_cache()

    def build_prompt(self, candidates: list[str]) -> str:
        """Chat prompt asking the judge model to name the outlier, up to (but not
        including) the assistant's answer."""
        listing = ", ".join(candidates)
        messages = [
            {"role": "user",
             "content": f"Which word does not belong with the others? {listing}. "
                        "Answer with only the word."
             },
        ]
        # messages = [
        #     {"role": "system",
        #      "content": "You are a helpful assistant tasked with identifying the outlier in a list of words."},
        #     {"role": "user", "content": "Which word does not belong? apple, banana, orange, car, grape."},
        #     {"role": "assistant", "content": "car"},
        #     {"role": "user", "content": "Which word does not belong? like, love, message, adore, hate."},
        #     {"role": "assistant", "content": "message"},
        #     {"role": "user", "content": "Which word does not belong? similar, mail, same, equal, different."},
        #     {"role": "assistant", "content": "mail"},
        #     {"role": "user", "content": f"Which word does not belong? {listing}. Answer with only the word."},
        # ]
        return self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
            temperature=0.7, top_p=0.8, top_k=20, min_p=0.0, presence_penalty=1.5, repetition_penalty=1.0,
            extra_body={
                "chat_template_kwargs": {"enable_thinking": False},
            },
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

        The window size is capped at ``self.chunk`` but shrinks below it under GPU
        memory pressure: each window is first probed at up to ``self.chunk`` rows
        to find its worst-case padded length, then re-sized against currently-free
        VRAM via ``_gpu_free_bytes``/``_estimate_chunk_rows`` (same machinery as
        the CuPy-side batch estimators in distance.py). This is what keeps a
        chunk from overshooting available memory when a batch happens to contain
        an unusually long prompt/completion.
        """
        prefix_ids = [self.tokenizer(p, add_special_tokens=False).input_ids for p in prompts]
        comp_ids   = [self.tokenizer(c, add_special_tokens=False).input_ids for c in completions]
        row_lens = [len(p) + len(c) for p, c in zip(prefix_ids, comp_ids)]

        is_cuda = self.device.type == "cuda"
        vocab_size = self.model.config.vocab_size if is_cuda else None

        out = [0.0] * len(prompts)
        n = len(prompts)
        s = 0
        while s < n:
            if is_cuda:
                probe_end = min(s + self.chunk, n)
                probe_L = max(row_lens[s:probe_end])
                free_b = _gpu_free_bytes(self.device)
                max_rows = _estimate_chunk_rows(vocab_size, probe_L, free_b)
                window = max(1, min(self.chunk, max_rows))
            else:
                window = self.chunk
            e = min(s + window, n)

            pref = prefix_ids[s:e]
            comp = comp_ids[s:e]
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
            s = e
        return out

    def _evaluate_tasks(self, tasks: list[dict], answer_key: str,
                        verbose: bool = False) -> int:
        """Shared batched scoring + evaluation for a list of outlier tasks.

        Each task must carry a "candidates" list; `answer_key` names the task
        key holding the true outlier. Tasks are mutated in place with a
        "scores" dict, a "predicted" pick, and a "correct" bool. Returns the
        number of tasks the judge got right.

        Used by `score()` (candidates = a Tucker dimension's top words + one
        random vocab word), `score_similarity_consistency()` (candidates = a
        query word's nearest neighbours + one random vocab word), and
        `benchmark()` (candidates = a labelled category's words + one word
        from a different category), so the batched-sweep machinery is
        written once.
        """
        prompts, completions, owner = [], [], []
        for t_idx, t in enumerate(tasks):
            prompt = self.build_prompt(t["candidates"])
            for c in t["candidates"]:
                prompts.append(prompt)
                completions.append(c)
                owner.append(t_idx)

        flat_scores = self.score_sequences(prompts, completions)

        for t in tasks:
            t["scores"] = {}
        for t_idx, comp, sc in zip(owner, completions, flat_scores):
            tasks[t_idx]["scores"][comp] = sc

        # The model is *asked* to name the word that doesn't belong, so the
        # candidate it is most likely to emit (highest log-prob) is the
        # predicted outlier -> max, not min.
        correct = 0
        for t in tasks:
            scores = t["scores"]
            predicted = max(scores, key=scores.get)
            t["predicted"] = predicted
            t["correct"] = predicted == t[answer_key]
            correct += t["correct"]
            if verbose:
                print(f"  {t[answer_key]!r} added to {t['candidates']}")
                print("  scores:", {k: round(v, 3) for k, v in scores.items()})
                print("  predicted:", predicted)
        return correct

    def score(self,
              tucker_decomp,
              intruder_choice_option: str = "teaLeaves",
              seed: int = 1,
              role: Optional[str] = None,
              verbose: bool = False,
              return_details: bool = False) -> dict:
        """Score every dimension of `role` (default: first role) with the outlier task.

        The candidate/outlier draws come from a local RNG seeded with `seed`, so the
        global random state is untouched and the same tasks are posed at every
        semantic check of a run (scores stay comparable across iterations).

        `intruder_choice_option` picks where the injected outlier comes from:
          "random"     a uniform draw from the vocabulary, absent from the top words.
          "teaLeaves"  a word that ranks in this dimension's bottom half but in the
                       top 10% of some dimension -- salient elsewhere, weak here, so
                       the judge cannot win by spotting a rare/odd token.

        The current `efficient` implementation for "teaLeaves": True ranks the whole (N, R) factor with
        one np.argsort, while the old reference implementation calls
        get_top_words_for_dimension once per dimension (which re-materializes the
        factor as numpy every time). The two can disagree on words with tied
        loadings, the rows under our sparsity setting, mostly and commonly, because torch.topk
        and argsort break ties differently, so many dimensions may draw a
        different intruder
        -> This hurts reproducibility, so we only ship the efficient version which is correct

        Returns a dict of [0,1] scores:
          dim_consistency           final score (accuracy × diversity multiplier when
                                    diversity_aware; the prototype's raw correct count
                                    is this × rank)
          dim_consistency_raw       plain accuracy before diversity rescaling
          dim_consistency_diversity distinct top words seen / max possible
                                    (only when diversity_aware)

        With `return_details` the dict additionally carries a "details" list with one
        entry per dimension (top words, injected outlier, the judge's pick, verdict,
        per-candidate log-probs) for inspection UIs. Keep it off inside the training
        loop, where the returned dict is merged into sem_out and JSON-dumped whole.
        """
        self._ensure_loaded()
        rng = random.Random(seed)

        role = role if role is not None else tucker_decomp.roles[0]
        role_idx = tucker_decomp.get_role_index(role)
        rank = int(tucker_decomp.core.shape[role_idx])
        vocab_list = tucker_decomp.vocab[f"vocab_{role}"]
        dims = tucker_decomp.get_dims()[role_idx]

        # 1. Build one outlier task per dimension.
        tasks = []
        all_dim_words = set() if self.diversity_aware else None


        # Option 1: random from vocab
        if intruder_choice_option == "random":
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
                tasks.append({"dim": i, "words": words, "candidates": candidates,
                              "random_word": random_word})
        # Option 2: a word from the bottom 50% of dimension i that is ALSO in the
        # top 10% of some dimension: plausible-looking but wrong for this dimension.
        elif intruder_choice_option == "teaLeaves":
            # Deferred import: tucker_tensor pulls in tensorly/CuPy and imports
            # this module lazily itself, so keep it out of judge's import time.

            from tensormet.tucker_tensor import _to_np
            n_words = dims                       # vocabulary size of this mode
            n_bottom = n_words // 2              # first index of the bottom half
            n_top = max(1, n_words // 10)

            # One descending argsort of the whole (N, R) factor, against `rank`
            # separate topk calls that each convert the factor to numpy again.
            factor = _to_np(tucker_decomp.factors[role_idx])[:, :rank]  # (N, R)
            order = np.argsort(-factor, axis=0, kind="stable")          # (N, R) vocab ids
            top_dim_words = {
                i: [vocab_list[j] for j in order[:self.num_dim_words, i]]
                for i in range(rank)
            }
            bottom_50_percents = {
                i: {vocab_list[j] for j in order[n_bottom:, i]}
                for i in range(rank)
            }
            top_10_percent_of_some_dimension_words = {
                vocab_list[j] for j in order[:n_top, :].ravel()
            }

            # # Reference implementation: one full topk per dimension.
            # top_10_percent_of_some_dimension_words = set()
            # top_dim_words = {}
            # bottom_50_percents = {}
            # for i in range(rank):
            #     seq = tucker_decomp.get_top_words_for_dimension(role, i, n_words)
            #     scored_list = [w for w, score in seq]
            #     bottom_50_percents[i] = set(scored_list[n_bottom:])
            #     top_dim_words[i] = scored_list[:self.num_dim_words]
            #     top_10_percent_of_some_dimension_words.update(scored_list[:n_top])

            if verbose:
                print(f"{len(top_10_percent_of_some_dimension_words)} candidate "
                      f"intruder words to choose from")

            for i in range(rank):
                words = top_dim_words[i]
                if self.diversity_aware:
                    all_dim_words.update(words)

                # Intersection, not union: salient elsewhere, weak here.
                # sorted() because set order over strings varies with PYTHONHASHSEED,
                # which would break seed reproducibility across processes.
                pool = sorted(
                    bottom_50_percents[i] & top_10_percent_of_some_dimension_words
                )
                if not pool:
                    raise ValueError(
                        f"No teaLeaves intruder available for dimension {i} of role "
                        f"{role!r}: no word is both in this dimension's bottom half "
                        f"and in the top 10% of another dimension."
                    )
                random_word = rng.choice(pool)

                candidates = words + [random_word]
                rng.shuffle(candidates)
                tasks.append({"dim": i, "words": words, "candidates": candidates,
                              "random_word": random_word})
        else:
            raise ValueError(
                f"intruder_choice_option must be 'random' or 'teaLeaves'; "
                f"got {intruder_choice_option!r}"
            )


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
        details = []
        for t in tasks:
            scores = t["scores"]
            outlier = max(scores, key=scores.get)
            if verbose:
                print(f"dim {t['dim']}: {t['random_word']} added to {t['candidates']}")
                print("  scores:", {k: round(v, 3) for k, v in scores.items()})
                print("  outlier:", outlier)
            is_correct = outlier == t["random_word"]
            if is_correct:
                correct += 1
            if return_details:
                details.append({
                    "dim": t["dim"],
                    "words": t["words"],
                    "outlier": t["random_word"],
                    "predicted": outlier,
                    "correct": is_correct,
                    "scores": {k: round(v, 3) for k, v in scores.items()},
                })

        raw = correct / rank
        out = {"dim_consistency": raw, "dim_consistency_raw": raw}
        if self.diversity_aware:
            mult = len(all_dim_words) / (rank * self.num_dim_words)
            out["dim_consistency"] = raw * mult
            out["dim_consistency_diversity"] = mult
        if return_details:
            out["details"] = details
        if verbose:
            print(f"dimension consistency: {correct}/{rank} correct -> "
                  f"{ {k: v for k, v in out.items() if k != 'details'} }")
        return out

    def score_similarity_consistency(self,
                                      tucker_decomp,
                                      query_words: Optional[list[str]] = None,
                                      role: Optional[str] = None,
                                      top_k: Optional[int] = None,
                                      seed: int = 1,
                                      verbose: bool = False,
                                      return_details: bool = False) -> dict:
        """Score consistency of `get_most_similar_elements` neighbourhoods with
        the same outlier-detection task as `score()`.

        Where `score()` probes latent *dimensions* (top-k words by loading),
        this probes the nearest-neighbour structure directly: for every word in
        `query_words` that is present in the vocabulary, fetch its top-`top_k`
        most similar words, inject one random vocab word absent from that
        neighbourhood, and ask the judge which word does not belong.
        Useful for checking neighbourhood quality on words that matter for downstream
        analysis (e.g. concepts drawn from an external list like the Master
        Metaphor List) rather than on arbitrary dimensions.

        `query_words` entries missing from the vocabulary are skipped (reported
        in `out["skipped"]` when `return_details`); `top_k` defaults to
        `self.num_dim_words` if not given. If `query_words` is omitted entirely,
        it defaults to the Master Metaphor List's concept words (fetched once
        and cached to disk -- see `tensormet.experimental.parse_master_metaphor_list.load_concepts`). That import
        is deferred here (like the judge model itself) so importing tensormet
        never requires the scraper's `requests`/`bs4` dependencies -- only
        calling this method without an explicit `query_words` does.

        Returns a dict of [0,1] scores, mirroring `score()`:
          similarity_consistency            final score (accuracy x diversity
                                             multiplier when diversity_aware)
          similarity_consistency_raw        plain accuracy before diversity rescaling
          similarity_consistency_diversity  distinct neighbour words seen / max
                                             possible (only when diversity_aware)
          n_queries                         number of query words actually scored
                                             (after skipping OOV words)
        """
        self._ensure_loaded()
        rng = random.Random(seed)

        if query_words is None:
            from tensormet.experimental.parse_master_metaphor_list import load_concepts
            query_words = load_concepts()

        role = role if role is not None else tucker_decomp.roles[0]
        k = top_k if top_k is not None else self.num_dim_words
        vocab_list = tucker_decomp.vocab[f"vocab_{role}"]
        vocab_set = set(vocab_list)

        skipped = [w for w in query_words if w not in vocab_set]
        kept = [w for w in query_words if w in vocab_set]
        if verbose and skipped:
            print(f"skipping {len(skipped)} query word(s) not in vocab: {skipped}")
        if not kept:
            raise ValueError("None of the query words were found in the vocabulary.")

        # 1. Build one outlier task per query word.
        tasks = []
        all_neighbor_words = set() if self.diversity_aware else None
        for w in kept:
            neighbors = tucker_decomp.get_most_similar_elements(w, role, top_k=k)
            if self.diversity_aware:
                all_neighbor_words.update(neighbors)
            # Outlier must be absent from the neighbour list (and from the
            # query word itself, which get_most_similar_elements may return
            # as its own nearest neighbour), otherwise the candidate list
            # could contain a duplicate and the scores dict would silently
            # collapse it.
            pool = [v for v in vocab_list if v not in neighbors and v != w]
            random_word = rng.choice(pool)

            candidates = neighbors + [random_word]
            rng.shuffle(candidates)
            tasks.append({"query": w, "words": neighbors, "candidates": candidates,
                          "random_word": random_word})

        # 2. Batched scoring + evaluation (shared with score()/benchmark()).
        correct = self._evaluate_tasks(tasks, answer_key="random_word", verbose=verbose)

        n = len(tasks)
        raw = correct / n
        out = {"similarity_consistency": raw, "similarity_consistency_raw": raw,
               "n_queries": n}
        if self.diversity_aware:
            mult = len(all_neighbor_words) / (n * k)
            out["similarity_consistency"] = raw * mult
            out["similarity_consistency_diversity"] = mult
        if return_details:
            out["details"] = [{
                "query": t["query"], "words": t["words"], "outlier": t["random_word"],
                "predicted": t["predicted"], "correct": t["correct"],
                "scores": {kk: round(vv, 3) for kk, vv in t["scores"].items()},
            } for t in tasks]
            out["skipped"] = skipped
        if verbose:
            print(f"similarity consistency: {correct}/{n} correct -> "
                  f"{ {kk: vv for kk, vv in out.items() if kk not in ('details', 'skipped')} }")
        return out

    def benchmark(self,
                  categories: dict,
                  num_words: Optional[int] = None,
                  n_trials: int = 50,
                  seed: int = 1,
                  verbose: bool = False,
                  return_details: bool = False) -> dict:
        """
        Benchmark on some pre-defined categories
        """
        self._ensure_loaded()
        k = num_words if num_words is not None else self.num_dim_words
        too_small = [name for name, words in categories.items() if len(words) < k]
        if too_small:
            raise ValueError(f"num_words={k} exceeds category size for: {too_small}")

        rng = random.Random(seed)
        names = list(categories.keys())
        tasks = []
        for i in range(n_trials):
            cat = rng.choice(names)
            words = rng.sample(categories[cat], k)
            outlier_cat = rng.choice([n for n in names if n != cat])
            outlier = rng.choice(categories[outlier_cat])
            candidates = words + [outlier]
            rng.shuffle(candidates)
            tasks.append({"trial": i, "category": cat, "outlier_category": outlier_cat,
                          "words": words, "candidates": candidates, "random_word": outlier})

        correct = self._evaluate_tasks(tasks, answer_key="random_word", verbose=verbose)

        per_cat: dict = {}
        for t in tasks:
            stats = per_cat.setdefault(t["category"], [0, 0])
            stats[0] += t["correct"]
            stats[1] += 1

        out = {
            "accuracy": correct / n_trials,
            "n_trials": n_trials,
            "per_category": {c: n_ok / n for c, (n_ok, n) in per_cat.items()},
        }
        if return_details:
            out["details"] = [{
                "trial": t["trial"], "category": t["category"],
                "outlier_category": t["outlier_category"], "words": t["words"],
                "outlier": t["random_word"], "predicted": t["predicted"],
                "correct": t["correct"],
                "scores": {k: round(v, 3) for k, v in t["scores"].items()},
            } for t in tasks]
        if verbose:
            print(f"benchmark accuracy: {correct}/{n_trials} -> {out['accuracy']:.3f}")
        return out
