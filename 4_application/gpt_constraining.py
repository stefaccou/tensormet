import torch
import spacy
import xml.etree.ElementTree as ET
from html.entities import name2codepoint
from utils import DATA_DIR
nlp = spacy.load("nl_core_news_md")

# -------------------------------------------------
# Inflection utilities
# -------------------------------------------------

def build_inflection_bank(lexicon_path):
    parser = ET.XMLParser()
    parser.entity.update({name: chr(cp) for name, cp in name2codepoint.items()})
    tree = ET.parse(lexicon_path, parser=parser)
    root = tree.getroot()
    verbs_to_forms = {}
    for entry in root.findall(".//entry"):
        lemma = entry.find("lem")
        wordform_set = set()
        for word in entry.findall("wordf"):
            if word.find("pos").text.startswith("WW"):
                wordform = word.find("orth")
                wordform_set.add(wordform.text)
        if wordform_set:
            verbs_to_forms[lemma.text] = wordform_set

    return verbs_to_forms

verbs_to_forms = build_inflection_bank(DATA_DIR/"lexica/e-Lex-1.1.xml")

def get_dutch_inflections(verb_inf):
    """
    We will now get them directly from elex
    """
    forms = verbs_to_forms.get(verb_inf, set())
    # ook varianten met voorafgaande spatie
    spaced_forms = {" " + f for f in forms}
    forms |= spaced_forms

    return forms


# -------------------------------------------------
# Scoring/picking utilities
# -------------------------------------------------
# we add a candidate scoring function for multi-word candidates
@torch.no_grad()
def score_candidate(model, toks, candidate_ids):
    # toks: current prompt state
    # candidate_ids: tensor([id1, id2, ...])
    # returns total logprob of generating that exact sequence next
    input_ids = toks['input_ids']
    attn = toks['attention_mask']

    total_logprob = 0.0
    cur_input_ids = input_ids.clone()
    cur_attn = attn.clone()

    for next_id in candidate_ids:
        out = model(input_ids=cur_input_ids, attention_mask=cur_attn)
        logits_last = out.logits[0, -1, :]           # (vocab,)
        logprobs = torch.log_softmax(logits_last, dim=-1)
        total_logprob += logprobs[next_id].item()

        # append token
        next_id_tensor = next_id.view(1,1)
        cur_input_ids = torch.cat([cur_input_ids, next_id_tensor], dim=-1)
        cur_attn = torch.cat([cur_attn, torch.ones((1,1), dtype=torch.long)], dim=-1)

    # normalize a bit for length so longer verbs aren't unfairly punished
    avg_logprob = total_logprob / candidate_ids.size(0)
    return avg_logprob


@torch.no_grad()
def pick_word_inflection(model, tokenizer, base_words, toks):
    """
    Kies de meest waarschijnlijke inflectievorm (volgende token) voor
    eender welk woord in base_words.
    Scoring = enkel eerste token van de kandidaat.
    """
    output = model(**toks)
    logits = output.logits
    next_token_logits = logits[0, -1, :]

    best_form = None
    best_ids = None
    best_score = float('-inf')

    # loop over alle basiswoorden en hun inflecties
    for w in base_words:
        for form in get_dutch_inflections(w):
            token_ids = tokenizer(form, return_tensors="pt")['input_ids'][0]

            # skip als leeg
            if token_ids.numel() == 0:
                continue

            # score = logit van eerste token-id
            cand_score = next_token_logits[token_ids[0].item()].item()

            if cand_score > best_score:
                best_score = cand_score
                best_ids = token_ids
                best_form = form

    print(f"Picked word: {best_form} with score {best_score}")

    # plak gekozen vorm achter toks
    new_input_ids = torch.cat(
        [toks['input_ids'], best_ids.unsqueeze(0)],
        dim=-1
    )
    new_attention_mask = torch.cat(
        [toks['attention_mask'],
         torch.ones((1, best_ids.size(0)), dtype=torch.long)],
        dim=-1
    )

    return {
        'input_ids': new_input_ids,
        'attention_mask': new_attention_mask
    }
@torch.no_grad()
def pick_word_inflection(model, tokenizer, base_words, toks, top_n_check=5):
    """
    Fast path:
      1. Get logits for *next* token once.
      2. For each candidate form (and its tokenization),
         record the first-token id and that first-token logit.
      3. Keep only candidates whose first-token id is among the model's own
         top_n_check next-token ids.
      4. If none survive, just pick the single best first-token candidate
         (cheap).
      5. If some survive, run the expensive multi-token scoring ONLY on that
         shortlist using score_candidate(), then pick the best.

    Returns new toks with the chosen candidate appended.
    """

    device = toks['input_ids'].device

    # forward once
    output = model(**toks)
    logits = output.logits
    next_token_logits = logits[0, -1, :]  # (vocab,)

    # get model's own top-N next-token guesses
    topk_vals, topk_idx = torch.topk(next_token_logits, k=top_n_check)
    topk_idx_set = set(topk_idx.tolist())

    # we'll collect candidate infos here
    cand_list = []
    best_fast_form = None
    best_fast_ids = None
    best_fast_score = float('-inf')

    # build candidates (first token only, cheap)
    for w in base_words:
        for form in get_dutch_inflections(w):
            ids = tokenizer(form, return_tensors="pt")['input_ids'][0]

            if ids.numel() == 0:
                continue

            first_id = ids[0].item()
            first_score = next_token_logits[first_id].item()

            # track global best by first-token score (fast fallback)
            if first_score > best_fast_score:
                best_fast_score = first_score
                best_fast_form = form
                best_fast_ids = ids

            # collect for potential slow scoring if it's in top-N
            if first_id in topk_idx_set:
                cand_list.append((form, ids, first_score))

    # if nothing qualified for deeper check, just use fast best
    if not cand_list:
        chosen_form = best_fast_form
        chosen_ids = best_fast_ids
    else:
        # shortlist slow scoring with full autoregressive score
        best_full_score = float('-inf')
        chosen_form = None
        chosen_ids = None

        for form, ids, _first_score in cand_list:
            # IMPORTANT: run score_candidate only for the shortlist
            sc = score_candidate(model, toks, ids.to(device))

            if sc > best_full_score:
                best_full_score = sc
                chosen_form = form
                chosen_ids = ids

    # append chosen_ids to toks (device-safe)
    chosen_ids = chosen_ids.to(device)
    new_input_ids = torch.cat(
        [toks['input_ids'], chosen_ids.unsqueeze(0)], dim=-1
    )
    new_attention_mask = torch.cat(
        [
            toks['attention_mask'],
            torch.ones((1, chosen_ids.size(0)), dtype=torch.long, device=device)
        ],
        dim=-1
    )

    print(f"Picked word: {chosen_form}")

    return {
        'input_ids': new_input_ids,
        'attention_mask': new_attention_mask
    }

@torch.no_grad()
def pick_word_inflection_scored(
    model,
    tokenizer,
    scores,          # dict: base_word -> float bias
    toks,
    top_n_check=5    # shortlist size trigger for expensive scoring
):
    """
    Select the best inflected form among all base words in `scores`.

    Speed trick:
    1. Single forward pass to get next-token logits.
    2. Rank all candidate forms by (first-token logit + bias).
       Keep:
         - global best (fast fallback),
         - shortlist = forms whose first-token id is in the model's own
           top_n_check next-token ids.
    3. If shortlist non-empty: re-score only those with full autoregressive
       score_candidate(...) + bias. Otherwise use the fast fallback.

    This drastically reduces slow per-candidate scoring while still avoiding
    garbage like "hord"/"ette".
    """

    device = toks['input_ids'].device

    # forward once
    output = model(**toks)
    logits = output.logits
    next_token_logits = logits[0, -1, :]  # (vocab,)

    # model's own top-N guesses for next token
    k = min(top_n_check, next_token_logits.size(-1))
    topk_vals, topk_idx = torch.topk(next_token_logits, k=k)
    topk_idx_set = set(topk_idx.tolist())

    # tracking
    best_fast_form = None
    best_fast_ids = None
    best_fast_score = float('-inf')

    shortlist = []  # list of (form, token_ids, bias)

    # loop all base words + their inflections
    for w, bias in scores.items():
        for form in get_dutch_inflections(w):
            tokenized = tokenizer(form, return_tensors="pt")['input_ids'][0]

            if tokenized.numel() == 0:
                continue

            first_id = tokenized[0].item()
            first_logit = next_token_logits[first_id].item()

            # cheap combined score: first-token logit + bias
            cheap_score = first_logit + bias

            # track best cheap candidate globally (fallback)
            if cheap_score > best_fast_score:
                best_fast_score = cheap_score
                best_fast_form = form
                best_fast_ids = tokenized

            # if first token is already plausible for the LM, keep for deeper check
            if first_id in topk_idx_set:
                shortlist.append((form, tokenized, bias))

    # if no shortlist, just use the global best fast match
    if not shortlist:
        chosen_form = best_fast_form
        chosen_ids = best_fast_ids.to(device)

    else:
        # expensive scoring only on shortlist
        best_full_score = float('-inf')
        chosen_form = None
        chosen_ids = None

        for form, tokenized, bias in shortlist:
            # full autoregressive score across ALL subtokens
            full_seq_score = score_candidate(model, toks, tokenized.to(device))
            # add external bias
            total = full_seq_score + bias

            if total > best_full_score:
                best_full_score = total
                chosen_form = form
                chosen_ids = tokenized.to(device)

        # safety fallback just in case, though chosen_ids should be set
        if chosen_ids is None:
            chosen_form = best_fast_form
            chosen_ids = best_fast_ids.to(device)

    # append chosen_ids to toks in a device-safe way
    new_input_ids = torch.cat(
        [toks['input_ids'], chosen_ids.unsqueeze(0)], dim=-1
    )
    new_attention_mask = torch.cat(
        [
            toks['attention_mask'],
            torch.ones(
                (1, chosen_ids.size(0)),
                dtype=torch.long,
                device=device
            ),
        ],
        dim=-1
    )

    print(f"Picked word: {chosen_form}")
    return {
        'input_ids': new_input_ids,
        'attention_mask': new_attention_mask
    }


def _clone_state(toks):
    return {
        'input_ids': toks['input_ids'].clone(),
        'attention_mask': toks['attention_mask'].clone()
    }


def _sample_next_token(logits_step, toks, temperature=0.7, k=50, rep_penalty=1.2):
    """
    Sampling step with:
    - repetition penalty
    - temperature
    - top-k
    """
    ls = logits_step.clone()

    # repetition penalty
    for token_id in toks['input_ids'][0]:
        ls[token_id] /= rep_penalty

    # temperature
    ls = ls / temperature

    # top-k filter
    topk_vals, topk_idx = torch.topk(ls, k)
    probs = torch.softmax(topk_vals, dim=-1)

    choice_idx = torch.multinomial(probs, num_samples=1)
    next_token_id = topk_idx[choice_idx].view(1, 1)

    return next_token_id


@torch.no_grad()
def _generate_with_inject(
    model,
    tokenizer,
    input_text,
    inject_fn,
    max_length=20,
    repeat=False,
    temperature=0.7,
    top_k=50,
):
    """
    Autoregressieve generatie met:
    - sampling (temp/top-k/repetition penalty)
    - detecteer eerste werkwoord in de huidige zin via spaCy
    - vervang die plek door een gekozen (geïnfecteerde) target_word,
      via inject_fn(model, tokenizer, toks_prev) -> new_toks
    """

    toks = tokenizer(input_text, return_tensors="pt")

    saw_verb_this_sent = False
    injected_verb_this_sent = False

    for _ in range(max_length):
        prev_state = _clone_state(toks)

        # forward
        output = model(**toks)
        logits_step = output.logits[0, -1, :]

        # sample één token
        next_token_id = _sample_next_token(
            logits_step,
            toks,
            temperature=temperature,
            k=top_k,
            rep_penalty=1.2
        )

        # append dit token
        toks = {
            'input_ids': torch.cat([toks['input_ids'], next_token_id], dim=-1),
            'attention_mask': torch.cat(
                [toks['attention_mask'], torch.tensor([[1]])],
                dim=-1
            )
        }

        # decode huidig tot tekst
        current_text = tokenizer.decode(
            toks['input_ids'][0],
            skip_special_tokens=False
        )

        # laatste zin parsen met spaCy
        doc = nlp(current_text)
        last_sent = list(doc.sents)[-1]

        # check: zit er al een werkwoord in deze zin?
        has_verb = any(token.pos_ == "VERB" for token in last_sent)

        if has_verb and not injected_verb_this_sent:
            # rollback en injecteer custom candidate
            toks = prev_state
            toks = inject_fn(model, tokenizer, toks)

            injected_verb_this_sent = True
            saw_verb_this_sent = True
        else:
            saw_verb_this_sent = saw_verb_this_sent or has_verb

        # zin afgelopen? reset flags als repeat==True
        if current_text.rstrip().endswith(('.', '!', '?', ',', "'", '"')) and repeat:
            saw_verb_this_sent = False
            injected_verb_this_sent = False

    return tokenizer.decode(toks['input_ids'][0], skip_special_tokens=False)


# Thin wrappers that define the injection policy:

def _inject_unscored(target_words):
    @torch.no_grad()
    def _fn(model, tokenizer, toks):
        return pick_word_inflection(model, tokenizer, target_words, toks)
    return _fn


def _inject_scored(target_word_scores):
    @torch.no_grad()
    def _fn(model, tokenizer, toks):
        return pick_word_inflection_scored(model, tokenizer, target_word_scores, toks)
    return _fn


# -------------------------------------------------
# Main generation loop
# -------------------------------------------------


@torch.no_grad()
def generate_with_selection(model, tokenizer, input_text, target_words,
                            temperature=0.7, top_k=50,
                            max_length=20, repeat=False):
    """
    Convenience wrapper for unscored target_words (list/iterable).
    """
    return _generate_with_inject(
        model=model,
        tokenizer=tokenizer,
        input_text=input_text,
        inject_fn=_inject_unscored(target_words),
        max_length=max_length,
        repeat=repeat,
        temperature=temperature,
        top_k=top_k,
    )


@torch.no_grad()
def generate_with_selection_scored(model, tokenizer, input_text, target_word_scores,
                                   temperature=0.7, top_k=50,
                                   max_length=20, repeat=False):
    """
    Convenience wrapper voor gescoorde target_words (dict: word -> score).
    """
    return _generate_with_inject(
        model=model,
        tokenizer=tokenizer,
        input_text=input_text,
        inject_fn=_inject_scored(target_word_scores),
        max_length=max_length,
        repeat=repeat,
        temperature=temperature,
        top_k=top_k,
    )
