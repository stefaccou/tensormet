# Main version, checked 17/12/2025
import csv
import os
from datasets import load_dataset
import spacy
from tqdm import tqdm
import torch
from tensormet.utils import DATA_DIR, select_gpu, tee_output

# ---- helpers to stream just the text field ----

def gen_texts(dataset, dataset_column_name):
    for ex in dataset:
        # each ex is a dict like {"text": "...", ...}
        txt = ex.get(dataset_column_name)
        if txt:
            # let spaCy handle casing etc.; keep original text
            yield txt


def extract_vectors(doc, start_sent_id):
    """
    doc: spaCy Doc
    start_sent_id: int -> the id to assign to the FIRST sentence in this doc
    returns:
        rows: list of dicts, one per extracted vector
        count: how many vectors we added
        next_sent_id: the next free sentence id after this doc
    """

    S = doc.vocab.strings
    DEP_NSUBJ = S["nsubj"]
    DEP_OBJ = S["dobj"]
    DEP_IOBJ = S["dative"]
    DEP_OBL = S["obl"]
    ROOT_VERB = S["VERB"]

    rows = []
    vec_count = 0
    sent_id = start_sent_id

    for sent in doc.sents:
        root = sent.root
        # check that the root POS is a verb
        if root.pos != ROOT_VERB:  # equivalent to: if root.pos_ != "VERB"
            sent_id += 1
            continue

        nsubj = obj = obj2 = obl = "~"
        filled = 0  # how many slots filled

        for child in root.children:
            d = child.dep
            if d == DEP_NSUBJ and nsubj == "~":
                nsubj = child.lemma_
                filled += 1
            elif d == DEP_OBJ:
                if obj == "~":
                    obj = child.lemma_
                    filled += 1
                elif obj2 == "~":
                    obj2 = child.lemma_
                    filled += 1
            elif d == DEP_IOBJ:
                if obj2 == "~":
                    obj2 = child.lemma_
                    filled += 1
            elif d == DEP_OBL and obl == "~":
                obl = child.lemma_
                filled += 1

            if filled == 4:
                break

        root_lemma = root.lemma_

        # build the row for CSV
        vector_tuple = (root_lemma, nsubj, obj, obj2, obl)

        rows.append({
            "sent_id": sent_id,
            # "root_lemma": root_lemma,
            # "nsubj": nsubj,
            # "obj": obj,
            # "obj2": obj2,
            # "obl": obl,
            "vector": repr(vector_tuple),     ### NEW: store full tuple in one field
            "sentence_text": sent.text        ### NEW: store original sentence text
        })

        vec_count += 1
        sent_id += 1  # advance to next sentence id

    return rows, vec_count, sent_id


def pos_tag(target_vectors=10000,
            save_every=1000,
            output_path=DATA_DIR/"vectors/fineweb_english_vectors.csv",
            dataset_path="HuggingFaceFW/fineweb",
            dataset_config="CC-MAIN-2025-26",
            dataset_column_name="text",
            spacy_model="en_core_web_md"):
    from datasets import load_dataset
    nlp = spacy.load(spacy_model, disable=["ner", "textcat"])
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # --- dataset stream ---
    ds = load_dataset(dataset_path, dataset_config,
                      split="train", streaming=True)
    ds = ds.shuffle(seed=7, buffer_size=10_000)

    # init CSV once (now with extra columns)
    if not os.path.exists(output_path):
        with open(output_path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow([
                "sent_id",          ### NEW
                # "root_lemma",
                # "nsubj",
                # "obj",
                # "obj2",
                # "obl",
                "vector",           ### NEW
                "sentence_text"     ### NEW
            ])

    texts = gen_texts(ds, dataset_column_name)
    buffer = []
    vector_count = 0
    last_checkpoint_at = 0
    global_sent_id = 0          ### NEW: running sentence index

    try:
        # nlp.pipe yields Doc objects
        for doc in nlp.pipe(texts, batch_size=1000, n_process=40):
            rows, count, global_sent_id = extract_vectors(doc, global_sent_id)
            if count == 0:
                continue

            # append dict rows to buffer, but CSV writer wants flat lists
            for r in rows:

                buffer.append([
                    r["sent_id"],
                    # r["root_lemma"],
                    # r["nsubj"],
                    # r["obj"],
                    # r["obj2"],
                    # r["obl"],
                    r["vector"],
                    r["sentence_text"]
                ])

            vector_count += count
            if vector_count - last_checkpoint_at >= save_every:
                with open(output_path, "a", newline="", encoding="utf-8") as f:
                    w = csv.writer(f)
                    w.writerows(buffer)
                buffer.clear()
                last_checkpoint_at = vector_count

            if vector_count >= target_vectors:
                break

    except KeyboardInterrupt:
        pass
    finally:
        if buffer:
            with open(output_path, "a", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                w.writerows(buffer)

    print(f"Vectors written: {vector_count} (file: {output_path})")


if __name__ == "__main__":
    device = select_gpu()
    # pos_tag(target_vectors=1_000_000_000,
    #         output_path=DATA_DIR / "vectors/fineweb_english_1B.csv",
    #         dataset_path="epfml/FineWeb-HQ",
    #         dataset_config=None,
    #         dataset_column_name="text",
    #         spacy_model="en_core_web_md"
    #         )

    # Original:
    # pos_tag(target_vectors=10_000_000,
    #         output_path=DATA_DIR / "vectors/fineweb_english_vectors.csv",
    #         dataset_path="HuggingFaceFW/fineweb",
    #         dataset_config="CC-MAIN-2025-26",
    #         dataset_column_name="text",
    #         spacy_model="en_core_web_md"
    #         )

    with tee_output(DATA_DIR/"vectors"/"fineweb_vectors_10M_log.txt"):
        pos_tag(target_vectors=100_000_000,
                output_path=DATA_DIR / "vectors/fineweb_english_10M.csv",
                dataset_path="HuggingFaceFW/fineweb",
                dataset_config="CC-MAIN-2025-26",
                dataset_column_name="text",
                spacy_model="en_core_web_md"
                )

