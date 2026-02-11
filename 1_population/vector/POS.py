# Version 1, checked 17/12/2025
import csv
import os
from datasets import load_dataset
import spacy
from tqdm import tqdm
import torch
from utils import DATA_DIR


# ---- helpers to stream just the text field ----
def gen_texts(dataset, dataset_column_name):
    for ex in dataset:
        # each ex is a dict like {"text": "...", ...}
        txt = ex.get(dataset_column_name)
        if txt:
            # let spaCy handle casing etc.; keep original text
            yield txt
def gen_texts_from_dir(directory):
    for filename in os.listdir(directory):
        if filename.endswith(".txt.sent"):
            filepath = os.path.join(directory, filename)
            with open(filepath, "r", encoding="utf-8") as f:
                # there are many short lines, we join into sentences/paragraphs ending with .
                text = f.read().replace("\n", " ")
                for sent in text.split("."):
                    sent = sent.strip()
                    if sent:
                        yield sent + "."

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
    DEP_OBJ = S["obj"]
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

        nsubj = obj = obj2 = obl = None
        filled = 0  # how many slots filled

        for child in root.children:
            d = child.dep
            if d == DEP_NSUBJ and nsubj is None:
                nsubj = child.lemma_
                filled += 1
            elif d == DEP_OBJ:
                if obj is None:
                    obj = child.lemma_
                    filled += 1
                elif obj2 is None:
                    obj2 = child.lemma_
                    filled += 1
            elif d == DEP_OBL and obl is None:
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
            output_path=DATA_DIR/"vectors/fineweb_dutch_vectors.csv",
            dataset_path="epfml/FineWeb2-HQ",
            dataset_config="nld_Latn",
            dataset_column_name="text",
            seed=7):
    from datasets import load_dataset
    print("starting loop")
    nlp = spacy.load("nl_core_news_md", disable=["ner", "textcat"])
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # --- dataset stream ---
    print("loading dataset...")
    if dataset_config == "local_dir":
        texts = gen_texts_from_dir(dataset_path)
    else:
        ds = load_dataset(dataset_path, dataset_config,
                          split="train", streaming=True)
        ds = ds.shuffle(seed=seed, buffer_size=10_000)

        texts = gen_texts(ds, dataset_column_name)

    print("dataset loaded, starting processing...")
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


    buffer = []
    vector_count = 0
    last_checkpoint_at = 0
    global_sent_id = 0          ### NEW: running sentence index

    vbar = tqdm(
        total=target_vectors,
        desc="Vectors",
        mininterval=0.5,
        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} • ETA: {remaining}"
    )

    try:
        # nlp.pipe yields Doc objects
        for doc in nlp.pipe(texts, batch_size=1000, n_process=3):
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
            vbar.update(count)
            vbar.set_postfix_str(f"total={vector_count}")

            if vector_count - last_checkpoint_at >= save_every:
                with open(output_path, "a", newline="", encoding="utf-8") as f:
                    w = csv.writer(f)
                    w.writerows(buffer)
                buffer.clear()
                last_checkpoint_at = vector_count

            if vector_count >= target_vectors:
                print("Target reached, stopping.")
                break
        print("Corpus exhausted, stopping.")
        print(f"Total vectors extracted: {vector_count}")
    except KeyboardInterrupt:
        pass
    finally:
        if buffer:
            with open(output_path, "a", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                w.writerows(buffer)

    print(f"Vectors written: {vector_count} (file: {output_path})")


if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "3"
    # we explicitly set cuda here
    torch.cuda.set_device(3)

    # pos_tag(target_vectors=2_000_000,
    #         save_every=250_000,
    #         dataset_path="epfml/FineWeb2-HQ",
    #         dataset_config="nld_Latn",
    #         dataset_column_name="text",
    #         output_path=DATA_DIR/"vectors/fineweb_dutch_vectors_ids.csv",
    #         seed=42)
    # pos_tag(target_vectors=1000000,
    #         dataset_path="sentence-transformers/parallel-sentences-opensubtitles", dataset_config="en-nl",
    #         dataset_column_name="non_english",
    #         output_path=DATA_DIR/"opensubtitles_nl_vectors_ids.csv"
    #         )
    pos_tag(target_vectors=1_000_000,
            save_every=10_000,
            dataset_path=DATA_DIR/"karrewiet",
            dataset_config="local_dir",
            output_path=DATA_DIR/"vectors/karrewiet_vectors_ids.csv")
