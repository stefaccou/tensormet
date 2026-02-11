# Old version, checked 17/12/2025
import csv
import os
from datasets import load_dataset
import spacy
from tqdm import tqdm
from utils import DATA_DIR

# to not interfere with other processes on the server, we will connect to device 3
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
# the cpu should also be on this device


# ---- helpers to stream just the text field ----


def gen_texts(dataset, dataset_column_name):
    for ex in dataset:
        # each ex is a dict like {"text": "...", ...}
        txt = ex.get(dataset_column_name)
        if txt:
            yield txt.lower()


def extract_vectors(doc):

    # --- map labels to ints once ---
    S = doc.vocab.strings
    DEP_NSUBJ = S["nsubj"]
    DEP_OBJ = S["obj"]
    DEP_OBL = S["obl"]
    ROOT_VERB = S["VERB"]
    results = []
    vec_count = 0
    for sent in doc.sents:
        root = sent.root
        if root.pos != ROOT_VERB:  # or: if root.pos_ != "VERB"
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
                elif obj2 is None:  # no reliable way to discriminate without extra processing, we simplify to this
                    obj2 = child.lemma_
                    filled += 1
            elif d == DEP_OBL and obl is None:
                obl = child.lemma_
                filled += 1

            if filled == 4:  # all targets filled
                break

        results.append((root.lemma_, nsubj, obj, obj2, obl))
        vec_count += 1
    return results, vec_count


def pos_tag(target_vectors=10000,
            save_every=1000,      # change to 1_000 if you prefer more frequent checkpoints
            output_path=DATA_DIR/"fineweb_dutch_vectors.csv",
            dataset_path="epfml/FineWeb2-HQ",
            dataset_config="nld_Latn",
            dataset_column_name="text"):
    from datasets import load_dataset
    nlp = spacy.load("nl_core_news_md", disable=["ner", "textcat"])
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # --- dataset stream ---
    ds = load_dataset(dataset_path, dataset_config,
                      split="train", streaming=True)
    ds = ds.shuffle(seed=7, buffer_size=10_000)

    # init CSV once
    if not os.path.exists(output_path):
        with open(output_path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["root_lemma", "nsubj", "obj", "obj2", "obl"])

    texts = gen_texts(ds, dataset_column_name)
    buffer = []
    vector_count = 0
    last_checkpoint_at = 0


    vbar = tqdm(
        total=target_vectors,
        desc="Vectors",
        mininterval=0.5,
        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} • ETA: {remaining}"
    )

    try:
        for doc in nlp.pipe(texts, batch_size=1000, n_process=1):
            vectors, count = extract_vectors(doc)
            if count == 0:
                continue

            buffer.extend(vectors)
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

    pos_tag(target_vectors=1_000_000, save_every=20_000,
            # dataset_path="sentence-transformers/parallel-sentences-opensubtitles", dataset_config="en-nl",
            # dataset_column_name="non_english",
            output_path=DATA_DIR/"test.csv")
