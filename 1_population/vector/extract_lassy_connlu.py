from pathlib import Path
from typing import List, Optional, Dict, Iterator, Tuple
from utils import DATA_DIR
import csv
from tqdm import tqdm


def iter_sentence_blocks(path: Path) -> Iterator[List[str]]:
    """Yield a list of lines (without trailing newline) per sentence."""
    buf: List[str] = []
    with open(path, "rt", encoding="utf-8", newline="") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line:
                if buf:
                    yield buf
                    buf = []
            else:
                buf.append(line)
        if buf:
            yield buf


def is_real_token_id(tok_id: str) -> bool:
    # ignore multiword tokens (e.g. "1-2") and empty nodes (e.g. "3.1")
    return tok_id.isdigit()


def parse_token_line(line: str) -> Optional[Dict]:
    if not line or line[0] == "#":
        return None
    cols = line.split("\t")
    if len(cols) != 10:
        return None  # malformed line
    tok_id = cols[0]
    if not is_real_token_id(tok_id):
        return None
    try:
        return {
            "id": int(cols[0]),
            "form": cols[1],
            "lemma": cols[2] if cols[2] != "_" else None,
            "upos": cols[3],
            "head": int(cols[6]) if cols[6].isdigit() else None,
            "deprel": cols[7],
        }
    except ValueError:
        return None


def rel_base(deprel: str) -> str:
    # "nsubj:pass" -> "nsubj", "obl:in" -> "obl"
    return deprel.split(":", 1)[0] if deprel else ""


def extract_vector_from_block(block_lines: List[str]) -> Tuple[
    Optional[str], Optional[str], Optional[Tuple[str, Optional[str], Optional[str], Optional[str], Optional[str]]]]:
    """
    Returns: (sent_id_str, sent_text, vector_tuple or None if skipped)
    """
    sent_id_str = None
    sent_text = None
    tokens: List[Dict] = []

    for line in block_lines:
        if line.startswith("# sent_id"):
            # "# sent_id = XYZ"
            parts = line.split("=", 1)
            if len(parts) == 2:
                sent_id_str = parts[1].strip()
        elif line.startswith("# text"):
            parts = line.split("=", 1)
            if len(parts) == 2:
                sent_text = parts[1].strip()
        else:
            t = parse_token_line(line)
            if t:
                tokens.append(t)

    if not tokens:
        return sent_id_str, sent_text, None

    # Find root: HEAD==0 and DEPREL=="root"
    root = None
    for t in tokens:
        if t["head"] == 0 and t["deprel"] == "root":
            root = t
            break
    if root is None:
        return sent_id_str, sent_text, None

    if root["upos"] != "VERB":
        return sent_id_str, sent_text, None

    root_id = root["id"]

    nsubj = "~"
    obj = "~"
    obj2 = "~"
    obl = "~"

    # children are tokens whose HEAD is root_id
    for child in tokens:
        if child["head"] != root_id:
            continue
        r = rel_base(child["deprel"])
        lem = child["lemma"]
        if r == "nsubj" and nsubj == "~":
            nsubj = lem
        elif r == "obj":
            if obj == "~":
                obj = lem
            elif obj2 == "~":
                obj2 = lem
        elif r == "obl" and obl == "~":
            obl = lem

        if nsubj is not "~" and obj is not "~" and obj2 is not "~" and obl is not "~":
            break

    vector = (root["lemma"], nsubj, obj, obj2, obl)
    return sent_id_str, sent_text, vector


def iter_conllu_files(input_dir: Path) -> Iterator[Path]:
    for p in input_dir.rglob("*"):
        if p.is_file() and (p.suffix in {".conllu", ".conll", ".txt", ".data", ".gz"}):
            # if it's .gz, accept it; otherwise accept common extensions
            # (your example uses .data)
            yield p

if __name__ == "__main__":
    connlu_path = Path("/home/nobackup/corpora/lassy-large-ud2")
    target_vectors = 10_000_000
    output_csv = Path(DATA_DIR / 'vectors' / f'lassy_first_{target_vectors}.csv')
    save_every = 200_000

    write_header = not output_csv.exists()
    out_f = open(output_csv, "a", newline="", encoding="utf-8")
    w = csv.writer(out_f)

    if write_header:
        header = ["sent_id", "vector", "sentence_text", "orig_sent_id"]
        w.writerow(header)

    buffer: List[List[str]] = []
    vec_count = 0
    sent_id_numeric = 0

    print(f"Starting extraction from {connlu_path} to {output_csv}")

    try:
        for fp in tqdm(iter_conllu_files(connlu_path)):
            for block in tqdm(iter_sentence_blocks(fp)):
                orig_sent_id, sent_text, vector = extract_vector_from_block(block)

                if vector is None:
                    continue

                buffer.append([
                    str(sent_id_numeric),
                    repr(vector),
                    sent_text or "",
                    orig_sent_id or ""
                ])
                vec_count += 1
                sent_id_numeric += 1

                if len(buffer) >= save_every:
                    w.writerows(buffer)
                    out_f.flush()
                    buffer.clear()

                if vec_count >= target_vectors:
                    break
            if vec_count >= target_vectors:
                break
    finally:
        # ensure last partial buffer is persisted even on Ctrl+C / exceptions
        if buffer:
            w.writerows(buffer)
            out_f.flush()
            buffer.clear()
        out_f.close()

    print(f"Wrote {vec_count} vectors to {output_csv}")
