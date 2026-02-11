from math import log
from collections import Counter
import csv, os
import numpy as np
import time
import torch
from tqdm import tqdm
from utils import notify_discord, DATA_DIR
import pickle

def populate_tensors(path_to_vectors, top_k=750, max_size_in_bytes=None, path_to_tensors=None, save=True):
    print(f"Populating tensors for top_k={top_k} from {path_to_vectors}...")
    if not path_to_tensors:
        # dataset = path_to_vectors.split("/")[-1].split(".")[0]
        # path_to_tensors = DATA_DIR/f"tensors/{dataset}/"
        dataset = path_to_vectors.stem  # instead of split()
        path_to_tensors = DATA_DIR / f"tensors/{dataset}/"

    os.makedirs(path_to_tensors, exist_ok=True)
    os.makedirs(f"{path_to_tensors}/populated", exist_ok=True)
    os.makedirs(f"{path_to_tensors}/vocabularies", exist_ok=True)
    print(f"Tensors will be saved to {path_to_tensors}")

    to_compute = {"counting", "sii", "sc", "vocabularies"}
    # check for existing files
    if (path_to_tensors / f"populated/counting_{top_k}.pt").exists():
        to_compute.discard("counting")

    if (path_to_tensors / f"populated/sii_{top_k}.pt").exists():
        to_compute.discard("sii")

    if (path_to_tensors / f"populated/sc_{top_k}.pt").exists():
        to_compute.discard("sc")

    # vocabularies are stored as .pkl, not .pt
    if (path_to_tensors / f"vocabularies/{top_k}.pkl").exists():
        to_compute.discard("vocabularies")

    # no work needed?
    if not to_compute:
        print("All tensors already exist, skipping computation.")
        return

    p_x = Counter() # probabilities
    p_y = Counter()
    p_z = Counter()
    p_xy = Counter()
    p_yz = Counter()
    p_xz = Counter()
    p_xyz = Counter()


    def specific_interaction_information(x, y, z):
        if min(p_x[x], p_y[y], p_z[z], p_xy[(x,y)], p_yz[(y,z)], p_xz[(x,z)], p_xyz[(x,y,z)]) == 0:
            raise ValueError('One of the probabilities is zero, cannot compute SII')
        sii = log(
            (p_xy[(x,y)] * p_yz[(y,z)] * p_xz[(x,z)])
            /
            (p_x[x] * p_y[y] * p_z[z] * p_xyz[(x,y,z)])
        )
        return sii

    def specific_correlation(x, y, z):
        if min(p_x[x], p_y[y], p_z[z], p_xyz[(x,y,z)]) == 0:
            raise ValueError('One of the probabilities is zero, cannot compute SC')
        sc = log(
            p_xyz[(x,y,z)]
            /
            (p_x[x] * p_y[y] * p_z[z])
        )
        return sc




    # with open(path_to_vectors, "r", encoding="utf-8") as f:
    #     reader = csv.reader(f)
    #     entries = list(reader)
    #     vectors = []
    #     for row in entries:
    #         # entries consist of "id", "(v,s,o)", "sentence"
    #         vector = row[1]
    #         # vector is a string representation of a tuple, we convert it back to tuple
    #         vector = tuple(vector.strip("()").replace("'", "").split(", "))
    #         vectors.append(vector)
    vectors = []
    with open(path_to_vectors, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        next(reader, None)  # skip header if there is one
        for row in reader:
            vector = row[1]
            # vector is a string representation of a tuple, we convert it back to tuple
            vector = tuple(vector.strip("()").replace("'", "").split(", "))
            (v,s,o) = vector[:3]
            v = v or "~"
            s = s or "~"
            o = o or "~"
            vectors.append((v, s, o))

            p_x[v] += 1
            p_y[s] += 1
            p_z[o] += 1
            p_xy[(v, s)] += 1
            p_yz[(s, o)] += 1
            p_xz[(v, o)] += 1
            p_xyz[(v, s, o)] += 1

    total_len = len(vectors)
    # convert all counts to probabilities in place
    for counter in [p_x, p_y, p_z, p_xy, p_yz, p_xz, p_xyz]:
        for key in counter:
            counter[key] /= total_len
    print("Probabilities computed.")
    #print(f"Loaded {len(vectors)-1} vectors from {path_to_vectors}")

    # first we build the vocabularies and the clean vectors

    # for vector in vectors:
    #     v = vector[0] if vector[0] != "" else "~"
    #     s = vector[1] if vector[1] != "" else "~"
    #     o = vector[2] if vector[2] != "" else "~"
    #     vectors.append((v, s, o))
    # print(f"Cleaned vectors ({len(vectors)} total)")
    # # we print the first 5 clean vectors for sanity check
    # print("First 5 clean vectors:", vectors[:5])
    #
    #
    # # we build up the counts
    # total_len = len(vectors)
    # for v, s, o in vectors:
    #     p_x[v] += 1 / total_len
    #     p_y[s] += 1 / total_len
    #     p_z[o] += 1 / total_len
    #     p_xy[(v, s)] += 1 / total_len
    #     p_yz[(s, o)] += 1 / total_len
    #     p_xz[(v, o)] += 1 / total_len
    #     p_xyz[(v, s, o)] += 1 / total_len
    # print("Counts built")

    #
    # m_c_v = [v for (v, count) in p_x.most_common(top_k)]
    # m_c_s = [s for (s, count) in p_y.most_common(top_k)]
    # m_c_o = [o for (o, count) in p_z.most_common(top_k)]
    # subset_t = []
    # for (v, s, o, *rest) in vectors:
    #     if v in m_c_v and s in m_c_s and o in m_c_o:
    #         subset_t.append((v,s,o))
    #
    # # 1) Build vocabularies in frequency order (stable & useful later)
    # vocab_v = [v for v, _ in Counter([v for v, _, _ in subset_t]).most_common()]
    # vocab_s = [s for s, _ in Counter([s for _, s, _ in subset_t]).most_common()]
    # vocab_o = [o for o, _ in Counter([o for _, _, o in subset_t]).most_common()]
    #
    # V, S, O = len(vocab_v), len(vocab_s), len(vocab_o)
    #
    # v2i = {v: i for i, v in enumerate(vocab_v)}
    # s2i = {s: i for i, s in enumerate(vocab_s)}
    # o2i = {o: i for i, o in enumerate(vocab_o)}
    m_c_v = [v for (v, count) in p_x.most_common(top_k)]
    m_c_s = [s for (s, count) in p_y.most_common(top_k)]
    m_c_o = [o for (o, count) in p_z.most_common(top_k)]

    # use top-k directly as vocabs
    vocab_v = m_c_v
    vocab_s = m_c_s
    vocab_o = m_c_o

    V, S, O = len(vocab_v), len(vocab_s), len(vocab_o)

    v2i = {v: i for i, v in enumerate(vocab_v)}
    s2i = {s: i for i, s in enumerate(vocab_s)}
    o2i = {o: i for i, o in enumerate(vocab_o)}

    # for filtering triples, use sets for fast membership
    v_set = set(vocab_v)
    s_set = set(vocab_s)
    o_set = set(vocab_o)

    subset_t = []
    for v, s, o in vectors:
        if v in v_set and s in s_set and o in o_set:
            subset_t.append((v, s, o))

    total_size = V*S*O
    print(f"Tensor dims: V={V}, S={S}, O={O}")
    print(f"Total size: {total_size} entries (~{(V * S * O * 4) / (1024 ** 3):.2f} GB as float32)")
    if max_size_in_bytes:
        if total_size > max_size_in_bytes:
            # we input the user for explicit confirmation before proceeding
            print(f"Warning: The tensor size exceeds the maximum allowed size of {max_size_in_bytes} bytes")
            input("Press Enter to continue or Ctrl+C to abort...")

    count_tensor = np.zeros((V, S, O), dtype=np.float32)
    sii_tensor = np.zeros((V, S, O), dtype=np.float32)
    sc_tensor = np.zeros((V, S, O), dtype=np.float32)
    print("Populating tensors...")

    # for v, s, o in tqdm(subset_t):
    #     vi = v2i[v]
    #     si = s2i[s]
    #     oi = o2i[o]
    #     tensor[vi, si, oi] += 1  # regular counting
    #     sii_tensor[vi, si, oi] = specific_interaction_information(v, s,o)
    #     sc_tensor[vi, si, oi] = specific_correlation(v, s, o)
    triple_counts = Counter(subset_t)
    for (v, s, o), count in tqdm(triple_counts.items()):
        vi = v2i[v]
        si = s2i[s]
        oi = o2i[o]
        count_tensor[vi, si, oi] += count  # regular counting
        sii_tensor[vi, si, oi] = specific_interaction_information(v, s,o)
        sc_tensor[vi, si, oi] = specific_correlation(v, s, o)


    # we explicitly make them torch tensors and save to cpu
    count_tensor = torch.tensor(count_tensor, dtype=torch.float32, device="cpu")
    sii_tensor = torch.tensor(sii_tensor, dtype=torch.float32, device="cpu")
    sc_tensor = torch.tensor(sc_tensor, dtype=torch.float32, device="cpu")
    print(f"created tensors of shape {sc_tensor.shape}")


    # we save them to the correct output paths
    if save:
        print(f"Saving tensors to {path_to_tensors}...")
        torch.save(count_tensor, f"{path_to_tensors}/populated/counting_{top_k}.pt")
        torch.save(sii_tensor, f"{path_to_tensors}/populated/sii_{top_k}.pt")
        torch.save(sc_tensor, f"{path_to_tensors}/populated/sc_{top_k}.pt")
        with open(f"{path_to_tensors}/vocabularies/{top_k}.pkl", "wb") as f:
            vocab = {
                "vocab_v": vocab_v,
                "vocab_s": vocab_s,
                "vocab_o": vocab_o,
                "v2i": v2i,
                "s2i": s2i,
                "o2i": o2i
            }
            pickle.dump(vocab, f)
        print("Tensors and vocabularies saved.")
        print()

    return count_tensor, sii_tensor, sc_tensor, vocab

if __name__ == "__main__":
    # path_to_vectors = DATA_DIR/"vectors/fineweb_dutch_vectors_ids.csv"
    goal_ks = [1500]
    paths_to_vectors = [DATA_DIR/"vectors/fineweb_dutch_vectors_ids.csv", DATA_DIR/"vectors/karrewiet_vectors_ids.csv"]
    for path_to_vectors in paths_to_vectors:
        start_time = time.time()
        for goal_k in goal_ks:
            populate_tensors(path_to_vectors, top_k=goal_k)
        end_time = time.time()
        notify_discord(f"Finished populating {path_to_vectors}in {(end_time - start_time)} seconds.")
