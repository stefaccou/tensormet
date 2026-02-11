from math import log
from collections import Counter
import csv, os
import numpy as np
import time
import torch
from tqdm import tqdm
from utils import notify_discord, DATA_DIR, select_gpu
import pickle
import ast

def populate_tensors(path_to_vectors, top_ks, save=True, path_to_tensors=None):
    print(f"Populating tensors for top_k={top_ks} from {path_to_vectors}...")
    if not path_to_tensors:
        # dataset = path_to_vectors.split("/")[-1].split(".")[0]
        # path_to_tensors = DATA_DIR/f"tensors/{dataset}/"
        dataset = path_to_vectors.stem.split('_')[0]+"_sparse"  # instead of split()
        print(f"Dataset name inferred as: {dataset}")
        path_to_tensors = DATA_DIR / f"tensors/{dataset}/"

    os.makedirs(path_to_tensors, exist_ok=True)
    os.makedirs(f"{path_to_tensors}/populated", exist_ok=True)
    os.makedirs(f"{path_to_tensors}/vocabularies", exist_ok=True)
    print(f"Tensors will be saved to {path_to_tensors}")


    p_x = Counter()  # probabilities
    p_y = Counter()
    p_z = Counter()
    p_xy = Counter()
    p_yz = Counter()
    p_xz = Counter()
    p_xyz = Counter()

    def specific_interaction_information(x, y, z):
        if min(p_x[x], p_y[y], p_z[z], p_xy[(x, y)], p_yz[(y, z)], p_xz[(x, z)], p_xyz[(x, y, z)]) == 0:
            raise ValueError('One of the probabilities is zero, cannot compute SII')
        sii = log(
            (p_xy[(x, y)] * p_yz[(y, z)] * p_xz[(x, z)])
            /
            (p_x[x] * p_y[y] * p_z[z] * p_xyz[(x, y, z)])
        )
        return sii

    def specific_correlation(x, y, z):
        if min(p_x[x], p_y[y], p_z[z], p_xyz[(x, y, z)]) == 0:
            raise ValueError('One of the probabilities is zero, cannot compute SC')
        sc = log(
            p_xyz[(x, y, z)]
            /
            (p_x[x] * p_y[y] * p_z[z])
        )
        return sc


    vectors = []
    with open(path_to_vectors, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        next(reader, None)  # skip header if there is one
        for row in reader:
            vector = ast.literal_eval(row[1])
            # vector = row[1]
            # # vector is a string representation of a tuple, we convert it back to tuple
            # vector = tuple(vector.strip("()").replace("'", "").split(", "))
            (v, s, o) = vector[:3]
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


    # if top_k is a list, we do this for each. If not, we transform it to a list of one element
    if not isinstance(top_ks, list):
        top_ks = [top_ks]

    results = {}
    for top_k in top_ks:
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
            continue

        m_c_v = [v for (v, count) in p_x.most_common(top_k)]
        m_c_s = [s for (s, count) in p_y.most_common(top_k)]
        m_c_o = [o for (o, count) in p_z.most_common(top_k)]
        # we check if all have at least top_k counts
        if not len(m_c_v) == len(m_c_s) == len(m_c_o) == top_k:
            print(f"Warning: less than {top_k} unique elements in one of the dimensions.")

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

        total_size = V * S * O
        print(f"Tensor dims: V={V}, S={S}, O={O}")
        print(f"Dense size: {total_size} entries (~{(V * S * O * 4) / (1024 ** 3):.2f} GB as float32)")

        # --- NEW: build sparse representation directly ---
        triple_counts = Counter(subset_t)

        indices = []
        count_values = []
        sii_values = []
        sc_values = []

        for (v, s, o), count in tqdm(triple_counts.items()):
            vi = v2i[v]
            si = s2i[s]
            oi = o2i[o]

            # store indices (vi, si, oi)
            indices.append([vi, si, oi])
            count_values.append(float(count))

            # these may raise ValueError if some probability is zero (same behavior as before)
            sii_values.append(float(specific_interaction_information(v, s, o)))
            sc_values.append(float(specific_correlation(v, s, o)))

        if len(indices) == 0:
            # no non-zero entries; create empty sparse tensors
            indices_tensor = torch.empty((3, 0), dtype=torch.long)
            count_tensor = torch.sparse_coo_tensor(indices_tensor,
                                                   torch.empty((0,), dtype=torch.float32),
                                                   size=(V, S, O))
            sii_tensor = torch.sparse_coo_tensor(indices_tensor,
                                                 torch.empty((0,), dtype=torch.float32),
                                                 size=(V, S, O))
            sc_tensor = torch.sparse_coo_tensor(indices_tensor,
                                                torch.empty((0,), dtype=torch.float32),
                                                size=(V, S, O))
        else:
            indices_tensor = torch.tensor(indices, dtype=torch.long).t()  # shape (3, nnz)
            count_values_tensor = torch.tensor(count_values, dtype=torch.float32)
            sii_values_tensor = torch.tensor(sii_values, dtype=torch.float32)
            sc_values_tensor = torch.tensor(sc_values, dtype=torch.float32)

            size = (V, S, O)

            count_tensor = torch.sparse_coo_tensor(indices_tensor, count_values_tensor, size=size)
            sii_tensor = torch.sparse_coo_tensor(indices_tensor, sii_values_tensor, size=size)
            sc_tensor = torch.sparse_coo_tensor(indices_tensor, sc_values_tensor, size=size)

            # ensure canonical form (sorted indices, summed duplicates)
            count_tensor = count_tensor.coalesce()
            sii_tensor = sii_tensor.coalesce()
            sc_tensor = sc_tensor.coalesce()


        print(f"created sparse tensors of shape {count_tensor.shape} with {count_tensor._nnz()} non-zero entries")
        print(f"Approximate size in memory: {count_tensor._nnz() * 3 * 4 / (1024 ** 2):.2f} MB (indices + values) each")
        vocab = {
            "vocab_v": vocab_v,
            "vocab_s": vocab_s,
            "vocab_o": vocab_o,
            "v2i": v2i,
            "s2i": s2i,
            "o2i": o2i
        }
        if save:
            print(f"Saving tensors to {path_to_tensors}...")
            torch.save(count_tensor, f"{path_to_tensors}/populated/counting_{top_k}.pt")
            torch.save(sii_tensor, f"{path_to_tensors}/populated/sii_{top_k}.pt")
            torch.save(sc_tensor, f"{path_to_tensors}/populated/sc_{top_k}.pt")
            with open(f"{path_to_tensors}/vocabularies/{top_k}.pkl", "wb") as f:
                pickle.dump(vocab, f)
            print("Tensors and vocabularies saved.")
            print()
        else:
            results[top_k] = (count_tensor, sii_tensor, sc_tensor, vocab)

        notify_discord(f"Finished populating tensors for top_k={top_k} from {path_to_vectors}.")

    return results


if __name__ == "__main__":
    # path_to_vectors = DATA_DIR/"vectors/fineweb_dutch_vectors_ids.csv"
    goal_ks = [1000, 2000, 3000, 4000, 5000]
    paths_to_vectors = [DATA_DIR/"vectors/fineweb_french_10000000.csv"]
    path_to_tensors = DATA_DIR / "tensors" / "fineweb-fr"
    device = select_gpu()
    for path_to_vectors in paths_to_vectors:
        start_time = time.time()
        # for goal_k in goal_ks:
        #     populate_tensors(path_to_vectors, top_ks=goal_k)
        populate_tensors(path_to_vectors,
                         top_ks=goal_ks,
                         path_to_tensors=path_to_tensors)
        end_time = time.time()
        notify_discord(f"Finished populating {path_to_vectors} in {(end_time - start_time)} seconds.")


