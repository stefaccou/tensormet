from similarity import load_og_sentences
from utils import DATA_DIR
import os
from tucker_tensor import TuckerDecomposition, ExtendedTucker

dataset = "fineweb-en"
# method = "siiShifted"
methods = ["siiSoftPlus", "scSoftPlus"]

# divergence = "fr"
divergences = ["fr"]
# dims = 2000
dims = [1000,2000]

rank = 150
iterations = 500



min_count=50

vector_path = os.path.join(DATA_DIR, "vectors", "fineweb_english_vectors.csv")
full_dataset = load_og_sentences(vector_path=vector_path)

for method in methods:
    for divergence in divergences:
        for dim in dims:
            tucker = TuckerDecomposition.load_from_disk(
                dataset=dataset,
                method=method,
                divergence=divergence,
                dims=dim,
                rank=rank,
                iterations=iterations
            )
            extended_norm = ExtendedTucker.extend_tucker(
                tucker,
                dataset=full_dataset,
                roles=["verb", "subject", "object"],
                min_count=50,
                fraction_threads=0.66,
                normalize=True,
                normalize_mode="l2"
            )
            path = (DATA_DIR / "tensors" / dataset / "extension" /
                    f"{divergence}_{method}_{dim}d_{rank}r_{iterations}i_norm.pt")
            extended_norm.save_extensions(path)
            print("saved to", path)

            extended_minmax = ExtendedTucker.extend_tucker(
                tucker,
                dataset=full_dataset,
                roles=["verb", "subject", "object"],
                min_count=50,
                fraction_threads=0.66,
                normalize=True,
                normalize_mode="minmax"
            )
            path = (DATA_DIR / "tensors" / dataset / "extension" /
                    f"{divergence}_{method}_{dim}d_{rank}r_{iterations}i_minmax.pt")
            extended_minmax.save_extensions(path)
            print("saved to", path)

            extended = ExtendedTucker.extend_tucker(
                tucker,
                dataset=full_dataset,
                roles=["verb", "subject", "object"],
                min_count=50,
                fraction_threads=0.66,
                normalize=False
            )

            path = DATA_DIR/"tensors"/dataset/"extension"/f"{divergence}_{method}_{dim}d_{rank}r_{iterations}i.pt"
            extended.save_extensions(path)
            print("saved to", path)
            print()