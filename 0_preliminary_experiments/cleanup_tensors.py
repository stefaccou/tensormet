# file to cleanup tensor files in the data directory
# we will change them to correct file type and device
from utils import DATA_DIR, tree_to_device, torch_or_pickle_load
import torch
import os
import pickle
from pathlib import Path

# we go through all elements in DATA_DIR/tensors/fineweb_dutch_vectors_ids/decomposition/

def _cleanup_tensors(path):
    """Cleans up tensor files in the specified directory."""
    path = DATA_DIR / path
    for tensor_file in path.glob("*.pkl"):
        if not "tensor" in tensor_file.name.split("/")[-1]:
            continue
        print("Processing file:", tensor_file)
        data = torch_or_pickle_load(tensor_file)
        # with open(tensor_file, "rb") as f:
        #     data = pickle.load(f)

        # move to cpu
        data_cpu = tree_to_device(data, torch.device("cpu"))
        # save as torch file
        torch_file = tensor_file.with_suffix(".pt")
        torch.save(data_cpu, torch_file)
        print("Saved cleaned file:", torch_file)
        # optionally remove the old file
        os.remove(tensor_file)
        print("Removed old file:", tensor_file)

def _cleanup_vocabs(path):
    """Cleans up tensor files in the specified directory."""
    path = DATA_DIR / path
    for vocab_file in path.glob("*.pkl"):
        if not "vocabularies" in vocab_file.name.split("/")[-1]:
            continue
        print("Processing file:", vocab_file)
        data = torch_or_pickle_load(vocab_file)
        # move to cpu
        data_cpu = tree_to_device(data, torch.device("cpu"))
        print("moved to cpu")
        # # save as pickle file
        with open(vocab_file.with_suffix(".pkl"), "wb") as f:
            pickle.dump(data_cpu, f)
        print("Saved cleaned file:", vocab_file)

# TENSORS_DIR = DATA_DIR / "tensors" / "fineweb_dutch_vectors_ids" / "decomposition"
# for tensor_file in TENSORS_DIR.glob("*.pkl"):
#     print("Processing file:", tensor_file)
#     with open(tensor_file, "rb") as f:
#         data = pickle.load(f)
#     # move to cpu
#     data_cpu = tree_to_device(data, torch.device("cpu"))
#     # save as torch file
#     torch_file = tensor_file.with_suffix(".pt")
#     torch.save(data_cpu, torch_file)
#     print("Saved cleaned file:", torch_file)
#     # optionally remove the old file
#     os.remove(tensor_file)
#     print("Removed old file:", tensor_file)

# we allow the user to specify a subdirectory in DATA_DIR/tensors/ to cleanup
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Cleanup tensor files in the data directory.")
    parser.add_argument(
        "--subdir",
        type=str,
        required=True,
        help="Subdirectory in DATA_DIR/tensors/ to cleanup (e.g., fineweb_dutch_vectors_ids/decomposition)."
    )
    parser.add_argument(
        "--type",
        type=str,
        choices=["tensors", "vocabs"],
        default="tensors",
        help="Type of files to cleanup: 'tensors' or 'vocabs'."
    )
    args = parser.parse_args()
    if args.type == "vocabs":
        _cleanup_vocabs(Path("tensors") / Path(args.subdir))
    else:
        _cleanup_tensors(Path("tensors") / Path(args.subdir))