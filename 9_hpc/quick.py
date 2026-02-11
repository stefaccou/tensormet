
import sys
import importlib
import torch
import tensorly
from tensormet.utils import DATA_DIR

# Alias the top-level module
sys.modules["tensorly_custom"] = tensorly

# If your pickle references submodules, alias those too (common ones)
for name in [
    "decomposition",
    "tenalg",
    "backend",
    "base",
    "tucker_tensor",
    "cp_tensor",
]:
    try:
        sys.modules[f"tensorly_custom.{name}"] = importlib.import_module(f"tensorly.{name}")
    except Exception:
        pass  # ignore missing submodules

obj = torch.load(DATA_DIR/"tensors/tier1/decomposition/fr_siiSoftPlus_1000d_150r_500i.pt", map_location="cpu", weights_only=False)

torch.save(obj, DATA_DIR/"tensors/tier1/decomposition/fr_siiSoftPlus_1000d_150r_500i.pt")