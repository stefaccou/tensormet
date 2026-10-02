# TensorMet
Tensor Decomposition for Metaphor generation and interpretation.

## Replicating current code
1. Set the DATA_DIR variable to sample_dir as included in the repo
2. Load in one of the sample models:
```python
from tensormet.tucker_tensor import TuckerDecomposition
tk = TuckerDecomposition_load_from_disk()
```
