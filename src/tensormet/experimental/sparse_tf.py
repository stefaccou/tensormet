from __future__ import annotations
from tensormet.tucker_tensor import _to_np

class TuckerVizMixin:
    """Visualization and rarely-used inspection methods for TuckerDecomposition."""
    # Paste candidate methods from TuckerDecomposition here.


    def sparse_representation(self):
        import tensorflow as tf
        # we return the sparse representation of the tensor
        # we check if our tensor is a tensorflow tensor or make it one
        if not isinstance(self.core, tf.Tensor):
            core = tf.convert_to_tensor(self.core)
        else:
            core = self.core
        sparse_core = tf.sparse.from_dense(core)
        # we do the same for the factors
        sparse_factors = []
        for factor in self.factors:
            if not isinstance(factor, tf.Tensor):
                factor = tf.convert_to_tensor(factor)
            sparse_factor = tf.sparse.from_dense(factor)
            sparse_factors.append(sparse_factor)

        return sparse_core, sparse_factors


    def tensor_to_sparse(self):
        self.core, self.factors = self.sparse_representation()


    def tensor_to_dense(self):
        import tensorflow as tf
        # If we have TensorFlow sparse tensors
        if isinstance(self.core, tf.SparseTensor):
            self.core = tf.sparse.to_dense(self.core).numpy()
            self.factors = [
                tf.sparse.to_dense(f).numpy() if isinstance(f, tf.SparseTensor) else _to_np(f)
                for f in self.factors
            ]
        else:
            # If they’re already torch/np dense, just ensure NumPy
            self.core = _to_np(self.core)
            self.factors = [_to_np(f) for f in self.factors]