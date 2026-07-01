"""mlx.nn.embedding module."""

from zero_mlx.nn.base import Module
from zero_mlx.array import array
from zero_mlx.mlx_random import uniform
import ml_switcheroo_compiler.ops as sops
import math


class Embedding(Module):
    """A simple lookup table that stores embeddings of a fixed dictionary and size."""

    def __init__(self, num_embeddings: int, dims: int):
        """Initialize Embedding."""
        super().__init__()
        self.num_embeddings = num_embeddings
        self.dims = dims
        scale = math.sqrt(1.0 / dims)
        self.weight = uniform(low=-scale, high=scale, shape=(num_embeddings, dims))

    def __call__(self, x: array) -> array:
        """Call."""
        x_t = x._tensor if hasattr(x, "_tensor") else x
        w_t = self.weight._tensor if hasattr(self.weight, "_tensor") else self.weight
        if hasattr(sops, "embedding"):
            out = sops.embedding(x_t, w_t)
        elif hasattr(sops, "embedding_lookup"):
            out = sops.embedding_lookup(w_t, x_t)
        else:
            from ml_switcheroo_compiler.ops.nn.nlp import embedding

            out = embedding(x_t, w_t)
        return array(out)


class QuantizedEmbedding(Module):
    """A simple lookup table that stores quantized embeddings of a fixed dictionary and size."""

    def __init__(
        self,
        num_embeddings: int,
        dims: int,
        group_size: int = 64,
        bits: int = 4,
        mode: str = "affine",
    ):
        """Initialize QuantizedEmbedding."""
        super().__init__()
        self.num_embeddings = num_embeddings
        self.dims = dims
        self.group_size = group_size
        self.bits = bits
        self.mode = mode
        scale = math.sqrt(1.0 / dims)
        self.weight = uniform(low=-scale, high=scale, shape=(num_embeddings, dims))
        self.scales = uniform(
            low=0.0, high=1.0, shape=(num_embeddings, dims // group_size)
        )
        self.biases = uniform(
            low=-scale, high=scale, shape=(num_embeddings, dims // group_size)
        )

    def __call__(self, x: array) -> array:
        """Call."""
        x_t = x._tensor if hasattr(x, "_tensor") else x
        w_t = self.weight._tensor if hasattr(self.weight, "_tensor") else self.weight
        s_t = self.scales._tensor if hasattr(self.scales, "_tensor") else self.scales
        b_t = self.biases._tensor if hasattr(self.biases, "_tensor") else self.biases

        # Assume there's a quantized_embedding op in the backend, fallback if not
        if hasattr(sops, "quantized_embedding"):
            out = sops.quantized_embedding(
                x_t,
                w_t,
                scales=s_t,
                biases=b_t,
                group_size=self.group_size,
                bits=self.bits,
                mode=self.mode,
            )
        else:
            try:
                from ml_switcheroo_compiler.ops.nn.quantized_ops import (
                    quantized_embedding,
                )

                out = quantized_embedding(
                    x_t,
                    w_t,
                    scales=s_t,
                    biases=b_t,
                    group_size=self.group_size,
                    bits=self.bits,
                )
            except ImportError:
                # Fallback to standard embedding if quantized isn't available
                if hasattr(sops, "embedding_lookup"):
                    out = sops.embedding_lookup(w_t, x_t)
                else:
                    from ml_switcheroo_compiler.ops.nn.nlp import embedding

                    out = embedding(x_t, w_t)
        return array(out)


__all__ = ["Embedding", "QuantizedEmbedding"]
