"""mlx.nn.embedding module."""

from zero_mlx.nn.base import Module
from zero_mlx.array import array
from zero_mlx.mlx_random import uniform
import math
import zero_mlx as mx


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
        return self.weight[x]


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
            low=0.0, high=1.0, shape=(num_embeddings, max(1, dims // group_size))
        )
        self.biases = uniform(
            low=-scale, high=scale, shape=(num_embeddings, max(1, dims // group_size))
        )

    def __call__(self, x: array) -> array:
        """Call."""
        w = self.weight[x]
        s = self.scales[x]
        b = self.biases[x]
        s_rep = mx.repeat(s, self.group_size, axis=-1)[..., : w.shape[-1]]
        b_rep = mx.repeat(b, self.group_size, axis=-1)[..., : w.shape[-1]]
        return w * s_rep + b_rep


__all__ = ["Embedding", "QuantizedEmbedding"]
