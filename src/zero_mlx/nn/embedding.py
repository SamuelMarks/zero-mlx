"""Embedding and QuantizedEmbedding lookup layers (`mlx.nn.embedding`) for zero_mlx."""

from __future__ import annotations

import math

from zero_mlx.array import array
from zero_mlx.mlx_random import uniform
from zero_mlx.nn.base import Module
import zero_mlx.ops as mx


class Embedding(Module):
    """A lookup table that stores embeddings of a fixed dictionary and dimension."""

    def __init__(self, num_embeddings: int, dims: int) -> None:
        """Initialize `Embedding`.

        Args:
            num_embeddings: Size of the dictionary of embeddings.
            dims: Size of each embedding vector.

        """
        super().__init__()
        self.num_embeddings: int = num_embeddings
        self.dims: int = dims
        scale: float = math.sqrt(1.0 / float(dims))
        self.weight: array = uniform(low=-scale, high=scale, shape=(num_embeddings, dims))

    def __call__(self, x: array) -> array:
        """Look up embeddings for integer indices `x`.

        Args:
            x: Index `array`.

        Returns:
            Embedded `array` of shape `(*x.shape, dims)`.

        """
        return self.weight[x]

    def as_linear(self, x: array) -> array:
        """Project `x` using the transposed embedding weight matrix `x @ weight.T`.

        Args:
            x: Input `array` of shape `(..., dims)`.

        Returns:
            Logit `array` of shape `(..., num_embeddings)`.

        """
        return mx.matmul(x, self.weight.T)


class QuantizedEmbedding(Module):
    """A lookup table that stores quantized embeddings of a fixed dictionary and dimension."""

    def __init__(
        self,
        num_embeddings: int,
        dims: int,
        group_size: int = 64,
        bits: int = 4,
        mode: str = "affine",
    ) -> None:
        """Initialize `QuantizedEmbedding`.

        Args:
            num_embeddings: Size of the dictionary of embeddings.
            dims: Size of each embedding vector.
            group_size: Quantization group size.
            bits: Number of bits per quantized weight.
            mode: Quantization mode (`'affine'`, `'mxfp4'`, `'nvfp4'`, `'mxfp8'`).

        """
        super().__init__()
        self.num_embeddings: int = num_embeddings
        self.dims: int = dims
        self.group_size: int = group_size
        self.bits: int = bits
        self.mode: str = mode
        scale: float = math.sqrt(1.0 / float(dims))
        self.weight: array = uniform(low=-scale, high=scale, shape=(num_embeddings, dims))
        self.scales: array = uniform(
            low=0.0, high=1.0, shape=(num_embeddings, max(1, dims // group_size))
        )
        self.biases: array = uniform(
            low=-scale, high=scale, shape=(num_embeddings, max(1, dims // group_size))
        )

    def __call__(self, x: array) -> array:
        """Look up and dequantize embeddings for integer indices `x`.

        Args:
            x: Index `array`.

        Returns:
            Dequantized embedded `array` of shape `(*x.shape, dims)`.

        """
        w = self.weight[x]
        s = self.scales[x]
        b = self.biases[x]
        s_rep = mx.repeat(s, self.group_size, axis=-1)[..., : w.shape[-1]]
        b_rep = mx.repeat(b, self.group_size, axis=-1)[..., : w.shape[-1]]
        return w * s_rep + b_rep

    def as_linear(self, x: array) -> array:
        """Project `x` using the dequantized transposed embedding matrix.

        Args:
            x: Input `array` of shape `(..., dims)`.

        Returns:
            Logit `array` of shape `(..., num_embeddings)`.

        """
        s_rep = mx.repeat(self.scales, self.group_size, axis=-1)[..., : self.weight.shape[-1]]
        b_rep = mx.repeat(self.biases, self.group_size, axis=-1)[..., : self.weight.shape[-1]]
        w_deq = self.weight * s_rep + b_rep
        return mx.matmul(x, w_deq.T)


__all__ = ["Embedding", "QuantizedEmbedding"]
