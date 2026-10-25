"""Embedding and QuantizedEmbedding lookup layers (`mlx.nn.embedding`) for zero_mlx."""

from __future__ import annotations

import math

from zero_mlx.array import array
from zero_mlx.convolutions import dequantize, quantize, quantized_matmul
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
        w_init = uniform(low=-scale, high=scale, shape=(num_embeddings, dims))
        q_res = quantize(w_init, group_size=group_size, bits=bits, mode=mode)
        self.weight: array = q_res[0]
        self.scales: array = q_res[1]
        self.biases: array | None = q_res[2] if len(q_res) == 3 else None

    @classmethod
    def from_embedding(
        cls,
        embedding_layer: Embedding,
        group_size: int = 64,
        bits: int = 4,
        mode: str = "affine",
    ) -> QuantizedEmbedding:
        """Create a `QuantizedEmbedding` by quantizing an existing `Embedding` layer's weights.

        Args:
            embedding_layer: Source `Embedding` module.
            group_size: Quantization group size.
            bits: Bits per quantized weight.
            mode: Quantization mode.

        Returns:
            Quantized `QuantizedEmbedding` instance preserving `embedding_layer`'s weights.

        """
        qe = cls(
            embedding_layer.num_embeddings,
            embedding_layer.dims,
            group_size=group_size,
            bits=bits,
            mode=mode,
        )
        q_res = quantize(embedding_layer.weight, group_size=group_size, bits=bits, mode=mode)
        qe.weight = q_res[0]
        qe.scales = q_res[1]
        qe.biases = q_res[2] if len(q_res) == 3 else None
        return qe

    def __call__(self, x: array) -> array:
        """Look up and dequantize embeddings for integer indices `x`.

        Args:
            x: Index `array`.

        Returns:
            Dequantized embedded `array` of shape `(*x.shape, dims)`.

        """
        w = self.weight[x]
        s = self.scales[x]
        b = self.biases[x] if self.biases is not None else None
        deq = dequantize(
            w,
            s,
            biases=b,
            group_size=self.group_size,
            bits=self.bits,
            mode=self.mode,
        )
        return deq[..., : self.dims]

    def as_linear(self, x: array) -> array:
        """Project `x` using the dequantized transposed embedding matrix.

        Args:
            x: Input `array` of shape `(..., dims)`.

        Returns:
            Logit `array` of shape `(..., num_embeddings)`.

        """
        return quantized_matmul(
            x,
            self.weight,
            self.scales,
            biases=self.biases,
            transpose=True,
            group_size=self.group_size,
            bits=self.bits,
            mode=self.mode,
        )


__all__ = ["Embedding", "QuantizedEmbedding"]
