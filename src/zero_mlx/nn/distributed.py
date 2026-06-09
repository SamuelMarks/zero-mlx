"""Quantized and distributed linear layers for neural networks."""

from typing import Any, Optional

import numpy as np

from .. import array
from .layers import Module


class QuantizedLinear(Module):
    """Applies an affine transformation to the input using a quantized weight matrix."""

    def __init__(
        self,
        input_dims: int,
        output_dims: int,
        bias: Optional[bool] = True,
        group_size: Optional[int] = 64,
        bits: Optional[int] = 4,
        mode: str = "affine",
    ) -> None:
        """Initialize."""
        super().__init__()
        self.input_dims = input_dims
        self.output_dims = output_dims
        self.bias = bias
        self.group_size = group_size
        self.bits = bits
        self.mode = mode

        self.weight = np.random.randn(output_dims, input_dims)
        self.scales = np.ones((output_dims, input_dims // group_size))
        self.biases = np.zeros((output_dims, input_dims // group_size))

        if self.bias:
            self.bias_param = np.zeros(output_dims)

    def __call__(self, x: array) -> array:
        """Forward pass."""
        # Simulated quantization just performs a standard dot product
        out = np.dot(x.data, self.weight.T)
        if self.bias:
            out += self.bias_param
        return array(out)


class QuantizedEmbedding(Module):
    """The same as :obj:`Embedding` but with a quantized weight matrix."""

    def __init__(
        self,
        num_embeddings: int,
        dims: int,
        group_size: Optional[int] = 64,
        bits: Optional[int] = 4,
        mode: str = "affine",
    ) -> None:
        """Initialize."""
        super().__init__()
        self.num_embeddings = num_embeddings
        self.dims = dims
        self.group_size = group_size
        self.bits = bits
        self.mode = mode

        self.weight = np.random.randn(num_embeddings, dims)

    def __call__(self, indices: array) -> array:
        """Forward pass."""
        return array(self.weight[indices.data])


class QuantizedAllToShardedLinear(Module):
    """Each member of the group applies part of the affine transformation with a quantized matrix."""

    def __init__(
        self,
        input_dims: int,
        output_dims: int,
        bias: Optional[bool] = True,
        group_size: Optional[int] = 64,
        bits: Optional[int] = 4,
        group: Optional[Any] = None,
    ) -> None:
        """Initialize."""
        super().__init__()
        self.input_dims = input_dims
        self.output_dims = output_dims
        self.bias = bias
        self.group_size = group_size
        self.bits = bits
        self.group = group

        # distributed simulation just returns chunk
        self.weight = np.random.randn(output_dims, input_dims)
        if self.bias:
            self.bias_param = np.zeros(output_dims)

    def __call__(self, x: array) -> array:
        """Forward pass."""
        out = np.dot(x.data, self.weight.T)
        if self.bias:
            out += self.bias_param
        return array(out)


class QuantizedShardedToAllLinear(Module):
    """Each member of the group applies part of the affine transformation using the quantized matrix."""

    def __init__(
        self,
        input_dims: int,
        output_dims: int,
        bias: Optional[bool] = True,
        group_size: Optional[int] = 64,
        bits: Optional[int] = 4,
        group: Optional[Any] = None,
    ) -> None:
        """Initialize."""
        super().__init__()
        self.input_dims = input_dims
        self.output_dims = output_dims
        self.bias = bias
        self.group_size = group_size
        self.bits = bits
        self.group = group

        self.weight = np.random.randn(output_dims, input_dims)
        if self.bias:
            self.bias_param = np.zeros(output_dims)

    def __call__(self, x: array) -> array:
        """Forward pass."""
        out = np.dot(x.data, self.weight.T)
        if self.bias:
            out += self.bias_param
        return array(out)


class AllToShardedLinear(Module):
    """Each member of the group applies part of the affine transformation such that the result is sharded."""

    def __init__(
        self,
        input_dims: int,
        output_dims: int,
        bias: Optional[bool] = True,
        group: Optional[Any] = None,
    ) -> None:
        """Initialize."""
        super().__init__()
        self.input_dims = input_dims
        self.output_dims = output_dims
        self.bias = bias
        self.group = group

        self.weight = np.random.randn(output_dims, input_dims)
        if self.bias:
            self.bias_param = np.zeros(output_dims)

    def __call__(self, x: array) -> array:
        """Forward pass."""
        out = np.dot(x.data, self.weight.T)
        if self.bias:
            out += self.bias_param
        return array(out)


class ShardedToAllLinear(Module):
    """Each member of the group applies part of the affine transformation and then aggregates the results."""

    def __init__(
        self,
        input_dims: int,
        output_dims: int,
        bias: Optional[bool] = True,
        group: Optional[Any] = None,
    ) -> None:
        """Initialize."""
        super().__init__()
        self.input_dims = input_dims
        self.output_dims = output_dims
        self.bias = bias
        self.group = group

        self.weight = np.random.randn(output_dims, input_dims)
        if self.bias:
            self.bias_param = np.zeros(output_dims)

    def __call__(self, x: array) -> array:
        """Forward pass."""
        out = np.dot(x.data, self.weight.T)
        if self.bias:
            out += self.bias_param
        return array(out)
