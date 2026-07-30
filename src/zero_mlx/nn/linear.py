"""mlx.nn.linear module."""

from typing import Any, Optional
import math
from zero_mlx.nn.base import Module
from zero_mlx.array import array
from zero_mlx.mlx_random import uniform
import zero_mlx as mx


class Linear(Module):
    """Applies an affine transformation to the input."""

    def __init__(self, input_dims: int, output_dims: int, bias: bool = True):
        """Initialize Linear layer.

        Args:
            input_dims: Input dimension.
            output_dims: Output dimension.
            bias: Whether to use bias.

        """
        super().__init__()
        scale = math.sqrt(1.0 / input_dims)
        self.weight = uniform(low=-scale, high=scale, shape=(output_dims, input_dims))
        if bias:
            self.bias = uniform(low=-scale, high=scale, shape=(output_dims,))
        else:
            self.bias = None

    def __call__(self, x: array) -> array:
        """Call the layer.

        Args:
            x: Input array.

        Returns:
            Output array.

        """
        # x is (..., in_dims)
        # weight is (out_dims, in_dims)
        # out = x @ weight.T
        out = mx.matmul(x, self.weight.T)
        if getattr(self, "bias", None) is not None:
            out = out + self.bias
        return out


class Bilinear(Module):
    """Applies a bilinear transformation to the inputs."""

    def __init__(
        self, input1_dims: int, input2_dims: int, output_dims: int, bias: bool = True
    ):
        """Initialize Bilinear layer.

        Args:
            input1_dims: Input 1 dimension.
            input2_dims: Input 2 dimension.
            output_dims: Output dimension.
            bias: Whether to use bias.

        """
        super().__init__()
        scale = math.sqrt(1.0 / input1_dims)
        self.weight = uniform(
            low=-scale, high=scale, shape=(output_dims, input1_dims, input2_dims)
        )
        if bias:
            self.bias = uniform(low=-scale, high=scale, shape=(output_dims,))
        else:
            self.bias = None

    def __call__(self, x1: array, x2: array) -> array:
        """Call the layer.

        Args:
            x1: Input 1 array.
            x2: Input 2 array.

        Returns:
            Output array.

        """
        # weight is (out, in1, in2)
        # x1 is (..., in1), x2 is (..., in2)
        # result = sum(x1_i * x2_j * W_kij) -> einsum
        out = mx.einsum("...i, ...j, oij -> ...o", x1, x2, self.weight)
        if getattr(self, "bias", None) is not None:
            out = out + self.bias
        return out


class AllToShardedLinear(Module):
    """Each member of the group applies part of the affine transformation."""

    def __init__(
        self,
        input_dims: int,
        output_dims: int,
        bias: bool = True,
        group: Optional[Any] = None,
    ):
        """Initialize AllToShardedLinear layer.

        Args:
            input_dims: Input dimension.
            output_dims: Output dimension.
            bias: Whether to use bias.
            group: Distributed group.

        """
        super().__init__()
        scale = math.sqrt(1.0 / input_dims)
        self.weight = uniform(low=-scale, high=scale, shape=(output_dims, input_dims))
        if bias:
            self.bias = uniform(low=-scale, high=scale, shape=(output_dims,))
        else:
            self.bias = None

    def __call__(self, x: array) -> array:
        """Call the layer.

        Args:
            x: Input array.

        Returns:
            Output array.

        """
        out = mx.matmul(x, self.weight.T)
        if getattr(self, "bias", None) is not None:
            out = out + self.bias
        return out


class ShardedToAllLinear(Module):
    """Applies part of the affine transformation and aggregates results."""

    def __init__(
        self,
        input_dims: int,
        output_dims: int,
        bias: bool = True,
        group: Optional[Any] = None,
    ):
        """Initialize ShardedToAllLinear layer.

        Args:
            input_dims: Input dimension.
            output_dims: Output dimension.
            bias: Whether to use bias.
            group: Distributed group.

        """
        super().__init__()
        scale = math.sqrt(1.0 / input_dims)
        self.weight = uniform(low=-scale, high=scale, shape=(output_dims, input_dims))
        if bias:
            self.bias = uniform(low=-scale, high=scale, shape=(output_dims,))
        else:
            self.bias = None

    def __call__(self, x: array) -> array:
        """Call the layer.

        Args:
            x: Input array.

        Returns:
            Output array.

        """
        out = mx.matmul(x, self.weight.T)
        if getattr(self, "bias", None) is not None:
            out = out + self.bias
        return out


class QuantizedLinear(Module):
    """Applies an affine transformation to the input using a quantized weight matrix."""

    def __init__(
        self,
        input_dims: int,
        output_dims: int,
        bias: bool = True,
        group_size: int = 64,
        bits: int = 4,
        mode: str = "affine",
    ):
        """Initialize QuantizedLinear layer.

        Args:
            input_dims: Input dimension.
            output_dims: Output dimension.
            bias: Whether to use bias.
            group_size: Group size for quantization.
            bits: Number of bits.
            mode: Quantization mode.

        """
        super().__init__()
        scale = math.sqrt(1.0 / input_dims)
        self.weight = uniform(low=-scale, high=scale, shape=(output_dims, input_dims))
        self.scales = uniform(
            low=0.0, high=1.0, shape=(output_dims, max(1, input_dims // group_size))
        )
        if bias:
            self.bias = uniform(low=-scale, high=scale, shape=(output_dims,))
        else:
            self.bias = None
        self.group_size = group_size
        self.bits = bits
        self.mode = mode

    def __call__(self, x: array) -> array:
        """Call the layer.

        Args:
            x: Input array.

        Returns:
            Output array.

        """
        # Dequantize weights
        # weight: (out_dims, in_dims)
        # scales: (out_dims, in_dims // group_size)
        # We need scales to broadcast over the group_size
        s = mx.repeat(self.scales, self.group_size, axis=-1)[
            ..., : self.weight.shape[-1]
        ]
        w_deq = self.weight * s
        out = mx.matmul(x, w_deq.T)
        if getattr(self, "bias", None) is not None:
            out = out + self.bias
        return out


class QuantizedAllToShardedLinear(Module):
    """Quantized AllToShardedLinear layer."""

    def __init__(
        self,
        input_dims: int,
        output_dims: int,
        bias: bool = True,
        group_size: int = 64,
        bits: int = 4,
        group: Optional[Any] = None,
    ):
        """Initialize QuantizedAllToShardedLinear layer.

        Args:
            input_dims: Input dimension.
            output_dims: Output dimension.
            bias: Whether to use bias.
            group_size: Group size for quantization.
            bits: Number of bits.
            group: Distributed group.

        """
        super().__init__()
        scale = math.sqrt(1.0 / input_dims)
        self.weight = uniform(low=-scale, high=scale, shape=(output_dims, input_dims))
        self.scales = uniform(
            low=0.0, high=1.0, shape=(output_dims, max(1, input_dims // group_size))
        )
        if bias:
            self.bias = uniform(low=-scale, high=scale, shape=(output_dims,))
        else:
            self.bias = None
        self.group_size = group_size
        self.bits = bits

    def __call__(self, x: array) -> array:
        """Call the layer.

        Args:
            x: Input array.

        Returns:
            Output array.

        """
        s = mx.repeat(self.scales, self.group_size, axis=-1)[
            ..., : self.weight.shape[-1]
        ]
        w_deq = self.weight * s
        out = mx.matmul(x, w_deq.T)
        if getattr(self, "bias", None) is not None:
            out = out + self.bias
        return out


class QuantizedShardedToAllLinear(Module):
    """Quantized ShardedToAllLinear layer."""

    def __init__(
        self,
        input_dims: int,
        output_dims: int,
        bias: bool = True,
        group_size: int = 64,
        bits: int = 4,
        group: Optional[Any] = None,
    ):
        """Initialize QuantizedShardedToAllLinear layer.

        Args:
            input_dims: Input dimension.
            output_dims: Output dimension.
            bias: Whether to use bias.
            group_size: Group size for quantization.
            bits: Number of bits.
            group: Distributed group.

        """
        super().__init__()
        scale = math.sqrt(1.0 / input_dims)
        self.weight = uniform(low=-scale, high=scale, shape=(output_dims, input_dims))
        self.scales = uniform(
            low=0.0, high=1.0, shape=(output_dims, max(1, input_dims // group_size))
        )
        if bias:
            self.bias = uniform(low=-scale, high=scale, shape=(output_dims,))
        else:
            self.bias = None
        self.group_size = group_size
        self.bits = bits

    def __call__(self, x: array) -> array:
        """Call the layer.

        Args:
            x: Input array.

        Returns:
            Output array.

        """
        s = mx.repeat(self.scales, self.group_size, axis=-1)[
            ..., : self.weight.shape[-1]
        ]
        w_deq = self.weight * s
        out = mx.matmul(x, w_deq.T)
        if getattr(self, "bias", None) is not None:
            out = out + self.bias
        return out


__all__ = [
    "Linear",
    "Bilinear",
    "AllToShardedLinear",
    "ShardedToAllLinear",
    "QuantizedLinear",
    "QuantizedAllToShardedLinear",
    "QuantizedShardedToAllLinear",
]
