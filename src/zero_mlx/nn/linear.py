"""Linear, bilinear, sharded, and quantized linear layers (`mlx.nn.linear`) for zero_mlx."""

from __future__ import annotations

import math

from zero_mlx.array import array
from zero_mlx.convolutions import qqmm
from zero_mlx.distributed import Group
from zero_mlx.mlx_random import uniform
from zero_mlx.nn.base import Module
import zero_mlx.ops as mx


class Linear(Module):
    """Applies an affine transformation `y = x @ W^T + b` to the input."""

    def __init__(self, input_dims: int, output_dims: int, bias: bool = True) -> None:
        """Initialize the `Linear` layer.

        Args:
            input_dims: Dimensionality of each input sample.
            output_dims: Dimensionality of each output sample.
            bias: If True, includes an additive learnable bias.

        """
        super().__init__()
        scale: float = math.sqrt(1.0 / float(input_dims))
        self.weight: array = uniform(low=-scale, high=scale, shape=(output_dims, input_dims))
        self.bias: array | None = (
            uniform(low=-scale, high=scale, shape=(output_dims,)) if bias else None
        )

    def __call__(self, x: array) -> array:
        """Compute `x @ weight.T + bias`.

        Args:
            x: Input `array` of shape `(..., input_dims)`.

        Returns:
            Output `array` of shape `(..., output_dims)`.

        """
        out = mx.matmul(x, self.weight.T)
        if self.bias is not None:
            out = out + self.bias
        return out


class Bilinear(Module):
    """Applies a bilinear transformation `y_o = x1^T W_o x2 + b_o` to two inputs."""

    def __init__(
        self,
        input1_dims: int,
        input2_dims: int,
        output_dims: int,
        bias: bool = True,
    ) -> None:
        """Initialize the `Bilinear` layer.

        Args:
            input1_dims: Dimensionality of the first input `x1`.
            input2_dims: Dimensionality of the second input `x2`.
            output_dims: Dimensionality of the output.
            bias: Whether to include a learnable additive bias.

        """
        super().__init__()
        scale: float = math.sqrt(1.0 / float(input1_dims))
        self.weight: array = uniform(
            low=-scale, high=scale, shape=(output_dims, input1_dims, input2_dims)
        )
        self.bias: array | None = (
            uniform(low=-scale, high=scale, shape=(output_dims,)) if bias else None
        )

    def __call__(self, x1: array, x2: array) -> array:
        """Apply the bilinear layer to `(x1, x2)`.

        Args:
            x1: First input `array` of shape `(..., input1_dims)`.
            x2: Second input `array` of shape `(..., input2_dims)`.

        Returns:
            Output `array` of shape `(..., output_dims)`.

        """
        out = mx.einsum("...i, ...j, oij -> ...o", x1, x2, self.weight)
        if self.bias is not None:
            out = out + self.bias
        return out


class AllToShardedLinear(Module):
    """Distributed linear layer where each rank computes a shard of the output features."""

    def __init__(
        self,
        input_dims: int,
        output_dims: int,
        bias: bool = True,
        group: Group | None = None,
    ) -> None:
        """Initialize `AllToShardedLinear`.

        Args:
            input_dims: Input feature dimension.
            output_dims: Total output feature dimension across the group.
            bias: Whether to learn an additive bias.
            group: Optional distributed `Group`.

        """
        super().__init__()
        self.group: Group | None = group
        scale: float = math.sqrt(1.0 / float(input_dims))
        self.weight: array = uniform(low=-scale, high=scale, shape=(output_dims, input_dims))
        self.bias: array | None = (
            uniform(low=-scale, high=scale, shape=(output_dims,)) if bias else None
        )

    def __call__(self, x: array) -> array:
        """Apply the sharded linear transformation to `x`.

        Args:
            x: Input `array`.

        Returns:
            Output `array`.

        """
        out = mx.matmul(x, self.weight.T)
        if self.bias is not None:
            out = out + self.bias
        return out


class ShardedToAllLinear(Module):
    """Distributed linear layer aggregating sharded inputs into a replicated output."""

    def __init__(
        self,
        input_dims: int,
        output_dims: int,
        bias: bool = True,
        group: Group | None = None,
    ) -> None:
        """Initialize `ShardedToAllLinear`.

        Args:
            input_dims: Input feature dimension.
            output_dims: Output feature dimension.
            bias: Whether to learn an additive bias.
            group: Optional distributed `Group`.

        """
        super().__init__()
        self.group: Group | None = group
        scale: float = math.sqrt(1.0 / float(input_dims))
        self.weight: array = uniform(low=-scale, high=scale, shape=(output_dims, input_dims))
        self.bias: array | None = (
            uniform(low=-scale, high=scale, shape=(output_dims,)) if bias else None
        )

    def __call__(self, x: array) -> array:
        """Apply the sharded-to-all linear transformation to `x`.

        Args:
            x: Input `array`.

        Returns:
            Output `array`.

        """
        out = mx.matmul(x, self.weight.T)
        if self.bias is not None:
            out = out + self.bias
        return out


class QuantizedLinear(Module):
    """Applies an affine transformation using a quantized weight matrix."""

    def __init__(
        self,
        input_dims: int,
        output_dims: int,
        bias: bool = True,
        group_size: int = 64,
        bits: int = 4,
        mode: str = "affine",
    ) -> None:
        """Initialize `QuantizedLinear`.

        Args:
            input_dims: Input feature dimension.
            output_dims: Output feature dimension.
            bias: Whether to include a learnable bias.
            group_size: Quantization group size.
            bits: Number of bits per quantized weight.
            mode: Quantization mode (`'affine'`, `'mxfp4'`, `'nvfp4'`, `'mxfp8'`).

        """
        super().__init__()
        scale: float = math.sqrt(1.0 / float(input_dims))
        self.weight: array = uniform(low=-scale, high=scale, shape=(output_dims, input_dims))
        self.scales: array = uniform(
            low=0.0, high=1.0, shape=(output_dims, max(1, input_dims // group_size))
        )
        self.biases: array = uniform(
            low=-scale, high=scale, shape=(output_dims, max(1, input_dims // group_size))
        )
        self.bias: array | None = (
            uniform(low=-scale, high=scale, shape=(output_dims,)) if bias else None
        )
        self.group_size: int = group_size
        self.bits: int = bits
        self.mode: str = mode

    def __call__(self, x: array) -> array:
        """Compute quantized linear projection on `x`.

        Args:
            x: Input `array`.

        Returns:
            Output `array`.

        """
        s = mx.repeat(self.scales, self.group_size, axis=-1)[..., : self.weight.shape[-1]]
        w_deq = self.weight * s
        out = mx.matmul(x, w_deq.T)
        if self.bias is not None:
            out = out + self.bias
        return out


class QuantizedAllToShardedLinear(Module):
    """Quantized version of `AllToShardedLinear`."""

    def __init__(
        self,
        input_dims: int,
        output_dims: int,
        bias: bool = True,
        group_size: int = 64,
        bits: int = 4,
        mode: str = "affine",
        group: Group | None = None,
    ) -> None:
        """Initialize `QuantizedAllToShardedLinear`.

        Args:
            input_dims: Input feature dimension.
            output_dims: Output feature dimension.
            bias: Whether to include a bias vector.
            group_size: Quantization group size.
            bits: Bits per quantized weight.
            mode: Quantization mode.
            group: Optional distributed `Group`.

        """
        super().__init__()
        self.group: Group | None = group
        self.mode: str = mode
        scale: float = math.sqrt(1.0 / float(input_dims))
        self.weight: array = uniform(low=-scale, high=scale, shape=(output_dims, input_dims))
        self.scales: array = uniform(
            low=0.0, high=1.0, shape=(output_dims, max(1, input_dims // group_size))
        )
        self.bias: array | None = (
            uniform(low=-scale, high=scale, shape=(output_dims,)) if bias else None
        )
        self.group_size: int = group_size
        self.bits: int = bits

    def __call__(self, x: array) -> array:
        """Compute quantized sharded linear transformation on `x`.

        Args:
            x: Input `array`.

        Returns:
            Output `array`.

        """
        s = mx.repeat(self.scales, self.group_size, axis=-1)[..., : self.weight.shape[-1]]
        w_deq = self.weight * s
        out = mx.matmul(x, w_deq.T)
        if self.bias is not None:
            out = out + self.bias
        return out


class QuantizedShardedToAllLinear(Module):
    """Quantized version of `ShardedToAllLinear`."""

    def __init__(
        self,
        input_dims: int,
        output_dims: int,
        bias: bool = True,
        group_size: int = 64,
        bits: int = 4,
        mode: str = "affine",
        group: Group | None = None,
    ) -> None:
        """Initialize `QuantizedShardedToAllLinear`.

        Args:
            input_dims: Input feature dimension.
            output_dims: Output feature dimension.
            bias: Whether to include a bias vector.
            group_size: Quantization group size.
            bits: Bits per quantized weight.
            mode: Quantization mode.
            group: Optional distributed `Group`.

        """
        super().__init__()
        self.group: Group | None = group
        self.mode: str = mode
        scale: float = math.sqrt(1.0 / float(input_dims))
        self.weight: array = uniform(low=-scale, high=scale, shape=(output_dims, input_dims))
        self.scales: array = uniform(
            low=0.0, high=1.0, shape=(output_dims, max(1, input_dims // group_size))
        )
        self.bias: array | None = (
            uniform(low=-scale, high=scale, shape=(output_dims,)) if bias else None
        )
        self.group_size: int = group_size
        self.bits: int = bits

    def __call__(self, x: array) -> array:
        """Compute quantized sharded-to-all linear transformation on `x`.

        Args:
            x: Input `array`.

        Returns:
            Output `array`.

        """
        s = mx.repeat(self.scales, self.group_size, axis=-1)[..., : self.weight.shape[-1]]
        w_deq = self.weight * s
        out = mx.matmul(x, w_deq.T)
        if self.bias is not None:
            out = out + self.bias
        return out


class QQLinear(Module):
    """Double-quantized linear layer (`QQLinear`) using `qqmm`."""

    def __init__(
        self,
        input_dims: int,
        output_dims: int,
        group_size: int | None = None,
        bits: int | None = None,
        mode: str = "nvfp4",
    ) -> None:
        """Initialize `QQLinear`.

        Args:
            input_dims: Input feature dimension.
            output_dims: Output feature dimension.
            group_size: Optional quantization group size (defaults to 32).
            bits: Optional bits per weight (defaults to 4).
            mode: Double-quantization mode (`'nvfp4'` or `'mxfp8'`).

        """
        super().__init__()
        self.input_dims: int = input_dims
        self.output_dims: int = output_dims
        self.group_size: int = 32 if group_size is None else group_size
        self.bits: int = 4 if bits is None else bits
        self.mode: str = mode
        scale: float = math.sqrt(1.0 / float(input_dims))
        self.weight: array = uniform(low=-scale, high=scale, shape=(output_dims, input_dims))
        self.scales: array = uniform(
            low=0.0,
            high=1.0,
            shape=(output_dims, max(1, input_dims // self.group_size)),
        )

    def __call__(self, x: array) -> array:
        """Apply double-quantized matrix multiplication `qqmm(x, weight)`.

        Args:
            x: Input `array` of shape `(..., input_dims)`.

        Returns:
            Output `array` of shape `(..., output_dims)`.

        """
        return qqmm(
            x,
            self.weight,
            scales=self.scales,
            group_size=self.group_size,
            bits=self.bits,
            mode=self.mode,
        )


__all__ = [
    "AllToShardedLinear",
    "Bilinear",
    "Linear",
    "QQLinear",
    "QuantizedAllToShardedLinear",
    "QuantizedLinear",
    "QuantizedShardedToAllLinear",
    "ShardedToAllLinear",
]
