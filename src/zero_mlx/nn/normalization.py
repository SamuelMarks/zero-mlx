"""mlx.nn.normalization module."""

from typing import Optional, Union, Tuple
from zero_mlx.nn.base import Module
from zero_mlx.array import array
import zero_mlx as mx


class BatchNorm(Module):
    """Applies Batch Normalization over a 2D or 3D input."""

    def __init__(
        self,
        num_features: int,
        eps: float = 1e-5,
        momentum: float = 0.1,
        affine: bool = True,
        track_running_stats: bool = True,
    ):
        """Initialize BatchNorm."""
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.track_running_stats = track_running_stats

        if self.affine:
            self.weight = mx.ones((num_features,))
            self.bias = mx.zeros((num_features,))
        else:
            self.weight = None
            self.bias = None

        if self.track_running_stats:
            self.running_mean = mx.zeros((num_features,))
            self.running_var = mx.ones((num_features,))

    def __call__(self, x: array) -> array:
        """Call."""
        axis = tuple(range(len(x.shape) - 1))

        # for training we should update running stats, but we just implement the forward pass
        mean = mx.mean(x, axis=axis, keepdims=True)
        var = mx.var(x, axis=axis, keepdims=True, ddof=0)

        out = mx.divide(mx.subtract(x, mean), mx.sqrt(mx.add(var, self.eps)))

        if self.weight is not None:
            out = mx.multiply(out, self.weight)
        if self.bias is not None:
            out = mx.add(out, self.bias)

        return out


class GroupNorm(Module):
    """Applies Group Normalization over a mini-batch of inputs."""

    def __init__(
        self, num_groups: int, num_channels: int, eps: float = 1e-5, affine: bool = True
    ):
        """Initialize GroupNorm."""
        super().__init__()
        self.num_groups = num_groups
        self.num_channels = num_channels
        self.eps = eps
        self.affine = affine

        if self.affine:
            self.weight = mx.ones((num_channels,))
            self.bias = mx.zeros((num_channels,))
        else:
            self.weight = None
            self.bias = None

    def __call__(self, x: array) -> array:
        """Call."""
        shape = x.shape
        batch = shape[0]
        channels = (
            shape[-1] if len(shape) > 2 else shape[1]
        )  # assuming channel last for standard mlx or we just infer
        # Usually group norm shape is (N, C, H, W) or (N, H, W, C).
        # Let's assume channels last (N, ..., C) since it's common in MLX.

        reshaped_x = mx.reshape(
            x, (batch, -1, self.num_groups, channels // self.num_groups)
        )

        mean = mx.mean(reshaped_x, axis=(1, 3), keepdims=True)
        var = mx.var(reshaped_x, axis=(1, 3), keepdims=True, ddof=0)

        out = mx.divide(mx.subtract(reshaped_x, mean), mx.sqrt(mx.add(var, self.eps)))
        out = mx.reshape(out, shape)

        if self.weight is not None:
            w_shape = [1, self.num_channels] + [1] * (out.ndim - 2)
            w_t = mx.reshape(self.weight, w_shape)
            out = mx.multiply(out, w_t)
        if self.bias is not None:
            b_shape = [1, self.num_channels] + [1] * (out.ndim - 2)
            b_t = mx.reshape(self.bias, b_shape)
            out = mx.add(out, b_t)

        return out


class InstanceNorm(Module):
    """Applies Instance Normalization over a mini-batch of inputs."""

    def __init__(
        self,
        num_features: int,
        eps: float = 1e-5,
        affine: bool = False,
        track_running_stats: bool = False,
    ):
        """Initialize InstanceNorm."""
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        self.track_running_stats = track_running_stats

        if self.affine:
            self.weight = mx.ones((num_features,))
            self.bias = mx.zeros((num_features,))
        else:
            self.weight = None
            self.bias = None

    def __call__(self, x: array) -> array:
        """Call."""
        # Instance norm: mean and var over spatial dimensions
        axis = tuple(range(1, len(x.shape) - 1)) if len(x.shape) > 2 else (1,)

        mean = mx.mean(x, axis=axis, keepdims=True)
        var = mx.var(x, axis=axis, keepdims=True, ddof=0)

        out = mx.divide(mx.subtract(x, mean), mx.sqrt(mx.add(var, self.eps)))

        if self.weight is not None:
            w_shape = [1, self.num_features] + [1] * (out.ndim - 2)
            w_t = mx.reshape(self.weight, w_shape)
            out = mx.multiply(out, w_t)
        if self.bias is not None:
            b_shape = [1, self.num_features] + [1] * (out.ndim - 2)
            b_t = mx.reshape(self.bias, b_shape)
            out = mx.add(out, b_t)

        return out


class LayerNorm(Module):
    """Applies Layer Normalization over a mini-batch of inputs."""

    def __init__(
        self, dims: Union[int, Tuple[int, ...]], eps: float = 1e-5, affine: bool = True
    ):
        """Initialize LayerNorm."""
        super().__init__()
        self.dims = dims if isinstance(dims, tuple) else (dims,)
        self.eps = eps
        self.affine = affine

        if self.affine:
            self.weight = mx.ones(self.dims)
            self.bias = mx.zeros(self.dims)
        else:
            self.weight = None
            self.bias = None

    def __call__(self, x: array) -> array:
        """Call."""
        axis = tuple(range(len(x.shape) - len(self.dims), len(x.shape)))

        mean = mx.mean(x, axis=axis, keepdims=True)
        var = mx.var(x, axis=axis, keepdims=True, ddof=0)

        out = mx.divide(mx.subtract(x, mean), mx.sqrt(mx.add(var, self.eps)))

        if self.weight is not None:
            out = mx.multiply(out, self.weight)
        if self.bias is not None:
            out = mx.add(out, self.bias)

        return out


class RMSNorm(Module):
    """Applies Root Mean Square Normalization."""

    def __init__(self, dims: Union[int, Tuple[int, ...]], eps: float = 1e-5):
        """Initialize RMSNorm."""
        super().__init__()
        self.dims = dims if isinstance(dims, tuple) else (dims,)
        self.eps = eps
        self.weight = mx.ones(self.dims)

    def __call__(self, x: array) -> array:
        """Call."""
        axis = tuple(range(len(x.shape) - len(self.dims), len(x.shape)))

        var = mx.mean(mx.square(x), axis=axis, keepdims=True)
        out = mx.divide(x, mx.sqrt(mx.add(var, self.eps)))

        out = mx.multiply(out, self.weight)
        return out


__all__ = ["BatchNorm", "GroupNorm", "InstanceNorm", "LayerNorm", "RMSNorm"]
