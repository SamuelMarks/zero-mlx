"""Normalization layers (`BatchNorm`, `GroupNorm`, `InstanceNorm`, `LayerNorm`, `RMSNorm`) for `mlx.nn`."""

from __future__ import annotations

from zero_mlx.array import array
from zero_mlx.nn.base import Module
import zero_mlx.ops as mx


class BatchNorm(Module):
    """Applies Batch Normalization over a 2D or 3D input `(N, ..., C)`."""

    def __init__(
        self,
        num_features: int,
        eps: float = 1e-5,
        momentum: float = 0.1,
        affine: bool = True,
        track_running_stats: bool = True,
    ) -> None:
        """Initialize `BatchNorm`.

        Args:
            num_features: Number of feature channels `C`.
            eps: Value added to denominator for numerical stability.
            momentum: Momentum factor for running mean and variance updates.
            affine: Whether to learn `weight` and `bias` parameters.
            track_running_stats: Whether to track `running_mean` and `running_var`.

        """
        super().__init__()
        self.num_features: int = num_features
        self.eps: float = eps
        self.momentum: float = momentum
        self.affine: bool = affine
        self.track_running_stats: bool = track_running_stats

        self.weight: array | None = mx.ones((num_features,)) if affine else None
        self.bias: array | None = mx.zeros((num_features,)) if affine else None
        self.running_mean: array | None = mx.zeros((num_features,)) if track_running_stats else None
        self.running_var: array | None = mx.ones((num_features,)) if track_running_stats else None

    def __call__(self, x: array) -> array:
        """Apply Batch Normalization to `x`.

        Args:
            x: Input `array` of shape `(N, ..., C)`.

        Returns:
            Normalized `array`.

        """
        axis = tuple(range(len(x.shape) - 1))
        batch_mean = mx.mean(x, axis=axis, keepdims=True)
        batch_var = mx.var(x, axis=axis, keepdims=True, ddof=0)
        if (
            self.track_running_stats
            and self.running_mean is not None
            and self.running_var is not None
        ):
            if self.training:
                mean_1d = mx.mean(x, axis=axis, keepdims=False)
                var_1d = mx.var(x, axis=axis, keepdims=False, ddof=0)
                self.running_mean = (
                    1.0 - self.momentum
                ) * self.running_mean + self.momentum * mean_1d
                self.running_var = (1.0 - self.momentum) * self.running_var + self.momentum * var_1d
                mean = batch_mean
                var = batch_var
            else:
                mean = self.running_mean
                var = self.running_var
        else:
            mean = batch_mean
            var = batch_var
        out = mx.divide(mx.subtract(x, mean), mx.sqrt(mx.add(var, self.eps)))
        if self.weight is not None:
            out = mx.multiply(out, self.weight)
        if self.bias is not None:
            out = mx.add(out, self.bias)
        return out


class GroupNorm(Module):
    """Applies Group Normalization over a mini-batch of inputs."""

    def __init__(
        self,
        num_groups: int,
        dims: int,
        eps: float = 1e-5,
        affine: bool = True,
        pytorch_compatible: bool = False,
    ) -> None:
        """Initialize `GroupNorm`.

        Args:
            num_groups: Number of groups to separate the channels into.
            dims: Number of channels/features in the input.
            eps: Value added to denominator for numerical stability.
            affine: Whether to learn per-channel `weight` and `bias` parameters.
            pytorch_compatible: Whether to use PyTorch-compatible group layout.

        """
        super().__init__()
        self.num_groups: int = num_groups
        self.dims: int = dims
        self.num_channels: int = dims
        self.eps: float = eps
        self.affine: bool = affine
        self.pytorch_compatible: bool = pytorch_compatible

        self.weight: array | None = mx.ones((dims,)) if affine else None
        self.bias: array | None = mx.zeros((dims,)) if affine else None

    def __call__(self, x: array) -> array:
        """Apply Group Normalization to `x`.

        Args:
            x: Input `array`.

        Returns:
            Group-normalized `array`.

        """
        shape = x.shape
        batch = shape[0]
        group_size = self.dims // self.num_groups
        if self.pytorch_compatible:
            reshaped_x = mx.reshape(x, (batch, -1, self.num_groups, group_size))
            mean = mx.mean(reshaped_x, axis=(1, 3), keepdims=True)
            var = mx.var(reshaped_x, axis=(1, 3), keepdims=True, ddof=0)
        else:
            reshaped_x = mx.reshape(x, (batch, -1, group_size, self.num_groups))
            mean = mx.mean(reshaped_x, axis=(1, 2), keepdims=True)
            var = mx.var(reshaped_x, axis=(1, 2), keepdims=True, ddof=0)
        out = mx.divide(mx.subtract(reshaped_x, mean), mx.sqrt(mx.add(var, self.eps)))
        out = mx.reshape(out, shape)

        is_channel_last: bool = shape[-1] == self.dims
        if self.weight is not None:
            w_t = (
                self.weight
                if is_channel_last
                else mx.reshape(self.weight, [1, self.dims] + [1] * (out.ndim - 2))
            )
            out = mx.multiply(out, w_t)
        if self.bias is not None:
            b_t = (
                self.bias
                if is_channel_last
                else mx.reshape(self.bias, [1, self.dims] + [1] * (out.ndim - 2))
            )
            out = mx.add(out, b_t)
        return out


class InstanceNorm(Module):
    """Applies Instance Normalization over a mini-batch of inputs."""

    def __init__(
        self,
        dims: int,
        eps: float = 1e-5,
        affine: bool = False,
    ) -> None:
        """Initialize `InstanceNorm`.

        Args:
            dims: Number of features/channels `C`.
            eps: Value added to denominator for numerical stability.
            affine: If True, this module has learnable affine parameters.

        """
        super().__init__()
        self.dims: int = dims
        self.num_features: int = dims
        self.eps: float = eps
        self.affine: bool = affine

        self.weight: array | None = mx.ones((dims,)) if affine else None
        self.bias: array | None = mx.zeros((dims,)) if affine else None

    def __call__(self, x: array) -> array:
        """Apply Instance Normalization to `x`.

        Args:
            x: Input `array`.

        Returns:
            Instance-normalized `array`.

        """
        axis = tuple(range(1, len(x.shape) - 1)) if len(x.shape) > 2 else (1,)
        mean = mx.mean(x, axis=axis, keepdims=True)
        var = mx.var(x, axis=axis, keepdims=True, ddof=0)
        out = mx.divide(mx.subtract(x, mean), mx.sqrt(mx.add(var, self.eps)))

        is_channel_last: bool = x.shape[-1] == self.dims
        if self.weight is not None:
            w_t = (
                self.weight
                if is_channel_last
                else mx.reshape(self.weight, [1, self.dims] + [1] * (out.ndim - 2))
            )
            out = mx.multiply(out, w_t)
        if self.bias is not None:
            b_t = (
                self.bias
                if is_channel_last
                else mx.reshape(self.bias, [1, self.dims] + [1] * (out.ndim - 2))
            )
            out = mx.add(out, b_t)
        return out


class LayerNorm(Module):
    """Applies Layer Normalization over the last dimension(s) of the input."""

    def __init__(
        self,
        dims: int | tuple[int, ...],
        eps: float = 1e-5,
        affine: bool = True,
        bias: bool = True,
    ) -> None:
        """Initialize `LayerNorm`.

        Args:
            dims: Feature dimension or tuple of trailing dimensions to normalize over.
            eps: Value added to denominator for numerical stability.
            affine: Whether to learn multiplicative `weight` (and `bias` if `bias=True`).
            bias: Whether to include a learnable additive `bias` when `affine=True`.

        """
        super().__init__()
        self.dims: tuple[int, ...] = dims if isinstance(dims, tuple) else (dims,)
        self.eps: float = eps
        self.affine: bool = affine

        self.weight: array | None = mx.ones(self.dims) if affine else None
        self.bias: array | None = mx.zeros(self.dims) if (affine and bias) else None

    def __call__(self, x: array) -> array:
        """Apply Layer Normalization to `x`.

        Args:
            x: Input `array`.

        Returns:
            Layer-normalized `array`.

        """
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
    """Applies Root Mean Square Normalization over the last dimension(s) of the input."""

    def __init__(self, dims: int | tuple[int, ...], eps: float = 1e-5) -> None:
        """Initialize `RMSNorm`.

        Args:
            dims: Feature dimension or tuple of trailing dimensions.
            eps: Value added to mean square for numerical stability.

        """
        super().__init__()
        self.dims: tuple[int, ...] = dims if isinstance(dims, tuple) else (dims,)
        self.eps: float = eps
        self.weight: array = mx.ones(self.dims)

    def __call__(self, x: array) -> array:
        """Apply RMS Normalization to `x`.

        Args:
            x: Input `array`.

        Returns:
            RMS-normalized `array`.

        """
        axis = tuple(range(len(x.shape) - len(self.dims), len(x.shape)))
        var = mx.mean(mx.square(x), axis=axis, keepdims=True)
        out = mx.divide(x, mx.sqrt(mx.add(var, self.eps)))
        return mx.multiply(out, self.weight)


__all__ = ["BatchNorm", "GroupNorm", "InstanceNorm", "LayerNorm", "RMSNorm"]
