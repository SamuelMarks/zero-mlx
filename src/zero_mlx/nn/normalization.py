"""Normalization layers for neural networks."""

import numpy as np

from .. import array
from .layers import Module


class BatchNorm(Module):
    """Applies Batch Normalization over a 2D or 3D input."""

    def __init__(
        self,
        num_features: int,
        eps: float = 1e-05,
        momentum: float = 0.1,
        affine: bool = True,
        track_running_stats: bool = True,
    ) -> None:
        """Initialize the BatchNorm layer."""
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.track_running_stats = track_running_stats

        if self.affine:
            self.weight = np.ones(num_features)
            self.bias = np.zeros(num_features)

        if self.track_running_stats:
            self.running_mean = np.zeros(num_features)
            self.running_var = np.ones(num_features)

    def __call__(self, x: array) -> array:
        """Forward pass."""
        # Assume x is (N, ..., C) where C is the last dimension (MLX convention usually channels last)
        # If x is (N, C, ...) we would need to know the format. We assume channels last: (N, *, C)
        # To compute batch norm, we reduce over all dimensions except the last one (C).
        d = x.data
        reduce_axes = tuple(range(d.ndim - 1))

        # We can implement a simple pseudo-training mode based on whether running stats are updated.
        # But for MLX parity, usually it just uses batch stats if in training mode.
        # We don't have a global `training` flag in this stub module, so we assume training
        # if `track_running_stats` is True and we just update it.
        # Actually, standard behavior without a train/eval flag: compute batch stats, update running.

        # Calculate batch mean and var
        batch_mean = np.mean(d, axis=reduce_axes, keepdims=True)
        batch_var = np.var(d, axis=reduce_axes, keepdims=True)

        if self.track_running_stats:
            # Update running stats
            self.running_mean = (
                1 - self.momentum
            ) * self.running_mean + self.momentum * np.squeeze(batch_mean)
            # Unbiased variance for running stats
            n_elements = np.prod([d.shape[i] for i in reduce_axes])
            unbiased_var = (
                batch_var * (n_elements / (n_elements - 1))
                if n_elements > 1
                else batch_var
            )
            self.running_var = (
                1 - self.momentum
            ) * self.running_var + self.momentum * np.squeeze(unbiased_var)

            # Use running stats for normalization (as if eval mode) or batch stats (as if training)?
            # In many frameworks, `__call__` does batch norm with batch stats during training.
            # We'll use batch stats.
            mean_to_use = batch_mean
            var_to_use = batch_var
        else:
            mean_to_use = batch_mean
            var_to_use = batch_var

        # Normalize
        out = (d - mean_to_use) / np.sqrt(var_to_use + self.eps)

        if self.affine:
            out = out * self.weight + self.bias

        return array(out)


class GroupNorm(Module):
    """Applies Group Normalization to the inputs."""

    def __init__(
        self,
        num_groups: int,
        dims: int,
        eps: float = 1e-05,
        affine: bool = True,
        pytorch_compatible: bool = False,
    ) -> None:
        """Initialize the GroupNorm layer."""
        super().__init__()
        self.num_groups = num_groups
        self.dims = dims
        self.eps = eps
        self.affine = affine
        self.pytorch_compatible = pytorch_compatible

        if self.affine:
            self.weight = np.ones(dims)
            self.bias = np.zeros(dims)

    def __call__(self, x: array) -> array:
        """Forward pass."""
        d = x.data
        shape = d.shape
        N = shape[0]
        C = shape[-1]
        spatial_shape = shape[1:-1]

        assert C % self.num_groups == 0, "dims must be divisible by num_groups"
        group_channels = C // self.num_groups

        reshaped = d.reshape((N, *spatial_shape, self.num_groups, group_channels))

        reduce_axes = tuple(range(1, d.ndim - 1)) + (-1,)

        mean = np.mean(reshaped, axis=reduce_axes, keepdims=True)
        var = np.var(reshaped, axis=reduce_axes, keepdims=True)

        out = (reshaped - mean) / np.sqrt(var + self.eps)
        out = out.reshape(shape)

        if self.affine:
            out = out * self.weight + self.bias

        return array(out)


class InstanceNorm(Module):
    """Applies instance normalization on the inputs."""

    def __init__(self, dims: int, eps: float = 1e-05, affine: bool = False) -> None:
        """Initialize the InstanceNorm layer."""
        super().__init__()
        self.dims = dims
        self.eps = eps
        self.affine = affine

        if self.affine:
            self.weight = np.ones(dims)
            self.bias = np.zeros(dims)

    def __call__(self, x: array) -> array:
        """Forward pass."""
        d = x.data
        reduce_axes = tuple(range(1, d.ndim - 1))

        mean = np.mean(d, axis=reduce_axes, keepdims=True)
        var = np.var(d, axis=reduce_axes, keepdims=True)

        out = (d - mean) / np.sqrt(var + self.eps)

        if self.affine:
            out = out * self.weight + self.bias

        return array(out)


class LayerNorm(Module):
    """Applies layer normalization on the inputs."""

    def __init__(
        self, dims: int, eps: float = 1e-05, affine: bool = True, bias: bool = True
    ) -> None:
        """Initialize the LayerNorm layer."""
        super().__init__()
        self.dims = dims
        self.eps = eps
        self.affine = affine
        self.bias_flag = bias

        if self.affine:
            self.weight = np.ones(dims)
            if self.bias_flag:
                self.bias = np.zeros(dims)

    def __call__(self, x: array) -> array:
        """Forward pass."""
        d = x.data
        reduce_axes = (-1,)

        mean = np.mean(d, axis=reduce_axes, keepdims=True)
        var = np.var(d, axis=reduce_axes, keepdims=True)

        out = (d - mean) / np.sqrt(var + self.eps)

        if self.affine:
            if self.bias_flag:
                out = out * self.weight + self.bias
            else:
                out = out * self.weight

        return array(out)


class RMSNorm(Module):
    """Applies Root Mean Square normalization to the inputs."""

    def __init__(self, dims: int, eps: float = 1e-05) -> None:
        """Initialize the RMSNorm layer."""
        super().__init__()
        self.dims = dims
        self.eps = eps
        self.weight = np.ones(dims)

    def __call__(self, x: array) -> array:
        """Forward pass."""
        d = x.data
        reduce_axes = (-1,)

        rms = np.sqrt(np.mean(np.square(d), axis=reduce_axes, keepdims=True) + self.eps)
        out = (d / rms) * self.weight
        return array(out)
