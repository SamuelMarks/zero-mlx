"""mlx.nn.normalization module."""

from typing import Optional, Union, Tuple
from zero_mlx.nn.base import Module
from zero_mlx.array import array
from zero_mlx.mlx_random import uniform
import ml_switcheroo_compiler.ops as sops


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
            self.weight = sops.ones((num_features,))
            self.bias = sops.zeros((num_features,))
        else:
            self.weight = None
            self.bias = None

        if self.track_running_stats:
            self.running_mean = sops.zeros((num_features,))
            self.running_var = sops.ones((num_features,))

    def __call__(self, x: array) -> array:
        """Call."""
        x_t = x._tensor if hasattr(x, "_tensor") else x

        # We don't have training state routing for running_mean updates natively in the math op,
        # but for parity we route what we have.
        mean_val = getattr(self, "running_mean", sops.zeros((self.num_features,)))
        var_val = getattr(self, "running_var", sops.ones((self.num_features,)))
        mean_val_t = mean_val._tensor if hasattr(mean_val, "_tensor") else mean_val
        var_val_t = var_val._tensor if hasattr(var_val, "_tensor") else var_val

        config = sops.BatchNormConfig(
            offset=self.bias._tensor
            if hasattr(self.bias, "_tensor")
            else self.bias
            if self.bias is not None
            else None,
            scale=self.weight._tensor
            if hasattr(self.weight, "_tensor")
            else self.weight
            if self.weight is not None
            else None,
            epsilon=self.eps,
        )

        out = sops.batch_normalization(
            x_t,
            mean_val_t,
            var_val_t,
            axis=tuple(range(len(x.shape) - 1)),
            config=config,
        )
        return array(out)


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
            self.weight = sops.ones((num_channels,))
            self.bias = sops.zeros((num_channels,))
        else:
            self.weight = None
            self.bias = None

    def __call__(self, x: array) -> array:
        """Call."""
        x_t = x._tensor if hasattr(x, "_tensor") else x
        w_t = (
            self.weight._tensor
            if hasattr(self.weight, "_tensor")
            else self.weight
            if self.weight is not None
            else None
        )
        b_t = (
            self.bias._tensor
            if hasattr(self.bias, "_tensor")
            else self.bias
            if self.bias is not None
            else None
        )

        out = sops.group_norm(
            x_t, num_groups=self.num_groups, scale=w_t, offset=b_t, epsilon=self.eps
        )
        return array(out)


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
            self.weight = sops.ones((num_features,))
            self.bias = sops.zeros((num_features,))
        else:
            self.weight = None
            self.bias = None

    def __call__(self, x: array) -> array:
        """Call."""
        x_t = x._tensor if hasattr(x, "_tensor") else x
        w_t = (
            self.weight._tensor
            if hasattr(self.weight, "_tensor")
            else self.weight
            if self.weight is not None
            else None
        )
        b_t = (
            self.bias._tensor
            if hasattr(self.bias, "_tensor")
            else self.bias
            if self.bias is not None
            else None
        )

        out = sops.instance_norm(x_t, scale=w_t, offset=b_t, epsilon=self.eps)
        return array(out)


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
            self.weight = sops.ones(self.dims)
            self.bias = sops.zeros(self.dims)
        else:
            self.weight = None
            self.bias = None

    def __call__(self, x: array) -> array:
        """Call."""
        x_t = x._tensor if hasattr(x, "_tensor") else x
        w_t = (
            self.weight._tensor
            if hasattr(self.weight, "_tensor")
            else self.weight
            if self.weight is not None
            else None
        )
        b_t = (
            self.bias._tensor
            if hasattr(self.bias, "_tensor")
            else self.bias
            if self.bias is not None
            else None
        )

        out = sops.layer_norm(
            x_t, normalized_shape=self.dims, scale=w_t, offset=b_t, epsilon=self.eps
        )
        return array(out)


class RMSNorm(Module):
    """Applies Root Mean Square Normalization."""

    def __init__(self, dims: Union[int, Tuple[int, ...]], eps: float = 1e-5):
        """Initialize RMSNorm."""
        super().__init__()
        self.dims = dims if isinstance(dims, tuple) else (dims,)
        self.eps = eps
        self.weight = sops.ones(self.dims)

    def __call__(self, x: array) -> array:
        """Call."""
        x_t = x._tensor if hasattr(x, "_tensor") else x
        w_t = self.weight._tensor if hasattr(self.weight, "_tensor") else self.weight
        out = sops.rms_normalization(x_t, scale=w_t, epsilon=self.eps)
        return array(out)


__all__ = ["BatchNorm", "GroupNorm", "InstanceNorm", "LayerNorm", "RMSNorm"]
