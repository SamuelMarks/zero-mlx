"""Normalization layers for neural networks."""

import numpy as np
from .. import array, _to_tensor, _wrap
from .layers import Module


class BatchNorm(Module):
    def __init__(
        self,
        num_features: int,
        eps: float = 1e-05,
        momentum: float = 0.1,
        affine: bool = True,
        track_running_stats: bool = True,
    ) -> None:
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.track_running_stats = track_running_stats
        self.running_mean = np.zeros(num_features)
        self.running_var = np.ones(num_features)

    def __call__(self, x: array) -> array:
        from ml_switcheroo.core.config import config

        if config.eager_mode:
            data = np.array(_to_tensor(x).data)
            reduce_axes = tuple(range(len(data.shape) - 1))
            mean = np.mean(data, axis=reduce_axes, keepdims=True)
            var = np.var(data, axis=reduce_axes, keepdims=True)
            if self.track_running_stats:
                self.running_mean = (
                    1 - self.momentum
                ) * self.running_mean + self.momentum * np.squeeze(mean)
                self.running_var = (
                    1 - self.momentum
                ) * self.running_var + self.momentum * np.squeeze(var)
            res = (data - mean) / np.sqrt(var + self.eps)
            return _wrap(_to_tensor(res))
        return x


class GroupNorm(Module):
    def __init__(
        self,
        num_groups: int,
        dims: int,
        eps: float = 1e-05,
        affine: bool = True,
        pytorch_compatible: bool = False,
    ) -> None:
        self.num_groups = num_groups
        self.eps = eps

    def __call__(self, x: array) -> array:
        from ml_switcheroo.core.config import config

        if config.eager_mode:
            data = np.array(_to_tensor(x).data)
            orig_shape = data.shape
            new_shape = data.shape[:-1] + (
                self.num_groups,
                data.shape[-1] // self.num_groups,
            )
            reshaped = data.reshape(new_shape)
            reduce_axes = tuple(range(1, len(new_shape) - 2)) + (-1,)
            mean = np.mean(reshaped, axis=reduce_axes, keepdims=True)
            var = np.var(reshaped, axis=reduce_axes, keepdims=True)
            res = (reshaped - mean) / np.sqrt(var + self.eps)
            return _wrap(_to_tensor(res.reshape(orig_shape)))
        return x


class InstanceNorm(Module):
    def __init__(self, dims: int, eps: float = 1e-05, affine: bool = False) -> None:
        self.eps = eps

    def __call__(self, x: array) -> array:
        from ml_switcheroo.core.config import config

        if config.eager_mode:
            data = np.array(_to_tensor(x).data)
            reduce_axes = tuple(range(1, len(data.shape) - 1))
            mean = np.mean(data, axis=reduce_axes, keepdims=True)
            var = np.var(data, axis=reduce_axes, keepdims=True)
            res = (data - mean) / np.sqrt(var + self.eps)
            return _wrap(_to_tensor(res))
        return x


class LayerNorm(Module):
    def __init__(
        self, dims: int, eps: float = 1e-05, affine: bool = True, bias: bool = True
    ) -> None:
        self.eps = eps

    def __call__(self, x: array) -> array:
        from ml_switcheroo.core.config import config

        if config.eager_mode:
            data = np.array(_to_tensor(x).data)
            mean = np.mean(data, axis=-1, keepdims=True)
            var = np.var(data, axis=-1, keepdims=True)
            res = (data - mean) / np.sqrt(var + self.eps)
            return _wrap(_to_tensor(res))
        return x


class RMSNorm(Module):
    def __init__(self, dims: int, eps: float = 1e-05) -> None:
        self.eps = eps

    def __call__(self, x: array) -> array:
        from ml_switcheroo.core.config import config

        if config.eager_mode:
            data = np.array(_to_tensor(x).data)
            var = np.mean(data**2, axis=-1, keepdims=True)
            res = data / np.sqrt(var + self.eps)
            return _wrap(_to_tensor(res))
        return x
