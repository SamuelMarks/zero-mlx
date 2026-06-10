"""Distributed and Quantized layers for neural networks."""

import numpy as np
from .. import array, _to_tensor, _wrap
from .layers import Module


class AllToShardedLinear(Module):
    def __init__(self, input_dims: int, output_dims: int, bias: bool = True) -> None:
        self.output_dims = output_dims

    def __call__(self, x: array) -> array:
        from ml_switcheroo.core.config import config

        if config.eager_mode:
            data = np.array(_to_tensor(x).data)
            res = np.zeros(data.shape[:-1] + (self.output_dims,))
            return _wrap(_to_tensor(res))
        return x


class QuantizedAllToShardedLinear(Module):
    def __init__(
        self,
        input_dims: int,
        output_dims: int,
        bias: bool = True,
        group_size: int = 64,
        bits: int = 4,
    ) -> None:
        self.output_dims = output_dims

    def __call__(self, x: array) -> array:
        from ml_switcheroo.core.config import config

        if config.eager_mode:
            data = np.array(_to_tensor(x).data)
            res = np.zeros(data.shape[:-1] + (self.output_dims,))
            return _wrap(_to_tensor(res))
        return x


class QuantizedEmbedding(Module):
    def __init__(
        self, num_embeddings: int, dims: int, group_size: int = 64, bits: int = 4
    ) -> None:
        self.dims = dims

    def __call__(self, indices: array) -> array:
        from ml_switcheroo.core.config import config

        if config.eager_mode:
            data = np.array(_to_tensor(indices).data)
            res = np.zeros(data.shape + (self.dims,))
            return _wrap(_to_tensor(res))
        return indices


class QuantizedLinear(Module):
    def __init__(
        self,
        input_dims: int,
        output_dims: int,
        bias: bool = True,
        group_size: int = 64,
        bits: int = 4,
    ) -> None:
        self.output_dims = output_dims

    def __call__(self, x: array) -> array:
        from ml_switcheroo.core.config import config

        if config.eager_mode:
            data = np.array(_to_tensor(x).data)
            res = np.zeros(data.shape[:-1] + (self.output_dims,))
            return _wrap(_to_tensor(res))
        return x


class QuantizedShardedToAllLinear(Module):
    def __init__(
        self,
        input_dims: int,
        output_dims: int,
        bias: bool = True,
        group_size: int = 64,
        bits: int = 4,
    ) -> None:
        self.output_dims = output_dims

    def __call__(self, x: array) -> array:
        from ml_switcheroo.core.config import config

        if config.eager_mode:
            data = np.array(_to_tensor(x).data)
            res = np.zeros(data.shape[:-1] + (self.output_dims,))
            return _wrap(_to_tensor(res))
        return x


class ShardedToAllLinear(Module):
    def __init__(self, input_dims: int, output_dims: int, bias: bool = True) -> None:
        self.output_dims = output_dims

    def __call__(self, x: array) -> array:
        from ml_switcheroo.core.config import config

        if config.eager_mode:
            data = np.array(_to_tensor(x).data)
            res = np.zeros(data.shape[:-1] + (self.output_dims,))
            return _wrap(_to_tensor(res))
        return x
