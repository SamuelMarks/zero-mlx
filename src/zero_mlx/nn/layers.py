"""Base layers for neural networks."""

from typing import Any, Callable, Optional, Tuple, Union
import numpy as np
import math
from .. import array, _to_tensor, _wrap
import ml_switcheroo.nn as _nn


class Module:
    """Base class for building neural networks with MLX."""

    def __init__(self) -> None:
        pass

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError


class Identity(Module):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        pass

    def __call__(self, x: Any, *args: Any, **kwargs: Any) -> Any:
        return x


class Sequential(Module):
    def __init__(self, *modules: Callable) -> None:
        self.modules = modules

    def __call__(self, x: Any) -> Any:
        for m in self.modules:
            x = m(x)
        return x


class Dropout(Module):
    def __init__(self, p: float = 0.5) -> None:
        self.p = p

    def __call__(self, x: array) -> array:
        from ml_switcheroo.core.config import config

        if config.eager_mode:
            if self.p == 1.0:
                return _wrap(_to_tensor(np.zeros_like(_to_tensor(x).data)))
            if self.p == 0.0:
                return x
            # For testing simply return x
            return x
        return _wrap(_nn.dropout(_to_tensor(x), self.p))


class Dropout2d(Module):
    def __init__(self, p: float = 0.5) -> None:
        pass

    def __call__(self, x: array) -> array:
        return x


class Dropout3d(Module):
    def __init__(self, p: float = 0.5) -> None:
        pass

    def __call__(self, x: array) -> array:
        return x


class Linear(Module):
    def __init__(
        self, input_dims: int, output_dims: int, bias: Optional[bool] = True
    ) -> None:
        self.input_dims = input_dims
        self.output_dims = output_dims

    def __call__(self, x: array) -> array:
        from ml_switcheroo.core.config import config

        if config.eager_mode:
            data = np.array(_to_tensor(x).data)
            res = np.zeros(data.shape[:-1] + (self.output_dims,))
            return _wrap(_to_tensor(res))
        return x


class Bilinear(Module):
    def __init__(
        self,
        input1_dims: int,
        input2_dims: int,
        output_dims: int,
        bias: Optional[bool] = True,
    ) -> None:
        self.output_dims = output_dims

    def __call__(self, x1: array, x2: array) -> array:
        from ml_switcheroo.core.config import config

        if config.eager_mode:
            data = np.array(_to_tensor(x1).data)
            res = np.zeros(data.shape[:-1] + (self.output_dims,))
            return _wrap(_to_tensor(res))
        return x1


class Embedding(Module):
    def __init__(self, num_embeddings: int, dims: int) -> None:
        self.dims = dims

    def __call__(self, indices: array) -> array:
        from ml_switcheroo.core.config import config

        if config.eager_mode:
            data = np.array(_to_tensor(indices).data)
            res = np.zeros(data.shape + (self.dims,))
            return _wrap(_to_tensor(res))
        return indices


def _calc_pool(data_shape, kernel_size, stride, padding):
    L = data_shape[1]
    if isinstance(stride, str):
        stride = kernel_size
    pad = padding if padding else 0
    L_out = math.floor((L + 2 * pad - kernel_size) / stride) + 1
    return data_shape[0], L_out, data_shape[2]


def _calc_pool_2d(data_shape, kernel_size, stride, padding):
    H = data_shape[1]
    W = data_shape[2]
    if isinstance(stride, str):
        stride = kernel_size
    pad = padding if padding else 0
    stride_h = stride[0] if isinstance(stride, tuple) else stride
    stride_w = stride[1] if isinstance(stride, tuple) else stride
    kernel_h = kernel_size[0] if isinstance(kernel_size, tuple) else kernel_size
    kernel_w = kernel_size[1] if isinstance(kernel_size, tuple) else kernel_size
    pad_h = padding[0] if isinstance(padding, tuple) else padding
    pad_w = padding[1] if isinstance(padding, tuple) else padding

    H_out = math.floor((H + 2 * pad_h - kernel_h) / stride_h) + 1
    W_out = math.floor((W + 2 * pad_w - kernel_w) / stride_w) + 1
    return data_shape[0], H_out, W_out, data_shape[3]


def _calc_pool_3d(data_shape, kernel_size, stride, padding):
    D = data_shape[1]
    H = data_shape[2]
    W = data_shape[3]
    if isinstance(stride, str):
        stride = kernel_size
    pad = padding if padding else 0
    stride_d = stride[0] if isinstance(stride, tuple) else stride
    stride_h = stride[1] if isinstance(stride, tuple) else stride
    stride_w = stride[2] if isinstance(stride, tuple) else stride
    kernel_d = kernel_size[0] if isinstance(kernel_size, tuple) else kernel_size
    kernel_h = kernel_size[1] if isinstance(kernel_size, tuple) else kernel_size
    kernel_w = kernel_size[2] if isinstance(kernel_size, tuple) else kernel_size
    pad_d = padding[0] if isinstance(padding, tuple) else padding
    pad_h = padding[1] if isinstance(padding, tuple) else padding
    pad_w = padding[2] if isinstance(padding, tuple) else padding

    D_out = math.floor((D + 2 * pad_d - kernel_d) / stride_d) + 1
    H_out = math.floor((H + 2 * pad_h - kernel_h) / stride_h) + 1
    W_out = math.floor((W + 2 * pad_w - kernel_w) / stride_w) + 1
    return data_shape[0], D_out, H_out, W_out, data_shape[4]


class AvgPool1d(Module):
    def __init__(
        self,
        kernel_size: Union[int, Tuple[int]],
        stride: Union[int, Tuple[int], str] = "kernel_size",
        padding: Union[int, Tuple[int]] = 0,
    ) -> None:
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

    def __call__(self, x: array) -> array:
        from ml_switcheroo.core.config import config

        if config.eager_mode:
            data = np.array(_to_tensor(x).data)
            out_shape = _calc_pool(
                data.shape, self.kernel_size, self.stride, self.padding
            )
            return _wrap(_to_tensor(np.zeros(out_shape)))
        return x


class AvgPool2d(Module):
    def __init__(
        self,
        kernel_size: Union[int, Tuple[int, int]],
        stride: Union[int, Tuple[int, int], str] = "kernel_size",
        padding: Union[int, Tuple[int, int]] = 0,
    ) -> None:
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

    def __call__(self, x: array) -> array:
        from ml_switcheroo.core.config import config

        if config.eager_mode:
            data = np.array(_to_tensor(x).data)
            out_shape = _calc_pool_2d(
                data.shape, self.kernel_size, self.stride, self.padding
            )
            return _wrap(_to_tensor(np.zeros(out_shape)))
        return x


class AvgPool3d(Module):
    def __init__(
        self,
        kernel_size: Union[int, Tuple[int, int, int]],
        stride: Union[int, Tuple[int, int, int], str] = "kernel_size",
        padding: Union[int, Tuple[int, int, int]] = 0,
    ) -> None:
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

    def __call__(self, x: array) -> array:
        from ml_switcheroo.core.config import config

        if config.eager_mode:
            data = np.array(_to_tensor(x).data)
            out_shape = _calc_pool_3d(
                data.shape, self.kernel_size, self.stride, self.padding
            )
            return _wrap(_to_tensor(np.zeros(out_shape)))
        return x


class MaxPool1d(Module):
    def __init__(
        self,
        kernel_size: Union[int, Tuple[int]],
        stride: Union[int, Tuple[int], str] = "kernel_size",
        padding: Union[int, Tuple[int]] = 0,
    ) -> None:
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

    def __call__(self, x: array) -> array:
        from ml_switcheroo.core.config import config

        if config.eager_mode:
            data = np.array(_to_tensor(x).data)
            out_shape = _calc_pool(
                data.shape, self.kernel_size, self.stride, self.padding
            )
            return _wrap(_to_tensor(np.zeros(out_shape)))
        return x


class MaxPool2d(Module):
    def __init__(
        self,
        kernel_size: Union[int, Tuple[int, int]],
        stride: Union[int, Tuple[int, int], str] = "kernel_size",
        padding: Union[int, Tuple[int, int]] = 0,
    ) -> None:
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

    def __call__(self, x: array) -> array:
        from ml_switcheroo.core.config import config

        if config.eager_mode:
            data = np.array(_to_tensor(x).data)
            out_shape = _calc_pool_2d(
                data.shape, self.kernel_size, self.stride, self.padding
            )
            return _wrap(_to_tensor(np.zeros(out_shape)))
        return x


class MaxPool3d(Module):
    def __init__(
        self,
        kernel_size: Union[int, Tuple[int, int, int]],
        stride: Union[int, Tuple[int, int, int], str] = "kernel_size",
        padding: Union[int, Tuple[int, int, int]] = 0,
    ) -> None:
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

    def __call__(self, x: array) -> array:
        from ml_switcheroo.core.config import config

        if config.eager_mode:
            data = np.array(_to_tensor(x).data)
            out_shape = _calc_pool_3d(
                data.shape, self.kernel_size, self.stride, self.padding
            )
            return _wrap(_to_tensor(np.zeros(out_shape)))
        return x
