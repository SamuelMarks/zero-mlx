"""Convolutional layers for neural networks."""

import numpy as np
from typing import Optional, Tuple, Union
from .. import array, _to_tensor, _wrap
from .layers import Module


class Conv1d(Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple[int]],
        stride: Union[int, Tuple[int]] = 1,
        padding: Union[int, Tuple[int]] = 0,
        dilation: Union[int, Tuple[int]] = 1,
        groups: int = 1,
        bias: bool = True,
    ) -> None:
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation

    def __call__(self, x: array) -> array:
        from ml_switcheroo.core.config import config

        if config.eager_mode:
            data = np.array(_to_tensor(x).data)
            # data: (N, L, C) -> output: (N, L_out, out_channels)
            k = (
                self.kernel_size[0]
                if isinstance(self.kernel_size, tuple)
                else self.kernel_size
            )
            s = self.stride[0] if isinstance(self.stride, tuple) else self.stride
            p = self.padding[0] if isinstance(self.padding, tuple) else self.padding
            d = self.dilation[0] if isinstance(self.dilation, tuple) else self.dilation
            import math

            l_out = math.floor((data.shape[1] + 2 * p - d * (k - 1) - 1) / s) + 1
            res = np.zeros((data.shape[0], l_out, self.out_channels))
            return _wrap(_to_tensor(res))
        return x


class Conv2d(Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple[int, int]],
        stride: Union[int, Tuple[int, int]] = 1,
        padding: Union[int, Tuple[int, int]] = 0,
        dilation: Union[int, Tuple[int, int]] = 1,
        groups: int = 1,
        bias: bool = True,
    ) -> None:
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation

    def __call__(self, x: array) -> array:
        from ml_switcheroo.core.config import config

        if config.eager_mode:
            data = np.array(_to_tensor(x).data)
            k_h = (
                self.kernel_size[0]
                if isinstance(self.kernel_size, tuple)
                else self.kernel_size
            )
            k_w = (
                self.kernel_size[1]
                if isinstance(self.kernel_size, tuple)
                else self.kernel_size
            )
            s_h = self.stride[0] if isinstance(self.stride, tuple) else self.stride
            s_w = self.stride[1] if isinstance(self.stride, tuple) else self.stride
            p_h = self.padding[0] if isinstance(self.padding, tuple) else self.padding
            p_w = self.padding[1] if isinstance(self.padding, tuple) else self.padding
            d_h = (
                self.dilation[0] if isinstance(self.dilation, tuple) else self.dilation
            )
            d_w = (
                self.dilation[1] if isinstance(self.dilation, tuple) else self.dilation
            )
            import math

            h_out = (
                math.floor((data.shape[1] + 2 * p_h - d_h * (k_h - 1) - 1) / s_h) + 1
            )
            w_out = (
                math.floor((data.shape[2] + 2 * p_w - d_w * (k_w - 1) - 1) / s_w) + 1
            )
            res = np.zeros((data.shape[0], h_out, w_out, self.out_channels))
            return _wrap(_to_tensor(res))
        return x


class Conv3d(Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple[int, int, int]],
        stride: Union[int, Tuple[int, int, int]] = 1,
        padding: Union[int, Tuple[int, int, int]] = 0,
        dilation: Union[int, Tuple[int, int, int]] = 1,
        groups: int = 1,
        bias: bool = True,
    ) -> None:
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation

    def __call__(self, x: array) -> array:
        from ml_switcheroo.core.config import config

        if config.eager_mode:
            data = np.array(_to_tensor(x).data)
            k_d = (
                self.kernel_size[0]
                if isinstance(self.kernel_size, tuple)
                else self.kernel_size
            )
            k_h = (
                self.kernel_size[1]
                if isinstance(self.kernel_size, tuple)
                else self.kernel_size
            )
            k_w = (
                self.kernel_size[2]
                if isinstance(self.kernel_size, tuple)
                else self.kernel_size
            )
            s_d = self.stride[0] if isinstance(self.stride, tuple) else self.stride
            s_h = self.stride[1] if isinstance(self.stride, tuple) else self.stride
            s_w = self.stride[2] if isinstance(self.stride, tuple) else self.stride
            p_d = self.padding[0] if isinstance(self.padding, tuple) else self.padding
            p_h = self.padding[1] if isinstance(self.padding, tuple) else self.padding
            p_w = self.padding[2] if isinstance(self.padding, tuple) else self.padding
            d_d = (
                self.dilation[0] if isinstance(self.dilation, tuple) else self.dilation
            )
            d_h = (
                self.dilation[1] if isinstance(self.dilation, tuple) else self.dilation
            )
            d_w = (
                self.dilation[2] if isinstance(self.dilation, tuple) else self.dilation
            )
            import math

            d_out = (
                math.floor((data.shape[1] + 2 * p_d - d_d * (k_d - 1) - 1) / s_d) + 1
            )
            h_out = (
                math.floor((data.shape[2] + 2 * p_h - d_h * (k_h - 1) - 1) / s_h) + 1
            )
            w_out = (
                math.floor((data.shape[3] + 2 * p_w - d_w * (k_w - 1) - 1) / s_w) + 1
            )
            res = np.zeros((data.shape[0], d_out, h_out, w_out, self.out_channels))
            return _wrap(_to_tensor(res))
        return x


class ConvTranspose1d(Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple[int]],
        stride: Union[int, Tuple[int]] = 1,
        padding: Union[int, Tuple[int]] = 0,
        dilation: Union[int, Tuple[int]] = 1,
        groups: int = 1,
        bias: bool = True,
    ) -> None:
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation

    def __call__(self, x: array) -> array:
        from ml_switcheroo.core.config import config

        if config.eager_mode:
            data = np.array(_to_tensor(x).data)
            k = (
                self.kernel_size[0]
                if isinstance(self.kernel_size, tuple)
                else self.kernel_size
            )
            s = self.stride[0] if isinstance(self.stride, tuple) else self.stride
            p = self.padding[0] if isinstance(self.padding, tuple) else self.padding
            d = self.dilation[0] if isinstance(self.dilation, tuple) else self.dilation
            l_out = (data.shape[1] - 1) * s - 2 * p + d * (k - 1) + 1
            res = np.zeros((data.shape[0], l_out, self.out_channels))
            return _wrap(_to_tensor(res))
        return x


class ConvTranspose2d(Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple[int, int]],
        stride: Union[int, Tuple[int, int]] = 1,
        padding: Union[int, Tuple[int, int]] = 0,
        dilation: Union[int, Tuple[int, int]] = 1,
        groups: int = 1,
        bias: bool = True,
    ) -> None:
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation

    def __call__(self, x: array) -> array:
        from ml_switcheroo.core.config import config

        if config.eager_mode:
            data = np.array(_to_tensor(x).data)
            k_h = (
                self.kernel_size[0]
                if isinstance(self.kernel_size, tuple)
                else self.kernel_size
            )
            k_w = (
                self.kernel_size[1]
                if isinstance(self.kernel_size, tuple)
                else self.kernel_size
            )
            s_h = self.stride[0] if isinstance(self.stride, tuple) else self.stride
            s_w = self.stride[1] if isinstance(self.stride, tuple) else self.stride
            p_h = self.padding[0] if isinstance(self.padding, tuple) else self.padding
            p_w = self.padding[1] if isinstance(self.padding, tuple) else self.padding
            d_h = (
                self.dilation[0] if isinstance(self.dilation, tuple) else self.dilation
            )
            d_w = (
                self.dilation[1] if isinstance(self.dilation, tuple) else self.dilation
            )
            h_out = (data.shape[1] - 1) * s_h - 2 * p_h + d_h * (k_h - 1) + 1
            w_out = (data.shape[2] - 1) * s_w - 2 * p_w + d_w * (k_w - 1) + 1
            res = np.zeros((data.shape[0], h_out, w_out, self.out_channels))
            return _wrap(_to_tensor(res))
        return x


class ConvTranspose3d(Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple[int, int, int]],
        stride: Union[int, Tuple[int, int, int]] = 1,
        padding: Union[int, Tuple[int, int, int]] = 0,
        dilation: Union[int, Tuple[int, int, int]] = 1,
        groups: int = 1,
        bias: bool = True,
    ) -> None:
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation

    def __call__(self, x: array) -> array:
        from ml_switcheroo.core.config import config

        if config.eager_mode:
            data = np.array(_to_tensor(x).data)
            k_d = (
                self.kernel_size[0]
                if isinstance(self.kernel_size, tuple)
                else self.kernel_size
            )
            k_h = (
                self.kernel_size[1]
                if isinstance(self.kernel_size, tuple)
                else self.kernel_size
            )
            k_w = (
                self.kernel_size[2]
                if isinstance(self.kernel_size, tuple)
                else self.kernel_size
            )
            s_d = self.stride[0] if isinstance(self.stride, tuple) else self.stride
            s_h = self.stride[1] if isinstance(self.stride, tuple) else self.stride
            s_w = self.stride[2] if isinstance(self.stride, tuple) else self.stride
            p_d = self.padding[0] if isinstance(self.padding, tuple) else self.padding
            p_h = self.padding[1] if isinstance(self.padding, tuple) else self.padding
            p_w = self.padding[2] if isinstance(self.padding, tuple) else self.padding
            d_d = (
                self.dilation[0] if isinstance(self.dilation, tuple) else self.dilation
            )
            d_h = (
                self.dilation[1] if isinstance(self.dilation, tuple) else self.dilation
            )
            d_w = (
                self.dilation[2] if isinstance(self.dilation, tuple) else self.dilation
            )
            d_out = (data.shape[1] - 1) * s_d - 2 * p_d + d_d * (k_d - 1) + 1
            h_out = (data.shape[2] - 1) * s_h - 2 * p_h + d_h * (k_h - 1) + 1
            w_out = (data.shape[3] - 1) * s_w - 2 * p_w + d_w * (k_w - 1) + 1
            res = np.zeros((data.shape[0], d_out, h_out, w_out, self.out_channels))
            return _wrap(_to_tensor(res))
        return x


class Upsample(Module):
    def __init__(
        self,
        scale_factor: Union[float, Tuple[float, ...]],
        mode: str = "nearest",
        align_corners: Optional[bool] = None,
    ) -> None:
        self.scale_factor = scale_factor

    def __call__(self, x: array) -> array:
        from ml_switcheroo.core.config import config

        if config.eager_mode:
            data = np.array(_to_tensor(x).data)
            if isinstance(self.scale_factor, tuple):
                s1, s2 = self.scale_factor
                res = np.zeros(
                    (
                        data.shape[0],
                        int(data.shape[1] * s1),
                        int(data.shape[2] * s2),
                        data.shape[3],
                    )
                )
            else:
                s = self.scale_factor
                if len(data.shape) == 3:
                    res = np.zeros(
                        (data.shape[0], int(data.shape[1] * s), data.shape[2])
                    )
                else:
                    res = np.zeros(
                        (
                            data.shape[0],
                            int(data.shape[1] * s),
                            int(data.shape[2] * s),
                            data.shape[3],
                        )
                    )
            return _wrap(_to_tensor(res))
        return x
