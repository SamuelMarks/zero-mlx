"""mlx.nn.convolution module."""

import math
from typing import Optional, Union, Tuple
from zero_mlx.nn.base import Module
from zero_mlx.array import array
from zero_mlx.mlx_random import uniform
import ml_switcheroo_compiler.ops as sops


class Conv1d(Module):
    """Applies a 1D convolution over an input signal."""

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
    ):
        """Initialize Conv1d."""
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = (
            kernel_size if isinstance(kernel_size, tuple) else (kernel_size,)
        )
        self.stride = stride if isinstance(stride, tuple) else (stride,)
        self.padding = padding if isinstance(padding, tuple) else (padding,)
        self.dilation = dilation if isinstance(dilation, tuple) else (dilation,)
        self.groups = groups

        # MLX Conv1d weight shape: (out_channels, kernel_size, in_channels // groups)
        # However, for 1D, mlx often uses shape (out_channels, k_w, in_channels // groups)
        k_w = self.kernel_size[0]
        in_ch_per_group = in_channels // groups
        scale = math.sqrt(1.0 / (in_ch_per_group * k_w))
        self.weight = uniform(
            low=-scale, high=scale, shape=(out_channels, k_w, in_ch_per_group)
        )
        if bias:
            self.bias = uniform(low=-scale, high=scale, shape=(out_channels,))
        else:
            self.bias = None

    def __call__(self, x: array) -> array:
        """Call."""
        x_t = x._tensor if hasattr(x, "_tensor") else x
        w_t = self.weight._tensor if hasattr(self.weight, "_tensor") else self.weight

        # ml_switcheroo_compiler conv1d likely expects certain kwargs
        out = sops.conv1d(
            x_t,
            w_t,
            strides=self.stride,
            padding=self.padding,
            lhs_dilation=self.dilation,
            feature_group_count=self.groups,
        )
        if getattr(self, "bias", None) is not None:
            b_t = self.bias._tensor if hasattr(self.bias, "_tensor") else self.bias
            out = sops.add(out, b_t)
        return array(out)


class Conv2d(Module):
    """Applies a 2D convolution over an input signal."""

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
    ):
        """Initialize Conv2d."""
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = (
            kernel_size
            if isinstance(kernel_size, tuple)
            else (kernel_size, kernel_size)
        )
        self.stride = stride if isinstance(stride, tuple) else (stride, stride)
        self.padding = padding if isinstance(padding, tuple) else (padding, padding)
        self.dilation = (
            dilation if isinstance(dilation, tuple) else (dilation, dilation)
        )
        self.groups = groups

        k_h, k_w = self.kernel_size
        in_ch_per_group = in_channels // groups
        scale = math.sqrt(1.0 / (in_ch_per_group * k_h * k_w))
        self.weight = uniform(
            low=-scale, high=scale, shape=(out_channels, k_h, k_w, in_ch_per_group)
        )
        if bias:
            self.bias = uniform(low=-scale, high=scale, shape=(out_channels,))
        else:
            self.bias = None

    def __call__(self, x: array) -> array:
        """Call."""
        x_t = x._tensor if hasattr(x, "_tensor") else x
        w_t = self.weight._tensor if hasattr(self.weight, "_tensor") else self.weight

        out = sops.conv2d(
            x_t,
            w_t,
            strides=self.stride,
            padding=self.padding,
            lhs_dilation=self.dilation,
            feature_group_count=self.groups,
        )
        if getattr(self, "bias", None) is not None:
            b_t = self.bias._tensor if hasattr(self.bias, "_tensor") else self.bias
            out = sops.add(out, b_t)
        return array(out)


class Conv3d(Module):
    """Applies a 3D convolution over an input signal."""

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
    ):
        """Initialize Conv3d."""
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = (
            kernel_size
            if isinstance(kernel_size, tuple)
            else (kernel_size, kernel_size, kernel_size)
        )
        self.stride = stride if isinstance(stride, tuple) else (stride, stride, stride)
        self.padding = (
            padding if isinstance(padding, tuple) else (padding, padding, padding)
        )
        self.dilation = (
            dilation if isinstance(dilation, tuple) else (dilation, dilation, dilation)
        )
        self.groups = groups

        k_d, k_h, k_w = self.kernel_size
        in_ch_per_group = in_channels // groups
        scale = math.sqrt(1.0 / (in_ch_per_group * k_d * k_h * k_w))
        self.weight = uniform(
            low=-scale, high=scale, shape=(out_channels, k_d, k_h, k_w, in_ch_per_group)
        )
        if bias:
            self.bias = uniform(low=-scale, high=scale, shape=(out_channels,))
        else:
            self.bias = None

    def __call__(self, x: array) -> array:
        """Call."""
        x_t = x._tensor if hasattr(x, "_tensor") else x
        w_t = self.weight._tensor if hasattr(self.weight, "_tensor") else self.weight

        out = sops.conv3d(
            x_t,
            w_t,
            strides=self.stride,
            padding=self.padding,
            lhs_dilation=self.dilation,
            feature_group_count=self.groups,
        )
        if getattr(self, "bias", None) is not None:
            b_t = self.bias._tensor if hasattr(self.bias, "_tensor") else self.bias
            out = sops.add(out, b_t)
        return array(out)


__all__ = ["Conv1d", "Conv2d", "Conv3d"]
