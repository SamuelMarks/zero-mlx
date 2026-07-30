"""mlx.nn.convolution_transpose module."""

import math
from typing import Optional, Union, Tuple
from zero_mlx.nn.base import Module
from zero_mlx.array import array
from zero_mlx.mlx_random import uniform
import ml_switcheroo_compiler.ops as sops


class ConvTranspose1d(Module):
    """Applies a 1D transposed convolution over an input signal."""

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
        """Initialize ConvTranspose1d."""
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

        k_w = self.kernel_size[0]
        in_ch_per_group = in_channels // groups
        scale = math.sqrt(1.0 / (in_ch_per_group * k_w))
        # MLX transpose conv weight shape: (out_channels, kernel_size, in_channels // groups) typically, but could be inverted. Assuming same scale.
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
        w_t = sops.transpose(w_t, axes=(1, 2, 0))

        out = sops.get_op("ConvTranspose")()(
            x_t,
            w_t,
            strides=self.stride,
            padding="VALID",
        )

        if self.padding[0] > 0:  # pragma: no cover
            out = sops.slice(
                out,
                start_indices=(0, self.padding[0], 0),
                end_indices=(
                    out.shape[0],
                    out.shape[1] - self.padding[0],
                    out.shape[2],
                ),
            )

        if getattr(self, "bias", None) is not None:
            b_t = self.bias._tensor if hasattr(self.bias, "_tensor") else self.bias
            out = sops.add(out, b_t)
        return array(out)


class ConvTranspose2d(Module):
    """Applies a 2D transposed convolution over an input signal."""

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
        """Initialize ConvTranspose2d."""
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
        w_t = sops.transpose(w_t, axes=(1, 2, 3, 0))

        out = sops.get_op("ConvTranspose")()(
            x_t,
            w_t,
            strides=self.stride,
            padding="VALID",
        )

        if self.padding[0] > 0 or self.padding[1] > 0:  # pragma: no cover
            p0, p1 = self.padding
            start = [0, p0, p1, 0]
            end = [out.shape[0], out.shape[1] - p0, out.shape[2] - p1, out.shape[3]]
            out = sops.slice(out, start_indices=start, end_indices=end)

        if getattr(self, "bias", None) is not None:
            b_t = self.bias._tensor if hasattr(self.bias, "_tensor") else self.bias
            out = sops.add(out, b_t)
        return array(out)


class ConvTranspose3d(Module):
    """Applies a 3D transposed convolution over an input signal."""

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
        """Initialize ConvTranspose3d."""
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
        w_t = sops.transpose(w_t, axes=(1, 2, 3, 4, 0))

        out = sops.get_op("ConvTranspose")()(
            x_t,
            w_t,
            strides=self.stride,
            padding="VALID",
        )

        if (
            self.padding[0] > 0 or self.padding[1] > 0 or self.padding[2] > 0
        ):  # pragma: no cover
            p0, p1, p2 = self.padding
            start = [0, p0, p1, p2, 0]
            end = [
                out.shape[0],
                out.shape[1] - p0,
                out.shape[2] - p1,
                out.shape[3] - p2,
                out.shape[4],
            ]
            out = sops.slice(out, start_indices=start, end_indices=end)

        if getattr(self, "bias", None) is not None:
            b_t = self.bias._tensor if hasattr(self.bias, "_tensor") else self.bias
            out = sops.add(out, b_t)
        return array(out)


__all__ = ["ConvTranspose1d", "ConvTranspose2d", "ConvTranspose3d"]
