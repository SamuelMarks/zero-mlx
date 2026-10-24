"""1D, 2D, and 3D Convolution layers (`mlx.nn.convolution`) for zero_mlx."""

from __future__ import annotations

import math

import ml_switcheroo_compiler.ops as sops
import ml_switcheroo_compiler.ops.shape.frontend as sf

from zero_mlx.array import array
from zero_mlx.mlx_random import uniform
from zero_mlx.nn.base import Module


class Conv1d(Module):
    """Applies a 1D convolution over a multi-channel input signal `(N, L, C_in)`."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int | tuple[int],
        stride: int | tuple[int] = 1,
        padding: int | tuple[int] = 0,
        dilation: int | tuple[int] = 1,
        groups: int = 1,
        bias: bool = True,
    ) -> None:
        """Initialize `Conv1d`.

        Args:
            in_channels: Number of input channels.
            out_channels: Number of output channels.
            kernel_size: Size of the 1D convolving kernel.
            stride: Stride of the convolution.
            padding: Zero-padding added to both sides of the input.
            dilation: Spacing between kernel elements.
            groups: Number of blocked connections from input channels to output channels.
            bias: If True, adds a learnable bias to the output.

        """
        super().__init__()
        self.in_channels: int = in_channels
        self.out_channels: int = out_channels
        self.kernel_size: tuple[int] = (
            kernel_size if isinstance(kernel_size, tuple) else (kernel_size,)
        )
        self.stride: tuple[int] = stride if isinstance(stride, tuple) else (stride,)
        self.padding: tuple[int] = padding if isinstance(padding, tuple) else (padding,)
        self.dilation: tuple[int] = dilation if isinstance(dilation, tuple) else (dilation,)
        self.groups: int = groups

        k_w = self.kernel_size[0]
        in_ch_per_group = in_channels // groups
        scale = math.sqrt(1.0 / float(in_ch_per_group * k_w))
        self.weight: array = uniform(
            low=-scale, high=scale, shape=(out_channels, k_w, in_ch_per_group)
        )
        self.bias: array | None = (
            uniform(low=-scale, high=scale, shape=(out_channels,)) if bias else None
        )

    def __call__(self, x: array) -> array:
        """Apply 1D convolution to `x`.

        Args:
            x: Input `array` of shape `(N, L, C_in)`.

        Returns:
            Convolved `array` of shape `(N, L_out, C_out)`.

        """
        w_t = sf.permute(self.weight._tensor, dims=(1, 2, 0))
        p0 = self.padding[0]
        out = sops.nn.conv.conv1d(
            x._tensor,
            w_t,
            strides=self.stride,
            padding=[(p0, p0)],
            lhs_dilation=self.dilation,
            rhs_dilation=1,
        )
        if self.bias is not None:
            out = sops.add(out, self.bias._tensor)
        return array(out, dtype=x.dtype)


class Conv2d(Module):
    """Applies a 2D convolution over a multi-channel input signal `(N, H, W, C_in)`."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int | tuple[int, int],
        stride: int | tuple[int, int] = 1,
        padding: int | tuple[int, int] = 0,
        dilation: int | tuple[int, int] = 1,
        groups: int = 1,
        bias: bool = True,
    ) -> None:
        """Initialize `Conv2d`.

        Args:
            in_channels: Number of input channels.
            out_channels: Number of output channels.
            kernel_size: Size of the 2D convolving kernel (`int` or 2-tuple).
            stride: Stride of the convolution (`int` or 2-tuple).
            padding: Zero-padding added to both sides of the input (`int` or 2-tuple).
            dilation: Spacing between kernel elements (`int` or 2-tuple).
            groups: Number of channel groups.
            bias: If True, adds a learnable bias to the output.

        """
        super().__init__()
        self.in_channels: int = in_channels
        self.out_channels: int = out_channels
        self.kernel_size: tuple[int, int] = (
            kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
        )
        self.stride: tuple[int, int] = stride if isinstance(stride, tuple) else (stride, stride)
        self.padding: tuple[int, int] = (
            padding if isinstance(padding, tuple) else (padding, padding)
        )
        self.dilation: tuple[int, int] = (
            dilation if isinstance(dilation, tuple) else (dilation, dilation)
        )
        self.groups: int = groups

        k_h, k_w = self.kernel_size
        in_ch_per_group = in_channels // groups
        scale = math.sqrt(1.0 / float(in_ch_per_group * k_h * k_w))
        self.weight: array = uniform(
            low=-scale, high=scale, shape=(out_channels, k_h, k_w, in_ch_per_group)
        )
        self.bias: array | None = (
            uniform(low=-scale, high=scale, shape=(out_channels,)) if bias else None
        )

    def __call__(self, x: array) -> array:
        """Apply 2D convolution to `x`.

        Args:
            x: Input `array` of shape `(N, H, W, C_in)`.

        Returns:
            Convolved `array` of shape `(N, H_out, W_out, C_out)`.

        """
        w_t = sf.permute(self.weight._tensor, dims=(1, 2, 3, 0))
        p0, p1 = self.padding
        out = sops.nn.conv.conv2d(
            x._tensor,
            w_t,
            strides=self.stride,
            padding=[(p0, p0), (p1, p1)],
            lhs_dilation=self.dilation,
            rhs_dilation=1,
        )
        if self.bias is not None:
            out = sops.add(out, self.bias._tensor)
        return array(out, dtype=x.dtype)


class Conv3d(Module):
    """Applies a 3D convolution over a multi-channel input signal `(N, D, H, W, C_in)`."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int | tuple[int, int, int],
        stride: int | tuple[int, int, int] = 1,
        padding: int | tuple[int, int, int] = 0,
        dilation: int | tuple[int, int, int] = 1,
        groups: int = 1,
        bias: bool = True,
    ) -> None:
        """Initialize `Conv3d`.

        Args:
            in_channels: Number of input channels.
            out_channels: Number of output channels.
            kernel_size: Size of the 3D convolving kernel (`int` or 3-tuple).
            stride: Stride of the convolution (`int` or 3-tuple).
            padding: Zero-padding added to both sides of the input (`int` or 3-tuple).
            dilation: Spacing between kernel elements (`int` or 3-tuple).
            groups: Number of channel groups.
            bias: If True, adds a learnable bias to the output.

        """
        super().__init__()
        self.in_channels: int = in_channels
        self.out_channels: int = out_channels
        self.kernel_size: tuple[int, int, int] = (
            kernel_size
            if isinstance(kernel_size, tuple)
            else (kernel_size, kernel_size, kernel_size)
        )
        self.stride: tuple[int, int, int] = (
            stride if isinstance(stride, tuple) else (stride, stride, stride)
        )
        self.padding: tuple[int, int, int] = (
            padding if isinstance(padding, tuple) else (padding, padding, padding)
        )
        self.dilation: tuple[int, int, int] = (
            dilation if isinstance(dilation, tuple) else (dilation, dilation, dilation)
        )
        self.groups: int = groups

        k_d, k_h, k_w = self.kernel_size
        in_ch_per_group = in_channels // groups
        scale = math.sqrt(1.0 / float(in_ch_per_group * k_d * k_h * k_w))
        self.weight: array = uniform(
            low=-scale,
            high=scale,
            shape=(out_channels, k_d, k_h, k_w, in_ch_per_group),
        )
        self.bias: array | None = (
            uniform(low=-scale, high=scale, shape=(out_channels,)) if bias else None
        )

    def __call__(self, x: array) -> array:
        """Apply 3D convolution to `x`.

        Args:
            x: Input `array` of shape `(N, D, H, W, C_in)`.

        Returns:
            Convolved `array` of shape `(N, D_out, H_out, W_out, C_out)`.

        """
        w_t = sf.permute(self.weight._tensor, dims=(1, 2, 3, 4, 0))
        p0, p1, p2 = self.padding
        out = sops.nn.conv.conv3d(
            x._tensor,
            w_t,
            strides=self.stride,
            padding=[(p0, p0), (p1, p1), (p2, p2)],
            lhs_dilation=self.dilation,
            rhs_dilation=1,
        )
        if self.bias is not None:
            out = sops.add(out, self.bias._tensor)
        return array(out, dtype=x.dtype)


__all__ = ["Conv1d", "Conv2d", "Conv3d"]
