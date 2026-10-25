"""1D, 2D, and 3D Transposed Convolution layers (`mlx.nn.convolution_transpose`) for zero_mlx."""

from __future__ import annotations

import math

from zero_mlx.array import array
from zero_mlx.convolutions import conv_transpose1d, conv_transpose2d, conv_transpose3d
from zero_mlx.mlx_random import uniform
from zero_mlx.nn.base import Module


class ConvTranspose1d(Module):
    """Applies a 1D transposed convolution over an input signal `(N, L, C_in)`."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int | tuple[int],
        stride: int | tuple[int] = 1,
        padding: int | tuple[int] = 0,
        dilation: int | tuple[int] = 1,
        output_padding: int | tuple[int] = 0,
        groups: int = 1,
        bias: bool = True,
    ) -> None:
        """Initialize `ConvTranspose1d`.

        Args:
            in_channels: Number of input channels.
            out_channels: Number of output channels.
            kernel_size: Size of the convolving kernel.
            stride: Stride of the convolution.
            padding: Input zero-padding.
            dilation: Kernel dilation spacing.
            output_padding: Additional size added to one side of the output shape.
            groups: Number of blocked connections from input to output channels.
            bias: Whether to include a learnable bias.

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
        self.output_padding: tuple[int] = (
            output_padding if isinstance(output_padding, tuple) else (output_padding,)
        )
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
        """Apply 1D transposed convolution to `x`.

        Args:
            x: Input `array` of shape `(N, L, C_in)`.

        Returns:
            Output `array` of shape `(N, L_out, C_out)`.

        """
        out = conv_transpose1d(
            x,
            self.weight,
            stride=self.stride[0],
            padding=self.padding[0],
            dilation=self.dilation[0],
            output_padding=self.output_padding[0],
            groups=self.groups,
        )
        return out + self.bias if self.bias is not None else out


class ConvTranspose2d(Module):
    """Applies a 2D transposed convolution over an input signal `(N, H, W, C_in)`."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int | tuple[int, int],
        stride: int | tuple[int, int] = 1,
        padding: int | tuple[int, int] = 0,
        dilation: int | tuple[int, int] = 1,
        output_padding: int | tuple[int, int] = 0,
        groups: int = 1,
        bias: bool = True,
    ) -> None:
        """Initialize `ConvTranspose2d`.

        Args:
            in_channels: Number of input channels.
            out_channels: Number of output channels.
            kernel_size: Size of the 2D kernel (`int` or 2-tuple).
            stride: Convolution stride (`int` or 2-tuple).
            padding: Input zero-padding (`int` or 2-tuple).
            dilation: Kernel dilation (`int` or 2-tuple).
            output_padding: Additional output size (`int` or 2-tuple).
            groups: Number of channel groups.
            bias: Whether to include a learnable bias.

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
        self.output_padding: tuple[int, int] = (
            output_padding
            if isinstance(output_padding, tuple)
            else (output_padding, output_padding)
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
        """Apply 2D transposed convolution to `x`.

        Args:
            x: Input `array` of shape `(N, H, W, C_in)`.

        Returns:
            Output `array` of shape `(N, H_out, W_out, C_out)`.

        """
        out = conv_transpose2d(
            x,
            self.weight,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            output_padding=self.output_padding,
            groups=self.groups,
        )
        return out + self.bias if self.bias is not None else out


class ConvTranspose3d(Module):
    """Applies a 3D transposed convolution over an input signal `(N, D, H, W, C_in)`."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int | tuple[int, int, int],
        stride: int | tuple[int, int, int] = 1,
        padding: int | tuple[int, int, int] = 0,
        dilation: int | tuple[int, int, int] = 1,
        output_padding: int | tuple[int, int, int] = 0,
        groups: int = 1,
        bias: bool = True,
    ) -> None:
        """Initialize `ConvTranspose3d`.

        Args:
            in_channels: Number of input channels.
            out_channels: Number of output channels.
            kernel_size: Size of the 3D kernel (`int` or 3-tuple).
            stride: Convolution stride (`int` or 3-tuple).
            padding: Input zero-padding (`int` or 3-tuple).
            dilation: Kernel dilation (`int` or 3-tuple).
            output_padding: Additional output size (`int` or 3-tuple).
            groups: Number of channel groups.
            bias: Whether to include a learnable bias.

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
        self.output_padding: tuple[int, int, int] = (
            output_padding
            if isinstance(output_padding, tuple)
            else (output_padding, output_padding, output_padding)
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
        """Apply 3D transposed convolution to `x`.

        Args:
            x: Input `array` of shape `(N, D, H, W, C_in)`.

        Returns:
            Output `array` of shape `(N, D_out, H_out, W_out, C_out)`.

        """
        out = conv_transpose3d(
            x,
            self.weight,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            output_padding=self.output_padding,
            groups=self.groups,
        )
        return out + self.bias if self.bias is not None else out


__all__ = ["ConvTranspose1d", "ConvTranspose2d", "ConvTranspose3d"]
