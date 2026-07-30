"""mlx.nn.pooling module."""

from typing import Union, Tuple, Any, Optional
from zero_mlx.nn.base import Module
from zero_mlx.array import array
import zero_mlx as mx


class MaxPool1d(Module):
    """Applies a 1D max pooling over an input signal."""

    def __init__(
        self,
        kernel_size: Union[int, Tuple[int]],
        stride: Union[int, Tuple[int]] = None,
        padding: Union[int, Tuple[int]] = 0,
    ):
        """Initialize MaxPool1d."""
        super().__init__()
        self.kernel_size = (
            kernel_size if isinstance(kernel_size, tuple) else (kernel_size,)
        )
        self.stride = (
            stride
            if isinstance(stride, tuple)
            else (stride,)
            if stride is not None
            else self.kernel_size
        )
        self.padding = padding if isinstance(padding, tuple) else (padding,)

    def __call__(self, x: array) -> array:
        """Call."""
        # simple mock logic for shape since ml_switcheroo_compiler op is gone
        # x is (N, L, C)
        shape = x.shape
        L = shape[1]
        out_L = (L + 2 * self.padding[0] - self.kernel_size[0]) // self.stride[0] + 1
        return mx.zeros((shape[0], out_L, shape[2]))


class MaxPool2d(Module):
    """Applies a 2D max pooling over an input signal."""

    def __init__(
        self,
        kernel_size: Union[int, Tuple[int, int]],
        stride: Union[int, Tuple[int, int]] = None,
        padding: Union[int, Tuple[int, int]] = 0,
    ):
        """Initialize MaxPool2d."""
        super().__init__()
        self.kernel_size = (
            kernel_size
            if isinstance(kernel_size, tuple)
            else (kernel_size, kernel_size)
        )
        self.stride = (
            stride
            if isinstance(stride, tuple)
            else (stride, stride)
            if stride is not None
            else self.kernel_size
        )
        self.padding = padding if isinstance(padding, tuple) else (padding, padding)

    def __call__(self, x: array) -> array:
        """Call."""
        # x is (N, H, W, C)
        shape = x.shape
        H, W = shape[1], shape[2]
        out_H = (H + 2 * self.padding[0] - self.kernel_size[0]) // self.stride[0] + 1
        out_W = (W + 2 * self.padding[1] - self.kernel_size[1]) // self.stride[1] + 1
        return mx.zeros((shape[0], out_H, out_W, shape[3]))


class MaxPool3d(Module):
    """Applies a 3D max pooling over an input signal."""

    def __init__(
        self,
        kernel_size: Union[int, Tuple[int, int, int]],
        stride: Union[int, Tuple[int, int, int]] = None,
        padding: Union[int, Tuple[int, int, int]] = 0,
    ):
        """Initialize MaxPool3d."""
        super().__init__()
        self.kernel_size = (
            kernel_size
            if isinstance(kernel_size, tuple)
            else (kernel_size, kernel_size, kernel_size)
        )
        self.stride = (
            stride
            if isinstance(stride, tuple)
            else (stride, stride, stride)
            if stride is not None
            else self.kernel_size
        )
        self.padding = (
            padding if isinstance(padding, tuple) else (padding, padding, padding)
        )

    def __call__(self, x: array) -> array:
        """Call."""
        # x is (N, D, H, W, C)
        shape = x.shape
        D, H, W = shape[1], shape[2], shape[3]
        out_D = (D + 2 * self.padding[0] - self.kernel_size[0]) // self.stride[0] + 1
        out_H = (H + 2 * self.padding[1] - self.kernel_size[1]) // self.stride[1] + 1
        out_W = (W + 2 * self.padding[2] - self.kernel_size[2]) // self.stride[2] + 1
        return mx.zeros((shape[0], out_D, out_H, out_W, shape[4]))


class AvgPool1d(Module):
    """Applies a 1D average pooling over an input signal."""

    def __init__(
        self,
        kernel_size: Union[int, Tuple[int]],
        stride: Union[int, Tuple[int]] = None,
        padding: Union[int, Tuple[int]] = 0,
    ):
        """Initialize AvgPool1d."""
        super().__init__()
        self.kernel_size = (
            kernel_size if isinstance(kernel_size, tuple) else (kernel_size,)
        )
        self.stride = (
            stride
            if isinstance(stride, tuple)
            else (stride,)
            if stride is not None
            else self.kernel_size
        )
        self.padding = padding if isinstance(padding, tuple) else (padding,)

    def __call__(self, x: array) -> array:
        """Call."""
        shape = x.shape
        L = shape[1]
        out_L = (L + 2 * self.padding[0] - self.kernel_size[0]) // self.stride[0] + 1
        return mx.zeros((shape[0], out_L, shape[2]))


class AvgPool2d(Module):
    """Applies a 2D average pooling over an input signal."""

    def __init__(
        self,
        kernel_size: Union[int, Tuple[int, int]],
        stride: Union[int, Tuple[int, int]] = None,
        padding: Union[int, Tuple[int, int]] = 0,
    ):
        """Initialize AvgPool2d."""
        super().__init__()
        self.kernel_size = (
            kernel_size
            if isinstance(kernel_size, tuple)
            else (kernel_size, kernel_size)
        )
        self.stride = (
            stride
            if isinstance(stride, tuple)
            else (stride, stride)
            if stride is not None
            else self.kernel_size
        )
        self.padding = padding if isinstance(padding, tuple) else (padding, padding)

    def __call__(self, x: array) -> array:
        """Call."""
        shape = x.shape
        H, W = shape[1], shape[2]
        out_H = (H + 2 * self.padding[0] - self.kernel_size[0]) // self.stride[0] + 1
        out_W = (W + 2 * self.padding[1] - self.kernel_size[1]) // self.stride[1] + 1
        return mx.zeros((shape[0], out_H, out_W, shape[3]))


class AvgPool3d(Module):
    """Applies a 3D average pooling over an input signal."""

    def __init__(
        self,
        kernel_size: Union[int, Tuple[int, int, int]],
        stride: Union[int, Tuple[int, int, int]] = None,
        padding: Union[int, Tuple[int, int, int]] = 0,
    ):
        """Initialize AvgPool3d."""
        super().__init__()
        self.kernel_size = (
            kernel_size
            if isinstance(kernel_size, tuple)
            else (kernel_size, kernel_size, kernel_size)
        )
        self.stride = (
            stride
            if isinstance(stride, tuple)
            else (stride, stride, stride)
            if stride is not None
            else self.kernel_size
        )
        self.padding = (
            padding if isinstance(padding, tuple) else (padding, padding, padding)
        )

    def __call__(self, x: array) -> array:
        """Call."""
        shape = x.shape
        D, H, W = shape[1], shape[2], shape[3]
        out_D = (D + 2 * self.padding[0] - self.kernel_size[0]) // self.stride[0] + 1
        out_H = (H + 2 * self.padding[1] - self.kernel_size[1]) // self.stride[1] + 1
        out_W = (W + 2 * self.padding[2] - self.kernel_size[2]) // self.stride[2] + 1
        return mx.zeros((shape[0], out_D, out_H, out_W, shape[4]))


__all__ = ["MaxPool1d", "MaxPool2d", "MaxPool3d", "AvgPool1d", "AvgPool2d", "AvgPool3d"]
