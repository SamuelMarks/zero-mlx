"""mlx.nn.pooling module."""

from typing import Union, Tuple, Any
from zero_mlx.nn.base import Module
from zero_mlx.array import array
import ml_switcheroo_compiler.ops as sops


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
        x_t = x._tensor if hasattr(x, "_tensor") else x
        out = sops.pool1d(
            x_t,
            window_shape=self.kernel_size[0],
            strides=self.stride[0],
            padding=self.padding,
            pool_mode="max",
        )
        return array(out)


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
        x_t = x._tensor if hasattr(x, "_tensor") else x
        out = sops.pool2d(
            x_t,
            window_shape=self.kernel_size,
            strides=self.stride,
            padding=self.padding,
            pool_mode="max",
        )
        return array(out)


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
        x_t = x._tensor if hasattr(x, "_tensor") else x
        out = sops.pool3d(
            x_t,
            window_shape=self.kernel_size,
            strides=self.stride,
            padding=self.padding,
            pool_mode="max",
        )
        return array(out)


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
        x_t = x._tensor if hasattr(x, "_tensor") else x
        out = sops.pool1d(
            x_t,
            window_shape=self.kernel_size[0],
            strides=self.stride[0],
            padding=self.padding,
            pool_mode="avg",
        )
        return array(out)


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
        x_t = x._tensor if hasattr(x, "_tensor") else x
        out = sops.pool2d(
            x_t,
            window_shape=self.kernel_size,
            strides=self.stride,
            padding=self.padding,
            pool_mode="avg",
        )
        return array(out)


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
        x_t = x._tensor if hasattr(x, "_tensor") else x
        out = sops.pool3d(
            x_t,
            window_shape=self.kernel_size,
            strides=self.stride,
            padding=self.padding,
            pool_mode="avg",
        )
        return array(out)


__all__ = ["MaxPool1d", "MaxPool2d", "MaxPool3d", "AvgPool1d", "AvgPool2d", "AvgPool3d"]
