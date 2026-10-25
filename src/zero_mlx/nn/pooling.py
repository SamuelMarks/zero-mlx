"""1D, 2D, and 3D Max and Average Pooling layers (`mlx.nn.pooling`) for zero_mlx."""

from __future__ import annotations

import builtins

from zero_mlx.array import array
from zero_mlx.nn.base import Module
import zero_mlx.ops as mx


def _pool1d(
    x: array,
    kernel_size: int,
    stride: int,
    padding: int,
    mode: str,
) -> array:
    """Apply 1D spatial pooling (`max` or `avg`) over `(N, L, C)` or `(L, C)`.

    Args:
        x: Input `array` of shape `(N, L, C)` or `(L, C)`.
        kernel_size: Window size along `L`.
        stride: Window stride along `L`.
        padding: Zero/negative-inf padding added to both ends of `L`.
        mode: `'max'` or `'avg'`.

    Returns:
        Pooled `array`.

    """
    unbatched: bool = x.ndim == 2
    inp: array = mx.expand_dims(x, axis=0) if unbatched else x
    if padding > 0:
        pad_val = float("-inf") if mode == "max" else 0.0
        inp = mx.pad(inp, ((0, 0), (padding, padding), (0, 0)), constant_values=pad_val)
    l_in = inp.shape[1]
    out_l = (l_in - kernel_size) // stride + 1
    span_l = (out_l - 1) * stride + 1
    views = [inp[:, kl : span_l + kl : stride, :] for kl in range(kernel_size)]
    stacked = mx.stack(views, axis=0)
    out = mx.max(stacked, axis=0) if mode == "max" else mx.mean(stacked, axis=0)
    return mx.squeeze(out, axis=0) if unbatched else out


def _pool2d(
    x: array,
    kernel_size: tuple[int, int],
    stride: tuple[int, int],
    padding: tuple[int, int],
    mode: str,
) -> array:
    """Apply 2D spatial pooling (`max` or `avg`) over `(N, H, W, C)` or `(H, W, C)`.

    Args:
        x: Input `array`.
        kernel_size: `(kH, kW)` window size.
        stride: `(sH, sW)` window stride.
        padding: `(pH, pW)` spatial padding.
        mode: `'max'` or `'avg'`.

    Returns:
        Pooled `array`.

    """
    unbatched: bool = x.ndim == 3
    inp: array = mx.expand_dims(x, axis=0) if unbatched else x
    ph, pw = padding
    if ph > 0 or pw > 0:
        pad_val = float("-inf") if mode == "max" else 0.0
        inp = mx.pad(
            inp,
            ((0, 0), (ph, ph), (pw, pw), (0, 0)),
            constant_values=pad_val,
        )
    kh, kw = kernel_size
    sh, sw = stride
    h_in, w_in = inp.shape[1], inp.shape[2]
    out_h = (h_in - kh) // sh + 1
    out_w = (w_in - kw) // sw + 1
    span_h = (out_h - 1) * sh + 1
    span_w = (out_w - 1) * sw + 1
    views = [
        inp[:, dh : span_h + dh : sh, dw : span_w + dw : sw, :]
        for dh in range(kh)
        for dw in range(kw)
    ]
    stacked = mx.stack(views, axis=0)
    out = mx.max(stacked, axis=0) if mode == "max" else mx.mean(stacked, axis=0)
    return mx.squeeze(out, axis=0) if unbatched else out


def _pool3d(
    x: array,
    kernel_size: tuple[int, int, int],
    stride: tuple[int, int, int],
    padding: tuple[int, int, int],
    mode: str,
) -> array:
    """Apply 3D spatial pooling (`max` or `avg`) over `(N, D, H, W, C)` or `(D, H, W, C)`.

    Args:
        x: Input `array`.
        kernel_size: `(kD, kH, kW)` window size.
        stride: `(sD, sH, sW)` window stride.
        padding: `(pD, pH, pW)` spatial padding.
        mode: `'max'` or `'avg'`.

    Returns:
        Pooled `array`.

    """
    unbatched: bool = x.ndim == 4
    inp: array = mx.expand_dims(x, axis=0) if unbatched else x
    pd, ph, pw = padding
    if pd > 0 or ph > 0 or pw > 0:
        pad_val = float("-inf") if mode == "max" else 0.0
        inp = mx.pad(
            inp,
            ((0, 0), (pd, pd), (ph, ph), (pw, pw), (0, 0)),
            constant_values=pad_val,
        )
    kd, kh, kw = kernel_size
    sd, sh, sw = stride
    d_in, h_in, w_in = inp.shape[1], inp.shape[2], inp.shape[3]
    out_d = (d_in - kd) // sd + 1
    out_h = (h_in - kh) // sh + 1
    out_w = (w_in - kw) // sw + 1
    span_d = (out_d - 1) * sd + 1
    span_h = (out_h - 1) * sh + 1
    span_w = (out_w - 1) * sw + 1
    views = [
        inp[:, dd : span_d + dd : sd, dh : span_h + dh : sh, dw : span_w + dw : sw, :]
        for dd in range(kd)
        for dh in range(kh)
        for dw in range(kw)
    ]
    stacked = mx.stack(views, axis=0)
    out = mx.max(stacked, axis=0) if mode == "max" else mx.mean(stacked, axis=0)
    return mx.squeeze(out, axis=0) if unbatched else out


class MaxPool1d(Module):
    """Applies a 1D max pooling over an input signal `(N, L, C)`."""

    def __init__(
        self,
        kernel_size: int | tuple[int],
        stride: int | tuple[int] | None = None,
        padding: int | tuple[int] = 0,
    ) -> None:
        """Initialize `MaxPool1d`.

        Args:
            kernel_size: Size of the pooling window.
            stride: Stride of the pooling window (defaults to `kernel_size`).
            padding: Zero-padding added to both sides.

        """
        super().__init__()
        self.kernel_size: tuple[int] = (
            kernel_size if isinstance(kernel_size, tuple) else (kernel_size,)
        )
        self.stride: tuple[int] = (
            stride
            if isinstance(stride, tuple)
            else ((stride,) if stride is not None else self.kernel_size)
        )
        self.padding: tuple[int] = padding if isinstance(padding, tuple) else (padding,)

    def __call__(self, x: array) -> array:
        """Apply 1D max pooling to `x`.

        Args:
            x: Input `array`.

        Returns:
            Pooled `array`.

        """
        return _pool1d(x, self.kernel_size[0], self.stride[0], self.padding[0], mode="max")


class MaxPool2d(Module):
    """Applies a 2D max pooling over an input signal `(N, H, W, C)`."""

    def __init__(
        self,
        kernel_size: int | tuple[int, int],
        stride: int | tuple[int, int] | None = None,
        padding: int | tuple[int, int] = 0,
    ) -> None:
        """Initialize `MaxPool2d`.

        Args:
            kernel_size: Size of the 2D pooling window.
            stride: Stride of the pooling window (defaults to `kernel_size`).
            padding: Implicit padding on both sides.

        """
        super().__init__()
        self.kernel_size: tuple[int, int] = (
            kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
        )
        self.stride: tuple[int, int] = (
            stride
            if isinstance(stride, tuple)
            else ((stride, stride) if stride is not None else self.kernel_size)
        )
        self.padding: tuple[int, int] = (
            padding if isinstance(padding, tuple) else (padding, padding)
        )

    def __call__(self, x: array) -> array:
        """Apply 2D max pooling to `x`.

        Args:
            x: Input `array`.

        Returns:
            Pooled `array`.

        """
        return _pool2d(x, self.kernel_size, self.stride, self.padding, mode="max")


class MaxPool3d(Module):
    """Applies a 3D max pooling over an input signal `(N, D, H, W, C)`."""

    def __init__(
        self,
        kernel_size: int | tuple[int, int, int],
        stride: int | tuple[int, int, int] | None = None,
        padding: int | tuple[int, int, int] = 0,
    ) -> None:
        """Initialize `MaxPool3d`.

        Args:
            kernel_size: Size of the 3D pooling window.
            stride: Stride of the pooling window (defaults to `kernel_size`).
            padding: Implicit padding on all sides.

        """
        super().__init__()
        self.kernel_size: tuple[int, int, int] = (
            kernel_size
            if isinstance(kernel_size, tuple)
            else (kernel_size, kernel_size, kernel_size)
        )
        self.stride: tuple[int, int, int] = (
            stride
            if isinstance(stride, tuple)
            else ((stride, stride, stride) if stride is not None else self.kernel_size)
        )
        self.padding: tuple[int, int, int] = (
            padding if isinstance(padding, tuple) else (padding, padding, padding)
        )

    def __call__(self, x: array) -> array:
        """Apply 3D max pooling to `x`.

        Args:
            x: Input `array`.

        Returns:
            Pooled `array`.

        """
        return _pool3d(x, self.kernel_size, self.stride, self.padding, mode="max")


class AvgPool1d(Module):
    """Applies a 1D average pooling over an input signal `(N, L, C)`."""

    def __init__(
        self,
        kernel_size: int | tuple[int],
        stride: int | tuple[int] | None = None,
        padding: int | tuple[int] = 0,
    ) -> None:
        """Initialize `AvgPool1d`.

        Args:
            kernel_size: Size of the pooling window.
            stride: Stride of the pooling window (defaults to `kernel_size`).
            padding: Zero-padding added to both sides.

        """
        super().__init__()
        self.kernel_size: tuple[int] = (
            kernel_size if isinstance(kernel_size, tuple) else (kernel_size,)
        )
        self.stride: tuple[int] = (
            stride
            if isinstance(stride, tuple)
            else ((stride,) if stride is not None else self.kernel_size)
        )
        self.padding: tuple[int] = padding if isinstance(padding, tuple) else (padding,)

    def __call__(self, x: array) -> array:
        """Apply 1D average pooling to `x`.

        Args:
            x: Input `array`.

        Returns:
            Pooled `array`.

        """
        return _pool1d(x, self.kernel_size[0], self.stride[0], self.padding[0], mode="avg")


class AvgPool2d(Module):
    """Applies a 2D average pooling over an input signal `(N, H, W, C)`."""

    def __init__(
        self,
        kernel_size: int | tuple[int, int],
        stride: int | tuple[int, int] | None = None,
        padding: int | tuple[int, int] = 0,
    ) -> None:
        """Initialize `AvgPool2d`.

        Args:
            kernel_size: Size of the 2D pooling window.
            stride: Stride of the pooling window (defaults to `kernel_size`).
            padding: Zero-padding added to both sides.

        """
        super().__init__()
        self.kernel_size: tuple[int, int] = (
            kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
        )
        self.stride: tuple[int, int] = (
            stride
            if isinstance(stride, tuple)
            else ((stride, stride) if stride is not None else self.kernel_size)
        )
        self.padding: tuple[int, int] = (
            padding if isinstance(padding, tuple) else (padding, padding)
        )

    def __call__(self, x: array) -> array:
        """Apply 2D average pooling to `x`.

        Args:
            x: Input `array`.

        Returns:
            Pooled `array`.

        """
        return _pool2d(x, self.kernel_size, self.stride, self.padding, mode="avg")


class AvgPool3d(Module):
    """Applies a 3D average pooling over an input signal `(N, D, H, W, C)`."""

    def __init__(
        self,
        kernel_size: int | tuple[int, int, int],
        stride: int | tuple[int, int, int] | None = None,
        padding: int | tuple[int, int, int] = 0,
    ) -> None:
        """Initialize `AvgPool3d`.

        Args:
            kernel_size: Size of the 3D pooling window.
            stride: Stride of the pooling window (defaults to `kernel_size`).
            padding: Zero-padding added to all sides.

        """
        super().__init__()
        self.kernel_size: tuple[int, int, int] = (
            kernel_size
            if isinstance(kernel_size, tuple)
            else (kernel_size, kernel_size, kernel_size)
        )
        self.stride: tuple[int, int, int] = (
            stride
            if isinstance(stride, tuple)
            else ((stride, stride, stride) if stride is not None else self.kernel_size)
        )
        self.padding: tuple[int, int, int] = (
            padding if isinstance(padding, tuple) else (padding, padding, padding)
        )

    def __call__(self, x: array) -> array:
        """Apply 3D average pooling to `x`.

        Args:
            x: Input `array`.

        Returns:
            Pooled `array`.

        """
        return _pool3d(x, self.kernel_size, self.stride, self.padding, mode="avg")


__all__ = [
    "AvgPool1d",
    "AvgPool2d",
    "AvgPool3d",
    "MaxPool1d",
    "MaxPool2d",
    "MaxPool3d",
]
