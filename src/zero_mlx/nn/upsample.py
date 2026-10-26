"""Spatial upsampling layer (`Upsample`) for `mlx.nn`."""

from __future__ import annotations

from typing import Sequence

import ml_switcheroo_compiler.ops as sops

from zero_mlx.array import array
from zero_mlx.dtypes import float32, int32
from zero_mlx.nn.base import Module
import zero_mlx.ops as mx


def _interp_1d_axis(
    x: array,
    n_out: int,
    axis: int,
    mode: str,
    align_corners: bool,
) -> array:
    """Interpolate `x` along spatial dimension `axis` to length `n_out`.

    Args:
        x: Input `array`.
        n_out: Target size along `axis`.
        axis: Spatial dimension index (`1 <= axis < x.ndim - 1`).
        mode: `'linear'` or `'cubic'`.
        align_corners: Whether to align corner pixel centers.

    Returns:
        Interpolated `array`.

    """
    n_in = x.shape[axis]
    if n_in == n_out:
        return x
    if n_in == 1:
        return mx.repeat(x, n_out, axis=axis)

    idx_out = mx.arange(n_out, dtype=float32)
    if mode == "nearest":
        nearest_idx = mx.clip(
            mx.floor(idx_out * (float(n_in) / float(n_out))).astype(int32),
            0,
            n_in - 1,
        )
        return mx.take(x, nearest_idx, axis=axis)

    src_pos = (
        idx_out * (float(n_in - 1) / float(n_out - 1))
        if (align_corners and n_out > 1)
        else ((idx_out + 0.5) * (float(n_in) / float(n_out)) - 0.5)
    )
    bcast_shape = [n_out if i == axis else 1 for i in range(x.ndim)]

    if mode == "linear":
        clamped = mx.clip(src_pos, 0.0, float(n_in - 1))
        i0 = mx.clip(mx.floor(clamped).astype(int32), 0, n_in - 1)
        i1 = mx.clip(i0 + 1, 0, n_in - 1)
        w1 = (clamped - i0.astype(float32)).reshape(tuple(bcast_shape))
        v0 = mx.take(x, i0, axis=axis)
        v1 = mx.take(x, i1, axis=axis)
        return (v0 * (1.0 - w1) + v1 * w1).astype(x.dtype)

    base = mx.floor(src_pos)
    t = (src_pos - base).reshape(tuple(bcast_shape))
    t2 = t * t
    t3 = t2 * t
    a: float = -0.75
    w_m1 = a * t3 - (2.0 * a) * t2 + a * t
    w_0 = (a + 2.0) * t3 - (a + 3.0) * t2 + 1.0
    w_1 = -(a + 2.0) * t3 + (2.0 * a + 3.0) * t2 - a * t
    w_2 = -a * t3 + a * t2

    im1 = mx.clip((base - 1.0).astype(int32), 0, n_in - 1)
    i0 = mx.clip(base.astype(int32), 0, n_in - 1)
    i1 = mx.clip((base + 1.0).astype(int32), 0, n_in - 1)
    i2 = mx.clip((base + 2.0).astype(int32), 0, n_in - 1)
    res = (
        mx.take(x, im1, axis=axis) * w_m1
        + mx.take(x, i0, axis=axis) * w_0
        + mx.take(x, i1, axis=axis) * w_1
        + mx.take(x, i2, axis=axis) * w_2
    )
    return res.astype(x.dtype)


class Upsample(Module):
    """Upsamples a multi-channel 1D, 2D, or 3D input signal."""

    def __init__(
        self,
        scale_factor: float | tuple[float, ...] | None = None,
        size: int | tuple[int, ...] | None = None,
        mode: str = "nearest",
        align_corners: bool = False,
        antialias: bool = False,
    ) -> None:
        """Initialize `Upsample`.

        Args:
            scale_factor: Multiplier for spatial size (`float` or tuple of `float`).
            size: Optional explicit output spatial size (`int` or tuple of `int`).
            mode: Interpolation mode (`'nearest'`, `'linear'`, `'cubic'`).
            align_corners: Whether to align corner pixels in linear/cubic interpolation.
            antialias: Whether to apply anti-aliasing when downsampling.

        """
        super().__init__()
        if mode not in ("nearest", "linear", "cubic"):
            raise ValueError(
                f"Unsupported Upsample mode '{mode}', expected 'nearest', 'linear', or 'cubic'"
            )
        if size is not None and scale_factor is not None:
            raise ValueError("Only one of size or scale_factor may be specified")
        self.scale_factor: float | tuple[float, ...] | None = scale_factor
        self.size: int | tuple[int, ...] | None = size
        self.mode: str = mode
        self.align_corners: bool = align_corners
        self.antialias: bool = antialias

    def __call__(self, x: array) -> array:
        """Upsample `x` according to `size` or `scale_factor`.

        Args:
            x: Input `array` of shape `(N, ..., C)`.

        Returns:
            Upsampled `array`.

        Raises:
            ValueError: If `x.ndim < 3` or `scale_factor`/`size` dimensions do not match `x.ndim - 2`.

        """
        if x.ndim < 3:
            raise ValueError(f"Upsample expects at least 3D input (N, ..., C), got {x.ndim}D")
        spatial_shape = x.shape[1:-1]
        if self.size is not None:
            target_size = (
                (self.size,) * len(spatial_shape)
                if isinstance(self.size, int)
                else tuple(int(s) for s in self.size)
            )
            if len(target_size) != len(spatial_shape):
                raise ValueError(
                    f"size length {len(target_size)} does not match spatial dims {len(spatial_shape)}"
                )
        elif self.scale_factor is not None:
            if isinstance(self.scale_factor, Sequence):
                if len(self.scale_factor) != len(spatial_shape):
                    raise ValueError(
                        f"scale_factor length {len(self.scale_factor)} does not match spatial dims {len(spatial_shape)}"
                    )
                target_size = tuple(int(s * f) for s, f in zip(spatial_shape, self.scale_factor))
            else:
                target_size = tuple(int(s * float(self.scale_factor)) for s in spatial_shape)
        else:
            target_size = spatial_shape

        if self.mode == "nearest" and len(target_size) == 2:
            out = sops.resize_nearest(x._tensor, size=target_size, align_corners=self.align_corners)
            return array(out, dtype=x.dtype)

        res = x
        for sp_idx, n_out in enumerate(target_size):
            res = _interp_1d_axis(
                res,
                n_out,
                axis=sp_idx + 1,
                mode=self.mode,
                align_corners=self.align_corners,
            )
        return res


__all__ = ["Upsample"]
