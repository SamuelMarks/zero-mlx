"""Spatial upsampling layer (`Upsample`) for `mlx.nn`."""

from __future__ import annotations

from typing import Sequence

import ml_switcheroo_compiler.ops as sops

from zero_mlx.array import array
from zero_mlx.nn.base import Module


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

        out = sops.resize_nearest(x._tensor, size=target_size, align_corners=self.align_corners)
        return array(out, dtype=x.dtype)


__all__ = ["Upsample"]
