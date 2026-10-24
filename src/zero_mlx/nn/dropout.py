"""Dropout layers (`Dropout`, `Dropout2d`, `Dropout3d`) for `mlx.nn`."""

from __future__ import annotations

from zero_mlx.array import array
from zero_mlx.dtypes import float32
from zero_mlx.mlx_random import uniform
from zero_mlx.nn.base import Module
import zero_mlx.ops as mx


class Dropout(Module):
    """During training, randomly zeroes elements of the input tensor with probability `p`."""

    def __init__(self, p: float = 0.5) -> None:
        """Initialize `Dropout`.

        Args:
            p: Probability of an element to be zeroed (in `[0, 1)`).

        Raises:
            ValueError: If `p < 0` or `p >= 1`.

        """
        super().__init__()
        if p < 0.0 or p >= 1.0:
            raise ValueError(f"Dropout probability p={p} is not in [0, 1)")
        self.p: float = p

    def __call__(self, x: array) -> array:
        """Apply dropout to `x` if in training mode.

        Args:
            x: Input `array`.

        Returns:
            Dropped-out `array`.

        """
        if not self.training or self.p == 0.0:
            return x
        mask = uniform(shape=x.shape, dtype=float32) > self.p
        return (x * mask) / (1.0 - self.p)


class Dropout2d(Dropout):
    """Applies 2D channel-wise dropout to 3D `(L, H, C)` or 4D `(N, H, W, C)` inputs."""

    def __call__(self, x: array) -> array:
        """Apply 2D channel-wise dropout to `x` if in training mode.

        Args:
            x: Input `array` of 3D `(H, W, C)` or 4D `(N, H, W, C)`.

        Returns:
            Channel-dropped `array`.

        Raises:
            ValueError: If `x.ndim` is not 3 or 4.

        """
        if x.ndim not in (3, 4):
            raise ValueError(f"Expected 3D or 4D input for Dropout2d, got {x.ndim}D")
        if not self.training or self.p == 0.0:
            return x
        mask_shape: tuple[int, ...] = (
            (1, 1, x.shape[-1]) if x.ndim == 3 else (x.shape[0], 1, 1, x.shape[-1])
        )
        mask = uniform(shape=mask_shape, dtype=float32) > self.p
        return (x * mask) / (1.0 - self.p)


class Dropout3d(Dropout):
    """Applies 3D channel-wise dropout to 4D `(D, H, W, C)` or 5D `(N, D, H, W, C)` inputs."""

    def __call__(self, x: array) -> array:
        """Apply 3D channel-wise dropout to `x` if in training mode.

        Args:
            x: Input `array` of 4D `(D, H, W, C)` or 5D `(N, D, H, W, C)`.

        Returns:
            Channel-dropped `array`.

        Raises:
            ValueError: If `x.ndim` is not 4 or 5.

        """
        if x.ndim not in (4, 5):
            raise ValueError(f"Expected 4D or 5D input for Dropout3d, got {x.ndim}D")
        if not self.training or self.p == 0.0:
            return x
        mask_shape: tuple[int, ...] = (
            (1, 1, 1, x.shape[-1]) if x.ndim == 4 else (x.shape[0], 1, 1, 1, x.shape[-1])
        )
        mask = uniform(shape=mask_shape, dtype=float32) > self.p
        return (x * mask) / (1.0 - self.p)


__all__ = ["Dropout", "Dropout2d", "Dropout3d"]
