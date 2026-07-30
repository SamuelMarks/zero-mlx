"""mlx.nn.dropout module."""

from zero_mlx.nn.base import Module
from zero_mlx.array import array
import zero_mlx as mx
from zero_mlx.mlx_random import uniform


class Dropout(Module):
    """Applies dropout to the input."""

    def __init__(self, p: float = 0.5):
        """Initialize Dropout."""
        super().__init__()
        self.p = p

    def __call__(self, x: array) -> array:
        """Call."""
        if not self.training or self.p == 0.0:
            return x
        if self.p == 1.0:
            return mx.zeros_like(x)  # pragma: no cover
        mask = uniform(shape=x.shape, dtype=mx.float32) > self.p
        return (x * mask) / (1.0 - self.p)


class Dropout2d(Module):
    """Applies 2D dropout to the input."""

    def __init__(self, p: float = 0.5):
        """Initialize Dropout2d."""
        super().__init__()
        self.p = p

    def __call__(self, x: array) -> array:
        """Call."""
        if not self.training or self.p == 0.0:
            return x
        if self.p == 1.0:
            return mx.zeros_like(x)  # pragma: no cover
        mask = uniform(shape=x.shape, dtype=mx.float32) > self.p
        return (x * mask) / (1.0 - self.p)


class Dropout3d(Module):
    """Applies 3D dropout to the input."""

    def __init__(self, p: float = 0.5):
        """Initialize Dropout3d."""
        super().__init__()
        self.p = p

    def __call__(self, x: array) -> array:
        """Call."""
        if not self.training or self.p == 0.0:
            return x
        if self.p == 1.0:
            return mx.zeros_like(x)  # pragma: no cover
        mask = uniform(shape=x.shape, dtype=mx.float32) > self.p
        return (x * mask) / (1.0 - self.p)


__all__ = ["Dropout", "Dropout2d", "Dropout3d"]
