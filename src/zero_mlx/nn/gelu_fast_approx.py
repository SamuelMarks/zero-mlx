"""Fast sigmoid-based approximation to the Gaussian Error Linear Unit (`gelu_fast_approx`)."""

from __future__ import annotations

from zero_mlx._types import ArrayInput
from zero_mlx.array import array
import zero_mlx.ops as mx


def gelu_fast_approx(x: ArrayInput) -> array:
    """Compute the fast sigmoid approximation `x * sigmoid(1.702 * x)` to GELU.

    Args:
        x: Input array.

    Returns:
        Fast approximate GELU `array`.

    """
    arr = x if isinstance(x, array) else array(x)
    return arr * mx.sigmoid(arr * 1.702)


__all__ = ["gelu_fast_approx"]
