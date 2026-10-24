"""Approximation to the Gaussian Error Linear Unit (`gelu_approx`)."""

from __future__ import annotations

import math

from zero_mlx._types import ArrayInput
from zero_mlx.array import array
import zero_mlx.ops as mx


def gelu_approx(x: ArrayInput) -> array:
    """Compute the tanh-based approximation to the Gaussian Error Linear Unit (GELU).

    Args:
        x: Input array.

    Returns:
        Approximate GELU `array`.

    """
    arr = x if isinstance(x, array) else array(x)
    c = math.sqrt(2.0 / math.pi)
    return 0.5 * arr * (1.0 + mx.tanh(c * (arr + 0.044715 * (arr**3))))


__all__ = ["gelu_approx"]
