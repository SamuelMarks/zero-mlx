"""Unit test for `zero_mlx.nan_to_num` operation."""

from __future__ import annotations

import zero_mlx as mx
from zero_mlx.array import array


def test_nan_to_num() -> None:
    """Verify `nan_to_num` replaces NaN, positive infinity, and negative infinity values."""
    x = array([float("nan"), float("inf"), float("-inf"), 1.0], dtype=mx.float32)
    out = mx.nan_to_num(x, nan=0.0, posinf=1e30, neginf=-1e30)
    expected = array([0.0, 1e30, -1e30, 1.0], dtype=mx.float32)
    assert bool(mx.allclose(out, expected, rtol=1e-5, atol=1e-5).item()) is True
