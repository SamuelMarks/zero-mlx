"""Behavioral smoke tests for `ArrayLike`, `average_gradients`, `gelu_fast_approx`, and `eval`."""

from __future__ import annotations

import math

import zero_mlx as mx
from zero_mlx.array import array
from zero_mlx.array_like import ArrayLike
from zero_mlx.new_ops import eval as z_eval
from zero_mlx.nn.average_gradients import average_gradients
from zero_mlx.nn.gelu_fast_approx import gelu_fast_approx


class _SampleArrayLike:
    """Custom `ArrayLike` implementation for protocol verification."""

    def __mlx_array__(self) -> array:
        """Convert to a 1D `zero_mlx.array`.

        Returns:
            Converted `zero_mlx.array`.

        """
        return array([1.0, 2.0])


def test_module_imports() -> None:
    """Verify behavioral contracts of `ArrayLike`, `average_gradients`, `gelu_fast_approx`, and `eval`."""
    sample = _SampleArrayLike()
    assert isinstance(sample, ArrayLike)
    converted = sample.__mlx_array__()
    z_eval(converted)
    assert converted.tolist() == [1.0, 2.0]

    grads: mx._types.PyTree = {"w": array([2.0, 4.0]), "meta": 1}
    avg = average_gradients(grads)
    assert (
        isinstance(avg, dict)
        and isinstance(avg["w"], array)
        and avg["w"].tolist() == [2.0, 4.0]
        and avg["meta"] == 1
    )

    g_out = gelu_fast_approx(array([0.0, 1.0]))
    expected = array([0.0, 1.0 / (1.0 + math.exp(-1.702))])
    assert bool(mx.allclose(g_out, expected, rtol=1e-5, atol=1e-5).item()) is True
