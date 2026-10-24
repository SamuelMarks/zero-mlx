"""Smoke tests verifying top-level module imports for `zero_mlx`."""

from __future__ import annotations

from zero_mlx.array_like import ArrayLike
from zero_mlx.new_ops import eval as z_eval
from zero_mlx.nn.average_gradients import average_gradients
from zero_mlx.nn.gelu_fast_approx import gelu_fast_approx


def test_module_imports() -> None:
    """Verify that core submodules import and expose their expected symbols."""
    assert ArrayLike is not None
    assert callable(average_gradients)
    assert callable(gelu_fast_approx)
    assert callable(z_eval)
