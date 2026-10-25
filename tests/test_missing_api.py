"""Unit tests for `checkpoint`, `cuda`, and `custom_function` API parity."""

from __future__ import annotations

import zero_mlx as mx
from zero_mlx.array import array
import zero_mlx.cuda as cuda


def test_checkpoint_parity() -> None:
    """Test `mx.checkpoint` function wrapper."""

    def f(*args: array, **kwargs: array) -> array:
        """Double input.

        Args:
            *args: Positional array arguments.
            **kwargs: Keyword array arguments.

        Returns:
            Doubled `array`.

        """
        _ = kwargs
        return args[0] * 2

    mock_f = mx.checkpoint(f)
    res = mock_f(array(2))
    assert isinstance(res, array)
    assert res.item() == 4


def test_cuda_parity() -> None:
    """Test `zero_mlx.cuda` backend status and memory functions."""
    assert cuda.is_available() is False
    assert cuda.device_count() == 0
    assert cuda.memory_info() == {}
    cuda.clear_cache()


def test_custom_function_parity() -> None:
    """Test `@mx.custom_function` decorator and `.vjp`, `.jvp`, `.vmap` registration."""

    @mx.custom_function
    def my_f(*args: array, **kwargs: array) -> array:
        """Double input.

        Args:
            *args: Positional array arguments.
            **kwargs: Keyword array arguments.

        Returns:
            Doubled `array`.

        """
        _ = kwargs
        return args[0] * 2

    @my_f.vjp
    def my_f_vjp(*args: array, **kwargs: array) -> array:
        """Custom VJP rule.

        Args:
            *args: Positional array arguments.
            **kwargs: Keyword array arguments.

        Returns:
            VJP gradient `array`.

        """
        _ = kwargs
        return args[0] * 2

    @my_f.jvp
    def my_f_jvp(*args: array, **kwargs: array) -> array:
        """Custom JVP rule.

        Args:
            *args: Positional array arguments.
            **kwargs: Keyword array arguments.

        Returns:
            JVP tangent `array`.

        """
        _ = kwargs
        return args[0] * 2

    @my_f.vmap
    def my_f_vmap(*args: array, **kwargs: array) -> array:
        """Custom vmap rule.

        Args:
            *args: Positional array arguments.
            **kwargs: Keyword array arguments.

        Returns:
            Vectorized `array`.

        """
        _ = kwargs
        return args[0] * 2

    assert my_f(array(3)).item() == 6
    outs_v, grads_v = mx.vjp(my_f, [array(3.0)], [array(1.0)])
    assert outs_v[0].item() == 6.0 and grads_v[0].item() == 6.0
    outs_j, tangs_j = mx.jvp(my_f, [array(3.0)], [array(1.0)])
    assert outs_j[0].item() == 6.0 and tangs_j[0].item() == 6.0
    assert mx.vmap(my_f)(array([1.0, 2.0])).tolist() == [2.0, 4.0]
