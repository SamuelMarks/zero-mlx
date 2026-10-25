"""Unit tests for `Conv1d`, `Conv2d`, `Conv3d`, `ConvTranspose1d`, `ConvTranspose2d`, and `ConvTranspose3d`."""

from __future__ import annotations

import math
from typing import Callable

import zero_mlx as mx
from zero_mlx.array import array
import zero_mlx.nn as z_nn


def _test_module_shape(
    m: Callable[[array], array],
    x_shape: tuple[int, ...],
    expected_shape: tuple[int, ...],
) -> None:
    """Run module `m` on a deterministic input of `x_shape` and assert `expected_shape`.

    Args:
        m: Convolutional `Module` to test.
        x_shape: Input tensor shape.
        expected_shape: Expected output tensor shape.

    """
    x = mx.linspace(0.1, 1.0, math.prod(x_shape), dtype=mx.float32).reshape(x_shape)
    out = m(x)
    assert out.shape == expected_shape
    assert bool(mx.all(mx.isfinite(out)).item()) is True
    assert bool(mx.allclose(out, m(x), rtol=1e-5, atol=1e-5).item()) is True


def test_conv1d() -> None:
    """Test `Conv1d` with bias."""
    m = z_nn.Conv1d(3, 4, 2)
    _test_module_shape(m, (2, 5, 3), (2, 4, 4))


def test_conv1d_no_bias() -> None:
    """Test `Conv1d` without bias."""
    m = z_nn.Conv1d(3, 4, 2, bias=False)
    _test_module_shape(m, (2, 5, 3), (2, 4, 4))


def test_conv2d() -> None:
    """Test `Conv2d` with bias."""
    m = z_nn.Conv2d(3, 4, 2)
    _test_module_shape(m, (2, 5, 5, 3), (2, 4, 4, 4))


def test_conv2d_no_bias() -> None:
    """Test `Conv2d` without bias."""
    m = z_nn.Conv2d(3, 4, 2, bias=False)
    _test_module_shape(m, (2, 5, 5, 3), (2, 4, 4, 4))


def test_conv3d() -> None:
    """Test `Conv3d` with bias."""
    m = z_nn.Conv3d(3, 4, 2)
    _test_module_shape(m, (2, 5, 5, 5, 3), (2, 4, 4, 4, 4))


def test_conv3d_no_bias() -> None:
    """Test `Conv3d` without bias."""
    m = z_nn.Conv3d(3, 4, 2, bias=False)
    _test_module_shape(m, (2, 5, 5, 5, 3), (2, 4, 4, 4, 4))


def test_conv_transpose1d() -> None:
    """Test `ConvTranspose1d` with bias."""
    m = z_nn.ConvTranspose1d(3, 4, 2)
    _test_module_shape(m, (2, 5, 3), (2, 6, 4))


def test_conv_transpose1d_no_bias() -> None:
    """Test `ConvTranspose1d` without bias."""
    m = z_nn.ConvTranspose1d(3, 4, 2, bias=False)
    _test_module_shape(m, (2, 5, 3), (2, 6, 4))


def test_conv_transpose2d() -> None:
    """Test `ConvTranspose2d` with bias."""
    m = z_nn.ConvTranspose2d(3, 4, 2)
    _test_module_shape(m, (2, 5, 5, 3), (2, 6, 6, 4))


def test_conv_transpose2d_no_bias() -> None:
    """Test `ConvTranspose2d` without bias."""
    m = z_nn.ConvTranspose2d(3, 4, 2, bias=False)
    _test_module_shape(m, (2, 5, 5, 3), (2, 6, 6, 4))


def test_conv_transpose3d() -> None:
    """Test `ConvTranspose3d` with bias."""
    m = z_nn.ConvTranspose3d(3, 4, 2)
    _test_module_shape(m, (2, 5, 5, 5, 3), (2, 6, 6, 6, 4))


def test_conv_transpose3d_no_bias() -> None:
    """Test `ConvTranspose3d` without bias."""
    m = z_nn.ConvTranspose3d(3, 4, 2, bias=False)
    _test_module_shape(m, (2, 5, 5, 5, 3), (2, 6, 6, 6, 4))
