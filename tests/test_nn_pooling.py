"""Unit tests for `MaxPool1d/2d/3d`, `AvgPool1d/2d/3d`, and `Upsample`."""

from __future__ import annotations

import math
from typing import Callable

import pytest

import zero_mlx as mx
from zero_mlx.array import array
import zero_mlx.nn as z_nn


def _test_module_shape(
    m: Callable[[array], array],
    x_shape: tuple[int, ...],
    expected_shape: tuple[int, ...],
) -> None:
    """Run pooling/upsampling module `m` on an input of `x_shape` and assert `expected_shape`.

    Args:
        m: Pooling or upsampling `Module` to test.
        x_shape: Input tensor shape.
        expected_shape: Expected output tensor shape.

    """
    x = mx.linspace(0.0, 4.0, math.prod(x_shape), dtype=mx.float32).reshape(x_shape)
    out = m(x)
    assert out.shape == expected_shape
    assert bool(mx.all(mx.isfinite(out)).item()) is True
    assert float(mx.min(out).item().real) >= -1.0 and float(mx.max(out).item().real) <= 5.0


def test_maxpool1d() -> None:
    """Test `MaxPool1d` with int and tuple kernel sizes."""
    m = z_nn.MaxPool1d(2)
    _test_module_shape(m, (2, 5, 3), (2, 2, 3))
    m_tup = z_nn.MaxPool1d((2,))
    _test_module_shape(m_tup, (2, 5, 3), (2, 2, 3))


def test_maxpool2d() -> None:
    """Test `MaxPool2d` with int and tuple kernel sizes."""
    m = z_nn.MaxPool2d(2)
    _test_module_shape(m, (2, 5, 5, 3), (2, 2, 2, 3))
    m_tup = z_nn.MaxPool2d((2, 2))
    _test_module_shape(m_tup, (2, 5, 5, 3), (2, 2, 2, 3))


def test_maxpool3d() -> None:
    """Test `MaxPool3d` with int and tuple kernel sizes."""
    m = z_nn.MaxPool3d(2)
    _test_module_shape(m, (2, 5, 5, 5, 3), (2, 2, 2, 2, 3))
    m_tup = z_nn.MaxPool3d((2, 2, 2))
    _test_module_shape(m_tup, (2, 5, 5, 5, 3), (2, 2, 2, 2, 3))


def test_avgpool1d() -> None:
    """Test `AvgPool1d` with int and tuple kernel sizes."""
    m = z_nn.AvgPool1d(2)
    _test_module_shape(m, (2, 5, 3), (2, 2, 3))
    m_tup = z_nn.AvgPool1d((2,))
    _test_module_shape(m_tup, (2, 5, 3), (2, 2, 3))


def test_avgpool2d() -> None:
    """Test `AvgPool2d` with int and tuple kernel sizes."""
    m = z_nn.AvgPool2d(2)
    _test_module_shape(m, (2, 5, 5, 3), (2, 2, 2, 3))
    m_tup = z_nn.AvgPool2d((2, 2))
    _test_module_shape(m_tup, (2, 5, 5, 3), (2, 2, 2, 3))


def test_avgpool3d() -> None:
    """Test `AvgPool3d` with int and tuple kernel sizes."""
    m = z_nn.AvgPool3d(2)
    _test_module_shape(m, (2, 5, 5, 5, 3), (2, 2, 2, 2, 3))
    m_tup = z_nn.AvgPool3d((2, 2, 2))
    _test_module_shape(m_tup, (2, 5, 5, 5, 3), (2, 2, 2, 2, 3))


def test_upsample() -> None:
    """Test `Upsample` with `scale_factor=2.0` and validation errors."""
    m = z_nn.Upsample(scale_factor=2.0)
    _test_module_shape(m, (2, 2, 2, 3), (2, 4, 4, 3))
    with pytest.raises(ValueError):
        z_nn.Upsample(mode="invalid_mode")
    with pytest.raises(ValueError):
        m(array([1.0, 2.0]))
    with pytest.raises(ValueError):
        z_nn.Upsample(size=(4, 4, 4))(mx.ones((2, 2, 2, 3), dtype=mx.float32))
    with pytest.raises(ValueError):
        z_nn.Upsample(scale_factor=(2.0, 2.0, 2.0))(mx.ones((2, 2, 2, 3), dtype=mx.float32))
