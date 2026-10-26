"""Unit tests for `MaxPool1d/2d/3d`, `AvgPool1d/2d/3d`, and `Upsample`."""

from __future__ import annotations

import pytest

import zero_mlx as mx
from zero_mlx.array import array
import zero_mlx.nn as z_nn


def test_maxpool1d() -> None:
    """Test `MaxPool1d` computes exact window maxima."""
    m = z_nn.MaxPool1d(2)
    x = array([[[1.0], [4.0], [2.0], [3.0]]], dtype=mx.float32)
    out = m(x)
    assert out.shape == (1, 2, 1)
    assert bool(mx.allclose(out, array([[[4.0], [3.0]]]), rtol=1e-5, atol=1e-5).item()) is True


def test_maxpool2d() -> None:
    """Test `MaxPool2d` computes exact 2x2 window maxima."""
    m = z_nn.MaxPool2d(2)
    x = array([[[[1.0], [2.0]], [[3.0], [4.0]]]], dtype=mx.float32)
    out = m(x)
    assert out.shape == (1, 1, 1, 1)
    assert float(out.item().real) == 4.0


def test_maxpool3d() -> None:
    """Test `MaxPool3d` computes exact 2x2x2 window maxima."""
    m = z_nn.MaxPool3d(2)
    x = mx.arange(0, 8, dtype=mx.float32).reshape((1, 2, 2, 2, 1))
    out = m(x)
    assert out.shape == (1, 1, 1, 1, 1)
    assert float(out.item().real) == 7.0


def test_avgpool1d() -> None:
    """Test `AvgPool1d` computes exact window averages."""
    m = z_nn.AvgPool1d(2)
    x = array([[[1.0], [3.0], [2.0], [6.0]]], dtype=mx.float32)
    out = m(x)
    assert out.shape == (1, 2, 1)
    assert bool(mx.allclose(out, array([[[2.0], [4.0]]]), rtol=1e-5, atol=1e-5).item()) is True


def test_avgpool2d() -> None:
    """Test `AvgPool2d` computes exact 2x2 window averages."""
    m = z_nn.AvgPool2d(2)
    x = array([[[[1.0], [2.0]], [[3.0], [4.0]]]], dtype=mx.float32)
    out = m(x)
    assert out.shape == (1, 1, 1, 1)
    assert abs(float(out.item().real) - 2.5) < 1e-5


def test_avgpool3d() -> None:
    """Test `AvgPool3d` computes exact 2x2x2 window averages."""
    m = z_nn.AvgPool3d(2)
    x = mx.arange(0, 8, dtype=mx.float32).reshape((1, 2, 2, 2, 1))
    out = m(x)
    assert out.shape == (1, 1, 1, 1, 1)
    assert abs(float(out.item().real) - 3.5) < 1e-5


def test_upsample() -> None:
    """Test `Upsample` nearest and linear interpolation values and validation errors."""
    m = z_nn.Upsample(scale_factor=2.0, mode="nearest")
    x = array([[[1.0], [2.0]]], dtype=mx.float32)
    out = m(x)
    assert out.shape == (1, 4, 1)
    assert (
        bool(mx.allclose(out, array([[[1.0], [1.0], [2.0], [2.0]]]), rtol=1e-5, atol=1e-5).item())
        is True
    )

    m_lin = z_nn.Upsample(size=(3,), mode="linear", align_corners=True)
    out_lin = m_lin(x)
    assert out_lin.shape == (1, 3, 1)
    assert (
        bool(mx.allclose(out_lin, array([[[1.0], [1.5], [2.0]]]), rtol=1e-5, atol=1e-5).item())
        is True
    )

    with pytest.raises(ValueError):
        z_nn.Upsample(mode="invalid_mode")
    with pytest.raises(ValueError):
        z_nn.Upsample(size=(4,), scale_factor=2.0)
    with pytest.raises(ValueError):
        m(array([1.0, 2.0]))
    with pytest.raises(ValueError):
        z_nn.Upsample(size=(4, 4, 4))(mx.ones((2, 2, 2, 3), dtype=mx.float32))
    with pytest.raises(ValueError):
        z_nn.Upsample(scale_factor=(2.0, 2.0, 2.0))(mx.ones((2, 2, 2, 3), dtype=mx.float32))
