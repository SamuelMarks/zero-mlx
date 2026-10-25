"""Unit tests for `Dropout`, `Embedding`, `QuantizedEmbedding`, `ALiBi`, `RoPE`, and `SinusoidalPositionalEncoding`."""

from __future__ import annotations

import pytest

import zero_mlx as mx
from zero_mlx.array import array
import zero_mlx.nn as z_nn


def test_dropout() -> None:
    """Test `Dropout` in training and evaluation modes."""
    m = z_nn.Dropout(0.5)
    x = mx.random.normal((2, 3))
    assert m(x).shape == (2, 3)
    m.eval()
    assert m(x).shape == (2, 3)


def test_dropout2d() -> None:
    """Test `Dropout2d` in training and evaluation modes."""
    m = z_nn.Dropout2d(0.5)
    x = mx.random.normal((2, 3, 3, 4))
    assert m(x).shape == (2, 3, 3, 4)
    assert m(mx.random.normal((3, 3, 4))).shape == (3, 3, 4)
    m.eval()
    assert m(x).shape == (2, 3, 3, 4)
    with pytest.raises(ValueError):
        m(array([1.0, 2.0]))


def test_dropout3d() -> None:
    """Test `Dropout3d` in training and evaluation modes."""
    m = z_nn.Dropout3d(0.5)
    x = mx.random.normal((2, 3, 3, 3, 4))
    assert m(x).shape == (2, 3, 3, 3, 4)
    assert m(mx.random.normal((3, 3, 3, 4))).shape == (3, 3, 3, 4)
    m.eval()
    assert m(x).shape == (2, 3, 3, 3, 4)
    with pytest.raises(ValueError):
        m(array([1.0, 2.0]))


def test_embedding() -> None:
    """Test `Embedding` lookup."""
    m = z_nn.Embedding(10, 3)
    x = mx.random.randint(0, 10, shape=(2, 4), dtype=mx.int32)
    out = m(x)
    assert out.shape == (2, 4, 3)


def test_quantized_embedding() -> None:
    """Test `QuantizedEmbedding` lookup."""
    m = z_nn.QuantizedEmbedding(10, 64)
    x = mx.random.randint(0, 10, shape=(2, 4), dtype=mx.int32)
    out = m(x)
    assert out.shape == (2, 4, 64)


def test_alibi() -> None:
    """Test `ALiBi` positional bias module."""
    m = z_nn.ALiBi()
    x = mx.random.normal((2, 4, 5, 5))
    out = m(x)
    assert out.shape == (2, 4, 5, 5)


def test_rope() -> None:
    """Test `RoPE` rotary positional encoding module."""
    m = z_nn.RoPE(4)
    x = mx.random.normal((2, 5, 4))
    out = m(x)
    assert out.shape == (2, 5, 4)
    m_trad = z_nn.RoPE(4, traditional=True)
    assert m_trad(x).shape == (2, 5, 4)


def test_sinusoidal() -> None:
    """Test `SinusoidalPositionalEncoding` module."""
    m = z_nn.SinusoidalPositionalEncoding(4)
    x = mx.random.normal((2, 5, 4))
    out = m(x)
    assert out.shape == (2, 5, 4)
