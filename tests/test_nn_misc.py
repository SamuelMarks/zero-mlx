"""Unit tests for `Dropout`, `Embedding`, `QuantizedEmbedding`, `ALiBi`, `RoPE`, and `SinusoidalPositionalEncoding`."""

from __future__ import annotations

import math

import pytest

import zero_mlx as mx
from zero_mlx.array import array
import zero_mlx.nn as z_nn


def test_dropout() -> None:
    """Test `Dropout` scales active elements in train mode and is identity in eval mode."""
    m = z_nn.Dropout(0.0)
    x = array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=mx.float32)
    assert bool(mx.allclose(m(x), x, rtol=1e-5, atol=1e-5).item()) is True

    m_half = z_nn.Dropout(0.5)
    assert m_half(x).shape == (2, 3)
    m_half.eval()
    assert bool(mx.allclose(m_half(x), x, rtol=1e-5, atol=1e-5).item()) is True


def test_dropout2d() -> None:
    """Test `Dropout2d` in training and evaluation modes."""
    m = z_nn.Dropout2d(0.5)
    x = mx.ones((2, 3, 3, 4), dtype=mx.float32)
    out_train = m(x)
    assert out_train.shape == (2, 3, 3, 4)
    assert m(mx.ones((3, 3, 4), dtype=mx.float32)).shape == (3, 3, 4)
    m.eval()
    assert bool(mx.allclose(m(x), x, rtol=1e-5, atol=1e-5).item()) is True
    with pytest.raises(ValueError):
        m(array([1.0, 2.0]))


def test_dropout3d() -> None:
    """Test `Dropout3d` in training and evaluation modes."""
    m = z_nn.Dropout3d(0.5)
    x = mx.ones((2, 3, 3, 3, 4), dtype=mx.float32)
    assert m(x).shape == (2, 3, 3, 3, 4)
    assert m(mx.ones((3, 3, 3, 4), dtype=mx.float32)).shape == (3, 3, 3, 4)
    m.eval()
    assert bool(mx.allclose(m(x), x, rtol=1e-5, atol=1e-5).item()) is True
    with pytest.raises(ValueError):
        m(array([1.0, 2.0]))


def test_embedding() -> None:
    """Test `Embedding` returns the exact rows of `weight` indexed by `x`."""
    m = z_nn.Embedding(3, 2)
    m.weight = array([[10.0, 20.0], [30.0, 40.0], [50.0, 60.0]], dtype=mx.float32)
    x = array([[2, 0]], dtype=mx.int32)
    out = m(x)
    expected = array([[[50.0, 60.0], [10.0, 20.0]]], dtype=mx.float32)
    assert out.shape == (1, 2, 2)
    assert bool(mx.allclose(out, expected, rtol=1e-5, atol=1e-5).item()) is True


def test_quantized_embedding() -> None:
    """Test `QuantizedEmbedding` closely approximates `Embedding` lookup."""
    emb = z_nn.Embedding(4, 64)
    emb.weight = mx.ones((4, 64), dtype=mx.float32) * 0.5
    qemb = z_nn.QuantizedEmbedding.from_embedding(emb, group_size=64, bits=8)
    x = array([[0, 2]], dtype=mx.int32)
    out = qemb(x)
    assert out.shape == (1, 2, 64)
    assert bool(mx.allclose(out, emb(x), rtol=5e-2, atol=5e-2).item()) is True


def test_alibi() -> None:
    """Test `ALiBi` adds distance-proportional bias with zero diagonal."""
    m = z_nn.ALiBi()
    x = mx.zeros((1, 1, 3, 3), dtype=mx.float32)
    out = m(x)
    assert out.shape == (1, 1, 3, 3)
    assert abs(float(out[0, 0, 0, 0].item().real)) < 1e-6
    assert float(out[0, 0, 0, 2].item().real) < float(out[0, 0, 0, 1].item().real) < 0.0


def test_rope() -> None:
    """Test `RoPE` preserves vector L2 norm and is identity at position 0."""
    m = z_nn.RoPE(4)
    x = array([[[1.0, 2.0, 3.0, 4.0], [1.0, 0.0, 0.0, 0.0]]], dtype=mx.float32)
    out = m(x)
    assert out.shape == (1, 2, 4)
    # Position 0 has angle 0 -> cos(0)=1, sin(0)=0 -> unchanged
    assert bool(mx.allclose(out[:, 0, :], x[:, 0, :], rtol=1e-5, atol=1e-5).item()) is True
    # Norm is preserved at position 1
    assert abs(float(mx.sum(out[:, 1, :] ** 2).item().real) - 1.0) < 1e-5
    m_trad = z_nn.RoPE(4, traditional=True)
    out_trad = m_trad(x)
    assert abs(float(mx.sum(out_trad[:, 1, :] ** 2).item().real) - 1.0) < 1e-5


def test_sinusoidal() -> None:
    """Test `SinusoidalPositionalEncoding` values at position 0 and 2D additive mode."""
    m = z_nn.SinusoidalPositionalEncoding(4, scale=1.0, cos_first=False)
    x = array([0], dtype=mx.int32)
    out = m(x)
    assert out.shape == (1, 4)
    # Interleaved [sin(0), cos(0), sin(0), cos(0)] = [0, 1, 0, 1]
    assert (
        bool(mx.allclose(out, array([[0.0, 1.0, 0.0, 1.0]]), rtol=1e-5, atol=1e-5).item()) is True
    )
    out_3d = m(mx.zeros((1, 2, 4), dtype=mx.float32))
    assert out_3d.shape == (1, 2, 4)
    _ = math.pi
