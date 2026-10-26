"""Unit tests for `BatchNorm`, `GroupNorm`, `InstanceNorm`, `LayerNorm`, and `RMSNorm`."""

from __future__ import annotations

import zero_mlx as mx
from zero_mlx.array import array
import zero_mlx.nn as z_nn


def test_batchnorm() -> None:
    """Test `BatchNorm` normalizes each feature channel to zero mean and unit variance."""
    m = z_nn.BatchNorm(2, eps=0.0)
    x = array([[1.0, 10.0], [3.0, 20.0]], dtype=mx.float32)
    out = m(x)
    expected = array([[-1.0, -1.0], [1.0, 1.0]], dtype=mx.float32)
    assert out.shape == (2, 2)
    assert bool(mx.allclose(out, expected, rtol=1e-4, atol=1e-4).item()) is True
    m.eval()
    out_eval = m(x)
    assert out_eval.shape == (2, 2)
    assert bool(mx.all(mx.isfinite(out_eval)).item()) is True

    m_no_affine = z_nn.BatchNorm(2, eps=0.0, affine=False, track_running_stats=False)
    out_na = m_no_affine(x)
    assert bool(mx.allclose(out_na, expected, rtol=1e-4, atol=1e-4).item()) is True


def test_groupnorm() -> None:
    """Test `GroupNorm` normalizes across channel groups and respects `pytorch_compatible`."""
    m = z_nn.GroupNorm(2, 4, eps=0.0, affine=True, pytorch_compatible=True)
    x = array([[[[1.0, 3.0, 10.0, 20.0]]]], dtype=mx.float32)
    out = m(x)
    expected = array([[[[-1.0, 1.0, -1.0, 1.0]]]], dtype=mx.float32)
    assert out.shape == (1, 1, 1, 4)
    assert bool(mx.allclose(out, expected, rtol=1e-4, atol=1e-4).item()) is True

    m_non_pt = z_nn.GroupNorm(2, 4, eps=0.0, affine=False, pytorch_compatible=False)
    out_non_pt = m_non_pt(x)
    assert out_non_pt.shape == (1, 1, 1, 4)
    assert abs(float(mx.mean(out_non_pt).item().real)) < 1e-5


def test_instancenorm() -> None:
    """Test `InstanceNorm` normalizes spatial dimensions per sample and channel."""
    m = z_nn.InstanceNorm(2, eps=0.0, affine=True)
    x = array([[[1.0, 10.0], [3.0, 20.0]]], dtype=mx.float32)
    out = m(x)
    expected = array([[[-1.0, -1.0], [1.0, 1.0]]], dtype=mx.float32)
    assert out.shape == (1, 2, 2)
    assert bool(mx.allclose(out, expected, rtol=1e-4, atol=1e-4).item()) is True

    m_no_affine = z_nn.InstanceNorm(2, eps=0.0, affine=False)
    assert bool(mx.allclose(m_no_affine(x), expected, rtol=1e-4, atol=1e-4).item()) is True


def test_layernorm() -> None:
    """Test `LayerNorm` normalizes the last dimension(s) to zero mean and unit variance."""
    m = z_nn.LayerNorm(2, eps=0.0, affine=True)
    x = array([[1.0, 3.0], [10.0, 20.0]], dtype=mx.float32)
    out = m(x)
    expected = array([[-1.0, 1.0], [-1.0, 1.0]], dtype=mx.float32)
    assert out.shape == (2, 2)
    assert bool(mx.allclose(out, expected, rtol=1e-4, atol=1e-4).item()) is True

    m.weight = array([2.0, 2.0], dtype=mx.float32)
    m.bias = array([0.5, 0.5], dtype=mx.float32)
    out_scaled = m(x)
    expected_scaled = array([[-1.5, 2.5], [-1.5, 2.5]], dtype=mx.float32)
    assert bool(mx.allclose(out_scaled, expected_scaled, rtol=1e-4, atol=1e-4).item()) is True

    m_no_affine = z_nn.LayerNorm(2, eps=0.0, affine=False)
    assert bool(mx.allclose(m_no_affine(x), expected, rtol=1e-4, atol=1e-4).item()) is True


def test_rmsnorm() -> None:
    """Test `RMSNorm` divides by root-mean-square `sqrt(mean(x^2) + eps)`."""
    m = z_nn.RMSNorm(2, eps=0.0)
    x = array([[3.0, 3.0], [4.0, -4.0]], dtype=mx.float32)
    out = m(x)
    expected = array([[1.0, 1.0], [1.0, -1.0]], dtype=mx.float32)
    assert out.shape == (2, 2)
    assert bool(mx.allclose(out, expected, rtol=1e-4, atol=1e-4).item()) is True
