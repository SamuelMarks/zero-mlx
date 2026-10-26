"""Unit tests for linear, bilinear, sharded, and quantized linear layers in `zero_mlx.nn.linear`."""

from __future__ import annotations

import zero_mlx as mx
from zero_mlx.array import array
import zero_mlx.nn as z_nn


def test_linear() -> None:
    """Test `Linear` with bias computes `x @ weight.T + bias`."""
    m = z_nn.Linear(2, 2, bias=True)
    m.weight = array([[1.0, 2.0], [3.0, 4.0]], dtype=mx.float32)
    m.bias = array([0.5, -0.5], dtype=mx.float32)
    x = array([[1.0, 1.0], [2.0, 0.0]], dtype=mx.float32)
    out = m(x)
    expected = array([[3.5, 6.5], [2.5, 5.5]], dtype=mx.float32)
    assert out.shape == (2, 2)
    assert bool(mx.allclose(out, expected, rtol=1e-5, atol=1e-5).item()) is True


def test_linear_no_bias() -> None:
    """Test `Linear` without bias computes `x @ weight.T`."""
    m = z_nn.Linear(2, 2, bias=False)
    m.weight = array([[1.0, 2.0], [3.0, 4.0]], dtype=mx.float32)
    x = array([[1.0, 1.0], [2.0, 0.0]], dtype=mx.float32)
    out = m(x)
    expected = array([[3.0, 7.0], [2.0, 6.0]], dtype=mx.float32)
    assert out.shape == (2, 2)
    assert bool(mx.allclose(out, expected, rtol=1e-5, atol=1e-5).item()) is True


def test_bilinear() -> None:
    """Test `Bilinear` with bias computes `x1^T W x2 + bias`."""
    m = z_nn.Bilinear(2, 2, 1, bias=True)
    m.weight = mx.ones((1, 2, 2), dtype=mx.float32)
    m.bias = array([1.0], dtype=mx.float32)
    x1 = array([[1.0, 2.0]], dtype=mx.float32)
    x2 = array([[3.0, 4.0]], dtype=mx.float32)
    out = m(x1, x2)
    # (1+2)*(3+4) + 1 = 3*7 + 1 = 22.0
    assert out.shape == (1, 1)
    assert bool(mx.allclose(out, array([[22.0]]), rtol=1e-5, atol=1e-5).item()) is True


def test_bilinear_no_bias() -> None:
    """Test `Bilinear` without bias computes `x1^T W x2`."""
    m = z_nn.Bilinear(2, 2, 1, bias=False)
    m.weight = mx.ones((1, 2, 2), dtype=mx.float32)
    x1 = array([[1.0, 2.0]], dtype=mx.float32)
    x2 = array([[3.0, 4.0]], dtype=mx.float32)
    out = m(x1, x2)
    assert out.shape == (1, 1)
    assert bool(mx.allclose(out, array([[21.0]]), rtol=1e-5, atol=1e-5).item()) is True


def test_all_to_sharded() -> None:
    """Test `AllToShardedLinear` with bias computes `x @ weight.T + bias`."""
    m = z_nn.AllToShardedLinear(2, 2, bias=True)
    m.weight = array([[2.0, 0.0], [0.0, 3.0]], dtype=mx.float32)
    m.bias = array([1.0, 2.0], dtype=mx.float32)
    x = array([[4.0, 5.0]], dtype=mx.float32)
    out = m(x)
    assert out.shape == (1, 2)
    assert bool(mx.allclose(out, array([[9.0, 17.0]]), rtol=1e-5, atol=1e-5).item()) is True


def test_all_to_sharded_no_bias() -> None:
    """Test `AllToShardedLinear` without bias computes `x @ weight.T`."""
    m = z_nn.AllToShardedLinear(2, 2, bias=False)
    m.weight = array([[2.0, 0.0], [0.0, 3.0]], dtype=mx.float32)
    x = array([[4.0, 5.0]], dtype=mx.float32)
    out = m(x)
    assert out.shape == (1, 2)
    assert bool(mx.allclose(out, array([[8.0, 15.0]]), rtol=1e-5, atol=1e-5).item()) is True


def test_sharded_to_all() -> None:
    """Test `ShardedToAllLinear` with bias computes `all_sum(x @ weight.T) + bias`."""
    m = z_nn.ShardedToAllLinear(2, 2, bias=True)
    m.weight = array([[2.0, 0.0], [0.0, 3.0]], dtype=mx.float32)
    m.bias = array([1.0, 2.0], dtype=mx.float32)
    x = array([[4.0, 5.0]], dtype=mx.float32)
    out = m(x)
    assert out.shape == (1, 2)
    assert bool(mx.allclose(out, array([[9.0, 17.0]]), rtol=1e-5, atol=1e-5).item()) is True


def test_sharded_to_all_no_bias() -> None:
    """Test `ShardedToAllLinear` without bias computes `all_sum(x @ weight.T)`."""
    m = z_nn.ShardedToAllLinear(2, 2, bias=False)
    m.weight = array([[2.0, 0.0], [0.0, 3.0]], dtype=mx.float32)
    x = array([[4.0, 5.0]], dtype=mx.float32)
    out = m(x)
    assert out.shape == (1, 2)
    assert bool(mx.allclose(out, array([[8.0, 15.0]]), rtol=1e-5, atol=1e-5).item()) is True


def test_quantized_linear() -> None:
    """Test `QuantizedLinear` with bias closely approximates `Linear`."""
    lin = z_nn.Linear(64, 4, bias=True)
    lin.weight = mx.ones((4, 64), dtype=mx.float32) * 0.5
    lin.bias = array([1.0, 2.0, 3.0, 4.0], dtype=mx.float32)
    qlin = z_nn.QuantizedLinear.from_linear(lin, group_size=64, bits=8)
    x = mx.ones((2, 64), dtype=mx.float32)
    out = qlin(x)
    assert out.shape == (2, 4)
    assert bool(mx.allclose(out, lin(x), rtol=5e-2, atol=5e-2).item()) is True


def test_quantized_linear_no_bias() -> None:
    """Test `QuantizedLinear` without bias closely approximates `Linear`."""
    lin = z_nn.Linear(64, 4, bias=False)
    lin.weight = mx.ones((4, 64), dtype=mx.float32) * 0.25
    qlin = z_nn.QuantizedLinear.from_linear(lin, group_size=64, bits=8)
    x = mx.ones((2, 64), dtype=mx.float32)
    out = qlin(x)
    assert out.shape == (2, 4)
    assert bool(mx.allclose(out, mx.full((2, 4), 16.0), rtol=5e-2, atol=5e-2).item()) is True


def test_quantized_all_to_sharded() -> None:
    """Test `QuantizedAllToShardedLinear` with bias."""
    lin = z_nn.Linear(64, 4, bias=True)
    lin.weight = mx.ones((4, 64), dtype=mx.float32) * 0.25
    lin.bias = array([1.0, 1.0, 1.0, 1.0], dtype=mx.float32)
    m = z_nn.QuantizedAllToShardedLinear.from_quantized_linear(
        z_nn.QuantizedLinear.from_linear(lin, group_size=64, bits=8)
    )
    x = mx.ones((2, 64), dtype=mx.float32)
    out = m(x)
    assert out.shape == (2, 4)
    assert bool(mx.allclose(out, mx.full((2, 4), 17.0), rtol=5e-2, atol=5e-2).item()) is True


def test_quantized_all_to_sharded_no_bias() -> None:
    """Test `QuantizedAllToShardedLinear` without bias."""
    m = z_nn.QuantizedAllToShardedLinear(64, 4, bias=False, group_size=64, bits=8)
    out = m(mx.zeros((2, 64), dtype=mx.float32))
    assert out.shape == (2, 4)
    assert bool(mx.allclose(out, mx.zeros((2, 4)), rtol=1e-5, atol=1e-5).item()) is True


def test_quantized_sharded_to_all() -> None:
    """Test `QuantizedShardedToAllLinear` with bias."""
    lin = z_nn.Linear(64, 4, bias=True)
    lin.weight = mx.ones((4, 64), dtype=mx.float32) * 0.25
    lin.bias = array([2.0, 2.0, 2.0, 2.0], dtype=mx.float32)
    m = z_nn.QuantizedShardedToAllLinear.from_quantized_linear(
        z_nn.QuantizedLinear.from_linear(lin, group_size=64, bits=8)
    )
    x = mx.ones((2, 64), dtype=mx.float32)
    out = m(x)
    assert out.shape == (2, 4)
    assert bool(mx.allclose(out, mx.full((2, 4), 18.0), rtol=5e-2, atol=5e-2).item()) is True


def test_quantized_sharded_to_all_no_bias() -> None:
    """Test `QuantizedShardedToAllLinear` without bias."""
    m = z_nn.QuantizedShardedToAllLinear(64, 4, bias=False, group_size=64, bits=8)
    out = m(mx.zeros((2, 64), dtype=mx.float32))
    assert out.shape == (2, 4)
    assert bool(mx.allclose(out, mx.zeros((2, 4)), rtol=1e-5, atol=1e-5).item()) is True
