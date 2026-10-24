"""Unit tests for linear, bilinear, sharded, and quantized linear layers in `zero_mlx.nn.linear`."""

from __future__ import annotations

import numpy as np

from zero_mlx.array import array
import zero_mlx.nn as z_nn


def test_linear() -> None:
    """Test `Linear` with bias."""
    m = z_nn.Linear(3, 4)
    x = array(np.random.randn(2, 3).astype(np.float32))
    out = m(x)
    assert out.shape == (2, 4)


def test_linear_no_bias() -> None:
    """Test `Linear` without bias."""
    m = z_nn.Linear(3, 4, bias=False)
    x = array(np.random.randn(2, 3).astype(np.float32))
    out = m(x)
    assert out.shape == (2, 4)


def test_bilinear() -> None:
    """Test `Bilinear` with bias."""
    m = z_nn.Bilinear(3, 4, 5)
    x1 = array(np.random.randn(2, 3).astype(np.float32))
    x2 = array(np.random.randn(2, 4).astype(np.float32))
    out = m(x1, x2)
    assert out.shape == (2, 5)


def test_bilinear_no_bias() -> None:
    """Test `Bilinear` without bias."""
    m = z_nn.Bilinear(3, 4, 5, bias=False)
    x1 = array(np.random.randn(2, 3).astype(np.float32))
    x2 = array(np.random.randn(2, 4).astype(np.float32))
    out = m(x1, x2)
    assert out.shape == (2, 5)


def test_all_to_sharded() -> None:
    """Test `AllToShardedLinear` with bias."""
    m = z_nn.AllToShardedLinear(3, 4)
    x = array(np.random.randn(2, 3).astype(np.float32))
    out = m(x)
    assert out.shape == (2, 4)


def test_all_to_sharded_no_bias() -> None:
    """Test `AllToShardedLinear` without bias."""
    m = z_nn.AllToShardedLinear(3, 4, bias=False)
    x = array(np.random.randn(2, 3).astype(np.float32))
    out = m(x)
    assert out.shape == (2, 4)


def test_sharded_to_all() -> None:
    """Test `ShardedToAllLinear` with bias."""
    m = z_nn.ShardedToAllLinear(3, 4)
    x = array(np.random.randn(2, 3).astype(np.float32))
    out = m(x)
    assert out.shape == (2, 4)


def test_sharded_to_all_no_bias() -> None:
    """Test `ShardedToAllLinear` without bias."""
    m = z_nn.ShardedToAllLinear(3, 4, bias=False)
    x = array(np.random.randn(2, 3).astype(np.float32))
    out = m(x)
    assert out.shape == (2, 4)


def test_quantized_linear() -> None:
    """Test `QuantizedLinear` with bias."""
    m = z_nn.QuantizedLinear(3, 4)
    x = array(np.random.randn(2, 3).astype(np.float32))
    out = m(x)
    assert out.shape == (2, 4)


def test_quantized_linear_no_bias() -> None:
    """Test `QuantizedLinear` without bias."""
    m = z_nn.QuantizedLinear(3, 4, bias=False)
    x = array(np.random.randn(2, 3).astype(np.float32))
    out = m(x)
    assert out.shape == (2, 4)


def test_quantized_all_to_sharded() -> None:
    """Test `QuantizedAllToShardedLinear` with bias."""
    m = z_nn.QuantizedAllToShardedLinear(3, 4)
    x = array(np.random.randn(2, 3).astype(np.float32))
    out = m(x)
    assert out.shape == (2, 4)


def test_quantized_all_to_sharded_no_bias() -> None:
    """Test `QuantizedAllToShardedLinear` without bias."""
    m = z_nn.QuantizedAllToShardedLinear(3, 4, bias=False)
    x = array(np.random.randn(2, 3).astype(np.float32))
    out = m(x)
    assert out.shape == (2, 4)


def test_quantized_sharded_to_all() -> None:
    """Test `QuantizedShardedToAllLinear` with bias."""
    m = z_nn.QuantizedShardedToAllLinear(3, 4)
    x = array(np.random.randn(2, 3).astype(np.float32))
    out = m(x)
    assert out.shape == (2, 4)


def test_quantized_sharded_to_all_no_bias() -> None:
    """Test `QuantizedShardedToAllLinear` without bias."""
    m = z_nn.QuantizedShardedToAllLinear(3, 4, bias=False)
    x = array(np.random.randn(2, 3).astype(np.float32))
    out = m(x)
    assert out.shape == (2, 4)
