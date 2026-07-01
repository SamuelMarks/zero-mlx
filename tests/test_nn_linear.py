import pytest
import numpy as np

import zero_mlx.nn as z_nn
from zero_mlx.array import array


def test_linear():
    m = z_nn.Linear(3, 4)
    x = array(np.random.randn(2, 3).astype(np.float32))
    out = m(x)
    assert out.shape == (2, 4)


def test_linear_no_bias():
    m = z_nn.Linear(3, 4, bias=False)
    x = array(np.random.randn(2, 3).astype(np.float32))
    out = m(x)
    assert out.shape == (2, 4)


def test_bilinear():
    m = z_nn.Bilinear(3, 2, 4)
    x1 = array(np.random.randn(5, 3).astype(np.float32))
    x2 = array(np.random.randn(5, 2).astype(np.float32))
    out = m(x1, x2)
    assert out.shape == (5, 4)


def test_bilinear_no_bias():
    m = z_nn.Bilinear(3, 2, 4, bias=False)
    x1 = array(np.random.randn(5, 3).astype(np.float32))
    x2 = array(np.random.randn(5, 2).astype(np.float32))
    out = m(x1, x2)
    assert out.shape == (5, 4)


def test_all_to_sharded():
    m = z_nn.AllToShardedLinear(3, 4)
    x = array(np.random.randn(2, 3).astype(np.float32))
    out = m(x)
    assert out.shape == (2, 4)


def test_all_to_sharded_no_bias():
    m = z_nn.AllToShardedLinear(3, 4, bias=False)
    x = array(np.random.randn(2, 3).astype(np.float32))
    out = m(x)
    assert out.shape == (2, 4)


def test_sharded_to_all():
    m = z_nn.ShardedToAllLinear(3, 4)
    x = array(np.random.randn(2, 3).astype(np.float32))
    out = m(x)
    assert out.shape == (2, 4)


def test_sharded_to_all_no_bias():
    m = z_nn.ShardedToAllLinear(3, 4, bias=False)
    x = array(np.random.randn(2, 3).astype(np.float32))
    out = m(x)
    assert out.shape == (2, 4)


def test_quantized_linear():
    m = z_nn.QuantizedLinear(3, 4)
    x = array(np.random.randn(2, 3).astype(np.float32))
    out = m(x)
    assert out.shape == (2, 4)


def test_quantized_linear_no_bias():
    m = z_nn.QuantizedLinear(3, 4, bias=False)
    x = array(np.random.randn(2, 3).astype(np.float32))
    out = m(x)
    assert out.shape == (2, 4)


def test_quantized_all_to_sharded():
    m = z_nn.QuantizedAllToShardedLinear(3, 4)
    x = array(np.random.randn(2, 3).astype(np.float32))
    out = m(x)
    assert out.shape == (2, 4)


def test_quantized_all_to_sharded_no_bias():
    m = z_nn.QuantizedAllToShardedLinear(3, 4, bias=False)
    x = array(np.random.randn(2, 3).astype(np.float32))
    out = m(x)
    assert out.shape == (2, 4)


def test_quantized_sharded_to_all():
    m = z_nn.QuantizedShardedToAllLinear(3, 4)
    x = array(np.random.randn(2, 3).astype(np.float32))
    out = m(x)
    assert out.shape == (2, 4)


def test_quantized_sharded_to_all_no_bias():
    m = z_nn.QuantizedShardedToAllLinear(3, 4, bias=False)
    x = array(np.random.randn(2, 3).astype(np.float32))
    out = m(x)
    assert out.shape == (2, 4)
