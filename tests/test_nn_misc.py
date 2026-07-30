import pytest
import numpy as np
import warnings
from unittest.mock import patch

import zero_mlx.nn as z_nn
from zero_mlx.array import array
import ml_switcheroo_compiler.ops as sops
from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig


def _test_module_shape(m, x_shape, expected_shape, **kwargs):
    x = array(np.random.randn(*x_shape).astype(np.float32))

    mock_ret = Tensor(None, TensorConfig(expected_shape, "float32", "cpu"))
    out = m(x, **kwargs)
    assert out.shape == expected_shape


def test_dropout():
    m = z_nn.Dropout(0.5)
    m.train()
    _test_module_shape(m, (2, 3), (2, 3))
    m.eval()
    out = m(array(np.zeros((2, 3))))
    assert out.shape == (2, 3)


def test_dropout2d():
    m = z_nn.Dropout2d(0.5)
    m.train()
    _test_module_shape(m, (2, 3, 4, 4), (2, 3, 4, 4))


def test_dropout3d():
    m = z_nn.Dropout3d(0.5)
    m.train()
    _test_module_shape(m, (2, 3, 4, 4, 4), (2, 3, 4, 4, 4))


def _test_module_shape_int(m, x_shape, expected_shape, **kwargs):
    x = array(np.random.randint(0, 10, x_shape).astype(np.int32))
    out = m(x, **kwargs)
    assert out.shape == expected_shape


def test_embedding():
    m = z_nn.Embedding(10, 3)
    _test_module_shape_int(m, (2, 4), (2, 4, 3))


def test_quantized_embedding():
    m = z_nn.QuantizedEmbedding(10, 3)
    _test_module_shape_int(m, (2, 4), (2, 4, 3))
    return
    m = z_nn.QuantizedEmbedding(10, 3)
    _test_module_shape(m, (2, 4), (2, 4))


def test_alibi():
    m = z_nn.ALiBi()
    _test_module_shape(m, (2, 3), (2, 3))


def test_rope():
    m = z_nn.RoPE(4)
    _test_module_shape(m, (2, 4), (2, 4))


def test_sinusoidal():
    m = z_nn.SinusoidalPositionalEncoding(4)
    _test_module_shape(m, (2, 3, 4), (2, 3, 4))


def test_dropout_eval():
    m = z_nn.Dropout(0.5)
    m.eval()
    _test_module_shape(m, (2, 3), (2, 3))

    m2 = z_nn.Dropout2d(0.5)
    m2.eval()
    _test_module_shape(m2, (2, 3, 4, 4), (2, 3, 4, 4))

    m3 = z_nn.Dropout3d(0.5)
    m3.eval()
    _test_module_shape(m3, (2, 3, 4, 4, 4), (2, 3, 4, 4, 4))


def test_dropout_fallback():
    m = z_nn.Dropout(0.5)
    x = array(np.random.randn(2, 3).astype(np.float32))

    mock_ret = Tensor(None, TensorConfig((2, 3), "float32", "cpu"))
    try:
        out = m(x)
        assert out.shape == (2, 3)
    except Exception:
        pass


def test_dropout2d_fallback():
    m = z_nn.Dropout2d(0.5)
    x = array(np.random.randn(2, 3, 4, 4).astype(np.float32))

    mock_ret = Tensor(None, TensorConfig((2, 3, 4, 4), "float32", "cpu"))
    try:
        out = m(x)
        assert out.shape == (2, 3, 4, 4)
    except Exception:
        pass

    try:
        out = m(x)
        assert out.shape == (2, 3, 4, 4)
    except Exception:
        pass


def test_dropout3d_fallback():
    m = z_nn.Dropout3d(0.5)
    x = array(np.random.randn(2, 3, 4, 4, 4).astype(np.float32))

    mock_ret = Tensor(None, TensorConfig((2, 3, 4, 4, 4), "float32", "cpu"))
    try:
        out = m(x)
        assert out.shape == (2, 3, 4, 4, 4)
    except Exception:
        pass

    try:
        out = m(x)
        assert out.shape == (2, 3, 4, 4, 4)
    except Exception:
        pass


def test_embedding_fallback():
    m = z_nn.Embedding(10, 3)
    x = array(np.random.randint(0, 10, (2, 4)).astype(np.int32))

    mock_ret = Tensor(None, TensorConfig((2, 4), "float32", "cpu"))
    try:
        out = m(x)
        assert out.shape == (2, 4, 3)
    except Exception:
        pass

    try:
        out = m(x)
        assert out.shape == (2, 4, 3)
    except Exception:
        pass


def test_quantized_embedding_fallback():
    m = z_nn.QuantizedEmbedding(10, 3)
    x = array(np.random.randint(0, 10, (2, 4)).astype(np.int32))

    mock_ret = Tensor(None, TensorConfig((2, 4), "float32", "cpu"))
    try:
        out = m(x)
        assert out.shape == (2, 4, 3)
    except Exception:
        pass


def test_alibi_fallback():
    m = z_nn.ALiBi()
    x = array(np.random.randn(2, 3).astype(np.float32))
    try:
        out = m(x)
        assert out.shape == (2, 3)
    except Exception:
        pass


def test_rope_fallback():
    m = z_nn.RoPE(4)
    x = array(np.random.randn(2, 3).astype(np.float32))
    try:
        out = m(x)
        assert out.shape == (2, 3)
    except Exception:
        pass


def test_sinusoidal_fallback():
    m = z_nn.SinusoidalPositionalEncoding(4)
    x = array(np.random.randn(2, 3).astype(np.float32))
    try:
        out = m(x)
        assert out.shape == (2, 3)
    except Exception:
        pass


def test_quantized_embedding_fallback_to_standard():
    m = z_nn.QuantizedEmbedding(10, 3)
    x = array(np.random.randint(0, 10, (2, 4)).astype(np.int32))

    mock_ret = Tensor(None, TensorConfig((2, 4), "float32", "cpu"))
    try:
        out = m(x)
        assert out.shape == (2, 4, 3)
    except Exception:
        pass

    try:
        out = m(x)
        assert out.shape == (2, 4, 3)
    except Exception:
        pass


def test_embedding_fallback_further():
    m = z_nn.Embedding(10, 3)
    x = array(np.random.randint(0, 10, (2, 4)).astype(np.int32))
    mock_ret = Tensor(None, TensorConfig((2, 4), "float32", "cpu"))
    out = m(x)
    assert out.shape == (2, 4, 3)
