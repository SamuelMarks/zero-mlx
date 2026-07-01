import pytest
import numpy as np
import warnings
from unittest.mock import patch

import zero_mlx.nn as z_nn
from zero_mlx.array import array
import ml_switcheroo_compiler.ops as sops
from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig


def _test_func_shape(func, x_shape, expected_shape, **kwargs):
    x = array(np.random.randn(*x_shape).astype(np.float32))

    mock_ret = Tensor(None, TensorConfig(expected_shape, "float32", "cpu"))
    with (
        patch.object(sops, "celu", return_value=mock_ret),
        patch.object(sops, "elu", return_value=mock_ret),
        patch.object(sops, "gelu", return_value=mock_ret),
        patch.object(sops, "glu", return_value=mock_ret),
        patch.object(sops, "leaky_relu", return_value=mock_ret),
        patch.object(sops, "log_sigmoid", return_value=mock_ret),
        patch.object(sops, "log_softmax", return_value=mock_ret),
        patch.object(sops, "relu", return_value=mock_ret),
        patch.object(sops, "relu6", return_value=mock_ret),
        patch.object(sops, "selu", return_value=mock_ret),
        patch.object(sops, "silu", return_value=mock_ret),
        patch.object(sops, "sigmoid", return_value=mock_ret),
        patch.object(sops, "softmax", return_value=mock_ret),
        patch.object(sops, "softplus", return_value=mock_ret),
        patch.object(sops, "softsign", return_value=mock_ret),
        patch.object(sops, "heaviside", return_value=mock_ret),
        patch.object(sops, "tanh", return_value=mock_ret),
        patch.object(sops, "hard_shrink", return_value=mock_ret),
        patch.object(sops, "hard_tanh", return_value=mock_ret),
        patch.object(sops, "hardswish", return_value=mock_ret),
    ):
        out = func(x, **kwargs)
        assert out.shape == expected_shape


def test_celu():
    _test_func_shape(z_nn.celu, (2, 3), (2, 3))
    _test_func_shape(z_nn.CELU(), (2, 3), (2, 3))


def test_elu():
    _test_func_shape(z_nn.elu, (2, 3), (2, 3))
    _test_func_shape(z_nn.ELU(), (2, 3), (2, 3))


def test_gelu():
    _test_func_shape(z_nn.gelu, (2, 3), (2, 3))
    _test_func_shape(z_nn.GELU("fast"), (2, 3), (2, 3))


def test_glu():
    _test_func_shape(z_nn.glu, (2, 3), (2, 3))
    _test_func_shape(z_nn.GLU(), (2, 3), (2, 3))


def test_leaky_relu():
    _test_func_shape(z_nn.leaky_relu, (2, 3), (2, 3))
    _test_func_shape(z_nn.LeakyReLU(), (2, 3), (2, 3))


def test_log_sigmoid():
    _test_func_shape(z_nn.log_sigmoid, (2, 3), (2, 3))
    _test_func_shape(z_nn.LogSigmoid(), (2, 3), (2, 3))


def test_log_softmax():
    _test_func_shape(z_nn.log_softmax, (2, 3), (2, 3))
    _test_func_shape(z_nn.LogSoftmax(), (2, 3), (2, 3))


def test_mish():
    m_ret = Tensor(None, TensorConfig((2, 3), "float32", "cpu"))
    with (
        patch.object(sops, "softplus", return_value=m_ret),
        patch.object(sops, "tanh", return_value=m_ret),
        patch.object(sops, "multiply", return_value=m_ret, create=True),
    ):
        _test_func_shape(z_nn.mish, (2, 3), (2, 3))
        _test_func_shape(z_nn.Mish(), (2, 3), (2, 3))


def test_prelu():
    m = z_nn.PReLU(3)
    x = array(np.random.randn(2, 3).astype(np.float32))
    m_ret = Tensor(None, TensorConfig((2, 3), "float32", "cpu"))
    with (
        patch.object(sops, "relu", return_value=m_ret),
        patch.object(sops, "multiply", return_value=m_ret, create=True),
        patch.object(sops, "minimum", return_value=m_ret, create=True),
        patch.object(sops, "zeros_like", return_value=m_ret, create=True),
        patch.object(sops, "add", return_value=m_ret, create=True),
    ):
        out = m(x)
        assert out.shape == (2, 3)


def test_relu():
    _test_func_shape(z_nn.relu, (2, 3), (2, 3))
    _test_func_shape(z_nn.ReLU(), (2, 3), (2, 3))


def test_relu2():
    m = z_nn.ReLU2()
    x = array(np.random.randn(2, 3).astype(np.float32))
    m_ret = Tensor(None, TensorConfig((2, 3), "float32", "cpu"))
    with (
        patch.object(sops, "maximum", return_value=m_ret, create=True),
        patch.object(sops, "minimum", return_value=m_ret, create=True),
        patch.object(sops, "zeros_like", return_value=m_ret, create=True),
        patch.object(sops, "full_like", return_value=m_ret, create=True),
    ):
        out = m(x)
        assert out.shape == (2, 3)
        out2 = z_nn.relu2(x)
        assert out2.shape == (2, 3)


def test_relu6():
    _test_func_shape(z_nn.relu6, (2, 3), (2, 3))
    _test_func_shape(z_nn.ReLU6(), (2, 3), (2, 3))


def test_selu():
    _test_func_shape(z_nn.selu, (2, 3), (2, 3))
    _test_func_shape(z_nn.SELU(), (2, 3), (2, 3))


def test_silu():
    _test_func_shape(z_nn.silu, (2, 3), (2, 3))
    _test_func_shape(z_nn.SiLU(), (2, 3), (2, 3))


def test_sigmoid():
    _test_func_shape(z_nn.sigmoid, (2, 3), (2, 3))
    _test_func_shape(z_nn.Sigmoid(), (2, 3), (2, 3))


def test_softmax():
    _test_func_shape(z_nn.softmax, (2, 3), (2, 3))
    _test_func_shape(z_nn.Softmax(), (2, 3), (2, 3))


def test_softmin():
    m = z_nn.Softmin()
    x = array(np.random.randn(2, 3).astype(np.float32))
    m_ret = Tensor(None, TensorConfig((2, 3), "float32", "cpu"))
    with (
        patch.object(sops, "negative", return_value=m_ret),
        patch.object(sops, "softmax", return_value=m_ret),
    ):
        out = m(x)
        assert out.shape == (2, 3)
        out2 = z_nn.softmin(x)
        assert out2.shape == (2, 3)


def test_softplus():
    _test_func_shape(z_nn.softplus, (2, 3), (2, 3))
    _test_func_shape(z_nn.Softplus(), (2, 3), (2, 3))


def test_softshrink():
    m = z_nn.Softshrink()
    x = array(np.random.randn(2, 3).astype(np.float32))
    m_ret = Tensor(None, TensorConfig((2, 3), "float32", "cpu"))
    with (
        patch.object(sops, "full_like", return_value=m_ret, create=True),
        patch.object(sops, "greater", return_value=m_ret, create=True),
        patch.object(sops, "less", return_value=m_ret, create=True),
        patch.object(sops, "subtract", return_value=m_ret),
        patch.object(sops, "add", return_value=m_ret, create=True),
        patch.object(sops, "multiply", return_value=m_ret, create=True),
    ):
        out = m(x)
        assert out.shape == (2, 3)
        out2 = z_nn.softshrink(x)
        assert out2.shape == (2, 3)


def test_softsign():
    _test_func_shape(z_nn.softsign, (2, 3), (2, 3))
    _test_func_shape(z_nn.Softsign(), (2, 3), (2, 3))


def test_step():
    m_ret = Tensor(None, TensorConfig((2, 3), "float32", "cpu"))
    with (
        patch.object(sops, "full_like", return_value=m_ret, create=True),
        patch.object(sops, "heaviside", return_value=m_ret),
    ):
        _test_func_shape(z_nn.step, (2, 3), (2, 3))
        _test_func_shape(z_nn.Step(), (2, 3), (2, 3))


def test_tanh():
    _test_func_shape(z_nn.tanh, (2, 3), (2, 3))
    _test_func_shape(z_nn.Tanh(), (2, 3), (2, 3))


def test_hard_shrink():
    # If the backend has hard_shrink:
    _test_func_shape(z_nn.hard_shrink, (2, 3), (2, 3))
    _test_func_shape(z_nn.HardShrink(), (2, 3), (2, 3))

    # Fallback path if backend doesn't have hard_shrink
    m = z_nn.HardShrink()
    x = array(np.random.randn(2, 3).astype(np.float32))
    m_ret = Tensor(None, TensorConfig((2, 3), "float32", "cpu"))
    with (
        patch("builtins.hasattr", return_value=False),
        patch.object(sops, "full_like", return_value=m_ret, create=True),
        patch.object(sops, "greater", return_value=m_ret, create=True),
        patch.object(sops, "less", return_value=m_ret, create=True),
        patch.object(sops, "logical_or", return_value=m_ret, create=True),
        patch.object(sops, "zeros_like", return_value=m_ret, create=True),
        patch.object(sops, "where", return_value=m_ret, create=True),
    ):
        out = z_nn.hard_shrink(x)
        assert out.shape == (2, 3)


def test_hard_tanh():
    _test_func_shape(z_nn.hard_tanh, (2, 3), (2, 3))
    _test_func_shape(z_nn.HardTanh(), (2, 3), (2, 3))

    m = z_nn.HardTanh()
    x = array(np.random.randn(2, 3).astype(np.float32))
    m_ret = Tensor(None, TensorConfig((2, 3), "float32", "cpu"))
    with (
        patch("builtins.hasattr", return_value=False),
        patch.object(sops, "full_like", return_value=m_ret, create=True),
        patch.object(sops, "maximum", return_value=m_ret, create=True),
        patch.object(sops, "minimum", return_value=m_ret, create=True),
    ):
        out = z_nn.hard_tanh(x)
        assert out.shape == (2, 3)


def test_hardswish():
    _test_func_shape(z_nn.hardswish, (2, 3), (2, 3))
    _test_func_shape(z_nn.Hardswish(), (2, 3), (2, 3))

    m = z_nn.Hardswish()
    x = array(np.random.randn(2, 3).astype(np.float32))
    m_ret = Tensor(None, TensorConfig((2, 3), "float32", "cpu"))
    with (
        patch("builtins.hasattr", return_value=False),
        patch.object(sops, "full_like", return_value=m_ret, create=True),
        patch.object(sops, "add", return_value=m_ret, create=True),
        patch.object(sops, "relu6", return_value=m_ret, create=True),
        patch.object(sops, "multiply", return_value=m_ret, create=True),
        patch.object(sops, "divide", return_value=m_ret, create=True),
    ):
        out = z_nn.hardswish(x)
        assert out.shape == (2, 3)
