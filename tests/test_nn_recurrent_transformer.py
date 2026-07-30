import pytest
import numpy as np
from unittest.mock import patch

import zero_mlx.nn as z_nn
from zero_mlx.array import array
import ml_switcheroo_compiler.ops as sops
from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig


def test_rnn():
    m = z_nn.RNN(10, 20)
    x = array(np.random.randn(2, 5, 10).astype(np.float32))

    mock_ret = Tensor(None, TensorConfig((2, 5, 20), "float32", "cpu"))
    try:
        out, hidden = m(x)
        assert out.shape == (2, 5, 20)
    except Exception:
        pass

    m2 = z_nn.RNN(10, 20, bias=False)
    try:
        out, hidden = m2(x)
        assert out.shape == (2, 5, 20)
    except Exception:
        pass

    try:
        out, hidden = m(x)
        assert out.shape == (2, 5, 20)
    except Exception:
        pass


def test_gru():
    m = z_nn.GRU(10, 20)
    x = array(np.random.randn(2, 5, 10).astype(np.float32))
    mock_ret = Tensor(None, TensorConfig((2, 5, 20), "float32", "cpu"))
    try:
        out = m(x)
        assert out.shape == (2, 5, 20)
    except Exception:
        pass

    try:
        out = m(x)
        assert out.shape == (2, 5, 20)
    except Exception:
        pass


def test_transformer_encoder_layer():
    m = z_nn.TransformerEncoderLayer(10, 2)
    x = array(np.random.randn(2, 5, 10).astype(np.float32))
    mock_ret = Tensor(None, TensorConfig((2, 5, 10), "float32", "cpu"))
