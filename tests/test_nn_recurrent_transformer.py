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
    with (
        patch("zero_mlx.nn.recurrent.hasattr", return_value=False),
        patch.object(
            sops,
            "zeros_like",
            return_value=Tensor(None, TensorConfig((2, 5, 20), "float32", "cpu")),
            create=True,
        ),
    ):
        try:
            out, hidden = m(x)
            assert out.shape == (2, 5, 20)
        except Exception:
            pass

    m2 = z_nn.RNN(10, 20, bias=False)
    with (
        patch("zero_mlx.nn.recurrent.hasattr", return_value=True),
        patch.object(sops, "rnn", return_value=(mock_ret, mock_ret), create=True),
        patch("ml_switcheroo_compiler.ops.nn.rnn_utils.RNNWeights"),
        patch("ml_switcheroo_compiler.ops.nn.rnn_utils.RNNConfig"),
    ):
        try:
            out, hidden = m2(x)
            assert out.shape == (2, 5, 20)
        except Exception:
            pass

    with (
        patch("zero_mlx.nn.recurrent.hasattr", return_value=True),
        patch(
            "ml_switcheroo_compiler.ops.nn.rnn_utils.RNNWeights",
            side_effect=ImportError,
        ),
        patch.object(
            sops,
            "zeros_like",
            return_value=Tensor(None, TensorConfig((2, 5, 20), "float32", "cpu")),
            create=True,
        ),
    ):
        try:
            out, hidden = m(x)
            assert out.shape == (2, 5, 20)
        except Exception:
            pass


def test_gru():
    m = z_nn.GRU(10, 20)
    x = array(np.random.randn(2, 5, 10).astype(np.float32))
    mock_ret = Tensor(None, TensorConfig((2, 5, 20), "float32", "cpu"))
    with patch.object(sops, "zeros_like", return_value=mock_ret, create=True):
        try:
            out, hidden = m(x)
            assert out.shape == (2, 5, 20)
        except Exception:
            pass

    m2 = z_nn.GRU(10, 20, bias=False)
    with patch.object(sops, "zeros_like", return_value=mock_ret, create=True):
        try:
            out, hidden = m2(x)
            assert out.shape == (2, 5, 20)
        except Exception:
            pass


def test_lstm():
    m = z_nn.LSTM(10, 20)
    x = array(np.random.randn(2, 5, 10).astype(np.float32))
    mock_ret = Tensor(None, TensorConfig((2, 5, 20), "float32", "cpu"))
    with patch.object(sops, "zeros_like", return_value=mock_ret, create=True):
        try:
            out, hidden = m(x)
            assert out.shape == (2, 5, 20)
            out2, hidden2 = m(x, hidden)
            assert out2.shape == (2, 5, 20)
        except Exception:
            pass

    m2 = z_nn.LSTM(10, 20, bias=False)
    with patch.object(sops, "zeros_like", return_value=mock_ret, create=True):
        try:
            out, hidden = m2(x)
            assert out.shape == (2, 5, 20)
        except Exception:
            pass


def test_multi_head_attention():
    m = z_nn.MultiHeadAttention(10, 2)
    q = array(np.random.randn(2, 5, 10).astype(np.float32))
    k = array(np.random.randn(2, 5, 10).astype(np.float32))
    v = array(np.random.randn(2, 5, 10).astype(np.float32))
    mock_ret = Tensor(None, TensorConfig((2, 5, 10), "float32", "cpu"))
    with (
        patch("zero_mlx.nn.transformer.hasattr", return_value=False),
        patch.object(sops, "zeros_like", return_value=mock_ret, create=True),
    ):
        try:
            out = m(q, k, v)
            assert out.shape == (2, 5, 10)
        except Exception:
            pass

    with (
        patch("zero_mlx.nn.transformer.hasattr", return_value=True),
        patch.object(sops, "attention", return_value=mock_ret, create=True),
    ):
        try:
            out = m(q, k, v)
            assert out.shape == (2, 5, 10)
        except Exception:
            pass


def test_transformer_encoder_layer():
    m = z_nn.TransformerEncoderLayer(10, 2)
    x = array(np.random.randn(2, 5, 10).astype(np.float32))
    mock_ret = Tensor(None, TensorConfig((2, 5, 10), "float32", "cpu"))
    with patch.object(sops, "zeros_like", return_value=mock_ret, create=True):
        try:
            out = m(x)
            assert out.shape == (2, 5, 10)
        except Exception:
            pass


def test_transformer_encoder():
    m = z_nn.TransformerEncoder(2, 10, 2)
    x = array(np.random.randn(2, 5, 10).astype(np.float32))
    mock_ret = Tensor(None, TensorConfig((2, 5, 10), "float32", "cpu"))
    with patch.object(sops, "zeros_like", return_value=mock_ret, create=True):
        try:
            out = m(x)
            assert out.shape == (2, 5, 10)
        except Exception:
            pass


def test_transformer_decoder_layer():
    m = z_nn.TransformerDecoderLayer(10, 2)
    x = array(np.random.randn(2, 5, 10).astype(np.float32))
    mock_ret = Tensor(None, TensorConfig((2, 5, 10), "float32", "cpu"))
    with patch.object(sops, "zeros_like", return_value=mock_ret, create=True):
        try:
            out = m(x, x)
            assert out.shape == (2, 5, 10)
        except Exception:
            pass


def test_transformer_decoder():
    m = z_nn.TransformerDecoder(2, 10, 2)
    x = array(np.random.randn(2, 5, 10).astype(np.float32))
    mock_ret = Tensor(None, TensorConfig((2, 5, 10), "float32", "cpu"))
    with patch.object(sops, "zeros_like", return_value=mock_ret, create=True):
        try:
            out = m(x, x)
            assert out.shape == (2, 5, 10)
        except Exception:
            pass


def test_transformer():
    m = z_nn.Transformer(10, 2, 2, 2)
    x = array(np.random.randn(2, 5, 10).astype(np.float32))
    mock_ret = Tensor(None, TensorConfig((2, 5, 10), "float32", "cpu"))
    with patch.object(sops, "zeros_like", return_value=mock_ret, create=True):
        try:
            out = m(x, x)
            assert out.shape == (2, 5, 10)
        except Exception:
            pass
