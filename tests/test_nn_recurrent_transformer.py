"""Unit tests for `RNN`, `GRU`, `LSTM`, and `TransformerEncoderLayer`."""

from __future__ import annotations

import numpy as np

from zero_mlx.array import array
import zero_mlx.nn as z_nn


def test_rnn() -> None:
    """Test `RNN` forward pass on 3D batched sequence input."""
    m = z_nn.RNN(3, 4)
    x = array(np.random.randn(2, 5, 3).astype(np.float32))
    out, h = m(x)
    assert out.shape == (2, 5, 4)
    assert h.shape == (2, 4)


def test_gru() -> None:
    """Test `GRU` forward pass on 3D batched sequence input."""
    m = z_nn.GRU(3, 4)
    x = array(np.random.randn(2, 5, 3).astype(np.float32))
    out, h = m(x)
    assert out.shape == (2, 5, 4)
    assert h.shape == (2, 4)


def test_transformer_encoder_layer() -> None:
    """Test `TransformerEncoderLayer` forward pass."""
    m = z_nn.TransformerEncoderLayer(10, 2)
    x = array(np.random.randn(2, 5, 10).astype(np.float32))
    out = m(x)
    assert out.shape == (2, 5, 10)
