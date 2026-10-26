"""Unit tests for `RNN`, `GRU`, `LSTM`, and `TransformerEncoderLayer`."""

from __future__ import annotations

import math

import zero_mlx as mx
from zero_mlx.array import array
import zero_mlx.nn as z_nn


def test_rnn() -> None:
    """Test `RNN` forward pass computes exact `tanh(x @ W_ih.T + h @ W_hh.T + b)` recurrence."""
    m = z_nn.RNN(1, 1, bias=False)
    m.weight_ih = array([[1.0]], dtype=mx.float32)
    m.weight_hh = array([[0.5]], dtype=mx.float32)
    x = array([[[1.0], [1.0]]], dtype=mx.float32)
    out, h = m(x)
    h1 = math.tanh(1.0)
    h2 = math.tanh(1.0 + 0.5 * h1)
    assert out.shape == (1, 2, 1)
    assert h.shape == (1, 1)
    assert bool(mx.allclose(out, array([[[h1], [h2]]]), rtol=1e-5, atol=1e-5).item()) is True
    assert abs(float(h.item().real) - h2) < 1e-5


def test_gru() -> None:
    """Test `GRU` forward pass with zero weights yields zero hidden state."""
    m = z_nn.GRU(2, 2, bias=False)
    m.weight_ih = mx.zeros((6, 2), dtype=mx.float32)
    m.weight_hh = mx.zeros((6, 2), dtype=mx.float32)
    x = mx.ones((1, 3, 2), dtype=mx.float32)
    out, h = m(x)
    assert out.shape == (1, 3, 2)
    assert h.shape == (1, 2)
    assert bool(mx.allclose(out, mx.zeros((1, 3, 2)), rtol=1e-5, atol=1e-5).item()) is True

    m_bias = z_nn.GRU(2, 2, bias=True)
    out_b, h_b = m_bias(x)
    assert out_b.shape == (1, 3, 2) and h_b.shape == (1, 2)


def test_transformer_encoder_layer() -> None:
    """Test `TransformerEncoderLayer`, `TransformerEncoder`, and `TransformerDecoder` with checkpointing."""
    m = z_nn.TransformerEncoderLayer(4, 2)
    x = array([[[1.0, -1.0, 0.5, -0.5], [0.2, 0.4, -0.2, -0.4]]], dtype=mx.float32)
    out = m(x)
    assert out.shape == (1, 2, 4)
    assert bool(mx.all(mx.isfinite(out)).item()) is True
    # Verify MultiHeadAttention uses initialized Linear projections rather than all-zero weights
    attn_out = m.attention(x, x, x)
    assert float(mx.sum(mx.abs(attn_out)).item().real) > 0.0

    enc_ckpt = z_nn.TransformerEncoder(1, 4, 2, checkpoint=True)
    assert enc_ckpt(x).shape == (1, 2, 4)
    dec_ckpt = z_nn.TransformerDecoder(1, 4, 2, checkpoint=True)
    assert dec_ckpt(x, x).shape == (1, 2, 4)
