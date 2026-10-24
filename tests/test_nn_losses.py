"""Unit tests for `zero_mlx.nn.losses` loss functions."""

from __future__ import annotations

import mlx.core as mx
import mlx.nn.losses as m_losses
import numpy as np

from zero_mlx.array import array
import zero_mlx.nn.losses as z_losses


def test_cosine_similarity_loss() -> None:
    """Test `cosine_similarity_loss` output shape and numerical values."""
    x1_np = np.array([[1.0, 0.0, 0.0], [0.0, 2.0, 0.0]], dtype=np.float32)
    x2_np = np.array([[1.0, 0.0, 0.0], [0.0, -2.0, 0.0]], dtype=np.float32)
    loss = z_losses.cosine_similarity_loss(array(x1_np), array(x2_np))
    assert loss.shape == (2,)
    np.testing.assert_allclose(loss.tolist(), [0.0, 2.0], rtol=1e-5, atol=1e-5)


def test_gaussian_nll_loss() -> None:
    """Test `gaussian_nll_loss` output shape and numerical parity with `mlx.nn.losses`."""
    i_np = np.random.randn(2, 3).astype(np.float32)
    t_np = np.random.randn(2, 3).astype(np.float32)
    v_np = np.abs(np.random.randn(2, 3)).astype(np.float32) + 0.1
    loss = z_losses.gaussian_nll_loss(array(i_np), array(t_np), array(v_np), reduction="none")
    m_loss = m_losses.gaussian_nll_loss(
        mx.array(i_np), mx.array(t_np), mx.array(v_np), reduction="none"
    )
    assert loss.shape == (2, 3)
    np.testing.assert_allclose(loss.tolist(), np.array(m_loss), rtol=1e-4, atol=1e-4)


def test_hinge_loss() -> None:
    """Test `hinge_loss` output shape and numerical parity with `mlx.nn.losses`."""
    i_np = np.random.randn(2, 3).astype(np.float32)
    t_np = np.ones((2, 3), dtype=np.float32)
    loss = z_losses.hinge_loss(array(i_np), array(t_np), reduction="none")
    m_loss = m_losses.hinge_loss(mx.array(i_np), mx.array(t_np), reduction="none")
    assert loss.shape == (2, 3)
    np.testing.assert_allclose(loss.tolist(), np.array(m_loss), rtol=1e-5, atol=1e-5)


def test_huber_loss() -> None:
    """Test `huber_loss` output shape and numerical parity with `mlx.nn.losses`."""
    i_np = np.random.randn(2, 3).astype(np.float32)
    t_np = np.random.randn(2, 3).astype(np.float32)
    loss = z_losses.huber_loss(array(i_np), array(t_np), reduction="none")
    m_loss = m_losses.huber_loss(mx.array(i_np), mx.array(t_np), reduction="none")
    assert loss.shape == (2, 3)
    np.testing.assert_allclose(loss.tolist(), np.array(m_loss), rtol=1e-5, atol=1e-5)


def test_kl_div_loss() -> None:
    """Test `kl_div_loss` output shape and numerical values across reductions."""
    i_np = np.log(np.array([[0.5, 0.5], [0.25, 0.75]], dtype=np.float32))
    t_np = np.array([[0.5, 0.5], [0.25, 0.75]], dtype=np.float32)
    i = array(i_np)
    t = array(t_np)
    assert z_losses.kl_div_loss(i, t, reduction="none").shape == (2, 2)
    np.testing.assert_allclose(
        z_losses.kl_div_loss(i, t, reduction="mean").item(),
        0.0,
        atol=1e-5,
    )
    assert z_losses.kl_div_loss(i, t, reduction="sum").ndim == 0
