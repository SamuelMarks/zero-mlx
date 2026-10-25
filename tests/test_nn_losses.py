"""Unit tests for `zero_mlx.nn.losses` loss functions."""

from __future__ import annotations

import zero_mlx as mx
from zero_mlx.array import array
import zero_mlx.nn.losses as z_losses


def test_cosine_similarity_loss() -> None:
    """Test `cosine_similarity_loss` output shape and numerical values."""
    x1 = array([[1.0, 0.0, 0.0], [0.0, 2.0, 0.0]], dtype=mx.float32)
    x2 = array([[1.0, 0.0, 0.0], [0.0, -2.0, 0.0]], dtype=mx.float32)
    loss = z_losses.cosine_similarity_loss(x1, x2)
    assert loss.shape == (2,)
    assert bool(mx.allclose(loss, array([0.0, 2.0]), rtol=1e-5, atol=1e-5).item()) is True


def test_gaussian_nll_loss() -> None:
    """Test `gaussian_nll_loss` output shape and analytical numerical parity."""
    i = array([[0.5, -0.5, 1.0], [1.5, -1.0, 0.0]], dtype=mx.float32)
    t = array([[0.2, -0.3, 0.8], [1.0, -0.5, 0.2]], dtype=mx.float32)
    v = array([[0.5, 0.8, 1.2], [0.3, 0.9, 1.5]], dtype=mx.float32)
    loss = z_losses.gaussian_nll_loss(i, t, v, reduction="none")
    v_clamped = mx.maximum(v, 1e-6)
    expected = 0.5 * (mx.log(v_clamped) + ((t - i) ** 2.0) / v_clamped)
    assert loss.shape == (2, 3)
    assert bool(mx.allclose(loss, expected, rtol=1e-4, atol=1e-4).item()) is True


def test_hinge_loss() -> None:
    """Test `hinge_loss` output shape and analytical numerical parity."""
    i = array([[0.5, -0.5, 1.2], [-1.0, 0.2, 2.0]], dtype=mx.float32)
    t = mx.ones((2, 3), dtype=mx.float32)
    loss = z_losses.hinge_loss(i, t, reduction="none")
    expected = mx.maximum(1.0 - i * t, 0.0)
    assert loss.shape == (2, 3)
    assert bool(mx.allclose(loss, expected, rtol=1e-5, atol=1e-5).item()) is True


def test_huber_loss() -> None:
    """Test `huber_loss` output shape and analytical numerical parity."""
    i = array([[0.5, -1.5, 0.2], [2.0, -0.3, 1.0]], dtype=mx.float32)
    t = array([[0.0, 0.0, 0.0], [0.5, 0.0, -0.5]], dtype=mx.float32)
    loss = z_losses.huber_loss(i, t, reduction="none")
    abs_err = mx.abs(i - t)
    expected = mx.where(abs_err <= 1.0, 0.5 * (abs_err**2.0), abs_err - 0.5)
    assert loss.shape == (2, 3)
    assert bool(mx.allclose(loss, expected, rtol=1e-5, atol=1e-5).item()) is True


def test_kl_div_loss() -> None:
    """Test `kl_div_loss` output shape and numerical values across reductions."""
    t = array([[0.5, 0.5], [0.25, 0.75]], dtype=mx.float32)
    i = mx.log(t)
    assert z_losses.kl_div_loss(i, t, reduction="none").shape == (2, 2)
    mean_loss = float(z_losses.kl_div_loss(i, t, reduction="mean").item().real)
    assert abs(mean_loss) < 1e-5
    assert z_losses.kl_div_loss(i, t, reduction="sum").ndim == 0
