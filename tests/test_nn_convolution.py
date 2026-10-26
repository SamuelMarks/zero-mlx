"""Unit tests for `Conv1d`, `Conv2d`, `Conv3d`, `ConvTranspose1d`, `ConvTranspose2d`, and `ConvTranspose3d`."""

from __future__ import annotations

import zero_mlx as mx
from zero_mlx.array import array
import zero_mlx.nn as z_nn


def test_conv1d() -> None:
    """Test `Conv1d` with deterministic kernel and bias values."""
    m = z_nn.Conv1d(1, 1, 2, bias=True)
    m.weight = mx.ones((1, 2, 1), dtype=mx.float32)
    m.bias = array([0.5], dtype=mx.float32)
    x = array([[[1.0], [2.0], [3.0]]], dtype=mx.float32)
    out = m(x)
    assert out.shape == (1, 2, 1)
    assert bool(mx.allclose(out, array([[[3.5], [5.5]]]), rtol=1e-5, atol=1e-5).item()) is True


def test_conv1d_no_bias() -> None:
    """Test `Conv1d` without bias on deterministic sum kernel."""
    m = z_nn.Conv1d(1, 1, 2, bias=False)
    m.weight = mx.ones((1, 2, 1), dtype=mx.float32)
    x = array([[[1.0], [2.0], [3.0]]], dtype=mx.float32)
    out = m(x)
    assert out.shape == (1, 2, 1)
    assert bool(mx.allclose(out, array([[[3.0], [5.0]]]), rtol=1e-5, atol=1e-5).item()) is True


def test_conv2d() -> None:
    """Test `Conv2d` with bias on a 2x2 all-ones kernel."""
    m = z_nn.Conv2d(1, 1, 2, bias=True)
    m.weight = mx.ones((1, 2, 2, 1), dtype=mx.float32)
    m.bias = array([1.0], dtype=mx.float32)
    x = mx.ones((1, 3, 3, 1), dtype=mx.float32) * 2.0
    out = m(x)
    assert out.shape == (1, 2, 2, 1)
    assert bool(mx.allclose(out, mx.full((1, 2, 2, 1), 9.0), rtol=1e-5, atol=1e-5).item()) is True


def test_conv2d_no_bias() -> None:
    """Test `Conv2d` without bias on a 2x2 all-ones kernel."""
    m = z_nn.Conv2d(1, 1, 2, bias=False)
    m.weight = mx.ones((1, 2, 2, 1), dtype=mx.float32)
    x = mx.ones((1, 3, 3, 1), dtype=mx.float32) * 2.0
    out = m(x)
    assert out.shape == (1, 2, 2, 1)
    assert bool(mx.allclose(out, mx.full((1, 2, 2, 1), 8.0), rtol=1e-5, atol=1e-5).item()) is True


def test_conv3d() -> None:
    """Test `Conv3d` with bias on a 2x2x2 all-ones kernel."""
    m = z_nn.Conv3d(1, 1, 2, bias=True)
    m.weight = mx.ones((1, 2, 2, 2, 1), dtype=mx.float32)
    m.bias = array([0.5], dtype=mx.float32)
    x = mx.ones((1, 3, 3, 3, 1), dtype=mx.float32)
    out = m(x)
    assert out.shape == (1, 2, 2, 2, 1)
    assert (
        bool(mx.allclose(out, mx.full((1, 2, 2, 2, 1), 8.5), rtol=1e-5, atol=1e-5).item()) is True
    )


def test_conv3d_no_bias() -> None:
    """Test `Conv3d` without bias on a 2x2x2 all-ones kernel."""
    m = z_nn.Conv3d(1, 1, 2, bias=False)
    m.weight = mx.ones((1, 2, 2, 2, 1), dtype=mx.float32)
    x = mx.ones((1, 3, 3, 3, 1), dtype=mx.float32)
    out = m(x)
    assert out.shape == (1, 2, 2, 2, 1)
    assert (
        bool(mx.allclose(out, mx.full((1, 2, 2, 2, 1), 8.0), rtol=1e-5, atol=1e-5).item()) is True
    )


def test_conv_transpose1d() -> None:
    """Test `ConvTranspose1d` with bias on deterministic impulse input."""
    m = z_nn.ConvTranspose1d(1, 1, 2, bias=True)
    m.weight = mx.ones((1, 2, 1), dtype=mx.float32)
    m.bias = array([0.5], dtype=mx.float32)
    x = array([[[1.0], [2.0]]], dtype=mx.float32)
    out = m(x)
    assert out.shape == (1, 3, 1)
    assert (
        bool(mx.allclose(out, array([[[1.5], [3.5], [2.5]]]), rtol=1e-5, atol=1e-5).item()) is True
    )


def test_conv_transpose1d_no_bias() -> None:
    """Test `ConvTranspose1d` without bias on deterministic impulse input."""
    m = z_nn.ConvTranspose1d(1, 1, 2, bias=False)
    m.weight = mx.ones((1, 2, 1), dtype=mx.float32)
    x = array([[[1.0], [2.0]]], dtype=mx.float32)
    out = m(x)
    assert out.shape == (1, 3, 1)
    assert (
        bool(mx.allclose(out, array([[[1.0], [3.0], [2.0]]]), rtol=1e-5, atol=1e-5).item()) is True
    )


def test_conv_transpose2d() -> None:
    """Test `ConvTranspose2d` with bias on a single spatial pixel."""
    m = z_nn.ConvTranspose2d(1, 1, 2, bias=True)
    m.weight = mx.ones((1, 2, 2, 1), dtype=mx.float32) * 2.0
    m.bias = array([1.0], dtype=mx.float32)
    x = mx.ones((1, 1, 1, 1), dtype=mx.float32) * 3.0
    out = m(x)
    assert out.shape == (1, 2, 2, 1)
    assert bool(mx.allclose(out, mx.full((1, 2, 2, 1), 7.0), rtol=1e-5, atol=1e-5).item()) is True


def test_conv_transpose2d_no_bias() -> None:
    """Test `ConvTranspose2d` without bias on a single spatial pixel."""
    m = z_nn.ConvTranspose2d(1, 1, 2, bias=False)
    m.weight = mx.ones((1, 2, 2, 1), dtype=mx.float32) * 2.0
    x = mx.ones((1, 1, 1, 1), dtype=mx.float32) * 3.0
    out = m(x)
    assert out.shape == (1, 2, 2, 1)
    assert bool(mx.allclose(out, mx.full((1, 2, 2, 1), 6.0), rtol=1e-5, atol=1e-5).item()) is True


def test_conv_transpose3d() -> None:
    """Test `ConvTranspose3d` with bias on a single 3D voxel."""
    m = z_nn.ConvTranspose3d(1, 1, 2, bias=True)
    m.weight = mx.ones((1, 2, 2, 2, 1), dtype=mx.float32)
    m.bias = array([0.5], dtype=mx.float32)
    x = mx.ones((1, 1, 1, 1, 1), dtype=mx.float32) * 4.0
    out = m(x)
    assert out.shape == (1, 2, 2, 2, 1)
    assert (
        bool(mx.allclose(out, mx.full((1, 2, 2, 2, 1), 4.5), rtol=1e-5, atol=1e-5).item()) is True
    )


def test_conv_transpose3d_no_bias() -> None:
    """Test `ConvTranspose3d` without bias on a single 3D voxel."""
    m = z_nn.ConvTranspose3d(1, 1, 2, bias=False)
    m.weight = mx.ones((1, 2, 2, 2, 1), dtype=mx.float32)
    x = mx.ones((1, 1, 1, 1, 1), dtype=mx.float32) * 4.0
    out = m(x)
    assert out.shape == (1, 2, 2, 2, 1)
    assert (
        bool(mx.allclose(out, mx.full((1, 2, 2, 2, 1), 4.0), rtol=1e-5, atol=1e-5).item()) is True
    )
