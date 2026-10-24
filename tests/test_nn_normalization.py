"""Unit tests for `BatchNorm`, `GroupNorm`, `InstanceNorm`, `LayerNorm`, and `RMSNorm`."""

from __future__ import annotations

from typing import Callable

import numpy as np

from zero_mlx.array import array
import zero_mlx.nn as z_nn


def _test_module_shape(
    m: Callable[[array], array],
    x_shape: tuple[int, ...],
    expected_shape: tuple[int, ...],
) -> None:
    """Run normalization module `m` on a random input of `x_shape` and assert `expected_shape`.

    Args:
        m: Normalization `Module` to test.
        x_shape: Input tensor shape.
        expected_shape: Expected output tensor shape.

    """
    x = array(np.random.randn(*x_shape).astype(np.float32))
    out = m(x)
    assert out.shape == expected_shape


def test_batchnorm() -> None:
    """Test `BatchNorm` with and without affine/running stats in train and eval modes."""
    m = z_nn.BatchNorm(3)
    _test_module_shape(m, (2, 5, 3), (2, 5, 3))
    m.eval()
    _test_module_shape(m, (2, 5, 3), (2, 5, 3))
    m_no_affine = z_nn.BatchNorm(3, affine=False, track_running_stats=False)
    _test_module_shape(m_no_affine, (2, 5, 3), (2, 5, 3))


def test_groupnorm() -> None:
    """Test `GroupNorm` with and without affine parameters."""
    m = z_nn.GroupNorm(2, 4)
    _test_module_shape(m, (2, 4, 3, 3), (2, 4, 3, 3))
    _test_module_shape(m, (2, 3, 3, 4), (2, 3, 3, 4))
    m_no_affine = z_nn.GroupNorm(2, 4, affine=False)
    _test_module_shape(m_no_affine, (2, 4, 3, 3), (2, 4, 3, 3))


def test_instancenorm() -> None:
    """Test `InstanceNorm` with and without affine parameters."""
    m = z_nn.InstanceNorm(3, affine=True)
    _test_module_shape(m, (2, 3, 4, 4), (2, 3, 4, 4))
    _test_module_shape(m, (2, 4, 4, 3), (2, 4, 4, 3))
    m_no_affine = z_nn.InstanceNorm(3, affine=False)
    _test_module_shape(m_no_affine, (2, 3, 4, 4), (2, 3, 4, 4))


def test_layernorm() -> None:
    """Test `LayerNorm` with int/tuple dims and with/without affine."""
    m = z_nn.LayerNorm(3)
    _test_module_shape(m, (2, 5, 3), (2, 5, 3))
    m_tup = z_nn.LayerNorm((5, 3))
    _test_module_shape(m_tup, (2, 5, 3), (2, 5, 3))
    m_no_affine = z_nn.LayerNorm(3, affine=False)
    _test_module_shape(m_no_affine, (2, 5, 3), (2, 5, 3))


def test_rmsnorm() -> None:
    """Test `RMSNorm` with int and tuple dims."""
    m = z_nn.RMSNorm(3)
    _test_module_shape(m, (2, 5, 3), (2, 5, 3))
    m_tup = z_nn.RMSNorm((5, 3))
    _test_module_shape(m_tup, (2, 5, 3), (2, 5, 3))
