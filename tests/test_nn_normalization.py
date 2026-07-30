import pytest
import numpy as np
import warnings
from unittest.mock import patch

import zero_mlx.nn as z_nn
from zero_mlx.array import array
import ml_switcheroo_compiler.ops as sops


def _test_module_shape(m, x_shape, expected_shape):
    x = array(np.random.randn(*x_shape).astype(np.float32))

    out = m(x)
    assert out.shape == expected_shape


def test_batchnorm():
    m = z_nn.BatchNorm(3)
    _test_module_shape(m, (2, 5, 3), (2, 5, 3))
    m = z_nn.BatchNorm(3, affine=False)
    _test_module_shape(m, (2, 5, 3), (2, 5, 3))
    m = z_nn.BatchNorm(3, track_running_stats=False)
    _test_module_shape(m, (2, 5, 3), (2, 5, 3))


def test_groupnorm():
    m = z_nn.GroupNorm(2, 4)
    _test_module_shape(m, (2, 4, 5, 5), (2, 4, 5, 5))
    m = z_nn.GroupNorm(2, 4, affine=False)
    _test_module_shape(m, (2, 4, 5, 5), (2, 4, 5, 5))


def test_instancenorm():
    m = z_nn.InstanceNorm(4)
    _test_module_shape(m, (2, 4, 5, 5), (2, 4, 5, 5))
    m = z_nn.InstanceNorm(4, affine=True)
    _test_module_shape(m, (2, 4, 5, 5), (2, 4, 5, 5))


def test_layernorm():
    m = z_nn.LayerNorm(3)
    _test_module_shape(m, (2, 5, 3), (2, 5, 3))
    m = z_nn.LayerNorm((5, 3))
    _test_module_shape(m, (2, 5, 3), (2, 5, 3))
    m = z_nn.LayerNorm(3, affine=False)
    _test_module_shape(m, (2, 5, 3), (2, 5, 3))


def test_rmsnorm():
    m = z_nn.RMSNorm(3)
    _test_module_shape(m, (2, 5, 3), (2, 5, 3))
    m = z_nn.RMSNorm((5, 3))
    _test_module_shape(m, (2, 5, 3), (2, 5, 3))
