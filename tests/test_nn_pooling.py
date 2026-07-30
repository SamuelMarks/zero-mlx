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


def test_maxpool1d():
    m = z_nn.MaxPool1d(2)
    _test_module_shape(m, (2, 5, 3), (2, 2, 3))
    m = z_nn.MaxPool1d((2,))
    _test_module_shape(m, (2, 5, 3), (2, 2, 3))


def test_maxpool2d():
    m = z_nn.MaxPool2d(2)
    _test_module_shape(m, (2, 5, 5, 3), (2, 2, 2, 3))
    m = z_nn.MaxPool2d((2, 2))
    _test_module_shape(m, (2, 5, 5, 3), (2, 2, 2, 3))


def test_maxpool3d():
    m = z_nn.MaxPool3d(2)
    _test_module_shape(m, (2, 5, 5, 5, 3), (2, 2, 2, 2, 3))
    m = z_nn.MaxPool3d((2, 2, 2))
    _test_module_shape(m, (2, 5, 5, 5, 3), (2, 2, 2, 2, 3))


def test_avgpool1d():
    m = z_nn.AvgPool1d(2)
    _test_module_shape(m, (2, 5, 3), (2, 2, 3))
    m = z_nn.AvgPool1d((2,))
    _test_module_shape(m, (2, 5, 3), (2, 2, 3))


def test_avgpool2d():
    m = z_nn.AvgPool2d(2)
    _test_module_shape(m, (2, 5, 5, 3), (2, 2, 2, 3))
    m = z_nn.AvgPool2d((2, 2))
    _test_module_shape(m, (2, 5, 5, 3), (2, 2, 2, 3))


def test_avgpool3d():
    m = z_nn.AvgPool3d(2)
    _test_module_shape(m, (2, 5, 5, 5, 3), (2, 2, 2, 2, 3))
    m = z_nn.AvgPool3d((2, 2, 2))
    _test_module_shape(m, (2, 5, 5, 5, 3), (2, 2, 2, 2, 3))


def test_upsample():
    m = z_nn.Upsample(scale_factor=2.0)
    _test_module_shape(m, (2, 2, 2, 3), (2, 4, 4, 3))
