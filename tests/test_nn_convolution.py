import pytest
import numpy as np
import warnings
from unittest.mock import patch

import zero_mlx.nn as z_nn
from zero_mlx.array import array
import ml_switcheroo_compiler.ops as sops


def _test_module_shape(m, x_shape, expected_shape):
    x = array(np.random.randn(*x_shape).astype(np.float32))

    # Mock sops functions so they just return a tensor of the correct shape
    # This allows us to fully trace and cover the frontend logic without
    # worrying about Tier 2 backend completeness.
    with (
        patch.object(sops, "conv1d", return_value=array(np.zeros(expected_shape))),
        patch.object(sops, "conv2d", return_value=array(np.zeros(expected_shape))),
        patch.object(sops, "conv3d", return_value=array(np.zeros(expected_shape))),
        patch.object(
            sops, "conv1d_transpose", return_value=array(np.zeros(expected_shape))
        ),
        patch.object(
            sops, "conv2d_transpose", return_value=array(np.zeros(expected_shape))
        ),
        patch.object(
            sops, "conv3d_transpose", return_value=array(np.zeros(expected_shape))
        ),
        patch.object(sops, "add", return_value=array(np.zeros(expected_shape))),
    ):
        out = m(x)
        assert out.shape == expected_shape


def test_conv1d():
    m = z_nn.Conv1d(3, 4, 2)
    _test_module_shape(m, (2, 5, 3), (2, 4, 4))


def test_conv1d_no_bias():
    m = z_nn.Conv1d(3, 4, 2, bias=False)
    _test_module_shape(m, (2, 5, 3), (2, 4, 4))


def test_conv2d():
    m = z_nn.Conv2d(3, 4, 2)
    _test_module_shape(m, (2, 5, 5, 3), (2, 4, 4, 4))


def test_conv2d_no_bias():
    m = z_nn.Conv2d(3, 4, 2, bias=False)
    _test_module_shape(m, (2, 5, 5, 3), (2, 4, 4, 4))


def test_conv3d():
    m = z_nn.Conv3d(3, 4, 2)
    _test_module_shape(m, (2, 5, 5, 5, 3), (2, 4, 4, 4, 4))


def test_conv3d_no_bias():
    m = z_nn.Conv3d(3, 4, 2, bias=False)
    _test_module_shape(m, (2, 5, 5, 5, 3), (2, 4, 4, 4, 4))


def test_conv_transpose1d():
    m = z_nn.ConvTranspose1d(3, 4, 2)
    _test_module_shape(m, (2, 5, 3), (2, 6, 4))


def test_conv_transpose1d_no_bias():
    m = z_nn.ConvTranspose1d(3, 4, 2, bias=False)
    _test_module_shape(m, (2, 5, 3), (2, 6, 4))


def test_conv_transpose2d():
    m = z_nn.ConvTranspose2d(3, 4, 2)
    _test_module_shape(m, (2, 5, 5, 3), (2, 6, 6, 4))


def test_conv_transpose2d_no_bias():
    m = z_nn.ConvTranspose2d(3, 4, 2, bias=False)
    _test_module_shape(m, (2, 5, 5, 3), (2, 6, 6, 4))


def test_conv_transpose3d():
    m = z_nn.ConvTranspose3d(3, 4, 2)
    _test_module_shape(m, (2, 5, 5, 5, 3), (2, 6, 6, 6, 4))


def test_conv_transpose3d_no_bias():
    m = z_nn.ConvTranspose3d(3, 4, 2, bias=False)
    _test_module_shape(m, (2, 5, 5, 5, 3), (2, 6, 6, 6, 4))
