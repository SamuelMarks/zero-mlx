import pytest
import numpy as np
from unittest.mock import patch

import zero_mlx.nn.losses as z_losses
from zero_mlx.array import array
import ml_switcheroo_compiler.ops as sops
from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig


def test_cosine_similarity_loss():
    x1 = array(np.random.randn(2, 3).astype(np.float32))
    x2 = array(np.random.randn(2, 3).astype(np.float32))
    mock_ret = Tensor(None, TensorConfig((2,), "float32", "cpu"))
    with patch.object(
        sops, "cosine_similarity_loss", return_value=mock_ret, create=True
    ):
        out = z_losses.cosine_similarity_loss(x1, x2)
        assert out.shape == (2,)


def test_gaussian_nll_loss():
    i = array(np.random.randn(2, 3).astype(np.float32))
    t = array(np.random.randn(2, 3).astype(np.float32))
    v = array(np.random.randn(2, 3).astype(np.float32))
    mock_ret = Tensor(None, TensorConfig((2, 3), "float32", "cpu"))
    with patch.object(sops, "gaussian_nll_loss", return_value=mock_ret, create=True):
        out = z_losses.gaussian_nll_loss(i, t, v)
        assert out.shape == (2, 3)


def test_hinge_loss():
    i = array(np.random.randn(2, 3).astype(np.float32))
    t = array(np.random.randn(2, 3).astype(np.float32))
    mock_ret = Tensor(None, TensorConfig((2, 3), "float32", "cpu"))
    with patch.object(sops, "hinge_loss", return_value=mock_ret, create=True):
        out = z_losses.hinge_loss(i, t)
        assert out.shape == (2, 3)


def test_huber_loss():
    i = array(np.random.randn(2, 3).astype(np.float32))
    t = array(np.random.randn(2, 3).astype(np.float32))
    mock_ret = Tensor(None, TensorConfig((2, 3), "float32", "cpu"))
    with patch.object(sops, "huber_loss", return_value=mock_ret, create=True):
        out = z_losses.huber_loss(i, t)
        assert out.shape == (2, 3)


def test_kl_div_loss():
    i = array(np.random.randn(2, 3).astype(np.float32))
    t = array(np.random.randn(2, 3).astype(np.float32))
    mock_ret = Tensor(None, TensorConfig((2, 3), "float32", "cpu"))
    with patch.object(sops, "kl_div_loss", return_value=mock_ret, create=True):
        out = z_losses.kl_div_loss(i, t)
        assert out.shape == (2, 3)


def test_l1_loss():
    p = array(np.random.randn(2, 3).astype(np.float32))
    t = array(np.random.randn(2, 3).astype(np.float32))
    mock_ret = Tensor(None, TensorConfig((2, 3), "float32", "cpu"))
    with patch.object(sops, "l1_loss", return_value=mock_ret, create=True):
        out = z_losses.l1_loss(p, t)
        assert out.shape == (2, 3)


def test_log_cosh_loss():
    p = array(np.random.randn(2, 3).astype(np.float32))
    t = array(np.random.randn(2, 3).astype(np.float32))
    mock_ret = Tensor(None, TensorConfig((2, 3), "float32", "cpu"))
    with patch.object(sops, "log_cosh_loss", return_value=mock_ret, create=True):
        out = z_losses.log_cosh_loss(p, t)
        assert out.shape == (2, 3)


def test_margin_ranking_loss():
    i1 = array(np.random.randn(2, 3).astype(np.float32))
    i2 = array(np.random.randn(2, 3).astype(np.float32))
    t = array(np.random.randn(2, 3).astype(np.float32))
    mock_ret = Tensor(None, TensorConfig((2, 3), "float32", "cpu"))
    with patch.object(sops, "margin_ranking_loss", return_value=mock_ret, create=True):
        out = z_losses.margin_ranking_loss(i1, i2, t)
        assert out.shape == (2, 3)


def test_mse_loss():
    p = array(np.random.randn(2, 3).astype(np.float32))
    t = array(np.random.randn(2, 3).astype(np.float32))
    mock_ret = Tensor(None, TensorConfig((2, 3), "float32", "cpu"))
    with patch.object(sops, "mse_loss", return_value=mock_ret, create=True):
        out = z_losses.mse_loss(p, t)
        assert out.shape == (2, 3)


def test_nll_loss():
    i = array(np.random.randn(2, 3).astype(np.float32))
    t = array(np.random.randint(0, 3, (2,)).astype(np.int32))
    mock_ret = Tensor(None, TensorConfig((2,), "float32", "cpu"))
    with patch.object(sops, "nll_loss", return_value=mock_ret, create=True):
        out = z_losses.nll_loss(i, t)
        assert out.shape == (2,)

    w = array(np.random.randn(3).astype(np.float32))
    with patch.object(sops, "nll_loss", return_value=mock_ret, create=True):
        out = z_losses.nll_loss(i, t, weight=w)
        assert out.shape == (2,)


def test_smooth_l1_loss():
    p = array(np.random.randn(2, 3).astype(np.float32))
    t = array(np.random.randn(2, 3).astype(np.float32))
    mock_ret = Tensor(None, TensorConfig((2, 3), "float32", "cpu"))
    with patch.object(sops, "smooth_l1_loss", return_value=mock_ret, create=True):
        out = z_losses.smooth_l1_loss(p, t)
        assert out.shape == (2, 3)


def test_triplet_loss():
    a = array(np.random.randn(2, 3).astype(np.float32))
    p = array(np.random.randn(2, 3).astype(np.float32))
    n = array(np.random.randn(2, 3).astype(np.float32))
    mock_ret = Tensor(None, TensorConfig((2,), "float32", "cpu"))
    with patch.object(sops, "triplet_loss", return_value=mock_ret, create=True):
        out = z_losses.triplet_loss(a, p, n)
        assert out.shape == (2,)
