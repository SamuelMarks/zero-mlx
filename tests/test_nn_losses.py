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
    out = z_losses.cosine_similarity_loss(x1, x2)
    assert out.shape == (2,)


def test_gaussian_nll_loss():
    i = array(np.random.randn(2, 3).astype(np.float32))
    t = array(np.random.randn(2, 3).astype(np.float32))
    v = array(np.random.randn(2, 3).astype(np.float32))
    mock_ret = Tensor(None, TensorConfig((2, 3), "float32", "cpu"))
