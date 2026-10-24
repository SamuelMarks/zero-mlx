"""Regression test for `nn.Upsample` output spatial dimensions."""

from __future__ import annotations

import numpy as np

from zero_mlx.array import array
import zero_mlx.nn as z_nn


def test_upsample_print() -> None:
    """Verify `nn.Upsample(scale_factor=2.0)` doubles the spatial dimensions of a 4D NHWC tensor."""
    m = z_nn.Upsample(scale_factor=2.0)
    x = array(np.random.randn(2, 2, 2, 3).astype(np.float32))
    out = m(x)
    assert out.shape == (2, 4, 4, 3)
