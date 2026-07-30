import pytest
import numpy as np
from zero_mlx.array import array
import zero_mlx.nn as z_nn


def test_upsample_print():
    m = z_nn.Upsample(scale_factor=2.0)
    x = array(np.random.randn(2, 2, 2, 3).astype(np.float32))

    import ml_switcheroo_compiler.backends.registry as reg

    backend = reg.get_active_backend()
    data = backend.execute_op(
        "ResizeNearest", x._tensor.data, size=(4, 4), align_corners=False
    )
    print("BACKEND DATA SHAPE", data.shape)
