import pytest
import numpy as np

try:
    import mlx.core as mx
    import mlx.nn as mx_nn
except ImportError:
    mx = None
    mx_nn = None

import zero_mlx.nn as z_nn
from zero_mlx.array import array


@pytest.mark.skipif(mx is None, reason="mlx not installed")
def test_module():
    m = z_nn.Module()
    # Test __setattr__
    sub_m = z_nn.Module()
    m.sub = sub_m
    m.param = array([1, 2, 3])

    assert "sub" in m._modules
    assert "param" in m._parameters

    # Test parameters()
    params = m.parameters()
    assert "param" in params
    assert "sub" in params

    # Test update()
    m.update({"param": array([4, 5, 6]), "sub": {}})
    np.testing.assert_allclose(m.param.tolist(), [4, 5, 6])

    # Test update with missing key (branch 59->56)
    m.update({"missing_key": array([7, 8, 9])})

    # Test train/eval
    assert m.training
    m.eval()
    assert not m.training
    assert not sub_m.training
    m.train()
    assert m.training
    assert sub_m.training

    # Test call raises NotImplementedError
    with pytest.raises(NotImplementedError):
        m()

    # Test missing attributes branches
    class BadModule(z_nn.Module):
        def __init__(self):
            # Do not call super().__init__()
            self.early_sub = z_nn.Module()
            self.early_param = array([1])

    bad_m = BadModule()
    assert "early_sub" in bad_m._modules
    assert "early_param" in bad_m._parameters

    # Test parameters without _parameters and _modules (delete them to simulate)
    del bad_m._parameters
    del bad_m._modules
    assert bad_m.parameters() == {}

    bad_m.train()


@pytest.mark.skipif(mx is None, reason="mlx not installed")
def test_identity():
    ident = z_nn.Identity()
    x = array([1, 2, 3])
    out = ident(x, something=True)
    np.testing.assert_allclose(out.tolist(), [1, 2, 3])


@pytest.mark.skipif(mx is None, reason="mlx not installed")
def test_sequential():
    layer1 = z_nn.Identity()
    layer2 = z_nn.Identity()
    seq = z_nn.Sequential(layer1, layer2)

    x = array([1, 2, 3])
    out = seq(x, something=True)
    np.testing.assert_allclose(out.tolist(), [1, 2, 3])
