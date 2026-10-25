"""Unit tests for `zero_mlx.nn.base` and `zero_mlx.nn.containers`."""

from __future__ import annotations

import pytest

from zero_mlx.array import array
import zero_mlx.nn as z_nn


def test_module() -> None:
    """Test `Module` parameter registration, submodule updates, and `train`/`eval` modes."""

    class SubModule(z_nn.Module):
        """Child test module."""

        sub_param: array

    class MainModule(z_nn.Module):
        """Parent test module."""

        param: array
        sub: SubModule

    m = MainModule()
    assert m.training is True

    m.param = array([1, 2, 3])
    sub = SubModule()
    sub.sub_param = array([4, 5])
    m.sub = sub

    params = m.parameters()
    assert "param" in params
    assert "sub" in params
    sub_dict = params["sub"]
    assert isinstance(sub_dict, dict)
    assert "sub_param" in sub_dict

    m.update({"param": array([4, 5, 6]), "sub": {"sub_param": array([7, 8])}})
    assert m.param.tolist() == [4, 5, 6]
    assert m.sub.sub_param.tolist() == [7, 8]

    m.eval()
    assert m.training is False
    assert m.sub.training is False

    m.train()
    assert m.training is True
    assert m.sub.training is True

    class BadModule(z_nn.Module):
        """Subclass that omits `super().__init__()`."""

        m: z_nn.Module
        p: array

        def __init__(self) -> None:
            """Initialize without calling super."""

    bm = BadModule()
    bm.m = z_nn.Module()
    bm.p = array([1])
    assert "m" in bm._modules
    assert "p" in bm._parameters
    assert "p" in bm
    assert isinstance(bm["p"], array)
    bm["extra"] = array([2])
    assert isinstance(bm.extra, array)
    bm.update_modules({"m": z_nn.Identity()})
    assert isinstance(bm.m, z_nn.Identity)
    with pytest.raises(AttributeError):
        _ = bm.nonexistent_attr
    with pytest.raises(KeyError):
        _ = bm["nonexistent_key"]
    with pytest.raises(NotImplementedError):
        z_nn.Module()._call_forward(array([1.0]))
    with pytest.raises(ValueError):
        bm.update({"unknown_param": array([1.0])}, strict=True)
    bm.update({"unknown_param": array([1.0])}, strict=False)
    bm.list_attr = [z_nn.Identity(), array([1.0])]
    bm.dict_attr = {"id": z_nn.Identity(), "arr": array([1.0])}
    assert "list_attr.0" in bm._modules
    assert "dict_attr.id" in bm._modules


def test_identity() -> None:
    """Test `Identity` module passthrough with arbitrary extra arguments."""
    id_layer = z_nn.Identity(1, 2, a=3)
    x = array([1, 2, 3])
    out = id_layer(x, 4, b=5)
    assert out.tolist() == [1, 2, 3]


def test_sequential() -> None:
    """Test `Sequential` container chaining multiple layers."""
    layer1 = z_nn.Identity()
    layer2 = z_nn.Identity()
    seq = z_nn.Sequential(layer1, layer2, lambda t: t * 2)

    x = array([1, 2, 3])
    out = seq(x, something=True)
    assert out.tolist() == [2, 4, 6]
