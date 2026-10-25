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
        _ = z_nn.Module().extra
    from zero_mlx.nn.base import _ModuleAttrDescriptor

    desc = _ModuleAttrDescriptor("weight")
    assert desc.__get__(None, z_nn.Module) is desc
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


def test_module_dict_and_traversal_methods() -> None:
    """Test dict-like methods, filters, leaf_modules, set_dtype, and layer factory methods."""
    import zero_mlx as mx
    import zero_mlx.core as mx_core

    assert mx_core.array([1.0, 2.0]).diag().shape == (2, 2)
    assert mx.linspace(0.0, 1.0, 4, endpoint=False).shape == (4,)

    arr = array([1.0, 2.0])
    for op_fn in (
        arr.at.subtract,
        arr.at.multiply,
        arr.at.divide,
        arr.at.maximum,
        arr.at.minimum,
    ):
        with pytest.raises(ValueError):
            op_fn(1.0)

    dev = mx.Device(mx.cpu, 0)
    dev.type = mx.gpu
    assert dev.type == mx.gpu
    st = mx.Stream(dev, 0)
    st.device = mx.Device(mx.cpu, 0)
    assert st.device.type == mx.cpu

    class Composite(z_nn.Module):
        """Composite module for testing traversal and dict methods."""

        def __init__(self) -> None:
            """Initialize composite module."""
            super().__init__()
            self.lin = z_nn.Linear(64, 4)
            self.nested = z_nn.Sequential(z_nn.Linear(4, 2))
            self._priv = 123

    comp = Composite()
    assert isinstance(comp.state, dict)
    assert "lin" in list(comp.keys())
    assert len(comp.values()) >= 2
    assert len(comp.items()) >= 2
    assert comp.get("nonexistent", 99) == 99
    assert comp.setdefault("custom_flag", True) is True
    assert comp.pop("custom_flag") is True
    comp["temp_k"] = 1
    popped_k, _ = comp.popitem()
    assert isinstance(popped_k, str)
    copied = comp.copy()
    assert isinstance(copied, dict)
    fk = z_nn.Module.fromkeys(["a", "b"], 0)
    assert fk == {"a": 0, "b": 0}
    assert z_nn.Module.is_module(comp) is True
    assert z_nn.Module.is_module(123) is False
    assert z_nn.Module.valid_child_filter(comp, "lin", comp.lin) is True
    assert z_nn.Module.valid_child_filter(comp, "_priv", 123) is False
    assert z_nn.Module.valid_parameter_filter(comp, "lin", comp.lin) is True
    assert z_nn.Module.trainable_parameter_filter(comp, "lin", comp.lin) is True

    mapped = comp.filter_and_map(
        z_nn.Module.valid_parameter_filter,
        map_fn=lambda v: v,
    )
    assert "lin" in mapped
    leaves = comp.leaf_modules()
    assert "lin" in leaves
    visited: list[str] = []
    comp.apply_to_modules(lambda path, mod: visited.append(path))
    assert "lin" in visited
    comp.set_dtype("float16")
    assert comp.lin.weight.dtype == mx.float16
    empty_m = z_nn.Module()
    empty_m.clear()
    assert empty_m.keys() == []

    lin = z_nn.Linear(64, 4)
    qlin = lin.to_quantized()
    assert isinstance(qlin, z_nn.QuantizedLinear)
    qqlin = lin.to_quantized(quantize_input=True)
    assert isinstance(qqlin, z_nn.QQLinear)
    a2s = z_nn.AllToShardedLinear.from_linear(lin)
    assert isinstance(a2s, z_nn.AllToShardedLinear)
    s2a = z_nn.ShardedToAllLinear.from_linear(lin)
    assert isinstance(s2a, z_nn.ShardedToAllLinear)
    qa2s = z_nn.QuantizedAllToShardedLinear.from_quantized_linear(qlin)
    assert isinstance(qa2s, z_nn.QuantizedAllToShardedLinear)
    qs2a = z_nn.QuantizedShardedToAllLinear.from_quantized_linear(qlin)
    assert isinstance(qs2a, z_nn.QuantizedShardedToAllLinear)

    emb = z_nn.Embedding(8, 64)
    qemb = emb.to_quantized()
    assert isinstance(qemb, z_nn.QuantizedEmbedding)

    rnn_fn = z_nn.RNN(4, 4, nonlinearity=lambda h: h * 0.5)
    out_rnn, _ = rnn_fn(mx.ones((2, 3, 4)))
    assert out_rnn.shape == (2, 3, 4)
