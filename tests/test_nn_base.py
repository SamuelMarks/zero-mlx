"""Unit tests for `zero_mlx.nn.base` and `zero_mlx.nn.containers`."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

import zero_mlx as mx
from zero_mlx.array import array
import zero_mlx.nn as z_nn
from zero_mlx.nn.base import ModuleAttr, _ModuleAttrDescriptor
import zero_mlx.optimizers as z_opt


def test_module() -> None:
    """Test `Module` parameter registration, submodule updates, and `train`/`eval` modes."""

    class SubModule(z_nn.Module):
        """Child test module."""

        if TYPE_CHECKING:
            sub_param: array
        else:
            sub_param = ModuleAttr("sub_param")

    class MainModule(z_nn.Module):
        """Parent test module."""

        if TYPE_CHECKING:
            param: array
            sub: SubModule
        else:
            param = ModuleAttr("param")
            sub = ModuleAttr("sub")

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

        if TYPE_CHECKING:
            m: z_nn.Module
            p: array
            extra: array
            list_attr: list[z_nn.Module | array]
            dict_attr: dict[str, z_nn.Module | array]
        else:
            m = ModuleAttr("m")
            p = ModuleAttr("p")
            extra = ModuleAttr("extra")
            list_attr = ModuleAttr("list_attr")
            dict_attr = ModuleAttr("dict_attr")

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
        _ = BadModule().extra

    desc = _ModuleAttrDescriptor("weight")
    assert desc.__get__(None, z_nn.Module) is desc
    with pytest.raises(KeyError):
        _ = bm["nonexistent_key"]
    with pytest.raises(NotImplementedError):
        z_nn.Module()._call_forward(array([1.0]))
    with pytest.raises(ValueError):
        bm.update({"unknown_param": array([1.0])}, strict=True)
    bm.update({"unknown_param": array([1.0])}, strict=False)
    bm.list_attr = [z_nn.Linear(2, 2), array([1.0])]
    bm.dict_attr = {"id": z_nn.Linear(2, 2), "arr": array([1.0])}
    assert "list_attr.0" in bm._modules
    assert "dict_attr.id" in bm._modules
    bm_params = bm.parameters()
    assert "list_attr" in bm_params and "dict_attr" in bm_params
    bm.apply(lambda t: t * 2.0)
    bm.update(bm.parameters())


def test_identity() -> None:
    """Test `Identity` module passthrough with arbitrary extra arguments."""
    id_layer = z_nn.Identity(1, 2, a=3)
    x = array([1, 2, 3])
    out = id_layer(x, 4, b=5)
    assert out.tolist() == [1, 2, 3]


def test_sequential() -> None:
    """Test `Sequential` container chaining multiple layers and recursive parameter operations."""
    layer1 = z_nn.Identity()
    layer2 = z_nn.Identity()
    seq = z_nn.Sequential(layer1, layer2, lambda t: t * 2)

    x = array([1, 2, 3])
    out = seq(x, something=True)
    assert out.tolist() == [2, 4, 6]

    lin1 = z_nn.Linear(2, 2, bias=True)
    lin1.weight = array([[1.0, 0.0], [0.0, 1.0]])
    lin1.bias = array([0.5, -0.5])
    seq_lin = z_nn.Sequential(lin1)
    seq_params = seq_lin.parameters()
    assert "layers" in seq_params
    assert isinstance(seq_params["layers"], list)
    seq_lin.apply(lambda w: w * 2.0)
    assert lin1.bias is not None and lin1.bias.tolist() == [1.0, -1.0]


def test_module_dict_and_traversal_methods() -> None:
    """Test dict-like methods, filters, leaf_modules, set_dtype, and analytical `nn.value_and_grad`."""
    import zero_mlx.core as mx_core

    assert mx_core.array([1.0, 2.0]).diag().tolist() == [[1.0, 0.0], [0.0, 2.0]]
    assert mx.linspace(0.0, 1.0, 4, endpoint=False).tolist() == [0.0, 0.25, 0.5, 0.75]

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

        if TYPE_CHECKING:
            lin: z_nn.Linear
            nested: z_nn.Sequential
            _priv: int
        else:
            lin = ModuleAttr("lin")
            nested = ModuleAttr("nested")
            _priv = ModuleAttr("_priv")

        def __init__(self) -> None:
            """Initialize composite module."""
            super().__init__()
            self.lin = z_nn.Linear(64, 4)
            self.nested = z_nn.Sequential(z_nn.Linear(4, 2))
            self._priv = 123

    comp = Composite()
    assert isinstance(comp.state, dict)
    assert "lin" in list(comp.keys())
    assert "lin" in list(iter(comp))
    assert len(comp.values()) >= 2
    assert len(comp.items()) >= 2
    assert comp.get("nonexistent", 99) == 99
    assert comp.setdefault("custom_flag", True) is True
    assert comp.pop("custom_flag") is True
    comp["temp_k"] = 1
    assert "lin" in comp.trainable_parameters()
    comp.apply(lambda t: t)
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
    assert "lin" in mapped and "nested" in mapped
    leaves = comp.leaf_modules()
    assert "lin" in leaves
    visited: list[str] = []
    comp.apply_to_modules(lambda path, mod: visited.append(path))
    assert "lin" in visited
    comp.set_dtype(mx.float16)
    assert comp.lin.weight.dtype == mx.float16
    empty_m = z_nn.Module()
    empty_m.clear()
    assert empty_m.keys() == []

    # Test real analytical gradient computation in nn.value_and_grad and optimizer step
    lin_train = z_nn.Linear(2, 1, bias=True)
    lin_train.weight = array([[2.0, -1.0]])
    lin_train.bias = array([0.5])
    x_in = array([[1.0, 2.0]])
    y_tgt = array([[3.0]])

    def _mse(*args: array | z_nn.Module, **kwargs: array) -> array:
        """Compute squared error loss `(lin_train(x_b) - y_b)^2`.

        Args:
            *args: Positional input batch and target batch.
            **kwargs: Unused keyword arguments.

        Returns:
            Scalar squared error `array`.

        """
        _ = kwargs
        x_b = args[0] if isinstance(args[0], array) else array(0.0)
        y_b = args[1] if isinstance(args[1], array) else array(0.0)
        pred = lin_train(x_b)
        return mx.sum((pred - y_b) ** 2)

    vng_fn = z_nn.value_and_grad(lin_train, _mse)
    loss0, grads0 = vng_fn(x_in, y_tgt)
    # pred = 2*1 + (-1)*2 + 0.5 = 0.5; err = 0.5 - 3.0 = -2.5; loss = 6.25
    # dL/dW = 2 * (-2.5) * [1.0, 2.0] = [-5.0, -10.0]; dL/db = [-5.0]
    assert abs(float(loss0.item().real) - 6.25) < 1e-4
    w_grad = grads0["weight"]
    b_grad = grads0["bias"]
    assert isinstance(w_grad, array) and isinstance(b_grad, array)
    assert bool(mx.allclose(w_grad, array([[-5.0, -10.0]]), rtol=1e-3, atol=1e-3).item()) is True
    assert bool(mx.allclose(b_grad, array([-5.0]), rtol=1e-3, atol=1e-3).item()) is True

    sgd = z_opt.SGD(learning_rate=0.05)
    sgd.update(lin_train, grads0)
    loss1, _ = vng_fn(x_in, y_tgt)
    assert float(loss1.item().real) < float(loss0.item().real)

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
