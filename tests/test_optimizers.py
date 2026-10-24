"""Unit tests for `zero_mlx.optimizers` and `zero_mlx.utils` PyTree functions."""

from __future__ import annotations

import numpy as np
import pytest

from zero_mlx._types import PyTree
from zero_mlx.array import array
from zero_mlx.nn.base import Module
from zero_mlx.nn.linear import Linear
import zero_mlx.optimizers as optimizers


def test_optimizer_base() -> None:
    """Test base `Optimizer` initialization, `init`, `apply_single`, `apply_gradients`, and `update`."""
    opt = optimizers.Optimizer()
    assert opt.schedulers is None
    opt_with_sched = optimizers.Optimizer(
        schedulers={"lr": optimizers.linear_schedule(0.1, 0.01, 10)}
    )
    assert opt_with_sched.schedulers is not None

    model = Linear(2, 2, bias=False)
    model.weight = array([[1.0, 2.0], [3.0, 4.0]])
    opt.init(model.parameters())
    assert "weight" in opt.state

    grads: dict[str, array] = {"weight": array([[0.5, 0.5], [0.5, 0.5]])}
    opt.update(model, grads)
    assert model.weight.shape == (2, 2)


def test_adadelta() -> None:
    """Test `AdaDelta` optimizer single-parameter update."""
    opt = optimizers.AdaDelta(learning_rate=0.1, rho=0.9, eps=1e-6)
    p = array([1.0, 2.0])
    g = array([0.1, 0.2])
    new_p, new_state = opt.apply_single(p, g, {})
    assert isinstance(new_p, array)
    assert new_p.shape == (2,)
    assert isinstance(new_state, dict)


def test_adafactor() -> None:
    """Test `Adafactor` optimizer with default and scheduled learning rate."""
    opt = optimizers.Adafactor(learning_rate=0.1)
    p = array([[1.0, 2.0], [3.0, 4.0]])
    g = array([[0.1, 0.2], [0.3, 0.4]])
    new_p, new_state = opt.apply_single(p, g, {})
    assert isinstance(new_p, array)
    assert new_p.shape == (2, 2)
    assert isinstance(new_state, dict)

    opt_none = optimizers.Adafactor(learning_rate=None)
    new_p2, _ = opt_none.apply_single(p, g, new_state)
    assert new_p2.shape == (2, 2)


def test_adagrad() -> None:
    """Test `Adagrad` optimizer single-parameter update."""
    opt = optimizers.Adagrad(learning_rate=0.1, eps=1e-8)
    p = array([1.0, 2.0])
    g = array([0.1, 0.2])
    new_p, new_state = opt.apply_single(p, g, {})
    assert isinstance(new_p, array)
    assert new_p.shape == (2,)
    assert isinstance(new_state, dict)


def test_adam() -> None:
    """Test `Adam` optimizer single-parameter update."""
    opt = optimizers.Adam(learning_rate=0.1, betas=[0.9, 0.999], eps=1e-8, bias_correction=True)
    p = array([1.0, 2.0])
    g = array([0.1, 0.2])
    new_p, new_state = opt.apply_single(p, g, {})
    assert isinstance(new_p, array)
    assert new_p.shape == (2,)
    assert isinstance(new_state, dict)


def test_adamw() -> None:
    """Test `AdamW` optimizer single-parameter update."""
    opt = optimizers.AdamW(
        learning_rate=0.1,
        betas=[0.9, 0.999],
        eps=1e-8,
        weight_decay=0.01,
        bias_correction=True,
    )
    p = array([1.0, 2.0])
    g = array([0.1, 0.2])
    new_p, new_state = opt.apply_single(p, g, {})
    assert isinstance(new_p, array)
    assert new_p.shape == (2,)
    assert isinstance(new_state, dict)


def test_adamax() -> None:
    """Test `Adamax` optimizer single-parameter update."""
    opt = optimizers.Adamax(learning_rate=0.1, betas=[0.9, 0.999], eps=1e-8)
    p = array([1.0, 2.0])
    g = array([0.1, 0.2])
    new_p, new_state = opt.apply_single(p, g, {})
    assert isinstance(new_p, array)
    assert new_p.shape == (2,)
    assert isinstance(new_state, dict)


def test_lion() -> None:
    """Test `Lion` optimizer single-parameter update."""
    opt = optimizers.Lion(learning_rate=0.1, betas=[0.9, 0.99], weight_decay=0.0)
    p = array([1.0, 2.0])
    g = array([0.1, 0.2])
    new_p, new_state = opt.apply_single(p, g, {})
    assert isinstance(new_p, array)
    assert new_p.shape == (2,)
    assert isinstance(new_state, dict)


def test_multi_optimizer() -> None:
    """Test `MultiOptimizer` initialization, filter routing, and state persistence."""
    opt1 = optimizers.SGD(learning_rate=0.1, momentum=0.9)
    opt2 = optimizers.Adam(learning_rate=0.01)
    opt_base = optimizers.Optimizer()
    m_opt = optimizers.MultiOptimizer([opt1, opt2])
    assert m_opt.filters == []
    m_opt2 = optimizers.MultiOptimizer(
        [opt1, opt_base, opt2],
        filters=[lambda k, g: k.startswith("w1"), lambda k, g: k.startswith("w2")],
    )
    assert len(m_opt2.filters) == 2
    params: PyTree = {
        "w1": array([1.0, 2.0]),
        "w2": array([2.0, 3.0]),
        "w3": array([3.0, 4.0]),
        "meta": 1,
    }
    grads: PyTree = {
        "w1": array([0.1, 0.1]),
        "w2": array([0.2, 0.2]),
        "w3": array([0.3, 0.3]),
        "meta": 1,
    }
    m_opt2.init(params)
    updated = m_opt2.apply_gradients(grads, params)
    assert isinstance(updated, dict)
    assert "w1" in opt1.state
    assert "w3" in opt2.state

    adam_opt = optimizers.Adam(learning_rate=0.1)
    p_tree: PyTree = {"w": array([1.0, 2.0])}
    g_tree: PyTree = {"w": array([0.5, 0.5])}
    p_step1 = adam_opt.apply_gradients(g_tree, p_tree)
    assert "w" in adam_opt.state and len(adam_opt.state["w"]) > 0
    p_step2 = adam_opt.apply_gradients(g_tree, p_step1)
    assert isinstance(p_step2, dict)


def test_muon() -> None:
    """Test `Muon` optimizer single-parameter update."""
    opt = optimizers.Muon(
        learning_rate=0.1, momentum=0.95, weight_decay=0.01, nesterov=True, ns_steps=5
    )
    p = array([[1.0, 2.0], [3.0, 4.0]])
    g = array([[0.1, 0.2], [0.3, 0.4]])
    new_p, new_state = opt.apply_single(p, g, {})
    assert isinstance(new_p, array)
    assert new_p.shape == (2, 2)
    assert isinstance(new_state, dict)


def test_rmsprop() -> None:
    """Test `RMSprop` optimizer single-parameter update."""
    opt = optimizers.RMSprop(learning_rate=0.1, alpha=0.99, eps=1e-8)
    p = array([1.0, 2.0])
    g = array([0.1, 0.2])
    new_p, new_state = opt.apply_single(p, g, {})
    assert isinstance(new_p, array)
    assert new_p.shape == (2,)
    assert isinstance(new_state, dict)


def test_sgd() -> None:
    """Test `SGD` optimizer with both constant and scheduled learning rates."""
    sched = optimizers.exponential_decay(0.1, 0.9)
    opt = optimizers.SGD(
        learning_rate=sched,
        momentum=0.9,
        weight_decay=0.01,
        dampening=0.0,
        nesterov=True,
    )
    p = array([1.0, 2.0])
    g = array([0.1, 0.2])
    new_p, new_state = opt.apply_single(p, g, {})
    assert isinstance(new_p, array)
    assert new_p.shape == (2,)
    assert isinstance(new_state, dict)


def test_clip_grad_norm() -> None:
    """Test `clip_grad_norm` both when norm exceeds `max_norm` and when it is below `max_norm`."""
    grads: PyTree = {"w": array([3.0, 4.0]), "b": 1}
    clipped, total_norm = optimizers.clip_grad_norm(grads, max_norm=2.5)
    assert isinstance(clipped, dict)
    assert isinstance(total_norm, array)
    assert pytest.approx(float(total_norm.item().real), rel=1e-4) == 5.0
    w_clipped = clipped["w"]
    assert isinstance(w_clipped, array)
    np.testing.assert_allclose(w_clipped.tolist(), [1.5, 2.0], rtol=1e-4)

    unchanged, norm2 = optimizers.clip_grad_norm(grads, max_norm=10.0)
    assert isinstance(unchanged, dict)
    w_unchanged = unchanged["w"]
    assert isinstance(w_unchanged, array)
    np.testing.assert_allclose(w_unchanged.tolist(), [3.0, 4.0], rtol=1e-4)
    assert pytest.approx(float(norm2.item().real), rel=1e-4) == 5.0


def test_schedules() -> None:
    """Test all learning rate schedule functions in `zero_mlx.optimizers.schedulers`."""
    s1 = optimizers.cosine_decay(1.0, 10, 0.1)
    assert s1(0) == pytest.approx(1.0)
    assert s1(10) == pytest.approx(0.1)

    s2 = optimizers.exponential_decay(1.0, 0.5)
    assert s2(2) == pytest.approx(0.25)

    s3 = optimizers.linear_schedule(0.0, 1.0, 10)
    assert s3(5) == pytest.approx(0.5)

    s4 = optimizers.step_decay(1.0, 0.5, 5)
    assert s4(6) == pytest.approx(0.5)

    s_join = optimizers.join_schedules([s3, s2], [5])
    assert s_join(2) == pytest.approx(s3(2))
    assert s_join(6) == pytest.approx(s2(1))


def test_trees() -> None:
    """Test `tree_flatten`, `tree_unflatten`, `tree_map`, `tree_merge`, and `tree_reduce`."""
    t: PyTree = {"a": 1, "b": {"c": 2}, "d": [3, (4, 5)]}
    f = optimizers.tree_flatten(t)
    assert f == [("a", 1), ("b.c", 2), ("d.0", 3), ("d.1.0", 4), ("d.1.1", 5)]

    dest_list: list[tuple[str, PyTree]] = []
    res_list = optimizers.tree_flatten({"a": 1}, destination=dest_list)
    assert res_list == [("a", 1)]

    dest_dict: dict[str, PyTree] = {}
    res_dict = optimizers.tree_flatten({"a": 1, "b": {"c": 2}}, destination=dest_dict)
    assert res_dict == {"a": 1, "b.c": 2}

    unflat = optimizers.tree_unflatten(f)
    assert unflat == {"a": 1, "b": {"c": 2}, "d": [3, [4, 5]]}
    assert optimizers.tree_unflatten({}) == {}
    assert optimizers.tree_unflatten([("", 42)]) == 42
    assert optimizers.tree_unflatten(1) == 1
    assert optimizers.tree_unflatten([1, 2]) == [1, 2]
    assert optimizers.tree_unflatten([("0", 10), ("2", 30)]) == [10, None, 30]

    mapped = optimizers.tree_map(lambda x: int(x) + 1 if isinstance(x, int) else x, t)
    assert mapped == {"a": 2, "b": {"c": 3}, "d": [4, (5, 6)]}

    merged = optimizers.tree_merge({"a": 1, "b": [1, 2]}, {"b": [3, 4], "c": 5})
    assert merged == {"a": 1, "b": [3, 4], "c": 5}
    merged_fn = optimizers.tree_merge(
        {"a": (1, 2)},
        {"a": (3, 4)},
        merge_fn=lambda a, b: int(a) + int(b) if isinstance(a, int) and isinstance(b, int) else b,
    )
    assert merged_fn == {"a": (4, 6)}
    assert optimizers.tree_merge({}, {"a": 1}) == {"a": 1}
    assert optimizers.tree_merge({"a": 1}, {}) == {"a": 1}
    assert optimizers.tree_merge(None, {"a": 1}) == {"a": 1}
    assert optimizers.tree_merge({"a": 1}, None) == {"a": 1}
    with pytest.raises(ValueError):
        optimizers.tree_merge([1], [1, 2])
    with pytest.raises(ValueError):
        optimizers.tree_merge((1,), (1, 2))

    reduced = optimizers.tree_reduce(
        lambda acc, x: (
            (int(acc) if isinstance(acc, int) else 0) + (int(x) if isinstance(x, int) else 0)
        ),
        t,
        0,
    )
    assert reduced == 15
    reduced_no_init = optimizers.tree_reduce(
        lambda acc, x: (
            (int(acc) if isinstance(acc, int) else 0) + (int(x) if isinstance(x, int) else 0)
        ),
        {"a": 2, "b": 3},
    )
    assert reduced_no_init == 5
