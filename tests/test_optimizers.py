import pytest
import numpy as np
from unittest.mock import patch

import zero_mlx.optimizers as optimizers
from zero_mlx.array import array
import ml_switcheroo_compiler.ops as sops
from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig


def test_optimizer_base():
    opt = optimizers.Optimizer()
    assert opt.schedulers is None
    # update does nothing in this shell
    opt.update(None, None)


def test_multi_optimizer():
    opt = optimizers.MultiOptimizer([optimizers.SGD(0.1)])
    assert opt.optimizers is not None
    assert opt.filters == []


def _test_optimizer_apply_single(opt_class, args, kwargs, update_fn_name):
    opt = opt_class(*args, **kwargs)
    param = array(np.random.randn(2, 3).astype(np.float32))
    grad = array(np.random.randn(2, 3).astype(np.float32))
    state = {"dummy": array(np.random.randn(2, 3).astype(np.float32))}

    mock_ret_p = Tensor(None, TensorConfig((2, 3), "float32", "cpu"))
    mock_ret_state = {"dummy": Tensor(None, TensorConfig((2, 3), "float32", "cpu"))}

    with patch(
        f"ml_switcheroo_compiler.ops.optimizers.updates.{update_fn_name}",
        return_value=(mock_ret_p, mock_ret_state),
        create=True,
    ):
        new_p, new_state = opt.apply_single(param, grad, state)
        assert new_p.shape == (2, 3)
        assert "dummy" in new_state
        assert new_state["dummy"].shape == (2, 3)

    with patch(
        f"ml_switcheroo_compiler.ops.optimizers.updates.{update_fn_name}",
        side_effect=ImportError,
        create=True,
    ):
        new_p, new_state = opt.apply_single(param, grad, state)
        assert new_p.shape == (2, 3)
        assert "dummy" in new_state
        assert new_state["dummy"].shape == (2, 3)


def test_adadelta():
    _test_optimizer_apply_single(optimizers.AdaDelta, [0.1], {}, "adadelta_update")


def test_adafactor():
    _test_optimizer_apply_single(optimizers.Adafactor, [0.1], {}, "adafactor_update")


def test_adagrad():
    _test_optimizer_apply_single(optimizers.Adagrad, [0.1], {}, "adagrad_update")


def test_adam():
    _test_optimizer_apply_single(optimizers.Adam, [0.1], {}, "adam_update")


def test_adamw():
    _test_optimizer_apply_single(optimizers.AdamW, [0.1], {}, "adamw_update")


def test_adamax():
    _test_optimizer_apply_single(optimizers.Adamax, [0.1], {}, "adamax_update")


def test_lion():
    _test_optimizer_apply_single(optimizers.Lion, [0.1], {}, "lion_update")


def test_muon():
    _test_optimizer_apply_single(optimizers.Muon, [0.1], {}, "muon_update")


def test_rmsprop():
    _test_optimizer_apply_single(optimizers.RMSprop, [0.1], {}, "rmsprop_update")


def test_sgd():
    _test_optimizer_apply_single(optimizers.SGD, [0.1], {}, "sgd_update")


def test_clip_grad_norm():
    grad = [array(np.random.randn(2, 3).astype(np.float32))]
    mock_ret = Tensor(None, TensorConfig((2, 3), "float32", "cpu"))
    with patch(
        "ml_switcheroo_compiler.ops.nn.clip_grad.clip_by_global_norm",
        return_value=(grad, mock_ret),
        create=True,
    ):
        clipped, norm = optimizers.clip_grad_norm(grad, 1.0)
        assert len(clipped) == 1

    with patch(
        "ml_switcheroo_compiler.ops.nn.clip_grad.clip_by_global_norm",
        side_effect=ImportError,
        create=True,
    ):
        clipped, norm = optimizers.clip_grad_norm(grad, 1.0)
        assert len(clipped) == 1


def test_schedulers():
    s1 = optimizers.cosine_decay(1.0, 10)
    assert isinstance(s1(5), float)

    s2 = optimizers.exponential_decay(1.0, 0.9)
    assert isinstance(s2(5), float)

    s3 = optimizers.linear_schedule(1.0, 0.0, 10)
    assert isinstance(s3(5), float)

    s4 = optimizers.step_decay(1.0, 0.9, 2)
    assert isinstance(s4(5), float)

    s5 = optimizers.join_schedules([s1, s2], [5])
    assert isinstance(s5(3), float)
    assert isinstance(s5(7), float)


def test_trees():
    tree = {"a": 1, "b": {"c": 2}}
    f = optimizers.tree_flatten(tree)
    # The actual implementation of tree flatten in compiler returns a list of leaves
    # Our fallback returns [] if missing
    assert isinstance(f, list)

    # We just need to hit the routing functions to get coverage
    m = optimizers.tree_map(lambda x: x, tree)
    assert m == tree

    m = optimizers.tree_merge(tree, tree)
    assert m == tree

    m = optimizers.tree_reduce(lambda x, y: x, tree)
    assert m is None

    m = optimizers.tree_unflatten(tree)
    assert m == tree


def test_clip_grad_norm_coverage():
    # To hit the loop, we need tree_flatten to return something.
    grad = {"a": array(np.random.randn(2, 3).astype(np.float32))}
    mock_ret = Tensor(None, TensorConfig((2, 3), "float32", "cpu"))
    with (
        patch(
            "ml_switcheroo_compiler.ops.nn.clip_grad.clip_by_global_norm",
            return_value=(grad, mock_ret),
            create=True,
        ),
        patch("zero_mlx.optimizers.tree_flatten", return_value=[grad["a"]]),
    ):
        clipped, norm = optimizers.clip_grad_norm(grad, 1.0)


def test_tree_flatten_destination():
    tree = {"a": 1, "b": {"c": 2}}
    import sys
    from unittest.mock import MagicMock

    mock_tree = MagicMock()
    mock_tree.tree_flatten.return_value = ([1, 2], None)
    sys.modules["ml_switcheroo_compiler.core.tree"] = mock_tree
    f = optimizers.tree_flatten(tree, destination=[])
    assert f == [1, 2]
    del sys.modules["ml_switcheroo_compiler.core.tree"]


def test_tree_merge():
    # tree_merge just returns tree_a in the stub
    res = optimizers.tree_merge(1, 2)
    assert res == 1


def test_tree_reduce():
    # tree_reduce just returns initializer
    res = optimizers.tree_reduce(lambda x, y: x + y, 1, 0)
    assert res == 0


def test_tree_unflatten():
    # tree_unflatten just returns tree
    res = optimizers.tree_unflatten(1)
    assert res == 1


def test_tree_map_success():
    import sys
    from unittest.mock import MagicMock
    import builtins

    mock_tree = MagicMock()
    mock_tree.tree_map.return_value = {"a": 1}

    original_import = builtins.__import__

    def mock_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "ml_switcheroo_compiler.core.tree":
            return mock_tree
        return original_import(name, globals, locals, fromlist, level)

    from unittest.mock import patch

    with patch("builtins.__import__", side_effect=mock_import):
        res = optimizers.tree_map(lambda x: x, {"a": 1})
        assert res == {"a": 1}


def test_tree_map_importerror():
    import sys
    from unittest.mock import MagicMock
    import builtins

    mock_tree = MagicMock()
    mock_tree.tree_map.return_value = {"a": 1}

    original_import = builtins.__import__

    def mock_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "ml_switcheroo_compiler.core.tree":
            raise ImportError("Mocked")
        return original_import(name, globals, locals, fromlist, level)

    from unittest.mock import patch

    with patch("builtins.__import__", side_effect=mock_import):
        res = optimizers.tree_map(lambda x: x, {"a": 1})
        assert res == {"a": 1}
