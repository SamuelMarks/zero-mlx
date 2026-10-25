"""Unit tests for `zero_mlx.nn.activations` functions and module classes."""

from __future__ import annotations

import math
from typing import Callable

import zero_mlx as mx
from zero_mlx._types import ArrayInput
from zero_mlx.array import array
import zero_mlx.nn as z_nn
from zero_mlx.nn.base import Module

ActivationFn = Callable[[ArrayInput], array]


def _test_func_shape(
    func: ActivationFn | Module,
    x_shape: tuple[int, ...],
    expected_shape: tuple[int, ...],
) -> None:
    """Invoke `func` on a deterministic array of `x_shape` and assert its output shape and finite numerical values.

    Args:
        func: Activation function or `Module` to test.
        x_shape: Input array shape.
        expected_shape: Expected output array shape.

    """
    x = mx.linspace(-1.5, 1.5, math.prod(x_shape), dtype=mx.float32).reshape(x_shape)
    raw_out: array | tuple[array, ...] = (
        func._call_forward(x) if isinstance(func, Module) else func(x)
    )
    out: array = raw_out[0] if isinstance(raw_out, tuple) else raw_out
    assert out.shape == expected_shape
    assert bool(mx.all(mx.isfinite(out)).item()) is True
    raw_out2: array | tuple[array, ...] = (
        func._call_forward(x) if isinstance(func, Module) else func(x)
    )
    out2: array = raw_out2[0] if isinstance(raw_out2, tuple) else raw_out2
    assert bool(mx.allclose(out, out2, rtol=1e-5, atol=1e-5).item()) is True


def test_celu() -> None:
    """Test `celu` and `CELU`."""
    _test_func_shape(z_nn.celu, (2, 3), (2, 3))
    _test_func_shape(z_nn.CELU(), (2, 3), (2, 3))


def test_elu() -> None:
    """Test `elu` and `ELU`."""
    _test_func_shape(z_nn.elu, (2, 3), (2, 3))
    _test_func_shape(z_nn.ELU(), (2, 3), (2, 3))


def test_gelu() -> None:
    """Test `gelu` and `GELU` across approximation modes."""
    _test_func_shape(z_nn.gelu, (2, 3), (2, 3))
    _test_func_shape(z_nn.GELU(), (2, 3), (2, 3))
    _test_func_shape(z_nn.GELU("approx"), (2, 3), (2, 3))
    _test_func_shape(z_nn.GELU("fast"), (2, 3), (2, 3))


def test_glu() -> None:
    """Test `glu` and `GLU`."""
    _test_func_shape(z_nn.glu, (2, 4), (2, 2))
    _test_func_shape(z_nn.GLU(), (2, 4), (2, 2))


def test_leaky_relu() -> None:
    """Test `leaky_relu` and `LeakyReLU`."""
    _test_func_shape(z_nn.leaky_relu, (2, 3), (2, 3))
    _test_func_shape(z_nn.LeakyReLU(), (2, 3), (2, 3))


def test_log_sigmoid() -> None:
    """Test `log_sigmoid` and `LogSigmoid`."""
    _test_func_shape(z_nn.log_sigmoid, (2, 3), (2, 3))
    _test_func_shape(z_nn.LogSigmoid(), (2, 3), (2, 3))


def test_log_softmax() -> None:
    """Test `log_softmax` and `LogSoftmax`."""
    _test_func_shape(z_nn.log_softmax, (2, 3), (2, 3))
    _test_func_shape(z_nn.LogSoftmax(), (2, 3), (2, 3))


def test_mish() -> None:
    """Test `mish` and `Mish`."""
    _test_func_shape(z_nn.mish, (2, 3), (2, 3))
    _test_func_shape(z_nn.Mish(), (2, 3), (2, 3))


def test_prelu() -> None:
    """Test `prelu` and `PReLU`."""
    w = array([0.1, 0.2, 0.3], dtype=mx.float32)
    _test_func_shape(lambda x: z_nn.prelu(x, w), (2, 3), (2, 3))
    _test_func_shape(z_nn.PReLU(3), (2, 3), (2, 3))


def test_relu() -> None:
    """Test `relu` and `ReLU`."""
    _test_func_shape(z_nn.relu, (2, 3), (2, 3))
    _test_func_shape(z_nn.ReLU(), (2, 3), (2, 3))


def test_relu2() -> None:
    """Test `relu2` and `ReLU2`."""
    _test_func_shape(z_nn.relu2, (2, 3), (2, 3))
    _test_func_shape(z_nn.ReLU2(), (2, 3), (2, 3))


def test_relu6() -> None:
    """Test `relu6` and `ReLU6`."""
    _test_func_shape(z_nn.relu6, (2, 3), (2, 3))
    _test_func_shape(z_nn.ReLU6(), (2, 3), (2, 3))


def test_selu() -> None:
    """Test `selu` and `SELU`."""
    _test_func_shape(z_nn.selu, (2, 3), (2, 3))
    _test_func_shape(z_nn.SELU(), (2, 3), (2, 3))


def test_silu() -> None:
    """Test `silu` and `SiLU`."""
    _test_func_shape(z_nn.silu, (2, 3), (2, 3))
    _test_func_shape(z_nn.SiLU(), (2, 3), (2, 3))


def test_sigmoid() -> None:
    """Test `sigmoid` and `Sigmoid`."""
    _test_func_shape(z_nn.sigmoid, (2, 3), (2, 3))
    _test_func_shape(z_nn.Sigmoid(), (2, 3), (2, 3))


def test_softmax() -> None:
    """Test `softmax` and `Softmax`."""
    _test_func_shape(z_nn.softmax, (2, 3), (2, 3))
    _test_func_shape(z_nn.Softmax(), (2, 3), (2, 3))


def test_softmin() -> None:
    """Test `softmin` and `Softmin`."""
    _test_func_shape(z_nn.softmin, (2, 3), (2, 3))
    _test_func_shape(z_nn.Softmin(), (2, 3), (2, 3))


def test_softplus() -> None:
    """Test `softplus` and `Softplus`."""
    _test_func_shape(z_nn.softplus, (2, 3), (2, 3))
    _test_func_shape(z_nn.Softplus(), (2, 3), (2, 3))


def test_softshrink() -> None:
    """Test `softshrink` and `Softshrink`."""
    _test_func_shape(z_nn.softshrink, (2, 3), (2, 3))
    _test_func_shape(z_nn.Softshrink(), (2, 3), (2, 3))


def test_softsign() -> None:
    """Test `softsign` and `Softsign`."""
    _test_func_shape(z_nn.softsign, (2, 3), (2, 3))
    _test_func_shape(z_nn.Softsign(), (2, 3), (2, 3))


def test_step() -> None:
    """Test `step` and `Step`."""
    _test_func_shape(z_nn.step, (2, 3), (2, 3))
    _test_func_shape(z_nn.Step(), (2, 3), (2, 3))


def test_tanh() -> None:
    """Test `tanh` and `Tanh`."""
    _test_func_shape(z_nn.tanh, (2, 3), (2, 3))
    _test_func_shape(z_nn.Tanh(), (2, 3), (2, 3))


def test_hard_shrink() -> None:
    """Test `hard_shrink` and `HardShrink`."""
    _test_func_shape(z_nn.hard_shrink, (2, 3), (2, 3))
    _test_func_shape(z_nn.HardShrink(), (2, 3), (2, 3))


def test_hard_tanh() -> None:
    """Test `hard_tanh` and `HardTanh`."""
    _test_func_shape(z_nn.hard_tanh, (2, 3), (2, 3))
    _test_func_shape(z_nn.HardTanh(), (2, 3), (2, 3))


def test_hardswish() -> None:
    """Test `hardswish` and `Hardswish`."""
    _test_func_shape(z_nn.hardswish, (2, 3), (2, 3))
    _test_func_shape(z_nn.Hardswish(), (2, 3), (2, 3))
