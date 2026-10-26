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


def assert_activation_matches(
    fn: ActivationFn,
    mod: Module,
    x: array,
    expected: list[float],
    rtol: float = 1e-4,
    atol: float = 1e-4,
) -> None:
    """Assert both functional `fn(x)` and module `mod(x)` match `expected` values.

    Args:
        fn: Functional activation.
        mod: Activation `Module` instance.
        x: Input 1D `array`.
        expected: Reference scalar outputs computed via closed-form `math` formulas.
        rtol: Relative tolerance.
        atol: Absolute tolerance.

    """
    out_fn = fn(x)
    raw_mod = mod._call_forward(x)
    out_mod = raw_mod[0] if isinstance(raw_mod, tuple) else raw_mod
    exp_arr = array(expected, dtype=mx.float32)
    assert out_fn.shape == exp_arr.shape
    assert out_mod.shape == exp_arr.shape
    assert bool(mx.allclose(out_fn, exp_arr, rtol=rtol, atol=atol).item()) is True
    assert bool(mx.allclose(out_mod, exp_arr, rtol=rtol, atol=atol).item()) is True


def test_celu() -> None:
    """Test `celu` and `CELU` against `max(0, x) + min(0, alpha * (exp(x / alpha) - 1))`."""
    x = array([-2.0, 0.0, 2.0], dtype=mx.float32)
    expected = [math.exp(-2.0) - 1.0, 0.0, 2.0]
    assert_activation_matches(z_nn.celu, z_nn.CELU(1.0), x, expected)


def test_elu() -> None:
    """Test `elu` and `ELU` against `x if x > 0 else alpha * (exp(x) - 1)`."""
    x = array([-2.0, 0.0, 2.0], dtype=mx.float32)
    expected = [math.exp(-2.0) - 1.0, 0.0, 2.0]
    assert_activation_matches(z_nn.elu, z_nn.ELU(1.0), x, expected)


def test_gelu() -> None:
    """Test `gelu` and `GELU` across exact, approx, and fast approximation modes."""
    x = array([-1.0, 0.0, 1.0], dtype=mx.float32)
    exact_exp = [v * 0.5 * (1.0 + math.erf(v / math.sqrt(2.0))) for v in (-1.0, 0.0, 1.0)]
    assert_activation_matches(z_nn.gelu, z_nn.GELU("none"), x, exact_exp)

    c = math.sqrt(2.0 / math.pi)
    approx_exp = [
        0.5 * v * (1.0 + math.tanh(c * (v + 0.044715 * (v**3)))) for v in (-1.0, 0.0, 1.0)
    ]
    assert_activation_matches(z_nn.gelu_approx, z_nn.GELU("approx"), x, approx_exp)

    fast_exp = [v / (1.0 + math.exp(-1.702 * v)) for v in (-1.0, 0.0, 1.0)]
    assert_activation_matches(z_nn.gelu_fast_approx, z_nn.GELU("fast"), x, fast_exp)


def test_glu() -> None:
    """Test `glu` and `GLU` against `a * sigmoid(b)`."""
    x = array([[2.0, 4.0, 0.0, 0.0]], dtype=mx.float32)
    out_fn = z_nn.glu(x, axis=-1)
    out_mod = z_nn.GLU(axis=-1)(x)
    assert out_fn.shape == (1, 2)
    assert bool(mx.allclose(out_fn, array([[1.0, 2.0]]), rtol=1e-5, atol=1e-5).item()) is True
    assert bool(mx.allclose(out_mod, array([[1.0, 2.0]]), rtol=1e-5, atol=1e-5).item()) is True


def test_leaky_relu() -> None:
    """Test `leaky_relu` and `LeakyReLU` against `max(0.01 * x, x)`."""
    x = array([-2.0, 0.0, 2.0], dtype=mx.float32)
    assert_activation_matches(z_nn.leaky_relu, z_nn.LeakyReLU(0.01), x, [-0.02, 0.0, 2.0])


def test_log_sigmoid() -> None:
    """Test `log_sigmoid` and `LogSigmoid` against `-log(1 + exp(-x))`."""
    x = array([-1.0, 0.0, 1.0], dtype=mx.float32)
    expected = [-math.log(1.0 + math.exp(-v)) for v in (-1.0, 0.0, 1.0)]
    assert_activation_matches(z_nn.log_sigmoid, z_nn.LogSigmoid(), x, expected)


def test_log_softmax() -> None:
    """Test `log_softmax` and `LogSoftmax` against `x - log(sum(exp(x)))`."""
    x = array([0.0, 0.0], dtype=mx.float32)
    expected = [-math.log(2.0), -math.log(2.0)]
    assert_activation_matches(z_nn.log_softmax, z_nn.LogSoftmax(), x, expected)


def test_mish() -> None:
    """Test `mish` and `Mish` against `x * tanh(log(1 + exp(x)))`."""
    x = array([-1.0, 0.0, 1.0], dtype=mx.float32)
    expected = [v * math.tanh(math.log(1.0 + math.exp(v))) for v in (-1.0, 0.0, 1.0)]
    assert_activation_matches(z_nn.mish, z_nn.Mish(), x, expected)


def test_prelu() -> None:
    """Test `prelu` and `PReLU` against `x if x >= 0 else w * x`."""
    x = array([-2.0, 0.0, 3.0], dtype=mx.float32)
    w = array([0.25, 0.25, 0.25], dtype=mx.float32)
    mod = z_nn.PReLU(3, init=0.25)
    assert_activation_matches(lambda a: z_nn.prelu(a, w), mod, x, [-0.5, 0.0, 3.0])


def test_relu() -> None:
    """Test `relu` and `ReLU` against `max(0, x)`."""
    x = array([-2.0, 0.0, 3.0], dtype=mx.float32)
    assert_activation_matches(z_nn.relu, z_nn.ReLU(), x, [0.0, 0.0, 3.0])


def test_relu2() -> None:
    """Test `relu2` and `ReLU2` against `max(0, x)^2`."""
    x = array([-2.0, 0.0, 3.0], dtype=mx.float32)
    assert_activation_matches(z_nn.relu2, z_nn.ReLU2(), x, [0.0, 0.0, 9.0])


def test_relu6() -> None:
    """Test `relu6` and `ReLU6` against `min(max(0, x), 6)`."""
    x = array([-2.0, 3.0, 8.0], dtype=mx.float32)
    assert_activation_matches(z_nn.relu6, z_nn.ReLU6(), x, [0.0, 3.0, 6.0])


def test_selu() -> None:
    """Test `selu` and `SELU` against scaled exponential linear unit formula."""
    alpha = 1.6732632423543772
    scale = 1.0507009873554805
    x = array([-1.0, 0.0, 2.0], dtype=mx.float32)
    expected = [scale * alpha * (math.exp(-1.0) - 1.0), 0.0, scale * 2.0]
    assert_activation_matches(z_nn.selu, z_nn.SELU(), x, expected)


def test_silu() -> None:
    """Test `silu` and `SiLU` against `x / (1 + exp(-x))`."""
    x = array([-1.0, 0.0, 1.0], dtype=mx.float32)
    expected = [v / (1.0 + math.exp(-v)) for v in (-1.0, 0.0, 1.0)]
    assert_activation_matches(z_nn.silu, z_nn.SiLU(), x, expected)


def test_sigmoid() -> None:
    """Test `sigmoid` and `Sigmoid` against `1 / (1 + exp(-x))`."""
    x = array([-1.0, 0.0, 1.0], dtype=mx.float32)
    expected = [1.0 / (1.0 + math.exp(-v)) for v in (-1.0, 0.0, 1.0)]
    assert_activation_matches(z_nn.sigmoid, z_nn.Sigmoid(), x, expected)


def test_softmax() -> None:
    """Test `softmax` and `Softmax` against `exp(x) / sum(exp(x))`."""
    x = array([0.0, math.log(3.0)], dtype=mx.float32)
    assert_activation_matches(z_nn.softmax, z_nn.Softmax(), x, [0.25, 0.75])


def test_softmin() -> None:
    """Test `softmin` and `Softmin` against `exp(-x) / sum(exp(-x))`."""
    x = array([0.0, math.log(3.0)], dtype=mx.float32)
    assert_activation_matches(z_nn.softmin, z_nn.Softmin(), x, [0.75, 0.25])


def test_softplus() -> None:
    """Test `softplus` and `Softplus` against `log(1 + exp(x))`."""
    x = array([-1.0, 0.0, 1.0], dtype=mx.float32)
    expected = [math.log(1.0 + math.exp(v)) for v in (-1.0, 0.0, 1.0)]
    assert_activation_matches(z_nn.softplus, z_nn.Softplus(), x, expected)


def test_softshrink() -> None:
    """Test `softshrink` and `Softshrink` with threshold `lambd=0.5`."""
    x = array([-1.5, 0.2, 1.5], dtype=mx.float32)
    assert_activation_matches(z_nn.softshrink, z_nn.Softshrink(0.5), x, [-1.0, 0.0, 1.0])


def test_softsign() -> None:
    """Test `softsign` and `Softsign` against `x / (1 + |x|)`."""
    x = array([-1.0, 0.0, 3.0], dtype=mx.float32)
    assert_activation_matches(z_nn.softsign, z_nn.Softsign(), x, [-0.5, 0.0, 0.75])


def test_step() -> None:
    """Test `step` and `Step` against `1.0 if x > threshold else 0.0`."""
    x = array([-1.0, 0.0, 1.0], dtype=mx.float32)
    assert_activation_matches(z_nn.step, z_nn.Step(0.0), x, [0.0, 0.0, 1.0])


def test_tanh() -> None:
    """Test `tanh` and `Tanh` against `math.tanh(x)`."""
    x = array([-1.0, 0.0, 1.0], dtype=mx.float32)
    expected = [math.tanh(v) for v in (-1.0, 0.0, 1.0)]
    assert_activation_matches(z_nn.tanh, z_nn.Tanh(), x, expected)


def test_hard_shrink() -> None:
    """Test `hard_shrink` and `HardShrink` against `x if |x| > lambd else 0`."""
    x = array([-1.0, 0.2, 1.0], dtype=mx.float32)
    assert_activation_matches(z_nn.hard_shrink, z_nn.HardShrink(0.5), x, [-1.0, 0.0, 1.0])


def test_hard_tanh() -> None:
    """Test `hard_tanh` and `HardTanh` against `clip(x, -1, 1)`."""
    x = array([-2.5, 0.5, 2.5], dtype=mx.float32)
    assert_activation_matches(z_nn.hard_tanh, z_nn.HardTanh(), x, [-1.0, 0.5, 1.0])


def test_hardswish() -> None:
    """Test `hardswish` and `Hardswish` against `x * clip(x + 3, 0, 6) / 6`."""
    x = array([-4.0, 0.0, 3.0], dtype=mx.float32)
    assert_activation_matches(z_nn.hardswish, z_nn.Hardswish(), x, [0.0, 0.0, 3.0])
