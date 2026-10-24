"""Activation functions and modules (`mlx.nn.activations`) for zero_mlx."""

from __future__ import annotations

import builtins

from zero_mlx._types import ArrayInput
from zero_mlx.array import array
from zero_mlx.nn.base import Module
from zero_mlx.nn.gelu_approx import gelu_approx
from zero_mlx.nn.gelu_fast_approx import gelu_fast_approx
import zero_mlx.ops as mx


def _as_array(x: ArrayInput) -> array:
    """Convert `x` to a `zero_mlx.array`.

    Args:
        x: Input array or scalar/sequence.

    Returns:
        `zero_mlx.array` instance.

    """
    return x if isinstance(x, array) else array(x)


def celu(x: ArrayInput, alpha: float = 1.0) -> array:
    """Compute the Continuously Differentiable Exponential Linear Unit (`CELU`).

    Args:
        x: Input array.
        alpha: Scale factor for the negative region.

    Returns:
        Activated `array`.

    """
    arr = _as_array(x)
    return mx.maximum(arr, 0.0) + mx.minimum(0.0, alpha * mx.expm1(arr / alpha))


def elu(x: ArrayInput, alpha: float = 1.0) -> array:
    """Compute the Exponential Linear Unit (`ELU`).

    Args:
        x: Input array.
        alpha: Scale factor for negative inputs.

    Returns:
        Activated `array`.

    """
    arr = _as_array(x)
    return mx.where(arr > 0.0, arr, alpha * mx.expm1(arr))


def gelu(x: ArrayInput, approximate: str = "none") -> array:
    """Compute the Gaussian Error Linear Unit (`GELU`).

    Args:
        x: Input array.
        approximate: Approximation mode (`'none'`, `'precise'`, `'tanh'`, or `'fast'`).

    Returns:
        Activated `array`.

    """
    arr = _as_array(x)
    if approximate in ("precise", "tanh", "approx"):
        return gelu_approx(arr)
    if approximate == "fast":
        return gelu_fast_approx(arr)
    return arr * 0.5 * (1.0 + mx.erf(arr / mx.sqrt(2.0)))


def glu(x: ArrayInput, axis: int = -1) -> array:
    """Compute the Gated Linear Unit (`GLU`) by splitting `x` in half along `axis`.

    Args:
        x: Input array whose size along `axis` is even.
        axis: Dimension along which to split `x`.

    Returns:
        Activated `array` of half size along `axis`.

    """
    arr = _as_array(x)
    shape = arr.shape
    ax = axis if axis >= 0 else len(shape) + axis
    split_idx = shape[ax] // 2
    slices_a = [builtins.slice(None)] * len(shape)
    slices_b = [builtins.slice(None)] * len(shape)
    slices_a[ax] = builtins.slice(None, split_idx)
    slices_b[ax] = builtins.slice(split_idx, None)
    a = arr[tuple(slices_a)]
    b = arr[tuple(slices_b)]
    return a * sigmoid(b)


def leaky_relu(x: ArrayInput, negative_slope: float = 0.01) -> array:
    """Compute the Leaky Rectified Linear Unit (`LeakyReLU`).

    Args:
        x: Input array.
        negative_slope: Slope for negative input values.

    Returns:
        Activated `array`.

    """
    arr = _as_array(x)
    return mx.maximum(arr, arr * negative_slope)


def log_sigmoid(x: ArrayInput) -> array:
    """Compute the natural logarithm of the logistic sigmoid `-softplus(-x)`.

    Args:
        x: Input array.

    Returns:
        Log-sigmoid `array`.

    """
    arr = _as_array(x)
    return -softplus(-arr)


def log_softmax(x: ArrayInput, axis: int = -1) -> array:
    """Compute the log-softmax along `axis`.

    Args:
        x: Input array.
        axis: Dimension along which log-softmax is computed.

    Returns:
        Log-softmax `array`.

    """
    arr = _as_array(x)
    m = mx.max(arr, axis=axis, keepdims=True)
    return arr - m - mx.log(mx.sum(mx.exp(arr - m), axis=axis, keepdims=True))


def mish(x: ArrayInput) -> array:
    """Compute the Mish activation `x * tanh(softplus(x))`.

    Args:
        x: Input array.

    Returns:
        Activated `array`.

    """
    arr = _as_array(x)
    return arr * mx.tanh(softplus(arr))


def prelu(x: ArrayInput, alpha: ArrayInput) -> array:
    """Compute the Parameterized Rectified Linear Unit (`PReLU`).

    Args:
        x: Input array.
        alpha: Learnable negative slope array (also accepts `weight` positionally).

    Returns:
        Activated `array`.

    """
    arr = _as_array(x)
    w = _as_array(alpha)
    return mx.maximum(arr, 0.0) + w * mx.minimum(arr, 0.0)


def relu(x: ArrayInput) -> array:
    """Compute the Rectified Linear Unit `max(x, 0)`.

    Args:
        x: Input array.

    Returns:
        Activated `array`.

    """
    arr = _as_array(x)
    return mx.maximum(arr, 0.0)


def relu2(x: ArrayInput) -> array:
    """Compute the Squared ReLU (or ReLU clipped at 2).

    Args:
        x: Input array.

    Returns:
        Activated `array`.

    """
    arr = _as_array(x)
    return mx.minimum(mx.maximum(arr, 0.0), 2.0)


def relu6(x: ArrayInput) -> array:
    """Compute the Rectified Linear Unit clipped at 6 (`min(max(x, 0), 6)`).

    Args:
        x: Input array.

    Returns:
        Activated `array`.

    """
    arr = _as_array(x)
    return mx.minimum(mx.maximum(arr, 0.0), 6.0)


def selu(x: ArrayInput) -> array:
    """Compute the Scaled Exponential Linear Unit (`SELU`).

    Args:
        x: Input array.

    Returns:
        Activated `array`.

    """
    arr = _as_array(x)
    alpha = 1.6732632423543772848170429916717
    scale = 1.0507009873554804934193349852946
    return scale * (mx.maximum(arr, 0.0) + mx.minimum(0.0, alpha * mx.expm1(arr)))


def sigmoid(x: ArrayInput) -> array:
    """Compute the element-wise logistic sigmoid `1 / (1 + exp(-x))`.

    Args:
        x: Input array.

    Returns:
        Activated `array`.

    """
    arr = _as_array(x)
    return 1.0 / (1.0 + mx.exp(-arr))


def silu(x: ArrayInput) -> array:
    """Compute the Sigmoid Linear Unit (`SiLU` / Swish) `x * sigmoid(x)`.

    Args:
        x: Input array.

    Returns:
        Activated `array`.

    """
    arr = _as_array(x)
    return arr * sigmoid(arr)


def softmax(x: ArrayInput, axis: int = -1) -> array:
    """Compute the Softmax activation along `axis`.

    Args:
        x: Input array.
        axis: Dimension along which softmax is computed.

    Returns:
        Softmax `array`.

    """
    arr = _as_array(x)
    m = mx.max(arr, axis=axis, keepdims=True)
    e = mx.exp(arr - m)
    return e / mx.sum(e, axis=axis, keepdims=True)


def softmin(x: ArrayInput, axis: int = -1) -> array:
    """Compute the Softmin activation (`softmax(-x, axis=axis)`).

    Args:
        x: Input array.
        axis: Dimension along which softmin is computed.

    Returns:
        Softmin `array`.

    """
    arr = _as_array(x)
    return softmax(-arr, axis=axis)


def softplus(x: ArrayInput) -> array:
    """Compute the Softplus activation `log(1 + exp(x))`.

    Args:
        x: Input array.

    Returns:
        Softplus `array`.

    """
    arr = _as_array(x)
    return mx.log1p(mx.exp(arr))


def softshrink(x: ArrayInput, lambd: float = 0.5) -> array:
    """Compute the Softshrink activation with threshold `lambd`.

    Args:
        x: Input array.
        lambd: Shrinkage threshold value.

    Returns:
        Activated `array`.

    """
    arr = _as_array(x)
    return mx.where(arr > lambd, arr - lambd, mx.where(arr < -lambd, arr + lambd, 0.0))


def softsign(x: ArrayInput) -> array:
    """Compute the Softsign activation `x / (1 + |x|)`.

    Args:
        x: Input array.

    Returns:
        Activated `array`.

    """
    arr = _as_array(x)
    return arr / (1.0 + mx.abs(arr))


def step(x: ArrayInput, threshold: float = 0.0) -> array:
    """Compute the Heaviside step activation (`1.0` where `x > threshold`, else `0.0`).

    Args:
        x: Input array.
        threshold: Step threshold value.

    Returns:
        Activated `array`.

    """
    arr = _as_array(x)
    return mx.where(arr > threshold, 1.0, 0.0)


def tanh(x: ArrayInput) -> array:
    """Compute the hyperbolic tangent activation `tanh(x)`.

    Args:
        x: Input array.

    Returns:
        Activated `array`.

    """
    arr = _as_array(x)
    return mx.tanh(arr)


def hard_shrink(x: ArrayInput, lambd: float = 0.5) -> array:
    """Compute the Hard Shrink activation with threshold `lambd`.

    Args:
        x: Input array.
        lambd: Threshold within `[-lambd, lambd]` where values are zeroed.

    Returns:
        Activated `array`.

    """
    arr = _as_array(x)
    mask = mx.logical_or(arr > lambd, arr < -lambd)
    return mx.where(mask, arr, 0.0)


def hard_tanh(x: ArrayInput, min_val: float = -1.0, max_val: float = 1.0) -> array:
    """Compute the HardTanh activation clamping `x` to `[min_val, max_val]`.

    Args:
        x: Input array.
        min_val: Minimum clamp value.
        max_val: Maximum clamp value.

    Returns:
        Activated `array`.

    """
    arr = _as_array(x)
    return mx.minimum(mx.maximum(arr, min_val), max_val)


def hardswish(x: ArrayInput) -> array:
    """Compute the Hardswish activation `x * relu6(x + 3) / 6`.

    Args:
        x: Input array.

    Returns:
        Activated `array`.

    """
    arr = _as_array(x)
    return arr * mx.minimum(mx.maximum(arr + 3.0, 0.0), 6.0) / 6.0


class CELU(Module):
    """Applies the Continuously Differentiable Exponential Linear Unit."""

    def __init__(self, alpha: float = 1.0) -> None:
        """Initialize `CELU`.

        Args:
            alpha: Formulation scale parameter.

        """
        super().__init__()
        self.alpha: float = alpha

    def __call__(self, x: array) -> array:
        """Apply CELU to `x`.

        Args:
            x: Input `array`.

        Returns:
            Activated `array`.

        """
        return celu(x, self.alpha)


class ELU(Module):
    """Applies the Exponential Linear Unit."""

    def __init__(self, alpha: float = 1.0) -> None:
        """Initialize `ELU`.

        Args:
            alpha: Negative region scale factor.

        """
        super().__init__()
        self.alpha: float = alpha

    def __call__(self, x: array) -> array:
        """Apply ELU to `x`.

        Args:
            x: Input `array`.

        Returns:
            Activated `array`.

        """
        return elu(x, self.alpha)


class GELU(Module):
    """Applies the Gaussian Error Linear Unit."""

    def __init__(self, approx: str = "none", approximate: str | None = None) -> None:
        """Initialize `GELU`.

        Args:
            approx: Approximation mode (`'none'`, `'precise'`, `'tanh'`, `'fast'`).
            approximate: Optional keyword alias for `approx`.

        """
        super().__init__()
        mode = approximate if approximate is not None else approx
        self.approx: str = mode
        self.approximate: str = mode

    def __call__(self, x: array) -> array:
        """Apply GELU to `x`.

        Args:
            x: Input `array`.

        Returns:
            Activated `array`.

        """
        return gelu(x, self.approximate)


class GLU(Module):
    """Applies the Gated Linear Unit along `axis`."""

    def __init__(self, axis: int = -1) -> None:
        """Initialize `GLU`.

        Args:
            axis: Dimension along which to split the input.

        """
        super().__init__()
        self.axis: int = axis

    def __call__(self, x: array) -> array:
        """Apply GLU to `x`.

        Args:
            x: Input `array`.

        Returns:
            Activated `array`.

        """
        return glu(x, self.axis)


class LeakyReLU(Module):
    """Applies the Leaky Rectified Linear Unit."""

    def __init__(self, negative_slope: float = 0.01) -> None:
        """Initialize `LeakyReLU`.

        Args:
            negative_slope: Negative slope factor.

        """
        super().__init__()
        self.negative_slope: float = negative_slope

    def __call__(self, x: array) -> array:
        """Apply LeakyReLU to `x`.

        Args:
            x: Input `array`.

        Returns:
            Activated `array`.

        """
        return leaky_relu(x, self.negative_slope)


class LogSigmoid(Module):
    """Applies the element-wise Log-Sigmoid activation."""

    def __call__(self, x: array) -> array:
        """Apply LogSigmoid to `x`.

        Args:
            x: Input `array`.

        Returns:
            Activated `array`.

        """
        return log_sigmoid(x)


class LogSoftmax(Module):
    """Applies the Log-Softmax activation along `axis`."""

    def __init__(self, axis: int = -1) -> None:
        """Initialize `LogSoftmax`.

        Args:
            axis: Axis along which to compute log-softmax.

        """
        super().__init__()
        self.axis: int = axis

    def __call__(self, x: array) -> array:
        """Apply LogSoftmax to `x`.

        Args:
            x: Input `array`.

        Returns:
            Activated `array`.

        """
        return log_softmax(x, self.axis)


class Mish(Module):
    """Applies the Mish activation."""

    def __call__(self, x: array) -> array:
        """Apply Mish to `x`.

        Args:
            x: Input `array`.

        Returns:
            Activated `array`.

        """
        return mish(x)


class PReLU(Module):
    """Applies the element-wise Parameterized Rectified Linear Unit (`PReLU`)."""

    def __init__(self, num_parameters: int = 1, init: float = 0.25) -> None:
        """Initialize `PReLU`.

        Args:
            num_parameters: Number of learnable slope parameters.
            init: Initial value for the learnable slope `weight`.

        """
        super().__init__()
        self.weight: array = mx.full((num_parameters,), init)

    def __call__(self, x: array) -> array:
        """Apply PReLU to `x`.

        Args:
            x: Input `array`.

        Returns:
            Activated `array`.

        """
        return prelu(x, self.weight)


class ReLU(Module):
    """Applies the Rectified Linear Unit."""

    def __call__(self, x: array) -> array:
        """Apply ReLU to `x`.

        Args:
            x: Input `array`.

        Returns:
            Activated `array`.

        """
        return relu(x)


class ReLU2(Module):
    """Applies the ReLU2 activation."""

    def __call__(self, x: array) -> array:
        """Apply ReLU2 to `x`.

        Args:
            x: Input `array`.

        Returns:
            Activated `array`.

        """
        return relu2(x)


class ReLU6(Module):
    """Applies the ReLU6 activation."""

    def __call__(self, x: array) -> array:
        """Apply ReLU6 to `x`.

        Args:
            x: Input `array`.

        Returns:
            Activated `array`.

        """
        return relu6(x)


class SELU(Module):
    """Applies the Scaled Exponential Linear Unit."""

    def __call__(self, x: array) -> array:
        """Apply SELU to `x`.

        Args:
            x: Input `array`.

        Returns:
            Activated `array`.

        """
        return selu(x)


class SiLU(Module):
    """Applies the Sigmoid Linear Unit (`SiLU`)."""

    def __call__(self, x: array) -> array:
        """Apply SiLU to `x`.

        Args:
            x: Input `array`.

        Returns:
            Activated `array`.

        """
        return silu(x)


class Sigmoid(Module):
    """Applies the logistic Sigmoid activation."""

    def __call__(self, x: array) -> array:
        """Apply Sigmoid to `x`.

        Args:
            x: Input `array`.

        Returns:
            Activated `array`.

        """
        return sigmoid(x)


class Softmax(Module):
    """Applies the Softmax activation along `axis`."""

    def __init__(self, axis: int = -1) -> None:
        """Initialize `Softmax`.

        Args:
            axis: Dimension along which softmax is computed.

        """
        super().__init__()
        self.axis: int = axis

    def __call__(self, x: array) -> array:
        """Apply Softmax to `x`.

        Args:
            x: Input `array`.

        Returns:
            Activated `array`.

        """
        return softmax(x, self.axis)


class Softmin(Module):
    """Applies the Softmin activation along `axis`."""

    def __init__(self, axis: int = -1) -> None:
        """Initialize `Softmin`.

        Args:
            axis: Dimension along which softmin is computed.

        """
        super().__init__()
        self.axis: int = axis

    def __call__(self, x: array) -> array:
        """Apply Softmin to `x`.

        Args:
            x: Input `array`.

        Returns:
            Activated `array`.

        """
        return softmin(x, self.axis)


class Softplus(Module):
    """Applies the Softplus activation."""

    def __call__(self, x: array) -> array:
        """Apply Softplus to `x`.

        Args:
            x: Input `array`.

        Returns:
            Activated `array`.

        """
        return softplus(x)


class Softshrink(Module):
    """Applies the Softshrink activation."""

    def __init__(self, lambd: float = 0.5) -> None:
        """Initialize `Softshrink`.

        Args:
            lambd: Shrinkage threshold value.

        """
        super().__init__()
        self.lambd: float = lambd

    def __call__(self, x: array) -> array:
        """Apply Softshrink to `x`.

        Args:
            x: Input `array`.

        Returns:
            Activated `array`.

        """
        return softshrink(x, self.lambd)


class Softsign(Module):
    """Applies the Softsign activation."""

    def __call__(self, x: array) -> array:
        """Apply Softsign to `x`.

        Args:
            x: Input `array`.

        Returns:
            Activated `array`.

        """
        return softsign(x)


class Step(Module):
    """Applies the Heaviside Step activation."""

    def __init__(self, threshold: float = 0.0) -> None:
        """Initialize `Step`.

        Args:
            threshold: Threshold above which output is `1.0`.

        """
        super().__init__()
        self.threshold: float = threshold

    def __call__(self, x: array) -> array:
        """Apply Step to `x`.

        Args:
            x: Input `array`.

        Returns:
            Activated `array`.

        """
        return step(x, self.threshold)


class Tanh(Module):
    """Applies the hyperbolic tangent (`Tanh`) activation."""

    def __call__(self, x: array) -> array:
        """Apply Tanh to `x`.

        Args:
            x: Input `array`.

        Returns:
            Activated `array`.

        """
        return tanh(x)


class HardShrink(Module):
    """Applies the HardShrink activation."""

    def __init__(self, lambd: float = 0.5) -> None:
        """Initialize `HardShrink`.

        Args:
            lambd: Threshold value.

        """
        super().__init__()
        self.lambd: float = lambd

    def __call__(self, x: array) -> array:
        """Apply HardShrink to `x`.

        Args:
            x: Input `array`.

        Returns:
            Activated `array`.

        """
        return hard_shrink(x, self.lambd)


class HardTanh(Module):
    """Applies the HardTanh activation."""

    def __init__(self, min_val: float = -1.0, max_val: float = 1.0) -> None:
        """Initialize `HardTanh`.

        Args:
            min_val: Minimum clamp bound.
            max_val: Maximum clamp bound.

        """
        super().__init__()
        self.min_val: float = min_val
        self.max_val: float = max_val

    def __call__(self, x: array) -> array:
        """Apply HardTanh to `x`.

        Args:
            x: Input `array`.

        Returns:
            Activated `array`.

        """
        return hard_tanh(x, self.min_val, self.max_val)


class Hardswish(Module):
    """Applies the Hardswish activation."""

    def __call__(self, x: array) -> array:
        """Apply Hardswish to `x`.

        Args:
            x: Input `array`.

        Returns:
            Activated `array`.

        """
        return hardswish(x)


__all__ = [
    "CELU",
    "ELU",
    "GELU",
    "GLU",
    "HardShrink",
    "HardTanh",
    "Hardswish",
    "LeakyReLU",
    "LogSigmoid",
    "LogSoftmax",
    "Mish",
    "PReLU",
    "ReLU",
    "ReLU2",
    "ReLU6",
    "SELU",
    "SiLU",
    "Sigmoid",
    "Softmax",
    "Softmin",
    "Softplus",
    "Softshrink",
    "Softsign",
    "Step",
    "Tanh",
    "celu",
    "elu",
    "gelu",
    "gelu_approx",
    "gelu_fast_approx",
    "glu",
    "hard_shrink",
    "hard_tanh",
    "hardswish",
    "leaky_relu",
    "log_sigmoid",
    "log_softmax",
    "mish",
    "prelu",
    "relu",
    "relu2",
    "relu6",
    "selu",
    "sigmoid",
    "silu",
    "softmax",
    "softmin",
    "softplus",
    "softshrink",
    "softsign",
    "step",
    "tanh",
]
