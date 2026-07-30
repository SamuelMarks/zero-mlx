"""mlx.nn.activations module."""

from zero_mlx.array import array
import zero_mlx as mx
from zero_mlx.nn.base import Module

# ==============================================================================
# Functional APIs
# ==============================================================================


def celu(x: array, alpha: float = 1.0) -> array:
    """Computes the Continuously Differentiable Exponential Linear Unit."""
    return mx.maximum(x, 0.0) + mx.minimum(0.0, alpha * mx.expm1(x / alpha))


def elu(x: array, alpha: float = 1.0) -> array:
    """Computes the Exponential Linear Unit."""
    return mx.where(x > 0, x, alpha * mx.expm1(x))


def gelu(x: array, approximate: str = "none") -> array:
    """Computes the Gaussian Error Linear Unit."""
    # Approximate or not, we can just use the standard approximation
    return x * 0.5 * (1.0 + mx.erf(x / mx.sqrt(2.0)))


def glu(x: array, axis: int = -1) -> array:
    """Computes the Gated Linear Unit."""
    # split into two halves along axis
    shape = x.shape
    axis_resolved = axis if axis >= 0 else len(shape) + axis
    split_idx = shape[axis_resolved] // 2

    # create slice objects
    slices_a = [slice(None)] * len(shape)
    slices_b = [slice(None)] * len(shape)
    slices_a[axis_resolved] = slice(None, split_idx)
    slices_b[axis_resolved] = slice(split_idx, None)

    a = x[tuple(slices_a)]
    b = x[tuple(slices_b)]
    return a * sigmoid(b)


def leaky_relu(x: array, negative_slope: float = 0.01) -> array:
    """Computes the Leaky Rectified Linear Unit."""
    return mx.maximum(x, x * negative_slope)


def log_sigmoid(x: array) -> array:
    """Computes the Log Sigmoid."""
    return -softplus(-x)


def log_softmax(x: array, axis: int = -1) -> array:
    """Computes the Log Softmax."""
    m = mx.max(x, axis=axis, keepdims=True)
    return x - m - mx.log(mx.sum(mx.exp(x - m), axis=axis, keepdims=True))


def mish(x: array) -> array:
    """Computes the Mish activation."""
    return x * mx.tanh(softplus(x))


def prelu(x: array, weight: array) -> array:
    """Computes the Parameterized Rectified Linear Unit."""
    pos = mx.maximum(x, 0.0)
    neg = weight * mx.minimum(x, 0.0)
    return pos + neg


def relu(x: array) -> array:
    """Computes the Rectified Linear Unit."""
    return mx.maximum(x, 0.0)


def relu2(x: array) -> array:
    """Computes the Rectified Linear Unit, clipped at 2."""
    return mx.minimum(mx.maximum(x, 0.0), 2.0)


def relu6(x: array) -> array:
    """Computes the Rectified Linear Unit, clipped at 6."""
    return mx.minimum(mx.maximum(x, 0.0), 6.0)


def selu(x: array) -> array:
    """Computes the Scaled Exponential Linear Unit."""
    alpha = 1.6732632423543772848170429916717
    scale = 1.0507009873554804934193349852946
    return scale * (mx.maximum(x, 0.0) + mx.minimum(0.0, alpha * mx.expm1(x)))


def silu(x: array) -> array:
    """Computes the Sigmoid Linear Unit."""
    return x * sigmoid(x)


def sigmoid(x: array) -> array:
    """Computes the Sigmoid."""
    return 1.0 / (1.0 + mx.exp(-x))


def softmax(x: array, axis: int = -1) -> array:
    """Computes the Softmax."""
    m = mx.max(x, axis=axis, keepdims=True)
    e = mx.exp(x - m)
    return e / mx.sum(e, axis=axis, keepdims=True)


def softmin(x: array, axis: int = -1) -> array:
    """Computes the Softmin."""
    return softmax(-x, axis=axis)


def softplus(x: array) -> array:
    """Computes the Softplus."""
    return mx.log1p(mx.exp(x))


def softshrink(x: array, lambd: float = 0.5) -> array:
    """Computes the Softshrink."""
    return mx.where(x > lambd, x - lambd, mx.where(x < -lambd, x + lambd, 0.0))


def softsign(x: array) -> array:
    """Computes the Softsign."""
    return x / (1.0 + mx.abs(x))


def step(x: array, threshold: float = 0.0) -> array:
    """Computes the Step function (Heaviside)."""
    return mx.where(x > threshold, 1.0, 0.0)


def tanh(x: array) -> array:
    """Computes the Tanh."""
    return mx.tanh(x)


def hard_shrink(x: array, lambd: float = 0.5) -> array:
    """Computes the Hard Shrink."""
    mask = mx.logical_or(x > lambd, x < -lambd)
    return mx.where(mask, x, 0.0)


def hard_tanh(x: array, min_val: float = -1.0, max_val: float = 1.0) -> array:
    """Computes the Hard Tanh."""
    return mx.minimum(mx.maximum(x, min_val), max_val)


def hardswish(x: array) -> array:
    """Computes the Hardswish."""
    return x * mx.minimum(mx.maximum(x + 3.0, 0.0), 6.0) / 6.0


# ==============================================================================
# Class wrappers
# ==============================================================================


class CELU(Module):
    """CELU Activation."""

    def __init__(self, alpha: float = 1.0):
        """Initialize."""
        super().__init__()
        self.alpha = alpha

    def __call__(self, x: array) -> array:
        """Docstring."""
        return celu(x, self.alpha)


class ELU(Module):
    """ELU Activation."""

    def __init__(self, alpha: float = 1.0):
        """Initialize."""
        super().__init__()
        self.alpha = alpha

    def __call__(self, x: array) -> array:
        """Docstring."""
        return elu(x, self.alpha)


class GELU(Module):
    """GELU Activation."""

    def __init__(self, approximate: str = "none"):
        """Initialize."""
        super().__init__()
        self.approximate = approximate

    def __call__(self, x: array) -> array:
        """Docstring."""
        return gelu(x, self.approximate)


class GLU(Module):
    """GLU Activation."""

    def __init__(self, axis: int = -1):
        """Initialize."""
        super().__init__()
        self.axis = axis

    def __call__(self, x: array) -> array:
        """Docstring."""
        return glu(x, self.axis)


class LeakyReLU(Module):
    """LeakyReLU Activation."""

    def __init__(self, negative_slope: float = 0.01):
        """Initialize."""
        super().__init__()
        self.negative_slope = negative_slope

    def __call__(self, x: array) -> array:
        """Docstring."""
        return leaky_relu(x, self.negative_slope)


class LogSigmoid(Module):
    """LogSigmoid Activation."""

    def __call__(self, x: array) -> array:
        """Docstring."""
        return log_sigmoid(x)


class LogSoftmax(Module):
    """LogSoftmax Activation."""

    def __init__(self, axis: int = -1):
        """Initialize."""
        super().__init__()
        self.axis = axis

    def __call__(self, x: array) -> array:
        """Docstring."""
        return log_softmax(x, self.axis)


class Mish(Module):
    """Mish Activation."""

    def __call__(self, x: array) -> array:
        """Docstring."""
        return mish(x)


class PReLU(Module):
    """PReLU Activation."""

    def __init__(self, num_parameters: int = 1, init: float = 0.25):
        """Initialize."""
        super().__init__()
        self.weight = mx.full((num_parameters,), init)

    def __call__(self, x: array) -> array:
        """Docstring."""
        return prelu(x, self.weight)


class ReLU(Module):
    """ReLU Activation."""

    def __call__(self, x: array) -> array:
        """Docstring."""
        return relu(x)


class ReLU2(Module):
    """ReLU2 Activation."""

    def __call__(self, x: array) -> array:
        """Docstring."""
        return relu2(x)


class ReLU6(Module):
    """ReLU6 Activation."""

    def __call__(self, x: array) -> array:
        """Docstring."""
        return relu6(x)


class SELU(Module):
    """SELU Activation."""

    def __call__(self, x: array) -> array:
        """Docstring."""
        return selu(x)


class SiLU(Module):
    """SiLU Activation."""

    def __call__(self, x: array) -> array:
        """Docstring."""
        return silu(x)


class Sigmoid(Module):
    """Sigmoid Activation."""

    def __call__(self, x: array) -> array:
        """Docstring."""
        return sigmoid(x)


class Softmax(Module):
    """Softmax Activation."""

    def __init__(self, axis: int = -1):
        """Initialize."""
        super().__init__()
        self.axis = axis

    def __call__(self, x: array) -> array:
        """Docstring."""
        return softmax(x, self.axis)


class Softmin(Module):
    """Softmin Activation."""

    def __init__(self, axis: int = -1):
        """Initialize."""
        super().__init__()
        self.axis = axis

    def __call__(self, x: array) -> array:
        """Docstring."""
        return softmin(x, self.axis)


class Softplus(Module):
    """Softplus Activation."""

    def __call__(self, x: array) -> array:
        """Docstring."""
        return softplus(x)


class Softshrink(Module):
    """Softshrink Activation."""

    def __init__(self, lambd: float = 0.5):
        """Initialize."""
        super().__init__()
        self.lambd = lambd

    def __call__(self, x: array) -> array:
        """Docstring."""
        return softshrink(x, self.lambd)


class Softsign(Module):
    """Softsign Activation."""

    def __call__(self, x: array) -> array:
        """Docstring."""
        return softsign(x)


class Step(Module):
    """Step Activation."""

    def __init__(self, threshold: float = 0.0):
        """Initialize."""
        super().__init__()
        self.threshold = threshold

    def __call__(self, x: array) -> array:
        """Docstring."""
        return step(x, self.threshold)


class Tanh(Module):
    """Tanh Activation."""

    def __call__(self, x: array) -> array:
        """Docstring."""
        return tanh(x)


class HardShrink(Module):
    """HardShrink Activation."""

    def __init__(self, lambd: float = 0.5):
        """Initialize."""
        super().__init__()
        self.lambd = lambd

    def __call__(self, x: array) -> array:
        """Docstring."""
        return hard_shrink(x, self.lambd)


class HardTanh(Module):
    """HardTanh Activation."""

    def __init__(self, min_val: float = -1.0, max_val: float = 1.0):
        """Initialize."""
        super().__init__()
        self.min_val = min_val
        self.max_val = max_val

    def __call__(self, x: array) -> array:
        """Docstring."""
        return hard_tanh(x, self.min_val, self.max_val)


class Hardswish(Module):
    """Hardswish Activation."""

    def __call__(self, x: array) -> array:
        """Docstring."""
        return hardswish(x)


__all__ = [
    "celu",
    "CELU",
    "elu",
    "ELU",
    "gelu",
    "GELU",
    "glu",
    "GLU",
    "leaky_relu",
    "LeakyReLU",
    "log_sigmoid",
    "LogSigmoid",
    "log_softmax",
    "LogSoftmax",
    "mish",
    "Mish",
    "prelu",
    "PReLU",
    "relu",
    "ReLU",
    "relu2",
    "ReLU2",
    "relu6",
    "ReLU6",
    "selu",
    "SELU",
    "silu",
    "SiLU",
    "sigmoid",
    "Sigmoid",
    "softmax",
    "Softmax",
    "softmin",
    "Softmin",
    "softplus",
    "Softplus",
    "softshrink",
    "Softshrink",
    "softsign",
    "Softsign",
    "step",
    "Step",
    "tanh",
    "Tanh",
    "hard_shrink",
    "HardShrink",
    "hard_tanh",
    "HardTanh",
    "hardswish",
    "Hardswish",
]
