"""mlx.nn.activations module."""

from zero_mlx.array import array
from zero_mlx.nn.base import Module
import ml_switcheroo_compiler.ops as sops
from typing import Optional, Union, Any

# ==============================================================================
# Functional APIs
# ==============================================================================


def celu(x: array, alpha: float = 1.0) -> array:
    """Computes the Continuously Differentiable Exponential Linear Unit."""
    x_t = x._tensor if hasattr(x, "_tensor") else x
    return array(sops.celu(x_t, alpha=alpha))


def elu(x: array, alpha: float = 1.0) -> array:
    """Computes the Exponential Linear Unit."""
    x_t = x._tensor if hasattr(x, "_tensor") else x
    return array(sops.elu(x_t, alpha=alpha))


def gelu(x: array, approximate: str = "none") -> array:
    """Computes the Gaussian Error Linear Unit."""
    x_t = x._tensor if hasattr(x, "_tensor") else x
    approx = True if approximate != "none" else False
    return array(sops.gelu(x_t, approximate=approx))


def glu(x: array, axis: int = -1) -> array:
    """Computes the Gated Linear Unit."""
    x_t = x._tensor if hasattr(x, "_tensor") else x
    return array(sops.glu(x_t, axis=axis))


def leaky_relu(x: array, negative_slope: float = 0.01) -> array:
    """Computes the Leaky Rectified Linear Unit."""
    x_t = x._tensor if hasattr(x, "_tensor") else x
    return array(sops.leaky_relu(x_t, alpha=negative_slope))


def log_sigmoid(x: array) -> array:
    """Computes the Log Sigmoid."""
    x_t = x._tensor if hasattr(x, "_tensor") else x
    return array(sops.log_sigmoid(x_t))


def log_softmax(x: array, axis: int = -1) -> array:
    """Computes the Log Softmax."""
    x_t = x._tensor if hasattr(x, "_tensor") else x
    return array(sops.log_softmax(x_t, axis=axis))


def mish(x: array) -> array:
    """Computes the Mish activation."""
    x_t = x._tensor if hasattr(x, "_tensor") else x
    return array(sops.multiply(x_t, sops.tanh(sops.softplus(x_t))))


def prelu(x: array, weight: array) -> array:
    """Computes the Parameterized Rectified Linear Unit."""
    x_t = x._tensor if hasattr(x, "_tensor") else x
    w_t = weight._tensor if hasattr(weight, "_tensor") else weight
    # PReLU: max(0, x) + w * min(0, x)
    pos = sops.relu(x_t)
    neg = sops.multiply(w_t, sops.minimum(x_t, sops.zeros_like(x_t)))
    return array(sops.add(pos, neg))


def relu(x: array) -> array:
    """Computes the Rectified Linear Unit."""
    x_t = x._tensor if hasattr(x, "_tensor") else x
    return array(sops.relu(x_t))


def relu2(x: array) -> array:
    """Computes the Rectified Linear Unit, clipped at 2."""
    x_t = x._tensor if hasattr(x, "_tensor") else x
    out = sops.minimum(
        sops.maximum(x_t, sops.zeros_like(x_t)), sops.full_like(x_t, 2.0)
    )
    return array(out)


def relu6(x: array) -> array:
    """Computes the Rectified Linear Unit, clipped at 6."""
    x_t = x._tensor if hasattr(x, "_tensor") else x
    return array(sops.relu6(x_t))


def selu(x: array) -> array:
    """Computes the Scaled Exponential Linear Unit."""
    x_t = x._tensor if hasattr(x, "_tensor") else x
    return array(sops.selu(x_t))


def silu(x: array) -> array:
    """Computes the Sigmoid Linear Unit."""
    x_t = x._tensor if hasattr(x, "_tensor") else x
    return array(sops.silu(x_t))


def sigmoid(x: array) -> array:
    """Computes the Sigmoid."""
    x_t = x._tensor if hasattr(x, "_tensor") else x
    return array(sops.sigmoid(x_t))


def softmax(x: array, axis: int = -1) -> array:
    """Computes the Softmax."""
    x_t = x._tensor if hasattr(x, "_tensor") else x
    return array(sops.softmax(x_t, axis=axis))


def softmin(x: array, axis: int = -1) -> array:
    """Computes the Softmin."""
    x_t = x._tensor if hasattr(x, "_tensor") else x
    return array(sops.softmax(sops.negative(x_t), axis=axis))


def softplus(x: array) -> array:
    """Computes the Softplus."""
    x_t = x._tensor if hasattr(x, "_tensor") else x
    return array(sops.softplus(x_t))


def softshrink(x: array, lambd: float = 0.5) -> array:
    """Computes the Softshrink."""
    x_t = x._tensor if hasattr(x, "_tensor") else x
    # sops may not have softshrink natively, implement via primitives
    lam_t = sops.full_like(x_t, lambd)
    n_lam_t = sops.full_like(x_t, -lambd)

    pos_mask = sops.greater(x_t, lam_t)
    neg_mask = sops.less(x_t, n_lam_t)

    pos_val = sops.multiply(pos_mask, sops.subtract(x_t, lam_t))
    neg_val = sops.multiply(neg_mask, sops.add(x_t, lam_t))

    return array(sops.add(pos_val, neg_val))


def softsign(x: array) -> array:
    """Computes the Softsign."""
    x_t = x._tensor if hasattr(x, "_tensor") else x
    return array(sops.softsign(x_t))


def step(x: array, threshold: float = 0.0) -> array:
    """Computes the Step function (Heaviside)."""
    x_t = x._tensor if hasattr(x, "_tensor") else x
    thresh_t = sops.full_like(x_t, threshold)
    return array(sops.heaviside(x_t, thresh_t))


def tanh(x: array) -> array:
    """Computes the Tanh."""
    x_t = x._tensor if hasattr(x, "_tensor") else x
    return array(sops.tanh(x_t))


def hard_shrink(x: array, lambd: float = 0.5) -> array:
    """Computes the Hard Shrink."""
    x_t = x._tensor if hasattr(x, "_tensor") else x
    if hasattr(sops, "hard_shrink"):
        return array(sops.hard_shrink(x_t, lambd=lambd))
    # Primitive fallback
    lam_t = sops.full_like(x_t, lambd)
    n_lam_t = sops.full_like(x_t, -lambd)
    pos_mask = sops.greater(x_t, lam_t)
    neg_mask = sops.less(x_t, n_lam_t)
    mask = sops.logical_or(pos_mask, neg_mask)
    return array(sops.where(mask, x_t, sops.zeros_like(x_t)))


def hard_tanh(x: array, min_val: float = -1.0, max_val: float = 1.0) -> array:
    """Computes the Hard Tanh."""
    x_t = x._tensor if hasattr(x, "_tensor") else x
    if hasattr(sops, "hard_tanh"):
        return array(sops.hard_tanh(x_t, min_val=min_val, max_val=max_val))
    return array(
        sops.minimum(
            sops.maximum(x_t, sops.full_like(x_t, min_val)),
            sops.full_like(x_t, max_val),
        )
    )


def hardswish(x: array) -> array:
    """Computes the Hardswish."""
    x_t = x._tensor if hasattr(x, "_tensor") else x
    # hardswish(x) = x * relu6(x + 3) / 6
    if hasattr(sops, "hardswish"):
        return array(sops.hardswish(x_t))
    out = sops.add(x_t, sops.full_like(x_t, 3.0))
    out = sops.relu6(out)
    out = sops.multiply(x_t, out)
    out = sops.divide(out, sops.full_like(x_t, 6.0))
    return array(out)


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
        self.weight = array(sops.full((num_parameters,), init))

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
