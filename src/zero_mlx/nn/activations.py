"""Activation functions for neural networks."""

import math
import numpy as np

from .. import array
from .layers import Module


def relu(x: array) -> array:
    """Applies the Rectified Linear Unit."""
    return array(np.maximum(x.data, 0))


class ReLU(Module):
    """Applies the Rectified Linear Unit."""

    def __init__(self) -> None:
        """Initialize the ReLU layer."""
        super().__init__()

    def __call__(self, x: array) -> array:
        """Forward pass."""
        return relu(x)


def sigmoid(x: array) -> array:
    """Applies the sigmoid function."""
    return array(1 / (1 + np.exp(-x.data)))


class Sigmoid(Module):
    """Applies the sigmoid function, element-wise."""

    def __init__(self) -> None:
        """Initialize the Sigmoid layer."""
        super().__init__()

    def __call__(self, x: array) -> array:
        """Forward pass."""
        return sigmoid(x)


def silu(x: array) -> array:
    """Applies the Sigmoid Linear Unit. Also known as Swish."""
    return array(x.data * (1 / (1 + np.exp(-x.data))))


class SiLU(Module):
    """Applies the Sigmoid Linear Unit. Also known as Swish."""

    def __init__(self) -> None:
        """Initialize the SiLU layer."""
        super().__init__()

    def __call__(self, x: array) -> array:
        """Forward pass."""
        return silu(x)


def softmax(x: array, axis: int = -1) -> array:
    """Applies the Softmax function."""
    e_x = np.exp(x.data - np.max(x.data, axis=axis, keepdims=True))
    return array(e_x / e_x.sum(axis=axis, keepdims=True))


class Softmax(Module):
    """Applies the Softmax function."""

    def __init__(self) -> None:
        """Initialize the Softmax layer."""
        super().__init__()

    def __call__(self, x: array) -> array:
        """Forward pass."""
        return softmax(x)


def tanh(x: array) -> array:
    """Applies the hyperbolic tangent function."""
    return array(np.tanh(x.data))


class Tanh(Module):
    """Applies the hyperbolic tangent function."""

    def __init__(self) -> None:
        """Initialize the Tanh layer."""
        super().__init__()

    def __call__(self, x: array) -> array:
        """Forward pass."""
        return tanh(x)


def elu(x: array, alpha: float = 1.0) -> array:
    """Applies the Exponential Linear Unit."""
    return array(np.where(x.data > 0, x.data, alpha * (np.exp(x.data) - 1)))


class ELU(Module):
    """Applies the Exponential Linear Unit."""

    def __init__(self, alpha: float = 1.0) -> None:
        """Initialize the ELU layer."""
        super().__init__()
        self.alpha = alpha

    def __call__(self, x: array) -> array:
        """Forward pass."""
        return elu(x, self.alpha)


def gelu(x: array) -> array:
    """Applies the Gaussian Error Linear Units function."""
    erf_vec = np.vectorize(math.erf)
    return array(x.data * 0.5 * (1.0 + erf_vec(x.data / np.sqrt(2.0))))


class GELU(Module):
    """Applies the Gaussian Error Linear Units."""

    def __init__(self, approx: str = "none") -> None:
        """Initialize the GELU layer."""
        super().__init__()
        self.approx = approx

    def __call__(self, x: array) -> array:
        """Forward pass."""
        if self.approx == "precise" or self.approx == "none":
            return gelu(x)
        elif self.approx == "fast":
            # 0.5 * x * (1 + tanh(sqrt(2 / pi) * (x + 0.044715 * x^3)))
            inner = np.sqrt(2.0 / np.pi) * (x.data + 0.044715 * np.power(x.data, 3))
            return array(0.5 * x.data * (1.0 + np.tanh(inner)))
        else:
            return gelu(x)


class CELU(Module):
    """Applies the Continuously Differentiable Exponential Linear Unit."""

    def __init__(self, alpha: float = 1.0) -> None:
        """Initialize the CELU layer."""
        super().__init__()
        self.alpha = alpha

    def __call__(self, x: array) -> array:
        """Forward pass."""
        return array(
            np.maximum(0, x.data)
            + np.minimum(0, self.alpha * (np.exp(x.data / self.alpha) - 1))
        )


class GLU(Module):
    """Applies the gated linear unit function."""

    def __init__(self, axis: int = -1) -> None:
        """Initialize the GLU layer."""
        super().__init__()
        self.axis = axis

    def __call__(self, x: array) -> array:
        """Forward pass."""
        # Split in half along the given axis
        splits = np.split(x.data, 2, axis=self.axis)
        a, b = splits[0], splits[1]
        return array(a * (1 / (1 + np.exp(-b))))


class LogSigmoid(Module):
    """Applies the Log Sigmoid function."""

    def __init__(self) -> None:
        """Initialize the LogSigmoid layer."""
        super().__init__()

    def __call__(self, x: array) -> array:
        """Forward pass."""
        return array(-np.log(1 + np.exp(-x.data)))


class LogSoftmax(Module):
    """Applies the Log Softmax function."""

    def __init__(self) -> None:
        """Initialize the LogSoftmax layer."""
        super().__init__()

    def __call__(self, x: array) -> array:
        """Forward pass."""
        axis = -1
        e_x = np.exp(x.data - np.max(x.data, axis=axis, keepdims=True))
        softmax_val = e_x / e_x.sum(axis=axis, keepdims=True)
        return array(np.log(softmax_val))


class Mish(Module):
    """Applies the Mish function, element-wise."""

    def __init__(self) -> None:
        """Initialize the Mish layer."""
        super().__init__()

    def __call__(self, x: array) -> array:
        """Forward pass."""
        return array(x.data * np.tanh(np.log(1 + np.exp(x.data))))


class PReLU(Module):
    """Applies the element-wise parametric ReLU."""

    def __init__(self, num_parameters: int = 1, init: float = 0.25) -> None:
        """Initialize the PReLU layer."""
        super().__init__()
        self.num_parameters = num_parameters
        self.weight = np.full(num_parameters, init)

    def __call__(self, x: array) -> array:
        """Forward pass."""
        return array(np.maximum(0, x.data) + self.weight * np.minimum(0, x.data))


class ReLU2(Module):
    """Applies the ReLU² activation function."""

    def __init__(self) -> None:
        """Initialize the ReLU2 layer."""
        super().__init__()

    def __call__(self, x: array) -> array:
        """Forward pass."""
        relu_x = np.maximum(0, x.data)
        return array(relu_x**2)


class ReLU6(Module):
    """Applies the Rectified Linear Unit 6."""

    def __init__(self) -> None:
        """Initialize the ReLU6 layer."""
        super().__init__()

    def __call__(self, x: array) -> array:
        """Forward pass."""
        return array(np.minimum(np.maximum(0, x.data), 6))


class SELU(Module):
    """Applies the Scaled Exponential Linear Unit."""

    def __init__(self) -> None:
        """Initialize the SELU layer."""
        super().__init__()
        self.alpha = 1.6732632423543772848170429916717
        self.scale = 1.0507009873554804934193349852946

    def __call__(self, x: array) -> array:
        """Forward pass."""
        return array(
            self.scale
            * (np.maximum(0, x.data) + np.minimum(0, self.alpha * (np.exp(x.data) - 1)))
        )


class HardShrink(Module):
    """Applies the HardShrink function."""

    def __init__(self, lambd: float = 0.5) -> None:
        """Initialize."""
        super().__init__()
        self.lambd = lambd

    def __call__(self, x: array) -> array:
        """Forward pass."""
        return array(
            np.where((x.data > self.lambd) | (x.data < -self.lambd), x.data, 0.0)
        )


class HardTanh(Module):
    """Applies the HardTanh function."""

    def __init__(self) -> None:
        """Initialize."""
        super().__init__()

    def __call__(self, x: array) -> array:
        """Forward pass."""
        return array(np.clip(x.data, -1.0, 1.0))


class Hardswish(Module):
    """Applies the hardswish function, element-wise."""

    def __init__(self) -> None:
        """Initialize."""
        super().__init__()

    def __call__(self, x: array) -> array:
        """Forward pass."""
        inner = np.clip(x.data + 3.0, 0.0, 6.0)
        return array(x.data * inner / 6.0)


class LeakyReLU(Module):
    """Applies the Leaky Rectified Linear Unit."""

    def __init__(self, negative_slope: float = 0.01) -> None:
        """Initialize."""
        super().__init__()
        self.negative_slope = negative_slope

    def __call__(self, x: array) -> array:
        """Forward pass."""
        return array(
            np.maximum(0, x.data) + self.negative_slope * np.minimum(0, x.data)
        )


class Softshrink(Module):
    """Applies the Softshrink function."""

    def __init__(self, lambd: float = 0.5) -> None:
        """Initialize."""
        super().__init__()
        self.lambd = lambd

    def __call__(self, x: array) -> array:
        """Forward pass."""
        return array(
            np.where(
                x.data > self.lambd,
                x.data - self.lambd,
                np.where(x.data < -self.lambd, x.data + self.lambd, 0.0),
            )
        )


class Softmin(Module):
    """Applies the Softmin function."""

    def __init__(self) -> None:
        """Initialize the Softmin layer."""
        super().__init__()

    def __call__(self, x: array) -> array:
        """Forward pass."""
        neg_x = -x.data
        axis = -1
        e_x = np.exp(neg_x - np.max(neg_x, axis=axis, keepdims=True))
        return array(e_x / e_x.sum(axis=axis, keepdims=True))


class Softplus(Module):
    """Applies the Softplus function."""

    def __init__(self) -> None:
        """Initialize the Softplus layer."""
        super().__init__()

    def __call__(self, x: array) -> array:
        """Forward pass."""
        return array(np.log(1 + np.exp(x.data)))


class Softsign(Module):
    """Applies the Softsign function."""

    def __init__(self) -> None:
        """Initialize the Softsign layer."""
        super().__init__()

    def __call__(self, x: array) -> array:
        """Forward pass."""
        return array(x.data / (1 + np.abs(x.data)))


class Step(Module):
    """Applies the Step Activation Function."""

    def __init__(self, threshold: float = 0.0) -> None:
        """Initialize the Step layer."""
        super().__init__()
        self.threshold = threshold

    def __call__(self, x: array) -> array:
        """Forward pass."""
        return array(np.where(x.data >= self.threshold, 1.0, 0.0))
