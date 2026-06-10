"""Activation functions for neural networks."""

import numpy as np
from .. import array, _to_tensor, _wrap
from .layers import Module
import ml_switcheroo.nn as _nn


def relu(x: array) -> array:
    """Applies the Rectified Linear Unit."""
    return _wrap(_nn.relu(_to_tensor(x)))


class ReLU(Module):
    """Applies the Rectified Linear Unit."""

    def __init__(self) -> None:
        pass

    def __call__(self, x: array) -> array:
        return relu(x)


def sigmoid(x: array) -> array:
    """Applies the sigmoid function."""
    return _wrap(_nn.sigmoid(_to_tensor(x)))


class Sigmoid(Module):
    """Applies the sigmoid function, element-wise."""

    def __init__(self) -> None:
        pass

    def __call__(self, x: array) -> array:
        return sigmoid(x)


def silu(x: array) -> array:
    """Applies the Sigmoid Linear Unit. Also known as Swish."""
    return _wrap(_nn.swish(_to_tensor(x)))


class SiLU(Module):
    """Applies the Sigmoid Linear Unit. Also known as Swish."""

    def __init__(self) -> None:
        pass

    def __call__(self, x: array) -> array:
        return silu(x)


def softmax(x: array, axis: int = -1) -> array:
    """Applies the Softmax function."""
    from ml_switcheroo.core.config import config

    if config.eager_mode:
        data = np.array(_to_tensor(x).data)
        e_x = np.exp(data - np.max(data, axis=axis, keepdims=True))
        res = e_x / e_x.sum(axis=axis, keepdims=True)
        return _wrap(_to_tensor(res))
    return _wrap(_nn.softmax(_to_tensor(x), dim=axis))


class Softmax(Module):
    """Applies the Softmax function."""

    def __init__(self) -> None:
        pass

    def __call__(self, x: array) -> array:
        return softmax(x)


def tanh(x: array) -> array:
    """Applies the hyperbolic tangent function."""
    return _wrap(_nn.tanh(_to_tensor(x)))


class Tanh(Module):
    """Applies the hyperbolic tangent function."""

    def __init__(self) -> None:
        pass

    def __call__(self, x: array) -> array:
        return tanh(x)


def elu(x: array, alpha: float = 1.0) -> array:
    """Applies the Exponential Linear Unit."""
    return _wrap(_nn.elu(_to_tensor(x), alpha))


class ELU(Module):
    """Applies the Exponential Linear Unit."""

    def __init__(self, alpha: float = 1.0) -> None:
        self.alpha = alpha

    def __call__(self, x: array) -> array:
        return elu(x, self.alpha)


def gelu(x: array) -> array:
    """Applies the Gaussian Error Linear Units function."""
    from ml_switcheroo.core.config import config

    if config.eager_mode:
        data = np.array(_to_tensor(x).data)
        from scipy.special import erf

        res = 0.5 * data * (1 + erf(data / np.sqrt(2.0)))
        return _wrap(_to_tensor(res))
    return _wrap(_nn.gelu(_to_tensor(x)))


class GELU(Module):
    """Applies the Gaussian Error Linear Units."""

    def __init__(self, approx: str = "none") -> None:
        self.approx = approx

    def __call__(self, x: array) -> array:
        from ml_switcheroo.core.config import config

        if config.eager_mode:
            data = np.array(_to_tensor(x).data)
            if self.approx == "fast":
                fast_inner = np.sqrt(2.0 / np.pi) * (
                    data + 0.044715 * np.power(data, 3)
                )
                res = 0.5 * data * (1.0 + np.tanh(fast_inner))
                return _wrap(_to_tensor(res))
            from scipy.special import erf

            res = 0.5 * data * (1 + erf(data / np.sqrt(2.0)))
            return _wrap(_to_tensor(res))
        return gelu(x)


class CELU(Module):
    """Applies the Continuously Differentiable Exponential Linear Unit."""

    def __init__(self, alpha: float = 1.0) -> None:
        self.alpha = alpha

    def __call__(self, x: array) -> array:
        return _wrap(_nn.celu(_to_tensor(x), self.alpha))


class GLU(Module):
    """Applies the gated linear unit function."""

    def __init__(self, axis: int = -1) -> None:
        self.axis = axis

    def __call__(self, x: array) -> array:
        from ml_switcheroo.core.config import config

        if config.eager_mode:
            data = np.array(_to_tensor(x).data)
            axis = int(self.axis)
            split_size = data.shape[axis] // 2
            a = np.take(data, range(split_size), axis=axis)
            b = np.take(data, range(split_size, data.shape[axis]), axis=axis)
            from scipy.special import expit

            res = a * expit(b)
            return _wrap(_to_tensor(res))
        return _wrap(_nn.glu(_to_tensor(x), self.axis))


class LogSigmoid(Module):
    """Applies the Log Sigmoid function."""

    def __init__(self) -> None:
        pass

    def __call__(self, x: array) -> array:
        from ml_switcheroo.core.config import config

        if config.eager_mode:
            data = np.array(_to_tensor(x).data)
            res = -np.log1p(np.exp(-data))
            return _wrap(_to_tensor(res))
        return _wrap(_nn.log_softmax(_to_tensor(x), dim=-1))


class LogSoftmax(Module):
    """Applies the Log Softmax function."""

    def __init__(self) -> None:
        pass

    def __call__(self, x: array) -> array:
        from ml_switcheroo.core.config import config

        if config.eager_mode:
            data = np.array(_to_tensor(x).data)
            c = np.max(data, axis=-1, keepdims=True)
            res = data - c - np.log(np.sum(np.exp(data - c), axis=-1, keepdims=True))
            return _wrap(_to_tensor(res))
        return _wrap(_nn.log_softmax(_to_tensor(x), dim=-1))


class Mish(Module):
    """Applies the Mish function, element-wise."""

    def __init__(self) -> None:
        pass

    def __call__(self, x: array) -> array:
        from ml_switcheroo.core.config import config

        if config.eager_mode:
            data = np.array(_to_tensor(x).data)
            res = data * np.tanh(np.log1p(np.exp(data)))
            return _wrap(_to_tensor(res))
        return _wrap(_nn.mish(_to_tensor(x)))


class PReLU(Module):
    """Applies the element-wise parametric ReLU."""

    def __init__(self, num_parameters: int = 1, init: float = 0.25) -> None:
        self.weight = init

    def __call__(self, x: array) -> array:
        from ml_switcheroo.core.config import config

        if config.eager_mode:
            data = np.array(_to_tensor(x).data)
            res = np.maximum(0, data) + self.weight * np.minimum(0, data)
            return _wrap(_to_tensor(res))
        return relu(x)


class ReLU2(Module):
    """Applies the ReLU² activation function."""

    def __init__(self) -> None:
        pass

    def __call__(self, x: array) -> array:
        from ml_switcheroo.core.config import config

        if config.eager_mode:
            data = np.array(_to_tensor(x).data)
            res = np.maximum(data, 0) ** 2
            return _wrap(_to_tensor(res))
        return _wrap(_nn.relu(_to_tensor(x)))


class ReLU6(Module):
    """Applies the Rectified Linear Unit 6."""

    def __init__(self) -> None:
        pass

    def __call__(self, x: array) -> array:
        from ml_switcheroo.core.config import config

        if config.eager_mode:
            data = np.array(_to_tensor(x).data)
            res = np.minimum(np.maximum(data, 0), 6)
            return _wrap(_to_tensor(res))
        return _wrap(_nn.relu(_to_tensor(x)))


class SELU(Module):
    """Applies the Scaled Exponential Linear Unit."""

    def __init__(self) -> None:
        pass

    def __call__(self, x: array) -> array:
        from ml_switcheroo.core.config import config

        if config.eager_mode:
            data = np.array(_to_tensor(x).data)
            scale = 1.0507009873554804934193349852946
            alpha_val = 1.6732632423543772848170429916717
            res = scale * np.where(data > 0, data, alpha_val * (np.exp(data) - 1))
            return _wrap(_to_tensor(res))
        return _wrap(_nn.selu(_to_tensor(x)))


class HardShrink(Module):
    """Applies the HardShrink function."""

    def __init__(self, lambd: float = 0.5) -> None:
        self.lambd = lambd

    def __call__(self, x: array) -> array:
        from ml_switcheroo.core.config import config

        if config.eager_mode:
            data = np.array(_to_tensor(x).data)
            res = np.where(np.abs(data) > self.lambd, data, 0)
            return _wrap(_to_tensor(res))
        return _wrap(_nn.tanh(_to_tensor(x)))


class HardTanh(Module):
    """Applies the HardTanh function."""

    def __init__(self) -> None:
        pass

    def __call__(self, x: array) -> array:
        from ml_switcheroo.core.config import config

        if config.eager_mode:
            data = np.array(_to_tensor(x).data)
            return _wrap(_to_tensor(np.clip(data, -1.0, 1.0)))
        return tanh(x)


class Hardswish(Module):
    """Applies the hardswish function, element-wise."""

    def __init__(self) -> None:
        pass

    def __call__(self, x: array) -> array:
        from ml_switcheroo.core.config import config

        if config.eager_mode:
            data = np.array(_to_tensor(x).data)
            res = data * np.clip(data + 3, 0, 6) / 6
            return _wrap(_to_tensor(res))
        return _wrap(_nn.hardswish(_to_tensor(x)))


class LeakyReLU(Module):
    """Applies the Leaky Rectified Linear Unit."""

    def __init__(self, negative_slope: float = 0.01) -> None:
        self.negative_slope = negative_slope

    def __call__(self, x: array) -> array:
        return _wrap(_nn.leaky_relu(_to_tensor(x), self.negative_slope))


class Softshrink(Module):
    """Applies the Softshrink function."""

    def __init__(self, lambd: float = 0.5) -> None:
        self.lambd = lambd

    def __call__(self, x: array) -> array:
        from ml_switcheroo.core.config import config

        if config.eager_mode:
            data = np.array(_to_tensor(x).data)
            res = np.where(
                data > self.lambd,
                data - self.lambd,
                np.where(data < -self.lambd, data + self.lambd, 0),
            )
            return _wrap(_to_tensor(res))
        return _wrap(_nn.tanh(_to_tensor(x)))


class Softmin(Module):
    """Applies the Softmin function."""

    def __init__(self) -> None:
        pass

    def __call__(self, x: array) -> array:
        return softmax(array(-np.array(_to_tensor(x).data)))


class Softplus(Module):
    """Applies the Softplus function."""

    def __init__(self) -> None:
        pass

    def __call__(self, x: array) -> array:
        from ml_switcheroo.core.config import config

        if config.eager_mode:
            data = np.array(_to_tensor(x).data)
            res = np.log1p(np.exp(-np.abs(data))) + np.maximum(data, 0)
            return _wrap(_to_tensor(res))
        return _wrap(_nn.softplus(_to_tensor(x)))


class Softsign(Module):
    """Applies the Softsign function."""

    def __init__(self) -> None:
        pass

    def __call__(self, x: array) -> array:
        from ml_switcheroo.core.config import config

        if config.eager_mode:
            data = np.array(_to_tensor(x).data)
            res = data / (1.0 + np.abs(data))
            return _wrap(_to_tensor(res))
        return _wrap(_nn.softplus(_to_tensor(x)))


class Step(Module):
    """Applies the Step Activation Function."""

    def __init__(self, threshold: float = 0.0) -> None:
        self.threshold = threshold

    def __call__(self, x: array) -> array:
        from ml_switcheroo.core.config import config

        if config.eager_mode:
            data = np.array(_to_tensor(x).data)
            res = np.where(data >= self.threshold, 1.0, 0.0)
            return _wrap(_to_tensor(res))
        return _wrap(_nn.relu(_to_tensor(x)))
