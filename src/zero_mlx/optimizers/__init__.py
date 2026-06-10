"""Optimizers for neural networks."""

import ml_switcheroo
from typing import Any, Callable, List, Optional, Tuple, Union
from .. import array


class Optimizer:
    def __init__(self, schedulers: Any = None, **kwargs) -> None:
        self.schedulers = schedulers
        self.state = {}
        for k, v in kwargs.items():
            setattr(self, k, v)

    def apply_gradients(self, grads: Any, model: Any) -> None:
        pass

    def update(self, model: Any, grads: Any) -> None:
        pass


class AdaDelta(Optimizer):
    def __init__(
        self,
        learning_rate: Union[float, Callable],
        rho: float = 0.9,
        eps: float = 1e-08,
    ) -> None:
        self.learning_rate = learning_rate
        self.rho = rho
        self.eps = eps


class Adafactor(Optimizer):
    def __init__(
        self,
        learning_rate: Optional[Union[float, Callable]] = None,
        eps: Tuple[float, float] = (1e-30, 0.001),
        clip_threshold: float = 1.0,
        decay_rate: float = -0.8,
        beta_1: Optional[float] = None,
        weight_decay: float = 0.0,
        scale_parameter: bool = True,
        relative_step: bool = True,
        warmup_init: bool = False,
    ) -> None:
        self.learning_rate = learning_rate
        self.eps = eps
        self.clip_threshold = clip_threshold
        self.decay_rate = decay_rate
        self.beta_1 = beta_1
        self.weight_decay = weight_decay
        self.scale_parameter = scale_parameter
        self.relative_step = relative_step
        self.warmup_init = warmup_init


class Adagrad(Optimizer):
    def __init__(
        self, learning_rate: Union[float, Callable], eps: float = 1e-08
    ) -> None:
        self.learning_rate = learning_rate
        self.eps = eps


class Adam(Optimizer):
    def __init__(
        self,
        learning_rate: Union[float, Callable],
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-08,
        bias_correction: bool = False,
    ) -> None:
        self.learning_rate = learning_rate
        self.betas = betas
        self.eps = eps
        self.bias_correction = bias_correction


class AdamW(Optimizer):
    def __init__(
        self,
        learning_rate: Union[float, Callable],
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-08,
        weight_decay: float = 0.01,
        bias_correction: bool = False,
    ) -> None:
        self.learning_rate = learning_rate
        self.betas = betas
        self.eps = eps
        self.weight_decay = weight_decay
        self.bias_correction = bias_correction


class Adamax(Optimizer):
    def __init__(
        self,
        learning_rate: Union[float, Callable],
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-08,
    ) -> None:
        self.learning_rate = learning_rate
        self.betas = betas
        self.eps = eps


class Lion(Optimizer):
    def __init__(
        self,
        learning_rate: Union[float, Callable],
        betas: Tuple[float, float] = (0.9, 0.99),
        weight_decay: float = 0.0,
    ) -> None:
        self.learning_rate = learning_rate
        self.betas = betas
        self.weight_decay = weight_decay


class Module:
    def __init__(self) -> None:
        pass

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError


class MultiOptimizer(Optimizer):
    def __init__(
        self,
        optimizers: List[Optimizer],
        filters: Optional[List[Callable[[str, array], bool]]] = None,
    ) -> None:
        self.optimizers = optimizers
        self.filters = filters if filters is not None else []


class Muon(Optimizer):
    def __init__(
        self,
        learning_rate: Union[float, Callable],
        momentum: float = 0.95,
        weight_decay: float = 0.01,
        nesterov: bool = True,
        ns_steps: int = 5,
    ) -> None:
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.nesterov = nesterov
        self.ns_steps = ns_steps


class RMSprop(Optimizer):
    def __init__(
        self,
        learning_rate: Union[float, Callable],
        alpha: float = 0.99,
        eps: float = 1e-08,
    ) -> None:
        self.learning_rate = learning_rate
        self.alpha = alpha
        self.eps = eps


class SGD(Optimizer):
    def __init__(
        self,
        learning_rate: Union[float, Callable],
        momentum: float = 0.0,
        weight_decay: float = 0.0,
        dampening: float = 0.0,
        nesterov: bool = False,
    ) -> None:
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.dampening = dampening
        self.nesterov = nesterov


def step_decay(
    init: float, decay_rate: float, step_size: int
) -> Callable[[int], float]:
    def schedule(step: int) -> float:
        return init * (decay_rate ** (step // step_size))

    return schedule


def exponential_decay(init: float, decay_rate: float) -> Callable[[int], float]:
    def schedule(step: int) -> float:
        return init * (decay_rate**step)

    return schedule


def cosine_decay(init: float, decay_steps: int) -> Callable[[int], float]:
    import math

    def schedule(step: int) -> float:
        step = min(step, decay_steps)
        cosine_decay = 0.5 * (1 + math.cos(math.pi * step / decay_steps))
        return init * cosine_decay

    return schedule


def linear_decay(init: float, end: float, steps: int) -> Callable[[int], float]:
    def schedule(step: int) -> float:
        step = min(step, steps)
        return init + (end - init) * (step / steps)

    return schedule


__all__ = [
    "AdaDelta",
    "Adafactor",
    "Adagrad",
    "Adam",
    "AdamW",
    "Adamax",
    "Lion",
    "Module",
    "MultiOptimizer",
    "Muon",
    "Optimizer",
    "RMSprop",
    "SGD",
    "step_decay",
    "exponential_decay",
    "cosine_decay",
    "linear_decay",
]
