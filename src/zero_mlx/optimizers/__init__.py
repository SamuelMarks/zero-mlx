"""mlx.optimizers module stub."""

import typing
from typing import Callable, List, Optional, Tuple, Union, Any, Dict

import zero_mlx.optimizers.math as math
import zero_mlx.optimizers.mx as mx
import zero_mlx.optimizers.optimizers as optimizers
import zero_mlx.optimizers.schedulers as schedulers

# Redefine standard types for MLX compatibility
Callable = Callable
List = List
Optional = Optional
Tuple = Tuple
Union = Union
math = math
mx = mx
optimizers = optimizers
schedulers = schedulers


class Optimizer:
    def __init__(self, schedulers=None):
        raise NotImplementedError("Optimizer not implemented")


class AdaDelta(Optimizer):
    def __init__(
        self,
        learning_rate: Union[float, Callable],
        rho: float = 0.9,
        eps: float = 1e-06,
    ):
        super().__init__()
        raise NotImplementedError("AdaDelta not implemented")


class Adafactor(Optimizer):
    def __init__(
        self,
        learning_rate: Union[float, Callable, None] = None,
        eps: Tuple[float, float] = (1e-30, 0.001),
        clip_threshold: float = 1.0,
        decay_rate: float = -0.8,
        beta_1: Any = None,
    ):
        super().__init__()
        raise NotImplementedError("Adafactor not implemented")


class Adagrad(Optimizer):
    def __init__(self, learning_rate: Union[float, Callable], eps: float = 1e-08):
        super().__init__()
        raise NotImplementedError("Adagrad not implemented")


class Adam(Optimizer):
    def __init__(
        self,
        learning_rate: Union[float, Callable],
        betas: List[float] = [0.9, 0.999],
        eps: float = 1e-08,
        bias_correction: bool = False,
    ):
        super().__init__()
        raise NotImplementedError("Adam not implemented")


class AdamW(Optimizer):
    def __init__(
        self,
        learning_rate: Union[float, Callable],
        betas: List[float] = [0.9, 0.999],
        eps: float = 1e-08,
        weight_decay: float = 0.01,
        bias_correction: bool = False,
    ):
        super().__init__()
        raise NotImplementedError("AdamW not implemented")


class Adamax(Optimizer):
    def __init__(
        self,
        learning_rate: Union[float, Callable],
        betas: List[float] = [0.9, 0.999],
        eps: float = 1e-08,
    ):
        super().__init__()
        raise NotImplementedError("Adamax not implemented")


class Lion(Optimizer):
    def __init__(
        self,
        learning_rate: Union[float, Callable],
        betas: List[float] = [0.9, 0.99],
        weight_decay: float = 0.0,
    ):
        super().__init__()
        raise NotImplementedError("Lion not implemented")


class MultiOptimizer(Optimizer):
    def __init__(self, optimizers, filters: list = []):
        super().__init__()
        raise NotImplementedError("MultiOptimizer not implemented")


class Muon(Optimizer):
    def __init__(
        self,
        learning_rate: Union[float, Callable],
        momentum: float = 0.95,
        weight_decay: float = 0.01,
        nesterov: bool = True,
        ns_steps: int = 5,
    ):
        super().__init__()
        raise NotImplementedError("Muon not implemented")


class RMSprop(Optimizer):
    def __init__(
        self,
        learning_rate: Union[float, Callable],
        alpha: float = 0.99,
        eps: float = 1e-08,
    ):
        super().__init__()
        raise NotImplementedError("RMSprop not implemented")


class SGD(Optimizer):
    def __init__(
        self,
        learning_rate: Union[float, Callable],
        momentum: float = 0.0,
        weight_decay: float = 0.0,
        dampening: float = 0.0,
        nesterov: bool = False,
    ):
        super().__init__()
        raise NotImplementedError("SGD not implemented")


def clip_grad_norm(grads, max_norm):
    raise NotImplementedError("clip_grad_norm not implemented")


def cosine_decay(init: float, decay_steps: int, end: float = 0.0) -> Callable:
    raise NotImplementedError("cosine_decay not implemented")


def exponential_decay(init: float, decay_rate: float) -> Callable:
    raise NotImplementedError("exponential_decay not implemented")


def join_schedules(schedules: List[Callable], boundaries: List[int]) -> Callable:
    raise NotImplementedError("join_schedules not implemented")


def linear_schedule(init: float, end: float, steps: int) -> Callable:
    raise NotImplementedError("linear_schedule not implemented")


def step_decay(init: float, decay_rate: float, step_size: int) -> Callable:
    raise NotImplementedError("step_decay not implemented")


def tree_flatten(
    tree: Any,
    prefix: str = "",
    is_leaf: Optional[Callable] = None,
    destination: Union[List[Tuple[str, Any]], Dict[str, Any], None] = None,
) -> Union[List[Tuple[str, Any]], Dict[str, Any]]:
    raise NotImplementedError("tree_flatten not implemented")


def tree_map(
    fn: Callable, tree: Any, *rest: Any, is_leaf: Optional[Callable] = None
) -> Any:
    raise NotImplementedError("tree_map not implemented")


def tree_merge(tree_a, tree_b, merge_fn=None):
    raise NotImplementedError("tree_merge not implemented")


def tree_reduce(fn, tree, initializer=None, is_leaf=None):
    raise NotImplementedError("tree_reduce not implemented")


def tree_unflatten(tree: Union[List[Tuple[str, Any]], Dict[str, Any]]) -> Any:
    raise NotImplementedError("tree_unflatten not implemented")


__all__ = [
    "math",
    "mx",
    "optimizers",
    "schedulers",
    "Optimizer",
    "AdaDelta",
    "Adafactor",
    "Adagrad",
    "Adam",
    "AdamW",
    "Adamax",
    "Lion",
    "MultiOptimizer",
    "Muon",
    "RMSprop",
    "SGD",
    "clip_grad_norm",
    "cosine_decay",
    "exponential_decay",
    "join_schedules",
    "linear_schedule",
    "step_decay",
    "tree_flatten",
    "tree_map",
    "tree_merge",
    "tree_reduce",
    "tree_unflatten",
    "Callable",
    "List",
    "Optional",
    "Tuple",
    "Union",
]
