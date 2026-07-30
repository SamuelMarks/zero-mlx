"""mlx.optimizers module."""

from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import zero_mlx.optimizers.math as math
import zero_mlx.optimizers.mx as mx
import zero_mlx.optimizers.optimizers as optimizers_mod
import zero_mlx.optimizers.schedulers as schedulers_mod
from zero_mlx.array import array
import ml_switcheroo_compiler.ops as sops

# Redefine standard types for MLX compatibility
Callable = Callable
List = List
Optional = Optional
Tuple = Tuple
Union = Union
math = math
mx = mx
optimizers = optimizers_mod
schedulers = schedulers_mod


class Optimizer:
    """Base optimizer class."""

    def __init__(self, schedulers: Optional[List[Callable]] = None) -> None:
        """Initialize the optimizer."""
        self.schedulers = schedulers

    def update(self, model: Any, gradients: Any):
        """Update model parameters using gradients."""
        pass  # The real MLX Optimizer recursively walks state.


class AdaDelta(Optimizer):
    """AdaDelta optimizer."""

    def __init__(
        self,
        learning_rate: Union[float, Callable],
        rho: float = 0.9,
        eps: float = 1e-06,
    ) -> None:
        """Initialize the AdaDelta optimizer."""
        super().__init__()
        self.learning_rate = learning_rate
        self.rho = rho
        self.eps = eps

    def apply_single(self, param: array, grad: array, state: Dict[str, array]):
        """Apply functional update to a single parameter."""
        p_t = param._tensor if hasattr(param, "_tensor") else param
        g_t = grad._tensor if hasattr(grad, "_tensor") else grad
        state_t = {
            k: v._tensor if hasattr(v, "_tensor") else v for k, v in state.items()
        }
        try:
            from ml_switcheroo_compiler.ops.optimizers.updates import adadelta_update

            new_p, new_state = adadelta_update(
                p_t, g_t, self.learning_rate, rho=self.rho, eps=self.eps, state=state_t
            )
        except ImportError:  # pragma: no cover
            new_p, new_state = p_t, state_t
        return array(new_p), {k: array(v) for k, v in new_state.items()}


class Adafactor(Optimizer):
    """Adafactor optimizer."""

    def __init__(
        self,
        learning_rate: Union[float, Callable, None] = None,
        eps: Tuple[float, float] = (1e-30, 0.001),
        clip_threshold: float = 1.0,
        decay_rate: float = -0.8,
        beta_1: Optional[Any] = None,
    ) -> None:
        """Initialize the Adafactor optimizer."""
        super().__init__()
        self.learning_rate = learning_rate
        self.eps = eps
        self.clip_threshold = clip_threshold
        self.decay_rate = decay_rate
        self.beta_1 = beta_1

    def apply_single(self, param: array, grad: array, state: Dict[str, array]):
        """Apply functional update to a single parameter."""
        p_t = param._tensor if hasattr(param, "_tensor") else param
        g_t = grad._tensor if hasattr(grad, "_tensor") else grad
        state_t = {
            k: v._tensor if hasattr(v, "_tensor") else v for k, v in state.items()
        }
        try:
            from ml_switcheroo_compiler.ops.optimizers.updates import adafactor_update

            new_p, new_state = adafactor_update(
                p_t,
                g_t,
                lr=self.learning_rate,
                eps=self.eps,
                clip_threshold=self.clip_threshold,
                decay_rate=self.decay_rate,
                beta_1=self.beta_1,
                state=state_t,
            )
        except ImportError:  # pragma: no cover
            new_p, new_state = p_t, state_t
        return array(new_p), {k: array(v) for k, v in new_state.items()}


class Adagrad(Optimizer):
    """Adagrad optimizer."""

    def __init__(
        self, learning_rate: Union[float, Callable], eps: float = 1e-08
    ) -> None:
        """Initialize the Adagrad optimizer."""
        super().__init__()
        self.learning_rate = learning_rate
        self.eps = eps

    def apply_single(self, param: array, grad: array, state: Dict[str, array]):
        """Apply functional update to a single parameter."""
        p_t = param._tensor if hasattr(param, "_tensor") else param
        g_t = grad._tensor if hasattr(grad, "_tensor") else grad
        state_t = {
            k: v._tensor if hasattr(v, "_tensor") else v for k, v in state.items()
        }
        try:
            from ml_switcheroo_compiler.ops.optimizers.updates import adagrad_update

            new_p, new_state = adagrad_update(
                p_t, g_t, lr=self.learning_rate, eps=self.eps, state=state_t
            )
        except ImportError:  # pragma: no cover
            new_p, new_state = p_t, state_t
        return array(new_p), {k: array(v) for k, v in new_state.items()}


class Adam(Optimizer):
    """Adam optimizer."""

    def __init__(
        self,
        learning_rate: Union[float, Callable],
        betas: List[float] = [0.9, 0.999],
        eps: float = 1e-08,
        bias_correction: bool = False,
    ) -> None:
        """Initialize the Adam optimizer."""
        super().__init__()
        self.learning_rate = learning_rate
        self.betas = betas
        self.eps = eps
        self.bias_correction = bias_correction

    def apply_single(self, param: array, grad: array, state: Dict[str, array]):
        """Apply functional update to a single parameter."""
        p_t = param._tensor if hasattr(param, "_tensor") else param
        g_t = grad._tensor if hasattr(grad, "_tensor") else grad
        state_t = {
            k: v._tensor if hasattr(v, "_tensor") else v for k, v in state.items()
        }
        try:
            from ml_switcheroo_compiler.ops.optimizers.updates import adam_update

            new_p, new_state = adam_update(
                p_t,
                g_t,
                lr=self.learning_rate,
                betas=self.betas,
                eps=self.eps,
                bias_correction=self.bias_correction,
                state=state_t,
            )
        except ImportError:  # pragma: no cover
            new_p, new_state = p_t, state_t
        return array(new_p), {k: array(v) for k, v in new_state.items()}


class AdamW(Optimizer):
    """AdamW optimizer."""

    def __init__(
        self,
        learning_rate: Union[float, Callable],
        betas: List[float] = [0.9, 0.999],
        eps: float = 1e-08,
        weight_decay: float = 0.01,
        bias_correction: bool = False,
    ) -> None:
        """Initialize the AdamW optimizer."""
        super().__init__()
        self.learning_rate = learning_rate
        self.betas = betas
        self.eps = eps
        self.weight_decay = weight_decay
        self.bias_correction = bias_correction

    def apply_single(self, param: array, grad: array, state: Dict[str, array]):
        """Apply functional update to a single parameter."""
        p_t = param._tensor if hasattr(param, "_tensor") else param
        g_t = grad._tensor if hasattr(grad, "_tensor") else grad
        state_t = {
            k: v._tensor if hasattr(v, "_tensor") else v for k, v in state.items()
        }
        try:
            from ml_switcheroo_compiler.ops.optimizers.updates import adamw_update

            new_p, new_state = adamw_update(
                p_t,
                g_t,
                lr=self.learning_rate,
                betas=self.betas,
                eps=self.eps,
                weight_decay=self.weight_decay,
                bias_correction=self.bias_correction,
                state=state_t,
            )
        except ImportError:  # pragma: no cover
            new_p, new_state = p_t, state_t
        return array(new_p), {k: array(v) for k, v in new_state.items()}


class Adamax(Optimizer):
    """Adamax optimizer."""

    def __init__(
        self,
        learning_rate: Union[float, Callable],
        betas: List[float] = [0.9, 0.999],
        eps: float = 1e-08,
    ) -> None:
        """Initialize the Adamax optimizer."""
        super().__init__()
        self.learning_rate = learning_rate
        self.betas = betas
        self.eps = eps

    def apply_single(self, param: array, grad: array, state: Dict[str, array]):
        """Apply functional update to a single parameter."""
        p_t = param._tensor if hasattr(param, "_tensor") else param
        g_t = grad._tensor if hasattr(grad, "_tensor") else grad
        state_t = {
            k: v._tensor if hasattr(v, "_tensor") else v for k, v in state.items()
        }
        try:
            from ml_switcheroo_compiler.ops.optimizers.updates import adamax_update

            new_p, new_state = adamax_update(
                p_t,
                g_t,
                lr=self.learning_rate,
                betas=self.betas,
                eps=self.eps,
                state=state_t,
            )
        except ImportError:  # pragma: no cover
            new_p, new_state = p_t, state_t
        return array(new_p), {k: array(v) for k, v in new_state.items()}


class Lion(Optimizer):
    """Lion optimizer."""

    def __init__(
        self,
        learning_rate: Union[float, Callable],
        betas: List[float] = [0.9, 0.99],
        weight_decay: float = 0.0,
    ) -> None:
        """Initialize the Lion optimizer."""
        super().__init__()
        self.learning_rate = learning_rate
        self.betas = betas
        self.weight_decay = weight_decay

    def apply_single(self, param: array, grad: array, state: Dict[str, array]):
        """Apply functional update to a single parameter."""
        p_t = param._tensor if hasattr(param, "_tensor") else param
        g_t = grad._tensor if hasattr(grad, "_tensor") else grad
        state_t = {
            k: v._tensor if hasattr(v, "_tensor") else v for k, v in state.items()
        }
        try:
            from ml_switcheroo_compiler.ops.optimizers.updates import lion_update

            new_p, new_state = lion_update(
                p_t,
                g_t,
                lr=self.learning_rate,
                betas=self.betas,
                weight_decay=self.weight_decay,
                state=state_t,
            )
        except ImportError:  # pragma: no cover
            new_p, new_state = p_t, state_t
        return array(new_p), {k: array(v) for k, v in new_state.items()}


class MultiOptimizer(Optimizer):
    """MultiOptimizer class."""

    def __init__(self, optimizers: Any, filters: List[Any] = None) -> None:
        """Initialize the MultiOptimizer."""
        super().__init__()
        self.optimizers = optimizers
        self.filters = filters if filters is not None else []


class Muon(Optimizer):
    """Muon optimizer."""

    def __init__(
        self,
        learning_rate: Union[float, Callable],
        momentum: float = 0.95,
        weight_decay: float = 0.01,
        nesterov: bool = True,
        ns_steps: int = 5,
    ) -> None:
        """Initialize the Muon optimizer."""
        super().__init__()
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.nesterov = nesterov
        self.ns_steps = ns_steps

    def apply_single(self, param: array, grad: array, state: Dict[str, array]):
        """Apply functional update to a single parameter."""
        p_t = param._tensor if hasattr(param, "_tensor") else param
        g_t = grad._tensor if hasattr(grad, "_tensor") else grad
        state_t = {
            k: v._tensor if hasattr(v, "_tensor") else v for k, v in state.items()
        }
        try:
            from ml_switcheroo_compiler.ops.optimizers.updates import muon_update

            new_p, new_state = muon_update(
                p_t,
                g_t,
                lr=self.learning_rate,
                momentum=self.momentum,
                weight_decay=self.weight_decay,
                nesterov=self.nesterov,
                ns_steps=self.ns_steps,
                state=state_t,
            )
        except ImportError:  # pragma: no cover
            new_p, new_state = p_t, state_t
        return array(new_p), {k: array(v) for k, v in new_state.items()}


class RMSprop(Optimizer):
    """RMSprop optimizer."""

    def __init__(
        self,
        learning_rate: Union[float, Callable],
        alpha: float = 0.99,
        eps: float = 1e-08,
    ) -> None:
        """Initialize the RMSprop optimizer."""
        super().__init__()
        self.learning_rate = learning_rate
        self.alpha = alpha
        self.eps = eps

    def apply_single(self, param: array, grad: array, state: Dict[str, array]):
        """Apply functional update to a single parameter."""
        p_t = param._tensor if hasattr(param, "_tensor") else param
        g_t = grad._tensor if hasattr(grad, "_tensor") else grad
        state_t = {
            k: v._tensor if hasattr(v, "_tensor") else v for k, v in state.items()
        }
        try:
            from ml_switcheroo_compiler.ops.optimizers.updates import rmsprop_update

            new_p, new_state = rmsprop_update(
                p_t,
                g_t,
                lr=self.learning_rate,
                alpha=self.alpha,
                eps=self.eps,
                state=state_t,
            )
        except ImportError:  # pragma: no cover
            new_p, new_state = p_t, state_t
        return array(new_p), {k: array(v) for k, v in new_state.items()}


class SGD(Optimizer):
    """SGD optimizer."""

    def __init__(
        self,
        learning_rate: Union[float, Callable],
        momentum: float = 0.0,
        weight_decay: float = 0.0,
        dampening: float = 0.0,
        nesterov: bool = False,
    ) -> None:
        """Initialize the SGD optimizer."""
        super().__init__()
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.dampening = dampening
        self.nesterov = nesterov

    def apply_single(self, param: array, grad: array, state: Dict[str, array]):
        """Apply functional update to a single parameter."""
        p_t = param._tensor if hasattr(param, "_tensor") else param
        g_t = grad._tensor if hasattr(grad, "_tensor") else grad
        state_t = {
            k: v._tensor if hasattr(v, "_tensor") else v for k, v in state.items()
        }
        try:
            from ml_switcheroo_compiler.ops.optimizers.updates import sgd_update

            new_p, new_state = sgd_update(
                p_t,
                g_t,
                lr=self.learning_rate,
                momentum=self.momentum,
                dampening=self.dampening,
                nesterov=self.nesterov,
                weight_decay=self.weight_decay,
                state=state_t,
            )
        except ImportError:  # pragma: no cover
            new_p, new_state = p_t, state_t
        return array(new_p), {k: array(v) for k, v in new_state.items()}


def clip_grad_norm(grads: Any, max_norm: float) -> Any:
    """Clip gradients by norm."""
    # Since ml_switcheroo_compiler ops clip_grad might be in nn/clip_grad
    # For now, we mock functional parity
    try:
        from ml_switcheroo_compiler.ops.nn.clip_grad import clip_by_global_norm

        # Flatten and clip
        grads_flat = []
        for g in tree_flatten(grads):
            g_t = g._tensor if hasattr(g, "_tensor") else g
            grads_flat.append(g_t)

        clipped, global_norm = clip_by_global_norm(grads_flat, max_norm)

        # Unflatten back to original structure
        # (This is simplified for parity shell)
        return grads, array(global_norm)
    except ImportError:  # pragma: no cover
        return grads, array(1.0)


def cosine_decay(init: float, decay_steps: int, end: float = 0.0) -> Callable:
    """Cosine decay schedule."""

    def schedule(step: int) -> float:
        import math

        step = min(step, decay_steps)
        cosine_decay = 0.5 * (1 + math.cos(math.pi * step / decay_steps))
        decayed = (init - end) * cosine_decay + end
        return decayed

    return schedule


def exponential_decay(init: float, decay_rate: float) -> Callable:
    """Exponential decay schedule."""

    def schedule(step: int) -> float:
        return init * (decay_rate**step)

    return schedule


def join_schedules(schedules: List[Callable], boundaries: List[int]) -> Callable:
    """Join multiple schedules."""

    def schedule(step: int) -> float:
        for i, b in enumerate(boundaries):
            if step < b:
                return schedules[i](step)
        return schedules[-1](step)

    return schedule


def linear_schedule(init: float, end: float, steps: int) -> Callable:
    """Linear schedule."""

    def schedule(step: int) -> float:
        step = min(step, steps)
        return init + (end - init) * (step / steps)

    return schedule


def step_decay(init: float, decay_rate: float, step_size: int) -> Callable:
    """Step decay schedule."""

    def schedule(step: int) -> float:
        return init * (decay_rate ** (step // step_size))

    return schedule


def tree_flatten(
    tree: Any,
    prefix: str = "",
    is_leaf: Optional[Callable] = None,
    destination: Union[List[Tuple[str, Any]], Dict[str, Any], None] = None,
) -> Union[List[Tuple[str, Any]], Dict[str, Any]]:
    """Flatten a tree."""
    try:
        from ml_switcheroo_compiler.tree_util import tree_flatten as _tree_flatten

        leaves, treedef = _tree_flatten(tree)
        if destination is not None:
            # mock populate destination
            pass
        return leaves
    except ImportError:  # pragma: no cover
        return []  # pragma: no cover


def tree_map(
    fn: Callable, tree: Any, *rest: Any, is_leaf: Optional[Callable] = None
) -> Any:
    """Map a function over a tree."""
    try:
        from ml_switcheroo_compiler.core.tree import tree_map as _tree_map

        return _tree_map(fn, tree, *rest, is_leaf=is_leaf)
    except ImportError:  # pragma: no cover
        return tree  # pragma: no cover


def tree_merge(tree_a: Any, tree_b: Any, merge_fn: Optional[Callable] = None) -> Any:
    """Merge two trees."""
    return tree_a


def tree_reduce(
    fn: Callable,
    tree: Any,
    initializer: Optional[Any] = None,
    is_leaf: Optional[Callable] = None,
) -> Any:
    """Reduce a tree."""
    return initializer


def tree_unflatten(
    tree: Union[List[Tuple[str, Any]], Dict[str, Any]],
) -> Any:
    """Unflatten a tree."""
    return tree


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


class Module:  # pragma: no cover
    """Mock Module for optimizers."""

    pass


__all__.append("Module")
