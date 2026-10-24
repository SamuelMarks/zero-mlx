"""Optimizer classes and gradient clipping utilities (`mlx.optimizers.optimizers`) for zero_mlx."""

from __future__ import annotations

from typing import Callable, Sequence, Union

from ml_switcheroo_compiler.ops.optimizers.updates import (
    AdadeltaConfig,
    AdagradConfig,
    AdamHyperparams,
    AdamWHyperparams,
    AdamaxHyperparams,
    LionConfig,
    RMSPropHyperparams,
    SGDConfig,
    adadelta_update,
    adafactor_update,
    adagrad_update,
    adam_update,
    adamax_update,
    adamw_update,
    lion_update,
    muon_update,
    rmsprop_update,
    sgd_update,
)

from zero_mlx._types import PyTree
from zero_mlx.array import array
from zero_mlx.nn.base import Module
from zero_mlx.ops import sqrt, square, sum as mx_sum
from zero_mlx.optimizers.schedulers import ScheduleFn
from zero_mlx.utils import tree_flatten, tree_map

LearningRateOrSchedule = Union[float, ScheduleFn]
StateDict = dict[str, array]


def _resolve_lr(lr: LearningRateOrSchedule | None, step: int = 0) -> float:
    """Evaluate a float or schedule callable into a scalar learning rate.

    Args:
        lr: Float learning rate, schedule callable, or None.
        step: Current optimization step.

    Returns:
        Evaluated float learning rate.

    """
    if lr is None:
        return 0.01
    if callable(lr):
        return float(lr(step))
    return float(lr)


class Optimizer:
    """Base class for all `zero_mlx` first-order optimizers."""

    def __init__(self, schedulers: dict[str, ScheduleFn] | list[ScheduleFn] | None = None) -> None:
        """Initialize the base `Optimizer`.

        Args:
            schedulers: Optional dictionary or list of parameter schedule functions.

        """
        self.schedulers: dict[str, ScheduleFn] | list[ScheduleFn] | None = schedulers
        self.state: dict[str, StateDict] = {}
        self.step: int = 0

    def init(self, parameters: PyTree) -> None:
        """Initialize optimizer state for a parameter PyTree.

        Args:
            parameters: Parameter dictionary or PyTree.

        """
        flat = tree_flatten(parameters)
        for k, _ in flat if isinstance(flat, list) else []:
            self.state.setdefault(k, {})

    def apply_single(
        self, gradient: array, parameter: array, state: StateDict
    ) -> array | tuple[array, StateDict]:
        """Apply optimizer step to a single parameter and return the updated parameter.

        Args:
            gradient: Gradient `array` for the parameter.
            parameter: Current parameter `array`.
            state: Mutable state dictionary for the parameter.

        Returns:
            Updated parameter `array` or `(updated_parameter, updated_state)`.

        """
        _ = state
        return parameter - gradient * 0.01

    def apply_gradients(self, gradients: PyTree, parameters: PyTree) -> PyTree:
        """Apply gradients to a PyTree of `parameters` and return the updated PyTree.

        Args:
            gradients: PyTree of gradients matching `parameters`.
            parameters: PyTree of model parameters.

        Returns:
            Updated parameter PyTree.

        """
        from zero_mlx.utils import tree_map_with_path

        self.step += 1

        def _step_leaf(path: str, param: PyTree, grad: PyTree) -> PyTree:
            """Update a single `(param, grad)` leaf pair at `path` and persist its state.

            Args:
                path: Dotted parameter path.
                param: Parameter leaf.
                grad: Gradient leaf.

            Returns:
                Updated parameter leaf.

            """
            if isinstance(param, array) and isinstance(grad, array):
                param_state: StateDict = self.state.setdefault(path, {})
                res = self.apply_single(param, grad, param_state)
                if isinstance(res, tuple):
                    new_param, new_state = res
                    self.state[path] = new_state
                    return new_param
                return res
            return param

        return tree_map_with_path(_step_leaf, parameters, gradients)

    def update(self, model: Module, gradients: PyTree) -> None:
        """Apply `gradients` to `model.parameters()` and update `model` in-place.

        Args:
            model: Neural network `Module` to update.
            gradients: PyTree of parameter gradients.

        """
        updated = self.apply_gradients(gradients, model.parameters())
        model.update(updated if isinstance(updated, dict) else {})


class SGD(Optimizer):
    """Stochastic Gradient Descent (SGD) optimizer with optional momentum and weight decay."""

    def __init__(
        self,
        learning_rate: LearningRateOrSchedule,
        momentum: float = 0.0,
        weight_decay: float = 0.0,
        dampening: float = 0.0,
        nesterov: bool = False,
    ) -> None:
        """Initialize the `SGD` optimizer.

        Args:
            learning_rate: Learning rate or schedule callable.
            momentum: Momentum factor.
            weight_decay: Weight decay (L2 penalty).
            dampening: Dampening for momentum.
            nesterov: Whether to enable Nesterov momentum.

        """
        super().__init__()
        self.learning_rate: LearningRateOrSchedule = learning_rate
        self.momentum: float = momentum
        self.weight_decay: float = weight_decay
        self.dampening: float = dampening
        self.nesterov: bool = nesterov

    def apply_single(self, param: array, grad: array, state: StateDict) -> tuple[array, StateDict]:
        """Apply SGD update to a single parameter and its state.

        Args:
            param: Parameter `array`.
            grad: Gradient `array`.
            state: Optimizer state dictionary for this parameter.

        Returns:
            Tuple `(updated_param, updated_state)`.

        """
        state_t = {k: v._tensor for k, v in state.items()}
        lr_val = _resolve_lr(self.learning_rate, self.step)
        cfg = SGDConfig(
            lr=lr_val,
            momentum=self.momentum,
            dampening=self.dampening,
            weight_decay=self.weight_decay,
            nesterov=self.nesterov,
        )
        new_p, new_state = sgd_update(
            param._tensor,
            grad._tensor,
            cfg,
            state=state_t,
        )
        return array(new_p, dtype=param.dtype), {
            k: array(v, dtype=param.dtype) for k, v in new_state.items()
        }


class RMSprop(Optimizer):
    """RMSprop optimizer."""

    def __init__(
        self,
        learning_rate: LearningRateOrSchedule,
        alpha: float = 0.99,
        eps: float = 1e-8,
    ) -> None:
        """Initialize the `RMSprop` optimizer.

        Args:
            learning_rate: Learning rate or schedule.
            alpha: Smoothing constant.
            eps: Term added to the denominator to improve numerical stability.

        """
        super().__init__()
        self.learning_rate: LearningRateOrSchedule = learning_rate
        self.alpha: float = alpha
        self.eps: float = eps

    def apply_single(self, param: array, grad: array, state: StateDict) -> tuple[array, StateDict]:
        """Apply RMSprop update to a single parameter.

        Args:
            param: Parameter `array`.
            grad: Gradient `array`.
            state: Optimizer state dictionary.

        Returns:
            Tuple `(updated_param, updated_state)`.

        """
        state_t = {k: v._tensor for k, v in state.items()}
        lr_val = _resolve_lr(self.learning_rate, self.step)
        hp = RMSPropHyperparams(lr=lr_val, alpha=self.alpha, eps=self.eps)
        new_p, new_state = rmsprop_update(
            param._tensor,
            grad._tensor,
            hp,
            state=state_t,
        )
        return array(new_p, dtype=param.dtype), {
            k: array(v, dtype=param.dtype) for k, v in new_state.items()
        }


class Adagrad(Optimizer):
    """Adagrad optimizer."""

    def __init__(
        self,
        learning_rate: LearningRateOrSchedule,
        eps: float = 1e-8,
    ) -> None:
        """Initialize the `Adagrad` optimizer.

        Args:
            learning_rate: Learning rate or schedule.
            eps: Term added to the denominator to improve numerical stability.

        """
        super().__init__()
        self.learning_rate: LearningRateOrSchedule = learning_rate
        self.eps: float = eps

    def apply_single(self, param: array, grad: array, state: StateDict) -> tuple[array, StateDict]:
        """Apply Adagrad update to a single parameter.

        Args:
            param: Parameter `array`.
            grad: Gradient `array`.
            state: Optimizer state dictionary.

        Returns:
            Tuple `(updated_param, updated_state)`.

        """
        state_t = {k: v._tensor for k, v in state.items()}
        lr_val = _resolve_lr(self.learning_rate, self.step)
        cfg = AdagradConfig(lr=lr_val, eps=self.eps, step=max(1, self.step))
        new_p, new_state = adagrad_update(
            param._tensor,
            grad._tensor,
            cfg,
            state=state_t,
        )
        return array(new_p, dtype=param.dtype), {
            k: array(v, dtype=param.dtype) for k, v in new_state.items()
        }


class AdaDelta(Optimizer):
    """AdaDelta optimizer with adaptive learning rate."""

    def __init__(
        self,
        learning_rate: LearningRateOrSchedule = 1.0,
        rho: float = 0.9,
        eps: float = 1e-6,
    ) -> None:
        """Initialize the `AdaDelta` optimizer.

        Args:
            learning_rate: Learning rate or schedule.
            rho: Coefficient used for computing a running average of squared gradients.
            eps: Term added to the denominator to improve numerical stability.

        """
        super().__init__()
        self.learning_rate: LearningRateOrSchedule = learning_rate
        self.rho: float = rho
        self.eps: float = eps

    def apply_single(self, param: array, grad: array, state: StateDict) -> tuple[array, StateDict]:
        """Apply AdaDelta update to a single parameter.

        Args:
            param: Parameter `array`.
            grad: Gradient `array`.
            state: Optimizer state dictionary.

        Returns:
            Tuple `(updated_param, updated_state)`.

        """
        state_t = {k: v._tensor for k, v in state.items()}
        lr_val = _resolve_lr(self.learning_rate, self.step)
        cfg = AdadeltaConfig(lr=lr_val, rho=self.rho, eps=self.eps)
        new_p, new_state = adadelta_update(
            param._tensor,
            grad._tensor,
            cfg,
            state=state_t,
        )
        return array(new_p, dtype=param.dtype), {
            k: array(v, dtype=param.dtype) for k, v in new_state.items()
        }


class Adam(Optimizer):
    """Adam optimizer."""

    def __init__(
        self,
        learning_rate: LearningRateOrSchedule,
        betas: Sequence[float] = (0.9, 0.999),
        eps: float = 1e-8,
        bias_correction: bool = False,
    ) -> None:
        """Initialize the `Adam` optimizer.

        Args:
            learning_rate: Learning rate or schedule.
            betas: Coefficients `(beta1, beta2)` used for computing running averages of gradient and its square.
            eps: Term added to denominator for numerical stability.
            bias_correction: Whether to apply bias correction to first and second moment estimates.

        """
        super().__init__()
        self.learning_rate: LearningRateOrSchedule = learning_rate
        self.betas: list[float] = list(betas)
        self.eps: float = eps
        self.bias_correction: bool = bias_correction

    def apply_single(self, param: array, grad: array, state: StateDict) -> tuple[array, StateDict]:
        """Apply Adam update to a single parameter.

        Args:
            param: Parameter `array`.
            grad: Gradient `array`.
            state: Optimizer state dictionary.

        Returns:
            Tuple `(updated_param, updated_state)`.

        """
        state_t = {k: v._tensor for k, v in state.items()}
        lr_val = _resolve_lr(self.learning_rate, self.step)
        hp = AdamHyperparams(
            lr=lr_val,
            beta1=self.betas[0],
            beta2=self.betas[1],
            eps=self.eps,
            step=max(1, self.step),
        )
        new_p, new_state = adam_update(
            param._tensor,
            grad._tensor,
            hp,
            state=state_t,
        )
        return array(new_p, dtype=param.dtype), {
            k: array(v, dtype=param.dtype) for k, v in new_state.items()
        }


class AdamW(Optimizer):
    """AdamW optimizer with decoupled weight decay."""

    def __init__(
        self,
        learning_rate: LearningRateOrSchedule,
        betas: Sequence[float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.01,
        bias_correction: bool = False,
    ) -> None:
        """Initialize the `AdamW` optimizer.

        Args:
            learning_rate: Learning rate or schedule.
            betas: Coefficients `(beta1, beta2)` for running averages.
            eps: Term added to denominator for numerical stability.
            weight_decay: Decoupled weight decay coefficient.
            bias_correction: Whether to apply bias correction.

        """
        super().__init__()
        self.learning_rate: LearningRateOrSchedule = learning_rate
        self.betas: list[float] = list(betas)
        self.eps: float = eps
        self.weight_decay: float = weight_decay
        self.bias_correction: bool = bias_correction

    def apply_single(self, param: array, grad: array, state: StateDict) -> tuple[array, StateDict]:
        """Apply AdamW update to a single parameter.

        Args:
            param: Parameter `array`.
            grad: Gradient `array`.
            state: Optimizer state dictionary.

        Returns:
            Tuple `(updated_param, updated_state)`.

        """
        state_t = {k: v._tensor for k, v in state.items()}
        lr_val = _resolve_lr(self.learning_rate, self.step)
        hp = AdamWHyperparams(
            lr=lr_val,
            beta1=self.betas[0],
            beta2=self.betas[1],
            eps=self.eps,
            weight_decay=self.weight_decay,
            step=max(1, self.step),
        )
        new_p, new_state = adamw_update(
            param._tensor,
            grad._tensor,
            hp,
            state=state_t,
        )
        return array(new_p, dtype=param.dtype), {
            k: array(v, dtype=param.dtype) for k, v in new_state.items()
        }


class Adamax(Optimizer):
    """Adamax optimizer (variant of Adam based on infinity norm)."""

    def __init__(
        self,
        learning_rate: LearningRateOrSchedule,
        betas: Sequence[float] = (0.9, 0.999),
        eps: float = 1e-8,
    ) -> None:
        """Initialize the `Adamax` optimizer.

        Args:
            learning_rate: Learning rate or schedule.
            betas: Coefficients `(beta1, beta2)`.
            eps: Numerical stability constant.

        """
        super().__init__()
        self.learning_rate: LearningRateOrSchedule = learning_rate
        self.betas: list[float] = list(betas)
        self.eps: float = eps

    def apply_single(self, param: array, grad: array, state: StateDict) -> tuple[array, StateDict]:
        """Apply Adamax update to a single parameter.

        Args:
            param: Parameter `array`.
            grad: Gradient `array`.
            state: Optimizer state dictionary.

        Returns:
            Tuple `(updated_param, updated_state)`.

        """
        state_t = {k: v._tensor for k, v in state.items()}
        lr_val = _resolve_lr(self.learning_rate, self.step)
        hp = AdamaxHyperparams(
            lr=lr_val,
            beta1=self.betas[0],
            beta2=self.betas[1],
            eps=self.eps,
            step=max(1, self.step),
        )
        new_p, new_state = adamax_update(
            param._tensor,
            grad._tensor,
            hp,
            state=state_t,
        )
        return array(new_p, dtype=param.dtype), {
            k: array(v, dtype=param.dtype) for k, v in new_state.items()
        }


class Lion(Optimizer):
    """Lion (EvoLved Sign Momentum) optimizer."""

    def __init__(
        self,
        learning_rate: LearningRateOrSchedule,
        betas: Sequence[float] = (0.9, 0.99),
        weight_decay: float = 0.0,
    ) -> None:
        """Initialize the `Lion` optimizer.

        Args:
            learning_rate: Learning rate or schedule.
            betas: Coefficients `(beta1, beta2)` for update interpolation and momentum decay.
            weight_decay: Decoupled weight decay factor.

        """
        super().__init__()
        self.learning_rate: LearningRateOrSchedule = learning_rate
        self.betas: list[float] = list(betas)
        self.weight_decay: float = weight_decay

    def apply_single(self, param: array, grad: array, state: StateDict) -> tuple[array, StateDict]:
        """Apply Lion update to a single parameter.

        Args:
            param: Parameter `array`.
            grad: Gradient `array`.
            state: Optimizer state dictionary.

        Returns:
            Tuple `(updated_param, updated_state)`.

        """
        state_t = {k: v._tensor for k, v in state.items()}
        lr_val = _resolve_lr(self.learning_rate, self.step)
        cfg = LionConfig(
            lr=lr_val,
            beta1=self.betas[0],
            beta2=self.betas[1],
            weight_decay=self.weight_decay,
        )
        new_p, new_state = lion_update(
            param._tensor,
            grad._tensor,
            cfg,
            state=state_t,
        )
        return array(new_p, dtype=param.dtype), {
            k: array(v, dtype=param.dtype) for k, v in new_state.items()
        }


class Adafactor(Optimizer):
    """Adafactor optimizer with factored second-moment estimation."""

    def __init__(
        self,
        learning_rate: LearningRateOrSchedule | None = None,
        eps: tuple[float, float] = (1e-30, 1e-3),
        clip_threshold: float = 1.0,
        decay_rate: float = -0.8,
        beta_1: float | None = None,
        weight_decay: float = 0.0,
        scale_parameter: bool = True,
        relative_step: bool = True,
        warmup_init: bool = False,
    ) -> None:
        """Initialize the `Adafactor` optimizer.

        Args:
            learning_rate: Optional learning rate or schedule.
            eps: Regularization constants `(eps1, eps2)`.
            clip_threshold: Threshold for root-mean-square gradient clipping.
            decay_rate: Coefficient used to compute running averages of square gradient.
            beta_1: Optional first moment coefficient.
            weight_decay: Weight decay coefficient.
            scale_parameter: If True, learning rate is scaled by root-mean-square of parameter.
            relative_step: If True, time-dependent learning rate is computed instead of external `learning_rate`.
            warmup_init: Time-dependent learning rate computation depends on whether warm-up initialization is being used.

        """
        super().__init__()
        self.learning_rate: LearningRateOrSchedule | None = learning_rate
        self.eps: tuple[float, float] = eps
        self.clip_threshold: float = clip_threshold
        self.decay_rate: float = decay_rate
        self.beta_1: float | None = beta_1
        self.weight_decay: float = weight_decay
        self.scale_parameter: bool = scale_parameter
        self.relative_step: bool = relative_step
        self.warmup_init: bool = warmup_init

    def apply_single(self, param: array, grad: array, state: StateDict) -> tuple[array, StateDict]:
        """Apply Adafactor update to a single parameter.

        Args:
            param: Parameter `array`.
            grad: Gradient `array`.
            state: Optimizer state dictionary.

        Returns:
            Tuple `(updated_param, updated_state)`.

        """
        state_t = {k: v._tensor for k, v in state.items()}
        lr_val = _resolve_lr(self.learning_rate, self.step)
        new_p, new_state = adafactor_update(
            param._tensor,
            grad._tensor,
            lr=lr_val,
            state=state_t,
        )
        return array(new_p, dtype=param.dtype), {
            k: array(v, dtype=param.dtype) for k, v in new_state.items()
        }


class Muon(Optimizer):
    """Muon (MomentUm Orthogonalized by Newton-schulz) optimizer."""

    def __init__(
        self,
        learning_rate: LearningRateOrSchedule,
        momentum: float = 0.95,
        weight_decay: float = 0.01,
        nesterov: bool = True,
        ns_steps: int = 5,
    ) -> None:
        """Initialize the `Muon` optimizer.

        Args:
            learning_rate: Learning rate or schedule.
            momentum: Momentum decay factor.
            weight_decay: Weight decay coefficient.
            nesterov: Whether to use Nesterov momentum.
            ns_steps: Number of Newton-Schulz iteration steps.

        """
        super().__init__()
        self.learning_rate: LearningRateOrSchedule = learning_rate
        self.momentum: float = momentum
        self.weight_decay: float = weight_decay
        self.nesterov: bool = nesterov
        self.ns_steps: int = ns_steps

    def apply_single(self, param: array, grad: array, state: StateDict) -> tuple[array, StateDict]:
        """Apply Muon update to a single parameter.

        Args:
            param: Parameter `array`.
            grad: Gradient `array`.
            state: Optimizer state dictionary.

        Returns:
            Tuple `(updated_param, updated_state)`.

        """
        state_t = {k: v._tensor for k, v in state.items()}
        lr_val = _resolve_lr(self.learning_rate, self.step)
        new_p, new_state = muon_update(
            param._tensor,
            grad._tensor,
            lr=lr_val,
            momentum=self.momentum,
            state=state_t,
        )
        return array(new_p, dtype=param.dtype), {
            k: array(v, dtype=param.dtype) for k, v in new_state.items()
        }


FilterFn = Callable[[str, PyTree], bool]


class MultiOptimizer(Optimizer):
    """Wraps a list of optimizers with corresponding weight predicates/filters."""

    def __init__(
        self,
        optimizers: Sequence[Optimizer],
        filters: Sequence[FilterFn] = (),
    ) -> None:
        """Initialize `MultiOptimizer`.

        Args:
            optimizers: Sequence of `Optimizer` instances.
            filters: Sequence of filter predicates `(path, grad) -> bool` of length `len(optimizers) - 1`.

        """
        super().__init__()
        self.optimizers: list[Optimizer] = list(optimizers)
        self.filters: list[FilterFn] = list(filters)

    def _select_optimizer(self, path: str, leaf: PyTree) -> Optimizer:
        """Select the sub-optimizer matching `(path, leaf)` via `self.filters`.

        Args:
            path: Dotted parameter key path.
            leaf: Parameter or gradient leaf.

        Returns:
            Selected `Optimizer` instance.

        """
        for filt, opt in zip(self.filters, self.optimizers):
            if filt(path, leaf):
                return opt
        return self.optimizers[-1]

    def init(self, parameters: PyTree) -> None:
        """Initialize state across all sub-optimizers.

        Args:
            parameters: Parameter PyTree.

        """
        for opt in self.optimizers:
            opt.init(parameters)

    def apply_gradients(self, gradients: PyTree, parameters: PyTree) -> PyTree:
        """Route each `(param, grad)` leaf to its matching sub-optimizer and return updated parameters.

        Args:
            gradients: Gradient PyTree.
            parameters: Parameter PyTree.

        Returns:
            Updated parameter PyTree.

        """
        from zero_mlx.utils import tree_map_with_path

        self.step += 1
        for opt in self.optimizers:
            opt.step = self.step

        def _route_leaf(path: str, param: PyTree, grad: PyTree) -> PyTree:
            """Update a single `(param, grad)` leaf using the filter-selected sub-optimizer.

            Args:
                path: Dotted parameter path.
                param: Parameter leaf.
                grad: Gradient leaf.

            Returns:
                Updated parameter leaf.

            """
            if isinstance(param, array) and isinstance(grad, array):
                sub_opt = self._select_optimizer(path, grad)
                param_state = sub_opt.state.setdefault(path, {})
                res = sub_opt.apply_single(param, grad, param_state)
                if isinstance(res, tuple):
                    new_p, new_s = res
                    sub_opt.state[path] = new_s
                    return new_p
                return res
            return param

        return tree_map_with_path(_route_leaf, parameters, gradients)


def clip_grad_norm(grads: PyTree, max_norm: float) -> tuple[PyTree, array]:
    """Clip the global L2 norm of a PyTree of gradients `grads` to at most `max_norm`.

    Args:
        grads: PyTree containing gradient `array` leaves.
        max_norm: Maximum permitted global L2 norm.

    Returns:
        Tuple `(clipped_grads, total_norm)`.

    """
    flat = tree_flatten(grads)
    sq_sum: array = array(0.0)
    for _, g in flat if isinstance(flat, list) else []:
        if isinstance(g, array):
            sq_sum = sq_sum + mx_sum(square(g))
    total_norm: array = sqrt(sq_sum)
    norm_val: float = float(total_norm.item().real)
    scale: float = max_norm / (norm_val + 1e-6) if norm_val > max_norm else 1.0

    def _scale_leaf(g: PyTree) -> PyTree:
        """Scale a gradient leaf by `scale`.

        Args:
            g: Gradient leaf.

        Returns:
            Scaled gradient leaf.

        """
        if isinstance(g, array) and scale < 1.0:
            return g * scale
        return g

    return tree_map(_scale_leaf, grads), total_norm


__all__ = [
    "AdaDelta",
    "Adafactor",
    "Adagrad",
    "Adam",
    "AdamW",
    "Adamax",
    "FilterFn",
    "LearningRateOrSchedule",
    "Lion",
    "Module",
    "MultiOptimizer",
    "Muon",
    "Optimizer",
    "RMSprop",
    "SGD",
    "StateDict",
    "clip_grad_norm",
]
