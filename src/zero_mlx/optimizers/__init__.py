"""Optimizers for neural networks."""

from typing import Any, Callable, List, Optional, Tuple, Union

from .. import array


class Optimizer:
    """The base class for all optimizers."""

    def __init__(self, schedulers: Any = None) -> None:
        """Initialize the optimizer."""
        self.schedulers = schedulers


class AdaDelta(Optimizer):
    """The AdaDelta optimizer with a learning rate [1]."""

    def __init__(
        self,
        learning_rate: Union[float, Callable],
        rho: float = 0.9,
        eps: float = 1e-08,
    ) -> None:
        """Initialize the AdaDelta optimizer."""
        super().__init__()
        self.learning_rate = learning_rate
        self.rho = rho
        self.eps = eps


class Adafactor(Optimizer):
    """The Adafactor optimizer."""

    def __init__(
        self,
        learning_rate: Optional[Union[float, Callable]] = None,
        eps: Tuple[float, float] = (1e-30, 1e-3),
        clip_threshold: float = 1.0,
        decay_rate: float = -0.8,
        beta_1: Optional[float] = None,
        weight_decay: float = 0.0,
        scale_parameter: bool = True,
        relative_step: bool = True,
        warmup_init: bool = False,
    ) -> None:
        """Initialize the Adafactor optimizer."""
        super().__init__()
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
    """The Adagrad optimizer [1]."""

    def __init__(
        self, learning_rate: Union[float, Callable], eps: float = 1e-08
    ) -> None:
        """Initialize the Adagrad optimizer."""
        super().__init__()
        self.learning_rate = learning_rate
        self.eps = eps


class Adam(Optimizer):
    """The Adam optimizer [1]."""

    def __init__(
        self,
        learning_rate: Union[float, Callable],
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-08,
        bias_correction: bool = False,
    ) -> None:
        """Initialize the Adam optimizer."""
        super().__init__()
        self.learning_rate = learning_rate
        self.betas = betas
        self.eps = eps
        self.bias_correction = bias_correction


class AdamW(Optimizer):
    """The AdamW optimizer [1]."""

    def __init__(
        self,
        learning_rate: Union[float, Callable],
        betas: Tuple[float, float] = (0.9, 0.999),
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


class Adamax(Optimizer):
    """The Adamax optimizer, a variant of Adam based on the infinity norm [1]."""

    def __init__(
        self,
        learning_rate: Union[float, Callable],
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-08,
    ) -> None:
        """Initialize the Adamax optimizer."""
        super().__init__()
        self.learning_rate = learning_rate
        self.betas = betas
        self.eps = eps


class Lion(Optimizer):
    """The Lion optimizer [1]."""

    def __init__(
        self,
        learning_rate: Union[float, Callable],
        betas: Tuple[float, float] = (0.9, 0.99),
        weight_decay: float = 0.0,
    ) -> None:
        """Initialize the Lion optimizer."""
        super().__init__()
        self.learning_rate = learning_rate
        self.betas = betas
        self.weight_decay = weight_decay


class Module:
    """Base class for building neural networks with MLX."""

    def __init__(self) -> None:
        """Initialize the module."""
        pass

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """Forward pass of the module."""
        raise NotImplementedError


class MultiOptimizer(Optimizer):
    """Wraps a list of optimizers with corresponding weight predicates/filters."""

    def __init__(
        self,
        optimizers: List[Optimizer],
        filters: Optional[List[Callable[[str, array], bool]]] = None,
    ) -> None:
        """Initialize the MultiOptimizer."""
        super().__init__()
        self.optimizers = optimizers
        self.filters = filters if filters is not None else []


class Muon(Optimizer):
    """The Muon optimizer."""

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


class RMSprop(Optimizer):
    """The RMSprop optimizer [1]."""

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


class SGD(Optimizer):
    """The stochastic gradient descent optimizer."""

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
]
