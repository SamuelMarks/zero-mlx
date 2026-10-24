"""Optimizers and learning-rate schedules (`mlx.optimizers`) for zero_mlx."""

from __future__ import annotations

from typing import Callable, Dict, List, Optional, Tuple, Union

from zero_mlx.nn.base import Module
import zero_mlx.optimizers.math as math
import zero_mlx.optimizers.mx as mx
import zero_mlx.optimizers.optimizers as optimizers
from zero_mlx.optimizers.optimizers import (
    AdaDelta,
    Adafactor,
    Adagrad,
    Adam,
    AdamW,
    Adamax,
    Lion,
    MultiOptimizer,
    Muon,
    Optimizer,
    RMSprop,
    SGD,
    clip_grad_norm,
)
import zero_mlx.optimizers.schedulers as schedulers
from zero_mlx.optimizers.schedulers import (
    cosine_decay,
    exponential_decay,
    join_schedules,
    linear_schedule,
    step_decay,
)
from zero_mlx.utils import (
    tree_flatten,
    tree_map,
    tree_merge,
    tree_reduce,
    tree_unflatten,
)

__all__ = [
    "AdaDelta",
    "Adafactor",
    "Adagrad",
    "Adam",
    "AdamW",
    "Adamax",
    "Callable",
    "Dict",
    "Lion",
    "List",
    "Module",
    "MultiOptimizer",
    "Muon",
    "Optimizer",
    "Optional",
    "RMSprop",
    "SGD",
    "Tuple",
    "Union",
    "clip_grad_norm",
    "cosine_decay",
    "exponential_decay",
    "join_schedules",
    "linear_schedule",
    "math",
    "mx",
    "optimizers",
    "schedulers",
    "step_decay",
    "tree_flatten",
    "tree_map",
    "tree_merge",
    "tree_reduce",
    "tree_unflatten",
]
