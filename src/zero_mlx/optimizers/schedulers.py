"""Learning rate scheduler functions (`mlx.optimizers.schedulers`) for zero_mlx."""

from __future__ import annotations

import math
from typing import Callable, Sequence

ScheduleFn = Callable[[int], float]


def cosine_decay(init: float, decay_steps: int, end: float = 0.0) -> ScheduleFn:
    """Create a cosine decay learning rate schedule.

    Args:
        init: Initial learning rate value.
        decay_steps: Number of steps over which to decay.
        end: Final minimum learning rate value.

    Returns:
        Callable mapping `step: int` to the scheduled learning rate `float`.

    """

    def _schedule(step: int) -> float:
        """Compute learning rate at `step`.

        Args:
            step: Current optimization step.

        Returns:
            Learning rate at `step`.

        """
        s = min(step, decay_steps)
        cos_factor = 0.5 * (1.0 + math.cos(math.pi * s / max(1, decay_steps)))
        return (init - end) * cos_factor + end

    return _schedule


def exponential_decay(init: float, decay_rate: float) -> ScheduleFn:
    """Create an exponential decay learning rate schedule (`init * decay_rate**step`).

    Args:
        init: Initial learning rate value.
        decay_rate: Multiplicative decay factor per step.

    Returns:
        Callable mapping `step: int` to the scheduled learning rate `float`.

    """

    def _schedule(step: int) -> float:
        """Compute learning rate at `step`.

        Args:
            step: Current optimization step.

        Returns:
            Learning rate at `step`.

        """
        return float(init * (decay_rate**step))

    return _schedule


def step_decay(init: float, decay_rate: float, step_size: int) -> ScheduleFn:
    """Create a step-wise exponential decay learning rate schedule.

    Args:
        init: Initial learning rate value.
        decay_rate: Multiplicative decay factor applied every `step_size` steps.
        step_size: Number of steps between decay updates.

    Returns:
        Callable mapping `step: int` to the scheduled learning rate `float`.

    """

    def _schedule(step: int) -> float:
        """Compute learning rate at `step`.

        Args:
            step: Current optimization step.

        Returns:
            Learning rate at `step`.

        """
        return float(init * (decay_rate ** (step // max(1, step_size))))

    return _schedule


def linear_schedule(init: float, end: float, steps: int) -> ScheduleFn:
    """Create a linear interpolation learning rate schedule from `init` to `end` over `steps`.

    Args:
        init: Initial learning rate value.
        end: Final learning rate value after `steps`.
        steps: Total number of interpolation steps.

    Returns:
        Callable mapping `step: int` to the scheduled learning rate `float`.

    """

    def _schedule(step: int) -> float:
        """Compute learning rate at `step`.

        Args:
            step: Current optimization step.

        Returns:
            Learning rate at `step`.

        """
        s = min(step, steps)
        return float(init + (end - init) * (s / max(1, steps)))

    return _schedule


def join_schedules(schedules: Sequence[ScheduleFn], boundaries: Sequence[int]) -> ScheduleFn:
    """Join multiple learning rate schedules across step `boundaries`.

    Args:
        schedules: Sequence of `N` schedule callables.
        boundaries: Sequence of `N - 1` increasing step boundary thresholds.

    Returns:
        Composite schedule callable.

    """

    def _schedule(step: int) -> float:
        """Compute learning rate at `step`.

        Args:
            step: Current optimization step.

        Returns:
            Learning rate at `step`.

        """
        for i, boundary in enumerate(boundaries):
            if step < boundary:
                offset = 0 if i == 0 else boundaries[i - 1]
                return schedules[i](step - offset)
        last_offset = boundaries[-1] if boundaries else 0
        return schedules[-1](step - last_offset)

    return _schedule


__all__ = [
    "ScheduleFn",
    "cosine_decay",
    "exponential_decay",
    "join_schedules",
    "linear_schedule",
    "step_decay",
]
