"""Global PRNG state wrapper (`mlx.core.random.state`) for zero_mlx."""

from __future__ import annotations

from zero_mlx.array import array
from zero_mlx.dtypes import uint32

_GLOBAL_SEED: int = 0


def set_global_seed(seed_val: int) -> None:
    """Update the global PRNG seed value.

    Args:
        seed_val: New integer seed value.

    """
    global _GLOBAL_SEED
    _GLOBAL_SEED = seed_val


def state(new_state: array | None = None) -> array:
    """Get or set the global pseudo-random number generator key state.

    Args:
        new_state: Optional new PRNG key `array` to set as the global state.

    Returns:
        Current PRNG key `array` of shape `(2,)` and dtype `uint32`.

    """
    global _GLOBAL_SEED
    if new_state is not None:
        return new_state.astype(uint32)
    return array([0, _GLOBAL_SEED & 0xFFFFFFFF], dtype=uint32)
