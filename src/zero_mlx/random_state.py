"""Global PRNG state wrapper (`mlx.core.random.state`) for zero_mlx."""

from __future__ import annotations

from zero_mlx.array import array
from zero_mlx.dtypes import uint32

_GLOBAL_SEED: int = 0
_GLOBAL_KEY: tuple[int, int] = (0, 0)


def set_global_seed(seed_val: int) -> None:
    """Update the global PRNG seed value.

    Args:
        seed_val: New integer seed value.

    """
    global _GLOBAL_SEED, _GLOBAL_KEY
    _GLOBAL_SEED = seed_val
    _GLOBAL_KEY = ((seed_val >> 32) & 0xFFFFFFFF, seed_val & 0xFFFFFFFF)


def state(new_state: array | None = None) -> array:
    """Get or set the global pseudo-random number generator key state.

    Args:
        new_state: Optional new PRNG key `array` to set as the global state.

    Returns:
        Current PRNG key `array` of shape `(2,)` and dtype `uint32`.

    """
    global _GLOBAL_SEED, _GLOBAL_KEY
    if new_state is not None:
        flat = new_state.astype(uint32).reshape(-1)
        k0 = int(flat[0].item().real) if flat.size > 0 else 0
        k1 = int(flat[1].item().real) if flat.size > 1 else 0
        _GLOBAL_KEY = (k0 & 0xFFFFFFFF, k1 & 0xFFFFFFFF)
        _GLOBAL_SEED = (_GLOBAL_KEY[0] << 32) | _GLOBAL_KEY[1]
    return array([_GLOBAL_KEY[0], _GLOBAL_KEY[1]], dtype=uint32)


def next_global_key() -> array:
    """Return the current global PRNG key and advance the global state deterministically.

    Returns:
        PRNG key `array` of shape `(2,)` and dtype `uint32`.

    """
    global _GLOBAL_SEED, _GLOBAL_KEY
    k0, k1 = _GLOBAL_KEY
    out_arr = array([k0, k1], dtype=uint32)
    next_k0 = ((k0 ^ 0x9E3779B9) * 0x85EBCA6B) & 0xFFFFFFFF
    next_k1 = ((k1 + 0x7F4A7C15) * 0xC2B2AE35) & 0xFFFFFFFF
    _GLOBAL_KEY = (next_k0, next_k1)
    _GLOBAL_SEED = (next_k0 << 32) | next_k1
    return out_arr
