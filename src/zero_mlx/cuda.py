"""CUDA backend status and memory utilities for zero_mlx."""

from __future__ import annotations


def is_available() -> bool:
    """Check if the CUDA backend is available.

    Returns:
        False, as native CUDA hardware execution is not required in zero_mlx.

    """
    return False


def device_count() -> int:
    """Return the number of available CUDA devices.

    Returns:
        0 in pure-Python mode.

    """
    return 0


def memory_info() -> dict[str, int]:
    """Return memory statistics for the current CUDA device.

    Returns:
        Dictionary of CUDA memory counters.

    """
    return {}


def clear_cache() -> None:
    """Clear the CUDA memory cache."""
    return None
