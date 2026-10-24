"""Metal backend status, capture, and memory utilities for zero_mlx."""

from __future__ import annotations

from zero_mlx.device import (
    clear_cache as _clear_cache,
    device_info as _device_info,
    get_active_memory as _get_active_memory,
    get_cache_memory as _get_cache_memory,
    get_peak_memory as _get_peak_memory,
    reset_peak_memory as _reset_peak_memory,
    set_cache_limit as _set_cache_limit,
    set_memory_limit as _set_memory_limit,
    set_wired_limit as _set_wired_limit,
)


def is_available() -> bool:
    """Check if the Metal backend is available.

    Returns:
        False in pure-Python zero-dependency mode.

    """
    return False


def get_active_memory() -> int:
    """Return the active Metal memory in bytes.

    Returns:
        Active memory in bytes.

    """
    return _get_active_memory()


def get_peak_memory() -> int:
    """Return the peak Metal memory in bytes.

    Returns:
        Peak memory in bytes.

    """
    return _get_peak_memory()


def reset_peak_memory() -> None:
    """Reset the recorded peak Metal memory usage to zero."""
    _reset_peak_memory()


def get_cache_memory() -> int:
    """Return the current Metal cache memory size in bytes.

    Returns:
        Cache memory in bytes.

    """
    return _get_cache_memory()


def set_cache_limit(limit: int) -> int:
    """Set the Metal cache memory limit in bytes.

    Args:
        limit: Cache limit in bytes.

    Returns:
        Previous cache limit in bytes.

    """
    return _set_cache_limit(limit)


def set_memory_limit(limit: int) -> int:
    """Set the Metal memory limit in bytes.

    Args:
        limit: Memory limit in bytes.

    Returns:
        Previous memory limit in bytes.

    """
    return _set_memory_limit(limit)


def set_wired_limit(limit: int) -> int:
    """Set the Metal wired memory limit in bytes.

    Args:
        limit: Wired memory limit in bytes.

    Returns:
        Previous wired memory limit in bytes.

    """
    return _set_wired_limit(limit)


def clear_cache() -> None:
    """Clear the Metal memory cache."""
    _clear_cache()


def device_info() -> dict[str, str | int]:
    """Return information about the Metal device.

    Returns:
        Dictionary with Metal device metadata.

    """
    return _device_info()


def start_capture(path: str) -> None:
    """Start a Metal trace capture to the specified file path.

    Args:
        path: Output `.gputrace` path.

    """
    _ = path


def stop_capture() -> None:
    """Stop an active Metal trace capture."""
    return None
