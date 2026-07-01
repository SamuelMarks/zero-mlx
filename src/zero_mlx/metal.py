"""Metal stub."""


def is_available() -> bool:  # pragma: no cover
    """Check if metal is available.

    Returns:
        bool: False, as Metal is not available in this stub.

    """
    return False


def get_active_memory() -> int:  # pragma: no cover
    """Get the active memory in use.

    Returns:
        int: The active memory in bytes, defaults to 0 in this stub.

    """
    return 0


def get_cache_memory() -> int:  # pragma: no cover
    """Get the current cache memory size.

    Returns:
        int: The cache memory in bytes, defaults to 0 in this stub.

    """
    return 0


def set_cache_limit(limit: int) -> int:  # pragma: no cover
    """Set the cache memory limit.

    Args:
        limit (int): The cache limit to set.

    Returns:
        int: The set cache limit.

    """
    return limit


def reset_peak_memory() -> None:  # pragma: no cover
    """Reset the recorded peak memory usage."""
    pass
