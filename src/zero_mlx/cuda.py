"""mlx.cuda stub."""


def is_available() -> bool:  # pragma: no cover
    """Check if CUDA is available.

    Returns:
        bool: False, as CUDA is not natively available in this stub.

    """
    return False


def device_count() -> int:  # pragma: no cover
    """Get the number of available CUDA devices.

    Returns:
        int: 0, as CUDA is not natively available in this stub.

    """
    return 0


def memory_info() -> dict:  # pragma: no cover
    """Get memory info for the current CUDA device.

    Returns:
        dict: Empty dictionary.

    """
    return {}


def clear_cache() -> None:  # pragma: no cover
    """Clear the CUDA memory cache."""
    pass
