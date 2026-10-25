"""Metal backend utilities submodule (`mlx.core.metal` / `zero_mlx.core.metal`)."""

from __future__ import annotations

from zero_mlx.metal import (
    clear_cache,
    device_info,
    get_active_memory,
    get_cache_memory,
    get_peak_memory,
    is_available,
    reset_peak_memory,
    set_cache_limit,
    set_memory_limit,
    set_wired_limit,
    start_capture,
    stop_capture,
)

__all__ = [
    "clear_cache",
    "device_info",
    "get_active_memory",
    "get_cache_memory",
    "get_peak_memory",
    "is_available",
    "reset_peak_memory",
    "set_cache_limit",
    "set_memory_limit",
    "set_wired_limit",
    "start_capture",
    "stop_capture",
]
