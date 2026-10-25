"""CUDA backend utilities submodule (`mlx.core.cuda` / `zero_mlx.core.cuda`)."""

from __future__ import annotations

from zero_mlx.cuda import clear_cache, device_count, is_available, memory_info

__all__ = ["clear_cache", "device_count", "is_available", "memory_info"]
