"""Distributed communication submodule (`mlx.core.distributed` / `zero_mlx.core.distributed`)."""

from __future__ import annotations

from zero_mlx.distributed import (
    Group,
    all_gather,
    all_max,
    all_min,
    all_sum,
    init,
    is_available,
    recv,
    recv_like,
    send,
    sum_scatter,
)

__all__ = [
    "Group",
    "all_gather",
    "all_max",
    "all_min",
    "all_sum",
    "init",
    "is_available",
    "recv",
    "recv_like",
    "send",
    "sum_scatter",
]
