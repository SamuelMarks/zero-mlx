"""Evaluation and graph export compatibility wrappers for zero_mlx."""

from __future__ import annotations

from zero_mlx.export import export_to_dot
from zero_mlx.new_ops import async_eval, eval


def patch_ops() -> None:
    """No-op compatibility stub replacing legacy runtime monkey-patching."""
    return None


__all__ = ["async_eval", "eval", "export_to_dot", "patch_ops"]
