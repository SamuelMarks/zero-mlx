"""Core `zero_mlx` alias (`mx`) for `mlx.optimizers`."""

from __future__ import annotations

from zero_mlx.array import array
from zero_mlx.ops import (
    abs,
    add,
    clip,
    divide,
    maximum,
    minimum,
    multiply,
    ones_like,
    rsqrt,
    sign,
    sqrt,
    square,
    subtract,
    sum,
    where,
    zeros_like,
)

__all__ = [
    "abs",
    "add",
    "array",
    "clip",
    "divide",
    "maximum",
    "minimum",
    "multiply",
    "ones_like",
    "rsqrt",
    "sign",
    "sqrt",
    "square",
    "subtract",
    "sum",
    "where",
    "zeros_like",
]
