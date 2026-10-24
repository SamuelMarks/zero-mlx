"""Array string formatting and printoptions management for zero_mlx."""

from __future__ import annotations

from contextlib import contextmanager
from typing import TYPE_CHECKING, Iterator, Sequence

from zero_mlx._types import NestedSequence, Scalar

if TYPE_CHECKING:
    from zero_mlx.array import array

_printoptions: dict[str, int | bool] = {
    "precision": 5,
    "threshold": 1000,
    "edgeitems": 3,
    "linewidth": 75,
    "suppress": False,
}

NL: str = chr(10)


def set_printoptions(
    precision: int | None = None,
    threshold: int | None = None,
    edgeitems: int | None = None,
    linewidth: int | None = None,
    suppress: bool | None = None,
) -> None:
    """Set printing options for `zero_mlx.array` string formatting.

    Args:
        precision: Number of digits of precision for floating point output.
        threshold: Total number of array elements which trigger summarization.
        edgeitems: Number of array items in summary at beginning and end of each dimension.
        linewidth: Number of characters per line for the purpose of inserting line breaks.
        suppress: Whether to suppress printing of small floating point values using scientific notation.

    """
    updates: dict[str, int | bool | None] = {
        "precision": precision,
        "threshold": threshold,
        "edgeitems": edgeitems,
        "linewidth": linewidth,
        "suppress": suppress,
    }
    for k, v in updates.items():
        if v is not None:
            _printoptions[k] = v


def get_printoptions() -> dict[str, int | bool]:
    """Return a copy of the current print options dictionary.

    Returns:
        Dictionary containing `precision`, `threshold`, `edgeitems`, `linewidth`, and `suppress`.

    """
    return dict(_printoptions)


@contextmanager
def printoptions(
    precision: int | None = None,
    threshold: int | None = None,
    edgeitems: int | None = None,
    linewidth: int | None = None,
    suppress: bool | None = None,
) -> Iterator[dict[str, int | bool]]:
    """Context manager for temporarily setting array print options.

    Args:
        precision: Number of digits of precision for floating point output.
        threshold: Total number of array elements which trigger summarization.
        edgeitems: Number of array items in summary at beginning and end of each dimension.
        linewidth: Number of characters per line.
        suppress: Whether to suppress scientific notation for small floats.

    Yields:
        Active print options dictionary within the context.

    """
    old_opts: dict[str, int | bool] = get_printoptions()
    set_printoptions(
        precision=precision,
        threshold=threshold,
        edgeitems=edgeitems,
        linewidth=linewidth,
        suppress=suppress,
    )
    try:
        yield get_printoptions()
    finally:
        _printoptions.update(old_opts)


def _format_scalar(val: Scalar, precision: int = 5) -> str:
    """Format a scalar value according to MLX representation rules.

    Args:
        val: Scalar value (`bool`, `int`, `float`, or `complex`).
        precision: Floating-point precision digits.

    Returns:
        Formatted string representation of the scalar.

    """
    if isinstance(val, bool):
        return "True" if val else "False"
    if isinstance(val, int):
        return str(val)
    if isinstance(val, float):
        if val != val:
            return "nan"
        if val == float("inf"):
            return "inf"
        if val == float("-inf"):
            return "-inf"
        if val == int(val):
            return str(int(val))
        formatted: str = f"{val:.{precision}f}"
        if precision == 5 and formatted.endswith("0"):
            return formatted.rstrip("0").rstrip(".")
        return formatted
    r_str: str = (
        str(int(val.real)) if val.real == int(val.real) else str(round(val.real, precision))
    )
    i_str: str = (
        str(int(val.imag)) if val.imag == int(val.imag) else str(round(val.imag, precision))
    )
    sign: str = "+" if val.imag >= 0 and not i_str.startswith("-") else ""
    return f"{r_str}{sign}{i_str}j"


def _format_recursive(
    data: NestedSequence | Scalar,
    current_shape: tuple[int, ...],
    depth: int,
    precision: int,
) -> str:
    """Recursively format nested sequence data given its shape.

    Args:
        data: Nested list or scalar element.
        current_shape: Remaining dimensions of the subarray.
        depth: Current recursion nesting depth.
        precision: Floating-point precision digits.

    Returns:
        Formatted bracketed string for the subarray.

    """
    if not current_shape or not isinstance(data, Sequence):
        scalar_val: Scalar = data if not isinstance(data, Sequence) else 0
        return _format_scalar(scalar_val, precision=precision)

    if current_shape[0] == 0:
        return "[]"

    elements: list[str] = []
    n_items: int = current_shape[0]
    if n_items > 6:
        for i in range(3):
            elements.append(_format_recursive(data[i], current_shape[1:], depth + 1, precision))
        elements.append("...")
        for i in range(n_items - 3, n_items):
            elements.append(_format_recursive(data[i], current_shape[1:], depth + 1, precision))
    else:
        for i in range(n_items):
            elements.append(_format_recursive(data[i], current_shape[1:], depth + 1, precision))

    if len(current_shape) == 1:
        return "[" + ", ".join(elements) + "]"
    indent: str = " " * (depth + 7)
    sep: str = "," + NL + indent
    return "[" + sep.join(elements) + "]"


def _format_list(
    data: NestedSequence | Scalar,
    shape: tuple[int, ...],
    threshold: int = 1000,
    precision: int = 5,
) -> str:
    """Format a nested list or scalar representing an array.

    Args:
        data: Nested sequence or scalar value.
        shape: Shape tuple of the array.
        threshold: Summarization threshold.
        precision: Floating-point precision digits.

    Returns:
        Formatted string representation of the array contents.

    """
    _ = threshold
    if not shape:
        scalar_val: Scalar = data if not isinstance(data, Sequence) else 0
        return _format_scalar(scalar_val, precision=precision)
    return _format_recursive(data, shape, 0, precision)


def format_array_repr(arr: array) -> str:
    """Format a `zero_mlx.array` into its canonical `array(..., dtype=...)` string.

    Args:
        arr: The `zero_mlx.array` instance to format.

    Returns:
        String representation matching MLX `array.__repr__`.

    """
    precision: int = int(_printoptions.get("precision", 5))
    threshold: int = int(_printoptions.get("threshold", 1000))
    dt_name: str = arr.dtype.value

    if arr.ndim == 0:
        val_str: str = _format_scalar(arr.item(), precision=precision)
        return f"array({val_str}, dtype={dt_name})"

    data: NestedSequence | Scalar = arr.tolist()
    arr_str: str = _format_list(data, arr.shape, threshold=threshold, precision=precision)
    return f"array({arr_str}, dtype={dt_name})"
