"""Numpy array representation injection."""

from typing import Any, Tuple, Optional


def _format_list(  # pragma: no cover
    data: Any, shape: Tuple[int, ...], threshold: int = 1000, precision: int = 5
) -> str:
    """Format a nested list representing an array."""
    if not shape:
        return str(round(data, precision)) if isinstance(data, float) else str(data)

    def _get_size(shp: Tuple[int, ...]) -> int:  # pragma: no cover
        s = 1
        for x in shp:
            s *= x
        return s

    total_size = _get_size(shape)

    def _format_recursive(  # pragma: no cover
        data: Any, current_shape: Tuple[int, ...], depth: int
    ) -> str:  # pragma: no cover
        if not current_shape:
            if isinstance(data, float):
                val = round(data, precision)
                if data == int(data):
                    return str(int(data))

                return (
                    f"{data:.{precision}f}".rstrip("0").rstrip(".")
                    if f"{data:.{precision}f}".endswith("0") and precision == 5
                    else f"{data:.{precision}f}"
                )
            if isinstance(data, complex):
                r = (
                    str(int(data.real))
                    if data.real == int(data.real)
                    else str(round(data.real, precision))
                )
                i = (
                    str(int(data.imag))
                    if data.imag == int(data.imag)
                    else str(round(data.imag, precision))
                )
                sign = "+" if data.imag >= 0 and not i.startswith("-") else ""
                return f"{r}{sign}{i}j"
            return str(data)

        if current_shape[0] == 0:
            return "[]"

        elements = []
        is_large = current_shape[0] > 6

        if is_large:
            head_len = 3
            tail_len = 3
            for i in range(head_len):
                elements.append(
                    _format_recursive(data[i], current_shape[1:], depth + 1)
                )
            elements.append("...")
            for i in range(current_shape[0] - tail_len, current_shape[0]):
                elements.append(
                    _format_recursive(data[i], current_shape[1:], depth + 1)
                )
        else:
            for i in range(current_shape[0]):
                elements.append(
                    _format_recursive(data[i], current_shape[1:], depth + 1)
                )

        indent = " " * (depth + 7)  # "array([" is 7 chars
        if len(current_shape) == 1:
            return "[" + ", ".join(elements) + "]"
        else:
            sep = ",\n" + indent
            return "[" + sep.join(elements) + "]"

    return _format_recursive(data, shape, 0)


def inject_repr(cls: Any) -> None:  # pragma: no cover
    """Inject __repr__ to mlx array class."""

    def __repr__(self: Any) -> str:  # pragma: no cover
        """Get string representation of the array."""
        import zero_mlx

        # Fallback to default precision if not set
        precision = (
            getattr(zero_mlx, "_printoptions_precision", 5)
            if hasattr(zero_mlx, "_printoptions_precision")
            else 5
        )

        dt_name = self.dtype.name
        if dt_name == "bool_":
            dt_name = "bool"

        # Fast path for eagerly evaluated / scalar tensors
        if getattr(self, "ndim", 0) == 0:
            val = self.item()
            if isinstance(val, float):
                if val == int(val):
                    val_str = str(int(val))
                else:
                    val_str = (
                        f"{val:.{precision}f}".rstrip("0").rstrip(".")
                        if precision == 5
                        else f"{val:.{precision}f}"
                    )
            elif isinstance(val, complex):
                r = (
                    str(int(val.real))
                    if val.real == int(val.real)
                    else str(round(val.real, precision))
                )
                i = (
                    str(int(val.imag))
                    if val.imag == int(val.imag)
                    else str(round(val.imag, precision))
                )
                sign = "+" if val.imag >= 0 and not i.startswith("-") else ""
                val_str = f"{r}{sign}{i}j"
            else:
                val_str = str(val)
            return f"array({val_str}, dtype={dt_name})"

        # Formatting
        try:
            data = self.tolist()
        except Exception:  # pragma: no cover
            data = None  # pragma: no cover

        if data is None:  # pragma: no cover
            arr_str = "[]"  # pragma: no cover
        else:
            try:
                arr_str = _format_list(
                    data, tuple(self.shape), threshold=1000, precision=precision
                )
            except Exception:  # pragma: no cover
                arr_str = str(data)  # pragma: no cover

        return f"array({arr_str}, dtype={dt_name})"

    cls.__repr__ = __repr__
    cls.__str__ = __repr__
