"""Dtype info."""

from typing import Any
from zero_mlx.dtypes import DType


class _Info:  # pragma: no cover
    """Class representing type information."""

    def __init__(  # pragma: no cover
        self, min_val: Any, max_val: Any, eps_val: Any, dtype: DType
    ):  # pragma: no cover
        """Initialize the info object.

        Args:
            min_val: Minimum representable value.
            max_val: Maximum representable value.
            eps_val: Epsilon value.
            dtype: MLX data type.

        """
        self._min = min_val
        self._max = max_val
        self._eps = eps_val
        self.dtype = dtype

    @property
    def min(self) -> Any:  # pragma: no cover
        """Get the minimum representable value.

        Returns:
            The minimum representable value.

        """
        return self._min

    @property
    def max(self) -> Any:  # pragma: no cover
        """Get the maximum representable value.

        Returns:
            The maximum representable value.

        """
        return self._max

    @property
    def eps(self) -> Any:  # pragma: no cover
        """Get the epsilon value.

        Returns:
            The epsilon value.

        """
        return self._eps


def finfo(dtype: DType) -> _Info:  # pragma: no cover
    """Get float info."""
    if not dtype.value.startswith("float") and not dtype.value.startswith("bfloat"):
        raise ValueError("finfo only accepts float dtypes")

    if dtype.value == "float32":
        return _Info(
            -3.4028234663852886e38, 3.4028234663852886e38, 1.1920928955078125e-07, dtype
        )
    if dtype.value == "float16":
        return _Info(-65504.0, 65504.0, 0.0009765625, dtype)
    if dtype.value == "bfloat16":
        return _Info(-3.389531389251535e38, 3.389531389251535e38, 0.0078125, dtype)
    return _Info(
        -1.7976931348623157e308, 1.7976931348623157e308, 2.220446049250313e-16, dtype
    )


def iinfo(dtype: DType) -> _Info:  # pragma: no cover
    """Get int info."""
    if not dtype.value.startswith("int") and not dtype.value.startswith("uint"):
        raise ValueError("iinfo only accepts int dtypes")

    if dtype.value == "int8":
        return _Info(-128, 127, None, dtype)
    if dtype.value == "uint8":
        return _Info(0, 255, None, dtype)
    if dtype.value == "int16":
        return _Info(-32768, 32767, None, dtype)
    if dtype.value == "uint16":
        return _Info(0, 65535, None, dtype)
    if dtype.value == "int32":
        return _Info(-2147483648, 2147483647, None, dtype)
    if dtype.value == "uint32":
        return _Info(0, 4294967295, None, dtype)
    if dtype.value == "int64":
        return _Info(-9223372036854775808, 9223372036854775807, None, dtype)
    if dtype.value == "uint64":
        return _Info(0, 18446744073709551615, None, dtype)

    return _Info(-2147483648, 2147483647, None, dtype)
