"""Numeric limits and precision metadata (`finfo` and `iinfo`) for MLX data types."""

from __future__ import annotations

from zero_mlx.dtypes import DType, DTypeInput, resolve_dtype


class FInfo:
    """Machine limits for floating-point data types."""

    def __init__(self, min_val: float, max_val: float, eps_val: float, dtype: DType) -> None:
        """Initialize floating-point type limits.

        Args:
            min_val: Smallest representable number.
            max_val: Largest representable number.
            eps_val: Machine epsilon value.
            dtype: Associated MLX floating-point `DType`.

        """
        self._min: float = min_val
        self._max: float = max_val
        self._eps: float = eps_val
        self.dtype: DType = dtype
        self.bits: int = dtype.size * 8

    @property
    def min(self) -> float:
        """Return the minimum representable floating-point value.

        Returns:
            Minimum representable float.

        """
        return self._min

    @property
    def max(self) -> float:
        """Return the maximum representable floating-point value.

        Returns:
            Maximum representable float.

        """
        return self._max

    @property
    def eps(self) -> float:
        """Return the machine epsilon for this floating-point type.

        Returns:
            Floating-point epsilon value.

        """
        return self._eps


class IInfo:
    """Machine limits for integer data types."""

    def __init__(self, min_val: int, max_val: int, dtype: DType) -> None:
        """Initialize integer type limits.

        Args:
            min_val: Minimum representable integer value.
            max_val: Maximum representable integer value.
            dtype: Associated MLX integer `DType`.

        """
        self._min: int = min_val
        self._max: int = max_val
        self.dtype: DType = dtype
        self.bits: int = dtype.size * 8

    @property
    def min(self) -> int:
        """Return the minimum representable integer value.

        Returns:
            Minimum representable integer.

        """
        return self._min

    @property
    def max(self) -> int:
        """Return the maximum representable integer value.

        Returns:
            Maximum representable integer.

        """
        return self._max


def finfo(dtype: DTypeInput) -> FInfo:
    """Return floating-point machine limits for an MLX floating-point `dtype`.

    Args:
        dtype: Floating-point `DType` or convertible representation.

    Returns:
        `FInfo` instance describing `min`, `max`, `eps`, and `dtype`.

    Raises:
        ValueError: If `dtype` is not a floating-point data type.

    """
    dt: DType = resolve_dtype(dtype)
    if not dt.value.startswith("float") and not dt.value.startswith("bfloat"):
        raise ValueError("finfo only accepts float dtypes")

    if dt == DType.float32:
        return FInfo(
            -3.4028234663852886e38,
            3.4028234663852886e38,
            1.1920928955078125e-07,
            dt,
        )
    if dt == DType.float16:
        return FInfo(-65504.0, 65504.0, 0.0009765625, dt)
    if dt == DType.bfloat16:
        return FInfo(-3.389531389251535e38, 3.389531389251535e38, 0.0078125, dt)
    return FInfo(
        -1.7976931348623157e308,
        1.7976931348623157e308,
        2.220446049250313e-16,
        dt,
    )


def iinfo(dtype: DTypeInput) -> IInfo:
    """Return integer machine limits for an MLX integer `dtype`.

    Args:
        dtype: Integer `DType` or convertible representation.

    Returns:
        `IInfo` instance describing `min`, `max`, and `dtype`.

    Raises:
        ValueError: If `dtype` is not an integer data type.

    """
    dt: DType = resolve_dtype(dtype)
    if not dt.value.startswith("int") and not dt.value.startswith("uint"):
        raise ValueError("iinfo only accepts int dtypes")

    limits: dict[DType, tuple[int, int]] = {
        DType.int8: (-128, 127),
        DType.uint8: (0, 255),
        DType.int16: (-32768, 32767),
        DType.uint16: (0, 65535),
        DType.int32: (-2147483648, 2147483647),
        DType.uint32: (0, 4294967295),
        DType.int64: (-9223372036854775808, 9223372036854775807),
        DType.uint64: (0, 18446744073709551615),
    }
    min_val, max_val = limits[dt]
    return IInfo(min_val, max_val, dt)
