"""Dtype info."""

from typing import Any
from zero_mlx.dtypes import DType
import numpy as np


class _Info:
    """Class representing type information."""

    def __init__(self, np_info: Any, dtype: DType):
        """Initialize the info object.

        Args:
            np_info: NumPy type info object.
            dtype: MLX data type.

        """
        self._np_info = np_info
        self.dtype = dtype

    @property
    def min(self) -> Any:
        """Get the minimum representable value.

        Returns:
            The minimum representable value.

        """
        return self._np_info.min

    @property
    def max(self) -> Any:
        """Get the maximum representable value.

        Returns:
            The maximum representable value.

        """
        return self._np_info.max

    @property
    def eps(self) -> Any:
        """Get the epsilon value.

        Returns:
            The epsilon value.

        """
        return getattr(self._np_info, "eps", None)


class _Bfloat16Info:
    """Class representing bfloat16 type information."""

    @property
    def min(self) -> float:
        return -3.389531389251535e38

    @property
    def max(self) -> float:
        return 3.389531389251535e38

    @property
    def eps(self) -> float:
        return 0.0078125


def finfo(dtype: DType) -> _Info:
    """Get float info."""
    if not dtype.value.startswith("float") and not dtype.value.startswith("bfloat"):
        raise ValueError("finfo only accepts float dtypes")

    np_val = dtype.value
    if np_val == "bfloat16":
        return _Info(_Bfloat16Info(), dtype)

    return _Info(np.finfo(np_val), dtype)


def iinfo(dtype: DType) -> _Info:
    """Get int info."""
    if not dtype.value.startswith("int") and not dtype.value.startswith("uint"):
        raise ValueError("iinfo only accepts int dtypes")
    return _Info(np.iinfo(dtype.value), dtype)
