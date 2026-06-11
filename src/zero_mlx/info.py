"""Dtype info."""

from typing import Any
from zero_mlx.dtypes import DType
from ml_switcheroo.core.info import finfo as _ml_finfo, iinfo as _ml_iinfo
from ml_switcheroo.core.dtype import DType as SwitcherooDType


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


def finfo(dtype: DType) -> _Info:
    """Get float info."""
    if not dtype.value.startswith("float") and not dtype.value.startswith("bfloat"):
        raise ValueError("finfo only accepts float dtypes")

    np_val = dtype.value
    if np_val == "bfloat16":
        np_val = "bfloat16"  # fixed since we support it now  # pragma: no cover

    s_dtype = SwitcherooDType(np_val)
    return _Info(_ml_finfo(s_dtype), dtype)


def iinfo(dtype: DType) -> _Info:
    """Get int info."""
    if not dtype.value.startswith("int") and not dtype.value.startswith("uint"):
        raise ValueError("iinfo only accepts int dtypes")
    s_dtype = SwitcherooDType(dtype.value)
    return _Info(_ml_iinfo(s_dtype), dtype)
