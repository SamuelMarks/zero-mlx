"""DLPack capsule and `from_dlpack` exchange implementation for zero_mlx."""

from __future__ import annotations

from zero_mlx._types import (
    DLPackCapsule,
    DLPackConvertible,
    NestedSequence,
    Scalar,
    StreamOrDevice,
)
from zero_mlx.array import array
from zero_mlx.dtypes import DType, float32


def _get_dummy_dlpack_capsule(
    data: NestedSequence | Scalar = 0.0,
    dtype: DType = float32,
) -> DLPackCapsule:
    """Create a `DLPackCapsule` wrapping the provided data and dtype.

    Args:
        data: Array payload to wrap in the capsule.
        dtype: Data type of the wrapped array.

    Returns:
        A `DLPackCapsule` instance.

    """
    return DLPackCapsule(data=data, dtype=dtype)


def from_dlpack(
    x: array | DLPackConvertible | DLPackCapsule | Scalar | str,
    *,
    copy: bool | None = None,
    stream: StreamOrDevice = None,
) -> array:
    """Create a `zero_mlx.array` from an object implementing the DLPack protocol.

    Args:
        x: An `array`, `DLPackConvertible` object, or `DLPackCapsule`.
        copy: Optional boolean requesting a copy of the underlying buffer.
        stream: Optional stream or device on which to execute the import.

    Returns:
        A new `zero_mlx.array` wrapping the imported buffer.

    Raises:
        TypeError: If `x` does not support the DLPack protocol.

    """
    _ = (copy, stream)
    if isinstance(x, array):
        return array(x.tolist(), dtype=x.dtype)
    if isinstance(x, DLPackCapsule):
        return array(x.data, dtype=x.dtype)
    if isinstance(x, DLPackConvertible):
        capsule: DLPackCapsule = x.__dlpack__()
        return array(capsule.data, dtype=capsule.dtype)
    raise TypeError(f"Object {type(x)} does not support DLPack import")
