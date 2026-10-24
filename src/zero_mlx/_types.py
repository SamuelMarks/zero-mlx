"""Strict shared type definitions and protocols for zero_mlx."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal, Mapping, Protocol, Sequence, Union, runtime_checkable

from zero_mlx.device import Device, DeviceType, Stream, StreamOrDeviceInput
from zero_mlx.dtypes import DType, DTypeInput, DtypeCategory

if TYPE_CHECKING:
    from builtins import ellipsis
    from zero_mlx.array import array

Scalar = Union[int, float, bool, complex]
NestedSequence = Sequence[Union[Scalar, "NestedSequence"]]
ShapeLike = Union[int, Sequence[int], None]
AxesLike = Union[int, Sequence[int], None]
DTypeLike = DTypeInput
StreamOrDevice = StreamOrDeviceInput
IntSlice = slice


@runtime_checkable
class MlxArrayConvertible(Protocol):
    """Protocol for objects providing `__mlx_array__` conversion."""

    def __mlx_array__(self) -> array:
        """Convert this object into a `zero_mlx.array`.

        Returns:
            The converted `zero_mlx.array` instance.

        """


@runtime_checkable
class DLPackConvertible(Protocol):
    """Protocol for objects supporting the DLPack exchange interface."""

    def __dlpack__(self, stream: int | None = None) -> DLPackCapsule:
        """Export a DLPack capsule representing the underlying buffer.

        Args:
            stream: Optional stream index for synchronization.

        Returns:
            A `DLPackCapsule` wrapping the array data.

        """

    def __dlpack_device__(self) -> tuple[int, int]:
        """Return the `(device_type, device_id)` tuple for DLPack.

        Returns:
            Tuple of `(device_type_id, device_index)`.

        """


class DLPackCapsule:
    """Pure-Python DLPack capsule representation for zero-dependency array exchange."""

    def __init__(
        self,
        data: NestedSequence | Scalar,
        dtype: DType,
        device_info: tuple[int, int] = (1, 0),
    ) -> None:
        """Initialize a `DLPackCapsule` holding array data and dtype metadata.

        Args:
            data: Nested list or scalar data payload.
            dtype: MLX `DType` of the buffer.
            device_info: Tuple of `(dlpack_device_type, device_id)`.

        """
        self.data: NestedSequence | Scalar = data
        self.dtype: DType = dtype
        self.device_info: tuple[int, int] = device_info


class ExternalDTypeProtocol(Protocol):
    """Protocol for external data-type descriptors (such as `numpy.dtype`) exposing `.name`."""

    @property
    def name(self) -> str:
        """Return the string name of the data type."""


@runtime_checkable
class ArrayProtocol(Protocol):
    """Protocol for external array-like objects (such as `numpy.ndarray`) exposing `.tolist()` and `.dtype`."""

    @property
    def dtype(self) -> str | DType | ExternalDTypeProtocol:
        """Return the array data type."""

    def tolist(self) -> Scalar | list[Scalar | NestedSequence]:
        """Convert the array into a Python scalar or nested list."""


ArrayInput = Union[
    "array",
    Scalar,
    NestedSequence,
    MlxArrayConvertible,
    DLPackConvertible,
    ArrayProtocol,
]

IndexElement = Union[
    int, "slice[int | None, int | None, int | None]", "array", list[int], "ellipsis", None
]
IndexLike = Union[IndexElement, tuple[IndexElement, ...]]

PyTree = Union[
    "array",
    Scalar,
    str,
    list["PyTree"],
    tuple["PyTree", ...],
    Sequence["PyTree"],
    dict[str, "PyTree"],
    Mapping[str, "PyTree"],
    None,
]

ParameterTree = dict[str, Union["ParameterTree", list[Union["ParameterTree", "array"]], "array"]]

LossReduction = str
PadMode = Literal["constant", "edge", "reflect", "symmetric", "wrap"]
QuantizationMode = Literal["affine", "mxfp4", "nvfp4", "mxfp8"]
ConvPadding = Union[int, tuple[int, ...], str]

__all__ = [
    "ArrayInput",
    "AxesLike",
    "ConvPadding",
    "DLPackCapsule",
    "DLPackConvertible",
    "DType",
    "DTypeLike",
    "DtypeCategory",
    "Device",
    "DeviceType",
    "IndexElement",
    "IndexLike",
    "LossReduction",
    "MlxArrayConvertible",
    "NestedSequence",
    "PadMode",
    "ParameterTree",
    "PyTree",
    "QuantizationMode",
    "Scalar",
    "ShapeLike",
    "Stream",
    "StreamOrDevice",
]
