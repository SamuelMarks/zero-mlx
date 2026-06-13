# ruff: noqa
"""Dtypes for zero_mlx."""

from enum import Enum
from ml_switcheroo_compiler.core.dtype import DType as SwitcherooDType


class DType(Enum):
    """MLX data types."""

    float32 = "float32"
    float16 = "float16"
    bfloat16 = "bfloat16"
    float64 = "float64"
    complex64 = "complex64"
    complex128 = "complex128"
    int64 = "int64"
    int32 = "int32"
    int16 = "int16"
    int8 = "int8"
    uint64 = "uint64"
    uint32 = "uint32"
    uint16 = "uint16"
    uint8 = "uint8"
    bool_ = "bool"

    @property
    def size(self) -> int:
        """Get size in bytes."""
        sizes = {
            "float32": 4,
            "float16": 2,
            "bfloat16": 2,
            "float64": 8,
            "complex64": 8,
            "complex128": 16,
            "int64": 8,
            "int32": 4,
            "int16": 2,
            "int8": 1,
            "uint64": 8,
            "uint32": 4,
            "uint16": 2,
            "uint8": 1,
            "bool": 1,
        }
        return sizes[self.value]

    def __repr__(self) -> str:
        """Repr."""
        return f"mlx.core.{self.value}"  # pragma: no cover

    def __str__(self) -> str:
        """Str."""
        return f"mlx.core.{self.value}"


def to_switcheroo_dtype(dtype: DType) -> SwitcherooDType:
    """Convert MLX dtype to ml_switcheroo dtype."""
    if hasattr(dtype, "value"):
        val = dtype.value
    elif hasattr(dtype, "name"):  # pragma: no cover
        val = dtype.name  # pragma: no cover
    else:
        val = str(dtype)  # pragma: no cover
    if val.startswith("uint") and val != "uint8":
        val = val[1:]
    return SwitcherooDType(val)


bool_ = DType.bool_
uint8 = DType.uint8
uint16 = DType.uint16
uint32 = DType.uint32
uint64 = DType.uint64
int8 = DType.int8
int16 = DType.int16
int32 = DType.int32
int64 = DType.int64
float16 = DType.float16
bfloat16 = DType.bfloat16
float32 = DType.float32
float64 = DType.float64
complex64 = DType.complex64
complex128 = DType.complex128


def to_mlx_dtype(dtype: SwitcherooDType) -> DType:
    """Convert ml_switcheroo dtype to MLX dtype."""
    val = dtype.value
    if val == "bool":
        return DType.bool_
    return DType(val)


class DtypeCategory(Enum):
    """Type to hold categories of dtypes."""

    complexfloating = "complexfloating"
    floating = "floating"
    inexact = "inexact"
    signedinteger = "signedinteger"
    unsignedinteger = "unsignedinteger"
    integer = "integer"
    number = "number"
    generic = "generic"


Dtype = DType
