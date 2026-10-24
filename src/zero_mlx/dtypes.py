"""Data types and type-checking helpers for zero_mlx."""

from __future__ import annotations

from enum import Enum
from typing import Union

from ml_switcheroo_compiler.core.dtype import DType as SwitcherooDType


class DType(Enum):
    """Enumeration of MLX numeric and boolean data types."""

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
        """Return the size of the data type in bytes.

        Returns:
            Number of bytes per element for this data type.

        """
        sizes: dict[str, int] = {
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
        """Return the official string representation of the MLX data type.

        Returns:
            String formatted as `mlx.core.<dtype>`.

        """
        return f"mlx.core.{self.value}"

    def __str__(self) -> str:
        """Return the human-readable string representation of the MLX data type.

        Returns:
            String formatted as `mlx.core.<dtype>`.

        """
        return f"mlx.core.{self.value}"


class DtypeCategory(Enum):
    """Enumeration of MLX data type hierarchy categories."""

    complexfloating = "complexfloating"
    floating = "floating"
    inexact = "inexact"
    signedinteger = "signedinteger"
    unsignedinteger = "unsignedinteger"
    integer = "integer"
    number = "number"
    generic = "generic"


Dtype = DType

bool_: DType = DType.bool_
uint8: DType = DType.uint8
uint16: DType = DType.uint16
uint32: DType = DType.uint32
uint64: DType = DType.uint64
int8: DType = DType.int8
int16: DType = DType.int16
int32: DType = DType.int32
int64: DType = DType.int64
float16: DType = DType.float16
bfloat16: DType = DType.bfloat16
float32: DType = DType.float32
float64: DType = DType.float64
complex64: DType = DType.complex64
complex128: DType = DType.complex128

complexfloating: DtypeCategory = DtypeCategory.complexfloating
floating: DtypeCategory = DtypeCategory.floating
inexact: DtypeCategory = DtypeCategory.inexact
signedinteger: DtypeCategory = DtypeCategory.signedinteger
unsignedinteger: DtypeCategory = DtypeCategory.unsignedinteger
integer: DtypeCategory = DtypeCategory.integer
number: DtypeCategory = DtypeCategory.number
generic: DtypeCategory = DtypeCategory.generic

DTypeInput = Union[
    DType,
    DtypeCategory,
    SwitcherooDType,
    str,
    type[bool],
    type[int],
    type[float],
    type[complex],
    None,
]
DTypeLike = DTypeInput

_FLOATING_DTYPES: frozenset[DType] = frozenset(
    {DType.float16, DType.bfloat16, DType.float32, DType.float64}
)
_COMPLEX_DTYPES: frozenset[DType] = frozenset({DType.complex64, DType.complex128})
_SIGNED_INT_DTYPES: frozenset[DType] = frozenset(
    {DType.int8, DType.int16, DType.int32, DType.int64}
)
_UNSIGNED_INT_DTYPES: frozenset[DType] = frozenset(
    {DType.uint8, DType.uint16, DType.uint32, DType.uint64}
)
_INTEGER_DTYPES: frozenset[DType] = _SIGNED_INT_DTYPES | _UNSIGNED_INT_DTYPES
_INEXACT_DTYPES: frozenset[DType] = _FLOATING_DTYPES | _COMPLEX_DTYPES
_NUMBER_DTYPES: frozenset[DType] = _INTEGER_DTYPES | _INEXACT_DTYPES
_ALL_DTYPES: frozenset[DType] = _NUMBER_DTYPES | frozenset({DType.bool_})

_CATEGORY_MAP: dict[str, frozenset[DType]] = {
    "bool": frozenset({DType.bool_}),
    "bool_": frozenset({DType.bool_}),
    "signedinteger": _SIGNED_INT_DTYPES,
    "signed integer": _SIGNED_INT_DTYPES,
    "unsignedinteger": _UNSIGNED_INT_DTYPES,
    "unsigned integer": _UNSIGNED_INT_DTYPES,
    "integer": _INTEGER_DTYPES,
    "integral": _INTEGER_DTYPES,
    "floating": _FLOATING_DTYPES,
    "real floating": _FLOATING_DTYPES,
    "complexfloating": _COMPLEX_DTYPES,
    "complex floating": _COMPLEX_DTYPES,
    "inexact": _INEXACT_DTYPES,
    "number": _NUMBER_DTYPES,
    "numeric": _NUMBER_DTYPES,
    "generic": _ALL_DTYPES,
}


def resolve_dtype(dtype: DTypeInput, default: DType = DType.float32) -> DType:
    """Resolve a dtype-like input value into a concrete `DType` enum member.

    Args:
        dtype: Input dtype specification (DType, SwitcherooDType, str, Python type, or None).
        default: Default `DType` to return when `dtype` is None.

    Returns:
        Resolved `DType` enum member.

    Raises:
        ValueError: If the input string or category cannot be resolved to a concrete `DType`.

    """
    if dtype is None:
        return default
    if isinstance(dtype, DType):
        return dtype
    if isinstance(dtype, SwitcherooDType):
        return to_mlx_dtype(dtype)
    if dtype is bool:
        return DType.bool_
    if dtype is int:
        return DType.int32
    if dtype is float:
        return DType.float32
    if dtype is complex:
        return DType.complex64
    if isinstance(dtype, DtypeCategory):
        cat_defaults: dict[DtypeCategory, DType] = {
            DtypeCategory.floating: DType.float32,
            DtypeCategory.inexact: DType.float32,
            DtypeCategory.complexfloating: DType.complex64,
            DtypeCategory.unsignedinteger: DType.uint32,
        }
        return cat_defaults.get(dtype, DType.int32)

    val: str = str(dtype)
    val = val[len("mlx.core.") :] if val.startswith("mlx.core.") else val
    if val in ("bool", "bool_"):
        return DType.bool_
    return DType(val)


def to_switcheroo_dtype(dtype: DTypeInput) -> SwitcherooDType:
    """Convert an MLX `DType` or dtype-like object to `ml_switcheroo_compiler.core.dtype.DType`.

    Args:
        dtype: The MLX data type to convert.

    Returns:
        The corresponding `SwitcherooDType` enum member.

    """
    if isinstance(dtype, SwitcherooDType):
        return dtype
    resolved: DType = resolve_dtype(dtype)
    return SwitcherooDType(resolved.value)


def to_mlx_dtype(dtype: SwitcherooDType | DType | str) -> DType:
    """Convert an `ml_switcheroo_compiler` dtype or string to an MLX `DType`.

    Args:
        dtype: The compiler `SwitcherooDType`, `DType`, or string name.

    Returns:
        The corresponding MLX `DType` enum member.

    """
    if isinstance(dtype, DType):
        return dtype
    val: str = dtype.value if isinstance(dtype, SwitcherooDType) else str(dtype)
    if val.startswith("mlx.core."):
        val = val[len("mlx.core.") :]
    if val in ("bool", "bool_"):
        return DType.bool_
    return DType(val)


def _category_dtypes(
    cat: DTypeInput,
) -> frozenset[DType]:
    """Return the set of concrete `DType` members belonging to a dtype or category.

    Args:
        cat: A concrete `DType`, `DtypeCategory`, Python type, or category name.

    Returns:
        Frozenset of matching `DType` members.

    """
    if isinstance(cat, DType):
        return frozenset({cat})
    if isinstance(cat, DtypeCategory):
        return _CATEGORY_MAP.get(cat.value, frozenset())
    if cat is bool:
        return frozenset({DType.bool_})
    if cat is int:
        return _INTEGER_DTYPES
    if cat is float:
        return _FLOATING_DTYPES
    if cat is complex:
        return _COMPLEX_DTYPES
    if isinstance(cat, str):
        clean: str = cat[len("mlx.core.") :] if cat.startswith("mlx.core.") else cat
        if clean in _CATEGORY_MAP:
            return _CATEGORY_MAP[clean]
        try:
            return frozenset({DType(clean)})
        except ValueError:
            return frozenset()
    return frozenset({to_mlx_dtype(cat)}) if isinstance(cat, SwitcherooDType) else frozenset()


def issubdtype(arg1: DTypeInput, arg2: DTypeInput) -> bool:
    """Return True if the first dtype argument is a subdtype of the second dtype or category.

    Args:
        arg1: First dtype or category to test.
        arg2: Target dtype or category to test against.

    Returns:
        True if `arg1` is a subdtype of `arg2`, False otherwise.

    """
    set1: frozenset[DType] = _category_dtypes(arg1)
    set2: frozenset[DType] = _category_dtypes(arg2)
    if not set1 or not set2:
        return False
    return set1.issubset(set2)


def isdtype(
    dtype: DTypeInput,
    kind: DTypeInput | tuple[DTypeInput, ...],
) -> bool:
    """Determine whether a given `dtype` belongs to the specified `kind` or tuple of kinds.

    Args:
        dtype: Data type to inspect.
        kind: Single dtype/category name or tuple of dtypes/category names.

    Returns:
        True if `dtype` matches `kind`, False otherwise.

    """
    resolved: DType = resolve_dtype(dtype)
    kinds: tuple[DTypeInput, ...] = kind if isinstance(kind, tuple) else (kind,)
    return any(resolved in _category_dtypes(k) for k in kinds)


_PROMOTION_ORDER: tuple[DType, ...] = (
    DType.bool_,
    DType.uint8,
    DType.int8,
    DType.uint16,
    DType.int16,
    DType.uint32,
    DType.int32,
    DType.uint64,
    DType.int64,
    DType.float16,
    DType.bfloat16,
    DType.float32,
    DType.float64,
    DType.complex64,
    DType.complex128,
)


def promote_dtypes(dt1: DType, dt2: DType) -> DType:
    """Compute the promoted MLX `DType` for two input data types.

    Args:
        dt1: First `DType`.
        dt2: Second `DType`.

    Returns:
        The promoted `DType` following MLX type promotion rules.

    """
    if dt1 == dt2:
        return dt1
    if {dt1, dt2} == {DType.float16, DType.bfloat16}:
        return DType.float32
    if dt1 in _SIGNED_INT_DTYPES and dt2 in _UNSIGNED_INT_DTYPES:
        if dt2 == DType.uint8 and dt1 in (DType.int16, DType.int32, DType.int64):
            return dt1
        if dt2 == DType.uint16 and dt1 in (DType.int32, DType.int64):
            return dt1
        if dt2 == DType.uint32 and dt1 == DType.int64:
            return DType.int64
        return (
            DType.float32
            if dt2 == DType.uint64
            else _PROMOTION_ORDER[max(_PROMOTION_ORDER.index(dt1), _PROMOTION_ORDER.index(dt2))]
        )
    if dt2 in _SIGNED_INT_DTYPES and dt1 in _UNSIGNED_INT_DTYPES:
        return promote_dtypes(dt2, dt1)
    idx1: int = _PROMOTION_ORDER.index(dt1)
    idx2: int = _PROMOTION_ORDER.index(dt2)
    return _PROMOTION_ORDER[max(idx1, idx2)]
