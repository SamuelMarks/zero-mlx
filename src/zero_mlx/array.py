"""N-dimensional array class (`zero_mlx.array`) wrapping `ml_switcheroo_compiler.Tensor`."""

from __future__ import annotations

import math
import types
from typing import TYPE_CHECKING, Sequence

import ml_switcheroo_compiler as ml_switcheroo
from ml_switcheroo_compiler.core.dtype import DType as CDType
import ml_switcheroo_compiler.ops as sops
import ml_switcheroo_compiler.ops.shape.frontend as sf
import struct

if TYPE_CHECKING:
    from zero_mlx.array_iterator import ArrayIterator
    from zero_mlx.at_mocker import ArrayAt

from zero_mlx._types import (
    ArrayInput,
    ArrayProtocol,
    AxesLike,
    DLPackCapsule,
    DLPackConvertible,
    DTypeLike,
    IndexLike,
    MlxArrayConvertible,
    NestedSequence,
    Scalar,
    StreamOrDevice,
)
from zero_mlx.dtypes import (
    DType,
    promote_dtypes,
    resolve_dtype,
    to_mlx_dtype,
    to_switcheroo_dtype,
)


def _infer_scalar_or_seq_dtype(val: Scalar | NestedSequence | array | ArrayProtocol) -> DType:
    """Infer the default MLX `DType` for a Python scalar, nested sequence, or `array`.

    Args:
        val: Input scalar, nested sequence, or `array`.

    Returns:
        Inferred MLX `DType`.

    """
    if isinstance(val, array):
        return val.dtype
    if isinstance(val, bool):
        return DType.bool_
    if isinstance(val, int):
        if val < -2147483648 or val > 2147483647:
            return DType.int64
        return DType.int32
    if isinstance(val, float):
        return DType.float32
    if isinstance(val, complex):
        return DType.complex64
    if isinstance(val, str):
        raise ValueError("Cannot convert string to mlx array")
    if isinstance(val, ArrayProtocol):
        dt_str = str(val.dtype).split(".")[-1]
        if dt_str in ("float64", "complex128"):
            return DType.float32 if dt_str == "float64" else DType.complex64
        return resolve_dtype(dt_str)
    if len(val) == 0:
        return DType.float32
    cur_dt: DType = _infer_scalar_or_seq_dtype(val[0])
    for item in val[1:]:
        cur_dt = promote_dtypes(cur_dt, _infer_scalar_or_seq_dtype(item))
    return cur_dt


def _unwrap_nested_arrays(
    val: Scalar | NestedSequence | array | ArrayProtocol,
) -> Scalar | list[Scalar | NestedSequence]:
    """Recursively convert any nested `zero_mlx.array` elements inside sequences to Python lists/scalars.

    Args:
        val: Input scalar, sequence, or `array`.

    Returns:
        Pure Python scalar or nested list representation.

    Raises:
        ValueError: If sub-sequences have inhomogeneous lengths.

    """
    if isinstance(val, (array, ArrayProtocol)):
        return val.tolist()
    if isinstance(val, (bool, int, float, complex)):
        return val
    converted: list[Scalar | NestedSequence] = [_unwrap_nested_arrays(x) for x in val]
    if converted and any(isinstance(x, list) for x in converted):
        if not all(isinstance(x, list) for x in converted):
            raise ValueError("Initialization encountered non-uniform length.")
        first_len: int = len(converted[0]) if isinstance(converted[0], list) else 0
        for sub in converted[1:]:
            if isinstance(sub, list) and len(sub) != first_len:
                raise ValueError("Initialization encountered non-uniform length.")
    return converted


def _to_tensor(
    x: ArrayInput | ml_switcheroo.Tensor,
    dtype: DTypeLike = None,
) -> tuple[ml_switcheroo.Tensor, DType]:
    """Convert an `ArrayInput` or `Tensor` into a `(Tensor, DType)` pair.

    Args:
        x: Input array, scalar, sequence, or compiler `Tensor`.
        dtype: Optional target `DTypeLike`.

    Returns:
        Tuple of `(compiler_tensor, resolved_mlx_dtype)`.

    Raises:
        ValueError: If input contains invalid elements or inhomogeneous sequences.

    """
    if isinstance(x, array):
        target_dt: DType = resolve_dtype(dtype, default=x.dtype)
        return (
            (sops.cast(x._tensor, to_switcheroo_dtype(target_dt)), target_dt)
            if target_dt != x.dtype
            else (x._tensor, target_dt)
        )

    if isinstance(x, ml_switcheroo.Tensor):
        inferred_mlx: DType = to_mlx_dtype(x.dtype)
        default_dt = DType.float32 if inferred_mlx == DType.float64 else inferred_mlx
        target_dt = resolve_dtype(dtype, default=default_dt)
        return x, target_dt

    if isinstance(x, MlxArrayConvertible):
        arr_conv: array = x.__mlx_array__()
        return _to_tensor(arr_conv, dtype=dtype)

    if isinstance(x, ArrayProtocol):
        arr_dt = _infer_scalar_or_seq_dtype(x)
        target_dt = resolve_dtype(dtype, default=arr_dt)
        return (
            sops.array(x.tolist(), dtype=to_switcheroo_dtype(target_dt)),
            target_dt,
        )

    if isinstance(x, DLPackConvertible):
        cap = x.__dlpack__()
        return _to_tensor(cap.data, dtype=dtype if dtype is not None else cap.dtype)

    if isinstance(x, (bytes, bytearray)):
        target_dt = resolve_dtype(dtype, default=DType.uint8)
        raw_list: list[int] = list(x)
        return (
            sops.array(raw_list, dtype=to_switcheroo_dtype(target_dt)),
            target_dt,
        )

    inferred_dt: DType = _infer_scalar_or_seq_dtype(x)
    target_dt = resolve_dtype(dtype, default=inferred_dt)
    clean_data = _unwrap_nested_arrays(x)
    sw_dt: CDType = to_switcheroo_dtype(target_dt)
    tensor: ml_switcheroo.Tensor = sops.array(clean_data, dtype=sw_dt)
    return tensor, target_dt


class array:
    """An N-dimensional array object compatible with `mlx.core.array`."""

    def __init__(
        self,
        data: ArrayInput | ml_switcheroo.Tensor,
        dtype: DTypeLike = None,
    ) -> None:
        """Initialize an `array` from scalar, sequence, `array`, or compiler `Tensor`.

        Args:
            data: Input scalar, nested sequence, `array`, or compiler `Tensor`.
            dtype: Optional data type override.

        """
        tensor, resolved_dt = _to_tensor(data, dtype=dtype)
        self._tensor: ml_switcheroo.Tensor = tensor
        self._original_dtype: DType = resolved_dt

    def _coerce_other(
        self, other: ArrayInput, allow_list: bool = False
    ) -> tuple[ml_switcheroo.Tensor, DType]:
        """Convert a binary operand into a `(Tensor, DType)` pair.

        Args:
            other: Operand value (`array`, scalar, or sequence if `allow_list` is True).
            allow_list: Whether Python `list`/`tuple` operands are permitted.

        Returns:
            Tuple of `(operand_tensor, promoted_dtype)`.

        Raises:
            ValueError: If `other` is a disallowed sequence or overflowing integer scalar.

        """
        if isinstance(other, str):
            raise ValueError("Unsupported type for array operation.")
        if isinstance(other, array):
            return other._tensor, promote_dtypes(self.dtype, other.dtype)
        if isinstance(other, (list, tuple)):
            if not allow_list:
                raise ValueError(f"Invalid type {type(other)} for array operation.")
            other_arr = array(other)
            return other_arr._tensor, promote_dtypes(self.dtype, other_arr.dtype)
        if isinstance(other, (bool, int, float, complex)):
            if (
                isinstance(other, int)
                and not isinstance(other, bool)
                and self.dtype == DType.int32
                and (other < -2147483648 or other > 2147483647)
            ):
                raise ValueError(f"Converting {other} to int32 would result in overflow.")
            other_dt: DType = _infer_scalar_or_seq_dtype(other)
            if (
                isinstance(other, int)
                and not isinstance(other, bool)
                and self.dtype
                in (
                    DType.int8,
                    DType.int16,
                    DType.int32,
                    DType.int64,
                    DType.uint8,
                    DType.uint16,
                    DType.uint32,
                    DType.uint64,
                    DType.float16,
                    DType.bfloat16,
                    DType.float32,
                    DType.float64,
                )
            ) or (
                isinstance(other, float)
                and self.dtype
                in (
                    DType.float16,
                    DType.bfloat16,
                    DType.float32,
                    DType.float64,
                )
            ):
                other_dt = self.dtype
            promoted: DType = promote_dtypes(self.dtype, other_dt)
            t = sops.array(other, dtype=to_switcheroo_dtype(other_dt))
            return t, promoted
        other_arr = array(other)
        return other_arr._tensor, promote_dtypes(self.dtype, other_arr.dtype)

    def _inplace_assign(self, new_arr: array) -> array:
        """Assign the tensor and dtype from `new_arr` into `self` for in-place operations.

        Args:
            new_arr: Resulting array of the operation.

        Returns:
            `self` updated in-place.

        """
        self._tensor = new_arr._tensor
        self._original_dtype = new_arr._original_dtype
        return self

    def __repr__(self) -> str:
        """Return the official string representation of the array.

        Returns:
            Formatted string `array(..., dtype=...)`.

        """
        from zero_mlx.array_repr import format_array_repr

        return format_array_repr(self)

    def __str__(self) -> str:
        """Return the string representation of the array.

        Returns:
            Formatted string `array(..., dtype=...)`.

        """
        return self.__repr__()

    def __format__(self, format_spec: str) -> str:
        """Format a scalar array according to `format_spec`.

        Args:
            format_spec: Standard Python format specification string.

        Returns:
            Formatted string.

        Raises:
            TypeError: If `format_spec` is non-empty and the array has more than 1 element.

        """
        if format_spec == "":
            return str(self)
        if self.size != 1:
            raise TypeError("unsupported format string passed to array.__format__")
        return format(self.item(), format_spec)

    def __add__(self, other: ArrayInput) -> array:
        """Compute element-wise addition `self + other`.

        Args:
            other: Right-hand operand.

        Returns:
            Resulting `array`.

        """
        t_other, dt = self._coerce_other(other)
        return array(self._tensor + t_other, dtype=dt)

    def __radd__(self, other: ArrayInput) -> array:
        """Compute reverse element-wise addition `other + self`.

        Args:
            other: Left-hand operand.

        Returns:
            Resulting `array`.

        """
        t_other, dt = self._coerce_other(other)
        return array(t_other + self._tensor, dtype=dt)

    def __iadd__(self, other: ArrayInput) -> array:
        """Compute in-place addition `self += other`.

        Args:
            other: Right-hand operand.

        Returns:
            Updated `self`.

        """
        return self._inplace_assign(self + other)

    def __sub__(self, other: ArrayInput) -> array:
        """Compute element-wise subtraction `self - other`.

        Args:
            other: Right-hand operand.

        Returns:
            Resulting `array`.

        """
        t_other, dt = self._coerce_other(other)
        return array(self._tensor - t_other, dtype=dt)

    def __rsub__(self, other: ArrayInput) -> array:
        """Compute reverse element-wise subtraction `other - self`.

        Args:
            other: Left-hand operand.

        Returns:
            Resulting `array`.

        """
        t_other, dt = self._coerce_other(other)
        return array(t_other - self._tensor, dtype=dt)

    def __isub__(self, other: ArrayInput) -> array:
        """Compute in-place subtraction `self -= other`.

        Args:
            other: Right-hand operand.

        Returns:
            Updated `self`.

        """
        return self._inplace_assign(self - other)

    def __mul__(self, other: ArrayInput) -> array:
        """Compute element-wise multiplication `self * other`.

        Args:
            other: Right-hand operand.

        Returns:
            Resulting `array`.

        """
        t_other, dt = self._coerce_other(other)
        return array(self._tensor * t_other, dtype=dt)

    def __rmul__(self, other: ArrayInput) -> array:
        """Compute reverse element-wise multiplication `other * self`.

        Args:
            other: Left-hand operand.

        Returns:
            Resulting `array`.

        """
        t_other, dt = self._coerce_other(other)
        return array(t_other * self._tensor, dtype=dt)

    def __imul__(self, other: ArrayInput) -> array:
        """Compute in-place multiplication `self *= other`.

        Args:
            other: Right-hand operand.

        Returns:
            Updated `self`.

        """
        return self._inplace_assign(self * other)

    def __truediv__(self, other: ArrayInput) -> array:
        """Compute element-wise true division `self / other`.

        Args:
            other: Right-hand operand.

        Returns:
            Resulting `array`.

        """
        t_other, dt = self._coerce_other(other)
        out_dt = (
            DType.float32
            if dt
            in (
                DType.bool_,
                DType.int8,
                DType.int16,
                DType.int32,
                DType.int64,
                DType.uint8,
                DType.uint16,
                DType.uint32,
                DType.uint64,
            )
            else dt
        )
        return array(self._tensor / t_other, dtype=out_dt)

    def __rtruediv__(self, other: ArrayInput) -> array:
        """Compute reverse element-wise true division `other / self`.

        Args:
            other: Left-hand operand.

        Returns:
            Resulting `array`.

        """
        t_other, dt = self._coerce_other(other)
        out_dt = (
            DType.float32
            if dt
            in (
                DType.bool_,
                DType.int8,
                DType.int16,
                DType.int32,
                DType.int64,
                DType.uint8,
                DType.uint16,
                DType.uint32,
                DType.uint64,
            )
            else dt
        )
        return array(t_other / self._tensor, dtype=out_dt)

    def __itruediv__(self, other: ArrayInput) -> array:
        """Compute in-place true division `self /= other`.

        Args:
            other: Right-hand operand.

        Returns:
            Updated `self`.

        """
        return self._inplace_assign(self / other)

    def __floordiv__(self, other: ArrayInput) -> array:
        """Compute element-wise floor division `self // other`.

        Args:
            other: Right-hand operand.

        Returns:
            Resulting `array`.

        """
        t_other, dt = self._coerce_other(other)
        return array(self._tensor // t_other, dtype=dt)

    def __rfloordiv__(self, other: ArrayInput) -> array:
        """Compute reverse element-wise floor division `other // self`.

        Args:
            other: Left-hand operand.

        Returns:
            Resulting `array`.

        """
        t_other, dt = self._coerce_other(other)
        return array(t_other // self._tensor, dtype=dt)

    def __ifloordiv__(self, other: ArrayInput) -> array:
        """Compute in-place floor division `self //= other`.

        Args:
            other: Right-hand operand.

        Returns:
            Updated `self`.

        """
        return self._inplace_assign(self // other)

    def __mod__(self, other: ArrayInput) -> array:
        """Compute element-wise remainder `self % other`.

        Args:
            other: Right-hand operand.

        Returns:
            Resulting `array`.

        """
        t_other, dt = self._coerce_other(other)
        return array(self._tensor % t_other, dtype=dt)

    def __rmod__(self, other: ArrayInput) -> array:
        """Compute reverse element-wise remainder `other % self`.

        Args:
            other: Left-hand operand.

        Returns:
            Resulting `array`.

        """
        t_other, dt = self._coerce_other(other)
        return array(t_other % self._tensor, dtype=dt)

    def __imod__(self, other: ArrayInput) -> array:
        """Compute in-place remainder `self %= other`.

        Args:
            other: Right-hand operand.

        Returns:
            Updated `self`.

        """
        return self._inplace_assign(self % other)

    def __pow__(self, other: ArrayInput) -> array:
        """Compute element-wise exponentiation `self ** other`.

        Args:
            other: Exponent operand.

        Returns:
            Resulting `array`.

        """
        t_other, dt = self._coerce_other(other)
        return array(self._tensor**t_other, dtype=dt)

    def __rpow__(self, other: ArrayInput) -> array:
        """Compute reverse element-wise exponentiation `other ** self`.

        Args:
            other: Base operand.

        Returns:
            Resulting `array`.

        """
        t_other, dt = self._coerce_other(other)
        return array(t_other**self._tensor, dtype=dt)

    def __ipow__(self, other: ArrayInput) -> array:
        """Compute in-place exponentiation `self **= other`.

        Args:
            other: Exponent operand.

        Returns:
            Updated `self`.

        """
        return self._inplace_assign(self**other)

    def __and__(self, other: ArrayInput) -> array:
        """Compute element-wise bitwise AND `self & other`.

        Args:
            other: Right-hand operand.

        Returns:
            Resulting `array`.

        """
        t_other, dt = self._coerce_other(other)
        return array(self._tensor & t_other, dtype=dt)

    def __rand__(self, other: ArrayInput) -> array:
        """Compute reverse element-wise bitwise AND `other & self`.

        Args:
            other: Left-hand operand.

        Returns:
            Resulting `array`.

        """
        t_other, dt = self._coerce_other(other)
        return array(t_other & self._tensor, dtype=dt)

    def __iand__(self, other: ArrayInput) -> array:
        """Compute in-place bitwise AND `self &= other`.

        Args:
            other: Right-hand operand.

        Returns:
            Updated `self`.

        """
        return self._inplace_assign(self & other)

    def __or__(self, other: ArrayInput) -> array:
        """Compute element-wise bitwise OR `self | other`.

        Args:
            other: Right-hand operand.

        Returns:
            Resulting `array`.

        """
        t_other, dt = self._coerce_other(other)
        return array(self._tensor | t_other, dtype=dt)

    def __ror__(self, other: ArrayInput) -> array:
        """Compute reverse element-wise bitwise OR `other | self`.

        Args:
            other: Left-hand operand.

        Returns:
            Resulting `array`.

        """
        t_other, dt = self._coerce_other(other)
        return array(t_other | self._tensor, dtype=dt)

    def __ior__(self, other: ArrayInput) -> array:
        """Compute in-place bitwise OR `self |= other`.

        Args:
            other: Right-hand operand.

        Returns:
            Updated `self`.

        """
        return self._inplace_assign(self | other)

    def __xor__(self, other: ArrayInput) -> array:
        """Compute element-wise bitwise XOR `self ^ other`.

        Args:
            other: Right-hand operand.

        Returns:
            Resulting `array`.

        """
        t_other, dt = self._coerce_other(other)
        return array(self._tensor ^ t_other, dtype=dt)

    def __rxor__(self, other: ArrayInput) -> array:
        """Compute reverse element-wise bitwise XOR `other ^ self`.

        Args:
            other: Left-hand operand.

        Returns:
            Resulting `array`.

        """
        t_other, dt = self._coerce_other(other)
        return array(t_other ^ self._tensor, dtype=dt)

    def __ixor__(self, other: ArrayInput) -> array:
        """Compute in-place bitwise XOR `self ^= other`.

        Args:
            other: Right-hand operand.

        Returns:
            Updated `self`.

        """
        return self._inplace_assign(self ^ other)

    def __lshift__(self, other: ArrayInput) -> array:
        """Compute element-wise left bit shift `self << other`.

        Args:
            other: Shift count operand.

        Returns:
            Resulting `array`.

        """
        import zero_mlx.ops as _ops

        return _ops.left_shift(self, other)

    def __rshift__(self, other: ArrayInput) -> array:
        """Compute element-wise right bit shift `self >> other`.

        Args:
            other: Shift count operand.

        Returns:
            Resulting `array`.

        """
        import zero_mlx.ops as _ops

        return _ops.right_shift(self, other)

    def __matmul__(self, other: ArrayInput) -> array:
        """Compute matrix multiplication `self @ other`.

        Args:
            other: Right-hand matrix/vector operand.

        Returns:
            Matrix product `array`.

        """
        t_other, dt = self._coerce_other(other)
        return array(sops.matmul(self._tensor, t_other), dtype=dt)

    def __imatmul__(self, other: ArrayInput) -> array:
        """Compute in-place matrix multiplication `self @= other`.

        Args:
            other: Right-hand matrix operand.

        Returns:
            Updated `self`.

        """
        return self._inplace_assign(self @ other)

    def __lt__(self, other: ArrayInput) -> array:
        """Compute element-wise less-than `self < other`.

        Args:
            other: Right-hand operand.

        Returns:
            Boolean `array`.

        """
        t_other, _ = self._coerce_other(other)
        return array(self._tensor < t_other, dtype=DType.bool_)

    def __le__(self, other: ArrayInput) -> array:
        """Compute element-wise less-than-or-equal `self <= other`.

        Args:
            other: Right-hand operand.

        Returns:
            Boolean `array`.

        """
        t_other, _ = self._coerce_other(other)
        return array(self._tensor <= t_other, dtype=DType.bool_)

    def __gt__(self, other: ArrayInput) -> array:
        """Compute element-wise greater-than `self > other`.

        Args:
            other: Right-hand operand.

        Returns:
            Boolean `array`.

        """
        t_other, _ = self._coerce_other(other)
        return array(self._tensor > t_other, dtype=DType.bool_)

    def __ge__(self, other: ArrayInput) -> array:
        """Compute element-wise greater-than-or-equal `self >= other`.

        Args:
            other: Right-hand operand.

        Returns:
            Boolean `array`.

        """
        t_other, _ = self._coerce_other(other)
        return array(self._tensor >= t_other, dtype=DType.bool_)

    def __eq__(self, other: object) -> array:  # type: ignore[override]
        """Compute element-wise equality `self == other`.

        Args:
            other: Value or `array` to compare against.

        Returns:
            Boolean `array`.

        """
        if isinstance(other, (list, tuple)) or other is None:
            return array(False, dtype=DType.bool_)
        if isinstance(other, (array, int, float, bool, complex)):
            t_other, _ = self._coerce_other(other)
            return array(self._tensor == t_other, dtype=DType.bool_)
        return array(False, dtype=DType.bool_)

    def __ne__(self, other: object) -> array:  # type: ignore[override]
        """Compute element-wise inequality `self != other`.

        Args:
            other: Value or `array` to compare against.

        Returns:
            Boolean `array`.

        """
        if isinstance(other, (list, tuple)) or other is None:
            return array(True, dtype=DType.bool_)
        if isinstance(other, (array, int, float, bool, complex)):
            t_other, _ = self._coerce_other(other)
            return array(self._tensor != t_other, dtype=DType.bool_)
        return array(True, dtype=DType.bool_)

    def __neg__(self) -> array:
        """Compute element-wise negation `-self`.

        Returns:
            Negated `array`.

        """
        return array(sops.negative(self._tensor), dtype=self.dtype)

    def __pos__(self) -> array:
        """Compute unary positive `+self`.

        Returns:
            Copy of `self`.

        """
        return self.copy()

    def __abs__(self) -> array:
        """Compute element-wise absolute value `abs(self)`.

        Returns:
            Absolute value `array`.

        """
        return array(sops.abs(self._tensor), dtype=self.dtype)

    def __invert__(self) -> array:
        """Compute element-wise bitwise inversion `~self`.

        Returns:
            Bitwise inverted `array`.

        """
        return array(sops.bitwise_not(self._tensor), dtype=self.dtype)

    def __dlpack_device__(self) -> tuple[int, int]:
        """Return the DLPack device tuple `(device_type, device_id)`.

        Returns:
            Tuple `(1, 0)` representing CPU/unified memory.

        """
        return (1, 0)

    def __dlpack__(self, stream: int | None = None) -> DLPackCapsule:
        """Export a `DLPackCapsule` for zero-copy/pure-Python DLPack exchange.

        Args:
            stream: Optional stream index.

        Returns:
            A `DLPackCapsule` containing this array's data and dtype.

        """
        _ = stream
        return DLPackCapsule(
            data=self.tolist(), dtype=self.dtype, device_info=self.__dlpack_device__()
        )

    def __array_namespace__(self, *, api_version: str | None = None) -> types.ModuleType:
        """Return the Python Array API namespace module (`zero_mlx`).

        Args:
            api_version: Optional Array API specification version string.

        Returns:
            The `zero_mlx` top-level module.

        """
        _ = api_version
        import zero_mlx as mx

        return mx

    def astype(self, dtype: DTypeLike, stream: StreamOrDevice = None) -> array:
        """Cast the array to a specified `dtype`.

        Args:
            dtype: Target data type.
            stream: Optional execution stream or device.

        Returns:
            New `array` cast to `dtype`.

        """
        _ = stream
        resolved: DType = resolve_dtype(dtype, default=self.dtype)
        return array(
            sops.cast(self._tensor, dtype=to_switcheroo_dtype(resolved)),
            dtype=resolved,
        )

    def reshape(self, *shape: int | Sequence[int]) -> array:
        """Return an array with the same data reshaped to `shape`.

        Args:
            *shape: Target dimensions as integers or a single sequence of integers.

        Returns:
            Reshaped `array`.

        """
        target_shape: tuple[int, ...]
        if len(shape) == 1 and isinstance(shape[0], Sequence):
            target_shape = tuple(int(s) for s in shape[0])
        else:
            target_shape = tuple(int(s) for s in shape if isinstance(s, int))
        return array(sf.reshape(self._tensor, target_shape), dtype=self.dtype)

    def squeeze(self, axis: AxesLike = None, *, stream: StreamOrDevice = None) -> array:
        """Remove single-dimensional axes from the array shape.

        Args:
            axis: Optional axis or sequence of axes of size 1 to remove.
            stream: Optional execution stream or device.

        Returns:
            Squeezed `array`.

        """
        _ = stream
        axes_tuple: tuple[int, ...]
        if axis is None:
            axes_tuple = tuple(i for i, s in enumerate(self.shape) if s == 1)
        elif isinstance(axis, int):
            axes_tuple = (axis,)
        else:
            axes_tuple = tuple(axis)
        tensor = self._tensor
        for a in reversed(sorted(axes_tuple)):
            tensor = sf.squeeze(tensor, axis=a)
        return array(tensor, dtype=self.dtype)

    def tolist(self) -> Scalar | list[Scalar | NestedSequence]:
        """Convert the array into a Python scalar (if 0D) or nested Python list.

        Returns:
            Python scalar or nested list of scalars.

        """
        from zero_mlx.new_ops import eval

        eval(self)
        data_obj = self._tensor.data
        raw: Scalar | list[Scalar | NestedSequence] = (
            data_obj.tolist() if isinstance(data_obj, ArrayProtocol) else self._tensor.item()
        )
        if isinstance(raw, list):
            return raw
        if self.dtype == DType.bool_:
            return bool(raw)
        if self.dtype.value.startswith(("int", "uint")):
            return int(raw.real)
        if self.dtype.value.startswith("complex"):
            return complex(raw)
        return float(raw.real)

    def item(self) -> Scalar:
        """Extract the Python scalar from a single-element array.

        Returns:
            Python `bool`, `int`, `float`, or `complex` scalar.

        Raises:
            ValueError: If `self.size != 1`.

        """
        if self.size != 1:
            raise ValueError("can only convert an array of size 1 to a Python scalar")
        raw_val: Scalar | NestedSequence = self.tolist()
        while isinstance(raw_val, Sequence):
            raw_val = raw_val[0]
        if self.dtype == DType.bool_:
            return bool(raw_val)
        if self.dtype.value.startswith(("int", "uint")):
            return int(raw_val.real)
        if self.dtype.value.startswith("complex"):
            return complex(raw_val)
        return float(raw_val.real)

    def __int__(self) -> int:
        """Convert a single-element array to a Python `int`.

        Returns:
            Integer scalar value.

        """
        return int(self.item().real)

    def __float__(self) -> float:
        """Convert a single-element array to a Python `float`.

        Returns:
            Float scalar value.

        """
        return float(self.item().real)

    def __bool__(self) -> bool:
        """Convert a single-element array to a Python `bool`.

        Returns:
            Boolean truth value.

        Raises:
            ValueError: If the array contains more than one element.

        """
        if self.size != 1:
            raise ValueError("The truth value of an array with more than one element is ambiguous.")
        return bool(self.item())

    def __len__(self) -> int:
        """Return the size of the first dimension.

        Returns:
            Length of dimension 0.

        Raises:
            TypeError: If the array is 0-dimensional.

        """
        if self.ndim == 0:
            raise TypeError("len() of unsized object")
        return self.shape[0]

    def _check_large_index(self, idx: IndexLike | list[int]) -> None:
        """Validate that integer or slice indices fit within 32-bit signed bounds.

        Args:
            idx: Index expression to validate.

        Raises:
            ValueError: If an index exceeds `[-2**31 + 1, 2**31 - 1]`.

        """
        limit: int = 2**31
        if isinstance(idx, slice):
            for bound in (idx.start, idx.stop, idx.step):
                if isinstance(bound, int) and (bound >= limit or bound <= -limit):
                    raise ValueError("Large index")
        elif isinstance(idx, int) and not isinstance(idx, bool):
            if idx >= limit or idx <= -limit:
                raise ValueError("Large index")
        elif isinstance(idx, tuple):
            for sub in idx:
                self._check_large_index(sub)

    def __getitem__(self, idx: IndexLike | list[int]) -> array:
        """Slice or index the array.

        Args:
            idx: Index specification (`int`, `slice`, `array`, `None`, `...`, or tuple).

        Returns:
            Indexed `array`.

        """
        self._check_large_index(idx)
        if isinstance(idx, list):
            idx_unwrapped = array(idx)._tensor
        elif isinstance(idx, array):
            idx_unwrapped = idx._tensor
        elif isinstance(idx, tuple):
            idx_unwrapped = tuple(
                i._tensor
                if isinstance(i, array)
                else (array(i)._tensor if isinstance(i, list) else i)
                for i in idx
            )
        else:
            idx_unwrapped = idx

        return array(self._tensor[idx_unwrapped], dtype=self.dtype)

    def __setitem__(self, idx: IndexLike | list[int], value: ArrayInput) -> None:
        """Update elements at `idx` in-place.

        Args:
            idx: Index specification.
            value: Replacement value or array.

        Raises:
            ValueError: If indexing a scalar with invalid indices or using raw `bool` index.

        """
        self._check_large_index(idx)
        if isinstance(idx, bool) or (
            isinstance(idx, tuple) and any(isinstance(i, bool) for i in idx)
        ):
            raise ValueError("Cannot index mlx array using the given type")

        if self.ndim == 0:
            if idx is None or (isinstance(idx, tuple) and all(i is None for i in idx)):
                t_val, _ = _to_tensor(value, dtype=self.dtype)
                self._tensor = t_val
                return
            raise ValueError("too many indices for array")

        val_tensor, _ = _to_tensor(value, dtype=self.dtype)
        if isinstance(idx, list):
            self._tensor[array(idx)._tensor] = val_tensor
        elif isinstance(idx, tuple):
            unwrapped_tuple = tuple(i._tensor if isinstance(i, array) else i for i in idx)
            self._tensor[unwrapped_tuple] = val_tensor
        else:
            idx_target = idx._tensor if isinstance(idx, array) else idx
            self._tensor[idx_target] = val_tensor

    def __iter__(self) -> ArrayIterator:
        """Return an `ArrayIterator` over the first dimension of the array.

        Returns:
            `ArrayIterator` instance.

        """
        from zero_mlx.array_iterator import ArrayIterator

        return ArrayIterator(self)

    def copy(self) -> array:
        """Return a copy of the array.

        Returns:
            Copied `array`.

        """
        return array(self.tolist(), dtype=self.dtype)

    def __copy__(self) -> array:
        """Support shallow copy via `copy.copy()`.

        Returns:
            Copied `array`.

        """
        return self.copy()

    def __deepcopy__(self, memo: dict[int, array] | None = None) -> array:
        """Support deep copy via `copy.deepcopy()`.

        Args:
            memo: Optional deepcopy memoization dictionary.

        Returns:
            Deep-copied `array`.

        """
        _ = memo
        return self.copy()

    @property
    def shape(self) -> tuple[int, ...]:
        """Return the dimensions of the array as a tuple of ints.

        Returns:
            Shape tuple.

        """
        return tuple(int(d) for d in self._tensor.shape)

    @property
    def size(self) -> int:
        """Return the total number of elements in the array.

        Returns:
            Total element count.

        """
        return math.prod(self.shape) if self.shape else 1

    @property
    def ndim(self) -> int:
        """Return the number of dimensions of the array.

        Returns:
            Number of dimensions.

        """
        return len(self.shape)

    @property
    def dtype(self) -> DType:
        """Return the MLX `DType` of the array.

        Returns:
            The array's `DType`.

        """
        return self._original_dtype

    @property
    def itemsize(self) -> int:
        """Return the size of each array element in bytes.

        Returns:
            Element size in bytes.

        """
        return self.dtype.size

    @property
    def nbytes(self) -> int:
        """Return the total number of bytes consumed by the array's elements.

        Returns:
            Total byte size (`self.size * self.itemsize`).

        """
        return self.size * self.itemsize

    @property
    def data(self) -> memoryview:
        """Return a pure-Python `memoryview` of the evaluated underlying data buffer.

        Returns:
            Read-write `memoryview` over the packed bytes of the array.

        """
        from zero_mlx.new_ops import eval

        eval(self)
        flat_arr: array = self.reshape(-1) if self.ndim > 0 else self
        raw_vals: Scalar | list[Scalar | NestedSequence] = flat_arr.tolist()
        items: list[Scalar | NestedSequence] = (
            raw_vals if isinstance(raw_vals, list) else [raw_vals]
        )
        buf = bytearray()
        dt = self.dtype
        for item in items:
            sc: Scalar = item if isinstance(item, (bool, int, float, complex)) else 0.0
            if dt == DType.bool_:
                buf.extend(struct.pack("<?", bool(sc)))
            elif dt == DType.int8:
                buf.extend(struct.pack("<b", int(sc.real)))
            elif dt == DType.uint8:
                buf.extend(struct.pack("<B", int(sc.real) & 0xFF))
            elif dt == DType.int16:
                buf.extend(struct.pack("<h", int(sc.real)))
            elif dt == DType.uint16:
                buf.extend(struct.pack("<H", int(sc.real) & 0xFFFF))
            elif dt == DType.int32:
                buf.extend(struct.pack("<i", int(sc.real)))
            elif dt == DType.uint32:
                buf.extend(struct.pack("<I", int(sc.real) & 0xFFFFFFFF))
            elif dt == DType.int64:
                buf.extend(struct.pack("<q", int(sc.real)))
            elif dt == DType.uint64:
                buf.extend(struct.pack("<Q", int(sc.real) & 0xFFFFFFFFFFFFFFFF))
            elif dt in (DType.float16, DType.bfloat16):
                buf.extend(struct.pack("<e", float(sc.real)))
            elif dt == DType.float64:
                buf.extend(struct.pack("<d", float(sc.real)))
            elif dt == DType.complex64:
                c = complex(sc)
                buf.extend(struct.pack("<ff", float(c.real), float(c.imag)))
            elif dt == DType.complex128:
                c = complex(sc)
                buf.extend(struct.pack("<dd", float(c.real), float(c.imag)))
            else:
                buf.extend(struct.pack("<f", float(sc.real)))
        return memoryview(buf)

    def __array__(
        self, dtype: DTypeLike = None, copy: bool | None = None
    ) -> Scalar | list[Scalar | NestedSequence]:
        """Support array conversion protocol without importing third-party libraries.

        Args:
            dtype: Optional target dtype override.
            copy: Optional boolean copy flag.

        Returns:
            Pure-Python scalar or nested list representation.

        """
        _ = copy
        from zero_mlx.new_ops import eval

        eval(self)
        target = self.astype(dtype) if dtype is not None else self
        return target.tolist()

    @property
    def T(self) -> array:
        """Return the transposed array (reversing all axes).

        Returns:
            Transposed `array`.

        """
        return self.transpose()

    @property
    def at(self) -> ArrayAt:
        """Return an `ArrayAt` helper for functional indexed updates (`.at[idx].add(...)`).

        Returns:
            `ArrayAt` proxy bound to `self`.

        """
        from zero_mlx.at_mocker import ArrayAt

        return ArrayAt(self)

    @property
    def real(self) -> array:
        """Return the real part of the array.

        Returns:
            Real-part `array`.

        """
        out_dt = DType.float32 if self.dtype in (DType.complex64, DType.complex128) else self.dtype
        return array(sops.real(self._tensor), dtype=out_dt)

    @property
    def imag(self) -> array:
        """Return the imaginary part of the array.

        Returns:
            Imaginary-part `array`.

        """
        if self.dtype in (DType.complex64, DType.complex128):
            return array(sops.imag(self._tensor), dtype=DType.float32)
        import zero_mlx.ops as _ops

        return _ops.zeros_like(self)

    def view(self, dtype: DTypeLike, *, stream: StreamOrDevice = None) -> array:
        """Return a view of the array with a new `dtype`.

        Args:
            dtype: Target `DTypeLike`.
            stream: Optional stream or device.

        Returns:
            `array` viewed with the target `dtype`.

        """
        _ = stream
        resolved = resolve_dtype(dtype, default=self.dtype)
        return array(self.tolist(), dtype=resolved)

    def abs(self, *, stream: StreamOrDevice = None) -> array:
        """Return the element-wise absolute value of the array.

        Args:
            stream: Optional stream or device.

        Returns:
            Absolute value `array`.

        """
        _ = stream
        return abs(self)

    def all(
        self,
        axis: AxesLike = None,
        keepdims: bool = False,
        *,
        stream: StreamOrDevice = None,
    ) -> array:
        """Evaluate whether all elements along `axis` are truthy.

        Args:
            axis: Axis or axes to reduce over.
            keepdims: Whether to retain reduced dimensions of size 1.
            stream: Optional stream or device.

        Returns:
            Boolean reduction `array`.

        """
        import zero_mlx.ops as _ops

        return _ops.all(self, axis=axis, keepdims=keepdims, stream=stream)

    def any(
        self,
        axis: AxesLike = None,
        keepdims: bool = False,
        *,
        stream: StreamOrDevice = None,
    ) -> array:
        """Evaluate whether any element along `axis` is truthy.

        Args:
            axis: Axis or axes to reduce over.
            keepdims: Whether to retain reduced dimensions of size 1.
            stream: Optional stream or device.

        Returns:
            Boolean reduction `array`.

        """
        import zero_mlx.ops as _ops

        return _ops.any(self, axis=axis, keepdims=keepdims, stream=stream)

    def argmax(
        self,
        axis: int | None = None,
        keepdims: bool = False,
        *,
        stream: StreamOrDevice = None,
    ) -> array:
        """Return the indices of the maximum values along `axis`.

        Args:
            axis: Axis along which to find the maximum indices.
            keepdims: Whether to keep the reduced dimension.
            stream: Optional stream or device.

        Returns:
            Index `array`.

        """
        import zero_mlx.ops as _ops

        return _ops.argmax(self, axis=axis, keepdims=keepdims, stream=stream)

    def argmin(
        self,
        axis: int | None = None,
        keepdims: bool = False,
        *,
        stream: StreamOrDevice = None,
    ) -> array:
        """Return the indices of the minimum values along `axis`.

        Args:
            axis: Axis along which to find the minimum indices.
            keepdims: Whether to keep the reduced dimension.
            stream: Optional stream or device.

        Returns:
            Index `array`.

        """
        import zero_mlx.ops as _ops

        return _ops.argmin(self, axis=axis, keepdims=keepdims, stream=stream)

    def conj(self, *, stream: StreamOrDevice = None) -> array:
        """Return the complex conjugate of the array.

        Args:
            stream: Optional stream or device.

        Returns:
            Complex-conjugated `array`.

        """
        import zero_mlx.ops as _ops

        return _ops.conjugate(self, stream=stream)

    def cummax(
        self,
        axis: int | None = None,
        reverse: bool = False,
        inclusive: bool = True,
        *,
        stream: StreamOrDevice = None,
    ) -> array:
        """Return the cumulative maximum of elements along `axis`.

        Args:
            axis: Axis along which to compute the cumulative maximum.
            reverse: Whether to accumulate in reverse order.
            inclusive: Whether the i-th output includes the i-th input.
            stream: Optional stream or device.

        Returns:
            Cumulative maximum `array`.

        """
        import zero_mlx.ops as _ops

        return _ops.cummax(self, axis=axis, reverse=reverse, inclusive=inclusive, stream=stream)

    def cummin(
        self,
        axis: int | None = None,
        reverse: bool = False,
        inclusive: bool = True,
        *,
        stream: StreamOrDevice = None,
    ) -> array:
        """Return the cumulative minimum of elements along `axis`.

        Args:
            axis: Axis along which to compute the cumulative minimum.
            reverse: Whether to accumulate in reverse order.
            inclusive: Whether the i-th output includes the i-th input.
            stream: Optional stream or device.

        Returns:
            Cumulative minimum `array`.

        """
        import zero_mlx.ops as _ops

        return _ops.cummin(self, axis=axis, reverse=reverse, inclusive=inclusive, stream=stream)

    def cumprod(
        self,
        axis: int | None = None,
        reverse: bool = False,
        inclusive: bool = True,
        *,
        stream: StreamOrDevice = None,
    ) -> array:
        """Return the cumulative product of elements along `axis`.

        Args:
            axis: Axis along which to compute the cumulative product.
            reverse: Whether to accumulate in reverse order.
            inclusive: Whether the i-th output includes the i-th input.
            stream: Optional stream or device.

        Returns:
            Cumulative product `array`.

        """
        import zero_mlx.ops as _ops

        return _ops.cumprod(self, axis=axis, reverse=reverse, inclusive=inclusive, stream=stream)

    def cumsum(
        self,
        axis: int | None = None,
        reverse: bool = False,
        inclusive: bool = True,
        *,
        stream: StreamOrDevice = None,
    ) -> array:
        """Return the cumulative sum of elements along `axis`.

        Args:
            axis: Axis along which to compute the cumulative sum.
            reverse: Whether to accumulate in reverse order.
            inclusive: Whether the i-th output includes the i-th input.
            stream: Optional stream or device.

        Returns:
            Cumulative sum `array`.

        """
        import zero_mlx.ops as _ops

        return _ops.cumsum(self, axis=axis, reverse=reverse, inclusive=inclusive, stream=stream)

    def diagonal(
        self,
        offset: int = 0,
        axis1: int = 0,
        axis2: int = 1,
        stream: StreamOrDevice = None,
    ) -> array:
        """Return specified diagonals of the array.

        Args:
            offset: Offset of the diagonal from the main diagonal.
            axis1: First axis of the 2D sub-arrays from which diagonals are taken.
            axis2: Second axis of the 2D sub-arrays from which diagonals are taken.
            stream: Optional stream or device.

        Returns:
            Diagonal `array`.

        """
        import zero_mlx.ops as _ops

        return _ops.diagonal(self, offset=offset, axis1=axis1, axis2=axis2, stream=stream)

    def exp(self, *, stream: StreamOrDevice = None) -> array:
        """Return the element-wise exponential of the array.

        Args:
            stream: Optional stream or device.

        Returns:
            Exponential `array`.

        """
        import zero_mlx.ops as _ops

        return _ops.exp(self, stream=stream)

    def flatten(
        self,
        start_axis: int = 0,
        end_axis: int = -1,
        *,
        stream: StreamOrDevice = None,
    ) -> array:
        """Flatten dimensions between `start_axis` and `end_axis`.

        Args:
            start_axis: First dimension to flatten.
            end_axis: Last dimension to flatten.
            stream: Optional stream or device.

        Returns:
            Flattened `array`.

        """
        import zero_mlx.ops as _ops

        return _ops.flatten(self, start_axis=start_axis, end_axis=end_axis, stream=stream)

    def log(self, *, stream: StreamOrDevice = None) -> array:
        """Return the element-wise natural logarithm of the array.

        Args:
            stream: Optional stream or device.

        Returns:
            Natural logarithm `array`.

        """
        import zero_mlx.ops as _ops

        return _ops.log(self, stream=stream)

    def log10(self, *, stream: StreamOrDevice = None) -> array:
        """Return the element-wise base-10 logarithm of the array.

        Args:
            stream: Optional stream or device.

        Returns:
            Base-10 logarithm `array`.

        """
        import zero_mlx.ops as _ops

        return _ops.log10(self, stream=stream)

    def log1p(self, *, stream: StreamOrDevice = None) -> array:
        """Return the element-wise natural logarithm of one plus the array.

        Args:
            stream: Optional stream or device.

        Returns:
            `log(1 + self)` `array`.

        """
        import zero_mlx.ops as _ops

        return _ops.log1p(self, stream=stream)

    def log2(self, *, stream: StreamOrDevice = None) -> array:
        """Return the element-wise base-2 logarithm of the array.

        Args:
            stream: Optional stream or device.

        Returns:
            Base-2 logarithm `array`.

        """
        import zero_mlx.ops as _ops

        return _ops.log2(self, stream=stream)

    def logcumsumexp(
        self,
        axis: int | None = None,
        reverse: bool = False,
        inclusive: bool = True,
        *,
        stream: StreamOrDevice = None,
    ) -> array:
        """Return the cumulative log-sum-exp along `axis`.

        Args:
            axis: Axis along which to compute the cumulative logsumexp.
            reverse: Whether to accumulate in reverse order.
            inclusive: Whether the i-th output includes the i-th input.
            stream: Optional stream or device.

        Returns:
            Cumulative logsumexp `array`.

        """
        import zero_mlx.ops as _ops

        return _ops.logcumsumexp(
            self, axis=axis, reverse=reverse, inclusive=inclusive, stream=stream
        )

    def logsumexp(
        self,
        axis: AxesLike = None,
        keepdims: bool = False,
        *,
        stream: StreamOrDevice = None,
    ) -> array:
        """Return the log-sum-exp reduction along `axis`.

        Args:
            axis: Axis or axes to reduce over.
            keepdims: Whether to retain reduced dimensions.
            stream: Optional stream or device.

        Returns:
            Reduced `array`.

        """
        import zero_mlx.ops as _ops

        return _ops.logsumexp(self, axis=axis, keepdims=keepdims, stream=stream)

    def max(
        self,
        axis: AxesLike = None,
        keepdims: bool = False,
        *,
        stream: StreamOrDevice = None,
    ) -> array:
        """Return the maximum along `axis`.

        Args:
            axis: Axis or axes to reduce over.
            keepdims: Whether to retain reduced dimensions.
            stream: Optional stream or device.

        Returns:
            Maximum `array`.

        """
        import zero_mlx.ops as _ops

        return _ops.max(self, axis=axis, keepdims=keepdims, stream=stream)

    def mean(
        self,
        axis: AxesLike = None,
        keepdims: bool = False,
        *,
        stream: StreamOrDevice = None,
    ) -> array:
        """Return the arithmetic mean along `axis`.

        Args:
            axis: Axis or axes to reduce over.
            keepdims: Whether to retain reduced dimensions.
            stream: Optional stream or device.

        Returns:
            Mean `array`.

        """
        import zero_mlx.ops as _ops

        return _ops.mean(self, axis=axis, keepdims=keepdims, stream=stream)

    def min(
        self,
        axis: AxesLike = None,
        keepdims: bool = False,
        *,
        stream: StreamOrDevice = None,
    ) -> array:
        """Return the minimum along `axis`.

        Args:
            axis: Axis or axes to reduce over.
            keepdims: Whether to retain reduced dimensions.
            stream: Optional stream or device.

        Returns:
            Minimum `array`.

        """
        import zero_mlx.ops as _ops

        return _ops.min(self, axis=axis, keepdims=keepdims, stream=stream)

    def moveaxis(
        self,
        source: int,
        destination: int,
        *,
        stream: StreamOrDevice = None,
    ) -> array:
        """Move an axis of the array to a new position.

        Args:
            source: Original index of the axis.
            destination: Target index of the axis.
            stream: Optional stream or device.

        Returns:
            Reordered `array`.

        """
        import zero_mlx.ops as _ops

        return _ops.moveaxis(self, source=source, destination=destination, stream=stream)

    def prod(
        self,
        axis: AxesLike = None,
        keepdims: bool = False,
        *,
        stream: StreamOrDevice = None,
    ) -> array:
        """Return the product of elements along `axis`.

        Args:
            axis: Axis or axes to reduce over.
            keepdims: Whether to retain reduced dimensions.
            stream: Optional stream or device.

        Returns:
            Product `array`.

        """
        import zero_mlx.ops as _ops

        return _ops.prod(self, axis=axis, keepdims=keepdims, stream=stream)

    def reciprocal(self, *, stream: StreamOrDevice = None) -> array:
        """Return the element-wise reciprocal `1 / self`.

        Args:
            stream: Optional stream or device.

        Returns:
            Reciprocal `array`.

        """
        import zero_mlx.ops as _ops

        return _ops.reciprocal(self, stream=stream)

    def round(self, decimals: int = 0, *, stream: StreamOrDevice = None) -> array:
        """Round elements to the given number of decimals.

        Args:
            decimals: Number of decimal places to round to.
            stream: Optional stream or device.

        Returns:
            Rounded `array`.

        """
        import zero_mlx.ops as _ops

        return _ops.round(self, decimals=decimals, stream=stream)

    def sin(self, *, stream: StreamOrDevice = None) -> array:
        """Return the element-wise sine of the array.

        Args:
            stream: Optional stream or device.

        Returns:
            Sine `array`.

        """
        import zero_mlx.ops as _ops

        return _ops.sin(self, stream=stream)

    def cos(self, *, stream: StreamOrDevice = None) -> array:
        """Return the element-wise cosine of the array.

        Args:
            stream: Optional stream or device.

        Returns:
            Cosine `array`.

        """
        import zero_mlx.ops as _ops

        return _ops.cos(self, stream=stream)

    def sqrt(self, *, stream: StreamOrDevice = None) -> array:
        """Return the element-wise square root of the array.

        Args:
            stream: Optional stream or device.

        Returns:
            Square root `array`.

        """
        import zero_mlx.ops as _ops

        return _ops.sqrt(self, stream=stream)

    def rsqrt(self, *, stream: StreamOrDevice = None) -> array:
        """Return the element-wise reciprocal square root of the array.

        Args:
            stream: Optional stream or device.

        Returns:
            Reciprocal square root `array`.

        """
        import zero_mlx.ops as _ops

        return _ops.rsqrt(self, stream=stream)

    def square(self, *, stream: StreamOrDevice = None) -> array:
        """Return the element-wise square of the array.

        Args:
            stream: Optional stream or device.

        Returns:
            Squared `array`.

        """
        import zero_mlx.ops as _ops

        return _ops.square(self, stream=stream)

    def std(
        self,
        axis: AxesLike = None,
        keepdims: bool = False,
        ddof: int = 0,
        *,
        stream: StreamOrDevice = None,
    ) -> array:
        """Return the standard deviation along `axis`.

        Args:
            axis: Axis or axes to reduce over.
            keepdims: Whether to retain reduced dimensions.
            ddof: Delta degrees of freedom.
            stream: Optional stream or device.

        Returns:
            Standard deviation `array`.

        """
        import zero_mlx.ops as _ops

        return _ops.std(self, axis=axis, keepdims=keepdims, ddof=ddof, stream=stream)

    def sum(
        self,
        axis: AxesLike = None,
        keepdims: bool = False,
        *,
        stream: StreamOrDevice = None,
    ) -> array:
        """Return the sum of elements along `axis`.

        Args:
            axis: Axis or axes to reduce over.
            keepdims: Whether to retain reduced dimensions.
            stream: Optional stream or device.

        Returns:
            Sum `array`.

        """
        import zero_mlx.ops as _ops

        return _ops.sum(self, axis=axis, keepdims=keepdims, stream=stream)

    def swapaxes(
        self,
        axis1: int,
        axis2: int,
        *,
        stream: StreamOrDevice = None,
    ) -> array:
        """Interchange two axes of the array.

        Args:
            axis1: First axis index.
            axis2: Second axis index.
            stream: Optional stream or device.

        Returns:
            `array` with `axis1` and `axis2` swapped.

        """
        import zero_mlx.ops as _ops

        return _ops.swapaxes(self, axis1=axis1, axis2=axis2, stream=stream)

    def transpose(
        self,
        *axes: int | Sequence[int],
        stream: StreamOrDevice = None,
    ) -> array:
        """Permute the dimensions of the array.

        Args:
            *axes: Optional permutation of axes as integers or a sequence of integers.
            stream: Optional stream or device.

        Returns:
            Transposed `array`.

        """
        import zero_mlx.ops as _ops

        if len(axes) == 0:
            return _ops.transpose(self, axes=None, stream=stream)
        if len(axes) == 1 and isinstance(axes[0], Sequence):
            return _ops.transpose(self, axes=tuple(int(a) for a in axes[0]), stream=stream)
        return _ops.transpose(
            self,
            axes=tuple(int(a) for a in axes if isinstance(a, int)),
            stream=stream,
        )

    def var(
        self,
        axis: AxesLike = None,
        keepdims: bool = False,
        ddof: int = 0,
        *,
        stream: StreamOrDevice = None,
    ) -> array:
        """Return the variance along `axis`.

        Args:
            axis: Axis or axes to reduce over.
            keepdims: Whether to retain reduced dimensions.
            ddof: Delta degrees of freedom.
            stream: Optional stream or device.

        Returns:
            Variance `array`.

        """
        import zero_mlx.ops as _ops

        return _ops.var(self, axis=axis, keepdims=keepdims, ddof=ddof, stream=stream)

    def split(
        self,
        indices_or_sections: int | Sequence[int],
        axis: int = 0,
        *,
        stream: StreamOrDevice = None,
    ) -> list[array]:
        """Split the array into multiple sub-arrays along `axis`.

        Args:
            indices_or_sections: Number of equal sections or split indices.
            axis: Axis along which to split.
            stream: Optional stream or device.

        Returns:
            List of split `array` objects.

        """
        import zero_mlx.ops as _ops

        return _ops.split(self, indices_or_sections, axis=axis, stream=stream)
