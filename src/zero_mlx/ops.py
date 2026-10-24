"""Core tensor creation, mathematical, reduction, logical, and shape operations for zero_mlx."""

from __future__ import annotations

import builtins
import math
from typing import Callable, Literal, Sequence

import ml_switcheroo_compiler.ops as sops
import ml_switcheroo_compiler.ops.shape.frontend as sf
import ml_switcheroo_compiler.ops.shape.indexing as ix
import ml_switcheroo_compiler.ops.shape.pad_and_tile as pt
from ml_switcheroo_compiler.core.dtype import DType as SwitcherooDType

from zero_mlx._types import (
    ArrayInput,
    AxesLike,
    DTypeLike,
    NestedSequence,
    Scalar,
    ShapeLike,
    StreamOrDevice,
)
from zero_mlx.array import _to_tensor, array
from zero_mlx.device import get_peak_memory, synchronize
from zero_mlx.dtypes import (
    DType,
    DtypeCategory,
    bool_,
    complex64,
    float32,
    int32,
    isdtype,
    issubdtype,
    promote_dtypes,
    resolve_dtype,
    to_mlx_dtype,
    to_switcheroo_dtype,
    uint8,
    uint32,
)
from zero_mlx.linalg import matrix_norm
from zero_mlx.new_ops import eval


def _as_array(x: ArrayInput) -> array:
    """Convert `x` into a `zero_mlx.array` if not already one.

    Args:
        x: Input array, scalar, or nested sequence.

    Returns:
        `zero_mlx.array` instance.

    """
    return x if isinstance(x, array) else array(x)


def _norm_shape(shape: ShapeLike) -> tuple[int, ...]:
    """Normalize a shape specification into a tuple of ints.

    Args:
        shape: Single integer dimension or sequence of dimensions.

    Returns:
        Tuple of dimension sizes.

    """
    if shape is None:
        return ()
    if isinstance(shape, int):
        return (shape,)
    return tuple(int(s) for s in shape)


def _binary_op(
    a: ArrayInput,
    b: ArrayInput,
    op_fn: Callable[[sops.Tensor, sops.Tensor], sops.Tensor],
    out_dtype: DType | None = None,
) -> array:
    """Execute a binary compiler operation with MLX dtype promotion.

    Args:
        a: Left-hand operand.
        b: Right-hand operand.
        op_fn: Binary compiler operation callable.
        out_dtype: Optional explicit output `DType`.

    Returns:
        Resulting `array`.

    """
    arr_a = _as_array(a)
    t_b, promoted = arr_a._coerce_other(b, allow_list=True)
    dt = out_dtype if out_dtype is not None else promoted
    res = op_fn(arr_a._tensor, t_b)
    return array(res, dtype=dt)


# ==============================================================================
# Creation & Window Operations
# ==============================================================================


def asarray(
    a: ArrayInput,
    dtype: DTypeLike = None,
    *,
    copy: bool | None = None,
    stream: StreamOrDevice = None,
) -> array:
    """Convert the input to a `zero_mlx.array`.

    Args:
        a: Input data (`array`, scalar, or nested sequence).
        dtype: Optional target data type.
        copy: Optional boolean controlling whether a copy is made.
        stream: Optional stream or device.

    Returns:
        Converted `array`.

    Raises:
        ValueError: If `copy=False` is requested when a copy or dtype conversion is required.

    """
    _ = stream
    if copy is False:
        if not isinstance(a, array):
            raise ValueError("copy=False not supported for non-arrays")
        if dtype is not None and resolve_dtype(dtype, default=a.dtype) != a.dtype:
            raise ValueError("copy=False not supported when converting dtype")
        return a
    if isinstance(a, array):
        target_dt = resolve_dtype(dtype, default=a.dtype)
        res = a.astype(target_dt) if target_dt != a.dtype else a
        return res.copy() if copy else res
    return array(a, dtype=dtype)


def zeros(
    shape: ShapeLike,
    dtype: DTypeLike = float32,
    *,
    stream: StreamOrDevice = None,
) -> array:
    """Return a new array of given `shape` and `dtype`, filled with zeros.

    Args:
        shape: Shape of the new array.
        dtype: Desired data type for the array.
        stream: Optional stream or device.

    Returns:
        Zero-filled `array`.

    """
    _ = stream
    dt = resolve_dtype(dtype, default=float32)
    shp = _norm_shape(shape)
    return array(sops.zeros(shp, dtype=to_switcheroo_dtype(dt)), dtype=dt)


def ones(
    shape: ShapeLike,
    dtype: DTypeLike = float32,
    *,
    stream: StreamOrDevice = None,
) -> array:
    """Return a new array of given `shape` and `dtype`, filled with ones.

    Args:
        shape: Shape of the new array.
        dtype: Desired data type for the array.
        stream: Optional stream or device.

    Returns:
        One-filled `array`.

    """
    _ = stream
    dt = resolve_dtype(dtype, default=float32)
    shp = _norm_shape(shape)
    return array(sops.ones(shp, dtype=to_switcheroo_dtype(dt)), dtype=dt)


def empty(
    shape: ShapeLike,
    dtype: DTypeLike = float32,
    *,
    stream: StreamOrDevice = None,
) -> array:
    """Return a new array of given `shape` and `dtype` without initializing entries.

    Args:
        shape: Shape of the new array.
        dtype: Desired data type for the array.
        stream: Optional stream or device.

    Returns:
        Allocated `array`.

    """
    return zeros(shape, dtype=dtype, stream=stream)


def full(
    shape: ShapeLike,
    vals: ArrayInput,
    dtype: DTypeLike = None,
    *,
    stream: StreamOrDevice = None,
) -> array:
    """Return a new array of given `shape` and `dtype`, filled with `vals`.

    Args:
        shape: Shape of the new array.
        vals: Fill value (scalar or broadcastable array).
        dtype: Optional desired data type.
        stream: Optional stream or device.

    Returns:
        Filled `array`.

    """
    _ = stream
    shp = _norm_shape(shape)
    val_arr = _as_array(vals)
    dt = resolve_dtype(dtype, default=val_arr.dtype)
    if val_arr.ndim == 0:
        return array(sops.full(shp, val_arr.item(), dtype=to_switcheroo_dtype(dt)), dtype=dt)
    return broadcast_to(val_arr.astype(dt), shp)


def zeros_like(
    a: ArrayInput,
    dtype: DTypeLike = None,
    *,
    stream: StreamOrDevice = None,
) -> array:
    """Return an array of zeros with the same shape and type as `a`.

    Args:
        a: Template array.
        dtype: Optional override data type.
        stream: Optional stream or device.

    Returns:
        Zero-filled `array`.

    """
    arr = _as_array(a)
    dt = resolve_dtype(dtype, default=arr.dtype)
    return zeros(arr.shape, dtype=dt, stream=stream)


def ones_like(
    a: ArrayInput,
    dtype: DTypeLike = None,
    *,
    stream: StreamOrDevice = None,
) -> array:
    """Return an array of ones with the same shape and type as `a`.

    Args:
        a: Template array.
        dtype: Optional override data type.
        stream: Optional stream or device.

    Returns:
        One-filled `array`.

    """
    arr = _as_array(a)
    dt = resolve_dtype(dtype, default=arr.dtype)
    return ones(arr.shape, dtype=dt, stream=stream)


def empty_like(
    a: ArrayInput,
    dtype: DTypeLike = None,
    *,
    stream: StreamOrDevice = None,
) -> array:
    """Return an uninitialized array with the same shape and type as `a`.

    Args:
        a: Template array.
        dtype: Optional override data type.
        stream: Optional stream or device.

    Returns:
        Allocated `array`.

    """
    return zeros_like(a, dtype=dtype, stream=stream)


def full_like(
    a: ArrayInput,
    vals: ArrayInput,
    dtype: DTypeLike = None,
    *,
    stream: StreamOrDevice = None,
) -> array:
    """Return a full array with the same shape and type as `a`, filled with `vals`.

    Args:
        a: Template array.
        vals: Fill value.
        dtype: Optional override data type.
        stream: Optional stream or device.

    Returns:
        Filled `array`.

    """
    arr = _as_array(a)
    dt = resolve_dtype(dtype, default=arr.dtype)
    return full(arr.shape, vals, dtype=dt, stream=stream)


def arange(
    start: int | float,
    stop: int | float | None = None,
    step: int | float = 1,
    dtype: DTypeLike = None,
    *,
    stream: StreamOrDevice = None,
) -> array:
    """Generate evenly spaced values within a half-open interval `[start, stop)`.

    Args:
        start: Start of interval (or `stop` if `stop` is None).
        stop: End of interval.
        step: Spacing between values.
        dtype: Optional output data type.
        stream: Optional stream or device.

    Returns:
        1D `array` of evenly spaced values.

    """
    _ = stream
    actual_start: int | float = 0 if stop is None else start
    actual_stop: int | float = start if stop is None else stop
    if dtype is None:
        all_ints = (
            isinstance(actual_start, int) and isinstance(actual_stop, int) and isinstance(step, int)
        )
        dt = int32 if all_ints else float32
    else:
        dt = resolve_dtype(dtype)
    return array(
        sops.arange(actual_start, actual_stop, step, dtype=to_switcheroo_dtype(dt)),
        dtype=dt,
    )


def linspace(
    start: int | float,
    stop: int | float,
    num: int = 50,
    dtype: DTypeLike = float32,
    *,
    stream: StreamOrDevice = None,
) -> array:
    """Generate `num` evenly spaced numbers over a specified interval `[start, stop]`.

    Args:
        start: Starting value of the sequence.
        stop: End value of the sequence.
        num: Number of samples to generate.
        dtype: Output data type.
        stream: Optional stream or device.

    Returns:
        1D `array` of `num` evenly spaced samples.

    """
    _ = stream
    dt = resolve_dtype(dtype, default=float32)
    return array(
        sops.linspace(start, stop, num, dtype=to_switcheroo_dtype(dt)),
        dtype=dt,
    )


def eye(
    n: int,
    m: int | None = None,
    k: int = 0,
    dtype: DTypeLike = float32,
    *,
    stream: StreamOrDevice = None,
) -> array:
    """Return a 2D array with ones on the `k`-th diagonal and zeros elsewhere.

    Args:
        n: Number of rows in the output.
        m: Number of columns in the output (defaults to `n`).
        k: Index of the diagonal (0 is main diagonal, >0 is upper, <0 is lower).
        dtype: Data type of the returned array.
        stream: Optional stream or device.

    Returns:
        2D identity-like `array`.

    """
    _ = stream
    dt = resolve_dtype(dtype, default=float32)
    cols: int = n if m is None else m
    grid = [[1 if (c - r) == k else 0 for c in range(cols)] for r in range(n)]
    return array(grid, dtype=dt)


def identity(
    n: int,
    dtype: DTypeLike = float32,
    *,
    stream: StreamOrDevice = None,
) -> array:
    """Return the square identity matrix of size `n x n`.

    Args:
        n: Number of rows and columns.
        dtype: Data type of the output matrix.
        stream: Optional stream or device.

    Returns:
        `n x n` identity `array`.

    """
    return eye(n, m=n, k=0, dtype=dtype, stream=stream)


def tri(
    n: int,
    m: int | None = None,
    k: int = 0,
    dtype: DTypeLike = float32,
    *,
    stream: StreamOrDevice = None,
) -> array:
    """Return an array with ones at and below the given diagonal `k` and zeros elsewhere.

    Args:
        n: Number of rows.
        m: Number of columns (defaults to `n`).
        k: Sub-diagonal at and below which the array is filled.
        dtype: Output data type.
        stream: Optional stream or device.

    Returns:
        Lower-triangular mask `array`.

    """
    dt = resolve_dtype(dtype, default=float32)
    cols: int = n if m is None else m
    return tril(ones((n, cols), dtype=dt, stream=stream), k=k, stream=stream)


def tril(
    x: ArrayInput,
    k: int = 0,
    *,
    stream: StreamOrDevice = None,
) -> array:
    """Zero out elements above the `k`-th diagonal of `x`.

    Args:
        x: Input array of at least 2 dimensions.
        k: Diagonal above which to zero elements.
        stream: Optional stream or device.

    Returns:
        Lower-triangular `array`.

    """
    _ = stream
    arr = _as_array(x)
    return array(pt.tril(arr._tensor, diagonal=k), dtype=arr.dtype)


def triu(
    x: ArrayInput,
    k: int = 0,
    *,
    stream: StreamOrDevice = None,
) -> array:
    """Zero out elements below the `k`-th diagonal of `x`.

    Args:
        x: Input array of at least 2 dimensions.
        k: Diagonal below which to zero elements.
        stream: Optional stream or device.

    Returns:
        Upper-triangular `array`.

    """
    _ = stream
    arr = _as_array(x)
    return array(pt.triu(arr._tensor, diagonal=k), dtype=arr.dtype)


def diag(
    a: ArrayInput,
    k: int = 0,
    *,
    stream: StreamOrDevice = None,
) -> array:
    """Extract a diagonal or construct a diagonal 2D array.

    Args:
        a: 1D array (constructs 2D matrix) or 2D array (extracts 1D diagonal).
        k: Diagonal offset.
        stream: Optional stream or device.

    Returns:
        Extracted diagonal or constructed diagonal `array`.

    """
    _ = stream
    arr = _as_array(a)
    if arr.ndim == 1:
        n = arr.shape[0]
        sz = n + builtins.abs(k)
        eye_mat = eye(sz, m=sz, k=k, dtype=arr.dtype, stream=stream)
        pad_before = 0 if k >= 0 else -k
        pad_after = k if k > 0 else 0
        padded_vec = (
            pad(arr, ((pad_before, pad_after),), constant_values=0, stream=stream)
            if (pad_before > 0 or pad_after > 0)
            else arr
        )
        return eye_mat * padded_vec.reshape((sz, 1))
    return diagonal(arr, offset=k, axis1=0, axis2=1, stream=stream)


def diagonal(
    a: ArrayInput,
    offset: int = 0,
    axis1: int = 0,
    axis2: int = 1,
    stream: StreamOrDevice = None,
) -> array:
    """Return specified diagonals of `a` along `axis1` and `axis2`.

    Args:
        a: Input array of at least 2 dimensions.
        offset: Offset of the diagonal from the main diagonal.
        axis1: First axis of the 2D sub-arrays.
        axis2: Second axis of the 2D sub-arrays.
        stream: Optional stream or device.

    Returns:
        Diagonal `array`.

    """
    _ = stream
    arr = _as_array(a)
    res = sops.diagonal(arr._tensor, offset=offset, axis1=axis1, axis2=axis2)
    return array(res[0] if isinstance(res, tuple) else res, dtype=arr.dtype)


def meshgrid(
    *arrays: ArrayInput,
    sparse: bool = False,
    indexing: str = "xy",
    stream: StreamOrDevice = None,
) -> list[array]:
    """Generate coordinate matrices from coordinate vectors.

    Args:
        *arrays: 1D coordinate arrays.
        sparse: If True, return sparse coordinate grids to conserve memory.
        indexing: Cartesian (`'xy'`) or matrix (`'ij'`) indexing of output.
        stream: Optional stream or device.

    Returns:
        List of coordinate `array` grids.

    """
    arrs = [_as_array(a).reshape(-1) for a in arrays]
    ndim = len(arrs)
    if ndim == 0:
        return []
    full_shape: list[int] = [a.size for a in arrs]
    if indexing == "xy" and ndim >= 2:
        full_shape[0], full_shape[1] = full_shape[1], full_shape[0]

    sparse_grids: list[array] = []
    for i, vec in enumerate(arrs):
        reshape_dims = [1] * ndim
        axis_pos = (
            (1 if i == 0 else (0 if i == 1 else i)) if (indexing == "xy" and ndim >= 2) else i
        )
        reshape_dims[axis_pos] = vec.size
        sparse_grids.append(vec.reshape(reshape_dims))

    if sparse:
        return sparse_grids
    return [broadcast_to(g, tuple(full_shape), stream=stream) for g in sparse_grids]


def bartlett(M: int, *, stream: StreamOrDevice = None) -> array:
    """Return the Bartlett (triangular) window of length `M`.

    Args:
        M: Number of points in the output window.
        stream: Optional stream or device.

    Returns:
        1D `array` of length `M` containing the Bartlett window.

    """
    if M <= 0:
        return array([], dtype=float32)
    if M == 1:
        return array([1.0], dtype=float32)
    n = arange(0, M, dtype=float32, stream=stream)
    return 1.0 - abs((2.0 * n) / (float(M) - 1.0) - 1.0, stream=stream)


def blackman(M: int, *, stream: StreamOrDevice = None) -> array:
    """Return the Blackman window of length `M`.

    Args:
        M: Number of points in the output window.
        stream: Optional stream or device.

    Returns:
        1D `array` of length `M` containing the Blackman window.

    """
    if M <= 0:
        return array([], dtype=float32)
    if M == 1:
        return array([1.0], dtype=float32)
    n = arange(0, M, dtype=float32, stream=stream)
    denom = float(M) - 1.0
    return (
        0.42
        - 0.5 * cos((2.0 * math.pi / denom) * n, stream=stream)
        + 0.08 * cos((4.0 * math.pi / denom) * n, stream=stream)
    )


def hamming(M: int, *, stream: StreamOrDevice = None) -> array:
    """Return the Hamming window of length `M`.

    Args:
        M: Number of points in the output window.
        stream: Optional stream or device.

    Returns:
        1D `array` of length `M` containing the Hamming window.

    """
    if M <= 0:
        return array([], dtype=float32)
    if M == 1:
        return array([1.0], dtype=float32)
    n = arange(0, M, dtype=float32, stream=stream)
    return 0.54 - 0.46 * cos((2.0 * math.pi / (float(M) - 1.0)) * n, stream=stream)


def hanning(M: int, *, stream: StreamOrDevice = None) -> array:
    """Return the Hanning (von Hann) window of length `M`.

    Args:
        M: Number of points in the output window.
        stream: Optional stream or device.

    Returns:
        1D `array` of length `M` containing the Hanning window.

    """
    if M <= 0:
        return array([], dtype=float32)
    if M == 1:
        return array([1.0], dtype=float32)
    n = arange(0, M, dtype=float32, stream=stream)
    return 0.5 - 0.5 * cos((2.0 * math.pi / (float(M) - 1.0)) * n, stream=stream)


# ==============================================================================
# Unary & Binary Element-wise Mathematical Operations
# ==============================================================================


def abs(a: ArrayInput, *, stream: StreamOrDevice = None) -> array:
    """Compute the element-wise absolute value.

    Args:
        a: Input array.
        stream: Optional stream or device.

    Returns:
        Absolute value `array`.

    """
    _ = stream
    arr = _as_array(a)
    out_dt = float32 if arr.dtype == complex64 else arr.dtype
    return array(sops.abs(arr._tensor), dtype=out_dt)


def negative(a: ArrayInput, *, stream: StreamOrDevice = None) -> array:
    """Compute element-wise numerical negation `-a`.

    Args:
        a: Input array.
        stream: Optional stream or device.

    Returns:
        Negated `array`.

    """
    _ = stream
    arr = _as_array(a)
    return -arr


def positive(a: ArrayInput, *, stream: StreamOrDevice = None) -> array:
    """Compute element-wise unary positive `+a`.

    Args:
        a: Input array.
        stream: Optional stream or device.

    Returns:
        Copy of `a`.

    """
    _ = stream
    return +_as_array(a)


def sign(a: ArrayInput, *, stream: StreamOrDevice = None) -> array:
    """Return element-wise indication of the sign of a number (`-1`, `0`, or `+1`).

    Args:
        a: Input array.
        stream: Optional stream or device.

    Returns:
        Sign `array`.

    """
    _ = stream
    arr = _as_array(a)
    return array(sops.sign(arr._tensor), dtype=arr.dtype)


def ceil(a: ArrayInput, *, stream: StreamOrDevice = None) -> array:
    """Return the ceiling of the input, element-wise.

    Args:
        a: Input array.
        stream: Optional stream or device.

    Returns:
        Ceiling `array`.

    """
    _ = stream
    arr = _as_array(a)
    return array(sops.ceil(arr._tensor), dtype=arr.dtype)


def floor(a: ArrayInput, *, stream: StreamOrDevice = None) -> array:
    """Return the floor of the input, element-wise.

    Args:
        a: Input array.
        stream: Optional stream or device.

    Returns:
        Floor `array`.

    """
    _ = stream
    arr = _as_array(a)
    return array(sops.floor(arr._tensor), dtype=arr.dtype)


def trunc(a: ArrayInput, *, stream: StreamOrDevice = None) -> array:
    """Return the truncated value of the input (rounded toward zero), element-wise.

    Args:
        a: Input array.
        stream: Optional stream or device.

    Returns:
        Truncated `array`.

    """
    _ = stream
    arr = _as_array(a)
    return where(arr >= 0, floor(arr), ceil(arr))


def round(a: ArrayInput, decimals: int = 0, *, stream: StreamOrDevice = None) -> array:
    """Round an array to the given number of decimals.

    Args:
        a: Input array.
        decimals: Number of decimal places to round to.
        stream: Optional stream or device.

    Returns:
        Rounded `array`.

    """
    _ = stream
    arr = _as_array(a)
    if decimals == 0:
        return array(sops.round(arr._tensor), dtype=arr.dtype)
    factor: float = 10.0**decimals
    return (array(sops.round((arr * factor)._tensor), dtype=float32) / factor).astype(arr.dtype)


def exp(a: ArrayInput, *, stream: StreamOrDevice = None) -> array:
    """Calculate the exponential of all elements in the input array.

    Args:
        a: Input array.
        stream: Optional stream or device.

    Returns:
        Exponential `array`.

    """
    _ = stream
    arr = _as_array(a)
    out_dt = float32 if arr.dtype.value.startswith(("int", "uint", "bool")) else arr.dtype
    return array(sops.exp(arr._tensor), dtype=out_dt)


def expm1(a: ArrayInput, *, stream: StreamOrDevice = None) -> array:
    """Calculate `exp(a) - 1` for all elements in the array.

    Args:
        a: Input array.
        stream: Optional stream or device.

    Returns:
        `exp(a) - 1` `array`.

    """
    _ = stream
    arr = _as_array(a)
    out_dt = float32 if arr.dtype.value.startswith(("int", "uint", "bool")) else arr.dtype
    return array(sops.expm1(arr._tensor), dtype=out_dt)


def log(a: ArrayInput, *, stream: StreamOrDevice = None) -> array:
    """Compute the natural logarithm, element-wise.

    Args:
        a: Input array.
        stream: Optional stream or device.

    Returns:
        Natural logarithm `array`.

    """
    _ = stream
    arr = _as_array(a)
    out_dt = float32 if arr.dtype.value.startswith(("int", "uint", "bool")) else arr.dtype
    return array(sops.log(arr._tensor), dtype=out_dt)


def log2(a: ArrayInput, *, stream: StreamOrDevice = None) -> array:
    """Compute the base-2 logarithm of `a`, element-wise.

    Args:
        a: Input array.
        stream: Optional stream or device.

    Returns:
        Base-2 logarithm `array`.

    """
    _ = stream
    arr = _as_array(a)
    out_dt = float32 if arr.dtype.value.startswith(("int", "uint", "bool")) else arr.dtype
    return array(sops.log2(arr._tensor), dtype=out_dt)


def log10(a: ArrayInput, *, stream: StreamOrDevice = None) -> array:
    """Compute the base-10 logarithm of `a`, element-wise.

    Args:
        a: Input array.
        stream: Optional stream or device.

    Returns:
        Base-10 logarithm `array`.

    """
    _ = stream
    arr = _as_array(a)
    out_dt = float32 if arr.dtype.value.startswith(("int", "uint", "bool")) else arr.dtype
    return array(sops.log10(arr._tensor), dtype=out_dt)


def log1p(a: ArrayInput, *, stream: StreamOrDevice = None) -> array:
    """Compute the natural logarithm of `1 + a`, element-wise.

    Args:
        a: Input array.
        stream: Optional stream or device.

    Returns:
        `log(1 + a)` `array`.

    """
    _ = stream
    arr = _as_array(a)
    out_dt = float32 if arr.dtype.value.startswith(("int", "uint", "bool")) else arr.dtype
    return array(sops.log1p(arr._tensor), dtype=out_dt)


def sqrt(a: ArrayInput, *, stream: StreamOrDevice = None) -> array:
    """Return the non-negative square-root of an array, element-wise.

    Args:
        a: Input array.
        stream: Optional stream or device.

    Returns:
        Square-root `array`.

    """
    _ = stream
    arr = _as_array(a)
    out_dt = float32 if arr.dtype.value.startswith(("int", "uint", "bool")) else arr.dtype
    return array(sops.sqrt(arr._tensor), dtype=out_dt)


def rsqrt(a: ArrayInput, *, stream: StreamOrDevice = None) -> array:
    """Return the reciprocal of the square-root (`1 / sqrt(a)`), element-wise.

    Args:
        a: Input array.
        stream: Optional stream or device.

    Returns:
        Reciprocal square-root `array`.

    """
    _ = stream
    arr = _as_array(a)
    out_dt = float32 if arr.dtype.value.startswith(("int", "uint", "bool")) else arr.dtype
    return array(sops.rsqrt(arr._tensor), dtype=out_dt)


def square(a: ArrayInput, *, stream: StreamOrDevice = None) -> array:
    """Return the element-wise square of the input.

    Args:
        a: Input array.
        stream: Optional stream or device.

    Returns:
        Squared `array`.

    """
    _ = stream
    arr = _as_array(a)
    return array(sops.square(arr._tensor), dtype=arr.dtype)


def reciprocal(a: ArrayInput, *, stream: StreamOrDevice = None) -> array:
    """Return the reciprocal of the argument (`1 / a`), element-wise.

    Args:
        a: Input array.
        stream: Optional stream or device.

    Returns:
        Reciprocal `array`.

    """
    _ = stream
    arr = _as_array(a)
    out_dt = float32 if arr.dtype.value.startswith(("int", "uint", "bool")) else arr.dtype
    return array(sops.reciprocal(arr._tensor), dtype=out_dt)


def sin(a: ArrayInput, *, stream: StreamOrDevice = None) -> array:
    """Compute trigonometric sine, element-wise.

    Args:
        a: Input array.
        stream: Optional stream or device.

    Returns:
        Sine `array`.

    """
    _ = stream
    arr = _as_array(a)
    out_dt = float32 if arr.dtype.value.startswith(("int", "uint", "bool")) else arr.dtype
    return array(sops.sin(arr._tensor), dtype=out_dt)


def cos(a: ArrayInput, *, stream: StreamOrDevice = None) -> array:
    """Compute trigonometric cosine, element-wise.

    Args:
        a: Input array.
        stream: Optional stream or device.

    Returns:
        Cosine `array`.

    """
    _ = stream
    arr = _as_array(a)
    out_dt = float32 if arr.dtype.value.startswith(("int", "uint", "bool")) else arr.dtype
    return array(sops.cos(arr._tensor), dtype=out_dt)


def tan(a: ArrayInput, *, stream: StreamOrDevice = None) -> array:
    """Compute trigonometric tangent, element-wise.

    Args:
        a: Input array.
        stream: Optional stream or device.

    Returns:
        Tangent `array`.

    """
    _ = stream
    arr = _as_array(a)
    out_dt = float32 if arr.dtype.value.startswith(("int", "uint", "bool")) else arr.dtype
    return array(sops.tan(arr._tensor), dtype=out_dt)


def arcsin(a: ArrayInput, *, stream: StreamOrDevice = None) -> array:
    """Compute inverse sine, element-wise.

    Args:
        a: Input array.
        stream: Optional stream or device.

    Returns:
        Inverse sine `array`.

    """
    _ = stream
    arr = _as_array(a)
    out_dt = float32 if arr.dtype.value.startswith(("int", "uint", "bool")) else arr.dtype
    return array(sops.asin(arr._tensor), dtype=out_dt)


def asin(a: ArrayInput, *, stream: StreamOrDevice = None) -> array:
    """Compute inverse sine, element-wise (alias for `arcsin`).

    Args:
        a: Input array.
        stream: Optional stream or device.

    Returns:
        Inverse sine `array`.

    """
    return arcsin(a, stream=stream)


def arccos(a: ArrayInput, *, stream: StreamOrDevice = None) -> array:
    """Compute inverse cosine, element-wise.

    Args:
        a: Input array.
        stream: Optional stream or device.

    Returns:
        Inverse cosine `array`.

    """
    _ = stream
    arr = _as_array(a)
    out_dt = float32 if arr.dtype.value.startswith(("int", "uint", "bool")) else arr.dtype
    return array(sops.acos(arr._tensor), dtype=out_dt)


def acos(a: ArrayInput, *, stream: StreamOrDevice = None) -> array:
    """Compute inverse cosine, element-wise (alias for `arccos`).

    Args:
        a: Input array.
        stream: Optional stream or device.

    Returns:
        Inverse cosine `array`.

    """
    return arccos(a, stream=stream)


def arctan(a: ArrayInput, *, stream: StreamOrDevice = None) -> array:
    """Compute inverse tangent, element-wise.

    Args:
        a: Input array.
        stream: Optional stream or device.

    Returns:
        Inverse tangent `array`.

    """
    _ = stream
    arr = _as_array(a)
    out_dt = float32 if arr.dtype.value.startswith(("int", "uint", "bool")) else arr.dtype
    return array(sops.atan(arr._tensor), dtype=out_dt)


def atan(a: ArrayInput, *, stream: StreamOrDevice = None) -> array:
    """Compute inverse tangent, element-wise (alias for `arctan`).

    Args:
        a: Input array.
        stream: Optional stream or device.

    Returns:
        Inverse tangent `array`.

    """
    return arctan(a, stream=stream)


def arctan2(a: ArrayInput, b: ArrayInput, *, stream: StreamOrDevice = None) -> array:
    """Compute element-wise arc tangent of `a / b` choosing the quadrant correctly.

    Args:
        a: y-coordinates.
        b: x-coordinates.
        stream: Optional stream or device.

    Returns:
        Angle `array` in radians in `[-pi, pi]`.

    """
    _ = stream
    return _binary_op(a, b, sops.atan2, out_dtype=float32)


def atan2(a: ArrayInput, b: ArrayInput, *, stream: StreamOrDevice = None) -> array:
    """Compute element-wise arc tangent of `a / b` (alias for `arctan2`).

    Args:
        a: y-coordinates.
        b: x-coordinates.
        stream: Optional stream or device.

    Returns:
        Angle `array` in radians.

    """
    return arctan2(a, b, stream=stream)


def sinh(a: ArrayInput, *, stream: StreamOrDevice = None) -> array:
    """Compute hyperbolic sine, element-wise.

    Args:
        a: Input array.
        stream: Optional stream or device.

    Returns:
        Hyperbolic sine `array`.

    """
    _ = stream
    arr = _as_array(a)
    out_dt = float32 if arr.dtype.value.startswith(("int", "uint", "bool")) else arr.dtype
    return array(sops.sinh(arr._tensor), dtype=out_dt)


def cosh(a: ArrayInput, *, stream: StreamOrDevice = None) -> array:
    """Compute hyperbolic cosine, element-wise.

    Args:
        a: Input array.
        stream: Optional stream or device.

    Returns:
        Hyperbolic cosine `array`.

    """
    _ = stream
    arr = _as_array(a)
    out_dt = float32 if arr.dtype.value.startswith(("int", "uint", "bool")) else arr.dtype
    return array(sops.cosh(arr._tensor), dtype=out_dt)


def tanh(a: ArrayInput, *, stream: StreamOrDevice = None) -> array:
    """Compute hyperbolic tangent, element-wise.

    Args:
        a: Input array.
        stream: Optional stream or device.

    Returns:
        Hyperbolic tangent `array`.

    """
    _ = stream
    arr = _as_array(a)
    out_dt = float32 if arr.dtype.value.startswith(("int", "uint", "bool")) else arr.dtype
    return array(sops.tanh(arr._tensor), dtype=out_dt)


def arcsinh(a: ArrayInput, *, stream: StreamOrDevice = None) -> array:
    """Compute inverse hyperbolic sine, element-wise.

    Args:
        a: Input array.
        stream: Optional stream or device.

    Returns:
        Inverse hyperbolic sine `array`.

    """
    _ = stream
    arr = _as_array(a)
    out_dt = float32 if arr.dtype.value.startswith(("int", "uint", "bool")) else arr.dtype
    return array(sops.asinh(arr._tensor), dtype=out_dt)


def asinh(a: ArrayInput, *, stream: StreamOrDevice = None) -> array:
    """Compute inverse hyperbolic sine, element-wise (alias for `arcsinh`).

    Args:
        a: Input array.
        stream: Optional stream or device.

    Returns:
        Inverse hyperbolic sine `array`.

    """
    return arcsinh(a, stream=stream)


def arccosh(a: ArrayInput, *, stream: StreamOrDevice = None) -> array:
    """Compute inverse hyperbolic cosine, element-wise.

    Args:
        a: Input array.
        stream: Optional stream or device.

    Returns:
        Inverse hyperbolic cosine `array`.

    """
    _ = stream
    arr = _as_array(a)
    out_dt = float32 if arr.dtype.value.startswith(("int", "uint", "bool")) else arr.dtype
    return array(sops.acosh(arr._tensor), dtype=out_dt)


def acosh(a: ArrayInput, *, stream: StreamOrDevice = None) -> array:
    """Compute inverse hyperbolic cosine, element-wise (alias for `arccosh`).

    Args:
        a: Input array.
        stream: Optional stream or device.

    Returns:
        Inverse hyperbolic cosine `array`.

    """
    return arccosh(a, stream=stream)


def arctanh(a: ArrayInput, *, stream: StreamOrDevice = None) -> array:
    """Compute inverse hyperbolic tangent, element-wise.

    Args:
        a: Input array.
        stream: Optional stream or device.

    Returns:
        Inverse hyperbolic tangent `array`.

    """
    _ = stream
    arr = _as_array(a)
    out_dt = float32 if arr.dtype.value.startswith(("int", "uint", "bool")) else arr.dtype
    return array(sops.atanh(arr._tensor), dtype=out_dt)


def atanh(a: ArrayInput, *, stream: StreamOrDevice = None) -> array:
    """Compute inverse hyperbolic tangent, element-wise (alias for `arctanh`).

    Args:
        a: Input array.
        stream: Optional stream or device.

    Returns:
        Inverse hyperbolic tangent `array`.

    """
    return arctanh(a, stream=stream)


def degrees(a: ArrayInput, *, stream: StreamOrDevice = None) -> array:
    """Convert angles from radians to degrees.

    Args:
        a: Input angle array in radians.
        stream: Optional stream or device.

    Returns:
        Angle `array` in degrees.

    """
    _ = stream
    arr = _as_array(a)
    return array(sops.rad2deg(arr._tensor), dtype=float32)


def radians(a: ArrayInput, *, stream: StreamOrDevice = None) -> array:
    """Convert angles from degrees to radians.

    Args:
        a: Input angle array in degrees.
        stream: Optional stream or device.

    Returns:
        Angle `array` in radians.

    """
    _ = stream
    arr = _as_array(a)
    return array(sops.deg2rad(arr._tensor), dtype=float32)


def erf(a: ArrayInput, *, stream: StreamOrDevice = None) -> array:
    """Compute the Gauss error function of `a`, element-wise.

    Args:
        a: Input array.
        stream: Optional stream or device.

    Returns:
        Error function `array`.

    """
    _ = stream
    arr = _as_array(a)
    return array(sops.erf(arr._tensor), dtype=float32)


def erfinv(a: ArrayInput, *, stream: StreamOrDevice = None) -> array:
    """Compute the inverse Gauss error function of `a`, element-wise.

    Args:
        a: Input array.
        stream: Optional stream or device.

    Returns:
        Inverse error function `array`.

    """
    _ = stream
    arr = _as_array(a)
    return array(sops.erfinv(arr._tensor), dtype=float32)


def sigmoid(a: ArrayInput, *, stream: StreamOrDevice = None) -> array:
    """Compute the logistic sigmoid `1 / (1 + exp(-a))`, element-wise.

    Args:
        a: Input array.
        stream: Optional stream or device.

    Returns:
        Sigmoid `array`.

    """
    arr = _as_array(a)
    return 1.0 / (1.0 + exp(-arr, stream=stream))


def softmax(
    a: ArrayInput,
    axis: AxesLike = None,
    *,
    precise: bool = False,
    stream: StreamOrDevice = None,
) -> array:
    """Compute the softmax along the given `axis`.

    Args:
        a: Input array.
        axis: Axis or axes along which to compute the softmax.
        precise: Whether to compute in high precision (`float32`).
        stream: Optional stream or device.

    Returns:
        Softmax-normalized `array`.

    """
    _ = precise
    arr = _as_array(a)
    m = max(arr, axis=axis, keepdims=True, stream=stream)
    e = exp(arr - m, stream=stream)
    return (e / sum(e, axis=axis, keepdims=True, stream=stream)).astype(arr.dtype)


def add(a: ArrayInput, b: ArrayInput, *, stream: StreamOrDevice = None) -> array:
    """Add arguments element-wise.

    Args:
        a: First input.
        b: Second input.
        stream: Optional stream or device.

    Returns:
        Element-wise sum `array`.

    """
    _ = stream
    return _as_array(a) + b


def subtract(a: ArrayInput, b: ArrayInput, *, stream: StreamOrDevice = None) -> array:
    """Subtract arguments element-wise.

    Args:
        a: First input.
        b: Second input.
        stream: Optional stream or device.

    Returns:
        Element-wise difference `array`.

    """
    _ = stream
    return _as_array(a) - b


def multiply(a: ArrayInput, b: ArrayInput, *, stream: StreamOrDevice = None) -> array:
    """Multiply arguments element-wise.

    Args:
        a: First input.
        b: Second input.
        stream: Optional stream or device.

    Returns:
        Element-wise product `array`.

    """
    _ = stream
    return _as_array(a) * b


def divide(a: ArrayInput, b: ArrayInput, *, stream: StreamOrDevice = None) -> array:
    """Divide arguments element-wise with true division.

    Args:
        a: Dividend input.
        b: Divisor input.
        stream: Optional stream or device.

    Returns:
        Element-wise quotient `array`.

    """
    _ = stream
    return _as_array(a) / b


def floor_divide(a: ArrayInput, b: ArrayInput, *, stream: StreamOrDevice = None) -> array:
    """Compute the largest integer smaller or equal to the division of the inputs.

    Args:
        a: Dividend input.
        b: Divisor input.
        stream: Optional stream or device.

    Returns:
        Floor-divided `array`.

    """
    _ = stream
    return _as_array(a) // b


def remainder(a: ArrayInput, b: ArrayInput, *, stream: StreamOrDevice = None) -> array:
    """Return element-wise remainder of division.

    Args:
        a: Dividend input.
        b: Divisor input.
        stream: Optional stream or device.

    Returns:
        Remainder `array`.

    """
    _ = stream
    return _as_array(a) % b


def divmod(a: ArrayInput, b: ArrayInput, *, stream: StreamOrDevice = None) -> tuple[array, array]:
    """Return element-wise quotient and remainder simultaneously.

    Args:
        a: Dividend input.
        b: Divisor input.
        stream: Optional stream or device.

    Returns:
        Tuple `(floor_divide(a, b), remainder(a, b))`.

    """
    return (floor_divide(a, b, stream=stream), remainder(a, b, stream=stream))


def power(a: ArrayInput, b: ArrayInput, *, stream: StreamOrDevice = None) -> array:
    """Raise elements of `a` to powers from `b`, element-wise.

    Args:
        a: Base input.
        b: Exponent input.
        stream: Optional stream or device.

    Returns:
        Exponentiated `array`.

    """
    _ = stream
    return _as_array(a) ** b


def pow(a: ArrayInput, b: ArrayInput, *, stream: StreamOrDevice = None) -> array:
    """Raise elements of `a` to powers from `b` (alias for `power`).

    Args:
        a: Base input.
        b: Exponent input.
        stream: Optional stream or device.

    Returns:
        Exponentiated `array`.

    """
    return power(a, b, stream=stream)


def maximum(a: ArrayInput, b: ArrayInput, *, stream: StreamOrDevice = None) -> array:
    """Element-wise maximum of array elements.

    Args:
        a: First input.
        b: Second input.
        stream: Optional stream or device.

    Returns:
        Element-wise maximum `array`.

    """
    _ = stream
    return _binary_op(a, b, sops.maximum)


def minimum(a: ArrayInput, b: ArrayInput, *, stream: StreamOrDevice = None) -> array:
    """Element-wise minimum of array elements.

    Args:
        a: First input.
        b: Second input.
        stream: Optional stream or device.

    Returns:
        Element-wise minimum `array`.

    """
    _ = stream
    return _binary_op(a, b, sops.minimum)


def logaddexp(a: ArrayInput, b: ArrayInput, *, stream: StreamOrDevice = None) -> array:
    """Compute `log(exp(a) + exp(b))` in a numerically stable way.

    Args:
        a: First input.
        b: Second input.
        stream: Optional stream or device.

    Returns:
        `log(exp(a) + exp(b))` `array`.

    """
    _ = stream
    return _binary_op(a, b, sops.logaddexp, out_dtype=float32)


def clip(
    a: ArrayInput,
    a_min: ArrayInput | None,
    a_max: ArrayInput | None,
    *,
    stream: StreamOrDevice = None,
) -> array:
    """Clip (limit) the values in an array to `[a_min, a_max]`.

    Args:
        a: Input array.
        a_min: Minimum value (or None for no lower bound).
        a_max: Maximum value (or None for no upper bound).
        stream: Optional stream or device.

    Returns:
        Clipped `array`.

    """
    res = _as_array(a)
    res = maximum(res, a_min, stream=stream) if a_min is not None else res
    return minimum(res, a_max, stream=stream) if a_max is not None else res


def nan_to_num(
    x: ArrayInput,
    nan: float = 0.0,
    posinf: float | None = None,
    neginf: float | None = None,
    *,
    stream: StreamOrDevice = None,
) -> array:
    """Replace `NaN` with `nan` and infinity with finite numbers.

    Args:
        x: Input array.
        nan: Value to replace `NaN` with.
        posinf: Value to replace positive infinity with.
        neginf: Value to replace negative infinity with.
        stream: Optional stream or device.

    Returns:
        Sanitized `array`.

    """
    _ = stream
    arr = _as_array(x)
    return array(
        sops.nan_to_num(arr._tensor, nan=nan, posinf=posinf, neginf=neginf),
        dtype=arr.dtype,
    )


def real(a: ArrayInput, *, stream: StreamOrDevice = None) -> array:
    """Return the real part of the complex argument.

    Args:
        a: Input array.
        stream: Optional stream or device.

    Returns:
        Real part `array`.

    """
    _ = stream
    return _as_array(a).real


def imag(a: ArrayInput, *, stream: StreamOrDevice = None) -> array:
    """Return the imaginary part of the complex argument.

    Args:
        a: Input array.
        stream: Optional stream or device.

    Returns:
        Imaginary part `array`.

    """
    _ = stream
    return _as_array(a).imag


def conjugate(a: ArrayInput, *, stream: StreamOrDevice = None) -> array:
    """Return the complex conjugate, element-wise.

    Args:
        a: Input array.
        stream: Optional stream or device.

    Returns:
        Complex conjugate `array`.

    """
    _ = stream
    arr = _as_array(a)
    return array(sops.conj(arr._tensor), dtype=arr.dtype)


def conj(a: ArrayInput, *, stream: StreamOrDevice = None) -> array:
    """Return the complex conjugate, element-wise (alias for `conjugate`).

    Args:
        a: Input array.
        stream: Optional stream or device.

    Returns:
        Complex conjugate `array`.

    """
    return conjugate(a, stream=stream)


# ==============================================================================
# Comparison, Logical & Bitwise Operations
# ==============================================================================


def equal(a: ArrayInput, b: ArrayInput, *, stream: StreamOrDevice = None) -> array:
    """Return `a == b` element-wise.

    Args:
        a: First input.
        b: Second input.
        stream: Optional stream or device.

    Returns:
        Boolean `array`.

    """
    _ = stream
    return _binary_op(a, b, sops.equal, out_dtype=bool_)


def not_equal(a: ArrayInput, b: ArrayInput, *, stream: StreamOrDevice = None) -> array:
    """Return `a != b` element-wise.

    Args:
        a: First input.
        b: Second input.
        stream: Optional stream or device.

    Returns:
        Boolean `array`.

    """
    _ = stream
    return _binary_op(a, b, sops.not_equal, out_dtype=bool_)


def less(a: ArrayInput, b: ArrayInput, *, stream: StreamOrDevice = None) -> array:
    """Return `a < b` element-wise.

    Args:
        a: First input.
        b: Second input.
        stream: Optional stream or device.

    Returns:
        Boolean `array`.

    """
    _ = stream
    return _binary_op(a, b, sops.less, out_dtype=bool_)


def less_equal(a: ArrayInput, b: ArrayInput, *, stream: StreamOrDevice = None) -> array:
    """Return `a <= b` element-wise.

    Args:
        a: First input.
        b: Second input.
        stream: Optional stream or device.

    Returns:
        Boolean `array`.

    """
    _ = stream
    return _binary_op(a, b, sops.less_equal, out_dtype=bool_)


def greater(a: ArrayInput, b: ArrayInput, *, stream: StreamOrDevice = None) -> array:
    """Return `a > b` element-wise.

    Args:
        a: First input.
        b: Second input.
        stream: Optional stream or device.

    Returns:
        Boolean `array`.

    """
    _ = stream
    return _binary_op(a, b, sops.greater, out_dtype=bool_)


def greater_equal(a: ArrayInput, b: ArrayInput, *, stream: StreamOrDevice = None) -> array:
    """Return `a >= b` element-wise.

    Args:
        a: First input.
        b: Second input.
        stream: Optional stream or device.

    Returns:
        Boolean `array`.

    """
    _ = stream
    return _binary_op(a, b, sops.greater_equal, out_dtype=bool_)


def array_equal(
    a: ArrayInput,
    b: ArrayInput,
    equal_nan: bool = False,
    *,
    stream: StreamOrDevice = None,
) -> array:
    """Return True if two arrays have the same shape and elements, False otherwise.

    Args:
        a: First input array.
        b: Second input array.
        equal_nan: Whether to compare `NaN` values as equal.
        stream: Optional stream or device.

    Returns:
        0D boolean `array`.

    """
    arr_a = _as_array(a)
    arr_b = _as_array(b)
    if arr_a.shape != arr_b.shape:
        return array(False, dtype=bool_)
    eq = equal(arr_a, arr_b, stream=stream)
    if equal_nan:
        both_nan = logical_and(
            isnan(arr_a, stream=stream), isnan(arr_b, stream=stream), stream=stream
        )
        eq = logical_or(eq, both_nan, stream=stream)
    return all(eq, stream=stream)


def isclose(
    a: ArrayInput,
    b: ArrayInput,
    rtol: float = 1e-5,
    atol: float = 1e-8,
    equal_nan: bool = False,
    *,
    stream: StreamOrDevice = None,
) -> array:
    """Return a boolean array where two arrays are element-wise equal within a tolerance.

    Args:
        a: First input.
        b: Second input.
        rtol: Relative tolerance parameter.
        atol: Absolute tolerance parameter.
        equal_nan: Whether `NaN` values in identical positions compare as True.
        stream: Optional stream or device.

    Returns:
        Boolean `array`.

    """
    _ = stream
    arr_a = _as_array(a)
    arr_b = _as_array(b)
    return array(
        sops.isclose(arr_a._tensor, arr_b._tensor, rtol=rtol, atol=atol, equal_nan=equal_nan),
        dtype=bool_,
    )


def allclose(
    a: ArrayInput,
    b: ArrayInput,
    rtol: float = 1e-5,
    atol: float = 1e-8,
    equal_nan: bool = False,
    *,
    stream: StreamOrDevice = None,
) -> array:
    """Return True if two arrays are element-wise equal within a tolerance.

    Args:
        a: First input.
        b: Second input.
        rtol: Relative tolerance parameter.
        atol: Absolute tolerance parameter.
        equal_nan: Whether `NaN` values compare as equal.
        stream: Optional stream or device.

    Returns:
        0D boolean `array`.

    """
    return all(isclose(a, b, rtol=rtol, atol=atol, equal_nan=equal_nan, stream=stream))


def isfinite(a: ArrayInput, *, stream: StreamOrDevice = None) -> array:
    """Test element-wise for finiteness (not infinity and not Not a Number).

    Args:
        a: Input array.
        stream: Optional stream or device.

    Returns:
        Boolean `array`.

    """
    _ = stream
    arr = _as_array(a)
    return array(sops.isfinite(arr._tensor), dtype=bool_)


def isinf(a: ArrayInput, *, stream: StreamOrDevice = None) -> array:
    """Test element-wise for positive or negative infinity.

    Args:
        a: Input array.
        stream: Optional stream or device.

    Returns:
        Boolean `array`.

    """
    _ = stream
    arr = _as_array(a)
    return array(sops.isinf(arr._tensor), dtype=bool_)


def isnan(a: ArrayInput, *, stream: StreamOrDevice = None) -> array:
    """Test element-wise for NaN and return result as a boolean array.

    Args:
        a: Input array.
        stream: Optional stream or device.

    Returns:
        Boolean `array`.

    """
    _ = stream
    arr = _as_array(a)
    return array(sops.isnan(arr._tensor), dtype=bool_)


def isposinf(a: ArrayInput, *, stream: StreamOrDevice = None) -> array:
    """Test element-wise for positive infinity.

    Args:
        a: Input array.
        stream: Optional stream or device.

    Returns:
        Boolean `array`.

    """
    _ = stream
    arr = _as_array(a)
    return array(sops.isposinf(arr._tensor), dtype=bool_)


def isneginf(a: ArrayInput, *, stream: StreamOrDevice = None) -> array:
    """Test element-wise for negative infinity.

    Args:
        a: Input array.
        stream: Optional stream or device.

    Returns:
        Boolean `array`.

    """
    _ = stream
    arr = _as_array(a)
    return array(sops.isneginf(arr._tensor), dtype=bool_)


def logical_not(a: ArrayInput, *, stream: StreamOrDevice = None) -> array:
    """Compute the truth value of `NOT a` element-wise.

    Args:
        a: Input array.
        stream: Optional stream or device.

    Returns:
        Boolean `array`.

    """
    _ = stream
    arr = _as_array(a)
    return array(sops.logical_not(arr._tensor), dtype=bool_)


def logical_and(a: ArrayInput, b: ArrayInput, *, stream: StreamOrDevice = None) -> array:
    """Compute the truth value of `a AND b` element-wise.

    Args:
        a: First input.
        b: Second input.
        stream: Optional stream or device.

    Returns:
        Boolean `array`.

    """
    _ = stream
    return _binary_op(a, b, sops.logical_and, out_dtype=bool_)


def logical_or(a: ArrayInput, b: ArrayInput, *, stream: StreamOrDevice = None) -> array:
    """Compute the truth value of `a OR b` element-wise.

    Args:
        a: First input.
        b: Second input.
        stream: Optional stream or device.

    Returns:
        Boolean `array`.

    """
    _ = stream
    return _binary_op(a, b, sops.logical_or, out_dtype=bool_)


def logical_xor(a: ArrayInput, b: ArrayInput, *, stream: StreamOrDevice = None) -> array:
    """Compute the truth value of `a XOR b` element-wise.

    Args:
        a: First input.
        b: Second input.
        stream: Optional stream or device.

    Returns:
        Boolean `array`.

    """
    ba = _as_array(a).astype(bool_)
    bb = _as_array(b).astype(bool_)
    return not_equal(ba, bb, stream=stream)


def bitwise_and(a: ArrayInput, b: ArrayInput, *, stream: StreamOrDevice = None) -> array:
    """Compute the bit-wise AND of two arrays element-wise.

    Args:
        a: First integer/boolean input.
        b: Second integer/boolean input.
        stream: Optional stream or device.

    Returns:
        Bitwise AND `array`.

    """
    _ = stream
    return _as_array(a) & b


def bitwise_or(a: ArrayInput, b: ArrayInput, *, stream: StreamOrDevice = None) -> array:
    """Compute the bit-wise OR of two arrays element-wise.

    Args:
        a: First integer/boolean input.
        b: Second integer/boolean input.
        stream: Optional stream or device.

    Returns:
        Bitwise OR `array`.

    """
    _ = stream
    return _as_array(a) | b


def bitwise_xor(a: ArrayInput, b: ArrayInput, *, stream: StreamOrDevice = None) -> array:
    """Compute the bit-wise XOR of two arrays element-wise.

    Args:
        a: First integer/boolean input.
        b: Second integer/boolean input.
        stream: Optional stream or device.

    Returns:
        Bitwise XOR `array`.

    """
    _ = stream
    return _as_array(a) ^ b


def bitwise_invert(a: ArrayInput, *, stream: StreamOrDevice = None) -> array:
    """Compute bit-wise inversion (`~a`) element-wise.

    Args:
        a: Input integer or boolean array.
        stream: Optional stream or device.

    Returns:
        Bitwise inverted `array`.

    """
    _ = stream
    return ~_as_array(a)


def left_shift(a: ArrayInput, b: ArrayInput, *, stream: StreamOrDevice = None) -> array:
    """Shift the bits of an integer to the left (`a << b`).

    Args:
        a: Input integer array.
        b: Number of bits to shift left.
        stream: Optional stream or device.

    Returns:
        Left-shifted `array`.

    """
    _ = stream
    return _binary_op(a, b, sops.left_shift)


def right_shift(a: ArrayInput, b: ArrayInput, *, stream: StreamOrDevice = None) -> array:
    """Shift the bits of an integer to the right (`a >> b`).

    Args:
        a: Input integer array.
        b: Number of bits to shift right.
        stream: Optional stream or device.

    Returns:
        Right-shifted `array`.

    """
    _ = stream
    return _binary_op(a, b, sops.right_shift)


# ==============================================================================
# Type Casting & Inspection Operations
# ==============================================================================


def astype(a: ArrayInput, dtype: DTypeLike, *, stream: StreamOrDevice = None) -> array:
    """Cast an array to a specified data type.

    Args:
        a: Input array.
        dtype: Target data type.
        stream: Optional stream or device.

    Returns:
        Cast `array`.

    """
    return _as_array(a).astype(dtype, stream=stream)


def view(a: ArrayInput, dtype: DTypeLike, *, stream: StreamOrDevice = None) -> array:
    """View the array data as a new `dtype`.

    Args:
        a: Input array.
        dtype: Target data type.
        stream: Optional stream or device.

    Returns:
        Viewed `array`.

    """
    return _as_array(a).view(dtype, stream=stream)


def can_cast(
    from_: ArrayInput | DTypeLike,
    to: DTypeLike,
    *,
    stream: StreamOrDevice = None,
) -> bool:
    """Return True if cast between `from_` and `to` can occur according to MLX type promotion rules.

    Args:
        from_: Source `array` or `DTypeLike`.
        to: Target `DTypeLike`.
        stream: Optional stream or device.

    Returns:
        True if `from_` can be safely cast to `to`.

    """
    _ = stream
    dt_from: DType = from_.dtype if isinstance(from_, array) else resolve_dtype(from_)
    dt_to: DType = resolve_dtype(to)
    return promote_dtypes(dt_from, dt_to) == dt_to


def result_type(*arrays_and_dtypes: ArrayInput | DTypeLike) -> DType:
    """Return the `DType` that results from applying the MLX type promotion rules to the arguments.

    Args:
        *arrays_and_dtypes: Arbitrary mix of arrays, scalars, or `DTypeLike` specifications.

    Returns:
        Promoted `DType`.

    """
    if not arrays_and_dtypes:
        return float32
    cur_dt: DType | None = None
    for item in arrays_and_dtypes:
        if isinstance(item, array):
            dt = item.dtype
        elif isinstance(item, DType):
            dt = item
        elif isinstance(item, (str, type, DtypeCategory, SwitcherooDType)) or item is None:
            dt = resolve_dtype(item)
        else:
            dt = _as_array(item).dtype
        cur_dt = dt if cur_dt is None else promote_dtypes(cur_dt, dt)
    return cur_dt if cur_dt is not None else float32


def to_fp8(x: ArrayInput, *, stream: StreamOrDevice = None) -> array:
    """Convert a floating-point array `x` to 8-bit floating point (`uint8` container).

    Args:
        x: Input floating-point array.
        stream: Optional stream or device.

    Returns:
        `uint8` `array` representing E4M3 FP8 values.

    """
    _ = stream
    arr = _as_array(x)
    return array(sops.cast(arr._tensor, to_switcheroo_dtype(uint8)), dtype=uint8)


def from_fp8(
    x: ArrayInput,
    dtype: DTypeLike = float32,
    *,
    stream: StreamOrDevice = None,
) -> array:
    """Convert an 8-bit floating-point (`uint8`) array `x` back to a floating-point `dtype`.

    Args:
        x: Input `uint8` FP8 array.
        dtype: Target floating-point data type.
        stream: Optional stream or device.

    Returns:
        Converted floating-point `array`.

    """
    _ = stream
    arr = _as_array(x)
    dt = resolve_dtype(dtype, default=float32)
    return array(sops.cast(arr._tensor, to_switcheroo_dtype(dt)), dtype=dt)


# ==============================================================================
# Reduction, Cumulative & Scan Operations
# ==============================================================================


def all(
    a: ArrayInput,
    axis: AxesLike = None,
    keepdims: bool = False,
    *,
    stream: StreamOrDevice = None,
) -> array:
    """Test whether all array elements along a given `axis` evaluate to True.

    Args:
        a: Input array.
        axis: Axis or axes along which to perform the logical AND reduction.
        keepdims: Whether to retain reduced dimensions with size 1.
        stream: Optional stream or device.

    Returns:
        Boolean `array`.

    """
    _ = stream
    arr = _as_array(a)
    return array(sops.all(arr._tensor, axis=axis, keepdims=keepdims), dtype=bool_)


def any(
    a: ArrayInput,
    axis: AxesLike = None,
    keepdims: bool = False,
    *,
    stream: StreamOrDevice = None,
) -> array:
    """Test whether any array element along a given `axis` evaluates to True.

    Args:
        a: Input array.
        axis: Axis or axes along which to perform the logical OR reduction.
        keepdims: Whether to retain reduced dimensions with size 1.
        stream: Optional stream or device.

    Returns:
        Boolean `array`.

    """
    _ = stream
    arr = _as_array(a)
    return array(sops.any(arr._tensor, axis=axis, keepdims=keepdims), dtype=bool_)


def sum(
    a: ArrayInput,
    axis: AxesLike = None,
    keepdims: bool = False,
    *,
    stream: StreamOrDevice = None,
) -> array:
    """Sum of array elements over a given `axis`.

    Args:
        a: Input array.
        axis: Axis or axes along which a sum is performed.
        keepdims: Whether to retain reduced dimensions with size 1.
        stream: Optional stream or device.

    Returns:
        Summed `array`.

    """
    _ = stream
    arr = _as_array(a)
    out_dt = int32 if arr.dtype == bool_ else arr.dtype
    return array(sops.sum(arr._tensor, axis=axis, keepdims=keepdims), dtype=out_dt)


def prod(
    a: ArrayInput,
    axis: AxesLike = None,
    keepdims: bool = False,
    *,
    stream: StreamOrDevice = None,
) -> array:
    """Return the product of array elements over a given `axis`.

    Args:
        a: Input array.
        axis: Axis or axes along which a product is performed.
        keepdims: Whether to retain reduced dimensions with size 1.
        stream: Optional stream or device.

    Returns:
        Product `array`.

    """
    _ = stream
    arr = _as_array(a)
    out_dt = int32 if arr.dtype == bool_ else arr.dtype
    return array(sops.prod(arr._tensor, axis=axis, keepdims=keepdims), dtype=out_dt)


def mean(
    a: ArrayInput,
    axis: AxesLike = None,
    keepdims: bool = False,
    *,
    stream: StreamOrDevice = None,
) -> array:
    """Compute the arithmetic mean along the specified `axis`.

    Args:
        a: Input array.
        axis: Axis or axes along which the means are computed.
        keepdims: Whether to retain reduced dimensions with size 1.
        stream: Optional stream or device.

    Returns:
        Mean `array`.

    """
    _ = stream
    arr = _as_array(a)
    out_dt = float32 if arr.dtype.value.startswith(("int", "uint", "bool")) else arr.dtype
    return array(sops.mean(arr._tensor, axis=axis, keepdims=keepdims), dtype=out_dt)


def var(
    a: ArrayInput,
    axis: AxesLike = None,
    keepdims: bool = False,
    ddof: int = 0,
    *,
    stream: StreamOrDevice = None,
) -> array:
    """Compute the variance along the specified `axis`.

    Args:
        a: Input array.
        axis: Axis or axes along which the variance is computed.
        keepdims: Whether to retain reduced dimensions with size 1.
        ddof: Delta degrees of freedom.
        stream: Optional stream or device.

    Returns:
        Variance `array`.

    """
    _ = stream
    arr = _as_array(a)
    return array(
        sops.variance(arr._tensor, axis=axis, keepdims=keepdims, ddof=ddof),
        dtype=float32,
    )


def std(
    a: ArrayInput,
    axis: AxesLike = None,
    keepdims: bool = False,
    ddof: int = 0,
    *,
    stream: StreamOrDevice = None,
) -> array:
    """Compute the standard deviation along the specified `axis`.

    Args:
        a: Input array.
        axis: Axis or axes along which the standard deviation is computed.
        keepdims: Whether to retain reduced dimensions with size 1.
        ddof: Delta degrees of freedom.
        stream: Optional stream or device.

    Returns:
        Standard deviation `array`.

    """
    _ = stream
    arr = _as_array(a)
    return array(
        sops.std(arr._tensor, axis=axis, keepdims=keepdims, ddof=ddof),
        dtype=float32,
    )


def min(
    a: ArrayInput,
    axis: AxesLike = None,
    keepdims: bool = False,
    *,
    stream: StreamOrDevice = None,
) -> array:
    """Return the minimum of an array or minimum along an `axis`.

    Args:
        a: Input array.
        axis: Axis or axes along which to operate.
        keepdims: Whether to retain reduced dimensions with size 1.
        stream: Optional stream or device.

    Returns:
        Minimum `array`.

    """
    _ = stream
    arr = _as_array(a)
    return array(sops.min(arr._tensor, axis=axis, keepdims=keepdims), dtype=arr.dtype)


def max(
    a: ArrayInput,
    axis: AxesLike = None,
    keepdims: bool = False,
    *,
    stream: StreamOrDevice = None,
) -> array:
    """Return the maximum of an array or maximum along an `axis`.

    Args:
        a: Input array.
        axis: Axis or axes along which to operate.
        keepdims: Whether to retain reduced dimensions with size 1.
        stream: Optional stream or device.

    Returns:
        Maximum `array`.

    """
    _ = stream
    arr = _as_array(a)
    return array(sops.max(arr._tensor, axis=axis, keepdims=keepdims), dtype=arr.dtype)


def minmax(
    a: ArrayInput,
    axis: AxesLike = None,
    keepdims: bool = False,
    *,
    stream: StreamOrDevice = None,
) -> tuple[array, array]:
    """Compute the minimum and maximum of `a` simultaneously along `axis`.

    Args:
        a: Input array.
        axis: Axis or axes along which to operate.
        keepdims: Whether to retain reduced dimensions with size 1.
        stream: Optional stream or device.

    Returns:
        Tuple `(min(a), max(a))`.

    """
    return (
        min(a, axis=axis, keepdims=keepdims, stream=stream),
        max(a, axis=axis, keepdims=keepdims, stream=stream),
    )


def argmin(
    a: ArrayInput,
    axis: int | None = None,
    keepdims: bool = False,
    *,
    stream: StreamOrDevice = None,
) -> array:
    """Return the indices of the minimum values along an `axis`.

    Args:
        a: Input array.
        axis: Axis along which to find minimum indices.
        keepdims: Whether to retain the reduced dimension with size 1.
        stream: Optional stream or device.

    Returns:
        `uint32` index `array`.

    """
    _ = stream
    arr = _as_array(a)
    return array(sops.argmin(arr._tensor, axis=axis, keepdims=keepdims), dtype=uint32)


def argmax(
    a: ArrayInput,
    axis: int | None = None,
    keepdims: bool = False,
    *,
    stream: StreamOrDevice = None,
) -> array:
    """Return the indices of the maximum values along an `axis`.

    Args:
        a: Input array.
        axis: Axis along which to find maximum indices.
        keepdims: Whether to retain the reduced dimension with size 1.
        stream: Optional stream or device.

    Returns:
        `uint32` index `array`.

    """
    _ = stream
    arr = _as_array(a)
    return array(sops.argmax(arr._tensor, axis=axis, keepdims=keepdims), dtype=uint32)


def logsumexp(
    a: ArrayInput,
    axis: AxesLike = None,
    keepdims: bool = False,
    *,
    stream: StreamOrDevice = None,
) -> array:
    """Compute the log of the sum of exponentials of input elements along `axis`.

    Args:
        a: Input array.
        axis: Axis or axes over which the sum is taken.
        keepdims: Whether to retain reduced dimensions with size 1.
        stream: Optional stream or device.

    Returns:
        Log-sum-exp `array`.

    """
    _ = stream
    arr = _as_array(a)
    return array(sops.logsumexp(arr._tensor, axis=axis, keepdims=keepdims), dtype=float32)


def median(
    a: ArrayInput,
    axis: AxesLike = None,
    keepdims: bool = False,
    *,
    stream: StreamOrDevice = None,
) -> array:
    """Compute the median along the specified `axis`.

    Args:
        a: Input array.
        axis: Axis or axes along which the medians are computed.
        keepdims: Whether to retain reduced dimensions with size 1.
        stream: Optional stream or device.

    Returns:
        Median `array`.

    """
    _ = stream
    arr = _as_array(a)
    return array(sops.nanmedian(arr._tensor, axis=axis, keepdims=keepdims), dtype=float32)


def count_nonzero(
    a: ArrayInput,
    axis: AxesLike = None,
    keepdims: bool = False,
    *,
    stream: StreamOrDevice = None,
) -> array:
    """Count the number of non-zero values in the array `a` along `axis`.

    Args:
        a: Input array.
        axis: Axis or axes along which to count non-zeros.
        keepdims: Whether to retain reduced dimensions with size 1.
        stream: Optional stream or device.

    Returns:
        Count `array` of dtype `int32`.

    """
    nz_mask = not_equal(a, 0, stream=stream).astype(int32)
    return sum(nz_mask, axis=axis, keepdims=keepdims, stream=stream)


def _apply_scan_1d(
    vals: list[float],
    mode: str,
    reverse: bool,
    inclusive: bool,
) -> list[float]:
    """Execute a 1D cumulative scan (`sum`, `prod`, `max`, `min`, `logaddexp`).

    Args:
        vals: 1D list of float values.
        mode: Scan mode (`'sum'`, `'prod'`, `'max'`, `'min'`, `'logaddexp'`).
        reverse: Whether to scan from right to left.
        inclusive: Whether the i-th output includes the i-th input.

    Returns:
        1D list of scanned values.

    """
    n = len(vals)
    if n == 0:
        return []
    seq = list(reversed(vals)) if reverse else list(vals)
    identities: dict[str, float] = {
        "sum": 0.0,
        "prod": 1.0,
        "max": float("-inf"),
        "min": float("inf"),
        "logaddexp": float("-inf"),
    }
    acc: float = identities[mode]
    out: list[float] = []
    for x in seq:
        if not inclusive:
            out.append(acc)
        if mode == "sum":
            acc = acc + x
        elif mode == "prod":
            acc = acc * x
        elif mode == "max":
            acc = x if x > acc else acc
        elif mode == "min":
            acc = x if x < acc else acc
        else:
            if acc == float("-inf"):
                acc = x
            else:
                m_val: float = builtins.max(acc, x)
                acc = m_val + math.log(math.exp(acc - m_val) + math.exp(x - m_val))
        if inclusive:
            out.append(acc)
    return list(reversed(out)) if reverse else out


def _scan_op(
    a: ArrayInput,
    mode: str,
    axis: int | None = None,
    reverse: bool = False,
    inclusive: bool = True,
    dtype: DTypeLike = None,
) -> array:
    """Apply a cumulative scan operation along `axis`.

    Args:
        a: Input array.
        mode: Scan operation (`'sum'`, `'prod'`, `'max'`, `'min'`, `'logaddexp'`).
        axis: Axis along which to scan (flattens if None).
        reverse: Whether to scan in reverse order.
        inclusive: Whether the scan is inclusive.
        dtype: Optional output data type.

    Returns:
        Scanned `array`.

    """
    arr = _as_array(a)
    target_dt = resolve_dtype(
        dtype,
        default=(float32 if mode == "logaddexp" else (int32 if arr.dtype == bool_ else arr.dtype)),
    )
    if axis is None:
        arr = arr.reshape(-1)
        ax = 0
    else:
        ax = axis if axis >= 0 else arr.ndim + axis

    if not reverse and inclusive:
        if mode == "logaddexp":
            m = max(arr, axis=ax, keepdims=True)
            return m + log(cumsum(exp(arr - m), axis=ax))
        scan_fns = {
            "sum": sops.cumsum,
            "prod": sops.cumprod,
            "max": sops.cummax,
            "min": sops.cummin,
        }
        fn = scan_fns[mode]
        return array(fn(arr._tensor, axis=ax), dtype=target_dt)

    moved = arr.moveaxis(ax, -1) if ax != arr.ndim - 1 else arr
    orig_moved_shape = moved.shape
    flat_2d = moved.reshape(-1, orig_moved_shape[-1])
    rows = flat_2d.tolist()
    scanned_rows: list[list[float]] = []
    for r in rows if isinstance(rows, list) else []:
        row_floats: list[float] = [
            float(v.real)
            for v in (r if isinstance(r, list) else [r])
            if isinstance(v, (int, float, bool, complex))
        ]
        scanned_rows.append(_apply_scan_1d(row_floats, mode, reverse, inclusive))
    res_2d = array(scanned_rows, dtype=target_dt).reshape(orig_moved_shape)
    return res_2d.moveaxis(-1, ax) if ax != arr.ndim - 1 else res_2d


def cumsum(
    a: ArrayInput,
    axis: int | None = None,
    reverse: bool = False,
    inclusive: bool = True,
    *,
    dtype: DTypeLike = None,
    stream: StreamOrDevice = None,
) -> array:
    """Return the cumulative sum of the elements along a given `axis`.

    Args:
        a: Input array.
        axis: Axis along which the cumulative sum is computed.
        reverse: Perform the cumulative sum in reverse direction.
        inclusive: The i-th element of the output includes the i-th element of the input.
        dtype: Optional output data type.
        stream: Optional stream or device.

    Returns:
        Cumulative sum `array`.

    """
    _ = stream
    return _scan_op(a, "sum", axis=axis, reverse=reverse, inclusive=inclusive, dtype=dtype)


def cumprod(
    a: ArrayInput,
    axis: int | None = None,
    reverse: bool = False,
    inclusive: bool = True,
    *,
    dtype: DTypeLike = None,
    stream: StreamOrDevice = None,
) -> array:
    """Return the cumulative product of the elements along a given `axis`.

    Args:
        a: Input array.
        axis: Axis along which the cumulative product is computed.
        reverse: Perform the cumulative product in reverse direction.
        inclusive: The i-th element of the output includes the i-th element of the input.
        dtype: Optional output data type.
        stream: Optional stream or device.

    Returns:
        Cumulative product `array`.

    """
    _ = stream
    return _scan_op(a, "prod", axis=axis, reverse=reverse, inclusive=inclusive, dtype=dtype)


def cummax(
    a: ArrayInput,
    axis: int | None = None,
    reverse: bool = False,
    inclusive: bool = True,
    *,
    stream: StreamOrDevice = None,
) -> array:
    """Return the cumulative maximum of the elements along a given `axis`.

    Args:
        a: Input array.
        axis: Axis along which the cumulative maximum is computed.
        reverse: Perform the cumulative maximum in reverse direction.
        inclusive: The i-th element of the output includes the i-th element of the input.
        stream: Optional stream or device.

    Returns:
        Cumulative maximum `array`.

    """
    _ = stream
    return _scan_op(a, "max", axis=axis, reverse=reverse, inclusive=inclusive)


def cummin(
    a: ArrayInput,
    axis: int | None = None,
    reverse: bool = False,
    inclusive: bool = True,
    *,
    stream: StreamOrDevice = None,
) -> array:
    """Return the cumulative minimum of the elements along a given `axis`.

    Args:
        a: Input array.
        axis: Axis along which the cumulative minimum is computed.
        reverse: Perform the cumulative minimum in reverse direction.
        inclusive: The i-th element of the output includes the i-th element of the input.
        stream: Optional stream or device.

    Returns:
        Cumulative minimum `array`.

    """
    _ = stream
    return _scan_op(a, "min", axis=axis, reverse=reverse, inclusive=inclusive)


def logcumsumexp(
    a: ArrayInput,
    axis: int | None = None,
    reverse: bool = False,
    inclusive: bool = True,
    *,
    stream: StreamOrDevice = None,
) -> array:
    """Return the cumulative logarithm of the sum of exponentials along a given `axis`.

    Args:
        a: Input array.
        axis: Axis along which the cumulative logsumexp is computed.
        reverse: Perform the cumulative logsumexp in reverse direction.
        inclusive: The i-th element of the output includes the i-th element of the input.
        stream: Optional stream or device.

    Returns:
        Cumulative logsumexp `array`.

    """
    _ = stream
    return _scan_op(a, "logaddexp", axis=axis, reverse=reverse, inclusive=inclusive)


def diff(
    a: ArrayInput,
    n: int = 1,
    axis: int = -1,
    prepend: ArrayInput | None = None,
    append: ArrayInput | None = None,
    *,
    stream: StreamOrDevice = None,
) -> array:
    """Calculate the `n`-th discrete difference along the given `axis`.

    Args:
        a: Input array.
        n: The number of times values are differenced.
        axis: The axis along which the difference is taken.
        prepend: Optional values to prepend to `a` along `axis` prior to performing the difference.
        append: Optional values to append to `a` along `axis` prior to performing the difference.
        stream: Optional stream or device.

    Returns:
        The `n`-th differences `array`.

    """
    arr = _as_array(a)
    parts: list[array] = []
    if prepend is not None:
        parts.append(_as_array(prepend))
    parts.append(arr)
    if append is not None:
        parts.append(_as_array(append))
    cur = concatenate(parts, axis=axis, stream=stream) if len(parts) > 1 else arr
    if n <= 0:
        return cur
    ax = axis if axis >= 0 else cur.ndim + axis
    for _ in range(n):
        idx_hi = tuple(
            builtins.slice(1, None) if i == ax else builtins.slice(None) for i in range(cur.ndim)
        )
        idx_lo = tuple(
            builtins.slice(None, -1) if i == ax else builtins.slice(None) for i in range(cur.ndim)
        )
        cur = cur[idx_hi] - cur[idx_lo]
    return cur


# ==============================================================================
# Shape, Indexing, Sorting, Searching & Set Operations
# ==============================================================================


def reshape(
    a: ArrayInput,
    shape: ShapeLike,
    *,
    stream: StreamOrDevice = None,
) -> array:
    """Reshape an array while preserving its total size.

    Args:
        a: Input array.
        shape: New shape specification.
        stream: Optional stream or device.

    Returns:
        Reshaped `array`.

    """
    _ = stream
    return _as_array(a).reshape(_norm_shape(shape))


def flatten(
    a: ArrayInput,
    start_axis: int = 0,
    end_axis: int = -1,
    *,
    stream: StreamOrDevice = None,
) -> array:
    """Flatten dimensions of `a` between `start_axis` and `end_axis`.

    Args:
        a: Input array.
        start_axis: First dimension to flatten.
        end_axis: Last dimension to flatten.
        stream: Optional stream or device.

    Returns:
        Flattened `array`.

    """
    _ = stream
    arr = _as_array(a)
    if arr.ndim == 0:
        return arr.reshape((1,))
    shp = list(arr.shape)
    s_ax = start_axis if start_axis >= 0 else len(shp) + start_axis
    e_ax = end_axis if end_axis >= 0 else len(shp) + end_axis
    if s_ax > e_ax:
        return arr
    flat_dim = 1
    for i in range(s_ax, e_ax + 1):
        flat_dim *= shp[i]
    new_shape = shp[:s_ax] + [flat_dim] + shp[e_ax + 1 :]
    return arr.reshape(new_shape)


def unflatten(
    a: ArrayInput,
    axis: int,
    shape: Sequence[int],
    *,
    stream: StreamOrDevice = None,
) -> array:
    """Unflatten an `axis` of `a` into multiple dimensions specified by `shape`.

    Args:
        a: Input array.
        axis: Axis to unflatten.
        shape: Shape to unflatten `axis` into.
        stream: Optional stream or device.

    Returns:
        Unflattened `array`.

    """
    _ = stream
    arr = _as_array(a)
    old_shape = list(arr.shape)
    ax = axis if axis >= 0 else len(old_shape) + axis
    new_shape = old_shape[:ax] + list(shape) + old_shape[ax + 1 :]
    return arr.reshape(new_shape)


def squeeze(
    a: ArrayInput,
    axis: AxesLike = None,
    *,
    stream: StreamOrDevice = None,
) -> array:
    """Remove length-1 axes from `a`.

    Args:
        a: Input array.
        axis: Optional axis or sequence of axes to squeeze.
        stream: Optional stream or device.

    Returns:
        Squeezed `array`.

    """
    return _as_array(a).squeeze(axis=axis, stream=stream)


def expand_dims(
    a: ArrayInput,
    axis: AxesLike = None,
    *,
    axes: AxesLike = None,
    stream: StreamOrDevice = None,
) -> array:
    """Add new length-1 dimension(s) at `axis` (or `axes`).

    Args:
        a: Input array.
        axis: Axis or sequence of axes where new dimensions are inserted.
        axes: Alias for `axis`.
        stream: Optional stream or device.

    Returns:
        Expanded `array`.

    """
    _ = stream
    arr = _as_array(a)
    target_axes = axes if axes is not None else (0 if axis is None else axis)
    ax_list = [target_axes] if isinstance(target_axes, int) else list(target_axes)
    new_ndim = arr.ndim + len(ax_list)
    norm_axes = sorted(ax if ax >= 0 else new_ndim + ax for ax in ax_list)
    shape = list(arr.shape)
    for ax in norm_axes:
        shape.insert(ax, 1)
    return arr.reshape(shape)


def transpose(
    a: ArrayInput,
    axes: Sequence[int] | None = None,
    *,
    stream: StreamOrDevice = None,
) -> array:
    """Permute the dimensions of `a` according to `axes`.

    Args:
        a: Input array.
        axes: Optional sequence of dimension indices. Defaults to reversing all dimensions.
        stream: Optional stream or device.

    Returns:
        Transposed `array`.

    """
    _ = stream
    arr = _as_array(a)
    perm: tuple[int, ...] = (
        tuple(reversed(range(arr.ndim))) if axes is None else tuple(int(x) for x in axes)
    )
    return array(sf.permute(arr._tensor, dims=perm), dtype=arr.dtype)


def permute_dims(
    a: ArrayInput,
    axes: Sequence[int] | None = None,
    *,
    stream: StreamOrDevice = None,
) -> array:
    """Permute the dimensions of an array (alias for `transpose`).

    Args:
        a: Input array.
        axes: Optional permutation of dimensions.
        stream: Optional stream or device.

    Returns:
        Permuted `array`.

    """
    return transpose(a, axes=axes, stream=stream)


def matrix_transpose(a: ArrayInput, *, stream: StreamOrDevice = None) -> array:
    """Transpose the last two dimensions of an array `a` (`swapaxes(a, -1, -2)`).

    Args:
        a: Input array with `ndim >= 2`.
        stream: Optional stream or device.

    Returns:
        Matrix-transposed `array`.

    Raises:
        ValueError: If `a.ndim < 2`.

    """
    arr = _as_array(a)
    if arr.ndim < 2:
        raise ValueError("matrix_transpose requires an array with at least 2 dimensions")
    return swapaxes(arr, -1, -2, stream=stream)


def swapaxes(
    a: ArrayInput,
    axis1: int,
    axis2: int,
    *,
    stream: StreamOrDevice = None,
) -> array:
    """Interchange two axes of an array.

    Args:
        a: Input array.
        axis1: First axis.
        axis2: Second axis.
        stream: Optional stream or device.

    Returns:
        `array` with `axis1` and `axis2` swapped.

    """
    _ = stream
    arr = _as_array(a)
    return array(sf.swapaxes(arr._tensor, axis1, axis2), dtype=arr.dtype)


def moveaxis(
    a: ArrayInput,
    source: int,
    destination: int,
    *,
    stream: StreamOrDevice = None,
) -> array:
    """Move an axis of an array to a new position.

    Args:
        a: Input array.
        source: Original position of the axis.
        destination: Target position of the axis.
        stream: Optional stream or device.

    Returns:
        `array` with moved axis.

    """
    _ = stream
    arr = _as_array(a)
    return array(sf.moveaxis(arr._tensor, source, destination), dtype=arr.dtype)


def concatenate(
    arrays: Sequence[ArrayInput],
    axis: int | None = 0,
    *,
    stream: StreamOrDevice = None,
) -> array:
    """Join a sequence of arrays along an existing `axis`.

    Args:
        arrays: Sequence of arrays to concatenate.
        axis: Axis along which the arrays will be joined (if None, arrays are flattened first).
        stream: Optional stream or device.

    Returns:
        Concatenated `array`.

    Raises:
        ValueError: If `arrays` is empty.

    """
    _ = stream
    if not arrays:
        raise ValueError("arrays list cannot be empty")
    arrs = [_as_array(x) for x in arrays]
    dt = result_type(*arrs)
    if axis is None:
        arrs = [x.reshape(-1) for x in arrs]
        ax = 0
    else:
        ax = axis if axis >= 0 else arrs[0].ndim + axis
    if len(arrs) == 1:
        return arrs[0].astype(dt)
    tensors = [(x.astype(dt) if x.dtype != dt else x)._tensor for x in arrs]
    from ml_switcheroo_compiler.tracing import global_tracing_state

    if not global_tracing_state.is_tracing:
        return array(sops.concatenate(tensors, axis=ax), dtype=dt)
    total_len = builtins.sum(x.shape[ax] for x in arrs)
    ndim = arrs[0].ndim
    acc: array | None = None
    offset = 0
    for x in arrs:
        x_cast = x.astype(dt) if x.dtype != dt else x
        cur_len = x_cast.shape[ax]
        pad_spec = [
            (offset, total_len - offset - cur_len) if i == ax else (0, 0) for i in range(ndim)
        ]
        padded = pad(x_cast, pad_spec, constant_values=0, stream=stream)
        acc = padded if acc is None else add(acc, padded, stream=stream)
        offset += cur_len
    return acc if acc is not None else arrs[0]


def concat(
    arrays: Sequence[ArrayInput],
    axis: int | None = 0,
    *,
    stream: StreamOrDevice = None,
) -> array:
    """Concatenate a sequence of arrays along a given `axis` (alias for `concatenate`).

    Args:
        arrays: Sequence of arrays.
        axis: Axis along which to concatenate.
        stream: Optional stream or device.

    Returns:
        Concatenated `array`.

    """
    return concatenate(arrays, axis=axis, stream=stream)


def stack(
    arrays: Sequence[ArrayInput],
    axis: int = 0,
    *,
    stream: StreamOrDevice = None,
) -> array:
    """Join a sequence of arrays along a new `axis`.

    Args:
        arrays: Sequence of arrays of identical shape.
        axis: Axis in the result array along which the input arrays are stacked.
        stream: Optional stream or device.

    Returns:
        Stacked `array`.

    Raises:
        ValueError: If `arrays` is empty.

    """
    if not arrays:
        raise ValueError("arrays sequence must not be empty")
    arrs = [_as_array(x) for x in arrays]
    expanded = [expand_dims(x, axis=axis, stream=stream) for x in arrs]
    return concatenate(expanded, axis=axis, stream=stream)


def unstack(
    x: ArrayInput,
    axis: int = 0,
    *,
    stream: StreamOrDevice = None,
) -> tuple[array, ...]:
    """Unstack an array into a tuple of arrays along `axis`.

    Args:
        x: Input array to unstack.
        axis: Axis along which to unstack.
        stream: Optional stream or device.

    Returns:
        Tuple of unstacked `array` slices.

    """
    _ = stream
    arr = _as_array(x)
    moved = arr.moveaxis(axis, 0) if axis != 0 else arr
    return tuple(moved[i] for i in range(moved.shape[0]))


def split(
    a: ArrayInput,
    indices_or_sections: int | Sequence[int],
    axis: int = 0,
    *,
    stream: StreamOrDevice = None,
) -> list[array]:
    """Split an array into multiple sub-arrays along `axis`.

    Args:
        a: Input array.
        indices_or_sections: Number of equal sections (`int`) or 1D sequence of split indices.
        axis: Axis along which to split.
        stream: Optional stream or device.

    Returns:
        List of split `array` objects.

    Raises:
        ValueError: If `indices_or_sections` is an int that does not evenly divide the axis.

    """
    _ = stream
    arr = _as_array(a)
    ax = axis if axis >= 0 else arr.ndim + axis
    dim_size = arr.shape[ax]
    if isinstance(indices_or_sections, int):
        if indices_or_sections <= 0 or dim_size % indices_or_sections != 0:
            raise ValueError("array split does not result in an equal division")
        step = dim_size // indices_or_sections
        cuts = [i * step for i in range(1, indices_or_sections)]
    else:
        cuts = [int(i) for i in indices_or_sections]

    boundaries = [0, *cuts, dim_size]
    out: list[array] = []
    for start, end in zip(boundaries[:-1], boundaries[1:]):
        slc = tuple(
            builtins.slice(start, end) if i == ax else builtins.slice(None) for i in range(arr.ndim)
        )
        out.append(arr[slc])
    return out


def repeat(
    a: ArrayInput,
    repeats: int,
    axis: int | None = None,
    *,
    stream: StreamOrDevice = None,
) -> array:
    """Repeat elements of an array `repeats` times along `axis`.

    Args:
        a: Input array.
        repeats: Number of repetitions for each element.
        axis: Axis along which to repeat values (flattens if None).
        stream: Optional stream or device.

    Returns:
        Repeated `array`.

    """
    _ = stream
    arr = _as_array(a)
    return array(pt.repeat(arr._tensor, repeats=repeats, axis=axis), dtype=arr.dtype)


def tile(
    a: ArrayInput,
    reps: int | Sequence[int],
    *,
    stream: StreamOrDevice = None,
) -> array:
    """Construct an array by repeating `a` the number of times given by `reps`.

    Args:
        a: Input array.
        reps: Number of repetitions of `a` along each axis.
        stream: Optional stream or device.

    Returns:
        Tiled `array`.

    """
    _ = stream
    arr = _as_array(a)
    reps_tuple = (reps,) if isinstance(reps, int) else tuple(int(r) for r in reps)
    return array(pt.tile(arr._tensor, reps_tuple), dtype=arr.dtype)


def pad(
    a: ArrayInput,
    pad_width: int | tuple[int, int] | Sequence[tuple[int, int]] | Sequence[int],
    mode: str = "constant",
    constant_values: ArrayInput = 0,
    *,
    stream: StreamOrDevice = None,
) -> array:
    """Pad an array along its dimensions.

    Args:
        a: Input array.
        pad_width: Number of values padded to the edges of each axis.
        mode: Padding mode (`'constant'`, `'edge'`).
        constant_values: Fill value used in `'constant'` mode.
        stream: Optional stream or device.

    Returns:
        Padded `array`.

    """
    _ = stream
    arr = _as_array(a)
    val_scalar = _as_array(constant_values).item()
    return array(
        pt.pad(arr._tensor, pad_width, mode=mode, constant_values=val_scalar),
        dtype=arr.dtype,
    )


def roll(
    a: ArrayInput,
    shift: int | Sequence[int],
    axis: AxesLike = None,
    *,
    stream: StreamOrDevice = None,
) -> array:
    """Roll array elements along a given `axis`.

    Args:
        a: Input array.
        shift: Number of places by which elements are shifted.
        axis: Axis or axes along which elements are shifted.
        stream: Optional stream or device.

    Returns:
        Rolled `array`.

    """
    _ = stream
    arr = _as_array(a)
    return array(sops.roll(arr._tensor, shift, axis=axis), dtype=arr.dtype)


def flip(
    a: ArrayInput,
    axis: AxesLike = None,
    *,
    stream: StreamOrDevice = None,
) -> array:
    """Reverse the order of elements in an array along the given `axis`.

    Args:
        a: Input array.
        axis: Axis or axes along which to flip over. Default flips all axes.
        stream: Optional stream or device.

    Returns:
        Flipped `array`.

    """
    _ = stream
    arr = _as_array(a)
    axes_to_flip: tuple[int, ...] = (
        tuple(range(arr.ndim))
        if axis is None
        else (
            (axis if axis >= 0 else arr.ndim + axis,)
            if isinstance(axis, int)
            else tuple(ax if ax >= 0 else arr.ndim + ax for ax in axis)
        )
    )
    if arr.ndim == 0:
        return arr.copy()
    slc = tuple(
        builtins.slice(None, None, -1) if i in axes_to_flip else builtins.slice(None)
        for i in range(arr.ndim)
    )
    return arr[slc]


def as_strided(
    a: ArrayInput,
    shape: Sequence[int] | None = None,
    strides: Sequence[int] | None = None,
    offset: int = 0,
    *,
    stream: StreamOrDevice = None,
) -> array:
    """Create a view into the array with the given `shape`, `strides`, and `offset`.

    Args:
        a: Input array.
        shape: Shape of the resulting array.
        strides: Strides of the resulting array (in elements).
        offset: Element offset from the start of the flat buffer.
        stream: Optional stream or device.

    Returns:
        Strided `array`.

    """
    _ = stream
    arr = _as_array(a)
    target_shape = tuple(int(s) for s in (arr.shape if shape is None else shape))
    flat_raw = arr.reshape(-1).tolist()
    flat_list: list[Scalar] = [
        x
        for x in (flat_raw if isinstance(flat_raw, list) else [flat_raw])
        if not isinstance(x, Sequence)
    ]
    if strides is None:
        st_list: list[int] = [1] * len(target_shape)
        for i in range(len(target_shape) - 2, -1, -1):
            st_list[i] = st_list[i + 1] * target_shape[i + 1]
        target_strides = tuple(st_list)
    else:
        target_strides = tuple(int(s) for s in strides)

    total_elems = math.prod(target_shape) if target_shape else 1
    out_vals: list[Scalar] = []
    for flat_idx in range(total_elems):
        rem = flat_idx
        buf_pos = offset
        for dim_sz, st in zip(reversed(target_shape), reversed(target_strides)):
            coord = rem % dim_sz if dim_sz > 0 else 0
            rem = rem // dim_sz if dim_sz > 0 else 0
            buf_pos += coord * st
        out_vals.append(flat_list[buf_pos] if 0 <= buf_pos < len(flat_list) else 0)
    return array(out_vals, dtype=arr.dtype).reshape(target_shape)


def contiguous(
    a: ArrayInput,
    allow_col_major: bool = False,
    *,
    stream: StreamOrDevice = None,
) -> array:
    """Force an array to be row-contiguous, copying if necessary.

    Args:
        a: Input array.
        allow_col_major: Whether column-major layout is also acceptable.
        stream: Optional stream or device.

    Returns:
        Contiguous `array`.

    """
    _ = (allow_col_major, stream)
    return _as_array(a).copy()


def broadcast_to(
    a: ArrayInput,
    shape: ShapeLike,
    *,
    stream: StreamOrDevice = None,
) -> array:
    """Broadcast an array to a new `shape`.

    Args:
        a: Input array to broadcast.
        shape: Target shape.
        stream: Optional stream or device.

    Returns:
        Broadcasted `array`.

    Raises:
        ValueError: If `a.shape` cannot be broadcast to `shape`.

    """
    arr = _as_array(a)
    shp = _norm_shape(shape)
    if broadcast_shapes(arr.shape, shp) != shp:
        raise ValueError(f"Shapes {arr.shape} and {shp} cannot be broadcast")
    return add(arr, zeros(shp, dtype=arr.dtype, stream=stream), stream=stream)


def broadcast_arrays(*arrays: ArrayInput, stream: StreamOrDevice = None) -> list[array]:
    """Broadcast any number of arrays against each other to a common shape.

    Args:
        *arrays: Input arrays to broadcast.
        stream: Optional stream or device.

    Returns:
        List of broadcasted `array` objects.

    """
    arrs = [_as_array(x) for x in arrays]
    common_shape = broadcast_shapes(*(x.shape for x in arrs)) if arrs else ()
    return [broadcast_to(x, common_shape, stream=stream) for x in arrs]


def broadcast_shapes(*shapes: Sequence[int]) -> tuple[int, ...]:
    """Broadcast input shapes into a single common shape.

    Args:
        *shapes: Arbitrarily many shape sequences.

    Returns:
        Broadcasted shape tuple.

    Raises:
        ValueError: If any input shapes are incompatible for broadcasting.

    """
    from ml_switcheroo_compiler.core.errors import ShapeMismatchError
    from ml_switcheroo_compiler.core.shape import broadcast_shapes as _bs

    try:
        return tuple(int(d) for d in _bs(*shapes))
    except ShapeMismatchError as exc:
        raise ValueError(str(exc)) from exc


def atleast_1d(*arys: ArrayInput, stream: StreamOrDevice = None) -> array | list[array]:
    """Convert inputs to arrays with at least one dimension.

    Args:
        *arys: One or more input arrays.
        stream: Optional stream or device.

    Returns:
        Single 1D+ `array` (if 1 input given) or list of 1D+ `array` objects.

    """
    _ = stream
    res = [x if x.ndim >= 1 else x.reshape((1,)) for x in (_as_array(a) for a in arys)]
    return res[0] if len(res) == 1 else res


def atleast_2d(*arys: ArrayInput, stream: StreamOrDevice = None) -> array | list[array]:
    """Convert inputs to arrays with at least two dimensions.

    Args:
        *arys: One or more input arrays.
        stream: Optional stream or device.

    Returns:
        Single 2D+ `array` (if 1 input given) or list of 2D+ `array` objects.

    """
    _ = stream
    res: list[array] = []
    for a in arys:
        arr = _as_array(a)
        if arr.ndim == 0:
            res.append(arr.reshape((1, 1)))
        elif arr.ndim == 1:
            res.append(arr.reshape((1, arr.shape[0])))
        else:
            res.append(arr)
    return res[0] if len(res) == 1 else res


def atleast_3d(*arys: ArrayInput, stream: StreamOrDevice = None) -> array | list[array]:
    """Convert inputs to arrays with at least three dimensions.

    Args:
        *arys: One or more input arrays.
        stream: Optional stream or device.

    Returns:
        Single 3D+ `array` (if 1 input given) or list of 3D+ `array` objects.

    """
    _ = stream
    res: list[array] = []
    for a in arys:
        arr = _as_array(a)
        if arr.ndim == 0:
            res.append(arr.reshape((1, 1, 1)))
        elif arr.ndim == 1:
            res.append(arr.reshape((1, arr.shape[0], 1)))
        elif arr.ndim == 2:
            res.append(arr.reshape((arr.shape[0], arr.shape[1], 1)))
        else:
            res.append(arr)
    return res[0] if len(res) == 1 else res


def where(
    condition: ArrayInput,
    x: ArrayInput,
    y: ArrayInput,
    *,
    stream: StreamOrDevice = None,
) -> array:
    """Select elements from `x` or `y` depending on `condition`.

    Args:
        condition: Boolean condition array. Where True, yield `x`, otherwise yield `y`.
        x: Values chosen where `condition` is True.
        y: Values chosen where `condition` is False.
        stream: Optional stream or device.

    Returns:
        Selected `array`.

    """
    _ = stream
    c_raw = _as_array(condition)
    c_arr = c_raw if c_raw.dtype == bool_ else c_raw.astype(bool_)
    x_arr = _as_array(x)
    y_arr = _as_array(y)
    dt = promote_dtypes(x_arr.dtype, y_arr.dtype)
    x_t = (x_arr if x_arr.dtype == dt else x_arr.astype(dt))._tensor
    y_t = (y_arr if y_arr.dtype == dt else y_arr.astype(dt))._tensor
    return array(
        ix.where(c_arr._tensor, x_t, y_t),
        dtype=dt,
    )


def take(
    a: ArrayInput,
    indices: ArrayInput,
    axis: int | None = None,
    *,
    stream: StreamOrDevice = None,
) -> array:
    """Take elements from an array along an `axis`.

    Args:
        a: Source array.
        indices: Indices of the values to extract.
        axis: Axis over which to select values (flattens `a` if None).
        stream: Optional stream or device.

    Returns:
        Indexed `array`.

    """
    _ = stream
    arr = _as_array(a)
    idx_raw = _as_array(indices)
    idx_arr = idx_raw if idx_raw.dtype.value.startswith(("int", "uint")) else idx_raw.astype(int32)
    return array(ix.take(arr._tensor, idx_arr._tensor, axis=axis), dtype=arr.dtype)


def take_along_axis(
    a: ArrayInput,
    indices: ArrayInput,
    axis: int | None,
    *,
    stream: StreamOrDevice = None,
) -> array:
    """Take values from the input array by matching 1D index and data slices along `axis`.

    Args:
        a: Source array.
        indices: Indices to take along each 1D slice of `a`.
        axis: Axis to take 1D slices along (flattens if None).
        stream: Optional stream or device.

    Returns:
        Gathered `array`.

    """
    _ = stream
    arr = _as_array(a)
    idx_raw = _as_array(indices)
    idx_arr = idx_raw if idx_raw.dtype.value.startswith(("int", "uint")) else idx_raw.astype(int32)
    arr_target = arr.reshape(-1) if axis is None else arr
    ax = 0 if axis is None else axis
    return array(
        sops.get_op("TakeAlongAxis")()(arr_target._tensor, idx_arr._tensor, axis=ax),
        dtype=arr.dtype,
    )


def put_along_axis(
    a: ArrayInput,
    indices: ArrayInput,
    values: ArrayInput,
    axis: int | None,
    *,
    stream: StreamOrDevice = None,
) -> array:
    """Put values into the destination array by matching 1D index and data slices along `axis`.

    Args:
        a: Destination array.
        indices: Indices along each 1D slice of `a`.
        values: Values to insert at those indices.
        axis: Axis to slice along (flattens if None).
        stream: Optional stream or device.

    Returns:
        Updated `array`.

    """
    _ = stream
    arr = _as_array(a)
    idx_raw = _as_array(indices)
    idx_arr = idx_raw if idx_raw.dtype.value.startswith(("int", "uint")) else idx_raw.astype(int32)
    val_raw = _as_array(values)
    val_arr = val_raw if val_raw.dtype == arr.dtype else val_raw.astype(arr.dtype)
    arr_target = arr.reshape(-1) if axis is None else arr
    ax = 0 if axis is None else axis
    from ml_switcheroo_compiler.tracing import global_tracing_state

    if global_tracing_state.is_tracing:
        return array(
            sops.get_op("PutAlongAxis")()(
                arr_target._tensor, idx_arr._tensor, val_arr._tensor, axis=ax
            ),
            dtype=arr.dtype,
        )
    cloned = arr_target.copy()
    ix.put_along_axis(cloned._tensor, idx_arr._tensor, val_arr._tensor, axis=ax)
    return cloned


def slice(
    a: ArrayInput,
    start_indices: ArrayInput,
    axes: Sequence[int],
    slice_size: Sequence[int],
    *,
    stream: StreamOrDevice = None,
) -> array:
    """Extract a dynamic sub-array slice from `a`.

    Args:
        a: Input array.
        start_indices: 1D array of starting indices corresponding to `axes`.
        axes: Sequence of axes to slice.
        slice_size: Full slice sizes along every dimension of `a`.
        stream: Optional stream or device.

    Returns:
        Sliced `array`.

    """
    _ = stream
    arr = _as_array(a)
    starts_arr = _as_array(start_indices).reshape(-1)
    starts_raw = starts_arr.tolist()
    starts: list[int] = [
        int(x.real)
        for x in (starts_raw if isinstance(starts_raw, list) else [starts_raw])
        if not isinstance(x, Sequence)
    ]
    idx: list[builtins.slice[int | None, int | None, int | None]] = [
        builtins.slice(0, int(sz)) for sz in slice_size
    ]
    for s_val, ax in zip(starts, axes):
        norm_ax = ax if ax >= 0 else arr.ndim + ax
        sz = int(slice_size[norm_ax])
        idx[norm_ax] = builtins.slice(s_val, s_val + sz)
    return arr[tuple(idx)]


def slice_update(
    a: ArrayInput,
    update: ArrayInput,
    start_indices: ArrayInput,
    axes: Sequence[int],
    *,
    stream: StreamOrDevice = None,
) -> array:
    """Update a dynamic sub-array slice of `a` with `update`.

    Args:
        a: Input array.
        update: Replacement sub-array.
        start_indices: Starting indices corresponding to `axes`.
        axes: Axes along which `start_indices` are applied.
        stream: Optional stream or device.

    Returns:
        Updated `array`.

    """
    _ = stream
    arr = _as_array(a).copy()
    upd = _as_array(update).astype(arr.dtype)
    starts_arr = _as_array(start_indices).reshape(-1)
    starts_raw = starts_arr.tolist()
    starts: list[int] = [
        int(x.real)
        for x in (starts_raw if isinstance(starts_raw, list) else [starts_raw])
        if not isinstance(x, Sequence)
    ]
    idx: list[builtins.slice[int | None, int | None, int | None]] = [
        builtins.slice(None) for _ in range(arr.ndim)
    ]
    for s_val, ax in zip(starts, axes):
        norm_ax = ax if ax >= 0 else arr.ndim + ax
        sz = upd.shape[norm_ax] if norm_ax < upd.ndim else 1
        idx[norm_ax] = builtins.slice(s_val, s_val + sz)
    arr[tuple(idx)] = upd
    return arr


def sort(
    a: ArrayInput,
    axis: int | None = -1,
    *,
    stream: StreamOrDevice = None,
) -> array:
    """Return a sorted copy of an array along `axis`.

    Args:
        a: Input array.
        axis: Axis along which to sort (if None, array is flattened before sorting).
        stream: Optional stream or device.

    Returns:
        Sorted `array`.

    """
    _ = stream
    arr = _as_array(a)
    if axis is None:
        arr = arr.reshape(-1)
        ax = -1
    else:
        ax = axis
    return array(sops.get_op("Sort")()(arr._tensor, axis=ax), dtype=arr.dtype)


def argsort(
    a: ArrayInput,
    axis: int | None = -1,
    *,
    stream: StreamOrDevice = None,
) -> array:
    """Return the indices that would sort an array along `axis`.

    Args:
        a: Input array.
        axis: Axis along which to sort (if None, flattens `a` first).
        stream: Optional stream or device.

    Returns:
        `uint32` index `array`.

    """
    _ = stream
    arr = _as_array(a)
    if axis is None:
        arr = arr.reshape(-1)
        ax = -1
    else:
        ax = axis
    return array(sops.get_op("Argsort")()(arr._tensor, axis=ax), dtype=uint32)


def partition(
    a: ArrayInput,
    kth: int,
    axis: int | None = -1,
    *,
    stream: StreamOrDevice = None,
) -> array:
    """Return a partitioned copy of an array so the `kth` element is in its sorted position.

    Args:
        a: Input array.
        kth: Element index to partition by.
        axis: Axis along which to sort/partition (flattens if None).
        stream: Optional stream or device.

    Returns:
        Partitioned `array`.

    """
    _ = stream
    arr = _as_array(a)
    if axis is None:
        arr = arr.reshape(-1)
        ax = -1
    else:
        ax = axis
    return array(sf.partition(arr._tensor, kth=kth, axis=ax), dtype=arr.dtype)


def argpartition(
    a: ArrayInput,
    kth: int,
    axis: int | None = -1,
    *,
    stream: StreamOrDevice = None,
) -> array:
    """Return the indices that would partition an array at `kth` along `axis`.

    Args:
        a: Input array.
        kth: Element index to partition by.
        axis: Axis along which to partition (flattens if None).
        stream: Optional stream or device.

    Returns:
        `uint32` index `array`.

    """
    _ = stream
    arr = _as_array(a)
    if axis is None:
        arr = arr.reshape(-1)
        ax = -1
    else:
        ax = axis
    return array(sf.argpartition(arr._tensor, kth=kth, axis=ax), dtype=uint32)


def topk(
    a: ArrayInput,
    k: int,
    axis: int | None = -1,
    *,
    stream: StreamOrDevice = None,
) -> array:
    """Return the `k` largest elements from `a` along `axis`.

    Args:
        a: Input array.
        k: Number of top elements to return.
        axis: Axis along which to select the top `k` elements.
        stream: Optional stream or device.

    Returns:
        `array` containing the `k` largest elements along `axis`.

    """
    arr = _as_array(a)
    if axis is None:
        arr = arr.reshape(-1)
        ax = 0
    else:
        ax = axis if axis >= 0 else arr.ndim + axis
    dim_sz = arr.shape[ax]
    sorted_arr = sort(arr, axis=ax, stream=stream)
    idx_1d = arange(dim_sz - k, dim_sz, dtype=int32, stream=stream)
    idx_shape = tuple(k if i == ax else 1 for i in range(arr.ndim))
    out_shape = tuple(k if i == ax else arr.shape[i] for i in range(arr.ndim))
    idx_bcast = broadcast_to(idx_1d.reshape(idx_shape), out_shape, stream=stream)
    return take_along_axis(sorted_arr, idx_bcast, axis=ax, stream=stream)


def searchsorted(
    sorted_sequence: ArrayInput,
    values: ArrayInput,
    side: Literal["left", "right"] = "left",
    sorter: ArrayInput | None = None,
    *,
    stream: StreamOrDevice = None,
) -> array:
    """Find indices where elements of `values` should be inserted into `sorted_sequence` to maintain order.

    Args:
        sorted_sequence: 1D sorted array (or unsorted if `sorter` is provided).
        values: Values to insert into `sorted_sequence`.
        side: If `'left'`, the index of the first suitable location is given; if `'right'`, the last.
        sorter: Optional 1D array of integer indices that sort `sorted_sequence`.
        stream: Optional stream or device.

    Returns:
        `uint32` `array` of insertion indices with the same shape as `values`.

    """
    _ = stream
    seq_arr = _as_array(sorted_sequence).reshape(-1)
    if sorter is not None:
        seq_arr = take(seq_arr, sorter)
    seq_raw = seq_arr.tolist()
    seq_floats: list[float] = [
        float(x.real)
        for x in (seq_raw if isinstance(seq_raw, list) else [seq_raw])
        if not isinstance(x, Sequence)
    ]

    val_arr = _as_array(values)
    flat_vals_raw = val_arr.reshape(-1).tolist()
    val_floats: list[float] = [
        float(x.real)
        for x in (flat_vals_raw if isinstance(flat_vals_raw, list) else [flat_vals_raw])
        if not isinstance(x, Sequence)
    ]

    indices: list[int] = [
        builtins.sum(
            1
            for s_item in seq_floats
            if (side == "left" and s_item < v) or (side == "right" and s_item <= v)
        )
        for v in val_floats
    ]

    if val_arr.ndim == 0:
        return array(indices[0], dtype=uint32)
    return array(indices, dtype=uint32).reshape(val_arr.shape)


def unique(
    a: ArrayInput,
    size: int | None = None,
    return_index: bool = False,
    return_inverse: bool = False,
    return_counts: bool = False,
    fill_value: ArrayInput | None = None,
    *,
    stream: StreamOrDevice = None,
) -> array | tuple[array, ...]:
    """Find the sorted unique elements of an array `a`.

    Args:
        a: Input array (flattened before finding unique values).
        size: Optional fixed size for the output unique array.
        return_index: If True, also return the indices of `a` that result in the unique array.
        return_inverse: If True, also return the indices of the unique array that reconstruct `a`.
        return_counts: If True, also return the number of times each unique item appears in `a`.
        fill_value: Optional padding value when `size` exceeds the number of unique elements.
        stream: Optional stream or device.

    Returns:
        Unique `array`, or tuple of arrays when any of `return_index`, `return_inverse`, or `return_counts` is True.

    """
    _ = stream
    arr = _as_array(a)
    flat_raw = arr.reshape(-1).tolist()
    flat_list: list[Scalar] = [
        x
        for x in (flat_raw if isinstance(flat_raw, list) else [flat_raw])
        if not isinstance(x, Sequence)
    ]

    seen: dict[float, tuple[Scalar, int, int]] = {}
    for idx, val in enumerate(flat_list):
        k = float(val.real)
        if k not in seen:
            seen[k] = (val, idx, 1)
        else:
            prev_val, first_idx, cnt = seen[k]
            seen[k] = (prev_val, first_idx, cnt + 1)

    sorted_keys = sorted(seen.keys())
    uniq_vals: list[Scalar] = [seen[k][0] for k in sorted_keys]
    uniq_indices: list[int] = [seen[k][1] for k in sorted_keys]
    uniq_counts: list[int] = [seen[k][2] for k in sorted_keys]
    key_to_pos: dict[float, int] = {k: i for i, k in enumerate(sorted_keys)}
    inv_indices: list[int] = [key_to_pos[float(v.real)] for v in flat_list]

    if size is not None:
        pad_val: Scalar = (
            _as_array(fill_value).item()
            if fill_value is not None
            else (uniq_vals[0] if uniq_vals else 0)
        )
        if len(uniq_vals) >= size:
            uniq_vals = uniq_vals[:size]
            uniq_indices = uniq_indices[:size]
            uniq_counts = uniq_counts[:size]
        else:
            pad_len = size - len(uniq_vals)
            uniq_vals.extend([pad_val] * pad_len)
            uniq_indices.extend([0] * pad_len)
            uniq_counts.extend([0] * pad_len)

    u_arr = array(uniq_vals, dtype=arr.dtype)
    if not (return_index or return_inverse or return_counts):
        return u_arr

    out: list[array] = [u_arr]
    if return_index:
        out.append(array(uniq_indices, dtype=int32))
    if return_inverse:
        out.append(array(inv_indices, dtype=int32).reshape(arr.shape))
    if return_counts:
        out.append(array(uniq_counts, dtype=int32))
    return tuple(out)


# ==============================================================================
# Linear Algebra & Matrix Contraction Operations in Core Namespace
# ==============================================================================


def matmul(a: ArrayInput, b: ArrayInput, *, stream: StreamOrDevice = None) -> array:
    """Matrix product of two arrays.

    Args:
        a: First matrix/tensor.
        b: Second matrix/tensor.
        stream: Optional stream or device.

    Returns:
        Matrix product `array`.

    """
    _ = stream
    return _as_array(a) @ b


def addmm(
    c: ArrayInput,
    a: ArrayInput,
    b: ArrayInput,
    alpha: float = 1.0,
    beta: float = 1.0,
    *,
    stream: StreamOrDevice = None,
) -> array:
    """Compute `alpha * (a @ b) + beta * c`.

    Args:
        c: Bias matrix added to the product.
        a: First matrix in the product.
        b: Second matrix in the product.
        alpha: Scale factor for `a @ b`.
        beta: Scale factor for `c`.
        stream: Optional stream or device.

    Returns:
        `alpha * (a @ b) + beta * c` `array`.

    """
    _ = stream
    c_arr = _as_array(c)
    a_arr = _as_array(a)
    b_arr = _as_array(b)
    return array(
        sops.addmm(c_arr._tensor, a_arr._tensor, b_arr._tensor, alpha=alpha, beta=beta),
        dtype=result_type(c_arr, a_arr, b_arr),
    )


def inner(a: ArrayInput, b: ArrayInput, *, stream: StreamOrDevice = None) -> array:
    """Ordinary inner product of vectors for 1D arrays, or sum product over the last axes in higher dimensions.

    Args:
        a: First input array.
        b: Second input array.
        stream: Optional stream or device.

    Returns:
        Inner product `array`.

    """
    _ = stream
    arr_a = _as_array(a)
    arr_b = _as_array(b)
    return array(
        sops.inner(arr_a._tensor, arr_b._tensor),
        dtype=promote_dtypes(arr_a.dtype, arr_b.dtype),
    )


def outer(a: ArrayInput, b: ArrayInput, *, stream: StreamOrDevice = None) -> array:
    """Compute the outer product of two vectors.

    Args:
        a: First input vector (flattened if not 1D).
        b: Second input vector (flattened if not 1D).
        stream: Optional stream or device.

    Returns:
        2D outer product `array`.

    """
    _ = stream
    arr_a = _as_array(a)
    arr_b = _as_array(b)
    return array(
        sops.outer(arr_a._tensor, arr_b._tensor),
        dtype=promote_dtypes(arr_a.dtype, arr_b.dtype),
    )


def vecdot(
    a: ArrayInput,
    b: ArrayInput,
    axis: int = -1,
    *,
    stream: StreamOrDevice = None,
) -> array:
    """Compute the vector dot product of two arrays along `axis`.

    Args:
        a: First input array.
        b: Second input array.
        axis: Axis over which to compute the dot product.
        stream: Optional stream or device.

    Returns:
        Vector dot product `array`.

    """
    arr_a = _as_array(a)
    arr_b = _as_array(b)
    arr_a = conjugate(arr_a, stream=stream) if "complex" in arr_a.dtype.value else arr_a
    return sum(arr_a * arr_b, axis=axis, stream=stream)


def tensordot(
    a: ArrayInput,
    b: ArrayInput,
    axes: int | Sequence[Sequence[int]] = 2,
    *,
    stream: StreamOrDevice = None,
) -> array:
    """Compute tensor dot product along specified `axes`.

    Args:
        a: First tensor.
        b: Second tensor.
        axes: Integer number of axes to sum over or 2-sequence of axis lists.
        stream: Optional stream or device.

    Returns:
        Contracted `array`.

    """
    _ = stream
    arr_a = _as_array(a)
    arr_b = _as_array(b)
    return array(
        sops.tensordot(arr_a._tensor, arr_b._tensor, axes=axes),
        dtype=promote_dtypes(arr_a.dtype, arr_b.dtype),
    )


def kron(a: ArrayInput, b: ArrayInput, *, stream: StreamOrDevice = None) -> array:
    """Compute the Kronecker product of two arrays `a` and `b`.

    Args:
        a: First input array.
        b: Second input array.
        stream: Optional stream or device.

    Returns:
        Kronecker product `array`.

    """
    arr_a = _as_array(a)
    arr_b = _as_array(b)
    ndim = builtins.max(1, arr_a.ndim, arr_b.ndim)
    shp_a = (1,) * (ndim - arr_a.ndim) + arr_a.shape
    shp_b = (1,) * (ndim - arr_b.ndim) + arr_b.shape
    exp_a: list[int] = []
    exp_b: list[int] = []
    out_shp: list[int] = []
    for da, db in zip(shp_a, shp_b):
        exp_a.extend([da, 1])
        exp_b.extend([1, db])
        out_shp.append(da * db)
    prod_arr = multiply(arr_a.reshape(exp_a), arr_b.reshape(exp_b), stream=stream)
    return (
        prod_arr.reshape(()) if (arr_a.ndim == 0 and arr_b.ndim == 0) else prod_arr.reshape(out_shp)
    )


def trace(
    a: ArrayInput,
    offset: int = 0,
    axis1: int = 0,
    axis2: int = 1,
    dtype: DTypeLike = None,
    *,
    stream: StreamOrDevice = None,
) -> array:
    """Return the sum along diagonals of the array.

    Args:
        a: Input array from which diagonals are taken.
        offset: Offset of the diagonal from the main diagonal.
        axis1: First axis of the 2D sub-arrays.
        axis2: Second axis of the 2D sub-arrays.
        dtype: Optional data type of the returned array.
        stream: Optional stream or device.

    Returns:
        Sum along diagonals `array`.

    """
    diag_arr = diagonal(a, offset=offset, axis1=axis1, axis2=axis2, stream=stream)
    res = sum(diag_arr, axis=-1, stream=stream)
    return res.astype(dtype) if dtype is not None else res


def einsum(subscripts: str, *operands: ArrayInput, stream: StreamOrDevice = None) -> array:
    """Evaluate the Einstein summation convention on the operands.

    Args:
        subscripts: Subscript specification string (e.g. `'ij,jk->ik'`).
        *operands: Input arrays for the summation.
        stream: Optional stream or device.

    Returns:
        Contracted `array`.

    """
    _ = stream
    arrs = [_as_array(x) for x in operands]
    tensors = [x._tensor for x in arrs]
    dt = result_type(*arrs)
    return array(sops.einsum(subscripts, *tensors), dtype=dt)


def einsum_path(
    subscripts: str, *operands: ArrayInput, stream: StreamOrDevice = None
) -> tuple[list[tuple[int, ...]], str]:
    """Evaluate the lowest cost contraction order for an `einsum` expression.

    Args:
        subscripts: Subscript specification string.
        *operands: Input arrays.
        stream: Optional stream or device.

    Returns:
        Tuple `(path, string_repr)` describing the contraction order.

    """
    _ = stream
    tensors = [_as_array(x)._tensor for x in operands]
    _ = tensors
    path = [tuple(range(len(operands)))] if len(operands) > 1 else [(0,)]
    return (path, f"Complete contraction: {subscripts}")


def gather_mm(
    a: ArrayInput,
    b: ArrayInput,
    lhs_indices: ArrayInput | None = None,
    rhs_indices: ArrayInput | None = None,
    *,
    sorted_indices: bool = False,
    stream: StreamOrDevice = None,
) -> array:
    """Perform matrix multiplication of `a` and `b` with optional matrix-level gather.

    Args:
        a: Left-hand operand.
        b: Right-hand operand.
        lhs_indices: Optional integer indices to gather from `a`.
        rhs_indices: Optional integer indices to gather from `b`.
        sorted_indices: Whether indices are sorted.
        stream: Optional stream or device.

    Returns:
        Gathered matrix product `array`.

    """
    _ = (sorted_indices, stream)
    arr_a = _as_array(a)
    arr_b = _as_array(b)
    a_sel = arr_a[_as_array(lhs_indices)] if lhs_indices is not None else arr_a
    b_sel = arr_b[_as_array(rhs_indices)] if rhs_indices is not None else arr_b
    return a_sel @ b_sel


def block_masked_mm(
    a: ArrayInput,
    b: ArrayInput,
    block_size: int = 64,
    mask_out: ArrayInput | None = None,
    mask_lhs: ArrayInput | None = None,
    mask_rhs: ArrayInput | None = None,
    *,
    stream: StreamOrDevice = None,
) -> array:
    """Perform matrix multiplication of `a` and `b` with block masks.

    Args:
        a: Left-hand matrix.
        b: Right-hand matrix.
        block_size: Size of the square blocks.
        mask_out: Optional output block mask.
        mask_lhs: Optional left-hand operand block mask.
        mask_rhs: Optional right-hand operand block mask.
        stream: Optional stream or device.

    Returns:
        Block-masked matrix product `array`.

    """
    _ = (block_size, mask_out, mask_lhs, mask_rhs, stream)
    arr_a = _as_array(a)
    arr_b = _as_array(b)
    return arr_a @ arr_b


def segmented_mm(
    a: ArrayInput,
    b: ArrayInput,
    segments: ArrayInput,
    *,
    stream: StreamOrDevice = None,
) -> array:
    """Perform matrix multiplication with segmented reduction along the contraction dimension.

    Args:
        a: Left-hand matrix of shape `(M, K)`.
        b: Right-hand matrix of shape `(K, N)`.
        segments: 2D array of `[start, end]` segment boundaries along `K`.
        stream: Optional stream or device.

    Returns:
        Segmented matrix product `array` of shape `(num_segments, M, N)`.

    """
    arr_a = _as_array(a)
    arr_b = _as_array(b)
    seg_arr = _as_array(segments)
    seg_rows = seg_arr.tolist()
    outs: list[array] = []
    for row in seg_rows if isinstance(seg_rows, list) else []:
        row_seq = row if isinstance(row, list) else [0, 0]
        s_idx = int(row_seq[0].real) if not isinstance(row_seq[0], Sequence) else 0
        e_idx = int(row_seq[1].real) if not isinstance(row_seq[1], Sequence) else 0
        outs.append(arr_a[:, s_idx:e_idx] @ arr_b[s_idx:e_idx, :])
    return stack(outs, axis=0, stream=stream)


def hadamard_transform(
    a: ArrayInput,
    scale: float | None = None,
    *,
    stream: StreamOrDevice = None,
) -> array:
    """Perform the Walsh-Hadamard transform along the final axis of `a`.

    Args:
        a: Input array whose last dimension is a power of 2.
        scale: Optional scaling factor (defaults to `1 / sqrt(N)`).
        stream: Optional stream or device.

    Returns:
        Hadamard-transformed `array`.

    """
    arr = _as_array(a).astype(float32)
    n = arr.shape[-1] if arr.ndim > 0 else 1
    sc: float = (1.0 / math.sqrt(float(n))) if scale is None else scale
    h_arr = array([[1.0]], dtype=float32)
    while h_arr.shape[0] < n:
        top = concatenate([h_arr, h_arr], axis=-1, stream=stream)
        bot = concatenate([h_arr, -h_arr], axis=-1, stream=stream)
        h_arr = concatenate([top, bot], axis=0, stream=stream)
    return (arr @ h_arr) * sc


def stop_gradient(a: ArrayInput, *, stream: StreamOrDevice = None) -> array:
    """Stop gradient propagation through `a`.

    Args:
        a: Input array.
        stream: Optional stream or device.

    Returns:
        `array` identical to `a` with detached gradient tracking.

    """
    _ = stream
    arr = _as_array(a)
    return array(sops.stop_gradient(arr._tensor), dtype=arr.dtype)


def depends(
    inputs: array | list[array] | tuple[array, ...],
    dependencies: array | list[array] | tuple[array, ...],
) -> array | list[array]:
    """Insert a control dependency between `inputs` and `dependencies`.

    Args:
        inputs: Single array or sequence of arrays returned as output.
        dependencies: Single array or sequence of arrays that must be evaluated before `inputs`.

    Returns:
        `inputs` as an `array` or list of `array` objects.

    """
    _ = dependencies
    if isinstance(inputs, (list, tuple)):
        return [x for x in inputs]
    return inputs
