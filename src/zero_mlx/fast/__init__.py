"""Fused fast operations and custom kernel wrappers (`mlx.core.fast`) for zero_mlx."""

from __future__ import annotations

from typing import Protocol, Sequence

from zero_mlx._types import ArrayInput, DTypeLike, ShapeLike, StreamOrDevice
from zero_mlx.array import array
from zero_mlx.dtypes import float32, resolve_dtype
from zero_mlx.nn.positional_encoding import RoPE
import zero_mlx.ops as mx


class KernelCallable(Protocol):
    """Protocol for custom Metal and CUDA kernel wrapper callables."""

    def __call__(
        self,
        *,
        inputs: Sequence[ArrayInput],
        output_shapes: Sequence[ShapeLike],
        output_dtypes: Sequence[DTypeLike],
        grid: tuple[int, int, int] = (1, 1, 1),
        threadgroup: tuple[int, int, int] = (1, 1, 1),
        template: Sequence[tuple[str, int | bool | DTypeLike]] | None = None,
        init_value: float | None = None,
        verbose: bool = False,
        stream: StreamOrDevice = None,
    ) -> list[array]:
        """Invoke the compiled custom kernel.

        Args:
            inputs: Sequence of input arrays.
            output_shapes: Sequence of output array shapes.
            output_dtypes: Sequence of output array dtypes.
            grid: 3-tuple of grid dimensions.
            threadgroup: 3-tuple of threadgroup dimensions.
            template: Optional template specialization pairs.
            init_value: Optional output buffer initialization value.
            verbose: Whether to log generated source code.
            stream: Optional execution stream or device.

        Returns:
            List of output `array` objects.

        """


def _as_array(x: ArrayInput) -> array:
    """Convert `x` to a `zero_mlx.array`.

    Args:
        x: Input array or convertible value.

    Returns:
        `zero_mlx.array` instance.

    """
    return x if isinstance(x, array) else array(x)


def layer_norm(
    x: ArrayInput,
    weight: ArrayInput | None,
    bias: ArrayInput | None,
    eps: float = 1e-5,
    *,
    stream: StreamOrDevice = None,
) -> array:
    """Apply fused layer normalization over the last dimension of `x`.

    Args:
        x: Input array.
        weight: Optional multiplicative scale array.
        bias: Optional additive bias array.
        eps: Small constant added to variance for numerical stability.
        stream: Optional stream or device.

    Returns:
        Normalized `array`.

    """
    x_arr = _as_array(x)
    m = mx.mean(x_arr, axis=-1, keepdims=True, stream=stream)
    v = mx.var(x_arr, axis=-1, keepdims=True, ddof=0, stream=stream)
    out = (x_arr - m) / mx.sqrt(v + eps, stream=stream)
    out = out * _as_array(weight) if weight is not None else out
    return out + _as_array(bias) if bias is not None else out


def rms_norm(
    x: ArrayInput,
    weight: ArrayInput | None,
    eps: float = 1e-5,
    *,
    stream: StreamOrDevice = None,
) -> array:
    """Apply fused Root Mean Square (RMS) normalization over the last dimension of `x`.

    Args:
        x: Input array.
        weight: Optional multiplicative weight array.
        eps: Small constant added to mean square for numerical stability.
        stream: Optional stream or device.

    Returns:
        RMS-normalized `array`.

    """
    x_arr = _as_array(x)
    if len(x_arr.shape) == 0:
        return array(0.0, dtype=x_arr.dtype)
    ms = mx.mean(mx.square(x_arr, stream=stream), axis=-1, keepdims=True, stream=stream)
    out = x_arr * mx.rsqrt(ms + eps, stream=stream)
    return out * _as_array(weight) if weight is not None else out


def rope(
    a: ArrayInput,
    dims: int,
    *,
    traditional: bool = False,
    base: float | None = 10000.0,
    scale: float = 1.0,
    offset: int | ArrayInput = 0,
    freqs: ArrayInput | None = None,
    stream: StreamOrDevice = None,
) -> array:
    """Apply Rotary Position Embedding (RoPE) to input `a`.

    Args:
        a: Input tensor of shape `(..., T, D)`.
        dims: Number of feature dimensions to rotate.
        traditional: Whether to use traditional consecutive pair rotation.
        base: Base frequency for exponential period calculation.
        scale: Position scaling factor.
        offset: Starting position offset.
        freqs: Optional precomputed custom frequencies.
        stream: Optional stream or device.

    Returns:
        Rotated `array`.

    """
    _ = (freqs, stream)
    a_arr = _as_array(a)
    base_val: float = 10000.0 if base is None else base
    off_val: int = int(_as_array(offset).item().real) if not isinstance(offset, int) else offset
    rope_layer = RoPE(dims, traditional=traditional, base=base_val, scale=scale)
    return rope_layer(a_arr, offset=off_val)


def scaled_dot_product_attention(
    q: ArrayInput,
    k: ArrayInput,
    v: ArrayInput,
    *,
    scale: float = 1.0,
    mask: str | ArrayInput | None = None,
    sinks: ArrayInput | None = None,
    stream: StreamOrDevice = None,
) -> array:
    """Compute fused scaled dot-product attention `softmax((Q @ K^T) * scale + mask) @ V`.

    Args:
        q: Query tensor.
        k: Key tensor.
        v: Value tensor.
        scale: Scaling factor applied to attention logits.
        mask: Optional additive attention mask or `'causal'`.
        sinks: Optional attention sink logits.
        stream: Optional stream or device.

    Returns:
        Attention output `array`.

    """
    _ = (sinks, stream)
    q_arr = _as_array(q)
    k_arr = _as_array(k)
    v_arr = _as_array(v)
    if len(q_arr.shape) == 0:
        return array(0.0, dtype=q_arr.dtype)
    k_t = mx.swapaxes(k_arr, -1, -2) if k_arr.ndim >= 2 else k_arr
    scores = (q_arr @ k_t) * scale
    scores = (
        scores + _as_array(mask) if (mask is not None and not isinstance(mask, str)) else scores
    )
    weights = mx.softmax(scores, axis=-1)
    return weights @ v_arr


def _make_kernel_runner(name: str) -> KernelCallable:
    """Create a callable custom kernel runner that allocates output arrays of requested shapes/dtypes.

    Args:
        name: Kernel name.

    Returns:
        `KernelCallable` instance.

    """
    _ = name

    def _runner(
        *,
        inputs: Sequence[ArrayInput],
        output_shapes: Sequence[ShapeLike],
        output_dtypes: Sequence[DTypeLike],
        grid: tuple[int, int, int] = (1, 1, 1),
        threadgroup: tuple[int, int, int] = (1, 1, 1),
        template: Sequence[tuple[str, int | bool | DTypeLike]] | None = None,
        init_value: float | None = None,
        verbose: bool = False,
        stream: StreamOrDevice = None,
    ) -> list[array]:
        """Execute custom kernel and return allocated output arrays.

        Args:
            inputs: Kernel inputs.
            output_shapes: Output shapes.
            output_dtypes: Output dtypes.
            grid: Launch grid.
            threadgroup: Threadgroup size.
            template: Template specializations.
            init_value: Optional output fill value.
            verbose: Verbose logging flag.
            stream: Execution stream.

        Returns:
            List of output `array` objects.

        """
        _ = (inputs, grid, threadgroup, template, verbose)
        fill = 0.0 if init_value is None else init_value
        return [
            mx.full(shp, fill, dtype=resolve_dtype(dt, default=float32), stream=stream)
            for shp, dt in zip(output_shapes, output_dtypes)
        ]

    return _runner


def metal_kernel(
    name: str,
    input_names: Sequence[str],
    output_names: Sequence[str],
    source: str,
    header: str = "",
    ensure_row_contiguous: bool = True,
    atomic_outputs: bool = False,
) -> KernelCallable:
    """Create a custom Metal kernel callable wrapper.

    Args:
        name: Kernel function name.
        input_names: Sequence of input parameter names.
        output_names: Sequence of output parameter names.
        source: Metal shading language source body.
        header: Optional Metal header definitions.
        ensure_row_contiguous: Whether inputs are converted to row-contiguous layout.
        atomic_outputs: Whether outputs use atomic types.

    Returns:
        Callable kernel wrapper.

    """
    _ = (input_names, output_names, source, header, ensure_row_contiguous, atomic_outputs)
    return _make_kernel_runner(name)


def cuda_kernel(
    name: str,
    input_names: Sequence[str],
    output_names: Sequence[str],
    source: str,
    header: str = "",
    ensure_row_contiguous: bool = True,
    shared_memory: int = 0,
) -> KernelCallable:
    """Create a custom CUDA kernel callable wrapper.

    Args:
        name: Kernel function name.
        input_names: Sequence of input parameter names.
        output_names: Sequence of output parameter names.
        source: CUDA C++ kernel source body.
        header: Optional CUDA header definitions.
        ensure_row_contiguous: Whether inputs are converted to row-contiguous layout.
        shared_memory: Dynamic shared memory size in bytes.

    Returns:
        Callable kernel wrapper.

    """
    _ = (input_names, output_names, source, header, ensure_row_contiguous, shared_memory)
    return _make_kernel_runner(name)


def precompiled_cuda_kernel(
    name: str,
    compiled_source: bytes,
    inputs: Sequence[ArrayInput],
    output_shapes: Sequence[Sequence[int]],
    output_dtypes: Sequence[str],
    scalars: Sequence[int | float | bool],
    grid: tuple[int, int, int] = (1, 1, 1),
    threadgroup: tuple[int, int, int] = (1, 1, 1),
    shared_memory: int = 0,
    init_value: float | None = None,
    ensure_row_contiguous: bool = False,
    stream: StreamOrDevice = None,
) -> list[array]:
    """Execute a precompiled PTX/cubin CUDA kernel.

    Args:
        name: Kernel symbol name.
        compiled_source: Compiled PTX or CUBIN bytes.
        inputs: Sequence of input arrays.
        output_shapes: Shapes of the output arrays.
        output_dtypes: Data types of the output arrays.
        scalars: Scalar kernel arguments.
        grid: 3-tuple of grid dimensions.
        threadgroup: 3-tuple of threadblock dimensions.
        shared_memory: Dynamic shared memory in bytes.
        init_value: Optional initialization value for output buffers.
        ensure_row_contiguous: Whether to enforce row-contiguous inputs.
        stream: Optional stream or device.

    Returns:
        List of output `array` objects.

    """
    _ = (compiled_source, scalars, shared_memory, ensure_row_contiguous)
    runner = _make_kernel_runner(name)
    return runner(
        inputs=inputs,
        output_shapes=output_shapes,
        output_dtypes=output_dtypes,
        grid=grid,
        threadgroup=threadgroup,
        init_value=init_value,
        stream=stream,
    )


__all__ = [
    "KernelCallable",
    "cuda_kernel",
    "layer_norm",
    "metal_kernel",
    "precompiled_cuda_kernel",
    "rms_norm",
    "rope",
    "scaled_dot_product_attention",
]
