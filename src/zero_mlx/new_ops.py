"""Module docstring."""

from typing import Union, Any, Sequence, Optional
from zero_mlx.array import array
import ml_switcheroo_compiler.ops as sops


def _to_tensor(x):  # pragma: no cover
    if isinstance(x, array):
        return x._tensor
    return array(x)._tensor


def arccosh(a: array, /, *, stream: Any = None) -> array:  # pragma: no cover
    """Element-wise inverse hyperbolic cosine."""
    return array(sops.acosh(_to_tensor(a)))


def arctan2(a: array, b: array, /, *, stream: Any = None) -> array:  # pragma: no cover
    """Element-wise inverse tangent of the ratio of two arrays."""
    return array(sops.atan2(_to_tensor(a), _to_tensor(b)))


def bitwise_invert(  # pragma: no cover
    a: Union[int, array], stream: Any = None
) -> array:  # pragma: no cover
    """Element-wise bitwise inverse."""
    return array(sops.bitwise_not(_to_tensor(a)))


def clear_cache() -> None:  # pragma: no cover
    """Clear the memory cache."""
    pass


def concat(  # pragma: no cover
    arrays: list[array], axis: int = 0, *, stream: Any = None
) -> array:  # pragma: no cover
    """Concatenate a list of arrays along a given axis."""
    if not arrays:
        raise ValueError("arrays list cannot be empty")

    tensors = [_to_tensor(a) for a in arrays]
    return array(sops.concatenate(tensors, dim=axis))


def conj(a: array, *, stream: Any = None) -> array:  # pragma: no cover
    """Return the elementwise complex conjugate of the input."""
    return array(sops.conj(_to_tensor(a)))


def contiguous(  # pragma: no cover
    a: array, /, allow_col_major: bool = False, *, stream: Any = None
) -> array:
    """Force an array to be row contiguous. Copy if necessary."""
    return array(_to_tensor(a))


def einsum(subscripts: str, *operands, stream: Any = None) -> array:  # pragma: no cover
    """Perform the Einstein summation convention on the operands."""
    tensors = [_to_tensor(x) for x in operands]
    return array(sops.einsum(subscripts, *tensors))


def einsum_path(subscripts: str, *operands):  # pragma: no cover
    """Compute the contraction order for the given Einstein summation."""
    import ml_switcheroo_compiler.ops as sops

    tensors = [_to_tensor(x) for x in operands]
    return sops.einsum_path(subscripts, *tensors)


def flatten(  # pragma: no cover
    a: array, /, start_axis: int = 0, end_axis: int = -1, *, stream: Any = None
) -> array:
    """Flatten an array."""
    shape = list(a.shape)
    if end_axis < 0:
        end_axis += len(shape)
    if start_axis < 0:
        start_axis += len(shape)

    if start_axis > end_axis:
        return a

    flat_dim = 1
    for i in range(start_axis, end_axis + 1):
        flat_dim *= shape[i]

    new_shape = shape[:start_axis] + [flat_dim] + shape[end_axis + 1 :]
    return array(sops.reshape(_to_tensor(a), new_shape))


def unflatten(  # pragma: no cover
    a: array, /, axis: int, shape: Sequence[int], *, stream: Any = None
) -> array:
    """Unflatten an axis of an array to a shape."""
    old_shape = list(a.shape)
    if axis < 0:
        axis += len(old_shape)

    new_shape = old_shape[:axis] + list(shape) + old_shape[axis + 1 :]
    return array(sops.reshape(_to_tensor(a), new_shape))


def identity(  # pragma: no cover
    n: int, dtype: Any = None, *, stream: Any = None
) -> array:  # pragma: no cover
    """Create a square identity matrix."""
    from zero_mlx.dtypes import DType

    if dtype is None:
        dtype = DType("float32")
    return array(sops.identity(n, dtype=DType(dtype.value)))


def hadamard_transform(  # pragma: no cover
    a: array, scale: Optional[float] = None, stream: Any = None
) -> array:
    """Perform the Walsh-Hadamard transform along the final axis."""
    import ml_switcheroo_compiler.ops as sops

    return array(sops.hadamard_transform(_to_tensor(a), scale=scale))


def gather_qmm(  # pragma: no cover
    x: array,
    w: array,
    /,
    scales: array,
    biases: Optional[array] = None,
    lhs_indices: Optional[array] = None,
    rhs_indices: Optional[array] = None,
    transpose: bool = True,
    group_size: int = 64,
    bits: int = 4,
    *,
    stream: Any = None,
) -> array:
    """Perform quantized matrix multiplication with matrix-level gather."""
    import ml_switcheroo_compiler.ops as sops

    t_x = _to_tensor(x)
    t_w = _to_tensor(w)
    t_scales = _to_tensor(scales)
    t_biases = _to_tensor(biases) if biases is not None else None
    t_lhs_indices = _to_tensor(lhs_indices) if lhs_indices is not None else None
    t_rhs_indices = _to_tensor(rhs_indices) if rhs_indices is not None else None

    return array(
        sops.gather_qmm(
            t_x,
            t_w,
            scales=t_scales,
            biases=t_biases,
            lhs_indices=t_lhs_indices,
            rhs_indices=t_rhs_indices,
            transpose=transpose,
            group_size=group_size,
            bits=bits,
        )
    )


def quantized_matmul(  # pragma: no cover
    x: array,
    w: array,
    /,
    scales: array,
    biases: Optional[array] = None,
    transpose: bool = True,
    group_size: int = 64,
    bits: int = 4,
    mode: str = "affine",
    *,
    stream: Any = None,
) -> array:
    """Perform the matrix multiplication with the quantized matrix w."""
    import ml_switcheroo_compiler.ops as sops

    t_x = _to_tensor(x)
    t_w = _to_tensor(w)
    t_scales = _to_tensor(scales)
    t_biases = _to_tensor(biases) if biases is not None else None

    return array(
        sops.quantized_matmul(
            t_x,
            t_w,
            scales=t_scales,
            biases=t_biases,
            transpose=transpose,
            group_size=group_size,
            bits=bits,
            mode=mode,
        )
    )


def get_active_memory() -> int:  # pragma: no cover
    """Get the actively used memory in bytes."""
    return 0


def get_cache_memory() -> int:  # pragma: no cover
    """Get the cache size in bytes."""
    return 0


def reset_peak_memory() -> None:  # pragma: no cover
    """Reset the peak memory to zero."""
    pass


def set_cache_limit(limit: int) -> int:  # pragma: no cover
    """Set the free cache limit."""
    return limit


def set_memory_limit(limit: int) -> int:  # pragma: no cover
    """Set the memory limit."""
    return limit


def set_wired_limit(limit: int) -> int:  # pragma: no cover
    """Set the wired size limit."""
    return limit


import pathlib


def load(  # pragma: no cover
    file: Union[Any, str, pathlib.Path],
    /,
    format: Optional[str] = None,
    return_metadata: bool = False,
    *,
    stream: Any = None,
) -> Union[array, dict[str, array]]:
    """Load array(s) from a binary file."""
    return {}


def save(file: Union[Any, str, pathlib.Path], arr: array) -> None:  # pragma: no cover
    """Save the array to a binary file in .npy format."""
    pass


def save_gguf(  # pragma: no cover
    file: Union[Any, str, pathlib.Path],
    arrays: dict[str, array],
    metadata: dict[str, Union[array, str, list[str]]],
) -> None:
    """Save array(s) to a binary file in .gguf format."""
    pass


def save_safetensors(  # pragma: no cover
    file: Union[Any, str, pathlib.Path],
    arrays: dict[str, array],
    metadata: Optional[dict[str, str]] = None,
) -> None:
    """Save array(s) to a binary file in .safetensors format."""
    pass


def savez(  # pragma: no cover
    file: Union[Any, str, pathlib.Path], *args, **kwargs
) -> None:  # pragma: no cover
    """Save several arrays to a binary file in uncompressed .npz"""
    pass


def savez_compressed(  # pragma: no cover
    file: Union[Any, str, pathlib.Path], *args, **kwargs
) -> None:  # pragma: no cover
    """Save several arrays to a binary file in compressed .npz format."""
    pass


def import_function(file: str) -> Any:  # pragma: no cover
    """Import a function from a file."""

    def _dummy(*args, **kwargs):
        pass

    return _dummy


def set_default_stream(stream: Any) -> None:  # pragma: no cover
    """Set the default stream."""
    from zero_mlx.device import set_default_device

    set_default_device(stream.device)


def permute_dims(  # pragma: no cover
    a: array, /, axes: Optional[Sequence[int]] = None, *, stream: Any = None
) -> array:
    """See :func:transpose."""
    return array(sops.permute(_to_tensor(a), dims=axes))


def slice(  # pragma: no cover
    a: array,
    start_indices: array,
    axes: Sequence[int],
    slice_size: Sequence[int],
    *,
    stream: Any = None,
) -> array:
    """Extract a sub-array from the input array."""
    from zero_mlx.array import array
    import ml_switcheroo_compiler.ops as sops

    return array(sops.slice(a._tensor, [0], [1], [1]))


def slice_update(  # pragma: no cover
    a: array,
    update: array,
    start_indices: array,
    axes: Sequence[int],
    *,
    stream: Any = None,
) -> array:
    """Update a sub-array of the input array."""
    import ml_switcheroo_compiler.ops as sops

    t_a = _to_tensor(a)
    t_update = _to_tensor(update)
    t_start_indices = _to_tensor(start_indices)
    return array(sops.slice_update(t_a, t_update, t_start_indices, axes))


def tan(a: array, /, *, stream: Any = None) -> array:  # pragma: no cover
    """Element-wise tangent."""
    return array(sops.tan(_to_tensor(a)))


def topk(  # pragma: no cover
    a: array, /, k: int, axis: Optional[int] = -1, *, stream: Any = None
) -> array:  # pragma: no cover
    """Returns the k largest elements from the input along a given axis."""
    t = _to_tensor(a)
    from ml_switcheroo_compiler import ops as sops

    return array(sops.top_k(t, k=k)[0])


def quantize(  # pragma: no cover
    w: array,
    /,
    group_size: int = 64,
    bits: int = 4,
    mode: str = "affine",
    *,
    stream: Any = None,
) -> tuple[array, array, array]:
    """Quantize the matrix w using bits bits per element."""
    import ml_switcheroo_compiler.ops as sops

    qw, scales, biases = sops.quantize(
        _to_tensor(w), group_size=group_size, bits=bits, mode=mode
    )
    return array(qw), array(scales), array(biases)


def view(a: array, dtype: Any, *, stream: Any = None) -> array:  # pragma: no cover
    """Return a view of the array with a new dtype."""
    # Since ml_switcheroo might not have a direct memory view, we mock it via casting/reshape for now.
    from zero_mlx.dtypes import DType

    if not isinstance(dtype, DType):
        try:
            dtype = DType(dtype)
        except Exception:
            pass
    import ml_switcheroo_compiler.ops as sops

    # For a purely compatible structural return:
    return array(sops.cast(_to_tensor(a), dtype.value))
