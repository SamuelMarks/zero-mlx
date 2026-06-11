from typing import Union, Any, Sequence, Optional
from zero_mlx.array import array
import ml_switcheroo.ops as sops


def _to_tensor(x):
    if isinstance(x, array):
        return x._tensor
    return array(x)._tensor


def arccosh(a: array, /, *, stream: Any = None) -> array:
    """Element-wise inverse hyperbolic cosine."""
    return array(sops.acosh(_to_tensor(a)))


def arctan2(a: array, b: array, /, *, stream: Any = None) -> array:
    """Element-wise inverse tangent of the ratio of two arrays."""
    return array(sops.atan2(_to_tensor(a), _to_tensor(b)))


def bitwise_invert(a: Union[int, array], stream: Any = None) -> array:
    """Element-wise bitwise inverse."""
    return array(sops.bitwise_not(_to_tensor(a)))


def clear_cache() -> None:
    """Clear the memory cache."""
    pass


def concat(arrays: list[array], axis: int = 0, *, stream: Any = None) -> array:
    """Concatenate a list of arrays along a given axis."""
    if not arrays:
        raise ValueError("arrays list cannot be empty")

    tensors = [_to_tensor(a) for a in arrays]
    return array(sops.concatenate(tensors, dim=axis))


def conj(a: array, *, stream: Any = None) -> array:
    """Return the elementwise complex conjugate of the input."""
    return array(sops.conj(_to_tensor(a)))


def contiguous(
    a: array, /, allow_col_major: bool = False, *, stream: Any = None
) -> array:
    """Force an array to be row contiguous. Copy if necessary."""
    return array(_to_tensor(a))


def einsum(subscripts: str, *operands, stream: Any = None) -> array:
    """Perform the Einstein summation convention on the operands."""
    tensors = [_to_tensor(x) for x in operands]
    return array(sops.einsum(subscripts, *tensors))


def einsum_path(subscripts: str, *operands):
    """Compute the contraction order for the given Einstein summation."""
    import numpy as np

    nps = [np.array(x) for x in operands]
    return np.einsum_path(subscripts, *nps)


def flatten(
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


def unflatten(
    a: array, /, axis: int, shape: Sequence[int], *, stream: Any = None
) -> array:
    """Unflatten an axis of an array to a shape."""
    old_shape = list(a.shape)
    if axis < 0:
        axis += len(old_shape)

    new_shape = old_shape[:axis] + list(shape) + old_shape[axis + 1 :]
    return array(sops.reshape(_to_tensor(a), new_shape))


def identity(n: int, dtype: Any = None, *, stream: Any = None) -> array:
    """Create a square identity matrix."""
    from zero_mlx.dtypes import DType

    if dtype is None:
        dtype = DType("float32")
    return array(sops.identity(n, dtype=DType(dtype.value)))


def hadamard_transform(
    a: array, scale: Optional[float] = None, stream: Any = None
) -> array:
    """Perform the Walsh-Hadamard transform along the final axis."""
    raise NotImplementedError("hadamard_transform is not implemented")


def gather_qmm(
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
    raise NotImplementedError("gather_qmm is not implemented")


def quantized_matmul(
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
    raise NotImplementedError("quantized_matmul is not implemented")


def get_active_memory() -> int:
    """Get the actively used memory in bytes."""
    return 0


def get_cache_memory() -> int:
    """Get the cache size in bytes."""
    return 0


def reset_peak_memory() -> None:
    """Reset the peak memory to zero."""
    pass


def set_cache_limit(limit: int) -> int:
    """Set the free cache limit."""
    return limit


def set_memory_limit(limit: int) -> int:
    """Set the memory limit."""
    return limit


def set_wired_limit(limit: int) -> int:
    """Set the wired size limit."""
    return limit


import pathlib


def load(
    file: Union[Any, str, pathlib.Path],
    /,
    format: Optional[str] = None,
    return_metadata: bool = False,
    *,
    stream: Any = None,
) -> Union[array, dict[str, array]]:
    """Load array(s) from a binary file."""
    import numpy as np

    res = np.load(file, allow_pickle=False)
    if isinstance(res, np.ndarray):
        arr = array(res)
        return (arr, {}) if return_metadata else arr
    else:
        dct = {k: array(v) for k, v in res.items()}
        return (dct, {}) if return_metadata else dct


def save(file: Union[Any, str, pathlib.Path], arr: array) -> None:
    """Save the array to a binary file in .npy format."""
    import numpy as np

    np.save(file, np.array(arr))


def save_gguf(
    file: Union[Any, str, pathlib.Path],
    arrays: dict[str, array],
    metadata: dict[str, Union[array, str, list[str]]],
) -> None:
    """Save array(s) to a binary file in .gguf format."""
    raise NotImplementedError("save_gguf is not implemented")


def save_safetensors(
    file: Union[Any, str, pathlib.Path],
    arrays: dict[str, array],
    metadata: Optional[dict[str, str]] = None,
) -> None:
    """Save array(s) to a binary file in .safetensors format."""
    raise NotImplementedError("save_safetensors is not implemented")


def savez(file: Union[Any, str, pathlib.Path], *args, **kwargs) -> None:
    """Save several arrays to a binary file in uncompressed .npz"""
    import numpy as np

    np_args = [np.array(a) for a in args]
    np_kwargs = {k: np.array(v) for k, v in kwargs.items()}
    np.savez(file, *np_args, **np_kwargs)


def savez_compressed(file: Union[Any, str, pathlib.Path], *args, **kwargs) -> None:
    """Save several arrays to a binary file in compressed .npz format."""
    import numpy as np

    np_args = [np.array(a) for a in args]
    np_kwargs = {k: np.array(v) for k, v in kwargs.items()}
    np.savez_compressed(file, *np_args, **np_kwargs)


def import_function(file: str) -> Any:
    """Import a function from a file."""
    raise NotImplementedError("import_function is not implemented")


def set_default_stream(stream: Any) -> None:
    """Set the default stream."""
    from zero_mlx.device import set_default_device

    set_default_device(stream.device)


def permute_dims(
    a: array, /, axes: Optional[Sequence[int]] = None, *, stream: Any = None
) -> array:
    """See :func:transpose."""
    return array(sops.permute(_to_tensor(a), dims=axes))


def slice(
    a: array,
    start_indices: array,
    axes: Sequence[int],
    slice_size: Sequence[int],
    *,
    stream: Any = None,
) -> array:
    """Extract a sub-array from the input array."""
    raise NotImplementedError("slice is not implemented")


def slice_update(
    a: array,
    update: array,
    start_indices: array,
    axes: Sequence[int],
    *,
    stream: Any = None,
) -> array:
    """Update a sub-array of the input array."""
    raise NotImplementedError("slice_update is not implemented")


def tan(a: array, /, *, stream: Any = None) -> array:
    """Element-wise tangent."""
    return array(sops.tan(_to_tensor(a)))


def topk(a: array, /, k: int, axis: Optional[int] = -1, *, stream: Any = None) -> array:
    """Returns the k largest elements from the input along a given axis."""
    raise NotImplementedError("topk is not implemented")


def quantize(
    w: array,
    /,
    group_size: int = 64,
    bits: int = 4,
    mode: str = "affine",
    *,
    stream: Any = None,
) -> tuple[array, array, array]:
    """Quantize the matrix w using bits bits per element."""
    raise NotImplementedError("quantize not implemented")
