from typing import Union, Tuple, Optional, Sequence, Any
from zero_mlx.array import array
import ml_switcheroo.nn as mnn
import ml_switcheroo.ops as sops


def _to_tensor(x):
    if isinstance(x, array):
        return x._tensor
    return sops.array(x)


def conv1d(
    input: array,
    weight: array,
    /,
    stride: int = 1,
    padding: int = 0,
    dilation: int = 1,
    groups: int = 1,
    *,
    stream: Any = None,
) -> array:
    """1D convolution over an input with several channels."""
    res = mnn.conv1d(
        _to_tensor(input),
        _to_tensor(weight),
        stride=stride,
        padding=padding,
        dilation=dilation,
        groups=groups,
    )
    return array(res)


def conv2d(
    input: array,
    weight: array,
    /,
    stride: Union[int, tuple[int, int]] = 1,
    padding: Union[int, tuple[int, int]] = 0,
    dilation: Union[int, tuple[int, int]] = 1,
    groups: int = 1,
    *,
    stream: Any = None,
) -> array:
    """2D convolution over an input with several channels."""
    res = mnn.conv2d(
        _to_tensor(input),
        _to_tensor(weight),
        stride=stride,
        padding=padding,
        dilation=dilation,
        groups=groups,
    )
    return array(res)


def conv3d(
    input: array,
    weight: array,
    /,
    stride: Union[int, tuple[int, int, int]] = 1,
    padding: Union[int, tuple[int, int, int]] = 0,
    dilation: Union[int, tuple[int, int, int]] = 1,
    groups: int = 1,
    *,
    stream: Any = None,
) -> array:
    """3D convolution over an input with several channels."""
    res = mnn.conv3d(
        _to_tensor(input),
        _to_tensor(weight),
        stride=stride,
        padding=padding,
        dilation=dilation,
        groups=groups,
    )
    return array(res)


def conv_transpose1d(
    input: array,
    weight: array,
    /,
    stride: int = 1,
    padding: int = 0,
    dilation: int = 1,
    output_padding: int = 0,
    groups: int = 1,
    *,
    stream: Any = None,
) -> array:
    """1D transposed convolution over an input with several channels."""
    res = mnn.conv_transpose1d(
        _to_tensor(input),
        _to_tensor(weight),
        stride=stride,
        padding=padding,
        output_padding=output_padding,
        groups=groups,
        dilation=dilation,
    )
    return array(res)


def conv_transpose2d(
    input: array,
    weight: array,
    /,
    stride: Union[int, Tuple[int, int]] = 1,
    padding: Union[int, Tuple[int, int]] = 0,
    dilation: Union[int, Tuple[int, int]] = 1,
    output_padding: Union[int, Tuple[int, int]] = 0,
    groups: int = 1,
    *,
    stream: Any = None,
) -> array:
    """2D transposed convolution over an input with several channels."""
    res = mnn.conv_transpose2d(
        _to_tensor(input),
        _to_tensor(weight),
        stride=stride,
        padding=padding,
        output_padding=output_padding,
        groups=groups,
        dilation=dilation,
    )
    return array(res)


def conv_transpose3d(
    input: array,
    weight: array,
    /,
    stride: Union[int, Tuple[int, int, int]] = 1,
    padding: Union[int, Tuple[int, int, int]] = 0,
    dilation: Union[int, Tuple[int, int, int]] = 1,
    output_padding: Union[int, Tuple[int, int, int]] = 0,
    groups: int = 1,
    *,
    stream: Any = None,
) -> array:
    """3D transposed convolution over an input with several channels."""
    res = mnn.conv_transpose3d(
        _to_tensor(input),
        _to_tensor(weight),
        stride=stride,
        padding=padding,
        output_padding=output_padding,
        groups=groups,
        dilation=dilation,
    )
    return array(res)


def conv_general(
    input: array,
    weight: array,
    /,
    stride: Union[int, Sequence[int]] = 1,
    padding: Union[int, Sequence[int], tuple[Sequence[int], Sequence[int]]] = 0,
    kernel_dilation: Union[int, Sequence[int]] = 1,
    input_dilation: Union[int, Sequence[int]] = 1,
    groups: int = 1,
    flip: bool = False,
    *,
    stream: Any = None,
) -> array:
    """General convolution over an input with several channels."""
    # ml_switcheroo doesn't have a direct conv_general. We simulate it with the specific ones
    ndims = len(input.shape) - 2
    if ndims == 1:
        return conv1d(
            input,
            weight,
            stride=stride,
            padding=padding,
            dilation=kernel_dilation,
            groups=groups,
        )
    elif ndims == 2:
        return conv2d(
            input,
            weight,
            stride=stride,
            padding=padding,
            dilation=kernel_dilation,
            groups=groups,
        )
    elif ndims == 3:
        return conv3d(
            input,
            weight,
            stride=stride,
            padding=padding,
            dilation=kernel_dilation,
            groups=groups,
        )
    raise NotImplementedError(f"conv_general not implemented for {ndims}D")


def convolve(a: array, v: array, /, mode: str = "full", *, stream: Any = None) -> array:
    """The discrete convolution of 1D arrays."""
    # A simple numpy wrapper since this is a 1D general convolve
    import numpy as np
    import ml_switcheroo.core.tensor as tensor

    res = np.convolve(np.array(a), np.array(v), mode=mode)
    dt = a.dtype
    # promote dtype
    if v.dtype.size > dt.size:
        dt = v.dtype
    return array(res, dtype=dt)


def dequantize(
    w: array,
    /,
    scales: array,
    biases: Optional[array] = None,
    group_size: int = 64,
    bits: int = 4,
    mode: str = "affine",
    *,
    stream: Any = None,
) -> array:
    """Dequantize the matrix w using quantization parameters."""
    raise NotImplementedError("dequantize is not implemented")
