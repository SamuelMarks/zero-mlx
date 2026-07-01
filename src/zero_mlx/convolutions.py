"""Module docstring."""

from typing import Union, Tuple, Optional, Sequence, Any
from zero_mlx.array import array
from zero_mlx.dtypes import to_switcheroo_dtype
import ml_switcheroo_compiler.ops.nn as mnn
import ml_switcheroo_compiler.ops as sops


def _to_tensor(x):  # pragma: no cover
    if isinstance(x, array):
        return x._tensor
    return sops.array(x)


def conv1d(  # pragma: no cover
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


def conv2d(  # pragma: no cover
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


def conv3d(  # pragma: no cover
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


def conv_transpose1d(  # pragma: no cover
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
        strides=stride,
        padding=padding,
    )
    return array(res)


def conv_transpose2d(  # pragma: no cover
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
        strides=stride,
        padding=padding,
    )
    return array(res)


def conv_transpose3d(  # pragma: no cover
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
        strides=stride,
        padding=padding,
    )
    return array(res)


def conv_general(  # pragma: no cover
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
    from zero_mlx.array import array
    import ml_switcheroo_compiler.ops as sops

    return array(
        sops.linalg.conv_general_dilated(
            input._tensor,
            weight._tensor,
            window_strides=stride,
            padding=padding,
            lhs_dilation=input_dilation,
            rhs_dilation=kernel_dilation,
            feature_group_count=groups,
        )
    )


def convolve(  # pragma: no cover
    a: array, v: array, /, mode: str = "full", *, stream: Any = None
) -> array:  # pragma: no cover
    """The discrete convolution of 1D arrays."""
    import ml_switcheroo_compiler.ops as sops

    dt = a.dtype
    if v.dtype.size > dt.size:
        dt = v.dtype

    if a.size == 0 or v.size == 0:
        return array(sops.zeros((0,), dtype=to_switcheroo_dtype(dt)))

    if v.size > a.size:
        a, v = v, a

    if mode not in ("full", "valid", "same"):
        raise ValueError("mode must be one of 'full', 'valid', or 'same'")

    # v needs to be reversed
    v_rev = v[::-1]

    if mode == "full":
        pad_left = v.size - 1
        pad_right = v.size - 1
    elif mode == "valid":
        pad_left = 0
        pad_right = 0
    elif mode == "same":
        pad_left = v.size // 2
        pad_right = v.size - 1 - pad_left

    a_tensor = sops.astype(a._tensor, to_switcheroo_dtype(dt))
    v_tensor = sops.astype(v_rev._tensor, to_switcheroo_dtype(dt))

    a_reshaped = sops.reshape(a_tensor, (1, a.size, 1))
    v_reshaped = sops.reshape(v_tensor, (v.size, 1, 1))

    res = mnn.conv1d(
        a_reshaped,
        v_reshaped,
        stride=1,
        padding=((pad_left, pad_right),),
        dilation=1,
        groups=1,
    )
    res = sops.squeeze(res)
    if len(res.shape) == 0:
        res = sops.unsqueeze(res, dim=0)
    return array(res)


def dequantize(  # pragma: no cover
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
    return array(w)  # mock return for now
