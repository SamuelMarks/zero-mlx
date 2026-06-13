from typing import Union, Tuple, Optional, Sequence, Any
from zero_mlx.array import array
from zero_mlx.dtypes import to_switcheroo_dtype
import ml_switcheroo_compiler.nn as mnn
import ml_switcheroo_compiler.ops as sops


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
    import ml_switcheroo_compiler.ops as sops

    dt = a.dtype
    if v.dtype.size > dt.size:
        dt = v.dtype

    if mode != "full":
        raise NotImplementedError(
            "Only mode='full' is currently supported for convolve"
        )

    if a.size == 0 or v.size == 0:
        return array(sops.zeros((0,), dtype=to_switcheroo_dtype(dt)))

    # v needs to be reversed
    v_rev = v[::-1]

    # Pad a
    pad_width = v.size - 1
    # We must pad `a` on both sides. sops.pad(tensor, [(pad_left, pad_right)])
    # Wait, does sops have pad? Yes! sops.pad(tensor, pad_width)
    a_tensor = sops.astype(a._tensor, to_switcheroo_dtype(dt))
    v_tensor = sops.astype(v_rev._tensor, to_switcheroo_dtype(dt))

    a_reshaped = sops.reshape(a_tensor, (1, a.size, 1))
    v_reshaped = sops.reshape(v_tensor, (1, v.size, 1))

    # We use conv1d and let it pad, or pad manually.
    # mnn.conv1d uses stride 1 and padding
    res = mnn.conv1d(
        a_reshaped,
        v_reshaped,
        stride=1,
        padding=pad_width,
        dilation=1,
        groups=1,
    )
    res = sops.squeeze(res)
    if res.ndim == 0:
        res = sops.unsqueeze(res, dim=0)
    return array(res)


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
