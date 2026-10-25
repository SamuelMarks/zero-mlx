"""Convolution and quantized matrix multiplication operations for zero_mlx."""

from __future__ import annotations

from typing import Sequence

import ml_switcheroo_compiler.ops as sops
import ml_switcheroo_compiler.ops.shape.frontend as sf

from zero_mlx._types import ArrayInput, DTypeLike, StreamOrDevice
from zero_mlx.array import _to_tensor, array
from zero_mlx.dtypes import DType, float32, resolve_dtype, to_switcheroo_dtype, uint32


def _as_array(x: ArrayInput) -> array:
    """Ensure `x` is a `zero_mlx.array`.

    Args:
        x: Input array or scalar/sequence.

    Returns:
        `zero_mlx.array` instance.

    """
    return x if isinstance(x, array) else array(x)


def _dilate_spatial_1d(arr: array, dilation: int, axis: int) -> array:
    """Insert `dilation - 1` zeros between adjacent elements of `arr` along `axis`.

    Args:
        arr: Input `array`.
        dilation: Dilation spacing factor (`>= 1`).
        axis: Spatial axis to dilate.

    Returns:
        Dilated `array`.

    """
    if dilation <= 1 or arr.shape[axis] <= 1:
        return arr
    import zero_mlx.ops as _ops

    n = arr.shape[axis]
    new_len = (n - 1) * dilation + 1
    parts: list[array] = []
    zero_slice = _ops.zeros_like(_ops.take(arr, array([0]), axis=axis))
    for i in range(n):
        parts.append(_ops.take(arr, array([i]), axis=axis))
        if i < n - 1:
            for _ in range(dilation - 1):
                parts.append(zero_slice)
    res = _ops.concatenate(parts, axis=axis)
    return res if res.shape[axis] == new_len else res


def _conv1d_with_pads(
    in_arr: array,
    w_arr: array,
    stride: int,
    pad_pair: tuple[int, int],
    dilation: int,
    groups: int,
) -> array:
    """Execute 1D convolution with explicit `(pad_lo, pad_hi)` and `groups`.

    Args:
        in_arr: Input `array` of shape `(N, L, C_in)` or `(L, C_in)`.
        w_arr: Weight `array` of shape `(C_out, K, C_in / groups)`.
        stride: Convolution stride.
        pad_pair: `(pad_left, pad_right)` padding pair.
        dilation: Kernel dilation.
        groups: Number of channel groups.

    Returns:
        Convolved `array`.

    """
    import zero_mlx.ops as _ops

    unbatched = in_arr.ndim == 2
    x_b = _ops.expand_dims(in_arr, axis=0) if unbatched else in_arr
    if groups <= 1:
        w_t = sf.permute(w_arr._tensor, dims=(1, 2, 0))
        res = sops.nn.conv.conv1d(
            x_b._tensor,
            w_t,
            strides=(stride,),
            padding=[pad_pair],
            lhs_dilation=(dilation,),
            rhs_dilation=1,
        )
        out = array(res, dtype=in_arr.dtype)
        return _ops.squeeze(out, axis=0) if unbatched else out

    c_in_g = x_b.shape[-1] // groups
    c_out_g = w_arr.shape[0] // groups
    group_outs: list[array] = []
    for g in range(groups):
        x_g = x_b[:, :, g * c_in_g : (g + 1) * c_in_g]
        w_g = w_arr[g * c_out_g : (g + 1) * c_out_g]
        w_t = sf.permute(w_g._tensor, dims=(1, 2, 0))
        res_g = sops.nn.conv.conv1d(
            x_g._tensor,
            w_t,
            strides=(stride,),
            padding=[pad_pair],
            lhs_dilation=(dilation,),
            rhs_dilation=1,
        )
        group_outs.append(array(res_g, dtype=in_arr.dtype))
    out_all = _ops.concatenate(group_outs, axis=-1)
    return _ops.squeeze(out_all, axis=0) if unbatched else out_all


def _conv2d_with_pads(
    in_arr: array,
    w_arr: array,
    stride: tuple[int, int],
    pad_pairs: list[tuple[int, int]],
    dilation: tuple[int, int],
    groups: int,
) -> array:
    """Execute 2D convolution with explicit padding pairs and `groups`.

    Args:
        in_arr: Input `array` of shape `(N, H, W, C_in)` or `(H, W, C_in)`.
        w_arr: Weight `array` of shape `(C_out, KH, KW, C_in / groups)`.
        stride: 2-tuple of spatial strides.
        pad_pairs: List of two `(pad_lo, pad_hi)` pairs.
        dilation: 2-tuple of kernel dilations.
        groups: Number of channel groups.

    Returns:
        Convolved `array`.

    """
    import zero_mlx.ops as _ops

    unbatched = in_arr.ndim == 3
    x_b = _ops.expand_dims(in_arr, axis=0) if unbatched else in_arr
    if groups <= 1:
        w_t = sf.permute(w_arr._tensor, dims=(1, 2, 3, 0))
        res = sops.nn.conv.conv2d(
            x_b._tensor,
            w_t,
            strides=stride,
            padding=pad_pairs,
            lhs_dilation=dilation,
            rhs_dilation=1,
        )
        out = array(res, dtype=in_arr.dtype)
        return _ops.squeeze(out, axis=0) if unbatched else out

    c_in_g = x_b.shape[-1] // groups
    c_out_g = w_arr.shape[0] // groups
    group_outs: list[array] = []
    for g in range(groups):
        x_g = x_b[:, :, :, g * c_in_g : (g + 1) * c_in_g]
        w_g = w_arr[g * c_out_g : (g + 1) * c_out_g]
        w_t = sf.permute(w_g._tensor, dims=(1, 2, 3, 0))
        res_g = sops.nn.conv.conv2d(
            x_g._tensor,
            w_t,
            strides=stride,
            padding=pad_pairs,
            lhs_dilation=dilation,
            rhs_dilation=1,
        )
        group_outs.append(array(res_g, dtype=in_arr.dtype))
    out_all = _ops.concatenate(group_outs, axis=-1)
    return _ops.squeeze(out_all, axis=0) if unbatched else out_all


def _conv3d_with_pads(
    in_arr: array,
    w_arr: array,
    stride: tuple[int, int, int],
    pad_pairs: list[tuple[int, int]],
    dilation: tuple[int, int, int],
    groups: int,
) -> array:
    """Execute 3D convolution with explicit padding pairs and `groups`.

    Args:
        in_arr: Input `array` of shape `(N, D, H, W, C_in)` or `(D, H, W, C_in)`.
        w_arr: Weight `array` of shape `(C_out, KD, KH, KW, C_in / groups)`.
        stride: 3-tuple of spatial strides.
        pad_pairs: List of three `(pad_lo, pad_hi)` pairs.
        dilation: 3-tuple of kernel dilations.
        groups: Number of channel groups.

    Returns:
        Convolved `array`.

    """
    import zero_mlx.ops as _ops

    unbatched = in_arr.ndim == 4
    x_b = _ops.expand_dims(in_arr, axis=0) if unbatched else in_arr
    if groups <= 1:
        w_t = sf.permute(w_arr._tensor, dims=(1, 2, 3, 4, 0))
        res = sops.nn.conv.conv3d(
            x_b._tensor,
            w_t,
            strides=stride,
            padding=pad_pairs,
            lhs_dilation=dilation,
            rhs_dilation=1,
        )
        out = array(res, dtype=in_arr.dtype)
        return _ops.squeeze(out, axis=0) if unbatched else out

    c_in_g = x_b.shape[-1] // groups
    c_out_g = w_arr.shape[0] // groups
    group_outs: list[array] = []
    for g in range(groups):
        x_g = x_b[:, :, :, :, g * c_in_g : (g + 1) * c_in_g]
        w_g = w_arr[g * c_out_g : (g + 1) * c_out_g]
        w_t = sf.permute(w_g._tensor, dims=(1, 2, 3, 4, 0))
        res_g = sops.nn.conv.conv3d(
            x_g._tensor,
            w_t,
            strides=stride,
            padding=pad_pairs,
            lhs_dilation=dilation,
            rhs_dilation=1,
        )
        group_outs.append(array(res_g, dtype=in_arr.dtype))
    out_all = _ops.concatenate(group_outs, axis=-1)
    return _ops.squeeze(out_all, axis=0) if unbatched else out_all


def conv1d(
    input: ArrayInput,
    weight: ArrayInput,
    stride: int = 1,
    padding: int = 0,
    dilation: int = 1,
    groups: int = 1,
    *,
    stream: StreamOrDevice = None,
) -> array:
    """Apply 1D convolution over an input with several channels.

    Args:
        input: Input tensor of shape `(N, L, C_in)` or `(L, C_in)`.
        weight: Convolution kernel of shape `(C_out, K, C_in / groups)`.
        stride: Convolution stride.
        padding: Implicit zero-padding on both sides of the input.
        dilation: Kernel dilation spacing.
        groups: Number of blocked connections from input channels to output channels.
        stream: Optional stream or device.

    Returns:
        Convolved `array`.

    """
    _ = stream
    return _conv1d_with_pads(
        _as_array(input),
        _as_array(weight),
        stride=stride,
        pad_pair=(padding, padding),
        dilation=dilation,
        groups=groups,
    )


def conv2d(
    input: ArrayInput,
    weight: ArrayInput,
    stride: int | tuple[int, int] = 1,
    padding: int | tuple[int, int] = 0,
    dilation: int | tuple[int, int] = 1,
    groups: int = 1,
    *,
    stream: StreamOrDevice = None,
) -> array:
    """Apply 2D convolution over an input with several channels.

    Args:
        input: Input tensor of shape `(N, H, W, C_in)` or `(H, W, C_in)`.
        weight: Convolution kernel of shape `(C_out, KH, KW, C_in / groups)`.
        stride: Convolution stride (`int` or 2-tuple).
        padding: Implicit zero-padding (`int` or 2-tuple).
        dilation: Kernel dilation (`int` or 2-tuple).
        groups: Number of channel groups.
        stream: Optional stream or device.

    Returns:
        Convolved `array`.

    """
    _ = stream
    s_tup = stride if isinstance(stride, tuple) else (stride, stride)
    p_tup = padding if isinstance(padding, tuple) else (padding, padding)
    d_tup = dilation if isinstance(dilation, tuple) else (dilation, dilation)
    return _conv2d_with_pads(
        _as_array(input),
        _as_array(weight),
        stride=s_tup,
        pad_pairs=[(p_tup[0], p_tup[0]), (p_tup[1], p_tup[1])],
        dilation=d_tup,
        groups=groups,
    )


def conv3d(
    input: ArrayInput,
    weight: ArrayInput,
    stride: int | tuple[int, int, int] = 1,
    padding: int | tuple[int, int, int] = 0,
    dilation: int | tuple[int, int, int] = 1,
    groups: int = 1,
    *,
    stream: StreamOrDevice = None,
) -> array:
    """Apply 3D convolution over an input with several channels.

    Args:
        input: Input tensor of shape `(N, D, H, W, C_in)` or `(D, H, W, C_in)`.
        weight: Convolution kernel of shape `(C_out, KD, KH, KW, C_in / groups)`.
        stride: Convolution stride (`int` or 3-tuple).
        padding: Implicit zero-padding (`int` or 3-tuple).
        dilation: Kernel dilation (`int` or 3-tuple).
        groups: Number of channel groups.
        stream: Optional stream or device.

    Returns:
        Convolved `array`.

    """
    _ = stream
    s_tup = stride if isinstance(stride, tuple) else (stride, stride, stride)
    p_tup = padding if isinstance(padding, tuple) else (padding, padding, padding)
    d_tup = dilation if isinstance(dilation, tuple) else (dilation, dilation, dilation)
    return _conv3d_with_pads(
        _as_array(input),
        _as_array(weight),
        stride=s_tup,
        pad_pairs=[(p_tup[0], p_tup[0]), (p_tup[1], p_tup[1]), (p_tup[2], p_tup[2])],
        dilation=d_tup,
        groups=groups,
    )


def conv_transpose1d(
    input: ArrayInput,
    weight: ArrayInput,
    stride: int = 1,
    padding: int = 0,
    dilation: int = 1,
    output_padding: int = 0,
    groups: int = 1,
    *,
    stream: StreamOrDevice = None,
) -> array:
    """Apply 1D transposed convolution over an input with several channels.

    Args:
        input: Input tensor of shape `(N, L, C_in)` or `(L, C_in)`.
        weight: Kernel tensor of shape `(C_out, K, C_in / groups)`.
        stride: Convolution stride.
        padding: Input padding.
        dilation: Kernel dilation.
        output_padding: Additional size added to one side of the output shape.
        groups: Number of channel groups.
        stream: Optional stream or device.

    Returns:
        Transposed-convolved `array`.

    """
    _ = stream
    import zero_mlx.ops as _ops

    in_arr = _as_array(input)
    w_arr = _dilate_spatial_1d(_as_array(weight), dilation, axis=1)
    unbatched = in_arr.ndim == 2
    x_b = _ops.expand_dims(in_arr, axis=0) if unbatched else in_arr

    c_in_g = x_b.shape[-1] // max(1, groups)
    c_out_g = w_arr.shape[0] // max(1, groups)
    group_outs: list[array] = []
    for g in range(max(1, groups)):
        x_g = x_b[:, :, g * c_in_g : (g + 1) * c_in_g]
        w_g = w_arr[g * c_out_g : (g + 1) * c_out_g]
        x_ncl = sf.permute(x_g._tensor, dims=(0, 2, 1))
        w_oi = sf.permute(w_g._tensor, dims=(0, 2, 1))
        out_ncl = sops.get_op("ConvTranspose")()(x_ncl, w_oi, (stride,), "VALID")
        group_outs.append(array(sf.permute(out_ncl, dims=(0, 2, 1)), dtype=in_arr.dtype))

    out_arr = group_outs[0] if len(group_outs) == 1 else _ops.concatenate(group_outs, axis=-1)
    if padding > 0:
        out_arr = out_arr[:, padding : out_arr.shape[1] - padding, :]
    if output_padding > 0:
        out_arr = _ops.pad(out_arr, ((0, 0), (0, output_padding), (0, 0)), mode="constant")
    return _ops.squeeze(out_arr, axis=0) if unbatched else out_arr


def conv_transpose2d(
    input: ArrayInput,
    weight: ArrayInput,
    stride: int | tuple[int, int] = 1,
    padding: int | tuple[int, int] = 0,
    dilation: int | tuple[int, int] = 1,
    output_padding: int | tuple[int, int] = 0,
    groups: int = 1,
    *,
    stream: StreamOrDevice = None,
) -> array:
    """Apply 2D transposed convolution over an input with several channels.

    Args:
        input: Input tensor of shape `(N, H, W, C_in)` or `(H, W, C_in)`.
        weight: Kernel tensor of shape `(C_out, KH, KW, C_in / groups)`.
        stride: Convolution stride (`int` or 2-tuple).
        padding: Input padding (`int` or 2-tuple).
        dilation: Kernel dilation (`int` or 2-tuple).
        output_padding: Output padding (`int` or 2-tuple).
        groups: Number of channel groups.
        stream: Optional stream or device.

    Returns:
        Transposed-convolved `array`.

    """
    _ = stream
    import zero_mlx.ops as _ops

    in_arr = _as_array(input)
    s_tup = stride if isinstance(stride, tuple) else (stride, stride)
    p_tup = padding if isinstance(padding, tuple) else (padding, padding)
    d_tup = dilation if isinstance(dilation, tuple) else (dilation, dilation)
    op_tup = (
        output_padding if isinstance(output_padding, tuple) else (output_padding, output_padding)
    )
    w_arr = _dilate_spatial_1d(_as_array(weight), d_tup[0], axis=1)
    w_arr = _dilate_spatial_1d(w_arr, d_tup[1], axis=2)
    unbatched = in_arr.ndim == 3
    x_b = _ops.expand_dims(in_arr, axis=0) if unbatched else in_arr

    c_in_g = x_b.shape[-1] // max(1, groups)
    c_out_g = w_arr.shape[0] // max(1, groups)
    group_outs: list[array] = []
    for g in range(max(1, groups)):
        x_g = x_b[:, :, :, g * c_in_g : (g + 1) * c_in_g]
        w_g = w_arr[g * c_out_g : (g + 1) * c_out_g]
        x_nchw = sf.permute(x_g._tensor, dims=(0, 3, 1, 2))
        w_oihw = sf.permute(w_g._tensor, dims=(0, 3, 1, 2))
        out_nchw = sops.get_op("ConvTranspose")()(x_nchw, w_oihw, s_tup, "VALID")
        group_outs.append(array(sf.permute(out_nchw, dims=(0, 2, 3, 1)), dtype=in_arr.dtype))

    out_arr = group_outs[0] if len(group_outs) == 1 else _ops.concatenate(group_outs, axis=-1)
    if p_tup[0] > 0 or p_tup[1] > 0:
        p0, p1 = p_tup
        out_arr = out_arr[:, p0 : out_arr.shape[1] - p0, p1 : out_arr.shape[2] - p1, :]
    if op_tup[0] > 0 or op_tup[1] > 0:
        out_arr = _ops.pad(
            out_arr,
            ((0, 0), (0, op_tup[0]), (0, op_tup[1]), (0, 0)),
            mode="constant",
        )
    return _ops.squeeze(out_arr, axis=0) if unbatched else out_arr


def conv_transpose3d(
    input: ArrayInput,
    weight: ArrayInput,
    stride: int | tuple[int, int, int] = 1,
    padding: int | tuple[int, int, int] = 0,
    dilation: int | tuple[int, int, int] = 1,
    output_padding: int | tuple[int, int, int] = 0,
    groups: int = 1,
    *,
    stream: StreamOrDevice = None,
) -> array:
    """Apply 3D transposed convolution over an input with several channels.

    Args:
        input: Input tensor of shape `(N, D, H, W, C_in)` or `(D, H, W, C_in)`.
        weight: Kernel tensor of shape `(C_out, KD, KH, KW, C_in / groups)`.
        stride: Convolution stride (`int` or 3-tuple).
        padding: Input padding (`int` or 3-tuple).
        dilation: Kernel dilation (`int` or 3-tuple).
        output_padding: Output padding (`int` or 3-tuple).
        groups: Number of channel groups.
        stream: Optional stream or device.

    Returns:
        Transposed-convolved `array`.

    """
    _ = stream
    import zero_mlx.ops as _ops

    in_arr = _as_array(input)
    s_tup = stride if isinstance(stride, tuple) else (stride, stride, stride)
    p_tup = padding if isinstance(padding, tuple) else (padding, padding, padding)
    d_tup = dilation if isinstance(dilation, tuple) else (dilation, dilation, dilation)
    op_tup = (
        output_padding
        if isinstance(output_padding, tuple)
        else (output_padding, output_padding, output_padding)
    )
    w_arr = _dilate_spatial_1d(_as_array(weight), d_tup[0], axis=1)
    w_arr = _dilate_spatial_1d(w_arr, d_tup[1], axis=2)
    w_arr = _dilate_spatial_1d(w_arr, d_tup[2], axis=3)
    unbatched = in_arr.ndim == 4
    x_b = _ops.expand_dims(in_arr, axis=0) if unbatched else in_arr

    c_in_g = x_b.shape[-1] // max(1, groups)
    c_out_g = w_arr.shape[0] // max(1, groups)
    group_outs: list[array] = []
    for g in range(max(1, groups)):
        x_g = x_b[:, :, :, :, g * c_in_g : (g + 1) * c_in_g]
        w_g = w_arr[g * c_out_g : (g + 1) * c_out_g]
        x_ncdhw = sf.permute(x_g._tensor, dims=(0, 4, 1, 2, 3))
        w_oidhw = sf.permute(w_g._tensor, dims=(0, 4, 1, 2, 3))
        out_ncdhw = sops.get_op("ConvTranspose")()(x_ncdhw, w_oidhw, s_tup, "VALID")
        group_outs.append(array(sf.permute(out_ncdhw, dims=(0, 2, 3, 4, 1)), dtype=in_arr.dtype))

    out_arr = group_outs[0] if len(group_outs) == 1 else _ops.concatenate(group_outs, axis=-1)
    if p_tup[0] > 0 or p_tup[1] > 0 or p_tup[2] > 0:
        p0, p1, p2 = p_tup
        out_arr = out_arr[
            :,
            p0 : out_arr.shape[1] - p0,
            p1 : out_arr.shape[2] - p1,
            p2 : out_arr.shape[3] - p2,
            :,
        ]
    if op_tup[0] > 0 or op_tup[1] > 0 or op_tup[2] > 0:
        out_arr = _ops.pad(
            out_arr,
            ((0, 0), (0, op_tup[0]), (0, op_tup[1]), (0, op_tup[2]), (0, 0)),
            mode="constant",
        )
    return _ops.squeeze(out_arr, axis=0) if unbatched else out_arr


def _normalize_conv_padding(
    padding: int | Sequence[int] | Sequence[Sequence[int]],
    ndims: int,
) -> list[tuple[int, int]]:
    """Normalize `conv_general` padding into `ndims` `(pad_lo, pad_hi)` pairs.

    Args:
        padding: Integer, flat sequence, or nested sequence of `(lo, hi)` pairs.
        ndims: Number of spatial dimensions.

    Returns:
        List of `ndims` `(pad_lo, pad_hi)` integer pairs.

    """
    if isinstance(padding, int):
        return [(padding, padding) for _ in range(ndims)]
    if len(padding) == 2 and all(isinstance(x, int) for x in padding):
        p0 = int(padding[0]) if isinstance(padding[0], int) else 0
        p1 = int(padding[1]) if isinstance(padding[1], int) else 0
        if ndims == 1:
            return [(p0, p1)]
    pairs: list[tuple[int, int]] = []
    for elem in padding:
        if isinstance(elem, Sequence):
            lo = int(elem[0])
            hi = int(elem[1]) if len(elem) > 1 else lo
            pairs.append((lo, hi))
        else:
            v = int(elem)
            pairs.append((v, v))
    while len(pairs) < ndims:
        pairs.append(pairs[-1] if pairs else (0, 0))
    return pairs[:ndims]


def conv_general(
    input: ArrayInput,
    weight: ArrayInput,
    stride: int | Sequence[int] = 1,
    padding: int | Sequence[int] | Sequence[Sequence[int]] = 0,
    kernel_dilation: int | Sequence[int] = 1,
    input_dilation: int | Sequence[int] = 1,
    groups: int = 1,
    flip: bool = False,
    *,
    stream: StreamOrDevice = None,
) -> array:
    """Apply general N-D convolution over an input with several channels.

    Args:
        input: Input tensor of shape `(N, ..., C_in)` or `(..., C_in)`.
        weight: Kernel tensor of shape `(C_out, ..., C_in / groups)`.
        stride: Spatial stride(s).
        padding: Spatial padding specification.
        kernel_dilation: Kernel dilation factor(s).
        input_dilation: Input dilation factor(s).
        groups: Number of channel groups.
        flip: Whether to flip the spatial kernel dimensions.
        stream: Optional stream or device.

    Returns:
        Convolved `array`.

    """
    _ = stream
    import zero_mlx.ops as _ops

    in_arr = _as_array(input)
    w_arr = _as_array(weight)
    ndims: int = w_arr.ndim - 2
    unbatched = in_arr.ndim == ndims + 1
    x_b = _ops.expand_dims(in_arr, axis=0) if unbatched else in_arr

    if flip and ndims >= 1:
        spatial_axes = tuple(range(1, ndims + 1))
        w_arr = _ops.flip(w_arr, axis=spatial_axes)

    in_dil = (
        [input_dilation] * ndims
        if isinstance(input_dilation, int)
        else [int(d) for d in input_dilation]
    )
    for ax_i in range(ndims):
        d_val = in_dil[ax_i] if ax_i < len(in_dil) else in_dil[-1]
        if d_val > 1:
            x_b = _dilate_spatial_1d(x_b, d_val, axis=ax_i + 1)

    pad_pairs = _normalize_conv_padding(padding, ndims)
    if ndims == 1:
        s1: int = int(stride[0]) if isinstance(stride, Sequence) else stride
        d1: int = (
            int(kernel_dilation[0]) if isinstance(kernel_dilation, Sequence) else kernel_dilation
        )
        out = _conv1d_with_pads(
            x_b, w_arr, stride=s1, pad_pair=pad_pairs[0], dilation=d1, groups=groups
        )
        return _ops.squeeze(out, axis=0) if unbatched else out
    if ndims == 2:
        s2: tuple[int, int] = (
            (int(stride[0]), int(stride[1] if len(stride) > 1 else stride[0]))
            if isinstance(stride, Sequence)
            else (stride, stride)
        )
        d2: tuple[int, int] = (
            (
                int(kernel_dilation[0]),
                int(kernel_dilation[1] if len(kernel_dilation) > 1 else kernel_dilation[0]),
            )
            if isinstance(kernel_dilation, Sequence)
            else (kernel_dilation, kernel_dilation)
        )
        out = _conv2d_with_pads(
            x_b, w_arr, stride=s2, pad_pairs=pad_pairs, dilation=d2, groups=groups
        )
        return _ops.squeeze(out, axis=0) if unbatched else out
    s3: tuple[int, int, int] = (
        (
            int(stride[0]),
            int(stride[1] if len(stride) > 1 else stride[0]),
            int(stride[2] if len(stride) > 2 else stride[0]),
        )
        if isinstance(stride, Sequence)
        else (stride, stride, stride)
    )
    d3: tuple[int, int, int] = (
        (
            int(kernel_dilation[0]),
            int(kernel_dilation[1] if len(kernel_dilation) > 1 else kernel_dilation[0]),
            int(kernel_dilation[2] if len(kernel_dilation) > 2 else kernel_dilation[0]),
        )
        if isinstance(kernel_dilation, Sequence)
        else (kernel_dilation, kernel_dilation, kernel_dilation)
    )
    out = _conv3d_with_pads(x_b, w_arr, stride=s3, pad_pairs=pad_pairs, dilation=d3, groups=groups)
    return _ops.squeeze(out, axis=0) if unbatched else out


def convolve(
    a: ArrayInput,
    v: ArrayInput,
    mode: str = "full",
    *,
    stream: StreamOrDevice = None,
) -> array:
    """Compute the discrete linear 1D convolution of `a` and `v`.

    Args:
        a: First 1D input array.
        v: Second 1D filter array.
        mode: Convolution mode (`'full'`, `'valid'`, or `'same'`).
        stream: Optional stream or device.

    Returns:
        1D convolved `array`.

    Raises:
        ValueError: If `mode` is not `'full'`, `'valid'`, or `'same'`, or inputs are not 1D.

    """
    _ = stream
    arr_a = _as_array(a)
    arr_v = _as_array(v)
    if mode not in ("full", "valid", "same"):
        raise ValueError("mode must be one of 'full', 'valid', or 'same'")
    if arr_a.ndim != 1 or arr_v.ndim != 1:
        raise ValueError("convolve expects 1D arrays")

    dt: DType = arr_a.dtype if arr_a.dtype.size >= arr_v.dtype.size else arr_v.dtype
    if arr_a.size == 0 or arr_v.size == 0:
        return array([], dtype=dt)

    if arr_v.size > arr_a.size:
        arr_a, arr_v = arr_v, arr_a

    n: int = arr_a.size
    m: int = arr_v.size
    full_len: int = n + m - 1
    import zero_mlx.ops as _ops

    full_1d = _ops.zeros((full_len,), dtype=float32)
    a_f = arr_a.astype(float32)
    v_f = arr_v.astype(float32)
    for j in range(m):
        pad_left = j
        pad_right = (m - 1) - j
        term = a_f * v_f[j]
        padded_term = (
            _ops.pad(term, ((pad_left, pad_right),), constant_values=0.0)
            if (pad_left > 0 or pad_right > 0)
            else term
        )
        full_1d = full_1d + padded_term

    if mode == "full":
        sliced = full_1d
    elif mode == "valid":
        start = m - 1
        end = n
        sliced = full_1d[start:end]
    else:
        start = (m - 1) // 2
        sliced = full_1d[start : start + n]

    if dt.value.startswith(("int", "uint")):
        return array(sops.round(sliced._tensor), dtype=dt)
    return sliced.astype(dt)


def quantize(
    w: ArrayInput,
    group_size: int | None = None,
    bits: int | None = None,
    mode: str = "affine",
    global_scale: ArrayInput | None = None,
    *,
    stream: StreamOrDevice = None,
) -> tuple[array, array] | tuple[array, array, array]:
    """Quantize a weight matrix `w` using `bits` bits per element and `group_size`.

    Args:
        w: Weight matrix to quantize.
        group_size: Quantization group size along the last dimension.
        bits: Number of bits per quantized element.
        mode: Quantization mode (`'affine'`, `'mxfp4'`, `'nvfp4'`, `'mxfp8'`).
        global_scale: Optional global scale factor.
        stream: Optional stream or device.

    Returns:
        Tuple `(w_q, scales, biases)` for `'affine'` mode or `(w_q, scales)` for non-affine modes.

    """
    _ = stream
    import zero_mlx.ops as _ops

    w_arr = _as_array(w)
    orig_dt = w_arr.dtype if "float" in w_arr.dtype.value else float32
    g_size: int = (
        (32 if mode in ("mxfp4", "nvfp4", "mxfp8") else 64) if group_size is None else group_size
    )
    b_bits: int = (8 if mode == "mxfp8" else 4) if bits is None else bits
    pack_factor: int = max(1, 32 // b_bits)

    if w_arr.ndim == 0 or w_arr.size == 0:
        empty_s = array(sops.ones((1,)), dtype=orig_dt)
        empty_b = array(sops.zeros((1,)), dtype=orig_dt)
        return (
            (w_arr.astype(uint32), empty_s)
            if mode != "affine"
            else (w_arr.astype(uint32), empty_s, empty_b)
        )

    last_dim = w_arr.shape[-1]
    n_groups = max(1, (last_dim + g_size - 1) // g_size)
    padded_last = n_groups * g_size
    w_f = w_arr.astype(float32)
    if global_scale is not None:
        w_f = w_f * _as_array(global_scale).astype(float32)
    if padded_last > last_dim:
        pad_spec = [(0, 0)] * (w_f.ndim - 1) + [(0, padded_last - last_dim)]
        w_f = _ops.pad(w_f, pad_spec, mode="constant")

    lead_shape = w_f.shape[:-1]
    grouped = w_f.reshape((*lead_shape, n_groups, g_size))
    q_max = float((1 << b_bits) - 1)

    if mode == "affine":
        biases_f = _ops.min(grouped, axis=-1)
        w_max = _ops.max(grouped, axis=-1)
        scales_f = (w_max - biases_f) / max(1.0, q_max)
        safe_scales = _ops.where(scales_f == 0.0, array(1.0, dtype=float32), scales_f)
        normed = (grouped - _ops.expand_dims(biases_f, axis=-1)) / _ops.expand_dims(
            safe_scales, axis=-1
        )
        codes = _ops.clip(_ops.round(normed), 0.0, q_max).astype(uint32)
    else:
        biases_f = _ops.zeros((*lead_shape, n_groups), dtype=float32)
        abs_max = _ops.max(_ops.abs(grouped), axis=-1)
        scales_f = _ops.where(abs_max == 0.0, array(1.0, dtype=float32), abs_max / max(1.0, q_max))
        normed = _ops.clip(grouped / _ops.expand_dims(scales_f, axis=-1), 0.0, q_max)
        codes = _ops.round(normed).astype(uint32)

    flat_codes = codes.reshape((*lead_shape, padded_last))
    if padded_last % pack_factor != 0:
        extra = pack_factor - (padded_last % pack_factor)
        pad_c = [(0, 0)] * (flat_codes.ndim - 1) + [(0, extra)]
        flat_codes = _ops.pad(flat_codes, pad_c, mode="constant")
    n_words = flat_codes.shape[-1] // pack_factor
    codes_pack_view = flat_codes.reshape((*lead_shape, n_words, pack_factor))

    packed = _ops.zeros((*lead_shape, n_words), dtype=uint32)
    for k in range(pack_factor):
        c_k = codes_pack_view[..., k]
        shift_mult = int(1 << (k * b_bits))
        packed = packed + (c_k * shift_mult).astype(uint32)

    scales_out = scales_f.astype(orig_dt)
    biases_out = biases_f.astype(orig_dt)
    if mode != "affine":
        return (packed, scales_out)
    return (packed, scales_out, biases_out)


def dequantize(
    w: ArrayInput,
    scales: ArrayInput,
    biases: ArrayInput | None = None,
    group_size: int | None = None,
    bits: int | None = None,
    mode: str = "affine",
    global_scale: ArrayInput | None = None,
    dtype: DTypeLike = None,
    *,
    stream: StreamOrDevice = None,
) -> array:
    """Dequantize a packed weight matrix `w` using `scales` and optional `biases`.

    Args:
        w: Quantized matrix.
        scales: Per-group scale factors.
        biases: Optional per-group biases (used in `'affine'` mode).
        group_size: Quantization group size.
        bits: Number of bits per quantized weight.
        mode: Quantization mode (`'affine'`, `'mxfp4'`, `'nvfp4'`, `'mxfp8'`).
        global_scale: Optional global scale factor.
        dtype: Optional output floating-point `DType`.
        stream: Optional stream or device.

    Returns:
        Dequantized floating-point `array`.

    """
    _ = (mode, stream)
    import zero_mlx.ops as _ops

    w_arr = _as_array(w)
    s_arr = _as_array(scales)
    target_dt: DType = resolve_dtype(
        dtype, default=s_arr.dtype if s_arr.dtype != uint32 else float32
    )
    g_size: int = (
        (32 if mode in ("mxfp4", "nvfp4", "mxfp8") else 64) if group_size is None else group_size
    )
    b_bits: int = (8 if mode == "mxfp8" else 4) if bits is None else bits
    pack_factor: int = max(1, 32 // b_bits)

    if w_arr.ndim == 0 or s_arr.ndim == 0:
        return w_arr.astype(target_dt)

    expected_unpacked = s_arr.shape[-1] * g_size
    if w_arr.dtype == uint32 and w_arr.shape[-1] <= expected_unpacked:
        mask_val = int((1 << b_bits) - 1)
        unpacked_parts: list[array] = []
        w_u32 = w_arr.astype(uint32)
        for k in range(pack_factor):
            divisor = int(1 << (k * b_bits))
            shifted = _ops.floor_divide(w_u32, divisor)
            code_k = _ops.bitwise_and(shifted, mask_val).astype(float32)
            unpacked_parts.append(code_k)
        stacked = _ops.stack(unpacked_parts, axis=-1)
        w_unpacked = stacked.reshape((*w_arr.shape[:-1], w_arr.shape[-1] * pack_factor))
    else:
        w_unpacked = w_arr.astype(float32)

    out_cols = w_unpacked.shape[-1]
    s_rep = _ops.repeat(s_arr.astype(float32), g_size, axis=-1)[..., :out_cols]
    res = w_unpacked * s_rep
    if biases is not None:
        b_arr = _as_array(biases).astype(float32)
        b_rep = _ops.repeat(b_arr, g_size, axis=-1)[..., :out_cols]
        res = res + b_rep
    if global_scale is not None:
        res = res / _as_array(global_scale).astype(float32)
    return res.astype(target_dt)


def quantized_matmul(
    x: ArrayInput,
    w: ArrayInput,
    scales: ArrayInput,
    biases: ArrayInput | None = None,
    transpose: bool = True,
    group_size: int | None = None,
    bits: int | None = None,
    mode: str = "affine",
    *,
    stream: StreamOrDevice = None,
) -> array:
    """Perform matrix multiplication of `x` with quantized weight matrix `w`.

    Args:
        x: Floating-point activation input.
        w: Quantized weight matrix.
        scales: Per-group scale factors.
        biases: Optional per-group biases.
        transpose: Whether `w` is transposed.
        group_size: Quantization group size.
        bits: Bits per quantized element.
        mode: Quantization mode.
        stream: Optional stream or device.

    Returns:
        Matrix multiplication result `array`.

    """
    x_arr = _as_array(x)
    w_deq = dequantize(
        w,
        scales,
        biases=biases,
        group_size=group_size,
        bits=bits,
        mode=mode,
        dtype=x_arr.dtype,
        stream=stream,
    )
    if transpose and w_deq.ndim >= 2 and x_arr.ndim >= 1 and w_deq.shape[-1] > x_arr.shape[-1]:
        w_deq = w_deq[..., : x_arr.shape[-1]]
    return x_arr @ (w_deq.T if transpose else w_deq)


def gather_qmm(
    x: ArrayInput,
    w: ArrayInput,
    scales: ArrayInput,
    biases: ArrayInput | None = None,
    lhs_indices: ArrayInput | None = None,
    rhs_indices: ArrayInput | None = None,
    transpose: bool = True,
    group_size: int | None = None,
    bits: int | None = None,
    mode: str = "affine",
    *,
    global_scale: ArrayInput | None = None,
    sorted_indices: bool = False,
    stream: StreamOrDevice = None,
) -> array:
    """Perform quantized matrix multiplication with matrix-level gather.

    Args:
        x: Input activation tensor.
        w: Quantized weight tensor.
        scales: Quantization scale factors.
        biases: Optional quantization biases.
        lhs_indices: Optional gather indices for `x`.
        rhs_indices: Optional gather indices for `w`.
        transpose: Whether `w` is transposed.
        group_size: Quantization group size.
        bits: Bits per quantized element.
        mode: Quantization mode.
        global_scale: Optional global scale factor.
        sorted_indices: Whether `rhs_indices` are pre-sorted.
        stream: Optional stream or device.

    Returns:
        Gathered quantized matmul `array`.

    """
    _ = (global_scale, sorted_indices)
    x_arr = _as_array(x)
    w_arr = _as_array(w)
    s_arr = _as_array(scales)
    x_sel = x_arr[_as_array(lhs_indices)] if lhs_indices is not None else x_arr
    w_sel = w_arr[_as_array(rhs_indices)] if rhs_indices is not None else w_arr
    s_sel = (
        s_arr[_as_array(rhs_indices)]
        if (rhs_indices is not None and s_arr.shape[0] == w_arr.shape[0])
        else s_arr
    )
    b_sel: ArrayInput | None = (
        _as_array(biases)[_as_array(rhs_indices)]
        if (
            rhs_indices is not None
            and biases is not None
            and _as_array(biases).shape[0] == w_arr.shape[0]
        )
        else biases
    )
    return quantized_matmul(
        x_sel,
        w_sel,
        s_sel,
        biases=b_sel,
        transpose=transpose,
        group_size=group_size,
        bits=bits,
        mode=mode,
        stream=stream,
    )


def qqmm(
    x: ArrayInput,
    w: ArrayInput,
    scales: ArrayInput | None = None,
    group_size: int | None = None,
    bits: int | None = None,
    mode: str = "nvfp4",
    global_scale_x: ArrayInput | None = None,
    global_scale_w: ArrayInput | None = None,
    *,
    global_scale: ArrayInput | None = None,
    stream: StreamOrDevice = None,
) -> array:
    """Perform double-quantized matrix multiplication (`qqmm`) of `x` and `w`.

    Args:
        x: Input activation array.
        w: Weight array (quantized or unquantized).
        scales: Optional quantization scales for `w`.
        group_size: Quantization group size.
        bits: Number of bits per quantized weight.
        mode: Quantization mode (`'nvfp4'` or `'mxfp8'`).
        global_scale_x: Optional global scale for `x`.
        global_scale_w: Optional global scale for `w`.
        global_scale: Optional alias for `global_scale_x`.
        stream: Optional stream or device.

    Returns:
        Matrix product `array`.

    """
    eff_gs_x = global_scale if global_scale is not None else global_scale_x
    x_arr = _as_array(x).astype(float32)
    if eff_gs_x is not None:
        x_arr = x_arr / _as_array(eff_gs_x).astype(float32)
    w_raw = _as_array(w)
    w_arr = (
        dequantize(
            w_raw,
            scales,
            group_size=group_size,
            bits=bits,
            mode=mode,
            global_scale=global_scale_w,
            dtype=float32,
            stream=stream,
        )
        if scales is not None
        else (
            w_raw.astype(float32) / _as_array(global_scale_w).astype(float32)
            if global_scale_w is not None
            else w_raw.astype(float32)
        )
    )
    if w_arr.ndim >= 2 and x_arr.ndim >= 1 and w_arr.shape[-1] > x_arr.shape[-1]:
        w_arr = w_arr[..., : x_arr.shape[-1]]
    w_mat = (
        w_arr.T
        if (
            w_arr.ndim >= 2
            and x_arr.shape[-1] == w_arr.shape[-1]
            and x_arr.shape[-1] != w_arr.shape[-2]
        )
        else w_arr
    )
    return x_arr @ w_mat


def gather_qqmm(
    x: ArrayInput,
    w: ArrayInput,
    scales: ArrayInput | None = None,
    lhs_indices: ArrayInput | None = None,
    rhs_indices: ArrayInput | None = None,
    group_size: int | None = None,
    bits: int | None = None,
    mode: str = "nvfp4",
    global_scale_x: ArrayInput | None = None,
    global_scale_w: ArrayInput | None = None,
    *,
    sorted_indices: bool = False,
    stream: StreamOrDevice = None,
) -> array:
    """Perform double-quantized matrix multiplication with matrix-level gather.

    Args:
        x: Input activation array.
        w: Weight array.
        scales: Optional quantization scales.
        lhs_indices: Optional gather indices for `x`.
        rhs_indices: Optional gather indices for `w`.
        group_size: Quantization group size.
        bits: Bits per quantized weight.
        mode: Quantization mode.
        global_scale_x: Optional global scale for `x`.
        global_scale_w: Optional global scale for `w`.
        sorted_indices: Whether indices are sorted.
        stream: Optional stream or device.

    Returns:
        Gathered matrix product `array`.

    """
    _ = sorted_indices
    x_arr = _as_array(x)
    w_arr = _as_array(w)
    x_sel = x_arr[_as_array(lhs_indices)] if lhs_indices is not None else x_arr
    w_sel = w_arr[_as_array(rhs_indices)] if rhs_indices is not None else w_arr
    s_sel: ArrayInput | None = (
        _as_array(scales)[_as_array(rhs_indices)]
        if (
            rhs_indices is not None
            and scales is not None
            and _as_array(scales).shape[0] == w_arr.shape[0]
        )
        else scales
    )
    return qqmm(
        x_sel,
        w_sel,
        scales=s_sel,
        group_size=group_size,
        bits=bits,
        mode=mode,
        global_scale_x=global_scale_x,
        global_scale_w=global_scale_w,
        stream=stream,
    )


__all__ = [
    "conv1d",
    "conv2d",
    "conv3d",
    "conv_general",
    "conv_transpose1d",
    "conv_transpose2d",
    "conv_transpose3d",
    "convolve",
    "dequantize",
    "gather_qmm",
    "gather_qqmm",
    "qqmm",
    "quantize",
    "quantized_matmul",
]
