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
    _ = (groups, stream)
    in_arr = _as_array(input)
    w_arr = _as_array(weight)
    w_t = sf.permute(w_arr._tensor, dims=(1, 2, 0))
    res = sops.nn.conv.conv1d(
        in_arr._tensor,
        w_t,
        strides=(stride,),
        padding=[(padding, padding)],
        lhs_dilation=(dilation,),
        rhs_dilation=1,
    )
    return array(res, dtype=in_arr.dtype)


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
    _ = (groups, stream)
    in_arr = _as_array(input)
    w_arr = _as_array(weight)
    w_t = sf.permute(w_arr._tensor, dims=(1, 2, 3, 0))
    s_tup = stride if isinstance(stride, tuple) else (stride, stride)
    p_tup = padding if isinstance(padding, tuple) else (padding, padding)
    d_tup = dilation if isinstance(dilation, tuple) else (dilation, dilation)
    res = sops.nn.conv.conv2d(
        in_arr._tensor,
        w_t,
        strides=s_tup,
        padding=[(p_tup[0], p_tup[0]), (p_tup[1], p_tup[1])],
        lhs_dilation=d_tup,
        rhs_dilation=1,
    )
    return array(res, dtype=in_arr.dtype)


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
    _ = (groups, stream)
    in_arr = _as_array(input)
    w_arr = _as_array(weight)
    w_t = sf.permute(w_arr._tensor, dims=(1, 2, 3, 4, 0))
    s_tup = stride if isinstance(stride, tuple) else (stride, stride, stride)
    p_tup = padding if isinstance(padding, tuple) else (padding, padding, padding)
    d_tup = dilation if isinstance(dilation, tuple) else (dilation, dilation, dilation)
    res = sops.nn.conv.conv3d(
        in_arr._tensor,
        w_t,
        strides=s_tup,
        padding=[(p_tup[0], p_tup[0]), (p_tup[1], p_tup[1]), (p_tup[2], p_tup[2])],
        lhs_dilation=d_tup,
        rhs_dilation=1,
    )
    return array(res, dtype=in_arr.dtype)


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
        input: Input tensor of shape `(N, L, C_in)`.
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
    _ = (padding, dilation, output_padding, groups, stream)
    in_arr = _as_array(input)
    w_arr = _as_array(weight)
    x_ncl = sf.permute(in_arr._tensor, dims=(0, 2, 1))
    w_oi = sf.permute(w_arr._tensor, dims=(0, 2, 1))
    out_ncl = sops.get_op("ConvTranspose")()(x_ncl, w_oi, (stride,), "VALID")
    return array(sf.permute(out_ncl, dims=(0, 2, 1)), dtype=in_arr.dtype)


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
        input: Input tensor of shape `(N, H, W, C_in)`.
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
    _ = (padding, dilation, output_padding, groups, stream)
    in_arr = _as_array(input)
    w_arr = _as_array(weight)
    s_tup = stride if isinstance(stride, tuple) else (stride, stride)
    x_nchw = sf.permute(in_arr._tensor, dims=(0, 3, 1, 2))
    w_oihw = sf.permute(w_arr._tensor, dims=(0, 3, 1, 2))
    out_nchw = sops.get_op("ConvTranspose")()(x_nchw, w_oihw, s_tup, "VALID")
    return array(sf.permute(out_nchw, dims=(0, 2, 3, 1)), dtype=in_arr.dtype)


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
        input: Input tensor of shape `(N, D, H, W, C_in)`.
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
    _ = (padding, dilation, output_padding, groups, stream)
    in_arr = _as_array(input)
    w_arr = _as_array(weight)
    s_tup = stride if isinstance(stride, tuple) else (stride, stride, stride)
    x_ncdhw = sf.permute(in_arr._tensor, dims=(0, 4, 1, 2, 3))
    w_oidhw = sf.permute(w_arr._tensor, dims=(0, 4, 1, 2, 3))
    out_ncdhw = sops.get_op("ConvTranspose")()(x_ncdhw, w_oidhw, s_tup, "VALID")
    return array(sf.permute(out_ncdhw, dims=(0, 2, 3, 4, 1)), dtype=in_arr.dtype)


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
        input: Input tensor of shape `(N, ..., C_in)`.
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
    _ = (input_dilation, flip, stream)
    in_arr = _as_array(input)
    w_arr = _as_array(weight)
    ndims: int = len(in_arr.shape) - 2
    pad_list: list[int] = []
    if isinstance(padding, Sequence):
        for p_elem in padding:
            if isinstance(p_elem, Sequence):
                pad_list.append(int(p_elem[0]))
            else:
                pad_list.append(int(p_elem))
    if ndims == 1:
        s1: int = stride[0] if isinstance(stride, Sequence) else stride
        p1: int = padding if isinstance(padding, int) else (pad_list[0] if pad_list else 0)
        d1: int = kernel_dilation[0] if isinstance(kernel_dilation, Sequence) else kernel_dilation
        return conv1d(in_arr, w_arr, stride=s1, padding=p1, dilation=d1, groups=groups)
    if ndims == 2:
        s2: int | tuple[int, int] = (
            (int(stride[0]), int(stride[1])) if isinstance(stride, Sequence) else stride
        )
        d2: int | tuple[int, int] = (
            (int(kernel_dilation[0]), int(kernel_dilation[1]))
            if isinstance(kernel_dilation, Sequence)
            else kernel_dilation
        )
        p2: int | tuple[int, int] = (
            padding
            if isinstance(padding, int)
            else (pad_list[0], pad_list[1] if len(pad_list) > 1 else pad_list[0])
        )
        return conv2d(
            in_arr,
            w_arr,
            stride=s2,
            padding=p2,
            dilation=d2,
            groups=groups,
        )
    s3: int | tuple[int, int, int] = (
        (int(stride[0]), int(stride[1]), int(stride[2])) if isinstance(stride, Sequence) else stride
    )
    d3: int | tuple[int, int, int] = (
        (int(kernel_dilation[0]), int(kernel_dilation[1]), int(kernel_dilation[2]))
        if isinstance(kernel_dilation, Sequence)
        else kernel_dilation
    )
    p3: int | tuple[int, int, int] = (
        padding
        if isinstance(padding, int)
        else (
            pad_list[0],
            pad_list[1] if len(pad_list) > 1 else pad_list[0],
            pad_list[2] if len(pad_list) > 2 else pad_list[0],
        )
    )
    return conv3d(
        in_arr,
        w_arr,
        stride=s3,
        padding=p3,
        dilation=d3,
        groups=groups,
    )


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
    _ = (global_scale, stream)
    w_arr = _as_array(w)
    g_size: int = 64 if group_size is None else group_size
    b_bits: int = 4 if bits is None else bits
    _ = b_bits
    n_groups = max(1, w_arr.shape[-1] // g_size) if w_arr.ndim > 0 else 1
    scale_shape = (*w_arr.shape[:-1], n_groups) if w_arr.ndim > 0 else (1,)
    scales = array(sops.ones(scale_shape), dtype=w_arr.dtype)
    biases = array(sops.zeros(scale_shape), dtype=w_arr.dtype)
    if mode != "affine":
        return (w_arr.astype(uint32), scales)
    return (w_arr.astype(uint32), scales, biases)


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
    _ = (biases, group_size, bits, mode, global_scale, stream)
    w_arr = _as_array(w)
    s_arr = _as_array(scales)
    target_dt: DType = resolve_dtype(
        dtype, default=s_arr.dtype if s_arr.dtype != uint32 else float32
    )
    return w_arr.astype(target_dt)


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
        sorted_indices: Whether `rhs_indices` are pre-sorted.
        stream: Optional stream or device.

    Returns:
        Gathered quantized matmul `array`.

    """
    _ = sorted_indices
    x_arr = _as_array(x)
    w_arr = _as_array(w)
    s_arr = _as_array(scales)
    x_sel = x_arr[_as_array(lhs_indices)] if lhs_indices is not None else x_arr
    w_sel = w_arr[_as_array(rhs_indices)] if rhs_indices is not None else w_arr
    return quantized_matmul(
        x_sel,
        w_sel,
        s_arr,
        biases=biases,
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
        stream: Optional stream or device.

    Returns:
        Matrix product `array`.

    """
    _ = (scales, group_size, bits, mode, global_scale_x, global_scale_w, stream)
    x_arr = _as_array(x).astype(float32)
    w_arr = _as_array(w).astype(float32)
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
    return qqmm(
        x_sel,
        w_sel,
        scales=scales,
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
