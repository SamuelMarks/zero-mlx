"""Fast Fourier Transform operations (`mlx.core.fft`) for zero_mlx."""

from __future__ import annotations

import math
from typing import Sequence

import ml_switcheroo_compiler.ops.signal as sig

from zero_mlx._types import ArrayInput, AxesLike, StreamOrDevice
from zero_mlx.array import array
from zero_mlx.dtypes import complex64, float32


def _as_array(x: ArrayInput) -> array:
    """Convert `x` to a `zero_mlx.array` if not already one.

    Args:
        x: Input array or sequence/scalar.

    Returns:
        `zero_mlx.array` instance.

    """
    return x if isinstance(x, array) else array(x)


def fft(
    a: ArrayInput,
    n: int | None = None,
    axis: int = -1,
    *,
    norm: str | None = None,
    stream: StreamOrDevice = None,
) -> array:
    """Compute the one-dimensional discrete Fourier Transform.

    Args:
        a: Input array.
        n: Length of the transformed axis of the output.
        axis: Axis over which to compute the FFT.
        norm: Optional normalization mode (`'backward'`, `'ortho'`, `'forward'`).
        stream: Optional stream or device.

    Returns:
        Complex output `array` of dtype `complex64`.

    """
    _ = (norm, stream)
    arr = _as_array(a)
    return array(sig.fft(arr._tensor, n=n, axis=axis), dtype=complex64)


def ifft(
    a: ArrayInput,
    n: int | None = None,
    axis: int = -1,
    *,
    norm: str | None = None,
    stream: StreamOrDevice = None,
) -> array:
    """Compute the one-dimensional inverse discrete Fourier Transform.

    Args:
        a: Input array.
        n: Length of the transformed axis of the output.
        axis: Axis over which to compute the inverse FFT.
        norm: Optional normalization mode (`'backward'`, `'ortho'`, `'forward'`).
        stream: Optional stream or device.

    Returns:
        Complex output `array` of dtype `complex64`.

    """
    _ = (norm, stream)
    arr = _as_array(a)
    return array(sig.ifft(arr._tensor, n=n, axis=axis), dtype=complex64)


def fft2(
    a: ArrayInput,
    s: Sequence[int] | None = None,
    axes: Sequence[int] | None = (-2, -1),
    *,
    norm: str | None = None,
    stream: StreamOrDevice = None,
) -> array:
    """Compute the two-dimensional discrete Fourier Transform.

    Args:
        a: Input array.
        s: Shape (length of each transformed axis) of the output.
        axes: Axes over which to compute the 2D FFT.
        norm: Optional normalization mode (`'backward'`, `'ortho'`, `'forward'`).
        stream: Optional stream or device.

    Returns:
        Complex output `array` of dtype `complex64`.

    """
    _ = (norm, stream)
    arr = _as_array(a)
    return array(sig.Fft2()(arr._tensor, s=s, axes=axes), dtype=complex64)


def ifft2(
    a: ArrayInput,
    s: Sequence[int] | None = None,
    axes: Sequence[int] | None = (-2, -1),
    *,
    norm: str | None = None,
    stream: StreamOrDevice = None,
) -> array:
    """Compute the two-dimensional inverse discrete Fourier Transform.

    Args:
        a: Input array.
        s: Shape (length of each transformed axis) of the output.
        axes: Axes over which to compute the 2D inverse FFT.
        norm: Optional normalization mode (`'backward'`, `'ortho'`, `'forward'`).
        stream: Optional stream or device.

    Returns:
        Complex output `array` of dtype `complex64`.

    """
    _ = (norm, stream)
    arr = _as_array(a)
    return array(sig.ifft2(arr._tensor, s=s, axes=axes), dtype=complex64)


def fftn(
    a: ArrayInput,
    s: Sequence[int] | None = None,
    axes: Sequence[int] | None = None,
    *,
    norm: str | None = None,
    stream: StreamOrDevice = None,
) -> array:
    """Compute the N-dimensional discrete Fourier Transform.

    Args:
        a: Input array.
        s: Shape (length of each transformed axis) of the output.
        axes: Axes over which to compute the N-D FFT.
        norm: Optional normalization mode (`'backward'`, `'ortho'`, `'forward'`).
        stream: Optional stream or device.

    Returns:
        Complex output `array` of dtype `complex64`.

    """
    _ = (norm, stream)
    arr = _as_array(a)
    return array(sig.fftn(arr._tensor, s=s, axes=axes), dtype=complex64)


def ifftn(
    a: ArrayInput,
    s: Sequence[int] | None = None,
    axes: Sequence[int] | None = None,
    *,
    norm: str | None = None,
    stream: StreamOrDevice = None,
) -> array:
    """Compute the N-dimensional inverse discrete Fourier Transform.

    Args:
        a: Input array.
        s: Shape (length of each transformed axis) of the output.
        axes: Axes over which to compute the N-D inverse FFT.
        norm: Optional normalization mode (`'backward'`, `'ortho'`, `'forward'`).
        stream: Optional stream or device.

    Returns:
        Complex output `array` of dtype `complex64`.

    """
    _ = (norm, stream)
    arr = _as_array(a)
    return array(sig.ifftn(arr._tensor, s=s, axes=axes), dtype=complex64)


def rfft(
    a: ArrayInput,
    n: int | None = None,
    axis: int = -1,
    *,
    norm: str | None = None,
    stream: StreamOrDevice = None,
) -> array:
    """Compute the one-dimensional discrete Fourier Transform for real input.

    Args:
        a: Real-valued input array.
        n: Number of points along transformation axis in the input to use.
        axis: Axis over which to compute the RFFT.
        norm: Optional normalization mode (`'backward'`, `'ortho'`, `'forward'`).
        stream: Optional stream or device.

    Returns:
        Complex output `array` of dtype `complex64`.

    """
    _ = (norm, stream)
    arr = _as_array(a)
    return array(sig.Rfft()(arr._tensor, n=n, axis=axis), dtype=complex64)


def irfft(
    a: ArrayInput,
    n: int | None = None,
    axis: int = -1,
    *,
    norm: str | None = None,
    stream: StreamOrDevice = None,
) -> array:
    """Compute the inverse of `rfft` (1D real inverse FFT).

    Args:
        a: Complex-valued input array.
        n: Length of the transformed axis of the output.
        axis: Axis over which to compute the IRFFT.
        norm: Optional normalization mode (`'backward'`, `'ortho'`, `'forward'`).
        stream: Optional stream or device.

    Returns:
        Real output `array` of dtype `float32`.

    """
    _ = (norm, stream)
    arr = _as_array(a)
    return array(sig.Irfft()(arr._tensor, n=n, axis=axis), dtype=float32)


def rfft2(
    a: ArrayInput,
    s: Sequence[int] | None = None,
    axes: Sequence[int] | None = (-2, -1),
    *,
    norm: str | None = None,
    stream: StreamOrDevice = None,
) -> array:
    """Compute the two-dimensional discrete Fourier Transform for real input.

    Args:
        a: Real-valued input array.
        s: Shape of the FFT.
        axes: Axes over which to compute the 2D RFFT.
        norm: Optional normalization mode (`'backward'`, `'ortho'`, `'forward'`).
        stream: Optional stream or device.

    Returns:
        Complex output `array` of dtype `complex64`.

    """
    _ = (norm, stream)
    arr = _as_array(a)
    return array(sig.rfft2(arr._tensor, s=s, axes=axes), dtype=complex64)


def irfft2(
    a: ArrayInput,
    s: Sequence[int] | None = None,
    axes: Sequence[int] | None = (-2, -1),
    *,
    norm: str | None = None,
    stream: StreamOrDevice = None,
) -> array:
    """Compute the inverse of `rfft2` (2D real inverse FFT).

    Args:
        a: Complex-valued input array.
        s: Shape of the real output.
        axes: Axes over which to compute the 2D IRFFT.
        norm: Optional normalization mode (`'backward'`, `'ortho'`, `'forward'`).
        stream: Optional stream or device.

    Returns:
        Real output `array` of dtype `float32`.

    """
    _ = (norm, stream)
    arr = _as_array(a)
    return array(sig.irfft2(arr._tensor, s=s, axes=axes), dtype=float32)


def rfftn(
    a: ArrayInput,
    s: Sequence[int] | None = None,
    axes: Sequence[int] | None = None,
    *,
    norm: str | None = None,
    stream: StreamOrDevice = None,
) -> array:
    """Compute the N-dimensional discrete Fourier Transform for real input.

    Args:
        a: Real-valued input array.
        s: Shape of the FFT.
        axes: Axes over which to compute the N-D RFFT.
        norm: Optional normalization mode (`'backward'`, `'ortho'`, `'forward'`).
        stream: Optional stream or device.

    Returns:
        Complex output `array` of dtype `complex64`.

    """
    _ = (norm, stream)
    arr = _as_array(a)
    return array(sig.rfftn(arr._tensor, s=s, axes=axes), dtype=complex64)


def irfftn(
    a: ArrayInput,
    s: Sequence[int] | None = None,
    axes: Sequence[int] | None = None,
    *,
    norm: str | None = None,
    stream: StreamOrDevice = None,
) -> array:
    """Compute the inverse of `rfftn` (N-dimensional real inverse FFT).

    Args:
        a: Complex-valued input array.
        s: Shape of the real output.
        axes: Axes over which to compute the N-D IRFFT.
        norm: Optional normalization mode (`'backward'`, `'ortho'`, `'forward'`).
        stream: Optional stream or device.

    Returns:
        Real output `array` of dtype `float32`.

    """
    _ = (norm, stream)
    arr = _as_array(a)
    return array(sig.irfftn(arr._tensor, s=s, axes=axes), dtype=float32)


def fftfreq(
    n: int,
    d: float = 1.0,
    *,
    stream: StreamOrDevice = None,
) -> array:
    """Return the Discrete Fourier Transform sample frequencies.

    Args:
        n: Window length.
        d: Sample spacing (inverse of the sampling rate).
        stream: Optional stream or device.

    Returns:
        Array of length `n` containing the sample frequencies.

    """
    _ = stream
    return array(sig.Fftfreq()(n, d=d), dtype=float32)


def rfftfreq(
    n: int,
    d: float = 1.0,
    *,
    stream: StreamOrDevice = None,
) -> array:
    """Return the Discrete Fourier Transform sample frequencies for `rfft`.

    Args:
        n: Window length.
        d: Sample spacing (inverse of the sampling rate).
        stream: Optional stream or device.

    Returns:
        Array of length `n // 2 + 1` containing the sample frequencies.

    """
    _ = stream
    return array(sig.rfftfreq(n, d=d), dtype=float32)


def fftshift(
    a: ArrayInput,
    axes: AxesLike = None,
    *,
    stream: StreamOrDevice = None,
) -> array:
    """Shift the zero-frequency component to the center of the spectrum.

    Args:
        a: Input array.
        axes: Axes over which to shift. Default is None, which shifts all axes.
        stream: Optional stream or device.

    Returns:
        The shifted `array`.

    """
    _ = stream
    arr = _as_array(a)
    return array(sig.fftshift(arr._tensor, axes=axes), dtype=arr.dtype)


def ifftshift(
    a: ArrayInput,
    axes: AxesLike = None,
    *,
    stream: StreamOrDevice = None,
) -> array:
    """Inverse of `fftshift`.

    Args:
        a: Input array.
        axes: Axes over which to calculate. Default is None, which shifts all axes.
        stream: Optional stream or device.

    Returns:
        The shifted `array`.

    """
    _ = stream
    arr = _as_array(a)
    return array(sig.ifftshift(arr._tensor, axes=axes), dtype=arr.dtype)


def _pad_stft_signal(sig_2d: array, pad_len: int, pad_mode: str) -> array:
    """Pad a 2D `(B, L)` signal on both ends by `pad_len` according to `pad_mode`.

    Args:
        sig_2d: Input signal of shape `(B, L)`.
        pad_len: Number of samples to pad on each side.
        pad_mode: Padding mode (`'reflect'`, `'constant'`, or `'edge'`).

    Returns:
        Padded 2D `array` of shape `(B, L + 2 * pad_len)`.

    """
    import zero_mlx.ops as _ops

    batch, sig_len = sig_2d.shape
    if pad_len == 0 or sig_len == 0:
        return sig_2d
    if pad_mode == "reflect" and sig_len > 1:
        period = 2 * (sig_len - 1)
        indices: list[int] = []
        for idx in range(-pad_len, sig_len + pad_len):
            mod_idx = idx % period
            refl_idx = period - mod_idx if mod_idx >= sig_len else mod_idx
            indices.append(refl_idx)
        return _ops.take(sig_2d, array(indices), axis=1)
    mode_lit = "edge" if (pad_mode in ("edge", "reflect") and sig_len >= 1) else "constant"
    _ = batch
    return _ops.pad(sig_2d, ((0, 0), (pad_len, pad_len)), mode=mode_lit)


def _prepare_stft_window(
    window: ArrayInput | None,
    w_len: int,
    n_fft: int,
) -> array:
    """Construct and center-pad a 1D STFT window of length `n_fft`.

    Args:
        window: Optional window array of length `w_len`.
        w_len: Active window length.
        n_fft: Total FFT frame size.

    Returns:
        1D `float32` `array` of shape `(n_fft,)`.

    """
    import zero_mlx.ops as _ops

    win_arr: array = (
        _as_array(window).astype(float32).reshape(-1)
        if window is not None
        else _ops.ones((w_len,), dtype=float32)
    )
    if win_arr.size < n_fft:
        left_pad = (n_fft - win_arr.size) // 2
        right_pad = n_fft - win_arr.size - left_pad
        return _ops.pad(win_arr, ((left_pad, right_pad),), mode="constant")
    return win_arr[:n_fft]


def stft(
    x: ArrayInput,
    n_fft: int = 2048,
    hop_length: int | None = None,
    win_length: int | None = None,
    window: ArrayInput | None = None,
    center: bool = True,
    pad_mode: str = "reflect",
    norm: str = "backward",
    onesided: bool | None = None,
    *,
    stream: StreamOrDevice = None,
) -> array:
    """Compute the Short-Time Fourier Transform (STFT) of a 1D or 2D signal `x`.

    Args:
        x: Input signal array of shape `(L,)` or `(B, L)`.
        n_fft: Size of Fourier transform.
        hop_length: Distance between neighboring sliding window frames.
        win_length: Size of window frame and STFT filter.
        window: Optional window array of length `win_length`.
        center: Whether to pad `x` on both sides so frames are centered.
        pad_mode: Padding mode when `center` is True (`'reflect'`, `'constant'`, `'edge'`).
        norm: Normalization mode (`'backward'`, `'ortho'`, `'forward'`).
        onesided: Whether to return one-sided spectrum (`n_fft // 2 + 1` bins).
        stream: Optional stream or device.

    Returns:
        Complex STFT spectrogram `array`.

    """
    import zero_mlx.ops as _ops

    arr = _as_array(x).astype(float32)
    w_len: int = win_length if win_length is not None else n_fft
    hop: int = hop_length if hop_length is not None else max(1, w_len // 4)
    is_onesided: bool = True if onesided is None else onesided

    orig_ndim = arr.ndim
    lead_shape = arr.shape[:-1] if orig_ndim > 1 else ()
    sig_len = arr.shape[-1] if orig_ndim > 0 else 1
    batch_size = int(math.prod(lead_shape)) if lead_shape else 1
    sig_2d = arr.reshape((batch_size, sig_len))

    pad_len = (n_fft // 2) if center else 0
    padded_2d = _pad_stft_signal(sig_2d, pad_len, pad_mode) if pad_len > 0 else sig_2d
    total_len = padded_2d.shape[-1]
    if total_len < n_fft:
        padded_2d = _ops.pad(padded_2d, ((0, 0), (0, n_fft - total_len)), mode="constant")
        total_len = n_fft

    n_frames: int = max(1, 1 + (total_len - n_fft) // hop)
    win_padded = _prepare_stft_window(window, w_len, n_fft)
    scale: float = (
        1.0 / math.sqrt(float(n_fft))
        if norm == "ortho"
        else (1.0 / float(n_fft) if norm == "forward" else 1.0)
    )

    frames_list: list[array] = [
        padded_2d[:, f_idx * hop : f_idx * hop + n_fft] * win_padded for f_idx in range(n_frames)
    ]
    frames_stacked = _ops.stack(frames_list, axis=2)
    spec_2d = (
        rfft(frames_stacked, n=n_fft, axis=1, stream=stream)
        if is_onesided
        else fft(frames_stacked, n=n_fft, axis=1, stream=stream)
    )
    if scale != 1.0:
        spec_2d = spec_2d * scale
    freq_bins = spec_2d.shape[1]
    out_shape: tuple[int, ...] = (
        (*lead_shape, freq_bins, n_frames) if orig_ndim > 1 else (freq_bins, n_frames)
    )
    return spec_2d.reshape(out_shape).astype(complex64)


def istft(
    stft_matrix: ArrayInput,
    n_fft: int | None = None,
    hop_length: int | None = None,
    win_length: int | None = None,
    window: ArrayInput | None = None,
    center: bool = True,
    norm: str = "backward",
    onesided: bool | None = None,
    length: int | None = None,
    *,
    stream: StreamOrDevice = None,
) -> array:
    """Compute the inverse Short-Time Fourier Transform (iSTFT).

    Args:
        stft_matrix: Complex STFT spectrogram of shape `(..., F, T)`.
        n_fft: Size of Fourier transform.
        hop_length: Distance between neighboring sliding window frames.
        win_length: Size of window frame.
        window: Optional window array.
        center: Whether the STFT was padded on both sides.
        norm: Normalization mode (`'backward'`, `'ortho'`, `'forward'`).
        onesided: Whether `stft_matrix` represents a one-sided spectrum.
        length: Optional desired output signal length.
        stream: Optional stream or device.

    Returns:
        Reconstructed time-domain signal `array` of dtype `float32`.

    """
    import zero_mlx.ops as _ops

    spec = _as_array(stft_matrix)
    is_onesided: bool = True if onesided is None else onesided
    freq_bins: int = spec.shape[-2] if spec.ndim >= 2 else 1
    n_frames: int = spec.shape[-1] if spec.ndim >= 1 else 1
    fft_size: int = (
        n_fft
        if n_fft is not None
        else ((freq_bins - 1) * 2 if is_onesided and freq_bins > 1 else freq_bins)
    )
    fft_size = max(1, fft_size)
    w_len: int = win_length if win_length is not None else fft_size
    hop: int = hop_length if hop_length is not None else max(1, w_len // 4)

    orig_ndim = spec.ndim
    lead_shape = spec.shape[:-2] if orig_ndim > 2 else ()
    batch_size = int(math.prod(lead_shape)) if lead_shape else 1
    spec_3d = spec.reshape((batch_size, freq_bins, n_frames))

    inv_scale: float = (
        math.sqrt(float(fft_size))
        if norm == "ortho"
        else (float(fft_size) if norm == "forward" else 1.0)
    )
    scaled_spec = spec_3d * inv_scale if inv_scale != 1.0 else spec_3d
    time_frames = (
        irfft(scaled_spec, n=fft_size, axis=1, stream=stream)
        if is_onesided
        else ifft(scaled_spec, n=fft_size, axis=1, stream=stream).real
    )
    win_padded = _prepare_stft_window(window, w_len, fft_size)
    win_sq = win_padded * win_padded

    raw_len = fft_size + hop * max(0, n_frames - 1)
    y_acc = _ops.zeros((batch_size, raw_len), dtype=float32)
    w_acc = _ops.zeros((raw_len,), dtype=float32)
    for f_idx in range(n_frames):
        start = f_idx * hop
        pad_right = raw_len - start - fft_size
        frame_w = time_frames[:, :, f_idx] * win_padded
        padded_frame = (
            _ops.pad(frame_w, ((0, 0), (start, pad_right)), mode="constant")
            if (start > 0 or pad_right > 0)
            else frame_w
        )
        padded_wsq = (
            _ops.pad(win_sq, ((start, pad_right),), mode="constant")
            if (start > 0 or pad_right > 0)
            else win_sq
        )
        y_acc = y_acc + padded_frame
        w_acc = w_acc + padded_wsq

    y_norm = y_acc / _ops.maximum(w_acc, 1e-8)
    pad_trim = (fft_size // 2) if (center and raw_len > fft_size) else 0
    if pad_trim > 0:
        y_norm = y_norm[:, pad_trim : raw_len - pad_trim]

    curr_len = y_norm.shape[-1]
    if length is not None:
        if length <= curr_len:
            y_norm = y_norm[:, :length]
        else:
            y_norm = _ops.pad(y_norm, ((0, 0), (0, length - curr_len)), mode="constant")
        out_len = length
    else:
        out_len = max(1, curr_len)

    out_shape: tuple[int, ...] = (*lead_shape, out_len) if orig_ndim > 2 else (out_len,)
    return y_norm.reshape(out_shape).astype(float32)


__all__ = [
    "fft",
    "fft2",
    "fftfreq",
    "fftn",
    "fftshift",
    "ifft",
    "ifft2",
    "ifftn",
    "ifftshift",
    "irfft",
    "irfft2",
    "irfftn",
    "istft",
    "rfft",
    "rfft2",
    "rfftfreq",
    "rfftn",
    "stft",
]
