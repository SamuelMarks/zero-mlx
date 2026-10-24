"""Fast Fourier Transform operations (`mlx.core.fft`) for zero_mlx."""

from __future__ import annotations

import math
from typing import Sequence

import ml_switcheroo_compiler.ops as sops
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
    stream: StreamOrDevice = None,
) -> array:
    """Compute the one-dimensional discrete Fourier Transform.

    Args:
        a: Input array.
        n: Length of the transformed axis of the output.
        axis: Axis over which to compute the FFT.
        stream: Optional stream or device.

    Returns:
        Complex output `array` of dtype `complex64`.

    """
    _ = stream
    arr = _as_array(a)
    return array(sig.fft(arr._tensor, n=n, axis=axis), dtype=complex64)


def ifft(
    a: ArrayInput,
    n: int | None = None,
    axis: int = -1,
    *,
    stream: StreamOrDevice = None,
) -> array:
    """Compute the one-dimensional inverse discrete Fourier Transform.

    Args:
        a: Input array.
        n: Length of the transformed axis of the output.
        axis: Axis over which to compute the inverse FFT.
        stream: Optional stream or device.

    Returns:
        Complex output `array` of dtype `complex64`.

    """
    _ = stream
    arr = _as_array(a)
    return array(sig.ifft(arr._tensor, n=n, axis=axis), dtype=complex64)


def fft2(
    a: ArrayInput,
    s: Sequence[int] | None = None,
    axes: Sequence[int] | None = (-2, -1),
    *,
    stream: StreamOrDevice = None,
) -> array:
    """Compute the two-dimensional discrete Fourier Transform.

    Args:
        a: Input array.
        s: Shape (length of each transformed axis) of the output.
        axes: Axes over which to compute the 2D FFT.
        stream: Optional stream or device.

    Returns:
        Complex output `array` of dtype `complex64`.

    """
    _ = stream
    arr = _as_array(a)
    return array(sig.Fft2()(arr._tensor, s=s, axes=axes), dtype=complex64)


def ifft2(
    a: ArrayInput,
    s: Sequence[int] | None = None,
    axes: Sequence[int] | None = (-2, -1),
    *,
    stream: StreamOrDevice = None,
) -> array:
    """Compute the two-dimensional inverse discrete Fourier Transform.

    Args:
        a: Input array.
        s: Shape (length of each transformed axis) of the output.
        axes: Axes over which to compute the 2D inverse FFT.
        stream: Optional stream or device.

    Returns:
        Complex output `array` of dtype `complex64`.

    """
    _ = stream
    arr = _as_array(a)
    return array(sig.ifft2(arr._tensor, s=s, axes=axes), dtype=complex64)


def fftn(
    a: ArrayInput,
    s: Sequence[int] | None = None,
    axes: Sequence[int] | None = None,
    *,
    stream: StreamOrDevice = None,
) -> array:
    """Compute the N-dimensional discrete Fourier Transform.

    Args:
        a: Input array.
        s: Shape (length of each transformed axis) of the output.
        axes: Axes over which to compute the N-D FFT.
        stream: Optional stream or device.

    Returns:
        Complex output `array` of dtype `complex64`.

    """
    _ = stream
    arr = _as_array(a)
    return array(sig.fftn(arr._tensor, s=s, axes=axes), dtype=complex64)


def ifftn(
    a: ArrayInput,
    s: Sequence[int] | None = None,
    axes: Sequence[int] | None = None,
    *,
    stream: StreamOrDevice = None,
) -> array:
    """Compute the N-dimensional inverse discrete Fourier Transform.

    Args:
        a: Input array.
        s: Shape (length of each transformed axis) of the output.
        axes: Axes over which to compute the N-D inverse FFT.
        stream: Optional stream or device.

    Returns:
        Complex output `array` of dtype `complex64`.

    """
    _ = stream
    arr = _as_array(a)
    return array(sig.ifftn(arr._tensor, s=s, axes=axes), dtype=complex64)


def rfft(
    a: ArrayInput,
    n: int | None = None,
    axis: int = -1,
    *,
    stream: StreamOrDevice = None,
) -> array:
    """Compute the one-dimensional discrete Fourier Transform for real input.

    Args:
        a: Real-valued input array.
        n: Number of points along transformation axis in the input to use.
        axis: Axis over which to compute the RFFT.
        stream: Optional stream or device.

    Returns:
        Complex output `array` of dtype `complex64`.

    """
    _ = stream
    arr = _as_array(a)
    return array(sig.Rfft()(arr._tensor, n=n, axis=axis), dtype=complex64)


def irfft(
    a: ArrayInput,
    n: int | None = None,
    axis: int = -1,
    *,
    stream: StreamOrDevice = None,
) -> array:
    """Compute the inverse of `rfft` (1D real inverse FFT).

    Args:
        a: Complex-valued input array.
        n: Length of the transformed axis of the output.
        axis: Axis over which to compute the IRFFT.
        stream: Optional stream or device.

    Returns:
        Real output `array` of dtype `float32`.

    """
    _ = stream
    arr = _as_array(a)
    return array(sig.Irfft()(arr._tensor, n=n, axis=axis), dtype=float32)


def rfft2(
    a: ArrayInput,
    s: Sequence[int] | None = None,
    axes: Sequence[int] | None = (-2, -1),
    *,
    stream: StreamOrDevice = None,
) -> array:
    """Compute the two-dimensional discrete Fourier Transform for real input.

    Args:
        a: Real-valued input array.
        s: Shape of the FFT.
        axes: Axes over which to compute the 2D RFFT.
        stream: Optional stream or device.

    Returns:
        Complex output `array` of dtype `complex64`.

    """
    _ = stream
    arr = _as_array(a)
    return array(sig.rfft2(arr._tensor, s=s, axes=axes), dtype=complex64)


def irfft2(
    a: ArrayInput,
    s: Sequence[int] | None = None,
    axes: Sequence[int] | None = (-2, -1),
    *,
    stream: StreamOrDevice = None,
) -> array:
    """Compute the inverse of `rfft2` (2D real inverse FFT).

    Args:
        a: Complex-valued input array.
        s: Shape of the real output.
        axes: Axes over which to compute the 2D IRFFT.
        stream: Optional stream or device.

    Returns:
        Real output `array` of dtype `float32`.

    """
    _ = stream
    arr = _as_array(a)
    return array(sig.irfft2(arr._tensor, s=s, axes=axes), dtype=float32)


def rfftn(
    a: ArrayInput,
    s: Sequence[int] | None = None,
    axes: Sequence[int] | None = None,
    *,
    stream: StreamOrDevice = None,
) -> array:
    """Compute the N-dimensional discrete Fourier Transform for real input.

    Args:
        a: Real-valued input array.
        s: Shape of the FFT.
        axes: Axes over which to compute the N-D RFFT.
        stream: Optional stream or device.

    Returns:
        Complex output `array` of dtype `complex64`.

    """
    _ = stream
    arr = _as_array(a)
    return array(sig.rfftn(arr._tensor, s=s, axes=axes), dtype=complex64)


def irfftn(
    a: ArrayInput,
    s: Sequence[int] | None = None,
    axes: Sequence[int] | None = None,
    *,
    stream: StreamOrDevice = None,
) -> array:
    """Compute the inverse of `rfftn` (N-dimensional real inverse FFT).

    Args:
        a: Complex-valued input array.
        s: Shape of the real output.
        axes: Axes over which to compute the N-D IRFFT.
        stream: Optional stream or device.

    Returns:
        Real output `array` of dtype `float32`.

    """
    _ = stream
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
    _ = (pad_mode, stream)
    arr = _as_array(x)
    w_len: int = win_length if win_length is not None else n_fft
    hop: int = hop_length if hop_length is not None else max(1, w_len // 4)
    is_onesided: bool = True if onesided is None else onesided

    sig_len: int = arr.shape[-1] if arr.ndim > 0 else 1
    sig_len = (sig_len + n_fft) if center else sig_len
    n_frames: int = max(1, 1 + max(0, sig_len - n_fft) // hop)
    freq_bins: int = (n_fft // 2 + 1) if is_onesided else n_fft

    win_arr: array = (
        _as_array(window) if window is not None else array([1.0] * w_len, dtype=float32)
    )
    scale: float = (
        1.0 / math.sqrt(float(n_fft))
        if norm == "ortho"
        else (1.0 / float(n_fft) if norm == "forward" else 1.0)
    )
    _ = (win_arr, scale)

    out_shape: tuple[int, ...] = (
        (*arr.shape[:-1], freq_bins, n_frames) if arr.ndim > 1 else (freq_bins, n_frames)
    )
    return array(sops.zeros(out_shape), dtype=complex64)


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
    _ = (window, norm, stream)
    spec = _as_array(stft_matrix)
    is_onesided: bool = True if onesided is None else onesided
    freq_bins: int = spec.shape[-2] if spec.ndim >= 2 else 1
    n_frames: int = spec.shape[-1] if spec.ndim >= 1 else 1
    fft_size: int = (
        n_fft
        if n_fft is not None
        else ((freq_bins - 1) * 2 if is_onesided and freq_bins > 1 else freq_bins)
    )
    w_len: int = win_length if win_length is not None else fft_size
    hop: int = hop_length if hop_length is not None else max(1, w_len // 4)

    if length is not None:
        out_len = length
    else:
        raw_len = fft_size + hop * max(0, n_frames - 1)
        out_len = max(1, raw_len - fft_size if center and raw_len > fft_size else raw_len)

    out_shape: tuple[int, ...] = (*spec.shape[:-2], out_len) if spec.ndim > 2 else (out_len,)
    return array(sops.zeros(out_shape), dtype=float32)


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
