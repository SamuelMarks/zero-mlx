"""Compiler operation parity tests for `fft` operations."""

from __future__ import annotations

import numpy as np

from tests.conftest import check_parity


def test_Fft_parity() -> None:
    """Test parity for Fft."""
    check_parity(
        "Fft",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
        kwargs_generator=lambda: {"n": 3},
    )


def test_Rfft_parity() -> None:
    """Test parity for Rfft."""
    check_parity("Rfft", lambda: [np.random.randn(2, 3).astype(np.float32)])


def test_Dct_parity() -> None:
    """Test parity for Dct."""
    check_parity(
        "Dct",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_Fft2d_parity() -> None:
    """Test parity for Fft2d."""
    check_parity(
        "Fft2d",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_Fft3d_parity() -> None:
    """Test parity for Fft3d."""
    check_parity(
        "Fft3d",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_Fftfreq_parity() -> None:
    """Test parity for Fftfreq."""
    check_parity(
        "Fftfreq",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_Fftnd_parity() -> None:
    """Test parity for Fftnd."""
    check_parity(
        "Fftnd",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_Fftshift_parity() -> None:
    """Test parity for Fftshift."""
    check_parity(
        "Fftshift",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_Hfft_parity() -> None:
    """Test parity for Hfft."""
    check_parity(
        "Hfft",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_Idct_parity() -> None:
    """Test parity for Idct."""
    check_parity(
        "Idct",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_Ifft_parity() -> None:
    """Test parity for Ifft."""
    check_parity(
        "Ifft",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_Ifft2d_parity() -> None:
    """Test parity for Ifft2d."""
    check_parity(
        "Ifft2d",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_Ifft3d_parity() -> None:
    """Test parity for Ifft3d."""
    check_parity(
        "Ifft3d",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_Ifftnd_parity() -> None:
    """Test parity for Ifftnd."""
    check_parity(
        "Ifftnd",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_Ifftshift_parity() -> None:
    """Test parity for Ifftshift."""
    check_parity(
        "Ifftshift",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_Ihfft_parity() -> None:
    """Test parity for Ihfft."""
    check_parity(
        "Ihfft",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_InverseMdct_parity() -> None:
    """Test parity for InverseMdct."""
    check_parity(
        "InverseMdct",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_Irfft_parity() -> None:
    """Test parity for Irfft."""
    check_parity(
        "Irfft",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_Irfft2d_parity() -> None:
    """Test parity for Irfft2d."""
    check_parity(
        "Irfft2d",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_Irfft3d_parity() -> None:
    """Test parity for Irfft3d."""
    check_parity(
        "Irfft3d",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_Irfftnd_parity() -> None:
    """Test parity for Irfftnd."""
    check_parity(
        "Irfftnd",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_Istft_parity() -> None:
    """Test parity for Istft."""
    check_parity(
        "Istft",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_Mdct_parity() -> None:
    """Test parity for Mdct."""
    check_parity(
        "Mdct",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_Rfft2d_parity() -> None:
    """Test parity for Rfft2d."""
    check_parity(
        "Rfft2d",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_Rfft3d_parity() -> None:
    """Test parity for Rfft3d."""
    check_parity(
        "Rfft3d",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_Rfftfreq_parity() -> None:
    """Test parity for Rfftfreq."""
    check_parity(
        "Rfftfreq",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_Rfftnd_parity() -> None:
    """Test parity for Rfftnd."""
    check_parity(
        "Rfftnd",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_Stft_parity() -> None:
    """Test parity for Stft."""
    check_parity(
        "Stft",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )
