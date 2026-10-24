"""Compiler operation parity tests for `creation` operations."""

from __future__ import annotations

import numpy as np

from tests.conftest import check_parity


def test_Arange_parity() -> None:
    """Test parity for Arange."""
    check_parity("Arange", lambda: [10])


def test_Full_parity() -> None:
    """Test parity for Full."""
    check_parity("Full", lambda: [np.array([2, 3], dtype=np.int32), 3.14])


def test_Ones_parity() -> None:
    """Test parity for Ones."""
    check_parity("Ones", lambda: [(2, 3)])


def test_Zeros_parity() -> None:
    """Test parity for Zeros."""
    check_parity("Zeros", lambda: [(2, 3)])


def test_AffineGrid_parity() -> None:
    """Test parity for AffineGrid."""
    check_parity(
        "AffineGrid",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_Clone_parity() -> None:
    """Test parity for Clone."""
    check_parity(
        "Clone",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_Exponential_parity() -> None:
    """Test parity for Exponential."""
    check_parity(
        "Exponential",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_GridSample_parity() -> None:
    """Test parity for GridSample."""
    check_parity(
        "GridSample",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_Meshgrid_parity() -> None:
    """Test parity for Meshgrid."""
    check_parity(
        "Meshgrid",
        lambda: [
            np.array([1, 2, 3], dtype=np.float32),
            np.array([4, 5, 6], dtype=np.float32),
        ],
    )


def test_RegexFullMatch_parity() -> None:
    """Test parity for RegexFullMatch."""
    check_parity(
        "RegexFullMatch",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_SparseEye_parity() -> None:
    """Test parity for SparseEye."""
    check_parity(
        "SparseEye",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_SparseFillEmptyRows_parity() -> None:
    """Test parity for SparseFillEmptyRows."""
    check_parity(
        "SparseFillEmptyRows",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_ZeroFraction_parity() -> None:
    """Test parity for ZeroFraction."""
    check_parity(
        "ZeroFraction",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )
