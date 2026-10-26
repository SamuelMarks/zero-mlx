"""Compiler operation parity tests for `creation` operations."""

from __future__ import annotations

from tests.conftest import check_parity, np


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


def test_BetaCdf_parity() -> None:
    """Test parity for BetaCdf."""
    check_parity(
        "BetaCdf",
        lambda: [np.array([0.2, 0.5], dtype=np.float32)],
    )


def test_BetaPdf_parity() -> None:
    """Test parity for BetaPdf."""
    check_parity(
        "BetaPdf",
        lambda: [np.array([0.2, 0.5], dtype=np.float32)],
    )


def test_BinomCdf_parity() -> None:
    """Test parity for BinomCdf."""
    check_parity(
        "BinomCdf",
        lambda: [np.array([1.0, 2.0], dtype=np.float32)],
    )


def test_BinomPmf_parity() -> None:
    """Test parity for BinomPmf."""
    check_parity(
        "BinomPmf",
        lambda: [np.array([1.0, 2.0], dtype=np.float32)],
    )


def test_C_parity() -> None:
    """Test parity for C."""
    check_parity(
        "C",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_CreateToken_parity() -> None:
    """Test parity for CreateToken."""
    check_parity(
        "CreateToken",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_Descriptive_parity() -> None:
    """Test parity for Descriptive."""
    check_parity(
        "Descriptive",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_DeviceContext_parity() -> None:
    """Test parity for DeviceContext."""
    check_parity(
        "DeviceContext",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_DeviceTransfer_parity() -> None:
    """Test parity for DeviceTransfer."""
    check_parity(
        "DeviceTransfer",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_Distributions_parity() -> None:
    """Test parity for Distributions."""
    check_parity(
        "Distributions",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_FromDlpack_parity() -> None:
    """Test parity for FromDlpack."""
    check_parity(
        "FromDlpack",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_Fromfile_parity() -> None:
    """Test parity for Fromfile."""
    check_parity(
        "Fromfile",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_Fromfunction_parity() -> None:
    """Test parity for Fromfunction."""
    check_parity(
        "Fromfunction",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_Fromiter_parity() -> None:
    """Test parity for Fromiter."""
    check_parity(
        "Fromiter",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_Frompyfunc_parity() -> None:
    """Test parity for Frompyfunc."""
    check_parity(
        "Frompyfunc",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_Fromstring_parity() -> None:
    """Test parity for Fromstring."""
    check_parity(
        "Fromstring",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_GammaCdf_parity() -> None:
    """Test parity for GammaCdf."""
    check_parity(
        "GammaCdf",
        lambda: [np.array([1.0, 2.0], dtype=np.float32)],
    )


def test_GammaPdf_parity() -> None:
    """Test parity for GammaPdf."""
    check_parity(
        "GammaPdf",
        lambda: [np.array([1.0, 2.0], dtype=np.float32)],
    )


def test_Geometric_parity() -> None:
    """Test parity for Geometric."""
    check_parity(
        "Geometric",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_Geomspace_parity() -> None:
    """Test parity for Geomspace."""
    check_parity(
        "Geomspace",
        lambda: [np.array([1.0, 10.0], dtype=np.float32)],
    )


def test_Load_parity() -> None:
    """Test parity for Load."""
    check_parity(
        "Load",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_NonblockingLoad_parity() -> None:
    """Test parity for NonblockingLoad."""
    check_parity(
        "NonblockingLoad",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_NonblockingSave_parity() -> None:
    """Test parity for NonblockingSave."""
    check_parity(
        "NonblockingSave",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_NormCdf_parity() -> None:
    """Test parity for NormCdf."""
    check_parity(
        "NormCdf",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_NormPdf_parity() -> None:
    """Test parity for NormPdf."""
    check_parity(
        "NormPdf",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_PoissonCdf_parity() -> None:
    """Test parity for PoissonCdf."""
    check_parity(
        "PoissonCdf",
        lambda: [np.array([1.0, 2.0], dtype=np.float32)],
    )


def test_PoissonPmf_parity() -> None:
    """Test parity for PoissonPmf."""
    check_parity(
        "PoissonPmf",
        lambda: [np.array([1.0, 2.0], dtype=np.float32)],
    )


def test_Save_parity() -> None:
    """Test parity for Save."""
    check_parity(
        "Save",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_SaveGguf_parity() -> None:
    """Test parity for SaveGguf."""
    check_parity(
        "SaveGguf",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_Savez_parity() -> None:
    """Test parity for Savez."""
    check_parity(
        "Savez",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_SavezCompressed_parity() -> None:
    """Test parity for SavezCompressed."""
    check_parity(
        "SavezCompressed",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_binomial_parity() -> None:
    """Test parity for binomial."""
    check_parity(
        "binomial",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_categorical_parity() -> None:
    """Test parity for categorical."""
    check_parity(
        "categorical",
        lambda: [np.array([[1.0, 2.0, 3.0]], dtype=np.float32)],
    )


def test_choice_parity() -> None:
    """Test parity for choice."""
    check_parity(
        "choice",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_dirichlet_parity() -> None:
    """Test parity for dirichlet."""
    check_parity(
        "dirichlet",
        lambda: [np.array([1.0, 2.0], dtype=np.float32)],
    )


def test_permutation_parity() -> None:
    """Test parity for permutation."""
    check_parity(
        "permutation",
        lambda: [np.array([1, 2, 3, 4], dtype=np.int32)],
    )


def test_truncated_normal_parity() -> None:
    """Test parity for truncated_normal."""
    check_parity(
        "truncated_normal",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )
