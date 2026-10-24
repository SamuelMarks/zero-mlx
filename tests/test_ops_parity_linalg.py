"""Compiler operation parity tests for `linalg` operations."""

from __future__ import annotations

import numpy as np
import pytest

from tests.conftest import check_parity


def test_Dot_parity() -> None:
    """Test parity for Dot."""
    check_parity(
        "Dot",
        lambda: [
            np.random.randn(3).astype(np.float32),
            np.random.randn(3).astype(np.float32),
        ],
    )


def test_DotGeneral_parity() -> None:
    """Test parity for DotGeneral."""
    check_parity("Dot_General", lambda: [])  # Skip complex for now


def test_Erfinv_parity() -> None:
    """Test parity for Erfinv."""
    check_parity(
        "Erfinv",
        lambda: [np.random.uniform(-0.9, 0.9, size=(2, 3)).astype(np.float32)],
        rtol=1e-4,
        atol=1e-4,
    )


def test_Matmul_parity() -> None:
    """Test parity for Matmul."""
    check_parity(
        "Matmul",
        lambda: [
            np.random.randn(2, 3).astype(np.float32),
            np.random.randn(3, 4).astype(np.float32),
        ],
    )


def test_Rsqrt_parity() -> None:
    """Test parity for Rsqrt."""
    check_parity("Rsqrt", lambda: [np.random.uniform(0.1, 5, size=(2, 3)).astype(np.float32)])


def test_Sqrt_parity() -> None:
    """Test parity for Sqrt."""
    check_parity("Sqrt", lambda: [np.random.uniform(0.1, 5, size=(2, 3)).astype(np.float32)])


def test_AsString_parity() -> None:
    """Test parity for AsString."""
    check_parity(
        "AsString",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_BandedTriangularSolve_parity() -> None:
    """Test parity for BandedTriangularSolve."""
    check_parity(
        "BandedTriangularSolve",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_Cholesky_parity() -> None:
    """Test parity for Cholesky."""
    check_parity(
        "Cholesky",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_CholeskySolve_parity() -> None:
    """Test parity for CholeskySolve."""
    check_parity(
        "CholeskySolve",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_ConfusionMatrix_parity() -> None:
    """Test parity for ConfusionMatrix."""
    check_parity(
        "ConfusionMatrix",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_Cross_parity() -> None:
    """Test parity for Cross."""
    check_parity(
        "Cross",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_CustomLinearSolve_parity() -> None:
    """Test parity for CustomLinearSolve."""
    check_parity(
        "CustomLinearSolve",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_Det_parity() -> None:
    """Test parity for Det."""
    check_parity(
        "Det",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_Diag_parity() -> None:
    """Test parity for Diag."""
    check_parity(
        "Diag",
        lambda: [np.random.randn(3).astype(np.float32)],
    )


def test_DiagIndices_parity() -> None:
    """Test parity for DiagIndices."""
    check_parity(
        "DiagIndices",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_DiagIndicesFrom_parity() -> None:
    """Test parity for DiagIndicesFrom."""
    check_parity(
        "DiagIndicesFrom",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_Diagflat_parity() -> None:
    """Test parity for Diagflat."""
    check_parity(
        "Diagflat",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_Diagonal_parity() -> None:
    """Test parity for Diagonal."""
    check_parity(
        "Diagonal",
        lambda: [np.random.randn(3, 3).astype(np.float32)],
    )


def test_Eigh_parity() -> None:
    """Test parity for Eigh."""
    check_parity(
        "Eigh",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_EighTridiagonal_parity() -> None:
    """Test parity for EighTridiagonal."""
    check_parity(
        "EighTridiagonal",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_Eigvalsh_parity() -> None:
    """Test parity for Eigvalsh."""
    check_parity(
        "Eigvalsh",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_ErfInv_parity() -> None:
    """Test parity for ErfInv."""
    check_parity(
        "ErfInv",
        lambda: [np.random.uniform(-0.9, 0.9, (2, 2)).astype(np.float32)],
        rtol=1e-4,
        atol=1e-4,
    )


def test_Erfcinv_parity() -> None:
    """Test parity for Erfcinv."""
    check_parity(
        "Erfcinv",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_FlipLeftRight_parity() -> None:
    """Test parity for FlipLeftRight."""
    check_parity(
        "FlipLeftRight",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_Inv_parity() -> None:
    """Test parity for Inv."""
    check_parity(
        "Inv",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_Invert_parity() -> None:
    """Test parity for Invert."""
    check_parity(
        "Invert",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_InvertPermutation_parity() -> None:
    """Test parity for InvertPermutation."""
    check_parity(
        "InvertPermutation",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_IsStrictlyIncreasing_parity() -> None:
    """Test parity for IsStrictlyIncreasing."""
    check_parity(
        "IsStrictlyIncreasing",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_LuSolve_parity() -> None:
    """Test parity for LuSolve."""
    check_parity(
        "LuSolve",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_MatrixExponential_parity() -> None:
    """Test parity for MatrixExponential."""
    check_parity(
        "MatrixExponential",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_MatrixNorm_parity() -> None:
    """Test parity for MatrixNorm."""
    check_parity(
        "MatrixNorm",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_MatrixPower_parity() -> None:
    """Test parity for MatrixPower."""
    check_parity(
        "MatrixPower",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_MatrixRank_parity() -> None:
    """Test parity for MatrixRank."""
    check_parity(
        "MatrixRank",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_MatrixTranspose_parity() -> None:
    """Test parity for MatrixTranspose."""
    check_parity(
        "MatrixTranspose",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_MultiDot_parity() -> None:
    """Test parity for MultiDot."""
    check_parity(
        "MultiDot",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_Ndtri_parity() -> None:
    """Test parity for Ndtri."""
    check_parity(
        "Ndtri",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_Pdot_parity() -> None:
    """Test parity for Pdot."""
    check_parity(
        "Pdot",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_Pinv_parity() -> None:
    """Test parity for Pinv."""
    check_parity(
        "Pinv",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_Qr_parity() -> None:
    """Test parity for Qr."""
    check_parity(
        "Qr",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_RaggedCrossHashed_parity() -> None:
    """Test parity for RaggedCrossHashed."""
    check_parity(
        "RaggedCrossHashed",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_RaggedDot_parity() -> None:
    """Test parity for RaggedDot."""
    check_parity(
        "RaggedDot",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_RaggedMatMul_parity() -> None:
    """Test parity for RaggedMatMul."""
    check_parity(
        "RaggedMatMul",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_RawMatMul_parity() -> None:
    """Test parity for RawMatMul."""
    check_parity(
        "RawMatMul",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_Rayleigh_parity() -> None:
    """Test parity for Rayleigh."""
    check_parity(
        "Rayleigh",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_Slogdet_parity() -> None:
    """Test parity for Slogdet."""
    check_parity(
        "Slogdet",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_Solve_parity() -> None:
    """Test parity for Solve."""
    check_parity(
        "Solve",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_SparseCrossHashed_parity() -> None:
    """Test parity for SparseCrossHashed."""
    check_parity(
        "SparseCrossHashed",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_SparseDenseMatMul_parity() -> None:
    """Test parity for SparseDenseMatMul."""
    check_parity(
        "SparseDenseMatMul",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_SparseSegmentSqrtN_parity() -> None:
    """Test parity for SparseSegmentSqrtN."""
    check_parity(
        "SparseSegmentSqrtN",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_Sqrtm_parity() -> None:
    """Test parity for Sqrtm."""
    check_parity(
        "Sqrtm",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_StridedSlice_parity() -> None:
    """Test parity for StridedSlice."""
    check_parity(
        "StridedSlice",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_StringJoin_parity() -> None:
    """Test parity for StringJoin."""
    check_parity(
        "StringJoin",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_StringLength_parity() -> None:
    """Test parity for StringLength."""
    check_parity(
        "StringLength",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_StringLookup_parity() -> None:
    """Test parity for StringLookup."""
    check_parity(
        "StringLookup",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_StringLower_parity() -> None:
    """Test parity for StringLower."""
    check_parity(
        "StringLower",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_StringSplit_parity() -> None:
    """Test parity for StringSplit."""
    check_parity(
        "StringSplit",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_StringSubstr_parity() -> None:
    """Test parity for StringSubstr."""
    check_parity(
        "StringSubstr",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_StringToHash_parity() -> None:
    """Test parity for StringToHash."""
    check_parity(
        "StringToHash",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_StringToNumber_parity() -> None:
    """Test parity for StringToNumber."""
    check_parity(
        "StringToNumber",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_StringUpper_parity() -> None:
    """Test parity for StringUpper."""
    check_parity(
        "StringUpper",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_Svd_parity() -> None:
    """Test parity for Svd."""
    check_parity(
        "Svd",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_Svdvals_parity() -> None:
    """Test parity for Svdvals."""
    check_parity(
        "Svdvals",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_Tensordot_parity() -> None:
    """Test parity for Tensordot."""
    check_parity(
        "Tensordot",
        lambda: [
            np.random.randn(2, 3, 4).astype(np.float32),
            np.random.randn(3, 4, 2).astype(np.float32),
        ],
    )


def test_Tensorinv_parity() -> None:
    """Test parity for Tensorinv."""
    check_parity(
        "Tensorinv",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_Tensorsolve_parity() -> None:
    """Test parity for Tensorsolve."""
    check_parity(
        "Tensorsolve",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_TimeDistributed_parity() -> None:
    """Test parity for TimeDistributed."""
    check_parity(
        "TimeDistributed",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_Trace_parity() -> None:
    """Test parity for Trace."""
    check_parity(
        "Trace",
        lambda: [np.random.randn(3, 3).astype(np.float32)],
    )


def test_TriInv_parity() -> None:
    """Test parity for TriInv."""
    check_parity(
        "TriInv",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_Triangular_parity() -> None:
    """Test parity for Triangular."""
    check_parity(
        "Triangular",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_TriangularSolve_parity() -> None:
    """Test parity for TriangularSolve."""
    check_parity(
        "TriangularSolve",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_Tridiagonal_parity() -> None:
    """Test parity for Tridiagonal."""
    check_parity(
        "Tridiagonal",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_TridiagonalSolve_parity() -> None:
    """Test parity for TridiagonalSolve."""
    check_parity(
        "TridiagonalSolve",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_Tril_parity() -> None:
    """Test parity for Tril."""
    check_parity(
        "Tril",
        lambda: [np.random.randn(3, 3).astype(np.float32)],
    )


def test_TrilIndices_parity() -> None:
    """Test parity for TrilIndices."""
    check_parity(
        "TrilIndices",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_TrilIndicesFrom_parity() -> None:
    """Test parity for TrilIndicesFrom."""
    check_parity(
        "TrilIndicesFrom",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_Triu_parity() -> None:
    """Test parity for Triu."""
    check_parity(
        "Triu",
        lambda: [np.random.randn(3, 3).astype(np.float32)],
    )


def test_TriuIndices_parity() -> None:
    """Test parity for TriuIndices."""
    check_parity(
        "TriuIndices",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_TriuIndicesFrom_parity() -> None:
    """Test parity for TriuIndicesFrom."""
    check_parity(
        "TriuIndicesFrom",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_UniqueInverse_parity() -> None:
    """Test parity for UniqueInverse."""
    check_parity(
        "UniqueInverse",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_UnsortedSegmentSqrtN_parity() -> None:
    """Test parity for UnsortedSegmentSqrtN."""
    check_parity(
        "UnsortedSegmentSqrtN",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_Vdot_parity() -> None:
    """Test parity for Vdot."""
    check_parity(
        "Vdot",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_Vecdot_parity() -> None:
    """Test parity for Vecdot."""
    check_parity(
        "Vecdot",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )
