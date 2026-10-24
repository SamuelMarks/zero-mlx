"""Compiler operation parity tests for `nn` operations."""

from __future__ import annotations

import numpy as np

from tests.conftest import check_parity


def test_ConvGeneralDilated_parity() -> None:
    """Test parity for ConvGeneralDilated."""
    check_parity(
        "Conv_General_Dilated", lambda: []
    )  # Skip complex tensor generation for now, just stub


def test_Norm_parity() -> None:
    """Test parity for Norm."""
    check_parity("Norm", lambda: [np.random.randn(2, 3).astype(np.float32)])


def test_AdaptiveAvgPool2D_parity() -> None:
    """Test parity for AdaptiveAvgPool2D."""
    check_parity(
        "AdaptiveAvgPool2D",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_AdaptiveMaxPool2D_parity() -> None:
    """Test parity for AdaptiveMaxPool2D."""
    check_parity(
        "AdaptiveMaxPool2D",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_AlphaDropout_parity() -> None:
    """Test parity for AlphaDropout."""
    check_parity(
        "AlphaDropout",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_BitcastConvertType_parity() -> None:
    """Test parity for BitcastConvertType."""
    check_parity(
        "BitcastConvertType",
        lambda: [np.array([1.5], dtype=np.float32)],
        lambda: {"new_dtype": np.int32},
    )


def test_CTCLoss_parity() -> None:
    """Test parity for CTCLoss."""
    check_parity(
        "CTCLoss",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_Conv_parity() -> None:
    """Test parity for Conv."""
    check_parity(
        "Conv",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_ConvGeneralDilatedLocal_parity() -> None:
    """Test parity for ConvGeneralDilatedLocal."""
    check_parity(
        "ConvGeneralDilatedLocal",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_ConvGeneralDilatedPatches_parity() -> None:
    """Test parity for ConvGeneralDilatedPatches."""
    check_parity(
        "ConvGeneralDilatedPatches",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_ConvTranspose_parity() -> None:
    """Test parity for ConvTranspose."""
    check_parity(
        "ConvTranspose",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_ConvWithGeneralPadding_parity() -> None:
    """Test parity for ConvWithGeneralPadding."""
    check_parity(
        "ConvWithGeneralPadding",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_Convolve_parity() -> None:
    """Test parity for Convolve."""
    check_parity(
        "Convolve",
        lambda: [
            np.random.randn(5).astype(np.float32),
            np.random.randn(3).astype(np.float32),
        ],
    )


def test_Dropout_parity() -> None:
    """Test parity for Dropout."""
    check_parity(
        "Dropout",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_FractionalMaxPool2D_parity() -> None:
    """Test parity for FractionalMaxPool2D."""
    check_parity(
        "FractionalMaxPool2D",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_GeneralizedNormal_parity() -> None:
    """Test parity for GeneralizedNormal."""
    check_parity(
        "GeneralizedNormal",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_GroupNorm_parity() -> None:
    """Test parity for GroupNorm."""
    check_parity(
        "GroupNorm",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_L2Normalize_parity() -> None:
    """Test parity for L2Normalize."""
    check_parity(
        "L2Normalize",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_Lognormal_parity() -> None:
    """Test parity for Lognormal."""
    check_parity(
        "Lognormal",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_MultivariateNormal_parity() -> None:
    """Test parity for MultivariateNormal."""
    check_parity(
        "MultivariateNormal",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_RawConv2D_parity() -> None:
    """Test parity for RawConv2D."""
    check_parity(
        "RawConv2D",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_ReduceEuclideanNorm_parity() -> None:
    """Test parity for ReduceEuclideanNorm."""
    check_parity(
        "ReduceEuclideanNorm",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_SparseSoftmax_parity() -> None:
    """Test parity for SparseSoftmax."""
    check_parity(
        "SparseSoftmax",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_VectorNorm_parity() -> None:
    """Test parity for VectorNorm."""
    check_parity(
        "VectorNorm",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_Dropout2d_parity() -> None:
    """Test parity for Dropout2d."""
    check_parity("Dropout2d", lambda: [np.random.randn(2, 3).astype(np.float32)])


def test_Dropout3d_parity() -> None:
    """Test parity for Dropout3d."""
    check_parity("Dropout3d", lambda: [np.random.randn(2, 3).astype(np.float32)])


def test_Gru_parity() -> None:
    """Test parity for Gru."""
    check_parity("Gru", lambda: [np.random.randn(2, 3).astype(np.float32)])


def test_Rope_parity() -> None:
    """Test parity for Rope."""
    check_parity("Rope", lambda: [np.random.randn(2, 3).astype(np.float32)])


def test_CudaKernel_parity() -> None:
    """Test parity for CudaKernel."""
    check_parity("CudaKernel", lambda: [np.random.randn(2, 3).astype(np.float32)])


def test_MetalKernel_parity() -> None:
    """Test parity for MetalKernel."""
    check_parity("MetalKernel", lambda: [np.random.randn(2, 3).astype(np.float32)])


def test_PrecompiledCudaKernel_parity() -> None:
    """Test parity for PrecompiledCudaKernel."""
    check_parity("PrecompiledCudaKernel", lambda: [np.random.randn(2, 3).astype(np.float32)])
