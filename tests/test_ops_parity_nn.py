"""Compiler operation parity tests for `nn` operations."""

from __future__ import annotations

from tests.conftest import check_parity, np


def test_ConvGeneralDilated_parity() -> None:
    """Test parity for ConvGeneralDilated."""
    check_parity(
        "Conv_General_Dilated",
        lambda: [
            np.ones((1, 4, 4, 2), dtype=np.float32),
            np.ones((3, 2, 2, 2), dtype=np.float32),
        ],
    )


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
    check_parity(
        "Rope",
        lambda: [np.ones((1, 2, 4, 8), dtype=np.float32), 8],
        kwargs_generator=lambda: {
            "traditional": False,
            "base": 10000.0,
            "scale": 1.0,
            "offset": 0,
        },
    )


def test_CudaKernel_parity() -> None:
    """Test parity for CudaKernel."""
    check_parity("CudaKernel", lambda: [np.random.randn(2, 3).astype(np.float32)])


def test_MetalKernel_parity() -> None:
    """Test parity for MetalKernel."""
    check_parity("MetalKernel", lambda: [np.random.randn(2, 3).astype(np.float32)])


def test_PrecompiledCudaKernel_parity() -> None:
    """Test parity for PrecompiledCudaKernel."""
    check_parity("PrecompiledCudaKernel", lambda: [np.random.randn(2, 3).astype(np.float32)])


def test_AdamaxHyperparams_parity() -> None:
    """Test parity for AdamaxHyperparams."""
    check_parity("AdamaxHyperparams", lambda: [np.random.randn(2, 3).astype(np.float32)])


def test_AdaptiveAvgPool3D_parity() -> None:
    """Test parity for AdaptiveAvgPool3D."""
    check_parity("AdaptiveAvgPool3D", lambda: [np.random.randn(1, 2, 2, 2, 1).astype(np.float32)])


def test_AdaptiveMaxPool3D_parity() -> None:
    """Test parity for AdaptiveMaxPool3D."""
    check_parity("AdaptiveMaxPool3D", lambda: [np.random.randn(1, 2, 2, 2, 1).astype(np.float32)])


def test_AdaptiveMaxPool3D_Indices_parity() -> None:
    """Test parity for AdaptiveMaxPool3D_Indices."""
    check_parity(
        "AdaptiveMaxPool3D_Indices",
        lambda: [np.random.randn(1, 2, 2, 2, 1).astype(np.float32)],
    )


def test_ApplyAdagrad_parity() -> None:
    """Test parity for ApplyAdagrad."""
    check_parity("ApplyAdagrad", lambda: [np.random.randn(2, 3).astype(np.float32)])


def test_ApplyAdam_parity() -> None:
    """Test parity for ApplyAdam."""
    check_parity("ApplyAdam", lambda: [np.random.randn(2, 3).astype(np.float32)])


def test_ApplyFtrl_parity() -> None:
    """Test parity for ApplyFtrl."""
    check_parity("ApplyFtrl", lambda: [np.random.randn(2, 3).astype(np.float32)])


def test_ApplyRMSProp_parity() -> None:
    """Test parity for ApplyRMSProp."""
    check_parity("ApplyRMSProp", lambda: [np.random.randn(2, 3).astype(np.float32)])


def test_ConvTransposeShapeTuple_parity() -> None:
    """Test parity for ConvTransposeShapeTuple."""
    check_parity("ConvTransposeShapeTuple", lambda: [np.random.randn(2, 3).astype(np.float32)])


def test_Dropout1d_parity() -> None:
    """Test parity for Dropout1d."""
    check_parity("Dropout1d", lambda: [np.random.randn(2, 3).astype(np.float32)])


def test_FractionalAvgPool_parity() -> None:
    """Test parity for FractionalAvgPool."""
    check_parity("FractionalAvgPool", lambda: [np.random.randn(1, 4, 4, 1).astype(np.float32)])


def test_FractionalMaxPool_parity() -> None:
    """Test parity for FractionalMaxPool."""
    check_parity("FractionalMaxPool", lambda: [np.random.randn(1, 4, 4, 1).astype(np.float32)])


def test_FractionalMaxPool3D_parity() -> None:
    """Test parity for FractionalMaxPool3D."""
    check_parity(
        "FractionalMaxPool3D",
        lambda: [np.random.randn(1, 2, 2, 2, 1).astype(np.float32)],
    )


def test_FractionalMaxPool3D_Indices_parity() -> None:
    """Test parity for FractionalMaxPool3D_Indices."""
    check_parity(
        "FractionalMaxPool3D_Indices",
        lambda: [np.random.randn(1, 2, 2, 2, 1).astype(np.float32)],
    )


def test_Iou_parity() -> None:
    """Test parity for Iou."""
    check_parity("Iou", lambda: [np.random.randn(2, 4).astype(np.float32)])


def test_LionConfig_parity() -> None:
    """Test parity for LionConfig."""
    check_parity("LionConfig", lambda: [np.random.randn(2, 3).astype(np.float32)])


def test_LogSigmoid_parity() -> None:
    """Test parity for LogSigmoid."""
    check_parity("LogSigmoid", lambda: [np.random.randn(2, 3).astype(np.float32)])


def test_LogSoftmax_parity() -> None:
    """Test parity for LogSoftmax."""
    check_parity("LogSoftmax", lambda: [np.random.randn(2, 3).astype(np.float32)])


def test_MaxPoolWithIndices_parity() -> None:
    """Test parity for MaxPoolWithIndices."""
    check_parity("MaxPoolWithIndices", lambda: [np.random.randn(1, 2, 2, 1).astype(np.float32)])


def test_MaxPoolWithIndices_Indices_parity() -> None:
    """Test parity for MaxPoolWithIndices_Indices."""
    check_parity(
        "MaxPoolWithIndices_Indices",
        lambda: [np.random.randn(1, 2, 2, 1).astype(np.float32)],
    )


def test_MaxUnpool1D_parity() -> None:
    """Test parity for MaxUnpool1D."""
    check_parity("MaxUnpool1D", lambda: [np.random.randn(1, 2, 1).astype(np.float32)])


def test_MaxUnpool2D_parity() -> None:
    """Test parity for MaxUnpool2D."""
    check_parity("MaxUnpool2D", lambda: [np.random.randn(1, 2, 2, 1).astype(np.float32)])


def test_MaxUnpool3D_parity() -> None:
    """Test parity for MaxUnpool3D."""
    check_parity("MaxUnpool3D", lambda: [np.random.randn(1, 2, 2, 2, 1).astype(np.float32)])


def test_Nms_parity() -> None:
    """Test parity for Nms."""
    check_parity("Nms", lambda: [np.random.randn(2, 4).astype(np.float32)])


def test_Normalization_parity() -> None:
    """Test parity for Normalization."""
    check_parity("Normalization", lambda: [np.random.randn(2, 3).astype(np.float32)])


def test_ScaledDotProductAttention_parity() -> None:
    """Test parity for ScaledDotProductAttention."""
    check_parity(
        "ScaledDotProductAttention",
        lambda: [
            np.ones((1, 2, 3, 4), dtype=np.float32),
            np.ones((1, 2, 3, 4), dtype=np.float32),
            np.ones((1, 2, 3, 4), dtype=np.float32),
        ],
    )


def test_Smm_parity() -> None:
    """Test parity for Smm."""
    check_parity("Smm", lambda: [np.random.randn(2, 2).astype(np.float32)])


def test_Softsign_parity() -> None:
    """Test parity for Softsign."""
    check_parity("Softsign", lambda: [np.random.randn(2, 3).astype(np.float32)])


def test_SpaceToBatch_parity() -> None:
    """Test parity for SpaceToBatch."""
    check_parity("SpaceToBatch", lambda: [np.random.randn(1, 2, 2, 1).astype(np.float32)])


def test_SpaceToBatchND_parity() -> None:
    """Test parity for SpaceToBatchND."""
    check_parity("SpaceToBatchND", lambda: [np.random.randn(1, 2, 2, 1).astype(np.float32)])


def test_SparseConcat_parity() -> None:
    """Test parity for SparseConcat."""
    check_parity("SparseConcat", lambda: [np.random.randn(2, 3).astype(np.float32)])


def test_SparsePlus_parity() -> None:
    """Test parity for SparsePlus."""
    check_parity("SparsePlus", lambda: [np.random.randn(2, 3).astype(np.float32)])


def test_SparseSampledAdd_parity() -> None:
    """Test parity for SparseSampledAdd."""
    check_parity("SparseSampledAdd", lambda: [np.random.randn(2, 3).astype(np.float32)])


def test_SparseSigmoid_parity() -> None:
    """Test parity for SparseSigmoid."""
    check_parity("SparseSigmoid", lambda: [np.random.randn(2, 3).astype(np.float32)])


def test_SparseSplit_parity() -> None:
    """Test parity for SparseSplit."""
    check_parity("SparseSplit", lambda: [np.random.randn(2, 3).astype(np.float32)])


def test_SparseToDense_parity() -> None:
    """Test parity for SparseToDense."""
    check_parity("SparseToDense", lambda: [np.random.randn(2, 3).astype(np.float32)])
