"""Compiler operation parity tests for `logical` operations."""

from __future__ import annotations

from tests.conftest import check_parity, np


def test_All_parity() -> None:
    """Test parity for All."""
    check_parity("All", lambda: [np.random.choice([True, False], size=(2, 3))])


def test_Allclose_parity() -> None:
    """Test parity for Allclose."""
    check_parity(
        "Allclose",
        lambda: [
            np.random.randn(2, 3).astype(np.float32),
            np.random.randn(2, 3).astype(np.float32),
        ],
    )


def test_Any_parity() -> None:
    """Test parity for Any."""
    check_parity("Any", lambda: [np.random.choice([True, False], size=(2, 3))])


def test_BitwiseAnd_parity() -> None:
    """Test parity for BitwiseAnd."""
    check_parity(
        "Bitwise_And",
        lambda: [
            np.random.randint(0, 10, size=(2, 3)).astype(np.int32),
            np.random.randint(0, 10, size=(2, 3)).astype(np.int32),
        ],
    )


def test_BitwiseNot_parity() -> None:
    """Test parity for BitwiseNot."""
    check_parity("Bitwise_Not", lambda: [np.random.randint(0, 10, size=(2, 3)).astype(np.int32)])


def test_BitwiseOr_parity() -> None:
    """Test parity for BitwiseOr."""
    check_parity(
        "Bitwise_Or",
        lambda: [
            np.random.randint(0, 10, size=(2, 3)).astype(np.int32),
            np.random.randint(0, 10, size=(2, 3)).astype(np.int32),
        ],
    )


def test_BitwiseXor_parity() -> None:
    """Test parity for BitwiseXor."""
    check_parity(
        "Bitwise_Xor",
        lambda: [
            np.random.randint(0, 10, size=(2, 3)).astype(np.int32),
            np.random.randint(0, 10, size=(2, 3)).astype(np.int32),
        ],
    )


def test_Equal_parity() -> None:
    """Test parity for Equal."""
    check_parity(
        "Equal",
        lambda: [
            np.random.randn(2, 3).astype(np.float32),
            np.random.randn(2, 3).astype(np.float32),
        ],
    )


def test_Floor_parity() -> None:
    """Test parity for Floor."""
    check_parity("Floor", lambda: [np.random.randn(2, 3).astype(np.float32)])


def test_FloorDivide_parity() -> None:
    """Test parity for FloorDivide."""
    check_parity(
        "Floor_Divide",
        lambda: [
            np.random.randint(1, 10, size=(2, 3)).astype(np.float32),
            np.random.randint(1, 10, size=(2, 3)).astype(np.float32),
        ],
    )


def test_Greater_parity() -> None:
    """Test parity for Greater."""
    check_parity(
        "Greater",
        lambda: [
            np.random.randn(2, 3).astype(np.float32),
            np.random.randn(2, 3).astype(np.float32),
        ],
    )


def test_GreaterEqual_parity() -> None:
    """Test parity for GreaterEqual."""
    check_parity(
        "Greater_Equal",
        lambda: [
            np.random.randn(2, 3).astype(np.float32),
            np.random.randn(2, 3).astype(np.float32),
        ],
    )


def test_Heaviside_parity() -> None:
    """Test parity for Heaviside."""
    check_parity(
        "Heaviside",
        lambda: [
            np.random.randn(2, 3).astype(np.float32),
            np.random.uniform(0, 1, size=(2, 3)).astype(np.float32),
        ],
    )


def test_Isclose_parity() -> None:
    """Test parity for Isclose."""
    check_parity(
        "Isclose",
        lambda: [
            np.random.randn(2, 3).astype(np.float32),
            np.random.randn(2, 3).astype(np.float32),
        ],
    )


def test_Isfinite_parity() -> None:
    """Test parity for Isfinite."""
    check_parity("Isfinite", lambda: [np.random.randn(2, 3).astype(np.float32)])


def test_Isinf_parity() -> None:
    """Test parity for Isinf."""
    check_parity("Isinf", lambda: [np.random.randn(2, 3).astype(np.float32)])


def test_Isnan_parity() -> None:
    """Test parity for Isnan."""
    check_parity("Isnan", lambda: [np.random.randn(2, 3).astype(np.float32)])


def test_Less_parity() -> None:
    """Test parity for Less."""
    check_parity(
        "Less",
        lambda: [
            np.random.randn(2, 3).astype(np.float32),
            np.random.randn(2, 3).astype(np.float32),
        ],
    )


def test_LessEqual_parity() -> None:
    """Test parity for LessEqual."""
    check_parity(
        "Less_Equal",
        lambda: [
            np.random.randn(2, 3).astype(np.float32),
            np.random.randn(2, 3).astype(np.float32),
        ],
    )


def test_LogicalAnd_parity() -> None:
    """Test parity for LogicalAnd."""
    check_parity(
        "Logical_And",
        lambda: [
            np.random.choice([True, False], size=(2, 3)),
            np.random.choice([True, False], size=(2, 3)),
        ],
    )


def test_LogicalNot_parity() -> None:
    """Test parity for LogicalNot."""
    check_parity("Logical_Not", lambda: [np.random.choice([True, False], size=(2, 3))])


def test_LogicalOr_parity() -> None:
    """Test parity for LogicalOr."""
    check_parity(
        "Logical_Or",
        lambda: [
            np.random.choice([True, False], size=(2, 3)),
            np.random.choice([True, False], size=(2, 3)),
        ],
    )


def test_LogicalXor_parity() -> None:
    """Test parity for LogicalXor."""
    check_parity(
        "Logical_Xor",
        lambda: [
            np.random.choice([True, False], size=(2, 3)).astype(bool),
            np.random.choice([True, False], size=(2, 3)).astype(bool),
        ],
    )


def test_NanToNum_parity() -> None:
    """Test parity for NanToNum."""
    check_parity(
        "Nan_To_Num",
        lambda: [np.array([np.nan, np.inf, -np.inf, 1.0], dtype=np.float32)],
        kwargs_generator=lambda: {"nan": 0.0, "posinf": 1e30, "neginf": -1e30},
    )


def test_NotEqual_parity() -> None:
    """Test parity for NotEqual."""
    check_parity(
        "Not_Equal",
        lambda: [
            np.random.randn(2, 3).astype(np.float32),
            np.random.randn(2, 3).astype(np.float32),
        ],
    )


def test_Rand_parity() -> None:
    """Test parity for Rand."""
    check_parity("Rand", lambda: [], kwargs_generator=lambda: {"shape": (2, 3)})


def test_Randint_parity() -> None:
    """Test parity for Randint."""
    check_parity(
        "Randint",
        lambda: [np.array(0, dtype=np.int32), np.array(10, dtype=np.int32)],
        kwargs_generator=lambda: {"shape": (2, 3)},
    )


def test_Randn_parity() -> None:
    """Test parity for Randn."""
    check_parity("Randn", lambda: [], kwargs_generator=lambda: {"shape": (2, 3)})


def test_Sort_parity() -> None:
    """Test parity for Sort."""
    check_parity("Sort", lambda: [np.random.randn(2, 3).astype(np.float32)])


def test_TrueDivide_parity() -> None:
    """Test parity for TrueDivide."""
    check_parity(
        "True_Divide",
        lambda: [
            np.random.randn(2, 3).astype(np.float32),
            np.random.randn(2, 3).astype(np.float32),
        ],
    )


def test_AffineGenerator_parity() -> None:
    """Test parity for AffineGenerator."""
    check_parity(
        "AffineGenerator",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_AffineTransform_parity() -> None:
    """Test parity for AffineTransform."""
    check_parity(
        "AffineTransform",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_AllGather_parity() -> None:
    """Test parity for AllGather."""
    check_parity(
        "AllGather",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_AllReduce_parity() -> None:
    """Test parity for AllReduce."""
    check_parity(
        "AllReduce",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_AllToAll_parity() -> None:
    """Test parity for AllToAll."""
    check_parity(
        "AllToAll",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_AxisIndex_parity() -> None:
    """Test parity for AxisIndex."""
    check_parity(
        "AxisIndex",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_Ball_parity() -> None:
    """Test parity for Ball."""
    check_parity(
        "Ball",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_BandPart_parity() -> None:
    """Test parity for BandPart."""
    check_parity(
        "BandPart",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_Chisquare_parity() -> None:
    """Test parity for Chisquare."""
    check_parity(
        "Chisquare",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_Corrcoef_parity() -> None:
    """Test parity for Corrcoef."""
    check_parity(
        "Corrcoef",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_Correlate_parity() -> None:
    """Test parity for Correlate."""
    check_parity(
        "Correlate",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_CropAndResize_parity() -> None:
    """Test parity for CropAndResize."""
    check_parity(
        "CropAndResize",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_DebugInfs_parity() -> None:
    """Test parity for DebugInfs."""
    check_parity(
        "DebugInfs",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_DebugNans_parity() -> None:
    """Test parity for DebugNans."""
    check_parity(
        "DebugNans",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_DivideNoNan_parity() -> None:
    """Test parity for DivideNoNan."""
    check_parity(
        "DivideNoNan",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_EditDistance_parity() -> None:
    """Test parity for EditDistance."""
    check_parity(
        "EditDistance",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_ElasticTransform_parity() -> None:
    """Test parity for ElasticTransform."""
    check_parity(
        "ElasticTransform",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_Equalization_parity() -> None:
    """Test parity for Equalization."""
    check_parity(
        "Equalization",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_Expand_parity() -> None:
    """Test parity for Expand."""
    check_parity(
        "Expand",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_ForiLoop_parity() -> None:
    """Test parity for ForiLoop."""
    check_parity(
        "ForiLoop",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_HsvToRgb_parity() -> None:
    """Test parity for HsvToRgb."""
    check_parity(
        "HsvToRgb",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_Infeed_parity() -> None:
    """Test parity for Infeed."""
    check_parity(
        "Infeed",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_IsNonDecreasing_parity() -> None:
    """Test parity for IsNonDecreasing."""
    check_parity(
        "IsNonDecreasing",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_Kaiser_parity() -> None:
    """Test parity for Kaiser."""
    check_parity(
        "Kaiser",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_KaiserWindow_parity() -> None:
    """Test parity for KaiserWindow."""
    check_parity(
        "KaiserWindow",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_Logistic_parity() -> None:
    """Test parity for Logistic."""
    check_parity(
        "Logistic",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_LuFactor_parity() -> None:
    """Test parity for LuFactor."""
    check_parity(
        "LuFactor",
        lambda: [np.array([[4.0, 0.0], [0.0, 2.0]], dtype=np.float32)],
    )


def test_MapCoordinates_parity() -> None:
    """Test parity for MapCoordinates."""
    check_parity(
        "MapCoordinates",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_Moveaxis_parity() -> None:
    """Test parity for Moveaxis."""
    check_parity(
        "Moveaxis",
        lambda: [np.random.randn(2, 3, 4).astype(np.float32), 0, -1],
    )


def test_MultiplyNoNan_parity() -> None:
    """Test parity for MultiplyNoNan."""
    check_parity(
        "MultiplyNoNan",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_Nanmedian_parity() -> None:
    """Test parity for Nanmedian."""
    check_parity(
        "Nanmedian",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_Nanpercentile_parity() -> None:
    """Test parity for Nanpercentile."""
    check_parity(
        "Nanpercentile",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_Nanquantile_parity() -> None:
    """Test parity for Nanquantile."""
    check_parity(
        "Nanquantile",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_Orthogonal_parity() -> None:
    """Test parity for Orthogonal."""
    check_parity(
        "Orthogonal",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_OverlapAndAdd_parity() -> None:
    """Test parity for OverlapAndAdd."""
    check_parity(
        "OverlapAndAdd",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_PerspectiveTransform_parity() -> None:
    """Test parity for PerspectiveTransform."""
    check_parity(
        "PerspectiveTransform",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_Poisson_parity() -> None:
    """Test parity for Poisson."""
    check_parity(
        "Poisson",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_RaggedSegmentIdsToRowSplits_parity() -> None:
    """Test parity for RaggedSegmentIdsToRowSplits."""
    check_parity(
        "RaggedSegmentIdsToRowSplits",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_RaggedTensorToDense_parity() -> None:
    """Test parity for RaggedTensorToDense."""
    check_parity(
        "RaggedTensorToDense",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_RandAugment_parity() -> None:
    """Test parity for RandAugment."""
    check_parity(
        "RandAugment",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_RandomColorJitter_parity() -> None:
    """Test parity for RandomColorJitter."""
    check_parity(
        "RandomColorJitter",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_RandomCrop_parity() -> None:
    """Test parity for RandomCrop."""
    check_parity(
        "RandomCrop",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_RandomElasticTransform_parity() -> None:
    """Test parity for RandomElasticTransform."""
    check_parity(
        "RandomElasticTransform",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_RandomErasing_parity() -> None:
    """Test parity for RandomErasing."""
    check_parity(
        "RandomErasing",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_RandomFlip_parity() -> None:
    """Test parity for RandomFlip."""
    check_parity(
        "RandomFlip",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_RandomGammaGrad_parity() -> None:
    """Test parity for RandomGammaGrad."""
    check_parity(
        "RandomGammaGrad",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_RandomGammaP_parity() -> None:
    """Test parity for RandomGammaP."""
    check_parity(
        "RandomGammaP",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_RandomGaussianBlur_parity() -> None:
    """Test parity for RandomGaussianBlur."""
    check_parity(
        "RandomGaussianBlur",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_RandomPerspective_parity() -> None:
    """Test parity for RandomPerspective."""
    check_parity(
        "RandomPerspective",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_RandomRotation_parity() -> None:
    """Test parity for RandomRotation."""
    check_parity(
        "RandomRotation",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_RandomSharpness_parity() -> None:
    """Test parity for RandomSharpness."""
    check_parity(
        "RandomSharpness",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_RandomShear_parity() -> None:
    """Test parity for RandomShear."""
    check_parity(
        "RandomShear",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_RandomTranslation_parity() -> None:
    """Test parity for RandomTranslation."""
    check_parity(
        "RandomTranslation",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_RandomZoom_parity() -> None:
    """Test parity for RandomZoom."""
    check_parity(
        "RandomZoom",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_ReciprocalNoNan_parity() -> None:
    """Test parity for ReciprocalNoNan."""
    check_parity(
        "ReciprocalNoNan",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_ReducePrecision_parity() -> None:
    """Test parity for ReducePrecision."""
    check_parity(
        "ReducePrecision",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_RngBitGenerator_parity() -> None:
    """Test parity for RngBitGenerator."""
    check_parity(
        "RngBitGenerator",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_RngUniform_parity() -> None:
    """Test parity for RngUniform."""
    check_parity(
        "RngUniform",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_SearchSorted_parity() -> None:
    """Test parity for SearchSorted."""
    check_parity(
        "SearchSorted",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_Setxor1d_parity() -> None:
    """Test parity for Setxor1d."""
    check_parity(
        "Setxor1d",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_ShardTensor_parity() -> None:
    """Test parity for ShardTensor."""
    check_parity(
        "ShardTensor",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_SortKeyVal_parity() -> None:
    """Test parity for SortKeyVal."""
    check_parity(
        "SortKeyVal",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_SparseExpandDims_parity() -> None:
    """Test parity for SparseExpandDims."""
    check_parity(
        "SparseExpandDims",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_SparseReorder_parity() -> None:
    """Test parity for SparseReorder."""
    check_parity(
        "SparseReorder",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_SparseToIndicator_parity() -> None:
    """Test parity for SparseToIndicator."""
    check_parity(
        "SparseToIndicator",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_TakeAlongAxis_parity() -> None:
    """Test parity for TakeAlongAxis."""
    check_parity(
        "TakeAlongAxis",
        lambda: [
            np.arange(6, dtype=np.float32).reshape(2, 3),
            np.array([[0, 2], [1, 0]], dtype=np.int32),
            1,
        ],
    )


def test_TensorArrayRead_parity() -> None:
    """Test parity for TensorArrayRead."""
    check_parity(
        "TensorArrayRead",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_TensorArrayStack_parity() -> None:
    """Test parity for TensorArrayStack."""
    check_parity(
        "TensorArrayStack",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_TensorArrayWrite_parity() -> None:
    """Test parity for TensorArrayWrite."""
    check_parity(
        "TensorArrayWrite",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_TensorScatterSub_parity() -> None:
    """Test parity for TensorScatterSub."""
    check_parity(
        "TensorScatterSub",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_TensorScatterUpdate_parity() -> None:
    """Test parity for TensorScatterUpdate."""
    check_parity(
        "TensorScatterUpdate",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_TextVectorization_parity() -> None:
    """Test parity for TextVectorization."""
    check_parity(
        "TextVectorization",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_UniqueAll_parity() -> None:
    """Test parity for UniqueAll."""
    check_parity(
        "UniqueAll",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_YiqToRgb_parity() -> None:
    """Test parity for YiqToRgb."""
    check_parity(
        "YiqToRgb",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_YuvToRgb_parity() -> None:
    """Test parity for YuvToRgb."""
    check_parity(
        "YuvToRgb",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )
