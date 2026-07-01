import pytest
import numpy as np

try:
    import mlx.core as mx
except ImportError:
    mx = None

import zero_mlx
from ml_switcheroo_compiler.tracing import _tracer


def assert_allclose_mlx(z_res, m_res, rtol=1e-5, atol=1e-5):
    import numpy as np

    if isinstance(z_res, (tuple, list)) and isinstance(m_res, (tuple, list)):
        for z, m in zip(z_res, m_res):
            assert_allclose_mlx(z, m, rtol, atol)
    else:
        # Extract underlying tensor data if it's a zero_mlx array
        if hasattr(z_res, "_tensor"):
            z_res = z_res._tensor.data
        if hasattr(m_res, "_tensor"):
            m_res = m_res._tensor.data
        np.testing.assert_allclose(
            np.array(z_res), np.array(m_res), rtol=rtol, atol=atol
        )


def check_parity(op_name, args_generator, kwargs_generator=None, rtol=1e-5, atol=1e-5):
    if mx is None:
        pytest.skip("MLX not available")
    func_name = op_name.lower()
    if not hasattr(zero_mlx, func_name) or not hasattr(mx, func_name):
        pytest.skip(f"{func_name} not mapped in zero_mlx or mx")

    if func_name == "fft":
        z_func = zero_mlx.fft.fft
        try:
            m_func = mx.fft.fft
            if not callable(m_func):
                m_func = getattr(mx, "fft")
        except AttributeError:
            m_func = getattr(mx, "fft")
    elif func_name == "rfft":
        z_func = zero_mlx.fft.rfft
        try:
            m_func = mx.fft.rfft
            if not callable(m_func):
                m_func = getattr(mx, "rfft")
        except AttributeError:
            m_func = getattr(mx, "rfft")
    else:
        z_func = getattr(zero_mlx, func_name)
        m_func = getattr(mx, func_name)

    args = args_generator()
    kwargs = kwargs_generator() if kwargs_generator else {}

    _tracer.start_tracing("parity_test")

    def unwrap(x):
        if isinstance(x, np.generic):
            return x.item()
        return x

    z_args = [
        zero_mlx.array(a) if isinstance(a, np.ndarray) else unwrap(a) for a in args
    ]
    m_args = [mx.array(a) if isinstance(a, np.ndarray) else unwrap(a) for a in args]

    try:
        z_res = z_func(*z_args, **kwargs)
        if hasattr(zero_mlx, "eval"):
            zero_mlx.eval(z_res)
        _tracer.stop_tracing()
        m_res = m_func(*m_args, **kwargs)
        assert_allclose_mlx(z_res, m_res, rtol=rtol, atol=atol)
    except Exception as e:
        try:
            if callable(getattr(_tracer, "is_tracing", None)):
                if _tracer.is_tracing():
                    _tracer.stop_tracing()
            elif getattr(_tracer, "is_tracing", False):
                _tracer.stop_tracing()
        except Exception:
            pass
        if False:
            _tracer.stop_tracing()
        raise e


def test_All_parity():
    """Test parity for All."""
    check_parity("All", lambda: [np.random.choice([True, False], size=(2, 3))])


def test_Allclose_parity():
    """Test parity for Allclose."""
    check_parity(
        "Allclose",
        lambda: [
            np.random.randn(2, 3).astype(np.float32),
            np.random.randn(2, 3).astype(np.float32),
        ],
    )


def test_Any_parity():
    """Test parity for Any."""
    check_parity("Any", lambda: [np.random.choice([True, False], size=(2, 3))])


def test_BitwiseAnd_parity():
    """Test parity for BitwiseAnd."""
    check_parity(
        "Bitwise_And",
        lambda: [
            np.random.randint(0, 10, size=(2, 3)).astype(np.int32),
            np.random.randint(0, 10, size=(2, 3)).astype(np.int32),
        ],
    )


def test_BitwiseNot_parity():
    """Test parity for BitwiseNot."""
    check_parity(
        "Bitwise_Not", lambda: [np.random.randint(0, 10, size=(2, 3)).astype(np.int32)]
    )


def test_BitwiseOr_parity():
    """Test parity for BitwiseOr."""
    check_parity(
        "Bitwise_Or",
        lambda: [
            np.random.randint(0, 10, size=(2, 3)).astype(np.int32),
            np.random.randint(0, 10, size=(2, 3)).astype(np.int32),
        ],
    )


def test_BitwiseXor_parity():
    """Test parity for BitwiseXor."""
    check_parity(
        "Bitwise_Xor",
        lambda: [
            np.random.randint(0, 10, size=(2, 3)).astype(np.int32),
            np.random.randint(0, 10, size=(2, 3)).astype(np.int32),
        ],
    )


def test_Equal_parity():
    """Test parity for Equal."""
    check_parity(
        "Equal",
        lambda: [
            np.random.randn(2, 3).astype(np.float32),
            np.random.randn(2, 3).astype(np.float32),
        ],
    )


def test_Floor_parity():
    """Test parity for Floor."""
    check_parity("Floor", lambda: [np.random.randn(2, 3).astype(np.float32)])


def test_FloorDivide_parity():
    """Test parity for FloorDivide."""
    check_parity(
        "Floor_Divide",
        lambda: [
            np.random.randint(1, 10, size=(2, 3)).astype(np.float32),
            np.random.randint(1, 10, size=(2, 3)).astype(np.float32),
        ],
    )


def test_Greater_parity():
    """Test parity for Greater."""
    check_parity(
        "Greater",
        lambda: [
            np.random.randn(2, 3).astype(np.float32),
            np.random.randn(2, 3).astype(np.float32),
        ],
    )


def test_GreaterEqual_parity():
    """Test parity for GreaterEqual."""
    check_parity(
        "Greater_Equal",
        lambda: [
            np.random.randn(2, 3).astype(np.float32),
            np.random.randn(2, 3).astype(np.float32),
        ],
    )


def test_Heaviside_parity():
    """Test parity for Heaviside."""
    check_parity(
        "Heaviside",
        lambda: [
            np.random.randn(2, 3).astype(np.float32),
            np.random.uniform(0, 1, size=(2, 3)).astype(np.float32),
        ],
    )


def test_Isclose_parity():
    """Test parity for Isclose."""
    check_parity(
        "Isclose",
        lambda: [
            np.random.randn(2, 3).astype(np.float32),
            np.random.randn(2, 3).astype(np.float32),
        ],
    )


def test_Isfinite_parity():
    """Test parity for Isfinite."""
    check_parity("Isfinite", lambda: [np.random.randn(2, 3).astype(np.float32)])


def test_Isinf_parity():
    """Test parity for Isinf."""
    check_parity("Isinf", lambda: [np.random.randn(2, 3).astype(np.float32)])


def test_Isnan_parity():
    """Test parity for Isnan."""
    check_parity("Isnan", lambda: [np.random.randn(2, 3).astype(np.float32)])


def test_Less_parity():
    """Test parity for Less."""
    check_parity(
        "Less",
        lambda: [
            np.random.randn(2, 3).astype(np.float32),
            np.random.randn(2, 3).astype(np.float32),
        ],
    )


def test_LessEqual_parity():
    """Test parity for LessEqual."""
    check_parity(
        "Less_Equal",
        lambda: [
            np.random.randn(2, 3).astype(np.float32),
            np.random.randn(2, 3).astype(np.float32),
        ],
    )


def test_LogicalAnd_parity():
    """Test parity for LogicalAnd."""
    check_parity(
        "Logical_And",
        lambda: [
            np.random.choice([True, False], size=(2, 3)),
            np.random.choice([True, False], size=(2, 3)),
        ],
    )


def test_LogicalNot_parity():
    """Test parity for LogicalNot."""
    check_parity("Logical_Not", lambda: [np.random.choice([True, False], size=(2, 3))])


def test_LogicalOr_parity():
    """Test parity for LogicalOr."""
    check_parity(
        "Logical_Or",
        lambda: [
            np.random.choice([True, False], size=(2, 3)),
            np.random.choice([True, False], size=(2, 3)),
        ],
    )


def test_LogicalXor_parity():
    """Test parity for LogicalXor."""
    check_parity(
        "Logical_Xor",
        lambda: [
            np.random.choice([True, False], size=(2, 3)).astype(bool),
            np.random.choice([True, False], size=(2, 3)).astype(bool),
        ],
    )


def test_NanToNum_parity():
    """Test parity for NanToNum."""
    check_parity(
        "Nan_To_Num",
        lambda: [np.array([np.nan, np.inf, -np.inf, 1.0], dtype=np.float32)],
        kwargs_generator=lambda: {"nan": 0.0, "posinf": 1e30, "neginf": -1e30},
    )


def test_NotEqual_parity():
    """Test parity for NotEqual."""
    check_parity(
        "Not_Equal",
        lambda: [
            np.random.randn(2, 3).astype(np.float32),
            np.random.randn(2, 3).astype(np.float32),
        ],
    )


def test_Rand_parity():
    """Test parity for Rand."""
    check_parity("Rand", lambda: [], kwargs_generator=lambda: {"shape": (2, 3)})


def test_Randint_parity():
    """Test parity for Randint."""
    check_parity(
        "Randint",
        lambda: [np.array(0, dtype=np.int32), np.array(10, dtype=np.int32)],
        kwargs_generator=lambda: {"shape": (2, 3)},
    )


def test_Randn_parity():
    """Test parity for Randn."""
    check_parity("Randn", lambda: [], kwargs_generator=lambda: {"shape": (2, 3)})


def test_Sort_parity():
    """Test parity for Sort."""
    check_parity("Sort", lambda: [np.random.randn(2, 3).astype(np.float32)])


def test_TrueDivide_parity():
    """Test parity for TrueDivide."""
    check_parity(
        "True_Divide",
        lambda: [
            np.random.randn(2, 3).astype(np.float32),
            np.random.randn(2, 3).astype(np.float32),
        ],
    )


def test_AffineGenerator_parity():
    """Test parity for AffineGenerator."""
    check_parity(
        "AffineGenerator",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_AffineTransform_parity():
    """Test parity for AffineTransform."""
    check_parity(
        "AffineTransform",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_AllGather_parity():
    """Test parity for AllGather."""
    check_parity(
        "AllGather",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_AllReduce_parity():
    """Test parity for AllReduce."""
    check_parity(
        "AllReduce",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_AllToAll_parity():
    """Test parity for AllToAll."""
    check_parity(
        "AllToAll",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_AxisIndex_parity():
    """Test parity for AxisIndex."""
    check_parity(
        "AxisIndex",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_Ball_parity():
    """Test parity for Ball."""
    check_parity(
        "Ball",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_BandPart_parity():
    """Test parity for BandPart."""
    check_parity(
        "BandPart",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_Chisquare_parity():
    """Test parity for Chisquare."""
    check_parity(
        "Chisquare",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_Corrcoef_parity():
    """Test parity for Corrcoef."""
    check_parity(
        "Corrcoef",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_Correlate_parity():
    """Test parity for Correlate."""
    check_parity(
        "Correlate",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_CropAndResize_parity():
    """Test parity for CropAndResize."""
    check_parity(
        "CropAndResize",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_DebugInfs_parity():
    """Test parity for DebugInfs."""
    check_parity(
        "DebugInfs",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_DebugNans_parity():
    """Test parity for DebugNans."""
    check_parity(
        "DebugNans",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_DivideNoNan_parity():
    """Test parity for DivideNoNan."""
    check_parity(
        "DivideNoNan",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_EditDistance_parity():
    """Test parity for EditDistance."""
    check_parity(
        "EditDistance",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_ElasticTransform_parity():
    """Test parity for ElasticTransform."""
    check_parity(
        "ElasticTransform",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_Equalization_parity():
    """Test parity for Equalization."""
    check_parity(
        "Equalization",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_Expand_parity():
    """Test parity for Expand."""
    check_parity(
        "Expand",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_ForiLoop_parity():
    """Test parity for ForiLoop."""
    check_parity(
        "ForiLoop",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_HsvToRgb_parity():
    """Test parity for HsvToRgb."""
    check_parity(
        "HsvToRgb",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_Infeed_parity():
    """Test parity for Infeed."""
    check_parity(
        "Infeed",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_IsNonDecreasing_parity():
    """Test parity for IsNonDecreasing."""
    check_parity(
        "IsNonDecreasing",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_Kaiser_parity():
    """Test parity for Kaiser."""
    check_parity(
        "Kaiser",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_KaiserWindow_parity():
    """Test parity for KaiserWindow."""
    check_parity(
        "KaiserWindow",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_Logistic_parity():
    """Test parity for Logistic."""
    check_parity(
        "Logistic",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_LuFactor_parity():
    """Test parity for LuFactor."""
    check_parity(
        "LuFactor",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_MapCoordinates_parity():
    """Test parity for MapCoordinates."""
    check_parity(
        "MapCoordinates",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_Moveaxis_parity():
    """Test parity for Moveaxis."""
    check_parity(
        "Moveaxis",
        lambda: [np.random.randn(2, 3, 4).astype(np.float32), 0, -1],
    )


def test_MultiplyNoNan_parity():
    """Test parity for MultiplyNoNan."""
    check_parity(
        "MultiplyNoNan",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_Nanmedian_parity():
    """Test parity for Nanmedian."""
    check_parity(
        "Nanmedian",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_Nanpercentile_parity():
    """Test parity for Nanpercentile."""
    check_parity(
        "Nanpercentile",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_Nanquantile_parity():
    """Test parity for Nanquantile."""
    check_parity(
        "Nanquantile",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_Orthogonal_parity():
    """Test parity for Orthogonal."""
    check_parity(
        "Orthogonal",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_OverlapAndAdd_parity():
    """Test parity for OverlapAndAdd."""
    check_parity(
        "OverlapAndAdd",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_PerspectiveTransform_parity():
    """Test parity for PerspectiveTransform."""
    check_parity(
        "PerspectiveTransform",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_Poisson_parity():
    """Test parity for Poisson."""
    check_parity(
        "Poisson",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_RaggedSegmentIdsToRowSplits_parity():
    """Test parity for RaggedSegmentIdsToRowSplits."""
    check_parity(
        "RaggedSegmentIdsToRowSplits",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_RaggedTensorToDense_parity():
    """Test parity for RaggedTensorToDense."""
    check_parity(
        "RaggedTensorToDense",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_RandAugment_parity():
    """Test parity for RandAugment."""
    check_parity(
        "RandAugment",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_RandomColorJitter_parity():
    """Test parity for RandomColorJitter."""
    check_parity(
        "RandomColorJitter",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_RandomCrop_parity():
    """Test parity for RandomCrop."""
    check_parity(
        "RandomCrop",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_RandomElasticTransform_parity():
    """Test parity for RandomElasticTransform."""
    check_parity(
        "RandomElasticTransform",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_RandomErasing_parity():
    """Test parity for RandomErasing."""
    check_parity(
        "RandomErasing",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_RandomFlip_parity():
    """Test parity for RandomFlip."""
    check_parity(
        "RandomFlip",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_RandomGammaGrad_parity():
    """Test parity for RandomGammaGrad."""
    check_parity(
        "RandomGammaGrad",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_RandomGammaP_parity():
    """Test parity for RandomGammaP."""
    check_parity(
        "RandomGammaP",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_RandomGaussianBlur_parity():
    """Test parity for RandomGaussianBlur."""
    check_parity(
        "RandomGaussianBlur",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_RandomPerspective_parity():
    """Test parity for RandomPerspective."""
    check_parity(
        "RandomPerspective",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_RandomRotation_parity():
    """Test parity for RandomRotation."""
    check_parity(
        "RandomRotation",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_RandomSharpness_parity():
    """Test parity for RandomSharpness."""
    check_parity(
        "RandomSharpness",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_RandomShear_parity():
    """Test parity for RandomShear."""
    check_parity(
        "RandomShear",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_RandomTranslation_parity():
    """Test parity for RandomTranslation."""
    check_parity(
        "RandomTranslation",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_RandomZoom_parity():
    """Test parity for RandomZoom."""
    check_parity(
        "RandomZoom",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_ReciprocalNoNan_parity():
    """Test parity for ReciprocalNoNan."""
    check_parity(
        "ReciprocalNoNan",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_ReducePrecision_parity():
    """Test parity for ReducePrecision."""
    check_parity(
        "ReducePrecision",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_RngBitGenerator_parity():
    """Test parity for RngBitGenerator."""
    check_parity(
        "RngBitGenerator",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_RngUniform_parity():
    """Test parity for RngUniform."""
    check_parity(
        "RngUniform",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_SearchSorted_parity():
    """Test parity for SearchSorted."""
    check_parity(
        "SearchSorted",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_Setxor1d_parity():
    """Test parity for Setxor1d."""
    check_parity(
        "Setxor1d",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_ShardTensor_parity():
    """Test parity for ShardTensor."""
    check_parity(
        "ShardTensor",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_SortKeyVal_parity():
    """Test parity for SortKeyVal."""
    check_parity(
        "SortKeyVal",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_SparseExpandDims_parity():
    """Test parity for SparseExpandDims."""
    check_parity(
        "SparseExpandDims",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_SparseReorder_parity():
    """Test parity for SparseReorder."""
    check_parity(
        "SparseReorder",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_SparseToIndicator_parity():
    """Test parity for SparseToIndicator."""
    check_parity(
        "SparseToIndicator",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_TakeAlongAxis_parity():
    """Test parity for TakeAlongAxis."""
    check_parity(
        "TakeAlongAxis",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_TensorArrayRead_parity():
    """Test parity for TensorArrayRead."""
    check_parity(
        "TensorArrayRead",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_TensorArrayStack_parity():
    """Test parity for TensorArrayStack."""
    check_parity(
        "TensorArrayStack",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_TensorArrayWrite_parity():
    """Test parity for TensorArrayWrite."""
    check_parity(
        "TensorArrayWrite",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_TensorScatterSub_parity():
    """Test parity for TensorScatterSub."""
    check_parity(
        "TensorScatterSub",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_TensorScatterUpdate_parity():
    """Test parity for TensorScatterUpdate."""
    check_parity(
        "TensorScatterUpdate",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_TextVectorization_parity():
    """Test parity for TextVectorization."""
    check_parity(
        "TextVectorization",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_UniqueAll_parity():
    """Test parity for UniqueAll."""
    check_parity(
        "UniqueAll",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_YiqToRgb_parity():
    """Test parity for YiqToRgb."""
    check_parity(
        "YiqToRgb",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_YuvToRgb_parity():
    """Test parity for YuvToRgb."""
    check_parity(
        "YuvToRgb",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )
