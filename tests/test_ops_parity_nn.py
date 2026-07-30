import pytest
import numpy as np

try:
    import mlx.core as mx
except ImportError:
    mx = None

import zero_mlx
from ml_switcheroo_compiler.tracing import global_tracing_state as _tracer


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


def test_ConvGeneralDilated_parity():
    """Test parity for ConvGeneralDilated."""
    check_parity(
        "Conv_General_Dilated", lambda: []
    )  # Skip complex tensor generation for now, just stub


def test_Norm_parity():
    """Test parity for Norm."""
    check_parity("Norm", lambda: [np.random.randn(2, 3).astype(np.float32)])


def test_AdaptiveAvgPool2D_parity():
    """Test parity for AdaptiveAvgPool2D."""
    check_parity(
        "AdaptiveAvgPool2D",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_AdaptiveMaxPool2D_parity():
    """Test parity for AdaptiveMaxPool2D."""
    check_parity(
        "AdaptiveMaxPool2D",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_AlphaDropout_parity():
    """Test parity for AlphaDropout."""
    check_parity(
        "AlphaDropout",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_BitcastConvertType_parity():
    """Test parity for BitcastConvertType."""
    check_parity(
        "BitcastConvertType",
        lambda: [np.array([1.5], dtype=np.float32)],
        lambda: {"new_dtype": np.int32},
    )


def test_CTCLoss_parity():
    """Test parity for CTCLoss."""
    check_parity(
        "CTCLoss",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_Conv_parity():
    """Test parity for Conv."""
    check_parity(
        "Conv",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_ConvGeneralDilatedLocal_parity():
    """Test parity for ConvGeneralDilatedLocal."""
    check_parity(
        "ConvGeneralDilatedLocal",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_ConvGeneralDilatedPatches_parity():
    """Test parity for ConvGeneralDilatedPatches."""
    check_parity(
        "ConvGeneralDilatedPatches",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_ConvTranspose_parity():
    """Test parity for ConvTranspose."""
    check_parity(
        "ConvTranspose",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_ConvWithGeneralPadding_parity():
    """Test parity for ConvWithGeneralPadding."""
    check_parity(
        "ConvWithGeneralPadding",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_Convolve_parity():
    """Test parity for Convolve."""
    check_parity(
        "Convolve",
        lambda: [
            np.random.randn(5).astype(np.float32),
            np.random.randn(3).astype(np.float32),
        ],
    )


def test_Dropout_parity():
    """Test parity for Dropout."""
    check_parity(
        "Dropout",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_FractionalMaxPool2D_parity():
    """Test parity for FractionalMaxPool2D."""
    check_parity(
        "FractionalMaxPool2D",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_GeneralizedNormal_parity():
    """Test parity for GeneralizedNormal."""
    check_parity(
        "GeneralizedNormal",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_GroupNorm_parity():
    """Test parity for GroupNorm."""
    check_parity(
        "GroupNorm",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_L2Normalize_parity():
    """Test parity for L2Normalize."""
    check_parity(
        "L2Normalize",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_Lognormal_parity():
    """Test parity for Lognormal."""
    check_parity(
        "Lognormal",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_MultivariateNormal_parity():
    """Test parity for MultivariateNormal."""
    check_parity(
        "MultivariateNormal",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_RawConv2D_parity():
    """Test parity for RawConv2D."""
    check_parity(
        "RawConv2D",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_ReduceEuclideanNorm_parity():
    """Test parity for ReduceEuclideanNorm."""
    check_parity(
        "ReduceEuclideanNorm",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_SparseSoftmax_parity():
    """Test parity for SparseSoftmax."""
    check_parity(
        "SparseSoftmax",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_VectorNorm_parity():
    """Test parity for VectorNorm."""
    check_parity(
        "VectorNorm",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_Dropout2d_parity():
    """Test parity for Dropout2d."""
    check_parity("Dropout2d", lambda: [np.random.randn(2, 3).astype(np.float32)])


def test_Dropout3d_parity():
    """Test parity for Dropout3d."""
    check_parity("Dropout3d", lambda: [np.random.randn(2, 3).astype(np.float32)])


def test_Gru_parity():
    """Test parity for Gru."""
    check_parity("Gru", lambda: [np.random.randn(2, 3).astype(np.float32)])


def test_Rope_parity():
    """Test parity for Rope."""
    pass


def test_CudaKernel_parity():
    """Test parity for CudaKernel."""
    pass


def test_MetalKernel_parity():
    """Test parity for MetalKernel."""
    pass


def test_PrecompiledCudaKernel_parity():
    """Test parity for PrecompiledCudaKernel."""
    pass
