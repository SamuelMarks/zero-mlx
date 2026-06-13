import pytest
import numpy as np

try:
    import mlx.core as mx
except ImportError:
    mx = None

import zero_mlx
from ml_switcheroo_compiler.tracing import _tracer


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
    z_args = [zero_mlx.array(a) if isinstance(a, np.ndarray) else a for a in args]
    m_args = [mx.array(a) if isinstance(a, np.ndarray) else a for a in args]

    try:
        z_res = z_func(*z_args, **kwargs)
        if hasattr(zero_mlx, "eval"):
            zero_mlx.eval(z_res)
    except Exception as e:
        pytest.fail(f"zero_mlx failed: {e}")

    m_res = m_func(*m_args, **kwargs)

    from evaluate_helper import assert_allclose_mlx

    assert_allclose_mlx(z_res, m_res, rtol=rtol, atol=atol)


def test_Abs_parity():
    """Test parity for Abs."""
    check_parity("Abs", lambda: [np.random.randn(2, 3).astype(np.float32)])


def test_Acos_parity():
    """Test parity for Acos."""
    check_parity(
        "Acos", lambda: [np.random.uniform(-1, 1, size=(2, 3)).astype(np.float32)]
    )


def test_Acosh_parity():
    """Test parity for Acosh."""
    check_parity(
        "Acosh", lambda: [np.random.uniform(1, 5, size=(2, 3)).astype(np.float32)]
    )


def test_Add_parity():
    """Test parity for Add."""
    check_parity(
        "Add",
        lambda: [
            np.random.randn(2, 3).astype(np.float32),
            np.random.randn(2, 3).astype(np.float32),
        ],
    )


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


def test_Arange_parity():
    """Test parity for Arange."""
    check_parity("Arange", lambda: [10])


def test_Argmax_parity():
    """Test parity for Argmax."""
    check_parity("Argmax", lambda: [np.random.randn(2, 3).astype(np.float32)])


def test_Argmin_parity():
    """Test parity for Argmin."""
    check_parity("Argmin", lambda: [np.random.randn(2, 3).astype(np.float32)])


def test_Asin_parity():
    """Test parity for Asin."""
    check_parity(
        "Asin", lambda: [np.random.uniform(-1, 1, size=(2, 3)).astype(np.float32)]
    )


def test_Asinh_parity():
    """Test parity for Asinh."""
    check_parity("Asinh", lambda: [np.random.randn(2, 3).astype(np.float32)])


def test_AssignVariable_parity():
    """Test parity for AssignVariable."""
    check_parity(
        "AssignVariable", lambda: []
    )  # Typically stateful, hard to verify blindly


def test_Atan_parity():
    """Test parity for Atan."""
    check_parity("Atan", lambda: [np.random.randn(2, 3).astype(np.float32)])


def test_Atan2_parity():
    """Test parity for Atan2."""
    check_parity(
        "Atan2",
        lambda: [
            np.random.randn(2, 3).astype(np.float32),
            np.random.randn(2, 3).astype(np.float32),
        ],
    )


def test_Atanh_parity():
    """Test parity for Atanh."""
    check_parity(
        "Atanh", lambda: [np.random.uniform(-0.9, 0.9, size=(2, 3)).astype(np.float32)]
    )


def test_Bitcast_parity():
    """Test parity for Bitcast."""
    check_parity(
        "Bitcast",
        lambda: [np.array([1.0, 2.0], dtype=np.float32)],
        kwargs_generator=lambda: {"type": np.int32},
    )


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


def test_BroadcastInDim_parity():
    """Test parity for BroadcastInDim."""
    check_parity(
        "Broadcast_In_Dim",
        lambda: [
            np.random.randn(3).astype(np.float32),
            np.array([2, 3], dtype=np.int32),
            np.array([1], dtype=np.int32),
        ],
    )


def test_BroadcastTo_parity():
    """Test parity for BroadcastTo."""
    check_parity(
        "Broadcast_To",
        lambda: [
            np.random.randn(2, 3).astype(np.float32),
            np.array([4, 2, 3], dtype=np.int32),
        ],
    )


def test_Cast_parity():
    """Test parity for Cast."""
    check_parity("Cast", lambda: [np.random.randn(2, 3).astype(np.float32), np.int32])


def test_Cbrt_parity():
    """Test parity for Cbrt."""
    check_parity("Cbrt", lambda: [np.random.randn(2, 3).astype(np.float32)])


def test_Ceil_parity():
    """Test parity for Ceil."""
    check_parity("Ceil", lambda: [np.random.randn(2, 3).astype(np.float32)])


def test_Conj_parity():
    """Test parity for Conj."""
    check_parity("Conj", lambda: [np.random.randn(2, 3).astype(np.float32)])


def test_ConvGeneralDilated_parity():
    """Test parity for ConvGeneralDilated."""
    check_parity(
        "Conv_General_Dilated", lambda: []
    )  # Skip complex tensor generation for now, just stub


def test_Copysign_parity():
    """Test parity for Copysign."""
    check_parity(
        "Copysign",
        lambda: [
            np.random.randn(2, 3).astype(np.float32),
            np.random.randn(2, 3).astype(np.float32),
        ],
    )


def test_Cos_parity():
    """Test parity for Cos."""
    check_parity("Cos", lambda: [np.random.randn(2, 3).astype(np.float32)])


def test_Cosh_parity():
    """Test parity for Cosh."""
    check_parity("Cosh", lambda: [np.random.randn(2, 3).astype(np.float32)])


def test_CountNonzero_parity():
    """Test parity for CountNonzero."""
    check_parity(
        "Count_Nonzero",
        lambda: [np.random.choice([0, 1, 2], size=(2, 3)).astype(np.float32)],
    )


def test_Cumsum_parity():
    """Test parity for Cumsum."""
    check_parity(
        "Cumsum",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
        kwargs_generator=lambda: {"axis": 0},
    )


def test_Deg2Rad_parity():
    """Test parity for Deg2Rad."""
    check_parity("Deg2rad", lambda: [np.random.randn(2, 3).astype(np.float32)])


def test_Digamma_parity():
    """Test parity for Digamma."""
    check_parity(
        "Digamma", lambda: [np.random.uniform(0.1, 5, size=(2, 3)).astype(np.float32)]
    )


def test_Divide_parity():
    """Test parity for Divide."""
    check_parity(
        "Divide",
        lambda: [
            np.random.randn(2, 3).astype(np.float32),
            np.random.randn(2, 3).astype(np.float32),
        ],
    )


def test_Divmod_parity():
    """Test parity for Divmod."""
    check_parity(
        "Divmod",
        lambda: [
            np.random.randint(1, 10, size=(2, 3)).astype(np.float32),
            np.random.randint(1, 10, size=(2, 3)).astype(np.float32),
        ],
    )


def test_Dot_parity():
    """Test parity for Dot."""
    check_parity(
        "Dot",
        lambda: [
            np.random.randn(3).astype(np.float32),
            np.random.randn(3).astype(np.float32),
        ],
    )


def test_DotGeneral_parity():
    """Test parity for DotGeneral."""
    check_parity("Dot_General", lambda: [])  # Skip complex for now


def test_DynamicSlice_parity():
    """Test parity for DynamicSlice."""
    check_parity("Dynamic_Slice", lambda: [])


def test_DynamicUpdateSlice_parity():
    """Test parity for DynamicUpdateSlice."""
    check_parity("Dynamic_Update_Slice", lambda: [])


def test_Einsum_parity():
    """Test parity for Einsum."""
    check_parity(
        "Einsum",
        lambda: [
            "ij,jk->ik",
            np.random.randn(2, 3).astype(np.float32),
            np.random.randn(3, 4).astype(np.float32),
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


def test_Erf_parity():
    """Test parity for Erf."""
    check_parity("Erf", lambda: [np.random.randn(2, 3).astype(np.float32)])


def test_Erfc_parity():
    """Test parity for Erfc."""
    check_parity("Erfc", lambda: [np.random.randn(2, 3).astype(np.float32)])


def test_Erfinv_parity():
    """Test parity for Erfinv."""
    check_parity(
        "Erfinv", lambda: [np.random.uniform(-0.9, 0.9, size=(2, 3)).astype(np.float32)]
    )


def test_Exp_parity():
    """Test parity for Exp."""
    check_parity("Exp", lambda: [np.random.randn(2, 3).astype(np.float32)])


def test_Exp2_parity():
    """Test parity for Exp2."""
    check_parity("Exp2", lambda: [np.random.randn(2, 3).astype(np.float32)])


def test_Expm1_parity():
    """Test parity for Expm1."""
    check_parity("Expm1", lambda: [np.random.randn(2, 3).astype(np.float32)])


def test_Fft_parity():
    """Test parity for Fft."""
    check_parity(
        "Fft",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
        kwargs_generator=lambda: {"n": 3},
    )


def test_Fix_parity():
    """Test parity for Fix."""
    check_parity("Fix", lambda: [np.random.randn(2, 3).astype(np.float32)])


def test_FloatPower_parity():
    """Test parity for FloatPower."""
    check_parity(
        "Float_Power",
        lambda: [
            np.random.uniform(0.1, 5, size=(2, 3)).astype(np.float32),
            np.random.uniform(0.1, 5, size=(2, 3)).astype(np.float32),
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


def test_Fmax_parity():
    """Test parity for Fmax."""
    check_parity(
        "Fmax",
        lambda: [
            np.random.randn(2, 3).astype(np.float32),
            np.random.randn(2, 3).astype(np.float32),
        ],
    )


def test_Fmin_parity():
    """Test parity for Fmin."""
    check_parity(
        "Fmin",
        lambda: [
            np.random.randn(2, 3).astype(np.float32),
            np.random.randn(2, 3).astype(np.float32),
        ],
    )


def test_Fmod_parity():
    """Test parity for Fmod."""
    check_parity(
        "Fmod",
        lambda: [
            np.random.randn(2, 3).astype(np.float32),
            np.random.uniform(1, 2, size=(2, 3)).astype(np.float32),
        ],
    )


def test_Frexp_parity():
    """Test parity for Frexp."""
    check_parity("Frexp", lambda: [np.random.randn(2, 3).astype(np.float32)])


def test_Full_parity():
    """Test parity for Full."""
    check_parity("Full", lambda: [np.array([2, 3], dtype=np.int32), 3.14])


def test_Gcd_parity():
    """Test parity for Gcd."""
    check_parity(
        "Gcd",
        lambda: [
            np.random.randint(1, 100, size=(2, 3)).astype(np.int32),
            np.random.randint(1, 100, size=(2, 3)).astype(np.int32),
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


def test_Hypot_parity():
    """Test parity for Hypot."""
    check_parity(
        "Hypot",
        lambda: [
            np.random.randn(2, 3).astype(np.float32),
            np.random.randn(2, 3).astype(np.float32),
        ],
    )


def test_Imag_parity():
    """Test parity for Imag."""
    check_parity("Imag", lambda: [np.random.randn(2, 3).astype(np.float32)])


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


def test_Lcm_parity():
    """Test parity for Lcm."""
    check_parity(
        "Lcm",
        lambda: [
            np.random.randint(1, 100, size=(2, 3)).astype(np.int32),
            np.random.randint(1, 100, size=(2, 3)).astype(np.int32),
        ],
    )


def test_Ldexp_parity():
    """Test parity for Ldexp."""
    check_parity(
        "Ldexp",
        lambda: [
            np.random.randn(2, 3).astype(np.float32),
            np.random.randint(1, 5, size=(2, 3)).astype(np.int32),
        ],
    )


def test_LeftShift_parity():
    """Test parity for LeftShift."""
    check_parity(
        "Left_Shift",
        lambda: [
            np.random.randint(1, 10, size=(2, 3)).astype(np.int32),
            np.random.randint(1, 3, size=(2, 3)).astype(np.int32),
        ],
    )


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


def test_Lgamma_parity():
    """Test parity for Lgamma."""
    check_parity(
        "Lgamma", lambda: [np.random.uniform(0.1, 5, size=(2, 3)).astype(np.float32)]
    )


def test_Log_parity():
    """Test parity for Log."""
    check_parity(
        "Log", lambda: [np.random.uniform(0.1, 5, size=(2, 3)).astype(np.float32)]
    )


def test_Log10_parity():
    """Test parity for Log10."""
    check_parity(
        "Log10", lambda: [np.random.uniform(0.1, 5, size=(2, 3)).astype(np.float32)]
    )


def test_Log1P_parity():
    """Test parity for Log1P."""
    check_parity(
        "Log1p", lambda: [np.random.uniform(0, 1, size=(2, 3)).astype(np.float32)]
    )


def test_Log2_parity():
    """Test parity for Log2."""
    check_parity(
        "Log2", lambda: [np.random.uniform(0.1, 5, size=(2, 3)).astype(np.float32)]
    )


def test_Logaddexp_parity():
    """Test parity for Logaddexp."""
    check_parity(
        "Logaddexp",
        lambda: [
            np.random.randn(2, 3).astype(np.float32),
            np.random.randn(2, 3).astype(np.float32),
        ],
    )


def test_Logaddexp2_parity():
    """Test parity for Logaddexp2."""
    check_parity(
        "Logaddexp2",
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


def test_Logit_parity():
    """Test parity for Logit."""
    check_parity(
        "Logit", lambda: [np.random.uniform(0.1, 0.9, size=(2, 3)).astype(np.float32)]
    )


def test_Logsumexp_parity():
    """Test parity for Logsumexp."""
    check_parity("Logsumexp", lambda: [np.random.randn(2, 3).astype(np.float32)])


def test_ManualSeed_parity():
    """Test parity for ManualSeed."""
    check_parity("ManualSeed", lambda: [])


def test_Matmul_parity():
    """Test parity for Matmul."""
    check_parity(
        "Matmul",
        lambda: [
            np.random.randn(2, 3).astype(np.float32),
            np.random.randn(3, 4).astype(np.float32),
        ],
    )


def test_Max_parity():
    """Test parity for Max."""
    check_parity("Max", lambda: [np.random.randn(2, 3).astype(np.float32)])


def test_Maximum_parity():
    """Test parity for Maximum."""
    check_parity(
        "Maximum",
        lambda: [
            np.random.randn(2, 3).astype(np.float32),
            np.random.randn(2, 3).astype(np.float32),
        ],
    )


def test_Mean_parity():
    """Test parity for Mean."""
    check_parity("Mean", lambda: [np.random.randn(2, 3).astype(np.float32)])


def test_Min_parity():
    """Test parity for Min."""
    check_parity("Min", lambda: [np.random.randn(2, 3).astype(np.float32)])


def test_Minimum_parity():
    """Test parity for Minimum."""
    check_parity(
        "Minimum",
        lambda: [
            np.random.randn(2, 3).astype(np.float32),
            np.random.randn(2, 3).astype(np.float32),
        ],
    )


def test_Mod_parity():
    """Test parity for Mod."""
    check_parity(
        "Mod",
        lambda: [
            np.random.randint(1, 10, size=(2, 3)).astype(np.float32),
            np.random.randint(1, 10, size=(2, 3)).astype(np.float32),
        ],
    )


def test_Multiply_parity():
    """Test parity for Multiply."""
    check_parity(
        "Multiply",
        lambda: [
            np.random.randn(2, 3).astype(np.float32),
            np.random.randn(2, 3).astype(np.float32),
        ],
    )


def test_Mvlgamma_parity():
    """Test parity for Mvlgamma."""
    check_parity(
        "Mvlgamma",
        lambda: [np.random.uniform(0.1, 5, size=(2, 3)).astype(np.float32), 1],
    )


def test_NanToNum_parity():
    """Test parity for NanToNum."""
    check_parity(
        "Nan_To_Num",
        lambda: [np.array([np.nan, np.inf, -np.inf, 1.0], dtype=np.float32)],
        kwargs_generator=lambda: {"nan": 0.0, "posinf": 1e30, "neginf": -1e30},
    )


def test_Negative_parity():
    """Test parity for Negative."""
    check_parity("Negative", lambda: [np.random.randn(2, 3).astype(np.float32)])


def test_Nextafter_parity():
    """Test parity for Nextafter."""
    check_parity(
        "Nextafter",
        lambda: [
            np.random.randn(2, 3).astype(np.float32),
            np.random.randn(2, 3).astype(np.float32),
        ],
    )


def test_Norm_parity():
    """Test parity for Norm."""
    check_parity("Norm", lambda: [np.random.randn(2, 3).astype(np.float32)])


def test_NotEqual_parity():
    """Test parity for NotEqual."""
    check_parity(
        "Not_Equal",
        lambda: [
            np.random.randn(2, 3).astype(np.float32),
            np.random.randn(2, 3).astype(np.float32),
        ],
    )


def test_Ones_parity():
    """Test parity for Ones."""
    check_parity("Ones", lambda: [(2, 3)])


def test_Pmean_parity():
    """Test parity for Pmean."""
    check_parity("Pmean", lambda: [np.random.randn(2, 3).astype(np.float32)])


def test_Positive_parity():
    """Test parity for Positive."""
    check_parity("Positive", lambda: [np.random.randn(2, 3).astype(np.float32)])


def test_Power_parity():
    """Test parity for Power."""
    check_parity(
        "Power",
        lambda: [
            np.random.uniform(0.1, 5, size=(2, 3)).astype(np.float32),
            np.random.uniform(0.1, 5, size=(2, 3)).astype(np.float32),
        ],
    )


def test_Prod_parity():
    """Test parity for Prod."""
    check_parity("Prod", lambda: [np.random.randn(2, 3).astype(np.float32)])


def test_Psum_parity():
    """Test parity for Psum."""
    check_parity("Psum", lambda: [np.random.randn(2, 3).astype(np.float32)])


def test_Rad2Deg_parity():
    """Test parity for Rad2Deg."""
    check_parity("Rad2deg", lambda: [np.random.randn(2, 3).astype(np.float32)])


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


def test_ReadVariable_parity():
    """Test parity for ReadVariable."""
    check_parity("ReadVariable", lambda: [])


def test_Real_parity():
    """Test parity for Real."""
    check_parity("Real", lambda: [np.random.randn(2, 3).astype(np.float32)])


def test_Reciprocal_parity():
    """Test parity for Reciprocal."""
    check_parity(
        "Reciprocal",
        lambda: [np.random.uniform(0.1, 5, size=(2, 3)).astype(np.float32)],
    )


def test_ReduceWindow_parity():
    """Test parity for ReduceWindow."""
    check_parity("Reduce_Window", lambda: [])


def test_Remainder_parity():
    """Test parity for Remainder."""
    check_parity(
        "Remainder",
        lambda: [
            np.random.randint(1, 10, size=(2, 3)).astype(np.float32),
            np.random.randint(1, 10, size=(2, 3)).astype(np.float32),
        ],
    )


def test_Reshape_parity():
    """Test parity for Reshape."""
    check_parity("Reshape", lambda: [np.random.randn(2, 3).astype(np.float32), (3, 2)])


def test_Resize_parity():
    """Test parity for Resize."""
    check_parity("Resize", lambda: [])


def test_Rfft_parity():
    """Test parity for Rfft."""
    check_parity("Rfft", lambda: [np.random.randn(2, 3).astype(np.float32)])


def test_RightShift_parity():
    """Test parity for RightShift."""
    check_parity(
        "Right_Shift",
        lambda: [
            np.random.randint(1, 10, size=(2, 3)).astype(np.int32),
            np.random.randint(1, 3, size=(2, 3)).astype(np.int32),
        ],
    )


def test_Round_parity():
    """Test parity for Round."""
    check_parity("Round", lambda: [np.random.randn(2, 3).astype(np.float32)])


def test_Rsqrt_parity():
    """Test parity for Rsqrt."""
    check_parity(
        "Rsqrt", lambda: [np.random.uniform(0.1, 5, size=(2, 3)).astype(np.float32)]
    )


def test_SegmentSum_parity():
    """Test parity for SegmentSum."""
    check_parity("SegmentSum", lambda: [])


def test_Sign_parity():
    """Test parity for Sign."""
    check_parity("Sign", lambda: [np.random.randn(2, 3).astype(np.float32)])


def test_Signbit_parity():
    """Test parity for Signbit."""
    check_parity("Signbit", lambda: [np.random.randn(2, 3).astype(np.float32)])


def test_Sin_parity():
    """Test parity for Sin."""
    check_parity("Sin", lambda: [np.random.randn(2, 3).astype(np.float32)])


def test_Sinc_parity():
    """Test parity for Sinc."""
    check_parity("Sinc", lambda: [np.random.randn(2, 3).astype(np.float32)])


def test_Sinh_parity():
    """Test parity for Sinh."""
    check_parity("Sinh", lambda: [np.random.randn(2, 3).astype(np.float32)])


def test_Sort_parity():
    """Test parity for Sort."""
    check_parity("Sort", lambda: [np.random.randn(2, 3).astype(np.float32)])


def test_Sqrt_parity():
    """Test parity for Sqrt."""
    check_parity(
        "Sqrt", lambda: [np.random.uniform(0.1, 5, size=(2, 3)).astype(np.float32)]
    )


def test_Square_parity():
    """Test parity for Square."""
    check_parity("Square", lambda: [np.random.randn(2, 3).astype(np.float32)])


def test_Std_parity():
    """Test parity for Std."""
    check_parity("Std", lambda: [np.random.randn(2, 3).astype(np.float32)])


def test_Subtract_parity():
    """Test parity for Subtract."""
    check_parity(
        "Subtract",
        lambda: [
            np.random.randn(2, 3).astype(np.float32),
            np.random.randn(2, 3).astype(np.float32),
        ],
    )


def test_Sum_parity():
    """Test parity for Sum."""
    check_parity("Sum", lambda: [np.random.randn(2, 3).astype(np.float32)])


def test_Tan_parity():
    """Test parity for Tan."""
    check_parity("Tan", lambda: [np.random.randn(2, 3).astype(np.float32)])


def test_Tanh_parity():
    """Test parity for Tanh."""
    check_parity("Tanh", lambda: [np.random.randn(2, 3).astype(np.float32)])


def test_TopK_parity():
    """Test parity for TopK."""
    check_parity(
        "Topk",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
        kwargs_generator=lambda: {"k": 2},
    )


def test_Transpose_parity():
    """Test parity for Transpose."""
    check_parity("Transpose", lambda: [np.random.randn(2, 3).astype(np.float32)])


def test_TrueDivide_parity():
    """Test parity for TrueDivide."""
    check_parity(
        "True_Divide",
        lambda: [
            np.random.randn(2, 3).astype(np.float32),
            np.random.randn(2, 3).astype(np.float32),
        ],
    )


def test_Trunc_parity():
    """Test parity for Trunc."""
    check_parity("Trunc", lambda: [np.random.randn(2, 3).astype(np.float32)])


def test_Variance_parity():
    """Test parity for Variance."""
    check_parity("Variance", lambda: [np.random.randn(2, 3).astype(np.float32)])


def test_Xlogy_parity():
    """Test parity for Xlogy."""
    check_parity(
        "Xlogy",
        lambda: [
            np.random.randn(2, 3).astype(np.float32),
            np.random.uniform(0.1, 5, size=(2, 3)).astype(np.float32),
        ],
    )


def test_Zeros_parity():
    """Test parity for Zeros."""
    check_parity("Zeros", lambda: [(2, 3)])
