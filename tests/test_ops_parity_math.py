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


def test_Asin_parity():
    """Test parity for Asin."""
    check_parity(
        "Asin", lambda: [np.random.uniform(-1, 1, size=(2, 3)).astype(np.float32)]
    )


def test_Asinh_parity():
    """Test parity for Asinh."""
    check_parity("Asinh", lambda: [np.random.randn(2, 3).astype(np.float32)])


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


def test_Cbrt_parity():
    """Test parity for Cbrt."""
    check_parity("Cbrt", lambda: [np.random.randn(2, 3).astype(np.float32)])


def test_Ceil_parity():
    """Test parity for Ceil."""
    check_parity("Ceil", lambda: [np.random.randn(2, 3).astype(np.float32)])


def test_Conj_parity():
    """Test parity for Conj."""
    check_parity("Conj", lambda: [np.random.randn(2, 3).astype(np.float32)])


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


def test_Erf_parity():
    import pytest

    pytest.skip("Aborts full test suite")
    """Test parity for Erf."""
    check_parity("Erf", lambda: [np.random.randn(2, 3).astype(np.float32)])


def test_Erfc_parity():
    """Test parity for Erfc."""
    check_parity("Erfc", lambda: [np.random.randn(2, 3).astype(np.float32)])


def test_Exp_parity():
    """Test parity for Exp."""
    check_parity("Exp", lambda: [np.random.randn(2, 3).astype(np.float32)])


def test_Exp2_parity():
    """Test parity for Exp2."""
    check_parity("Exp2", lambda: [np.random.randn(2, 3).astype(np.float32)])


@pytest.mark.skip("Crashes MLX natively")
def test_Expm1_parity():
    """Test parity for Expm1."""
    check_parity("Expm1", lambda: [np.random.randn(2, 3).astype(np.float32)])


def test_FloatPower_parity():
    """Test parity for FloatPower."""
    check_parity(
        "Float_Power",
        lambda: [
            np.random.uniform(0.1, 5, size=(2, 3)).astype(np.float32),
            np.random.uniform(0.1, 5, size=(2, 3)).astype(np.float32),
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


def test_Imag_parity():
    """Test parity for Imag."""
    check_parity("Imag", lambda: [np.random.randn(2, 3).astype(np.float32)])


def test_Ldexp_parity():
    """Test parity for Ldexp."""
    check_parity(
        "Ldexp",
        lambda: [
            np.random.randn(2, 3).astype(np.float32),
            np.random.randint(1, 5, size=(2, 3)).astype(np.int32),
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


def test_Logit_parity():
    """Test parity for Logit."""
    check_parity(
        "Logit", lambda: [np.random.uniform(0.1, 0.9, size=(2, 3)).astype(np.float32)]
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


def test_Power_parity():
    """Test parity for Power."""
    check_parity(
        "Power",
        lambda: [
            np.random.uniform(0.1, 5, size=(2, 3)).astype(np.float32),
            np.random.uniform(0.1, 5, size=(2, 3)).astype(np.float32),
        ],
    )


def test_Real_parity():
    """Test parity for Real."""
    check_parity("Real", lambda: [np.random.randn(2, 3).astype(np.float32)])


def test_Round_parity():
    """Test parity for Round."""
    check_parity("Round", lambda: [np.random.randn(2, 3).astype(np.float32)])


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


def test_Subtract_parity():
    """Test parity for Subtract."""
    check_parity(
        "Subtract",
        lambda: [
            np.random.randn(2, 3).astype(np.float32),
            np.random.randn(2, 3).astype(np.float32),
        ],
    )


def test_Tan_parity():
    """Test parity for Tan."""
    check_parity("Tan", lambda: [np.random.randn(2, 3).astype(np.float32)])


def test_Tanh_parity():
    """Test parity for Tanh."""
    check_parity("Tanh", lambda: [np.random.randn(2, 3).astype(np.float32)])


def test_Xlogy_parity():
    """Test parity for Xlogy."""
    check_parity(
        "Xlogy",
        lambda: [
            np.random.randn(2, 3).astype(np.float32),
            np.random.uniform(0.1, 5, size=(2, 3)).astype(np.float32),
        ],
    )


def test_AccumulateN_parity():
    """Test parity for AccumulateN."""
    check_parity(
        "AccumulateN",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_AddN_parity():
    """Test parity for AddN."""
    check_parity(
        "AddN",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_Assign_parity():
    """Test parity for Assign."""
    check_parity(
        "Assign",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_AssignAdd_parity():
    """Test parity for AssignAdd."""
    check_parity(
        "AssignAdd",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_AssignSub_parity():
    """Test parity for AssignSub."""
    check_parity(
        "AssignSub",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_BesselI0_parity():
    """Test parity for BesselI0."""
    check_parity(
        "BesselI0",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_BesselI0e_parity():
    """Test parity for BesselI0e."""
    check_parity(
        "BesselI0e",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_BesselI1_parity():
    """Test parity for BesselI1."""
    check_parity(
        "BesselI1",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_BesselI1e_parity():
    """Test parity for BesselI1e."""
    check_parity(
        "BesselI1e",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_BesselJ0_parity():
    """Test parity for BesselJ0."""
    check_parity(
        "BesselJ0",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_BesselJ1_parity():
    """Test parity for BesselJ1."""
    check_parity(
        "BesselJ1",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_BesselJn_parity():
    """Test parity for BesselJn."""
    check_parity(
        "BesselJn",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_BesselK0_parity():
    """Test parity for BesselK0."""
    check_parity(
        "BesselK0",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_BesselK0e_parity():
    """Test parity for BesselK0e."""
    check_parity(
        "BesselK0e",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_BesselK1_parity():
    """Test parity for BesselK1."""
    check_parity(
        "BesselK1",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_BesselK1e_parity():
    """Test parity for BesselK1e."""
    check_parity(
        "BesselK1e",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_BesselY0_parity():
    """Test parity for BesselY0."""
    check_parity(
        "BesselY0",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_BesselY1_parity():
    """Test parity for BesselY1."""
    check_parity(
        "BesselY1",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_Beta_parity():
    """Test parity for Beta."""
    check_parity(
        "Beta",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_Betainc_parity():
    """Test parity for Betainc."""
    check_parity(
        "Betainc",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_ConstantOfShape_parity():
    """Test parity for ConstantOfShape."""
    check_parity(
        "ConstantOfShape",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_CropImages_parity():
    """Test parity for CropImages."""
    check_parity(
        "CropImages",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_Expint_parity():
    """Test parity for Expint."""
    check_parity(
        "Expint",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_FresnelCos_parity():
    """Test parity for FresnelCos."""
    check_parity(
        "FresnelCos",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_FresnelSin_parity():
    """Test parity for FresnelSin."""
    check_parity(
        "FresnelSin",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_Gamma_parity():
    """Test parity for Gamma."""
    check_parity(
        "Gamma",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_Igamma_parity():
    """Test parity for Igamma."""
    check_parity(
        "Igamma",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_IgammaGradA_parity():
    """Test parity for IgammaGradA."""
    check_parity(
        "IgammaGradA",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_Igammac_parity():
    """Test parity for Igammac."""
    check_parity(
        "Igammac",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_Lbeta_parity():
    """Test parity for Lbeta."""
    check_parity(
        "Lbeta",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_Loggamma_parity():
    """Test parity for Loggamma."""
    check_parity(
        "Loggamma",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_MfccsFromLogMelSpectrograms_parity():
    """Test parity for MfccsFromLogMelSpectrograms."""
    check_parity(
        "MfccsFromLogMelSpectrograms",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_PadImages_parity():
    """Test parity for PadImages."""
    check_parity(
        "PadImages",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_Polyadd_parity():
    """Test parity for Polyadd."""
    check_parity(
        "Polyadd",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_Polydiv_parity():
    """Test parity for Polydiv."""
    check_parity(
        "Polydiv",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_Polygamma_parity():
    """Test parity for Polygamma."""
    check_parity(
        "Polygamma",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_Polymul_parity():
    """Test parity for Polymul."""
    check_parity(
        "Polymul",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_Polysub_parity():
    """Test parity for Polysub."""
    check_parity(
        "Polysub",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_PowerIteration_parity():
    """Test parity for PowerIteration."""
    check_parity(
        "PowerIteration",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_RaggedAdd_parity():
    """Test parity for RaggedAdd."""
    check_parity(
        "RaggedAdd",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_RaggedConstant_parity():
    """Test parity for RaggedConstant."""
    check_parity(
        "RaggedConstant",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_ScatterAdd_parity():
    """Test parity for ScatterAdd."""
    check_parity(
        "ScatterAdd",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_ScatterMul_parity():
    """Test parity for ScatterMul."""
    check_parity(
        "ScatterMul",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_SparseAdd_parity():
    """Test parity for SparseAdd."""
    check_parity(
        "SparseAdd",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_SpecialGamma_parity():
    """Test parity for SpecialGamma."""
    check_parity(
        "SpecialGamma",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_TruncateDiv_parity():
    """Test parity for TruncateDiv."""
    check_parity(
        "TruncateDiv",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_TruncateMod_parity():
    """Test parity for TruncateMod."""
    check_parity(
        "TruncateMod",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_Xdivy_parity():
    """Test parity for Xdivy."""
    check_parity(
        "Xdivy",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_Xlog1py_parity():
    """Test parity for Xlog1py."""
    check_parity(
        "Xlog1py",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )
