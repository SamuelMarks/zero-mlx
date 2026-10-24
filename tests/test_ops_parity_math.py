"""Compiler operation parity tests for `math` operations."""

from __future__ import annotations

import numpy as np
import pytest

from tests.conftest import check_parity


def test_Abs_parity() -> None:
    """Test parity for Abs."""
    check_parity("Abs", lambda: [np.random.randn(2, 3).astype(np.float32)])


def test_Acos_parity() -> None:
    """Test parity for Acos."""
    check_parity("Acos", lambda: [np.random.uniform(-1, 1, size=(2, 3)).astype(np.float32)])


def test_Acosh_parity() -> None:
    """Test parity for Acosh."""
    check_parity("Acosh", lambda: [np.random.uniform(1, 5, size=(2, 3)).astype(np.float32)])


def test_Add_parity() -> None:
    """Test parity for Add."""
    check_parity(
        "Add",
        lambda: [
            np.random.randn(2, 3).astype(np.float32),
            np.random.randn(2, 3).astype(np.float32),
        ],
    )


def test_Asin_parity() -> None:
    """Test parity for Asin."""
    check_parity("Asin", lambda: [np.random.uniform(-1, 1, size=(2, 3)).astype(np.float32)])


def test_Asinh_parity() -> None:
    """Test parity for Asinh."""
    check_parity("Asinh", lambda: [np.random.randn(2, 3).astype(np.float32)])


def test_Atan_parity() -> None:
    """Test parity for Atan."""
    check_parity("Atan", lambda: [np.random.randn(2, 3).astype(np.float32)])


def test_Atan2_parity() -> None:
    """Test parity for Atan2."""
    check_parity(
        "Atan2",
        lambda: [
            np.random.randn(2, 3).astype(np.float32),
            np.random.randn(2, 3).astype(np.float32),
        ],
    )


def test_Atanh_parity() -> None:
    """Test parity for Atanh."""
    check_parity("Atanh", lambda: [np.random.uniform(-0.9, 0.9, size=(2, 3)).astype(np.float32)])


def test_Cbrt_parity() -> None:
    """Test parity for Cbrt."""
    check_parity("Cbrt", lambda: [np.random.randn(2, 3).astype(np.float32)])


def test_Ceil_parity() -> None:
    """Test parity for Ceil."""
    check_parity("Ceil", lambda: [np.random.randn(2, 3).astype(np.float32)])


def test_Conj_parity() -> None:
    """Test parity for Conj."""
    check_parity("Conj", lambda: [np.random.randn(2, 3).astype(np.float32)])


def test_Copysign_parity() -> None:
    """Test parity for Copysign."""
    check_parity(
        "Copysign",
        lambda: [
            np.random.randn(2, 3).astype(np.float32),
            np.random.randn(2, 3).astype(np.float32),
        ],
    )


def test_Cos_parity() -> None:
    """Test parity for Cos."""
    check_parity("Cos", lambda: [np.random.randn(2, 3).astype(np.float32)])


def test_Cosh_parity() -> None:
    """Test parity for Cosh."""
    check_parity("Cosh", lambda: [np.random.randn(2, 3).astype(np.float32)])


def test_Digamma_parity() -> None:
    """Test parity for Digamma."""
    check_parity("Digamma", lambda: [np.random.uniform(0.1, 5, size=(2, 3)).astype(np.float32)])


def test_Divide_parity() -> None:
    """Test parity for Divide."""
    check_parity(
        "Divide",
        lambda: [
            np.random.randn(2, 3).astype(np.float32),
            np.random.randn(2, 3).astype(np.float32),
        ],
    )


def test_Divmod_parity() -> None:
    """Test parity for Divmod."""
    check_parity(
        "Divmod",
        lambda: [
            np.random.randint(1, 10, size=(2, 3)).astype(np.float32),
            np.random.randint(1, 10, size=(2, 3)).astype(np.float32),
        ],
    )


def test_Erf_parity() -> None:
    """Test parity for Erf."""
    check_parity("Erf", lambda: [np.random.randn(2, 3).astype(np.float32)])


def test_Erfc_parity() -> None:
    """Test parity for Erfc."""
    check_parity("Erfc", lambda: [np.random.randn(2, 3).astype(np.float32)])


def test_Exp_parity() -> None:
    """Test parity for Exp."""
    check_parity("Exp", lambda: [np.random.randn(2, 3).astype(np.float32)])


def test_Exp2_parity() -> None:
    """Test parity for Exp2."""
    check_parity("Exp2", lambda: [np.random.randn(2, 3).astype(np.float32)])


def test_Expm1_parity() -> None:
    """Test parity for Expm1."""
    check_parity("Expm1", lambda: [np.random.randn(2, 3).astype(np.float32)])


def test_FloatPower_parity() -> None:
    """Test parity for FloatPower."""
    check_parity(
        "Float_Power",
        lambda: [
            np.random.uniform(0.1, 5, size=(2, 3)).astype(np.float32),
            np.random.uniform(0.1, 5, size=(2, 3)).astype(np.float32),
        ],
    )


def test_Fmod_parity() -> None:
    """Test parity for Fmod."""
    check_parity(
        "Fmod",
        lambda: [
            np.random.randn(2, 3).astype(np.float32),
            np.random.uniform(1, 2, size=(2, 3)).astype(np.float32),
        ],
    )


def test_Frexp_parity() -> None:
    """Test parity for Frexp."""
    check_parity("Frexp", lambda: [np.random.randn(2, 3).astype(np.float32)])


def test_Imag_parity() -> None:
    """Test parity for Imag."""
    check_parity("Imag", lambda: [np.random.randn(2, 3).astype(np.float32)])


def test_Ldexp_parity() -> None:
    """Test parity for Ldexp."""
    check_parity(
        "Ldexp",
        lambda: [
            np.random.randn(2, 3).astype(np.float32),
            np.random.randint(1, 5, size=(2, 3)).astype(np.int32),
        ],
    )


def test_Lgamma_parity() -> None:
    """Test parity for Lgamma."""
    check_parity("Lgamma", lambda: [np.random.uniform(0.1, 5, size=(2, 3)).astype(np.float32)])


def test_Log_parity() -> None:
    """Test parity for Log."""
    check_parity("Log", lambda: [np.random.uniform(0.1, 5, size=(2, 3)).astype(np.float32)])


def test_Log10_parity() -> None:
    """Test parity for Log10."""
    check_parity("Log10", lambda: [np.random.uniform(0.1, 5, size=(2, 3)).astype(np.float32)])


def test_Log1P_parity() -> None:
    """Test parity for Log1P."""
    check_parity("Log1p", lambda: [np.random.uniform(0, 1, size=(2, 3)).astype(np.float32)])


def test_Log2_parity() -> None:
    """Test parity for Log2."""
    check_parity("Log2", lambda: [np.random.uniform(0.1, 5, size=(2, 3)).astype(np.float32)])


def test_Logaddexp_parity() -> None:
    """Test parity for Logaddexp."""
    check_parity(
        "Logaddexp",
        lambda: [
            np.random.randn(2, 3).astype(np.float32),
            np.random.randn(2, 3).astype(np.float32),
        ],
    )


def test_Logaddexp2_parity() -> None:
    """Test parity for Logaddexp2."""
    check_parity(
        "Logaddexp2",
        lambda: [
            np.random.randn(2, 3).astype(np.float32),
            np.random.randn(2, 3).astype(np.float32),
        ],
    )


def test_Logit_parity() -> None:
    """Test parity for Logit."""
    check_parity("Logit", lambda: [np.random.uniform(0.1, 0.9, size=(2, 3)).astype(np.float32)])


def test_Mod_parity() -> None:
    """Test parity for Mod."""
    check_parity(
        "Mod",
        lambda: [
            np.random.randint(1, 10, size=(2, 3)).astype(np.float32),
            np.random.randint(1, 10, size=(2, 3)).astype(np.float32),
        ],
    )


def test_Multiply_parity() -> None:
    """Test parity for Multiply."""
    check_parity(
        "Multiply",
        lambda: [
            np.random.randn(2, 3).astype(np.float32),
            np.random.randn(2, 3).astype(np.float32),
        ],
    )


def test_Mvlgamma_parity() -> None:
    """Test parity for Mvlgamma."""
    check_parity(
        "Mvlgamma",
        lambda: [np.random.uniform(0.1, 5, size=(2, 3)).astype(np.float32), 1],
    )


def test_Power_parity() -> None:
    """Test parity for Power."""
    check_parity(
        "Power",
        lambda: [
            np.random.uniform(0.1, 5, size=(2, 3)).astype(np.float32),
            np.random.uniform(0.1, 5, size=(2, 3)).astype(np.float32),
        ],
    )


def test_Real_parity() -> None:
    """Test parity for Real."""
    check_parity("Real", lambda: [np.random.randn(2, 3).astype(np.float32)])


def test_Round_parity() -> None:
    """Test parity for Round."""
    check_parity("Round", lambda: [np.random.randn(2, 3).astype(np.float32)])


def test_Sign_parity() -> None:
    """Test parity for Sign."""
    check_parity("Sign", lambda: [np.random.randn(2, 3).astype(np.float32)])


def test_Signbit_parity() -> None:
    """Test parity for Signbit."""
    check_parity("Signbit", lambda: [np.random.randn(2, 3).astype(np.float32)])


def test_Sin_parity() -> None:
    """Test parity for Sin."""
    check_parity("Sin", lambda: [np.random.randn(2, 3).astype(np.float32)])


def test_Sinc_parity() -> None:
    """Test parity for Sinc."""
    check_parity("Sinc", lambda: [np.random.randn(2, 3).astype(np.float32)])


def test_Sinh_parity() -> None:
    """Test parity for Sinh."""
    check_parity("Sinh", lambda: [np.random.randn(2, 3).astype(np.float32)])


def test_Subtract_parity() -> None:
    """Test parity for Subtract."""
    check_parity(
        "Subtract",
        lambda: [
            np.random.randn(2, 3).astype(np.float32),
            np.random.randn(2, 3).astype(np.float32),
        ],
    )


def test_Tan_parity() -> None:
    """Test parity for Tan."""
    check_parity("Tan", lambda: [np.random.randn(2, 3).astype(np.float32)])


def test_Tanh_parity() -> None:
    """Test parity for Tanh."""
    check_parity("Tanh", lambda: [np.random.randn(2, 3).astype(np.float32)])


def test_Xlogy_parity() -> None:
    """Test parity for Xlogy."""
    check_parity(
        "Xlogy",
        lambda: [
            np.random.randn(2, 3).astype(np.float32),
            np.random.uniform(0.1, 5, size=(2, 3)).astype(np.float32),
        ],
    )


def test_AccumulateN_parity() -> None:
    """Test parity for AccumulateN."""
    check_parity(
        "AccumulateN",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_AddN_parity() -> None:
    """Test parity for AddN."""
    check_parity(
        "AddN",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_Assign_parity() -> None:
    """Test parity for Assign."""
    check_parity(
        "Assign",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_AssignAdd_parity() -> None:
    """Test parity for AssignAdd."""
    check_parity(
        "AssignAdd",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_AssignSub_parity() -> None:
    """Test parity for AssignSub."""
    check_parity(
        "AssignSub",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_BesselI0_parity() -> None:
    """Test parity for BesselI0."""
    check_parity(
        "BesselI0",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_BesselI0e_parity() -> None:
    """Test parity for BesselI0e."""
    check_parity(
        "BesselI0e",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_BesselI1_parity() -> None:
    """Test parity for BesselI1."""
    check_parity(
        "BesselI1",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_BesselI1e_parity() -> None:
    """Test parity for BesselI1e."""
    check_parity(
        "BesselI1e",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_BesselJ0_parity() -> None:
    """Test parity for BesselJ0."""
    check_parity(
        "BesselJ0",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_BesselJ1_parity() -> None:
    """Test parity for BesselJ1."""
    check_parity(
        "BesselJ1",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_BesselJn_parity() -> None:
    """Test parity for BesselJn."""
    check_parity(
        "BesselJn",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_BesselK0_parity() -> None:
    """Test parity for BesselK0."""
    check_parity(
        "BesselK0",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_BesselK0e_parity() -> None:
    """Test parity for BesselK0e."""
    check_parity(
        "BesselK0e",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_BesselK1_parity() -> None:
    """Test parity for BesselK1."""
    check_parity(
        "BesselK1",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_BesselK1e_parity() -> None:
    """Test parity for BesselK1e."""
    check_parity(
        "BesselK1e",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_BesselY0_parity() -> None:
    """Test parity for BesselY0."""
    check_parity(
        "BesselY0",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_BesselY1_parity() -> None:
    """Test parity for BesselY1."""
    check_parity(
        "BesselY1",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_Beta_parity() -> None:
    """Test parity for Beta."""
    check_parity(
        "Beta",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_Betainc_parity() -> None:
    """Test parity for Betainc."""
    check_parity(
        "Betainc",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_ConstantOfShape_parity() -> None:
    """Test parity for ConstantOfShape."""
    check_parity(
        "ConstantOfShape",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_CropImages_parity() -> None:
    """Test parity for CropImages."""
    check_parity(
        "CropImages",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_Expint_parity() -> None:
    """Test parity for Expint."""
    check_parity(
        "Expint",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_FresnelCos_parity() -> None:
    """Test parity for FresnelCos."""
    check_parity(
        "FresnelCos",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_FresnelSin_parity() -> None:
    """Test parity for FresnelSin."""
    check_parity(
        "FresnelSin",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_Gamma_parity() -> None:
    """Test parity for Gamma."""
    check_parity(
        "Gamma",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_Igamma_parity() -> None:
    """Test parity for Igamma."""
    check_parity(
        "Igamma",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_IgammaGradA_parity() -> None:
    """Test parity for IgammaGradA."""
    check_parity(
        "IgammaGradA",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_Igammac_parity() -> None:
    """Test parity for Igammac."""
    check_parity(
        "Igammac",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_Lbeta_parity() -> None:
    """Test parity for Lbeta."""
    check_parity(
        "Lbeta",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_Loggamma_parity() -> None:
    """Test parity for Loggamma."""
    check_parity(
        "Loggamma",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_MfccsFromLogMelSpectrograms_parity() -> None:
    """Test parity for MfccsFromLogMelSpectrograms."""
    check_parity(
        "MfccsFromLogMelSpectrograms",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_PadImages_parity() -> None:
    """Test parity for PadImages."""
    check_parity(
        "PadImages",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_Polyadd_parity() -> None:
    """Test parity for Polyadd."""
    check_parity(
        "Polyadd",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_Polydiv_parity() -> None:
    """Test parity for Polydiv."""
    check_parity(
        "Polydiv",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_Polygamma_parity() -> None:
    """Test parity for Polygamma."""
    check_parity(
        "Polygamma",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_Polymul_parity() -> None:
    """Test parity for Polymul."""
    check_parity(
        "Polymul",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_Polysub_parity() -> None:
    """Test parity for Polysub."""
    check_parity(
        "Polysub",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_PowerIteration_parity() -> None:
    """Test parity for PowerIteration."""
    check_parity(
        "PowerIteration",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_RaggedAdd_parity() -> None:
    """Test parity for RaggedAdd."""
    check_parity(
        "RaggedAdd",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_RaggedConstant_parity() -> None:
    """Test parity for RaggedConstant."""
    check_parity(
        "RaggedConstant",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_ScatterAdd_parity() -> None:
    """Test parity for ScatterAdd."""
    check_parity(
        "ScatterAdd",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_ScatterMul_parity() -> None:
    """Test parity for ScatterMul."""
    check_parity(
        "ScatterMul",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_SparseAdd_parity() -> None:
    """Test parity for SparseAdd."""
    check_parity(
        "SparseAdd",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_SpecialGamma_parity() -> None:
    """Test parity for SpecialGamma."""
    check_parity(
        "SpecialGamma",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_TruncateDiv_parity() -> None:
    """Test parity for TruncateDiv."""
    check_parity(
        "TruncateDiv",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_TruncateMod_parity() -> None:
    """Test parity for TruncateMod."""
    check_parity(
        "TruncateMod",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_Xdivy_parity() -> None:
    """Test parity for Xdivy."""
    check_parity(
        "Xdivy",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_Xlog1py_parity() -> None:
    """Test parity for Xlog1py."""
    check_parity(
        "Xlog1py",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )
