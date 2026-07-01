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


@pytest.mark.skip("Crashes MLX natively")
def test_Erfinv_parity():
    """Test parity for Erfinv."""
    check_parity(
        "Erfinv", lambda: [np.random.uniform(-0.9, 0.9, size=(2, 3)).astype(np.float32)]
    )


def test_Matmul_parity():
    """Test parity for Matmul."""
    check_parity(
        "Matmul",
        lambda: [
            np.random.randn(2, 3).astype(np.float32),
            np.random.randn(3, 4).astype(np.float32),
        ],
    )


def test_Rsqrt_parity():
    """Test parity for Rsqrt."""
    check_parity(
        "Rsqrt", lambda: [np.random.uniform(0.1, 5, size=(2, 3)).astype(np.float32)]
    )


def test_Sqrt_parity():
    """Test parity for Sqrt."""
    check_parity(
        "Sqrt", lambda: [np.random.uniform(0.1, 5, size=(2, 3)).astype(np.float32)]
    )


def test_AsString_parity():
    """Test parity for AsString."""
    check_parity(
        "AsString",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_BandedTriangularSolve_parity():
    """Test parity for BandedTriangularSolve."""
    check_parity(
        "BandedTriangularSolve",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_Cholesky_parity():
    """Test parity for Cholesky."""
    check_parity(
        "Cholesky",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_CholeskySolve_parity():
    """Test parity for CholeskySolve."""
    check_parity(
        "CholeskySolve",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_ConfusionMatrix_parity():
    """Test parity for ConfusionMatrix."""
    check_parity(
        "ConfusionMatrix",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_Cross_parity():
    """Test parity for Cross."""
    check_parity(
        "Cross",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_CustomLinearSolve_parity():
    """Test parity for CustomLinearSolve."""
    check_parity(
        "CustomLinearSolve",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_Det_parity():
    """Test parity for Det."""
    check_parity(
        "Det",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_Diag_parity():
    """Test parity for Diag."""
    check_parity(
        "Diag",
        lambda: [np.random.randn(3).astype(np.float32)],
    )


def test_DiagIndices_parity():
    """Test parity for DiagIndices."""
    check_parity(
        "DiagIndices",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_DiagIndicesFrom_parity():
    """Test parity for DiagIndicesFrom."""
    check_parity(
        "DiagIndicesFrom",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_Diagflat_parity():
    """Test parity for Diagflat."""
    check_parity(
        "Diagflat",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_Diagonal_parity():
    """Test parity for Diagonal."""
    check_parity(
        "Diagonal",
        lambda: [np.random.randn(3, 3).astype(np.float32)],
    )


def test_Eigh_parity():
    """Test parity for Eigh."""
    check_parity(
        "Eigh",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_EighTridiagonal_parity():
    """Test parity for EighTridiagonal."""
    check_parity(
        "EighTridiagonal",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_Eigvalsh_parity():
    """Test parity for Eigvalsh."""
    check_parity(
        "Eigvalsh",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


@pytest.mark.skip("Crashes MLX natively")
def test_ErfInv_parity():
    """Test parity for ErfInv."""
    check_parity(
        "ErfInv",
        lambda: [np.random.uniform(-0.9, 0.9, (2, 2)).astype(np.float32)],
    )


@pytest.mark.skip("Crashes MLX natively")
def test_Erfcinv_parity():
    """Test parity for Erfcinv."""
    check_parity(
        "Erfcinv",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_FlipLeftRight_parity():
    """Test parity for FlipLeftRight."""
    check_parity(
        "FlipLeftRight",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_Inv_parity():
    """Test parity for Inv."""
    check_parity(
        "Inv",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_Invert_parity():
    """Test parity for Invert."""
    check_parity(
        "Invert",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_InvertPermutation_parity():
    """Test parity for InvertPermutation."""
    check_parity(
        "InvertPermutation",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_IsStrictlyIncreasing_parity():
    """Test parity for IsStrictlyIncreasing."""
    check_parity(
        "IsStrictlyIncreasing",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_LuSolve_parity():
    """Test parity for LuSolve."""
    check_parity(
        "LuSolve",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_MatrixExponential_parity():
    """Test parity for MatrixExponential."""
    check_parity(
        "MatrixExponential",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_MatrixNorm_parity():
    """Test parity for MatrixNorm."""
    check_parity(
        "MatrixNorm",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_MatrixPower_parity():
    """Test parity for MatrixPower."""
    check_parity(
        "MatrixPower",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_MatrixRank_parity():
    """Test parity for MatrixRank."""
    check_parity(
        "MatrixRank",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_MatrixTranspose_parity():
    """Test parity for MatrixTranspose."""
    check_parity(
        "MatrixTranspose",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_MultiDot_parity():
    """Test parity for MultiDot."""
    check_parity(
        "MultiDot",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_Ndtri_parity():
    """Test parity for Ndtri."""
    check_parity(
        "Ndtri",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_Pdot_parity():
    """Test parity for Pdot."""
    check_parity(
        "Pdot",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_Pinv_parity():
    """Test parity for Pinv."""
    check_parity(
        "Pinv",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_Qr_parity():
    """Test parity for Qr."""
    check_parity(
        "Qr",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_RaggedCrossHashed_parity():
    """Test parity for RaggedCrossHashed."""
    check_parity(
        "RaggedCrossHashed",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_RaggedDot_parity():
    """Test parity for RaggedDot."""
    check_parity(
        "RaggedDot",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_RaggedMatMul_parity():
    """Test parity for RaggedMatMul."""
    check_parity(
        "RaggedMatMul",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_RawMatMul_parity():
    """Test parity for RawMatMul."""
    check_parity(
        "RawMatMul",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_Rayleigh_parity():
    """Test parity for Rayleigh."""
    check_parity(
        "Rayleigh",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_Slogdet_parity():
    """Test parity for Slogdet."""
    check_parity(
        "Slogdet",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_Solve_parity():
    """Test parity for Solve."""
    check_parity(
        "Solve",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_SparseCrossHashed_parity():
    """Test parity for SparseCrossHashed."""
    check_parity(
        "SparseCrossHashed",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_SparseDenseMatMul_parity():
    """Test parity for SparseDenseMatMul."""
    check_parity(
        "SparseDenseMatMul",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_SparseSegmentSqrtN_parity():
    """Test parity for SparseSegmentSqrtN."""
    check_parity(
        "SparseSegmentSqrtN",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_Sqrtm_parity():
    """Test parity for Sqrtm."""
    check_parity(
        "Sqrtm",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_StridedSlice_parity():
    """Test parity for StridedSlice."""
    check_parity(
        "StridedSlice",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_StringJoin_parity():
    """Test parity for StringJoin."""
    check_parity(
        "StringJoin",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_StringLength_parity():
    """Test parity for StringLength."""
    check_parity(
        "StringLength",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_StringLookup_parity():
    """Test parity for StringLookup."""
    check_parity(
        "StringLookup",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_StringLower_parity():
    """Test parity for StringLower."""
    check_parity(
        "StringLower",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_StringSplit_parity():
    """Test parity for StringSplit."""
    check_parity(
        "StringSplit",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_StringSubstr_parity():
    """Test parity for StringSubstr."""
    check_parity(
        "StringSubstr",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_StringToHash_parity():
    """Test parity for StringToHash."""
    check_parity(
        "StringToHash",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_StringToNumber_parity():
    """Test parity for StringToNumber."""
    check_parity(
        "StringToNumber",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_StringUpper_parity():
    """Test parity for StringUpper."""
    check_parity(
        "StringUpper",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_Svd_parity():
    """Test parity for Svd."""
    check_parity(
        "Svd",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_Svdvals_parity():
    """Test parity for Svdvals."""
    check_parity(
        "Svdvals",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_Tensordot_parity():
    """Test parity for Tensordot."""
    check_parity(
        "Tensordot",
        lambda: [
            np.random.randn(2, 3, 4).astype(np.float32),
            np.random.randn(3, 4, 2).astype(np.float32),
        ],
    )


def test_Tensorinv_parity():
    """Test parity for Tensorinv."""
    check_parity(
        "Tensorinv",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_Tensorsolve_parity():
    """Test parity for Tensorsolve."""
    check_parity(
        "Tensorsolve",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_TimeDistributed_parity():
    """Test parity for TimeDistributed."""
    check_parity(
        "TimeDistributed",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_Trace_parity():
    """Test parity for Trace."""
    check_parity(
        "Trace",
        lambda: [np.random.randn(3, 3).astype(np.float32)],
    )


def test_TriInv_parity():
    """Test parity for TriInv."""
    check_parity(
        "TriInv",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_Triangular_parity():
    """Test parity for Triangular."""
    check_parity(
        "Triangular",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_TriangularSolve_parity():
    """Test parity for TriangularSolve."""
    check_parity(
        "TriangularSolve",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_Tridiagonal_parity():
    """Test parity for Tridiagonal."""
    check_parity(
        "Tridiagonal",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_TridiagonalSolve_parity():
    """Test parity for TridiagonalSolve."""
    check_parity(
        "TridiagonalSolve",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_Tril_parity():
    """Test parity for Tril."""
    check_parity(
        "Tril",
        lambda: [np.random.randn(3, 3).astype(np.float32)],
    )


def test_TrilIndices_parity():
    """Test parity for TrilIndices."""
    check_parity(
        "TrilIndices",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_TrilIndicesFrom_parity():
    """Test parity for TrilIndicesFrom."""
    check_parity(
        "TrilIndicesFrom",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_Triu_parity():
    """Test parity for Triu."""
    check_parity(
        "Triu",
        lambda: [np.random.randn(3, 3).astype(np.float32)],
    )


def test_TriuIndices_parity():
    """Test parity for TriuIndices."""
    check_parity(
        "TriuIndices",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_TriuIndicesFrom_parity():
    """Test parity for TriuIndicesFrom."""
    check_parity(
        "TriuIndicesFrom",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_UniqueInverse_parity():
    """Test parity for UniqueInverse."""
    check_parity(
        "UniqueInverse",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_UnsortedSegmentSqrtN_parity():
    """Test parity for UnsortedSegmentSqrtN."""
    check_parity(
        "UnsortedSegmentSqrtN",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_Vdot_parity():
    """Test parity for Vdot."""
    check_parity(
        "Vdot",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_Vecdot_parity():
    """Test parity for Vecdot."""
    check_parity(
        "Vecdot",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )
