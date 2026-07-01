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


def test_Argmax_parity():
    """Test parity for Argmax."""
    check_parity("Argmax", lambda: [np.random.randn(2, 3).astype(np.float32)])


def test_Argmin_parity():
    """Test parity for Argmin."""
    check_parity("Argmin", lambda: [np.random.randn(2, 3).astype(np.float32)])


def test_AssignVariable_parity():
    """Test parity for AssignVariable."""
    check_parity(
        "AssignVariable", lambda: []
    )  # Typically stateful, hard to verify blindly


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


def test_Logsumexp_parity():
    """Test parity for Logsumexp."""
    check_parity("Logsumexp", lambda: [np.random.randn(2, 3).astype(np.float32)])


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


def test_Pmean_parity():
    """Test parity for Pmean."""
    check_parity("Pmean", lambda: [np.random.randn(2, 3).astype(np.float32)])


def test_Prod_parity():
    """Test parity for Prod."""
    check_parity("Prod", lambda: [np.random.randn(2, 3).astype(np.float32)])


def test_Psum_parity():
    """Test parity for Psum."""
    check_parity("Psum", lambda: [np.random.randn(2, 3).astype(np.float32)])


def test_ReadVariable_parity():
    """Test parity for ReadVariable."""
    check_parity("ReadVariable", lambda: [])


def test_SegmentSum_parity():
    """Test parity for SegmentSum."""
    check_parity("SegmentSum", lambda: [])


def test_Std_parity():
    """Test parity for Std."""
    check_parity("Std", lambda: [np.random.randn(2, 3).astype(np.float32)])


def test_Sum_parity():
    """Test parity for Sum."""
    check_parity("Sum", lambda: [np.random.randn(2, 3).astype(np.float32)])


def test_Variance_parity():
    """Test parity for Variance."""
    check_parity("Variance", lambda: [np.random.randn(2, 3).astype(np.float32)])


def test_ApproxMaxK_parity():
    """Test parity for ApproxMaxK."""
    check_parity(
        "ApproxMaxK",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_ApproxMaxKIndices_parity():
    """Test parity for ApproxMaxKIndices."""
    check_parity(
        "ApproxMaxKIndices",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_ApproxMinK_parity():
    """Test parity for ApproxMinK."""
    check_parity(
        "ApproxMinK",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_ApproxMinKIndices_parity():
    """Test parity for ApproxMinKIndices."""
    check_parity(
        "ApproxMinKIndices",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_ArgSort_parity():
    """Test parity for ArgSort."""
    check_parity(
        "ArgSort",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_Argpartition_parity():
    """Test parity for Argpartition."""
    check_parity(
        "Argpartition",
        lambda: [np.random.randn(5).astype(np.float32), 2],
    )


def test_Argwhere_parity():
    """Test parity for Argwhere."""
    check_parity(
        "Argwhere",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_Bincount_parity():
    """Test parity for Bincount."""
    check_parity(
        "Bincount",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_BitwiseCount_parity():
    """Test parity for BitwiseCount."""
    check_parity(
        "BitwiseCount",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_Cumlogsumexp_parity():
    """Test parity for Cumlogsumexp."""
    check_parity(
        "Cumlogsumexp",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_Cummax_parity():
    """Test parity for Cummax."""
    check_parity(
        "Cummax",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_Cummin_parity():
    """Test parity for Cummin."""
    check_parity(
        "Cummin",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_Cumprod_parity():
    """Test parity for Cumprod."""
    check_parity(
        "Cumprod",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_CumulativeLogsumexp_parity():
    """Test parity for CumulativeLogsumexp."""
    check_parity(
        "CumulativeLogsumexp",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_DoubleSidedMaxwell_parity():
    """Test parity for DoubleSidedMaxwell."""
    check_parity(
        "DoubleSidedMaxwell",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_GroupMean_parity():
    """Test parity for GroupMean."""
    check_parity(
        "GroupMean",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_GroupVariance_parity():
    """Test parity for GroupVariance."""
    check_parity(
        "GroupVariance",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_Hamming_parity():
    """Test parity for Hamming."""
    check_parity(
        "Hamming",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_HammingWindow_parity():
    """Test parity for HammingWindow."""
    check_parity(
        "HammingWindow",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_HouseholderProduct_parity():
    """Test parity for HouseholderProduct."""
    check_parity(
        "HouseholderProduct",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_Maxwell_parity():
    """Test parity for Maxwell."""
    check_parity(
        "Maxwell",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_Nanargmax_parity():
    """Test parity for Nanargmax."""
    check_parity(
        "Nanargmax",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_Nanargmin_parity():
    """Test parity for Nanargmin."""
    check_parity(
        "Nanargmin",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_Nancumprod_parity():
    """Test parity for Nancumprod."""
    check_parity(
        "Nancumprod",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_Nancumsum_parity():
    """Test parity for Nancumsum."""
    check_parity(
        "Nancumsum",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_Nanmax_parity():
    """Test parity for Nanmax."""
    check_parity(
        "Nanmax",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_Nanmean_parity():
    """Test parity for Nanmean."""
    check_parity(
        "Nanmean",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_Nanmin_parity():
    """Test parity for Nanmin."""
    check_parity(
        "Nanmin",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_Nanprod_parity():
    """Test parity for Nanprod."""
    check_parity(
        "Nanprod",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_Nanstd_parity():
    """Test parity for Nanstd."""
    check_parity(
        "Nanstd",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_Nansum_parity():
    """Test parity for Nansum."""
    check_parity(
        "Nansum",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_Nanvar_parity():
    """Test parity for Nanvar."""
    check_parity(
        "Nanvar",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_NonMaxSuppression_parity():
    """Test parity for NonMaxSuppression."""
    check_parity(
        "NonMaxSuppression",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_Pmax_parity():
    """Test parity for Pmax."""
    check_parity(
        "Pmax",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_Pmin_parity():
    """Test parity for Pmin."""
    check_parity(
        "Pmin",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_PopulationCount_parity():
    """Test parity for PopulationCount."""
    check_parity(
        "PopulationCount",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_PsumScatter_parity():
    """Test parity for PsumScatter."""
    check_parity(
        "PsumScatter",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_ScatterMax_parity():
    """Test parity for ScatterMax."""
    check_parity(
        "ScatterMax",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_ScatterMin_parity():
    """Test parity for ScatterMin."""
    check_parity(
        "ScatterMin",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_SegmentMax_parity():
    """Test parity for SegmentMax."""
    check_parity(
        "SegmentMax",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_SegmentMean_parity():
    """Test parity for SegmentMean."""
    check_parity(
        "SegmentMean",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_SegmentMin_parity():
    """Test parity for SegmentMin."""
    check_parity(
        "SegmentMin",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_SegmentProd_parity():
    """Test parity for SegmentProd."""
    check_parity(
        "SegmentProd",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_SparseBincount_parity():
    """Test parity for SparseBincount."""
    check_parity(
        "SparseBincount",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_SparseMaximum_parity():
    """Test parity for SparseMaximum."""
    check_parity(
        "SparseMaximum",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_SparseMinimum_parity():
    """Test parity for SparseMinimum."""
    check_parity(
        "SparseMinimum",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_SparseReduceMax_parity():
    """Test parity for SparseReduceMax."""
    check_parity(
        "SparseReduceMax",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_SparseReduceSum_parity():
    """Test parity for SparseReduceSum."""
    check_parity(
        "SparseReduceSum",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_SparseSegmentMean_parity():
    """Test parity for SparseSegmentMean."""
    check_parity(
        "SparseSegmentMean",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_SparseSegmentSum_parity():
    """Test parity for SparseSegmentSum."""
    check_parity(
        "SparseSegmentSum",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_UniqueCounts_parity():
    """Test parity for UniqueCounts."""
    check_parity(
        "UniqueCounts",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_UnsortedSegmentMax_parity():
    """Test parity for UnsortedSegmentMax."""
    check_parity(
        "UnsortedSegmentMax",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_UnsortedSegmentMean_parity():
    """Test parity for UnsortedSegmentMean."""
    check_parity(
        "UnsortedSegmentMean",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_UnsortedSegmentMin_parity():
    """Test parity for UnsortedSegmentMin."""
    check_parity(
        "UnsortedSegmentMin",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_UnsortedSegmentProd_parity():
    """Test parity for UnsortedSegmentProd."""
    check_parity(
        "UnsortedSegmentProd",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_UnsortedSegmentSum_parity():
    """Test parity for UnsortedSegmentSum."""
    check_parity(
        "UnsortedSegmentSum",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_WeibullMin_parity():
    """Test parity for WeibullMin."""
    check_parity(
        "WeibullMin",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )
