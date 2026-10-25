"""Compiler operation parity tests for `reductions` operations."""

from __future__ import annotations

import pytest

from tests.conftest import check_parity, np


def test_Argmax_parity() -> None:
    """Test parity for Argmax."""
    check_parity("Argmax", lambda: [np.random.randn(2, 3).astype(np.float32)])


def test_Argmin_parity() -> None:
    """Test parity for Argmin."""
    check_parity("Argmin", lambda: [np.random.randn(2, 3).astype(np.float32)])


def test_AssignVariable_parity() -> None:
    """Test parity for AssignVariable."""
    check_parity("AssignVariable", lambda: [])  # Typically stateful, hard to verify blindly


def test_CountNonzero_parity() -> None:
    """Test parity for CountNonzero."""
    check_parity(
        "Count_Nonzero",
        lambda: [np.random.choice([0, 1, 2], size=(2, 3)).astype(np.float32)],
    )


def test_Cumsum_parity() -> None:
    """Test parity for Cumsum."""
    check_parity(
        "Cumsum",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
        kwargs_generator=lambda: {"axis": 0},
    )


def test_Einsum_parity() -> None:
    """Test parity for Einsum."""
    check_parity(
        "Einsum",
        lambda: [
            "ij,jk->ik",
            np.random.randn(2, 3).astype(np.float32),
            np.random.randn(3, 4).astype(np.float32),
        ],
    )


def test_Fmax_parity() -> None:
    """Test parity for Fmax."""
    check_parity(
        "Fmax",
        lambda: [
            np.random.randn(2, 3).astype(np.float32),
            np.random.randn(2, 3).astype(np.float32),
        ],
    )


def test_Fmin_parity() -> None:
    """Test parity for Fmin."""
    check_parity(
        "Fmin",
        lambda: [
            np.random.randn(2, 3).astype(np.float32),
            np.random.randn(2, 3).astype(np.float32),
        ],
    )


def test_Logsumexp_parity() -> None:
    """Test parity for Logsumexp."""
    check_parity("Logsumexp", lambda: [np.random.randn(2, 3).astype(np.float32)])


def test_Max_parity() -> None:
    """Test parity for Max."""
    check_parity("Max", lambda: [np.random.randn(2, 3).astype(np.float32)])


def test_Maximum_parity() -> None:
    """Test parity for Maximum."""
    check_parity(
        "Maximum",
        lambda: [
            np.random.randn(2, 3).astype(np.float32),
            np.random.randn(2, 3).astype(np.float32),
        ],
    )


def test_Mean_parity() -> None:
    """Test parity for Mean."""
    check_parity("Mean", lambda: [np.random.randn(2, 3).astype(np.float32)])


def test_Min_parity() -> None:
    """Test parity for Min."""
    check_parity("Min", lambda: [np.random.randn(2, 3).astype(np.float32)])


def test_Minimum_parity() -> None:
    """Test parity for Minimum."""
    check_parity(
        "Minimum",
        lambda: [
            np.random.randn(2, 3).astype(np.float32),
            np.random.randn(2, 3).astype(np.float32),
        ],
    )


def test_Pmean_parity() -> None:
    """Test parity for Pmean."""
    check_parity("Pmean", lambda: [np.random.randn(2, 3).astype(np.float32)])


def test_Prod_parity() -> None:
    """Test parity for Prod."""
    check_parity("Prod", lambda: [np.random.randn(2, 3).astype(np.float32)])


def test_Psum_parity() -> None:
    """Test parity for Psum."""
    check_parity("Psum", lambda: [np.random.randn(2, 3).astype(np.float32)])


def test_ReadVariable_parity() -> None:
    """Test parity for ReadVariable."""
    check_parity("ReadVariable", lambda: [])


def test_SegmentSum_parity() -> None:
    """Test parity for SegmentSum."""
    check_parity("SegmentSum", lambda: [])


def test_Std_parity() -> None:
    """Test parity for Std."""
    check_parity("Std", lambda: [np.random.randn(2, 3).astype(np.float32)])


def test_Sum_parity() -> None:
    """Test parity for Sum."""
    check_parity("Sum", lambda: [np.random.randn(2, 3).astype(np.float32)])


def test_Variance_parity() -> None:
    """Test parity for Variance."""
    check_parity("Variance", lambda: [np.random.randn(2, 3).astype(np.float32)])


def test_ApproxMaxK_parity() -> None:
    """Test parity for ApproxMaxK."""
    check_parity(
        "ApproxMaxK",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_ApproxMaxKIndices_parity() -> None:
    """Test parity for ApproxMaxKIndices."""
    check_parity(
        "ApproxMaxKIndices",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_ApproxMinK_parity() -> None:
    """Test parity for ApproxMinK."""
    check_parity(
        "ApproxMinK",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_ApproxMinKIndices_parity() -> None:
    """Test parity for ApproxMinKIndices."""
    check_parity(
        "ApproxMinKIndices",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_ArgSort_parity() -> None:
    """Test parity for ArgSort."""
    check_parity(
        "ArgSort",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_Argpartition_parity() -> None:
    """Test parity for Argpartition."""
    import zero_mlx

    arr_np = np.array([3.0, 1.0, 4.0, 1.5, 2.0], dtype=np.float32)
    z_idx = zero_mlx.argpartition(zero_mlx.array(arr_np), 2)
    m_idx = np.argpartition(arr_np, 2)
    z_part = arr_np[np.array(z_idx.tolist(), dtype=np.int64)]
    m_part = arr_np[np.array(m_idx, dtype=np.int64)]
    assert z_part[2].item() == pytest.approx(m_part[2].item())
    assert np.all(z_part[:2] <= z_part[2])
    assert np.all(z_part[2:] >= z_part[2])
    np.testing.assert_allclose(np.sort(z_idx.tolist()), np.sort(np.array(m_idx)))


def test_Argwhere_parity() -> None:
    """Test parity for Argwhere."""
    check_parity(
        "Argwhere",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_Bincount_parity() -> None:
    """Test parity for Bincount."""
    check_parity(
        "Bincount",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_BitwiseCount_parity() -> None:
    """Test parity for BitwiseCount."""
    check_parity(
        "BitwiseCount",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_Cumlogsumexp_parity() -> None:
    """Test parity for Cumlogsumexp."""
    check_parity(
        "Cumlogsumexp",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_Cummax_parity() -> None:
    """Test parity for Cummax."""
    check_parity(
        "Cummax",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_Cummin_parity() -> None:
    """Test parity for Cummin."""
    check_parity(
        "Cummin",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_Cumprod_parity() -> None:
    """Test parity for Cumprod."""
    check_parity(
        "Cumprod",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_CumulativeLogsumexp_parity() -> None:
    """Test parity for CumulativeLogsumexp."""
    check_parity(
        "CumulativeLogsumexp",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_DoubleSidedMaxwell_parity() -> None:
    """Test parity for DoubleSidedMaxwell."""
    check_parity(
        "DoubleSidedMaxwell",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_GroupMean_parity() -> None:
    """Test parity for GroupMean."""
    check_parity(
        "GroupMean",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_GroupVariance_parity() -> None:
    """Test parity for GroupVariance."""
    check_parity(
        "GroupVariance",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_Hamming_parity() -> None:
    """Test parity for Hamming."""
    check_parity(
        "Hamming",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_HammingWindow_parity() -> None:
    """Test parity for HammingWindow."""
    check_parity(
        "HammingWindow",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_HouseholderProduct_parity() -> None:
    """Test parity for HouseholderProduct."""
    check_parity(
        "HouseholderProduct",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_Maxwell_parity() -> None:
    """Test parity for Maxwell."""
    check_parity(
        "Maxwell",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_Nanargmax_parity() -> None:
    """Test parity for Nanargmax."""
    check_parity(
        "Nanargmax",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_Nanargmin_parity() -> None:
    """Test parity for Nanargmin."""
    check_parity(
        "Nanargmin",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_Nancumprod_parity() -> None:
    """Test parity for Nancumprod."""
    check_parity(
        "Nancumprod",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_Nancumsum_parity() -> None:
    """Test parity for Nancumsum."""
    check_parity(
        "Nancumsum",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_Nanmax_parity() -> None:
    """Test parity for Nanmax."""
    check_parity(
        "Nanmax",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_Nanmean_parity() -> None:
    """Test parity for Nanmean."""
    check_parity(
        "Nanmean",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_Nanmin_parity() -> None:
    """Test parity for Nanmin."""
    check_parity(
        "Nanmin",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_Nanprod_parity() -> None:
    """Test parity for Nanprod."""
    check_parity(
        "Nanprod",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_Nanstd_parity() -> None:
    """Test parity for Nanstd."""
    check_parity(
        "Nanstd",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_Nansum_parity() -> None:
    """Test parity for Nansum."""
    check_parity(
        "Nansum",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_Nanvar_parity() -> None:
    """Test parity for Nanvar."""
    check_parity(
        "Nanvar",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_NonMaxSuppression_parity() -> None:
    """Test parity for NonMaxSuppression."""
    check_parity(
        "NonMaxSuppression",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_Pmax_parity() -> None:
    """Test parity for Pmax."""
    check_parity(
        "Pmax",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_Pmin_parity() -> None:
    """Test parity for Pmin."""
    check_parity(
        "Pmin",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_PopulationCount_parity() -> None:
    """Test parity for PopulationCount."""
    check_parity(
        "PopulationCount",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_PsumScatter_parity() -> None:
    """Test parity for PsumScatter."""
    check_parity(
        "PsumScatter",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_ScatterMax_parity() -> None:
    """Test parity for ScatterMax."""
    check_parity(
        "ScatterMax",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_ScatterMin_parity() -> None:
    """Test parity for ScatterMin."""
    check_parity(
        "ScatterMin",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_SegmentMax_parity() -> None:
    """Test parity for SegmentMax."""
    check_parity(
        "SegmentMax",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_SegmentMean_parity() -> None:
    """Test parity for SegmentMean."""
    check_parity(
        "SegmentMean",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_SegmentMin_parity() -> None:
    """Test parity for SegmentMin."""
    check_parity(
        "SegmentMin",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_SegmentProd_parity() -> None:
    """Test parity for SegmentProd."""
    check_parity(
        "SegmentProd",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_SparseBincount_parity() -> None:
    """Test parity for SparseBincount."""
    check_parity(
        "SparseBincount",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_SparseMaximum_parity() -> None:
    """Test parity for SparseMaximum."""
    check_parity(
        "SparseMaximum",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_SparseMinimum_parity() -> None:
    """Test parity for SparseMinimum."""
    check_parity(
        "SparseMinimum",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_SparseReduceMax_parity() -> None:
    """Test parity for SparseReduceMax."""
    check_parity(
        "SparseReduceMax",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_SparseReduceSum_parity() -> None:
    """Test parity for SparseReduceSum."""
    check_parity(
        "SparseReduceSum",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_SparseSegmentMean_parity() -> None:
    """Test parity for SparseSegmentMean."""
    check_parity(
        "SparseSegmentMean",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_SparseSegmentSum_parity() -> None:
    """Test parity for SparseSegmentSum."""
    check_parity(
        "SparseSegmentSum",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_UniqueCounts_parity() -> None:
    """Test parity for UniqueCounts."""
    check_parity(
        "UniqueCounts",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_UnsortedSegmentMax_parity() -> None:
    """Test parity for UnsortedSegmentMax."""
    check_parity(
        "UnsortedSegmentMax",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_UnsortedSegmentMean_parity() -> None:
    """Test parity for UnsortedSegmentMean."""
    check_parity(
        "UnsortedSegmentMean",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_UnsortedSegmentMin_parity() -> None:
    """Test parity for UnsortedSegmentMin."""
    check_parity(
        "UnsortedSegmentMin",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_UnsortedSegmentProd_parity() -> None:
    """Test parity for UnsortedSegmentProd."""
    check_parity(
        "UnsortedSegmentProd",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_UnsortedSegmentSum_parity() -> None:
    """Test parity for UnsortedSegmentSum."""
    check_parity(
        "UnsortedSegmentSum",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_WeibullMin_parity() -> None:
    """Test parity for WeibullMin."""
    check_parity(
        "WeibullMin",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )
