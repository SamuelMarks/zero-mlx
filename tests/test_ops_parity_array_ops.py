"""Compiler operation parity tests for `array_ops` operations."""

from __future__ import annotations

import numpy as np
import pytest

from tests.conftest import check_parity


def test_Bitcast_parity() -> None:
    """Test parity for Bitcast."""
    check_parity(
        "Bitcast",
        lambda: [np.array([1.0, 2.0], dtype=np.float32)],
        kwargs_generator=lambda: {"type": np.int32},
    )


def test_BroadcastInDim_parity() -> None:
    """Test parity for BroadcastInDim."""
    check_parity(
        "Broadcast_In_Dim",
        lambda: [
            np.random.randn(3).astype(np.float32),
            [2, 3],
            [1],
        ],
    )


def test_BroadcastTo_parity() -> None:
    """Test parity for BroadcastTo."""
    check_parity(
        "Broadcast_To",
        lambda: [
            np.random.randn(2, 3).astype(np.float32),
            [4, 2, 3],
        ],
    )


def test_Cast_parity() -> None:
    """Test parity for Cast."""
    check_parity("Cast", lambda: [np.random.randn(2, 3).astype(np.float32), np.int32])


def test_Deg2Rad_parity() -> None:
    """Test parity for Deg2Rad."""
    check_parity("Deg2rad", lambda: [np.random.randn(2, 3).astype(np.float32)])


def test_DynamicSlice_parity() -> None:
    """Test parity for DynamicSlice."""
    check_parity("Dynamic_Slice", lambda: [])


def test_DynamicUpdateSlice_parity() -> None:
    """Test parity for DynamicUpdateSlice."""
    check_parity("Dynamic_Update_Slice", lambda: [])


def test_Fix_parity() -> None:
    """Test parity for Fix."""
    check_parity("Fix", lambda: [np.random.randn(2, 3).astype(np.float32)])


def test_Gcd_parity() -> None:
    """Test parity for Gcd."""
    check_parity(
        "Gcd",
        lambda: [
            np.random.randint(1, 100, size=(2, 3)).astype(np.int32),
            np.random.randint(1, 100, size=(2, 3)).astype(np.int32),
        ],
    )


def test_Hypot_parity() -> None:
    """Test parity for Hypot."""
    check_parity(
        "Hypot",
        lambda: [
            np.random.randn(2, 3).astype(np.float32),
            np.random.randn(2, 3).astype(np.float32),
        ],
    )


def test_Lcm_parity() -> None:
    """Test parity for Lcm."""
    check_parity(
        "Lcm",
        lambda: [
            np.random.randint(1, 100, size=(2, 3)).astype(np.int32),
            np.random.randint(1, 100, size=(2, 3)).astype(np.int32),
        ],
    )


def test_LeftShift_parity() -> None:
    """Test parity for LeftShift."""
    check_parity(
        "Left_Shift",
        lambda: [
            np.random.randint(1, 10, size=(2, 3)).astype(np.int32),
            np.random.randint(1, 3, size=(2, 3)).astype(np.int32),
        ],
    )


def test_ManualSeed_parity() -> None:
    """Test parity for ManualSeed."""
    check_parity("ManualSeed", lambda: [])


def test_Negative_parity() -> None:
    """Test parity for Negative."""
    check_parity("Negative", lambda: [np.random.randn(2, 3).astype(np.float32)])


def test_Nextafter_parity() -> None:
    """Test parity for Nextafter."""
    check_parity(
        "Nextafter",
        lambda: [
            np.random.randn(2, 3).astype(np.float32),
            np.random.randn(2, 3).astype(np.float32),
        ],
    )


def test_Positive_parity() -> None:
    """Test parity for Positive."""
    check_parity("Positive", lambda: [np.random.randn(2, 3).astype(np.float32)])


def test_Rad2Deg_parity() -> None:
    """Test parity for Rad2Deg."""
    check_parity("Rad2deg", lambda: [np.random.randn(2, 3).astype(np.float32)])


def test_Reciprocal_parity() -> None:
    """Test parity for Reciprocal."""
    check_parity(
        "Reciprocal",
        lambda: [np.random.uniform(0.1, 5, size=(2, 3)).astype(np.float32)],
    )


def test_ReduceWindow_parity() -> None:
    """Test parity for ReduceWindow."""
    check_parity("Reduce_Window", lambda: [])


def test_Remainder_parity() -> None:
    """Test parity for Remainder."""
    check_parity(
        "Remainder",
        lambda: [
            np.random.randint(1, 10, size=(2, 3)).astype(np.float32),
            np.random.randint(1, 10, size=(2, 3)).astype(np.float32),
        ],
    )


def test_Reshape_parity() -> None:
    """Test parity for Reshape."""
    check_parity("Reshape", lambda: [np.random.randn(2, 3).astype(np.float32), (3, 2)])


def test_Resize_parity() -> None:
    """Test parity for Resize."""
    check_parity("Resize", lambda: [])


def test_RightShift_parity() -> None:
    """Test parity for RightShift."""
    check_parity(
        "Right_Shift",
        lambda: [
            np.random.randint(1, 10, size=(2, 3)).astype(np.int32),
            np.random.randint(1, 3, size=(2, 3)).astype(np.int32),
        ],
    )


def test_Square_parity() -> None:
    """Test parity for Square."""
    check_parity("Square", lambda: [np.random.randn(2, 3).astype(np.float32)])


def test_TopK_parity() -> None:
    """Test parity for TopK."""
    check_parity(
        "Topk",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
        kwargs_generator=lambda: {"k": 2},
    )


def test_Transpose_parity() -> None:
    """Test parity for Transpose."""
    check_parity("Transpose", lambda: [np.random.randn(2, 3).astype(np.float32)])


def test_Trunc_parity() -> None:
    """Test parity for Trunc."""
    check_parity("Trunc", lambda: [np.random.randn(2, 3).astype(np.float32)])


def test_ActivityRegularization_parity() -> None:
    """Test parity for ActivityRegularization."""
    check_parity(
        "ActivityRegularization",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_Adjoint_parity() -> None:
    """Test parity for Adjoint."""
    check_parity(
        "Adjoint",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_AdjustBrightness_parity() -> None:
    """Test parity for AdjustBrightness."""
    check_parity(
        "AdjustBrightness",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_AdjustContrast_parity() -> None:
    """Test parity for AdjustContrast."""
    check_parity(
        "AdjustContrast",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_AdjustHue_parity() -> None:
    """Test parity for AdjustHue."""
    check_parity(
        "AdjustHue",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_AdjustSaturation_parity() -> None:
    """Test parity for AdjustSaturation."""
    check_parity(
        "AdjustSaturation",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_Angle_parity() -> None:
    """Test parity for Angle."""
    check_parity(
        "Angle",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_Assert_parity() -> None:
    """Test parity for Assert."""
    check_parity(
        "Assert",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_AssociativeScan_parity() -> None:
    """Test parity for AssociativeScan."""
    check_parity(
        "AssociativeScan",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_AugMix_parity() -> None:
    """Test parity for AugMix."""
    check_parity(
        "AugMix",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_AutoContrast_parity() -> None:
    """Test parity for AutoContrast."""
    check_parity(
        "AutoContrast",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_Bartlett_parity() -> None:
    """Test parity for Bartlett."""
    check_parity(
        "Bartlett",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_Binomial_parity() -> None:
    """Test parity for Binomial."""
    check_parity(
        "Binomial",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_Bits_parity() -> None:
    """Test parity for Bits."""
    check_parity(
        "Bits",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_Blackman_parity() -> None:
    """Test parity for Blackman."""
    check_parity(
        "Blackman",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_BooleanMask_parity() -> None:
    """Test parity for BooleanMask."""
    check_parity(
        "BooleanMask",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_Cauchy_parity() -> None:
    """Test parity for Cauchy."""
    check_parity(
        "Cauchy",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_Choose_parity() -> None:
    """Test parity for Choose."""
    check_parity(
        "Choose",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_Clz_parity() -> None:
    """Test parity for Clz."""
    check_parity(
        "Clz",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_ColumnStack_parity() -> None:
    """Test parity for ColumnStack."""
    check_parity(
        "ColumnStack",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_Compress_parity() -> None:
    """Test parity for Compress."""
    check_parity(
        "Compress",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_Concatenate_parity() -> None:
    """Test parity for Concatenate."""
    check_parity(
        "Concatenate",
        lambda: [
            [
                np.random.randn(2, 2).astype(np.float32),
                np.random.randn(2, 2).astype(np.float32),
            ]
        ],
    )


def test_Cov_parity() -> None:
    """Test parity for Cov."""
    check_parity(
        "Cov",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_Crop_parity() -> None:
    """Test parity for Crop."""
    check_parity(
        "Crop",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_CustomRoot_parity() -> None:
    """Test parity for CustomRoot."""
    check_parity(
        "CustomRoot",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_Cutmix_parity() -> None:
    """Test parity for Cutmix."""
    check_parity(
        "Cutmix",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_Dawsn_parity() -> None:
    """Test parity for Dawsn."""
    check_parity(
        "Dawsn",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_Degeneration_parity() -> None:
    """Test parity for Degeneration."""
    check_parity(
        "Degeneration",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_Delete_parity() -> None:
    """Test parity for Delete."""
    check_parity(
        "Delete",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_DevicePutReplicated_parity() -> None:
    """Test parity for DevicePutReplicated."""
    check_parity(
        "DevicePutReplicated",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_DevicePutSharded_parity() -> None:
    """Test parity for DevicePutSharded."""
    check_parity(
        "DevicePutSharded",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_Diff_parity() -> None:
    """Test parity for Diff."""
    check_parity(
        "Diff",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_Digitize_parity() -> None:
    """Test parity for Digitize."""
    check_parity(
        "Digitize",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_Dirichlet_parity() -> None:
    """Test parity for Dirichlet."""
    check_parity(
        "Dirichlet",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_DrawBoundingBoxes_parity() -> None:
    """Test parity for DrawBoundingBoxes."""
    check_parity(
        "DrawBoundingBoxes",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_Dsplit_parity() -> None:
    """Test parity for Dsplit."""
    check_parity(
        "Dsplit",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_Dstack_parity() -> None:
    """Test parity for Dstack."""
    check_parity(
        "Dstack",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_DynamicIndexInDim_parity() -> None:
    """Test parity for DynamicIndexInDim."""
    check_parity(
        "DynamicIndexInDim",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_DynamicPartition_parity() -> None:
    """Test parity for DynamicPartition."""
    check_parity(
        "DynamicPartition",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_DynamicShape_parity() -> None:
    """Test parity for DynamicShape."""
    check_parity(
        "DynamicShape",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_DynamicSliceInDim_parity() -> None:
    """Test parity for DynamicSliceInDim."""
    check_parity(
        "DynamicSliceInDim",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_DynamicStitch_parity() -> None:
    """Test parity for DynamicStitch."""
    check_parity(
        "DynamicStitch",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_DynamicUpdateIndexInDim_parity() -> None:
    """Test parity for DynamicUpdateIndexInDim."""
    check_parity(
        "DynamicUpdateIndexInDim",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_DynamicUpdateSliceInDim_parity() -> None:
    """Test parity for DynamicUpdateSliceInDim."""
    check_parity(
        "DynamicUpdateSliceInDim",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_ExtractBoundingBoxes_parity() -> None:
    """Test parity for ExtractBoundingBoxes."""
    check_parity(
        "ExtractBoundingBoxes",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_ExtractPatches_parity() -> None:
    """Test parity for ExtractPatches."""
    check_parity(
        "ExtractPatches",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_ExtractVolumePatches_parity() -> None:
    """Test parity for ExtractVolumePatches."""
    check_parity(
        "ExtractVolumePatches",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_F_parity() -> None:
    """Test parity for F."""
    check_parity(
        "F",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_Flatten_parity() -> None:
    """Test parity for Flatten."""
    check_parity(
        "Flatten",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_FlipUpDown_parity() -> None:
    """Test parity for FlipUpDown."""
    check_parity(
        "FlipUpDown",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_Fold_parity() -> None:
    """Test parity for Fold."""
    check_parity(
        "Fold",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_Frame_parity() -> None:
    """Test parity for Frame."""
    check_parity(
        "Frame",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_Gather_parity() -> None:
    """Test parity for Gather."""
    check_parity(
        "Gather",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_GatherNd_parity() -> None:
    """Test parity for GatherNd."""
    check_parity(
        "GatherNd",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_GaussianBlur_parity() -> None:
    """Test parity for GaussianBlur."""
    check_parity(
        "GaussianBlur",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_Gumbel_parity() -> None:
    """Test parity for Gumbel."""
    check_parity(
        "Gumbel",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_HannWindow_parity() -> None:
    """Test parity for HannWindow."""
    check_parity(
        "HannWindow",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_Hanning_parity() -> None:
    """Test parity for Hanning."""
    check_parity(
        "Hanning",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_Hashing_parity() -> None:
    """Test parity for Hashing."""
    check_parity(
        "Hashing",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_Hessenberg_parity() -> None:
    """Test parity for Hessenberg."""
    check_parity(
        "Hessenberg",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_Hsplit_parity() -> None:
    """Test parity for Hsplit."""
    check_parity(
        "Hsplit",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_Hstack_parity() -> None:
    """Test parity for Hstack."""
    check_parity(
        "Hstack",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_Inner_parity() -> None:
    """Test parity for Inner."""
    check_parity(
        "Inner",
        lambda: [
            np.random.randn(2, 3).astype(np.float32),
            np.random.randn(2, 3).astype(np.float32),
        ],
    )


def test_IntegerLookup_parity() -> None:
    """Test parity for IntegerLookup."""
    check_parity(
        "IntegerLookup",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_IoU_parity() -> None:
    """Test parity for IoU."""
    check_parity(
        "IoU",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_Key_parity() -> None:
    """Test parity for Key."""
    check_parity(
        "Key",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_KeyData_parity() -> None:
    """Test parity for KeyData."""
    check_parity(
        "KeyData",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_KeyImpl_parity() -> None:
    """Test parity for KeyImpl."""
    check_parity(
        "KeyImpl",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_Laplace_parity() -> None:
    """Test parity for Laplace."""
    check_parity(
        "Laplace",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_Lookup_parity() -> None:
    """Test parity for Lookup."""
    check_parity(
        "Lookup",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_Lu_parity() -> None:
    """Test parity for Lu."""
    check_parity(
        "Lu",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_LuPivotsToPermutation_parity() -> None:
    """Test parity for LuPivotsToPermutation."""
    check_parity(
        "LuPivotsToPermutation",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_MedianFilter_parity() -> None:
    """Test parity for MedianFilter."""
    check_parity(
        "MedianFilter",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_MelFilterbank_parity() -> None:
    """Test parity for MelFilterbank."""
    check_parity(
        "MelFilterbank",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_MelSpectrogram_parity() -> None:
    """Test parity for MelSpectrogram."""
    check_parity(
        "MelSpectrogram",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_Mfcc_parity() -> None:
    """Test parity for Mfcc."""
    check_parity(
        "Mfcc",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_Mixup_parity() -> None:
    """Test parity for Mixup."""
    check_parity(
        "Mixup",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_Outer_parity() -> None:
    """Test parity for Outer."""
    check_parity(
        "Outer",
        lambda: [
            np.random.randn(3).astype(np.float32),
            np.random.randn(4).astype(np.float32),
        ],
    )


def test_Outfeed_parity() -> None:
    """Test parity for Outfeed."""
    check_parity(
        "Outfeed",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_Packbits_parity() -> None:
    """Test parity for Packbits."""
    check_parity(
        "Packbits",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_PadToBoundingBox_parity() -> None:
    """Test parity for PadToBoundingBox."""
    check_parity(
        "PadToBoundingBox",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_Pareto_parity() -> None:
    """Test parity for Pareto."""
    check_parity(
        "Pareto",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_Partition_parity() -> None:
    """Test parity for Partition."""
    import mlx.core as mx
    import zero_mlx

    arr_np = np.array([3.0, 1.0, 4.0, 1.5, 2.0], dtype=np.float32)
    z_out = zero_mlx.partition(zero_mlx.array(arr_np), 2)
    m_out = mx.partition(mx.array(arr_np), 2)
    z_vals = np.array(z_out.tolist())
    m_vals = np.array(m_out)
    assert z_vals[2] == pytest.approx(m_vals[2])
    assert np.all(z_vals[:2] <= z_vals[2])
    assert np.all(z_vals[2:] >= z_vals[2])
    np.testing.assert_allclose(np.sort(z_vals), np.sort(m_vals))


def test_Pbroadcast_parity() -> None:
    """Test parity for Pbroadcast."""
    check_parity(
        "Pbroadcast",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_Permute_parity() -> None:
    """Test parity for Permute."""
    check_parity(
        "Permute",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_Polar_parity() -> None:
    """Test parity for Polar."""
    check_parity(
        "Polar",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_Poly_parity() -> None:
    """Test parity for Poly."""
    check_parity(
        "Poly",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_Polyder_parity() -> None:
    """Test parity for Polyder."""
    check_parity(
        "Polyder",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_Polyfit_parity() -> None:
    """Test parity for Polyfit."""
    check_parity(
        "Polyfit",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_Polyint_parity() -> None:
    """Test parity for Polyint."""
    check_parity(
        "Polyint",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_Polyval_parity() -> None:
    """Test parity for Polyval."""
    check_parity(
        "Polyval",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_Posterize_parity() -> None:
    """Test parity for Posterize."""
    check_parity(
        "Posterize",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_Ppermute_parity() -> None:
    """Test parity for Ppermute."""
    check_parity(
        "Ppermute",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_Pshuffle_parity() -> None:
    """Test parity for Pshuffle."""
    check_parity(
        "Pshuffle",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_Pswapaxes_parity() -> None:
    """Test parity for Pswapaxes."""
    check_parity(
        "Pswapaxes",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_RaggedDynamicBroadcast_parity() -> None:
    """Test parity for RaggedDynamicBroadcast."""
    check_parity(
        "RaggedDynamicBroadcast",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_RaggedGather_parity() -> None:
    """Test parity for RaggedGather."""
    check_parity(
        "RaggedGather",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_RaggedRange_parity() -> None:
    """Test parity for RaggedRange."""
    check_parity(
        "RaggedRange",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_RaggedRowSplitsToSegmentIds_parity() -> None:
    """Test parity for RaggedRowSplitsToSegmentIds."""
    check_parity(
        "RaggedRowSplitsToSegmentIds",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_RaggedStack_parity() -> None:
    """Test parity for RaggedStack."""
    check_parity(
        "RaggedStack",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_RaggedStackDynamicPartitions_parity() -> None:
    """Test parity for RaggedStackDynamicPartitions."""
    check_parity(
        "RaggedStackDynamicPartitions",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_Range_parity() -> None:
    """Test parity for Range."""
    check_parity(
        "Range",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_RawMerge_parity() -> None:
    """Test parity for RawMerge."""
    check_parity(
        "RawMerge",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_RawSwitch_parity() -> None:
    """Test parity for RawSwitch."""
    check_parity(
        "RawSwitch",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_ReduceScatter_parity() -> None:
    """Test parity for ReduceScatter."""
    check_parity(
        "ReduceScatter",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_RegexReplace_parity() -> None:
    """Test parity for RegexReplace."""
    check_parity(
        "RegexReplace",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_Repeat_parity() -> None:
    """Test parity for Repeat."""
    check_parity(
        "Repeat",
        lambda: [np.random.randn(2, 2).astype(np.float32), 2],
    )


def test_ResizeBicubic_parity() -> None:
    """Test parity for ResizeBicubic."""
    check_parity(
        "ResizeBicubic",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_ResizeBilinear_parity() -> None:
    """Test parity for ResizeBilinear."""
    check_parity(
        "ResizeBilinear",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_ResizeLanczos3_parity() -> None:
    """Test parity for ResizeLanczos3."""
    check_parity(
        "ResizeLanczos3",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_ResizeLanczos5_parity() -> None:
    """Test parity for ResizeLanczos5."""
    check_parity(
        "ResizeLanczos5",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_ResizeNearest_parity() -> None:
    """Test parity for ResizeNearest."""
    check_parity(
        "ResizeNearest",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_RgbToGrayscale_parity() -> None:
    """Test parity for RgbToGrayscale."""
    check_parity(
        "RgbToGrayscale",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_RgbToHsv_parity() -> None:
    """Test parity for RgbToHsv."""
    check_parity(
        "RgbToHsv",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_RgbToYiq_parity() -> None:
    """Test parity for RgbToYiq."""
    check_parity(
        "RgbToYiq",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_RgbToYuv_parity() -> None:
    """Test parity for RgbToYuv."""
    check_parity(
        "RgbToYuv",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_Roll_parity() -> None:
    """Test parity for Roll."""
    check_parity(
        "Roll",
        lambda: [np.random.randn(2, 2).astype(np.float32), 1],
    )


def test_Roots_parity() -> None:
    """Test parity for Roots."""
    check_parity(
        "Roots",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_RowStack_parity() -> None:
    """Test parity for RowStack."""
    check_parity(
        "RowStack",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_Scan_parity() -> None:
    """Test parity for Scan."""
    check_parity(
        "Scan",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_ScanBind_parity() -> None:
    """Test parity for ScanBind."""
    check_parity(
        "ScanBind",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_Scatter_parity() -> None:
    """Test parity for Scatter."""
    check_parity(
        "Scatter",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_ScatterApply_parity() -> None:
    """Test parity for ScatterApply."""
    check_parity(
        "ScatterApply",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_ScatterNd_parity() -> None:
    """Test parity for ScatterNd."""
    check_parity(
        "ScatterNd",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_Schur_parity() -> None:
    """Test parity for Schur."""
    check_parity(
        "Schur",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_Select_parity() -> None:
    """Test parity for Select."""
    check_parity(
        "Select",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_Setdiff1d_parity() -> None:
    """Test parity for Setdiff1d."""
    check_parity(
        "Setdiff1d",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_Sharpen_parity() -> None:
    """Test parity for Sharpen."""
    check_parity(
        "Sharpen",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_Slice_parity() -> None:
    """Test parity for Slice."""
    check_parity(
        "Slice",
        lambda: [
            np.random.randn(5).astype(np.float32),
            np.array([1], dtype=np.int32),
            [0],
            [3],
        ],
    )


def test_SliceInDim_parity() -> None:
    """Test parity for SliceInDim."""
    check_parity(
        "SliceInDim",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_SobolSample_parity() -> None:
    """Test parity for SobolSample."""
    check_parity(
        "SobolSample",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_Solarize_parity() -> None:
    """Test parity for Solarize."""
    check_parity(
        "Solarize",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_SparseMapValues_parity() -> None:
    """Test parity for SparseMapValues."""
    check_parity(
        "SparseMapValues",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_SparseMask_parity() -> None:
    """Test parity for SparseMask."""
    check_parity(
        "SparseMask",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_SparseResetShape_parity() -> None:
    """Test parity for SparseResetShape."""
    check_parity(
        "SparseResetShape",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_SparseReshape_parity() -> None:
    """Test parity for SparseReshape."""
    check_parity(
        "SparseReshape",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_SparseRetain_parity() -> None:
    """Test parity for SparseRetain."""
    check_parity(
        "SparseRetain",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_SparseSlice_parity() -> None:
    """Test parity for SparseSlice."""
    check_parity(
        "SparseSlice",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_SparseTranspose_parity() -> None:
    """Test parity for SparseTranspose."""
    check_parity(
        "SparseTranspose",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_Spence_parity() -> None:
    """Test parity for Spence."""
    check_parity(
        "Spence",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_Split_parity() -> None:
    """Test parity for Split."""
    check_parity(
        "Split",
        lambda: [np.random.randn(4).astype(np.float32), 2],
    )


def test_SquaredDifference_parity() -> None:
    """Test parity for SquaredDifference."""
    check_parity(
        "SquaredDifference",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_Squeeze_parity() -> None:
    """Test parity for Squeeze."""
    check_parity(
        "Squeeze",
        lambda: [np.random.randn(2, 1, 3).astype(np.float32)],
    )


def test_Stack_parity() -> None:
    """Test parity for Stack."""
    check_parity(
        "Stack",
        lambda: [
            [
                np.random.randn(2, 2).astype(np.float32),
                np.random.randn(2, 2).astype(np.float32),
            ]
        ],
    )


def test_Swapaxes_parity() -> None:
    """Test parity for Swapaxes."""
    check_parity(
        "Swapaxes",
        lambda: [np.random.randn(2, 3, 4).astype(np.float32), 0, 2],
    )


def test_Switch_parity() -> None:
    """Test parity for Switch."""
    check_parity(
        "Switch",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_T_parity() -> None:
    """Test parity for T."""
    check_parity(
        "T",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_Take_parity() -> None:
    """Test parity for Take."""
    check_parity(
        "Take",
        lambda: [np.random.randn(5).astype(np.float32), np.array([0, 2, 4])],
    )


def test_Tile_parity() -> None:
    """Test parity for Tile."""
    check_parity(
        "Tile",
        lambda: [np.random.randn(2, 2).astype(np.float32), (2, 2)],
    )


def test_TrapezoidalIntegral_parity() -> None:
    """Test parity for TrapezoidalIntegral."""
    check_parity(
        "TrapezoidalIntegral",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_Unfold_parity() -> None:
    """Test parity for Unfold."""
    check_parity(
        "Unfold",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_Union1d_parity() -> None:
    """Test parity for Union1d."""
    check_parity(
        "Union1d",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_UniqueValues_parity() -> None:
    """Test parity for UniqueValues."""
    check_parity(
        "UniqueValues",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_Unpackbits_parity() -> None:
    """Test parity for Unpackbits."""
    check_parity(
        "Unpackbits",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_UnravelIndex_parity() -> None:
    """Test parity for UnravelIndex."""
    check_parity(
        "UnravelIndex",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_Vsplit_parity() -> None:
    """Test parity for Vsplit."""
    check_parity(
        "Vsplit",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_Vstack_parity() -> None:
    """Test parity for Vstack."""
    check_parity(
        "Vstack",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_Wald_parity() -> None:
    """Test parity for Wald."""
    check_parity(
        "Wald",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_Where_parity() -> None:
    """Test parity for Where."""
    check_parity(
        "Where",
        lambda: [
            np.random.rand(2, 2) > 0.5,
            np.random.randn(2, 2).astype(np.float32),
            np.random.randn(2, 2).astype(np.float32),
        ],
    )


def test_WhileLoop_parity() -> None:
    """Test parity for WhileLoop."""
    check_parity(
        "WhileLoop",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_WithShardingConstraint_parity() -> None:
    """Test parity for WithShardingConstraint."""
    check_parity(
        "WithShardingConstraint",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_WrapKeyData_parity() -> None:
    """Test parity for WrapKeyData."""
    check_parity(
        "WrapKeyData",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_Zeta_parity() -> None:
    """Test parity for Zeta."""
    check_parity(
        "Zeta",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_BlockMaskedMm_parity() -> None:
    """Test parity for BlockMaskedMm."""
    check_parity("BlockMaskedMm", lambda: [np.random.randn(2, 3).astype(np.float32)])


def test_GatherMm_parity() -> None:
    """Test parity for GatherMm."""
    check_parity("GatherMm", lambda: [np.random.randn(2, 3).astype(np.float32)])


def test_SegmentedMm_parity() -> None:
    """Test parity for SegmentedMm."""
    check_parity("SegmentedMm", lambda: [np.random.randn(2, 3).astype(np.float32)])


def test_Logcumsumexp_parity() -> None:
    """Test parity for Logcumsumexp."""
    check_parity("Logcumsumexp", lambda: [np.random.randn(2, 3).astype(np.float32)])


def test_PutAlongAxis_parity() -> None:
    """Test parity for PutAlongAxis."""
    check_parity("PutAlongAxis", lambda: [np.random.randn(2, 3).astype(np.float32)])


def test_GetItem_parity() -> None:
    """Test parity for GetItem."""
    check_parity("GetItem", lambda: [np.random.randn(2, 3).astype(np.float32)])
