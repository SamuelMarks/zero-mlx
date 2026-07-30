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

    def process_arg(a, array_fn):
        if isinstance(a, np.ndarray):
            return array_fn(a)
        elif isinstance(a, list):
            return [process_arg(i, array_fn) for i in a]
        elif isinstance(a, tuple):
            return tuple(process_arg(i, array_fn) for i in a)
        return unwrap(a)

    z_args = [process_arg(a, zero_mlx.array) for a in args]
    m_args = [process_arg(a, mx.array) for a in args]

    try:
        z_res = z_func(*z_args, **kwargs)
        if hasattr(zero_mlx, "eval"):
            if isinstance(z_res, (list, tuple)):
                zero_mlx.eval(*z_res)
            else:
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


def test_Bitcast_parity():
    """Test parity for Bitcast."""
    check_parity(
        "Bitcast",
        lambda: [np.array([1.0, 2.0], dtype=np.float32)],
        kwargs_generator=lambda: {"type": np.int32},
    )


def test_BroadcastInDim_parity():
    """Test parity for BroadcastInDim."""
    check_parity(
        "Broadcast_In_Dim",
        lambda: [
            np.random.randn(3).astype(np.float32),
            [2, 3],
            [1],
        ],
    )


def test_BroadcastTo_parity():
    """Test parity for BroadcastTo."""
    check_parity(
        "Broadcast_To",
        lambda: [
            np.random.randn(2, 3).astype(np.float32),
            [4, 2, 3],
        ],
    )


def test_Cast_parity():
    """Test parity for Cast."""
    check_parity("Cast", lambda: [np.random.randn(2, 3).astype(np.float32), np.int32])


def test_Deg2Rad_parity():
    """Test parity for Deg2Rad."""
    check_parity("Deg2rad", lambda: [np.random.randn(2, 3).astype(np.float32)])


def test_DynamicSlice_parity():
    """Test parity for DynamicSlice."""
    check_parity("Dynamic_Slice", lambda: [])


def test_DynamicUpdateSlice_parity():
    """Test parity for DynamicUpdateSlice."""
    check_parity("Dynamic_Update_Slice", lambda: [])


def test_Fix_parity():
    """Test parity for Fix."""
    check_parity("Fix", lambda: [np.random.randn(2, 3).astype(np.float32)])


def test_Gcd_parity():
    """Test parity for Gcd."""
    check_parity(
        "Gcd",
        lambda: [
            np.random.randint(1, 100, size=(2, 3)).astype(np.int32),
            np.random.randint(1, 100, size=(2, 3)).astype(np.int32),
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


def test_Lcm_parity():
    """Test parity for Lcm."""
    check_parity(
        "Lcm",
        lambda: [
            np.random.randint(1, 100, size=(2, 3)).astype(np.int32),
            np.random.randint(1, 100, size=(2, 3)).astype(np.int32),
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


def test_ManualSeed_parity():
    """Test parity for ManualSeed."""
    check_parity("ManualSeed", lambda: [])


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


def test_Positive_parity():
    """Test parity for Positive."""
    check_parity("Positive", lambda: [np.random.randn(2, 3).astype(np.float32)])


def test_Rad2Deg_parity():
    """Test parity for Rad2Deg."""
    check_parity("Rad2deg", lambda: [np.random.randn(2, 3).astype(np.float32)])


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


def test_RightShift_parity():
    """Test parity for RightShift."""
    check_parity(
        "Right_Shift",
        lambda: [
            np.random.randint(1, 10, size=(2, 3)).astype(np.int32),
            np.random.randint(1, 3, size=(2, 3)).astype(np.int32),
        ],
    )


def test_Square_parity():
    """Test parity for Square."""
    check_parity("Square", lambda: [np.random.randn(2, 3).astype(np.float32)])


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


def test_Trunc_parity():
    """Test parity for Trunc."""
    check_parity("Trunc", lambda: [np.random.randn(2, 3).astype(np.float32)])


def test_ActivityRegularization_parity():
    """Test parity for ActivityRegularization."""
    check_parity(
        "ActivityRegularization",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_Adjoint_parity():
    """Test parity for Adjoint."""
    check_parity(
        "Adjoint",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_AdjustBrightness_parity():
    """Test parity for AdjustBrightness."""
    check_parity(
        "AdjustBrightness",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_AdjustContrast_parity():
    """Test parity for AdjustContrast."""
    check_parity(
        "AdjustContrast",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_AdjustHue_parity():
    """Test parity for AdjustHue."""
    check_parity(
        "AdjustHue",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_AdjustSaturation_parity():
    """Test parity for AdjustSaturation."""
    check_parity(
        "AdjustSaturation",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_Angle_parity():
    """Test parity for Angle."""
    check_parity(
        "Angle",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_Assert_parity():
    """Test parity for Assert."""
    check_parity(
        "Assert",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_AssociativeScan_parity():
    """Test parity for AssociativeScan."""
    check_parity(
        "AssociativeScan",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_AugMix_parity():
    """Test parity for AugMix."""
    check_parity(
        "AugMix",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_AutoContrast_parity():
    """Test parity for AutoContrast."""
    check_parity(
        "AutoContrast",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_Bartlett_parity():
    """Test parity for Bartlett."""
    check_parity(
        "Bartlett",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_Binomial_parity():
    """Test parity for Binomial."""
    check_parity(
        "Binomial",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_Bits_parity():
    """Test parity for Bits."""
    check_parity(
        "Bits",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_Blackman_parity():
    """Test parity for Blackman."""
    check_parity(
        "Blackman",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_BooleanMask_parity():
    """Test parity for BooleanMask."""
    check_parity(
        "BooleanMask",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_Cauchy_parity():
    """Test parity for Cauchy."""
    check_parity(
        "Cauchy",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_Choose_parity():
    """Test parity for Choose."""
    check_parity(
        "Choose",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_Clz_parity():
    """Test parity for Clz."""
    check_parity(
        "Clz",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_ColumnStack_parity():
    """Test parity for ColumnStack."""
    check_parity(
        "ColumnStack",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_Compress_parity():
    """Test parity for Compress."""
    check_parity(
        "Compress",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_Concatenate_parity():
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


def test_Cov_parity():
    """Test parity for Cov."""
    check_parity(
        "Cov",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_Crop_parity():
    """Test parity for Crop."""
    check_parity(
        "Crop",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_CustomRoot_parity():
    """Test parity for CustomRoot."""
    check_parity(
        "CustomRoot",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_Cutmix_parity():
    """Test parity for Cutmix."""
    check_parity(
        "Cutmix",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_Dawsn_parity():
    """Test parity for Dawsn."""
    check_parity(
        "Dawsn",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_Degeneration_parity():
    """Test parity for Degeneration."""
    check_parity(
        "Degeneration",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_Delete_parity():
    """Test parity for Delete."""
    check_parity(
        "Delete",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_DevicePutReplicated_parity():
    """Test parity for DevicePutReplicated."""
    check_parity(
        "DevicePutReplicated",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_DevicePutSharded_parity():
    """Test parity for DevicePutSharded."""
    check_parity(
        "DevicePutSharded",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_Diff_parity():
    """Test parity for Diff."""
    check_parity(
        "Diff",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_Digitize_parity():
    """Test parity for Digitize."""
    check_parity(
        "Digitize",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_Dirichlet_parity():
    """Test parity for Dirichlet."""
    check_parity(
        "Dirichlet",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_DrawBoundingBoxes_parity():
    """Test parity for DrawBoundingBoxes."""
    check_parity(
        "DrawBoundingBoxes",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_Dsplit_parity():
    """Test parity for Dsplit."""
    check_parity(
        "Dsplit",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_Dstack_parity():
    """Test parity for Dstack."""
    check_parity(
        "Dstack",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_DynamicIndexInDim_parity():
    """Test parity for DynamicIndexInDim."""
    check_parity(
        "DynamicIndexInDim",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_DynamicPartition_parity():
    """Test parity for DynamicPartition."""
    check_parity(
        "DynamicPartition",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_DynamicShape_parity():
    """Test parity for DynamicShape."""
    check_parity(
        "DynamicShape",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_DynamicSliceInDim_parity():
    """Test parity for DynamicSliceInDim."""
    check_parity(
        "DynamicSliceInDim",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_DynamicStitch_parity():
    """Test parity for DynamicStitch."""
    check_parity(
        "DynamicStitch",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_DynamicUpdateIndexInDim_parity():
    """Test parity for DynamicUpdateIndexInDim."""
    check_parity(
        "DynamicUpdateIndexInDim",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_DynamicUpdateSliceInDim_parity():
    """Test parity for DynamicUpdateSliceInDim."""
    check_parity(
        "DynamicUpdateSliceInDim",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_ExtractBoundingBoxes_parity():
    """Test parity for ExtractBoundingBoxes."""
    check_parity(
        "ExtractBoundingBoxes",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_ExtractPatches_parity():
    """Test parity for ExtractPatches."""
    check_parity(
        "ExtractPatches",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_ExtractVolumePatches_parity():
    """Test parity for ExtractVolumePatches."""
    check_parity(
        "ExtractVolumePatches",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_F_parity():
    """Test parity for F."""
    check_parity(
        "F",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_Flatten_parity():
    """Test parity for Flatten."""
    check_parity(
        "Flatten",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_FlipUpDown_parity():
    """Test parity for FlipUpDown."""
    check_parity(
        "FlipUpDown",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_Fold_parity():
    """Test parity for Fold."""
    check_parity(
        "Fold",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_Frame_parity():
    """Test parity for Frame."""
    check_parity(
        "Frame",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_Gather_parity():
    """Test parity for Gather."""
    check_parity(
        "Gather",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_GatherNd_parity():
    """Test parity for GatherNd."""
    check_parity(
        "GatherNd",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_GaussianBlur_parity():
    """Test parity for GaussianBlur."""
    check_parity(
        "GaussianBlur",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_Gumbel_parity():
    """Test parity for Gumbel."""
    check_parity(
        "Gumbel",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_HannWindow_parity():
    """Test parity for HannWindow."""
    check_parity(
        "HannWindow",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_Hanning_parity():
    """Test parity for Hanning."""
    check_parity(
        "Hanning",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_Hashing_parity():
    """Test parity for Hashing."""
    check_parity(
        "Hashing",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_Hessenberg_parity():
    """Test parity for Hessenberg."""
    check_parity(
        "Hessenberg",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_Hsplit_parity():
    """Test parity for Hsplit."""
    check_parity(
        "Hsplit",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_Hstack_parity():
    """Test parity for Hstack."""
    check_parity(
        "Hstack",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_Inner_parity():
    """Test parity for Inner."""
    check_parity(
        "Inner",
        lambda: [
            np.random.randn(2, 3).astype(np.float32),
            np.random.randn(2, 3).astype(np.float32),
        ],
    )


def test_IntegerLookup_parity():
    """Test parity for IntegerLookup."""
    check_parity(
        "IntegerLookup",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_IoU_parity():
    """Test parity for IoU."""
    check_parity(
        "IoU",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_Key_parity():
    """Test parity for Key."""
    check_parity(
        "Key",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_KeyData_parity():
    """Test parity for KeyData."""
    check_parity(
        "KeyData",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_KeyImpl_parity():
    """Test parity for KeyImpl."""
    check_parity(
        "KeyImpl",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_Laplace_parity():
    """Test parity for Laplace."""
    check_parity(
        "Laplace",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_Lookup_parity():
    """Test parity for Lookup."""
    check_parity(
        "Lookup",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_Lu_parity():
    """Test parity for Lu."""
    check_parity(
        "Lu",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_LuPivotsToPermutation_parity():
    """Test parity for LuPivotsToPermutation."""
    check_parity(
        "LuPivotsToPermutation",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_MedianFilter_parity():
    """Test parity for MedianFilter."""
    check_parity(
        "MedianFilter",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_MelFilterbank_parity():
    """Test parity for MelFilterbank."""
    check_parity(
        "MelFilterbank",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_MelSpectrogram_parity():
    """Test parity for MelSpectrogram."""
    check_parity(
        "MelSpectrogram",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_Mfcc_parity():
    """Test parity for Mfcc."""
    check_parity(
        "Mfcc",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_Mixup_parity():
    """Test parity for Mixup."""
    check_parity(
        "Mixup",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_Outer_parity():
    """Test parity for Outer."""
    check_parity(
        "Outer",
        lambda: [
            np.random.randn(3).astype(np.float32),
            np.random.randn(4).astype(np.float32),
        ],
    )


def test_Outfeed_parity():
    """Test parity for Outfeed."""
    check_parity(
        "Outfeed",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_Packbits_parity():
    """Test parity for Packbits."""
    check_parity(
        "Packbits",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_PadToBoundingBox_parity():
    """Test parity for PadToBoundingBox."""
    check_parity(
        "PadToBoundingBox",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_Pareto_parity():
    """Test parity for Pareto."""
    check_parity(
        "Pareto",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_Partition_parity():
    """Test parity for Partition."""
    import tests.test_ops_parity_array_ops as module

    orig = module.assert_allclose_mlx
    module.assert_allclose_mlx = lambda z, m, **k: orig(np.sort(z), np.sort(m), **k)
    try:
        check_parity(
            "Partition",
            lambda: [np.random.randn(5).astype(np.float32), 2],
        )
    finally:
        module.assert_allclose_mlx = orig


def test_Pbroadcast_parity():
    """Test parity for Pbroadcast."""
    check_parity(
        "Pbroadcast",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_Permute_parity():
    """Test parity for Permute."""
    check_parity(
        "Permute",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_Polar_parity():
    """Test parity for Polar."""
    check_parity(
        "Polar",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_Poly_parity():
    """Test parity for Poly."""
    check_parity(
        "Poly",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_Polyder_parity():
    """Test parity for Polyder."""
    check_parity(
        "Polyder",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_Polyfit_parity():
    """Test parity for Polyfit."""
    check_parity(
        "Polyfit",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_Polyint_parity():
    """Test parity for Polyint."""
    check_parity(
        "Polyint",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_Polyval_parity():
    """Test parity for Polyval."""
    check_parity(
        "Polyval",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_Posterize_parity():
    """Test parity for Posterize."""
    check_parity(
        "Posterize",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_Ppermute_parity():
    """Test parity for Ppermute."""
    check_parity(
        "Ppermute",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_Pshuffle_parity():
    """Test parity for Pshuffle."""
    check_parity(
        "Pshuffle",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_Pswapaxes_parity():
    """Test parity for Pswapaxes."""
    check_parity(
        "Pswapaxes",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_RaggedDynamicBroadcast_parity():
    """Test parity for RaggedDynamicBroadcast."""
    check_parity(
        "RaggedDynamicBroadcast",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_RaggedGather_parity():
    """Test parity for RaggedGather."""
    check_parity(
        "RaggedGather",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_RaggedRange_parity():
    """Test parity for RaggedRange."""
    check_parity(
        "RaggedRange",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_RaggedRowSplitsToSegmentIds_parity():
    """Test parity for RaggedRowSplitsToSegmentIds."""
    check_parity(
        "RaggedRowSplitsToSegmentIds",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_RaggedStack_parity():
    """Test parity for RaggedStack."""
    check_parity(
        "RaggedStack",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_RaggedStackDynamicPartitions_parity():
    """Test parity for RaggedStackDynamicPartitions."""
    check_parity(
        "RaggedStackDynamicPartitions",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_Range_parity():
    """Test parity for Range."""
    check_parity(
        "Range",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_RawMerge_parity():
    """Test parity for RawMerge."""
    check_parity(
        "RawMerge",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_RawSwitch_parity():
    """Test parity for RawSwitch."""
    check_parity(
        "RawSwitch",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_ReduceScatter_parity():
    """Test parity for ReduceScatter."""
    check_parity(
        "ReduceScatter",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_RegexReplace_parity():
    """Test parity for RegexReplace."""
    check_parity(
        "RegexReplace",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_Repeat_parity():
    """Test parity for Repeat."""
    check_parity(
        "Repeat",
        lambda: [np.random.randn(2, 2).astype(np.float32), 2],
    )


def test_ResizeBicubic_parity():
    """Test parity for ResizeBicubic."""
    check_parity(
        "ResizeBicubic",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_ResizeBilinear_parity():
    """Test parity for ResizeBilinear."""
    check_parity(
        "ResizeBilinear",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_ResizeLanczos3_parity():
    """Test parity for ResizeLanczos3."""
    check_parity(
        "ResizeLanczos3",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_ResizeLanczos5_parity():
    """Test parity for ResizeLanczos5."""
    check_parity(
        "ResizeLanczos5",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_ResizeNearest_parity():
    """Test parity for ResizeNearest."""
    check_parity(
        "ResizeNearest",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_RgbToGrayscale_parity():
    """Test parity for RgbToGrayscale."""
    check_parity(
        "RgbToGrayscale",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_RgbToHsv_parity():
    """Test parity for RgbToHsv."""
    check_parity(
        "RgbToHsv",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_RgbToYiq_parity():
    """Test parity for RgbToYiq."""
    check_parity(
        "RgbToYiq",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_RgbToYuv_parity():
    """Test parity for RgbToYuv."""
    check_parity(
        "RgbToYuv",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_Roll_parity():
    """Test parity for Roll."""
    check_parity(
        "Roll",
        lambda: [np.random.randn(2, 2).astype(np.float32), 1],
    )


def test_Roots_parity():
    """Test parity for Roots."""
    check_parity(
        "Roots",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_RowStack_parity():
    """Test parity for RowStack."""
    check_parity(
        "RowStack",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_Scan_parity():
    """Test parity for Scan."""
    check_parity(
        "Scan",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_ScanBind_parity():
    """Test parity for ScanBind."""
    check_parity(
        "ScanBind",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_Scatter_parity():
    """Test parity for Scatter."""
    check_parity(
        "Scatter",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_ScatterApply_parity():
    """Test parity for ScatterApply."""
    check_parity(
        "ScatterApply",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_ScatterNd_parity():
    """Test parity for ScatterNd."""
    check_parity(
        "ScatterNd",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_Schur_parity():
    """Test parity for Schur."""
    check_parity(
        "Schur",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_Select_parity():
    """Test parity for Select."""
    check_parity(
        "Select",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_Setdiff1d_parity():
    """Test parity for Setdiff1d."""
    check_parity(
        "Setdiff1d",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_Sharpen_parity():
    """Test parity for Sharpen."""
    check_parity(
        "Sharpen",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_Slice_parity():
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


def test_SliceInDim_parity():
    """Test parity for SliceInDim."""
    check_parity(
        "SliceInDim",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_SobolSample_parity():
    """Test parity for SobolSample."""
    check_parity(
        "SobolSample",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_Solarize_parity():
    """Test parity for Solarize."""
    check_parity(
        "Solarize",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_SparseMapValues_parity():
    """Test parity for SparseMapValues."""
    check_parity(
        "SparseMapValues",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_SparseMask_parity():
    """Test parity for SparseMask."""
    check_parity(
        "SparseMask",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_SparseResetShape_parity():
    """Test parity for SparseResetShape."""
    check_parity(
        "SparseResetShape",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_SparseReshape_parity():
    """Test parity for SparseReshape."""
    check_parity(
        "SparseReshape",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_SparseRetain_parity():
    """Test parity for SparseRetain."""
    check_parity(
        "SparseRetain",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_SparseSlice_parity():
    """Test parity for SparseSlice."""
    check_parity(
        "SparseSlice",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_SparseTranspose_parity():
    """Test parity for SparseTranspose."""
    check_parity(
        "SparseTranspose",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_Spence_parity():
    """Test parity for Spence."""
    check_parity(
        "Spence",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_Split_parity():
    """Test parity for Split."""
    check_parity(
        "Split",
        lambda: [np.random.randn(4).astype(np.float32), 2],
    )


def test_SquaredDifference_parity():
    """Test parity for SquaredDifference."""
    check_parity(
        "SquaredDifference",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_Squeeze_parity():
    """Test parity for Squeeze."""
    check_parity(
        "Squeeze",
        lambda: [np.random.randn(2, 1, 3).astype(np.float32)],
    )


def test_Stack_parity():
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


def test_Swapaxes_parity():
    """Test parity for Swapaxes."""
    check_parity(
        "Swapaxes",
        lambda: [np.random.randn(2, 3, 4).astype(np.float32), 0, 2],
    )


def test_Switch_parity():
    """Test parity for Switch."""
    check_parity(
        "Switch",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_T_parity():
    """Test parity for T."""
    check_parity(
        "T",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_Take_parity():
    """Test parity for Take."""
    check_parity(
        "Take",
        lambda: [np.random.randn(5).astype(np.float32), np.array([0, 2, 4])],
    )


def test_Tile_parity():
    """Test parity for Tile."""
    check_parity(
        "Tile",
        lambda: [np.random.randn(2, 2).astype(np.float32), (2, 2)],
    )


def test_TrapezoidalIntegral_parity():
    """Test parity for TrapezoidalIntegral."""
    check_parity(
        "TrapezoidalIntegral",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_Unfold_parity():
    """Test parity for Unfold."""
    check_parity(
        "Unfold",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_Union1d_parity():
    """Test parity for Union1d."""
    check_parity(
        "Union1d",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_UniqueValues_parity():
    """Test parity for UniqueValues."""
    check_parity(
        "UniqueValues",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_Unpackbits_parity():
    """Test parity for Unpackbits."""
    check_parity(
        "Unpackbits",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_UnravelIndex_parity():
    """Test parity for UnravelIndex."""
    check_parity(
        "UnravelIndex",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_Vsplit_parity():
    """Test parity for Vsplit."""
    check_parity(
        "Vsplit",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_Vstack_parity():
    """Test parity for Vstack."""
    check_parity(
        "Vstack",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_Wald_parity():
    """Test parity for Wald."""
    check_parity(
        "Wald",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_Where_parity():
    """Test parity for Where."""
    check_parity(
        "Where",
        lambda: [
            np.random.rand(2, 2) > 0.5,
            np.random.randn(2, 2).astype(np.float32),
            np.random.randn(2, 2).astype(np.float32),
        ],
    )


def test_WhileLoop_parity():
    """Test parity for WhileLoop."""
    check_parity(
        "WhileLoop",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_WithShardingConstraint_parity():
    """Test parity for WithShardingConstraint."""
    check_parity(
        "WithShardingConstraint",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_WrapKeyData_parity():
    """Test parity for WrapKeyData."""
    check_parity(
        "WrapKeyData",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_Zeta_parity():
    """Test parity for Zeta."""
    check_parity(
        "Zeta",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_BlockMaskedMm_parity():
    """Test parity for BlockMaskedMm."""
    check_parity("BlockMaskedMm", lambda: [np.random.randn(2, 3).astype(np.float32)])


def test_GatherMm_parity():
    """Test parity for GatherMm."""
    check_parity("GatherMm", lambda: [np.random.randn(2, 3).astype(np.float32)])


def test_SegmentedMm_parity():
    """Test parity for SegmentedMm."""
    check_parity("SegmentedMm", lambda: [np.random.randn(2, 3).astype(np.float32)])


def test_Logcumsumexp_parity():
    """Test parity for Logcumsumexp."""
    check_parity("Logcumsumexp", lambda: [np.random.randn(2, 3).astype(np.float32)])


def test_PutAlongAxis_parity():
    """Test parity for PutAlongAxis."""
    check_parity("PutAlongAxis", lambda: [np.random.randn(2, 3).astype(np.float32)])


def test_GetItem_parity():
    """Test parity for GetItem."""
    check_parity("GetItem", lambda: [np.random.randn(2, 3).astype(np.float32)])
