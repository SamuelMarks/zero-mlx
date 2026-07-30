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


def test_Fft_parity():
    """Test parity for Fft."""
    check_parity(
        "Fft",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
        kwargs_generator=lambda: {"n": 3},
    )


def test_Rfft_parity():
    """Test parity for Rfft."""
    check_parity("Rfft", lambda: [np.random.randn(2, 3).astype(np.float32)])


def test_Dct_parity():
    """Test parity for Dct."""
    check_parity(
        "Dct",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_Fft2d_parity():
    """Test parity for Fft2d."""
    check_parity(
        "Fft2d",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_Fft3d_parity():
    """Test parity for Fft3d."""
    check_parity(
        "Fft3d",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_Fftfreq_parity():
    """Test parity for Fftfreq."""
    check_parity(
        "Fftfreq",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_Fftnd_parity():
    """Test parity for Fftnd."""
    check_parity(
        "Fftnd",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_Fftshift_parity():
    """Test parity for Fftshift."""
    check_parity(
        "Fftshift",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_Hfft_parity():
    """Test parity for Hfft."""
    check_parity(
        "Hfft",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_Idct_parity():
    """Test parity for Idct."""
    check_parity(
        "Idct",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_Ifft_parity():
    """Test parity for Ifft."""
    check_parity(
        "Ifft",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_Ifft2d_parity():
    """Test parity for Ifft2d."""
    check_parity(
        "Ifft2d",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_Ifft3d_parity():
    """Test parity for Ifft3d."""
    check_parity(
        "Ifft3d",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_Ifftnd_parity():
    """Test parity for Ifftnd."""
    check_parity(
        "Ifftnd",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_Ifftshift_parity():
    """Test parity for Ifftshift."""
    check_parity(
        "Ifftshift",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_Ihfft_parity():
    """Test parity for Ihfft."""
    check_parity(
        "Ihfft",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_InverseMdct_parity():
    """Test parity for InverseMdct."""
    check_parity(
        "InverseMdct",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_Irfft_parity():
    """Test parity for Irfft."""
    check_parity(
        "Irfft",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_Irfft2d_parity():
    """Test parity for Irfft2d."""
    check_parity(
        "Irfft2d",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_Irfft3d_parity():
    """Test parity for Irfft3d."""
    check_parity(
        "Irfft3d",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_Irfftnd_parity():
    """Test parity for Irfftnd."""
    check_parity(
        "Irfftnd",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_Istft_parity():
    """Test parity for Istft."""
    check_parity(
        "Istft",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_Mdct_parity():
    """Test parity for Mdct."""
    check_parity(
        "Mdct",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_Rfft2d_parity():
    """Test parity for Rfft2d."""
    check_parity(
        "Rfft2d",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_Rfft3d_parity():
    """Test parity for Rfft3d."""
    check_parity(
        "Rfft3d",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_Rfftfreq_parity():
    """Test parity for Rfftfreq."""
    check_parity(
        "Rfftfreq",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_Rfftnd_parity():
    """Test parity for Rfftnd."""
    check_parity(
        "Rfftnd",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_Stft_parity():
    """Test parity for Stft."""
    check_parity(
        "Stft",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )
