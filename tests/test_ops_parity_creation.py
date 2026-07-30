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


def test_Arange_parity():
    """Test parity for Arange."""
    check_parity("Arange", lambda: [10])


def test_Full_parity():
    """Test parity for Full."""
    check_parity("Full", lambda: [np.array([2, 3], dtype=np.int32), 3.14])


def test_Ones_parity():
    """Test parity for Ones."""
    check_parity("Ones", lambda: [(2, 3)])


def test_Zeros_parity():
    """Test parity for Zeros."""
    check_parity("Zeros", lambda: [(2, 3)])


def test_AffineGrid_parity():
    """Test parity for AffineGrid."""
    check_parity(
        "AffineGrid",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_Clone_parity():
    """Test parity for Clone."""
    check_parity(
        "Clone",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_Exponential_parity():
    """Test parity for Exponential."""
    check_parity(
        "Exponential",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_GridSample_parity():
    """Test parity for GridSample."""
    check_parity(
        "GridSample",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_Meshgrid_parity():
    """Test parity for Meshgrid."""
    check_parity(
        "Meshgrid",
        lambda: [
            np.array([1, 2, 3], dtype=np.float32),
            np.array([4, 5, 6], dtype=np.float32),
        ],
    )


def test_RegexFullMatch_parity():
    """Test parity for RegexFullMatch."""
    check_parity(
        "RegexFullMatch",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_SparseEye_parity():
    """Test parity for SparseEye."""
    check_parity(
        "SparseEye",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_SparseFillEmptyRows_parity():
    """Test parity for SparseFillEmptyRows."""
    check_parity(
        "SparseFillEmptyRows",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )


def test_ZeroFraction_parity():
    """Test parity for ZeroFraction."""
    check_parity(
        "ZeroFraction",
        lambda: [np.random.randn(2, 3).astype(np.float32)],
    )
