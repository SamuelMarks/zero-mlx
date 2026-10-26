"""Comprehensive unit tests for `mlx_random`, `device`, `metal`, `dlpack`, `export`, `distributed`, `fast`, `utils`, `dtypes`, `info`, `array`, `ops`, `convolutions`, `linalg`, `fft`, and `nn`."""

from __future__ import annotations

import copy
import io
import pathlib
import pytest

import zero_mlx as mx
from zero_mlx._types import DLPackCapsule, NestedSequence, Scalar
from zero_mlx.array import array
from zero_mlx.array_like import ArrayLike
import zero_mlx.convolutions as z_conv
import zero_mlx.distributed as dist
import zero_mlx.dlpack as z_dlpack
import zero_mlx.dtypes as z_dt
import zero_mlx.export as z_export
import zero_mlx.fast as z_fast
import zero_mlx.fft as z_fft
import zero_mlx.info as z_info
import zero_mlx.linalg as z_linalg
import zero_mlx.metal as z_metal
import zero_mlx.mlx_random as z_rand
import zero_mlx.new_ops as z_new_ops
import zero_mlx.nn as z_nn
import zero_mlx.ops as z_ops
import zero_mlx.ops_patch as z_patch
import zero_mlx.random_state as z_rstate
import zero_mlx.utils as z_utils


def test_dtypes_and_info() -> None:
    """Test all `dtypes.py` and `info.py` methods, conversions, categories, and promotion rules."""
    for dt in z_dt.DType:
        assert dt.size in (1, 2, 4, 8, 16)
        assert repr(dt).startswith("mlx.core.")
        assert str(dt).startswith("mlx.core.")
        sw = z_dt.to_switcheroo_dtype(dt)
        assert z_dt.resolve_dtype(sw) == dt
        assert z_dt.to_switcheroo_dtype(sw) == sw
        assert z_dt.to_mlx_dtype(sw) == dt
        assert z_dt.to_mlx_dtype(dt) == dt
        assert z_dt.to_mlx_dtype(str(dt)) == dt

    assert z_dt.resolve_dtype(None, default=mx.int16) == mx.int16
    assert z_dt.resolve_dtype(bool) == mx.bool_
    assert z_dt.resolve_dtype(int) == mx.int32
    assert z_dt.resolve_dtype(float) == mx.float32
    assert z_dt.resolve_dtype(complex) == mx.complex64
    assert z_dt.resolve_dtype(z_dt.floating) == mx.float32
    assert z_dt.resolve_dtype(z_dt.complexfloating) == mx.complex64
    assert z_dt.resolve_dtype(z_dt.unsignedinteger) == mx.uint32
    assert z_dt.resolve_dtype(z_dt.signedinteger) == mx.int32
    assert z_dt.resolve_dtype("mlx.core.float32") == mx.float32
    assert z_dt.resolve_dtype("bool_") == mx.bool_

    assert z_dt.issubdtype(mx.float32, z_dt.floating) is True
    assert z_dt.issubdtype(mx.int32, int) is True
    assert z_dt.issubdtype(mx.bool_, bool) is True
    assert z_dt.issubdtype(mx.float32, float) is True
    assert z_dt.issubdtype(mx.complex64, complex) is True
    assert z_dt.issubdtype("float32", "floating") is True
    assert z_dt.issubdtype("invalid_dtype_xyz", "floating") is False
    assert z_dt.issubdtype(z_dt.to_switcheroo_dtype(mx.int32), z_dt.integer) is True
    assert z_dt.issubdtype(None, z_dt.number) is False

    assert z_dt.isdtype(mx.float32, "real floating") is True
    assert z_dt.isdtype(mx.int32, ("bool", "integral")) is True
    assert z_dt.isdtype(mx.bool_, "floating") is False

    assert z_dt.promote_dtypes(mx.float16, mx.bfloat16) == mx.float32
    assert z_dt.promote_dtypes(mx.int16, mx.uint8) == mx.int16
    assert z_dt.promote_dtypes(mx.int32, mx.uint16) == mx.int32
    assert z_dt.promote_dtypes(mx.int64, mx.uint32) == mx.int64
    assert z_dt.promote_dtypes(mx.int32, mx.uint64) == mx.float32
    assert z_dt.promote_dtypes(mx.uint8, mx.int16) == mx.int16

    for f_dt in (mx.float16, mx.bfloat16, mx.float32, mx.float64):
        fi = z_info.finfo(f_dt)
        assert fi.min < 0.0
        assert fi.max > 0.0
        assert fi.eps > 0.0
        assert fi.dtype == f_dt
        assert fi.bits in (16, 32, 64)

    with pytest.raises(ValueError):
        z_info.finfo(mx.int32)

    for i_dt in (
        mx.int8,
        mx.uint8,
        mx.int16,
        mx.uint16,
        mx.int32,
        mx.uint32,
        mx.int64,
        mx.uint64,
    ):
        ii = z_info.iinfo(i_dt)
        assert ii.max > ii.min
        assert ii.dtype == i_dt

    with pytest.raises(ValueError):
        z_info.iinfo(mx.float32)


def test_device_and_metal_and_dlpack() -> None:
    """Test all `device.py`, `metal.py`, and `dlpack.py` classes and functions."""
    assert str(mx.cpu) == "cpu"
    assert repr(mx.cpu) == "DeviceType.cpu"
    assert mx.cpu == mx.Device(mx.cpu)
    assert mx.cpu != "invalid"
    assert hash(mx.cpu) == hash("cpu")

    d_cpu = mx.Device("cpu", 0)
    d_copy = mx.Device(d_cpu)
    assert d_cpu == d_copy
    assert d_cpu == mx.cpu
    assert d_cpu != "not_a_device"
    assert hash(d_cpu) == hash(d_copy)
    assert repr(d_cpu) == "Device(cpu, 0)"
    with pytest.raises(ValueError):
        mx.Device("invalid_dev")
    with pytest.raises(ValueError):
        mx.Device(123)

    mx.set_default_device(mx.cpu)
    assert mx.default_device() == d_cpu
    s1 = mx.new_stream(mx.cpu)
    s2 = mx.new_thread_local_stream(mx.gpu)
    s3 = mx.new_thread_unsafe_stream("cpu")
    assert s1 == mx.Stream(mx.cpu, s1.index)
    assert s1 != "not_a_stream"
    assert "Stream(" in repr(s1)
    mx.set_default_stream(s1)
    assert mx.default_stream(mx.cpu) == s1
    with s2:
        assert mx.default_device() == mx.gpu
    with mx.stream(s1):
        assert mx.default_device() == mx.cpu
    with mx.stream(mx.gpu):
        assert mx.default_device() == mx.gpu
    mx.synchronize(s3)
    mx.clear_streams()
    assert mx.is_available(mx.cpu) is True
    assert mx.device_count(mx.cpu) == 1
    info = mx.device_info(mx.cpu)
    assert info["device_name"] == "zero_mlx_cpu"

    assert z_metal.is_available() is False
    assert z_metal.get_active_memory() == 0
    assert z_metal.get_peak_memory() == 0
    z_metal.reset_peak_memory()
    assert z_metal.get_cache_memory() == 0
    prev_cache = z_metal.set_cache_limit(1024)
    assert isinstance(prev_cache, int)
    prev_mem = z_metal.set_memory_limit(2048)
    assert isinstance(prev_mem, int)
    assert mx.get_memory_limit() == 2048
    prev_wired = z_metal.set_wired_limit(512)
    assert isinstance(prev_wired, int)
    z_metal.clear_cache()
    assert "architecture" in z_metal.device_info()
    z_metal.start_capture("trace.gputrace")
    z_metal.stop_capture()

    cap = z_dlpack._get_dummy_dlpack_capsule([1.0, 2.0], mx.float32)
    a_from_cap = mx.from_dlpack(cap)
    assert a_from_cap.tolist() == [1.0, 2.0]
    a_from_arr = mx.from_dlpack(a_from_cap)
    assert a_from_arr.tolist() == [1.0, 2.0]

    class CustomDLPack:
        """Custom DLPack convertible test class."""

        def __dlpack__(self, stream: int | None = None) -> DLPackCapsule:
            """Return capsule.

            Args:
                stream: Optional stream index.

            Returns:
                Wrapped `DLPackCapsule`.

            """
            _ = stream
            return DLPackCapsule([3.0, 4.0], mx.float32)

        def __dlpack_device__(self) -> tuple[int, int]:
            """Return device tuple."""
            return (1, 0)

    a_custom = mx.from_dlpack(CustomDLPack())
    assert a_custom.tolist() == [3.0, 4.0]
    assert array(CustomDLPack()).tolist() == [3.0, 4.0]
    with pytest.raises(TypeError):
        mx.from_dlpack(123)


def test_array_repr_and_at_and_iterator() -> None:
    """Test `array_repr.py`, `at_mocker.py`, `array_iterator.py`, and `array.py` methods."""
    mx.set_printoptions(precision=3, threshold=500, edgeitems=2, linewidth=80, suppress=True)
    opts = mx.get_printoptions()
    assert opts["precision"] == 3
    with mx.printoptions(precision=4):
        assert mx.get_printoptions()["precision"] == 4
    assert mx.get_printoptions()["precision"] == 3
    mx.set_printoptions(precision=5)

    assert "True" in repr(array(True))
    assert "42" in repr(array(42))
    assert "nan" in repr(array(float("nan")))
    assert "inf" in repr(array(float("inf")))
    assert "-inf" in repr(array(float("-inf")))
    assert "2" in repr(array(2.0))
    assert "1.25" in repr(array(1.25))
    with mx.printoptions(precision=2):
        assert "1.25" in repr(array(1.25))
    assert "j" in repr(array(1.0 + 2.0j))
    assert "[]" in repr(array([]))
    assert "..." in repr(mx.arange(10))
    assert str(mx.ones((2, 2))).startswith("array(")
    from zero_mlx.array_repr import _format_list, _format_scalar

    assert _format_list(1.5, ()) == "1.5"
    assert _format_scalar(1.5 - 2.5j) == "1.5-2.5j"

    arr = array([10.0, 20.0, 30.0])
    it = iter(arr)
    assert iter(it) is it
    assert [float(x.item().real) for x in it] == [10.0, 20.0, 30.0]

    assert arr.at[1].set(99.0).tolist() == [10.0, 99.0, 30.0]
    assert arr.at[0].add(5.0).tolist() == [15.0, 20.0, 30.0]
    assert arr.at[0].subtract(5.0).tolist() == [5.0, 20.0, 30.0]
    assert arr.at[0].multiply(2.0).tolist() == [20.0, 20.0, 30.0]
    assert arr.at[0].divide(2.0).tolist() == [5.0, 20.0, 30.0]
    assert arr.at[0].power(2.0).tolist() == [100.0, 20.0, 30.0]
    assert arr.at[0].maximum(50.0).tolist() == [50.0, 20.0, 30.0]
    assert arr.at[0].minimum(5.0).tolist() == [5.0, 20.0, 30.0]
    assert arr.at[0:2].add(array([1.0, 2.0])).tolist() == [11.0, 22.0, 30.0]
    assert arr.at[[0, 2]].add(1.0).tolist() == [11.0, 20.0, 31.0]
    assert arr.at[array([0, 1])].add(array([5.0])).tolist() == [15.0, 25.0, 30.0]
    assert arr.at[array(0)].add(1.0).tolist() == [11.0, 20.0, 30.0]

    int_arr = array([2, 3, 4], dtype=mx.int32)
    assert int_arr.at[0].add(1).tolist() == [3, 3, 4]
    assert int_arr.at[0].subtract(1).tolist() == [1, 3, 4]
    assert int_arr.at[0].multiply(2).tolist() == [4, 3, 4]
    assert int_arr.at[0].power(2).tolist() == [4, 3, 4]

    c_arr = array([1.0 + 1.0j, 2.0 + 0.0j], dtype=mx.complex64)
    assert len(c_arr.at[0].add(1.0 + 2.0j)) == 2
    assert len(c_arr.at[0].subtract(1.0 + 2.0j)) == 2
    assert len(c_arr.at[0].multiply(2.0 + 0.0j)) == 2
    assert len(c_arr.at[0].divide(2.0 + 0.0j)) == 2
    assert len(c_arr.at[0].power(2.0 + 0.0j)) == 2

    arr2d = array([[1.0, 2.0], [3.0, 4.0]])
    assert arr2d.at[0, 1].set(9.0).tolist() == [[1.0, 9.0], [3.0, 4.0]]
    assert arr2d.at[0, 1].add(1.0).tolist() == [[1.0, 3.0], [3.0, 4.0]]
    assert arr2d.at[0, 1].subtract(1.0).tolist() == [[1.0, 1.0], [3.0, 4.0]]
    assert arr2d.at[0, 1].multiply(2.0).tolist() == [[1.0, 4.0], [3.0, 4.0]]
    assert arr2d.at[0, 1].divide(2.0).tolist() == [[1.0, 1.0], [3.0, 4.0]]
    assert arr2d.at[0, 1].power(2.0).tolist() == [[1.0, 4.0], [3.0, 4.0]]
    assert arr2d.at[0, 1].maximum(5.0).tolist() == [[1.0, 5.0], [3.0, 4.0]]
    assert arr2d.at[0, 1].minimum(0.0).tolist() == [[1.0, 0.0], [3.0, 4.0]]

    with pytest.raises(ValueError):
        arr.at.add(1.0)


def test_array_operators_and_methods() -> None:
    """Test all `array.py` operators, properties, indexing, and delegating methods."""

    class CustomMlxArray:
        """Custom MlxArrayConvertible object."""

        def __mlx_array__(self) -> array:
            """Convert to array."""
            return array([1.0, 2.0])

    class _DummyExternalArray:
        """Custom `ArrayProtocol` implementation for testing external array conversion."""

        def __init__(self, data: list[Scalar | NestedSequence], dtype: str) -> None:
            """Initialize dummy external array.

            Args:
                data: Underlying list of scalars.
                dtype: String dtype name.

            """
            self._data: list[Scalar | NestedSequence] = data
            self._dtype: str = dtype

        @property
        def dtype(self) -> str:
            """Return string dtype."""
            return self._dtype

        def tolist(self) -> list[Scalar | NestedSequence]:
            """Return list representation."""
            return self._data

    a_conv = array(CustomMlxArray())
    assert a_conv.tolist() == [1.0, 2.0]
    assert (a_conv + CustomMlxArray()).tolist() == [2.0, 4.0]
    assert mx.maximum(a_conv, [0.5, 3.0]).tolist() == [1.0, 3.0]
    assert (array([True, False]) + 1).tolist() == [2, 1]
    assert array(_DummyExternalArray([1.0, 2.0], "float64")).dtype == mx.float32
    assert array(_DummyExternalArray([1.0 + 2.0j], "complex128")).dtype == mx.complex64
    assert array(_DummyExternalArray([1, 2], "int16")).dtype == mx.int16
    assert array([[1.0]]).item() == 1.0
    a_bytes = array(b"\x01\x02")
    assert a_bytes.tolist() == [1, 2]
    a_big_int = array(3000000000)
    assert a_big_int.dtype == mx.int64
    a_nested_arr = array([array([1.0, 2.0]), _DummyExternalArray([3.0, 4.0], "float32")])
    assert a_nested_arr.shape == (2, 2)

    with pytest.raises(ValueError):
        array("bad_str")
    with pytest.raises(ValueError):
        array([[1, 2], 3])
    with pytest.raises(ValueError):
        array([[1, 2], [3]])

    x = array([2.0, 4.0])
    y = array([1.0, 2.0])
    assert (x + y).tolist() == [3.0, 6.0]
    assert (1.0 + x).tolist() == [3.0, 5.0]
    x_i = x.copy()
    x_i += 1.0
    assert x_i.tolist() == [3.0, 5.0]
    assert (x - y).tolist() == [1.0, 2.0]
    assert (10.0 - x).tolist() == [8.0, 6.0]
    x_i -= 1.0
    assert x_i.tolist() == [2.0, 4.0]
    assert (x * y).tolist() == [2.0, 8.0]
    assert (2.0 * x).tolist() == [4.0, 8.0]
    x_i *= 2.0
    assert x_i.tolist() == [4.0, 8.0]
    assert (x / y).tolist() == [2.0, 2.0]
    assert (4.0 / x).tolist() == [2.0, 1.0]
    x_i /= 2.0
    assert x_i.tolist() == [2.0, 4.0]
    assert (x // y).tolist() == [2.0, 2.0]
    assert (5.0 // x).tolist() == [2.0, 1.0]
    x_i //= 2.0
    assert x_i.tolist() == [1.0, 2.0]
    assert (x % y).tolist() == [0.0, 0.0]
    assert (5.0 % x).tolist() == [1.0, 1.0]
    x_i %= 2.0
    assert x_i.tolist() == [1.0, 0.0]
    assert (x**2.0).tolist() == [4.0, 16.0]
    assert (2.0**y).tolist() == [2.0, 4.0]
    y_i = y.copy()
    y_i **= 2.0
    assert y_i.tolist() == [1.0, 4.0]

    b1 = array([1, 3], dtype=mx.int32)
    b2 = array([3, 1], dtype=mx.int32)
    assert (b1 & b2).tolist() == [1, 1]
    assert (3 & b1).tolist() == [1, 3]
    b_i = b1.copy()
    b_i &= b2
    assert b_i.tolist() == [1, 1]
    assert (b1 | b2).tolist() == [3, 3]
    assert (2 | b1).tolist() == [3, 3]
    b_i |= b2
    assert b_i.tolist() == [3, 1]
    assert (b1 ^ b2).tolist() == [2, 2]
    assert (1 ^ b1).tolist() == [0, 2]
    b_i ^= b2
    assert b_i.tolist() == [0, 0]
    assert (b1 << 1).tolist() == [2, 6]
    assert (b1 >> 1).tolist() == [0, 1]
    assert (~b1).tolist() == [-2, -4]

    m1 = mx.eye(2)
    m2 = mx.ones((2, 2))
    assert (m1 @ m2).tolist() == [[1.0, 1.0], [1.0, 1.0]]
    m_i = m1.copy()
    m_i @= m2
    assert m_i.tolist() == [[1.0, 1.0], [1.0, 1.0]]

    assert (x < y).tolist() == [False, False]
    assert (x <= y).tolist() == [False, False]
    assert (x > y).tolist() == [True, True]
    assert (x >= y).tolist() == [True, True]
    assert (x == x).tolist() == [True, True]
    assert (x == [2.0, 4.0]).item() is False
    assert (x == "other").item() is False
    assert (x != y).tolist() == [True, True]
    assert (x != [2.0, 4.0]).item() is True
    assert (x != "other").item() is True
    assert (-x).tolist() == [-2.0, -4.0]
    assert (+x).tolist() == [2.0, 4.0]
    assert abs(-x).tolist() == [2.0, 4.0]

    with pytest.raises(ValueError):
        _ = x + "bad"
    with pytest.raises(ValueError):
        _ = x + [1.0, 2.0]
    with pytest.raises(ValueError):
        _ = b1 + 3000000000

    sc = array(3.5)
    assert f"{sc:.1f}" == "3.5"
    assert f"{sc}" == repr(sc)
    with pytest.raises(TypeError):
        _ = f"{x:.1f}"
    assert int(sc) == 3
    assert float(sc) == 3.5
    assert bool(sc) is True
    with pytest.raises(ValueError):
        bool(x)
    with pytest.raises(ValueError):
        x.item()
    with pytest.raises(TypeError):
        len(sc)
    assert len(x) == 2

    assert sc.__dlpack_device__() == (1, 0)
    assert isinstance(sc.__dlpack__(), DLPackCapsule)
    assert sc.__array_namespace__() is mx
    assert isinstance(sc.data, memoryview)
    assert sc.__array__() == 3.5
    assert sc.__array__(dtype=mx.int32) == 3
    for dt_test in (
        mx.bool_,
        mx.int8,
        mx.uint8,
        mx.int16,
        mx.uint16,
        mx.int32,
        mx.uint32,
        mx.int64,
        mx.uint64,
        mx.float16,
        mx.bfloat16,
        mx.float64,
        mx.complex64,
        mx.complex128,
    ):
        assert isinstance(array([1], dtype=dt_test).data, memoryview)
    assert copy.copy(x).tolist() == [2.0, 4.0]
    assert copy.deepcopy(x).tolist() == [2.0, 4.0]

    with pytest.raises(ValueError):
        _ = x[2**31]
    with pytest.raises(ValueError):
        _ = x[0 : 2**31]
    with pytest.raises(ValueError):
        x[True] = 1.0
    with pytest.raises(ValueError):
        sc[0] = 1.0
    sc[None] = 9.0
    assert sc.item() == 9.0
    x_set = array([1.0, 2.0, 3.0])
    x_set[array([0, 2])] = array([7.0, 8.0])
    assert x_set.tolist() == [7.0, 2.0, 8.0]
    x_set[[0, 1]] = array([5.0, 6.0])
    assert x_set.tolist() == [5.0, 6.0, 8.0]
    assert x_set[[0, 2]].tolist() == [5.0, 8.0]

    # Test all array delegating methods
    a_m = array([[1.0, 4.0], [9.0, 16.0]])
    assert a_m.itemsize == 4
    assert a_m.nbytes == 16
    assert a_m.T.shape == (2, 2)
    assert a_m.real.shape == (2, 2)
    assert a_m.imag.tolist() == [[0.0, 0.0], [0.0, 0.0]]
    assert array([1.0 + 2.0j]).imag.tolist() == [2.0]
    assert a_m.view(mx.float32).shape == (2, 2)
    assert a_m.abs().shape == (2, 2)
    assert bool(a_m.all().item()) is True
    assert bool(a_m.any().item()) is True
    assert int(a_m.argmax().item().real) == 3
    assert int(a_m.argmin().item().real) == 0
    assert a_m.conj().shape == (2, 2)
    assert a_m.cummax(axis=0).shape == (2, 2)
    assert a_m.cummin(axis=0).shape == (2, 2)
    assert a_m.cumprod(axis=0).shape == (2, 2)
    assert a_m.cumsum(axis=0).shape == (2, 2)
    assert a_m.diagonal().tolist() == [1.0, 16.0]
    assert a_m.exp().shape == (2, 2)
    assert a_m.flatten().shape == (4,)
    assert a_m.log().shape == (2, 2)
    assert a_m.log10().shape == (2, 2)
    assert a_m.log1p().shape == (2, 2)
    assert a_m.log2().shape == (2, 2)
    assert a_m.logcumsumexp(axis=0).shape == (2, 2)
    assert a_m.logsumexp().ndim == 0
    assert float(a_m.max().item().real) == 16.0
    assert float(a_m.mean().item().real) == 7.5
    assert float(a_m.min().item().real) == 1.0
    assert a_m.moveaxis(0, 1).shape == (2, 2)
    assert float(a_m.prod().item().real) == 576.0
    assert a_m.reciprocal().shape == (2, 2)
    assert a_m.round().shape == (2, 2)
    assert a_m.sin().shape == (2, 2)
    assert a_m.cos().shape == (2, 2)
    assert a_m.sqrt().tolist() == [[1.0, 2.0], [3.0, 4.0]]
    assert a_m.rsqrt().shape == (2, 2)
    assert a_m.square().shape == (2, 2)
    assert a_m.std().ndim == 0
    assert float(a_m.sum().item().real) == 30.0
    assert a_m.swapaxes(0, 1).shape == (2, 2)
    assert a_m.transpose(1, 0).shape == (2, 2)
    assert a_m.transpose([1, 0]).shape == (2, 2)
    assert a_m.var().ndim == 0
    assert len(a_m.split(2)) == 2
    assert array([[[1.0]]]).squeeze().ndim == 0
    assert array([[[1.0]]]).squeeze(axis=0).shape == (1, 1)
    assert array([[[1.0]]]).squeeze(axis=(0, 1)).shape == (1,)


def test_random_and_distributed_and_fast_and_export(tmp_path: pathlib.Path) -> None:
    """Test `mlx_random`, `random_state`, `distributed`, `fast`, `export`, `utils`, and `new_ops`.

    Args:
        tmp_path: Pytest temporary directory fixture.

    """
    z_rand.seed(123)
    st = z_rstate.state()
    assert st.shape == (2,)
    assert z_rstate.state(array([1, 2], dtype=mx.uint32)).tolist() == [1, 2]
    pk = z_rand.PRNGKey(42)
    assert z_rand.split(pk, num=3).shape == (3, 2)
    assert z_rand.split(z_rand.key(7), num=2).shape == (2, 2)
    assert z_rand.uniform(0.0, 1.0, shape=None, key=pk).shape == ()
    assert z_rand.uniform(0.0, 1.0, shape=3, key=10).shape == (3,)
    assert z_rand.normal((2, 2), loc=1.0, scale=2.0).shape == (2, 2)
    assert z_rand.randint(0, 10, shape=(2, 3)).shape == (2, 3)
    assert z_rand.bernoulli(0.5, shape=(2, 2)).shape == (2, 2)
    assert z_rand.bernoulli(array([0.2, 0.8])).shape == (2,)
    with pytest.raises(ValueError):
        z_rand.bernoulli(1)
    assert z_rand.categorical(array([[1.0, 2.0, 3.0]])).shape == (1,)
    assert z_rand.categorical(array(1.0)).shape == ()
    assert z_rand.categorical(array([[1.0, 2.0, 3.0]]), num_samples=4).shape == (1, 4)
    assert z_rand.categorical(array([1.0, 2.0]), shape=(2, 2)).shape == (2, 2)
    with pytest.raises(ValueError):
        z_rand.categorical(array([1.0, 2.0]), shape=(2,), num_samples=2)
    assert z_rand.gumbel((2, 2)).shape == (2, 2)
    assert z_rand.laplace((2, 2), loc=1.0, scale=2.0).shape == (2, 2)
    assert z_rand.multivariate_normal(array([0.0, 0.0]), mx.eye(2), shape=(3,)).shape == (3, 2)
    with pytest.raises(ValueError):
        z_rand.multivariate_normal(array([0.0, 0.0]), mx.eye(2), dtype=mx.int32)
    assert z_rand.permutation(5).shape == (5,)
    assert z_rand.permutation(mx.ones((3, 4)), axis=1).shape == (3, 4)
    assert z_rand.truncated_normal(-1.0, 1.0, shape=(2, 3)).shape == (2, 3)
    assert z_rand.truncated_normal(array([1.0, -1.0]), array([-1.0, 1.0])).shape == (2,)

    g = dist.init()
    assert g.rank() == 0
    assert g.size() == 1
    assert g.split(0).size() == 1
    assert dist.is_available() is False
    x_d = array([1.0, 2.0])
    assert dist.all_gather(x_d).shape[0] >= 2
    assert dist.all_max(x_d).tolist() == [1.0, 2.0]
    assert dist.all_min(x_d).tolist() == [1.0, 2.0]
    assert dist.all_sum(x_d).tolist() == [1.0, 2.0]
    assert dist.sum_scatter(x_d).tolist() == [1.0, 2.0]
    assert dist.send(x_d, 0).tolist() == [1.0, 2.0]
    assert dist.recv((2,), mx.float32, 0).shape == (2,)
    assert dist.recv(2, mx.float32, 0).shape == (2,)
    assert dist.recv_like(x_d, 0).shape == (2,)

    x_f = mx.ones((2, 4))
    w_f = mx.ones((4,))
    b_f = mx.zeros((4,))
    assert z_fast.layer_norm(x_f, w_f, b_f).shape == (2, 4)
    assert z_fast.rms_norm(x_f, w_f).shape == (2, 4)
    assert z_fast.rms_norm(array(1.0), None).item() == 0.0
    assert z_fast.rope(mx.ones((1, 2, 4, 8)), dims=8, offset=array(1)).shape == (1, 2, 4, 8)
    assert z_fast.rope(
        mx.ones((1, 2, 4, 8)), dims=6, traditional=True, freqs=mx.ones((3,))
    ).shape == (1, 2, 4, 8)
    assert z_fast.rope(
        mx.ones((1, 2, 4, 8)), dims=8, traditional=False, freqs=mx.ones((4,))
    ).shape == (1, 2, 4, 8)
    q = mx.ones((1, 2, 4, 8))
    assert z_fast.scaled_dot_product_attention(q, q, q).shape == (1, 2, 4, 8)
    assert z_fast.scaled_dot_product_attention(
        q, q, q, scale=0.5, mask=mx.zeros((1, 2, 4, 4))
    ).shape == (1, 2, 4, 8)
    assert z_fast.scaled_dot_product_attention(
        q, q, q, scale=0.5, mask=mx.ones((1, 2, 4, 4), dtype=mx.bool_)
    ).shape == (1, 2, 4, 8)
    assert z_fast.scaled_dot_product_attention(
        q, q, q, scale=0.5, mask="causal", sinks=mx.zeros((1, 2))
    ).shape == (1, 2, 4, 8)
    assert z_fast.scaled_dot_product_attention(array(1.0), array(1.0), array(1.0)).item() == 0.0
    mk = z_fast.metal_kernel("k", ["a"], ["b"], "source")
    assert mk(inputs=[x_f], output_shapes=[(2, 4)], output_dtypes=[mx.float32])[0].shape == (2, 4)
    ck = z_fast.cuda_kernel("k", ["a"], ["b"], "source")
    assert callable(ck)
    pck = z_fast.precompiled_cuda_kernel("k", b"ptx", [x_f], [(2, 4)], ["float32"], [1])
    assert isinstance(pck, list)

    fn_path = str(tmp_path / "func.mlxfn")

    def sample_fn(*args: array, **kw: array) -> array:
        """Sample exportable function.

        Args:
            *args: Positional array arguments.
            **kw: Keyword array arguments.

        Returns:
            Sum `array`.

        """
        a = args[0]
        return a + kw.get("b", array(1.0))

    z_export.export_function(
        fn_path, sample_fn, array([1.0, 2.0]), metadata={"v": "1"}, b=array([3.0, 4.0])
    )
    imp_res = z_export.import_function(fn_path, return_metadata=True)
    assert isinstance(imp_res, tuple)
    imp_fn, meta = imp_res
    assert meta == {"v": "1"}
    out_tup = imp_fn(array([1.0, 2.0]), b=array([3.0, 4.0]))
    assert out_tup[0].tolist() == [4.0, 6.0]

    def tuple_fn(*args: array, **kw: array) -> tuple[array, ...]:
        """Tuple returning exportable function.

        Args:
            *args: Positional array arguments.
            **kw: Keyword array arguments.

        Returns:
            Tuple of output arrays.

        """
        _ = kw
        return (args[0] * 2.0, args[0] + 1.0)

    fn_path2 = str(tmp_path / "func2.mlxfn")
    with z_export.exporter(fn_path2, tuple_fn) as exp:
        exp(array([2.0]))
    with pytest.raises(RuntimeError):
        exp(array([2.0]))
    imp2 = z_export.import_function(fn_path2)
    assert not isinstance(imp2, tuple)
    res_imp2 = imp2(array([5.0]))
    assert len(res_imp2) == 2
    assert res_imp2[0].tolist() == [10.0] and res_imp2[1].tolist() == [6.0]
    with pytest.raises(ValueError):
        imp2(array([1.0, 2.0]))

    unreg_path = str(tmp_path / "unreg.mlxfn")
    with z_export.exporter(unreg_path, None) as exp_none:
        exp_none(array([1.0]))
    pathlib.Path(unreg_path).write_text(
        '{"shapeless": false, "metadata": {"k": "v"}}', encoding="utf-8"
    )
    imp_unreg = z_export.import_function(unreg_path)
    assert not isinstance(imp_unreg, tuple)
    assert len(imp_unreg(array([5.0]))) == 1
    with pytest.raises(FileNotFoundError):
        z_export.import_function(str(tmp_path / "missing.mlxfn"))

    cb_bytes: list[bytes] = []
    z_export.export_function(cb_bytes.append, sample_fn, array([1.0]))
    assert len(cb_bytes) == 1

    dot_path = str(tmp_path / "graph.dot")
    z_export.export_to_dot(dot_path, array([1.0]), named=array([2.0]))
    assert "digraph G" in pathlib.Path(dot_path).read_text(encoding="utf-8")
    buf = io.StringIO()
    z_export.export_to_dot(buf, array([1.0]))
    assert "digraph G" in buf.getvalue()

    assert z_utils.get_array_buffer_size({"a": array([1.0, 2.0]), "b": [array([3.0])]}) == 12
    mapped_path = z_utils.tree_map_with_path(lambda p, v: f"{p}:{v}", {"a": 1, "b": [2, (3,)]})
    assert mapped_path == {"a": "a:1", "b": ["b.0:2", ("b.1.0:3",)]}
    assert z_utils.tree_flatten({"a": [1, 2]}, is_leaf=lambda x: isinstance(x, list)) == [
        ("a", [1, 2])
    ]

    z_patch.patch_ops()
    z_new_ops.async_eval(array([1.0]))
    z_new_ops.disable_compile()
    z_new_ops.enable_compile()
    c_fn = z_new_ops.compile(sample_fn)
    c_res = c_fn(array([1.0]))
    assert isinstance(c_res, array) and c_res.tolist() == [2.0]

    def sq_fn(*args: array, **kwargs: array) -> array:
        """Square sum function.

        Args:
            *args: Positional array arguments.
            **kwargs: Keyword array arguments.

        Returns:
            Scalar objective `array`.

        """
        x_in = args[0]
        y_in = kwargs.get("y", array(0.0))
        return mx.sum(x_in * x_in) + mx.sum(y_in * 3.0)

    g_fn = mx.grad(sq_fn)
    g_val = g_fn(array([2.0, 3.0]))
    assert isinstance(g_val, array)
    assert bool(mx.allclose(g_val, array([4.0, 6.0]), rtol=1e-2, atol=1e-2).item()) is True

    def _sq_sc(*args: array, **kwargs: array) -> array:
        """Scalar square function.

        Args:
            *args: Positional array arguments.
            **kwargs: Keyword array arguments.

        Returns:
            Squared scalar `array`.

        """
        _ = kwargs
        return args[0] * args[0]

    g_sc = mx.grad(_sq_sc)(array(3.0))
    assert isinstance(g_sc, array)
    assert float(g_sc.item().real) == pytest.approx(6.0, rel=1e-2)

    def _prod_fn(*args: array, **kwargs: array) -> array:
        """Product sum function.

        Args:
            *args: Positional array arguments.
            **kwargs: Keyword array arguments.

        Returns:
            Sum of elementwise product `array`.

        """
        _ = kwargs
        return mx.sum(args[0] * args[1])

    _, g_multi = mx.value_and_grad(_prod_fn, argnums=(0, 1))(array([2.0]), array([5.0]))
    assert isinstance(g_multi, tuple) and len(g_multi) == 2
    _, g_kw = mx.value_and_grad(sq_fn, argnames="y")(array([1.0]), y=array([2.0]))
    assert isinstance(g_kw, dict) and "y" in g_kw
    _, g_both = mx.value_and_grad(sq_fn, argnums=0, argnames="y")(array([1.0]), y=array([2.0]))
    assert isinstance(g_both, tuple) and len(g_both) == 2
    with pytest.raises(ValueError):
        mx.grad(lambda *a, **kw: a[0])(array([1.0, 2.0]))

    def _vjp_fn(*primals: array) -> list[array]:
        """Two-output primal function.

        Args:
            *primals: Primal array inputs.

        Returns:
            List of two scaled arrays.

        """
        return [primals[0] * 2.0, primals[0] * 3.0]

    def _jvp_fn(*primals: array) -> array:
        """Single-output primal function.

        Args:
            *primals: Primal array inputs.

        Returns:
            Scaled `array`.

        """
        return primals[0] * 2.0

    outs_vjp, grads_vjp = mx.vjp(
        _vjp_fn, [array([1.0, 2.0])], [array([1.0, 1.0]), array([1.0, 1.0])]
    )
    assert len(outs_vjp) == 2 and len(grads_vjp) == 1
    outs_jvp, tang_jvp = mx.jvp(_jvp_fn, [array([1.0, 2.0])], [array([1.0, 1.0])])
    assert len(outs_jvp) == 1 and len(tang_jvp) == 1

    def _add_vmap(*args: array, **kwargs: array) -> array:
        """Add two arrays for vmap.

        Args:
            *args: Positional array arguments.
            **kwargs: Keyword array arguments.

        Returns:
            Elementwise sum `array`.

        """
        _ = kwargs
        return args[0] + args[1]

    def _mul2_vmap(*args: array, **kwargs: array) -> array:
        """Multiply array by 2 for vmap.

        Args:
            *args: Positional array arguments.
            **kwargs: Keyword array arguments.

        Returns:
            Doubled `array`.

        """
        _ = kwargs
        return args[0] * 2.0

    vm_fn = mx.vmap(_add_vmap, in_axes=(0, None))
    assert vm_fn(mx.ones((3, 2)), array([1.0, 2.0])).shape == (3, 2)
    vm_fn2 = mx.vmap(_mul2_vmap, in_axes=1)
    assert vm_fn2(mx.ones((2, 3))).shape == (3, 2)
    vm_none = mx.vmap(_mul2_vmap, in_axes=0, out_axes=None)
    assert isinstance(vm_none(mx.ones((2, 3))), array)
    vm_seq = mx.vmap(_mul2_vmap, in_axes=0, out_axes=(0,))
    assert isinstance(vm_seq(mx.ones((2, 3))), array)
    with pytest.raises(ValueError):
        mx.vmap(_add_vmap, in_axes=(0, 0))(mx.ones((2, 2)), mx.ones((3, 2)))

    comp_fn = mx.compile(_mul2_vmap)
    assert isinstance(comp_fn(array([1.0, 2.0])), array)
    mx.disable_compile()
    assert isinstance(comp_fn(array([1.0, 2.0])), array)
    mx.enable_compile()

    npy_file = tmp_path / "arr.npy"
    mx.save(npy_file, array([1.0, 2.0]))
    mx.save(tmp_path / "scalar.npy", array(3.0))
    loaded_npy = mx.load(npy_file)
    assert isinstance(loaded_npy, array) and loaded_npy.tolist() == [1.0, 2.0]
    loaded_disk = mx.load(npy_file, format="npy")
    assert isinstance(loaded_disk, array) and loaded_disk.tolist() == [1.0, 2.0]
    loaded_disk_meta = mx.load(tmp_path / "scalar.npy", return_metadata=True)
    assert isinstance(loaded_disk_meta, tuple)
    with pytest.raises(FileNotFoundError):
        mx.load(tmp_path / "missing_file.npy")
    with pytest.raises(ValueError):
        mx.load(npy_file, format="invalid_fmt")

    # Test binary stream save/load and complex/int64/bool dtypes
    stream_buf = io.BytesIO()
    mx.save(stream_buf, array([True, False], dtype=mx.bool_))
    stream_buf.seek(0)
    s_loaded = mx.load(stream_buf)
    assert isinstance(s_loaded, array) and s_loaded.tolist() == [True, False]
    stream_buf_meta = io.BytesIO(stream_buf.getvalue())
    assert isinstance(mx.load(stream_buf_meta, return_metadata=True), tuple)

    c_npy = tmp_path / "complex.npy"
    mx.save(c_npy, array([1.0 + 2.0j], dtype=mx.complex64))
    c_loaded = mx.load(c_npy)
    assert isinstance(c_loaded, array) and c_loaded.tolist() == [1.0 + 2.0j]

    i64_npy = tmp_path / "int64.npy"
    mx.save(i64_npy, array([10000000000], dtype=mx.int64))
    i64_loaded = mx.load(i64_npy)
    assert isinstance(i64_loaded, array) and i64_loaded.tolist() == [10000000000]

    npz_file = tmp_path / "bundle"
    mx.savez(npz_file, array([1.0]), named=array([[2.0, 3.0]]))
    load_res = mx.load(npz_file, return_metadata=True)
    assert isinstance(load_res, tuple)
    loaded_npz, meta_npz = load_res
    assert isinstance(loaded_npz, dict) and isinstance(meta_npz, dict)
    loaded_npz_disk = mx.load(npz_file, return_metadata=True)
    assert isinstance(loaded_npz_disk, tuple) and "named" in loaded_npz_disk[0]
    mx.savez_compressed(npz_file, array([1.0]), named=array([2.0]))
    assert isinstance(mx.load(npz_file), dict)

    st_file = tmp_path / "weights"
    mx.save_safetensors(st_file, {"w": array([1.0, 2.0])}, metadata={"f": "mlx"})
    st_loaded, st_meta = mx.load(st_file, return_metadata=True)
    assert isinstance(st_loaded, dict) and st_meta == {"f": "mlx"}
    assert isinstance(mx.load(st_file, return_metadata=False), dict)

    gguf_file = tmp_path / "weights_gguf"
    mx.save_gguf(gguf_file, {"w": array([1.0])}, metadata={"arch": "llama"})
    gguf_loaded, gguf_meta = mx.load(gguf_file, return_metadata=True)
    assert isinstance(gguf_loaded, dict) and gguf_meta == {"arch": "llama"}
    mx.save_safetensors(io.BytesIO(), {"w": array([1.0])})
    mx.save_gguf(io.BytesIO(), {"w": array([1.0])})

    legacy_j1 = tmp_path / "leg1.json"
    legacy_j1.write_text('{"dtype": "float32", "data": [1.0, 2.0]}', encoding="utf-8")
    assert isinstance(mx.load(legacy_j1), array)
    legacy_j2 = tmp_path / "leg2.json"
    legacy_j2.write_text(
        '{"arrays": {"w": {"dtype": "float32", "data": [1.0]}}, "metadata": {"m": "1"}}',
        encoding="utf-8",
    )
    assert isinstance(mx.load(legacy_j2), dict)
    legacy_j3 = tmp_path / "leg3.json"
    legacy_j3.write_text('{"w": {"dtype": "float32", "data": [1.0]}}', encoding="utf-8")
    assert isinstance(mx.load(legacy_j3), dict)


def test_convolutions_and_linalg_and_fft() -> None:
    """Test all functions in `convolutions.py`, `linalg.py`, and `fft.py`."""
    x1 = mx.ones((1, 4, 2))
    w1 = mx.ones((3, 2, 2))
    assert z_conv.conv1d(x1, w1).shape[0] == 1
    assert z_conv.conv1d(mx.ones((1, 4, 4)), mx.ones((4, 2, 2)), groups=2).shape == (1, 3, 4)
    assert z_conv.conv_transpose1d(x1, w1).shape[0] == 1
    assert (
        z_conv.conv_transpose1d(
            mx.ones((1, 4, 2)),
            mx.ones((2, 2, 1)),
            padding=1,
            dilation=2,
            output_padding=1,
            groups=2,
        ).shape[0]
        == 1
    )
    assert (
        z_conv.conv_general(
            x1, w1, stride=[1], padding=(0, 0), kernel_dilation=[1], input_dilation=[2], flip=True
        ).shape[0]
        == 1
    )

    x2 = mx.ones((1, 4, 4, 2))
    w2 = mx.ones((3, 2, 2, 2))
    assert z_conv.conv2d(x2, w2).shape[0] == 1
    assert z_conv.conv2d(mx.ones((1, 4, 4, 4)), mx.ones((4, 2, 2, 2)), groups=2).shape[0] == 1
    assert z_conv.conv_transpose2d(x2, w2).shape[0] == 1
    assert (
        z_conv.conv_transpose2d(
            mx.ones((1, 4, 4, 2)),
            mx.ones((2, 2, 2, 1)),
            padding=1,
            dilation=2,
            output_padding=1,
            groups=2,
        ).shape[0]
        == 1
    )
    assert z_conv.conv_general(x2, w2).shape[0] == 1
    assert z_conv.conv_general(x2, w2, padding=(1, 1)).shape[0] == 1
    assert z_conv.conv_general(x2, w2, padding=((1, 1), (1, 1))).shape[0] == 1
    assert z_conv.conv_general(x2, w2, padding=[1]).shape[0] == 1

    x3 = mx.ones((1, 3, 3, 3, 2))
    w3 = mx.ones((2, 2, 2, 2, 2))
    assert z_conv.conv3d(x3, w3).shape[0] == 1
    assert z_conv.conv3d(mx.ones((1, 3, 3, 3, 4)), mx.ones((4, 2, 2, 2, 2)), groups=2).shape[0] == 1
    assert z_conv.conv_transpose3d(x3, w3).shape[0] == 1
    assert (
        z_conv.conv_transpose3d(
            mx.ones((1, 3, 3, 3, 2)),
            mx.ones((2, 2, 2, 2, 1)),
            padding=1,
            dilation=2,
            output_padding=1,
            groups=2,
        ).shape[0]
        == 1
    )
    assert z_conv.conv_general(x3, w3).shape[0] == 1
    assert z_conv.conv_general(x3, w3, stride=[1], kernel_dilation=[1]).shape[0] == 1

    assert z_conv.convolve(array([1.0, 2.0, 3.0]), array([1.0, 1.0]), mode="full").tolist() == [
        1.0,
        3.0,
        5.0,
        3.0,
    ]
    assert z_conv.convolve(array([1.0, 2.0, 3.0]), array([1.0, 1.0]), mode="valid").tolist() == [
        3.0,
        5.0,
    ]
    assert z_conv.convolve(array([1.0, 2.0, 3.0]), array([1.0, 1.0]), mode="same").tolist() == [
        1.0,
        3.0,
        5.0,
    ]
    assert (
        z_conv.convolve(array([1, 2], dtype=mx.int32), array([1, 2, 3], dtype=mx.int32)).dtype
        == mx.int32
    )
    assert z_conv.convolve(array([]), array([1.0])).size == 0
    with pytest.raises(ValueError):
        z_conv.convolve(array([1.0]), array([1.0]), mode="invalid")
    with pytest.raises(ValueError):
        z_conv.convolve(mx.ones((2, 2)), array([1.0]))

    w_q_in = mx.ones((4, 64))
    q_aff = z_conv.quantize(w_q_in, group_size=64, bits=4, mode="affine")
    assert len(q_aff) == 3
    q_nv = z_conv.quantize(w_q_in, group_size=32, bits=4, mode="nvfp4", global_scale=array(1.0))
    assert len(q_nv) == 2
    assert len(z_conv.quantize(array(1.0), mode="affine")) == 3
    assert len(z_conv.quantize(mx.ones((2, 5)), group_size=4, bits=6, mode="affine")) == 3
    deq = z_conv.dequantize(q_aff[0], q_aff[1], q_aff[2], global_scale=array(1.0))
    assert deq.shape == w_q_in.shape
    assert z_conv.dequantize(array(1.0), array(1.0)).ndim == 0
    assert z_conv.dequantize(w_q_in, mx.ones((4, 1))).shape == w_q_in.shape
    qmm = z_conv.quantized_matmul(mx.ones((2, 64)), q_aff[0], q_aff[1], q_aff[2])
    assert qmm.shape[0] == 2
    gqmm = z_conv.gather_qmm(
        mx.ones((2, 64)),
        q_aff[0],
        q_aff[1],
        q_aff[2],
        lhs_indices=array([0, 1]),
        rhs_indices=array([0, 1]),
    )
    assert gqmm.shape[0] == 2
    assert z_conv.qqmm(
        mx.ones((2, 4)), mx.ones((3, 4)), global_scale_x=array(1.0), global_scale_w=array(1.0)
    ).shape == (2, 3)
    assert z_conv.qqmm(
        mx.ones((2, 64)), q_nv[0], scales=q_nv[1], group_size=32, bits=4, mode="nvfp4"
    ).shape == (2, 4)
    assert z_conv.gather_qqmm(
        mx.ones((2, 4)),
        mx.ones((3, 4)),
        scales=mx.ones((3, 1)),
        lhs_indices=array([0]),
        rhs_indices=array([0, 1]),
    ).shape == (1, 2)

    m = array([[4.0, 1.0], [1.0, 3.0]])
    assert z_linalg.norm(m).ndim == 0
    assert z_linalg.matrix_norm(m).ndim == 0
    for ord_val in (1, -1, 2, -2, float("inf"), float("-inf"), "nuc"):
        assert z_linalg.matrix_norm(m, ord=ord_val).ndim == 0
        assert z_linalg.matrix_norm(m, ord=ord_val, keepdims=True).shape == (1, 1)
    with pytest.raises(ValueError):
        z_linalg.matrix_norm(m, ord="bad")
    with pytest.raises(ValueError):
        z_linalg.matrix_norm(array([1.0, 2.0]))
    q_mat, r_mat = z_linalg.qr(m)
    assert q_mat.shape == (2, 2) and r_mat.shape == (2, 2)
    with pytest.raises(ValueError):
        z_linalg.qr(array([1.0, 2.0]))
    with pytest.raises(ValueError):
        z_linalg.qr(array([[1, 2], [3, 4]], dtype=mx.int32))
    svd_res = z_linalg.svd(m, compute_uv=True)
    assert isinstance(svd_res, tuple)
    u, s, vt = svd_res
    assert u.shape == (2, 2) and s.shape == (2,) and vt.shape == (2, 2)
    s_only = z_linalg.svd(m, compute_uv=False)
    assert isinstance(s_only, array) and s_only.shape == (2,)
    assert z_linalg.inv(m).shape == (2, 2)
    assert z_linalg.tri_inv(m, upper=False).shape == (2, 2)
    assert z_linalg.tri_inv(m, upper=True).shape == (2, 2)
    assert z_linalg.cholesky(m, upper=False).shape == (2, 2)
    assert z_linalg.cholesky_inv(m, upper=False).shape == (2, 2)
    assert z_linalg.pinv(m).shape == (2, 2)
    assert z_linalg.cross(array([1.0, 0.0, 0.0]), array([0.0, 1.0, 0.0])).tolist() == [
        0.0,
        0.0,
        1.0,
    ]
    assert z_linalg.cross(mx.ones((3, 2)), mx.ones((3, 2)), axis=0).shape == (3, 2)
    w_eig, v_eig = z_linalg.eig(m)
    assert w_eig.shape == (2,) and v_eig.shape == (2, 2)
    assert z_linalg.eigvals(m).shape == (2,)
    w_eigh, v_eigh = z_linalg.eigh(m)
    assert w_eigh.shape == (2,) and v_eigh.shape == (2, 2)
    assert z_linalg.eigvalsh(m).shape == (2,)
    p_lu, l_lu, u_lu = z_linalg.lu(m)
    assert p_lu.shape == (2,) and l_lu.shape == (2, 2) and u_lu.shape == (2, 2)
    with pytest.raises(ValueError):
        z_linalg.lu(array([1.0, 2.0]))
    with pytest.raises(ValueError):
        z_linalg.lu(array([[1, 2], [3, 4]], dtype=mx.int32))
    lu_f, piv_f = z_linalg.lu_factor(m)
    assert lu_f.shape == (2, 2) and piv_f.shape == (2,)
    assert z_linalg.lu_solve(lu_f, array([1.0, 2.0])).shape == (2,)
    assert z_linalg.lu_solve((lu_f, piv_f), array([1.0, 2.0])).shape == (2,)
    with pytest.raises(ValueError):
        z_linalg.lu_solve(array([1.0, 2.0]), array([1.0, 2.0]))
    assert z_linalg.solve(m, array([1.0, 2.0])).shape == (2,)
    assert z_linalg.solve_triangular(m, array([1.0, 2.0])).shape == (2,)
    assert z_linalg.det(m).ndim == 0
    sign_d, logdet_d = z_linalg.slogdet(m)
    assert sign_d.ndim == 0 and logdet_d.ndim == 0
    for fn_det in (z_linalg.det, z_linalg.slogdet):
        with pytest.raises(ValueError):
            fn_det(array([1.0, 2.0]))
        with pytest.raises(ValueError):
            fn_det(mx.ones((2, 3)))
        with pytest.raises(ValueError):
            fn_det(array([[1.0 + 1.0j, 0.0], [0.0, 1.0]], dtype=mx.complex64))

    sig1 = array([1.0, 2.0, 3.0, 4.0])
    sig2 = mx.ones((4, 4))
    assert z_fft.fft(sig1).shape == (4,)
    assert z_fft.ifft(sig1).shape == (4,)
    assert z_fft.fft2(sig2).shape == (4, 4)
    assert z_fft.ifft2(sig2).shape == (4, 4)
    assert z_fft.fftn(sig2).shape == (4, 4)
    assert z_fft.ifftn(sig2).shape == (4, 4)
    assert z_fft.rfft(sig1).shape == (3,)
    assert z_fft.irfft(z_fft.rfft(sig1)).shape == (4,)
    assert z_fft.rfft2(sig2).shape == (4, 3)
    assert z_fft.irfft2(z_fft.rfft2(sig2)).shape == (4, 4)
    assert z_fft.rfftn(sig2).shape == (4, 3)
    assert z_fft.irfftn(z_fft.rfftn(sig2)).shape == (4, 4)
    assert z_fft.fftfreq(4).shape == (4,)
    assert z_fft.rfftfreq(4).shape == (3,)
    assert z_fft.fftshift(sig1).shape == (4,)
    assert z_fft.ifftshift(sig1).shape == (4,)
    spec = z_fft.stft(mx.ones((16,)), n_fft=8, hop_length=4, norm="ortho")
    assert spec.shape[0] == 5
    spec_short = z_fft.stft(
        array([1.0]),
        n_fft=4,
        hop_length=2,
        win_length=2,
        window=mx.ones((2,)),
        center=False,
        onesided=False,
        norm="forward",
    )
    assert spec_short.shape[0] == 4
    assert z_fft.stft(array([1.0]), n_fft=4, pad_mode="edge").shape[0] == 3
    assert z_fft._pad_stft_signal(mx.ones((1, 4)), 0, "reflect").shape == (1, 4)
    rec = z_fft.istft(spec, n_fft=8, hop_length=4, norm="ortho")
    assert rec.ndim == 1
    rec_len = z_fft.istft(mx.ones((2, 5, 4)), length=2, norm="forward")
    assert rec_len.shape == (2, 2)
    rec_long = z_fft.istft(mx.ones((5, 1)), length=32, center=False, onesided=False)
    assert rec_long.shape == (32,)


def test_ops_and_nn_comprehensive(tmp_path: pathlib.Path) -> None:
    """Test remaining functions and edge cases in `ops.py` and `nn/*`.

    Args:
        tmp_path: Pytest temporary directory fixture.

    """
    a = array([1.0, 2.0, 3.0])
    assert mx.asarray(a, copy=False) is a
    assert mx.asarray(a, dtype=mx.float32, copy=True).tolist() == [1.0, 2.0, 3.0]
    assert mx.asarray([1.0, 2.0]).tolist() == [1.0, 2.0]
    with pytest.raises(ValueError):
        mx.asarray([1.0, 2.0], copy=False)
    with pytest.raises(ValueError):
        mx.asarray(a, dtype=mx.int32, copy=False)

    assert mx.empty((2, 2)).shape == (2, 2)
    assert mx.arange(1, 5, 2, dtype=mx.float32).tolist() == [1.0, 3.0]
    assert mx.linspace(0.0, 1.0, 3).shape == (3,)
    assert z_rand._get_key_tensor(None).shape == (2,)
    assert z_rand._get_key_tensor(z_rand.key(1)).shape == (2,)
    assert z_nn.gelu_approx(a).shape == (3,)
    assert z_nn.gelu(a, "precise").shape == (3,)

    def _chk_id(*args: array, **kwargs: array) -> array:
        """Identity function for checkpoint.

        Args:
            *args: Positional array arguments.
            **kwargs: Keyword array arguments.

        Returns:
            First positional `array`.

        """
        _ = kwargs
        return args[0]

    chk_out = z_new_ops.checkpoint(_chk_id)(a)
    assert isinstance(chk_out, array) and chk_out.tolist() == [1.0, 2.0, 3.0]
    assert mx.logcumsumexp(array([1.0, 2.0]), reverse=True, inclusive=False).shape == (2,)
    assert mx.concatenate([a]).tolist() == [1.0, 2.0, 3.0]
    assert len(mx.unstack(mx.ones((2, 3)), axis=0)) == 2
    assert mx.vecdot(a, a).item() == 14.0
    assert mx.kron(array([1.0, 2.0]), array([3.0, 4.0])).tolist() == [3.0, 4.0, 6.0, 8.0]
    import ml_switcheroo_compiler.tracing as _tr
    from ml_switcheroo_compiler.core.config import config as _sw_cfg

    _sw_cfg.eager_mode = False
    _tr.global_tracing_state.start_tracing("cov_trace")
    try:
        t_tr = mx.abs(array([-1.0, 2.0]))
        z_new_ops.eval(t_tr)
        z_new_ops.eval(123)
        _ = mx.put_along_axis(array([1.0, 2.0]), array([0]), array([5.0]), axis=0)
    finally:
        _tr.global_tracing_state.stop_tracing()
        _sw_cfg.eager_mode = True

    opt_base = mx.optimizers.Optimizer()
    opt_res = opt_base.apply_gradients({"a": array([2.0]), "b": 1}, {"a": array([1.0]), "b": 1})
    assert isinstance(opt_res, dict) and opt_res["b"] == 1
    assert mx.empty_like(a).shape == (3,)
    assert mx.full((2, 2), array([1.0, 2.0])).shape == (2, 2)
    assert mx.full_like(a, 5.0).tolist() == [5.0, 5.0, 5.0]
    assert mx.ones_like(a).tolist() == [1.0, 1.0, 1.0]
    assert mx.identity(3).shape == (3, 3)
    assert mx.tri(3, 4, k=1).shape == (3, 4)
    assert mx.tril(mx.ones((3, 3))).shape == (3, 3)
    assert mx.triu(mx.ones((3, 3))).shape == (3, 3)
    assert mx.diag(array([1.0, 2.0]), k=1).shape == (3, 3)
    assert mx.diag(array([1.0, 2.0]), k=-1).shape == (3, 3)
    assert mx.diag(mx.eye(2)).tolist() == [1.0, 1.0]
    assert mx.meshgrid() == []
    assert len(mx.meshgrid(a, a, sparse=True, indexing="ij")) == 2
    for win_fn in (mx.bartlett, mx.blackman, mx.hamming, mx.hanning):
        assert win_fn(0).size == 0
        assert win_fn(1).tolist() == [1.0]
        assert win_fn(5).shape == (5,)

    assert mx.positive(a).tolist() == [1.0, 2.0, 3.0]
    assert mx.trunc(array([-1.7, 1.7])).tolist() == [-1.0, 1.0]
    assert mx.round(array([1.234]), decimals=2).shape == (1,)
    assert mx.asin(array([0.0])).tolist() == [0.0]
    assert mx.acos(array([1.0])).tolist() == [0.0]
    assert mx.atan(array([0.0])).tolist() == [0.0]
    assert mx.atan2(array([0.0]), array([1.0])).tolist() == [0.0]
    assert mx.asinh(array([0.0])).tolist() == [0.0]
    assert mx.acosh(array([1.0])).tolist() == [0.0]
    assert mx.atanh(array([0.0])).tolist() == [0.0]
    assert mx.degrees(array([0.0])).tolist() == [0.0]
    assert mx.radians(array([0.0])).tolist() == [0.0]
    assert mx.erfinv(array([0.0])).tolist() == [0.0]
    assert mx.pow(a, 2.0).tolist() == [1.0, 4.0, 9.0]
    assert mx.clip(a, 1.5, 2.5).tolist() == [1.5, 2.0, 2.5]
    assert mx.real(a).tolist() == [1.0, 2.0, 3.0]
    assert mx.imag(a).tolist() == [0.0, 0.0, 0.0]
    assert mx.conj(a).tolist() == [1.0, 2.0, 3.0]
    assert bool(mx.array_equal(a, a).item()) is True
    assert bool(mx.array_equal(a, array([1.0, 2.0])).item()) is False
    assert (
        bool(mx.array_equal(array([float("nan")]), array([float("nan")]), equal_nan=True).item())
        is True
    )
    assert bool(mx.allclose(a, a).item()) is True
    assert mx.isposinf(array([float("inf"), -1.0])).tolist() == [True, False]
    assert mx.isneginf(array([float("-inf"), 1.0])).tolist() == [True, False]
    assert mx.logical_xor(array([True, False]), array([True, True])).tolist() == [False, True]
    assert mx.bitwise_and(array([3]), array([1])).tolist() == [1]
    assert mx.bitwise_or(array([2]), array([1])).tolist() == [3]
    assert mx.bitwise_xor(array([3]), array([1])).tolist() == [2]
    assert mx.bitwise_invert(array([0], dtype=mx.int32)).tolist() == [-1]
    assert mx.astype(a, mx.int32).tolist() == [1, 2, 3]
    assert mx.view(a, mx.float32).shape == (3,)
    assert mx.can_cast(mx.int16, mx.float32) is True
    assert mx.can_cast(a, mx.int8) is False
    assert mx.result_type() == mx.float32
    assert mx.result_type(a, mx.int32, "float32", 1) == mx.float32
    fp8 = mx.to_fp8(a)
    assert mx.from_fp8(fp8).shape == (3,)
    assert mx.minmax(a)[0].item() == 1.0 and mx.minmax(a)[1].item() == 3.0
    assert mx.median(a).item() == 2.0
    assert mx.count_nonzero(array([0, 1, 2])).item() == 2

    # Reverse/exclusive scans
    assert mx.cumsum(a, reverse=True, inclusive=False).shape == (3,)
    assert mx.cumprod(a, reverse=True, inclusive=False).shape == (3,)
    assert mx.cummax(a, reverse=True, inclusive=False).shape == (3,)
    assert mx.cummin(a, reverse=True, inclusive=False).shape == (3,)
    assert mx.logcumsumexp(a, reverse=True, inclusive=False).shape == (3,)
    assert mx.cumsum(array([]), inclusive=False).size == 0
    assert mx.diff(a, n=1, prepend=array([0.0]), append=array([4.0])).tolist() == [
        1.0,
        1.0,
        1.0,
        1.0,
    ]
    assert mx.diff(a, n=0).tolist() == [1.0, 2.0, 3.0]

    assert mx.flatten(array(1.0)).shape == (1,)
    assert mx.flatten(mx.ones((2, 3)), start_axis=1, end_axis=0).shape == (2, 3)
    assert mx.unflatten(mx.ones((6,)), 0, (2, 3)).shape == (2, 3)
    assert mx.permute_dims(mx.ones((2, 3))).shape == (3, 2)
    assert mx.matrix_transpose(mx.ones((2, 3))).shape == (3, 2)
    with pytest.raises(ValueError):
        mx.matrix_transpose(a)
    with pytest.raises(ValueError):
        mx.concatenate([])
    assert mx.concatenate([a, a], axis=None).shape == (6,)
    assert mx.concat([a, a]).shape == (6,)
    with pytest.raises(ValueError):
        mx.stack([])
    assert len(mx.unstack(mx.ones((2, 3)), axis=1)) == 3
    assert len(mx.split(mx.arange(6), [2, 4])) == 3
    with pytest.raises(ValueError):
        mx.split(mx.arange(5), 2)
    assert mx.flip(array(1.0)).item() == 1.0
    assert mx.flip(mx.ones((2, 3))).shape == (2, 3)
    assert mx.flip(mx.ones((2, 3)), axis=0).shape == (2, 3)
    assert mx.flip(mx.ones((2, 3)), axis=(0, 1)).shape == (2, 3)
    assert mx.as_strided(mx.arange(6), shape=(2, 2), strides=(2, 1), offset=1).shape == (2, 2)
    assert mx.as_strided(mx.arange(4), shape=(2, 2)).shape == (2, 2)
    assert mx.contiguous(a).tolist() == [1.0, 2.0, 3.0]
    with pytest.raises(ValueError):
        mx.broadcast_to(mx.ones((3,)), (2,))
    with pytest.raises(ValueError):
        mx.broadcast_to(mx.ones((2, 3)), (2, 1))
    assert mx.broadcast_arrays() == []
    assert len(mx.broadcast_arrays(array([1.0]), mx.ones((2, 2)))) == 2
    a1d = mx.atleast_1d(array(1.0))
    assert isinstance(a1d, array) and a1d.shape == (1,)
    a1d_list = mx.atleast_1d(array(1.0), a)
    assert isinstance(a1d_list, list) and len(a1d_list) == 2
    a2d = mx.atleast_2d(array(1.0))
    assert isinstance(a2d, array) and a2d.shape == (1, 1)
    a2d_list = mx.atleast_2d(a, mx.ones((2, 2)))
    assert isinstance(a2d_list, list) and len(a2d_list) == 2
    a3d = mx.atleast_3d(array(1.0))
    assert isinstance(a3d, array) and a3d.shape == (1, 1, 1)
    a3d_list = mx.atleast_3d(a, mx.ones((2, 2)), mx.ones((2, 2, 2)))
    assert isinstance(a3d_list, list) and len(a3d_list) == 3
    assert mx.put_along_axis(array([1.0, 2.0, 3.0]), array([1]), array([9.0]), axis=0).tolist() == [
        1.0,
        9.0,
        3.0,
    ]
    assert mx.slice(mx.arange(6).reshape((2, 3)), array([0, 1]), [0, 1], [1, 2]).shape == (1, 2)
    assert mx.slice_update(mx.zeros((2, 3)), mx.ones((1, 2)), array([0, 1]), [0, 1]).shape == (2, 3)
    assert mx.sort(a, axis=None).tolist() == [1.0, 2.0, 3.0]
    assert mx.argsort(a, axis=None).tolist() == [0, 1, 2]
    assert mx.partition(a, kth=1, axis=None).shape == (3,)
    assert mx.argpartition(a, kth=1, axis=None).shape == (3,)
    assert mx.topk(a, k=2, axis=None).tolist() == [2.0, 3.0]
    assert mx.searchsorted(array([1.0, 3.0, 5.0]), array(2.0)).item() == 1
    assert mx.searchsorted(
        array([5.0, 1.0, 3.0]), array([2.0, 4.0]), side="right", sorter=array([1, 2, 0])
    ).tolist() == [1, 2]
    u_only = mx.unique(array([3.0, 1.0, 2.0, 1.0]))
    assert isinstance(u_only, array) and u_only.tolist() == [1.0, 2.0, 3.0]
    u_tup = mx.unique(
        array([2.0, 1.0, 2.0]),
        size=4,
        return_index=True,
        return_inverse=True,
        return_counts=True,
        fill_value=0.0,
    )
    assert isinstance(u_tup, tuple) and len(u_tup) == 4
    u_trunc = mx.unique(array([1.0, 2.0, 3.0]), size=2)
    assert isinstance(u_trunc, array) and u_trunc.shape == (2,)
    assert mx.addmm(mx.zeros((2, 2)), mx.eye(2), mx.ones((2, 2))).tolist() == [
        [1.0, 1.0],
        [1.0, 1.0],
    ]
    assert mx.vecdot(array([1.0 + 1.0j]), array([1.0 + 1.0j])).real.item() == 2.0
    assert mx.trace(mx.eye(3), dtype=mx.float32).item() == 3.0
    assert mx.einsum("ij,jk->ik", mx.eye(2), mx.ones((2, 2))).shape == (2, 2)
    path_ein, _ = mx.einsum_path("ij,jk->ik", mx.eye(2), mx.ones((2, 2)))
    assert len(path_ein) >= 1
    assert mx.gather_mm(mx.eye(2), mx.ones((2, 2))).shape == (2, 2)
    assert mx.block_masked_mm(mx.eye(2), mx.ones((2, 2)), block_size=2).shape == (2, 2)
    assert mx.segmented_mm(
        mx.ones((2, 4)), mx.ones((4, 2)), array([[0, 2], [2, 4]], dtype=mx.int32)
    ).shape == (2, 2, 2)
    assert mx.hadamard_transform(array([1.0, 1.0])).shape == (2,)
    assert mx.stop_gradient(a).tolist() == [1.0, 2.0, 3.0]
    dep_a = mx.depends(a, a)
    assert isinstance(dep_a, array) and dep_a.tolist() == [1.0, 2.0, 3.0]
    dep_list = mx.depends([a], [a])
    assert isinstance(dep_list, list) and len(dep_list) == 1

    # NN initializers, quantized, base, layers, transformer, losses, pooling, upsample
    t_mat = mx.zeros((4, 4))
    assert z_nn.init.constant(2.0)(t_mat).shape == (4, 4)
    assert z_nn.init.normal()(t_mat).shape == (4, 4)
    assert z_nn.init.uniform()(t_mat).shape == (4, 4)
    assert z_nn.init.identity()(t_mat).shape == (4, 4)
    with pytest.raises(ValueError):
        z_nn.init.identity()(mx.zeros((2, 3)))
    with pytest.raises(ValueError):
        z_nn.init.glorot_normal()(array([1.0, 2.0]))
    assert z_nn.init.glorot_normal()(mx.zeros((2, 3, 4))).shape == (2, 3, 4)
    assert z_nn.init.glorot_uniform()(t_mat).shape == (4, 4)
    assert z_nn.init.he_normal()(t_mat, mode="fan_out").shape == (4, 4)
    with pytest.raises(ValueError):
        z_nn.init.he_normal()(t_mat, mode="bad")
    assert z_nn.init.he_uniform()(t_mat, mode="fan_in").shape == (4, 4)
    with pytest.raises(ValueError):
        z_nn.init.he_uniform()(t_mat, mode="bad")
    assert z_nn.init.orthogonal()(t_mat).shape == (4, 4)
    with pytest.raises(ValueError):
        z_nn.init.orthogonal()(array([1.0]))
    assert z_nn.init.sparse(0.5)(t_mat).shape == (4, 4)
    with pytest.raises(ValueError):
        z_nn.init.sparse(0.5)(array([1.0]))

    from typing import TYPE_CHECKING

    class SubMod(z_nn.Module):
        """Test composite module."""

        if TYPE_CHECKING:
            lin: z_nn.Linear | z_nn.QuantizedLinear | z_nn.QQLinear
            emb: z_nn.Embedding | z_nn.QuantizedEmbedding
            seq: z_nn.Sequential
            extra_val: int
        else:
            lin = z_nn.base.ModuleAttr("lin")
            emb = z_nn.base.ModuleAttr("emb")
            seq = z_nn.base.ModuleAttr("seq")
            extra_val = z_nn.base.ModuleAttr("extra_val")

        def __init__(self) -> None:
            """Initialize composite module."""
            super().__init__()
            self.lin = z_nn.Linear(64, 4)
            self.emb = z_nn.Embedding(8, 64)
            self.seq = z_nn.Sequential(z_nn.Linear(4, 4))

        def __call__(self, x: array) -> array:
            """Forward pass.

            Args:
                x: Input tensor.

            Returns:
                Linear projection output `array`.

            """
            return self.lin(x)

    sm = SubMod()
    assert len(sm.children()) >= 2
    assert len(sm.modules()) >= 3
    assert len(sm.named_modules()) >= 3
    sm.freeze(keys="lin", strict=True)
    assert "lin" not in sm.trainable_parameters()
    with pytest.raises(KeyError):
        sm.freeze(keys="nonexistent", strict=True)
    sm.freeze()
    assert sm.trainable_parameters() == {"emb": {}, "seq": {}}
    sm.unfreeze(keys="lin", strict=True)
    with pytest.raises(KeyError):
        sm.unfreeze(keys="nonexistent", strict=True)
    sm.unfreeze()
    sm.apply(lambda p: p * 1.0)
    w_npz = str(tmp_path / "m.npz")
    w_st = str(tmp_path / "m.safetensors")
    sm.save_weights(w_npz)
    sm.load_weights(w_npz)
    sm.save_weights(w_st)
    sm.load_weights({"lin.weight": sm.lin.weight})
    sm.load_weights([("lin.weight", sm.lin.weight)])

    z_nn.quantize(sm, class_predicate=lambda k, m: {"group_size": 32, "bits": 4, "mode": "affine"})
    assert isinstance(sm.lin, z_nn.QuantizedLinear)
    assert isinstance(sm.emb, z_nn.QuantizedEmbedding)
    sm2 = SubMod()
    sm2.extra_val = 42
    z_nn.quantize(sm2, class_predicate=lambda k, m: False)
    z_nn.quantize(sm2, quantize_input=True, mode="nvfp4")
    assert isinstance(sm2.lin, z_nn.QQLinear)
    assert sm2.lin(mx.ones((2, 64))).shape == (2, 4)
    assert sm.emb.as_linear(mx.ones((2, 64))).shape == (2, 8)
    assert z_nn.Embedding(8, 4).as_linear(mx.ones((2, 4))).shape == (2, 8)

    fsdp = z_nn.FullyShardedModule(z_nn.Linear(4, 2))
    fsdp_out = fsdp(mx.ones((2, 4)))
    assert isinstance(fsdp_out, array) and fsdp_out.shape == (2, 2)
    assert z_nn.average_gradients({"w": array([1.0])}) == {"w": array([1.0])}

    def _loss1(*args: array | z_nn.Module, **kwargs: array) -> array:
        """Loss taking input array.

        Args:
            *args: Positional arguments.
            **kwargs: Keyword arguments.

        Returns:
            Scalar loss `array`.

        """
        _ = kwargs
        if isinstance(args[0], array):
            return mx.sum(sm(args[0]))
        raise TypeError("Expected array")

    def _loss2(*args: array | z_nn.Module, **kwargs: array) -> array:
        """Loss taking (model, x).

        Args:
            *args: Positional arguments.
            **kwargs: Keyword arguments.

        Returns:
            Scalar loss `array`.

        """
        _ = kwargs
        if isinstance(args[0], z_nn.Module) and len(args) > 1 and isinstance(args[1], array):
            res = args[0]._call_forward(args[1])
            return mx.sum(res[0] if isinstance(res, tuple) else res)
        raise TypeError("Expected (Module, array)")

    def _chk_sm(*args: array, **kwargs: array) -> array:
        """Checkpoint wrapper around sm.

        Args:
            *args: Positional array arguments.
            **kwargs: Keyword array arguments.

        Returns:
            Module output `array`.

        """
        _ = kwargs
        return sm(args[0])

    vng_nn = z_nn.value_and_grad(sm, _loss1)
    loss_v, grads_v = vng_nn(mx.ones((2, 64)))
    assert loss_v.ndim == 0 and isinstance(grads_v, dict)
    vng_nn2 = z_nn.value_and_grad(sm, _loss2)
    loss_v2, _ = vng_nn2(mx.ones((2, 64)))
    assert loss_v2.ndim == 0
    chk_nn = z_nn.checkpoint(sm)
    assert chk_nn(mx.ones((2, 64))).shape == (2, 4)
    chk_nn2 = z_nn.checkpoint(sm, _chk_sm)
    assert chk_nn2(mx.ones((2, 64))).shape == (2, 4)

    with pytest.raises(ValueError):
        z_nn.Dropout(-0.1)
    assert z_nn.ConvTranspose1d(2, 2, 2, padding=1)(mx.ones((1, 4, 2))).shape == (1, 3, 2)
    assert z_nn.ConvTranspose2d(2, 2, 2, padding=1)(mx.ones((1, 4, 4, 2))).shape == (1, 3, 3, 2)
    assert z_nn.ConvTranspose3d(2, 2, 2, padding=1)(mx.ones((1, 4, 4, 4, 2))).shape == (
        1,
        3,
        3,
        3,
        2,
    )
    assert z_nn.MaxPool1d(2, padding=1)(mx.ones((1, 4, 2))).shape == (1, 3, 2)
    assert z_nn.MaxPool2d(2, padding=1)(mx.ones((1, 4, 4, 2))).shape == (1, 3, 3, 2)
    assert z_nn.MaxPool3d(2, padding=1)(mx.ones((1, 4, 4, 4, 2))).shape == (1, 3, 3, 3, 2)
    assert z_nn.Upsample(size=4)(mx.ones((1, 2, 2, 2))).shape == (1, 4, 4, 2)
    assert z_nn.Upsample(scale_factor=(2.0, 2.0))(mx.ones((1, 2, 2, 2))).shape == (1, 4, 4, 2)
    assert z_nn.Upsample(scale_factor=2.0, mode="linear", align_corners=True)(
        mx.ones((1, 2, 2, 2))
    ).shape == (1, 4, 4, 2)
    assert z_nn.Upsample(scale_factor=2.0, mode="cubic", align_corners=False)(
        mx.ones((1, 2, 2, 2))
    ).shape == (1, 4, 4, 2)
    assert z_nn.Upsample(size=3, mode="linear")(mx.ones((1, 1, 2))).shape == (1, 3, 2)
    assert z_nn.Upsample(size=2, mode="linear")(mx.ones((1, 2, 2))).shape == (1, 2, 2)
    assert z_nn.Upsample(scale_factor=None, size=None)(mx.ones((1, 2, 2, 2))).shape == (1, 2, 2, 2)

    alibi = z_nn.ALiBi()
    assert alibi(mx.zeros((1, 3, 4, 4)), mask=mx.zeros((1, 3, 4, 4))).shape == (1, 3, 4, 4)
    assert alibi(mx.zeros((4, 4)), mask=mx.zeros((4, 4))).shape == (4, 4)
    assert z_nn.RoPE(4)(mx.ones((1, 3, 8))).shape == (1, 3, 8)
    assert z_nn.SinusoidalPositionalEncoding(4, cos_first=True, full_turns=True)(
        array([0, 1, 2])
    ).shape == (3, 4)

    rnn = z_nn.RNN(4, 4, bias=False, nonlinearity="relu")
    assert rnn(mx.ones((3, 4)), hidden=mx.zeros((4,)))[0].shape == (3, 4)
    gru = z_nn.GRU(4, 4, bias=False)
    assert gru(mx.ones((3, 4)), hidden=mx.zeros((4,)))[0].shape == (3, 4)
    lstm = z_nn.LSTM(4, 4, bias=True)
    assert lstm(mx.ones((2, 3, 4)))[0].shape == (2, 3, 4)
    lstm_nb = z_nn.LSTM(4, 4, bias=False)
    assert lstm_nb(mx.ones((3, 4)), hidden=(mx.zeros((1, 4)), mx.zeros((1, 4))))[0].shape == (3, 4)
    assert lstm_nb(mx.ones((3, 4)), hidden=mx.zeros((4,)), cell=mx.zeros((4,)))[0].shape == (3, 4)

    with pytest.raises(ValueError):
        z_nn.MultiHeadAttention(10, 3)
    with pytest.raises(ValueError):
        z_nn.MultiHeadAttention(8, 2, value_dims=5)
    assert z_nn.MultiHeadAttention.create_additive_causal_mask(4).shape == (4, 4)
    mha = z_nn.MultiHeadAttention(8, 2)
    assert mha(
        mx.ones((1, 3, 8)), mx.ones((1, 3, 8)), mx.ones((1, 3, 8)), mask=mx.zeros((1, 2, 3, 3))
    ).shape == (1, 3, 8)
    enc_post = z_nn.TransformerEncoderLayer(8, 2, norm_first=False)
    assert enc_post(mx.ones((1, 3, 8))).shape == (1, 3, 8)
    dec_post = z_nn.TransformerDecoderLayer(8, 2, norm_first=False)
    assert dec_post(mx.ones((1, 3, 8)), mx.ones((1, 3, 8))).shape == (1, 3, 8)
    tf_model = z_nn.Transformer(dims=8, num_heads=2, num_encoder_layers=1, num_decoder_layers=1)
    assert tf_model(mx.ones((1, 3, 8)), mx.ones((1, 3, 8))).shape == (1, 3, 8)

    # Losses
    logits = array([[2.0, 1.0], [0.5, 1.5]])
    targets_idx = array([0, 1])
    targets_prob = array([[1.0, 0.0], [0.0, 1.0]])
    assert (
        z_nn.losses.cross_entropy(logits, targets_idx, label_smoothing=0.1, reduction="mean").ndim
        == 0
    )
    assert (
        z_nn.losses.cross_entropy(
            logits, targets_prob, weights=array([1.0, 2.0]), reduction="sum"
        ).ndim
        == 0
    )
    with pytest.raises(ValueError):
        z_nn.losses.cross_entropy(logits, targets_idx, label_smoothing=1.5)
    with pytest.raises(ValueError):
        z_nn.losses.cross_entropy(logits, targets_idx, reduction="invalid")
    assert (
        z_nn.losses.binary_cross_entropy(
            array([0.5, -0.5]), array([1.0, 0.0]), weights=array([1.0, 1.0]), with_logits=True
        ).ndim
        == 0
    )
    assert (
        z_nn.losses.binary_cross_entropy(
            array([0.8, 0.2]), array([1.0, 0.0]), with_logits=False
        ).ndim
        == 0
    )
    assert z_nn.losses.gaussian_nll_loss(
        array([1.0]), array([1.0]), array([1.0]), full=True
    ).shape == (1,)
    assert z_nn.losses.hinge_loss(array([1.0, -1.0]), array([1.0, -1.0])).shape == (2,)
    assert z_nn.losses.huber_loss(array([0.5, 2.0]), array([0.0, 0.0]), delta=1.0).shape == (2,)
    assert z_nn.losses.kl_div_loss(array([[-0.5, -0.5]]), array([[0.5, 0.5]])).shape == (1,)
    assert z_nn.losses.l1_loss(array([1.0, 2.0]), array([0.0, 2.0])).ndim == 0
    assert z_nn.losses.log_cosh_loss(array([1.0, 2.0]), array([0.0, 1.0])).shape == (2,)
    assert z_nn.losses.margin_ranking_loss(array([1.0]), array([0.5]), array([1.0])).shape == (1,)
    assert z_nn.losses.mse_loss(array([1.0, 2.0]), array([1.0, 0.0])).ndim == 0
    assert (
        z_nn.losses.nll_loss(logits, targets_idx, weight=array([1.0, 2.0]), reduction="mean").ndim
        == 0
    )
    assert z_nn.losses.nll_loss(logits, targets_idx, reduction="sum").ndim == 0
    assert z_nn.losses.nll_loss(logits, targets_idx, reduction="none").shape == (2,)
    assert z_nn.losses.smooth_l1_loss(array([0.2, 2.0]), array([0.0, 0.0])).ndim == 0
    assert z_nn.losses.triplet_loss(mx.ones((2, 3)), mx.ones((2, 3)), mx.zeros((2, 3))).shape == (
        2,
    )
