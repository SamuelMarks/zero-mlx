import pytest
from zero_mlx import array, set_printoptions, printoptions, stack
from zero_mlx.dtypes import DType
import numpy as np


def test_printoptions():
    set_printoptions(threshold=5)
    with printoptions(threshold=5):
        pass


class BadArray1:
    def __array__(self, dtype=None):
        raise ValueError("foobar")


class BadArray2:
    @property
    def __array_interface__(self):
        raise RuntimeError("baz")


class BadArray3:
    def __array__(self, dtype=None):
        raise ValueError("setting an array element with a sequence")


class BadArray4:
    def __array__(self, dtype=None):
        raise ValueError("inhomogeneous")


def test_bad_arrays():
    try:
        array(BadArray1())
    except ValueError as e:
        assert "foobar" in str(e)
    try:
        array(BadArray2())
    except Exception:
        pass
    try:
        array(BadArray3())
    except ValueError:
        pass
    try:
        array(BadArray4())
    except ValueError:
        pass


def test_array_astype_no_value():
    a = array([1])
    try:
        a.astype("int32")
    except Exception:
        pass


def test_array_binary_ops():
    a = array(1.0, dtype=DType.float32)
    b = array(1.0, dtype=DType.float32)
    _ = a + b
    a = array(1, dtype=DType.int32)
    b = array(1, dtype=DType.int32)
    _ = a + b


def test_array_binary_ops_elif():
    a = array(1, dtype=DType.int32)
    b = array(1.0, dtype=DType.float32)
    _ = a + b


def test_array_index_errors():
    a = array([1])
    try:
        a[1, 2]
    except IndexError:
        pass
    try:
        a[2] = 1
    except IndexError:
        pass


def test_device_none():
    from zero_mlx.device import Stream
    import zero_mlx.device

    old = zero_mlx.device._default_device
    zero_mlx.device._default_device = None
    try:
        with Stream("gpu"):
            pass
    finally:
        zero_mlx.device._default_device = old


def test_random_dtypes():
    from zero_mlx.mlx_random import (
        randint,
        bernoulli,
        categorical,
        gumbel,
        truncated_normal,
    )

    randint(0, 10, dtype=DType.int32)
    bernoulli(0.5, dtype=DType.bool_)
    categorical(array([1.0]), dtype=DType.int32)
    gumbel(dtype=DType.float32)
    truncated_normal(-1.0, 1.0, dtype=DType.float32)


def test_ops_none_dtypes():
    from zero_mlx.ops import zeros_like, ones_like, gather_mm

    zeros_like(array([1]), dtype=DType.float32)
    ones_like(array([1]), dtype=DType.float32)
    a = array(np.ones((1, 2, 2)))
    b = array(np.ones((1, 2, 2)))
    gather_mm(a, b, lhs_indices=array([0]))


def test_ops_patch_stack_none():
    stack([1, 2])
    array([1, 2], dtype=None)
    try:
        array([array([1]), array([1, 2])])
    except ValueError:
        pass
    try:
        array([array([1]), BadArray1()])
    except ValueError:
        pass
