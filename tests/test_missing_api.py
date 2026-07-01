import pytest
import zero_mlx as mx
import zero_mlx.cuda as cuda


def test_checkpoint_parity():
    def f(x):
        return x * 2

    mock_f = mx.checkpoint(f)
    assert mock_f(2) == 4


def test_cuda_parity():
    assert cuda.is_available() is False
    assert cuda.device_count() == 0
    assert cuda.memory_info() == {}
    assert cuda.clear_cache() is None


def test_custom_function_parity():
    @mx.custom_function
    def my_f(x):
        return x * 2

    @my_f.vjp
    def my_f_vjp(inputs, outputs, cotangents):
        return (cotangents[0] * 2,)

    @my_f.jvp
    def my_f_jvp(inputs, tangents):
        return (my_f(*inputs), tangents[0] * 2)

    @my_f.vmap
    def my_f_vmap(inputs, axes):
        return (my_f(*inputs), axes)

    assert my_f(3) == 6
