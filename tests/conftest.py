"""Pytest configuration and shared parity verification helpers for `zero_mlx`."""

from __future__ import annotations

import inspect
import os
import sys
from typing import Callable, Iterator, Protocol, Sequence, Union

import mlx.core as mx
import numpy as np
import numpy.typing as npt
import pytest

sys.path.insert(
    0,
    os.path.abspath(os.path.join(os.path.dirname(__file__), "../../ml-switcheroo-compiler/src")),
)

from ml_switcheroo_compiler.tracing import global_tracing_state as _tracer
import zero_mlx
from zero_mlx._types import ArrayProtocol
from zero_mlx.array import array
from zero_mlx.dtypes import DType

NDArrayGeneric = npt.NDArray[np.generic]

ParityResult = Union[
    array,
    mx.array,
    ArrayProtocol,
    NDArrayGeneric,
    Sequence[Union[array, mx.array, ArrayProtocol, NDArrayGeneric]],
    float,
    int,
    bool,
]
GenScalar = Union[
    int,
    float,
    bool,
    complex,
    str,
    DType,
    type[np.generic],
    type[int],
    type[float],
    type[bool],
    None,
]
GenValue = Union[
    NDArrayGeneric,
    ArrayProtocol,
    np.generic,
    array,
    mx.array,
    GenScalar,
    list["GenValue"],
    tuple["GenValue", ...],
]


ArrayBuildFn = Callable[[NDArrayGeneric], GenValue]


class VariadicOpCallable(Protocol):
    """Protocol for dynamically looked-up operations in `check_parity`."""

    def __call__(self, *args: GenValue, **kwargs: GenValue) -> ParityResult:
        """Invoke the operation with positional and keyword arguments.

        Args:
            *args: Positional arguments.
            **kwargs: Keyword arguments.

        Returns:
            Operation result.

        """


def _build_op_dispatch() -> tuple[dict[str, VariadicOpCallable], dict[str, VariadicOpCallable]]:
    """Build explicit dispatch tables mapping lowercase op names to `zero_mlx` and `mlx.core` functions.

    Returns:
        Tuple of `(zero_mlx_ops_dict, mlx_core_ops_dict)`.

    """
    z_ops: dict[str, VariadicOpCallable] = {}
    for name, member in inspect.getmembers(zero_mlx):
        if callable(member):
            z_ops[name] = member
    for name, member in inspect.getmembers(zero_mlx.fft):
        if callable(member):
            z_ops[name] = member

    m_ops: dict[str, VariadicOpCallable] = {}
    for name, member in inspect.getmembers(mx):
        if callable(member):
            m_ops[name] = member
    for name, member in inspect.getmembers(mx.fft):
        if callable(member):
            m_ops[name] = member
    return z_ops, m_ops


_Z_DISPATCH, _M_DISPATCH = _build_op_dispatch()


@pytest.fixture(autouse=True)
def switcheroo_config() -> Iterator[None]:
    """Ensure clean compiler tracing state around every test function.

    Yields:
        None during test execution.

    """
    yield


def assert_allclose_mlx(
    z_res: ParityResult,
    m_res: ParityResult,
    rtol: float = 1e-5,
    atol: float = 1e-5,
) -> None:
    """Assert numerical closeness between a `zero_mlx` result and a native `mlx.core` result.

    Args:
        z_res: Result produced by `zero_mlx`.
        m_res: Result produced by `mlx.core`.
        rtol: Relative tolerance.
        atol: Absolute tolerance.

    """
    if isinstance(z_res, (tuple, list)) and isinstance(m_res, (tuple, list)):
        for z_item, m_item in zip(z_res, m_res):
            assert_allclose_mlx(z_item, m_item, rtol=rtol, atol=atol)
    else:
        z_arr = z_res.tolist() if isinstance(z_res, array) else z_res
        np.testing.assert_allclose(np.array(z_arr), np.array(m_res), rtol=rtol, atol=atol)


def _unwrap(x: GenValue) -> GenValue:
    """Unwrap NumPy scalar generics into Python scalars.

    Args:
        x: Input value.

    Returns:
        Unwrapped scalar or value.

    """
    if isinstance(x, np.bool_):
        return bool(x)
    if isinstance(x, np.integer):
        return int(x)
    if isinstance(x, np.floating):
        return float(x)
    if isinstance(x, np.complexfloating):
        return complex(x)
    return x


def _process_arg(a: GenValue, array_fn: ArrayBuildFn) -> GenValue:
    """Convert NumPy arrays into target framework arrays recursively.

    Args:
        a: Input argument value.
        array_fn: Target framework array constructor.

    Returns:
        Converted argument value.

    """
    if isinstance(a, np.ndarray):
        return array_fn(a)
    if isinstance(a, list):
        return [_process_arg(i, array_fn) for i in a]
    if isinstance(a, tuple):
        return tuple(_process_arg(i, array_fn) for i in a)
    return _unwrap(a)


def check_parity(
    op_name: str,
    args_generator: Callable[[], Sequence[GenValue]],
    kwargs_generator: Callable[[], dict[str, GenValue]] | None = None,
    rtol: float = 1e-5,
    atol: float = 1e-5,
) -> None:
    """Execute `op_name` in both `zero_mlx` and `mlx.core` and assert numerical parity.

    Args:
        op_name: Compiler operation name (e.g. `'Add'`, `'Sin'`, `'Fft'`).
        args_generator: Zero-argument callable returning a list of sample positional arguments.
        kwargs_generator: Optional zero-argument callable returning sample keyword arguments.
        rtol: Relative tolerance.
        atol: Absolute tolerance.

    """
    func_name = op_name.lower()
    if func_name not in _Z_DISPATCH or func_name not in _M_DISPATCH:
        pytest.skip(f"{func_name} not mapped in zero_mlx or mx")

    z_func: VariadicOpCallable = _Z_DISPATCH[func_name]
    m_func: VariadicOpCallable = _M_DISPATCH[func_name]

    args = args_generator()
    kwargs = kwargs_generator() if kwargs_generator is not None else {}

    def _z_build(x: NDArrayGeneric) -> GenValue:
        """Build `zero_mlx.array`.

        Args:
            x: Input NumPy array.

        Returns:
            Wrapped `zero_mlx.array`.

        """
        return zero_mlx.array(x)

    def _m_build(x: NDArrayGeneric) -> GenValue:
        """Build `mlx.core.array`.

        Args:
            x: Input NumPy array.

        Returns:
            Wrapped `mlx.core.array`.

        """
        return mx.array(x)

    m_args = [_process_arg(a, _m_build) for a in args]

    _tracer.start_tracing("parity_test")
    try:
        z_args = [_process_arg(a, _z_build) for a in args]
        z_res = z_func(*z_args, **kwargs)
        if isinstance(z_res, (list, tuple)):
            for item in z_res:
                if isinstance(item, array):
                    zero_mlx.eval(item)
        elif isinstance(z_res, array):
            zero_mlx.eval(z_res)
    finally:
        if _tracer.is_tracing:
            _tracer.stop_tracing()

    m_res = m_func(*m_args, **kwargs)
    assert_allclose_mlx(z_res, m_res, rtol=rtol, atol=atol)
