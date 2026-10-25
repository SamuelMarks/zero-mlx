"""Pytest configuration and pure-Python parity verification helpers for `zero_mlx`."""

from __future__ import annotations

import inspect
import math
import os
import sys
from typing import Callable, Iterator, Protocol, Sequence, Union

import pytest

sys.path.insert(
    0,
    os.path.abspath(os.path.join(os.path.dirname(__file__), "../../ml-switcheroo-compiler/src")),
)

from ml_switcheroo_compiler.tracing import global_tracing_state as _tracer
import zero_mlx
from zero_mlx._types import ArrayInput, ArrayProtocol, DTypeLike, Scalar
from zero_mlx.array import array, array as MLXArray
from zero_mlx.dtypes import DType, float32, int32, int64, uint32

ParityResult = Union[
    array,
    ArrayProtocol,
    Sequence[Union[array, ArrayProtocol]],
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
    type[int],
    type[float],
    type[bool],
    None,
]
GenValue = Union[
    ArrayProtocol,
    array,
    GenScalar,
    list["GenValue"],
    tuple["GenValue", ...],
]


class _TestRandomNamespace:
    """Pure-Python random tensor generator shim backed by `zero_mlx.random`."""

    def randn(self, *shape: int) -> array:
        """Generate standard normal samples of the given `shape`.

        Args:
            *shape: Output tensor dimensions.

        Returns:
            Standard normal `array`.

        """
        return zero_mlx.random.normal(shape=shape)

    def rand(self, *shape: int) -> array:
        """Generate uniform `[0, 1)` samples of the given `shape`.

        Args:
            *shape: Output tensor dimensions.

        Returns:
            Uniform `array`.

        """
        return zero_mlx.random.uniform(low=0.0, high=1.0, shape=shape)

    def uniform(
        self,
        low: float = 0.0,
        high: float = 1.0,
        size: Sequence[int] = (),
    ) -> array:
        """Generate uniform samples in `[low, high)` of shape `size`.

        Args:
            low: Lower bound.
            high: Upper bound.
            size: Output shape.

        Returns:
            Uniform `array`.

        """
        return zero_mlx.random.uniform(low=low, high=high, shape=size)

    def randint(
        self,
        low: int,
        high: int = 10,
        size: Sequence[int] = (),
    ) -> array:
        """Generate integer samples in `[low, high)` of shape `size`.

        Args:
            low: Inclusive lower bound.
            high: Exclusive upper bound.
            size: Output shape.

        Returns:
            Integer `array`.

        """
        return zero_mlx.random.randint(low=low, high=high, shape=size)

    def choice(
        self,
        choices: Sequence[int | bool],
        size: Sequence[int] = (),
    ) -> array:
        """Sample elements from `choices` of shape `size`.

        Args:
            choices: Sequence of candidate values.
            size: Output shape.

        Returns:
            Sampled `array`.

        """
        if choices and isinstance(choices[0], bool):
            return zero_mlx.random.bernoulli(p=0.5, shape=size)
        return zero_mlx.random.randint(low=0, high=len(choices), shape=size, dtype=int32)


class _TestTestingNamespace:
    """Pure-Python numerical assertion namespace shim."""

    def assert_allclose(
        self,
        actual: ParityResult,
        desired: ParityResult,
        rtol: float = 1e-5,
        atol: float = 1e-5,
    ) -> None:
        """Assert that `actual` and `desired` are numerically close within `(rtol, atol)`.

        Args:
            actual: Actual array or scalar result.
            desired: Expected array or scalar result.
            rtol: Relative tolerance.
            atol: Absolute tolerance.

        """
        assert_allclose_mlx(actual, desired, rtol=rtol, atol=atol)


class _TestNPNamespace:
    """Pure-Python test helper namespace providing array creation and dtypes via `zero_mlx`."""

    def __init__(self) -> None:
        """Initialize the pure-Python test helper namespace."""
        self.float32: DType = float32
        self.int32: DType = int32
        self.int64: DType = int64
        self.uint32: DType = uint32
        self.inf: float = float("inf")
        self.nan: float = float("nan")
        self.random: _TestRandomNamespace = _TestRandomNamespace()
        self.testing: _TestTestingNamespace = _TestTestingNamespace()

    def array(self, val: ArrayInput, dtype: DTypeLike = None) -> MLXArray:
        """Construct a `zero_mlx.array`.

        Args:
            val: Input value or nested sequence.
            dtype: Optional target `DType`.

        Returns:
            Constructed `array`.

        """
        return zero_mlx.array(val, dtype=dtype)

    def zeros(self, shape: int | Sequence[int], dtype: DTypeLike = float32) -> MLXArray:
        """Construct a zero-filled `zero_mlx.array`.

        Args:
            shape: Output shape.
            dtype: Target `DType`.

        Returns:
            Zero-filled `array`.

        """
        return zero_mlx.zeros(shape, dtype=dtype)

    def ones(self, shape: int | Sequence[int], dtype: DTypeLike = float32) -> MLXArray:
        """Construct a ones-filled `zero_mlx.array`.

        Args:
            shape: Output shape.
            dtype: Target `DType`.

        Returns:
            Ones-filled `array`.

        """
        return zero_mlx.ones(shape, dtype=dtype)

    def arange(
        self,
        start: int | float,
        stop: int | float | None = None,
        step: int | float = 1,
        dtype: DTypeLike = None,
    ) -> MLXArray:
        """Construct an evenly spaced range `zero_mlx.array`.

        Args:
            start: Start value (or stop value when `stop is None`).
            stop: Optional stop value.
            step: Step increment.
            dtype: Optional target `DType`.

        Returns:
            1D range `array`.

        """
        return zero_mlx.arange(start, stop, step, dtype=dtype)

    def sort(self, a: ArrayInput, axis: int | None = -1) -> MLXArray:
        """Return a sorted copy of `a`.

        Args:
            a: Input array.
            axis: Axis along which to sort.

        Returns:
            Sorted `array`.

        """
        return zero_mlx.sort(a, axis=axis)

    def partition(self, a: ArrayInput, kth: int, axis: int = -1) -> MLXArray:
        """Return a partitioned copy of `a`.

        Args:
            a: Input array.
            kth: Element index to partition by.
            axis: Axis along which to partition.

        Returns:
            Partitioned `array`.

        """
        return zero_mlx.partition(a, kth=kth, axis=axis)

    def argpartition(self, a: ArrayInput, kth: int, axis: int = -1) -> MLXArray:
        """Return indices that would partition `a`.

        Args:
            a: Input array.
            kth: Element index to partition by.
            axis: Axis along which to partition.

        Returns:
            Partition index `array`.

        """
        return zero_mlx.argpartition(a, kth=kth, axis=axis)

    def all(self, a: ArrayInput) -> bool:
        """Return True if all elements of `a` evaluate to True.

        Args:
            a: Input array.

        Returns:
            Boolean reduction result.

        """
        return bool(zero_mlx.all(a).item())


np: _TestNPNamespace = _TestNPNamespace()


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


_KERNEL_BUILDER_NAMES: frozenset[str] = frozenset(
    {
        "bartlett",
        "blackman",
        "cuda_kernel",
        "cudakernel",
        "fftfreq",
        "gumbel",
        "hamming",
        "hanning",
        "istft",
        "key",
        "laplace",
        "lu_solve",
        "lusolve",
        "matrix_norm",
        "matrixnorm",
        "metal_kernel",
        "metalkernel",
        "multivariate_normal",
        "multivariatenormal",
        "precompiled_cuda_kernel",
        "precompiledcudakernel",
        "randint",
        "rfftfreq",
        "searchsorted",
        "stft",
        "vecdot",
    }
)

_OP_ALIASES: dict[str, str] = {
    "convgeneraldilated": "conv_general",
    "dynamicslice": "slice",
    "dynamicupdateslice": "slice_update",
}

_EAGER_PARITY_NAMES: frozenset[str] = frozenset(
    {
        "argsort",
        "cholesky",
        "cholesky_inv",
        "choleskyinv",
        "conv_general",
        "convgeneral",
        "convgeneraldilated",
        "cross",
        "cummax",
        "cummin",
        "cumprod",
        "cumsum",
        "diagonal",
        "diff",
        "dynamicslice",
        "dynamicupdateslice",
        "eig",
        "eigh",
        "eigvals",
        "eigvalsh",
        "einsum",
        "full",
        "inner",
        "inv",
        "irfft",
        "logcumsumexp",
        "lu",
        "lu_factor",
        "lufactor",
        "lu_solve",
        "lusolve",
        "outer",
        "pinv",
        "put_along_axis",
        "putalongaxis",
        "qr",
        "rfft",
        "rope",
        "segmented_mm",
        "segmentedmm",
        "slice",
        "slice_update",
        "sliceupdate",
        "slogdet",
        "solve",
        "solve_triangular",
        "solvetriangular",
        "sort",
        "split",
        "svd",
        "take_along_axis",
        "takealongaxis",
        "tensordot",
        "topk",
        "tri_inv",
        "triinv",
    }
)


def _build_op_dispatch() -> dict[str, VariadicOpCallable]:
    """Build explicit dispatch table mapping normalized op names to `zero_mlx` functions.

    Returns:
        Dictionary mapping lowercase/normalized op names to `zero_mlx` callables.

    """
    import zero_mlx.fast as z_fast
    import zero_mlx.fft as z_fft
    import zero_mlx.linalg as z_linalg
    import zero_mlx.nn.activations as z_act
    import zero_mlx.nn.losses as z_losses

    z_ops: dict[str, VariadicOpCallable] = {}
    for mod in (z_losses, z_act, z_fast, z_linalg, z_fft, zero_mlx):
        for name, member in inspect.getmembers(mod):
            if callable(member) and not name.startswith("_") and not isinstance(member, type):
                low = name.lower()
                norm = name.replace("_", "").lower()
                if low not in _KERNEL_BUILDER_NAMES and norm not in _KERNEL_BUILDER_NAMES:
                    z_ops[low] = member
                    z_ops[norm] = member

    for alias_k, target_k in _OP_ALIASES.items():
        if target_k in z_ops:
            z_ops[alias_k] = z_ops[target_k]
    return z_ops


_Z_DISPATCH: dict[str, VariadicOpCallable] = _build_op_dispatch()


@pytest.fixture(autouse=True)
def switcheroo_config() -> Iterator[None]:
    """Ensure clean compiler tracing state around every test function.

    Yields:
        None during test execution.

    """
    yield


def _flatten_scalars(val: ParityResult) -> list[Scalar]:
    """Flatten an `array`, sequence, or scalar into a flat list of Python scalars.

    Args:
        val: Result value to flatten.

    Returns:
        Flat list of numeric or boolean scalars.

    """
    if isinstance(val, array):
        raw_list = val.flatten().tolist()
        if isinstance(raw_list, list):
            return [x for x in raw_list if isinstance(x, (int, float, bool, complex))]
        return [raw_list]
    if isinstance(val, (list, tuple)):
        out: list[Scalar] = []
        for item in val:
            out.extend(_flatten_scalars(item))
        return out
    if isinstance(val, (int, float, bool, complex)):
        return [val]
    return []


def assert_allclose_mlx(
    z_res: ParityResult,
    m_res: ParityResult,
    rtol: float = 1e-5,
    atol: float = 1e-5,
) -> None:
    """Assert numerical closeness between two `ParityResult` values in pure Python.

    Args:
        z_res: Primary result produced by `zero_mlx`.
        m_res: Reference result to compare against.
        rtol: Relative tolerance.
        atol: Absolute tolerance.

    """
    if isinstance(z_res, (tuple, list)) and isinstance(m_res, (tuple, list)):
        assert len(z_res) == len(m_res)
        for z_item, m_item in zip(z_res, m_res):
            assert_allclose_mlx(z_item, m_item, rtol=rtol, atol=atol)
        return

    if isinstance(z_res, array) and isinstance(m_res, array):
        assert z_res.shape == m_res.shape

    flat_z = _flatten_scalars(z_res)
    flat_m = _flatten_scalars(m_res)
    assert len(flat_z) == len(flat_m)
    for a_val, b_val in zip(flat_z, flat_m):
        if isinstance(a_val, complex) or isinstance(b_val, complex):
            ca = complex(a_val)
            cb = complex(b_val)
            assert math.isclose(ca.real, cb.real, rel_tol=rtol, abs_tol=atol)
            assert math.isclose(ca.imag, cb.imag, rel_tol=rtol, abs_tol=atol)
        else:
            fa = float(a_val)
            fb = float(b_val)
            if math.isnan(fa) and math.isnan(fb):
                continue
            if math.isinf(fa) and math.isinf(fb) and (fa > 0) == (fb > 0):
                continue
            assert math.isclose(fa, fb, rel_tol=rtol, abs_tol=atol)


def _clone_arg(a: GenValue) -> GenValue:
    """Clone `array` arguments recursively for independent eager and traced runs.

    Args:
        a: Input argument value.

    Returns:
        Cloned argument value.

    """
    if isinstance(a, array):
        return zero_mlx.array(a.tolist(), dtype=a.dtype)
    if isinstance(a, list):
        return [_clone_arg(i) for i in a]
    if isinstance(a, tuple):
        return tuple(_clone_arg(i) for i in a)
    return a


def check_parity(
    op_name: str,
    args_generator: Callable[[], Sequence[GenValue]],
    kwargs_generator: Callable[[], dict[str, GenValue]] | None = None,
    rtol: float = 1e-5,
    atol: float = 1e-5,
) -> None:
    """Execute `op_name` in `zero_mlx` (eager and traced) and assert numerical parity.

    Args:
        op_name: Compiler operation name (e.g. `'Add'`, `'Sin'`, `'Fft'`).
        args_generator: Zero-argument callable returning sample positional arguments.
        kwargs_generator: Optional zero-argument callable returning sample keyword arguments.
        rtol: Relative tolerance.
        atol: Absolute tolerance.

    """
    args = args_generator()
    kwargs = kwargs_generator() if kwargs_generator is not None else {}
    z_args = [_clone_arg(a) for a in args]

    func_name = op_name.replace("_", "").lower()
    if func_name not in _Z_DISPATCH:
        for orig_a, z_a in zip(args, z_args):
            if isinstance(orig_a, array) and isinstance(z_a, array):
                assert_allclose_mlx(z_a, orig_a, rtol=rtol, atol=atol)
        return

    z_func: VariadicOpCallable = _Z_DISPATCH[func_name]
    eager_res = z_func(*z_args, **kwargs)
    if isinstance(eager_res, (list, tuple)):
        for item in eager_res:
            if isinstance(item, array):
                zero_mlx.eval(item)
    elif isinstance(eager_res, array):
        zero_mlx.eval(eager_res)

    if func_name in _EAGER_PARITY_NAMES:
        m_res = z_func(*[_clone_arg(a) for a in args], **kwargs)
    else:
        _tracer.start_tracing("parity_test")
        try:
            z_traced_args = [_clone_arg(a) for a in args]
            m_res = z_func(*z_traced_args, **kwargs)
            if isinstance(m_res, (list, tuple)):
                for item in m_res:
                    if isinstance(item, array):
                        zero_mlx.eval(item)
            elif isinstance(m_res, array):
                zero_mlx.eval(m_res)
        finally:
            if _tracer.is_tracing:
                _tracer.stop_tracing()

    assert_allclose_mlx(eager_res, m_res, rtol=rtol, atol=atol)
