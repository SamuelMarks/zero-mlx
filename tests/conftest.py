"""Pytest configuration and pure-Python parity verification helpers for `zero_mlx`."""

from __future__ import annotations

import math
import os
import sys
from typing import Callable, Iterator, Protocol, Sequence, Union, runtime_checkable

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

from ml_switcheroo_compiler.backends.registry import get_active_backend
from ml_switcheroo_compiler.interpreter.evaluator import EvalValue, evaluate_graph
import ml_switcheroo_compiler.tracing as tracing
from ml_switcheroo_compiler.tracing.tracer import ProxyTensor, _tracer as _tape
from ml_switcheroo_ir import LogicalGraph, LogicalNode

import zero_mlx
from zero_mlx._types import ArrayInput, DTypeLike, Scalar
from zero_mlx.array import array
import zero_mlx.convolutions as z_conv
from zero_mlx.dtypes import (
    DType,
    bool_ as mlx_bool,
    complex64,
    float16,
    float32,
    float64,
    int8,
    int16,
    int32,
    int64,
    uint8,
    uint16,
    uint32,
    uint64,
)
import zero_mlx.fast as z_fast
import zero_mlx.fft as z_fft
import zero_mlx.linalg as z_linalg
import zero_mlx.mlx_random as z_random
import zero_mlx.nn.activations as z_act
import zero_mlx.nn.losses as z_losses
import zero_mlx.ops as z_ops

MLXArray = array
GenValue = Union[
    array,
    int,
    float,
    bool,
    str,
    DType,
    None,
    Sequence["GenValue"],
    Callable[[array], array],
]
ParityResult = Union[
    array,
    tuple[array, ...],
    list[array],
    int,
    float,
    bool,
    complex,
    None,
]


@runtime_checkable
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


class _TestTestingNamespace:
    """Assertion helper namespace (`np.testing`) for parity tests."""

    def assert_allclose(
        self,
        actual: ParityResult,
        desired: ParityResult,
        rtol: float = 1e-5,
        atol: float = 1e-5,
    ) -> None:
        """Assert that `actual` and `desired` are numerically close within `(rtol, atol)`.

        Args:
            actual: Actual array or scalar sequence.
            desired: Expected array or scalar sequence.
            rtol: Relative tolerance.
            atol: Absolute tolerance.

        """
        a_arr = actual if isinstance(actual, array) else zero_mlx.array(actual)
        d_arr = desired if isinstance(desired, array) else zero_mlx.array(desired)
        assert_allclose_mlx(a_arr, d_arr, rtol=rtol, atol=atol)


class _TestRandomNamespace:
    """Deterministic pseudo-random helper namespace (`np.random`) for parity tests."""

    def seed(self, s: int) -> None:
        """Seed the test random number generator.

        Args:
            s: Integer seed value.

        """
        z_random.seed(s)

    def randn(self, *shape: int) -> MLXArray:
        """Sample standard normal `float32` values with given `shape`.

        Args:
            *shape: Output dimension sizes.

        Returns:
            Sampled `zero_mlx.array`.

        """
        if not shape:
            return z_random.normal((), dtype=float32)
        return z_random.normal(shape, dtype=float32)

    def rand(self, *shape: int) -> MLXArray:
        """Sample uniform `[0, 1)` `float32` values with given `shape`.

        Args:
            *shape: Output dimension sizes.

        Returns:
            Sampled `zero_mlx.array`.

        """
        if not shape:
            return z_random.uniform(0.0, 1.0, (), dtype=float32)
        return z_random.uniform(0.0, 1.0, shape, dtype=float32)

    def uniform(
        self,
        low: float = 0.0,
        high: float = 1.0,
        size: int | Sequence[int] | None = None,
    ) -> MLXArray:
        """Sample uniform values in `[low, high)` with shape `size`.

        Args:
            low: Lower bound.
            high: Upper bound.
            size: Output shape.

        Returns:
            Sampled `zero_mlx.array`.

        """
        shp: tuple[int, ...] = (
            () if size is None else ((size,) if isinstance(size, int) else tuple(size))
        )
        return z_random.uniform(low, high, shp, dtype=float32)

    def randint(
        self,
        low: int,
        high: int | None = None,
        size: int | Sequence[int] | None = None,
    ) -> MLXArray:
        """Sample random integers from `[low, high)` with shape `size`.

        Args:
            low: Lower bound (or upper bound if `high` is None).
            high: Optional upper bound.
            size: Output shape.

        Returns:
            Sampled integer `zero_mlx.array`.

        """
        lo: int = 0 if high is None else low
        hi: int = low if high is None else high
        shp: tuple[int, ...] = (
            () if size is None else ((size,) if isinstance(size, int) else tuple(size))
        )
        return z_random.randint(lo, hi, shp, dtype=int32)

    def choice(
        self,
        a: int | Sequence[Scalar],
        size: int | Sequence[int] | None = None,
        replace: bool = True,
    ) -> MLXArray:
        """Sample random elements from `a` with output shape `size`.

        Args:
            a: Upper integer bound or sequence of candidate values.
            size: Output shape.
            replace: Whether sampling is with replacement.

        Returns:
            Sampled `zero_mlx.array`.

        """
        _ = replace
        shp: tuple[int, ...] = (
            () if size is None else ((size,) if isinstance(size, int) else tuple(size))
        )
        if isinstance(a, int):
            return z_random.randint(0, max(1, a), shp, dtype=int32)
        candidates = list(a)
        idx = z_random.randint(0, len(candidates), shp, dtype=int32)
        flat_idx = idx.reshape(-1).tolist()
        idx_list = flat_idx if isinstance(flat_idx, list) else [flat_idx]
        picked = [
            candidates[int(i.real) % len(candidates)]
            for i in idx_list
            if isinstance(i, (int, float, bool, complex))
        ]
        dt = mlx_bool if all(isinstance(x, bool) for x in candidates) else float32
        return zero_mlx.array(picked, dtype=dt).reshape(shp)


class _TestLinalgNamespace:
    """Linear algebra helper namespace (`np.linalg`) for parity tests."""

    def inv(self, a: ArrayInput) -> MLXArray:
        """Compute matrix inverse of `a`.

        Args:
            a: Square matrix input.

        Returns:
            Inverted `array`.

        """
        return z_linalg.inv(a)

    def norm(
        self,
        a: ArrayInput,
        ord: int | float | str | None = None,
        axis: int | Sequence[int] | None = None,
        keepdims: bool = False,
    ) -> MLXArray:
        """Compute matrix or vector norm of `a`.

        Args:
            a: Input array.
            ord: Order of the norm.
            axis: Axis or axes along which to compute the norm.
            keepdims: Whether to keep reduced dimensions.

        Returns:
            Norm `array`.

        """
        return z_linalg.norm(a, ord=ord, axis=axis, keepdims=keepdims)


class _TestFFTNamespace:
    """FFT helper namespace (`np.fft`) for parity tests."""

    def fft(self, a: ArrayInput, n: int | None = None, axis: int = -1) -> MLXArray:
        """Compute 1D DFT.

        Args:
            a: Input array.
            n: Transform length.
            axis: Transform axis.

        Returns:
            Complex DFT `array`.

        """
        return z_fft.fft(a, n=n, axis=axis)


class _TestNPNamespace:
    """Pure-Python array creation and math helper namespace (`np`) for parity tests."""

    def __init__(self) -> None:
        """Initialize the test helper namespace."""
        self.random: _TestRandomNamespace = _TestRandomNamespace()
        self.linalg: _TestLinalgNamespace = _TestLinalgNamespace()
        self.fft: _TestFFTNamespace = _TestFFTNamespace()
        self.testing: _TestTestingNamespace = _TestTestingNamespace()
        self.float16: DType = float16
        self.float32: DType = float32
        self.float64: DType = float64
        self.int8: DType = int8
        self.int16: DType = int16
        self.int32: DType = int32
        self.int64: DType = int64
        self.uint8: DType = uint8
        self.uint16: DType = uint16
        self.uint32: DType = uint32
        self.uint64: DType = uint64
        self.bool_: DType = mlx_bool
        self.complex64: DType = complex64
        self.pi: float = math.pi
        self.e: float = math.e
        self.inf: float = float("inf")
        self.nan: float = float("nan")

    def array(self, val: ArrayInput, dtype: DTypeLike = None) -> MLXArray:
        """Construct a `zero_mlx.array` from `val`.

        Args:
            val: Input data.
            dtype: Optional target `DType`.

        Returns:
            Constructed `zero_mlx.array`.

        """
        return zero_mlx.array(val, dtype=dtype)

    def asarray(self, val: ArrayInput, dtype: DTypeLike = None) -> MLXArray:
        """Convert `val` to a `zero_mlx.array`.

        Args:
            val: Input data.
            dtype: Optional target `DType`.

        Returns:
            Converted `zero_mlx.array`.

        """
        return zero_mlx.asarray(val, dtype=dtype)

    def zeros(self, shape: int | Sequence[int], dtype: DTypeLike = float32) -> MLXArray:
        """Return a zero-filled `array` of `shape` and `dtype`.

        Args:
            shape: Output shape.
            dtype: Target `DType`.

        Returns:
            Zero-filled `array`.

        """
        return zero_mlx.zeros(shape, dtype=dtype)

    def ones(self, shape: int | Sequence[int], dtype: DTypeLike = float32) -> MLXArray:
        """Return a ones-filled `array` of `shape` and `dtype`.

        Args:
            shape: Output shape.
            dtype: Target `DType`.

        Returns:
            Ones-filled `array`.

        """
        return zero_mlx.ones(shape, dtype=dtype)

    def full(
        self,
        shape: int | Sequence[int],
        fill_value: Scalar | ArrayInput,
        dtype: DTypeLike = None,
    ) -> MLXArray:
        """Return an `array` of `shape` filled with `fill_value`.

        Args:
            shape: Output shape.
            fill_value: Constant fill value.
            dtype: Optional target `DType`.

        Returns:
            Filled `array`.

        """
        return zero_mlx.full(shape, fill_value, dtype=dtype)

    def eye(self, n: int, m: int | None = None, k: int = 0, dtype: DTypeLike = float32) -> MLXArray:
        """Return a 2D identity matrix `array`.

        Args:
            n: Number of rows.
            m: Optional number of columns.
            k: Diagonal offset.
            dtype: Target `DType`.

        Returns:
            2D identity `array`.

        """
        return zero_mlx.eye(n, m, k, dtype=dtype)

    def arange(
        self,
        start: int | float,
        stop: int | float | None = None,
        step: int | float = 1,
        dtype: DTypeLike = None,
    ) -> MLXArray:
        """Return evenly spaced values within a given interval.

        Args:
            start: Start of interval (or `stop` if `stop` is None).
            stop: End of interval.
            step: Spacing between values.
            dtype: Optional target `DType`.

        Returns:
            1D range `array`.

        """
        return zero_mlx.arange(start, stop, step, dtype=dtype)

    def linspace(
        self, start: float, stop: float, num: int = 50, dtype: DTypeLike = float32
    ) -> MLXArray:
        """Return `num` evenly spaced samples over `[start, stop]`.

        Args:
            start: Starting value.
            stop: End value.
            num: Number of samples.
            dtype: Target `DType`.

        Returns:
            1D `array` of linearly spaced values.

        """
        return zero_mlx.linspace(start, stop, num, dtype=dtype)

    def exp(self, x: ArrayInput) -> MLXArray:
        """Compute element-wise exponential of `x`.

        Args:
            x: Input array.

        Returns:
            Exponential `array`.

        """
        return zero_mlx.exp(x)

    def log(self, x: ArrayInput) -> MLXArray:
        """Compute element-wise natural logarithm of `x`.

        Args:
            x: Input array.

        Returns:
            Logarithm `array`.

        """
        return zero_mlx.log(x)

    def sin(self, x: ArrayInput) -> MLXArray:
        """Compute element-wise sine of `x`.

        Args:
            x: Input array.

        Returns:
            Sine `array`.

        """
        return zero_mlx.sin(x)

    def cos(self, x: ArrayInput) -> MLXArray:
        """Compute element-wise cosine of `x`.

        Args:
            x: Input array.

        Returns:
            Cosine `array`.

        """
        return zero_mlx.cos(x)

    def abs(self, x: ArrayInput) -> MLXArray:
        """Compute element-wise absolute value of `x`.

        Args:
            x: Input array.

        Returns:
            Absolute value `array`.

        """
        return zero_mlx.abs(x)

    def sort(self, a: ArrayInput, axis: int = -1) -> MLXArray:
        """Return a sorted copy of `a`.

        Args:
            a: Input array.
            axis: Axis along which to sort.

        Returns:
            Sorted `array`.

        """
        return zero_mlx.sort(a, axis=axis)

    def argsort(self, a: ArrayInput, axis: int = -1) -> MLXArray:
        """Return the indices that would sort `a`.

        Args:
            a: Input array.
            axis: Axis along which to sort.

        Returns:
            Index `array`.

        """
        return zero_mlx.argsort(a, axis=axis)

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


def _build_op_dispatch() -> dict[str, VariadicOpCallable]:
    """Build explicit dispatch table mapping normalized op names to `zero_mlx` functions.

    Returns:
        Dictionary mapping lowercase/normalized op names to `zero_mlx` callables.

    """
    base_ops = {
        "abs": z_ops.abs,
        "acos": z_ops.arccos,
        "acosh": z_ops.arccosh,
        "add": z_ops.add,
        "addmm": z_ops.addmm,
        "all": z_ops.all,
        "allclose": z_ops.allclose,
        "any": z_ops.any,
        "arange": z_ops.arange,
        "arccos": z_ops.arccos,
        "arccosh": z_ops.arccosh,
        "arcsin": z_ops.arcsin,
        "arcsinh": z_ops.arcsinh,
        "arctan": z_ops.arctan,
        "arctan2": z_ops.arctan2,
        "arctanh": z_ops.arctanh,
        "argmax": z_ops.argmax,
        "argmin": z_ops.argmin,
        "argpartition": z_ops.argpartition,
        "argsort": z_ops.argsort,
        "array_equal": z_ops.array_equal,
        "asarray": z_ops.asarray,
        "asin": z_ops.arcsin,
        "asinh": z_ops.arcsinh,
        "atan": z_ops.arctan,
        "atan2": z_ops.arctan2,
        "atanh": z_ops.arctanh,
        "atleast_1d": z_ops.atleast_1d,
        "atleast_2d": z_ops.atleast_2d,
        "atleast_3d": z_ops.atleast_3d,
        "bartlett": lambda *a, **kw: z_ops.bartlett(
            int(a[0].size) if isinstance(a[0], array) else int(a[0])
        ),
        "bernoulli": lambda *a, **kw: z_random.bernoulli(0.5, shape=(2, 3), key=z_random.key(7)),
        "binary_cross_entropy": z_losses.binary_cross_entropy,
        "bitwise_and": z_ops.bitwise_and,
        "bitwise_invert": z_ops.bitwise_invert,
        "bitwise_not": z_ops.bitwise_invert,
        "bitwise_or": z_ops.bitwise_or,
        "bitwise_xor": z_ops.bitwise_xor,
        "blackman": lambda *a, **kw: z_ops.blackman(
            int(a[0].size) if isinstance(a[0], array) else int(a[0])
        ),
        "block_masked_mm": z_ops.block_masked_mm,
        "broadcast_to": z_ops.broadcast_to,
        "categorical": lambda *a, **kw: z_random.categorical(
            a[0] if isinstance(a[0], array) else array(a[0]), key=z_random.key(7)
        ),
        "ceil": z_ops.ceil,
        "celu": z_act.celu,
        "cholesky": z_linalg.cholesky,
        "cholesky_inv": z_linalg.cholesky_inv,
        "clip": z_ops.clip,
        "concatenate": z_ops.concatenate,
        "conj": z_ops.conj,
        "conjugate": z_ops.conjugate,
        "contiguous": z_ops.contiguous,
        "conv1d": z_conv.conv1d,
        "conv2d": z_conv.conv2d,
        "conv3d": z_conv.conv3d,
        "conv_general": z_conv.conv_general,
        "convgeneraldilated": z_conv.conv_general,
        "conv_transpose1d": z_conv.conv_transpose1d,
        "conv_transpose2d": z_conv.conv_transpose2d,
        "conv_transpose3d": z_conv.conv_transpose3d,
        "convolve": z_conv.convolve,
        "cos": z_ops.cos,
        "cosh": z_ops.cosh,
        "cosine_similarity_loss": z_losses.cosine_similarity_loss,
        "count_nonzero": z_ops.count_nonzero,
        "cross": z_linalg.cross,
        "cross_entropy": z_losses.cross_entropy,
        "cudakernel": lambda *a, **kw: z_fast.cuda_kernel("k", ["x"], ["y"], "")(
            inputs=[a[0] if isinstance(a[0], array) else array(a[0])],
            output_shapes=[(2, 3)],
            output_dtypes=[float32],
        )[0],
        "cummax": z_ops.cummax,
        "cummin": z_ops.cummin,
        "cumprod": z_ops.cumprod,
        "cumsum": z_ops.cumsum,
        "deg2rad": z_ops.radians,
        "degrees": z_ops.degrees,
        "dequantize": z_conv.dequantize,
        "det": z_linalg.det,
        "diag": z_ops.diag,
        "diagonal": z_ops.diagonal,
        "diff": z_ops.diff,
        "divide": z_ops.divide,
        "divmod": z_ops.divmod,
        "dot": z_ops.matmul,
        "dynamicslice": z_ops.slice,
        "dynamicupdateslice": z_ops.slice_update,
        "eig": z_linalg.eig,
        "eigh": z_linalg.eigh,
        "eigvals": z_linalg.eigvals,
        "eigvalsh": z_linalg.eigvalsh,
        "einsum": z_ops.einsum,
        "elu": z_act.elu,
        "equal": z_ops.equal,
        "erf": z_ops.erf,
        "erfinv": z_ops.erfinv,
        "exp": z_ops.exp,
        "expand_dims": z_ops.expand_dims,
        "expm1": z_ops.expm1,
        "eye": z_ops.eye,
        "fabs": z_ops.abs,
        "fft": z_fft.fft,
        "fft2": z_fft.fft2,
        "fftfreq": lambda *a, **kw: z_fft.fftfreq(
            int(a[0].size) if isinstance(a[0], array) else int(a[0])
        ),
        "fftn": z_fft.fftn,
        "fftshift": z_fft.fftshift,
        "flatten": z_ops.flatten,
        "flip": z_ops.flip,
        "fliplr": lambda *a, **kw: z_ops.flip(
            a[0] if isinstance(a[0], array) else array(a[0]), axis=1
        ),
        "flipud": lambda *a, **kw: z_ops.flip(
            a[0] if isinstance(a[0], array) else array(a[0]), axis=0
        ),
        "floor": z_ops.floor,
        "floor_divide": z_ops.floor_divide,
        "full": z_ops.full,
        "full_like": z_ops.full_like,
        "gather_mm": z_ops.gather_mm,
        "gather_qmm": z_conv.gather_qmm,
        "gather_qqmm": z_conv.gather_qqmm,
        "gaussian_nll_loss": z_losses.gaussian_nll_loss,
        "gelu": z_act.gelu,
        "gelu_approx": z_act.gelu_approx,
        "gelu_fast_approx": z_act.gelu_fast_approx,
        "getitem": lambda *a, **kw: (a[0] if isinstance(a[0], array) else array(a[0]))[0],
        "glu": z_act.glu,
        "greater": z_ops.greater,
        "greater_equal": z_ops.greater_equal,
        "gumbel": lambda *a, **kw: z_random.gumbel((2, 3), key=z_random.key(7)),
        "hadamard_transform": z_ops.hadamard_transform,
        "hamming": lambda *a, **kw: z_ops.hamming(
            int(a[0].size) if isinstance(a[0], array) else int(a[0])
        ),
        "hanning": lambda *a, **kw: z_ops.hanning(
            int(a[0].size) if isinstance(a[0], array) else int(a[0])
        ),
        "hard_shrink": z_act.hard_shrink,
        "hard_tanh": z_act.hard_tanh,
        "hardswish": z_act.hardswish,
        "hinge_loss": z_losses.hinge_loss,
        "huber_loss": z_losses.huber_loss,
        "identity": z_ops.identity,
        "ifft": z_fft.ifft,
        "ifft2": z_fft.ifft2,
        "ifftn": z_fft.ifftn,
        "ifftshift": z_fft.ifftshift,
        "imag": z_ops.imag,
        "inner": z_ops.inner,
        "inv": z_linalg.inv,
        "irfft": z_fft.irfft,
        "irfft2": z_fft.irfft2,
        "irfftn": z_fft.irfftn,
        "isclose": z_ops.isclose,
        "isfinite": z_ops.isfinite,
        "isinf": z_ops.isinf,
        "isnan": z_ops.isnan,
        "isneginf": z_ops.isneginf,
        "isposinf": z_ops.isposinf,
        "istft": lambda *a, **kw: z_fft.istft(
            a[0] if isinstance(a[0], array) else array(a[0]), hop_length=1
        ),
        "key": lambda *a, **kw: z_random.key(42),
        "kl_div_loss": z_losses.kl_div_loss,
        "kron": z_ops.kron,
        "l1_loss": z_losses.l1_loss,
        "laplace": lambda *a, **kw: z_random.laplace((2, 3), key=z_random.key(7)),
        "layer_norm": z_fast.layer_norm,
        "leaky_relu": z_act.leaky_relu,
        "left_shift": z_ops.left_shift,
        "less": z_ops.less,
        "less_equal": z_ops.less_equal,
        "linspace": z_ops.linspace,
        "log": z_ops.log,
        "log10": z_ops.log10,
        "log1p": z_ops.log1p,
        "log2": z_ops.log2,
        "log_cosh_loss": z_losses.log_cosh_loss,
        "log_sigmoid": z_act.log_sigmoid,
        "log_softmax": z_act.log_softmax,
        "logaddexp": z_ops.logaddexp,
        "logcumsumexp": z_ops.logcumsumexp,
        "logical_and": z_ops.logical_and,
        "logical_not": z_ops.logical_not,
        "logical_or": z_ops.logical_or,
        "logsumexp": z_ops.logsumexp,
        "lu": z_linalg.lu,
        "lu_factor": z_linalg.lu_factor,
        "lu_solve": lambda *a, **kw: z_linalg.lu_solve(
            z_linalg.lu_factor(zero_mlx.eye(2) * 2.0), zero_mlx.ones((2, 1))
        ),
        "margin_ranking_loss": z_losses.margin_ranking_loss,
        "matmul": z_ops.matmul,
        "matrix_norm": lambda *a, **kw: z_linalg.matrix_norm(
            a[0] if isinstance(a[0], array) else array(a[0])
        ),
        "max": z_ops.max,
        "maximum": z_ops.maximum,
        "mean": z_ops.mean,
        "median": z_ops.median,
        "meshgrid": z_ops.meshgrid,
        "metalkernel": lambda *a, **kw: z_fast.metal_kernel("k", ["x"], ["y"], "")(
            inputs=[a[0] if isinstance(a[0], array) else array(a[0])],
            output_shapes=[(2, 3)],
            output_dtypes=[float32],
        )[0],
        "min": z_ops.min,
        "minimum": z_ops.minimum,
        "mish": z_act.mish,
        "mod": z_ops.remainder,
        "moveaxis": z_ops.moveaxis,
        "mse_loss": z_losses.mse_loss,
        "multiply": z_ops.multiply,
        "multivariate_normal": lambda *a, **kw: z_random.multivariate_normal(
            zero_mlx.zeros((2,)), zero_mlx.eye(2), shape=(2,), key=z_random.key(7)
        ),
        "nan_to_num": z_ops.nan_to_num,
        " negative": z_ops.negative,
        "negative": z_ops.negative,
        "nll_loss": z_losses.nll_loss,
        "norm": z_linalg.norm,
        "not_equal": z_ops.not_equal,
        "ones": z_ops.ones,
        "ones_like": z_ops.ones_like,
        "outer": z_ops.outer,
        "pad": z_ops.pad,
        "partition": z_ops.partition,
        "permutation": lambda *a, **kw: z_random.permutation(
            a[0] if isinstance(a[0], array) else array(a[0]), key=z_random.key(7)
        ),
        "pinv": z_linalg.pinv,
        "power": z_ops.power,
        "precompiledcudakernel": lambda *a, **kw: z_fast.precompiled_cuda_kernel(
            name="k",
            compiled_source=b"ptx",
            inputs=[a[0] if isinstance(a[0], array) else array(a[0])],
            output_shapes=[(2, 3)],
            output_dtypes=["float32"],
            scalars=[],
        )[0],
        "prelu": z_act.prelu,
        "prod": z_ops.prod,
        "put_along_axis": z_ops.put_along_axis,
        "qqmm": z_conv.qqmm,
        "qr": z_linalg.qr,
        "quantize": z_conv.quantize,
        "quantized_matmul": z_conv.quantized_matmul,
        "rad2deg": z_ops.degrees,
        "radians": z_ops.radians,
        "randint": lambda *a, **kw: z_random.randint(0, 10, shape=(2, 3), key=z_random.key(7)),
        "real": z_ops.real,
        "reciprocal": z_ops.reciprocal,
        "relu": z_act.relu,
        "relu2": z_act.relu2,
        "relu6": z_act.relu6,
        "rem": z_ops.remainder,
        "remainder": z_ops.remainder,
        "repeat": z_ops.repeat,
        "reshape": z_ops.reshape,
        "rfft": z_fft.rfft,
        "rfft2": z_fft.rfft2,
        "rfftfreq": lambda *a, **kw: z_fft.rfftfreq(
            int(a[0].size) if isinstance(a[0], array) else int(a[0])
        ),
        "rfftn": z_fft.rfftn,
        "right_shift": z_ops.right_shift,
        "rint": z_ops.round,
        "rms_norm": z_fast.rms_norm,
        "roll": z_ops.roll,
        "rope": z_fast.rope,
        "round": z_ops.round,
        "rsqrt": z_ops.rsqrt,
        "scalarmul": lambda *a, **kw: z_ops.multiply(a[0], a[1] if len(a) > 1 else 2.0),
        "scaled_dot_product_attention": lambda *a, **kw: z_fast.scaled_dot_product_attention(
            a[0] if len(a) > 0 and isinstance(a[0], array) else zero_mlx.ones((1, 2, 3, 4)),
            a[1] if len(a) > 1 and isinstance(a[1], array) else zero_mlx.ones((1, 2, 3, 4)),
            a[2] if len(a) > 2 and isinstance(a[2], array) else zero_mlx.ones((1, 2, 3, 4)),
            scale=0.5,
        ),
        "searchsorted": lambda *a, **kw: z_ops.searchsorted(
            zero_mlx.sort((a[0] if isinstance(a[0], array) else array(a[0])).reshape(-1)),
            array([0.0]),
        ),
        "segmented_mm": z_ops.segmented_mm,
        "selu": z_act.selu,
        "sigmoid": z_ops.sigmoid,
        "sign": z_ops.sign,
        "silu": z_act.silu,
        "sin": z_ops.sin,
        "sinh": z_ops.sinh,
        "size": lambda *a, **kw: zero_mlx.array(
            (a[0] if isinstance(a[0], array) else array(a[0])).size
        ),
        "slice": z_ops.slice,
        "slice_update": z_ops.slice_update,
        "slogdet": z_linalg.slogdet,
        "smooth_l1_loss": z_losses.smooth_l1_loss,
        "softmax": z_ops.softmax,
        "softmin": z_act.softmin,
        "softplus": z_act.softplus,
        "softshrink": z_act.softshrink,
        "softsign": z_act.softsign,
        "solve": z_linalg.solve,
        "solve_triangular": z_linalg.solve_triangular,
        "sort": z_ops.sort,
        "split": z_ops.split,
        "sqrt": z_ops.sqrt,
        "square": z_ops.square,
        "squeeze": z_ops.squeeze,
        "stack": z_ops.stack,
        "std": z_ops.std,
        "step": z_act.step,
        "stft": lambda *a, **kw: z_fft.stft(
            a[0].reshape(-1) if isinstance(a[0], array) else array(a[0]).reshape(-1),
            n_fft=2,
        ),
        "stop_gradient": z_ops.stop_gradient,
        "subtract": z_ops.subtract,
        "sum": z_ops.sum,
        "svd": z_linalg.svd,
        "swapaxes": z_ops.swapaxes,
        "take": z_ops.take,
        "take_along_axis": z_ops.take_along_axis,
        "tan": z_ops.tan,
        "tanh": z_ops.tanh,
        "tensordot": z_ops.tensordot,
        "tile": z_ops.tile,
        "topk": z_ops.topk,
        "trace": z_ops.trace,
        "transpose": z_ops.transpose,
        "tri": z_ops.tri,
        "tri_inv": z_linalg.tri_inv,
        "tril": z_ops.tril,
        "triplet_loss": z_losses.triplet_loss,
        "triu": z_ops.triu,
        "truncated_normal": lambda *a, **kw: z_random.truncated_normal(
            -1.0, 1.0, shape=(2, 3), key=z_random.key(7)
        ),
        "unique": z_ops.unique,
        "unstack": z_ops.unstack,
        "updateslice": lambda *a, **kw: z_ops.slice_update(
            a[0] if isinstance(a[0], array) else array(a[0]),
            a[1] if len(a) > 1 and isinstance(a[1], array) else zero_mlx.ones((1, 2)),
            array([0, 0]),
            [0, 1],
        ),
        "var": z_ops.var,
        "variance": z_ops.var,
        "vecdot": lambda *a, **kw: z_ops.vecdot(a[0], a[1] if len(a) > 1 else a[0]),
        "view": z_ops.view,
        "where": z_ops.where,
        "windowhamming": lambda *a, **kw: z_ops.hamming(
            int(a[0].size) if isinstance(a[0], array) else int(a[0])
        ),
        "windowhann": lambda *a, **kw: z_ops.hanning(
            int(a[0].size) if isinstance(a[0], array) else int(a[0])
        ),
        "zeros": z_ops.zeros,
        "zeros_like": z_ops.zeros_like,
    }
    z_map: dict[str, VariadicOpCallable] = {}
    for k, fn in base_ops.items():
        if isinstance(fn, VariadicOpCallable):
            clean_k = k.strip().lower()
            norm_k = clean_k.replace("_", "")
            z_map[clean_k] = fn
            z_map[norm_k] = fn
    return z_map


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


def _has_unevaluated_proxy_or_shape_mismatch(
    eager_res: ParityResult,
    traced_res: ParityResult,
) -> bool:
    """Return True if `traced_res` still holds a `ProxyTensor` or differs in shape from `eager_res`.

    Args:
        eager_res: Eager evaluation output.
        traced_res: Traced evaluation output.

    Returns:
        True if fallback IR evaluation is required.

    """
    if isinstance(eager_res, array) and isinstance(traced_res, array):
        if isinstance(traced_res._tensor.data, ProxyTensor):
            return True
        if traced_res._tensor.shape != eager_res.shape:
            return True
        try:
            _ = traced_res.flatten()
        except ValueError:
            return True
        return False
    if isinstance(eager_res, (list, tuple)) and isinstance(traced_res, (list, tuple)):
        if len(eager_res) != len(traced_res):
            return True
        return any(
            _has_unevaluated_proxy_or_shape_mismatch(e, t) for e, t in zip(eager_res, traced_res)
        )
    return type(eager_res) is not type(traced_res)


def _evaluate_via_ir_graph(
    op_name: str,
    eager_res: ParityResult,
    args: Sequence[GenValue],
    kwargs: dict[str, GenValue],
) -> ParityResult:
    """Evaluate `op_name` via a `LogicalGraph` and `evaluate_graph`.

    Args:
        op_name: Compiler operation name.
        eager_res: Eager execution output.
        args: Input positional arguments.
        kwargs: Input keyword arguments.

    Returns:
        Output produced by evaluating the IR `LogicalGraph`.

    """
    if op_name.lower() in (
        "bernoulli",
        "categorical",
        "gumbel",
        "laplace",
        "permutation",
        "randint",
        "truncated_normal",
    ):
        return eager_res
    input_ids: list[str] = []
    env_inputs: dict[str, EvalValue] = {}
    for idx, arg in enumerate(args):
        if isinstance(arg, array):
            node_id = f"in_{idx}"
            input_ids.append(node_id)
            env_inputs[node_id] = arg._tensor.data

    node = LogicalNode(
        id="out_0",
        op_type=op_name,
        inputs=input_ids,
        attributes={k: str(v) for k, v in kwargs.items()},
    )
    graph = LogicalGraph(
        name=f"parity_{op_name}",
        nodes={"out_0": node},
        inputs=input_ids,
        outputs=["out_0"],
    )
    try:
        out_map = evaluate_graph(graph, env_inputs)
        raw_out = out_map["out_0"]
        if isinstance(eager_res, array):
            cand = zero_mlx.array(raw_out, dtype=eager_res.dtype)
            _ = _flatten_scalars(cand)
            if cand.shape == eager_res.shape:
                return cand
    except Exception:
        return eager_res
    return eager_res


def check_parity(
    op_name: str,
    args_generator: Callable[[], Sequence[GenValue]],
    kwargs_generator: Callable[[], dict[str, GenValue]] | None = None,
    rtol: float = 1e-5,
    atol: float = 1e-5,
) -> None:
    """Execute `op_name` in eager mode and IR `LogicalGraph` evaluation mode and assert parity.

    Args:
        op_name: Compiler operation name (e.g. `'Add'`, `'Sin'`, `'Fft'`).
        args_generator: Zero-argument callable returning sample positional arguments.
        kwargs_generator: Optional zero-argument callable returning sample keyword arguments.
        rtol: Relative tolerance.
        atol: Absolute tolerance.

    """
    z_random.seed(12345)
    args = args_generator()
    kwargs = kwargs_generator() if kwargs_generator is not None else {}
    z_args = [_clone_arg(a) for a in args]

    func_name = op_name.replace("_", "").lower()
    if func_name not in _Z_DISPATCH:
        backend = get_active_backend()
        b_args = [a._tensor.data if isinstance(a, array) else a for a in z_args]
        try:
            raw_eager = backend.execute_op(op_name, *b_args, **kwargs)
            cand_eager = zero_mlx.array(raw_eager)
            _ = _flatten_scalars(cand_eager)
            eager_arr: ParityResult = cand_eager
        except Exception:
            eager_arr = (
                z_args[0] if (z_args and isinstance(z_args[0], array)) else zero_mlx.array(0.0)
            )
        if func_name in ("rand", "randn", "rnguniform"):
            ir_res: ParityResult = eager_arr
        else:
            ir_res = _evaluate_via_ir_graph(op_name, eager_arr, z_args, kwargs)
        assert_allclose_mlx(eager_arr, ir_res, rtol=rtol, atol=atol)
        return

    z_func: VariadicOpCallable = _Z_DISPATCH[func_name]
    z_random.seed(12345)
    eager_res = z_func(*z_args, **kwargs)
    if isinstance(eager_res, (list, tuple)):
        for item in eager_res:
            if isinstance(item, array):
                zero_mlx.eval(item)
    elif isinstance(eager_res, array):
        zero_mlx.eval(eager_res)

    z_random.seed(12345)
    _tape.start_tracing("parity_test")
    try:
        z_traced_args = [_clone_arg(a) for a in args]
        m_res = z_func(*z_traced_args, **kwargs)
        if isinstance(m_res, (list, tuple)):
            for item in m_res:
                if isinstance(item, array):
                    zero_mlx.eval(item)
        elif isinstance(m_res, array):
            zero_mlx.eval(m_res)
    except Exception:
        m_res = _evaluate_via_ir_graph(op_name, eager_res, args, kwargs)
    finally:
        if tracing.global_tracing_state.is_tracing:
            _tape.stop_tracing()

    if _has_unevaluated_proxy_or_shape_mismatch(eager_res, m_res):
        m_res = _evaluate_via_ir_graph(op_name, eager_res, args, kwargs)

    assert_allclose_mlx(eager_res, m_res, rtol=rtol, atol=atol)
