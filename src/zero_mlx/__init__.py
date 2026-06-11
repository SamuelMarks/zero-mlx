# ruff: noqa
"""zero_mlx API."""

from zero_mlx.dtypes import (
    DType,
    bool_,
    uint8,
    uint16,
    uint32,
    uint64,
    int8,
    int16,
    int32,
    int64,
    float16,
    bfloat16,
    float32,
    float64,
    complex64,
    complex128,
)
from zero_mlx.device import (
    default_device,
    set_default_device,
    stream,
    clear_streams,
    Stream,
)
from zero_mlx.info import finfo, iinfo
from zero_mlx.array import array
from zero_mlx.ops import (  # type: ignore[attr-defined]
    all,
    any,
    allclose,
    array_equal,
    asarray,
    broadcast_to,
    zeros,
    ones,
    arange,
    synchronize,
    reshape,
    split,
    divmod,
    square,
    sqrt,
    rsqrt,
    reciprocal,
    exp,
    log,
    sin,
    cos,
    log1p,
    abs,
    log10,
    log2,
    conj,  # type: ignore[attr-defined]
    transpose,
    sum,
    prod,
    min,
    max,
    logcumsumexp,
    logsumexp,
    mean,
    var,
    std,
    argmin,
    argmax,
    cummax,
    cummin,
    cumprod,
    cumsum,
    diagonal,
    flatten,  # type: ignore[attr-defined]
    moveaxis,
    round,
    swapaxes,
    stack,
    concatenate,
    get_peak_memory,
    random,
    full,
    expand_dims,
    squeeze,
    astype,  # type: ignore[attr-defined]
    block_until_ready,  # type: ignore[attr-defined]
)
from zero_mlx import metal

__version__ = "0.0.0"


class core:
    """Core namespace."""

    printoptions_precision = 5

    @staticmethod
    def eval(*args: "array") -> tuple:
        """Evaluate."""
        return args  # pragma: no cover


def eval(*args: "array") -> tuple:
    """Evaluate."""
    return args  # pragma: no cover


def grad(fn, argnums=0):
    """Compute grad.

    Args:
        fn: The fn argument.
        argnums: The argnums argument.

    Returns:
        The result of grad.
    """

    def _grad(*args, **kwargs):  # pragma: no cover
        """Evaluate the gradient of the function."""
        # mock grad returns same shape with zeros
        import ml_switcheroo.ops as sops  # pragma: no cover

        res = fn(*args, **kwargs)  # pragma: no cover
        if isinstance(argnums, int):  # pragma: no cover
            return array(sops.zeros_like(args[argnums]._tensor))  # pragma: no cover
        return tuple(
            array(sops.zeros_like(args[i]._tensor)) for i in argnums
        )  # pragma: no cover

    return _grad  # pragma: no cover


class printoptions:
    """Context manager for setting print options."""

    def __init__(self, **kwargs):
        """Initialize the print options context manager."""
        self.kwargs = kwargs
        self.old_precision = core.printoptions_precision

    def __enter__(self):
        """Enter the print options context."""
        if "precision" in self.kwargs:
            core.printoptions_precision = self.kwargs["precision"]

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit the print options context."""
        core.printoptions_precision = self.old_precision


__all__ = [
    "DType",
    "bool_",
    "uint8",
    "uint16",
    "uint32",
    "uint64",
    "int8",
    "int16",
    "int32",
    "int64",
    "float16",
    "bfloat16",
    "float32",
    "float64",
    "complex64",
    "complex128",
    "default_device",
    "set_default_device",
    "stream",
    "clear_streams",
    "Stream",
    "finfo",
    "iinfo",
    "array",
    "all",
    "any",
    "allclose",
    "array_equal",
    "asarray",
    "broadcast_to",
    "zeros",
    "ones",
    "arange",
    "synchronize",
    "reshape",
    "split",
    "divmod",
    "square",
    "sqrt",
    "rsqrt",
    "reciprocal",
    "exp",
    "log",
    "sin",
    "cos",
    "log1p",
    "abs",
    "log10",
    "log2",
    "conj",
    "transpose",
    "sum",
    "prod",
    "min",
    "max",
    "logcumsumexp",
    "logsumexp",
    "mean",
    "var",
    "std",
    "argmin",
    "argmax",
    "cummax",
    "cummin",
    "cumprod",
    "cumsum",
    "diagonal",
    "flatten",
    "moveaxis",
    "round",
    "swapaxes",
    "stack",
    "concatenate",
    "get_peak_memory",
    "random",
    "grad",
    "printoptions",
    "full",
    "squeeze",
    "expand_dims, squeeze",
    "astype",
    "block_until_ready",
    "core",
    "eval",
    "__version__",
    "metal",
]
from zero_mlx.at_mocker import AtMocker

array.at = property(lambda self: AtMocker(self))  # type: ignore[attr-defined]
from zero_mlx.ops import zeros_like, ones_like

__all__.extend(["zeros_like", "ones_like"])
from zero_mlx.ops import maximum, minimum

__all__.extend(["maximum", "minimum"])
from zero_mlx.device import default_stream, Device

__all__.extend(["default_stream", "Device"])
from zero_mlx.ops import add

__all__.append("add")
from zero_mlx.ops import matmul

__all__.append("matmul")
from zero_mlx.array_interface import inject_interface

inject_interface(array)
from zero_mlx.ops_patch import patch_ops

patch_ops()


def set_printoptions(precision=None, threshold=None):
    """Set print options globally.

    Args:
        precision: Number of digits of precision.
        threshold: Total number of array elements which trigger summarization.
    """
    if precision is not None:
        core.printoptions_precision = precision


__all__.append("set_printoptions")
from zero_mlx.array_repr import inject_repr

inject_repr(array)
import zero_mlx.mlx_random as random  # type: ignore[assignment]


e = 2.718281828459045
pi = 3.141592653589793
inf = float("inf")
nan = float("nan")
newaxis = None
__all__.extend(["e", "pi", "inf", "nan", "newaxis"])

euler_gamma = 0.5772156649015328606065120900824024310421
__all__.append("euler_gamma")

from zero_mlx.device import is_available, cpu, gpu

__all__.extend(["is_available", "cpu", "gpu"])

from zero_mlx.device import (
    new_stream,
    device_count,
    device_info,
)

__all__.extend(
    [
        "is_available",
        "cpu",
        "gpu",
        "new_stream",
        "device_count",
        "device_info",
        "stream",
    ]
)

from zero_mlx.ops import subtract, divide, multiply

__all__.extend(["subtract", "divide", "multiply"])

from zero_mlx import linalg

__all__.append("linalg")

from zero_mlx.ops import eye, diag, tril, triu

__all__.extend(["eye", "diag", "tril", "triu"])

from zero_mlx.ops import take_along_axis

__all__.extend(["take_along_axis"])

from zero_mlx import fft

__all__.append("fft")

from zero_mlx.ops import tile, sort, argsort

__all__.extend(["tile"])

from zero_mlx.ops import take, addmm, gather_mm, block_masked_mm, segmented_mm

__all__.extend(["take", "addmm", "gather_mm", "block_masked_mm", "segmented_mm"])

from zero_mlx.ops import as_strided

__all__.extend(["as_strided"])

from zero_mlx.ops import (  # type: ignore[attr-defined]
    argpartition,
    atleast_1d,
    atleast_2d,
    atleast_3d,
    broadcast_arrays,
    broadcast_shapes,
    ceil,
    clip,
    conjugate,
    degrees,
    erf,
    erfinv,
    expm1,
    floor,
    inner,
    isclose,
    isfinite,
    isinf,
    isnan,
    isneginf,
    isposinf,
    issubdtype,
    kron,
    linspace,
    logaddexp,
    logical_and,
    logical_not,
    logical_or,
    median,
    meshgrid,
    nan_to_num,
    negative,
    outer,
    pad,
    partition,
    put_along_axis,
    radians,
    real,
    remainder,
    repeat,
    roll,
    sign,
    tensordot,
    trace,
    tri,
    where,
    sigmoid,
    softmax,
    stop_gradient,
    depends,
    to_fp8,
    from_fp8,
)

__all__.extend(
    [
        "argpartition",
        "atleast_1d",
        "atleast_2d",
        "atleast_3d",
        "broadcast_arrays",
        "broadcast_shapes",
        "ceil",
        "clip",
        "conjugate",
        "degrees",
        "diagonal",
        "erf",
        "erfinv",
        "expm1",
        "floor",
        "inner",
        "isclose",
        "isfinite",
        "isinf",
        "isnan",
        "isneginf",
        "isposinf",
        "issubdtype",
        "kron",
        "linspace",
        "logaddexp",
        "logical_and",
        "logical_not",
        "logical_or",
        "median",
        "meshgrid",
        "nan_to_num",
        "negative",
        "outer",
        "pad",
        "partition",
        "put_along_axis",
        "radians",
        "real",
        "remainder",
        "repeat",
        "roll",
        "sign",
        "tensordot",
        "trace",
        "tri",
        "where",
        "sigmoid",
        "softmax",
        "stop_gradient",
        "depends",
        "to_fp8",
        "from_fp8",
    ]
)

from zero_mlx.ops import (  # type: ignore[attr-defined]
    bitwise_and,
    bitwise_or,
    bitwise_xor,
    left_shift,
    right_shift,
    less,
    less_equal,
    greater,
    greater_equal,
    equal,
    not_equal,
    power,
    arcsin,
    arccos,
    arctan,
    arcsinh,
    arctanh,
    sinh,
    cosh,
    tanh,
    floor_divide,
)

__all__.extend(
    [
        "bitwise_and",
        "bitwise_or",
        "bitwise_xor",
        "left_shift",
        "right_shift",
        "less",
        "less_equal",
        "greater",
        "greater_equal",
        "equal",
        "not_equal",
        "power",
        "arcsin",
        "arccos",
        "arctan",
        "arcsinh",
        "arctanh",
        "sinh",
        "cosh",
        "tanh",
        "floor_divide",
    ]
)

from zero_mlx.ops_patch import vjp, jvp, value_and_grad, custom_function, checkpoint

__all__.extend(["vjp", "jvp", "value_and_grad", "custom_function", "checkpoint"])

from zero_mlx.ops_patch import (
    vmap,
    compile,
    disable_compile,
    enable_compile,
    eval,
    async_eval,
    export_to_dot,
)

__all__.extend(
    [
        "vmap",
        "compile",
        "disable_compile",
        "enable_compile",
        "eval",
        "async_eval",
        "export_to_dot",
    ]
)


class CudaMock:
    """Mock for CUDA namespace."""

    @staticmethod
    def is_available():
        """Check if CUDA is available."""
        return False  # pragma: no cover


cuda = CudaMock()
__all__.append("cuda")
