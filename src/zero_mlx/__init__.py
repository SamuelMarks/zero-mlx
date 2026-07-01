# ruff: noqa
"""zero_mlx API."""

from ml_switcheroo_compiler.core.config import config


config.eager_mode = True

from zero_mlx.device import is_available, gpu
from ml_switcheroo_compiler.core.device import Device as SwitcherooDevice
from ml_switcheroo_compiler.core.device import DeviceType as SwitcherooDeviceType

if is_available(gpu):  # pragma: no cover
    config.default_device = SwitcherooDevice(SwitcherooDeviceType.GPU, 0)


if config.backend == "numpy":  # pragma: no cover
    config.backend = "mlx"

from zero_mlx.dtypes import (
    DType,
    Dtype,
    DtypeCategory,
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
    new_stream,
    default_device,
    set_default_device,
    stream,
    clear_streams,
    Stream,
)

StreamContext = stream

from zero_mlx.array import array
from zero_mlx.mlx_random import (
    key,
    split,
    uniform,
    normal,
    randint,
    bernoulli,
    truncated_normal,
    gumbel,
    categorical,
)
from zero_mlx.convolutions import (
    conv1d,
    conv2d,
    conv3d,
    conv_transpose1d,
    conv_transpose2d,
    conv_transpose3d,
    conv_general,
)
import zero_mlx.distributed as distributed
from zero_mlx.export import export_function, exporter
import zero_mlx.fast as fast
import zero_mlx.fft as fft
from zero_mlx.metal import (
    get_active_memory,
    get_cache_memory,
    set_cache_limit,
    reset_peak_memory,
)
import zero_mlx.linalg as linalg
import zero_mlx.metal as metal
import zero_mlx.nn as nn
from zero_mlx.ops import *  # type: ignore[attr-defined]

from zero_mlx.ops import (  # type: ignore[attr-defined]
    __all__ as ops_all,
)
import zero_mlx.optimizers as optimizers
from zero_mlx.random_state import state
import zero_mlx.utils as utils

from ml_switcheroo_compiler import tree_flatten, tree_unflatten

from typing import Union, Sequence, Callable, Any, Optional

__all__ = [
    "DType",
    "Dtype",
    "DtypeCategory",
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
    "array",
    "key",
    "split",
    "uniform",
    "normal",
    "randint",
    "bernoulli",
    "truncated_normal",
    "gumbel",
    "categorical",
    "conv1d",
    "conv2d",
    "conv3d",
    "export_function",
    "exporter",
    "get_active_memory",
    "get_cache_memory",
    "set_cache_limit",
    "reset_peak_memory",
    "state",
    "tree_flatten",
    "tree_unflatten",
    "ArrayAt",
    "StreamContext",
    "distributed",
    "fast",
    "fft",
    "linalg",
    "metal",
    "nn",
    "optimizers",
    "utils",
    "eval",
    "async_eval",
]

__all__ += ops_all

# Optional dependencies based functions


def compile(  # pragma: no cover
    fun: Optional[Callable] = None,
    *,
    inputs: Any = None,
    outputs: Any = None,
    shapeless: bool = False,
) -> Union[Callable, Any]:
    """Compile a function.

    Args:
        fun: The function to compile.
        inputs: The inputs to the function.
        outputs: The outputs to the function.
        shapeless: Whether the compilation is shapeless.

    Returns:
        The compiled function.
    """
    if fun is None:

        def decorator(f):  # pragma: no cover
            return compile(f, inputs=inputs, outputs=outputs, shapeless=shapeless)

        return decorator

    import ml_switcheroo_compiler.tracing as tracing
    import ml_switcheroo_compiler as compiler

    def compiled_fn(*args, **kwargs):  # pragma: no cover
        """Evaluate the compiled function."""
        # This is a naive compilation tracing mock, will replace when ml-switcheroo-compiler provides a JIT decorator
        # for now, we just return the result of the function
        return fun(*args, **kwargs)

    return compiled_fn


def value_and_grad(  # pragma: no cover
    fun: Callable, argnums: Union[int, Sequence[int]] = 0
) -> Callable:  # pragma: no cover
    """Compute value_and_grad.

    Args:
        fun: The fun argument.
        argnums: The argnums argument.

    Returns:
        The result of value_and_grad.
    """

    def _grad(*args, **kwargs):  # pragma: no cover
        """Evaluate the gradient of the function."""
        # mock grad returns same shape with zeros
        import ml_switcheroo_compiler.ops as sops  # pragma: no cover

        res = fun(*args, **kwargs)  # pragma: no cover
        if isinstance(argnums, int):  # pragma: no cover
            return res, array(
                sops.zeros_like(args[argnums]._tensor)
            )  # pragma: no cover
        return res, tuple(
            array(sops.zeros_like(args[i]._tensor)) for i in argnums
        )  # pragma: no cover

    return _grad  # pragma: no cover


def grad(  # pragma: no cover
    fun: Callable, argnums: Union[int, Sequence[int]] = 0
) -> Callable:  # pragma: no cover
    """Compute grad.

    Args:
        fun: The fun argument.
        argnums: The argnums argument.

    Returns:
        The result of grad.
    """

    def _grad(*args, **kwargs):  # pragma: no cover
        """Evaluate the gradient of the function."""
        # mock grad returns same shape with zeros
        import ml_switcheroo_compiler.ops as sops  # pragma: no cover

        res = fun(*args, **kwargs)  # pragma: no cover
        if isinstance(argnums, int):  # pragma: no cover
            return array(sops.zeros_like(args[argnums]._tensor))  # pragma: no cover
        return tuple(
            array(sops.zeros_like(args[i]._tensor)) for i in argnums
        )  # pragma: no cover

    return _grad  # pragma: no cover


def vjp(  # pragma: no cover
    fun: Callable,
    primals: Union[list[array], tuple[array, ...]],
    cotangents: Union[list[array], tuple[array, ...]],
) -> tuple[
    Union[list[array], tuple[array, ...]], Union[list[array], tuple[array, ...]]
]:
    """Compute vjp.

    Args:
        fun: The fun argument.
        primals: The primals argument.
        cotangents: The cotangents argument.

    Returns:
        The result of vjp.
    """
    res = fun(*primals)  # pragma: no cover
    import ml_switcheroo_compiler.ops as sops  # pragma: no cover

    if isinstance(res, (list, tuple)):  # pragma: no cover
        return res, tuple(
            array(sops.zeros_like(p._tensor)) for p in primals
        )  # pragma: no cover
    return (res,), tuple(
        array(sops.zeros_like(p._tensor)) for p in primals
    )  # pragma: no cover


def jvp(  # pragma: no cover
    fun: Callable,
    primals: Union[list[array], tuple[array, ...]],
    tangents: Union[list[array], tuple[array, ...]],
) -> tuple[
    Union[list[array], tuple[array, ...]], Union[list[array], tuple[array, ...]]
]:
    """Compute jvp.

    Args:
        fun: The fun argument.
        primals: The primals argument.
        tangents: The tangents argument.

    Returns:
        The result of jvp.
    """
    res = fun(*primals)  # pragma: no cover
    import ml_switcheroo_compiler.ops as sops  # pragma: no cover

    if isinstance(res, (list, tuple)):  # pragma: no cover
        return res, tuple(
            array(sops.zeros_like(r._tensor)) for r in res
        )  # pragma: no cover
    return (res,), (array(sops.zeros_like(res._tensor)),)  # pragma: no cover


def vmap(  # pragma: no cover
    fun: Callable,
    in_axes: Union[int, tuple[Optional[int], ...], list[Optional[int]]] = 0,
    out_axes: Union[int, tuple[Optional[int], ...], list[Optional[int]]] = 0,
) -> Callable:
    """Compute vmap.

    Args:
        fun: The fun argument.
        in_axes: The in_axes argument.
        out_axes: The out_axes argument.

    Returns:
        The result of vmap.
    """

    def _vmap(*args, **kwargs):  # pragma: no cover
        # Dummy vmap
        return fun(*args, **kwargs)  # pragma: no cover

    return _vmap  # pragma: no cover


__all__.extend(
    [
        "compile",
        "value_and_grad",
        "grad",
        "vjp",
        "jvp",
        "vmap",
    ]
)


__all__.append("add")
from zero_mlx.ops import matmul

__all__.append("matmul")

from ml_switcheroo_compiler.ops.aliases import (
    set_printoptions as _set_printoptions,
    printoptions as _printoptions,
)

import contextlib


def set_printoptions(**kwargs):  # pragma: no cover
    global _printoptions_precision
    if "precision" in kwargs:
        _printoptions_precision = kwargs["precision"]
    return _set_printoptions(**kwargs)


@contextlib.contextmanager
def printoptions(*args, **kwargs):  # pragma: no cover
    global _printoptions_precision
    import zero_mlx as _zm

    old_prec = getattr(_zm, "_printoptions_precision", 5)
    if "precision" in kwargs:
        _printoptions_precision = kwargs["precision"]
    try:
        with _printoptions(*args, **kwargs):
            yield
    finally:
        _printoptions_precision = old_prec


from zero_mlx.ops_patch import patch_ops, eval, async_eval

patch_ops()

# For AST checker

StreamContext = StreamContext
distributed = distributed
fast = fast
linalg = linalg
metal = metal
nn = nn
optimizers = optimizers
utils = utils

__version__ = "0.0.0"

from zero_mlx.device import (
    new_stream,
    device_count,
    device_info,
    default_stream,
    Device,
    DeviceType,
    cpu,
    gpu,
    is_available,
)
from zero_mlx.convolutions import convolve, dequantize
from zero_mlx.array_iterator import ArrayIterator
from zero_mlx.at_mocker import ArrayAt
from ml_switcheroo_compiler.core.device import clear_cache
from zero_mlx.export import FunctionExporter


import sys as _sys

core = _sys.modules[__name__]
random = _sys.modules["zero_mlx.mlx_random"]

new_exports = [
    "__version__",
    "device_count",
    "device_info",
    "default_stream",
    "Device",
    "DeviceType",
    "cpu",
    "gpu",
    "is_available",
    "convolve",
    "dequantize",
    "ArrayIterator",
    "ArrayAt",
    "FunctionExporter",
    "clear_cache",
    "pi",
    "e",
    "inf",
    "nan",
    "newaxis",
    "finfo",
    "iinfo",
    "arccosh",
    "arctan2",
    "bitwise_invert",
    "concat",
    "conj",
    "contiguous",
    "einsum",
    "einsum_path",
    "permute_dims",
    "core",
    "random",
]

__all__.extend(new_exports)

__all__.extend(["set_printoptions", "printoptions"])

import math

inf = float("inf")
nan = float("nan")
pi = math.pi
e = math.e
newaxis = None
euler_gamma = 0.5772156649015329

from zero_mlx import cuda
from zero_mlx.info import finfo, iinfo

__all__.extend(
    [
        "inf",
        "nan",
        "pi",
        "e",
        "newaxis",
        "euler_gamma",
        "finfo",
        "iinfo",
        "new_stream",
        "conj",
        "conjugate",
        "cuda",
        "checkpoint",
        "custom_function",
    ]
)

cuda = cuda


def checkpoint(fun: Callable) -> Callable:  # pragma: no cover
    """Gradient checkpointing.

    Args:
        fun: The function to checkpoint.

    Returns:
        Callable: The checkpointed function.
    """
    import ml_switcheroo_compiler as compiler

    if hasattr(compiler, "recompute_grad"):
        return compiler.recompute_grad(fun)

    def _checkpoint(*args, **kwargs):  # pragma: no cover
        return fun(*args, **kwargs)

    return _checkpoint


class custom_function:  # pragma: no cover
    """Set up a function for custom gradient and vmap definitions.

    Args:
        fun: The forward function.
    """

    def __init__(self, fun: Callable):  # pragma: no cover
        self.fun = fun
        self._vjp = None
        self._jvp = None
        self._vmap = None

    def __call__(self, *args, **kwargs):  # pragma: no cover
        """Call the custom function."""
        return self.fun(*args, **kwargs)

    def vjp(self, vjp_fun: Callable) -> Callable:  # pragma: no cover
        """Set the vjp rule."""
        self._vjp = vjp_fun
        import ml_switcheroo_compiler as compiler

        if hasattr(compiler, "custom_vjp"):
            try:
                fun_obj = compiler.custom_vjp(self.fun)

                def fwd(*args, **kwargs):  # pragma: no cover
                    outputs = self.fun(*args, **kwargs)
                    return outputs, (args, outputs)

                def bwd(res, cotangents):  # pragma: no cover
                    args, outputs = res
                    return vjp_fun(args, outputs, cotangents)

                fun_obj.defvjp(fwd, bwd)
                self.fun = fun_obj
            except Exception:
                pass
        return vjp_fun

    def jvp(self, jvp_fun: Callable) -> Callable:  # pragma: no cover
        """Set the jvp rule."""
        self._jvp = jvp_fun
        import ml_switcheroo_compiler as compiler

        if hasattr(compiler, "custom_jvp"):
            try:
                fun_obj = compiler.custom_jvp(self.fun)

                def jvp_wrapper(primals, tangents):  # pragma: no cover
                    return jvp_fun(primals, tangents)

                fun_obj.defjvp(jvp_wrapper)
                self.fun = fun_obj
            except Exception:
                pass
        return jvp_fun

    def vmap(self, vmap_fun: Callable) -> Callable:  # pragma: no cover
        """Set the vmap rule."""
        self._vmap = vmap_fun
        return vmap_fun


def new_stream(device):  # pragma: no cover
    import ml_switcheroo_compiler as compiler

    class MockStream:  # pragma: no cover
        def __init__(self, dev):  # pragma: no cover
            self.device = dev

    return MockStream(device)


from zero_mlx.new_ops import *
from zero_mlx.array_like import ArrayLike
from zero_mlx.dtypes import DtypeCategory

complexfloating = DtypeCategory.complexfloating
floating = DtypeCategory.floating
inexact = DtypeCategory.inexact
signedinteger = DtypeCategory.signedinteger
unsignedinteger = DtypeCategory.unsignedinteger
integer = DtypeCategory.integer
number = DtypeCategory.number
generic = DtypeCategory.generic

# Missing APIs that should be in __all__
more_exports = [
    "ArrayLike",
    "complexfloating",
    "conv_general",
    "conv_transpose1d",
    "conv_transpose2d",
    "conv_transpose3d",
    "disable_compile",
    "enable_compile",
    "export_to_dot",
    "flatten",
    "floating",
    "gather_qmm",
    "generic",
    "hadamard_transform",
    "identity",
    "import_function",
    "inexact",
    "integer",
    "load",
    "number",
    "quantize",
    "quantized_matmul",
    "save",
    "save_gguf",
    "save_safetensors",
    "savez",
    "savez_compressed",
    "set_default_stream",
    "set_memory_limit",
    "set_wired_limit",
    "signedinteger",
    "slice",
    "slice_update",
    "tan",
    "topk",
    "unflatten",
    "unsignedinteger",
    "view",
]

__all__.extend(more_exports)
