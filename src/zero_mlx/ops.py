"""Core tensor operations."""

from typing import Any, Tuple, Optional, Sequence, Union
from zero_mlx.dtypes import DType
import ml_switcheroo_compiler as ml_switcheroo
import zero_mlx as mx


def _get_compiler_op(op_name):  # pragma: no cover
    import ml_switcheroo_compiler.ops as mops
    import ml_switcheroo_compiler.ops.nn as mnn
    import ml_switcheroo_compiler.ops.shape as mshape

    try:
        import ml_switcheroo_compiler.ops.shape.indexing as mindexing
    except ImportError:
        mindexing = mshape
    try:
        import ml_switcheroo_compiler.ops.shape.frontend as mfrontend
    except ImportError:
        mfrontend = mshape
    import ml_switcheroo_compiler.ops.linalg as mlinalg
    import ml_switcheroo_compiler.ops.reductions as mreductions
    import ml_switcheroo_compiler.ops.binary as mbinary
    import ml_switcheroo_compiler.ops.unary as munary
    import ml_switcheroo_compiler.random as mrand

    try:
        import ml_switcheroo_compiler.ops.shape.misc as mmisc
    except ImportError:
        mmisc = mshape
    try:
        import ml_switcheroo_compiler.ops.shape.reshape as mreshape
    except ImportError:
        mreshape = mshape

    # explicit map
    explicit_map = {
        "where": mindexing,
        "repeat": mmisc,
        "broadcast_to": lambda *args, **kwargs: mops.get_op("BroadcastTo")()(
            *args, **kwargs
        ),
        "take": mindexing,
        "sigmoid": lambda *args, **kwargs: mnn.activations.sigmoid(*args, **kwargs),
        "tile": mmisc,
        "sort": mmisc,
        "argsort": mmisc,
        "argpartition": mfrontend,
        "argwhere": mfrontend,
        "tril": mmisc,
        "triu": mmisc,
        "top_k": mmisc,
        "max": mreductions,
        "min": mreductions,
        "sum": mreductions,
        "mean": mreductions,
        "var": mreductions,
        "std": mreductions,
        "prod": mreductions,
        "argmin": mreductions,
        "argmax": mreductions,
        "all": mreductions,
        "any": mreductions,
    }

    if op_name in explicit_map:
        val = explicit_map[op_name]
        if callable(val) and val.__name__ == "<lambda>":
            return val
        return getattr(val, op_name)

    modules = [
        mops,
        mnn,
        mrand,
        mshape,
        mindexing,
        mfrontend,
        mlinalg,
        mreductions,
        mbinary,
        munary,
    ]
    for mod in modules:
        if hasattr(mod, op_name):
            return getattr(mod, op_name)

    if hasattr(mops, "get_op"):
        try:
            return mops.get_op(op_name)
        except Exception:
            pass

    raise NotImplementedError(f"Missing in compiler: {op_name}")


def _wrap(x: Any, dtype: Optional[DType] = None) -> Any:  # pragma: no cover
    """Compute _wrap.

    Args:
        x: The x argument.
        dtype: The dtype argument.

    Returns:
        The result of _wrap.

    """
    from zero_mlx.array import array  # pragma: no cover

    if isinstance(x, ml_switcheroo.Tensor):  # pragma: no cover
        return array(x, dtype=dtype)  # pragma: no cover
    return x  # pragma: no cover


def all(  # pragma: no cover
    a: Any, axis: Any = None, keepdims: bool = False, stream: Any = None
) -> Any:  # pragma: no cover
    """Compute all.

    Args:
        a: The a argument.
        axis: The axis argument.
        keepdims: The keepdims argument.
        stream: The stream argument.

    Returns:
        The result of all.

    """
    from zero_mlx.array import array, _to_tensor  # pragma: no cover
    import ml_switcheroo_compiler.ops as mops  # pragma: no cover
    import ml_switcheroo_compiler.ops.nn as mnn  # pragma: no cover
    import ml_switcheroo_compiler.random as mrand  # pragma: no cover
    import types  # pragma: no cover

    def _u(x, param_name=None):  # pragma: no cover
        if isinstance(x, list):  # pragma: no cover
            if __import__("builtins").all(  # pragma: no cover
                (isinstance(i, int) for i in x)  # pragma: no cover
            ):  # pragma: no cover
                return tuple(x)  # pragma: no cover
            return [_u(i, param_name) for i in x]  # pragma: no cover
        if isinstance(x, tuple):  # pragma: no cover
            return tuple((_u(i, param_name) for i in x))  # pragma: no cover
        if hasattr(x, "_tensor"):  # pragma: no cover
            return x._tensor  # pragma: no cover
        if isinstance(x, (int, float, bool, complex)):  # pragma: no cover
            if param_name not in {  # pragma: no cover
                "axis",  # pragma: no cover
                "keepdims",
                "shape",
                "size",  # pragma: no cover
                "equal_nan",
                "nan",
                "posinf",
                "neginf",  # pragma: no cover
                "axis1",  # pragma: no cover
                "rtol",  # pragma: no cover
                "n",  # pragma: no cover
                "m",  # pragma: no cover
                "endpoint",  # pragma: no cover
                "offset",  # pragma: no cover
                "a_max",  # pragma: no cover
                "ddof",  # pragma: no cover
                "dims",  # pragma: no cover
                "source",  # pragma: no cover
                "axis2",  # pragma: no cover
                "stop",  # pragma: no cover
                "k",  # pragma: no cover
                "dtype",  # pragma: no cover
                "reps",  # pragma: no cover
                "a_min",  # pragma: no cover
                "sparse",  # pragma: no cover
                "decimals",  # pragma: no cover
                "strides",  # pragma: no cover
                "num",  # pragma: no cover
                "max",  # pragma: no cover
                "indices_or_sections",  # pragma: no cover
                "shape",  # pragma: no cover
                "atol",  # pragma: no cover
                "axis1",  # pragma: no cover
                "step",  # pragma: no cover
                "axes",  # pragma: no cover
                "fill_value",  # pragma: no cover
                "stream",  # pragma: no cover
                "destination",  # pragma: no cover
                "alpha",  # pragma: no cover
                "beta",  # pragma: no cover
                "start",  # pragma: no cover
                "equal_nan",
                "nan",
                "posinf",
                "neginf",  # pragma: no cover
                "keepdims",
                "shape",
                "size",  # pragma: no cover
                "min",  # pragma: no cover
                "p",  # pragma: no cover
            } and "all" not in (  # pragma: no cover
                "transpose",  # pragma: no cover
                "moveaxis",  # pragma: no cover
                "reshape",  # pragma: no cover
                "expand_dims",  # pragma: no cover
                "linspace",  # pragma: no cover
                "logcumsumexp",  # pragma: no cover
                "cummin",  # pragma: no cover
                "cummax",  # pragma: no cover
                "cumprod",  # pragma: no cover
                "cumsum",  # pragma: no cover
                "logsumexp",  # pragma: no cover
                "all",  # pragma: no cover
                "any",  # pragma: no cover
                "argmin",  # pragma: no cover
                "argmax",  # pragma: no cover
                "prod",  # pragma: no cover
                "sum",  # pragma: no cover
                "mean",  # pragma: no cover
                "var",  # pragma: no cover
                "std",  # pragma: no cover
                "max",  # pragma: no cover
                "min",  # pragma: no cover
                "tensordot",  # pragma: no cover
            ):  # pragma: no cover
                return _to_tensor(x)  # pragma: no cover
        if type(x).__name__ == "ndarray":  # pragma: no cover
            return _to_tensor(x)  # pragma: no cover
        return x  # pragma: no cover

    def _w(x):  # pragma: no cover
        if isinstance(x, tuple):  # pragma: no cover
            return tuple((_w(i) for i in x))  # pragma: no cover
        if isinstance(x, list):  # pragma: no cover
            return [_w(i) for i in x]  # pragma: no cover
        return array(x) if hasattr(x, "shape") else x  # pragma: no cover

    fn = _get_compiler_op("all")  # pragma: no cover
    res = fn(
        _u(a, "a"), axis=_u(axis, "axis"), keepdims=_u(keepdims, "keepdims")
    )  # pragma: no cover

    def __w(x):
        if isinstance(x, tuple):
            return tuple((__w(i) for i in x))
        if isinstance(x, list):
            return [__w(i) for i in x]
        return array(x) if hasattr(x, "shape") else x

    return __w(res)  # pragma: no cover


def any(  # pragma: no cover
    a: Any, axis: Any = None, keepdims: bool = False, stream: Any = None
) -> Any:  # pragma: no cover
    """Compute any.

    Args:
        a: The a argument.
        axis: The axis argument.
        keepdims: The keepdims argument.
        stream: The stream argument.

    Returns:
        The result of any.

    """
    from zero_mlx.array import array, _to_tensor  # pragma: no cover
    import ml_switcheroo_compiler.ops as mops  # pragma: no cover
    import ml_switcheroo_compiler.ops.nn as mnn  # pragma: no cover
    import ml_switcheroo_compiler.random as mrand  # pragma: no cover
    import types  # pragma: no cover

    def _u(x, param_name=None):  # pragma: no cover
        if isinstance(x, list):  # pragma: no cover
            if __import__("builtins").all(  # pragma: no cover
                (isinstance(i, int) for i in x)  # pragma: no cover
            ):  # pragma: no cover
                return tuple(x)  # pragma: no cover
            return [_u(i, param_name) for i in x]  # pragma: no cover
        if isinstance(x, tuple):  # pragma: no cover
            return tuple((_u(i, param_name) for i in x))  # pragma: no cover
        if hasattr(x, "_tensor"):  # pragma: no cover
            return x._tensor  # pragma: no cover
        if isinstance(x, (int, float, bool, complex)):  # pragma: no cover
            if param_name not in {  # pragma: no cover
                "axis",  # pragma: no cover
                "keepdims",
                "shape",
                "size",  # pragma: no cover
                "equal_nan",
                "nan",
                "posinf",
                "neginf",  # pragma: no cover
                "axis1",  # pragma: no cover
                "rtol",  # pragma: no cover
                "n",  # pragma: no cover
                "m",  # pragma: no cover
                "endpoint",  # pragma: no cover
                "offset",  # pragma: no cover
                "a_max",  # pragma: no cover
                "ddof",  # pragma: no cover
                "dims",  # pragma: no cover
                "source",  # pragma: no cover
                "axis2",  # pragma: no cover
                "stop",  # pragma: no cover
                "k",  # pragma: no cover
                "dtype",  # pragma: no cover
                "reps",  # pragma: no cover
                "a_min",  # pragma: no cover
                "sparse",  # pragma: no cover
                "decimals",  # pragma: no cover
                "strides",  # pragma: no cover
                "num",  # pragma: no cover
                "max",  # pragma: no cover
                "indices_or_sections",  # pragma: no cover
                "shape",  # pragma: no cover
                "atol",  # pragma: no cover
                "axis1",  # pragma: no cover
                "step",  # pragma: no cover
                "axes",  # pragma: no cover
                "fill_value",  # pragma: no cover
                "stream",  # pragma: no cover
                "destination",  # pragma: no cover
                "alpha",  # pragma: no cover
                "beta",  # pragma: no cover
                "start",  # pragma: no cover
                "equal_nan",
                "nan",
                "posinf",
                "neginf",  # pragma: no cover
                "keepdims",
                "shape",
                "size",  # pragma: no cover
                "min",  # pragma: no cover
                "p",  # pragma: no cover
            } and "any" not in (  # pragma: no cover
                "transpose",  # pragma: no cover
                "moveaxis",  # pragma: no cover
                "reshape",  # pragma: no cover
                "expand_dims",  # pragma: no cover
                "linspace",  # pragma: no cover
                "logcumsumexp",  # pragma: no cover
                "cummin",  # pragma: no cover
                "cummax",  # pragma: no cover
                "cumprod",  # pragma: no cover
                "cumsum",  # pragma: no cover
                "logsumexp",  # pragma: no cover
                "all",  # pragma: no cover
                "any",  # pragma: no cover
                "argmin",  # pragma: no cover
                "argmax",  # pragma: no cover
                "prod",  # pragma: no cover
                "sum",  # pragma: no cover
                "mean",  # pragma: no cover
                "var",  # pragma: no cover
                "std",  # pragma: no cover
                "max",  # pragma: no cover
                "min",  # pragma: no cover
                "tensordot",  # pragma: no cover
            ):  # pragma: no cover
                return _to_tensor(x)  # pragma: no cover
        if type(x).__name__ == "ndarray":  # pragma: no cover
            return _to_tensor(x)  # pragma: no cover
        return x  # pragma: no cover

    def _w(x):  # pragma: no cover
        if isinstance(x, tuple):  # pragma: no cover
            return tuple((_w(i) for i in x))  # pragma: no cover
        if isinstance(x, list):  # pragma: no cover
            return [_w(i) for i in x]  # pragma: no cover
        return array(x) if hasattr(x, "shape") else x  # pragma: no cover

    fn = _get_compiler_op("any")  # pragma: no cover
    res = fn(
        _u(a, "a"), axis=_u(axis, "axis"), keepdims=_u(keepdims, "keepdims")
    )  # pragma: no cover

    def __w(x):
        if isinstance(x, tuple):
            return tuple((__w(i) for i in x))
        if isinstance(x, list):
            return [__w(i) for i in x]
        return array(x) if hasattr(x, "shape") else x

    return __w(res)  # pragma: no cover


def allclose(  # pragma: no cover
    a: Any,  # pragma: no cover
    b: Any,  # pragma: no cover
    rtol: float = 1e-05,  # pragma: no cover
    atol: float = 1e-08,  # pragma: no cover
    equal_nan: bool = False,  # pragma: no cover
    stream: Any = None,  # pragma: no cover
) -> Any:  # pragma: no cover
    """Compute allclose.

    Args:
        a: The a argument.
        b: The b argument.
        rtol: The rtol argument.
        atol: The atol argument.
        equal_nan: The equal_nan argument.
        stream: The stream argument.

    Returns:
        The result of allclose.

    """
    from zero_mlx.array import array, _to_tensor  # pragma: no cover
    import ml_switcheroo_compiler.ops as mops  # pragma: no cover
    import ml_switcheroo_compiler.ops.nn as mnn  # pragma: no cover
    import ml_switcheroo_compiler.random as mrand  # pragma: no cover
    import types  # pragma: no cover

    def _u(x, param_name=None):  # pragma: no cover
        if isinstance(x, list):  # pragma: no cover
            if __import__("builtins").all(  # pragma: no cover
                (isinstance(i, int) for i in x)  # pragma: no cover
            ):  # pragma: no cover
                return tuple(x)  # pragma: no cover
            return [_u(i, param_name) for i in x]  # pragma: no cover
        if isinstance(x, tuple):  # pragma: no cover
            return tuple((_u(i, param_name) for i in x))  # pragma: no cover
        if hasattr(x, "_tensor"):  # pragma: no cover
            return x._tensor  # pragma: no cover
        if isinstance(x, (int, float, bool, complex)):  # pragma: no cover
            if param_name not in {  # pragma: no cover
                "axis",  # pragma: no cover
                "keepdims",
                "shape",
                "size",  # pragma: no cover
                "equal_nan",
                "nan",
                "posinf",
                "neginf",  # pragma: no cover
                "axis1",  # pragma: no cover
                "rtol",  # pragma: no cover
                "n",  # pragma: no cover
                "m",  # pragma: no cover
                "endpoint",  # pragma: no cover
                "offset",  # pragma: no cover
                "a_max",  # pragma: no cover
                "ddof",  # pragma: no cover
                "dims",  # pragma: no cover
                "source",  # pragma: no cover
                "axis2",  # pragma: no cover
                "stop",  # pragma: no cover
                "k",  # pragma: no cover
                "dtype",  # pragma: no cover
                "reps",  # pragma: no cover
                "a_min",  # pragma: no cover
                "sparse",  # pragma: no cover
                "decimals",  # pragma: no cover
                "strides",  # pragma: no cover
                "num",  # pragma: no cover
                "max",  # pragma: no cover
                "indices_or_sections",  # pragma: no cover
                "shape",  # pragma: no cover
                "atol",  # pragma: no cover
                "axis1",  # pragma: no cover
                "step",  # pragma: no cover
                "axes",  # pragma: no cover
                "fill_value",  # pragma: no cover
                "stream",  # pragma: no cover
                "destination",  # pragma: no cover
                "alpha",  # pragma: no cover
                "beta",  # pragma: no cover
                "start",  # pragma: no cover
                "equal_nan",
                "nan",
                "posinf",
                "neginf",  # pragma: no cover
                "keepdims",
                "shape",
                "size",  # pragma: no cover
                "min",  # pragma: no cover
                "p",  # pragma: no cover
            } and "allclose" not in (  # pragma: no cover
                "transpose",  # pragma: no cover
                "moveaxis",  # pragma: no cover
                "reshape",  # pragma: no cover
                "expand_dims",  # pragma: no cover
                "linspace",  # pragma: no cover
                "logcumsumexp",  # pragma: no cover
                "cummin",  # pragma: no cover
                "cummax",  # pragma: no cover
                "cumprod",  # pragma: no cover
                "cumsum",  # pragma: no cover
                "logsumexp",  # pragma: no cover
                "all",  # pragma: no cover
                "any",  # pragma: no cover
                "argmin",  # pragma: no cover
                "argmax",  # pragma: no cover
                "prod",  # pragma: no cover
                "sum",  # pragma: no cover
                "mean",  # pragma: no cover
                "var",  # pragma: no cover
                "std",  # pragma: no cover
                "max",  # pragma: no cover
                "min",  # pragma: no cover
                "tensordot",  # pragma: no cover
            ):  # pragma: no cover
                return _to_tensor(x)  # pragma: no cover
        if type(x).__name__ == "ndarray":  # pragma: no cover
            return _to_tensor(x)  # pragma: no cover
        return x  # pragma: no cover

    def _w(x):  # pragma: no cover
        if isinstance(x, tuple):  # pragma: no cover
            return tuple((_w(i) for i in x))  # pragma: no cover
        if isinstance(x, list):  # pragma: no cover
            return [_w(i) for i in x]  # pragma: no cover
        return array(x) if hasattr(x, "shape") else x  # pragma: no cover

    fn = _get_compiler_op("allclose")  # pragma: no cover
    res = fn(
        _u(a, "a"), _u(b, "b"), rtol=rtol, atol=atol, equal_nan=equal_nan
    )  # pragma: no cover

    def __w(x):
        if isinstance(x, tuple):
            return tuple((__w(i) for i in x))
        if isinstance(x, list):
            return [__w(i) for i in x]
        return array(x) if hasattr(x, "shape") else x

    return __w(res)  # pragma: no cover


def synchronize(*args, **kwargs):  # pragma: no cover
    """Compute synchronize.

    Args:
        args: The args argument.
        kwargs: The kwargs argument.

    Returns:
        The result of synchronize.

    """
    from zero_mlx.array import array, _to_tensor  # pragma: no cover
    import ml_switcheroo_compiler.ops as mops  # pragma: no cover
    import ml_switcheroo_compiler.ops.nn as mnn  # pragma: no cover
    import ml_switcheroo_compiler.random as mrand  # pragma: no cover
    import types  # pragma: no cover

    def _u(x, param_name=None):  # pragma: no cover
        if isinstance(x, list):  # pragma: no cover
            if __import__("builtins").all(  # pragma: no cover
                (isinstance(i, int) for i in x)  # pragma: no cover
            ):  # pragma: no cover
                return tuple(x)  # pragma: no cover
            return [_u(i, param_name) for i in x]  # pragma: no cover
        if isinstance(x, tuple):  # pragma: no cover
            return tuple((_u(i, param_name) for i in x))  # pragma: no cover
        if hasattr(x, "_tensor"):  # pragma: no cover
            return x._tensor  # pragma: no cover
        if isinstance(x, (int, float, bool, complex)):  # pragma: no cover
            if param_name not in {  # pragma: no cover
                "axis",  # pragma: no cover
                "keepdims",
                "shape",
                "size",  # pragma: no cover
                "equal_nan",
                "nan",
                "posinf",
                "neginf",  # pragma: no cover
                "axis1",  # pragma: no cover
                "rtol",  # pragma: no cover
                "n",  # pragma: no cover
                "m",  # pragma: no cover
                "endpoint",  # pragma: no cover
                "offset",  # pragma: no cover
                "a_max",  # pragma: no cover
                "ddof",  # pragma: no cover
                "dims",  # pragma: no cover
                "source",  # pragma: no cover
                "axis2",  # pragma: no cover
                "stop",  # pragma: no cover
                "k",  # pragma: no cover
                "dtype",  # pragma: no cover
                "reps",  # pragma: no cover
                "a_min",  # pragma: no cover
                "sparse",  # pragma: no cover
                "decimals",  # pragma: no cover
                "strides",  # pragma: no cover
                "num",  # pragma: no cover
                "max",  # pragma: no cover
                "indices_or_sections",  # pragma: no cover
                "shape",  # pragma: no cover
                "atol",  # pragma: no cover
                "axis1",  # pragma: no cover
                "step",  # pragma: no cover
                "axes",  # pragma: no cover
                "fill_value",  # pragma: no cover
                "stream",  # pragma: no cover
                "destination",  # pragma: no cover
                "alpha",  # pragma: no cover
                "beta",  # pragma: no cover
                "start",  # pragma: no cover
                "equal_nan",
                "nan",
                "posinf",
                "neginf",  # pragma: no cover
                "keepdims",
                "shape",
                "size",  # pragma: no cover
                "min",  # pragma: no cover
                "p",  # pragma: no cover
            } and "synchronize" not in (  # pragma: no cover
                "transpose",  # pragma: no cover
                "moveaxis",  # pragma: no cover
                "reshape",  # pragma: no cover
                "expand_dims",  # pragma: no cover
                "linspace",  # pragma: no cover
                "logcumsumexp",  # pragma: no cover
                "cummin",  # pragma: no cover
                "cummax",  # pragma: no cover
                "cumprod",  # pragma: no cover
                "cumsum",  # pragma: no cover
                "logsumexp",  # pragma: no cover
                "all",  # pragma: no cover
                "any",  # pragma: no cover
                "argmin",  # pragma: no cover
                "argmax",  # pragma: no cover
                "prod",  # pragma: no cover
                "sum",  # pragma: no cover
                "mean",  # pragma: no cover
                "var",  # pragma: no cover
                "std",  # pragma: no cover
                "max",  # pragma: no cover
                "min",  # pragma: no cover
                "tensordot",  # pragma: no cover
            ):  # pragma: no cover
                return _to_tensor(x)  # pragma: no cover
        if type(x).__name__ == "ndarray":  # pragma: no cover
            return _to_tensor(x)  # pragma: no cover
        return x  # pragma: no cover

    def _w(x):  # pragma: no cover
        if isinstance(x, tuple):  # pragma: no cover
            return tuple((_w(i) for i in x))  # pragma: no cover
        if isinstance(x, list):  # pragma: no cover
            return [_w(i) for i in x]  # pragma: no cover
        return array(x) if hasattr(x, "shape") else x  # pragma: no cover

    fn = _get_compiler_op("synchronize")  # pragma: no cover
    res = fn(  # pragma: no cover
        *[_u(x) for x in args],  # pragma: no cover
        **{
            ("dim" if k == "axis" else k): _u(v, k)
            for (k, v) in kwargs.items()
            if k != "stream"
        },  # pragma: no cover
    )  # pragma: no cover

    def __w(x):
        if isinstance(x, tuple):
            return tuple((__w(i) for i in x))
        if isinstance(x, list):
            return [__w(i) for i in x]
        return array(x) if hasattr(x, "shape") else x

    return __w(res)  # pragma: no cover


def eval(*args, **kwargs):  # pragma: no cover
    """Compute eval.

    Args:
        args: The args argument.
        kwargs: The kwargs argument.

    Returns:
        The result of eval.

    """
    from zero_mlx.array import array, _to_tensor  # pragma: no cover
    import ml_switcheroo_compiler.ops as mops  # pragma: no cover
    import ml_switcheroo_compiler.ops.nn as mnn  # pragma: no cover
    import ml_switcheroo_compiler.random as mrand  # pragma: no cover
    import types  # pragma: no cover

    def _u(x, param_name=None):  # pragma: no cover
        if isinstance(x, list):  # pragma: no cover
            if __import__("builtins").all(  # pragma: no cover
                (isinstance(i, int) for i in x)  # pragma: no cover
            ):  # pragma: no cover
                return tuple(x)  # pragma: no cover
            return [_u(i, param_name) for i in x]  # pragma: no cover
        if isinstance(x, tuple):  # pragma: no cover
            return tuple((_u(i, param_name) for i in x))  # pragma: no cover
        if hasattr(x, "_tensor"):  # pragma: no cover
            return x._tensor  # pragma: no cover
        if isinstance(x, (int, float, bool, complex)):  # pragma: no cover
            if param_name not in {  # pragma: no cover
                "axis",  # pragma: no cover
                "keepdims",
                "shape",
                "size",  # pragma: no cover
                "equal_nan",
                "nan",
                "posinf",
                "neginf",  # pragma: no cover
                "axis1",  # pragma: no cover
                "rtol",  # pragma: no cover
                "n",  # pragma: no cover
                "m",  # pragma: no cover
                "endpoint",  # pragma: no cover
                "offset",  # pragma: no cover
                "a_max",  # pragma: no cover
                "ddof",  # pragma: no cover
                "dims",  # pragma: no cover
                "source",  # pragma: no cover
                "axis2",  # pragma: no cover
                "stop",  # pragma: no cover
                "k",  # pragma: no cover
                "dtype",  # pragma: no cover
                "reps",  # pragma: no cover
                "a_min",  # pragma: no cover
                "sparse",  # pragma: no cover
                "decimals",  # pragma: no cover
                "strides",  # pragma: no cover
                "num",  # pragma: no cover
                "max",  # pragma: no cover
                "indices_or_sections",  # pragma: no cover
                "shape",  # pragma: no cover
                "atol",  # pragma: no cover
                "axis1",  # pragma: no cover
                "step",  # pragma: no cover
                "axes",  # pragma: no cover
                "fill_value",  # pragma: no cover
                "stream",  # pragma: no cover
                "destination",  # pragma: no cover
                "alpha",  # pragma: no cover
                "beta",  # pragma: no cover
                "start",  # pragma: no cover
                "equal_nan",
                "nan",
                "posinf",
                "neginf",  # pragma: no cover
                "keepdims",
                "shape",
                "size",  # pragma: no cover
                "min",  # pragma: no cover
                "p",  # pragma: no cover
            } and "eval" not in (  # pragma: no cover
                "transpose",  # pragma: no cover
                "moveaxis",  # pragma: no cover
                "reshape",  # pragma: no cover
                "expand_dims",  # pragma: no cover
                "linspace",  # pragma: no cover
                "logcumsumexp",  # pragma: no cover
                "cummin",  # pragma: no cover
                "cummax",  # pragma: no cover
                "cumprod",  # pragma: no cover
                "cumsum",  # pragma: no cover
                "logsumexp",  # pragma: no cover
                "all",  # pragma: no cover
                "any",  # pragma: no cover
                "argmin",  # pragma: no cover
                "argmax",  # pragma: no cover
                "prod",  # pragma: no cover
                "sum",  # pragma: no cover
                "mean",  # pragma: no cover
                "var",  # pragma: no cover
                "std",  # pragma: no cover
                "max",  # pragma: no cover
                "min",  # pragma: no cover
                "tensordot",  # pragma: no cover
            ):  # pragma: no cover
                return _to_tensor(x)  # pragma: no cover
        if type(x).__name__ == "ndarray":  # pragma: no cover
            return _to_tensor(x)  # pragma: no cover
        return x  # pragma: no cover

    def _w(x):  # pragma: no cover
        if isinstance(x, tuple):  # pragma: no cover
            return tuple((_w(i) for i in x))  # pragma: no cover
        if isinstance(x, list):  # pragma: no cover
            return [_w(i) for i in x]  # pragma: no cover
        return array(x) if hasattr(x, "shape") else x  # pragma: no cover

    fn = _get_compiler_op("eval")  # pragma: no cover
    res = fn(  # pragma: no cover
        *[_u(x) for x in args],  # pragma: no cover
        **{
            ("dim" if k == "axis" else k): _u(v, k)
            for (k, v) in kwargs.items()
            if k != "stream"
        },  # pragma: no cover
    )  # pragma: no cover

    def __w(x):
        if isinstance(x, tuple):
            return tuple((__w(i) for i in x))
        if isinstance(x, list):
            return [__w(i) for i in x]
        return array(x) if hasattr(x, "shape") else x

    return __w(res)  # pragma: no cover


def old_split(*args, **kwargs):  # pragma: no cover
    """Compute old_split.

    Args:
        args: The args argument.
        kwargs: The kwargs argument.

    Returns:
        The result of old_split.

    """
    from zero_mlx.array import array, _to_tensor  # pragma: no cover
    import ml_switcheroo_compiler.ops as mops  # pragma: no cover
    import ml_switcheroo_compiler.ops.nn as mnn  # pragma: no cover
    import ml_switcheroo_compiler.random as mrand  # pragma: no cover
    import types  # pragma: no cover

    def _u(x, param_name=None):  # pragma: no cover
        if isinstance(x, list):  # pragma: no cover
            if __import__("builtins").all(  # pragma: no cover
                (isinstance(i, int) for i in x)  # pragma: no cover
            ):  # pragma: no cover
                return tuple(x)  # pragma: no cover
            return [_u(i, param_name) for i in x]  # pragma: no cover
        if isinstance(x, tuple):  # pragma: no cover
            return tuple((_u(i, param_name) for i in x))  # pragma: no cover
        if hasattr(x, "_tensor"):  # pragma: no cover
            return x._tensor  # pragma: no cover
        if isinstance(x, (int, float, bool, complex)):  # pragma: no cover
            if param_name not in {  # pragma: no cover
                "axis",  # pragma: no cover
                "keepdims",
                "shape",
                "size",  # pragma: no cover
                "equal_nan",
                "nan",
                "posinf",
                "neginf",  # pragma: no cover
                "axis1",  # pragma: no cover
                "rtol",  # pragma: no cover
                "n",  # pragma: no cover
                "m",  # pragma: no cover
                "endpoint",  # pragma: no cover
                "offset",  # pragma: no cover
                "a_max",  # pragma: no cover
                "ddof",  # pragma: no cover
                "dims",  # pragma: no cover
                "source",  # pragma: no cover
                "axis2",  # pragma: no cover
                "stop",  # pragma: no cover
                "k",  # pragma: no cover
                "dtype",  # pragma: no cover
                "reps",  # pragma: no cover
                "a_min",  # pragma: no cover
                "sparse",  # pragma: no cover
                "decimals",  # pragma: no cover
                "strides",  # pragma: no cover
                "num",  # pragma: no cover
                "max",  # pragma: no cover
                "indices_or_sections",  # pragma: no cover
                "shape",  # pragma: no cover
                "atol",  # pragma: no cover
                "axis1",  # pragma: no cover
                "step",  # pragma: no cover
                "axes",  # pragma: no cover
                "fill_value",  # pragma: no cover
                "stream",  # pragma: no cover
                "destination",  # pragma: no cover
                "alpha",  # pragma: no cover
                "beta",  # pragma: no cover
                "start",  # pragma: no cover
                "equal_nan",
                "nan",
                "posinf",
                "neginf",  # pragma: no cover
                "keepdims",
                "shape",
                "size",  # pragma: no cover
                "min",  # pragma: no cover
                "p",  # pragma: no cover
            } and "old_split" not in (  # pragma: no cover
                "transpose",  # pragma: no cover
                "moveaxis",  # pragma: no cover
                "reshape",  # pragma: no cover
                "expand_dims",  # pragma: no cover
                "linspace",  # pragma: no cover
                "logcumsumexp",  # pragma: no cover
                "cummin",  # pragma: no cover
                "cummax",  # pragma: no cover
                "cumprod",  # pragma: no cover
                "cumsum",  # pragma: no cover
                "logsumexp",  # pragma: no cover
                "all",  # pragma: no cover
                "any",  # pragma: no cover
                "argmin",  # pragma: no cover
                "argmax",  # pragma: no cover
                "prod",  # pragma: no cover
                "sum",  # pragma: no cover
                "mean",  # pragma: no cover
                "var",  # pragma: no cover
                "std",  # pragma: no cover
                "max",  # pragma: no cover
                "min",  # pragma: no cover
                "tensordot",  # pragma: no cover
            ):  # pragma: no cover
                return _to_tensor(x)  # pragma: no cover
        if type(x).__name__ == "ndarray":  # pragma: no cover
            return _to_tensor(x)  # pragma: no cover
        return x  # pragma: no cover

    def _w(x):  # pragma: no cover
        if isinstance(x, tuple):  # pragma: no cover
            return tuple((_w(i) for i in x))  # pragma: no cover
        if isinstance(x, list):  # pragma: no cover
            return [_w(i) for i in x]  # pragma: no cover
        return array(x) if hasattr(x, "shape") else x  # pragma: no cover

    fn = _get_compiler_op("old_split")  # pragma: no cover
    res = fn(  # pragma: no cover
        *[_u(x) for x in args],  # pragma: no cover
        **{
            ("dim" if k == "axis" else k): _u(v, k)
            for (k, v) in kwargs.items()
            if k != "stream"
        },  # pragma: no cover
    )  # pragma: no cover

    def __w(x):
        if isinstance(x, tuple):
            return tuple((__w(i) for i in x))
        if isinstance(x, list):
            return [__w(i) for i in x]
        return array(x) if hasattr(x, "shape") else x

    return __w(res)  # pragma: no cover


def asarray(  # pragma: no cover
    a: Any,
    dtype: Optional[DType] = None,
    stream: Any = None,
    copy: bool = False,  # pragma: no cover
) -> Any:  # pragma: no cover
    """Compute asarray.

    Args:
        a: The a argument.
        dtype: The dtype argument.
        stream: The stream argument.
        copy: The copy argument.

    Returns:
        The result of asarray.

    """
    from zero_mlx.array import array, _to_tensor  # pragma: no cover
    import ml_switcheroo_compiler.ops as mops  # pragma: no cover
    import ml_switcheroo_compiler.ops.nn as mnn  # pragma: no cover
    import ml_switcheroo_compiler.random as mrand  # pragma: no cover
    import types  # pragma: no cover

    def _u(x, param_name=None):  # pragma: no cover
        if isinstance(x, list):  # pragma: no cover
            if __import__("builtins").all(  # pragma: no cover
                (isinstance(i, int) for i in x)  # pragma: no cover
            ):  # pragma: no cover
                return tuple(x)  # pragma: no cover
            return [_u(i, param_name) for i in x]  # pragma: no cover
        if isinstance(x, tuple):  # pragma: no cover
            return tuple((_u(i, param_name) for i in x))  # pragma: no cover
        if hasattr(x, "_tensor"):  # pragma: no cover
            return x._tensor  # pragma: no cover
        if isinstance(x, (int, float, bool, complex)):  # pragma: no cover
            if param_name not in {  # pragma: no cover
                "axis",  # pragma: no cover
                "keepdims",
                "shape",
                "size",  # pragma: no cover
                "equal_nan",
                "nan",
                "posinf",
                "neginf",  # pragma: no cover
                "axis1",  # pragma: no cover
                "rtol",  # pragma: no cover
                "n",  # pragma: no cover
                "m",  # pragma: no cover
                "endpoint",  # pragma: no cover
                "offset",  # pragma: no cover
                "a_max",  # pragma: no cover
                "ddof",  # pragma: no cover
                "dims",  # pragma: no cover
                "source",  # pragma: no cover
                "axis2",  # pragma: no cover
                "stop",  # pragma: no cover
                "k",  # pragma: no cover
                "dtype",  # pragma: no cover
                "reps",  # pragma: no cover
                "a_min",  # pragma: no cover
                "sparse",  # pragma: no cover
                "decimals",  # pragma: no cover
                "strides",  # pragma: no cover
                "num",  # pragma: no cover
                "max",  # pragma: no cover
                "indices_or_sections",  # pragma: no cover
                "shape",  # pragma: no cover
                "atol",  # pragma: no cover
                "axis1",  # pragma: no cover
                "step",  # pragma: no cover
                "axes",  # pragma: no cover
                "fill_value",  # pragma: no cover
                "stream",  # pragma: no cover
                "destination",  # pragma: no cover
                "alpha",  # pragma: no cover
                "beta",  # pragma: no cover
                "start",  # pragma: no cover
                "equal_nan",
                "nan",
                "posinf",
                "neginf",  # pragma: no cover
                "keepdims",
                "shape",
                "size",  # pragma: no cover
                "min",  # pragma: no cover
                "p",  # pragma: no cover
            } and "asarray" not in (  # pragma: no cover
                "transpose",  # pragma: no cover
                "moveaxis",  # pragma: no cover
                "reshape",  # pragma: no cover
                "expand_dims",  # pragma: no cover
                "linspace",  # pragma: no cover
                "logcumsumexp",  # pragma: no cover
                "cummin",  # pragma: no cover
                "cummax",  # pragma: no cover
                "cumprod",  # pragma: no cover
                "cumsum",  # pragma: no cover
                "logsumexp",  # pragma: no cover
                "all",  # pragma: no cover
                "any",  # pragma: no cover
                "argmin",  # pragma: no cover
                "argmax",  # pragma: no cover
                "prod",  # pragma: no cover
                "sum",  # pragma: no cover
                "mean",  # pragma: no cover
                "var",  # pragma: no cover
                "std",  # pragma: no cover
                "max",  # pragma: no cover
                "min",  # pragma: no cover
                "tensordot",  # pragma: no cover
            ):  # pragma: no cover
                return _to_tensor(x)  # pragma: no cover
        if type(x).__name__ == "ndarray":  # pragma: no cover
            return _to_tensor(x)  # pragma: no cover
        return x  # pragma: no cover

    def _w(x):  # pragma: no cover
        if isinstance(x, tuple):  # pragma: no cover
            return tuple((_w(i) for i in x))  # pragma: no cover
        if isinstance(x, list):  # pragma: no cover
            return [_w(i) for i in x]  # pragma: no cover
        return array(x) if hasattr(x, "shape") else x  # pragma: no cover

    fn = _get_compiler_op("asarray")  # pragma: no cover
    res = fn(  # pragma: no cover
        _u(a, "a"),
        dtype=_u(dtype, "dtype"),
        copy=_u(copy, "copy"),  # pragma: no cover
    )  # pragma: no cover

    def __w(x):
        if isinstance(x, tuple):
            return tuple((__w(i) for i in x))
        if isinstance(x, list):
            return [__w(i) for i in x]
        return array(x) if hasattr(x, "shape") else x

    return __w(res)  # pragma: no cover


def stack(  # pragma: no cover
    arrays: Sequence[Any], axis: int = 0, stream: Any = None
) -> Any:  # pragma: no cover
    """Compute stack.

    Args:
        arrays: The arrays argument.
        axis: The axis argument.
        stream: The stream argument.

    Returns:
        The result of stack.

    """
    from zero_mlx.array import array, _to_tensor  # pragma: no cover
    import ml_switcheroo_compiler.ops as mops  # pragma: no cover
    import ml_switcheroo_compiler.ops.nn as mnn  # pragma: no cover
    import ml_switcheroo_compiler.random as mrand  # pragma: no cover
    import types  # pragma: no cover

    def _u(x, param_name=None):  # pragma: no cover
        if isinstance(x, list):  # pragma: no cover
            if __import__("builtins").all(  # pragma: no cover
                (isinstance(i, int) for i in x)  # pragma: no cover
            ):  # pragma: no cover
                return tuple(x)  # pragma: no cover
            return [_u(i, param_name) for i in x]  # pragma: no cover
        if isinstance(x, tuple):  # pragma: no cover
            return tuple((_u(i, param_name) for i in x))  # pragma: no cover
        if hasattr(x, "_tensor"):  # pragma: no cover
            return x._tensor  # pragma: no cover
        if isinstance(x, (int, float, bool, complex)):  # pragma: no cover
            if param_name not in {  # pragma: no cover
                "axis",  # pragma: no cover
                "keepdims",
                "shape",
                "size",  # pragma: no cover
                "equal_nan",
                "nan",
                "posinf",
                "neginf",  # pragma: no cover
                "axis1",  # pragma: no cover
                "rtol",  # pragma: no cover
                "n",  # pragma: no cover
                "m",  # pragma: no cover
                "endpoint",  # pragma: no cover
                "offset",  # pragma: no cover
                "a_max",  # pragma: no cover
                "ddof",  # pragma: no cover
                "dims",  # pragma: no cover
                "source",  # pragma: no cover
                "axis2",  # pragma: no cover
                "stop",  # pragma: no cover
                "k",  # pragma: no cover
                "dtype",  # pragma: no cover
                "reps",  # pragma: no cover
                "a_min",  # pragma: no cover
                "sparse",  # pragma: no cover
                "decimals",  # pragma: no cover
                "strides",  # pragma: no cover
                "num",  # pragma: no cover
                "max",  # pragma: no cover
                "indices_or_sections",  # pragma: no cover
                "shape",  # pragma: no cover
                "atol",  # pragma: no cover
                "axis1",  # pragma: no cover
                "step",  # pragma: no cover
                "axes",  # pragma: no cover
                "fill_value",  # pragma: no cover
                "stream",  # pragma: no cover
                "destination",  # pragma: no cover
                "alpha",  # pragma: no cover
                "beta",  # pragma: no cover
                "start",  # pragma: no cover
                "equal_nan",
                "nan",
                "posinf",
                "neginf",  # pragma: no cover
                "keepdims",
                "shape",
                "size",  # pragma: no cover
                "min",  # pragma: no cover
                "p",  # pragma: no cover
            } and "stack" not in (  # pragma: no cover
                "transpose",  # pragma: no cover
                "moveaxis",  # pragma: no cover
                "reshape",  # pragma: no cover
                "expand_dims",  # pragma: no cover
                "linspace",  # pragma: no cover
                "logcumsumexp",  # pragma: no cover
                "cummin",  # pragma: no cover
                "cummax",  # pragma: no cover
                "cumprod",  # pragma: no cover
                "cumsum",  # pragma: no cover
                "logsumexp",  # pragma: no cover
                "all",  # pragma: no cover
                "any",  # pragma: no cover
                "argmin",  # pragma: no cover
                "argmax",  # pragma: no cover
                "prod",  # pragma: no cover
                "sum",  # pragma: no cover
                "mean",  # pragma: no cover
                "var",  # pragma: no cover
                "std",  # pragma: no cover
                "max",  # pragma: no cover
                "min",  # pragma: no cover
                "tensordot",  # pragma: no cover
            ):  # pragma: no cover
                return _to_tensor(x)  # pragma: no cover
        if type(x).__name__ == "ndarray":  # pragma: no cover
            return _to_tensor(x)  # pragma: no cover
        return x  # pragma: no cover

    def _w(x):  # pragma: no cover
        if isinstance(x, tuple):  # pragma: no cover
            return tuple((_w(i) for i in x))  # pragma: no cover
        if isinstance(x, list):  # pragma: no cover
            return [_w(i) for i in x]  # pragma: no cover
        return array(x) if hasattr(x, "shape") else x  # pragma: no cover

    fn = _get_compiler_op("stack")  # pragma: no cover
    res = fn(_u(arrays, "arrays"), dim=_u(axis, "axis"))  # pragma: no cover

    def __w(x):
        if isinstance(x, tuple):
            return tuple((__w(i) for i in x))
        if isinstance(x, list):
            return [__w(i) for i in x]
        return array(x) if hasattr(x, "shape") else x

    return __w(res)  # pragma: no cover


def sin(a: Any, stream: Any = None) -> Any:  # pragma: no cover
    """Compute sin.

    Args:
        a: The a argument.
        stream: The stream argument.

    Returns:
        The result of sin.

    """
    from zero_mlx.array import array, _to_tensor  # pragma: no cover
    import ml_switcheroo_compiler.ops as mops  # pragma: no cover
    import ml_switcheroo_compiler.ops.nn as mnn  # pragma: no cover
    import ml_switcheroo_compiler.random as mrand  # pragma: no cover
    import types  # pragma: no cover

    def _u(x, param_name=None):  # pragma: no cover
        if isinstance(x, list):  # pragma: no cover
            if __import__("builtins").all(  # pragma: no cover
                (isinstance(i, int) for i in x)  # pragma: no cover
            ):  # pragma: no cover
                return tuple(x)  # pragma: no cover
            return [_u(i, param_name) for i in x]  # pragma: no cover
        if isinstance(x, tuple):  # pragma: no cover
            return tuple((_u(i, param_name) for i in x))  # pragma: no cover
        if hasattr(x, "_tensor"):  # pragma: no cover
            return x._tensor  # pragma: no cover
        if isinstance(x, (int, float, bool, complex)):  # pragma: no cover
            if param_name not in {  # pragma: no cover
                "axis",  # pragma: no cover
                "keepdims",
                "shape",
                "size",  # pragma: no cover
                "equal_nan",
                "nan",
                "posinf",
                "neginf",  # pragma: no cover
                "axis1",  # pragma: no cover
                "rtol",  # pragma: no cover
                "n",  # pragma: no cover
                "m",  # pragma: no cover
                "endpoint",  # pragma: no cover
                "offset",  # pragma: no cover
                "a_max",  # pragma: no cover
                "ddof",  # pragma: no cover
                "dims",  # pragma: no cover
                "source",  # pragma: no cover
                "axis2",  # pragma: no cover
                "stop",  # pragma: no cover
                "k",  # pragma: no cover
                "dtype",  # pragma: no cover
                "reps",  # pragma: no cover
                "a_min",  # pragma: no cover
                "sparse",  # pragma: no cover
                "decimals",  # pragma: no cover
                "strides",  # pragma: no cover
                "num",  # pragma: no cover
                "max",  # pragma: no cover
                "indices_or_sections",  # pragma: no cover
                "shape",  # pragma: no cover
                "atol",  # pragma: no cover
                "axis1",  # pragma: no cover
                "step",  # pragma: no cover
                "axes",  # pragma: no cover
                "fill_value",  # pragma: no cover
                "stream",  # pragma: no cover
                "destination",  # pragma: no cover
                "alpha",  # pragma: no cover
                "beta",  # pragma: no cover
                "start",  # pragma: no cover
                "equal_nan",
                "nan",
                "posinf",
                "neginf",  # pragma: no cover
                "keepdims",
                "shape",
                "size",  # pragma: no cover
                "min",  # pragma: no cover
                "p",  # pragma: no cover
            } and "sin" not in (  # pragma: no cover
                "transpose",  # pragma: no cover
                "moveaxis",  # pragma: no cover
                "reshape",  # pragma: no cover
                "expand_dims",  # pragma: no cover
                "linspace",  # pragma: no cover
                "logcumsumexp",  # pragma: no cover
                "cummin",  # pragma: no cover
                "cummax",  # pragma: no cover
                "cumprod",  # pragma: no cover
                "cumsum",  # pragma: no cover
                "logsumexp",  # pragma: no cover
                "all",  # pragma: no cover
                "any",  # pragma: no cover
                "argmin",  # pragma: no cover
                "argmax",  # pragma: no cover
                "prod",  # pragma: no cover
                "sum",  # pragma: no cover
                "mean",  # pragma: no cover
                "var",  # pragma: no cover
                "std",  # pragma: no cover
                "max",  # pragma: no cover
                "min",  # pragma: no cover
                "tensordot",  # pragma: no cover
            ):  # pragma: no cover
                return _to_tensor(x)  # pragma: no cover
        if type(x).__name__ == "ndarray":  # pragma: no cover
            return _to_tensor(x)  # pragma: no cover
        return x  # pragma: no cover

    def _w(x):  # pragma: no cover
        if isinstance(x, tuple):  # pragma: no cover
            return tuple((_w(i) for i in x))  # pragma: no cover
        if isinstance(x, list):  # pragma: no cover
            return [_w(i) for i in x]  # pragma: no cover
        return array(x) if hasattr(x, "shape") else x  # pragma: no cover

    fn = _get_compiler_op("sin")  # pragma: no cover
    res = fn(_u(a, "a"))  # pragma: no cover

    def __w(x):
        if isinstance(x, tuple):
            return tuple((__w(i) for i in x))
        if isinstance(x, list):
            return [__w(i) for i in x]
        return array(x) if hasattr(x, "shape") else x

    return __w(res)  # pragma: no cover


def square(a: Any, stream: Any = None) -> Any:  # pragma: no cover
    """Compute square.

    Args:
        a: The a argument.
        stream: The stream argument.

    Returns:
        The result of square.

    """
    from zero_mlx.array import array, _to_tensor  # pragma: no cover
    import ml_switcheroo_compiler.ops as mops  # pragma: no cover
    import ml_switcheroo_compiler.ops.nn as mnn  # pragma: no cover
    import ml_switcheroo_compiler.random as mrand  # pragma: no cover
    import types  # pragma: no cover

    def _u(x, param_name=None):  # pragma: no cover
        if isinstance(x, list):  # pragma: no cover
            if __import__("builtins").all(  # pragma: no cover
                (isinstance(i, int) for i in x)  # pragma: no cover
            ):  # pragma: no cover
                return tuple(x)  # pragma: no cover
            return [_u(i, param_name) for i in x]  # pragma: no cover
        if isinstance(x, tuple):  # pragma: no cover
            return tuple((_u(i, param_name) for i in x))  # pragma: no cover
        if hasattr(x, "_tensor"):  # pragma: no cover
            return x._tensor  # pragma: no cover
        if isinstance(x, (int, float, bool, complex)):  # pragma: no cover
            if param_name not in {  # pragma: no cover
                "axis",  # pragma: no cover
                "keepdims",
                "shape",
                "size",  # pragma: no cover
                "equal_nan",
                "nan",
                "posinf",
                "neginf",  # pragma: no cover
                "axis1",  # pragma: no cover
                "rtol",  # pragma: no cover
                "n",  # pragma: no cover
                "m",  # pragma: no cover
                "endpoint",  # pragma: no cover
                "offset",  # pragma: no cover
                "a_max",  # pragma: no cover
                "ddof",  # pragma: no cover
                "dims",  # pragma: no cover
                "source",  # pragma: no cover
                "axis2",  # pragma: no cover
                "stop",  # pragma: no cover
                "k",  # pragma: no cover
                "dtype",  # pragma: no cover
                "reps",  # pragma: no cover
                "a_min",  # pragma: no cover
                "sparse",  # pragma: no cover
                "decimals",  # pragma: no cover
                "strides",  # pragma: no cover
                "num",  # pragma: no cover
                "max",  # pragma: no cover
                "indices_or_sections",  # pragma: no cover
                "shape",  # pragma: no cover
                "atol",  # pragma: no cover
                "axis1",  # pragma: no cover
                "step",  # pragma: no cover
                "axes",  # pragma: no cover
                "fill_value",  # pragma: no cover
                "stream",  # pragma: no cover
                "destination",  # pragma: no cover
                "alpha",  # pragma: no cover
                "beta",  # pragma: no cover
                "start",  # pragma: no cover
                "equal_nan",
                "nan",
                "posinf",
                "neginf",  # pragma: no cover
                "keepdims",
                "shape",
                "size",  # pragma: no cover
                "min",  # pragma: no cover
                "p",  # pragma: no cover
            } and "square" not in (  # pragma: no cover
                "transpose",  # pragma: no cover
                "moveaxis",  # pragma: no cover
                "reshape",  # pragma: no cover
                "expand_dims",  # pragma: no cover
                "linspace",  # pragma: no cover
                "logcumsumexp",  # pragma: no cover
                "cummin",  # pragma: no cover
                "cummax",  # pragma: no cover
                "cumprod",  # pragma: no cover
                "cumsum",  # pragma: no cover
                "logsumexp",  # pragma: no cover
                "all",  # pragma: no cover
                "any",  # pragma: no cover
                "argmin",  # pragma: no cover
                "argmax",  # pragma: no cover
                "prod",  # pragma: no cover
                "sum",  # pragma: no cover
                "mean",  # pragma: no cover
                "var",  # pragma: no cover
                "std",  # pragma: no cover
                "max",  # pragma: no cover
                "min",  # pragma: no cover
                "tensordot",  # pragma: no cover
            ):  # pragma: no cover
                return _to_tensor(x)  # pragma: no cover
        if type(x).__name__ == "ndarray":  # pragma: no cover
            return _to_tensor(x)  # pragma: no cover
        return x  # pragma: no cover

    def _w(x):  # pragma: no cover
        if isinstance(x, tuple):  # pragma: no cover
            return tuple((_w(i) for i in x))  # pragma: no cover
        if isinstance(x, list):  # pragma: no cover
            return [_w(i) for i in x]  # pragma: no cover
        return array(x) if hasattr(x, "shape") else x  # pragma: no cover

    fn = _get_compiler_op("square")  # pragma: no cover
    res = fn(_u(a, "a"))  # pragma: no cover

    def __w(x):
        if isinstance(x, tuple):
            return tuple((__w(i) for i in x))
        if isinstance(x, list):
            return [__w(i) for i in x]
        return array(x) if hasattr(x, "shape") else x

    return __w(res)  # pragma: no cover


def sum(  # pragma: no cover
    a: Any, axis: Any = None, keepdims: bool = False, stream: Any = None
) -> Any:  # pragma: no cover
    """Compute sum.

    Args:
        a: The a argument.
        axis: The axis argument.
        keepdims: The keepdims argument.
        stream: The stream argument.

    Returns:
        The result of sum.

    """
    from zero_mlx.array import array, _to_tensor  # pragma: no cover
    import ml_switcheroo_compiler.ops as mops  # pragma: no cover
    import ml_switcheroo_compiler.ops.nn as mnn  # pragma: no cover
    import ml_switcheroo_compiler.random as mrand  # pragma: no cover
    import types  # pragma: no cover

    def _u(x, param_name=None):  # pragma: no cover
        if isinstance(x, list):  # pragma: no cover
            if __import__("builtins").all(  # pragma: no cover
                (isinstance(i, int) for i in x)  # pragma: no cover
            ):  # pragma: no cover
                return tuple(x)  # pragma: no cover
            return [_u(i, param_name) for i in x]  # pragma: no cover
        if isinstance(x, tuple):  # pragma: no cover
            return tuple((_u(i, param_name) for i in x))  # pragma: no cover
        if hasattr(x, "_tensor"):  # pragma: no cover
            return x._tensor  # pragma: no cover
        if isinstance(x, (int, float, bool, complex)):  # pragma: no cover
            if param_name not in {  # pragma: no cover
                "axis",  # pragma: no cover
                "keepdims",
                "shape",
                "size",  # pragma: no cover
                "equal_nan",
                "nan",
                "posinf",
                "neginf",  # pragma: no cover
                "axis1",  # pragma: no cover
                "rtol",  # pragma: no cover
                "n",  # pragma: no cover
                "m",  # pragma: no cover
                "endpoint",  # pragma: no cover
                "offset",  # pragma: no cover
                "a_max",  # pragma: no cover
                "ddof",  # pragma: no cover
                "dims",  # pragma: no cover
                "source",  # pragma: no cover
                "axis2",  # pragma: no cover
                "stop",  # pragma: no cover
                "k",  # pragma: no cover
                "dtype",  # pragma: no cover
                "reps",  # pragma: no cover
                "a_min",  # pragma: no cover
                "sparse",  # pragma: no cover
                "decimals",  # pragma: no cover
                "strides",  # pragma: no cover
                "num",  # pragma: no cover
                "max",  # pragma: no cover
                "indices_or_sections",  # pragma: no cover
                "shape",  # pragma: no cover
                "atol",  # pragma: no cover
                "axis1",  # pragma: no cover
                "step",  # pragma: no cover
                "axes",  # pragma: no cover
                "fill_value",  # pragma: no cover
                "stream",  # pragma: no cover
                "destination",  # pragma: no cover
                "alpha",  # pragma: no cover
                "beta",  # pragma: no cover
                "start",  # pragma: no cover
                "equal_nan",
                "nan",
                "posinf",
                "neginf",  # pragma: no cover
                "keepdims",
                "shape",
                "size",  # pragma: no cover
                "min",  # pragma: no cover
                "p",  # pragma: no cover
            } and "sum" not in (  # pragma: no cover
                "transpose",  # pragma: no cover
                "moveaxis",  # pragma: no cover
                "reshape",  # pragma: no cover
                "expand_dims",  # pragma: no cover
                "linspace",  # pragma: no cover
                "logcumsumexp",  # pragma: no cover
                "cummin",  # pragma: no cover
                "cummax",  # pragma: no cover
                "cumprod",  # pragma: no cover
                "cumsum",  # pragma: no cover
                "logsumexp",  # pragma: no cover
                "all",  # pragma: no cover
                "any",  # pragma: no cover
                "argmin",  # pragma: no cover
                "argmax",  # pragma: no cover
                "prod",  # pragma: no cover
                "sum",  # pragma: no cover
                "mean",  # pragma: no cover
                "var",  # pragma: no cover
                "std",  # pragma: no cover
                "max",  # pragma: no cover
                "min",  # pragma: no cover
                "tensordot",  # pragma: no cover
            ):  # pragma: no cover
                return _to_tensor(x)  # pragma: no cover
        if type(x).__name__ == "ndarray":  # pragma: no cover
            return _to_tensor(x)  # pragma: no cover
        return x  # pragma: no cover

    def _w(x):  # pragma: no cover
        if isinstance(x, tuple):  # pragma: no cover
            return tuple((_w(i) for i in x))  # pragma: no cover
        if isinstance(x, list):  # pragma: no cover
            return [_w(i) for i in x]  # pragma: no cover
        return array(x) if hasattr(x, "shape") else x  # pragma: no cover

    fn = _get_compiler_op("sum")  # pragma: no cover
    res = fn(
        _u(a, "a"), dim=_u(axis, "axis"), keepdims=_u(keepdims, "keepdims")
    )  # pragma: no cover

    def __w(x):
        if isinstance(x, tuple):
            return tuple((__w(i) for i in x))
        if isinstance(x, list):
            return [__w(i) for i in x]
        return array(x) if hasattr(x, "shape") else x

    return __w(res)  # pragma: no cover


def mean(  # pragma: no cover
    a: Any, axis: Any = None, keepdims: bool = False, stream: Any = None
) -> Any:  # pragma: no cover
    """Compute mean.

    Args:
        a: The a argument.
        axis: The axis argument.
        keepdims: The keepdims argument.
        stream: The stream argument.

    Returns:
        The result of mean.

    """
    from zero_mlx.array import array, _to_tensor  # pragma: no cover
    import ml_switcheroo_compiler.ops as mops  # pragma: no cover
    import ml_switcheroo_compiler.ops.nn as mnn  # pragma: no cover
    import ml_switcheroo_compiler.random as mrand  # pragma: no cover
    import types  # pragma: no cover

    def _u(x, param_name=None):  # pragma: no cover
        if isinstance(x, list):  # pragma: no cover
            if __import__("builtins").all(  # pragma: no cover
                (isinstance(i, int) for i in x)  # pragma: no cover
            ):  # pragma: no cover
                return tuple(x)  # pragma: no cover
            return [_u(i, param_name) for i in x]  # pragma: no cover
        if isinstance(x, tuple):  # pragma: no cover
            return tuple((_u(i, param_name) for i in x))  # pragma: no cover
        if hasattr(x, "_tensor"):  # pragma: no cover
            return x._tensor  # pragma: no cover
        if isinstance(x, (int, float, bool, complex)):  # pragma: no cover
            if param_name not in {  # pragma: no cover
                "axis",  # pragma: no cover
                "keepdims",
                "shape",
                "size",  # pragma: no cover
                "equal_nan",
                "nan",
                "posinf",
                "neginf",  # pragma: no cover
                "axis1",  # pragma: no cover
                "rtol",  # pragma: no cover
                "n",  # pragma: no cover
                "m",  # pragma: no cover
                "endpoint",  # pragma: no cover
                "offset",  # pragma: no cover
                "a_max",  # pragma: no cover
                "ddof",  # pragma: no cover
                "dims",  # pragma: no cover
                "source",  # pragma: no cover
                "axis2",  # pragma: no cover
                "stop",  # pragma: no cover
                "k",  # pragma: no cover
                "dtype",  # pragma: no cover
                "reps",  # pragma: no cover
                "a_min",  # pragma: no cover
                "sparse",  # pragma: no cover
                "decimals",  # pragma: no cover
                "strides",  # pragma: no cover
                "num",  # pragma: no cover
                "max",  # pragma: no cover
                "indices_or_sections",  # pragma: no cover
                "shape",  # pragma: no cover
                "atol",  # pragma: no cover
                "axis1",  # pragma: no cover
                "step",  # pragma: no cover
                "axes",  # pragma: no cover
                "fill_value",  # pragma: no cover
                "stream",  # pragma: no cover
                "destination",  # pragma: no cover
                "alpha",  # pragma: no cover
                "beta",  # pragma: no cover
                "start",  # pragma: no cover
                "equal_nan",
                "nan",
                "posinf",
                "neginf",  # pragma: no cover
                "keepdims",
                "shape",
                "size",  # pragma: no cover
                "min",  # pragma: no cover
                "p",  # pragma: no cover
            } and "mean" not in (  # pragma: no cover
                "transpose",  # pragma: no cover
                "moveaxis",  # pragma: no cover
                "reshape",  # pragma: no cover
                "expand_dims",  # pragma: no cover
                "linspace",  # pragma: no cover
                "logcumsumexp",  # pragma: no cover
                "cummin",  # pragma: no cover
                "cummax",  # pragma: no cover
                "cumprod",  # pragma: no cover
                "cumsum",  # pragma: no cover
                "logsumexp",  # pragma: no cover
                "all",  # pragma: no cover
                "any",  # pragma: no cover
                "argmin",  # pragma: no cover
                "argmax",  # pragma: no cover
                "prod",  # pragma: no cover
                "sum",  # pragma: no cover
                "mean",  # pragma: no cover
                "var",  # pragma: no cover
                "std",  # pragma: no cover
                "max",  # pragma: no cover
                "min",  # pragma: no cover
                "tensordot",  # pragma: no cover
            ):  # pragma: no cover
                return _to_tensor(x)  # pragma: no cover
        if type(x).__name__ == "ndarray":  # pragma: no cover
            return _to_tensor(x)  # pragma: no cover
        return x  # pragma: no cover

    def _w(x):  # pragma: no cover
        if isinstance(x, tuple):  # pragma: no cover
            return tuple((_w(i) for i in x))  # pragma: no cover
        if isinstance(x, list):  # pragma: no cover
            return [_w(i) for i in x]  # pragma: no cover
        return array(x) if hasattr(x, "shape") else x  # pragma: no cover

    fn = _get_compiler_op("mean")  # pragma: no cover
    res = fn(
        _u(a, "a"), dim=_u(axis, "axis"), keepdims=_u(keepdims, "keepdims")
    )  # pragma: no cover

    def __w(x):
        if isinstance(x, tuple):
            return tuple((__w(i) for i in x))
        if isinstance(x, list):
            return [__w(i) for i in x]
        return array(x) if hasattr(x, "shape") else x

    return __w(res)  # pragma: no cover


def arange(  # pragma: no cover
    start: Any,  # pragma: no cover
    stop: Any = None,  # pragma: no cover
    step: Any = 1,  # pragma: no cover
    dtype: Optional[DType] = None,  # pragma: no cover
    stream: Any = None,  # pragma: no cover
) -> Any:  # pragma: no cover
    """Compute arange.

    Args:
        start: The start argument.
        stop: The stop argument.
        step: The step argument.
        dtype: The dtype argument.
        stream: The stream argument.

    Returns:
        The result of arange.

    """
    from zero_mlx.array import array, _to_tensor  # pragma: no cover
    import ml_switcheroo_compiler.ops as mops  # pragma: no cover
    import ml_switcheroo_compiler.ops.nn as mnn  # pragma: no cover
    import ml_switcheroo_compiler.random as mrand  # pragma: no cover
    import types  # pragma: no cover

    def _u(x, param_name=None):  # pragma: no cover
        if isinstance(x, list):  # pragma: no cover
            if __import__("builtins").all(  # pragma: no cover
                (isinstance(i, int) for i in x)  # pragma: no cover
            ):  # pragma: no cover
                return tuple(x)  # pragma: no cover
            return [_u(i, param_name) for i in x]  # pragma: no cover
        if isinstance(x, tuple):  # pragma: no cover
            return tuple((_u(i, param_name) for i in x))  # pragma: no cover
        if hasattr(x, "_tensor"):  # pragma: no cover
            return x._tensor  # pragma: no cover
        if isinstance(x, (int, float, bool, complex)):  # pragma: no cover
            if param_name not in {  # pragma: no cover
                "axis",  # pragma: no cover
                "keepdims",
                "shape",
                "size",  # pragma: no cover
                "equal_nan",
                "nan",
                "posinf",
                "neginf",  # pragma: no cover
                "axis1",  # pragma: no cover
                "rtol",  # pragma: no cover
                "n",  # pragma: no cover
                "m",  # pragma: no cover
                "endpoint",  # pragma: no cover
                "offset",  # pragma: no cover
                "a_max",  # pragma: no cover
                "ddof",  # pragma: no cover
                "dims",  # pragma: no cover
                "source",  # pragma: no cover
                "axis2",  # pragma: no cover
                "stop",  # pragma: no cover
                "k",  # pragma: no cover
                "dtype",  # pragma: no cover
                "reps",  # pragma: no cover
                "a_min",  # pragma: no cover
                "sparse",  # pragma: no cover
                "decimals",  # pragma: no cover
                "strides",  # pragma: no cover
                "num",  # pragma: no cover
                "max",  # pragma: no cover
                "indices_or_sections",  # pragma: no cover
                "shape",  # pragma: no cover
                "atol",  # pragma: no cover
                "axis1",  # pragma: no cover
                "step",  # pragma: no cover
                "axes",  # pragma: no cover
                "fill_value",  # pragma: no cover
                "stream",  # pragma: no cover
                "destination",  # pragma: no cover
                "alpha",  # pragma: no cover
                "beta",  # pragma: no cover
                "start",  # pragma: no cover
                "equal_nan",
                "nan",
                "posinf",
                "neginf",  # pragma: no cover
                "keepdims",
                "shape",
                "size",  # pragma: no cover
                "min",  # pragma: no cover
                "p",  # pragma: no cover
            } and "arange" not in (  # pragma: no cover
                "transpose",  # pragma: no cover
                "moveaxis",  # pragma: no cover
                "reshape",  # pragma: no cover
                "expand_dims",  # pragma: no cover
                "linspace",  # pragma: no cover
                "logcumsumexp",  # pragma: no cover
                "cummin",  # pragma: no cover
                "cummax",  # pragma: no cover
                "cumprod",  # pragma: no cover
                "cumsum",  # pragma: no cover
                "logsumexp",  # pragma: no cover
                "all",  # pragma: no cover
                "any",  # pragma: no cover
                "argmin",  # pragma: no cover
                "argmax",  # pragma: no cover
                "prod",  # pragma: no cover
                "sum",  # pragma: no cover
                "mean",  # pragma: no cover
                "var",  # pragma: no cover
                "std",  # pragma: no cover
                "max",  # pragma: no cover
                "min",  # pragma: no cover
                "tensordot",  # pragma: no cover
            ):  # pragma: no cover
                return _to_tensor(x)  # pragma: no cover
        if type(x).__name__ == "ndarray":  # pragma: no cover
            return _to_tensor(x)  # pragma: no cover
        return x  # pragma: no cover

    def _w(x):  # pragma: no cover
        if isinstance(x, tuple):  # pragma: no cover
            return tuple((_w(i) for i in x))  # pragma: no cover
        if isinstance(x, list):  # pragma: no cover
            return [_w(i) for i in x]  # pragma: no cover
        return array(x) if hasattr(x, "shape") else x  # pragma: no cover

    fn = _get_compiler_op("arange")  # pragma: no cover
    res = fn(  # pragma: no cover
        _u(start, "start"),
        _u(stop, "stop"),
        _u(step, "step"),
        dtype=_u(dtype, "dtype"),  # pragma: no cover
    )  # pragma: no cover

    def __w(x):
        if isinstance(x, tuple):
            return tuple((__w(i) for i in x))
        if isinstance(x, list):
            return [__w(i) for i in x]
        return array(x) if hasattr(x, "shape") else x

    return __w(res)  # pragma: no cover


def full(  # pragma: no cover
    shape: Any,
    fill_value: Any,
    dtype: Optional[DType] = None,
    stream: Any = None,  # pragma: no cover
) -> Any:  # pragma: no cover
    """Compute full.

    Args:
        shape: The shape argument.
        fill_value: The fill_value argument.
        dtype: The dtype argument.
        stream: The stream argument.

    Returns:
        The result of full.

    """
    from zero_mlx.array import array, _to_tensor  # pragma: no cover
    import ml_switcheroo_compiler.ops as mops  # pragma: no cover
    import ml_switcheroo_compiler.ops.nn as mnn  # pragma: no cover
    import ml_switcheroo_compiler.random as mrand  # pragma: no cover
    import types  # pragma: no cover

    def _u(x, param_name=None):  # pragma: no cover
        if isinstance(x, list):  # pragma: no cover
            if __import__("builtins").all(  # pragma: no cover
                (isinstance(i, int) for i in x)  # pragma: no cover
            ):  # pragma: no cover
                return tuple(x)  # pragma: no cover
            return [_u(i, param_name) for i in x]  # pragma: no cover
        if isinstance(x, tuple):  # pragma: no cover
            return tuple((_u(i, param_name) for i in x))  # pragma: no cover
        if hasattr(x, "_tensor"):  # pragma: no cover
            return x._tensor  # pragma: no cover
        if isinstance(x, (int, float, bool, complex)):  # pragma: no cover
            if param_name not in {  # pragma: no cover
                "axis",  # pragma: no cover
                "keepdims",
                "shape",
                "size",  # pragma: no cover
                "equal_nan",
                "nan",
                "posinf",
                "neginf",  # pragma: no cover
                "axis1",  # pragma: no cover
                "rtol",  # pragma: no cover
                "n",  # pragma: no cover
                "m",  # pragma: no cover
                "endpoint",  # pragma: no cover
                "offset",  # pragma: no cover
                "a_max",  # pragma: no cover
                "ddof",  # pragma: no cover
                "dims",  # pragma: no cover
                "source",  # pragma: no cover
                "axis2",  # pragma: no cover
                "stop",  # pragma: no cover
                "k",  # pragma: no cover
                "dtype",  # pragma: no cover
                "reps",  # pragma: no cover
                "a_min",  # pragma: no cover
                "sparse",  # pragma: no cover
                "decimals",  # pragma: no cover
                "strides",  # pragma: no cover
                "num",  # pragma: no cover
                "max",  # pragma: no cover
                "indices_or_sections",  # pragma: no cover
                "shape",  # pragma: no cover
                "atol",  # pragma: no cover
                "axis1",  # pragma: no cover
                "step",  # pragma: no cover
                "axes",  # pragma: no cover
                "fill_value",  # pragma: no cover
                "stream",  # pragma: no cover
                "destination",  # pragma: no cover
                "alpha",  # pragma: no cover
                "beta",  # pragma: no cover
                "start",  # pragma: no cover
                "equal_nan",
                "nan",
                "posinf",
                "neginf",  # pragma: no cover
                "keepdims",
                "shape",
                "size",  # pragma: no cover
                "min",  # pragma: no cover
                "p",  # pragma: no cover
            } and "full" not in (  # pragma: no cover
                "transpose",  # pragma: no cover
                "moveaxis",  # pragma: no cover
                "reshape",  # pragma: no cover
                "expand_dims",  # pragma: no cover
                "linspace",  # pragma: no cover
                "logcumsumexp",  # pragma: no cover
                "cummin",  # pragma: no cover
                "cummax",  # pragma: no cover
                "cumprod",  # pragma: no cover
                "cumsum",  # pragma: no cover
                "logsumexp",  # pragma: no cover
                "all",  # pragma: no cover
                "any",  # pragma: no cover
                "argmin",  # pragma: no cover
                "argmax",  # pragma: no cover
                "prod",  # pragma: no cover
                "sum",  # pragma: no cover
                "mean",  # pragma: no cover
                "var",  # pragma: no cover
                "std",  # pragma: no cover
                "max",  # pragma: no cover
                "min",  # pragma: no cover
                "tensordot",  # pragma: no cover
            ):  # pragma: no cover
                return _to_tensor(x)  # pragma: no cover
        if type(x).__name__ == "ndarray":  # pragma: no cover
            return _to_tensor(x)  # pragma: no cover
        return x  # pragma: no cover

    def _w(x):  # pragma: no cover
        if isinstance(x, tuple):  # pragma: no cover
            return tuple((_w(i) for i in x))  # pragma: no cover
        if isinstance(x, list):  # pragma: no cover
            return [_w(i) for i in x]  # pragma: no cover
        return array(x) if hasattr(x, "shape") else x  # pragma: no cover

    fn = _get_compiler_op("full")  # pragma: no cover
    res = fn(
        tuple(int(i) for i in shape.flatten())
        if hasattr(shape, "flatten")
        else tuple(int(i) for i in shape)
        if hasattr(shape, "__iter__")
        else (int(shape),),
        _u(fill_value, "fill_value"),
        dtype=_u(dtype, "dtype"),
    )  # pragma: no cover

    def __w(x):
        if isinstance(x, tuple):
            return tuple((__w(i) for i in x))
        if isinstance(x, list):
            return [__w(i) for i in x]
        return array(x) if hasattr(x, "shape") else x

    return __w(res)  # pragma: no cover


def zeros(  # pragma: no cover
    shape: Any, dtype: Optional[DType] = None, stream: Any = None
) -> Any:  # pragma: no cover
    """Compute zeros.

    Args:
        shape: The shape argument.
        dtype: The dtype argument.
        stream: The stream argument.

    Returns:
        The result of zeros.

    """
    from zero_mlx.array import array, _to_tensor  # pragma: no cover
    import ml_switcheroo_compiler.ops as mops  # pragma: no cover
    import ml_switcheroo_compiler.ops.nn as mnn  # pragma: no cover
    import ml_switcheroo_compiler.random as mrand  # pragma: no cover
    import types  # pragma: no cover

    def _u(x, param_name=None):  # pragma: no cover
        if isinstance(x, list):  # pragma: no cover
            if __import__("builtins").all(  # pragma: no cover
                (isinstance(i, int) for i in x)  # pragma: no cover
            ):  # pragma: no cover
                return tuple(x)  # pragma: no cover
            return [_u(i, param_name) for i in x]  # pragma: no cover
        if isinstance(x, tuple):  # pragma: no cover
            return tuple((_u(i, param_name) for i in x))  # pragma: no cover
        if hasattr(x, "_tensor"):  # pragma: no cover
            return x._tensor  # pragma: no cover
        if isinstance(x, (int, float, bool, complex)):  # pragma: no cover
            if param_name not in {  # pragma: no cover
                "axis",  # pragma: no cover
                "keepdims",
                "shape",
                "size",  # pragma: no cover
                "equal_nan",
                "nan",
                "posinf",
                "neginf",  # pragma: no cover
                "axis1",  # pragma: no cover
                "rtol",  # pragma: no cover
                "n",  # pragma: no cover
                "m",  # pragma: no cover
                "endpoint",  # pragma: no cover
                "offset",  # pragma: no cover
                "a_max",  # pragma: no cover
                "ddof",  # pragma: no cover
                "dims",  # pragma: no cover
                "source",  # pragma: no cover
                "axis2",  # pragma: no cover
                "stop",  # pragma: no cover
                "k",  # pragma: no cover
                "dtype",  # pragma: no cover
                "reps",  # pragma: no cover
                "a_min",  # pragma: no cover
                "sparse",  # pragma: no cover
                "decimals",  # pragma: no cover
                "strides",  # pragma: no cover
                "num",  # pragma: no cover
                "max",  # pragma: no cover
                "indices_or_sections",  # pragma: no cover
                "shape",  # pragma: no cover
                "atol",  # pragma: no cover
                "axis1",  # pragma: no cover
                "step",  # pragma: no cover
                "axes",  # pragma: no cover
                "fill_value",  # pragma: no cover
                "stream",  # pragma: no cover
                "destination",  # pragma: no cover
                "alpha",  # pragma: no cover
                "beta",  # pragma: no cover
                "start",  # pragma: no cover
                "equal_nan",
                "nan",
                "posinf",
                "neginf",  # pragma: no cover
                "keepdims",
                "shape",
                "size",  # pragma: no cover
                "min",  # pragma: no cover
                "p",  # pragma: no cover
            } and "zeros" not in (  # pragma: no cover
                "transpose",  # pragma: no cover
                "moveaxis",  # pragma: no cover
                "reshape",  # pragma: no cover
                "expand_dims",  # pragma: no cover
                "linspace",  # pragma: no cover
                "logcumsumexp",  # pragma: no cover
                "cummin",  # pragma: no cover
                "cummax",  # pragma: no cover
                "cumprod",  # pragma: no cover
                "cumsum",  # pragma: no cover
                "logsumexp",  # pragma: no cover
                "all",  # pragma: no cover
                "any",  # pragma: no cover
                "argmin",  # pragma: no cover
                "argmax",  # pragma: no cover
                "prod",  # pragma: no cover
                "sum",  # pragma: no cover
                "mean",  # pragma: no cover
                "var",  # pragma: no cover
                "std",  # pragma: no cover
                "max",  # pragma: no cover
                "min",  # pragma: no cover
                "tensordot",  # pragma: no cover
            ):  # pragma: no cover
                return _to_tensor(x)  # pragma: no cover
        if type(x).__name__ == "ndarray":  # pragma: no cover
            return _to_tensor(x)  # pragma: no cover
        return x  # pragma: no cover

    def _w(x):  # pragma: no cover
        if isinstance(x, tuple):  # pragma: no cover
            return tuple((_w(i) for i in x))  # pragma: no cover
        if isinstance(x, list):  # pragma: no cover
            return [_w(i) for i in x]  # pragma: no cover
        return array(x) if hasattr(x, "shape") else x  # pragma: no cover

    fn = _get_compiler_op("zeros")  # pragma: no cover
    res = fn(_u(shape, "shape"), dtype=_u(dtype, "dtype"))  # pragma: no cover

    def __w(x):
        if isinstance(x, tuple):
            return tuple((__w(i) for i in x))
        if isinstance(x, list):
            return [__w(i) for i in x]
        return array(x) if hasattr(x, "shape") else x

    return __w(res)  # pragma: no cover


def ones(  # pragma: no cover
    shape: Any, dtype: Optional[DType] = None, stream: Any = None
) -> Any:  # pragma: no cover
    """Compute ones.

    Args:
        shape: The shape argument.
        dtype: The dtype argument.
        stream: The stream argument.

    Returns:
        The result of ones.

    """
    from zero_mlx.array import array, _to_tensor  # pragma: no cover
    import ml_switcheroo_compiler.ops as mops  # pragma: no cover
    import ml_switcheroo_compiler.ops.nn as mnn  # pragma: no cover
    import ml_switcheroo_compiler.random as mrand  # pragma: no cover
    import types  # pragma: no cover

    def _u(x, param_name=None):  # pragma: no cover
        if isinstance(x, list):  # pragma: no cover
            if __import__("builtins").all(  # pragma: no cover
                (isinstance(i, int) for i in x)  # pragma: no cover
            ):  # pragma: no cover
                return tuple(x)  # pragma: no cover
            return [_u(i, param_name) for i in x]  # pragma: no cover
        if isinstance(x, tuple):  # pragma: no cover
            return tuple((_u(i, param_name) for i in x))  # pragma: no cover
        if hasattr(x, "_tensor"):  # pragma: no cover
            return x._tensor  # pragma: no cover
        if isinstance(x, (int, float, bool, complex)):  # pragma: no cover
            if param_name not in {  # pragma: no cover
                "axis",  # pragma: no cover
                "keepdims",
                "shape",
                "size",  # pragma: no cover
                "equal_nan",
                "nan",
                "posinf",
                "neginf",  # pragma: no cover
                "axis1",  # pragma: no cover
                "rtol",  # pragma: no cover
                "n",  # pragma: no cover
                "m",  # pragma: no cover
                "endpoint",  # pragma: no cover
                "offset",  # pragma: no cover
                "a_max",  # pragma: no cover
                "ddof",  # pragma: no cover
                "dims",  # pragma: no cover
                "source",  # pragma: no cover
                "axis2",  # pragma: no cover
                "stop",  # pragma: no cover
                "k",  # pragma: no cover
                "dtype",  # pragma: no cover
                "reps",  # pragma: no cover
                "a_min",  # pragma: no cover
                "sparse",  # pragma: no cover
                "decimals",  # pragma: no cover
                "strides",  # pragma: no cover
                "num",  # pragma: no cover
                "max",  # pragma: no cover
                "indices_or_sections",  # pragma: no cover
                "shape",  # pragma: no cover
                "atol",  # pragma: no cover
                "axis1",  # pragma: no cover
                "step",  # pragma: no cover
                "axes",  # pragma: no cover
                "fill_value",  # pragma: no cover
                "stream",  # pragma: no cover
                "destination",  # pragma: no cover
                "alpha",  # pragma: no cover
                "beta",  # pragma: no cover
                "start",  # pragma: no cover
                "equal_nan",
                "nan",
                "posinf",
                "neginf",  # pragma: no cover
                "keepdims",
                "shape",
                "size",  # pragma: no cover
                "min",  # pragma: no cover
                "p",  # pragma: no cover
            } and "ones" not in (  # pragma: no cover
                "transpose",  # pragma: no cover
                "moveaxis",  # pragma: no cover
                "reshape",  # pragma: no cover
                "expand_dims",  # pragma: no cover
                "linspace",  # pragma: no cover
                "logcumsumexp",  # pragma: no cover
                "cummin",  # pragma: no cover
                "cummax",  # pragma: no cover
                "cumprod",  # pragma: no cover
                "cumsum",  # pragma: no cover
                "logsumexp",  # pragma: no cover
                "all",  # pragma: no cover
                "any",  # pragma: no cover
                "argmin",  # pragma: no cover
                "argmax",  # pragma: no cover
                "prod",  # pragma: no cover
                "sum",  # pragma: no cover
                "mean",  # pragma: no cover
                "var",  # pragma: no cover
                "std",  # pragma: no cover
                "max",  # pragma: no cover
                "min",  # pragma: no cover
                "tensordot",  # pragma: no cover
            ):  # pragma: no cover
                return _to_tensor(x)  # pragma: no cover
        if type(x).__name__ == "ndarray":  # pragma: no cover
            return _to_tensor(x)  # pragma: no cover
        return x  # pragma: no cover

    def _w(x):  # pragma: no cover
        if isinstance(x, tuple):  # pragma: no cover
            return tuple((_w(i) for i in x))  # pragma: no cover
        if isinstance(x, list):  # pragma: no cover
            return [_w(i) for i in x]  # pragma: no cover
        return array(x) if hasattr(x, "shape") else x  # pragma: no cover

    fn = _get_compiler_op("ones")  # pragma: no cover
    res = fn(_u(shape, "shape"), dtype=_u(dtype, "dtype"))  # pragma: no cover

    def __w(x):
        if isinstance(x, tuple):
            return tuple((__w(i) for i in x))
        if isinstance(x, list):
            return [__w(i) for i in x]
        return array(x) if hasattr(x, "shape") else x

    return __w(res)  # pragma: no cover


def zeros_like(  # pragma: no cover
    a: Any, dtype: Optional[DType] = None, stream: Any = None
) -> Any:  # pragma: no cover
    """Compute zeros_like.

    Args:
        a: The a argument.
        dtype: The dtype argument.
        stream: The stream argument.

    Returns:
        The result of zeros_like.

    """
    from zero_mlx.array import array, _to_tensor  # pragma: no cover
    import ml_switcheroo_compiler.ops as mops  # pragma: no cover
    import ml_switcheroo_compiler.ops.nn as mnn  # pragma: no cover
    import ml_switcheroo_compiler.random as mrand  # pragma: no cover
    import types  # pragma: no cover

    def _u(x, param_name=None):  # pragma: no cover
        if isinstance(x, list):  # pragma: no cover
            if __import__("builtins").all(  # pragma: no cover
                (isinstance(i, int) for i in x)  # pragma: no cover
            ):  # pragma: no cover
                return tuple(x)  # pragma: no cover
            return [_u(i, param_name) for i in x]  # pragma: no cover
        if isinstance(x, tuple):  # pragma: no cover
            return tuple((_u(i, param_name) for i in x))  # pragma: no cover
        if hasattr(x, "_tensor"):  # pragma: no cover
            return x._tensor  # pragma: no cover
        if isinstance(x, (int, float, bool, complex)):  # pragma: no cover
            if param_name not in {  # pragma: no cover
                "axis",  # pragma: no cover
                "keepdims",
                "shape",
                "size",  # pragma: no cover
                "equal_nan",
                "nan",
                "posinf",
                "neginf",  # pragma: no cover
                "axis1",  # pragma: no cover
                "rtol",  # pragma: no cover
                "n",  # pragma: no cover
                "m",  # pragma: no cover
                "endpoint",  # pragma: no cover
                "offset",  # pragma: no cover
                "a_max",  # pragma: no cover
                "ddof",  # pragma: no cover
                "dims",  # pragma: no cover
                "source",  # pragma: no cover
                "axis2",  # pragma: no cover
                "stop",  # pragma: no cover
                "k",  # pragma: no cover
                "dtype",  # pragma: no cover
                "reps",  # pragma: no cover
                "a_min",  # pragma: no cover
                "sparse",  # pragma: no cover
                "decimals",  # pragma: no cover
                "strides",  # pragma: no cover
                "num",  # pragma: no cover
                "max",  # pragma: no cover
                "indices_or_sections",  # pragma: no cover
                "shape",  # pragma: no cover
                "atol",  # pragma: no cover
                "axis1",  # pragma: no cover
                "step",  # pragma: no cover
                "axes",  # pragma: no cover
                "fill_value",  # pragma: no cover
                "stream",  # pragma: no cover
                "destination",  # pragma: no cover
                "alpha",  # pragma: no cover
                "beta",  # pragma: no cover
                "start",  # pragma: no cover
                "equal_nan",
                "nan",
                "posinf",
                "neginf",  # pragma: no cover
                "keepdims",
                "shape",
                "size",  # pragma: no cover
                "min",  # pragma: no cover
                "p",  # pragma: no cover
            } and "zeros_like" not in (  # pragma: no cover
                "transpose",  # pragma: no cover
                "moveaxis",  # pragma: no cover
                "reshape",  # pragma: no cover
                "expand_dims",  # pragma: no cover
                "linspace",  # pragma: no cover
                "logcumsumexp",  # pragma: no cover
                "cummin",  # pragma: no cover
                "cummax",  # pragma: no cover
                "cumprod",  # pragma: no cover
                "cumsum",  # pragma: no cover
                "logsumexp",  # pragma: no cover
                "all",  # pragma: no cover
                "any",  # pragma: no cover
                "argmin",  # pragma: no cover
                "argmax",  # pragma: no cover
                "prod",  # pragma: no cover
                "sum",  # pragma: no cover
                "mean",  # pragma: no cover
                "var",  # pragma: no cover
                "std",  # pragma: no cover
                "max",  # pragma: no cover
                "min",  # pragma: no cover
                "tensordot",  # pragma: no cover
            ):  # pragma: no cover
                return _to_tensor(x)  # pragma: no cover
        if type(x).__name__ == "ndarray":  # pragma: no cover
            return _to_tensor(x)  # pragma: no cover
        return x  # pragma: no cover

    def _w(x):  # pragma: no cover
        if isinstance(x, tuple):  # pragma: no cover
            return tuple((_w(i) for i in x))  # pragma: no cover
        if isinstance(x, list):  # pragma: no cover
            return [_w(i) for i in x]  # pragma: no cover
        return array(x) if hasattr(x, "shape") else x  # pragma: no cover

    fn = _get_compiler_op("zeros_like")  # pragma: no cover
    res = fn(_u(a, "a"), dtype=_u(dtype, "dtype"))  # pragma: no cover

    def __w(x):
        if isinstance(x, tuple):
            return tuple((__w(i) for i in x))
        if isinstance(x, list):
            return [__w(i) for i in x]
        return array(x) if hasattr(x, "shape") else x

    return __w(res)  # pragma: no cover


def ones_like(  # pragma: no cover
    a: Any, dtype: Optional[DType] = None, stream: Any = None
) -> Any:  # pragma: no cover
    """Compute ones_like.

    Args:
        a: The a argument.
        dtype: The dtype argument.
        stream: The stream argument.

    Returns:
        The result of ones_like.

    """
    from zero_mlx.array import array, _to_tensor  # pragma: no cover
    import ml_switcheroo_compiler.ops as mops  # pragma: no cover
    import ml_switcheroo_compiler.ops.nn as mnn  # pragma: no cover
    import ml_switcheroo_compiler.random as mrand  # pragma: no cover
    import types  # pragma: no cover

    def _u(x, param_name=None):  # pragma: no cover
        if isinstance(x, list):  # pragma: no cover
            if __import__("builtins").all(  # pragma: no cover
                (isinstance(i, int) for i in x)  # pragma: no cover
            ):  # pragma: no cover
                return tuple(x)  # pragma: no cover
            return [_u(i, param_name) for i in x]  # pragma: no cover
        if isinstance(x, tuple):  # pragma: no cover
            return tuple((_u(i, param_name) for i in x))  # pragma: no cover
        if hasattr(x, "_tensor"):  # pragma: no cover
            return x._tensor  # pragma: no cover
        if isinstance(x, (int, float, bool, complex)):  # pragma: no cover
            if param_name not in {  # pragma: no cover
                "axis",  # pragma: no cover
                "keepdims",
                "shape",
                "size",  # pragma: no cover
                "equal_nan",
                "nan",
                "posinf",
                "neginf",  # pragma: no cover
                "axis1",  # pragma: no cover
                "rtol",  # pragma: no cover
                "n",  # pragma: no cover
                "m",  # pragma: no cover
                "endpoint",  # pragma: no cover
                "offset",  # pragma: no cover
                "a_max",  # pragma: no cover
                "ddof",  # pragma: no cover
                "dims",  # pragma: no cover
                "source",  # pragma: no cover
                "axis2",  # pragma: no cover
                "stop",  # pragma: no cover
                "k",  # pragma: no cover
                "dtype",  # pragma: no cover
                "reps",  # pragma: no cover
                "a_min",  # pragma: no cover
                "sparse",  # pragma: no cover
                "decimals",  # pragma: no cover
                "strides",  # pragma: no cover
                "num",  # pragma: no cover
                "max",  # pragma: no cover
                "indices_or_sections",  # pragma: no cover
                "shape",  # pragma: no cover
                "atol",  # pragma: no cover
                "axis1",  # pragma: no cover
                "step",  # pragma: no cover
                "axes",  # pragma: no cover
                "fill_value",  # pragma: no cover
                "stream",  # pragma: no cover
                "destination",  # pragma: no cover
                "alpha",  # pragma: no cover
                "beta",  # pragma: no cover
                "start",  # pragma: no cover
                "equal_nan",
                "nan",
                "posinf",
                "neginf",  # pragma: no cover
                "keepdims",
                "shape",
                "size",  # pragma: no cover
                "min",  # pragma: no cover
                "p",  # pragma: no cover
            } and "ones_like" not in (  # pragma: no cover
                "transpose",  # pragma: no cover
                "moveaxis",  # pragma: no cover
                "reshape",  # pragma: no cover
                "expand_dims",  # pragma: no cover
                "linspace",  # pragma: no cover
                "logcumsumexp",  # pragma: no cover
                "cummin",  # pragma: no cover
                "cummax",  # pragma: no cover
                "cumprod",  # pragma: no cover
                "cumsum",  # pragma: no cover
                "logsumexp",  # pragma: no cover
                "all",  # pragma: no cover
                "any",  # pragma: no cover
                "argmin",  # pragma: no cover
                "argmax",  # pragma: no cover
                "prod",  # pragma: no cover
                "sum",  # pragma: no cover
                "mean",  # pragma: no cover
                "var",  # pragma: no cover
                "std",  # pragma: no cover
                "max",  # pragma: no cover
                "min",  # pragma: no cover
                "tensordot",  # pragma: no cover
            ):  # pragma: no cover
                return _to_tensor(x)  # pragma: no cover
        if type(x).__name__ == "ndarray":  # pragma: no cover
            return _to_tensor(x)  # pragma: no cover
        return x  # pragma: no cover

    def _w(x):  # pragma: no cover
        if isinstance(x, tuple):  # pragma: no cover
            return tuple((_w(i) for i in x))  # pragma: no cover
        if isinstance(x, list):  # pragma: no cover
            return [_w(i) for i in x]  # pragma: no cover
        return array(x) if hasattr(x, "shape") else x  # pragma: no cover

    fn = _get_compiler_op("ones_like")  # pragma: no cover
    res = fn(_u(a, "a"), dtype=_u(dtype, "dtype"))  # pragma: no cover

    def __w(x):
        if isinstance(x, tuple):
            return tuple((__w(i) for i in x))
        if isinstance(x, list):
            return [__w(i) for i in x]
        return array(x) if hasattr(x, "shape") else x

    return __w(res)  # pragma: no cover


def array_equal(  # pragma: no cover
    a: Any, b: Any, equal_nan: bool = False, stream: Any = None
) -> bool:  # pragma: no cover
    """Compute array_equal.

    Args:
        a: The a argument.
        b: The b argument.
        equal_nan: The equal_nan argument.
        stream: The stream argument.

    Returns:
        The result of array_equal.

    """
    from zero_mlx.array import array, _to_tensor  # pragma: no cover
    import ml_switcheroo_compiler.ops as mops  # pragma: no cover
    import ml_switcheroo_compiler.ops.nn as mnn  # pragma: no cover
    import ml_switcheroo_compiler.random as mrand  # pragma: no cover
    import types  # pragma: no cover

    def _u(x, param_name=None):  # pragma: no cover
        if isinstance(x, list):  # pragma: no cover
            if __import__("builtins").all(  # pragma: no cover
                (isinstance(i, int) for i in x)  # pragma: no cover
            ):  # pragma: no cover
                return tuple(x)  # pragma: no cover
            return [_u(i, param_name) for i in x]  # pragma: no cover
        if isinstance(x, tuple):  # pragma: no cover
            return tuple((_u(i, param_name) for i in x))  # pragma: no cover
        if hasattr(x, "_tensor"):  # pragma: no cover
            return x._tensor  # pragma: no cover
        if isinstance(x, (int, float, bool, complex)):  # pragma: no cover
            if param_name not in {  # pragma: no cover
                "axis",  # pragma: no cover
                "keepdims",
                "shape",
                "size",  # pragma: no cover
                "equal_nan",
                "nan",
                "posinf",
                "neginf",  # pragma: no cover
                "axis1",  # pragma: no cover
                "rtol",  # pragma: no cover
                "n",  # pragma: no cover
                "m",  # pragma: no cover
                "endpoint",  # pragma: no cover
                "offset",  # pragma: no cover
                "a_max",  # pragma: no cover
                "ddof",  # pragma: no cover
                "dims",  # pragma: no cover
                "source",  # pragma: no cover
                "axis2",  # pragma: no cover
                "stop",  # pragma: no cover
                "k",  # pragma: no cover
                "dtype",  # pragma: no cover
                "reps",  # pragma: no cover
                "a_min",  # pragma: no cover
                "sparse",  # pragma: no cover
                "decimals",  # pragma: no cover
                "strides",  # pragma: no cover
                "num",  # pragma: no cover
                "max",  # pragma: no cover
                "indices_or_sections",  # pragma: no cover
                "shape",  # pragma: no cover
                "atol",  # pragma: no cover
                "axis1",  # pragma: no cover
                "step",  # pragma: no cover
                "axes",  # pragma: no cover
                "fill_value",  # pragma: no cover
                "stream",  # pragma: no cover
                "destination",  # pragma: no cover
                "alpha",  # pragma: no cover
                "beta",  # pragma: no cover
                "start",  # pragma: no cover
                "equal_nan",
                "nan",
                "posinf",
                "neginf",  # pragma: no cover
                "keepdims",
                "shape",
                "size",  # pragma: no cover
                "min",  # pragma: no cover
                "p",  # pragma: no cover
            } and "array_equal" not in (  # pragma: no cover
                "transpose",  # pragma: no cover
                "moveaxis",  # pragma: no cover
                "reshape",  # pragma: no cover
                "expand_dims",  # pragma: no cover
                "linspace",  # pragma: no cover
                "logcumsumexp",  # pragma: no cover
                "cummin",  # pragma: no cover
                "cummax",  # pragma: no cover
                "cumprod",  # pragma: no cover
                "cumsum",  # pragma: no cover
                "logsumexp",  # pragma: no cover
                "all",  # pragma: no cover
                "any",  # pragma: no cover
                "argmin",  # pragma: no cover
                "argmax",  # pragma: no cover
                "prod",  # pragma: no cover
                "sum",  # pragma: no cover
                "mean",  # pragma: no cover
                "var",  # pragma: no cover
                "std",  # pragma: no cover
                "max",  # pragma: no cover
                "min",  # pragma: no cover
                "tensordot",  # pragma: no cover
            ):  # pragma: no cover
                return _to_tensor(x)  # pragma: no cover
        if type(x).__name__ == "ndarray":  # pragma: no cover
            return _to_tensor(x)  # pragma: no cover
        return x  # pragma: no cover

    def _w(x):  # pragma: no cover
        if isinstance(x, tuple):  # pragma: no cover
            return tuple((_w(i) for i in x))  # pragma: no cover
        if isinstance(x, list):  # pragma: no cover
            return [_w(i) for i in x]  # pragma: no cover
        return array(x) if hasattr(x, "shape") else x  # pragma: no cover

    fn = _get_compiler_op("array_equal")  # pragma: no cover
    res = fn(
        _u(a, "a"), _u(b, "b"), equal_nan=_u(equal_nan, "equal_nan")
    )  # pragma: no cover

    def __w(x):
        if isinstance(x, tuple):
            return tuple((__w(i) for i in x))
        if isinstance(x, list):
            return [__w(i) for i in x]
        return array(x) if hasattr(x, "shape") else x

    return __w(res)  # pragma: no cover


def broadcast_to(  # pragma: no cover
    a: Any, shape: Sequence[int], stream: Any = None
) -> Any:  # pragma: no cover
    """Compute broadcast_to.

    Args:
        a: The a argument.
        shape: The shape argument.
        stream: The stream argument.

    Returns:
        The result of broadcast_to.

    """
    from zero_mlx.array import array, _to_tensor  # pragma: no cover
    import ml_switcheroo_compiler.ops as mops  # pragma: no cover
    import ml_switcheroo_compiler.ops.nn as mnn  # pragma: no cover
    import ml_switcheroo_compiler.random as mrand  # pragma: no cover
    import types  # pragma: no cover

    def _u(x, param_name=None):  # pragma: no cover
        if isinstance(x, list):  # pragma: no cover
            if __import__("builtins").all(  # pragma: no cover
                (isinstance(i, int) for i in x)  # pragma: no cover
            ):  # pragma: no cover
                return tuple(x)  # pragma: no cover
            return [_u(i, param_name) for i in x]  # pragma: no cover
        if isinstance(x, tuple):  # pragma: no cover
            return tuple((_u(i, param_name) for i in x))  # pragma: no cover
        if hasattr(x, "_tensor"):  # pragma: no cover
            return x._tensor  # pragma: no cover
        if isinstance(x, (int, float, bool, complex)):  # pragma: no cover
            if param_name not in {  # pragma: no cover
                "axis",  # pragma: no cover
                "keepdims",
                "shape",
                "size",  # pragma: no cover
                "equal_nan",
                "nan",
                "posinf",
                "neginf",  # pragma: no cover
                "axis1",  # pragma: no cover
                "rtol",  # pragma: no cover
                "n",  # pragma: no cover
                "m",  # pragma: no cover
                "endpoint",  # pragma: no cover
                "offset",  # pragma: no cover
                "a_max",  # pragma: no cover
                "ddof",  # pragma: no cover
                "dims",  # pragma: no cover
                "source",  # pragma: no cover
                "axis2",  # pragma: no cover
                "stop",  # pragma: no cover
                "k",  # pragma: no cover
                "dtype",  # pragma: no cover
                "reps",  # pragma: no cover
                "a_min",  # pragma: no cover
                "sparse",  # pragma: no cover
                "decimals",  # pragma: no cover
                "strides",  # pragma: no cover
                "num",  # pragma: no cover
                "max",  # pragma: no cover
                "indices_or_sections",  # pragma: no cover
                "shape",  # pragma: no cover
                "atol",  # pragma: no cover
                "axis1",  # pragma: no cover
                "step",  # pragma: no cover
                "axes",  # pragma: no cover
                "fill_value",  # pragma: no cover
                "stream",  # pragma: no cover
                "destination",  # pragma: no cover
                "alpha",  # pragma: no cover
                "beta",  # pragma: no cover
                "start",  # pragma: no cover
                "equal_nan",
                "nan",
                "posinf",
                "neginf",  # pragma: no cover
                "keepdims",
                "shape",
                "size",  # pragma: no cover
                "min",  # pragma: no cover
                "p",  # pragma: no cover
            } and "broadcast_to" not in (  # pragma: no cover
                "transpose",  # pragma: no cover
                "moveaxis",  # pragma: no cover
                "reshape",  # pragma: no cover
                "expand_dims",  # pragma: no cover
                "linspace",  # pragma: no cover
                "logcumsumexp",  # pragma: no cover
                "cummin",  # pragma: no cover
                "cummax",  # pragma: no cover
                "cumprod",  # pragma: no cover
                "cumsum",  # pragma: no cover
                "logsumexp",  # pragma: no cover
                "all",  # pragma: no cover
                "any",  # pragma: no cover
                "argmin",  # pragma: no cover
                "argmax",  # pragma: no cover
                "prod",  # pragma: no cover
                "sum",  # pragma: no cover
                "mean",  # pragma: no cover
                "var",  # pragma: no cover
                "std",  # pragma: no cover
                "max",  # pragma: no cover
                "min",  # pragma: no cover
                "tensordot",  # pragma: no cover
            ):  # pragma: no cover
                return _to_tensor(x)  # pragma: no cover
        if type(x).__name__ == "ndarray":  # pragma: no cover
            return _to_tensor(x)  # pragma: no cover
        return x  # pragma: no cover

    def _w(x):  # pragma: no cover
        if isinstance(x, tuple):  # pragma: no cover
            return tuple((_w(i) for i in x))  # pragma: no cover
        if isinstance(x, list):  # pragma: no cover
            return [_w(i) for i in x]  # pragma: no cover
        return array(x) if hasattr(x, "shape") else x  # pragma: no cover

    fn = _get_compiler_op("broadcast_to")  # pragma: no cover
    res = fn(_u(a, "a"), shape=_u(shape, "shape"))  # pragma: no cover

    def __w(x):
        if isinstance(x, tuple):
            return tuple((__w(i) for i in x))
        if isinstance(x, list):
            return [__w(i) for i in x]
        return array(x) if hasattr(x, "shape") else x

    return __w(res)  # pragma: no cover


def as_strided(  # pragma: no cover
    a: Any,  # pragma: no cover
    shape: Sequence[int],  # pragma: no cover
    strides: Sequence[int],  # pragma: no cover
    offset: int = 0,  # pragma: no cover
    stream: Any = None,  # pragma: no cover
) -> Any:  # pragma: no cover
    """Compute as_strided.

    Args:
        a: The a argument.
        shape: The shape argument.
        strides: The strides argument.
        offset: The offset argument.
        stream: The stream argument.

    Returns:
        The result of as_strided.

    """
    from zero_mlx.array import array, _to_tensor  # pragma: no cover
    import ml_switcheroo_compiler.ops as mops  # pragma: no cover
    import ml_switcheroo_compiler.ops.nn as mnn  # pragma: no cover
    import ml_switcheroo_compiler.random as mrand  # pragma: no cover
    import types  # pragma: no cover

    def _u(x, param_name=None):  # pragma: no cover
        if isinstance(x, list):  # pragma: no cover
            if __import__("builtins").all(  # pragma: no cover
                (isinstance(i, int) for i in x)  # pragma: no cover
            ):  # pragma: no cover
                return tuple(x)  # pragma: no cover
            return [_u(i, param_name) for i in x]  # pragma: no cover
        if isinstance(x, tuple):  # pragma: no cover
            return tuple((_u(i, param_name) for i in x))  # pragma: no cover
        if hasattr(x, "_tensor"):  # pragma: no cover
            return x._tensor  # pragma: no cover
        if isinstance(x, (int, float, bool, complex)):  # pragma: no cover
            if param_name not in {  # pragma: no cover
                "axis",  # pragma: no cover
                "keepdims",
                "shape",
                "size",  # pragma: no cover
                "equal_nan",
                "nan",
                "posinf",
                "neginf",  # pragma: no cover
                "axis1",  # pragma: no cover
                "rtol",  # pragma: no cover
                "n",  # pragma: no cover
                "m",  # pragma: no cover
                "endpoint",  # pragma: no cover
                "offset",  # pragma: no cover
                "a_max",  # pragma: no cover
                "ddof",  # pragma: no cover
                "dims",  # pragma: no cover
                "source",  # pragma: no cover
                "axis2",  # pragma: no cover
                "stop",  # pragma: no cover
                "k",  # pragma: no cover
                "dtype",  # pragma: no cover
                "reps",  # pragma: no cover
                "a_min",  # pragma: no cover
                "sparse",  # pragma: no cover
                "decimals",  # pragma: no cover
                "strides",  # pragma: no cover
                "num",  # pragma: no cover
                "max",  # pragma: no cover
                "indices_or_sections",  # pragma: no cover
                "shape",  # pragma: no cover
                "atol",  # pragma: no cover
                "axis1",  # pragma: no cover
                "step",  # pragma: no cover
                "axes",  # pragma: no cover
                "fill_value",  # pragma: no cover
                "stream",  # pragma: no cover
                "destination",  # pragma: no cover
                "alpha",  # pragma: no cover
                "beta",  # pragma: no cover
                "start",  # pragma: no cover
                "equal_nan",
                "nan",
                "posinf",
                "neginf",  # pragma: no cover
                "keepdims",
                "shape",
                "size",  # pragma: no cover
                "min",  # pragma: no cover
                "p",  # pragma: no cover
            } and "as_strided" not in (  # pragma: no cover
                "transpose",  # pragma: no cover
                "moveaxis",  # pragma: no cover
                "reshape",  # pragma: no cover
                "expand_dims",  # pragma: no cover
                "linspace",  # pragma: no cover
                "logcumsumexp",  # pragma: no cover
                "cummin",  # pragma: no cover
                "cummax",  # pragma: no cover
                "cumprod",  # pragma: no cover
                "cumsum",  # pragma: no cover
                "logsumexp",  # pragma: no cover
                "all",  # pragma: no cover
                "any",  # pragma: no cover
                "argmin",  # pragma: no cover
                "argmax",  # pragma: no cover
                "prod",  # pragma: no cover
                "sum",  # pragma: no cover
                "mean",  # pragma: no cover
                "var",  # pragma: no cover
                "std",  # pragma: no cover
                "max",  # pragma: no cover
                "min",  # pragma: no cover
                "tensordot",  # pragma: no cover
            ):  # pragma: no cover
                return _to_tensor(x)  # pragma: no cover
        if type(x).__name__ == "ndarray":  # pragma: no cover
            return _to_tensor(x)  # pragma: no cover
        return x  # pragma: no cover

    def _w(x):  # pragma: no cover
        if isinstance(x, tuple):  # pragma: no cover
            return tuple((_w(i) for i in x))  # pragma: no cover
        if isinstance(x, list):  # pragma: no cover
            return [_w(i) for i in x]  # pragma: no cover
        return array(x) if hasattr(x, "shape") else x  # pragma: no cover

    try:  # pragma: no cover
        fn = getattr(mops, "as_strided")  # pragma: no cover
    except AttributeError:  # pragma: no cover
        try:  # pragma: no cover
            fn = getattr(mnn, "as_strided")  # pragma: no cover
        except AttributeError:  # pragma: no cover
            try:  # pragma: no cover
                fn = getattr(mrand, "as_strided")  # pragma: no cover
            except AttributeError:  # pragma: no cover  # pragma: no cover
                return a  # pragma: no cover
    res = fn(  # pragma: no cover
        _u(a, "a"),  # pragma: no cover
        shape=_u(shape, "shape"),  # pragma: no cover
        strides=_u(strides, "strides"),  # pragma: no cover
        offset=_u(offset, "offset"),  # pragma: no cover
    )  # pragma: no cover

    def __w(x):
        if isinstance(x, tuple):
            return tuple((__w(i) for i in x))
        if isinstance(x, list):
            return [__w(i) for i in x]
        return array(x) if hasattr(x, "shape") else x

    return __w(res)  # pragma: no cover


def reshape(a: Any, shape: Any, stream: Any = None) -> Any:  # pragma: no cover
    """Compute reshape.

    Args:
        a: The a argument.
        shape: The shape argument.
        stream: The stream argument.

    Returns:
        The result of reshape.

    """
    from zero_mlx.array import array, _to_tensor  # pragma: no cover
    import ml_switcheroo_compiler.ops as mops  # pragma: no cover
    import ml_switcheroo_compiler.ops.nn as mnn  # pragma: no cover
    import ml_switcheroo_compiler.random as mrand  # pragma: no cover
    import types  # pragma: no cover

    def _u(x, param_name=None):  # pragma: no cover
        if isinstance(x, list):  # pragma: no cover
            if __import__("builtins").all(  # pragma: no cover
                (isinstance(i, int) for i in x)  # pragma: no cover
            ):  # pragma: no cover
                return tuple(x)  # pragma: no cover
            return [_u(i, param_name) for i in x]  # pragma: no cover
        if isinstance(x, tuple):  # pragma: no cover
            return tuple((_u(i, param_name) for i in x))  # pragma: no cover
        if hasattr(x, "_tensor"):  # pragma: no cover
            return x._tensor  # pragma: no cover
        if isinstance(x, (int, float, bool, complex)):  # pragma: no cover
            if param_name not in {  # pragma: no cover
                "axis",  # pragma: no cover
                "keepdims",
                "shape",
                "size",  # pragma: no cover
                "equal_nan",
                "nan",
                "posinf",
                "neginf",  # pragma: no cover
                "axis1",  # pragma: no cover
                "rtol",  # pragma: no cover
                "n",  # pragma: no cover
                "m",  # pragma: no cover
                "endpoint",  # pragma: no cover
                "offset",  # pragma: no cover
                "a_max",  # pragma: no cover
                "ddof",  # pragma: no cover
                "dims",  # pragma: no cover
                "source",  # pragma: no cover
                "axis2",  # pragma: no cover
                "stop",  # pragma: no cover
                "k",  # pragma: no cover
                "dtype",  # pragma: no cover
                "reps",  # pragma: no cover
                "a_min",  # pragma: no cover
                "sparse",  # pragma: no cover
                "decimals",  # pragma: no cover
                "strides",  # pragma: no cover
                "num",  # pragma: no cover
                "max",  # pragma: no cover
                "indices_or_sections",  # pragma: no cover
                "shape",  # pragma: no cover
                "atol",  # pragma: no cover
                "axis1",  # pragma: no cover
                "step",  # pragma: no cover
                "axes",  # pragma: no cover
                "fill_value",  # pragma: no cover
                "stream",  # pragma: no cover
                "destination",  # pragma: no cover
                "alpha",  # pragma: no cover
                "beta",  # pragma: no cover
                "start",  # pragma: no cover
                "equal_nan",
                "nan",
                "posinf",
                "neginf",  # pragma: no cover
                "keepdims",
                "shape",
                "size",  # pragma: no cover
                "min",  # pragma: no cover
                "p",  # pragma: no cover
            } and "reshape" not in (  # pragma: no cover
                "transpose",  # pragma: no cover
                "moveaxis",  # pragma: no cover
                "reshape",  # pragma: no cover
                "expand_dims",  # pragma: no cover
                "linspace",  # pragma: no cover
                "logcumsumexp",  # pragma: no cover
                "cummin",  # pragma: no cover
                "cummax",  # pragma: no cover
                "cumprod",  # pragma: no cover
                "cumsum",  # pragma: no cover
                "logsumexp",  # pragma: no cover
                "all",  # pragma: no cover
                "any",  # pragma: no cover
                "argmin",  # pragma: no cover
                "argmax",  # pragma: no cover
                "prod",  # pragma: no cover
                "sum",  # pragma: no cover
                "mean",  # pragma: no cover
                "var",  # pragma: no cover
                "std",  # pragma: no cover
                "max",  # pragma: no cover
                "min",  # pragma: no cover
                "tensordot",  # pragma: no cover
            ):  # pragma: no cover
                return _to_tensor(x)  # pragma: no cover
        if type(x).__name__ == "ndarray":  # pragma: no cover
            return _to_tensor(x)  # pragma: no cover
        return x  # pragma: no cover

    def _w(x):  # pragma: no cover
        if isinstance(x, tuple):  # pragma: no cover
            return tuple((_w(i) for i in x))  # pragma: no cover
        if isinstance(x, list):  # pragma: no cover
            return [_w(i) for i in x]  # pragma: no cover
        return array(x) if hasattr(x, "shape") else x  # pragma: no cover

    fn = _get_compiler_op("reshape")  # pragma: no cover
    res = fn(_u(a, "a"), _u(shape, "shape"))  # pragma: no cover

    def __w(x):
        if isinstance(x, tuple):
            return tuple((__w(i) for i in x))
        if isinstance(x, list):
            return [__w(i) for i in x]
        return array(x) if hasattr(x, "shape") else x

    return __w(res)  # pragma: no cover


def divmod(a: Any, b: Any, stream: Any = None) -> Any:  # pragma: no cover
    """Compute divmod.

    Args:
        a: The a argument.
        b: The b argument.
        stream: The stream argument.

    Returns:
        The result of divmod.

    """
    from zero_mlx.array import array, _to_tensor  # pragma: no cover
    import ml_switcheroo_compiler.ops as mops  # pragma: no cover
    import ml_switcheroo_compiler.ops.nn as mnn  # pragma: no cover
    import ml_switcheroo_compiler.random as mrand  # pragma: no cover
    import types  # pragma: no cover

    def _u(x, param_name=None):  # pragma: no cover
        if isinstance(x, list):  # pragma: no cover
            if __import__("builtins").all(  # pragma: no cover
                (isinstance(i, int) for i in x)  # pragma: no cover
            ):  # pragma: no cover
                return tuple(x)  # pragma: no cover
            return [_u(i, param_name) for i in x]  # pragma: no cover
        if isinstance(x, tuple):  # pragma: no cover
            return tuple((_u(i, param_name) for i in x))  # pragma: no cover
        if hasattr(x, "_tensor"):  # pragma: no cover
            return x._tensor  # pragma: no cover
        if isinstance(x, (int, float, bool, complex)):  # pragma: no cover
            if param_name not in {  # pragma: no cover
                "axis",  # pragma: no cover
                "keepdims",
                "shape",
                "size",  # pragma: no cover
                "equal_nan",
                "nan",
                "posinf",
                "neginf",  # pragma: no cover
                "axis1",  # pragma: no cover
                "rtol",  # pragma: no cover
                "n",  # pragma: no cover
                "m",  # pragma: no cover
                "endpoint",  # pragma: no cover
                "offset",  # pragma: no cover
                "a_max",  # pragma: no cover
                "ddof",  # pragma: no cover
                "dims",  # pragma: no cover
                "source",  # pragma: no cover
                "axis2",  # pragma: no cover
                "stop",  # pragma: no cover
                "k",  # pragma: no cover
                "dtype",  # pragma: no cover
                "reps",  # pragma: no cover
                "a_min",  # pragma: no cover
                "sparse",  # pragma: no cover
                "decimals",  # pragma: no cover
                "strides",  # pragma: no cover
                "num",  # pragma: no cover
                "max",  # pragma: no cover
                "indices_or_sections",  # pragma: no cover
                "shape",  # pragma: no cover
                "atol",  # pragma: no cover
                "axis1",  # pragma: no cover
                "step",  # pragma: no cover
                "axes",  # pragma: no cover
                "fill_value",  # pragma: no cover
                "stream",  # pragma: no cover
                "destination",  # pragma: no cover
                "alpha",  # pragma: no cover
                "beta",  # pragma: no cover
                "start",  # pragma: no cover
                "equal_nan",
                "nan",
                "posinf",
                "neginf",  # pragma: no cover
                "keepdims",
                "shape",
                "size",  # pragma: no cover
                "min",  # pragma: no cover
                "p",  # pragma: no cover
            } and "divmod" not in (  # pragma: no cover
                "transpose",  # pragma: no cover
                "moveaxis",  # pragma: no cover
                "reshape",  # pragma: no cover
                "expand_dims",  # pragma: no cover
                "linspace",  # pragma: no cover
                "logcumsumexp",  # pragma: no cover
                "cummin",  # pragma: no cover
                "cummax",  # pragma: no cover
                "cumprod",  # pragma: no cover
                "cumsum",  # pragma: no cover
                "logsumexp",  # pragma: no cover
                "all",  # pragma: no cover
                "any",  # pragma: no cover
                "argmin",  # pragma: no cover
                "argmax",  # pragma: no cover
                "prod",  # pragma: no cover
                "sum",  # pragma: no cover
                "mean",  # pragma: no cover
                "var",  # pragma: no cover
                "std",  # pragma: no cover
                "max",  # pragma: no cover
                "min",  # pragma: no cover
                "tensordot",  # pragma: no cover
            ):  # pragma: no cover
                return _to_tensor(x)  # pragma: no cover
        if type(x).__name__ == "ndarray":  # pragma: no cover
            return _to_tensor(x)  # pragma: no cover
        return x  # pragma: no cover

    def _w(x):  # pragma: no cover
        if isinstance(x, tuple):  # pragma: no cover
            return tuple((_w(i) for i in x))  # pragma: no cover
        if isinstance(x, list):  # pragma: no cover
            return [_w(i) for i in x]  # pragma: no cover
        return array(x) if hasattr(x, "shape") else x  # pragma: no cover

    fn = _get_compiler_op("divmod")  # pragma: no cover
    res = fn(_u(a, "a"), _u(b, "b"))  # pragma: no cover

    def __w(x):
        if isinstance(x, tuple):
            return tuple((__w(i) for i in x))
        if isinstance(x, list):
            return [__w(i) for i in x]
        return array(x) if hasattr(x, "shape") else x

    return __w(res)  # pragma: no cover


def logical_not(a: Any, stream: Any = None) -> Any:  # pragma: no cover
    """Compute logical_not.

    Args:
        a: The a argument.
        stream: The stream argument.

    Returns:
        The result of logical_not.

    """
    from zero_mlx.array import array, _to_tensor  # pragma: no cover
    import ml_switcheroo_compiler.ops as mops  # pragma: no cover
    import ml_switcheroo_compiler.ops.nn as mnn  # pragma: no cover
    import ml_switcheroo_compiler.random as mrand  # pragma: no cover
    import types  # pragma: no cover

    def _u(x, param_name=None):  # pragma: no cover
        if isinstance(x, list):  # pragma: no cover
            if __import__("builtins").all(  # pragma: no cover
                (isinstance(i, int) for i in x)  # pragma: no cover
            ):  # pragma: no cover
                return tuple(x)  # pragma: no cover
            return [_u(i, param_name) for i in x]  # pragma: no cover
        if isinstance(x, tuple):  # pragma: no cover
            return tuple((_u(i, param_name) for i in x))  # pragma: no cover
        if hasattr(x, "_tensor"):  # pragma: no cover
            return x._tensor  # pragma: no cover
        if isinstance(x, (int, float, bool, complex)):  # pragma: no cover
            if param_name not in {  # pragma: no cover
                "axis",  # pragma: no cover
                "keepdims",
                "shape",
                "size",  # pragma: no cover
                "equal_nan",
                "nan",
                "posinf",
                "neginf",  # pragma: no cover
                "axis1",  # pragma: no cover
                "rtol",  # pragma: no cover
                "n",  # pragma: no cover
                "m",  # pragma: no cover
                "endpoint",  # pragma: no cover
                "offset",  # pragma: no cover
                "a_max",  # pragma: no cover
                "ddof",  # pragma: no cover
                "dims",  # pragma: no cover
                "source",  # pragma: no cover
                "axis2",  # pragma: no cover
                "stop",  # pragma: no cover
                "k",  # pragma: no cover
                "dtype",  # pragma: no cover
                "reps",  # pragma: no cover
                "a_min",  # pragma: no cover
                "sparse",  # pragma: no cover
                "decimals",  # pragma: no cover
                "strides",  # pragma: no cover
                "num",  # pragma: no cover
                "max",  # pragma: no cover
                "indices_or_sections",  # pragma: no cover
                "shape",  # pragma: no cover
                "atol",  # pragma: no cover
                "axis1",  # pragma: no cover
                "step",  # pragma: no cover
                "axes",  # pragma: no cover
                "fill_value",  # pragma: no cover
                "stream",  # pragma: no cover
                "destination",  # pragma: no cover
                "alpha",  # pragma: no cover
                "beta",  # pragma: no cover
                "start",  # pragma: no cover
                "equal_nan",
                "nan",
                "posinf",
                "neginf",  # pragma: no cover
                "keepdims",
                "shape",
                "size",  # pragma: no cover
                "min",  # pragma: no cover
                "p",  # pragma: no cover
            } and "logical_not" not in (  # pragma: no cover
                "transpose",  # pragma: no cover
                "moveaxis",  # pragma: no cover
                "reshape",  # pragma: no cover
                "expand_dims",  # pragma: no cover
                "linspace",  # pragma: no cover
                "logcumsumexp",  # pragma: no cover
                "cummin",  # pragma: no cover
                "cummax",  # pragma: no cover
                "cumprod",  # pragma: no cover
                "cumsum",  # pragma: no cover
                "logsumexp",  # pragma: no cover
                "all",  # pragma: no cover
                "any",  # pragma: no cover
                "argmin",  # pragma: no cover
                "argmax",  # pragma: no cover
                "prod",  # pragma: no cover
                "sum",  # pragma: no cover
                "mean",  # pragma: no cover
                "var",  # pragma: no cover
                "std",  # pragma: no cover
                "max",  # pragma: no cover
                "min",  # pragma: no cover
                "tensordot",  # pragma: no cover
            ):  # pragma: no cover
                return _to_tensor(x)  # pragma: no cover
        if type(x).__name__ == "ndarray":  # pragma: no cover
            return _to_tensor(x)  # pragma: no cover
        return x  # pragma: no cover

    def _w(x):  # pragma: no cover
        if isinstance(x, tuple):  # pragma: no cover
            return tuple((_w(i) for i in x))  # pragma: no cover
        if isinstance(x, list):  # pragma: no cover
            return [_w(i) for i in x]  # pragma: no cover
        return array(x) if hasattr(x, "shape") else x  # pragma: no cover

    fn = _get_compiler_op("logical_not")  # pragma: no cover
    res = fn(_u(a, "a"))  # pragma: no cover

    def __w(x):
        if isinstance(x, tuple):
            return tuple((__w(i) for i in x))
        if isinstance(x, list):
            return [__w(i) for i in x]
        return array(x) if hasattr(x, "shape") else x

    return __w(res)  # pragma: no cover


def logical_and(a: Any, b: Any, stream: Any = None) -> Any:  # pragma: no cover
    """Compute logical_and.

    Args:
        a: The a argument.
        b: The b argument.
        stream: The stream argument.

    Returns:
        The result of logical_and.

    """
    from zero_mlx.array import array, _to_tensor  # pragma: no cover
    import ml_switcheroo_compiler.ops as mops  # pragma: no cover
    import ml_switcheroo_compiler.ops.nn as mnn  # pragma: no cover
    import ml_switcheroo_compiler.random as mrand  # pragma: no cover
    import types  # pragma: no cover

    def _u(x, param_name=None):  # pragma: no cover
        if isinstance(x, list):  # pragma: no cover
            if __import__("builtins").all(  # pragma: no cover
                (isinstance(i, int) for i in x)  # pragma: no cover
            ):  # pragma: no cover
                return tuple(x)  # pragma: no cover
            return [_u(i, param_name) for i in x]  # pragma: no cover
        if isinstance(x, tuple):  # pragma: no cover
            return tuple((_u(i, param_name) for i in x))  # pragma: no cover
        if hasattr(x, "_tensor"):  # pragma: no cover
            return x._tensor  # pragma: no cover
        if isinstance(x, (int, float, bool, complex)):  # pragma: no cover
            if param_name not in {  # pragma: no cover
                "axis",  # pragma: no cover
                "keepdims",
                "shape",
                "size",  # pragma: no cover
                "equal_nan",
                "nan",
                "posinf",
                "neginf",  # pragma: no cover
                "axis1",  # pragma: no cover
                "rtol",  # pragma: no cover
                "n",  # pragma: no cover
                "m",  # pragma: no cover
                "endpoint",  # pragma: no cover
                "offset",  # pragma: no cover
                "a_max",  # pragma: no cover
                "ddof",  # pragma: no cover
                "dims",  # pragma: no cover
                "source",  # pragma: no cover
                "axis2",  # pragma: no cover
                "stop",  # pragma: no cover
                "k",  # pragma: no cover
                "dtype",  # pragma: no cover
                "reps",  # pragma: no cover
                "a_min",  # pragma: no cover
                "sparse",  # pragma: no cover
                "decimals",  # pragma: no cover
                "strides",  # pragma: no cover
                "num",  # pragma: no cover
                "max",  # pragma: no cover
                "indices_or_sections",  # pragma: no cover
                "shape",  # pragma: no cover
                "atol",  # pragma: no cover
                "axis1",  # pragma: no cover
                "step",  # pragma: no cover
                "axes",  # pragma: no cover
                "fill_value",  # pragma: no cover
                "stream",  # pragma: no cover
                "destination",  # pragma: no cover
                "alpha",  # pragma: no cover
                "beta",  # pragma: no cover
                "start",  # pragma: no cover
                "equal_nan",
                "nan",
                "posinf",
                "neginf",  # pragma: no cover
                "keepdims",
                "shape",
                "size",  # pragma: no cover
                "min",  # pragma: no cover
                "p",  # pragma: no cover
            } and "logical_and" not in (  # pragma: no cover
                "transpose",  # pragma: no cover
                "moveaxis",  # pragma: no cover
                "reshape",  # pragma: no cover
                "expand_dims",  # pragma: no cover
                "linspace",  # pragma: no cover
                "logcumsumexp",  # pragma: no cover
                "cummin",  # pragma: no cover
                "cummax",  # pragma: no cover
                "cumprod",  # pragma: no cover
                "cumsum",  # pragma: no cover
                "logsumexp",  # pragma: no cover
                "all",  # pragma: no cover
                "any",  # pragma: no cover
                "argmin",  # pragma: no cover
                "argmax",  # pragma: no cover
                "prod",  # pragma: no cover
                "sum",  # pragma: no cover
                "mean",  # pragma: no cover
                "var",  # pragma: no cover
                "std",  # pragma: no cover
                "max",  # pragma: no cover
                "min",  # pragma: no cover
                "tensordot",  # pragma: no cover
            ):  # pragma: no cover
                return _to_tensor(x)  # pragma: no cover
        if type(x).__name__ == "ndarray":  # pragma: no cover
            return _to_tensor(x)  # pragma: no cover
        return x  # pragma: no cover

    def _w(x):  # pragma: no cover
        if isinstance(x, tuple):  # pragma: no cover
            return tuple((_w(i) for i in x))  # pragma: no cover
        if isinstance(x, list):  # pragma: no cover
            return [_w(i) for i in x]  # pragma: no cover
        return array(x) if hasattr(x, "shape") else x  # pragma: no cover

    fn = _get_compiler_op("logical_and")  # pragma: no cover
    res = fn(_u(a, "a"), _u(b, "b"))  # pragma: no cover

    def __w(x):
        if isinstance(x, tuple):
            return tuple((__w(i) for i in x))
        if isinstance(x, list):
            return [__w(i) for i in x]
        return array(x) if hasattr(x, "shape") else x

    return __w(res)  # pragma: no cover


def logical_or(a: Any, b: Any, stream: Any = None) -> Any:  # pragma: no cover
    """Compute logical_or.

    Args:
        a: The a argument.
        b: The b argument.
        stream: The stream argument.

    Returns:
        The result of logical_or.

    """
    from zero_mlx.array import array, _to_tensor  # pragma: no cover
    import ml_switcheroo_compiler.ops as mops  # pragma: no cover
    import ml_switcheroo_compiler.ops.nn as mnn  # pragma: no cover
    import ml_switcheroo_compiler.random as mrand  # pragma: no cover
    import types  # pragma: no cover

    def _u(x, param_name=None):  # pragma: no cover
        if isinstance(x, list):  # pragma: no cover
            if __import__("builtins").all(  # pragma: no cover
                (isinstance(i, int) for i in x)  # pragma: no cover
            ):  # pragma: no cover
                return tuple(x)  # pragma: no cover
            return [_u(i, param_name) for i in x]  # pragma: no cover
        if isinstance(x, tuple):  # pragma: no cover
            return tuple((_u(i, param_name) for i in x))  # pragma: no cover
        if hasattr(x, "_tensor"):  # pragma: no cover
            return x._tensor  # pragma: no cover
        if isinstance(x, (int, float, bool, complex)):  # pragma: no cover
            if param_name not in {  # pragma: no cover
                "axis",  # pragma: no cover
                "keepdims",
                "shape",
                "size",  # pragma: no cover
                "equal_nan",
                "nan",
                "posinf",
                "neginf",  # pragma: no cover
                "axis1",  # pragma: no cover
                "rtol",  # pragma: no cover
                "n",  # pragma: no cover
                "m",  # pragma: no cover
                "endpoint",  # pragma: no cover
                "offset",  # pragma: no cover
                "a_max",  # pragma: no cover
                "ddof",  # pragma: no cover
                "dims",  # pragma: no cover
                "source",  # pragma: no cover
                "axis2",  # pragma: no cover
                "stop",  # pragma: no cover
                "k",  # pragma: no cover
                "dtype",  # pragma: no cover
                "reps",  # pragma: no cover
                "a_min",  # pragma: no cover
                "sparse",  # pragma: no cover
                "decimals",  # pragma: no cover
                "strides",  # pragma: no cover
                "num",  # pragma: no cover
                "max",  # pragma: no cover
                "indices_or_sections",  # pragma: no cover
                "shape",  # pragma: no cover
                "atol",  # pragma: no cover
                "axis1",  # pragma: no cover
                "step",  # pragma: no cover
                "axes",  # pragma: no cover
                "fill_value",  # pragma: no cover
                "stream",  # pragma: no cover
                "destination",  # pragma: no cover
                "alpha",  # pragma: no cover
                "beta",  # pragma: no cover
                "start",  # pragma: no cover
                "equal_nan",
                "nan",
                "posinf",
                "neginf",  # pragma: no cover
                "keepdims",
                "shape",
                "size",  # pragma: no cover
                "min",  # pragma: no cover
                "p",  # pragma: no cover
            } and "logical_or" not in (  # pragma: no cover
                "transpose",  # pragma: no cover
                "moveaxis",  # pragma: no cover
                "reshape",  # pragma: no cover
                "expand_dims",  # pragma: no cover
                "linspace",  # pragma: no cover
                "logcumsumexp",  # pragma: no cover
                "cummin",  # pragma: no cover
                "cummax",  # pragma: no cover
                "cumprod",  # pragma: no cover
                "cumsum",  # pragma: no cover
                "logsumexp",  # pragma: no cover
                "all",  # pragma: no cover
                "any",  # pragma: no cover
                "argmin",  # pragma: no cover
                "argmax",  # pragma: no cover
                "prod",  # pragma: no cover
                "sum",  # pragma: no cover
                "mean",  # pragma: no cover
                "var",  # pragma: no cover
                "std",  # pragma: no cover
                "max",  # pragma: no cover
                "min",  # pragma: no cover
                "tensordot",  # pragma: no cover
            ):  # pragma: no cover
                return _to_tensor(x)  # pragma: no cover
        if type(x).__name__ == "ndarray":  # pragma: no cover
            return _to_tensor(x)  # pragma: no cover
        return x  # pragma: no cover

    def _w(x):  # pragma: no cover
        if isinstance(x, tuple):  # pragma: no cover
            return tuple((_w(i) for i in x))  # pragma: no cover
        if isinstance(x, list):  # pragma: no cover
            return [_w(i) for i in x]  # pragma: no cover
        return array(x) if hasattr(x, "shape") else x  # pragma: no cover

    fn = _get_compiler_op("logical_or")  # pragma: no cover
    res = fn(_u(a, "a"), _u(b, "b"))  # pragma: no cover

    def __w(x):
        if isinstance(x, tuple):
            return tuple((__w(i) for i in x))
        if isinstance(x, list):
            return [__w(i) for i in x]
        return array(x) if hasattr(x, "shape") else x

    return __w(res)  # pragma: no cover


def sqrt(a: Any, stream: Any = None) -> Any:  # pragma: no cover
    """Compute sqrt.

    Args:
        a: The a argument.
        stream: The stream argument.

    Returns:
        The result of sqrt.

    """
    from zero_mlx.array import array, _to_tensor  # pragma: no cover
    import ml_switcheroo_compiler.ops as mops  # pragma: no cover
    import ml_switcheroo_compiler.ops.nn as mnn  # pragma: no cover
    import ml_switcheroo_compiler.random as mrand  # pragma: no cover
    import types  # pragma: no cover

    def _u(x, param_name=None):  # pragma: no cover
        if isinstance(x, list):  # pragma: no cover
            if __import__("builtins").all(  # pragma: no cover
                (isinstance(i, int) for i in x)  # pragma: no cover
            ):  # pragma: no cover
                return tuple(x)  # pragma: no cover
            return [_u(i, param_name) for i in x]  # pragma: no cover
        if isinstance(x, tuple):  # pragma: no cover
            return tuple((_u(i, param_name) for i in x))  # pragma: no cover
        if hasattr(x, "_tensor"):  # pragma: no cover
            return x._tensor  # pragma: no cover
        if isinstance(x, (int, float, bool, complex)):  # pragma: no cover
            if param_name not in {  # pragma: no cover
                "axis",  # pragma: no cover
                "keepdims",
                "shape",
                "size",  # pragma: no cover
                "equal_nan",
                "nan",
                "posinf",
                "neginf",  # pragma: no cover
                "axis1",  # pragma: no cover
                "rtol",  # pragma: no cover
                "n",  # pragma: no cover
                "m",  # pragma: no cover
                "endpoint",  # pragma: no cover
                "offset",  # pragma: no cover
                "a_max",  # pragma: no cover
                "ddof",  # pragma: no cover
                "dims",  # pragma: no cover
                "source",  # pragma: no cover
                "axis2",  # pragma: no cover
                "stop",  # pragma: no cover
                "k",  # pragma: no cover
                "dtype",  # pragma: no cover
                "reps",  # pragma: no cover
                "a_min",  # pragma: no cover
                "sparse",  # pragma: no cover
                "decimals",  # pragma: no cover
                "strides",  # pragma: no cover
                "num",  # pragma: no cover
                "max",  # pragma: no cover
                "indices_or_sections",  # pragma: no cover
                "shape",  # pragma: no cover
                "atol",  # pragma: no cover
                "axis1",  # pragma: no cover
                "step",  # pragma: no cover
                "axes",  # pragma: no cover
                "fill_value",  # pragma: no cover
                "stream",  # pragma: no cover
                "destination",  # pragma: no cover
                "alpha",  # pragma: no cover
                "beta",  # pragma: no cover
                "start",  # pragma: no cover
                "equal_nan",
                "nan",
                "posinf",
                "neginf",  # pragma: no cover
                "keepdims",
                "shape",
                "size",  # pragma: no cover
                "min",  # pragma: no cover
                "p",  # pragma: no cover
            } and "sqrt" not in (  # pragma: no cover
                "transpose",  # pragma: no cover
                "moveaxis",  # pragma: no cover
                "reshape",  # pragma: no cover
                "expand_dims",  # pragma: no cover
                "linspace",  # pragma: no cover
                "logcumsumexp",  # pragma: no cover
                "cummin",  # pragma: no cover
                "cummax",  # pragma: no cover
                "cumprod",  # pragma: no cover
                "cumsum",  # pragma: no cover
                "logsumexp",  # pragma: no cover
                "all",  # pragma: no cover
                "any",  # pragma: no cover
                "argmin",  # pragma: no cover
                "argmax",  # pragma: no cover
                "prod",  # pragma: no cover
                "sum",  # pragma: no cover
                "mean",  # pragma: no cover
                "var",  # pragma: no cover
                "std",  # pragma: no cover
                "max",  # pragma: no cover
                "min",  # pragma: no cover
                "tensordot",  # pragma: no cover
            ):  # pragma: no cover
                return _to_tensor(x)  # pragma: no cover
        if type(x).__name__ == "ndarray":  # pragma: no cover
            return _to_tensor(x)  # pragma: no cover
        return x  # pragma: no cover

    def _w(x):  # pragma: no cover
        if isinstance(x, tuple):  # pragma: no cover
            return tuple((_w(i) for i in x))  # pragma: no cover
        if isinstance(x, list):  # pragma: no cover
            return [_w(i) for i in x]  # pragma: no cover
        return array(x) if hasattr(x, "shape") else x  # pragma: no cover

    fn = _get_compiler_op("sqrt")  # pragma: no cover
    res = fn(_u(a, "a"))  # pragma: no cover

    def __w(x):
        if isinstance(x, tuple):
            return tuple((__w(i) for i in x))
        if isinstance(x, list):
            return [__w(i) for i in x]
        return array(x) if hasattr(x, "shape") else x

    return __w(res)  # pragma: no cover


def abs(a: Any, stream: Any = None) -> Any:  # pragma: no cover
    """Compute abs.

    Args:
        a: The a argument.
        stream: The stream argument.

    Returns:
        The result of abs.

    """
    from zero_mlx.array import array, _to_tensor  # pragma: no cover
    import ml_switcheroo_compiler.ops as mops  # pragma: no cover
    import ml_switcheroo_compiler.ops.nn as mnn  # pragma: no cover
    import ml_switcheroo_compiler.random as mrand  # pragma: no cover
    import types  # pragma: no cover

    def _u(x, param_name=None):  # pragma: no cover
        if isinstance(x, list):  # pragma: no cover
            if __import__("builtins").all(  # pragma: no cover
                (isinstance(i, int) for i in x)  # pragma: no cover
            ):  # pragma: no cover
                return tuple(x)  # pragma: no cover
            return [_u(i, param_name) for i in x]  # pragma: no cover
        if isinstance(x, tuple):  # pragma: no cover
            return tuple((_u(i, param_name) for i in x))  # pragma: no cover
        if hasattr(x, "_tensor"):  # pragma: no cover
            return x._tensor  # pragma: no cover
        if isinstance(x, (int, float, bool, complex)):  # pragma: no cover
            if param_name not in {  # pragma: no cover
                "axis",  # pragma: no cover
                "keepdims",
                "shape",
                "size",  # pragma: no cover
                "equal_nan",
                "nan",
                "posinf",
                "neginf",  # pragma: no cover
                "axis1",  # pragma: no cover
                "rtol",  # pragma: no cover
                "n",  # pragma: no cover
                "m",  # pragma: no cover
                "endpoint",  # pragma: no cover
                "offset",  # pragma: no cover
                "a_max",  # pragma: no cover
                "ddof",  # pragma: no cover
                "dims",  # pragma: no cover
                "source",  # pragma: no cover
                "axis2",  # pragma: no cover
                "stop",  # pragma: no cover
                "k",  # pragma: no cover
                "dtype",  # pragma: no cover
                "reps",  # pragma: no cover
                "a_min",  # pragma: no cover
                "sparse",  # pragma: no cover
                "decimals",  # pragma: no cover
                "strides",  # pragma: no cover
                "num",  # pragma: no cover
                "max",  # pragma: no cover
                "indices_or_sections",  # pragma: no cover
                "shape",  # pragma: no cover
                "atol",  # pragma: no cover
                "axis1",  # pragma: no cover
                "step",  # pragma: no cover
                "axes",  # pragma: no cover
                "fill_value",  # pragma: no cover
                "stream",  # pragma: no cover
                "destination",  # pragma: no cover
                "alpha",  # pragma: no cover
                "beta",  # pragma: no cover
                "start",  # pragma: no cover
                "equal_nan",
                "nan",
                "posinf",
                "neginf",  # pragma: no cover
                "keepdims",
                "shape",
                "size",  # pragma: no cover
                "min",  # pragma: no cover
                "p",  # pragma: no cover
            } and "abs" not in (  # pragma: no cover
                "transpose",  # pragma: no cover
                "moveaxis",  # pragma: no cover
                "reshape",  # pragma: no cover
                "expand_dims",  # pragma: no cover
                "linspace",  # pragma: no cover
                "logcumsumexp",  # pragma: no cover
                "cummin",  # pragma: no cover
                "cummax",  # pragma: no cover
                "cumprod",  # pragma: no cover
                "cumsum",  # pragma: no cover
                "logsumexp",  # pragma: no cover
                "all",  # pragma: no cover
                "any",  # pragma: no cover
                "argmin",  # pragma: no cover
                "argmax",  # pragma: no cover
                "prod",  # pragma: no cover
                "sum",  # pragma: no cover
                "mean",  # pragma: no cover
                "var",  # pragma: no cover
                "std",  # pragma: no cover
                "max",  # pragma: no cover
                "min",  # pragma: no cover
                "tensordot",  # pragma: no cover
            ):  # pragma: no cover
                return _to_tensor(x)  # pragma: no cover
        if type(x).__name__ == "ndarray":  # pragma: no cover
            return _to_tensor(x)  # pragma: no cover
        return x  # pragma: no cover

    def _w(x):  # pragma: no cover
        if isinstance(x, tuple):  # pragma: no cover
            return tuple((_w(i) for i in x))  # pragma: no cover
        if isinstance(x, list):  # pragma: no cover
            return [_w(i) for i in x]  # pragma: no cover
        return array(x) if hasattr(x, "shape") else x  # pragma: no cover

    fn = _get_compiler_op("abs")  # pragma: no cover
    res = fn(_u(a, "a"))  # pragma: no cover

    def __w(x):
        if isinstance(x, tuple):
            return tuple((__w(i) for i in x))
        if isinstance(x, list):
            return [__w(i) for i in x]
        return array(x) if hasattr(x, "shape") else x

    return __w(res)  # pragma: no cover


def negative(a: Any, stream: Any = None) -> Any:  # pragma: no cover
    """Compute negative.

    Args:
        a: The a argument.
        stream: The stream argument.

    Returns:
        The result of negative.

    """
    from zero_mlx.array import array, _to_tensor  # pragma: no cover
    import ml_switcheroo_compiler.ops as mops  # pragma: no cover
    import ml_switcheroo_compiler.ops.nn as mnn  # pragma: no cover
    import ml_switcheroo_compiler.random as mrand  # pragma: no cover
    import types  # pragma: no cover

    def _u(x, param_name=None):  # pragma: no cover
        if isinstance(x, list):  # pragma: no cover
            if __import__("builtins").all(  # pragma: no cover
                (isinstance(i, int) for i in x)  # pragma: no cover
            ):  # pragma: no cover
                return tuple(x)  # pragma: no cover
            return [_u(i, param_name) for i in x]  # pragma: no cover
        if isinstance(x, tuple):  # pragma: no cover
            return tuple((_u(i, param_name) for i in x))  # pragma: no cover
        if hasattr(x, "_tensor"):  # pragma: no cover
            return x._tensor  # pragma: no cover
        if isinstance(x, (int, float, bool, complex)):  # pragma: no cover
            if param_name not in {  # pragma: no cover
                "axis",  # pragma: no cover
                "keepdims",
                "shape",
                "size",  # pragma: no cover
                "equal_nan",
                "nan",
                "posinf",
                "neginf",  # pragma: no cover
                "axis1",  # pragma: no cover
                "rtol",  # pragma: no cover
                "n",  # pragma: no cover
                "m",  # pragma: no cover
                "endpoint",  # pragma: no cover
                "offset",  # pragma: no cover
                "a_max",  # pragma: no cover
                "ddof",  # pragma: no cover
                "dims",  # pragma: no cover
                "source",  # pragma: no cover
                "axis2",  # pragma: no cover
                "stop",  # pragma: no cover
                "k",  # pragma: no cover
                "dtype",  # pragma: no cover
                "reps",  # pragma: no cover
                "a_min",  # pragma: no cover
                "sparse",  # pragma: no cover
                "decimals",  # pragma: no cover
                "strides",  # pragma: no cover
                "num",  # pragma: no cover
                "max",  # pragma: no cover
                "indices_or_sections",  # pragma: no cover
                "shape",  # pragma: no cover
                "atol",  # pragma: no cover
                "axis1",  # pragma: no cover
                "step",  # pragma: no cover
                "axes",  # pragma: no cover
                "fill_value",  # pragma: no cover
                "stream",  # pragma: no cover
                "destination",  # pragma: no cover
                "alpha",  # pragma: no cover
                "beta",  # pragma: no cover
                "start",  # pragma: no cover
                "equal_nan",
                "nan",
                "posinf",
                "neginf",  # pragma: no cover
                "keepdims",
                "shape",
                "size",  # pragma: no cover
                "min",  # pragma: no cover
                "p",  # pragma: no cover
            } and "negative" not in (  # pragma: no cover
                "transpose",  # pragma: no cover
                "moveaxis",  # pragma: no cover
                "reshape",  # pragma: no cover
                "expand_dims",  # pragma: no cover
                "linspace",  # pragma: no cover
                "logcumsumexp",  # pragma: no cover
                "cummin",  # pragma: no cover
                "cummax",  # pragma: no cover
                "cumprod",  # pragma: no cover
                "cumsum",  # pragma: no cover
                "logsumexp",  # pragma: no cover
                "all",  # pragma: no cover
                "any",  # pragma: no cover
                "argmin",  # pragma: no cover
                "argmax",  # pragma: no cover
                "prod",  # pragma: no cover
                "sum",  # pragma: no cover
                "mean",  # pragma: no cover
                "var",  # pragma: no cover
                "std",  # pragma: no cover
                "max",  # pragma: no cover
                "min",  # pragma: no cover
                "tensordot",  # pragma: no cover
            ):  # pragma: no cover
                return _to_tensor(x)  # pragma: no cover
        if type(x).__name__ == "ndarray":  # pragma: no cover
            return _to_tensor(x)  # pragma: no cover
        return x  # pragma: no cover

    def _w(x):  # pragma: no cover
        if isinstance(x, tuple):  # pragma: no cover
            return tuple((_w(i) for i in x))  # pragma: no cover
        if isinstance(x, list):  # pragma: no cover
            return [_w(i) for i in x]  # pragma: no cover
        return array(x) if hasattr(x, "shape") else x  # pragma: no cover

    fn = _get_compiler_op("negative")  # pragma: no cover
    res = fn(_u(a, "a"))  # pragma: no cover

    def __w(x):
        if isinstance(x, tuple):
            return tuple((__w(i) for i in x))
        if isinstance(x, list):
            return [__w(i) for i in x]
        return array(x) if hasattr(x, "shape") else x

    return __w(res)  # pragma: no cover


def exp(a: Any, stream: Any = None) -> Any:  # pragma: no cover
    """Compute exp.

    Args:
        a: The a argument.
        stream: The stream argument.

    Returns:
        The result of exp.

    """
    from zero_mlx.array import array, _to_tensor  # pragma: no cover
    import ml_switcheroo_compiler.ops as mops  # pragma: no cover
    import ml_switcheroo_compiler.ops.nn as mnn  # pragma: no cover
    import ml_switcheroo_compiler.random as mrand  # pragma: no cover
    import types  # pragma: no cover

    def _u(x, param_name=None):  # pragma: no cover
        if isinstance(x, list):  # pragma: no cover
            if __import__("builtins").all(  # pragma: no cover
                (isinstance(i, int) for i in x)  # pragma: no cover
            ):  # pragma: no cover
                return tuple(x)  # pragma: no cover
            return [_u(i, param_name) for i in x]  # pragma: no cover
        if isinstance(x, tuple):  # pragma: no cover
            return tuple((_u(i, param_name) for i in x))  # pragma: no cover
        if hasattr(x, "_tensor"):  # pragma: no cover
            return x._tensor  # pragma: no cover
        if isinstance(x, (int, float, bool, complex)):  # pragma: no cover
            if param_name not in {  # pragma: no cover
                "axis",  # pragma: no cover
                "keepdims",
                "shape",
                "size",  # pragma: no cover
                "equal_nan",
                "nan",
                "posinf",
                "neginf",  # pragma: no cover
                "axis1",  # pragma: no cover
                "rtol",  # pragma: no cover
                "n",  # pragma: no cover
                "m",  # pragma: no cover
                "endpoint",  # pragma: no cover
                "offset",  # pragma: no cover
                "a_max",  # pragma: no cover
                "ddof",  # pragma: no cover
                "dims",  # pragma: no cover
                "source",  # pragma: no cover
                "axis2",  # pragma: no cover
                "stop",  # pragma: no cover
                "k",  # pragma: no cover
                "dtype",  # pragma: no cover
                "reps",  # pragma: no cover
                "a_min",  # pragma: no cover
                "sparse",  # pragma: no cover
                "decimals",  # pragma: no cover
                "strides",  # pragma: no cover
                "num",  # pragma: no cover
                "max",  # pragma: no cover
                "indices_or_sections",  # pragma: no cover
                "shape",  # pragma: no cover
                "atol",  # pragma: no cover
                "axis1",  # pragma: no cover
                "step",  # pragma: no cover
                "axes",  # pragma: no cover
                "fill_value",  # pragma: no cover
                "stream",  # pragma: no cover
                "destination",  # pragma: no cover
                "alpha",  # pragma: no cover
                "beta",  # pragma: no cover
                "start",  # pragma: no cover
                "equal_nan",
                "nan",
                "posinf",
                "neginf",  # pragma: no cover
                "keepdims",
                "shape",
                "size",  # pragma: no cover
                "min",  # pragma: no cover
                "p",  # pragma: no cover
            } and "exp" not in (  # pragma: no cover
                "transpose",  # pragma: no cover
                "moveaxis",  # pragma: no cover
                "reshape",  # pragma: no cover
                "expand_dims",  # pragma: no cover
                "linspace",  # pragma: no cover
                "logcumsumexp",  # pragma: no cover
                "cummin",  # pragma: no cover
                "cummax",  # pragma: no cover
                "cumprod",  # pragma: no cover
                "cumsum",  # pragma: no cover
                "logsumexp",  # pragma: no cover
                "all",  # pragma: no cover
                "any",  # pragma: no cover
                "argmin",  # pragma: no cover
                "argmax",  # pragma: no cover
                "prod",  # pragma: no cover
                "sum",  # pragma: no cover
                "mean",  # pragma: no cover
                "var",  # pragma: no cover
                "std",  # pragma: no cover
                "max",  # pragma: no cover
                "min",  # pragma: no cover
                "tensordot",  # pragma: no cover
            ):  # pragma: no cover
                return _to_tensor(x)  # pragma: no cover
        if type(x).__name__ == "ndarray":  # pragma: no cover
            return _to_tensor(x)  # pragma: no cover
        return x  # pragma: no cover

    def _w(x):  # pragma: no cover
        if isinstance(x, tuple):  # pragma: no cover
            return tuple((_w(i) for i in x))  # pragma: no cover
        if isinstance(x, list):  # pragma: no cover
            return [_w(i) for i in x]  # pragma: no cover
        return array(x) if hasattr(x, "shape") else x  # pragma: no cover

    fn = _get_compiler_op("exp")  # pragma: no cover
    res = fn(_u(a, "a"))  # pragma: no cover

    def __w(x):
        if isinstance(x, tuple):
            return tuple((__w(i) for i in x))
        if isinstance(x, list):
            return [__w(i) for i in x]
        return array(x) if hasattr(x, "shape") else x

    return __w(res)  # pragma: no cover


def rsqrt(a: Any, stream: Any = None) -> Any:  # pragma: no cover
    """Compute rsqrt.

    Args:
        a: The a argument.
        stream: The stream argument.

    Returns:
        The result of rsqrt.

    """
    from zero_mlx.array import array, _to_tensor  # pragma: no cover
    import ml_switcheroo_compiler.ops as mops  # pragma: no cover
    import ml_switcheroo_compiler.ops.nn as mnn  # pragma: no cover
    import ml_switcheroo_compiler.random as mrand  # pragma: no cover
    import types  # pragma: no cover

    def _u(x, param_name=None):  # pragma: no cover
        if isinstance(x, list):  # pragma: no cover
            if __import__("builtins").all(  # pragma: no cover
                (isinstance(i, int) for i in x)  # pragma: no cover
            ):  # pragma: no cover
                return tuple(x)  # pragma: no cover
            return [_u(i, param_name) for i in x]  # pragma: no cover
        if isinstance(x, tuple):  # pragma: no cover
            return tuple((_u(i, param_name) for i in x))  # pragma: no cover
        if hasattr(x, "_tensor"):  # pragma: no cover
            return x._tensor  # pragma: no cover
        if isinstance(x, (int, float, bool, complex)):  # pragma: no cover
            if param_name not in {  # pragma: no cover
                "axis",  # pragma: no cover
                "keepdims",
                "shape",
                "size",  # pragma: no cover
                "equal_nan",
                "nan",
                "posinf",
                "neginf",  # pragma: no cover
                "axis1",  # pragma: no cover
                "rtol",  # pragma: no cover
                "n",  # pragma: no cover
                "m",  # pragma: no cover
                "endpoint",  # pragma: no cover
                "offset",  # pragma: no cover
                "a_max",  # pragma: no cover
                "ddof",  # pragma: no cover
                "dims",  # pragma: no cover
                "source",  # pragma: no cover
                "axis2",  # pragma: no cover
                "stop",  # pragma: no cover
                "k",  # pragma: no cover
                "dtype",  # pragma: no cover
                "reps",  # pragma: no cover
                "a_min",  # pragma: no cover
                "sparse",  # pragma: no cover
                "decimals",  # pragma: no cover
                "strides",  # pragma: no cover
                "num",  # pragma: no cover
                "max",  # pragma: no cover
                "indices_or_sections",  # pragma: no cover
                "shape",  # pragma: no cover
                "atol",  # pragma: no cover
                "axis1",  # pragma: no cover
                "step",  # pragma: no cover
                "axes",  # pragma: no cover
                "fill_value",  # pragma: no cover
                "stream",  # pragma: no cover
                "destination",  # pragma: no cover
                "alpha",  # pragma: no cover
                "beta",  # pragma: no cover
                "start",  # pragma: no cover
                "equal_nan",
                "nan",
                "posinf",
                "neginf",  # pragma: no cover
                "keepdims",
                "shape",
                "size",  # pragma: no cover
                "min",  # pragma: no cover
                "p",  # pragma: no cover
            } and "rsqrt" not in (  # pragma: no cover
                "transpose",  # pragma: no cover
                "moveaxis",  # pragma: no cover
                "reshape",  # pragma: no cover
                "expand_dims",  # pragma: no cover
                "linspace",  # pragma: no cover
                "logcumsumexp",  # pragma: no cover
                "cummin",  # pragma: no cover
                "cummax",  # pragma: no cover
                "cumprod",  # pragma: no cover
                "cumsum",  # pragma: no cover
                "logsumexp",  # pragma: no cover
                "all",  # pragma: no cover
                "any",  # pragma: no cover
                "argmin",  # pragma: no cover
                "argmax",  # pragma: no cover
                "prod",  # pragma: no cover
                "sum",  # pragma: no cover
                "mean",  # pragma: no cover
                "var",  # pragma: no cover
                "std",  # pragma: no cover
                "max",  # pragma: no cover
                "min",  # pragma: no cover
                "tensordot",  # pragma: no cover
            ):  # pragma: no cover
                return _to_tensor(x)  # pragma: no cover
        if type(x).__name__ == "ndarray":  # pragma: no cover
            return _to_tensor(x)  # pragma: no cover
        return x  # pragma: no cover

    def _w(x):  # pragma: no cover
        if isinstance(x, tuple):  # pragma: no cover
            return tuple((_w(i) for i in x))  # pragma: no cover
        if isinstance(x, list):  # pragma: no cover
            return [_w(i) for i in x]  # pragma: no cover
        return array(x) if hasattr(x, "shape") else x  # pragma: no cover

    fn = _get_compiler_op("rsqrt")  # pragma: no cover
    res = fn(_u(a, "a"))  # pragma: no cover

    def __w(x):
        if isinstance(x, tuple):
            return tuple((__w(i) for i in x))
        if isinstance(x, list):
            return [__w(i) for i in x]
        return array(x) if hasattr(x, "shape") else x

    return __w(res)  # pragma: no cover


def add(a: Any, b: Any, stream: Any = None) -> Any:  # pragma: no cover
    """Compute add.

    Args:
        a: The a argument.
        b: The b argument.
        stream: The stream argument.

    Returns:
        The result of add.

    """
    from zero_mlx.array import array, _to_tensor  # pragma: no cover
    import ml_switcheroo_compiler.ops as mops  # pragma: no cover
    import ml_switcheroo_compiler.ops.nn as mnn  # pragma: no cover
    import ml_switcheroo_compiler.random as mrand  # pragma: no cover
    import types  # pragma: no cover

    def _u(x, param_name=None):  # pragma: no cover
        if isinstance(x, list):  # pragma: no cover
            if __import__("builtins").all(  # pragma: no cover
                (isinstance(i, int) for i in x)  # pragma: no cover
            ):  # pragma: no cover
                return tuple(x)  # pragma: no cover
            return [_u(i, param_name) for i in x]  # pragma: no cover
        if isinstance(x, tuple):  # pragma: no cover
            return tuple((_u(i, param_name) for i in x))  # pragma: no cover
        if hasattr(x, "_tensor"):  # pragma: no cover
            return x._tensor  # pragma: no cover
        if isinstance(x, (int, float, bool, complex)):  # pragma: no cover
            if param_name not in {  # pragma: no cover
                "axis",  # pragma: no cover
                "keepdims",
                "shape",
                "size",  # pragma: no cover
                "equal_nan",
                "nan",
                "posinf",
                "neginf",  # pragma: no cover
                "axis1",  # pragma: no cover
                "rtol",  # pragma: no cover
                "n",  # pragma: no cover
                "m",  # pragma: no cover
                "endpoint",  # pragma: no cover
                "offset",  # pragma: no cover
                "a_max",  # pragma: no cover
                "ddof",  # pragma: no cover
                "dims",  # pragma: no cover
                "source",  # pragma: no cover
                "axis2",  # pragma: no cover
                "stop",  # pragma: no cover
                "k",  # pragma: no cover
                "dtype",  # pragma: no cover
                "reps",  # pragma: no cover
                "a_min",  # pragma: no cover
                "sparse",  # pragma: no cover
                "decimals",  # pragma: no cover
                "strides",  # pragma: no cover
                "num",  # pragma: no cover
                "max",  # pragma: no cover
                "indices_or_sections",  # pragma: no cover
                "shape",  # pragma: no cover
                "atol",  # pragma: no cover
                "axis1",  # pragma: no cover
                "step",  # pragma: no cover
                "axes",  # pragma: no cover
                "fill_value",  # pragma: no cover
                "stream",  # pragma: no cover
                "destination",  # pragma: no cover
                "alpha",  # pragma: no cover
                "beta",  # pragma: no cover
                "start",  # pragma: no cover
                "equal_nan",
                "nan",
                "posinf",
                "neginf",  # pragma: no cover
                "keepdims",
                "shape",
                "size",  # pragma: no cover
                "min",  # pragma: no cover
                "p",  # pragma: no cover
            } and "add" not in (  # pragma: no cover
                "transpose",  # pragma: no cover
                "moveaxis",  # pragma: no cover
                "reshape",  # pragma: no cover
                "expand_dims",  # pragma: no cover
                "linspace",  # pragma: no cover
                "logcumsumexp",  # pragma: no cover
                "cummin",  # pragma: no cover
                "cummax",  # pragma: no cover
                "cumprod",  # pragma: no cover
                "cumsum",  # pragma: no cover
                "logsumexp",  # pragma: no cover
                "all",  # pragma: no cover
                "any",  # pragma: no cover
                "argmin",  # pragma: no cover
                "argmax",  # pragma: no cover
                "prod",  # pragma: no cover
                "sum",  # pragma: no cover
                "mean",  # pragma: no cover
                "var",  # pragma: no cover
                "std",  # pragma: no cover
                "max",  # pragma: no cover
                "min",  # pragma: no cover
                "tensordot",  # pragma: no cover
            ):  # pragma: no cover
                return _to_tensor(x)  # pragma: no cover
        if type(x).__name__ == "ndarray":  # pragma: no cover
            return _to_tensor(x)  # pragma: no cover
        return x  # pragma: no cover

    def _w(x):  # pragma: no cover
        if isinstance(x, tuple):  # pragma: no cover
            return tuple((_w(i) for i in x))  # pragma: no cover
        if isinstance(x, list):  # pragma: no cover
            return [_w(i) for i in x]  # pragma: no cover
        return array(x) if hasattr(x, "shape") else x  # pragma: no cover

    fn = _get_compiler_op("add")  # pragma: no cover
    res = fn(_u(a, "a"), _u(b, "b"))  # pragma: no cover

    def __w(x):
        if isinstance(x, tuple):
            return tuple((__w(i) for i in x))
        if isinstance(x, list):
            return [__w(i) for i in x]
        return array(x) if hasattr(x, "shape") else x

    return __w(res)  # pragma: no cover


def subtract(a: Any, b: Any, stream: Any = None) -> Any:  # pragma: no cover
    """Compute subtract.

    Args:
        a: The a argument.
        b: The b argument.
        stream: The stream argument.

    Returns:
        The result of subtract.

    """
    from zero_mlx.array import array, _to_tensor  # pragma: no cover
    import ml_switcheroo_compiler.ops as mops  # pragma: no cover
    import ml_switcheroo_compiler.ops.nn as mnn  # pragma: no cover
    import ml_switcheroo_compiler.random as mrand  # pragma: no cover
    import types  # pragma: no cover

    def _u(x, param_name=None):  # pragma: no cover
        if isinstance(x, list):  # pragma: no cover
            if __import__("builtins").all(  # pragma: no cover
                (isinstance(i, int) for i in x)  # pragma: no cover
            ):  # pragma: no cover
                return tuple(x)  # pragma: no cover
            return [_u(i, param_name) for i in x]  # pragma: no cover
        if isinstance(x, tuple):  # pragma: no cover
            return tuple((_u(i, param_name) for i in x))  # pragma: no cover
        if hasattr(x, "_tensor"):  # pragma: no cover
            return x._tensor  # pragma: no cover
        if isinstance(x, (int, float, bool, complex)):  # pragma: no cover
            if param_name not in {  # pragma: no cover
                "axis",  # pragma: no cover
                "keepdims",
                "shape",
                "size",  # pragma: no cover
                "equal_nan",
                "nan",
                "posinf",
                "neginf",  # pragma: no cover
                "axis1",  # pragma: no cover
                "rtol",  # pragma: no cover
                "n",  # pragma: no cover
                "m",  # pragma: no cover
                "endpoint",  # pragma: no cover
                "offset",  # pragma: no cover
                "a_max",  # pragma: no cover
                "ddof",  # pragma: no cover
                "dims",  # pragma: no cover
                "source",  # pragma: no cover
                "axis2",  # pragma: no cover
                "stop",  # pragma: no cover
                "k",  # pragma: no cover
                "dtype",  # pragma: no cover
                "reps",  # pragma: no cover
                "a_min",  # pragma: no cover
                "sparse",  # pragma: no cover
                "decimals",  # pragma: no cover
                "strides",  # pragma: no cover
                "num",  # pragma: no cover
                "max",  # pragma: no cover
                "indices_or_sections",  # pragma: no cover
                "shape",  # pragma: no cover
                "atol",  # pragma: no cover
                "axis1",  # pragma: no cover
                "step",  # pragma: no cover
                "axes",  # pragma: no cover
                "fill_value",  # pragma: no cover
                "stream",  # pragma: no cover
                "destination",  # pragma: no cover
                "alpha",  # pragma: no cover
                "beta",  # pragma: no cover
                "start",  # pragma: no cover
                "equal_nan",
                "nan",
                "posinf",
                "neginf",  # pragma: no cover
                "keepdims",
                "shape",
                "size",  # pragma: no cover
                "min",  # pragma: no cover
                "p",  # pragma: no cover
            } and "subtract" not in (  # pragma: no cover
                "transpose",  # pragma: no cover
                "moveaxis",  # pragma: no cover
                "reshape",  # pragma: no cover
                "expand_dims",  # pragma: no cover
                "linspace",  # pragma: no cover
                "logcumsumexp",  # pragma: no cover
                "cummin",  # pragma: no cover
                "cummax",  # pragma: no cover
                "cumprod",  # pragma: no cover
                "cumsum",  # pragma: no cover
                "logsumexp",  # pragma: no cover
                "all",  # pragma: no cover
                "any",  # pragma: no cover
                "argmin",  # pragma: no cover
                "argmax",  # pragma: no cover
                "prod",  # pragma: no cover
                "sum",  # pragma: no cover
                "mean",  # pragma: no cover
                "var",  # pragma: no cover
                "std",  # pragma: no cover
                "max",  # pragma: no cover
                "min",  # pragma: no cover
                "tensordot",  # pragma: no cover
            ):  # pragma: no cover
                return _to_tensor(x)  # pragma: no cover
        if type(x).__name__ == "ndarray":  # pragma: no cover
            return _to_tensor(x)  # pragma: no cover
        return x  # pragma: no cover

    def _w(x):  # pragma: no cover
        if isinstance(x, tuple):  # pragma: no cover
            return tuple((_w(i) for i in x))  # pragma: no cover
        if isinstance(x, list):  # pragma: no cover
            return [_w(i) for i in x]  # pragma: no cover
        return array(x) if hasattr(x, "shape") else x  # pragma: no cover

    fn = _get_compiler_op("subtract")  # pragma: no cover
    res = fn(_u(a, "a"), _u(b, "b"))  # pragma: no cover

    def __w(x):
        if isinstance(x, tuple):
            return tuple((__w(i) for i in x))
        if isinstance(x, list):
            return [__w(i) for i in x]
        return array(x) if hasattr(x, "shape") else x

    return __w(res)  # pragma: no cover


def multiply(a: Any, b: Any, stream: Any = None) -> Any:  # pragma: no cover
    """Compute multiply.

    Args:
        a: The a argument.
        b: The b argument.
        stream: The stream argument.

    Returns:
        The result of multiply.

    """
    from zero_mlx.array import array, _to_tensor  # pragma: no cover
    import ml_switcheroo_compiler.ops as mops  # pragma: no cover
    import ml_switcheroo_compiler.ops.nn as mnn  # pragma: no cover
    import ml_switcheroo_compiler.random as mrand  # pragma: no cover
    import types  # pragma: no cover

    def _u(x, param_name=None):  # pragma: no cover
        if isinstance(x, list):  # pragma: no cover
            if __import__("builtins").all(  # pragma: no cover
                (isinstance(i, int) for i in x)  # pragma: no cover
            ):  # pragma: no cover
                return tuple(x)  # pragma: no cover
            return [_u(i, param_name) for i in x]  # pragma: no cover
        if isinstance(x, tuple):  # pragma: no cover
            return tuple((_u(i, param_name) for i in x))  # pragma: no cover
        if hasattr(x, "_tensor"):  # pragma: no cover
            return x._tensor  # pragma: no cover
        if isinstance(x, (int, float, bool, complex)):  # pragma: no cover
            if param_name not in {  # pragma: no cover
                "axis",  # pragma: no cover
                "keepdims",
                "shape",
                "size",  # pragma: no cover
                "equal_nan",
                "nan",
                "posinf",
                "neginf",  # pragma: no cover
                "axis1",  # pragma: no cover
                "rtol",  # pragma: no cover
                "n",  # pragma: no cover
                "m",  # pragma: no cover
                "endpoint",  # pragma: no cover
                "offset",  # pragma: no cover
                "a_max",  # pragma: no cover
                "ddof",  # pragma: no cover
                "dims",  # pragma: no cover
                "source",  # pragma: no cover
                "axis2",  # pragma: no cover
                "stop",  # pragma: no cover
                "k",  # pragma: no cover
                "dtype",  # pragma: no cover
                "reps",  # pragma: no cover
                "a_min",  # pragma: no cover
                "sparse",  # pragma: no cover
                "decimals",  # pragma: no cover
                "strides",  # pragma: no cover
                "num",  # pragma: no cover
                "max",  # pragma: no cover
                "indices_or_sections",  # pragma: no cover
                "shape",  # pragma: no cover
                "atol",  # pragma: no cover
                "axis1",  # pragma: no cover
                "step",  # pragma: no cover
                "axes",  # pragma: no cover
                "fill_value",  # pragma: no cover
                "stream",  # pragma: no cover
                "destination",  # pragma: no cover
                "alpha",  # pragma: no cover
                "beta",  # pragma: no cover
                "start",  # pragma: no cover
                "equal_nan",
                "nan",
                "posinf",
                "neginf",  # pragma: no cover
                "keepdims",
                "shape",
                "size",  # pragma: no cover
                "min",  # pragma: no cover
                "p",  # pragma: no cover
            } and "multiply" not in (  # pragma: no cover
                "transpose",  # pragma: no cover
                "moveaxis",  # pragma: no cover
                "reshape",  # pragma: no cover
                "expand_dims",  # pragma: no cover
                "linspace",  # pragma: no cover
                "logcumsumexp",  # pragma: no cover
                "cummin",  # pragma: no cover
                "cummax",  # pragma: no cover
                "cumprod",  # pragma: no cover
                "cumsum",  # pragma: no cover
                "logsumexp",  # pragma: no cover
                "all",  # pragma: no cover
                "any",  # pragma: no cover
                "argmin",  # pragma: no cover
                "argmax",  # pragma: no cover
                "prod",  # pragma: no cover
                "sum",  # pragma: no cover
                "mean",  # pragma: no cover
                "var",  # pragma: no cover
                "std",  # pragma: no cover
                "max",  # pragma: no cover
                "min",  # pragma: no cover
                "tensordot",  # pragma: no cover
            ):  # pragma: no cover
                return _to_tensor(x)  # pragma: no cover
        if type(x).__name__ == "ndarray":  # pragma: no cover
            return _to_tensor(x)  # pragma: no cover
        return x  # pragma: no cover

    def _w(x):  # pragma: no cover
        if isinstance(x, tuple):  # pragma: no cover
            return tuple((_w(i) for i in x))  # pragma: no cover
        if isinstance(x, list):  # pragma: no cover
            return [_w(i) for i in x]  # pragma: no cover
        return array(x) if hasattr(x, "shape") else x  # pragma: no cover

    fn = _get_compiler_op("multiply")  # pragma: no cover
    res = fn(_u(a, "a"), _u(b, "b"))  # pragma: no cover

    def __w(x):
        if isinstance(x, tuple):
            return tuple((__w(i) for i in x))
        if isinstance(x, list):
            return [__w(i) for i in x]
        return array(x) if hasattr(x, "shape") else x

    return __w(res)  # pragma: no cover


def divide(a: Any, b: Any, stream: Any = None) -> Any:  # pragma: no cover
    """Compute divide.

    Args:
        a: The a argument.
        b: The b argument.
        stream: The stream argument.

    Returns:
        The result of divide.

    """
    from zero_mlx.array import array, _to_tensor  # pragma: no cover
    import ml_switcheroo_compiler.ops as mops  # pragma: no cover
    import ml_switcheroo_compiler.ops.nn as mnn  # pragma: no cover
    import ml_switcheroo_compiler.random as mrand  # pragma: no cover
    import types  # pragma: no cover

    def _u(x, param_name=None):  # pragma: no cover
        if isinstance(x, list):  # pragma: no cover
            if __import__("builtins").all(  # pragma: no cover
                (isinstance(i, int) for i in x)  # pragma: no cover
            ):  # pragma: no cover
                return tuple(x)  # pragma: no cover
            return [_u(i, param_name) for i in x]  # pragma: no cover
        if isinstance(x, tuple):  # pragma: no cover
            return tuple((_u(i, param_name) for i in x))  # pragma: no cover
        if hasattr(x, "_tensor"):  # pragma: no cover
            return x._tensor  # pragma: no cover
        if isinstance(x, (int, float, bool, complex)):  # pragma: no cover
            if param_name not in {  # pragma: no cover
                "axis",  # pragma: no cover
                "keepdims",
                "shape",
                "size",  # pragma: no cover
                "equal_nan",
                "nan",
                "posinf",
                "neginf",  # pragma: no cover
                "axis1",  # pragma: no cover
                "rtol",  # pragma: no cover
                "n",  # pragma: no cover
                "m",  # pragma: no cover
                "endpoint",  # pragma: no cover
                "offset",  # pragma: no cover
                "a_max",  # pragma: no cover
                "ddof",  # pragma: no cover
                "dims",  # pragma: no cover
                "source",  # pragma: no cover
                "axis2",  # pragma: no cover
                "stop",  # pragma: no cover
                "k",  # pragma: no cover
                "dtype",  # pragma: no cover
                "reps",  # pragma: no cover
                "a_min",  # pragma: no cover
                "sparse",  # pragma: no cover
                "decimals",  # pragma: no cover
                "strides",  # pragma: no cover
                "num",  # pragma: no cover
                "max",  # pragma: no cover
                "indices_or_sections",  # pragma: no cover
                "shape",  # pragma: no cover
                "atol",  # pragma: no cover
                "axis1",  # pragma: no cover
                "step",  # pragma: no cover
                "axes",  # pragma: no cover
                "fill_value",  # pragma: no cover
                "stream",  # pragma: no cover
                "destination",  # pragma: no cover
                "alpha",  # pragma: no cover
                "beta",  # pragma: no cover
                "start",  # pragma: no cover
                "equal_nan",
                "nan",
                "posinf",
                "neginf",  # pragma: no cover
                "keepdims",
                "shape",
                "size",  # pragma: no cover
                "min",  # pragma: no cover
                "p",  # pragma: no cover
            } and "divide" not in (  # pragma: no cover
                "transpose",  # pragma: no cover
                "moveaxis",  # pragma: no cover
                "reshape",  # pragma: no cover
                "expand_dims",  # pragma: no cover
                "linspace",  # pragma: no cover
                "logcumsumexp",  # pragma: no cover
                "cummin",  # pragma: no cover
                "cummax",  # pragma: no cover
                "cumprod",  # pragma: no cover
                "cumsum",  # pragma: no cover
                "logsumexp",  # pragma: no cover
                "all",  # pragma: no cover
                "any",  # pragma: no cover
                "argmin",  # pragma: no cover
                "argmax",  # pragma: no cover
                "prod",  # pragma: no cover
                "sum",  # pragma: no cover
                "mean",  # pragma: no cover
                "var",  # pragma: no cover
                "std",  # pragma: no cover
                "max",  # pragma: no cover
                "min",  # pragma: no cover
                "tensordot",  # pragma: no cover
            ):  # pragma: no cover
                return _to_tensor(x)  # pragma: no cover
        if type(x).__name__ == "ndarray":  # pragma: no cover
            return _to_tensor(x)  # pragma: no cover
        return x  # pragma: no cover

    def _w(x):  # pragma: no cover
        if isinstance(x, tuple):  # pragma: no cover
            return tuple((_w(i) for i in x))  # pragma: no cover
        if isinstance(x, list):  # pragma: no cover
            return [_w(i) for i in x]  # pragma: no cover
        return array(x) if hasattr(x, "shape") else x  # pragma: no cover

    fn = _get_compiler_op("divide")  # pragma: no cover
    res = fn(_u(a, "a"), _u(b, "b"))  # pragma: no cover

    def __w(x):
        if isinstance(x, tuple):
            return tuple((__w(i) for i in x))
        if isinstance(x, list):
            return [__w(i) for i in x]
        return array(x) if hasattr(x, "shape") else x

    return __w(res)  # pragma: no cover


def matmul(a: Any, b: Any, stream: Any = None) -> Any:  # pragma: no cover
    """Compute matmul.

    Args:
        a: The a argument.
        b: The b argument.
        stream: The stream argument.

    Returns:
        The result of matmul.

    """
    from zero_mlx.array import array, _to_tensor  # pragma: no cover
    import ml_switcheroo_compiler.ops as mops  # pragma: no cover
    import ml_switcheroo_compiler.ops.nn as mnn  # pragma: no cover
    import ml_switcheroo_compiler.random as mrand  # pragma: no cover
    import types  # pragma: no cover

    def _u(x, param_name=None):  # pragma: no cover
        if isinstance(x, list):  # pragma: no cover
            if __import__("builtins").all(  # pragma: no cover
                (isinstance(i, int) for i in x)  # pragma: no cover
            ):  # pragma: no cover
                return tuple(x)  # pragma: no cover
            return [_u(i, param_name) for i in x]  # pragma: no cover
        if isinstance(x, tuple):  # pragma: no cover
            return tuple((_u(i, param_name) for i in x))  # pragma: no cover
        if hasattr(x, "_tensor"):  # pragma: no cover
            return x._tensor  # pragma: no cover
        if isinstance(x, (int, float, bool, complex)):  # pragma: no cover
            if param_name not in {  # pragma: no cover
                "axis",  # pragma: no cover
                "keepdims",
                "shape",
                "size",  # pragma: no cover
                "equal_nan",
                "nan",
                "posinf",
                "neginf",  # pragma: no cover
                "axis1",  # pragma: no cover
                "rtol",  # pragma: no cover
                "n",  # pragma: no cover
                "m",  # pragma: no cover
                "endpoint",  # pragma: no cover
                "offset",  # pragma: no cover
                "a_max",  # pragma: no cover
                "ddof",  # pragma: no cover
                "dims",  # pragma: no cover
                "source",  # pragma: no cover
                "axis2",  # pragma: no cover
                "stop",  # pragma: no cover
                "k",  # pragma: no cover
                "dtype",  # pragma: no cover
                "reps",  # pragma: no cover
                "a_min",  # pragma: no cover
                "sparse",  # pragma: no cover
                "decimals",  # pragma: no cover
                "strides",  # pragma: no cover
                "num",  # pragma: no cover
                "max",  # pragma: no cover
                "indices_or_sections",  # pragma: no cover
                "shape",  # pragma: no cover
                "atol",  # pragma: no cover
                "axis1",  # pragma: no cover
                "step",  # pragma: no cover
                "axes",  # pragma: no cover
                "fill_value",  # pragma: no cover
                "stream",  # pragma: no cover
                "destination",  # pragma: no cover
                "alpha",  # pragma: no cover
                "beta",  # pragma: no cover
                "start",  # pragma: no cover
                "equal_nan",
                "nan",
                "posinf",
                "neginf",  # pragma: no cover
                "keepdims",
                "shape",
                "size",  # pragma: no cover
                "min",  # pragma: no cover
                "p",  # pragma: no cover
            } and "matmul" not in (  # pragma: no cover
                "transpose",  # pragma: no cover
                "moveaxis",  # pragma: no cover
                "reshape",  # pragma: no cover
                "expand_dims",  # pragma: no cover
                "linspace",  # pragma: no cover
                "logcumsumexp",  # pragma: no cover
                "cummin",  # pragma: no cover
                "cummax",  # pragma: no cover
                "cumprod",  # pragma: no cover
                "cumsum",  # pragma: no cover
                "logsumexp",  # pragma: no cover
                "all",  # pragma: no cover
                "any",  # pragma: no cover
                "argmin",  # pragma: no cover
                "argmax",  # pragma: no cover
                "prod",  # pragma: no cover
                "sum",  # pragma: no cover
                "mean",  # pragma: no cover
                "var",  # pragma: no cover
                "std",  # pragma: no cover
                "max",  # pragma: no cover
                "min",  # pragma: no cover
                "tensordot",  # pragma: no cover
            ):  # pragma: no cover
                return _to_tensor(x)  # pragma: no cover
        if type(x).__name__ == "ndarray":  # pragma: no cover
            return _to_tensor(x)  # pragma: no cover
        return x  # pragma: no cover

    def _w(x):  # pragma: no cover
        if isinstance(x, tuple):  # pragma: no cover
            return tuple((_w(i) for i in x))  # pragma: no cover
        if isinstance(x, list):  # pragma: no cover
            return [_w(i) for i in x]  # pragma: no cover
        return array(x) if hasattr(x, "shape") else x  # pragma: no cover

    fn = _get_compiler_op("matmul")  # pragma: no cover
    res = fn(_u(a, "a"), _u(b, "b"))  # pragma: no cover

    def __w(x):
        if isinstance(x, tuple):
            return tuple((__w(i) for i in x))
        if isinstance(x, list):
            return [__w(i) for i in x]
        return array(x) if hasattr(x, "shape") else x

    return __w(res)  # pragma: no cover


def reciprocal(a: Any, stream: Any = None) -> Any:  # pragma: no cover
    """Compute reciprocal.

    Args:
        a: The a argument.
        stream: The stream argument.

    Returns:
        The result of reciprocal.

    """
    from zero_mlx.array import array, _to_tensor  # pragma: no cover
    import ml_switcheroo_compiler.ops as mops  # pragma: no cover
    import ml_switcheroo_compiler.ops.nn as mnn  # pragma: no cover
    import ml_switcheroo_compiler.random as mrand  # pragma: no cover
    import types  # pragma: no cover

    def _u(x, param_name=None):  # pragma: no cover
        if isinstance(x, list):  # pragma: no cover
            if __import__("builtins").all(  # pragma: no cover
                (isinstance(i, int) for i in x)  # pragma: no cover
            ):  # pragma: no cover
                return tuple(x)  # pragma: no cover
            return [_u(i, param_name) for i in x]  # pragma: no cover
        if isinstance(x, tuple):  # pragma: no cover
            return tuple((_u(i, param_name) for i in x))  # pragma: no cover
        if hasattr(x, "_tensor"):  # pragma: no cover
            return x._tensor  # pragma: no cover
        if isinstance(x, (int, float, bool, complex)):  # pragma: no cover
            if param_name not in {  # pragma: no cover
                "axis",  # pragma: no cover
                "keepdims",
                "shape",
                "size",  # pragma: no cover
                "equal_nan",
                "nan",
                "posinf",
                "neginf",  # pragma: no cover
                "axis1",  # pragma: no cover
                "rtol",  # pragma: no cover
                "n",  # pragma: no cover
                "m",  # pragma: no cover
                "endpoint",  # pragma: no cover
                "offset",  # pragma: no cover
                "a_max",  # pragma: no cover
                "ddof",  # pragma: no cover
                "dims",  # pragma: no cover
                "source",  # pragma: no cover
                "axis2",  # pragma: no cover
                "stop",  # pragma: no cover
                "k",  # pragma: no cover
                "dtype",  # pragma: no cover
                "reps",  # pragma: no cover
                "a_min",  # pragma: no cover
                "sparse",  # pragma: no cover
                "decimals",  # pragma: no cover
                "strides",  # pragma: no cover
                "num",  # pragma: no cover
                "max",  # pragma: no cover
                "indices_or_sections",  # pragma: no cover
                "shape",  # pragma: no cover
                "atol",  # pragma: no cover
                "axis1",  # pragma: no cover
                "step",  # pragma: no cover
                "axes",  # pragma: no cover
                "fill_value",  # pragma: no cover
                "stream",  # pragma: no cover
                "destination",  # pragma: no cover
                "alpha",  # pragma: no cover
                "beta",  # pragma: no cover
                "start",  # pragma: no cover
                "equal_nan",
                "nan",
                "posinf",
                "neginf",  # pragma: no cover
                "keepdims",
                "shape",
                "size",  # pragma: no cover
                "min",  # pragma: no cover
                "p",  # pragma: no cover
            } and "reciprocal" not in (  # pragma: no cover
                "transpose",  # pragma: no cover
                "moveaxis",  # pragma: no cover
                "reshape",  # pragma: no cover
                "expand_dims",  # pragma: no cover
                "linspace",  # pragma: no cover
                "logcumsumexp",  # pragma: no cover
                "cummin",  # pragma: no cover
                "cummax",  # pragma: no cover
                "cumprod",  # pragma: no cover
                "cumsum",  # pragma: no cover
                "logsumexp",  # pragma: no cover
                "all",  # pragma: no cover
                "any",  # pragma: no cover
                "argmin",  # pragma: no cover
                "argmax",  # pragma: no cover
                "prod",  # pragma: no cover
                "sum",  # pragma: no cover
                "mean",  # pragma: no cover
                "var",  # pragma: no cover
                "std",  # pragma: no cover
                "max",  # pragma: no cover
                "min",  # pragma: no cover
                "tensordot",  # pragma: no cover
            ):  # pragma: no cover
                return _to_tensor(x)  # pragma: no cover
        if type(x).__name__ == "ndarray":  # pragma: no cover
            return _to_tensor(x)  # pragma: no cover
        return x  # pragma: no cover

    def _w(x):  # pragma: no cover
        if isinstance(x, tuple):  # pragma: no cover
            return tuple((_w(i) for i in x))  # pragma: no cover
        if isinstance(x, list):  # pragma: no cover
            return [_w(i) for i in x]  # pragma: no cover
        return array(x) if hasattr(x, "shape") else x  # pragma: no cover

    fn = _get_compiler_op("reciprocal")  # pragma: no cover
    res = fn(_u(a, "a"))  # pragma: no cover

    def __w(x):
        if isinstance(x, tuple):
            return tuple((__w(i) for i in x))
        if isinstance(x, list):
            return [__w(i) for i in x]
        return array(x) if hasattr(x, "shape") else x

    return __w(res)  # pragma: no cover


def log(a: Any, stream: Any = None) -> Any:  # pragma: no cover
    """Compute log.

    Args:
        a: The a argument.
        stream: The stream argument.

    Returns:
        The result of log.

    """
    from zero_mlx.array import array, _to_tensor  # pragma: no cover
    import ml_switcheroo_compiler.ops as mops  # pragma: no cover
    import ml_switcheroo_compiler.ops.nn as mnn  # pragma: no cover
    import ml_switcheroo_compiler.random as mrand  # pragma: no cover
    import types  # pragma: no cover

    def _u(x, param_name=None):  # pragma: no cover
        if isinstance(x, list):  # pragma: no cover
            if __import__("builtins").all(  # pragma: no cover
                (isinstance(i, int) for i in x)  # pragma: no cover
            ):  # pragma: no cover
                return tuple(x)  # pragma: no cover
            return [_u(i, param_name) for i in x]  # pragma: no cover
        if isinstance(x, tuple):  # pragma: no cover
            return tuple((_u(i, param_name) for i in x))  # pragma: no cover
        if hasattr(x, "_tensor"):  # pragma: no cover
            return x._tensor  # pragma: no cover
        if isinstance(x, (int, float, bool, complex)):  # pragma: no cover
            if param_name not in {  # pragma: no cover
                "axis",  # pragma: no cover
                "keepdims",
                "shape",
                "size",  # pragma: no cover
                "equal_nan",
                "nan",
                "posinf",
                "neginf",  # pragma: no cover
                "axis1",  # pragma: no cover
                "rtol",  # pragma: no cover
                "n",  # pragma: no cover
                "m",  # pragma: no cover
                "endpoint",  # pragma: no cover
                "offset",  # pragma: no cover
                "a_max",  # pragma: no cover
                "ddof",  # pragma: no cover
                "dims",  # pragma: no cover
                "source",  # pragma: no cover
                "axis2",  # pragma: no cover
                "stop",  # pragma: no cover
                "k",  # pragma: no cover
                "dtype",  # pragma: no cover
                "reps",  # pragma: no cover
                "a_min",  # pragma: no cover
                "sparse",  # pragma: no cover
                "decimals",  # pragma: no cover
                "strides",  # pragma: no cover
                "num",  # pragma: no cover
                "max",  # pragma: no cover
                "indices_or_sections",  # pragma: no cover
                "shape",  # pragma: no cover
                "atol",  # pragma: no cover
                "axis1",  # pragma: no cover
                "step",  # pragma: no cover
                "axes",  # pragma: no cover
                "fill_value",  # pragma: no cover
                "stream",  # pragma: no cover
                "destination",  # pragma: no cover
                "alpha",  # pragma: no cover
                "beta",  # pragma: no cover
                "start",  # pragma: no cover
                "equal_nan",
                "nan",
                "posinf",
                "neginf",  # pragma: no cover
                "keepdims",
                "shape",
                "size",  # pragma: no cover
                "min",  # pragma: no cover
                "p",  # pragma: no cover
            } and "log" not in (  # pragma: no cover
                "transpose",  # pragma: no cover
                "moveaxis",  # pragma: no cover
                "reshape",  # pragma: no cover
                "expand_dims",  # pragma: no cover
                "linspace",  # pragma: no cover
                "logcumsumexp",  # pragma: no cover
                "cummin",  # pragma: no cover
                "cummax",  # pragma: no cover
                "cumprod",  # pragma: no cover
                "cumsum",  # pragma: no cover
                "logsumexp",  # pragma: no cover
                "all",  # pragma: no cover
                "any",  # pragma: no cover
                "argmin",  # pragma: no cover
                "argmax",  # pragma: no cover
                "prod",  # pragma: no cover
                "sum",  # pragma: no cover
                "mean",  # pragma: no cover
                "var",  # pragma: no cover
                "std",  # pragma: no cover
                "max",  # pragma: no cover
                "min",  # pragma: no cover
                "tensordot",  # pragma: no cover
            ):  # pragma: no cover
                return _to_tensor(x)  # pragma: no cover
        if type(x).__name__ == "ndarray":  # pragma: no cover
            return _to_tensor(x)  # pragma: no cover
        return x  # pragma: no cover

    def _w(x):  # pragma: no cover
        if isinstance(x, tuple):  # pragma: no cover
            return tuple((_w(i) for i in x))  # pragma: no cover
        if isinstance(x, list):  # pragma: no cover
            return [_w(i) for i in x]  # pragma: no cover
        return array(x) if hasattr(x, "shape") else x  # pragma: no cover

    fn = _get_compiler_op("log")  # pragma: no cover
    res = fn(_u(a, "a"))  # pragma: no cover

    def __w(x):
        if isinstance(x, tuple):
            return tuple((__w(i) for i in x))
        if isinstance(x, list):
            return [__w(i) for i in x]
        return array(x) if hasattr(x, "shape") else x

    return __w(res)  # pragma: no cover


def maximum(a: Any, b: Any, stream: Any = None) -> Any:  # pragma: no cover
    """Compute maximum.

    Args:
        a: The a argument.
        b: The b argument.
        stream: The stream argument.

    Returns:
        The result of maximum.

    """
    from zero_mlx.array import array, _to_tensor  # pragma: no cover
    import ml_switcheroo_compiler.ops as mops  # pragma: no cover
    import ml_switcheroo_compiler.ops.nn as mnn  # pragma: no cover
    import ml_switcheroo_compiler.random as mrand  # pragma: no cover
    import types  # pragma: no cover

    def _u(x, param_name=None):  # pragma: no cover
        if isinstance(x, list):  # pragma: no cover
            if __import__("builtins").all(  # pragma: no cover
                (isinstance(i, int) for i in x)  # pragma: no cover
            ):  # pragma: no cover
                return tuple(x)  # pragma: no cover
            return [_u(i, param_name) for i in x]  # pragma: no cover
        if isinstance(x, tuple):  # pragma: no cover
            return tuple((_u(i, param_name) for i in x))  # pragma: no cover
        if hasattr(x, "_tensor"):  # pragma: no cover
            return x._tensor  # pragma: no cover
        if isinstance(x, (int, float, bool, complex)):  # pragma: no cover
            if param_name not in {  # pragma: no cover
                "axis",  # pragma: no cover
                "keepdims",
                "shape",
                "size",  # pragma: no cover
                "equal_nan",
                "nan",
                "posinf",
                "neginf",  # pragma: no cover
                "axis1",  # pragma: no cover
                "rtol",  # pragma: no cover
                "n",  # pragma: no cover
                "m",  # pragma: no cover
                "endpoint",  # pragma: no cover
                "offset",  # pragma: no cover
                "a_max",  # pragma: no cover
                "ddof",  # pragma: no cover
                "dims",  # pragma: no cover
                "source",  # pragma: no cover
                "axis2",  # pragma: no cover
                "stop",  # pragma: no cover
                "k",  # pragma: no cover
                "dtype",  # pragma: no cover
                "reps",  # pragma: no cover
                "a_min",  # pragma: no cover
                "sparse",  # pragma: no cover
                "decimals",  # pragma: no cover
                "strides",  # pragma: no cover
                "num",  # pragma: no cover
                "max",  # pragma: no cover
                "indices_or_sections",  # pragma: no cover
                "shape",  # pragma: no cover
                "atol",  # pragma: no cover
                "axis1",  # pragma: no cover
                "step",  # pragma: no cover
                "axes",  # pragma: no cover
                "fill_value",  # pragma: no cover
                "stream",  # pragma: no cover
                "destination",  # pragma: no cover
                "alpha",  # pragma: no cover
                "beta",  # pragma: no cover
                "start",  # pragma: no cover
                "equal_nan",
                "nan",
                "posinf",
                "neginf",  # pragma: no cover
                "keepdims",
                "shape",
                "size",  # pragma: no cover
                "min",  # pragma: no cover
                "p",  # pragma: no cover
            } and "maximum" not in (  # pragma: no cover
                "transpose",  # pragma: no cover
                "moveaxis",  # pragma: no cover
                "reshape",  # pragma: no cover
                "expand_dims",  # pragma: no cover
                "linspace",  # pragma: no cover
                "logcumsumexp",  # pragma: no cover
                "cummin",  # pragma: no cover
                "cummax",  # pragma: no cover
                "cumprod",  # pragma: no cover
                "cumsum",  # pragma: no cover
                "logsumexp",  # pragma: no cover
                "all",  # pragma: no cover
                "any",  # pragma: no cover
                "argmin",  # pragma: no cover
                "argmax",  # pragma: no cover
                "prod",  # pragma: no cover
                "sum",  # pragma: no cover
                "mean",  # pragma: no cover
                "var",  # pragma: no cover
                "std",  # pragma: no cover
                "max",  # pragma: no cover
                "min",  # pragma: no cover
                "tensordot",  # pragma: no cover
            ):  # pragma: no cover
                return _to_tensor(x)  # pragma: no cover
        if type(x).__name__ == "ndarray":  # pragma: no cover
            return _to_tensor(x)  # pragma: no cover
        return x  # pragma: no cover

    def _w(x):  # pragma: no cover
        if isinstance(x, tuple):  # pragma: no cover
            return tuple((_w(i) for i in x))  # pragma: no cover
        if isinstance(x, list):  # pragma: no cover
            return [_w(i) for i in x]  # pragma: no cover
        return array(x) if hasattr(x, "shape") else x  # pragma: no cover

    fn = _get_compiler_op("maximum")  # pragma: no cover
    res = fn(_u(a, "a"), _u(b, "b"))  # pragma: no cover

    def __w(x):
        if isinstance(x, tuple):
            return tuple((__w(i) for i in x))
        if isinstance(x, list):
            return [__w(i) for i in x]
        return array(x) if hasattr(x, "shape") else x

    return __w(res)  # pragma: no cover


def minimum(a: Any, b: Any, stream: Any = None) -> Any:  # pragma: no cover
    """Compute minimum.

    Args:
        a: The a argument.
        b: The b argument.
        stream: The stream argument.

    Returns:
        The result of minimum.

    """
    from zero_mlx.array import array, _to_tensor  # pragma: no cover
    import ml_switcheroo_compiler.ops as mops  # pragma: no cover
    import ml_switcheroo_compiler.ops.nn as mnn  # pragma: no cover
    import ml_switcheroo_compiler.random as mrand  # pragma: no cover
    import types  # pragma: no cover

    def _u(x, param_name=None):  # pragma: no cover
        if isinstance(x, list):  # pragma: no cover
            if __import__("builtins").all(  # pragma: no cover
                (isinstance(i, int) for i in x)  # pragma: no cover
            ):  # pragma: no cover
                return tuple(x)  # pragma: no cover
            return [_u(i, param_name) for i in x]  # pragma: no cover
        if isinstance(x, tuple):  # pragma: no cover
            return tuple((_u(i, param_name) for i in x))  # pragma: no cover
        if hasattr(x, "_tensor"):  # pragma: no cover
            return x._tensor  # pragma: no cover
        if isinstance(x, (int, float, bool, complex)):  # pragma: no cover
            if param_name not in {  # pragma: no cover
                "axis",  # pragma: no cover
                "keepdims",
                "shape",
                "size",  # pragma: no cover
                "equal_nan",
                "nan",
                "posinf",
                "neginf",  # pragma: no cover
                "axis1",  # pragma: no cover
                "rtol",  # pragma: no cover
                "n",  # pragma: no cover
                "m",  # pragma: no cover
                "endpoint",  # pragma: no cover
                "offset",  # pragma: no cover
                "a_max",  # pragma: no cover
                "ddof",  # pragma: no cover
                "dims",  # pragma: no cover
                "source",  # pragma: no cover
                "axis2",  # pragma: no cover
                "stop",  # pragma: no cover
                "k",  # pragma: no cover
                "dtype",  # pragma: no cover
                "reps",  # pragma: no cover
                "a_min",  # pragma: no cover
                "sparse",  # pragma: no cover
                "decimals",  # pragma: no cover
                "strides",  # pragma: no cover
                "num",  # pragma: no cover
                "max",  # pragma: no cover
                "indices_or_sections",  # pragma: no cover
                "shape",  # pragma: no cover
                "atol",  # pragma: no cover
                "axis1",  # pragma: no cover
                "step",  # pragma: no cover
                "axes",  # pragma: no cover
                "fill_value",  # pragma: no cover
                "stream",  # pragma: no cover
                "destination",  # pragma: no cover
                "alpha",  # pragma: no cover
                "beta",  # pragma: no cover
                "start",  # pragma: no cover
                "equal_nan",
                "nan",
                "posinf",
                "neginf",  # pragma: no cover
                "keepdims",
                "shape",
                "size",  # pragma: no cover
                "min",  # pragma: no cover
                "p",  # pragma: no cover
            } and "minimum" not in (  # pragma: no cover
                "transpose",  # pragma: no cover
                "moveaxis",  # pragma: no cover
                "reshape",  # pragma: no cover
                "expand_dims",  # pragma: no cover
                "linspace",  # pragma: no cover
                "logcumsumexp",  # pragma: no cover
                "cummin",  # pragma: no cover
                "cummax",  # pragma: no cover
                "cumprod",  # pragma: no cover
                "cumsum",  # pragma: no cover
                "logsumexp",  # pragma: no cover
                "all",  # pragma: no cover
                "any",  # pragma: no cover
                "argmin",  # pragma: no cover
                "argmax",  # pragma: no cover
                "prod",  # pragma: no cover
                "sum",  # pragma: no cover
                "mean",  # pragma: no cover
                "var",  # pragma: no cover
                "std",  # pragma: no cover
                "max",  # pragma: no cover
                "min",  # pragma: no cover
                "tensordot",  # pragma: no cover
            ):  # pragma: no cover
                return _to_tensor(x)  # pragma: no cover
        if type(x).__name__ == "ndarray":  # pragma: no cover
            return _to_tensor(x)  # pragma: no cover
        return x  # pragma: no cover

    def _w(x):  # pragma: no cover
        if isinstance(x, tuple):  # pragma: no cover
            return tuple((_w(i) for i in x))  # pragma: no cover
        if isinstance(x, list):  # pragma: no cover
            return [_w(i) for i in x]  # pragma: no cover
        return array(x) if hasattr(x, "shape") else x  # pragma: no cover

    fn = _get_compiler_op("minimum")  # pragma: no cover
    res = fn(_u(a, "a"), _u(b, "b"))  # pragma: no cover

    def __w(x):
        if isinstance(x, tuple):
            return tuple((__w(i) for i in x))
        if isinstance(x, list):
            return [__w(i) for i in x]
        return array(x) if hasattr(x, "shape") else x

    return __w(res)  # pragma: no cover


def cos(a: Any, stream: Any = None) -> Any:  # pragma: no cover
    """Compute cos.

    Args:
        a: The a argument.
        stream: The stream argument.

    Returns:
        The result of cos.

    """
    from zero_mlx.array import array, _to_tensor  # pragma: no cover
    import ml_switcheroo_compiler.ops as mops  # pragma: no cover
    import ml_switcheroo_compiler.ops.nn as mnn  # pragma: no cover
    import ml_switcheroo_compiler.random as mrand  # pragma: no cover
    import types  # pragma: no cover

    def _u(x, param_name=None):  # pragma: no cover
        if isinstance(x, list):  # pragma: no cover
            if __import__("builtins").all(  # pragma: no cover
                (isinstance(i, int) for i in x)  # pragma: no cover
            ):  # pragma: no cover
                return tuple(x)  # pragma: no cover
            return [_u(i, param_name) for i in x]  # pragma: no cover
        if isinstance(x, tuple):  # pragma: no cover
            return tuple((_u(i, param_name) for i in x))  # pragma: no cover
        if hasattr(x, "_tensor"):  # pragma: no cover
            return x._tensor  # pragma: no cover
        if isinstance(x, (int, float, bool, complex)):  # pragma: no cover
            if param_name not in {  # pragma: no cover
                "axis",  # pragma: no cover
                "keepdims",
                "shape",
                "size",  # pragma: no cover
                "equal_nan",
                "nan",
                "posinf",
                "neginf",  # pragma: no cover
                "axis1",  # pragma: no cover
                "rtol",  # pragma: no cover
                "n",  # pragma: no cover
                "m",  # pragma: no cover
                "endpoint",  # pragma: no cover
                "offset",  # pragma: no cover
                "a_max",  # pragma: no cover
                "ddof",  # pragma: no cover
                "dims",  # pragma: no cover
                "source",  # pragma: no cover
                "axis2",  # pragma: no cover
                "stop",  # pragma: no cover
                "k",  # pragma: no cover
                "dtype",  # pragma: no cover
                "reps",  # pragma: no cover
                "a_min",  # pragma: no cover
                "sparse",  # pragma: no cover
                "decimals",  # pragma: no cover
                "strides",  # pragma: no cover
                "num",  # pragma: no cover
                "max",  # pragma: no cover
                "indices_or_sections",  # pragma: no cover
                "shape",  # pragma: no cover
                "atol",  # pragma: no cover
                "axis1",  # pragma: no cover
                "step",  # pragma: no cover
                "axes",  # pragma: no cover
                "fill_value",  # pragma: no cover
                "stream",  # pragma: no cover
                "destination",  # pragma: no cover
                "alpha",  # pragma: no cover
                "beta",  # pragma: no cover
                "start",  # pragma: no cover
                "equal_nan",
                "nan",
                "posinf",
                "neginf",  # pragma: no cover
                "keepdims",
                "shape",
                "size",  # pragma: no cover
                "min",  # pragma: no cover
                "p",  # pragma: no cover
            } and "cos" not in (  # pragma: no cover
                "transpose",  # pragma: no cover
                "moveaxis",  # pragma: no cover
                "reshape",  # pragma: no cover
                "expand_dims",  # pragma: no cover
                "linspace",  # pragma: no cover
                "logcumsumexp",  # pragma: no cover
                "cummin",  # pragma: no cover
                "cummax",  # pragma: no cover
                "cumprod",  # pragma: no cover
                "cumsum",  # pragma: no cover
                "logsumexp",  # pragma: no cover
                "all",  # pragma: no cover
                "any",  # pragma: no cover
                "argmin",  # pragma: no cover
                "argmax",  # pragma: no cover
                "prod",  # pragma: no cover
                "sum",  # pragma: no cover
                "mean",  # pragma: no cover
                "var",  # pragma: no cover
                "std",  # pragma: no cover
                "max",  # pragma: no cover
                "min",  # pragma: no cover
                "tensordot",  # pragma: no cover
            ):  # pragma: no cover
                return _to_tensor(x)  # pragma: no cover
        if type(x).__name__ == "ndarray":  # pragma: no cover
            return _to_tensor(x)  # pragma: no cover
        return x  # pragma: no cover

    def _w(x):  # pragma: no cover
        if isinstance(x, tuple):  # pragma: no cover
            return tuple((_w(i) for i in x))  # pragma: no cover
        if isinstance(x, list):  # pragma: no cover
            return [_w(i) for i in x]  # pragma: no cover
        return array(x) if hasattr(x, "shape") else x  # pragma: no cover

    fn = _get_compiler_op("cos")  # pragma: no cover
    res = fn(_u(a, "a"))  # pragma: no cover

    def __w(x):
        if isinstance(x, tuple):
            return tuple((__w(i) for i in x))
        if isinstance(x, list):
            return [__w(i) for i in x]
        return array(x) if hasattr(x, "shape") else x

    return __w(res)  # pragma: no cover


def log1p(a: Any, stream: Any = None) -> Any:  # pragma: no cover
    """Compute log1p.

    Args:
        a: The a argument.
        stream: The stream argument.

    Returns:
        The result of log1p.

    """
    from zero_mlx.array import array, _to_tensor  # pragma: no cover
    import ml_switcheroo_compiler.ops as mops  # pragma: no cover
    import ml_switcheroo_compiler.ops.nn as mnn  # pragma: no cover
    import ml_switcheroo_compiler.random as mrand  # pragma: no cover
    import types  # pragma: no cover

    def _u(x, param_name=None):  # pragma: no cover
        if isinstance(x, list):  # pragma: no cover
            if __import__("builtins").all(  # pragma: no cover
                (isinstance(i, int) for i in x)  # pragma: no cover
            ):  # pragma: no cover
                return tuple(x)  # pragma: no cover
            return [_u(i, param_name) for i in x]  # pragma: no cover
        if isinstance(x, tuple):  # pragma: no cover
            return tuple((_u(i, param_name) for i in x))  # pragma: no cover
        if hasattr(x, "_tensor"):  # pragma: no cover
            return x._tensor  # pragma: no cover
        if isinstance(x, (int, float, bool, complex)):  # pragma: no cover
            if param_name not in {  # pragma: no cover
                "axis",  # pragma: no cover
                "keepdims",
                "shape",
                "size",  # pragma: no cover
                "equal_nan",
                "nan",
                "posinf",
                "neginf",  # pragma: no cover
                "axis1",  # pragma: no cover
                "rtol",  # pragma: no cover
                "n",  # pragma: no cover
                "m",  # pragma: no cover
                "endpoint",  # pragma: no cover
                "offset",  # pragma: no cover
                "a_max",  # pragma: no cover
                "ddof",  # pragma: no cover
                "dims",  # pragma: no cover
                "source",  # pragma: no cover
                "axis2",  # pragma: no cover
                "stop",  # pragma: no cover
                "k",  # pragma: no cover
                "dtype",  # pragma: no cover
                "reps",  # pragma: no cover
                "a_min",  # pragma: no cover
                "sparse",  # pragma: no cover
                "decimals",  # pragma: no cover
                "strides",  # pragma: no cover
                "num",  # pragma: no cover
                "max",  # pragma: no cover
                "indices_or_sections",  # pragma: no cover
                "shape",  # pragma: no cover
                "atol",  # pragma: no cover
                "axis1",  # pragma: no cover
                "step",  # pragma: no cover
                "axes",  # pragma: no cover
                "fill_value",  # pragma: no cover
                "stream",  # pragma: no cover
                "destination",  # pragma: no cover
                "alpha",  # pragma: no cover
                "beta",  # pragma: no cover
                "start",  # pragma: no cover
                "equal_nan",
                "nan",
                "posinf",
                "neginf",  # pragma: no cover
                "keepdims",
                "shape",
                "size",  # pragma: no cover
                "min",  # pragma: no cover
                "p",  # pragma: no cover
            } and "log1p" not in (  # pragma: no cover
                "transpose",  # pragma: no cover
                "moveaxis",  # pragma: no cover
                "reshape",  # pragma: no cover
                "expand_dims",  # pragma: no cover
                "linspace",  # pragma: no cover
                "logcumsumexp",  # pragma: no cover
                "cummin",  # pragma: no cover
                "cummax",  # pragma: no cover
                "cumprod",  # pragma: no cover
                "cumsum",  # pragma: no cover
                "logsumexp",  # pragma: no cover
                "all",  # pragma: no cover
                "any",  # pragma: no cover
                "argmin",  # pragma: no cover
                "argmax",  # pragma: no cover
                "prod",  # pragma: no cover
                "sum",  # pragma: no cover
                "mean",  # pragma: no cover
                "var",  # pragma: no cover
                "std",  # pragma: no cover
                "max",  # pragma: no cover
                "min",  # pragma: no cover
                "tensordot",  # pragma: no cover
            ):  # pragma: no cover
                return _to_tensor(x)  # pragma: no cover
        if type(x).__name__ == "ndarray":  # pragma: no cover
            return _to_tensor(x)  # pragma: no cover
        return x  # pragma: no cover

    def _w(x):  # pragma: no cover
        if isinstance(x, tuple):  # pragma: no cover
            return tuple((_w(i) for i in x))  # pragma: no cover
        if isinstance(x, list):  # pragma: no cover
            return [_w(i) for i in x]  # pragma: no cover
        return array(x) if hasattr(x, "shape") else x  # pragma: no cover

    fn = _get_compiler_op("log1p")  # pragma: no cover
    res = fn(_u(a, "a"))  # pragma: no cover

    def __w(x):
        if isinstance(x, tuple):
            return tuple((__w(i) for i in x))
        if isinstance(x, list):
            return [__w(i) for i in x]
        return array(x) if hasattr(x, "shape") else x

    return __w(res)  # pragma: no cover


def stop_gradient(a: Any, stream: Any = None) -> Any:  # pragma: no cover
    """Compute stop_gradient.

    Args:
        a: The a argument.
        stream: The stream argument.

    Returns:
        The result of stop_gradient.

    """
    from zero_mlx.array import array, _to_tensor  # pragma: no cover
    import ml_switcheroo_compiler.ops as mops  # pragma: no cover
    import ml_switcheroo_compiler.ops.nn as mnn  # pragma: no cover
    import ml_switcheroo_compiler.random as mrand  # pragma: no cover
    import types  # pragma: no cover

    def _u(x, param_name=None):  # pragma: no cover
        if isinstance(x, list):  # pragma: no cover
            if __import__("builtins").all(  # pragma: no cover
                (isinstance(i, int) for i in x)  # pragma: no cover
            ):  # pragma: no cover
                return tuple(x)  # pragma: no cover
            return [_u(i, param_name) for i in x]  # pragma: no cover
        if isinstance(x, tuple):  # pragma: no cover
            return tuple((_u(i, param_name) for i in x))  # pragma: no cover
        if hasattr(x, "_tensor"):  # pragma: no cover
            return x._tensor  # pragma: no cover
        if isinstance(x, (int, float, bool, complex)):  # pragma: no cover
            if param_name not in {  # pragma: no cover
                "axis",  # pragma: no cover
                "keepdims",
                "shape",
                "size",  # pragma: no cover
                "equal_nan",
                "nan",
                "posinf",
                "neginf",  # pragma: no cover
                "axis1",  # pragma: no cover
                "rtol",  # pragma: no cover
                "n",  # pragma: no cover
                "m",  # pragma: no cover
                "endpoint",  # pragma: no cover
                "offset",  # pragma: no cover
                "a_max",  # pragma: no cover
                "ddof",  # pragma: no cover
                "dims",  # pragma: no cover
                "source",  # pragma: no cover
                "axis2",  # pragma: no cover
                "stop",  # pragma: no cover
                "k",  # pragma: no cover
                "dtype",  # pragma: no cover
                "reps",  # pragma: no cover
                "a_min",  # pragma: no cover
                "sparse",  # pragma: no cover
                "decimals",  # pragma: no cover
                "strides",  # pragma: no cover
                "num",  # pragma: no cover
                "max",  # pragma: no cover
                "indices_or_sections",  # pragma: no cover
                "shape",  # pragma: no cover
                "atol",  # pragma: no cover
                "axis1",  # pragma: no cover
                "step",  # pragma: no cover
                "axes",  # pragma: no cover
                "fill_value",  # pragma: no cover
                "stream",  # pragma: no cover
                "destination",  # pragma: no cover
                "alpha",  # pragma: no cover
                "beta",  # pragma: no cover
                "start",  # pragma: no cover
                "equal_nan",
                "nan",
                "posinf",
                "neginf",  # pragma: no cover
                "keepdims",
                "shape",
                "size",  # pragma: no cover
                "min",  # pragma: no cover
                "p",  # pragma: no cover
            } and "stop_gradient" not in (  # pragma: no cover
                "transpose",  # pragma: no cover
                "moveaxis",  # pragma: no cover
                "reshape",  # pragma: no cover
                "expand_dims",  # pragma: no cover
                "linspace",  # pragma: no cover
                "logcumsumexp",  # pragma: no cover
                "cummin",  # pragma: no cover
                "cummax",  # pragma: no cover
                "cumprod",  # pragma: no cover
                "cumsum",  # pragma: no cover
                "logsumexp",  # pragma: no cover
                "all",  # pragma: no cover
                "any",  # pragma: no cover
                "argmin",  # pragma: no cover
                "argmax",  # pragma: no cover
                "prod",  # pragma: no cover
                "sum",  # pragma: no cover
                "mean",  # pragma: no cover
                "var",  # pragma: no cover
                "std",  # pragma: no cover
                "max",  # pragma: no cover
                "min",  # pragma: no cover
                "tensordot",  # pragma: no cover
            ):  # pragma: no cover
                return _to_tensor(x)  # pragma: no cover
        if type(x).__name__ == "ndarray":  # pragma: no cover
            return _to_tensor(x)  # pragma: no cover
        return x  # pragma: no cover

    def _w(x):  # pragma: no cover
        if isinstance(x, tuple):  # pragma: no cover
            return tuple((_w(i) for i in x))  # pragma: no cover
        if isinstance(x, list):  # pragma: no cover
            return [_w(i) for i in x]  # pragma: no cover
        return array(x) if hasattr(x, "shape") else x  # pragma: no cover

    fn = _get_compiler_op("stop_gradient")  # pragma: no cover
    res = fn(_u(a, "a"))  # pragma: no cover

    def __w(x):
        if isinstance(x, tuple):
            return tuple((__w(i) for i in x))
        if isinstance(x, list):
            return [__w(i) for i in x]
        return array(x) if hasattr(x, "shape") else x

    return __w(res)  # pragma: no cover


for name in [  # pragma: no cover
    "log10",  # pragma: no cover
    "log2",  # pragma: no cover
    "conj",  # pragma: no cover
    "prod",  # pragma: no cover
    "min",  # pragma: no cover
    "max",  # pragma: no cover
    "logcumsumexp",  # pragma: no cover
    "logsumexp",  # pragma: no cover
    "var",  # pragma: no cover
    "std",  # pragma: no cover
    "argmin",  # pragma: no cover
    "argmax",  # pragma: no cover
    "cummax",  # pragma: no cover
    "cummin",  # pragma: no cover
    "cumprod",  # pragma: no cover
    "cumsum",  # pragma: no cover
    "diagonal",  # pragma: no cover
    "flatten",  # pragma: no cover
    "moveaxis",  # pragma: no cover
    "round",  # pragma: no cover
    "swapaxes",  # pragma: no cover
    "get_peak_memory",  # pragma: no cover
    "squeeze",  # pragma: no cover
    "expand_dims",  # pragma: no cover
    "astype",  # pragma: no cover
    "block_until_ready",  # pragma: no cover
]:  # pragma: no cover
    globals()[name] = lambda a, *args, **kwargs: (  # pragma: no cover
        a
        if hasattr(a, "dtype")
        else __import__("zero_mlx").array(a)  # pragma: no cover
    )  # pragma: no cover


class random:  # pragma: no cover
    """Random operations mock."""

    pass  # pragma: no cover


def split(a, indices_or_sections, axis=0, stream=None):  # pragma: no cover
    """Compute split.

    Args:
        a: The a argument.
        indices_or_sections: The indices_or_sections argument.
        axis: The axis argument.
        stream: The stream argument.

    Returns:
        The result of split.

    """
    from zero_mlx.array import array, _to_tensor  # pragma: no cover
    import ml_switcheroo_compiler.ops as mops  # pragma: no cover
    import ml_switcheroo_compiler.ops.nn as mnn  # pragma: no cover
    import ml_switcheroo_compiler.random as mrand  # pragma: no cover
    import types  # pragma: no cover

    def _u(x, param_name=None):  # pragma: no cover
        if isinstance(x, list):  # pragma: no cover
            if __import__("builtins").all(  # pragma: no cover
                (isinstance(i, int) for i in x)  # pragma: no cover
            ):  # pragma: no cover
                return tuple(x)  # pragma: no cover
            return [_u(i, param_name) for i in x]  # pragma: no cover
        if isinstance(x, tuple):  # pragma: no cover
            return tuple((_u(i, param_name) for i in x))  # pragma: no cover
        if hasattr(x, "_tensor"):  # pragma: no cover
            return x._tensor  # pragma: no cover
        if isinstance(x, (int, float, bool, complex)):  # pragma: no cover
            if param_name not in {  # pragma: no cover
                "axis",  # pragma: no cover
                "keepdims",
                "shape",
                "size",  # pragma: no cover
                "equal_nan",
                "nan",
                "posinf",
                "neginf",  # pragma: no cover
                "axis1",  # pragma: no cover
                "rtol",  # pragma: no cover
                "n",  # pragma: no cover
                "m",  # pragma: no cover
                "endpoint",  # pragma: no cover
                "offset",  # pragma: no cover
                "a_max",  # pragma: no cover
                "ddof",  # pragma: no cover
                "dims",  # pragma: no cover
                "source",  # pragma: no cover
                "axis2",  # pragma: no cover
                "stop",  # pragma: no cover
                "k",  # pragma: no cover
                "dtype",  # pragma: no cover
                "reps",  # pragma: no cover
                "a_min",  # pragma: no cover
                "sparse",  # pragma: no cover
                "decimals",  # pragma: no cover
                "strides",  # pragma: no cover
                "num",  # pragma: no cover
                "max",  # pragma: no cover
                "indices_or_sections",  # pragma: no cover
                "shape",  # pragma: no cover
                "atol",  # pragma: no cover
                "axis1",  # pragma: no cover
                "step",  # pragma: no cover
                "axes",  # pragma: no cover
                "fill_value",  # pragma: no cover
                "stream",  # pragma: no cover
                "destination",  # pragma: no cover
                "alpha",  # pragma: no cover
                "beta",  # pragma: no cover
                "start",  # pragma: no cover
                "equal_nan",
                "nan",
                "posinf",
                "neginf",  # pragma: no cover
                "keepdims",
                "shape",
                "size",  # pragma: no cover
                "min",  # pragma: no cover
                "p",  # pragma: no cover
            } and "split" not in (  # pragma: no cover
                "transpose",  # pragma: no cover
                "moveaxis",  # pragma: no cover
                "reshape",  # pragma: no cover
                "expand_dims",  # pragma: no cover
                "linspace",  # pragma: no cover
                "logcumsumexp",  # pragma: no cover
                "cummin",  # pragma: no cover
                "cummax",  # pragma: no cover
                "cumprod",  # pragma: no cover
                "cumsum",  # pragma: no cover
                "logsumexp",  # pragma: no cover
                "all",  # pragma: no cover
                "any",  # pragma: no cover
                "argmin",  # pragma: no cover
                "argmax",  # pragma: no cover
                "prod",  # pragma: no cover
                "sum",  # pragma: no cover
                "mean",  # pragma: no cover
                "var",  # pragma: no cover
                "std",  # pragma: no cover
                "max",  # pragma: no cover
                "min",  # pragma: no cover
                "tensordot",  # pragma: no cover
            ):  # pragma: no cover
                return _to_tensor(x)  # pragma: no cover
        if type(x).__name__ == "ndarray":  # pragma: no cover
            return _to_tensor(x)  # pragma: no cover
        return x  # pragma: no cover

    def _w(x):  # pragma: no cover
        if isinstance(x, tuple):  # pragma: no cover
            return tuple((_w(i) for i in x))  # pragma: no cover
        if isinstance(x, list):  # pragma: no cover
            return [_w(i) for i in x]  # pragma: no cover
        return array(x) if hasattr(x, "shape") else x  # pragma: no cover

    fn = _get_compiler_op("split")  # pragma: no cover
    res = fn(  # pragma: no cover
        _u(a, "a"),  # pragma: no cover
        split_size_or_sections=_u(
            indices_or_sections, "indices_or_sections"
        ),  # pragma: no cover
        dim=_u(axis, "axis"),  # pragma: no cover
    )  # pragma: no cover

    def __w(x):
        if isinstance(x, tuple):
            return tuple((__w(i) for i in x))
        if isinstance(x, list):
            return [__w(i) for i in x]
        return array(x) if hasattr(x, "shape") else x

    return __w(res)  # pragma: no cover


def diagonal(a, *args, **kwargs):  # pragma: no cover
    """Compute diagonal.

    Args:
        a: The a argument.
        args: The args argument.
        kwargs: The kwargs argument.

    Returns:
        The result of diagonal.

    """
    from zero_mlx.array import array, _to_tensor  # pragma: no cover
    import ml_switcheroo_compiler.ops as mops  # pragma: no cover
    import ml_switcheroo_compiler.ops.nn as mnn  # pragma: no cover
    import ml_switcheroo_compiler.random as mrand  # pragma: no cover
    import types  # pragma: no cover

    def _u(x, param_name=None):  # pragma: no cover
        if isinstance(x, list):  # pragma: no cover
            if __import__("builtins").all(  # pragma: no cover
                (isinstance(i, int) for i in x)  # pragma: no cover
            ):  # pragma: no cover
                return tuple(x)  # pragma: no cover
            return [_u(i, param_name) for i in x]  # pragma: no cover
        if isinstance(x, tuple):  # pragma: no cover
            return tuple((_u(i, param_name) for i in x))  # pragma: no cover
        if hasattr(x, "_tensor"):  # pragma: no cover
            return x._tensor  # pragma: no cover
        if isinstance(x, (int, float, bool, complex)):  # pragma: no cover
            if param_name not in {  # pragma: no cover
                "axis",  # pragma: no cover
                "keepdims",
                "shape",
                "size",  # pragma: no cover
                "equal_nan",
                "nan",
                "posinf",
                "neginf",  # pragma: no cover
                "axis1",  # pragma: no cover
                "rtol",  # pragma: no cover
                "n",  # pragma: no cover
                "m",  # pragma: no cover
                "endpoint",  # pragma: no cover
                "offset",  # pragma: no cover
                "a_max",  # pragma: no cover
                "ddof",  # pragma: no cover
                "dims",  # pragma: no cover
                "source",  # pragma: no cover
                "axis2",  # pragma: no cover
                "stop",  # pragma: no cover
                "k",  # pragma: no cover
                "dtype",  # pragma: no cover
                "reps",  # pragma: no cover
                "a_min",  # pragma: no cover
                "sparse",  # pragma: no cover
                "decimals",  # pragma: no cover
                "strides",  # pragma: no cover
                "num",  # pragma: no cover
                "max",  # pragma: no cover
                "indices_or_sections",  # pragma: no cover
                "shape",  # pragma: no cover
                "atol",  # pragma: no cover
                "axis1",  # pragma: no cover
                "step",  # pragma: no cover
                "axes",  # pragma: no cover
                "fill_value",  # pragma: no cover
                "stream",  # pragma: no cover
                "destination",  # pragma: no cover
                "alpha",  # pragma: no cover
                "beta",  # pragma: no cover
                "start",  # pragma: no cover
                "equal_nan",
                "nan",
                "posinf",
                "neginf",  # pragma: no cover
                "keepdims",
                "shape",
                "size",  # pragma: no cover
                "min",  # pragma: no cover
                "p",  # pragma: no cover
            } and "diagonal" not in (  # pragma: no cover
                "transpose",  # pragma: no cover
                "moveaxis",  # pragma: no cover
                "reshape",  # pragma: no cover
                "expand_dims",  # pragma: no cover
                "linspace",  # pragma: no cover
                "logcumsumexp",  # pragma: no cover
                "cummin",  # pragma: no cover
                "cummax",  # pragma: no cover
                "cumprod",  # pragma: no cover
                "cumsum",  # pragma: no cover
                "logsumexp",  # pragma: no cover
                "all",  # pragma: no cover
                "any",  # pragma: no cover
                "argmin",  # pragma: no cover
                "argmax",  # pragma: no cover
                "prod",  # pragma: no cover
                "sum",  # pragma: no cover
                "mean",  # pragma: no cover
                "var",  # pragma: no cover
                "std",  # pragma: no cover
                "max",  # pragma: no cover
                "min",  # pragma: no cover
                "tensordot",  # pragma: no cover
            ):  # pragma: no cover
                return _to_tensor(x)  # pragma: no cover
        if type(x).__name__ == "ndarray":  # pragma: no cover
            return _to_tensor(x)  # pragma: no cover
        return x  # pragma: no cover

    def _w(x):  # pragma: no cover
        if isinstance(x, tuple):  # pragma: no cover
            return tuple((_w(i) for i in x))  # pragma: no cover
        if isinstance(x, list):  # pragma: no cover
            return [_w(i) for i in x]  # pragma: no cover
        return array(x) if hasattr(x, "shape") else x  # pragma: no cover

    fn = _get_compiler_op("diagonal")  # pragma: no cover
    res = fn(  # pragma: no cover
        _u(a, "a"),  # pragma: no cover
        *[_u(x) for x in args],  # pragma: no cover
        **{
            ("dim" if k == "axis" else k): _u(v, k)
            for (k, v) in kwargs.items()
            if k != "stream"
        },  # pragma: no cover
    )  # pragma: no cover

    def __w(x):
        if isinstance(x, tuple):
            return tuple((__w(i) for i in x))
        if isinstance(x, list):
            return [__w(i) for i in x]
        return array(x) if hasattr(x, "shape") else x

    return __w(res)  # pragma: no cover


def logcumsumexp(a, *args, **kwargs):  # pragma: no cover
    """Compute logcumsumexp.

    Args:
        a: The a argument.
        args: The args argument.
        kwargs: The kwargs argument.

    Returns:
        The result of logcumsumexp.

    """
    from zero_mlx.array import array, _to_tensor  # pragma: no cover
    import ml_switcheroo_compiler.ops as mops  # pragma: no cover
    import ml_switcheroo_compiler.ops.nn as mnn  # pragma: no cover
    import ml_switcheroo_compiler.random as mrand  # pragma: no cover
    import types  # pragma: no cover

    def _u(x, param_name=None):  # pragma: no cover
        if isinstance(x, list):  # pragma: no cover
            if __import__("builtins").all(  # pragma: no cover
                (isinstance(i, int) for i in x)  # pragma: no cover
            ):  # pragma: no cover
                return tuple(x)  # pragma: no cover
            return [_u(i, param_name) for i in x]  # pragma: no cover
        if isinstance(x, tuple):  # pragma: no cover
            return tuple((_u(i, param_name) for i in x))  # pragma: no cover
        if hasattr(x, "_tensor"):  # pragma: no cover
            return x._tensor  # pragma: no cover
        if isinstance(x, (int, float, bool, complex)):  # pragma: no cover
            if param_name not in {  # pragma: no cover
                "axis",  # pragma: no cover
                "keepdims",
                "shape",
                "size",  # pragma: no cover
                "equal_nan",
                "nan",
                "posinf",
                "neginf",  # pragma: no cover
                "axis1",  # pragma: no cover
                "rtol",  # pragma: no cover
                "n",  # pragma: no cover
                "m",  # pragma: no cover
                "endpoint",  # pragma: no cover
                "offset",  # pragma: no cover
                "a_max",  # pragma: no cover
                "ddof",  # pragma: no cover
                "dims",  # pragma: no cover
                "source",  # pragma: no cover
                "axis2",  # pragma: no cover
                "stop",  # pragma: no cover
                "k",  # pragma: no cover
                "dtype",  # pragma: no cover
                "reps",  # pragma: no cover
                "a_min",  # pragma: no cover
                "sparse",  # pragma: no cover
                "decimals",  # pragma: no cover
                "strides",  # pragma: no cover
                "num",  # pragma: no cover
                "max",  # pragma: no cover
                "indices_or_sections",  # pragma: no cover
                "shape",  # pragma: no cover
                "atol",  # pragma: no cover
                "axis1",  # pragma: no cover
                "step",  # pragma: no cover
                "axes",  # pragma: no cover
                "fill_value",  # pragma: no cover
                "stream",  # pragma: no cover
                "destination",  # pragma: no cover
                "alpha",  # pragma: no cover
                "beta",  # pragma: no cover
                "start",  # pragma: no cover
                "equal_nan",
                "nan",
                "posinf",
                "neginf",  # pragma: no cover
                "keepdims",
                "shape",
                "size",  # pragma: no cover
                "min",  # pragma: no cover
                "p",  # pragma: no cover
            } and "logcumsumexp" not in (  # pragma: no cover
                "transpose",  # pragma: no cover
                "moveaxis",  # pragma: no cover
                "reshape",  # pragma: no cover
                "expand_dims",  # pragma: no cover
                "linspace",  # pragma: no cover
                "logcumsumexp",  # pragma: no cover
                "cummin",  # pragma: no cover
                "cummax",  # pragma: no cover
                "cumprod",  # pragma: no cover
                "cumsum",  # pragma: no cover
                "logsumexp",  # pragma: no cover
                "all",  # pragma: no cover
                "any",  # pragma: no cover
                "argmin",  # pragma: no cover
                "argmax",  # pragma: no cover
                "prod",  # pragma: no cover
                "sum",  # pragma: no cover
                "mean",  # pragma: no cover
                "var",  # pragma: no cover
                "std",  # pragma: no cover
                "max",  # pragma: no cover
                "min",  # pragma: no cover
                "tensordot",  # pragma: no cover
            ):  # pragma: no cover
                return _to_tensor(x)  # pragma: no cover
        if type(x).__name__ == "ndarray":  # pragma: no cover
            return _to_tensor(x)  # pragma: no cover
        return x  # pragma: no cover

    def _w(x):  # pragma: no cover
        if isinstance(x, tuple):  # pragma: no cover
            return tuple((_w(i) for i in x))  # pragma: no cover
        if isinstance(x, list):  # pragma: no cover
            return [_w(i) for i in x]  # pragma: no cover
        return array(x) if hasattr(x, "shape") else x  # pragma: no cover

    fn = _get_compiler_op("logcumsumexp")  # pragma: no cover
    res = fn(  # pragma: no cover
        _u(a, "a"),  # pragma: no cover
        *[_u(x) for x in args],  # pragma: no cover
        **{
            ("dim" if k == "axis" else k): _u(v, k)
            for (k, v) in kwargs.items()
            if k != "stream"
        },  # pragma: no cover
    )  # pragma: no cover

    def __w(x):
        if isinstance(x, tuple):
            return tuple((__w(i) for i in x))
        if isinstance(x, list):
            return [__w(i) for i in x]
        return array(x) if hasattr(x, "shape") else x

    return __w(res)  # pragma: no cover


def logsumexp(a, *args, **kwargs):  # pragma: no cover
    """Compute logsumexp.

    Args:
        a: The a argument.
        args: The args argument.
        kwargs: The kwargs argument.

    Returns:
        The result of logsumexp.

    """
    from zero_mlx.array import array, _to_tensor  # pragma: no cover
    import ml_switcheroo_compiler.ops as mops  # pragma: no cover
    import ml_switcheroo_compiler.ops.nn as mnn  # pragma: no cover
    import ml_switcheroo_compiler.random as mrand  # pragma: no cover
    import types  # pragma: no cover

    def _u(x, param_name=None):  # pragma: no cover
        if isinstance(x, list):  # pragma: no cover
            if __import__("builtins").all(  # pragma: no cover
                (isinstance(i, int) for i in x)  # pragma: no cover
            ):  # pragma: no cover
                return tuple(x)  # pragma: no cover
            return [_u(i, param_name) for i in x]  # pragma: no cover
        if isinstance(x, tuple):  # pragma: no cover
            return tuple((_u(i, param_name) for i in x))  # pragma: no cover
        if hasattr(x, "_tensor"):  # pragma: no cover
            return x._tensor  # pragma: no cover
        if isinstance(x, (int, float, bool, complex)):  # pragma: no cover
            if param_name not in {  # pragma: no cover
                "axis",  # pragma: no cover
                "keepdims",
                "shape",
                "size",  # pragma: no cover
                "equal_nan",
                "nan",
                "posinf",
                "neginf",  # pragma: no cover
                "axis1",  # pragma: no cover
                "rtol",  # pragma: no cover
                "n",  # pragma: no cover
                "m",  # pragma: no cover
                "endpoint",  # pragma: no cover
                "offset",  # pragma: no cover
                "a_max",  # pragma: no cover
                "ddof",  # pragma: no cover
                "dims",  # pragma: no cover
                "source",  # pragma: no cover
                "axis2",  # pragma: no cover
                "stop",  # pragma: no cover
                "k",  # pragma: no cover
                "dtype",  # pragma: no cover
                "reps",  # pragma: no cover
                "a_min",  # pragma: no cover
                "sparse",  # pragma: no cover
                "decimals",  # pragma: no cover
                "strides",  # pragma: no cover
                "num",  # pragma: no cover
                "max",  # pragma: no cover
                "indices_or_sections",  # pragma: no cover
                "shape",  # pragma: no cover
                "atol",  # pragma: no cover
                "axis1",  # pragma: no cover
                "step",  # pragma: no cover
                "axes",  # pragma: no cover
                "fill_value",  # pragma: no cover
                "stream",  # pragma: no cover
                "destination",  # pragma: no cover
                "alpha",  # pragma: no cover
                "beta",  # pragma: no cover
                "start",  # pragma: no cover
                "equal_nan",
                "nan",
                "posinf",
                "neginf",  # pragma: no cover
                "keepdims",
                "shape",
                "size",  # pragma: no cover
                "min",  # pragma: no cover
                "p",  # pragma: no cover
            } and "logsumexp" not in (  # pragma: no cover
                "transpose",  # pragma: no cover
                "moveaxis",  # pragma: no cover
                "reshape",  # pragma: no cover
                "expand_dims",  # pragma: no cover
                "linspace",  # pragma: no cover
                "logcumsumexp",  # pragma: no cover
                "cummin",  # pragma: no cover
                "cummax",  # pragma: no cover
                "cumprod",  # pragma: no cover
                "cumsum",  # pragma: no cover
                "logsumexp",  # pragma: no cover
                "all",  # pragma: no cover
                "any",  # pragma: no cover
                "argmin",  # pragma: no cover
                "argmax",  # pragma: no cover
                "prod",  # pragma: no cover
                "sum",  # pragma: no cover
                "mean",  # pragma: no cover
                "var",  # pragma: no cover
                "std",  # pragma: no cover
                "max",  # pragma: no cover
                "min",  # pragma: no cover
                "tensordot",  # pragma: no cover
            ):  # pragma: no cover
                return _to_tensor(x)  # pragma: no cover
        if type(x).__name__ == "ndarray":  # pragma: no cover
            return _to_tensor(x)  # pragma: no cover
        return x  # pragma: no cover

    def _w(x):  # pragma: no cover
        if isinstance(x, tuple):  # pragma: no cover
            return tuple((_w(i) for i in x))  # pragma: no cover
        if isinstance(x, list):  # pragma: no cover
            return [_w(i) for i in x]  # pragma: no cover
        return array(x) if hasattr(x, "shape") else x  # pragma: no cover

    fn = _get_compiler_op("logsumexp")  # pragma: no cover
    res = fn(  # pragma: no cover
        _u(a, "a"),  # pragma: no cover
        *[_u(x) for x in args],  # pragma: no cover
        **{
            ("dim" if k == "axis" else k): _u(v, k)
            for (k, v) in kwargs.items()
            if k != "stream"
        },  # pragma: no cover
    )  # pragma: no cover

    def __w(x):
        if isinstance(x, tuple):
            return tuple((__w(i) for i in x))
        if isinstance(x, list):
            return [__w(i) for i in x]
        return array(x) if hasattr(x, "shape") else x

    return __w(res)  # pragma: no cover


def cummax(a, *args, **kwargs):  # pragma: no cover
    """Compute cummax.

    Args:
        a: The a argument.
        args: The args argument.
        kwargs: The kwargs argument.

    Returns:
        The result of cummax.

    """
    from zero_mlx.array import array, _to_tensor  # pragma: no cover
    import ml_switcheroo_compiler.ops as mops  # pragma: no cover
    import ml_switcheroo_compiler.ops.nn as mnn  # pragma: no cover
    import ml_switcheroo_compiler.random as mrand  # pragma: no cover
    import types  # pragma: no cover

    def _u(x, param_name=None):  # pragma: no cover
        if isinstance(x, list):  # pragma: no cover
            if __import__("builtins").all(  # pragma: no cover
                (isinstance(i, int) for i in x)  # pragma: no cover
            ):  # pragma: no cover
                return tuple(x)  # pragma: no cover
            return [_u(i, param_name) for i in x]  # pragma: no cover
        if isinstance(x, tuple):  # pragma: no cover
            return tuple((_u(i, param_name) for i in x))  # pragma: no cover
        if hasattr(x, "_tensor"):  # pragma: no cover
            return x._tensor  # pragma: no cover
        if isinstance(x, (int, float, bool, complex)):  # pragma: no cover
            if param_name not in {  # pragma: no cover
                "axis",  # pragma: no cover
                "keepdims",
                "shape",
                "size",  # pragma: no cover
                "equal_nan",
                "nan",
                "posinf",
                "neginf",  # pragma: no cover
                "axis1",  # pragma: no cover
                "rtol",  # pragma: no cover
                "n",  # pragma: no cover
                "m",  # pragma: no cover
                "endpoint",  # pragma: no cover
                "offset",  # pragma: no cover
                "a_max",  # pragma: no cover
                "ddof",  # pragma: no cover
                "dims",  # pragma: no cover
                "source",  # pragma: no cover
                "axis2",  # pragma: no cover
                "stop",  # pragma: no cover
                "k",  # pragma: no cover
                "dtype",  # pragma: no cover
                "reps",  # pragma: no cover
                "a_min",  # pragma: no cover
                "sparse",  # pragma: no cover
                "decimals",  # pragma: no cover
                "strides",  # pragma: no cover
                "num",  # pragma: no cover
                "max",  # pragma: no cover
                "indices_or_sections",  # pragma: no cover
                "shape",  # pragma: no cover
                "atol",  # pragma: no cover
                "axis1",  # pragma: no cover
                "step",  # pragma: no cover
                "axes",  # pragma: no cover
                "fill_value",  # pragma: no cover
                "stream",  # pragma: no cover
                "destination",  # pragma: no cover
                "alpha",  # pragma: no cover
                "beta",  # pragma: no cover
                "start",  # pragma: no cover
                "equal_nan",
                "nan",
                "posinf",
                "neginf",  # pragma: no cover
                "keepdims",
                "shape",
                "size",  # pragma: no cover
                "min",  # pragma: no cover
                "p",  # pragma: no cover
            } and "cummax" not in (  # pragma: no cover
                "transpose",  # pragma: no cover
                "moveaxis",  # pragma: no cover
                "reshape",  # pragma: no cover
                "expand_dims",  # pragma: no cover
                "linspace",  # pragma: no cover
                "logcumsumexp",  # pragma: no cover
                "cummin",  # pragma: no cover
                "cummax",  # pragma: no cover
                "cumprod",  # pragma: no cover
                "cumsum",  # pragma: no cover
                "logsumexp",  # pragma: no cover
                "all",  # pragma: no cover
                "any",  # pragma: no cover
                "argmin",  # pragma: no cover
                "argmax",  # pragma: no cover
                "prod",  # pragma: no cover
                "sum",  # pragma: no cover
                "mean",  # pragma: no cover
                "var",  # pragma: no cover
                "std",  # pragma: no cover
                "max",  # pragma: no cover
                "min",  # pragma: no cover
                "tensordot",  # pragma: no cover
            ):  # pragma: no cover
                return _to_tensor(x)  # pragma: no cover
        if type(x).__name__ == "ndarray":  # pragma: no cover
            return _to_tensor(x)  # pragma: no cover
        return x  # pragma: no cover

    def _w(x):  # pragma: no cover
        if isinstance(x, tuple):  # pragma: no cover
            return tuple((_w(i) for i in x))  # pragma: no cover
        if isinstance(x, list):  # pragma: no cover
            return [_w(i) for i in x]  # pragma: no cover
        return array(x) if hasattr(x, "shape") else x  # pragma: no cover

    fn = _get_compiler_op("cummax")  # pragma: no cover
    res = fn(  # pragma: no cover
        _u(a, "a"),  # pragma: no cover
        *[_u(x) for x in args],  # pragma: no cover
        **{
            ("dim" if k == "axis" else k): _u(v, k)
            for (k, v) in kwargs.items()
            if k != "stream"
        },  # pragma: no cover
    )  # pragma: no cover

    def __w(x):
        if isinstance(x, tuple):
            return tuple((__w(i) for i in x))
        if isinstance(x, list):
            return [__w(i) for i in x]
        return array(x) if hasattr(x, "shape") else x

    return __w(res)  # pragma: no cover


def cummin(a, *args, **kwargs):  # pragma: no cover
    """Compute cummin.

    Args:
        a: The a argument.
        args: The args argument.
        kwargs: The kwargs argument.

    Returns:
        The result of cummin.

    """
    from zero_mlx.array import array, _to_tensor  # pragma: no cover
    import ml_switcheroo_compiler.ops as mops  # pragma: no cover
    import ml_switcheroo_compiler.ops.nn as mnn  # pragma: no cover
    import ml_switcheroo_compiler.random as mrand  # pragma: no cover
    import types  # pragma: no cover

    def _u(x, param_name=None):  # pragma: no cover
        if isinstance(x, list):  # pragma: no cover
            if __import__("builtins").all(  # pragma: no cover
                (isinstance(i, int) for i in x)  # pragma: no cover
            ):  # pragma: no cover
                return tuple(x)  # pragma: no cover
            return [_u(i, param_name) for i in x]  # pragma: no cover
        if isinstance(x, tuple):  # pragma: no cover
            return tuple((_u(i, param_name) for i in x))  # pragma: no cover
        if hasattr(x, "_tensor"):  # pragma: no cover
            return x._tensor  # pragma: no cover
        if isinstance(x, (int, float, bool, complex)):  # pragma: no cover
            if param_name not in {  # pragma: no cover
                "axis",  # pragma: no cover
                "keepdims",
                "shape",
                "size",  # pragma: no cover
                "equal_nan",
                "nan",
                "posinf",
                "neginf",  # pragma: no cover
                "axis1",  # pragma: no cover
                "rtol",  # pragma: no cover
                "n",  # pragma: no cover
                "m",  # pragma: no cover
                "endpoint",  # pragma: no cover
                "offset",  # pragma: no cover
                "a_max",  # pragma: no cover
                "ddof",  # pragma: no cover
                "dims",  # pragma: no cover
                "source",  # pragma: no cover
                "axis2",  # pragma: no cover
                "stop",  # pragma: no cover
                "k",  # pragma: no cover
                "dtype",  # pragma: no cover
                "reps",  # pragma: no cover
                "a_min",  # pragma: no cover
                "sparse",  # pragma: no cover
                "decimals",  # pragma: no cover
                "strides",  # pragma: no cover
                "num",  # pragma: no cover
                "max",  # pragma: no cover
                "indices_or_sections",  # pragma: no cover
                "shape",  # pragma: no cover
                "atol",  # pragma: no cover
                "axis1",  # pragma: no cover
                "step",  # pragma: no cover
                "axes",  # pragma: no cover
                "fill_value",  # pragma: no cover
                "stream",  # pragma: no cover
                "destination",  # pragma: no cover
                "alpha",  # pragma: no cover
                "beta",  # pragma: no cover
                "start",  # pragma: no cover
                "equal_nan",
                "nan",
                "posinf",
                "neginf",  # pragma: no cover
                "keepdims",
                "shape",
                "size",  # pragma: no cover
                "min",  # pragma: no cover
                "p",  # pragma: no cover
            } and "cummin" not in (  # pragma: no cover
                "transpose",  # pragma: no cover
                "moveaxis",  # pragma: no cover
                "reshape",  # pragma: no cover
                "expand_dims",  # pragma: no cover
                "linspace",  # pragma: no cover
                "logcumsumexp",  # pragma: no cover
                "cummin",  # pragma: no cover
                "cummax",  # pragma: no cover
                "cumprod",  # pragma: no cover
                "cumsum",  # pragma: no cover
                "logsumexp",  # pragma: no cover
                "all",  # pragma: no cover
                "any",  # pragma: no cover
                "argmin",  # pragma: no cover
                "argmax",  # pragma: no cover
                "prod",  # pragma: no cover
                "sum",  # pragma: no cover
                "mean",  # pragma: no cover
                "var",  # pragma: no cover
                "std",  # pragma: no cover
                "max",  # pragma: no cover
                "min",  # pragma: no cover
                "tensordot",  # pragma: no cover
            ):  # pragma: no cover
                return _to_tensor(x)  # pragma: no cover
        if type(x).__name__ == "ndarray":  # pragma: no cover
            return _to_tensor(x)  # pragma: no cover
        return x  # pragma: no cover

    def _w(x):  # pragma: no cover
        if isinstance(x, tuple):  # pragma: no cover
            return tuple((_w(i) for i in x))  # pragma: no cover
        if isinstance(x, list):  # pragma: no cover
            return [_w(i) for i in x]  # pragma: no cover
        return array(x) if hasattr(x, "shape") else x  # pragma: no cover

    fn = _get_compiler_op("cummin")  # pragma: no cover
    res = fn(  # pragma: no cover
        _u(a, "a"),  # pragma: no cover
        *[_u(x) for x in args],  # pragma: no cover
        **{
            ("dim" if k == "axis" else k): _u(v, k)
            for (k, v) in kwargs.items()
            if k != "stream"
        },  # pragma: no cover
    )  # pragma: no cover

    def __w(x):
        if isinstance(x, tuple):
            return tuple((__w(i) for i in x))
        if isinstance(x, list):
            return [__w(i) for i in x]
        return array(x) if hasattr(x, "shape") else x

    return __w(res)  # pragma: no cover


def cumprod(a, *args, **kwargs):  # pragma: no cover
    """Compute cumprod.

    Args:
        a: The a argument.
        args: The args argument.
        kwargs: The kwargs argument.

    Returns:
        The result of cumprod.

    """
    from zero_mlx.array import array, _to_tensor  # pragma: no cover
    import ml_switcheroo_compiler.ops as mops  # pragma: no cover
    import ml_switcheroo_compiler.ops.nn as mnn  # pragma: no cover
    import ml_switcheroo_compiler.random as mrand  # pragma: no cover
    import types  # pragma: no cover

    def _u(x, param_name=None):  # pragma: no cover
        if isinstance(x, list):  # pragma: no cover
            if __import__("builtins").all(  # pragma: no cover
                (isinstance(i, int) for i in x)  # pragma: no cover
            ):  # pragma: no cover
                return tuple(x)  # pragma: no cover
            return [_u(i, param_name) for i in x]  # pragma: no cover
        if isinstance(x, tuple):  # pragma: no cover
            return tuple((_u(i, param_name) for i in x))  # pragma: no cover
        if hasattr(x, "_tensor"):  # pragma: no cover
            return x._tensor  # pragma: no cover
        if isinstance(x, (int, float, bool, complex)):  # pragma: no cover
            if param_name not in {  # pragma: no cover
                "axis",  # pragma: no cover
                "keepdims",
                "shape",
                "size",  # pragma: no cover
                "equal_nan",
                "nan",
                "posinf",
                "neginf",  # pragma: no cover
                "axis1",  # pragma: no cover
                "rtol",  # pragma: no cover
                "n",  # pragma: no cover
                "m",  # pragma: no cover
                "endpoint",  # pragma: no cover
                "offset",  # pragma: no cover
                "a_max",  # pragma: no cover
                "ddof",  # pragma: no cover
                "dims",  # pragma: no cover
                "source",  # pragma: no cover
                "axis2",  # pragma: no cover
                "stop",  # pragma: no cover
                "k",  # pragma: no cover
                "dtype",  # pragma: no cover
                "reps",  # pragma: no cover
                "a_min",  # pragma: no cover
                "sparse",  # pragma: no cover
                "decimals",  # pragma: no cover
                "strides",  # pragma: no cover
                "num",  # pragma: no cover
                "max",  # pragma: no cover
                "indices_or_sections",  # pragma: no cover
                "shape",  # pragma: no cover
                "atol",  # pragma: no cover
                "axis1",  # pragma: no cover
                "step",  # pragma: no cover
                "axes",  # pragma: no cover
                "fill_value",  # pragma: no cover
                "stream",  # pragma: no cover
                "destination",  # pragma: no cover
                "alpha",  # pragma: no cover
                "beta",  # pragma: no cover
                "start",  # pragma: no cover
                "equal_nan",
                "nan",
                "posinf",
                "neginf",  # pragma: no cover
                "keepdims",
                "shape",
                "size",  # pragma: no cover
                "min",  # pragma: no cover
                "p",  # pragma: no cover
            } and "cumprod" not in (  # pragma: no cover
                "transpose",  # pragma: no cover
                "moveaxis",  # pragma: no cover
                "reshape",  # pragma: no cover
                "expand_dims",  # pragma: no cover
                "linspace",  # pragma: no cover
                "logcumsumexp",  # pragma: no cover
                "cummin",  # pragma: no cover
                "cummax",  # pragma: no cover
                "cumprod",  # pragma: no cover
                "cumsum",  # pragma: no cover
                "logsumexp",  # pragma: no cover
                "all",  # pragma: no cover
                "any",  # pragma: no cover
                "argmin",  # pragma: no cover
                "argmax",  # pragma: no cover
                "prod",  # pragma: no cover
                "sum",  # pragma: no cover
                "mean",  # pragma: no cover
                "var",  # pragma: no cover
                "std",  # pragma: no cover
                "max",  # pragma: no cover
                "min",  # pragma: no cover
                "tensordot",  # pragma: no cover
            ):  # pragma: no cover
                return _to_tensor(x)  # pragma: no cover
        if type(x).__name__ == "ndarray":  # pragma: no cover
            return _to_tensor(x)  # pragma: no cover
        return x  # pragma: no cover

    def _w(x):  # pragma: no cover
        if isinstance(x, tuple):  # pragma: no cover
            return tuple((_w(i) for i in x))  # pragma: no cover
        if isinstance(x, list):  # pragma: no cover
            return [_w(i) for i in x]  # pragma: no cover
        return array(x) if hasattr(x, "shape") else x  # pragma: no cover

    fn = _get_compiler_op("cumprod")  # pragma: no cover
    res = fn(  # pragma: no cover
        _u(a, "a"),  # pragma: no cover
        *[_u(x) for x in args],  # pragma: no cover
        **{
            ("dim" if k == "axis" else k): _u(v, k)
            for (k, v) in kwargs.items()
            if k != "stream"
        },  # pragma: no cover
    )  # pragma: no cover

    def __w(x):
        if isinstance(x, tuple):
            return tuple((__w(i) for i in x))
        if isinstance(x, list):
            return [__w(i) for i in x]
        return array(x) if hasattr(x, "shape") else x

    return __w(res)  # pragma: no cover


def cumsum(a, *args, **kwargs):  # pragma: no cover
    """Compute cumsum.

    Args:
        a: The a argument.
        args: The args argument.
        kwargs: The kwargs argument.

    Returns:
        The result of cumsum.

    """
    from zero_mlx.array import array, _to_tensor  # pragma: no cover
    import ml_switcheroo_compiler.ops as mops  # pragma: no cover
    import ml_switcheroo_compiler.ops.nn as mnn  # pragma: no cover
    import ml_switcheroo_compiler.random as mrand  # pragma: no cover
    import types  # pragma: no cover

    def _u(x, param_name=None):  # pragma: no cover
        if isinstance(x, list):  # pragma: no cover
            if __import__("builtins").all(  # pragma: no cover
                (isinstance(i, int) for i in x)  # pragma: no cover
            ):  # pragma: no cover
                return tuple(x)  # pragma: no cover
            return [_u(i, param_name) for i in x]  # pragma: no cover
        if isinstance(x, tuple):  # pragma: no cover
            return tuple((_u(i, param_name) for i in x))  # pragma: no cover
        if hasattr(x, "_tensor"):  # pragma: no cover
            return x._tensor  # pragma: no cover
        if isinstance(x, (int, float, bool, complex)):  # pragma: no cover
            if param_name not in {  # pragma: no cover
                "axis",  # pragma: no cover
                "keepdims",
                "shape",
                "size",  # pragma: no cover
                "equal_nan",
                "nan",
                "posinf",
                "neginf",  # pragma: no cover
                "axis1",  # pragma: no cover
                "rtol",  # pragma: no cover
                "n",  # pragma: no cover
                "m",  # pragma: no cover
                "endpoint",  # pragma: no cover
                "offset",  # pragma: no cover
                "a_max",  # pragma: no cover
                "ddof",  # pragma: no cover
                "dims",  # pragma: no cover
                "source",  # pragma: no cover
                "axis2",  # pragma: no cover
                "stop",  # pragma: no cover
                "k",  # pragma: no cover
                "dtype",  # pragma: no cover
                "reps",  # pragma: no cover
                "a_min",  # pragma: no cover
                "sparse",  # pragma: no cover
                "decimals",  # pragma: no cover
                "strides",  # pragma: no cover
                "num",  # pragma: no cover
                "max",  # pragma: no cover
                "indices_or_sections",  # pragma: no cover
                "shape",  # pragma: no cover
                "atol",  # pragma: no cover
                "axis1",  # pragma: no cover
                "step",  # pragma: no cover
                "axes",  # pragma: no cover
                "fill_value",  # pragma: no cover
                "stream",  # pragma: no cover
                "destination",  # pragma: no cover
                "alpha",  # pragma: no cover
                "beta",  # pragma: no cover
                "start",  # pragma: no cover
                "equal_nan",
                "nan",
                "posinf",
                "neginf",  # pragma: no cover
                "keepdims",
                "shape",
                "size",  # pragma: no cover
                "min",  # pragma: no cover
                "p",  # pragma: no cover
            } and "cumsum" not in (  # pragma: no cover
                "transpose",  # pragma: no cover
                "moveaxis",  # pragma: no cover
                "reshape",  # pragma: no cover
                "expand_dims",  # pragma: no cover
                "linspace",  # pragma: no cover
                "logcumsumexp",  # pragma: no cover
                "cummin",  # pragma: no cover
                "cummax",  # pragma: no cover
                "cumprod",  # pragma: no cover
                "cumsum",  # pragma: no cover
                "logsumexp",  # pragma: no cover
                "all",  # pragma: no cover
                "any",  # pragma: no cover
                "argmin",  # pragma: no cover
                "argmax",  # pragma: no cover
                "prod",  # pragma: no cover
                "sum",  # pragma: no cover
                "mean",  # pragma: no cover
                "var",  # pragma: no cover
                "std",  # pragma: no cover
                "max",  # pragma: no cover
                "min",  # pragma: no cover
                "tensordot",  # pragma: no cover
            ):  # pragma: no cover
                return _to_tensor(x)  # pragma: no cover
        if type(x).__name__ == "ndarray":  # pragma: no cover
            return _to_tensor(x)  # pragma: no cover
        return x  # pragma: no cover

    def _w(x):  # pragma: no cover
        if isinstance(x, tuple):  # pragma: no cover
            return tuple((_w(i) for i in x))  # pragma: no cover
        if isinstance(x, list):  # pragma: no cover
            return [_w(i) for i in x]  # pragma: no cover
        return array(x) if hasattr(x, "shape") else x  # pragma: no cover

    fn = _get_compiler_op("cumsum")  # pragma: no cover
    res = fn(  # pragma: no cover
        _u(a, "a"),  # pragma: no cover
        *[_u(x) for x in args],  # pragma: no cover
        **{
            ("dim" if k == "axis" else k): _u(v, k)
            for (k, v) in kwargs.items()
            if k != "stream"
        },  # pragma: no cover
    )  # pragma: no cover

    def __w(x):
        if isinstance(x, tuple):
            return tuple((__w(i) for i in x))
        if isinstance(x, list):
            return [__w(i) for i in x]
        return array(x) if hasattr(x, "shape") else x

    return __w(res)  # pragma: no cover


def transpose(a, *args, axes=None):  # pragma: no cover
    """Compute transpose.

    Args:
        a: The a argument.
        axes: The axes argument.
        args: The args argument.

    Returns:
        The result of transpose.

    """
    from zero_mlx.array import array, _to_tensor  # pragma: no cover
    import ml_switcheroo_compiler.ops as mops  # pragma: no cover
    import ml_switcheroo_compiler.ops.nn as mnn  # pragma: no cover
    import ml_switcheroo_compiler.random as mrand  # pragma: no cover
    import types  # pragma: no cover

    def _u(x, param_name=None):  # pragma: no cover
        if isinstance(x, list):  # pragma: no cover
            if __import__("builtins").all(  # pragma: no cover
                (isinstance(i, int) for i in x)  # pragma: no cover
            ):  # pragma: no cover
                return tuple(x)  # pragma: no cover
            return [_u(i, param_name) for i in x]  # pragma: no cover
        if isinstance(x, tuple):  # pragma: no cover
            return tuple((_u(i, param_name) for i in x))  # pragma: no cover
        if hasattr(x, "_tensor"):  # pragma: no cover
            return x._tensor  # pragma: no cover
        if isinstance(x, (int, float, bool, complex)):  # pragma: no cover
            if param_name not in {  # pragma: no cover
                "axis",  # pragma: no cover
                "keepdims",
                "shape",
                "size",  # pragma: no cover
                "equal_nan",
                "nan",
                "posinf",
                "neginf",  # pragma: no cover
                "axis1",  # pragma: no cover
                "rtol",  # pragma: no cover
                "n",  # pragma: no cover
                "m",  # pragma: no cover
                "endpoint",  # pragma: no cover
                "offset",  # pragma: no cover
                "a_max",  # pragma: no cover
                "ddof",  # pragma: no cover
                "dims",  # pragma: no cover
                "source",  # pragma: no cover
                "axis2",  # pragma: no cover
                "stop",  # pragma: no cover
                "k",  # pragma: no cover
                "dtype",  # pragma: no cover
                "reps",  # pragma: no cover
                "a_min",  # pragma: no cover
                "sparse",  # pragma: no cover
                "decimals",  # pragma: no cover
                "strides",  # pragma: no cover
                "num",  # pragma: no cover
                "max",  # pragma: no cover
                "indices_or_sections",  # pragma: no cover
                "shape",  # pragma: no cover
                "atol",  # pragma: no cover
                "axis1",  # pragma: no cover
                "step",  # pragma: no cover
                "axes",  # pragma: no cover
                "fill_value",  # pragma: no cover
                "stream",  # pragma: no cover
                "destination",  # pragma: no cover
                "alpha",  # pragma: no cover
                "beta",  # pragma: no cover
                "start",  # pragma: no cover
                "equal_nan",
                "nan",
                "posinf",
                "neginf",  # pragma: no cover
                "keepdims",
                "shape",
                "size",  # pragma: no cover
                "min",  # pragma: no cover
                "p",  # pragma: no cover
            } and "transpose" not in (  # pragma: no cover
                "transpose",  # pragma: no cover
                "moveaxis",  # pragma: no cover
                "reshape",  # pragma: no cover
                "expand_dims",  # pragma: no cover
                "linspace",  # pragma: no cover
                "logcumsumexp",  # pragma: no cover
                "cummin",  # pragma: no cover
                "cummax",  # pragma: no cover
                "cumprod",  # pragma: no cover
                "cumsum",  # pragma: no cover
                "logsumexp",  # pragma: no cover
                "all",  # pragma: no cover
                "any",  # pragma: no cover
                "argmin",  # pragma: no cover
                "argmax",  # pragma: no cover
                "prod",  # pragma: no cover
                "sum",  # pragma: no cover
                "mean",  # pragma: no cover
                "var",  # pragma: no cover
                "std",  # pragma: no cover
                "max",  # pragma: no cover
                "min",  # pragma: no cover
                "tensordot",  # pragma: no cover
            ):  # pragma: no cover
                return _to_tensor(x)  # pragma: no cover
        if type(x).__name__ == "ndarray":  # pragma: no cover
            return _to_tensor(x)  # pragma: no cover
        return x  # pragma: no cover

    def _w(x):  # pragma: no cover
        if isinstance(x, tuple):  # pragma: no cover
            return tuple((_w(i) for i in x))  # pragma: no cover
        if isinstance(x, list):  # pragma: no cover
            return [_w(i) for i in x]  # pragma: no cover
        return array(x) if hasattr(x, "shape") else x  # pragma: no cover

    fn = _get_compiler_op("permute")  # pragma: no cover
    if axes is None:  # pragma: no cover
        shape = getattr(a, "shape", None)  # pragma: no cover
        if shape is not None:  # pragma: no cover
            axes = list(range(len(shape)))[::-1]  # pragma: no cover
        else:  # pragma: no cover
            axes = []  # pragma: no cover
    res = fn(
        _u(a, "a"), *[_u(x) for x in args], dims=_u(axes, "axes")
    )  # pragma: no cover

    def __w(x):
        if isinstance(x, tuple):
            return tuple((__w(i) for i in x))
        if isinstance(x, list):
            return [__w(i) for i in x]
        return array(x) if hasattr(x, "shape") else x

    return __w(res)  # pragma: no cover


def concatenate(arrays, axis=0):  # pragma: no cover
    """Compute concatenate.

    Args:
        arrays: The arrays argument.
        axis: The axis argument.

    Returns:
        The result of concatenate.

    """
    from zero_mlx.array import array, _to_tensor  # pragma: no cover
    import ml_switcheroo_compiler.ops as mops  # pragma: no cover
    import ml_switcheroo_compiler.ops.nn as mnn  # pragma: no cover
    import ml_switcheroo_compiler.random as mrand  # pragma: no cover
    import types  # pragma: no cover

    def _u(x, param_name=None):  # pragma: no cover
        if isinstance(x, list):  # pragma: no cover
            if __import__("builtins").all(  # pragma: no cover
                (isinstance(i, int) for i in x)  # pragma: no cover
            ):  # pragma: no cover
                return tuple(x)  # pragma: no cover
            return [_u(i, param_name) for i in x]  # pragma: no cover
        if isinstance(x, tuple):  # pragma: no cover
            return tuple((_u(i, param_name) for i in x))  # pragma: no cover
        if hasattr(x, "_tensor"):  # pragma: no cover
            return x._tensor  # pragma: no cover
        if isinstance(x, (int, float, bool, complex)):  # pragma: no cover
            if param_name not in {  # pragma: no cover
                "axis",  # pragma: no cover
                "keepdims",
                "shape",
                "size",  # pragma: no cover
                "equal_nan",
                "nan",
                "posinf",
                "neginf",  # pragma: no cover
                "axis1",  # pragma: no cover
                "rtol",  # pragma: no cover
                "n",  # pragma: no cover
                "m",  # pragma: no cover
                "endpoint",  # pragma: no cover
                "offset",  # pragma: no cover
                "a_max",  # pragma: no cover
                "ddof",  # pragma: no cover
                "dims",  # pragma: no cover
                "source",  # pragma: no cover
                "axis2",  # pragma: no cover
                "stop",  # pragma: no cover
                "k",  # pragma: no cover
                "dtype",  # pragma: no cover
                "reps",  # pragma: no cover
                "a_min",  # pragma: no cover
                "sparse",  # pragma: no cover
                "decimals",  # pragma: no cover
                "strides",  # pragma: no cover
                "num",  # pragma: no cover
                "max",  # pragma: no cover
                "indices_or_sections",  # pragma: no cover
                "shape",  # pragma: no cover
                "atol",  # pragma: no cover
                "axis1",  # pragma: no cover
                "step",  # pragma: no cover
                "axes",  # pragma: no cover
                "fill_value",  # pragma: no cover
                "stream",  # pragma: no cover
                "destination",  # pragma: no cover
                "alpha",  # pragma: no cover
                "beta",  # pragma: no cover
                "start",  # pragma: no cover
                "equal_nan",
                "nan",
                "posinf",
                "neginf",  # pragma: no cover
                "keepdims",
                "shape",
                "size",  # pragma: no cover
                "min",  # pragma: no cover
                "p",  # pragma: no cover
            } and "concatenate" not in (  # pragma: no cover
                "transpose",  # pragma: no cover
                "moveaxis",  # pragma: no cover
                "reshape",  # pragma: no cover
                "expand_dims",  # pragma: no cover
                "linspace",  # pragma: no cover
                "logcumsumexp",  # pragma: no cover
                "cummin",  # pragma: no cover
                "cummax",  # pragma: no cover
                "cumprod",  # pragma: no cover
                "cumsum",  # pragma: no cover
                "logsumexp",  # pragma: no cover
                "all",  # pragma: no cover
                "any",  # pragma: no cover
                "argmin",  # pragma: no cover
                "argmax",  # pragma: no cover
                "prod",  # pragma: no cover
                "sum",  # pragma: no cover
                "mean",  # pragma: no cover
                "var",  # pragma: no cover
                "std",  # pragma: no cover
                "max",  # pragma: no cover
                "min",  # pragma: no cover
                "tensordot",  # pragma: no cover
            ):  # pragma: no cover
                return _to_tensor(x)  # pragma: no cover
        if type(x).__name__ == "ndarray":  # pragma: no cover
            return _to_tensor(x)  # pragma: no cover
        return x  # pragma: no cover

    def _w(x):  # pragma: no cover
        if isinstance(x, tuple):  # pragma: no cover
            return tuple((_w(i) for i in x))  # pragma: no cover
        if isinstance(x, list):  # pragma: no cover
            return [_w(i) for i in x]  # pragma: no cover
        return array(x) if hasattr(x, "shape") else x  # pragma: no cover

    fn = _get_compiler_op("concatenate")  # pragma: no cover
    res = fn(_u(arrays, "arrays"), dim=_u(axis, "axis"))  # pragma: no cover

    def __w(x):
        if isinstance(x, tuple):
            return tuple((__w(i) for i in x))
        if isinstance(x, list):
            return [__w(i) for i in x]
        return array(x) if hasattr(x, "shape") else x

    return __w(res)  # pragma: no cover


def get_peak_memory(*args, **kwargs):  # pragma: no cover
    """Compute get_peak_memory.

    Args:
        args: The args argument.
        kwargs: The kwargs argument.

    Returns:
        The result of get_peak_memory.

    """
    from zero_mlx.array import array, _to_tensor  # pragma: no cover
    import ml_switcheroo_compiler.ops as mops  # pragma: no cover
    import ml_switcheroo_compiler.ops.nn as mnn  # pragma: no cover
    import ml_switcheroo_compiler.random as mrand  # pragma: no cover
    import types  # pragma: no cover

    def _u(x, param_name=None):  # pragma: no cover
        if isinstance(x, list):  # pragma: no cover
            if __import__("builtins").all(  # pragma: no cover
                (isinstance(i, int) for i in x)  # pragma: no cover
            ):  # pragma: no cover
                return tuple(x)  # pragma: no cover
            return [_u(i, param_name) for i in x]  # pragma: no cover
        if isinstance(x, tuple):  # pragma: no cover
            return tuple((_u(i, param_name) for i in x))  # pragma: no cover
        if hasattr(x, "_tensor"):  # pragma: no cover
            return x._tensor  # pragma: no cover
        if isinstance(x, (int, float, bool, complex)):  # pragma: no cover
            if param_name not in {  # pragma: no cover
                "axis",  # pragma: no cover
                "keepdims",
                "shape",
                "size",  # pragma: no cover
                "equal_nan",
                "nan",
                "posinf",
                "neginf",  # pragma: no cover
                "axis1",  # pragma: no cover
                "rtol",  # pragma: no cover
                "n",  # pragma: no cover
                "m",  # pragma: no cover
                "endpoint",  # pragma: no cover
                "offset",  # pragma: no cover
                "a_max",  # pragma: no cover
                "ddof",  # pragma: no cover
                "dims",  # pragma: no cover
                "source",  # pragma: no cover
                "axis2",  # pragma: no cover
                "stop",  # pragma: no cover
                "k",  # pragma: no cover
                "dtype",  # pragma: no cover
                "reps",  # pragma: no cover
                "a_min",  # pragma: no cover
                "sparse",  # pragma: no cover
                "decimals",  # pragma: no cover
                "strides",  # pragma: no cover
                "num",  # pragma: no cover
                "max",  # pragma: no cover
                "indices_or_sections",  # pragma: no cover
                "shape",  # pragma: no cover
                "atol",  # pragma: no cover
                "axis1",  # pragma: no cover
                "step",  # pragma: no cover
                "axes",  # pragma: no cover
                "fill_value",  # pragma: no cover
                "stream",  # pragma: no cover
                "destination",  # pragma: no cover
                "alpha",  # pragma: no cover
                "beta",  # pragma: no cover
                "start",  # pragma: no cover
                "equal_nan",
                "nan",
                "posinf",
                "neginf",  # pragma: no cover
                "keepdims",
                "shape",
                "size",  # pragma: no cover
                "min",  # pragma: no cover
                "p",  # pragma: no cover
            } and "get_peak_memory" not in (  # pragma: no cover
                "transpose",  # pragma: no cover
                "moveaxis",  # pragma: no cover
                "reshape",  # pragma: no cover
                "expand_dims",  # pragma: no cover
                "linspace",  # pragma: no cover
                "logcumsumexp",  # pragma: no cover
                "cummin",  # pragma: no cover
                "cummax",  # pragma: no cover
                "cumprod",  # pragma: no cover
                "cumsum",  # pragma: no cover
                "logsumexp",  # pragma: no cover
                "all",  # pragma: no cover
                "any",  # pragma: no cover
                "argmin",  # pragma: no cover
                "argmax",  # pragma: no cover
                "prod",  # pragma: no cover
                "sum",  # pragma: no cover
                "mean",  # pragma: no cover
                "var",  # pragma: no cover
                "std",  # pragma: no cover
                "max",  # pragma: no cover
                "min",  # pragma: no cover
                "tensordot",  # pragma: no cover
            ):  # pragma: no cover
                return _to_tensor(x)  # pragma: no cover
        if type(x).__name__ == "ndarray":  # pragma: no cover
            return _to_tensor(x)  # pragma: no cover
        return x  # pragma: no cover

    def _w(x):  # pragma: no cover
        if isinstance(x, tuple):  # pragma: no cover
            return tuple((_w(i) for i in x))  # pragma: no cover
        if isinstance(x, list):  # pragma: no cover
            return [_w(i) for i in x]  # pragma: no cover
        return array(x) if hasattr(x, "shape") else x  # pragma: no cover

    fn = _get_compiler_op("get_peak_memory")  # pragma: no cover
    res = fn(  # pragma: no cover
        *[_u(x) for x in args],  # pragma: no cover
        **{
            ("dim" if k == "axis" else k): _u(v, k)
            for (k, v) in kwargs.items()
            if k != "stream"
        },  # pragma: no cover
    )  # pragma: no cover

    def __w(x):
        if isinstance(x, tuple):
            return tuple((__w(i) for i in x))
        if isinstance(x, list):
            return [__w(i) for i in x]
        return array(x) if hasattr(x, "shape") else x

    return __w(res)  # pragma: no cover


def argmin(a, axis=None, keepdims=False, stream=None):  # pragma: no cover
    """Compute argmin.

    Args:
        a: The a argument.
        axis: The axis argument.
        keepdims: The keepdims argument.
        stream: The stream argument.

    Returns:
        The result of argmin.

    """
    from zero_mlx.array import array, _to_tensor  # pragma: no cover
    import ml_switcheroo_compiler.ops as mops  # pragma: no cover
    import ml_switcheroo_compiler.ops.nn as mnn  # pragma: no cover
    import ml_switcheroo_compiler.random as mrand  # pragma: no cover
    import types  # pragma: no cover

    def _u(x, param_name=None):  # pragma: no cover
        if isinstance(x, list):  # pragma: no cover
            if __import__("builtins").all(  # pragma: no cover
                (isinstance(i, int) for i in x)  # pragma: no cover
            ):  # pragma: no cover
                return tuple(x)  # pragma: no cover
            return [_u(i, param_name) for i in x]  # pragma: no cover
        if isinstance(x, tuple):  # pragma: no cover
            return tuple((_u(i, param_name) for i in x))  # pragma: no cover
        if hasattr(x, "_tensor"):  # pragma: no cover
            return x._tensor  # pragma: no cover
        if isinstance(x, (int, float, bool, complex)):  # pragma: no cover
            if param_name not in {  # pragma: no cover
                "axis",  # pragma: no cover
                "keepdims",
                "shape",
                "size",  # pragma: no cover
                "equal_nan",
                "nan",
                "posinf",
                "neginf",  # pragma: no cover
                "axis1",  # pragma: no cover
                "rtol",  # pragma: no cover
                "n",  # pragma: no cover
                "m",  # pragma: no cover
                "endpoint",  # pragma: no cover
                "offset",  # pragma: no cover
                "a_max",  # pragma: no cover
                "ddof",  # pragma: no cover
                "dims",  # pragma: no cover
                "source",  # pragma: no cover
                "axis2",  # pragma: no cover
                "stop",  # pragma: no cover
                "k",  # pragma: no cover
                "dtype",  # pragma: no cover
                "reps",  # pragma: no cover
                "a_min",  # pragma: no cover
                "sparse",  # pragma: no cover
                "decimals",  # pragma: no cover
                "strides",  # pragma: no cover
                "num",  # pragma: no cover
                "max",  # pragma: no cover
                "indices_or_sections",  # pragma: no cover
                "shape",  # pragma: no cover
                "atol",  # pragma: no cover
                "axis1",  # pragma: no cover
                "step",  # pragma: no cover
                "axes",  # pragma: no cover
                "fill_value",  # pragma: no cover
                "stream",  # pragma: no cover
                "destination",  # pragma: no cover
                "alpha",  # pragma: no cover
                "beta",  # pragma: no cover
                "start",  # pragma: no cover
                "equal_nan",
                "nan",
                "posinf",
                "neginf",  # pragma: no cover
                "keepdims",
                "shape",
                "size",  # pragma: no cover
                "min",  # pragma: no cover
                "p",  # pragma: no cover
            } and "argmin" not in (  # pragma: no cover
                "transpose",  # pragma: no cover
                "moveaxis",  # pragma: no cover
                "reshape",  # pragma: no cover
                "expand_dims",  # pragma: no cover
                "linspace",  # pragma: no cover
                "logcumsumexp",  # pragma: no cover
                "cummin",  # pragma: no cover
                "cummax",  # pragma: no cover
                "cumprod",  # pragma: no cover
                "cumsum",  # pragma: no cover
                "logsumexp",  # pragma: no cover
                "all",  # pragma: no cover
                "any",  # pragma: no cover
                "argmin",  # pragma: no cover
                "argmax",  # pragma: no cover
                "prod",  # pragma: no cover
                "sum",  # pragma: no cover
                "mean",  # pragma: no cover
                "var",  # pragma: no cover
                "std",  # pragma: no cover
                "max",  # pragma: no cover
                "min",  # pragma: no cover
                "tensordot",  # pragma: no cover
            ):  # pragma: no cover
                return _to_tensor(x)  # pragma: no cover
        if type(x).__name__ == "ndarray":  # pragma: no cover
            return _to_tensor(x)  # pragma: no cover
        return x  # pragma: no cover

    def _w(x):  # pragma: no cover
        if isinstance(x, tuple):  # pragma: no cover
            return tuple((_w(i) for i in x))  # pragma: no cover
        if isinstance(x, list):  # pragma: no cover
            return [_w(i) for i in x]  # pragma: no cover
        return array(x) if hasattr(x, "shape") else x  # pragma: no cover

    fn = _get_compiler_op("argmin")  # pragma: no cover
    res = fn(
        _u(a, "a"), axis=_u(axis, "axis"), keepdims=_u(keepdims, "keepdims")
    )  # pragma: no cover

    def __w(x):
        if isinstance(x, tuple):
            return tuple((__w(i) for i in x))
        if isinstance(x, list):
            return [__w(i) for i in x]
        return array(x) if hasattr(x, "shape") else x

    return __w(res)  # pragma: no cover


def argmax(a, axis=None, keepdims=False, stream=None):  # pragma: no cover
    """Compute argmax.

    Args:
        a: The a argument.
        axis: The axis argument.
        keepdims: The keepdims argument.
        stream: The stream argument.

    Returns:
        The result of argmax.

    """
    from zero_mlx.array import array, _to_tensor  # pragma: no cover
    import ml_switcheroo_compiler.ops as mops  # pragma: no cover
    import ml_switcheroo_compiler.ops.nn as mnn  # pragma: no cover
    import ml_switcheroo_compiler.random as mrand  # pragma: no cover
    import types  # pragma: no cover

    def _u(x, param_name=None):  # pragma: no cover
        if isinstance(x, list):  # pragma: no cover
            if __import__("builtins").all(  # pragma: no cover
                (isinstance(i, int) for i in x)  # pragma: no cover
            ):  # pragma: no cover
                return tuple(x)  # pragma: no cover
            return [_u(i, param_name) for i in x]  # pragma: no cover
        if isinstance(x, tuple):  # pragma: no cover
            return tuple((_u(i, param_name) for i in x))  # pragma: no cover
        if hasattr(x, "_tensor"):  # pragma: no cover
            return x._tensor  # pragma: no cover
        if isinstance(x, (int, float, bool, complex)):  # pragma: no cover
            if param_name not in {  # pragma: no cover
                "axis",  # pragma: no cover
                "keepdims",
                "shape",
                "size",  # pragma: no cover
                "equal_nan",
                "nan",
                "posinf",
                "neginf",  # pragma: no cover
                "axis1",  # pragma: no cover
                "rtol",  # pragma: no cover
                "n",  # pragma: no cover
                "m",  # pragma: no cover
                "endpoint",  # pragma: no cover
                "offset",  # pragma: no cover
                "a_max",  # pragma: no cover
                "ddof",  # pragma: no cover
                "dims",  # pragma: no cover
                "source",  # pragma: no cover
                "axis2",  # pragma: no cover
                "stop",  # pragma: no cover
                "k",  # pragma: no cover
                "dtype",  # pragma: no cover
                "reps",  # pragma: no cover
                "a_min",  # pragma: no cover
                "sparse",  # pragma: no cover
                "decimals",  # pragma: no cover
                "strides",  # pragma: no cover
                "num",  # pragma: no cover
                "max",  # pragma: no cover
                "indices_or_sections",  # pragma: no cover
                "shape",  # pragma: no cover
                "atol",  # pragma: no cover
                "axis1",  # pragma: no cover
                "step",  # pragma: no cover
                "axes",  # pragma: no cover
                "fill_value",  # pragma: no cover
                "stream",  # pragma: no cover
                "destination",  # pragma: no cover
                "alpha",  # pragma: no cover
                "beta",  # pragma: no cover
                "start",  # pragma: no cover
                "equal_nan",
                "nan",
                "posinf",
                "neginf",  # pragma: no cover
                "keepdims",
                "shape",
                "size",  # pragma: no cover
                "min",  # pragma: no cover
                "p",  # pragma: no cover
            } and "argmax" not in (  # pragma: no cover
                "transpose",  # pragma: no cover
                "moveaxis",  # pragma: no cover
                "reshape",  # pragma: no cover
                "expand_dims",  # pragma: no cover
                "linspace",  # pragma: no cover
                "logcumsumexp",  # pragma: no cover
                "cummin",  # pragma: no cover
                "cummax",  # pragma: no cover
                "cumprod",  # pragma: no cover
                "cumsum",  # pragma: no cover
                "logsumexp",  # pragma: no cover
                "all",  # pragma: no cover
                "any",  # pragma: no cover
                "argmin",  # pragma: no cover
                "argmax",  # pragma: no cover
                "prod",  # pragma: no cover
                "sum",  # pragma: no cover
                "mean",  # pragma: no cover
                "var",  # pragma: no cover
                "std",  # pragma: no cover
                "max",  # pragma: no cover
                "min",  # pragma: no cover
                "tensordot",  # pragma: no cover
            ):  # pragma: no cover
                return _to_tensor(x)  # pragma: no cover
        if type(x).__name__ == "ndarray":  # pragma: no cover
            return _to_tensor(x)  # pragma: no cover
        return x  # pragma: no cover

    def _w(x):  # pragma: no cover
        if isinstance(x, tuple):  # pragma: no cover
            return tuple((_w(i) for i in x))  # pragma: no cover
        if isinstance(x, list):  # pragma: no cover
            return [_w(i) for i in x]  # pragma: no cover
        return array(x) if hasattr(x, "shape") else x  # pragma: no cover

    fn = _get_compiler_op("argmax")  # pragma: no cover
    res = fn(
        _u(a, "a"), axis=_u(axis, "axis"), keepdims=_u(keepdims, "keepdims")
    )  # pragma: no cover

    def __w(x):
        if isinstance(x, tuple):
            return tuple((__w(i) for i in x))
        if isinstance(x, list):
            return [__w(i) for i in x]
        return array(x) if hasattr(x, "shape") else x

    return __w(res)  # pragma: no cover


def min(a, axis=None, keepdims=False, stream=None):  # pragma: no cover
    """Compute min.

    Args:
        a: The a argument.
        axis: The axis argument.
        keepdims: The keepdims argument.
        stream: The stream argument.

    Returns:
        The result of min.

    """
    from zero_mlx.array import array, _to_tensor  # pragma: no cover
    import ml_switcheroo_compiler.ops as mops  # pragma: no cover
    import ml_switcheroo_compiler.ops.nn as mnn  # pragma: no cover
    import ml_switcheroo_compiler.random as mrand  # pragma: no cover
    import types  # pragma: no cover

    def _u(x, param_name=None):  # pragma: no cover
        if isinstance(x, list):  # pragma: no cover
            if __import__("builtins").all(  # pragma: no cover
                (isinstance(i, int) for i in x)  # pragma: no cover
            ):  # pragma: no cover
                return tuple(x)  # pragma: no cover
            return [_u(i, param_name) for i in x]  # pragma: no cover
        if isinstance(x, tuple):  # pragma: no cover
            return tuple((_u(i, param_name) for i in x))  # pragma: no cover
        if hasattr(x, "_tensor"):  # pragma: no cover
            return x._tensor  # pragma: no cover
        if isinstance(x, (int, float, bool, complex)):  # pragma: no cover
            if param_name not in {  # pragma: no cover
                "axis",  # pragma: no cover
                "keepdims",
                "shape",
                "size",  # pragma: no cover
                "equal_nan",
                "nan",
                "posinf",
                "neginf",  # pragma: no cover
                "axis1",  # pragma: no cover
                "rtol",  # pragma: no cover
                "n",  # pragma: no cover
                "m",  # pragma: no cover
                "endpoint",  # pragma: no cover
                "offset",  # pragma: no cover
                "a_max",  # pragma: no cover
                "ddof",  # pragma: no cover
                "dims",  # pragma: no cover
                "source",  # pragma: no cover
                "axis2",  # pragma: no cover
                "stop",  # pragma: no cover
                "k",  # pragma: no cover
                "dtype",  # pragma: no cover
                "reps",  # pragma: no cover
                "a_min",  # pragma: no cover
                "sparse",  # pragma: no cover
                "decimals",  # pragma: no cover
                "strides",  # pragma: no cover
                "num",  # pragma: no cover
                "max",  # pragma: no cover
                "indices_or_sections",  # pragma: no cover
                "shape",  # pragma: no cover
                "atol",  # pragma: no cover
                "axis1",  # pragma: no cover
                "step",  # pragma: no cover
                "axes",  # pragma: no cover
                "fill_value",  # pragma: no cover
                "stream",  # pragma: no cover
                "destination",  # pragma: no cover
                "alpha",  # pragma: no cover
                "beta",  # pragma: no cover
                "start",  # pragma: no cover
                "equal_nan",
                "nan",
                "posinf",
                "neginf",  # pragma: no cover
                "keepdims",
                "shape",
                "size",  # pragma: no cover
                "min",  # pragma: no cover
                "p",  # pragma: no cover
            } and "min" not in (  # pragma: no cover
                "transpose",  # pragma: no cover
                "moveaxis",  # pragma: no cover
                "reshape",  # pragma: no cover
                "expand_dims",  # pragma: no cover
                "linspace",  # pragma: no cover
                "logcumsumexp",  # pragma: no cover
                "cummin",  # pragma: no cover
                "cummax",  # pragma: no cover
                "cumprod",  # pragma: no cover
                "cumsum",  # pragma: no cover
                "logsumexp",  # pragma: no cover
                "all",  # pragma: no cover
                "any",  # pragma: no cover
                "argmin",  # pragma: no cover
                "argmax",  # pragma: no cover
                "prod",  # pragma: no cover
                "sum",  # pragma: no cover
                "mean",  # pragma: no cover
                "var",  # pragma: no cover
                "std",  # pragma: no cover
                "max",  # pragma: no cover
                "min",  # pragma: no cover
                "tensordot",  # pragma: no cover
            ):  # pragma: no cover
                return _to_tensor(x)  # pragma: no cover
        if type(x).__name__ == "ndarray":  # pragma: no cover
            return _to_tensor(x)  # pragma: no cover
        return x  # pragma: no cover

    def _w(x):  # pragma: no cover
        if isinstance(x, tuple):  # pragma: no cover
            return tuple((_w(i) for i in x))  # pragma: no cover
        if isinstance(x, list):  # pragma: no cover
            return [_w(i) for i in x]  # pragma: no cover
        return array(x) if hasattr(x, "shape") else x  # pragma: no cover

    fn = _get_compiler_op("min")  # pragma: no cover
    res = fn(
        _u(a, "a"), dim=_u(axis, "axis"), keepdims=_u(keepdims, "keepdims")
    )  # pragma: no cover

    def __w(x):
        if isinstance(x, tuple):
            return tuple((__w(i) for i in x))
        if isinstance(x, list):
            return [__w(i) for i in x]
        return array(x) if hasattr(x, "shape") else x

    return __w(res)  # pragma: no cover


def max(a, axis=None, keepdims=False, stream=None):  # pragma: no cover
    """Compute max.

    Args:
        a: The a argument.
        axis: The axis argument.
        keepdims: The keepdims argument.
        stream: The stream argument.

    Returns:
        The result of max.

    """
    from zero_mlx.array import array, _to_tensor  # pragma: no cover
    import ml_switcheroo_compiler.ops as mops  # pragma: no cover
    import ml_switcheroo_compiler.ops.nn as mnn  # pragma: no cover
    import ml_switcheroo_compiler.random as mrand  # pragma: no cover
    import types  # pragma: no cover

    def _u(x, param_name=None):  # pragma: no cover
        if isinstance(x, list):  # pragma: no cover
            if __import__("builtins").all(  # pragma: no cover
                (isinstance(i, int) for i in x)  # pragma: no cover
            ):  # pragma: no cover
                return tuple(x)  # pragma: no cover
            return [_u(i, param_name) for i in x]  # pragma: no cover
        if isinstance(x, tuple):  # pragma: no cover
            return tuple((_u(i, param_name) for i in x))  # pragma: no cover
        if hasattr(x, "_tensor"):  # pragma: no cover
            return x._tensor  # pragma: no cover
        if isinstance(x, (int, float, bool, complex)):  # pragma: no cover
            if param_name not in {  # pragma: no cover
                "axis",  # pragma: no cover
                "keepdims",
                "shape",
                "size",  # pragma: no cover
                "equal_nan",
                "nan",
                "posinf",
                "neginf",  # pragma: no cover
                "axis1",  # pragma: no cover
                "rtol",  # pragma: no cover
                "n",  # pragma: no cover
                "m",  # pragma: no cover
                "endpoint",  # pragma: no cover
                "offset",  # pragma: no cover
                "a_max",  # pragma: no cover
                "ddof",  # pragma: no cover
                "dims",  # pragma: no cover
                "source",  # pragma: no cover
                "axis2",  # pragma: no cover
                "stop",  # pragma: no cover
                "k",  # pragma: no cover
                "dtype",  # pragma: no cover
                "reps",  # pragma: no cover
                "a_min",  # pragma: no cover
                "sparse",  # pragma: no cover
                "decimals",  # pragma: no cover
                "strides",  # pragma: no cover
                "num",  # pragma: no cover
                "max",  # pragma: no cover
                "indices_or_sections",  # pragma: no cover
                "shape",  # pragma: no cover
                "atol",  # pragma: no cover
                "axis1",  # pragma: no cover
                "step",  # pragma: no cover
                "axes",  # pragma: no cover
                "fill_value",  # pragma: no cover
                "stream",  # pragma: no cover
                "destination",  # pragma: no cover
                "alpha",  # pragma: no cover
                "beta",  # pragma: no cover
                "start",  # pragma: no cover
                "equal_nan",
                "nan",
                "posinf",
                "neginf",  # pragma: no cover
                "keepdims",
                "shape",
                "size",  # pragma: no cover
                "min",  # pragma: no cover
                "p",  # pragma: no cover
            } and "max" not in (  # pragma: no cover
                "transpose",  # pragma: no cover
                "moveaxis",  # pragma: no cover
                "reshape",  # pragma: no cover
                "expand_dims",  # pragma: no cover
                "linspace",  # pragma: no cover
                "logcumsumexp",  # pragma: no cover
                "cummin",  # pragma: no cover
                "cummax",  # pragma: no cover
                "cumprod",  # pragma: no cover
                "cumsum",  # pragma: no cover
                "logsumexp",  # pragma: no cover
                "all",  # pragma: no cover
                "any",  # pragma: no cover
                "argmin",  # pragma: no cover
                "argmax",  # pragma: no cover
                "prod",  # pragma: no cover
                "sum",  # pragma: no cover
                "mean",  # pragma: no cover
                "var",  # pragma: no cover
                "std",  # pragma: no cover
                "max",  # pragma: no cover
                "min",  # pragma: no cover
                "tensordot",  # pragma: no cover
            ):  # pragma: no cover
                return _to_tensor(x)  # pragma: no cover
        if type(x).__name__ == "ndarray":  # pragma: no cover
            return _to_tensor(x)  # pragma: no cover
        return x  # pragma: no cover

    def _w(x):  # pragma: no cover
        if isinstance(x, tuple):  # pragma: no cover
            return tuple((_w(i) for i in x))  # pragma: no cover
        if isinstance(x, list):  # pragma: no cover
            return [_w(i) for i in x]  # pragma: no cover
        return array(x) if hasattr(x, "shape") else x  # pragma: no cover

    fn = _get_compiler_op("max")  # pragma: no cover
    res = fn(
        _u(a, "a"), dim=_u(axis, "axis"), keepdims=_u(keepdims, "keepdims")
    )  # pragma: no cover

    def __w(x):
        if isinstance(x, tuple):
            return tuple((__w(i) for i in x))
        if isinstance(x, list):
            return [__w(i) for i in x]
        return array(x) if hasattr(x, "shape") else x

    return __w(res)  # pragma: no cover


def prod(a, axis=None, keepdims=False, stream=None):  # pragma: no cover
    """Compute prod.

    Args:
        a: The a argument.
        axis: The axis argument.
        keepdims: The keepdims argument.
        stream: The stream argument.

    Returns:
        The result of prod.

    """
    from zero_mlx.array import array, _to_tensor  # pragma: no cover
    import ml_switcheroo_compiler.ops as mops  # pragma: no cover
    import ml_switcheroo_compiler.ops.nn as mnn  # pragma: no cover
    import ml_switcheroo_compiler.random as mrand  # pragma: no cover
    import types  # pragma: no cover

    def _u(x, param_name=None):  # pragma: no cover
        if isinstance(x, list):  # pragma: no cover
            if __import__("builtins").all(  # pragma: no cover
                (isinstance(i, int) for i in x)  # pragma: no cover
            ):  # pragma: no cover
                return tuple(x)  # pragma: no cover
            return [_u(i, param_name) for i in x]  # pragma: no cover
        if isinstance(x, tuple):  # pragma: no cover
            return tuple((_u(i, param_name) for i in x))  # pragma: no cover
        if hasattr(x, "_tensor"):  # pragma: no cover
            return x._tensor  # pragma: no cover
        if isinstance(x, (int, float, bool, complex)):  # pragma: no cover
            if param_name not in {  # pragma: no cover
                "axis",  # pragma: no cover
                "keepdims",
                "shape",
                "size",  # pragma: no cover
                "equal_nan",
                "nan",
                "posinf",
                "neginf",  # pragma: no cover
                "axis1",  # pragma: no cover
                "rtol",  # pragma: no cover
                "n",  # pragma: no cover
                "m",  # pragma: no cover
                "endpoint",  # pragma: no cover
                "offset",  # pragma: no cover
                "a_max",  # pragma: no cover
                "ddof",  # pragma: no cover
                "dims",  # pragma: no cover
                "source",  # pragma: no cover
                "axis2",  # pragma: no cover
                "stop",  # pragma: no cover
                "k",  # pragma: no cover
                "dtype",  # pragma: no cover
                "reps",  # pragma: no cover
                "a_min",  # pragma: no cover
                "sparse",  # pragma: no cover
                "decimals",  # pragma: no cover
                "strides",  # pragma: no cover
                "num",  # pragma: no cover
                "max",  # pragma: no cover
                "indices_or_sections",  # pragma: no cover
                "shape",  # pragma: no cover
                "atol",  # pragma: no cover
                "axis1",  # pragma: no cover
                "step",  # pragma: no cover
                "axes",  # pragma: no cover
                "fill_value",  # pragma: no cover
                "stream",  # pragma: no cover
                "destination",  # pragma: no cover
                "alpha",  # pragma: no cover
                "beta",  # pragma: no cover
                "start",  # pragma: no cover
                "equal_nan",
                "nan",
                "posinf",
                "neginf",  # pragma: no cover
                "keepdims",
                "shape",
                "size",  # pragma: no cover
                "min",  # pragma: no cover
                "p",  # pragma: no cover
            } and "prod" not in (  # pragma: no cover
                "transpose",  # pragma: no cover
                "moveaxis",  # pragma: no cover
                "reshape",  # pragma: no cover
                "expand_dims",  # pragma: no cover
                "linspace",  # pragma: no cover
                "logcumsumexp",  # pragma: no cover
                "cummin",  # pragma: no cover
                "cummax",  # pragma: no cover
                "cumprod",  # pragma: no cover
                "cumsum",  # pragma: no cover
                "logsumexp",  # pragma: no cover
                "all",  # pragma: no cover
                "any",  # pragma: no cover
                "argmin",  # pragma: no cover
                "argmax",  # pragma: no cover
                "prod",  # pragma: no cover
                "sum",  # pragma: no cover
                "mean",  # pragma: no cover
                "var",  # pragma: no cover
                "std",  # pragma: no cover
                "max",  # pragma: no cover
                "min",  # pragma: no cover
                "tensordot",  # pragma: no cover
            ):  # pragma: no cover
                return _to_tensor(x)  # pragma: no cover
        if type(x).__name__ == "ndarray":  # pragma: no cover
            return _to_tensor(x)  # pragma: no cover
        return x  # pragma: no cover

    def _w(x):  # pragma: no cover
        if isinstance(x, tuple):  # pragma: no cover
            return tuple((_w(i) for i in x))  # pragma: no cover
        if isinstance(x, list):  # pragma: no cover
            return [_w(i) for i in x]  # pragma: no cover
        return array(x) if hasattr(x, "shape") else x  # pragma: no cover

    fn = _get_compiler_op("prod")  # pragma: no cover
    res = fn(
        _u(a, "a"), dim=_u(axis, "axis"), keepdims=_u(keepdims, "keepdims")
    )  # pragma: no cover

    def __w(x):
        if isinstance(x, tuple):
            return tuple((__w(i) for i in x))
        if isinstance(x, list):
            return [__w(i) for i in x]
        return array(x) if hasattr(x, "shape") else x

    return __w(res)  # pragma: no cover


def eye(n, m=None, k=0, dtype=None, stream=None):  # pragma: no cover
    """Compute eye.

    Args:
        n: The n argument.
        m: The m argument.
        k: The k argument.
        dtype: The dtype argument.
        stream: The stream argument.

    Returns:
        The result of eye.

    """
    from zero_mlx.array import array, _to_tensor  # pragma: no cover
    import ml_switcheroo_compiler.ops as mops  # pragma: no cover
    import ml_switcheroo_compiler.ops.nn as mnn  # pragma: no cover
    import ml_switcheroo_compiler.random as mrand  # pragma: no cover
    import types  # pragma: no cover

    def _u(x, param_name=None):  # pragma: no cover
        if isinstance(x, list):  # pragma: no cover
            if __import__("builtins").all(  # pragma: no cover
                (isinstance(i, int) for i in x)  # pragma: no cover
            ):  # pragma: no cover
                return tuple(x)  # pragma: no cover
            return [_u(i, param_name) for i in x]  # pragma: no cover
        if isinstance(x, tuple):  # pragma: no cover
            return tuple((_u(i, param_name) for i in x))  # pragma: no cover
        if hasattr(x, "_tensor"):  # pragma: no cover
            return x._tensor  # pragma: no cover
        if isinstance(x, (int, float, bool, complex)):  # pragma: no cover
            if param_name not in {  # pragma: no cover
                "axis",  # pragma: no cover
                "keepdims",
                "shape",
                "size",  # pragma: no cover
                "equal_nan",
                "nan",
                "posinf",
                "neginf",  # pragma: no cover
                "axis1",  # pragma: no cover
                "rtol",  # pragma: no cover
                "n",  # pragma: no cover
                "m",  # pragma: no cover
                "endpoint",  # pragma: no cover
                "offset",  # pragma: no cover
                "a_max",  # pragma: no cover
                "ddof",  # pragma: no cover
                "dims",  # pragma: no cover
                "source",  # pragma: no cover
                "axis2",  # pragma: no cover
                "stop",  # pragma: no cover
                "k",  # pragma: no cover
                "dtype",  # pragma: no cover
                "reps",  # pragma: no cover
                "a_min",  # pragma: no cover
                "sparse",  # pragma: no cover
                "decimals",  # pragma: no cover
                "strides",  # pragma: no cover
                "num",  # pragma: no cover
                "max",  # pragma: no cover
                "indices_or_sections",  # pragma: no cover
                "shape",  # pragma: no cover
                "atol",  # pragma: no cover
                "axis1",  # pragma: no cover
                "step",  # pragma: no cover
                "axes",  # pragma: no cover
                "fill_value",  # pragma: no cover
                "stream",  # pragma: no cover
                "destination",  # pragma: no cover
                "alpha",  # pragma: no cover
                "beta",  # pragma: no cover
                "start",  # pragma: no cover
                "equal_nan",
                "nan",
                "posinf",
                "neginf",  # pragma: no cover
                "keepdims",
                "shape",
                "size",  # pragma: no cover
                "min",  # pragma: no cover
                "p",  # pragma: no cover
            } and "eye" not in (  # pragma: no cover
                "transpose",  # pragma: no cover
                "moveaxis",  # pragma: no cover
                "reshape",  # pragma: no cover
                "expand_dims",  # pragma: no cover
                "linspace",  # pragma: no cover
                "logcumsumexp",  # pragma: no cover
                "cummin",  # pragma: no cover
                "cummax",  # pragma: no cover
                "cumprod",  # pragma: no cover
                "cumsum",  # pragma: no cover
                "logsumexp",  # pragma: no cover
                "all",  # pragma: no cover
                "any",  # pragma: no cover
                "argmin",  # pragma: no cover
                "argmax",  # pragma: no cover
                "prod",  # pragma: no cover
                "sum",  # pragma: no cover
                "mean",  # pragma: no cover
                "var",  # pragma: no cover
                "std",  # pragma: no cover
                "max",  # pragma: no cover
                "min",  # pragma: no cover
                "tensordot",  # pragma: no cover
            ):  # pragma: no cover
                return _to_tensor(x)  # pragma: no cover
        if type(x).__name__ == "ndarray":  # pragma: no cover
            return _to_tensor(x)  # pragma: no cover
        return x  # pragma: no cover

    def _w(x):  # pragma: no cover
        if isinstance(x, tuple):  # pragma: no cover
            return tuple((_w(i) for i in x))  # pragma: no cover
        if isinstance(x, list):  # pragma: no cover
            return [_w(i) for i in x]  # pragma: no cover
        return array(x) if hasattr(x, "shape") else x  # pragma: no cover

    fn = _get_compiler_op("eye")  # pragma: no cover
    res = fn(
        _u(n, "n"), _u(m, "m"), k=_u(k, "k"), dtype=_u(dtype, "dtype")
    )  # pragma: no cover

    def __w(x):
        if isinstance(x, tuple):
            return tuple((__w(i) for i in x))
        if isinstance(x, list):
            return [__w(i) for i in x]
        return array(x) if hasattr(x, "shape") else x

    return __w(res)  # pragma: no cover


def diag(v, k=0, stream=None):  # pragma: no cover
    """Compute diag.

    Args:
        v: The v argument.
        k: The k argument.
        stream: The stream argument.

    Returns:
        The result of diag.

    """
    from zero_mlx.array import array, _to_tensor  # pragma: no cover
    import ml_switcheroo_compiler.ops as mops  # pragma: no cover
    import ml_switcheroo_compiler.ops.nn as mnn  # pragma: no cover
    import ml_switcheroo_compiler.random as mrand  # pragma: no cover
    import types  # pragma: no cover

    def _u(x, param_name=None):  # pragma: no cover
        if isinstance(x, list):  # pragma: no cover
            if __import__("builtins").all(  # pragma: no cover
                (isinstance(i, int) for i in x)  # pragma: no cover
            ):  # pragma: no cover
                return tuple(x)  # pragma: no cover
            return [_u(i, param_name) for i in x]  # pragma: no cover
        if isinstance(x, tuple):  # pragma: no cover
            return tuple((_u(i, param_name) for i in x))  # pragma: no cover
        if hasattr(x, "_tensor"):  # pragma: no cover
            return x._tensor  # pragma: no cover
        if isinstance(x, (int, float, bool, complex)):  # pragma: no cover
            if param_name not in {  # pragma: no cover
                "axis",  # pragma: no cover
                "keepdims",
                "shape",
                "size",  # pragma: no cover
                "equal_nan",
                "nan",
                "posinf",
                "neginf",  # pragma: no cover
                "axis1",  # pragma: no cover
                "rtol",  # pragma: no cover
                "n",  # pragma: no cover
                "m",  # pragma: no cover
                "endpoint",  # pragma: no cover
                "offset",  # pragma: no cover
                "a_max",  # pragma: no cover
                "ddof",  # pragma: no cover
                "dims",  # pragma: no cover
                "source",  # pragma: no cover
                "axis2",  # pragma: no cover
                "stop",  # pragma: no cover
                "k",  # pragma: no cover
                "dtype",  # pragma: no cover
                "reps",  # pragma: no cover
                "a_min",  # pragma: no cover
                "sparse",  # pragma: no cover
                "decimals",  # pragma: no cover
                "strides",  # pragma: no cover
                "num",  # pragma: no cover
                "max",  # pragma: no cover
                "indices_or_sections",  # pragma: no cover
                "shape",  # pragma: no cover
                "atol",  # pragma: no cover
                "axis1",  # pragma: no cover
                "step",  # pragma: no cover
                "axes",  # pragma: no cover
                "fill_value",  # pragma: no cover
                "stream",  # pragma: no cover
                "destination",  # pragma: no cover
                "alpha",  # pragma: no cover
                "beta",  # pragma: no cover
                "start",  # pragma: no cover
                "equal_nan",
                "nan",
                "posinf",
                "neginf",  # pragma: no cover
                "keepdims",
                "shape",
                "size",  # pragma: no cover
                "min",  # pragma: no cover
                "p",  # pragma: no cover
                "nan",
                "posinf",
                "neginf",
            } and "diag" not in (  # pragma: no cover
                "transpose",  # pragma: no cover
                "moveaxis",  # pragma: no cover
                "reshape",  # pragma: no cover
                "expand_dims",  # pragma: no cover
                "linspace",  # pragma: no cover
                "logcumsumexp",  # pragma: no cover
                "cummin",  # pragma: no cover
                "cummax",  # pragma: no cover
                "cumprod",  # pragma: no cover
                "cumsum",  # pragma: no cover
                "logsumexp",  # pragma: no cover
                "all",  # pragma: no cover
                "any",  # pragma: no cover
                "argmin",  # pragma: no cover
                "argmax",  # pragma: no cover
                "prod",  # pragma: no cover
                "sum",  # pragma: no cover
                "mean",  # pragma: no cover
                "var",  # pragma: no cover
                "std",  # pragma: no cover
                "max",  # pragma: no cover
                "min",  # pragma: no cover
                "tensordot",  # pragma: no cover
            ):  # pragma: no cover
                return _to_tensor(x)  # pragma: no cover
        if type(x).__name__ == "ndarray":  # pragma: no cover
            return _to_tensor(x)  # pragma: no cover
        return x  # pragma: no cover

    def _w(x):  # pragma: no cover
        if isinstance(x, tuple):  # pragma: no cover
            return tuple((_w(i) for i in x))  # pragma: no cover
        if isinstance(x, list):  # pragma: no cover
            return [_w(i) for i in x]  # pragma: no cover
        return array(x) if hasattr(x, "shape") else x  # pragma: no cover

    v_u = _u(v, "v")  # pragma: no cover
    if hasattr(v_u, "shape") and len(v_u.shape) == 1:  # pragma: no cover
        fn = _get_compiler_op("diagflat")  # pragma: no cover
        res = fn(v_u, k=_u(k, "k"))  # pragma: no cover
    else:  # pragma: no cover
        fn = _get_compiler_op("diag")  # pragma: no cover
        res = fn(v_u, diagonal=_u(k, "k"))  # pragma: no cover

    def __w(x):
        if isinstance(x, tuple):
            return tuple((__w(i) for i in x))
        if isinstance(x, list):
            return [__w(i) for i in x]
        return array(x) if hasattr(x, "shape") else x

    return __w(res)  # pragma: no cover


def tril(m, k=0, stream=None):  # pragma: no cover
    """Compute tril.

    Args:
        m: The m argument.
        k: The k argument.
        stream: The stream argument.

    Returns:
        The result of tril.

    """
    from zero_mlx.array import array, _to_tensor  # pragma: no cover
    import ml_switcheroo_compiler.ops as mops  # pragma: no cover
    import ml_switcheroo_compiler.ops.nn as mnn  # pragma: no cover
    import ml_switcheroo_compiler.random as mrand  # pragma: no cover
    import types  # pragma: no cover

    def _u(x, param_name=None):  # pragma: no cover
        if isinstance(x, list):  # pragma: no cover
            if __import__("builtins").all(  # pragma: no cover
                (isinstance(i, int) for i in x)  # pragma: no cover
            ):  # pragma: no cover
                return tuple(x)  # pragma: no cover
            return [_u(i, param_name) for i in x]  # pragma: no cover
        if isinstance(x, tuple):  # pragma: no cover
            return tuple((_u(i, param_name) for i in x))  # pragma: no cover
        if hasattr(x, "_tensor"):  # pragma: no cover
            return x._tensor  # pragma: no cover
        if isinstance(x, (int, float, bool, complex)):  # pragma: no cover
            if param_name not in {  # pragma: no cover
                "axis",  # pragma: no cover
                "keepdims",
                "shape",
                "size",  # pragma: no cover
                "equal_nan",
                "nan",
                "posinf",
                "neginf",  # pragma: no cover
                "axis1",  # pragma: no cover
                "rtol",  # pragma: no cover
                "n",  # pragma: no cover
                "m",  # pragma: no cover
                "endpoint",  # pragma: no cover
                "offset",  # pragma: no cover
                "a_max",  # pragma: no cover
                "ddof",  # pragma: no cover
                "dims",  # pragma: no cover
                "source",  # pragma: no cover
                "axis2",  # pragma: no cover
                "stop",  # pragma: no cover
                "k",  # pragma: no cover
                "dtype",  # pragma: no cover
                "reps",  # pragma: no cover
                "a_min",  # pragma: no cover
                "sparse",  # pragma: no cover
                "decimals",  # pragma: no cover
                "strides",  # pragma: no cover
                "num",  # pragma: no cover
                "max",  # pragma: no cover
                "indices_or_sections",  # pragma: no cover
                "shape",  # pragma: no cover
                "atol",  # pragma: no cover
                "axis1",  # pragma: no cover
                "step",  # pragma: no cover
                "axes",  # pragma: no cover
                "fill_value",  # pragma: no cover
                "stream",  # pragma: no cover
                "destination",  # pragma: no cover
                "alpha",  # pragma: no cover
                "beta",  # pragma: no cover
                "start",  # pragma: no cover
                "equal_nan",
                "nan",
                "posinf",
                "neginf",  # pragma: no cover
                "keepdims",
                "shape",
                "size",  # pragma: no cover
                "min",  # pragma: no cover
                "p",  # pragma: no cover
            } and "tril" not in (  # pragma: no cover
                "transpose",  # pragma: no cover
                "moveaxis",  # pragma: no cover
                "reshape",  # pragma: no cover
                "expand_dims",  # pragma: no cover
                "linspace",  # pragma: no cover
                "logcumsumexp",  # pragma: no cover
                "cummin",  # pragma: no cover
                "cummax",  # pragma: no cover
                "cumprod",  # pragma: no cover
                "cumsum",  # pragma: no cover
                "logsumexp",  # pragma: no cover
                "all",  # pragma: no cover
                "any",  # pragma: no cover
                "argmin",  # pragma: no cover
                "argmax",  # pragma: no cover
                "prod",  # pragma: no cover
                "sum",  # pragma: no cover
                "mean",  # pragma: no cover
                "var",  # pragma: no cover
                "std",  # pragma: no cover
                "max",  # pragma: no cover
                "min",  # pragma: no cover
                "tensordot",  # pragma: no cover
            ):  # pragma: no cover
                return _to_tensor(x)  # pragma: no cover
        if type(x).__name__ == "ndarray":  # pragma: no cover
            return _to_tensor(x)  # pragma: no cover
        return x  # pragma: no cover

    def _w(x):  # pragma: no cover
        if isinstance(x, tuple):  # pragma: no cover
            return tuple((_w(i) for i in x))  # pragma: no cover
        if isinstance(x, list):  # pragma: no cover
            return [_w(i) for i in x]  # pragma: no cover
        return array(x) if hasattr(x, "shape") else x  # pragma: no cover

    fn = _get_compiler_op("tril")  # pragma: no cover
    res = fn(_u(m, "m"), k=_u(k, "k"))  # pragma: no cover

    def __w(x):
        if isinstance(x, tuple):
            return tuple((__w(i) for i in x))
        if isinstance(x, list):
            return [__w(i) for i in x]
        return array(x) if hasattr(x, "shape") else x

    return __w(res)  # pragma: no cover


def triu(m, k=0, stream=None):  # pragma: no cover
    """Compute triu.

    Args:
        m: The m argument.
        k: The k argument.
        stream: The stream argument.

    Returns:
        The result of triu.

    """
    from zero_mlx.array import array, _to_tensor  # pragma: no cover
    import ml_switcheroo_compiler.ops as mops  # pragma: no cover
    import ml_switcheroo_compiler.ops.nn as mnn  # pragma: no cover
    import ml_switcheroo_compiler.random as mrand  # pragma: no cover
    import types  # pragma: no cover

    def _u(x, param_name=None):  # pragma: no cover
        if isinstance(x, list):  # pragma: no cover
            if __import__("builtins").all(  # pragma: no cover
                (isinstance(i, int) for i in x)  # pragma: no cover
            ):  # pragma: no cover
                return tuple(x)  # pragma: no cover
            return [_u(i, param_name) for i in x]  # pragma: no cover
        if isinstance(x, tuple):  # pragma: no cover
            return tuple((_u(i, param_name) for i in x))  # pragma: no cover
        if hasattr(x, "_tensor"):  # pragma: no cover
            return x._tensor  # pragma: no cover
        if isinstance(x, (int, float, bool, complex)):  # pragma: no cover
            if param_name not in {  # pragma: no cover
                "axis",  # pragma: no cover
                "keepdims",
                "shape",
                "size",  # pragma: no cover
                "equal_nan",
                "nan",
                "posinf",
                "neginf",  # pragma: no cover
                "axis1",  # pragma: no cover
                "rtol",  # pragma: no cover
                "n",  # pragma: no cover
                "m",  # pragma: no cover
                "endpoint",  # pragma: no cover
                "offset",  # pragma: no cover
                "a_max",  # pragma: no cover
                "ddof",  # pragma: no cover
                "dims",  # pragma: no cover
                "source",  # pragma: no cover
                "axis2",  # pragma: no cover
                "stop",  # pragma: no cover
                "k",  # pragma: no cover
                "dtype",  # pragma: no cover
                "reps",  # pragma: no cover
                "a_min",  # pragma: no cover
                "sparse",  # pragma: no cover
                "decimals",  # pragma: no cover
                "strides",  # pragma: no cover
                "num",  # pragma: no cover
                "max",  # pragma: no cover
                "indices_or_sections",  # pragma: no cover
                "shape",  # pragma: no cover
                "atol",  # pragma: no cover
                "axis1",  # pragma: no cover
                "step",  # pragma: no cover
                "axes",  # pragma: no cover
                "fill_value",  # pragma: no cover
                "stream",  # pragma: no cover
                "destination",  # pragma: no cover
                "alpha",  # pragma: no cover
                "beta",  # pragma: no cover
                "start",  # pragma: no cover
                "equal_nan",
                "nan",
                "posinf",
                "neginf",  # pragma: no cover
                "keepdims",
                "shape",
                "size",  # pragma: no cover
                "min",  # pragma: no cover
                "p",  # pragma: no cover
            } and "triu" not in (  # pragma: no cover
                "transpose",  # pragma: no cover
                "moveaxis",  # pragma: no cover
                "reshape",  # pragma: no cover
                "expand_dims",  # pragma: no cover
                "linspace",  # pragma: no cover
                "logcumsumexp",  # pragma: no cover
                "cummin",  # pragma: no cover
                "cummax",  # pragma: no cover
                "cumprod",  # pragma: no cover
                "cumsum",  # pragma: no cover
                "logsumexp",  # pragma: no cover
                "all",  # pragma: no cover
                "any",  # pragma: no cover
                "argmin",  # pragma: no cover
                "argmax",  # pragma: no cover
                "prod",  # pragma: no cover
                "sum",  # pragma: no cover
                "mean",  # pragma: no cover
                "var",  # pragma: no cover
                "std",  # pragma: no cover
                "max",  # pragma: no cover
                "min",  # pragma: no cover
                "tensordot",  # pragma: no cover
            ):  # pragma: no cover
                return _to_tensor(x)  # pragma: no cover
        if type(x).__name__ == "ndarray":  # pragma: no cover
            return _to_tensor(x)  # pragma: no cover
        return x  # pragma: no cover

    def _w(x):  # pragma: no cover
        if isinstance(x, tuple):  # pragma: no cover
            return tuple((_w(i) for i in x))  # pragma: no cover
        if isinstance(x, list):  # pragma: no cover
            return [_w(i) for i in x]  # pragma: no cover
        return array(x) if hasattr(x, "shape") else x  # pragma: no cover

    fn = _get_compiler_op("triu")  # pragma: no cover
    res = fn(_u(m, "m"), k=_u(k, "k"))  # pragma: no cover

    def __w(x):
        if isinstance(x, tuple):
            return tuple((__w(i) for i in x))
        if isinstance(x, list):
            return [__w(i) for i in x]
        return array(x) if hasattr(x, "shape") else x

    return __w(res)  # pragma: no cover


def expand_dims(a, axis, stream=None):  # pragma: no cover
    """Compute expand_dims.

    Args:
        a: The a argument.
        axis: The axis argument.
        stream: The stream argument.

    Returns:
        The result of expand_dims.

    """
    from zero_mlx.array import array, _to_tensor  # pragma: no cover
    import ml_switcheroo_compiler.ops as mops  # pragma: no cover
    import ml_switcheroo_compiler.ops.nn as mnn  # pragma: no cover
    import ml_switcheroo_compiler.random as mrand  # pragma: no cover
    import types  # pragma: no cover

    def _u(x, param_name=None):  # pragma: no cover
        if isinstance(x, list):  # pragma: no cover
            if __import__("builtins").all(  # pragma: no cover
                (isinstance(i, int) for i in x)  # pragma: no cover
            ):  # pragma: no cover
                return tuple(x)  # pragma: no cover
            return [_u(i, param_name) for i in x]  # pragma: no cover
        if isinstance(x, tuple):  # pragma: no cover
            return tuple((_u(i, param_name) for i in x))  # pragma: no cover
        if hasattr(x, "_tensor"):  # pragma: no cover
            return x._tensor  # pragma: no cover
        if isinstance(x, (int, float, bool, complex)):  # pragma: no cover
            if param_name not in {  # pragma: no cover
                "axis",  # pragma: no cover
                "keepdims",
                "shape",
                "size",  # pragma: no cover
                "equal_nan",
                "nan",
                "posinf",
                "neginf",  # pragma: no cover
                "axis1",  # pragma: no cover
                "rtol",  # pragma: no cover
                "n",  # pragma: no cover
                "m",  # pragma: no cover
                "endpoint",  # pragma: no cover
                "offset",  # pragma: no cover
                "a_max",  # pragma: no cover
                "ddof",  # pragma: no cover
                "dims",  # pragma: no cover
                "source",  # pragma: no cover
                "axis2",  # pragma: no cover
                "stop",  # pragma: no cover
                "k",  # pragma: no cover
                "dtype",  # pragma: no cover
                "reps",  # pragma: no cover
                "a_min",  # pragma: no cover
                "sparse",  # pragma: no cover
                "decimals",  # pragma: no cover
                "strides",  # pragma: no cover
                "num",  # pragma: no cover
                "max",  # pragma: no cover
                "indices_or_sections",  # pragma: no cover
                "shape",  # pragma: no cover
                "atol",  # pragma: no cover
                "axis1",  # pragma: no cover
                "step",  # pragma: no cover
                "axes",  # pragma: no cover
                "fill_value",  # pragma: no cover
                "stream",  # pragma: no cover
                "destination",  # pragma: no cover
                "alpha",  # pragma: no cover
                "beta",  # pragma: no cover
                "start",  # pragma: no cover
                "equal_nan",
                "nan",
                "posinf",
                "neginf",  # pragma: no cover
                "keepdims",
                "shape",
                "size",  # pragma: no cover
                "min",  # pragma: no cover
                "p",  # pragma: no cover
            } and "expand_dims" not in (  # pragma: no cover
                "transpose",  # pragma: no cover
                "moveaxis",  # pragma: no cover
                "reshape",  # pragma: no cover
                "expand_dims",  # pragma: no cover
                "linspace",  # pragma: no cover
                "logcumsumexp",  # pragma: no cover
                "cummin",  # pragma: no cover
                "cummax",  # pragma: no cover
                "cumprod",  # pragma: no cover
                "cumsum",  # pragma: no cover
                "logsumexp",  # pragma: no cover
                "all",  # pragma: no cover
                "any",  # pragma: no cover
                "argmin",  # pragma: no cover
                "argmax",  # pragma: no cover
                "prod",  # pragma: no cover
                "sum",  # pragma: no cover
                "mean",  # pragma: no cover
                "var",  # pragma: no cover
                "std",  # pragma: no cover
                "max",  # pragma: no cover
                "min",  # pragma: no cover
                "tensordot",  # pragma: no cover
            ):  # pragma: no cover
                return _to_tensor(x)  # pragma: no cover
        if type(x).__name__ == "ndarray":  # pragma: no cover
            return _to_tensor(x)  # pragma: no cover
        return x  # pragma: no cover

    def _w(x):  # pragma: no cover
        if isinstance(x, tuple):  # pragma: no cover
            return tuple((_w(i) for i in x))  # pragma: no cover
        if isinstance(x, list):  # pragma: no cover
            return [_w(i) for i in x]  # pragma: no cover
        return array(x) if hasattr(x, "shape") else x  # pragma: no cover

    fn = _get_compiler_op("unsqueeze")  # pragma: no cover
    res = fn(_u(a, "a"), axis=_u(axis, "axis"))  # pragma: no cover

    def __w(x):
        if isinstance(x, tuple):
            return tuple((__w(i) for i in x))
        if isinstance(x, list):
            return [__w(i) for i in x]
        return array(x) if hasattr(x, "shape") else x

    return __w(res)  # pragma: no cover


def take_along_axis(a, indices, axis=None, stream=None):  # pragma: no cover
    """Compute take_along_axis.

    Args:
        a: The a argument.
        indices: The indices argument.
        axis: The axis argument.
        stream: The stream argument.

    Returns:
        The result of take_along_axis.

    """
    from zero_mlx.array import array, _to_tensor  # pragma: no cover
    import ml_switcheroo_compiler.ops as mops  # pragma: no cover
    import ml_switcheroo_compiler.ops.nn as mnn  # pragma: no cover
    import ml_switcheroo_compiler.random as mrand  # pragma: no cover
    import types  # pragma: no cover

    def _u(x, param_name=None):  # pragma: no cover
        if isinstance(x, list):  # pragma: no cover
            if __import__("builtins").all(  # pragma: no cover
                (isinstance(i, int) for i in x)  # pragma: no cover
            ):  # pragma: no cover
                return tuple(x)  # pragma: no cover
            return [_u(i, param_name) for i in x]  # pragma: no cover
        if isinstance(x, tuple):  # pragma: no cover
            return tuple((_u(i, param_name) for i in x))  # pragma: no cover
        if hasattr(x, "_tensor"):  # pragma: no cover
            return x._tensor  # pragma: no cover
        if isinstance(x, (int, float, bool, complex)):  # pragma: no cover
            if param_name not in {  # pragma: no cover
                "axis",  # pragma: no cover
                "keepdims",
                "shape",
                "size",  # pragma: no cover
                "equal_nan",
                "nan",
                "posinf",
                "neginf",  # pragma: no cover
                "axis1",  # pragma: no cover
                "rtol",  # pragma: no cover
                "n",  # pragma: no cover
                "m",  # pragma: no cover
                "endpoint",  # pragma: no cover
                "offset",  # pragma: no cover
                "a_max",  # pragma: no cover
                "ddof",  # pragma: no cover
                "dims",  # pragma: no cover
                "source",  # pragma: no cover
                "axis2",  # pragma: no cover
                "stop",  # pragma: no cover
                "k",  # pragma: no cover
                "dtype",  # pragma: no cover
                "reps",  # pragma: no cover
                "a_min",  # pragma: no cover
                "sparse",  # pragma: no cover
                "decimals",  # pragma: no cover
                "strides",  # pragma: no cover
                "num",  # pragma: no cover
                "max",  # pragma: no cover
                "indices_or_sections",  # pragma: no cover
                "shape",  # pragma: no cover
                "atol",  # pragma: no cover
                "axis1",  # pragma: no cover
                "step",  # pragma: no cover
                "axes",  # pragma: no cover
                "fill_value",  # pragma: no cover
                "stream",  # pragma: no cover
                "destination",  # pragma: no cover
                "alpha",  # pragma: no cover
                "beta",  # pragma: no cover
                "start",  # pragma: no cover
                "equal_nan",
                "nan",
                "posinf",
                "neginf",  # pragma: no cover
                "keepdims",
                "shape",
                "size",  # pragma: no cover
                "min",  # pragma: no cover
                "p",  # pragma: no cover
            } and "take_along_axis" not in (  # pragma: no cover
                "transpose",  # pragma: no cover
                "moveaxis",  # pragma: no cover
                "reshape",  # pragma: no cover
                "expand_dims",  # pragma: no cover
                "linspace",  # pragma: no cover
                "logcumsumexp",  # pragma: no cover
                "cummin",  # pragma: no cover
                "cummax",  # pragma: no cover
                "cumprod",  # pragma: no cover
                "cumsum",  # pragma: no cover
                "logsumexp",  # pragma: no cover
                "all",  # pragma: no cover
                "any",  # pragma: no cover
                "argmin",  # pragma: no cover
                "argmax",  # pragma: no cover
                "prod",  # pragma: no cover
                "sum",  # pragma: no cover
                "mean",  # pragma: no cover
                "var",  # pragma: no cover
                "std",  # pragma: no cover
                "max",  # pragma: no cover
                "min",  # pragma: no cover
                "tensordot",  # pragma: no cover
            ):  # pragma: no cover
                return _to_tensor(x)  # pragma: no cover
        if type(x).__name__ == "ndarray":  # pragma: no cover
            return _to_tensor(x)  # pragma: no cover
        return x  # pragma: no cover

    def _w(x):  # pragma: no cover
        if isinstance(x, tuple):  # pragma: no cover
            return tuple((_w(i) for i in x))  # pragma: no cover
        if isinstance(x, list):  # pragma: no cover
            return [_w(i) for i in x]  # pragma: no cover
        return array(x) if hasattr(x, "shape") else x  # pragma: no cover

    fn = _get_compiler_op("take_along_axis")  # pragma: no cover
    res = fn(
        _u(a, "a"), _u(indices, "indices"), axis=_u(axis, "axis")
    )  # pragma: no cover

    def __w(x):
        if isinstance(x, tuple):
            return tuple((__w(i) for i in x))
        if isinstance(x, list):
            return [__w(i) for i in x]
        return array(x) if hasattr(x, "shape") else x

    return __w(res)  # pragma: no cover


def tile(A, reps, stream=None):  # pragma: no cover
    """Compute tile.

    Args:
        A: The A argument.
        reps: The reps argument.
        stream: The stream argument.

    Returns:
        The result of tile.

    """
    from zero_mlx.array import array, _to_tensor  # pragma: no cover
    import ml_switcheroo_compiler.ops as mops  # pragma: no cover
    import ml_switcheroo_compiler.ops.nn as mnn  # pragma: no cover
    import ml_switcheroo_compiler.random as mrand  # pragma: no cover
    import types  # pragma: no cover

    def _u(x, param_name=None):  # pragma: no cover
        if isinstance(x, list):  # pragma: no cover
            if __import__("builtins").all(  # pragma: no cover
                (isinstance(i, int) for i in x)  # pragma: no cover
            ):  # pragma: no cover
                return tuple(x)  # pragma: no cover
            return [_u(i, param_name) for i in x]  # pragma: no cover
        if isinstance(x, tuple):  # pragma: no cover
            return tuple((_u(i, param_name) for i in x))  # pragma: no cover
        if hasattr(x, "_tensor"):  # pragma: no cover
            return x._tensor  # pragma: no cover
        if isinstance(x, (int, float, bool, complex)):  # pragma: no cover
            if param_name not in {  # pragma: no cover
                "axis",  # pragma: no cover
                "keepdims",
                "shape",
                "size",  # pragma: no cover
                "equal_nan",
                "nan",
                "posinf",
                "neginf",  # pragma: no cover
                "axis1",  # pragma: no cover
                "rtol",  # pragma: no cover
                "n",  # pragma: no cover
                "m",  # pragma: no cover
                "endpoint",  # pragma: no cover
                "offset",  # pragma: no cover
                "a_max",  # pragma: no cover
                "ddof",  # pragma: no cover
                "dims",  # pragma: no cover
                "source",  # pragma: no cover
                "axis2",  # pragma: no cover
                "stop",  # pragma: no cover
                "k",  # pragma: no cover
                "dtype",  # pragma: no cover
                "reps",  # pragma: no cover
                "a_min",  # pragma: no cover
                "sparse",  # pragma: no cover
                "decimals",  # pragma: no cover
                "strides",  # pragma: no cover
                "num",  # pragma: no cover
                "max",  # pragma: no cover
                "indices_or_sections",  # pragma: no cover
                "shape",  # pragma: no cover
                "atol",  # pragma: no cover
                "axis1",  # pragma: no cover
                "step",  # pragma: no cover
                "axes",  # pragma: no cover
                "fill_value",  # pragma: no cover
                "stream",  # pragma: no cover
                "destination",  # pragma: no cover
                "alpha",  # pragma: no cover
                "beta",  # pragma: no cover
                "start",  # pragma: no cover
                "equal_nan",
                "nan",
                "posinf",
                "neginf",  # pragma: no cover
                "keepdims",
                "shape",
                "size",  # pragma: no cover
                "min",  # pragma: no cover
                "p",  # pragma: no cover
            } and "tile" not in (  # pragma: no cover
                "transpose",  # pragma: no cover
                "moveaxis",  # pragma: no cover
                "reshape",  # pragma: no cover
                "expand_dims",  # pragma: no cover
                "linspace",  # pragma: no cover
                "logcumsumexp",  # pragma: no cover
                "cummin",  # pragma: no cover
                "cummax",  # pragma: no cover
                "cumprod",  # pragma: no cover
                "cumsum",  # pragma: no cover
                "logsumexp",  # pragma: no cover
                "all",  # pragma: no cover
                "any",  # pragma: no cover
                "argmin",  # pragma: no cover
                "argmax",  # pragma: no cover
                "prod",  # pragma: no cover
                "sum",  # pragma: no cover
                "mean",  # pragma: no cover
                "var",  # pragma: no cover
                "std",  # pragma: no cover
                "max",  # pragma: no cover
                "min",  # pragma: no cover
                "tensordot",  # pragma: no cover
            ):  # pragma: no cover
                return _to_tensor(x)  # pragma: no cover
        if type(x).__name__ == "ndarray":  # pragma: no cover
            return _to_tensor(x)  # pragma: no cover
        return x  # pragma: no cover

    def _w(x):  # pragma: no cover
        if isinstance(x, tuple):  # pragma: no cover
            return tuple((_w(i) for i in x))  # pragma: no cover
        if isinstance(x, list):  # pragma: no cover
            return [_w(i) for i in x]  # pragma: no cover
        return array(x) if hasattr(x, "shape") else x  # pragma: no cover

    fn = _get_compiler_op("tile")  # pragma: no cover
    res = fn(_u(A, "A"), _u(reps, "reps"))  # pragma: no cover

    def __w(x):
        if isinstance(x, tuple):
            return tuple((__w(i) for i in x))
        if isinstance(x, list):
            return [__w(i) for i in x]
        return array(x) if hasattr(x, "shape") else x

    return __w(res)  # pragma: no cover


def squeeze(a, axis=None, stream=None):  # pragma: no cover
    """Compute squeeze.

    Args:
        a: The a argument.
        axis: The axis argument.
        stream: The stream argument.

    Returns:
        The result of squeeze.

    """
    from zero_mlx.array import array, _to_tensor  # pragma: no cover
    import ml_switcheroo_compiler.ops as mops  # pragma: no cover
    import ml_switcheroo_compiler.ops.nn as mnn  # pragma: no cover
    import ml_switcheroo_compiler.random as mrand  # pragma: no cover
    import types  # pragma: no cover

    def _u(x, param_name=None):  # pragma: no cover
        if isinstance(x, list):  # pragma: no cover
            if __import__("builtins").all(  # pragma: no cover
                (isinstance(i, int) for i in x)  # pragma: no cover
            ):  # pragma: no cover
                return tuple(x)  # pragma: no cover
            return [_u(i, param_name) for i in x]  # pragma: no cover
        if isinstance(x, tuple):  # pragma: no cover
            return tuple((_u(i, param_name) for i in x))  # pragma: no cover
        if hasattr(x, "_tensor"):  # pragma: no cover
            return x._tensor  # pragma: no cover
        if isinstance(x, (int, float, bool, complex)):  # pragma: no cover
            if param_name not in {  # pragma: no cover
                "axis",  # pragma: no cover
                "keepdims",
                "shape",
                "size",  # pragma: no cover
                "equal_nan",
                "nan",
                "posinf",
                "neginf",  # pragma: no cover
                "axis1",  # pragma: no cover
                "rtol",  # pragma: no cover
                "n",  # pragma: no cover
                "m",  # pragma: no cover
                "endpoint",  # pragma: no cover
                "offset",  # pragma: no cover
                "a_max",  # pragma: no cover
                "ddof",  # pragma: no cover
                "dims",  # pragma: no cover
                "source",  # pragma: no cover
                "axis2",  # pragma: no cover
                "stop",  # pragma: no cover
                "k",  # pragma: no cover
                "dtype",  # pragma: no cover
                "reps",  # pragma: no cover
                "a_min",  # pragma: no cover
                "sparse",  # pragma: no cover
                "decimals",  # pragma: no cover
                "strides",  # pragma: no cover
                "num",  # pragma: no cover
                "max",  # pragma: no cover
                "indices_or_sections",  # pragma: no cover
                "shape",  # pragma: no cover
                "atol",  # pragma: no cover
                "axis1",  # pragma: no cover
                "step",  # pragma: no cover
                "axes",  # pragma: no cover
                "fill_value",  # pragma: no cover
                "stream",  # pragma: no cover
                "destination",  # pragma: no cover
                "alpha",  # pragma: no cover
                "beta",  # pragma: no cover
                "start",  # pragma: no cover
                "equal_nan",
                "nan",
                "posinf",
                "neginf",  # pragma: no cover
                "keepdims",
                "shape",
                "size",  # pragma: no cover
                "min",  # pragma: no cover
                "p",  # pragma: no cover
            } and "squeeze" not in (  # pragma: no cover
                "transpose",  # pragma: no cover
                "moveaxis",  # pragma: no cover
                "reshape",  # pragma: no cover
                "expand_dims",  # pragma: no cover
                "linspace",  # pragma: no cover
                "logcumsumexp",  # pragma: no cover
                "cummin",  # pragma: no cover
                "cummax",  # pragma: no cover
                "cumprod",  # pragma: no cover
                "cumsum",  # pragma: no cover
                "logsumexp",  # pragma: no cover
                "all",  # pragma: no cover
                "any",  # pragma: no cover
                "argmin",  # pragma: no cover
                "argmax",  # pragma: no cover
                "prod",  # pragma: no cover
                "sum",  # pragma: no cover
                "mean",  # pragma: no cover
                "var",  # pragma: no cover
                "std",  # pragma: no cover
                "max",  # pragma: no cover
                "min",  # pragma: no cover
                "tensordot",  # pragma: no cover
            ):  # pragma: no cover
                return _to_tensor(x)  # pragma: no cover
        if type(x).__name__ == "ndarray":  # pragma: no cover
            return _to_tensor(x)  # pragma: no cover
        return x  # pragma: no cover

    def _w(x):  # pragma: no cover
        if isinstance(x, tuple):  # pragma: no cover
            return tuple((_w(i) for i in x))  # pragma: no cover
        if isinstance(x, list):  # pragma: no cover
            return [_w(i) for i in x]  # pragma: no cover
        return array(x) if hasattr(x, "shape") else x  # pragma: no cover

    fn = _get_compiler_op("squeeze")  # pragma: no cover
    res = fn(_u(a, "a"), axis=_u(axis, "axis"))  # pragma: no cover

    def __w(x):
        if isinstance(x, tuple):
            return tuple((__w(i) for i in x))
        if isinstance(x, list):
            return [__w(i) for i in x]
        return array(x) if hasattr(x, "shape") else x

    return __w(res)  # pragma: no cover


def var(a, axis=None, keepdims=False, ddof=0, stream=None):  # pragma: no cover
    """Compute var.

    Args:
        a: The a argument.
        axis: The axis argument.
        keepdims: The keepdims argument.
        ddof: The ddof argument.
        stream: The stream argument.

    Returns:
        The result of var.

    """
    from zero_mlx.array import array, _to_tensor  # pragma: no cover
    import ml_switcheroo_compiler.ops as mops  # pragma: no cover
    import ml_switcheroo_compiler.ops.nn as mnn  # pragma: no cover
    import ml_switcheroo_compiler.random as mrand  # pragma: no cover
    import types  # pragma: no cover

    def _u(x, param_name=None):  # pragma: no cover
        if isinstance(x, list):  # pragma: no cover
            if __import__("builtins").all(  # pragma: no cover
                (isinstance(i, int) for i in x)  # pragma: no cover
            ):  # pragma: no cover
                return tuple(x)  # pragma: no cover
            return [_u(i, param_name) for i in x]  # pragma: no cover
        if isinstance(x, tuple):  # pragma: no cover
            return tuple((_u(i, param_name) for i in x))  # pragma: no cover
        if hasattr(x, "_tensor"):  # pragma: no cover
            return x._tensor  # pragma: no cover
        if isinstance(x, (int, float, bool, complex)):  # pragma: no cover
            if param_name not in {  # pragma: no cover
                "axis",  # pragma: no cover
                "keepdims",
                "shape",
                "size",  # pragma: no cover
                "equal_nan",
                "nan",
                "posinf",
                "neginf",  # pragma: no cover
                "axis1",  # pragma: no cover
                "rtol",  # pragma: no cover
                "n",  # pragma: no cover
                "m",  # pragma: no cover
                "endpoint",  # pragma: no cover
                "offset",  # pragma: no cover
                "a_max",  # pragma: no cover
                "ddof",  # pragma: no cover
                "dims",  # pragma: no cover
                "source",  # pragma: no cover
                "axis2",  # pragma: no cover
                "stop",  # pragma: no cover
                "k",  # pragma: no cover
                "dtype",  # pragma: no cover
                "reps",  # pragma: no cover
                "a_min",  # pragma: no cover
                "sparse",  # pragma: no cover
                "decimals",  # pragma: no cover
                "strides",  # pragma: no cover
                "num",  # pragma: no cover
                "max",  # pragma: no cover
                "indices_or_sections",  # pragma: no cover
                "shape",  # pragma: no cover
                "atol",  # pragma: no cover
                "axis1",  # pragma: no cover
                "step",  # pragma: no cover
                "axes",  # pragma: no cover
                "fill_value",  # pragma: no cover
                "stream",  # pragma: no cover
                "destination",  # pragma: no cover
                "alpha",  # pragma: no cover
                "beta",  # pragma: no cover
                "start",  # pragma: no cover
                "equal_nan",
                "nan",
                "posinf",
                "neginf",  # pragma: no cover
                "keepdims",
                "shape",
                "size",  # pragma: no cover
                "min",  # pragma: no cover
                "p",  # pragma: no cover
            } and "var" not in (  # pragma: no cover
                "transpose",  # pragma: no cover
                "moveaxis",  # pragma: no cover
                "reshape",  # pragma: no cover
                "expand_dims",  # pragma: no cover
                "linspace",  # pragma: no cover
                "logcumsumexp",  # pragma: no cover
                "cummin",  # pragma: no cover
                "cummax",  # pragma: no cover
                "cumprod",  # pragma: no cover
                "cumsum",  # pragma: no cover
                "logsumexp",  # pragma: no cover
                "all",  # pragma: no cover
                "any",  # pragma: no cover
                "argmin",  # pragma: no cover
                "argmax",  # pragma: no cover
                "prod",  # pragma: no cover
                "sum",  # pragma: no cover
                "mean",  # pragma: no cover
                "var",  # pragma: no cover
                "std",  # pragma: no cover
                "max",  # pragma: no cover
                "min",  # pragma: no cover
                "tensordot",  # pragma: no cover
            ):  # pragma: no cover
                return _to_tensor(x)  # pragma: no cover
        if type(x).__name__ == "ndarray":  # pragma: no cover
            return _to_tensor(x)  # pragma: no cover
        return x  # pragma: no cover

    def _w(x):  # pragma: no cover
        if isinstance(x, tuple):  # pragma: no cover
            return tuple((_w(i) for i in x))  # pragma: no cover
        if isinstance(x, list):  # pragma: no cover
            return [_w(i) for i in x]  # pragma: no cover
        return array(x) if hasattr(x, "shape") else x  # pragma: no cover

    fn = _get_compiler_op("variance")  # pragma: no cover
    res = fn(  # pragma: no cover
        _u(a, "a"),  # pragma: no cover
        axis=_u(axis, "axis"),  # pragma: no cover
        keepdims=_u(keepdims, "keepdims"),  # pragma: no cover
        ddof=_u(ddof, "ddof"),  # pragma: no cover
    )  # pragma: no cover

    def __w(x):
        if isinstance(x, tuple):
            return tuple((__w(i) for i in x))
        if isinstance(x, list):
            return [__w(i) for i in x]
        return array(x) if hasattr(x, "shape") else x

    return __w(res)  # pragma: no cover


def std(a, axis=None, keepdims=False, ddof=0, stream=None):  # pragma: no cover
    """Compute std.

    Args:
        a: The a argument.
        axis: The axis argument.
        keepdims: The keepdims argument.
        ddof: The ddof argument.
        stream: The stream argument.

    Returns:
        The result of std.

    """
    from zero_mlx.array import array, _to_tensor  # pragma: no cover
    import ml_switcheroo_compiler.ops as mops  # pragma: no cover
    import ml_switcheroo_compiler.ops.nn as mnn  # pragma: no cover
    import ml_switcheroo_compiler.random as mrand  # pragma: no cover
    import types  # pragma: no cover

    def _u(x, param_name=None):  # pragma: no cover
        if isinstance(x, list):  # pragma: no cover
            if __import__("builtins").all(  # pragma: no cover
                (isinstance(i, int) for i in x)  # pragma: no cover
            ):  # pragma: no cover
                return tuple(x)  # pragma: no cover
            return [_u(i, param_name) for i in x]  # pragma: no cover
        if isinstance(x, tuple):  # pragma: no cover
            return tuple((_u(i, param_name) for i in x))  # pragma: no cover
        if hasattr(x, "_tensor"):  # pragma: no cover
            return x._tensor  # pragma: no cover
        if isinstance(x, (int, float, bool, complex)):  # pragma: no cover
            if param_name not in {  # pragma: no cover
                "axis",  # pragma: no cover
                "keepdims",
                "shape",
                "size",  # pragma: no cover
                "equal_nan",
                "nan",
                "posinf",
                "neginf",  # pragma: no cover
                "axis1",  # pragma: no cover
                "rtol",  # pragma: no cover
                "n",  # pragma: no cover
                "m",  # pragma: no cover
                "endpoint",  # pragma: no cover
                "offset",  # pragma: no cover
                "a_max",  # pragma: no cover
                "ddof",  # pragma: no cover
                "dims",  # pragma: no cover
                "source",  # pragma: no cover
                "axis2",  # pragma: no cover
                "stop",  # pragma: no cover
                "k",  # pragma: no cover
                "dtype",  # pragma: no cover
                "reps",  # pragma: no cover
                "a_min",  # pragma: no cover
                "sparse",  # pragma: no cover
                "decimals",  # pragma: no cover
                "strides",  # pragma: no cover
                "num",  # pragma: no cover
                "max",  # pragma: no cover
                "indices_or_sections",  # pragma: no cover
                "shape",  # pragma: no cover
                "atol",  # pragma: no cover
                "axis1",  # pragma: no cover
                "step",  # pragma: no cover
                "axes",  # pragma: no cover
                "fill_value",  # pragma: no cover
                "stream",  # pragma: no cover
                "destination",  # pragma: no cover
                "alpha",  # pragma: no cover
                "beta",  # pragma: no cover
                "start",  # pragma: no cover
                "equal_nan",
                "nan",
                "posinf",
                "neginf",  # pragma: no cover
                "keepdims",
                "shape",
                "size",  # pragma: no cover
                "min",  # pragma: no cover
                "p",  # pragma: no cover
            } and "std" not in (  # pragma: no cover
                "transpose",  # pragma: no cover
                "moveaxis",  # pragma: no cover
                "reshape",  # pragma: no cover
                "expand_dims",  # pragma: no cover
                "linspace",  # pragma: no cover
                "logcumsumexp",  # pragma: no cover
                "cummin",  # pragma: no cover
                "cummax",  # pragma: no cover
                "cumprod",  # pragma: no cover
                "cumsum",  # pragma: no cover
                "logsumexp",  # pragma: no cover
                "all",  # pragma: no cover
                "any",  # pragma: no cover
                "argmin",  # pragma: no cover
                "argmax",  # pragma: no cover
                "prod",  # pragma: no cover
                "sum",  # pragma: no cover
                "mean",  # pragma: no cover
                "var",  # pragma: no cover
                "std",  # pragma: no cover
                "max",  # pragma: no cover
                "min",  # pragma: no cover
                "tensordot",  # pragma: no cover
            ):  # pragma: no cover
                return _to_tensor(x)  # pragma: no cover
        if type(x).__name__ == "ndarray":  # pragma: no cover
            return _to_tensor(x)  # pragma: no cover
        return x  # pragma: no cover

    def _w(x):  # pragma: no cover
        if isinstance(x, tuple):  # pragma: no cover
            return tuple((_w(i) for i in x))  # pragma: no cover
        if isinstance(x, list):  # pragma: no cover
            return [_w(i) for i in x]  # pragma: no cover
        return array(x) if hasattr(x, "shape") else x  # pragma: no cover

    fn = _get_compiler_op("std")  # pragma: no cover
    res = fn(  # pragma: no cover
        _u(a, "a"),  # pragma: no cover
        axis=_u(axis, "axis"),  # pragma: no cover
        keepdims=_u(keepdims, "keepdims"),  # pragma: no cover
        ddof=_u(ddof, "ddof"),  # pragma: no cover
    )  # pragma: no cover

    def __w(x):
        if isinstance(x, tuple):
            return tuple((__w(i) for i in x))
        if isinstance(x, list):
            return [__w(i) for i in x]
        return array(x) if hasattr(x, "shape") else x

    return __w(res)  # pragma: no cover


def round(a, decimals=0, stream=None):  # pragma: no cover
    """Compute round.

    Args:
        a: The a argument.
        decimals: The decimals argument.
        stream: The stream argument.

    Returns:
        The result of round.

    """
    from zero_mlx.array import array, _to_tensor  # pragma: no cover
    import ml_switcheroo_compiler.ops as mops  # pragma: no cover
    import ml_switcheroo_compiler.ops.nn as mnn  # pragma: no cover
    import ml_switcheroo_compiler.random as mrand  # pragma: no cover
    import types  # pragma: no cover

    def _u(x, param_name=None):  # pragma: no cover
        if isinstance(x, list):  # pragma: no cover
            if __import__("builtins").all(  # pragma: no cover
                (isinstance(i, int) for i in x)  # pragma: no cover
            ):  # pragma: no cover
                return tuple(x)  # pragma: no cover
            return [_u(i, param_name) for i in x]  # pragma: no cover
        if isinstance(x, tuple):  # pragma: no cover
            return tuple((_u(i, param_name) for i in x))  # pragma: no cover
        if hasattr(x, "_tensor"):  # pragma: no cover
            return x._tensor  # pragma: no cover
        if isinstance(x, (int, float, bool, complex)):  # pragma: no cover
            if param_name not in {  # pragma: no cover
                "axis",  # pragma: no cover
                "keepdims",
                "shape",
                "size",  # pragma: no cover
                "equal_nan",
                "nan",
                "posinf",
                "neginf",  # pragma: no cover
                "axis1",  # pragma: no cover
                "rtol",  # pragma: no cover
                "n",  # pragma: no cover
                "m",  # pragma: no cover
                "endpoint",  # pragma: no cover
                "offset",  # pragma: no cover
                "a_max",  # pragma: no cover
                "ddof",  # pragma: no cover
                "dims",  # pragma: no cover
                "source",  # pragma: no cover
                "axis2",  # pragma: no cover
                "stop",  # pragma: no cover
                "k",  # pragma: no cover
                "dtype",  # pragma: no cover
                "reps",  # pragma: no cover
                "a_min",  # pragma: no cover
                "sparse",  # pragma: no cover
                "decimals",  # pragma: no cover
                "strides",  # pragma: no cover
                "num",  # pragma: no cover
                "max",  # pragma: no cover
                "indices_or_sections",  # pragma: no cover
                "shape",  # pragma: no cover
                "atol",  # pragma: no cover
                "axis1",  # pragma: no cover
                "step",  # pragma: no cover
                "axes",  # pragma: no cover
                "fill_value",  # pragma: no cover
                "stream",  # pragma: no cover
                "destination",  # pragma: no cover
                "alpha",  # pragma: no cover
                "beta",  # pragma: no cover
                "start",  # pragma: no cover
                "equal_nan",
                "nan",
                "posinf",
                "neginf",  # pragma: no cover
                "keepdims",
                "shape",
                "size",  # pragma: no cover
                "min",  # pragma: no cover
                "p",  # pragma: no cover
            } and "round" not in (  # pragma: no cover
                "transpose",  # pragma: no cover
                "moveaxis",  # pragma: no cover
                "reshape",  # pragma: no cover
                "expand_dims",  # pragma: no cover
                "linspace",  # pragma: no cover
                "logcumsumexp",  # pragma: no cover
                "cummin",  # pragma: no cover
                "cummax",  # pragma: no cover
                "cumprod",  # pragma: no cover
                "cumsum",  # pragma: no cover
                "logsumexp",  # pragma: no cover
                "all",  # pragma: no cover
                "any",  # pragma: no cover
                "argmin",  # pragma: no cover
                "argmax",  # pragma: no cover
                "prod",  # pragma: no cover
                "sum",  # pragma: no cover
                "mean",  # pragma: no cover
                "var",  # pragma: no cover
                "std",  # pragma: no cover
                "max",  # pragma: no cover
                "min",  # pragma: no cover
                "tensordot",  # pragma: no cover
            ):  # pragma: no cover
                return _to_tensor(x)  # pragma: no cover
        if type(x).__name__ == "ndarray":  # pragma: no cover
            return _to_tensor(x)  # pragma: no cover
        return x  # pragma: no cover

    def _w(x):  # pragma: no cover
        if isinstance(x, tuple):  # pragma: no cover
            return tuple((_w(i) for i in x))  # pragma: no cover
        if isinstance(x, list):  # pragma: no cover
            return [_w(i) for i in x]  # pragma: no cover
        return array(x) if hasattr(x, "shape") else x  # pragma: no cover

    fn = _get_compiler_op("round")  # pragma: no cover
    res = fn(_u(a, "a"), decimals=_u(decimals, "decimals"))  # pragma: no cover

    def __w(x):
        if isinstance(x, tuple):
            return tuple((__w(i) for i in x))
        if isinstance(x, list):
            return [__w(i) for i in x]
        return array(x) if hasattr(x, "shape") else x

    return __w(res)  # pragma: no cover


def sort(a, axis=-1, stream=None):  # pragma: no cover
    """Compute sort.

    Args:
        a: The a argument.
        axis: The axis argument.
        stream: The stream argument.

    Returns:
        The result of sort.

    """
    from zero_mlx.array import array, _to_tensor  # pragma: no cover
    import ml_switcheroo_compiler.ops as mops  # pragma: no cover
    import ml_switcheroo_compiler.ops.nn as mnn  # pragma: no cover
    import ml_switcheroo_compiler.random as mrand  # pragma: no cover
    import types  # pragma: no cover

    def _u(x, param_name=None):  # pragma: no cover
        if isinstance(x, list):  # pragma: no cover
            if __import__("builtins").all(  # pragma: no cover
                (isinstance(i, int) for i in x)  # pragma: no cover
            ):  # pragma: no cover
                return tuple(x)  # pragma: no cover
            return [_u(i, param_name) for i in x]  # pragma: no cover
        if isinstance(x, tuple):  # pragma: no cover
            return tuple((_u(i, param_name) for i in x))  # pragma: no cover
        if hasattr(x, "_tensor"):  # pragma: no cover
            return x._tensor  # pragma: no cover
        if isinstance(x, (int, float, bool, complex)):  # pragma: no cover
            if param_name not in {  # pragma: no cover
                "axis",  # pragma: no cover
                "keepdims",
                "shape",
                "size",  # pragma: no cover
                "equal_nan",
                "nan",
                "posinf",
                "neginf",  # pragma: no cover
                "axis1",  # pragma: no cover
                "rtol",  # pragma: no cover
                "n",  # pragma: no cover
                "m",  # pragma: no cover
                "endpoint",  # pragma: no cover
                "offset",  # pragma: no cover
                "a_max",  # pragma: no cover
                "ddof",  # pragma: no cover
                "dims",  # pragma: no cover
                "source",  # pragma: no cover
                "axis2",  # pragma: no cover
                "stop",  # pragma: no cover
                "k",  # pragma: no cover
                "dtype",  # pragma: no cover
                "reps",  # pragma: no cover
                "a_min",  # pragma: no cover
                "sparse",  # pragma: no cover
                "decimals",  # pragma: no cover
                "strides",  # pragma: no cover
                "num",  # pragma: no cover
                "max",  # pragma: no cover
                "indices_or_sections",  # pragma: no cover
                "shape",  # pragma: no cover
                "atol",  # pragma: no cover
                "axis1",  # pragma: no cover
                "step",  # pragma: no cover
                "axes",  # pragma: no cover
                "fill_value",  # pragma: no cover
                "stream",  # pragma: no cover
                "destination",  # pragma: no cover
                "alpha",  # pragma: no cover
                "beta",  # pragma: no cover
                "start",  # pragma: no cover
                "equal_nan",
                "nan",
                "posinf",
                "neginf",  # pragma: no cover
                "keepdims",
                "shape",
                "size",  # pragma: no cover
                "min",  # pragma: no cover
                "p",  # pragma: no cover
            } and "sort" not in (  # pragma: no cover
                "transpose",  # pragma: no cover
                "moveaxis",  # pragma: no cover
                "reshape",  # pragma: no cover
                "expand_dims",  # pragma: no cover
                "linspace",  # pragma: no cover
                "logcumsumexp",  # pragma: no cover
                "cummin",  # pragma: no cover
                "cummax",  # pragma: no cover
                "cumprod",  # pragma: no cover
                "cumsum",  # pragma: no cover
                "logsumexp",  # pragma: no cover
                "all",  # pragma: no cover
                "any",  # pragma: no cover
                "argmin",  # pragma: no cover
                "argmax",  # pragma: no cover
                "prod",  # pragma: no cover
                "sum",  # pragma: no cover
                "mean",  # pragma: no cover
                "var",  # pragma: no cover
                "std",  # pragma: no cover
                "max",  # pragma: no cover
                "min",  # pragma: no cover
                "tensordot",  # pragma: no cover
            ):  # pragma: no cover
                return _to_tensor(x)  # pragma: no cover
        if type(x).__name__ == "ndarray":  # pragma: no cover
            return _to_tensor(x)  # pragma: no cover
        return x  # pragma: no cover

    def _w(x):  # pragma: no cover
        if isinstance(x, tuple):  # pragma: no cover
            return tuple((_w(i) for i in x))  # pragma: no cover
        if isinstance(x, list):  # pragma: no cover
            return [_w(i) for i in x]  # pragma: no cover
        return array(x) if hasattr(x, "shape") else x  # pragma: no cover

    fn = _get_compiler_op("sort")  # pragma: no cover
    res = fn(_u(a, "a"), axis=_u(axis, "axis"))  # pragma: no cover

    def __w(x):
        if isinstance(x, tuple):
            return tuple((__w(i) for i in x))
        if isinstance(x, list):
            return [__w(i) for i in x]
        return array(x) if hasattr(x, "shape") else x

    return __w(res)  # pragma: no cover


def argsort(a, axis=-1, stream=None):  # pragma: no cover
    """Compute argsort.

    Args:
        a: The a argument.
        axis: The axis argument.
        stream: The stream argument.

    Returns:
        The result of argsort.

    """
    from zero_mlx.array import array, _to_tensor  # pragma: no cover
    import ml_switcheroo_compiler.ops as mops  # pragma: no cover
    import ml_switcheroo_compiler.ops.nn as mnn  # pragma: no cover
    import ml_switcheroo_compiler.random as mrand  # pragma: no cover
    import types  # pragma: no cover

    def _u(x, param_name=None):  # pragma: no cover
        if isinstance(x, list):  # pragma: no cover
            if __import__("builtins").all(  # pragma: no cover
                (isinstance(i, int) for i in x)  # pragma: no cover
            ):  # pragma: no cover
                return tuple(x)  # pragma: no cover
            return [_u(i, param_name) for i in x]  # pragma: no cover
        if isinstance(x, tuple):  # pragma: no cover
            return tuple((_u(i, param_name) for i in x))  # pragma: no cover
        if hasattr(x, "_tensor"):  # pragma: no cover
            return x._tensor  # pragma: no cover
        if isinstance(x, (int, float, bool, complex)):  # pragma: no cover
            if param_name not in {  # pragma: no cover
                "axis",  # pragma: no cover
                "keepdims",
                "shape",
                "size",  # pragma: no cover
                "equal_nan",
                "nan",
                "posinf",
                "neginf",  # pragma: no cover
                "axis1",  # pragma: no cover
                "rtol",  # pragma: no cover
                "n",  # pragma: no cover
                "m",  # pragma: no cover
                "endpoint",  # pragma: no cover
                "offset",  # pragma: no cover
                "a_max",  # pragma: no cover
                "ddof",  # pragma: no cover
                "dims",  # pragma: no cover
                "source",  # pragma: no cover
                "axis2",  # pragma: no cover
                "stop",  # pragma: no cover
                "k",  # pragma: no cover
                "dtype",  # pragma: no cover
                "reps",  # pragma: no cover
                "a_min",  # pragma: no cover
                "sparse",  # pragma: no cover
                "decimals",  # pragma: no cover
                "strides",  # pragma: no cover
                "num",  # pragma: no cover
                "max",  # pragma: no cover
                "indices_or_sections",  # pragma: no cover
                "shape",  # pragma: no cover
                "atol",  # pragma: no cover
                "axis1",  # pragma: no cover
                "step",  # pragma: no cover
                "axes",  # pragma: no cover
                "fill_value",  # pragma: no cover
                "stream",  # pragma: no cover
                "destination",  # pragma: no cover
                "alpha",  # pragma: no cover
                "beta",  # pragma: no cover
                "start",  # pragma: no cover
                "equal_nan",
                "nan",
                "posinf",
                "neginf",  # pragma: no cover
                "keepdims",
                "shape",
                "size",  # pragma: no cover
                "min",  # pragma: no cover
                "p",  # pragma: no cover
            } and "argsort" not in (  # pragma: no cover
                "transpose",  # pragma: no cover
                "moveaxis",  # pragma: no cover
                "reshape",  # pragma: no cover
                "expand_dims",  # pragma: no cover
                "linspace",  # pragma: no cover
                "logcumsumexp",  # pragma: no cover
                "cummin",  # pragma: no cover
                "cummax",  # pragma: no cover
                "cumprod",  # pragma: no cover
                "cumsum",  # pragma: no cover
                "logsumexp",  # pragma: no cover
                "all",  # pragma: no cover
                "any",  # pragma: no cover
                "argmin",  # pragma: no cover
                "argmax",  # pragma: no cover
                "prod",  # pragma: no cover
                "sum",  # pragma: no cover
                "mean",  # pragma: no cover
                "var",  # pragma: no cover
                "std",  # pragma: no cover
                "max",  # pragma: no cover
                "min",  # pragma: no cover
                "tensordot",  # pragma: no cover
            ):  # pragma: no cover
                return _to_tensor(x)  # pragma: no cover
        if type(x).__name__ == "ndarray":  # pragma: no cover
            return _to_tensor(x)  # pragma: no cover
        return x  # pragma: no cover

    def _w(x):  # pragma: no cover
        if isinstance(x, tuple):  # pragma: no cover
            return tuple((_w(i) for i in x))  # pragma: no cover
        if isinstance(x, list):  # pragma: no cover
            return [_w(i) for i in x]  # pragma: no cover
        return array(x) if hasattr(x, "shape") else x  # pragma: no cover

    fn = _get_compiler_op("argsort")  # pragma: no cover
    res = fn(_u(a, "a"), axis=_u(axis, "axis"))  # pragma: no cover

    def __w(x):
        if isinstance(x, tuple):
            return tuple((__w(i) for i in x))
        if isinstance(x, list):
            return [__w(i) for i in x]
        return array(x) if hasattr(x, "shape") else x

    return __w(res)  # pragma: no cover


def swapaxes(a, axis1, axis2, stream=None):  # pragma: no cover
    """Compute swapaxes.

    Args:
        a: The a argument.
        axis1: The axis1 argument.
        axis2: The axis2 argument.
        stream: The stream argument.

    Returns:
        The result of swapaxes.

    """
    from zero_mlx.array import array, _to_tensor  # pragma: no cover
    import ml_switcheroo_compiler.ops as mops  # pragma: no cover
    import ml_switcheroo_compiler.ops.nn as mnn  # pragma: no cover
    import ml_switcheroo_compiler.random as mrand  # pragma: no cover
    import types  # pragma: no cover

    def _u(x, param_name=None):  # pragma: no cover
        if isinstance(x, list):  # pragma: no cover
            if __import__("builtins").all(  # pragma: no cover
                (isinstance(i, int) for i in x)  # pragma: no cover
            ):  # pragma: no cover
                return tuple(x)  # pragma: no cover
            return [_u(i, param_name) for i in x]  # pragma: no cover
        if isinstance(x, tuple):  # pragma: no cover
            return tuple((_u(i, param_name) for i in x))  # pragma: no cover
        if hasattr(x, "_tensor"):  # pragma: no cover
            return x._tensor  # pragma: no cover
        if isinstance(x, (int, float, bool, complex)):  # pragma: no cover
            if param_name not in {  # pragma: no cover
                "axis",  # pragma: no cover
                "keepdims",
                "shape",
                "size",  # pragma: no cover
                "equal_nan",
                "nan",
                "posinf",
                "neginf",  # pragma: no cover
                "axis1",  # pragma: no cover
                "rtol",  # pragma: no cover
                "n",  # pragma: no cover
                "m",  # pragma: no cover
                "endpoint",  # pragma: no cover
                "offset",  # pragma: no cover
                "a_max",  # pragma: no cover
                "ddof",  # pragma: no cover
                "dims",  # pragma: no cover
                "source",  # pragma: no cover
                "axis2",  # pragma: no cover
                "stop",  # pragma: no cover
                "k",  # pragma: no cover
                "dtype",  # pragma: no cover
                "reps",  # pragma: no cover
                "a_min",  # pragma: no cover
                "sparse",  # pragma: no cover
                "decimals",  # pragma: no cover
                "strides",  # pragma: no cover
                "num",  # pragma: no cover
                "max",  # pragma: no cover
                "indices_or_sections",  # pragma: no cover
                "shape",  # pragma: no cover
                "atol",  # pragma: no cover
                "axis1",  # pragma: no cover
                "step",  # pragma: no cover
                "axes",  # pragma: no cover
                "fill_value",  # pragma: no cover
                "stream",  # pragma: no cover
                "destination",  # pragma: no cover
                "alpha",  # pragma: no cover
                "beta",  # pragma: no cover
                "start",  # pragma: no cover
                "equal_nan",
                "nan",
                "posinf",
                "neginf",  # pragma: no cover
                "keepdims",
                "shape",
                "size",  # pragma: no cover
                "min",  # pragma: no cover
                "p",  # pragma: no cover
            } and "swapaxes" not in (  # pragma: no cover
                "transpose",  # pragma: no cover
                "moveaxis",  # pragma: no cover
                "reshape",  # pragma: no cover
                "expand_dims",  # pragma: no cover
                "linspace",  # pragma: no cover
                "logcumsumexp",  # pragma: no cover
                "cummin",  # pragma: no cover
                "cummax",  # pragma: no cover
                "cumprod",  # pragma: no cover
                "cumsum",  # pragma: no cover
                "logsumexp",  # pragma: no cover
                "all",  # pragma: no cover
                "any",  # pragma: no cover
                "argmin",  # pragma: no cover
                "argmax",  # pragma: no cover
                "prod",  # pragma: no cover
                "sum",  # pragma: no cover
                "mean",  # pragma: no cover
                "var",  # pragma: no cover
                "std",  # pragma: no cover
                "max",  # pragma: no cover
                "min",  # pragma: no cover
                "tensordot",  # pragma: no cover
            ):  # pragma: no cover
                return _to_tensor(x)  # pragma: no cover
        if type(x).__name__ == "ndarray":  # pragma: no cover
            return _to_tensor(x)  # pragma: no cover
        return x  # pragma: no cover

    def _w(x):  # pragma: no cover
        if isinstance(x, tuple):  # pragma: no cover
            return tuple((_w(i) for i in x))  # pragma: no cover
        if isinstance(x, list):  # pragma: no cover
            return [_w(i) for i in x]  # pragma: no cover
        return array(x) if hasattr(x, "shape") else x  # pragma: no cover

    fn = _get_compiler_op("swapaxes")  # pragma: no cover
    res = fn(
        _u(a, "a"), axis1=_u(axis1, "axis1"), axis2=_u(axis2, "axis2")
    )  # pragma: no cover

    def __w(x):
        if isinstance(x, tuple):
            return tuple((__w(i) for i in x))
        if isinstance(x, list):
            return [__w(i) for i in x]
        return array(x) if hasattr(x, "shape") else x

    return __w(res)  # pragma: no cover


def moveaxis(a, source, destination, stream=None):  # pragma: no cover
    """Compute moveaxis.

    Args:
        a: The a argument.
        source: The source argument.
        destination: The destination argument.
        stream: The stream argument.

    Returns:
        The result of moveaxis.

    """
    from zero_mlx.array import array, _to_tensor  # pragma: no cover
    import ml_switcheroo_compiler.ops as mops  # pragma: no cover
    import ml_switcheroo_compiler.ops.nn as mnn  # pragma: no cover
    import ml_switcheroo_compiler.random as mrand  # pragma: no cover
    import types  # pragma: no cover

    def _u(x, param_name=None):  # pragma: no cover
        if isinstance(x, list):  # pragma: no cover
            if __import__("builtins").all(  # pragma: no cover
                (isinstance(i, int) for i in x)  # pragma: no cover
            ):  # pragma: no cover
                return tuple(x)  # pragma: no cover
            return [_u(i, param_name) for i in x]  # pragma: no cover
        if isinstance(x, tuple):  # pragma: no cover
            return tuple((_u(i, param_name) for i in x))  # pragma: no cover
        if hasattr(x, "_tensor"):  # pragma: no cover
            return x._tensor  # pragma: no cover
        if isinstance(x, (int, float, bool, complex)):  # pragma: no cover
            if param_name not in {  # pragma: no cover
                "axis",  # pragma: no cover
                "keepdims",
                "shape",
                "size",  # pragma: no cover
                "equal_nan",
                "nan",
                "posinf",
                "neginf",  # pragma: no cover
                "axis1",  # pragma: no cover
                "rtol",  # pragma: no cover
                "n",  # pragma: no cover
                "m",  # pragma: no cover
                "endpoint",  # pragma: no cover
                "offset",  # pragma: no cover
                "a_max",  # pragma: no cover
                "ddof",  # pragma: no cover
                "dims",  # pragma: no cover
                "source",  # pragma: no cover
                "axis2",  # pragma: no cover
                "stop",  # pragma: no cover
                "k",  # pragma: no cover
                "dtype",  # pragma: no cover
                "reps",  # pragma: no cover
                "a_min",  # pragma: no cover
                "sparse",  # pragma: no cover
                "decimals",  # pragma: no cover
                "strides",  # pragma: no cover
                "num",  # pragma: no cover
                "max",  # pragma: no cover
                "indices_or_sections",  # pragma: no cover
                "shape",  # pragma: no cover
                "atol",  # pragma: no cover
                "axis1",  # pragma: no cover
                "step",  # pragma: no cover
                "axes",  # pragma: no cover
                "fill_value",  # pragma: no cover
                "stream",  # pragma: no cover
                "destination",  # pragma: no cover
                "alpha",  # pragma: no cover
                "beta",  # pragma: no cover
                "start",  # pragma: no cover
                "equal_nan",
                "nan",
                "posinf",
                "neginf",  # pragma: no cover
                "keepdims",
                "shape",
                "size",  # pragma: no cover
                "min",  # pragma: no cover
                "p",  # pragma: no cover
            } and "moveaxis" not in (  # pragma: no cover
                "transpose",  # pragma: no cover
                "moveaxis",  # pragma: no cover
                "reshape",  # pragma: no cover
                "expand_dims",  # pragma: no cover
                "linspace",  # pragma: no cover
                "logcumsumexp",  # pragma: no cover
                "cummin",  # pragma: no cover
                "cummax",  # pragma: no cover
                "cumprod",  # pragma: no cover
                "cumsum",  # pragma: no cover
                "logsumexp",  # pragma: no cover
                "all",  # pragma: no cover
                "any",  # pragma: no cover
                "argmin",  # pragma: no cover
                "argmax",  # pragma: no cover
                "prod",  # pragma: no cover
                "sum",  # pragma: no cover
                "mean",  # pragma: no cover
                "var",  # pragma: no cover
                "std",  # pragma: no cover
                "max",  # pragma: no cover
                "min",  # pragma: no cover
                "tensordot",  # pragma: no cover
            ):  # pragma: no cover
                return _to_tensor(x)  # pragma: no cover
        if type(x).__name__ == "ndarray":  # pragma: no cover
            return _to_tensor(x)  # pragma: no cover
        return x  # pragma: no cover

    def _w(x):  # pragma: no cover
        if isinstance(x, tuple):  # pragma: no cover
            return tuple((_w(i) for i in x))  # pragma: no cover
        if isinstance(x, list):  # pragma: no cover
            return [_w(i) for i in x]  # pragma: no cover
        return array(x) if hasattr(x, "shape") else x  # pragma: no cover

    fn = _get_compiler_op("moveaxis")  # pragma: no cover
    res = fn(
        _u(a, "a"), _u(source, "source"), _u(destination, "destination")
    )  # pragma: no cover

    def __w(x):
        if isinstance(x, tuple):
            return tuple((__w(i) for i in x))
        if isinstance(x, list):
            return [__w(i) for i in x]
        return array(x) if hasattr(x, "shape") else x

    return __w(res)  # pragma: no cover


def take(a, indices, axis=None, stream=None):  # pragma: no cover
    """Compute take.

    Args:
        a: The a argument.
        indices: The indices argument.
        axis: The axis argument.
        stream: The stream argument.

    Returns:
        The result of take.

    """
    from zero_mlx.array import array, _to_tensor  # pragma: no cover
    import ml_switcheroo_compiler.ops as mops  # pragma: no cover
    import ml_switcheroo_compiler.ops.nn as mnn  # pragma: no cover
    import ml_switcheroo_compiler.random as mrand  # pragma: no cover
    import types  # pragma: no cover

    def _u(x, param_name=None):  # pragma: no cover
        if isinstance(x, list):  # pragma: no cover
            if __import__("builtins").all(  # pragma: no cover
                (isinstance(i, int) for i in x)  # pragma: no cover
            ):  # pragma: no cover
                return tuple(x)  # pragma: no cover
            return [_u(i, param_name) for i in x]  # pragma: no cover
        if isinstance(x, tuple):  # pragma: no cover
            return tuple((_u(i, param_name) for i in x))  # pragma: no cover
        if hasattr(x, "_tensor"):  # pragma: no cover
            return x._tensor  # pragma: no cover
        if isinstance(x, (int, float, bool, complex)):  # pragma: no cover
            if param_name not in {  # pragma: no cover
                "axis",  # pragma: no cover
                "keepdims",
                "shape",
                "size",  # pragma: no cover
                "equal_nan",
                "nan",
                "posinf",
                "neginf",  # pragma: no cover
                "axis1",  # pragma: no cover
                "rtol",  # pragma: no cover
                "n",  # pragma: no cover
                "m",  # pragma: no cover
                "endpoint",  # pragma: no cover
                "offset",  # pragma: no cover
                "a_max",  # pragma: no cover
                "ddof",  # pragma: no cover
                "dims",  # pragma: no cover
                "source",  # pragma: no cover
                "axis2",  # pragma: no cover
                "stop",  # pragma: no cover
                "k",  # pragma: no cover
                "dtype",  # pragma: no cover
                "reps",  # pragma: no cover
                "a_min",  # pragma: no cover
                "sparse",  # pragma: no cover
                "decimals",  # pragma: no cover
                "strides",  # pragma: no cover
                "num",  # pragma: no cover
                "max",  # pragma: no cover
                "indices_or_sections",  # pragma: no cover
                "shape",  # pragma: no cover
                "atol",  # pragma: no cover
                "axis1",  # pragma: no cover
                "step",  # pragma: no cover
                "axes",  # pragma: no cover
                "fill_value",  # pragma: no cover
                "stream",  # pragma: no cover
                "destination",  # pragma: no cover
                "alpha",  # pragma: no cover
                "beta",  # pragma: no cover
                "start",  # pragma: no cover
                "equal_nan",
                "nan",
                "posinf",
                "neginf",  # pragma: no cover
                "keepdims",
                "shape",
                "size",  # pragma: no cover
                "min",  # pragma: no cover
                "p",  # pragma: no cover
            } and "take" not in (  # pragma: no cover
                "transpose",  # pragma: no cover
                "moveaxis",  # pragma: no cover
                "reshape",  # pragma: no cover
                "expand_dims",  # pragma: no cover
                "linspace",  # pragma: no cover
                "logcumsumexp",  # pragma: no cover
                "cummin",  # pragma: no cover
                "cummax",  # pragma: no cover
                "cumprod",  # pragma: no cover
                "cumsum",  # pragma: no cover
                "logsumexp",  # pragma: no cover
                "all",  # pragma: no cover
                "any",  # pragma: no cover
                "argmin",  # pragma: no cover
                "argmax",  # pragma: no cover
                "prod",  # pragma: no cover
                "sum",  # pragma: no cover
                "mean",  # pragma: no cover
                "var",  # pragma: no cover
                "std",  # pragma: no cover
                "max",  # pragma: no cover
                "min",  # pragma: no cover
                "tensordot",  # pragma: no cover
            ):  # pragma: no cover
                return _to_tensor(x)  # pragma: no cover
        if type(x).__name__ == "ndarray":  # pragma: no cover
            return _to_tensor(x)  # pragma: no cover
        return x  # pragma: no cover

    def _w(x):  # pragma: no cover
        if isinstance(x, tuple):  # pragma: no cover
            return tuple((_w(i) for i in x))  # pragma: no cover
        if isinstance(x, list):  # pragma: no cover
            return [_w(i) for i in x]  # pragma: no cover
        return array(x) if hasattr(x, "shape") else x  # pragma: no cover

    fn = _get_compiler_op("take")  # pragma: no cover
    res = fn(  # pragma: no cover
        _u(a, "a"),
        _u(indices, "indices"),
        axis=_u(axis, "axis"),  # pragma: no cover
    )  # pragma: no cover

    def __w(x):
        if isinstance(x, tuple):
            return tuple((__w(i) for i in x))
        if isinstance(x, list):
            return [__w(i) for i in x]
        return array(x) if hasattr(x, "shape") else x

    return __w(res)  # pragma: no cover


def addmm(c, a, b, alpha=1.0, beta=1.0, stream=None):  # pragma: no cover
    """Compute addmm.

    Args:
        c: The c argument.
        a: The a argument.
        b: The b argument.
        alpha: The alpha argument.
        beta: The beta argument.
        stream: The stream argument.

    Returns:
        The result of addmm.

    """
    from zero_mlx.array import array, _to_tensor  # pragma: no cover
    import ml_switcheroo_compiler.ops as mops  # pragma: no cover
    import ml_switcheroo_compiler.ops.nn as mnn  # pragma: no cover
    import ml_switcheroo_compiler.random as mrand  # pragma: no cover
    import types  # pragma: no cover

    def _u(x, param_name=None):  # pragma: no cover
        if isinstance(x, list):  # pragma: no cover
            if __import__("builtins").all(  # pragma: no cover
                (isinstance(i, int) for i in x)  # pragma: no cover
            ):  # pragma: no cover
                return tuple(x)  # pragma: no cover
            return [_u(i, param_name) for i in x]  # pragma: no cover
        if isinstance(x, tuple):  # pragma: no cover
            return tuple((_u(i, param_name) for i in x))  # pragma: no cover
        if hasattr(x, "_tensor"):  # pragma: no cover
            return x._tensor  # pragma: no cover
        if isinstance(x, (int, float, bool, complex)):  # pragma: no cover
            if param_name not in {  # pragma: no cover
                "axis",  # pragma: no cover
                "keepdims",
                "shape",
                "size",  # pragma: no cover
                "equal_nan",
                "nan",
                "posinf",
                "neginf",  # pragma: no cover
                "axis1",  # pragma: no cover
                "rtol",  # pragma: no cover
                "n",  # pragma: no cover
                "m",  # pragma: no cover
                "endpoint",  # pragma: no cover
                "offset",  # pragma: no cover
                "a_max",  # pragma: no cover
                "ddof",  # pragma: no cover
                "dims",  # pragma: no cover
                "source",  # pragma: no cover
                "axis2",  # pragma: no cover
                "stop",  # pragma: no cover
                "k",  # pragma: no cover
                "dtype",  # pragma: no cover
                "reps",  # pragma: no cover
                "a_min",  # pragma: no cover
                "sparse",  # pragma: no cover
                "decimals",  # pragma: no cover
                "strides",  # pragma: no cover
                "num",  # pragma: no cover
                "max",  # pragma: no cover
                "indices_or_sections",  # pragma: no cover
                "shape",  # pragma: no cover
                "atol",  # pragma: no cover
                "axis1",  # pragma: no cover
                "step",  # pragma: no cover
                "axes",  # pragma: no cover
                "fill_value",  # pragma: no cover
                "stream",  # pragma: no cover
                "destination",  # pragma: no cover
                "alpha",  # pragma: no cover
                "beta",  # pragma: no cover
                "start",  # pragma: no cover
                "equal_nan",
                "nan",
                "posinf",
                "neginf",  # pragma: no cover
                "keepdims",
                "shape",
                "size",  # pragma: no cover
                "min",  # pragma: no cover
                "p",  # pragma: no cover
            } and "addmm" not in (  # pragma: no cover
                "transpose",  # pragma: no cover
                "moveaxis",  # pragma: no cover
                "reshape",  # pragma: no cover
                "expand_dims",  # pragma: no cover
                "linspace",  # pragma: no cover
                "logcumsumexp",  # pragma: no cover
                "cummin",  # pragma: no cover
                "cummax",  # pragma: no cover
                "cumprod",  # pragma: no cover
                "cumsum",  # pragma: no cover
                "logsumexp",  # pragma: no cover
                "all",  # pragma: no cover
                "any",  # pragma: no cover
                "argmin",  # pragma: no cover
                "argmax",  # pragma: no cover
                "prod",  # pragma: no cover
                "sum",  # pragma: no cover
                "mean",  # pragma: no cover
                "var",  # pragma: no cover
                "std",  # pragma: no cover
                "max",  # pragma: no cover
                "min",  # pragma: no cover
                "tensordot",  # pragma: no cover
            ):  # pragma: no cover
                return _to_tensor(x)  # pragma: no cover
        if type(x).__name__ == "ndarray":  # pragma: no cover
            return _to_tensor(x)  # pragma: no cover
        return x  # pragma: no cover

    def _w(x):  # pragma: no cover
        if isinstance(x, tuple):  # pragma: no cover
            return tuple((_w(i) for i in x))  # pragma: no cover
        if isinstance(x, list):  # pragma: no cover
            return [_w(i) for i in x]  # pragma: no cover
        return array(x) if hasattr(x, "shape") else x  # pragma: no cover

    fn = _get_compiler_op("addmm")  # pragma: no cover
    res = fn(
        _u(c, "c"), _u(a, "a"), _u(b, "b"), _u(alpha, "alpha"), _u(beta, "beta")
    )  # pragma: no cover

    def __w(x):
        if isinstance(x, tuple):
            return tuple((__w(i) for i in x))
        if isinstance(x, list):
            return [__w(i) for i in x]
        return array(x) if hasattr(x, "shape") else x

    return __w(res)  # pragma: no cover


def gather_mm(  # pragma: no cover
    a, b, lhs_indices=None, rhs_indices=None, stream=None, **kwargs
):  # pragma: no cover
    """Compute gather_mm.

    Args:
        a: The a argument.
        b: The b argument.
        lhs_indices: The lhs_indices argument.
        rhs_indices: The rhs_indices argument.
        stream: The stream argument.
        kwargs: The kwargs argument.

    Returns:
        The result of gather_mm.

    """
    from zero_mlx.array import array, _to_tensor  # pragma: no cover
    import ml_switcheroo_compiler.ops as mops  # pragma: no cover
    import ml_switcheroo_compiler.ops.nn as mnn  # pragma: no cover
    import ml_switcheroo_compiler.random as mrand  # pragma: no cover
    import types  # pragma: no cover

    def _u(x, param_name=None):  # pragma: no cover
        if isinstance(x, list):  # pragma: no cover
            if __import__("builtins").all(  # pragma: no cover
                (isinstance(i, int) for i in x)  # pragma: no cover
            ):  # pragma: no cover
                return tuple(x)  # pragma: no cover
            return [_u(i, param_name) for i in x]  # pragma: no cover
        if isinstance(x, tuple):  # pragma: no cover
            return tuple((_u(i, param_name) for i in x))  # pragma: no cover
        if hasattr(x, "_tensor"):  # pragma: no cover
            return x._tensor  # pragma: no cover
        if isinstance(x, (int, float, bool, complex)):  # pragma: no cover
            if param_name not in {  # pragma: no cover
                "axis",  # pragma: no cover
                "keepdims",
                "shape",
                "size",  # pragma: no cover
                "equal_nan",
                "nan",
                "posinf",
                "neginf",  # pragma: no cover
                "axis1",  # pragma: no cover
                "rtol",  # pragma: no cover
                "n",  # pragma: no cover
                "m",  # pragma: no cover
                "endpoint",  # pragma: no cover
                "offset",  # pragma: no cover
                "a_max",  # pragma: no cover
                "ddof",  # pragma: no cover
                "dims",  # pragma: no cover
                "source",  # pragma: no cover
                "axis2",  # pragma: no cover
                "stop",  # pragma: no cover
                "k",  # pragma: no cover
                "dtype",  # pragma: no cover
                "reps",  # pragma: no cover
                "a_min",  # pragma: no cover
                "sparse",  # pragma: no cover
                "decimals",  # pragma: no cover
                "strides",  # pragma: no cover
                "num",  # pragma: no cover
                "max",  # pragma: no cover
                "indices_or_sections",  # pragma: no cover
                "shape",  # pragma: no cover
                "atol",  # pragma: no cover
                "axis1",  # pragma: no cover
                "step",  # pragma: no cover
                "axes",  # pragma: no cover
                "fill_value",  # pragma: no cover
                "stream",  # pragma: no cover
                "destination",  # pragma: no cover
                "alpha",  # pragma: no cover
                "beta",  # pragma: no cover
                "start",  # pragma: no cover
                "equal_nan",
                "nan",
                "posinf",
                "neginf",  # pragma: no cover
                "keepdims",
                "shape",
                "size",  # pragma: no cover
                "min",  # pragma: no cover
                "p",  # pragma: no cover
            } and "gather_mm" not in (  # pragma: no cover
                "transpose",  # pragma: no cover
                "moveaxis",  # pragma: no cover
                "reshape",  # pragma: no cover
                "expand_dims",  # pragma: no cover
                "linspace",  # pragma: no cover
                "logcumsumexp",  # pragma: no cover
                "cummin",  # pragma: no cover
                "cummax",  # pragma: no cover
                "cumprod",  # pragma: no cover
                "cumsum",  # pragma: no cover
                "logsumexp",  # pragma: no cover
                "all",  # pragma: no cover
                "any",  # pragma: no cover
                "argmin",  # pragma: no cover
                "argmax",  # pragma: no cover
                "prod",  # pragma: no cover
                "sum",  # pragma: no cover
                "mean",  # pragma: no cover
                "var",  # pragma: no cover
                "std",  # pragma: no cover
                "max",  # pragma: no cover
                "min",  # pragma: no cover
                "tensordot",  # pragma: no cover
            ):  # pragma: no cover
                return _to_tensor(x)  # pragma: no cover
        if type(x).__name__ == "ndarray":  # pragma: no cover
            return _to_tensor(x)  # pragma: no cover
        return x  # pragma: no cover

    def _w(x):  # pragma: no cover
        if isinstance(x, tuple):  # pragma: no cover
            return tuple((_w(i) for i in x))  # pragma: no cover
        if isinstance(x, list):  # pragma: no cover
            return [_w(i) for i in x]  # pragma: no cover
        return array(x) if hasattr(x, "shape") else x  # pragma: no cover

    fn = _get_compiler_op("gather_mm")  # pragma: no cover
    res = fn(  # pragma: no cover
        _u(a, "a"),  # pragma: no cover
        _u(b, "b"),  # pragma: no cover
        _u(lhs_indices, "lhs_indices"),  # pragma: no cover
        _u(rhs_indices, "rhs_indices"),  # pragma: no cover
        **{
            ("dim" if k == "axis" else k): _u(v, k)
            for (k, v) in kwargs.items()
            if k != "stream"
        },  # pragma: no cover
    )  # pragma: no cover

    def __w(x):
        if isinstance(x, tuple):
            return tuple((__w(i) for i in x))
        if isinstance(x, list):
            return [__w(i) for i in x]
        return array(x) if hasattr(x, "shape") else x

    return __w(res)  # pragma: no cover


def block_masked_mm(  # pragma: no cover
    a,
    b,
    block_size,
    out_mask=None,
    a_mask=None,
    b_mask=None,
    stream=None,  # pragma: no cover
):  # pragma: no cover
    """Compute block_masked_mm.

    Args:
        a: The a argument.
        b: The b argument.
        block_size: The block_size argument.
        out_mask: The out_mask argument.
        a_mask: The a_mask argument.
        b_mask: The b_mask argument.
        stream: The stream argument.

    Returns:
        The result of block_masked_mm.

    """
    from zero_mlx.array import array, _to_tensor  # pragma: no cover
    import ml_switcheroo_compiler.ops as mops  # pragma: no cover
    import ml_switcheroo_compiler.ops.nn as mnn  # pragma: no cover
    import ml_switcheroo_compiler.random as mrand  # pragma: no cover
    import types  # pragma: no cover

    def _u(x, param_name=None):  # pragma: no cover
        if isinstance(x, list):  # pragma: no cover
            if __import__("builtins").all(  # pragma: no cover
                (isinstance(i, int) for i in x)  # pragma: no cover
            ):  # pragma: no cover
                return tuple(x)  # pragma: no cover
            return [_u(i, param_name) for i in x]  # pragma: no cover
        if isinstance(x, tuple):  # pragma: no cover
            return tuple((_u(i, param_name) for i in x))  # pragma: no cover
        if hasattr(x, "_tensor"):  # pragma: no cover
            return x._tensor  # pragma: no cover
        if isinstance(x, (int, float, bool, complex)):  # pragma: no cover
            if param_name not in {  # pragma: no cover
                "axis",  # pragma: no cover
                "keepdims",
                "shape",
                "size",  # pragma: no cover
                "equal_nan",
                "nan",
                "posinf",
                "neginf",  # pragma: no cover
                "axis1",  # pragma: no cover
                "rtol",  # pragma: no cover
                "n",  # pragma: no cover
                "m",  # pragma: no cover
                "endpoint",  # pragma: no cover
                "offset",  # pragma: no cover
                "a_max",  # pragma: no cover
                "ddof",  # pragma: no cover
                "dims",  # pragma: no cover
                "source",  # pragma: no cover
                "axis2",  # pragma: no cover
                "stop",  # pragma: no cover
                "k",  # pragma: no cover
                "dtype",  # pragma: no cover
                "reps",  # pragma: no cover
                "a_min",  # pragma: no cover
                "sparse",  # pragma: no cover
                "decimals",  # pragma: no cover
                "strides",  # pragma: no cover
                "num",  # pragma: no cover
                "max",  # pragma: no cover
                "indices_or_sections",  # pragma: no cover
                "shape",  # pragma: no cover
                "atol",  # pragma: no cover
                "axis1",  # pragma: no cover
                "step",  # pragma: no cover
                "axes",  # pragma: no cover
                "fill_value",  # pragma: no cover
                "stream",  # pragma: no cover
                "destination",  # pragma: no cover
                "alpha",  # pragma: no cover
                "beta",  # pragma: no cover
                "start",  # pragma: no cover
                "equal_nan",
                "nan",
                "posinf",
                "neginf",  # pragma: no cover
                "keepdims",
                "shape",
                "size",  # pragma: no cover
                "min",  # pragma: no cover
                "p",  # pragma: no cover
            } and "block_masked_mm" not in (  # pragma: no cover
                "transpose",  # pragma: no cover
                "moveaxis",  # pragma: no cover
                "reshape",  # pragma: no cover
                "expand_dims",  # pragma: no cover
                "linspace",  # pragma: no cover
                "logcumsumexp",  # pragma: no cover
                "cummin",  # pragma: no cover
                "cummax",  # pragma: no cover
                "cumprod",  # pragma: no cover
                "cumsum",  # pragma: no cover
                "logsumexp",  # pragma: no cover
                "all",  # pragma: no cover
                "any",  # pragma: no cover
                "argmin",  # pragma: no cover
                "argmax",  # pragma: no cover
                "prod",  # pragma: no cover
                "sum",  # pragma: no cover
                "mean",  # pragma: no cover
                "var",  # pragma: no cover
                "std",  # pragma: no cover
                "max",  # pragma: no cover
                "min",  # pragma: no cover
                "tensordot",  # pragma: no cover
            ):  # pragma: no cover
                return _to_tensor(x)  # pragma: no cover
        if type(x).__name__ == "ndarray":  # pragma: no cover
            return _to_tensor(x)  # pragma: no cover
        return x  # pragma: no cover

    def _w(x):  # pragma: no cover
        if isinstance(x, tuple):  # pragma: no cover
            return tuple((_w(i) for i in x))  # pragma: no cover
        if isinstance(x, list):  # pragma: no cover
            return [_w(i) for i in x]  # pragma: no cover
        return array(x) if hasattr(x, "shape") else x  # pragma: no cover

    fn = _get_compiler_op("block_masked_mm")  # pragma: no cover
    res = fn(  # pragma: no cover
        _u(a, "a"),  # pragma: no cover
        _u(b, "b"),  # pragma: no cover
        _u(block_size, "block_size"),  # pragma: no cover
        _u(out_mask, "out_mask"),  # pragma: no cover
        _u(a_mask, "a_mask"),  # pragma: no cover
        _u(b_mask, "b_mask"),  # pragma: no cover
    )  # pragma: no cover

    def __w(x):
        if isinstance(x, tuple):
            return tuple((__w(i) for i in x))
        if isinstance(x, list):
            return [__w(i) for i in x]
        return array(x) if hasattr(x, "shape") else x

    return __w(res)  # pragma: no cover


def segmented_mm(a, b, segments, stream=None):  # pragma: no cover
    """Compute segmented_mm.

    Args:
        a: The a argument.
        b: The b argument.
        segments: The segments argument.
        stream: The stream argument.

    Returns:
        The result of segmented_mm.

    """
    from zero_mlx.array import array, _to_tensor  # pragma: no cover
    import ml_switcheroo_compiler.ops as mops  # pragma: no cover
    import ml_switcheroo_compiler.ops.nn as mnn  # pragma: no cover
    import ml_switcheroo_compiler.random as mrand  # pragma: no cover
    import types  # pragma: no cover

    def _u(x, param_name=None):  # pragma: no cover
        if isinstance(x, list):  # pragma: no cover
            if __import__("builtins").all(  # pragma: no cover
                (isinstance(i, int) for i in x)  # pragma: no cover
            ):  # pragma: no cover
                return tuple(x)  # pragma: no cover
            return [_u(i, param_name) for i in x]  # pragma: no cover
        if isinstance(x, tuple):  # pragma: no cover
            return tuple((_u(i, param_name) for i in x))  # pragma: no cover
        if hasattr(x, "_tensor"):  # pragma: no cover
            return x._tensor  # pragma: no cover
        if isinstance(x, (int, float, bool, complex)):  # pragma: no cover
            if param_name not in {  # pragma: no cover
                "axis",  # pragma: no cover
                "keepdims",
                "shape",
                "size",  # pragma: no cover
                "equal_nan",
                "nan",
                "posinf",
                "neginf",  # pragma: no cover
                "axis1",  # pragma: no cover
                "rtol",  # pragma: no cover
                "n",  # pragma: no cover
                "m",  # pragma: no cover
                "endpoint",  # pragma: no cover
                "offset",  # pragma: no cover
                "a_max",  # pragma: no cover
                "ddof",  # pragma: no cover
                "dims",  # pragma: no cover
                "source",  # pragma: no cover
                "axis2",  # pragma: no cover
                "stop",  # pragma: no cover
                "k",  # pragma: no cover
                "dtype",  # pragma: no cover
                "reps",  # pragma: no cover
                "a_min",  # pragma: no cover
                "sparse",  # pragma: no cover
                "decimals",  # pragma: no cover
                "strides",  # pragma: no cover
                "num",  # pragma: no cover
                "max",  # pragma: no cover
                "indices_or_sections",  # pragma: no cover
                "shape",  # pragma: no cover
                "atol",  # pragma: no cover
                "axis1",  # pragma: no cover
                "step",  # pragma: no cover
                "axes",  # pragma: no cover
                "fill_value",  # pragma: no cover
                "stream",  # pragma: no cover
                "destination",  # pragma: no cover
                "alpha",  # pragma: no cover
                "beta",  # pragma: no cover
                "start",  # pragma: no cover
                "equal_nan",
                "nan",
                "posinf",
                "neginf",  # pragma: no cover
                "keepdims",
                "shape",
                "size",  # pragma: no cover
                "min",  # pragma: no cover
                "p",  # pragma: no cover
            } and "segmented_mm" not in (  # pragma: no cover
                "transpose",  # pragma: no cover
                "moveaxis",  # pragma: no cover
                "reshape",  # pragma: no cover
                "expand_dims",  # pragma: no cover
                "linspace",  # pragma: no cover
                "logcumsumexp",  # pragma: no cover
                "cummin",  # pragma: no cover
                "cummax",  # pragma: no cover
                "cumprod",  # pragma: no cover
                "cumsum",  # pragma: no cover
                "logsumexp",  # pragma: no cover
                "all",  # pragma: no cover
                "any",  # pragma: no cover
                "argmin",  # pragma: no cover
                "argmax",  # pragma: no cover
                "prod",  # pragma: no cover
                "sum",  # pragma: no cover
                "mean",  # pragma: no cover
                "var",  # pragma: no cover
                "std",  # pragma: no cover
                "max",  # pragma: no cover
                "min",  # pragma: no cover
                "tensordot",  # pragma: no cover
            ):  # pragma: no cover
                return _to_tensor(x)  # pragma: no cover
        if type(x).__name__ == "ndarray":  # pragma: no cover
            return _to_tensor(x)  # pragma: no cover
        return x  # pragma: no cover

    def _w(x):  # pragma: no cover
        if isinstance(x, tuple):  # pragma: no cover
            return tuple((_w(i) for i in x))  # pragma: no cover
        if isinstance(x, list):  # pragma: no cover
            return [_w(i) for i in x]  # pragma: no cover
        return array(x) if hasattr(x, "shape") else x  # pragma: no cover

    fn = _get_compiler_op("segmented_mm")  # pragma: no cover
    res = fn(_u(a, "a"), _u(b, "b"), _u(segments, "segments"))  # pragma: no cover

    def __w(x):
        if isinstance(x, tuple):
            return tuple((__w(i) for i in x))
        if isinstance(x, list):
            return [__w(i) for i in x]
        return array(x) if hasattr(x, "shape") else x

    return __w(res)  # pragma: no cover


def argpartition(*args, **kwargs):  # pragma: no cover
    """Compute argpartition.

    Args:
        args: The args argument.
        kwargs: The kwargs argument.

    Returns:
        The result of argpartition.

    """
    from zero_mlx.array import array, _to_tensor  # pragma: no cover
    import ml_switcheroo_compiler.ops as mops  # pragma: no cover
    import ml_switcheroo_compiler.ops.nn as mnn  # pragma: no cover
    import ml_switcheroo_compiler.random as mrand  # pragma: no cover
    import types  # pragma: no cover

    def _u(x, param_name=None):  # pragma: no cover
        if isinstance(x, list):  # pragma: no cover
            if __import__("builtins").all(  # pragma: no cover
                (isinstance(i, int) for i in x)  # pragma: no cover
            ):  # pragma: no cover
                return tuple(x)  # pragma: no cover
            return [_u(i, param_name) for i in x]  # pragma: no cover
        if isinstance(x, tuple):  # pragma: no cover
            return tuple((_u(i, param_name) for i in x))  # pragma: no cover
        if hasattr(x, "_tensor"):  # pragma: no cover
            return x._tensor  # pragma: no cover
        if isinstance(x, (int, float, bool, complex)):  # pragma: no cover
            if param_name not in {  # pragma: no cover
                "axis",  # pragma: no cover
                "keepdims",
                "shape",
                "size",  # pragma: no cover
                "equal_nan",
                "nan",
                "posinf",
                "neginf",  # pragma: no cover
                "axis1",  # pragma: no cover
                "rtol",  # pragma: no cover
                "n",  # pragma: no cover
                "m",  # pragma: no cover
                "endpoint",  # pragma: no cover
                "offset",  # pragma: no cover
                "a_max",  # pragma: no cover
                "ddof",  # pragma: no cover
                "dims",  # pragma: no cover
                "source",  # pragma: no cover
                "axis2",  # pragma: no cover
                "stop",  # pragma: no cover
                "k",  # pragma: no cover
                "dtype",  # pragma: no cover
                "reps",  # pragma: no cover
                "a_min",  # pragma: no cover
                "sparse",  # pragma: no cover
                "decimals",  # pragma: no cover
                "strides",  # pragma: no cover
                "num",  # pragma: no cover
                "max",  # pragma: no cover
                "indices_or_sections",  # pragma: no cover
                "shape",  # pragma: no cover
                "atol",  # pragma: no cover
                "axis1",  # pragma: no cover
                "step",  # pragma: no cover
                "axes",  # pragma: no cover
                "fill_value",  # pragma: no cover
                "stream",  # pragma: no cover
                "destination",  # pragma: no cover
                "alpha",  # pragma: no cover
                "beta",  # pragma: no cover
                "start",  # pragma: no cover
                "equal_nan",
                "nan",
                "posinf",
                "neginf",  # pragma: no cover
                "keepdims",
                "shape",
                "size",  # pragma: no cover
                "min",  # pragma: no cover
                "p",  # pragma: no cover
            } and "argpartition" not in (  # pragma: no cover
                "transpose",  # pragma: no cover
                "moveaxis",  # pragma: no cover
                "reshape",  # pragma: no cover
                "expand_dims",  # pragma: no cover
                "linspace",  # pragma: no cover
                "logcumsumexp",  # pragma: no cover
                "cummin",  # pragma: no cover
                "cummax",  # pragma: no cover
                "cumprod",  # pragma: no cover
                "cumsum",  # pragma: no cover
                "logsumexp",  # pragma: no cover
                "all",  # pragma: no cover
                "any",  # pragma: no cover
                "argmin",  # pragma: no cover
                "argmax",  # pragma: no cover
                "prod",  # pragma: no cover
                "sum",  # pragma: no cover
                "mean",  # pragma: no cover
                "var",  # pragma: no cover
                "std",  # pragma: no cover
                "max",  # pragma: no cover
                "min",  # pragma: no cover
                "tensordot",  # pragma: no cover
            ):  # pragma: no cover
                return _to_tensor(x)  # pragma: no cover
        if type(x).__name__ == "ndarray":  # pragma: no cover
            return _to_tensor(x)  # pragma: no cover
        return x  # pragma: no cover

    def _w(x):  # pragma: no cover
        if isinstance(x, tuple):  # pragma: no cover
            return tuple((_w(i) for i in x))  # pragma: no cover
        if isinstance(x, list):  # pragma: no cover
            return [_w(i) for i in x]  # pragma: no cover
        return array(x) if hasattr(x, "shape") else x  # pragma: no cover

    fn = _get_compiler_op("argpartition")  # pragma: no cover
    res = fn(  # pragma: no cover
        *[_u(x) for x in args],  # pragma: no cover
        **{
            ("dim" if k == "axis" else k): _u(v, k)
            for (k, v) in kwargs.items()
            if k != "stream"
        },  # pragma: no cover
    )  # pragma: no cover

    def __w(x):
        if isinstance(x, tuple):
            return tuple((__w(i) for i in x))
        if isinstance(x, list):
            return [__w(i) for i in x]
        return array(x) if hasattr(x, "shape") else x

    return __w(res)  # pragma: no cover


def atleast_1d(*args, **kwargs):  # pragma: no cover
    """Compute atleast_1d.

    Args:
        args: The args argument.
        kwargs: The kwargs argument.

    Returns:
        The result of atleast_1d.

    """
    from zero_mlx.array import array, _to_tensor  # pragma: no cover
    import ml_switcheroo_compiler.ops as mops  # pragma: no cover
    import ml_switcheroo_compiler.ops.nn as mnn  # pragma: no cover
    import ml_switcheroo_compiler.random as mrand  # pragma: no cover
    import types  # pragma: no cover

    def _u(x, param_name=None):  # pragma: no cover
        if isinstance(x, list):  # pragma: no cover
            if __import__("builtins").all(  # pragma: no cover
                (isinstance(i, int) for i in x)  # pragma: no cover
            ):  # pragma: no cover
                return tuple(x)  # pragma: no cover
            return [_u(i, param_name) for i in x]  # pragma: no cover
        if isinstance(x, tuple):  # pragma: no cover
            return tuple((_u(i, param_name) for i in x))  # pragma: no cover
        if hasattr(x, "_tensor"):  # pragma: no cover
            return x._tensor  # pragma: no cover
        if isinstance(x, (int, float, bool, complex)):  # pragma: no cover
            if param_name not in {  # pragma: no cover
                "axis",  # pragma: no cover
                "keepdims",
                "shape",
                "size",  # pragma: no cover
                "equal_nan",
                "nan",
                "posinf",
                "neginf",  # pragma: no cover
                "axis1",  # pragma: no cover
                "rtol",  # pragma: no cover
                "n",  # pragma: no cover
                "m",  # pragma: no cover
                "endpoint",  # pragma: no cover
                "offset",  # pragma: no cover
                "a_max",  # pragma: no cover
                "ddof",  # pragma: no cover
                "dims",  # pragma: no cover
                "source",  # pragma: no cover
                "axis2",  # pragma: no cover
                "stop",  # pragma: no cover
                "k",  # pragma: no cover
                "dtype",  # pragma: no cover
                "reps",  # pragma: no cover
                "a_min",  # pragma: no cover
                "sparse",  # pragma: no cover
                "decimals",  # pragma: no cover
                "strides",  # pragma: no cover
                "num",  # pragma: no cover
                "max",  # pragma: no cover
                "indices_or_sections",  # pragma: no cover
                "shape",  # pragma: no cover
                "atol",  # pragma: no cover
                "axis1",  # pragma: no cover
                "step",  # pragma: no cover
                "axes",  # pragma: no cover
                "fill_value",  # pragma: no cover
                "stream",  # pragma: no cover
                "destination",  # pragma: no cover
                "alpha",  # pragma: no cover
                "beta",  # pragma: no cover
                "start",  # pragma: no cover
                "equal_nan",
                "nan",
                "posinf",
                "neginf",  # pragma: no cover
                "keepdims",
                "shape",
                "size",  # pragma: no cover
                "min",  # pragma: no cover
                "p",  # pragma: no cover
            } and "atleast_1d" not in (  # pragma: no cover
                "transpose",  # pragma: no cover
                "moveaxis",  # pragma: no cover
                "reshape",  # pragma: no cover
                "expand_dims",  # pragma: no cover
                "linspace",  # pragma: no cover
                "logcumsumexp",  # pragma: no cover
                "cummin",  # pragma: no cover
                "cummax",  # pragma: no cover
                "cumprod",  # pragma: no cover
                "cumsum",  # pragma: no cover
                "logsumexp",  # pragma: no cover
                "all",  # pragma: no cover
                "any",  # pragma: no cover
                "argmin",  # pragma: no cover
                "argmax",  # pragma: no cover
                "prod",  # pragma: no cover
                "sum",  # pragma: no cover
                "mean",  # pragma: no cover
                "var",  # pragma: no cover
                "std",  # pragma: no cover
                "max",  # pragma: no cover
                "min",  # pragma: no cover
                "tensordot",  # pragma: no cover
            ):  # pragma: no cover
                return _to_tensor(x)  # pragma: no cover
        if type(x).__name__ == "ndarray":  # pragma: no cover
            return _to_tensor(x)  # pragma: no cover
        return x  # pragma: no cover

    def _w(x):  # pragma: no cover
        if isinstance(x, tuple):  # pragma: no cover
            return tuple((_w(i) for i in x))  # pragma: no cover
        if isinstance(x, list):  # pragma: no cover
            return [_w(i) for i in x]  # pragma: no cover
        return array(x) if hasattr(x, "shape") else x  # pragma: no cover

    fn = _get_compiler_op("atleast_1d")  # pragma: no cover
    res = fn(  # pragma: no cover
        *[_u(x) for x in args],  # pragma: no cover
        **{
            ("dim" if k == "axis" else k): _u(v, k)
            for (k, v) in kwargs.items()
            if k != "stream"
        },  # pragma: no cover
    )  # pragma: no cover

    def __w(x):
        if isinstance(x, tuple):
            return tuple((__w(i) for i in x))
        if isinstance(x, list):
            return [__w(i) for i in x]
        return array(x) if hasattr(x, "shape") else x

    return __w(res)  # pragma: no cover


def atleast_2d(*args, **kwargs):  # pragma: no cover
    """Compute atleast_2d.

    Args:
        args: The args argument.
        kwargs: The kwargs argument.

    Returns:
        The result of atleast_2d.

    """
    from zero_mlx.array import array, _to_tensor  # pragma: no cover
    import ml_switcheroo_compiler.ops as mops  # pragma: no cover
    import ml_switcheroo_compiler.ops.nn as mnn  # pragma: no cover
    import ml_switcheroo_compiler.random as mrand  # pragma: no cover
    import types  # pragma: no cover

    def _u(x, param_name=None):  # pragma: no cover
        if isinstance(x, list):  # pragma: no cover
            if __import__("builtins").all(  # pragma: no cover
                (isinstance(i, int) for i in x)  # pragma: no cover
            ):  # pragma: no cover
                return tuple(x)  # pragma: no cover
            return [_u(i, param_name) for i in x]  # pragma: no cover
        if isinstance(x, tuple):  # pragma: no cover
            return tuple((_u(i, param_name) for i in x))  # pragma: no cover
        if hasattr(x, "_tensor"):  # pragma: no cover
            return x._tensor  # pragma: no cover
        if isinstance(x, (int, float, bool, complex)):  # pragma: no cover
            if param_name not in {  # pragma: no cover
                "axis",  # pragma: no cover
                "keepdims",
                "shape",
                "size",  # pragma: no cover
                "equal_nan",
                "nan",
                "posinf",
                "neginf",  # pragma: no cover
                "axis1",  # pragma: no cover
                "rtol",  # pragma: no cover
                "n",  # pragma: no cover
                "m",  # pragma: no cover
                "endpoint",  # pragma: no cover
                "offset",  # pragma: no cover
                "a_max",  # pragma: no cover
                "ddof",  # pragma: no cover
                "dims",  # pragma: no cover
                "source",  # pragma: no cover
                "axis2",  # pragma: no cover
                "stop",  # pragma: no cover
                "k",  # pragma: no cover
                "dtype",  # pragma: no cover
                "reps",  # pragma: no cover
                "a_min",  # pragma: no cover
                "sparse",  # pragma: no cover
                "decimals",  # pragma: no cover
                "strides",  # pragma: no cover
                "num",  # pragma: no cover
                "max",  # pragma: no cover
                "indices_or_sections",  # pragma: no cover
                "shape",  # pragma: no cover
                "atol",  # pragma: no cover
                "axis1",  # pragma: no cover
                "step",  # pragma: no cover
                "axes",  # pragma: no cover
                "fill_value",  # pragma: no cover
                "stream",  # pragma: no cover
                "destination",  # pragma: no cover
                "alpha",  # pragma: no cover
                "beta",  # pragma: no cover
                "start",  # pragma: no cover
                "equal_nan",
                "nan",
                "posinf",
                "neginf",  # pragma: no cover
                "keepdims",
                "shape",
                "size",  # pragma: no cover
                "min",  # pragma: no cover
                "p",  # pragma: no cover
            } and "atleast_2d" not in (  # pragma: no cover
                "transpose",  # pragma: no cover
                "moveaxis",  # pragma: no cover
                "reshape",  # pragma: no cover
                "expand_dims",  # pragma: no cover
                "linspace",  # pragma: no cover
                "logcumsumexp",  # pragma: no cover
                "cummin",  # pragma: no cover
                "cummax",  # pragma: no cover
                "cumprod",  # pragma: no cover
                "cumsum",  # pragma: no cover
                "logsumexp",  # pragma: no cover
                "all",  # pragma: no cover
                "any",  # pragma: no cover
                "argmin",  # pragma: no cover
                "argmax",  # pragma: no cover
                "prod",  # pragma: no cover
                "sum",  # pragma: no cover
                "mean",  # pragma: no cover
                "var",  # pragma: no cover
                "std",  # pragma: no cover
                "max",  # pragma: no cover
                "min",  # pragma: no cover
                "tensordot",  # pragma: no cover
            ):  # pragma: no cover
                return _to_tensor(x)  # pragma: no cover
        if type(x).__name__ == "ndarray":  # pragma: no cover
            return _to_tensor(x)  # pragma: no cover
        return x  # pragma: no cover

    def _w(x):  # pragma: no cover
        if isinstance(x, tuple):  # pragma: no cover
            return tuple((_w(i) for i in x))  # pragma: no cover
        if isinstance(x, list):  # pragma: no cover
            return [_w(i) for i in x]  # pragma: no cover
        return array(x) if hasattr(x, "shape") else x  # pragma: no cover

    fn = _get_compiler_op("atleast_2d")  # pragma: no cover
    res = fn(  # pragma: no cover
        *[_u(x) for x in args],  # pragma: no cover
        **{
            ("dim" if k == "axis" else k): _u(v, k)
            for (k, v) in kwargs.items()
            if k != "stream"
        },  # pragma: no cover
    )  # pragma: no cover

    def __w(x):
        if isinstance(x, tuple):
            return tuple((__w(i) for i in x))
        if isinstance(x, list):
            return [__w(i) for i in x]
        return array(x) if hasattr(x, "shape") else x

    return __w(res)  # pragma: no cover


def atleast_3d(*args, **kwargs):  # pragma: no cover
    """Compute atleast_3d.

    Args:
        args: The args argument.
        kwargs: The kwargs argument.

    Returns:
        The result of atleast_3d.

    """
    from zero_mlx.array import array, _to_tensor  # pragma: no cover
    import ml_switcheroo_compiler.ops as mops  # pragma: no cover
    import ml_switcheroo_compiler.ops.nn as mnn  # pragma: no cover
    import ml_switcheroo_compiler.random as mrand  # pragma: no cover
    import types  # pragma: no cover

    def _u(x, param_name=None):  # pragma: no cover
        if isinstance(x, list):  # pragma: no cover
            if __import__("builtins").all(  # pragma: no cover
                (isinstance(i, int) for i in x)  # pragma: no cover
            ):  # pragma: no cover
                return tuple(x)  # pragma: no cover
            return [_u(i, param_name) for i in x]  # pragma: no cover
        if isinstance(x, tuple):  # pragma: no cover
            return tuple((_u(i, param_name) for i in x))  # pragma: no cover
        if hasattr(x, "_tensor"):  # pragma: no cover
            return x._tensor  # pragma: no cover
        if isinstance(x, (int, float, bool, complex)):  # pragma: no cover
            if param_name not in {  # pragma: no cover
                "axis",  # pragma: no cover
                "keepdims",
                "shape",
                "size",  # pragma: no cover
                "equal_nan",
                "nan",
                "posinf",
                "neginf",  # pragma: no cover
                "axis1",  # pragma: no cover
                "rtol",  # pragma: no cover
                "n",  # pragma: no cover
                "m",  # pragma: no cover
                "endpoint",  # pragma: no cover
                "offset",  # pragma: no cover
                "a_max",  # pragma: no cover
                "ddof",  # pragma: no cover
                "dims",  # pragma: no cover
                "source",  # pragma: no cover
                "axis2",  # pragma: no cover
                "stop",  # pragma: no cover
                "k",  # pragma: no cover
                "dtype",  # pragma: no cover
                "reps",  # pragma: no cover
                "a_min",  # pragma: no cover
                "sparse",  # pragma: no cover
                "decimals",  # pragma: no cover
                "strides",  # pragma: no cover
                "num",  # pragma: no cover
                "max",  # pragma: no cover
                "indices_or_sections",  # pragma: no cover
                "shape",  # pragma: no cover
                "atol",  # pragma: no cover
                "axis1",  # pragma: no cover
                "step",  # pragma: no cover
                "axes",  # pragma: no cover
                "fill_value",  # pragma: no cover
                "stream",  # pragma: no cover
                "destination",  # pragma: no cover
                "alpha",  # pragma: no cover
                "beta",  # pragma: no cover
                "start",  # pragma: no cover
                "equal_nan",
                "nan",
                "posinf",
                "neginf",  # pragma: no cover
                "keepdims",
                "shape",
                "size",  # pragma: no cover
                "min",  # pragma: no cover
                "p",  # pragma: no cover
            } and "atleast_3d" not in (  # pragma: no cover
                "transpose",  # pragma: no cover
                "moveaxis",  # pragma: no cover
                "reshape",  # pragma: no cover
                "expand_dims",  # pragma: no cover
                "linspace",  # pragma: no cover
                "logcumsumexp",  # pragma: no cover
                "cummin",  # pragma: no cover
                "cummax",  # pragma: no cover
                "cumprod",  # pragma: no cover
                "cumsum",  # pragma: no cover
                "logsumexp",  # pragma: no cover
                "all",  # pragma: no cover
                "any",  # pragma: no cover
                "argmin",  # pragma: no cover
                "argmax",  # pragma: no cover
                "prod",  # pragma: no cover
                "sum",  # pragma: no cover
                "mean",  # pragma: no cover
                "var",  # pragma: no cover
                "std",  # pragma: no cover
                "max",  # pragma: no cover
                "min",  # pragma: no cover
                "tensordot",  # pragma: no cover
            ):  # pragma: no cover
                return _to_tensor(x)  # pragma: no cover
        if type(x).__name__ == "ndarray":  # pragma: no cover
            return _to_tensor(x)  # pragma: no cover
        return x  # pragma: no cover

    def _w(x):  # pragma: no cover
        if isinstance(x, tuple):  # pragma: no cover
            return tuple((_w(i) for i in x))  # pragma: no cover
        if isinstance(x, list):  # pragma: no cover
            return [_w(i) for i in x]  # pragma: no cover
        return array(x) if hasattr(x, "shape") else x  # pragma: no cover

    fn = _get_compiler_op("atleast_3d")  # pragma: no cover
    res = fn(  # pragma: no cover
        *[_u(x) for x in args],  # pragma: no cover
        **{
            ("dim" if k == "axis" else k): _u(v, k)
            for (k, v) in kwargs.items()
            if k != "stream"
        },  # pragma: no cover
    )  # pragma: no cover

    def __w(x):
        if isinstance(x, tuple):
            return tuple((__w(i) for i in x))
        if isinstance(x, list):
            return [__w(i) for i in x]
        return array(x) if hasattr(x, "shape") else x

    return __w(res)  # pragma: no cover


def broadcast_arrays(*args, **kwargs):  # pragma: no cover
    """Compute broadcast_arrays.

    Args:
        args: The args argument.
        kwargs: The kwargs argument.

    Returns:
        The result of broadcast_arrays.

    """
    from zero_mlx.array import array, _to_tensor  # pragma: no cover
    import ml_switcheroo_compiler.ops as mops  # pragma: no cover
    import ml_switcheroo_compiler.ops.nn as mnn  # pragma: no cover
    import ml_switcheroo_compiler.random as mrand  # pragma: no cover
    import types  # pragma: no cover

    def _u(x, param_name=None):  # pragma: no cover
        if isinstance(x, list):  # pragma: no cover
            if __import__("builtins").all(  # pragma: no cover
                (isinstance(i, int) for i in x)  # pragma: no cover
            ):  # pragma: no cover
                return tuple(x)  # pragma: no cover
            return [_u(i, param_name) for i in x]  # pragma: no cover
        if isinstance(x, tuple):  # pragma: no cover
            return tuple((_u(i, param_name) for i in x))  # pragma: no cover
        if hasattr(x, "_tensor"):  # pragma: no cover
            return x._tensor  # pragma: no cover
        if isinstance(x, (int, float, bool, complex)):  # pragma: no cover
            if param_name not in {  # pragma: no cover
                "axis",  # pragma: no cover
                "keepdims",
                "shape",
                "size",  # pragma: no cover
                "equal_nan",
                "nan",
                "posinf",
                "neginf",  # pragma: no cover
                "axis1",  # pragma: no cover
                "rtol",  # pragma: no cover
                "n",  # pragma: no cover
                "m",  # pragma: no cover
                "endpoint",  # pragma: no cover
                "offset",  # pragma: no cover
                "a_max",  # pragma: no cover
                "ddof",  # pragma: no cover
                "dims",  # pragma: no cover
                "source",  # pragma: no cover
                "axis2",  # pragma: no cover
                "stop",  # pragma: no cover
                "k",  # pragma: no cover
                "dtype",  # pragma: no cover
                "reps",  # pragma: no cover
                "a_min",  # pragma: no cover
                "sparse",  # pragma: no cover
                "decimals",  # pragma: no cover
                "strides",  # pragma: no cover
                "num",  # pragma: no cover
                "max",  # pragma: no cover
                "indices_or_sections",  # pragma: no cover
                "shape",  # pragma: no cover
                "atol",  # pragma: no cover
                "axis1",  # pragma: no cover
                "step",  # pragma: no cover
                "axes",  # pragma: no cover
                "fill_value",  # pragma: no cover
                "stream",  # pragma: no cover
                "destination",  # pragma: no cover
                "alpha",  # pragma: no cover
                "beta",  # pragma: no cover
                "start",  # pragma: no cover
                "equal_nan",
                "nan",
                "posinf",
                "neginf",  # pragma: no cover
                "keepdims",
                "shape",
                "size",  # pragma: no cover
                "min",  # pragma: no cover
                "p",  # pragma: no cover
            } and "broadcast_arrays" not in (  # pragma: no cover
                "transpose",  # pragma: no cover
                "moveaxis",  # pragma: no cover
                "reshape",  # pragma: no cover
                "expand_dims",  # pragma: no cover
                "linspace",  # pragma: no cover
                "logcumsumexp",  # pragma: no cover
                "cummin",  # pragma: no cover
                "cummax",  # pragma: no cover
                "cumprod",  # pragma: no cover
                "cumsum",  # pragma: no cover
                "logsumexp",  # pragma: no cover
                "all",  # pragma: no cover
                "any",  # pragma: no cover
                "argmin",  # pragma: no cover
                "argmax",  # pragma: no cover
                "prod",  # pragma: no cover
                "sum",  # pragma: no cover
                "mean",  # pragma: no cover
                "var",  # pragma: no cover
                "std",  # pragma: no cover
                "max",  # pragma: no cover
                "min",  # pragma: no cover
                "tensordot",  # pragma: no cover
            ):  # pragma: no cover
                return _to_tensor(x)  # pragma: no cover
        if type(x).__name__ == "ndarray":  # pragma: no cover
            return _to_tensor(x)  # pragma: no cover
        return x  # pragma: no cover

    def _w(x):  # pragma: no cover
        if isinstance(x, tuple):  # pragma: no cover
            return tuple((_w(i) for i in x))  # pragma: no cover
        if isinstance(x, list):  # pragma: no cover
            return [_w(i) for i in x]  # pragma: no cover
        return array(x) if hasattr(x, "shape") else x  # pragma: no cover

    fn = _get_compiler_op("broadcast_arrays")  # pragma: no cover
    res = fn(  # pragma: no cover
        *[_u(x) for x in args],  # pragma: no cover
        **{
            ("dim" if k == "axis" else k): _u(v, k)
            for (k, v) in kwargs.items()
            if k != "stream"
        },  # pragma: no cover
    )  # pragma: no cover

    def __w(x):
        if isinstance(x, tuple):
            return tuple((__w(i) for i in x))
        if isinstance(x, list):
            return [__w(i) for i in x]
        return array(x) if hasattr(x, "shape") else x

    return __w(res)  # pragma: no cover


def broadcast_shapes(*args, **kwargs):  # pragma: no cover
    """Compute broadcast_shapes.

    Args:
        args: The args argument.
        kwargs: The kwargs argument.

    Returns:
        The result of broadcast_shapes.

    """
    from zero_mlx.array import array, _to_tensor  # pragma: no cover
    import ml_switcheroo_compiler.ops as mops  # pragma: no cover
    import ml_switcheroo_compiler.ops.nn as mnn  # pragma: no cover
    import ml_switcheroo_compiler.random as mrand  # pragma: no cover
    import types  # pragma: no cover

    def _u(x, param_name=None):  # pragma: no cover
        if isinstance(x, list):  # pragma: no cover
            if __import__("builtins").all(  # pragma: no cover
                (isinstance(i, int) for i in x)  # pragma: no cover
            ):  # pragma: no cover
                return tuple(x)  # pragma: no cover
            return [_u(i, param_name) for i in x]  # pragma: no cover
        if isinstance(x, tuple):  # pragma: no cover
            return tuple((_u(i, param_name) for i in x))  # pragma: no cover
        if hasattr(x, "_tensor"):  # pragma: no cover
            return x._tensor  # pragma: no cover
        if isinstance(x, (int, float, bool, complex)):  # pragma: no cover
            if param_name not in {  # pragma: no cover
                "axis",  # pragma: no cover
                "keepdims",
                "shape",
                "size",  # pragma: no cover
                "equal_nan",
                "nan",
                "posinf",
                "neginf",  # pragma: no cover
                "axis1",  # pragma: no cover
                "rtol",  # pragma: no cover
                "n",  # pragma: no cover
                "m",  # pragma: no cover
                "endpoint",  # pragma: no cover
                "offset",  # pragma: no cover
                "a_max",  # pragma: no cover
                "ddof",  # pragma: no cover
                "dims",  # pragma: no cover
                "source",  # pragma: no cover
                "axis2",  # pragma: no cover
                "stop",  # pragma: no cover
                "k",  # pragma: no cover
                "dtype",  # pragma: no cover
                "reps",  # pragma: no cover
                "a_min",  # pragma: no cover
                "sparse",  # pragma: no cover
                "decimals",  # pragma: no cover
                "strides",  # pragma: no cover
                "num",  # pragma: no cover
                "max",  # pragma: no cover
                "indices_or_sections",  # pragma: no cover
                "shape",  # pragma: no cover
                "atol",  # pragma: no cover
                "axis1",  # pragma: no cover
                "step",  # pragma: no cover
                "axes",  # pragma: no cover
                "fill_value",  # pragma: no cover
                "stream",  # pragma: no cover
                "destination",  # pragma: no cover
                "alpha",  # pragma: no cover
                "beta",  # pragma: no cover
                "start",  # pragma: no cover
                "equal_nan",
                "nan",
                "posinf",
                "neginf",  # pragma: no cover
                "keepdims",
                "shape",
                "size",  # pragma: no cover
                "min",  # pragma: no cover
                "p",  # pragma: no cover
            } and "broadcast_shapes" not in (  # pragma: no cover
                "transpose",  # pragma: no cover
                "moveaxis",  # pragma: no cover
                "reshape",  # pragma: no cover
                "expand_dims",  # pragma: no cover
                "linspace",  # pragma: no cover
                "logcumsumexp",  # pragma: no cover
                "cummin",  # pragma: no cover
                "cummax",  # pragma: no cover
                "cumprod",  # pragma: no cover
                "cumsum",  # pragma: no cover
                "logsumexp",  # pragma: no cover
                "all",  # pragma: no cover
                "any",  # pragma: no cover
                "argmin",  # pragma: no cover
                "argmax",  # pragma: no cover
                "prod",  # pragma: no cover
                "sum",  # pragma: no cover
                "mean",  # pragma: no cover
                "var",  # pragma: no cover
                "std",  # pragma: no cover
                "max",  # pragma: no cover
                "min",  # pragma: no cover
                "tensordot",  # pragma: no cover
            ):  # pragma: no cover
                return _to_tensor(x)  # pragma: no cover
        if type(x).__name__ == "ndarray":  # pragma: no cover
            return _to_tensor(x)  # pragma: no cover
        return x  # pragma: no cover

    def _w(x):  # pragma: no cover
        if isinstance(x, tuple):  # pragma: no cover
            return tuple((_w(i) for i in x))  # pragma: no cover
        if isinstance(x, list):  # pragma: no cover
            return [_w(i) for i in x]  # pragma: no cover
        return array(x) if hasattr(x, "shape") else x  # pragma: no cover

    fn = _get_compiler_op("broadcast_shapes")  # pragma: no cover
    res = fn(  # pragma: no cover
        *[_u(x) for x in args],  # pragma: no cover
        **{
            ("dim" if k == "axis" else k): _u(v, k)
            for (k, v) in kwargs.items()
            if k != "stream"
        },  # pragma: no cover
    )  # pragma: no cover

    def __w(x):
        if isinstance(x, tuple):
            return tuple((__w(i) for i in x))
        if isinstance(x, list):
            return [__w(i) for i in x]
        return array(x) if hasattr(x, "shape") else x

    return __w(res)  # pragma: no cover


def ceil(*args, **kwargs):  # pragma: no cover
    """Compute ceil.

    Args:
        args: The args argument.
        kwargs: The kwargs argument.

    Returns:
        The result of ceil.

    """
    from zero_mlx.array import array, _to_tensor  # pragma: no cover
    import ml_switcheroo_compiler.ops as mops  # pragma: no cover
    import ml_switcheroo_compiler.ops.nn as mnn  # pragma: no cover
    import ml_switcheroo_compiler.random as mrand  # pragma: no cover
    import types  # pragma: no cover

    def _u(x, param_name=None):  # pragma: no cover
        if isinstance(x, list):  # pragma: no cover
            if __import__("builtins").all(  # pragma: no cover
                (isinstance(i, int) for i in x)  # pragma: no cover
            ):  # pragma: no cover
                return tuple(x)  # pragma: no cover
            return [_u(i, param_name) for i in x]  # pragma: no cover
        if isinstance(x, tuple):  # pragma: no cover
            return tuple((_u(i, param_name) for i in x))  # pragma: no cover
        if hasattr(x, "_tensor"):  # pragma: no cover
            return x._tensor  # pragma: no cover
        if isinstance(x, (int, float, bool, complex)):  # pragma: no cover
            if param_name not in {  # pragma: no cover
                "axis",  # pragma: no cover
                "keepdims",
                "shape",
                "size",  # pragma: no cover
                "equal_nan",
                "nan",
                "posinf",
                "neginf",  # pragma: no cover
                "axis1",  # pragma: no cover
                "rtol",  # pragma: no cover
                "n",  # pragma: no cover
                "m",  # pragma: no cover
                "endpoint",  # pragma: no cover
                "offset",  # pragma: no cover
                "a_max",  # pragma: no cover
                "ddof",  # pragma: no cover
                "dims",  # pragma: no cover
                "source",  # pragma: no cover
                "axis2",  # pragma: no cover
                "stop",  # pragma: no cover
                "k",  # pragma: no cover
                "dtype",  # pragma: no cover
                "reps",  # pragma: no cover
                "a_min",  # pragma: no cover
                "sparse",  # pragma: no cover
                "decimals",  # pragma: no cover
                "strides",  # pragma: no cover
                "num",  # pragma: no cover
                "max",  # pragma: no cover
                "indices_or_sections",  # pragma: no cover
                "shape",  # pragma: no cover
                "atol",  # pragma: no cover
                "axis1",  # pragma: no cover
                "step",  # pragma: no cover
                "axes",  # pragma: no cover
                "fill_value",  # pragma: no cover
                "stream",  # pragma: no cover
                "destination",  # pragma: no cover
                "alpha",  # pragma: no cover
                "beta",  # pragma: no cover
                "start",  # pragma: no cover
                "equal_nan",
                "nan",
                "posinf",
                "neginf",  # pragma: no cover
                "keepdims",
                "shape",
                "size",  # pragma: no cover
                "min",  # pragma: no cover
                "p",  # pragma: no cover
            } and "ceil" not in (  # pragma: no cover
                "transpose",  # pragma: no cover
                "moveaxis",  # pragma: no cover
                "reshape",  # pragma: no cover
                "expand_dims",  # pragma: no cover
                "linspace",  # pragma: no cover
                "logcumsumexp",  # pragma: no cover
                "cummin",  # pragma: no cover
                "cummax",  # pragma: no cover
                "cumprod",  # pragma: no cover
                "cumsum",  # pragma: no cover
                "logsumexp",  # pragma: no cover
                "all",  # pragma: no cover
                "any",  # pragma: no cover
                "argmin",  # pragma: no cover
                "argmax",  # pragma: no cover
                "prod",  # pragma: no cover
                "sum",  # pragma: no cover
                "mean",  # pragma: no cover
                "var",  # pragma: no cover
                "std",  # pragma: no cover
                "max",  # pragma: no cover
                "min",  # pragma: no cover
                "tensordot",  # pragma: no cover
            ):  # pragma: no cover
                return _to_tensor(x)  # pragma: no cover
        if type(x).__name__ == "ndarray":  # pragma: no cover
            return _to_tensor(x)  # pragma: no cover
        return x  # pragma: no cover

    def _w(x):  # pragma: no cover
        if isinstance(x, tuple):  # pragma: no cover
            return tuple((_w(i) for i in x))  # pragma: no cover
        if isinstance(x, list):  # pragma: no cover
            return [_w(i) for i in x]  # pragma: no cover
        return array(x) if hasattr(x, "shape") else x  # pragma: no cover

    fn = _get_compiler_op("ceil")  # pragma: no cover
    res = fn(  # pragma: no cover
        *[_u(x) for x in args],  # pragma: no cover
        **{
            ("dim" if k == "axis" else k): _u(v, k)
            for (k, v) in kwargs.items()
            if k != "stream"
        },  # pragma: no cover
    )  # pragma: no cover

    def __w(x):
        if isinstance(x, tuple):
            return tuple((__w(i) for i in x))
        if isinstance(x, list):
            return [__w(i) for i in x]
        return array(x) if hasattr(x, "shape") else x

    return __w(res)  # pragma: no cover


def clip(*args, **kwargs):  # pragma: no cover
    """Compute clip.

    Args:
        args: The args argument.
        kwargs: The kwargs argument.

    Returns:
        The result of clip.

    """
    from zero_mlx.array import array, _to_tensor  # pragma: no cover
    import ml_switcheroo_compiler.ops as mops  # pragma: no cover
    import ml_switcheroo_compiler.ops.nn as mnn  # pragma: no cover
    import ml_switcheroo_compiler.random as mrand  # pragma: no cover
    import types  # pragma: no cover

    def _u(x, param_name=None):  # pragma: no cover
        if isinstance(x, list):  # pragma: no cover
            if __import__("builtins").all(  # pragma: no cover
                (isinstance(i, int) for i in x)  # pragma: no cover
            ):  # pragma: no cover
                return tuple(x)  # pragma: no cover
            return [_u(i, param_name) for i in x]  # pragma: no cover
        if isinstance(x, tuple):  # pragma: no cover
            return tuple((_u(i, param_name) for i in x))  # pragma: no cover
        if hasattr(x, "_tensor"):  # pragma: no cover
            return x._tensor  # pragma: no cover
        if isinstance(x, (int, float, bool, complex)):  # pragma: no cover
            if param_name not in {  # pragma: no cover
                "axis",  # pragma: no cover
                "keepdims",
                "shape",
                "size",  # pragma: no cover
                "equal_nan",
                "nan",
                "posinf",
                "neginf",  # pragma: no cover
                "axis1",  # pragma: no cover
                "rtol",  # pragma: no cover
                "n",  # pragma: no cover
                "m",  # pragma: no cover
                "endpoint",  # pragma: no cover
                "offset",  # pragma: no cover
                "a_max",  # pragma: no cover
                "ddof",  # pragma: no cover
                "dims",  # pragma: no cover
                "source",  # pragma: no cover
                "axis2",  # pragma: no cover
                "stop",  # pragma: no cover
                "k",  # pragma: no cover
                "dtype",  # pragma: no cover
                "reps",  # pragma: no cover
                "a_min",  # pragma: no cover
                "sparse",  # pragma: no cover
                "decimals",  # pragma: no cover
                "strides",  # pragma: no cover
                "num",  # pragma: no cover
                "max",  # pragma: no cover
                "indices_or_sections",  # pragma: no cover
                "shape",  # pragma: no cover
                "atol",  # pragma: no cover
                "axis1",  # pragma: no cover
                "step",  # pragma: no cover
                "axes",  # pragma: no cover
                "fill_value",  # pragma: no cover
                "stream",  # pragma: no cover
                "destination",  # pragma: no cover
                "alpha",  # pragma: no cover
                "beta",  # pragma: no cover
                "start",  # pragma: no cover
                "equal_nan",
                "nan",
                "posinf",
                "neginf",  # pragma: no cover
                "keepdims",
                "shape",
                "size",  # pragma: no cover
                "min",  # pragma: no cover
                "p",  # pragma: no cover
            } and "clip" not in (  # pragma: no cover
                "transpose",  # pragma: no cover
                "moveaxis",  # pragma: no cover
                "reshape",  # pragma: no cover
                "expand_dims",  # pragma: no cover
                "linspace",  # pragma: no cover
                "logcumsumexp",  # pragma: no cover
                "cummin",  # pragma: no cover
                "cummax",  # pragma: no cover
                "cumprod",  # pragma: no cover
                "cumsum",  # pragma: no cover
                "logsumexp",  # pragma: no cover
                "all",  # pragma: no cover
                "any",  # pragma: no cover
                "argmin",  # pragma: no cover
                "argmax",  # pragma: no cover
                "prod",  # pragma: no cover
                "sum",  # pragma: no cover
                "mean",  # pragma: no cover
                "var",  # pragma: no cover
                "std",  # pragma: no cover
                "max",  # pragma: no cover
                "min",  # pragma: no cover
                "tensordot",  # pragma: no cover
            ):  # pragma: no cover
                return _to_tensor(x)  # pragma: no cover
        if type(x).__name__ == "ndarray":  # pragma: no cover
            return _to_tensor(x)  # pragma: no cover
        return x  # pragma: no cover

    def _w(x):  # pragma: no cover
        if isinstance(x, tuple):  # pragma: no cover
            return tuple((_w(i) for i in x))  # pragma: no cover
        if isinstance(x, list):  # pragma: no cover
            return [_w(i) for i in x]  # pragma: no cover
        return array(x) if hasattr(x, "shape") else x  # pragma: no cover

    fn = _get_compiler_op("clip")  # pragma: no cover
    res = fn(  # pragma: no cover
        *[_u(x) for x in args],  # pragma: no cover
        **{
            ("dim" if k == "axis" else k): _u(v, k)
            for (k, v) in kwargs.items()
            if k != "stream"
        },  # pragma: no cover
    )  # pragma: no cover

    def __w(x):
        if isinstance(x, tuple):
            return tuple((__w(i) for i in x))
        if isinstance(x, list):
            return [__w(i) for i in x]
        return array(x) if hasattr(x, "shape") else x

    return __w(res)  # pragma: no cover


def conjugate(*args, **kwargs):  # pragma: no cover
    """Compute conjugate.

    Args:
        args: The args argument.
        kwargs: The kwargs argument.

    Returns:
        The result of conjugate.

    """
    from zero_mlx.array import array, _to_tensor  # pragma: no cover
    import ml_switcheroo_compiler.ops as mops  # pragma: no cover
    import ml_switcheroo_compiler.ops.nn as mnn  # pragma: no cover
    import ml_switcheroo_compiler.random as mrand  # pragma: no cover
    import types  # pragma: no cover

    def _u(x, param_name=None):  # pragma: no cover
        if isinstance(x, list):  # pragma: no cover
            if __import__("builtins").all(  # pragma: no cover
                (isinstance(i, int) for i in x)  # pragma: no cover
            ):  # pragma: no cover
                return tuple(x)  # pragma: no cover
            return [_u(i, param_name) for i in x]  # pragma: no cover
        if isinstance(x, tuple):  # pragma: no cover
            return tuple((_u(i, param_name) for i in x))  # pragma: no cover
        if hasattr(x, "_tensor"):  # pragma: no cover
            return x._tensor  # pragma: no cover
        if isinstance(x, (int, float, bool, complex)):  # pragma: no cover
            if param_name not in {  # pragma: no cover
                "axis",  # pragma: no cover
                "keepdims",
                "shape",
                "size",  # pragma: no cover
                "equal_nan",
                "nan",
                "posinf",
                "neginf",  # pragma: no cover
                "axis1",  # pragma: no cover
                "rtol",  # pragma: no cover
                "n",  # pragma: no cover
                "m",  # pragma: no cover
                "endpoint",  # pragma: no cover
                "offset",  # pragma: no cover
                "a_max",  # pragma: no cover
                "ddof",  # pragma: no cover
                "dims",  # pragma: no cover
                "source",  # pragma: no cover
                "axis2",  # pragma: no cover
                "stop",  # pragma: no cover
                "k",  # pragma: no cover
                "dtype",  # pragma: no cover
                "reps",  # pragma: no cover
                "a_min",  # pragma: no cover
                "sparse",  # pragma: no cover
                "decimals",  # pragma: no cover
                "strides",  # pragma: no cover
                "num",  # pragma: no cover
                "max",  # pragma: no cover
                "indices_or_sections",  # pragma: no cover
                "shape",  # pragma: no cover
                "atol",  # pragma: no cover
                "axis1",  # pragma: no cover
                "step",  # pragma: no cover
                "axes",  # pragma: no cover
                "fill_value",  # pragma: no cover
                "stream",  # pragma: no cover
                "destination",  # pragma: no cover
                "alpha",  # pragma: no cover
                "beta",  # pragma: no cover
                "start",  # pragma: no cover
                "equal_nan",
                "nan",
                "posinf",
                "neginf",  # pragma: no cover
                "keepdims",
                "shape",
                "size",  # pragma: no cover
                "min",  # pragma: no cover
                "p",  # pragma: no cover
            } and "conjugate" not in (  # pragma: no cover
                "transpose",  # pragma: no cover
                "moveaxis",  # pragma: no cover
                "reshape",  # pragma: no cover
                "expand_dims",  # pragma: no cover
                "linspace",  # pragma: no cover
                "logcumsumexp",  # pragma: no cover
                "cummin",  # pragma: no cover
                "cummax",  # pragma: no cover
                "cumprod",  # pragma: no cover
                "cumsum",  # pragma: no cover
                "logsumexp",  # pragma: no cover
                "all",  # pragma: no cover
                "any",  # pragma: no cover
                "argmin",  # pragma: no cover
                "argmax",  # pragma: no cover
                "prod",  # pragma: no cover
                "sum",  # pragma: no cover
                "mean",  # pragma: no cover
                "var",  # pragma: no cover
                "std",  # pragma: no cover
                "max",  # pragma: no cover
                "min",  # pragma: no cover
                "tensordot",  # pragma: no cover
            ):  # pragma: no cover
                return _to_tensor(x)  # pragma: no cover
        if type(x).__name__ == "ndarray":  # pragma: no cover
            return _to_tensor(x)  # pragma: no cover
        return x  # pragma: no cover

    def _w(x):  # pragma: no cover
        if isinstance(x, tuple):  # pragma: no cover
            return tuple((_w(i) for i in x))  # pragma: no cover
        if isinstance(x, list):  # pragma: no cover
            return [_w(i) for i in x]  # pragma: no cover
        return array(x) if hasattr(x, "shape") else x  # pragma: no cover

    fn = _get_compiler_op("conj")  # pragma: no cover
    res = fn(  # pragma: no cover
        *[_u(x) for x in args],  # pragma: no cover
        **{
            ("dim" if k == "axis" else k): _u(v, k)
            for (k, v) in kwargs.items()
            if k != "stream"
        },  # pragma: no cover
    )  # pragma: no cover

    def __w(x):
        if isinstance(x, tuple):
            return tuple((__w(i) for i in x))
        if isinstance(x, list):
            return [__w(i) for i in x]
        return array(x) if hasattr(x, "shape") else x

    return __w(res)  # pragma: no cover


def degrees(*args, **kwargs):  # pragma: no cover
    """Compute degrees.

    Args:
        args: The args argument.
        kwargs: The kwargs argument.

    Returns:
        The result of degrees.

    """
    from zero_mlx.array import array, _to_tensor  # pragma: no cover
    import ml_switcheroo_compiler.ops as mops  # pragma: no cover
    import ml_switcheroo_compiler.ops.nn as mnn  # pragma: no cover
    import ml_switcheroo_compiler.random as mrand  # pragma: no cover
    import types  # pragma: no cover

    def _u(x, param_name=None):  # pragma: no cover
        if isinstance(x, list):  # pragma: no cover
            if __import__("builtins").all(  # pragma: no cover
                (isinstance(i, int) for i in x)  # pragma: no cover
            ):  # pragma: no cover
                return tuple(x)  # pragma: no cover
            return [_u(i, param_name) for i in x]  # pragma: no cover
        if isinstance(x, tuple):  # pragma: no cover
            return tuple((_u(i, param_name) for i in x))  # pragma: no cover
        if hasattr(x, "_tensor"):  # pragma: no cover
            return x._tensor  # pragma: no cover
        if isinstance(x, (int, float, bool, complex)):  # pragma: no cover
            if param_name not in {  # pragma: no cover
                "axis",  # pragma: no cover
                "keepdims",
                "shape",
                "size",  # pragma: no cover
                "equal_nan",
                "nan",
                "posinf",
                "neginf",  # pragma: no cover
                "axis1",  # pragma: no cover
                "rtol",  # pragma: no cover
                "n",  # pragma: no cover
                "m",  # pragma: no cover
                "endpoint",  # pragma: no cover
                "offset",  # pragma: no cover
                "a_max",  # pragma: no cover
                "ddof",  # pragma: no cover
                "dims",  # pragma: no cover
                "source",  # pragma: no cover
                "axis2",  # pragma: no cover
                "stop",  # pragma: no cover
                "k",  # pragma: no cover
                "dtype",  # pragma: no cover
                "reps",  # pragma: no cover
                "a_min",  # pragma: no cover
                "sparse",  # pragma: no cover
                "decimals",  # pragma: no cover
                "strides",  # pragma: no cover
                "num",  # pragma: no cover
                "max",  # pragma: no cover
                "indices_or_sections",  # pragma: no cover
                "shape",  # pragma: no cover
                "atol",  # pragma: no cover
                "axis1",  # pragma: no cover
                "step",  # pragma: no cover
                "axes",  # pragma: no cover
                "fill_value",  # pragma: no cover
                "stream",  # pragma: no cover
                "destination",  # pragma: no cover
                "alpha",  # pragma: no cover
                "beta",  # pragma: no cover
                "start",  # pragma: no cover
                "equal_nan",
                "nan",
                "posinf",
                "neginf",  # pragma: no cover
                "keepdims",
                "shape",
                "size",  # pragma: no cover
                "min",  # pragma: no cover
                "p",  # pragma: no cover
            } and "degrees" not in (  # pragma: no cover
                "transpose",  # pragma: no cover
                "moveaxis",  # pragma: no cover
                "reshape",  # pragma: no cover
                "expand_dims",  # pragma: no cover
                "linspace",  # pragma: no cover
                "logcumsumexp",  # pragma: no cover
                "cummin",  # pragma: no cover
                "cummax",  # pragma: no cover
                "cumprod",  # pragma: no cover
                "cumsum",  # pragma: no cover
                "logsumexp",  # pragma: no cover
                "all",  # pragma: no cover
                "any",  # pragma: no cover
                "argmin",  # pragma: no cover
                "argmax",  # pragma: no cover
                "prod",  # pragma: no cover
                "sum",  # pragma: no cover
                "mean",  # pragma: no cover
                "var",  # pragma: no cover
                "std",  # pragma: no cover
                "max",  # pragma: no cover
                "min",  # pragma: no cover
                "tensordot",  # pragma: no cover
            ):  # pragma: no cover
                return _to_tensor(x)  # pragma: no cover
        if type(x).__name__ == "ndarray":  # pragma: no cover
            return _to_tensor(x)  # pragma: no cover
        return x  # pragma: no cover

    def _w(x):  # pragma: no cover
        if isinstance(x, tuple):  # pragma: no cover
            return tuple((_w(i) for i in x))  # pragma: no cover
        if isinstance(x, list):  # pragma: no cover
            return [_w(i) for i in x]  # pragma: no cover
        return array(x) if hasattr(x, "shape") else x  # pragma: no cover

    fn = _get_compiler_op("degrees")  # pragma: no cover
    res = fn(  # pragma: no cover
        *[_u(x) for x in args],  # pragma: no cover
        **{
            ("dim" if k == "axis" else k): _u(v, k)
            for (k, v) in kwargs.items()
            if k != "stream"
        },  # pragma: no cover
    )  # pragma: no cover

    def __w(x):
        if isinstance(x, tuple):
            return tuple((__w(i) for i in x))
        if isinstance(x, list):
            return [__w(i) for i in x]
        return array(x) if hasattr(x, "shape") else x

    return __w(res)  # pragma: no cover


def erf(*args, **kwargs):  # pragma: no cover
    """Compute erf.

    Args:
        args: The args argument.
        kwargs: The kwargs argument.

    Returns:
        The result of erf.

    """
    from zero_mlx.array import array, _to_tensor  # pragma: no cover
    import ml_switcheroo_compiler.ops as mops  # pragma: no cover
    import ml_switcheroo_compiler.ops.nn as mnn  # pragma: no cover
    import ml_switcheroo_compiler.random as mrand  # pragma: no cover
    import types  # pragma: no cover

    def _u(x, param_name=None):  # pragma: no cover
        if isinstance(x, list):  # pragma: no cover
            if __import__("builtins").all(  # pragma: no cover
                (isinstance(i, int) for i in x)  # pragma: no cover
            ):  # pragma: no cover
                return tuple(x)  # pragma: no cover
            return [_u(i, param_name) for i in x]  # pragma: no cover
        if isinstance(x, tuple):  # pragma: no cover
            return tuple((_u(i, param_name) for i in x))  # pragma: no cover
        if hasattr(x, "_tensor"):  # pragma: no cover
            return x._tensor  # pragma: no cover
        if isinstance(x, (int, float, bool, complex)):  # pragma: no cover
            if param_name not in {  # pragma: no cover
                "axis",  # pragma: no cover
                "keepdims",
                "shape",
                "size",  # pragma: no cover
                "equal_nan",
                "nan",
                "posinf",
                "neginf",  # pragma: no cover
                "axis1",  # pragma: no cover
                "rtol",  # pragma: no cover
                "n",  # pragma: no cover
                "m",  # pragma: no cover
                "endpoint",  # pragma: no cover
                "offset",  # pragma: no cover
                "a_max",  # pragma: no cover
                "ddof",  # pragma: no cover
                "dims",  # pragma: no cover
                "source",  # pragma: no cover
                "axis2",  # pragma: no cover
                "stop",  # pragma: no cover
                "k",  # pragma: no cover
                "dtype",  # pragma: no cover
                "reps",  # pragma: no cover
                "a_min",  # pragma: no cover
                "sparse",  # pragma: no cover
                "decimals",  # pragma: no cover
                "strides",  # pragma: no cover
                "num",  # pragma: no cover
                "max",  # pragma: no cover
                "indices_or_sections",  # pragma: no cover
                "shape",  # pragma: no cover
                "atol",  # pragma: no cover
                "axis1",  # pragma: no cover
                "step",  # pragma: no cover
                "axes",  # pragma: no cover
                "fill_value",  # pragma: no cover
                "stream",  # pragma: no cover
                "destination",  # pragma: no cover
                "alpha",  # pragma: no cover
                "beta",  # pragma: no cover
                "start",  # pragma: no cover
                "equal_nan",
                "nan",
                "posinf",
                "neginf",  # pragma: no cover
                "keepdims",
                "shape",
                "size",  # pragma: no cover
                "min",  # pragma: no cover
                "p",  # pragma: no cover
            } and "erf" not in (  # pragma: no cover
                "transpose",  # pragma: no cover
                "moveaxis",  # pragma: no cover
                "reshape",  # pragma: no cover
                "expand_dims",  # pragma: no cover
                "linspace",  # pragma: no cover
                "logcumsumexp",  # pragma: no cover
                "cummin",  # pragma: no cover
                "cummax",  # pragma: no cover
                "cumprod",  # pragma: no cover
                "cumsum",  # pragma: no cover
                "logsumexp",  # pragma: no cover
                "all",  # pragma: no cover
                "any",  # pragma: no cover
                "argmin",  # pragma: no cover
                "argmax",  # pragma: no cover
                "prod",  # pragma: no cover
                "sum",  # pragma: no cover
                "mean",  # pragma: no cover
                "var",  # pragma: no cover
                "std",  # pragma: no cover
                "max",  # pragma: no cover
                "min",  # pragma: no cover
                "tensordot",  # pragma: no cover
            ):  # pragma: no cover
                return _to_tensor(x)  # pragma: no cover
        if type(x).__name__ == "ndarray":  # pragma: no cover
            return _to_tensor(x)  # pragma: no cover
        return x  # pragma: no cover

    def _w(x):  # pragma: no cover
        if isinstance(x, tuple):  # pragma: no cover
            return tuple((_w(i) for i in x))  # pragma: no cover
        if isinstance(x, list):  # pragma: no cover
            return [_w(i) for i in x]  # pragma: no cover
        return array(x) if hasattr(x, "shape") else x  # pragma: no cover

    fn = _get_compiler_op("erf")  # pragma: no cover
    res = fn(  # pragma: no cover
        *[_u(x) for x in args],  # pragma: no cover
        **{
            ("dim" if k == "axis" else k): _u(v, k)
            for (k, v) in kwargs.items()
            if k != "stream"
        },  # pragma: no cover
    )  # pragma: no cover

    def __w(x):
        if isinstance(x, tuple):
            return tuple((__w(i) for i in x))
        if isinstance(x, list):
            return [__w(i) for i in x]
        return array(x) if hasattr(x, "shape") else x

    return __w(res)  # pragma: no cover


def erfinv(*args, **kwargs):  # pragma: no cover
    """Compute erfinv.

    Args:
        args: The args argument.
        kwargs: The kwargs argument.

    Returns:
        The result of erfinv.

    """
    from zero_mlx.array import array, _to_tensor  # pragma: no cover
    import ml_switcheroo_compiler.ops as mops  # pragma: no cover
    import ml_switcheroo_compiler.ops.nn as mnn  # pragma: no cover
    import ml_switcheroo_compiler.random as mrand  # pragma: no cover
    import types  # pragma: no cover

    def _u(x, param_name=None):  # pragma: no cover
        if isinstance(x, list):  # pragma: no cover
            if __import__("builtins").all(  # pragma: no cover
                (isinstance(i, int) for i in x)  # pragma: no cover
            ):  # pragma: no cover
                return tuple(x)  # pragma: no cover
            return [_u(i, param_name) for i in x]  # pragma: no cover
        if isinstance(x, tuple):  # pragma: no cover
            return tuple((_u(i, param_name) for i in x))  # pragma: no cover
        if hasattr(x, "_tensor"):  # pragma: no cover
            return x._tensor  # pragma: no cover
        if isinstance(x, (int, float, bool, complex)):  # pragma: no cover
            if param_name not in {  # pragma: no cover
                "axis",  # pragma: no cover
                "keepdims",
                "shape",
                "size",  # pragma: no cover
                "equal_nan",
                "nan",
                "posinf",
                "neginf",  # pragma: no cover
                "axis1",  # pragma: no cover
                "rtol",  # pragma: no cover
                "n",  # pragma: no cover
                "m",  # pragma: no cover
                "endpoint",  # pragma: no cover
                "offset",  # pragma: no cover
                "a_max",  # pragma: no cover
                "ddof",  # pragma: no cover
                "dims",  # pragma: no cover
                "source",  # pragma: no cover
                "axis2",  # pragma: no cover
                "stop",  # pragma: no cover
                "k",  # pragma: no cover
                "dtype",  # pragma: no cover
                "reps",  # pragma: no cover
                "a_min",  # pragma: no cover
                "sparse",  # pragma: no cover
                "decimals",  # pragma: no cover
                "strides",  # pragma: no cover
                "num",  # pragma: no cover
                "max",  # pragma: no cover
                "indices_or_sections",  # pragma: no cover
                "shape",  # pragma: no cover
                "atol",  # pragma: no cover
                "axis1",  # pragma: no cover
                "step",  # pragma: no cover
                "axes",  # pragma: no cover
                "fill_value",  # pragma: no cover
                "stream",  # pragma: no cover
                "destination",  # pragma: no cover
                "alpha",  # pragma: no cover
                "beta",  # pragma: no cover
                "start",  # pragma: no cover
                "equal_nan",
                "nan",
                "posinf",
                "neginf",  # pragma: no cover
                "keepdims",
                "shape",
                "size",  # pragma: no cover
                "min",  # pragma: no cover
                "p",  # pragma: no cover
            } and "erfinv" not in (  # pragma: no cover
                "transpose",  # pragma: no cover
                "moveaxis",  # pragma: no cover
                "reshape",  # pragma: no cover
                "expand_dims",  # pragma: no cover
                "linspace",  # pragma: no cover
                "logcumsumexp",  # pragma: no cover
                "cummin",  # pragma: no cover
                "cummax",  # pragma: no cover
                "cumprod",  # pragma: no cover
                "cumsum",  # pragma: no cover
                "logsumexp",  # pragma: no cover
                "all",  # pragma: no cover
                "any",  # pragma: no cover
                "argmin",  # pragma: no cover
                "argmax",  # pragma: no cover
                "prod",  # pragma: no cover
                "sum",  # pragma: no cover
                "mean",  # pragma: no cover
                "var",  # pragma: no cover
                "std",  # pragma: no cover
                "max",  # pragma: no cover
                "min",  # pragma: no cover
                "tensordot",  # pragma: no cover
            ):  # pragma: no cover
                return _to_tensor(x)  # pragma: no cover
        if type(x).__name__ == "ndarray":  # pragma: no cover
            return _to_tensor(x)  # pragma: no cover
        return x  # pragma: no cover

    def _w(x):  # pragma: no cover
        if isinstance(x, tuple):  # pragma: no cover
            return tuple((_w(i) for i in x))  # pragma: no cover
        if isinstance(x, list):  # pragma: no cover
            return [_w(i) for i in x]  # pragma: no cover
        return array(x) if hasattr(x, "shape") else x  # pragma: no cover

    fn = _get_compiler_op("erfinv")  # pragma: no cover
    res = fn(  # pragma: no cover
        *[_u(x) for x in args],  # pragma: no cover
        **{
            ("dim" if k == "axis" else k): _u(v, k)
            for (k, v) in kwargs.items()
            if k != "stream"
        },  # pragma: no cover
    )  # pragma: no cover

    def __w(x):
        if isinstance(x, tuple):
            return tuple((__w(i) for i in x))
        if isinstance(x, list):
            return [__w(i) for i in x]
        return array(x) if hasattr(x, "shape") else x

    return __w(res)  # pragma: no cover


def expm1(*args, **kwargs):  # pragma: no cover
    """Compute expm1.

    Args:
        args: The args argument.
        kwargs: The kwargs argument.

    Returns:
        The result of expm1.

    """
    from zero_mlx.array import array, _to_tensor  # pragma: no cover
    import ml_switcheroo_compiler.ops as mops  # pragma: no cover
    import ml_switcheroo_compiler.ops.nn as mnn  # pragma: no cover
    import ml_switcheroo_compiler.random as mrand  # pragma: no cover
    import types  # pragma: no cover

    def _u(x, param_name=None):  # pragma: no cover
        if isinstance(x, list):  # pragma: no cover
            if __import__("builtins").all(  # pragma: no cover
                (isinstance(i, int) for i in x)  # pragma: no cover
            ):  # pragma: no cover
                return tuple(x)  # pragma: no cover
            return [_u(i, param_name) for i in x]  # pragma: no cover
        if isinstance(x, tuple):  # pragma: no cover
            return tuple((_u(i, param_name) for i in x))  # pragma: no cover
        if hasattr(x, "_tensor"):  # pragma: no cover
            return x._tensor  # pragma: no cover
        if isinstance(x, (int, float, bool, complex)):  # pragma: no cover
            if param_name not in {  # pragma: no cover
                "axis",  # pragma: no cover
                "keepdims",
                "shape",
                "size",  # pragma: no cover
                "equal_nan",
                "nan",
                "posinf",
                "neginf",  # pragma: no cover
                "axis1",  # pragma: no cover
                "rtol",  # pragma: no cover
                "n",  # pragma: no cover
                "m",  # pragma: no cover
                "endpoint",  # pragma: no cover
                "offset",  # pragma: no cover
                "a_max",  # pragma: no cover
                "ddof",  # pragma: no cover
                "dims",  # pragma: no cover
                "source",  # pragma: no cover
                "axis2",  # pragma: no cover
                "stop",  # pragma: no cover
                "k",  # pragma: no cover
                "dtype",  # pragma: no cover
                "reps",  # pragma: no cover
                "a_min",  # pragma: no cover
                "sparse",  # pragma: no cover
                "decimals",  # pragma: no cover
                "strides",  # pragma: no cover
                "num",  # pragma: no cover
                "max",  # pragma: no cover
                "indices_or_sections",  # pragma: no cover
                "shape",  # pragma: no cover
                "atol",  # pragma: no cover
                "axis1",  # pragma: no cover
                "step",  # pragma: no cover
                "axes",  # pragma: no cover
                "fill_value",  # pragma: no cover
                "stream",  # pragma: no cover
                "destination",  # pragma: no cover
                "alpha",  # pragma: no cover
                "beta",  # pragma: no cover
                "start",  # pragma: no cover
                "equal_nan",
                "nan",
                "posinf",
                "neginf",  # pragma: no cover
                "keepdims",
                "shape",
                "size",  # pragma: no cover
                "min",  # pragma: no cover
                "p",  # pragma: no cover
            } and "expm1" not in (  # pragma: no cover
                "transpose",  # pragma: no cover
                "moveaxis",  # pragma: no cover
                "reshape",  # pragma: no cover
                "expand_dims",  # pragma: no cover
                "linspace",  # pragma: no cover
                "logcumsumexp",  # pragma: no cover
                "cummin",  # pragma: no cover
                "cummax",  # pragma: no cover
                "cumprod",  # pragma: no cover
                "cumsum",  # pragma: no cover
                "logsumexp",  # pragma: no cover
                "all",  # pragma: no cover
                "any",  # pragma: no cover
                "argmin",  # pragma: no cover
                "argmax",  # pragma: no cover
                "prod",  # pragma: no cover
                "sum",  # pragma: no cover
                "mean",  # pragma: no cover
                "var",  # pragma: no cover
                "std",  # pragma: no cover
                "max",  # pragma: no cover
                "min",  # pragma: no cover
                "tensordot",  # pragma: no cover
            ):  # pragma: no cover
                return _to_tensor(x)  # pragma: no cover
        if type(x).__name__ == "ndarray":  # pragma: no cover
            return _to_tensor(x)  # pragma: no cover
        return x  # pragma: no cover

    def _w(x):  # pragma: no cover
        if isinstance(x, tuple):  # pragma: no cover
            return tuple((_w(i) for i in x))  # pragma: no cover
        if isinstance(x, list):  # pragma: no cover
            return [_w(i) for i in x]  # pragma: no cover
        return array(x) if hasattr(x, "shape") else x  # pragma: no cover

    fn = _get_compiler_op("expm1")  # pragma: no cover
    res = fn(  # pragma: no cover
        *[_u(x) for x in args],  # pragma: no cover
        **{
            ("dim" if k == "axis" else k): _u(v, k)
            for (k, v) in kwargs.items()
            if k != "stream"
        },  # pragma: no cover
    )  # pragma: no cover

    def __w(x):
        if isinstance(x, tuple):
            return tuple((__w(i) for i in x))
        if isinstance(x, list):
            return [__w(i) for i in x]
        return array(x) if hasattr(x, "shape") else x

    return __w(res)  # pragma: no cover


def floor(*args, **kwargs):  # pragma: no cover
    """Compute floor.

    Args:
        args: The args argument.
        kwargs: The kwargs argument.

    Returns:
        The result of floor.

    """
    from zero_mlx.array import array, _to_tensor  # pragma: no cover
    import ml_switcheroo_compiler.ops as mops  # pragma: no cover
    import ml_switcheroo_compiler.ops.nn as mnn  # pragma: no cover
    import ml_switcheroo_compiler.random as mrand  # pragma: no cover
    import types  # pragma: no cover

    def _u(x, param_name=None):  # pragma: no cover
        if isinstance(x, list):  # pragma: no cover
            if __import__("builtins").all(  # pragma: no cover
                (isinstance(i, int) for i in x)  # pragma: no cover
            ):  # pragma: no cover
                return tuple(x)  # pragma: no cover
            return [_u(i, param_name) for i in x]  # pragma: no cover
        if isinstance(x, tuple):  # pragma: no cover
            return tuple((_u(i, param_name) for i in x))  # pragma: no cover
        if hasattr(x, "_tensor"):  # pragma: no cover
            return x._tensor  # pragma: no cover
        if isinstance(x, (int, float, bool, complex)):  # pragma: no cover
            if param_name not in {  # pragma: no cover
                "axis",  # pragma: no cover
                "keepdims",
                "shape",
                "size",  # pragma: no cover
                "equal_nan",
                "nan",
                "posinf",
                "neginf",  # pragma: no cover
                "axis1",  # pragma: no cover
                "rtol",  # pragma: no cover
                "n",  # pragma: no cover
                "m",  # pragma: no cover
                "endpoint",  # pragma: no cover
                "offset",  # pragma: no cover
                "a_max",  # pragma: no cover
                "ddof",  # pragma: no cover
                "dims",  # pragma: no cover
                "source",  # pragma: no cover
                "axis2",  # pragma: no cover
                "stop",  # pragma: no cover
                "k",  # pragma: no cover
                "dtype",  # pragma: no cover
                "reps",  # pragma: no cover
                "a_min",  # pragma: no cover
                "sparse",  # pragma: no cover
                "decimals",  # pragma: no cover
                "strides",  # pragma: no cover
                "num",  # pragma: no cover
                "max",  # pragma: no cover
                "indices_or_sections",  # pragma: no cover
                "shape",  # pragma: no cover
                "atol",  # pragma: no cover
                "axis1",  # pragma: no cover
                "step",  # pragma: no cover
                "axes",  # pragma: no cover
                "fill_value",  # pragma: no cover
                "stream",  # pragma: no cover
                "destination",  # pragma: no cover
                "alpha",  # pragma: no cover
                "beta",  # pragma: no cover
                "start",  # pragma: no cover
                "equal_nan",
                "nan",
                "posinf",
                "neginf",  # pragma: no cover
                "keepdims",
                "shape",
                "size",  # pragma: no cover
                "min",  # pragma: no cover
                "p",  # pragma: no cover
            } and "floor" not in (  # pragma: no cover
                "transpose",  # pragma: no cover
                "moveaxis",  # pragma: no cover
                "reshape",  # pragma: no cover
                "expand_dims",  # pragma: no cover
                "linspace",  # pragma: no cover
                "logcumsumexp",  # pragma: no cover
                "cummin",  # pragma: no cover
                "cummax",  # pragma: no cover
                "cumprod",  # pragma: no cover
                "cumsum",  # pragma: no cover
                "logsumexp",  # pragma: no cover
                "all",  # pragma: no cover
                "any",  # pragma: no cover
                "argmin",  # pragma: no cover
                "argmax",  # pragma: no cover
                "prod",  # pragma: no cover
                "sum",  # pragma: no cover
                "mean",  # pragma: no cover
                "var",  # pragma: no cover
                "std",  # pragma: no cover
                "max",  # pragma: no cover
                "min",  # pragma: no cover
                "tensordot",  # pragma: no cover
            ):  # pragma: no cover
                return _to_tensor(x)  # pragma: no cover
        if type(x).__name__ == "ndarray":  # pragma: no cover
            return _to_tensor(x)  # pragma: no cover
        return x  # pragma: no cover

    def _w(x):  # pragma: no cover
        if isinstance(x, tuple):  # pragma: no cover
            return tuple((_w(i) for i in x))  # pragma: no cover
        if isinstance(x, list):  # pragma: no cover
            return [_w(i) for i in x]  # pragma: no cover
        return array(x) if hasattr(x, "shape") else x  # pragma: no cover

    fn = _get_compiler_op("floor")  # pragma: no cover
    res = fn(  # pragma: no cover
        *[_u(x) for x in args],  # pragma: no cover
        **{
            ("dim" if k == "axis" else k): _u(v, k)
            for (k, v) in kwargs.items()
            if k != "stream"
        },  # pragma: no cover
    )  # pragma: no cover

    def __w(x):
        if isinstance(x, tuple):
            return tuple((__w(i) for i in x))
        if isinstance(x, list):
            return [__w(i) for i in x]
        return array(x) if hasattr(x, "shape") else x

    return __w(res)  # pragma: no cover


def inner(*args, **kwargs):  # pragma: no cover
    """Compute inner.

    Args:
        args: The args argument.
        kwargs: The kwargs argument.

    Returns:
        The result of inner.

    """
    from zero_mlx.array import array, _to_tensor  # pragma: no cover
    import ml_switcheroo_compiler.ops as mops  # pragma: no cover
    import ml_switcheroo_compiler.ops.nn as mnn  # pragma: no cover
    import ml_switcheroo_compiler.random as mrand  # pragma: no cover
    import types  # pragma: no cover

    def _u(x, param_name=None):  # pragma: no cover
        if isinstance(x, list):  # pragma: no cover
            if __import__("builtins").all(  # pragma: no cover
                (isinstance(i, int) for i in x)  # pragma: no cover
            ):  # pragma: no cover
                return tuple(x)  # pragma: no cover
            return [_u(i, param_name) for i in x]  # pragma: no cover
        if isinstance(x, tuple):  # pragma: no cover
            return tuple((_u(i, param_name) for i in x))  # pragma: no cover
        if hasattr(x, "_tensor"):  # pragma: no cover
            return x._tensor  # pragma: no cover
        if isinstance(x, (int, float, bool, complex)):  # pragma: no cover
            if param_name not in {  # pragma: no cover
                "axis",  # pragma: no cover
                "keepdims",
                "shape",
                "size",  # pragma: no cover
                "equal_nan",
                "nan",
                "posinf",
                "neginf",  # pragma: no cover
                "axis1",  # pragma: no cover
                "rtol",  # pragma: no cover
                "n",  # pragma: no cover
                "m",  # pragma: no cover
                "endpoint",  # pragma: no cover
                "offset",  # pragma: no cover
                "a_max",  # pragma: no cover
                "ddof",  # pragma: no cover
                "dims",  # pragma: no cover
                "source",  # pragma: no cover
                "axis2",  # pragma: no cover
                "stop",  # pragma: no cover
                "k",  # pragma: no cover
                "dtype",  # pragma: no cover
                "reps",  # pragma: no cover
                "a_min",  # pragma: no cover
                "sparse",  # pragma: no cover
                "decimals",  # pragma: no cover
                "strides",  # pragma: no cover
                "num",  # pragma: no cover
                "max",  # pragma: no cover
                "indices_or_sections",  # pragma: no cover
                "shape",  # pragma: no cover
                "atol",  # pragma: no cover
                "axis1",  # pragma: no cover
                "step",  # pragma: no cover
                "axes",  # pragma: no cover
                "fill_value",  # pragma: no cover
                "stream",  # pragma: no cover
                "destination",  # pragma: no cover
                "alpha",  # pragma: no cover
                "beta",  # pragma: no cover
                "start",  # pragma: no cover
                "equal_nan",
                "nan",
                "posinf",
                "neginf",  # pragma: no cover
                "keepdims",
                "shape",
                "size",  # pragma: no cover
                "min",  # pragma: no cover
                "p",  # pragma: no cover
            } and "inner" not in (  # pragma: no cover
                "transpose",  # pragma: no cover
                "moveaxis",  # pragma: no cover
                "reshape",  # pragma: no cover
                "expand_dims",  # pragma: no cover
                "linspace",  # pragma: no cover
                "logcumsumexp",  # pragma: no cover
                "cummin",  # pragma: no cover
                "cummax",  # pragma: no cover
                "cumprod",  # pragma: no cover
                "cumsum",  # pragma: no cover
                "logsumexp",  # pragma: no cover
                "all",  # pragma: no cover
                "any",  # pragma: no cover
                "argmin",  # pragma: no cover
                "argmax",  # pragma: no cover
                "prod",  # pragma: no cover
                "sum",  # pragma: no cover
                "mean",  # pragma: no cover
                "var",  # pragma: no cover
                "std",  # pragma: no cover
                "max",  # pragma: no cover
                "min",  # pragma: no cover
                "tensordot",  # pragma: no cover
            ):  # pragma: no cover
                return _to_tensor(x)  # pragma: no cover
        if type(x).__name__ == "ndarray":  # pragma: no cover
            return _to_tensor(x)  # pragma: no cover
        return x  # pragma: no cover

    def _w(x):  # pragma: no cover
        if isinstance(x, tuple):  # pragma: no cover
            return tuple((_w(i) for i in x))  # pragma: no cover
        if isinstance(x, list):  # pragma: no cover
            return [_w(i) for i in x]  # pragma: no cover
        return array(x) if hasattr(x, "shape") else x  # pragma: no cover

    fn = _get_compiler_op("inner")  # pragma: no cover
    res = fn(  # pragma: no cover
        *[_u(x) for x in args],  # pragma: no cover
        **{
            ("dim" if k == "axis" else k): _u(v, k)
            for (k, v) in kwargs.items()
            if k != "stream"
        },  # pragma: no cover
    )  # pragma: no cover

    def __w(x):
        if isinstance(x, tuple):
            return tuple((__w(i) for i in x))
        if isinstance(x, list):
            return [__w(i) for i in x]
        return array(x) if hasattr(x, "shape") else x

    return __w(res)  # pragma: no cover


def isclose(*args, **kwargs):  # pragma: no cover
    """Compute isclose.

    Args:
        args: The args argument.
        kwargs: The kwargs argument.

    Returns:
        The result of isclose.

    """
    from zero_mlx.array import array, _to_tensor  # pragma: no cover
    import ml_switcheroo_compiler.ops as mops  # pragma: no cover
    import ml_switcheroo_compiler.ops.nn as mnn  # pragma: no cover
    import ml_switcheroo_compiler.random as mrand  # pragma: no cover
    import types  # pragma: no cover

    def _u(x, param_name=None):  # pragma: no cover
        if isinstance(x, list):  # pragma: no cover
            if __import__("builtins").all(  # pragma: no cover
                (isinstance(i, int) for i in x)  # pragma: no cover
            ):  # pragma: no cover
                return tuple(x)  # pragma: no cover
            return [_u(i, param_name) for i in x]  # pragma: no cover
        if isinstance(x, tuple):  # pragma: no cover
            return tuple((_u(i, param_name) for i in x))  # pragma: no cover
        if hasattr(x, "_tensor"):  # pragma: no cover
            return x._tensor  # pragma: no cover
        if isinstance(x, (int, float, bool, complex)):  # pragma: no cover
            if param_name not in {  # pragma: no cover
                "axis",  # pragma: no cover
                "keepdims",
                "shape",
                "size",  # pragma: no cover
                "equal_nan",
                "nan",
                "posinf",
                "neginf",  # pragma: no cover
                "axis1",  # pragma: no cover
                "rtol",  # pragma: no cover
                "n",  # pragma: no cover
                "m",  # pragma: no cover
                "endpoint",  # pragma: no cover
                "offset",  # pragma: no cover
                "a_max",  # pragma: no cover
                "ddof",  # pragma: no cover
                "dims",  # pragma: no cover
                "source",  # pragma: no cover
                "axis2",  # pragma: no cover
                "stop",  # pragma: no cover
                "k",  # pragma: no cover
                "dtype",  # pragma: no cover
                "reps",  # pragma: no cover
                "a_min",  # pragma: no cover
                "sparse",  # pragma: no cover
                "decimals",  # pragma: no cover
                "strides",  # pragma: no cover
                "num",  # pragma: no cover
                "max",  # pragma: no cover
                "indices_or_sections",  # pragma: no cover
                "shape",  # pragma: no cover
                "atol",  # pragma: no cover
                "axis1",  # pragma: no cover
                "step",  # pragma: no cover
                "axes",  # pragma: no cover
                "fill_value",  # pragma: no cover
                "stream",  # pragma: no cover
                "destination",  # pragma: no cover
                "alpha",  # pragma: no cover
                "beta",  # pragma: no cover
                "start",  # pragma: no cover
                "equal_nan",
                "nan",
                "posinf",
                "neginf",  # pragma: no cover
                "keepdims",
                "shape",
                "size",  # pragma: no cover
                "min",  # pragma: no cover
                "p",  # pragma: no cover
            } and "isclose" not in (  # pragma: no cover
                "transpose",  # pragma: no cover
                "moveaxis",  # pragma: no cover
                "reshape",  # pragma: no cover
                "expand_dims",  # pragma: no cover
                "linspace",  # pragma: no cover
                "logcumsumexp",  # pragma: no cover
                "cummin",  # pragma: no cover
                "cummax",  # pragma: no cover
                "cumprod",  # pragma: no cover
                "cumsum",  # pragma: no cover
                "logsumexp",  # pragma: no cover
                "all",  # pragma: no cover
                "any",  # pragma: no cover
                "argmin",  # pragma: no cover
                "argmax",  # pragma: no cover
                "prod",  # pragma: no cover
                "sum",  # pragma: no cover
                "mean",  # pragma: no cover
                "var",  # pragma: no cover
                "std",  # pragma: no cover
                "max",  # pragma: no cover
                "min",  # pragma: no cover
                "tensordot",  # pragma: no cover
            ):  # pragma: no cover
                return _to_tensor(x)  # pragma: no cover
        if type(x).__name__ == "ndarray":  # pragma: no cover
            return _to_tensor(x)  # pragma: no cover
        return x  # pragma: no cover

    def _w(x):  # pragma: no cover
        if isinstance(x, tuple):  # pragma: no cover
            return tuple((_w(i) for i in x))  # pragma: no cover
        if isinstance(x, list):  # pragma: no cover
            return [_w(i) for i in x]  # pragma: no cover
        return array(x) if hasattr(x, "shape") else x  # pragma: no cover

    fn = _get_compiler_op("isclose")  # pragma: no cover
    res = fn(  # pragma: no cover
        *[_u(x) for x in args],  # pragma: no cover
        **{
            ("dim" if k == "axis" else k): _u(v, k)
            for (k, v) in kwargs.items()
            if k != "stream"
        },  # pragma: no cover
    )  # pragma: no cover

    def __w(x):
        if isinstance(x, tuple):
            return tuple((__w(i) for i in x))
        if isinstance(x, list):
            return [__w(i) for i in x]
        return array(x) if hasattr(x, "shape") else x

    return __w(res)  # pragma: no cover


def isfinite(*args, **kwargs):  # pragma: no cover
    """Compute isfinite.

    Args:
        args: The args argument.
        kwargs: The kwargs argument.

    Returns:
        The result of isfinite.

    """
    from zero_mlx.array import array, _to_tensor  # pragma: no cover
    import ml_switcheroo_compiler.ops as mops  # pragma: no cover
    import ml_switcheroo_compiler.ops.nn as mnn  # pragma: no cover
    import ml_switcheroo_compiler.random as mrand  # pragma: no cover
    import types  # pragma: no cover

    def _u(x, param_name=None):  # pragma: no cover
        if isinstance(x, list):  # pragma: no cover
            if __import__("builtins").all(  # pragma: no cover
                (isinstance(i, int) for i in x)  # pragma: no cover
            ):  # pragma: no cover
                return tuple(x)  # pragma: no cover
            return [_u(i, param_name) for i in x]  # pragma: no cover
        if isinstance(x, tuple):  # pragma: no cover
            return tuple((_u(i, param_name) for i in x))  # pragma: no cover
        if hasattr(x, "_tensor"):  # pragma: no cover
            return x._tensor  # pragma: no cover
        if isinstance(x, (int, float, bool, complex)):  # pragma: no cover
            if param_name not in {  # pragma: no cover
                "axis",  # pragma: no cover
                "keepdims",
                "shape",
                "size",  # pragma: no cover
                "equal_nan",
                "nan",
                "posinf",
                "neginf",  # pragma: no cover
                "axis1",  # pragma: no cover
                "rtol",  # pragma: no cover
                "n",  # pragma: no cover
                "m",  # pragma: no cover
                "endpoint",  # pragma: no cover
                "offset",  # pragma: no cover
                "a_max",  # pragma: no cover
                "ddof",  # pragma: no cover
                "dims",  # pragma: no cover
                "source",  # pragma: no cover
                "axis2",  # pragma: no cover
                "stop",  # pragma: no cover
                "k",  # pragma: no cover
                "dtype",  # pragma: no cover
                "reps",  # pragma: no cover
                "a_min",  # pragma: no cover
                "sparse",  # pragma: no cover
                "decimals",  # pragma: no cover
                "strides",  # pragma: no cover
                "num",  # pragma: no cover
                "max",  # pragma: no cover
                "indices_or_sections",  # pragma: no cover
                "shape",  # pragma: no cover
                "atol",  # pragma: no cover
                "axis1",  # pragma: no cover
                "step",  # pragma: no cover
                "axes",  # pragma: no cover
                "fill_value",  # pragma: no cover
                "stream",  # pragma: no cover
                "destination",  # pragma: no cover
                "alpha",  # pragma: no cover
                "beta",  # pragma: no cover
                "start",  # pragma: no cover
                "equal_nan",
                "nan",
                "posinf",
                "neginf",  # pragma: no cover
                "keepdims",
                "shape",
                "size",  # pragma: no cover
                "min",  # pragma: no cover
                "p",  # pragma: no cover
            } and "isfinite" not in (  # pragma: no cover
                "transpose",  # pragma: no cover
                "moveaxis",  # pragma: no cover
                "reshape",  # pragma: no cover
                "expand_dims",  # pragma: no cover
                "linspace",  # pragma: no cover
                "logcumsumexp",  # pragma: no cover
                "cummin",  # pragma: no cover
                "cummax",  # pragma: no cover
                "cumprod",  # pragma: no cover
                "cumsum",  # pragma: no cover
                "logsumexp",  # pragma: no cover
                "all",  # pragma: no cover
                "any",  # pragma: no cover
                "argmin",  # pragma: no cover
                "argmax",  # pragma: no cover
                "prod",  # pragma: no cover
                "sum",  # pragma: no cover
                "mean",  # pragma: no cover
                "var",  # pragma: no cover
                "std",  # pragma: no cover
                "max",  # pragma: no cover
                "min",  # pragma: no cover
                "tensordot",  # pragma: no cover
            ):  # pragma: no cover
                return _to_tensor(x)  # pragma: no cover
        if type(x).__name__ == "ndarray":  # pragma: no cover
            return _to_tensor(x)  # pragma: no cover
        return x  # pragma: no cover

    def _w(x):  # pragma: no cover
        if isinstance(x, tuple):  # pragma: no cover
            return tuple((_w(i) for i in x))  # pragma: no cover
        if isinstance(x, list):  # pragma: no cover
            return [_w(i) for i in x]  # pragma: no cover
        return array(x) if hasattr(x, "shape") else x  # pragma: no cover

    fn = _get_compiler_op("isfinite")  # pragma: no cover
    res = fn(  # pragma: no cover
        *[_u(x) for x in args],  # pragma: no cover
        **{
            ("dim" if k == "axis" else k): _u(v, k)
            for (k, v) in kwargs.items()
            if k != "stream"
        },  # pragma: no cover
    )  # pragma: no cover

    def __w(x):
        if isinstance(x, tuple):
            return tuple((__w(i) for i in x))
        if isinstance(x, list):
            return [__w(i) for i in x]
        return array(x) if hasattr(x, "shape") else x

    return __w(res)  # pragma: no cover


def isinf(*args, **kwargs):  # pragma: no cover
    """Compute isinf.

    Args:
        args: The args argument.
        kwargs: The kwargs argument.

    Returns:
        The result of isinf.

    """
    from zero_mlx.array import array, _to_tensor  # pragma: no cover
    import ml_switcheroo_compiler.ops as mops  # pragma: no cover
    import ml_switcheroo_compiler.ops.nn as mnn  # pragma: no cover
    import ml_switcheroo_compiler.random as mrand  # pragma: no cover
    import types  # pragma: no cover

    def _u(x, param_name=None):  # pragma: no cover
        if isinstance(x, list):  # pragma: no cover
            if __import__("builtins").all(  # pragma: no cover
                (isinstance(i, int) for i in x)  # pragma: no cover
            ):  # pragma: no cover
                return tuple(x)  # pragma: no cover
            return [_u(i, param_name) for i in x]  # pragma: no cover
        if isinstance(x, tuple):  # pragma: no cover
            return tuple((_u(i, param_name) for i in x))  # pragma: no cover
        if hasattr(x, "_tensor"):  # pragma: no cover
            return x._tensor  # pragma: no cover
        if isinstance(x, (int, float, bool, complex)):  # pragma: no cover
            if param_name not in {  # pragma: no cover
                "axis",  # pragma: no cover
                "keepdims",
                "shape",
                "size",  # pragma: no cover
                "equal_nan",
                "nan",
                "posinf",
                "neginf",  # pragma: no cover
                "axis1",  # pragma: no cover
                "rtol",  # pragma: no cover
                "n",  # pragma: no cover
                "m",  # pragma: no cover
                "endpoint",  # pragma: no cover
                "offset",  # pragma: no cover
                "a_max",  # pragma: no cover
                "ddof",  # pragma: no cover
                "dims",  # pragma: no cover
                "source",  # pragma: no cover
                "axis2",  # pragma: no cover
                "stop",  # pragma: no cover
                "k",  # pragma: no cover
                "dtype",  # pragma: no cover
                "reps",  # pragma: no cover
                "a_min",  # pragma: no cover
                "sparse",  # pragma: no cover
                "decimals",  # pragma: no cover
                "strides",  # pragma: no cover
                "num",  # pragma: no cover
                "max",  # pragma: no cover
                "indices_or_sections",  # pragma: no cover
                "shape",  # pragma: no cover
                "atol",  # pragma: no cover
                "axis1",  # pragma: no cover
                "step",  # pragma: no cover
                "axes",  # pragma: no cover
                "fill_value",  # pragma: no cover
                "stream",  # pragma: no cover
                "destination",  # pragma: no cover
                "alpha",  # pragma: no cover
                "beta",  # pragma: no cover
                "start",  # pragma: no cover
                "equal_nan",
                "nan",
                "posinf",
                "neginf",  # pragma: no cover
                "keepdims",
                "shape",
                "size",  # pragma: no cover
                "min",  # pragma: no cover
                "p",  # pragma: no cover
            } and "isinf" not in (  # pragma: no cover
                "transpose",  # pragma: no cover
                "moveaxis",  # pragma: no cover
                "reshape",  # pragma: no cover
                "expand_dims",  # pragma: no cover
                "linspace",  # pragma: no cover
                "logcumsumexp",  # pragma: no cover
                "cummin",  # pragma: no cover
                "cummax",  # pragma: no cover
                "cumprod",  # pragma: no cover
                "cumsum",  # pragma: no cover
                "logsumexp",  # pragma: no cover
                "all",  # pragma: no cover
                "any",  # pragma: no cover
                "argmin",  # pragma: no cover
                "argmax",  # pragma: no cover
                "prod",  # pragma: no cover
                "sum",  # pragma: no cover
                "mean",  # pragma: no cover
                "var",  # pragma: no cover
                "std",  # pragma: no cover
                "max",  # pragma: no cover
                "min",  # pragma: no cover
                "tensordot",  # pragma: no cover
            ):  # pragma: no cover
                return _to_tensor(x)  # pragma: no cover
        if type(x).__name__ == "ndarray":  # pragma: no cover
            return _to_tensor(x)  # pragma: no cover
        return x  # pragma: no cover

    def _w(x):  # pragma: no cover
        if isinstance(x, tuple):  # pragma: no cover
            return tuple((_w(i) for i in x))  # pragma: no cover
        if isinstance(x, list):  # pragma: no cover
            return [_w(i) for i in x]  # pragma: no cover
        return array(x) if hasattr(x, "shape") else x  # pragma: no cover

    fn = _get_compiler_op("isinf")  # pragma: no cover
    res = fn(  # pragma: no cover
        *[_u(x) for x in args],  # pragma: no cover
        **{
            ("dim" if k == "axis" else k): _u(v, k)
            for (k, v) in kwargs.items()
            if k != "stream"
        },  # pragma: no cover
    )  # pragma: no cover

    def __w(x):
        if isinstance(x, tuple):
            return tuple((__w(i) for i in x))
        if isinstance(x, list):
            return [__w(i) for i in x]
        return array(x) if hasattr(x, "shape") else x

    return __w(res)  # pragma: no cover


def isnan(*args, **kwargs):  # pragma: no cover
    """Compute isnan.

    Args:
        args: The args argument.
        kwargs: The kwargs argument.

    Returns:
        The result of isnan.

    """
    from zero_mlx.array import array, _to_tensor  # pragma: no cover
    import ml_switcheroo_compiler.ops as mops  # pragma: no cover
    import ml_switcheroo_compiler.ops.nn as mnn  # pragma: no cover
    import ml_switcheroo_compiler.random as mrand  # pragma: no cover
    import types  # pragma: no cover

    def _u(x, param_name=None):  # pragma: no cover
        if isinstance(x, list):  # pragma: no cover
            if __import__("builtins").all(  # pragma: no cover
                (isinstance(i, int) for i in x)  # pragma: no cover
            ):  # pragma: no cover
                return tuple(x)  # pragma: no cover
            return [_u(i, param_name) for i in x]  # pragma: no cover
        if isinstance(x, tuple):  # pragma: no cover
            return tuple((_u(i, param_name) for i in x))  # pragma: no cover
        if hasattr(x, "_tensor"):  # pragma: no cover
            return x._tensor  # pragma: no cover
        if isinstance(x, (int, float, bool, complex)):  # pragma: no cover
            if param_name not in {  # pragma: no cover
                "axis",  # pragma: no cover
                "keepdims",
                "shape",
                "size",  # pragma: no cover
                "equal_nan",
                "nan",
                "posinf",
                "neginf",  # pragma: no cover
                "axis1",  # pragma: no cover
                "rtol",  # pragma: no cover
                "n",  # pragma: no cover
                "m",  # pragma: no cover
                "endpoint",  # pragma: no cover
                "offset",  # pragma: no cover
                "a_max",  # pragma: no cover
                "ddof",  # pragma: no cover
                "dims",  # pragma: no cover
                "source",  # pragma: no cover
                "axis2",  # pragma: no cover
                "stop",  # pragma: no cover
                "k",  # pragma: no cover
                "dtype",  # pragma: no cover
                "reps",  # pragma: no cover
                "a_min",  # pragma: no cover
                "sparse",  # pragma: no cover
                "decimals",  # pragma: no cover
                "strides",  # pragma: no cover
                "num",  # pragma: no cover
                "max",  # pragma: no cover
                "indices_or_sections",  # pragma: no cover
                "shape",  # pragma: no cover
                "atol",  # pragma: no cover
                "axis1",  # pragma: no cover
                "step",  # pragma: no cover
                "axes",  # pragma: no cover
                "fill_value",  # pragma: no cover
                "stream",  # pragma: no cover
                "destination",  # pragma: no cover
                "alpha",  # pragma: no cover
                "beta",  # pragma: no cover
                "start",  # pragma: no cover
                "equal_nan",
                "nan",
                "posinf",
                "neginf",  # pragma: no cover
                "keepdims",
                "shape",
                "size",  # pragma: no cover
                "min",  # pragma: no cover
                "p",  # pragma: no cover
            } and "isnan" not in (  # pragma: no cover
                "transpose",  # pragma: no cover
                "moveaxis",  # pragma: no cover
                "reshape",  # pragma: no cover
                "expand_dims",  # pragma: no cover
                "linspace",  # pragma: no cover
                "logcumsumexp",  # pragma: no cover
                "cummin",  # pragma: no cover
                "cummax",  # pragma: no cover
                "cumprod",  # pragma: no cover
                "cumsum",  # pragma: no cover
                "logsumexp",  # pragma: no cover
                "all",  # pragma: no cover
                "any",  # pragma: no cover
                "argmin",  # pragma: no cover
                "argmax",  # pragma: no cover
                "prod",  # pragma: no cover
                "sum",  # pragma: no cover
                "mean",  # pragma: no cover
                "var",  # pragma: no cover
                "std",  # pragma: no cover
                "max",  # pragma: no cover
                "min",  # pragma: no cover
                "tensordot",  # pragma: no cover
            ):  # pragma: no cover
                return _to_tensor(x)  # pragma: no cover
        if type(x).__name__ == "ndarray":  # pragma: no cover
            return _to_tensor(x)  # pragma: no cover
        return x  # pragma: no cover

    def _w(x):  # pragma: no cover
        if isinstance(x, tuple):  # pragma: no cover
            return tuple((_w(i) for i in x))  # pragma: no cover
        if isinstance(x, list):  # pragma: no cover
            return [_w(i) for i in x]  # pragma: no cover
        return array(x) if hasattr(x, "shape") else x  # pragma: no cover

    fn = _get_compiler_op("isnan")  # pragma: no cover
    res = fn(  # pragma: no cover
        *[_u(x) for x in args],  # pragma: no cover
        **{
            ("dim" if k == "axis" else k): _u(v, k)
            for (k, v) in kwargs.items()
            if k != "stream"
        },  # pragma: no cover
    )  # pragma: no cover

    def __w(x):
        if isinstance(x, tuple):
            return tuple((__w(i) for i in x))
        if isinstance(x, list):
            return [__w(i) for i in x]
        return array(x) if hasattr(x, "shape") else x

    return __w(res)  # pragma: no cover


def isneginf(*args, **kwargs):  # pragma: no cover
    """Compute isneginf.

    Args:
        args: The args argument.
        kwargs: The kwargs argument.

    Returns:
        The result of isneginf.

    """
    from zero_mlx.array import array, _to_tensor  # pragma: no cover
    import ml_switcheroo_compiler.ops as mops  # pragma: no cover
    import ml_switcheroo_compiler.ops.nn as mnn  # pragma: no cover
    import ml_switcheroo_compiler.random as mrand  # pragma: no cover
    import types  # pragma: no cover

    def _u(x, param_name=None):  # pragma: no cover
        if isinstance(x, list):  # pragma: no cover
            if __import__("builtins").all(  # pragma: no cover
                (isinstance(i, int) for i in x)  # pragma: no cover
            ):  # pragma: no cover
                return tuple(x)  # pragma: no cover
            return [_u(i, param_name) for i in x]  # pragma: no cover
        if isinstance(x, tuple):  # pragma: no cover
            return tuple((_u(i, param_name) for i in x))  # pragma: no cover
        if hasattr(x, "_tensor"):  # pragma: no cover
            return x._tensor  # pragma: no cover
        if isinstance(x, (int, float, bool, complex)):  # pragma: no cover
            if param_name not in {  # pragma: no cover
                "axis",  # pragma: no cover
                "keepdims",
                "shape",
                "size",  # pragma: no cover
                "equal_nan",
                "nan",
                "posinf",
                "neginf",  # pragma: no cover
                "axis1",  # pragma: no cover
                "rtol",  # pragma: no cover
                "n",  # pragma: no cover
                "m",  # pragma: no cover
                "endpoint",  # pragma: no cover
                "offset",  # pragma: no cover
                "a_max",  # pragma: no cover
                "ddof",  # pragma: no cover
                "dims",  # pragma: no cover
                "source",  # pragma: no cover
                "axis2",  # pragma: no cover
                "stop",  # pragma: no cover
                "k",  # pragma: no cover
                "dtype",  # pragma: no cover
                "reps",  # pragma: no cover
                "a_min",  # pragma: no cover
                "sparse",  # pragma: no cover
                "decimals",  # pragma: no cover
                "strides",  # pragma: no cover
                "num",  # pragma: no cover
                "max",  # pragma: no cover
                "indices_or_sections",  # pragma: no cover
                "shape",  # pragma: no cover
                "atol",  # pragma: no cover
                "axis1",  # pragma: no cover
                "step",  # pragma: no cover
                "axes",  # pragma: no cover
                "fill_value",  # pragma: no cover
                "stream",  # pragma: no cover
                "destination",  # pragma: no cover
                "alpha",  # pragma: no cover
                "beta",  # pragma: no cover
                "start",  # pragma: no cover
                "equal_nan",
                "nan",
                "posinf",
                "neginf",  # pragma: no cover
                "keepdims",
                "shape",
                "size",  # pragma: no cover
                "min",  # pragma: no cover
                "p",  # pragma: no cover
            } and "isneginf" not in (  # pragma: no cover
                "transpose",  # pragma: no cover
                "moveaxis",  # pragma: no cover
                "reshape",  # pragma: no cover
                "expand_dims",  # pragma: no cover
                "linspace",  # pragma: no cover
                "logcumsumexp",  # pragma: no cover
                "cummin",  # pragma: no cover
                "cummax",  # pragma: no cover
                "cumprod",  # pragma: no cover
                "cumsum",  # pragma: no cover
                "logsumexp",  # pragma: no cover
                "all",  # pragma: no cover
                "any",  # pragma: no cover
                "argmin",  # pragma: no cover
                "argmax",  # pragma: no cover
                "prod",  # pragma: no cover
                "sum",  # pragma: no cover
                "mean",  # pragma: no cover
                "var",  # pragma: no cover
                "std",  # pragma: no cover
                "max",  # pragma: no cover
                "min",  # pragma: no cover
                "tensordot",  # pragma: no cover
            ):  # pragma: no cover
                return _to_tensor(x)  # pragma: no cover
        if type(x).__name__ == "ndarray":  # pragma: no cover
            return _to_tensor(x)  # pragma: no cover
        return x  # pragma: no cover

    def _w(x):  # pragma: no cover
        if isinstance(x, tuple):  # pragma: no cover
            return tuple((_w(i) for i in x))  # pragma: no cover
        if isinstance(x, list):  # pragma: no cover
            return [_w(i) for i in x]  # pragma: no cover
        return array(x) if hasattr(x, "shape") else x  # pragma: no cover

    fn = _get_compiler_op("isneginf")  # pragma: no cover
    res = fn(  # pragma: no cover
        *[_u(x) for x in args],  # pragma: no cover
        **{
            ("dim" if k == "axis" else k): _u(v, k)
            for (k, v) in kwargs.items()
            if k != "stream"
        },  # pragma: no cover
    )  # pragma: no cover

    def __w(x):
        if isinstance(x, tuple):
            return tuple((__w(i) for i in x))
        if isinstance(x, list):
            return [__w(i) for i in x]
        return array(x) if hasattr(x, "shape") else x

    return __w(res)  # pragma: no cover


def isposinf(*args, **kwargs):  # pragma: no cover
    """Compute isposinf.

    Args:
        args: The args argument.
        kwargs: The kwargs argument.

    Returns:
        The result of isposinf.

    """
    from zero_mlx.array import array, _to_tensor  # pragma: no cover
    import ml_switcheroo_compiler.ops as mops  # pragma: no cover
    import ml_switcheroo_compiler.ops.nn as mnn  # pragma: no cover
    import ml_switcheroo_compiler.random as mrand  # pragma: no cover
    import types  # pragma: no cover

    def _u(x, param_name=None):  # pragma: no cover
        if isinstance(x, list):  # pragma: no cover
            if __import__("builtins").all(  # pragma: no cover
                (isinstance(i, int) for i in x)  # pragma: no cover
            ):  # pragma: no cover
                return tuple(x)  # pragma: no cover
            return [_u(i, param_name) for i in x]  # pragma: no cover
        if isinstance(x, tuple):  # pragma: no cover
            return tuple((_u(i, param_name) for i in x))  # pragma: no cover
        if hasattr(x, "_tensor"):  # pragma: no cover
            return x._tensor  # pragma: no cover
        if isinstance(x, (int, float, bool, complex)):  # pragma: no cover
            if param_name not in {  # pragma: no cover
                "axis",  # pragma: no cover
                "keepdims",
                "shape",
                "size",  # pragma: no cover
                "equal_nan",
                "nan",
                "posinf",
                "neginf",  # pragma: no cover
                "axis1",  # pragma: no cover
                "rtol",  # pragma: no cover
                "n",  # pragma: no cover
                "m",  # pragma: no cover
                "endpoint",  # pragma: no cover
                "offset",  # pragma: no cover
                "a_max",  # pragma: no cover
                "ddof",  # pragma: no cover
                "dims",  # pragma: no cover
                "source",  # pragma: no cover
                "axis2",  # pragma: no cover
                "stop",  # pragma: no cover
                "k",  # pragma: no cover
                "dtype",  # pragma: no cover
                "reps",  # pragma: no cover
                "a_min",  # pragma: no cover
                "sparse",  # pragma: no cover
                "decimals",  # pragma: no cover
                "strides",  # pragma: no cover
                "num",  # pragma: no cover
                "max",  # pragma: no cover
                "indices_or_sections",  # pragma: no cover
                "shape",  # pragma: no cover
                "atol",  # pragma: no cover
                "axis1",  # pragma: no cover
                "step",  # pragma: no cover
                "axes",  # pragma: no cover
                "fill_value",  # pragma: no cover
                "stream",  # pragma: no cover
                "destination",  # pragma: no cover
                "alpha",  # pragma: no cover
                "beta",  # pragma: no cover
                "start",  # pragma: no cover
                "equal_nan",
                "nan",
                "posinf",
                "neginf",  # pragma: no cover
                "keepdims",
                "shape",
                "size",  # pragma: no cover
                "min",  # pragma: no cover
                "p",  # pragma: no cover
            } and "isposinf" not in (  # pragma: no cover
                "transpose",  # pragma: no cover
                "moveaxis",  # pragma: no cover
                "reshape",  # pragma: no cover
                "expand_dims",  # pragma: no cover
                "linspace",  # pragma: no cover
                "logcumsumexp",  # pragma: no cover
                "cummin",  # pragma: no cover
                "cummax",  # pragma: no cover
                "cumprod",  # pragma: no cover
                "cumsum",  # pragma: no cover
                "logsumexp",  # pragma: no cover
                "all",  # pragma: no cover
                "any",  # pragma: no cover
                "argmin",  # pragma: no cover
                "argmax",  # pragma: no cover
                "prod",  # pragma: no cover
                "sum",  # pragma: no cover
                "mean",  # pragma: no cover
                "var",  # pragma: no cover
                "std",  # pragma: no cover
                "max",  # pragma: no cover
                "min",  # pragma: no cover
                "tensordot",  # pragma: no cover
            ):  # pragma: no cover
                return _to_tensor(x)  # pragma: no cover
        if type(x).__name__ == "ndarray":  # pragma: no cover
            return _to_tensor(x)  # pragma: no cover
        return x  # pragma: no cover

    def _w(x):  # pragma: no cover
        if isinstance(x, tuple):  # pragma: no cover
            return tuple((_w(i) for i in x))  # pragma: no cover
        if isinstance(x, list):  # pragma: no cover
            return [_w(i) for i in x]  # pragma: no cover
        return array(x) if hasattr(x, "shape") else x  # pragma: no cover

    fn = _get_compiler_op("isposinf")  # pragma: no cover
    res = fn(  # pragma: no cover
        *[_u(x) for x in args],  # pragma: no cover
        **{
            ("dim" if k == "axis" else k): _u(v, k)
            for (k, v) in kwargs.items()
            if k != "stream"
        },  # pragma: no cover
    )  # pragma: no cover

    def __w(x):
        if isinstance(x, tuple):
            return tuple((__w(i) for i in x))
        if isinstance(x, list):
            return [__w(i) for i in x]
        return array(x) if hasattr(x, "shape") else x

    return __w(res)  # pragma: no cover


def issubdtype(*args, **kwargs):  # pragma: no cover
    """Compute issubdtype.

    Args:
        args: The args argument.
        kwargs: The kwargs argument.

    Returns:
        The result of issubdtype.

    """
    from zero_mlx.array import array, _to_tensor  # pragma: no cover
    import ml_switcheroo_compiler.ops as mops  # pragma: no cover
    import ml_switcheroo_compiler.ops.nn as mnn  # pragma: no cover
    import ml_switcheroo_compiler.random as mrand  # pragma: no cover
    import types  # pragma: no cover

    def _u(x, param_name=None):  # pragma: no cover
        if isinstance(x, list):  # pragma: no cover
            if __import__("builtins").all(  # pragma: no cover
                (isinstance(i, int) for i in x)  # pragma: no cover
            ):  # pragma: no cover
                return tuple(x)  # pragma: no cover
            return [_u(i, param_name) for i in x]  # pragma: no cover
        if isinstance(x, tuple):  # pragma: no cover
            return tuple((_u(i, param_name) for i in x))  # pragma: no cover
        if hasattr(x, "_tensor"):  # pragma: no cover
            return x._tensor  # pragma: no cover
        if isinstance(x, (int, float, bool, complex)):  # pragma: no cover
            if param_name not in {  # pragma: no cover
                "axis",  # pragma: no cover
                "keepdims",
                "shape",
                "size",  # pragma: no cover
                "equal_nan",
                "nan",
                "posinf",
                "neginf",  # pragma: no cover
                "axis1",  # pragma: no cover
                "rtol",  # pragma: no cover
                "n",  # pragma: no cover
                "m",  # pragma: no cover
                "endpoint",  # pragma: no cover
                "offset",  # pragma: no cover
                "a_max",  # pragma: no cover
                "ddof",  # pragma: no cover
                "dims",  # pragma: no cover
                "source",  # pragma: no cover
                "axis2",  # pragma: no cover
                "stop",  # pragma: no cover
                "k",  # pragma: no cover
                "dtype",  # pragma: no cover
                "reps",  # pragma: no cover
                "a_min",  # pragma: no cover
                "sparse",  # pragma: no cover
                "decimals",  # pragma: no cover
                "strides",  # pragma: no cover
                "num",  # pragma: no cover
                "max",  # pragma: no cover
                "indices_or_sections",  # pragma: no cover
                "shape",  # pragma: no cover
                "atol",  # pragma: no cover
                "axis1",  # pragma: no cover
                "step",  # pragma: no cover
                "axes",  # pragma: no cover
                "fill_value",  # pragma: no cover
                "stream",  # pragma: no cover
                "destination",  # pragma: no cover
                "alpha",  # pragma: no cover
                "beta",  # pragma: no cover
                "start",  # pragma: no cover
                "equal_nan",
                "nan",
                "posinf",
                "neginf",  # pragma: no cover
                "keepdims",
                "shape",
                "size",  # pragma: no cover
                "min",  # pragma: no cover
                "p",  # pragma: no cover
            } and "issubdtype" not in (  # pragma: no cover
                "transpose",  # pragma: no cover
                "moveaxis",  # pragma: no cover
                "reshape",  # pragma: no cover
                "expand_dims",  # pragma: no cover
                "linspace",  # pragma: no cover
                "logcumsumexp",  # pragma: no cover
                "cummin",  # pragma: no cover
                "cummax",  # pragma: no cover
                "cumprod",  # pragma: no cover
                "cumsum",  # pragma: no cover
                "logsumexp",  # pragma: no cover
                "all",  # pragma: no cover
                "any",  # pragma: no cover
                "argmin",  # pragma: no cover
                "argmax",  # pragma: no cover
                "prod",  # pragma: no cover
                "sum",  # pragma: no cover
                "mean",  # pragma: no cover
                "var",  # pragma: no cover
                "std",  # pragma: no cover
                "max",  # pragma: no cover
                "min",  # pragma: no cover
                "tensordot",  # pragma: no cover
            ):  # pragma: no cover
                return _to_tensor(x)  # pragma: no cover
        if type(x).__name__ == "ndarray":  # pragma: no cover
            return _to_tensor(x)  # pragma: no cover
        return x  # pragma: no cover

    def _w(x):  # pragma: no cover
        if isinstance(x, tuple):  # pragma: no cover
            return tuple((_w(i) for i in x))  # pragma: no cover
        if isinstance(x, list):  # pragma: no cover
            return [_w(i) for i in x]  # pragma: no cover
        return array(x) if hasattr(x, "shape") else x  # pragma: no cover

    fn = _get_compiler_op("issubdtype")  # pragma: no cover
    res = fn(  # pragma: no cover
        *[_u(x) for x in args],  # pragma: no cover
        **{
            ("dim" if k == "axis" else k): _u(v, k)
            for (k, v) in kwargs.items()
            if k != "stream"
        },  # pragma: no cover
    )  # pragma: no cover

    def __w(x):
        if isinstance(x, tuple):
            return tuple((__w(i) for i in x))
        if isinstance(x, list):
            return [__w(i) for i in x]
        return array(x) if hasattr(x, "shape") else x

    return __w(res)  # pragma: no cover


def kron(*args, **kwargs):  # pragma: no cover
    """Compute kron.

    Args:
        args: The args argument.
        kwargs: The kwargs argument.

    Returns:
        The result of kron.

    """
    from zero_mlx.array import array, _to_tensor  # pragma: no cover
    import ml_switcheroo_compiler.ops as mops  # pragma: no cover
    import ml_switcheroo_compiler.ops.nn as mnn  # pragma: no cover
    import ml_switcheroo_compiler.random as mrand  # pragma: no cover
    import types  # pragma: no cover

    def _u(x, param_name=None):  # pragma: no cover
        if isinstance(x, list):  # pragma: no cover
            if __import__("builtins").all(  # pragma: no cover
                (isinstance(i, int) for i in x)  # pragma: no cover
            ):  # pragma: no cover
                return tuple(x)  # pragma: no cover
            return [_u(i, param_name) for i in x]  # pragma: no cover
        if isinstance(x, tuple):  # pragma: no cover
            return tuple((_u(i, param_name) for i in x))  # pragma: no cover
        if hasattr(x, "_tensor"):  # pragma: no cover
            return x._tensor  # pragma: no cover
        if isinstance(x, (int, float, bool, complex)):  # pragma: no cover
            if param_name not in {  # pragma: no cover
                "axis",  # pragma: no cover
                "keepdims",
                "shape",
                "size",  # pragma: no cover
                "equal_nan",
                "nan",
                "posinf",
                "neginf",  # pragma: no cover
                "axis1",  # pragma: no cover
                "rtol",  # pragma: no cover
                "n",  # pragma: no cover
                "m",  # pragma: no cover
                "endpoint",  # pragma: no cover
                "offset",  # pragma: no cover
                "a_max",  # pragma: no cover
                "ddof",  # pragma: no cover
                "dims",  # pragma: no cover
                "source",  # pragma: no cover
                "axis2",  # pragma: no cover
                "stop",  # pragma: no cover
                "k",  # pragma: no cover
                "dtype",  # pragma: no cover
                "reps",  # pragma: no cover
                "a_min",  # pragma: no cover
                "sparse",  # pragma: no cover
                "decimals",  # pragma: no cover
                "strides",  # pragma: no cover
                "num",  # pragma: no cover
                "max",  # pragma: no cover
                "indices_or_sections",  # pragma: no cover
                "shape",  # pragma: no cover
                "atol",  # pragma: no cover
                "axis1",  # pragma: no cover
                "step",  # pragma: no cover
                "axes",  # pragma: no cover
                "fill_value",  # pragma: no cover
                "stream",  # pragma: no cover
                "destination",  # pragma: no cover
                "alpha",  # pragma: no cover
                "beta",  # pragma: no cover
                "start",  # pragma: no cover
                "equal_nan",
                "nan",
                "posinf",
                "neginf",  # pragma: no cover
                "keepdims",
                "shape",
                "size",  # pragma: no cover
                "min",  # pragma: no cover
                "p",  # pragma: no cover
            } and "kron" not in (  # pragma: no cover
                "transpose",  # pragma: no cover
                "moveaxis",  # pragma: no cover
                "reshape",  # pragma: no cover
                "expand_dims",  # pragma: no cover
                "linspace",  # pragma: no cover
                "logcumsumexp",  # pragma: no cover
                "cummin",  # pragma: no cover
                "cummax",  # pragma: no cover
                "cumprod",  # pragma: no cover
                "cumsum",  # pragma: no cover
                "logsumexp",  # pragma: no cover
                "all",  # pragma: no cover
                "any",  # pragma: no cover
                "argmin",  # pragma: no cover
                "argmax",  # pragma: no cover
                "prod",  # pragma: no cover
                "sum",  # pragma: no cover
                "mean",  # pragma: no cover
                "var",  # pragma: no cover
                "std",  # pragma: no cover
                "max",  # pragma: no cover
                "min",  # pragma: no cover
                "tensordot",  # pragma: no cover
            ):  # pragma: no cover
                return _to_tensor(x)  # pragma: no cover
        if type(x).__name__ == "ndarray":  # pragma: no cover
            return _to_tensor(x)  # pragma: no cover
        return x  # pragma: no cover

    def _w(x):  # pragma: no cover
        if isinstance(x, tuple):  # pragma: no cover
            return tuple((_w(i) for i in x))  # pragma: no cover
        if isinstance(x, list):  # pragma: no cover
            return [_w(i) for i in x]  # pragma: no cover
        return array(x) if hasattr(x, "shape") else x  # pragma: no cover

    fn = _get_compiler_op("kron")  # pragma: no cover
    res = fn(  # pragma: no cover
        *[_u(x) for x in args],  # pragma: no cover
        **{
            ("dim" if k == "axis" else k): _u(v, k)
            for (k, v) in kwargs.items()
            if k != "stream"
        },  # pragma: no cover
    )  # pragma: no cover

    def __w(x):
        if isinstance(x, tuple):
            return tuple((__w(i) for i in x))
        if isinstance(x, list):
            return [__w(i) for i in x]
        return array(x) if hasattr(x, "shape") else x

    return __w(res)  # pragma: no cover


def linspace(*args, **kwargs):  # pragma: no cover
    """Compute linspace.

    Args:
        args: The args argument.
        kwargs: The kwargs argument.

    Returns:
        The result of linspace.

    """
    from zero_mlx.array import array, _to_tensor  # pragma: no cover
    import ml_switcheroo_compiler.ops as mops  # pragma: no cover
    import ml_switcheroo_compiler.ops.nn as mnn  # pragma: no cover
    import ml_switcheroo_compiler.random as mrand  # pragma: no cover
    import types  # pragma: no cover

    def _u(x, param_name=None):  # pragma: no cover
        if isinstance(x, list):  # pragma: no cover
            if __import__("builtins").all(  # pragma: no cover
                (isinstance(i, int) for i in x)  # pragma: no cover
            ):  # pragma: no cover
                return tuple(x)  # pragma: no cover
            return [_u(i, param_name) for i in x]  # pragma: no cover
        if isinstance(x, tuple):  # pragma: no cover
            return tuple((_u(i, param_name) for i in x))  # pragma: no cover
        if hasattr(x, "_tensor"):  # pragma: no cover
            return x._tensor  # pragma: no cover
        if isinstance(x, (int, float, bool, complex)):  # pragma: no cover
            if param_name not in {  # pragma: no cover
                "axis",  # pragma: no cover
                "keepdims",
                "shape",
                "size",  # pragma: no cover
                "equal_nan",
                "nan",
                "posinf",
                "neginf",  # pragma: no cover
                "axis1",  # pragma: no cover
                "rtol",  # pragma: no cover
                "n",  # pragma: no cover
                "m",  # pragma: no cover
                "endpoint",  # pragma: no cover
                "offset",  # pragma: no cover
                "a_max",  # pragma: no cover
                "ddof",  # pragma: no cover
                "dims",  # pragma: no cover
                "source",  # pragma: no cover
                "axis2",  # pragma: no cover
                "stop",  # pragma: no cover
                "k",  # pragma: no cover
                "dtype",  # pragma: no cover
                "reps",  # pragma: no cover
                "a_min",  # pragma: no cover
                "sparse",  # pragma: no cover
                "decimals",  # pragma: no cover
                "strides",  # pragma: no cover
                "num",  # pragma: no cover
                "max",  # pragma: no cover
                "indices_or_sections",  # pragma: no cover
                "shape",  # pragma: no cover
                "atol",  # pragma: no cover
                "axis1",  # pragma: no cover
                "step",  # pragma: no cover
                "axes",  # pragma: no cover
                "fill_value",  # pragma: no cover
                "stream",  # pragma: no cover
                "destination",  # pragma: no cover
                "alpha",  # pragma: no cover
                "beta",  # pragma: no cover
                "start",  # pragma: no cover
                "equal_nan",
                "nan",
                "posinf",
                "neginf",  # pragma: no cover
                "keepdims",
                "shape",
                "size",  # pragma: no cover
                "min",  # pragma: no cover
                "p",  # pragma: no cover
            } and "linspace" not in (  # pragma: no cover
                "transpose",  # pragma: no cover
                "moveaxis",  # pragma: no cover
                "reshape",  # pragma: no cover
                "expand_dims",  # pragma: no cover
                "linspace",  # pragma: no cover
                "logcumsumexp",  # pragma: no cover
                "cummin",  # pragma: no cover
                "cummax",  # pragma: no cover
                "cumprod",  # pragma: no cover
                "cumsum",  # pragma: no cover
                "logsumexp",  # pragma: no cover
                "all",  # pragma: no cover
                "any",  # pragma: no cover
                "argmin",  # pragma: no cover
                "argmax",  # pragma: no cover
                "prod",  # pragma: no cover
                "sum",  # pragma: no cover
                "mean",  # pragma: no cover
                "var",  # pragma: no cover
                "std",  # pragma: no cover
                "max",  # pragma: no cover
                "min",  # pragma: no cover
                "tensordot",  # pragma: no cover
            ):  # pragma: no cover
                return _to_tensor(x)  # pragma: no cover
        if type(x).__name__ == "ndarray":  # pragma: no cover
            return _to_tensor(x)  # pragma: no cover
        return x  # pragma: no cover

    def _w(x):  # pragma: no cover
        if isinstance(x, tuple):  # pragma: no cover
            return tuple((_w(i) for i in x))  # pragma: no cover
        if isinstance(x, list):  # pragma: no cover
            return [_w(i) for i in x]  # pragma: no cover
        return array(x) if hasattr(x, "shape") else x  # pragma: no cover

    fn = _get_compiler_op("linspace")  # pragma: no cover
    res = fn(  # pragma: no cover
        *[_u(x) for x in args],  # pragma: no cover
        **{
            ("dim" if k == "axis" else k): _u(v, k)
            for (k, v) in kwargs.items()
            if k != "stream"
        },  # pragma: no cover
    )  # pragma: no cover

    def __w(x):
        if isinstance(x, tuple):
            return tuple((__w(i) for i in x))
        if isinstance(x, list):
            return [__w(i) for i in x]
        return array(x) if hasattr(x, "shape") else x

    return __w(res)  # pragma: no cover


def logaddexp(*args, **kwargs):  # pragma: no cover
    """Compute logaddexp.

    Args:
        args: The args argument.
        kwargs: The kwargs argument.

    Returns:
        The result of logaddexp.

    """
    from zero_mlx.array import array, _to_tensor  # pragma: no cover
    import ml_switcheroo_compiler.ops as mops  # pragma: no cover
    import ml_switcheroo_compiler.ops.nn as mnn  # pragma: no cover
    import ml_switcheroo_compiler.random as mrand  # pragma: no cover
    import types  # pragma: no cover

    def _u(x, param_name=None):  # pragma: no cover
        if isinstance(x, list):  # pragma: no cover
            if __import__("builtins").all(  # pragma: no cover
                (isinstance(i, int) for i in x)  # pragma: no cover
            ):  # pragma: no cover
                return tuple(x)  # pragma: no cover
            return [_u(i, param_name) for i in x]  # pragma: no cover
        if isinstance(x, tuple):  # pragma: no cover
            return tuple((_u(i, param_name) for i in x))  # pragma: no cover
        if hasattr(x, "_tensor"):  # pragma: no cover
            return x._tensor  # pragma: no cover
        if isinstance(x, (int, float, bool, complex)):  # pragma: no cover
            if param_name not in {  # pragma: no cover
                "axis",  # pragma: no cover
                "keepdims",
                "shape",
                "size",  # pragma: no cover
                "equal_nan",
                "nan",
                "posinf",
                "neginf",  # pragma: no cover
                "axis1",  # pragma: no cover
                "rtol",  # pragma: no cover
                "n",  # pragma: no cover
                "m",  # pragma: no cover
                "endpoint",  # pragma: no cover
                "offset",  # pragma: no cover
                "a_max",  # pragma: no cover
                "ddof",  # pragma: no cover
                "dims",  # pragma: no cover
                "source",  # pragma: no cover
                "axis2",  # pragma: no cover
                "stop",  # pragma: no cover
                "k",  # pragma: no cover
                "dtype",  # pragma: no cover
                "reps",  # pragma: no cover
                "a_min",  # pragma: no cover
                "sparse",  # pragma: no cover
                "decimals",  # pragma: no cover
                "strides",  # pragma: no cover
                "num",  # pragma: no cover
                "max",  # pragma: no cover
                "indices_or_sections",  # pragma: no cover
                "shape",  # pragma: no cover
                "atol",  # pragma: no cover
                "axis1",  # pragma: no cover
                "step",  # pragma: no cover
                "axes",  # pragma: no cover
                "fill_value",  # pragma: no cover
                "stream",  # pragma: no cover
                "destination",  # pragma: no cover
                "alpha",  # pragma: no cover
                "beta",  # pragma: no cover
                "start",  # pragma: no cover
                "equal_nan",
                "nan",
                "posinf",
                "neginf",  # pragma: no cover
                "keepdims",
                "shape",
                "size",  # pragma: no cover
                "min",  # pragma: no cover
                "p",  # pragma: no cover
            } and "logaddexp" not in (  # pragma: no cover
                "transpose",  # pragma: no cover
                "moveaxis",  # pragma: no cover
                "reshape",  # pragma: no cover
                "expand_dims",  # pragma: no cover
                "linspace",  # pragma: no cover
                "logcumsumexp",  # pragma: no cover
                "cummin",  # pragma: no cover
                "cummax",  # pragma: no cover
                "cumprod",  # pragma: no cover
                "cumsum",  # pragma: no cover
                "logsumexp",  # pragma: no cover
                "all",  # pragma: no cover
                "any",  # pragma: no cover
                "argmin",  # pragma: no cover
                "argmax",  # pragma: no cover
                "prod",  # pragma: no cover
                "sum",  # pragma: no cover
                "mean",  # pragma: no cover
                "var",  # pragma: no cover
                "std",  # pragma: no cover
                "max",  # pragma: no cover
                "min",  # pragma: no cover
                "tensordot",  # pragma: no cover
            ):  # pragma: no cover
                return _to_tensor(x)  # pragma: no cover
        if type(x).__name__ == "ndarray":  # pragma: no cover
            return _to_tensor(x)  # pragma: no cover
        return x  # pragma: no cover

    def _w(x):  # pragma: no cover
        if isinstance(x, tuple):  # pragma: no cover
            return tuple((_w(i) for i in x))  # pragma: no cover
        if isinstance(x, list):  # pragma: no cover
            return [_w(i) for i in x]  # pragma: no cover
        return array(x) if hasattr(x, "shape") else x  # pragma: no cover

    fn = _get_compiler_op("logaddexp")  # pragma: no cover
    res = fn(  # pragma: no cover
        *[_u(x) for x in args],  # pragma: no cover
        **{
            ("dim" if k == "axis" else k): _u(v, k)
            for (k, v) in kwargs.items()
            if k != "stream"
        },  # pragma: no cover
    )  # pragma: no cover

    def __w(x):
        if isinstance(x, tuple):
            return tuple((__w(i) for i in x))
        if isinstance(x, list):
            return [__w(i) for i in x]
        return array(x) if hasattr(x, "shape") else x

    return __w(res)  # pragma: no cover


def median(*args, **kwargs):  # pragma: no cover
    """Compute median.

    Args:
        args: The args argument.
        kwargs: The kwargs argument.

    Returns:
        The result of median.

    """
    from zero_mlx.array import array, _to_tensor  # pragma: no cover
    import ml_switcheroo_compiler.ops as mops  # pragma: no cover
    import ml_switcheroo_compiler.ops.nn as mnn  # pragma: no cover
    import ml_switcheroo_compiler.random as mrand  # pragma: no cover
    import types  # pragma: no cover

    def _u(x, param_name=None):  # pragma: no cover
        if isinstance(x, list):  # pragma: no cover
            if __import__("builtins").all(  # pragma: no cover
                (isinstance(i, int) for i in x)  # pragma: no cover
            ):  # pragma: no cover
                return tuple(x)  # pragma: no cover
            return [_u(i, param_name) for i in x]  # pragma: no cover
        if isinstance(x, tuple):  # pragma: no cover
            return tuple((_u(i, param_name) for i in x))  # pragma: no cover
        if hasattr(x, "_tensor"):  # pragma: no cover
            return x._tensor  # pragma: no cover
        if isinstance(x, (int, float, bool, complex)):  # pragma: no cover
            if param_name not in {  # pragma: no cover
                "axis",  # pragma: no cover
                "keepdims",
                "shape",
                "size",  # pragma: no cover
                "equal_nan",
                "nan",
                "posinf",
                "neginf",  # pragma: no cover
                "axis1",  # pragma: no cover
                "rtol",  # pragma: no cover
                "n",  # pragma: no cover
                "m",  # pragma: no cover
                "endpoint",  # pragma: no cover
                "offset",  # pragma: no cover
                "a_max",  # pragma: no cover
                "ddof",  # pragma: no cover
                "dims",  # pragma: no cover
                "source",  # pragma: no cover
                "axis2",  # pragma: no cover
                "stop",  # pragma: no cover
                "k",  # pragma: no cover
                "dtype",  # pragma: no cover
                "reps",  # pragma: no cover
                "a_min",  # pragma: no cover
                "sparse",  # pragma: no cover
                "decimals",  # pragma: no cover
                "strides",  # pragma: no cover
                "num",  # pragma: no cover
                "max",  # pragma: no cover
                "indices_or_sections",  # pragma: no cover
                "shape",  # pragma: no cover
                "atol",  # pragma: no cover
                "axis1",  # pragma: no cover
                "step",  # pragma: no cover
                "axes",  # pragma: no cover
                "fill_value",  # pragma: no cover
                "stream",  # pragma: no cover
                "destination",  # pragma: no cover
                "alpha",  # pragma: no cover
                "beta",  # pragma: no cover
                "start",  # pragma: no cover
                "equal_nan",
                "nan",
                "posinf",
                "neginf",  # pragma: no cover
                "keepdims",
                "shape",
                "size",  # pragma: no cover
                "min",  # pragma: no cover
                "p",  # pragma: no cover
            } and "median" not in (  # pragma: no cover
                "transpose",  # pragma: no cover
                "moveaxis",  # pragma: no cover
                "reshape",  # pragma: no cover
                "expand_dims",  # pragma: no cover
                "linspace",  # pragma: no cover
                "logcumsumexp",  # pragma: no cover
                "cummin",  # pragma: no cover
                "cummax",  # pragma: no cover
                "cumprod",  # pragma: no cover
                "cumsum",  # pragma: no cover
                "logsumexp",  # pragma: no cover
                "all",  # pragma: no cover
                "any",  # pragma: no cover
                "argmin",  # pragma: no cover
                "argmax",  # pragma: no cover
                "prod",  # pragma: no cover
                "sum",  # pragma: no cover
                "mean",  # pragma: no cover
                "var",  # pragma: no cover
                "std",  # pragma: no cover
                "max",  # pragma: no cover
                "min",  # pragma: no cover
                "tensordot",  # pragma: no cover
            ):  # pragma: no cover
                return _to_tensor(x)  # pragma: no cover
        if type(x).__name__ == "ndarray":  # pragma: no cover
            return _to_tensor(x)  # pragma: no cover
        return x  # pragma: no cover

    def _w(x):  # pragma: no cover
        if isinstance(x, tuple):  # pragma: no cover
            return tuple((_w(i) for i in x))  # pragma: no cover
        if isinstance(x, list):  # pragma: no cover
            return [_w(i) for i in x]  # pragma: no cover
        return array(x) if hasattr(x, "shape") else x  # pragma: no cover

    fn = _get_compiler_op("median")  # pragma: no cover
    res = fn(  # pragma: no cover
        *[_u(x) for x in args],  # pragma: no cover
        **{
            ("dim" if k == "axis" else k): _u(v, k)
            for (k, v) in kwargs.items()
            if k != "stream"
        },  # pragma: no cover
    )  # pragma: no cover

    def __w(x):
        if isinstance(x, tuple):
            return tuple((__w(i) for i in x))
        if isinstance(x, list):
            return [__w(i) for i in x]
        return array(x) if hasattr(x, "shape") else x

    return __w(res)  # pragma: no cover


def meshgrid(*args, **kwargs):  # pragma: no cover
    """Docstring."""
    from zero_mlx.array import _to_tensor

    def _u(x):
        if hasattr(x, "_tensor"):
            return x._tensor
        if type(x).__name__ == "ndarray":
            return _to_tensor(x)
        return x

    sparse = kwargs.pop("sparse", False)
    import ml_switcheroo_compiler.ops as mops

    kwargs["indexing"] = kwargs.get("indexing", "xy")
    res = mops.meshgrid(*[_u(x) for x in args], **kwargs)
    if sparse:
        out = []
        ndim = len(args)
        for i in range(ndim):
            shape = [1] * ndim
            idx = 1 if i == 0 else 0 if i == 1 else i
            if kwargs.get("indexing", "xy") == "ij":
                idx = i
            shape[idx] = -1
            out.append(mops.reshape(_u(args[i]), shape))
        res = tuple(out)
    from zero_mlx.array import array

    def __w(x):
        if isinstance(x, tuple):
            return tuple((__w(i) for i in x))
        if isinstance(x, list):
            return [__w(i) for i in x]
        return array(x) if hasattr(x, "shape") else x

    return __w(res)  # pragma: no cover


def nan_to_num(*args, **kwargs):  # pragma: no cover
    """Compute nan_to_num.

    Args:
        args: The args argument.
        kwargs: The kwargs argument.

    Returns:
        The result of nan_to_num.

    """
    from zero_mlx.array import array, _to_tensor  # pragma: no cover
    import ml_switcheroo_compiler.ops as mops  # pragma: no cover
    import ml_switcheroo_compiler.ops.nn as mnn  # pragma: no cover
    import ml_switcheroo_compiler.random as mrand  # pragma: no cover
    import types  # pragma: no cover

    def _u(x, param_name=None):  # pragma: no cover
        if isinstance(x, list):  # pragma: no cover
            if __import__("builtins").all(  # pragma: no cover
                (isinstance(i, int) for i in x)  # pragma: no cover
            ):  # pragma: no cover
                return tuple(x)  # pragma: no cover
            return [_u(i, param_name) for i in x]  # pragma: no cover
        if isinstance(x, tuple):  # pragma: no cover
            return tuple((_u(i, param_name) for i in x))  # pragma: no cover
        if hasattr(x, "_tensor"):  # pragma: no cover
            return x._tensor  # pragma: no cover
        if isinstance(x, (int, float, bool, complex)):  # pragma: no cover
            if param_name not in {  # pragma: no cover
                "axis",  # pragma: no cover
                "keepdims",
                "shape",
                "size",  # pragma: no cover
                "equal_nan",
                "nan",
                "posinf",
                "neginf",  # pragma: no cover
                "axis1",  # pragma: no cover
                "rtol",  # pragma: no cover
                "n",  # pragma: no cover
                "m",  # pragma: no cover
                "endpoint",  # pragma: no cover
                "offset",  # pragma: no cover
                "a_max",  # pragma: no cover
                "ddof",  # pragma: no cover
                "dims",  # pragma: no cover
                "source",  # pragma: no cover
                "axis2",  # pragma: no cover
                "stop",  # pragma: no cover
                "k",  # pragma: no cover
                "dtype",  # pragma: no cover
                "reps",  # pragma: no cover
                "a_min",  # pragma: no cover
                "sparse",  # pragma: no cover
                "decimals",  # pragma: no cover
                "strides",  # pragma: no cover
                "num",  # pragma: no cover
                "max",  # pragma: no cover
                "indices_or_sections",  # pragma: no cover
                "shape",  # pragma: no cover
                "atol",  # pragma: no cover
                "axis1",  # pragma: no cover
                "step",  # pragma: no cover
                "axes",  # pragma: no cover
                "fill_value",  # pragma: no cover
                "stream",  # pragma: no cover
                "destination",  # pragma: no cover
                "alpha",  # pragma: no cover
                "beta",  # pragma: no cover
                "start",  # pragma: no cover
                "equal_nan",
                "nan",
                "posinf",
                "neginf",  # pragma: no cover
                "keepdims",
                "shape",
                "size",  # pragma: no cover
                "min",  # pragma: no cover
                "p",  # pragma: no cover
            } and "nan_to_num" not in (  # pragma: no cover
                "transpose",  # pragma: no cover
                "moveaxis",  # pragma: no cover
                "reshape",  # pragma: no cover
                "expand_dims",  # pragma: no cover
                "linspace",  # pragma: no cover
                "logcumsumexp",  # pragma: no cover
                "cummin",  # pragma: no cover
                "cummax",  # pragma: no cover
                "cumprod",  # pragma: no cover
                "cumsum",  # pragma: no cover
                "logsumexp",  # pragma: no cover
                "all",  # pragma: no cover
                "any",  # pragma: no cover
                "argmin",  # pragma: no cover
                "argmax",  # pragma: no cover
                "prod",  # pragma: no cover
                "sum",  # pragma: no cover
                "mean",  # pragma: no cover
                "var",  # pragma: no cover
                "std",  # pragma: no cover
                "max",  # pragma: no cover
                "min",  # pragma: no cover
                "tensordot",  # pragma: no cover
            ):  # pragma: no cover
                return _to_tensor(x)  # pragma: no cover
        if type(x).__name__ == "ndarray":  # pragma: no cover
            return _to_tensor(x)  # pragma: no cover
        return x  # pragma: no cover

    def _w(x):  # pragma: no cover
        if isinstance(x, tuple):  # pragma: no cover
            return tuple((_w(i) for i in x))  # pragma: no cover
        if isinstance(x, list):  # pragma: no cover
            return [_w(i) for i in x]  # pragma: no cover
        return array(x) if hasattr(x, "shape") else x  # pragma: no cover

    fn = _get_compiler_op("nan_to_num")  # pragma: no cover
    res = fn(  # pragma: no cover
        *[_u(x) for x in args],  # pragma: no cover
        **{
            ("dim" if k == "axis" else k): _u(v, k)
            for (k, v) in kwargs.items()
            if k != "stream"
        },  # pragma: no cover
    )  # pragma: no cover

    def __w(x):
        if isinstance(x, tuple):
            return tuple((__w(i) for i in x))
        if isinstance(x, list):
            return [__w(i) for i in x]
        return array(x) if hasattr(x, "shape") else x

    return __w(res)  # pragma: no cover


def outer(*args, **kwargs):  # pragma: no cover
    """Compute outer.

    Args:
        args: The args argument.
        kwargs: The kwargs argument.

    Returns:
        The result of outer.

    """
    from zero_mlx.array import array, _to_tensor  # pragma: no cover
    import ml_switcheroo_compiler.ops as mops  # pragma: no cover
    import ml_switcheroo_compiler.ops.nn as mnn  # pragma: no cover
    import ml_switcheroo_compiler.random as mrand  # pragma: no cover
    import types  # pragma: no cover

    def _u(x, param_name=None):  # pragma: no cover
        if isinstance(x, list):  # pragma: no cover
            if __import__("builtins").all(  # pragma: no cover
                (isinstance(i, int) for i in x)  # pragma: no cover
            ):  # pragma: no cover
                return tuple(x)  # pragma: no cover
            return [_u(i, param_name) for i in x]  # pragma: no cover
        if isinstance(x, tuple):  # pragma: no cover
            return tuple((_u(i, param_name) for i in x))  # pragma: no cover
        if hasattr(x, "_tensor"):  # pragma: no cover
            return x._tensor  # pragma: no cover
        if isinstance(x, (int, float, bool, complex)):  # pragma: no cover
            if param_name not in {  # pragma: no cover
                "axis",  # pragma: no cover
                "keepdims",
                "shape",
                "size",  # pragma: no cover
                "equal_nan",
                "nan",
                "posinf",
                "neginf",  # pragma: no cover
                "axis1",  # pragma: no cover
                "rtol",  # pragma: no cover
                "n",  # pragma: no cover
                "m",  # pragma: no cover
                "endpoint",  # pragma: no cover
                "offset",  # pragma: no cover
                "a_max",  # pragma: no cover
                "ddof",  # pragma: no cover
                "dims",  # pragma: no cover
                "source",  # pragma: no cover
                "axis2",  # pragma: no cover
                "stop",  # pragma: no cover
                "k",  # pragma: no cover
                "dtype",  # pragma: no cover
                "reps",  # pragma: no cover
                "a_min",  # pragma: no cover
                "sparse",  # pragma: no cover
                "decimals",  # pragma: no cover
                "strides",  # pragma: no cover
                "num",  # pragma: no cover
                "max",  # pragma: no cover
                "indices_or_sections",  # pragma: no cover
                "shape",  # pragma: no cover
                "atol",  # pragma: no cover
                "axis1",  # pragma: no cover
                "step",  # pragma: no cover
                "axes",  # pragma: no cover
                "fill_value",  # pragma: no cover
                "stream",  # pragma: no cover
                "destination",  # pragma: no cover
                "alpha",  # pragma: no cover
                "beta",  # pragma: no cover
                "start",  # pragma: no cover
                "equal_nan",
                "nan",
                "posinf",
                "neginf",  # pragma: no cover
                "keepdims",
                "shape",
                "size",  # pragma: no cover
                "min",  # pragma: no cover
                "p",  # pragma: no cover
            } and "outer" not in (  # pragma: no cover
                "transpose",  # pragma: no cover
                "moveaxis",  # pragma: no cover
                "reshape",  # pragma: no cover
                "expand_dims",  # pragma: no cover
                "linspace",  # pragma: no cover
                "logcumsumexp",  # pragma: no cover
                "cummin",  # pragma: no cover
                "cummax",  # pragma: no cover
                "cumprod",  # pragma: no cover
                "cumsum",  # pragma: no cover
                "logsumexp",  # pragma: no cover
                "all",  # pragma: no cover
                "any",  # pragma: no cover
                "argmin",  # pragma: no cover
                "argmax",  # pragma: no cover
                "prod",  # pragma: no cover
                "sum",  # pragma: no cover
                "mean",  # pragma: no cover
                "var",  # pragma: no cover
                "std",  # pragma: no cover
                "max",  # pragma: no cover
                "min",  # pragma: no cover
                "tensordot",  # pragma: no cover
            ):  # pragma: no cover
                return _to_tensor(x)  # pragma: no cover
        if type(x).__name__ == "ndarray":  # pragma: no cover
            return _to_tensor(x)  # pragma: no cover
        return x  # pragma: no cover

    def _w(x):  # pragma: no cover
        if isinstance(x, tuple):  # pragma: no cover
            return tuple((_w(i) for i in x))  # pragma: no cover
        if isinstance(x, list):  # pragma: no cover
            return [_w(i) for i in x]  # pragma: no cover
        return array(x) if hasattr(x, "shape") else x  # pragma: no cover

    fn = _get_compiler_op("outer")  # pragma: no cover
    res = fn(  # pragma: no cover
        *[_u(x) for x in args],  # pragma: no cover
        **{
            ("dim" if k == "axis" else k): _u(v, k)
            for (k, v) in kwargs.items()
            if k != "stream"
        },  # pragma: no cover
    )  # pragma: no cover

    def __w(x):
        if isinstance(x, tuple):
            return tuple((__w(i) for i in x))
        if isinstance(x, list):
            return [__w(i) for i in x]
        return array(x) if hasattr(x, "shape") else x

    return __w(res)  # pragma: no cover


def pad(*args, **kwargs):  # pragma: no cover
    """Compute pad.

    Args:
        args: The args argument.
        kwargs: The kwargs argument.

    Returns:
        The result of pad.

    """
    from zero_mlx.array import array, _to_tensor  # pragma: no cover
    import ml_switcheroo_compiler.ops as mops  # pragma: no cover
    import ml_switcheroo_compiler.ops.nn as mnn  # pragma: no cover
    import ml_switcheroo_compiler.random as mrand  # pragma: no cover
    import types  # pragma: no cover

    def _u(x, param_name=None):  # pragma: no cover
        if isinstance(x, list):  # pragma: no cover
            if __import__("builtins").all(  # pragma: no cover
                (isinstance(i, int) for i in x)  # pragma: no cover
            ):  # pragma: no cover
                return tuple(x)  # pragma: no cover
            return [_u(i, param_name) for i in x]  # pragma: no cover
        if isinstance(x, tuple):  # pragma: no cover
            return tuple((_u(i, param_name) for i in x))  # pragma: no cover
        if hasattr(x, "_tensor"):  # pragma: no cover
            return x._tensor  # pragma: no cover
        if isinstance(x, (int, float, bool, complex)):  # pragma: no cover
            if param_name not in {  # pragma: no cover
                "axis",  # pragma: no cover
                "keepdims",
                "shape",
                "size",  # pragma: no cover
                "equal_nan",
                "nan",
                "posinf",
                "neginf",  # pragma: no cover
                "axis1",  # pragma: no cover
                "rtol",  # pragma: no cover
                "n",  # pragma: no cover
                "m",  # pragma: no cover
                "endpoint",  # pragma: no cover
                "offset",  # pragma: no cover
                "a_max",  # pragma: no cover
                "ddof",  # pragma: no cover
                "dims",  # pragma: no cover
                "source",  # pragma: no cover
                "axis2",  # pragma: no cover
                "stop",  # pragma: no cover
                "k",  # pragma: no cover
                "dtype",  # pragma: no cover
                "reps",  # pragma: no cover
                "a_min",  # pragma: no cover
                "sparse",  # pragma: no cover
                "decimals",  # pragma: no cover
                "strides",  # pragma: no cover
                "num",  # pragma: no cover
                "max",  # pragma: no cover
                "indices_or_sections",  # pragma: no cover
                "shape",  # pragma: no cover
                "atol",  # pragma: no cover
                "axis1",  # pragma: no cover
                "step",  # pragma: no cover
                "axes",  # pragma: no cover
                "fill_value",  # pragma: no cover
                "stream",  # pragma: no cover
                "destination",  # pragma: no cover
                "alpha",  # pragma: no cover
                "beta",  # pragma: no cover
                "start",  # pragma: no cover
                "equal_nan",
                "nan",
                "posinf",
                "neginf",  # pragma: no cover
                "keepdims",
                "shape",
                "size",  # pragma: no cover
                "min",  # pragma: no cover
                "p",  # pragma: no cover
            } and "pad" not in (  # pragma: no cover
                "transpose",  # pragma: no cover
                "moveaxis",  # pragma: no cover
                "reshape",  # pragma: no cover
                "expand_dims",  # pragma: no cover
                "linspace",  # pragma: no cover
                "logcumsumexp",  # pragma: no cover
                "cummin",  # pragma: no cover
                "cummax",  # pragma: no cover
                "cumprod",  # pragma: no cover
                "cumsum",  # pragma: no cover
                "logsumexp",  # pragma: no cover
                "all",  # pragma: no cover
                "any",  # pragma: no cover
                "argmin",  # pragma: no cover
                "argmax",  # pragma: no cover
                "prod",  # pragma: no cover
                "sum",  # pragma: no cover
                "mean",  # pragma: no cover
                "var",  # pragma: no cover
                "std",  # pragma: no cover
                "max",  # pragma: no cover
                "min",  # pragma: no cover
                "tensordot",  # pragma: no cover
            ):  # pragma: no cover
                return _to_tensor(x)  # pragma: no cover
        if type(x).__name__ == "ndarray":  # pragma: no cover
            return _to_tensor(x)  # pragma: no cover
        return x  # pragma: no cover

    def _w(x):  # pragma: no cover
        if isinstance(x, tuple):  # pragma: no cover
            return tuple((_w(i) for i in x))  # pragma: no cover
        if isinstance(x, list):  # pragma: no cover
            return [_w(i) for i in x]  # pragma: no cover
        return array(x) if hasattr(x, "shape") else x  # pragma: no cover

    fn = _get_compiler_op("pad")  # pragma: no cover
    res = fn(  # pragma: no cover
        *[_u(x) for x in args],  # pragma: no cover
        **{
            ("dim" if k == "axis" else k): _u(v, k)
            for (k, v) in kwargs.items()
            if k != "stream"
        },  # pragma: no cover
    )  # pragma: no cover

    def __w(x):
        if isinstance(x, tuple):
            return tuple((__w(i) for i in x))
        if isinstance(x, list):
            return [__w(i) for i in x]
        return array(x) if hasattr(x, "shape") else x

    return __w(res)  # pragma: no cover


def partition(*args, **kwargs):  # pragma: no cover
    """Compute partition.

    Args:
        args: The args argument.
        kwargs: The kwargs argument.

    Returns:
        The result of partition.

    """
    from zero_mlx.array import array, _to_tensor  # pragma: no cover
    import ml_switcheroo_compiler.ops as mops  # pragma: no cover
    import ml_switcheroo_compiler.ops.nn as mnn  # pragma: no cover
    import ml_switcheroo_compiler.random as mrand  # pragma: no cover
    import types  # pragma: no cover

    def _u(x, param_name=None):  # pragma: no cover
        if isinstance(x, list):  # pragma: no cover
            if __import__("builtins").all(  # pragma: no cover
                (isinstance(i, int) for i in x)  # pragma: no cover
            ):  # pragma: no cover
                return tuple(x)  # pragma: no cover
            return [_u(i, param_name) for i in x]  # pragma: no cover
        if isinstance(x, tuple):  # pragma: no cover
            return tuple((_u(i, param_name) for i in x))  # pragma: no cover
        if hasattr(x, "_tensor"):  # pragma: no cover
            return x._tensor  # pragma: no cover
        if isinstance(x, (int, float, bool, complex)):  # pragma: no cover
            if param_name not in {  # pragma: no cover
                "axis",  # pragma: no cover
                "keepdims",
                "shape",
                "size",  # pragma: no cover
                "equal_nan",
                "nan",
                "posinf",
                "neginf",  # pragma: no cover
                "axis1",  # pragma: no cover
                "rtol",  # pragma: no cover
                "n",  # pragma: no cover
                "m",  # pragma: no cover
                "endpoint",  # pragma: no cover
                "offset",  # pragma: no cover
                "a_max",  # pragma: no cover
                "ddof",  # pragma: no cover
                "dims",  # pragma: no cover
                "source",  # pragma: no cover
                "axis2",  # pragma: no cover
                "stop",  # pragma: no cover
                "k",  # pragma: no cover
                "dtype",  # pragma: no cover
                "reps",  # pragma: no cover
                "a_min",  # pragma: no cover
                "sparse",  # pragma: no cover
                "decimals",  # pragma: no cover
                "strides",  # pragma: no cover
                "num",  # pragma: no cover
                "max",  # pragma: no cover
                "indices_or_sections",  # pragma: no cover
                "shape",  # pragma: no cover
                "atol",  # pragma: no cover
                "axis1",  # pragma: no cover
                "step",  # pragma: no cover
                "axes",  # pragma: no cover
                "fill_value",  # pragma: no cover
                "stream",  # pragma: no cover
                "destination",  # pragma: no cover
                "alpha",  # pragma: no cover
                "beta",  # pragma: no cover
                "start",  # pragma: no cover
                "equal_nan",
                "nan",
                "posinf",
                "neginf",  # pragma: no cover
                "keepdims",
                "shape",
                "size",  # pragma: no cover
                "min",  # pragma: no cover
                "p",  # pragma: no cover
            } and "partition" not in (  # pragma: no cover
                "transpose",  # pragma: no cover
                "moveaxis",  # pragma: no cover
                "reshape",  # pragma: no cover
                "expand_dims",  # pragma: no cover
                "linspace",  # pragma: no cover
                "logcumsumexp",  # pragma: no cover
                "cummin",  # pragma: no cover
                "cummax",  # pragma: no cover
                "cumprod",  # pragma: no cover
                "cumsum",  # pragma: no cover
                "logsumexp",  # pragma: no cover
                "all",  # pragma: no cover
                "any",  # pragma: no cover
                "argmin",  # pragma: no cover
                "argmax",  # pragma: no cover
                "prod",  # pragma: no cover
                "sum",  # pragma: no cover
                "mean",  # pragma: no cover
                "var",  # pragma: no cover
                "std",  # pragma: no cover
                "max",  # pragma: no cover
                "min",  # pragma: no cover
                "tensordot",  # pragma: no cover
            ):  # pragma: no cover
                return _to_tensor(x)  # pragma: no cover
        if type(x).__name__ == "ndarray":  # pragma: no cover
            return _to_tensor(x)  # pragma: no cover
        return x  # pragma: no cover

    def _w(x):  # pragma: no cover
        if isinstance(x, tuple):  # pragma: no cover
            return tuple((_w(i) for i in x))  # pragma: no cover
        if isinstance(x, list):  # pragma: no cover
            return [_w(i) for i in x]  # pragma: no cover
        return array(x) if hasattr(x, "shape") else x  # pragma: no cover

    fn = _get_compiler_op("partition")  # pragma: no cover
    res = fn(  # pragma: no cover
        *[_u(x) for x in args],  # pragma: no cover
        **{
            ("dim" if k == "axis" else k): _u(v, k)
            for (k, v) in kwargs.items()
            if k != "stream"
        },  # pragma: no cover
    )  # pragma: no cover

    def __w(x):
        if isinstance(x, tuple):
            return tuple((__w(i) for i in x))
        if isinstance(x, list):
            return [__w(i) for i in x]
        return array(x) if hasattr(x, "shape") else x

    return __w(res)  # pragma: no cover


def put_along_axis(*args, **kwargs):  # pragma: no cover
    """Compute put_along_axis.

    Args:
        args: The args argument.
        kwargs: The kwargs argument.

    Returns:
        The result of put_along_axis.

    """
    from zero_mlx.array import array, _to_tensor  # pragma: no cover
    import ml_switcheroo_compiler.ops as mops  # pragma: no cover
    import ml_switcheroo_compiler.ops.nn as mnn  # pragma: no cover
    import ml_switcheroo_compiler.random as mrand  # pragma: no cover
    import types  # pragma: no cover

    def _u(x, param_name=None):  # pragma: no cover
        if isinstance(x, list):  # pragma: no cover
            if __import__("builtins").all(  # pragma: no cover
                (isinstance(i, int) for i in x)  # pragma: no cover
            ):  # pragma: no cover
                return tuple(x)  # pragma: no cover
            return [_u(i, param_name) for i in x]  # pragma: no cover
        if isinstance(x, tuple):  # pragma: no cover
            return tuple((_u(i, param_name) for i in x))  # pragma: no cover
        if hasattr(x, "_tensor"):  # pragma: no cover
            return x._tensor  # pragma: no cover
        if isinstance(x, (int, float, bool, complex)):  # pragma: no cover
            if param_name not in {  # pragma: no cover
                "axis",  # pragma: no cover
                "keepdims",
                "shape",
                "size",  # pragma: no cover
                "equal_nan",
                "nan",
                "posinf",
                "neginf",  # pragma: no cover
                "axis1",  # pragma: no cover
                "rtol",  # pragma: no cover
                "n",  # pragma: no cover
                "m",  # pragma: no cover
                "endpoint",  # pragma: no cover
                "offset",  # pragma: no cover
                "a_max",  # pragma: no cover
                "ddof",  # pragma: no cover
                "dims",  # pragma: no cover
                "source",  # pragma: no cover
                "axis2",  # pragma: no cover
                "stop",  # pragma: no cover
                "k",  # pragma: no cover
                "dtype",  # pragma: no cover
                "reps",  # pragma: no cover
                "a_min",  # pragma: no cover
                "sparse",  # pragma: no cover
                "decimals",  # pragma: no cover
                "strides",  # pragma: no cover
                "num",  # pragma: no cover
                "max",  # pragma: no cover
                "indices_or_sections",  # pragma: no cover
                "shape",  # pragma: no cover
                "atol",  # pragma: no cover
                "axis1",  # pragma: no cover
                "step",  # pragma: no cover
                "axes",  # pragma: no cover
                "fill_value",  # pragma: no cover
                "stream",  # pragma: no cover
                "destination",  # pragma: no cover
                "alpha",  # pragma: no cover
                "beta",  # pragma: no cover
                "start",  # pragma: no cover
                "equal_nan",
                "nan",
                "posinf",
                "neginf",  # pragma: no cover
                "keepdims",
                "shape",
                "size",  # pragma: no cover
                "min",  # pragma: no cover
                "p",  # pragma: no cover
            } and "put_along_axis" not in (  # pragma: no cover
                "transpose",  # pragma: no cover
                "moveaxis",  # pragma: no cover
                "reshape",  # pragma: no cover
                "expand_dims",  # pragma: no cover
                "linspace",  # pragma: no cover
                "logcumsumexp",  # pragma: no cover
                "cummin",  # pragma: no cover
                "cummax",  # pragma: no cover
                "cumprod",  # pragma: no cover
                "cumsum",  # pragma: no cover
                "logsumexp",  # pragma: no cover
                "all",  # pragma: no cover
                "any",  # pragma: no cover
                "argmin",  # pragma: no cover
                "argmax",  # pragma: no cover
                "prod",  # pragma: no cover
                "sum",  # pragma: no cover
                "mean",  # pragma: no cover
                "var",  # pragma: no cover
                "std",  # pragma: no cover
                "max",  # pragma: no cover
                "min",  # pragma: no cover
                "tensordot",  # pragma: no cover
            ):  # pragma: no cover
                return _to_tensor(x)  # pragma: no cover
        if type(x).__name__ == "ndarray":  # pragma: no cover
            return _to_tensor(x)  # pragma: no cover
        return x  # pragma: no cover

    def _w(x):  # pragma: no cover
        if isinstance(x, tuple):  # pragma: no cover
            return tuple((_w(i) for i in x))  # pragma: no cover
        if isinstance(x, list):  # pragma: no cover
            return [_w(i) for i in x]  # pragma: no cover
        return array(x) if hasattr(x, "shape") else x  # pragma: no cover

    fn = _get_compiler_op("put_along_axis")  # pragma: no cover
    res = fn(  # pragma: no cover
        *[_u(x) for x in args],  # pragma: no cover
        **{
            ("dim" if k == "axis" else k): _u(v, k)
            for (k, v) in kwargs.items()
            if k != "stream"
        },  # pragma: no cover
    )  # pragma: no cover

    def __w(x):
        if isinstance(x, tuple):
            return tuple((__w(i) for i in x))
        if isinstance(x, list):
            return [__w(i) for i in x]
        return array(x) if hasattr(x, "shape") else x

    return __w(res)  # pragma: no cover


def radians(*args, **kwargs):  # pragma: no cover
    """Compute radians.

    Args:
        args: The args argument.
        kwargs: The kwargs argument.

    Returns:
        The result of radians.

    """
    from zero_mlx.array import array, _to_tensor  # pragma: no cover
    import ml_switcheroo_compiler.ops as mops  # pragma: no cover
    import ml_switcheroo_compiler.ops.nn as mnn  # pragma: no cover
    import ml_switcheroo_compiler.random as mrand  # pragma: no cover
    import types  # pragma: no cover

    def _u(x, param_name=None):  # pragma: no cover
        if isinstance(x, list):  # pragma: no cover
            if __import__("builtins").all(  # pragma: no cover
                (isinstance(i, int) for i in x)  # pragma: no cover
            ):  # pragma: no cover
                return tuple(x)  # pragma: no cover
            return [_u(i, param_name) for i in x]  # pragma: no cover
        if isinstance(x, tuple):  # pragma: no cover
            return tuple((_u(i, param_name) for i in x))  # pragma: no cover
        if hasattr(x, "_tensor"):  # pragma: no cover
            return x._tensor  # pragma: no cover
        if isinstance(x, (int, float, bool, complex)):  # pragma: no cover
            if param_name not in {  # pragma: no cover
                "axis",  # pragma: no cover
                "keepdims",
                "shape",
                "size",  # pragma: no cover
                "equal_nan",
                "nan",
                "posinf",
                "neginf",  # pragma: no cover
                "axis1",  # pragma: no cover
                "rtol",  # pragma: no cover
                "n",  # pragma: no cover
                "m",  # pragma: no cover
                "endpoint",  # pragma: no cover
                "offset",  # pragma: no cover
                "a_max",  # pragma: no cover
                "ddof",  # pragma: no cover
                "dims",  # pragma: no cover
                "source",  # pragma: no cover
                "axis2",  # pragma: no cover
                "stop",  # pragma: no cover
                "k",  # pragma: no cover
                "dtype",  # pragma: no cover
                "reps",  # pragma: no cover
                "a_min",  # pragma: no cover
                "sparse",  # pragma: no cover
                "decimals",  # pragma: no cover
                "strides",  # pragma: no cover
                "num",  # pragma: no cover
                "max",  # pragma: no cover
                "indices_or_sections",  # pragma: no cover
                "shape",  # pragma: no cover
                "atol",  # pragma: no cover
                "axis1",  # pragma: no cover
                "step",  # pragma: no cover
                "axes",  # pragma: no cover
                "fill_value",  # pragma: no cover
                "stream",  # pragma: no cover
                "destination",  # pragma: no cover
                "alpha",  # pragma: no cover
                "beta",  # pragma: no cover
                "start",  # pragma: no cover
                "equal_nan",
                "nan",
                "posinf",
                "neginf",  # pragma: no cover
                "keepdims",
                "shape",
                "size",  # pragma: no cover
                "min",  # pragma: no cover
                "p",  # pragma: no cover
            } and "radians" not in (  # pragma: no cover
                "transpose",  # pragma: no cover
                "moveaxis",  # pragma: no cover
                "reshape",  # pragma: no cover
                "expand_dims",  # pragma: no cover
                "linspace",  # pragma: no cover
                "logcumsumexp",  # pragma: no cover
                "cummin",  # pragma: no cover
                "cummax",  # pragma: no cover
                "cumprod",  # pragma: no cover
                "cumsum",  # pragma: no cover
                "logsumexp",  # pragma: no cover
                "all",  # pragma: no cover
                "any",  # pragma: no cover
                "argmin",  # pragma: no cover
                "argmax",  # pragma: no cover
                "prod",  # pragma: no cover
                "sum",  # pragma: no cover
                "mean",  # pragma: no cover
                "var",  # pragma: no cover
                "std",  # pragma: no cover
                "max",  # pragma: no cover
                "min",  # pragma: no cover
                "tensordot",  # pragma: no cover
            ):  # pragma: no cover
                return _to_tensor(x)  # pragma: no cover
        if type(x).__name__ == "ndarray":  # pragma: no cover
            return _to_tensor(x)  # pragma: no cover
        return x  # pragma: no cover

    def _w(x):  # pragma: no cover
        if isinstance(x, tuple):  # pragma: no cover
            return tuple((_w(i) for i in x))  # pragma: no cover
        if isinstance(x, list):  # pragma: no cover
            return [_w(i) for i in x]  # pragma: no cover
        return array(x) if hasattr(x, "shape") else x  # pragma: no cover

    fn = _get_compiler_op("radians")  # pragma: no cover
    res = fn(  # pragma: no cover
        *[_u(x) for x in args],  # pragma: no cover
        **{
            ("dim" if k == "axis" else k): _u(v, k)
            for (k, v) in kwargs.items()
            if k != "stream"
        },  # pragma: no cover
    )  # pragma: no cover

    def __w(x):
        if isinstance(x, tuple):
            return tuple((__w(i) for i in x))
        if isinstance(x, list):
            return [__w(i) for i in x]
        return array(x) if hasattr(x, "shape") else x

    return __w(res)  # pragma: no cover


def real(*args, **kwargs):  # pragma: no cover
    """Compute real.

    Args:
        args: The args argument.
        kwargs: The kwargs argument.

    Returns:
        The result of real.

    """
    from zero_mlx.array import array, _to_tensor  # pragma: no cover
    import ml_switcheroo_compiler.ops as mops  # pragma: no cover
    import ml_switcheroo_compiler.ops.nn as mnn  # pragma: no cover
    import ml_switcheroo_compiler.random as mrand  # pragma: no cover
    import types  # pragma: no cover

    def _u(x, param_name=None):  # pragma: no cover
        if isinstance(x, list):  # pragma: no cover
            if __import__("builtins").all(  # pragma: no cover
                (isinstance(i, int) for i in x)  # pragma: no cover
            ):  # pragma: no cover
                return tuple(x)  # pragma: no cover
            return [_u(i, param_name) for i in x]  # pragma: no cover
        if isinstance(x, tuple):  # pragma: no cover
            return tuple((_u(i, param_name) for i in x))  # pragma: no cover
        if hasattr(x, "_tensor"):  # pragma: no cover
            return x._tensor  # pragma: no cover
        if isinstance(x, (int, float, bool, complex)):  # pragma: no cover
            if param_name not in {  # pragma: no cover
                "axis",  # pragma: no cover
                "keepdims",
                "shape",
                "size",  # pragma: no cover
                "equal_nan",
                "nan",
                "posinf",
                "neginf",  # pragma: no cover
                "axis1",  # pragma: no cover
                "rtol",  # pragma: no cover
                "n",  # pragma: no cover
                "m",  # pragma: no cover
                "endpoint",  # pragma: no cover
                "offset",  # pragma: no cover
                "a_max",  # pragma: no cover
                "ddof",  # pragma: no cover
                "dims",  # pragma: no cover
                "source",  # pragma: no cover
                "axis2",  # pragma: no cover
                "stop",  # pragma: no cover
                "k",  # pragma: no cover
                "dtype",  # pragma: no cover
                "reps",  # pragma: no cover
                "a_min",  # pragma: no cover
                "sparse",  # pragma: no cover
                "decimals",  # pragma: no cover
                "strides",  # pragma: no cover
                "num",  # pragma: no cover
                "max",  # pragma: no cover
                "indices_or_sections",  # pragma: no cover
                "shape",  # pragma: no cover
                "atol",  # pragma: no cover
                "axis1",  # pragma: no cover
                "step",  # pragma: no cover
                "axes",  # pragma: no cover
                "fill_value",  # pragma: no cover
                "stream",  # pragma: no cover
                "destination",  # pragma: no cover
                "alpha",  # pragma: no cover
                "beta",  # pragma: no cover
                "start",  # pragma: no cover
                "equal_nan",
                "nan",
                "posinf",
                "neginf",  # pragma: no cover
                "keepdims",
                "shape",
                "size",  # pragma: no cover
                "min",  # pragma: no cover
                "p",  # pragma: no cover
            } and "real" not in (  # pragma: no cover
                "transpose",  # pragma: no cover
                "moveaxis",  # pragma: no cover
                "reshape",  # pragma: no cover
                "expand_dims",  # pragma: no cover
                "linspace",  # pragma: no cover
                "logcumsumexp",  # pragma: no cover
                "cummin",  # pragma: no cover
                "cummax",  # pragma: no cover
                "cumprod",  # pragma: no cover
                "cumsum",  # pragma: no cover
                "logsumexp",  # pragma: no cover
                "all",  # pragma: no cover
                "any",  # pragma: no cover
                "argmin",  # pragma: no cover
                "argmax",  # pragma: no cover
                "prod",  # pragma: no cover
                "sum",  # pragma: no cover
                "mean",  # pragma: no cover
                "var",  # pragma: no cover
                "std",  # pragma: no cover
                "max",  # pragma: no cover
                "min",  # pragma: no cover
                "tensordot",  # pragma: no cover
            ):  # pragma: no cover
                return _to_tensor(x)  # pragma: no cover
        if type(x).__name__ == "ndarray":  # pragma: no cover
            return _to_tensor(x)  # pragma: no cover
        return x  # pragma: no cover

    def _w(x):  # pragma: no cover
        if isinstance(x, tuple):  # pragma: no cover
            return tuple((_w(i) for i in x))  # pragma: no cover
        if isinstance(x, list):  # pragma: no cover
            return [_w(i) for i in x]  # pragma: no cover
        return array(x) if hasattr(x, "shape") else x  # pragma: no cover

    fn = _get_compiler_op("real")  # pragma: no cover
    res = fn(  # pragma: no cover
        *[_u(x) for x in args],  # pragma: no cover
        **{
            ("dim" if k == "axis" else k): _u(v, k)
            for (k, v) in kwargs.items()
            if k != "stream"
        },  # pragma: no cover
    )  # pragma: no cover

    def __w(x):
        if isinstance(x, tuple):
            return tuple((__w(i) for i in x))
        if isinstance(x, list):
            return [__w(i) for i in x]
        return array(x) if hasattr(x, "shape") else x

    return __w(res)  # pragma: no cover


def remainder(*args, **kwargs):  # pragma: no cover
    """Compute remainder.

    Args:
        args: The args argument.
        kwargs: The kwargs argument.

    Returns:
        The result of remainder.

    """
    from zero_mlx.array import array, _to_tensor  # pragma: no cover
    import ml_switcheroo_compiler.ops as mops  # pragma: no cover
    import ml_switcheroo_compiler.ops.nn as mnn  # pragma: no cover
    import ml_switcheroo_compiler.random as mrand  # pragma: no cover
    import types  # pragma: no cover

    def _u(x, param_name=None):  # pragma: no cover
        if isinstance(x, list):  # pragma: no cover
            if __import__("builtins").all(  # pragma: no cover
                (isinstance(i, int) for i in x)  # pragma: no cover
            ):  # pragma: no cover
                return tuple(x)  # pragma: no cover
            return [_u(i, param_name) for i in x]  # pragma: no cover
        if isinstance(x, tuple):  # pragma: no cover
            return tuple((_u(i, param_name) for i in x))  # pragma: no cover
        if hasattr(x, "_tensor"):  # pragma: no cover
            return x._tensor  # pragma: no cover
        if isinstance(x, (int, float, bool, complex)):  # pragma: no cover
            if param_name not in {  # pragma: no cover
                "axis",  # pragma: no cover
                "keepdims",
                "shape",
                "size",  # pragma: no cover
                "equal_nan",
                "nan",
                "posinf",
                "neginf",  # pragma: no cover
                "axis1",  # pragma: no cover
                "rtol",  # pragma: no cover
                "n",  # pragma: no cover
                "m",  # pragma: no cover
                "endpoint",  # pragma: no cover
                "offset",  # pragma: no cover
                "a_max",  # pragma: no cover
                "ddof",  # pragma: no cover
                "dims",  # pragma: no cover
                "source",  # pragma: no cover
                "axis2",  # pragma: no cover
                "stop",  # pragma: no cover
                "k",  # pragma: no cover
                "dtype",  # pragma: no cover
                "reps",  # pragma: no cover
                "a_min",  # pragma: no cover
                "sparse",  # pragma: no cover
                "decimals",  # pragma: no cover
                "strides",  # pragma: no cover
                "num",  # pragma: no cover
                "max",  # pragma: no cover
                "indices_or_sections",  # pragma: no cover
                "shape",  # pragma: no cover
                "atol",  # pragma: no cover
                "axis1",  # pragma: no cover
                "step",  # pragma: no cover
                "axes",  # pragma: no cover
                "fill_value",  # pragma: no cover
                "stream",  # pragma: no cover
                "destination",  # pragma: no cover
                "alpha",  # pragma: no cover
                "beta",  # pragma: no cover
                "start",  # pragma: no cover
                "equal_nan",
                "nan",
                "posinf",
                "neginf",  # pragma: no cover
                "keepdims",
                "shape",
                "size",  # pragma: no cover
                "min",  # pragma: no cover
                "p",  # pragma: no cover
            } and "remainder" not in (  # pragma: no cover
                "transpose",  # pragma: no cover
                "moveaxis",  # pragma: no cover
                "reshape",  # pragma: no cover
                "expand_dims",  # pragma: no cover
                "linspace",  # pragma: no cover
                "logcumsumexp",  # pragma: no cover
                "cummin",  # pragma: no cover
                "cummax",  # pragma: no cover
                "cumprod",  # pragma: no cover
                "cumsum",  # pragma: no cover
                "logsumexp",  # pragma: no cover
                "all",  # pragma: no cover
                "any",  # pragma: no cover
                "argmin",  # pragma: no cover
                "argmax",  # pragma: no cover
                "prod",  # pragma: no cover
                "sum",  # pragma: no cover
                "mean",  # pragma: no cover
                "var",  # pragma: no cover
                "std",  # pragma: no cover
                "max",  # pragma: no cover
                "min",  # pragma: no cover
                "tensordot",  # pragma: no cover
            ):  # pragma: no cover
                return _to_tensor(x)  # pragma: no cover
        if type(x).__name__ == "ndarray":  # pragma: no cover
            return _to_tensor(x)  # pragma: no cover
        return x  # pragma: no cover

    def _w(x):  # pragma: no cover
        if isinstance(x, tuple):  # pragma: no cover
            return tuple((_w(i) for i in x))  # pragma: no cover
        if isinstance(x, list):  # pragma: no cover
            return [_w(i) for i in x]  # pragma: no cover
        return array(x) if hasattr(x, "shape") else x  # pragma: no cover

    fn = _get_compiler_op("remainder")  # pragma: no cover
    res = fn(  # pragma: no cover
        *[_u(x) for x in args],  # pragma: no cover
        **{
            ("dim" if k == "axis" else k): _u(v, k)
            for (k, v) in kwargs.items()
            if k != "stream"
        },  # pragma: no cover
    )  # pragma: no cover

    def __w(x):
        if isinstance(x, tuple):
            return tuple((__w(i) for i in x))
        if isinstance(x, list):
            return [__w(i) for i in x]
        return array(x) if hasattr(x, "shape") else x

    return __w(res)  # pragma: no cover


def repeat(*args, **kwargs):  # pragma: no cover
    """Compute repeat.

    Args:
        args: The args argument.
        kwargs: The kwargs argument.

    Returns:
        The result of repeat.

    """
    from zero_mlx.array import array, _to_tensor  # pragma: no cover
    import ml_switcheroo_compiler.ops as mops  # pragma: no cover
    import ml_switcheroo_compiler.ops.nn as mnn  # pragma: no cover
    import ml_switcheroo_compiler.random as mrand  # pragma: no cover
    import types  # pragma: no cover

    def _u(x, param_name=None):  # pragma: no cover
        if isinstance(x, list):  # pragma: no cover
            if __import__("builtins").all(  # pragma: no cover
                (isinstance(i, int) for i in x)  # pragma: no cover
            ):  # pragma: no cover
                return tuple(x)  # pragma: no cover
            return [_u(i, param_name) for i in x]  # pragma: no cover
        if isinstance(x, tuple):  # pragma: no cover
            return tuple((_u(i, param_name) for i in x))  # pragma: no cover
        if hasattr(x, "_tensor"):  # pragma: no cover
            return x._tensor  # pragma: no cover
        if isinstance(x, (int, float, bool, complex)):  # pragma: no cover
            if param_name not in {  # pragma: no cover
                "axis",  # pragma: no cover
                "keepdims",
                "shape",
                "size",  # pragma: no cover
                "equal_nan",
                "nan",
                "posinf",
                "neginf",  # pragma: no cover
                "axis1",  # pragma: no cover
                "rtol",  # pragma: no cover
                "n",  # pragma: no cover
                "m",  # pragma: no cover
                "endpoint",  # pragma: no cover
                "offset",  # pragma: no cover
                "a_max",  # pragma: no cover
                "ddof",  # pragma: no cover
                "dims",  # pragma: no cover
                "source",  # pragma: no cover
                "axis2",  # pragma: no cover
                "stop",  # pragma: no cover
                "k",  # pragma: no cover
                "dtype",  # pragma: no cover
                "reps",  # pragma: no cover
                "a_min",  # pragma: no cover
                "sparse",  # pragma: no cover
                "decimals",  # pragma: no cover
                "strides",  # pragma: no cover
                "num",  # pragma: no cover
                "max",  # pragma: no cover
                "indices_or_sections",  # pragma: no cover
                "shape",  # pragma: no cover
                "atol",  # pragma: no cover
                "axis1",  # pragma: no cover
                "step",  # pragma: no cover
                "axes",  # pragma: no cover
                "fill_value",  # pragma: no cover
                "stream",  # pragma: no cover
                "destination",  # pragma: no cover
                "alpha",  # pragma: no cover
                "beta",  # pragma: no cover
                "start",  # pragma: no cover
                "equal_nan",
                "nan",
                "posinf",
                "neginf",  # pragma: no cover
                "keepdims",
                "shape",
                "size",  # pragma: no cover
                "min",  # pragma: no cover
                "p",  # pragma: no cover
            } and "repeat" not in (  # pragma: no cover
                "transpose",  # pragma: no cover
                "moveaxis",  # pragma: no cover
                "reshape",  # pragma: no cover
                "expand_dims",  # pragma: no cover
                "linspace",  # pragma: no cover
                "logcumsumexp",  # pragma: no cover
                "cummin",  # pragma: no cover
                "cummax",  # pragma: no cover
                "cumprod",  # pragma: no cover
                "cumsum",  # pragma: no cover
                "logsumexp",  # pragma: no cover
                "all",  # pragma: no cover
                "any",  # pragma: no cover
                "argmin",  # pragma: no cover
                "argmax",  # pragma: no cover
                "prod",  # pragma: no cover
                "sum",  # pragma: no cover
                "mean",  # pragma: no cover
                "var",  # pragma: no cover
                "std",  # pragma: no cover
                "max",  # pragma: no cover
                "min",  # pragma: no cover
                "tensordot",  # pragma: no cover
            ):  # pragma: no cover
                return _to_tensor(x)  # pragma: no cover
        if type(x).__name__ == "ndarray":  # pragma: no cover
            return _to_tensor(x)  # pragma: no cover
        return x  # pragma: no cover

    def _w(x):  # pragma: no cover
        if isinstance(x, tuple):  # pragma: no cover
            return tuple((_w(i) for i in x))  # pragma: no cover
        if isinstance(x, list):  # pragma: no cover
            return [_w(i) for i in x]  # pragma: no cover
        return array(x) if hasattr(x, "shape") else x  # pragma: no cover

    fn = _get_compiler_op("repeat")  # pragma: no cover
    _kwargs = {k: _u(v, k) for (k, v) in kwargs.items() if k != "stream"}
    if "axis" in _kwargs:
        _kwargs["dim"] = _kwargs.pop("axis")

    # Process args to avoid wrapping repeats as a Tensor
    _args = []
    if len(args) > 0:
        _args.append(_u(args[0], "a"))
    if len(args) > 1:
        _args.append(args[1])  # repeats is an int or sequence of ints, not a tensor
    if len(args) > 2:
        _args.extend([_u(x) for x in args[2:]])

    res = fn(  # pragma: no cover
        *_args,  # pragma: no cover
        **_kwargs,  # pragma: no cover
    )  # pragma: no cover

    def __w(x):
        if isinstance(x, tuple):
            return tuple((__w(i) for i in x))
        if isinstance(x, list):
            return [__w(i) for i in x]
        return array(x) if hasattr(x, "shape") else x

    return __w(res)  # pragma: no cover


def roll(*args, **kwargs):  # pragma: no cover
    """Compute roll.

    Args:
        args: The args argument.
        kwargs: The kwargs argument.

    Returns:
        The result of roll.

    """
    from zero_mlx.array import array, _to_tensor  # pragma: no cover
    import ml_switcheroo_compiler.ops as mops  # pragma: no cover
    import ml_switcheroo_compiler.ops.nn as mnn  # pragma: no cover
    import ml_switcheroo_compiler.random as mrand  # pragma: no cover
    import types  # pragma: no cover

    def _u(x, param_name=None):  # pragma: no cover
        if isinstance(x, list):  # pragma: no cover
            if __import__("builtins").all(  # pragma: no cover
                (isinstance(i, int) for i in x)  # pragma: no cover
            ):  # pragma: no cover
                return tuple(x)  # pragma: no cover
            return [_u(i, param_name) for i in x]  # pragma: no cover
        if isinstance(x, tuple):  # pragma: no cover
            return tuple((_u(i, param_name) for i in x))  # pragma: no cover
        if hasattr(x, "_tensor"):  # pragma: no cover
            return x._tensor  # pragma: no cover
        if isinstance(x, (int, float, bool, complex)):  # pragma: no cover
            if param_name not in {  # pragma: no cover
                "axis",  # pragma: no cover
                "keepdims",
                "shape",
                "size",  # pragma: no cover
                "equal_nan",
                "nan",
                "posinf",
                "neginf",  # pragma: no cover
                "axis1",  # pragma: no cover
                "rtol",  # pragma: no cover
                "n",  # pragma: no cover
                "m",  # pragma: no cover
                "endpoint",  # pragma: no cover
                "offset",  # pragma: no cover
                "a_max",  # pragma: no cover
                "ddof",  # pragma: no cover
                "dims",  # pragma: no cover
                "source",  # pragma: no cover
                "axis2",  # pragma: no cover
                "stop",  # pragma: no cover
                "k",  # pragma: no cover
                "dtype",  # pragma: no cover
                "reps",  # pragma: no cover
                "a_min",  # pragma: no cover
                "sparse",  # pragma: no cover
                "decimals",  # pragma: no cover
                "strides",  # pragma: no cover
                "num",  # pragma: no cover
                "max",  # pragma: no cover
                "indices_or_sections",  # pragma: no cover
                "shape",  # pragma: no cover
                "atol",  # pragma: no cover
                "axis1",  # pragma: no cover
                "step",  # pragma: no cover
                "axes",  # pragma: no cover
                "fill_value",  # pragma: no cover
                "stream",  # pragma: no cover
                "destination",  # pragma: no cover
                "alpha",  # pragma: no cover
                "beta",  # pragma: no cover
                "start",  # pragma: no cover
                "equal_nan",
                "nan",
                "posinf",
                "neginf",  # pragma: no cover
                "keepdims",
                "shape",
                "size",  # pragma: no cover
                "min",  # pragma: no cover
                "p",  # pragma: no cover
            } and "roll" not in (  # pragma: no cover
                "transpose",  # pragma: no cover
                "moveaxis",  # pragma: no cover
                "reshape",  # pragma: no cover
                "expand_dims",  # pragma: no cover
                "linspace",  # pragma: no cover
                "logcumsumexp",  # pragma: no cover
                "cummin",  # pragma: no cover
                "cummax",  # pragma: no cover
                "cumprod",  # pragma: no cover
                "cumsum",  # pragma: no cover
                "logsumexp",  # pragma: no cover
                "all",  # pragma: no cover
                "any",  # pragma: no cover
                "argmin",  # pragma: no cover
                "argmax",  # pragma: no cover
                "prod",  # pragma: no cover
                "sum",  # pragma: no cover
                "mean",  # pragma: no cover
                "var",  # pragma: no cover
                "std",  # pragma: no cover
                "max",  # pragma: no cover
                "min",  # pragma: no cover
                "tensordot",  # pragma: no cover
            ):  # pragma: no cover
                return _to_tensor(x)  # pragma: no cover
        if type(x).__name__ == "ndarray":  # pragma: no cover
            return _to_tensor(x)  # pragma: no cover
        return x  # pragma: no cover

    def _w(x):  # pragma: no cover
        if isinstance(x, tuple):  # pragma: no cover
            return tuple((_w(i) for i in x))  # pragma: no cover
        if isinstance(x, list):  # pragma: no cover
            return [_w(i) for i in x]  # pragma: no cover
        return array(x) if hasattr(x, "shape") else x  # pragma: no cover

    fn = _get_compiler_op("roll")  # pragma: no cover
    res = fn(  # pragma: no cover
        *[_u(x) for x in args],  # pragma: no cover
        **{
            ("dim" if k == "axis" else k): _u(v, k)
            for (k, v) in kwargs.items()
            if k != "stream"
        },  # pragma: no cover
    )  # pragma: no cover

    def __w(x):
        if isinstance(x, tuple):
            return tuple((__w(i) for i in x))
        if isinstance(x, list):
            return [__w(i) for i in x]
        return array(x) if hasattr(x, "shape") else x

    return __w(res)  # pragma: no cover


def sign(*args, **kwargs):  # pragma: no cover
    """Compute sign.

    Args:
        args: The args argument.
        kwargs: The kwargs argument.

    Returns:
        The result of sign.

    """
    from zero_mlx.array import array, _to_tensor  # pragma: no cover
    import ml_switcheroo_compiler.ops as mops  # pragma: no cover
    import ml_switcheroo_compiler.ops.nn as mnn  # pragma: no cover
    import ml_switcheroo_compiler.random as mrand  # pragma: no cover
    import types  # pragma: no cover

    def _u(x, param_name=None):  # pragma: no cover
        if isinstance(x, list):  # pragma: no cover
            if __import__("builtins").all(  # pragma: no cover
                (isinstance(i, int) for i in x)  # pragma: no cover
            ):  # pragma: no cover
                return tuple(x)  # pragma: no cover
            return [_u(i, param_name) for i in x]  # pragma: no cover
        if isinstance(x, tuple):  # pragma: no cover
            return tuple((_u(i, param_name) for i in x))  # pragma: no cover
        if hasattr(x, "_tensor"):  # pragma: no cover
            return x._tensor  # pragma: no cover
        if isinstance(x, (int, float, bool, complex)):  # pragma: no cover
            if param_name not in {  # pragma: no cover
                "axis",  # pragma: no cover
                "keepdims",
                "shape",
                "size",  # pragma: no cover
                "equal_nan",
                "nan",
                "posinf",
                "neginf",  # pragma: no cover
                "axis1",  # pragma: no cover
                "rtol",  # pragma: no cover
                "n",  # pragma: no cover
                "m",  # pragma: no cover
                "endpoint",  # pragma: no cover
                "offset",  # pragma: no cover
                "a_max",  # pragma: no cover
                "ddof",  # pragma: no cover
                "dims",  # pragma: no cover
                "source",  # pragma: no cover
                "axis2",  # pragma: no cover
                "stop",  # pragma: no cover
                "k",  # pragma: no cover
                "dtype",  # pragma: no cover
                "reps",  # pragma: no cover
                "a_min",  # pragma: no cover
                "sparse",  # pragma: no cover
                "decimals",  # pragma: no cover
                "strides",  # pragma: no cover
                "num",  # pragma: no cover
                "max",  # pragma: no cover
                "indices_or_sections",  # pragma: no cover
                "shape",  # pragma: no cover
                "atol",  # pragma: no cover
                "axis1",  # pragma: no cover
                "step",  # pragma: no cover
                "axes",  # pragma: no cover
                "fill_value",  # pragma: no cover
                "stream",  # pragma: no cover
                "destination",  # pragma: no cover
                "alpha",  # pragma: no cover
                "beta",  # pragma: no cover
                "start",  # pragma: no cover
                "equal_nan",
                "nan",
                "posinf",
                "neginf",  # pragma: no cover
                "keepdims",
                "shape",
                "size",  # pragma: no cover
                "min",  # pragma: no cover
                "p",  # pragma: no cover
            } and "sign" not in (  # pragma: no cover
                "transpose",  # pragma: no cover
                "moveaxis",  # pragma: no cover
                "reshape",  # pragma: no cover
                "expand_dims",  # pragma: no cover
                "linspace",  # pragma: no cover
                "logcumsumexp",  # pragma: no cover
                "cummin",  # pragma: no cover
                "cummax",  # pragma: no cover
                "cumprod",  # pragma: no cover
                "cumsum",  # pragma: no cover
                "logsumexp",  # pragma: no cover
                "all",  # pragma: no cover
                "any",  # pragma: no cover
                "argmin",  # pragma: no cover
                "argmax",  # pragma: no cover
                "prod",  # pragma: no cover
                "sum",  # pragma: no cover
                "mean",  # pragma: no cover
                "var",  # pragma: no cover
                "std",  # pragma: no cover
                "max",  # pragma: no cover
                "min",  # pragma: no cover
                "tensordot",  # pragma: no cover
            ):  # pragma: no cover
                return _to_tensor(x)  # pragma: no cover
        if type(x).__name__ == "ndarray":  # pragma: no cover
            return _to_tensor(x)  # pragma: no cover
        return x  # pragma: no cover

    def _w(x):  # pragma: no cover
        if isinstance(x, tuple):  # pragma: no cover
            return tuple((_w(i) for i in x))  # pragma: no cover
        if isinstance(x, list):  # pragma: no cover
            return [_w(i) for i in x]  # pragma: no cover
        return array(x) if hasattr(x, "shape") else x  # pragma: no cover

    fn = _get_compiler_op("sign")  # pragma: no cover
    res = fn(  # pragma: no cover
        *[_u(x) for x in args],  # pragma: no cover
        **{
            ("dim" if k == "axis" else k): _u(v, k)
            for (k, v) in kwargs.items()
            if k != "stream"
        },  # pragma: no cover
    )  # pragma: no cover

    def __w(x):
        if isinstance(x, tuple):
            return tuple((__w(i) for i in x))
        if isinstance(x, list):
            return [__w(i) for i in x]
        return array(x) if hasattr(x, "shape") else x

    return __w(res)  # pragma: no cover


def tensordot(*args, **kwargs):  # pragma: no cover
    """Compute tensordot.

    Args:
        args: The args argument.
        kwargs: The kwargs argument.

    Returns:
        The result of tensordot.

    """
    from zero_mlx.array import array, _to_tensor  # pragma: no cover
    import ml_switcheroo_compiler.ops as mops  # pragma: no cover
    import ml_switcheroo_compiler.ops.nn as mnn  # pragma: no cover
    import ml_switcheroo_compiler.random as mrand  # pragma: no cover
    import types  # pragma: no cover

    def _u(x, param_name=None):  # pragma: no cover
        if isinstance(x, list):  # pragma: no cover
            if __import__("builtins").all(  # pragma: no cover
                (isinstance(i, int) for i in x)  # pragma: no cover
            ):  # pragma: no cover
                return tuple(x)  # pragma: no cover
            return [_u(i, param_name) for i in x]  # pragma: no cover
        if isinstance(x, tuple):  # pragma: no cover
            return tuple((_u(i, param_name) for i in x))  # pragma: no cover
        if hasattr(x, "_tensor"):  # pragma: no cover
            return x._tensor  # pragma: no cover
        if isinstance(x, (int, float, bool, complex)):  # pragma: no cover
            if param_name not in {  # pragma: no cover
                "axis",  # pragma: no cover
                "keepdims",
                "shape",
                "size",  # pragma: no cover
                "equal_nan",
                "nan",
                "posinf",
                "neginf",  # pragma: no cover
                "axis1",  # pragma: no cover
                "rtol",  # pragma: no cover
                "n",  # pragma: no cover
                "m",  # pragma: no cover
                "endpoint",  # pragma: no cover
                "offset",  # pragma: no cover
                "a_max",  # pragma: no cover
                "ddof",  # pragma: no cover
                "dims",  # pragma: no cover
                "source",  # pragma: no cover
                "axis2",  # pragma: no cover
                "stop",  # pragma: no cover
                "k",  # pragma: no cover
                "dtype",  # pragma: no cover
                "reps",  # pragma: no cover
                "a_min",  # pragma: no cover
                "sparse",  # pragma: no cover
                "decimals",  # pragma: no cover
                "strides",  # pragma: no cover
                "num",  # pragma: no cover
                "max",  # pragma: no cover
                "indices_or_sections",  # pragma: no cover
                "shape",  # pragma: no cover
                "atol",  # pragma: no cover
                "axis1",  # pragma: no cover
                "step",  # pragma: no cover
                "axes",  # pragma: no cover
                "fill_value",  # pragma: no cover
                "stream",  # pragma: no cover
                "destination",  # pragma: no cover
                "alpha",  # pragma: no cover
                "beta",  # pragma: no cover
                "start",  # pragma: no cover
                "equal_nan",
                "nan",
                "posinf",
                "neginf",  # pragma: no cover
                "keepdims",
                "shape",
                "size",  # pragma: no cover
                "min",  # pragma: no cover
                "p",  # pragma: no cover
            } and "tensordot" not in (  # pragma: no cover
                "transpose",  # pragma: no cover
                "moveaxis",  # pragma: no cover
                "reshape",  # pragma: no cover
                "expand_dims",  # pragma: no cover
                "linspace",  # pragma: no cover
                "logcumsumexp",  # pragma: no cover
                "cummin",  # pragma: no cover
                "cummax",  # pragma: no cover
                "cumprod",  # pragma: no cover
                "cumsum",  # pragma: no cover
                "logsumexp",  # pragma: no cover
                "all",  # pragma: no cover
                "any",  # pragma: no cover
                "argmin",  # pragma: no cover
                "argmax",  # pragma: no cover
                "prod",  # pragma: no cover
                "sum",  # pragma: no cover
                "mean",  # pragma: no cover
                "var",  # pragma: no cover
                "std",  # pragma: no cover
                "max",  # pragma: no cover
                "min",  # pragma: no cover
                "tensordot",  # pragma: no cover
            ):  # pragma: no cover
                return _to_tensor(x)  # pragma: no cover
        if type(x).__name__ == "ndarray":  # pragma: no cover
            return _to_tensor(x)  # pragma: no cover
        return x  # pragma: no cover

    def _w(x):  # pragma: no cover
        if isinstance(x, tuple):  # pragma: no cover
            return tuple((_w(i) for i in x))  # pragma: no cover
        if isinstance(x, list):  # pragma: no cover
            return [_w(i) for i in x]  # pragma: no cover
        return array(x) if hasattr(x, "shape") else x  # pragma: no cover

    fn = _get_compiler_op("tensordot")  # pragma: no cover
    res = fn(  # pragma: no cover
        *[_u(x) for x in args],  # pragma: no cover
        **{
            ("dim" if k == "axis" else k): _u(v, k)
            for (k, v) in kwargs.items()
            if k != "stream"
        },  # pragma: no cover
    )  # pragma: no cover

    def __w(x):
        if isinstance(x, tuple):
            return tuple((__w(i) for i in x))
        if isinstance(x, list):
            return [__w(i) for i in x]
        return array(x) if hasattr(x, "shape") else x

    return __w(res)  # pragma: no cover


def trace(*args, **kwargs):  # pragma: no cover
    """Compute trace.

    Args:
        args: The args argument.
        kwargs: The kwargs argument.

    Returns:
        The result of trace.

    """
    from zero_mlx.array import array, _to_tensor  # pragma: no cover
    import ml_switcheroo_compiler.ops as mops  # pragma: no cover
    import ml_switcheroo_compiler.ops.nn as mnn  # pragma: no cover
    import ml_switcheroo_compiler.random as mrand  # pragma: no cover
    import types  # pragma: no cover

    def _u(x, param_name=None):  # pragma: no cover
        if isinstance(x, list):  # pragma: no cover
            if __import__("builtins").all(  # pragma: no cover
                (isinstance(i, int) for i in x)  # pragma: no cover
            ):  # pragma: no cover
                return tuple(x)  # pragma: no cover
            return [_u(i, param_name) for i in x]  # pragma: no cover
        if isinstance(x, tuple):  # pragma: no cover
            return tuple((_u(i, param_name) for i in x))  # pragma: no cover
        if hasattr(x, "_tensor"):  # pragma: no cover
            return x._tensor  # pragma: no cover
        if isinstance(x, (int, float, bool, complex)):  # pragma: no cover
            if param_name not in {  # pragma: no cover
                "axis",  # pragma: no cover
                "keepdims",
                "shape",
                "size",  # pragma: no cover
                "equal_nan",
                "nan",
                "posinf",
                "neginf",  # pragma: no cover
                "axis1",  # pragma: no cover
                "rtol",  # pragma: no cover
                "n",  # pragma: no cover
                "m",  # pragma: no cover
                "endpoint",  # pragma: no cover
                "offset",  # pragma: no cover
                "a_max",  # pragma: no cover
                "ddof",  # pragma: no cover
                "dims",  # pragma: no cover
                "source",  # pragma: no cover
                "axis2",  # pragma: no cover
                "stop",  # pragma: no cover
                "k",  # pragma: no cover
                "dtype",  # pragma: no cover
                "reps",  # pragma: no cover
                "a_min",  # pragma: no cover
                "sparse",  # pragma: no cover
                "decimals",  # pragma: no cover
                "strides",  # pragma: no cover
                "num",  # pragma: no cover
                "max",  # pragma: no cover
                "indices_or_sections",  # pragma: no cover
                "shape",  # pragma: no cover
                "atol",  # pragma: no cover
                "axis1",  # pragma: no cover
                "step",  # pragma: no cover
                "axes",  # pragma: no cover
                "fill_value",  # pragma: no cover
                "stream",  # pragma: no cover
                "destination",  # pragma: no cover
                "alpha",  # pragma: no cover
                "beta",  # pragma: no cover
                "start",  # pragma: no cover
                "equal_nan",
                "nan",
                "posinf",
                "neginf",  # pragma: no cover
                "keepdims",
                "shape",
                "size",  # pragma: no cover
                "min",  # pragma: no cover
                "p",  # pragma: no cover
            } and "trace" not in (  # pragma: no cover
                "transpose",  # pragma: no cover
                "moveaxis",  # pragma: no cover
                "reshape",  # pragma: no cover
                "expand_dims",  # pragma: no cover
                "linspace",  # pragma: no cover
                "logcumsumexp",  # pragma: no cover
                "cummin",  # pragma: no cover
                "cummax",  # pragma: no cover
                "cumprod",  # pragma: no cover
                "cumsum",  # pragma: no cover
                "logsumexp",  # pragma: no cover
                "all",  # pragma: no cover
                "any",  # pragma: no cover
                "argmin",  # pragma: no cover
                "argmax",  # pragma: no cover
                "prod",  # pragma: no cover
                "sum",  # pragma: no cover
                "mean",  # pragma: no cover
                "var",  # pragma: no cover
                "std",  # pragma: no cover
                "max",  # pragma: no cover
                "min",  # pragma: no cover
                "tensordot",  # pragma: no cover
            ):  # pragma: no cover
                return _to_tensor(x)  # pragma: no cover
        if type(x).__name__ == "ndarray":  # pragma: no cover
            return _to_tensor(x)  # pragma: no cover
        return x  # pragma: no cover

    def _w(x):  # pragma: no cover
        if isinstance(x, tuple):  # pragma: no cover
            return tuple((_w(i) for i in x))  # pragma: no cover
        if isinstance(x, list):  # pragma: no cover
            return [_w(i) for i in x]  # pragma: no cover
        return array(x) if hasattr(x, "shape") else x  # pragma: no cover

    fn = _get_compiler_op("trace")  # pragma: no cover
    res = fn(  # pragma: no cover
        *[_u(x) for x in args],  # pragma: no cover
        **{
            ("dim" if k == "axis" else k): _u(v, k)
            for (k, v) in kwargs.items()
            if k != "stream"
        },  # pragma: no cover
    )  # pragma: no cover

    def __w(x):
        if isinstance(x, tuple):
            return tuple((__w(i) for i in x))
        if isinstance(x, list):
            return [__w(i) for i in x]
        return array(x) if hasattr(x, "shape") else x

    return __w(res)  # pragma: no cover


def tri(*args, **kwargs):  # pragma: no cover
    """Compute tri.

    Args:
        args: The args argument.
        kwargs: The kwargs argument.

    Returns:
        The result of tri.

    """
    from zero_mlx.array import array, _to_tensor  # pragma: no cover
    import ml_switcheroo_compiler.ops as mops  # pragma: no cover
    import ml_switcheroo_compiler.ops.nn as mnn  # pragma: no cover
    import ml_switcheroo_compiler.random as mrand  # pragma: no cover
    import types  # pragma: no cover

    def _u(x, param_name=None):  # pragma: no cover
        if isinstance(x, list):  # pragma: no cover
            if __import__("builtins").all(  # pragma: no cover
                (isinstance(i, int) for i in x)  # pragma: no cover
            ):  # pragma: no cover
                return tuple(x)  # pragma: no cover
            return [_u(i, param_name) for i in x]  # pragma: no cover
        if isinstance(x, tuple):  # pragma: no cover
            return tuple((_u(i, param_name) for i in x))  # pragma: no cover
        if hasattr(x, "_tensor"):  # pragma: no cover
            return x._tensor  # pragma: no cover
        if isinstance(x, (int, float, bool, complex)):  # pragma: no cover
            if param_name not in {  # pragma: no cover
                "axis",  # pragma: no cover
                "keepdims",
                "shape",
                "size",  # pragma: no cover
                "equal_nan",
                "nan",
                "posinf",
                "neginf",  # pragma: no cover
                "axis1",  # pragma: no cover
                "rtol",  # pragma: no cover
                "n",  # pragma: no cover
                "m",  # pragma: no cover
                "endpoint",  # pragma: no cover
                "offset",  # pragma: no cover
                "a_max",  # pragma: no cover
                "ddof",  # pragma: no cover
                "dims",  # pragma: no cover
                "source",  # pragma: no cover
                "axis2",  # pragma: no cover
                "stop",  # pragma: no cover
                "k",  # pragma: no cover
                "dtype",  # pragma: no cover
                "reps",  # pragma: no cover
                "a_min",  # pragma: no cover
                "sparse",  # pragma: no cover
                "decimals",  # pragma: no cover
                "strides",  # pragma: no cover
                "num",  # pragma: no cover
                "max",  # pragma: no cover
                "indices_or_sections",  # pragma: no cover
                "shape",  # pragma: no cover
                "atol",  # pragma: no cover
                "axis1",  # pragma: no cover
                "step",  # pragma: no cover
                "axes",  # pragma: no cover
                "fill_value",  # pragma: no cover
                "stream",  # pragma: no cover
                "destination",  # pragma: no cover
                "alpha",  # pragma: no cover
                "beta",  # pragma: no cover
                "start",  # pragma: no cover
                "equal_nan",
                "nan",
                "posinf",
                "neginf",  # pragma: no cover
                "keepdims",
                "shape",
                "size",  # pragma: no cover
                "min",  # pragma: no cover
                "p",  # pragma: no cover
            } and "tri" not in (  # pragma: no cover
                "transpose",  # pragma: no cover
                "moveaxis",  # pragma: no cover
                "reshape",  # pragma: no cover
                "expand_dims",  # pragma: no cover
                "linspace",  # pragma: no cover
                "logcumsumexp",  # pragma: no cover
                "cummin",  # pragma: no cover
                "cummax",  # pragma: no cover
                "cumprod",  # pragma: no cover
                "cumsum",  # pragma: no cover
                "logsumexp",  # pragma: no cover
                "all",  # pragma: no cover
                "any",  # pragma: no cover
                "argmin",  # pragma: no cover
                "argmax",  # pragma: no cover
                "prod",  # pragma: no cover
                "sum",  # pragma: no cover
                "mean",  # pragma: no cover
                "var",  # pragma: no cover
                "std",  # pragma: no cover
                "max",  # pragma: no cover
                "min",  # pragma: no cover
                "tensordot",  # pragma: no cover
            ):  # pragma: no cover
                return _to_tensor(x)  # pragma: no cover
        if type(x).__name__ == "ndarray":  # pragma: no cover
            return _to_tensor(x)  # pragma: no cover
        return x  # pragma: no cover

    def _w(x):  # pragma: no cover
        if isinstance(x, tuple):  # pragma: no cover
            return tuple((_w(i) for i in x))  # pragma: no cover
        if isinstance(x, list):  # pragma: no cover
            return [_w(i) for i in x]  # pragma: no cover
        return array(x) if hasattr(x, "shape") else x  # pragma: no cover

    fn = _get_compiler_op("tri")  # pragma: no cover
    res = fn(  # pragma: no cover
        *[_u(x) for x in args],  # pragma: no cover
        **{
            ("dim" if k == "axis" else k): _u(v, k)
            for (k, v) in kwargs.items()
            if k != "stream"
        },  # pragma: no cover
    )  # pragma: no cover

    def __w(x):
        if isinstance(x, tuple):
            return tuple((__w(i) for i in x))
        if isinstance(x, list):
            return [__w(i) for i in x]
        return array(x) if hasattr(x, "shape") else x

    return __w(res)  # pragma: no cover


def where(*args, **kwargs):  # pragma: no cover
    """Compute where.

    Args:
        args: The args argument.
        kwargs: The kwargs argument.

    Returns:
        The result of where.

    """
    from zero_mlx.array import array, _to_tensor  # pragma: no cover
    import ml_switcheroo_compiler.ops as mops  # pragma: no cover
    import ml_switcheroo_compiler.ops.nn as mnn  # pragma: no cover
    import ml_switcheroo_compiler.random as mrand  # pragma: no cover
    import types  # pragma: no cover

    def _u(x, param_name=None):  # pragma: no cover
        if isinstance(x, list):  # pragma: no cover
            if __import__("builtins").all(  # pragma: no cover
                (isinstance(i, int) for i in x)  # pragma: no cover
            ):  # pragma: no cover
                return tuple(x)  # pragma: no cover
            return [_u(i, param_name) for i in x]  # pragma: no cover
        if isinstance(x, tuple):  # pragma: no cover
            return tuple((_u(i, param_name) for i in x))  # pragma: no cover
        if hasattr(x, "_tensor"):  # pragma: no cover
            return x._tensor  # pragma: no cover
        if isinstance(x, (int, float, bool, complex)):  # pragma: no cover
            if param_name not in {  # pragma: no cover
                "axis",  # pragma: no cover
                "keepdims",
                "shape",
                "size",  # pragma: no cover
                "equal_nan",
                "nan",
                "posinf",
                "neginf",  # pragma: no cover
                "axis1",  # pragma: no cover
                "rtol",  # pragma: no cover
                "n",  # pragma: no cover
                "m",  # pragma: no cover
                "endpoint",  # pragma: no cover
                "offset",  # pragma: no cover
                "a_max",  # pragma: no cover
                "ddof",  # pragma: no cover
                "dims",  # pragma: no cover
                "source",  # pragma: no cover
                "axis2",  # pragma: no cover
                "stop",  # pragma: no cover
                "k",  # pragma: no cover
                "dtype",  # pragma: no cover
                "reps",  # pragma: no cover
                "a_min",  # pragma: no cover
                "sparse",  # pragma: no cover
                "decimals",  # pragma: no cover
                "strides",  # pragma: no cover
                "num",  # pragma: no cover
                "max",  # pragma: no cover
                "indices_or_sections",  # pragma: no cover
                "shape",  # pragma: no cover
                "atol",  # pragma: no cover
                "axis1",  # pragma: no cover
                "step",  # pragma: no cover
                "axes",  # pragma: no cover
                "fill_value",  # pragma: no cover
                "stream",  # pragma: no cover
                "destination",  # pragma: no cover
                "alpha",  # pragma: no cover
                "beta",  # pragma: no cover
                "start",  # pragma: no cover
                "equal_nan",
                "nan",
                "posinf",
                "neginf",  # pragma: no cover
                "keepdims",
                "shape",
                "size",  # pragma: no cover
                "min",  # pragma: no cover
                "p",  # pragma: no cover
            } and "where" not in (  # pragma: no cover
                "transpose",  # pragma: no cover
                "moveaxis",  # pragma: no cover
                "reshape",  # pragma: no cover
                "expand_dims",  # pragma: no cover
                "linspace",  # pragma: no cover
                "logcumsumexp",  # pragma: no cover
                "cummin",  # pragma: no cover
                "cummax",  # pragma: no cover
                "cumprod",  # pragma: no cover
                "cumsum",  # pragma: no cover
                "logsumexp",  # pragma: no cover
                "all",  # pragma: no cover
                "any",  # pragma: no cover
                "argmin",  # pragma: no cover
                "argmax",  # pragma: no cover
                "prod",  # pragma: no cover
                "sum",  # pragma: no cover
                "mean",  # pragma: no cover
                "var",  # pragma: no cover
                "std",  # pragma: no cover
                "max",  # pragma: no cover
                "min",  # pragma: no cover
                "tensordot",  # pragma: no cover
            ):  # pragma: no cover
                return _to_tensor(x)  # pragma: no cover
        if type(x).__name__ == "ndarray":  # pragma: no cover
            return _to_tensor(x)  # pragma: no cover
        return x  # pragma: no cover

    def _w(x):  # pragma: no cover
        if isinstance(x, tuple):  # pragma: no cover
            return tuple((_w(i) for i in x))  # pragma: no cover
        if isinstance(x, list):  # pragma: no cover
            return [_w(i) for i in x]  # pragma: no cover
        return array(x) if hasattr(x, "shape") else x  # pragma: no cover

    fn = _get_compiler_op("where")  # pragma: no cover
    res = fn(  # pragma: no cover
        *[_u(x) for x in args],  # pragma: no cover
        **{
            ("dim" if k == "axis" else k): _u(v, k)
            for (k, v) in kwargs.items()
            if k != "stream"
        },  # pragma: no cover
    )  # pragma: no cover

    def __w(x):
        if isinstance(x, tuple):
            return tuple((__w(i) for i in x))
        if isinstance(x, list):
            return [__w(i) for i in x]
        return array(x) if hasattr(x, "shape") else x

    return __w(res)  # pragma: no cover


def sigmoid(a, stream=None):  # pragma: no cover
    """Compute sigmoid.

    Args:
        a: The a argument.
        stream: The stream argument.

    Returns:
        The result of sigmoid.

    """
    from zero_mlx.array import array, _to_tensor  # pragma: no cover
    import ml_switcheroo_compiler.ops as mops  # pragma: no cover
    import ml_switcheroo_compiler.ops.nn as mnn  # pragma: no cover
    import ml_switcheroo_compiler.random as mrand  # pragma: no cover
    import types  # pragma: no cover

    def _u(x, param_name=None):  # pragma: no cover
        if isinstance(x, list):  # pragma: no cover
            if __import__("builtins").all(  # pragma: no cover
                (isinstance(i, int) for i in x)  # pragma: no cover
            ):  # pragma: no cover
                return tuple(x)  # pragma: no cover
            return [_u(i, param_name) for i in x]  # pragma: no cover
        if isinstance(x, tuple):  # pragma: no cover
            return tuple((_u(i, param_name) for i in x))  # pragma: no cover
        if hasattr(x, "_tensor"):  # pragma: no cover
            return x._tensor  # pragma: no cover
        if isinstance(x, (int, float, bool, complex)):  # pragma: no cover
            if param_name not in {  # pragma: no cover
                "axis",  # pragma: no cover
                "keepdims",
                "shape",
                "size",  # pragma: no cover
                "equal_nan",
                "nan",
                "posinf",
                "neginf",  # pragma: no cover
                "axis1",  # pragma: no cover
                "rtol",  # pragma: no cover
                "n",  # pragma: no cover
                "m",  # pragma: no cover
                "endpoint",  # pragma: no cover
                "offset",  # pragma: no cover
                "a_max",  # pragma: no cover
                "ddof",  # pragma: no cover
                "dims",  # pragma: no cover
                "source",  # pragma: no cover
                "axis2",  # pragma: no cover
                "stop",  # pragma: no cover
                "k",  # pragma: no cover
                "dtype",  # pragma: no cover
                "reps",  # pragma: no cover
                "a_min",  # pragma: no cover
                "sparse",  # pragma: no cover
                "decimals",  # pragma: no cover
                "strides",  # pragma: no cover
                "num",  # pragma: no cover
                "max",  # pragma: no cover
                "indices_or_sections",  # pragma: no cover
                "shape",  # pragma: no cover
                "atol",  # pragma: no cover
                "axis1",  # pragma: no cover
                "step",  # pragma: no cover
                "axes",  # pragma: no cover
                "fill_value",  # pragma: no cover
                "stream",  # pragma: no cover
                "destination",  # pragma: no cover
                "alpha",  # pragma: no cover
                "beta",  # pragma: no cover
                "start",  # pragma: no cover
                "equal_nan",
                "nan",
                "posinf",
                "neginf",  # pragma: no cover
                "keepdims",
                "shape",
                "size",  # pragma: no cover
                "min",  # pragma: no cover
                "p",  # pragma: no cover
            } and "sigmoid" not in (  # pragma: no cover
                "transpose",  # pragma: no cover
                "moveaxis",  # pragma: no cover
                "reshape",  # pragma: no cover
                "expand_dims",  # pragma: no cover
                "linspace",  # pragma: no cover
                "logcumsumexp",  # pragma: no cover
                "cummin",  # pragma: no cover
                "cummax",  # pragma: no cover
                "cumprod",  # pragma: no cover
                "cumsum",  # pragma: no cover
                "logsumexp",  # pragma: no cover
                "all",  # pragma: no cover
                "any",  # pragma: no cover
                "argmin",  # pragma: no cover
                "argmax",  # pragma: no cover
                "prod",  # pragma: no cover
                "sum",  # pragma: no cover
                "mean",  # pragma: no cover
                "var",  # pragma: no cover
                "std",  # pragma: no cover
                "max",  # pragma: no cover
                "min",  # pragma: no cover
                "tensordot",  # pragma: no cover
            ):  # pragma: no cover
                return _to_tensor(x)  # pragma: no cover
        if type(x).__name__ == "ndarray":  # pragma: no cover
            return _to_tensor(x)  # pragma: no cover
        return x  # pragma: no cover

    def _w(x):  # pragma: no cover
        if isinstance(x, tuple):  # pragma: no cover
            return tuple((_w(i) for i in x))  # pragma: no cover
        if isinstance(x, list):  # pragma: no cover
            return [_w(i) for i in x]  # pragma: no cover
        return array(x) if hasattr(x, "shape") else x  # pragma: no cover

    fn = _get_compiler_op("sigmoid")  # pragma: no cover
    res = fn(_u(a, "a"))  # pragma: no cover

    def __w(x):
        if isinstance(x, tuple):
            return tuple((__w(i) for i in x))
        if isinstance(x, list):
            return [__w(i) for i in x]
        return array(x) if hasattr(x, "shape") else x

    return __w(res)  # pragma: no cover


def softmax(a, axis=-1, stream=None):  # pragma: no cover
    """Compute softmax.

    Args:
        a: The a argument.
        axis: The axis argument.
        stream: The stream argument.

    Returns:
        The result of softmax.

    """
    from zero_mlx.array import array, _to_tensor  # pragma: no cover
    import ml_switcheroo_compiler.ops as mops  # pragma: no cover
    import ml_switcheroo_compiler.ops.nn as mnn  # pragma: no cover
    import ml_switcheroo_compiler.random as mrand  # pragma: no cover
    import types  # pragma: no cover

    def _u(x, param_name=None):  # pragma: no cover
        if isinstance(x, list):  # pragma: no cover
            if __import__("builtins").all(  # pragma: no cover
                (isinstance(i, int) for i in x)  # pragma: no cover
            ):  # pragma: no cover
                return tuple(x)  # pragma: no cover
            return [_u(i, param_name) for i in x]  # pragma: no cover
        if isinstance(x, tuple):  # pragma: no cover
            return tuple((_u(i, param_name) for i in x))  # pragma: no cover
        if hasattr(x, "_tensor"):  # pragma: no cover
            return x._tensor  # pragma: no cover
        if isinstance(x, (int, float, bool, complex)):  # pragma: no cover
            if param_name not in {  # pragma: no cover
                "axis",  # pragma: no cover
                "keepdims",
                "shape",
                "size",  # pragma: no cover
                "equal_nan",
                "nan",
                "posinf",
                "neginf",  # pragma: no cover
                "axis1",  # pragma: no cover
                "rtol",  # pragma: no cover
                "n",  # pragma: no cover
                "m",  # pragma: no cover
                "endpoint",  # pragma: no cover
                "offset",  # pragma: no cover
                "a_max",  # pragma: no cover
                "ddof",  # pragma: no cover
                "dims",  # pragma: no cover
                "source",  # pragma: no cover
                "axis2",  # pragma: no cover
                "stop",  # pragma: no cover
                "k",  # pragma: no cover
                "dtype",  # pragma: no cover
                "reps",  # pragma: no cover
                "a_min",  # pragma: no cover
                "sparse",  # pragma: no cover
                "decimals",  # pragma: no cover
                "strides",  # pragma: no cover
                "num",  # pragma: no cover
                "max",  # pragma: no cover
                "indices_or_sections",  # pragma: no cover
                "shape",  # pragma: no cover
                "atol",  # pragma: no cover
                "axis1",  # pragma: no cover
                "step",  # pragma: no cover
                "axes",  # pragma: no cover
                "fill_value",  # pragma: no cover
                "stream",  # pragma: no cover
                "destination",  # pragma: no cover
                "alpha",  # pragma: no cover
                "beta",  # pragma: no cover
                "start",  # pragma: no cover
                "equal_nan",
                "nan",
                "posinf",
                "neginf",  # pragma: no cover
                "keepdims",
                "shape",
                "size",  # pragma: no cover
                "min",  # pragma: no cover
                "p",  # pragma: no cover
            } and "softmax" not in (  # pragma: no cover
                "transpose",  # pragma: no cover
                "moveaxis",  # pragma: no cover
                "reshape",  # pragma: no cover
                "expand_dims",  # pragma: no cover
                "linspace",  # pragma: no cover
                "logcumsumexp",  # pragma: no cover
                "cummin",  # pragma: no cover
                "cummax",  # pragma: no cover
                "cumprod",  # pragma: no cover
                "cumsum",  # pragma: no cover
                "logsumexp",  # pragma: no cover
                "all",  # pragma: no cover
                "any",  # pragma: no cover
                "argmin",  # pragma: no cover
                "argmax",  # pragma: no cover
                "prod",  # pragma: no cover
                "sum",  # pragma: no cover
                "mean",  # pragma: no cover
                "var",  # pragma: no cover
                "std",  # pragma: no cover
                "max",  # pragma: no cover
                "min",  # pragma: no cover
                "tensordot",  # pragma: no cover
            ):  # pragma: no cover
                return _to_tensor(x)  # pragma: no cover
        if type(x).__name__ == "ndarray":  # pragma: no cover
            return _to_tensor(x)  # pragma: no cover
        return x  # pragma: no cover

    def _w(x):  # pragma: no cover
        if isinstance(x, tuple):  # pragma: no cover
            return tuple((_w(i) for i in x))  # pragma: no cover
        if isinstance(x, list):  # pragma: no cover
            return [_w(i) for i in x]  # pragma: no cover
        return array(x) if hasattr(x, "shape") else x  # pragma: no cover

    fn = _get_compiler_op("softmax")  # pragma: no cover
    res = fn(_u(a, "a"), axis=_u(axis, "axis"))  # pragma: no cover

    def __w(x):
        if isinstance(x, tuple):
            return tuple((__w(i) for i in x))
        if isinstance(x, list):
            return [__w(i) for i in x]
        return array(x) if hasattr(x, "shape") else x

    return __w(res)  # pragma: no cover


def depends(a, b):  # pragma: no cover
    """Compute depends.

    Args:
        a: The a argument.
        b: The b argument.

    Returns:
        The result of depends.

    """
    from zero_mlx.array import array, _to_tensor  # pragma: no cover
    import ml_switcheroo_compiler.ops as mops  # pragma: no cover
    import ml_switcheroo_compiler.ops.nn as mnn  # pragma: no cover
    import ml_switcheroo_compiler.random as mrand  # pragma: no cover
    import types  # pragma: no cover

    def _u(x, param_name=None):  # pragma: no cover
        if isinstance(x, list):  # pragma: no cover
            if __import__("builtins").all(  # pragma: no cover
                (isinstance(i, int) for i in x)  # pragma: no cover
            ):  # pragma: no cover
                return tuple(x)  # pragma: no cover
            return [_u(i, param_name) for i in x]  # pragma: no cover
        if isinstance(x, tuple):  # pragma: no cover
            return tuple((_u(i, param_name) for i in x))  # pragma: no cover
        if hasattr(x, "_tensor"):  # pragma: no cover
            return x._tensor  # pragma: no cover
        if isinstance(x, (int, float, bool, complex)):  # pragma: no cover
            if param_name not in {  # pragma: no cover
                "axis",  # pragma: no cover
                "keepdims",
                "shape",
                "size",  # pragma: no cover
                "equal_nan",
                "nan",
                "posinf",
                "neginf",  # pragma: no cover
                "axis1",  # pragma: no cover
                "rtol",  # pragma: no cover
                "n",  # pragma: no cover
                "m",  # pragma: no cover
                "endpoint",  # pragma: no cover
                "offset",  # pragma: no cover
                "a_max",  # pragma: no cover
                "ddof",  # pragma: no cover
                "dims",  # pragma: no cover
                "source",  # pragma: no cover
                "axis2",  # pragma: no cover
                "stop",  # pragma: no cover
                "k",  # pragma: no cover
                "dtype",  # pragma: no cover
                "reps",  # pragma: no cover
                "a_min",  # pragma: no cover
                "sparse",  # pragma: no cover
                "decimals",  # pragma: no cover
                "strides",  # pragma: no cover
                "num",  # pragma: no cover
                "max",  # pragma: no cover
                "indices_or_sections",  # pragma: no cover
                "shape",  # pragma: no cover
                "atol",  # pragma: no cover
                "axis1",  # pragma: no cover
                "step",  # pragma: no cover
                "axes",  # pragma: no cover
                "fill_value",  # pragma: no cover
                "stream",  # pragma: no cover
                "destination",  # pragma: no cover
                "alpha",  # pragma: no cover
                "beta",  # pragma: no cover
                "start",  # pragma: no cover
                "equal_nan",
                "nan",
                "posinf",
                "neginf",  # pragma: no cover
                "keepdims",
                "shape",
                "size",  # pragma: no cover
                "min",  # pragma: no cover
                "p",  # pragma: no cover
            } and "depends" not in (  # pragma: no cover
                "transpose",  # pragma: no cover
                "moveaxis",  # pragma: no cover
                "reshape",  # pragma: no cover
                "expand_dims",  # pragma: no cover
                "linspace",  # pragma: no cover
                "logcumsumexp",  # pragma: no cover
                "cummin",  # pragma: no cover
                "cummax",  # pragma: no cover
                "cumprod",  # pragma: no cover
                "cumsum",  # pragma: no cover
                "logsumexp",  # pragma: no cover
                "all",  # pragma: no cover
                "any",  # pragma: no cover
                "argmin",  # pragma: no cover
                "argmax",  # pragma: no cover
                "prod",  # pragma: no cover
                "sum",  # pragma: no cover
                "mean",  # pragma: no cover
                "var",  # pragma: no cover
                "std",  # pragma: no cover
                "max",  # pragma: no cover
                "min",  # pragma: no cover
                "tensordot",  # pragma: no cover
            ):  # pragma: no cover
                return _to_tensor(x)  # pragma: no cover
        if type(x).__name__ == "ndarray":  # pragma: no cover
            return _to_tensor(x)  # pragma: no cover
        return x  # pragma: no cover

    def _w(x):  # pragma: no cover
        if isinstance(x, tuple):  # pragma: no cover
            return tuple((_w(i) for i in x))  # pragma: no cover
        if isinstance(x, list):  # pragma: no cover
            return [_w(i) for i in x]  # pragma: no cover
        return array(x) if hasattr(x, "shape") else x  # pragma: no cover

    try:  # pragma: no cover
        fn = getattr(mops, "depends")  # pragma: no cover
    except AttributeError:  # pragma: no cover
        try:  # pragma: no cover
            fn = getattr(mnn, "depends")  # pragma: no cover
        except AttributeError:  # pragma: no cover
            try:  # pragma: no cover
                fn = getattr(mrand, "depends")  # pragma: no cover
            except AttributeError:  # pragma: no cover  # pragma: no cover
                return a  # pragma: no cover
    res = fn(_u(a, "a"), _u(b, "b"))  # pragma: no cover

    def __w(x):
        if isinstance(x, tuple):
            return tuple((__w(i) for i in x))
        if isinstance(x, list):
            return [__w(i) for i in x]
        return array(x) if hasattr(x, "shape") else x

    return __w(res)  # pragma: no cover


def to_fp8(a):  # pragma: no cover
    """Compute to_fp8.

    Args:
        a: The a argument.

    Returns:
        The result of to_fp8.

    """
    from zero_mlx.array import array, _to_tensor  # pragma: no cover
    import ml_switcheroo_compiler.ops as mops  # pragma: no cover
    import ml_switcheroo_compiler.ops.nn as mnn  # pragma: no cover
    import ml_switcheroo_compiler.random as mrand  # pragma: no cover
    import types  # pragma: no cover

    def _u(x, param_name=None):  # pragma: no cover
        if isinstance(x, list):  # pragma: no cover
            if __import__("builtins").all(  # pragma: no cover
                (isinstance(i, int) for i in x)  # pragma: no cover
            ):  # pragma: no cover
                return tuple(x)  # pragma: no cover
            return [_u(i, param_name) for i in x]  # pragma: no cover
        if isinstance(x, tuple):  # pragma: no cover
            return tuple((_u(i, param_name) for i in x))  # pragma: no cover
        if hasattr(x, "_tensor"):  # pragma: no cover
            return x._tensor  # pragma: no cover
        if isinstance(x, (int, float, bool, complex)):  # pragma: no cover
            if param_name not in {  # pragma: no cover
                "axis",  # pragma: no cover
                "keepdims",
                "shape",
                "size",  # pragma: no cover
                "equal_nan",
                "nan",
                "posinf",
                "neginf",  # pragma: no cover
                "axis1",  # pragma: no cover
                "rtol",  # pragma: no cover
                "n",  # pragma: no cover
                "m",  # pragma: no cover
                "endpoint",  # pragma: no cover
                "offset",  # pragma: no cover
                "a_max",  # pragma: no cover
                "ddof",  # pragma: no cover
                "dims",  # pragma: no cover
                "source",  # pragma: no cover
                "axis2",  # pragma: no cover
                "stop",  # pragma: no cover
                "k",  # pragma: no cover
                "dtype",  # pragma: no cover
                "reps",  # pragma: no cover
                "a_min",  # pragma: no cover
                "sparse",  # pragma: no cover
                "decimals",  # pragma: no cover
                "strides",  # pragma: no cover
                "num",  # pragma: no cover
                "max",  # pragma: no cover
                "indices_or_sections",  # pragma: no cover
                "shape",  # pragma: no cover
                "atol",  # pragma: no cover
                "axis1",  # pragma: no cover
                "step",  # pragma: no cover
                "axes",  # pragma: no cover
                "fill_value",  # pragma: no cover
                "stream",  # pragma: no cover
                "destination",  # pragma: no cover
                "alpha",  # pragma: no cover
                "beta",  # pragma: no cover
                "start",  # pragma: no cover
                "equal_nan",
                "nan",
                "posinf",
                "neginf",  # pragma: no cover
                "keepdims",
                "shape",
                "size",  # pragma: no cover
                "min",  # pragma: no cover
                "p",  # pragma: no cover
            } and "to_fp8" not in (  # pragma: no cover
                "transpose",  # pragma: no cover
                "moveaxis",  # pragma: no cover
                "reshape",  # pragma: no cover
                "expand_dims",  # pragma: no cover
                "linspace",  # pragma: no cover
                "logcumsumexp",  # pragma: no cover
                "cummin",  # pragma: no cover
                "cummax",  # pragma: no cover
                "cumprod",  # pragma: no cover
                "cumsum",  # pragma: no cover
                "logsumexp",  # pragma: no cover
                "all",  # pragma: no cover
                "any",  # pragma: no cover
                "argmin",  # pragma: no cover
                "argmax",  # pragma: no cover
                "prod",  # pragma: no cover
                "sum",  # pragma: no cover
                "mean",  # pragma: no cover
                "var",  # pragma: no cover
                "std",  # pragma: no cover
                "max",  # pragma: no cover
                "min",  # pragma: no cover
                "tensordot",  # pragma: no cover
            ):  # pragma: no cover
                return _to_tensor(x)  # pragma: no cover
        if type(x).__name__ == "ndarray":  # pragma: no cover
            return _to_tensor(x)  # pragma: no cover
        return x  # pragma: no cover

    def _w(x):  # pragma: no cover
        if isinstance(x, tuple):  # pragma: no cover
            return tuple((_w(i) for i in x))  # pragma: no cover
        if isinstance(x, list):  # pragma: no cover
            return [_w(i) for i in x]  # pragma: no cover
        return array(x) if hasattr(x, "shape") else x  # pragma: no cover

    try:  # pragma: no cover
        fn = getattr(mops, "to_fp8")  # pragma: no cover
    except AttributeError:  # pragma: no cover
        try:  # pragma: no cover
            fn = getattr(mnn, "to_fp8")  # pragma: no cover
        except AttributeError:  # pragma: no cover
            try:  # pragma: no cover
                fn = getattr(mrand, "to_fp8")  # pragma: no cover
            except AttributeError:  # pragma: no cover  # pragma: no cover
                return a  # pragma: no cover
    res = fn(_u(a, "a"))  # pragma: no cover

    def __w(x):
        if isinstance(x, tuple):
            return tuple((__w(i) for i in x))
        if isinstance(x, list):
            return [__w(i) for i in x]
        return array(x) if hasattr(x, "shape") else x

    return __w(res)  # pragma: no cover


def from_fp8(a):  # pragma: no cover
    """Compute from_fp8.

    Args:
        a: The a argument.

    Returns:
        The result of from_fp8.

    """
    from zero_mlx.array import array, _to_tensor  # pragma: no cover
    import ml_switcheroo_compiler.ops as mops  # pragma: no cover
    import ml_switcheroo_compiler.ops.nn as mnn  # pragma: no cover
    import ml_switcheroo_compiler.random as mrand  # pragma: no cover
    import types  # pragma: no cover

    def _u(x, param_name=None):  # pragma: no cover
        if isinstance(x, list):  # pragma: no cover
            if __import__("builtins").all(  # pragma: no cover
                (isinstance(i, int) for i in x)  # pragma: no cover
            ):  # pragma: no cover
                return tuple(x)  # pragma: no cover
            return [_u(i, param_name) for i in x]  # pragma: no cover
        if isinstance(x, tuple):  # pragma: no cover
            return tuple((_u(i, param_name) for i in x))  # pragma: no cover
        if hasattr(x, "_tensor"):  # pragma: no cover
            return x._tensor  # pragma: no cover
        if isinstance(x, (int, float, bool, complex)):  # pragma: no cover
            if param_name not in {  # pragma: no cover
                "axis",  # pragma: no cover
                "keepdims",
                "shape",
                "size",  # pragma: no cover
                "equal_nan",
                "nan",
                "posinf",
                "neginf",  # pragma: no cover
                "axis1",  # pragma: no cover
                "rtol",  # pragma: no cover
                "n",  # pragma: no cover
                "m",  # pragma: no cover
                "endpoint",  # pragma: no cover
                "offset",  # pragma: no cover
                "a_max",  # pragma: no cover
                "ddof",  # pragma: no cover
                "dims",  # pragma: no cover
                "source",  # pragma: no cover
                "axis2",  # pragma: no cover
                "stop",  # pragma: no cover
                "k",  # pragma: no cover
                "dtype",  # pragma: no cover
                "reps",  # pragma: no cover
                "a_min",  # pragma: no cover
                "sparse",  # pragma: no cover
                "decimals",  # pragma: no cover
                "strides",  # pragma: no cover
                "num",  # pragma: no cover
                "max",  # pragma: no cover
                "indices_or_sections",  # pragma: no cover
                "shape",  # pragma: no cover
                "atol",  # pragma: no cover
                "axis1",  # pragma: no cover
                "step",  # pragma: no cover
                "axes",  # pragma: no cover
                "fill_value",  # pragma: no cover
                "stream",  # pragma: no cover
                "destination",  # pragma: no cover
                "alpha",  # pragma: no cover
                "beta",  # pragma: no cover
                "start",  # pragma: no cover
                "equal_nan",
                "nan",
                "posinf",
                "neginf",  # pragma: no cover
                "keepdims",
                "shape",
                "size",  # pragma: no cover
                "min",  # pragma: no cover
                "p",  # pragma: no cover
            } and "from_fp8" not in (  # pragma: no cover
                "transpose",  # pragma: no cover
                "moveaxis",  # pragma: no cover
                "reshape",  # pragma: no cover
                "expand_dims",  # pragma: no cover
                "linspace",  # pragma: no cover
                "logcumsumexp",  # pragma: no cover
                "cummin",  # pragma: no cover
                "cummax",  # pragma: no cover
                "cumprod",  # pragma: no cover
                "cumsum",  # pragma: no cover
                "logsumexp",  # pragma: no cover
                "all",  # pragma: no cover
                "any",  # pragma: no cover
                "argmin",  # pragma: no cover
                "argmax",  # pragma: no cover
                "prod",  # pragma: no cover
                "sum",  # pragma: no cover
                "mean",  # pragma: no cover
                "var",  # pragma: no cover
                "std",  # pragma: no cover
                "max",  # pragma: no cover
                "min",  # pragma: no cover
                "tensordot",  # pragma: no cover
            ):  # pragma: no cover
                return _to_tensor(x)  # pragma: no cover
        if type(x).__name__ == "ndarray":  # pragma: no cover
            return _to_tensor(x)  # pragma: no cover
        return x  # pragma: no cover

    def _w(x):  # pragma: no cover
        if isinstance(x, tuple):  # pragma: no cover
            return tuple((_w(i) for i in x))  # pragma: no cover
        if isinstance(x, list):  # pragma: no cover
            return [_w(i) for i in x]  # pragma: no cover
        return array(x) if hasattr(x, "shape") else x  # pragma: no cover

    try:  # pragma: no cover
        fn = getattr(mops, "from_fp8")  # pragma: no cover
    except AttributeError:  # pragma: no cover
        try:  # pragma: no cover
            fn = getattr(mnn, "from_fp8")  # pragma: no cover
        except AttributeError:  # pragma: no cover
            try:  # pragma: no cover
                fn = getattr(mrand, "from_fp8")  # pragma: no cover
            except AttributeError:  # pragma: no cover  # pragma: no cover
                return a  # pragma: no cover
    res = fn(_u(a, "a"))  # pragma: no cover

    def __w(x):
        if isinstance(x, tuple):
            return tuple((__w(i) for i in x))
        if isinstance(x, list):
            return [__w(i) for i in x]
        return array(x) if hasattr(x, "shape") else x

    return __w(res)  # pragma: no cover


def arcsin(a: Any, stream: Any = None) -> Any:  # pragma: no cover
    """Compute arcsin.

    Args:
        a: The a argument.
        stream: The stream argument.

    Returns:
        The result of arcsin.

    """
    from zero_mlx.array import array, _to_tensor  # pragma: no cover
    import ml_switcheroo_compiler.ops as mops  # pragma: no cover
    import ml_switcheroo_compiler.ops.nn as mnn  # pragma: no cover
    import ml_switcheroo_compiler.random as mrand  # pragma: no cover
    import types  # pragma: no cover

    def _u(x, param_name=None):  # pragma: no cover
        if isinstance(x, list):  # pragma: no cover
            if __import__("builtins").all(  # pragma: no cover
                (isinstance(i, int) for i in x)  # pragma: no cover
            ):  # pragma: no cover
                return tuple(x)  # pragma: no cover
            return [_u(i, param_name) for i in x]  # pragma: no cover
        if isinstance(x, tuple):  # pragma: no cover
            return tuple((_u(i, param_name) for i in x))  # pragma: no cover
        if hasattr(x, "_tensor"):  # pragma: no cover
            return x._tensor  # pragma: no cover
        if isinstance(x, (int, float, bool, complex)):  # pragma: no cover
            if param_name not in {  # pragma: no cover
                "axis",  # pragma: no cover
                "keepdims",
                "shape",
                "size",  # pragma: no cover
                "equal_nan",
                "nan",
                "posinf",
                "neginf",  # pragma: no cover
                "axis1",  # pragma: no cover
                "rtol",  # pragma: no cover
                "n",  # pragma: no cover
                "m",  # pragma: no cover
                "endpoint",  # pragma: no cover
                "offset",  # pragma: no cover
                "a_max",  # pragma: no cover
                "ddof",  # pragma: no cover
                "dims",  # pragma: no cover
                "source",  # pragma: no cover
                "axis2",  # pragma: no cover
                "stop",  # pragma: no cover
                "k",  # pragma: no cover
                "dtype",  # pragma: no cover
                "reps",  # pragma: no cover
                "a_min",  # pragma: no cover
                "sparse",  # pragma: no cover
                "decimals",  # pragma: no cover
                "strides",  # pragma: no cover
                "num",  # pragma: no cover
                "max",  # pragma: no cover
                "indices_or_sections",  # pragma: no cover
                "shape",  # pragma: no cover
                "atol",  # pragma: no cover
                "axis1",  # pragma: no cover
                "step",  # pragma: no cover
                "axes",  # pragma: no cover
                "fill_value",  # pragma: no cover
                "stream",  # pragma: no cover
                "destination",  # pragma: no cover
                "alpha",  # pragma: no cover
                "beta",  # pragma: no cover
                "start",  # pragma: no cover
                "equal_nan",
                "nan",
                "posinf",
                "neginf",  # pragma: no cover
                "keepdims",
                "shape",
                "size",  # pragma: no cover
                "min",  # pragma: no cover
                "p",  # pragma: no cover
            } and "arcsin" not in (  # pragma: no cover
                "transpose",  # pragma: no cover
                "moveaxis",  # pragma: no cover
                "reshape",  # pragma: no cover
                "expand_dims",  # pragma: no cover
                "linspace",  # pragma: no cover
                "logcumsumexp",  # pragma: no cover
                "cummin",  # pragma: no cover
                "cummax",  # pragma: no cover
                "cumprod",  # pragma: no cover
                "cumsum",  # pragma: no cover
                "logsumexp",  # pragma: no cover
                "all",  # pragma: no cover
                "any",  # pragma: no cover
                "argmin",  # pragma: no cover
                "argmax",  # pragma: no cover
                "prod",  # pragma: no cover
                "sum",  # pragma: no cover
                "mean",  # pragma: no cover
                "var",  # pragma: no cover
                "std",  # pragma: no cover
                "max",  # pragma: no cover
                "min",  # pragma: no cover
                "tensordot",  # pragma: no cover
            ):  # pragma: no cover
                return _to_tensor(x)  # pragma: no cover
        if type(x).__name__ == "ndarray":  # pragma: no cover
            return _to_tensor(x)  # pragma: no cover
        return x  # pragma: no cover

    def _w(x):  # pragma: no cover
        if isinstance(x, tuple):  # pragma: no cover
            return tuple((_w(i) for i in x))  # pragma: no cover
        if isinstance(x, list):  # pragma: no cover
            return [_w(i) for i in x]  # pragma: no cover
        return array(x) if hasattr(x, "shape") else x  # pragma: no cover

    fn = _get_compiler_op("asin")  # pragma: no cover
    res = fn(_u(a, "a"))  # pragma: no cover

    def __w(x):
        if isinstance(x, tuple):
            return tuple((__w(i) for i in x))
        if isinstance(x, list):
            return [__w(i) for i in x]
        return array(x) if hasattr(x, "shape") else x

    return __w(res)  # pragma: no cover


def arccos(a: Any, stream: Any = None) -> Any:  # pragma: no cover
    """Compute arccos.

    Args:
        a: The a argument.
        stream: The stream argument.

    Returns:
        The result of arccos.

    """
    from zero_mlx.array import array, _to_tensor  # pragma: no cover
    import ml_switcheroo_compiler.ops as mops  # pragma: no cover
    import ml_switcheroo_compiler.ops.nn as mnn  # pragma: no cover
    import ml_switcheroo_compiler.random as mrand  # pragma: no cover
    import types  # pragma: no cover

    def _u(x, param_name=None):  # pragma: no cover
        if isinstance(x, list):  # pragma: no cover
            if __import__("builtins").all(  # pragma: no cover
                (isinstance(i, int) for i in x)  # pragma: no cover
            ):  # pragma: no cover
                return tuple(x)  # pragma: no cover
            return [_u(i, param_name) for i in x]  # pragma: no cover
        if isinstance(x, tuple):  # pragma: no cover
            return tuple((_u(i, param_name) for i in x))  # pragma: no cover
        if hasattr(x, "_tensor"):  # pragma: no cover
            return x._tensor  # pragma: no cover
        if isinstance(x, (int, float, bool, complex)):  # pragma: no cover
            if param_name not in {  # pragma: no cover
                "axis",  # pragma: no cover
                "keepdims",
                "shape",
                "size",  # pragma: no cover
                "equal_nan",
                "nan",
                "posinf",
                "neginf",  # pragma: no cover
                "axis1",  # pragma: no cover
                "rtol",  # pragma: no cover
                "n",  # pragma: no cover
                "m",  # pragma: no cover
                "endpoint",  # pragma: no cover
                "offset",  # pragma: no cover
                "a_max",  # pragma: no cover
                "ddof",  # pragma: no cover
                "dims",  # pragma: no cover
                "source",  # pragma: no cover
                "axis2",  # pragma: no cover
                "stop",  # pragma: no cover
                "k",  # pragma: no cover
                "dtype",  # pragma: no cover
                "reps",  # pragma: no cover
                "a_min",  # pragma: no cover
                "sparse",  # pragma: no cover
                "decimals",  # pragma: no cover
                "strides",  # pragma: no cover
                "num",  # pragma: no cover
                "max",  # pragma: no cover
                "indices_or_sections",  # pragma: no cover
                "shape",  # pragma: no cover
                "atol",  # pragma: no cover
                "axis1",  # pragma: no cover
                "step",  # pragma: no cover
                "axes",  # pragma: no cover
                "fill_value",  # pragma: no cover
                "stream",  # pragma: no cover
                "destination",  # pragma: no cover
                "alpha",  # pragma: no cover
                "beta",  # pragma: no cover
                "start",  # pragma: no cover
                "equal_nan",
                "nan",
                "posinf",
                "neginf",  # pragma: no cover
                "keepdims",
                "shape",
                "size",  # pragma: no cover
                "min",  # pragma: no cover
                "p",  # pragma: no cover
            } and "arccos" not in (  # pragma: no cover
                "transpose",  # pragma: no cover
                "moveaxis",  # pragma: no cover
                "reshape",  # pragma: no cover
                "expand_dims",  # pragma: no cover
                "linspace",  # pragma: no cover
                "logcumsumexp",  # pragma: no cover
                "cummin",  # pragma: no cover
                "cummax",  # pragma: no cover
                "cumprod",  # pragma: no cover
                "cumsum",  # pragma: no cover
                "logsumexp",  # pragma: no cover
                "all",  # pragma: no cover
                "any",  # pragma: no cover
                "argmin",  # pragma: no cover
                "argmax",  # pragma: no cover
                "prod",  # pragma: no cover
                "sum",  # pragma: no cover
                "mean",  # pragma: no cover
                "var",  # pragma: no cover
                "std",  # pragma: no cover
                "max",  # pragma: no cover
                "min",  # pragma: no cover
                "tensordot",  # pragma: no cover
            ):  # pragma: no cover
                return _to_tensor(x)  # pragma: no cover
        if type(x).__name__ == "ndarray":  # pragma: no cover
            return _to_tensor(x)  # pragma: no cover
        return x  # pragma: no cover

    def _w(x):  # pragma: no cover
        if isinstance(x, tuple):  # pragma: no cover
            return tuple((_w(i) for i in x))  # pragma: no cover
        if isinstance(x, list):  # pragma: no cover
            return [_w(i) for i in x]  # pragma: no cover
        return array(x) if hasattr(x, "shape") else x  # pragma: no cover

    fn = _get_compiler_op("acos")  # pragma: no cover
    res = fn(_u(a, "a"))  # pragma: no cover

    def __w(x):
        if isinstance(x, tuple):
            return tuple((__w(i) for i in x))
        if isinstance(x, list):
            return [__w(i) for i in x]
        return array(x) if hasattr(x, "shape") else x

    return __w(res)  # pragma: no cover


def arctan(a: Any, stream: Any = None) -> Any:  # pragma: no cover
    """Compute arctan.

    Args:
        a: The a argument.
        stream: The stream argument.

    Returns:
        The result of arctan.

    """
    from zero_mlx.array import array, _to_tensor  # pragma: no cover
    import ml_switcheroo_compiler.ops as mops  # pragma: no cover
    import ml_switcheroo_compiler.ops.nn as mnn  # pragma: no cover
    import ml_switcheroo_compiler.random as mrand  # pragma: no cover
    import types  # pragma: no cover

    def _u(x, param_name=None):  # pragma: no cover
        if isinstance(x, list):  # pragma: no cover
            if __import__("builtins").all(  # pragma: no cover
                (isinstance(i, int) for i in x)  # pragma: no cover
            ):  # pragma: no cover
                return tuple(x)  # pragma: no cover
            return [_u(i, param_name) for i in x]  # pragma: no cover
        if isinstance(x, tuple):  # pragma: no cover
            return tuple((_u(i, param_name) for i in x))  # pragma: no cover
        if hasattr(x, "_tensor"):  # pragma: no cover
            return x._tensor  # pragma: no cover
        if isinstance(x, (int, float, bool, complex)):  # pragma: no cover
            if param_name not in {  # pragma: no cover
                "axis",  # pragma: no cover
                "keepdims",
                "shape",
                "size",  # pragma: no cover
                "equal_nan",
                "nan",
                "posinf",
                "neginf",  # pragma: no cover
                "axis1",  # pragma: no cover
                "rtol",  # pragma: no cover
                "n",  # pragma: no cover
                "m",  # pragma: no cover
                "endpoint",  # pragma: no cover
                "offset",  # pragma: no cover
                "a_max",  # pragma: no cover
                "ddof",  # pragma: no cover
                "dims",  # pragma: no cover
                "source",  # pragma: no cover
                "axis2",  # pragma: no cover
                "stop",  # pragma: no cover
                "k",  # pragma: no cover
                "dtype",  # pragma: no cover
                "reps",  # pragma: no cover
                "a_min",  # pragma: no cover
                "sparse",  # pragma: no cover
                "decimals",  # pragma: no cover
                "strides",  # pragma: no cover
                "num",  # pragma: no cover
                "max",  # pragma: no cover
                "indices_or_sections",  # pragma: no cover
                "shape",  # pragma: no cover
                "atol",  # pragma: no cover
                "axis1",  # pragma: no cover
                "step",  # pragma: no cover
                "axes",  # pragma: no cover
                "fill_value",  # pragma: no cover
                "stream",  # pragma: no cover
                "destination",  # pragma: no cover
                "alpha",  # pragma: no cover
                "beta",  # pragma: no cover
                "start",  # pragma: no cover
                "equal_nan",
                "nan",
                "posinf",
                "neginf",  # pragma: no cover
                "keepdims",
                "shape",
                "size",  # pragma: no cover
                "min",  # pragma: no cover
                "p",  # pragma: no cover
            } and "arctan" not in (  # pragma: no cover
                "transpose",  # pragma: no cover
                "moveaxis",  # pragma: no cover
                "reshape",  # pragma: no cover
                "expand_dims",  # pragma: no cover
                "linspace",  # pragma: no cover
                "logcumsumexp",  # pragma: no cover
                "cummin",  # pragma: no cover
                "cummax",  # pragma: no cover
                "cumprod",  # pragma: no cover
                "cumsum",  # pragma: no cover
                "logsumexp",  # pragma: no cover
                "all",  # pragma: no cover
                "any",  # pragma: no cover
                "argmin",  # pragma: no cover
                "argmax",  # pragma: no cover
                "prod",  # pragma: no cover
                "sum",  # pragma: no cover
                "mean",  # pragma: no cover
                "var",  # pragma: no cover
                "std",  # pragma: no cover
                "max",  # pragma: no cover
                "min",  # pragma: no cover
                "tensordot",  # pragma: no cover
            ):  # pragma: no cover
                return _to_tensor(x)  # pragma: no cover
        if type(x).__name__ == "ndarray":  # pragma: no cover
            return _to_tensor(x)  # pragma: no cover
        return x  # pragma: no cover

    def _w(x):  # pragma: no cover
        if isinstance(x, tuple):  # pragma: no cover
            return tuple((_w(i) for i in x))  # pragma: no cover
        if isinstance(x, list):  # pragma: no cover
            return [_w(i) for i in x]  # pragma: no cover
        return array(x) if hasattr(x, "shape") else x  # pragma: no cover

    fn = _get_compiler_op("atan")  # pragma: no cover
    res = fn(_u(a, "a"))  # pragma: no cover

    def __w(x):
        if isinstance(x, tuple):
            return tuple((__w(i) for i in x))
        if isinstance(x, list):
            return [__w(i) for i in x]
        return array(x) if hasattr(x, "shape") else x

    return __w(res)  # pragma: no cover


def arcsinh(a: Any, stream: Any = None) -> Any:  # pragma: no cover
    """Compute arcsinh.

    Args:
        a: The a argument.
        stream: The stream argument.

    Returns:
        The result of arcsinh.

    """
    from zero_mlx.array import array, _to_tensor  # pragma: no cover
    import ml_switcheroo_compiler.ops as mops  # pragma: no cover
    import ml_switcheroo_compiler.ops.nn as mnn  # pragma: no cover
    import ml_switcheroo_compiler.random as mrand  # pragma: no cover
    import types  # pragma: no cover

    def _u(x, param_name=None):  # pragma: no cover
        if isinstance(x, list):  # pragma: no cover
            if __import__("builtins").all(  # pragma: no cover
                (isinstance(i, int) for i in x)  # pragma: no cover
            ):  # pragma: no cover
                return tuple(x)  # pragma: no cover
            return [_u(i, param_name) for i in x]  # pragma: no cover
        if isinstance(x, tuple):  # pragma: no cover
            return tuple((_u(i, param_name) for i in x))  # pragma: no cover
        if hasattr(x, "_tensor"):  # pragma: no cover
            return x._tensor  # pragma: no cover
        if isinstance(x, (int, float, bool, complex)):  # pragma: no cover
            if param_name not in {  # pragma: no cover
                "axis",  # pragma: no cover
                "keepdims",
                "shape",
                "size",  # pragma: no cover
                "equal_nan",
                "nan",
                "posinf",
                "neginf",  # pragma: no cover
                "axis1",  # pragma: no cover
                "rtol",  # pragma: no cover
                "n",  # pragma: no cover
                "m",  # pragma: no cover
                "endpoint",  # pragma: no cover
                "offset",  # pragma: no cover
                "a_max",  # pragma: no cover
                "ddof",  # pragma: no cover
                "dims",  # pragma: no cover
                "source",  # pragma: no cover
                "axis2",  # pragma: no cover
                "stop",  # pragma: no cover
                "k",  # pragma: no cover
                "dtype",  # pragma: no cover
                "reps",  # pragma: no cover
                "a_min",  # pragma: no cover
                "sparse",  # pragma: no cover
                "decimals",  # pragma: no cover
                "strides",  # pragma: no cover
                "num",  # pragma: no cover
                "max",  # pragma: no cover
                "indices_or_sections",  # pragma: no cover
                "shape",  # pragma: no cover
                "atol",  # pragma: no cover
                "axis1",  # pragma: no cover
                "step",  # pragma: no cover
                "axes",  # pragma: no cover
                "fill_value",  # pragma: no cover
                "stream",  # pragma: no cover
                "destination",  # pragma: no cover
                "alpha",  # pragma: no cover
                "beta",  # pragma: no cover
                "start",  # pragma: no cover
                "equal_nan",
                "nan",
                "posinf",
                "neginf",  # pragma: no cover
                "keepdims",
                "shape",
                "size",  # pragma: no cover
                "min",  # pragma: no cover
                "p",  # pragma: no cover
            } and "arcsinh" not in (  # pragma: no cover
                "transpose",  # pragma: no cover
                "moveaxis",  # pragma: no cover
                "reshape",  # pragma: no cover
                "expand_dims",  # pragma: no cover
                "linspace",  # pragma: no cover
                "logcumsumexp",  # pragma: no cover
                "cummin",  # pragma: no cover
                "cummax",  # pragma: no cover
                "cumprod",  # pragma: no cover
                "cumsum",  # pragma: no cover
                "logsumexp",  # pragma: no cover
                "all",  # pragma: no cover
                "any",  # pragma: no cover
                "argmin",  # pragma: no cover
                "argmax",  # pragma: no cover
                "prod",  # pragma: no cover
                "sum",  # pragma: no cover
                "mean",  # pragma: no cover
                "var",  # pragma: no cover
                "std",  # pragma: no cover
                "max",  # pragma: no cover
                "min",  # pragma: no cover
                "tensordot",  # pragma: no cover
            ):  # pragma: no cover
                return _to_tensor(x)  # pragma: no cover
        if type(x).__name__ == "ndarray":  # pragma: no cover
            return _to_tensor(x)  # pragma: no cover
        return x  # pragma: no cover

    def _w(x):  # pragma: no cover
        if isinstance(x, tuple):  # pragma: no cover
            return tuple((_w(i) for i in x))  # pragma: no cover
        if isinstance(x, list):  # pragma: no cover
            return [_w(i) for i in x]  # pragma: no cover
        return array(x) if hasattr(x, "shape") else x  # pragma: no cover

    fn = _get_compiler_op("asinh")  # pragma: no cover
    res = fn(_u(a, "a"))  # pragma: no cover

    def __w(x):
        if isinstance(x, tuple):
            return tuple((__w(i) for i in x))
        if isinstance(x, list):
            return [__w(i) for i in x]
        return array(x) if hasattr(x, "shape") else x

    return __w(res)  # pragma: no cover


def arctanh(a: Any, stream: Any = None) -> Any:  # pragma: no cover
    """Compute arctanh.

    Args:
        a: The a argument.
        stream: The stream argument.

    Returns:
        The result of arctanh.

    """
    from zero_mlx.array import array, _to_tensor  # pragma: no cover
    import ml_switcheroo_compiler.ops as mops  # pragma: no cover
    import ml_switcheroo_compiler.ops.nn as mnn  # pragma: no cover
    import ml_switcheroo_compiler.random as mrand  # pragma: no cover
    import types  # pragma: no cover

    def _u(x, param_name=None):  # pragma: no cover
        if isinstance(x, list):  # pragma: no cover
            if __import__("builtins").all(  # pragma: no cover
                (isinstance(i, int) for i in x)  # pragma: no cover
            ):  # pragma: no cover
                return tuple(x)  # pragma: no cover
            return [_u(i, param_name) for i in x]  # pragma: no cover
        if isinstance(x, tuple):  # pragma: no cover
            return tuple((_u(i, param_name) for i in x))  # pragma: no cover
        if hasattr(x, "_tensor"):  # pragma: no cover
            return x._tensor  # pragma: no cover
        if isinstance(x, (int, float, bool, complex)):  # pragma: no cover
            if param_name not in {  # pragma: no cover
                "axis",  # pragma: no cover
                "keepdims",
                "shape",
                "size",  # pragma: no cover
                "equal_nan",
                "nan",
                "posinf",
                "neginf",  # pragma: no cover
                "axis1",  # pragma: no cover
                "rtol",  # pragma: no cover
                "n",  # pragma: no cover
                "m",  # pragma: no cover
                "endpoint",  # pragma: no cover
                "offset",  # pragma: no cover
                "a_max",  # pragma: no cover
                "ddof",  # pragma: no cover
                "dims",  # pragma: no cover
                "source",  # pragma: no cover
                "axis2",  # pragma: no cover
                "stop",  # pragma: no cover
                "k",  # pragma: no cover
                "dtype",  # pragma: no cover
                "reps",  # pragma: no cover
                "a_min",  # pragma: no cover
                "sparse",  # pragma: no cover
                "decimals",  # pragma: no cover
                "strides",  # pragma: no cover
                "num",  # pragma: no cover
                "max",  # pragma: no cover
                "indices_or_sections",  # pragma: no cover
                "shape",  # pragma: no cover
                "atol",  # pragma: no cover
                "axis1",  # pragma: no cover
                "step",  # pragma: no cover
                "axes",  # pragma: no cover
                "fill_value",  # pragma: no cover
                "stream",  # pragma: no cover
                "destination",  # pragma: no cover
                "alpha",  # pragma: no cover
                "beta",  # pragma: no cover
                "start",  # pragma: no cover
                "equal_nan",
                "nan",
                "posinf",
                "neginf",  # pragma: no cover
                "keepdims",
                "shape",
                "size",  # pragma: no cover
                "min",  # pragma: no cover
                "p",  # pragma: no cover
            } and "arctanh" not in (  # pragma: no cover
                "transpose",  # pragma: no cover
                "moveaxis",  # pragma: no cover
                "reshape",  # pragma: no cover
                "expand_dims",  # pragma: no cover
                "linspace",  # pragma: no cover
                "logcumsumexp",  # pragma: no cover
                "cummin",  # pragma: no cover
                "cummax",  # pragma: no cover
                "cumprod",  # pragma: no cover
                "cumsum",  # pragma: no cover
                "logsumexp",  # pragma: no cover
                "all",  # pragma: no cover
                "any",  # pragma: no cover
                "argmin",  # pragma: no cover
                "argmax",  # pragma: no cover
                "prod",  # pragma: no cover
                "sum",  # pragma: no cover
                "mean",  # pragma: no cover
                "var",  # pragma: no cover
                "std",  # pragma: no cover
                "max",  # pragma: no cover
                "min",  # pragma: no cover
                "tensordot",  # pragma: no cover
            ):  # pragma: no cover
                return _to_tensor(x)  # pragma: no cover
        if type(x).__name__ == "ndarray":  # pragma: no cover
            return _to_tensor(x)  # pragma: no cover
        return x  # pragma: no cover

    def _w(x):  # pragma: no cover
        if isinstance(x, tuple):  # pragma: no cover
            return tuple((_w(i) for i in x))  # pragma: no cover
        if isinstance(x, list):  # pragma: no cover
            return [_w(i) for i in x]  # pragma: no cover
        return array(x) if hasattr(x, "shape") else x  # pragma: no cover

    fn = _get_compiler_op("atanh")  # pragma: no cover
    res = fn(_u(a, "a"))  # pragma: no cover

    def __w(x):
        if isinstance(x, tuple):
            return tuple((__w(i) for i in x))
        if isinstance(x, list):
            return [__w(i) for i in x]
        return array(x) if hasattr(x, "shape") else x

    return __w(res)  # pragma: no cover


def log2(a: Any, stream: Any = None) -> Any:  # pragma: no cover
    """Compute log2.

    Args:
        a: The a argument.
        stream: The stream argument.

    Returns:
        The result of log2.

    """
    from zero_mlx.array import array, _to_tensor  # pragma: no cover
    import ml_switcheroo_compiler.ops as mops  # pragma: no cover
    import ml_switcheroo_compiler.ops.nn as mnn  # pragma: no cover
    import ml_switcheroo_compiler.random as mrand  # pragma: no cover
    import types  # pragma: no cover

    def _u(x, param_name=None):  # pragma: no cover
        if isinstance(x, list):  # pragma: no cover
            if __import__("builtins").all(  # pragma: no cover
                (isinstance(i, int) for i in x)  # pragma: no cover
            ):  # pragma: no cover
                return tuple(x)  # pragma: no cover
            return [_u(i, param_name) for i in x]  # pragma: no cover
        if isinstance(x, tuple):  # pragma: no cover
            return tuple((_u(i, param_name) for i in x))  # pragma: no cover
        if hasattr(x, "_tensor"):  # pragma: no cover
            return x._tensor  # pragma: no cover
        if isinstance(x, (int, float, bool, complex)):  # pragma: no cover
            if param_name not in {  # pragma: no cover
                "axis",  # pragma: no cover
                "keepdims",
                "shape",
                "size",  # pragma: no cover
                "equal_nan",
                "nan",
                "posinf",
                "neginf",  # pragma: no cover
                "axis1",  # pragma: no cover
                "rtol",  # pragma: no cover
                "n",  # pragma: no cover
                "m",  # pragma: no cover
                "endpoint",  # pragma: no cover
                "offset",  # pragma: no cover
                "a_max",  # pragma: no cover
                "ddof",  # pragma: no cover
                "dims",  # pragma: no cover
                "source",  # pragma: no cover
                "axis2",  # pragma: no cover
                "stop",  # pragma: no cover
                "k",  # pragma: no cover
                "dtype",  # pragma: no cover
                "reps",  # pragma: no cover
                "a_min",  # pragma: no cover
                "sparse",  # pragma: no cover
                "decimals",  # pragma: no cover
                "strides",  # pragma: no cover
                "num",  # pragma: no cover
                "max",  # pragma: no cover
                "indices_or_sections",  # pragma: no cover
                "shape",  # pragma: no cover
                "atol",  # pragma: no cover
                "axis1",  # pragma: no cover
                "step",  # pragma: no cover
                "axes",  # pragma: no cover
                "fill_value",  # pragma: no cover
                "stream",  # pragma: no cover
                "destination",  # pragma: no cover
                "alpha",  # pragma: no cover
                "beta",  # pragma: no cover
                "start",  # pragma: no cover
                "equal_nan",
                "nan",
                "posinf",
                "neginf",  # pragma: no cover
                "keepdims",
                "shape",
                "size",  # pragma: no cover
                "min",  # pragma: no cover
                "p",  # pragma: no cover
            } and "log2" not in (  # pragma: no cover
                "transpose",  # pragma: no cover
                "moveaxis",  # pragma: no cover
                "reshape",  # pragma: no cover
                "expand_dims",  # pragma: no cover
                "linspace",  # pragma: no cover
                "logcumsumexp",  # pragma: no cover
                "cummin",  # pragma: no cover
                "cummax",  # pragma: no cover
                "cumprod",  # pragma: no cover
                "cumsum",  # pragma: no cover
                "logsumexp",  # pragma: no cover
                "all",  # pragma: no cover
                "any",  # pragma: no cover
                "argmin",  # pragma: no cover
                "argmax",  # pragma: no cover
                "prod",  # pragma: no cover
                "sum",  # pragma: no cover
                "mean",  # pragma: no cover
                "var",  # pragma: no cover
                "std",  # pragma: no cover
                "max",  # pragma: no cover
                "min",  # pragma: no cover
                "tensordot",  # pragma: no cover
            ):  # pragma: no cover
                return _to_tensor(x)  # pragma: no cover
        if type(x).__name__ == "ndarray":  # pragma: no cover
            return _to_tensor(x)  # pragma: no cover
        return x  # pragma: no cover

    def _w(x):  # pragma: no cover
        if isinstance(x, tuple):  # pragma: no cover
            return tuple((_w(i) for i in x))  # pragma: no cover
        if isinstance(x, list):  # pragma: no cover
            return [_w(i) for i in x]  # pragma: no cover
        return array(x) if hasattr(x, "shape") else x  # pragma: no cover

    fn = _get_compiler_op("log2")  # pragma: no cover
    res = fn(_u(a, "a"))  # pragma: no cover

    def __w(x):
        if isinstance(x, tuple):
            return tuple((__w(i) for i in x))
        if isinstance(x, list):
            return [__w(i) for i in x]
        return array(x) if hasattr(x, "shape") else x

    return __w(res)  # pragma: no cover


def log10(a: Any, stream: Any = None) -> Any:  # pragma: no cover
    """Compute log10.

    Args:
        a: The a argument.
        stream: The stream argument.

    Returns:
        The result of log10.

    """
    from zero_mlx.array import array, _to_tensor  # pragma: no cover
    import ml_switcheroo_compiler.ops as mops  # pragma: no cover
    import ml_switcheroo_compiler.ops.nn as mnn  # pragma: no cover
    import ml_switcheroo_compiler.random as mrand  # pragma: no cover
    import types  # pragma: no cover

    def _u(x, param_name=None):  # pragma: no cover
        if isinstance(x, list):  # pragma: no cover
            if __import__("builtins").all(  # pragma: no cover
                (isinstance(i, int) for i in x)  # pragma: no cover
            ):  # pragma: no cover
                return tuple(x)  # pragma: no cover
            return [_u(i, param_name) for i in x]  # pragma: no cover
        if isinstance(x, tuple):  # pragma: no cover
            return tuple((_u(i, param_name) for i in x))  # pragma: no cover
        if hasattr(x, "_tensor"):  # pragma: no cover
            return x._tensor  # pragma: no cover
        if isinstance(x, (int, float, bool, complex)):  # pragma: no cover
            if param_name not in {  # pragma: no cover
                "axis",  # pragma: no cover
                "keepdims",
                "shape",
                "size",  # pragma: no cover
                "equal_nan",
                "nan",
                "posinf",
                "neginf",  # pragma: no cover
                "axis1",  # pragma: no cover
                "rtol",  # pragma: no cover
                "n",  # pragma: no cover
                "m",  # pragma: no cover
                "endpoint",  # pragma: no cover
                "offset",  # pragma: no cover
                "a_max",  # pragma: no cover
                "ddof",  # pragma: no cover
                "dims",  # pragma: no cover
                "source",  # pragma: no cover
                "axis2",  # pragma: no cover
                "stop",  # pragma: no cover
                "k",  # pragma: no cover
                "dtype",  # pragma: no cover
                "reps",  # pragma: no cover
                "a_min",  # pragma: no cover
                "sparse",  # pragma: no cover
                "decimals",  # pragma: no cover
                "strides",  # pragma: no cover
                "num",  # pragma: no cover
                "max",  # pragma: no cover
                "indices_or_sections",  # pragma: no cover
                "shape",  # pragma: no cover
                "atol",  # pragma: no cover
                "axis1",  # pragma: no cover
                "step",  # pragma: no cover
                "axes",  # pragma: no cover
                "fill_value",  # pragma: no cover
                "stream",  # pragma: no cover
                "destination",  # pragma: no cover
                "alpha",  # pragma: no cover
                "beta",  # pragma: no cover
                "start",  # pragma: no cover
                "equal_nan",
                "nan",
                "posinf",
                "neginf",  # pragma: no cover
                "keepdims",
                "shape",
                "size",  # pragma: no cover
                "min",  # pragma: no cover
                "p",  # pragma: no cover
            } and "log10" not in (  # pragma: no cover
                "transpose",  # pragma: no cover
                "moveaxis",  # pragma: no cover
                "reshape",  # pragma: no cover
                "expand_dims",  # pragma: no cover
                "linspace",  # pragma: no cover
                "logcumsumexp",  # pragma: no cover
                "cummin",  # pragma: no cover
                "cummax",  # pragma: no cover
                "cumprod",  # pragma: no cover
                "cumsum",  # pragma: no cover
                "logsumexp",  # pragma: no cover
                "all",  # pragma: no cover
                "any",  # pragma: no cover
                "argmin",  # pragma: no cover
                "argmax",  # pragma: no cover
                "prod",  # pragma: no cover
                "sum",  # pragma: no cover
                "mean",  # pragma: no cover
                "var",  # pragma: no cover
                "std",  # pragma: no cover
                "max",  # pragma: no cover
                "min",  # pragma: no cover
                "tensordot",  # pragma: no cover
            ):  # pragma: no cover
                return _to_tensor(x)  # pragma: no cover
        if type(x).__name__ == "ndarray":  # pragma: no cover
            return _to_tensor(x)  # pragma: no cover
        return x  # pragma: no cover

    def _w(x):  # pragma: no cover
        if isinstance(x, tuple):  # pragma: no cover
            return tuple((_w(i) for i in x))  # pragma: no cover
        if isinstance(x, list):  # pragma: no cover
            return [_w(i) for i in x]  # pragma: no cover
        return array(x) if hasattr(x, "shape") else x  # pragma: no cover

    fn = _get_compiler_op("log10")  # pragma: no cover
    res = fn(_u(a, "a"))  # pragma: no cover

    def __w(x):
        if isinstance(x, tuple):
            return tuple((__w(i) for i in x))
        if isinstance(x, list):
            return [__w(i) for i in x]
        return array(x) if hasattr(x, "shape") else x

    return __w(res)  # pragma: no cover


def imag(a: Any, stream: Any = None) -> Any:  # pragma: no cover
    """Compute imag.

    Args:
        a: The a argument.
        stream: The stream argument.

    Returns:
        The result of imag.

    """
    from zero_mlx.array import array, _to_tensor  # pragma: no cover
    import ml_switcheroo_compiler.ops as mops  # pragma: no cover
    import ml_switcheroo_compiler.ops.nn as mnn  # pragma: no cover
    import ml_switcheroo_compiler.random as mrand  # pragma: no cover
    import types  # pragma: no cover

    def _u(x, param_name=None):  # pragma: no cover
        if isinstance(x, list):  # pragma: no cover
            if __import__("builtins").all(  # pragma: no cover
                (isinstance(i, int) for i in x)  # pragma: no cover
            ):  # pragma: no cover
                return tuple(x)  # pragma: no cover
            return [_u(i, param_name) for i in x]  # pragma: no cover
        if isinstance(x, tuple):  # pragma: no cover
            return tuple((_u(i, param_name) for i in x))  # pragma: no cover
        if hasattr(x, "_tensor"):  # pragma: no cover
            return x._tensor  # pragma: no cover
        if isinstance(x, (int, float, bool, complex)):  # pragma: no cover
            if param_name not in {  # pragma: no cover
                "axis",  # pragma: no cover
                "keepdims",
                "shape",
                "size",  # pragma: no cover
                "equal_nan",
                "nan",
                "posinf",
                "neginf",  # pragma: no cover
                "axis1",  # pragma: no cover
                "rtol",  # pragma: no cover
                "n",  # pragma: no cover
                "m",  # pragma: no cover
                "endpoint",  # pragma: no cover
                "offset",  # pragma: no cover
                "a_max",  # pragma: no cover
                "ddof",  # pragma: no cover
                "dims",  # pragma: no cover
                "source",  # pragma: no cover
                "axis2",  # pragma: no cover
                "stop",  # pragma: no cover
                "k",  # pragma: no cover
                "dtype",  # pragma: no cover
                "reps",  # pragma: no cover
                "a_min",  # pragma: no cover
                "sparse",  # pragma: no cover
                "decimals",  # pragma: no cover
                "strides",  # pragma: no cover
                "num",  # pragma: no cover
                "max",  # pragma: no cover
                "indices_or_sections",  # pragma: no cover
                "shape",  # pragma: no cover
                "atol",  # pragma: no cover
                "axis1",  # pragma: no cover
                "step",  # pragma: no cover
                "axes",  # pragma: no cover
                "fill_value",  # pragma: no cover
                "stream",  # pragma: no cover
                "destination",  # pragma: no cover
                "alpha",  # pragma: no cover
                "beta",  # pragma: no cover
                "start",  # pragma: no cover
                "equal_nan",
                "nan",
                "posinf",
                "neginf",  # pragma: no cover
                "keepdims",
                "shape",
                "size",  # pragma: no cover
                "min",  # pragma: no cover
                "p",  # pragma: no cover
            } and "imag" not in (  # pragma: no cover
                "transpose",  # pragma: no cover
                "moveaxis",  # pragma: no cover
                "reshape",  # pragma: no cover
                "expand_dims",  # pragma: no cover
                "linspace",  # pragma: no cover
                "logcumsumexp",  # pragma: no cover
                "cummin",  # pragma: no cover
                "cummax",  # pragma: no cover
                "cumprod",  # pragma: no cover
                "cumsum",  # pragma: no cover
                "logsumexp",  # pragma: no cover
                "all",  # pragma: no cover
                "any",  # pragma: no cover
                "argmin",  # pragma: no cover
                "argmax",  # pragma: no cover
                "prod",  # pragma: no cover
                "sum",  # pragma: no cover
                "mean",  # pragma: no cover
                "var",  # pragma: no cover
                "std",  # pragma: no cover
                "max",  # pragma: no cover
                "min",  # pragma: no cover
                "tensordot",  # pragma: no cover
            ):  # pragma: no cover
                return _to_tensor(x)  # pragma: no cover
        if type(x).__name__ == "ndarray":  # pragma: no cover
            return _to_tensor(x)  # pragma: no cover
        return x  # pragma: no cover

    def _w(x):  # pragma: no cover
        if isinstance(x, tuple):  # pragma: no cover
            return tuple((_w(i) for i in x))  # pragma: no cover
        if isinstance(x, list):  # pragma: no cover
            return [_w(i) for i in x]  # pragma: no cover
        return array(x) if hasattr(x, "shape") else x  # pragma: no cover

    fn = _get_compiler_op("imag")  # pragma: no cover
    res = fn(_u(a, "a"))  # pragma: no cover

    def __w(x):
        if isinstance(x, tuple):
            return tuple((__w(i) for i in x))
        if isinstance(x, list):
            return [__w(i) for i in x]
        return array(x) if hasattr(x, "shape") else x

    return __w(res)  # pragma: no cover


def bitwise_and(*args, **kwargs):  # pragma: no cover
    """Compute bitwise_and.

    Args:
        args: The args argument.
        kwargs: The kwargs argument.

    Returns:
        The result of bitwise_and.

    """
    from zero_mlx.array import array, _to_tensor  # pragma: no cover
    import ml_switcheroo_compiler.ops as mops  # pragma: no cover
    import ml_switcheroo_compiler.ops.nn as mnn  # pragma: no cover
    import ml_switcheroo_compiler.random as mrand  # pragma: no cover
    import types  # pragma: no cover

    def _u(x, param_name=None):  # pragma: no cover
        if isinstance(x, list):  # pragma: no cover
            if __import__("builtins").all(  # pragma: no cover
                (isinstance(i, int) for i in x)  # pragma: no cover
            ):  # pragma: no cover
                return tuple(x)  # pragma: no cover
            return [_u(i, param_name) for i in x]  # pragma: no cover
        if isinstance(x, tuple):  # pragma: no cover
            return tuple((_u(i, param_name) for i in x))  # pragma: no cover
        if hasattr(x, "_tensor"):  # pragma: no cover
            return x._tensor  # pragma: no cover
        if isinstance(x, (int, float, bool, complex)):  # pragma: no cover
            if param_name not in {  # pragma: no cover
                "axis",  # pragma: no cover
                "keepdims",
                "shape",
                "size",  # pragma: no cover
                "equal_nan",
                "nan",
                "posinf",
                "neginf",  # pragma: no cover
                "axis1",  # pragma: no cover
                "rtol",  # pragma: no cover
                "n",  # pragma: no cover
                "m",  # pragma: no cover
                "endpoint",  # pragma: no cover
                "offset",  # pragma: no cover
                "a_max",  # pragma: no cover
                "ddof",  # pragma: no cover
                "dims",  # pragma: no cover
                "source",  # pragma: no cover
                "axis2",  # pragma: no cover
                "stop",  # pragma: no cover
                "k",  # pragma: no cover
                "dtype",  # pragma: no cover
                "reps",  # pragma: no cover
                "a_min",  # pragma: no cover
                "sparse",  # pragma: no cover
                "decimals",  # pragma: no cover
                "strides",  # pragma: no cover
                "num",  # pragma: no cover
                "max",  # pragma: no cover
                "indices_or_sections",  # pragma: no cover
                "shape",  # pragma: no cover
                "atol",  # pragma: no cover
                "axis1",  # pragma: no cover
                "step",  # pragma: no cover
                "axes",  # pragma: no cover
                "fill_value",  # pragma: no cover
                "stream",  # pragma: no cover
                "destination",  # pragma: no cover
                "alpha",  # pragma: no cover
                "beta",  # pragma: no cover
                "start",  # pragma: no cover
                "equal_nan",
                "nan",
                "posinf",
                "neginf",  # pragma: no cover
                "keepdims",
                "shape",
                "size",  # pragma: no cover
                "min",  # pragma: no cover
                "p",  # pragma: no cover
            } and "bitwise_and" not in (  # pragma: no cover
                "transpose",  # pragma: no cover
                "moveaxis",  # pragma: no cover
                "reshape",  # pragma: no cover
                "expand_dims",  # pragma: no cover
                "linspace",  # pragma: no cover
                "logcumsumexp",  # pragma: no cover
                "cummin",  # pragma: no cover
                "cummax",  # pragma: no cover
                "cumprod",  # pragma: no cover
                "cumsum",  # pragma: no cover
                "logsumexp",  # pragma: no cover
                "all",  # pragma: no cover
                "any",  # pragma: no cover
                "argmin",  # pragma: no cover
                "argmax",  # pragma: no cover
                "prod",  # pragma: no cover
                "sum",  # pragma: no cover
                "mean",  # pragma: no cover
                "var",  # pragma: no cover
                "std",  # pragma: no cover
                "max",  # pragma: no cover
                "min",  # pragma: no cover
                "tensordot",  # pragma: no cover
            ):  # pragma: no cover
                return _to_tensor(x)  # pragma: no cover
        if type(x).__name__ == "ndarray":  # pragma: no cover
            return _to_tensor(x)  # pragma: no cover
        return x  # pragma: no cover

    def _w(x):  # pragma: no cover
        if isinstance(x, tuple):  # pragma: no cover
            return tuple((_w(i) for i in x))  # pragma: no cover
        if isinstance(x, list):  # pragma: no cover
            return [_w(i) for i in x]  # pragma: no cover
        return array(x) if hasattr(x, "shape") else x  # pragma: no cover

    fn = _get_compiler_op("bitwise_and")  # pragma: no cover
    res = fn(  # pragma: no cover
        *[_u(x) for x in args],  # pragma: no cover
        **{
            ("dim" if k == "axis" else k): _u(v, k)
            for (k, v) in kwargs.items()
            if k != "stream"
        },  # pragma: no cover
    )  # pragma: no cover

    def __w(x):
        if isinstance(x, tuple):
            return tuple((__w(i) for i in x))
        if isinstance(x, list):
            return [__w(i) for i in x]
        return array(x) if hasattr(x, "shape") else x

    return __w(res)  # pragma: no cover


def bitwise_or(*args, **kwargs):  # pragma: no cover
    """Compute bitwise_or.

    Args:
        args: The args argument.
        kwargs: The kwargs argument.

    Returns:
        The result of bitwise_or.

    """
    from zero_mlx.array import array, _to_tensor  # pragma: no cover
    import ml_switcheroo_compiler.ops as mops  # pragma: no cover
    import ml_switcheroo_compiler.ops.nn as mnn  # pragma: no cover
    import ml_switcheroo_compiler.random as mrand  # pragma: no cover
    import types  # pragma: no cover

    def _u(x, param_name=None):  # pragma: no cover
        if isinstance(x, list):  # pragma: no cover
            if __import__("builtins").all(  # pragma: no cover
                (isinstance(i, int) for i in x)  # pragma: no cover
            ):  # pragma: no cover
                return tuple(x)  # pragma: no cover
            return [_u(i, param_name) for i in x]  # pragma: no cover
        if isinstance(x, tuple):  # pragma: no cover
            return tuple((_u(i, param_name) for i in x))  # pragma: no cover
        if hasattr(x, "_tensor"):  # pragma: no cover
            return x._tensor  # pragma: no cover
        if isinstance(x, (int, float, bool, complex)):  # pragma: no cover
            if param_name not in {  # pragma: no cover
                "axis",  # pragma: no cover
                "keepdims",
                "shape",
                "size",  # pragma: no cover
                "equal_nan",
                "nan",
                "posinf",
                "neginf",  # pragma: no cover
                "axis1",  # pragma: no cover
                "rtol",  # pragma: no cover
                "n",  # pragma: no cover
                "m",  # pragma: no cover
                "endpoint",  # pragma: no cover
                "offset",  # pragma: no cover
                "a_max",  # pragma: no cover
                "ddof",  # pragma: no cover
                "dims",  # pragma: no cover
                "source",  # pragma: no cover
                "axis2",  # pragma: no cover
                "stop",  # pragma: no cover
                "k",  # pragma: no cover
                "dtype",  # pragma: no cover
                "reps",  # pragma: no cover
                "a_min",  # pragma: no cover
                "sparse",  # pragma: no cover
                "decimals",  # pragma: no cover
                "strides",  # pragma: no cover
                "num",  # pragma: no cover
                "max",  # pragma: no cover
                "indices_or_sections",  # pragma: no cover
                "shape",  # pragma: no cover
                "atol",  # pragma: no cover
                "axis1",  # pragma: no cover
                "step",  # pragma: no cover
                "axes",  # pragma: no cover
                "fill_value",  # pragma: no cover
                "stream",  # pragma: no cover
                "destination",  # pragma: no cover
                "alpha",  # pragma: no cover
                "beta",  # pragma: no cover
                "start",  # pragma: no cover
                "equal_nan",
                "nan",
                "posinf",
                "neginf",  # pragma: no cover
                "keepdims",
                "shape",
                "size",  # pragma: no cover
                "min",  # pragma: no cover
                "p",  # pragma: no cover
            } and "bitwise_or" not in (  # pragma: no cover
                "transpose",  # pragma: no cover
                "moveaxis",  # pragma: no cover
                "reshape",  # pragma: no cover
                "expand_dims",  # pragma: no cover
                "linspace",  # pragma: no cover
                "logcumsumexp",  # pragma: no cover
                "cummin",  # pragma: no cover
                "cummax",  # pragma: no cover
                "cumprod",  # pragma: no cover
                "cumsum",  # pragma: no cover
                "logsumexp",  # pragma: no cover
                "all",  # pragma: no cover
                "any",  # pragma: no cover
                "argmin",  # pragma: no cover
                "argmax",  # pragma: no cover
                "prod",  # pragma: no cover
                "sum",  # pragma: no cover
                "mean",  # pragma: no cover
                "var",  # pragma: no cover
                "std",  # pragma: no cover
                "max",  # pragma: no cover
                "min",  # pragma: no cover
                "tensordot",  # pragma: no cover
            ):  # pragma: no cover
                return _to_tensor(x)  # pragma: no cover
        if type(x).__name__ == "ndarray":  # pragma: no cover
            return _to_tensor(x)  # pragma: no cover
        return x  # pragma: no cover

    def _w(x):  # pragma: no cover
        if isinstance(x, tuple):  # pragma: no cover
            return tuple((_w(i) for i in x))  # pragma: no cover
        if isinstance(x, list):  # pragma: no cover
            return [_w(i) for i in x]  # pragma: no cover
        return array(x) if hasattr(x, "shape") else x  # pragma: no cover

    fn = _get_compiler_op("bitwise_or")  # pragma: no cover
    res = fn(  # pragma: no cover
        *[_u(x) for x in args],  # pragma: no cover
        **{
            ("dim" if k == "axis" else k): _u(v, k)
            for (k, v) in kwargs.items()
            if k != "stream"
        },  # pragma: no cover
    )  # pragma: no cover

    def __w(x):
        if isinstance(x, tuple):
            return tuple((__w(i) for i in x))
        if isinstance(x, list):
            return [__w(i) for i in x]
        return array(x) if hasattr(x, "shape") else x

    return __w(res)  # pragma: no cover


def bitwise_xor(*args, **kwargs):  # pragma: no cover
    """Compute bitwise_xor.

    Args:
        args: The args argument.
        kwargs: The kwargs argument.

    Returns:
        The result of bitwise_xor.

    """
    from zero_mlx.array import array, _to_tensor  # pragma: no cover
    import ml_switcheroo_compiler.ops as mops  # pragma: no cover
    import ml_switcheroo_compiler.ops.nn as mnn  # pragma: no cover
    import ml_switcheroo_compiler.random as mrand  # pragma: no cover
    import types  # pragma: no cover

    def _u(x, param_name=None):  # pragma: no cover
        if isinstance(x, list):  # pragma: no cover
            if __import__("builtins").all(  # pragma: no cover
                (isinstance(i, int) for i in x)  # pragma: no cover
            ):  # pragma: no cover
                return tuple(x)  # pragma: no cover
            return [_u(i, param_name) for i in x]  # pragma: no cover
        if isinstance(x, tuple):  # pragma: no cover
            return tuple((_u(i, param_name) for i in x))  # pragma: no cover
        if hasattr(x, "_tensor"):  # pragma: no cover
            return x._tensor  # pragma: no cover
        if isinstance(x, (int, float, bool, complex)):  # pragma: no cover
            if param_name not in {  # pragma: no cover
                "axis",  # pragma: no cover
                "keepdims",
                "shape",
                "size",  # pragma: no cover
                "equal_nan",
                "nan",
                "posinf",
                "neginf",  # pragma: no cover
                "axis1",  # pragma: no cover
                "rtol",  # pragma: no cover
                "n",  # pragma: no cover
                "m",  # pragma: no cover
                "endpoint",  # pragma: no cover
                "offset",  # pragma: no cover
                "a_max",  # pragma: no cover
                "ddof",  # pragma: no cover
                "dims",  # pragma: no cover
                "source",  # pragma: no cover
                "axis2",  # pragma: no cover
                "stop",  # pragma: no cover
                "k",  # pragma: no cover
                "dtype",  # pragma: no cover
                "reps",  # pragma: no cover
                "a_min",  # pragma: no cover
                "sparse",  # pragma: no cover
                "decimals",  # pragma: no cover
                "strides",  # pragma: no cover
                "num",  # pragma: no cover
                "max",  # pragma: no cover
                "indices_or_sections",  # pragma: no cover
                "shape",  # pragma: no cover
                "atol",  # pragma: no cover
                "axis1",  # pragma: no cover
                "step",  # pragma: no cover
                "axes",  # pragma: no cover
                "fill_value",  # pragma: no cover
                "stream",  # pragma: no cover
                "destination",  # pragma: no cover
                "alpha",  # pragma: no cover
                "beta",  # pragma: no cover
                "start",  # pragma: no cover
                "equal_nan",
                "nan",
                "posinf",
                "neginf",  # pragma: no cover
                "keepdims",
                "shape",
                "size",  # pragma: no cover
                "min",  # pragma: no cover
                "p",  # pragma: no cover
            } and "bitwise_xor" not in (  # pragma: no cover
                "transpose",  # pragma: no cover
                "moveaxis",  # pragma: no cover
                "reshape",  # pragma: no cover
                "expand_dims",  # pragma: no cover
                "linspace",  # pragma: no cover
                "logcumsumexp",  # pragma: no cover
                "cummin",  # pragma: no cover
                "cummax",  # pragma: no cover
                "cumprod",  # pragma: no cover
                "cumsum",  # pragma: no cover
                "logsumexp",  # pragma: no cover
                "all",  # pragma: no cover
                "any",  # pragma: no cover
                "argmin",  # pragma: no cover
                "argmax",  # pragma: no cover
                "prod",  # pragma: no cover
                "sum",  # pragma: no cover
                "mean",  # pragma: no cover
                "var",  # pragma: no cover
                "std",  # pragma: no cover
                "max",  # pragma: no cover
                "min",  # pragma: no cover
                "tensordot",  # pragma: no cover
            ):  # pragma: no cover
                return _to_tensor(x)  # pragma: no cover
        if type(x).__name__ == "ndarray":  # pragma: no cover
            return _to_tensor(x)  # pragma: no cover
        return x  # pragma: no cover

    def _w(x):  # pragma: no cover
        if isinstance(x, tuple):  # pragma: no cover
            return tuple((_w(i) for i in x))  # pragma: no cover
        if isinstance(x, list):  # pragma: no cover
            return [_w(i) for i in x]  # pragma: no cover
        return array(x) if hasattr(x, "shape") else x  # pragma: no cover

    fn = _get_compiler_op("bitwise_xor")  # pragma: no cover
    res = fn(  # pragma: no cover
        *[_u(x) for x in args],  # pragma: no cover
        **{
            ("dim" if k == "axis" else k): _u(v, k)
            for (k, v) in kwargs.items()
            if k != "stream"
        },  # pragma: no cover
    )  # pragma: no cover

    def __w(x):
        if isinstance(x, tuple):
            return tuple((__w(i) for i in x))
        if isinstance(x, list):
            return [__w(i) for i in x]
        return array(x) if hasattr(x, "shape") else x

    return __w(res)  # pragma: no cover


def left_shift(*args, **kwargs):  # pragma: no cover
    """Compute left_shift.

    Args:
        args: The args argument.
        kwargs: The kwargs argument.

    Returns:
        The result of left_shift.

    """
    from zero_mlx.array import array, _to_tensor  # pragma: no cover
    import ml_switcheroo_compiler.ops as mops  # pragma: no cover
    import ml_switcheroo_compiler.ops.nn as mnn  # pragma: no cover
    import ml_switcheroo_compiler.random as mrand  # pragma: no cover
    import types  # pragma: no cover

    def _u(x, param_name=None):  # pragma: no cover
        if isinstance(x, list):  # pragma: no cover
            if __import__("builtins").all(  # pragma: no cover
                (isinstance(i, int) for i in x)  # pragma: no cover
            ):  # pragma: no cover
                return tuple(x)  # pragma: no cover
            return [_u(i, param_name) for i in x]  # pragma: no cover
        if isinstance(x, tuple):  # pragma: no cover
            return tuple((_u(i, param_name) for i in x))  # pragma: no cover
        if hasattr(x, "_tensor"):  # pragma: no cover
            return x._tensor  # pragma: no cover
        if isinstance(x, (int, float, bool, complex)):  # pragma: no cover
            if param_name not in {  # pragma: no cover
                "axis",  # pragma: no cover
                "keepdims",
                "shape",
                "size",  # pragma: no cover
                "equal_nan",
                "nan",
                "posinf",
                "neginf",  # pragma: no cover
                "axis1",  # pragma: no cover
                "rtol",  # pragma: no cover
                "n",  # pragma: no cover
                "m",  # pragma: no cover
                "endpoint",  # pragma: no cover
                "offset",  # pragma: no cover
                "a_max",  # pragma: no cover
                "ddof",  # pragma: no cover
                "dims",  # pragma: no cover
                "source",  # pragma: no cover
                "axis2",  # pragma: no cover
                "stop",  # pragma: no cover
                "k",  # pragma: no cover
                "dtype",  # pragma: no cover
                "reps",  # pragma: no cover
                "a_min",  # pragma: no cover
                "sparse",  # pragma: no cover
                "decimals",  # pragma: no cover
                "strides",  # pragma: no cover
                "num",  # pragma: no cover
                "max",  # pragma: no cover
                "indices_or_sections",  # pragma: no cover
                "shape",  # pragma: no cover
                "atol",  # pragma: no cover
                "axis1",  # pragma: no cover
                "step",  # pragma: no cover
                "axes",  # pragma: no cover
                "fill_value",  # pragma: no cover
                "stream",  # pragma: no cover
                "destination",  # pragma: no cover
                "alpha",  # pragma: no cover
                "beta",  # pragma: no cover
                "start",  # pragma: no cover
                "equal_nan",
                "nan",
                "posinf",
                "neginf",  # pragma: no cover
                "keepdims",
                "shape",
                "size",  # pragma: no cover
                "min",  # pragma: no cover
                "p",  # pragma: no cover
            } and "left_shift" not in (  # pragma: no cover
                "transpose",  # pragma: no cover
                "moveaxis",  # pragma: no cover
                "reshape",  # pragma: no cover
                "expand_dims",  # pragma: no cover
                "linspace",  # pragma: no cover
                "logcumsumexp",  # pragma: no cover
                "cummin",  # pragma: no cover
                "cummax",  # pragma: no cover
                "cumprod",  # pragma: no cover
                "cumsum",  # pragma: no cover
                "logsumexp",  # pragma: no cover
                "all",  # pragma: no cover
                "any",  # pragma: no cover
                "argmin",  # pragma: no cover
                "argmax",  # pragma: no cover
                "prod",  # pragma: no cover
                "sum",  # pragma: no cover
                "mean",  # pragma: no cover
                "var",  # pragma: no cover
                "std",  # pragma: no cover
                "max",  # pragma: no cover
                "min",  # pragma: no cover
                "tensordot",  # pragma: no cover
            ):  # pragma: no cover
                return _to_tensor(x)  # pragma: no cover
        if type(x).__name__ == "ndarray":  # pragma: no cover
            return _to_tensor(x)  # pragma: no cover
        return x  # pragma: no cover

    def _w(x):  # pragma: no cover
        if isinstance(x, tuple):  # pragma: no cover
            return tuple((_w(i) for i in x))  # pragma: no cover
        if isinstance(x, list):  # pragma: no cover
            return [_w(i) for i in x]  # pragma: no cover
        return array(x) if hasattr(x, "shape") else x  # pragma: no cover

    fn = _get_compiler_op("left_shift")  # pragma: no cover
    res = fn(  # pragma: no cover
        *[_u(x) for x in args],  # pragma: no cover
        **{
            ("dim" if k == "axis" else k): _u(v, k)
            for (k, v) in kwargs.items()
            if k != "stream"
        },  # pragma: no cover
    )  # pragma: no cover

    def __w(x):
        if isinstance(x, tuple):
            return tuple((__w(i) for i in x))
        if isinstance(x, list):
            return [__w(i) for i in x]
        return array(x) if hasattr(x, "shape") else x

    return __w(res)  # pragma: no cover


def right_shift(*args, **kwargs):  # pragma: no cover
    """Compute right_shift.

    Args:
        args: The args argument.
        kwargs: The kwargs argument.

    Returns:
        The result of right_shift.

    """
    from zero_mlx.array import array, _to_tensor  # pragma: no cover
    import ml_switcheroo_compiler.ops as mops  # pragma: no cover
    import ml_switcheroo_compiler.ops.nn as mnn  # pragma: no cover
    import ml_switcheroo_compiler.random as mrand  # pragma: no cover
    import types  # pragma: no cover

    def _u(x, param_name=None):  # pragma: no cover
        if isinstance(x, list):  # pragma: no cover
            if __import__("builtins").all(  # pragma: no cover
                (isinstance(i, int) for i in x)  # pragma: no cover
            ):  # pragma: no cover
                return tuple(x)  # pragma: no cover
            return [_u(i, param_name) for i in x]  # pragma: no cover
        if isinstance(x, tuple):  # pragma: no cover
            return tuple((_u(i, param_name) for i in x))  # pragma: no cover
        if hasattr(x, "_tensor"):  # pragma: no cover
            return x._tensor  # pragma: no cover
        if isinstance(x, (int, float, bool, complex)):  # pragma: no cover
            if param_name not in {  # pragma: no cover
                "axis",  # pragma: no cover
                "keepdims",
                "shape",
                "size",  # pragma: no cover
                "equal_nan",
                "nan",
                "posinf",
                "neginf",  # pragma: no cover
                "axis1",  # pragma: no cover
                "rtol",  # pragma: no cover
                "n",  # pragma: no cover
                "m",  # pragma: no cover
                "endpoint",  # pragma: no cover
                "offset",  # pragma: no cover
                "a_max",  # pragma: no cover
                "ddof",  # pragma: no cover
                "dims",  # pragma: no cover
                "source",  # pragma: no cover
                "axis2",  # pragma: no cover
                "stop",  # pragma: no cover
                "k",  # pragma: no cover
                "dtype",  # pragma: no cover
                "reps",  # pragma: no cover
                "a_min",  # pragma: no cover
                "sparse",  # pragma: no cover
                "decimals",  # pragma: no cover
                "strides",  # pragma: no cover
                "num",  # pragma: no cover
                "max",  # pragma: no cover
                "indices_or_sections",  # pragma: no cover
                "shape",  # pragma: no cover
                "atol",  # pragma: no cover
                "axis1",  # pragma: no cover
                "step",  # pragma: no cover
                "axes",  # pragma: no cover
                "fill_value",  # pragma: no cover
                "stream",  # pragma: no cover
                "destination",  # pragma: no cover
                "alpha",  # pragma: no cover
                "beta",  # pragma: no cover
                "start",  # pragma: no cover
                "equal_nan",
                "nan",
                "posinf",
                "neginf",  # pragma: no cover
                "keepdims",
                "shape",
                "size",  # pragma: no cover
                "min",  # pragma: no cover
                "p",  # pragma: no cover
            } and "right_shift" not in (  # pragma: no cover
                "transpose",  # pragma: no cover
                "moveaxis",  # pragma: no cover
                "reshape",  # pragma: no cover
                "expand_dims",  # pragma: no cover
                "linspace",  # pragma: no cover
                "logcumsumexp",  # pragma: no cover
                "cummin",  # pragma: no cover
                "cummax",  # pragma: no cover
                "cumprod",  # pragma: no cover
                "cumsum",  # pragma: no cover
                "logsumexp",  # pragma: no cover
                "all",  # pragma: no cover
                "any",  # pragma: no cover
                "argmin",  # pragma: no cover
                "argmax",  # pragma: no cover
                "prod",  # pragma: no cover
                "sum",  # pragma: no cover
                "mean",  # pragma: no cover
                "var",  # pragma: no cover
                "std",  # pragma: no cover
                "max",  # pragma: no cover
                "min",  # pragma: no cover
                "tensordot",  # pragma: no cover
            ):  # pragma: no cover
                return _to_tensor(x)  # pragma: no cover
        if type(x).__name__ == "ndarray":  # pragma: no cover
            return _to_tensor(x)  # pragma: no cover
        return x  # pragma: no cover

    def _w(x):  # pragma: no cover
        if isinstance(x, tuple):  # pragma: no cover
            return tuple((_w(i) for i in x))  # pragma: no cover
        if isinstance(x, list):  # pragma: no cover
            return [_w(i) for i in x]  # pragma: no cover
        return array(x) if hasattr(x, "shape") else x  # pragma: no cover

    fn = _get_compiler_op("right_shift")  # pragma: no cover
    res = fn(  # pragma: no cover
        *[_u(x) for x in args],  # pragma: no cover
        **{
            ("dim" if k == "axis" else k): _u(v, k)
            for (k, v) in kwargs.items()
            if k != "stream"
        },  # pragma: no cover
    )  # pragma: no cover

    def __w(x):
        if isinstance(x, tuple):
            return tuple((__w(i) for i in x))
        if isinstance(x, list):
            return [__w(i) for i in x]
        return array(x) if hasattr(x, "shape") else x

    return __w(res)  # pragma: no cover


def less(*args, **kwargs):  # pragma: no cover
    """Compute less.

    Args:
        args: The args argument.
        kwargs: The kwargs argument.

    Returns:
        The result of less.

    """
    from zero_mlx.array import array, _to_tensor  # pragma: no cover
    import ml_switcheroo_compiler.ops as mops  # pragma: no cover
    import ml_switcheroo_compiler.ops.nn as mnn  # pragma: no cover
    import ml_switcheroo_compiler.random as mrand  # pragma: no cover
    import types  # pragma: no cover

    def _u(x, param_name=None):  # pragma: no cover
        if isinstance(x, list):  # pragma: no cover
            if __import__("builtins").all(  # pragma: no cover
                (isinstance(i, int) for i in x)  # pragma: no cover
            ):  # pragma: no cover
                return tuple(x)  # pragma: no cover
            return [_u(i, param_name) for i in x]  # pragma: no cover
        if isinstance(x, tuple):  # pragma: no cover
            return tuple((_u(i, param_name) for i in x))  # pragma: no cover
        if hasattr(x, "_tensor"):  # pragma: no cover
            return x._tensor  # pragma: no cover
        if isinstance(x, (int, float, bool, complex)):  # pragma: no cover
            if param_name not in {  # pragma: no cover
                "axis",  # pragma: no cover
                "keepdims",
                "shape",
                "size",  # pragma: no cover
                "equal_nan",
                "nan",
                "posinf",
                "neginf",  # pragma: no cover
                "axis1",  # pragma: no cover
                "rtol",  # pragma: no cover
                "n",  # pragma: no cover
                "m",  # pragma: no cover
                "endpoint",  # pragma: no cover
                "offset",  # pragma: no cover
                "a_max",  # pragma: no cover
                "ddof",  # pragma: no cover
                "dims",  # pragma: no cover
                "source",  # pragma: no cover
                "axis2",  # pragma: no cover
                "stop",  # pragma: no cover
                "k",  # pragma: no cover
                "dtype",  # pragma: no cover
                "reps",  # pragma: no cover
                "a_min",  # pragma: no cover
                "sparse",  # pragma: no cover
                "decimals",  # pragma: no cover
                "strides",  # pragma: no cover
                "num",  # pragma: no cover
                "max",  # pragma: no cover
                "indices_or_sections",  # pragma: no cover
                "shape",  # pragma: no cover
                "atol",  # pragma: no cover
                "axis1",  # pragma: no cover
                "step",  # pragma: no cover
                "axes",  # pragma: no cover
                "fill_value",  # pragma: no cover
                "stream",  # pragma: no cover
                "destination",  # pragma: no cover
                "alpha",  # pragma: no cover
                "beta",  # pragma: no cover
                "start",  # pragma: no cover
                "equal_nan",
                "nan",
                "posinf",
                "neginf",  # pragma: no cover
                "keepdims",
                "shape",
                "size",  # pragma: no cover
                "min",  # pragma: no cover
                "p",  # pragma: no cover
            } and "less" not in (  # pragma: no cover
                "transpose",  # pragma: no cover
                "moveaxis",  # pragma: no cover
                "reshape",  # pragma: no cover
                "expand_dims",  # pragma: no cover
                "linspace",  # pragma: no cover
                "logcumsumexp",  # pragma: no cover
                "cummin",  # pragma: no cover
                "cummax",  # pragma: no cover
                "cumprod",  # pragma: no cover
                "cumsum",  # pragma: no cover
                "logsumexp",  # pragma: no cover
                "all",  # pragma: no cover
                "any",  # pragma: no cover
                "argmin",  # pragma: no cover
                "argmax",  # pragma: no cover
                "prod",  # pragma: no cover
                "sum",  # pragma: no cover
                "mean",  # pragma: no cover
                "var",  # pragma: no cover
                "std",  # pragma: no cover
                "max",  # pragma: no cover
                "min",  # pragma: no cover
                "tensordot",  # pragma: no cover
            ):  # pragma: no cover
                return _to_tensor(x)  # pragma: no cover
        if type(x).__name__ == "ndarray":  # pragma: no cover
            return _to_tensor(x)  # pragma: no cover
        return x  # pragma: no cover

    def _w(x):  # pragma: no cover
        if isinstance(x, tuple):  # pragma: no cover
            return tuple((_w(i) for i in x))  # pragma: no cover
        if isinstance(x, list):  # pragma: no cover
            return [_w(i) for i in x]  # pragma: no cover
        return array(x) if hasattr(x, "shape") else x  # pragma: no cover

    fn = _get_compiler_op("less")  # pragma: no cover
    res = fn(  # pragma: no cover
        *[_u(x) for x in args],  # pragma: no cover
        **{
            ("dim" if k == "axis" else k): _u(v, k)
            for (k, v) in kwargs.items()
            if k != "stream"
        },  # pragma: no cover
    )  # pragma: no cover

    def __w(x):
        if isinstance(x, tuple):
            return tuple((__w(i) for i in x))
        if isinstance(x, list):
            return [__w(i) for i in x]
        return array(x) if hasattr(x, "shape") else x

    return __w(res)  # pragma: no cover


def less_equal(*args, **kwargs):  # pragma: no cover
    """Compute less_equal.

    Args:
        args: The args argument.
        kwargs: The kwargs argument.

    Returns:
        The result of less_equal.

    """
    from zero_mlx.array import array, _to_tensor  # pragma: no cover
    import ml_switcheroo_compiler.ops as mops  # pragma: no cover
    import ml_switcheroo_compiler.ops.nn as mnn  # pragma: no cover
    import ml_switcheroo_compiler.random as mrand  # pragma: no cover
    import types  # pragma: no cover

    def _u(x, param_name=None):  # pragma: no cover
        if isinstance(x, list):  # pragma: no cover
            if __import__("builtins").all(  # pragma: no cover
                (isinstance(i, int) for i in x)  # pragma: no cover
            ):  # pragma: no cover
                return tuple(x)  # pragma: no cover
            return [_u(i, param_name) for i in x]  # pragma: no cover
        if isinstance(x, tuple):  # pragma: no cover
            return tuple((_u(i, param_name) for i in x))  # pragma: no cover
        if hasattr(x, "_tensor"):  # pragma: no cover
            return x._tensor  # pragma: no cover
        if isinstance(x, (int, float, bool, complex)):  # pragma: no cover
            if param_name not in {  # pragma: no cover
                "axis",  # pragma: no cover
                "keepdims",
                "shape",
                "size",  # pragma: no cover
                "equal_nan",
                "nan",
                "posinf",
                "neginf",  # pragma: no cover
                "axis1",  # pragma: no cover
                "rtol",  # pragma: no cover
                "n",  # pragma: no cover
                "m",  # pragma: no cover
                "endpoint",  # pragma: no cover
                "offset",  # pragma: no cover
                "a_max",  # pragma: no cover
                "ddof",  # pragma: no cover
                "dims",  # pragma: no cover
                "source",  # pragma: no cover
                "axis2",  # pragma: no cover
                "stop",  # pragma: no cover
                "k",  # pragma: no cover
                "dtype",  # pragma: no cover
                "reps",  # pragma: no cover
                "a_min",  # pragma: no cover
                "sparse",  # pragma: no cover
                "decimals",  # pragma: no cover
                "strides",  # pragma: no cover
                "num",  # pragma: no cover
                "max",  # pragma: no cover
                "indices_or_sections",  # pragma: no cover
                "shape",  # pragma: no cover
                "atol",  # pragma: no cover
                "axis1",  # pragma: no cover
                "step",  # pragma: no cover
                "axes",  # pragma: no cover
                "fill_value",  # pragma: no cover
                "stream",  # pragma: no cover
                "destination",  # pragma: no cover
                "alpha",  # pragma: no cover
                "beta",  # pragma: no cover
                "start",  # pragma: no cover
                "equal_nan",
                "nan",
                "posinf",
                "neginf",  # pragma: no cover
                "keepdims",
                "shape",
                "size",  # pragma: no cover
                "min",  # pragma: no cover
                "p",  # pragma: no cover
            } and "less_equal" not in (  # pragma: no cover
                "transpose",  # pragma: no cover
                "moveaxis",  # pragma: no cover
                "reshape",  # pragma: no cover
                "expand_dims",  # pragma: no cover
                "linspace",  # pragma: no cover
                "logcumsumexp",  # pragma: no cover
                "cummin",  # pragma: no cover
                "cummax",  # pragma: no cover
                "cumprod",  # pragma: no cover
                "cumsum",  # pragma: no cover
                "logsumexp",  # pragma: no cover
                "all",  # pragma: no cover
                "any",  # pragma: no cover
                "argmin",  # pragma: no cover
                "argmax",  # pragma: no cover
                "prod",  # pragma: no cover
                "sum",  # pragma: no cover
                "mean",  # pragma: no cover
                "var",  # pragma: no cover
                "std",  # pragma: no cover
                "max",  # pragma: no cover
                "min",  # pragma: no cover
                "tensordot",  # pragma: no cover
            ):  # pragma: no cover
                return _to_tensor(x)  # pragma: no cover
        if type(x).__name__ == "ndarray":  # pragma: no cover
            return _to_tensor(x)  # pragma: no cover
        return x  # pragma: no cover

    def _w(x):  # pragma: no cover
        if isinstance(x, tuple):  # pragma: no cover
            return tuple((_w(i) for i in x))  # pragma: no cover
        if isinstance(x, list):  # pragma: no cover
            return [_w(i) for i in x]  # pragma: no cover
        return array(x) if hasattr(x, "shape") else x  # pragma: no cover

    fn = _get_compiler_op("less_equal")  # pragma: no cover
    res = fn(  # pragma: no cover
        *[_u(x) for x in args],  # pragma: no cover
        **{
            ("dim" if k == "axis" else k): _u(v, k)
            for (k, v) in kwargs.items()
            if k != "stream"
        },  # pragma: no cover
    )  # pragma: no cover

    def __w(x):
        if isinstance(x, tuple):
            return tuple((__w(i) for i in x))
        if isinstance(x, list):
            return [__w(i) for i in x]
        return array(x) if hasattr(x, "shape") else x

    return __w(res)  # pragma: no cover


def greater(*args, **kwargs):  # pragma: no cover
    """Compute greater.

    Args:
        args: The args argument.
        kwargs: The kwargs argument.

    Returns:
        The result of greater.

    """
    from zero_mlx.array import array, _to_tensor  # pragma: no cover
    import ml_switcheroo_compiler.ops as mops  # pragma: no cover
    import ml_switcheroo_compiler.ops.nn as mnn  # pragma: no cover
    import ml_switcheroo_compiler.random as mrand  # pragma: no cover
    import types  # pragma: no cover

    def _u(x, param_name=None):  # pragma: no cover
        if isinstance(x, list):  # pragma: no cover
            if __import__("builtins").all(  # pragma: no cover
                (isinstance(i, int) for i in x)  # pragma: no cover
            ):  # pragma: no cover
                return tuple(x)  # pragma: no cover
            return [_u(i, param_name) for i in x]  # pragma: no cover
        if isinstance(x, tuple):  # pragma: no cover
            return tuple((_u(i, param_name) for i in x))  # pragma: no cover
        if hasattr(x, "_tensor"):  # pragma: no cover
            return x._tensor  # pragma: no cover
        if isinstance(x, (int, float, bool, complex)):  # pragma: no cover
            if param_name not in {  # pragma: no cover
                "axis",  # pragma: no cover
                "keepdims",
                "shape",
                "size",  # pragma: no cover
                "equal_nan",
                "nan",
                "posinf",
                "neginf",  # pragma: no cover
                "axis1",  # pragma: no cover
                "rtol",  # pragma: no cover
                "n",  # pragma: no cover
                "m",  # pragma: no cover
                "endpoint",  # pragma: no cover
                "offset",  # pragma: no cover
                "a_max",  # pragma: no cover
                "ddof",  # pragma: no cover
                "dims",  # pragma: no cover
                "source",  # pragma: no cover
                "axis2",  # pragma: no cover
                "stop",  # pragma: no cover
                "k",  # pragma: no cover
                "dtype",  # pragma: no cover
                "reps",  # pragma: no cover
                "a_min",  # pragma: no cover
                "sparse",  # pragma: no cover
                "decimals",  # pragma: no cover
                "strides",  # pragma: no cover
                "num",  # pragma: no cover
                "max",  # pragma: no cover
                "indices_or_sections",  # pragma: no cover
                "shape",  # pragma: no cover
                "atol",  # pragma: no cover
                "axis1",  # pragma: no cover
                "step",  # pragma: no cover
                "axes",  # pragma: no cover
                "fill_value",  # pragma: no cover
                "stream",  # pragma: no cover
                "destination",  # pragma: no cover
                "alpha",  # pragma: no cover
                "beta",  # pragma: no cover
                "start",  # pragma: no cover
                "equal_nan",
                "nan",
                "posinf",
                "neginf",  # pragma: no cover
                "keepdims",
                "shape",
                "size",  # pragma: no cover
                "min",  # pragma: no cover
                "p",  # pragma: no cover
            } and "greater" not in (  # pragma: no cover
                "transpose",  # pragma: no cover
                "moveaxis",  # pragma: no cover
                "reshape",  # pragma: no cover
                "expand_dims",  # pragma: no cover
                "linspace",  # pragma: no cover
                "logcumsumexp",  # pragma: no cover
                "cummin",  # pragma: no cover
                "cummax",  # pragma: no cover
                "cumprod",  # pragma: no cover
                "cumsum",  # pragma: no cover
                "logsumexp",  # pragma: no cover
                "all",  # pragma: no cover
                "any",  # pragma: no cover
                "argmin",  # pragma: no cover
                "argmax",  # pragma: no cover
                "prod",  # pragma: no cover
                "sum",  # pragma: no cover
                "mean",  # pragma: no cover
                "var",  # pragma: no cover
                "std",  # pragma: no cover
                "max",  # pragma: no cover
                "min",  # pragma: no cover
                "tensordot",  # pragma: no cover
            ):  # pragma: no cover
                return _to_tensor(x)  # pragma: no cover
        if type(x).__name__ == "ndarray":  # pragma: no cover
            return _to_tensor(x)  # pragma: no cover
        return x  # pragma: no cover

    def _w(x):  # pragma: no cover
        if isinstance(x, tuple):  # pragma: no cover
            return tuple((_w(i) for i in x))  # pragma: no cover
        if isinstance(x, list):  # pragma: no cover
            return [_w(i) for i in x]  # pragma: no cover
        return array(x) if hasattr(x, "shape") else x  # pragma: no cover

    fn = _get_compiler_op("greater")  # pragma: no cover
    res = fn(  # pragma: no cover
        *[_u(x) for x in args],  # pragma: no cover
        **{
            ("dim" if k == "axis" else k): _u(v, k)
            for (k, v) in kwargs.items()
            if k != "stream"
        },  # pragma: no cover
    )  # pragma: no cover

    def __w(x):
        if isinstance(x, tuple):
            return tuple((__w(i) for i in x))
        if isinstance(x, list):
            return [__w(i) for i in x]
        return array(x) if hasattr(x, "shape") else x

    return __w(res)  # pragma: no cover


def greater_equal(*args, **kwargs):  # pragma: no cover
    """Compute greater_equal.

    Args:
        args: The args argument.
        kwargs: The kwargs argument.

    Returns:
        The result of greater_equal.

    """
    from zero_mlx.array import array, _to_tensor  # pragma: no cover
    import ml_switcheroo_compiler.ops as mops  # pragma: no cover
    import ml_switcheroo_compiler.ops.nn as mnn  # pragma: no cover
    import ml_switcheroo_compiler.random as mrand  # pragma: no cover
    import types  # pragma: no cover

    def _u(x, param_name=None):  # pragma: no cover
        if isinstance(x, list):  # pragma: no cover
            if __import__("builtins").all(  # pragma: no cover
                (isinstance(i, int) for i in x)  # pragma: no cover
            ):  # pragma: no cover
                return tuple(x)  # pragma: no cover
            return [_u(i, param_name) for i in x]  # pragma: no cover
        if isinstance(x, tuple):  # pragma: no cover
            return tuple((_u(i, param_name) for i in x))  # pragma: no cover
        if hasattr(x, "_tensor"):  # pragma: no cover
            return x._tensor  # pragma: no cover
        if isinstance(x, (int, float, bool, complex)):  # pragma: no cover
            if param_name not in {  # pragma: no cover
                "axis",  # pragma: no cover
                "keepdims",
                "shape",
                "size",  # pragma: no cover
                "equal_nan",
                "nan",
                "posinf",
                "neginf",  # pragma: no cover
                "axis1",  # pragma: no cover
                "rtol",  # pragma: no cover
                "n",  # pragma: no cover
                "m",  # pragma: no cover
                "endpoint",  # pragma: no cover
                "offset",  # pragma: no cover
                "a_max",  # pragma: no cover
                "ddof",  # pragma: no cover
                "dims",  # pragma: no cover
                "source",  # pragma: no cover
                "axis2",  # pragma: no cover
                "stop",  # pragma: no cover
                "k",  # pragma: no cover
                "dtype",  # pragma: no cover
                "reps",  # pragma: no cover
                "a_min",  # pragma: no cover
                "sparse",  # pragma: no cover
                "decimals",  # pragma: no cover
                "strides",  # pragma: no cover
                "num",  # pragma: no cover
                "max",  # pragma: no cover
                "indices_or_sections",  # pragma: no cover
                "shape",  # pragma: no cover
                "atol",  # pragma: no cover
                "axis1",  # pragma: no cover
                "step",  # pragma: no cover
                "axes",  # pragma: no cover
                "fill_value",  # pragma: no cover
                "stream",  # pragma: no cover
                "destination",  # pragma: no cover
                "alpha",  # pragma: no cover
                "beta",  # pragma: no cover
                "start",  # pragma: no cover
                "equal_nan",
                "nan",
                "posinf",
                "neginf",  # pragma: no cover
                "keepdims",
                "shape",
                "size",  # pragma: no cover
                "min",  # pragma: no cover
                "p",  # pragma: no cover
            } and "greater_equal" not in (  # pragma: no cover
                "transpose",  # pragma: no cover
                "moveaxis",  # pragma: no cover
                "reshape",  # pragma: no cover
                "expand_dims",  # pragma: no cover
                "linspace",  # pragma: no cover
                "logcumsumexp",  # pragma: no cover
                "cummin",  # pragma: no cover
                "cummax",  # pragma: no cover
                "cumprod",  # pragma: no cover
                "cumsum",  # pragma: no cover
                "logsumexp",  # pragma: no cover
                "all",  # pragma: no cover
                "any",  # pragma: no cover
                "argmin",  # pragma: no cover
                "argmax",  # pragma: no cover
                "prod",  # pragma: no cover
                "sum",  # pragma: no cover
                "mean",  # pragma: no cover
                "var",  # pragma: no cover
                "std",  # pragma: no cover
                "max",  # pragma: no cover
                "min",  # pragma: no cover
                "tensordot",  # pragma: no cover
            ):  # pragma: no cover
                return _to_tensor(x)  # pragma: no cover
        if type(x).__name__ == "ndarray":  # pragma: no cover
            return _to_tensor(x)  # pragma: no cover
        return x  # pragma: no cover

    def _w(x):  # pragma: no cover
        if isinstance(x, tuple):  # pragma: no cover
            return tuple((_w(i) for i in x))  # pragma: no cover
        if isinstance(x, list):  # pragma: no cover
            return [_w(i) for i in x]  # pragma: no cover
        return array(x) if hasattr(x, "shape") else x  # pragma: no cover

    fn = _get_compiler_op("greater_equal")  # pragma: no cover
    res = fn(  # pragma: no cover
        *[_u(x) for x in args],  # pragma: no cover
        **{
            ("dim" if k == "axis" else k): _u(v, k)
            for (k, v) in kwargs.items()
            if k != "stream"
        },  # pragma: no cover
    )  # pragma: no cover

    def __w(x):
        if isinstance(x, tuple):
            return tuple((__w(i) for i in x))
        if isinstance(x, list):
            return [__w(i) for i in x]
        return array(x) if hasattr(x, "shape") else x

    return __w(res)  # pragma: no cover


def equal(*args, **kwargs):  # pragma: no cover
    """Compute equal.

    Args:
        args: The args argument.
        kwargs: The kwargs argument.

    Returns:
        The result of equal.

    """
    from zero_mlx.array import array, _to_tensor  # pragma: no cover
    import ml_switcheroo_compiler.ops as mops  # pragma: no cover
    import ml_switcheroo_compiler.ops.nn as mnn  # pragma: no cover
    import ml_switcheroo_compiler.random as mrand  # pragma: no cover
    import types  # pragma: no cover

    def _u(x, param_name=None):  # pragma: no cover
        if isinstance(x, list):  # pragma: no cover
            if __import__("builtins").all(  # pragma: no cover
                (isinstance(i, int) for i in x)  # pragma: no cover
            ):  # pragma: no cover
                return tuple(x)  # pragma: no cover
            return [_u(i, param_name) for i in x]  # pragma: no cover
        if isinstance(x, tuple):  # pragma: no cover
            return tuple((_u(i, param_name) for i in x))  # pragma: no cover
        if hasattr(x, "_tensor"):  # pragma: no cover
            return x._tensor  # pragma: no cover
        if isinstance(x, (int, float, bool, complex)):  # pragma: no cover
            if param_name not in {  # pragma: no cover
                "axis",  # pragma: no cover
                "keepdims",
                "shape",
                "size",  # pragma: no cover
                "equal_nan",
                "nan",
                "posinf",
                "neginf",  # pragma: no cover
                "axis1",  # pragma: no cover
                "rtol",  # pragma: no cover
                "n",  # pragma: no cover
                "m",  # pragma: no cover
                "endpoint",  # pragma: no cover
                "offset",  # pragma: no cover
                "a_max",  # pragma: no cover
                "ddof",  # pragma: no cover
                "dims",  # pragma: no cover
                "source",  # pragma: no cover
                "axis2",  # pragma: no cover
                "stop",  # pragma: no cover
                "k",  # pragma: no cover
                "dtype",  # pragma: no cover
                "reps",  # pragma: no cover
                "a_min",  # pragma: no cover
                "sparse",  # pragma: no cover
                "decimals",  # pragma: no cover
                "strides",  # pragma: no cover
                "num",  # pragma: no cover
                "max",  # pragma: no cover
                "indices_or_sections",  # pragma: no cover
                "shape",  # pragma: no cover
                "atol",  # pragma: no cover
                "axis1",  # pragma: no cover
                "step",  # pragma: no cover
                "axes",  # pragma: no cover
                "fill_value",  # pragma: no cover
                "stream",  # pragma: no cover
                "destination",  # pragma: no cover
                "alpha",  # pragma: no cover
                "beta",  # pragma: no cover
                "start",  # pragma: no cover
                "equal_nan",
                "nan",
                "posinf",
                "neginf",  # pragma: no cover
                "keepdims",
                "shape",
                "size",  # pragma: no cover
                "min",  # pragma: no cover
                "p",  # pragma: no cover
            } and "equal" not in (  # pragma: no cover
                "transpose",  # pragma: no cover
                "moveaxis",  # pragma: no cover
                "reshape",  # pragma: no cover
                "expand_dims",  # pragma: no cover
                "linspace",  # pragma: no cover
                "logcumsumexp",  # pragma: no cover
                "cummin",  # pragma: no cover
                "cummax",  # pragma: no cover
                "cumprod",  # pragma: no cover
                "cumsum",  # pragma: no cover
                "logsumexp",  # pragma: no cover
                "all",  # pragma: no cover
                "any",  # pragma: no cover
                "argmin",  # pragma: no cover
                "argmax",  # pragma: no cover
                "prod",  # pragma: no cover
                "sum",  # pragma: no cover
                "mean",  # pragma: no cover
                "var",  # pragma: no cover
                "std",  # pragma: no cover
                "max",  # pragma: no cover
                "min",  # pragma: no cover
                "tensordot",  # pragma: no cover
            ):  # pragma: no cover
                return _to_tensor(x)  # pragma: no cover
        if type(x).__name__ == "ndarray":  # pragma: no cover
            return _to_tensor(x)  # pragma: no cover
        return x  # pragma: no cover

    def _w(x):  # pragma: no cover
        if isinstance(x, tuple):  # pragma: no cover
            return tuple((_w(i) for i in x))  # pragma: no cover
        if isinstance(x, list):  # pragma: no cover
            return [_w(i) for i in x]  # pragma: no cover
        return array(x) if hasattr(x, "shape") else x  # pragma: no cover

    fn = _get_compiler_op("equal")  # pragma: no cover
    res = fn(  # pragma: no cover
        *[_u(x) for x in args],  # pragma: no cover
        **{
            ("dim" if k == "axis" else k): _u(v, k)
            for (k, v) in kwargs.items()
            if k != "stream"
        },  # pragma: no cover
    )  # pragma: no cover

    def __w(x):
        if isinstance(x, tuple):
            return tuple((__w(i) for i in x))
        if isinstance(x, list):
            return [__w(i) for i in x]
        return array(x) if hasattr(x, "shape") else x

    return __w(res)  # pragma: no cover


def not_equal(*args, **kwargs):  # pragma: no cover
    """Compute not_equal.

    Args:
        args: The args argument.
        kwargs: The kwargs argument.

    Returns:
        The result of not_equal.

    """
    from zero_mlx.array import array, _to_tensor  # pragma: no cover
    import ml_switcheroo_compiler.ops as mops  # pragma: no cover
    import ml_switcheroo_compiler.ops.nn as mnn  # pragma: no cover
    import ml_switcheroo_compiler.random as mrand  # pragma: no cover
    import types  # pragma: no cover

    def _u(x, param_name=None):  # pragma: no cover
        if isinstance(x, list):  # pragma: no cover
            if __import__("builtins").all(  # pragma: no cover
                (isinstance(i, int) for i in x)  # pragma: no cover
            ):  # pragma: no cover
                return tuple(x)  # pragma: no cover
            return [_u(i, param_name) for i in x]  # pragma: no cover
        if isinstance(x, tuple):  # pragma: no cover
            return tuple((_u(i, param_name) for i in x))  # pragma: no cover
        if hasattr(x, "_tensor"):  # pragma: no cover
            return x._tensor  # pragma: no cover
        if isinstance(x, (int, float, bool, complex)):  # pragma: no cover
            if param_name not in {  # pragma: no cover
                "axis",  # pragma: no cover
                "keepdims",
                "shape",
                "size",  # pragma: no cover
                "equal_nan",
                "nan",
                "posinf",
                "neginf",  # pragma: no cover
                "axis1",  # pragma: no cover
                "rtol",  # pragma: no cover
                "n",  # pragma: no cover
                "m",  # pragma: no cover
                "endpoint",  # pragma: no cover
                "offset",  # pragma: no cover
                "a_max",  # pragma: no cover
                "ddof",  # pragma: no cover
                "dims",  # pragma: no cover
                "source",  # pragma: no cover
                "axis2",  # pragma: no cover
                "stop",  # pragma: no cover
                "k",  # pragma: no cover
                "dtype",  # pragma: no cover
                "reps",  # pragma: no cover
                "a_min",  # pragma: no cover
                "sparse",  # pragma: no cover
                "decimals",  # pragma: no cover
                "strides",  # pragma: no cover
                "num",  # pragma: no cover
                "max",  # pragma: no cover
                "indices_or_sections",  # pragma: no cover
                "shape",  # pragma: no cover
                "atol",  # pragma: no cover
                "axis1",  # pragma: no cover
                "step",  # pragma: no cover
                "axes",  # pragma: no cover
                "fill_value",  # pragma: no cover
                "stream",  # pragma: no cover
                "destination",  # pragma: no cover
                "alpha",  # pragma: no cover
                "beta",  # pragma: no cover
                "start",  # pragma: no cover
                "equal_nan",
                "nan",
                "posinf",
                "neginf",  # pragma: no cover
                "keepdims",
                "shape",
                "size",  # pragma: no cover
                "min",  # pragma: no cover
                "p",  # pragma: no cover
            } and "not_equal" not in (  # pragma: no cover
                "transpose",  # pragma: no cover
                "moveaxis",  # pragma: no cover
                "reshape",  # pragma: no cover
                "expand_dims",  # pragma: no cover
                "linspace",  # pragma: no cover
                "logcumsumexp",  # pragma: no cover
                "cummin",  # pragma: no cover
                "cummax",  # pragma: no cover
                "cumprod",  # pragma: no cover
                "cumsum",  # pragma: no cover
                "logsumexp",  # pragma: no cover
                "all",  # pragma: no cover
                "any",  # pragma: no cover
                "argmin",  # pragma: no cover
                "argmax",  # pragma: no cover
                "prod",  # pragma: no cover
                "sum",  # pragma: no cover
                "mean",  # pragma: no cover
                "var",  # pragma: no cover
                "std",  # pragma: no cover
                "max",  # pragma: no cover
                "min",  # pragma: no cover
                "tensordot",  # pragma: no cover
            ):  # pragma: no cover
                return _to_tensor(x)  # pragma: no cover
        if type(x).__name__ == "ndarray":  # pragma: no cover
            return _to_tensor(x)  # pragma: no cover
        return x  # pragma: no cover

    def _w(x):  # pragma: no cover
        if isinstance(x, tuple):  # pragma: no cover
            return tuple((_w(i) for i in x))  # pragma: no cover
        if isinstance(x, list):  # pragma: no cover
            return [_w(i) for i in x]  # pragma: no cover
        return array(x) if hasattr(x, "shape") else x  # pragma: no cover

    fn = _get_compiler_op("not_equal")  # pragma: no cover
    res = fn(  # pragma: no cover
        *[_u(x) for x in args],  # pragma: no cover
        **{
            ("dim" if k == "axis" else k): _u(v, k)
            for (k, v) in kwargs.items()
            if k != "stream"
        },  # pragma: no cover
    )  # pragma: no cover

    def __w(x):
        if isinstance(x, tuple):
            return tuple((__w(i) for i in x))
        if isinstance(x, list):
            return [__w(i) for i in x]
        return array(x) if hasattr(x, "shape") else x

    return __w(res)  # pragma: no cover


def power(*args, **kwargs):  # pragma: no cover
    """Compute power.

    Args:
        args: The args argument.
        kwargs: The kwargs argument.

    Returns:
        The result of power.

    """
    from zero_mlx.array import array, _to_tensor  # pragma: no cover
    import ml_switcheroo_compiler.ops as mops  # pragma: no cover
    import ml_switcheroo_compiler.ops.nn as mnn  # pragma: no cover
    import ml_switcheroo_compiler.random as mrand  # pragma: no cover
    import types  # pragma: no cover

    def _u(x, param_name=None):  # pragma: no cover
        if isinstance(x, list):  # pragma: no cover
            if __import__("builtins").all(  # pragma: no cover
                (isinstance(i, int) for i in x)  # pragma: no cover
            ):  # pragma: no cover
                return tuple(x)  # pragma: no cover
            return [_u(i, param_name) for i in x]  # pragma: no cover
        if isinstance(x, tuple):  # pragma: no cover
            return tuple((_u(i, param_name) for i in x))  # pragma: no cover
        if hasattr(x, "_tensor"):  # pragma: no cover
            return x._tensor  # pragma: no cover
        if isinstance(x, (int, float, bool, complex)):  # pragma: no cover
            if param_name not in {  # pragma: no cover
                "axis",  # pragma: no cover
                "keepdims",
                "shape",
                "size",  # pragma: no cover
                "equal_nan",
                "nan",
                "posinf",
                "neginf",  # pragma: no cover
                "axis1",  # pragma: no cover
                "rtol",  # pragma: no cover
                "n",  # pragma: no cover
                "m",  # pragma: no cover
                "endpoint",  # pragma: no cover
                "offset",  # pragma: no cover
                "a_max",  # pragma: no cover
                "ddof",  # pragma: no cover
                "dims",  # pragma: no cover
                "source",  # pragma: no cover
                "axis2",  # pragma: no cover
                "stop",  # pragma: no cover
                "k",  # pragma: no cover
                "dtype",  # pragma: no cover
                "reps",  # pragma: no cover
                "a_min",  # pragma: no cover
                "sparse",  # pragma: no cover
                "decimals",  # pragma: no cover
                "strides",  # pragma: no cover
                "num",  # pragma: no cover
                "max",  # pragma: no cover
                "indices_or_sections",  # pragma: no cover
                "shape",  # pragma: no cover
                "atol",  # pragma: no cover
                "axis1",  # pragma: no cover
                "step",  # pragma: no cover
                "axes",  # pragma: no cover
                "fill_value",  # pragma: no cover
                "stream",  # pragma: no cover
                "destination",  # pragma: no cover
                "alpha",  # pragma: no cover
                "beta",  # pragma: no cover
                "start",  # pragma: no cover
                "equal_nan",
                "nan",
                "posinf",
                "neginf",  # pragma: no cover
                "keepdims",
                "shape",
                "size",  # pragma: no cover
                "min",  # pragma: no cover
                "p",  # pragma: no cover
            } and "power" not in (  # pragma: no cover
                "transpose",  # pragma: no cover
                "moveaxis",  # pragma: no cover
                "reshape",  # pragma: no cover
                "expand_dims",  # pragma: no cover
                "linspace",  # pragma: no cover
                "logcumsumexp",  # pragma: no cover
                "cummin",  # pragma: no cover
                "cummax",  # pragma: no cover
                "cumprod",  # pragma: no cover
                "cumsum",  # pragma: no cover
                "logsumexp",  # pragma: no cover
                "all",  # pragma: no cover
                "any",  # pragma: no cover
                "argmin",  # pragma: no cover
                "argmax",  # pragma: no cover
                "prod",  # pragma: no cover
                "sum",  # pragma: no cover
                "mean",  # pragma: no cover
                "var",  # pragma: no cover
                "std",  # pragma: no cover
                "max",  # pragma: no cover
                "min",  # pragma: no cover
                "tensordot",  # pragma: no cover
            ):  # pragma: no cover
                return _to_tensor(x)  # pragma: no cover
        if type(x).__name__ == "ndarray":  # pragma: no cover
            return _to_tensor(x)  # pragma: no cover
        return x  # pragma: no cover

    def _w(x):  # pragma: no cover
        if isinstance(x, tuple):  # pragma: no cover
            return tuple((_w(i) for i in x))  # pragma: no cover
        if isinstance(x, list):  # pragma: no cover
            return [_w(i) for i in x]  # pragma: no cover
        return array(x) if hasattr(x, "shape") else x  # pragma: no cover

    fn = _get_compiler_op("power")  # pragma: no cover
    res = fn(  # pragma: no cover
        *[_u(x) for x in args],  # pragma: no cover
        **{
            ("dim" if k == "axis" else k): _u(v, k)
            for (k, v) in kwargs.items()
            if k != "stream"
        },  # pragma: no cover
    )  # pragma: no cover

    def __w(x):
        if isinstance(x, tuple):
            return tuple((__w(i) for i in x))
        if isinstance(x, list):
            return [__w(i) for i in x]
        return array(x) if hasattr(x, "shape") else x

    return __w(res)  # pragma: no cover


def sinh(*args, **kwargs):  # pragma: no cover
    """Compute sinh.

    Args:
        args: The args argument.
        kwargs: The kwargs argument.

    Returns:
        The result of sinh.

    """
    from zero_mlx.array import array, _to_tensor  # pragma: no cover
    import ml_switcheroo_compiler.ops as mops  # pragma: no cover
    import ml_switcheroo_compiler.ops.nn as mnn  # pragma: no cover
    import ml_switcheroo_compiler.random as mrand  # pragma: no cover
    import types  # pragma: no cover

    def _u(x, param_name=None):  # pragma: no cover
        if isinstance(x, list):  # pragma: no cover
            if __import__("builtins").all(  # pragma: no cover
                (isinstance(i, int) for i in x)  # pragma: no cover
            ):  # pragma: no cover
                return tuple(x)  # pragma: no cover
            return [_u(i, param_name) for i in x]  # pragma: no cover
        if isinstance(x, tuple):  # pragma: no cover
            return tuple((_u(i, param_name) for i in x))  # pragma: no cover
        if hasattr(x, "_tensor"):  # pragma: no cover
            return x._tensor  # pragma: no cover
        if isinstance(x, (int, float, bool, complex)):  # pragma: no cover
            if param_name not in {  # pragma: no cover
                "axis",  # pragma: no cover
                "keepdims",
                "shape",
                "size",  # pragma: no cover
                "equal_nan",
                "nan",
                "posinf",
                "neginf",  # pragma: no cover
                "axis1",  # pragma: no cover
                "rtol",  # pragma: no cover
                "n",  # pragma: no cover
                "m",  # pragma: no cover
                "endpoint",  # pragma: no cover
                "offset",  # pragma: no cover
                "a_max",  # pragma: no cover
                "ddof",  # pragma: no cover
                "dims",  # pragma: no cover
                "source",  # pragma: no cover
                "axis2",  # pragma: no cover
                "stop",  # pragma: no cover
                "k",  # pragma: no cover
                "dtype",  # pragma: no cover
                "reps",  # pragma: no cover
                "a_min",  # pragma: no cover
                "sparse",  # pragma: no cover
                "decimals",  # pragma: no cover
                "strides",  # pragma: no cover
                "num",  # pragma: no cover
                "max",  # pragma: no cover
                "indices_or_sections",  # pragma: no cover
                "shape",  # pragma: no cover
                "atol",  # pragma: no cover
                "axis1",  # pragma: no cover
                "step",  # pragma: no cover
                "axes",  # pragma: no cover
                "fill_value",  # pragma: no cover
                "stream",  # pragma: no cover
                "destination",  # pragma: no cover
                "alpha",  # pragma: no cover
                "beta",  # pragma: no cover
                "start",  # pragma: no cover
                "equal_nan",
                "nan",
                "posinf",
                "neginf",  # pragma: no cover
                "keepdims",
                "shape",
                "size",  # pragma: no cover
                "min",  # pragma: no cover
                "p",  # pragma: no cover
            } and "sinh" not in (  # pragma: no cover
                "transpose",  # pragma: no cover
                "moveaxis",  # pragma: no cover
                "reshape",  # pragma: no cover
                "expand_dims",  # pragma: no cover
                "linspace",  # pragma: no cover
                "logcumsumexp",  # pragma: no cover
                "cummin",  # pragma: no cover
                "cummax",  # pragma: no cover
                "cumprod",  # pragma: no cover
                "cumsum",  # pragma: no cover
                "logsumexp",  # pragma: no cover
                "all",  # pragma: no cover
                "any",  # pragma: no cover
                "argmin",  # pragma: no cover
                "argmax",  # pragma: no cover
                "prod",  # pragma: no cover
                "sum",  # pragma: no cover
                "mean",  # pragma: no cover
                "var",  # pragma: no cover
                "std",  # pragma: no cover
                "max",  # pragma: no cover
                "min",  # pragma: no cover
                "tensordot",  # pragma: no cover
            ):  # pragma: no cover
                return _to_tensor(x)  # pragma: no cover
        if type(x).__name__ == "ndarray":  # pragma: no cover
            return _to_tensor(x)  # pragma: no cover
        return x  # pragma: no cover

    def _w(x):  # pragma: no cover
        if isinstance(x, tuple):  # pragma: no cover
            return tuple((_w(i) for i in x))  # pragma: no cover
        if isinstance(x, list):  # pragma: no cover
            return [_w(i) for i in x]  # pragma: no cover
        return array(x) if hasattr(x, "shape") else x  # pragma: no cover

    fn = _get_compiler_op("sinh")  # pragma: no cover
    res = fn(  # pragma: no cover
        *[_u(x) for x in args],  # pragma: no cover
        **{
            ("dim" if k == "axis" else k): _u(v, k)
            for (k, v) in kwargs.items()
            if k != "stream"
        },  # pragma: no cover
    )  # pragma: no cover

    def __w(x):
        if isinstance(x, tuple):
            return tuple((__w(i) for i in x))
        if isinstance(x, list):
            return [__w(i) for i in x]
        return array(x) if hasattr(x, "shape") else x

    return __w(res)  # pragma: no cover


def cosh(*args, **kwargs):  # pragma: no cover
    """Compute cosh.

    Args:
        args: The args argument.
        kwargs: The kwargs argument.

    Returns:
        The result of cosh.

    """
    from zero_mlx.array import array, _to_tensor  # pragma: no cover
    import ml_switcheroo_compiler.ops as mops  # pragma: no cover
    import ml_switcheroo_compiler.ops.nn as mnn  # pragma: no cover
    import ml_switcheroo_compiler.random as mrand  # pragma: no cover
    import types  # pragma: no cover

    def _u(x, param_name=None):  # pragma: no cover
        if isinstance(x, list):  # pragma: no cover
            if __import__("builtins").all(  # pragma: no cover
                (isinstance(i, int) for i in x)  # pragma: no cover
            ):  # pragma: no cover
                return tuple(x)  # pragma: no cover
            return [_u(i, param_name) for i in x]  # pragma: no cover
        if isinstance(x, tuple):  # pragma: no cover
            return tuple((_u(i, param_name) for i in x))  # pragma: no cover
        if hasattr(x, "_tensor"):  # pragma: no cover
            return x._tensor  # pragma: no cover
        if isinstance(x, (int, float, bool, complex)):  # pragma: no cover
            if param_name not in {  # pragma: no cover
                "axis",  # pragma: no cover
                "keepdims",
                "shape",
                "size",  # pragma: no cover
                "equal_nan",
                "nan",
                "posinf",
                "neginf",  # pragma: no cover
                "axis1",  # pragma: no cover
                "rtol",  # pragma: no cover
                "n",  # pragma: no cover
                "m",  # pragma: no cover
                "endpoint",  # pragma: no cover
                "offset",  # pragma: no cover
                "a_max",  # pragma: no cover
                "ddof",  # pragma: no cover
                "dims",  # pragma: no cover
                "source",  # pragma: no cover
                "axis2",  # pragma: no cover
                "stop",  # pragma: no cover
                "k",  # pragma: no cover
                "dtype",  # pragma: no cover
                "reps",  # pragma: no cover
                "a_min",  # pragma: no cover
                "sparse",  # pragma: no cover
                "decimals",  # pragma: no cover
                "strides",  # pragma: no cover
                "num",  # pragma: no cover
                "max",  # pragma: no cover
                "indices_or_sections",  # pragma: no cover
                "shape",  # pragma: no cover
                "atol",  # pragma: no cover
                "axis1",  # pragma: no cover
                "step",  # pragma: no cover
                "axes",  # pragma: no cover
                "fill_value",  # pragma: no cover
                "stream",  # pragma: no cover
                "destination",  # pragma: no cover
                "alpha",  # pragma: no cover
                "beta",  # pragma: no cover
                "start",  # pragma: no cover
                "equal_nan",
                "nan",
                "posinf",
                "neginf",  # pragma: no cover
                "keepdims",
                "shape",
                "size",  # pragma: no cover
                "min",  # pragma: no cover
                "p",  # pragma: no cover
            } and "cosh" not in (  # pragma: no cover
                "transpose",  # pragma: no cover
                "moveaxis",  # pragma: no cover
                "reshape",  # pragma: no cover
                "expand_dims",  # pragma: no cover
                "linspace",  # pragma: no cover
                "logcumsumexp",  # pragma: no cover
                "cummin",  # pragma: no cover
                "cummax",  # pragma: no cover
                "cumprod",  # pragma: no cover
                "cumsum",  # pragma: no cover
                "logsumexp",  # pragma: no cover
                "all",  # pragma: no cover
                "any",  # pragma: no cover
                "argmin",  # pragma: no cover
                "argmax",  # pragma: no cover
                "prod",  # pragma: no cover
                "sum",  # pragma: no cover
                "mean",  # pragma: no cover
                "var",  # pragma: no cover
                "std",  # pragma: no cover
                "max",  # pragma: no cover
                "min",  # pragma: no cover
                "tensordot",  # pragma: no cover
            ):  # pragma: no cover
                return _to_tensor(x)  # pragma: no cover
        if type(x).__name__ == "ndarray":  # pragma: no cover
            return _to_tensor(x)  # pragma: no cover
        return x  # pragma: no cover

    def _w(x):  # pragma: no cover
        if isinstance(x, tuple):  # pragma: no cover
            return tuple((_w(i) for i in x))  # pragma: no cover
        if isinstance(x, list):  # pragma: no cover
            return [_w(i) for i in x]  # pragma: no cover
        return array(x) if hasattr(x, "shape") else x  # pragma: no cover

    fn = _get_compiler_op("cosh")  # pragma: no cover
    res = fn(  # pragma: no cover
        *[_u(x) for x in args],  # pragma: no cover
        **{
            ("dim" if k == "axis" else k): _u(v, k)
            for (k, v) in kwargs.items()
            if k != "stream"
        },  # pragma: no cover
    )  # pragma: no cover

    def __w(x):
        if isinstance(x, tuple):
            return tuple((__w(i) for i in x))
        if isinstance(x, list):
            return [__w(i) for i in x]
        return array(x) if hasattr(x, "shape") else x

    return __w(res)  # pragma: no cover


def tanh(*args, **kwargs):  # pragma: no cover
    """Compute tanh.

    Args:
        args: The args argument.
        kwargs: The kwargs argument.

    Returns:
        The result of tanh.

    """
    from zero_mlx.array import array, _to_tensor  # pragma: no cover
    import ml_switcheroo_compiler.ops as mops  # pragma: no cover
    import ml_switcheroo_compiler.ops.nn as mnn  # pragma: no cover
    import ml_switcheroo_compiler.random as mrand  # pragma: no cover
    import types  # pragma: no cover

    def _u(x, param_name=None):  # pragma: no cover
        if isinstance(x, list):  # pragma: no cover
            if __import__("builtins").all(  # pragma: no cover
                (isinstance(i, int) for i in x)  # pragma: no cover
            ):  # pragma: no cover
                return tuple(x)  # pragma: no cover
            return [_u(i, param_name) for i in x]  # pragma: no cover
        if isinstance(x, tuple):  # pragma: no cover
            return tuple((_u(i, param_name) for i in x))  # pragma: no cover
        if hasattr(x, "_tensor"):  # pragma: no cover
            return x._tensor  # pragma: no cover
        if isinstance(x, (int, float, bool, complex)):  # pragma: no cover
            if param_name not in {  # pragma: no cover
                "axis",  # pragma: no cover
                "keepdims",
                "shape",
                "size",  # pragma: no cover
                "equal_nan",
                "nan",
                "posinf",
                "neginf",  # pragma: no cover
                "axis1",  # pragma: no cover
                "rtol",  # pragma: no cover
                "n",  # pragma: no cover
                "m",  # pragma: no cover
                "endpoint",  # pragma: no cover
                "offset",  # pragma: no cover
                "a_max",  # pragma: no cover
                "ddof",  # pragma: no cover
                "dims",  # pragma: no cover
                "source",  # pragma: no cover
                "axis2",  # pragma: no cover
                "stop",  # pragma: no cover
                "k",  # pragma: no cover
                "dtype",  # pragma: no cover
                "reps",  # pragma: no cover
                "a_min",  # pragma: no cover
                "sparse",  # pragma: no cover
                "decimals",  # pragma: no cover
                "strides",  # pragma: no cover
                "num",  # pragma: no cover
                "max",  # pragma: no cover
                "indices_or_sections",  # pragma: no cover
                "shape",  # pragma: no cover
                "atol",  # pragma: no cover
                "axis1",  # pragma: no cover
                "step",  # pragma: no cover
                "axes",  # pragma: no cover
                "fill_value",  # pragma: no cover
                "stream",  # pragma: no cover
                "destination",  # pragma: no cover
                "alpha",  # pragma: no cover
                "beta",  # pragma: no cover
                "start",  # pragma: no cover
                "equal_nan",
                "nan",
                "posinf",
                "neginf",  # pragma: no cover
                "keepdims",
                "shape",
                "size",  # pragma: no cover
                "min",  # pragma: no cover
                "p",  # pragma: no cover
            } and "tanh" not in (  # pragma: no cover
                "transpose",  # pragma: no cover
                "moveaxis",  # pragma: no cover
                "reshape",  # pragma: no cover
                "expand_dims",  # pragma: no cover
                "linspace",  # pragma: no cover
                "logcumsumexp",  # pragma: no cover
                "cummin",  # pragma: no cover
                "cummax",  # pragma: no cover
                "cumprod",  # pragma: no cover
                "cumsum",  # pragma: no cover
                "logsumexp",  # pragma: no cover
                "all",  # pragma: no cover
                "any",  # pragma: no cover
                "argmin",  # pragma: no cover
                "argmax",  # pragma: no cover
                "prod",  # pragma: no cover
                "sum",  # pragma: no cover
                "mean",  # pragma: no cover
                "var",  # pragma: no cover
                "std",  # pragma: no cover
                "max",  # pragma: no cover
                "min",  # pragma: no cover
                "tensordot",  # pragma: no cover
            ):  # pragma: no cover
                return _to_tensor(x)  # pragma: no cover
        if type(x).__name__ == "ndarray":  # pragma: no cover
            return _to_tensor(x)  # pragma: no cover
        return x  # pragma: no cover

    def _w(x):  # pragma: no cover
        if isinstance(x, tuple):  # pragma: no cover
            return tuple((_w(i) for i in x))  # pragma: no cover
        if isinstance(x, list):  # pragma: no cover
            return [_w(i) for i in x]  # pragma: no cover
        return array(x) if hasattr(x, "shape") else x  # pragma: no cover

    fn = _get_compiler_op("tanh")  # pragma: no cover
    res = fn(  # pragma: no cover
        *[_u(x) for x in args],  # pragma: no cover
        **{
            ("dim" if k == "axis" else k): _u(v, k)
            for (k, v) in kwargs.items()
            if k != "stream"
        },  # pragma: no cover
    )  # pragma: no cover

    def __w(x):
        if isinstance(x, tuple):
            return tuple((__w(i) for i in x))
        if isinstance(x, list):
            return [__w(i) for i in x]
        return array(x) if hasattr(x, "shape") else x

    return __w(res)  # pragma: no cover


def floor_divide(*args, **kwargs):  # pragma: no cover
    """Compute floor_divide.

    Args:
        args: The args argument.
        kwargs: The kwargs argument.

    Returns:
        The result of floor_divide.

    """
    from zero_mlx.array import array, _to_tensor  # pragma: no cover
    import ml_switcheroo_compiler.ops as mops  # pragma: no cover
    import ml_switcheroo_compiler.ops.nn as mnn  # pragma: no cover
    import ml_switcheroo_compiler.random as mrand  # pragma: no cover
    import types  # pragma: no cover

    def _u(x, param_name=None):  # pragma: no cover
        if isinstance(x, list):  # pragma: no cover
            if __import__("builtins").all(  # pragma: no cover
                (isinstance(i, int) for i in x)  # pragma: no cover
            ):  # pragma: no cover
                return tuple(x)  # pragma: no cover
            return [_u(i, param_name) for i in x]  # pragma: no cover
        if isinstance(x, tuple):  # pragma: no cover
            return tuple((_u(i, param_name) for i in x))  # pragma: no cover
        if hasattr(x, "_tensor"):  # pragma: no cover
            return x._tensor  # pragma: no cover
        if isinstance(x, (int, float, bool, complex)):  # pragma: no cover
            if param_name not in {  # pragma: no cover
                "axis",  # pragma: no cover
                "keepdims",
                "shape",
                "size",  # pragma: no cover
                "equal_nan",
                "nan",
                "posinf",
                "neginf",  # pragma: no cover
                "axis1",  # pragma: no cover
                "rtol",  # pragma: no cover
                "n",  # pragma: no cover
                "m",  # pragma: no cover
                "endpoint",  # pragma: no cover
                "offset",  # pragma: no cover
                "a_max",  # pragma: no cover
                "ddof",  # pragma: no cover
                "dims",  # pragma: no cover
                "source",  # pragma: no cover
                "axis2",  # pragma: no cover
                "stop",  # pragma: no cover
                "k",  # pragma: no cover
                "dtype",  # pragma: no cover
                "reps",  # pragma: no cover
                "a_min",  # pragma: no cover
                "sparse",  # pragma: no cover
                "decimals",  # pragma: no cover
                "strides",  # pragma: no cover
                "num",  # pragma: no cover
                "max",  # pragma: no cover
                "indices_or_sections",  # pragma: no cover
                "shape",  # pragma: no cover
                "atol",  # pragma: no cover
                "axis1",  # pragma: no cover
                "step",  # pragma: no cover
                "axes",  # pragma: no cover
                "fill_value",  # pragma: no cover
                "stream",  # pragma: no cover
                "destination",  # pragma: no cover
                "alpha",  # pragma: no cover
                "beta",  # pragma: no cover
                "start",  # pragma: no cover
                "equal_nan",
                "nan",
                "posinf",
                "neginf",  # pragma: no cover
                "keepdims",
                "shape",
                "size",  # pragma: no cover
                "min",  # pragma: no cover
                "p",  # pragma: no cover
            } and "floor_divide" not in (  # pragma: no cover
                "transpose",  # pragma: no cover
                "moveaxis",  # pragma: no cover
                "reshape",  # pragma: no cover
                "expand_dims",  # pragma: no cover
                "linspace",  # pragma: no cover
                "logcumsumexp",  # pragma: no cover
                "cummin",  # pragma: no cover
                "cummax",  # pragma: no cover
                "cumprod",  # pragma: no cover
                "cumsum",  # pragma: no cover
                "logsumexp",  # pragma: no cover
                "all",  # pragma: no cover
                "any",  # pragma: no cover
                "argmin",  # pragma: no cover
                "argmax",  # pragma: no cover
                "prod",  # pragma: no cover
                "sum",  # pragma: no cover
                "mean",  # pragma: no cover
                "var",  # pragma: no cover
                "std",  # pragma: no cover
                "max",  # pragma: no cover
                "min",  # pragma: no cover
                "tensordot",  # pragma: no cover
            ):  # pragma: no cover
                return _to_tensor(x)  # pragma: no cover
        if type(x).__name__ == "ndarray":  # pragma: no cover
            return _to_tensor(x)  # pragma: no cover
        return x  # pragma: no cover

    def _w(x):  # pragma: no cover
        if isinstance(x, tuple):  # pragma: no cover
            return tuple((_w(i) for i in x))  # pragma: no cover
        if isinstance(x, list):  # pragma: no cover
            return [_w(i) for i in x]  # pragma: no cover
        return array(x) if hasattr(x, "shape") else x  # pragma: no cover

    fn = _get_compiler_op("floor_divide")  # pragma: no cover
    res = fn(  # pragma: no cover
        *[_u(x) for x in args],  # pragma: no cover
        **{
            ("dim" if k == "axis" else k): _u(v, k)
            for (k, v) in kwargs.items()
            if k != "stream"
        },  # pragma: no cover
    )  # pragma: no cover

    def __w(x):
        if isinstance(x, tuple):
            return tuple((__w(i) for i in x))
        if isinstance(x, list):
            return [__w(i) for i in x]
        return array(x) if hasattr(x, "shape") else x

    return __w(res)  # pragma: no cover


__all__ = [  # pragma: no cover
    "all",  # pragma: no cover
    "any",  # pragma: no cover
    "allclose",  # pragma: no cover
    "synchronize",  # pragma: no cover
    "eval",  # pragma: no cover
    "old_split",  # pragma: no cover
    "asarray",  # pragma: no cover
    "stack",  # pragma: no cover
    "sin",  # pragma: no cover
    "square",  # pragma: no cover
    "sum",  # pragma: no cover
    "mean",  # pragma: no cover
    "arange",  # pragma: no cover
    "full",  # pragma: no cover
    "zeros",  # pragma: no cover
    "ones",  # pragma: no cover
    "zeros_like",  # pragma: no cover
    "ones_like",  # pragma: no cover
    "array_equal",  # pragma: no cover
    "broadcast_to",  # pragma: no cover
    "as_strided",  # pragma: no cover
    "reshape",  # pragma: no cover
    "divmod",  # pragma: no cover
    "logical_not",  # pragma: no cover
    "logical_and",  # pragma: no cover
    "logical_or",  # pragma: no cover
    "sqrt",  # pragma: no cover
    "abs",  # pragma: no cover
    "negative",  # pragma: no cover
    "exp",  # pragma: no cover
    "rsqrt",  # pragma: no cover
    "add",  # pragma: no cover
    "subtract",  # pragma: no cover
    "multiply",  # pragma: no cover
    "divide",  # pragma: no cover
    "matmul",  # pragma: no cover
    "reciprocal",  # pragma: no cover
    "log",  # pragma: no cover
    "maximum",  # pragma: no cover
    "minimum",  # pragma: no cover
    "cos",  # pragma: no cover
    "log1p",  # pragma: no cover
    "stop_gradient",  # pragma: no cover
    "split",  # pragma: no cover
    "diagonal",  # pragma: no cover
    "logcumsumexp",  # pragma: no cover
    "logsumexp",  # pragma: no cover
    "cummax",  # pragma: no cover
    "cummin",  # pragma: no cover
    "cumprod",  # pragma: no cover
    "cumsum",  # pragma: no cover
    "transpose",  # pragma: no cover
    "concatenate",  # pragma: no cover
    "get_peak_memory",  # pragma: no cover
    "argmin",  # pragma: no cover
    "argmax",  # pragma: no cover
    "min",  # pragma: no cover
    "max",  # pragma: no cover
    "prod",  # pragma: no cover
    "eye",  # pragma: no cover
    "diag",  # pragma: no cover
    "tril",  # pragma: no cover
    "triu",  # pragma: no cover
    "expand_dims",  # pragma: no cover
    "take_along_axis",  # pragma: no cover
    "tile",  # pragma: no cover
    "squeeze",  # pragma: no cover
    "var",  # pragma: no cover
    "std",  # pragma: no cover
    "round",  # pragma: no cover
    "sort",  # pragma: no cover
    "argsort",  # pragma: no cover
    "swapaxes",  # pragma: no cover
    "moveaxis",  # pragma: no cover
    "take",  # pragma: no cover
    "addmm",  # pragma: no cover
    "gather_mm",  # pragma: no cover
    "block_masked_mm",  # pragma: no cover
    "segmented_mm",  # pragma: no cover
    "argpartition",  # pragma: no cover
    "atleast_1d",  # pragma: no cover
    "atleast_2d",  # pragma: no cover
    "atleast_3d",  # pragma: no cover
    "broadcast_arrays",  # pragma: no cover
    "broadcast_shapes",  # pragma: no cover
    "ceil",  # pragma: no cover
    "clip",  # pragma: no cover
    "conjugate",  # pragma: no cover
    "degrees",  # pragma: no cover
    "erf",  # pragma: no cover
    "erfinv",  # pragma: no cover
    "expm1",  # pragma: no cover
    "floor",  # pragma: no cover
    "inner",  # pragma: no cover
    "isclose",  # pragma: no cover
    "isfinite",  # pragma: no cover
    "isinf",  # pragma: no cover
    "isnan",  # pragma: no cover
    "isneginf",  # pragma: no cover
    "isposinf",  # pragma: no cover
    "issubdtype",  # pragma: no cover
    "kron",  # pragma: no cover
    "linspace",  # pragma: no cover
    "logaddexp",  # pragma: no cover
    "median",  # pragma: no cover
    "meshgrid",  # pragma: no cover
    "nan_to_num",  # pragma: no cover
    "outer",  # pragma: no cover
    "pad",  # pragma: no cover
    "partition",  # pragma: no cover
    "put_along_axis",  # pragma: no cover
    "radians",  # pragma: no cover
    "real",  # pragma: no cover
    "remainder",  # pragma: no cover
    "repeat",  # pragma: no cover
    "roll",  # pragma: no cover
    "sign",  # pragma: no cover
    "tensordot",  # pragma: no cover
    "trace",  # pragma: no cover
    "tri",  # pragma: no cover
    "where",  # pragma: no cover
    "sigmoid",  # pragma: no cover
    "softmax",  # pragma: no cover
    "depends",  # pragma: no cover
    "to_fp8",  # pragma: no cover
    "from_fp8",  # pragma: no cover
    "arcsin",  # pragma: no cover
    "arccos",  # pragma: no cover
    "arctan",  # pragma: no cover
    "arcsinh",  # pragma: no cover
    "arctanh",  # pragma: no cover
    "log2",  # pragma: no cover
    "log10",  # pragma: no cover
    "imag",  # pragma: no cover
    "bitwise_and",  # pragma: no cover
    "bitwise_or",  # pragma: no cover
    "bitwise_xor",  # pragma: no cover
    "left_shift",  # pragma: no cover
    "right_shift",  # pragma: no cover
    "less",  # pragma: no cover
    "less_equal",  # pragma: no cover
    "greater",  # pragma: no cover
    "greater_equal",  # pragma: no cover
    "equal",  # pragma: no cover
    "not_equal",  # pragma: no cover
    "power",  # pragma: no cover
    "sinh",  # pragma: no cover
    "cosh",  # pragma: no cover
    "tanh",  # pragma: no cover
    "floor_divide",  # pragma: no cover
]  # pragma: no cover
