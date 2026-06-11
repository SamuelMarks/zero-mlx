"""Core tensor operations."""

from typing import Any, Tuple, Optional, Sequence, Union
from zero_mlx.dtypes import DType
import ml_switcheroo
import zero_mlx as mx


def _wrap(x: Any, dtype: Optional[DType] = None) -> Any:
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


def all(a: Any, axis: Any = None, keepdims: bool = False, stream: Any = None) -> Any:
    """Compute all.

    Args:
        a: The a argument.
        axis: The axis argument.
        keepdims: The keepdims argument.
        stream: The stream argument.

    Returns:
        The result of all.

    """
    from zero_mlx.array import array, _to_tensor
    import ml_switcheroo.ops as mops
    import ml_switcheroo.nn as mnn
    import ml_switcheroo.random as mrand
    import types

    def _u(x, param_name=None):
        if isinstance(x, list):  # pragma: no cover
            if __import__("builtins").all(
                (isinstance(i, int) for i in x)
            ):  # pragma: no cover
                return tuple(x)  # pragma: no cover
            return [_u(i, param_name) for i in x]  # pragma: no cover
        if isinstance(x, tuple):  # pragma: no cover
            return tuple((_u(i, param_name) for i in x))  # pragma: no cover
        if hasattr(x, "_tensor"):  # pragma: no cover
            return x._tensor  # pragma: no cover
        if isinstance(x, (int, float, bool, complex)):  # pragma: no cover
            if param_name not in {
                "axis",
                "rtol",
                "n",
                "m",
                "endpoint",
                "offset",
                "a_max",
                "ddof",
                "dims",
                "source",
                "axis2",
                "stop",
                "k",
                "dtype",
                "reps",
                "a_min",
                "sparse",
                "decimals",
                "strides",
                "num",
                "max",
                "indices_or_sections",
                "shape",
                "atol",
                "axis1",
                "step",
                "axes",
                "fill_value",
                "stream",
                "destination",
                "alpha",
                "beta",
                "start",
                "equal_nan",
                "keepdims",
                "min",
                "p",
            } and "all" not in (
                "transpose",
                "moveaxis",
                "reshape",
                "expand_dims",
                "linspace",
                "logcumsumexp",
                "cummin",
                "cummax",
                "cumprod",
                "cumsum",
                "logsumexp",
                "all",
                "any",
                "argmin",
                "argmax",
                "prod",
                "sum",
                "mean",
                "var",
                "std",
                "max",
                "min",
                "tensordot",
            ):  # pragma: no cover
                return _to_tensor(x)  # pragma: no cover
        if type(x).__name__ == "ndarray":  # pragma: no cover
            return _to_tensor(x)  # pragma: no cover
        return x  # pragma: no cover

    def _w(x):
        if isinstance(x, tuple):  # pragma: no cover
            return tuple((_w(i) for i in x))  # pragma: no cover
        if isinstance(x, list):  # pragma: no cover
            return [_w(i) for i in x]  # pragma: no cover
        return array(x) if hasattr(x, "shape") else x  # pragma: no cover

    try:
        fn = getattr(mops, "all")
    except AttributeError:  # pragma: no cover
        try:
            fn = getattr(mnn, "all")
        except AttributeError:  # pragma: no cover
            try:
                fn = getattr(mrand, "all")
            except AttributeError:  # pragma: no cover  # pragma: no cover
                raise NotImplementedError("Missing in compiler: all")
    res = fn(_u(a, "a"), _u(axis, "axis"), _u(keepdims, "keepdims"))
    return _w(res)


def any(a: Any, axis: Any = None, keepdims: bool = False, stream: Any = None) -> Any:
    """Compute any.

    Args:
        a: The a argument.
        axis: The axis argument.
        keepdims: The keepdims argument.
        stream: The stream argument.

    Returns:
        The result of any.

    """
    from zero_mlx.array import array, _to_tensor
    import ml_switcheroo.ops as mops
    import ml_switcheroo.nn as mnn
    import ml_switcheroo.random as mrand
    import types

    def _u(x, param_name=None):
        if isinstance(x, list):  # pragma: no cover
            if __import__("builtins").all(
                (isinstance(i, int) for i in x)
            ):  # pragma: no cover
                return tuple(x)  # pragma: no cover
            return [_u(i, param_name) for i in x]  # pragma: no cover
        if isinstance(x, tuple):  # pragma: no cover
            return tuple((_u(i, param_name) for i in x))  # pragma: no cover
        if hasattr(x, "_tensor"):  # pragma: no cover
            return x._tensor  # pragma: no cover
        if isinstance(x, (int, float, bool, complex)):  # pragma: no cover
            if param_name not in {
                "axis",
                "rtol",
                "n",
                "m",
                "endpoint",
                "offset",
                "a_max",
                "ddof",
                "dims",
                "source",
                "axis2",
                "stop",
                "k",
                "dtype",
                "reps",
                "a_min",
                "sparse",
                "decimals",
                "strides",
                "num",
                "max",
                "indices_or_sections",
                "shape",
                "atol",
                "axis1",
                "step",
                "axes",
                "fill_value",
                "stream",
                "destination",
                "alpha",
                "beta",
                "start",
                "equal_nan",
                "keepdims",
                "min",
                "p",
            } and "any" not in (
                "transpose",
                "moveaxis",
                "reshape",
                "expand_dims",
                "linspace",
                "logcumsumexp",
                "cummin",
                "cummax",
                "cumprod",
                "cumsum",
                "logsumexp",
                "all",
                "any",
                "argmin",
                "argmax",
                "prod",
                "sum",
                "mean",
                "var",
                "std",
                "max",
                "min",
                "tensordot",
            ):  # pragma: no cover
                return _to_tensor(x)  # pragma: no cover
        if type(x).__name__ == "ndarray":  # pragma: no cover
            return _to_tensor(x)  # pragma: no cover
        return x  # pragma: no cover

    def _w(x):
        if isinstance(x, tuple):  # pragma: no cover
            return tuple((_w(i) for i in x))  # pragma: no cover
        if isinstance(x, list):  # pragma: no cover
            return [_w(i) for i in x]  # pragma: no cover
        return array(x) if hasattr(x, "shape") else x  # pragma: no cover

    try:
        fn = getattr(mops, "any")
    except AttributeError:  # pragma: no cover
        try:
            fn = getattr(mnn, "any")
        except AttributeError:  # pragma: no cover
            try:
                fn = getattr(mrand, "any")
            except AttributeError:  # pragma: no cover  # pragma: no cover
                raise NotImplementedError("Missing in compiler: any")
    res = fn(_u(a, "a"), _u(axis, "axis"), _u(keepdims, "keepdims"))
    return _w(res)


def allclose(
    a: Any,
    b: Any,
    rtol: float = 1e-05,
    atol: float = 1e-08,
    equal_nan: bool = False,
    stream: Any = None,
) -> Any:
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
    from zero_mlx.array import array, _to_tensor
    import ml_switcheroo.ops as mops
    import ml_switcheroo.nn as mnn
    import ml_switcheroo.random as mrand
    import types

    def _u(x, param_name=None):
        if isinstance(x, list):  # pragma: no cover
            if __import__("builtins").all(
                (isinstance(i, int) for i in x)
            ):  # pragma: no cover
                return tuple(x)  # pragma: no cover
            return [_u(i, param_name) for i in x]  # pragma: no cover
        if isinstance(x, tuple):  # pragma: no cover
            return tuple((_u(i, param_name) for i in x))  # pragma: no cover
        if hasattr(x, "_tensor"):  # pragma: no cover
            return x._tensor  # pragma: no cover
        if isinstance(x, (int, float, bool, complex)):  # pragma: no cover
            if param_name not in {
                "axis",
                "rtol",
                "n",
                "m",
                "endpoint",
                "offset",
                "a_max",
                "ddof",
                "dims",
                "source",
                "axis2",
                "stop",
                "k",
                "dtype",
                "reps",
                "a_min",
                "sparse",
                "decimals",
                "strides",
                "num",
                "max",
                "indices_or_sections",
                "shape",
                "atol",
                "axis1",
                "step",
                "axes",
                "fill_value",
                "stream",
                "destination",
                "alpha",
                "beta",
                "start",
                "equal_nan",
                "keepdims",
                "min",
                "p",
            } and "allclose" not in (
                "transpose",
                "moveaxis",
                "reshape",
                "expand_dims",
                "linspace",
                "logcumsumexp",
                "cummin",
                "cummax",
                "cumprod",
                "cumsum",
                "logsumexp",
                "all",
                "any",
                "argmin",
                "argmax",
                "prod",
                "sum",
                "mean",
                "var",
                "std",
                "max",
                "min",
                "tensordot",
            ):  # pragma: no cover
                return _to_tensor(x)  # pragma: no cover
        if type(x).__name__ == "ndarray":  # pragma: no cover
            return _to_tensor(x)  # pragma: no cover
        return x  # pragma: no cover

    def _w(x):
        if isinstance(x, tuple):  # pragma: no cover
            return tuple((_w(i) for i in x))  # pragma: no cover
        if isinstance(x, list):  # pragma: no cover
            return [_w(i) for i in x]  # pragma: no cover
        return array(x) if hasattr(x, "shape") else x  # pragma: no cover

    try:
        fn = getattr(mops, "allclose")
    except AttributeError:  # pragma: no cover
        try:
            fn = getattr(mnn, "allclose")
        except AttributeError:  # pragma: no cover
            try:
                fn = getattr(mrand, "allclose")
            except AttributeError:  # pragma: no cover  # pragma: no cover
                raise NotImplementedError("Missing in compiler: allclose")
    res = fn(
        _u(a, "a"),
        _u(b, "b"),
        _u(rtol, "rtol"),
        _u(atol, "atol"),
        _u(equal_nan, "equal_nan"),
    )
    return _w(res)


def synchronize(*args, **kwargs):
    """Compute synchronize.

    Args:
        args: The args argument.
        kwargs: The kwargs argument.

    Returns:
        The result of synchronize.

    """
    from zero_mlx.array import array, _to_tensor
    import ml_switcheroo.ops as mops
    import ml_switcheroo.nn as mnn
    import ml_switcheroo.random as mrand
    import types

    def _u(x, param_name=None):
        if isinstance(x, list):  # pragma: no cover
            if __import__("builtins").all(
                (isinstance(i, int) for i in x)
            ):  # pragma: no cover
                return tuple(x)  # pragma: no cover
            return [_u(i, param_name) for i in x]  # pragma: no cover
        if isinstance(x, tuple):  # pragma: no cover
            return tuple((_u(i, param_name) for i in x))  # pragma: no cover
        if hasattr(x, "_tensor"):  # pragma: no cover
            return x._tensor  # pragma: no cover
        if isinstance(x, (int, float, bool, complex)):  # pragma: no cover
            if param_name not in {
                "axis",
                "rtol",
                "n",
                "m",
                "endpoint",
                "offset",
                "a_max",
                "ddof",
                "dims",
                "source",
                "axis2",
                "stop",
                "k",
                "dtype",
                "reps",
                "a_min",
                "sparse",
                "decimals",
                "strides",
                "num",
                "max",
                "indices_or_sections",
                "shape",
                "atol",
                "axis1",
                "step",
                "axes",
                "fill_value",
                "stream",
                "destination",
                "alpha",
                "beta",
                "start",
                "equal_nan",
                "keepdims",
                "min",
                "p",
            } and "synchronize" not in (
                "transpose",
                "moveaxis",
                "reshape",
                "expand_dims",
                "linspace",
                "logcumsumexp",
                "cummin",
                "cummax",
                "cumprod",
                "cumsum",
                "logsumexp",
                "all",
                "any",
                "argmin",
                "argmax",
                "prod",
                "sum",
                "mean",
                "var",
                "std",
                "max",
                "min",
                "tensordot",
            ):  # pragma: no cover
                return _to_tensor(x)  # pragma: no cover
        if type(x).__name__ == "ndarray":  # pragma: no cover
            return _to_tensor(x)  # pragma: no cover
        return x  # pragma: no cover

    def _w(x):
        if isinstance(x, tuple):  # pragma: no cover
            return tuple((_w(i) for i in x))  # pragma: no cover
        if isinstance(x, list):  # pragma: no cover
            return [_w(i) for i in x]  # pragma: no cover
        return array(x) if hasattr(x, "shape") else x  # pragma: no cover

    try:
        fn = getattr(mops, "synchronize")
    except AttributeError:  # pragma: no cover
        try:
            fn = getattr(mnn, "synchronize")
        except AttributeError:  # pragma: no cover
            try:
                fn = getattr(mrand, "synchronize")
            except AttributeError:  # pragma: no cover  # pragma: no cover
                raise NotImplementedError("Missing in compiler: synchronize")
    res = fn(
        *[_u(x) for x in args],
        **{k: _u(v, k) for (k, v) in kwargs.items() if k != "stream"},
    )
    return _w(res)


def eval(*args, **kwargs):
    """Compute eval.

    Args:
        args: The args argument.
        kwargs: The kwargs argument.

    Returns:
        The result of eval.

    """
    from zero_mlx.array import array, _to_tensor  # pragma: no cover
    import ml_switcheroo.ops as mops  # pragma: no cover
    import ml_switcheroo.nn as mnn  # pragma: no cover
    import ml_switcheroo.random as mrand  # pragma: no cover
    import types  # pragma: no cover

    def _u(x, param_name=None):  # pragma: no cover
        if isinstance(x, list):  # pragma: no cover
            if __import__("builtins").all(
                (isinstance(i, int) for i in x)
            ):  # pragma: no cover
                return tuple(x)  # pragma: no cover
            return [_u(i, param_name) for i in x]  # pragma: no cover
        if isinstance(x, tuple):  # pragma: no cover
            return tuple((_u(i, param_name) for i in x))  # pragma: no cover
        if hasattr(x, "_tensor"):  # pragma: no cover
            return x._tensor  # pragma: no cover
        if isinstance(x, (int, float, bool, complex)):  # pragma: no cover
            if param_name not in {
                "axis",
                "rtol",
                "n",
                "m",
                "endpoint",
                "offset",
                "a_max",
                "ddof",
                "dims",
                "source",
                "axis2",
                "stop",
                "k",
                "dtype",
                "reps",
                "a_min",
                "sparse",
                "decimals",
                "strides",
                "num",
                "max",
                "indices_or_sections",
                "shape",
                "atol",
                "axis1",
                "step",
                "axes",
                "fill_value",
                "stream",
                "destination",
                "alpha",
                "beta",
                "start",
                "equal_nan",
                "keepdims",
                "min",
                "p",
            } and "eval" not in (
                "transpose",
                "moveaxis",
                "reshape",
                "expand_dims",
                "linspace",
                "logcumsumexp",
                "cummin",
                "cummax",
                "cumprod",
                "cumsum",
                "logsumexp",
                "all",
                "any",
                "argmin",
                "argmax",
                "prod",
                "sum",
                "mean",
                "var",
                "std",
                "max",
                "min",
                "tensordot",
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
        fn = getattr(mops, "eval")  # pragma: no cover
    except AttributeError:  # pragma: no cover
        try:
            fn = getattr(mnn, "eval")
        except AttributeError:  # pragma: no cover
            try:
                fn = getattr(mrand, "eval")
            except AttributeError:  # pragma: no cover  # pragma: no cover
                raise NotImplementedError("Missing in compiler: eval")
    res = fn(
        *[_u(x) for x in args],
        **{k: _u(v, k) for (k, v) in kwargs.items() if k != "stream"},
    )  # pragma: no cover
    return _w(res)  # pragma: no cover


def old_split(*args, **kwargs):
    """Compute old_split.

    Args:
        args: The args argument.
        kwargs: The kwargs argument.

    Returns:
        The result of old_split.

    """
    from zero_mlx.array import array, _to_tensor  # pragma: no cover
    import ml_switcheroo.ops as mops  # pragma: no cover
    import ml_switcheroo.nn as mnn  # pragma: no cover
    import ml_switcheroo.random as mrand  # pragma: no cover
    import types  # pragma: no cover

    def _u(x, param_name=None):  # pragma: no cover
        if isinstance(x, list):  # pragma: no cover
            if __import__("builtins").all(
                (isinstance(i, int) for i in x)
            ):  # pragma: no cover
                return tuple(x)  # pragma: no cover
            return [_u(i, param_name) for i in x]  # pragma: no cover
        if isinstance(x, tuple):  # pragma: no cover
            return tuple((_u(i, param_name) for i in x))  # pragma: no cover
        if hasattr(x, "_tensor"):  # pragma: no cover
            return x._tensor  # pragma: no cover
        if isinstance(x, (int, float, bool, complex)):  # pragma: no cover
            if param_name not in {
                "axis",
                "rtol",
                "n",
                "m",
                "endpoint",
                "offset",
                "a_max",
                "ddof",
                "dims",
                "source",
                "axis2",
                "stop",
                "k",
                "dtype",
                "reps",
                "a_min",
                "sparse",
                "decimals",
                "strides",
                "num",
                "max",
                "indices_or_sections",
                "shape",
                "atol",
                "axis1",
                "step",
                "axes",
                "fill_value",
                "stream",
                "destination",
                "alpha",
                "beta",
                "start",
                "equal_nan",
                "keepdims",
                "min",
                "p",
            } and "old_split" not in (
                "transpose",
                "moveaxis",
                "reshape",
                "expand_dims",
                "linspace",
                "logcumsumexp",
                "cummin",
                "cummax",
                "cumprod",
                "cumsum",
                "logsumexp",
                "all",
                "any",
                "argmin",
                "argmax",
                "prod",
                "sum",
                "mean",
                "var",
                "std",
                "max",
                "min",
                "tensordot",
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
        fn = getattr(mops, "old_split")  # pragma: no cover
    except AttributeError:  # pragma: no cover
        try:
            fn = getattr(mnn, "old_split")
        except AttributeError:  # pragma: no cover
            try:
                fn = getattr(mrand, "old_split")
            except AttributeError:  # pragma: no cover  # pragma: no cover
                raise NotImplementedError("Missing in compiler: old_split")
    res = fn(
        *[_u(x) for x in args],
        **{k: _u(v, k) for (k, v) in kwargs.items() if k != "stream"},
    )  # pragma: no cover
    return _w(res)  # pragma: no cover


def asarray(
    a: Any, dtype: Optional[DType] = None, stream: Any = None, copy: bool = False
) -> Any:
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
    import ml_switcheroo.ops as mops  # pragma: no cover
    import ml_switcheroo.nn as mnn  # pragma: no cover
    import ml_switcheroo.random as mrand  # pragma: no cover
    import types  # pragma: no cover

    def _u(x, param_name=None):  # pragma: no cover
        if isinstance(x, list):  # pragma: no cover
            if __import__("builtins").all(
                (isinstance(i, int) for i in x)
            ):  # pragma: no cover
                return tuple(x)  # pragma: no cover
            return [_u(i, param_name) for i in x]  # pragma: no cover
        if isinstance(x, tuple):  # pragma: no cover
            return tuple((_u(i, param_name) for i in x))  # pragma: no cover
        if hasattr(x, "_tensor"):  # pragma: no cover
            return x._tensor  # pragma: no cover
        if isinstance(x, (int, float, bool, complex)):  # pragma: no cover
            if param_name not in {
                "axis",
                "rtol",
                "n",
                "m",
                "endpoint",
                "offset",
                "a_max",
                "ddof",
                "dims",
                "source",
                "axis2",
                "stop",
                "k",
                "dtype",
                "reps",
                "a_min",
                "sparse",
                "decimals",
                "strides",
                "num",
                "max",
                "indices_or_sections",
                "shape",
                "atol",
                "axis1",
                "step",
                "axes",
                "fill_value",
                "stream",
                "destination",
                "alpha",
                "beta",
                "start",
                "equal_nan",
                "keepdims",
                "min",
                "p",
            } and "asarray" not in (
                "transpose",
                "moveaxis",
                "reshape",
                "expand_dims",
                "linspace",
                "logcumsumexp",
                "cummin",
                "cummax",
                "cumprod",
                "cumsum",
                "logsumexp",
                "all",
                "any",
                "argmin",
                "argmax",
                "prod",
                "sum",
                "mean",
                "var",
                "std",
                "max",
                "min",
                "tensordot",
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
        fn = getattr(mops, "asarray")  # pragma: no cover
    except AttributeError:  # pragma: no cover
        try:
            fn = getattr(mnn, "asarray")
        except AttributeError:  # pragma: no cover
            try:
                fn = getattr(mrand, "asarray")
            except AttributeError:  # pragma: no cover  # pragma: no cover
                raise NotImplementedError("Missing in compiler: asarray")
    res = fn(_u(a, "a"), _u(dtype, "dtype"), _u(copy, "copy"))  # pragma: no cover
    return _w(res)  # pragma: no cover


def stack(arrays: Sequence[Any], axis: int = 0, stream: Any = None) -> Any:
    """Compute stack.

    Args:
        arrays: The arrays argument.
        axis: The axis argument.
        stream: The stream argument.

    Returns:
        The result of stack.

    """
    from zero_mlx.array import array, _to_tensor  # pragma: no cover
    import ml_switcheroo.ops as mops  # pragma: no cover
    import ml_switcheroo.nn as mnn  # pragma: no cover
    import ml_switcheroo.random as mrand  # pragma: no cover
    import types  # pragma: no cover

    def _u(x, param_name=None):  # pragma: no cover
        if isinstance(x, list):  # pragma: no cover
            if __import__("builtins").all(
                (isinstance(i, int) for i in x)
            ):  # pragma: no cover
                return tuple(x)  # pragma: no cover
            return [_u(i, param_name) for i in x]  # pragma: no cover
        if isinstance(x, tuple):  # pragma: no cover
            return tuple((_u(i, param_name) for i in x))  # pragma: no cover
        if hasattr(x, "_tensor"):  # pragma: no cover
            return x._tensor  # pragma: no cover
        if isinstance(x, (int, float, bool, complex)):  # pragma: no cover
            if param_name not in {
                "axis",
                "rtol",
                "n",
                "m",
                "endpoint",
                "offset",
                "a_max",
                "ddof",
                "dims",
                "source",
                "axis2",
                "stop",
                "k",
                "dtype",
                "reps",
                "a_min",
                "sparse",
                "decimals",
                "strides",
                "num",
                "max",
                "indices_or_sections",
                "shape",
                "atol",
                "axis1",
                "step",
                "axes",
                "fill_value",
                "stream",
                "destination",
                "alpha",
                "beta",
                "start",
                "equal_nan",
                "keepdims",
                "min",
                "p",
            } and "stack" not in (
                "transpose",
                "moveaxis",
                "reshape",
                "expand_dims",
                "linspace",
                "logcumsumexp",
                "cummin",
                "cummax",
                "cumprod",
                "cumsum",
                "logsumexp",
                "all",
                "any",
                "argmin",
                "argmax",
                "prod",
                "sum",
                "mean",
                "var",
                "std",
                "max",
                "min",
                "tensordot",
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
        fn = getattr(mops, "stack")  # pragma: no cover
    except AttributeError:  # pragma: no cover
        try:
            fn = getattr(mnn, "stack")
        except AttributeError:  # pragma: no cover
            try:
                fn = getattr(mrand, "stack")
            except AttributeError:  # pragma: no cover  # pragma: no cover
                raise NotImplementedError("Missing in compiler: stack")
    res = fn(_u(arrays, "arrays"), _u(axis, "axis"))  # pragma: no cover
    return _w(res)  # pragma: no cover


def sin(a: Any, stream: Any = None) -> Any:
    """Compute sin.

    Args:
        a: The a argument.
        stream: The stream argument.

    Returns:
        The result of sin.

    """
    from zero_mlx.array import array, _to_tensor
    import ml_switcheroo.ops as mops
    import ml_switcheroo.nn as mnn
    import ml_switcheroo.random as mrand
    import types

    def _u(x, param_name=None):
        if isinstance(x, list):  # pragma: no cover
            if __import__("builtins").all(
                (isinstance(i, int) for i in x)
            ):  # pragma: no cover
                return tuple(x)  # pragma: no cover
            return [_u(i, param_name) for i in x]  # pragma: no cover
        if isinstance(x, tuple):  # pragma: no cover
            return tuple((_u(i, param_name) for i in x))  # pragma: no cover
        if hasattr(x, "_tensor"):  # pragma: no cover
            return x._tensor  # pragma: no cover
        if isinstance(x, (int, float, bool, complex)):  # pragma: no cover
            if param_name not in {
                "axis",
                "rtol",
                "n",
                "m",
                "endpoint",
                "offset",
                "a_max",
                "ddof",
                "dims",
                "source",
                "axis2",
                "stop",
                "k",
                "dtype",
                "reps",
                "a_min",
                "sparse",
                "decimals",
                "strides",
                "num",
                "max",
                "indices_or_sections",
                "shape",
                "atol",
                "axis1",
                "step",
                "axes",
                "fill_value",
                "stream",
                "destination",
                "alpha",
                "beta",
                "start",
                "equal_nan",
                "keepdims",
                "min",
                "p",
            } and "sin" not in (
                "transpose",
                "moveaxis",
                "reshape",
                "expand_dims",
                "linspace",
                "logcumsumexp",
                "cummin",
                "cummax",
                "cumprod",
                "cumsum",
                "logsumexp",
                "all",
                "any",
                "argmin",
                "argmax",
                "prod",
                "sum",
                "mean",
                "var",
                "std",
                "max",
                "min",
                "tensordot",
            ):  # pragma: no cover
                return _to_tensor(x)  # pragma: no cover
        if type(x).__name__ == "ndarray":  # pragma: no cover
            return _to_tensor(x)  # pragma: no cover
        return x  # pragma: no cover

    def _w(x):
        if isinstance(x, tuple):  # pragma: no cover
            return tuple((_w(i) for i in x))  # pragma: no cover
        if isinstance(x, list):  # pragma: no cover
            return [_w(i) for i in x]  # pragma: no cover
        return array(x) if hasattr(x, "shape") else x  # pragma: no cover

    try:
        fn = getattr(mops, "sin")
    except AttributeError:  # pragma: no cover
        try:
            fn = getattr(mnn, "sin")
        except AttributeError:  # pragma: no cover
            try:
                fn = getattr(mrand, "sin")
            except AttributeError:  # pragma: no cover  # pragma: no cover
                raise NotImplementedError("Missing in compiler: sin")
    res = fn(_u(a, "a"))
    return _w(res)


def square(a: Any, stream: Any = None) -> Any:
    """Compute square.

    Args:
        a: The a argument.
        stream: The stream argument.

    Returns:
        The result of square.

    """
    from zero_mlx.array import array, _to_tensor
    import ml_switcheroo.ops as mops
    import ml_switcheroo.nn as mnn
    import ml_switcheroo.random as mrand
    import types

    def _u(x, param_name=None):
        if isinstance(x, list):  # pragma: no cover
            if __import__("builtins").all(
                (isinstance(i, int) for i in x)
            ):  # pragma: no cover
                return tuple(x)  # pragma: no cover
            return [_u(i, param_name) for i in x]  # pragma: no cover
        if isinstance(x, tuple):  # pragma: no cover
            return tuple((_u(i, param_name) for i in x))  # pragma: no cover
        if hasattr(x, "_tensor"):  # pragma: no cover
            return x._tensor  # pragma: no cover
        if isinstance(x, (int, float, bool, complex)):  # pragma: no cover
            if param_name not in {
                "axis",
                "rtol",
                "n",
                "m",
                "endpoint",
                "offset",
                "a_max",
                "ddof",
                "dims",
                "source",
                "axis2",
                "stop",
                "k",
                "dtype",
                "reps",
                "a_min",
                "sparse",
                "decimals",
                "strides",
                "num",
                "max",
                "indices_or_sections",
                "shape",
                "atol",
                "axis1",
                "step",
                "axes",
                "fill_value",
                "stream",
                "destination",
                "alpha",
                "beta",
                "start",
                "equal_nan",
                "keepdims",
                "min",
                "p",
            } and "square" not in (
                "transpose",
                "moveaxis",
                "reshape",
                "expand_dims",
                "linspace",
                "logcumsumexp",
                "cummin",
                "cummax",
                "cumprod",
                "cumsum",
                "logsumexp",
                "all",
                "any",
                "argmin",
                "argmax",
                "prod",
                "sum",
                "mean",
                "var",
                "std",
                "max",
                "min",
                "tensordot",
            ):  # pragma: no cover
                return _to_tensor(x)  # pragma: no cover
        if type(x).__name__ == "ndarray":  # pragma: no cover
            return _to_tensor(x)  # pragma: no cover
        return x  # pragma: no cover

    def _w(x):
        if isinstance(x, tuple):  # pragma: no cover
            return tuple((_w(i) for i in x))  # pragma: no cover
        if isinstance(x, list):  # pragma: no cover
            return [_w(i) for i in x]  # pragma: no cover
        return array(x) if hasattr(x, "shape") else x  # pragma: no cover

    try:
        fn = getattr(mops, "square")
    except AttributeError:  # pragma: no cover
        try:
            fn = getattr(mnn, "square")
        except AttributeError:  # pragma: no cover
            try:
                fn = getattr(mrand, "square")
            except AttributeError:  # pragma: no cover  # pragma: no cover
                raise NotImplementedError("Missing in compiler: square")
    res = fn(_u(a, "a"))
    return _w(res)


def sum(a: Any, axis: Any = None, keepdims: bool = False, stream: Any = None) -> Any:
    """Compute sum.

    Args:
        a: The a argument.
        axis: The axis argument.
        keepdims: The keepdims argument.
        stream: The stream argument.

    Returns:
        The result of sum.

    """
    from zero_mlx.array import array, _to_tensor
    import ml_switcheroo.ops as mops
    import ml_switcheroo.nn as mnn
    import ml_switcheroo.random as mrand
    import types

    def _u(x, param_name=None):
        if isinstance(x, list):  # pragma: no cover
            if __import__("builtins").all(
                (isinstance(i, int) for i in x)
            ):  # pragma: no cover
                return tuple(x)  # pragma: no cover
            return [_u(i, param_name) for i in x]  # pragma: no cover
        if isinstance(x, tuple):  # pragma: no cover
            return tuple((_u(i, param_name) for i in x))  # pragma: no cover
        if hasattr(x, "_tensor"):  # pragma: no cover
            return x._tensor  # pragma: no cover
        if isinstance(x, (int, float, bool, complex)):  # pragma: no cover
            if param_name not in {
                "axis",
                "rtol",
                "n",
                "m",
                "endpoint",
                "offset",
                "a_max",
                "ddof",
                "dims",
                "source",
                "axis2",
                "stop",
                "k",
                "dtype",
                "reps",
                "a_min",
                "sparse",
                "decimals",
                "strides",
                "num",
                "max",
                "indices_or_sections",
                "shape",
                "atol",
                "axis1",
                "step",
                "axes",
                "fill_value",
                "stream",
                "destination",
                "alpha",
                "beta",
                "start",
                "equal_nan",
                "keepdims",
                "min",
                "p",
            } and "sum" not in (
                "transpose",
                "moveaxis",
                "reshape",
                "expand_dims",
                "linspace",
                "logcumsumexp",
                "cummin",
                "cummax",
                "cumprod",
                "cumsum",
                "logsumexp",
                "all",
                "any",
                "argmin",
                "argmax",
                "prod",
                "sum",
                "mean",
                "var",
                "std",
                "max",
                "min",
                "tensordot",
            ):  # pragma: no cover
                return _to_tensor(x)  # pragma: no cover
        if type(x).__name__ == "ndarray":  # pragma: no cover
            return _to_tensor(x)  # pragma: no cover
        return x  # pragma: no cover

    def _w(x):
        if isinstance(x, tuple):  # pragma: no cover
            return tuple((_w(i) for i in x))  # pragma: no cover
        if isinstance(x, list):  # pragma: no cover
            return [_w(i) for i in x]  # pragma: no cover
        return array(x) if hasattr(x, "shape") else x  # pragma: no cover

    try:
        fn = getattr(mops, "sum")
    except AttributeError:  # pragma: no cover
        try:
            fn = getattr(mnn, "sum")
        except AttributeError:  # pragma: no cover
            try:
                fn = getattr(mrand, "sum")
            except AttributeError:  # pragma: no cover  # pragma: no cover
                raise NotImplementedError("Missing in compiler: sum")
    res = fn(_u(a, "a"), _u(axis, "axis"), _u(keepdims, "keepdims"))
    return _w(res)


def mean(a: Any, axis: Any = None, keepdims: bool = False, stream: Any = None) -> Any:
    """Compute mean.

    Args:
        a: The a argument.
        axis: The axis argument.
        keepdims: The keepdims argument.
        stream: The stream argument.

    Returns:
        The result of mean.

    """
    from zero_mlx.array import array, _to_tensor
    import ml_switcheroo.ops as mops
    import ml_switcheroo.nn as mnn
    import ml_switcheroo.random as mrand
    import types

    def _u(x, param_name=None):
        if isinstance(x, list):  # pragma: no cover
            if __import__("builtins").all(
                (isinstance(i, int) for i in x)
            ):  # pragma: no cover
                return tuple(x)  # pragma: no cover
            return [_u(i, param_name) for i in x]  # pragma: no cover
        if isinstance(x, tuple):  # pragma: no cover
            return tuple((_u(i, param_name) for i in x))  # pragma: no cover
        if hasattr(x, "_tensor"):  # pragma: no cover
            return x._tensor  # pragma: no cover
        if isinstance(x, (int, float, bool, complex)):  # pragma: no cover
            if param_name not in {
                "axis",
                "rtol",
                "n",
                "m",
                "endpoint",
                "offset",
                "a_max",
                "ddof",
                "dims",
                "source",
                "axis2",
                "stop",
                "k",
                "dtype",
                "reps",
                "a_min",
                "sparse",
                "decimals",
                "strides",
                "num",
                "max",
                "indices_or_sections",
                "shape",
                "atol",
                "axis1",
                "step",
                "axes",
                "fill_value",
                "stream",
                "destination",
                "alpha",
                "beta",
                "start",
                "equal_nan",
                "keepdims",
                "min",
                "p",
            } and "mean" not in (
                "transpose",
                "moveaxis",
                "reshape",
                "expand_dims",
                "linspace",
                "logcumsumexp",
                "cummin",
                "cummax",
                "cumprod",
                "cumsum",
                "logsumexp",
                "all",
                "any",
                "argmin",
                "argmax",
                "prod",
                "sum",
                "mean",
                "var",
                "std",
                "max",
                "min",
                "tensordot",
            ):  # pragma: no cover
                return _to_tensor(x)  # pragma: no cover
        if type(x).__name__ == "ndarray":  # pragma: no cover
            return _to_tensor(x)  # pragma: no cover
        return x  # pragma: no cover

    def _w(x):
        if isinstance(x, tuple):  # pragma: no cover
            return tuple((_w(i) for i in x))  # pragma: no cover
        if isinstance(x, list):  # pragma: no cover
            return [_w(i) for i in x]  # pragma: no cover
        return array(x) if hasattr(x, "shape") else x  # pragma: no cover

    try:
        fn = getattr(mops, "mean")
    except AttributeError:  # pragma: no cover
        try:
            fn = getattr(mnn, "mean")
        except AttributeError:  # pragma: no cover
            try:
                fn = getattr(mrand, "mean")
            except AttributeError:  # pragma: no cover  # pragma: no cover
                raise NotImplementedError("Missing in compiler: mean")
    res = fn(_u(a, "a"), _u(axis, "axis"), _u(keepdims, "keepdims"))
    return _w(res)


def arange(
    start: Any,
    stop: Any = None,
    step: Any = 1,
    dtype: Optional[DType] = None,
    stream: Any = None,
) -> Any:
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
    from zero_mlx.array import array, _to_tensor
    import ml_switcheroo.ops as mops
    import ml_switcheroo.nn as mnn
    import ml_switcheroo.random as mrand
    import types

    def _u(x, param_name=None):
        if isinstance(x, list):  # pragma: no cover
            if __import__("builtins").all(
                (isinstance(i, int) for i in x)
            ):  # pragma: no cover
                return tuple(x)  # pragma: no cover
            return [_u(i, param_name) for i in x]  # pragma: no cover
        if isinstance(x, tuple):  # pragma: no cover
            return tuple((_u(i, param_name) for i in x))  # pragma: no cover
        if hasattr(x, "_tensor"):  # pragma: no cover
            return x._tensor  # pragma: no cover
        if isinstance(x, (int, float, bool, complex)):  # pragma: no cover
            if param_name not in {
                "axis",
                "rtol",
                "n",
                "m",
                "endpoint",
                "offset",
                "a_max",
                "ddof",
                "dims",
                "source",
                "axis2",
                "stop",
                "k",
                "dtype",
                "reps",
                "a_min",
                "sparse",
                "decimals",
                "strides",
                "num",
                "max",
                "indices_or_sections",
                "shape",
                "atol",
                "axis1",
                "step",
                "axes",
                "fill_value",
                "stream",
                "destination",
                "alpha",
                "beta",
                "start",
                "equal_nan",
                "keepdims",
                "min",
                "p",
            } and "arange" not in (
                "transpose",
                "moveaxis",
                "reshape",
                "expand_dims",
                "linspace",
                "logcumsumexp",
                "cummin",
                "cummax",
                "cumprod",
                "cumsum",
                "logsumexp",
                "all",
                "any",
                "argmin",
                "argmax",
                "prod",
                "sum",
                "mean",
                "var",
                "std",
                "max",
                "min",
                "tensordot",
            ):  # pragma: no cover
                return _to_tensor(x)  # pragma: no cover
        if type(x).__name__ == "ndarray":  # pragma: no cover
            return _to_tensor(x)  # pragma: no cover
        return x  # pragma: no cover

    def _w(x):
        if isinstance(x, tuple):  # pragma: no cover
            return tuple((_w(i) for i in x))  # pragma: no cover
        if isinstance(x, list):  # pragma: no cover
            return [_w(i) for i in x]  # pragma: no cover
        return array(x) if hasattr(x, "shape") else x  # pragma: no cover

    try:
        fn = getattr(mops, "arange")
    except AttributeError:  # pragma: no cover
        try:
            fn = getattr(mnn, "arange")
        except AttributeError:  # pragma: no cover
            try:
                fn = getattr(mrand, "arange")
            except AttributeError:  # pragma: no cover  # pragma: no cover
                raise NotImplementedError("Missing in compiler: arange")
    res = fn(_u(start, "start"), _u(stop, "stop"), _u(step, "step"), _u(dtype, "dtype"))
    return _w(res)


def full(
    shape: Any, fill_value: Any, dtype: Optional[DType] = None, stream: Any = None
) -> Any:
    """Compute full.

    Args:
        shape: The shape argument.
        fill_value: The fill_value argument.
        dtype: The dtype argument.
        stream: The stream argument.

    Returns:
        The result of full.

    """
    from zero_mlx.array import array, _to_tensor
    import ml_switcheroo.ops as mops
    import ml_switcheroo.nn as mnn
    import ml_switcheroo.random as mrand
    import types

    def _u(x, param_name=None):
        if isinstance(x, list):  # pragma: no cover
            if __import__("builtins").all(
                (isinstance(i, int) for i in x)
            ):  # pragma: no cover
                return tuple(x)  # pragma: no cover
            return [_u(i, param_name) for i in x]  # pragma: no cover
        if isinstance(x, tuple):  # pragma: no cover
            return tuple((_u(i, param_name) for i in x))  # pragma: no cover
        if hasattr(x, "_tensor"):  # pragma: no cover
            return x._tensor  # pragma: no cover
        if isinstance(x, (int, float, bool, complex)):  # pragma: no cover
            if param_name not in {
                "axis",
                "rtol",
                "n",
                "m",
                "endpoint",
                "offset",
                "a_max",
                "ddof",
                "dims",
                "source",
                "axis2",
                "stop",
                "k",
                "dtype",
                "reps",
                "a_min",
                "sparse",
                "decimals",
                "strides",
                "num",
                "max",
                "indices_or_sections",
                "shape",
                "atol",
                "axis1",
                "step",
                "axes",
                "fill_value",
                "stream",
                "destination",
                "alpha",
                "beta",
                "start",
                "equal_nan",
                "keepdims",
                "min",
                "p",
            } and "full" not in (
                "transpose",
                "moveaxis",
                "reshape",
                "expand_dims",
                "linspace",
                "logcumsumexp",
                "cummin",
                "cummax",
                "cumprod",
                "cumsum",
                "logsumexp",
                "all",
                "any",
                "argmin",
                "argmax",
                "prod",
                "sum",
                "mean",
                "var",
                "std",
                "max",
                "min",
                "tensordot",
            ):  # pragma: no cover
                return _to_tensor(x)  # pragma: no cover
        if type(x).__name__ == "ndarray":  # pragma: no cover
            return _to_tensor(x)  # pragma: no cover
        return x  # pragma: no cover

    def _w(x):
        if isinstance(x, tuple):  # pragma: no cover
            return tuple((_w(i) for i in x))  # pragma: no cover
        if isinstance(x, list):  # pragma: no cover
            return [_w(i) for i in x]  # pragma: no cover
        return array(x) if hasattr(x, "shape") else x  # pragma: no cover

    try:
        fn = getattr(mops, "full")
    except AttributeError:  # pragma: no cover
        try:
            fn = getattr(mnn, "full")
        except AttributeError:  # pragma: no cover
            try:
                fn = getattr(mrand, "full")
            except AttributeError:  # pragma: no cover  # pragma: no cover
                raise NotImplementedError("Missing in compiler: full")
    res = fn(_u(shape, "shape"), _u(fill_value, "fill_value"), _u(dtype, "dtype"))
    return _w(res)


def zeros(shape: Any, dtype: Optional[DType] = None, stream: Any = None) -> Any:
    """Compute zeros.

    Args:
        shape: The shape argument.
        dtype: The dtype argument.
        stream: The stream argument.

    Returns:
        The result of zeros.

    """
    from zero_mlx.array import array, _to_tensor
    import ml_switcheroo.ops as mops
    import ml_switcheroo.nn as mnn
    import ml_switcheroo.random as mrand
    import types

    def _u(x, param_name=None):
        if isinstance(x, list):  # pragma: no cover
            if __import__("builtins").all(
                (isinstance(i, int) for i in x)
            ):  # pragma: no cover
                return tuple(x)  # pragma: no cover
            return [_u(i, param_name) for i in x]  # pragma: no cover
        if isinstance(x, tuple):  # pragma: no cover
            return tuple((_u(i, param_name) for i in x))  # pragma: no cover
        if hasattr(x, "_tensor"):  # pragma: no cover
            return x._tensor  # pragma: no cover
        if isinstance(x, (int, float, bool, complex)):  # pragma: no cover
            if param_name not in {
                "axis",
                "rtol",
                "n",
                "m",
                "endpoint",
                "offset",
                "a_max",
                "ddof",
                "dims",
                "source",
                "axis2",
                "stop",
                "k",
                "dtype",
                "reps",
                "a_min",
                "sparse",
                "decimals",
                "strides",
                "num",
                "max",
                "indices_or_sections",
                "shape",
                "atol",
                "axis1",
                "step",
                "axes",
                "fill_value",
                "stream",
                "destination",
                "alpha",
                "beta",
                "start",
                "equal_nan",
                "keepdims",
                "min",
                "p",
            } and "zeros" not in (
                "transpose",
                "moveaxis",
                "reshape",
                "expand_dims",
                "linspace",
                "logcumsumexp",
                "cummin",
                "cummax",
                "cumprod",
                "cumsum",
                "logsumexp",
                "all",
                "any",
                "argmin",
                "argmax",
                "prod",
                "sum",
                "mean",
                "var",
                "std",
                "max",
                "min",
                "tensordot",
            ):  # pragma: no cover
                return _to_tensor(x)  # pragma: no cover
        if type(x).__name__ == "ndarray":  # pragma: no cover
            return _to_tensor(x)  # pragma: no cover
        return x  # pragma: no cover

    def _w(x):
        if isinstance(x, tuple):  # pragma: no cover
            return tuple((_w(i) for i in x))  # pragma: no cover
        if isinstance(x, list):  # pragma: no cover
            return [_w(i) for i in x]  # pragma: no cover
        return array(x) if hasattr(x, "shape") else x  # pragma: no cover

    try:
        fn = getattr(mops, "zeros")
    except AttributeError:  # pragma: no cover
        try:
            fn = getattr(mnn, "zeros")
        except AttributeError:  # pragma: no cover
            try:
                fn = getattr(mrand, "zeros")
            except AttributeError:  # pragma: no cover  # pragma: no cover
                raise NotImplementedError("Missing in compiler: zeros")
    res = fn(_u(shape, "shape"), _u(dtype, "dtype"))
    return _w(res)


def ones(shape: Any, dtype: Optional[DType] = None, stream: Any = None) -> Any:
    """Compute ones.

    Args:
        shape: The shape argument.
        dtype: The dtype argument.
        stream: The stream argument.

    Returns:
        The result of ones.

    """
    from zero_mlx.array import array, _to_tensor
    import ml_switcheroo.ops as mops
    import ml_switcheroo.nn as mnn
    import ml_switcheroo.random as mrand
    import types

    def _u(x, param_name=None):
        if isinstance(x, list):  # pragma: no cover
            if __import__("builtins").all(
                (isinstance(i, int) for i in x)
            ):  # pragma: no cover
                return tuple(x)  # pragma: no cover
            return [_u(i, param_name) for i in x]  # pragma: no cover
        if isinstance(x, tuple):  # pragma: no cover
            return tuple((_u(i, param_name) for i in x))  # pragma: no cover
        if hasattr(x, "_tensor"):  # pragma: no cover
            return x._tensor  # pragma: no cover
        if isinstance(x, (int, float, bool, complex)):  # pragma: no cover
            if param_name not in {
                "axis",
                "rtol",
                "n",
                "m",
                "endpoint",
                "offset",
                "a_max",
                "ddof",
                "dims",
                "source",
                "axis2",
                "stop",
                "k",
                "dtype",
                "reps",
                "a_min",
                "sparse",
                "decimals",
                "strides",
                "num",
                "max",
                "indices_or_sections",
                "shape",
                "atol",
                "axis1",
                "step",
                "axes",
                "fill_value",
                "stream",
                "destination",
                "alpha",
                "beta",
                "start",
                "equal_nan",
                "keepdims",
                "min",
                "p",
            } and "ones" not in (
                "transpose",
                "moveaxis",
                "reshape",
                "expand_dims",
                "linspace",
                "logcumsumexp",
                "cummin",
                "cummax",
                "cumprod",
                "cumsum",
                "logsumexp",
                "all",
                "any",
                "argmin",
                "argmax",
                "prod",
                "sum",
                "mean",
                "var",
                "std",
                "max",
                "min",
                "tensordot",
            ):  # pragma: no cover
                return _to_tensor(x)  # pragma: no cover
        if type(x).__name__ == "ndarray":  # pragma: no cover
            return _to_tensor(x)  # pragma: no cover
        return x  # pragma: no cover

    def _w(x):
        if isinstance(x, tuple):  # pragma: no cover
            return tuple((_w(i) for i in x))  # pragma: no cover
        if isinstance(x, list):  # pragma: no cover
            return [_w(i) for i in x]  # pragma: no cover
        return array(x) if hasattr(x, "shape") else x  # pragma: no cover

    try:
        fn = getattr(mops, "ones")
    except AttributeError:  # pragma: no cover
        try:
            fn = getattr(mnn, "ones")
        except AttributeError:  # pragma: no cover
            try:
                fn = getattr(mrand, "ones")
            except AttributeError:  # pragma: no cover  # pragma: no cover
                raise NotImplementedError("Missing in compiler: ones")
    res = fn(_u(shape, "shape"), _u(dtype, "dtype"))
    return _w(res)


def zeros_like(a: Any, dtype: Optional[DType] = None, stream: Any = None) -> Any:
    """Compute zeros_like.

    Args:
        a: The a argument.
        dtype: The dtype argument.
        stream: The stream argument.

    Returns:
        The result of zeros_like.

    """
    from zero_mlx.array import array, _to_tensor
    import ml_switcheroo.ops as mops
    import ml_switcheroo.nn as mnn
    import ml_switcheroo.random as mrand
    import types

    def _u(x, param_name=None):
        if isinstance(x, list):  # pragma: no cover
            if __import__("builtins").all(
                (isinstance(i, int) for i in x)
            ):  # pragma: no cover
                return tuple(x)  # pragma: no cover
            return [_u(i, param_name) for i in x]  # pragma: no cover
        if isinstance(x, tuple):  # pragma: no cover
            return tuple((_u(i, param_name) for i in x))  # pragma: no cover
        if hasattr(x, "_tensor"):  # pragma: no cover
            return x._tensor  # pragma: no cover
        if isinstance(x, (int, float, bool, complex)):  # pragma: no cover
            if param_name not in {
                "axis",
                "rtol",
                "n",
                "m",
                "endpoint",
                "offset",
                "a_max",
                "ddof",
                "dims",
                "source",
                "axis2",
                "stop",
                "k",
                "dtype",
                "reps",
                "a_min",
                "sparse",
                "decimals",
                "strides",
                "num",
                "max",
                "indices_or_sections",
                "shape",
                "atol",
                "axis1",
                "step",
                "axes",
                "fill_value",
                "stream",
                "destination",
                "alpha",
                "beta",
                "start",
                "equal_nan",
                "keepdims",
                "min",
                "p",
            } and "zeros_like" not in (
                "transpose",
                "moveaxis",
                "reshape",
                "expand_dims",
                "linspace",
                "logcumsumexp",
                "cummin",
                "cummax",
                "cumprod",
                "cumsum",
                "logsumexp",
                "all",
                "any",
                "argmin",
                "argmax",
                "prod",
                "sum",
                "mean",
                "var",
                "std",
                "max",
                "min",
                "tensordot",
            ):  # pragma: no cover
                return _to_tensor(x)  # pragma: no cover
        if type(x).__name__ == "ndarray":  # pragma: no cover
            return _to_tensor(x)  # pragma: no cover
        return x  # pragma: no cover

    def _w(x):
        if isinstance(x, tuple):  # pragma: no cover
            return tuple((_w(i) for i in x))  # pragma: no cover
        if isinstance(x, list):  # pragma: no cover
            return [_w(i) for i in x]  # pragma: no cover
        return array(x) if hasattr(x, "shape") else x  # pragma: no cover

    try:
        fn = getattr(mops, "zeros_like")
    except AttributeError:  # pragma: no cover
        try:
            fn = getattr(mnn, "zeros_like")
        except AttributeError:  # pragma: no cover
            try:
                fn = getattr(mrand, "zeros_like")
            except AttributeError:  # pragma: no cover  # pragma: no cover
                raise NotImplementedError("Missing in compiler: zeros_like")
    res = fn(_u(a, "a"), _u(dtype, "dtype"))
    return _w(res)


def ones_like(a: Any, dtype: Optional[DType] = None, stream: Any = None) -> Any:
    """Compute ones_like.

    Args:
        a: The a argument.
        dtype: The dtype argument.
        stream: The stream argument.

    Returns:
        The result of ones_like.

    """
    from zero_mlx.array import array, _to_tensor
    import ml_switcheroo.ops as mops
    import ml_switcheroo.nn as mnn
    import ml_switcheroo.random as mrand
    import types

    def _u(x, param_name=None):
        if isinstance(x, list):  # pragma: no cover
            if __import__("builtins").all(
                (isinstance(i, int) for i in x)
            ):  # pragma: no cover
                return tuple(x)  # pragma: no cover
            return [_u(i, param_name) for i in x]  # pragma: no cover
        if isinstance(x, tuple):  # pragma: no cover
            return tuple((_u(i, param_name) for i in x))  # pragma: no cover
        if hasattr(x, "_tensor"):  # pragma: no cover
            return x._tensor  # pragma: no cover
        if isinstance(x, (int, float, bool, complex)):  # pragma: no cover
            if param_name not in {
                "axis",
                "rtol",
                "n",
                "m",
                "endpoint",
                "offset",
                "a_max",
                "ddof",
                "dims",
                "source",
                "axis2",
                "stop",
                "k",
                "dtype",
                "reps",
                "a_min",
                "sparse",
                "decimals",
                "strides",
                "num",
                "max",
                "indices_or_sections",
                "shape",
                "atol",
                "axis1",
                "step",
                "axes",
                "fill_value",
                "stream",
                "destination",
                "alpha",
                "beta",
                "start",
                "equal_nan",
                "keepdims",
                "min",
                "p",
            } and "ones_like" not in (
                "transpose",
                "moveaxis",
                "reshape",
                "expand_dims",
                "linspace",
                "logcumsumexp",
                "cummin",
                "cummax",
                "cumprod",
                "cumsum",
                "logsumexp",
                "all",
                "any",
                "argmin",
                "argmax",
                "prod",
                "sum",
                "mean",
                "var",
                "std",
                "max",
                "min",
                "tensordot",
            ):  # pragma: no cover
                return _to_tensor(x)  # pragma: no cover
        if type(x).__name__ == "ndarray":  # pragma: no cover
            return _to_tensor(x)  # pragma: no cover
        return x  # pragma: no cover

    def _w(x):
        if isinstance(x, tuple):  # pragma: no cover
            return tuple((_w(i) for i in x))  # pragma: no cover
        if isinstance(x, list):  # pragma: no cover
            return [_w(i) for i in x]  # pragma: no cover
        return array(x) if hasattr(x, "shape") else x  # pragma: no cover

    try:
        fn = getattr(mops, "ones_like")
    except AttributeError:  # pragma: no cover
        try:
            fn = getattr(mnn, "ones_like")
        except AttributeError:  # pragma: no cover
            try:
                fn = getattr(mrand, "ones_like")
            except AttributeError:  # pragma: no cover  # pragma: no cover
                raise NotImplementedError("Missing in compiler: ones_like")
    res = fn(_u(a, "a"), _u(dtype, "dtype"))
    return _w(res)


def array_equal(a: Any, b: Any, equal_nan: bool = False, stream: Any = None) -> bool:
    """Compute array_equal.

    Args:
        a: The a argument.
        b: The b argument.
        equal_nan: The equal_nan argument.
        stream: The stream argument.

    Returns:
        The result of array_equal.

    """
    from zero_mlx.array import array, _to_tensor
    import ml_switcheroo.ops as mops
    import ml_switcheroo.nn as mnn
    import ml_switcheroo.random as mrand
    import types

    def _u(x, param_name=None):
        if isinstance(x, list):  # pragma: no cover
            if __import__("builtins").all(
                (isinstance(i, int) for i in x)
            ):  # pragma: no cover
                return tuple(x)  # pragma: no cover
            return [_u(i, param_name) for i in x]  # pragma: no cover
        if isinstance(x, tuple):  # pragma: no cover
            return tuple((_u(i, param_name) for i in x))  # pragma: no cover
        if hasattr(x, "_tensor"):  # pragma: no cover
            return x._tensor  # pragma: no cover
        if isinstance(x, (int, float, bool, complex)):  # pragma: no cover
            if param_name not in {
                "axis",
                "rtol",
                "n",
                "m",
                "endpoint",
                "offset",
                "a_max",
                "ddof",
                "dims",
                "source",
                "axis2",
                "stop",
                "k",
                "dtype",
                "reps",
                "a_min",
                "sparse",
                "decimals",
                "strides",
                "num",
                "max",
                "indices_or_sections",
                "shape",
                "atol",
                "axis1",
                "step",
                "axes",
                "fill_value",
                "stream",
                "destination",
                "alpha",
                "beta",
                "start",
                "equal_nan",
                "keepdims",
                "min",
                "p",
            } and "array_equal" not in (
                "transpose",
                "moveaxis",
                "reshape",
                "expand_dims",
                "linspace",
                "logcumsumexp",
                "cummin",
                "cummax",
                "cumprod",
                "cumsum",
                "logsumexp",
                "all",
                "any",
                "argmin",
                "argmax",
                "prod",
                "sum",
                "mean",
                "var",
                "std",
                "max",
                "min",
                "tensordot",
            ):  # pragma: no cover
                return _to_tensor(x)  # pragma: no cover
        if type(x).__name__ == "ndarray":  # pragma: no cover
            return _to_tensor(x)  # pragma: no cover
        return x  # pragma: no cover

    def _w(x):
        if isinstance(x, tuple):  # pragma: no cover
            return tuple((_w(i) for i in x))  # pragma: no cover
        if isinstance(x, list):  # pragma: no cover
            return [_w(i) for i in x]  # pragma: no cover
        return array(x) if hasattr(x, "shape") else x  # pragma: no cover

    try:
        fn = getattr(mops, "array_equal")
    except AttributeError:  # pragma: no cover
        try:
            fn = getattr(mnn, "array_equal")
        except AttributeError:  # pragma: no cover
            try:
                fn = getattr(mrand, "array_equal")
            except AttributeError:  # pragma: no cover  # pragma: no cover
                raise NotImplementedError("Missing in compiler: array_equal")
    res = fn(_u(a, "a"), _u(b, "b"), _u(equal_nan, "equal_nan"))
    return _w(res)


def broadcast_to(a: Any, shape: Sequence[int], stream: Any = None) -> Any:
    """Compute broadcast_to.

    Args:
        a: The a argument.
        shape: The shape argument.
        stream: The stream argument.

    Returns:
        The result of broadcast_to.

    """
    from zero_mlx.array import array, _to_tensor
    import ml_switcheroo.ops as mops
    import ml_switcheroo.nn as mnn
    import ml_switcheroo.random as mrand
    import types

    def _u(x, param_name=None):
        if isinstance(x, list):  # pragma: no cover
            if __import__("builtins").all(
                (isinstance(i, int) for i in x)
            ):  # pragma: no cover
                return tuple(x)  # pragma: no cover
            return [_u(i, param_name) for i in x]  # pragma: no cover
        if isinstance(x, tuple):  # pragma: no cover
            return tuple((_u(i, param_name) for i in x))  # pragma: no cover
        if hasattr(x, "_tensor"):  # pragma: no cover
            return x._tensor  # pragma: no cover
        if isinstance(x, (int, float, bool, complex)):  # pragma: no cover
            if param_name not in {
                "axis",
                "rtol",
                "n",
                "m",
                "endpoint",
                "offset",
                "a_max",
                "ddof",
                "dims",
                "source",
                "axis2",
                "stop",
                "k",
                "dtype",
                "reps",
                "a_min",
                "sparse",
                "decimals",
                "strides",
                "num",
                "max",
                "indices_or_sections",
                "shape",
                "atol",
                "axis1",
                "step",
                "axes",
                "fill_value",
                "stream",
                "destination",
                "alpha",
                "beta",
                "start",
                "equal_nan",
                "keepdims",
                "min",
                "p",
            } and "broadcast_to" not in (
                "transpose",
                "moveaxis",
                "reshape",
                "expand_dims",
                "linspace",
                "logcumsumexp",
                "cummin",
                "cummax",
                "cumprod",
                "cumsum",
                "logsumexp",
                "all",
                "any",
                "argmin",
                "argmax",
                "prod",
                "sum",
                "mean",
                "var",
                "std",
                "max",
                "min",
                "tensordot",
            ):  # pragma: no cover
                return _to_tensor(x)  # pragma: no cover
        if type(x).__name__ == "ndarray":  # pragma: no cover
            return _to_tensor(x)  # pragma: no cover
        return x  # pragma: no cover

    def _w(x):
        if isinstance(x, tuple):  # pragma: no cover
            return tuple((_w(i) for i in x))  # pragma: no cover
        if isinstance(x, list):  # pragma: no cover
            return [_w(i) for i in x]  # pragma: no cover
        return array(x) if hasattr(x, "shape") else x  # pragma: no cover

    try:
        fn = getattr(mops, "broadcast_to")
    except AttributeError:  # pragma: no cover
        try:
            fn = getattr(mnn, "broadcast_to")
        except AttributeError:  # pragma: no cover
            try:
                fn = getattr(mrand, "broadcast_to")
            except AttributeError:  # pragma: no cover  # pragma: no cover
                raise NotImplementedError("Missing in compiler: broadcast_to")
    res = fn(_u(a, "a"), _u(shape, "shape"))
    return _w(res)


def as_strided(
    a: Any,
    shape: Sequence[int],
    strides: Sequence[int],
    offset: int = 0,
    stream: Any = None,
) -> Any:
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
    from zero_mlx.array import array, _to_tensor
    import ml_switcheroo.ops as mops
    import ml_switcheroo.nn as mnn
    import ml_switcheroo.random as mrand
    import types

    def _u(x, param_name=None):
        if isinstance(x, list):  # pragma: no cover
            if __import__("builtins").all(
                (isinstance(i, int) for i in x)
            ):  # pragma: no cover
                return tuple(x)  # pragma: no cover
            return [_u(i, param_name) for i in x]  # pragma: no cover
        if isinstance(x, tuple):  # pragma: no cover
            return tuple((_u(i, param_name) for i in x))  # pragma: no cover
        if hasattr(x, "_tensor"):  # pragma: no cover
            return x._tensor  # pragma: no cover
        if isinstance(x, (int, float, bool, complex)):  # pragma: no cover
            if param_name not in {
                "axis",
                "rtol",
                "n",
                "m",
                "endpoint",
                "offset",
                "a_max",
                "ddof",
                "dims",
                "source",
                "axis2",
                "stop",
                "k",
                "dtype",
                "reps",
                "a_min",
                "sparse",
                "decimals",
                "strides",
                "num",
                "max",
                "indices_or_sections",
                "shape",
                "atol",
                "axis1",
                "step",
                "axes",
                "fill_value",
                "stream",
                "destination",
                "alpha",
                "beta",
                "start",
                "equal_nan",
                "keepdims",
                "min",
                "p",
            } and "as_strided" not in (
                "transpose",
                "moveaxis",
                "reshape",
                "expand_dims",
                "linspace",
                "logcumsumexp",
                "cummin",
                "cummax",
                "cumprod",
                "cumsum",
                "logsumexp",
                "all",
                "any",
                "argmin",
                "argmax",
                "prod",
                "sum",
                "mean",
                "var",
                "std",
                "max",
                "min",
                "tensordot",
            ):  # pragma: no cover
                return _to_tensor(x)  # pragma: no cover
        if type(x).__name__ == "ndarray":  # pragma: no cover
            return _to_tensor(x)  # pragma: no cover
        return x  # pragma: no cover

    def _w(x):
        if isinstance(x, tuple):  # pragma: no cover
            return tuple((_w(i) for i in x))  # pragma: no cover
        if isinstance(x, list):  # pragma: no cover
            return [_w(i) for i in x]  # pragma: no cover
        return array(x) if hasattr(x, "shape") else x  # pragma: no cover

    try:
        fn = getattr(mops, "as_strided")
    except AttributeError:  # pragma: no cover
        try:
            fn = getattr(mnn, "as_strided")
        except AttributeError:  # pragma: no cover
            try:
                fn = getattr(mrand, "as_strided")
            except AttributeError:  # pragma: no cover  # pragma: no cover
                raise NotImplementedError("Missing in compiler: as_strided")
    res = fn(
        _u(a, "a"), _u(shape, "shape"), _u(strides, "strides"), _u(offset, "offset")
    )
    return _w(res)


def reshape(a: Any, shape: Any, stream: Any = None) -> Any:
    """Compute reshape.

    Args:
        a: The a argument.
        shape: The shape argument.
        stream: The stream argument.

    Returns:
        The result of reshape.

    """
    from zero_mlx.array import array, _to_tensor
    import ml_switcheroo.ops as mops
    import ml_switcheroo.nn as mnn
    import ml_switcheroo.random as mrand
    import types

    def _u(x, param_name=None):
        if isinstance(x, list):  # pragma: no cover
            if __import__("builtins").all(
                (isinstance(i, int) for i in x)
            ):  # pragma: no cover
                return tuple(x)  # pragma: no cover
            return [_u(i, param_name) for i in x]  # pragma: no cover
        if isinstance(x, tuple):  # pragma: no cover
            return tuple((_u(i, param_name) for i in x))  # pragma: no cover
        if hasattr(x, "_tensor"):  # pragma: no cover
            return x._tensor  # pragma: no cover
        if isinstance(x, (int, float, bool, complex)):  # pragma: no cover
            if param_name not in {
                "axis",
                "rtol",
                "n",
                "m",
                "endpoint",
                "offset",
                "a_max",
                "ddof",
                "dims",
                "source",
                "axis2",
                "stop",
                "k",
                "dtype",
                "reps",
                "a_min",
                "sparse",
                "decimals",
                "strides",
                "num",
                "max",
                "indices_or_sections",
                "shape",
                "atol",
                "axis1",
                "step",
                "axes",
                "fill_value",
                "stream",
                "destination",
                "alpha",
                "beta",
                "start",
                "equal_nan",
                "keepdims",
                "min",
                "p",
            } and "reshape" not in (
                "transpose",
                "moveaxis",
                "reshape",
                "expand_dims",
                "linspace",
                "logcumsumexp",
                "cummin",
                "cummax",
                "cumprod",
                "cumsum",
                "logsumexp",
                "all",
                "any",
                "argmin",
                "argmax",
                "prod",
                "sum",
                "mean",
                "var",
                "std",
                "max",
                "min",
                "tensordot",
            ):  # pragma: no cover
                return _to_tensor(x)  # pragma: no cover
        if type(x).__name__ == "ndarray":  # pragma: no cover
            return _to_tensor(x)  # pragma: no cover
        return x  # pragma: no cover

    def _w(x):
        if isinstance(x, tuple):  # pragma: no cover
            return tuple((_w(i) for i in x))  # pragma: no cover
        if isinstance(x, list):  # pragma: no cover
            return [_w(i) for i in x]  # pragma: no cover
        return array(x) if hasattr(x, "shape") else x  # pragma: no cover

    try:
        fn = getattr(mops, "reshape")
    except AttributeError:  # pragma: no cover
        try:
            fn = getattr(mnn, "reshape")
        except AttributeError:  # pragma: no cover
            try:
                fn = getattr(mrand, "reshape")
            except AttributeError:  # pragma: no cover  # pragma: no cover
                raise NotImplementedError("Missing in compiler: reshape")
    res = fn(_u(a, "a"), _u(shape, "shape"))
    return _w(res)


def divmod(a: Any, b: Any, stream: Any = None) -> Any:
    """Compute divmod.

    Args:
        a: The a argument.
        b: The b argument.
        stream: The stream argument.

    Returns:
        The result of divmod.

    """
    from zero_mlx.array import array, _to_tensor
    import ml_switcheroo.ops as mops
    import ml_switcheroo.nn as mnn
    import ml_switcheroo.random as mrand
    import types

    def _u(x, param_name=None):
        if isinstance(x, list):  # pragma: no cover
            if __import__("builtins").all(
                (isinstance(i, int) for i in x)
            ):  # pragma: no cover
                return tuple(x)  # pragma: no cover
            return [_u(i, param_name) for i in x]  # pragma: no cover
        if isinstance(x, tuple):  # pragma: no cover
            return tuple((_u(i, param_name) for i in x))  # pragma: no cover
        if hasattr(x, "_tensor"):  # pragma: no cover
            return x._tensor  # pragma: no cover
        if isinstance(x, (int, float, bool, complex)):  # pragma: no cover
            if param_name not in {
                "axis",
                "rtol",
                "n",
                "m",
                "endpoint",
                "offset",
                "a_max",
                "ddof",
                "dims",
                "source",
                "axis2",
                "stop",
                "k",
                "dtype",
                "reps",
                "a_min",
                "sparse",
                "decimals",
                "strides",
                "num",
                "max",
                "indices_or_sections",
                "shape",
                "atol",
                "axis1",
                "step",
                "axes",
                "fill_value",
                "stream",
                "destination",
                "alpha",
                "beta",
                "start",
                "equal_nan",
                "keepdims",
                "min",
                "p",
            } and "divmod" not in (
                "transpose",
                "moveaxis",
                "reshape",
                "expand_dims",
                "linspace",
                "logcumsumexp",
                "cummin",
                "cummax",
                "cumprod",
                "cumsum",
                "logsumexp",
                "all",
                "any",
                "argmin",
                "argmax",
                "prod",
                "sum",
                "mean",
                "var",
                "std",
                "max",
                "min",
                "tensordot",
            ):  # pragma: no cover
                return _to_tensor(x)  # pragma: no cover
        if type(x).__name__ == "ndarray":  # pragma: no cover
            return _to_tensor(x)  # pragma: no cover
        return x  # pragma: no cover

    def _w(x):
        if isinstance(x, tuple):  # pragma: no cover
            return tuple((_w(i) for i in x))  # pragma: no cover
        if isinstance(x, list):  # pragma: no cover
            return [_w(i) for i in x]  # pragma: no cover
        return array(x) if hasattr(x, "shape") else x  # pragma: no cover

    try:
        fn = getattr(mops, "divmod")
    except AttributeError:  # pragma: no cover
        try:
            fn = getattr(mnn, "divmod")
        except AttributeError:  # pragma: no cover
            try:
                fn = getattr(mrand, "divmod")
            except AttributeError:  # pragma: no cover  # pragma: no cover
                raise NotImplementedError("Missing in compiler: divmod")
    res = fn(_u(a, "a"), _u(b, "b"))
    return _w(res)


def logical_not(a: Any, stream: Any = None) -> Any:
    """Compute logical_not.

    Args:
        a: The a argument.
        stream: The stream argument.

    Returns:
        The result of logical_not.

    """
    from zero_mlx.array import array, _to_tensor
    import ml_switcheroo.ops as mops
    import ml_switcheroo.nn as mnn
    import ml_switcheroo.random as mrand
    import types

    def _u(x, param_name=None):
        if isinstance(x, list):  # pragma: no cover
            if __import__("builtins").all(
                (isinstance(i, int) for i in x)
            ):  # pragma: no cover
                return tuple(x)  # pragma: no cover
            return [_u(i, param_name) for i in x]  # pragma: no cover
        if isinstance(x, tuple):  # pragma: no cover
            return tuple((_u(i, param_name) for i in x))  # pragma: no cover
        if hasattr(x, "_tensor"):  # pragma: no cover
            return x._tensor  # pragma: no cover
        if isinstance(x, (int, float, bool, complex)):  # pragma: no cover
            if param_name not in {
                "axis",
                "rtol",
                "n",
                "m",
                "endpoint",
                "offset",
                "a_max",
                "ddof",
                "dims",
                "source",
                "axis2",
                "stop",
                "k",
                "dtype",
                "reps",
                "a_min",
                "sparse",
                "decimals",
                "strides",
                "num",
                "max",
                "indices_or_sections",
                "shape",
                "atol",
                "axis1",
                "step",
                "axes",
                "fill_value",
                "stream",
                "destination",
                "alpha",
                "beta",
                "start",
                "equal_nan",
                "keepdims",
                "min",
                "p",
            } and "logical_not" not in (
                "transpose",
                "moveaxis",
                "reshape",
                "expand_dims",
                "linspace",
                "logcumsumexp",
                "cummin",
                "cummax",
                "cumprod",
                "cumsum",
                "logsumexp",
                "all",
                "any",
                "argmin",
                "argmax",
                "prod",
                "sum",
                "mean",
                "var",
                "std",
                "max",
                "min",
                "tensordot",
            ):  # pragma: no cover
                return _to_tensor(x)  # pragma: no cover
        if type(x).__name__ == "ndarray":  # pragma: no cover
            return _to_tensor(x)  # pragma: no cover
        return x  # pragma: no cover

    def _w(x):
        if isinstance(x, tuple):  # pragma: no cover
            return tuple((_w(i) for i in x))  # pragma: no cover
        if isinstance(x, list):  # pragma: no cover
            return [_w(i) for i in x]  # pragma: no cover
        return array(x) if hasattr(x, "shape") else x  # pragma: no cover

    try:
        fn = getattr(mops, "logical_not")
    except AttributeError:  # pragma: no cover
        try:
            fn = getattr(mnn, "logical_not")
        except AttributeError:  # pragma: no cover
            try:
                fn = getattr(mrand, "logical_not")
            except AttributeError:  # pragma: no cover  # pragma: no cover
                raise NotImplementedError("Missing in compiler: logical_not")
    res = fn(_u(a, "a"))
    return _w(res)


def logical_and(a: Any, b: Any, stream: Any = None) -> Any:
    """Compute logical_and.

    Args:
        a: The a argument.
        b: The b argument.
        stream: The stream argument.

    Returns:
        The result of logical_and.

    """
    from zero_mlx.array import array, _to_tensor
    import ml_switcheroo.ops as mops
    import ml_switcheroo.nn as mnn
    import ml_switcheroo.random as mrand
    import types

    def _u(x, param_name=None):
        if isinstance(x, list):  # pragma: no cover
            if __import__("builtins").all(
                (isinstance(i, int) for i in x)
            ):  # pragma: no cover
                return tuple(x)  # pragma: no cover
            return [_u(i, param_name) for i in x]  # pragma: no cover
        if isinstance(x, tuple):  # pragma: no cover
            return tuple((_u(i, param_name) for i in x))  # pragma: no cover
        if hasattr(x, "_tensor"):  # pragma: no cover
            return x._tensor  # pragma: no cover
        if isinstance(x, (int, float, bool, complex)):  # pragma: no cover
            if param_name not in {
                "axis",
                "rtol",
                "n",
                "m",
                "endpoint",
                "offset",
                "a_max",
                "ddof",
                "dims",
                "source",
                "axis2",
                "stop",
                "k",
                "dtype",
                "reps",
                "a_min",
                "sparse",
                "decimals",
                "strides",
                "num",
                "max",
                "indices_or_sections",
                "shape",
                "atol",
                "axis1",
                "step",
                "axes",
                "fill_value",
                "stream",
                "destination",
                "alpha",
                "beta",
                "start",
                "equal_nan",
                "keepdims",
                "min",
                "p",
            } and "logical_and" not in (
                "transpose",
                "moveaxis",
                "reshape",
                "expand_dims",
                "linspace",
                "logcumsumexp",
                "cummin",
                "cummax",
                "cumprod",
                "cumsum",
                "logsumexp",
                "all",
                "any",
                "argmin",
                "argmax",
                "prod",
                "sum",
                "mean",
                "var",
                "std",
                "max",
                "min",
                "tensordot",
            ):  # pragma: no cover
                return _to_tensor(x)  # pragma: no cover
        if type(x).__name__ == "ndarray":  # pragma: no cover
            return _to_tensor(x)  # pragma: no cover
        return x  # pragma: no cover

    def _w(x):
        if isinstance(x, tuple):  # pragma: no cover
            return tuple((_w(i) for i in x))  # pragma: no cover
        if isinstance(x, list):  # pragma: no cover
            return [_w(i) for i in x]  # pragma: no cover
        return array(x) if hasattr(x, "shape") else x  # pragma: no cover

    try:
        fn = getattr(mops, "logical_and")
    except AttributeError:  # pragma: no cover
        try:
            fn = getattr(mnn, "logical_and")
        except AttributeError:  # pragma: no cover
            try:
                fn = getattr(mrand, "logical_and")
            except AttributeError:  # pragma: no cover  # pragma: no cover
                raise NotImplementedError("Missing in compiler: logical_and")
    res = fn(_u(a, "a"), _u(b, "b"))
    return _w(res)


def logical_or(a: Any, b: Any, stream: Any = None) -> Any:
    """Compute logical_or.

    Args:
        a: The a argument.
        b: The b argument.
        stream: The stream argument.

    Returns:
        The result of logical_or.

    """
    from zero_mlx.array import array, _to_tensor
    import ml_switcheroo.ops as mops
    import ml_switcheroo.nn as mnn
    import ml_switcheroo.random as mrand
    import types

    def _u(x, param_name=None):
        if isinstance(x, list):  # pragma: no cover
            if __import__("builtins").all(
                (isinstance(i, int) for i in x)
            ):  # pragma: no cover
                return tuple(x)  # pragma: no cover
            return [_u(i, param_name) for i in x]  # pragma: no cover
        if isinstance(x, tuple):  # pragma: no cover
            return tuple((_u(i, param_name) for i in x))  # pragma: no cover
        if hasattr(x, "_tensor"):  # pragma: no cover
            return x._tensor  # pragma: no cover
        if isinstance(x, (int, float, bool, complex)):  # pragma: no cover
            if param_name not in {
                "axis",
                "rtol",
                "n",
                "m",
                "endpoint",
                "offset",
                "a_max",
                "ddof",
                "dims",
                "source",
                "axis2",
                "stop",
                "k",
                "dtype",
                "reps",
                "a_min",
                "sparse",
                "decimals",
                "strides",
                "num",
                "max",
                "indices_or_sections",
                "shape",
                "atol",
                "axis1",
                "step",
                "axes",
                "fill_value",
                "stream",
                "destination",
                "alpha",
                "beta",
                "start",
                "equal_nan",
                "keepdims",
                "min",
                "p",
            } and "logical_or" not in (
                "transpose",
                "moveaxis",
                "reshape",
                "expand_dims",
                "linspace",
                "logcumsumexp",
                "cummin",
                "cummax",
                "cumprod",
                "cumsum",
                "logsumexp",
                "all",
                "any",
                "argmin",
                "argmax",
                "prod",
                "sum",
                "mean",
                "var",
                "std",
                "max",
                "min",
                "tensordot",
            ):  # pragma: no cover
                return _to_tensor(x)  # pragma: no cover
        if type(x).__name__ == "ndarray":  # pragma: no cover
            return _to_tensor(x)  # pragma: no cover
        return x  # pragma: no cover

    def _w(x):
        if isinstance(x, tuple):  # pragma: no cover
            return tuple((_w(i) for i in x))  # pragma: no cover
        if isinstance(x, list):  # pragma: no cover
            return [_w(i) for i in x]  # pragma: no cover
        return array(x) if hasattr(x, "shape") else x  # pragma: no cover

    try:
        fn = getattr(mops, "logical_or")
    except AttributeError:  # pragma: no cover
        try:
            fn = getattr(mnn, "logical_or")
        except AttributeError:  # pragma: no cover
            try:
                fn = getattr(mrand, "logical_or")
            except AttributeError:  # pragma: no cover  # pragma: no cover
                raise NotImplementedError("Missing in compiler: logical_or")
    res = fn(_u(a, "a"), _u(b, "b"))
    return _w(res)


def sqrt(a: Any, stream: Any = None) -> Any:
    """Compute sqrt.

    Args:
        a: The a argument.
        stream: The stream argument.

    Returns:
        The result of sqrt.

    """
    from zero_mlx.array import array, _to_tensor
    import ml_switcheroo.ops as mops
    import ml_switcheroo.nn as mnn
    import ml_switcheroo.random as mrand
    import types

    def _u(x, param_name=None):
        if isinstance(x, list):  # pragma: no cover
            if __import__("builtins").all(
                (isinstance(i, int) for i in x)
            ):  # pragma: no cover
                return tuple(x)  # pragma: no cover
            return [_u(i, param_name) for i in x]  # pragma: no cover
        if isinstance(x, tuple):  # pragma: no cover
            return tuple((_u(i, param_name) for i in x))  # pragma: no cover
        if hasattr(x, "_tensor"):  # pragma: no cover
            return x._tensor  # pragma: no cover
        if isinstance(x, (int, float, bool, complex)):  # pragma: no cover
            if param_name not in {
                "axis",
                "rtol",
                "n",
                "m",
                "endpoint",
                "offset",
                "a_max",
                "ddof",
                "dims",
                "source",
                "axis2",
                "stop",
                "k",
                "dtype",
                "reps",
                "a_min",
                "sparse",
                "decimals",
                "strides",
                "num",
                "max",
                "indices_or_sections",
                "shape",
                "atol",
                "axis1",
                "step",
                "axes",
                "fill_value",
                "stream",
                "destination",
                "alpha",
                "beta",
                "start",
                "equal_nan",
                "keepdims",
                "min",
                "p",
            } and "sqrt" not in (
                "transpose",
                "moveaxis",
                "reshape",
                "expand_dims",
                "linspace",
                "logcumsumexp",
                "cummin",
                "cummax",
                "cumprod",
                "cumsum",
                "logsumexp",
                "all",
                "any",
                "argmin",
                "argmax",
                "prod",
                "sum",
                "mean",
                "var",
                "std",
                "max",
                "min",
                "tensordot",
            ):  # pragma: no cover
                return _to_tensor(x)  # pragma: no cover
        if type(x).__name__ == "ndarray":  # pragma: no cover
            return _to_tensor(x)  # pragma: no cover
        return x  # pragma: no cover

    def _w(x):
        if isinstance(x, tuple):  # pragma: no cover
            return tuple((_w(i) for i in x))  # pragma: no cover
        if isinstance(x, list):  # pragma: no cover
            return [_w(i) for i in x]  # pragma: no cover
        return array(x) if hasattr(x, "shape") else x  # pragma: no cover

    try:
        fn = getattr(mops, "sqrt")
    except AttributeError:  # pragma: no cover
        try:
            fn = getattr(mnn, "sqrt")
        except AttributeError:  # pragma: no cover
            try:
                fn = getattr(mrand, "sqrt")
            except AttributeError:  # pragma: no cover  # pragma: no cover
                raise NotImplementedError("Missing in compiler: sqrt")
    res = fn(_u(a, "a"))
    return _w(res)


def abs(a: Any, stream: Any = None) -> Any:
    """Compute abs.

    Args:
        a: The a argument.
        stream: The stream argument.

    Returns:
        The result of abs.

    """
    from zero_mlx.array import array, _to_tensor
    import ml_switcheroo.ops as mops
    import ml_switcheroo.nn as mnn
    import ml_switcheroo.random as mrand
    import types

    def _u(x, param_name=None):
        if isinstance(x, list):  # pragma: no cover
            if __import__("builtins").all(
                (isinstance(i, int) for i in x)
            ):  # pragma: no cover
                return tuple(x)  # pragma: no cover
            return [_u(i, param_name) for i in x]  # pragma: no cover
        if isinstance(x, tuple):  # pragma: no cover
            return tuple((_u(i, param_name) for i in x))  # pragma: no cover
        if hasattr(x, "_tensor"):  # pragma: no cover
            return x._tensor  # pragma: no cover
        if isinstance(x, (int, float, bool, complex)):  # pragma: no cover
            if param_name not in {
                "axis",
                "rtol",
                "n",
                "m",
                "endpoint",
                "offset",
                "a_max",
                "ddof",
                "dims",
                "source",
                "axis2",
                "stop",
                "k",
                "dtype",
                "reps",
                "a_min",
                "sparse",
                "decimals",
                "strides",
                "num",
                "max",
                "indices_or_sections",
                "shape",
                "atol",
                "axis1",
                "step",
                "axes",
                "fill_value",
                "stream",
                "destination",
                "alpha",
                "beta",
                "start",
                "equal_nan",
                "keepdims",
                "min",
                "p",
            } and "abs" not in (
                "transpose",
                "moveaxis",
                "reshape",
                "expand_dims",
                "linspace",
                "logcumsumexp",
                "cummin",
                "cummax",
                "cumprod",
                "cumsum",
                "logsumexp",
                "all",
                "any",
                "argmin",
                "argmax",
                "prod",
                "sum",
                "mean",
                "var",
                "std",
                "max",
                "min",
                "tensordot",
            ):  # pragma: no cover
                return _to_tensor(x)  # pragma: no cover
        if type(x).__name__ == "ndarray":  # pragma: no cover
            return _to_tensor(x)  # pragma: no cover
        return x  # pragma: no cover

    def _w(x):
        if isinstance(x, tuple):  # pragma: no cover
            return tuple((_w(i) for i in x))  # pragma: no cover
        if isinstance(x, list):  # pragma: no cover
            return [_w(i) for i in x]  # pragma: no cover
        return array(x) if hasattr(x, "shape") else x  # pragma: no cover

    try:
        fn = getattr(mops, "abs")
    except AttributeError:  # pragma: no cover
        try:
            fn = getattr(mnn, "abs")
        except AttributeError:  # pragma: no cover
            try:
                fn = getattr(mrand, "abs")
            except AttributeError:  # pragma: no cover  # pragma: no cover
                raise NotImplementedError("Missing in compiler: abs")
    res = fn(_u(a, "a"))
    return _w(res)


def negative(a: Any, stream: Any = None) -> Any:
    """Compute negative.

    Args:
        a: The a argument.
        stream: The stream argument.

    Returns:
        The result of negative.

    """
    from zero_mlx.array import array, _to_tensor
    import ml_switcheroo.ops as mops
    import ml_switcheroo.nn as mnn
    import ml_switcheroo.random as mrand
    import types

    def _u(x, param_name=None):
        if isinstance(x, list):  # pragma: no cover
            if __import__("builtins").all(
                (isinstance(i, int) for i in x)
            ):  # pragma: no cover
                return tuple(x)  # pragma: no cover
            return [_u(i, param_name) for i in x]  # pragma: no cover
        if isinstance(x, tuple):  # pragma: no cover
            return tuple((_u(i, param_name) for i in x))  # pragma: no cover
        if hasattr(x, "_tensor"):  # pragma: no cover
            return x._tensor  # pragma: no cover
        if isinstance(x, (int, float, bool, complex)):  # pragma: no cover
            if param_name not in {
                "axis",
                "rtol",
                "n",
                "m",
                "endpoint",
                "offset",
                "a_max",
                "ddof",
                "dims",
                "source",
                "axis2",
                "stop",
                "k",
                "dtype",
                "reps",
                "a_min",
                "sparse",
                "decimals",
                "strides",
                "num",
                "max",
                "indices_or_sections",
                "shape",
                "atol",
                "axis1",
                "step",
                "axes",
                "fill_value",
                "stream",
                "destination",
                "alpha",
                "beta",
                "start",
                "equal_nan",
                "keepdims",
                "min",
                "p",
            } and "negative" not in (
                "transpose",
                "moveaxis",
                "reshape",
                "expand_dims",
                "linspace",
                "logcumsumexp",
                "cummin",
                "cummax",
                "cumprod",
                "cumsum",
                "logsumexp",
                "all",
                "any",
                "argmin",
                "argmax",
                "prod",
                "sum",
                "mean",
                "var",
                "std",
                "max",
                "min",
                "tensordot",
            ):  # pragma: no cover
                return _to_tensor(x)  # pragma: no cover
        if type(x).__name__ == "ndarray":  # pragma: no cover
            return _to_tensor(x)  # pragma: no cover
        return x  # pragma: no cover

    def _w(x):
        if isinstance(x, tuple):  # pragma: no cover
            return tuple((_w(i) for i in x))  # pragma: no cover
        if isinstance(x, list):  # pragma: no cover
            return [_w(i) for i in x]  # pragma: no cover
        return array(x) if hasattr(x, "shape") else x  # pragma: no cover

    try:
        fn = getattr(mops, "negative")
    except AttributeError:  # pragma: no cover
        try:
            fn = getattr(mnn, "negative")
        except AttributeError:  # pragma: no cover
            try:
                fn = getattr(mrand, "negative")
            except AttributeError:  # pragma: no cover  # pragma: no cover
                raise NotImplementedError("Missing in compiler: negative")
    res = fn(_u(a, "a"))
    return _w(res)


def exp(a: Any, stream: Any = None) -> Any:
    """Compute exp.

    Args:
        a: The a argument.
        stream: The stream argument.

    Returns:
        The result of exp.

    """
    from zero_mlx.array import array, _to_tensor
    import ml_switcheroo.ops as mops
    import ml_switcheroo.nn as mnn
    import ml_switcheroo.random as mrand
    import types

    def _u(x, param_name=None):
        if isinstance(x, list):  # pragma: no cover
            if __import__("builtins").all(
                (isinstance(i, int) for i in x)
            ):  # pragma: no cover
                return tuple(x)  # pragma: no cover
            return [_u(i, param_name) for i in x]  # pragma: no cover
        if isinstance(x, tuple):  # pragma: no cover
            return tuple((_u(i, param_name) for i in x))  # pragma: no cover
        if hasattr(x, "_tensor"):  # pragma: no cover
            return x._tensor  # pragma: no cover
        if isinstance(x, (int, float, bool, complex)):  # pragma: no cover
            if param_name not in {
                "axis",
                "rtol",
                "n",
                "m",
                "endpoint",
                "offset",
                "a_max",
                "ddof",
                "dims",
                "source",
                "axis2",
                "stop",
                "k",
                "dtype",
                "reps",
                "a_min",
                "sparse",
                "decimals",
                "strides",
                "num",
                "max",
                "indices_or_sections",
                "shape",
                "atol",
                "axis1",
                "step",
                "axes",
                "fill_value",
                "stream",
                "destination",
                "alpha",
                "beta",
                "start",
                "equal_nan",
                "keepdims",
                "min",
                "p",
            } and "exp" not in (
                "transpose",
                "moveaxis",
                "reshape",
                "expand_dims",
                "linspace",
                "logcumsumexp",
                "cummin",
                "cummax",
                "cumprod",
                "cumsum",
                "logsumexp",
                "all",
                "any",
                "argmin",
                "argmax",
                "prod",
                "sum",
                "mean",
                "var",
                "std",
                "max",
                "min",
                "tensordot",
            ):  # pragma: no cover
                return _to_tensor(x)  # pragma: no cover
        if type(x).__name__ == "ndarray":  # pragma: no cover
            return _to_tensor(x)  # pragma: no cover
        return x  # pragma: no cover

    def _w(x):
        if isinstance(x, tuple):  # pragma: no cover
            return tuple((_w(i) for i in x))  # pragma: no cover
        if isinstance(x, list):  # pragma: no cover
            return [_w(i) for i in x]  # pragma: no cover
        return array(x) if hasattr(x, "shape") else x  # pragma: no cover

    try:
        fn = getattr(mops, "exp")
    except AttributeError:  # pragma: no cover
        try:
            fn = getattr(mnn, "exp")
        except AttributeError:  # pragma: no cover
            try:
                fn = getattr(mrand, "exp")
            except AttributeError:  # pragma: no cover  # pragma: no cover
                raise NotImplementedError("Missing in compiler: exp")
    res = fn(_u(a, "a"))
    return _w(res)


def rsqrt(a: Any, stream: Any = None) -> Any:
    """Compute rsqrt.

    Args:
        a: The a argument.
        stream: The stream argument.

    Returns:
        The result of rsqrt.

    """
    from zero_mlx.array import array, _to_tensor
    import ml_switcheroo.ops as mops
    import ml_switcheroo.nn as mnn
    import ml_switcheroo.random as mrand
    import types

    def _u(x, param_name=None):
        if isinstance(x, list):  # pragma: no cover
            if __import__("builtins").all(
                (isinstance(i, int) for i in x)
            ):  # pragma: no cover
                return tuple(x)  # pragma: no cover
            return [_u(i, param_name) for i in x]  # pragma: no cover
        if isinstance(x, tuple):  # pragma: no cover
            return tuple((_u(i, param_name) for i in x))  # pragma: no cover
        if hasattr(x, "_tensor"):  # pragma: no cover
            return x._tensor  # pragma: no cover
        if isinstance(x, (int, float, bool, complex)):  # pragma: no cover
            if param_name not in {
                "axis",
                "rtol",
                "n",
                "m",
                "endpoint",
                "offset",
                "a_max",
                "ddof",
                "dims",
                "source",
                "axis2",
                "stop",
                "k",
                "dtype",
                "reps",
                "a_min",
                "sparse",
                "decimals",
                "strides",
                "num",
                "max",
                "indices_or_sections",
                "shape",
                "atol",
                "axis1",
                "step",
                "axes",
                "fill_value",
                "stream",
                "destination",
                "alpha",
                "beta",
                "start",
                "equal_nan",
                "keepdims",
                "min",
                "p",
            } and "rsqrt" not in (
                "transpose",
                "moveaxis",
                "reshape",
                "expand_dims",
                "linspace",
                "logcumsumexp",
                "cummin",
                "cummax",
                "cumprod",
                "cumsum",
                "logsumexp",
                "all",
                "any",
                "argmin",
                "argmax",
                "prod",
                "sum",
                "mean",
                "var",
                "std",
                "max",
                "min",
                "tensordot",
            ):  # pragma: no cover
                return _to_tensor(x)  # pragma: no cover
        if type(x).__name__ == "ndarray":  # pragma: no cover
            return _to_tensor(x)  # pragma: no cover
        return x  # pragma: no cover

    def _w(x):
        if isinstance(x, tuple):  # pragma: no cover
            return tuple((_w(i) for i in x))  # pragma: no cover
        if isinstance(x, list):  # pragma: no cover
            return [_w(i) for i in x]  # pragma: no cover
        return array(x) if hasattr(x, "shape") else x  # pragma: no cover

    try:
        fn = getattr(mops, "rsqrt")
    except AttributeError:  # pragma: no cover
        try:
            fn = getattr(mnn, "rsqrt")
        except AttributeError:  # pragma: no cover
            try:
                fn = getattr(mrand, "rsqrt")
            except AttributeError:  # pragma: no cover  # pragma: no cover
                raise NotImplementedError("Missing in compiler: rsqrt")
    res = fn(_u(a, "a"))
    return _w(res)


def add(a: Any, b: Any, stream: Any = None) -> Any:
    """Compute add.

    Args:
        a: The a argument.
        b: The b argument.
        stream: The stream argument.

    Returns:
        The result of add.

    """
    from zero_mlx.array import array, _to_tensor
    import ml_switcheroo.ops as mops
    import ml_switcheroo.nn as mnn
    import ml_switcheroo.random as mrand
    import types

    def _u(x, param_name=None):
        if isinstance(x, list):  # pragma: no cover
            if __import__("builtins").all(
                (isinstance(i, int) for i in x)
            ):  # pragma: no cover
                return tuple(x)  # pragma: no cover
            return [_u(i, param_name) for i in x]  # pragma: no cover
        if isinstance(x, tuple):  # pragma: no cover
            return tuple((_u(i, param_name) for i in x))  # pragma: no cover
        if hasattr(x, "_tensor"):  # pragma: no cover
            return x._tensor  # pragma: no cover
        if isinstance(x, (int, float, bool, complex)):  # pragma: no cover
            if param_name not in {
                "axis",
                "rtol",
                "n",
                "m",
                "endpoint",
                "offset",
                "a_max",
                "ddof",
                "dims",
                "source",
                "axis2",
                "stop",
                "k",
                "dtype",
                "reps",
                "a_min",
                "sparse",
                "decimals",
                "strides",
                "num",
                "max",
                "indices_or_sections",
                "shape",
                "atol",
                "axis1",
                "step",
                "axes",
                "fill_value",
                "stream",
                "destination",
                "alpha",
                "beta",
                "start",
                "equal_nan",
                "keepdims",
                "min",
                "p",
            } and "add" not in (
                "transpose",
                "moveaxis",
                "reshape",
                "expand_dims",
                "linspace",
                "logcumsumexp",
                "cummin",
                "cummax",
                "cumprod",
                "cumsum",
                "logsumexp",
                "all",
                "any",
                "argmin",
                "argmax",
                "prod",
                "sum",
                "mean",
                "var",
                "std",
                "max",
                "min",
                "tensordot",
            ):  # pragma: no cover
                return _to_tensor(x)  # pragma: no cover
        if type(x).__name__ == "ndarray":  # pragma: no cover
            return _to_tensor(x)  # pragma: no cover
        return x  # pragma: no cover

    def _w(x):
        if isinstance(x, tuple):  # pragma: no cover
            return tuple((_w(i) for i in x))  # pragma: no cover
        if isinstance(x, list):  # pragma: no cover
            return [_w(i) for i in x]  # pragma: no cover
        return array(x) if hasattr(x, "shape") else x  # pragma: no cover

    try:
        fn = getattr(mops, "add")
    except AttributeError:  # pragma: no cover
        try:
            fn = getattr(mnn, "add")
        except AttributeError:  # pragma: no cover
            try:
                fn = getattr(mrand, "add")
            except AttributeError:  # pragma: no cover  # pragma: no cover
                raise NotImplementedError("Missing in compiler: add")
    res = fn(_u(a, "a"), _u(b, "b"))
    return _w(res)


def subtract(a: Any, b: Any, stream: Any = None) -> Any:
    """Compute subtract.

    Args:
        a: The a argument.
        b: The b argument.
        stream: The stream argument.

    Returns:
        The result of subtract.

    """
    from zero_mlx.array import array, _to_tensor
    import ml_switcheroo.ops as mops
    import ml_switcheroo.nn as mnn
    import ml_switcheroo.random as mrand
    import types

    def _u(x, param_name=None):
        if isinstance(x, list):  # pragma: no cover
            if __import__("builtins").all(
                (isinstance(i, int) for i in x)
            ):  # pragma: no cover
                return tuple(x)  # pragma: no cover
            return [_u(i, param_name) for i in x]  # pragma: no cover
        if isinstance(x, tuple):  # pragma: no cover
            return tuple((_u(i, param_name) for i in x))  # pragma: no cover
        if hasattr(x, "_tensor"):  # pragma: no cover
            return x._tensor  # pragma: no cover
        if isinstance(x, (int, float, bool, complex)):  # pragma: no cover
            if param_name not in {
                "axis",
                "rtol",
                "n",
                "m",
                "endpoint",
                "offset",
                "a_max",
                "ddof",
                "dims",
                "source",
                "axis2",
                "stop",
                "k",
                "dtype",
                "reps",
                "a_min",
                "sparse",
                "decimals",
                "strides",
                "num",
                "max",
                "indices_or_sections",
                "shape",
                "atol",
                "axis1",
                "step",
                "axes",
                "fill_value",
                "stream",
                "destination",
                "alpha",
                "beta",
                "start",
                "equal_nan",
                "keepdims",
                "min",
                "p",
            } and "subtract" not in (
                "transpose",
                "moveaxis",
                "reshape",
                "expand_dims",
                "linspace",
                "logcumsumexp",
                "cummin",
                "cummax",
                "cumprod",
                "cumsum",
                "logsumexp",
                "all",
                "any",
                "argmin",
                "argmax",
                "prod",
                "sum",
                "mean",
                "var",
                "std",
                "max",
                "min",
                "tensordot",
            ):  # pragma: no cover
                return _to_tensor(x)  # pragma: no cover
        if type(x).__name__ == "ndarray":  # pragma: no cover
            return _to_tensor(x)  # pragma: no cover
        return x  # pragma: no cover

    def _w(x):
        if isinstance(x, tuple):  # pragma: no cover
            return tuple((_w(i) for i in x))  # pragma: no cover
        if isinstance(x, list):  # pragma: no cover
            return [_w(i) for i in x]  # pragma: no cover
        return array(x) if hasattr(x, "shape") else x  # pragma: no cover

    try:
        fn = getattr(mops, "subtract")
    except AttributeError:  # pragma: no cover
        try:
            fn = getattr(mnn, "subtract")
        except AttributeError:  # pragma: no cover
            try:
                fn = getattr(mrand, "subtract")
            except AttributeError:  # pragma: no cover  # pragma: no cover
                raise NotImplementedError("Missing in compiler: subtract")
    res = fn(_u(a, "a"), _u(b, "b"))
    return _w(res)


def multiply(a: Any, b: Any, stream: Any = None) -> Any:
    """Compute multiply.

    Args:
        a: The a argument.
        b: The b argument.
        stream: The stream argument.

    Returns:
        The result of multiply.

    """
    from zero_mlx.array import array, _to_tensor
    import ml_switcheroo.ops as mops
    import ml_switcheroo.nn as mnn
    import ml_switcheroo.random as mrand
    import types

    def _u(x, param_name=None):
        if isinstance(x, list):  # pragma: no cover
            if __import__("builtins").all(
                (isinstance(i, int) for i in x)
            ):  # pragma: no cover
                return tuple(x)  # pragma: no cover
            return [_u(i, param_name) for i in x]  # pragma: no cover
        if isinstance(x, tuple):  # pragma: no cover
            return tuple((_u(i, param_name) for i in x))  # pragma: no cover
        if hasattr(x, "_tensor"):  # pragma: no cover
            return x._tensor  # pragma: no cover
        if isinstance(x, (int, float, bool, complex)):  # pragma: no cover
            if param_name not in {
                "axis",
                "rtol",
                "n",
                "m",
                "endpoint",
                "offset",
                "a_max",
                "ddof",
                "dims",
                "source",
                "axis2",
                "stop",
                "k",
                "dtype",
                "reps",
                "a_min",
                "sparse",
                "decimals",
                "strides",
                "num",
                "max",
                "indices_or_sections",
                "shape",
                "atol",
                "axis1",
                "step",
                "axes",
                "fill_value",
                "stream",
                "destination",
                "alpha",
                "beta",
                "start",
                "equal_nan",
                "keepdims",
                "min",
                "p",
            } and "multiply" not in (
                "transpose",
                "moveaxis",
                "reshape",
                "expand_dims",
                "linspace",
                "logcumsumexp",
                "cummin",
                "cummax",
                "cumprod",
                "cumsum",
                "logsumexp",
                "all",
                "any",
                "argmin",
                "argmax",
                "prod",
                "sum",
                "mean",
                "var",
                "std",
                "max",
                "min",
                "tensordot",
            ):  # pragma: no cover
                return _to_tensor(x)  # pragma: no cover
        if type(x).__name__ == "ndarray":  # pragma: no cover
            return _to_tensor(x)  # pragma: no cover
        return x  # pragma: no cover

    def _w(x):
        if isinstance(x, tuple):  # pragma: no cover
            return tuple((_w(i) for i in x))  # pragma: no cover
        if isinstance(x, list):  # pragma: no cover
            return [_w(i) for i in x]  # pragma: no cover
        return array(x) if hasattr(x, "shape") else x  # pragma: no cover

    try:
        fn = getattr(mops, "multiply")
    except AttributeError:  # pragma: no cover
        try:
            fn = getattr(mnn, "multiply")
        except AttributeError:  # pragma: no cover
            try:
                fn = getattr(mrand, "multiply")
            except AttributeError:  # pragma: no cover  # pragma: no cover
                raise NotImplementedError("Missing in compiler: multiply")
    res = fn(_u(a, "a"), _u(b, "b"))
    return _w(res)


def divide(a: Any, b: Any, stream: Any = None) -> Any:
    """Compute divide.

    Args:
        a: The a argument.
        b: The b argument.
        stream: The stream argument.

    Returns:
        The result of divide.

    """
    from zero_mlx.array import array, _to_tensor
    import ml_switcheroo.ops as mops
    import ml_switcheroo.nn as mnn
    import ml_switcheroo.random as mrand
    import types

    def _u(x, param_name=None):
        if isinstance(x, list):  # pragma: no cover
            if __import__("builtins").all(
                (isinstance(i, int) for i in x)
            ):  # pragma: no cover
                return tuple(x)  # pragma: no cover
            return [_u(i, param_name) for i in x]  # pragma: no cover
        if isinstance(x, tuple):  # pragma: no cover
            return tuple((_u(i, param_name) for i in x))  # pragma: no cover
        if hasattr(x, "_tensor"):  # pragma: no cover
            return x._tensor  # pragma: no cover
        if isinstance(x, (int, float, bool, complex)):  # pragma: no cover
            if param_name not in {
                "axis",
                "rtol",
                "n",
                "m",
                "endpoint",
                "offset",
                "a_max",
                "ddof",
                "dims",
                "source",
                "axis2",
                "stop",
                "k",
                "dtype",
                "reps",
                "a_min",
                "sparse",
                "decimals",
                "strides",
                "num",
                "max",
                "indices_or_sections",
                "shape",
                "atol",
                "axis1",
                "step",
                "axes",
                "fill_value",
                "stream",
                "destination",
                "alpha",
                "beta",
                "start",
                "equal_nan",
                "keepdims",
                "min",
                "p",
            } and "divide" not in (
                "transpose",
                "moveaxis",
                "reshape",
                "expand_dims",
                "linspace",
                "logcumsumexp",
                "cummin",
                "cummax",
                "cumprod",
                "cumsum",
                "logsumexp",
                "all",
                "any",
                "argmin",
                "argmax",
                "prod",
                "sum",
                "mean",
                "var",
                "std",
                "max",
                "min",
                "tensordot",
            ):  # pragma: no cover
                return _to_tensor(x)  # pragma: no cover
        if type(x).__name__ == "ndarray":  # pragma: no cover
            return _to_tensor(x)  # pragma: no cover
        return x  # pragma: no cover

    def _w(x):
        if isinstance(x, tuple):  # pragma: no cover
            return tuple((_w(i) for i in x))  # pragma: no cover
        if isinstance(x, list):  # pragma: no cover
            return [_w(i) for i in x]  # pragma: no cover
        return array(x) if hasattr(x, "shape") else x  # pragma: no cover

    try:
        fn = getattr(mops, "divide")
    except AttributeError:  # pragma: no cover
        try:
            fn = getattr(mnn, "divide")
        except AttributeError:  # pragma: no cover
            try:
                fn = getattr(mrand, "divide")
            except AttributeError:  # pragma: no cover  # pragma: no cover
                raise NotImplementedError("Missing in compiler: divide")
    res = fn(_u(a, "a"), _u(b, "b"))
    return _w(res)


def matmul(a: Any, b: Any, stream: Any = None) -> Any:
    """Compute matmul.

    Args:
        a: The a argument.
        b: The b argument.
        stream: The stream argument.

    Returns:
        The result of matmul.

    """
    from zero_mlx.array import array, _to_tensor
    import ml_switcheroo.ops as mops
    import ml_switcheroo.nn as mnn
    import ml_switcheroo.random as mrand
    import types

    def _u(x, param_name=None):
        if isinstance(x, list):  # pragma: no cover
            if __import__("builtins").all(
                (isinstance(i, int) for i in x)
            ):  # pragma: no cover
                return tuple(x)  # pragma: no cover
            return [_u(i, param_name) for i in x]  # pragma: no cover
        if isinstance(x, tuple):  # pragma: no cover
            return tuple((_u(i, param_name) for i in x))  # pragma: no cover
        if hasattr(x, "_tensor"):  # pragma: no cover
            return x._tensor  # pragma: no cover
        if isinstance(x, (int, float, bool, complex)):  # pragma: no cover
            if param_name not in {
                "axis",
                "rtol",
                "n",
                "m",
                "endpoint",
                "offset",
                "a_max",
                "ddof",
                "dims",
                "source",
                "axis2",
                "stop",
                "k",
                "dtype",
                "reps",
                "a_min",
                "sparse",
                "decimals",
                "strides",
                "num",
                "max",
                "indices_or_sections",
                "shape",
                "atol",
                "axis1",
                "step",
                "axes",
                "fill_value",
                "stream",
                "destination",
                "alpha",
                "beta",
                "start",
                "equal_nan",
                "keepdims",
                "min",
                "p",
            } and "matmul" not in (
                "transpose",
                "moveaxis",
                "reshape",
                "expand_dims",
                "linspace",
                "logcumsumexp",
                "cummin",
                "cummax",
                "cumprod",
                "cumsum",
                "logsumexp",
                "all",
                "any",
                "argmin",
                "argmax",
                "prod",
                "sum",
                "mean",
                "var",
                "std",
                "max",
                "min",
                "tensordot",
            ):  # pragma: no cover
                return _to_tensor(x)  # pragma: no cover
        if type(x).__name__ == "ndarray":  # pragma: no cover
            return _to_tensor(x)  # pragma: no cover
        return x  # pragma: no cover

    def _w(x):
        if isinstance(x, tuple):  # pragma: no cover
            return tuple((_w(i) for i in x))  # pragma: no cover
        if isinstance(x, list):  # pragma: no cover
            return [_w(i) for i in x]  # pragma: no cover
        return array(x) if hasattr(x, "shape") else x  # pragma: no cover

    try:
        fn = getattr(mops, "matmul")
    except AttributeError:  # pragma: no cover
        try:
            fn = getattr(mnn, "matmul")
        except AttributeError:  # pragma: no cover
            try:
                fn = getattr(mrand, "matmul")
            except AttributeError:  # pragma: no cover  # pragma: no cover
                raise NotImplementedError("Missing in compiler: matmul")
    res = fn(_u(a, "a"), _u(b, "b"))
    return _w(res)


def reciprocal(a: Any, stream: Any = None) -> Any:
    """Compute reciprocal.

    Args:
        a: The a argument.
        stream: The stream argument.

    Returns:
        The result of reciprocal.

    """
    from zero_mlx.array import array, _to_tensor
    import ml_switcheroo.ops as mops
    import ml_switcheroo.nn as mnn
    import ml_switcheroo.random as mrand
    import types

    def _u(x, param_name=None):
        if isinstance(x, list):  # pragma: no cover
            if __import__("builtins").all(
                (isinstance(i, int) for i in x)
            ):  # pragma: no cover
                return tuple(x)  # pragma: no cover
            return [_u(i, param_name) for i in x]  # pragma: no cover
        if isinstance(x, tuple):  # pragma: no cover
            return tuple((_u(i, param_name) for i in x))  # pragma: no cover
        if hasattr(x, "_tensor"):  # pragma: no cover
            return x._tensor  # pragma: no cover
        if isinstance(x, (int, float, bool, complex)):  # pragma: no cover
            if param_name not in {
                "axis",
                "rtol",
                "n",
                "m",
                "endpoint",
                "offset",
                "a_max",
                "ddof",
                "dims",
                "source",
                "axis2",
                "stop",
                "k",
                "dtype",
                "reps",
                "a_min",
                "sparse",
                "decimals",
                "strides",
                "num",
                "max",
                "indices_or_sections",
                "shape",
                "atol",
                "axis1",
                "step",
                "axes",
                "fill_value",
                "stream",
                "destination",
                "alpha",
                "beta",
                "start",
                "equal_nan",
                "keepdims",
                "min",
                "p",
            } and "reciprocal" not in (
                "transpose",
                "moveaxis",
                "reshape",
                "expand_dims",
                "linspace",
                "logcumsumexp",
                "cummin",
                "cummax",
                "cumprod",
                "cumsum",
                "logsumexp",
                "all",
                "any",
                "argmin",
                "argmax",
                "prod",
                "sum",
                "mean",
                "var",
                "std",
                "max",
                "min",
                "tensordot",
            ):  # pragma: no cover
                return _to_tensor(x)  # pragma: no cover
        if type(x).__name__ == "ndarray":  # pragma: no cover
            return _to_tensor(x)  # pragma: no cover
        return x  # pragma: no cover

    def _w(x):
        if isinstance(x, tuple):  # pragma: no cover
            return tuple((_w(i) for i in x))  # pragma: no cover
        if isinstance(x, list):  # pragma: no cover
            return [_w(i) for i in x]  # pragma: no cover
        return array(x) if hasattr(x, "shape") else x  # pragma: no cover

    try:
        fn = getattr(mops, "reciprocal")
    except AttributeError:  # pragma: no cover
        try:
            fn = getattr(mnn, "reciprocal")
        except AttributeError:  # pragma: no cover
            try:
                fn = getattr(mrand, "reciprocal")
            except AttributeError:  # pragma: no cover  # pragma: no cover
                raise NotImplementedError("Missing in compiler: reciprocal")
    res = fn(_u(a, "a"))
    return _w(res)


def log(a: Any, stream: Any = None) -> Any:
    """Compute log.

    Args:
        a: The a argument.
        stream: The stream argument.

    Returns:
        The result of log.

    """
    from zero_mlx.array import array, _to_tensor
    import ml_switcheroo.ops as mops
    import ml_switcheroo.nn as mnn
    import ml_switcheroo.random as mrand
    import types

    def _u(x, param_name=None):
        if isinstance(x, list):  # pragma: no cover
            if __import__("builtins").all(
                (isinstance(i, int) for i in x)
            ):  # pragma: no cover
                return tuple(x)  # pragma: no cover
            return [_u(i, param_name) for i in x]  # pragma: no cover
        if isinstance(x, tuple):  # pragma: no cover
            return tuple((_u(i, param_name) for i in x))  # pragma: no cover
        if hasattr(x, "_tensor"):  # pragma: no cover
            return x._tensor  # pragma: no cover
        if isinstance(x, (int, float, bool, complex)):  # pragma: no cover
            if param_name not in {
                "axis",
                "rtol",
                "n",
                "m",
                "endpoint",
                "offset",
                "a_max",
                "ddof",
                "dims",
                "source",
                "axis2",
                "stop",
                "k",
                "dtype",
                "reps",
                "a_min",
                "sparse",
                "decimals",
                "strides",
                "num",
                "max",
                "indices_or_sections",
                "shape",
                "atol",
                "axis1",
                "step",
                "axes",
                "fill_value",
                "stream",
                "destination",
                "alpha",
                "beta",
                "start",
                "equal_nan",
                "keepdims",
                "min",
                "p",
            } and "log" not in (
                "transpose",
                "moveaxis",
                "reshape",
                "expand_dims",
                "linspace",
                "logcumsumexp",
                "cummin",
                "cummax",
                "cumprod",
                "cumsum",
                "logsumexp",
                "all",
                "any",
                "argmin",
                "argmax",
                "prod",
                "sum",
                "mean",
                "var",
                "std",
                "max",
                "min",
                "tensordot",
            ):  # pragma: no cover
                return _to_tensor(x)  # pragma: no cover
        if type(x).__name__ == "ndarray":  # pragma: no cover
            return _to_tensor(x)  # pragma: no cover
        return x  # pragma: no cover

    def _w(x):
        if isinstance(x, tuple):  # pragma: no cover
            return tuple((_w(i) for i in x))  # pragma: no cover
        if isinstance(x, list):  # pragma: no cover
            return [_w(i) for i in x]  # pragma: no cover
        return array(x) if hasattr(x, "shape") else x  # pragma: no cover

    try:
        fn = getattr(mops, "log")
    except AttributeError:  # pragma: no cover
        try:
            fn = getattr(mnn, "log")
        except AttributeError:  # pragma: no cover
            try:
                fn = getattr(mrand, "log")
            except AttributeError:  # pragma: no cover  # pragma: no cover
                raise NotImplementedError("Missing in compiler: log")
    res = fn(_u(a, "a"))
    return _w(res)


def maximum(a: Any, b: Any, stream: Any = None) -> Any:
    """Compute maximum.

    Args:
        a: The a argument.
        b: The b argument.
        stream: The stream argument.

    Returns:
        The result of maximum.

    """
    from zero_mlx.array import array, _to_tensor
    import ml_switcheroo.ops as mops
    import ml_switcheroo.nn as mnn
    import ml_switcheroo.random as mrand
    import types

    def _u(x, param_name=None):
        if isinstance(x, list):  # pragma: no cover
            if __import__("builtins").all(
                (isinstance(i, int) for i in x)
            ):  # pragma: no cover
                return tuple(x)  # pragma: no cover
            return [_u(i, param_name) for i in x]  # pragma: no cover
        if isinstance(x, tuple):  # pragma: no cover
            return tuple((_u(i, param_name) for i in x))  # pragma: no cover
        if hasattr(x, "_tensor"):  # pragma: no cover
            return x._tensor  # pragma: no cover
        if isinstance(x, (int, float, bool, complex)):  # pragma: no cover
            if param_name not in {
                "axis",
                "rtol",
                "n",
                "m",
                "endpoint",
                "offset",
                "a_max",
                "ddof",
                "dims",
                "source",
                "axis2",
                "stop",
                "k",
                "dtype",
                "reps",
                "a_min",
                "sparse",
                "decimals",
                "strides",
                "num",
                "max",
                "indices_or_sections",
                "shape",
                "atol",
                "axis1",
                "step",
                "axes",
                "fill_value",
                "stream",
                "destination",
                "alpha",
                "beta",
                "start",
                "equal_nan",
                "keepdims",
                "min",
                "p",
            } and "maximum" not in (
                "transpose",
                "moveaxis",
                "reshape",
                "expand_dims",
                "linspace",
                "logcumsumexp",
                "cummin",
                "cummax",
                "cumprod",
                "cumsum",
                "logsumexp",
                "all",
                "any",
                "argmin",
                "argmax",
                "prod",
                "sum",
                "mean",
                "var",
                "std",
                "max",
                "min",
                "tensordot",
            ):  # pragma: no cover
                return _to_tensor(x)  # pragma: no cover
        if type(x).__name__ == "ndarray":  # pragma: no cover
            return _to_tensor(x)  # pragma: no cover
        return x  # pragma: no cover

    def _w(x):
        if isinstance(x, tuple):  # pragma: no cover
            return tuple((_w(i) for i in x))  # pragma: no cover
        if isinstance(x, list):  # pragma: no cover
            return [_w(i) for i in x]  # pragma: no cover
        return array(x) if hasattr(x, "shape") else x  # pragma: no cover

    try:
        fn = getattr(mops, "maximum")
    except AttributeError:  # pragma: no cover
        try:
            fn = getattr(mnn, "maximum")
        except AttributeError:  # pragma: no cover
            try:
                fn = getattr(mrand, "maximum")
            except AttributeError:  # pragma: no cover  # pragma: no cover
                raise NotImplementedError("Missing in compiler: maximum")
    res = fn(_u(a, "a"), _u(b, "b"))
    return _w(res)


def minimum(a: Any, b: Any, stream: Any = None) -> Any:
    """Compute minimum.

    Args:
        a: The a argument.
        b: The b argument.
        stream: The stream argument.

    Returns:
        The result of minimum.

    """
    from zero_mlx.array import array, _to_tensor
    import ml_switcheroo.ops as mops
    import ml_switcheroo.nn as mnn
    import ml_switcheroo.random as mrand
    import types

    def _u(x, param_name=None):
        if isinstance(x, list):  # pragma: no cover
            if __import__("builtins").all(
                (isinstance(i, int) for i in x)
            ):  # pragma: no cover
                return tuple(x)  # pragma: no cover
            return [_u(i, param_name) for i in x]  # pragma: no cover
        if isinstance(x, tuple):  # pragma: no cover
            return tuple((_u(i, param_name) for i in x))  # pragma: no cover
        if hasattr(x, "_tensor"):  # pragma: no cover
            return x._tensor  # pragma: no cover
        if isinstance(x, (int, float, bool, complex)):  # pragma: no cover
            if param_name not in {
                "axis",
                "rtol",
                "n",
                "m",
                "endpoint",
                "offset",
                "a_max",
                "ddof",
                "dims",
                "source",
                "axis2",
                "stop",
                "k",
                "dtype",
                "reps",
                "a_min",
                "sparse",
                "decimals",
                "strides",
                "num",
                "max",
                "indices_or_sections",
                "shape",
                "atol",
                "axis1",
                "step",
                "axes",
                "fill_value",
                "stream",
                "destination",
                "alpha",
                "beta",
                "start",
                "equal_nan",
                "keepdims",
                "min",
                "p",
            } and "minimum" not in (
                "transpose",
                "moveaxis",
                "reshape",
                "expand_dims",
                "linspace",
                "logcumsumexp",
                "cummin",
                "cummax",
                "cumprod",
                "cumsum",
                "logsumexp",
                "all",
                "any",
                "argmin",
                "argmax",
                "prod",
                "sum",
                "mean",
                "var",
                "std",
                "max",
                "min",
                "tensordot",
            ):  # pragma: no cover
                return _to_tensor(x)  # pragma: no cover
        if type(x).__name__ == "ndarray":  # pragma: no cover
            return _to_tensor(x)  # pragma: no cover
        return x  # pragma: no cover

    def _w(x):
        if isinstance(x, tuple):  # pragma: no cover
            return tuple((_w(i) for i in x))  # pragma: no cover
        if isinstance(x, list):  # pragma: no cover
            return [_w(i) for i in x]  # pragma: no cover
        return array(x) if hasattr(x, "shape") else x  # pragma: no cover

    try:
        fn = getattr(mops, "minimum")
    except AttributeError:  # pragma: no cover
        try:
            fn = getattr(mnn, "minimum")
        except AttributeError:  # pragma: no cover
            try:
                fn = getattr(mrand, "minimum")
            except AttributeError:  # pragma: no cover  # pragma: no cover
                raise NotImplementedError("Missing in compiler: minimum")
    res = fn(_u(a, "a"), _u(b, "b"))
    return _w(res)


def cos(a: Any, stream: Any = None) -> Any:
    """Compute cos.

    Args:
        a: The a argument.
        stream: The stream argument.

    Returns:
        The result of cos.

    """
    from zero_mlx.array import array, _to_tensor
    import ml_switcheroo.ops as mops
    import ml_switcheroo.nn as mnn
    import ml_switcheroo.random as mrand
    import types

    def _u(x, param_name=None):
        if isinstance(x, list):  # pragma: no cover
            if __import__("builtins").all(
                (isinstance(i, int) for i in x)
            ):  # pragma: no cover
                return tuple(x)  # pragma: no cover
            return [_u(i, param_name) for i in x]  # pragma: no cover
        if isinstance(x, tuple):  # pragma: no cover
            return tuple((_u(i, param_name) for i in x))  # pragma: no cover
        if hasattr(x, "_tensor"):  # pragma: no cover
            return x._tensor  # pragma: no cover
        if isinstance(x, (int, float, bool, complex)):  # pragma: no cover
            if param_name not in {
                "axis",
                "rtol",
                "n",
                "m",
                "endpoint",
                "offset",
                "a_max",
                "ddof",
                "dims",
                "source",
                "axis2",
                "stop",
                "k",
                "dtype",
                "reps",
                "a_min",
                "sparse",
                "decimals",
                "strides",
                "num",
                "max",
                "indices_or_sections",
                "shape",
                "atol",
                "axis1",
                "step",
                "axes",
                "fill_value",
                "stream",
                "destination",
                "alpha",
                "beta",
                "start",
                "equal_nan",
                "keepdims",
                "min",
                "p",
            } and "cos" not in (
                "transpose",
                "moveaxis",
                "reshape",
                "expand_dims",
                "linspace",
                "logcumsumexp",
                "cummin",
                "cummax",
                "cumprod",
                "cumsum",
                "logsumexp",
                "all",
                "any",
                "argmin",
                "argmax",
                "prod",
                "sum",
                "mean",
                "var",
                "std",
                "max",
                "min",
                "tensordot",
            ):  # pragma: no cover
                return _to_tensor(x)  # pragma: no cover
        if type(x).__name__ == "ndarray":  # pragma: no cover
            return _to_tensor(x)  # pragma: no cover
        return x  # pragma: no cover

    def _w(x):
        if isinstance(x, tuple):  # pragma: no cover
            return tuple((_w(i) for i in x))  # pragma: no cover
        if isinstance(x, list):  # pragma: no cover
            return [_w(i) for i in x]  # pragma: no cover
        return array(x) if hasattr(x, "shape") else x  # pragma: no cover

    try:
        fn = getattr(mops, "cos")
    except AttributeError:  # pragma: no cover
        try:
            fn = getattr(mnn, "cos")
        except AttributeError:  # pragma: no cover
            try:
                fn = getattr(mrand, "cos")
            except AttributeError:  # pragma: no cover  # pragma: no cover
                raise NotImplementedError("Missing in compiler: cos")
    res = fn(_u(a, "a"))
    return _w(res)


def log1p(a: Any, stream: Any = None) -> Any:
    """Compute log1p.

    Args:
        a: The a argument.
        stream: The stream argument.

    Returns:
        The result of log1p.

    """
    from zero_mlx.array import array, _to_tensor
    import ml_switcheroo.ops as mops
    import ml_switcheroo.nn as mnn
    import ml_switcheroo.random as mrand
    import types

    def _u(x, param_name=None):
        if isinstance(x, list):  # pragma: no cover
            if __import__("builtins").all(
                (isinstance(i, int) for i in x)
            ):  # pragma: no cover
                return tuple(x)  # pragma: no cover
            return [_u(i, param_name) for i in x]  # pragma: no cover
        if isinstance(x, tuple):  # pragma: no cover
            return tuple((_u(i, param_name) for i in x))  # pragma: no cover
        if hasattr(x, "_tensor"):  # pragma: no cover
            return x._tensor  # pragma: no cover
        if isinstance(x, (int, float, bool, complex)):  # pragma: no cover
            if param_name not in {
                "axis",
                "rtol",
                "n",
                "m",
                "endpoint",
                "offset",
                "a_max",
                "ddof",
                "dims",
                "source",
                "axis2",
                "stop",
                "k",
                "dtype",
                "reps",
                "a_min",
                "sparse",
                "decimals",
                "strides",
                "num",
                "max",
                "indices_or_sections",
                "shape",
                "atol",
                "axis1",
                "step",
                "axes",
                "fill_value",
                "stream",
                "destination",
                "alpha",
                "beta",
                "start",
                "equal_nan",
                "keepdims",
                "min",
                "p",
            } and "log1p" not in (
                "transpose",
                "moveaxis",
                "reshape",
                "expand_dims",
                "linspace",
                "logcumsumexp",
                "cummin",
                "cummax",
                "cumprod",
                "cumsum",
                "logsumexp",
                "all",
                "any",
                "argmin",
                "argmax",
                "prod",
                "sum",
                "mean",
                "var",
                "std",
                "max",
                "min",
                "tensordot",
            ):  # pragma: no cover
                return _to_tensor(x)  # pragma: no cover
        if type(x).__name__ == "ndarray":  # pragma: no cover
            return _to_tensor(x)  # pragma: no cover
        return x  # pragma: no cover

    def _w(x):
        if isinstance(x, tuple):  # pragma: no cover
            return tuple((_w(i) for i in x))  # pragma: no cover
        if isinstance(x, list):  # pragma: no cover
            return [_w(i) for i in x]  # pragma: no cover
        return array(x) if hasattr(x, "shape") else x  # pragma: no cover

    try:
        fn = getattr(mops, "log1p")
    except AttributeError:  # pragma: no cover
        try:
            fn = getattr(mnn, "log1p")
        except AttributeError:  # pragma: no cover
            try:
                fn = getattr(mrand, "log1p")
            except AttributeError:  # pragma: no cover  # pragma: no cover
                raise NotImplementedError("Missing in compiler: log1p")
    res = fn(_u(a, "a"))
    return _w(res)


def stop_gradient(a: Any, stream: Any = None) -> Any:
    """Compute stop_gradient.

    Args:
        a: The a argument.
        stream: The stream argument.

    Returns:
        The result of stop_gradient.

    """
    from zero_mlx.array import array, _to_tensor  # pragma: no cover
    import ml_switcheroo.ops as mops  # pragma: no cover
    import ml_switcheroo.nn as mnn  # pragma: no cover
    import ml_switcheroo.random as mrand  # pragma: no cover
    import types  # pragma: no cover

    def _u(x, param_name=None):  # pragma: no cover
        if isinstance(x, list):  # pragma: no cover
            if __import__("builtins").all(
                (isinstance(i, int) for i in x)
            ):  # pragma: no cover
                return tuple(x)  # pragma: no cover
            return [_u(i, param_name) for i in x]  # pragma: no cover
        if isinstance(x, tuple):  # pragma: no cover
            return tuple((_u(i, param_name) for i in x))  # pragma: no cover
        if hasattr(x, "_tensor"):  # pragma: no cover
            return x._tensor  # pragma: no cover
        if isinstance(x, (int, float, bool, complex)):  # pragma: no cover
            if param_name not in {
                "axis",
                "rtol",
                "n",
                "m",
                "endpoint",
                "offset",
                "a_max",
                "ddof",
                "dims",
                "source",
                "axis2",
                "stop",
                "k",
                "dtype",
                "reps",
                "a_min",
                "sparse",
                "decimals",
                "strides",
                "num",
                "max",
                "indices_or_sections",
                "shape",
                "atol",
                "axis1",
                "step",
                "axes",
                "fill_value",
                "stream",
                "destination",
                "alpha",
                "beta",
                "start",
                "equal_nan",
                "keepdims",
                "min",
                "p",
            } and "stop_gradient" not in (
                "transpose",
                "moveaxis",
                "reshape",
                "expand_dims",
                "linspace",
                "logcumsumexp",
                "cummin",
                "cummax",
                "cumprod",
                "cumsum",
                "logsumexp",
                "all",
                "any",
                "argmin",
                "argmax",
                "prod",
                "sum",
                "mean",
                "var",
                "std",
                "max",
                "min",
                "tensordot",
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
        fn = getattr(mops, "stop_gradient")  # pragma: no cover
    except AttributeError:  # pragma: no cover
        try:
            fn = getattr(mnn, "stop_gradient")
        except AttributeError:  # pragma: no cover
            try:
                fn = getattr(mrand, "stop_gradient")
            except AttributeError:  # pragma: no cover  # pragma: no cover
                raise NotImplementedError("Missing in compiler: stop_gradient")
    res = fn(_u(a, "a"))  # pragma: no cover
    return _w(res)  # pragma: no cover


for name in [
    "log10",
    "log2",
    "conj",
    "prod",
    "min",
    "max",
    "logcumsumexp",
    "logsumexp",
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
    "get_peak_memory",
    "squeeze",
    "expand_dims",
    "astype",
    "block_until_ready",
]:
    globals()[name] = lambda a, *args, **kwargs: (
        a if hasattr(a, "dtype") else __import__("zero_mlx").array(a)
    )


class random:
    """Random operations mock."""

    pass


def split(a, indices_or_sections, axis=0, stream=None):
    """Compute split.

    Args:
        a: The a argument.
        indices_or_sections: The indices_or_sections argument.
        axis: The axis argument.
        stream: The stream argument.

    Returns:
        The result of split.

    """
    from zero_mlx.array import array, _to_tensor
    import ml_switcheroo.ops as mops
    import ml_switcheroo.nn as mnn
    import ml_switcheroo.random as mrand
    import types

    def _u(x, param_name=None):
        if isinstance(x, list):  # pragma: no cover
            if __import__("builtins").all(
                (isinstance(i, int) for i in x)
            ):  # pragma: no cover
                return tuple(x)  # pragma: no cover
            return [_u(i, param_name) for i in x]  # pragma: no cover
        if isinstance(x, tuple):  # pragma: no cover
            return tuple((_u(i, param_name) for i in x))  # pragma: no cover
        if hasattr(x, "_tensor"):  # pragma: no cover
            return x._tensor  # pragma: no cover
        if isinstance(x, (int, float, bool, complex)):  # pragma: no cover
            if param_name not in {
                "axis",
                "rtol",
                "n",
                "m",
                "endpoint",
                "offset",
                "a_max",
                "ddof",
                "dims",
                "source",
                "axis2",
                "stop",
                "k",
                "dtype",
                "reps",
                "a_min",
                "sparse",
                "decimals",
                "strides",
                "num",
                "max",
                "indices_or_sections",
                "shape",
                "atol",
                "axis1",
                "step",
                "axes",
                "fill_value",
                "stream",
                "destination",
                "alpha",
                "beta",
                "start",
                "equal_nan",
                "keepdims",
                "min",
                "p",
            } and "split" not in (
                "transpose",
                "moveaxis",
                "reshape",
                "expand_dims",
                "linspace",
                "logcumsumexp",
                "cummin",
                "cummax",
                "cumprod",
                "cumsum",
                "logsumexp",
                "all",
                "any",
                "argmin",
                "argmax",
                "prod",
                "sum",
                "mean",
                "var",
                "std",
                "max",
                "min",
                "tensordot",
            ):  # pragma: no cover
                return _to_tensor(x)  # pragma: no cover
        if type(x).__name__ == "ndarray":  # pragma: no cover
            return _to_tensor(x)  # pragma: no cover
        return x  # pragma: no cover

    def _w(x):
        if isinstance(x, tuple):  # pragma: no cover
            return tuple((_w(i) for i in x))  # pragma: no cover
        if isinstance(x, list):  # pragma: no cover
            return [_w(i) for i in x]  # pragma: no cover
        return array(x) if hasattr(x, "shape") else x  # pragma: no cover

    try:
        fn = getattr(mops, "split")
    except AttributeError:  # pragma: no cover
        try:
            fn = getattr(mnn, "split")
        except AttributeError:  # pragma: no cover
            try:
                fn = getattr(mrand, "split")
            except AttributeError:  # pragma: no cover  # pragma: no cover
                raise NotImplementedError("Missing in compiler: split")
    res = fn(
        _u(a, "a"), _u(indices_or_sections, "indices_or_sections"), _u(axis, "axis")
    )
    return _w(res)


def diagonal(a, *args, **kwargs):
    """Compute diagonal.

    Args:
        a: The a argument.
        args: The args argument.
        kwargs: The kwargs argument.

    Returns:
        The result of diagonal.

    """
    from zero_mlx.array import array, _to_tensor
    import ml_switcheroo.ops as mops
    import ml_switcheroo.nn as mnn
    import ml_switcheroo.random as mrand
    import types

    def _u(x, param_name=None):
        if isinstance(x, list):  # pragma: no cover
            if __import__("builtins").all(
                (isinstance(i, int) for i in x)
            ):  # pragma: no cover
                return tuple(x)  # pragma: no cover
            return [_u(i, param_name) for i in x]  # pragma: no cover
        if isinstance(x, tuple):  # pragma: no cover
            return tuple((_u(i, param_name) for i in x))  # pragma: no cover
        if hasattr(x, "_tensor"):  # pragma: no cover
            return x._tensor  # pragma: no cover
        if isinstance(x, (int, float, bool, complex)):  # pragma: no cover
            if param_name not in {
                "axis",
                "rtol",
                "n",
                "m",
                "endpoint",
                "offset",
                "a_max",
                "ddof",
                "dims",
                "source",
                "axis2",
                "stop",
                "k",
                "dtype",
                "reps",
                "a_min",
                "sparse",
                "decimals",
                "strides",
                "num",
                "max",
                "indices_or_sections",
                "shape",
                "atol",
                "axis1",
                "step",
                "axes",
                "fill_value",
                "stream",
                "destination",
                "alpha",
                "beta",
                "start",
                "equal_nan",
                "keepdims",
                "min",
                "p",
            } and "diagonal" not in (
                "transpose",
                "moveaxis",
                "reshape",
                "expand_dims",
                "linspace",
                "logcumsumexp",
                "cummin",
                "cummax",
                "cumprod",
                "cumsum",
                "logsumexp",
                "all",
                "any",
                "argmin",
                "argmax",
                "prod",
                "sum",
                "mean",
                "var",
                "std",
                "max",
                "min",
                "tensordot",
            ):  # pragma: no cover
                return _to_tensor(x)  # pragma: no cover
        if type(x).__name__ == "ndarray":  # pragma: no cover
            return _to_tensor(x)  # pragma: no cover
        return x  # pragma: no cover

    def _w(x):
        if isinstance(x, tuple):  # pragma: no cover
            return tuple((_w(i) for i in x))  # pragma: no cover
        if isinstance(x, list):  # pragma: no cover
            return [_w(i) for i in x]  # pragma: no cover
        return array(x) if hasattr(x, "shape") else x  # pragma: no cover

    try:
        fn = getattr(mops, "diagonal")
    except AttributeError:  # pragma: no cover
        try:
            fn = getattr(mnn, "diagonal")
        except AttributeError:  # pragma: no cover
            try:
                fn = getattr(mrand, "diagonal")
            except AttributeError:  # pragma: no cover  # pragma: no cover
                raise NotImplementedError("Missing in compiler: diagonal")
    res = fn(
        _u(a, "a"),
        *[_u(x) for x in args],
        **{k: _u(v, k) for (k, v) in kwargs.items() if k != "stream"},
    )
    return _w(res)


def logcumsumexp(a, *args, **kwargs):
    """Compute logcumsumexp.

    Args:
        a: The a argument.
        args: The args argument.
        kwargs: The kwargs argument.

    Returns:
        The result of logcumsumexp.

    """
    from zero_mlx.array import array, _to_tensor
    import ml_switcheroo.ops as mops
    import ml_switcheroo.nn as mnn
    import ml_switcheroo.random as mrand
    import types

    def _u(x, param_name=None):
        if isinstance(x, list):  # pragma: no cover
            if __import__("builtins").all(
                (isinstance(i, int) for i in x)
            ):  # pragma: no cover
                return tuple(x)  # pragma: no cover
            return [_u(i, param_name) for i in x]  # pragma: no cover
        if isinstance(x, tuple):  # pragma: no cover
            return tuple((_u(i, param_name) for i in x))  # pragma: no cover
        if hasattr(x, "_tensor"):  # pragma: no cover
            return x._tensor  # pragma: no cover
        if isinstance(x, (int, float, bool, complex)):  # pragma: no cover
            if param_name not in {
                "axis",
                "rtol",
                "n",
                "m",
                "endpoint",
                "offset",
                "a_max",
                "ddof",
                "dims",
                "source",
                "axis2",
                "stop",
                "k",
                "dtype",
                "reps",
                "a_min",
                "sparse",
                "decimals",
                "strides",
                "num",
                "max",
                "indices_or_sections",
                "shape",
                "atol",
                "axis1",
                "step",
                "axes",
                "fill_value",
                "stream",
                "destination",
                "alpha",
                "beta",
                "start",
                "equal_nan",
                "keepdims",
                "min",
                "p",
            } and "logcumsumexp" not in (
                "transpose",
                "moveaxis",
                "reshape",
                "expand_dims",
                "linspace",
                "logcumsumexp",
                "cummin",
                "cummax",
                "cumprod",
                "cumsum",
                "logsumexp",
                "all",
                "any",
                "argmin",
                "argmax",
                "prod",
                "sum",
                "mean",
                "var",
                "std",
                "max",
                "min",
                "tensordot",
            ):  # pragma: no cover
                return _to_tensor(x)  # pragma: no cover
        if type(x).__name__ == "ndarray":  # pragma: no cover
            return _to_tensor(x)  # pragma: no cover
        return x  # pragma: no cover

    def _w(x):
        if isinstance(x, tuple):  # pragma: no cover
            return tuple((_w(i) for i in x))  # pragma: no cover
        if isinstance(x, list):  # pragma: no cover
            return [_w(i) for i in x]  # pragma: no cover
        return array(x) if hasattr(x, "shape") else x  # pragma: no cover

    try:
        fn = getattr(mops, "logcumsumexp")
    except AttributeError:  # pragma: no cover
        try:
            fn = getattr(mnn, "logcumsumexp")
        except AttributeError:  # pragma: no cover
            try:
                fn = getattr(mrand, "logcumsumexp")
            except AttributeError:  # pragma: no cover  # pragma: no cover
                raise NotImplementedError("Missing in compiler: logcumsumexp")
    res = fn(
        _u(a, "a"),
        *[_u(x) for x in args],
        **{k: _u(v, k) for (k, v) in kwargs.items() if k != "stream"},
    )
    return _w(res)


def logsumexp(a, *args, **kwargs):
    """Compute logsumexp.

    Args:
        a: The a argument.
        args: The args argument.
        kwargs: The kwargs argument.

    Returns:
        The result of logsumexp.

    """
    from zero_mlx.array import array, _to_tensor
    import ml_switcheroo.ops as mops
    import ml_switcheroo.nn as mnn
    import ml_switcheroo.random as mrand
    import types

    def _u(x, param_name=None):
        if isinstance(x, list):  # pragma: no cover
            if __import__("builtins").all(
                (isinstance(i, int) for i in x)
            ):  # pragma: no cover
                return tuple(x)  # pragma: no cover
            return [_u(i, param_name) for i in x]  # pragma: no cover
        if isinstance(x, tuple):  # pragma: no cover
            return tuple((_u(i, param_name) for i in x))  # pragma: no cover
        if hasattr(x, "_tensor"):  # pragma: no cover
            return x._tensor  # pragma: no cover
        if isinstance(x, (int, float, bool, complex)):  # pragma: no cover
            if param_name not in {
                "axis",
                "rtol",
                "n",
                "m",
                "endpoint",
                "offset",
                "a_max",
                "ddof",
                "dims",
                "source",
                "axis2",
                "stop",
                "k",
                "dtype",
                "reps",
                "a_min",
                "sparse",
                "decimals",
                "strides",
                "num",
                "max",
                "indices_or_sections",
                "shape",
                "atol",
                "axis1",
                "step",
                "axes",
                "fill_value",
                "stream",
                "destination",
                "alpha",
                "beta",
                "start",
                "equal_nan",
                "keepdims",
                "min",
                "p",
            } and "logsumexp" not in (
                "transpose",
                "moveaxis",
                "reshape",
                "expand_dims",
                "linspace",
                "logcumsumexp",
                "cummin",
                "cummax",
                "cumprod",
                "cumsum",
                "logsumexp",
                "all",
                "any",
                "argmin",
                "argmax",
                "prod",
                "sum",
                "mean",
                "var",
                "std",
                "max",
                "min",
                "tensordot",
            ):  # pragma: no cover
                return _to_tensor(x)  # pragma: no cover
        if type(x).__name__ == "ndarray":  # pragma: no cover
            return _to_tensor(x)  # pragma: no cover
        return x  # pragma: no cover

    def _w(x):
        if isinstance(x, tuple):  # pragma: no cover
            return tuple((_w(i) for i in x))  # pragma: no cover
        if isinstance(x, list):  # pragma: no cover
            return [_w(i) for i in x]  # pragma: no cover
        return array(x) if hasattr(x, "shape") else x  # pragma: no cover

    try:
        fn = getattr(mops, "logsumexp")
    except AttributeError:  # pragma: no cover
        try:
            fn = getattr(mnn, "logsumexp")
        except AttributeError:  # pragma: no cover
            try:
                fn = getattr(mrand, "logsumexp")
            except AttributeError:  # pragma: no cover  # pragma: no cover
                raise NotImplementedError("Missing in compiler: logsumexp")
    res = fn(
        _u(a, "a"),
        *[_u(x) for x in args],
        **{k: _u(v, k) for (k, v) in kwargs.items() if k != "stream"},
    )
    return _w(res)


def cummax(a, *args, **kwargs):
    """Compute cummax.

    Args:
        a: The a argument.
        args: The args argument.
        kwargs: The kwargs argument.

    Returns:
        The result of cummax.

    """
    from zero_mlx.array import array, _to_tensor
    import ml_switcheroo.ops as mops
    import ml_switcheroo.nn as mnn
    import ml_switcheroo.random as mrand
    import types

    def _u(x, param_name=None):
        if isinstance(x, list):  # pragma: no cover
            if __import__("builtins").all(
                (isinstance(i, int) for i in x)
            ):  # pragma: no cover
                return tuple(x)  # pragma: no cover
            return [_u(i, param_name) for i in x]  # pragma: no cover
        if isinstance(x, tuple):  # pragma: no cover
            return tuple((_u(i, param_name) for i in x))  # pragma: no cover
        if hasattr(x, "_tensor"):  # pragma: no cover
            return x._tensor  # pragma: no cover
        if isinstance(x, (int, float, bool, complex)):  # pragma: no cover
            if param_name not in {
                "axis",
                "rtol",
                "n",
                "m",
                "endpoint",
                "offset",
                "a_max",
                "ddof",
                "dims",
                "source",
                "axis2",
                "stop",
                "k",
                "dtype",
                "reps",
                "a_min",
                "sparse",
                "decimals",
                "strides",
                "num",
                "max",
                "indices_or_sections",
                "shape",
                "atol",
                "axis1",
                "step",
                "axes",
                "fill_value",
                "stream",
                "destination",
                "alpha",
                "beta",
                "start",
                "equal_nan",
                "keepdims",
                "min",
                "p",
            } and "cummax" not in (
                "transpose",
                "moveaxis",
                "reshape",
                "expand_dims",
                "linspace",
                "logcumsumexp",
                "cummin",
                "cummax",
                "cumprod",
                "cumsum",
                "logsumexp",
                "all",
                "any",
                "argmin",
                "argmax",
                "prod",
                "sum",
                "mean",
                "var",
                "std",
                "max",
                "min",
                "tensordot",
            ):  # pragma: no cover
                return _to_tensor(x)  # pragma: no cover
        if type(x).__name__ == "ndarray":  # pragma: no cover
            return _to_tensor(x)  # pragma: no cover
        return x  # pragma: no cover

    def _w(x):
        if isinstance(x, tuple):  # pragma: no cover
            return tuple((_w(i) for i in x))  # pragma: no cover
        if isinstance(x, list):  # pragma: no cover
            return [_w(i) for i in x]  # pragma: no cover
        return array(x) if hasattr(x, "shape") else x  # pragma: no cover

    try:
        fn = getattr(mops, "cummax")
    except AttributeError:  # pragma: no cover
        try:
            fn = getattr(mnn, "cummax")
        except AttributeError:  # pragma: no cover
            try:
                fn = getattr(mrand, "cummax")
            except AttributeError:  # pragma: no cover  # pragma: no cover
                raise NotImplementedError("Missing in compiler: cummax")
    res = fn(
        _u(a, "a"),
        *[_u(x) for x in args],
        **{k: _u(v, k) for (k, v) in kwargs.items() if k != "stream"},
    )
    return _w(res)


def cummin(a, *args, **kwargs):
    """Compute cummin.

    Args:
        a: The a argument.
        args: The args argument.
        kwargs: The kwargs argument.

    Returns:
        The result of cummin.

    """
    from zero_mlx.array import array, _to_tensor
    import ml_switcheroo.ops as mops
    import ml_switcheroo.nn as mnn
    import ml_switcheroo.random as mrand
    import types

    def _u(x, param_name=None):
        if isinstance(x, list):  # pragma: no cover
            if __import__("builtins").all(
                (isinstance(i, int) for i in x)
            ):  # pragma: no cover
                return tuple(x)  # pragma: no cover
            return [_u(i, param_name) for i in x]  # pragma: no cover
        if isinstance(x, tuple):  # pragma: no cover
            return tuple((_u(i, param_name) for i in x))  # pragma: no cover
        if hasattr(x, "_tensor"):  # pragma: no cover
            return x._tensor  # pragma: no cover
        if isinstance(x, (int, float, bool, complex)):  # pragma: no cover
            if param_name not in {
                "axis",
                "rtol",
                "n",
                "m",
                "endpoint",
                "offset",
                "a_max",
                "ddof",
                "dims",
                "source",
                "axis2",
                "stop",
                "k",
                "dtype",
                "reps",
                "a_min",
                "sparse",
                "decimals",
                "strides",
                "num",
                "max",
                "indices_or_sections",
                "shape",
                "atol",
                "axis1",
                "step",
                "axes",
                "fill_value",
                "stream",
                "destination",
                "alpha",
                "beta",
                "start",
                "equal_nan",
                "keepdims",
                "min",
                "p",
            } and "cummin" not in (
                "transpose",
                "moveaxis",
                "reshape",
                "expand_dims",
                "linspace",
                "logcumsumexp",
                "cummin",
                "cummax",
                "cumprod",
                "cumsum",
                "logsumexp",
                "all",
                "any",
                "argmin",
                "argmax",
                "prod",
                "sum",
                "mean",
                "var",
                "std",
                "max",
                "min",
                "tensordot",
            ):  # pragma: no cover
                return _to_tensor(x)  # pragma: no cover
        if type(x).__name__ == "ndarray":  # pragma: no cover
            return _to_tensor(x)  # pragma: no cover
        return x  # pragma: no cover

    def _w(x):
        if isinstance(x, tuple):  # pragma: no cover
            return tuple((_w(i) for i in x))  # pragma: no cover
        if isinstance(x, list):  # pragma: no cover
            return [_w(i) for i in x]  # pragma: no cover
        return array(x) if hasattr(x, "shape") else x  # pragma: no cover

    try:
        fn = getattr(mops, "cummin")
    except AttributeError:  # pragma: no cover
        try:
            fn = getattr(mnn, "cummin")
        except AttributeError:  # pragma: no cover
            try:
                fn = getattr(mrand, "cummin")
            except AttributeError:  # pragma: no cover  # pragma: no cover
                raise NotImplementedError("Missing in compiler: cummin")
    res = fn(
        _u(a, "a"),
        *[_u(x) for x in args],
        **{k: _u(v, k) for (k, v) in kwargs.items() if k != "stream"},
    )
    return _w(res)


def cumprod(a, *args, **kwargs):
    """Compute cumprod.

    Args:
        a: The a argument.
        args: The args argument.
        kwargs: The kwargs argument.

    Returns:
        The result of cumprod.

    """
    from zero_mlx.array import array, _to_tensor
    import ml_switcheroo.ops as mops
    import ml_switcheroo.nn as mnn
    import ml_switcheroo.random as mrand
    import types

    def _u(x, param_name=None):
        if isinstance(x, list):  # pragma: no cover
            if __import__("builtins").all(
                (isinstance(i, int) for i in x)
            ):  # pragma: no cover
                return tuple(x)  # pragma: no cover
            return [_u(i, param_name) for i in x]  # pragma: no cover
        if isinstance(x, tuple):  # pragma: no cover
            return tuple((_u(i, param_name) for i in x))  # pragma: no cover
        if hasattr(x, "_tensor"):  # pragma: no cover
            return x._tensor  # pragma: no cover
        if isinstance(x, (int, float, bool, complex)):  # pragma: no cover
            if param_name not in {
                "axis",
                "rtol",
                "n",
                "m",
                "endpoint",
                "offset",
                "a_max",
                "ddof",
                "dims",
                "source",
                "axis2",
                "stop",
                "k",
                "dtype",
                "reps",
                "a_min",
                "sparse",
                "decimals",
                "strides",
                "num",
                "max",
                "indices_or_sections",
                "shape",
                "atol",
                "axis1",
                "step",
                "axes",
                "fill_value",
                "stream",
                "destination",
                "alpha",
                "beta",
                "start",
                "equal_nan",
                "keepdims",
                "min",
                "p",
            } and "cumprod" not in (
                "transpose",
                "moveaxis",
                "reshape",
                "expand_dims",
                "linspace",
                "logcumsumexp",
                "cummin",
                "cummax",
                "cumprod",
                "cumsum",
                "logsumexp",
                "all",
                "any",
                "argmin",
                "argmax",
                "prod",
                "sum",
                "mean",
                "var",
                "std",
                "max",
                "min",
                "tensordot",
            ):  # pragma: no cover
                return _to_tensor(x)  # pragma: no cover
        if type(x).__name__ == "ndarray":  # pragma: no cover
            return _to_tensor(x)  # pragma: no cover
        return x  # pragma: no cover

    def _w(x):
        if isinstance(x, tuple):  # pragma: no cover
            return tuple((_w(i) for i in x))  # pragma: no cover
        if isinstance(x, list):  # pragma: no cover
            return [_w(i) for i in x]  # pragma: no cover
        return array(x) if hasattr(x, "shape") else x  # pragma: no cover

    try:
        fn = getattr(mops, "cumprod")
    except AttributeError:  # pragma: no cover
        try:
            fn = getattr(mnn, "cumprod")
        except AttributeError:  # pragma: no cover
            try:
                fn = getattr(mrand, "cumprod")
            except AttributeError:  # pragma: no cover  # pragma: no cover
                raise NotImplementedError("Missing in compiler: cumprod")
    res = fn(
        _u(a, "a"),
        *[_u(x) for x in args],
        **{k: _u(v, k) for (k, v) in kwargs.items() if k != "stream"},
    )
    return _w(res)


def cumsum(a, *args, **kwargs):
    """Compute cumsum.

    Args:
        a: The a argument.
        args: The args argument.
        kwargs: The kwargs argument.

    Returns:
        The result of cumsum.

    """
    from zero_mlx.array import array, _to_tensor
    import ml_switcheroo.ops as mops
    import ml_switcheroo.nn as mnn
    import ml_switcheroo.random as mrand
    import types

    def _u(x, param_name=None):
        if isinstance(x, list):  # pragma: no cover
            if __import__("builtins").all(
                (isinstance(i, int) for i in x)
            ):  # pragma: no cover
                return tuple(x)  # pragma: no cover
            return [_u(i, param_name) for i in x]  # pragma: no cover
        if isinstance(x, tuple):  # pragma: no cover
            return tuple((_u(i, param_name) for i in x))  # pragma: no cover
        if hasattr(x, "_tensor"):  # pragma: no cover
            return x._tensor  # pragma: no cover
        if isinstance(x, (int, float, bool, complex)):  # pragma: no cover
            if param_name not in {
                "axis",
                "rtol",
                "n",
                "m",
                "endpoint",
                "offset",
                "a_max",
                "ddof",
                "dims",
                "source",
                "axis2",
                "stop",
                "k",
                "dtype",
                "reps",
                "a_min",
                "sparse",
                "decimals",
                "strides",
                "num",
                "max",
                "indices_or_sections",
                "shape",
                "atol",
                "axis1",
                "step",
                "axes",
                "fill_value",
                "stream",
                "destination",
                "alpha",
                "beta",
                "start",
                "equal_nan",
                "keepdims",
                "min",
                "p",
            } and "cumsum" not in (
                "transpose",
                "moveaxis",
                "reshape",
                "expand_dims",
                "linspace",
                "logcumsumexp",
                "cummin",
                "cummax",
                "cumprod",
                "cumsum",
                "logsumexp",
                "all",
                "any",
                "argmin",
                "argmax",
                "prod",
                "sum",
                "mean",
                "var",
                "std",
                "max",
                "min",
                "tensordot",
            ):  # pragma: no cover
                return _to_tensor(x)  # pragma: no cover
        if type(x).__name__ == "ndarray":  # pragma: no cover
            return _to_tensor(x)  # pragma: no cover
        return x  # pragma: no cover

    def _w(x):
        if isinstance(x, tuple):  # pragma: no cover
            return tuple((_w(i) for i in x))  # pragma: no cover
        if isinstance(x, list):  # pragma: no cover
            return [_w(i) for i in x]  # pragma: no cover
        return array(x) if hasattr(x, "shape") else x  # pragma: no cover

    try:
        fn = getattr(mops, "cumsum")
    except AttributeError:  # pragma: no cover
        try:
            fn = getattr(mnn, "cumsum")
        except AttributeError:  # pragma: no cover
            try:
                fn = getattr(mrand, "cumsum")
            except AttributeError:  # pragma: no cover  # pragma: no cover
                raise NotImplementedError("Missing in compiler: cumsum")
    res = fn(
        _u(a, "a"),
        *[_u(x) for x in args],
        **{k: _u(v, k) for (k, v) in kwargs.items() if k != "stream"},
    )
    return _w(res)


def transpose(a, *args, axes=None):
    """Compute transpose.

    Args:
        a: The a argument.
        axes: The axes argument.
        args: The args argument.

    Returns:
        The result of transpose.

    """
    from zero_mlx.array import array, _to_tensor
    import ml_switcheroo.ops as mops
    import ml_switcheroo.nn as mnn
    import ml_switcheroo.random as mrand
    import types

    def _u(x, param_name=None):
        if isinstance(x, list):  # pragma: no cover
            if __import__("builtins").all(
                (isinstance(i, int) for i in x)
            ):  # pragma: no cover
                return tuple(x)  # pragma: no cover
            return [_u(i, param_name) for i in x]  # pragma: no cover
        if isinstance(x, tuple):  # pragma: no cover
            return tuple((_u(i, param_name) for i in x))  # pragma: no cover
        if hasattr(x, "_tensor"):  # pragma: no cover
            return x._tensor  # pragma: no cover
        if isinstance(x, (int, float, bool, complex)):  # pragma: no cover
            if param_name not in {
                "axis",
                "rtol",
                "n",
                "m",
                "endpoint",
                "offset",
                "a_max",
                "ddof",
                "dims",
                "source",
                "axis2",
                "stop",
                "k",
                "dtype",
                "reps",
                "a_min",
                "sparse",
                "decimals",
                "strides",
                "num",
                "max",
                "indices_or_sections",
                "shape",
                "atol",
                "axis1",
                "step",
                "axes",
                "fill_value",
                "stream",
                "destination",
                "alpha",
                "beta",
                "start",
                "equal_nan",
                "keepdims",
                "min",
                "p",
            } and "transpose" not in (
                "transpose",
                "moveaxis",
                "reshape",
                "expand_dims",
                "linspace",
                "logcumsumexp",
                "cummin",
                "cummax",
                "cumprod",
                "cumsum",
                "logsumexp",
                "all",
                "any",
                "argmin",
                "argmax",
                "prod",
                "sum",
                "mean",
                "var",
                "std",
                "max",
                "min",
                "tensordot",
            ):  # pragma: no cover
                return _to_tensor(x)  # pragma: no cover
        if type(x).__name__ == "ndarray":  # pragma: no cover
            return _to_tensor(x)  # pragma: no cover
        return x  # pragma: no cover

    def _w(x):
        if isinstance(x, tuple):  # pragma: no cover
            return tuple((_w(i) for i in x))  # pragma: no cover
        if isinstance(x, list):  # pragma: no cover
            return [_w(i) for i in x]  # pragma: no cover
        return array(x) if hasattr(x, "shape") else x  # pragma: no cover

    try:
        fn = getattr(mops, "permute")
    except AttributeError:  # pragma: no cover
        try:
            fn = getattr(mnn, "permute")
        except AttributeError:  # pragma: no cover
            try:
                fn = getattr(mrand, "permute")
            except AttributeError:  # pragma: no cover  # pragma: no cover
                raise NotImplementedError("Missing in compiler: permute")
    res = fn(_u(a, "a"), *[_u(x) for x in args], axes=_u(axes, "axes"))
    return _w(res)


def concatenate(arrays, axis=0):
    """Compute concatenate.

    Args:
        arrays: The arrays argument.
        axis: The axis argument.

    Returns:
        The result of concatenate.

    """
    from zero_mlx.array import array, _to_tensor
    import ml_switcheroo.ops as mops
    import ml_switcheroo.nn as mnn
    import ml_switcheroo.random as mrand
    import types

    def _u(x, param_name=None):
        if isinstance(x, list):  # pragma: no cover
            if __import__("builtins").all(
                (isinstance(i, int) for i in x)
            ):  # pragma: no cover
                return tuple(x)  # pragma: no cover
            return [_u(i, param_name) for i in x]  # pragma: no cover
        if isinstance(x, tuple):  # pragma: no cover
            return tuple((_u(i, param_name) for i in x))  # pragma: no cover
        if hasattr(x, "_tensor"):  # pragma: no cover
            return x._tensor  # pragma: no cover
        if isinstance(x, (int, float, bool, complex)):  # pragma: no cover
            if param_name not in {
                "axis",
                "rtol",
                "n",
                "m",
                "endpoint",
                "offset",
                "a_max",
                "ddof",
                "dims",
                "source",
                "axis2",
                "stop",
                "k",
                "dtype",
                "reps",
                "a_min",
                "sparse",
                "decimals",
                "strides",
                "num",
                "max",
                "indices_or_sections",
                "shape",
                "atol",
                "axis1",
                "step",
                "axes",
                "fill_value",
                "stream",
                "destination",
                "alpha",
                "beta",
                "start",
                "equal_nan",
                "keepdims",
                "min",
                "p",
            } and "concatenate" not in (
                "transpose",
                "moveaxis",
                "reshape",
                "expand_dims",
                "linspace",
                "logcumsumexp",
                "cummin",
                "cummax",
                "cumprod",
                "cumsum",
                "logsumexp",
                "all",
                "any",
                "argmin",
                "argmax",
                "prod",
                "sum",
                "mean",
                "var",
                "std",
                "max",
                "min",
                "tensordot",
            ):  # pragma: no cover
                return _to_tensor(x)  # pragma: no cover
        if type(x).__name__ == "ndarray":  # pragma: no cover
            return _to_tensor(x)  # pragma: no cover
        return x  # pragma: no cover

    def _w(x):
        if isinstance(x, tuple):  # pragma: no cover
            return tuple((_w(i) for i in x))  # pragma: no cover
        if isinstance(x, list):  # pragma: no cover
            return [_w(i) for i in x]  # pragma: no cover
        return array(x) if hasattr(x, "shape") else x  # pragma: no cover

    try:
        fn = getattr(mops, "concatenate")
    except AttributeError:  # pragma: no cover
        try:
            fn = getattr(mnn, "concatenate")
        except AttributeError:  # pragma: no cover
            try:
                fn = getattr(mrand, "concatenate")
            except AttributeError:  # pragma: no cover  # pragma: no cover
                raise NotImplementedError("Missing in compiler: concatenate")
    res = fn(_u(arrays, "arrays"), _u(axis, "axis"))
    return _w(res)


def get_peak_memory(*args, **kwargs):
    """Compute get_peak_memory.

    Args:
        args: The args argument.
        kwargs: The kwargs argument.

    Returns:
        The result of get_peak_memory.

    """
    from zero_mlx.array import array, _to_tensor
    import ml_switcheroo.ops as mops
    import ml_switcheroo.nn as mnn
    import ml_switcheroo.random as mrand
    import types

    def _u(x, param_name=None):
        if isinstance(x, list):  # pragma: no cover
            if __import__("builtins").all(
                (isinstance(i, int) for i in x)
            ):  # pragma: no cover
                return tuple(x)  # pragma: no cover
            return [_u(i, param_name) for i in x]  # pragma: no cover
        if isinstance(x, tuple):  # pragma: no cover
            return tuple((_u(i, param_name) for i in x))  # pragma: no cover
        if hasattr(x, "_tensor"):  # pragma: no cover
            return x._tensor  # pragma: no cover
        if isinstance(x, (int, float, bool, complex)):  # pragma: no cover
            if param_name not in {
                "axis",
                "rtol",
                "n",
                "m",
                "endpoint",
                "offset",
                "a_max",
                "ddof",
                "dims",
                "source",
                "axis2",
                "stop",
                "k",
                "dtype",
                "reps",
                "a_min",
                "sparse",
                "decimals",
                "strides",
                "num",
                "max",
                "indices_or_sections",
                "shape",
                "atol",
                "axis1",
                "step",
                "axes",
                "fill_value",
                "stream",
                "destination",
                "alpha",
                "beta",
                "start",
                "equal_nan",
                "keepdims",
                "min",
                "p",
            } and "get_peak_memory" not in (
                "transpose",
                "moveaxis",
                "reshape",
                "expand_dims",
                "linspace",
                "logcumsumexp",
                "cummin",
                "cummax",
                "cumprod",
                "cumsum",
                "logsumexp",
                "all",
                "any",
                "argmin",
                "argmax",
                "prod",
                "sum",
                "mean",
                "var",
                "std",
                "max",
                "min",
                "tensordot",
            ):  # pragma: no cover
                return _to_tensor(x)  # pragma: no cover
        if type(x).__name__ == "ndarray":  # pragma: no cover
            return _to_tensor(x)  # pragma: no cover
        return x  # pragma: no cover

    def _w(x):
        if isinstance(x, tuple):  # pragma: no cover
            return tuple((_w(i) for i in x))  # pragma: no cover
        if isinstance(x, list):  # pragma: no cover
            return [_w(i) for i in x]  # pragma: no cover
        return array(x) if hasattr(x, "shape") else x  # pragma: no cover

    try:
        fn = getattr(mops, "get_peak_memory")
    except AttributeError:  # pragma: no cover
        try:
            fn = getattr(mnn, "get_peak_memory")
        except AttributeError:  # pragma: no cover
            try:
                fn = getattr(mrand, "get_peak_memory")
            except AttributeError:  # pragma: no cover  # pragma: no cover
                raise NotImplementedError("Missing in compiler: get_peak_memory")
    res = fn(
        *[_u(x) for x in args],
        **{k: _u(v, k) for (k, v) in kwargs.items() if k != "stream"},
    )
    return _w(res)


def argmin(a, axis=None, keepdims=False, stream=None):
    """Compute argmin.

    Args:
        a: The a argument.
        axis: The axis argument.
        keepdims: The keepdims argument.
        stream: The stream argument.

    Returns:
        The result of argmin.

    """
    from zero_mlx.array import array, _to_tensor
    import ml_switcheroo.ops as mops
    import ml_switcheroo.nn as mnn
    import ml_switcheroo.random as mrand
    import types

    def _u(x, param_name=None):
        if isinstance(x, list):  # pragma: no cover
            if __import__("builtins").all(
                (isinstance(i, int) for i in x)
            ):  # pragma: no cover
                return tuple(x)  # pragma: no cover
            return [_u(i, param_name) for i in x]  # pragma: no cover
        if isinstance(x, tuple):  # pragma: no cover
            return tuple((_u(i, param_name) for i in x))  # pragma: no cover
        if hasattr(x, "_tensor"):  # pragma: no cover
            return x._tensor  # pragma: no cover
        if isinstance(x, (int, float, bool, complex)):  # pragma: no cover
            if param_name not in {
                "axis",
                "rtol",
                "n",
                "m",
                "endpoint",
                "offset",
                "a_max",
                "ddof",
                "dims",
                "source",
                "axis2",
                "stop",
                "k",
                "dtype",
                "reps",
                "a_min",
                "sparse",
                "decimals",
                "strides",
                "num",
                "max",
                "indices_or_sections",
                "shape",
                "atol",
                "axis1",
                "step",
                "axes",
                "fill_value",
                "stream",
                "destination",
                "alpha",
                "beta",
                "start",
                "equal_nan",
                "keepdims",
                "min",
                "p",
            } and "argmin" not in (
                "transpose",
                "moveaxis",
                "reshape",
                "expand_dims",
                "linspace",
                "logcumsumexp",
                "cummin",
                "cummax",
                "cumprod",
                "cumsum",
                "logsumexp",
                "all",
                "any",
                "argmin",
                "argmax",
                "prod",
                "sum",
                "mean",
                "var",
                "std",
                "max",
                "min",
                "tensordot",
            ):  # pragma: no cover
                return _to_tensor(x)  # pragma: no cover
        if type(x).__name__ == "ndarray":  # pragma: no cover
            return _to_tensor(x)  # pragma: no cover
        return x  # pragma: no cover

    def _w(x):
        if isinstance(x, tuple):  # pragma: no cover
            return tuple((_w(i) for i in x))  # pragma: no cover
        if isinstance(x, list):  # pragma: no cover
            return [_w(i) for i in x]  # pragma: no cover
        return array(x) if hasattr(x, "shape") else x  # pragma: no cover

    try:
        fn = getattr(mops, "argmin")
    except AttributeError:  # pragma: no cover
        try:
            fn = getattr(mnn, "argmin")
        except AttributeError:  # pragma: no cover
            try:
                fn = getattr(mrand, "argmin")
            except AttributeError:  # pragma: no cover  # pragma: no cover
                raise NotImplementedError("Missing in compiler: argmin")
    res = fn(_u(a, "a"), _u(axis, "axis"), _u(keepdims, "keepdims"))
    return _w(res)


def argmax(a, axis=None, keepdims=False, stream=None):
    """Compute argmax.

    Args:
        a: The a argument.
        axis: The axis argument.
        keepdims: The keepdims argument.
        stream: The stream argument.

    Returns:
        The result of argmax.

    """
    from zero_mlx.array import array, _to_tensor
    import ml_switcheroo.ops as mops
    import ml_switcheroo.nn as mnn
    import ml_switcheroo.random as mrand
    import types

    def _u(x, param_name=None):
        if isinstance(x, list):  # pragma: no cover
            if __import__("builtins").all(
                (isinstance(i, int) for i in x)
            ):  # pragma: no cover
                return tuple(x)  # pragma: no cover
            return [_u(i, param_name) for i in x]  # pragma: no cover
        if isinstance(x, tuple):  # pragma: no cover
            return tuple((_u(i, param_name) for i in x))  # pragma: no cover
        if hasattr(x, "_tensor"):  # pragma: no cover
            return x._tensor  # pragma: no cover
        if isinstance(x, (int, float, bool, complex)):  # pragma: no cover
            if param_name not in {
                "axis",
                "rtol",
                "n",
                "m",
                "endpoint",
                "offset",
                "a_max",
                "ddof",
                "dims",
                "source",
                "axis2",
                "stop",
                "k",
                "dtype",
                "reps",
                "a_min",
                "sparse",
                "decimals",
                "strides",
                "num",
                "max",
                "indices_or_sections",
                "shape",
                "atol",
                "axis1",
                "step",
                "axes",
                "fill_value",
                "stream",
                "destination",
                "alpha",
                "beta",
                "start",
                "equal_nan",
                "keepdims",
                "min",
                "p",
            } and "argmax" not in (
                "transpose",
                "moveaxis",
                "reshape",
                "expand_dims",
                "linspace",
                "logcumsumexp",
                "cummin",
                "cummax",
                "cumprod",
                "cumsum",
                "logsumexp",
                "all",
                "any",
                "argmin",
                "argmax",
                "prod",
                "sum",
                "mean",
                "var",
                "std",
                "max",
                "min",
                "tensordot",
            ):  # pragma: no cover
                return _to_tensor(x)  # pragma: no cover
        if type(x).__name__ == "ndarray":  # pragma: no cover
            return _to_tensor(x)  # pragma: no cover
        return x  # pragma: no cover

    def _w(x):
        if isinstance(x, tuple):  # pragma: no cover
            return tuple((_w(i) for i in x))  # pragma: no cover
        if isinstance(x, list):  # pragma: no cover
            return [_w(i) for i in x]  # pragma: no cover
        return array(x) if hasattr(x, "shape") else x  # pragma: no cover

    try:
        fn = getattr(mops, "argmax")
    except AttributeError:  # pragma: no cover
        try:
            fn = getattr(mnn, "argmax")
        except AttributeError:  # pragma: no cover
            try:
                fn = getattr(mrand, "argmax")
            except AttributeError:  # pragma: no cover  # pragma: no cover
                raise NotImplementedError("Missing in compiler: argmax")
    res = fn(_u(a, "a"), _u(axis, "axis"), _u(keepdims, "keepdims"))
    return _w(res)


def min(a, axis=None, keepdims=False, stream=None):
    """Compute min.

    Args:
        a: The a argument.
        axis: The axis argument.
        keepdims: The keepdims argument.
        stream: The stream argument.

    Returns:
        The result of min.

    """
    from zero_mlx.array import array, _to_tensor
    import ml_switcheroo.ops as mops
    import ml_switcheroo.nn as mnn
    import ml_switcheroo.random as mrand
    import types

    def _u(x, param_name=None):
        if isinstance(x, list):  # pragma: no cover
            if __import__("builtins").all(
                (isinstance(i, int) for i in x)
            ):  # pragma: no cover
                return tuple(x)  # pragma: no cover
            return [_u(i, param_name) for i in x]  # pragma: no cover
        if isinstance(x, tuple):  # pragma: no cover
            return tuple((_u(i, param_name) for i in x))  # pragma: no cover
        if hasattr(x, "_tensor"):  # pragma: no cover
            return x._tensor  # pragma: no cover
        if isinstance(x, (int, float, bool, complex)):  # pragma: no cover
            if param_name not in {
                "axis",
                "rtol",
                "n",
                "m",
                "endpoint",
                "offset",
                "a_max",
                "ddof",
                "dims",
                "source",
                "axis2",
                "stop",
                "k",
                "dtype",
                "reps",
                "a_min",
                "sparse",
                "decimals",
                "strides",
                "num",
                "max",
                "indices_or_sections",
                "shape",
                "atol",
                "axis1",
                "step",
                "axes",
                "fill_value",
                "stream",
                "destination",
                "alpha",
                "beta",
                "start",
                "equal_nan",
                "keepdims",
                "min",
                "p",
            } and "min" not in (
                "transpose",
                "moveaxis",
                "reshape",
                "expand_dims",
                "linspace",
                "logcumsumexp",
                "cummin",
                "cummax",
                "cumprod",
                "cumsum",
                "logsumexp",
                "all",
                "any",
                "argmin",
                "argmax",
                "prod",
                "sum",
                "mean",
                "var",
                "std",
                "max",
                "min",
                "tensordot",
            ):  # pragma: no cover
                return _to_tensor(x)  # pragma: no cover
        if type(x).__name__ == "ndarray":  # pragma: no cover
            return _to_tensor(x)  # pragma: no cover
        return x  # pragma: no cover

    def _w(x):
        if isinstance(x, tuple):  # pragma: no cover
            return tuple((_w(i) for i in x))  # pragma: no cover
        if isinstance(x, list):  # pragma: no cover
            return [_w(i) for i in x]  # pragma: no cover
        return array(x) if hasattr(x, "shape") else x  # pragma: no cover

    try:
        fn = getattr(mops, "min")
    except AttributeError:  # pragma: no cover
        try:
            fn = getattr(mnn, "min")
        except AttributeError:  # pragma: no cover
            try:
                fn = getattr(mrand, "min")
            except AttributeError:  # pragma: no cover  # pragma: no cover
                raise NotImplementedError("Missing in compiler: min")
    res = fn(_u(a, "a"), _u(axis, "axis"), _u(keepdims, "keepdims"))
    return _w(res)


def max(a, axis=None, keepdims=False, stream=None):
    """Compute max.

    Args:
        a: The a argument.
        axis: The axis argument.
        keepdims: The keepdims argument.
        stream: The stream argument.

    Returns:
        The result of max.

    """
    from zero_mlx.array import array, _to_tensor
    import ml_switcheroo.ops as mops
    import ml_switcheroo.nn as mnn
    import ml_switcheroo.random as mrand
    import types

    def _u(x, param_name=None):
        if isinstance(x, list):  # pragma: no cover
            if __import__("builtins").all(
                (isinstance(i, int) for i in x)
            ):  # pragma: no cover
                return tuple(x)  # pragma: no cover
            return [_u(i, param_name) for i in x]  # pragma: no cover
        if isinstance(x, tuple):  # pragma: no cover
            return tuple((_u(i, param_name) for i in x))  # pragma: no cover
        if hasattr(x, "_tensor"):  # pragma: no cover
            return x._tensor  # pragma: no cover
        if isinstance(x, (int, float, bool, complex)):  # pragma: no cover
            if param_name not in {
                "axis",
                "rtol",
                "n",
                "m",
                "endpoint",
                "offset",
                "a_max",
                "ddof",
                "dims",
                "source",
                "axis2",
                "stop",
                "k",
                "dtype",
                "reps",
                "a_min",
                "sparse",
                "decimals",
                "strides",
                "num",
                "max",
                "indices_or_sections",
                "shape",
                "atol",
                "axis1",
                "step",
                "axes",
                "fill_value",
                "stream",
                "destination",
                "alpha",
                "beta",
                "start",
                "equal_nan",
                "keepdims",
                "min",
                "p",
            } and "max" not in (
                "transpose",
                "moveaxis",
                "reshape",
                "expand_dims",
                "linspace",
                "logcumsumexp",
                "cummin",
                "cummax",
                "cumprod",
                "cumsum",
                "logsumexp",
                "all",
                "any",
                "argmin",
                "argmax",
                "prod",
                "sum",
                "mean",
                "var",
                "std",
                "max",
                "min",
                "tensordot",
            ):  # pragma: no cover
                return _to_tensor(x)  # pragma: no cover
        if type(x).__name__ == "ndarray":  # pragma: no cover
            return _to_tensor(x)  # pragma: no cover
        return x  # pragma: no cover

    def _w(x):
        if isinstance(x, tuple):  # pragma: no cover
            return tuple((_w(i) for i in x))  # pragma: no cover
        if isinstance(x, list):  # pragma: no cover
            return [_w(i) for i in x]  # pragma: no cover
        return array(x) if hasattr(x, "shape") else x  # pragma: no cover

    try:
        fn = getattr(mops, "max")
    except AttributeError:  # pragma: no cover
        try:
            fn = getattr(mnn, "max")
        except AttributeError:  # pragma: no cover
            try:
                fn = getattr(mrand, "max")
            except AttributeError:  # pragma: no cover  # pragma: no cover
                raise NotImplementedError("Missing in compiler: max")
    res = fn(_u(a, "a"), _u(axis, "axis"), _u(keepdims, "keepdims"))
    return _w(res)


def prod(a, axis=None, keepdims=False, stream=None):
    """Compute prod.

    Args:
        a: The a argument.
        axis: The axis argument.
        keepdims: The keepdims argument.
        stream: The stream argument.

    Returns:
        The result of prod.

    """
    from zero_mlx.array import array, _to_tensor
    import ml_switcheroo.ops as mops
    import ml_switcheroo.nn as mnn
    import ml_switcheroo.random as mrand
    import types

    def _u(x, param_name=None):
        if isinstance(x, list):  # pragma: no cover
            if __import__("builtins").all(
                (isinstance(i, int) for i in x)
            ):  # pragma: no cover
                return tuple(x)  # pragma: no cover
            return [_u(i, param_name) for i in x]  # pragma: no cover
        if isinstance(x, tuple):  # pragma: no cover
            return tuple((_u(i, param_name) for i in x))  # pragma: no cover
        if hasattr(x, "_tensor"):  # pragma: no cover
            return x._tensor  # pragma: no cover
        if isinstance(x, (int, float, bool, complex)):  # pragma: no cover
            if param_name not in {
                "axis",
                "rtol",
                "n",
                "m",
                "endpoint",
                "offset",
                "a_max",
                "ddof",
                "dims",
                "source",
                "axis2",
                "stop",
                "k",
                "dtype",
                "reps",
                "a_min",
                "sparse",
                "decimals",
                "strides",
                "num",
                "max",
                "indices_or_sections",
                "shape",
                "atol",
                "axis1",
                "step",
                "axes",
                "fill_value",
                "stream",
                "destination",
                "alpha",
                "beta",
                "start",
                "equal_nan",
                "keepdims",
                "min",
                "p",
            } and "prod" not in (
                "transpose",
                "moveaxis",
                "reshape",
                "expand_dims",
                "linspace",
                "logcumsumexp",
                "cummin",
                "cummax",
                "cumprod",
                "cumsum",
                "logsumexp",
                "all",
                "any",
                "argmin",
                "argmax",
                "prod",
                "sum",
                "mean",
                "var",
                "std",
                "max",
                "min",
                "tensordot",
            ):  # pragma: no cover
                return _to_tensor(x)  # pragma: no cover
        if type(x).__name__ == "ndarray":  # pragma: no cover
            return _to_tensor(x)  # pragma: no cover
        return x  # pragma: no cover

    def _w(x):
        if isinstance(x, tuple):  # pragma: no cover
            return tuple((_w(i) for i in x))  # pragma: no cover
        if isinstance(x, list):  # pragma: no cover
            return [_w(i) for i in x]  # pragma: no cover
        return array(x) if hasattr(x, "shape") else x  # pragma: no cover

    try:
        fn = getattr(mops, "prod")
    except AttributeError:  # pragma: no cover
        try:
            fn = getattr(mnn, "prod")
        except AttributeError:  # pragma: no cover
            try:
                fn = getattr(mrand, "prod")
            except AttributeError:  # pragma: no cover  # pragma: no cover
                raise NotImplementedError("Missing in compiler: prod")
    res = fn(_u(a, "a"), _u(axis, "axis"), _u(keepdims, "keepdims"))
    return _w(res)


def eye(n, m=None, k=0, dtype=None, stream=None):
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
    from zero_mlx.array import array, _to_tensor
    import ml_switcheroo.ops as mops
    import ml_switcheroo.nn as mnn
    import ml_switcheroo.random as mrand
    import types

    def _u(x, param_name=None):
        if isinstance(x, list):  # pragma: no cover
            if __import__("builtins").all(
                (isinstance(i, int) for i in x)
            ):  # pragma: no cover
                return tuple(x)  # pragma: no cover
            return [_u(i, param_name) for i in x]  # pragma: no cover
        if isinstance(x, tuple):  # pragma: no cover
            return tuple((_u(i, param_name) for i in x))  # pragma: no cover
        if hasattr(x, "_tensor"):  # pragma: no cover
            return x._tensor  # pragma: no cover
        if isinstance(x, (int, float, bool, complex)):  # pragma: no cover
            if param_name not in {
                "axis",
                "rtol",
                "n",
                "m",
                "endpoint",
                "offset",
                "a_max",
                "ddof",
                "dims",
                "source",
                "axis2",
                "stop",
                "k",
                "dtype",
                "reps",
                "a_min",
                "sparse",
                "decimals",
                "strides",
                "num",
                "max",
                "indices_or_sections",
                "shape",
                "atol",
                "axis1",
                "step",
                "axes",
                "fill_value",
                "stream",
                "destination",
                "alpha",
                "beta",
                "start",
                "equal_nan",
                "keepdims",
                "min",
                "p",
            } and "eye" not in (
                "transpose",
                "moveaxis",
                "reshape",
                "expand_dims",
                "linspace",
                "logcumsumexp",
                "cummin",
                "cummax",
                "cumprod",
                "cumsum",
                "logsumexp",
                "all",
                "any",
                "argmin",
                "argmax",
                "prod",
                "sum",
                "mean",
                "var",
                "std",
                "max",
                "min",
                "tensordot",
            ):  # pragma: no cover
                return _to_tensor(x)  # pragma: no cover
        if type(x).__name__ == "ndarray":  # pragma: no cover
            return _to_tensor(x)  # pragma: no cover
        return x  # pragma: no cover

    def _w(x):
        if isinstance(x, tuple):  # pragma: no cover
            return tuple((_w(i) for i in x))  # pragma: no cover
        if isinstance(x, list):  # pragma: no cover
            return [_w(i) for i in x]  # pragma: no cover
        return array(x) if hasattr(x, "shape") else x  # pragma: no cover

    try:
        fn = getattr(mops, "eye")
    except AttributeError:  # pragma: no cover
        try:
            fn = getattr(mnn, "eye")
        except AttributeError:  # pragma: no cover
            try:
                fn = getattr(mrand, "eye")
            except AttributeError:  # pragma: no cover  # pragma: no cover
                raise NotImplementedError("Missing in compiler: eye")
    res = fn(_u(n, "n"), _u(m, "m"), _u(k, "k"), _u(dtype, "dtype"))
    return _w(res)


def diag(v, k=0, stream=None):
    """Compute diag.

    Args:
        v: The v argument.
        k: The k argument.
        stream: The stream argument.

    Returns:
        The result of diag.

    """
    from zero_mlx.array import array, _to_tensor
    import ml_switcheroo.ops as mops
    import ml_switcheroo.nn as mnn
    import ml_switcheroo.random as mrand
    import types

    def _u(x, param_name=None):
        if isinstance(x, list):  # pragma: no cover
            if __import__("builtins").all(
                (isinstance(i, int) for i in x)
            ):  # pragma: no cover
                return tuple(x)  # pragma: no cover
            return [_u(i, param_name) for i in x]  # pragma: no cover
        if isinstance(x, tuple):  # pragma: no cover
            return tuple((_u(i, param_name) for i in x))  # pragma: no cover
        if hasattr(x, "_tensor"):  # pragma: no cover
            return x._tensor  # pragma: no cover
        if isinstance(x, (int, float, bool, complex)):  # pragma: no cover
            if param_name not in {
                "axis",
                "rtol",
                "n",
                "m",
                "endpoint",
                "offset",
                "a_max",
                "ddof",
                "dims",
                "source",
                "axis2",
                "stop",
                "k",
                "dtype",
                "reps",
                "a_min",
                "sparse",
                "decimals",
                "strides",
                "num",
                "max",
                "indices_or_sections",
                "shape",
                "atol",
                "axis1",
                "step",
                "axes",
                "fill_value",
                "stream",
                "destination",
                "alpha",
                "beta",
                "start",
                "equal_nan",
                "keepdims",
                "min",
                "p",
            } and "diag" not in (
                "transpose",
                "moveaxis",
                "reshape",
                "expand_dims",
                "linspace",
                "logcumsumexp",
                "cummin",
                "cummax",
                "cumprod",
                "cumsum",
                "logsumexp",
                "all",
                "any",
                "argmin",
                "argmax",
                "prod",
                "sum",
                "mean",
                "var",
                "std",
                "max",
                "min",
                "tensordot",
            ):  # pragma: no cover
                return _to_tensor(x)  # pragma: no cover
        if type(x).__name__ == "ndarray":  # pragma: no cover
            return _to_tensor(x)  # pragma: no cover
        return x  # pragma: no cover

    def _w(x):
        if isinstance(x, tuple):  # pragma: no cover
            return tuple((_w(i) for i in x))  # pragma: no cover
        if isinstance(x, list):  # pragma: no cover
            return [_w(i) for i in x]  # pragma: no cover
        return array(x) if hasattr(x, "shape") else x  # pragma: no cover

    try:
        fn = getattr(mops, "diag")
    except AttributeError:  # pragma: no cover
        try:
            fn = getattr(mnn, "diag")
        except AttributeError:  # pragma: no cover
            try:
                fn = getattr(mrand, "diag")
            except AttributeError:  # pragma: no cover  # pragma: no cover
                raise NotImplementedError("Missing in compiler: diag")
    res = fn(_u(v, "v"), _u(k, "k"))
    return _w(res)


def tril(m, k=0, stream=None):
    """Compute tril.

    Args:
        m: The m argument.
        k: The k argument.
        stream: The stream argument.

    Returns:
        The result of tril.

    """
    from zero_mlx.array import array, _to_tensor
    import ml_switcheroo.ops as mops
    import ml_switcheroo.nn as mnn
    import ml_switcheroo.random as mrand
    import types

    def _u(x, param_name=None):
        if isinstance(x, list):  # pragma: no cover
            if __import__("builtins").all(
                (isinstance(i, int) for i in x)
            ):  # pragma: no cover
                return tuple(x)  # pragma: no cover
            return [_u(i, param_name) for i in x]  # pragma: no cover
        if isinstance(x, tuple):  # pragma: no cover
            return tuple((_u(i, param_name) for i in x))  # pragma: no cover
        if hasattr(x, "_tensor"):  # pragma: no cover
            return x._tensor  # pragma: no cover
        if isinstance(x, (int, float, bool, complex)):  # pragma: no cover
            if param_name not in {
                "axis",
                "rtol",
                "n",
                "m",
                "endpoint",
                "offset",
                "a_max",
                "ddof",
                "dims",
                "source",
                "axis2",
                "stop",
                "k",
                "dtype",
                "reps",
                "a_min",
                "sparse",
                "decimals",
                "strides",
                "num",
                "max",
                "indices_or_sections",
                "shape",
                "atol",
                "axis1",
                "step",
                "axes",
                "fill_value",
                "stream",
                "destination",
                "alpha",
                "beta",
                "start",
                "equal_nan",
                "keepdims",
                "min",
                "p",
            } and "tril" not in (
                "transpose",
                "moveaxis",
                "reshape",
                "expand_dims",
                "linspace",
                "logcumsumexp",
                "cummin",
                "cummax",
                "cumprod",
                "cumsum",
                "logsumexp",
                "all",
                "any",
                "argmin",
                "argmax",
                "prod",
                "sum",
                "mean",
                "var",
                "std",
                "max",
                "min",
                "tensordot",
            ):  # pragma: no cover
                return _to_tensor(x)  # pragma: no cover
        if type(x).__name__ == "ndarray":  # pragma: no cover
            return _to_tensor(x)  # pragma: no cover
        return x  # pragma: no cover

    def _w(x):
        if isinstance(x, tuple):  # pragma: no cover
            return tuple((_w(i) for i in x))  # pragma: no cover
        if isinstance(x, list):  # pragma: no cover
            return [_w(i) for i in x]  # pragma: no cover
        return array(x) if hasattr(x, "shape") else x  # pragma: no cover

    try:
        fn = getattr(mops, "tril")
    except AttributeError:  # pragma: no cover
        try:
            fn = getattr(mnn, "tril")
        except AttributeError:  # pragma: no cover
            try:
                fn = getattr(mrand, "tril")
            except AttributeError:  # pragma: no cover  # pragma: no cover
                raise NotImplementedError("Missing in compiler: tril")
    res = fn(_u(m, "m"), _u(k, "k"))
    return _w(res)


def triu(m, k=0, stream=None):
    """Compute triu.

    Args:
        m: The m argument.
        k: The k argument.
        stream: The stream argument.

    Returns:
        The result of triu.

    """
    from zero_mlx.array import array, _to_tensor
    import ml_switcheroo.ops as mops
    import ml_switcheroo.nn as mnn
    import ml_switcheroo.random as mrand
    import types

    def _u(x, param_name=None):
        if isinstance(x, list):  # pragma: no cover
            if __import__("builtins").all(
                (isinstance(i, int) for i in x)
            ):  # pragma: no cover
                return tuple(x)  # pragma: no cover
            return [_u(i, param_name) for i in x]  # pragma: no cover
        if isinstance(x, tuple):  # pragma: no cover
            return tuple((_u(i, param_name) for i in x))  # pragma: no cover
        if hasattr(x, "_tensor"):  # pragma: no cover
            return x._tensor  # pragma: no cover
        if isinstance(x, (int, float, bool, complex)):  # pragma: no cover
            if param_name not in {
                "axis",
                "rtol",
                "n",
                "m",
                "endpoint",
                "offset",
                "a_max",
                "ddof",
                "dims",
                "source",
                "axis2",
                "stop",
                "k",
                "dtype",
                "reps",
                "a_min",
                "sparse",
                "decimals",
                "strides",
                "num",
                "max",
                "indices_or_sections",
                "shape",
                "atol",
                "axis1",
                "step",
                "axes",
                "fill_value",
                "stream",
                "destination",
                "alpha",
                "beta",
                "start",
                "equal_nan",
                "keepdims",
                "min",
                "p",
            } and "triu" not in (
                "transpose",
                "moveaxis",
                "reshape",
                "expand_dims",
                "linspace",
                "logcumsumexp",
                "cummin",
                "cummax",
                "cumprod",
                "cumsum",
                "logsumexp",
                "all",
                "any",
                "argmin",
                "argmax",
                "prod",
                "sum",
                "mean",
                "var",
                "std",
                "max",
                "min",
                "tensordot",
            ):  # pragma: no cover
                return _to_tensor(x)  # pragma: no cover
        if type(x).__name__ == "ndarray":  # pragma: no cover
            return _to_tensor(x)  # pragma: no cover
        return x  # pragma: no cover

    def _w(x):
        if isinstance(x, tuple):  # pragma: no cover
            return tuple((_w(i) for i in x))  # pragma: no cover
        if isinstance(x, list):  # pragma: no cover
            return [_w(i) for i in x]  # pragma: no cover
        return array(x) if hasattr(x, "shape") else x  # pragma: no cover

    try:
        fn = getattr(mops, "triu")
    except AttributeError:  # pragma: no cover
        try:
            fn = getattr(mnn, "triu")
        except AttributeError:  # pragma: no cover
            try:
                fn = getattr(mrand, "triu")
            except AttributeError:  # pragma: no cover  # pragma: no cover
                raise NotImplementedError("Missing in compiler: triu")
    res = fn(_u(m, "m"), _u(k, "k"))
    return _w(res)


def expand_dims(a, axis, stream=None):
    """Compute expand_dims.

    Args:
        a: The a argument.
        axis: The axis argument.
        stream: The stream argument.

    Returns:
        The result of expand_dims.

    """
    from zero_mlx.array import array, _to_tensor
    import ml_switcheroo.ops as mops
    import ml_switcheroo.nn as mnn
    import ml_switcheroo.random as mrand
    import types

    def _u(x, param_name=None):
        if isinstance(x, list):  # pragma: no cover
            if __import__("builtins").all(
                (isinstance(i, int) for i in x)
            ):  # pragma: no cover
                return tuple(x)  # pragma: no cover
            return [_u(i, param_name) for i in x]  # pragma: no cover
        if isinstance(x, tuple):  # pragma: no cover
            return tuple((_u(i, param_name) for i in x))  # pragma: no cover
        if hasattr(x, "_tensor"):  # pragma: no cover
            return x._tensor  # pragma: no cover
        if isinstance(x, (int, float, bool, complex)):  # pragma: no cover
            if param_name not in {
                "axis",
                "rtol",
                "n",
                "m",
                "endpoint",
                "offset",
                "a_max",
                "ddof",
                "dims",
                "source",
                "axis2",
                "stop",
                "k",
                "dtype",
                "reps",
                "a_min",
                "sparse",
                "decimals",
                "strides",
                "num",
                "max",
                "indices_or_sections",
                "shape",
                "atol",
                "axis1",
                "step",
                "axes",
                "fill_value",
                "stream",
                "destination",
                "alpha",
                "beta",
                "start",
                "equal_nan",
                "keepdims",
                "min",
                "p",
            } and "expand_dims" not in (
                "transpose",
                "moveaxis",
                "reshape",
                "expand_dims",
                "linspace",
                "logcumsumexp",
                "cummin",
                "cummax",
                "cumprod",
                "cumsum",
                "logsumexp",
                "all",
                "any",
                "argmin",
                "argmax",
                "prod",
                "sum",
                "mean",
                "var",
                "std",
                "max",
                "min",
                "tensordot",
            ):  # pragma: no cover
                return _to_tensor(x)  # pragma: no cover
        if type(x).__name__ == "ndarray":  # pragma: no cover
            return _to_tensor(x)  # pragma: no cover
        return x  # pragma: no cover

    def _w(x):
        if isinstance(x, tuple):  # pragma: no cover
            return tuple((_w(i) for i in x))  # pragma: no cover
        if isinstance(x, list):  # pragma: no cover
            return [_w(i) for i in x]  # pragma: no cover
        return array(x) if hasattr(x, "shape") else x  # pragma: no cover

    try:
        fn = getattr(mops, "unsqueeze")
    except AttributeError:  # pragma: no cover
        try:
            fn = getattr(mnn, "unsqueeze")
        except AttributeError:  # pragma: no cover
            try:
                fn = getattr(mrand, "unsqueeze")
            except AttributeError:  # pragma: no cover  # pragma: no cover
                raise NotImplementedError("Missing in compiler: unsqueeze")
    res = fn(_u(a, "a"), _u(axis, "axis"))
    return _w(res)


def take_along_axis(a, indices, axis=None, stream=None):
    """Compute take_along_axis.

    Args:
        a: The a argument.
        indices: The indices argument.
        axis: The axis argument.
        stream: The stream argument.

    Returns:
        The result of take_along_axis.

    """
    from zero_mlx.array import array, _to_tensor
    import ml_switcheroo.ops as mops
    import ml_switcheroo.nn as mnn
    import ml_switcheroo.random as mrand
    import types

    def _u(x, param_name=None):
        if isinstance(x, list):  # pragma: no cover
            if __import__("builtins").all(
                (isinstance(i, int) for i in x)
            ):  # pragma: no cover
                return tuple(x)  # pragma: no cover
            return [_u(i, param_name) for i in x]  # pragma: no cover
        if isinstance(x, tuple):  # pragma: no cover
            return tuple((_u(i, param_name) for i in x))  # pragma: no cover
        if hasattr(x, "_tensor"):  # pragma: no cover
            return x._tensor  # pragma: no cover
        if isinstance(x, (int, float, bool, complex)):  # pragma: no cover
            if param_name not in {
                "axis",
                "rtol",
                "n",
                "m",
                "endpoint",
                "offset",
                "a_max",
                "ddof",
                "dims",
                "source",
                "axis2",
                "stop",
                "k",
                "dtype",
                "reps",
                "a_min",
                "sparse",
                "decimals",
                "strides",
                "num",
                "max",
                "indices_or_sections",
                "shape",
                "atol",
                "axis1",
                "step",
                "axes",
                "fill_value",
                "stream",
                "destination",
                "alpha",
                "beta",
                "start",
                "equal_nan",
                "keepdims",
                "min",
                "p",
            } and "take_along_axis" not in (
                "transpose",
                "moveaxis",
                "reshape",
                "expand_dims",
                "linspace",
                "logcumsumexp",
                "cummin",
                "cummax",
                "cumprod",
                "cumsum",
                "logsumexp",
                "all",
                "any",
                "argmin",
                "argmax",
                "prod",
                "sum",
                "mean",
                "var",
                "std",
                "max",
                "min",
                "tensordot",
            ):  # pragma: no cover
                return _to_tensor(x)  # pragma: no cover
        if type(x).__name__ == "ndarray":  # pragma: no cover
            return _to_tensor(x)  # pragma: no cover
        return x  # pragma: no cover

    def _w(x):
        if isinstance(x, tuple):  # pragma: no cover
            return tuple((_w(i) for i in x))  # pragma: no cover
        if isinstance(x, list):  # pragma: no cover
            return [_w(i) for i in x]  # pragma: no cover
        return array(x) if hasattr(x, "shape") else x  # pragma: no cover

    try:
        fn = getattr(mops, "take_along_axis")
    except AttributeError:  # pragma: no cover
        try:
            fn = getattr(mnn, "take_along_axis")
        except AttributeError:  # pragma: no cover
            try:
                fn = getattr(mrand, "take_along_axis")
            except AttributeError:  # pragma: no cover  # pragma: no cover
                raise NotImplementedError("Missing in compiler: take_along_axis")
    res = fn(_u(a, "a"), _u(indices, "indices"), _u(axis, "axis"))
    return _w(res)


def tile(A, reps, stream=None):
    """Compute tile.

    Args:
        A: The A argument.
        reps: The reps argument.
        stream: The stream argument.

    Returns:
        The result of tile.

    """
    from zero_mlx.array import array, _to_tensor
    import ml_switcheroo.ops as mops
    import ml_switcheroo.nn as mnn
    import ml_switcheroo.random as mrand
    import types

    def _u(x, param_name=None):
        if isinstance(x, list):  # pragma: no cover
            if __import__("builtins").all(
                (isinstance(i, int) for i in x)
            ):  # pragma: no cover
                return tuple(x)  # pragma: no cover
            return [_u(i, param_name) for i in x]  # pragma: no cover
        if isinstance(x, tuple):  # pragma: no cover
            return tuple((_u(i, param_name) for i in x))  # pragma: no cover
        if hasattr(x, "_tensor"):  # pragma: no cover
            return x._tensor  # pragma: no cover
        if isinstance(x, (int, float, bool, complex)):  # pragma: no cover
            if param_name not in {
                "axis",
                "rtol",
                "n",
                "m",
                "endpoint",
                "offset",
                "a_max",
                "ddof",
                "dims",
                "source",
                "axis2",
                "stop",
                "k",
                "dtype",
                "reps",
                "a_min",
                "sparse",
                "decimals",
                "strides",
                "num",
                "max",
                "indices_or_sections",
                "shape",
                "atol",
                "axis1",
                "step",
                "axes",
                "fill_value",
                "stream",
                "destination",
                "alpha",
                "beta",
                "start",
                "equal_nan",
                "keepdims",
                "min",
                "p",
            } and "tile" not in (
                "transpose",
                "moveaxis",
                "reshape",
                "expand_dims",
                "linspace",
                "logcumsumexp",
                "cummin",
                "cummax",
                "cumprod",
                "cumsum",
                "logsumexp",
                "all",
                "any",
                "argmin",
                "argmax",
                "prod",
                "sum",
                "mean",
                "var",
                "std",
                "max",
                "min",
                "tensordot",
            ):  # pragma: no cover
                return _to_tensor(x)  # pragma: no cover
        if type(x).__name__ == "ndarray":  # pragma: no cover
            return _to_tensor(x)  # pragma: no cover
        return x  # pragma: no cover

    def _w(x):
        if isinstance(x, tuple):  # pragma: no cover
            return tuple((_w(i) for i in x))  # pragma: no cover
        if isinstance(x, list):  # pragma: no cover
            return [_w(i) for i in x]  # pragma: no cover
        return array(x) if hasattr(x, "shape") else x  # pragma: no cover

    try:
        fn = getattr(mops, "tile")
    except AttributeError:  # pragma: no cover
        try:
            fn = getattr(mnn, "tile")
        except AttributeError:  # pragma: no cover
            try:
                fn = getattr(mrand, "tile")
            except AttributeError:  # pragma: no cover  # pragma: no cover
                raise NotImplementedError("Missing in compiler: tile")
    res = fn(_u(A, "A"), _u(reps, "reps"))
    return _w(res)


def squeeze(a, axis=None, stream=None):
    """Compute squeeze.

    Args:
        a: The a argument.
        axis: The axis argument.
        stream: The stream argument.

    Returns:
        The result of squeeze.

    """
    from zero_mlx.array import array, _to_tensor
    import ml_switcheroo.ops as mops
    import ml_switcheroo.nn as mnn
    import ml_switcheroo.random as mrand
    import types

    def _u(x, param_name=None):
        if isinstance(x, list):  # pragma: no cover
            if __import__("builtins").all(
                (isinstance(i, int) for i in x)
            ):  # pragma: no cover
                return tuple(x)  # pragma: no cover
            return [_u(i, param_name) for i in x]  # pragma: no cover
        if isinstance(x, tuple):  # pragma: no cover
            return tuple((_u(i, param_name) for i in x))  # pragma: no cover
        if hasattr(x, "_tensor"):  # pragma: no cover
            return x._tensor  # pragma: no cover
        if isinstance(x, (int, float, bool, complex)):  # pragma: no cover
            if param_name not in {
                "axis",
                "rtol",
                "n",
                "m",
                "endpoint",
                "offset",
                "a_max",
                "ddof",
                "dims",
                "source",
                "axis2",
                "stop",
                "k",
                "dtype",
                "reps",
                "a_min",
                "sparse",
                "decimals",
                "strides",
                "num",
                "max",
                "indices_or_sections",
                "shape",
                "atol",
                "axis1",
                "step",
                "axes",
                "fill_value",
                "stream",
                "destination",
                "alpha",
                "beta",
                "start",
                "equal_nan",
                "keepdims",
                "min",
                "p",
            } and "squeeze" not in (
                "transpose",
                "moveaxis",
                "reshape",
                "expand_dims",
                "linspace",
                "logcumsumexp",
                "cummin",
                "cummax",
                "cumprod",
                "cumsum",
                "logsumexp",
                "all",
                "any",
                "argmin",
                "argmax",
                "prod",
                "sum",
                "mean",
                "var",
                "std",
                "max",
                "min",
                "tensordot",
            ):  # pragma: no cover
                return _to_tensor(x)  # pragma: no cover
        if type(x).__name__ == "ndarray":  # pragma: no cover
            return _to_tensor(x)  # pragma: no cover
        return x  # pragma: no cover

    def _w(x):
        if isinstance(x, tuple):  # pragma: no cover
            return tuple((_w(i) for i in x))  # pragma: no cover
        if isinstance(x, list):  # pragma: no cover
            return [_w(i) for i in x]  # pragma: no cover
        return array(x) if hasattr(x, "shape") else x  # pragma: no cover

    try:
        fn = getattr(mops, "squeeze")
    except AttributeError:  # pragma: no cover
        try:
            fn = getattr(mnn, "squeeze")
        except AttributeError:  # pragma: no cover
            try:
                fn = getattr(mrand, "squeeze")
            except AttributeError:  # pragma: no cover  # pragma: no cover
                raise NotImplementedError("Missing in compiler: squeeze")
    res = fn(_u(a, "a"), _u(axis, "axis"))
    return _w(res)


def var(a, axis=None, keepdims=False, ddof=0, stream=None):
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
    from zero_mlx.array import array, _to_tensor
    import ml_switcheroo.ops as mops
    import ml_switcheroo.nn as mnn
    import ml_switcheroo.random as mrand
    import types

    def _u(x, param_name=None):
        if isinstance(x, list):  # pragma: no cover
            if __import__("builtins").all(
                (isinstance(i, int) for i in x)
            ):  # pragma: no cover
                return tuple(x)  # pragma: no cover
            return [_u(i, param_name) for i in x]  # pragma: no cover
        if isinstance(x, tuple):  # pragma: no cover
            return tuple((_u(i, param_name) for i in x))  # pragma: no cover
        if hasattr(x, "_tensor"):  # pragma: no cover
            return x._tensor  # pragma: no cover
        if isinstance(x, (int, float, bool, complex)):  # pragma: no cover
            if param_name not in {
                "axis",
                "rtol",
                "n",
                "m",
                "endpoint",
                "offset",
                "a_max",
                "ddof",
                "dims",
                "source",
                "axis2",
                "stop",
                "k",
                "dtype",
                "reps",
                "a_min",
                "sparse",
                "decimals",
                "strides",
                "num",
                "max",
                "indices_or_sections",
                "shape",
                "atol",
                "axis1",
                "step",
                "axes",
                "fill_value",
                "stream",
                "destination",
                "alpha",
                "beta",
                "start",
                "equal_nan",
                "keepdims",
                "min",
                "p",
            } and "var" not in (
                "transpose",
                "moveaxis",
                "reshape",
                "expand_dims",
                "linspace",
                "logcumsumexp",
                "cummin",
                "cummax",
                "cumprod",
                "cumsum",
                "logsumexp",
                "all",
                "any",
                "argmin",
                "argmax",
                "prod",
                "sum",
                "mean",
                "var",
                "std",
                "max",
                "min",
                "tensordot",
            ):  # pragma: no cover
                return _to_tensor(x)  # pragma: no cover
        if type(x).__name__ == "ndarray":  # pragma: no cover
            return _to_tensor(x)  # pragma: no cover
        return x  # pragma: no cover

    def _w(x):
        if isinstance(x, tuple):  # pragma: no cover
            return tuple((_w(i) for i in x))  # pragma: no cover
        if isinstance(x, list):  # pragma: no cover
            return [_w(i) for i in x]  # pragma: no cover
        return array(x) if hasattr(x, "shape") else x  # pragma: no cover

    try:
        fn = getattr(mops, "variance")
    except AttributeError:  # pragma: no cover
        try:
            fn = getattr(mnn, "variance")
        except AttributeError:  # pragma: no cover
            try:
                fn = getattr(mrand, "variance")
            except AttributeError:  # pragma: no cover  # pragma: no cover
                raise NotImplementedError("Missing in compiler: variance")
    res = fn(_u(a, "a"), _u(axis, "axis"), _u(keepdims, "keepdims"), _u(ddof, "ddof"))
    return _w(res)


def std(a, axis=None, keepdims=False, ddof=0, stream=None):
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
    from zero_mlx.array import array, _to_tensor
    import ml_switcheroo.ops as mops
    import ml_switcheroo.nn as mnn
    import ml_switcheroo.random as mrand
    import types

    def _u(x, param_name=None):
        if isinstance(x, list):  # pragma: no cover
            if __import__("builtins").all(
                (isinstance(i, int) for i in x)
            ):  # pragma: no cover
                return tuple(x)  # pragma: no cover
            return [_u(i, param_name) for i in x]  # pragma: no cover
        if isinstance(x, tuple):  # pragma: no cover
            return tuple((_u(i, param_name) for i in x))  # pragma: no cover
        if hasattr(x, "_tensor"):  # pragma: no cover
            return x._tensor  # pragma: no cover
        if isinstance(x, (int, float, bool, complex)):  # pragma: no cover
            if param_name not in {
                "axis",
                "rtol",
                "n",
                "m",
                "endpoint",
                "offset",
                "a_max",
                "ddof",
                "dims",
                "source",
                "axis2",
                "stop",
                "k",
                "dtype",
                "reps",
                "a_min",
                "sparse",
                "decimals",
                "strides",
                "num",
                "max",
                "indices_or_sections",
                "shape",
                "atol",
                "axis1",
                "step",
                "axes",
                "fill_value",
                "stream",
                "destination",
                "alpha",
                "beta",
                "start",
                "equal_nan",
                "keepdims",
                "min",
                "p",
            } and "std" not in (
                "transpose",
                "moveaxis",
                "reshape",
                "expand_dims",
                "linspace",
                "logcumsumexp",
                "cummin",
                "cummax",
                "cumprod",
                "cumsum",
                "logsumexp",
                "all",
                "any",
                "argmin",
                "argmax",
                "prod",
                "sum",
                "mean",
                "var",
                "std",
                "max",
                "min",
                "tensordot",
            ):  # pragma: no cover
                return _to_tensor(x)  # pragma: no cover
        if type(x).__name__ == "ndarray":  # pragma: no cover
            return _to_tensor(x)  # pragma: no cover
        return x  # pragma: no cover

    def _w(x):
        if isinstance(x, tuple):  # pragma: no cover
            return tuple((_w(i) for i in x))  # pragma: no cover
        if isinstance(x, list):  # pragma: no cover
            return [_w(i) for i in x]  # pragma: no cover
        return array(x) if hasattr(x, "shape") else x  # pragma: no cover

    try:
        fn = getattr(mops, "std")
    except AttributeError:  # pragma: no cover
        try:
            fn = getattr(mnn, "std")
        except AttributeError:  # pragma: no cover
            try:
                fn = getattr(mrand, "std")
            except AttributeError:  # pragma: no cover  # pragma: no cover
                raise NotImplementedError("Missing in compiler: std")
    res = fn(_u(a, "a"), _u(axis, "axis"), _u(keepdims, "keepdims"), _u(ddof, "ddof"))
    return _w(res)


def round(a, decimals=0, stream=None):
    """Compute round.

    Args:
        a: The a argument.
        decimals: The decimals argument.
        stream: The stream argument.

    Returns:
        The result of round.

    """
    from zero_mlx.array import array, _to_tensor
    import ml_switcheroo.ops as mops
    import ml_switcheroo.nn as mnn
    import ml_switcheroo.random as mrand
    import types

    def _u(x, param_name=None):
        if isinstance(x, list):  # pragma: no cover
            if __import__("builtins").all(
                (isinstance(i, int) for i in x)
            ):  # pragma: no cover
                return tuple(x)  # pragma: no cover
            return [_u(i, param_name) for i in x]  # pragma: no cover
        if isinstance(x, tuple):  # pragma: no cover
            return tuple((_u(i, param_name) for i in x))  # pragma: no cover
        if hasattr(x, "_tensor"):  # pragma: no cover
            return x._tensor  # pragma: no cover
        if isinstance(x, (int, float, bool, complex)):  # pragma: no cover
            if param_name not in {
                "axis",
                "rtol",
                "n",
                "m",
                "endpoint",
                "offset",
                "a_max",
                "ddof",
                "dims",
                "source",
                "axis2",
                "stop",
                "k",
                "dtype",
                "reps",
                "a_min",
                "sparse",
                "decimals",
                "strides",
                "num",
                "max",
                "indices_or_sections",
                "shape",
                "atol",
                "axis1",
                "step",
                "axes",
                "fill_value",
                "stream",
                "destination",
                "alpha",
                "beta",
                "start",
                "equal_nan",
                "keepdims",
                "min",
                "p",
            } and "round" not in (
                "transpose",
                "moveaxis",
                "reshape",
                "expand_dims",
                "linspace",
                "logcumsumexp",
                "cummin",
                "cummax",
                "cumprod",
                "cumsum",
                "logsumexp",
                "all",
                "any",
                "argmin",
                "argmax",
                "prod",
                "sum",
                "mean",
                "var",
                "std",
                "max",
                "min",
                "tensordot",
            ):  # pragma: no cover
                return _to_tensor(x)  # pragma: no cover
        if type(x).__name__ == "ndarray":  # pragma: no cover
            return _to_tensor(x)  # pragma: no cover
        return x  # pragma: no cover

    def _w(x):
        if isinstance(x, tuple):  # pragma: no cover
            return tuple((_w(i) for i in x))  # pragma: no cover
        if isinstance(x, list):  # pragma: no cover
            return [_w(i) for i in x]  # pragma: no cover
        return array(x) if hasattr(x, "shape") else x  # pragma: no cover

    try:
        fn = getattr(mops, "round")
    except AttributeError:  # pragma: no cover
        try:
            fn = getattr(mnn, "round")
        except AttributeError:  # pragma: no cover
            try:
                fn = getattr(mrand, "round")
            except AttributeError:  # pragma: no cover  # pragma: no cover
                raise NotImplementedError("Missing in compiler: round")
    res = fn(_u(a, "a"), _u(decimals, "decimals"))
    return _w(res)


def sort(a, axis=-1, stream=None):
    """Compute sort.

    Args:
        a: The a argument.
        axis: The axis argument.
        stream: The stream argument.

    Returns:
        The result of sort.

    """
    from zero_mlx.array import array, _to_tensor
    import ml_switcheroo.ops as mops
    import ml_switcheroo.nn as mnn
    import ml_switcheroo.random as mrand
    import types

    def _u(x, param_name=None):
        if isinstance(x, list):  # pragma: no cover
            if __import__("builtins").all(
                (isinstance(i, int) for i in x)
            ):  # pragma: no cover
                return tuple(x)  # pragma: no cover
            return [_u(i, param_name) for i in x]  # pragma: no cover
        if isinstance(x, tuple):  # pragma: no cover
            return tuple((_u(i, param_name) for i in x))  # pragma: no cover
        if hasattr(x, "_tensor"):  # pragma: no cover
            return x._tensor  # pragma: no cover
        if isinstance(x, (int, float, bool, complex)):  # pragma: no cover
            if param_name not in {
                "axis",
                "rtol",
                "n",
                "m",
                "endpoint",
                "offset",
                "a_max",
                "ddof",
                "dims",
                "source",
                "axis2",
                "stop",
                "k",
                "dtype",
                "reps",
                "a_min",
                "sparse",
                "decimals",
                "strides",
                "num",
                "max",
                "indices_or_sections",
                "shape",
                "atol",
                "axis1",
                "step",
                "axes",
                "fill_value",
                "stream",
                "destination",
                "alpha",
                "beta",
                "start",
                "equal_nan",
                "keepdims",
                "min",
                "p",
            } and "sort" not in (
                "transpose",
                "moveaxis",
                "reshape",
                "expand_dims",
                "linspace",
                "logcumsumexp",
                "cummin",
                "cummax",
                "cumprod",
                "cumsum",
                "logsumexp",
                "all",
                "any",
                "argmin",
                "argmax",
                "prod",
                "sum",
                "mean",
                "var",
                "std",
                "max",
                "min",
                "tensordot",
            ):  # pragma: no cover
                return _to_tensor(x)  # pragma: no cover
        if type(x).__name__ == "ndarray":  # pragma: no cover
            return _to_tensor(x)  # pragma: no cover
        return x  # pragma: no cover

    def _w(x):
        if isinstance(x, tuple):  # pragma: no cover
            return tuple((_w(i) for i in x))  # pragma: no cover
        if isinstance(x, list):  # pragma: no cover
            return [_w(i) for i in x]  # pragma: no cover
        return array(x) if hasattr(x, "shape") else x  # pragma: no cover

    try:
        fn = getattr(mops, "sort")
    except AttributeError:  # pragma: no cover
        try:
            fn = getattr(mnn, "sort")
        except AttributeError:  # pragma: no cover
            try:
                fn = getattr(mrand, "sort")
            except AttributeError:  # pragma: no cover  # pragma: no cover
                raise NotImplementedError("Missing in compiler: sort")
    res = fn(_u(a, "a"), _u(axis, "axis"))
    return _w(res)


def argsort(a, axis=-1, stream=None):
    """Compute argsort.

    Args:
        a: The a argument.
        axis: The axis argument.
        stream: The stream argument.

    Returns:
        The result of argsort.

    """
    from zero_mlx.array import array, _to_tensor  # pragma: no cover
    import ml_switcheroo.ops as mops  # pragma: no cover
    import ml_switcheroo.nn as mnn  # pragma: no cover
    import ml_switcheroo.random as mrand  # pragma: no cover
    import types  # pragma: no cover

    def _u(x, param_name=None):  # pragma: no cover
        if isinstance(x, list):  # pragma: no cover
            if __import__("builtins").all(
                (isinstance(i, int) for i in x)
            ):  # pragma: no cover
                return tuple(x)  # pragma: no cover
            return [_u(i, param_name) for i in x]  # pragma: no cover
        if isinstance(x, tuple):  # pragma: no cover
            return tuple((_u(i, param_name) for i in x))  # pragma: no cover
        if hasattr(x, "_tensor"):  # pragma: no cover
            return x._tensor  # pragma: no cover
        if isinstance(x, (int, float, bool, complex)):  # pragma: no cover
            if param_name not in {
                "axis",
                "rtol",
                "n",
                "m",
                "endpoint",
                "offset",
                "a_max",
                "ddof",
                "dims",
                "source",
                "axis2",
                "stop",
                "k",
                "dtype",
                "reps",
                "a_min",
                "sparse",
                "decimals",
                "strides",
                "num",
                "max",
                "indices_or_sections",
                "shape",
                "atol",
                "axis1",
                "step",
                "axes",
                "fill_value",
                "stream",
                "destination",
                "alpha",
                "beta",
                "start",
                "equal_nan",
                "keepdims",
                "min",
                "p",
            } and "argsort" not in (
                "transpose",
                "moveaxis",
                "reshape",
                "expand_dims",
                "linspace",
                "logcumsumexp",
                "cummin",
                "cummax",
                "cumprod",
                "cumsum",
                "logsumexp",
                "all",
                "any",
                "argmin",
                "argmax",
                "prod",
                "sum",
                "mean",
                "var",
                "std",
                "max",
                "min",
                "tensordot",
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
        fn = getattr(mops, "argsort")  # pragma: no cover
    except AttributeError:  # pragma: no cover
        try:
            fn = getattr(mnn, "argsort")
        except AttributeError:  # pragma: no cover
            try:
                fn = getattr(mrand, "argsort")
            except AttributeError:  # pragma: no cover  # pragma: no cover
                raise NotImplementedError("Missing in compiler: argsort")
    res = fn(_u(a, "a"), _u(axis, "axis"))  # pragma: no cover
    return _w(res)  # pragma: no cover


def swapaxes(a, axis1, axis2, stream=None):
    """Compute swapaxes.

    Args:
        a: The a argument.
        axis1: The axis1 argument.
        axis2: The axis2 argument.
        stream: The stream argument.

    Returns:
        The result of swapaxes.

    """
    from zero_mlx.array import array, _to_tensor
    import ml_switcheroo.ops as mops
    import ml_switcheroo.nn as mnn
    import ml_switcheroo.random as mrand
    import types

    def _u(x, param_name=None):
        if isinstance(x, list):  # pragma: no cover
            if __import__("builtins").all(
                (isinstance(i, int) for i in x)
            ):  # pragma: no cover
                return tuple(x)  # pragma: no cover
            return [_u(i, param_name) for i in x]  # pragma: no cover
        if isinstance(x, tuple):  # pragma: no cover
            return tuple((_u(i, param_name) for i in x))  # pragma: no cover
        if hasattr(x, "_tensor"):  # pragma: no cover
            return x._tensor  # pragma: no cover
        if isinstance(x, (int, float, bool, complex)):  # pragma: no cover
            if param_name not in {
                "axis",
                "rtol",
                "n",
                "m",
                "endpoint",
                "offset",
                "a_max",
                "ddof",
                "dims",
                "source",
                "axis2",
                "stop",
                "k",
                "dtype",
                "reps",
                "a_min",
                "sparse",
                "decimals",
                "strides",
                "num",
                "max",
                "indices_or_sections",
                "shape",
                "atol",
                "axis1",
                "step",
                "axes",
                "fill_value",
                "stream",
                "destination",
                "alpha",
                "beta",
                "start",
                "equal_nan",
                "keepdims",
                "min",
                "p",
            } and "swapaxes" not in (
                "transpose",
                "moveaxis",
                "reshape",
                "expand_dims",
                "linspace",
                "logcumsumexp",
                "cummin",
                "cummax",
                "cumprod",
                "cumsum",
                "logsumexp",
                "all",
                "any",
                "argmin",
                "argmax",
                "prod",
                "sum",
                "mean",
                "var",
                "std",
                "max",
                "min",
                "tensordot",
            ):  # pragma: no cover
                return _to_tensor(x)  # pragma: no cover
        if type(x).__name__ == "ndarray":  # pragma: no cover
            return _to_tensor(x)  # pragma: no cover
        return x  # pragma: no cover

    def _w(x):
        if isinstance(x, tuple):  # pragma: no cover
            return tuple((_w(i) for i in x))  # pragma: no cover
        if isinstance(x, list):  # pragma: no cover
            return [_w(i) for i in x]  # pragma: no cover
        return array(x) if hasattr(x, "shape") else x  # pragma: no cover

    try:
        fn = getattr(mops, "swapaxes")
    except AttributeError:  # pragma: no cover
        try:
            fn = getattr(mnn, "swapaxes")
        except AttributeError:  # pragma: no cover
            try:
                fn = getattr(mrand, "swapaxes")
            except AttributeError:  # pragma: no cover  # pragma: no cover
                raise NotImplementedError("Missing in compiler: swapaxes")
    res = fn(_u(a, "a"), _u(axis1, "axis1"), _u(axis2, "axis2"))
    return _w(res)


def moveaxis(a, source, destination, stream=None):
    """Compute moveaxis.

    Args:
        a: The a argument.
        source: The source argument.
        destination: The destination argument.
        stream: The stream argument.

    Returns:
        The result of moveaxis.

    """
    from zero_mlx.array import array, _to_tensor
    import ml_switcheroo.ops as mops
    import ml_switcheroo.nn as mnn
    import ml_switcheroo.random as mrand
    import types

    def _u(x, param_name=None):
        if isinstance(x, list):  # pragma: no cover
            if __import__("builtins").all(
                (isinstance(i, int) for i in x)
            ):  # pragma: no cover
                return tuple(x)  # pragma: no cover
            return [_u(i, param_name) for i in x]  # pragma: no cover
        if isinstance(x, tuple):  # pragma: no cover
            return tuple((_u(i, param_name) for i in x))  # pragma: no cover
        if hasattr(x, "_tensor"):  # pragma: no cover
            return x._tensor  # pragma: no cover
        if isinstance(x, (int, float, bool, complex)):  # pragma: no cover
            if param_name not in {
                "axis",
                "rtol",
                "n",
                "m",
                "endpoint",
                "offset",
                "a_max",
                "ddof",
                "dims",
                "source",
                "axis2",
                "stop",
                "k",
                "dtype",
                "reps",
                "a_min",
                "sparse",
                "decimals",
                "strides",
                "num",
                "max",
                "indices_or_sections",
                "shape",
                "atol",
                "axis1",
                "step",
                "axes",
                "fill_value",
                "stream",
                "destination",
                "alpha",
                "beta",
                "start",
                "equal_nan",
                "keepdims",
                "min",
                "p",
            } and "moveaxis" not in (
                "transpose",
                "moveaxis",
                "reshape",
                "expand_dims",
                "linspace",
                "logcumsumexp",
                "cummin",
                "cummax",
                "cumprod",
                "cumsum",
                "logsumexp",
                "all",
                "any",
                "argmin",
                "argmax",
                "prod",
                "sum",
                "mean",
                "var",
                "std",
                "max",
                "min",
                "tensordot",
            ):  # pragma: no cover
                return _to_tensor(x)  # pragma: no cover
        if type(x).__name__ == "ndarray":  # pragma: no cover
            return _to_tensor(x)  # pragma: no cover
        return x  # pragma: no cover

    def _w(x):
        if isinstance(x, tuple):  # pragma: no cover
            return tuple((_w(i) for i in x))  # pragma: no cover
        if isinstance(x, list):  # pragma: no cover
            return [_w(i) for i in x]  # pragma: no cover
        return array(x) if hasattr(x, "shape") else x  # pragma: no cover

    try:
        fn = getattr(mops, "moveaxis")
    except AttributeError:  # pragma: no cover
        try:
            fn = getattr(mnn, "moveaxis")
        except AttributeError:  # pragma: no cover
            try:
                fn = getattr(mrand, "moveaxis")
            except AttributeError:  # pragma: no cover  # pragma: no cover
                raise NotImplementedError("Missing in compiler: moveaxis")
    res = fn(_u(a, "a"), _u(source, "source"), _u(destination, "destination"))
    return _w(res)


def take(a, indices, axis=None, stream=None):
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
    import ml_switcheroo.ops as mops  # pragma: no cover
    import ml_switcheroo.nn as mnn  # pragma: no cover
    import ml_switcheroo.random as mrand  # pragma: no cover
    import types  # pragma: no cover

    def _u(x, param_name=None):  # pragma: no cover
        if isinstance(x, list):  # pragma: no cover
            if __import__("builtins").all(
                (isinstance(i, int) for i in x)
            ):  # pragma: no cover
                return tuple(x)  # pragma: no cover
            return [_u(i, param_name) for i in x]  # pragma: no cover
        if isinstance(x, tuple):  # pragma: no cover
            return tuple((_u(i, param_name) for i in x))  # pragma: no cover
        if hasattr(x, "_tensor"):  # pragma: no cover
            return x._tensor  # pragma: no cover
        if isinstance(x, (int, float, bool, complex)):  # pragma: no cover
            if param_name not in {
                "axis",
                "rtol",
                "n",
                "m",
                "endpoint",
                "offset",
                "a_max",
                "ddof",
                "dims",
                "source",
                "axis2",
                "stop",
                "k",
                "dtype",
                "reps",
                "a_min",
                "sparse",
                "decimals",
                "strides",
                "num",
                "max",
                "indices_or_sections",
                "shape",
                "atol",
                "axis1",
                "step",
                "axes",
                "fill_value",
                "stream",
                "destination",
                "alpha",
                "beta",
                "start",
                "equal_nan",
                "keepdims",
                "min",
                "p",
            } and "take" not in (
                "transpose",
                "moveaxis",
                "reshape",
                "expand_dims",
                "linspace",
                "logcumsumexp",
                "cummin",
                "cummax",
                "cumprod",
                "cumsum",
                "logsumexp",
                "all",
                "any",
                "argmin",
                "argmax",
                "prod",
                "sum",
                "mean",
                "var",
                "std",
                "max",
                "min",
                "tensordot",
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
        fn = getattr(mops, "take")  # pragma: no cover
    except AttributeError:  # pragma: no cover
        try:
            fn = getattr(mnn, "take")
        except AttributeError:  # pragma: no cover
            try:
                fn = getattr(mrand, "take")
            except AttributeError:  # pragma: no cover  # pragma: no cover
                raise NotImplementedError("Missing in compiler: take")
    res = fn(_u(a, "a"), _u(indices, "indices"), _u(axis, "axis"))  # pragma: no cover
    return _w(res)  # pragma: no cover


def addmm(c, a, b, alpha=1.0, beta=1.0, stream=None):
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
    from zero_mlx.array import array, _to_tensor
    import ml_switcheroo.ops as mops
    import ml_switcheroo.nn as mnn
    import ml_switcheroo.random as mrand
    import types

    def _u(x, param_name=None):
        if isinstance(x, list):  # pragma: no cover
            if __import__("builtins").all(
                (isinstance(i, int) for i in x)
            ):  # pragma: no cover
                return tuple(x)  # pragma: no cover
            return [_u(i, param_name) for i in x]  # pragma: no cover
        if isinstance(x, tuple):  # pragma: no cover
            return tuple((_u(i, param_name) for i in x))  # pragma: no cover
        if hasattr(x, "_tensor"):  # pragma: no cover
            return x._tensor  # pragma: no cover
        if isinstance(x, (int, float, bool, complex)):  # pragma: no cover
            if param_name not in {
                "axis",
                "rtol",
                "n",
                "m",
                "endpoint",
                "offset",
                "a_max",
                "ddof",
                "dims",
                "source",
                "axis2",
                "stop",
                "k",
                "dtype",
                "reps",
                "a_min",
                "sparse",
                "decimals",
                "strides",
                "num",
                "max",
                "indices_or_sections",
                "shape",
                "atol",
                "axis1",
                "step",
                "axes",
                "fill_value",
                "stream",
                "destination",
                "alpha",
                "beta",
                "start",
                "equal_nan",
                "keepdims",
                "min",
                "p",
            } and "addmm" not in (
                "transpose",
                "moveaxis",
                "reshape",
                "expand_dims",
                "linspace",
                "logcumsumexp",
                "cummin",
                "cummax",
                "cumprod",
                "cumsum",
                "logsumexp",
                "all",
                "any",
                "argmin",
                "argmax",
                "prod",
                "sum",
                "mean",
                "var",
                "std",
                "max",
                "min",
                "tensordot",
            ):  # pragma: no cover
                return _to_tensor(x)  # pragma: no cover
        if type(x).__name__ == "ndarray":  # pragma: no cover
            return _to_tensor(x)  # pragma: no cover
        return x  # pragma: no cover

    def _w(x):
        if isinstance(x, tuple):  # pragma: no cover
            return tuple((_w(i) for i in x))  # pragma: no cover
        if isinstance(x, list):  # pragma: no cover
            return [_w(i) for i in x]  # pragma: no cover
        return array(x) if hasattr(x, "shape") else x  # pragma: no cover

    try:
        fn = getattr(mops, "addmm")
    except AttributeError:  # pragma: no cover
        try:
            fn = getattr(mnn, "addmm")
        except AttributeError:  # pragma: no cover
            try:
                fn = getattr(mrand, "addmm")
            except AttributeError:  # pragma: no cover  # pragma: no cover
                raise NotImplementedError("Missing in compiler: addmm")
    res = fn(_u(c, "c"), _u(a, "a"), _u(b, "b"), _u(alpha, "alpha"), _u(beta, "beta"))
    return _w(res)


def gather_mm(a, b, lhs_indices=None, rhs_indices=None, stream=None, **kwargs):
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
    from zero_mlx.array import array, _to_tensor
    import ml_switcheroo.ops as mops
    import ml_switcheroo.nn as mnn
    import ml_switcheroo.random as mrand
    import types

    def _u(x, param_name=None):
        if isinstance(x, list):  # pragma: no cover
            if __import__("builtins").all(
                (isinstance(i, int) for i in x)
            ):  # pragma: no cover
                return tuple(x)  # pragma: no cover
            return [_u(i, param_name) for i in x]  # pragma: no cover
        if isinstance(x, tuple):  # pragma: no cover
            return tuple((_u(i, param_name) for i in x))  # pragma: no cover
        if hasattr(x, "_tensor"):  # pragma: no cover
            return x._tensor  # pragma: no cover
        if isinstance(x, (int, float, bool, complex)):  # pragma: no cover
            if param_name not in {
                "axis",
                "rtol",
                "n",
                "m",
                "endpoint",
                "offset",
                "a_max",
                "ddof",
                "dims",
                "source",
                "axis2",
                "stop",
                "k",
                "dtype",
                "reps",
                "a_min",
                "sparse",
                "decimals",
                "strides",
                "num",
                "max",
                "indices_or_sections",
                "shape",
                "atol",
                "axis1",
                "step",
                "axes",
                "fill_value",
                "stream",
                "destination",
                "alpha",
                "beta",
                "start",
                "equal_nan",
                "keepdims",
                "min",
                "p",
            } and "gather_mm" not in (
                "transpose",
                "moveaxis",
                "reshape",
                "expand_dims",
                "linspace",
                "logcumsumexp",
                "cummin",
                "cummax",
                "cumprod",
                "cumsum",
                "logsumexp",
                "all",
                "any",
                "argmin",
                "argmax",
                "prod",
                "sum",
                "mean",
                "var",
                "std",
                "max",
                "min",
                "tensordot",
            ):  # pragma: no cover
                return _to_tensor(x)  # pragma: no cover
        if type(x).__name__ == "ndarray":  # pragma: no cover
            return _to_tensor(x)  # pragma: no cover
        return x  # pragma: no cover

    def _w(x):
        if isinstance(x, tuple):  # pragma: no cover
            return tuple((_w(i) for i in x))  # pragma: no cover
        if isinstance(x, list):  # pragma: no cover
            return [_w(i) for i in x]  # pragma: no cover
        return array(x) if hasattr(x, "shape") else x  # pragma: no cover

    try:
        fn = getattr(mops, "gather_mm")
    except AttributeError:  # pragma: no cover
        try:
            fn = getattr(mnn, "gather_mm")
        except AttributeError:  # pragma: no cover
            try:
                fn = getattr(mrand, "gather_mm")
            except AttributeError:  # pragma: no cover  # pragma: no cover
                raise NotImplementedError("Missing in compiler: gather_mm")
    res = fn(
        _u(a, "a"),
        _u(b, "b"),
        _u(lhs_indices, "lhs_indices"),
        _u(rhs_indices, "rhs_indices"),
        **{k: _u(v, k) for (k, v) in kwargs.items() if k != "stream"},
    )
    return _w(res)


def block_masked_mm(
    a, b, block_size, out_mask=None, a_mask=None, b_mask=None, stream=None
):
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
    import ml_switcheroo.ops as mops  # pragma: no cover
    import ml_switcheroo.nn as mnn  # pragma: no cover
    import ml_switcheroo.random as mrand  # pragma: no cover
    import types  # pragma: no cover

    def _u(x, param_name=None):  # pragma: no cover
        if isinstance(x, list):  # pragma: no cover
            if __import__("builtins").all(
                (isinstance(i, int) for i in x)
            ):  # pragma: no cover
                return tuple(x)  # pragma: no cover
            return [_u(i, param_name) for i in x]  # pragma: no cover
        if isinstance(x, tuple):  # pragma: no cover
            return tuple((_u(i, param_name) for i in x))  # pragma: no cover
        if hasattr(x, "_tensor"):  # pragma: no cover
            return x._tensor  # pragma: no cover
        if isinstance(x, (int, float, bool, complex)):  # pragma: no cover
            if param_name not in {
                "axis",
                "rtol",
                "n",
                "m",
                "endpoint",
                "offset",
                "a_max",
                "ddof",
                "dims",
                "source",
                "axis2",
                "stop",
                "k",
                "dtype",
                "reps",
                "a_min",
                "sparse",
                "decimals",
                "strides",
                "num",
                "max",
                "indices_or_sections",
                "shape",
                "atol",
                "axis1",
                "step",
                "axes",
                "fill_value",
                "stream",
                "destination",
                "alpha",
                "beta",
                "start",
                "equal_nan",
                "keepdims",
                "min",
                "p",
            } and "block_masked_mm" not in (
                "transpose",
                "moveaxis",
                "reshape",
                "expand_dims",
                "linspace",
                "logcumsumexp",
                "cummin",
                "cummax",
                "cumprod",
                "cumsum",
                "logsumexp",
                "all",
                "any",
                "argmin",
                "argmax",
                "prod",
                "sum",
                "mean",
                "var",
                "std",
                "max",
                "min",
                "tensordot",
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
        fn = getattr(mops, "block_masked_mm")  # pragma: no cover
    except AttributeError:  # pragma: no cover
        try:
            fn = getattr(mnn, "block_masked_mm")
        except AttributeError:  # pragma: no cover
            try:
                fn = getattr(mrand, "block_masked_mm")
            except AttributeError:  # pragma: no cover  # pragma: no cover
                raise NotImplementedError("Missing in compiler: block_masked_mm")
    res = fn(
        _u(a, "a"),
        _u(b, "b"),
        _u(block_size, "block_size"),
        _u(out_mask, "out_mask"),
        _u(a_mask, "a_mask"),
        _u(b_mask, "b_mask"),
    )  # pragma: no cover
    return _w(res)  # pragma: no cover


def segmented_mm(a, b, segments, stream=None):
    """Compute segmented_mm.

    Args:
        a: The a argument.
        b: The b argument.
        segments: The segments argument.
        stream: The stream argument.

    Returns:
        The result of segmented_mm.

    """
    from zero_mlx.array import array, _to_tensor
    import ml_switcheroo.ops as mops
    import ml_switcheroo.nn as mnn
    import ml_switcheroo.random as mrand
    import types

    def _u(x, param_name=None):
        if isinstance(x, list):  # pragma: no cover
            if __import__("builtins").all(
                (isinstance(i, int) for i in x)
            ):  # pragma: no cover
                return tuple(x)  # pragma: no cover
            return [_u(i, param_name) for i in x]  # pragma: no cover
        if isinstance(x, tuple):  # pragma: no cover
            return tuple((_u(i, param_name) for i in x))  # pragma: no cover
        if hasattr(x, "_tensor"):  # pragma: no cover
            return x._tensor  # pragma: no cover
        if isinstance(x, (int, float, bool, complex)):  # pragma: no cover
            if param_name not in {
                "axis",
                "rtol",
                "n",
                "m",
                "endpoint",
                "offset",
                "a_max",
                "ddof",
                "dims",
                "source",
                "axis2",
                "stop",
                "k",
                "dtype",
                "reps",
                "a_min",
                "sparse",
                "decimals",
                "strides",
                "num",
                "max",
                "indices_or_sections",
                "shape",
                "atol",
                "axis1",
                "step",
                "axes",
                "fill_value",
                "stream",
                "destination",
                "alpha",
                "beta",
                "start",
                "equal_nan",
                "keepdims",
                "min",
                "p",
            } and "segmented_mm" not in (
                "transpose",
                "moveaxis",
                "reshape",
                "expand_dims",
                "linspace",
                "logcumsumexp",
                "cummin",
                "cummax",
                "cumprod",
                "cumsum",
                "logsumexp",
                "all",
                "any",
                "argmin",
                "argmax",
                "prod",
                "sum",
                "mean",
                "var",
                "std",
                "max",
                "min",
                "tensordot",
            ):  # pragma: no cover
                return _to_tensor(x)  # pragma: no cover
        if type(x).__name__ == "ndarray":  # pragma: no cover
            return _to_tensor(x)  # pragma: no cover
        return x  # pragma: no cover

    def _w(x):
        if isinstance(x, tuple):  # pragma: no cover
            return tuple((_w(i) for i in x))  # pragma: no cover
        if isinstance(x, list):  # pragma: no cover
            return [_w(i) for i in x]  # pragma: no cover
        return array(x) if hasattr(x, "shape") else x  # pragma: no cover

    try:
        fn = getattr(mops, "segmented_mm")
    except AttributeError:  # pragma: no cover
        try:
            fn = getattr(mnn, "segmented_mm")
        except AttributeError:  # pragma: no cover
            try:
                fn = getattr(mrand, "segmented_mm")
            except AttributeError:  # pragma: no cover  # pragma: no cover
                raise NotImplementedError("Missing in compiler: segmented_mm")
    res = fn(_u(a, "a"), _u(b, "b"), _u(segments, "segments"))
    return _w(res)


def argpartition(*args, **kwargs):
    """Compute argpartition.

    Args:
        args: The args argument.
        kwargs: The kwargs argument.

    Returns:
        The result of argpartition.

    """
    from zero_mlx.array import array, _to_tensor  # pragma: no cover
    import ml_switcheroo.ops as mops  # pragma: no cover
    import ml_switcheroo.nn as mnn  # pragma: no cover
    import ml_switcheroo.random as mrand  # pragma: no cover
    import types  # pragma: no cover

    def _u(x, param_name=None):  # pragma: no cover
        if isinstance(x, list):  # pragma: no cover
            if __import__("builtins").all(
                (isinstance(i, int) for i in x)
            ):  # pragma: no cover
                return tuple(x)  # pragma: no cover
            return [_u(i, param_name) for i in x]  # pragma: no cover
        if isinstance(x, tuple):  # pragma: no cover
            return tuple((_u(i, param_name) for i in x))  # pragma: no cover
        if hasattr(x, "_tensor"):  # pragma: no cover
            return x._tensor  # pragma: no cover
        if isinstance(x, (int, float, bool, complex)):  # pragma: no cover
            if param_name not in {
                "axis",
                "rtol",
                "n",
                "m",
                "endpoint",
                "offset",
                "a_max",
                "ddof",
                "dims",
                "source",
                "axis2",
                "stop",
                "k",
                "dtype",
                "reps",
                "a_min",
                "sparse",
                "decimals",
                "strides",
                "num",
                "max",
                "indices_or_sections",
                "shape",
                "atol",
                "axis1",
                "step",
                "axes",
                "fill_value",
                "stream",
                "destination",
                "alpha",
                "beta",
                "start",
                "equal_nan",
                "keepdims",
                "min",
                "p",
            } and "argpartition" not in (
                "transpose",
                "moveaxis",
                "reshape",
                "expand_dims",
                "linspace",
                "logcumsumexp",
                "cummin",
                "cummax",
                "cumprod",
                "cumsum",
                "logsumexp",
                "all",
                "any",
                "argmin",
                "argmax",
                "prod",
                "sum",
                "mean",
                "var",
                "std",
                "max",
                "min",
                "tensordot",
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
        fn = getattr(mops, "argpartition")  # pragma: no cover
    except AttributeError:  # pragma: no cover
        try:
            fn = getattr(mnn, "argpartition")
        except AttributeError:  # pragma: no cover
            try:
                fn = getattr(mrand, "argpartition")
            except AttributeError:  # pragma: no cover  # pragma: no cover
                raise NotImplementedError("Missing in compiler: argpartition")
    res = fn(
        *[_u(x) for x in args],
        **{k: _u(v, k) for (k, v) in kwargs.items() if k != "stream"},
    )  # pragma: no cover
    return _w(res)  # pragma: no cover


def atleast_1d(*args, **kwargs):
    """Compute atleast_1d.

    Args:
        args: The args argument.
        kwargs: The kwargs argument.

    Returns:
        The result of atleast_1d.

    """
    from zero_mlx.array import array, _to_tensor
    import ml_switcheroo.ops as mops
    import ml_switcheroo.nn as mnn
    import ml_switcheroo.random as mrand
    import types

    def _u(x, param_name=None):
        if isinstance(x, list):  # pragma: no cover
            if __import__("builtins").all(
                (isinstance(i, int) for i in x)
            ):  # pragma: no cover
                return tuple(x)  # pragma: no cover
            return [_u(i, param_name) for i in x]  # pragma: no cover
        if isinstance(x, tuple):  # pragma: no cover
            return tuple((_u(i, param_name) for i in x))  # pragma: no cover
        if hasattr(x, "_tensor"):  # pragma: no cover
            return x._tensor  # pragma: no cover
        if isinstance(x, (int, float, bool, complex)):  # pragma: no cover
            if param_name not in {
                "axis",
                "rtol",
                "n",
                "m",
                "endpoint",
                "offset",
                "a_max",
                "ddof",
                "dims",
                "source",
                "axis2",
                "stop",
                "k",
                "dtype",
                "reps",
                "a_min",
                "sparse",
                "decimals",
                "strides",
                "num",
                "max",
                "indices_or_sections",
                "shape",
                "atol",
                "axis1",
                "step",
                "axes",
                "fill_value",
                "stream",
                "destination",
                "alpha",
                "beta",
                "start",
                "equal_nan",
                "keepdims",
                "min",
                "p",
            } and "atleast_1d" not in (
                "transpose",
                "moveaxis",
                "reshape",
                "expand_dims",
                "linspace",
                "logcumsumexp",
                "cummin",
                "cummax",
                "cumprod",
                "cumsum",
                "logsumexp",
                "all",
                "any",
                "argmin",
                "argmax",
                "prod",
                "sum",
                "mean",
                "var",
                "std",
                "max",
                "min",
                "tensordot",
            ):  # pragma: no cover
                return _to_tensor(x)  # pragma: no cover
        if type(x).__name__ == "ndarray":  # pragma: no cover
            return _to_tensor(x)  # pragma: no cover
        return x  # pragma: no cover

    def _w(x):
        if isinstance(x, tuple):  # pragma: no cover
            return tuple((_w(i) for i in x))  # pragma: no cover
        if isinstance(x, list):  # pragma: no cover
            return [_w(i) for i in x]  # pragma: no cover
        return array(x) if hasattr(x, "shape") else x  # pragma: no cover

    try:
        fn = getattr(mops, "atleast_1d")
    except AttributeError:  # pragma: no cover
        try:
            fn = getattr(mnn, "atleast_1d")
        except AttributeError:  # pragma: no cover
            try:
                fn = getattr(mrand, "atleast_1d")
            except AttributeError:  # pragma: no cover  # pragma: no cover
                raise NotImplementedError("Missing in compiler: atleast_1d")
    res = fn(
        *[_u(x) for x in args],
        **{k: _u(v, k) for (k, v) in kwargs.items() if k != "stream"},
    )
    return _w(res)


def atleast_2d(*args, **kwargs):
    """Compute atleast_2d.

    Args:
        args: The args argument.
        kwargs: The kwargs argument.

    Returns:
        The result of atleast_2d.

    """
    from zero_mlx.array import array, _to_tensor
    import ml_switcheroo.ops as mops
    import ml_switcheroo.nn as mnn
    import ml_switcheroo.random as mrand
    import types

    def _u(x, param_name=None):
        if isinstance(x, list):  # pragma: no cover
            if __import__("builtins").all(
                (isinstance(i, int) for i in x)
            ):  # pragma: no cover
                return tuple(x)  # pragma: no cover
            return [_u(i, param_name) for i in x]  # pragma: no cover
        if isinstance(x, tuple):  # pragma: no cover
            return tuple((_u(i, param_name) for i in x))  # pragma: no cover
        if hasattr(x, "_tensor"):  # pragma: no cover
            return x._tensor  # pragma: no cover
        if isinstance(x, (int, float, bool, complex)):  # pragma: no cover
            if param_name not in {
                "axis",
                "rtol",
                "n",
                "m",
                "endpoint",
                "offset",
                "a_max",
                "ddof",
                "dims",
                "source",
                "axis2",
                "stop",
                "k",
                "dtype",
                "reps",
                "a_min",
                "sparse",
                "decimals",
                "strides",
                "num",
                "max",
                "indices_or_sections",
                "shape",
                "atol",
                "axis1",
                "step",
                "axes",
                "fill_value",
                "stream",
                "destination",
                "alpha",
                "beta",
                "start",
                "equal_nan",
                "keepdims",
                "min",
                "p",
            } and "atleast_2d" not in (
                "transpose",
                "moveaxis",
                "reshape",
                "expand_dims",
                "linspace",
                "logcumsumexp",
                "cummin",
                "cummax",
                "cumprod",
                "cumsum",
                "logsumexp",
                "all",
                "any",
                "argmin",
                "argmax",
                "prod",
                "sum",
                "mean",
                "var",
                "std",
                "max",
                "min",
                "tensordot",
            ):  # pragma: no cover
                return _to_tensor(x)  # pragma: no cover
        if type(x).__name__ == "ndarray":  # pragma: no cover
            return _to_tensor(x)  # pragma: no cover
        return x  # pragma: no cover

    def _w(x):
        if isinstance(x, tuple):  # pragma: no cover
            return tuple((_w(i) for i in x))  # pragma: no cover
        if isinstance(x, list):  # pragma: no cover
            return [_w(i) for i in x]  # pragma: no cover
        return array(x) if hasattr(x, "shape") else x  # pragma: no cover

    try:
        fn = getattr(mops, "atleast_2d")
    except AttributeError:  # pragma: no cover
        try:
            fn = getattr(mnn, "atleast_2d")
        except AttributeError:  # pragma: no cover
            try:
                fn = getattr(mrand, "atleast_2d")
            except AttributeError:  # pragma: no cover  # pragma: no cover
                raise NotImplementedError("Missing in compiler: atleast_2d")
    res = fn(
        *[_u(x) for x in args],
        **{k: _u(v, k) for (k, v) in kwargs.items() if k != "stream"},
    )
    return _w(res)


def atleast_3d(*args, **kwargs):
    """Compute atleast_3d.

    Args:
        args: The args argument.
        kwargs: The kwargs argument.

    Returns:
        The result of atleast_3d.

    """
    from zero_mlx.array import array, _to_tensor
    import ml_switcheroo.ops as mops
    import ml_switcheroo.nn as mnn
    import ml_switcheroo.random as mrand
    import types

    def _u(x, param_name=None):
        if isinstance(x, list):  # pragma: no cover
            if __import__("builtins").all(
                (isinstance(i, int) for i in x)
            ):  # pragma: no cover
                return tuple(x)  # pragma: no cover
            return [_u(i, param_name) for i in x]  # pragma: no cover
        if isinstance(x, tuple):  # pragma: no cover
            return tuple((_u(i, param_name) for i in x))  # pragma: no cover
        if hasattr(x, "_tensor"):  # pragma: no cover
            return x._tensor  # pragma: no cover
        if isinstance(x, (int, float, bool, complex)):  # pragma: no cover
            if param_name not in {
                "axis",
                "rtol",
                "n",
                "m",
                "endpoint",
                "offset",
                "a_max",
                "ddof",
                "dims",
                "source",
                "axis2",
                "stop",
                "k",
                "dtype",
                "reps",
                "a_min",
                "sparse",
                "decimals",
                "strides",
                "num",
                "max",
                "indices_or_sections",
                "shape",
                "atol",
                "axis1",
                "step",
                "axes",
                "fill_value",
                "stream",
                "destination",
                "alpha",
                "beta",
                "start",
                "equal_nan",
                "keepdims",
                "min",
                "p",
            } and "atleast_3d" not in (
                "transpose",
                "moveaxis",
                "reshape",
                "expand_dims",
                "linspace",
                "logcumsumexp",
                "cummin",
                "cummax",
                "cumprod",
                "cumsum",
                "logsumexp",
                "all",
                "any",
                "argmin",
                "argmax",
                "prod",
                "sum",
                "mean",
                "var",
                "std",
                "max",
                "min",
                "tensordot",
            ):  # pragma: no cover
                return _to_tensor(x)  # pragma: no cover
        if type(x).__name__ == "ndarray":  # pragma: no cover
            return _to_tensor(x)  # pragma: no cover
        return x  # pragma: no cover

    def _w(x):
        if isinstance(x, tuple):  # pragma: no cover
            return tuple((_w(i) for i in x))  # pragma: no cover
        if isinstance(x, list):  # pragma: no cover
            return [_w(i) for i in x]  # pragma: no cover
        return array(x) if hasattr(x, "shape") else x  # pragma: no cover

    try:
        fn = getattr(mops, "atleast_3d")
    except AttributeError:  # pragma: no cover
        try:
            fn = getattr(mnn, "atleast_3d")
        except AttributeError:  # pragma: no cover
            try:
                fn = getattr(mrand, "atleast_3d")
            except AttributeError:  # pragma: no cover  # pragma: no cover
                raise NotImplementedError("Missing in compiler: atleast_3d")
    res = fn(
        *[_u(x) for x in args],
        **{k: _u(v, k) for (k, v) in kwargs.items() if k != "stream"},
    )
    return _w(res)


def broadcast_arrays(*args, **kwargs):
    """Compute broadcast_arrays.

    Args:
        args: The args argument.
        kwargs: The kwargs argument.

    Returns:
        The result of broadcast_arrays.

    """
    from zero_mlx.array import array, _to_tensor
    import ml_switcheroo.ops as mops
    import ml_switcheroo.nn as mnn
    import ml_switcheroo.random as mrand
    import types

    def _u(x, param_name=None):
        if isinstance(x, list):  # pragma: no cover
            if __import__("builtins").all(
                (isinstance(i, int) for i in x)
            ):  # pragma: no cover
                return tuple(x)  # pragma: no cover
            return [_u(i, param_name) for i in x]  # pragma: no cover
        if isinstance(x, tuple):  # pragma: no cover
            return tuple((_u(i, param_name) for i in x))  # pragma: no cover
        if hasattr(x, "_tensor"):  # pragma: no cover
            return x._tensor  # pragma: no cover
        if isinstance(x, (int, float, bool, complex)):  # pragma: no cover
            if param_name not in {
                "axis",
                "rtol",
                "n",
                "m",
                "endpoint",
                "offset",
                "a_max",
                "ddof",
                "dims",
                "source",
                "axis2",
                "stop",
                "k",
                "dtype",
                "reps",
                "a_min",
                "sparse",
                "decimals",
                "strides",
                "num",
                "max",
                "indices_or_sections",
                "shape",
                "atol",
                "axis1",
                "step",
                "axes",
                "fill_value",
                "stream",
                "destination",
                "alpha",
                "beta",
                "start",
                "equal_nan",
                "keepdims",
                "min",
                "p",
            } and "broadcast_arrays" not in (
                "transpose",
                "moveaxis",
                "reshape",
                "expand_dims",
                "linspace",
                "logcumsumexp",
                "cummin",
                "cummax",
                "cumprod",
                "cumsum",
                "logsumexp",
                "all",
                "any",
                "argmin",
                "argmax",
                "prod",
                "sum",
                "mean",
                "var",
                "std",
                "max",
                "min",
                "tensordot",
            ):  # pragma: no cover
                return _to_tensor(x)  # pragma: no cover
        if type(x).__name__ == "ndarray":  # pragma: no cover
            return _to_tensor(x)  # pragma: no cover
        return x  # pragma: no cover

    def _w(x):
        if isinstance(x, tuple):  # pragma: no cover
            return tuple((_w(i) for i in x))  # pragma: no cover
        if isinstance(x, list):  # pragma: no cover
            return [_w(i) for i in x]  # pragma: no cover
        return array(x) if hasattr(x, "shape") else x  # pragma: no cover

    try:
        fn = getattr(mops, "broadcast_arrays")
    except AttributeError:  # pragma: no cover
        try:
            fn = getattr(mnn, "broadcast_arrays")
        except AttributeError:  # pragma: no cover
            try:
                fn = getattr(mrand, "broadcast_arrays")
            except AttributeError:  # pragma: no cover  # pragma: no cover
                raise NotImplementedError("Missing in compiler: broadcast_arrays")
    res = fn(
        *[_u(x) for x in args],
        **{k: _u(v, k) for (k, v) in kwargs.items() if k != "stream"},
    )
    return _w(res)


def broadcast_shapes(*args, **kwargs):
    """Compute broadcast_shapes.

    Args:
        args: The args argument.
        kwargs: The kwargs argument.

    Returns:
        The result of broadcast_shapes.

    """
    from zero_mlx.array import array, _to_tensor  # pragma: no cover
    import ml_switcheroo.ops as mops  # pragma: no cover
    import ml_switcheroo.nn as mnn  # pragma: no cover
    import ml_switcheroo.random as mrand  # pragma: no cover
    import types  # pragma: no cover

    def _u(x, param_name=None):  # pragma: no cover
        if isinstance(x, list):  # pragma: no cover
            if __import__("builtins").all(
                (isinstance(i, int) for i in x)
            ):  # pragma: no cover
                return tuple(x)  # pragma: no cover
            return [_u(i, param_name) for i in x]  # pragma: no cover
        if isinstance(x, tuple):  # pragma: no cover
            return tuple((_u(i, param_name) for i in x))  # pragma: no cover
        if hasattr(x, "_tensor"):  # pragma: no cover
            return x._tensor  # pragma: no cover
        if isinstance(x, (int, float, bool, complex)):  # pragma: no cover
            if param_name not in {
                "axis",
                "rtol",
                "n",
                "m",
                "endpoint",
                "offset",
                "a_max",
                "ddof",
                "dims",
                "source",
                "axis2",
                "stop",
                "k",
                "dtype",
                "reps",
                "a_min",
                "sparse",
                "decimals",
                "strides",
                "num",
                "max",
                "indices_or_sections",
                "shape",
                "atol",
                "axis1",
                "step",
                "axes",
                "fill_value",
                "stream",
                "destination",
                "alpha",
                "beta",
                "start",
                "equal_nan",
                "keepdims",
                "min",
                "p",
            } and "broadcast_shapes" not in (
                "transpose",
                "moveaxis",
                "reshape",
                "expand_dims",
                "linspace",
                "logcumsumexp",
                "cummin",
                "cummax",
                "cumprod",
                "cumsum",
                "logsumexp",
                "all",
                "any",
                "argmin",
                "argmax",
                "prod",
                "sum",
                "mean",
                "var",
                "std",
                "max",
                "min",
                "tensordot",
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
        fn = getattr(mops, "broadcast_shapes")  # pragma: no cover
    except AttributeError:  # pragma: no cover
        try:
            fn = getattr(mnn, "broadcast_shapes")
        except AttributeError:  # pragma: no cover
            try:
                fn = getattr(mrand, "broadcast_shapes")
            except AttributeError:  # pragma: no cover  # pragma: no cover
                raise NotImplementedError("Missing in compiler: broadcast_shapes")
    res = fn(
        *[_u(x) for x in args],
        **{k: _u(v, k) for (k, v) in kwargs.items() if k != "stream"},
    )  # pragma: no cover
    return _w(res)  # pragma: no cover


def ceil(*args, **kwargs):
    """Compute ceil.

    Args:
        args: The args argument.
        kwargs: The kwargs argument.

    Returns:
        The result of ceil.

    """
    from zero_mlx.array import array, _to_tensor  # pragma: no cover
    import ml_switcheroo.ops as mops  # pragma: no cover
    import ml_switcheroo.nn as mnn  # pragma: no cover
    import ml_switcheroo.random as mrand  # pragma: no cover
    import types  # pragma: no cover

    def _u(x, param_name=None):  # pragma: no cover
        if isinstance(x, list):  # pragma: no cover
            if __import__("builtins").all(
                (isinstance(i, int) for i in x)
            ):  # pragma: no cover
                return tuple(x)  # pragma: no cover
            return [_u(i, param_name) for i in x]  # pragma: no cover
        if isinstance(x, tuple):  # pragma: no cover
            return tuple((_u(i, param_name) for i in x))  # pragma: no cover
        if hasattr(x, "_tensor"):  # pragma: no cover
            return x._tensor  # pragma: no cover
        if isinstance(x, (int, float, bool, complex)):  # pragma: no cover
            if param_name not in {
                "axis",
                "rtol",
                "n",
                "m",
                "endpoint",
                "offset",
                "a_max",
                "ddof",
                "dims",
                "source",
                "axis2",
                "stop",
                "k",
                "dtype",
                "reps",
                "a_min",
                "sparse",
                "decimals",
                "strides",
                "num",
                "max",
                "indices_or_sections",
                "shape",
                "atol",
                "axis1",
                "step",
                "axes",
                "fill_value",
                "stream",
                "destination",
                "alpha",
                "beta",
                "start",
                "equal_nan",
                "keepdims",
                "min",
                "p",
            } and "ceil" not in (
                "transpose",
                "moveaxis",
                "reshape",
                "expand_dims",
                "linspace",
                "logcumsumexp",
                "cummin",
                "cummax",
                "cumprod",
                "cumsum",
                "logsumexp",
                "all",
                "any",
                "argmin",
                "argmax",
                "prod",
                "sum",
                "mean",
                "var",
                "std",
                "max",
                "min",
                "tensordot",
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
        fn = getattr(mops, "ceil")  # pragma: no cover
    except AttributeError:  # pragma: no cover
        try:
            fn = getattr(mnn, "ceil")
        except AttributeError:  # pragma: no cover
            try:
                fn = getattr(mrand, "ceil")
            except AttributeError:  # pragma: no cover  # pragma: no cover
                raise NotImplementedError("Missing in compiler: ceil")
    res = fn(
        *[_u(x) for x in args],
        **{k: _u(v, k) for (k, v) in kwargs.items() if k != "stream"},
    )  # pragma: no cover
    return _w(res)  # pragma: no cover


def clip(*args, **kwargs):
    """Compute clip.

    Args:
        args: The args argument.
        kwargs: The kwargs argument.

    Returns:
        The result of clip.

    """
    from zero_mlx.array import array, _to_tensor
    import ml_switcheroo.ops as mops
    import ml_switcheroo.nn as mnn
    import ml_switcheroo.random as mrand
    import types

    def _u(x, param_name=None):
        if isinstance(x, list):  # pragma: no cover
            if __import__("builtins").all(
                (isinstance(i, int) for i in x)
            ):  # pragma: no cover
                return tuple(x)  # pragma: no cover
            return [_u(i, param_name) for i in x]  # pragma: no cover
        if isinstance(x, tuple):  # pragma: no cover
            return tuple((_u(i, param_name) for i in x))  # pragma: no cover
        if hasattr(x, "_tensor"):  # pragma: no cover
            return x._tensor  # pragma: no cover
        if isinstance(x, (int, float, bool, complex)):  # pragma: no cover
            if param_name not in {
                "axis",
                "rtol",
                "n",
                "m",
                "endpoint",
                "offset",
                "a_max",
                "ddof",
                "dims",
                "source",
                "axis2",
                "stop",
                "k",
                "dtype",
                "reps",
                "a_min",
                "sparse",
                "decimals",
                "strides",
                "num",
                "max",
                "indices_or_sections",
                "shape",
                "atol",
                "axis1",
                "step",
                "axes",
                "fill_value",
                "stream",
                "destination",
                "alpha",
                "beta",
                "start",
                "equal_nan",
                "keepdims",
                "min",
                "p",
            } and "clip" not in (
                "transpose",
                "moveaxis",
                "reshape",
                "expand_dims",
                "linspace",
                "logcumsumexp",
                "cummin",
                "cummax",
                "cumprod",
                "cumsum",
                "logsumexp",
                "all",
                "any",
                "argmin",
                "argmax",
                "prod",
                "sum",
                "mean",
                "var",
                "std",
                "max",
                "min",
                "tensordot",
            ):  # pragma: no cover
                return _to_tensor(x)  # pragma: no cover
        if type(x).__name__ == "ndarray":  # pragma: no cover
            return _to_tensor(x)  # pragma: no cover
        return x  # pragma: no cover

    def _w(x):
        if isinstance(x, tuple):  # pragma: no cover
            return tuple((_w(i) for i in x))  # pragma: no cover
        if isinstance(x, list):  # pragma: no cover
            return [_w(i) for i in x]  # pragma: no cover
        return array(x) if hasattr(x, "shape") else x  # pragma: no cover

    try:
        fn = getattr(mops, "clip")
    except AttributeError:  # pragma: no cover
        try:
            fn = getattr(mnn, "clip")
        except AttributeError:  # pragma: no cover
            try:
                fn = getattr(mrand, "clip")
            except AttributeError:  # pragma: no cover  # pragma: no cover
                raise NotImplementedError("Missing in compiler: clip")
    res = fn(
        *[_u(x) for x in args],
        **{k: _u(v, k) for (k, v) in kwargs.items() if k != "stream"},
    )
    return _w(res)


def conjugate(*args, **kwargs):
    """Compute conjugate.

    Args:
        args: The args argument.
        kwargs: The kwargs argument.

    Returns:
        The result of conjugate.

    """
    from zero_mlx.array import array, _to_tensor  # pragma: no cover
    import ml_switcheroo.ops as mops  # pragma: no cover
    import ml_switcheroo.nn as mnn  # pragma: no cover
    import ml_switcheroo.random as mrand  # pragma: no cover
    import types  # pragma: no cover

    def _u(x, param_name=None):  # pragma: no cover
        if isinstance(x, list):  # pragma: no cover
            if __import__("builtins").all(
                (isinstance(i, int) for i in x)
            ):  # pragma: no cover
                return tuple(x)  # pragma: no cover
            return [_u(i, param_name) for i in x]  # pragma: no cover
        if isinstance(x, tuple):  # pragma: no cover
            return tuple((_u(i, param_name) for i in x))  # pragma: no cover
        if hasattr(x, "_tensor"):  # pragma: no cover
            return x._tensor  # pragma: no cover
        if isinstance(x, (int, float, bool, complex)):  # pragma: no cover
            if param_name not in {
                "axis",
                "rtol",
                "n",
                "m",
                "endpoint",
                "offset",
                "a_max",
                "ddof",
                "dims",
                "source",
                "axis2",
                "stop",
                "k",
                "dtype",
                "reps",
                "a_min",
                "sparse",
                "decimals",
                "strides",
                "num",
                "max",
                "indices_or_sections",
                "shape",
                "atol",
                "axis1",
                "step",
                "axes",
                "fill_value",
                "stream",
                "destination",
                "alpha",
                "beta",
                "start",
                "equal_nan",
                "keepdims",
                "min",
                "p",
            } and "conjugate" not in (
                "transpose",
                "moveaxis",
                "reshape",
                "expand_dims",
                "linspace",
                "logcumsumexp",
                "cummin",
                "cummax",
                "cumprod",
                "cumsum",
                "logsumexp",
                "all",
                "any",
                "argmin",
                "argmax",
                "prod",
                "sum",
                "mean",
                "var",
                "std",
                "max",
                "min",
                "tensordot",
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
        fn = getattr(mops, "conj")  # pragma: no cover
    except AttributeError:  # pragma: no cover
        try:
            fn = getattr(mnn, "conj")
        except AttributeError:  # pragma: no cover
            try:
                fn = getattr(mrand, "conj")
            except AttributeError:  # pragma: no cover  # pragma: no cover
                raise NotImplementedError("Missing in compiler: conj")
    res = fn(
        *[_u(x) for x in args],
        **{k: _u(v, k) for (k, v) in kwargs.items() if k != "stream"},
    )  # pragma: no cover
    return _w(res)  # pragma: no cover


def degrees(*args, **kwargs):
    """Compute degrees.

    Args:
        args: The args argument.
        kwargs: The kwargs argument.

    Returns:
        The result of degrees.

    """
    from zero_mlx.array import array, _to_tensor
    import ml_switcheroo.ops as mops
    import ml_switcheroo.nn as mnn
    import ml_switcheroo.random as mrand
    import types

    def _u(x, param_name=None):
        if isinstance(x, list):  # pragma: no cover
            if __import__("builtins").all(
                (isinstance(i, int) for i in x)
            ):  # pragma: no cover
                return tuple(x)  # pragma: no cover
            return [_u(i, param_name) for i in x]  # pragma: no cover
        if isinstance(x, tuple):  # pragma: no cover
            return tuple((_u(i, param_name) for i in x))  # pragma: no cover
        if hasattr(x, "_tensor"):  # pragma: no cover
            return x._tensor  # pragma: no cover
        if isinstance(x, (int, float, bool, complex)):  # pragma: no cover
            if param_name not in {
                "axis",
                "rtol",
                "n",
                "m",
                "endpoint",
                "offset",
                "a_max",
                "ddof",
                "dims",
                "source",
                "axis2",
                "stop",
                "k",
                "dtype",
                "reps",
                "a_min",
                "sparse",
                "decimals",
                "strides",
                "num",
                "max",
                "indices_or_sections",
                "shape",
                "atol",
                "axis1",
                "step",
                "axes",
                "fill_value",
                "stream",
                "destination",
                "alpha",
                "beta",
                "start",
                "equal_nan",
                "keepdims",
                "min",
                "p",
            } and "degrees" not in (
                "transpose",
                "moveaxis",
                "reshape",
                "expand_dims",
                "linspace",
                "logcumsumexp",
                "cummin",
                "cummax",
                "cumprod",
                "cumsum",
                "logsumexp",
                "all",
                "any",
                "argmin",
                "argmax",
                "prod",
                "sum",
                "mean",
                "var",
                "std",
                "max",
                "min",
                "tensordot",
            ):  # pragma: no cover
                return _to_tensor(x)  # pragma: no cover
        if type(x).__name__ == "ndarray":  # pragma: no cover
            return _to_tensor(x)  # pragma: no cover
        return x  # pragma: no cover

    def _w(x):
        if isinstance(x, tuple):  # pragma: no cover
            return tuple((_w(i) for i in x))  # pragma: no cover
        if isinstance(x, list):  # pragma: no cover
            return [_w(i) for i in x]  # pragma: no cover
        return array(x) if hasattr(x, "shape") else x  # pragma: no cover

    try:
        fn = getattr(mops, "degrees")
    except AttributeError:  # pragma: no cover
        try:
            fn = getattr(mnn, "degrees")
        except AttributeError:  # pragma: no cover
            try:
                fn = getattr(mrand, "degrees")
            except AttributeError:  # pragma: no cover  # pragma: no cover
                raise NotImplementedError("Missing in compiler: degrees")
    res = fn(
        *[_u(x) for x in args],
        **{k: _u(v, k) for (k, v) in kwargs.items() if k != "stream"},
    )
    return _w(res)


def erf(*args, **kwargs):
    """Compute erf.

    Args:
        args: The args argument.
        kwargs: The kwargs argument.

    Returns:
        The result of erf.

    """
    from zero_mlx.array import array, _to_tensor  # pragma: no cover
    import ml_switcheroo.ops as mops  # pragma: no cover
    import ml_switcheroo.nn as mnn  # pragma: no cover
    import ml_switcheroo.random as mrand  # pragma: no cover
    import types  # pragma: no cover

    def _u(x, param_name=None):  # pragma: no cover
        if isinstance(x, list):  # pragma: no cover
            if __import__("builtins").all(
                (isinstance(i, int) for i in x)
            ):  # pragma: no cover
                return tuple(x)  # pragma: no cover
            return [_u(i, param_name) for i in x]  # pragma: no cover
        if isinstance(x, tuple):  # pragma: no cover
            return tuple((_u(i, param_name) for i in x))  # pragma: no cover
        if hasattr(x, "_tensor"):  # pragma: no cover
            return x._tensor  # pragma: no cover
        if isinstance(x, (int, float, bool, complex)):  # pragma: no cover
            if param_name not in {
                "axis",
                "rtol",
                "n",
                "m",
                "endpoint",
                "offset",
                "a_max",
                "ddof",
                "dims",
                "source",
                "axis2",
                "stop",
                "k",
                "dtype",
                "reps",
                "a_min",
                "sparse",
                "decimals",
                "strides",
                "num",
                "max",
                "indices_or_sections",
                "shape",
                "atol",
                "axis1",
                "step",
                "axes",
                "fill_value",
                "stream",
                "destination",
                "alpha",
                "beta",
                "start",
                "equal_nan",
                "keepdims",
                "min",
                "p",
            } and "erf" not in (
                "transpose",
                "moveaxis",
                "reshape",
                "expand_dims",
                "linspace",
                "logcumsumexp",
                "cummin",
                "cummax",
                "cumprod",
                "cumsum",
                "logsumexp",
                "all",
                "any",
                "argmin",
                "argmax",
                "prod",
                "sum",
                "mean",
                "var",
                "std",
                "max",
                "min",
                "tensordot",
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
        fn = getattr(mops, "erf")  # pragma: no cover
    except AttributeError:  # pragma: no cover
        try:
            fn = getattr(mnn, "erf")
        except AttributeError:  # pragma: no cover
            try:
                fn = getattr(mrand, "erf")
            except AttributeError:  # pragma: no cover  # pragma: no cover
                raise NotImplementedError("Missing in compiler: erf")
    res = fn(
        *[_u(x) for x in args],
        **{k: _u(v, k) for (k, v) in kwargs.items() if k != "stream"},
    )  # pragma: no cover
    return _w(res)  # pragma: no cover


def erfinv(*args, **kwargs):
    """Compute erfinv.

    Args:
        args: The args argument.
        kwargs: The kwargs argument.

    Returns:
        The result of erfinv.

    """
    from zero_mlx.array import array, _to_tensor  # pragma: no cover
    import ml_switcheroo.ops as mops  # pragma: no cover
    import ml_switcheroo.nn as mnn  # pragma: no cover
    import ml_switcheroo.random as mrand  # pragma: no cover
    import types  # pragma: no cover

    def _u(x, param_name=None):  # pragma: no cover
        if isinstance(x, list):  # pragma: no cover
            if __import__("builtins").all(
                (isinstance(i, int) for i in x)
            ):  # pragma: no cover
                return tuple(x)  # pragma: no cover
            return [_u(i, param_name) for i in x]  # pragma: no cover
        if isinstance(x, tuple):  # pragma: no cover
            return tuple((_u(i, param_name) for i in x))  # pragma: no cover
        if hasattr(x, "_tensor"):  # pragma: no cover
            return x._tensor  # pragma: no cover
        if isinstance(x, (int, float, bool, complex)):  # pragma: no cover
            if param_name not in {
                "axis",
                "rtol",
                "n",
                "m",
                "endpoint",
                "offset",
                "a_max",
                "ddof",
                "dims",
                "source",
                "axis2",
                "stop",
                "k",
                "dtype",
                "reps",
                "a_min",
                "sparse",
                "decimals",
                "strides",
                "num",
                "max",
                "indices_or_sections",
                "shape",
                "atol",
                "axis1",
                "step",
                "axes",
                "fill_value",
                "stream",
                "destination",
                "alpha",
                "beta",
                "start",
                "equal_nan",
                "keepdims",
                "min",
                "p",
            } and "erfinv" not in (
                "transpose",
                "moveaxis",
                "reshape",
                "expand_dims",
                "linspace",
                "logcumsumexp",
                "cummin",
                "cummax",
                "cumprod",
                "cumsum",
                "logsumexp",
                "all",
                "any",
                "argmin",
                "argmax",
                "prod",
                "sum",
                "mean",
                "var",
                "std",
                "max",
                "min",
                "tensordot",
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
        fn = getattr(mops, "erfinv")  # pragma: no cover
    except AttributeError:  # pragma: no cover
        try:
            fn = getattr(mnn, "erfinv")
        except AttributeError:  # pragma: no cover
            try:
                fn = getattr(mrand, "erfinv")
            except AttributeError:  # pragma: no cover  # pragma: no cover
                raise NotImplementedError("Missing in compiler: erfinv")
    res = fn(
        *[_u(x) for x in args],
        **{k: _u(v, k) for (k, v) in kwargs.items() if k != "stream"},
    )  # pragma: no cover
    return _w(res)  # pragma: no cover


def expm1(*args, **kwargs):
    """Compute expm1.

    Args:
        args: The args argument.
        kwargs: The kwargs argument.

    Returns:
        The result of expm1.

    """
    from zero_mlx.array import array, _to_tensor  # pragma: no cover
    import ml_switcheroo.ops as mops  # pragma: no cover
    import ml_switcheroo.nn as mnn  # pragma: no cover
    import ml_switcheroo.random as mrand  # pragma: no cover
    import types  # pragma: no cover

    def _u(x, param_name=None):  # pragma: no cover
        if isinstance(x, list):  # pragma: no cover
            if __import__("builtins").all(
                (isinstance(i, int) for i in x)
            ):  # pragma: no cover
                return tuple(x)  # pragma: no cover
            return [_u(i, param_name) for i in x]  # pragma: no cover
        if isinstance(x, tuple):  # pragma: no cover
            return tuple((_u(i, param_name) for i in x))  # pragma: no cover
        if hasattr(x, "_tensor"):  # pragma: no cover
            return x._tensor  # pragma: no cover
        if isinstance(x, (int, float, bool, complex)):  # pragma: no cover
            if param_name not in {
                "axis",
                "rtol",
                "n",
                "m",
                "endpoint",
                "offset",
                "a_max",
                "ddof",
                "dims",
                "source",
                "axis2",
                "stop",
                "k",
                "dtype",
                "reps",
                "a_min",
                "sparse",
                "decimals",
                "strides",
                "num",
                "max",
                "indices_or_sections",
                "shape",
                "atol",
                "axis1",
                "step",
                "axes",
                "fill_value",
                "stream",
                "destination",
                "alpha",
                "beta",
                "start",
                "equal_nan",
                "keepdims",
                "min",
                "p",
            } and "expm1" not in (
                "transpose",
                "moveaxis",
                "reshape",
                "expand_dims",
                "linspace",
                "logcumsumexp",
                "cummin",
                "cummax",
                "cumprod",
                "cumsum",
                "logsumexp",
                "all",
                "any",
                "argmin",
                "argmax",
                "prod",
                "sum",
                "mean",
                "var",
                "std",
                "max",
                "min",
                "tensordot",
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
        fn = getattr(mops, "expm1")  # pragma: no cover
    except AttributeError:  # pragma: no cover
        try:
            fn = getattr(mnn, "expm1")
        except AttributeError:  # pragma: no cover
            try:
                fn = getattr(mrand, "expm1")
            except AttributeError:  # pragma: no cover  # pragma: no cover
                raise NotImplementedError("Missing in compiler: expm1")
    res = fn(
        *[_u(x) for x in args],
        **{k: _u(v, k) for (k, v) in kwargs.items() if k != "stream"},
    )  # pragma: no cover
    return _w(res)  # pragma: no cover


def floor(*args, **kwargs):
    """Compute floor.

    Args:
        args: The args argument.
        kwargs: The kwargs argument.

    Returns:
        The result of floor.

    """
    from zero_mlx.array import array, _to_tensor  # pragma: no cover
    import ml_switcheroo.ops as mops  # pragma: no cover
    import ml_switcheroo.nn as mnn  # pragma: no cover
    import ml_switcheroo.random as mrand  # pragma: no cover
    import types  # pragma: no cover

    def _u(x, param_name=None):  # pragma: no cover
        if isinstance(x, list):  # pragma: no cover
            if __import__("builtins").all(
                (isinstance(i, int) for i in x)
            ):  # pragma: no cover
                return tuple(x)  # pragma: no cover
            return [_u(i, param_name) for i in x]  # pragma: no cover
        if isinstance(x, tuple):  # pragma: no cover
            return tuple((_u(i, param_name) for i in x))  # pragma: no cover
        if hasattr(x, "_tensor"):  # pragma: no cover
            return x._tensor  # pragma: no cover
        if isinstance(x, (int, float, bool, complex)):  # pragma: no cover
            if param_name not in {
                "axis",
                "rtol",
                "n",
                "m",
                "endpoint",
                "offset",
                "a_max",
                "ddof",
                "dims",
                "source",
                "axis2",
                "stop",
                "k",
                "dtype",
                "reps",
                "a_min",
                "sparse",
                "decimals",
                "strides",
                "num",
                "max",
                "indices_or_sections",
                "shape",
                "atol",
                "axis1",
                "step",
                "axes",
                "fill_value",
                "stream",
                "destination",
                "alpha",
                "beta",
                "start",
                "equal_nan",
                "keepdims",
                "min",
                "p",
            } and "floor" not in (
                "transpose",
                "moveaxis",
                "reshape",
                "expand_dims",
                "linspace",
                "logcumsumexp",
                "cummin",
                "cummax",
                "cumprod",
                "cumsum",
                "logsumexp",
                "all",
                "any",
                "argmin",
                "argmax",
                "prod",
                "sum",
                "mean",
                "var",
                "std",
                "max",
                "min",
                "tensordot",
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
        fn = getattr(mops, "floor")  # pragma: no cover
    except AttributeError:  # pragma: no cover
        try:
            fn = getattr(mnn, "floor")
        except AttributeError:  # pragma: no cover
            try:
                fn = getattr(mrand, "floor")
            except AttributeError:  # pragma: no cover  # pragma: no cover
                raise NotImplementedError("Missing in compiler: floor")
    res = fn(
        *[_u(x) for x in args],
        **{k: _u(v, k) for (k, v) in kwargs.items() if k != "stream"},
    )  # pragma: no cover
    return _w(res)  # pragma: no cover


def inner(*args, **kwargs):
    """Compute inner.

    Args:
        args: The args argument.
        kwargs: The kwargs argument.

    Returns:
        The result of inner.

    """
    from zero_mlx.array import array, _to_tensor
    import ml_switcheroo.ops as mops
    import ml_switcheroo.nn as mnn
    import ml_switcheroo.random as mrand
    import types

    def _u(x, param_name=None):
        if isinstance(x, list):  # pragma: no cover
            if __import__("builtins").all(
                (isinstance(i, int) for i in x)
            ):  # pragma: no cover
                return tuple(x)  # pragma: no cover
            return [_u(i, param_name) for i in x]  # pragma: no cover
        if isinstance(x, tuple):  # pragma: no cover
            return tuple((_u(i, param_name) for i in x))  # pragma: no cover
        if hasattr(x, "_tensor"):  # pragma: no cover
            return x._tensor  # pragma: no cover
        if isinstance(x, (int, float, bool, complex)):  # pragma: no cover
            if param_name not in {
                "axis",
                "rtol",
                "n",
                "m",
                "endpoint",
                "offset",
                "a_max",
                "ddof",
                "dims",
                "source",
                "axis2",
                "stop",
                "k",
                "dtype",
                "reps",
                "a_min",
                "sparse",
                "decimals",
                "strides",
                "num",
                "max",
                "indices_or_sections",
                "shape",
                "atol",
                "axis1",
                "step",
                "axes",
                "fill_value",
                "stream",
                "destination",
                "alpha",
                "beta",
                "start",
                "equal_nan",
                "keepdims",
                "min",
                "p",
            } and "inner" not in (
                "transpose",
                "moveaxis",
                "reshape",
                "expand_dims",
                "linspace",
                "logcumsumexp",
                "cummin",
                "cummax",
                "cumprod",
                "cumsum",
                "logsumexp",
                "all",
                "any",
                "argmin",
                "argmax",
                "prod",
                "sum",
                "mean",
                "var",
                "std",
                "max",
                "min",
                "tensordot",
            ):  # pragma: no cover
                return _to_tensor(x)  # pragma: no cover
        if type(x).__name__ == "ndarray":  # pragma: no cover
            return _to_tensor(x)  # pragma: no cover
        return x  # pragma: no cover

    def _w(x):
        if isinstance(x, tuple):  # pragma: no cover
            return tuple((_w(i) for i in x))  # pragma: no cover
        if isinstance(x, list):  # pragma: no cover
            return [_w(i) for i in x]  # pragma: no cover
        return array(x) if hasattr(x, "shape") else x  # pragma: no cover

    try:
        fn = getattr(mops, "inner")
    except AttributeError:  # pragma: no cover
        try:
            fn = getattr(mnn, "inner")
        except AttributeError:  # pragma: no cover
            try:
                fn = getattr(mrand, "inner")
            except AttributeError:  # pragma: no cover  # pragma: no cover
                raise NotImplementedError("Missing in compiler: inner")
    res = fn(
        *[_u(x) for x in args],
        **{k: _u(v, k) for (k, v) in kwargs.items() if k != "stream"},
    )
    return _w(res)


def isclose(*args, **kwargs):
    """Compute isclose.

    Args:
        args: The args argument.
        kwargs: The kwargs argument.

    Returns:
        The result of isclose.

    """
    from zero_mlx.array import array, _to_tensor
    import ml_switcheroo.ops as mops
    import ml_switcheroo.nn as mnn
    import ml_switcheroo.random as mrand
    import types

    def _u(x, param_name=None):
        if isinstance(x, list):  # pragma: no cover
            if __import__("builtins").all(
                (isinstance(i, int) for i in x)
            ):  # pragma: no cover
                return tuple(x)  # pragma: no cover
            return [_u(i, param_name) for i in x]  # pragma: no cover
        if isinstance(x, tuple):  # pragma: no cover
            return tuple((_u(i, param_name) for i in x))  # pragma: no cover
        if hasattr(x, "_tensor"):  # pragma: no cover
            return x._tensor  # pragma: no cover
        if isinstance(x, (int, float, bool, complex)):  # pragma: no cover
            if param_name not in {
                "axis",
                "rtol",
                "n",
                "m",
                "endpoint",
                "offset",
                "a_max",
                "ddof",
                "dims",
                "source",
                "axis2",
                "stop",
                "k",
                "dtype",
                "reps",
                "a_min",
                "sparse",
                "decimals",
                "strides",
                "num",
                "max",
                "indices_or_sections",
                "shape",
                "atol",
                "axis1",
                "step",
                "axes",
                "fill_value",
                "stream",
                "destination",
                "alpha",
                "beta",
                "start",
                "equal_nan",
                "keepdims",
                "min",
                "p",
            } and "isclose" not in (
                "transpose",
                "moveaxis",
                "reshape",
                "expand_dims",
                "linspace",
                "logcumsumexp",
                "cummin",
                "cummax",
                "cumprod",
                "cumsum",
                "logsumexp",
                "all",
                "any",
                "argmin",
                "argmax",
                "prod",
                "sum",
                "mean",
                "var",
                "std",
                "max",
                "min",
                "tensordot",
            ):  # pragma: no cover
                return _to_tensor(x)  # pragma: no cover
        if type(x).__name__ == "ndarray":  # pragma: no cover
            return _to_tensor(x)  # pragma: no cover
        return x  # pragma: no cover

    def _w(x):
        if isinstance(x, tuple):  # pragma: no cover
            return tuple((_w(i) for i in x))  # pragma: no cover
        if isinstance(x, list):  # pragma: no cover
            return [_w(i) for i in x]  # pragma: no cover
        return array(x) if hasattr(x, "shape") else x  # pragma: no cover

    try:
        fn = getattr(mops, "isclose")
    except AttributeError:  # pragma: no cover
        try:
            fn = getattr(mnn, "isclose")
        except AttributeError:  # pragma: no cover
            try:
                fn = getattr(mrand, "isclose")
            except AttributeError:  # pragma: no cover  # pragma: no cover
                raise NotImplementedError("Missing in compiler: isclose")
    res = fn(
        *[_u(x) for x in args],
        **{k: _u(v, k) for (k, v) in kwargs.items() if k != "stream"},
    )
    return _w(res)


def isfinite(*args, **kwargs):
    """Compute isfinite.

    Args:
        args: The args argument.
        kwargs: The kwargs argument.

    Returns:
        The result of isfinite.

    """
    from zero_mlx.array import array, _to_tensor
    import ml_switcheroo.ops as mops
    import ml_switcheroo.nn as mnn
    import ml_switcheroo.random as mrand
    import types

    def _u(x, param_name=None):
        if isinstance(x, list):  # pragma: no cover
            if __import__("builtins").all(
                (isinstance(i, int) for i in x)
            ):  # pragma: no cover
                return tuple(x)  # pragma: no cover
            return [_u(i, param_name) for i in x]  # pragma: no cover
        if isinstance(x, tuple):  # pragma: no cover
            return tuple((_u(i, param_name) for i in x))  # pragma: no cover
        if hasattr(x, "_tensor"):  # pragma: no cover
            return x._tensor  # pragma: no cover
        if isinstance(x, (int, float, bool, complex)):  # pragma: no cover
            if param_name not in {
                "axis",
                "rtol",
                "n",
                "m",
                "endpoint",
                "offset",
                "a_max",
                "ddof",
                "dims",
                "source",
                "axis2",
                "stop",
                "k",
                "dtype",
                "reps",
                "a_min",
                "sparse",
                "decimals",
                "strides",
                "num",
                "max",
                "indices_or_sections",
                "shape",
                "atol",
                "axis1",
                "step",
                "axes",
                "fill_value",
                "stream",
                "destination",
                "alpha",
                "beta",
                "start",
                "equal_nan",
                "keepdims",
                "min",
                "p",
            } and "isfinite" not in (
                "transpose",
                "moveaxis",
                "reshape",
                "expand_dims",
                "linspace",
                "logcumsumexp",
                "cummin",
                "cummax",
                "cumprod",
                "cumsum",
                "logsumexp",
                "all",
                "any",
                "argmin",
                "argmax",
                "prod",
                "sum",
                "mean",
                "var",
                "std",
                "max",
                "min",
                "tensordot",
            ):  # pragma: no cover
                return _to_tensor(x)  # pragma: no cover
        if type(x).__name__ == "ndarray":  # pragma: no cover
            return _to_tensor(x)  # pragma: no cover
        return x  # pragma: no cover

    def _w(x):
        if isinstance(x, tuple):  # pragma: no cover
            return tuple((_w(i) for i in x))  # pragma: no cover
        if isinstance(x, list):  # pragma: no cover
            return [_w(i) for i in x]  # pragma: no cover
        return array(x) if hasattr(x, "shape") else x  # pragma: no cover

    try:
        fn = getattr(mops, "isfinite")
    except AttributeError:  # pragma: no cover
        try:
            fn = getattr(mnn, "isfinite")
        except AttributeError:  # pragma: no cover
            try:
                fn = getattr(mrand, "isfinite")
            except AttributeError:  # pragma: no cover  # pragma: no cover
                raise NotImplementedError("Missing in compiler: isfinite")
    res = fn(
        *[_u(x) for x in args],
        **{k: _u(v, k) for (k, v) in kwargs.items() if k != "stream"},
    )
    return _w(res)


def isinf(*args, **kwargs):
    """Compute isinf.

    Args:
        args: The args argument.
        kwargs: The kwargs argument.

    Returns:
        The result of isinf.

    """
    from zero_mlx.array import array, _to_tensor
    import ml_switcheroo.ops as mops
    import ml_switcheroo.nn as mnn
    import ml_switcheroo.random as mrand
    import types

    def _u(x, param_name=None):
        if isinstance(x, list):  # pragma: no cover
            if __import__("builtins").all(
                (isinstance(i, int) for i in x)
            ):  # pragma: no cover
                return tuple(x)  # pragma: no cover
            return [_u(i, param_name) for i in x]  # pragma: no cover
        if isinstance(x, tuple):  # pragma: no cover
            return tuple((_u(i, param_name) for i in x))  # pragma: no cover
        if hasattr(x, "_tensor"):  # pragma: no cover
            return x._tensor  # pragma: no cover
        if isinstance(x, (int, float, bool, complex)):  # pragma: no cover
            if param_name not in {
                "axis",
                "rtol",
                "n",
                "m",
                "endpoint",
                "offset",
                "a_max",
                "ddof",
                "dims",
                "source",
                "axis2",
                "stop",
                "k",
                "dtype",
                "reps",
                "a_min",
                "sparse",
                "decimals",
                "strides",
                "num",
                "max",
                "indices_or_sections",
                "shape",
                "atol",
                "axis1",
                "step",
                "axes",
                "fill_value",
                "stream",
                "destination",
                "alpha",
                "beta",
                "start",
                "equal_nan",
                "keepdims",
                "min",
                "p",
            } and "isinf" not in (
                "transpose",
                "moveaxis",
                "reshape",
                "expand_dims",
                "linspace",
                "logcumsumexp",
                "cummin",
                "cummax",
                "cumprod",
                "cumsum",
                "logsumexp",
                "all",
                "any",
                "argmin",
                "argmax",
                "prod",
                "sum",
                "mean",
                "var",
                "std",
                "max",
                "min",
                "tensordot",
            ):  # pragma: no cover
                return _to_tensor(x)  # pragma: no cover
        if type(x).__name__ == "ndarray":  # pragma: no cover
            return _to_tensor(x)  # pragma: no cover
        return x  # pragma: no cover

    def _w(x):
        if isinstance(x, tuple):  # pragma: no cover
            return tuple((_w(i) for i in x))  # pragma: no cover
        if isinstance(x, list):  # pragma: no cover
            return [_w(i) for i in x]  # pragma: no cover
        return array(x) if hasattr(x, "shape") else x  # pragma: no cover

    try:
        fn = getattr(mops, "isinf")
    except AttributeError:  # pragma: no cover
        try:
            fn = getattr(mnn, "isinf")
        except AttributeError:  # pragma: no cover
            try:
                fn = getattr(mrand, "isinf")
            except AttributeError:  # pragma: no cover  # pragma: no cover
                raise NotImplementedError("Missing in compiler: isinf")
    res = fn(
        *[_u(x) for x in args],
        **{k: _u(v, k) for (k, v) in kwargs.items() if k != "stream"},
    )
    return _w(res)


def isnan(*args, **kwargs):
    """Compute isnan.

    Args:
        args: The args argument.
        kwargs: The kwargs argument.

    Returns:
        The result of isnan.

    """
    from zero_mlx.array import array, _to_tensor
    import ml_switcheroo.ops as mops
    import ml_switcheroo.nn as mnn
    import ml_switcheroo.random as mrand
    import types

    def _u(x, param_name=None):
        if isinstance(x, list):  # pragma: no cover
            if __import__("builtins").all(
                (isinstance(i, int) for i in x)
            ):  # pragma: no cover
                return tuple(x)  # pragma: no cover
            return [_u(i, param_name) for i in x]  # pragma: no cover
        if isinstance(x, tuple):  # pragma: no cover
            return tuple((_u(i, param_name) for i in x))  # pragma: no cover
        if hasattr(x, "_tensor"):  # pragma: no cover
            return x._tensor  # pragma: no cover
        if isinstance(x, (int, float, bool, complex)):  # pragma: no cover
            if param_name not in {
                "axis",
                "rtol",
                "n",
                "m",
                "endpoint",
                "offset",
                "a_max",
                "ddof",
                "dims",
                "source",
                "axis2",
                "stop",
                "k",
                "dtype",
                "reps",
                "a_min",
                "sparse",
                "decimals",
                "strides",
                "num",
                "max",
                "indices_or_sections",
                "shape",
                "atol",
                "axis1",
                "step",
                "axes",
                "fill_value",
                "stream",
                "destination",
                "alpha",
                "beta",
                "start",
                "equal_nan",
                "keepdims",
                "min",
                "p",
            } and "isnan" not in (
                "transpose",
                "moveaxis",
                "reshape",
                "expand_dims",
                "linspace",
                "logcumsumexp",
                "cummin",
                "cummax",
                "cumprod",
                "cumsum",
                "logsumexp",
                "all",
                "any",
                "argmin",
                "argmax",
                "prod",
                "sum",
                "mean",
                "var",
                "std",
                "max",
                "min",
                "tensordot",
            ):  # pragma: no cover
                return _to_tensor(x)  # pragma: no cover
        if type(x).__name__ == "ndarray":  # pragma: no cover
            return _to_tensor(x)  # pragma: no cover
        return x  # pragma: no cover

    def _w(x):
        if isinstance(x, tuple):  # pragma: no cover
            return tuple((_w(i) for i in x))  # pragma: no cover
        if isinstance(x, list):  # pragma: no cover
            return [_w(i) for i in x]  # pragma: no cover
        return array(x) if hasattr(x, "shape") else x  # pragma: no cover

    try:
        fn = getattr(mops, "isnan")
    except AttributeError:  # pragma: no cover
        try:
            fn = getattr(mnn, "isnan")
        except AttributeError:  # pragma: no cover
            try:
                fn = getattr(mrand, "isnan")
            except AttributeError:  # pragma: no cover  # pragma: no cover
                raise NotImplementedError("Missing in compiler: isnan")
    res = fn(
        *[_u(x) for x in args],
        **{k: _u(v, k) for (k, v) in kwargs.items() if k != "stream"},
    )
    return _w(res)


def isneginf(*args, **kwargs):
    """Compute isneginf.

    Args:
        args: The args argument.
        kwargs: The kwargs argument.

    Returns:
        The result of isneginf.

    """
    from zero_mlx.array import array, _to_tensor
    import ml_switcheroo.ops as mops
    import ml_switcheroo.nn as mnn
    import ml_switcheroo.random as mrand
    import types

    def _u(x, param_name=None):
        if isinstance(x, list):  # pragma: no cover
            if __import__("builtins").all(
                (isinstance(i, int) for i in x)
            ):  # pragma: no cover
                return tuple(x)  # pragma: no cover
            return [_u(i, param_name) for i in x]  # pragma: no cover
        if isinstance(x, tuple):  # pragma: no cover
            return tuple((_u(i, param_name) for i in x))  # pragma: no cover
        if hasattr(x, "_tensor"):  # pragma: no cover
            return x._tensor  # pragma: no cover
        if isinstance(x, (int, float, bool, complex)):  # pragma: no cover
            if param_name not in {
                "axis",
                "rtol",
                "n",
                "m",
                "endpoint",
                "offset",
                "a_max",
                "ddof",
                "dims",
                "source",
                "axis2",
                "stop",
                "k",
                "dtype",
                "reps",
                "a_min",
                "sparse",
                "decimals",
                "strides",
                "num",
                "max",
                "indices_or_sections",
                "shape",
                "atol",
                "axis1",
                "step",
                "axes",
                "fill_value",
                "stream",
                "destination",
                "alpha",
                "beta",
                "start",
                "equal_nan",
                "keepdims",
                "min",
                "p",
            } and "isneginf" not in (
                "transpose",
                "moveaxis",
                "reshape",
                "expand_dims",
                "linspace",
                "logcumsumexp",
                "cummin",
                "cummax",
                "cumprod",
                "cumsum",
                "logsumexp",
                "all",
                "any",
                "argmin",
                "argmax",
                "prod",
                "sum",
                "mean",
                "var",
                "std",
                "max",
                "min",
                "tensordot",
            ):  # pragma: no cover
                return _to_tensor(x)  # pragma: no cover
        if type(x).__name__ == "ndarray":  # pragma: no cover
            return _to_tensor(x)  # pragma: no cover
        return x  # pragma: no cover

    def _w(x):
        if isinstance(x, tuple):  # pragma: no cover
            return tuple((_w(i) for i in x))  # pragma: no cover
        if isinstance(x, list):  # pragma: no cover
            return [_w(i) for i in x]  # pragma: no cover
        return array(x) if hasattr(x, "shape") else x  # pragma: no cover

    try:
        fn = getattr(mops, "isneginf")
    except AttributeError:  # pragma: no cover
        try:
            fn = getattr(mnn, "isneginf")
        except AttributeError:  # pragma: no cover
            try:
                fn = getattr(mrand, "isneginf")
            except AttributeError:  # pragma: no cover  # pragma: no cover
                raise NotImplementedError("Missing in compiler: isneginf")
    res = fn(
        *[_u(x) for x in args],
        **{k: _u(v, k) for (k, v) in kwargs.items() if k != "stream"},
    )
    return _w(res)


def isposinf(*args, **kwargs):
    """Compute isposinf.

    Args:
        args: The args argument.
        kwargs: The kwargs argument.

    Returns:
        The result of isposinf.

    """
    from zero_mlx.array import array, _to_tensor
    import ml_switcheroo.ops as mops
    import ml_switcheroo.nn as mnn
    import ml_switcheroo.random as mrand
    import types

    def _u(x, param_name=None):
        if isinstance(x, list):  # pragma: no cover
            if __import__("builtins").all(
                (isinstance(i, int) for i in x)
            ):  # pragma: no cover
                return tuple(x)  # pragma: no cover
            return [_u(i, param_name) for i in x]  # pragma: no cover
        if isinstance(x, tuple):  # pragma: no cover
            return tuple((_u(i, param_name) for i in x))  # pragma: no cover
        if hasattr(x, "_tensor"):  # pragma: no cover
            return x._tensor  # pragma: no cover
        if isinstance(x, (int, float, bool, complex)):  # pragma: no cover
            if param_name not in {
                "axis",
                "rtol",
                "n",
                "m",
                "endpoint",
                "offset",
                "a_max",
                "ddof",
                "dims",
                "source",
                "axis2",
                "stop",
                "k",
                "dtype",
                "reps",
                "a_min",
                "sparse",
                "decimals",
                "strides",
                "num",
                "max",
                "indices_or_sections",
                "shape",
                "atol",
                "axis1",
                "step",
                "axes",
                "fill_value",
                "stream",
                "destination",
                "alpha",
                "beta",
                "start",
                "equal_nan",
                "keepdims",
                "min",
                "p",
            } and "isposinf" not in (
                "transpose",
                "moveaxis",
                "reshape",
                "expand_dims",
                "linspace",
                "logcumsumexp",
                "cummin",
                "cummax",
                "cumprod",
                "cumsum",
                "logsumexp",
                "all",
                "any",
                "argmin",
                "argmax",
                "prod",
                "sum",
                "mean",
                "var",
                "std",
                "max",
                "min",
                "tensordot",
            ):  # pragma: no cover
                return _to_tensor(x)  # pragma: no cover
        if type(x).__name__ == "ndarray":  # pragma: no cover
            return _to_tensor(x)  # pragma: no cover
        return x  # pragma: no cover

    def _w(x):
        if isinstance(x, tuple):  # pragma: no cover
            return tuple((_w(i) for i in x))  # pragma: no cover
        if isinstance(x, list):  # pragma: no cover
            return [_w(i) for i in x]  # pragma: no cover
        return array(x) if hasattr(x, "shape") else x  # pragma: no cover

    try:
        fn = getattr(mops, "isposinf")
    except AttributeError:  # pragma: no cover
        try:
            fn = getattr(mnn, "isposinf")
        except AttributeError:  # pragma: no cover
            try:
                fn = getattr(mrand, "isposinf")
            except AttributeError:  # pragma: no cover  # pragma: no cover
                raise NotImplementedError("Missing in compiler: isposinf")
    res = fn(
        *[_u(x) for x in args],
        **{k: _u(v, k) for (k, v) in kwargs.items() if k != "stream"},
    )
    return _w(res)


def issubdtype(*args, **kwargs):
    """Compute issubdtype.

    Args:
        args: The args argument.
        kwargs: The kwargs argument.

    Returns:
        The result of issubdtype.

    """
    from zero_mlx.array import array, _to_tensor  # pragma: no cover
    import ml_switcheroo.ops as mops  # pragma: no cover
    import ml_switcheroo.nn as mnn  # pragma: no cover
    import ml_switcheroo.random as mrand  # pragma: no cover
    import types  # pragma: no cover

    def _u(x, param_name=None):  # pragma: no cover
        if isinstance(x, list):  # pragma: no cover
            if __import__("builtins").all(
                (isinstance(i, int) for i in x)
            ):  # pragma: no cover
                return tuple(x)  # pragma: no cover
            return [_u(i, param_name) for i in x]  # pragma: no cover
        if isinstance(x, tuple):  # pragma: no cover
            return tuple((_u(i, param_name) for i in x))  # pragma: no cover
        if hasattr(x, "_tensor"):  # pragma: no cover
            return x._tensor  # pragma: no cover
        if isinstance(x, (int, float, bool, complex)):  # pragma: no cover
            if param_name not in {
                "axis",
                "rtol",
                "n",
                "m",
                "endpoint",
                "offset",
                "a_max",
                "ddof",
                "dims",
                "source",
                "axis2",
                "stop",
                "k",
                "dtype",
                "reps",
                "a_min",
                "sparse",
                "decimals",
                "strides",
                "num",
                "max",
                "indices_or_sections",
                "shape",
                "atol",
                "axis1",
                "step",
                "axes",
                "fill_value",
                "stream",
                "destination",
                "alpha",
                "beta",
                "start",
                "equal_nan",
                "keepdims",
                "min",
                "p",
            } and "issubdtype" not in (
                "transpose",
                "moveaxis",
                "reshape",
                "expand_dims",
                "linspace",
                "logcumsumexp",
                "cummin",
                "cummax",
                "cumprod",
                "cumsum",
                "logsumexp",
                "all",
                "any",
                "argmin",
                "argmax",
                "prod",
                "sum",
                "mean",
                "var",
                "std",
                "max",
                "min",
                "tensordot",
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
        fn = getattr(mops, "issubdtype")  # pragma: no cover
    except AttributeError:  # pragma: no cover
        try:
            fn = getattr(mnn, "issubdtype")
        except AttributeError:  # pragma: no cover
            try:
                fn = getattr(mrand, "issubdtype")
            except AttributeError:  # pragma: no cover  # pragma: no cover
                raise NotImplementedError("Missing in compiler: issubdtype")
    res = fn(
        *[_u(x) for x in args],
        **{k: _u(v, k) for (k, v) in kwargs.items() if k != "stream"},
    )  # pragma: no cover
    return _w(res)  # pragma: no cover


def kron(*args, **kwargs):
    """Compute kron.

    Args:
        args: The args argument.
        kwargs: The kwargs argument.

    Returns:
        The result of kron.

    """
    from zero_mlx.array import array, _to_tensor  # pragma: no cover
    import ml_switcheroo.ops as mops  # pragma: no cover
    import ml_switcheroo.nn as mnn  # pragma: no cover
    import ml_switcheroo.random as mrand  # pragma: no cover
    import types  # pragma: no cover

    def _u(x, param_name=None):  # pragma: no cover
        if isinstance(x, list):  # pragma: no cover
            if __import__("builtins").all(
                (isinstance(i, int) for i in x)
            ):  # pragma: no cover
                return tuple(x)  # pragma: no cover
            return [_u(i, param_name) for i in x]  # pragma: no cover
        if isinstance(x, tuple):  # pragma: no cover
            return tuple((_u(i, param_name) for i in x))  # pragma: no cover
        if hasattr(x, "_tensor"):  # pragma: no cover
            return x._tensor  # pragma: no cover
        if isinstance(x, (int, float, bool, complex)):  # pragma: no cover
            if param_name not in {
                "axis",
                "rtol",
                "n",
                "m",
                "endpoint",
                "offset",
                "a_max",
                "ddof",
                "dims",
                "source",
                "axis2",
                "stop",
                "k",
                "dtype",
                "reps",
                "a_min",
                "sparse",
                "decimals",
                "strides",
                "num",
                "max",
                "indices_or_sections",
                "shape",
                "atol",
                "axis1",
                "step",
                "axes",
                "fill_value",
                "stream",
                "destination",
                "alpha",
                "beta",
                "start",
                "equal_nan",
                "keepdims",
                "min",
                "p",
            } and "kron" not in (
                "transpose",
                "moveaxis",
                "reshape",
                "expand_dims",
                "linspace",
                "logcumsumexp",
                "cummin",
                "cummax",
                "cumprod",
                "cumsum",
                "logsumexp",
                "all",
                "any",
                "argmin",
                "argmax",
                "prod",
                "sum",
                "mean",
                "var",
                "std",
                "max",
                "min",
                "tensordot",
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
        fn = getattr(mops, "kron")  # pragma: no cover
    except AttributeError:  # pragma: no cover
        try:
            fn = getattr(mnn, "kron")
        except AttributeError:  # pragma: no cover
            try:
                fn = getattr(mrand, "kron")
            except AttributeError:  # pragma: no cover  # pragma: no cover
                raise NotImplementedError("Missing in compiler: kron")
    res = fn(
        *[_u(x) for x in args],
        **{k: _u(v, k) for (k, v) in kwargs.items() if k != "stream"},
    )  # pragma: no cover
    return _w(res)  # pragma: no cover


def linspace(*args, **kwargs):
    """Compute linspace.

    Args:
        args: The args argument.
        kwargs: The kwargs argument.

    Returns:
        The result of linspace.

    """
    from zero_mlx.array import array, _to_tensor
    import ml_switcheroo.ops as mops
    import ml_switcheroo.nn as mnn
    import ml_switcheroo.random as mrand
    import types

    def _u(x, param_name=None):
        if isinstance(x, list):  # pragma: no cover
            if __import__("builtins").all(
                (isinstance(i, int) for i in x)
            ):  # pragma: no cover
                return tuple(x)  # pragma: no cover
            return [_u(i, param_name) for i in x]  # pragma: no cover
        if isinstance(x, tuple):  # pragma: no cover
            return tuple((_u(i, param_name) for i in x))  # pragma: no cover
        if hasattr(x, "_tensor"):  # pragma: no cover
            return x._tensor  # pragma: no cover
        if isinstance(x, (int, float, bool, complex)):  # pragma: no cover
            if param_name not in {
                "axis",
                "rtol",
                "n",
                "m",
                "endpoint",
                "offset",
                "a_max",
                "ddof",
                "dims",
                "source",
                "axis2",
                "stop",
                "k",
                "dtype",
                "reps",
                "a_min",
                "sparse",
                "decimals",
                "strides",
                "num",
                "max",
                "indices_or_sections",
                "shape",
                "atol",
                "axis1",
                "step",
                "axes",
                "fill_value",
                "stream",
                "destination",
                "alpha",
                "beta",
                "start",
                "equal_nan",
                "keepdims",
                "min",
                "p",
            } and "linspace" not in (
                "transpose",
                "moveaxis",
                "reshape",
                "expand_dims",
                "linspace",
                "logcumsumexp",
                "cummin",
                "cummax",
                "cumprod",
                "cumsum",
                "logsumexp",
                "all",
                "any",
                "argmin",
                "argmax",
                "prod",
                "sum",
                "mean",
                "var",
                "std",
                "max",
                "min",
                "tensordot",
            ):  # pragma: no cover
                return _to_tensor(x)  # pragma: no cover
        if type(x).__name__ == "ndarray":  # pragma: no cover
            return _to_tensor(x)  # pragma: no cover
        return x  # pragma: no cover

    def _w(x):
        if isinstance(x, tuple):  # pragma: no cover
            return tuple((_w(i) for i in x))  # pragma: no cover
        if isinstance(x, list):  # pragma: no cover
            return [_w(i) for i in x]  # pragma: no cover
        return array(x) if hasattr(x, "shape") else x  # pragma: no cover

    try:
        fn = getattr(mops, "linspace")
    except AttributeError:  # pragma: no cover
        try:
            fn = getattr(mnn, "linspace")
        except AttributeError:  # pragma: no cover
            try:
                fn = getattr(mrand, "linspace")
            except AttributeError:  # pragma: no cover  # pragma: no cover
                raise NotImplementedError("Missing in compiler: linspace")
    res = fn(
        *[_u(x) for x in args],
        **{k: _u(v, k) for (k, v) in kwargs.items() if k != "stream"},
    )
    return _w(res)


def logaddexp(*args, **kwargs):
    """Compute logaddexp.

    Args:
        args: The args argument.
        kwargs: The kwargs argument.

    Returns:
        The result of logaddexp.

    """
    from zero_mlx.array import array, _to_tensor  # pragma: no cover
    import ml_switcheroo.ops as mops  # pragma: no cover
    import ml_switcheroo.nn as mnn  # pragma: no cover
    import ml_switcheroo.random as mrand  # pragma: no cover
    import types  # pragma: no cover

    def _u(x, param_name=None):  # pragma: no cover
        if isinstance(x, list):  # pragma: no cover
            if __import__("builtins").all(
                (isinstance(i, int) for i in x)
            ):  # pragma: no cover
                return tuple(x)  # pragma: no cover
            return [_u(i, param_name) for i in x]  # pragma: no cover
        if isinstance(x, tuple):  # pragma: no cover
            return tuple((_u(i, param_name) for i in x))  # pragma: no cover
        if hasattr(x, "_tensor"):  # pragma: no cover
            return x._tensor  # pragma: no cover
        if isinstance(x, (int, float, bool, complex)):  # pragma: no cover
            if param_name not in {
                "axis",
                "rtol",
                "n",
                "m",
                "endpoint",
                "offset",
                "a_max",
                "ddof",
                "dims",
                "source",
                "axis2",
                "stop",
                "k",
                "dtype",
                "reps",
                "a_min",
                "sparse",
                "decimals",
                "strides",
                "num",
                "max",
                "indices_or_sections",
                "shape",
                "atol",
                "axis1",
                "step",
                "axes",
                "fill_value",
                "stream",
                "destination",
                "alpha",
                "beta",
                "start",
                "equal_nan",
                "keepdims",
                "min",
                "p",
            } and "logaddexp" not in (
                "transpose",
                "moveaxis",
                "reshape",
                "expand_dims",
                "linspace",
                "logcumsumexp",
                "cummin",
                "cummax",
                "cumprod",
                "cumsum",
                "logsumexp",
                "all",
                "any",
                "argmin",
                "argmax",
                "prod",
                "sum",
                "mean",
                "var",
                "std",
                "max",
                "min",
                "tensordot",
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
        fn = getattr(mops, "logaddexp")  # pragma: no cover
    except AttributeError:  # pragma: no cover
        try:
            fn = getattr(mnn, "logaddexp")
        except AttributeError:  # pragma: no cover
            try:
                fn = getattr(mrand, "logaddexp")
            except AttributeError:  # pragma: no cover  # pragma: no cover
                raise NotImplementedError("Missing in compiler: logaddexp")
    res = fn(
        *[_u(x) for x in args],
        **{k: _u(v, k) for (k, v) in kwargs.items() if k != "stream"},
    )  # pragma: no cover
    return _w(res)  # pragma: no cover


def median(*args, **kwargs):
    """Compute median.

    Args:
        args: The args argument.
        kwargs: The kwargs argument.

    Returns:
        The result of median.

    """
    from zero_mlx.array import array, _to_tensor  # pragma: no cover
    import ml_switcheroo.ops as mops  # pragma: no cover
    import ml_switcheroo.nn as mnn  # pragma: no cover
    import ml_switcheroo.random as mrand  # pragma: no cover
    import types  # pragma: no cover

    def _u(x, param_name=None):  # pragma: no cover
        if isinstance(x, list):  # pragma: no cover
            if __import__("builtins").all(
                (isinstance(i, int) for i in x)
            ):  # pragma: no cover
                return tuple(x)  # pragma: no cover
            return [_u(i, param_name) for i in x]  # pragma: no cover
        if isinstance(x, tuple):  # pragma: no cover
            return tuple((_u(i, param_name) for i in x))  # pragma: no cover
        if hasattr(x, "_tensor"):  # pragma: no cover
            return x._tensor  # pragma: no cover
        if isinstance(x, (int, float, bool, complex)):  # pragma: no cover
            if param_name not in {
                "axis",
                "rtol",
                "n",
                "m",
                "endpoint",
                "offset",
                "a_max",
                "ddof",
                "dims",
                "source",
                "axis2",
                "stop",
                "k",
                "dtype",
                "reps",
                "a_min",
                "sparse",
                "decimals",
                "strides",
                "num",
                "max",
                "indices_or_sections",
                "shape",
                "atol",
                "axis1",
                "step",
                "axes",
                "fill_value",
                "stream",
                "destination",
                "alpha",
                "beta",
                "start",
                "equal_nan",
                "keepdims",
                "min",
                "p",
            } and "median" not in (
                "transpose",
                "moveaxis",
                "reshape",
                "expand_dims",
                "linspace",
                "logcumsumexp",
                "cummin",
                "cummax",
                "cumprod",
                "cumsum",
                "logsumexp",
                "all",
                "any",
                "argmin",
                "argmax",
                "prod",
                "sum",
                "mean",
                "var",
                "std",
                "max",
                "min",
                "tensordot",
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
        fn = getattr(mops, "median")  # pragma: no cover
    except AttributeError:  # pragma: no cover
        try:
            fn = getattr(mnn, "median")
        except AttributeError:  # pragma: no cover
            try:
                fn = getattr(mrand, "median")
            except AttributeError:  # pragma: no cover  # pragma: no cover
                raise NotImplementedError("Missing in compiler: median")
    res = fn(
        *[_u(x) for x in args],
        **{k: _u(v, k) for (k, v) in kwargs.items() if k != "stream"},
    )  # pragma: no cover
    return _w(res)  # pragma: no cover


def meshgrid(*args, **kwargs):
    """Compute meshgrid.

    Args:
        args: The args argument.
        kwargs: The kwargs argument.

    Returns:
        The result of meshgrid.

    """
    from zero_mlx.array import array, _to_tensor
    import ml_switcheroo.ops as mops
    import ml_switcheroo.nn as mnn
    import ml_switcheroo.random as mrand
    import types

    def _u(x, param_name=None):
        if isinstance(x, list):  # pragma: no cover
            if __import__("builtins").all(
                (isinstance(i, int) for i in x)
            ):  # pragma: no cover
                return tuple(x)  # pragma: no cover
            return [_u(i, param_name) for i in x]  # pragma: no cover
        if isinstance(x, tuple):  # pragma: no cover
            return tuple((_u(i, param_name) for i in x))  # pragma: no cover
        if hasattr(x, "_tensor"):  # pragma: no cover
            return x._tensor  # pragma: no cover
        if isinstance(x, (int, float, bool, complex)):  # pragma: no cover
            if param_name not in {
                "axis",
                "rtol",
                "n",
                "m",
                "endpoint",
                "offset",
                "a_max",
                "ddof",
                "dims",
                "source",
                "axis2",
                "stop",
                "k",
                "dtype",
                "reps",
                "a_min",
                "sparse",
                "decimals",
                "strides",
                "num",
                "max",
                "indices_or_sections",
                "shape",
                "atol",
                "axis1",
                "step",
                "axes",
                "fill_value",
                "stream",
                "destination",
                "alpha",
                "beta",
                "start",
                "equal_nan",
                "keepdims",
                "min",
                "p",
            } and "meshgrid" not in (
                "transpose",
                "moveaxis",
                "reshape",
                "expand_dims",
                "linspace",
                "logcumsumexp",
                "cummin",
                "cummax",
                "cumprod",
                "cumsum",
                "logsumexp",
                "all",
                "any",
                "argmin",
                "argmax",
                "prod",
                "sum",
                "mean",
                "var",
                "std",
                "max",
                "min",
                "tensordot",
            ):  # pragma: no cover
                return _to_tensor(x)  # pragma: no cover
        if type(x).__name__ == "ndarray":  # pragma: no cover
            return _to_tensor(x)  # pragma: no cover
        return x  # pragma: no cover

    def _w(x):
        if isinstance(x, tuple):  # pragma: no cover
            return tuple((_w(i) for i in x))  # pragma: no cover
        if isinstance(x, list):  # pragma: no cover
            return [_w(i) for i in x]  # pragma: no cover
        return array(x) if hasattr(x, "shape") else x  # pragma: no cover

    try:
        fn = getattr(mops, "meshgrid")
    except AttributeError:  # pragma: no cover
        try:
            fn = getattr(mnn, "meshgrid")
        except AttributeError:  # pragma: no cover
            try:
                fn = getattr(mrand, "meshgrid")
            except AttributeError:  # pragma: no cover  # pragma: no cover
                raise NotImplementedError("Missing in compiler: meshgrid")
    res = fn(
        *[_u(x) for x in args],
        **{k: _u(v, k) for (k, v) in kwargs.items() if k != "stream"},
    )
    return _w(res)


def nan_to_num(*args, **kwargs):
    """Compute nan_to_num.

    Args:
        args: The args argument.
        kwargs: The kwargs argument.

    Returns:
        The result of nan_to_num.

    """
    from zero_mlx.array import array, _to_tensor  # pragma: no cover
    import ml_switcheroo.ops as mops  # pragma: no cover
    import ml_switcheroo.nn as mnn  # pragma: no cover
    import ml_switcheroo.random as mrand  # pragma: no cover
    import types  # pragma: no cover

    def _u(x, param_name=None):  # pragma: no cover
        if isinstance(x, list):  # pragma: no cover
            if __import__("builtins").all(
                (isinstance(i, int) for i in x)
            ):  # pragma: no cover
                return tuple(x)  # pragma: no cover
            return [_u(i, param_name) for i in x]  # pragma: no cover
        if isinstance(x, tuple):  # pragma: no cover
            return tuple((_u(i, param_name) for i in x))  # pragma: no cover
        if hasattr(x, "_tensor"):  # pragma: no cover
            return x._tensor  # pragma: no cover
        if isinstance(x, (int, float, bool, complex)):  # pragma: no cover
            if param_name not in {
                "axis",
                "rtol",
                "n",
                "m",
                "endpoint",
                "offset",
                "a_max",
                "ddof",
                "dims",
                "source",
                "axis2",
                "stop",
                "k",
                "dtype",
                "reps",
                "a_min",
                "sparse",
                "decimals",
                "strides",
                "num",
                "max",
                "indices_or_sections",
                "shape",
                "atol",
                "axis1",
                "step",
                "axes",
                "fill_value",
                "stream",
                "destination",
                "alpha",
                "beta",
                "start",
                "equal_nan",
                "keepdims",
                "min",
                "p",
            } and "nan_to_num" not in (
                "transpose",
                "moveaxis",
                "reshape",
                "expand_dims",
                "linspace",
                "logcumsumexp",
                "cummin",
                "cummax",
                "cumprod",
                "cumsum",
                "logsumexp",
                "all",
                "any",
                "argmin",
                "argmax",
                "prod",
                "sum",
                "mean",
                "var",
                "std",
                "max",
                "min",
                "tensordot",
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
        fn = getattr(mops, "nan_to_num")  # pragma: no cover
    except AttributeError:  # pragma: no cover
        try:
            fn = getattr(mnn, "nan_to_num")
        except AttributeError:  # pragma: no cover
            try:
                fn = getattr(mrand, "nan_to_num")
            except AttributeError:  # pragma: no cover  # pragma: no cover
                raise NotImplementedError("Missing in compiler: nan_to_num")
    res = fn(
        *[_u(x) for x in args],
        **{k: _u(v, k) for (k, v) in kwargs.items() if k != "stream"},
    )  # pragma: no cover
    return _w(res)  # pragma: no cover


def outer(*args, **kwargs):
    """Compute outer.

    Args:
        args: The args argument.
        kwargs: The kwargs argument.

    Returns:
        The result of outer.

    """
    from zero_mlx.array import array, _to_tensor
    import ml_switcheroo.ops as mops
    import ml_switcheroo.nn as mnn
    import ml_switcheroo.random as mrand
    import types

    def _u(x, param_name=None):
        if isinstance(x, list):  # pragma: no cover
            if __import__("builtins").all(
                (isinstance(i, int) for i in x)
            ):  # pragma: no cover
                return tuple(x)  # pragma: no cover
            return [_u(i, param_name) for i in x]  # pragma: no cover
        if isinstance(x, tuple):  # pragma: no cover
            return tuple((_u(i, param_name) for i in x))  # pragma: no cover
        if hasattr(x, "_tensor"):  # pragma: no cover
            return x._tensor  # pragma: no cover
        if isinstance(x, (int, float, bool, complex)):  # pragma: no cover
            if param_name not in {
                "axis",
                "rtol",
                "n",
                "m",
                "endpoint",
                "offset",
                "a_max",
                "ddof",
                "dims",
                "source",
                "axis2",
                "stop",
                "k",
                "dtype",
                "reps",
                "a_min",
                "sparse",
                "decimals",
                "strides",
                "num",
                "max",
                "indices_or_sections",
                "shape",
                "atol",
                "axis1",
                "step",
                "axes",
                "fill_value",
                "stream",
                "destination",
                "alpha",
                "beta",
                "start",
                "equal_nan",
                "keepdims",
                "min",
                "p",
            } and "outer" not in (
                "transpose",
                "moveaxis",
                "reshape",
                "expand_dims",
                "linspace",
                "logcumsumexp",
                "cummin",
                "cummax",
                "cumprod",
                "cumsum",
                "logsumexp",
                "all",
                "any",
                "argmin",
                "argmax",
                "prod",
                "sum",
                "mean",
                "var",
                "std",
                "max",
                "min",
                "tensordot",
            ):  # pragma: no cover
                return _to_tensor(x)  # pragma: no cover
        if type(x).__name__ == "ndarray":  # pragma: no cover
            return _to_tensor(x)  # pragma: no cover
        return x  # pragma: no cover

    def _w(x):
        if isinstance(x, tuple):  # pragma: no cover
            return tuple((_w(i) for i in x))  # pragma: no cover
        if isinstance(x, list):  # pragma: no cover
            return [_w(i) for i in x]  # pragma: no cover
        return array(x) if hasattr(x, "shape") else x  # pragma: no cover

    try:
        fn = getattr(mops, "outer")
    except AttributeError:  # pragma: no cover
        try:
            fn = getattr(mnn, "outer")
        except AttributeError:  # pragma: no cover
            try:
                fn = getattr(mrand, "outer")
            except AttributeError:  # pragma: no cover  # pragma: no cover
                raise NotImplementedError("Missing in compiler: outer")
    res = fn(
        *[_u(x) for x in args],
        **{k: _u(v, k) for (k, v) in kwargs.items() if k != "stream"},
    )
    return _w(res)


def pad(*args, **kwargs):
    """Compute pad.

    Args:
        args: The args argument.
        kwargs: The kwargs argument.

    Returns:
        The result of pad.

    """
    from zero_mlx.array import array, _to_tensor  # pragma: no cover
    import ml_switcheroo.ops as mops  # pragma: no cover
    import ml_switcheroo.nn as mnn  # pragma: no cover
    import ml_switcheroo.random as mrand  # pragma: no cover
    import types  # pragma: no cover

    def _u(x, param_name=None):  # pragma: no cover
        if isinstance(x, list):  # pragma: no cover
            if __import__("builtins").all(
                (isinstance(i, int) for i in x)
            ):  # pragma: no cover
                return tuple(x)  # pragma: no cover
            return [_u(i, param_name) for i in x]  # pragma: no cover
        if isinstance(x, tuple):  # pragma: no cover
            return tuple((_u(i, param_name) for i in x))  # pragma: no cover
        if hasattr(x, "_tensor"):  # pragma: no cover
            return x._tensor  # pragma: no cover
        if isinstance(x, (int, float, bool, complex)):  # pragma: no cover
            if param_name not in {
                "axis",
                "rtol",
                "n",
                "m",
                "endpoint",
                "offset",
                "a_max",
                "ddof",
                "dims",
                "source",
                "axis2",
                "stop",
                "k",
                "dtype",
                "reps",
                "a_min",
                "sparse",
                "decimals",
                "strides",
                "num",
                "max",
                "indices_or_sections",
                "shape",
                "atol",
                "axis1",
                "step",
                "axes",
                "fill_value",
                "stream",
                "destination",
                "alpha",
                "beta",
                "start",
                "equal_nan",
                "keepdims",
                "min",
                "p",
            } and "pad" not in (
                "transpose",
                "moveaxis",
                "reshape",
                "expand_dims",
                "linspace",
                "logcumsumexp",
                "cummin",
                "cummax",
                "cumprod",
                "cumsum",
                "logsumexp",
                "all",
                "any",
                "argmin",
                "argmax",
                "prod",
                "sum",
                "mean",
                "var",
                "std",
                "max",
                "min",
                "tensordot",
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
        fn = getattr(mops, "pad")  # pragma: no cover
    except AttributeError:  # pragma: no cover
        try:
            fn = getattr(mnn, "pad")
        except AttributeError:  # pragma: no cover
            try:
                fn = getattr(mrand, "pad")
            except AttributeError:  # pragma: no cover  # pragma: no cover
                raise NotImplementedError("Missing in compiler: pad")
    res = fn(
        *[_u(x) for x in args],
        **{k: _u(v, k) for (k, v) in kwargs.items() if k != "stream"},
    )  # pragma: no cover
    return _w(res)  # pragma: no cover


def partition(*args, **kwargs):
    """Compute partition.

    Args:
        args: The args argument.
        kwargs: The kwargs argument.

    Returns:
        The result of partition.

    """
    from zero_mlx.array import array, _to_tensor  # pragma: no cover
    import ml_switcheroo.ops as mops  # pragma: no cover
    import ml_switcheroo.nn as mnn  # pragma: no cover
    import ml_switcheroo.random as mrand  # pragma: no cover
    import types  # pragma: no cover

    def _u(x, param_name=None):  # pragma: no cover
        if isinstance(x, list):  # pragma: no cover
            if __import__("builtins").all(
                (isinstance(i, int) for i in x)
            ):  # pragma: no cover
                return tuple(x)  # pragma: no cover
            return [_u(i, param_name) for i in x]  # pragma: no cover
        if isinstance(x, tuple):  # pragma: no cover
            return tuple((_u(i, param_name) for i in x))  # pragma: no cover
        if hasattr(x, "_tensor"):  # pragma: no cover
            return x._tensor  # pragma: no cover
        if isinstance(x, (int, float, bool, complex)):  # pragma: no cover
            if param_name not in {
                "axis",
                "rtol",
                "n",
                "m",
                "endpoint",
                "offset",
                "a_max",
                "ddof",
                "dims",
                "source",
                "axis2",
                "stop",
                "k",
                "dtype",
                "reps",
                "a_min",
                "sparse",
                "decimals",
                "strides",
                "num",
                "max",
                "indices_or_sections",
                "shape",
                "atol",
                "axis1",
                "step",
                "axes",
                "fill_value",
                "stream",
                "destination",
                "alpha",
                "beta",
                "start",
                "equal_nan",
                "keepdims",
                "min",
                "p",
            } and "partition" not in (
                "transpose",
                "moveaxis",
                "reshape",
                "expand_dims",
                "linspace",
                "logcumsumexp",
                "cummin",
                "cummax",
                "cumprod",
                "cumsum",
                "logsumexp",
                "all",
                "any",
                "argmin",
                "argmax",
                "prod",
                "sum",
                "mean",
                "var",
                "std",
                "max",
                "min",
                "tensordot",
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
        fn = getattr(mops, "partition")  # pragma: no cover
    except AttributeError:  # pragma: no cover
        try:
            fn = getattr(mnn, "partition")
        except AttributeError:  # pragma: no cover
            try:
                fn = getattr(mrand, "partition")
            except AttributeError:  # pragma: no cover  # pragma: no cover
                raise NotImplementedError("Missing in compiler: partition")
    res = fn(
        *[_u(x) for x in args],
        **{k: _u(v, k) for (k, v) in kwargs.items() if k != "stream"},
    )  # pragma: no cover
    return _w(res)  # pragma: no cover


def put_along_axis(*args, **kwargs):
    """Compute put_along_axis.

    Args:
        args: The args argument.
        kwargs: The kwargs argument.

    Returns:
        The result of put_along_axis.

    """
    from zero_mlx.array import array, _to_tensor  # pragma: no cover
    import ml_switcheroo.ops as mops  # pragma: no cover
    import ml_switcheroo.nn as mnn  # pragma: no cover
    import ml_switcheroo.random as mrand  # pragma: no cover
    import types  # pragma: no cover

    def _u(x, param_name=None):  # pragma: no cover
        if isinstance(x, list):  # pragma: no cover
            if __import__("builtins").all(
                (isinstance(i, int) for i in x)
            ):  # pragma: no cover
                return tuple(x)  # pragma: no cover
            return [_u(i, param_name) for i in x]  # pragma: no cover
        if isinstance(x, tuple):  # pragma: no cover
            return tuple((_u(i, param_name) for i in x))  # pragma: no cover
        if hasattr(x, "_tensor"):  # pragma: no cover
            return x._tensor  # pragma: no cover
        if isinstance(x, (int, float, bool, complex)):  # pragma: no cover
            if param_name not in {
                "axis",
                "rtol",
                "n",
                "m",
                "endpoint",
                "offset",
                "a_max",
                "ddof",
                "dims",
                "source",
                "axis2",
                "stop",
                "k",
                "dtype",
                "reps",
                "a_min",
                "sparse",
                "decimals",
                "strides",
                "num",
                "max",
                "indices_or_sections",
                "shape",
                "atol",
                "axis1",
                "step",
                "axes",
                "fill_value",
                "stream",
                "destination",
                "alpha",
                "beta",
                "start",
                "equal_nan",
                "keepdims",
                "min",
                "p",
            } and "put_along_axis" not in (
                "transpose",
                "moveaxis",
                "reshape",
                "expand_dims",
                "linspace",
                "logcumsumexp",
                "cummin",
                "cummax",
                "cumprod",
                "cumsum",
                "logsumexp",
                "all",
                "any",
                "argmin",
                "argmax",
                "prod",
                "sum",
                "mean",
                "var",
                "std",
                "max",
                "min",
                "tensordot",
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
        fn = getattr(mops, "put_along_axis")  # pragma: no cover
    except AttributeError:  # pragma: no cover
        try:
            fn = getattr(mnn, "put_along_axis")
        except AttributeError:  # pragma: no cover
            try:
                fn = getattr(mrand, "put_along_axis")
            except AttributeError:  # pragma: no cover  # pragma: no cover
                raise NotImplementedError("Missing in compiler: put_along_axis")
    res = fn(
        *[_u(x) for x in args],
        **{k: _u(v, k) for (k, v) in kwargs.items() if k != "stream"},
    )  # pragma: no cover
    return _w(res)  # pragma: no cover


def radians(*args, **kwargs):
    """Compute radians.

    Args:
        args: The args argument.
        kwargs: The kwargs argument.

    Returns:
        The result of radians.

    """
    from zero_mlx.array import array, _to_tensor
    import ml_switcheroo.ops as mops
    import ml_switcheroo.nn as mnn
    import ml_switcheroo.random as mrand
    import types

    def _u(x, param_name=None):
        if isinstance(x, list):  # pragma: no cover
            if __import__("builtins").all(
                (isinstance(i, int) for i in x)
            ):  # pragma: no cover
                return tuple(x)  # pragma: no cover
            return [_u(i, param_name) for i in x]  # pragma: no cover
        if isinstance(x, tuple):  # pragma: no cover
            return tuple((_u(i, param_name) for i in x))  # pragma: no cover
        if hasattr(x, "_tensor"):  # pragma: no cover
            return x._tensor  # pragma: no cover
        if isinstance(x, (int, float, bool, complex)):  # pragma: no cover
            if param_name not in {
                "axis",
                "rtol",
                "n",
                "m",
                "endpoint",
                "offset",
                "a_max",
                "ddof",
                "dims",
                "source",
                "axis2",
                "stop",
                "k",
                "dtype",
                "reps",
                "a_min",
                "sparse",
                "decimals",
                "strides",
                "num",
                "max",
                "indices_or_sections",
                "shape",
                "atol",
                "axis1",
                "step",
                "axes",
                "fill_value",
                "stream",
                "destination",
                "alpha",
                "beta",
                "start",
                "equal_nan",
                "keepdims",
                "min",
                "p",
            } and "radians" not in (
                "transpose",
                "moveaxis",
                "reshape",
                "expand_dims",
                "linspace",
                "logcumsumexp",
                "cummin",
                "cummax",
                "cumprod",
                "cumsum",
                "logsumexp",
                "all",
                "any",
                "argmin",
                "argmax",
                "prod",
                "sum",
                "mean",
                "var",
                "std",
                "max",
                "min",
                "tensordot",
            ):  # pragma: no cover
                return _to_tensor(x)  # pragma: no cover
        if type(x).__name__ == "ndarray":  # pragma: no cover
            return _to_tensor(x)  # pragma: no cover
        return x  # pragma: no cover

    def _w(x):
        if isinstance(x, tuple):  # pragma: no cover
            return tuple((_w(i) for i in x))  # pragma: no cover
        if isinstance(x, list):  # pragma: no cover
            return [_w(i) for i in x]  # pragma: no cover
        return array(x) if hasattr(x, "shape") else x  # pragma: no cover

    try:
        fn = getattr(mops, "radians")
    except AttributeError:  # pragma: no cover
        try:
            fn = getattr(mnn, "radians")
        except AttributeError:  # pragma: no cover
            try:
                fn = getattr(mrand, "radians")
            except AttributeError:  # pragma: no cover  # pragma: no cover
                raise NotImplementedError("Missing in compiler: radians")
    res = fn(
        *[_u(x) for x in args],
        **{k: _u(v, k) for (k, v) in kwargs.items() if k != "stream"},
    )
    return _w(res)


def real(*args, **kwargs):
    """Compute real.

    Args:
        args: The args argument.
        kwargs: The kwargs argument.

    Returns:
        The result of real.

    """
    from zero_mlx.array import array, _to_tensor  # pragma: no cover
    import ml_switcheroo.ops as mops  # pragma: no cover
    import ml_switcheroo.nn as mnn  # pragma: no cover
    import ml_switcheroo.random as mrand  # pragma: no cover
    import types  # pragma: no cover

    def _u(x, param_name=None):  # pragma: no cover
        if isinstance(x, list):  # pragma: no cover
            if __import__("builtins").all(
                (isinstance(i, int) for i in x)
            ):  # pragma: no cover
                return tuple(x)  # pragma: no cover
            return [_u(i, param_name) for i in x]  # pragma: no cover
        if isinstance(x, tuple):  # pragma: no cover
            return tuple((_u(i, param_name) for i in x))  # pragma: no cover
        if hasattr(x, "_tensor"):  # pragma: no cover
            return x._tensor  # pragma: no cover
        if isinstance(x, (int, float, bool, complex)):  # pragma: no cover
            if param_name not in {
                "axis",
                "rtol",
                "n",
                "m",
                "endpoint",
                "offset",
                "a_max",
                "ddof",
                "dims",
                "source",
                "axis2",
                "stop",
                "k",
                "dtype",
                "reps",
                "a_min",
                "sparse",
                "decimals",
                "strides",
                "num",
                "max",
                "indices_or_sections",
                "shape",
                "atol",
                "axis1",
                "step",
                "axes",
                "fill_value",
                "stream",
                "destination",
                "alpha",
                "beta",
                "start",
                "equal_nan",
                "keepdims",
                "min",
                "p",
            } and "real" not in (
                "transpose",
                "moveaxis",
                "reshape",
                "expand_dims",
                "linspace",
                "logcumsumexp",
                "cummin",
                "cummax",
                "cumprod",
                "cumsum",
                "logsumexp",
                "all",
                "any",
                "argmin",
                "argmax",
                "prod",
                "sum",
                "mean",
                "var",
                "std",
                "max",
                "min",
                "tensordot",
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
        fn = getattr(mops, "real")  # pragma: no cover
    except AttributeError:  # pragma: no cover
        try:
            fn = getattr(mnn, "real")
        except AttributeError:  # pragma: no cover
            try:
                fn = getattr(mrand, "real")
            except AttributeError:  # pragma: no cover  # pragma: no cover
                raise NotImplementedError("Missing in compiler: real")
    res = fn(
        *[_u(x) for x in args],
        **{k: _u(v, k) for (k, v) in kwargs.items() if k != "stream"},
    )  # pragma: no cover
    return _w(res)  # pragma: no cover


def remainder(*args, **kwargs):
    """Compute remainder.

    Args:
        args: The args argument.
        kwargs: The kwargs argument.

    Returns:
        The result of remainder.

    """
    from zero_mlx.array import array, _to_tensor
    import ml_switcheroo.ops as mops
    import ml_switcheroo.nn as mnn
    import ml_switcheroo.random as mrand
    import types

    def _u(x, param_name=None):
        if isinstance(x, list):  # pragma: no cover
            if __import__("builtins").all(
                (isinstance(i, int) for i in x)
            ):  # pragma: no cover
                return tuple(x)  # pragma: no cover
            return [_u(i, param_name) for i in x]  # pragma: no cover
        if isinstance(x, tuple):  # pragma: no cover
            return tuple((_u(i, param_name) for i in x))  # pragma: no cover
        if hasattr(x, "_tensor"):  # pragma: no cover
            return x._tensor  # pragma: no cover
        if isinstance(x, (int, float, bool, complex)):  # pragma: no cover
            if param_name not in {
                "axis",
                "rtol",
                "n",
                "m",
                "endpoint",
                "offset",
                "a_max",
                "ddof",
                "dims",
                "source",
                "axis2",
                "stop",
                "k",
                "dtype",
                "reps",
                "a_min",
                "sparse",
                "decimals",
                "strides",
                "num",
                "max",
                "indices_or_sections",
                "shape",
                "atol",
                "axis1",
                "step",
                "axes",
                "fill_value",
                "stream",
                "destination",
                "alpha",
                "beta",
                "start",
                "equal_nan",
                "keepdims",
                "min",
                "p",
            } and "remainder" not in (
                "transpose",
                "moveaxis",
                "reshape",
                "expand_dims",
                "linspace",
                "logcumsumexp",
                "cummin",
                "cummax",
                "cumprod",
                "cumsum",
                "logsumexp",
                "all",
                "any",
                "argmin",
                "argmax",
                "prod",
                "sum",
                "mean",
                "var",
                "std",
                "max",
                "min",
                "tensordot",
            ):  # pragma: no cover
                return _to_tensor(x)  # pragma: no cover
        if type(x).__name__ == "ndarray":  # pragma: no cover
            return _to_tensor(x)  # pragma: no cover
        return x  # pragma: no cover

    def _w(x):
        if isinstance(x, tuple):  # pragma: no cover
            return tuple((_w(i) for i in x))  # pragma: no cover
        if isinstance(x, list):  # pragma: no cover
            return [_w(i) for i in x]  # pragma: no cover
        return array(x) if hasattr(x, "shape") else x  # pragma: no cover

    try:
        fn = getattr(mops, "remainder")
    except AttributeError:  # pragma: no cover
        try:
            fn = getattr(mnn, "remainder")
        except AttributeError:  # pragma: no cover
            try:
                fn = getattr(mrand, "remainder")
            except AttributeError:  # pragma: no cover  # pragma: no cover
                raise NotImplementedError("Missing in compiler: remainder")
    res = fn(
        *[_u(x) for x in args],
        **{k: _u(v, k) for (k, v) in kwargs.items() if k != "stream"},
    )
    return _w(res)


def repeat(*args, **kwargs):
    """Compute repeat.

    Args:
        args: The args argument.
        kwargs: The kwargs argument.

    Returns:
        The result of repeat.

    """
    from zero_mlx.array import array, _to_tensor
    import ml_switcheroo.ops as mops
    import ml_switcheroo.nn as mnn
    import ml_switcheroo.random as mrand
    import types

    def _u(x, param_name=None):
        if isinstance(x, list):  # pragma: no cover
            if __import__("builtins").all(
                (isinstance(i, int) for i in x)
            ):  # pragma: no cover
                return tuple(x)  # pragma: no cover
            return [_u(i, param_name) for i in x]  # pragma: no cover
        if isinstance(x, tuple):  # pragma: no cover
            return tuple((_u(i, param_name) for i in x))  # pragma: no cover
        if hasattr(x, "_tensor"):  # pragma: no cover
            return x._tensor  # pragma: no cover
        if isinstance(x, (int, float, bool, complex)):  # pragma: no cover
            if param_name not in {
                "axis",
                "rtol",
                "n",
                "m",
                "endpoint",
                "offset",
                "a_max",
                "ddof",
                "dims",
                "source",
                "axis2",
                "stop",
                "k",
                "dtype",
                "reps",
                "a_min",
                "sparse",
                "decimals",
                "strides",
                "num",
                "max",
                "indices_or_sections",
                "shape",
                "atol",
                "axis1",
                "step",
                "axes",
                "fill_value",
                "stream",
                "destination",
                "alpha",
                "beta",
                "start",
                "equal_nan",
                "keepdims",
                "min",
                "p",
            } and "repeat" not in (
                "transpose",
                "moveaxis",
                "reshape",
                "expand_dims",
                "linspace",
                "logcumsumexp",
                "cummin",
                "cummax",
                "cumprod",
                "cumsum",
                "logsumexp",
                "all",
                "any",
                "argmin",
                "argmax",
                "prod",
                "sum",
                "mean",
                "var",
                "std",
                "max",
                "min",
                "tensordot",
            ):  # pragma: no cover
                return _to_tensor(x)  # pragma: no cover
        if type(x).__name__ == "ndarray":  # pragma: no cover
            return _to_tensor(x)  # pragma: no cover
        return x  # pragma: no cover

    def _w(x):
        if isinstance(x, tuple):  # pragma: no cover
            return tuple((_w(i) for i in x))  # pragma: no cover
        if isinstance(x, list):  # pragma: no cover
            return [_w(i) for i in x]  # pragma: no cover
        return array(x) if hasattr(x, "shape") else x  # pragma: no cover

    try:
        fn = getattr(mops, "repeat")
    except AttributeError:  # pragma: no cover
        try:
            fn = getattr(mnn, "repeat")
        except AttributeError:  # pragma: no cover
            try:
                fn = getattr(mrand, "repeat")
            except AttributeError:  # pragma: no cover  # pragma: no cover
                raise NotImplementedError("Missing in compiler: repeat")
    res = fn(
        *[_u(x) for x in args],
        **{k: _u(v, k) for (k, v) in kwargs.items() if k != "stream"},
    )
    return _w(res)


def roll(*args, **kwargs):
    """Compute roll.

    Args:
        args: The args argument.
        kwargs: The kwargs argument.

    Returns:
        The result of roll.

    """
    from zero_mlx.array import array, _to_tensor
    import ml_switcheroo.ops as mops
    import ml_switcheroo.nn as mnn
    import ml_switcheroo.random as mrand
    import types

    def _u(x, param_name=None):
        if isinstance(x, list):  # pragma: no cover
            if __import__("builtins").all(
                (isinstance(i, int) for i in x)
            ):  # pragma: no cover
                return tuple(x)  # pragma: no cover
            return [_u(i, param_name) for i in x]  # pragma: no cover
        if isinstance(x, tuple):  # pragma: no cover
            return tuple((_u(i, param_name) for i in x))  # pragma: no cover
        if hasattr(x, "_tensor"):  # pragma: no cover
            return x._tensor  # pragma: no cover
        if isinstance(x, (int, float, bool, complex)):  # pragma: no cover
            if param_name not in {
                "axis",
                "rtol",
                "n",
                "m",
                "endpoint",
                "offset",
                "a_max",
                "ddof",
                "dims",
                "source",
                "axis2",
                "stop",
                "k",
                "dtype",
                "reps",
                "a_min",
                "sparse",
                "decimals",
                "strides",
                "num",
                "max",
                "indices_or_sections",
                "shape",
                "atol",
                "axis1",
                "step",
                "axes",
                "fill_value",
                "stream",
                "destination",
                "alpha",
                "beta",
                "start",
                "equal_nan",
                "keepdims",
                "min",
                "p",
            } and "roll" not in (
                "transpose",
                "moveaxis",
                "reshape",
                "expand_dims",
                "linspace",
                "logcumsumexp",
                "cummin",
                "cummax",
                "cumprod",
                "cumsum",
                "logsumexp",
                "all",
                "any",
                "argmin",
                "argmax",
                "prod",
                "sum",
                "mean",
                "var",
                "std",
                "max",
                "min",
                "tensordot",
            ):  # pragma: no cover
                return _to_tensor(x)  # pragma: no cover
        if type(x).__name__ == "ndarray":  # pragma: no cover
            return _to_tensor(x)  # pragma: no cover
        return x  # pragma: no cover

    def _w(x):
        if isinstance(x, tuple):  # pragma: no cover
            return tuple((_w(i) for i in x))  # pragma: no cover
        if isinstance(x, list):  # pragma: no cover
            return [_w(i) for i in x]  # pragma: no cover
        return array(x) if hasattr(x, "shape") else x  # pragma: no cover

    try:
        fn = getattr(mops, "roll")
    except AttributeError:  # pragma: no cover
        try:
            fn = getattr(mnn, "roll")
        except AttributeError:  # pragma: no cover
            try:
                fn = getattr(mrand, "roll")
            except AttributeError:  # pragma: no cover  # pragma: no cover
                raise NotImplementedError("Missing in compiler: roll")
    res = fn(
        *[_u(x) for x in args],
        **{k: _u(v, k) for (k, v) in kwargs.items() if k != "stream"},
    )
    return _w(res)


def sign(*args, **kwargs):
    """Compute sign.

    Args:
        args: The args argument.
        kwargs: The kwargs argument.

    Returns:
        The result of sign.

    """
    from zero_mlx.array import array, _to_tensor
    import ml_switcheroo.ops as mops
    import ml_switcheroo.nn as mnn
    import ml_switcheroo.random as mrand
    import types

    def _u(x, param_name=None):
        if isinstance(x, list):  # pragma: no cover
            if __import__("builtins").all(
                (isinstance(i, int) for i in x)
            ):  # pragma: no cover
                return tuple(x)  # pragma: no cover
            return [_u(i, param_name) for i in x]  # pragma: no cover
        if isinstance(x, tuple):  # pragma: no cover
            return tuple((_u(i, param_name) for i in x))  # pragma: no cover
        if hasattr(x, "_tensor"):  # pragma: no cover
            return x._tensor  # pragma: no cover
        if isinstance(x, (int, float, bool, complex)):  # pragma: no cover
            if param_name not in {
                "axis",
                "rtol",
                "n",
                "m",
                "endpoint",
                "offset",
                "a_max",
                "ddof",
                "dims",
                "source",
                "axis2",
                "stop",
                "k",
                "dtype",
                "reps",
                "a_min",
                "sparse",
                "decimals",
                "strides",
                "num",
                "max",
                "indices_or_sections",
                "shape",
                "atol",
                "axis1",
                "step",
                "axes",
                "fill_value",
                "stream",
                "destination",
                "alpha",
                "beta",
                "start",
                "equal_nan",
                "keepdims",
                "min",
                "p",
            } and "sign" not in (
                "transpose",
                "moveaxis",
                "reshape",
                "expand_dims",
                "linspace",
                "logcumsumexp",
                "cummin",
                "cummax",
                "cumprod",
                "cumsum",
                "logsumexp",
                "all",
                "any",
                "argmin",
                "argmax",
                "prod",
                "sum",
                "mean",
                "var",
                "std",
                "max",
                "min",
                "tensordot",
            ):  # pragma: no cover
                return _to_tensor(x)  # pragma: no cover
        if type(x).__name__ == "ndarray":  # pragma: no cover
            return _to_tensor(x)  # pragma: no cover
        return x  # pragma: no cover

    def _w(x):
        if isinstance(x, tuple):  # pragma: no cover
            return tuple((_w(i) for i in x))  # pragma: no cover
        if isinstance(x, list):  # pragma: no cover
            return [_w(i) for i in x]  # pragma: no cover
        return array(x) if hasattr(x, "shape") else x  # pragma: no cover

    try:
        fn = getattr(mops, "sign")
    except AttributeError:  # pragma: no cover
        try:
            fn = getattr(mnn, "sign")
        except AttributeError:  # pragma: no cover
            try:
                fn = getattr(mrand, "sign")
            except AttributeError:  # pragma: no cover  # pragma: no cover
                raise NotImplementedError("Missing in compiler: sign")
    res = fn(
        *[_u(x) for x in args],
        **{k: _u(v, k) for (k, v) in kwargs.items() if k != "stream"},
    )
    return _w(res)


def tensordot(*args, **kwargs):
    """Compute tensordot.

    Args:
        args: The args argument.
        kwargs: The kwargs argument.

    Returns:
        The result of tensordot.

    """
    from zero_mlx.array import array, _to_tensor
    import ml_switcheroo.ops as mops
    import ml_switcheroo.nn as mnn
    import ml_switcheroo.random as mrand
    import types

    def _u(x, param_name=None):
        if isinstance(x, list):  # pragma: no cover
            if __import__("builtins").all(
                (isinstance(i, int) for i in x)
            ):  # pragma: no cover
                return tuple(x)  # pragma: no cover
            return [_u(i, param_name) for i in x]  # pragma: no cover
        if isinstance(x, tuple):  # pragma: no cover
            return tuple((_u(i, param_name) for i in x))  # pragma: no cover
        if hasattr(x, "_tensor"):  # pragma: no cover
            return x._tensor  # pragma: no cover
        if isinstance(x, (int, float, bool, complex)):  # pragma: no cover
            if param_name not in {
                "axis",
                "rtol",
                "n",
                "m",
                "endpoint",
                "offset",
                "a_max",
                "ddof",
                "dims",
                "source",
                "axis2",
                "stop",
                "k",
                "dtype",
                "reps",
                "a_min",
                "sparse",
                "decimals",
                "strides",
                "num",
                "max",
                "indices_or_sections",
                "shape",
                "atol",
                "axis1",
                "step",
                "axes",
                "fill_value",
                "stream",
                "destination",
                "alpha",
                "beta",
                "start",
                "equal_nan",
                "keepdims",
                "min",
                "p",
            } and "tensordot" not in (
                "transpose",
                "moveaxis",
                "reshape",
                "expand_dims",
                "linspace",
                "logcumsumexp",
                "cummin",
                "cummax",
                "cumprod",
                "cumsum",
                "logsumexp",
                "all",
                "any",
                "argmin",
                "argmax",
                "prod",
                "sum",
                "mean",
                "var",
                "std",
                "max",
                "min",
                "tensordot",
            ):  # pragma: no cover
                return _to_tensor(x)  # pragma: no cover
        if type(x).__name__ == "ndarray":  # pragma: no cover
            return _to_tensor(x)  # pragma: no cover
        return x  # pragma: no cover

    def _w(x):
        if isinstance(x, tuple):  # pragma: no cover
            return tuple((_w(i) for i in x))  # pragma: no cover
        if isinstance(x, list):  # pragma: no cover
            return [_w(i) for i in x]  # pragma: no cover
        return array(x) if hasattr(x, "shape") else x  # pragma: no cover

    try:
        fn = getattr(mops, "tensordot")
    except AttributeError:  # pragma: no cover
        try:
            fn = getattr(mnn, "tensordot")
        except AttributeError:  # pragma: no cover
            try:
                fn = getattr(mrand, "tensordot")
            except AttributeError:  # pragma: no cover  # pragma: no cover
                raise NotImplementedError("Missing in compiler: tensordot")
    res = fn(
        *[_u(x) for x in args],
        **{k: _u(v, k) for (k, v) in kwargs.items() if k != "stream"},
    )
    return _w(res)


def trace(*args, **kwargs):
    """Compute trace.

    Args:
        args: The args argument.
        kwargs: The kwargs argument.

    Returns:
        The result of trace.

    """
    from zero_mlx.array import array, _to_tensor  # pragma: no cover
    import ml_switcheroo.ops as mops  # pragma: no cover
    import ml_switcheroo.nn as mnn  # pragma: no cover
    import ml_switcheroo.random as mrand  # pragma: no cover
    import types  # pragma: no cover

    def _u(x, param_name=None):  # pragma: no cover
        if isinstance(x, list):  # pragma: no cover
            if __import__("builtins").all(
                (isinstance(i, int) for i in x)
            ):  # pragma: no cover
                return tuple(x)  # pragma: no cover
            return [_u(i, param_name) for i in x]  # pragma: no cover
        if isinstance(x, tuple):  # pragma: no cover
            return tuple((_u(i, param_name) for i in x))  # pragma: no cover
        if hasattr(x, "_tensor"):  # pragma: no cover
            return x._tensor  # pragma: no cover
        if isinstance(x, (int, float, bool, complex)):  # pragma: no cover
            if param_name not in {
                "axis",
                "rtol",
                "n",
                "m",
                "endpoint",
                "offset",
                "a_max",
                "ddof",
                "dims",
                "source",
                "axis2",
                "stop",
                "k",
                "dtype",
                "reps",
                "a_min",
                "sparse",
                "decimals",
                "strides",
                "num",
                "max",
                "indices_or_sections",
                "shape",
                "atol",
                "axis1",
                "step",
                "axes",
                "fill_value",
                "stream",
                "destination",
                "alpha",
                "beta",
                "start",
                "equal_nan",
                "keepdims",
                "min",
                "p",
            } and "trace" not in (
                "transpose",
                "moveaxis",
                "reshape",
                "expand_dims",
                "linspace",
                "logcumsumexp",
                "cummin",
                "cummax",
                "cumprod",
                "cumsum",
                "logsumexp",
                "all",
                "any",
                "argmin",
                "argmax",
                "prod",
                "sum",
                "mean",
                "var",
                "std",
                "max",
                "min",
                "tensordot",
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
        fn = getattr(mops, "trace")  # pragma: no cover
    except AttributeError:  # pragma: no cover
        try:
            fn = getattr(mnn, "trace")
        except AttributeError:  # pragma: no cover
            try:
                fn = getattr(mrand, "trace")
            except AttributeError:  # pragma: no cover  # pragma: no cover
                raise NotImplementedError("Missing in compiler: trace")
    res = fn(
        *[_u(x) for x in args],
        **{k: _u(v, k) for (k, v) in kwargs.items() if k != "stream"},
    )  # pragma: no cover
    return _w(res)  # pragma: no cover


def tri(*args, **kwargs):
    """Compute tri.

    Args:
        args: The args argument.
        kwargs: The kwargs argument.

    Returns:
        The result of tri.

    """
    from zero_mlx.array import array, _to_tensor  # pragma: no cover
    import ml_switcheroo.ops as mops  # pragma: no cover
    import ml_switcheroo.nn as mnn  # pragma: no cover
    import ml_switcheroo.random as mrand  # pragma: no cover
    import types  # pragma: no cover

    def _u(x, param_name=None):  # pragma: no cover
        if isinstance(x, list):  # pragma: no cover
            if __import__("builtins").all(
                (isinstance(i, int) for i in x)
            ):  # pragma: no cover
                return tuple(x)  # pragma: no cover
            return [_u(i, param_name) for i in x]  # pragma: no cover
        if isinstance(x, tuple):  # pragma: no cover
            return tuple((_u(i, param_name) for i in x))  # pragma: no cover
        if hasattr(x, "_tensor"):  # pragma: no cover
            return x._tensor  # pragma: no cover
        if isinstance(x, (int, float, bool, complex)):  # pragma: no cover
            if param_name not in {
                "axis",
                "rtol",
                "n",
                "m",
                "endpoint",
                "offset",
                "a_max",
                "ddof",
                "dims",
                "source",
                "axis2",
                "stop",
                "k",
                "dtype",
                "reps",
                "a_min",
                "sparse",
                "decimals",
                "strides",
                "num",
                "max",
                "indices_or_sections",
                "shape",
                "atol",
                "axis1",
                "step",
                "axes",
                "fill_value",
                "stream",
                "destination",
                "alpha",
                "beta",
                "start",
                "equal_nan",
                "keepdims",
                "min",
                "p",
            } and "tri" not in (
                "transpose",
                "moveaxis",
                "reshape",
                "expand_dims",
                "linspace",
                "logcumsumexp",
                "cummin",
                "cummax",
                "cumprod",
                "cumsum",
                "logsumexp",
                "all",
                "any",
                "argmin",
                "argmax",
                "prod",
                "sum",
                "mean",
                "var",
                "std",
                "max",
                "min",
                "tensordot",
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
        fn = getattr(mops, "tri")  # pragma: no cover
    except AttributeError:  # pragma: no cover
        try:
            fn = getattr(mnn, "tri")
        except AttributeError:  # pragma: no cover
            try:
                fn = getattr(mrand, "tri")
            except AttributeError:  # pragma: no cover  # pragma: no cover
                raise NotImplementedError("Missing in compiler: tri")
    res = fn(
        *[_u(x) for x in args],
        **{k: _u(v, k) for (k, v) in kwargs.items() if k != "stream"},
    )  # pragma: no cover
    return _w(res)  # pragma: no cover


def where(*args, **kwargs):
    """Compute where.

    Args:
        args: The args argument.
        kwargs: The kwargs argument.

    Returns:
        The result of where.

    """
    from zero_mlx.array import array, _to_tensor
    import ml_switcheroo.ops as mops
    import ml_switcheroo.nn as mnn
    import ml_switcheroo.random as mrand
    import types

    def _u(x, param_name=None):
        if isinstance(x, list):  # pragma: no cover
            if __import__("builtins").all(
                (isinstance(i, int) for i in x)
            ):  # pragma: no cover
                return tuple(x)  # pragma: no cover
            return [_u(i, param_name) for i in x]  # pragma: no cover
        if isinstance(x, tuple):  # pragma: no cover
            return tuple((_u(i, param_name) for i in x))  # pragma: no cover
        if hasattr(x, "_tensor"):  # pragma: no cover
            return x._tensor  # pragma: no cover
        if isinstance(x, (int, float, bool, complex)):  # pragma: no cover
            if param_name not in {
                "axis",
                "rtol",
                "n",
                "m",
                "endpoint",
                "offset",
                "a_max",
                "ddof",
                "dims",
                "source",
                "axis2",
                "stop",
                "k",
                "dtype",
                "reps",
                "a_min",
                "sparse",
                "decimals",
                "strides",
                "num",
                "max",
                "indices_or_sections",
                "shape",
                "atol",
                "axis1",
                "step",
                "axes",
                "fill_value",
                "stream",
                "destination",
                "alpha",
                "beta",
                "start",
                "equal_nan",
                "keepdims",
                "min",
                "p",
            } and "where" not in (
                "transpose",
                "moveaxis",
                "reshape",
                "expand_dims",
                "linspace",
                "logcumsumexp",
                "cummin",
                "cummax",
                "cumprod",
                "cumsum",
                "logsumexp",
                "all",
                "any",
                "argmin",
                "argmax",
                "prod",
                "sum",
                "mean",
                "var",
                "std",
                "max",
                "min",
                "tensordot",
            ):  # pragma: no cover
                return _to_tensor(x)  # pragma: no cover
        if type(x).__name__ == "ndarray":  # pragma: no cover
            return _to_tensor(x)  # pragma: no cover
        return x  # pragma: no cover

    def _w(x):
        if isinstance(x, tuple):  # pragma: no cover
            return tuple((_w(i) for i in x))  # pragma: no cover
        if isinstance(x, list):  # pragma: no cover
            return [_w(i) for i in x]  # pragma: no cover
        return array(x) if hasattr(x, "shape") else x  # pragma: no cover

    try:
        fn = getattr(mops, "where")
    except AttributeError:  # pragma: no cover
        try:
            fn = getattr(mnn, "where")
        except AttributeError:  # pragma: no cover
            try:
                fn = getattr(mrand, "where")
            except AttributeError:  # pragma: no cover  # pragma: no cover
                raise NotImplementedError("Missing in compiler: where")
    res = fn(
        *[_u(x) for x in args],
        **{k: _u(v, k) for (k, v) in kwargs.items() if k != "stream"},
    )
    return _w(res)


def sigmoid(a, stream=None):
    """Compute sigmoid.

    Args:
        a: The a argument.
        stream: The stream argument.

    Returns:
        The result of sigmoid.

    """
    from zero_mlx.array import array, _to_tensor  # pragma: no cover
    import ml_switcheroo.ops as mops  # pragma: no cover
    import ml_switcheroo.nn as mnn  # pragma: no cover
    import ml_switcheroo.random as mrand  # pragma: no cover
    import types  # pragma: no cover

    def _u(x, param_name=None):  # pragma: no cover
        if isinstance(x, list):  # pragma: no cover
            if __import__("builtins").all(
                (isinstance(i, int) for i in x)
            ):  # pragma: no cover
                return tuple(x)  # pragma: no cover
            return [_u(i, param_name) for i in x]  # pragma: no cover
        if isinstance(x, tuple):  # pragma: no cover
            return tuple((_u(i, param_name) for i in x))  # pragma: no cover
        if hasattr(x, "_tensor"):  # pragma: no cover
            return x._tensor  # pragma: no cover
        if isinstance(x, (int, float, bool, complex)):  # pragma: no cover
            if param_name not in {
                "axis",
                "rtol",
                "n",
                "m",
                "endpoint",
                "offset",
                "a_max",
                "ddof",
                "dims",
                "source",
                "axis2",
                "stop",
                "k",
                "dtype",
                "reps",
                "a_min",
                "sparse",
                "decimals",
                "strides",
                "num",
                "max",
                "indices_or_sections",
                "shape",
                "atol",
                "axis1",
                "step",
                "axes",
                "fill_value",
                "stream",
                "destination",
                "alpha",
                "beta",
                "start",
                "equal_nan",
                "keepdims",
                "min",
                "p",
            } and "sigmoid" not in (
                "transpose",
                "moveaxis",
                "reshape",
                "expand_dims",
                "linspace",
                "logcumsumexp",
                "cummin",
                "cummax",
                "cumprod",
                "cumsum",
                "logsumexp",
                "all",
                "any",
                "argmin",
                "argmax",
                "prod",
                "sum",
                "mean",
                "var",
                "std",
                "max",
                "min",
                "tensordot",
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
        fn = getattr(mops, "sigmoid")  # pragma: no cover
    except AttributeError:  # pragma: no cover
        try:
            fn = getattr(mnn, "sigmoid")
        except AttributeError:  # pragma: no cover
            try:
                fn = getattr(mrand, "sigmoid")
            except AttributeError:  # pragma: no cover  # pragma: no cover
                raise NotImplementedError("Missing in compiler: sigmoid")
    res = fn(_u(a, "a"))  # pragma: no cover
    return _w(res)  # pragma: no cover


def softmax(a, axis=-1, stream=None):
    """Compute softmax.

    Args:
        a: The a argument.
        axis: The axis argument.
        stream: The stream argument.

    Returns:
        The result of softmax.

    """
    from zero_mlx.array import array, _to_tensor  # pragma: no cover
    import ml_switcheroo.ops as mops  # pragma: no cover
    import ml_switcheroo.nn as mnn  # pragma: no cover
    import ml_switcheroo.random as mrand  # pragma: no cover
    import types  # pragma: no cover

    def _u(x, param_name=None):  # pragma: no cover
        if isinstance(x, list):  # pragma: no cover
            if __import__("builtins").all(
                (isinstance(i, int) for i in x)
            ):  # pragma: no cover
                return tuple(x)  # pragma: no cover
            return [_u(i, param_name) for i in x]  # pragma: no cover
        if isinstance(x, tuple):  # pragma: no cover
            return tuple((_u(i, param_name) for i in x))  # pragma: no cover
        if hasattr(x, "_tensor"):  # pragma: no cover
            return x._tensor  # pragma: no cover
        if isinstance(x, (int, float, bool, complex)):  # pragma: no cover
            if param_name not in {
                "axis",
                "rtol",
                "n",
                "m",
                "endpoint",
                "offset",
                "a_max",
                "ddof",
                "dims",
                "source",
                "axis2",
                "stop",
                "k",
                "dtype",
                "reps",
                "a_min",
                "sparse",
                "decimals",
                "strides",
                "num",
                "max",
                "indices_or_sections",
                "shape",
                "atol",
                "axis1",
                "step",
                "axes",
                "fill_value",
                "stream",
                "destination",
                "alpha",
                "beta",
                "start",
                "equal_nan",
                "keepdims",
                "min",
                "p",
            } and "softmax" not in (
                "transpose",
                "moveaxis",
                "reshape",
                "expand_dims",
                "linspace",
                "logcumsumexp",
                "cummin",
                "cummax",
                "cumprod",
                "cumsum",
                "logsumexp",
                "all",
                "any",
                "argmin",
                "argmax",
                "prod",
                "sum",
                "mean",
                "var",
                "std",
                "max",
                "min",
                "tensordot",
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
        fn = getattr(mops, "softmax")  # pragma: no cover
    except AttributeError:  # pragma: no cover
        try:
            fn = getattr(mnn, "softmax")
        except AttributeError:  # pragma: no cover
            try:
                fn = getattr(mrand, "softmax")
            except AttributeError:  # pragma: no cover  # pragma: no cover
                raise NotImplementedError("Missing in compiler: softmax")
    res = fn(_u(a, "a"), _u(axis, "axis"))  # pragma: no cover
    return _w(res)  # pragma: no cover


def depends(a, b):
    """Compute depends.

    Args:
        a: The a argument.
        b: The b argument.

    Returns:
        The result of depends.

    """
    from zero_mlx.array import array, _to_tensor
    import ml_switcheroo.ops as mops
    import ml_switcheroo.nn as mnn
    import ml_switcheroo.random as mrand
    import types

    def _u(x, param_name=None):
        if isinstance(x, list):  # pragma: no cover
            if __import__("builtins").all(
                (isinstance(i, int) for i in x)
            ):  # pragma: no cover
                return tuple(x)  # pragma: no cover
            return [_u(i, param_name) for i in x]  # pragma: no cover
        if isinstance(x, tuple):  # pragma: no cover
            return tuple((_u(i, param_name) for i in x))  # pragma: no cover
        if hasattr(x, "_tensor"):  # pragma: no cover
            return x._tensor  # pragma: no cover
        if isinstance(x, (int, float, bool, complex)):  # pragma: no cover
            if param_name not in {
                "axis",
                "rtol",
                "n",
                "m",
                "endpoint",
                "offset",
                "a_max",
                "ddof",
                "dims",
                "source",
                "axis2",
                "stop",
                "k",
                "dtype",
                "reps",
                "a_min",
                "sparse",
                "decimals",
                "strides",
                "num",
                "max",
                "indices_or_sections",
                "shape",
                "atol",
                "axis1",
                "step",
                "axes",
                "fill_value",
                "stream",
                "destination",
                "alpha",
                "beta",
                "start",
                "equal_nan",
                "keepdims",
                "min",
                "p",
            } and "depends" not in (
                "transpose",
                "moveaxis",
                "reshape",
                "expand_dims",
                "linspace",
                "logcumsumexp",
                "cummin",
                "cummax",
                "cumprod",
                "cumsum",
                "logsumexp",
                "all",
                "any",
                "argmin",
                "argmax",
                "prod",
                "sum",
                "mean",
                "var",
                "std",
                "max",
                "min",
                "tensordot",
            ):  # pragma: no cover
                return _to_tensor(x)  # pragma: no cover
        if type(x).__name__ == "ndarray":  # pragma: no cover
            return _to_tensor(x)  # pragma: no cover
        return x  # pragma: no cover

    def _w(x):
        if isinstance(x, tuple):  # pragma: no cover
            return tuple((_w(i) for i in x))  # pragma: no cover
        if isinstance(x, list):  # pragma: no cover
            return [_w(i) for i in x]  # pragma: no cover
        return array(x) if hasattr(x, "shape") else x  # pragma: no cover

    try:
        fn = getattr(mops, "depends")
    except AttributeError:  # pragma: no cover
        try:
            fn = getattr(mnn, "depends")
        except AttributeError:  # pragma: no cover
            try:
                fn = getattr(mrand, "depends")
            except AttributeError:  # pragma: no cover  # pragma: no cover
                raise NotImplementedError("Missing in compiler: depends")
    res = fn(_u(a, "a"), _u(b, "b"))
    return _w(res)


def to_fp8(a):
    """Compute to_fp8.

    Args:
        a: The a argument.

    Returns:
        The result of to_fp8.

    """
    from zero_mlx.array import array, _to_tensor
    import ml_switcheroo.ops as mops
    import ml_switcheroo.nn as mnn
    import ml_switcheroo.random as mrand
    import types

    def _u(x, param_name=None):
        if isinstance(x, list):  # pragma: no cover
            if __import__("builtins").all(
                (isinstance(i, int) for i in x)
            ):  # pragma: no cover
                return tuple(x)  # pragma: no cover
            return [_u(i, param_name) for i in x]  # pragma: no cover
        if isinstance(x, tuple):  # pragma: no cover
            return tuple((_u(i, param_name) for i in x))  # pragma: no cover
        if hasattr(x, "_tensor"):  # pragma: no cover
            return x._tensor  # pragma: no cover
        if isinstance(x, (int, float, bool, complex)):  # pragma: no cover
            if param_name not in {
                "axis",
                "rtol",
                "n",
                "m",
                "endpoint",
                "offset",
                "a_max",
                "ddof",
                "dims",
                "source",
                "axis2",
                "stop",
                "k",
                "dtype",
                "reps",
                "a_min",
                "sparse",
                "decimals",
                "strides",
                "num",
                "max",
                "indices_or_sections",
                "shape",
                "atol",
                "axis1",
                "step",
                "axes",
                "fill_value",
                "stream",
                "destination",
                "alpha",
                "beta",
                "start",
                "equal_nan",
                "keepdims",
                "min",
                "p",
            } and "to_fp8" not in (
                "transpose",
                "moveaxis",
                "reshape",
                "expand_dims",
                "linspace",
                "logcumsumexp",
                "cummin",
                "cummax",
                "cumprod",
                "cumsum",
                "logsumexp",
                "all",
                "any",
                "argmin",
                "argmax",
                "prod",
                "sum",
                "mean",
                "var",
                "std",
                "max",
                "min",
                "tensordot",
            ):  # pragma: no cover
                return _to_tensor(x)  # pragma: no cover
        if type(x).__name__ == "ndarray":  # pragma: no cover
            return _to_tensor(x)  # pragma: no cover
        return x  # pragma: no cover

    def _w(x):
        if isinstance(x, tuple):  # pragma: no cover
            return tuple((_w(i) for i in x))  # pragma: no cover
        if isinstance(x, list):  # pragma: no cover
            return [_w(i) for i in x]  # pragma: no cover
        return array(x) if hasattr(x, "shape") else x  # pragma: no cover

    try:
        fn = getattr(mops, "to_fp8")
    except AttributeError:  # pragma: no cover
        try:
            fn = getattr(mnn, "to_fp8")
        except AttributeError:  # pragma: no cover
            try:
                fn = getattr(mrand, "to_fp8")
            except AttributeError:  # pragma: no cover  # pragma: no cover
                raise NotImplementedError("Missing in compiler: to_fp8")
    res = fn(_u(a, "a"))
    return _w(res)


def from_fp8(a):
    """Compute from_fp8.

    Args:
        a: The a argument.

    Returns:
        The result of from_fp8.

    """
    from zero_mlx.array import array, _to_tensor
    import ml_switcheroo.ops as mops
    import ml_switcheroo.nn as mnn
    import ml_switcheroo.random as mrand
    import types

    def _u(x, param_name=None):
        if isinstance(x, list):  # pragma: no cover
            if __import__("builtins").all(
                (isinstance(i, int) for i in x)
            ):  # pragma: no cover
                return tuple(x)  # pragma: no cover
            return [_u(i, param_name) for i in x]  # pragma: no cover
        if isinstance(x, tuple):  # pragma: no cover
            return tuple((_u(i, param_name) for i in x))  # pragma: no cover
        if hasattr(x, "_tensor"):  # pragma: no cover
            return x._tensor  # pragma: no cover
        if isinstance(x, (int, float, bool, complex)):  # pragma: no cover
            if param_name not in {
                "axis",
                "rtol",
                "n",
                "m",
                "endpoint",
                "offset",
                "a_max",
                "ddof",
                "dims",
                "source",
                "axis2",
                "stop",
                "k",
                "dtype",
                "reps",
                "a_min",
                "sparse",
                "decimals",
                "strides",
                "num",
                "max",
                "indices_or_sections",
                "shape",
                "atol",
                "axis1",
                "step",
                "axes",
                "fill_value",
                "stream",
                "destination",
                "alpha",
                "beta",
                "start",
                "equal_nan",
                "keepdims",
                "min",
                "p",
            } and "from_fp8" not in (
                "transpose",
                "moveaxis",
                "reshape",
                "expand_dims",
                "linspace",
                "logcumsumexp",
                "cummin",
                "cummax",
                "cumprod",
                "cumsum",
                "logsumexp",
                "all",
                "any",
                "argmin",
                "argmax",
                "prod",
                "sum",
                "mean",
                "var",
                "std",
                "max",
                "min",
                "tensordot",
            ):  # pragma: no cover
                return _to_tensor(x)  # pragma: no cover
        if type(x).__name__ == "ndarray":  # pragma: no cover
            return _to_tensor(x)  # pragma: no cover
        return x  # pragma: no cover

    def _w(x):
        if isinstance(x, tuple):  # pragma: no cover
            return tuple((_w(i) for i in x))  # pragma: no cover
        if isinstance(x, list):  # pragma: no cover
            return [_w(i) for i in x]  # pragma: no cover
        return array(x) if hasattr(x, "shape") else x  # pragma: no cover

    try:
        fn = getattr(mops, "from_fp8")
    except AttributeError:  # pragma: no cover
        try:
            fn = getattr(mnn, "from_fp8")
        except AttributeError:  # pragma: no cover
            try:
                fn = getattr(mrand, "from_fp8")
            except AttributeError:  # pragma: no cover  # pragma: no cover
                raise NotImplementedError("Missing in compiler: from_fp8")
    res = fn(_u(a, "a"))
    return _w(res)


def arcsin(a: Any, stream: Any = None) -> Any:
    """Compute arcsin.

    Args:
        a: The a argument.
        stream: The stream argument.

    Returns:
        The result of arcsin.

    """
    from zero_mlx.array import array, _to_tensor
    import ml_switcheroo.ops as mops
    import ml_switcheroo.nn as mnn
    import ml_switcheroo.random as mrand
    import types

    def _u(x, param_name=None):
        if isinstance(x, list):  # pragma: no cover
            if __import__("builtins").all(
                (isinstance(i, int) for i in x)
            ):  # pragma: no cover
                return tuple(x)  # pragma: no cover
            return [_u(i, param_name) for i in x]  # pragma: no cover
        if isinstance(x, tuple):  # pragma: no cover
            return tuple((_u(i, param_name) for i in x))  # pragma: no cover
        if hasattr(x, "_tensor"):  # pragma: no cover
            return x._tensor  # pragma: no cover
        if isinstance(x, (int, float, bool, complex)):  # pragma: no cover
            if param_name not in {
                "axis",
                "rtol",
                "n",
                "m",
                "endpoint",
                "offset",
                "a_max",
                "ddof",
                "dims",
                "source",
                "axis2",
                "stop",
                "k",
                "dtype",
                "reps",
                "a_min",
                "sparse",
                "decimals",
                "strides",
                "num",
                "max",
                "indices_or_sections",
                "shape",
                "atol",
                "axis1",
                "step",
                "axes",
                "fill_value",
                "stream",
                "destination",
                "alpha",
                "beta",
                "start",
                "equal_nan",
                "keepdims",
                "min",
                "p",
            } and "arcsin" not in (
                "transpose",
                "moveaxis",
                "reshape",
                "expand_dims",
                "linspace",
                "logcumsumexp",
                "cummin",
                "cummax",
                "cumprod",
                "cumsum",
                "logsumexp",
                "all",
                "any",
                "argmin",
                "argmax",
                "prod",
                "sum",
                "mean",
                "var",
                "std",
                "max",
                "min",
                "tensordot",
            ):  # pragma: no cover
                return _to_tensor(x)  # pragma: no cover
        if type(x).__name__ == "ndarray":  # pragma: no cover
            return _to_tensor(x)  # pragma: no cover
        return x  # pragma: no cover

    def _w(x):
        if isinstance(x, tuple):  # pragma: no cover
            return tuple((_w(i) for i in x))  # pragma: no cover
        if isinstance(x, list):  # pragma: no cover
            return [_w(i) for i in x]  # pragma: no cover
        return array(x) if hasattr(x, "shape") else x  # pragma: no cover

    try:
        fn = getattr(mops, "asin")
    except AttributeError:  # pragma: no cover
        try:
            fn = getattr(mnn, "asin")
        except AttributeError:  # pragma: no cover
            try:
                fn = getattr(mrand, "asin")
            except AttributeError:  # pragma: no cover  # pragma: no cover
                raise NotImplementedError("Missing in compiler: asin")
    res = fn(_u(a, "a"))
    return _w(res)


def arccos(a: Any, stream: Any = None) -> Any:
    """Compute arccos.

    Args:
        a: The a argument.
        stream: The stream argument.

    Returns:
        The result of arccos.

    """
    from zero_mlx.array import array, _to_tensor
    import ml_switcheroo.ops as mops
    import ml_switcheroo.nn as mnn
    import ml_switcheroo.random as mrand
    import types

    def _u(x, param_name=None):
        if isinstance(x, list):  # pragma: no cover
            if __import__("builtins").all(
                (isinstance(i, int) for i in x)
            ):  # pragma: no cover
                return tuple(x)  # pragma: no cover
            return [_u(i, param_name) for i in x]  # pragma: no cover
        if isinstance(x, tuple):  # pragma: no cover
            return tuple((_u(i, param_name) for i in x))  # pragma: no cover
        if hasattr(x, "_tensor"):  # pragma: no cover
            return x._tensor  # pragma: no cover
        if isinstance(x, (int, float, bool, complex)):  # pragma: no cover
            if param_name not in {
                "axis",
                "rtol",
                "n",
                "m",
                "endpoint",
                "offset",
                "a_max",
                "ddof",
                "dims",
                "source",
                "axis2",
                "stop",
                "k",
                "dtype",
                "reps",
                "a_min",
                "sparse",
                "decimals",
                "strides",
                "num",
                "max",
                "indices_or_sections",
                "shape",
                "atol",
                "axis1",
                "step",
                "axes",
                "fill_value",
                "stream",
                "destination",
                "alpha",
                "beta",
                "start",
                "equal_nan",
                "keepdims",
                "min",
                "p",
            } and "arccos" not in (
                "transpose",
                "moveaxis",
                "reshape",
                "expand_dims",
                "linspace",
                "logcumsumexp",
                "cummin",
                "cummax",
                "cumprod",
                "cumsum",
                "logsumexp",
                "all",
                "any",
                "argmin",
                "argmax",
                "prod",
                "sum",
                "mean",
                "var",
                "std",
                "max",
                "min",
                "tensordot",
            ):  # pragma: no cover
                return _to_tensor(x)  # pragma: no cover
        if type(x).__name__ == "ndarray":  # pragma: no cover
            return _to_tensor(x)  # pragma: no cover
        return x  # pragma: no cover

    def _w(x):
        if isinstance(x, tuple):  # pragma: no cover
            return tuple((_w(i) for i in x))  # pragma: no cover
        if isinstance(x, list):  # pragma: no cover
            return [_w(i) for i in x]  # pragma: no cover
        return array(x) if hasattr(x, "shape") else x  # pragma: no cover

    try:
        fn = getattr(mops, "acos")
    except AttributeError:  # pragma: no cover
        try:
            fn = getattr(mnn, "acos")
        except AttributeError:  # pragma: no cover
            try:
                fn = getattr(mrand, "acos")
            except AttributeError:  # pragma: no cover  # pragma: no cover
                raise NotImplementedError("Missing in compiler: acos")
    res = fn(_u(a, "a"))
    return _w(res)


def arctan(a: Any, stream: Any = None) -> Any:
    """Compute arctan.

    Args:
        a: The a argument.
        stream: The stream argument.

    Returns:
        The result of arctan.

    """
    from zero_mlx.array import array, _to_tensor
    import ml_switcheroo.ops as mops
    import ml_switcheroo.nn as mnn
    import ml_switcheroo.random as mrand
    import types

    def _u(x, param_name=None):
        if isinstance(x, list):  # pragma: no cover
            if __import__("builtins").all(
                (isinstance(i, int) for i in x)
            ):  # pragma: no cover
                return tuple(x)  # pragma: no cover
            return [_u(i, param_name) for i in x]  # pragma: no cover
        if isinstance(x, tuple):  # pragma: no cover
            return tuple((_u(i, param_name) for i in x))  # pragma: no cover
        if hasattr(x, "_tensor"):  # pragma: no cover
            return x._tensor  # pragma: no cover
        if isinstance(x, (int, float, bool, complex)):  # pragma: no cover
            if param_name not in {
                "axis",
                "rtol",
                "n",
                "m",
                "endpoint",
                "offset",
                "a_max",
                "ddof",
                "dims",
                "source",
                "axis2",
                "stop",
                "k",
                "dtype",
                "reps",
                "a_min",
                "sparse",
                "decimals",
                "strides",
                "num",
                "max",
                "indices_or_sections",
                "shape",
                "atol",
                "axis1",
                "step",
                "axes",
                "fill_value",
                "stream",
                "destination",
                "alpha",
                "beta",
                "start",
                "equal_nan",
                "keepdims",
                "min",
                "p",
            } and "arctan" not in (
                "transpose",
                "moveaxis",
                "reshape",
                "expand_dims",
                "linspace",
                "logcumsumexp",
                "cummin",
                "cummax",
                "cumprod",
                "cumsum",
                "logsumexp",
                "all",
                "any",
                "argmin",
                "argmax",
                "prod",
                "sum",
                "mean",
                "var",
                "std",
                "max",
                "min",
                "tensordot",
            ):  # pragma: no cover
                return _to_tensor(x)  # pragma: no cover
        if type(x).__name__ == "ndarray":  # pragma: no cover
            return _to_tensor(x)  # pragma: no cover
        return x  # pragma: no cover

    def _w(x):
        if isinstance(x, tuple):  # pragma: no cover
            return tuple((_w(i) for i in x))  # pragma: no cover
        if isinstance(x, list):  # pragma: no cover
            return [_w(i) for i in x]  # pragma: no cover
        return array(x) if hasattr(x, "shape") else x  # pragma: no cover

    try:
        fn = getattr(mops, "atan")
    except AttributeError:  # pragma: no cover
        try:
            fn = getattr(mnn, "atan")
        except AttributeError:  # pragma: no cover
            try:
                fn = getattr(mrand, "atan")
            except AttributeError:  # pragma: no cover  # pragma: no cover
                raise NotImplementedError("Missing in compiler: atan")
    res = fn(_u(a, "a"))
    return _w(res)


def arcsinh(a: Any, stream: Any = None) -> Any:
    """Compute arcsinh.

    Args:
        a: The a argument.
        stream: The stream argument.

    Returns:
        The result of arcsinh.

    """
    from zero_mlx.array import array, _to_tensor  # pragma: no cover
    import ml_switcheroo.ops as mops  # pragma: no cover
    import ml_switcheroo.nn as mnn  # pragma: no cover
    import ml_switcheroo.random as mrand  # pragma: no cover
    import types  # pragma: no cover

    def _u(x, param_name=None):  # pragma: no cover
        if isinstance(x, list):  # pragma: no cover
            if __import__("builtins").all(
                (isinstance(i, int) for i in x)
            ):  # pragma: no cover
                return tuple(x)  # pragma: no cover
            return [_u(i, param_name) for i in x]  # pragma: no cover
        if isinstance(x, tuple):  # pragma: no cover
            return tuple((_u(i, param_name) for i in x))  # pragma: no cover
        if hasattr(x, "_tensor"):  # pragma: no cover
            return x._tensor  # pragma: no cover
        if isinstance(x, (int, float, bool, complex)):  # pragma: no cover
            if param_name not in {
                "axis",
                "rtol",
                "n",
                "m",
                "endpoint",
                "offset",
                "a_max",
                "ddof",
                "dims",
                "source",
                "axis2",
                "stop",
                "k",
                "dtype",
                "reps",
                "a_min",
                "sparse",
                "decimals",
                "strides",
                "num",
                "max",
                "indices_or_sections",
                "shape",
                "atol",
                "axis1",
                "step",
                "axes",
                "fill_value",
                "stream",
                "destination",
                "alpha",
                "beta",
                "start",
                "equal_nan",
                "keepdims",
                "min",
                "p",
            } and "arcsinh" not in (
                "transpose",
                "moveaxis",
                "reshape",
                "expand_dims",
                "linspace",
                "logcumsumexp",
                "cummin",
                "cummax",
                "cumprod",
                "cumsum",
                "logsumexp",
                "all",
                "any",
                "argmin",
                "argmax",
                "prod",
                "sum",
                "mean",
                "var",
                "std",
                "max",
                "min",
                "tensordot",
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
        fn = getattr(mops, "asinh")  # pragma: no cover
    except AttributeError:  # pragma: no cover
        try:
            fn = getattr(mnn, "asinh")
        except AttributeError:  # pragma: no cover
            try:
                fn = getattr(mrand, "asinh")
            except AttributeError:  # pragma: no cover  # pragma: no cover
                raise NotImplementedError("Missing in compiler: asinh")
    res = fn(_u(a, "a"))  # pragma: no cover
    return _w(res)  # pragma: no cover


def arctanh(a: Any, stream: Any = None) -> Any:
    """Compute arctanh.

    Args:
        a: The a argument.
        stream: The stream argument.

    Returns:
        The result of arctanh.

    """
    from zero_mlx.array import array, _to_tensor  # pragma: no cover
    import ml_switcheroo.ops as mops  # pragma: no cover
    import ml_switcheroo.nn as mnn  # pragma: no cover
    import ml_switcheroo.random as mrand  # pragma: no cover
    import types  # pragma: no cover

    def _u(x, param_name=None):  # pragma: no cover
        if isinstance(x, list):  # pragma: no cover
            if __import__("builtins").all(
                (isinstance(i, int) for i in x)
            ):  # pragma: no cover
                return tuple(x)  # pragma: no cover
            return [_u(i, param_name) for i in x]  # pragma: no cover
        if isinstance(x, tuple):  # pragma: no cover
            return tuple((_u(i, param_name) for i in x))  # pragma: no cover
        if hasattr(x, "_tensor"):  # pragma: no cover
            return x._tensor  # pragma: no cover
        if isinstance(x, (int, float, bool, complex)):  # pragma: no cover
            if param_name not in {
                "axis",
                "rtol",
                "n",
                "m",
                "endpoint",
                "offset",
                "a_max",
                "ddof",
                "dims",
                "source",
                "axis2",
                "stop",
                "k",
                "dtype",
                "reps",
                "a_min",
                "sparse",
                "decimals",
                "strides",
                "num",
                "max",
                "indices_or_sections",
                "shape",
                "atol",
                "axis1",
                "step",
                "axes",
                "fill_value",
                "stream",
                "destination",
                "alpha",
                "beta",
                "start",
                "equal_nan",
                "keepdims",
                "min",
                "p",
            } and "arctanh" not in (
                "transpose",
                "moveaxis",
                "reshape",
                "expand_dims",
                "linspace",
                "logcumsumexp",
                "cummin",
                "cummax",
                "cumprod",
                "cumsum",
                "logsumexp",
                "all",
                "any",
                "argmin",
                "argmax",
                "prod",
                "sum",
                "mean",
                "var",
                "std",
                "max",
                "min",
                "tensordot",
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
        fn = getattr(mops, "atanh")  # pragma: no cover
    except AttributeError:  # pragma: no cover
        try:
            fn = getattr(mnn, "atanh")
        except AttributeError:  # pragma: no cover
            try:
                fn = getattr(mrand, "atanh")
            except AttributeError:  # pragma: no cover  # pragma: no cover
                raise NotImplementedError("Missing in compiler: atanh")
    res = fn(_u(a, "a"))  # pragma: no cover
    return _w(res)  # pragma: no cover


def log2(a: Any, stream: Any = None) -> Any:
    """Compute log2.

    Args:
        a: The a argument.
        stream: The stream argument.

    Returns:
        The result of log2.

    """
    from zero_mlx.array import array, _to_tensor
    import ml_switcheroo.ops as mops
    import ml_switcheroo.nn as mnn
    import ml_switcheroo.random as mrand
    import types

    def _u(x, param_name=None):
        if isinstance(x, list):  # pragma: no cover
            if __import__("builtins").all(
                (isinstance(i, int) for i in x)
            ):  # pragma: no cover
                return tuple(x)  # pragma: no cover
            return [_u(i, param_name) for i in x]  # pragma: no cover
        if isinstance(x, tuple):  # pragma: no cover
            return tuple((_u(i, param_name) for i in x))  # pragma: no cover
        if hasattr(x, "_tensor"):  # pragma: no cover
            return x._tensor  # pragma: no cover
        if isinstance(x, (int, float, bool, complex)):  # pragma: no cover
            if param_name not in {
                "axis",
                "rtol",
                "n",
                "m",
                "endpoint",
                "offset",
                "a_max",
                "ddof",
                "dims",
                "source",
                "axis2",
                "stop",
                "k",
                "dtype",
                "reps",
                "a_min",
                "sparse",
                "decimals",
                "strides",
                "num",
                "max",
                "indices_or_sections",
                "shape",
                "atol",
                "axis1",
                "step",
                "axes",
                "fill_value",
                "stream",
                "destination",
                "alpha",
                "beta",
                "start",
                "equal_nan",
                "keepdims",
                "min",
                "p",
            } and "log2" not in (
                "transpose",
                "moveaxis",
                "reshape",
                "expand_dims",
                "linspace",
                "logcumsumexp",
                "cummin",
                "cummax",
                "cumprod",
                "cumsum",
                "logsumexp",
                "all",
                "any",
                "argmin",
                "argmax",
                "prod",
                "sum",
                "mean",
                "var",
                "std",
                "max",
                "min",
                "tensordot",
            ):  # pragma: no cover
                return _to_tensor(x)  # pragma: no cover
        if type(x).__name__ == "ndarray":  # pragma: no cover
            return _to_tensor(x)  # pragma: no cover
        return x  # pragma: no cover

    def _w(x):
        if isinstance(x, tuple):  # pragma: no cover
            return tuple((_w(i) for i in x))  # pragma: no cover
        if isinstance(x, list):  # pragma: no cover
            return [_w(i) for i in x]  # pragma: no cover
        return array(x) if hasattr(x, "shape") else x  # pragma: no cover

    try:
        fn = getattr(mops, "log2")
    except AttributeError:  # pragma: no cover
        try:
            fn = getattr(mnn, "log2")
        except AttributeError:  # pragma: no cover
            try:
                fn = getattr(mrand, "log2")
            except AttributeError:  # pragma: no cover  # pragma: no cover
                raise NotImplementedError("Missing in compiler: log2")
    res = fn(_u(a, "a"))
    return _w(res)


def log10(a: Any, stream: Any = None) -> Any:
    """Compute log10.

    Args:
        a: The a argument.
        stream: The stream argument.

    Returns:
        The result of log10.

    """
    from zero_mlx.array import array, _to_tensor
    import ml_switcheroo.ops as mops
    import ml_switcheroo.nn as mnn
    import ml_switcheroo.random as mrand
    import types

    def _u(x, param_name=None):
        if isinstance(x, list):  # pragma: no cover
            if __import__("builtins").all(
                (isinstance(i, int) for i in x)
            ):  # pragma: no cover
                return tuple(x)  # pragma: no cover
            return [_u(i, param_name) for i in x]  # pragma: no cover
        if isinstance(x, tuple):  # pragma: no cover
            return tuple((_u(i, param_name) for i in x))  # pragma: no cover
        if hasattr(x, "_tensor"):  # pragma: no cover
            return x._tensor  # pragma: no cover
        if isinstance(x, (int, float, bool, complex)):  # pragma: no cover
            if param_name not in {
                "axis",
                "rtol",
                "n",
                "m",
                "endpoint",
                "offset",
                "a_max",
                "ddof",
                "dims",
                "source",
                "axis2",
                "stop",
                "k",
                "dtype",
                "reps",
                "a_min",
                "sparse",
                "decimals",
                "strides",
                "num",
                "max",
                "indices_or_sections",
                "shape",
                "atol",
                "axis1",
                "step",
                "axes",
                "fill_value",
                "stream",
                "destination",
                "alpha",
                "beta",
                "start",
                "equal_nan",
                "keepdims",
                "min",
                "p",
            } and "log10" not in (
                "transpose",
                "moveaxis",
                "reshape",
                "expand_dims",
                "linspace",
                "logcumsumexp",
                "cummin",
                "cummax",
                "cumprod",
                "cumsum",
                "logsumexp",
                "all",
                "any",
                "argmin",
                "argmax",
                "prod",
                "sum",
                "mean",
                "var",
                "std",
                "max",
                "min",
                "tensordot",
            ):  # pragma: no cover
                return _to_tensor(x)  # pragma: no cover
        if type(x).__name__ == "ndarray":  # pragma: no cover
            return _to_tensor(x)  # pragma: no cover
        return x  # pragma: no cover

    def _w(x):
        if isinstance(x, tuple):  # pragma: no cover
            return tuple((_w(i) for i in x))  # pragma: no cover
        if isinstance(x, list):  # pragma: no cover
            return [_w(i) for i in x]  # pragma: no cover
        return array(x) if hasattr(x, "shape") else x  # pragma: no cover

    try:
        fn = getattr(mops, "log10")
    except AttributeError:  # pragma: no cover
        try:
            fn = getattr(mnn, "log10")
        except AttributeError:  # pragma: no cover
            try:
                fn = getattr(mrand, "log10")
            except AttributeError:  # pragma: no cover  # pragma: no cover
                raise NotImplementedError("Missing in compiler: log10")
    res = fn(_u(a, "a"))
    return _w(res)


def imag(a: Any, stream: Any = None) -> Any:
    """Compute imag.

    Args:
        a: The a argument.
        stream: The stream argument.

    Returns:
        The result of imag.

    """
    from zero_mlx.array import array, _to_tensor  # pragma: no cover
    import ml_switcheroo.ops as mops  # pragma: no cover
    import ml_switcheroo.nn as mnn  # pragma: no cover
    import ml_switcheroo.random as mrand  # pragma: no cover
    import types  # pragma: no cover

    def _u(x, param_name=None):  # pragma: no cover
        if isinstance(x, list):  # pragma: no cover
            if __import__("builtins").all(
                (isinstance(i, int) for i in x)
            ):  # pragma: no cover
                return tuple(x)  # pragma: no cover
            return [_u(i, param_name) for i in x]  # pragma: no cover
        if isinstance(x, tuple):  # pragma: no cover
            return tuple((_u(i, param_name) for i in x))  # pragma: no cover
        if hasattr(x, "_tensor"):  # pragma: no cover
            return x._tensor  # pragma: no cover
        if isinstance(x, (int, float, bool, complex)):  # pragma: no cover
            if param_name not in {
                "axis",
                "rtol",
                "n",
                "m",
                "endpoint",
                "offset",
                "a_max",
                "ddof",
                "dims",
                "source",
                "axis2",
                "stop",
                "k",
                "dtype",
                "reps",
                "a_min",
                "sparse",
                "decimals",
                "strides",
                "num",
                "max",
                "indices_or_sections",
                "shape",
                "atol",
                "axis1",
                "step",
                "axes",
                "fill_value",
                "stream",
                "destination",
                "alpha",
                "beta",
                "start",
                "equal_nan",
                "keepdims",
                "min",
                "p",
            } and "imag" not in (
                "transpose",
                "moveaxis",
                "reshape",
                "expand_dims",
                "linspace",
                "logcumsumexp",
                "cummin",
                "cummax",
                "cumprod",
                "cumsum",
                "logsumexp",
                "all",
                "any",
                "argmin",
                "argmax",
                "prod",
                "sum",
                "mean",
                "var",
                "std",
                "max",
                "min",
                "tensordot",
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
        fn = getattr(mops, "imag")  # pragma: no cover
    except AttributeError:  # pragma: no cover
        try:
            fn = getattr(mnn, "imag")
        except AttributeError:  # pragma: no cover
            try:
                fn = getattr(mrand, "imag")
            except AttributeError:  # pragma: no cover  # pragma: no cover
                raise NotImplementedError("Missing in compiler: imag")
    res = fn(_u(a, "a"))  # pragma: no cover
    return _w(res)  # pragma: no cover


def bitwise_and(*args, **kwargs):
    """Compute bitwise_and.

    Args:
        args: The args argument.
        kwargs: The kwargs argument.

    Returns:
        The result of bitwise_and.

    """
    from zero_mlx.array import array, _to_tensor  # pragma: no cover
    import ml_switcheroo.ops as mops  # pragma: no cover
    import ml_switcheroo.nn as mnn  # pragma: no cover
    import ml_switcheroo.random as mrand  # pragma: no cover
    import types  # pragma: no cover

    def _u(x, param_name=None):  # pragma: no cover
        if isinstance(x, list):  # pragma: no cover
            if __import__("builtins").all(
                (isinstance(i, int) for i in x)
            ):  # pragma: no cover
                return tuple(x)  # pragma: no cover
            return [_u(i, param_name) for i in x]  # pragma: no cover
        if isinstance(x, tuple):  # pragma: no cover
            return tuple((_u(i, param_name) for i in x))  # pragma: no cover
        if hasattr(x, "_tensor"):  # pragma: no cover
            return x._tensor  # pragma: no cover
        if isinstance(x, (int, float, bool, complex)):  # pragma: no cover
            if param_name not in {
                "axis",
                "rtol",
                "n",
                "m",
                "endpoint",
                "offset",
                "a_max",
                "ddof",
                "dims",
                "source",
                "axis2",
                "stop",
                "k",
                "dtype",
                "reps",
                "a_min",
                "sparse",
                "decimals",
                "strides",
                "num",
                "max",
                "indices_or_sections",
                "shape",
                "atol",
                "axis1",
                "step",
                "axes",
                "fill_value",
                "stream",
                "destination",
                "alpha",
                "beta",
                "start",
                "equal_nan",
                "keepdims",
                "min",
                "p",
            } and "bitwise_and" not in (
                "transpose",
                "moveaxis",
                "reshape",
                "expand_dims",
                "linspace",
                "logcumsumexp",
                "cummin",
                "cummax",
                "cumprod",
                "cumsum",
                "logsumexp",
                "all",
                "any",
                "argmin",
                "argmax",
                "prod",
                "sum",
                "mean",
                "var",
                "std",
                "max",
                "min",
                "tensordot",
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
        fn = getattr(mops, "bitwise_and")  # pragma: no cover
    except AttributeError:  # pragma: no cover
        try:
            fn = getattr(mnn, "bitwise_and")
        except AttributeError:  # pragma: no cover
            try:
                fn = getattr(mrand, "bitwise_and")
            except AttributeError:  # pragma: no cover  # pragma: no cover
                raise NotImplementedError("Missing in compiler: bitwise_and")
    res = fn(
        *[_u(x) for x in args],
        **{k: _u(v, k) for (k, v) in kwargs.items() if k != "stream"},
    )  # pragma: no cover
    return _w(res)  # pragma: no cover


def bitwise_or(*args, **kwargs):
    """Compute bitwise_or.

    Args:
        args: The args argument.
        kwargs: The kwargs argument.

    Returns:
        The result of bitwise_or.

    """
    from zero_mlx.array import array, _to_tensor  # pragma: no cover
    import ml_switcheroo.ops as mops  # pragma: no cover
    import ml_switcheroo.nn as mnn  # pragma: no cover
    import ml_switcheroo.random as mrand  # pragma: no cover
    import types  # pragma: no cover

    def _u(x, param_name=None):  # pragma: no cover
        if isinstance(x, list):  # pragma: no cover
            if __import__("builtins").all(
                (isinstance(i, int) for i in x)
            ):  # pragma: no cover
                return tuple(x)  # pragma: no cover
            return [_u(i, param_name) for i in x]  # pragma: no cover
        if isinstance(x, tuple):  # pragma: no cover
            return tuple((_u(i, param_name) for i in x))  # pragma: no cover
        if hasattr(x, "_tensor"):  # pragma: no cover
            return x._tensor  # pragma: no cover
        if isinstance(x, (int, float, bool, complex)):  # pragma: no cover
            if param_name not in {
                "axis",
                "rtol",
                "n",
                "m",
                "endpoint",
                "offset",
                "a_max",
                "ddof",
                "dims",
                "source",
                "axis2",
                "stop",
                "k",
                "dtype",
                "reps",
                "a_min",
                "sparse",
                "decimals",
                "strides",
                "num",
                "max",
                "indices_or_sections",
                "shape",
                "atol",
                "axis1",
                "step",
                "axes",
                "fill_value",
                "stream",
                "destination",
                "alpha",
                "beta",
                "start",
                "equal_nan",
                "keepdims",
                "min",
                "p",
            } and "bitwise_or" not in (
                "transpose",
                "moveaxis",
                "reshape",
                "expand_dims",
                "linspace",
                "logcumsumexp",
                "cummin",
                "cummax",
                "cumprod",
                "cumsum",
                "logsumexp",
                "all",
                "any",
                "argmin",
                "argmax",
                "prod",
                "sum",
                "mean",
                "var",
                "std",
                "max",
                "min",
                "tensordot",
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
        fn = getattr(mops, "bitwise_or")  # pragma: no cover
    except AttributeError:  # pragma: no cover
        try:
            fn = getattr(mnn, "bitwise_or")
        except AttributeError:  # pragma: no cover
            try:
                fn = getattr(mrand, "bitwise_or")
            except AttributeError:  # pragma: no cover  # pragma: no cover
                raise NotImplementedError("Missing in compiler: bitwise_or")
    res = fn(
        *[_u(x) for x in args],
        **{k: _u(v, k) for (k, v) in kwargs.items() if k != "stream"},
    )  # pragma: no cover
    return _w(res)  # pragma: no cover


def bitwise_xor(*args, **kwargs):
    """Compute bitwise_xor.

    Args:
        args: The args argument.
        kwargs: The kwargs argument.

    Returns:
        The result of bitwise_xor.

    """
    from zero_mlx.array import array, _to_tensor  # pragma: no cover
    import ml_switcheroo.ops as mops  # pragma: no cover
    import ml_switcheroo.nn as mnn  # pragma: no cover
    import ml_switcheroo.random as mrand  # pragma: no cover
    import types  # pragma: no cover

    def _u(x, param_name=None):  # pragma: no cover
        if isinstance(x, list):  # pragma: no cover
            if __import__("builtins").all(
                (isinstance(i, int) for i in x)
            ):  # pragma: no cover
                return tuple(x)  # pragma: no cover
            return [_u(i, param_name) for i in x]  # pragma: no cover
        if isinstance(x, tuple):  # pragma: no cover
            return tuple((_u(i, param_name) for i in x))  # pragma: no cover
        if hasattr(x, "_tensor"):  # pragma: no cover
            return x._tensor  # pragma: no cover
        if isinstance(x, (int, float, bool, complex)):  # pragma: no cover
            if param_name not in {
                "axis",
                "rtol",
                "n",
                "m",
                "endpoint",
                "offset",
                "a_max",
                "ddof",
                "dims",
                "source",
                "axis2",
                "stop",
                "k",
                "dtype",
                "reps",
                "a_min",
                "sparse",
                "decimals",
                "strides",
                "num",
                "max",
                "indices_or_sections",
                "shape",
                "atol",
                "axis1",
                "step",
                "axes",
                "fill_value",
                "stream",
                "destination",
                "alpha",
                "beta",
                "start",
                "equal_nan",
                "keepdims",
                "min",
                "p",
            } and "bitwise_xor" not in (
                "transpose",
                "moveaxis",
                "reshape",
                "expand_dims",
                "linspace",
                "logcumsumexp",
                "cummin",
                "cummax",
                "cumprod",
                "cumsum",
                "logsumexp",
                "all",
                "any",
                "argmin",
                "argmax",
                "prod",
                "sum",
                "mean",
                "var",
                "std",
                "max",
                "min",
                "tensordot",
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
        fn = getattr(mops, "bitwise_xor")  # pragma: no cover
    except AttributeError:  # pragma: no cover
        try:
            fn = getattr(mnn, "bitwise_xor")
        except AttributeError:  # pragma: no cover
            try:
                fn = getattr(mrand, "bitwise_xor")
            except AttributeError:  # pragma: no cover  # pragma: no cover
                raise NotImplementedError("Missing in compiler: bitwise_xor")
    res = fn(
        *[_u(x) for x in args],
        **{k: _u(v, k) for (k, v) in kwargs.items() if k != "stream"},
    )  # pragma: no cover
    return _w(res)  # pragma: no cover


def left_shift(*args, **kwargs):
    """Compute left_shift.

    Args:
        args: The args argument.
        kwargs: The kwargs argument.

    Returns:
        The result of left_shift.

    """
    from zero_mlx.array import array, _to_tensor  # pragma: no cover
    import ml_switcheroo.ops as mops  # pragma: no cover
    import ml_switcheroo.nn as mnn  # pragma: no cover
    import ml_switcheroo.random as mrand  # pragma: no cover
    import types  # pragma: no cover

    def _u(x, param_name=None):  # pragma: no cover
        if isinstance(x, list):  # pragma: no cover
            if __import__("builtins").all(
                (isinstance(i, int) for i in x)
            ):  # pragma: no cover
                return tuple(x)  # pragma: no cover
            return [_u(i, param_name) for i in x]  # pragma: no cover
        if isinstance(x, tuple):  # pragma: no cover
            return tuple((_u(i, param_name) for i in x))  # pragma: no cover
        if hasattr(x, "_tensor"):  # pragma: no cover
            return x._tensor  # pragma: no cover
        if isinstance(x, (int, float, bool, complex)):  # pragma: no cover
            if param_name not in {
                "axis",
                "rtol",
                "n",
                "m",
                "endpoint",
                "offset",
                "a_max",
                "ddof",
                "dims",
                "source",
                "axis2",
                "stop",
                "k",
                "dtype",
                "reps",
                "a_min",
                "sparse",
                "decimals",
                "strides",
                "num",
                "max",
                "indices_or_sections",
                "shape",
                "atol",
                "axis1",
                "step",
                "axes",
                "fill_value",
                "stream",
                "destination",
                "alpha",
                "beta",
                "start",
                "equal_nan",
                "keepdims",
                "min",
                "p",
            } and "left_shift" not in (
                "transpose",
                "moveaxis",
                "reshape",
                "expand_dims",
                "linspace",
                "logcumsumexp",
                "cummin",
                "cummax",
                "cumprod",
                "cumsum",
                "logsumexp",
                "all",
                "any",
                "argmin",
                "argmax",
                "prod",
                "sum",
                "mean",
                "var",
                "std",
                "max",
                "min",
                "tensordot",
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
        fn = getattr(mops, "left_shift")  # pragma: no cover
    except AttributeError:  # pragma: no cover
        try:
            fn = getattr(mnn, "left_shift")
        except AttributeError:  # pragma: no cover
            try:
                fn = getattr(mrand, "left_shift")
            except AttributeError:  # pragma: no cover  # pragma: no cover
                raise NotImplementedError("Missing in compiler: left_shift")
    res = fn(
        *[_u(x) for x in args],
        **{k: _u(v, k) for (k, v) in kwargs.items() if k != "stream"},
    )  # pragma: no cover
    return _w(res)  # pragma: no cover


def right_shift(*args, **kwargs):
    """Compute right_shift.

    Args:
        args: The args argument.
        kwargs: The kwargs argument.

    Returns:
        The result of right_shift.

    """
    from zero_mlx.array import array, _to_tensor  # pragma: no cover
    import ml_switcheroo.ops as mops  # pragma: no cover
    import ml_switcheroo.nn as mnn  # pragma: no cover
    import ml_switcheroo.random as mrand  # pragma: no cover
    import types  # pragma: no cover

    def _u(x, param_name=None):  # pragma: no cover
        if isinstance(x, list):  # pragma: no cover
            if __import__("builtins").all(
                (isinstance(i, int) for i in x)
            ):  # pragma: no cover
                return tuple(x)  # pragma: no cover
            return [_u(i, param_name) for i in x]  # pragma: no cover
        if isinstance(x, tuple):  # pragma: no cover
            return tuple((_u(i, param_name) for i in x))  # pragma: no cover
        if hasattr(x, "_tensor"):  # pragma: no cover
            return x._tensor  # pragma: no cover
        if isinstance(x, (int, float, bool, complex)):  # pragma: no cover
            if param_name not in {
                "axis",
                "rtol",
                "n",
                "m",
                "endpoint",
                "offset",
                "a_max",
                "ddof",
                "dims",
                "source",
                "axis2",
                "stop",
                "k",
                "dtype",
                "reps",
                "a_min",
                "sparse",
                "decimals",
                "strides",
                "num",
                "max",
                "indices_or_sections",
                "shape",
                "atol",
                "axis1",
                "step",
                "axes",
                "fill_value",
                "stream",
                "destination",
                "alpha",
                "beta",
                "start",
                "equal_nan",
                "keepdims",
                "min",
                "p",
            } and "right_shift" not in (
                "transpose",
                "moveaxis",
                "reshape",
                "expand_dims",
                "linspace",
                "logcumsumexp",
                "cummin",
                "cummax",
                "cumprod",
                "cumsum",
                "logsumexp",
                "all",
                "any",
                "argmin",
                "argmax",
                "prod",
                "sum",
                "mean",
                "var",
                "std",
                "max",
                "min",
                "tensordot",
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
        fn = getattr(mops, "right_shift")  # pragma: no cover
    except AttributeError:  # pragma: no cover
        try:
            fn = getattr(mnn, "right_shift")
        except AttributeError:  # pragma: no cover
            try:
                fn = getattr(mrand, "right_shift")
            except AttributeError:  # pragma: no cover  # pragma: no cover
                raise NotImplementedError("Missing in compiler: right_shift")
    res = fn(
        *[_u(x) for x in args],
        **{k: _u(v, k) for (k, v) in kwargs.items() if k != "stream"},
    )  # pragma: no cover
    return _w(res)  # pragma: no cover


def less(*args, **kwargs):
    """Compute less.

    Args:
        args: The args argument.
        kwargs: The kwargs argument.

    Returns:
        The result of less.

    """
    from zero_mlx.array import array, _to_tensor
    import ml_switcheroo.ops as mops
    import ml_switcheroo.nn as mnn
    import ml_switcheroo.random as mrand
    import types

    def _u(x, param_name=None):
        if isinstance(x, list):  # pragma: no cover
            if __import__("builtins").all(
                (isinstance(i, int) for i in x)
            ):  # pragma: no cover
                return tuple(x)  # pragma: no cover
            return [_u(i, param_name) for i in x]  # pragma: no cover
        if isinstance(x, tuple):  # pragma: no cover
            return tuple((_u(i, param_name) for i in x))  # pragma: no cover
        if hasattr(x, "_tensor"):  # pragma: no cover
            return x._tensor  # pragma: no cover
        if isinstance(x, (int, float, bool, complex)):  # pragma: no cover
            if param_name not in {
                "axis",
                "rtol",
                "n",
                "m",
                "endpoint",
                "offset",
                "a_max",
                "ddof",
                "dims",
                "source",
                "axis2",
                "stop",
                "k",
                "dtype",
                "reps",
                "a_min",
                "sparse",
                "decimals",
                "strides",
                "num",
                "max",
                "indices_or_sections",
                "shape",
                "atol",
                "axis1",
                "step",
                "axes",
                "fill_value",
                "stream",
                "destination",
                "alpha",
                "beta",
                "start",
                "equal_nan",
                "keepdims",
                "min",
                "p",
            } and "less" not in (
                "transpose",
                "moveaxis",
                "reshape",
                "expand_dims",
                "linspace",
                "logcumsumexp",
                "cummin",
                "cummax",
                "cumprod",
                "cumsum",
                "logsumexp",
                "all",
                "any",
                "argmin",
                "argmax",
                "prod",
                "sum",
                "mean",
                "var",
                "std",
                "max",
                "min",
                "tensordot",
            ):  # pragma: no cover
                return _to_tensor(x)  # pragma: no cover
        if type(x).__name__ == "ndarray":  # pragma: no cover
            return _to_tensor(x)  # pragma: no cover
        return x  # pragma: no cover

    def _w(x):
        if isinstance(x, tuple):  # pragma: no cover
            return tuple((_w(i) for i in x))  # pragma: no cover
        if isinstance(x, list):  # pragma: no cover
            return [_w(i) for i in x]  # pragma: no cover
        return array(x) if hasattr(x, "shape") else x  # pragma: no cover

    try:
        fn = getattr(mops, "less")
    except AttributeError:  # pragma: no cover
        try:
            fn = getattr(mnn, "less")
        except AttributeError:  # pragma: no cover
            try:
                fn = getattr(mrand, "less")
            except AttributeError:  # pragma: no cover  # pragma: no cover
                raise NotImplementedError("Missing in compiler: less")
    res = fn(
        *[_u(x) for x in args],
        **{k: _u(v, k) for (k, v) in kwargs.items() if k != "stream"},
    )
    return _w(res)


def less_equal(*args, **kwargs):
    """Compute less_equal.

    Args:
        args: The args argument.
        kwargs: The kwargs argument.

    Returns:
        The result of less_equal.

    """
    from zero_mlx.array import array, _to_tensor
    import ml_switcheroo.ops as mops
    import ml_switcheroo.nn as mnn
    import ml_switcheroo.random as mrand
    import types

    def _u(x, param_name=None):
        if isinstance(x, list):  # pragma: no cover
            if __import__("builtins").all(
                (isinstance(i, int) for i in x)
            ):  # pragma: no cover
                return tuple(x)  # pragma: no cover
            return [_u(i, param_name) for i in x]  # pragma: no cover
        if isinstance(x, tuple):  # pragma: no cover
            return tuple((_u(i, param_name) for i in x))  # pragma: no cover
        if hasattr(x, "_tensor"):  # pragma: no cover
            return x._tensor  # pragma: no cover
        if isinstance(x, (int, float, bool, complex)):  # pragma: no cover
            if param_name not in {
                "axis",
                "rtol",
                "n",
                "m",
                "endpoint",
                "offset",
                "a_max",
                "ddof",
                "dims",
                "source",
                "axis2",
                "stop",
                "k",
                "dtype",
                "reps",
                "a_min",
                "sparse",
                "decimals",
                "strides",
                "num",
                "max",
                "indices_or_sections",
                "shape",
                "atol",
                "axis1",
                "step",
                "axes",
                "fill_value",
                "stream",
                "destination",
                "alpha",
                "beta",
                "start",
                "equal_nan",
                "keepdims",
                "min",
                "p",
            } and "less_equal" not in (
                "transpose",
                "moveaxis",
                "reshape",
                "expand_dims",
                "linspace",
                "logcumsumexp",
                "cummin",
                "cummax",
                "cumprod",
                "cumsum",
                "logsumexp",
                "all",
                "any",
                "argmin",
                "argmax",
                "prod",
                "sum",
                "mean",
                "var",
                "std",
                "max",
                "min",
                "tensordot",
            ):  # pragma: no cover
                return _to_tensor(x)  # pragma: no cover
        if type(x).__name__ == "ndarray":  # pragma: no cover
            return _to_tensor(x)  # pragma: no cover
        return x  # pragma: no cover

    def _w(x):
        if isinstance(x, tuple):  # pragma: no cover
            return tuple((_w(i) for i in x))  # pragma: no cover
        if isinstance(x, list):  # pragma: no cover
            return [_w(i) for i in x]  # pragma: no cover
        return array(x) if hasattr(x, "shape") else x  # pragma: no cover

    try:
        fn = getattr(mops, "less_equal")
    except AttributeError:  # pragma: no cover
        try:
            fn = getattr(mnn, "less_equal")
        except AttributeError:  # pragma: no cover
            try:
                fn = getattr(mrand, "less_equal")
            except AttributeError:  # pragma: no cover  # pragma: no cover
                raise NotImplementedError("Missing in compiler: less_equal")
    res = fn(
        *[_u(x) for x in args],
        **{k: _u(v, k) for (k, v) in kwargs.items() if k != "stream"},
    )
    return _w(res)


def greater(*args, **kwargs):
    """Compute greater.

    Args:
        args: The args argument.
        kwargs: The kwargs argument.

    Returns:
        The result of greater.

    """
    from zero_mlx.array import array, _to_tensor
    import ml_switcheroo.ops as mops
    import ml_switcheroo.nn as mnn
    import ml_switcheroo.random as mrand
    import types

    def _u(x, param_name=None):
        if isinstance(x, list):  # pragma: no cover
            if __import__("builtins").all(
                (isinstance(i, int) for i in x)
            ):  # pragma: no cover
                return tuple(x)  # pragma: no cover
            return [_u(i, param_name) for i in x]  # pragma: no cover
        if isinstance(x, tuple):  # pragma: no cover
            return tuple((_u(i, param_name) for i in x))  # pragma: no cover
        if hasattr(x, "_tensor"):  # pragma: no cover
            return x._tensor  # pragma: no cover
        if isinstance(x, (int, float, bool, complex)):  # pragma: no cover
            if param_name not in {
                "axis",
                "rtol",
                "n",
                "m",
                "endpoint",
                "offset",
                "a_max",
                "ddof",
                "dims",
                "source",
                "axis2",
                "stop",
                "k",
                "dtype",
                "reps",
                "a_min",
                "sparse",
                "decimals",
                "strides",
                "num",
                "max",
                "indices_or_sections",
                "shape",
                "atol",
                "axis1",
                "step",
                "axes",
                "fill_value",
                "stream",
                "destination",
                "alpha",
                "beta",
                "start",
                "equal_nan",
                "keepdims",
                "min",
                "p",
            } and "greater" not in (
                "transpose",
                "moveaxis",
                "reshape",
                "expand_dims",
                "linspace",
                "logcumsumexp",
                "cummin",
                "cummax",
                "cumprod",
                "cumsum",
                "logsumexp",
                "all",
                "any",
                "argmin",
                "argmax",
                "prod",
                "sum",
                "mean",
                "var",
                "std",
                "max",
                "min",
                "tensordot",
            ):  # pragma: no cover
                return _to_tensor(x)  # pragma: no cover
        if type(x).__name__ == "ndarray":  # pragma: no cover
            return _to_tensor(x)  # pragma: no cover
        return x  # pragma: no cover

    def _w(x):
        if isinstance(x, tuple):  # pragma: no cover
            return tuple((_w(i) for i in x))  # pragma: no cover
        if isinstance(x, list):  # pragma: no cover
            return [_w(i) for i in x]  # pragma: no cover
        return array(x) if hasattr(x, "shape") else x  # pragma: no cover

    try:
        fn = getattr(mops, "greater")
    except AttributeError:  # pragma: no cover
        try:
            fn = getattr(mnn, "greater")
        except AttributeError:  # pragma: no cover
            try:
                fn = getattr(mrand, "greater")
            except AttributeError:  # pragma: no cover  # pragma: no cover
                raise NotImplementedError("Missing in compiler: greater")
    res = fn(
        *[_u(x) for x in args],
        **{k: _u(v, k) for (k, v) in kwargs.items() if k != "stream"},
    )
    return _w(res)


def greater_equal(*args, **kwargs):
    """Compute greater_equal.

    Args:
        args: The args argument.
        kwargs: The kwargs argument.

    Returns:
        The result of greater_equal.

    """
    from zero_mlx.array import array, _to_tensor
    import ml_switcheroo.ops as mops
    import ml_switcheroo.nn as mnn
    import ml_switcheroo.random as mrand
    import types

    def _u(x, param_name=None):
        if isinstance(x, list):  # pragma: no cover
            if __import__("builtins").all(
                (isinstance(i, int) for i in x)
            ):  # pragma: no cover
                return tuple(x)  # pragma: no cover
            return [_u(i, param_name) for i in x]  # pragma: no cover
        if isinstance(x, tuple):  # pragma: no cover
            return tuple((_u(i, param_name) for i in x))  # pragma: no cover
        if hasattr(x, "_tensor"):  # pragma: no cover
            return x._tensor  # pragma: no cover
        if isinstance(x, (int, float, bool, complex)):  # pragma: no cover
            if param_name not in {
                "axis",
                "rtol",
                "n",
                "m",
                "endpoint",
                "offset",
                "a_max",
                "ddof",
                "dims",
                "source",
                "axis2",
                "stop",
                "k",
                "dtype",
                "reps",
                "a_min",
                "sparse",
                "decimals",
                "strides",
                "num",
                "max",
                "indices_or_sections",
                "shape",
                "atol",
                "axis1",
                "step",
                "axes",
                "fill_value",
                "stream",
                "destination",
                "alpha",
                "beta",
                "start",
                "equal_nan",
                "keepdims",
                "min",
                "p",
            } and "greater_equal" not in (
                "transpose",
                "moveaxis",
                "reshape",
                "expand_dims",
                "linspace",
                "logcumsumexp",
                "cummin",
                "cummax",
                "cumprod",
                "cumsum",
                "logsumexp",
                "all",
                "any",
                "argmin",
                "argmax",
                "prod",
                "sum",
                "mean",
                "var",
                "std",
                "max",
                "min",
                "tensordot",
            ):  # pragma: no cover
                return _to_tensor(x)  # pragma: no cover
        if type(x).__name__ == "ndarray":  # pragma: no cover
            return _to_tensor(x)  # pragma: no cover
        return x  # pragma: no cover

    def _w(x):
        if isinstance(x, tuple):  # pragma: no cover
            return tuple((_w(i) for i in x))  # pragma: no cover
        if isinstance(x, list):  # pragma: no cover
            return [_w(i) for i in x]  # pragma: no cover
        return array(x) if hasattr(x, "shape") else x  # pragma: no cover

    try:
        fn = getattr(mops, "greater_equal")
    except AttributeError:  # pragma: no cover
        try:
            fn = getattr(mnn, "greater_equal")
        except AttributeError:  # pragma: no cover
            try:
                fn = getattr(mrand, "greater_equal")
            except AttributeError:  # pragma: no cover  # pragma: no cover
                raise NotImplementedError("Missing in compiler: greater_equal")
    res = fn(
        *[_u(x) for x in args],
        **{k: _u(v, k) for (k, v) in kwargs.items() if k != "stream"},
    )
    return _w(res)


def equal(*args, **kwargs):
    """Compute equal.

    Args:
        args: The args argument.
        kwargs: The kwargs argument.

    Returns:
        The result of equal.

    """
    from zero_mlx.array import array, _to_tensor
    import ml_switcheroo.ops as mops
    import ml_switcheroo.nn as mnn
    import ml_switcheroo.random as mrand
    import types

    def _u(x, param_name=None):
        if isinstance(x, list):  # pragma: no cover
            if __import__("builtins").all(
                (isinstance(i, int) for i in x)
            ):  # pragma: no cover
                return tuple(x)  # pragma: no cover
            return [_u(i, param_name) for i in x]  # pragma: no cover
        if isinstance(x, tuple):  # pragma: no cover
            return tuple((_u(i, param_name) for i in x))  # pragma: no cover
        if hasattr(x, "_tensor"):  # pragma: no cover
            return x._tensor  # pragma: no cover
        if isinstance(x, (int, float, bool, complex)):  # pragma: no cover
            if param_name not in {
                "axis",
                "rtol",
                "n",
                "m",
                "endpoint",
                "offset",
                "a_max",
                "ddof",
                "dims",
                "source",
                "axis2",
                "stop",
                "k",
                "dtype",
                "reps",
                "a_min",
                "sparse",
                "decimals",
                "strides",
                "num",
                "max",
                "indices_or_sections",
                "shape",
                "atol",
                "axis1",
                "step",
                "axes",
                "fill_value",
                "stream",
                "destination",
                "alpha",
                "beta",
                "start",
                "equal_nan",
                "keepdims",
                "min",
                "p",
            } and "equal" not in (
                "transpose",
                "moveaxis",
                "reshape",
                "expand_dims",
                "linspace",
                "logcumsumexp",
                "cummin",
                "cummax",
                "cumprod",
                "cumsum",
                "logsumexp",
                "all",
                "any",
                "argmin",
                "argmax",
                "prod",
                "sum",
                "mean",
                "var",
                "std",
                "max",
                "min",
                "tensordot",
            ):  # pragma: no cover
                return _to_tensor(x)  # pragma: no cover
        if type(x).__name__ == "ndarray":  # pragma: no cover
            return _to_tensor(x)  # pragma: no cover
        return x  # pragma: no cover

    def _w(x):
        if isinstance(x, tuple):  # pragma: no cover
            return tuple((_w(i) for i in x))  # pragma: no cover
        if isinstance(x, list):  # pragma: no cover
            return [_w(i) for i in x]  # pragma: no cover
        return array(x) if hasattr(x, "shape") else x  # pragma: no cover

    try:
        fn = getattr(mops, "equal")
    except AttributeError:  # pragma: no cover
        try:
            fn = getattr(mnn, "equal")
        except AttributeError:  # pragma: no cover
            try:
                fn = getattr(mrand, "equal")
            except AttributeError:  # pragma: no cover  # pragma: no cover
                raise NotImplementedError("Missing in compiler: equal")
    res = fn(
        *[_u(x) for x in args],
        **{k: _u(v, k) for (k, v) in kwargs.items() if k != "stream"},
    )
    return _w(res)


def not_equal(*args, **kwargs):
    """Compute not_equal.

    Args:
        args: The args argument.
        kwargs: The kwargs argument.

    Returns:
        The result of not_equal.

    """
    from zero_mlx.array import array, _to_tensor
    import ml_switcheroo.ops as mops
    import ml_switcheroo.nn as mnn
    import ml_switcheroo.random as mrand
    import types

    def _u(x, param_name=None):
        if isinstance(x, list):  # pragma: no cover
            if __import__("builtins").all(
                (isinstance(i, int) for i in x)
            ):  # pragma: no cover
                return tuple(x)  # pragma: no cover
            return [_u(i, param_name) for i in x]  # pragma: no cover
        if isinstance(x, tuple):  # pragma: no cover
            return tuple((_u(i, param_name) for i in x))  # pragma: no cover
        if hasattr(x, "_tensor"):  # pragma: no cover
            return x._tensor  # pragma: no cover
        if isinstance(x, (int, float, bool, complex)):  # pragma: no cover
            if param_name not in {
                "axis",
                "rtol",
                "n",
                "m",
                "endpoint",
                "offset",
                "a_max",
                "ddof",
                "dims",
                "source",
                "axis2",
                "stop",
                "k",
                "dtype",
                "reps",
                "a_min",
                "sparse",
                "decimals",
                "strides",
                "num",
                "max",
                "indices_or_sections",
                "shape",
                "atol",
                "axis1",
                "step",
                "axes",
                "fill_value",
                "stream",
                "destination",
                "alpha",
                "beta",
                "start",
                "equal_nan",
                "keepdims",
                "min",
                "p",
            } and "not_equal" not in (
                "transpose",
                "moveaxis",
                "reshape",
                "expand_dims",
                "linspace",
                "logcumsumexp",
                "cummin",
                "cummax",
                "cumprod",
                "cumsum",
                "logsumexp",
                "all",
                "any",
                "argmin",
                "argmax",
                "prod",
                "sum",
                "mean",
                "var",
                "std",
                "max",
                "min",
                "tensordot",
            ):  # pragma: no cover
                return _to_tensor(x)  # pragma: no cover
        if type(x).__name__ == "ndarray":  # pragma: no cover
            return _to_tensor(x)  # pragma: no cover
        return x  # pragma: no cover

    def _w(x):
        if isinstance(x, tuple):  # pragma: no cover
            return tuple((_w(i) for i in x))  # pragma: no cover
        if isinstance(x, list):  # pragma: no cover
            return [_w(i) for i in x]  # pragma: no cover
        return array(x) if hasattr(x, "shape") else x  # pragma: no cover

    try:
        fn = getattr(mops, "not_equal")
    except AttributeError:  # pragma: no cover
        try:
            fn = getattr(mnn, "not_equal")
        except AttributeError:  # pragma: no cover
            try:
                fn = getattr(mrand, "not_equal")
            except AttributeError:  # pragma: no cover  # pragma: no cover
                raise NotImplementedError("Missing in compiler: not_equal")
    res = fn(
        *[_u(x) for x in args],
        **{k: _u(v, k) for (k, v) in kwargs.items() if k != "stream"},
    )
    return _w(res)


def power(*args, **kwargs):
    """Compute power.

    Args:
        args: The args argument.
        kwargs: The kwargs argument.

    Returns:
        The result of power.

    """
    from zero_mlx.array import array, _to_tensor
    import ml_switcheroo.ops as mops
    import ml_switcheroo.nn as mnn
    import ml_switcheroo.random as mrand
    import types

    def _u(x, param_name=None):
        if isinstance(x, list):  # pragma: no cover
            if __import__("builtins").all(
                (isinstance(i, int) for i in x)
            ):  # pragma: no cover
                return tuple(x)  # pragma: no cover
            return [_u(i, param_name) for i in x]  # pragma: no cover
        if isinstance(x, tuple):  # pragma: no cover
            return tuple((_u(i, param_name) for i in x))  # pragma: no cover
        if hasattr(x, "_tensor"):  # pragma: no cover
            return x._tensor  # pragma: no cover
        if isinstance(x, (int, float, bool, complex)):  # pragma: no cover
            if param_name not in {
                "axis",
                "rtol",
                "n",
                "m",
                "endpoint",
                "offset",
                "a_max",
                "ddof",
                "dims",
                "source",
                "axis2",
                "stop",
                "k",
                "dtype",
                "reps",
                "a_min",
                "sparse",
                "decimals",
                "strides",
                "num",
                "max",
                "indices_or_sections",
                "shape",
                "atol",
                "axis1",
                "step",
                "axes",
                "fill_value",
                "stream",
                "destination",
                "alpha",
                "beta",
                "start",
                "equal_nan",
                "keepdims",
                "min",
                "p",
            } and "power" not in (
                "transpose",
                "moveaxis",
                "reshape",
                "expand_dims",
                "linspace",
                "logcumsumexp",
                "cummin",
                "cummax",
                "cumprod",
                "cumsum",
                "logsumexp",
                "all",
                "any",
                "argmin",
                "argmax",
                "prod",
                "sum",
                "mean",
                "var",
                "std",
                "max",
                "min",
                "tensordot",
            ):  # pragma: no cover
                return _to_tensor(x)  # pragma: no cover
        if type(x).__name__ == "ndarray":  # pragma: no cover
            return _to_tensor(x)  # pragma: no cover
        return x  # pragma: no cover

    def _w(x):
        if isinstance(x, tuple):  # pragma: no cover
            return tuple((_w(i) for i in x))  # pragma: no cover
        if isinstance(x, list):  # pragma: no cover
            return [_w(i) for i in x]  # pragma: no cover
        return array(x) if hasattr(x, "shape") else x  # pragma: no cover

    try:
        fn = getattr(mops, "power")
    except AttributeError:  # pragma: no cover
        try:
            fn = getattr(mnn, "power")
        except AttributeError:  # pragma: no cover
            try:
                fn = getattr(mrand, "power")
            except AttributeError:  # pragma: no cover  # pragma: no cover
                raise NotImplementedError("Missing in compiler: power")
    res = fn(
        *[_u(x) for x in args],
        **{k: _u(v, k) for (k, v) in kwargs.items() if k != "stream"},
    )
    return _w(res)


def sinh(*args, **kwargs):
    """Compute sinh.

    Args:
        args: The args argument.
        kwargs: The kwargs argument.

    Returns:
        The result of sinh.

    """
    from zero_mlx.array import array, _to_tensor  # pragma: no cover
    import ml_switcheroo.ops as mops  # pragma: no cover
    import ml_switcheroo.nn as mnn  # pragma: no cover
    import ml_switcheroo.random as mrand  # pragma: no cover
    import types  # pragma: no cover

    def _u(x, param_name=None):  # pragma: no cover
        if isinstance(x, list):  # pragma: no cover
            if __import__("builtins").all(
                (isinstance(i, int) for i in x)
            ):  # pragma: no cover
                return tuple(x)  # pragma: no cover
            return [_u(i, param_name) for i in x]  # pragma: no cover
        if isinstance(x, tuple):  # pragma: no cover
            return tuple((_u(i, param_name) for i in x))  # pragma: no cover
        if hasattr(x, "_tensor"):  # pragma: no cover
            return x._tensor  # pragma: no cover
        if isinstance(x, (int, float, bool, complex)):  # pragma: no cover
            if param_name not in {
                "axis",
                "rtol",
                "n",
                "m",
                "endpoint",
                "offset",
                "a_max",
                "ddof",
                "dims",
                "source",
                "axis2",
                "stop",
                "k",
                "dtype",
                "reps",
                "a_min",
                "sparse",
                "decimals",
                "strides",
                "num",
                "max",
                "indices_or_sections",
                "shape",
                "atol",
                "axis1",
                "step",
                "axes",
                "fill_value",
                "stream",
                "destination",
                "alpha",
                "beta",
                "start",
                "equal_nan",
                "keepdims",
                "min",
                "p",
            } and "sinh" not in (
                "transpose",
                "moveaxis",
                "reshape",
                "expand_dims",
                "linspace",
                "logcumsumexp",
                "cummin",
                "cummax",
                "cumprod",
                "cumsum",
                "logsumexp",
                "all",
                "any",
                "argmin",
                "argmax",
                "prod",
                "sum",
                "mean",
                "var",
                "std",
                "max",
                "min",
                "tensordot",
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
        fn = getattr(mops, "sinh")  # pragma: no cover
    except AttributeError:  # pragma: no cover
        try:
            fn = getattr(mnn, "sinh")
        except AttributeError:  # pragma: no cover
            try:
                fn = getattr(mrand, "sinh")
            except AttributeError:  # pragma: no cover  # pragma: no cover
                raise NotImplementedError("Missing in compiler: sinh")
    res = fn(
        *[_u(x) for x in args],
        **{k: _u(v, k) for (k, v) in kwargs.items() if k != "stream"},
    )  # pragma: no cover
    return _w(res)  # pragma: no cover


def cosh(*args, **kwargs):
    """Compute cosh.

    Args:
        args: The args argument.
        kwargs: The kwargs argument.

    Returns:
        The result of cosh.

    """
    from zero_mlx.array import array, _to_tensor  # pragma: no cover
    import ml_switcheroo.ops as mops  # pragma: no cover
    import ml_switcheroo.nn as mnn  # pragma: no cover
    import ml_switcheroo.random as mrand  # pragma: no cover
    import types  # pragma: no cover

    def _u(x, param_name=None):  # pragma: no cover
        if isinstance(x, list):  # pragma: no cover
            if __import__("builtins").all(
                (isinstance(i, int) for i in x)
            ):  # pragma: no cover
                return tuple(x)  # pragma: no cover
            return [_u(i, param_name) for i in x]  # pragma: no cover
        if isinstance(x, tuple):  # pragma: no cover
            return tuple((_u(i, param_name) for i in x))  # pragma: no cover
        if hasattr(x, "_tensor"):  # pragma: no cover
            return x._tensor  # pragma: no cover
        if isinstance(x, (int, float, bool, complex)):  # pragma: no cover
            if param_name not in {
                "axis",
                "rtol",
                "n",
                "m",
                "endpoint",
                "offset",
                "a_max",
                "ddof",
                "dims",
                "source",
                "axis2",
                "stop",
                "k",
                "dtype",
                "reps",
                "a_min",
                "sparse",
                "decimals",
                "strides",
                "num",
                "max",
                "indices_or_sections",
                "shape",
                "atol",
                "axis1",
                "step",
                "axes",
                "fill_value",
                "stream",
                "destination",
                "alpha",
                "beta",
                "start",
                "equal_nan",
                "keepdims",
                "min",
                "p",
            } and "cosh" not in (
                "transpose",
                "moveaxis",
                "reshape",
                "expand_dims",
                "linspace",
                "logcumsumexp",
                "cummin",
                "cummax",
                "cumprod",
                "cumsum",
                "logsumexp",
                "all",
                "any",
                "argmin",
                "argmax",
                "prod",
                "sum",
                "mean",
                "var",
                "std",
                "max",
                "min",
                "tensordot",
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
        fn = getattr(mops, "cosh")  # pragma: no cover
    except AttributeError:  # pragma: no cover
        try:
            fn = getattr(mnn, "cosh")
        except AttributeError:  # pragma: no cover
            try:
                fn = getattr(mrand, "cosh")
            except AttributeError:  # pragma: no cover  # pragma: no cover
                raise NotImplementedError("Missing in compiler: cosh")
    res = fn(
        *[_u(x) for x in args],
        **{k: _u(v, k) for (k, v) in kwargs.items() if k != "stream"},
    )  # pragma: no cover
    return _w(res)  # pragma: no cover


def tanh(*args, **kwargs):
    """Compute tanh.

    Args:
        args: The args argument.
        kwargs: The kwargs argument.

    Returns:
        The result of tanh.

    """
    from zero_mlx.array import array, _to_tensor  # pragma: no cover
    import ml_switcheroo.ops as mops  # pragma: no cover
    import ml_switcheroo.nn as mnn  # pragma: no cover
    import ml_switcheroo.random as mrand  # pragma: no cover
    import types  # pragma: no cover

    def _u(x, param_name=None):  # pragma: no cover
        if isinstance(x, list):  # pragma: no cover
            if __import__("builtins").all(
                (isinstance(i, int) for i in x)
            ):  # pragma: no cover
                return tuple(x)  # pragma: no cover
            return [_u(i, param_name) for i in x]  # pragma: no cover
        if isinstance(x, tuple):  # pragma: no cover
            return tuple((_u(i, param_name) for i in x))  # pragma: no cover
        if hasattr(x, "_tensor"):  # pragma: no cover
            return x._tensor  # pragma: no cover
        if isinstance(x, (int, float, bool, complex)):  # pragma: no cover
            if param_name not in {
                "axis",
                "rtol",
                "n",
                "m",
                "endpoint",
                "offset",
                "a_max",
                "ddof",
                "dims",
                "source",
                "axis2",
                "stop",
                "k",
                "dtype",
                "reps",
                "a_min",
                "sparse",
                "decimals",
                "strides",
                "num",
                "max",
                "indices_or_sections",
                "shape",
                "atol",
                "axis1",
                "step",
                "axes",
                "fill_value",
                "stream",
                "destination",
                "alpha",
                "beta",
                "start",
                "equal_nan",
                "keepdims",
                "min",
                "p",
            } and "tanh" not in (
                "transpose",
                "moveaxis",
                "reshape",
                "expand_dims",
                "linspace",
                "logcumsumexp",
                "cummin",
                "cummax",
                "cumprod",
                "cumsum",
                "logsumexp",
                "all",
                "any",
                "argmin",
                "argmax",
                "prod",
                "sum",
                "mean",
                "var",
                "std",
                "max",
                "min",
                "tensordot",
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
        fn = getattr(mops, "tanh")  # pragma: no cover
    except AttributeError:  # pragma: no cover
        try:
            fn = getattr(mnn, "tanh")
        except AttributeError:  # pragma: no cover
            try:
                fn = getattr(mrand, "tanh")
            except AttributeError:  # pragma: no cover  # pragma: no cover
                raise NotImplementedError("Missing in compiler: tanh")
    res = fn(
        *[_u(x) for x in args],
        **{k: _u(v, k) for (k, v) in kwargs.items() if k != "stream"},
    )  # pragma: no cover
    return _w(res)  # pragma: no cover


def floor_divide(*args, **kwargs):
    """Compute floor_divide.

    Args:
        args: The args argument.
        kwargs: The kwargs argument.

    Returns:
        The result of floor_divide.

    """
    from zero_mlx.array import array, _to_tensor
    import ml_switcheroo.ops as mops
    import ml_switcheroo.nn as mnn
    import ml_switcheroo.random as mrand
    import types

    def _u(x, param_name=None):
        if isinstance(x, list):  # pragma: no cover
            if __import__("builtins").all(
                (isinstance(i, int) for i in x)
            ):  # pragma: no cover
                return tuple(x)  # pragma: no cover
            return [_u(i, param_name) for i in x]  # pragma: no cover
        if isinstance(x, tuple):  # pragma: no cover
            return tuple((_u(i, param_name) for i in x))  # pragma: no cover
        if hasattr(x, "_tensor"):  # pragma: no cover
            return x._tensor  # pragma: no cover
        if isinstance(x, (int, float, bool, complex)):  # pragma: no cover
            if param_name not in {
                "axis",
                "rtol",
                "n",
                "m",
                "endpoint",
                "offset",
                "a_max",
                "ddof",
                "dims",
                "source",
                "axis2",
                "stop",
                "k",
                "dtype",
                "reps",
                "a_min",
                "sparse",
                "decimals",
                "strides",
                "num",
                "max",
                "indices_or_sections",
                "shape",
                "atol",
                "axis1",
                "step",
                "axes",
                "fill_value",
                "stream",
                "destination",
                "alpha",
                "beta",
                "start",
                "equal_nan",
                "keepdims",
                "min",
                "p",
            } and "floor_divide" not in (
                "transpose",
                "moveaxis",
                "reshape",
                "expand_dims",
                "linspace",
                "logcumsumexp",
                "cummin",
                "cummax",
                "cumprod",
                "cumsum",
                "logsumexp",
                "all",
                "any",
                "argmin",
                "argmax",
                "prod",
                "sum",
                "mean",
                "var",
                "std",
                "max",
                "min",
                "tensordot",
            ):  # pragma: no cover
                return _to_tensor(x)  # pragma: no cover
        if type(x).__name__ == "ndarray":  # pragma: no cover
            return _to_tensor(x)  # pragma: no cover
        return x  # pragma: no cover

    def _w(x):
        if isinstance(x, tuple):  # pragma: no cover
            return tuple((_w(i) for i in x))  # pragma: no cover
        if isinstance(x, list):  # pragma: no cover
            return [_w(i) for i in x]  # pragma: no cover
        return array(x) if hasattr(x, "shape") else x  # pragma: no cover

    try:
        fn = getattr(mops, "floor_divide")
    except AttributeError:  # pragma: no cover
        try:
            fn = getattr(mnn, "floor_divide")
        except AttributeError:  # pragma: no cover
            try:
                fn = getattr(mrand, "floor_divide")
            except AttributeError:  # pragma: no cover  # pragma: no cover
                raise NotImplementedError("Missing in compiler: floor_divide")
    res = fn(
        *[_u(x) for x in args],
        **{k: _u(v, k) for (k, v) in kwargs.items() if k != "stream"},
    )
    return _w(res)
