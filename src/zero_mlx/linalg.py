"""Linear algebra operations."""

from zero_mlx.array import array


def norm(a, ord=None, axis=None, keepdims=False, stream=None):
    """Compute norm.

    Args:
        a: The a argument.
        ord: The ord argument.
        axis: The axis argument.
        keepdims: The keepdims argument.
        stream: The stream argument.

    Returns:
        The result of norm.

    """
    from zero_mlx.array import _to_tensor
    import ml_switcheroo_compiler.ops as mops
    import ml_switcheroo_compiler.nn as mnn
    import ml_switcheroo_compiler.random as mrand
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
                "compute_uv",
                "upper",
                "lower",
                "k",
                "a_min",
                "offset",
                "beta",
                "destination",
                "axis1",
                "num",
                "endpoint",
                "fill_value",
                "lower",
                "alpha",
                "rtol",
                "axis",
                "equal_nan",
                "upper",
                "ddof",
                "s",
                "source",
                "sparse",
                "shape",
                "d",
                "norm",
                "indices_or_sections",
                "dims",
                "p",
                "stop",
                "dtype",
                "min",
                "atol",
                "ord",
                "axes",
                "axis2",
                "m",
                "reps",
                "stream",
                "a_max",
                "start",
                "seed_val",
                "step",
                "strides",
                "keepdims",
                "decimals",
                "max",
                "n",
            } and "norm" not in (
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
                "norm",
                "cross",
                "solve_triangular",
                "fftfreq",
                "rfftfreq",
                "fftshift",
                "ifftshift",
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
        fn = getattr(mops, "norm")
    except AttributeError:  # pragma: no cover
        try:
            fn = getattr(mnn, "norm")
        except AttributeError:  # pragma: no cover
            try:
                fn = getattr(mrand, "norm")
            except AttributeError:  # pragma: no cover  # pragma: no cover
                raise NotImplementedError("Missing in compiler: norm")
    res = fn(_u(a, "a"), _u(ord, "ord"), _u(axis, "axis"), _u(keepdims, "keepdims"))
    return _w(res)


def qr(a, stream=None):
    """Compute qr.

    Args:
        a: The a argument.
        stream: The stream argument.

    Returns:
        The result of qr.

    """
    from zero_mlx.array import _to_tensor
    import ml_switcheroo_compiler.ops as mops
    import ml_switcheroo_compiler.nn as mnn
    import ml_switcheroo_compiler.random as mrand
    import types
    import zero_mlx as mx

    if getattr(a, "dtype", None) not in (
        mx.float32,
        mx.float16,
        mx.bfloat16,
        getattr(mx, "float64", None),
    ):
        raise ValueError("QR requires float types")

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
                "compute_uv",
                "upper",
                "lower",
                "k",
                "a_min",
                "offset",
                "beta",
                "destination",
                "axis1",
                "num",
                "endpoint",
                "fill_value",
                "lower",
                "alpha",
                "rtol",
                "axis",
                "equal_nan",
                "upper",
                "ddof",
                "s",
                "source",
                "sparse",
                "shape",
                "d",
                "norm",
                "indices_or_sections",
                "dims",
                "p",
                "stop",
                "dtype",
                "min",
                "atol",
                "ord",
                "axes",
                "axis2",
                "m",
                "reps",
                "stream",
                "a_max",
                "start",
                "seed_val",
                "step",
                "strides",
                "keepdims",
                "decimals",
                "max",
                "n",
            } and "qr" not in (
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
                "norm",
                "cross",
                "solve_triangular",
                "fftfreq",
                "rfftfreq",
                "fftshift",
                "ifftshift",
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
        fn = getattr(mops, "qr")
    except AttributeError:  # pragma: no cover
        try:
            fn = getattr(mnn, "qr")
        except AttributeError:  # pragma: no cover
            try:
                fn = getattr(mrand, "qr")
            except AttributeError:  # pragma: no cover  # pragma: no cover
                raise NotImplementedError("Missing in compiler: qr")
    res = fn(_u(a, "a"))
    res_w = _w(res)
    return (
        res_w[0].astype(getattr(a, "dtype", res_w[0].dtype)),
        res_w[1].astype(getattr(a, "dtype", res_w[1].dtype)),
    )


def svd(a, compute_uv=True, stream=None):
    """Compute svd.

    Args:
        a: The a argument.
        compute_uv: The compute_uv argument.
        stream: The stream argument.

    Returns:
        The result of svd.

    """
    from zero_mlx.array import _to_tensor
    import ml_switcheroo_compiler.ops as mops
    import ml_switcheroo_compiler.nn as mnn
    import ml_switcheroo_compiler.random as mrand
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
                "compute_uv",
                "upper",
                "lower",
                "k",
                "a_min",
                "offset",
                "beta",
                "destination",
                "axis1",
                "num",
                "endpoint",
                "fill_value",
                "lower",
                "alpha",
                "rtol",
                "axis",
                "equal_nan",
                "upper",
                "ddof",
                "s",
                "source",
                "sparse",
                "shape",
                "d",
                "norm",
                "indices_or_sections",
                "dims",
                "p",
                "stop",
                "dtype",
                "min",
                "atol",
                "ord",
                "axes",
                "axis2",
                "m",
                "reps",
                "stream",
                "a_max",
                "start",
                "seed_val",
                "step",
                "strides",
                "keepdims",
                "decimals",
                "max",
                "n",
            } and "svd" not in (
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
                "norm",
                "cross",
                "solve_triangular",
                "fftfreq",
                "rfftfreq",
                "fftshift",
                "ifftshift",
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
        fn = getattr(mops, "svd")
    except AttributeError:  # pragma: no cover
        try:
            fn = getattr(mnn, "svd")
        except AttributeError:  # pragma: no cover
            try:
                fn = getattr(mrand, "svd")
            except AttributeError:  # pragma: no cover  # pragma: no cover
                raise NotImplementedError("Missing in compiler: svd")
    res = fn(_u(a, "a"), compute_uv=_u(compute_uv, "compute_uv"))
    return _w(res)


def inv(a, stream=None):
    """Compute inv.

    Args:
        a: The a argument.
        stream: The stream argument.

    Returns:
        The result of inv.

    """
    from zero_mlx.array import _to_tensor
    import ml_switcheroo_compiler.ops as mops
    import ml_switcheroo_compiler.nn as mnn
    import ml_switcheroo_compiler.random as mrand
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
                "compute_uv",
                "upper",
                "lower",
                "k",
                "a_min",
                "offset",
                "beta",
                "destination",
                "axis1",
                "num",
                "endpoint",
                "fill_value",
                "lower",
                "alpha",
                "rtol",
                "axis",
                "equal_nan",
                "upper",
                "ddof",
                "s",
                "source",
                "sparse",
                "shape",
                "d",
                "norm",
                "indices_or_sections",
                "dims",
                "p",
                "stop",
                "dtype",
                "min",
                "atol",
                "ord",
                "axes",
                "axis2",
                "m",
                "reps",
                "stream",
                "a_max",
                "start",
                "seed_val",
                "step",
                "strides",
                "keepdims",
                "decimals",
                "max",
                "n",
            } and "inv" not in (
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
                "norm",
                "cross",
                "solve_triangular",
                "fftfreq",
                "rfftfreq",
                "fftshift",
                "ifftshift",
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
        fn = getattr(mops, "inv")
    except AttributeError:  # pragma: no cover
        try:
            fn = getattr(mnn, "inv")
        except AttributeError:  # pragma: no cover
            try:
                fn = getattr(mrand, "inv")
            except AttributeError:  # pragma: no cover  # pragma: no cover
                raise NotImplementedError("Missing in compiler: inv")
    res = fn(_u(a, "a"))
    return _w(res)


def tri_inv(a, upper=False, stream=None):
    """Compute tri_inv.

    Args:
        a: The a argument.
        upper: The upper argument.
        stream: The stream argument.

    Returns:
        The result of tri_inv.

    """
    from zero_mlx.array import _to_tensor
    import ml_switcheroo_compiler.ops as mops
    import ml_switcheroo_compiler.nn as mnn
    import ml_switcheroo_compiler.random as mrand
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
                "compute_uv",
                "upper",
                "lower",
                "k",
                "a_min",
                "offset",
                "beta",
                "destination",
                "axis1",
                "num",
                "endpoint",
                "fill_value",
                "lower",
                "alpha",
                "rtol",
                "axis",
                "equal_nan",
                "upper",
                "ddof",
                "s",
                "source",
                "sparse",
                "shape",
                "d",
                "norm",
                "indices_or_sections",
                "dims",
                "p",
                "stop",
                "dtype",
                "min",
                "atol",
                "ord",
                "axes",
                "axis2",
                "m",
                "reps",
                "stream",
                "a_max",
                "start",
                "seed_val",
                "step",
                "strides",
                "keepdims",
                "decimals",
                "max",
                "n",
            } and "tri_inv" not in (
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
                "norm",
                "cross",
                "solve_triangular",
                "fftfreq",
                "rfftfreq",
                "fftshift",
                "ifftshift",
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
        fn = getattr(mops, "tri_inv")
    except AttributeError:  # pragma: no cover
        try:
            fn = getattr(mnn, "tri_inv")
        except AttributeError:  # pragma: no cover
            try:
                fn = getattr(mrand, "tri_inv")
            except AttributeError:  # pragma: no cover  # pragma: no cover
                raise NotImplementedError("Missing in compiler: tri_inv")
    res = fn(_u(a, "a"), _u(upper, "upper"))
    return _w(res)


def cholesky(a, upper=False, stream=None):
    """Compute cholesky.

    Args:
        a: The a argument.
        upper: The upper argument.
        stream: The stream argument.

    Returns:
        The result of cholesky.

    """
    from zero_mlx.array import _to_tensor  # pragma: no cover
    import ml_switcheroo_compiler.ops as mops  # pragma: no cover
    import ml_switcheroo_compiler.nn as mnn  # pragma: no cover
    import ml_switcheroo_compiler.random as mrand  # pragma: no cover
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
                "compute_uv",
                "upper",
                "lower",
                "k",
                "a_min",
                "offset",
                "beta",
                "destination",
                "axis1",
                "num",
                "endpoint",
                "fill_value",
                "lower",
                "alpha",
                "rtol",
                "axis",
                "equal_nan",
                "upper",
                "ddof",
                "s",
                "source",
                "sparse",
                "shape",
                "d",
                "norm",
                "indices_or_sections",
                "dims",
                "p",
                "stop",
                "dtype",
                "min",
                "atol",
                "ord",
                "axes",
                "axis2",
                "m",
                "reps",
                "stream",
                "a_max",
                "start",
                "seed_val",
                "step",
                "strides",
                "keepdims",
                "decimals",
                "max",
                "n",
            } and "cholesky" not in (
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
                "norm",
                "cross",
                "solve_triangular",
                "fftfreq",
                "rfftfreq",
                "fftshift",
                "ifftshift",
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
        fn = getattr(mops, "cholesky")  # pragma: no cover
    except AttributeError:  # pragma: no cover
        try:
            fn = getattr(mnn, "cholesky")
        except AttributeError:  # pragma: no cover
            try:
                fn = getattr(mrand, "cholesky")
            except AttributeError:  # pragma: no cover  # pragma: no cover
                raise NotImplementedError("Missing in compiler: cholesky")
    res = fn(_u(a, "a"), _u(upper, "upper"))  # pragma: no cover
    return _w(res)  # pragma: no cover


def cholesky_inv(a, upper=False, stream=None):
    """Compute cholesky_inv.

    Args:
        a: The a argument.
        upper: The upper argument.
        stream: The stream argument.

    Returns:
        The result of cholesky_inv.

    """
    from zero_mlx.array import _to_tensor  # pragma: no cover
    import ml_switcheroo_compiler.ops as mops  # pragma: no cover
    import ml_switcheroo_compiler.nn as mnn  # pragma: no cover
    import ml_switcheroo_compiler.random as mrand  # pragma: no cover
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
                "compute_uv",
                "upper",
                "lower",
                "k",
                "a_min",
                "offset",
                "beta",
                "destination",
                "axis1",
                "num",
                "endpoint",
                "fill_value",
                "lower",
                "alpha",
                "rtol",
                "axis",
                "equal_nan",
                "upper",
                "ddof",
                "s",
                "source",
                "sparse",
                "shape",
                "d",
                "norm",
                "indices_or_sections",
                "dims",
                "p",
                "stop",
                "dtype",
                "min",
                "atol",
                "ord",
                "axes",
                "axis2",
                "m",
                "reps",
                "stream",
                "a_max",
                "start",
                "seed_val",
                "step",
                "strides",
                "keepdims",
                "decimals",
                "max",
                "n",
            } and "cholesky_inv" not in (
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
                "norm",
                "cross",
                "solve_triangular",
                "fftfreq",
                "rfftfreq",
                "fftshift",
                "ifftshift",
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
        fn = getattr(mops, "cholesky_inv")  # pragma: no cover
    except AttributeError:  # pragma: no cover
        try:
            fn = getattr(mnn, "cholesky_inv")
        except AttributeError:  # pragma: no cover
            try:
                fn = getattr(mrand, "cholesky_inv")
            except AttributeError:  # pragma: no cover  # pragma: no cover
                raise NotImplementedError("Missing in compiler: cholesky_inv")
    res = fn(_u(a, "a"), _u(upper, "upper"))  # pragma: no cover
    return _w(res)  # pragma: no cover


def pinv(a, stream=None):
    """Compute pinv.

    Args:
        a: The a argument.
        stream: The stream argument.

    Returns:
        The result of pinv.

    """
    from zero_mlx.array import _to_tensor
    import ml_switcheroo_compiler.ops as mops
    import ml_switcheroo_compiler.nn as mnn
    import ml_switcheroo_compiler.random as mrand
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
                "compute_uv",
                "upper",
                "lower",
                "k",
                "a_min",
                "offset",
                "beta",
                "destination",
                "axis1",
                "num",
                "endpoint",
                "fill_value",
                "lower",
                "alpha",
                "rtol",
                "axis",
                "equal_nan",
                "upper",
                "ddof",
                "s",
                "source",
                "sparse",
                "shape",
                "d",
                "norm",
                "indices_or_sections",
                "dims",
                "p",
                "stop",
                "dtype",
                "min",
                "atol",
                "ord",
                "axes",
                "axis2",
                "m",
                "reps",
                "stream",
                "a_max",
                "start",
                "seed_val",
                "step",
                "strides",
                "keepdims",
                "decimals",
                "max",
                "n",
            } and "pinv" not in (
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
                "norm",
                "cross",
                "solve_triangular",
                "fftfreq",
                "rfftfreq",
                "fftshift",
                "ifftshift",
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
        fn = getattr(mops, "pinv")
    except AttributeError:  # pragma: no cover
        try:
            fn = getattr(mnn, "pinv")
        except AttributeError:  # pragma: no cover
            try:
                fn = getattr(mrand, "pinv")
            except AttributeError:  # pragma: no cover  # pragma: no cover
                raise NotImplementedError("Missing in compiler: pinv")
    res = fn(_u(a, "a"))
    return _w(res)


def cross(a, b, axis=-1, stream=None):
    """Compute cross.

    Args:
        a: The a argument.
        b: The b argument.
        axis: The axis argument.
        stream: The stream argument.

    Returns:
        The result of cross.

    """
    from zero_mlx.array import _to_tensor
    import ml_switcheroo_compiler.ops as mops
    import ml_switcheroo_compiler.nn as mnn
    import ml_switcheroo_compiler.random as mrand
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
                "compute_uv",
                "upper",
                "lower",
                "k",
                "a_min",
                "offset",
                "beta",
                "destination",
                "axis1",
                "num",
                "endpoint",
                "fill_value",
                "lower",
                "alpha",
                "rtol",
                "axis",
                "equal_nan",
                "upper",
                "ddof",
                "s",
                "source",
                "sparse",
                "shape",
                "d",
                "norm",
                "indices_or_sections",
                "dims",
                "p",
                "stop",
                "dtype",
                "min",
                "atol",
                "ord",
                "axes",
                "axis2",
                "m",
                "reps",
                "stream",
                "a_max",
                "start",
                "seed_val",
                "step",
                "strides",
                "keepdims",
                "decimals",
                "max",
                "n",
            } and "cross" not in (
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
                "norm",
                "cross",
                "solve_triangular",
                "fftfreq",
                "rfftfreq",
                "fftshift",
                "ifftshift",
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
        fn = getattr(mops, "cross")
    except AttributeError:  # pragma: no cover
        try:
            fn = getattr(mnn, "cross")
        except AttributeError:  # pragma: no cover
            try:
                fn = getattr(mrand, "cross")
            except AttributeError:  # pragma: no cover  # pragma: no cover
                raise NotImplementedError("Missing in compiler: cross")
    res = fn(_u(a, "a"), _u(b, "b"), _u(axis, "axis"))
    return _w(res)


def eig(a, stream=None):
    """Compute eig.

    Args:
        a: The a argument.
        stream: The stream argument.

    Returns:
        The result of eig.

    """
    from zero_mlx.array import _to_tensor  # pragma: no cover
    import ml_switcheroo_compiler.ops as mops  # pragma: no cover
    import ml_switcheroo_compiler.nn as mnn  # pragma: no cover
    import ml_switcheroo_compiler.random as mrand  # pragma: no cover
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
                "compute_uv",
                "upper",
                "lower",
                "k",
                "a_min",
                "offset",
                "beta",
                "destination",
                "axis1",
                "num",
                "endpoint",
                "fill_value",
                "lower",
                "alpha",
                "rtol",
                "axis",
                "equal_nan",
                "upper",
                "ddof",
                "s",
                "source",
                "sparse",
                "shape",
                "d",
                "norm",
                "indices_or_sections",
                "dims",
                "p",
                "stop",
                "dtype",
                "min",
                "atol",
                "ord",
                "axes",
                "axis2",
                "m",
                "reps",
                "stream",
                "a_max",
                "start",
                "seed_val",
                "step",
                "strides",
                "keepdims",
                "decimals",
                "max",
                "n",
            } and "eig" not in (
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
                "norm",
                "cross",
                "solve_triangular",
                "fftfreq",
                "rfftfreq",
                "fftshift",
                "ifftshift",
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
        fn = getattr(mops, "eig")  # pragma: no cover
    except AttributeError:  # pragma: no cover
        try:
            fn = getattr(mnn, "eig")
        except AttributeError:  # pragma: no cover
            try:
                fn = getattr(mrand, "eig")
            except AttributeError:  # pragma: no cover  # pragma: no cover
                raise NotImplementedError("Missing in compiler: eig")
    res = fn(_u(a, "a"))  # pragma: no cover
    return _w(res)  # pragma: no cover


def eigvals(a, stream=None):
    """Compute eigvals.

    Args:
        a: The a argument.
        stream: The stream argument.

    Returns:
        The result of eigvals.

    """
    from zero_mlx.array import _to_tensor  # pragma: no cover
    import ml_switcheroo_compiler.ops as mops  # pragma: no cover
    import ml_switcheroo_compiler.nn as mnn  # pragma: no cover
    import ml_switcheroo_compiler.random as mrand  # pragma: no cover
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
                "compute_uv",
                "upper",
                "lower",
                "k",
                "a_min",
                "offset",
                "beta",
                "destination",
                "axis1",
                "num",
                "endpoint",
                "fill_value",
                "lower",
                "alpha",
                "rtol",
                "axis",
                "equal_nan",
                "upper",
                "ddof",
                "s",
                "source",
                "sparse",
                "shape",
                "d",
                "norm",
                "indices_or_sections",
                "dims",
                "p",
                "stop",
                "dtype",
                "min",
                "atol",
                "ord",
                "axes",
                "axis2",
                "m",
                "reps",
                "stream",
                "a_max",
                "start",
                "seed_val",
                "step",
                "strides",
                "keepdims",
                "decimals",
                "max",
                "n",
            } and "eigvals" not in (
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
                "norm",
                "cross",
                "solve_triangular",
                "fftfreq",
                "rfftfreq",
                "fftshift",
                "ifftshift",
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
        fn = getattr(mops, "eigvals")  # pragma: no cover
    except AttributeError:  # pragma: no cover
        try:
            fn = getattr(mnn, "eigvals")
        except AttributeError:  # pragma: no cover
            try:
                fn = getattr(mrand, "eigvals")
            except AttributeError:  # pragma: no cover  # pragma: no cover
                raise NotImplementedError("Missing in compiler: eigvals")
    res = fn(_u(a, "a"))  # pragma: no cover
    return _w(res)  # pragma: no cover


def eigh(a, UPLO="L", stream=None):
    """Compute eigh.

    Args:
        a: The a argument.
        UPLO: The UPLO argument.
        stream: The stream argument.

    Returns:
        The result of eigh.

    """
    from zero_mlx.array import _to_tensor  # pragma: no cover
    import ml_switcheroo_compiler.ops as mops  # pragma: no cover
    import ml_switcheroo_compiler.nn as mnn  # pragma: no cover
    import ml_switcheroo_compiler.random as mrand  # pragma: no cover
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
                "compute_uv",
                "upper",
                "lower",
                "k",
                "a_min",
                "offset",
                "beta",
                "destination",
                "axis1",
                "num",
                "endpoint",
                "fill_value",
                "lower",
                "alpha",
                "rtol",
                "axis",
                "equal_nan",
                "upper",
                "ddof",
                "s",
                "source",
                "sparse",
                "shape",
                "d",
                "norm",
                "indices_or_sections",
                "dims",
                "p",
                "stop",
                "dtype",
                "min",
                "atol",
                "ord",
                "axes",
                "axis2",
                "m",
                "reps",
                "stream",
                "a_max",
                "start",
                "seed_val",
                "step",
                "strides",
                "keepdims",
                "decimals",
                "max",
                "n",
            } and "eigh" not in (
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
                "norm",
                "cross",
                "solve_triangular",
                "fftfreq",
                "rfftfreq",
                "fftshift",
                "ifftshift",
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
        fn = getattr(mops, "eigh")  # pragma: no cover
    except AttributeError:  # pragma: no cover
        try:
            fn = getattr(mnn, "eigh")
        except AttributeError:  # pragma: no cover
            try:
                fn = getattr(mrand, "eigh")
            except AttributeError:  # pragma: no cover  # pragma: no cover
                raise NotImplementedError("Missing in compiler: eigh")
    res = fn(_u(a, "a"), _u(UPLO, "UPLO"))  # pragma: no cover
    return _w(res)  # pragma: no cover


def eigvalsh(a, UPLO="L", stream=None):
    """Compute eigvalsh.

    Args:
        a: The a argument.
        UPLO: The UPLO argument.
        stream: The stream argument.

    Returns:
        The result of eigvalsh.

    """
    from zero_mlx.array import _to_tensor  # pragma: no cover
    import ml_switcheroo_compiler.ops as mops  # pragma: no cover
    import ml_switcheroo_compiler.nn as mnn  # pragma: no cover
    import ml_switcheroo_compiler.random as mrand  # pragma: no cover
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
                "compute_uv",
                "upper",
                "lower",
                "k",
                "a_min",
                "offset",
                "beta",
                "destination",
                "axis1",
                "num",
                "endpoint",
                "fill_value",
                "lower",
                "alpha",
                "rtol",
                "axis",
                "equal_nan",
                "upper",
                "ddof",
                "s",
                "source",
                "sparse",
                "shape",
                "d",
                "norm",
                "indices_or_sections",
                "dims",
                "p",
                "stop",
                "dtype",
                "min",
                "atol",
                "ord",
                "axes",
                "axis2",
                "m",
                "reps",
                "stream",
                "a_max",
                "start",
                "seed_val",
                "step",
                "strides",
                "keepdims",
                "decimals",
                "max",
                "n",
            } and "eigvalsh" not in (
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
                "norm",
                "cross",
                "solve_triangular",
                "fftfreq",
                "rfftfreq",
                "fftshift",
                "ifftshift",
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
        fn = getattr(mops, "eigvalsh")  # pragma: no cover
    except AttributeError:  # pragma: no cover
        try:
            fn = getattr(mnn, "eigvalsh")
        except AttributeError:  # pragma: no cover
            try:
                fn = getattr(mrand, "eigvalsh")
            except AttributeError:  # pragma: no cover  # pragma: no cover
                raise NotImplementedError("Missing in compiler: eigvalsh")
    res = fn(_u(a, "a"), _u(UPLO, "UPLO"))  # pragma: no cover
    return _w(res)  # pragma: no cover


def lu(a, stream=None):
    """Compute lu.

    Args:
        a: The a argument.
        stream: The stream argument.

    Returns:
        The result of lu.

    """
    from zero_mlx.array import _to_tensor
    import ml_switcheroo_compiler.ops as mops
    import ml_switcheroo_compiler.nn as mnn
    import ml_switcheroo_compiler.random as mrand
    import types
    import zero_mlx as mx

    if getattr(a, "dtype", None) not in (
        mx.float32,
        mx.float16,
        mx.bfloat16,
        getattr(mx, "float64", None),
    ):
        raise ValueError("LU requires float types")

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
                "compute_uv",
                "upper",
                "lower",
                "k",
                "a_min",
                "offset",
                "beta",
                "destination",
                "axis1",
                "num",
                "endpoint",
                "fill_value",
                "lower",
                "alpha",
                "rtol",
                "axis",
                "equal_nan",
                "upper",
                "ddof",
                "s",
                "source",
                "sparse",
                "shape",
                "d",
                "norm",
                "indices_or_sections",
                "dims",
                "p",
                "stop",
                "dtype",
                "min",
                "atol",
                "ord",
                "axes",
                "axis2",
                "m",
                "reps",
                "stream",
                "a_max",
                "start",
                "seed_val",
                "step",
                "strides",
                "keepdims",
                "decimals",
                "max",
                "n",
            } and "lu" not in (
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
                "norm",
                "cross",
                "solve_triangular",
                "fftfreq",
                "rfftfreq",
                "fftshift",
                "ifftshift",
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
        fn = getattr(mops, "lu")
    except AttributeError:  # pragma: no cover
        try:
            fn = getattr(mnn, "lu")
        except AttributeError:  # pragma: no cover
            try:
                fn = getattr(mrand, "lu")
            except AttributeError:  # pragma: no cover  # pragma: no cover
                raise NotImplementedError("Missing in compiler: lu")
    res = fn(_u(a, "a"))
    return _w(res)


def lu_factor(a, stream=None):
    """Compute lu_factor.

    Args:
        a: The a argument.
        stream: The stream argument.

    Returns:
        The result of lu_factor.

    """
    from zero_mlx.array import _to_tensor
    import ml_switcheroo_compiler.ops as mops
    import ml_switcheroo_compiler.nn as mnn
    import ml_switcheroo_compiler.random as mrand
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
                "compute_uv",
                "upper",
                "lower",
                "k",
                "a_min",
                "offset",
                "beta",
                "destination",
                "axis1",
                "num",
                "endpoint",
                "fill_value",
                "lower",
                "alpha",
                "rtol",
                "axis",
                "equal_nan",
                "upper",
                "ddof",
                "s",
                "source",
                "sparse",
                "shape",
                "d",
                "norm",
                "indices_or_sections",
                "dims",
                "p",
                "stop",
                "dtype",
                "min",
                "atol",
                "ord",
                "axes",
                "axis2",
                "m",
                "reps",
                "stream",
                "a_max",
                "start",
                "seed_val",
                "step",
                "strides",
                "keepdims",
                "decimals",
                "max",
                "n",
            } and "lu_factor" not in (
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
                "norm",
                "cross",
                "solve_triangular",
                "fftfreq",
                "rfftfreq",
                "fftshift",
                "ifftshift",
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
        fn = getattr(mops, "lu_factor")
    except AttributeError:  # pragma: no cover
        try:
            fn = getattr(mnn, "lu_factor")
        except AttributeError:  # pragma: no cover
            try:
                fn = getattr(mrand, "lu_factor")
            except AttributeError:  # pragma: no cover  # pragma: no cover
                raise NotImplementedError("Missing in compiler: lu_factor")
    res = fn(_u(a, "a"))
    return _w(res)


def lu_solve(lu_and_piv, b, stream=None):
    """Compute lu_solve.

    Args:
        lu_and_piv: The lu_and_piv argument.
        b: The b argument.
        stream: The stream argument.

    Returns:
        The result of lu_solve.

    """
    from zero_mlx.array import _to_tensor  # pragma: no cover
    import ml_switcheroo_compiler.ops as mops  # pragma: no cover
    import ml_switcheroo_compiler.nn as mnn  # pragma: no cover
    import ml_switcheroo_compiler.random as mrand  # pragma: no cover
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
                "compute_uv",
                "upper",
                "lower",
                "k",
                "a_min",
                "offset",
                "beta",
                "destination",
                "axis1",
                "num",
                "endpoint",
                "fill_value",
                "lower",
                "alpha",
                "rtol",
                "axis",
                "equal_nan",
                "upper",
                "ddof",
                "s",
                "source",
                "sparse",
                "shape",
                "d",
                "norm",
                "indices_or_sections",
                "dims",
                "p",
                "stop",
                "dtype",
                "min",
                "atol",
                "ord",
                "axes",
                "axis2",
                "m",
                "reps",
                "stream",
                "a_max",
                "start",
                "seed_val",
                "step",
                "strides",
                "keepdims",
                "decimals",
                "max",
                "n",
            } and "lu_solve" not in (
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
                "norm",
                "cross",
                "solve_triangular",
                "fftfreq",
                "rfftfreq",
                "fftshift",
                "ifftshift",
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
        fn = getattr(mops, "lu_solve")  # pragma: no cover
    except AttributeError:  # pragma: no cover
        try:
            fn = getattr(mnn, "lu_solve")
        except AttributeError:  # pragma: no cover
            try:
                fn = getattr(mrand, "lu_solve")
            except AttributeError:  # pragma: no cover  # pragma: no cover
                raise NotImplementedError("Missing in compiler: lu_solve")
    res = fn(_u(lu_and_piv, "lu_and_piv"), _u(b, "b"))  # pragma: no cover
    return _w(res)  # pragma: no cover


def solve(a, b, stream=None):
    """Compute solve.

    Args:
        a: The a argument.
        b: The b argument.
        stream: The stream argument.

    Returns:
        The result of solve.

    """
    from zero_mlx.array import _to_tensor
    import ml_switcheroo_compiler.ops as mops
    import ml_switcheroo_compiler.nn as mnn
    import ml_switcheroo_compiler.random as mrand
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
                "compute_uv",
                "upper",
                "lower",
                "k",
                "a_min",
                "offset",
                "beta",
                "destination",
                "axis1",
                "num",
                "endpoint",
                "fill_value",
                "lower",
                "alpha",
                "rtol",
                "axis",
                "equal_nan",
                "upper",
                "ddof",
                "s",
                "source",
                "sparse",
                "shape",
                "d",
                "norm",
                "indices_or_sections",
                "dims",
                "p",
                "stop",
                "dtype",
                "min",
                "atol",
                "ord",
                "axes",
                "axis2",
                "m",
                "reps",
                "stream",
                "a_max",
                "start",
                "seed_val",
                "step",
                "strides",
                "keepdims",
                "decimals",
                "max",
                "n",
            } and "solve" not in (
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
                "norm",
                "cross",
                "solve_triangular",
                "fftfreq",
                "rfftfreq",
                "fftshift",
                "ifftshift",
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
        fn = getattr(mops, "solve")
    except AttributeError:  # pragma: no cover
        try:
            fn = getattr(mnn, "solve")
        except AttributeError:  # pragma: no cover
            try:
                fn = getattr(mrand, "solve")
            except AttributeError:  # pragma: no cover  # pragma: no cover
                raise NotImplementedError("Missing in compiler: solve")
    res = fn(_u(a, "a"), _u(b, "b"))
    return _w(res)


def solve_triangular(a, b, upper=False, stream=None):
    """Compute solve_triangular.

    Args:
        a: The a argument.
        b: The b argument.
        upper: The upper argument.
        stream: The stream argument.

    Returns:
        The result of solve_triangular.

    """
    from zero_mlx.array import _to_tensor
    import ml_switcheroo_compiler.ops as mops
    import ml_switcheroo_compiler.nn as mnn
    import ml_switcheroo_compiler.random as mrand
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
                "compute_uv",
                "upper",
                "lower",
                "k",
                "a_min",
                "offset",
                "beta",
                "destination",
                "axis1",
                "num",
                "endpoint",
                "fill_value",
                "lower",
                "alpha",
                "rtol",
                "axis",
                "equal_nan",
                "upper",
                "ddof",
                "s",
                "source",
                "sparse",
                "shape",
                "d",
                "norm",
                "indices_or_sections",
                "dims",
                "p",
                "stop",
                "dtype",
                "min",
                "atol",
                "ord",
                "axes",
                "axis2",
                "m",
                "reps",
                "stream",
                "a_max",
                "start",
                "seed_val",
                "step",
                "strides",
                "keepdims",
                "decimals",
                "max",
                "n",
            } and "solve_triangular" not in (
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
                "norm",
                "cross",
                "solve_triangular",
                "fftfreq",
                "rfftfreq",
                "fftshift",
                "ifftshift",
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
        fn = getattr(mops, "solve_triangular")
    except AttributeError:  # pragma: no cover
        try:
            fn = getattr(mnn, "solve_triangular")
        except AttributeError:  # pragma: no cover
            try:
                fn = getattr(mrand, "solve_triangular")
            except AttributeError:  # pragma: no cover  # pragma: no cover
                raise NotImplementedError("Missing in compiler: solve_triangular")
    res = fn(_u(a, "a"), _u(b, "b"), _u(upper, "upper"))
    return _w(res)


def det(a, stream=None):
    """Compute det.

    Args:
        a: The a argument.
        stream: The stream argument.

    Returns:
        The result of det.

    """
    from zero_mlx.array import _to_tensor
    import ml_switcheroo_compiler.ops as mops
    import ml_switcheroo_compiler.nn as mnn
    import ml_switcheroo_compiler.random as mrand
    import types
    import zero_mlx as mx

    if getattr(a, "dtype", None) in (mx.complex64, getattr(mx, "complex128", None)):
        raise ValueError("complex unsupported")

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
                "compute_uv",
                "upper",
                "lower",
                "k",
                "a_min",
                "offset",
                "beta",
                "destination",
                "axis1",
                "num",
                "endpoint",
                "fill_value",
                "lower",
                "alpha",
                "rtol",
                "axis",
                "equal_nan",
                "upper",
                "ddof",
                "s",
                "source",
                "sparse",
                "shape",
                "d",
                "norm",
                "indices_or_sections",
                "dims",
                "p",
                "stop",
                "dtype",
                "min",
                "atol",
                "ord",
                "axes",
                "axis2",
                "m",
                "reps",
                "stream",
                "a_max",
                "start",
                "seed_val",
                "step",
                "strides",
                "keepdims",
                "decimals",
                "max",
                "n",
            } and "det" not in (
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
                "norm",
                "cross",
                "solve_triangular",
                "fftfreq",
                "rfftfreq",
                "fftshift",
                "ifftshift",
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
        fn = getattr(mops, "det")
    except AttributeError:  # pragma: no cover
        try:
            fn = getattr(mnn, "det")
        except AttributeError:  # pragma: no cover
            try:
                fn = getattr(mrand, "det")
            except AttributeError:  # pragma: no cover  # pragma: no cover
                raise NotImplementedError("Missing in compiler: det")
    res = fn(_u(a, "a"))
    return _w(res)


def slogdet(a, stream=None):
    """Compute slogdet.

    Args:
        a: The a argument.
        stream: The stream argument.

    Returns:
        The result of slogdet.

    """
    from zero_mlx.array import _to_tensor
    import ml_switcheroo_compiler.ops as mops
    import ml_switcheroo_compiler.nn as mnn
    import ml_switcheroo_compiler.random as mrand
    import types
    import zero_mlx as mx

    if getattr(a, "dtype", None) in (mx.complex64, getattr(mx, "complex128", None)):
        raise ValueError("complex unsupported")

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
                "compute_uv",
                "upper",
                "lower",
                "k",
                "a_min",
                "offset",
                "beta",
                "destination",
                "axis1",
                "num",
                "endpoint",
                "fill_value",
                "lower",
                "alpha",
                "rtol",
                "axis",
                "equal_nan",
                "upper",
                "ddof",
                "s",
                "source",
                "sparse",
                "shape",
                "d",
                "norm",
                "indices_or_sections",
                "dims",
                "p",
                "stop",
                "dtype",
                "min",
                "atol",
                "ord",
                "axes",
                "axis2",
                "m",
                "reps",
                "stream",
                "a_max",
                "start",
                "seed_val",
                "step",
                "strides",
                "keepdims",
                "decimals",
                "max",
                "n",
            } and "slogdet" not in (
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
                "norm",
                "cross",
                "solve_triangular",
                "fftfreq",
                "rfftfreq",
                "fftshift",
                "ifftshift",
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
        fn = getattr(mops, "slogdet")
    except AttributeError:  # pragma: no cover
        try:
            fn = getattr(mnn, "slogdet")
        except AttributeError:  # pragma: no cover
            try:
                fn = getattr(mrand, "slogdet")
            except AttributeError:  # pragma: no cover  # pragma: no cover
                raise NotImplementedError("Missing in compiler: slogdet")
    res = fn(_u(a, "a"))
    return _w(res)
