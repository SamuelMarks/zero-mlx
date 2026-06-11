"""Fast Fourier Transform operations."""

from zero_mlx.array import array


def fft(a, n=None, axis=-1, norm=None, stream=None):  # pragma: no cover
    """Compute fft.

    Args:
        a: The a argument.
        n: The n argument.
        axis: The axis argument.
        norm: The norm argument.
        stream: The stream argument.

    Returns:
        The result of fft.

    """
    from zero_mlx.array import _to_tensor
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
            } and "fft" not in (
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
        fn = getattr(mops, "fft")
    except AttributeError:  # pragma: no cover
        try:
            fn = getattr(mnn, "fft")
        except AttributeError:  # pragma: no cover
            try:
                fn = getattr(mrand, "fft")
            except AttributeError:  # pragma: no cover  # pragma: no cover
                raise NotImplementedError("Missing in compiler: fft")
    res = fn(_u(a, "a"), _u(n, "n"), _u(axis, "axis"), _u(norm, "norm"))
    return _w(res)


def ifft(a, n=None, axis=-1, norm=None, stream=None):  # pragma: no cover
    """Compute ifft.

    Args:
        a: The a argument.
        n: The n argument.
        axis: The axis argument.
        norm: The norm argument.
        stream: The stream argument.

    Returns:
        The result of ifft.

    """
    from zero_mlx.array import _to_tensor
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
            } and "ifft" not in (
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
        fn = getattr(mops, "ifft")
    except AttributeError:  # pragma: no cover
        try:
            fn = getattr(mnn, "ifft")
        except AttributeError:  # pragma: no cover
            try:
                fn = getattr(mrand, "ifft")
            except AttributeError:  # pragma: no cover  # pragma: no cover
                raise NotImplementedError("Missing in compiler: ifft")
    res = fn(_u(a, "a"), _u(n, "n"), _u(axis, "axis"), _u(norm, "norm"))
    return _w(res)


def fft2(a, s=None, axes=(-2, -1), norm=None, stream=None):  # pragma: no cover
    """Compute fft2.

    Args:
        a: The a argument.
        s: The s argument.
        axes: The axes argument.
        norm: The norm argument.
        stream: The stream argument.

    Returns:
        The result of fft2.

    """
    from zero_mlx.array import _to_tensor
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
            } and "fft2" not in (
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
        fn = getattr(mops, "fft2")
    except AttributeError:  # pragma: no cover
        try:
            fn = getattr(mnn, "fft2")
        except AttributeError:  # pragma: no cover
            try:
                fn = getattr(mrand, "fft2")
            except AttributeError:  # pragma: no cover  # pragma: no cover
                raise NotImplementedError("Missing in compiler: fft2")
    res = fn(_u(a, "a"), _u(s, "s"), _u(axes, "axes"), _u(norm, "norm"))
    return _w(res)


def ifft2(a, s=None, axes=(-2, -1), norm=None, stream=None):  # pragma: no cover
    """Compute ifft2.

    Args:
        a: The a argument.
        s: The s argument.
        axes: The axes argument.
        norm: The norm argument.
        stream: The stream argument.

    Returns:
        The result of ifft2.

    """
    from zero_mlx.array import _to_tensor
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
            } and "ifft2" not in (
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
        fn = getattr(mops, "ifft2")
    except AttributeError:  # pragma: no cover
        try:
            fn = getattr(mnn, "ifft2")
        except AttributeError:  # pragma: no cover
            try:
                fn = getattr(mrand, "ifft2")
            except AttributeError:  # pragma: no cover  # pragma: no cover
                raise NotImplementedError("Missing in compiler: ifft2")
    res = fn(_u(a, "a"), _u(s, "s"), _u(axes, "axes"), _u(norm, "norm"))
    return _w(res)


def fftn(a, s=None, axes=None, norm=None, stream=None):  # pragma: no cover
    """Compute fftn.

    Args:
        a: The a argument.
        s: The s argument.
        axes: The axes argument.
        norm: The norm argument.
        stream: The stream argument.

    Returns:
        The result of fftn.

    """
    from zero_mlx.array import _to_tensor
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
            } and "fftn" not in (
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
        fn = getattr(mops, "fftn")
    except AttributeError:  # pragma: no cover
        try:
            fn = getattr(mnn, "fftn")
        except AttributeError:  # pragma: no cover
            try:
                fn = getattr(mrand, "fftn")
            except AttributeError:  # pragma: no cover  # pragma: no cover
                raise NotImplementedError("Missing in compiler: fftn")
    res = fn(_u(a, "a"), _u(s, "s"), _u(axes, "axes"), _u(norm, "norm"))
    return _w(res)


def ifftn(a, s=None, axes=None, norm=None, stream=None):  # pragma: no cover
    """Compute ifftn.

    Args:
        a: The a argument.
        s: The s argument.
        axes: The axes argument.
        norm: The norm argument.
        stream: The stream argument.

    Returns:
        The result of ifftn.

    """
    from zero_mlx.array import _to_tensor
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
            } and "ifftn" not in (
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
        fn = getattr(mops, "ifftn")
    except AttributeError:  # pragma: no cover
        try:
            fn = getattr(mnn, "ifftn")
        except AttributeError:  # pragma: no cover
            try:
                fn = getattr(mrand, "ifftn")
            except AttributeError:  # pragma: no cover  # pragma: no cover
                raise NotImplementedError("Missing in compiler: ifftn")
    res = fn(_u(a, "a"), _u(s, "s"), _u(axes, "axes"), _u(norm, "norm"))
    return _w(res)


def rfft(a, n=None, axis=-1, norm=None, stream=None):  # pragma: no cover
    """Compute rfft.

    Args:
        a: The a argument.
        n: The n argument.
        axis: The axis argument.
        norm: The norm argument.
        stream: The stream argument.

    Returns:
        The result of rfft.

    """
    from zero_mlx.array import _to_tensor
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
            } and "rfft" not in (
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
        fn = getattr(mops, "rfft")
    except AttributeError:  # pragma: no cover
        try:
            fn = getattr(mnn, "rfft")
        except AttributeError:  # pragma: no cover
            try:
                fn = getattr(mrand, "rfft")
            except AttributeError:  # pragma: no cover  # pragma: no cover
                raise NotImplementedError("Missing in compiler: rfft")
    res = fn(_u(a, "a"), _u(n, "n"), _u(axis, "axis"), _u(norm, "norm"))
    return _w(res)


def irfft(a, n=None, axis=-1, norm=None, stream=None):  # pragma: no cover
    """Compute irfft.

    Args:
        a: The a argument.
        n: The n argument.
        axis: The axis argument.
        norm: The norm argument.
        stream: The stream argument.

    Returns:
        The result of irfft.

    """
    from zero_mlx.array import _to_tensor
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
            } and "irfft" not in (
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
        fn = getattr(mops, "irfft")
    except AttributeError:  # pragma: no cover
        try:
            fn = getattr(mnn, "irfft")
        except AttributeError:  # pragma: no cover
            try:
                fn = getattr(mrand, "irfft")
            except AttributeError:  # pragma: no cover  # pragma: no cover
                raise NotImplementedError("Missing in compiler: irfft")
    res = fn(_u(a, "a"), _u(n, "n"), _u(axis, "axis"), _u(norm, "norm"))
    return _w(res)


def rfft2(a, s=None, axes=(-2, -1), norm=None, stream=None):  # pragma: no cover
    """Compute rfft2.

    Args:
        a: The a argument.
        s: The s argument.
        axes: The axes argument.
        norm: The norm argument.
        stream: The stream argument.

    Returns:
        The result of rfft2.

    """
    from zero_mlx.array import _to_tensor
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
            } and "rfft2" not in (
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
        fn = getattr(mops, "rfft2")
    except AttributeError:  # pragma: no cover
        try:
            fn = getattr(mnn, "rfft2")
        except AttributeError:  # pragma: no cover
            try:
                fn = getattr(mrand, "rfft2")
            except AttributeError:  # pragma: no cover  # pragma: no cover
                raise NotImplementedError("Missing in compiler: rfft2")
    res = fn(_u(a, "a"), _u(s, "s"), _u(axes, "axes"), _u(norm, "norm"))
    return _w(res)


def irfft2(a, s=None, axes=(-2, -1), norm=None, stream=None):  # pragma: no cover
    """Compute irfft2.

    Args:
        a: The a argument.
        s: The s argument.
        axes: The axes argument.
        norm: The norm argument.
        stream: The stream argument.

    Returns:
        The result of irfft2.

    """
    from zero_mlx.array import _to_tensor
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
            } and "irfft2" not in (
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
        fn = getattr(mops, "irfft2")
    except AttributeError:  # pragma: no cover
        try:
            fn = getattr(mnn, "irfft2")
        except AttributeError:  # pragma: no cover
            try:
                fn = getattr(mrand, "irfft2")
            except AttributeError:  # pragma: no cover  # pragma: no cover
                raise NotImplementedError("Missing in compiler: irfft2")
    res = fn(_u(a, "a"), _u(s, "s"), _u(axes, "axes"), _u(norm, "norm"))
    return _w(res)


def rfftn(a, s=None, axes=None, norm=None, stream=None):  # pragma: no cover
    """Compute rfftn.

    Args:
        a: The a argument.
        s: The s argument.
        axes: The axes argument.
        norm: The norm argument.
        stream: The stream argument.

    Returns:
        The result of rfftn.

    """
    from zero_mlx.array import _to_tensor
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
            } and "rfftn" not in (
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
        fn = getattr(mops, "rfftn")
    except AttributeError:  # pragma: no cover
        try:
            fn = getattr(mnn, "rfftn")
        except AttributeError:  # pragma: no cover
            try:
                fn = getattr(mrand, "rfftn")
            except AttributeError:  # pragma: no cover  # pragma: no cover
                raise NotImplementedError("Missing in compiler: rfftn")
    res = fn(_u(a, "a"), _u(s, "s"), _u(axes, "axes"), _u(norm, "norm"))
    return _w(res)


def irfftn(a, s=None, axes=None, norm=None, stream=None):  # pragma: no cover
    """Compute irfftn.

    Args:
        a: The a argument.
        s: The s argument.
        axes: The axes argument.
        norm: The norm argument.
        stream: The stream argument.

    Returns:
        The result of irfftn.

    """
    from zero_mlx.array import _to_tensor
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
            } and "irfftn" not in (
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
        fn = getattr(mops, "irfftn")
    except AttributeError:  # pragma: no cover
        try:
            fn = getattr(mnn, "irfftn")
        except AttributeError:  # pragma: no cover
            try:
                fn = getattr(mrand, "irfftn")
            except AttributeError:  # pragma: no cover  # pragma: no cover
                raise NotImplementedError("Missing in compiler: irfftn")
    res = fn(_u(a, "a"), _u(s, "s"), _u(axes, "axes"), _u(norm, "norm"))
    return _w(res)


def fftfreq(n, d=1.0, stream=None):  # pragma: no cover
    """Compute fftfreq.

    Args:
        n: The n argument.
        d: The d argument.
        stream: The stream argument.

    Returns:
        The result of fftfreq.

    """
    from zero_mlx.array import _to_tensor
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
            } and "fftfreq" not in (
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
        fn = getattr(mops, "fftfreq")
    except AttributeError:  # pragma: no cover
        try:
            fn = getattr(mnn, "fftfreq")
        except AttributeError:  # pragma: no cover
            try:
                fn = getattr(mrand, "fftfreq")
            except AttributeError:  # pragma: no cover  # pragma: no cover
                raise NotImplementedError("Missing in compiler: fftfreq")
    res = fn(_u(n, "n"), _u(d, "d"))
    return _w(res)


def rfftfreq(n, d=1.0, stream=None):  # pragma: no cover
    """Compute rfftfreq.

    Args:
        n: The n argument.
        d: The d argument.
        stream: The stream argument.

    Returns:
        The result of rfftfreq.

    """
    from zero_mlx.array import _to_tensor
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
            } and "rfftfreq" not in (
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
        fn = getattr(mops, "rfftfreq")
    except AttributeError:  # pragma: no cover
        try:
            fn = getattr(mnn, "rfftfreq")
        except AttributeError:  # pragma: no cover
            try:
                fn = getattr(mrand, "rfftfreq")
            except AttributeError:  # pragma: no cover  # pragma: no cover
                raise NotImplementedError("Missing in compiler: rfftfreq")
    res = fn(_u(n, "n"), _u(d, "d"))
    return _w(res)


def fftshift(x, axes=None, stream=None):  # pragma: no cover
    """Compute fftshift.

    Args:
        x: The x argument.
        axes: The axes argument.
        stream: The stream argument.

    Returns:
        The result of fftshift.

    """
    from zero_mlx.array import _to_tensor
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
            } and "fftshift" not in (
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
        fn = getattr(mops, "fftshift")
    except AttributeError:  # pragma: no cover
        try:
            fn = getattr(mnn, "fftshift")
        except AttributeError:  # pragma: no cover
            try:
                fn = getattr(mrand, "fftshift")
            except AttributeError:  # pragma: no cover  # pragma: no cover
                raise NotImplementedError("Missing in compiler: fftshift")
    res = fn(_u(x, "x"), _u(axes, "axes"))
    return _w(res)


def ifftshift(x, axes=None, stream=None):  # pragma: no cover
    """Compute ifftshift.

    Args:
        x: The x argument.
        axes: The axes argument.
        stream: The stream argument.

    Returns:
        The result of ifftshift.

    """
    from zero_mlx.array import _to_tensor
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
            } and "ifftshift" not in (
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
        fn = getattr(mops, "ifftshift")
    except AttributeError:  # pragma: no cover
        try:
            fn = getattr(mnn, "ifftshift")
        except AttributeError:  # pragma: no cover
            try:
                fn = getattr(mrand, "ifftshift")
            except AttributeError:  # pragma: no cover  # pragma: no cover
                raise NotImplementedError("Missing in compiler: ifftshift")
    res = fn(_u(x, "x"), _u(axes, "axes"))
    return _w(res)
