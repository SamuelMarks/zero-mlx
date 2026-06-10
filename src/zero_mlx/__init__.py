"""zero_mlx API."""

import ml_switcheroo
import numpy as np
import ml_switcheroo.ops as _ops


def _to_tensor(x):
    if hasattr(x, "_tensor"):
        return x._tensor
    if isinstance(x, ml_switcheroo.Tensor):
        return x
    from ml_switcheroo.core.config import config

    return ml_switcheroo.Tensor(
        np.array(x),
        np.array(x).shape,
        config.default_float_dtype,
        config.default_device,
    )


def _wrap(x):
    if isinstance(x, ml_switcheroo.Tensor):
        return array(x)
    return x


class array:
    """Array class."""

    def __init__(self, data):
        """Initialize array."""
        if hasattr(data, "_tensor"):
            self._tensor = data._tensor
        elif isinstance(data, ml_switcheroo.Tensor):
            self._tensor = data
        else:
            self._tensor = _to_tensor(data)

    def __add__(self, other):
        """Add two arrays."""
        return _wrap(_ops.add(self._tensor, _to_tensor(other)))

    @property
    def shape(self):
        return self._tensor.shape

    @property
    def data(self):
        from ml_switcheroo.core.config import config

        if config.eager_mode:
            arr = np.array(self._tensor.data)
            if arr.shape == ():
                return arr.item()
            return arr
        return self._tensor.data


class core:
    """Core namespace."""

    @staticmethod
    def eval(*args):
        """Evaluate."""
        # No-op in eager mode
        return args


class Stream:
    """Stream context."""

    def __enter__(self):
        """Enter context manager."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit context manager."""
        pass


def stream():
    """Create a stream."""
    return Stream()
