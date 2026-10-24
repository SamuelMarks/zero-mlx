"""Weight initialization functions (`mlx.nn.init`) for zero_mlx."""

from __future__ import annotations

import math
from typing import Callable, Protocol

from zero_mlx.array import array
from zero_mlx.dtypes import DType, DTypeLike, float32, resolve_dtype
from zero_mlx.mlx_random import normal as rand_normal, uniform as rand_uniform
from zero_mlx.ops import eye, full, where, zeros_like

InitializerFn = Callable[[array], array]


class ModeInitializerFn(Protocol):
    """Protocol for He (Kaiming) initializers accepting optional `mode` and `gain`."""

    def __call__(
        self,
        a: array,
        mode: str = "fan_in",
        gain: float = 1.0,
    ) -> array:
        """Initialize `a` with Kaiming scaling.

        Args:
            a: Template `array`.
            mode: Either `'fan_in'` or `'fan_out'`.
            gain: Multiplicative scaling factor.

        Returns:
            Initialized `array`.

        """


def constant(value: float, dtype: DTypeLike = float32) -> InitializerFn:
    """Create an initializer that fills an array with a constant `value`.

    Args:
        value: Constant scalar value.
        dtype: Default data type for the initialized array.

    Returns:
        Initializer callable mapping a template `array` to a constant-filled `array`.

    """
    dt: DType = resolve_dtype(dtype, default=float32)

    def _initializer(a: array) -> array:
        """Fill `a.shape` with `value`.

        Args:
            a: Template `array`.

        Returns:
            Initialized `array`.

        """
        return full(a.shape, value, dtype=dt)

    return _initializer


def normal(mean: float = 0.0, std: float = 1.0, dtype: DTypeLike = float32) -> InitializerFn:
    """Create an initializer that samples from a normal distribution `N(mean, std^2)`.

    Args:
        mean: Mean of the normal distribution.
        std: Standard deviation of the normal distribution.
        dtype: Default data type for the initialized array.

    Returns:
        Initializer callable.

    """
    dt: DType = resolve_dtype(dtype, default=float32)

    def _initializer(a: array) -> array:
        """Sample normal weights of shape `a.shape`.

        Args:
            a: Template `array`.

        Returns:
            Initialized `array`.

        """
        return rand_normal(a.shape, dtype=dt, loc=mean, scale=std)

    return _initializer


def uniform(low: float = 0.0, high: float = 1.0, dtype: DTypeLike = float32) -> InitializerFn:
    """Create an initializer that samples from a uniform distribution `[low, high)`.

    Args:
        low: Lower bound of the uniform range.
        high: Upper bound of the uniform range.
        dtype: Default data type for the initialized array.

    Returns:
        Initializer callable.

    """
    dt: DType = resolve_dtype(dtype, default=float32)

    def _initializer(a: array) -> array:
        """Sample uniform weights of shape `a.shape`.

        Args:
            a: Template `array`.

        Returns:
            Initialized `array`.

        """
        return rand_uniform(low=low, high=high, shape=a.shape, dtype=dt)

    return _initializer


def identity(dtype: DTypeLike = float32) -> InitializerFn:
    """Create an initializer that returns a square identity matrix matching `a.shape`.

    Args:
        dtype: Default data type for the identity matrix.

    Returns:
        Initializer callable.

    """
    dt: DType = resolve_dtype(dtype, default=float32)

    def _initializer(a: array) -> array:
        """Create an identity matrix for 2D `a`.

        Args:
            a: 2D square template `array`.

        Returns:
            Identity `array`.

        Raises:
            ValueError: If `a` is not a 2D square array.

        """
        if a.ndim != 2 or a.shape[0] != a.shape[1]:
            raise ValueError(f"Input array must be a 2D square matrix, got shape {a.shape}")
        return eye(a.shape[0], dtype=dt)

    return _initializer


def _calculate_fan_in_fan_out(a: array) -> tuple[int, int]:
    """Compute `(fan_in, fan_out)` for a weight tensor `a` of at least 2 dimensions.

    Args:
        a: Weight template `array`.

    Returns:
        Tuple `(fan_in, fan_out)`.

    Raises:
        ValueError: If `a.ndim < 2`.

    """
    if a.ndim < 2:
        raise ValueError(f"Fan-in/fan-out requires at least 2D array, got {a.ndim}D")
    receptive_field_size: int = 1
    for s in a.shape[2:]:
        receptive_field_size *= s
    fan_in: int = a.shape[1] * receptive_field_size
    fan_out: int = a.shape[0] * receptive_field_size
    return fan_in, fan_out


def glorot_normal(dtype: DTypeLike = float32) -> InitializerFn:
    """Create a Glorot (Xavier) normal initializer.

    Args:
        dtype: Default floating-point data type.

    Returns:
        Glorot normal initializer callable.

    """
    dt: DType = resolve_dtype(dtype, default=float32)

    def _initializer(a: array, gain: float = 1.0) -> array:
        """Initialize `a` using Glorot normal scaling.

        Args:
            a: Template weight `array`.
            gain: Multiplicative scaling factor.

        Returns:
            Initialized `array`.

        """
        fan_in, fan_out = _calculate_fan_in_fan_out(a)
        std = gain * math.sqrt(2.0 / float(fan_in + fan_out))
        return rand_normal(a.shape, dtype=dt, scale=std)

    return _initializer


def glorot_uniform(dtype: DTypeLike = float32) -> InitializerFn:
    """Create a Glorot (Xavier) uniform initializer.

    Args:
        dtype: Default floating-point data type.

    Returns:
        Glorot uniform initializer callable.

    """
    dt: DType = resolve_dtype(dtype, default=float32)

    def _initializer(a: array, gain: float = 1.0) -> array:
        """Initialize `a` using Glorot uniform bounds.

        Args:
            a: Template weight `array`.
            gain: Multiplicative scaling factor.

        Returns:
            Initialized `array`.

        """
        fan_in, fan_out = _calculate_fan_in_fan_out(a)
        limit = gain * math.sqrt(6.0 / float(fan_in + fan_out))
        return rand_uniform(low=-limit, high=limit, shape=a.shape, dtype=dt)

    return _initializer


def he_normal(dtype: DTypeLike = float32) -> ModeInitializerFn:
    """Create a He (Kaiming) normal initializer.

    Args:
        dtype: Default floating-point data type.

    Returns:
        He normal initializer callable.

    """
    dt: DType = resolve_dtype(dtype, default=float32)

    def _initializer(a: array, mode: str = "fan_in", gain: float = 1.0) -> array:
        """Initialize `a` using He normal scaling.

        Args:
            a: Template weight `array`.
            mode: Either `'fan_in'` or `'fan_out'`.
            gain: Scaling factor.

        Returns:
            Initialized `array`.

        Raises:
            ValueError: If `mode` is not `'fan_in'` or `'fan_out'`.

        """
        if mode not in ("fan_in", "fan_out"):
            raise ValueError(f"Invalid mode '{mode}', expected 'fan_in' or 'fan_out'")
        fan_in, fan_out = _calculate_fan_in_fan_out(a)
        fan = fan_in if mode == "fan_in" else fan_out
        std = gain / math.sqrt(float(fan))
        return rand_normal(a.shape, dtype=dt, scale=std)

    return _initializer


def he_uniform(dtype: DTypeLike = float32) -> ModeInitializerFn:
    """Create a He (Kaiming) uniform initializer.

    Args:
        dtype: Default floating-point data type.

    Returns:
        He uniform initializer callable.

    """
    dt: DType = resolve_dtype(dtype, default=float32)

    def _initializer(a: array, mode: str = "fan_in", gain: float = 1.0) -> array:
        """Initialize `a` using He uniform bounds.

        Args:
            a: Template weight `array`.
            mode: Either `'fan_in'` or `'fan_out'`.
            gain: Scaling factor.

        Returns:
            Initialized `array`.

        Raises:
            ValueError: If `mode` is not `'fan_in'` or `'fan_out'`.

        """
        if mode not in ("fan_in", "fan_out"):
            raise ValueError(f"Invalid mode '{mode}', expected 'fan_in' or 'fan_out'")
        fan_in, fan_out = _calculate_fan_in_fan_out(a)
        fan = fan_in if mode == "fan_in" else fan_out
        limit = gain * math.sqrt(3.0 / float(fan))
        return rand_uniform(low=-limit, high=limit, shape=a.shape, dtype=dt)

    return _initializer


def orthogonal(gain: float = 1.0, dtype: DTypeLike = float32) -> InitializerFn:
    """Create an orthogonal matrix initializer via QR decomposition.

    Args:
        gain: Multiplicative scaling factor applied to the orthogonal matrix.
        dtype: Output floating-point data type.

    Returns:
        Orthogonal initializer callable.

    """
    dt: DType = resolve_dtype(dtype, default=float32)

    def _initializer(a: array) -> array:
        """Generate an orthogonal weight matrix of shape `a.shape`.

        Args:
            a: 2D template `array`.

        Returns:
            Orthogonal `array`.

        Raises:
            ValueError: If `a.ndim != 2`.

        """
        if a.ndim != 2:
            raise ValueError(f"Orthogonal initialization requires a 2D array, got {a.ndim}D")
        from zero_mlx.linalg import qr

        rows, cols = a.shape
        n = max(rows, cols)
        rmat = rand_normal((n, n), dtype=float32)
        q, _ = qr(rmat)
        q_sliced = q[:rows, :cols] * gain
        return q_sliced.astype(dt)

    return _initializer


def sparse(
    sparsity: float,
    mean: float = 0.0,
    std: float = 1.0,
    dtype: DTypeLike = float32,
) -> InitializerFn:
    """Create a 2D sparse normal initializer where a fraction `sparsity` of elements are zeroed.

    Args:
        sparsity: Fraction of elements in each column/row to set to zero (in `[0, 1)`).
        mean: Mean of the normal distribution for non-zero entries.
        std: Standard deviation of the normal distribution for non-zero entries.
        dtype: Output data type.

    Returns:
        Sparse initializer callable.

    """
    dt: DType = resolve_dtype(dtype, default=float32)

    def _initializer(a: array) -> array:
        """Generate a sparse 2D weight matrix of shape `a.shape`.

        Args:
            a: 2D template `array`.

        Returns:
            Sparse `array`.

        Raises:
            ValueError: If `a.ndim != 2`.

        """
        if a.ndim != 2:
            raise ValueError(f"Sparse initialization requires a 2D array, got {a.ndim}D")
        vals = rand_normal(a.shape, dtype=dt, loc=mean, scale=std)
        mask = rand_uniform(low=0.0, high=1.0, shape=a.shape, dtype=float32) >= sparsity
        return where(mask, vals, zeros_like(vals))

    return _initializer


__all__ = [
    "InitializerFn",
    "constant",
    "glorot_normal",
    "glorot_uniform",
    "he_normal",
    "he_uniform",
    "identity",
    "normal",
    "orthogonal",
    "sparse",
    "uniform",
]
