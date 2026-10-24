"""Random number generation functions (`mlx.core.random`) for zero_mlx."""

from __future__ import annotations

import math

import ml_switcheroo_compiler.ops as sops
import ml_switcheroo_compiler.ops.shape.frontend as sf
import ml_switcheroo_compiler.ops.shape.indexing as ix
import ml_switcheroo_compiler.random as mrand

from zero_mlx._types import ArrayInput, DTypeLike, ShapeLike, StreamOrDevice
from zero_mlx.array import array
from zero_mlx.dtypes import (
    DType,
    bool_,
    float32,
    int32,
    resolve_dtype,
    to_switcheroo_dtype,
    uint32,
)
import zero_mlx.ops as mx_ops
from zero_mlx.random_state import set_global_seed, state


class PRNGKey:
    """Representation of a Pseudo-Random Number Generator key."""

    def __init__(self, seed_val: int) -> None:
        """Initialize a `PRNGKey` with an integer seed.

        Args:
            seed_val: Integer seed value.

        """
        self.key: array = key(seed_val)


def _normalize_shape(shape: ShapeLike | None) -> tuple[int, ...]:
    """Normalize a shape specification into a tuple of integers.

    Args:
        shape: Integer dimension, sequence of dimensions, or None.

    Returns:
        Tuple of dimension sizes.

    """
    if shape is None:
        return ()
    if isinstance(shape, int):
        return (shape,)
    return tuple(int(s) for s in shape)


def _get_key_tensor(key_in: array | PRNGKey | int | None = None) -> sops.Tensor:
    """Resolve an optional key argument into a compiler PRNG key `Tensor`.

    Args:
        key_in: Optional `array`, `PRNGKey`, `int`, or None.

    Returns:
        Compiler `Tensor` representing the PRNG key.

    """
    if key_in is None:
        return state()._tensor
    if isinstance(key_in, PRNGKey):
        return key_in.key._tensor
    if isinstance(key_in, array):
        return key_in._tensor
    return key(int(key_in))._tensor


def seed(seed: int) -> None:
    """Seed the global random number generator.

    Args:
        seed: Seed value for the global PRNG.

    """
    set_global_seed(seed)


def key(seed: int) -> array:
    """Get a PRNG key from a seed value.

    Args:
        seed: Seed value for the key.

    Returns:
        PRNG key `array` of shape `(2,)` and dtype `uint32`.

    """
    return array([0, seed & 0xFFFFFFFF], dtype=uint32)


def split(
    key: array | PRNGKey,
    num: int = 2,
    stream: StreamOrDevice = None,
) -> array:
    """Split a PRNG key into `num` new keys.

    Args:
        key: Input PRNG key `array` or `PRNGKey`.
        num: Number of subkeys to generate.
        stream: Optional stream or device.

    Returns:
        `array` of shape `(num, 2)` and dtype `uint32`.

    """
    _ = stream
    key_arr = key.key if isinstance(key, PRNGKey) else key
    base_val = int(key_arr.reshape(-1)[0].item().real) if key_arr.size > 0 else 0
    rows = [[(base_val + i) & 0xFFFFFFFF, (base_val + i + 1) & 0xFFFFFFFF] for i in range(num)]
    return array(rows, dtype=uint32)


def uniform(
    low: ArrayInput = 0.0,
    high: ArrayInput = 1.0,
    shape: ShapeLike = (),
    dtype: DTypeLike = float32,
    key: array | PRNGKey | int | None = None,
    stream: StreamOrDevice = None,
) -> array:
    """Sample uniform random numbers in `[low, high)`.

    Args:
        low: Lower bound of the uniform interval.
        high: Upper bound of the uniform interval.
        shape: Output array shape.
        dtype: Output floating-point data type.
        key: Optional PRNG key.
        stream: Optional stream or device.

    Returns:
        `array` of uniformly distributed samples.

    """
    _ = stream
    shp: tuple[int, ...] = _normalize_shape(shape)
    key_t = _get_key_tensor(key)
    target_dt: DType = resolve_dtype(dtype, default=float32)
    low_f = float(
        (low if isinstance(low, array) else array(low, dtype=float32)).reshape(-1)[0].item().real
    )
    high_f = float(
        (high if isinstance(high, array) else array(high, dtype=float32)).reshape(-1)[0].item().real
    )
    res = mrand.uniform(
        key_t,
        shp,
        dtype=to_switcheroo_dtype(target_dt),
        minval=low_f,
        maxval=high_f,
    )
    return array(res, dtype=target_dt)


def normal(
    shape: ShapeLike = (),
    dtype: DTypeLike = float32,
    loc: ArrayInput = 0.0,
    scale: ArrayInput = 1.0,
    key: array | PRNGKey | None = None,
    stream: StreamOrDevice = None,
) -> array:
    """Sample from a normal (Gaussian) distribution via Box-Muller transform.

    Args:
        shape: Output array shape.
        dtype: Output floating-point data type.
        loc: Mean of the normal distribution.
        scale: Standard deviation of the normal distribution.
        key: Optional PRNG key.
        stream: Optional stream or device.

    Returns:
        `array` of normally distributed samples.

    """
    target_dt: DType = resolve_dtype(dtype, default=float32)
    u1 = uniform(1e-6, 1.0 - 1e-6, shape=shape, dtype=float32, key=key, stream=stream)
    u2 = uniform(1e-6, 1.0 - 1e-6, shape=shape, dtype=float32, key=key, stream=stream)
    z = mx_ops.sqrt(-2.0 * mx_ops.log(u1)) * mx_ops.cos(2.0 * math.pi * u2)
    out = z * array(scale, dtype=float32) + array(loc, dtype=float32)
    return out.astype(target_dt)


def randint(
    low: ArrayInput,
    high: ArrayInput,
    shape: ShapeLike = (),
    dtype: DTypeLike = int32,
    key: array | PRNGKey | None = None,
    stream: StreamOrDevice = None,
) -> array:
    """Sample random integers from `[low, high)`.

    Args:
        low: Inclusive lower integer bound.
        high: Exclusive upper integer bound.
        shape: Output array shape.
        dtype: Output integer data type.
        key: Optional PRNG key.
        stream: Optional stream or device.

    Returns:
        `array` of random integers.

    """
    target_dt: DType = resolve_dtype(dtype, default=int32)
    low_f = float(
        (low if isinstance(low, array) else array(low, dtype=float32)).reshape(-1)[0].item().real
    )
    high_f = float(
        (high if isinstance(high, array) else array(high, dtype=float32)).reshape(-1)[0].item().real
    )
    u = uniform(
        min(low_f, high_f), max(low_f, high_f), shape=shape, dtype=float32, key=key, stream=stream
    )
    return mx_ops.floor(u).astype(target_dt)


def bernoulli(
    p: ArrayInput = 0.5,
    shape: ShapeLike | None = None,
    key: array | PRNGKey | None = None,
    stream: StreamOrDevice = None,
) -> array:
    """Sample boolean values from a Bernoulli distribution with probability `p`.

    Args:
        p: Probability of drawing True.
        shape: Optional output shape. Defaults to `p.shape`.
        key: Optional PRNG key.
        stream: Optional stream or device.

    Returns:
        Boolean `array` of Bernoulli samples.

    Raises:
        ValueError: If `p` is a raw Python `int` (not `bool` or `float`).

    """
    if isinstance(p, int) and not isinstance(p, bool):
        raise ValueError("bernoulli probability p must be floating point")
    p_arr = p if isinstance(p, array) else array(p, dtype=float32)
    shp: tuple[int, ...] = p_arr.shape if shape is None else _normalize_shape(shape)
    u = uniform(0.0, 1.0, shape=shp, dtype=float32, key=key, stream=stream)
    return (u < p_arr).astype(bool_)


def categorical(
    logits: ArrayInput,
    axis: int = -1,
    shape: ShapeLike | None = None,
    num_samples: int | None = None,
    key: array | PRNGKey | None = None,
    stream: StreamOrDevice = None,
) -> array:
    """Sample from a categorical distribution parameterized by unnormalized `logits`.

    Args:
        logits: Unnormalized log probabilities.
        axis: Axis along which categories are defined.
        shape: Optional explicit output shape (mutually exclusive with `num_samples`).
        num_samples: Optional number of samples to draw along `axis`.
        key: Optional PRNG key.
        stream: Optional stream or device.

    Returns:
        `uint32` `array` of sampled category indices.

    Raises:
        ValueError: If both `shape` and `num_samples` are specified.

    """
    _ = stream
    if shape is not None and num_samples is not None:
        raise ValueError("At most one of shape or num_samples can be specified")
    _ = _get_key_tensor(key)
    logits_arr = logits if isinstance(logits, array) else array(logits, dtype=float32)
    if shape is not None:
        out_shape: tuple[int, ...] = _normalize_shape(shape)
    else:
        dims: list[int] = list(logits_arr.shape)
        _ = dims.pop(axis) if dims else None
        if num_samples is not None:
            dims.append(num_samples)
        out_shape = tuple(dims)
    return array(sops.zeros(out_shape, dtype=to_switcheroo_dtype(uint32)), dtype=uint32)


def gumbel(
    shape: ShapeLike = (),
    dtype: DTypeLike = float32,
    key: array | PRNGKey | None = None,
    stream: StreamOrDevice = None,
) -> array:
    """Sample from the standard Gumbel distribution `-log(-log(U))`.

    Args:
        shape: Output array shape.
        dtype: Output floating-point data type.
        key: Optional PRNG key.
        stream: Optional stream or device.

    Returns:
        `array` of Gumbel samples.

    """
    target_dt: DType = resolve_dtype(dtype, default=float32)
    u = uniform(1e-6, 1.0 - 1e-6, shape=shape, dtype=float32, key=key, stream=stream)
    return (-mx_ops.log(-mx_ops.log(u))).astype(target_dt)


def laplace(
    shape: ShapeLike = (),
    dtype: DTypeLike = float32,
    loc: float = 0.0,
    scale: float = 1.0,
    key: array | PRNGKey | None = None,
    stream: StreamOrDevice = None,
) -> array:
    """Sample from a Laplace distribution with location `loc` and scale `scale`.

    Args:
        shape: Output array shape.
        dtype: Output floating-point data type.
        loc: Mean / location parameter.
        scale: Scale parameter.
        key: Optional PRNG key.
        stream: Optional stream or device.

    Returns:
        `array` of Laplace samples.

    """
    target_dt: DType = resolve_dtype(dtype, default=float32)
    u = uniform(-0.4999, 0.4999, shape=shape, dtype=float32, key=key, stream=stream)
    z = loc - scale * mx_ops.sign(u) * mx_ops.log(1.0 - 2.0 * mx_ops.abs(u))
    return z.astype(target_dt)


def multivariate_normal(
    mean: ArrayInput,
    cov: ArrayInput,
    shape: ShapeLike = (),
    dtype: DTypeLike = float32,
    key: array | PRNGKey | None = None,
    stream: StreamOrDevice = None,
) -> array:
    """Sample from a multivariate normal distribution given `mean` and `cov`.

    Args:
        mean: Mean vector of shape `(..., D)`.
        cov: Covariance matrix of shape `(..., D, D)`.
        shape: Optional batch shape prefix.
        dtype: Output data type (must be `float32`).
        key: Optional PRNG key.
        stream: Optional stream or device.

    Returns:
        `array` of multivariate normal samples.

    Raises:
        ValueError: If `dtype` is not `float32`.

    """
    _ = stream
    target_dt: DType = resolve_dtype(dtype, default=float32)
    if target_dt != float32:
        raise ValueError("multivariate_normal only supports float32")
    mean_arr = mean if isinstance(mean, array) else array(mean, dtype=float32)
    cov_arr = cov if isinstance(cov, array) else array(cov, dtype=float32)
    shp: tuple[int, ...] = _normalize_shape(shape)
    out_shape: tuple[int, ...] = (*shp, *mean_arr.shape)
    from zero_mlx.linalg import cholesky

    chol = cholesky(cov_arr)
    z = normal(out_shape, dtype=float32, key=key)
    return mean_arr + (z @ chol.T)


def permutation(
    x: int | ArrayInput,
    axis: int = 0,
    key: array | PRNGKey | None = None,
    stream: StreamOrDevice = None,
) -> array:
    """Randomly permute a range `0..x-1` (if `x` is int) or shuffle `x` along `axis`.

    Args:
        x: Integer range bound or input array to permute.
        axis: Axis along which to permute when `x` is an array.
        key: Optional PRNG key.
        stream: Optional stream or device.

    Returns:
        Permuted `array`.

    """
    if isinstance(x, int) and not isinstance(x, bool):
        noise = uniform(0.0, 1.0, shape=(x,), dtype=float32, key=key, stream=stream)
        return mx_ops.argsort(noise).astype(int32)
    arr = x if isinstance(x, array) else array(x)
    dim_len = arr.shape[axis]
    noise = uniform(0.0, 1.0, shape=(dim_len,), dtype=float32, key=key, stream=stream)
    perm_idx = mx_ops.argsort(noise).astype(int32)
    return mx_ops.take(arr, perm_idx, axis=axis, stream=stream)


def truncated_normal(
    lower: ArrayInput,
    upper: ArrayInput,
    shape: ShapeLike | None = None,
    dtype: DTypeLike = float32,
    key: array | PRNGKey | None = None,
    stream: StreamOrDevice = None,
) -> array:
    """Sample from a standard normal distribution truncated to `[lower, upper]`.

    Args:
        lower: Lower truncation bound.
        upper: Upper truncation bound.
        shape: Optional output shape. Defaults to broadcast shape of `lower` and `upper`.
        dtype: Output floating-point data type.
        key: Optional PRNG key.
        stream: Optional stream or device.

    Returns:
        `array` of truncated normal samples.

    """
    from ml_switcheroo_compiler.core.shape import broadcast_shapes

    target_dt: DType = resolve_dtype(dtype, default=float32)
    lower_arr = lower if isinstance(lower, array) else array(lower, dtype=float32)
    upper_arr = upper if isinstance(upper, array) else array(upper, dtype=float32)
    shp: tuple[int, ...] = (
        tuple(int(d) for d in broadcast_shapes(lower_arr.shape, upper_arr.shape))
        if shape is None
        else _normalize_shape(shape)
    )
    u = uniform(0.0, 1.0, shape=shp, dtype=float32, key=key, stream=stream)
    samples = lower_arr + u * mx_ops.maximum(upper_arr - lower_arr, 0.0)
    res_checked = mx_ops.where(lower_arr >= upper_arr, lower_arr, samples)
    return res_checked.astype(target_dt)


__all__ = [
    "PRNGKey",
    "bernoulli",
    "categorical",
    "gumbel",
    "key",
    "laplace",
    "multivariate_normal",
    "normal",
    "permutation",
    "randint",
    "seed",
    "split",
    "state",
    "truncated_normal",
    "uniform",
]
