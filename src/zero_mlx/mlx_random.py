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


def _mix32(val: int) -> int:
    """Apply a 32-bit SplitMix/Murmur3 avalanche hash to an integer.

    Args:
        val: Input integer.

    Returns:
        Avalanched 32-bit unsigned integer.

    """
    x = val & 0xFFFFFFFF
    x = ((x ^ (x >> 16)) * 0x85EBCA6B) & 0xFFFFFFFF
    x = ((x ^ (x >> 13)) * 0xC2B2AE35) & 0xFFFFFFFF
    return (x ^ (x >> 16)) & 0xFFFFFFFF


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
    hi = (seed >> 32) & 0xFFFFFFFF
    lo = seed & 0xFFFFFFFF
    return array([hi, lo], dtype=uint32)


def split(
    key: array | PRNGKey,
    num: int = 2,
    stream: StreamOrDevice = None,
) -> array:
    """Split a PRNG key into `num` deterministic, uncorrelated subkeys.

    Args:
        key: Input PRNG key `array` or `PRNGKey`.
        num: Number of subkeys to generate.
        stream: Optional stream or device.

    Returns:
        `array` of shape `(num, 2)` and dtype `uint32`.

    """
    _ = stream
    key_arr = key.key if isinstance(key, PRNGKey) else key
    flat = key_arr.reshape(-1)
    k0 = int(flat[0].item().real) if flat.size > 0 else 0
    k1 = int(flat[1].item().real) if flat.size > 1 else 0
    rows = [
        [
            _mix32(k0 ^ (i * 0x9E3779B9)),
            _mix32(k1 ^ ((i + 1) * 0x7F4A7C15)),
        ]
        for i in range(num)
    ]
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
    from ml_switcheroo_compiler.core.shape import broadcast_shapes

    _ = stream
    req_shp: tuple[int, ...] = _normalize_shape(shape)
    low_arr = low if isinstance(low, array) else array(low, dtype=float32)
    high_arr = high if isinstance(high, array) else array(high, dtype=float32)
    bounds_shp = tuple(int(d) for d in broadcast_shapes(low_arr.shape, high_arr.shape))
    shp: tuple[int, ...] = (
        tuple(int(d) for d in broadcast_shapes(req_shp, bounds_shp)) if req_shp else bounds_shp
    )
    key_t = _get_key_tensor(key)
    target_dt: DType = resolve_dtype(dtype, default=float32)
    u01 = array(
        mrand.uniform(
            key_t,
            shp,
            dtype=to_switcheroo_dtype(float32),
            minval=0.0,
            maxval=1.0,
        ),
        dtype=float32,
    )
    res = low_arr.astype(float32) + u01 * (high_arr.astype(float32) - low_arr.astype(float32))
    return res.astype(target_dt)


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
    base_key: array = state() if key is None else (key.key if isinstance(key, PRNGKey) else key)
    subkeys = split(base_key, num=2, stream=stream)
    u1 = uniform(1e-6, 1.0 - 1e-6, shape=shape, dtype=float32, key=subkeys[0], stream=stream)
    u2 = uniform(1e-6, 1.0 - 1e-6, shape=shape, dtype=float32, key=subkeys[1], stream=stream)
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
    low_arr = (low if isinstance(low, array) else array(low, dtype=float32)).astype(float32)
    high_arr = (high if isinstance(high, array) else array(high, dtype=float32)).astype(float32)
    lo_b = mx_ops.minimum(low_arr, high_arr)
    hi_b = mx_ops.maximum(low_arr, high_arr)
    u = uniform(lo_b, hi_b, shape=shape, dtype=float32, key=key, stream=stream)
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
    if shape is not None and num_samples is not None:
        raise ValueError("At most one of shape or num_samples can be specified")
    _ = _get_key_tensor(key)
    logits_arr = (logits if isinstance(logits, array) else array(logits, dtype=float32)).astype(
        float32
    )
    if logits_arr.ndim == 0 or logits_arr.size == 0:
        out_empty = () if shape is None else _normalize_shape(shape)
        return array(sops.zeros(out_empty, dtype=to_switcheroo_dtype(uint32)), dtype=uint32)

    norm_ax = axis % logits_arr.ndim
    moved = logits_arr.moveaxis(norm_ax, -1) if norm_ax != logits_arr.ndim - 1 else logits_arr
    num_classes = moved.shape[-1]

    if shape is not None:
        out_shape: tuple[int, ...] = _normalize_shape(shape)
        g_noise = gumbel((*out_shape, num_classes), dtype=float32, key=key, stream=stream)
        return mx_ops.argmax(moved + g_noise, axis=-1, stream=stream).astype(uint32)

    if num_samples is not None:
        g_noise = gumbel(
            (*moved.shape[:-1], num_samples, num_classes), dtype=float32, key=key, stream=stream
        )
        perturbed = mx_ops.expand_dims(moved, axis=-2) + g_noise
        return mx_ops.argmax(perturbed, axis=-1, stream=stream).astype(uint32)

    g_noise = gumbel(moved.shape, dtype=float32, key=key, stream=stream)
    return mx_ops.argmax(moved + g_noise, axis=-1, stream=stream).astype(uint32)


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
    sqrt2 = math.sqrt(2.0)
    cdf_low = 0.5 * (1.0 + mx_ops.erf(lower_arr / sqrt2, stream=stream))
    cdf_high = 0.5 * (1.0 + mx_ops.erf(upper_arr / sqrt2, stream=stream))
    u = uniform(0.0, 1.0, shape=shp, dtype=float32, key=key, stream=stream)
    p = cdf_low + u * mx_ops.maximum(cdf_high - cdf_low, 0.0, stream=stream)
    p_clipped = mx_ops.clip(2.0 * p - 1.0, -0.999999, 0.999999, stream=stream)
    samples = mx_ops.clip(
        sqrt2 * mx_ops.erfinv(p_clipped, stream=stream),
        lower_arr,
        upper_arr,
        stream=stream,
    )
    res_checked = mx_ops.where(lower_arr >= upper_arr, lower_arr, samples, stream=stream)
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
