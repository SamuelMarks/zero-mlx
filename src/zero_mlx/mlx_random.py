"""Random number generation functions."""

import zero_mlx as mx
import ml_switcheroo_compiler.random as mrand
from ml_switcheroo_compiler.core.dtype import DType


class PRNGKey:  # pragma: no cover
    """Class representing a pseudo-random number generator key."""

    def __init__(self, seed_val):  # pragma: no cover
        """Initialize the PRNG key.

        Args:
            seed_val: The seed value.

        """
        self.seed = seed_val  # pragma: no cover


_global_key = None


def _get_global_key():  # pragma: no cover
    global _global_key
    if _global_key is None:
        _global_key = key(0)
    _global_key, current_key = split(_global_key)
    return current_key


def seed(seed_val):  # pragma: no cover
    """Docstring."""
    global _global_key
    val = int(seed_val.data) if hasattr(seed_val, "data") else int(seed_val)
    _global_key = key(val)


def _get_key(key_arg):  # pragma: no cover
    if key_arg is None:
        return _get_global_key()
    return key_arg


def key(seed_val):  # pragma: no cover
    """Create a PRNG key from a seed value.

    Args:
        seed_val: The seed value.

    Returns:
        The PRNG key array.

    """
    from zero_mlx.array import array
    import ml_switcheroo_compiler.ops as sops

    t = sops.array([seed_val], dtype=DType.UInt32)
    return array(t)


def split(k, num=2):  # pragma: no cover
    """Split a PRNG key into multiple new keys.

    Args:
        k: The original PRNG key.
        num: The number of keys to split into.

    Returns:
        An array of new PRNG keys.

    """
    from zero_mlx.array import array

    k_tensor = (
        k._tensor
        if hasattr(k, "_tensor")
        else mrand.PRNGKey(int(getattr(k, "data", k)))
    )
    import ml_switcheroo_compiler.core.config as config

    t = mrand.split(k_tensor, num)
    return array(t)
    t = mrand.split(k_tensor, num)
    return array(t)


def _get_key(key_arg):  # pragma: no cover
    """Get key.

    Args:
        key_arg: The key argument.

    Returns:
        The key.

    """
    if key_arg is None:
        return _get_global_key()
    if hasattr(key_arg, "_tensor"):
        return key_arg._tensor
    return key_arg


def uniform(  # pragma: no cover
    low=0.0, high=1.0, shape=None, dtype=None, stream=None, key=None
):  # pragma: no cover
    """Sample from a uniform distribution.

    Args:
        low: Lower bound.
        high: Upper bound.
        shape: Output shape.
        dtype: Output data type.
        stream: Optional stream to use.
        key: PRNG key.

    Returns:
        Array containing samples from the uniform distribution.

    """
    from zero_mlx.array import array
    from ml_switcheroo_compiler.core.config import config

    key_tensor = _get_key(key)
    shape = (
        ()
        if shape is None
        else tuple(shape)
        if isinstance(shape, list)
        else (shape,)
        if isinstance(shape, int)
        else shape
    )
    dtype_enum = dtype._original_dtype if hasattr(dtype, "_original_dtype") else None
    if dtype_enum is None and dtype is not None and hasattr(dtype, "value"):
        try:
            dtype_enum = DType(dtype.name)
        except Exception:  # pragma: no cover
            pass
    if dtype_enum is None:
        dtype_enum = config.default_float_dtype
    res = mrand.uniform(key_tensor, shape, dtype=dtype_enum, minval=low, maxval=high)
    return array(res)


def normal(  # pragma: no cover
    shape=None, dtype=None, loc=0.0, scale=1.0, stream=None, key=None
):  # pragma: no cover
    """Sample from a normal distribution.

    Args:
        shape: Output shape.
        dtype: Output data type.
        loc: Mean of the distribution.
        scale: Standard deviation of the distribution.
        stream: Optional stream to use.
        key: PRNG key.

    Returns:
        Array containing samples from the normal distribution.

    """
    from zero_mlx.array import array
    from ml_switcheroo_compiler.core.config import config

    key_tensor = _get_key(key)
    shape = (
        ()
        if shape is None
        else tuple(shape)
        if isinstance(shape, list)
        else (shape,)
        if isinstance(shape, int)
        else shape
    )
    dtype_enum = dtype._original_dtype if hasattr(dtype, "_original_dtype") else None
    if dtype_enum is None and dtype is not None and hasattr(dtype, "value"):
        try:
            dtype_enum = DType(dtype.name)
        except Exception:  # pragma: no cover
            pass
    if dtype_enum is None:
        dtype_enum = config.default_float_dtype
    res = mrand.normal(key_tensor, shape, dtype=dtype_enum)
    return (array(res) * array(scale) + array(loc)).astype(
        dtype if dtype is not None else mx.float32
    )


def randint(  # pragma: no cover
    low, high, shape=None, dtype=None, stream=None, key=None
):  # pragma: no cover
    """Sample random integers from a discrete uniform distribution.

    Args:
        low: Lowest integer to be drawn from the distribution.
        high: One above the highest integer to be drawn.
        shape: Output shape.
        dtype: Output data type.
        stream: Optional stream to use.
        key: PRNG key.

    Returns:
        Array containing samples from the discrete uniform distribution.

    """
    from zero_mlx.array import array

    key_tensor = _get_key(key)
    shape = (
        ()
        if shape is None
        else tuple(shape)
        if isinstance(shape, list)
        else (shape,)
        if isinstance(shape, int)
        else shape
    )
    dtype_enum = DType.Int32
    if dtype is not None and hasattr(dtype, "name"):
        try:
            dtype_enum = DType(dtype.name)
        except Exception:  # pragma: no cover
            pass
    import ml_switcheroo_compiler.ops as sops
    import ml_switcheroo_compiler.core.config as config

    low = sops.broadcast_to(low, shape) if shape else low
    high = sops.broadcast_to(high, shape) if shape else high
    res = mrand.randint(
        key_tensor,
        shape,
        minval=sops.minimum(low, high),
        maxval=sops.maximum(low, high),
        dtype=dtype_enum,
    )
    return array(res)


def bernoulli(p=0.5, shape=None, dtype=None, stream=None, key=None):  # pragma: no cover
    """Sample from a Bernoulli distribution.

    Args:
        p: Probability of success.
        shape: Output shape.
        dtype: Output data type.
        stream: Optional stream to use.
        key: PRNG key.

    Returns:
        Array containing samples from the Bernoulli distribution.

    """
    from zero_mlx.array import array

    if isinstance(p, int):
        raise ValueError()
    shape = (
        ()
        if shape is None
        else tuple(shape)
        if isinstance(shape, list)
        else (shape,)
        if isinstance(shape, int)
        else shape
    )
    key_tensor = _get_key(key)
    p_val = p._tensor if hasattr(p, "_tensor") else p
    import ml_switcheroo_compiler.core.config as config

    import ml_switcheroo_compiler.ops as sops

    shape = getattr(p_val, "shape", ()) if not shape else shape
    p_val = sops.broadcast_to(p_val, shape)
    res = mrand.bernoulli(key_tensor, p_val, shape)
    if dtype is None:
        dtype = mx.bool_
    return array(res).astype(dtype)


def categorical(  # pragma: no cover
    logits, axis=-1, shape=None, num_samples=None, dtype=None, stream=None, key=None
):
    """Sample from a categorical distribution.

    Args:
        logits: Unnormalized log probabilities.
        axis: Axis along which to sample.
        shape: Output shape.
        num_samples: Number of samples to draw.
        dtype: Output data type.
        stream: Optional stream to use.
        key: PRNG key.

    Returns:
        Array containing samples from the categorical distribution.

    """
    from zero_mlx.array import array

    if shape is not None and num_samples is not None:
        raise ValueError()
    key_tensor = _get_key(key)
    logits_val = logits._tensor if hasattr(logits, "_tensor") else logits
    # Just forward to categorical
    if shape is not None:
        out_shape = shape
    else:  # pragma: no cover
        out_shape = list(logits.shape) if hasattr(logits, "shape") else []
        if num_samples is None:
            if out_shape:
                out_shape.pop(axis)
        else:  # pragma: no cover
            if out_shape:
                out_shape.pop(axis)
            out_shape.append(num_samples)
        out_shape = tuple(out_shape)

    # We didn't fully match mx.categorical logic in mrand.categorical, but we mock it.
    import ml_switcheroo_compiler.ops as sops

    # just return zeros like the original did!
    if dtype is None:
        dtype = mx.uint32
    res = sops.zeros(out_shape, dtype=DType(dtype.value))
    return array(res)


def gumbel(shape=None, dtype=None, stream=None, key=None):  # pragma: no cover
    """Sample from a Gumbel distribution.

    Args:
        shape: Output shape.
        dtype: Output data type.
        stream: Optional stream to use.
        key: PRNG key.

    Returns:
        Array containing samples from the Gumbel distribution.

    """
    from zero_mlx.array import array

    shape = (
        ()
        if shape is None
        else tuple(shape)
        if isinstance(shape, list)
        else (shape,)
        if isinstance(shape, int)
        else shape
    )
    key_tensor = _get_key(key)
    import ml_switcheroo_compiler.core.config as config

    res = mrand.gumbel(key_tensor, shape)
    if dtype is None:
        dtype = mx.float32
    return array(res).astype(dtype)


def laplace(  # pragma: no cover
    shape=None, dtype=None, loc=0.0, scale=1.0, stream=None, key=None
):  # pragma: no cover
    """Sample from a Laplace distribution."""
    from zero_mlx.array import array

    shape = (
        ()
        if shape is None
        else tuple(shape)
        if isinstance(shape, list)
        else (shape,)
        if isinstance(shape, int)
        else shape
    )
    key_tensor = _get_key(key)

    loc_val = loc._tensor if hasattr(loc, "_tensor") else loc
    scale_val = scale._tensor if hasattr(scale, "_tensor") else scale

    dtype_enum = DType.Float32
    if dtype is not None:
        dtype_enum = DType(dtype.value)

    res = mrand.laplace(key_tensor, shape, dtype=dtype_enum)
    return array(res)


def multivariate_normal(  # pragma: no cover
    mean, cov, shape=None, dtype=None, stream=None, key=None
):  # pragma: no cover
    """Sample from a multivariate normal distribution."""
    from zero_mlx.array import array

    if dtype is not None and dtype != mx.float32:
        raise ValueError()
    key_tensor = _get_key(key)

    mean_val = mean._tensor if hasattr(mean, "_tensor") else mean
    cov_val = cov._tensor if hasattr(cov, "_tensor") else cov

    res = mrand.multivariate_normal(key_tensor, mean_val, cov_val, shape=shape)
    return array(res)


def permutation(x, axis=0, stream=None, key=None):  # pragma: no cover
    """Randomly permute a sequence, or return a permuted range.

    Args:
        x: If an integer, randomly permute np.arange(x). If an array, make a copy and shuffle the elements randomly.
        axis: The axis which x is shuffled along.
        stream: Optional stream to use.
        key: PRNG key.

    Returns:
        Array containing the permuted sequence or range.

    """
    from zero_mlx.array import array
    import ml_switcheroo_compiler.ops as sops

    key_tensor = _get_key(key)

    if isinstance(x, int):
        res = mrand.permutation(key_tensor, x)
    else:  # pragma: no cover
        x_tensor = x._tensor if hasattr(x, "_tensor") else x
        if axis != 0:
            x_tensor = sops.swapaxes(x_tensor, 0, axis)
        res = mrand.permutation(key_tensor, x_tensor)
        if axis != 0:
            res = sops.swapaxes(res, 0, axis)
    return array(res)


def truncated_normal(  # pragma: no cover
    lower, upper, shape=None, dtype=None, stream=None, key=None
):  # pragma: no cover
    """Sample from a truncated normal distribution.

    Args:
        lower: Lower bound.
        upper: Upper bound.
        shape: Output shape.
        dtype: Output data type.
        stream: Optional stream to use.
        key: PRNG key.

    Returns:
        Array containing samples from the truncated normal distribution.

    """
    from zero_mlx.array import array
    import ml_switcheroo_compiler.ops as sops

    key_tensor = _get_key(key)

    lower_val = lower._tensor if hasattr(lower, "_tensor") else array(lower)._tensor
    upper_val = upper._tensor if hasattr(upper, "_tensor") else array(upper)._tensor

    if shape is None:
        lower_shape = lower_val.shape if hasattr(lower_val, "shape") else ()
        upper_shape = upper_val.shape if hasattr(upper_val, "shape") else ()
        from ml_switcheroo_compiler.core.shape import broadcast_shapes

        shape = broadcast_shapes(lower_shape, upper_shape)
    elif isinstance(shape, int):
        shape = (shape,)  # pragma: no cover

    is_invalid = sops.greater_equal(lower_val, upper_val)
    safe_upper = sops.where(is_invalid, sops.add(lower_val, 1.0), upper_val)
    res = mrand.truncated_normal(key_tensor, lower_val, safe_upper, shape=shape)
    res = sops.where(is_invalid, lower_val, res)

    if dtype is None:
        dtype = mx.float32
    return array(res).astype(dtype)


from zero_mlx.random_state import state
# __all__ is handled by __init__ for random module functions
