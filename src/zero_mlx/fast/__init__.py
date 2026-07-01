"""mlx.core.fast: fast operations"""

from typing import Any


def cuda_kernel(*args: Any, **kwargs: Any) -> Any:  # pragma: no cover
    """Execute a CUDA kernel.

    Args:
        *args: Variable length argument list.
        **kwargs: Arbitrary keyword arguments.

    Returns:
        Any: The result of the kernel execution.

    """
    import ml_switcheroo_compiler.ops as sops

    return sops.cuda_kernel(*args, **kwargs)


def layer_norm(*args: Any, **kwargs: Any) -> Any:  # pragma: no cover
    """Apply layer normalization.

    Args:
        *args: Variable length argument list.
        **kwargs: Arbitrary keyword arguments.

    Returns:
        Any: The result of the layer normalization.

    """
    from zero_mlx.array import array
    import ml_switcheroo_compiler.ops as sops

    x = args[0] if len(args) > 0 else kwargs.get("x")
    if not hasattr(x, "_tensor"):
        x = array(x)
    weight = args[1] if len(args) > 1 else kwargs.get("weight")
    bias = args[2] if len(args) > 2 else kwargs.get("bias")
    eps = args[3] if len(args) > 3 else kwargs.get("eps", 1e-5)
    w_t = weight._tensor if weight is not None else None
    b_t = bias._tensor if bias is not None else None
    return array(
        sops.nn.layer_norm(
            x._tensor, [x.shape[-1]] if len(x.shape) > 0 else [], w_t, b_t, epsilon=eps
        )
    )


def metal_kernel(*args: Any, **kwargs: Any) -> Any:  # pragma: no cover
    """Execute a Metal kernel.

    Args:
        *args: Variable length argument list.
        **kwargs: Arbitrary keyword arguments.

    Returns:
        Any: The result of the kernel execution.

    """
    import ml_switcheroo_compiler.ops as sops

    return sops.metal_kernel(*args, **kwargs)


def precompiled_cuda_kernel(*args: Any, **kwargs: Any) -> Any:  # pragma: no cover
    """Execute a precompiled CUDA kernel.

    Args:
        *args: Variable length argument list.
        **kwargs: Arbitrary keyword arguments.

    Returns:
        Any: The result of the kernel execution.

    """
    import ml_switcheroo_compiler.ops as sops

    return sops.precompiled_cuda_kernel(*args, **kwargs)


def rms_norm(*args: Any, **kwargs: Any) -> Any:  # pragma: no cover
    """Apply RMS normalization.

    Args:
        *args: Variable length argument list.
        **kwargs: Arbitrary keyword arguments.

    Returns:
        Any: The result of the RMS normalization.

    """
    from zero_mlx.array import array
    import ml_switcheroo_compiler.ops as sops

    x = args[0] if len(args) > 0 else kwargs.get("x")
    if not hasattr(x, "_tensor"):
        x = array(x)
    weight = args[1] if len(args) > 1 else kwargs.get("weight")
    eps = args[2] if len(args) > 2 else kwargs.get("eps", 1e-5)
    return (
        array(
            sops.nn.rms_norm(
                x._tensor, weight._tensor if weight is not None else None, epsilon=eps
            )
        )
        if len(x.shape) > 0
        else array(0.0)
    )


def rope(*args: Any, **kwargs: Any) -> Any:  # pragma: no cover
    """Apply rotary positional embedding.

    Args:
        *args: Variable length argument list.
        **kwargs: Arbitrary keyword arguments.

    Returns:
        Any: The result of the RoPE application.

    """
    from zero_mlx.array import array
    import ml_switcheroo_compiler.ops as sops

    a = args[0] if len(args) > 0 else kwargs.get("a")
    if not hasattr(a, "_tensor"):
        a = array(a)
    dims = args[1] if len(args) > 1 else kwargs.get("dims")
    base = kwargs.get("base", 10000.0)
    offset = kwargs.get("offset", 0)
    # The compiler's rope expects (input, dim, base, offset).
    return array(sops.nn.rope(a._tensor, dim=dims, base=base, offset=offset))


def scaled_dot_product_attention(*args: Any, **kwargs: Any) -> Any:  # pragma: no cover
    """Compute scaled dot product attention.

    Args:
        *args: Variable length argument list.
        **kwargs: Arbitrary keyword arguments.

    Returns:
        Any: The result of the attention computation.

    Raises:
        NotImplementedError: Always raised as this is a stub.

    """
    from zero_mlx.array import array
    import ml_switcheroo_compiler.ops as sops

    q = args[0] if len(args) > 0 else kwargs.get("q")
    k = args[1] if len(args) > 1 else kwargs.get("k", q)
    v = args[2] if len(args) > 2 else kwargs.get("v", q)
    if k is None:
        k = q
    if v is None:
        v = q
    if not hasattr(q, "_tensor"):
        q = array(q)
    if not hasattr(k, "_tensor"):
        k = array(k)
    if not hasattr(v, "_tensor"):
        v = array(v)
    scale = args[3] if len(args) > 3 else kwargs.get("scale", None)
    mask = args[4] if len(args) > 4 else kwargs.get("mask", None)
    m_t = mask._tensor if mask is not None else None
    return (
        array(
            sops.nn.dot_product_attention(
                q._tensor,
                k._tensor,
                v._tensor,
                config=sops.nn.DotProductAttentionConfig(scale=scale, mask=m_t),
            )
        )
        if len(q.shape) > 0
        else array(0.0)
    )


__all__ = [
    "cuda_kernel",
    "layer_norm",
    "metal_kernel",
    "precompiled_cuda_kernel",
    "rms_norm",
    "rope",
    "scaled_dot_product_attention",
]
