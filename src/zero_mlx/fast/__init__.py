"""mlx.core.fast: fast operations"""

from typing import Any


def cuda_kernel(*args, **kwargs) -> Any:
    raise NotImplementedError("fast.cuda_kernel is not implemented")


def layer_norm(*args, **kwargs) -> Any:
    raise NotImplementedError("fast.layer_norm is not implemented")


def metal_kernel(*args, **kwargs) -> Any:
    raise NotImplementedError("fast.metal_kernel is not implemented")


def precompiled_cuda_kernel(*args, **kwargs) -> Any:
    raise NotImplementedError("fast.precompiled_cuda_kernel is not implemented")


def rms_norm(*args, **kwargs) -> Any:
    raise NotImplementedError("fast.rms_norm is not implemented")


def rope(*args, **kwargs) -> Any:
    raise NotImplementedError("fast.rope is not implemented")


def scaled_dot_product_attention(*args, **kwargs) -> Any:
    raise NotImplementedError("fast.scaled_dot_product_attention is not implemented")


__all__ = [
    "cuda_kernel",
    "layer_norm",
    "metal_kernel",
    "precompiled_cuda_kernel",
    "rms_norm",
    "rope",
    "scaled_dot_product_attention",
]
