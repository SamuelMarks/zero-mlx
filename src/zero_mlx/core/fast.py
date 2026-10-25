"""Fast fused operations submodule (`mlx.core.fast` / `zero_mlx.core.fast`)."""

from __future__ import annotations

from zero_mlx.fast import (
    KernelCallable,
    cuda_kernel,
    layer_norm,
    metal_kernel,
    rms_norm,
    rope,
    scaled_dot_product_attention,
)

__all__ = [
    "KernelCallable",
    "cuda_kernel",
    "layer_norm",
    "metal_kernel",
    "rms_norm",
    "rope",
    "scaled_dot_product_attention",
]
