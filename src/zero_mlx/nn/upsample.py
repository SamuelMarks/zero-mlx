"""mlx.nn.upsample module."""

from typing import Union, Tuple, Optional
from zero_mlx.nn.base import Module
from zero_mlx.array import array
import ml_switcheroo_compiler.ops as sops


class Upsample(Module):
    """Upsamples a given multi-channel data."""

    def __init__(
        self,
        scale_factor: Union[float, Tuple[float, ...]] = None,
        size: Union[int, Tuple[int, ...]] = None,
        mode: str = "nearest",
        align_corners: bool = False,
    ):
        """Initialize Upsample."""
        super().__init__()
        self.scale_factor = scale_factor
        self.size = size
        self.mode = mode
        self.align_corners = align_corners

    def __call__(self, x: array) -> array:
        """Call."""
        x_t = x._tensor if hasattr(x, "_tensor") else x
        out = sops.upsample(
            x_t,
            scale_factor=self.scale_factor,
            size=self.size,
            mode=self.mode,
            align_corners=self.align_corners,
        )
        return array(out)


__all__ = ["Upsample"]
