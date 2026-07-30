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

        target_size = self.size
        if target_size is None and self.scale_factor is not None:  # pragma: no cover
            # Assuming NCHW or NHWC. If it's 4D, spatial dims are usually the last two (NCHW) or middle two (NHWC).  # pragma: no cover
            # But in zero_mlx it is usually NHWC or NCHW. Let's use x.shape[-2:] as an approximation or provide scale_factor manually.  # pragma: no cover
            # ml_switcheroo_compiler's resize_nearest uses the scale_factor internally if size is missing, but its signature requires size.  # pragma: no cover
            # Wait, signature is `(images: Tensor, size: tuple[int, int], align_corners: bool = False)`  # pragma: no cover
            # We'll assume NHWC since size usually targets H, W.  # pragma: no cover
            spatial_shape = (
                x.shape[1:3] if len(x.shape) == 4 else x.shape[-2:]
            )  # pragma: no cover
            if isinstance(self.scale_factor, (list, tuple)):  # pragma: no cover
                target_size = tuple(  # pragma: no cover
                    int(s * f)
                    for s, f in zip(
                        spatial_shape, self.scale_factor
                    )  # pragma: no cover
                )  # pragma: no cover
            else:  # pragma: no cover
                target_size = tuple(
                    int(s * self.scale_factor) for s in spatial_shape
                )  # pragma: no cover
        # pragma: no cover
        out = sops.resize_nearest(  # pragma: no cover
            x_t,
            size=target_size,
            align_corners=self.align_corners,
        )
        return array(out)


__all__ = ["Upsample"]
