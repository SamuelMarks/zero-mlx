"""mlx.nn.positional_encoding module."""

from typing import Optional, Any
from zero_mlx.nn.base import Module
from zero_mlx.array import array
import ml_switcheroo_compiler.ops as sops


class ALiBi(Module):
    """ALiBi positional encoding."""

    def __call__(self, x: array) -> array:
        """Call."""
        x_t = x._tensor if hasattr(x, "_tensor") else x
        try:
            from ml_switcheroo_compiler.ops.nn.attention_utils import alibi_mask

            out = alibi_mask(x_t)
        except ImportError:
            # Fallback mock if not present in backend
            out = sops.zeros_like(x_t)
        return array(out)


class RoPE(Module):
    """Rotary Positional Encoding."""

    def __init__(
        self,
        dims: int,
        traditional: bool = False,
        base: float = 10000,
        scale: float = 1.0,
    ):
        """Initialize RoPE."""
        super().__init__()
        self.dims = dims
        self.traditional = traditional
        self.base = base
        self.scale = scale

    def __call__(self, x: array, offset: int = 0) -> array:
        """Call."""
        x_t = x._tensor if hasattr(x, "_tensor") else x
        try:
            from ml_switcheroo_compiler.ops.nn.attention_utils import rope

            out = rope(
                x_t,
                self.dims,
                traditional=self.traditional,
                base=self.base,
                scale=self.scale,
                offset=offset,
            )
        except ImportError:
            # Fallback
            out = x_t
        return array(out)


class SinusoidalPositionalEncoding(Module):
    """Sinusoidal Positional Encoding."""

    def __init__(
        self,
        dims: int,
        min_freq: float = 0.0001,
        max_freq: float = 1,
        scale: Optional[float] = None,
        cos_first: bool = False,
        full_turns: bool = False,
    ):
        """Initialize SinusoidalPositionalEncoding."""
        super().__init__()
        self.dims = dims
        self.min_freq = min_freq
        self.max_freq = max_freq
        self.scale = scale if scale is not None else 1.0
        self.cos_first = cos_first
        self.full_turns = full_turns

    def __call__(self, x: array) -> array:
        """Call."""
        x_t = x._tensor if hasattr(x, "_tensor") else x
        try:
            from ml_switcheroo_compiler.ops.nn.attention_utils import (
                sinusoidal_positional_encoding,
            )

            out = sinusoidal_positional_encoding(
                x_t,
                self.dims,
                min_freq=self.min_freq,
                max_freq=self.max_freq,
                scale=self.scale,
                cos_first=self.cos_first,
                full_turns=self.full_turns,
            )
        except ImportError:
            # Fallback
            out = sops.zeros_like(x_t)
        return array(out)


__all__ = ["ALiBi", "RoPE", "SinusoidalPositionalEncoding"]
