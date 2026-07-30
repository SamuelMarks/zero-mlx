"""mlx.nn.positional_encoding module."""

from typing import Optional, Any
from zero_mlx.nn.base import Module
from zero_mlx.array import array
import zero_mlx as mx
import math


class ALiBi(Module):
    """ALiBi positional encoding."""

    def __call__(self, x: array) -> array:
        """Call."""
        # Not a complete implementation, just identity for now.
        return x


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
        seq_len = x.shape[-2]
        position = mx.arange(offset, offset + seq_len)
        div_term = mx.exp(
            mx.arange(0, self.dims, 2) * (-math.log(self.base) / self.dims)
        )
        pe = mx.matmul(mx.expand_dims(position, 1), mx.expand_dims(div_term, 0))

        sin_pe = mx.sin(pe)
        cos_pe = mx.cos(pe)

        x_r = x[..., 0::2]
        x_i = x[..., 1::2]

        out_r = mx.subtract(mx.multiply(x_r, cos_pe), mx.multiply(x_i, sin_pe))
        out_i = mx.add(mx.multiply(x_r, sin_pe), mx.multiply(x_i, cos_pe))

        out = mx.concatenate(
            [mx.expand_dims(out_r, -1), mx.expand_dims(out_i, -1)], axis=-1
        )
        out = mx.reshape(out, x.shape)

        return out


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
        seq_len = x.shape[1] if x.ndim > 1 else x.shape[0]
        position = mx.arange(seq_len)
        div_term = mx.exp(
            mx.arange(0, self.dims, 2) * (-math.log(1.0 / self.min_freq) / self.dims)
        )
        pe = mx.matmul(mx.expand_dims(position, 1), mx.expand_dims(div_term, 0))

        sin_pe = mx.sin(pe)
        cos_pe = mx.cos(pe)

        if self.cos_first:
            out = mx.concatenate(  # pragma: no cover
                [mx.expand_dims(cos_pe, -1), mx.expand_dims(sin_pe, -1)], axis=-1
            )
        else:
            out = mx.concatenate(
                [mx.expand_dims(sin_pe, -1), mx.expand_dims(cos_pe, -1)], axis=-1
            )

        out = mx.reshape(out, (seq_len, self.dims))
        return mx.add(x, out)


__all__ = ["ALiBi", "RoPE", "SinusoidalPositionalEncoding"]
