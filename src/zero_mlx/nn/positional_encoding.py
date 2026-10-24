"""Positional encoding layers (`ALiBi`, `RoPE`, `SinusoidalPositionalEncoding`) for `mlx.nn`."""

from __future__ import annotations

import math

from zero_mlx.array import array
from zero_mlx.dtypes import float32
from zero_mlx.nn.base import Module
import zero_mlx.ops as mx


class ALiBi(Module):
    """Attention with Linear Biases (`ALiBi`) positional encoding."""

    @staticmethod
    def create_alibi_slopes(num_heads: int) -> array:
        """Compute the head-specific geometric slope array for `num_heads`.

        Args:
            num_heads: Number of attention heads.

        Returns:
            Slope `array` of shape `(num_heads, 1, 1)`.

        """
        closest_power = 2 ** math.floor(math.log2(max(1, num_heads)))
        base = 2.0 ** (-(2.0 ** -(math.log2(closest_power) - 3.0)))
        slopes: list[float] = [base ** (i + 1) for i in range(closest_power)]
        extra_base = 2.0 ** (-(2.0 ** -(math.log2(2 * closest_power) - 3.0)))
        num_rem = min(closest_power, num_heads - closest_power)
        for i in range(num_rem):
            slopes.append(extra_base ** (2 * i + 1))
        return array(slopes, dtype=float32).reshape((num_heads, 1, 1))

    @staticmethod
    def create_alibi_matrix(
        q_sequence_length: int,
        k_sequence_length: int,
        num_heads: int,
        offset: int = 0,
    ) -> array:
        """Create the `(num_heads, q_sequence_length, k_sequence_length)` ALiBi bias matrix.

        Args:
            q_sequence_length: Query sequence length.
            k_sequence_length: Key sequence length.
            num_heads: Number of attention heads.
            offset: Query position offset.

        Returns:
            ALiBi bias `array` of shape `(num_heads, q_sequence_length, k_sequence_length)`.

        """
        q_idx = mx.expand_dims(mx.arange(offset, offset + q_sequence_length, dtype=float32), axis=1)
        k_idx = mx.expand_dims(mx.arange(0, k_sequence_length, dtype=float32), axis=0)
        rel_dist = -mx.abs(q_idx - k_idx)
        slopes = ALiBi.create_alibi_slopes(num_heads)
        return mx.expand_dims(rel_dist, axis=0) * slopes

    def __call__(
        self,
        attention_scores: array,
        offset: int = 0,
        mask: array | None = None,
    ) -> array:
        """Add ALiBi positional bias (and optional `mask`) to `attention_scores`.

        Args:
            attention_scores: Attention logits of shape `(..., H, L_q, L_k)` or `(L_q, L_k)`.
            offset: Query position offset.
            mask: Optional additive attention mask.

        Returns:
            Biased attention score `array`.

        """
        if attention_scores.ndim < 3:
            return attention_scores + mask if mask is not None else attention_scores
        num_heads = attention_scores.shape[-3]
        q_len = attention_scores.shape[-2]
        k_len = attention_scores.shape[-1]
        alibi_bias = self.create_alibi_matrix(q_len, k_len, num_heads, offset=offset)
        out = attention_scores + alibi_bias
        return (out + mask) if mask is not None else out


class RoPE(Module):
    """Rotary Positional Encoding (`RoPE`) layer."""

    def __init__(
        self,
        dims: int,
        traditional: bool = False,
        base: float = 10000.0,
        scale: float = 1.0,
    ) -> None:
        """Initialize `RoPE`.

        Args:
            dims: Number of feature dimensions to rotate.
            traditional: Whether to rotate consecutive pairs rather than first/second halves.
            base: Exponential frequency base.
            scale: Position scaling factor.

        """
        super().__init__()
        self.dims: int = dims
        self.traditional: bool = traditional
        self.base: float = base
        self.scale: float = scale

    def __call__(self, x: array, offset: int = 0) -> array:
        """Apply Rotary Positional Encoding to `x`.

        Args:
            x: Input `array` of shape `(..., T, D)`.
            offset: Starting sequence position offset.

        Returns:
            Rotated `array` of shape `x.shape`.

        """
        seq_len = x.shape[-2]
        position = mx.arange(offset, offset + seq_len, dtype=float32) * self.scale
        div_term = mx.exp(
            mx.arange(0, self.dims, 2, dtype=float32) * (-math.log(self.base) / float(self.dims))
        )
        pe = mx.matmul(mx.expand_dims(position, axis=1), mx.expand_dims(div_term, axis=0))

        sin_pe = mx.sin(pe)
        cos_pe = mx.cos(pe)

        x_rot = x[..., : self.dims]
        x_pass = x[..., self.dims :] if self.dims < x.shape[-1] else None

        if self.traditional:
            x_r = x_rot[..., 0::2]
            x_i = x_rot[..., 1::2]
            out_r = mx.subtract(mx.multiply(x_r, cos_pe), mx.multiply(x_i, sin_pe))
            out_i = mx.add(mx.multiply(x_r, sin_pe), mx.multiply(x_i, cos_pe))
            out = mx.concatenate(
                [mx.expand_dims(out_r, axis=-1), mx.expand_dims(out_i, axis=-1)], axis=-1
            )
            out = mx.reshape(out, x_rot.shape)
        else:
            half = self.dims // 2
            x1 = x_rot[..., :half]
            x2 = x_rot[..., half:]
            out_1 = mx.subtract(mx.multiply(x1, cos_pe), mx.multiply(x2, sin_pe))
            out_2 = mx.add(mx.multiply(x1, sin_pe), mx.multiply(x2, cos_pe))
            out = mx.concatenate([out_1, out_2], axis=-1)

        if x_pass is not None:
            return mx.concatenate([out, x_pass], axis=-1)
        return out


class SinusoidalPositionalEncoding(Module):
    """Fixed Sinusoidal Positional Encoding layer."""

    def __init__(
        self,
        dims: int,
        min_freq: float = 0.0001,
        max_freq: float = 1.0,
        scale: float | None = None,
        cos_first: bool = False,
        full_turns: bool = False,
    ) -> None:
        """Initialize `SinusoidalPositionalEncoding`.

        Args:
            dims: Embedding dimensionality (must be even).
            min_freq: Minimum frequency.
            max_freq: Maximum frequency.
            scale: Optional multiplicative scale (defaults to `sqrt(2 / dims)` or `1.0`).
            cos_first: If True, place `cos` channels before `sin` channels.
            full_turns: If True, multiply frequencies by `2 * pi`.

        """
        super().__init__()
        self.dims: int = dims
        self.min_freq: float = min_freq
        self.max_freq: float = max_freq
        self.scale: float = scale if scale is not None else 1.0
        self.cos_first: bool = cos_first
        self.full_turns: bool = full_turns

    def __call__(self, x: array) -> array:
        """Compute sinusoidal positional encoding for positions or add to sequence `x`.

        Args:
            x: Input sequence `array` of shape `(..., L, dims)` or 1D position indices.

        Returns:
            Positionally encoded `array`.

        """
        seq_len = x.shape[1] if x.ndim > 1 else x.shape[0]
        position = mx.arange(0, seq_len, dtype=float32)
        if self.full_turns:
            position = position * (2.0 * math.pi)
        div_term = (
            mx.exp(
                mx.arange(0, self.dims, 2, dtype=float32)
                * (-math.log(self.max_freq / self.min_freq) / float(self.dims))
            )
            * self.max_freq
        )
        pe = mx.matmul(mx.expand_dims(position, axis=1), mx.expand_dims(div_term, axis=0))

        sin_pe = mx.sin(pe)
        cos_pe = mx.cos(pe)

        if self.cos_first:
            out = mx.concatenate(
                [mx.expand_dims(cos_pe, axis=-1), mx.expand_dims(sin_pe, axis=-1)],
                axis=-1,
            )
        else:
            out = mx.concatenate(
                [mx.expand_dims(sin_pe, axis=-1), mx.expand_dims(cos_pe, axis=-1)],
                axis=-1,
            )

        out = mx.reshape(out, (seq_len, self.dims)) * self.scale
        if x.ndim >= 2 and x.shape[-1] == self.dims:
            return mx.add(x, out)
        return out


__all__ = ["ALiBi", "RoPE", "SinusoidalPositionalEncoding"]
