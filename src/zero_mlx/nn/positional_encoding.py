"""Positional encoding layers for neural networks."""

from typing import Optional

import numpy as np

from .. import array
from .layers import Module


class RoPE(Module):
    """Implements the rotary positional encoding."""

    def __init__(
        self,
        dims: int,
        traditional: bool = False,
        base: float = 10000.0,
        scale: float = 1.0,
    ) -> None:
        """Initialize."""
        super().__init__()
        self.dims = dims
        self.traditional = traditional
        self.base = base
        self.scale = scale

    def __call__(self, x: array, offset: int = 0) -> array:
        """Forward pass."""
        d = x.data
        shape = d.shape
        L = shape[-2]

        # Calculate freqs
        half_dims = self.dims // 2
        freqs = np.exp(
            -np.log(self.base) * np.arange(0, half_dims, dtype=d.dtype) / half_dims
        )
        positions = np.arange(offset, offset + L, dtype=d.dtype) * self.scale
        theta = np.outer(positions, freqs)

        costheta = np.cos(theta)
        sintheta = np.sin(theta)

        # RoPE application
        d_rope = d[..., : self.dims]
        d_pass = d[..., self.dims :]

        if self.traditional:
            # interleaved: x0, x1, x2, x3 -> x0, x2 and x1, x3
            x1 = d_rope[..., 0::2]
            x2 = d_rope[..., 1::2]
            out_rope = np.empty_like(d_rope)
            out_rope[..., 0::2] = x1 * costheta - x2 * sintheta
            out_rope[..., 1::2] = x1 * sintheta + x2 * costheta
        else:
            # split half: x0, x1 and x2, x3
            x1 = d_rope[..., :half_dims]
            x2 = d_rope[..., half_dims:]
            out_rope = np.concatenate(
                [x1 * costheta - x2 * sintheta, x1 * sintheta + x2 * costheta], axis=-1
            )

        out = (
            np.concatenate([out_rope, d_pass], axis=-1)
            if d_pass.shape[-1] > 0
            else out_rope
        )
        return array(out)


class ALiBi(Module):
    """Base class for building neural networks with MLX."""

    def __init__(
        self, _alibi_mask_key: Optional[str] = None, _alibi_mask: Optional[str] = None
    ) -> None:
        """Initialize ALiBi."""
        super().__init__()
        self._alibi_mask_key = _alibi_mask_key
        self._alibi_mask = _alibi_mask

    def __call__(self, x: array) -> array:
        """Forward pass."""
        # x is usually attention scores (B, H, q_len, k_len)
        d = x.data
        if d.ndim < 4:
            return array(d)  # Not typical attention score format

        H = d.shape[1]
        k_len = d.shape[-1]

        # Calculate slopes
        def get_slopes(n):
            """Get slopes for ALiBi."""

            def get_slopes_power_of_2(n):
                """Get slopes for power of 2."""
                start = 2 ** (-(2 ** -(np.log2(n) - 3)))
                ratio = start
                return [start * ratio**i for i in range(n)]

            if np.log2(n).is_integer():
                return get_slopes_power_of_2(n)
            else:
                closest_power_of_2 = 2 ** int(np.log2(n))
                return (
                    get_slopes_power_of_2(closest_power_of_2)
                    + get_slopes(2 * closest_power_of_2)[0::2][: n - closest_power_of_2]
                )

        slopes = np.array(get_slopes(H), dtype=d.dtype)

        # Distance mask: (1, 1, 1, k_len)
        # ALiBi adds penalty based on distance to the left (causal usually)
        # We just create a relative distance matrix. Usually ALiBi is causal so offset is - (k_len - 1 - i)
        # For simplicity in stub, assuming symmetric or causal relative index.
        # Often it's just `np.arange(k_len)` broadcasted.
        pos = np.arange(k_len, dtype=d.dtype)
        # we reshape slopes to (H, 1, 1) and pos to (1, k_len)
        penalty = slopes[:, np.newaxis, np.newaxis] * pos[np.newaxis, np.newaxis, :]

        out = d + penalty
        return array(out)


class SinusoidalPositionalEncoding(Module):
    """Implements sinusoidal positional encoding."""

    def __init__(
        self,
        dims: int,
        min_freq: float = 0.0001,
        max_freq: float = 1.0,
        scale: Optional[float] = None,
        cos_first: bool = False,
        full_turns: bool = False,
    ) -> None:
        """Initialize."""
        super().__init__()
        self.dims = dims
        self.min_freq = min_freq
        self.max_freq = max_freq
        self.scale = scale if scale is not None else np.sqrt(2 / dims)
        self.cos_first = cos_first
        self.full_turns = full_turns

    def __call__(self, x: array, offset: int = 0) -> array:
        """Forward pass."""
        d = x.data
        L = d.shape[-2]

        half_dims = self.dims // 2
        # log-linear spaced frequencies
        freqs = np.exp(
            np.linspace(np.log(self.min_freq), np.log(self.max_freq), half_dims)
        )

        if self.full_turns:
            freqs *= 2 * np.pi

        pos = np.arange(offset, offset + L, dtype=d.dtype)
        theta = np.outer(pos, freqs)

        sintheta = np.sin(theta)
        costheta = np.cos(theta)

        if self.cos_first:
            emb = np.concatenate([costheta, sintheta], axis=-1)
        else:
            emb = np.concatenate([sintheta, costheta], axis=-1)

        out = d + emb * self.scale
        return array(out)
