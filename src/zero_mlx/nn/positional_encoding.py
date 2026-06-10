"""Positional encoding layers for neural networks."""

from typing import Optional
from .. import array
from .layers import Module


class RoPE(Module):
    def __init__(
        self,
        dims: int,
        traditional: bool = False,
        base: float = 10000.0,
        scale: float = 1.0,
    ) -> None:
        pass

    def __call__(self, x: array, offset: int = 0) -> array:
        return x


class ALiBi(Module):
    def __init__(
        self, _alibi_mask_key: Optional[str] = None, _alibi_mask: Optional[str] = None
    ) -> None:
        pass

    def __call__(self, x: array) -> array:
        return x


class SinusoidalPositionalEncoding(Module):
    def __init__(
        self,
        dims: int,
        min_freq: float = 0.0001,
        max_freq: float = 1.0,
        scale: Optional[float] = None,
        cos_first: bool = False,
        full_turns: bool = False,
    ) -> None:
        pass

    def __call__(self, x: array, offset: int = 0) -> array:
        return x
