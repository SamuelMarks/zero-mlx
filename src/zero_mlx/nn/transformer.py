"""Transformer layers for neural networks."""

from typing import Any, Optional
import numpy as np
from .. import array, _to_tensor, _wrap
from .layers import Module


class MultiHeadAttention(Module):
    def __init__(
        self,
        dims: int,
        num_heads: int,
        query_input_dims: Optional[int] = None,
        key_input_dims: Optional[int] = None,
        value_input_dims: Optional[int] = None,
        value_dims: Optional[int] = None,
        value_out_dims: Optional[int] = None,
        bias: bool = False,
    ) -> None:
        self.dims = dims
        self.num_heads = num_heads

    def __call__(
        self, queries: array, keys: array, values: array, mask: Optional[array] = None
    ) -> array:
        from ml_switcheroo.core.config import config

        if config.eager_mode:
            data = np.array(_to_tensor(queries).data)
            return _wrap(_to_tensor(np.zeros_like(data)))
        return queries


class TransformerEncoderLayer(Module):
    def __init__(
        self,
        dims: int,
        num_heads: int,
        mlp_dims: Optional[int] = None,
        norm_first: bool = False,
    ) -> None:
        pass

    def __call__(self, x: array, mask: Optional[array] = None) -> array:
        return x


class TransformerEncoder(Module):
    def __init__(
        self,
        num_layers: int,
        dims: int,
        num_heads: int,
        mlp_dims: Optional[int] = None,
        norm_first: bool = False,
    ) -> None:
        pass

    def __call__(self, x: array, mask: Optional[array] = None) -> array:
        return x


class TransformerDecoderLayer(Module):
    def __init__(
        self,
        dims: int,
        num_heads: int,
        mlp_dims: Optional[int] = None,
        norm_first: bool = False,
    ) -> None:
        pass

    def __call__(
        self,
        x: array,
        memory: array,
        x_mask: Optional[array] = None,
        memory_mask: Optional[array] = None,
    ) -> array:
        return x


class TransformerDecoder(Module):
    def __init__(
        self,
        num_layers: int,
        dims: int,
        num_heads: int,
        mlp_dims: Optional[int] = None,
        norm_first: bool = False,
    ) -> None:
        pass

    def __call__(
        self,
        x: array,
        memory: array,
        x_mask: Optional[array] = None,
        memory_mask: Optional[array] = None,
    ) -> array:
        return x


class Transformer(Module):
    def __init__(
        self,
        dims: int,
        num_heads: int,
        num_encoder_layers: int,
        num_decoder_layers: int,
        mlp_dims: Optional[int] = None,
        norm_first: bool = False,
        custom_encoder: Optional[Any] = None,
        custom_decoder: Optional[Any] = None,
    ) -> None:
        pass

    def __call__(
        self,
        src: array,
        tgt: array,
        src_mask: Optional[array] = None,
        tgt_mask: Optional[array] = None,
        memory_mask: Optional[array] = None,
    ) -> array:
        return tgt
