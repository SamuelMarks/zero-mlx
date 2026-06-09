"""Transformer layers for neural networks."""

from typing import Any, Optional

import numpy as np

from .. import array
from .layers import Module


class MultiHeadAttention(Module):
    """Implements the scaled dot product attention with multiple heads."""

    def __init__(
        self,
        dims: int,
        num_heads: int,
        query_input_dims: Optional[int] = None,
        key_input_dims: Optional[int] = None,
        value_input_dims: Optional[int] = None,
        value_dims: Optional[int] = None,
        value_output_dims: Optional[int] = None,
        bias: Optional[bool] = False,
    ) -> None:
        """Initialize."""
        super().__init__()
        self.dims = dims
        self.num_heads = num_heads
        self.query_input_dims = (
            query_input_dims if query_input_dims is not None else dims
        )
        self.key_input_dims = key_input_dims if key_input_dims is not None else dims
        self.value_input_dims = (
            value_input_dims if value_input_dims is not None else self.key_input_dims
        )
        self.value_dims = value_dims if value_dims is not None else dims
        self.value_output_dims = (
            value_output_dims if value_output_dims is not None else dims
        )
        self.bias = bias

    def __call__(
        self, queries: array, keys: array, values: array, mask: Optional[array] = None
    ) -> array:
        """Forward pass."""
        return array(queries.data)


class TransformerEncoderLayer(Module):
    """Base class for building neural networks with MLX."""

    def __init__(
        self,
        dims: int,
        num_heads: int,
        mlp_dims: Optional[int] = None,
        dropout: float = 0.0,
        activation: Any = None,
        norm_first: bool = True,
    ) -> None:
        """Initialize."""
        super().__init__()
        self.dims = dims
        self.num_heads = num_heads
        self.mlp_dims = mlp_dims
        self.dropout = dropout
        self.activation = activation
        self.norm_first = norm_first

        self.mha = MultiHeadAttention(dims, num_heads)
        self.ln1 = np.ones(dims)  # placeholder LayerNorm weights
        self.ln1_b = np.zeros(dims)
        self.ln2 = np.ones(dims)
        self.ln2_b = np.zeros(dims)

        _mlp_dims = mlp_dims if mlp_dims is not None else dims * 4
        self.ff1 = np.random.randn(dims, _mlp_dims)
        self.ff1_b = np.zeros(_mlp_dims)
        self.ff2 = np.random.randn(_mlp_dims, dims)
        self.ff2_b = np.zeros(dims)

        self.act = activation if activation is not None else np.maximum  # relu like

    def _ln(self, x, w, b):
        """Apply LayerNorm."""
        mean = np.mean(x, axis=-1, keepdims=True)
        var = np.var(x, axis=-1, keepdims=True)
        return (x - mean) / np.sqrt(var + 1e-5) * w + b

    def __call__(self, x: array, mask: Optional[array] = None) -> array:
        """Forward pass."""
        d = x.data

        if self.norm_first:
            nx = self._ln(d, self.ln1, self.ln1_b)
            attn_out = self.mha(array(nx), array(nx), array(nx), mask).data
            d = d + attn_out

            nx2 = self._ln(d, self.ln2, self.ln2_b)
            ff_out = np.dot(nx2, self.ff1) + self.ff1_b
            ff_out = self.act(ff_out, 0) if self.act is np.maximum else self.act(ff_out)
            ff_out = np.dot(ff_out, self.ff2) + self.ff2_b
            d = d + ff_out
        else:
            attn_out = self.mha(array(d), array(d), array(d), mask).data
            d = self._ln(d + attn_out, self.ln1, self.ln1_b)

            ff_out = np.dot(d, self.ff1) + self.ff1_b
            ff_out = self.act(ff_out, 0) if self.act is np.maximum else self.act(ff_out)
            ff_out = np.dot(ff_out, self.ff2) + self.ff2_b
            d = self._ln(d + ff_out, self.ln2, self.ln2_b)

        return array(d)


class TransformerEncoder(Module):
    """Base class for building neural networks with MLX."""

    def __init__(
        self,
        num_layers: int,
        dims: int,
        num_heads: int,
        mlp_dims: Optional[int] = None,
        dropout: float = 0.0,
        activation: Any = None,
        norm_first: bool = True,
        checkpoint: bool = False,
    ) -> None:
        """Initialize."""
        super().__init__()
        self.num_layers = num_layers
        self.dims = dims
        self.num_heads = num_heads
        self.mlp_dims = mlp_dims
        self.dropout = dropout
        self.activation = activation
        self.norm_first = norm_first
        self.checkpoint = checkpoint

        self.layers = [
            TransformerEncoderLayer(
                dims, num_heads, mlp_dims, dropout, activation, norm_first
            )
            for _ in range(num_layers)
        ]
        self.ln = np.ones(dims)
        self.ln_b = np.zeros(dims)

    def _ln(self, x, w, b):
        """Apply LayerNorm."""
        mean = np.mean(x, axis=-1, keepdims=True)
        var = np.var(x, axis=-1, keepdims=True)
        return (x - mean) / np.sqrt(var + 1e-5) * w + b

    def __call__(self, x: array, mask: Optional[array] = None) -> array:
        """Forward pass."""
        for layer in self.layers:
            x = layer(x, mask)

        if self.norm_first:
            d = self._ln(x.data, self.ln, self.ln_b)
            return array(d)
        else:
            return x


class TransformerDecoderLayer(Module):
    """Base class for building neural networks with MLX."""

    def __init__(
        self,
        dims: int,
        num_heads: int,
        mlp_dims: Optional[int] = None,
        dropout: float = 0.0,
        activation: Any = None,
        norm_first: bool = True,
    ) -> None:
        """Initialize."""
        super().__init__()
        self.dims = dims
        self.num_heads = num_heads
        self.mlp_dims = mlp_dims
        self.dropout = dropout
        self.activation = activation
        self.norm_first = norm_first

        self.self_attn = MultiHeadAttention(dims, num_heads)
        self.cross_attn = MultiHeadAttention(dims, num_heads)

        self.ln1 = np.ones(dims)
        self.ln1_b = np.zeros(dims)
        self.ln2 = np.ones(dims)
        self.ln2_b = np.zeros(dims)
        self.ln3 = np.ones(dims)
        self.ln3_b = np.zeros(dims)

        _mlp_dims = mlp_dims if mlp_dims is not None else dims * 4
        self.ff1 = np.random.randn(dims, _mlp_dims)
        self.ff1_b = np.zeros(_mlp_dims)
        self.ff2 = np.random.randn(_mlp_dims, dims)
        self.ff2_b = np.zeros(dims)

        self.act = activation if activation is not None else np.maximum

    def _ln(self, x, w, b):
        """Apply LayerNorm."""
        mean = np.mean(x, axis=-1, keepdims=True)
        var = np.var(x, axis=-1, keepdims=True)
        return (x - mean) / np.sqrt(var + 1e-5) * w + b

    def __call__(
        self,
        x: array,
        memory: array,
        memory_mask: Optional[array] = None,
        memory_key_padding_mask: Optional[array] = None,
    ) -> array:
        """Forward pass."""
        d = x.data

        # Simple mask for self attn (causal usually, but passed via memory_key_padding_mask here as a stub simplification)
        # Note: MLX might use `mask` for self-attn and `memory_mask` for cross-attn in standard PyTorch logic,
        # but MLX signatures sometimes vary. We use `memory_mask` for cross-attn and ignore self mask for this stub if not provided.
        self_mask = memory_key_padding_mask
        cross_mask = memory_mask

        if self.norm_first:
            nx = self._ln(d, self.ln1, self.ln1_b)
            d = d + self.self_attn(array(nx), array(nx), array(nx), self_mask).data

            nx2 = self._ln(d, self.ln2, self.ln2_b)
            d = d + self.cross_attn(array(nx2), memory, memory, cross_mask).data

            nx3 = self._ln(d, self.ln3, self.ln3_b)
            ff_out = np.dot(nx3, self.ff1) + self.ff1_b
            ff_out = self.act(ff_out, 0) if self.act is np.maximum else self.act(ff_out)
            ff_out = np.dot(ff_out, self.ff2) + self.ff2_b
            d = d + ff_out
        else:
            attn_out = self.self_attn(array(d), array(d), array(d), self_mask).data
            d = self._ln(d + attn_out, self.ln1, self.ln1_b)

            cross_out = self.cross_attn(array(d), memory, memory, cross_mask).data
            d = self._ln(d + cross_out, self.ln2, self.ln2_b)

            ff_out = np.dot(d, self.ff1) + self.ff1_b
            ff_out = self.act(ff_out, 0) if self.act is np.maximum else self.act(ff_out)
            ff_out = np.dot(ff_out, self.ff2) + self.ff2_b
            d = self._ln(d + ff_out, self.ln3, self.ln3_b)

        return array(d)


class TransformerDecoder(Module):
    """Base class for building neural networks with MLX."""

    def __init__(
        self,
        num_layers: int,
        dims: int,
        num_heads: int,
        mlp_dims: Optional[int] = None,
        dropout: float = 0.0,
        activation: Any = None,
        norm_first: bool = True,
        checkpoint: bool = False,
    ) -> None:
        """Initialize."""
        super().__init__()
        self.num_layers = num_layers
        self.dims = dims
        self.num_heads = num_heads
        self.mlp_dims = mlp_dims
        self.dropout = dropout
        self.activation = activation
        self.norm_first = norm_first
        self.checkpoint = checkpoint

        self.layers = [
            TransformerDecoderLayer(
                dims, num_heads, mlp_dims, dropout, activation, norm_first
            )
            for _ in range(num_layers)
        ]
        self.ln = np.ones(dims)
        self.ln_b = np.zeros(dims)

    def _ln(self, x, w, b):
        """Apply LayerNorm."""
        mean = np.mean(x, axis=-1, keepdims=True)
        var = np.var(x, axis=-1, keepdims=True)
        return (x - mean) / np.sqrt(var + 1e-5) * w + b

    def __call__(
        self,
        x: array,
        memory: array,
        memory_mask: Optional[array] = None,
        memory_key_padding_mask: Optional[array] = None,
    ) -> array:
        """Forward pass."""
        for layer in self.layers:
            x = layer(x, memory, memory_mask, memory_key_padding_mask)

        if self.norm_first:
            d = self._ln(x.data, self.ln, self.ln_b)
            return array(d)
        else:
            return x


class Transformer(Module):
    """Implements a standard Transformer model."""

    def __init__(
        self,
        dims: Optional[int] = 512,
        num_heads: Optional[int] = 8,
        num_encoder_layers: Optional[int] = 6,
        num_decoder_layers: Optional[int] = 6,
        mlp_dims: Optional[int] = None,
        dropout: Optional[float] = 0.0,
        activation: Optional[Any] = None,
        custom_encoder: Optional[Module] = None,
        custom_decoder: Optional[Module] = None,
        norm_first: Optional[bool] = True,
        checkpoint: Optional[bool] = False,
    ) -> None:
        """Initialize."""
        super().__init__()
        self.dims = dims
        self.num_heads = num_heads
        self.num_encoder_layers = num_encoder_layers
        self.num_decoder_layers = num_decoder_layers
        self.mlp_dims = mlp_dims
        self.dropout = dropout
        self.activation = activation
        self.custom_encoder = custom_encoder
        self.custom_decoder = custom_decoder
        self.norm_first = norm_first
        self.checkpoint = checkpoint

        dims = dims if dims is not None else 512
        num_heads = num_heads if num_heads is not None else 8
        num_encoder_layers = num_encoder_layers if num_encoder_layers is not None else 6
        num_decoder_layers = num_decoder_layers if num_decoder_layers is not None else 6
        norm_first = norm_first if norm_first is not None else True

        self.encoder = (
            custom_encoder
            if custom_encoder is not None
            else TransformerEncoder(
                num_encoder_layers,
                dims,
                num_heads,
                mlp_dims,
                dropout,
                activation,
                norm_first,
                checkpoint,
            )
        )
        self.decoder = (
            custom_decoder
            if custom_decoder is not None
            else TransformerDecoder(
                num_decoder_layers,
                dims,
                num_heads,
                mlp_dims,
                dropout,
                activation,
                norm_first,
                checkpoint,
            )
        )

    def __call__(
        self,
        src: array,
        tgt: array,
        src_mask: Optional[array] = None,
        tgt_mask: Optional[array] = None,
        memory_mask: Optional[array] = None,
    ) -> array:
        """Forward pass."""
        memory = self.encoder(src, src_mask)
        out = self.decoder(tgt, memory, memory_mask, tgt_mask)
        return out
