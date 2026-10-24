"""Transformer attention, encoder, and decoder modules (`mlx.nn.transformer`) for zero_mlx."""

from __future__ import annotations

import math
from typing import Callable

from zero_mlx.array import array
from zero_mlx.nn.activations import relu
from zero_mlx.nn.base import Module
from zero_mlx.nn.dropout import Dropout
from zero_mlx.nn.linear import Linear
from zero_mlx.nn.normalization import LayerNorm
import zero_mlx.ops as mx


class MultiHeadAttention(Module):
    """Computes scaled dot-product attention across multiple parallel heads."""

    def __init__(
        self,
        dims: int,
        num_heads: int,
        query_input_dims: int | None = None,
        key_input_dims: int | None = None,
        value_input_dims: int | None = None,
        value_dims: int | None = None,
        value_output_dims: int | None = None,
        bias: bool = False,
    ) -> None:
        """Initialize `MultiHeadAttention`.

        Args:
            dims: Model query/key total dimensionality.
            num_heads: Number of attention heads.
            query_input_dims: Input dimension for queries (defaults to `dims`).
            key_input_dims: Input dimension for keys (defaults to `dims`).
            value_input_dims: Input dimension for values (defaults to `key_input_dims`).
            value_dims: Internal value projection dimension (defaults to `dims`).
            value_output_dims: Final output projection dimension (defaults to `dims`).
            bias: Whether linear projections include additive bias.

        Raises:
            ValueError: If `dims` or `value_dims` is not divisible by `num_heads`.

        """
        super().__init__()
        if dims % num_heads != 0:
            raise ValueError(f"dims ({dims}) must be divisible by num_heads ({num_heads})")
        self.dims: int = dims
        self.num_heads: int = num_heads
        q_in = query_input_dims if query_input_dims is not None else dims
        k_in = key_input_dims if key_input_dims is not None else dims
        v_in = value_input_dims if value_input_dims is not None else k_in
        v_dims = value_dims if value_dims is not None else dims
        v_out = value_output_dims if value_output_dims is not None else dims
        if v_dims % num_heads != 0:
            raise ValueError(f"value_dims ({v_dims}) must be divisible by num_heads ({num_heads})")

        self.query_proj: array = mx.zeros((q_in, dims))
        self.key_proj: array = mx.zeros((k_in, dims))
        self.value_proj: array = mx.zeros((v_in, v_dims))
        self.out_proj: array = mx.zeros((v_dims, v_out))
        self.bias: bool = bias

    @staticmethod
    def create_additive_causal_mask(n: int) -> array:
        """Create an `n x n` upper-triangular additive causal attention mask (`0` and `-inf`).

        Args:
            n: Sequence length.

        Returns:
            Additive causal mask `array` of shape `(n, n)`.

        """
        indices = mx.arange(0, n)
        mask = mx.expand_dims(indices, axis=1) < mx.expand_dims(indices, axis=0)
        return mx.where(mask, float("-inf"), 0.0)

    def __call__(
        self,
        queries: array,
        keys: array,
        values: array,
        mask: array | None = None,
    ) -> array:
        """Compute multi-head attention over `(queries, keys, values)`.

        Args:
            queries: Query tensor of shape `(B, L, D_q)`.
            keys: Key tensor of shape `(B, S, D_k)`.
            values: Value tensor of shape `(B, S, D_v)`.
            mask: Optional additive mask of shape broadcastable to `(B, H, L, S)`.

        Returns:
            Attention output `array` of shape `(B, L, value_output_dims)`.

        """
        b_sz, l_len, _ = queries.shape
        _, s_len, _ = keys.shape

        q = mx.matmul(queries, self.query_proj)
        k = mx.matmul(keys, self.key_proj)
        v = mx.matmul(values, self.value_proj)

        q = mx.reshape(q, (b_sz, l_len, self.num_heads, -1))
        k = mx.reshape(k, (b_sz, s_len, self.num_heads, -1))
        v = mx.reshape(v, (b_sz, s_len, self.num_heads, -1))

        q = mx.transpose(q, (0, 2, 1, 3))
        k = mx.transpose(k, (0, 2, 3, 1))
        v = mx.transpose(v, (0, 2, 1, 3))

        scale = math.sqrt(float(q.shape[-1]))
        scores = mx.divide(mx.matmul(q, k), scale)
        if mask is not None:
            scores = mx.add(scores, mask)

        attn = mx.softmax(scores, axis=-1)
        out = mx.matmul(attn, v)
        out = mx.transpose(out, (0, 2, 1, 3))
        out = mx.reshape(out, (b_sz, l_len, -1))
        return mx.matmul(out, self.out_proj)


class TransformerEncoderLayer(Module):
    """A single Transformer encoder layer with self-attention and feed-forward network."""

    def __init__(
        self,
        dims: int,
        num_heads: int,
        mlp_dims: int | None = None,
        dropout: float = 0.0,
        activation: Callable[[array], array] = relu,
        norm_first: bool = True,
    ) -> None:
        """Initialize `TransformerEncoderLayer`.

        Args:
            dims: Model feature dimension.
            num_heads: Number of attention heads.
            mlp_dims: Hidden dimension of the feed-forward MLP (defaults to `4 * dims`).
            dropout: Dropout probability.
            activation: Activation function in the MLP block.
            norm_first: If True, apply pre-LayerNorm; otherwise post-LayerNorm.

        """
        super().__init__()
        self.dims: int = dims
        ff_dims: int = mlp_dims if mlp_dims is not None else dims * 4
        self.attention: MultiHeadAttention = MultiHeadAttention(dims, num_heads)
        self.ln1: LayerNorm = LayerNorm(dims)
        self.ln2: LayerNorm = LayerNorm(dims)
        self.linear1: Linear = Linear(dims, ff_dims)
        self.linear2: Linear = Linear(ff_dims, dims)
        self.dropout1: Dropout = Dropout(dropout)
        self.dropout2: Dropout = Dropout(dropout)
        self.activation: Callable[[array], array] = activation
        self.norm_first: bool = norm_first

    def __call__(self, x: array, mask: array | None = None) -> array:
        """Apply the Transformer encoder layer to `x`.

        Args:
            x: Input sequence `array` of shape `(B, L, dims)`.
            mask: Optional self-attention mask.

        Returns:
            Encoded `array` of shape `(B, L, dims)`.

        """
        if self.norm_first:
            y = self.ln1(x)
            y = self.attention(y, y, y, mask)
            x = x + self.dropout1(y)
            y = self.ln2(x)
            y = self.linear2(self.dropout2(self.activation(self.linear1(y))))
            return x + y

        y = self.attention(x, x, x, mask)
        x = self.ln1(x + self.dropout1(y))
        y = self.linear2(self.dropout2(self.activation(self.linear1(x))))
        return self.ln2(x + y)


class TransformerEncoder(Module):
    """A stack of `num_layers` `TransformerEncoderLayer` blocks."""

    def __init__(
        self,
        num_layers: int,
        dims: int,
        num_heads: int,
        mlp_dims: int | None = None,
        dropout: float = 0.0,
        activation: Callable[[array], array] = relu,
        norm_first: bool = True,
        checkpoint: bool = False,
    ) -> None:
        """Initialize `TransformerEncoder`.

        Args:
            num_layers: Number of stacked encoder layers.
            dims: Model dimension.
            num_heads: Number of attention heads per layer.
            mlp_dims: Feed-forward hidden dimension.
            dropout: Dropout probability.
            activation: MLP activation function.
            norm_first: Whether to use pre-norm.
            checkpoint: Whether to enable gradient checkpointing.

        """
        super().__init__()
        self.dims: int = dims
        self.checkpoint: bool = checkpoint
        self.layers: list[TransformerEncoderLayer] = [
            TransformerEncoderLayer(
                dims,
                num_heads,
                mlp_dims=mlp_dims,
                dropout=dropout,
                activation=activation,
                norm_first=norm_first,
            )
            for _ in range(num_layers)
        ]
        self.ln: LayerNorm = LayerNorm(dims)

    def __call__(self, x: array, mask: array | None = None) -> array:
        """Run the encoder stack on `x`.

        Args:
            x: Input `array` of shape `(B, L, dims)`.
            mask: Optional attention mask.

        Returns:
            Encoded output `array`.

        """
        for layer in self.layers:
            x = layer(x, mask)
        return self.ln(x)


class TransformerDecoderLayer(Module):
    """A single Transformer decoder layer with self-attention, cross-attention, and MLP."""

    def __init__(
        self,
        dims: int,
        num_heads: int,
        mlp_dims: int | None = None,
        dropout: float = 0.0,
        activation: Callable[[array], array] = relu,
        norm_first: bool = True,
    ) -> None:
        """Initialize `TransformerDecoderLayer`.

        Args:
            dims: Model feature dimension.
            num_heads: Number of attention heads.
            mlp_dims: Hidden dimension of the feed-forward MLP (defaults to `4 * dims`).
            dropout: Dropout probability.
            activation: Activation function in the MLP block.
            norm_first: Whether to apply LayerNorm before attention/MLP sublayers.

        """
        super().__init__()
        self.dims: int = dims
        ff_dims: int = mlp_dims if mlp_dims is not None else dims * 4
        self.self_attn: MultiHeadAttention = MultiHeadAttention(dims, num_heads)
        self.cross_attn: MultiHeadAttention = MultiHeadAttention(dims, num_heads)
        self.ln1: LayerNorm = LayerNorm(dims)
        self.ln2: LayerNorm = LayerNorm(dims)
        self.ln3: LayerNorm = LayerNorm(dims)
        self.linear1: Linear = Linear(dims, ff_dims)
        self.linear2: Linear = Linear(ff_dims, dims)
        self.dropout1: Dropout = Dropout(dropout)
        self.dropout2: Dropout = Dropout(dropout)
        self.dropout3: Dropout = Dropout(dropout)
        self.activation: Callable[[array], array] = activation
        self.norm_first: bool = norm_first

    def __call__(
        self,
        x: array,
        memory: array,
        x_mask: array | None = None,
        memory_mask: array | None = None,
    ) -> array:
        """Apply the Transformer decoder layer to `(x, memory)`.

        Args:
            x: Target sequence `array` of shape `(B, T, dims)`.
            memory: Encoded memory `array` of shape `(B, S, dims)`.
            x_mask: Optional target self-attention mask.
            memory_mask: Optional encoder-decoder cross-attention mask.

        Returns:
            Decoded `array` of shape `(B, T, dims)`.

        """
        if self.norm_first:
            y = self.ln1(x)
            x = x + self.dropout1(self.self_attn(y, y, y, x_mask))
            y = self.ln2(x)
            x = x + self.dropout2(self.cross_attn(y, memory, memory, memory_mask))
            y = self.ln3(x)
            y = self.linear2(self.dropout3(self.activation(self.linear1(y))))
            return x + y

        x = self.ln1(x + self.dropout1(self.self_attn(x, x, x, x_mask)))
        x = self.ln2(x + self.dropout2(self.cross_attn(x, memory, memory, memory_mask)))
        y = self.linear2(self.dropout3(self.activation(self.linear1(x))))
        return self.ln3(x + y)


class TransformerDecoder(Module):
    """A stack of `num_layers` `TransformerDecoderLayer` blocks."""

    def __init__(
        self,
        num_layers: int,
        dims: int,
        num_heads: int,
        mlp_dims: int | None = None,
        dropout: float = 0.0,
        activation: Callable[[array], array] = relu,
        norm_first: bool = True,
        checkpoint: bool = False,
    ) -> None:
        """Initialize `TransformerDecoder`.

        Args:
            num_layers: Number of stacked decoder layers.
            dims: Model dimension.
            num_heads: Number of attention heads.
            mlp_dims: Feed-forward hidden dimension.
            dropout: Dropout probability.
            activation: MLP activation function.
            norm_first: Whether to use pre-norm.
            checkpoint: Whether to enable gradient checkpointing.

        """
        super().__init__()
        self.dims: int = dims
        self.checkpoint: bool = checkpoint
        self.layers: list[TransformerDecoderLayer] = [
            TransformerDecoderLayer(
                dims,
                num_heads,
                mlp_dims=mlp_dims,
                dropout=dropout,
                activation=activation,
                norm_first=norm_first,
            )
            for _ in range(num_layers)
        ]
        self.ln: LayerNorm = LayerNorm(dims)

    def __call__(
        self,
        x: array,
        memory: array,
        x_mask: array | None = None,
        memory_mask: array | None = None,
    ) -> array:
        """Run the decoder stack on `(x, memory)`.

        Args:
            x: Target sequence `array`.
            memory: Encoder memory `array`.
            x_mask: Optional target self-attention mask.
            memory_mask: Optional cross-attention mask.

        Returns:
            Decoded `array`.

        """
        for layer in self.layers:
            x = layer(x, memory, x_mask, memory_mask)
        return self.ln(x)


class Transformer(Module):
    """Standard encoder-decoder Transformer model."""

    def __init__(
        self,
        dims: int = 512,
        num_heads: int = 8,
        num_encoder_layers: int = 6,
        num_decoder_layers: int = 6,
        mlp_dims: int | None = None,
        dropout: float = 0.0,
        activation: Callable[[array], array] = relu,
        custom_encoder: Module | None = None,
        custom_decoder: Module | None = None,
        norm_first: bool = True,
        checkpoint: bool = False,
    ) -> None:
        """Initialize `Transformer`.

        Args:
            dims: Model feature dimension.
            num_heads: Number of attention heads.
            num_encoder_layers: Number of encoder layers.
            num_decoder_layers: Number of decoder layers.
            mlp_dims: Feed-forward hidden dimension.
            dropout: Dropout probability.
            activation: MLP activation function.
            custom_encoder: Optional custom encoder `Module`.
            custom_decoder: Optional custom decoder `Module`.
            norm_first: Whether to use pre-norm.
            checkpoint: Whether to use gradient checkpointing.

        """
        super().__init__()
        self.dims: int = dims
        self.encoder: Module = (
            custom_encoder
            if custom_encoder is not None
            else TransformerEncoder(
                num_encoder_layers,
                dims,
                num_heads,
                mlp_dims=mlp_dims,
                dropout=dropout,
                activation=activation,
                norm_first=norm_first,
                checkpoint=checkpoint,
            )
        )
        self.decoder: Module = (
            custom_decoder
            if custom_decoder is not None
            else TransformerDecoder(
                num_decoder_layers,
                dims,
                num_heads,
                mlp_dims=mlp_dims,
                dropout=dropout,
                activation=activation,
                norm_first=norm_first,
                checkpoint=checkpoint,
            )
        )

    def __call__(
        self,
        src: array,
        tgt: array,
        src_mask: array | None = None,
        tgt_mask: array | None = None,
        memory_mask: array | None = None,
    ) -> array:
        """Run encoder-decoder forward pass on `(src, tgt)`.

        Args:
            src: Source sequence `array` of shape `(B, S, dims)`.
            tgt: Target sequence `array` of shape `(B, T, dims)`.
            src_mask: Optional source self-attention mask.
            tgt_mask: Optional target self-attention mask.
            memory_mask: Optional encoder-decoder cross-attention mask.

        Returns:
            Decoded output `array` of shape `(B, T, dims)`.

        """
        mem_res: array | tuple[array, ...] = self.encoder._call_forward(src, src_mask)
        memory: array = mem_res[0] if isinstance(mem_res, tuple) else mem_res
        out_res: array | tuple[array, ...] = self.decoder._call_forward(
            tgt, memory, tgt_mask, memory_mask
        )
        return out_res[0] if isinstance(out_res, tuple) else out_res


__all__ = [
    "MultiHeadAttention",
    "Transformer",
    "TransformerDecoder",
    "TransformerDecoderLayer",
    "TransformerEncoder",
    "TransformerEncoderLayer",
]
