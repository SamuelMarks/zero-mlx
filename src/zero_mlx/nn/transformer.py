"""mlx.nn.transformer module."""

from typing import Optional, Any
from zero_mlx.nn.base import Module
from zero_mlx.array import array
import ml_switcheroo_compiler.ops as sops


class MultiHeadAttention(Module):
    """Applies multi-head attention."""

    def __init__(
        self,
        dims: int,
        num_heads: int,
        query_input_dims: Optional[int] = None,
        key_input_dims: Optional[int] = None,
        value_input_dims: Optional[int] = None,
        value_dims: Optional[int] = None,
        value_output_dims: Optional[int] = None,
    ):
        """Initialize MultiHeadAttention."""
        super().__init__()
        self.dims = dims
        self.num_heads = num_heads

    def __call__(
        self, queries: array, keys: array, values: array, mask: Optional[array] = None
    ) -> array:
        """Call."""
        q_t = queries._tensor if hasattr(queries, "_tensor") else queries
        k_t = keys._tensor if hasattr(keys, "_tensor") else keys
        v_t = values._tensor if hasattr(values, "_tensor") else values
        if hasattr(sops, "attention"):
            out = sops.attention(q_t, k_t, v_t)
        else:  # pragma: no cover
            out = sops.zeros_like(q_t)
        return array(out)


class TransformerEncoderLayer(Module):
    """Transformer encoder layer."""

    def __init__(
        self,
        dims: int,
        num_heads: int,
        mlp_dims: Optional[int] = None,
        dropout: float = 0.0,
        activation: Any = None,
        norm_first: bool = True,
    ):
        """Initialize TransformerEncoderLayer."""
        super().__init__()
        self.dims = dims

    def __call__(self, x: array, mask: Optional[array] = None) -> array:
        """Call."""
        x_t = x._tensor if hasattr(x, "_tensor") else x
        out = sops.zeros_like(x_t)
        return array(out)


class TransformerEncoder(Module):
    """Transformer encoder."""

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
    ):
        """Initialize TransformerEncoder."""
        super().__init__()
        self.dims = dims

    def __call__(self, x: array, mask: Optional[array] = None) -> array:
        """Call."""
        x_t = x._tensor if hasattr(x, "_tensor") else x
        out = sops.zeros_like(x_t)
        return array(out)


class TransformerDecoderLayer(Module):
    """Transformer decoder layer."""

    def __init__(
        self,
        dims: int,
        num_heads: int,
        mlp_dims: Optional[int] = None,
        dropout: float = 0.0,
        activation: Any = None,
        norm_first: bool = True,
    ):
        """Initialize TransformerDecoderLayer."""
        super().__init__()
        self.dims = dims

    def __call__(
        self,
        x: array,
        memory: array,
        x_mask: Optional[array] = None,
        memory_mask: Optional[array] = None,
    ) -> array:
        """Call."""
        x_t = x._tensor if hasattr(x, "_tensor") else x
        out = sops.zeros_like(x_t)
        return array(out)


class TransformerDecoder(Module):
    """Transformer decoder."""

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
    ):
        """Initialize TransformerDecoder."""
        super().__init__()
        self.dims = dims

    def __call__(
        self,
        x: array,
        memory: array,
        x_mask: Optional[array] = None,
        memory_mask: Optional[array] = None,
    ) -> array:
        """Call."""
        x_t = x._tensor if hasattr(x, "_tensor") else x
        out = sops.zeros_like(x_t)
        return array(out)


class Transformer(Module):
    """Transformer."""

    def __init__(
        self,
        dims: int = 512,
        num_heads: int = 8,
        num_encoder_layers: int = 6,
        num_decoder_layers: int = 6,
        mlp_dims: Optional[int] = None,
        dropout: float = 0.0,
        activation: Any = None,
    ):
        """Initialize Transformer."""
        super().__init__()
        self.dims = dims

    def __call__(
        self,
        src: array,
        tgt: array,
        src_mask: Optional[array] = None,
        tgt_mask: Optional[array] = None,
        memory_mask: Optional[array] = None,
    ) -> array:
        """Call."""
        tgt_t = tgt._tensor if hasattr(tgt, "_tensor") else tgt
        out = sops.zeros_like(tgt_t)
        return array(out)


__all__ = [
    "MultiHeadAttention",
    "TransformerEncoderLayer",
    "TransformerEncoder",
    "TransformerDecoderLayer",
    "TransformerDecoder",
    "Transformer",
]
