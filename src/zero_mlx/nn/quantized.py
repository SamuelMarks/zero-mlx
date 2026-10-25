"""Model quantization utilities and quantized layers (`mlx.nn.quantized`) for zero_mlx."""

from __future__ import annotations

from typing import Callable, Union

from zero_mlx.nn.base import Module
from zero_mlx.nn.embedding import Embedding, QuantizedEmbedding
from zero_mlx.nn.linear import (
    AllToShardedLinear,
    Linear,
    QQLinear,
    QuantizedAllToShardedLinear,
    QuantizedLinear,
    QuantizedShardedToAllLinear,
    ShardedToAllLinear,
)

ClassPredicate = Callable[[str, Module], Union[bool, dict[str, Union[int, str]]]]


def quantize(
    model: Module,
    group_size: int | None = None,
    bits: int | None = None,
    *,
    mode: str = "affine",
    quantize_input: bool = False,
    class_predicate: ClassPredicate | None = None,
) -> Module:
    """Quantize the linear and embedding layers of `model` in-place.

    Args:
        model: Root `Module` whose layers will be quantized.
        group_size: Quantization group size (defaults to 64, or 32 for `quantize_input=True`).
        bits: Bits per quantized weight (defaults to 4).
        mode: Quantization mode (`'affine'`, `'mxfp4'`, `'nvfp4'`, `'mxfp8'`).
        quantize_input: If True, convert `Linear` layers to `QQLinear`.
        class_predicate: Optional predicate `(path, module) -> bool | dict` controlling which layers to quantize.

    Returns:
        The quantized `model`.

    """
    g_size: int = (32 if quantize_input else 64) if group_size is None else group_size
    b_bits: int = 4 if bits is None else bits

    for k, sub in list(model.children().items()):
        should_quantize: bool = True
        sub_g_size: int = g_size
        sub_bits: int = b_bits
        sub_mode: str = mode
        if class_predicate is not None:
            pred_res = class_predicate(k, sub)
            if isinstance(pred_res, dict):
                should_quantize = True
                sub_g_size = int(pred_res.get("group_size", g_size))
                sub_bits = int(pred_res.get("bits", b_bits))
                sub_mode = str(pred_res.get("mode", mode))
            else:
                should_quantize = bool(pred_res)
        if should_quantize:
            if isinstance(sub, Linear):
                if quantize_input:
                    q_mod: Module = QQLinear.from_linear(
                        sub,
                        group_size=sub_g_size,
                        bits=sub_bits,
                        mode=sub_mode,
                    )
                else:
                    q_mod = QuantizedLinear.from_linear(
                        sub,
                        group_size=sub_g_size,
                        bits=sub_bits,
                        mode=sub_mode,
                    )
                model[k] = q_mod
            elif isinstance(sub, Embedding):
                q_emb = QuantizedEmbedding.from_embedding(
                    sub,
                    group_size=sub_g_size,
                    bits=sub_bits,
                    mode=sub_mode,
                )
                model[k] = q_emb
            else:
                quantize(
                    sub,
                    group_size=g_size,
                    bits=b_bits,
                    mode=mode,
                    quantize_input=quantize_input,
                    class_predicate=class_predicate,
                )
    return model


__all__ = [
    "AllToShardedLinear",
    "ClassPredicate",
    "QQLinear",
    "QuantizedAllToShardedLinear",
    "QuantizedEmbedding",
    "QuantizedLinear",
    "QuantizedShardedToAllLinear",
    "ShardedToAllLinear",
    "quantize",
]
