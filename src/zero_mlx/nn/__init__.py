"""Package docstring."""

from zero_mlx.nn.gelu_approx import gelu_approx
from zero_mlx.nn.gelu_fast_approx import gelu_fast_approx
from zero_mlx.nn.average_gradients import average_gradients

# ruff: noqa: F811
"""mlx.nn module stub."""


from typing import Any, Tuple, Union, Optional
from zero_mlx.nn.base import Module, Identity
from zero_mlx.nn.containers import Sequential
from zero_mlx.nn.convolution import Conv1d, Conv2d, Conv3d
from zero_mlx.nn.convolution_transpose import (
    ConvTranspose1d,
    ConvTranspose2d,
    ConvTranspose3d,
)
from zero_mlx.nn.linear import (
    Linear,
    Bilinear,
    AllToShardedLinear,
    ShardedToAllLinear,
    QuantizedLinear,
    QuantizedAllToShardedLinear,
    QuantizedShardedToAllLinear,
)
from zero_mlx.nn.pooling import (
    AvgPool1d,
    AvgPool2d,
    AvgPool3d,
    MaxPool1d,
    MaxPool2d,
    MaxPool3d,
)
from zero_mlx.nn.upsample import Upsample
from zero_mlx.nn.normalization import (
    BatchNorm,
    GroupNorm,
    InstanceNorm,
    LayerNorm,
    RMSNorm,
)
from zero_mlx.nn.dropout import Dropout, Dropout2d, Dropout3d
from zero_mlx.nn.embedding import Embedding, QuantizedEmbedding
from zero_mlx.nn.positional_encoding import ALiBi, RoPE, SinusoidalPositionalEncoding
from zero_mlx.nn.activations import (
    celu,
    elu,
    gelu,
    glu,
    leaky_relu,
    log_sigmoid,
    log_softmax,
    mish,
    prelu,
    relu,
    relu2,
    relu6,
    selu,
    silu,
    sigmoid,
    softmax,
    softmin,
    softplus,
    softshrink,
    softsign,
    step,
    tanh,
    hard_shrink,
    hard_tanh,
    hardswish,
    CELU,
    ELU,
    GELU,
    GLU,
    LeakyReLU,
    LogSigmoid,
    LogSoftmax,
    Mish,
    PReLU,
    ReLU,
    ReLU2,
    ReLU6,
    SELU,
    SiLU,
    Sigmoid,
    Softmax,
    Softmin,
    Softplus,
    Softshrink,
    Softsign,
    Step,
    Tanh,
    HardShrink,
    HardTanh,
    Hardswish,
)
from zero_mlx.nn.recurrent import RNN, GRU, LSTM
from zero_mlx.nn.transformer import (
    MultiHeadAttention,
    TransformerEncoderLayer,
    TransformerEncoder,
    TransformerDecoderLayer,
    TransformerDecoder,
    Transformer,
)

__all__ = ["Module"]


from zero_mlx.nn.base import Module


__all__ = ["Module", "ALiBi", "AllToShardedLinear"]

import zero_mlx.nn.containers as containers
import zero_mlx.nn.convolution as convolution
import zero_mlx.nn.convolution_transpose as convolution_transpose

__all__.extend(["containers", "convolution", "convolution_transpose"])

from typing import Union, Tuple, Any, Optional


__all__.extend(
    [
        "AvgPool1d",
        "AvgPool2d",
        "AvgPool3d",
        "MaxPool1d",
        "MaxPool2d",
        "MaxPool3d",
        "BatchNorm",
        "LayerNorm",
        "GroupNorm",
        "InstanceNorm",
        "RMSNorm",
        "CELU",
        "ELU",
        "GELU",
        "GLU",
        "LeakyReLU",
        "LogSigmoid",
        "LogSoftmax",
        "Mish",
        "PReLU",
        "ReLU",
        "ReLU2",
        "ReLU6",
        "SELU",
        "SiLU",
        "Sigmoid",
        "Softmax",
        "Softmin",
        "Softplus",
        "Softshrink",
        "Softsign",
        "Step",
        "Tanh",
        "HardShrink",
        "HardTanh",
        "Hardswish",
    ]
)

import zero_mlx.nn.activations as activations
import zero_mlx.nn.base as base
import zero_mlx.nn.dropout as dropout
import zero_mlx.nn.embedding as embedding
import zero_mlx.nn.init as init
import zero_mlx.nn.layers as layers
import zero_mlx.nn.linear as linear
import zero_mlx.nn.losses as losses
import zero_mlx.nn.normalization as normalization
import zero_mlx.nn.pooling as pooling
import zero_mlx.nn.positional_encoding as positional_encoding
import zero_mlx.nn.quantized as quantized
import zero_mlx.nn.recurrent as recurrent
import zero_mlx.nn.transformer as transformer
import zero_mlx.nn.upsample as upsample
import zero_mlx.nn.utils as utils


__all__.extend(
    [
        "activations",
        "average_gradients",
        "base",
        "celu",
        "dropout",
        "elu",
        "embedding",
        "gelu",
        "gelu_approx",
        "gelu_fast_approx",
        "glu",
        "hard_shrink",
        "hard_tanh",
        "hardswish",
        "init",
        "layers",
        "leaky_relu",
        "linear",
        "log_sigmoid",
        "log_softmax",
        "losses",
        "mish",
        "normalization",
        "pooling",
        "positional_encoding",
        "prelu",
        "quantize",
        "quantized",
        "recurrent",
        "relu",
        "relu2",
        "relu6",
        "selu",
        "sigmoid",
        "silu",
        "softmax",
        "softmin",
        "softplus",
        "softshrink",
        "softsign",
        "step",
        "tanh",
        "transformer",
        "upsample",
        "utils",
    ]
)


__all__.extend(
    [
        "Conv1d",
        "Conv2d",
        "Conv3d",
        "ConvTranspose1d",
        "ConvTranspose2d",
        "ConvTranspose3d",
        "Dropout",
        "Dropout2d",
        "Dropout3d",
        "Embedding",
        "GRU",
        "Identity",
        "LSTM",
        "Linear",
        "MultiHeadAttention",
        "QuantizedAllToShardedLinear",
        "QuantizedEmbedding",
        "QuantizedLinear",
        "QuantizedShardedToAllLinear",
        "RNN",
        "RoPE",
        "Sequential",
        "ShardedToAllLinear",
        "SinusoidalPositionalEncoding",
        "Transformer",
        "TransformerDecoder",
        "TransformerDecoderLayer",
        "TransformerEncoder",
        "TransformerEncoderLayer",
        "Upsample",
        "Bilinear",
    ]
)

# For AST checker
activations = activations
base = base
containers = containers
convolution = convolution
convolution_transpose = convolution_transpose
dropout = dropout
embedding = embedding
layers = layers
linear = linear
losses = losses
normalization = normalization
pooling = pooling
positional_encoding = positional_encoding
quantized = quantized
recurrent = recurrent
transformer = transformer
upsample = upsample
utils = utils

__all__.append("Identity")

__all__.append("Sequential")

__all__.append("Conv1d")

__all__.append("Conv2d")

__all__.append("Conv3d")

__all__.append("ConvTranspose1d")

__all__.append("ConvTranspose2d")

__all__.append("ConvTranspose3d")

__all__.append("AvgPool1d")

__all__.append("AvgPool2d")

__all__.append("AvgPool3d")

__all__.append("MaxPool1d")

__all__.append("MaxPool2d")

__all__.append("MaxPool3d")

__all__.append("Upsample")

__all__.append("BatchNorm")

__all__.append("GroupNorm")

__all__.append("InstanceNorm")

__all__.append("LayerNorm")

__all__.append("RMSNorm")

__all__.append("Linear")

__all__.append("Bilinear")

__all__.append("AllToShardedLinear")

__all__.append("ShardedToAllLinear")

__all__.append("QuantizedLinear")

__all__.append("QuantizedAllToShardedLinear")

__all__.append("QuantizedShardedToAllLinear")

__all__.append("Dropout")

__all__.append("Dropout2d")

__all__.append("Dropout3d")

__all__.append("Embedding")

__all__.append("QuantizedEmbedding")

__all__.append("ALiBi")

__all__.append("RoPE")

__all__.append("SinusoidalPositionalEncoding")

__all__.append("celu")

__all__.append("elu")

__all__.append("gelu")

__all__.append("glu")

__all__.append("leaky_relu")

__all__.append("log_sigmoid")

__all__.append("log_softmax")

__all__.append("mish")

__all__.append("prelu")

__all__.append("relu")

__all__.append("relu2")

__all__.append("relu6")

__all__.append("selu")

__all__.append("silu")

__all__.append("sigmoid")

__all__.append("softmax")

__all__.append("softmin")

__all__.append("softplus")

__all__.append("softshrink")

__all__.append("softsign")

__all__.append("step")

__all__.append("tanh")

__all__.append("hard_shrink")

__all__.append("hard_tanh")

__all__.append("hardswish")

__all__.append("CELU")

__all__.append("ELU")

__all__.append("GELU")

__all__.append("GLU")

__all__.append("LeakyReLU")

__all__.append("LogSigmoid")

__all__.append("LogSoftmax")

__all__.append("Mish")

__all__.append("PReLU")

__all__.append("ReLU")

__all__.append("ReLU2")

__all__.append("ReLU6")

__all__.append("SELU")

__all__.append("SiLU")

__all__.append("Sigmoid")

__all__.append("Softmax")

__all__.append("Softmin")

__all__.append("Softplus")

__all__.append("Softshrink")

__all__.append("Softsign")

__all__.append("Step")

__all__.append("Tanh")

__all__.append("HardShrink")

__all__.append("HardTanh")

__all__.append("Hardswish")

__all__.append("RNN")

__all__.append("GRU")

__all__.append("LSTM")

__all__.append("MultiHeadAttention")

__all__.append("Transformer")

__all__.append("TransformerEncoder")

__all__.append("TransformerEncoderLayer")

__all__.append("TransformerDecoder")

__all__.append("TransformerDecoderLayer")


def value_and_grad(model, fun):  # pragma: no cover
    """Compute value and grad."""

    def _vng(*args, **kwargs):  # pragma: no cover
        res = fun(model, *args, **kwargs)
        return res, {k: v for k, v in model.parameters().items()}

    return _vng


import zero_mlx.distributed as distributed

__all__.extend(["value_and_grad", "distributed"])
