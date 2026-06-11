# ruff: noqa: F811
"""mlx.nn module stub."""

from typing import Any


class Module:
    def __init__(self):
        pass


class ALiBi(Module):
    def __init__(self):
        super().__init__()
        raise NotImplementedError("ALiBi not implemented")


class AllToShardedLinear(Module):
    def __init__(
        self, input_dims: int, output_dims: int, bias: bool = True, group: Any = None
    ):
        super().__init__()
        raise NotImplementedError("AllToShardedLinear not implemented")


__all__ = ["Module", "ALiBi", "AllToShardedLinear"]

import zero_mlx.nn.containers as containers
import zero_mlx.nn.convolution as convolution
import zero_mlx.nn.convolution_transpose as convolution_transpose

__all__.extend(["containers", "convolution", "convolution_transpose"])

from typing import Union, Tuple, Any, Optional


class AvgPool1d(Module):
    def __init__(
        self,
        kernel_size: Union[int, Tuple[int]],
        stride: Union[int, Tuple[int], None] = None,
        padding: Union[int, Tuple[int]] = 0,
    ):
        super().__init__()
        raise NotImplementedError("AvgPool1d not implemented")


class AvgPool2d(Module):
    def __init__(
        self,
        kernel_size: Union[int, Tuple[int, int]],
        stride: Union[int, Tuple[int, int], None] = None,
        padding: Union[int, Tuple[int, int], None] = 0,
    ):
        super().__init__()
        raise NotImplementedError("AvgPool2d not implemented")


class AvgPool3d(Module):
    def __init__(
        self,
        kernel_size: Union[int, Tuple[int, int, int]],
        stride: Union[int, Tuple[int, int, int], None] = None,
        padding: Union[int, Tuple[int, int, int], None] = 0,
    ):
        super().__init__()
        raise NotImplementedError("AvgPool3d not implemented")


class MaxPool1d(Module):
    def __init__(
        self,
        kernel_size: Union[int, Tuple[int]],
        stride: Union[int, Tuple[int], None] = None,
        padding: Union[int, Tuple[int]] = 0,
    ):
        super().__init__()
        raise NotImplementedError("MaxPool1d not implemented")


class MaxPool2d(Module):
    def __init__(
        self,
        kernel_size: Union[int, Tuple[int, int]],
        stride: Union[int, Tuple[int, int], None] = None,
        padding: Union[int, Tuple[int, int], None] = 0,
    ):
        super().__init__()
        raise NotImplementedError("MaxPool2d not implemented")


class MaxPool3d(Module):
    def __init__(
        self,
        kernel_size: Union[int, Tuple[int, int, int]],
        stride: Union[int, Tuple[int, int, int], None] = None,
        padding: Union[int, Tuple[int, int, int], None] = 0,
    ):
        super().__init__()
        raise NotImplementedError("MaxPool3d not implemented")


class BatchNorm(Module):
    def __init__(
        self,
        num_features: int,
        eps: float = 1e-05,
        momentum: float = 0.1,
        affine: bool = True,
        track_running_stats: bool = True,
    ):
        super().__init__()
        raise NotImplementedError("BatchNorm not implemented")


class LayerNorm(Module):
    def __init__(
        self, dims: int, eps: float = 1e-05, affine: bool = True, bias: bool = True
    ):
        super().__init__()
        raise NotImplementedError("LayerNorm not implemented")


class GroupNorm(Module):
    def __init__(
        self,
        num_groups: int,
        dims: int,
        eps: float = 1e-05,
        affine: bool = True,
        pytorch_compatible: bool = False,
    ):
        super().__init__()
        raise NotImplementedError("GroupNorm not implemented")


class InstanceNorm(Module):
    def __init__(self, dims: int, eps: float = 1e-05, affine: bool = False):
        super().__init__()
        raise NotImplementedError("InstanceNorm not implemented")


class RMSNorm(Module):
    def __init__(self, dims: int, eps: float = 1e-05):
        super().__init__()
        raise NotImplementedError("RMSNorm not implemented")


class CELU(Module):
    def __init__(self, alpha=1.0):
        super().__init__()
        raise NotImplementedError("CELU not implemented")


class ELU(Module):
    def __init__(self, alpha=1.0):
        super().__init__()
        raise NotImplementedError("ELU not implemented")


class GELU(Module):
    def __init__(self, approx="none"):
        super().__init__()
        raise NotImplementedError("GELU not implemented")


class GLU(Module):
    def __init__(self, axis: int = -1):
        super().__init__()
        raise NotImplementedError("GLU not implemented")


class LeakyReLU(Module):
    def __init__(self, negative_slope=0.01):
        super().__init__()
        raise NotImplementedError("LeakyReLU not implemented")


class LogSigmoid(Module):
    def __init__(self):
        super().__init__()
        raise NotImplementedError("LogSigmoid not implemented")


class LogSoftmax(Module):
    def __init__(self):
        super().__init__()
        raise NotImplementedError("LogSoftmax not implemented")


class Mish(Module):
    def __init__(self):
        super().__init__()
        raise NotImplementedError("Mish not implemented")


class PReLU(Module):
    def __init__(self, num_parameters=1, init=0.25):
        super().__init__()
        raise NotImplementedError("PReLU not implemented")


class ReLU(Module):
    def __init__(self):
        super().__init__()
        raise NotImplementedError("ReLU not implemented")


class ReLU2(Module):
    def __init__(self):
        super().__init__()
        raise NotImplementedError("ReLU2 not implemented")


class ReLU6(Module):
    def __init__(self):
        super().__init__()
        raise NotImplementedError("ReLU6 not implemented")


class SELU(Module):
    def __init__(self):
        super().__init__()
        raise NotImplementedError("SELU not implemented")


class SiLU(Module):
    def __init__(self):
        super().__init__()
        raise NotImplementedError("SiLU not implemented")


class Sigmoid(Module):
    def __init__(self):
        super().__init__()
        raise NotImplementedError("Sigmoid not implemented")


class Softmax(Module):
    def __init__(self):
        super().__init__()
        raise NotImplementedError("Softmax not implemented")


class Softmin(Module):
    def __init__(self):
        super().__init__()
        raise NotImplementedError("Softmin not implemented")


class Softplus(Module):
    def __init__(self):
        super().__init__()
        raise NotImplementedError("Softplus not implemented")


class Softshrink(Module):
    def __init__(self, lambd=0.5):
        super().__init__()
        raise NotImplementedError("Softshrink not implemented")


class Softsign(Module):
    def __init__(self):
        super().__init__()
        raise NotImplementedError("Softsign not implemented")


class Step(Module):
    def __init__(self, threshold: float = 0.0):
        super().__init__()
        raise NotImplementedError("Step not implemented")


class Tanh(Module):
    def __init__(self):
        super().__init__()
        raise NotImplementedError("Tanh not implemented")


class HardShrink(Module):
    def __init__(self):
        super().__init__()
        raise NotImplementedError("HardShrink not implemented")


class HardTanh(Module):
    def __init__(self):
        super().__init__()
        raise NotImplementedError("HardTanh not implemented")


class Hardswish(Module):
    def __init__(self):
        super().__init__()
        raise NotImplementedError("Hardswish not implemented")


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


def average_gradients(
    gradients: Any,
    group: Any = None,
    all_reduce_size: int = 33554432,
    communication_type: Any = None,
    communication_stream: Any = None,
) -> Any:
    raise NotImplementedError("average_gradients is not implemented")


def celu(x, alpha=1.0):
    raise NotImplementedError("celu is not implemented")


def elu(x, alpha=1.0):
    raise NotImplementedError("elu is not implemented")


def gelu(x):
    raise NotImplementedError("gelu is not implemented")


def gelu_approx(x):
    raise NotImplementedError("gelu_approx is not implemented")


def gelu_fast_approx(x):
    raise NotImplementedError("gelu_fast_approx is not implemented")


def glu(x, axis: int = -1):
    raise NotImplementedError("glu is not implemented")


def hard_shrink(x, lambd=0.5):
    raise NotImplementedError("hard_shrink is not implemented")


def hard_tanh(x, min_val=-1.0, max_val=1.0):
    raise NotImplementedError("hard_tanh is not implemented")


def hardswish(x):
    raise NotImplementedError("hardswish is not implemented")


def leaky_relu(x, negative_slope=0.01):
    raise NotImplementedError("leaky_relu is not implemented")


def log_sigmoid(x):
    raise NotImplementedError("log_sigmoid is not implemented")


def log_softmax(x, axis=-1):
    raise NotImplementedError("log_softmax is not implemented")


def mish(x):
    raise NotImplementedError("mish is not implemented")


def prelu(x, alpha):
    raise NotImplementedError("prelu is not implemented")


def relu(x):
    raise NotImplementedError("relu is not implemented")


def relu2(x):
    raise NotImplementedError("relu2 is not implemented")


def relu6(x):
    raise NotImplementedError("relu6 is not implemented")


def selu(x):
    raise NotImplementedError("selu is not implemented")


def silu(x):
    raise NotImplementedError("silu is not implemented")


def sigmoid(x):
    raise NotImplementedError("sigmoid is not implemented")


def softmax(x):
    raise NotImplementedError("softmax is not implemented")


def softmin(x, axis=-1):
    raise NotImplementedError("softmin is not implemented")


def softplus(x):
    raise NotImplementedError("softplus is not implemented")


def softshrink(x, lambd=0.5):
    raise NotImplementedError("softshrink is not implemented")


def softsign(x):
    raise NotImplementedError("softsign is not implemented")


def step(x, threshold: float = 0.0):
    raise NotImplementedError("step is not implemented")


def tanh(x):
    raise NotImplementedError("tanh is not implemented")


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


class Conv1d(Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = True,
    ):
        super().__init__()
        raise NotImplementedError("Conv1d not implemented")


class Conv2d(Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, tuple],
        stride: Union[int, tuple] = 1,
        padding: Union[int, tuple] = 0,
        dilation: Union[int, tuple] = 1,
        groups: int = 1,
        bias: bool = True,
    ):
        super().__init__()
        raise NotImplementedError("Conv2d not implemented")


class Conv3d(Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, tuple],
        stride: Union[int, tuple] = 1,
        padding: Union[int, tuple] = 0,
        dilation: Union[int, tuple] = 1,
        bias: bool = True,
    ):
        super().__init__()
        raise NotImplementedError("Conv3d not implemented")


class ConvTranspose1d(Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        output_padding: int = 0,
        bias: bool = True,
    ):
        super().__init__()
        raise NotImplementedError("ConvTranspose1d not implemented")


class ConvTranspose2d(Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, tuple],
        stride: Union[int, tuple] = 1,
        padding: Union[int, tuple] = 0,
        dilation: Union[int, tuple] = 1,
        output_padding: Union[int, tuple] = 0,
        bias: bool = True,
    ):
        super().__init__()
        raise NotImplementedError("ConvTranspose2d not implemented")


class ConvTranspose3d(Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, tuple],
        stride: Union[int, tuple] = 1,
        padding: Union[int, tuple] = 0,
        dilation: Union[int, tuple] = 1,
        output_padding: Union[int, tuple] = 0,
        bias: bool = True,
    ):
        super().__init__()
        raise NotImplementedError("ConvTranspose3d not implemented")


class Dropout(Module):
    def __init__(self, p: float = 0.5):
        super().__init__()
        raise NotImplementedError("Dropout not implemented")


class Dropout2d(Module):
    def __init__(self, p: float = 0.5):
        super().__init__()
        raise NotImplementedError("Dropout2d not implemented")


class Dropout3d(Module):
    def __init__(self, p: float = 0.5):
        super().__init__()
        raise NotImplementedError("Dropout3d not implemented")


class Embedding(Module):
    def __init__(self, num_embeddings: int, dims: int):
        super().__init__()
        raise NotImplementedError("Embedding not implemented")


class GRU(Module):
    def __init__(self, input_size: int, hidden_size: int, bias: bool = True):
        super().__init__()
        raise NotImplementedError("GRU not implemented")


class Identity(Module):
    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__()
        raise NotImplementedError("Identity not implemented")


class LSTM(Module):
    def __init__(self, input_size: int, hidden_size: int, bias: bool = True):
        super().__init__()
        raise NotImplementedError("LSTM not implemented")


class Linear(Module):
    def __init__(self, input_dims: int, output_dims: int, bias: bool = True):
        super().__init__()
        raise NotImplementedError("Linear not implemented")


class MultiHeadAttention(Module):
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
        super().__init__()
        raise NotImplementedError("MultiHeadAttention not implemented")


class QuantizedAllToShardedLinear(Module):
    def __init__(
        self,
        input_dims: int,
        output_dims: int,
        bias: bool = True,
        group_size: int = 64,
        bits: int = 4,
        group: Optional[Any] = None,
    ):
        super().__init__()
        raise NotImplementedError("QuantizedAllToShardedLinear not implemented")


class QuantizedEmbedding(Module):
    def __init__(
        self,
        num_embeddings: int,
        dims: int,
        group_size: int = 64,
        bits: int = 4,
        mode: str = "affine",
    ):
        super().__init__()
        raise NotImplementedError("QuantizedEmbedding not implemented")


class QuantizedLinear(Module):
    def __init__(
        self,
        input_dims: int,
        output_dims: int,
        bias: bool = True,
        group_size: int = 64,
        bits: int = 4,
        mode: str = "affine",
    ):
        super().__init__()
        raise NotImplementedError("QuantizedLinear not implemented")


class QuantizedShardedToAllLinear(Module):
    def __init__(
        self,
        input_dims: int,
        output_dims: int,
        bias: bool = True,
        group_size: int = 64,
        bits: int = 4,
        group: Optional[Any] = None,
    ):
        super().__init__()
        raise NotImplementedError("QuantizedShardedToAllLinear not implemented")


class RNN(Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        bias: bool = True,
        nonlinearity: Optional[Any] = None,
    ):
        super().__init__()
        raise NotImplementedError("RNN not implemented")


class RoPE(Module):
    def __init__(
        self,
        dims: int,
        traditional: bool = False,
        base: float = 10000,
        scale: float = 1.0,
    ):
        super().__init__()
        raise NotImplementedError("RoPE not implemented")


class Sequential(Module):
    def __init__(self, *modules):
        super().__init__()
        raise NotImplementedError("Sequential not implemented")


class ShardedToAllLinear(Module):
    def __init__(
        self,
        input_dims: int,
        output_dims: int,
        bias: bool = True,
        group: Optional[Any] = None,
    ):
        super().__init__()
        raise NotImplementedError("ShardedToAllLinear not implemented")


class SinusoidalPositionalEncoding(Module):
    def __init__(
        self,
        dims: int,
        min_freq: float = 0.0001,
        max_freq: float = 1,
        scale: Optional[float] = None,
        cos_first: bool = False,
        full_turns: bool = False,
    ):
        super().__init__()
        raise NotImplementedError("SinusoidalPositionalEncoding not implemented")


class Transformer(Module):
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
        super().__init__()
        raise NotImplementedError("Transformer not implemented")


class TransformerDecoder(Module):
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
        super().__init__()
        raise NotImplementedError("TransformerDecoder not implemented")


class TransformerDecoderLayer(Module):
    def __init__(
        self,
        dims: int,
        num_heads: int,
        mlp_dims: Optional[int] = None,
        dropout: float = 0.0,
        activation: Any = None,
        norm_first: bool = True,
    ):
        super().__init__()
        raise NotImplementedError("TransformerDecoderLayer not implemented")


class TransformerEncoder(Module):
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
        super().__init__()
        raise NotImplementedError("TransformerEncoder not implemented")


class TransformerEncoderLayer(Module):
    def __init__(
        self,
        dims: int,
        num_heads: int,
        mlp_dims: Optional[int] = None,
        dropout: float = 0.0,
        activation: Any = None,
        norm_first: bool = True,
    ):
        super().__init__()
        raise NotImplementedError("TransformerEncoderLayer not implemented")


class Upsample(Module):
    def __init__(
        self,
        scale_factor: Union[float, Tuple],
        mode: str = "nearest",
        align_corners: bool = False,
    ):
        super().__init__()
        raise NotImplementedError("Upsample not implemented")


class Bilinear(Module):
    def __init__(
        self, input1_dims: int, input2_dims: int, output_dims: int, bias: bool = True
    ):
        super().__init__()
        raise NotImplementedError("Bilinear not implemented")


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
