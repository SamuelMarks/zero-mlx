# Missing MLX APIs Implementation Plan

This document tracks the missing MLX APIs that need to be implemented in zero-mlx.

| Status | Name | Signature | Docstring |
|---|---|---|---|
| [x] | `mlx.core.ArrayAt` | `(...)` | A helper object to apply updates at specific indices. |
| [x] | `mlx.core.ArrayIterator` | `(...)` | A helper object to iterate over the 1st dimension of an array. |
| [x] | `mlx.core.ArrayLike` | `(...)` | Any Python object which has an __mlx__array__ method that |
| [x] | `mlx.core.DeviceType` | `(value, names=None, *, module=None, qualname=None, type=None, start=1)` | No documentation available. |
| [x] | `mlx.core.Dtype` | `(...)` | An object to hold the type of a :class:array. |
| [x] | `mlx.core.DtypeCategory` | `(value, names=None, *, module=None, qualname=None, type=None, start=1)` | Type to hold categories of :class:dtypes <Dtype>. |
| [x] | `mlx.core.FunctionExporter` | `(...)` | A context managing class for exporting multiple traces of the same |
| [x] | `mlx.core.StreamContext` | `(...)` | A context manager for setting the current device and stream. |
| [x] | `mlx.core.arccosh` | `(a: array, /, *, stream: Union[None, Stream, Device] = None) -> array` | Element-wise inverse hyperbolic cosine. |
| [x] | `mlx.core.arctan2` | `(a: array, b: array, /, *, stream: Union[None, Stream, Device] = None) -> array` | Element-wise inverse tangent of the ratio of two arrays. |
| [x] | `mlx.core.bitwise_invert` | `(a: Union[scalar, array], stream: Union[None, Stream, Device] = None) -> array` | Element-wise bitwise inverse. |
| [x] | `mlx.core.clear_cache` | `() -> None` | Clear the memory cache. |
| [x] | `mlx.core.complexfloating` | `(...)` | No documentation available. |
| [x] | `mlx.core.concat` | `(arrays: list[array], axis: Optional[int] = 0, *, stream: Union[None, Stream, Device] = None) -> array` | See :func:concatenate. |
| [x] | `mlx.core.conj` | `(a: array, *, stream: Union[None, Stream, Device] = None) -> array` | Return the elementwise complex conjugate of the input. |
| [x] | `mlx.core.contiguous` | `(a: array, /, allow_col_major: bool = False, *, stream: Union[None, Stream, Device] = None) -> array` | Force an array to be row contiguous. Copy if necessary. |
| [x] | `mlx.core.conv1d` | `(input: array, weight: array, /, stride: int = 1, padding: int = 0, dilation: int = 1, groups: int = 1, *, stream: Union[None, Stream, Device] = None) -> array` | 1D convolution over an input with several channels |
| [x] | `mlx.core.conv2d` | `(input: array, weight: array, /, stride: Union[int, tuple[int, int]] = 1, padding: Union[int, tuple[int, int]] = 0, dilation: Union[int, tuple[int, int]] = 1, groups: int = 1, *, stream: Union[None...` | 2D convolution over an input with several channels |
| [x] | `mlx.core.conv3d` | `(input: array, weight: array, /, stride: Union[int, tuple[int, int, int]] = 1, padding: Union[int, tuple[int, int, int]] = 0, dilation: Union[int, tuple[int, int, int]] = 1, groups: int = 1, *, str...` | 3D convolution over an input with several channels |
| [x] | `mlx.core.conv_general` | `(input: array, weight: array, /, stride: Union[int, Sequence[int]] = 1, padding: Union[int, Sequence[int], tuple[Sequence[int], Sequence[int]]] = 0, kernel_dilation: Union[int, Sequence[int]] = 1, ...` | General convolution over an input with several channels |
| [x] | `mlx.core.conv_transpose1d` | `(input: array, weight: array, /, stride: int = 1, padding: int = 0, dilation: int = 1, output_padding: int = 0, groups: int = 1, *, stream: Union[None, Stream, Device] = None) -> array` | 1D transposed convolution over an input with several channels |
| [x] | `mlx.core.conv_transpose2d` | `(input: array, weight: array, /, stride: Union[int, Tuple[int, int]] = 1, padding: Union[int, Tuple[int, int]] = 0, dilation: Union[int, Tuple[int, int]] = 1, output_padding: Union[int, Tuple[int, ...` | 2D transposed convolution over an input with several channels |
| [x] | `mlx.core.conv_transpose3d` | `(input: array, weight: array, /, stride: Union[int, Tuple[int, int, int]] = 1, padding: Union[int, Tuple[int, int, int]] = 0, dilation: Union[int, Tuple[int, int, int]] = 1, output_padding: Union[i...` | 3D transposed convolution over an input with several channels |
| [x] | `mlx.core.convolve` | `(a: array, v: array, /, mode: str = "full", *, stream: Union[None, Stream, Device] = None) -> array` | The discrete convolution of 1D arrays. |
| [x] | `mlx.core.dequantize` | `(w: array, /, scales: array, biases: Optional[array] = None, group_size: int = 64, bits: int = 4, mode: str = 'affine', *, stream: Union[None, Stream, Device] = None) -> array` | Dequantize the matrix w using quantization parameters. |
| [x] | `mlx.core.distributed` | `(...)` | mlx.core.distributed: Communication operations |
| [x] | `mlx.core.einsum` | `(subscripts: str, *operands, stream: Union[None, Stream, Device] = None) -> array` | Perform the Einstein summation convention on the operands. |
| [x] | `mlx.core.einsum_path` | `(subscripts: str, *operands)` | Compute the contraction order for the given Einstein summation. |
| [x] | `mlx.core.export_function` | `(arg0: object, fun: collections.abc.Callable, *args, shapeless: bool = False, **kwargs) -> None` | Export an MLX function. |
| [x] | `mlx.core.exporter` | `(file: str, fun: collections.abc.Callable, *, shapeless: bool = False) -> mlx.core.FunctionExporter` | Make a callable object to export multiple traces of a function to a file. |
| [x] | `mlx.core.fast` | `(...)` | mlx.core.fast: fast operations |
| [x] | `mlx.core.flatten` | `(a: array, /, start_axis: int = 0, end_axis: int = -1, *, stream: Union[None, Stream, Device] = None) -> array` | Flatten an array. |
| [x] | `mlx.core.floating` | `(...)` | No documentation available. |
| [x] | `mlx.core.gather_qmm` | `(x: array, w: array, /, scales: array, biases: Optional[array] = None, lhs_indices: Optional[array] = None, rhs_indices: Optional[array] = None, transpose: bool = True, group_size: int = 64, bits: ...` | Perform quantized matrix multiplication with matrix-level gather. |
| [x] | `mlx.core.generic` | `(...)` | No documentation available. |
| [x] | `mlx.core.get_active_memory` | `() -> int` | Get the actively used memory in bytes. |
| [x] | `mlx.core.get_cache_memory` | `() -> int` | Get the cache size in bytes. |
| [x] | `mlx.core.hadamard_transform` | `(a: array, scale: Optional[float] = None, stream: Union[None, Stream, Device] = None) -> array` | Perform the Walsh-Hadamard transform along the final axis. |
| [x] | `mlx.core.identity` | `(n: int, dtype: Optional[Dtype] = float32, *, stream: Union[None, Stream, Device] = None) -> array` | Create a square identity matrix. |
| [x] | `mlx.core.import_function` | `(file: str) -> Callable` | Import a function from a file. |
| [x] | `mlx.core.inexact` | `(...)` | No documentation available. |
| [x] | `mlx.core.integer` | `(...)` | No documentation available. |
| [x] | `mlx.core.linalg` | `(...)` | mlx.core.linalg: linear algebra routines. |
| [x] | `mlx.core.load` | `(file: Union[file, str, pathlib.Path], /, format: Optional[str] = None, return_metadata: bool = False, *, stream: Union[None, Stream, Device] = None) -> Union[array, dict[str, array]]` | Load array(s) from a binary file. |
| [x] | `mlx.core.metal` | `(...)` | mlx.metal |
| [x] | `mlx.core.number` | `(...)` | No documentation available. |
| [x] | `mlx.core.permute_dims` | `(a: array, /, axes: Optional[Sequence[int]] = None, *, stream: Union[None, Stream, Device] = None) -> array` | See :func:transpose. |
| [x] | `mlx.core.quantize` | `(w: array, /, group_size: int = 64, bits: int = 4, mode: str = 'affine', *, stream: Union[None, Stream, Device] = None) -> tuple[array, array, array]` | Quantize the matrix w using bits bits per element. |
| [x] | `mlx.core.quantized_matmul` | `(x: array, w: array, /, scales: array, biases: Optional[array] = None, transpose: bool = True, group_size: int = 64, bits: int = 4, mode: str = 'affine', *, stream: Union[None, Stream, Device] = No...` | Perform the matrix multiplication with the quantized matrix w. The |
| [x] | `mlx.core.random.state` | `(...)` | Built-in mutable sequence. |
| [x] | `mlx.core.reset_peak_memory` | `() -> None` | Reset the peak memory to zero. |
| [x] | `mlx.core.save` | `(file: Union[file, str, pathlib.Path], arr: array) -> None` | Save the array to a binary file in .npy format. |
| [x] | `mlx.core.save_gguf` | `(file: Union[file, str, pathlib.Path], arrays: dict[str, array], metadata: dict[str, Union[array, str, list[str]]])` | Save array(s) to a binary file in .gguf format. |
| [x] | `mlx.core.save_safetensors` | `(file: Union[file, str, pathlib.Path], arrays: dict[str, array], metadata: Optional[dict[str, str]] = None)` | Save array(s) to a binary file in .safetensors format. |
| [x] | `mlx.core.savez` | `(file: Union[file, str, pathlib.Path], *args, **kwargs)` | Save several arrays to a binary file in uncompressed .npz |
| [x] | `mlx.core.savez_compressed` | `(file: Union[file, str, pathlib.Path], *args, **kwargs)` | Save several arrays to a binary file in compressed .npz format. |
| [x] | `mlx.core.set_cache_limit` | `(limit: int) -> int` | Set the free cache limit. |
| [x] | `mlx.core.set_default_stream` | `(stream: mlx.core.Stream) -> None` | Set the default stream. |
| [x] | `mlx.core.set_memory_limit` | `(limit: int) -> int` | Set the memory limit. |
| [x] | `mlx.core.set_wired_limit` | `(limit: int) -> int` | Set the wired size limit. |
| [x] | `mlx.core.signedinteger` | `(...)` | No documentation available. |
| [x] | `mlx.core.slice` | `(a: array, start_indices: array, axes: Sequence[int], slice_size: Sequence[int], *, stream: Union[None, Stream, Device] = None) -> array` | Extract a sub-array from the input array. |
| [x] | `mlx.core.slice_update` | `(a: array, update: array, start_indices: array, axes: Sequence[int], *, stream: Union[None, Stream, Device] = None) -> array` | Update a sub-array of the input array. |
| [x] | `mlx.core.tan` | `(a: array, /, *, stream: Union[None, Stream, Device] = None) -> array` | Element-wise tangent. |
| [x] | `mlx.core.topk` | `(a: array, /, k: int, axis: Union[None, int] = -1, *, stream: Union[None, Stream, Device] = None) -> array` | Returns the k largest elements from the input along a given axis. |
| [x] | `mlx.core.unflatten` | `(a: array, /, axis: int, shape: Sequence[int], *, stream: Union[None, Stream, Device] = None) -> array` | Unflatten an axis of an array to a shape. |
| [x] | `mlx.core.unsignedinteger` | `(...)` | No documentation available. |
| [x] | `mlx.nn.ALiBi` | `()` | No documentation available. |
| [x] | `mlx.nn.AllToShardedLinear` | `(input_dims: int, output_dims: int, bias: bool = True, group: Optional[mlx.core.distributed.Group] = None)` | Each member of the group applies part of the affine transformation such |
| [x] | `mlx.nn.AvgPool1d` | `(kernel_size: Union[int, Tuple[int]], stride: Union[int, Tuple[int], NoneType] = None, padding: Union[int, Tuple[int]] = 0)` | Applies 1-dimensional average pooling. |
| [x] | `mlx.nn.AvgPool2d` | `(kernel_size: Union[int, Tuple[int, int]], stride: Union[int, Tuple[int, int], NoneType] = None, padding: Union[int, Tuple[int, int], NoneType] = 0)` | Applies 2-dimensional average pooling. |
| [x] | `mlx.nn.AvgPool3d` | `(kernel_size: Union[int, Tuple[int, int, int]], stride: Union[int, Tuple[int, int, int], NoneType] = None, padding: Union[int, Tuple[int, int, int], NoneType] = 0)` | Applies 3-dimensional average pooling. |
| [x] | `mlx.nn.BatchNorm` | `(num_features: int, eps: float = 1e-05, momentum: float = 0.1, affine: bool = True, track_running_stats: bool = True)` | Applies Batch Normalization over a 2D or 3D input. |
| [x] | `mlx.nn.Bilinear` | `(input1_dims: int, input2_dims: int, output_dims: int, bias: bool = True) -> None` | Applies a bilinear transformation to the inputs. |
| [x] | `mlx.nn.CELU` | `(alpha=1.0)` | Applies the Continuously Differentiable Exponential Linear Unit. |
| [x] | `mlx.nn.Conv1d` | `(in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, dilation: int = 1, groups: int = 1, bias: bool = True)` | Applies a 1-dimensional convolution over the multi-channel input sequence. |
| [x] | `mlx.nn.Conv2d` | `(in_channels: int, out_channels: int, kernel_size: Union[int, tuple], stride: Union[int, tuple] = 1, padding: Union[int, tuple] = 0, dilation: Union[int, tuple] = 1, groups: int = 1, bias: bool = T...` | Applies a 2-dimensional convolution over the multi-channel input image. |
| [x] | `mlx.nn.Conv3d` | `(in_channels: int, out_channels: int, kernel_size: Union[int, tuple], stride: Union[int, tuple] = 1, padding: Union[int, tuple] = 0, dilation: Union[int, tuple] = 1, bias: bool = True)` | Applies a 3-dimensional convolution over the multi-channel input image. |
| [x] | `mlx.nn.ConvTranspose1d` | `(in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, dilation: int = 1, output_padding: int = 0, bias: bool = True)` | Applies a 1-dimensional transposed convolution over the multi-channel input sequence. |
| [x] | `mlx.nn.ConvTranspose2d` | `(in_channels: int, out_channels: int, kernel_size: Union[int, tuple], stride: Union[int, tuple] = 1, padding: Union[int, tuple] = 0, dilation: Union[int, tuple] = 1, output_padding: Union[int, tupl...` | Applies a 2-dimensional transposed convolution over the multi-channel input image. |
| [x] | `mlx.nn.ConvTranspose3d` | `(in_channels: int, out_channels: int, kernel_size: Union[int, tuple], stride: Union[int, tuple] = 1, padding: Union[int, tuple] = 0, dilation: Union[int, tuple] = 1, output_padding: Union[int, tupl...` | Applies a 3-dimensional transposed convolution over the multi-channel input image. |
| [x] | `mlx.nn.Dropout` | `(p: float = 0.5)` | Randomly zero a portion of the elements during training. |
| [x] | `mlx.nn.Dropout2d` | `(p: float = 0.5)` | Apply 2D channel-wise dropout during training. |
| [x] | `mlx.nn.Dropout3d` | `(p: float = 0.5)` | Apply 3D channel-wise dropout during training. |
| [x] | `mlx.nn.ELU` | `(alpha=1.0)` | Applies the Exponential Linear Unit. |
| [x] | `mlx.nn.Embedding` | `(num_embeddings: int, dims: int)` | Implements a simple lookup table that maps each input integer to a |
| [x] | `mlx.nn.GELU` | `(approx='none')` | Applies the Gaussian Error Linear Units. |
| [x] | `mlx.nn.GLU` | `(axis: int = -1)` | Applies the gated linear unit function. |
| [x] | `mlx.nn.GRU` | `(input_size: int, hidden_size: int, bias: bool = True)` | A gated recurrent unit (GRU) RNN layer. |
| [x] | `mlx.nn.GroupNorm` | `(num_groups: int, dims: int, eps: float = 1e-05, affine: bool = True, pytorch_compatible: bool = False)` | Applies Group Normalization [1] to the inputs. |
| [x] | `mlx.nn.HardShrink` | `()` | Applies the HardShrink function. |
| [x] | `mlx.nn.HardTanh` | `()` | Applies the HardTanh function. |
| [x] | `mlx.nn.Hardswish` | `()` | Applies the hardswish function, element-wise. |
| [x] | `mlx.nn.Identity` | `(*args: Any, **kwargs: Any) -> None` | A placeholder identity operator that is argument-insensitive. |
| [x] | `mlx.nn.InstanceNorm` | `(dims: int, eps: float = 1e-05, affine: bool = False)` | Applies instance normalization [1] on the inputs. |
| [x] | `mlx.nn.LSTM` | `(input_size: int, hidden_size: int, bias: bool = True)` | An LSTM recurrent layer. |
| [x] | `mlx.nn.LayerNorm` | `(dims: int, eps: float = 1e-05, affine: bool = True, bias: bool = True)` | Applies layer normalization [1] on the inputs. |
| [x] | `mlx.nn.LeakyReLU` | `(negative_slope=0.01)` | Applies the Leaky Rectified Linear Unit. |
| [x] | `mlx.nn.Linear` | `(input_dims: int, output_dims: int, bias: bool = True) -> None` | Applies an affine transformation to the input. |
| [x] | `mlx.nn.LogSigmoid` | `()` | Applies the Log Sigmoid function. |
| [x] | `mlx.nn.LogSoftmax` | `()` | Applies the Log Softmax function. |
| [x] | `mlx.nn.MaxPool1d` | `(kernel_size: Union[int, Tuple[int]], stride: Union[int, Tuple[int], NoneType] = None, padding: Union[int, Tuple[int]] = 0)` | Applies 1-dimensional max pooling. |
| [x] | `mlx.nn.MaxPool2d` | `(kernel_size: Union[int, Tuple[int, int]], stride: Union[int, Tuple[int, int], NoneType] = None, padding: Union[int, Tuple[int, int], NoneType] = 0)` | Applies 2-dimensional max pooling. |
| [x] | `mlx.nn.MaxPool3d` | `(kernel_size: Union[int, Tuple[int, int, int]], stride: Union[int, Tuple[int, int, int], NoneType] = None, padding: Union[int, Tuple[int, int, int], NoneType] = 0)` | Applies 3-dimensional max pooling. |
| [x] | `mlx.nn.Mish` | `()` | Applies the Mish function, element-wise. |
| [x] | `mlx.nn.Module` | `()` | Base class for building neural networks with MLX. |
| [x] | `mlx.nn.MultiHeadAttention` | `(dims: int, num_heads: int, query_input_dims: Optional[int] = None, key_input_dims: Optional[int] = None, value_input_dims: Optional[int] = None, value_dims: Optional[int] = None, value_output_dims...` | Implements the scaled dot product attention with multiple heads. |
| [x] | `mlx.nn.PReLU` | `(num_parameters=1, init=0.25)` | Applies the element-wise parametric ReLU. |
| [x] | `mlx.nn.QuantizedAllToShardedLinear` | `(input_dims: int, output_dims: int, bias: bool = True, group_size: int = 64, bits: int = 4, group: Optional[mlx.core.distributed.Group] = None)` | Each member of the group applies part of the affine transformation with |
| [x] | `mlx.nn.QuantizedEmbedding` | `(num_embeddings: int, dims: int, group_size: int = 64, bits: int = 4, mode: str = 'affine')` | The same as :obj:Embedding but with a  quantized weight matrix. |
| [x] | `mlx.nn.QuantizedLinear` | `(input_dims: int, output_dims: int, bias: bool = True, group_size: int = 64, bits: int = 4, mode: str = 'affine')` | Applies an affine transformation to the input using a quantized weight matrix. |
| [x] | `mlx.nn.QuantizedShardedToAllLinear` | `(input_dims: int, output_dims: int, bias: bool = True, group_size: int = 64, bits: int = 4, group: Optional[mlx.core.distributed.Group] = None)` | Each member of the group applies part of the affine transformation using |
| [x] | `mlx.nn.RMSNorm` | `(dims: int, eps: float = 1e-05)` | Applies Root Mean Square normalization [1] to the inputs. |
| [x] | `mlx.nn.RNN` | `(input_size: int, hidden_size: int, bias: bool = True, nonlinearity: Optional[Callable] = None)` | An Elman recurrent layer. |
| [x] | `mlx.nn.ReLU` | `()` | Applies the Rectified Linear Unit. |
| [x] | `mlx.nn.ReLU2` | `()` | Applies the ReLU² activation function. |
| [x] | `mlx.nn.ReLU6` | `()` | Applies the Rectified Linear Unit 6. |
| [x] | `mlx.nn.RoPE` | `(dims: int, traditional: bool = False, base: float = 10000, scale: float = 1.0)` | Implements the rotary positional encoding. |
| [x] | `mlx.nn.SELU` | `()` | Applies the Scaled Exponential Linear Unit. |
| [x] | `mlx.nn.Sequential` | `(*modules)` | A layer that calls the passed callables in order. |
| [x] | `mlx.nn.ShardedToAllLinear` | `(input_dims: int, output_dims: int, bias: bool = True, group: Optional[mlx.core.distributed.Group] = None)` | Each member of the group applies part of the affine transformation and |
| [x] | `mlx.nn.SiLU` | `()` | Applies the Sigmoid Linear Unit. Also known as Swish. |
| [x] | `mlx.nn.Sigmoid` | `()` | Applies the sigmoid function, element-wise. |
| [x] | `mlx.nn.SinusoidalPositionalEncoding` | `(dims: int, min_freq: float = 0.0001, max_freq: float = 1, scale: Optional[float] = None, cos_first: bool = False, full_turns: bool = False)` | Implements sinusoidal positional encoding. |
| [x] | `mlx.nn.Softmax` | `()` | Applies the Softmax function. |
| [x] | `mlx.nn.Softmin` | `()` | Applies the Softmin function. |
| [x] | `mlx.nn.Softplus` | `()` | Applies the Softplus function. |
| [x] | `mlx.nn.Softshrink` | `(lambd=0.5)` | Applies the Softshrink function. |
| [x] | `mlx.nn.Softsign` | `()` | Applies the Softsign function. |
| [x] | `mlx.nn.Step` | `(threshold: float = 0.0)` | Applies the Step Activation Function. |
| [x] | `mlx.nn.Tanh` | `()` | Applies the hyperbolic tangent function. |
| [x] | `mlx.nn.Transformer` | `(dims: int = 512, num_heads: int = 8, num_encoder_layers: int = 6, num_decoder_layers: int = 6, mlp_dims: Optional[int] = None, dropout: float = 0.0, activation: Callable[[Any], Any] = <mlx.gc_func...` | Implements a standard Transformer model. |
| [x] | `mlx.nn.TransformerDecoder` | `(num_layers: int, dims: int, num_heads: int, mlp_dims: Optional[int] = None, dropout: float = 0.0, activation=<mlx.gc_func object at 0x10484bf90>, norm_first: bool = True, checkpoint: bool = False)` | No documentation available. |
| [x] | `mlx.nn.TransformerDecoderLayer` | `(dims: int, num_heads: int, mlp_dims: Optional[int] = None, dropout: float = 0.0, activation: Callable[[Any], Any] = <mlx.gc_func object at 0x10484bf90>, norm_first: bool = True)` | No documentation available. |
| [x] | `mlx.nn.TransformerEncoder` | `(num_layers: int, dims: int, num_heads: int, mlp_dims: Optional[int] = None, dropout: float = 0.0, activation=<mlx.gc_func object at 0x10484bf90>, norm_first: bool = True, checkpoint: bool = False)` | No documentation available. |
| [x] | `mlx.nn.TransformerEncoderLayer` | `(dims: int, num_heads: int, mlp_dims: Optional[int] = None, dropout: float = 0.0, activation: Callable[[Any], Any] = <mlx.gc_func object at 0x10484bf90>, norm_first: bool = True)` | No documentation available. |
| [x] | `mlx.nn.Upsample` | `(scale_factor: Union[float, Tuple], mode: Literal['nearest', 'linear', 'cubic'] = 'nearest', align_corners: bool = False)` | Upsample the input signal spatially. |
| [x] | `mlx.nn.activations` | `(...)` | No documentation available. |
| [x] | `mlx.nn.average_gradients` | `(gradients: Any, group: Optional[mlx.core.distributed.Group] = None, all_reduce_size: int = 33554432, communication_type: Optional[mlx.core.Dtype] = None, communication_stream: Optional[mlx.core.St...` | Average the gradients across the distributed processes in the passed group. |
| [x] | `mlx.nn.base` | `(...)` | No documentation available. |
| [x] | `mlx.nn.celu` | `(x, alpha=1.0)` | Applies the Continuously Differentiable Exponential Linear Unit. |
| [x] | `mlx.nn.containers` | `(...)` | No documentation available. |
| [x] | `mlx.nn.convolution` | `(...)` | No documentation available. |
| [x] | `mlx.nn.convolution_transpose` | `(...)` | No documentation available. |
| [x] | `mlx.nn.dropout` | `(...)` | No documentation available. |
| [x] | `mlx.nn.elu` | `(x, alpha=1.0)` | Applies the Exponential Linear Unit. |
| [x] | `mlx.nn.embedding` | `(...)` | No documentation available. |
| [x] | `mlx.nn.gelu` | `(x) -> mlx.core.array` | Applies the Gaussian Error Linear Units function. |
| [x] | `mlx.nn.gelu_approx` | `(x)` | An approximation to Gaussian Error Linear Unit. |
| [x] | `mlx.nn.gelu_fast_approx` | `(x)` | A fast approximation to Gaussian Error Linear Unit. |
| [x] | `mlx.nn.glu` | `(x: mlx.core.array, axis: int = -1) -> mlx.core.array` | Applies the gated linear unit function. |
| [x] | `mlx.nn.hard_shrink` | `(x, lambd=0.5)` | Applies the HardShrink activation function. |
| [x] | `mlx.nn.hard_tanh` | `(x, min_val=-1.0, max_val=1.0)` | Applies the HardTanh function. |
| [x] | `mlx.nn.hardswish` | `(x)` | Applies the hardswish function, element-wise. |
| [x] | `mlx.nn.init` | `(...)` | No documentation available. |
| [x] | `mlx.nn.layers` | `(...)` | No documentation available. |
| [x] | `mlx.nn.leaky_relu` | `(x, negative_slope=0.01)` | Applies the Leaky Rectified Linear Unit. |
| [x] | `mlx.nn.linear` | `(...)` | No documentation available. |
| [x] | `mlx.nn.log_sigmoid` | `(x)` | Applies the Log Sigmoid function. |
| [x] | `mlx.nn.log_softmax` | `(x, axis=-1)` | Applies the Log Softmax function. |
| [x] | `mlx.nn.losses` | `(...)` | No documentation available. |
| [x] | `mlx.nn.mish` | `(x: mlx.core.array) -> mlx.core.array` | Applies the Mish function, element-wise. |
| [x] | `mlx.nn.normalization` | `(...)` | No documentation available. |
| [x] | `mlx.nn.pooling` | `(...)` | No documentation available. |
| [x] | `mlx.nn.positional_encoding` | `(...)` | No documentation available. |
| [x] | `mlx.nn.prelu` | `(x: mlx.core.array, alpha: mlx.core.array) -> mlx.core.array` | Applies the element-wise parametric ReLU. |
| [x] | `mlx.nn.quantized` | `(...)` | No documentation available. |
| [x] | `mlx.nn.recurrent` | `(...)` | No documentation available. |
| [x] | `mlx.nn.relu` | `(x)` | Applies the Rectified Linear Unit. |
| [x] | `mlx.nn.relu2` | `(x)` | Applies the ReLU² activation function. |
| [x] | `mlx.nn.relu6` | `(x)` | Applies the Rectified Linear Unit 6. |
| [x] | `mlx.nn.selu` | `(x)` | Applies the Scaled Exponential Linear Unit. |
| [x] | `mlx.nn.silu` | `(x)` | Applies the Sigmoid Linear Unit. Also known as Swish. |
| [x] | `mlx.nn.softmin` | `(x, axis=-1)` | Applies the Softmin function. |
| [x] | `mlx.nn.softplus` | `(x)` | Applies the Softplus function. |
| [x] | `mlx.nn.softshrink` | `(x, lambd: float = 0.5)` | Applies the Softshrink activation function. |
| [x] | `mlx.nn.softsign` | `(x)` | Applies the Softsign function. |
| [x] | `mlx.nn.step` | `(x: mlx.core.array, threshold: float = 0.0)` | Applies the Step Activation Function. |
| [x] | `mlx.nn.transformer` | `(...)` | No documentation available. |
| [x] | `mlx.nn.upsample` | `(...)` | No documentation available. |
| [x] | `mlx.nn.utils` | `(...)` | No documentation available. |
| [x] | `mlx.optimizers.AdaDelta` | `(learning_rate: Union[float, Callable[[mlx.core.array], mlx.core.array]], rho: float = 0.9, eps: float = 1e-06)` | The AdaDelta optimizer with a learning rate [1]. |
| [x] | `mlx.optimizers.Adafactor` | `(learning_rate: Union[float, Callable[[mlx.core.array], mlx.core.array], NoneType] = None, eps: Tuple[float, float] = (1e-30, 0.001), clip_threshold: float = 1.0, decay_rate: float = -0.8, beta_1: ...` | The Adafactor optimizer. |
| [x] | `mlx.optimizers.Adagrad` | `(learning_rate: Union[float, Callable[[mlx.core.array], mlx.core.array]], eps: float = 1e-08)` | The Adagrad optimizer [1]. |
| [x] | `mlx.optimizers.Adam` | `(learning_rate: Union[float, Callable[[mlx.core.array], mlx.core.array]], betas: List[float] = [0.9, 0.999], eps: float = 1e-08, bias_correction: bool = False)` | The Adam optimizer [1]. In detail, |
| [x] | `mlx.optimizers.AdamW` | `(learning_rate: Union[float, Callable[[mlx.core.array], mlx.core.array]], betas: List[float] = [0.9, 0.999], eps: float = 1e-08, weight_decay: float = 0.01, bias_correction: bool = False)` | The AdamW optimizer [1]. We update the weights with a weight_decay |
| [x] | `mlx.optimizers.Adamax` | `(learning_rate: Union[float, Callable[[mlx.core.array], mlx.core.array]], betas: List[float] = [0.9, 0.999], eps: float = 1e-08)` | The Adamax optimizer, a variant of Adam based on the infinity norm [1]. |
| [x] | `mlx.optimizers.Callable` | `(*args, **kwargs)` | Callable type; Callable[[int], str] is a function of (int) -> str. |
| [x] | `mlx.optimizers.Lion` | `(learning_rate: Union[float, Callable[[mlx.core.array], mlx.core.array]], betas: List[float] = [0.9, 0.99], weight_decay: float = 0.0)` | The Lion optimizer [1]. |
| [x] | `mlx.optimizers.List` | `(*args, **kwargs)` | A generic version of list. |
| [x] | `mlx.optimizers.MultiOptimizer` | `(optimizers, filters: list = [])` | Wraps a list of optimizers with corresponding weight predicates/filters |
| [x] | `mlx.optimizers.Muon` | `(learning_rate: Union[float, Callable[[mlx.core.array], mlx.core.array]], momentum: float = 0.95, weight_decay: float = 0.01, nesterov: bool = True, ns_steps: int = 5)` | The Muon optimizer. |
| [x] | `mlx.optimizers.Optimizer` | `(schedulers=None)` | The base class for all optimizers. It allows us to implement an |
| [x] | `mlx.optimizers.Optional` | `(*args, **kwds)` | Optional type. |
| [x] | `mlx.optimizers.RMSprop` | `(learning_rate: Union[float, Callable[[mlx.core.array], mlx.core.array]], alpha: float = 0.99, eps: float = 1e-08)` | The RMSprop optimizer [1]. |
| [x] | `mlx.optimizers.SGD` | `(learning_rate: Union[float, Callable[[mlx.core.array], mlx.core.array]], momentum: float = 0.0, weight_decay: float = 0.0, dampening: float = 0.0, nesterov: bool = False)` | The stochastic gradient descent optimizer. |
| [x] | `mlx.optimizers.Tuple` | `(*args, **kwargs)` | Tuple type; Tuple[X, Y] is the cross-product type of X and Y. |
| [x] | `mlx.optimizers.Union` | `(*args, **kwds)` | Union type; Union[X, Y] means either X or Y. |
| [x] | `mlx.optimizers.clip_grad_norm` | `(grads, max_norm)` | Clips the global norm of the gradients. |
| [x] | `mlx.optimizers.cosine_decay` | `(init: float, decay_steps: int, end: float = 0.0) -> Callable` | Make a cosine decay scheduler. |
| [x] | `mlx.optimizers.exponential_decay` | `(init: float, decay_rate: float) -> Callable` | Make an exponential decay scheduler. |
| [x] | `mlx.optimizers.join_schedules` | `(schedules: List[Callable], boundaries: List[int]) -> Callable` | Join multiple schedules to create a new schedule. |
| [x] | `mlx.optimizers.linear_schedule` | `(init: float, end: float, steps: int) -> Callable` | Make a linear scheduler. |
| [x] | `mlx.optimizers.math` | `(...)` | This module provides access to the mathematical functions |
| [x] | `mlx.optimizers.mx` | `(...)` | mlx: A framework for machine learning on Apple silicon. |
| [x] | `mlx.optimizers.optimizers` | `(...)` | No documentation available. |
| [x] | `mlx.optimizers.schedulers` | `(...)` | No documentation available. |
| [x] | `mlx.optimizers.step_decay` | `(init: float, decay_rate: float, step_size: int) -> Callable` | Make a step decay scheduler. |
| [x] | `mlx.optimizers.tree_flatten` | `(tree: Any, prefix: str = '', is_leaf: Optional[Callable] = None, destination: Union[List[Tuple[str, Any]], Dict[str, Any], NoneType] = None) -> Union[List[Tuple[str, Any]], Dict[str, Any]]` | Flattens a Python tree to a list of key, value tuples. |
| [x] | `mlx.optimizers.tree_map` | `(fn: Callable, tree: Any, *rest: Any, is_leaf: Optional[Callable] = None) -> Any` | Applies fn to the leaves of the Python tree tree and |
| [x] | `mlx.optimizers.tree_merge` | `(tree_a, tree_b, merge_fn=None)` | Merge two Python trees in one containing the values of both. It can be |
| [x] | `mlx.optimizers.tree_reduce` | `(fn, tree, initializer=None, is_leaf=None)` | Applies a reduction to the leaves of a Python tree. |
| [x] | `mlx.optimizers.tree_unflatten` | `(tree: Union[List[Tuple[str, Any]], Dict[str, Any]]) -> Any` | Recreate a Python tree from its flat representation. |
| [x] | `mlx.utils.Any` | `(*args, **kwds)` | Special type indicating an unconstrained type. |
| [x] | `mlx.utils.Dict` | `(*args, **kwargs)` | A generic version of dict. |
| [x] | `mlx.utils.defaultdict` | `(default_factory=None, /, [...]) --> dict with default factory` | The default factory is called without arguments to produce |
| [x] | `mlx.utils.tree_map_with_path` | `(fn: Callable, tree: Any, *rest: Any, is_leaf: Optional[Callable] = None, path: Optional[Any] = None) -> Any` | Applies fn to the path and leaves of the Python tree tree and |
| [x] | `mlx.utils.zip_longest` | `(iter1 [,iter2 [...]], [fillvalue=None]) --> zip_longest object` | Return a zip_longest object whose .__next__() method returns a tuple where |
