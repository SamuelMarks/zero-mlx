Extracting target APIs from /Users/samuel/repos/zero-mlx/src...
Scoring compliance...

--- Compliance Report ---
Overall Compliance: 0.0%

Breakdown by Module:
  - mlx.nn: 0.0% (0/77)
  - mlx.nn.losses: 0.0% (0/12)
  - mlx.optimizers: 0.0% (0/13)

Missing APIs (102):

|   | Framework | Namespace | Symbol | FQN | Signature | Docstring |
|---|---|---|---|---|---|---|
| [x] | mlx | mlx.nn | ALiBi | mlx.nn.ALiBi | `(_alibi_mask_key='```(None)```', _alibi_mask='```(None)```')` | Base class for building neural networks with MLX. |
| [x] | mlx | mlx.nn | AllToShardedLinear | mlx.nn.AllToShardedLinear | `(input_dims: int, output_dims: int, bias: bool | None=True, group: mx.distributed.Group | None='```(None)```')` | Each member of the group applies part of the affine transformation such that the result is sharde... |
| [x] | mlx | mlx.nn | AvgPool1d | mlx.nn.AvgPool1d | `(kernel_size: int | tuple(int), stride: int | tuple(int) | optional='kernel_size', padding: int | tuple(int) | optional=0)` | Applies 1-dimensional average pooling. |
| [x] | mlx | mlx.nn | AvgPool2d | mlx.nn.AvgPool2d | `(kernel_size: int | tuple(int, int), stride: int | tuple(int, int) | optional='kernel_size', padding: int | tuple(int, int) | optional=0)` | Applies 2-dimensional average pooling. |
| [x] | mlx | mlx.nn | AvgPool3d | mlx.nn.AvgPool3d | `(kernel_size: int | tuple(int, int, int), stride: int | tuple(int, int, int) | optional='kernel_size', padding: int | tuple(int, int, int) | optional=0)` | Applies 3-dimensional average pooling. |
| [x] | mlx | mlx.nn | BatchNorm | mlx.nn.BatchNorm | `(num_features: int, eps: float | None=1e-05, momentum: float | None=0.1, affine: bool | None=True, track_running_stats: bool | None=True)` | Applies Batch Normalization over a 2D or 3D input. |
| [x] | mlx | mlx.nn | Bilinear | mlx.nn.Bilinear | `(input1_dims: int, input2_dims: int, output_dims: int, bias: bool | None=True) -> NoneType` | Applies a bilinear transformation to the inputs. |
| [x] | mlx | mlx.nn | CELU | mlx.nn.CELU | `(alpha: float=1.0)` | Applies the Continuously Differentiable Exponential Linear Unit.     Applies :math:`\max(0, x) + ... |
| [x] | mlx | mlx.nn | Conv1d | mlx.nn.Conv1d | `(in_channels: int, out_channels: int, kernel_size: int, stride: int | None=1, padding: int | None=0, dilation: int | None=1, groups: int | None=1, bias: bool | None=True)` | Applies a 1-dimensional convolution over the multi-channel input sequence. |
| [x] | mlx | mlx.nn | Conv2d | mlx.nn.Conv2d | `(in_channels: int, out_channels: int, kernel_size: int | tuple, stride: int | tuple | optional=1, padding: int | tuple | optional=0, dilation: int | tuple | optional=1, groups: int | None=1, bias: bool | None=True)` | Applies a 2-dimensional convolution over the multi-channel input image. |
| [x] | mlx | mlx.nn | Conv3d | mlx.nn.Conv3d | `(in_channels: int, out_channels: int, kernel_size: int | tuple, stride: int | tuple | optional=1, dilation: int | tuple | optional=1, padding: int | tuple | optional=0, bias: bool | None=True)` | Applies a 3-dimensional convolution over the multi-channel input image. |
| [x] | mlx | mlx.nn | ConvTranspose1d | mlx.nn.ConvTranspose1d | `(in_channels: int, out_channels: int, kernel_size: int, stride: int | None=1, padding: int | None=0, dilation: int | None=1, output_padding: int | None=0, bias: bool | None=True)` | Applies a 1-dimensional transposed convolution over the multi-channel input sequence. |
| [x] | mlx | mlx.nn | ConvTranspose2d | mlx.nn.ConvTranspose2d | `(in_channels: int, out_channels: int, kernel_size: int | tuple, stride: int | tuple | optional=1, padding: int | tuple | optional=0, dilation: int | tuple | optional=1, output_padding: int | tuple | optional=0, bias: bool | None=True)` | Applies a 2-dimensional transposed convolution over the multi-channel input image. |
| [x] | mlx | mlx.nn | ConvTranspose3d | mlx.nn.ConvTranspose3d | `(in_channels: int, out_channels: int, kernel_size: int | tuple, stride: int | tuple | optional=1, padding: int | tuple | optional=0, dilation: int | tuple | optional=1, output_padding: int | tuple | optional=0, bias: bool | None=True)` | Applies a 3-dimensional transposed convolution over the multi-channel input image. |
| [x] | mlx | mlx.nn | Dropout | mlx.nn.Dropout | `(p: float=0.5)` | Randomly zero a portion of the elements during training. |
| [x] | mlx | mlx.nn | Dropout2d | mlx.nn.Dropout2d | `(p: float=0.5)` | Apply 2D channel-wise dropout during training. |
| [x] | mlx | mlx.nn | Dropout3d | mlx.nn.Dropout3d | `(p: float=0.5)` | Apply 3D channel-wise dropout during training. |
| [x] | mlx | mlx.nn | ELU | mlx.nn.ELU | `(alpha: float=1.0)` | Applies the Exponential Linear Unit.     Simply ``mx.where(x > 0, x, alpha * (mx.exp(x) - 1))``. |
| [x] | mlx | mlx.nn | Embedding | mlx.nn.Embedding | `(num_embeddings: int, dims: int)` | Implements a simple lookup table that maps each input integer to a high-dimensional vector. |
| [x] | mlx | mlx.nn | GELU | mlx.nn.GELU | `(approx: 'none' | 'precise' | 'fast'='none')` | Applies the Gaussian Error Linear Units. |
| [x] | mlx | mlx.nn | GLU | mlx.nn.GLU | `(axis: int=-1)` | Applies the gated linear unit function. |
| [x] | mlx | mlx.nn | GRU | mlx.nn.GRU | `(input_size: int, hidden_size: int, bias: bool=True)` | A gated recurrent unit (GRU) RNN layer. |
| [x] | mlx | mlx.nn | GroupNorm | mlx.nn.GroupNorm | `(num_groups: int, dims: int, eps: float=1e-05, affine: bool=True, pytorch_compatible: bool=False)` | Applies Group Normalization [1] to the inputs. |
| [x] | mlx | mlx.nn | HardShrink | mlx.nn.HardShrink | `(lambd: float=0.5)` | Applies the HardShrink function. |
| [x] | mlx | mlx.nn | HardTanh | mlx.nn.HardTanh | `()` | Applies the HardTanh function. |
| [x] | mlx | mlx.nn | Hardswish | mlx.nn.Hardswish | `()` | Applies the hardswish function, element-wise. |
| [x] | mlx | mlx.nn | Identity | mlx.nn.Identity | `(args: Any, kwargs: dict | None) -> NoneType` | A placeholder identity operator that is argument-insensitive. |
| [x] | mlx | mlx.nn | InstanceNorm | mlx.nn.InstanceNorm | `(dims: int, eps: float=1e-05, affine: bool=False)` | Applies instance normalization [1] on the inputs. |
| [x] | mlx | mlx.nn | LSTM | mlx.nn.LSTM | `(input_size: int, hidden_size: int, bias: bool=True)` | An LSTM recurrent layer. |
| [x] | mlx | mlx.nn | LayerNorm | mlx.nn.LayerNorm | `(dims: int, eps: float=1e-05, affine: bool=True, bias: bool=True)` | Applies layer normalization [1] on the inputs. |
| [x] | mlx | mlx.nn | LeakyReLU | mlx.nn.LeakyReLU | `(negative_slope: float=0.01)` | Applies the Leaky Rectified Linear Unit. |
| [x] | mlx | mlx.nn | Linear | mlx.nn.Linear | `(input_dims: int, output_dims: int, bias: bool | None=True) -> NoneType` | Applies an affine transformation to the input. |
| [x] | mlx | mlx.nn | LogSigmoid | mlx.nn.LogSigmoid | `()` | Applies the Log Sigmoid function. |
| [x] | mlx | mlx.nn | LogSoftmax | mlx.nn.LogSoftmax | `()` | Applies the Log Softmax function. |
| [x] | mlx | mlx.nn | MaxPool1d | mlx.nn.MaxPool1d | `(kernel_size: int | tuple(int), stride: int | tuple(int) | optional='kernel_size', padding: int | tuple(int) | optional=0)` | Applies 1-dimensional max pooling. |
| [x] | mlx | mlx.nn | MaxPool2d | mlx.nn.MaxPool2d | `(kernel_size: int | tuple(int, int), stride: int | tuple(int, int) | optional='kernel_size', padding: int | tuple(int, int) | optional=0)` | Applies 2-dimensional max pooling. |
| [x] | mlx | mlx.nn | MaxPool3d | mlx.nn.MaxPool3d | `(kernel_size: int | tuple(int, int, int), stride: int | tuple(int, int, int) | optional='kernel_size', padding: int | tuple(int, int, int) | optional=0)` | Applies 3-dimensional max pooling. |
| [x] | mlx | mlx.nn | Mish | mlx.nn.Mish | `()` | Applies the Mish function, element-wise. |
| [x] | mlx | mlx.nn | Module | mlx.nn.Module | `(__call__: Callable)` | Base class for building neural networks with MLX. |
| [x] | mlx | mlx.nn | MultiHeadAttention | mlx.nn.MultiHeadAttention | `(dims: int, num_heads: int, query_input_dims: int | None='dims', key_input_dims: int | None='dims', value_input_dims: int | None='key_input_dims', value_dims: int | None='dims', value_output_dims: int | None='dims', bias: bool | None=False)` | Implements the scaled dot product attention with multiple heads. |
| [x] | mlx | mlx.nn | PReLU | mlx.nn.PReLU | `(num_parameters: int=1, init: float=0.25)` | Applies the element-wise parametric ReLU.     Applies :math:`\max(0, x) + a * \min(0, x)` element... |
| [x] | mlx | mlx.nn | QuantizedAllToShardedLinear | mlx.nn.QuantizedAllToShardedLinear | `(input_dims: int, output_dims: int, bias: bool | None=True, group_size: int | None=64, bits: int | None=4, group: mx.distributed.Group | None='```(None)```')` | Each member of the group applies part of the affine transformation with a quantized matrix such t... |
| [x] | mlx | mlx.nn | QuantizedEmbedding | mlx.nn.QuantizedEmbedding | `(num_embeddings: int, dims: int, group_size: int | None=64, bits: int | None=4, mode: str='affine')` | The same as :obj:`Embedding` but with a  quantized weight matrix. |
| [x] | mlx | mlx.nn | QuantizedLinear | mlx.nn.QuantizedLinear | `(input_dims: int, output_dims: int, bias: bool | None=True, group_size: int | None=64, bits: int | None=4, mode: str='affine')` | Applies an affine transformation to the input using a quantized weight matrix. |
| [x] | mlx | mlx.nn | QuantizedShardedToAllLinear | mlx.nn.QuantizedShardedToAllLinear | `(input_dims: int, output_dims: int, bias: bool | None=True, group_size: int | None=64, bits: int | None=4, group: mx.distributed.Group | None='```(None)```')` | Each member of the group applies part of the affine transformation using the quantized matrix and... |
| [x] | mlx | mlx.nn | RMSNorm | mlx.nn.RMSNorm | `(dims: int, eps: float=1e-05)` | Applies Root Mean Square normalization [1] to the inputs. |
| [x] | mlx | mlx.nn | RNN | mlx.nn.RNN | `(input_size: int, hidden_size: int, bias: bool | None=True, nonlinearity: callable | None='```(None)```')` | An Elman recurrent layer. |
| [x] | mlx | mlx.nn | ReLU | mlx.nn.ReLU | `()` | Applies the Rectified Linear Unit.     Simply ``mx.maximum(x, 0)``. |
| [x] | mlx | mlx.nn | ReLU2 | mlx.nn.ReLU2 | `()` | Applies the ReLU² activation function. |
| [x] | mlx | mlx.nn | ReLU6 | mlx.nn.ReLU6 | `()` | Applies the Rectified Linear Unit 6. |
| [x] | mlx | mlx.nn | RoPE | mlx.nn.RoPE | `(dims: int, traditional: bool | None=False, base: float | None=10000, scale: float | None=1.0)` | Implements the rotary positional encoding. |
| [x] | mlx | mlx.nn | SELU | mlx.nn.SELU | `()` | Applies the Scaled Exponential Linear Unit. |
| [x] | mlx | mlx.nn | Sequential | mlx.nn.Sequential | `(modules: tuple of Callables)` | A layer that calls the passed callables in order. |
| [x] | mlx | mlx.nn | ShardedToAllLinear | mlx.nn.ShardedToAllLinear | `(input_dims: int, output_dims: int, bias: bool | None=True, group: mx.distributed.Group | None='```(None)```')` | Each member of the group applies part of the affine transformation and then aggregates the results. |
| [x] | mlx | mlx.nn | SiLU | mlx.nn.SiLU | `()` | Applies the Sigmoid Linear Unit. Also known as Swish. |
| [x] | mlx | mlx.nn | Sigmoid | mlx.nn.Sigmoid | `()` | Applies the sigmoid function, element-wise. |
| [x] | mlx | mlx.nn | SinusoidalPositionalEncoding | mlx.nn.SinusoidalPositionalEncoding | `(dims: int, min_freq: float | None='\n            ``0.0001', max_freq: float | None='\n            ``1', scale: float | None='sqrt(2/dims)``.', cos_first: bool | None=False, full_turns: bool | None=False)` | Implements sinusoidal positional encoding. |
| [x] | mlx | mlx.nn | Softmax | mlx.nn.Softmax | `()` | Applies the Softmax function. |
| [x] | mlx | mlx.nn | Softmin | mlx.nn.Softmin | `()` | Applies the Softmin function. |
| [x] | mlx | mlx.nn | Softplus | mlx.nn.Softplus | `()` | Applies the Softplus function. |
| [x] | mlx | mlx.nn | Softshrink | mlx.nn.Softshrink | `(lambd: float=0.5)` | Applies the Softshrink function. |
| [x] | mlx | mlx.nn | Softsign | mlx.nn.Softsign | `()` | Applies the Softsign function. |
| [x] | mlx | mlx.nn | Step | mlx.nn.Step | `(threshold: float=0.0)` | Applies the Step Activation Function. |
| [x] | mlx | mlx.nn | Tanh | mlx.nn.Tanh | `()` | Applies the hyperbolic tangent function. |
| [x] | mlx | mlx.nn | Transformer | mlx.nn.Transformer | `(dims: int | None=512, num_heads: int | None='\n            ``8', num_encoder_layers: int | None=6, num_decoder_layers: int | None=6, mlp_dims: int | None='4*dims`` if not provided', dropout: float | None=0.0, activation: function | None=':func:`mlx', custom_encoder: nn.Module | None='```(None)```', custom_decoder: nn.Module | None='```(None)```', norm_first: bool | None=True, checkpoint: bool | None=False)` | Implements a standard Transformer model. |
| [x] | mlx | mlx.nn | TransformerDecoder | mlx.nn.TransformerDecoder | `(num_layers: int, dims: int, num_heads: int, mlp_dims: int | None='```(None)```', dropout: float=0.0, activation='```(relu)```', norm_first: bool=True, checkpoint: bool=False)` | Base class for building neural networks with MLX. |
| [x] | mlx | mlx.nn | TransformerDecoderLayer | mlx.nn.TransformerDecoderLayer | `(dims: int, num_heads: int, mlp_dims: int | None='```(None)```', dropout: float=0.0, activation: Callable[[Any], Any]='```(relu)```', norm_first: bool=True)` | Base class for building neural networks with MLX. |
| [x] | mlx | mlx.nn | TransformerEncoder | mlx.nn.TransformerEncoder | `(num_layers: int, dims: int, num_heads: int, mlp_dims: int | None='```(None)```', dropout: float=0.0, activation='```(relu)```', norm_first: bool=True, checkpoint: bool=False)` | Base class for building neural networks with MLX. |
| [x] | mlx | mlx.nn | TransformerEncoderLayer | mlx.nn.TransformerEncoderLayer | `(dims: int, num_heads: int, mlp_dims: int | None='```(None)```', dropout: float=0.0, activation: Callable[[Any], Any]='```(relu)```', norm_first: bool=True)` | Base class for building neural networks with MLX. |
| [x] | mlx | mlx.nn | Upsample | mlx.nn.Upsample | `(scale_factor: float | tuple, mode: Literal['nearest', 'linear', 'cubic']='nearest', align_corners: bool=False)` | Upsample the input signal spatially. |
| [x] | mlx | mlx.nn | elu | mlx.nn.elu | `(x, alpha: float=1.0)` | Applies the Exponential Linear Unit. |
| [x] | mlx | mlx.nn | gelu | mlx.nn.gelu | `(x) -> mlx.core.array` | Applies the Gaussian Error Linear Units function. |
| [x] | mlx | mlx.nn.losses | cosine_similarity_loss | mlx.nn.losses.cosine_similarity_loss | `(x1: mx.array, x2: mx.array, axis: int | None=1, eps: float | None=1e-08, reduction: str | None='none') -> mlx.core.array` | Computes the cosine similarity between the two inputs. |
| [x] | mlx | mlx.nn.losses | gaussian_nll_loss | mlx.nn.losses.gaussian_nll_loss | `(inputs: array, targets: array, vars: array, full: bool | None=False, eps: float | None=1e-06, reduction: str | None='none') -> mlx.core.array` | Computes the negative log likelihood loss for a Gaussian distribution. |
| [x] | mlx | mlx.nn.losses | hinge_loss | mlx.nn.losses.hinge_loss | `(inputs: array, targets: array, reduction: str | None='none') -> mlx.core.array` | Computes the hinge loss between inputs and targets. |
| [x] | mlx | mlx.nn.losses | huber_loss | mlx.nn.losses.huber_loss | `(inputs: array, targets: array, delta: float | None=1.0, reduction: str | None='none') -> mlx.core.array` | Computes the Huber loss between inputs and targets. |
| [x] | mlx | mlx.nn.losses | kl_div_loss | mlx.nn.losses.kl_div_loss | `(inputs: array, targets: array, axis: int | None=-1.0, reduction: str | None='none') -> mlx.core.array` | Computes the Kullback-Leibler divergence loss. |
| [x] | mlx | mlx.nn.losses | l1_loss | mlx.nn.losses.l1_loss | `(predictions: array, targets: array, reduction: str | None='mean') -> mlx.core.array` | Computes the L1 loss. |
| [x] | mlx | mlx.nn.losses | log_cosh_loss | mlx.nn.losses.log_cosh_loss | `(inputs: array, targets: array, reduction: str | None='none') -> mlx.core.array` | Computes the log cosh loss between inputs and targets. |
| [x] | mlx | mlx.nn.losses | margin_ranking_loss | mlx.nn.losses.margin_ranking_loss | `(inputs1: array, inputs2: array, targets: array, margin: float | None=0.0, reduction: str | None='none') -> mlx.core.array` | Calculate the margin ranking loss that loss given inputs :math:`x_1`, :math:`x_2` and a label :ma... |
| [x] | mlx | mlx.nn.losses | mse_loss | mlx.nn.losses.mse_loss | `(predictions: array, targets: array, reduction: str | None='mean') -> mlx.core.array` | Computes the mean squared error loss. |
| [x] | mlx | mlx.nn.losses | nll_loss | mlx.nn.losses.nll_loss | `(inputs: array, targets: array, axis: int | None=-1.0, reduction: str | None='none') -> mlx.core.array` | Computes the negative log likelihood loss. |
| [x] | mlx | mlx.nn.losses | smooth_l1_loss | mlx.nn.losses.smooth_l1_loss | `(predictions: array, targets: array, beta: float | None=1.0, reduction: str | None='mean') -> mlx.core.array` | Computes the smooth L1 loss. |
| [x] | mlx | mlx.nn.losses | triplet_loss | mlx.nn.losses.triplet_loss | `(anchors: array, positives: array, negatives: array, axis: int | None=-1.0, p: int | None=2, margin: float | None=1.0, eps: float | None=1e-06, reduction: str | None='none') -> mlx.core.array` | Computes the triplet loss for a set of anchor, positive, and negative samples. Margin is represen... |
| [x] | mlx | mlx.nn | relu | mlx.nn.relu | `(x)` | Applies the Rectified Linear Unit. |
| [x] | mlx | mlx.nn | sigmoid | mlx.nn.sigmoid | `(x)` | Applies the sigmoid function. |
| [x] | mlx | mlx.nn | silu | mlx.nn.silu | `(x)` | Applies the Sigmoid Linear Unit. Also known as Swish. |
| [x] | mlx | mlx.nn | softmax | mlx.nn.softmax | `(x, axis: int=-1)` | Applies the Softmax function. |
| [x] | mlx | mlx.nn | tanh | mlx.nn.tanh | `(x)` | Applies the hyperbolic tangent function. |
| [x] | mlx | mlx.optimizers | AdaDelta | mlx.optimizers.AdaDelta | `(learning_rate: float | callable, rho: float | None=0.9, eps: float | None=1e-08)` | The AdaDelta optimizer with a learning rate [1]. |
| [x] | mlx | mlx.optimizers | Adafactor | mlx.optimizers.Adafactor | `(learning_rate: float | callable | optional='```(None)```', eps: tuple(float, float) | None='(1e-30, 1e-3)``.', clip_threshold: float | None=1.0, decay_rate: float | None=-0.8, beta_1: float | None='```(None)```', weight_decay: float | None=0.0, scale_parameter: bool | None=True, relative_step: bool | None=True, warmup_init: bool | None='\n            ``False')` | The Adafactor optimizer. |
| [x] | mlx | mlx.optimizers | Adagrad | mlx.optimizers.Adagrad | `(learning_rate: float | callable, eps: float | None=1e-08)` | The Adagrad optimizer [1]. |
| [x] | mlx | mlx.optimizers | Adam | mlx.optimizers.Adam | `(learning_rate: float | callable, betas: tuple[float, float] | None='(0.9, 0.999)', eps: float | None=1e-08, bias_correction: bool | None=False)` | The Adam optimizer [1]. In detail, |
| [x] | mlx | mlx.optimizers | AdamW | mlx.optimizers.AdamW | `(learning_rate: float | callable, betas: tuple[float, float] | None='(0.9, 0.999)', eps: float | None=1e-08, weight_decay: float | None=0.01, bias_correction: bool | None=False)` | The AdamW optimizer [1]. We update the weights with a weight_decay (:math:`\lambda`) value: |
| [x] | mlx | mlx.optimizers | Adamax | mlx.optimizers.Adamax | `(learning_rate: float | callable, betas: tuple[float, float] | None='(0.9, 0.999)', eps: float | None=1e-08)` | The Adamax optimizer, a variant of Adam based on the infinity norm [1]. |
| [x] | mlx | mlx.optimizers | Lion | mlx.optimizers.Lion | `(learning_rate: float | callable, betas: tuple[float, float] | None='(0.9, 0.99)', weight_decay: float | None=0.0)` | The Lion optimizer [1]. |
| [x] | mlx | mlx.optimizers | Module | mlx.optimizers.Module | `(__call__: Callable)` | Base class for building neural networks with MLX. |
| [x] | mlx | mlx.optimizers | MultiOptimizer | mlx.optimizers.MultiOptimizer | `(optimizers: list[Optimizer], filters: list[Callable[[str, array], bool]=[])` | Wraps a list of optimizers with corresponding weight predicates/filters to make it easy to use di... |
| [x] | mlx | mlx.optimizers | Muon | mlx.optimizers.Muon | `(learning_rate: float | callable, momentum: float | None=0.95, weight_decay: float | None=0.01, nesterov: bool | None=True, ns_steps: int | None=5)` | The Muon optimizer. |
| [x] | mlx | mlx.optimizers | Optimizer | mlx.optimizers.Optimizer | `(schedulers='```(None)```')` | The base class for all optimizers. It allows us to implement an optimizer on a per-parameter basi... |
| [x] | mlx | mlx.optimizers | RMSprop | mlx.optimizers.RMSprop | `(learning_rate: float | callable, alpha: float | None=0.99, eps: float | None=1e-08)` | The RMSprop optimizer [1]. |
| [x] | mlx | mlx.optimizers | SGD | mlx.optimizers.SGD | `(learning_rate: float | callable, momentum: float | None=0, weight_decay: float | None=0, dampening: float | None=0, nesterov: bool | None=False)` | The stochastic gradient descent optimizer. |
