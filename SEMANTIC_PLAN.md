# Semantic Implementation Plan

This document outlines the roadmap for transitioning structurally compliant (stubbed) MLX layers into mathematically accurate implementations. Each layer requires a precise `numpy`-based forward pass and corresponding tests that verify the output against expected mathematical results.

## Normalization Layers
- [x] **BatchNorm**
  - [x] Implement training mode (batch statistics calculation).
  - [x] Implement evaluation mode (running statistics usage).
  - [x] Implement running mean/var updates (momentum).
  - [x] Implement affine transformation (weight/bias scaling).
  - [x] Write tests verifying exact normalization outputs across varying batch sizes.
- [x] **GroupNorm**
  - [x] Implement reshaping to `(N, num_groups, C // num_groups, ...)`.
  - [x] Calculate mean and variance over the spatial and group dimensions.
  - [x] Implement affine transformation.
  - [x] Write tests verifying normalization against expected values for different group sizes.
- [x] **InstanceNorm**
  - [x] Calculate mean and variance per instance and per channel.
  - [x] Implement affine transformation (if enabled).
  - [x] Write tests verifying exact normalization outputs.
- [x] **LayerNorm**
  - [x] Calculate mean and variance over the specified normalized dimensions.
  - [x] Implement affine transformation.
  - [x] Write tests verifying layer-wise normalization.
- [x] **RMSNorm**
  - [x] Calculate Root Mean Square over the specified dimensions.
  - [x] Implement scaling using the `weight` parameter.
  - [x] Write tests comparing RMS scaling against mathematical expectations.

## Pooling Layers
- [x] **AvgPool1d**
  - [x] Implement 1D sliding window average calculation.
  - [x] Handle stride and padding correctly.
  - [x] Write tests for various kernel sizes and strides.
- [x] **AvgPool2d**
  - [x] Implement 2D sliding window average calculation.
  - [x] Handle spatial stride and padding.
  - [x] Write tests for varied input shapes and parameters.
- [x] **AvgPool3d**
  - [x] Implement 3D sliding window average calculation.
  - [x] Handle volumetric stride and padding.
  - [x] Write tests for 3D tensors.
- [x] **MaxPool1d**
  - [x] Implement 1D sliding window max calculation.
  - [x] Handle stride and padding.
  - [x] Write tests verifying max extraction.
- [x] **MaxPool2d**
  - [x] Implement 2D sliding window max calculation.
  - [x] Handle stride and padding.
  - [x] Write tests verifying spatial max extraction.
- [x] **MaxPool3d**
  - [x] Implement 3D sliding window max calculation.
  - [x] Handle stride and padding.
  - [x] Write tests verifying volumetric max extraction.

## Convolutional & Spatial Layers
- [x] **Conv1d**
  - [x] Implement 1D cross-correlation/convolution using `numpy`.
  - [x] Support grouped convolutions.
  - [x] Support stride, padding, and dilation.
  - [x] Write tests verifying outputs against known kernel applications.
- [x] **Conv2d**
  - [x] Implement 2D spatial convolution (e.g., using `im2col` or `np.lib.stride_tricks`).
  - [x] Support groups, stride, padding, and dilation.
  - [x] Write tests for standard computer vision shapes.
- [x] **Conv3d**
  - [x] Implement 3D volumetric convolution.
  - [x] Support groups, stride, padding, and dilation.
  - [x] Write tests for 3D tensors.
- [x] **ConvTranspose1d**
  - [x] Implement 1D transposed convolution (fractionally strided convolution).
  - [x] Support output padding and groups.
  - [x] Write mathematical verification tests.
- [x] **ConvTranspose2d**
  - [x] Implement 2D transposed convolution.
  - [x] Support output padding and groups.
  - [x] Write mathematical verification tests.
- [x] **ConvTranspose3d**
  - [x] Implement 3D transposed convolution.
  - [x] Support output padding and groups.
  - [x] Write mathematical verification tests.
- [x] **Upsample**
  - [x] Implement `nearest` interpolation mode for N-D tensors.
  - [x] Implement `linear` (bilinear/trilinear) interpolation modes.
  - [x] Implement `cubic` interpolation mode.
  - [x] Support `align_corners` flag accurately.
  - [x] Write tests verifying scaling dimensions and interpolated values.

## Recurrent Layers
- [x] **RNN (Elman)**
  - [x] Implement sequential time-step iteration.
  - [x] Implement hidden state updates using `weight_ih`, `weight_hh`, and biases.
  - [x] Apply specified nonlinearity (tanh or relu).
  - [x] Write tests verifying hidden state accumulation over time.
- [x] **GRU (Gated Recurrent Unit)**
  - [x] Implement update gate ($z_t$) calculation.
  - [x] Implement reset gate ($r_t$) calculation.
  - [x] Implement candidate hidden state calculation.
  - [x] Implement final hidden state ($h_t$) calculation.
  - [x] Write tests verifying exact GRU mathematical steps.
- [x] **LSTM (Long Short-Term Memory)**
  - [x] Implement input gate ($i_t$) calculation.
  - [x] Implement forget gate ($f_t$) calculation.
  - [x] Implement cell state ($C_t$) update.
  - [x] Implement output gate ($o_t$) calculation.
  - [x] Implement hidden state ($h_t$) update.
  - [x] Write tests verifying both cell state and hidden state propagation.

## Positional Encodings
- [x] **RoPE (Rotary Positional Encoding)**
  - [x] Generate rotation matrices/frequencies based on `base` and `dims`.
  - [x] Implement application of rotation to queries and keys.
  - [x] Support `traditional` vs interleaved coordinate modes.
  - [x] Write mathematical verification tests.
- [x] **ALiBi (Attention with Linear Biases)**
  - [x] Calculate distance-based slopes based on the number of attention heads.
  - [x] Implement mask generation and addition to attention scores.
  - [x] Write tests verifying penalty slopes.
- [x] **SinusoidalPositionalEncoding**
  - [x] Generate sine and cosine embeddings based on frequencies.
  - [x] Support `cos_first` and `full_turns` flags.
  - [x] Write tests verifying exact waveform values.

## Transformer Layers
- [x] **MultiHeadAttention**
  - [x] Implement linear projections for queries, keys, and values.
  - [x] Implement splitting into multiple heads.
  - [x] Implement scaled dot-product attention calculation ($Softmax(QK^T/\sqrt{d})V$).
  - [x] Support attention masking (causal, padding).
  - [x] Implement output concatenation and linear projection.
  - [x] Write tests verifying exact attention matrix outputs.
- [x] **TransformerEncoderLayer**
  - [x] Assemble `MultiHeadAttention`, `LayerNorm`, and MLP (Linear layers + Activation).
  - [x] Implement residual connections correctly.
  - [x] Support `norm_first` (Pre-LN vs Post-LN).
  - [x] Write tests verifying end-to-end layer forward pass.
- [x] **TransformerDecoderLayer**
  - [x] Implement self-attention, cross-attention (over memory), and MLP.
  - [x] Implement residual connections and `LayerNorms`.
  - [x] Support distinct `mask` and `memory_mask`.
  - [x] Write tests verifying decoding logic.
- [x] **TransformerEncoder**
  - [x] Stack multiple `TransformerEncoderLayer` modules.
  - [x] Implement sequential forward pass.
  - [x] Write integration tests.
- [x] **TransformerDecoder**
  - [x] Stack multiple `TransformerDecoderLayer` modules.
  - [x] Implement sequential forward pass with memory integration.
  - [x] Write integration tests.
- [x] **Transformer (Full Model)**
  - [x] Combine Encoder and Decoder.
  - [x] Wire source/target inputs and masks correctly.
  - [x] Write end-to-end seq2seq integration tests.

## Distributed & Quantized Layers
*Note: Due to `numpy` constraints, these may be simulated semantically rather than providing actual hardware-level quantization/sharding.*
- [x] **QuantizedLinear / QuantizedEmbedding**
  - [x] Implement mock or exact group-wise scaling and de-quantization logic during the forward pass to simulate the mathematical impact of `group_size` and `bits`.
- [x] **Sharded/Distributed Linear**
  - [x] Implement mathematical simulation of tensor splitting, distributed calculation, and reduction/gathering (e.g., simulating what `AllToSharded` or `ShardedToAll` does to the tensor shape and values mathematically).
