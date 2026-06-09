import numpy as np
import pytest

from zero_mlx import array
from zero_mlx import nn


def test_activations():
    x = array(np.array([-1.0, 0.0, 1.0, 2.0]))

    # relu
    assert np.allclose(nn.relu(x).data, [0.0, 0.0, 1.0, 2.0])
    assert np.allclose(nn.ReLU()(x).data, [0.0, 0.0, 1.0, 2.0])

    # sigmoid
    sig_out = 1 / (1 + np.exp(-x.data))
    assert np.allclose(nn.sigmoid(x).data, sig_out)
    assert np.allclose(nn.Sigmoid()(x).data, sig_out)

    # silu
    assert np.allclose(nn.silu(x).data, x.data * sig_out)
    assert np.allclose(nn.SiLU()(x).data, x.data * sig_out)

    # softmax
    e_x = np.exp(x.data - np.max(x.data, axis=-1, keepdims=True))
    sm_out = e_x / e_x.sum(axis=-1, keepdims=True)
    assert np.allclose(nn.softmax(x).data, sm_out)
    assert np.allclose(nn.Softmax()(x).data, sm_out)

    # tanh
    assert np.allclose(nn.tanh(x).data, np.tanh(x.data))
    assert np.allclose(nn.Tanh()(x).data, np.tanh(x.data))

    # elu
    elu_out = np.where(x.data > 0, x.data, 1.0 * (np.exp(x.data) - 1))
    assert np.allclose(nn.elu(x).data, elu_out)
    assert np.allclose(nn.ELU()(x).data, elu_out)

    # gelu
    import math

    erf_vec = np.vectorize(math.erf)
    gelu_out = x.data * 0.5 * (1.0 + erf_vec(x.data / np.sqrt(2.0)))
    assert np.allclose(nn.gelu(x).data, gelu_out)
    assert np.allclose(nn.GELU()(x).data, gelu_out)
    assert np.allclose(nn.GELU(approx="precise")(x).data, gelu_out)
    fast_inner = np.sqrt(2.0 / np.pi) * (x.data + 0.044715 * np.power(x.data, 3))
    gelu_fast = 0.5 * x.data * (1.0 + np.tanh(fast_inner))
    assert np.allclose(nn.GELU(approx="fast")(x).data, gelu_fast)
    assert np.allclose(nn.GELU(approx="invalid")(x).data, gelu_out)

    # CELU
    celu_out = np.maximum(0, x.data) + np.minimum(0, 1.0 * (np.exp(x.data / 1.0) - 1))
    assert np.allclose(nn.CELU()(x).data, celu_out)

    # GLU
    x_glu = array(np.array([[1.0, 2.0], [3.0, 4.0]]))
    a, b = np.split(x_glu.data, 2, axis=-1)
    glu_out = a * (1 / (1 + np.exp(-b)))
    assert np.allclose(nn.GLU()(x_glu).data, glu_out)

    # LogSigmoid
    assert np.allclose(nn.LogSigmoid()(x).data, -np.log(1 + np.exp(-x.data)))

    # LogSoftmax
    assert np.allclose(nn.LogSoftmax()(x).data, np.log(sm_out))

    # Mish
    mish_out = x.data * np.tanh(np.log(1 + np.exp(x.data)))
    assert np.allclose(nn.Mish()(x).data, mish_out)

    # PReLU
    prelu = nn.PReLU()
    prelu_out = np.maximum(0, x.data) + prelu.weight * np.minimum(0, x.data)
    assert np.allclose(prelu(x).data, prelu_out)

    # ReLU2
    assert np.allclose(nn.ReLU2()(x).data, np.maximum(0, x.data) ** 2)

    # ReLU6
    assert np.allclose(nn.ReLU6()(x).data, np.minimum(np.maximum(0, x.data), 6))

    # HardShrink
    assert np.allclose(
        nn.HardShrink(lambd=0.5)(x).data,
        np.where((x.data > 0.5) | (x.data < -0.5), x.data, 0.0),
    )

    # HardTanh
    assert np.allclose(nn.HardTanh()(x).data, np.clip(x.data, -1.0, 1.0))

    # Hardswish
    inner = np.clip(x.data + 3.0, 0.0, 6.0)
    assert np.allclose(nn.Hardswish()(x).data, x.data * inner / 6.0)

    # LeakyReLU
    assert np.allclose(
        nn.LeakyReLU(negative_slope=0.1)(x).data,
        np.maximum(0, x.data) + 0.1 * np.minimum(0, x.data),
    )

    # Softshrink
    expected_softshrink = np.where(
        x.data > 0.5, x.data - 0.5, np.where(x.data < -0.5, x.data + 0.5, 0.0)
    )
    assert np.allclose(nn.Softshrink(lambd=0.5)(x).data, expected_softshrink)

    # SELU
    alpha = 1.6732632423543772848170429916717
    scale = 1.0507009873554804934193349852946
    selu_out = scale * (
        np.maximum(0, x.data) + np.minimum(0, alpha * (np.exp(x.data) - 1))
    )
    assert np.allclose(nn.SELU()(x).data, selu_out)

    # Softmin
    neg_x = -x.data
    e_neg_x = np.exp(neg_x - np.max(neg_x, axis=-1, keepdims=True))
    softmin_out = e_neg_x / e_neg_x.sum(axis=-1, keepdims=True)
    assert np.allclose(nn.Softmin()(x).data, softmin_out)

    # Softplus
    assert np.allclose(nn.Softplus()(x).data, np.log(1 + np.exp(x.data)))

    # Softsign
    assert np.allclose(nn.Softsign()(x).data, x.data / (1 + np.abs(x.data)))

    # Step
    assert np.allclose(
        nn.Step(threshold=1.0)(x).data, np.where(x.data >= 1.0, 1.0, 0.0)
    )


def test_layers():
    x = array(np.array([1.0, 2.0]))

    # Module
    mod = nn.Module()
    with pytest.raises(NotImplementedError):
        mod(x)

    # Identity
    ident = nn.Identity()
    assert np.allclose(ident(x).data, x.data)

    # Sequential
    seq = nn.Sequential(nn.ReLU(), nn.Step(1.5))
    out = seq(x)
    assert np.allclose(out.data, [0.0, 1.0])

    # Dropout
    x_drop = array(np.ones(10))
    drop = nn.Dropout(p=0.5)
    out_drop = drop(x_drop)
    assert out_drop.data.shape == (10,)
    drop_1 = nn.Dropout(p=1.0)
    assert np.allclose(drop_1(x_drop).data, np.zeros(10))

    # Dropout2d
    drop2d = nn.Dropout2d(p=0.5)
    assert np.allclose(drop2d(x).data, x.data)

    # Dropout3d
    drop3d = nn.Dropout3d(p=0.5)
    assert np.allclose(drop3d(x).data, x.data)

    # Linear
    linear = nn.Linear(2, 3, bias=True)
    out_lin = linear(array(np.array([[1.0, 2.0]])))
    assert out_lin.data.shape == (1, 3)
    linear_nobias = nn.Linear(2, 3, bias=False)
    out_lin_nobias = linear_nobias(array(np.array([[1.0, 2.0]])))
    assert out_lin_nobias.data.shape == (1, 3)

    # Bilinear
    bilinear = nn.Bilinear(2, 2, 3, bias=True)
    x1 = array(np.array([[1.0, 2.0]]))
    x2 = array(np.array([[3.0, 4.0]]))
    out_bilin = bilinear(x1, x2)
    assert out_bilin.data.shape == (1, 3)
    bilinear_nobias = nn.Bilinear(2, 2, 3, bias=False)
    assert bilinear_nobias(x1, x2).data.shape == (1, 3)

    # Embedding
    emb = nn.Embedding(10, 5)
    indices = array(np.array([1, 4]))
    out_emb = emb(indices)
    assert out_emb.data.shape == (2, 5)

    # Pooling
    x_pool1d = np.random.randn(2, 4, 3)  # (N, L, C)
    avg1d = nn.AvgPool1d(kernel_size=2, stride=2, padding=1)
    out_avg1d = avg1d(array(x_pool1d))
    assert out_avg1d.data.shape == (2, 3, 3)

    x_pool2d = np.random.randn(2, 4, 4, 3)  # (N, H, W, C)
    avg2d = nn.AvgPool2d(kernel_size=2, stride=2, padding=1)
    out_avg2d = avg2d(array(x_pool2d))
    assert out_avg2d.data.shape == (2, 3, 3, 3)

    x_pool3d = np.random.randn(2, 4, 4, 4, 3)  # (N, D, H, W, C)
    avg3d = nn.AvgPool3d(kernel_size=2, stride=2, padding=1)
    out_avg3d = avg3d(array(x_pool3d))
    assert out_avg3d.data.shape == (2, 3, 3, 3, 3)

    max1d = nn.MaxPool1d(kernel_size=2, stride=2, padding=1)
    out_max1d = max1d(array(x_pool1d))
    assert out_max1d.data.shape == (2, 3, 3)

    max2d = nn.MaxPool2d(kernel_size=2, stride=2, padding=1)
    out_max2d = max2d(array(x_pool2d))
    assert out_max2d.data.shape == (2, 3, 3, 3)

    max3d = nn.MaxPool3d(kernel_size=2, stride=2, padding=1)
    out_max3d = max3d(array(x_pool3d))
    assert out_max3d.data.shape == (2, 3, 3, 3, 3)

    # Normalization
    x_norm = np.random.randn(2, 4, 4, 4)  # (N, H, W, C)
    eps = 1e-5

    # BatchNorm
    bn = nn.BatchNorm(4)
    bn_out = bn(array(x_norm))
    reduce_axes = (0, 1, 2)
    bn_mean = np.mean(x_norm, axis=reduce_axes, keepdims=True)
    bn_var = np.var(x_norm, axis=reduce_axes, keepdims=True)
    bn_expected = (x_norm - bn_mean) / np.sqrt(bn_var + eps)
    assert np.allclose(bn_out.data, bn_expected)
    assert np.allclose(bn.running_mean, 0.1 * np.squeeze(bn_mean))

    bn_notrack = nn.BatchNorm(4, track_running_stats=False)
    assert np.allclose(bn_notrack(array(x_norm)).data, bn_expected)

    # GroupNorm
    gn = nn.GroupNorm(num_groups=2, dims=4)
    gn_out = gn(array(x_norm))
    reshaped_gn = x_norm.reshape(2, 4, 4, 2, 2)
    gn_mean = np.mean(reshaped_gn, axis=(1, 2, 4), keepdims=True)
    gn_var = np.var(reshaped_gn, axis=(1, 2, 4), keepdims=True)
    gn_expected = ((reshaped_gn - gn_mean) / np.sqrt(gn_var + eps)).reshape(
        x_norm.shape
    )
    assert np.allclose(gn_out.data, gn_expected)

    # InstanceNorm
    inn = nn.InstanceNorm(4)
    inn_out = inn(array(x_norm))
    in_mean = np.mean(x_norm, axis=(1, 2), keepdims=True)
    in_var = np.var(x_norm, axis=(1, 2), keepdims=True)
    in_expected = (x_norm - in_mean) / np.sqrt(in_var + eps)
    assert np.allclose(inn_out.data, in_expected)
    inn_affine = nn.InstanceNorm(4, affine=True)
    assert np.allclose(inn_affine(array(x_norm)).data, in_expected)

    # LayerNorm
    ln = nn.LayerNorm(4)
    ln_out = ln(array(x_norm))
    ln_mean = np.mean(x_norm, axis=(-1,), keepdims=True)
    ln_var = np.var(x_norm, axis=(-1,), keepdims=True)
    ln_expected = (x_norm - ln_mean) / np.sqrt(ln_var + eps)
    assert np.allclose(ln_out.data, ln_expected)
    ln_nobias = nn.LayerNorm(4, bias=False)
    assert np.allclose(ln_nobias(array(x_norm)).data, ln_expected)

    # RMSNorm
    rmsn = nn.RMSNorm(4)
    rmsn_out = rmsn(array(x_norm))
    rms_val = np.sqrt(np.mean(np.square(x_norm), axis=(-1,), keepdims=True) + eps)
    rmsn_expected = x_norm / rms_val
    assert np.allclose(rmsn_out.data, rmsn_expected)

    # Positional Encoding
    x_pe = np.random.randn(2, 4, 4)  # (N, L, D)

    rope = nn.RoPE(4)
    out_rope = rope(array(x_pe))
    assert out_rope.data.shape == (2, 4, 4)

    rope_trad = nn.RoPE(4, traditional=True)
    assert rope_trad(array(x_pe)).data.shape == (2, 4, 4)

    alibi = nn.ALiBi()
    x_attn = np.random.randn(2, 4, 4, 4)  # (B, H, q, k)
    out_alibi = alibi(array(x_attn))
    assert out_alibi.data.shape == (2, 4, 4, 4)
    assert alibi(array(x_pe)).data.shape == (2, 4, 4)  # Fallback branch

    # Also test ALiBi with non-power of two heads
    alibi_non_pow = nn.ALiBi()
    x_attn_np = np.random.randn(2, 3, 4, 4)  # (B, 3, q, k)
    assert alibi_non_pow(array(x_attn_np)).data.shape == (2, 3, 4, 4)

    sinpe = nn.SinusoidalPositionalEncoding(4)
    out_sinpe = sinpe(array(x_pe))
    assert out_sinpe.data.shape == (2, 4, 4)

    sinpe_cos = nn.SinusoidalPositionalEncoding(4, cos_first=True, full_turns=True)
    assert sinpe_cos(array(x_pe)).data.shape == (2, 4, 4)

    # Convolutions
    conv1d = nn.Conv1d(2, 3, kernel_size=2, padding=1)
    x_c1 = np.random.randn(2, 4, 2)
    out_c1 = conv1d(array(x_c1))
    assert out_c1.data.shape == (2, 5, 3)

    conv1d_nobias = nn.Conv1d(2, 3, kernel_size=2, bias=False)
    assert conv1d_nobias(array(x_c1)).data.shape == (2, 3, 3)

    conv2d = nn.Conv2d(2, 3, kernel_size=(2, 2), padding=1)
    x_c2 = np.random.randn(2, 4, 4, 2)
    out_c2 = conv2d(array(x_c2))
    assert out_c2.data.shape == (2, 5, 5, 3)

    conv2d_nobias = nn.Conv2d(2, 3, kernel_size=(2, 2), bias=False)
    assert conv2d_nobias(array(x_c2)).data.shape == (2, 3, 3, 3)

    conv3d = nn.Conv3d(2, 3, kernel_size=(2, 2, 2), padding=1)
    x_c3 = np.random.randn(2, 4, 4, 4, 2)
    out_c3 = conv3d(array(x_c3))
    assert out_c3.data.shape == (2, 5, 5, 5, 3)

    conv3d_nobias = nn.Conv3d(2, 3, kernel_size=(2, 2, 2), bias=False)
    assert conv3d_nobias(array(x_c3)).data.shape == (2, 3, 3, 3, 3)

    conv_t1 = nn.ConvTranspose1d(2, 3, 2)
    x_t1 = np.random.randn(2, 4, 2)
    out_t1 = conv_t1(array(x_t1))
    assert out_t1.data.shape == (2, 5, 3)

    conv_t1_nobias = nn.ConvTranspose1d(2, 3, 2, bias=False)
    assert conv_t1_nobias(array(x_t1)).data.shape == (2, 5, 3)

    conv_t2 = nn.ConvTranspose2d(2, 3, (2, 2))
    x_t2 = np.random.randn(2, 4, 4, 2)
    out_t2 = conv_t2(array(x_t2))
    assert out_t2.data.shape == (2, 5, 5, 3)

    conv_t2_nobias = nn.ConvTranspose2d(2, 3, (2, 2), bias=False)
    assert conv_t2_nobias(array(x_t2)).data.shape == (2, 5, 5, 3)

    conv_t3 = nn.ConvTranspose3d(2, 3, (2, 2, 2))
    x_t3 = np.random.randn(2, 4, 4, 4, 2)
    out_t3 = conv_t3(array(x_t3))
    assert out_t3.data.shape == (2, 5, 5, 5, 3)

    conv_t3_nobias = nn.ConvTranspose3d(2, 3, (2, 2, 2), bias=False)
    assert conv_t3_nobias(array(x_t3)).data.shape == (2, 5, 5, 5, 3)

    # Upsample
    x_up2 = np.random.randn(2, 4, 4, 3)
    up2 = nn.Upsample(scale_factor=2.0)
    out_up2 = up2(array(x_up2))
    assert out_up2.data.shape == (2, 8, 8, 3)
    up2_non_nearest = nn.Upsample(scale_factor=2.0, mode="linear")
    assert up2_non_nearest(array(x_up2)).data.shape == (2, 8, 8, 3)

    # Recurrent
    x_rnn = np.random.randn(2, 5, 4)  # (N, L, I)

    rnn = nn.RNN(4, 3)
    out_rnn, h_rnn = rnn(array(x_rnn))
    assert out_rnn.data.shape == (2, 5, 3)
    assert h_rnn.data.shape == (2, 3)

    rnn_nobias = nn.RNN(4, 3, bias=False)
    assert rnn_nobias(array(x_rnn))[0].data.shape == (2, 5, 3)

    # 2D case
    x_rnn_2d = np.random.randn(5, 4)
    out_rnn_2d, h_rnn_2d = rnn(array(x_rnn_2d))
    assert out_rnn_2d.data.shape == (5, 3)
    assert h_rnn_2d.data.shape == (3,)

    gru = nn.GRU(4, 3)
    out_gru, h_gru = gru(array(x_rnn))
    assert out_gru.data.shape == (2, 5, 3)
    assert h_gru.data.shape == (2, 3)

    gru_nobias = nn.GRU(4, 3, bias=False)
    assert gru_nobias(array(x_rnn))[0].data.shape == (2, 5, 3)

    out_gru_2d, h_gru_2d = gru(array(x_rnn_2d))
    assert out_gru_2d.data.shape == (5, 3)

    lstm = nn.LSTM(4, 3)
    out_lstm, (h_lstm, c_lstm) = lstm(array(x_rnn))
    assert out_lstm.data.shape == (2, 5, 3)
    assert h_lstm.data.shape == (2, 3)
    assert c_lstm.data.shape == (2, 3)

    lstm_nobias = nn.LSTM(4, 3, bias=False)
    assert lstm_nobias(array(x_rnn))[0].data.shape == (2, 5, 3)

    out_lstm_2d, (h_lstm_2d, c_lstm_2d) = lstm(array(x_rnn_2d))
    assert out_lstm_2d.data.shape == (5, 3)

    # With initial hidden state
    h0 = array(np.random.randn(2, 3))
    c0 = array(np.random.randn(2, 3))
    assert rnn(array(x_rnn), h0=h0)[0].data.shape == (2, 5, 3)
    assert gru(array(x_rnn), h0=h0)[0].data.shape == (2, 5, 3)
    assert lstm(array(x_rnn), hx=(h0, c0))[0].data.shape == (2, 5, 3)

    # 1D single batch
    x_rnn_1d = np.random.randn(5, 4)
    h0_1d = array(np.random.randn(3))
    c0_1d = array(np.random.randn(3))
    out_1d_rnn, h_1d_rnn = rnn(array(x_rnn_1d), h0=h0_1d)
    out_1d_gru, h_1d_gru = gru(array(x_rnn_1d), h0=h0_1d)
    out_1d_lstm, (h_1d_lstm, _) = lstm(array(x_rnn_1d), hx=(h0_1d, c0_1d))
    assert h_1d_rnn.data.shape == (3,)
    assert h_1d_gru.data.shape == (3,)
    assert h_1d_lstm.data.shape == (3,)

    # Transformers
    x_enc = array(np.random.randn(2, 5, 8))  # (N, L, D)
    x_dec = array(np.random.randn(2, 3, 8))

    mha = nn.MultiHeadAttention(8, 2)
    out_mha = mha(x_enc, x_enc, x_enc)
    assert out_mha.data.shape == (2, 5, 8)

    mask2d = array(np.random.randn(5, 5))
    mask3d = array(np.random.randn(2, 5, 5))
    out_mha_m2 = mha(x_enc, x_enc, x_enc, mask=mask2d)
    out_mha_m3 = mha(x_enc, x_enc, x_enc, mask=mask3d)
    assert out_mha_m2.data.shape == (2, 5, 8)
    assert out_mha_m3.data.shape == (2, 5, 8)

    x_enc_2d = array(np.random.randn(5, 8))
    assert mha(x_enc_2d, x_enc_2d, x_enc_2d).data.shape == (5, 8)

    enc_layer = nn.TransformerEncoderLayer(8, 2)
    out_enc_l = enc_layer(x_enc)
    assert out_enc_l.data.shape == (2, 5, 8)

    enc_layer_post = nn.TransformerEncoderLayer(8, 2, norm_first=False)
    assert enc_layer_post(x_enc).data.shape == (2, 5, 8)

    encoder = nn.TransformerEncoder(2, 8, 2)
    out_enc = encoder(x_enc)
    assert out_enc.data.shape == (2, 5, 8)

    dec_layer = nn.TransformerDecoderLayer(8, 2)
    out_dec_l = dec_layer(x_dec, x_enc)
    assert out_dec_l.data.shape == (2, 3, 8)

    dec_layer_post = nn.TransformerDecoderLayer(8, 2, norm_first=False)
    assert dec_layer_post(x_dec, x_enc).data.shape == (2, 3, 8)

    decoder = nn.TransformerDecoder(2, 8, 2)
    out_dec = decoder(x_dec, x_enc)
    assert out_dec.data.shape == (2, 3, 8)

    decoder_post = nn.TransformerDecoder(2, 8, 2, norm_first=False)
    assert decoder_post(x_dec, x_enc).data.shape == (2, 3, 8)

    encoder_post = nn.TransformerEncoder(2, 8, 2, norm_first=False)
    assert encoder_post(x_enc).data.shape == (2, 5, 8)

    transformer = nn.Transformer(
        dims=8, num_heads=2, num_encoder_layers=2, num_decoder_layers=2
    )
    out_trans = transformer(x_enc, x_dec)
    assert out_trans.data.shape == (2, 3, 8)

    # Distributed / Quantized
    x_dist = np.random.randn(2, 4, 4)
    qlin = nn.QuantizedLinear(4, 3)
    out_qlin = qlin(array(x_dist))
    assert out_qlin.data.shape == (2, 4, 3)

    qemb = nn.QuantizedEmbedding(10, 4)
    out_qemb = qemb(array(np.array([1, 2])))
    assert out_qemb.data.shape == (2, 4)

    q_all2sh = nn.QuantizedAllToShardedLinear(4, 3)
    out_q_all2sh = q_all2sh(array(x_dist))
    assert out_q_all2sh.data.shape == (2, 4, 3)

    q_sh2all = nn.QuantizedShardedToAllLinear(4, 3)
    out_q_sh2all = q_sh2all(array(x_dist))
    assert out_q_sh2all.data.shape == (2, 4, 3)

    all2sh = nn.AllToShardedLinear(4, 3)
    out_all2sh = all2sh(array(x_dist))
    assert out_all2sh.data.shape == (2, 4, 3)

    sh2all = nn.ShardedToAllLinear(4, 3)
    out_sh2all = sh2all(array(x_dist))
    assert out_sh2all.data.shape == (2, 4, 3)


def test_losses():
    x1 = array(np.array([[1.0, 0.0], [0.0, 1.0]]))
    x2 = array(np.array([[0.5, 0.5], [0.0, 1.0]]))

    # cosine_similarity_loss
    out = nn.losses.cosine_similarity_loss(x1, x2, reduction="none")
    assert np.allclose(out.data, [0.70710678, 1.0])

    # gaussian_nll_loss
    v = array(np.array([[1.0, 1.0], [1.0, 1.0]]))
    out = nn.losses.gaussian_nll_loss(x1, x2, v)
    assert out.data.shape == (2, 2)
    out_full = nn.losses.gaussian_nll_loss(x1, x2, v, full=True)
    assert out_full.data.shape == (2, 2)

    # hinge_loss
    out = nn.losses.hinge_loss(x1, x2)
    assert np.allclose(out.data, [[0.5, 1.0], [1.0, 0.0]])

    # huber_loss
    out = nn.losses.huber_loss(x1, x2)
    assert np.allclose(out.data, [[0.125, 0.125], [0.0, 0.0]])

    # kl_div_loss
    x1_kl = array(np.array([[-1.0, -2.0], [-3.0, -4.0]]))
    x2_kl = array(np.array([[0.5, 0.5], [0.1, 0.9]]))
    out = nn.losses.kl_div_loss(x1_kl, x2_kl)
    assert out.data.shape == (2, 2)

    # l1_loss
    out = nn.losses.l1_loss(x1, x2, reduction="mean")
    assert np.allclose(out.data, 0.25)

    # log_cosh_loss
    out = nn.losses.log_cosh_loss(x1, x2)
    assert np.allclose(out.data, [[0.1201145, 0.1201145], [0.0, 0.0]])

    # margin_ranking_loss
    y = array(np.array([[1.0, -1.0], [1.0, 1.0]]))
    out = nn.losses.margin_ranking_loss(x1, x2, y)
    assert np.allclose(out.data, [[0.0, 0.0], [0.0, 0.0]])

    # mse_loss
    out = nn.losses.mse_loss(x1, x2, reduction="mean")
    assert np.allclose(out.data, 0.125)

    out = nn.losses.mse_loss(x1, x2, reduction="sum")
    assert np.allclose(out.data, 0.5)

    # nll_loss
    inputs = array(np.array([[-1.0, -2.0], [-3.0, -0.5]]))
    targets = array(np.array([0, 1]))
    out = nn.losses.nll_loss(inputs, targets)
    assert np.allclose(out.data, [1.0, 0.5])

    # smooth_l1_loss
    out = nn.losses.smooth_l1_loss(x1, x2)
    assert np.allclose(out.data, 0.0625)

    # triplet_loss
    anc = array(np.array([[1.0, 0.0]]))
    pos = array(np.array([[0.8, 0.2]]))
    neg = array(np.array([[0.1, 0.9]]))
    out = nn.losses.triplet_loss(anc, pos, neg)

    dist_pos = np.linalg.norm(anc.data - pos.data, axis=-1)
    dist_neg = np.linalg.norm(anc.data - neg.data, axis=-1)
    expected = np.maximum(0, dist_pos - dist_neg + 1.0)
    assert np.allclose(out.data, expected)
