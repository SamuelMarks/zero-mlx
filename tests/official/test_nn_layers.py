import unittest
import zero_mlx as mx


class TestNNLayers(unittest.TestCase):
    def test_pooling_stubs(self):
        with self.assertRaises(NotImplementedError):
            mx.nn.AvgPool1d(2)
        with self.assertRaises(NotImplementedError):
            mx.nn.AvgPool2d(2)
        with self.assertRaises(NotImplementedError):
            mx.nn.AvgPool3d(2)
        with self.assertRaises(NotImplementedError):
            mx.nn.MaxPool1d(2)
        with self.assertRaises(NotImplementedError):
            mx.nn.MaxPool2d(2)
        with self.assertRaises(NotImplementedError):
            mx.nn.MaxPool3d(2)

    def test_norm_stubs(self):
        with self.assertRaises(NotImplementedError):
            mx.nn.BatchNorm(2)
        with self.assertRaises(NotImplementedError):
            mx.nn.LayerNorm(2)
        with self.assertRaises(NotImplementedError):
            mx.nn.GroupNorm(1, 2)
        with self.assertRaises(NotImplementedError):
            mx.nn.InstanceNorm(2)
        with self.assertRaises(NotImplementedError):
            mx.nn.RMSNorm(2)

    def test_activation_stubs(self):
        act_classes = [
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
        for name in act_classes:
            cls = getattr(mx.nn, name)
            with self.assertRaises(NotImplementedError):
                cls()

    def test_functional_stubs(self):
        act_funcs = [
            "celu",
            "elu",
            "gelu",
            "gelu_approx",
            "gelu_fast_approx",
            "glu",
            "hard_shrink",
            "hard_tanh",
            "hardswish",
            "leaky_relu",
            "log_sigmoid",
            "log_softmax",
            "mish",
            "relu",
            "relu2",
            "relu6",
            "selu",
            "silu",
            "sigmoid",
            "softmax",
            "softmin",
            "softplus",
            "softshrink",
            "softsign",
            "step",
            "tanh",
        ]
        for name in act_funcs:
            func = getattr(mx.nn, name)
            with self.assertRaises(NotImplementedError):
                func(mx.array(1))

        with self.assertRaises(NotImplementedError):
            mx.nn.prelu(mx.array(1), mx.array(1))

        with self.assertRaises(NotImplementedError):
            mx.nn.average_gradients(None)

    def test_more_namespaces(self):
        self.assertTrue(hasattr(mx.nn, "activations"))
        self.assertTrue(hasattr(mx.nn, "base"))
        self.assertTrue(hasattr(mx.nn, "dropout"))
        self.assertTrue(hasattr(mx.nn, "embedding"))
        self.assertTrue(hasattr(mx.nn, "init"))
        self.assertTrue(hasattr(mx.nn, "layers"))
        self.assertTrue(hasattr(mx.nn, "linear"))
        self.assertTrue(hasattr(mx.nn, "losses"))
        self.assertTrue(hasattr(mx.nn, "normalization"))
        self.assertTrue(hasattr(mx.nn, "pooling"))
        self.assertTrue(hasattr(mx.nn, "positional_encoding"))
        self.assertTrue(hasattr(mx.nn, "quantized"))
        self.assertTrue(hasattr(mx.nn, "recurrent"))
        self.assertTrue(hasattr(mx.nn, "transformer"))
        self.assertTrue(hasattr(mx.nn, "upsample"))
        self.assertTrue(hasattr(mx.nn, "utils"))

    def test_gelu_approx(self):
        with self.assertRaises(NotImplementedError):
            mx.nn.gelu_approx(1)
        with self.assertRaises(NotImplementedError):
            mx.nn.gelu_fast_approx(1)

    def test_missing_files_coverage(self):
        import zero_mlx.nn.gelu_approx as ga
        import zero_mlx.nn.gelu_fast_approx as gfa
        import zero_mlx.nn.average_gradients as ag

        with self.assertRaises(NotImplementedError):
            ga.gelu_approx(1)
        with self.assertRaises(NotImplementedError):
            gfa.gelu_fast_approx(1)
        with self.assertRaises(NotImplementedError):
            ag.average_gradients(None)


if __name__ == "__main__":
    unittest.main()

    def test_missing_nn_classes(self):
        classes = [
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
            "QuantizedEmbedding",
            "QuantizedLinear",
            "RNN",
            "RoPE",
            "Sequential",
            "SinusoidalPositionalEncoding",
            "Transformer",
            "TransformerDecoder",
            "TransformerDecoderLayer",
            "TransformerEncoder",
            "TransformerEncoderLayer",
            "Upsample",
            "Bilinear",
        ]
        for name in classes:
            cls = getattr(mx.nn, name)
            with self.assertRaises(NotImplementedError):
                if name == "Sequential":
                    cls()
                elif "Linear" in name and "Quantized" not in name:
                    cls(1, 1)
                elif "Embedding" in name:
                    cls(1, 1)
                elif "QuantizedLinear" in name:
                    cls(1, 1)
                elif name in ["GRU", "LSTM", "RNN"]:
                    cls(1, 1)
                elif name == "MultiHeadAttention":
                    cls(1, 1)
                elif name == "RoPE":
                    cls(1)
                elif name == "SinusoidalPositionalEncoding":
                    cls(1)
                elif "Transformer" in name and "Layer" in name:
                    cls(1, 1)
                elif "Transformer" in name:
                    if name == "Transformer":
                        cls()
                    else:
                        cls(1, 1, 1)
                elif name == "Upsample":
                    cls(1)
                elif name == "Bilinear":
                    cls(1, 1, 1)
                elif name == "Identity":
                    cls()
                elif "Conv" in name:
                    if "1d" in name:
                        cls(1, 1, 1)
                    elif "2d" in name:
                        cls(1, 1, 1)
                    else:
                        cls(1, 1, 1)
                else:
                    cls()

        # Sharded tests
        with self.assertRaises(NotImplementedError):
            mx.nn.QuantizedAllToShardedLinear(1, 1)
        with self.assertRaises(NotImplementedError):
            mx.nn.QuantizedShardedToAllLinear(1, 1)
        with self.assertRaises(NotImplementedError):
            mx.nn.ShardedToAllLinear(1, 1)
        with self.assertRaises(NotImplementedError):
            mx.nn.MultiHeadAttention(1, 1)
