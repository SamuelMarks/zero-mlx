import unittest
import zero_mlx as mx


class TestNNLayersTransformers(unittest.TestCase):
    def test_missing_nn_classes_transformer(self):
        with self.assertRaises(NotImplementedError):
            mx.nn.Transformer()
        with self.assertRaises(NotImplementedError):
            mx.nn.TransformerDecoder(1, 1, 1)
        with self.assertRaises(NotImplementedError):
            mx.nn.TransformerDecoderLayer(1, 1)
        with self.assertRaises(NotImplementedError):
            mx.nn.TransformerEncoder(1, 1, 1)
        with self.assertRaises(NotImplementedError):
            mx.nn.TransformerEncoderLayer(1, 1)


if __name__ == "__main__":
    unittest.main()
