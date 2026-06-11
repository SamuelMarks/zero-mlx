import unittest
import zero_mlx as mx


class TestNN(unittest.TestCase):
    def test_stubs(self):
        m = mx.nn.Module()
        self.assertIsInstance(m, mx.nn.Module)

        with self.assertRaises(NotImplementedError):
            mx.nn.ALiBi()

        with self.assertRaises(NotImplementedError):
            mx.nn.AllToShardedLinear(1, 1)

    def test_namespaces(self):
        self.assertTrue(hasattr(mx.nn, "containers"))
        self.assertTrue(hasattr(mx.nn, "convolution"))
        self.assertTrue(hasattr(mx.nn, "convolution_transpose"))


if __name__ == "__main__":
    unittest.main()
