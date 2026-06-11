import unittest
import numpy as np
import zero_mlx as mx
from ml_switcheroo.core.errors import UnimplementedMathError


class TestConvolutions(unittest.TestCase):
    def test_convolve(self):
        a = mx.array([1, 2, 3], dtype=mx.float32)
        v = mx.array([0, 1, 0.5], dtype=mx.float64)
        res = mx.convolve(a, v, mode="full")
        expected = np.convolve(np.array([1, 2, 3]), np.array([0, 1, 0.5]), mode="full")
        self.assertTrue(np.allclose(np.array(res), expected))
        self.assertEqual(res.dtype, mx.float64)

    def test_dequantize(self):
        w = mx.array([1, 2])
        scales = mx.array([1.0])
        with self.assertRaises(NotImplementedError):
            mx.dequantize(w, scales=scales)

    def test_conv1d(self):
        x = mx.random.uniform(shape=(1, 3, 5))
        w = mx.random.uniform(shape=(2, 3, 3))
        with self.assertRaises(UnimplementedMathError):
            mx.conv1d(x, w, padding=1)

    def test_conv2d(self):
        x = mx.random.uniform(shape=(1, 3, 5, 5))
        w = mx.random.uniform(shape=(2, 3, 3, 3))
        with self.assertRaises(UnimplementedMathError):
            mx.conv2d(x, w, padding=1)

    def test_conv3d(self):
        x = mx.random.uniform(shape=(1, 3, 5, 5, 5))
        w = mx.random.uniform(shape=(2, 3, 3, 3, 3))
        with self.assertRaises(UnimplementedMathError):
            mx.conv3d(x, w, padding=1)

    def test_conv_transpose1d(self):
        x = mx.random.uniform(shape=(1, 3, 5))
        w = mx.random.uniform(shape=(3, 2, 3))
        with self.assertRaises(UnimplementedMathError):
            mx.conv_transpose1d(x, w, padding=1)

    def test_conv_transpose2d(self):
        x = mx.random.uniform(shape=(1, 3, 5, 5))
        w = mx.random.uniform(shape=(3, 2, 3, 3))
        with self.assertRaises(UnimplementedMathError):
            mx.conv_transpose2d(x, w, padding=1)

    def test_conv_transpose3d(self):
        x = mx.random.uniform(shape=(1, 3, 5, 5, 5))
        w = mx.random.uniform(shape=(3, 2, 3, 3, 3))
        with self.assertRaises(UnimplementedMathError):
            mx.conv_transpose3d(x, w, padding=1)

    def test_conv_general(self):
        x = mx.random.uniform(shape=(1, 3, 5))
        w = mx.random.uniform(shape=(2, 3, 3))
        with self.assertRaises(UnimplementedMathError):
            mx.conv_general(x, w, padding=1)

        x2 = mx.random.uniform(shape=(1, 3, 5, 5))
        w2 = mx.random.uniform(shape=(2, 3, 3, 3))
        with self.assertRaises(UnimplementedMathError):
            mx.conv_general(x2, w2, padding=1)

        x3 = mx.random.uniform(shape=(1, 3, 5, 5, 5))
        w3 = mx.random.uniform(shape=(2, 3, 3, 3, 3))
        with self.assertRaises(UnimplementedMathError):
            mx.conv_general(x3, w3, padding=1)

        with self.assertRaises(NotImplementedError):
            mx.conv_general(mx.random.uniform(shape=(1, 3, 5, 5, 5, 5)), w3)


if __name__ == "__main__":
    unittest.main()

    def test_conv_general_flip(self):
        x = mx.random.uniform(shape=(1, 3, 5))
        w = mx.random.uniform(shape=(2, 3, 3))
        with self.assertRaises(UnimplementedMathError):
            mx.conv_general(x, w, padding=1, flip=True)
