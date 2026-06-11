import unittest
import numpy as np
import zero_mlx as mx


class TestNewOps(unittest.TestCase):
    def test_arccosh(self):
        x = mx.array([1.0, 2.0, 3.0])
        y = mx.arccosh(x)
        expected = np.arccosh(np.array([1.0, 2.0, 3.0]))
        self.assertTrue(np.allclose(np.array(y), expected))

    def test_arctan2(self):
        y = mx.array([1.0, -1.0])
        x = mx.array([1.0, 1.0])
        res = mx.arctan2(y, x)
        expected = np.arctan2(np.array([1.0, -1.0]), np.array([1.0, 1.0]))
        self.assertTrue(np.allclose(np.array(res), expected))

    def test_bitwise_invert(self):
        x = mx.array([1, 2, 3], dtype=mx.int32)
        y = mx.bitwise_invert(x)
        expected = np.bitwise_not(np.array([1, 2, 3], dtype=np.int32))
        self.assertTrue(np.array_equal(np.array(y), expected))

        y2 = mx.bitwise_invert(1)
        expected2 = np.bitwise_not(np.int32(1))
        self.assertEqual(np.array(y2).item(), expected2)

    def test_clear_cache(self):
        mx.clear_cache()
        self.assertTrue(True)

    def test_concat(self):
        x = mx.array([1, 2])
        y = mx.array([3, 4])
        res = mx.concat([x, y])
        self.assertTrue(np.array_equal(np.array(res), np.array([1, 2, 3, 4])))
        with self.assertRaises(ValueError):
            mx.concat([])

    def test_conj(self):
        x = mx.array([1 + 2j, 3 - 4j])
        res = mx.conj(x)
        self.assertTrue(np.array_equal(np.array(res), np.array([1 - 2j, 3 + 4j])))

    def test_contiguous(self):
        x = mx.array([1, 2, 3])
        res = mx.contiguous(x)
        self.assertTrue(np.array_equal(np.array(res), np.array(x)))

    def test_dtype_categories(self):
        self.assertEqual(mx.complexfloating.name, "complexfloating")
        self.assertEqual(mx.floating.name, "floating")
        self.assertEqual(mx.inexact.name, "inexact")
        self.assertEqual(mx.signedinteger.name, "signedinteger")
        self.assertEqual(mx.unsignedinteger.name, "unsignedinteger")
        self.assertEqual(mx.integer.name, "integer")
        self.assertEqual(mx.number.name, "number")
        self.assertEqual(mx.generic.name, "generic")

    def test_einsum(self):
        a = mx.array([[1, 2], [3, 4]])
        b = mx.array([[5, 6], [7, 8]])
        res = mx.einsum("ij,jk->ik", a, b)
        expected = np.einsum(
            "ij,jk->ik", np.array([[1, 2], [3, 4]]), np.array([[5, 6], [7, 8]])
        )
        self.assertTrue(np.allclose(np.array(res), expected))

    def test_einsum_path(self):
        a = mx.array([[1, 2], [3, 4]])
        b = mx.array([[5, 6], [7, 8]])
        path, info = mx.einsum_path("ij,jk->ik", a, b)
        self.assertIsInstance(path, list)
        self.assertIsInstance(info, str)

    def test_missing_ops(self):
        x = mx.array([[1, 2], [3, 4]])
        res = mx.permute_dims(x, axes=(1, 0))
        expected = np.transpose(np.array(x), axes=(1, 0))
        self.assertTrue(np.array_equal(np.array(res), expected))

        y = mx.tan(mx.array([0.0, 1.0]))
        expected_y = np.tan(np.array([0.0, 1.0]))
        self.assertTrue(np.allclose(np.array(y), expected_y))

        with self.assertRaises(NotImplementedError):
            mx.slice(x, mx.array(0), (0,), (1,))
        with self.assertRaises(NotImplementedError):
            mx.slice_update(x, x, mx.array(0), (0,))
        with self.assertRaises(NotImplementedError):
            mx.topk(x, 1)
        with self.assertRaises(NotImplementedError):
            mx.random.state()


if __name__ == "__main__":
    unittest.main()

    def test_quantize(self):
        with self.assertRaises(NotImplementedError):
            mx.quantize(mx.array(1))

    def test_core_types(self):
        self.assertTrue(hasattr(mx, "floating"))
        self.assertTrue(hasattr(mx, "generic"))
        self.assertTrue(hasattr(mx, "inexact"))
        self.assertTrue(hasattr(mx, "integer"))
        self.assertTrue(hasattr(mx, "number"))
        self.assertTrue(hasattr(mx, "signedinteger"))
        self.assertTrue(hasattr(mx, "unsignedinteger"))
