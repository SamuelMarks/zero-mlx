import unittest
import numpy as np
import zero_mlx as mx
import tempfile
import os


class TestMemoryIO(unittest.TestCase):
    def test_flatten_unflatten(self):
        a = mx.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
        f = mx.flatten(a, start_axis=1)
        self.assertEqual(f.shape, (2, 4))
        uf = mx.unflatten(f, axis=1, shape=(2, 2))
        self.assertEqual(uf.shape, (2, 2, 2))
        self.assertTrue(np.array_equal(np.array(a), np.array(uf)))

        # negative axes
        f2 = mx.flatten(a, start_axis=-2, end_axis=-1)
        self.assertEqual(f2.shape, (2, 4))

        f3 = mx.flatten(a, start_axis=2, end_axis=1)
        self.assertEqual(f3.shape, a.shape)

        uf2 = mx.unflatten(f, axis=-1, shape=(2, 2))
        self.assertEqual(uf2.shape, (2, 2, 2))

    def test_identity(self):
        a = mx.identity(3, dtype=mx.int32)
        expected = np.identity(3, dtype=np.int32)
        self.assertTrue(np.array_equal(np.array(a), expected))

    def test_stubs(self):
        with self.assertRaises(NotImplementedError):
            mx.hadamard_transform(mx.array(1))
        with self.assertRaises(NotImplementedError):
            mx.gather_qmm(mx.array(1), mx.array(1), scales=mx.array(1))
        with self.assertRaises(NotImplementedError):
            mx.quantized_matmul(mx.array(1), mx.array(1), scales=mx.array(1))
        with self.assertRaises(NotImplementedError):
            mx.save_gguf("test", {}, {})
        with self.assertRaises(NotImplementedError):
            mx.save_safetensors("test", {})
        with self.assertRaises(NotImplementedError):
            mx.import_function("test")

        self.assertEqual(mx.get_active_memory(), 0)
        self.assertEqual(mx.get_cache_memory(), 0)
        mx.reset_peak_memory()
        self.assertEqual(mx.set_cache_limit(10), 10)
        self.assertEqual(mx.set_memory_limit(20), 20)
        self.assertEqual(mx.set_wired_limit(30), 30)

        # default stream
        d = mx.Device(mx.cpu)
        s = mx.StreamContext(d)
        mx.set_default_stream(s)

    def test_load_save(self):
        with tempfile.TemporaryDirectory() as td:
            f1 = os.path.join(td, "arr.npy")
            a = mx.array([1, 2, 3])
            mx.save(f1, a)
            b = mx.load(f1)
            self.assertTrue(np.array_equal(np.array(a), np.array(b)))

            b2, meta = mx.load(f1, return_metadata=True)
            self.assertTrue(np.array_equal(np.array(a), np.array(b2)))
            self.assertEqual(meta, {})

            f2 = os.path.join(td, "arrs.npz")
            mx.savez(f2, a, y=a)
            b_dict = mx.load(f2)
            self.assertTrue(np.array_equal(np.array(a), np.array(b_dict["arr_0"])))
            self.assertTrue(np.array_equal(np.array(a), np.array(b_dict["y"])))

            b_dict2, meta = mx.load(f2, return_metadata=True)
            self.assertEqual(meta, {})

            f3 = os.path.join(td, "arrs_comp.npz")
            mx.savez_compressed(f3, a, y=a)
            b_dict = mx.load(f3)
            self.assertTrue(np.array_equal(np.array(a), np.array(b_dict["arr_0"])))
            self.assertTrue(np.array_equal(np.array(a), np.array(b_dict["y"])))


if __name__ == "__main__":
    unittest.main()
