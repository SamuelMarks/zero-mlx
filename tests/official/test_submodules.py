import unittest
import zero_mlx as mx


class TestSubmodules(unittest.TestCase):
    def test_distributed_stub(self):
        self.assertFalse(mx.distributed.is_available())
        g = mx.distributed.init()
        self.assertIsInstance(g, mx.distributed.Group)
        with self.assertRaises(NotImplementedError):
            mx.distributed.all_sum(mx.array(1))
        with self.assertRaises(NotImplementedError):
            mx.distributed.all_gather(mx.array(1))
        with self.assertRaises(NotImplementedError):
            mx.distributed.all_max(mx.array(1))
        with self.assertRaises(NotImplementedError):
            mx.distributed.all_min(mx.array(1))
        with self.assertRaises(NotImplementedError):
            mx.distributed.send(mx.array(1), 0)
        with self.assertRaises(NotImplementedError):
            mx.distributed.recv(mx.array(1), 0)
        with self.assertRaises(NotImplementedError):
            mx.distributed.recv_like(0, mx.array(1))

    def test_fast_stub(self):
        with self.assertRaises(NotImplementedError):
            mx.fast.layer_norm(1)
        with self.assertRaises(NotImplementedError):
            mx.fast.rms_norm(1)
        with self.assertRaises(NotImplementedError):
            mx.fast.rope(1)
        with self.assertRaises(NotImplementedError):
            mx.fast.scaled_dot_product_attention(1)
        with self.assertRaises(NotImplementedError):
            mx.fast.cuda_kernel(1)
        with self.assertRaises(NotImplementedError):
            mx.fast.metal_kernel(1)
        with self.assertRaises(NotImplementedError):
            mx.fast.precompiled_cuda_kernel(1)


if __name__ == "__main__":
    unittest.main()
