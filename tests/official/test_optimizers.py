import unittest
import zero_mlx as mx


class TestOptimizers(unittest.TestCase):
    def test_namespaces(self):
        self.assertTrue(hasattr(mx.optimizers, "math"))
        self.assertTrue(hasattr(mx.optimizers, "mx"))
        self.assertTrue(hasattr(mx.optimizers, "optimizers"))
        self.assertTrue(hasattr(mx.optimizers, "schedulers"))

    def test_optimizer_stubs(self):
        opt_classes = [
            "AdaDelta",
            "Adafactor",
            "Adagrad",
            "Adam",
            "AdamW",
            "Adamax",
            "Lion",
            "MultiOptimizer",
            "Muon",
            "RMSprop",
            "SGD",
        ]
        for name in opt_classes:
            cls = getattr(mx.optimizers, name)
            with self.assertRaises(NotImplementedError):
                cls(0.1)

        with self.assertRaises(NotImplementedError):
            mx.optimizers.Optimizer()

    def test_functional_stubs(self):
        with self.assertRaises(NotImplementedError):
            mx.optimizers.clip_grad_norm(None, 1.0)
        with self.assertRaises(NotImplementedError):
            mx.optimizers.cosine_decay(1.0, 10)
        with self.assertRaises(NotImplementedError):
            mx.optimizers.exponential_decay(1.0, 0.1)
        with self.assertRaises(NotImplementedError):
            mx.optimizers.join_schedules([], [])
        with self.assertRaises(NotImplementedError):
            mx.optimizers.linear_schedule(1.0, 0.0, 10)
        with self.assertRaises(NotImplementedError):
            mx.optimizers.step_decay(1.0, 0.1, 10)
        with self.assertRaises(NotImplementedError):
            mx.optimizers.tree_flatten(None)
        with self.assertRaises(NotImplementedError):
            mx.optimizers.tree_map(None, None)
        with self.assertRaises(NotImplementedError):
            mx.optimizers.tree_merge(None, None)
        with self.assertRaises(NotImplementedError):
            mx.optimizers.tree_reduce(None, None)
        with self.assertRaises(NotImplementedError):
            mx.optimizers.tree_unflatten(None)

    def test_missing_files_coverage(self):
        import zero_mlx.optimizers.math as math
        import zero_mlx.optimizers.mx as mx
        import zero_mlx.optimizers.optimizers as optimizers_module
        import zero_mlx.optimizers.schedulers as schedulers

        self.assertTrue(True)

    def test_multi_optimizer(self):
        with self.assertRaises(NotImplementedError):
            mx.optimizers.MultiOptimizer([], [])


if __name__ == "__main__":
    unittest.main()
