import unittest
import zero_mlx as mx


class TestMissingNamespaces(unittest.TestCase):
    def test_namespaces(self):
        self.assertTrue(hasattr(mx, "linalg"))
        self.assertTrue(hasattr(mx, "metal"))
        self.assertTrue(hasattr(mx, "fast"))

        # Test core aliases
        self.assertTrue(hasattr(mx.core, "linalg"))
        self.assertTrue(hasattr(mx.core, "metal"))
        self.assertTrue(hasattr(mx.core, "fast"))
        self.assertTrue(hasattr(mx.core, "StreamContext"))
        self.assertTrue(hasattr(mx.core, "ArrayAt"))


if __name__ == "__main__":
    unittest.main()
