import unittest
import zero_mlx as mx


class TestUtils(unittest.TestCase):
    def test_utils(self):
        self.assertTrue(hasattr(mx.utils, "Dict"))
        self.assertTrue(hasattr(mx.utils, "Any"))
        self.assertTrue(hasattr(mx.utils, "defaultdict"))
        self.assertTrue(hasattr(mx.utils, "zip_longest"))

        with self.assertRaises(NotImplementedError):
            mx.utils.tree_map_with_path(lambda x: x, None)


if __name__ == "__main__":
    unittest.main()
