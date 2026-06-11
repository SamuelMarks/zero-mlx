"""Tests for ArrayIterator and ArrayAt."""

import unittest
import numpy as np
import zero_mlx as mx


class TestArrayIteratorAndAt(unittest.TestCase):
    """Tests for ArrayIterator and ArrayAt."""

    def test_array_iterator(self):
        """Test iterating over an array."""
        arr = mx.array([1, 2, 3])
        it = iter(arr)
        self.assertIsInstance(it, mx.ArrayIterator)

        items = list(it)
        self.assertEqual(len(items), 3)
        self.assertEqual(items[0].item(), 1)
        self.assertEqual(items[1].item(), 2)
        self.assertEqual(items[2].item(), 3)

        # Test 2D array
        arr2d = mx.array([[1, 2], [3, 4]])
        items2d = list(arr2d)
        self.assertEqual(len(items2d), 2)
        self.assertTrue(np.array_equal(items2d[0], mx.array([1, 2])))
        self.assertTrue(np.array_equal(items2d[1], mx.array([3, 4])))

        # Test empty array
        arr_empty = mx.array([])
        items_empty = list(arr_empty)
        self.assertEqual(len(items_empty), 0)

    def test_array_at(self):
        """Test array.at property."""
        arr = mx.array([1, 2, 3])
        at_mocker = arr.at
        self.assertIsInstance(at_mocker, mx.ArrayAt)

        res = at_mocker[1].add(10)
        self.assertEqual(res.tolist(), [1, 12, 3])

        with self.assertRaises(ValueError):
            at_mocker.add(10)


if __name__ == "__main__":
    unittest.main()
