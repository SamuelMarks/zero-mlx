import unittest
import zero_mlx as mx


class TestExport(unittest.TestCase):
    def test_export(self):
        def f(x):
            return x

        with self.assertRaises(NotImplementedError):
            mx.export_function(mx.array([1]), f)

        exp = mx.exporter("test.txt", f)
        self.assertIsInstance(exp, mx.FunctionExporter)

        with exp as e:
            self.assertEqual(e, exp)

        with self.assertRaises(NotImplementedError):
            exp(mx.array([1]))

        exp.close()


if __name__ == "__main__":
    unittest.main()
