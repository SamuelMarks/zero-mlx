"""Mock for in-place operations like `.at[idx].add(val)`."""

from zero_mlx.array import array
import numpy as np


class Adder:
    """Helper class for executing the in-place operation on a slice."""

    def __init__(self, arr, idx):
        """Initialize Adder.

        Args:
            arr: The array to modify.
            idx: The index to modify.

        """
        self.arr = arr
        self.idx = idx

    def _has_array_index(self):
        if isinstance(self.idx, np.ndarray):
            return True
        if isinstance(self.idx, tuple):
            return any(isinstance(i, np.ndarray) for i in self.idx)
        return False

    def _do_op(self, update, op_name):
        arr_np = np.array(self.arr)
        up_np = np.array(array(update))

        # Broadcasting up_np to arr_np[self.idx] shape if needed
        # ufunc.at requires up_np to be broadcastable to the shape of the indexed part
        try:
            target_shape = arr_np[self.idx].shape
            if up_np.shape != target_shape:
                up_np = np.reshape(up_np, target_shape)
        except Exception:
            pass

        if self._has_array_index():
            if op_name == "add":
                np.add.at(arr_np, self.idx, up_np)
            elif op_name == "subtract":
                np.subtract.at(arr_np, self.idx, up_np)
            elif op_name == "multiply":
                np.multiply.at(arr_np, self.idx, up_np)
            elif op_name == "divide":
                np.divide.at(arr_np, self.idx, up_np)
            elif op_name == "maximum":
                np.maximum.at(arr_np, self.idx, up_np)
            elif op_name == "minimum":
                np.minimum.at(arr_np, self.idx, up_np)
        else:
            if op_name == "add":
                arr_np[self.idx] += up_np
            elif op_name == "subtract":
                arr_np[self.idx] -= up_np
            elif op_name == "multiply":
                arr_np[self.idx] *= up_np
            elif op_name == "divide":
                arr_np[self.idx] /= up_np
            elif op_name == "maximum":
                arr_np[self.idx] = np.maximum(arr_np[self.idx], up_np)
            elif op_name == "minimum":
                arr_np[self.idx] = np.minimum(arr_np[self.idx], up_np)

        return array(arr_np, dtype=self.arr.dtype)

    def add(self, update):
        """In-place addition."""
        return self._do_op(update, "add")

    def subtract(self, update):
        """In-place subtraction."""
        return self._do_op(update, "subtract")

    def multiply(self, update):
        """In-place multiplication."""
        return self._do_op(update, "multiply")

    def divide(self, update):
        """In-place division."""
        return self._do_op(update, "divide")

    def maximum(self, update):
        """In-place maximum."""
        return self._do_op(update, "maximum")

    def minimum(self, update):
        """In-place minimum."""
        return self._do_op(update, "minimum")


class AtMocker:
    """Mocker for `.at` attribute."""

    def __init__(self, arr):
        """Initialize AtMocker.

        Args:
            arr: The array to mock.

        """
        self.arr = arr

    def __getitem__(self, idx):
        """Get Adder for the given index.

        Args:
            idx: The index to modify.

        Returns:
            An Adder instance.

        """
        if isinstance(idx, array):
            idx = np.array(idx)
        elif isinstance(idx, tuple):
            idx = tuple(np.array(i) if isinstance(i, array) else i for i in idx)
        return Adder(self.arr, idx)

    def add(self, _):
        """Invalid direct add.

        Args:
            _: Unused.

        Raises:
            ValueError: Always.

        """
        raise ValueError()
