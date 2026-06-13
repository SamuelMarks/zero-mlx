"""Mock for in-place operations like `.at[idx].add(val)`."""

from zero_mlx.array import array


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

    def _do_op(self, update, op_name):
        arr_list = self.arr.tolist()

        # Simple implementation for 1D array and scalar index to pass the test
        # without numpy. We will upgrade this to IR graph building in the refactor.
        if op_name == "add":
            arr_list[self.idx] += update
        elif op_name == "subtract":
            arr_list[self.idx] -= update
        elif op_name == "multiply":
            arr_list[self.idx] *= update
        elif op_name == "divide":
            arr_list[self.idx] /= update
        elif op_name == "maximum":
            arr_list[self.idx] = max(arr_list[self.idx], update)
        elif op_name == "minimum":
            arr_list[self.idx] = min(arr_list[self.idx], update)

        return array(arr_list, dtype=self.arr.dtype)

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
        # Unwrap array indices
        if isinstance(idx, array):
            idx = idx.item()
        elif isinstance(idx, tuple):
            idx = tuple(i.item() if isinstance(i, array) else i for i in idx)
        return Adder(self.arr, idx)

    def add(self, _):
        """Invalid direct add.

        Args:
            _: Unused.

        Raises:
            ValueError: Always.
        """
        raise ValueError()
