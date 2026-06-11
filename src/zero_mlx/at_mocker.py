"""Mock for in-place operations like `.at[idx].add(val)`."""

from zero_mlx.array import array
import ml_switcheroo.ops as sops


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

    def add(self, update):
        """In-place addition.

        Args:
            update: The value to add.

        Returns:
            The updated array.

        """
        res_tensor = sops.do_update(
            self.arr._tensor, self.idx, array(update)._tensor, "add"
        )
        return array(res_tensor, dtype=self.arr.dtype)

    def subtract(self, update):
        """In-place subtraction.

        Args:
            update: The value to subtract.

        Returns:
            The updated array.

        """
        res_tensor = sops.do_update(
            self.arr._tensor, self.idx, array(update)._tensor, "subtract"
        )
        return array(res_tensor, dtype=self.arr.dtype)

    def multiply(self, update):
        """In-place multiplication.

        Args:
            update: The value to multiply.

        Returns:
            The updated array.

        """
        res_tensor = sops.do_update(
            self.arr._tensor, self.idx, array(update)._tensor, "multiply"
        )
        return array(res_tensor, dtype=self.arr.dtype)

    def divide(self, update):
        """In-place division.

        Args:
            update: The value to divide.

        Returns:
            The updated array.

        """
        res_tensor = sops.do_update(
            self.arr._tensor, self.idx, array(update)._tensor, "divide"
        )
        return array(res_tensor, dtype=self.arr.dtype)

    def maximum(self, update):
        """In-place maximum.

        Args:
            update: The value to take the maximum with.

        Returns:
            The updated array.

        """
        res_tensor = sops.do_update(
            self.arr._tensor, self.idx, array(update)._tensor, "maximum"
        )
        return array(res_tensor, dtype=self.arr.dtype)

    def minimum(self, update):
        """In-place minimum.

        Args:
            update: The value to take the minimum with.

        Returns:
            The updated array.

        """
        res_tensor = sops.do_update(
            self.arr._tensor, self.idx, array(update)._tensor, "minimum"
        )
        return array(res_tensor, dtype=self.arr.dtype)


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
            idx = idx._tensor.__array__()
        elif isinstance(idx, tuple):
            idx = tuple(
                i._tensor.__array__() if isinstance(i, array) else i for i in idx
            )
        return Adder(self.arr, idx)

    def add(self, _):
        """Invalid direct add.

        Args:
            _: Unused.

        Raises:
            ValueError: Always.

        """
        raise ValueError()
