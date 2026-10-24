"""Iterator for traversing the first dimension of a `zero_mlx.array`."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from zero_mlx.array import array


class ArrayIterator:
    """A helper iterator to traverse slices along the 1st dimension of an `array`."""

    def __init__(self, arr: array) -> None:
        """Initialize the `ArrayIterator`.

        Args:
            arr: The `zero_mlx.array` to iterate over.

        """
        self._arr: array = arr
        self._index: int = 0
        self._length: int = arr.shape[0] if len(arr.shape) > 0 else 0

    def __iter__(self) -> ArrayIterator:
        """Return the iterator object itself.

        Returns:
            The `ArrayIterator` instance.

        """
        return self

    def __next__(self) -> array:
        """Return the next slice along the first dimension of the array.

        Returns:
            The next element `array` along axis 0.

        Raises:
            StopIteration: When all elements along axis 0 have been visited.

        """
        if self._index >= self._length:
            raise StopIteration

        item: array = self._arr[self._index]
        self._index += 1
        return item
