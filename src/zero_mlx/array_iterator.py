"""ArrayIterator class."""

from zero_mlx.array import array


class ArrayIterator:  # pragma: no cover
    """A helper object to iterate over the 1st dimension of an array."""

    def __init__(self, arr: array):  # pragma: no cover
        """Initialize ArrayIterator.

        Args:
            arr: The array to iterate over.

        """
        self._arr = arr
        self._index = 0
        self._length = arr.shape[0] if len(arr.shape) > 0 else 0

    def __iter__(self):  # pragma: no cover
        """Return the iterator object itself.

        Returns:
            The iterator object.

        """
        return self

    def __next__(self) -> array:  # pragma: no cover
        """Get the next element in the array.

        Returns:
            The next array element.

        Raises:
            StopIteration: If there are no more elements.

        """
        if self._index >= self._length:
            raise StopIteration

        item = self._arr[self._index]
        self._index += 1
        return item
