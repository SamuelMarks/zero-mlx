"""Numpy array interface injection."""


def inject_interface(cls):
    """Inject __array__ interface to mlx array class.

    Args:
        cls: The class to inject the interface into.

    """
    original_array = cls.__array__

    def __array__(self, dtype=None, copy=None):
        """Implement numpy __array__ interface.

        Args:
            self: The array instance.
            dtype: Target data type.
            copy: Whether to copy the array.

        Returns:
            The array representation.

        """
        import numpy as np

        kwargs = {}
        if dtype is not None:
            kwargs["dtype"] = dtype
        if copy is not None:
            kwargs["copy"] = copy

        arr = np.array(self._tensor.data, **kwargs)

        if copy is False or copy is None:
            try:
                # We do not have array_with_base anymore, so just return arr
                return arr
            except Exception:  # pragma: no cover
                pass  # pragma: no cover
        return arr  # pragma: no cover

    cls.__array__ = __array__
