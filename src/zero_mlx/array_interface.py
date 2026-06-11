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
        arr = self._tensor.__array__(dtype=dtype, copy=copy)
        from ml_switcheroo.core.array_base_helper import fix_complex, array_with_base

        if dtype is None:
            arr = fix_complex(arr, self.dtype.value)

        if copy is False or copy is None:
            try:
                return array_with_base(arr, self)
            except Exception:  # pragma: no cover
                pass  # pragma: no cover
        return arr  # pragma: no cover

    cls.__array__ = __array__
