"""Numpy array representation injection."""


def inject_repr(cls):
    """Inject __repr__ to mlx array class.

    Args:
        cls: The class to inject the representation into.

    """

    def __repr__(self) -> str:
        """Get string representation of the array.

        Returns:
            The string representation.

        """
        from ml_switcheroo.core.repr_helper import get_repr
        import zero_mlx as mx

        precision = (
            mx.core.printoptions_precision
            if hasattr(mx.core, "printoptions_precision")
            else 5
        )
        threshold = 6 if self.ndim <= 1 else (12 if self.ndim >= 2 else 1000)

        dtype_val = self._original_dtype.value

        return get_repr(self._tensor, dtype_val, precision, threshold)

    cls.__repr__ = __repr__
    cls.__str__ = __repr__
