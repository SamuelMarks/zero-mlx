"""zero_mlx API."""


class array:
    """Array class."""

    def __init__(self, data):
        """Initialize array."""
        self.data = data

    def __add__(self, other):
        """Add two arrays."""
        return array(self.data + getattr(other, "data", other))


class core:
    """Core namespace."""

    @staticmethod
    def eval(*args):
        """Evaluate."""
        # Lazy evaluation mock
        pass


class Stream:
    """Stream context."""

    def __enter__(self):
        """Enter context manager."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit context manager."""
        pass


def stream():
    """Create a stream."""
    return Stream()
