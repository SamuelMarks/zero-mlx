"""mlx.core.random.state stub."""


class state:  # pragma: no cover
    """Random state stub.

    This class provides a stub for the random state management in mlx.
    """

    def __new__(cls) -> None:  # pragma: no cover
        """Initialize the random state stub.

        Raises:
            NotImplementedError: Always raised as this is a stub.

        """
        from ml_switcheroo_compiler.random.state import get_global_generator

        return get_global_generator()
