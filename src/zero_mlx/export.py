"""mlx.export module stub."""

from typing import Callable, Optional, Any


class FunctionExporter:  # pragma: no cover
    """A context managing class for exporting multiple traces of the same function to a file."""

    def __init__(  # pragma: no cover
        self, file: str, fun: Callable, shapeless: bool = False
    ) -> None:  # pragma: no cover
        """Initialize the FunctionExporter.

        Args:
            file (str): The file path.
            fun (Callable): The function to export.
            shapeless (bool, optional): Whether to export without shapes. Defaults to False.

        """
        self.file = file
        self.fun = fun
        self.shapeless = shapeless

    def __call__(self, *args: Any, **kwargs: Any) -> None:  # pragma: no cover
        """Call the FunctionExporter.

        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        """
        pass

    def __enter__(self) -> "FunctionExporter":  # pragma: no cover
        """Enter the context manager.

        Returns:
            FunctionExporter: The current instance.

        """
        return self

    def __exit__(  # pragma: no cover
        self,
        exc_type: Optional[Any] = None,
        exc_value: Optional[Any] = None,
        traceback: Optional[Any] = None,
    ) -> None:
        """Exit the context manager.

        Args:
            exc_type (Optional[Any], optional): Exception type. Defaults to None.
            exc_value (Optional[Any], optional): Exception value. Defaults to None.
            traceback (Optional[Any], optional): Traceback. Defaults to None.

        """
        pass

    def close(self) -> None:  # pragma: no cover
        """Close the exporter."""
        pass


def exporter(  # pragma: no cover
    file: str, fun: Callable, *, shapeless: bool = False
) -> FunctionExporter:  # pragma: no cover
    """Make a callable object to export multiple traces of a function to a file.

    Args:
        file (str): The file path.
        fun (Callable): The function to export.
        shapeless (bool, optional): Whether to export without shapes. Defaults to False.

    Returns:
        FunctionExporter: The exporter object.

    """
    return FunctionExporter(file, fun, shapeless)


def export_function(  # pragma: no cover
    arg0: Any, fun: Callable, *args: Any, shapeless: bool = False, **kwargs: Any
) -> None:
    """Export an MLX function.

    Args:
        arg0 (Any): The output file or object.
        fun (Callable): The function to export.
        *args: Variable length argument list.
        shapeless (bool, optional): Whether to export without shapes. Defaults to False.
        **kwargs: Arbitrary keyword arguments.

    """
    pass
