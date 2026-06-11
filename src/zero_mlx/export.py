from typing import Callable, Optional, Any


class FunctionExporter:
    """A context managing class for exporting multiple traces of the same function to a file."""

    def __init__(self, file: str, fun: Callable, shapeless: bool = False):
        self.file = file
        self.fun = fun
        self.shapeless = shapeless

    def __call__(self, *args, **kwargs) -> None:
        raise NotImplementedError("FunctionExporter is not implemented in zero_mlx.")

    def __enter__(self) -> "FunctionExporter":
        return self

    def __exit__(
        self,
        exc_type: Optional[Any] = None,
        exc_value: Optional[Any] = None,
        traceback: Optional[Any] = None,
    ) -> None:
        pass

    def close(self) -> None:
        pass


def exporter(file: str, fun: Callable, *, shapeless: bool = False) -> FunctionExporter:
    """Make a callable object to export multiple traces of a function to a file."""
    return FunctionExporter(file, fun, shapeless)


def export_function(
    arg0: Any, fun: Callable, *args, shapeless: bool = False, **kwargs
) -> None:
    """Export an MLX function."""
    raise NotImplementedError("export_function is not implemented in zero_mlx.")
