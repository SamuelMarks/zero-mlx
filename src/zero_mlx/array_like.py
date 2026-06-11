from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class ArrayLike(Protocol):
    """Any Python object which has an __mlx__array__ method that returns an array."""

    def __mlx_array__(self) -> Any: ...
