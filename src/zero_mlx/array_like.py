"""Protocol definition for MLX array-like objects."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from zero_mlx.array import array


@runtime_checkable
class ArrayLike(Protocol):
    """Any Python object which has an `__mlx_array__` method that returns an `array`."""

    def __mlx_array__(self) -> array:
        """Convert this object into a `zero_mlx.array`.

        Returns:
            The converted `zero_mlx.array` instance.

        """
