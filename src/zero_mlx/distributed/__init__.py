"""Distributed communication operations (`mlx.core.distributed`) for zero_mlx."""

from __future__ import annotations

from typing import Sequence

import ml_switcheroo_compiler.ops as sops

from zero_mlx._types import ArrayInput, ShapeLike, StreamOrDevice
from zero_mlx.array import array
from zero_mlx.dtypes import DTypeLike, float32, resolve_dtype, to_switcheroo_dtype


class Group:
    """Representation of a distributed communication group."""

    def __init__(self, rank: int = 0, size: int = 1) -> None:
        """Initialize a distributed `Group`.

        Args:
            rank: Process rank within the group.
            size: Total number of processes in the group.

        """
        self._rank: int = rank
        self._size: int = size

    def rank(self) -> int:
        """Return the rank of the current process in the group.

        Returns:
            Integer process rank.

        """
        return self._rank

    def size(self) -> int:
        """Return the total number of processes in the group.

        Returns:
            Group size.

        """
        return self._size

    def split(self, color: int, key: int = -1) -> Group:
        """Split the group based on `color` and `key`.

        Args:
            color: Subgroup color identifier.
            key: Rank ordering key within the subgroup.

        Returns:
            New split `Group` instance.

        """
        _ = (color, key)
        return Group(self._rank, self._size)


def init(strict: bool = False, backend: str = "any") -> Group:
    """Initialize the distributed communication runtime and return the global group.

    Args:
        strict: If True, raise an error if distributed initialization fails.
        backend: Distributed communication backend name.

    Returns:
        Global `Group` object.

    """
    _ = (strict, backend)
    return Group(0, 1)


def is_available(backend: str = "any") -> bool:
    """Check if distributed communication is available.

    Args:
        backend: Distributed backend to check.

    Returns:
        False in single-process pure-Python mode.

    """
    _ = backend
    return False


def all_gather(
    x: ArrayInput,
    *,
    group: Group | None = None,
    stream: StreamOrDevice = None,
) -> array:
    """Gather arrays from all processes in the group and concatenate along axis 0.

    Args:
        x: Input array to gather.
        group: Optional distributed `Group`.
        stream: Optional stream or device.

    Returns:
        Gathered `array`.

    """
    _ = (group, stream)
    arr = x if isinstance(x, array) else array(x)
    return arr.copy()


def all_max(
    x: ArrayInput,
    *,
    group: Group | None = None,
    stream: StreamOrDevice = None,
) -> array:
    """Compute the element-wise maximum across all processes in the group.

    Args:
        x: Input array.
        group: Optional distributed `Group`.
        stream: Optional stream or device.

    Returns:
        Element-wise maximum `array`.

    """
    _ = (group, stream)
    arr = x if isinstance(x, array) else array(x)
    return arr.copy()


def all_min(
    x: ArrayInput,
    *,
    group: Group | None = None,
    stream: StreamOrDevice = None,
) -> array:
    """Compute the element-wise minimum across all processes in the group.

    Args:
        x: Input array.
        group: Optional distributed `Group`.
        stream: Optional stream or device.

    Returns:
        Element-wise minimum `array`.

    """
    _ = (group, stream)
    arr = x if isinstance(x, array) else array(x)
    return arr.copy()


def all_sum(
    x: ArrayInput,
    *,
    group: Group | None = None,
    stream: StreamOrDevice = None,
) -> array:
    """Compute the element-wise sum across all processes in the group.

    Args:
        x: Input array.
        group: Optional distributed `Group`.
        stream: Optional stream or device.

    Returns:
        Element-wise sum `array`.

    """
    _ = (group, stream)
    arr = x if isinstance(x, array) else array(x)
    return arr.copy()


def sum_scatter(
    x: ArrayInput,
    *,
    group: Group | None = None,
    stream: StreamOrDevice = None,
) -> array:
    """Sum arrays across all processes and scatter the result blocks across ranks.

    Args:
        x: Input array.
        group: Optional distributed `Group`.
        stream: Optional stream or device.

    Returns:
        Sum-scattered `array`.

    """
    return all_sum(x, group=group, stream=stream)


def send(
    x: ArrayInput,
    dst: int,
    *,
    group: Group | None = None,
    stream: StreamOrDevice = None,
) -> array:
    """Send an array `x` to destination rank `dst`.

    Args:
        x: Input array to send.
        dst: Destination rank.
        group: Optional distributed `Group`.
        stream: Optional stream or device.

    Returns:
        The sent `array`.

    """
    _ = (dst, group, stream)
    return x if isinstance(x, array) else array(x)


def recv(
    shape: ShapeLike,
    dtype: DTypeLike,
    src: int,
    *,
    group: Group | None = None,
    stream: StreamOrDevice = None,
) -> array:
    """Receive an array of given `shape` and `dtype` from source rank `src`.

    Args:
        shape: Shape of the array to receive.
        dtype: Data type of the array to receive.
        src: Source process rank.
        group: Optional distributed `Group`.
        stream: Optional stream or device.

    Returns:
        Received `array` initialized to zeros.

    """
    _ = (src, group, stream)
    shp: tuple[int, ...] = (
        () if shape is None else ((shape,) if isinstance(shape, int) else tuple(shape))
    )
    dt = resolve_dtype(dtype, default=float32)
    return array(sops.zeros(shp, dtype=to_switcheroo_dtype(dt)), dtype=dt)


def recv_like(
    x: ArrayInput,
    src: int,
    *,
    group: Group | None = None,
    stream: StreamOrDevice = None,
) -> array:
    """Receive an array with the same shape and dtype as `x` from rank `src`.

    Args:
        x: Template array defining shape and dtype.
        src: Source process rank.
        group: Optional distributed `Group`.
        stream: Optional stream or device.

    Returns:
        Received `array`.

    """
    arr = x if isinstance(x, array) else array(x)
    return recv(arr.shape, arr.dtype, src, group=group, stream=stream)


__all__ = [
    "Group",
    "all_gather",
    "all_max",
    "all_min",
    "all_sum",
    "init",
    "is_available",
    "recv",
    "recv_like",
    "send",
    "sum_scatter",
]
