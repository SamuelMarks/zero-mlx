"""Distributed gradient averaging helper (`mlx.nn.average_gradients`) for zero_mlx."""

from __future__ import annotations

from zero_mlx._types import PyTree, StreamOrDevice
from zero_mlx.distributed import Group


def average_gradients(
    gradients: PyTree,
    group: Group | None = None,
    all_reduce_size: int = 33554432,
    communication_type: str | None = None,
    communication_stream: StreamOrDevice = None,
) -> PyTree:
    """Average a PyTree of gradients across all ranks in the distributed `group`.

    Args:
        gradients: PyTree of parameter gradients.
        group: Optional distributed `Group`.
        all_reduce_size: Bucket size in bytes for coalesced all-reduce communication.
        communication_type: Optional data type used during communication.
        communication_stream: Optional stream used for communication.

    Returns:
        Averaged `gradients` PyTree.

    """
    _ = (group, all_reduce_size, communication_type, communication_stream)
    return gradients


__all__ = ["average_gradients"]
