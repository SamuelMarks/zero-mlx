"""Distributed gradient averaging helper (`mlx.nn.average_gradients`) for zero_mlx."""

from __future__ import annotations

from zero_mlx._types import PyTree, StreamOrDevice
from zero_mlx.array import array
from zero_mlx.distributed import Group, all_sum
from zero_mlx.utils import tree_map


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
    _ = (all_reduce_size, communication_type)
    grp = group if group is not None else Group()
    world_size = max(1, grp.size())

    def _avg_leaf(leaf: PyTree) -> PyTree:
        """All-reduce and scale a single `array` gradient leaf.

        Args:
            leaf: PyTree leaf value.

        Returns:
            Averaged `array` if `leaf` is an `array`, else `leaf`.

        """
        if isinstance(leaf, array):
            summed = all_sum(leaf, group=grp, stream=communication_stream)
            return summed * (1.0 / float(world_size)) if world_size > 1 else summed
        return leaf

    return tree_map(_avg_leaf, gradients)


__all__ = ["average_gradients"]
