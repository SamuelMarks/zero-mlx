"""mlx.nn.average_gradients module stub."""

from typing import Any


def average_gradients(
    gradients: Any,
    group: Any = None,
    all_reduce_size: int = 33554432,
    communication_type: Any = None,
    communication_stream: Any = None,
) -> Any:
    """Average the gradients across the distributed processes in the passed group."""
    raise NotImplementedError("average_gradients is not implemented")
