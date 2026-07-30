"""mlx.nn.average_gradients module stub."""

from typing import Any, Optional


def average_gradients(  # pragma: no cover
    gradients: Any,
    group: Optional[Any] = None,
    all_reduce_size: int = 33554432,
    communication_type: Optional[Any] = None,
    communication_stream: Optional[Any] = None,
) -> Any:
    """Average the gradients across the distributed processes in the passed group.

    Args:
        gradients (Any): The gradients to average.
        group (Optional[Any], optional): The distributed group. Defaults to None.
        all_reduce_size (int, optional): The chunk size for all reduce. Defaults to 33554432.
        communication_type (Optional[Any], optional): Type of communication. Defaults to None.
        communication_stream (Optional[Any], optional): Stream for communication. Defaults to None.

    Returns:
        Any: The averaged gradients.


    """
    return gradients
