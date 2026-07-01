"""mlx.core.distributed: Communication operations"""

from typing import Any, Optional


class Group:  # pragma: no cover
    """Group class."""

    def __init__(self) -> None:  # pragma: no cover
        """Initialize the Group."""
        pass


def all_gather(  # pragma: no cover
    x: Any, group: Optional[Group] = None, *, stream: Optional[Any] = None
) -> Any:
    """Perform an all_gather operation.

    Args:
        x (Any): The input array.
        group (Optional[Group], optional): The distributed group. Defaults to None.
        stream (Optional[Any], optional): The stream. Defaults to None.

    Returns:
        Any: The gathered array.

    Raises:
        NotImplementedError: Always raised as this is a stub.

    """
    from zero_mlx.array import array
    import ml_switcheroo_compiler.ops as sops

    return array(sops.distributed.all_gather(x._tensor))


def all_max(  # pragma: no cover
    x: Any, group: Optional[Group] = None, *, stream: Optional[Any] = None
) -> Any:
    """Perform an all_max operation.

    Args:
        x (Any): The input array.
        group (Optional[Group], optional): The distributed group. Defaults to None.
        stream (Optional[Any], optional): The stream. Defaults to None.

    Returns:
        Any: The max array.

    Raises:
        NotImplementedError: Always raised as this is a stub.

    """
    from zero_mlx.array import array
    import ml_switcheroo_compiler.ops as sops

    return array(sops.distributed.pmax(x._tensor, axis_name="group"))


def all_min(  # pragma: no cover
    x: Any, group: Optional[Group] = None, *, stream: Optional[Any] = None
) -> Any:
    """Perform an all_min operation.

    Args:
        x (Any): The input array.
        group (Optional[Group], optional): The distributed group. Defaults to None.
        stream (Optional[Any], optional): The stream. Defaults to None.

    Returns:
        Any: The min array.

    Raises:
        NotImplementedError: Always raised as this is a stub.

    """
    from zero_mlx.array import array
    import ml_switcheroo_compiler.ops as sops

    return array(sops.distributed.pmin(x._tensor, axis_name="group"))


def all_sum(  # pragma: no cover
    x: Any, group: Optional[Group] = None, *, stream: Optional[Any] = None
) -> Any:
    """Perform an all_sum operation.

    Args:
        x (Any): The input array.
        group (Optional[Group], optional): The distributed group. Defaults to None.
        stream (Optional[Any], optional): The stream. Defaults to None.

    Returns:
        Any: The sum array.

    Raises:
        NotImplementedError: Always raised as this is a stub.

    """
    from zero_mlx.array import array
    import ml_switcheroo_compiler.ops as sops

    return array(sops.distributed.all_reduce(x._tensor))


def init(strict: bool = False) -> Group:  # pragma: no cover
    """Initialize the distributed group.

    Args:
        strict (bool, optional): Whether to initialize strictly. Defaults to False.

    Returns:
        Group: A new distributed Group object.

    """
    return Group()


def is_available() -> bool:  # pragma: no cover
    """Check if distributed operations are available.

    Returns:
        bool: False, as distributed operations are not natively available in this stub.

    """
    return False


def recv(dst: Any, src: int, stream: Optional[Any] = None) -> Any:  # pragma: no cover
    """Receive data from a source.

    Args:
        dst (Any): The destination array.
        src (int): The source rank.
        stream (Optional[Any], optional): The stream. Defaults to None.

    Returns:
        Any: The received array.

    Raises:
        NotImplementedError: Always raised as this is a stub.

    """
    from zero_mlx.array import array

    return array(0)


def recv_like(  # pragma: no cover
    src: int, template: Any, stream: Optional[Any] = None
) -> Any:  # pragma: no cover
    """Receive data shaped like a template from a source.

    Args:
        src (int): The source rank.
        template (Any): The template array.
        stream (Optional[Any], optional): The stream. Defaults to None.

    Returns:
        Any: The received array.

    Raises:
        NotImplementedError: Always raised as this is a stub.

    """
    from zero_mlx.array import array

    return array(0)


def send(x: Any, dst: int, stream: Optional[Any] = None) -> Any:  # pragma: no cover
    """Send data to a destination.

    Args:
        x (Any): The input array to send.
        dst (int): The destination rank.
        stream (Optional[Any], optional): The stream. Defaults to None.

    Returns:
        Any: The sent array.

    Raises:
        NotImplementedError: Always raised as this is a stub.

    """
    pass


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
]
