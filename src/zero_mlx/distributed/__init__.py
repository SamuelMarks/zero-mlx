"""mlx.core.distributed: Communication operations"""

import typing
from typing import Optional, Any
import ml_switcheroo_compiler.ops as sops


class Group:
    """Group class."""

    def __init__(self):
        pass


def all_gather(x: Any, group: Optional[Group] = None, *, stream: Any = None) -> Any:
    raise NotImplementedError("distributed.all_gather is not implemented")


def all_max(x: Any, group: Optional[Group] = None, *, stream: Any = None) -> Any:
    raise NotImplementedError("distributed.all_max is not implemented")


def all_min(x: Any, group: Optional[Group] = None, *, stream: Any = None) -> Any:
    raise NotImplementedError("distributed.all_min is not implemented")


def all_sum(x: Any, group: Optional[Group] = None, *, stream: Any = None) -> Any:
    raise NotImplementedError("distributed.all_sum is not implemented")


def init(strict: bool = False) -> Group:
    return Group()


def is_available() -> bool:
    return False


def recv(dst: Any, src: int, stream: Any = None) -> Any:
    raise NotImplementedError("distributed.recv is not implemented")


def recv_like(src: int, template: Any, stream: Any = None) -> Any:
    raise NotImplementedError("distributed.recv_like is not implemented")


def send(x: Any, dst: int, stream: Any = None) -> Any:
    raise NotImplementedError("distributed.send is not implemented")


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
