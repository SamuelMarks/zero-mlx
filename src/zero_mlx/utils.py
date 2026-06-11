"""mlx.utils module stub."""

import typing
from typing import Callable, Optional, Any
from collections import defaultdict
from itertools import zip_longest

Dict = typing.Dict
defaultdict = defaultdict
zip_longest = zip_longest


def tree_map_with_path(
    fn: Callable,
    tree: Any,
    *rest: Any,
    is_leaf: Optional[Callable] = None,
    path: Optional[Any] = None,
) -> Any:
    raise NotImplementedError("tree_map_with_path not implemented")


__all__ = ["Dict", "Any", "defaultdict", "zip_longest", "tree_map_with_path"]
