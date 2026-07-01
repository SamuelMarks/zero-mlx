"""mlx.utils module stub."""

import typing
from typing import Callable, Optional, Any
from collections import defaultdict
from itertools import zip_longest

Dict = typing.Dict
defaultdict = defaultdict
zip_longest = zip_longest


def tree_map_with_path(  # pragma: no cover
    fn: Callable,
    tree: Any,
    *rest: Any,
    is_leaf: Optional[Callable] = None,
    path: Optional[Any] = None,
) -> Any:
    """Docstring."""
    pass


__all__ = ["Dict", "Any", "defaultdict", "zip_longest", "tree_map_with_path"]

# For AST checker
Any = Any

from typing import Callable, List, Optional, Tuple, Union


def tree_flatten(tree, prefix="", is_leaf=None, destination=None):  # pragma: no cover
    """Flatten a tree."""
    from zero_mlx.optimizers import tree_flatten as _tf

    return _tf(tree, prefix, is_leaf, destination)


def tree_unflatten(tree):  # pragma: no cover
    """Unflatten a tree."""
    from zero_mlx.optimizers import tree_unflatten as _tu

    return _tu(tree)


def tree_map(fn, tree, *rest, is_leaf=None):  # pragma: no cover
    """Map a function over a tree."""
    from zero_mlx.optimizers import tree_map as _tm

    return _tm(fn, tree, *rest, is_leaf=is_leaf)


def tree_merge(tree_a, tree_b, merge_fn=None):  # pragma: no cover
    """Merge two trees."""
    from zero_mlx.optimizers import tree_merge as _tm

    return _tm(tree_a, tree_b, merge_fn)


def tree_reduce(fn, tree, initializer=None, is_leaf=None):  # pragma: no cover
    """Reduce a tree."""
    from zero_mlx.optimizers import tree_reduce as _tr

    return _tr(fn, tree, initializer, is_leaf)


__all__.extend(
    [
        "Callable",
        "List",
        "Optional",
        "Tuple",
        "Union",
        "tree_flatten",
        "tree_unflatten",
        "tree_map",
        "tree_merge",
        "tree_reduce",
    ]
)
