"""PyTree traversal and manipulation utilities (`mlx.utils`) for zero_mlx."""

from __future__ import annotations

from collections import defaultdict
from itertools import zip_longest
from typing import Callable, Dict, List, Optional, Protocol, Sequence, Tuple, Union

from zero_mlx._types import PyTree
from zero_mlx.array import array

LeafPredicate = Optional[Callable[[PyTree], bool]]


class TreeMapFn(Protocol):
    """Protocol for leaf transformation functions passed to `tree_map`."""

    def __call__(self, leaf: PyTree, *rest: PyTree) -> PyTree:
        """Transform a leaf value and optional parallel leaf values.

        Args:
            leaf: Primary leaf value.
            *rest: Parallel leaf values.

        Returns:
            Transformed leaf value.

        """


class TreeMapWithPathFn(Protocol):
    """Protocol for path-aware leaf transformation functions passed to `tree_map_with_path`."""

    def __call__(self, path: str, leaf: PyTree, *rest: PyTree) -> PyTree:
        """Transform a leaf value given its dot-separated `path`.

        Args:
            path: Dot-separated path string.
            leaf: Primary leaf value.
            *rest: Parallel leaf values.

        Returns:
            Transformed leaf value.

        """


def _is_leaf_node(tree: PyTree, is_leaf: LeafPredicate = None) -> bool:
    """Return True if `tree` should be treated as a leaf node.

    Args:
        tree: PyTree node to inspect.
        is_leaf: Optional user-supplied predicate returning True for leaves.

    Returns:
        True if `tree` is a leaf node, False if it is a container.

    """
    if is_leaf is not None and is_leaf(tree):
        return True
    return not isinstance(tree, (dict, list, tuple))


def tree_flatten(
    tree: PyTree,
    prefix: str = "",
    is_leaf: LeafPredicate = None,
    destination: list[tuple[str, PyTree]] | dict[str, PyTree] | None = None,
) -> list[tuple[str, PyTree]] | dict[str, PyTree]:
    """Flatten a nested Python tree into `(dot_path, leaf)` pairs or a flat dictionary.

    Args:
        tree: Nested dictionary, list, tuple, or leaf value.
        prefix: Dot-separated prefix prepended to keys.
        is_leaf: Optional predicate identifying custom leaf nodes.
        destination: Optional list or dict to populate in-place and return.

    Returns:
        List of `(key_path, value)` pairs, or `destination` if a dict was provided.

    """
    flat: list[tuple[str, PyTree]] = []

    def _recurse(curr: PyTree, p: str) -> None:
        """Recursively collect leaf nodes and their dotted paths.

        Args:
            curr: Current subtree.
            p: Current path prefix.

        """
        if _is_leaf_node(curr, is_leaf):
            key = p.lstrip(".")
            flat.append((key, curr))
            return
        if isinstance(curr, dict):
            for k, v in curr.items():
                _recurse(v, f"{p}.{k}" if p else str(k))
        else:
            seq_curr: Sequence[PyTree] = curr if isinstance(curr, (list, tuple)) else ()
            for i, v in enumerate(seq_curr):
                _recurse(v, f"{p}.{i}" if p else str(i))

    _recurse(tree, prefix)

    if isinstance(destination, list):
        destination.extend(flat)
        return destination
    if isinstance(destination, dict):
        for k, v in flat:
            destination[k] = v
        return destination
    return flat


def tree_unflatten(
    tree: Sequence[tuple[str, PyTree]] | dict[str, PyTree] | PyTree,
) -> PyTree:
    """Reconstruct a nested Python tree from flat `(dot_path, leaf)` pairs or a flat dict.

    Args:
        tree: Sequence of `(path, value)` pairs or mapping from `path` to `value`.

    Returns:
        Reconstructed nested Python tree (`dict` or `list`).

    """
    if not isinstance(tree, (dict, list, tuple)):
        return tree
    if isinstance(tree, dict):
        items: list[tuple[str, PyTree]] = list(tree.items())
    else:
        if tree and not isinstance(tree[0], tuple):
            return list(tree) if isinstance(tree, list) else tuple(tree)
        items = [
            (str(pair[0]), pair[1]) for pair in tree if isinstance(pair, tuple) and len(pair) == 2
        ]
    if not items:
        return {}
    if len(items) == 1 and items[0][0] == "":
        return items[0][1]

    is_root_list: bool = all(k.split(".", 1)[0].isdigit() for k, _ in items if k)
    if is_root_list:
        sub_groups: dict[int, list[tuple[str, PyTree]]] = {}
        direct_leaves: dict[int, PyTree] = {}
        max_idx: int = -1
        for k, v in items:
            parts = k.split(".", 1)
            idx = int(parts[0])
            if idx > max_idx:
                max_idx = idx
            if len(parts) == 1:
                direct_leaves[idx] = v
            else:
                sub_groups.setdefault(idx, []).append((parts[1], v))
        out_list: list[PyTree] = []
        for i in range(max_idx + 1):
            if i in direct_leaves:
                out_list.append(direct_leaves[i])
            elif i in sub_groups:
                out_list.append(tree_unflatten(sub_groups[i]))
            else:
                out_list.append(None)
        return out_list

    sub_dict_groups: dict[str, list[tuple[str, PyTree]]] = {}
    direct_dict_leaves: dict[str, PyTree] = {}
    key_order: list[str] = []
    for k, v in items:
        parts = k.split(".", 1)
        head = parts[0]
        if head not in key_order:
            key_order.append(head)
        if len(parts) == 1:
            direct_dict_leaves[head] = v
        else:
            sub_dict_groups.setdefault(head, []).append((parts[1], v))

    out_dict: dict[str, PyTree] = {}
    for head in key_order:
        if head in direct_dict_leaves:
            out_dict[head] = direct_dict_leaves[head]
        else:
            out_dict[head] = tree_unflatten(sub_dict_groups[head])
    return out_dict


TreeMapCallable = Union[
    TreeMapFn,
    Callable[[PyTree], PyTree],
    Callable[[PyTree, PyTree], PyTree],
]


def tree_map(
    fn: TreeMapCallable,
    tree: PyTree,
    *rest: PyTree,
    is_leaf: LeafPredicate = None,
) -> PyTree:
    """Apply `fn` to all leaves of `tree` (and any parallel trees in `rest`).

    Args:
        fn: Callable taking one argument per tree and returning the transformed leaf.
        tree: Primary PyTree structure.
        *rest: Additional parallel PyTrees with matching structure.
        is_leaf: Optional predicate identifying custom leaf nodes.

    Returns:
        New PyTree with the same structure and transformed leaves.

    """
    if _is_leaf_node(tree, is_leaf):
        return fn(tree, *rest)
    if isinstance(tree, dict):
        return {
            k: tree_map(
                fn,
                v,
                *(r[k] for r in rest if isinstance(r, dict)),
                is_leaf=is_leaf,
            )
            for k, v in tree.items()
        }
    if isinstance(tree, list):
        return [
            tree_map(
                fn,
                v,
                *(r[i] for r in rest if isinstance(r, (list, tuple))),
                is_leaf=is_leaf,
            )
            for i, v in enumerate(tree)
        ]
    seq_tree: Sequence[PyTree] = tree if isinstance(tree, tuple) else ()
    return tuple(
        tree_map(
            fn,
            v,
            *(r[i] for r in rest if isinstance(r, (list, tuple))),
            is_leaf=is_leaf,
        )
        for i, v in enumerate(seq_tree)
    )


TreeMapWithPathCallable = Union[
    TreeMapWithPathFn,
    Callable[[str, PyTree], PyTree],
    Callable[[str, PyTree, PyTree], PyTree],
]


def tree_map_with_path(
    fn: TreeMapWithPathCallable,
    tree: PyTree,
    *rest: PyTree,
    is_leaf: LeafPredicate = None,
    path: str | None = None,
) -> PyTree:
    """Apply `fn(path, leaf, *rest_leaves)` to every leaf in `tree`.

    Args:
        fn: Callable taking `(dot_path, leaf, *rest_leaves)` and returning a new leaf.
        tree: Primary PyTree structure.
        *rest: Additional parallel PyTrees.
        is_leaf: Optional predicate identifying custom leaf nodes.
        path: Current dot-separated path prefix.

    Returns:
        New PyTree with transformed leaves.

    """
    cur_path: str = "" if path is None else path
    if _is_leaf_node(tree, is_leaf):
        return fn(cur_path, tree, *rest)
    if isinstance(tree, dict):
        return {
            k: tree_map_with_path(
                fn,
                v,
                *(r[k] for r in rest if isinstance(r, dict)),
                is_leaf=is_leaf,
                path=f"{cur_path}.{k}" if cur_path else str(k),
            )
            for k, v in tree.items()
        }
    if isinstance(tree, list):
        return [
            tree_map_with_path(
                fn,
                v,
                *(r[i] for r in rest if isinstance(r, (list, tuple))),
                is_leaf=is_leaf,
                path=f"{cur_path}.{i}" if cur_path else str(i),
            )
            for i, v in enumerate(tree)
        ]
    seq_tree: Sequence[PyTree] = tree if isinstance(tree, tuple) else ()
    return tuple(
        tree_map_with_path(
            fn,
            v,
            *(r[i] for r in rest if isinstance(r, (list, tuple))),
            is_leaf=is_leaf,
            path=f"{cur_path}.{i}" if cur_path else str(i),
        )
        for i, v in enumerate(seq_tree)
    )


def tree_reduce(
    fn: Callable[[PyTree, PyTree], PyTree],
    tree: PyTree,
    initializer: PyTree = None,
    is_leaf: LeafPredicate = None,
) -> PyTree:
    """Apply a binary reduction function `fn(acc, leaf)` across all leaves of `tree`.

    Args:
        fn: Reduction function taking `(accumulator, leaf)`.
        tree: PyTree to reduce over.
        initializer: Initial accumulator value.
        is_leaf: Optional predicate identifying custom leaf nodes.

    Returns:
        Final accumulated value.

    """
    acc: PyTree = initializer
    if _is_leaf_node(tree, is_leaf):
        return tree if acc is None else fn(acc, tree)
    if isinstance(tree, dict):
        for v in tree.values():
            acc = tree_reduce(fn, v, acc, is_leaf)
    else:
        seq_tree: Sequence[PyTree] = tree if isinstance(tree, (list, tuple)) else ()
        for v in seq_tree:
            acc = tree_reduce(fn, v, acc, is_leaf)
    return acc


def tree_merge(
    tree_a: PyTree,
    tree_b: PyTree,
    merge_fn: Callable[[PyTree, PyTree], PyTree] | None = None,
) -> PyTree:
    """Merge two Python trees `tree_a` and `tree_b`.

    Args:
        tree_a: First PyTree.
        tree_b: Second PyTree.
        merge_fn: Optional function to combine colliding leaf values `(val_a, val_b)`.

    Returns:
        Merged PyTree.

    Raises:
        ValueError: If both trees are non-empty sequences of different lengths.

    """
    if isinstance(tree_a, dict) and not tree_a and tree_b is not None:
        return tree_b
    if isinstance(tree_b, dict) and not tree_b and tree_a is not None:
        return tree_a
    if tree_a is None:
        return tree_b
    if tree_b is None:
        return tree_a
    if isinstance(tree_a, dict) and isinstance(tree_b, dict):
        merged: dict[str, PyTree] = dict(tree_a)
        for k, v in tree_b.items():
            if k in merged:
                merged[k] = tree_merge(merged[k], v, merge_fn)
            else:
                merged[k] = v
        return merged
    if isinstance(tree_a, list) and isinstance(tree_b, list):
        if len(tree_a) != len(tree_b):
            raise ValueError("Cannot merge lists of different lengths")
        return [tree_merge(a, b, merge_fn) for a, b in zip(tree_a, tree_b)]
    if isinstance(tree_a, tuple) and isinstance(tree_b, tuple):
        if len(tree_a) != len(tree_b):
            raise ValueError("Cannot merge tuples of different lengths")
        return tuple(tree_merge(a, b, merge_fn) for a, b in zip(tree_a, tree_b))
    if merge_fn is not None:
        return merge_fn(tree_a, tree_b)
    return tree_b


def get_array_buffer_size(*args: PyTree) -> int:
    """Compute the total buffer size in bytes across all arrays in `*args`.

    Args:
        *args: Arbitrary PyTrees containing `zero_mlx.array` leaves.

    Returns:
        Total byte size across all discovered arrays.

    """
    total: int = 0
    for item in args:
        flat = tree_flatten(item)
        total += sum(
            leaf.nbytes
            for _, leaf in (flat if isinstance(flat, list) else [])
            if isinstance(leaf, array)
        )
    return total


__all__ = [
    "Callable",
    "Dict",
    "List",
    "Optional",
    "Tuple",
    "Union",
    "defaultdict",
    "get_array_buffer_size",
    "tree_flatten",
    "tree_map",
    "tree_map_with_path",
    "tree_merge",
    "tree_reduce",
    "tree_unflatten",
    "zip_longest",
]
