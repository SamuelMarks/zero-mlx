"""Indexed array update proxy (`.at[idx]`) for functional scatter operations."""

from __future__ import annotations

from typing import Sequence

from zero_mlx._types import ArrayInput, IndexLike, NestedSequence, Scalar
from zero_mlx.array import array


def _apply_scalar_op(lhs: Scalar, rhs: Scalar, op_name: str) -> Scalar:
    """Apply a binary update operation to two scalar values.

    Args:
        lhs: Current element value.
        rhs: Update element value.
        op_name: Name of the operation (`add`, `subtract`, `multiply`, `divide`, `power`, `maximum`, `minimum`, `set`).

    Returns:
        Updated scalar value.

    """
    if op_name == "set":
        return rhs
    if op_name == "add":
        if isinstance(lhs, complex) or isinstance(rhs, complex):
            return complex(lhs) + complex(rhs)
        if isinstance(lhs, float) or isinstance(rhs, float):
            return float(lhs.real) + float(rhs.real)
        return int(lhs.real) + int(rhs.real)
    if op_name == "subtract":
        if isinstance(lhs, complex) or isinstance(rhs, complex):
            return complex(lhs) - complex(rhs)
        if isinstance(lhs, float) or isinstance(rhs, float):
            return float(lhs.real) - float(rhs.real)
        return int(lhs.real) - int(rhs.real)
    if op_name == "multiply":
        if isinstance(lhs, complex) or isinstance(rhs, complex):
            return complex(lhs) * complex(rhs)
        if isinstance(lhs, float) or isinstance(rhs, float):
            return float(lhs.real) * float(rhs.real)
        return int(lhs.real) * int(rhs.real)
    if op_name == "divide":
        if isinstance(lhs, complex) or isinstance(rhs, complex):
            return complex(lhs) / complex(rhs)
        return float(lhs.real) / float(rhs.real)
    if op_name == "power":
        if isinstance(lhs, complex) or isinstance(rhs, complex):
            return complex(complex(lhs) ** complex(rhs))
        if isinstance(lhs, float) or isinstance(rhs, float):
            return float(float(lhs.real) ** float(rhs.real))
        return int(int(lhs.real) ** int(rhs.real))
    if op_name == "maximum":
        return lhs if float(lhs.real) >= float(rhs.real) else rhs
    return lhs if float(lhs.real) <= float(rhs.real) else rhs


def _update_1d_list(
    data: list[Scalar],
    indices: Sequence[int],
    updates: Sequence[Scalar],
    op_name: str,
) -> list[Scalar]:
    """Apply indexed scatter updates to a 1D list of scalars.

    Args:
        data: Mutable 1D list of scalar elements.
        indices: Sequence of integer target indices.
        updates: Sequence of update values matching `indices`.
        op_name: Operation name to apply at each index.

    Returns:
        Updated 1D list of scalars.

    """
    n: int = len(data)
    result: list[Scalar] = list(data)
    for idx, u_val in zip(indices, updates):
        norm_idx: int = idx if idx >= 0 else n + idx
        result[norm_idx] = _apply_scalar_op(result[norm_idx], u_val, op_name)
    return result


class Adder:
    """Helper class for executing functional indexed updates on a `zero_mlx.array`."""

    def __init__(self, arr: array, idx: IndexLike | list[int]) -> None:
        """Initialize `Adder` for a target array and index expression.

        Args:
            arr: The source `array` to update functionally.
            idx: Index expression specifying target positions.

        """
        self.arr: array = arr
        self.idx: IndexLike | list[int] = idx

    def _do_op(self, update: ArrayInput, op_name: str) -> array:
        """Execute the functional update operation and return a new `array`.

        Args:
            update: Value or array of values to apply at `self.idx`.
            op_name: Operation identifier (`add`, `subtract`, `multiply`, `divide`, `power`, `maximum`, `minimum`, `set`).

        Returns:
            Newly created `array` containing the applied updates.

        """
        upd_arr: array = (
            update if isinstance(update, array) else array(update, dtype=self.arr.dtype)
        )
        if self.arr.ndim == 1 and isinstance(self.idx, (int, list, slice, array)):
            raw_list = self.arr.tolist()
            flat_data: list[Scalar] = (
                [x if not isinstance(x, Sequence) else 0 for x in raw_list]
                if isinstance(raw_list, list)
                else [raw_list]
            )
            indices: list[int] = []
            if isinstance(self.idx, int):
                indices = [self.idx]
            elif isinstance(self.idx, slice):
                indices = list(range(*self.idx.indices(len(flat_data))))
            elif isinstance(self.idx, list):
                indices = [int(i) for i in self.idx]
            else:
                idx_vals = self.idx.tolist()
                if isinstance(idx_vals, list):
                    indices = [int(i.real) if not isinstance(i, Sequence) else 0 for i in idx_vals]
                else:
                    indices = [int(idx_vals.real)]

            if upd_arr.ndim == 0:
                upd_vals: list[Scalar] = [upd_arr.item()] * len(indices)
            else:
                u_list = upd_arr.tolist()
                upd_vals = (
                    [u if not isinstance(u, Sequence) else 0 for u in u_list]
                    if isinstance(u_list, list)
                    else [u_list]
                )
                if len(upd_vals) == 1 and len(indices) > 1:
                    upd_vals = upd_vals * len(indices)

            updated = _update_1d_list(flat_data, indices, upd_vals, op_name)
            return array(updated, dtype=self.arr.dtype)

        cloned: array = self.arr.copy()
        current_slice: array = cloned[self.idx]
        if op_name == "set":
            new_slice = upd_arr
        elif op_name == "add":
            new_slice = current_slice + upd_arr
        elif op_name == "subtract":
            new_slice = current_slice - upd_arr
        elif op_name == "multiply":
            new_slice = current_slice * upd_arr
        elif op_name == "divide":
            new_slice = current_slice / upd_arr
        elif op_name == "power":
            new_slice = current_slice**upd_arr
        elif op_name == "maximum":
            import zero_mlx.ops as _ops

            new_slice = _ops.maximum(current_slice, upd_arr)
        else:
            import zero_mlx.ops as _ops

            new_slice = _ops.minimum(current_slice, upd_arr)

        cloned[self.idx] = new_slice
        return cloned

    def add(self, update: ArrayInput) -> array:
        """Return a new array with `update` added at the specified indices.

        Args:
            update: Value or array to add.

        Returns:
            Updated `array`.

        """
        return self._do_op(update, "add")

    def subtract(self, update: ArrayInput) -> array:
        """Return a new array with `update` subtracted at the specified indices.

        Args:
            update: Value or array to subtract.

        Returns:
            Updated `array`.

        """
        return self._do_op(update, "subtract")

    def multiply(self, update: ArrayInput) -> array:
        """Return a new array with `update` multiplied at the specified indices.

        Args:
            update: Value or array to multiply by.

        Returns:
            Updated `array`.

        """
        return self._do_op(update, "multiply")

    def divide(self, update: ArrayInput) -> array:
        """Return a new array with `update` divided at the specified indices.

        Args:
            update: Value or array to divide by.

        Returns:
            Updated `array`.

        """
        return self._do_op(update, "divide")

    def power(self, update: ArrayInput) -> array:
        """Return a new array raised to `update` at the specified indices.

        Args:
            update: Exponent value or array.

        Returns:
            Updated `array`.

        """
        return self._do_op(update, "power")

    def maximum(self, update: ArrayInput) -> array:
        """Return a new array with element-wise maximum with `update` at the specified indices.

        Args:
            update: Value or array for maximum comparison.

        Returns:
            Updated `array`.

        """
        return self._do_op(update, "maximum")

    def minimum(self, update: ArrayInput) -> array:
        """Return a new array with element-wise minimum with `update` at the specified indices.

        Args:
            update: Value or array for minimum comparison.

        Returns:
            Updated `array`.

        """
        return self._do_op(update, "minimum")

    def set(self, update: ArrayInput) -> array:
        """Return a new array with `update` assigned at the specified indices.

        Args:
            update: Value or array to assign.

        Returns:
            Updated `array`.

        """
        return self._do_op(update, "set")


class ArrayAt:
    """Helper proxy accessed via `array.at` to perform indexed functional updates."""

    def __init__(self, arr: array) -> None:
        """Initialize `ArrayAt` for the given `array`.

        Args:
            arr: Target `zero_mlx.array`.

        """
        self.arr: array = arr

    def __getitem__(self, idx: IndexLike | list[int]) -> Adder:
        """Bind an index expression and return an `Adder` ready to apply an update.

        Args:
            idx: Index expression (`int`, `slice`, `array`, `list[int]`, or tuple of indices).

        Returns:
            An `Adder` instance bound to `self.arr` and `idx`.

        """
        return Adder(self.arr, idx)

    def add(self, update: ArrayInput) -> array:
        """Raise ValueError when `.add()` is called directly on `.at` without indexing.

        Args:
            update: Unused update value.

        Returns:
            Never returns.

        Raises:
            ValueError: Always raised because `.at` must be indexed first.

        """
        _ = update
        raise ValueError("Cannot call .add() directly on .at")
