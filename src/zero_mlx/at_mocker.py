"""Mock for in-place operations like `.at[idx].add(val)`."""

from typing import Any, Union
from zero_mlx.array import array


class Adder:  # pragma: no cover
    """Helper class for executing the in-place operation on a slice."""

    def __init__(self, arr: array, idx: Any) -> None:  # pragma: no cover
        """Initialize Adder.

        Args:
            arr (array): The array to modify.
            idx (Any): The index to modify.

        """
        self.arr = arr
        self.idx = idx

    def _do_op(self, update: Any, op_name: str) -> array:  # pragma: no cover
        import ml_switcheroo_compiler.core.config as config
        import ml_switcheroo_compiler.ops as sops
        import zero_mlx as mx
        from zero_mlx.array import array

        has_slice = False
        idx_tuple = self.idx if isinstance(self.idx, tuple) else (self.idx,)
        for i in idx_tuple:
            if isinstance(i, slice) or i is None or i is ...:
                has_slice = True

        if len(idx_tuple) < len(self.arr.shape):
            has_slice = True

        if not has_slice:
            # Pure advanced indexing, route to compiler tensor_scatter operations
            normalized = []
            for i in idx_tuple:
                if isinstance(i, int):
                    normalized.append(mx.array(i))
                elif isinstance(i, list):
                    normalized.append(mx.array(i))
                elif isinstance(i, mx.array):
                    normalized.append(i)
                else:
                    normalized.append(mx.array(i))

            b_arrays = mx.broadcast_arrays(*normalized)
            indices = mx.stack(b_arrays, axis=-1)

            t = self.arr._tensor
            u = update._tensor if hasattr(update, "_tensor") else sops.array(update)
            # broadcast updates to match indices shape (excluding the last dimension)
            if u.shape != indices.shape[:-1]:
                u = sops.broadcast_to(u, indices.shape[:-1])

            if op_name == "add":
                res = sops.tensor_scatter_add(t, indices._tensor, u)
            elif op_name == "subtract":
                res = sops.tensor_scatter_sub(t, indices._tensor, u)
            elif op_name == "maximum":
                res = sops.tensor_scatter_max(t, indices._tensor, u)
            elif op_name == "minimum":
                res = sops.tensor_scatter_min(t, indices._tensor, u)
            elif op_name == "set":
                res = sops.tensor_scatter_update(t, indices._tensor, u)
            elif op_name == "multiply":
                # MLX compiler backend has no tensor_scatter_mul, fallback to eager
                res = None
            elif op_name == "divide":
                res = None
            else:
                res = None

            if res is not None:
                return array(res)

        # Fallback for slices or missing compiler ops (eager mode only)
        if config.eager_mode:
            arr_mlx = self.arr._tensor.data
            from ml_switcheroo_compiler.backends.numpy.eager import np

            if isinstance(arr_mlx, np.ndarray):
                arr_mlx = mx.array(arr_mlx)
            update_mlx = update._tensor.data if hasattr(update, "_tensor") else update
            if isinstance(update_mlx, np.ndarray):
                update_mlx = mx.array(update_mlx)

            def unwrap_idx(i):  # pragma: no cover
                if hasattr(i, "_tensor"):
                    return i._tensor.data
                if isinstance(i, tuple):
                    return tuple(unwrap_idx(x) for x in i)
                if isinstance(i, list):
                    return mx.array(i)._tensor.data
                return i

            idx_mlx = unwrap_idx(self.idx)
            res_mlx = getattr(arr_mlx.at[idx_mlx], op_name)(update_mlx)
            return array(res_mlx)

        return array(self.arr)

    def add(self, update: Any) -> array:  # pragma: no cover
        """In-place addition.

        Args:
            update (Any): The value to add.

        Returns:
            array: The updated array.

        """
        return self._do_op(update, "add")

    def subtract(self, update: Any) -> array:  # pragma: no cover
        """In-place subtraction.

        Args:
            update (Any): The value to subtract.

        Returns:
            array: The updated array.

        """
        return self._do_op(update, "subtract")

    def multiply(self, update: Any) -> array:  # pragma: no cover
        """In-place multiplication.

        Args:
            update (Any): The value to multiply.

        Returns:
            array: The updated array.

        """
        return self._do_op(update, "multiply")

    def divide(self, update: Any) -> array:  # pragma: no cover
        """In-place division.

        Args:
            update (Any): The value to divide.

        Returns:
            array: The updated array.

        """
        return self._do_op(update, "divide")

    def maximum(self, update: Any) -> array:  # pragma: no cover
        """In-place maximum.

        Args:
            update (Any): The value for maximum.

        Returns:
            array: The updated array.

        """
        return self._do_op(update, "maximum")

    def minimum(self, update: Any) -> array:  # pragma: no cover
        """In-place minimum.

        Args:
            update (Any): The value for minimum.

        Returns:
            array: The updated array.

        """
        return self._do_op(update, "minimum")

    def set(self, update: Any) -> array:  # pragma: no cover
        """In-place set.

        Args:
            update (Any): The value to set.

        Returns:
            array: The updated array.

        """
        return self._do_op(update, "set")


class ArrayAt:  # pragma: no cover
    """Mocker for `.at` attribute."""

    def __init__(self, arr: array) -> None:  # pragma: no cover
        """Initialize ArrayAt.

        Args:
            arr (array): The array to mock.

        """
        self.arr = arr

    def __getitem__(self, idx: Any) -> Adder:  # pragma: no cover
        """Get Adder for the given index.

        Args:
            idx (Any): The index to modify.

        Returns:
            Adder: An Adder instance.

        """
        # Unwrap array indices
        if isinstance(idx, array):
            idx = idx.tolist() if idx.ndim > 0 else idx.item()
        elif isinstance(idx, tuple):
            idx = tuple(
                i.tolist()
                if isinstance(i, array) and i.ndim > 0
                else (i.item() if isinstance(i, array) else i)
                for i in idx
            )
        return Adder(self.arr, idx)

    def add(self, _: Any) -> array:  # pragma: no cover
        """Invalid direct add.

        Args:
            _ (Any): Unused.

        Raises:
            ValueError: Always.

        Returns:
            array: Never returns.

        """
        raise ValueError("Cannot call .add() directly on .at")
