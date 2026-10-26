"""Neural network functional and distributed utilities (`mlx.nn.utils`) for zero_mlx."""

from __future__ import annotations

from typing import Protocol

from zero_mlx._types import ArrayInput, DTypeLike, PyTree
from zero_mlx.array import array
from zero_mlx.distributed import Group
from zero_mlx.new_ops import _numerical_grad_single
from zero_mlx.nn.average_gradients import average_gradients
from zero_mlx.nn.base import Module
from zero_mlx.utils import tree_map_with_path, tree_unflatten


class LossCallable(Protocol):
    """Protocol for variadic loss functions differentiated by `nn.value_and_grad`."""

    def __call__(
        self,
        *args: array | Module,
        **kwargs: array,
    ) -> array:
        """Evaluate the loss function.

        Args:
            *args: Positional array or `Module` arguments.
            **kwargs: Keyword array arguments.

        Returns:
            Scalar or tensor loss `array`.

        """


class ValueAndGradCallable(Protocol):
    """Protocol for the function returned by `nn.value_and_grad`."""

    def __call__(self, *args: ArrayInput, **kwargs: ArrayInput) -> tuple[array, dict[str, PyTree]]:
        """Evaluate loss and gradients with respect to model trainable parameters.

        Args:
            *args: Positional arguments.
            **kwargs: Keyword arguments.

        Returns:
            Tuple `(loss_value, parameter_gradients_dict)`.

        """


class CheckpointCallable(Protocol):
    """Protocol for functions wrapped by `nn.checkpoint`."""

    def __call__(self, *args: array, **kwargs: array) -> array:
        """Execute the checkpointed forward function.

        Args:
            *args: Positional `array` arguments.
            **kwargs: Keyword `array` arguments.

        Returns:
            Output `array`.

        """


class FullyShardedModule(Module):
    """Wraps a `Module` for Fully Sharded Data Parallel (FSDP) execution across a distributed `Group`."""

    def __init__(
        self,
        module: Module,
        group: Group | None = None,
        compute_dtype: DTypeLike = None,
    ) -> None:
        """Initialize `FullyShardedModule`.

        Args:
            module: Wrapped neural network `Module`.
            group: Optional distributed communication `Group`.
            compute_dtype: Optional dtype used for forward/backward computation.

        """
        super().__init__()
        self.module: Module = module
        self.group: Group | None = group
        self.compute_dtype: DTypeLike = compute_dtype

    def __call__(self, *args: array, **kwargs: array) -> array | tuple[array, ...]:
        """Forward pass delegating to the wrapped `self.module`.

        Args:
            *args: Positional input arrays.
            **kwargs: Keyword input arrays.

        Returns:
            Output `array` or tuple of `array` objects from `self.module`.

        """
        res: array | tuple[array, ...] = self.module._call_forward(*args, **kwargs)
        return res


def _invoke_loss(
    fn: LossCallable,
    model: Module,
    args: tuple[array, ...],
    kwargs: dict[str, array],
) -> array:
    """Invoke a loss function with either `(*args, **kwargs)` or `(model, *args, **kwargs)`.

    Args:
        fn: Loss function callable.
        model: Neural network `Module`.
        args: Positional array arguments.
        kwargs: Keyword array arguments.

    Returns:
        Loss `array`.

    """
    try:
        res = fn(*args, **kwargs)
    except TypeError:
        res = fn(model, *args, **kwargs)
    return res if isinstance(res, array) else array(res)


def value_and_grad(model: Module, fn: LossCallable) -> ValueAndGradCallable:
    """Transform `fn` to compute both its value and gradients with respect to `model.trainable_parameters()`.

    Args:
        model: Neural network `Module` whose trainable parameters are differentiated.
        fn: Loss function accepting `*args, **kwargs` (or `(model, *args, **kwargs)`).

    Returns:
        Callable returning `(loss_value, parameter_gradients_dict)`.

    """

    def _wrapped(*args: ArrayInput, **kwargs: ArrayInput) -> tuple[array, dict[str, PyTree]]:
        """Evaluate `fn` and compute gradients for all trainable parameters in `model`.

        Args:
            *args: Positional arguments passed to `fn`.
            **kwargs: Keyword arguments passed to `fn`.

        Returns:
            Tuple `(value, gradients_tree)`.

        """
        arr_args = tuple(a if isinstance(a, array) else array(a) for a in args)
        arr_kwargs = {k: (v if isinstance(v, array) else array(v)) for k, v in kwargs.items()}
        val = _invoke_loss(fn, model, arr_args, arr_kwargs)

        def _grad_leaf(path: str, leaf: PyTree) -> PyTree:
            """Compute the parameter gradient for a single `array` leaf at `path`.

            Args:
                path: Dot-separated parameter path within `model`.
                leaf: Parameter leaf value.

            Returns:
                Gradient `array` matching `leaf`.

            """
            orig_param: array = leaf if isinstance(leaf, array) else array(leaf)

            def _loss_for_param(perturbed: array) -> array:
                """Evaluate loss with `path` set to `perturbed`.

                Args:
                    perturbed: Perturbed parameter `array`.

                Returns:
                    Scalar loss `array`.

                """
                patch = tree_unflatten([(path, perturbed)])
                model.update(patch if isinstance(patch, dict) else {}, strict=False)
                return _invoke_loss(fn, model, arr_args, arr_kwargs)

            grad_arr = _numerical_grad_single(_loss_for_param, orig_param)
            restore = tree_unflatten([(path, orig_param)])
            model.update(restore if isinstance(restore, dict) else {}, strict=False)
            return grad_arr

        grads = tree_map_with_path(_grad_leaf, model.trainable_parameters())
        return val, grads if isinstance(grads, dict) else {}

    return _wrapped


def checkpoint(module: Module, fn: CheckpointCallable | None = None) -> CheckpointCallable:
    """Wrap `fn` (or `module.__call__`) with activation checkpointing.

    Args:
        module: `Module` whose activations are checkpointed.
        fn: Optional function to checkpoint (defaults to `module`).

    Returns:
        Checkpointed callable.

    """

    def _checkpointed(*args: array, **kwargs: array) -> array:
        """Execute checkpointed forward pass.

        Args:
            *args: Positional arrays.
            **kwargs: Keyword arrays.

        Returns:
            Output `array`.

        """
        if fn is not None:
            return fn(*args, **kwargs)
        res = module._call_forward(*args, **kwargs)
        return res[0] if isinstance(res, tuple) else res

    return _checkpointed


__all__ = [
    "CheckpointCallable",
    "FullyShardedModule",
    "LossCallable",
    "ValueAndGradCallable",
    "average_gradients",
    "checkpoint",
    "value_and_grad",
]
