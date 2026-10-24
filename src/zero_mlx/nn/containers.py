"""Container modules (`Sequential`) for `mlx.nn`."""

from __future__ import annotations

from typing import Callable, Union

from zero_mlx.array import array
from zero_mlx.nn.base import Module

LayerCallable = Union[Module, Callable[[array], array]]


class Sequential(Module):
    """A container module that calls a sequence of layers/callables in order."""

    def __init__(self, *modules: LayerCallable) -> None:
        """Initialize `Sequential` with the given layers or callables.

        Args:
            *modules: Variable number of `Module` instances or callables.

        """
        super().__init__()
        self.layers: list[LayerCallable] = list(modules)

    def __call__(
        self,
        x: array,
        *args: array | int | float | str | bool,
        **kwargs: array | int | float | str | bool,
    ) -> array:
        """Apply each layer in `self.layers` sequentially to `x`.

        Args:
            x: Input `array` to the first layer.
            *args: Additional positional arguments passed to the first layer.
            **kwargs: Additional keyword arguments passed to the first layer.

        Returns:
            Final output `array` after passing through all layers.

        """
        out: array = x
        for i, layer in enumerate(self.layers):
            if isinstance(layer, Module):
                res = (
                    layer._call_forward(out, *args, **kwargs)
                    if i == 0
                    else layer._call_forward(out)
                )
            else:
                res = layer(out)
            out = res[0] if isinstance(res, tuple) else res
        return out


__all__ = ["LayerCallable", "Sequential"]
