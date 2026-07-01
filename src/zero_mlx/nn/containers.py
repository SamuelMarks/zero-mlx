"""mlx.nn.containers module."""

from typing import Any, Callable, Iterable
from zero_mlx.nn.base import Module


class Sequential(Module):
    """A layer that calls the passed callables in order."""

    def __init__(self, *modules: Callable) -> None:
        """Initialize the Sequential container.

        Args:
            *modules: The callables to be run in sequence.

        """
        super().__init__()
        self.layers = list(modules)
        for i, m in enumerate(modules):
            setattr(self, str(i), m)

    def __call__(self, x: Any, *args: Any, **kwargs: Any) -> Any:
        """Call the sequence of layers.

        Args:
            x: The input.
            *args: Additional arguments passed to the first layer.
            **kwargs: Keyword arguments passed to the first layer.

        Returns:
            The output of the sequence.

        """
        for i, layer in enumerate(self.layers):
            if i == 0:
                x = layer(x, *args, **kwargs)
            else:
                x = layer(x)
        return x
