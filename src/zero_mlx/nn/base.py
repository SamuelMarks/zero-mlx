"""Base Module class."""

from typing import Any, Dict
from zero_mlx.array import array


class Module:
    """Base class for building neural networks with MLX."""

    def __init__(self) -> None:
        """Initialize Module."""
        self._modules: Dict[str, "Module"] = {}
        self._parameters: Dict[str, array] = {}
        self.training: bool = True

    def __setattr__(self, name: str, value: Any) -> None:
        """Set an attribute.

        Args:
            name (str): The name of the attribute.
            value (Any): The value of the attribute.

        """
        if isinstance(value, Module):
            if not hasattr(self, "_modules"):
                self.__dict__["_modules"] = {}
            self._modules[name] = value
        elif isinstance(value, array):
            if not hasattr(self, "_parameters"):
                self.__dict__["_parameters"] = {}
            self._parameters[name] = value
        super().__setattr__(name, value)

    def parameters(self) -> Dict[str, Any]:
        """Return parameters of the module.

        Returns:
            Dict[str, Any]: A dictionary of parameters.

        """
        params: Dict[str, Any] = {}
        if hasattr(self, "_parameters"):
            params.update(self._parameters)
        if hasattr(self, "_modules"):
            for k, v in self._modules.items():
                params[k] = v.parameters()
        return params

    def update(self, parameters: Dict[str, Any]) -> "Module":
        """Update parameters of the module.

        Args:
            parameters (Dict[str, Any]): The parameters to update.

        Returns:
            Module: The updated module.

        """
        for k, v in parameters.items():
            if isinstance(v, dict) and k in getattr(self, "_modules", {}):
                self._modules[k].update(v)
            elif k in getattr(self, "_parameters", {}):
                self._parameters[k] = v
                setattr(self, k, v)
        return self

    def train(self, mode: bool = True) -> "Module":
        """Set the module in training mode.

        Args:
            mode: True to set to training mode, False for evaluation mode.

        Returns:
            The module itself.

        """
        self.training = mode
        if hasattr(self, "_modules"):
            for m in self._modules.values():
                m.train(mode)
        return self

    def eval(self) -> "Module":
        """Set the module in evaluation mode.

        Returns:
            The module itself.

        """
        return self.train(False)

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """Call the module.

        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Raises:
            NotImplementedError: If not implemented in subclass.

        """
        raise NotImplementedError("Module must implement __call__")


class Identity(Module):
    """A placeholder identity operator that is argument-insensitive."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Initialize Identity.

        Args:
            *args: Ignored.
            **kwargs: Ignored.

        """
        super().__init__()

    def __call__(self, x: Any, *args: Any, **kwargs: Any) -> Any:
        """Return the input unchanged.

        Args:
            x: The input.
            *args: Ignored additional arguments.
            **kwargs: Ignored keyword arguments.

        Returns:
            The input unchanged.

        """
        return x
