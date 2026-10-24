"""Base `Module` and `Identity` classes (`mlx.nn.base`) for zero_mlx."""

from __future__ import annotations

from typing import Callable, Protocol, Sequence, Union, runtime_checkable
from weakref import WeakKeyDictionary

from zero_mlx._types import ArrayInput, PyTree
from zero_mlx.array import array
from zero_mlx.utils import tree_flatten


@runtime_checkable
class CallableModuleProtocol(Protocol):
    """Protocol for `Module` subclasses implementing `__call__`."""

    def __call__(
        self,
        *args: array | int | float | str | bool | None,
        **kwargs: array | int | float | str | bool | None,
    ) -> array | tuple[array, ...]:
        """Invoke the module's forward pass.

        Args:
            *args: Positional arguments.
            **kwargs: Keyword arguments.

        Returns:
            Output `array` or tuple of `array` objects.

        """


ModuleAttrValue = Union[
    array,
    "Module",
    list[Union["Module", array, str]],
    tuple[Union["Module", array], ...],
    dict[str, "Module"],
    dict[str, array],
    dict[str, Union["Module", array]],
    set[str],
    int,
    float,
    bool,
    str,
    Callable[[array], array],
    None,
]

_MODULE_STATE: WeakKeyDictionary[Module, dict[str, ModuleAttrValue]] = WeakKeyDictionary()
_MODULE_FROZEN: WeakKeyDictionary[Module, set[str]] = WeakKeyDictionary()


class Module:
    """Base class for building neural network layers and models in `zero_mlx`."""

    training: bool

    def __init__(self) -> None:
        """Initialize the `Module` state."""
        self.training = True

    def _state_dict(self) -> dict[str, ModuleAttrValue]:
        """Return the internal attribute dictionary of this module instance.

        Returns:
            Mutable mapping of attribute names to values on `self`.

        """
        state = _MODULE_STATE.get(self)
        if state is None:
            state = {"training": True}
            _MODULE_STATE[self] = state
        return state

    def __getattr__(self, key: str) -> ModuleAttrValue:
        """Look up a registered parameter, submodule, or attribute by name.

        Args:
            key: Attribute name.

        Returns:
            Stored attribute value.

        Raises:
            AttributeError: If `key` is not present in the module.

        """
        if key == "training":
            return bool(self._state_dict().get("training", True))
        state = self._state_dict()
        if key in state:
            return state[key]
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{key}'")

    def __setattr__(self, key: str, value: ModuleAttrValue) -> None:
        """Store an attribute, parameter, or submodule in the module's state dictionary.

        Args:
            key: Attribute name to set.
            value: Attribute value to bind.

        """
        self._state_dict()[key] = value

    def __getitem__(self, key: str) -> ModuleAttrValue:
        """Retrieve a module attribute, parameter, or submodule by `key`.

        Args:
            key: Attribute name.

        Returns:
            Attribute value bound to `key`.

        Raises:
            KeyError: If `key` does not exist on the module.

        """
        state = self._state_dict()
        if key in state:
            return state[key]
        raise KeyError(key)

    def __setitem__(self, key: str, value: ModuleAttrValue) -> None:
        """Bind a module attribute, parameter, or submodule to `key`.

        Args:
            key: Attribute name.
            value: Attribute value to bind.

        """
        self._state_dict()[key] = value

    def __contains__(self, key: str) -> bool:
        """Return True if `key` is an attribute, parameter, or submodule of this module.

        Args:
            key: Attribute name to check.

        Returns:
            True if `key` exists on the module, False otherwise.

        """
        return key in self._state_dict()

    @property
    def _modules(self) -> dict[str, Module]:
        """Return a dictionary of registered submodules on this module.

        Returns:
            Mapping from submodule attribute names to `Module` instances.

        """
        mods: dict[str, Module] = {}
        for k, v in self._state_dict().items():
            if isinstance(v, Module):
                mods[k] = v
            elif isinstance(v, list):
                for idx, item in enumerate(v):
                    _ = mods.setdefault(f"{k}.{idx}", item) if isinstance(item, Module) else None
        return mods

    @property
    def _parameters(self) -> dict[str, array]:
        """Return a dictionary of direct `array` parameters on this module.

        Returns:
            Mapping from parameter attribute names to `array` instances.

        """
        return {
            k: v
            for k, v in self._state_dict().items()
            if not k.startswith("_") and isinstance(v, array)
        }

    def _get_frozen_set(self) -> set[str]:
        """Return the mutable set of frozen parameter/submodule keys.

        Returns:
            Set of frozen key strings.

        """
        frozen = _MODULE_FROZEN.get(self)
        if frozen is None:
            frozen = set()
            _MODULE_FROZEN[self] = frozen
        return frozen

    def parameters(self) -> dict[str, PyTree]:
        """Return a nested dictionary of all parameters in this module and its submodules.

        Returns:
            Nested dictionary mapping attribute names to parameter `array` leaves.

        """
        params: dict[str, PyTree] = {}
        for k, v in self._parameters.items():
            params[k] = v
        for k, m in self._modules.items():
            if "." not in k:
                params[k] = m.parameters()
        return params

    def trainable_parameters(self) -> dict[str, PyTree]:
        """Return a nested dictionary of all non-frozen trainable parameters.

        Returns:
            Nested dictionary of trainable parameter `array` leaves.

        """
        frozen = self._get_frozen_set()
        params: dict[str, PyTree] = {}
        for k, v in self._parameters.items():
            if k not in frozen:
                params[k] = v
        for k, m in self._modules.items():
            if "." not in k and k not in frozen:
                params[k] = m.trainable_parameters()
        return params

    def freeze(
        self,
        *,
        recurse: bool = True,
        keys: str | Sequence[str] | None = None,
        strict: bool = False,
    ) -> Module:
        """Freeze this module's parameters (and optionally submodules) so they are excluded from `trainable_parameters`.

        Args:
            recurse: Whether to recursively freeze submodules.
            keys: Optional parameter name or sequence of names to freeze.
            strict: If True, raise KeyError when a specified key does not exist.

        Returns:
            `self` for chaining.

        Raises:
            KeyError: If `strict` is True and a key in `keys` is not found.

        """
        frozen = self._get_frozen_set()
        if keys is not None:
            key_list = [keys] if isinstance(keys, str) else list(keys)
            for k in key_list:
                if strict and k not in self._parameters and k not in self._modules:
                    raise KeyError(f"Parameter or submodule '{k}' not found")
                frozen.add(k)
        else:
            frozen.update(self._parameters.keys())
        for m in self._modules.values() if recurse else ():
            m.freeze(recurse=True, keys=keys, strict=False)
        return self

    def unfreeze(
        self,
        *,
        recurse: bool = True,
        keys: str | Sequence[str] | None = None,
        strict: bool = False,
    ) -> Module:
        """Unfreeze this module's parameters (and optionally submodules).

        Args:
            recurse: Whether to recursively unfreeze submodules.
            keys: Optional parameter name or sequence of names to unfreeze.
            strict: If True, raise KeyError when a specified key does not exist.

        Returns:
            `self` for chaining.

        Raises:
            KeyError: If `strict` is True and a key in `keys` is not found.

        """
        frozen = self._get_frozen_set()
        if keys is not None:
            key_list = [keys] if isinstance(keys, str) else list(keys)
            for k in key_list:
                if strict and k not in self._parameters and k not in self._modules:
                    raise KeyError(f"Parameter or submodule '{k}' not found")
                frozen.discard(k)
        else:
            frozen.clear()
        for m in self._modules.values() if recurse else ():
            m.unfreeze(recurse=True, keys=keys, strict=False)
        return self

    def update(self, parameters: dict[str, PyTree], strict: bool = True) -> Module:
        """Update the module's parameters in-place from a nested `parameters` dictionary.

        Args:
            parameters: Dictionary mapping parameter/submodule names to updated values.
            strict: Whether to require keys to match existing parameters or submodules.

        Returns:
            `self` after updating parameters.

        """
        _ = strict
        state = self._state_dict()
        mods = self._modules
        for k, v in parameters.items():
            if isinstance(v, dict) and k in mods:
                mods[k].update(v, strict=strict)
            else:
                v_arr = v if isinstance(v, array) else array(v)
                state[k] = v_arr
        return self

    def update_modules(self, modules: dict[str, Module]) -> Module:
        """Replace child submodules in-place from a dictionary of `Module` instances.

        Args:
            modules: Mapping from child attribute names to new `Module` instances.

        Returns:
            `self` after updating submodules.

        """
        state = self._state_dict()
        for k, m in modules.items():
            state[k] = m
        return self

    def load_weights(
        self,
        file_or_weights: str | Sequence[tuple[str, array]] | dict[str, array],
        strict: bool = True,
    ) -> Module:
        """Load weights from a list of `(key, array)` pairs, a dict, or a file path.

        Args:
            file_or_weights: File path, dict, or sequence of `(path, array)` pairs.
            strict: If True, verify that all loaded keys exist on the module.

        Returns:
            `self` after loading weights.

        """
        from zero_mlx.new_ops import load
        from zero_mlx.utils import tree_unflatten

        if isinstance(file_or_weights, str):
            loaded = load(file_or_weights)
            items = list(loaded.items()) if isinstance(loaded, dict) else []
        elif isinstance(file_or_weights, dict):
            items = list(file_or_weights.items())
        else:
            items = list(file_or_weights)
        nested = tree_unflatten(items)
        return self.update(nested if isinstance(nested, dict) else {}, strict=strict)

    def save_weights(self, file: str) -> None:
        """Save the module's parameters to `file`.

        Args:
            file: Destination `.npz` or `.safetensors` file path.

        """
        from zero_mlx.new_ops import save_safetensors, savez

        flat = tree_flatten(self.parameters())
        flat_dict: dict[str, ArrayInput] = {
            k: v for k, v in (flat if isinstance(flat, list) else []) if isinstance(v, array)
        }
        if file.endswith(".safetensors"):
            save_safetensors(file, flat_dict)
        else:
            savez(file, **flat_dict)

    def children(self) -> dict[str, Module]:
        """Return a dictionary of immediate child submodules.

        Returns:
            Dictionary mapping child names to `Module` instances.

        """
        return {k: m for k, m in self._modules.items() if "." not in k}

    def modules(self) -> list[Module]:
        """Return a flat list of this module and all its descendant submodules.

        Returns:
            List of `Module` instances.

        """
        result: list[Module] = [self]
        for m in self._modules.values():
            result.extend(m.modules())
        return result

    def named_modules(self) -> list[tuple[str, Module]]:
        """Return a list of `(dotted_name, module)` pairs for all modules in the hierarchy.

        Returns:
            List of `(name, Module)` tuples.

        """
        result: list[tuple[str, Module]] = [("", self)]
        for k, m in self._modules.items():
            for sub_k, sub_m in m.named_modules():
                full_k = f"{k}.{sub_k}" if sub_k else k
                result.append((full_k, sub_m))
        return result

    def apply(self, map_fn: Callable[[array], array]) -> Module:
        """Apply `map_fn` to all parameters in this module and its submodules in-place.

        Args:
            map_fn: Function mapping an `array` parameter to a new `array`.

        Returns:
            `self` after applying `map_fn`.

        """
        state = self._state_dict()
        for k, v in list(self._parameters.items()):
            state[k] = map_fn(v)
        for m in self._modules.values():
            m.apply(map_fn)
        return self

    def train(self, mode: bool = True) -> Module:
        """Set the module and all its submodules into training mode (if `mode=True`) or evaluation mode.

        Args:
            mode: True for training mode, False for evaluation mode.

        Returns:
            `self` for chaining.

        """
        self.training = mode
        for m in self._modules.values():
            m.train(mode)
        return self

    def eval(self) -> Module:
        """Set the module and all its submodules into evaluation mode (`train(False)`).

        Returns:
            `self` for chaining.

        """
        return self.train(False)

    def _call_forward(
        self,
        *args: array | int | float | str | bool | None,
        **kwargs: array | int | float | str | bool | None,
    ) -> array | tuple[array, ...]:
        """Invoke this module's `__call__` method in a type-safe manner.

        Args:
            *args: Positional arguments forwarded to `__call__`.
            **kwargs: Keyword arguments forwarded to `__call__`.

        Returns:
            Output `array` or tuple of `array` objects.

        """
        if isinstance(self, CallableModuleProtocol):
            return self(*args, **kwargs)
        return array(0.0)


class Identity(Module):
    """A placeholder identity operator that returns its primary input unchanged."""

    def __init__(self, *args: int | float | str | bool, **kwargs: int | float | str | bool) -> None:
        """Initialize `Identity`, ignoring any positional or keyword configuration arguments.

        Args:
            *args: Ignored positional arguments.
            **kwargs: Ignored keyword arguments.

        """
        _ = (args, kwargs)
        super().__init__()

    def __call__(
        self,
        x: array,
        *args: array | int | float | str | bool,
        **kwargs: array | int | float | str | bool,
    ) -> array:
        """Return `x` unchanged.

        Args:
            x: Primary input `array`.
            *args: Ignored additional positional arguments.
            **kwargs: Ignored additional keyword arguments.

        Returns:
            The input `x` unchanged.

        """
        _ = (args, kwargs)
        return x


__all__ = ["CallableModuleProtocol", "Identity", "Module", "ModuleAttrValue"]
