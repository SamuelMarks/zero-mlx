"""Base `Module` and `Identity` classes (`mlx.nn.base`) for zero_mlx."""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable, Iterator, Protocol, Sequence, Union, runtime_checkable

from zero_mlx._types import ArrayInput, DTypeLike, PyTree
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
    list[Union["Module", array, str, "ModuleAttrValue"]],
    tuple[Union["Module", array, "ModuleAttrValue"], ...],
    dict[str, "Module"],
    dict[str, array],
    dict[str, Union["Module", array]],
    dict[str, "ModuleAttrValue"],
    int,
    float,
    bool,
    str,
    Callable[[array], array],
    None,
]


class ModuleAttr:
    """Data descriptor routing `Module` parameter and submodule attributes to `_state_dict()`."""

    def __init__(self, name: str) -> None:
        """Initialize the descriptor for attribute `name`.

        Args:
            name: Attribute key in the module's state dictionary.

        """
        self.name: str = name

    def __get__(
        self,
        instance: Module | None,
        owner: type[Module] | None = None,
    ) -> ModuleAttrValue | ModuleAttr:
        """Read the attribute value from `instance._state_dict()`.

        Args:
            instance: `Module` instance, or None when accessed on the class.
            owner: Owning `Module` class.

        Returns:
            Stored attribute value, or `self` when accessed on the class.

        Raises:
            AttributeError: If `self.name` is not set on `instance`.

        """
        _ = owner
        if instance is None:
            return self
        state = instance._state_dict()
        if self.name in state:
            return state[self.name]
        raise AttributeError(f"'{type(instance).__name__}' object has no attribute '{self.name}'")

    def __set__(self, instance: Module, value: ModuleAttrValue) -> None:
        """Write `value` into `instance._state_dict()[self.name]`.

        Args:
            instance: Target `Module` instance.
            value: Attribute value to store.

        """
        instance._state_dict()[self.name] = value


_ModuleAttrDescriptor = ModuleAttr


def _filter_and_map_container(
    parent: Module,
    container: (
        list[Module | array | str | ModuleAttrValue]
        | tuple[Module | array | ModuleAttrValue, ...]
        | dict[str, Module]
        | dict[str, array]
        | dict[str, Module | array]
        | dict[str, ModuleAttrValue]
    ),
    filter_fn: Callable[[Module, str, ModuleAttrValue], bool],
    map_fn: Callable[[ModuleAttrValue], ModuleAttrValue] | None = None,
    is_leaf_fn: Callable[[Module, str, ModuleAttrValue], bool] | None = None,
) -> ModuleAttrValue:
    """Recursively filter and map a nested `list`, `tuple`, or `dict` container on `parent`.

    Args:
        parent: Owning `Module`.
        container: Container attribute value (`list`, `tuple`, or `dict`).
        filter_fn: Predicate `(module, key, value) -> bool`.
        map_fn: Optional leaf transformation function.
        is_leaf_fn: Optional custom leaf predicate.

    Returns:
        Filtered and mapped container of the same collection kind.

    """
    if isinstance(container, (list, tuple)):
        out_seq: list[ModuleAttrValue] = []
        for item in container:
            if isinstance(item, Module):
                out_seq.append(item.filter_and_map(filter_fn, map_fn=map_fn, is_leaf_fn=is_leaf_fn))
            else:
                out_seq.append(map_fn(item) if map_fn is not None else item)
        return tuple(out_seq) if isinstance(container, tuple) else out_seq

    out_dict: dict[str, ModuleAttrValue] = {}
    for sub_k, sub_v in container.items():
        if isinstance(sub_v, Module):
            out_dict[sub_k] = sub_v.filter_and_map(filter_fn, map_fn=map_fn, is_leaf_fn=is_leaf_fn)
        else:
            out_dict[sub_k] = map_fn(sub_v) if map_fn is not None else sub_v
    return out_dict


def _extract_container_params(
    container: (
        list[Module | array | str | ModuleAttrValue]
        | tuple[Module | array | ModuleAttrValue, ...]
        | dict[str, Module]
        | dict[str, array]
        | dict[str, Module | array]
        | dict[str, ModuleAttrValue]
    ),
    trainable_only: bool,
) -> PyTree:
    """Recursively extract parameter PyTrees from a `list`, `tuple`, or `dict` container.

    Args:
        container: Sequence or mapping of `Module` or `array` elements.
        trainable_only: If True, call `trainable_parameters()` on nested modules.

    Returns:
        Extracted parameter `list` or `dict`.

    """
    if isinstance(container, (list, tuple)):
        return [
            (item.trainable_parameters() if trainable_only else item.parameters())
            if isinstance(item, Module)
            else item
            for item in container
            if isinstance(item, (Module, array))
        ]

    return {
        k: (v.trainable_parameters() if trainable_only else v.parameters())
        if isinstance(v, Module)
        else v
        for k, v in container.items()
        if isinstance(v, (Module, array))
    }


def _has_any_array_leaf(tree: PyTree) -> bool:
    """Return True if `tree` contains at least one `array` leaf.

    Args:
        tree: Nested PyTree structure.

    Returns:
        True if an `array` leaf is present in `tree`.

    """
    if isinstance(tree, array):
        return True
    if isinstance(tree, dict):
        return any(_has_any_array_leaf(v) for v in tree.values())
    return isinstance(tree, (list, tuple)) and any(_has_any_array_leaf(v) for v in tree)


class Module:
    """Base class for building neural network layers and models in `zero_mlx`."""

    _state: dict[str, ModuleAttrValue] | None = None
    _frozen: set[str] | None = None

    if TYPE_CHECKING:
        p: array | float
    else:
        weight = ModuleAttr("weight")
        bias = ModuleAttr("bias")
        scales = ModuleAttr("scales")
        biases = ModuleAttr("biases")
        weight_ih = ModuleAttr("weight_ih")
        weight_hh = ModuleAttr("weight_hh")
        bias_ih = ModuleAttr("bias_ih")
        bias_hh = ModuleAttr("bias_hh")
        running_mean = ModuleAttr("running_mean")
        running_var = ModuleAttr("running_var")
        layers = ModuleAttr("layers")
        query_proj = ModuleAttr("query_proj")
        key_proj = ModuleAttr("key_proj")
        value_proj = ModuleAttr("value_proj")
        out_proj = ModuleAttr("out_proj")
        attention = ModuleAttr("attention")
        self_attn = ModuleAttr("self_attn")
        cross_attn = ModuleAttr("cross_attn")
        ln = ModuleAttr("ln")
        ln1 = ModuleAttr("ln1")
        ln2 = ModuleAttr("ln2")
        ln3 = ModuleAttr("ln3")
        linear1 = ModuleAttr("linear1")
        linear2 = ModuleAttr("linear2")
        dropout1 = ModuleAttr("dropout1")
        dropout2 = ModuleAttr("dropout2")
        dropout3 = ModuleAttr("dropout3")
        encoder = ModuleAttr("encoder")
        decoder = ModuleAttr("decoder")
        module = ModuleAttr("module")
        p = ModuleAttr("p")

    def __init__(self) -> None:
        """Initialize the `Module` state."""
        self._state = {"training": True}
        self._frozen = set()

    @property
    def training(self) -> bool:
        """Return True if the module is in training mode, False if in evaluation mode.

        Returns:
            Boolean training mode flag.

        """
        return bool(self._state_dict().get("training", True))

    @property
    def state(self) -> dict[str, ModuleAttrValue]:
        """Return the mutable attribute state dictionary of this module.

        Returns:
            Mutable mapping of module attributes, parameters, and submodules.

        """
        return self._state_dict()

    def keys(self) -> list[str]:
        """Return the list of attribute keys stored in this module.

        Returns:
            List of attribute keys.

        """
        return list(self._state_dict().keys())

    def values(self) -> list[ModuleAttrValue]:
        """Return the list of attribute values stored in this module.

        Returns:
            List of attribute values.

        """
        return list(self._state_dict().values())

    def items(self) -> list[tuple[str, ModuleAttrValue]]:
        """Return the list of `(key, value)` pairs stored in this module.

        Returns:
            List of `(key, value)` tuples.

        """
        return list(self._state_dict().items())

    def __iter__(self) -> Iterator[str]:
        """Iterate over attribute keys stored in this module.

        Returns:
            Iterator over attribute key strings.

        """
        return iter(self._state_dict())

    def get(self, key: str, default: ModuleAttrValue = None) -> ModuleAttrValue:
        """Return the value for `key` if `key` is in the module, else `default`.

        Args:
            key: Attribute key to look up.
            default: Fallback value if `key` is not present.

        Returns:
            Stored value or `default`.

        """
        return self._state_dict().get(key, default)

    def pop(self, key: str, default: ModuleAttrValue = None) -> ModuleAttrValue:
        """Remove `key` from the module state and return its value (or `default`).

        Args:
            key: Attribute key to remove.
            default: Fallback value if `key` is not present.

        Returns:
            Removed value or `default`.

        """
        return self._state_dict().pop(key, default)

    def popitem(self) -> tuple[str, ModuleAttrValue]:
        """Remove and return the last `(key, value)` pair from the module state.

        Returns:
            Removed `(key, value)` tuple.

        """
        return self._state_dict().popitem()

    def clear(self) -> None:
        """Remove all keys from the module state."""
        self._state_dict().clear()

    def copy(self) -> dict[str, ModuleAttrValue]:
        """Return a shallow dictionary copy of the module state.

        Returns:
            Shallow copy of the state dictionary.

        """
        return self._state_dict().copy()

    def setdefault(self, key: str, default: ModuleAttrValue = None) -> ModuleAttrValue:
        """Return `self[key]` if present, otherwise set `self[key] = default` and return it.

        Args:
            key: Attribute key.
            default: Default value to store when `key` is absent.

        Returns:
            Existing or newly stored value.

        """
        return self._state_dict().setdefault(key, default)

    @classmethod
    def fromkeys(
        cls,
        iterable: Sequence[str],
        value: ModuleAttrValue = None,
    ) -> dict[str, ModuleAttrValue]:
        """Create a dictionary with keys from `iterable` and values set to `value`.

        Args:
            iterable: Sequence of key strings.
            value: Initial value for each key.

        Returns:
            Dictionary mapping each key to `value`.

        """
        return dict.fromkeys(iterable, value)

    @staticmethod
    def is_module(value: ModuleAttrValue) -> bool:
        """Return True if `value` is an instance of `Module`.

        Args:
            value: Value to test.

        Returns:
            True if `value` is a `Module`.

        """
        return isinstance(value, Module)

    @staticmethod
    def valid_child_filter(module: Module, key: str, value: ModuleAttrValue) -> bool:
        """Return True if `(key, value)` is a valid child attribute on `module`.

        Args:
            module: Parent `Module`.
            key: Attribute name.
            value: Attribute value.

        Returns:
            True if `key` does not start with `'_'` and `value` is a `Module`, `list`, `tuple`, or `dict`.

        """
        _ = module
        return not key.startswith("_") and isinstance(value, (Module, list, tuple, dict))

    @staticmethod
    def valid_parameter_filter(module: Module, key: str, value: ModuleAttrValue) -> bool:
        """Return True if `(key, value)` is a valid parameter or child container on `module`.

        Args:
            module: Parent `Module`.
            key: Attribute name.
            value: Attribute value.

        Returns:
            True if `key` is public and `value` is an `array` or child container.

        """
        _ = module
        return not key.startswith("_") and isinstance(value, (array, Module, list, tuple, dict))

    @staticmethod
    def trainable_parameter_filter(module: Module, key: str, value: ModuleAttrValue) -> bool:
        """Return True if `(key, value)` is a non-frozen trainable parameter on `module`.

        Args:
            module: Parent `Module`.
            key: Attribute name.
            value: Attribute value.

        Returns:
            True if `key` is a valid parameter and not frozen on `module`.

        """
        return (
            Module.valid_parameter_filter(module, key, value)
            and key not in module._get_frozen_set()
        )

    def filter_and_map(
        self,
        filter_fn: Callable[[Module, str, ModuleAttrValue], bool],
        map_fn: Callable[[ModuleAttrValue], ModuleAttrValue] | None = None,
        is_leaf_fn: Callable[[Module, str, ModuleAttrValue], bool] | None = None,
    ) -> dict[str, ModuleAttrValue]:
        """Recursively filter and optionally map attributes of this module.

        Args:
            filter_fn: Predicate `(module, key, value) -> bool` selecting attributes.
            map_fn: Optional transformation function applied to leaf values.
            is_leaf_fn: Optional predicate `(module, key, value) -> bool` identifying leaf values.

        Returns:
            Nested dictionary of filtered and mapped values.

        """
        result: dict[str, ModuleAttrValue] = {}
        for k, v in self._state_dict().items():
            if not filter_fn(self, k, v):
                continue
            is_leaf = (
                is_leaf_fn(self, k, v)
                if is_leaf_fn is not None
                else not isinstance(v, (Module, list, tuple, dict))
            )
            if not is_leaf and isinstance(v, Module):
                result[k] = v.filter_and_map(filter_fn, map_fn=map_fn, is_leaf_fn=is_leaf_fn)
            elif not is_leaf and isinstance(v, (list, tuple, dict)):
                result[k] = _filter_and_map_container(
                    self, v, filter_fn, map_fn=map_fn, is_leaf_fn=is_leaf_fn
                )
            else:
                result[k] = map_fn(v) if map_fn is not None else v
        return result

    def leaf_modules(self) -> dict[str, Module]:
        """Return a dictionary of leaf submodules (modules that have no child submodules).

        Returns:
            Dictionary mapping submodule names to leaf `Module` instances.

        """
        return {path: mod for path, mod in self.named_modules() if path and not mod._modules}

    def apply_to_modules(self, apply_fn: Callable[[str, Module], None]) -> Module:
        """Apply `apply_fn(path, module)` to this module and all its descendant submodules.

        Args:
            apply_fn: Callback receiving `(dotted_path, module)`.

        Returns:
            `self` for chaining.

        """
        for path, mod in self.named_modules():
            apply_fn(path, mod)
        return self

    def set_dtype(
        self,
        dtype: DTypeLike,
        predicate: Callable[[array], bool] | None = None,
    ) -> Module:
        """Cast parameters of this module matching `predicate` to `dtype`.

        Args:
            dtype: Target data type (`DType` or dtype string).
            predicate: Optional predicate filtering which `array` parameters to cast.

        Returns:
            `self` after casting parameters.

        """
        from zero_mlx.dtypes import floating, issubdtype

        pred = predicate if predicate is not None else (lambda x: issubdtype(x.dtype, floating))
        return self.apply(lambda x: x.astype(dtype) if pred(x) else x)

    def _state_dict(self) -> dict[str, ModuleAttrValue]:
        """Return the internal attribute dictionary of this module instance.

        Returns:
            Mutable mapping of attribute names to values on `self`.

        """
        if self._state is None:
            self._state = {"training": True}
            self._frozen = set()
        return self._state

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
            if k.startswith("_"):
                continue
            if isinstance(v, Module):
                mods[k] = v
            elif isinstance(v, (list, tuple)):
                for idx, item in enumerate(v):
                    if isinstance(item, Module):
                        mods[f"{k}.{idx}"] = item
            elif isinstance(v, dict):
                for sub_k, sub_item in v.items():
                    if isinstance(sub_item, Module):
                        mods[f"{k}.{sub_k}"] = sub_item
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
        _ = self._state_dict()
        return self._frozen if self._frozen is not None else set()

    def parameters(self) -> dict[str, PyTree]:
        """Return a nested dictionary of all parameters in this module and its submodules.

        Returns:
            Nested dictionary mapping attribute names to parameter `array` leaves.

        """
        params: dict[str, PyTree] = {}
        for k, v in self._state_dict().items():
            if k.startswith("_") or k == "training":
                continue
            if isinstance(v, array):
                params[k] = v
            elif isinstance(v, Module):
                params[k] = v.parameters()
            elif isinstance(v, (list, tuple, dict)):
                params[k] = _extract_container_params(v, trainable_only=False)
        return params

    def trainable_parameters(self) -> dict[str, PyTree]:
        """Return a nested dictionary of all non-frozen trainable parameters.

        Returns:
            Nested dictionary of trainable parameter `array` leaves.

        """
        frozen = self._get_frozen_set()
        params: dict[str, PyTree] = {}
        for k, v in self._state_dict().items():
            if k.startswith("_") or k == "training" or k in frozen:
                continue
            if isinstance(v, array):
                params[k] = v
            elif isinstance(v, Module):
                params[k] = v.trainable_parameters()
            elif isinstance(v, (list, tuple, dict)):
                extracted = _extract_container_params(v, trainable_only=True)
                if _has_any_array_leaf(extracted):
                    params[k] = extracted
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
        state = self._state_dict()
        if keys is not None:
            key_list = [keys] if isinstance(keys, str) else list(keys)
            for k in key_list:
                if (
                    strict
                    and k not in self._parameters
                    and k not in self._modules
                    and k not in state
                ):
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
        state = self._state_dict()
        if keys is not None:
            key_list = [keys] if isinstance(keys, str) else list(keys)
            for k in key_list:
                if (
                    strict
                    and k not in self._parameters
                    and k not in self._modules
                    and k not in state
                ):
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

        Raises:
            ValueError: If `strict` is True and a key does not exist on the module.

        """
        state = self._state_dict()
        mods = self._modules
        for k, v in parameters.items():
            if strict and k not in state and k not in mods:
                raise ValueError(
                    f"Module '{type(self).__name__}' has no parameter or submodule '{k}'"
                )
            curr = state.get(k)
            if isinstance(v, dict) and isinstance(curr, Module):
                curr.update(v, strict=strict)
            elif isinstance(v, dict) and isinstance(curr, dict):
                new_dict: dict[str, ModuleAttrValue] = dict(curr)
                for sub_k, sub_v in v.items():
                    sub_curr = new_dict.get(sub_k)
                    if isinstance(sub_curr, Module) and isinstance(sub_v, dict):
                        sub_curr.update(sub_v, strict=strict)
                    else:
                        new_dict[sub_k] = sub_v if isinstance(sub_v, array) else array(sub_v)
                state[k] = new_dict
            elif isinstance(v, (list, tuple)) and isinstance(curr, list):
                for idx, item_v in enumerate(v):
                    sub_item = curr[idx]
                    if isinstance(sub_item, Module) and isinstance(item_v, dict):
                        sub_item.update(item_v, strict=strict)
                    else:
                        curr[idx] = item_v if isinstance(item_v, array) else array(item_v)
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
        for k, v in list(state.items()):
            if k.startswith("_") or k == "training":
                continue
            if isinstance(v, array):
                state[k] = map_fn(v)
            elif isinstance(v, Module):
                v.apply(map_fn)
            elif isinstance(v, list):
                state[k] = [
                    item.apply(map_fn)
                    if isinstance(item, Module)
                    else (map_fn(item) if isinstance(item, array) else item)
                    for item in v
                ]
            elif isinstance(v, dict):
                state[k] = {
                    sub_k: (
                        sub_v.apply(map_fn)
                        if isinstance(sub_v, Module)
                        else (map_fn(sub_v) if isinstance(sub_v, array) else sub_v)
                    )
                    for sub_k, sub_v in v.items()
                }
        return self

    def train(self, mode: bool = True) -> Module:
        """Set the module and all its submodules into training (`mode=True`) or evaluation (`mode=False`) mode.

        Args:
            mode: True for training mode, False for evaluation mode.

        Returns:
            `self` for chaining.

        """
        self._state_dict()["training"] = mode
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

        Raises:
            NotImplementedError: If the module subclass does not implement `__call__`.

        """
        if isinstance(self, CallableModuleProtocol):
            return self(*args, **kwargs)
        raise NotImplementedError(f"Module '{type(self).__name__}' does not implement __call__")


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


__all__ = [
    "CallableModuleProtocol",
    "Identity",
    "Module",
    "ModuleAttr",
    "ModuleAttrValue",
]
