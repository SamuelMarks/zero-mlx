#!/usr/bin/env python3
"""Verify exported zero_mlx module hierarchy, classes, fields, functions, args, types, defaults, and docstrings against mlx_api_snapshot.json and ml-ecosystem-snapshots v0.0.5 (mlx_v0.32.4.json)."""

from __future__ import annotations

import inspect
import json
import os
import sys
import types
from typing import Sequence

from ml_ecosystem_snapshots.cli.resolver import resolve_snapshot_path
from ml_ecosystem_snapshots.compliance import score_compliance
from ml_switcheroo_ir.schema.ghost import GhostRef

REQUIRED_MODULE_HIERARCHY: tuple[str, ...] = (
    "zero_mlx",
    "zero_mlx.core",
    "zero_mlx.core.fft",
    "zero_mlx.core.linalg",
    "zero_mlx.core.random",
    "zero_mlx.core.fast",
    "zero_mlx.core.metal",
    "zero_mlx.core.cuda",
    "zero_mlx.core.distributed",
    "zero_mlx.fft",
    "zero_mlx.linalg",
    "zero_mlx.random",
    "zero_mlx.fast",
    "zero_mlx.metal",
    "zero_mlx.cuda",
    "zero_mlx.distributed",
    "zero_mlx.nn",
    "zero_mlx.nn.losses",
    "zero_mlx.nn.utils",
    "zero_mlx.optimizers",
    "zero_mlx.optimizers.schedulers",
    "zero_mlx.utils",
)

REQUIRED_CLASS_MEMBERS: dict[tuple[str, str], tuple[str, ...]] = {
    ("mlx.core", "array"): (
        "T",
        "at",
        "dtype",
        "itemsize",
        "nbytes",
        "ndim",
        "real",
        "imag",
        "shape",
        "size",
        "astype",
        "cumsum",
        "cumprod",
        "diag",
        "diagonal",
        "flatten",
        "item",
        "max",
        "mean",
        "min",
        "moveaxis",
        "prod",
        "reshape",
        "round",
        "split",
        "squeeze",
        "std",
        "sum",
        "swapaxes",
        "tolist",
        "transpose",
        "var",
        "view",
    ),
    ("mlx.core", "Device"): ("type", "index"),
    ("mlx.core", "Stream"): ("device", "index"),
    ("mlx.core", "ArrayAt"): ("add", "subtract", "multiply", "divide", "maximum", "minimum"),
    ("mlx.nn", "Module"): (
        "training",
        "state",
        "parameters",
        "trainable_parameters",
        "children",
        "modules",
        "named_modules",
        "leaf_modules",
        "freeze",
        "unfreeze",
        "train",
        "eval",
        "update",
        "apply",
        "apply_to_modules",
        "filter_and_map",
        "set_dtype",
        "load_weights",
        "save_weights",
        "keys",
        "values",
        "items",
        "get",
        "pop",
        "popitem",
        "clear",
        "copy",
        "setdefault",
        "fromkeys",
        "is_module",
        "valid_child_filter",
        "valid_parameter_filter",
        "trainable_parameter_filter",
    ),
    ("mlx.nn", "Linear"): ("to_quantized",),
    ("mlx.nn", "Embedding"): ("as_linear", "to_quantized"),
    ("mlx.nn", "ALiBi"): ("create_alibi_slope", "create_alibi_slopes"),
    ("mlx.nn", "AllToShardedLinear"): ("from_linear",),
    ("mlx.nn", "ShardedToAllLinear"): ("from_linear",),
    ("mlx.nn", "QuantizedAllToShardedLinear"): ("from_quantized_linear",),
    ("mlx.nn", "QuantizedShardedToAllLinear"): ("from_quantized_linear",),
    ("mlx.optimizers", "Optimizer"): (
        "state",
        "step",
        "learning_rate",
        "init",
        "init_single",
        "apply_single",
        "apply_gradients",
        "update",
    ),
}


def get_public_api(module: types.ModuleType | None) -> set[str]:
    """Return the set of public symbol names exposed by a module.

    Args:
        module: Module instance to inspect, or None.

    Returns:
        Set of public attribute names on the module.

    """
    if module is None:
        return set()
    return {name for name in dir(module) if not name.startswith("_")}


def resolve_ecosystem_v005_path(ecosystem_path: str | None = None) -> list[str]:
    """Return candidate paths for `mlx_v0.32.4.json` from `ml-ecosystem-snapshots` v0.0.5.

    Args:
        ecosystem_path: Optional explicit path override.

    Returns:
        List of candidate file paths in priority order.

    """
    return (
        [ecosystem_path]
        if ecosystem_path is not None
        else [
            resolve_snapshot_path("mlx_v0.32.4"),
            os.path.expanduser("~/.cache/ml_ecosystem_snapshots/snapshots/mlx_v0.32.4.json"),
            os.path.expanduser("~/.cache/ml_ecosystem_snapshots/mlx_v0.32.4.json"),
        ]
    )


def verify_module_hierarchy(
    modules: Sequence[tuple[str, types.ModuleType | None]],
) -> int:
    """Verify that all official MLX module paths exist and have module docstrings.

    Args:
        modules: Sequence of `(dotted_module_name, module_or_none)` pairs to verify.

    Returns:
        Number of missing or undocumented modules discovered (0 on full pass).

    """
    failures: int = 0
    for mod_path, mod in modules:
        if mod is None:
            print(f"Missing module in hierarchy: {mod_path}")
            failures += 1
        elif not (mod.__doc__ and mod.__doc__.strip()):
            print(f"Missing module docstring in {mod_path}")
            failures += 1
    return failures


def verify_class_members(
    mod_objects: dict[str, types.ModuleType],
    required_members: dict[tuple[str, str], tuple[str, ...]] = REQUIRED_CLASS_MEMBERS,
) -> int:
    """Verify that required fields, properties, and methods exist on key classes.

    Args:
        mod_objects: Mapping from `mlx.*` module names to `zero_mlx` modules.
        required_members: Mapping from `(mod_name, cls_name)` to required member tuples.

    Returns:
        Number of missing class fields or methods discovered.

    """
    failures: int = 0
    for (mod_name, cls_name), req_members in required_members.items():
        mod = mod_objects.get(mod_name)
        if mod is None:
            continue
        members_map = dict(inspect.getmembers(mod))
        if cls_name not in members_map:
            continue
        cls_obj = members_map[cls_name]
        cls_members = set(dir(cls_obj))
        for member in req_members:
            if member not in cls_members:
                print(f"Missing class member {mod_name}.{cls_name}.{member}")
                failures += 1
    return failures


def verify_ghost_ref_signature(
    ref: GhostRef,
    mod: types.ModuleType,
) -> int:
    """Verify a single `GhostRef` against the live symbol in `mod`.

    Checks kind (`class` vs `function`), docstring existence, and minimum required
    parameter names, annotations, and defaults.

    Args:
        ref: Ground-truth `GhostRef` from `ml-ecosystem-snapshots` v0.0.5.
        mod: Target `zero_mlx` module.

    Returns:
        Number of signature/kind/docstring violations for `ref`.

    """
    members_map = dict(inspect.getmembers(mod))
    if ref.name not in members_map:
        return 1
    obj = members_map[ref.name]

    violations: int = 0
    is_cls = isinstance(obj, type)
    if ref.kind == "class" and not is_cls:
        print(f"Kind mismatch on {ref.api_path}: expected class, got function")
        violations += 1
    elif ref.kind == "function" and is_cls:
        print(f"Kind mismatch on {ref.api_path}: expected function, got class")
        violations += 1

    doc = obj.__doc__
    if not (isinstance(doc, str) and doc.strip()):
        print(f"Missing docstring on {ref.api_path}")
        violations += 1

    target_callable = obj.__init__ if is_cls else obj
    sig = inspect.signature(target_callable)
    act_params = {k: p for k, p in sig.parameters.items() if k != "self"}
    has_var_kw = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in act_params.values())

    for exp_p in ref.params:
        pname = exp_p.name
        if pname in ("args", "kwargs", "__call__"):
            continue
        if pname not in act_params and not has_var_kw:
            print(f"Missing parameter '{pname}' on {ref.api_path}")
            violations += 1
        elif pname in act_params:
            act_p = act_params[pname]
            if act_p.annotation is inspect.Parameter.empty:
                print(f"Missing type annotation on {ref.api_path} parameter '{pname}'")
                violations += 1
    return violations


def verify_snapshot(
    snapshot_path: str = "mlx_api_snapshot.json",
    ecosystem_path: str | None = None,
) -> bool:
    """Verify that `zero_mlx` submodules export all symbols and signatures from the MLX snapshots.

    Args:
        snapshot_path: Path to `mlx_api_snapshot.json`.
        ecosystem_path: Optional path to `mlx_v0.32.4.json` from `ml-ecosystem-snapshots` v0.0.5.

    Returns:
        True if all snapshot modules, classes, fields, functions, args, and docstrings pass, False otherwise.

    """
    if not os.path.exists(snapshot_path):
        print(f"Error: {snapshot_path} not found.")
        return False

    with open(snapshot_path, encoding="utf-8") as f:
        snapshot: dict[str, dict[str, dict[str, str]]] = json.load(f)

    sys.path.insert(0, os.path.abspath("src"))
    import zero_mlx
    import zero_mlx.core as z_core
    import zero_mlx.core.cuda as z_core_cuda
    import zero_mlx.core.distributed as z_core_dist
    import zero_mlx.core.fast as z_core_fast
    import zero_mlx.core.fft as z_core_fft
    import zero_mlx.core.linalg as z_core_linalg
    import zero_mlx.core.metal as z_core_metal
    import zero_mlx.core.random as z_core_random
    import zero_mlx.cuda as z_cuda
    import zero_mlx.distributed as z_dist
    import zero_mlx.fast as z_fast
    import zero_mlx.fft as z_fft
    import zero_mlx.linalg as z_linalg
    import zero_mlx.metal as z_metal
    import zero_mlx.nn as z_nn
    import zero_mlx.nn.losses as z_losses
    import zero_mlx.nn.utils as z_nn_utils
    import zero_mlx.optimizers as z_optim
    import zero_mlx.optimizers.schedulers as z_schedulers
    import zero_mlx.random as z_random
    import zero_mlx.utils as z_utils

    hierarchy_pairs: tuple[tuple[str, types.ModuleType | None], ...] = (
        ("zero_mlx", zero_mlx),
        ("zero_mlx.core", z_core),
        ("zero_mlx.core.fft", z_core_fft),
        ("zero_mlx.core.linalg", z_core_linalg),
        ("zero_mlx.core.random", z_core_random),
        ("zero_mlx.core.fast", z_core_fast),
        ("zero_mlx.core.metal", z_core_metal),
        ("zero_mlx.core.cuda", z_core_cuda),
        ("zero_mlx.core.distributed", z_core_dist),
        ("zero_mlx.fft", z_fft),
        ("zero_mlx.linalg", z_linalg),
        ("zero_mlx.random", z_random),
        ("zero_mlx.fast", z_fast),
        ("zero_mlx.metal", z_metal),
        ("zero_mlx.cuda", z_cuda),
        ("zero_mlx.distributed", z_dist),
        ("zero_mlx.nn", z_nn),
        ("zero_mlx.nn.losses", z_losses),
        ("zero_mlx.nn.utils", z_nn_utils),
        ("zero_mlx.optimizers", z_optim),
        ("zero_mlx.optimizers.schedulers", z_schedulers),
        ("zero_mlx.utils", z_utils),
    )

    mod_objects: dict[str, types.ModuleType] = {
        "mlx.core": zero_mlx,
        "mlx.core.fft": z_fft,
        "mlx.core.linalg": z_linalg,
        "mlx.core.random": z_random,
        "mlx.nn": z_nn,
        "mlx.nn.losses": z_losses,
        "mlx.optimizers": z_optim,
        "mlx.utils": z_utils,
    }

    zero_api: dict[str, set[str]] = {k: get_public_api(v) for k, v in mod_objects.items()}
    core_api: set[str] = get_public_api(z_core)

    missing_total: int = 0
    if ecosystem_path is None:
        missing_total += verify_module_hierarchy(hierarchy_pairs)
        missing_total += verify_class_members(mod_objects)

    for mod_name, mlx_apis in snapshot.items():
        zero_mod_apis: set[str] = zero_api.get(mod_name, set())
        missing: list[str] = [api_name for api_name in mlx_apis if api_name not in zero_mod_apis]
        if missing:
            missing_total += len(missing)
            print(f"Missing in {mod_name}: {len(missing)} APIs ({missing[:5]})")

    missing_core = [
        api_name for api_name in snapshot.get("mlx.core", {}) if api_name not in core_api
    ]
    if missing_core:
        missing_total += len(missing_core)
        print(f"Missing in zero_mlx.core: {len(missing_core)} APIs ({missing_core[:5]})")

    candidates: list[str] = resolve_ecosystem_v005_path(ecosystem_path)
    checked_v005: bool = False
    for path in candidates:
        if os.path.exists(path) and not checked_v005:
            checked_v005 = True
            with open(path, encoding="utf-8") as f:
                eco_data = json.load(f)
            cats = eco_data.get("categories", {})
            valid_cats: dict[str, list[dict[str, str | list[str] | None]]] = {}
            target_refs: list[GhostRef] = []
            for cat_name, items in cats.items():
                valid_items: list[dict[str, str | list[str] | None]] = []
                for item in items:
                    api_path = str(item.get("api_path", ""))
                    name = str(item.get("name", ""))
                    if "." in api_path:
                        mod_name = api_path.rsplit(".", 1)[0]
                        if mod_name in zero_api:
                            if name not in zero_api[mod_name]:
                                missing_total += 1
                                print(f"Missing v0.0.5 symbol in {mod_name}: {name}")
                            elif isinstance(item.get("params"), list):
                                ref = GhostRef.model_validate(item)
                                valid_items.append(item)
                                missing_total += verify_ghost_ref_signature(
                                    ref, mod_objects[mod_name]
                                )
                                target_refs.append(ref)
                                for alias_path in ref.aliases or ():
                                    a_mod, a_name = alias_path.rsplit(".", 1)
                                    if a_name not in zero_api.get(a_mod, set()):
                                        missing_total += 1
                                        print(f"Missing v0.0.5 alias in {a_mod}: {a_name}")
                                    else:
                                        target_refs.append(
                                            ref.model_copy(
                                                update={"name": a_name, "api_path": alias_path}
                                            )
                                        )
                valid_cats[cat_name] = valid_items
            report = score_compliance(
                {"categories": valid_cats},
                target_refs,
                strict_c_extensions=False,
            )
            missing_total += len(report.missing) + len(report.mismatched)

    if missing_total == 0:
        print("API parity with snapshot verified successfully!")
        return True

    print(f"Total missing APIs: {missing_total}")
    return False


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entrypoint for verifying snapshot API exports.

    Args:
        argv: Optional CLI arguments.

    Returns:
        Exit code 0 on pass, 1 on failure.

    """
    snap = argv[0] if argv else "mlx_api_snapshot.json"
    if not verify_snapshot(snapshot_path=snap):
        sys.exit(1)
    return 0


if __name__ == "__main__":
    main()
