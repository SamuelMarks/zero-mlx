#!/usr/bin/env python3
"""Verify exported zero_mlx module namespaces against mlx_api_snapshot.json and mlx_v0.32.4.json."""

from __future__ import annotations

import json
import os
import sys
import types
from typing import Sequence


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


def verify_snapshot(
    snapshot_path: str = "mlx_api_snapshot.json",
    ecosystem_path: str | None = None,
) -> bool:
    """Verify that `zero_mlx` submodules export all symbols from the MLX snapshots.

    Args:
        snapshot_path: Path to `mlx_api_snapshot.json`.
        ecosystem_path: Optional path to `mlx_v0.32.4.json` from `ml-ecosystem-snapshots`.

    Returns:
        True if all snapshot symbols are exported on their corresponding modules, False otherwise.

    """
    if not os.path.exists(snapshot_path):
        print(f"Error: {snapshot_path} not found.")
        return False

    with open(snapshot_path, encoding="utf-8") as f:
        snapshot: dict[str, dict[str, dict[str, str]]] = json.load(f)

    sys.path.insert(0, os.path.abspath("src"))
    import zero_mlx
    import zero_mlx.fft as z_fft
    import zero_mlx.linalg as z_linalg
    import zero_mlx.mlx_random as z_random
    import zero_mlx.nn as z_nn
    import zero_mlx.nn.losses as z_losses
    import zero_mlx.optimizers as z_optim
    import zero_mlx.utils as z_utils

    zero_api: dict[str, set[str]] = {
        "mlx.core": get_public_api(zero_mlx),
        "mlx.core.fft": get_public_api(z_fft),
        "mlx.core.linalg": get_public_api(z_linalg),
        "mlx.core.random": get_public_api(z_random),
        "mlx.nn": get_public_api(z_nn),
        "mlx.nn.losses": get_public_api(z_losses),
        "mlx.optimizers": get_public_api(z_optim),
        "mlx.utils": get_public_api(z_utils),
    }

    missing_total: int = 0
    for mod_name, mlx_apis in snapshot.items():
        zero_mod_apis: set[str] = zero_api.get(mod_name, set())
        missing: list[str] = [api_name for api_name in mlx_apis if api_name not in zero_mod_apis]
        if missing:
            missing_total += len(missing)
            print(f"Missing in {mod_name}: {len(missing)} APIs ({missing[:5]})")

    candidates: list[str] = (
        [ecosystem_path]
        if ecosystem_path is not None
        else [
            os.path.expanduser("~/.cache/ml_ecosystem_snapshots/snapshots/mlx_v0.32.4.json"),
            os.path.expanduser("~/.cache/ml_ecosystem_snapshots/mlx_v0.32.4.json"),
        ]
    )
    for path in candidates:
        if os.path.exists(path):
            with open(path, encoding="utf-8") as f:
                eco_data = json.load(f)
            cats: dict[str, list[dict[str, str]]] = eco_data.get("categories", {})
            for items in cats.values():
                for item in items:
                    api_path = str(item.get("api_path", ""))
                    name = str(item.get("name", ""))
                    if "." in api_path:
                        mod_name = api_path.rsplit(".", 1)[0]
                        if mod_name in zero_api and name not in zero_api[mod_name]:
                            missing_total += 1
                            print(f"Missing v0.0.4 symbol in {mod_name}: {name}")

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
