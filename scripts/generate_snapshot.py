#!/usr/bin/env python3
"""Snapshot generator grounded in ml-switcheroo-ir GhostRef and ml-ecosystem-snapshots."""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Sequence

from ml_ecosystem_snapshots.frameworks.mlx import collect_api
from ml_switcheroo_ir.schema.ghost import GhostRef, SemanticTier


def collect_ghost_snapshot(
    include_nonpublic: bool = False,
) -> dict[str, list[dict[str, str | bool | list[str] | None]]]:
    """Collect live MLX API definitions across all semantic tiers using GhostRef.

    Args:
        include_nonpublic: Whether to include private/non-public API symbols.

    Returns:
        Dictionary mapping semantic tier category names to serialized GhostRef dictionaries.

    """
    categories: dict[str, list[dict[str, str | bool | list[str] | None]]] = {}
    for tier in (
        SemanticTier.ACTIVATION,
        SemanticTier.ARRAY_API,
        SemanticTier.LAYER,
        SemanticTier.LOSS,
        SemanticTier.OPTIMIZER,
    ):
        refs: list[GhostRef] = collect_api(tier, include_nonpublic=include_nonpublic)
        categories[tier.value] = [
            {
                "name": r.name,
                "api_path": r.api_path,
                "kind": r.kind,
                "is_public": r.is_public,
                "docstring": r.docstring,
            }
            for r in refs
        ]
    return categories


def load_cached_ecosystem_snapshot(
    cache_path: str | None = None,
) -> dict[str, dict[str, dict[str, str]]]:
    """Load the cached ml-ecosystem-snapshots v0.0.4 MLX JSON and normalize to module map.

    Args:
        cache_path: Optional path to the `mlx_v0.32.4.json` file.

    Returns:
        Normalized module dictionary mapping `mlx.*` modules to symbol metadata.

    """
    candidates: list[str] = (
        [cache_path]
        if cache_path is not None
        else [
            os.path.expanduser("~/.cache/ml_ecosystem_snapshots/snapshots/mlx_v0.32.4.json"),
            os.path.expanduser("~/.cache/ml_ecosystem_snapshots/mlx_v0.32.4.json"),
        ]
    )

    result: dict[str, dict[str, dict[str, str]]] = {}
    for path in candidates:
        if os.path.exists(path):
            with open(path, encoding="utf-8") as f:
                raw_data = json.load(f)
            cats: dict[str, list[dict[str, str | list[dict[str, str]] | None]]] = raw_data.get(
                "categories", {}
            )
            for items in cats.values():
                for item in items:
                    api_path = str(item.get("api_path", ""))
                    name = str(item.get("name", ""))
                    if "." in api_path:
                        mod_name = api_path.rsplit(".", 1)[0]
                        params_list = item.get("params", [])
                        param_names: list[str] = []
                        if isinstance(params_list, list):
                            for p in params_list:
                                if isinstance(p, dict) and "name" in p:
                                    param_names.append(str(p["name"]))
                        sig = "(" + ", ".join(param_names) + ")"
                        doc_str = str(item.get("docstring") or "No documentation available.")
                        first_line = (
                            doc_str.strip().splitlines()[0]
                            if doc_str.strip()
                            else "No documentation available."
                        )
                        if mod_name not in result:
                            result[mod_name] = {}
                        result[mod_name][name] = {
                            "type": str(item.get("kind", "function")),
                            "signature": sig,
                            "doc_str": first_line,
                        }
    return result


def generate_snapshot_file(
    output_path: str = "mlx_api_snapshot.json",
    cache_path: str | None = None,
) -> dict[str, dict[str, dict[str, str]]]:
    """Generate or refresh the MLX API snapshot JSON file.

    Args:
        output_path: Destination file path for the snapshot JSON.
        cache_path: Optional path to the cached `mlx_v0.32.4.json` snapshot.

    Returns:
        The generated snapshot dictionary.

    """
    snapshot = load_cached_ecosystem_snapshot(cache_path=cache_path)
    if snapshot:
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(snapshot, f, indent=2)
    return snapshot


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entrypoint for generating the MLX API snapshot.

    Args:
        argv: Optional CLI arguments. Defaults to `sys.argv[1:]`.

    Returns:
        Exit code 0 on success.

    """
    parser = argparse.ArgumentParser(
        description="Generate MLX API snapshot using ml-ecosystem-snapshots and ml-switcheroo-ir."
    )
    parser.add_argument(
        "--output",
        type=str,
        default="mlx_api_snapshot.json",
        help="Output JSON file path.",
    )
    parser.add_argument(
        "--cache-path",
        type=str,
        default=None,
        help="Optional path to mlx_v0.32.4.json.",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)
    generate_snapshot_file(output_path=str(args.output), cache_path=args.cache_path)
    print(f"Snapshot generated: {args.output}")
    return 0


if __name__ == "__main__":
    main()
