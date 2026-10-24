#!/usr/bin/env python3
"""Check API parity of zero_mlx against mlx_api_snapshot.json and ml-ecosystem-snapshots v0.0.4."""

from __future__ import annotations

import ast
import json
import os
import sys
from typing import Sequence

NL: str = chr(10)
CR: str = chr(13)


def parse_defs(filepath: str, zero_api: set[str]) -> None:
    """Parse public function, class, and assignment names from a Python source file.

    Args:
        filepath: Path to the Python file to parse.
        zero_api: Set of discovered API symbol names to populate.

    """
    with open(filepath, encoding="utf-8") as f:
        tree: ast.Module = ast.parse(f.read(), filename=filepath)
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and not node.name.startswith("_"):
            zero_api.add(node.name)
        elif isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and not target.id.startswith("_"):
                    zero_api.add(target.id)
        elif (
            isinstance(node, ast.AnnAssign)
            and isinstance(node.target, ast.Name)
            and not node.target.id.startswith("_")
        ):
            zero_api.add(node.target.id)
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            for alias in node.names:
                bound_name = alias.asname if alias.asname is not None else alias.name.split(".")[-1]
                if not bound_name.startswith("_"):
                    zero_api.add(bound_name)


def _load_ecosystem_v004_apis(
    ecosystem_path: str | None = None,
) -> dict[str, tuple[str, dict[str, str]]]:
    """Load additional API definitions from ml-ecosystem-snapshots v0.0.4 if available.

    Args:
        ecosystem_path: Optional explicit path to `mlx_v0.32.4.json`.

    Returns:
        Mapping from symbol name to `(full_api_path, metadata_dict)`.

    """
    candidates: list[str] = (
        [ecosystem_path]
        if ecosystem_path is not None
        else [
            os.path.expanduser("~/.cache/ml_ecosystem_snapshots/snapshots/mlx_v0.32.4.json"),
            os.path.expanduser("~/.cache/ml_ecosystem_snapshots/mlx_v0.32.4.json"),
        ]
    )
    apis: dict[str, tuple[str, dict[str, str]]] = {}
    for path in candidates:
        if os.path.exists(path):
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
            cats: dict[str, list[dict[str, str]]] = data.get("categories", {})
            for items in cats.values():
                for item in items:
                    name = str(item.get("name", ""))
                    api_path = str(item.get("api_path", ""))
                    if name and api_path and name not in apis:
                        apis[name] = (
                            api_path,
                            {
                                "signature": "(...)",
                                "doc_str": str(
                                    item.get("docstring") or "No documentation available."
                                ),
                            },
                        )
    return apis


def check_parity(
    snapshot_path: str = "mlx_api_snapshot.json",
    src_dir: str = "src/zero_mlx",
    missing_api_path: str = "MLX_MISSING_API.md",
    ecosystem_path: str | None = None,
) -> bool:
    """Verify that all APIs in the snapshot exist in `src/zero_mlx` and update `MLX_MISSING_API.md`.

    Args:
        snapshot_path: Path to `mlx_api_snapshot.json`.
        src_dir: Root directory of `zero_mlx` source files.
        missing_api_path: Output markdown file tracking missing APIs.
        ecosystem_path: Optional path to `mlx_v0.32.4.json`.

    Returns:
        True if `MLX_MISSING_API.md` is unchanged and zero APIs are missing, False otherwise.

    """
    if not os.path.exists(snapshot_path):
        print(f"Error: {snapshot_path} not found.")
        return False

    with open(snapshot_path, encoding="utf-8") as f:
        snapshot: dict[str, dict[str, dict[str, str]]] = json.load(f)

    mlx_apis: dict[str, tuple[str, dict[str, str]]] = {}
    for mod_name, apis in snapshot.items():
        for name, info in apis.items():
            if name not in mlx_apis:
                mlx_apis[name] = (f"{mod_name}.{name}", info)

    for name, val in _load_ecosystem_v004_apis(ecosystem_path).items():
        if name not in mlx_apis:
            mlx_apis[name] = val

    zero_api: set[str] = set()
    for root, _, files in os.walk(src_dir):
        for file in files:
            if file.endswith(".py"):
                parse_defs(os.path.join(root, file), zero_api)

    missing_apis: list[tuple[str, str, dict[str, str]]] = []
    for name, (full_name, info) in mlx_apis.items():
        if name not in zero_api:
            missing_apis.append((name, full_name, info))

    missing_apis.sort(key=lambda x: x[1])

    lines: list[str] = [
        "# Missing MLX APIs Implementation Plan",
        "",
        "This document tracks the missing MLX APIs that need to be implemented in zero-mlx.",
        "",
        "| Status | Name | Signature | Docstring |",
        "|---|---|---|---|",
    ]

    for _, full_name, info in missing_apis:
        sig: str = info.get("signature", "")
        sig = sig.replace("|", r"\|").replace(NL, " ").replace(CR, " ")
        if not sig.startswith("("):
            sig = "(" + sig + ")"
        if len(sig) > 200:
            sig = sig[:197] + "..."

        doc_str: str = info.get("doc_str", "")
        doc_str = doc_str.replace("|", r"\|").replace(NL, " ").replace(CR, " ").replace("`", "")
        if len(doc_str) > 200:
            doc_str = doc_str[:197] + "..."

        lines.append(f"| [ ] | `{full_name}` | `{sig}` | {doc_str} |")

    md_content: str = NL.join(lines) + NL

    existing_content: str = ""
    if os.path.exists(missing_api_path):
        with open(missing_api_path, encoding="utf-8") as f:
            existing_content = f.read()

    with open(missing_api_path, "w", encoding="utf-8") as f:
        f.write(md_content)

    if existing_content != md_content or len(missing_apis) > 0:
        print(f"API parity check failed: {len(missing_apis)} missing APIs ({missing_api_path}).")
        return False

    print("API parity check passed.")
    return True


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entrypoint for checking API parity.

    Args:
        argv: Optional CLI arguments.

    Returns:
        Exit code 0 on pass, 1 on failure.

    """
    snap = argv[0] if argv else "mlx_api_snapshot.json"
    if not check_parity(snapshot_path=snap):
        sys.exit(1)
    return 0


if __name__ == "__main__":
    main()
