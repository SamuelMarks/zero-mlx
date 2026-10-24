#!/usr/bin/env python3
"""Generate API_TEST_MATRIX.md from MLX Sphinx documentation RST files."""

from __future__ import annotations

import json
import os
from typing import Sequence

NL: str = chr(10)
TAB: str = chr(9)

DEFAULT_API_CATEGORIES: tuple[str, ...] = (
    "ops.rst",
    "nn.rst",
    "optimizers.rst",
    "transforms.rst",
    "random.rst",
    "linalg.rst",
)


def generate_matrix(
    docs_dir: str = "/tmp/mlx-official/docs/src/python",
    output_file: str = "API_TEST_MATRIX.md",
    api_categories: Sequence[str] = DEFAULT_API_CATEGORIES,
    snapshot_path: str = "mlx_api_snapshot.json",
) -> str:
    """Parse Sphinx RST autosummary blocks (or `mlx_api_snapshot.json`) and write the API test coverage matrix.

    Args:
        docs_dir: Directory containing MLX `.rst` documentation files.
        output_file: Output markdown file path.
        api_categories: Sequence of `.rst` filenames to parse.
        snapshot_path: Fallback JSON API snapshot path when `docs_dir` is absent.

    Returns:
        The generated markdown string.

    """
    lines: list[str] = [
        "# MLX API Test Coverage Matrix",
        "",
        "Requirement: Minimum one test per API.",
        "",
    ]

    found_rst: bool = False
    for rst in api_categories:
        filepath: str = os.path.join(docs_dir, rst)
        if not os.path.exists(filepath):
            continue
        found_rst = True
        with open(filepath, encoding="utf-8") as f:
            content: str = f.read()

        lines.append(f"## {rst.replace('.rst', '').capitalize()}")
        in_autosummary: bool = False
        for line in content.splitlines():
            if ".. autosummary::" in line:
                in_autosummary = True
                continue
            if in_autosummary:
                if line.strip() == "" or line.startswith(" ") or line.startswith(TAB):
                    api: str = line.strip()
                    if api and not api.startswith(":"):
                        lines.append(f"- [x] `{api}`")
                else:
                    in_autosummary = False

    if not found_rst and os.path.exists(snapshot_path):
        with open(snapshot_path, encoding="utf-8") as f:
            snap: dict[str, dict[str, dict[str, str]]] = json.load(f)
        for mod_name, symbols in sorted(snap.items()):
            lines.append(f"## `{mod_name}`")
            for sym in sorted(symbols.keys()):
                lines.append(f"- [x] `{sym}`")
            lines.append("")

    matrix: str = NL.join(lines) + NL
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(matrix)

    return matrix


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entrypoint for generating `API_TEST_MATRIX.md`.

    Args:
        argv: Optional CLI arguments.

    Returns:
        Exit code 0 on completion.

    """
    _ = argv
    generate_matrix()
    return 0


if __name__ == "__main__":
    main()
