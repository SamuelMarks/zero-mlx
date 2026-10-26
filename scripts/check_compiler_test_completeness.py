#!/usr/bin/env python3
"""Check that all ml-switcheroo-compiler operations have parity tests in zero_mlx."""

from __future__ import annotations

import ast
import os
import sys
from typing import Sequence

NL: str = chr(10)


def check_completeness(
    tests_dir: str = "tests",
    tracking_file: str = "COMPILER_TEST_PARITY.md",
) -> bool:
    """Verify compiler operation test completeness and update `COMPILER_TEST_PARITY.md`.

    Args:
        tests_dir: Directory containing `test_ops_parity_*.py` files.
        tracking_file: Markdown file tracking parity test status per operation.

    Returns:
        True if `COMPILER_TEST_PARITY.md` is up to date, False otherwise.

    """
    sys.path.insert(0, os.path.abspath("src"))
    import zero_mlx as _zmlx
    from ml_switcheroo_compiler.ops.registry import _OP_REGISTRY

    _ = _zmlx
    all_ops: set[str] = set(_OP_REGISTRY.keys())

    test_files: list[str] = []
    if os.path.exists(tests_dir):
        test_files = [
            os.path.join(tests_dir, f)
            for f in os.listdir(tests_dir)
            if f.startswith("test_ops_parity_") and f.endswith(".py")
        ]

    implemented_tests: set[str] = set()
    for test_file in test_files:
        with open(test_file, encoding="utf-8") as f:
            tree: ast.Module = ast.parse(f.read(), filename=test_file)
            for node in ast.walk(tree):
                if (
                    isinstance(node, ast.FunctionDef)
                    and node.name.startswith("test_")
                    and node.name.endswith("_parity")
                ):
                    op_name: str = node.name[5:-7]
                    implemented_tests.add(op_name)

    lines: list[str] = [
        "# Compiler Operations Test Parity",
        "",
        "This tracks which ml-switcheroo-compiler operations have a dedicated parity test against MLX in `tests/test_ops_parity_*.py`.",
        "",
        "| Status | Operation |",
        "|---|---|",
    ]

    for op in sorted(all_ops):
        status: str = "[x]" if op in implemented_tests else "[ ]"
        lines.append(f"| {status} | `{op}` |")

    md_content: str = NL.join(lines) + NL

    existing_content: str = ""
    if os.path.exists(tracking_file):
        with open(tracking_file, encoding="utf-8") as f:
            existing_content = f.read()

    with open(tracking_file, "w", encoding="utf-8") as f:
        f.write(md_content)

    if existing_content != md_content:
        print(
            f"Compiler test completeness check failed: {tracking_file} was updated. Please commit the changes."
        )
        return False

    print("Compiler test completeness check passed.")
    return True


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entrypoint for compiler test completeness check.

    Args:
        argv: Optional CLI arguments.

    Returns:
        Exit code 0 on pass, 1 on failure.

    """
    tracking = argv[0] if argv else "COMPILER_TEST_PARITY.md"
    if not check_completeness(tracking_file=tracking):
        sys.exit(1)
    return 0


if __name__ == "__main__":
    main()
