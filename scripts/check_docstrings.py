#!/usr/bin/env python3
"""AST-based docstring completeness verifier for modules, classes, functions, and arguments."""

from __future__ import annotations

import argparse
import ast
import os
import re
import sys
from typing import Sequence

DEFAULT_TARGET_DIRS: tuple[str, ...] = ("src/zero_mlx", "scripts", "tests")
PLACEHOLDER_DOCSTRINGS: frozenset[str] = frozenset(
    {
        "Module docstring.",
        "Package docstring.",
        "Docstring.",
        "Call.",
        "Execute implementation detail.",
    }
)


def _extract_documented_args(docstring: str) -> set[str]:
    """Parse documented parameter names from the Args section of a Google-style docstring.

    Args:
        docstring: Raw docstring text of a function or method.

    Returns:
        Set of parameter names documented under `Args:`.

    """
    documented: set[str] = set()
    lines: list[str] = docstring.splitlines()
    in_args: bool = False
    args_indent: int | None = None

    for line in lines:
        stripped: str = line.strip()
        if stripped in ("Args:", "Arguments:", "Parameters:"):
            in_args = True
            args_indent = len(line) - len(line.lstrip())
            continue
        if in_args:
            if not stripped:
                continue
            curr_indent: int = len(line) - len(line.lstrip())
            if args_indent is not None and curr_indent <= args_indent and stripped.endswith(":"):
                in_args = False
                continue
            match = re.match(r"^\*{0,2}([a-zA-Z_][a-zA-Z0-9_]*)\s*(?:\([^)]*\))?\s*:", stripped)
            if match:
                documented.add(match.group(1))
    return documented


def _check_function_docstring(
    filepath: str, node: ast.FunctionDef | ast.AsyncFunctionDef
) -> list[str]:
    """Verify that a function or method has a non-placeholder docstring and documents all parameters.

    Args:
        filepath: Path of the file being checked.
        node: Function or AsyncFunction AST node.

    Returns:
        List of docstring violation messages.

    """
    violations: list[str] = []
    doc: str | None = ast.get_docstring(node)
    if not doc or not doc.strip():
        violations.append(f"{filepath}:{node.lineno}: missing docstring for function '{node.name}'")
        return violations

    first_line: str = doc.strip().splitlines()[0].strip()
    if first_line in PLACEHOLDER_DOCSTRINGS:
        violations.append(
            f"{filepath}:{node.lineno}: placeholder docstring '{first_line}' in '{node.name}'"
        )

    params: list[str] = []
    for arg in [*node.args.posonlyargs, *node.args.args, *node.args.kwonlyargs]:
        if arg.arg not in ("self", "cls"):
            params.append(arg.arg)
    if node.args.vararg is not None:
        params.append(node.args.vararg.arg)
    if node.args.kwarg is not None:
        params.append(node.args.kwarg.arg)

    if params:
        documented: set[str] = _extract_documented_args(doc)
        for p in params:
            if p not in documented:
                violations.append(
                    f"{filepath}:{node.lineno}: undocumented parameter '{p}' in '{node.name}'"
                )

    return violations


def check_file_docstrings(filepath: str) -> list[str]:
    """Check a single Python file for module, class, function, and parameter docstrings.

    Args:
        filepath: Path to the Python file to inspect.

    Returns:
        List of violation messages discovered in the file.

    """
    with open(filepath, encoding="utf-8") as f:
        source: str = f.read()

    try:
        tree: ast.Module = ast.parse(source, filename=filepath)
    except SyntaxError as exc:
        return [f"{filepath}:{exc.lineno}: SyntaxError: {exc.msg}"]

    violations: list[str] = []
    mod_doc: str | None = ast.get_docstring(tree)
    if not mod_doc or not mod_doc.strip():
        violations.append(f"{filepath}:1: missing module docstring")
    elif mod_doc.strip() in PLACEHOLDER_DOCSTRINGS:
        violations.append(f"{filepath}:1: placeholder module docstring '{mod_doc.strip()}'")

    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            cls_doc: str | None = ast.get_docstring(node)
            if not cls_doc or not cls_doc.strip():
                violations.append(
                    f"{filepath}:{node.lineno}: missing docstring for class '{node.name}'"
                )
            elif cls_doc.strip() in PLACEHOLDER_DOCSTRINGS:
                violations.append(
                    f"{filepath}:{node.lineno}: placeholder docstring in class '{node.name}'"
                )
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            violations.extend(_check_function_docstring(filepath, node))

    return violations


def scan_path(target_path: str) -> list[str]:
    """Scan a file or directory recursively for docstring violations.

    Args:
        target_path: File or directory path to scan.

    Returns:
        List of docstring violation messages.

    """
    if not os.path.exists(target_path):
        return []
    if os.path.isfile(target_path):
        return check_file_docstrings(target_path)

    violations: list[str] = []
    for root, _, files in sorted(os.walk(target_path)):
        for fname in sorted(files):
            if fname.endswith(".py"):
                violations.extend(check_file_docstrings(os.path.join(root, fname)))
    return violations


def main(argv: Sequence[str] | None = None) -> int:
    """Run the AST docstring completeness verifier CLI.

    Args:
        argv: Optional CLI arguments. Defaults to `sys.argv[1:]`.

    Returns:
        Exit code 0 if 100% compliant, 1 otherwise.

    """
    parser = argparse.ArgumentParser(
        description="Verify 100% module, class, function, and parameter docstring coverage."
    )
    parser.add_argument(
        "paths",
        nargs="*",
        default=list(DEFAULT_TARGET_DIRS),
        help="Directories or files to check.",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)

    all_violations: list[str] = []
    for path in args.paths:
        all_violations.extend(scan_path(str(path)))

    if all_violations:
        for v in all_violations:
            print(v, file=sys.stderr)
        print(
            f"FAILED: Found {len(all_violations)} docstring violations.",
            file=sys.stderr,
        )
        sys.exit(1)

    print("100% docstring completeness check passed.")
    return 0


if __name__ == "__main__":
    main()
