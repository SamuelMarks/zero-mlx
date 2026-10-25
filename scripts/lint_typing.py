#!/usr/bin/env python3
"""AST-based linter banning Any, object, typing.cast, reflection, dunder hacks, and pragma: no cover."""

from __future__ import annotations

import argparse
import ast
import os
import sys
from typing import Sequence

DEFAULT_TARGET_DIRS: tuple[str, ...] = ("src/zero_mlx", "scripts", "tests")
BANNED_REFLECTION_FUNCS: frozenset[str] = frozenset(
    {"getattr", "setattr", "hasattr", "delattr", "__import__", "cast", "vars", "__setattr__"}
)
BANNED_TYPE_NAMES: frozenset[str] = frozenset({"Any", "object", "AnyType"})


def _check_annotation(node: ast.AST | None, filepath: str) -> list[str]:
    """Inspect a type annotation AST subtree for prohibited Any or object usages.

    Args:
        node: Annotation AST node to inspect.
        filepath: Path of the file being scanned.

    Returns:
        List of violation messages discovered in the annotation.

    """
    if node is None:
        return []
    violations: list[str] = []
    for sub in ast.walk(node):
        if isinstance(sub, ast.Name) and sub.id in BANNED_TYPE_NAMES:
            violations.append(f"{filepath}:{sub.lineno}: prohibited type annotation '{sub.id}'")
        elif isinstance(sub, ast.Attribute) and sub.attr == "Any":
            violations.append(f"{filepath}:{sub.lineno}: prohibited type annotation 'typing.Any'")
    return violations


def _check_function_node(filepath: str, node: ast.FunctionDef | ast.AsyncFunctionDef) -> list[str]:
    """Check function parameter and return annotations for prohibited types.

    Args:
        filepath: Path of the file being scanned.
        node: Function or AsyncFunction AST node.

    Returns:
        List of violation messages on the function signature.

    """
    violations: list[str] = []
    all_args: list[ast.arg] = [
        *node.args.posonlyargs,
        *node.args.args,
        *node.args.kwonlyargs,
    ]
    if node.args.vararg is not None:
        all_args.append(node.args.vararg)
    if node.args.kwarg is not None:
        all_args.append(node.args.kwarg)

    for arg in all_args:
        if arg.arg not in ("self", "cls") and arg.annotation is None:
            violations.append(
                f"{filepath}:{arg.lineno}: missing type annotation for parameter '{arg.arg}' in '{node.name}'"
            )
        if (
            node.name in ("__eq__", "__ne__")
            and arg.arg == "other"
            and isinstance(arg.annotation, ast.Name)
            and arg.annotation.id == "object"
        ):
            continue
        violations.extend(_check_annotation(arg.annotation, filepath))

    if node.returns is None:
        violations.append(
            f"{filepath}:{node.lineno}: missing return type annotation in '{node.name}'"
        )
    else:
        violations.extend(_check_annotation(node.returns, filepath))

    return violations


def _check_ast_node(filepath: str, node: ast.AST) -> list[str]:
    """Check a single AST node for typing, reflection, or dunder violations.

    Args:
        filepath: Path of the file being scanned.
        node: AST node to inspect.

    Returns:
        List of violation messages for the node.

    """
    violations: list[str] = []
    if isinstance(node, ast.ImportFrom):
        for alias in node.names:
            if alias.name in ("Any", "cast"):
                violations.append(f"{filepath}:{node.lineno}: prohibited import '{alias.name}'")
    elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
        violations.extend(_check_function_node(filepath, node))
    elif isinstance(node, ast.AnnAssign):
        violations.extend(_check_annotation(node.annotation, filepath))
    elif isinstance(node, ast.Assign):
        for target in node.targets:
            if isinstance(target, ast.Name) and target.id in BANNED_TYPE_NAMES:
                violations.append(
                    f"{filepath}:{node.lineno}: prohibited type alias assignment '{target.id}'"
                )
    elif isinstance(node, ast.Call):
        if isinstance(node.func, ast.Name) and node.func.id in BANNED_REFLECTION_FUNCS:
            violations.append(
                f"{filepath}:{node.lineno}: prohibited reflection/cast call '{node.func.id}()'"
            )
        elif (
            isinstance(node.func, ast.Attribute)
            and node.func.attr in BANNED_REFLECTION_FUNCS
            and not (
                node.func.attr == "cast"
                and isinstance(node.func.value, ast.Name)
                and node.func.value.id in ("sops", "mops", "ops")
            )
        ):
            violations.append(
                f"{filepath}:{node.lineno}: prohibited reflection/cast call '.{node.func.attr}()'"
            )
    elif isinstance(node, ast.Attribute) and node.attr == "__dict__":
        violations.append(f"{filepath}:{node.lineno}: prohibited dunder access '.__dict__'")
    return violations


def scan_file(filepath: str) -> list[str]:
    """Scan a single Python source file for typing, reflection, and pragma violations.

    Args:
        filepath: Path to the Python file to scan.

    Returns:
        List of violation messages discovered in the file.

    """
    with open(filepath, encoding="utf-8") as f:
        source: str = f.read()

    violations: list[str] = []
    for idx, line in enumerate(source.splitlines(), start=1):
        if "pragma: no cover" in line and not filepath.endswith("lint_typing.py"):
            violations.append(f"{filepath}:{idx}: prohibited '# pragma: no cover'")
        if ("# noqa" in line or "ruff: noqa" in line) and not filepath.endswith("lint_typing.py"):
            violations.append(f"{filepath}:{idx}: prohibited '# noqa'")

    try:
        tree: ast.Module = ast.parse(source, filename=filepath)
    except SyntaxError as exc:
        violations.append(f"{filepath}:{exc.lineno}: SyntaxError: {exc.msg}")
        return violations

    for node in ast.walk(tree):
        violations.extend(_check_ast_node(filepath, node))

    return violations


def scan_directory(target_dir: str) -> list[str]:
    """Recursively scan a directory for typing and reflection violations.

    Args:
        target_dir: Root directory path to scan.

    Returns:
        Aggregated list of violation messages across all Python files.

    """
    violations: list[str] = []
    if not os.path.exists(target_dir):
        return violations
    if os.path.isfile(target_dir):
        return scan_file(target_dir)

    for root, _, files in sorted(os.walk(target_dir)):
        for fname in sorted(files):
            if fname.endswith(".py"):
                violations.extend(scan_file(os.path.join(root, fname)))
    return violations


def main(argv: Sequence[str] | None = None) -> int:
    """Execute the strict AST typing and reflection linter CLI.

    Args:
        argv: Optional CLI arguments. Defaults to `sys.argv[1:]`.

    Returns:
        Exit code 0 if zero violations are found, 1 otherwise.

    """
    parser = argparse.ArgumentParser(
        description="Lint Python ASTs for Any, object, cast, getattr/setattr, and pragma: no cover."
    )
    parser.add_argument(
        "paths",
        nargs="*",
        default=list(DEFAULT_TARGET_DIRS),
        help="Directories or files to scan.",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)

    all_violations: list[str] = []
    for path in args.paths:
        all_violations.extend(scan_directory(str(path)))

    if all_violations:
        for v in all_violations:
            print(v, file=sys.stderr)
        print(
            f"FAILED: Found {len(all_violations)} typing/reflection violations.",
            file=sys.stderr,
        )
        sys.exit(1)

    print("Strict typing & reflection check passed.")
    return 0


if __name__ == "__main__":
    main()
