#!/usr/bin/env python3
"""AST-based verification script for enforcing 3rd-party import boundaries."""

from __future__ import annotations

import ast
import importlib.util
import sys
from typing import Sequence

ALLOWED_3RD_PARTY: set[str] = {
    "pydantic",
    "cdd",
    "cdd_python",
    "ml_switcheroo_ir",
    "ml_switcheroo_compiler",
    "ml_ecosystem_snapshots",
    "zero_keras",
    "zero_mlx",
}

FALLBACK_STDLIB: set[str] = {
    "__future__",
    "abc",
    "argparse",
    "ast",
    "asyncio",
    "base64",
    "builtins",
    "collections",
    "concurrent",
    "contextlib",
    "copy",
    "ctypes",
    "dataclasses",
    "datetime",
    "decimal",
    "enum",
    "functools",
    "glob",
    "hashlib",
    "importlib",
    "inspect",
    "io",
    "itertools",
    "json",
    "logging",
    "math",
    "multiprocessing",
    "numbers",
    "operator",
    "os",
    "pathlib",
    "pickle",
    "pprint",
    "random",
    "re",
    "shutil",
    "socket",
    "sqlite3",
    "string",
    "struct",
    "subprocess",
    "sys",
    "tempfile",
    "threading",
    "time",
    "traceback",
    "types",
    "typing",
    "unittest",
    "urllib",
    "uuid",
    "warnings",
    "weakref",
    "zipfile",
    "zoneinfo",
}


def is_stdlib(module_name: str) -> bool:
    """Check whether a module name belongs to the Python standard library.

    Args:
        module_name: Module name or dotted path to check.

    Returns:
        True if the module is a standard library or built-in module, False otherwise.

    """
    if not module_name:
        return True

    base_name: str = module_name.split(".")[0]

    if base_name in sys.builtin_module_names or base_name in FALLBACK_STDLIB:
        return True

    spec = importlib.util.find_spec(base_name)
    if spec is None or spec.origin is None:
        return False

    if "site-packages" in spec.origin or "dist-packages" in spec.origin:
        return False

    return spec.origin.startswith(sys.base_prefix) or spec.origin.startswith(sys.prefix)


def _extract_dynamic_arg(call_node: ast.Call) -> ast.expr | None:
    """Extract the target module expression from a dynamic import call node.

    Args:
        call_node: The AST Call node representing a dynamic import.

    Returns:
        The AST expression for the module argument if present, else None.

    """
    if call_node.args:
        return call_node.args[0]
    for kw in call_node.keywords:
        if kw.arg == "name":
            return kw.value
    return None


def _is_dynamic_import_call(func: ast.expr) -> bool:
    """Determine if a call target expression is a dynamic import function.

    Args:
        func: The function expression of an AST Call node.

    Returns:
        True if the call invokes `__import__`, `import_module`, or `sys.modules.get`.

    """
    if isinstance(func, ast.Name) and func.id in ("__import__", "import_module"):
        return True
    if isinstance(func, ast.Attribute):
        if func.attr in ("__import__", "import_module"):
            return True
        if (
            isinstance(func.value, ast.Attribute)
            and func.value.attr == "modules"
            and isinstance(func.value.value, ast.Name)
            and func.value.value.id == "sys"
        ):
            return True
    return False


def _check_module_allowed(module_name: str, filepath: str, lineno: int, errors: list[str]) -> None:
    """Verify a single imported module name against the allowed set and stdlib.

    Args:
        module_name: The imported module name.
        filepath: Path of the file being checked.
        lineno: Line number of the import statement.
        errors: Mutable list to append error messages to.

    """
    base_name: str = module_name.split(".")[0]
    if "backends.numpy" in module_name:
        errors.append(f"{filepath}:{lineno}: Forbidden 3rd-party backend import '{module_name}'")
        return
    if base_name not in ALLOWED_3RD_PARTY and not is_stdlib(base_name):
        errors.append(
            f"{filepath}:{lineno}: Forbidden 3rd-party import '{base_name}' (from '{module_name}')"
        )


def check_file(filepath: str) -> bool:
    """Check a single Python file for forbidden third-party imports.

    Args:
        filepath: Path to the Python source file to inspect.

    Returns:
        True if the file contains only allowed imports, False otherwise.

    """
    with open(filepath, encoding="utf-8") as f:
        try:
            tree: ast.Module = ast.parse(f.read(), filename=filepath)
        except SyntaxError:
            return False

    errors: list[str] = []

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                _check_module_allowed(alias.name, filepath, node.lineno, errors)
        elif isinstance(node, ast.ImportFrom):
            if node.level == 0 and node.module:
                _check_module_allowed(node.module, filepath, node.lineno, errors)
        elif isinstance(node, ast.Call) and _is_dynamic_import_call(node.func):
            module_val = _extract_dynamic_arg(node)
            if module_val is not None:
                if isinstance(module_val, ast.Constant) and isinstance(module_val.value, str):
                    _check_module_allowed(module_val.value, filepath, node.lineno, errors)
                else:
                    errors.append(
                        f"{filepath}:{node.lineno}: Dynamic import with non-constant argument is forbidden"
                    )
            else:
                errors.append(
                    f"{filepath}:{node.lineno}: Dynamic import without a clear module name is forbidden"
                )
        elif (
            isinstance(node, ast.Subscript)
            and isinstance(node.value, ast.Attribute)
            and node.value.attr == "modules"
            and isinstance(node.value.value, ast.Name)
            and node.value.value.id == "sys"
        ):
            slice_val = node.slice
            if isinstance(slice_val, ast.Constant) and isinstance(slice_val.value, str):
                _check_module_allowed(slice_val.value, filepath, node.lineno, errors)
            else:
                errors.append(
                    f"{filepath}:{node.lineno}: sys.modules access with non-constant key is forbidden"
                )

    for error in errors:
        print(error, file=sys.stderr)

    return len(errors) == 0


def main(argv: Sequence[str] | None = None) -> int:
    """Run the 3rd-party import checker across the specified files.

    Args:
        argv: Optional sequence of file paths to check. Defaults to `sys.argv[1:]`.

    Returns:
        Exit code 0 if all files pass, 1 if any forbidden import is detected.

    """
    files: Sequence[str] = argv if argv is not None else sys.argv[1:]
    all_passed: bool = True
    for filepath in files:
        if not check_file(filepath):
            all_passed = False

    if not all_passed:
        sys.exit(1)
    return 0


if __name__ == "__main__":
    main()
