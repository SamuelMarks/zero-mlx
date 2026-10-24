#!/usr/bin/env python3
"""AST-based linting pass for tautological assertions, bare pass blocks, and swallowed exceptions."""

from __future__ import annotations

import argparse
import ast
import os
import sys
from typing import Sequence


def _node_is_verification(sub: ast.AST) -> bool:
    """Return True if an AST node is an assertion, raise, or verification helper call.

    Args:
        sub: AST node to inspect.

    Returns:
        True if the node performs test verification.

    """
    if isinstance(sub, (ast.Assert, ast.Raise)):
        return True
    if isinstance(sub, ast.Call):
        target: ast.expr = sub.func
        if isinstance(target, ast.Call) and target.args:
            target = target.args[-1]
        if isinstance(target, ast.Name):
            return target.id.startswith(("assert", "_check", "check_", "verify", "_test", "_run"))
        if isinstance(target, ast.Attribute):
            return target.attr.startswith(("assert", "check", "verify")) or target.attr in (
                "raises",
                "warns",
                "fail",
                "approx",
            )
    return False


def _has_assertion_or_verification(
    func_node: ast.FunctionDef | ast.AsyncFunctionDef,
) -> bool:
    """Check whether a test function AST contains at least one assertion outside except blocks.

    Args:
        func_node: Function AST node to inspect.

    Returns:
        True if an assertion or verification call is present outside except handlers.

    """
    except_nodes: set[int] = set()
    for sub in ast.walk(func_node):
        if isinstance(sub, ast.ExceptHandler):
            for child in ast.walk(sub):
                except_nodes.add(id(child))

    for sub in ast.walk(func_node):
        if id(sub) in except_nodes:
            continue
        if _node_is_verification(sub):
            return True
    return False


def _is_excluded_middle_boolop(expr: ast.BoolOp) -> bool:
    """Return True if a BoolOp is an `or` tautology such as `x is None or x is not None`.

    Args:
        expr: BoolOp AST node to inspect.

    Returns:
        True if the BoolOp is a tautological disjunction.

    """
    if not isinstance(expr.op, ast.Or):
        return False
    for val in expr.values:
        if isinstance(val, ast.Constant) and val.value is True:
            return True
    lhs, rhs = expr.values[0], expr.values[1]
    return (
        len(expr.values) == 2
        and isinstance(lhs, ast.Compare)
        and isinstance(rhs, ast.Compare)
        and len(lhs.ops) == 1
        and len(rhs.ops) == 1
        and ast.dump(lhs.left) == ast.dump(rhs.left)
        and ast.dump(lhs.comparators[0]) == ast.dump(rhs.comparators[0])
        and {type(lhs.ops[0]), type(rhs.ops[0])} in ({ast.Is, ast.IsNot}, {ast.Eq, ast.NotEq})
    )


def _check_assert_node(path: str, node: ast.Assert) -> str | None:
    """Check an `ast.Assert` node for vacuous boolean constants or self-comparisons.

    Args:
        path: File path of the test module.
        node: Assert AST node.

    Returns:
        Error message if invalid, else None.

    """
    test_expr: ast.expr = node.test
    if isinstance(test_expr, ast.Constant) and isinstance(test_expr.value, bool):
        return f"FAILED: Vacuous 'assert {test_expr.value}' found in {path}:{node.lineno}"
    if isinstance(test_expr, ast.BoolOp) and _is_excluded_middle_boolop(test_expr):
        return f"FAILED: Tautological disjunction found in {path}:{node.lineno}"
    if (
        isinstance(test_expr, ast.Compare)
        and len(test_expr.ops) == 1
        and isinstance(test_expr.ops[0], (ast.Eq, ast.Is))
        and ast.dump(test_expr.left) == ast.dump(test_expr.comparators[0])
    ):
        return f"FAILED: Tautological comparison found in {path}:{node.lineno}"
    return None


def _check_except_handler(path: str, node: ast.ExceptHandler) -> list[str]:
    """Check an `ast.ExceptHandler` for swallowed exceptions or caught AssertionErrors.

    Args:
        path: File path of the test module.
        node: ExceptHandler AST node.

    Returns:
        List of violation messages.

    """
    errs: list[str] = []
    if node.type is None:
        errs.append(f"FAILED: Vacuous 'except:' found in {path}:{node.lineno}")
    else:
        for sub in ast.walk(node.type):
            if isinstance(sub, ast.Name) and sub.id == "AssertionError":
                errs.append(
                    f"FAILED: Catching AssertionError in except handler at {path}:{node.lineno}"
                )
        if len(node.body) == 1 and isinstance(node.body[0], ast.Pass):
            errs.append(f"FAILED: Vacuous 'except ...: pass' found in {path}:{node.lineno}")
    return errs


def check_test_file_ast(
    path: str,
    check_zero_assertions: bool = True,
    check_swallowed_exceptions: bool = True,
) -> list[str]:
    """Inspect a test file's AST for vacuous assertions and swallowed exceptions.

    Args:
        path: File path of the Python test module.
        check_zero_assertions: Whether to flag `test_*` functions with zero assertions.
        check_swallowed_exceptions: Whether to flag `try...except: pass` blocks.

    Returns:
        List of violation messages discovered in the file.

    """
    with open(path, encoding="utf-8") as file:
        content: str = file.read()

    try:
        tree: ast.Module = ast.parse(content, filename=path)
    except SyntaxError:
        return []

    errors: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Assert):
            err = _check_assert_node(path, node)
            if err is not None:
                errors.append(err)
        elif check_swallowed_exceptions and isinstance(node, ast.ExceptHandler):
            errors.extend(_check_except_handler(path, node))
        elif check_zero_assertions and isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if node.name.startswith("test_") and not _has_assertion_or_verification(node):
                errors.append(
                    f"FAILED: Zero assertions in test function '{node.name}' at {path}:{node.lineno}"
                )

    return errors


def main(argv: Sequence[str] | None = None) -> int:
    """Run AST-based lint pass across the test suite.

    Args:
        argv: Optional CLI arguments. Defaults to `sys.argv[1:]`.

    Returns:
        Exit code 0 on pass, 1 on failure.

    """
    parser = argparse.ArgumentParser(description="Run AST-based test quality lint checks.")
    parser.add_argument("--tests-dir", type=str, default="tests", help="Tests directory to scan.")
    parser.add_argument(
        "--check-zero-assertions",
        action="store_true",
        default=True,
        help="Flag test functions with zero assertions.",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)

    all_errors: list[str] = []
    if os.path.exists(args.tests_dir):
        for root, _, files in sorted(os.walk(args.tests_dir)):
            for fname in sorted(files):
                if fname.endswith(".py"):
                    path: str = os.path.join(root, fname)
                    all_errors.extend(
                        check_test_file_ast(
                            path,
                            check_zero_assertions=args.check_zero_assertions,
                        )
                    )

    if all_errors:
        for err in all_errors:
            print(err, file=sys.stderr)
        sys.exit(1)

    print("Test quality linting passed.")
    return 0


if __name__ == "__main__":
    main()
