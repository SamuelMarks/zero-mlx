#!/usr/bin/env python3
import ast
import sys
import importlib.util

ALLOWED_3RD_PARTY = {
    "pydantic",
    "cdd",
    "cdd_python",
    "ml_switcheroo_ir",
    "ml_switcheroo_compiler",
    "zero_keras",
    "zero_mlx",  # 1st party
}

FALLBACK_STDLIB = {
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


def is_stdlib(module_name):
    if not module_name:
        return True

    base_name = module_name.split(".")[0]

    if hasattr(sys, "stdlib_module_names"):
        if base_name in sys.stdlib_module_names:
            return True

    if base_name in sys.builtin_module_names:
        return True

    if base_name in FALLBACK_STDLIB:
        return True

    try:
        spec = importlib.util.find_spec(base_name)
        if spec is None:
            return False

        locations = []
        if spec.origin:
            if spec.origin == "built-in":
                return True
            locations.append(spec.origin)
        if spec.submodule_search_locations:
            locations.extend(spec.submodule_search_locations)

        for loc in locations:
            if "site-packages" in loc or "dist-packages" in loc:
                return False

        if locations:
            return any(
                loc.startswith(sys.base_prefix) or loc.startswith(sys.prefix)
                for loc in locations
            )

        return True
    except Exception:
        return False


def check_file(filepath):
    with open(filepath, "r", encoding="utf-8") as f:
        try:
            tree = ast.parse(f.read(), filename=filepath)
        except SyntaxError:
            return False

    errors = []

    def check_module(module_name, lineno):
        if not module_name:
            return
        base_name = module_name.split(".")[0]
        if base_name not in ALLOWED_3RD_PARTY and not is_stdlib(base_name):
            errors.append(
                f"{filepath}:{lineno}: Forbidden 3rd-party import '{base_name}' (from '{module_name}')"
            )

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                check_module(alias.name, node.lineno)
        elif isinstance(node, ast.ImportFrom):
            if node.level == 0 and node.module:
                check_module(node.module, node.lineno)
        elif isinstance(node, ast.Call):
            func = node.func

            def check_dynamic_import(call_node):
                module_val = None
                if call_node.args:
                    module_val = call_node.args[0]
                else:
                    for kw in call_node.keywords:
                        if kw.arg == "name":
                            module_val = kw.value
                            break

                if module_val is not None:
                    if isinstance(module_val, ast.Constant) and isinstance(
                        module_val.value, str
                    ):
                        check_module(module_val.value, call_node.lineno)
                    else:
                        errors.append(
                            f"{filepath}:{call_node.lineno}: Dynamic import with non-constant argument is forbidden"
                        )
                else:
                    errors.append(
                        f"{filepath}:{call_node.lineno}: Dynamic import without a clear module name is forbidden"
                    )

            if isinstance(func, ast.Name):
                if func.id in ("__import__", "import_module"):
                    check_dynamic_import(node)
            elif isinstance(func, ast.Attribute):
                if func.attr in ("__import__", "import_module"):
                    check_dynamic_import(node)
                elif (
                    isinstance(func.value, ast.Attribute)
                    and func.value.attr == "modules"
                ):
                    if (
                        isinstance(func.value.value, ast.Name)
                        and func.value.value.id == "sys"
                    ):
                        check_dynamic_import(node)

        elif isinstance(node, ast.Subscript):
            value = node.value
            if isinstance(value, ast.Attribute) and value.attr == "modules":
                if isinstance(value.value, ast.Name) and value.value.id == "sys":
                    slice_val = node.slice
                    if isinstance(slice_val, ast.Constant) and isinstance(
                        slice_val.value, str
                    ):
                        check_module(slice_val.value, node.lineno)
                    else:
                        errors.append(
                            f"{filepath}:{node.lineno}: sys.modules access with non-constant key is forbidden"
                        )

    for error in errors:
        print(error, file=sys.stderr)

    return len(errors) == 0


def main():
    files = sys.argv[1:]
    all_passed = True
    for filepath in files:
        if not check_file(filepath):
            all_passed = False

    if not all_passed:
        sys.exit(1)


if __name__ == "__main__":
    main()
