import sys
import os
import ast


def main():
    sys.path.insert(0, os.path.abspath("../ml-switcheroo-compiler/src"))

    try:
        import ml_switcheroo_compiler.ops
        from ml_switcheroo_compiler.ops.base import _OP_REGISTRY
    except ImportError:
        print(
            "Error: Could not import ml_switcheroo_compiler. Make sure it is in ../ml-switcheroo-compiler"
        )
        sys.exit(1)

    all_ops = set(_OP_REGISTRY.keys())

    test_file = "tests/test_ops_parity.py"
    implemented_tests = set()

    if os.path.exists(test_file):
        with open(test_file, "r") as f:
            tree = ast.parse(f.read())
            for node in ast.walk(tree):
                if (
                    isinstance(node, ast.FunctionDef)
                    and node.name.startswith("test_")
                    and node.name.endswith("_parity")
                ):
                    op_name = node.name[5:-7]  # remove test_ and _parity
                    implemented_tests.add(op_name)

    missing = all_ops - implemented_tests

    # Write to a file for tracking
    tracking_file = "COMPILER_TEST_PARITY.md"
    md_content = "# Compiler Operations Test Parity\n\n"
    md_content += "This tracks which ml-switcheroo-compiler operations have a dedicated parity test against MLX in `tests/test_ops_parity.py`.\n\n"
    md_content += "| Status | Operation |\n"
    md_content += "|---|---|\n"

    for op in sorted(all_ops):
        status = "[x]" if op in implemented_tests else "[ ]"
        md_content += f"| {status} | `{op}` |\n"

    existing_content = ""
    if os.path.exists(tracking_file):
        with open(tracking_file, "r") as f:
            existing_content = f.read()

    with open(tracking_file, "w") as f:
        f.write(md_content)

    if existing_content != md_content:
        print(
            f"Compiler test completeness check failed: {tracking_file} was updated. Please commit the changes."
        )
        sys.exit(1)
    else:
        print("Compiler test completeness check passed.")


if __name__ == "__main__":
    main()
