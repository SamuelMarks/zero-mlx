import ast
import json
import os
import sys


def parse_defs(filepath, zero_api):
    if not os.path.exists(filepath):
        return
    with open(filepath, "r") as f:
        tree = ast.parse(f.read())
    for node in ast.walk(tree):
        if isinstance(
            node, (ast.FunctionDef, ast.ClassDef)
        ) and not node.name.startswith("_"):
            zero_api.add(node.name)
        elif isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and not target.id.startswith("_"):
                    zero_api.add(target.id)


def main():
    if not os.path.exists("mlx_api_snapshot.json"):
        print("Error: mlx_api_snapshot.json not found.")
        sys.exit(1)

    with open("mlx_api_snapshot.json", "r") as f:
        snapshot = json.load(f)

    mlx_apis = {}
    for mod_name, apis in snapshot.items():
        for name, info in apis.items():
            if name not in mlx_apis:
                mlx_apis[name] = (f"{mod_name}.{name}", info)

    zero_api = set()
    for root, _, files in os.walk("src/zero_mlx"):
        for file in files:
            if file.endswith(".py"):
                parse_defs(os.path.join(root, file), zero_api)

    missing_apis = []
    for name, (full_name, info) in mlx_apis.items():
        if name not in zero_api:
            missing_apis.append((name, full_name, info))

    missing_apis.sort(key=lambda x: x[1])

    md_content = "# Missing MLX APIs Implementation Plan\n\n"
    md_content += "This document tracks the missing MLX APIs that need to be implemented in zero-mlx.\n\n"
    md_content += "| Status | Name | Signature | Docstring |\n"
    md_content += "|---|---|---|---|\n"

    for name, full_name, info in missing_apis:
        sig = info.get("signature", "")
        sig = sig.replace("|", "\\|").replace("\n", " ").replace("\r", " ")
        if not sig.startswith("("):
            sig = "(" + sig + ")"
        if len(sig) > 200:
            sig = sig[:197] + "..."

        doc_str = info.get("doc_str", "")
        doc_str = (
            doc_str.replace("|", "\\|")
            .replace("\n", " ")
            .replace("\r", " ")
            .replace("`", "")
        )
        if len(doc_str) > 200:
            doc_str = doc_str[:197] + "..."

        md_content += f"| [ ] | `{full_name}` | `{sig}` | {doc_str} |\n"

    missing_api_path = "MLX_MISSING_API.md"
    existing_content = ""
    if os.path.exists(missing_api_path):
        with open(missing_api_path, "r") as f:
            existing_content = f.read()

    with open(missing_api_path, "w") as f:
        f.write(md_content)

    if existing_content != md_content:
        print(
            "API parity check failed: MLX_MISSING_API.md was updated. Please commit the changes."
        )
        sys.exit(1)
    else:
        print("API parity check passed.")


if __name__ == "__main__":
    main()
