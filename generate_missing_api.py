import mlx.core
import mlx.nn
import mlx.optimizers
import mlx.utils
import mlx.core.fft
import mlx.core.linalg
import mlx.core.random
import ast
import os
import inspect


def get_public_api(module):
    return {
        name: getattr(module, name) for name in dir(module) if not name.startswith("_")
    }


mlx_apis = {}
for mod_name, mod in [
    ("mlx.core", mlx.core),
    ("mlx.core.fft", mlx.core.fft),
    ("mlx.core.linalg", mlx.core.linalg),
    ("mlx.core.random", mlx.core.random),
    ("mlx.nn", mlx.nn),
    ("mlx.optimizers", mlx.optimizers),
    ("mlx.utils", mlx.utils),
]:
    for name, obj in get_public_api(mod).items():
        if name not in mlx_apis:
            mlx_apis[name] = (f"{mod_name}.{name}", obj)

zero_api = set()


def parse_defs(filepath):
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


for file in os.listdir("src/zero_mlx"):
    if file.endswith(".py"):
        parse_defs(os.path.join("src/zero_mlx", file))

missing_apis = []
for name, (full_name, obj) in mlx_apis.items():
    if name not in zero_api:
        missing_apis.append((name, full_name, obj))

missing_apis.sort(key=lambda x: x[1])

md_content = "# Missing MLX APIs Implementation Plan\n\n"
md_content += "This document tracks the missing MLX APIs that need to be implemented in zero-mlx.\n\n"
md_content += "| Status | Name | Signature | Docstring |\n"
md_content += "|---|---|---|---|\n"

for name, full_name, obj in missing_apis:
    sig = ""
    try:
        sig = str(inspect.signature(obj))
    except Exception:
        doc = getattr(obj, "__doc__", "") or ""
        lines = doc.strip().split("\n")
        if lines and lines[0].startswith(name + "("):
            sig = lines[0][len(name) :].strip()
        else:
            sig = "(...)"

    sig = sig.replace("|", "\|").replace("\n", " ").replace("\r", " ")
    if not sig.startswith("("):
        sig = "(" + sig + ")"
    if len(sig) > 200:
        sig = sig[:197] + "..."

    doc_str = ""
    doc = getattr(obj, "__doc__", "") or ""
    lines = [line.strip() for line in doc.split("\n") if line.strip()]
    for line in lines:
        if (
            not line.startswith(name + "(")
            and not line.startswith("Args:")
            and not line.startswith("Returns:")
        ):
            doc_str = line
            break
    if not doc_str:
        doc_str = "No documentation available."

    doc_str = (
        doc_str.replace("|", "\|")
        .replace("\n", " ")
        .replace("\r", " ")
        .replace("`", "")
    )
    if len(doc_str) > 200:
        doc_str = doc_str[:197] + "..."

    md_content += f"| [ ] | `{full_name}` | `{sig}` | {doc_str} |\n"

with open("MLX_MISSING_API.md", "w") as f:
    f.write(md_content)
