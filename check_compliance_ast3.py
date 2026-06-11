import mlx.core
import mlx.nn
import mlx.optimizers
import mlx.utils
import mlx.core.fft
import mlx.core.linalg
import mlx.core.random
import ast
import os


def get_public_api(module):
    return {name for name in dir(module) if not name.startswith("_")}


mlx_core_api = (
    get_public_api(mlx.core)
    | get_public_api(mlx.core.fft)
    | get_public_api(mlx.core.linalg)
    | get_public_api(mlx.core.random)
)
mlx_nn_api = get_public_api(mlx.nn)
mlx_opt_api = get_public_api(mlx.optimizers)
mlx_utils_api = get_public_api(mlx.utils)

zero_api = set()


def parse_all(filepath):
    with open(filepath, "r") as f:
        tree = ast.parse(f.read())

    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == "__all__":
                    if isinstance(node.value, ast.List):
                        for elt in node.value.elts:
                            if isinstance(elt, ast.Constant):
                                zero_api.add(elt.value)
        elif isinstance(node, ast.Expr):
            if isinstance(node.value, ast.Call):
                if isinstance(node.value.func, ast.Attribute):
                    if (
                        isinstance(node.value.func.value, ast.Name)
                        and node.value.func.value.id == "__all__"
                    ):
                        if node.value.func.attr == "append":
                            if isinstance(node.value.args[0], ast.Constant):
                                zero_api.add(node.value.args[0].value)
                        elif node.value.func.attr == "extend":
                            if isinstance(node.value.args[0], ast.List):
                                for elt in node.value.args[0].elts:
                                    if isinstance(elt, ast.Constant):
                                        zero_api.add(elt.value)


parse_all("src/zero_mlx/__init__.py")
if os.path.exists("src/zero_mlx/linalg.py"):
    parse_all("src/zero_mlx/linalg.py")
if os.path.exists("src/zero_mlx/fft.py"):
    parse_all("src/zero_mlx/fft.py")
if os.path.exists("src/zero_mlx/mlx_random.py"):
    parse_all("src/zero_mlx/mlx_random.py")

print(f"MLX core APIs: {len(mlx_core_api)}")
print(f"Zero MLX APIs exposed in __all__: {len(zero_api)}")

core_match = zero_api.intersection(mlx_core_api)
print(f"Zero MLX implemented MLX core APIs: {len(core_match)}")

total_mlx_api = (
    len(mlx_core_api) + len(mlx_nn_api) + len(mlx_opt_api) + len(mlx_utils_api)
)
print(f"Total MLX APIs: {total_mlx_api}")
print(f"Overall compliance: {len(core_match) / total_mlx_api * 100:.2f}%")
print(f"Core compliance: {len(core_match) / len(mlx_core_api) * 100:.2f}%")
