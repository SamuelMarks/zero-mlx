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


def parse_defs(filepath):
    with open(filepath, "r") as f:
        tree = ast.parse(f.read())
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and not node.name.startswith("_"):
            zero_api.add(node.name)
        elif isinstance(node, ast.ClassDef) and not node.name.startswith("_"):
            zero_api.add(node.name)


for file in ["ops.py", "linalg.py", "fft.py", "mlx_random.py", "array.py"]:
    if os.path.exists("src/zero_mlx/" + file):
        parse_defs("src/zero_mlx/" + file)

print(f"MLX core APIs: {len(mlx_core_api)}")
print(f"Zero MLX APIs defined: {len(zero_api)}")

core_match = zero_api.intersection(mlx_core_api)
print(f"Zero MLX implemented MLX core APIs: {len(core_match)}")

total_mlx_api = (
    len(mlx_core_api) + len(mlx_nn_api) + len(mlx_opt_api) + len(mlx_utils_api)
)
print(f"Total MLX APIs: {total_mlx_api}")
print(f"Overall compliance: {len(core_match) / total_mlx_api * 100:.2f}%")
print(f"Core compliance: {len(core_match) / len(mlx_core_api) * 100:.2f}%")
