import mlx.core
import mlx.nn
import mlx.optimizers
import mlx.utils
import mlx.core.fft
import mlx.core.linalg
import mlx.core.random
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
with open("src/zero_mlx/__init__.py", "r") as f:
    for line in f:
        if line.startswith("__all__"):
            break

    in_all = False
    for line in f:
        if line.strip() == "[":
            in_all = True
            continue
        if in_all and line.strip() == "]":
            break
        if in_all and line.strip():
            api = line.strip().strip('",')
            zero_api.add(api)

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
