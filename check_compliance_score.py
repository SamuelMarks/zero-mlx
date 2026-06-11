import mlx.core
import mlx.nn
import mlx.optimizers
import mlx.utils
import zero_mlx


def get_public_api(module):
    return {name for name in dir(module) if not name.startswith("_")}


mlx_core_api = get_public_api(mlx.core)
mlx_nn_api = get_public_api(mlx.nn)
mlx_opt_api = get_public_api(mlx.optimizers)
mlx_utils_api = get_public_api(mlx.utils)

zero_core_api = get_public_api(zero_mlx)

core_match = zero_core_api.intersection(mlx_core_api)

print(f"MLX core APIs: {len(mlx_core_api)}")
print(f"Zero MLX core APIs implemented: {len(core_match)}")
print(f"Core compliance: {len(core_match) / len(mlx_core_api) * 100:.2f}%")

total_mlx_api = (
    len(mlx_core_api) + len(mlx_nn_api) + len(mlx_opt_api) + len(mlx_utils_api)
)
print(f"Total MLX APIs (core, nn, optimizers, utils): {total_mlx_api}")
print(f"Overall compliance: {len(core_match) / total_mlx_api * 100:.2f}%")
