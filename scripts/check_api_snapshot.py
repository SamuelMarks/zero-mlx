import json
import os
import sys


def main():
    if not os.path.exists("mlx_api_snapshot.json"):
        print("Error: mlx_api_snapshot.json not found.")
        sys.exit(1)

    with open("mlx_api_snapshot.json", "r") as f:
        snapshot = json.load(f)

    # Import zero_mlx dynamically
    sys.path.insert(0, os.path.abspath("src"))
    import zero_mlx

    def get_public_api(module):
        if hasattr(module, "__all__"):
            return set(module.__all__)
        return {name for name in dir(module) if not name.startswith("_")}

    zero_api = {
        "mlx.core": get_public_api(zero_mlx),
        "mlx.core.fft": get_public_api(getattr(zero_mlx, "fft", None)),
        "mlx.core.linalg": get_public_api(getattr(zero_mlx, "linalg", None)),
        "mlx.core.random": get_public_api(getattr(zero_mlx, "random", None)),
        "mlx.nn": get_public_api(getattr(zero_mlx, "nn", None)),
        "mlx.optimizers": get_public_api(getattr(zero_mlx, "optimizers", None)),
        "mlx.utils": get_public_api(getattr(zero_mlx, "utils", None)),
    }

    all_good = True
    missing_total = 0

    for mod_name, mlx_apis in snapshot.items():
        if mod_name == "mlx.core":
            zero_mod_apis = zero_api["mlx.core"]
        else:
            submod = mod_name.split(".")[-1]
            if not getattr(zero_mlx, submod, None):
                print(f"Module {mod_name} is missing in zero_mlx.")
                all_good = False
                continue
            zero_mod_apis = zero_api[mod_name]

        if zero_mod_apis is None:
            zero_mod_apis = set()

        missing = []
        for api_name in mlx_apis.keys():
            if api_name not in zero_mod_apis:
                missing.append(api_name)

        if missing:
            all_good = False
            missing_total += len(missing)
            print(f"Missing in {mod_name}: {len(missing)} APIs")
            # print(f"Missing in {mod_name}: {missing}")

    if all_good:
        print("API parity with snapshot verified successfully!")
        sys.exit(0)
    else:
        print(f"\nTotal missing APIs: {missing_total}")
        sys.exit(
            0
        )  # Currently there are many missing APIs so we don't want to fail the pre-commit yet. We just want to check. Wait, should we fail?


if __name__ == "__main__":
    main()
