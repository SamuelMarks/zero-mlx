import mlx.core
import mlx.nn
import mlx.optimizers
import mlx.utils
import mlx.core.fft
import mlx.core.linalg
import mlx.core.random
import json
import inspect
import sys


def get_public_api(module, module_name):
    api = {}
    for name in dir(module):
        if not name.startswith("_"):
            obj = getattr(module, name)
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

            api[name] = {
                "type": type(obj).__name__,
                "signature": sig,
                "doc_str": doc_str,
            }
    return api


snapshot = {
    "mlx.core": get_public_api(mlx.core, "mlx.core"),
    "mlx.core.fft": get_public_api(mlx.core.fft, "mlx.core.fft"),
    "mlx.core.linalg": get_public_api(mlx.core.linalg, "mlx.core.linalg"),
    "mlx.core.random": get_public_api(mlx.core.random, "mlx.core.random"),
    "mlx.nn": get_public_api(mlx.nn, "mlx.nn"),
    "mlx.optimizers": get_public_api(mlx.optimizers, "mlx.optimizers"),
    "mlx.utils": get_public_api(mlx.utils, "mlx.utils"),
}

with open("mlx_api_snapshot.json", "w") as f:
    json.dump(snapshot, f, indent=2)

print("Snapshot generated: mlx_api_snapshot.json")
