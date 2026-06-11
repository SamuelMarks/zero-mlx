import os

docs_dir = "/tmp/mlx-official/docs/src/python"
output_file = "API_TEST_MATRIX.md"

api_categories = [
    "ops.rst",
    "nn.rst",
    "optimizers.rst",
    "transforms.rst",
    "random.rst",
    "linalg.rst",
]
matrix = "# MLX API Test Coverage Matrix\n\nRequirement: Minimum one test per API.\n\n"

for rst in api_categories:
    filepath = os.path.join(docs_dir, rst)
    if not os.path.exists(filepath):
        continue
    with open(filepath, "r") as f:
        content = f.read()

    matrix += f"## {rst.replace('.rst', '').capitalize()}\n"
    in_autosummary = False
    for line in content.split("\n"):
        if ".. autosummary::" in line:
            in_autosummary = True
            continue
        if in_autosummary:
            if line.strip() == "" or line.startswith(" ") or line.startswith("\t"):
                api = line.strip()
                if api and not api.startswith(":"):
                    matrix += f"- [ ] `{api}`\n"
            elif (
                not line.startswith(" ")
                and not line.startswith("\t")
                and line.strip() != ""
            ):
                in_autosummary = False

with open(output_file, "w") as f:
    f.write(matrix)
