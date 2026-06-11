import re

with open("README.md", "r") as f:
    readme = f.read()

# Replace the pending features list or add a note
if "Implementing missing API" in readme:
    readme = readme.replace("Implementing missing API", "Implemented missing API")

with open("README.md", "w") as f:
    f.write(readme)
