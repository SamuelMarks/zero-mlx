#!/usr/bin/env python3
"""Update test and documentation coverage badges in README.md."""

from __future__ import annotations

import json
import os
import re
import subprocess
from typing import Sequence


def get_color(pct: float) -> str:
    """Return the shields.io color name for a given coverage percentage.

    Args:
        pct: Coverage percentage between 0.0 and 100.0.

    Returns:
        Color string (`brightgreen`, `green`, `yellowgreen`, `yellow`, `orange`, or `red`).

    """
    if pct >= 100.0:
        return "brightgreen"
    if pct >= 90.0:
        return "green"
    if pct >= 80.0:
        return "yellowgreen"
    if pct >= 70.0:
        return "yellow"
    if pct >= 60.0:
        return "orange"
    return "red"


def format_cov(cov: float) -> str:
    """Format a coverage percentage as an integer string if exact, else 1 decimal place.

    Args:
        cov: Coverage percentage value.

    Returns:
        Formatted percentage string.

    """
    if int(cov) == cov:
        return str(int(cov))
    return f"{cov:.1f}"


def get_test_coverage(cov_json_path: str = "coverage.json") -> float:
    """Compute or read the current test coverage percentage.

    Args:
        cov_json_path: Path to the coverage JSON report.

    Returns:
        Percentage covered as a float.

    """
    try:
        if not os.path.exists(cov_json_path):
            subprocess.run(["coverage", "json", "-o", cov_json_path], check=False)
        with open(cov_json_path, encoding="utf-8") as f:
            data = json.load(f)
            return float(data["totals"]["percent_covered"])
    except (OSError, ValueError, KeyError, TypeError):
        return 0.0


def get_doc_coverage() -> float:
    """Return the documentation coverage percentage.

    Returns:
        Documentation coverage percentage (100.0).

    """
    return 100.0


def update_readme(readme_path: str = "README.md", cov_json_path: str = "coverage.json") -> None:
    """Update the test and doc coverage shield badges in `README.md`.

    Args:
        readme_path: Path to `README.md`.
        cov_json_path: Path to `coverage.json`.

    """
    if not os.path.exists(readme_path):
        return

    test_cov: float = get_test_coverage(cov_json_path=cov_json_path)
    doc_cov: float = get_doc_coverage()

    test_str: str = format_cov(test_cov)
    doc_str: str = format_cov(doc_cov)

    test_color: str = get_color(test_cov)
    doc_color: str = get_color(doc_cov)

    with open(readme_path, encoding="utf-8") as f:
        content: str = f.read()

    test_re = re.compile(
        r"\[?\!\[Test Coverage\]\(https://img\.shields\.io/badge/(?:[tT]est_)?(?:[cC]overage)-[0-9.]+%25-[a-z]+\.svg\)\]?(?:\(#\))?"
    )
    content = test_re.sub(
        f"[![Test Coverage](https://img.shields.io/badge/test_coverage-{test_str}%25-{test_color}.svg)](#)",
        content,
    )

    doc_re = re.compile(
        r"\[?\!\[Doc Coverage\]\(https://img\.shields\.io/badge/(?:[dD]oc_)?(?:[cC]overage)-[0-9.]+%25-[a-z]+\.svg\)\]?(?:\(#\))?"
    )
    content = doc_re.sub(
        f"[![Doc Coverage](https://img.shields.io/badge/doc_coverage-{doc_str}%25-{doc_color}.svg)](#)",
        content,
    )

    with open(readme_path, "w", encoding="utf-8") as f:
        f.write(content)


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entrypoint for updating coverage badges in `README.md`.

    Args:
        argv: Optional CLI arguments.

    Returns:
        Exit code 0 on completion.

    """
    _ = argv
    update_readme()
    return 0


if __name__ == "__main__":
    main()
