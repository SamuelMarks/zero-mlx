"""Parity test for `nan_to_num` operation."""

from __future__ import annotations

import numpy as np

from tests.conftest import check_parity


def test_NanToNum_parity() -> None:
    """Verify `nan_to_num` parity between `zero_mlx` and `mlx.core`."""
    check_parity(
        "Nan_To_Num",
        lambda: [np.array([np.nan, np.inf, -np.inf, 1.0], dtype=np.float32)],
        kwargs_generator=lambda: {"nan": 0.0, "posinf": 1e30, "neginf": -1e30},
    )
