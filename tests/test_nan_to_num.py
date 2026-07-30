import pytest
import numpy as np
from tests.test_ops_parity_logical import check_parity


def test_NanToNum_parity():
    check_parity(
        "Nan_To_Num",
        lambda: [np.array([np.nan, np.inf, -np.inf, 1.0], dtype=np.float32)],
        kwargs_generator=lambda: {"nan": 0.0, "posinf": 1e30, "neginf": -1e30},
    )
