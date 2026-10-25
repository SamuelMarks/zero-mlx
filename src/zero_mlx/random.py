"""Random number generation submodule (`mlx.core.random` / `zero_mlx.random`)."""

from __future__ import annotations

from zero_mlx.mlx_random import (
    PRNGKey,
    bernoulli,
    categorical,
    gumbel,
    key,
    laplace,
    multivariate_normal,
    normal,
    permutation,
    randint,
    seed,
    split,
    state,
    truncated_normal,
    uniform,
)

__all__ = [
    "PRNGKey",
    "bernoulli",
    "categorical",
    "gumbel",
    "key",
    "laplace",
    "multivariate_normal",
    "normal",
    "permutation",
    "randint",
    "seed",
    "split",
    "state",
    "truncated_normal",
    "uniform",
]
