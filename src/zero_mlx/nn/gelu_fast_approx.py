"""mlx.nn.gelu_fast_approx module stub."""

from typing import Any


def gelu_fast_approx(x: Any) -> Any:  # pragma: no cover
    """A fast approximation to Gaussian Error Linear Unit.

    Args:
        x (Any): The input array.

    Returns:
        Any: The output array.

    Raises:
        NotImplementedError: Always raised as this is a stub.

    """
    from zero_mlx.array import array
    import ml_switcheroo_compiler.ops as sops

    if not hasattr(x, "_tensor"):
        x = array(x)
    return array(sops.nn.gelu(x._tensor, approximate=True))
