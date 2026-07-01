"""mlx.nn.gelu_approx module stub."""


def gelu_approx(x):  # pragma: no cover
    """An approximation to Gaussian Error Linear Unit."""
    from zero_mlx.array import array
    import ml_switcheroo_compiler.ops as sops

    if not hasattr(x, "_tensor"):
        x = array(x)
    return array(sops.nn.gelu(x._tensor, approximate=True))
