"""mlx.nn.losses module."""

from typing import Optional, Union, Tuple
from zero_mlx.array import array
import ml_switcheroo_compiler.ops as sops


def cosine_similarity_loss(
    x1: array, x2: array, axis: int = -1, eps: float = 1e-8
) -> array:
    """Computes the cosine similarity loss."""
    x1_t = x1._tensor if hasattr(x1, "_tensor") else x1
    x2_t = x2._tensor if hasattr(x2, "_tensor") else x2
    return array(sops.cosine_similarity_loss(x1_t, x2_t, axis=axis, eps=eps))


def gaussian_nll_loss(
    input: array, target: array, var: array, full: bool = False, eps: float = 1e-6
) -> array:
    """Computes the Gaussian negative log likelihood loss."""
    i_t = input._tensor if hasattr(input, "_tensor") else input
    t_t = target._tensor if hasattr(target, "_tensor") else target
    v_t = var._tensor if hasattr(var, "_tensor") else var
    return array(sops.gaussian_nll_loss(i_t, t_t, v_t, full=full, eps=eps))


def hinge_loss(inputs: array, targets: array, margin: float = 1.0) -> array:
    """Computes the hinge loss."""
    i_t = inputs._tensor if hasattr(inputs, "_tensor") else inputs
    t_t = targets._tensor if hasattr(targets, "_tensor") else targets
    return array(sops.hinge_loss(i_t, t_t, margin=margin))


def huber_loss(inputs: array, targets: array, delta: float = 1.0) -> array:
    """Computes the Huber loss."""
    i_t = inputs._tensor if hasattr(inputs, "_tensor") else inputs
    t_t = targets._tensor if hasattr(targets, "_tensor") else targets
    return array(sops.huber_loss(i_t, t_t, delta=delta))


def kl_div_loss(inputs: array, targets: array, reduction: str = "none") -> array:
    """Computes the Kullback-Leibler divergence loss."""
    i_t = inputs._tensor if hasattr(inputs, "_tensor") else inputs
    t_t = targets._tensor if hasattr(targets, "_tensor") else targets
    return array(sops.kl_div_loss(i_t, t_t, reduction=reduction))


def l1_loss(predictions: array, targets: array, reduction: str = "none") -> array:
    """Computes the L1 loss."""
    p_t = predictions._tensor if hasattr(predictions, "_tensor") else predictions
    t_t = targets._tensor if hasattr(targets, "_tensor") else targets
    return array(sops.l1_loss(p_t, t_t, reduction=reduction))


def log_cosh_loss(predictions: array, targets: array) -> array:
    """Computes the log cosh loss."""
    p_t = predictions._tensor if hasattr(predictions, "_tensor") else predictions
    t_t = targets._tensor if hasattr(targets, "_tensor") else targets
    return array(sops.log_cosh_loss(p_t, t_t))


def margin_ranking_loss(
    input1: array, input2: array, target: array, margin: float = 0.0
) -> array:
    """Computes the margin ranking loss."""
    i1_t = input1._tensor if hasattr(input1, "_tensor") else input1
    i2_t = input2._tensor if hasattr(input2, "_tensor") else input2
    t_t = target._tensor if hasattr(target, "_tensor") else target
    return array(sops.margin_ranking_loss(i1_t, i2_t, t_t, margin=margin))


def mse_loss(predictions: array, targets: array, reduction: str = "none") -> array:
    """Computes the mean squared error loss."""
    p_t = predictions._tensor if hasattr(predictions, "_tensor") else predictions
    t_t = targets._tensor if hasattr(targets, "_tensor") else targets
    return array(sops.mse_loss(p_t, t_t, reduction=reduction))


def nll_loss(
    inputs: array,
    targets: array,
    weight: Optional[array] = None,
    reduction: str = "none",
    ignore_index: int = -100,
) -> array:
    """Computes the negative log likelihood loss."""
    i_t = inputs._tensor if hasattr(inputs, "_tensor") else inputs
    t_t = targets._tensor if hasattr(targets, "_tensor") else targets
    w_t = (
        weight._tensor
        if hasattr(weight, "_tensor")
        else weight
        if weight is not None
        else None
    )
    return array(
        sops.nll_loss(
            i_t, t_t, weight=w_t, reduction=reduction, ignore_index=ignore_index
        )
    )


def smooth_l1_loss(
    predictions: array, targets: array, beta: float = 1.0, reduction: str = "none"
) -> array:
    """Computes the smooth L1 loss."""
    p_t = predictions._tensor if hasattr(predictions, "_tensor") else predictions
    t_t = targets._tensor if hasattr(targets, "_tensor") else targets
    return array(sops.smooth_l1_loss(p_t, t_t, beta=beta, reduction=reduction))


def triplet_loss(
    anchor: array, positive: array, negative: array, margin: float = 1.0, p: float = 2.0
) -> array:
    """Computes the triplet margin loss."""
    a_t = anchor._tensor if hasattr(anchor, "_tensor") else anchor
    p_t = positive._tensor if hasattr(positive, "_tensor") else positive
    n_t = negative._tensor if hasattr(negative, "_tensor") else negative
    return array(sops.triplet_loss(a_t, p_t, n_t, margin=margin, p=p))


__all__ = [
    "cosine_similarity_loss",
    "gaussian_nll_loss",
    "hinge_loss",
    "huber_loss",
    "kl_div_loss",
    "l1_loss",
    "log_cosh_loss",
    "margin_ranking_loss",
    "mse_loss",
    "nll_loss",
    "smooth_l1_loss",
    "triplet_loss",
]
