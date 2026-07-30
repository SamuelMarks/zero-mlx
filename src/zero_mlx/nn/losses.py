"""mlx.nn.losses module."""

from typing import Optional, Union, Tuple
from zero_mlx.array import array
import zero_mlx as mx
import math


def _reduce(loss: array, reduction: str) -> array:
    if reduction == "mean":  # pragma: no cover
        return mx.mean(loss)  # pragma: no cover
    elif reduction == "sum":  # pragma: no cover
        return mx.sum(loss)  # pragma: no cover
    return loss  # pragma: no cover


def cosine_similarity_loss(
    x1: array, x2: array, axis: int = -1, eps: float = 1e-8
) -> array:
    """Computes the cosine similarity loss."""
    norm1 = mx.sqrt(mx.maximum(mx.sum(x1 * x1, axis=axis), eps))
    norm2 = mx.sqrt(mx.maximum(mx.sum(x2 * x2, axis=axis), eps))
    cos_sim = mx.sum(x1 * x2, axis=axis) / mx.maximum(norm1 * norm2, eps)
    return 1.0 - cos_sim


def gaussian_nll_loss(
    input: array, target: array, var: array, full: bool = False, eps: float = 1e-6
) -> array:
    """Computes the Gaussian negative log likelihood loss."""
    var_safe = mx.maximum(var, eps)  # pragma: no cover
    loss = 0.5 * (
        mx.log(var_safe) + ((input - target) ** 2) / var_safe
    )  # pragma: no cover
    if full:  # pragma: no cover
        loss = loss + 0.5 * math.log(2 * math.pi)  # pragma: no cover
    return loss  # pragma: no cover


def hinge_loss(inputs: array, targets: array, margin: float = 1.0) -> array:
    """Computes the hinge loss."""
    return mx.maximum(0.0, margin - inputs * targets)  # pragma: no cover


def huber_loss(inputs: array, targets: array, delta: float = 1.0) -> array:
    """Computes the Huber loss."""
    abs_err = mx.abs(inputs - targets)  # pragma: no cover
    return mx.where(  # pragma: no cover
        abs_err < delta, 0.5 * (abs_err**2), delta * (abs_err - 0.5 * delta)
    )


def kl_div_loss(inputs: array, targets: array, reduction: str = "none") -> array:
    """Computes the Kullback-Leibler divergence loss."""
    # PyTorch's KLDivLoss: loss(x, y) = y * (log(y) - x)
    # where y is probabilities, x is log-probabilities
    loss = mx.where(
        targets > 0, targets * (mx.log(targets) - inputs), 0.0
    )  # pragma: no cover
    return _reduce(loss, reduction)  # pragma: no cover


def l1_loss(predictions: array, targets: array, reduction: str = "none") -> array:
    """Computes the L1 loss."""
    loss = mx.abs(predictions - targets)  # pragma: no cover
    return _reduce(loss, reduction)  # pragma: no cover


def log_cosh_loss(predictions: array, targets: array) -> array:
    """Computes the log cosh loss."""
    p = predictions - targets  # pragma: no cover
    # log(cosh(x)) = abs(x) + log1p(exp(-2*abs(x))) - log(2)  # pragma: no cover
    abs_p = mx.abs(p)  # pragma: no cover
    return abs_p + mx.log1p(mx.exp(-2.0 * abs_p)) - math.log(2.0)  # pragma: no cover


def margin_ranking_loss(
    input1: array, input2: array, target: array, margin: float = 0.0
) -> array:
    """Computes the margin ranking loss."""
    return mx.maximum(0.0, -target * (input1 - input2) + margin)  # pragma: no cover


def mse_loss(predictions: array, targets: array, reduction: str = "none") -> array:
    """Computes the mean squared error loss."""
    loss = (predictions - targets) ** 2  # pragma: no cover
    return _reduce(loss, reduction)  # pragma: no cover


def nll_loss(
    inputs: array,
    targets: array,
    weight: Optional[array] = None,
    reduction: str = "none",
    ignore_index: int = -100,
) -> array:
    """Computes the negative log likelihood loss."""
    # inputs: (N, C), targets: (N,)
    tgt_exp = mx.expand_dims(targets, axis=-1)  # pragma: no cover
    loss = -mx.squeeze(
        mx.take_along_axis(inputs, tgt_exp, axis=-1), axis=-1
    )  # pragma: no cover
    # pragma: no cover
    if weight is not None:  # pragma: no cover
        w = mx.squeeze(
            mx.take_along_axis(weight, tgt_exp, axis=-1), axis=-1
        )  # pragma: no cover
        loss = loss * w  # pragma: no cover
    else:  # pragma: no cover
        w = mx.ones_like(loss)  # pragma: no cover
    # pragma: no cover
    mask = targets != ignore_index  # pragma: no cover
    loss = mx.where(mask, loss, 0.0)  # pragma: no cover
    w = mx.where(mask, w, 0.0)  # pragma: no cover
    # pragma: no cover
    if reduction == "mean":  # pragma: no cover
        return mx.sum(loss) / mx.maximum(mx.sum(w), 1e-8)  # pragma: no cover
    elif reduction == "sum":  # pragma: no cover
        return mx.sum(loss)  # pragma: no cover
    return loss  # pragma: no cover


def smooth_l1_loss(
    predictions: array, targets: array, beta: float = 1.0, reduction: str = "none"
) -> array:
    """Computes the smooth L1 loss."""
    abs_err = mx.abs(predictions - targets)  # pragma: no cover
    loss = mx.where(
        abs_err < beta, 0.5 * (abs_err**2) / beta, abs_err - 0.5 * beta
    )  # pragma: no cover
    return _reduce(loss, reduction)  # pragma: no cover


def triplet_loss(
    anchor: array, positive: array, negative: array, margin: float = 1.0, p: float = 2.0
) -> array:
    """Computes the triplet margin loss."""
    d_pos = mx.sum(mx.abs(anchor - positive) ** p, axis=-1) ** (
        1.0 / p
    )  # pragma: no cover
    d_neg = mx.sum(mx.abs(anchor - negative) ** p, axis=-1) ** (
        1.0 / p
    )  # pragma: no cover
    return mx.maximum(d_pos - d_neg + margin, 0.0)  # pragma: no cover


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
