"""Loss functions (`mlx.nn.losses`) for zero_mlx."""

from __future__ import annotations

import math

from zero_mlx._types import ArrayInput, LossReduction
from zero_mlx.array import array
from zero_mlx.dtypes import float32, int32
from zero_mlx.nn.activations import log_softmax, sigmoid, softmax
import zero_mlx.ops as mx

Reduction = LossReduction


def _as_array(x: ArrayInput) -> array:
    """Convert `x` to a `zero_mlx.array`.

    Args:
        x: Input array or convertible value.

    Returns:
        `zero_mlx.array` instance.

    """
    return x if isinstance(x, array) else array(x)


def _reduce(loss: array, reduction: str) -> array:
    """Apply loss reduction (`'none'`, `'mean'`, or `'sum'`).

    Args:
        loss: Unreduced loss `array`.
        reduction: Reduction mode (`'none'`, `'mean'`, or `'sum'`).

    Returns:
        Reduced loss `array`.

    Raises:
        ValueError: If `reduction` is not one of `'none'`, `'mean'`, or `'sum'`.

    """
    if reduction == "mean":
        return mx.mean(loss)
    if reduction == "sum":
        return mx.sum(loss)
    if reduction == "none":
        return loss
    raise ValueError(f"Invalid reduction '{reduction}', expected 'none', 'mean', or 'sum'")


def cross_entropy(
    logits: ArrayInput,
    targets: ArrayInput,
    weights: ArrayInput | None = None,
    axis: int = -1,
    label_smoothing: float = 0.0,
    reduction: LossReduction = "none",
) -> array:
    """Compute cross-entropy loss between `logits` and class index or probability `targets`.

    Args:
        logits: Unnormalized logits.
        targets: Ground-truth class indices or class probabilities.
        weights: Optional per-sample weight array.
        axis: Class dimension axis.
        label_smoothing: Label smoothing factor in `[0, 1)`.
        reduction: Reduction mode (`'none'`, `'mean'`, `'sum'`).

    Returns:
        Cross-entropy loss `array`.

    Raises:
        ValueError: If `label_smoothing` is outside `[0, 1)`.

    """
    if label_smoothing < 0.0 or label_smoothing >= 1.0:
        raise ValueError(f"label_smoothing must be in [0, 1), got {label_smoothing}")
    l_arr = _as_array(logits)
    t_arr = _as_array(targets)
    log_probs = log_softmax(l_arr, axis=axis)
    if t_arr.ndim == l_arr.ndim:
        loss = -mx.sum(t_arr * log_probs, axis=axis)
    else:
        tgt_idx = mx.expand_dims(t_arr.astype(int32), axis=axis)
        picked = mx.squeeze(mx.take_along_axis(log_probs, tgt_idx, axis=axis), axis=axis)
        if label_smoothing > 0.0:
            smooth_loss = -mx.mean(log_probs, axis=axis)
            loss = (1.0 - label_smoothing) * (-picked) + label_smoothing * smooth_loss
        else:
            loss = -picked
    if weights is not None:
        loss = loss * _as_array(weights)
    return _reduce(loss, reduction)


def binary_cross_entropy(
    inputs: ArrayInput,
    targets: ArrayInput,
    weights: ArrayInput | None = None,
    with_logits: bool = True,
    reduction: LossReduction = "mean",
) -> array:
    """Compute binary cross-entropy loss between `inputs` and `targets`.

    Args:
        inputs: Predicted logits (if `with_logits=True`) or probabilities.
        targets: Binary target values in `[0, 1]`.
        weights: Optional element-wise or sample-wise weights.
        with_logits: Whether `inputs` are raw logits.
        reduction: Reduction mode (`'none'`, `'mean'`, `'sum'`).

    Returns:
        Binary cross-entropy loss `array`.

    """
    x = _as_array(inputs).astype(float32)
    y = _as_array(targets).astype(float32)
    if with_logits:
        loss = mx.maximum(x, 0.0) - x * y + mx.log1p(mx.exp(-mx.abs(x)))
    else:
        eps = 1e-7
        p = mx.clip(x, eps, 1.0 - eps)
        loss = -(y * mx.log(p) + (1.0 - y) * mx.log(1.0 - p))
    if weights is not None:
        loss = loss * _as_array(weights)
    return _reduce(loss, reduction)


def cosine_similarity_loss(
    x1: ArrayInput,
    x2: ArrayInput,
    axis: int = 1,
    eps: float = 1e-8,
    reduction: LossReduction = "none",
) -> array:
    """Compute the cosine similarity distance `1 - cos(x1, x2)` (or `cos(x1, x2)` similarity).

    Args:
        x1: First input tensor.
        x2: Second input tensor.
        axis: Dimension along which cosine similarity is computed.
        eps: Minimum denominator value to avoid division by zero.
        reduction: Reduction mode (`'none'`, `'mean'`, `'sum'`).

    Returns:
        Cosine similarity loss `array`.

    """
    a1 = _as_array(x1)
    a2 = _as_array(x2)
    norm1 = mx.sqrt(mx.maximum(mx.sum(a1 * a1, axis=axis), eps))
    norm2 = mx.sqrt(mx.maximum(mx.sum(a2 * a2, axis=axis), eps))
    cos_sim = mx.sum(a1 * a2, axis=axis) / mx.maximum(norm1 * norm2, eps)
    loss = 1.0 - cos_sim
    return _reduce(loss, reduction)


def gaussian_nll_loss(
    inputs: ArrayInput,
    targets: ArrayInput,
    vars: ArrayInput,
    full: bool = False,
    eps: float = 1e-6,
    reduction: LossReduction = "none",
) -> array:
    """Compute the Gaussian negative log-likelihood loss.

    Args:
        inputs: Predicted expectations of the Gaussian distribution.
        targets: Sampled target values.
        vars: Predicted variances (one for each expectation).
        full: Whether to include the constant `0.5 * log(2 * pi)` term.
        eps: Stability clamp floor for `vars`.
        reduction: Reduction mode (`'none'`, `'mean'`, `'sum'`).

    Returns:
        Gaussian NLL loss `array`.

    """
    inp = _as_array(inputs)
    tgt = _as_array(targets)
    v = _as_array(vars)
    var_safe = mx.maximum(v, eps)
    loss = 0.5 * (mx.log(var_safe) + ((inp - tgt) ** 2) / var_safe)
    loss = (loss + 0.5 * math.log(2.0 * math.pi)) if full else loss
    return _reduce(loss, reduction)


def hinge_loss(
    inputs: ArrayInput,
    targets: ArrayInput,
    reduction: LossReduction = "none",
    margin: float = 1.0,
) -> array:
    """Compute the hinge loss `max(0, margin - inputs * targets)`.

    Args:
        inputs: Predicted values.
        targets: Binary targets in `{-1, 1}`.
        reduction: Reduction mode (`'none'`, `'mean'`, `'sum'`).
        margin: Hinge margin (default `1.0`).

    Returns:
        Hinge loss `array`.

    """
    inp = _as_array(inputs)
    tgt = _as_array(targets)
    loss = mx.maximum(0.0, margin - inp * tgt)
    return _reduce(loss, reduction)


def huber_loss(
    inputs: ArrayInput,
    targets: ArrayInput,
    delta: float = 1.0,
    reduction: LossReduction = "none",
) -> array:
    """Compute the Huber loss between `inputs` and `targets`.

    Args:
        inputs: Predicted values.
        targets: Ground-truth values.
        delta: Threshold at which to change between delta-scaled L1 and L2 loss.
        reduction: Reduction mode (`'none'`, `'mean'`, `'sum'`).

    Returns:
        Huber loss `array`.

    """
    inp = _as_array(inputs)
    tgt = _as_array(targets)
    abs_err = mx.abs(inp - tgt)
    loss = mx.where(abs_err < delta, 0.5 * (abs_err**2), delta * (abs_err - 0.5 * delta))
    return _reduce(loss, reduction)


def kl_div_loss(
    inputs: ArrayInput,
    targets: ArrayInput,
    axis: int = -1,
    reduction: LossReduction = "none",
) -> array:
    """Compute the Kullback-Leibler divergence loss `sum(targets * (log(targets) - inputs))`.

    Args:
        inputs: Log-probabilities of the predicted distribution.
        targets: Probabilities of the target distribution.
        axis: Distribution axis (used when reducing across classes).
        reduction: Reduction mode (`'none'`, `'mean'`, `'sum'`).

    Returns:
        KL divergence loss `array`.

    """
    _ = axis
    inp = _as_array(inputs)
    tgt = _as_array(targets)
    loss = mx.where(tgt > 0.0, tgt * (mx.log(tgt) - inp), 0.0)
    return _reduce(loss, reduction)


def l1_loss(
    predictions: ArrayInput,
    targets: ArrayInput,
    reduction: LossReduction = "mean",
) -> array:
    """Compute the L1 (mean absolute error) loss `|predictions - targets|`.

    Args:
        predictions: Predicted values.
        targets: Target values.
        reduction: Reduction mode (`'none'`, `'mean'`, `'sum'`).

    Returns:
        L1 loss `array`.

    """
    pred = _as_array(predictions)
    tgt = _as_array(targets)
    loss = mx.abs(pred - tgt)
    return _reduce(loss, reduction)


def log_cosh_loss(
    inputs: ArrayInput,
    targets: ArrayInput,
    reduction: LossReduction = "none",
) -> array:
    """Compute the log-cosh loss `log(cosh(inputs - targets))`.

    Args:
        inputs: Predicted values.
        targets: Target values.
        reduction: Reduction mode (`'none'`, `'mean'`, `'sum'`).

    Returns:
        Log-cosh loss `array`.

    """
    inp = _as_array(inputs)
    tgt = _as_array(targets)
    abs_p = mx.abs(inp - tgt)
    loss = abs_p + mx.log1p(mx.exp(-2.0 * abs_p)) - math.log(2.0)
    return _reduce(loss, reduction)


def margin_ranking_loss(
    inputs1: ArrayInput,
    inputs2: ArrayInput,
    targets: ArrayInput,
    margin: float = 0.0,
    reduction: LossReduction = "none",
) -> array:
    """Compute the margin ranking loss `max(0, -targets * (inputs1 - inputs2) + margin)`.

    Args:
        inputs1: Scores for the first input.
        inputs2: Scores for the second input.
        targets: Ranking labels (`1` or `-1`).
        margin: Ranking margin offset.
        reduction: Reduction mode (`'none'`, `'mean'`, `'sum'`).

    Returns:
        Margin ranking loss `array`.

    """
    i1 = _as_array(inputs1)
    i2 = _as_array(inputs2)
    tgt = _as_array(targets)
    loss = mx.maximum(0.0, -tgt * (i1 - i2) + margin)
    return _reduce(loss, reduction)


def mse_loss(
    predictions: ArrayInput,
    targets: ArrayInput,
    reduction: LossReduction = "mean",
) -> array:
    """Compute the Mean Squared Error (L2) loss `(predictions - targets)^2`.

    Args:
        predictions: Predicted values.
        targets: Target values.
        reduction: Reduction mode (`'none'`, `'mean'`, `'sum'`).

    Returns:
        MSE loss `array`.

    """
    pred = _as_array(predictions)
    tgt = _as_array(targets)
    loss = (pred - tgt) ** 2
    return _reduce(loss, reduction)


def nll_loss(
    inputs: ArrayInput,
    targets: ArrayInput,
    axis: int = -1,
    reduction: LossReduction = "none",
    weight: ArrayInput | None = None,
    ignore_index: int = -100,
) -> array:
    """Compute the negative log-likelihood loss.

    Args:
        inputs: Log-probabilities of shape `(..., C)`.
        targets: Integer class indices.
        axis: Class dimension axis.
        reduction: Reduction mode (`'none'`, `'mean'`, `'sum'`).
        weight: Optional per-class weight vector.
        ignore_index: Target class value that is ignored and does not contribute to the loss.

    Returns:
        Negative log-likelihood loss `array`.

    """
    inp = _as_array(inputs)
    tgt = _as_array(targets).astype(int32)
    tgt_exp = mx.expand_dims(tgt, axis=axis)
    loss = -mx.squeeze(mx.take_along_axis(inp, tgt_exp, axis=axis), axis=axis)
    if weight is not None:
        w_arr = _as_array(weight)
        w = mx.squeeze(mx.take_along_axis(w_arr, tgt_exp, axis=-1), axis=-1)
        loss = loss * w
    else:
        w = mx.ones_like(loss)
    mask = tgt != ignore_index
    loss = mx.where(mask, loss, 0.0)
    w = mx.where(mask, w, 0.0)
    if reduction == "mean":
        return mx.sum(loss) / mx.maximum(mx.sum(w), 1e-8)
    if reduction == "sum":
        return mx.sum(loss)
    return loss


def smooth_l1_loss(
    predictions: ArrayInput,
    targets: ArrayInput,
    beta: float = 1.0,
    reduction: LossReduction = "mean",
) -> array:
    """Compute the smooth L1 loss with transition threshold `beta`.

    Args:
        predictions: Predicted values.
        targets: Ground-truth values.
        beta: L1-to-L2 transition point.
        reduction: Reduction mode (`'none'`, `'mean'`, `'sum'`).

    Returns:
        Smooth L1 loss `array`.

    """
    pred = _as_array(predictions)
    tgt = _as_array(targets)
    abs_err = mx.abs(pred - tgt)
    loss = mx.where(abs_err < beta, 0.5 * (abs_err**2) / beta, abs_err - 0.5 * beta)
    return _reduce(loss, reduction)


def triplet_loss(
    anchors: ArrayInput,
    positives: ArrayInput,
    negatives: ArrayInput,
    axis: int = -1,
    p: float = 2.0,
    margin: float = 1.0,
    eps: float = 1e-6,
    reduction: LossReduction = "none",
) -> array:
    """Compute the triplet margin loss for `(anchors, positives, negatives)`.

    Args:
        anchors: Anchor embeddings.
        positives: Positive embeddings.
        negatives: Negative embeddings.
        axis: Embedding feature dimension.
        p: Norm degree for pairwise distance.
        margin: Triplet margin.
        eps: Small epsilon added before root for numerical stability.
        reduction: Reduction mode (`'none'`, `'mean'`, `'sum'`).

    Returns:
        Triplet loss `array`.

    """
    anc = _as_array(anchors)
    pos = _as_array(positives)
    neg = _as_array(negatives)
    d_pos = (mx.sum(mx.abs(anc - pos) ** p, axis=axis) + eps) ** (1.0 / p)
    d_neg = (mx.sum(mx.abs(anc - neg) ** p, axis=axis) + eps) ** (1.0 / p)
    loss = mx.maximum(d_pos - d_neg + margin, 0.0)
    return _reduce(loss, reduction)


__all__ = [
    "Reduction",
    "binary_cross_entropy",
    "cosine_similarity_loss",
    "cross_entropy",
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
