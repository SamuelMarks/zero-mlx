"""Loss functions for neural networks."""

import numpy as np
from .. import array


def _reduce(x: np.ndarray, reduction: str = "none") -> array:
    """Reduce output array."""
    if reduction == "mean":
        return array(np.mean(x))
    elif reduction == "sum":
        return array(np.sum(x))
    return array(x)


def cosine_similarity_loss(
    x1: array, x2: array, axis: int = 1, eps: float = 1e-08, reduction: str = "none"
) -> array:
    """Computes the cosine similarity between the two inputs."""
    d1 = x1.data
    d2 = x2.data
    dot_product = np.sum(d1 * d2, axis=axis)
    norm1 = np.linalg.norm(d1, axis=axis)
    norm2 = np.linalg.norm(d2, axis=axis)
    similarity = dot_product / (np.maximum(norm1 * norm2, eps))
    return _reduce(similarity, reduction)


def gaussian_nll_loss(
    inputs: array,
    targets: array,
    vars: array,
    full: bool = False,
    eps: float = 1e-06,
    reduction: str = "none",
) -> array:
    """Computes the negative log likelihood loss for a Gaussian distribution."""
    i = inputs.data
    t = targets.data
    v = vars.data
    loss = 0.5 * (np.log(np.maximum(v, eps)) + ((i - t) ** 2) / np.maximum(v, eps))
    if full:
        loss += 0.5 * np.log(2 * np.pi)
    return _reduce(loss, reduction)


def hinge_loss(inputs: array, targets: array, reduction: str = "none") -> array:
    """Computes the hinge loss between inputs and targets."""
    loss = np.maximum(0, 1 - inputs.data * targets.data)
    return _reduce(loss, reduction)


def huber_loss(
    inputs: array, targets: array, delta: float = 1.0, reduction: str = "none"
) -> array:
    """Computes the Huber loss between inputs and targets."""
    diff = np.abs(inputs.data - targets.data)
    loss = np.where(diff < delta, 0.5 * (diff**2), delta * diff - 0.5 * (delta**2))
    return _reduce(loss, reduction)


def kl_div_loss(
    inputs: array, targets: array, axis: int = -1, reduction: str = "none"
) -> array:
    """Computes the Kullback-Leibler divergence loss."""
    # Assuming inputs are log-probabilities and targets are probabilities
    i = inputs.data
    t = targets.data
    # Avoid log(0) in targets
    mask = t > 0
    loss = np.zeros_like(t)
    loss[mask] = t[mask] * (np.log(t[mask]) - i[mask])
    return _reduce(loss, reduction)


def l1_loss(predictions: array, targets: array, reduction: str = "mean") -> array:
    """Computes the L1 loss."""
    loss = np.abs(predictions.data - targets.data)
    return _reduce(loss, reduction)


def log_cosh_loss(inputs: array, targets: array, reduction: str = "none") -> array:
    """Computes the log cosh loss between inputs and targets."""
    diff = inputs.data - targets.data
    loss = np.log(np.cosh(diff))
    return _reduce(loss, reduction)


def margin_ranking_loss(
    inputs1: array,
    inputs2: array,
    targets: array,
    margin: float = 0.0,
    reduction: str = "none",
) -> array:
    """Calculate the margin ranking loss."""
    loss = np.maximum(0, -targets.data * (inputs1.data - inputs2.data) + margin)
    return _reduce(loss, reduction)


def mse_loss(predictions: array, targets: array, reduction: str = "mean") -> array:
    """Computes the mean squared error loss."""
    loss = (predictions.data - targets.data) ** 2
    return _reduce(loss, reduction)


def nll_loss(
    inputs: array, targets: array, axis: int = -1, reduction: str = "none"
) -> array:
    """Computes the negative log likelihood loss."""
    # inputs are log probabilities, targets are class indices
    i = inputs.data
    t = targets.data

    # We create a mask to gather the log probabilities corresponding to the target classes
    # Assuming standard NLL: inputs (N, C), targets (N,)
    indices = tuple(np.indices(t.shape))
    loss = -i[(*indices, t)]

    return _reduce(loss, reduction)


def smooth_l1_loss(
    predictions: array, targets: array, beta: float = 1.0, reduction: str = "mean"
) -> array:
    """Computes the smooth L1 loss."""
    diff = np.abs(predictions.data - targets.data)
    loss = np.where(diff < beta, 0.5 * (diff**2) / beta, diff - 0.5 * beta)
    return _reduce(loss, reduction)


def triplet_loss(
    anchors: array,
    positives: array,
    negatives: array,
    axis: int = -1,
    p: int = 2,
    margin: float = 1.0,
    eps: float = 1e-06,
    reduction: str = "none",
) -> array:
    """Computes the triplet loss for a set of anchor, positive, and negative samples."""
    a = anchors.data
    pos = positives.data
    neg = negatives.data

    dist_pos = np.linalg.norm(a - pos, ord=p, axis=axis)
    dist_neg = np.linalg.norm(a - neg, ord=p, axis=axis)

    loss = np.maximum(0, dist_pos - dist_neg + margin)
    return _reduce(loss, reduction)
