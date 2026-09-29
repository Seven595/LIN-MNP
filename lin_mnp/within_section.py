"""Reusable within-section morphology propagation utilities.

These functions support the HEST, STImage and HESpotEx transfer analyses. They
share Lin-MNP's descriptor fusion and temperature but exclude the query spot
from a gallery restricted to one section.
"""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np

from lin_mnp import model


def radial_pool_weights(
    grid_size: int = 7,
    stride: float = 32.0,
    radius: float = 72.0,
    transition_width: float = 16.0,
) -> np.ndarray:
    """Return normalized soft-disk weights for a square token grid."""

    center = (grid_size - 1) / 2.0
    coordinate = (np.arange(grid_size, dtype=np.float64) - center) * stride
    vertical, horizontal = np.meshgrid(coordinate, coordinate, indexing="ij")
    distance = np.sqrt(horizontal**2 + vertical**2)
    weights = 1.0 / (1.0 + np.exp((distance - radius) / transition_width))
    weights /= weights.sum()
    return weights.reshape(-1).astype(np.float32)


def mean_gene_pcc_and_mse(
    target: np.ndarray, prediction: np.ndarray
) -> dict[str, float]:
    """Return STFlow-style mean gene-wise PCC and element-wise MSE."""

    left = np.asarray(target, dtype=np.float64)
    right = np.asarray(prediction, dtype=np.float64)
    if left.ndim != 2 or left.shape != right.shape:
        raise ValueError("target and prediction must have matching [spots, genes] shapes")
    centered_left = left - left.mean(axis=0, keepdims=True)
    centered_right = right - right.mean(axis=0, keepdims=True)
    denominator = np.sqrt(
        np.square(centered_left).sum(axis=0)
        * np.square(centered_right).sum(axis=0)
    )
    correlation = np.divide(
        (centered_left * centered_right).sum(axis=0),
        denominator,
        out=np.full(left.shape[1], np.nan, dtype=np.float64),
        where=denominator > 1e-12,
    )
    return {
        "pcc": float(np.nanmean(correlation)),
        "mse": float(np.square(left - right).mean()),
    }


def apply(
    prediction: np.ndarray,
    context: np.ndarray,
    local: np.ndarray,
    neighbour_count: int,
    *,
    fusion: str = "borda",
    weighting: str = "softmax",
) -> tuple[np.ndarray, int]:
    """Apply one morphology-propagation step within a single section."""

    context = model.normalize_rows(context)
    local = model.normalize_rows(local)
    count = min(int(neighbour_count), len(context) - 1)
    if count < 1:
        raise ValueError("within-section propagation requires at least two spots")
    index = np.arange(len(context))
    similarity = model.variant_similarity(context, local, index, index, fusion)
    similarity[index, index] = -np.inf
    selected = model.top(similarity, count)
    scores = np.take_along_axis(similarity, selected, axis=1)
    if weighting == "softmax":
        weights = model.softmax(scores)
    elif weighting == "uniform":
        weights = np.full(scores.shape, 1.0 / count, dtype=np.float32)
    else:
        raise ValueError(f"unknown neighbour weighting: {weighting}")
    values = np.asarray(prediction, dtype=np.float32)
    propagated = np.einsum(
        "nk,nkg->ng", weights, values[selected], optimize=True
    ).astype(np.float32)
    return propagated, count


def apply_grid(
    prediction: np.ndarray,
    context: np.ndarray,
    local: np.ndarray,
    neighbour_counts: Iterable[int],
    *,
    fusion: str = "borda",
    weighting: str = "softmax",
) -> dict[int, np.ndarray]:
    """Evaluate several neighbour counts while sorting similarities once."""

    counts = tuple(int(value) for value in neighbour_counts)
    if not counts or min(counts) < 1:
        raise ValueError("neighbour counts must be positive")
    context = model.normalize_rows(context)
    local = model.normalize_rows(local)
    if len(context) < 2:
        return {count: np.asarray(prediction, dtype=np.float32) for count in counts}
    index = np.arange(len(context))
    similarity = model.variant_similarity(context, local, index, index, fusion)
    similarity[index, index] = -np.inf
    ceiling = min(max(counts), len(context) - 1)
    ordered = model.top(similarity, ceiling)
    scores = np.take_along_axis(similarity, ordered, axis=1)
    values = np.asarray(prediction, dtype=np.float32)
    result = {}
    for requested in counts:
        count = min(requested, len(context) - 1)
        selected = ordered[:, :count]
        selected_scores = scores[:, :count]
        if weighting == "softmax":
            weights = model.softmax(selected_scores)
        elif weighting == "uniform":
            weights = np.full(selected_scores.shape, 1.0 / count, dtype=np.float32)
        else:
            raise ValueError(f"unknown neighbour weighting: {weighting}")
        result[requested] = np.einsum(
            "nk,nkg->ng", weights, values[selected], optimize=True
        ).astype(np.float32)
    return result


__all__ = ["apply", "apply_grid", "mean_gene_pcc_and_mse", "radial_pool_weights"]
