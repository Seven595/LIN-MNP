"""Transparent HESpotEx-style metrics for spot-level expression predictions.

Metrics are evaluated independently for every gene across the spots of one
section. The published HESpotEx Methods specify independent min--max scaling
of prediction and ground truth before MSE. The released training code shifts
each predicted section by its global minimum before returning it; we mirror
that non-negative postprocessing before interpreting each gene as a spatial
probability distribution for KL divergence.

The paper does not release its final metric implementation. A small, explicit
epsilon is therefore used here to make KL finite for biological zero counts.
"""

from __future__ import annotations

import numpy as np


DEFAULT_KL_EPSILON = 1e-10


def _matching_matrices(
    target: np.ndarray, prediction: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    target64 = np.asarray(target, dtype=np.float64)
    prediction64 = np.asarray(prediction, dtype=np.float64)
    if (
        target64.ndim != 2
        or prediction64.ndim != 2
        or target64.shape != prediction64.shape
    ):
        raise ValueError("metric inputs must have matching [spots, genes] shapes")
    if target64.shape[0] < 2 or target64.shape[1] < 1:
        raise ValueError("metric inputs require at least two spots and one gene")
    if not np.isfinite(target64).all() or not np.isfinite(prediction64).all():
        raise ValueError("metric inputs must be finite")
    return target64, prediction64


def per_gene_pcc(target: np.ndarray, prediction: np.ndarray) -> np.ndarray:
    """Ordinary Pearson correlation across spots, independently per gene."""

    target64, prediction64 = _matching_matrices(target, prediction)
    target_centered = target64 - target64.mean(axis=0, keepdims=True)
    prediction_centered = prediction64 - prediction64.mean(axis=0, keepdims=True)
    denominator = np.sqrt(
        np.square(target_centered).sum(axis=0)
        * np.square(prediction_centered).sum(axis=0)
    )
    return np.divide(
        np.sum(target_centered * prediction_centered, axis=0),
        denominator,
        out=np.full(target64.shape[1], np.nan, dtype=np.float64),
        where=denominator > 1e-12,
    )


def _column_minmax(values: np.ndarray) -> np.ndarray:
    minima = values.min(axis=0, keepdims=True)
    ranges = values.max(axis=0, keepdims=True) - minima
    return np.divide(
        values - minima,
        ranges,
        out=np.zeros_like(values, dtype=np.float64),
        where=ranges > 0,
    )


def per_gene_minmax_mse(target: np.ndarray, prediction: np.ndarray) -> np.ndarray:
    """MSE after separate per-gene [0, 1] scaling, as in HESpotEx Methods."""

    target64, prediction64 = _matching_matrices(target, prediction)
    difference = _column_minmax(target64) - _column_minmax(prediction64)
    return np.square(difference).mean(axis=0)


def per_gene_spatial_kl(
    target: np.ndarray,
    prediction: np.ndarray,
    *,
    epsilon: float = DEFAULT_KL_EPSILON,
) -> np.ndarray:
    """Return per-gene ``D_KL(target || prediction)`` across spatial spots.

    Target values must already be non-negative. Prediction values are shifted
    once by the section-wide minimum, matching the postprocessing at the end of
    the released HESpotEx ``train.main``. Epsilon is added to every spatial bin
    before normalization so zero-valued bins and all-zero genes are defined.
    """

    target64, prediction64 = _matching_matrices(target, prediction)
    if (target64 < 0).any():
        raise ValueError("KL target must be non-negative")
    if not np.isfinite(epsilon) or epsilon <= 0:
        raise ValueError("KL epsilon must be finite and positive")
    prediction_nonnegative = prediction64 - prediction64.min()
    target_mass = target64 + float(epsilon)
    prediction_mass = prediction_nonnegative + float(epsilon)
    target_probability = target_mass / target_mass.sum(axis=0, keepdims=True)
    prediction_probability = prediction_mass / prediction_mass.sum(
        axis=0, keepdims=True
    )
    return np.sum(
        target_probability
        * (np.log(target_probability) - np.log(prediction_probability)),
        axis=0,
    )
