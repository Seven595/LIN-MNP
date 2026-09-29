"""HEtoSGEBench-compatible spot-level expression metrics.

The benchmark computes every metric for one ``(slide, gene)`` vector across
spots.  In particular, its quantity named spatial SSIM is a *global* SSIM on
two independently quantised expression vectors; coordinates and image-style
local windows are not part of that definition.

The implementation intentionally preserves a few details of the reference R
code that are easy to miss:

* mutual information uses ``infotheo``-style equal-frequency bins and is
  normalised by the geometric mean of the two marginal entropies;
* only the prediction is clamped for Jensen--Shannon divergence;
* both range- and sample-standard-deviation-normalised RMSE are reported; and
* SSIM uses R ``cut(x, 256) - 1`` semantics after independent max scaling.

Undefined quantities are returned as ``NaN`` rather than silently replaced by
zero.  :func:`compute_metrics_with_reasons` additionally records why each
quantity is undefined so downstream aggregation can skip it explicitly.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Mapping, Sequence, Tuple

import numpy as np


METRIC_NAMES: Tuple[str, ...] = (
    "pcc",
    "mi",
    "jsd",
    "nrmse_range",
    "nrmse_sd",
    "ssim",
)


def _as_pair(
    true: Sequence[float] | np.ndarray,
    pred: Sequence[float] | np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    true_array = np.asarray(true, dtype=np.float64)
    pred_array = np.asarray(pred, dtype=np.float64)
    if true_array.ndim != 1 or pred_array.ndim != 1:
        raise ValueError("Metrics require one-dimensional spot vectors")
    if true_array.shape != pred_array.shape:
        raise ValueError("true and pred must have the same shape")
    return true_array, pred_array


def pearson_correlation(
    true: Sequence[float] | np.ndarray,
    pred: Sequence[float] | np.ndarray,
) -> float:
    """Ordinary gene-wise Pearson correlation across spots."""

    true_array, pred_array = _as_pair(true, pred)
    if (
        true_array.size < 2
        or not np.all(np.isfinite(true_array))
        or not np.all(np.isfinite(pred_array))
        or np.var(true_array, ddof=1) == 0.0
        or np.var(pred_array, ddof=1) == 0.0
    ):
        return float("nan")
    return float(np.corrcoef(true_array, pred_array)[0, 1])


def threshold_auc(
    true: Sequence[float] | np.ndarray,
    pred: Sequence[float] | np.ndarray,
    cutoff: float = 0.0,
) -> float:
    """HEtoSGEBench AUC for detecting ``true > cutoff`` from predictions.

    This is the Mann--Whitney form of ROC AUC with average ranks for ties.  It
    matches ``pROC::auc(response, predicted, direction='<')`` used by the
    benchmark while avoiding a heavyweight classifier dependency.  AUC is
    undefined when either class is absent or either vector is non-finite.
    """

    true_array, pred_array = _as_pair(true, pred)
    if (
        true_array.size == 0
        or not np.all(np.isfinite(true_array))
        or not np.all(np.isfinite(pred_array))
    ):
        return float("nan")
    positive = true_array > float(cutoff)
    n_positive = int(np.count_nonzero(positive))
    n_negative = int(positive.size - n_positive)
    if n_positive == 0 or n_negative == 0:
        return float("nan")

    order = np.argsort(pred_array, kind="stable")
    sorted_values = pred_array[order]
    ranks = np.empty(pred_array.size, dtype=np.float64)
    start = 0
    while start < sorted_values.size:
        stop = start + 1
        while stop < sorted_values.size and sorted_values[stop] == sorted_values[start]:
            stop += 1
        # One-based average rank, matching the Wilcoxon/Mann--Whitney formula.
        ranks[order[start:stop]] = 0.5 * ((start + 1) + stop)
        start = stop
    rank_sum_positive = float(np.sum(ranks[positive]))
    u_statistic = rank_sum_positive - n_positive * (n_positive + 1) / 2.0
    return float(u_statistic / (n_positive * n_negative))


def _equal_frequency_codes(values: np.ndarray, n_bins: int) -> np.ndarray:
    """Clone ``infotheo::discretize(..., disc='equalfreq')`` binning.

    The C++ implementation used by ``infotheo`` builds boundaries from the
    sorted observations and assigns a tied value to the first boundary greater
    than or equal to it. Callers reject ``n < 10``, for which the benchmark's
    requested minimum of ten bins is not safely defined.
    """

    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1:
        raise ValueError("Equal-frequency discretisation requires a vector")
    if values.size == 0:
        return np.empty(0, dtype=np.int64)
    n_bins = max(1, min(int(n_bins), values.size))
    ordered = np.sort(values)
    frequency, remainder = divmod(values.size, n_bins)
    splitpoint = frequency - 1
    boundaries = np.empty(n_bins, dtype=np.float64)
    for index in range(n_bins - 1):
        if remainder > 0:
            boundaries[index] = ordered[splitpoint + 1]
            remainder -= 1
        else:
            boundaries[index] = ordered[splitpoint]
        splitpoint += frequency
    boundaries[-1] = ordered[-1] + 0.01
    return np.searchsorted(boundaries, values, side="left").astype(np.int64)


def _entropy(codes: np.ndarray) -> float:
    if codes.size == 0:
        return float("nan")
    _, counts = np.unique(codes, return_counts=True)
    probabilities = counts.astype(np.float64) / codes.size
    return float(-np.sum(probabilities * np.log(probabilities)))


def normalized_mutual_information(
    true: Sequence[float] | np.ndarray,
    pred: Sequence[float] | np.ndarray,
) -> float:
    """Equal-frequency, entropy-normalised empirical mutual information.

    HEtoSGEBench uses ``B = floor(max(10, n_spots ** (1/3)))`` bins for each
    vector independently, followed by ``MI / sqrt(H_true * H_pred)``.
    """

    true_array, pred_array = _as_pair(true, pred)
    n_spots = true_array.size
    if (
        n_spots < 10
        or not np.all(np.isfinite(true_array))
        or not np.all(np.isfinite(pred_array))
    ):
        return float("nan")
    n_bins = int(max(10.0, n_spots ** (1.0 / 3.0)))
    true_codes = _equal_frequency_codes(true_array, n_bins)
    pred_codes = _equal_frequency_codes(pred_array, n_bins)
    true_entropy = _entropy(true_codes)
    pred_entropy = _entropy(pred_codes)
    denominator = np.sqrt(true_entropy * pred_entropy)
    if not np.isfinite(denominator) or denominator <= 0.0:
        return float("nan")
    joint_codes = np.column_stack((true_codes, pred_codes))
    _, joint_counts = np.unique(joint_codes, axis=0, return_counts=True)
    joint_probabilities = joint_counts.astype(np.float64) / n_spots
    joint_entropy = -np.sum(joint_probabilities * np.log(joint_probabilities))
    # Tiny negative values can arise only from floating-point cancellation.
    mutual_information = max(0.0, true_entropy + pred_entropy - joint_entropy)
    return float(mutual_information / denominator)


def jensen_shannon_divergence(
    true: Sequence[float] | np.ndarray,
    pred: Sequence[float] | np.ndarray,
) -> float:
    """Base-2 JSD after clamping the prediction, but not the truth, at zero."""

    true_array, pred_array = _as_pair(true, pred)
    if (
        true_array.size == 0
        or not np.all(np.isfinite(true_array))
        or not np.all(np.isfinite(pred_array))
        or np.any(true_array < 0.0)
    ):
        return float("nan")

    n_spots = true_array.size
    true_sum = float(np.sum(true_array))
    if true_sum == 0.0:
        true_probability = np.full(n_spots, 1.0 / n_spots)
    elif true_sum > 0.0:
        true_probability = true_array / true_sum
    else:  # Defensive guard; negative entries were already rejected above.
        return float("nan")

    nonnegative_pred = np.maximum(pred_array, 0.0)
    pred_sum = float(np.sum(nonnegative_pred))
    if pred_sum == 0.0:
        pred_probability = np.full(n_spots, 1.0 / n_spots)
    else:
        pred_probability = nonnegative_pred / pred_sum

    midpoint = 0.5 * (true_probability + pred_probability)

    def kl_base_two(probability: np.ndarray) -> float:
        positive = probability > 0.0
        return float(
            np.sum(
                probability[positive]
                * np.log2(probability[positive] / midpoint[positive])
            )
        )

    return float(0.5 * kl_base_two(true_probability) + 0.5 * kl_base_two(pred_probability))


def normalized_rmse(
    true: Sequence[float] | np.ndarray,
    pred: Sequence[float] | np.ndarray,
    normalization: str = "range",
) -> float:
    """RMSE divided by the truth range or its R-style sample SD (``ddof=1``)."""

    true_array, pred_array = _as_pair(true, pred)
    if (
        true_array.size == 0
        or not np.all(np.isfinite(true_array))
        or not np.all(np.isfinite(pred_array))
    ):
        return float("nan")
    rmse = float(np.sqrt(np.mean(np.square(pred_array - true_array))))
    key = normalization.lower()
    if key == "range":
        denominator = float(np.max(true_array) - np.min(true_array))
    elif key in {"sd", "std", "standard_deviation"}:
        if true_array.size < 2:
            return float("nan")
        denominator = float(np.std(true_array, ddof=1))
    else:
        raise ValueError("normalization must be 'range' or 'sd'")
    if denominator == 0.0 or not np.isfinite(denominator):
        return float("nan")
    return float(rmse / denominator)


def _r_cut_256_codes(values: np.ndarray) -> np.ndarray:
    """Return zero-based codes matching R ``cut(values, 256) - 1``."""

    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1:
        raise ValueError("R cut compatibility requires a vector")
    if values.size == 0:
        return np.empty(0, dtype=np.int64)
    lower = float(np.min(values))
    upper = float(np.max(values))
    span = upper - lower
    if span == 0.0:
        span = abs(lower) if lower != 0.0 else 1.0
        breaks = np.linspace(lower - span / 1000.0, upper + span / 1000.0, 257)
    else:
        breaks = np.linspace(lower, upper, 257)
        breaks[0] = lower - span / 1000.0
        breaks[-1] = upper + span / 1000.0
    # R cut defaults to right=TRUE, so a value exactly on an internal break is
    # assigned to the interval on its left.  labels=FALSE is one-based.
    codes = np.searchsorted(breaks, values, side="left") - 1
    return np.clip(codes, 0, 255).astype(np.int64)


def _ssim_quantize(values: np.ndarray) -> np.ndarray:
    maximum = float(np.max(values))
    scaled = np.zeros_like(values) if maximum == 0.0 else values / maximum
    return _r_cut_256_codes(scaled).astype(np.float64)


def spatial_ssim(
    true: Sequence[float] | np.ndarray,
    pred: Sequence[float] | np.ndarray,
) -> float:
    """HEtoSGEBench global vector SSIM (despite the historical name).

    No spot coordinates or neighbourhood windows enter the published metric.
    Negative values are valid and are deliberately not clipped.
    """

    true_array, pred_array = _as_pair(true, pred)
    if (
        true_array.size < 2
        or not np.all(np.isfinite(true_array))
        or not np.all(np.isfinite(pred_array))
    ):
        return float("nan")
    true_codes = _ssim_quantize(true_array)
    pred_codes = _ssim_quantize(pred_array)
    true_mean = float(np.mean(true_codes))
    pred_mean = float(np.mean(pred_codes))
    true_variance = float(np.var(true_codes, ddof=1))
    pred_variance = float(np.var(pred_codes, ddof=1))
    covariance = float(np.cov(true_codes, pred_codes, ddof=1)[0, 1])
    c1 = (0.01 * 255.0) ** 2
    c2 = (0.03 * 255.0) ** 2
    numerator = (2.0 * true_mean * pred_mean + c1) * (2.0 * covariance + c2)
    denominator = (
        (true_mean**2 + pred_mean**2 + c1)
        * (true_variance + pred_variance + c2)
    )
    if denominator == 0.0 or not np.isfinite(denominator):
        return float("nan")
    return float(numerator / denominator)


@dataclass(frozen=True)
class MetricBundle:
    """Metric values plus explicit invalidity reasons."""

    values: Mapping[str, float]
    invalid_reasons: Mapping[str, str]

    @property
    def valid(self) -> Dict[str, bool]:
        return {name: bool(np.isfinite(self.values[name])) for name in METRIC_NAMES}


def compute_metrics_with_reasons(
    true: Sequence[float] | np.ndarray,
    pred: Sequence[float] | np.ndarray,
) -> MetricBundle:
    """Compute all benchmark metrics and annotate every non-finite result."""

    true_array, pred_array = _as_pair(true, pred)
    values: Dict[str, float] = {name: float("nan") for name in METRIC_NAMES}
    reasons: Dict[str, str] = {name: "" for name in METRIC_NAMES}

    if true_array.size == 0:
        return MetricBundle(values, {name: "empty_input" for name in METRIC_NAMES})
    if not np.all(np.isfinite(true_array)) or not np.all(np.isfinite(pred_array)):
        return MetricBundle(values, {name: "nonfinite_input" for name in METRIC_NAMES})

    values.update(
        pcc=pearson_correlation(true_array, pred_array),
        mi=normalized_mutual_information(true_array, pred_array),
        jsd=jensen_shannon_divergence(true_array, pred_array),
        nrmse_range=normalized_rmse(true_array, pred_array, "range"),
        nrmse_sd=normalized_rmse(true_array, pred_array, "sd"),
        ssim=spatial_ssim(true_array, pred_array),
    )

    if true_array.size < 2:
        reasons["pcc"] = "insufficient_spots"
        reasons["nrmse_sd"] = "insufficient_spots"
        reasons["ssim"] = "insufficient_spots"
    elif np.var(true_array, ddof=1) == 0.0:
        reasons["pcc"] = "constant_true"
        reasons["nrmse_sd"] = "constant_true"
    if true_array.size >= 2 and np.var(pred_array, ddof=1) == 0.0:
        reasons["pcc"] = "constant_pred"
    if np.max(true_array) - np.min(true_array) == 0.0:
        reasons["nrmse_range"] = "zero_true_range"
    if np.any(true_array < 0.0):
        reasons["jsd"] = "negative_true_mass"

    # MI can be undefined even for nonconstant raw vectors when ties collapse
    # all equal-frequency bins into one empirical category.
    if not np.isfinite(values["mi"]):
        reasons["mi"] = (
            "insufficient_spots" if true_array.size < 10 else "zero_discrete_entropy"
        )
    for name, value in values.items():
        if not np.isfinite(value) and not reasons[name]:
            reasons[name] = "nonfinite_result"
    return MetricBundle(values, reasons)


def compute_metrics(
    true: Sequence[float] | np.ndarray,
    pred: Sequence[float] | np.ndarray,
) -> Dict[str, float]:
    """Return the six benchmark metrics as a plain dictionary."""

    return dict(compute_metrics_with_reasons(true, pred).values)


# Concise aliases used by reports and external callers.
gene_wise_pearson = pearson_correlation
mutual_information = normalized_mutual_information
js_divergence = jensen_shannon_divergence
nrmse = normalized_rmse
global_ssim = spatial_ssim


__all__ = [
    "METRIC_NAMES",
    "MetricBundle",
    "compute_metrics",
    "compute_metrics_with_reasons",
    "gene_wise_pearson",
    "global_ssim",
    "jensen_shannon_divergence",
    "js_divergence",
    "mutual_information",
    "normalized_mutual_information",
    "normalized_rmse",
    "nrmse",
    "pearson_correlation",
    "spatial_ssim",
    "threshold_auc",
]
