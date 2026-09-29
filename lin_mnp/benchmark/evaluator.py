"""Leakage-aware evaluation and hierarchical HEtoSGEBench aggregation.

Metrics are first computed across spots for every ``(slide, gene)``.  The
cross-slide reporting path then follows the strict, unweighted hierarchy

``slice metric -> patient mean -> gene mean -> gene-panel summary``.

This prevents patients with more sections from receiving extra weight.  The
confidence interval for a panel-level mean likewise resamples whole patients,
repeats the same hierarchy inside every bootstrap replicate, and therefore
preserves within-patient dependence between slides and genes.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Set, Tuple

import numpy as np
import pandas as pd

from .metrics import METRIC_NAMES, compute_metrics_with_reasons


_IDENTITY_COLUMNS = ["slide_id", "patient_id", "gene", "gene_set"]


def _to_numpy(value: Any) -> np.ndarray:
    """Convert NumPy-, Torch-, and array-like values without importing Torch."""

    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    return np.asarray(value)


_MISSING = object()


def _field(record: Any, name: str, default: Any = _MISSING) -> Any:
    if isinstance(record, Mapping) and name in record:
        return record[name]
    if hasattr(record, name):
        return getattr(record, name)
    if default is not _MISSING:
        return default
    raise ValueError(f"Slide record is missing required field {name!r}")


def _first_field(record: Any, names: Sequence[str]) -> Any:
    for name in names:
        if isinstance(record, Mapping) and name in record:
            return record[name]
        if hasattr(record, name):
            return getattr(record, name)
    raise ValueError(f"Slide record is missing all aliases {tuple(names)!r}")


def _canonical_gene_set(name: Any) -> Optional[str]:
    key = str(name).strip().lower()
    if key in {"all", "all_genes", "all genes"}:
        return "all"
    if key in {"hvg", "hvgs"}:
        return "HVG"
    if key in {"svg", "svgs"}:
        return "SVG"
    return None


def _case_insensitive_item(mapping: Mapping[Any, Any], name: str) -> Any:
    target = name.lower()
    for key, value in mapping.items():
        if str(key).lower() == target:
            return value
    return _MISSING


def _gene_names_from_spec(spec: Any, gene_names: Sequence[str]) -> Set[str]:
    if spec is None or spec is _MISSING:
        return set()
    if isinstance(spec, str):
        return {spec}
    if isinstance(spec, Mapping):
        # A gene -> bool dictionary is a useful compact manifest form.
        return {
            str(gene)
            for gene, selected in spec.items()
            if isinstance(selected, (bool, np.bool_)) and bool(selected)
        }
    array = _to_numpy(spec)
    if array.ndim == 1 and array.dtype.kind == "b":
        if array.size != len(gene_names):
            raise ValueError("Boolean gene-set mask must match len(gene_names)")
        return {gene for gene, selected in zip(gene_names, array) if bool(selected)}
    if array.ndim == 0:
        return {str(array.item())}
    return {str(value) for value in array.reshape(-1).tolist()}


def _select_slide_spec(spec: Any, slide_id: str, gene_names: Sequence[str]) -> Set[str]:
    if not isinstance(spec, Mapping):
        return _gene_names_from_spec(spec, gene_names)

    # Distinguish a gene->bool mask from a slide->gene-list mapping.
    if spec and all(isinstance(value, (bool, np.bool_)) for value in spec.values()):
        return _gene_names_from_spec(spec, gene_names)
    for candidate in (slide_id, "*", "global", "default"):
        selected = _case_insensitive_item(spec, candidate)
        if selected is not _MISSING:
            return _gene_names_from_spec(selected, gene_names)
    return set()


def _resolve_gene_sets(
    gene_sets: Any,
    slide_gene_sets: Any,
    slide_id: str,
    gene_names: Sequence[str],
) -> Tuple[Dict[str, Set[str]], Set[str]]:
    """Resolve global and slice-specific HVG/SVG specifications."""

    resolved: Dict[str, Set[str]] = {"all": set(gene_names)}
    requested: Set[str] = {"all"}

    if gene_sets is not None and hasattr(gene_sets, "genes_for"):
        for set_name in ("HVG", "SVG"):
            try:
                resolved[set_name] = set(
                    str(gene) for gene in gene_sets.genes_for(set_name, slide_id)
                )
            except (KeyError, ValueError):
                resolved[set_name] = set()
            requested.add(set_name)
    elif isinstance(gene_sets, Mapping):
        # Form 1: {"HVG": [...], "SVG": {slide_id: [...]}}
        for raw_name, spec in gene_sets.items():
            set_name = _canonical_gene_set(raw_name)
            if set_name in {"HVG", "SVG"}:
                resolved[set_name] = _select_slide_spec(spec, slide_id, gene_names)
                requested.add(set_name)
        # Form 2: {slide_id: {"HVG": [...], "SVG": [...]}}
        per_slide = _case_insensitive_item(gene_sets, slide_id)
        if isinstance(per_slide, Mapping):
            for raw_name, spec in per_slide.items():
                set_name = _canonical_gene_set(raw_name)
                if set_name in {"HVG", "SVG"}:
                    resolved[set_name] = _gene_names_from_spec(spec, gene_names)
                    requested.add(set_name)

    # A record-level specification overrides the corresponding global set.
    if isinstance(slide_gene_sets, Mapping):
        for raw_name, spec in slide_gene_sets.items():
            set_name = _canonical_gene_set(raw_name)
            if set_name in {"HVG", "SVG"}:
                resolved[set_name] = _select_slide_spec(spec, slide_id, gene_names)
                requested.add(set_name)

    known = set(gene_names)
    for set_name in list(resolved):
        resolved[set_name].intersection_update(known)
    return resolved, requested


def _normalise_mask(mask: Any, n_spots: int, n_genes: int) -> np.ndarray:
    if mask is None:
        return np.ones((n_spots, n_genes), dtype=bool)
    array = _to_numpy(mask).astype(bool, copy=False)
    if array.shape == (n_spots, n_genes):
        return array
    if array.shape == (n_spots,):
        return np.broadcast_to(array[:, None], (n_spots, n_genes))
    if array.shape == (n_genes,):
        return np.broadcast_to(array[None, :], (n_spots, n_genes))
    raise ValueError(
        "mask must have shape [spots, genes], [spots], or [genes]; "
        f"received {array.shape}"
    )


def _finite_summary(values: Iterable[float]) -> Dict[str, float | int]:
    array = np.asarray(list(values), dtype=np.float64)
    array = array[np.isfinite(array)]
    count = int(array.size)
    return {
        "mean": float(np.mean(array)) if count else float("nan"),
        "median": float(np.median(array)) if count else float("nan"),
        "std": float(np.std(array, ddof=1)) if count >= 2 else float("nan"),
        "n_valid": count,
    }


def _percentile_interval(replicates: Sequence[float]) -> Tuple[float, float]:
    values = np.asarray(replicates, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return float("nan"), float("nan")
    lower, upper = np.percentile(values, [2.5, 97.5])
    return float(lower), float(upper)


def _bootstrap_unit_mean(
    values: np.ndarray,
    bootstrap_samples: int,
    rng: np.random.Generator,
) -> Tuple[float, float]:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    # A non-parametric bootstrap needs at least two units. At the per-gene
    # level these units are patients; at the descriptive per-slide level they
    # are genes.
    if values.size < 2 or bootstrap_samples < 1:
        return float("nan"), float("nan")
    indices = rng.integers(0, values.size, size=(bootstrap_samples, values.size))
    replicates = np.mean(values[indices], axis=1)
    return _percentile_interval(replicates)


def _empty_per_slide_gene() -> pd.DataFrame:
    columns: List[str] = _IDENTITY_COLUMNS + ["n_spots", "n_masked_spots"]
    for metric in METRIC_NAMES:
        columns.extend([metric, f"{metric}_valid", f"{metric}_invalid_reason"])
    return pd.DataFrame(columns=columns)


def _build_per_slide_gene(
    slides: Sequence[Any],
    gene_sets: Any,
) -> Tuple[pd.DataFrame, Set[str], Set[str]]:
    rows: List[Dict[str, Any]] = []
    requested_sets: Set[str] = {"all"}
    all_genes: Set[str] = set()
    seen_slides: Set[str] = set()

    for record in slides:
        slide_id = str(_field(record, "slide_id"))
        patient_id = str(_field(record, "patient_id"))
        if slide_id in seen_slides:
            raise ValueError(f"Duplicate slide_id {slide_id!r}")
        seen_slides.add(slide_id)

        true = _to_numpy(_first_field(record, ("true", "expression", "y"))).astype(
            np.float64, copy=False
        )
        pred = _to_numpy(_first_field(record, ("pred", "prediction", "y_pred"))).astype(
            np.float64, copy=False
        )
        if true.ndim != 2 or pred.ndim != 2 or true.shape != pred.shape:
            raise ValueError(
                f"Slide {slide_id!r}: true and pred must have identical [spots, genes] shape"
            )
        gene_names = tuple(str(gene) for gene in _field(record, "gene_names"))
        if len(gene_names) != true.shape[1]:
            raise ValueError(f"Slide {slide_id!r}: len(gene_names) does not match matrices")
        if len(set(gene_names)) != len(gene_names):
            raise ValueError(f"Slide {slide_id!r}: gene_names must be unique")
        all_genes.update(gene_names)

        mask = _normalise_mask(
            _field(record, "mask", None), n_spots=true.shape[0], n_genes=true.shape[1]
        )
        resolved_sets, slide_requested = _resolve_gene_sets(
            gene_sets,
            _field(record, "gene_sets", None),
            slide_id,
            gene_names,
        )
        requested_sets.update(slide_requested)

        for gene_index, gene in enumerate(gene_names):
            selected = mask[:, gene_index]
            bundle = compute_metrics_with_reasons(
                true[selected, gene_index], pred[selected, gene_index]
            )
            base: Dict[str, Any] = {
                "slide_id": slide_id,
                "patient_id": patient_id,
                "gene": gene,
                "n_spots": int(true.shape[0]),
                "n_masked_spots": int(np.count_nonzero(selected)),
            }
            for metric in METRIC_NAMES:
                value = float(bundle.values[metric])
                base[metric] = value
                base[f"{metric}_valid"] = bool(np.isfinite(value))
                base[f"{metric}_invalid_reason"] = bundle.invalid_reasons[metric]
            for set_name, members in resolved_sets.items():
                if gene in members:
                    row = dict(base)
                    row["gene_set"] = set_name
                    rows.append(row)

    if not rows:
        return _empty_per_slide_gene(), requested_sets, all_genes
    frame = pd.DataFrame.from_records(rows)
    ordered_columns = _empty_per_slide_gene().columns.tolist()
    frame = frame.loc[:, ordered_columns]
    return frame.sort_values(_IDENTITY_COLUMNS, kind="stable").reset_index(drop=True), requested_sets, all_genes


def _aggregate_patient_gene(per_slide_gene: pd.DataFrame) -> pd.DataFrame:
    identifier_columns = ["patient_id", "gene", "gene_set"]
    columns = identifier_columns + ["n_slides"]
    for metric in METRIC_NAMES:
        columns.extend([metric, f"{metric}_n_valid_slides"])
    if per_slide_gene.empty:
        return pd.DataFrame(columns=columns)

    rows: List[Dict[str, Any]] = []
    grouped = per_slide_gene.groupby(identifier_columns, sort=True, dropna=False)
    for identifiers, group in grouped:
        row: Dict[str, Any] = dict(zip(identifier_columns, identifiers))
        row["n_slides"] = int(group["slide_id"].nunique())
        for metric in METRIC_NAMES:
            values = group[metric].to_numpy(dtype=np.float64)
            finite = values[np.isfinite(values)]
            row[metric] = float(np.mean(finite)) if finite.size else float("nan")
            row[f"{metric}_n_valid_slides"] = int(finite.size)
        rows.append(row)
    return pd.DataFrame.from_records(rows, columns=columns)


def _summarise_per_gene(
    per_patient_gene: pd.DataFrame,
    bootstrap_samples: int,
    rng: np.random.Generator,
) -> pd.DataFrame:
    columns = [
        "gene_set",
        "gene",
        "metric",
        "mean",
        "median",
        "std",
        "n_valid",
        "ci_low",
        "ci_high",
    ]
    if per_patient_gene.empty:
        return pd.DataFrame(columns=columns)
    rows: List[Dict[str, Any]] = []
    for (gene_set, gene), group in per_patient_gene.groupby(
        ["gene_set", "gene"], sort=True, dropna=False
    ):
        for metric in METRIC_NAMES:
            values = group[metric].to_numpy(dtype=np.float64)
            summary = _finite_summary(values)
            ci_low, ci_high = _bootstrap_unit_mean(values, bootstrap_samples, rng)
            rows.append(
                {
                    "gene_set": gene_set,
                    "gene": gene,
                    "metric": metric,
                    **summary,
                    "ci_low": ci_low,
                    "ci_high": ci_high,
                }
            )
    return pd.DataFrame.from_records(rows, columns=columns)


def _summarise_per_slide(
    per_slide_gene: pd.DataFrame,
    bootstrap_samples: int,
    rng: np.random.Generator,
) -> pd.DataFrame:
    columns = [
        "gene_set",
        "slide_id",
        "patient_id",
        "metric",
        "mean",
        "median",
        "std",
        "n_valid",
        "ci_low",
        "ci_high",
    ]
    if per_slide_gene.empty:
        return pd.DataFrame(columns=columns)
    rows: List[Dict[str, Any]] = []
    for (gene_set, slide_id, patient_id), group in per_slide_gene.groupby(
        ["gene_set", "slide_id", "patient_id"], sort=True, dropna=False
    ):
        for metric in METRIC_NAMES:
            values = group[metric].to_numpy(dtype=np.float64)
            # This is a descriptive within-slide gene bootstrap, not a claim
            # of independent biological samples. Patient-cluster resampling is
            # used for all cross-slide inferential summaries below.
            ci_low, ci_high = _bootstrap_unit_mean(values, bootstrap_samples, rng)
            rows.append(
                {
                    "gene_set": gene_set,
                    "slide_id": slide_id,
                    "patient_id": patient_id,
                    "metric": metric,
                    **_finite_summary(values),
                    "ci_low": ci_low,
                    "ci_high": ci_high,
                }
            )
    return pd.DataFrame.from_records(rows, columns=columns)


def _bootstrap_panel_mean(
    per_patient_gene: pd.DataFrame,
    gene_set: str,
    metric: str,
    bootstrap_samples: int,
    rng: np.random.Generator,
) -> Tuple[float, float]:
    selected = per_patient_gene.loc[
        per_patient_gene["gene_set"] == gene_set,
        ["patient_id", "gene", metric],
    ].copy()
    selected = selected[np.isfinite(selected[metric].to_numpy(dtype=np.float64))]
    patients = selected["patient_id"].drop_duplicates().to_numpy()
    if patients.size < 2 or bootstrap_samples < 1:
        return float("nan"), float("nan")

    by_patient = {
        patient: selected.loc[selected["patient_id"] == patient, ["gene", metric]]
        for patient in patients
    }
    replicates: List[float] = []
    for _ in range(bootstrap_samples):
        sampled = patients[rng.integers(0, patients.size, size=patients.size)]
        # Concatenating repeated patient frames intentionally gives a sampled
        # cluster its bootstrap multiplicity before gene means are recomputed.
        replicate_rows = pd.concat([by_patient[patient] for patient in sampled], ignore_index=True)
        gene_means = replicate_rows.groupby("gene", sort=False)[metric].mean()
        replicates.append(float(gene_means.mean()))
    return _percentile_interval(replicates)


def _summarise_panels(
    per_gene: pd.DataFrame,
    per_patient_gene: pd.DataFrame,
    requested_sets: Set[str],
    bootstrap_samples: int,
    rng: np.random.Generator,
) -> pd.DataFrame:
    columns = [
        "gene_set",
        "metric",
        "mean",
        "median",
        "std",
        "n_valid",
        "n_patients",
        "ci_low",
        "ci_high",
    ]
    rows: List[Dict[str, Any]] = []
    order = [name for name in ("all", "HVG", "SVG") if name in requested_sets]
    for gene_set in order:
        for metric in METRIC_NAMES:
            selected = per_gene.loc[
                (per_gene["gene_set"] == gene_set) & (per_gene["metric"] == metric), "mean"
            ]
            stats = _finite_summary(selected.to_numpy(dtype=np.float64))
            patient_rows = per_patient_gene.loc[per_patient_gene["gene_set"] == gene_set]
            if not patient_rows.empty:
                patient_rows = patient_rows[
                    np.isfinite(patient_rows[metric].to_numpy(dtype=np.float64))
                ]
            n_patients = int(patient_rows["patient_id"].nunique()) if not patient_rows.empty else 0
            ci_low, ci_high = _bootstrap_panel_mean(
                per_patient_gene,
                gene_set,
                metric,
                bootstrap_samples,
                rng,
            )
            rows.append(
                {
                    "gene_set": gene_set,
                    "metric": metric,
                    **stats,
                    "n_patients": n_patients,
                    "ci_low": ci_low,
                    "ci_high": ci_high,
                }
            )
    return pd.DataFrame.from_records(rows, columns=columns)


@dataclass
class EvaluationResult:
    """Five requested evaluation tables plus reproducibility metadata."""

    per_slide_gene: pd.DataFrame
    per_patient_gene: pd.DataFrame
    per_gene: pd.DataFrame
    per_slide: pd.DataFrame
    summary: pd.DataFrame
    metadata: Dict[str, Any] = field(default_factory=dict)

    def save(self, output_dir: str | Path) -> None:
        """Write all tables as CSV and the evaluation contract as JSON."""

        destination = Path(output_dir)
        destination.mkdir(parents=True, exist_ok=True)
        for name in (
            "per_slide_gene",
            "per_patient_gene",
            "per_gene",
            "per_slide",
            "summary",
        ):
            getattr(self, name).to_csv(destination / f"{name}.csv", index=False)
        with (destination / "evaluation_metadata.json").open("w", encoding="utf-8") as handle:
            json.dump(self.metadata, handle, indent=2, sort_keys=True, ensure_ascii=False)


def evaluate_slides(
    slides: Sequence[Any],
    gene_sets: Any = None,
    bootstrap_samples: int = 500,
    bootstrap_seed: int = 17,
) -> EvaluationResult:
    """Evaluate a collection of slide dictionaries or attribute-style objects.

    Required fields are ``slide_id``, ``patient_id``, ``true``, ``pred``, and
    ``gene_names``.  ``true``/``pred`` must be ``[spots, genes]``.  Optional
    ``mask`` values may be ``[spots, genes]``, ``[spots]``, or ``[genes]``.

    ``gene_sets`` accepts global lists, slice-specific mappings, or a
    ``FrozenGeneSets``-like object exposing ``genes_for(set_name, slide_id)``.
    A record-level ``gene_sets`` mapping may override global membership.
    """

    slides = list(slides)
    if not slides:
        raise ValueError("At least one slide record is required")
    if bootstrap_samples < 0:
        raise ValueError("bootstrap_samples must be non-negative")
    rng = np.random.default_rng(bootstrap_seed)
    per_slide_gene, requested_sets, genes = _build_per_slide_gene(slides, gene_sets)
    per_patient_gene = _aggregate_patient_gene(per_slide_gene)
    per_gene = _summarise_per_gene(per_patient_gene, bootstrap_samples, rng)
    per_slide = _summarise_per_slide(per_slide_gene, bootstrap_samples, rng)
    summary = _summarise_panels(
        per_gene,
        per_patient_gene,
        requested_sets,
        bootstrap_samples,
        rng,
    )

    metadata: Dict[str, Any] = {
        "benchmark_compatibility": "HEtoSGEBench per-(slide,gene) definitions",
        "aggregation": "unweighted slice -> patient -> gene -> gene-set summary",
        "bootstrap": (
            "patient-cluster percentile 95% CI of the mean for per_gene and summary; "
            "descriptive within-slide gene percentile bootstrap for per_slide"
        ),
        "per_slide_bootstrap": (
            "resample valid genes within each slide/gene_set; requires >=2 valid genes"
        ),
        "bootstrap_samples": int(bootstrap_samples),
        "bootstrap_seed": int(bootstrap_seed),
        "single_patient_ci": (
            "NaN for patient-cluster per_gene/summary CIs; per_slide uses gene bootstrap"
        ),
        "metrics": list(METRIC_NAMES),
        "gene_sets": [name for name in ("all", "HVG", "SVG") if name in requested_sets],
        "n_slides": int(per_slide_gene["slide_id"].nunique()),
        "n_patients": int(per_slide_gene["patient_id"].nunique()),
        "n_genes": int(len(genes)),
        "nonfinite_policy": "record invalid reason, exclude from aggregates",
    }
    return EvaluationResult(
        per_slide_gene=per_slide_gene,
        per_patient_gene=per_patient_gene,
        per_gene=per_gene,
        per_slide=per_slide,
        summary=summary,
        metadata=metadata,
    )


class Evaluator:
    """Stateful convenience wrapper around :func:`evaluate_slides`."""

    def __init__(
        self,
        gene_sets: Any = None,
        bootstrap_samples: int = 500,
        bootstrap_seed: int = 17,
    ) -> None:
        self.gene_sets = gene_sets
        self.bootstrap_samples = bootstrap_samples
        self.bootstrap_seed = bootstrap_seed

    def evaluate(self, slides: Sequence[Any], gene_sets: Any = None) -> EvaluationResult:
        return evaluate_slides(
            slides,
            gene_sets=self.gene_sets if gene_sets is None else gene_sets,
            bootstrap_samples=self.bootstrap_samples,
            bootstrap_seed=self.bootstrap_seed,
        )

    __call__ = evaluate


evaluate_predictions = evaluate_slides


__all__ = [
    "EvaluationResult",
    "Evaluator",
    "evaluate_predictions",
    "evaluate_slides",
]
