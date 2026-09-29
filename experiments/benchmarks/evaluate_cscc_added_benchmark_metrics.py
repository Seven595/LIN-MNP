#!/usr/bin/env python3
"""Evaluate the five added cSCC methods with the Section 9.2 metric suite.

All spot-level prediction identities and available checksums are verified before
any fixed-997 target shard is opened.  Aggregation is section/gene -> equally
weighted patient/gene -> equally weighted patient mean -> gene-panel mean.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
from typing import Any, Mapping

import numpy as np
import pandas as pd

from lin_mnp.paths import ARCHIVE_ROOT, DATA_ROOT, RESULTS_ROOT

ARCHIVE = ARCHIVE_ROOT
LABELS = DATA_ROOT / "cscc/labels"
ADDITIONAL_RESULTS = ARCHIVE / "benchmark/additional_comparators"
REMAINING_RESULTS = ARCHIVE / "benchmark/remaining_comparators"
STFLOW = (
    ARCHIVE
    / "legacy/component_analysis/stflow997_common_raw/ciga_224px"
)
OUTPUT = RESULTS_ROOT / "additional_comparators/cscc/evaluation"
METRIC_CODE = ARCHIVE / "tools/compute_unified_protocol_metrics.py"
METRICS = (
    "pcc", "mi", "auc_0", "mse_minmax", "ssim", "kl_spatial_softmax",
    "mse_native", "nrmse_range", "nrmse_sd",
)
_METRIC_MODULE = None


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(payload: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp-{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def metrics_module():
    global _METRIC_MODULE
    if _METRIC_MODULE is not None:
        return _METRIC_MODULE
    spec = importlib.util.spec_from_file_location("added_cscc_unified_metrics", METRIC_CODE)
    if spec is None or spec.loader is None:
        raise ImportError(METRIC_CODE)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    _METRIC_MODULE = module
    return module


def decode(value: np.ndarray) -> tuple[str, ...]:
    return tuple(map(str, np.asarray(value).reshape(-1)))


def section_zscore(value: np.ndarray) -> np.ndarray:
    value = np.asarray(value, dtype=np.float64)
    scale = value.std(axis=0, keepdims=True)
    return np.divide(
        value - value.mean(axis=0, keepdims=True),
        scale,
        out=np.zeros_like(value),
        where=scale > 1e-12,
    )


def align_prediction(
    prediction: np.ndarray,
    prediction_genes: tuple[str, ...],
    prediction_barcodes: tuple[str, ...],
    target_genes: tuple[str, ...],
    target_barcodes: tuple[str, ...],
) -> np.ndarray:
    if len(set(prediction_genes)) != len(prediction_genes):
        raise ValueError("duplicated prediction genes")
    if len(set(prediction_barcodes)) != len(prediction_barcodes):
        raise ValueError("duplicated prediction barcodes")
    gene_lookup = {value.casefold(): index for index, value in enumerate(prediction_genes)}
    barcode_lookup = {value: index for index, value in enumerate(prediction_barcodes)}
    if set(gene_lookup) != {value.casefold() for value in target_genes}:
        raise ValueError("prediction/target gene identities differ")
    if set(barcode_lookup) != set(target_barcodes):
        raise ValueError("prediction/target barcode identities differ")
    gene_order = [gene_lookup[value.casefold()] for value in target_genes]
    barcode_order = [barcode_lookup[value] for value in target_barcodes]
    return np.asarray(prediction, dtype=np.float64)[np.ix_(barcode_order, gene_order)]


def load_prediction(task: Mapping[str, Any]) -> tuple[np.ndarray, tuple[str, ...], tuple[str, ...]]:
    with np.load(task["prediction"], allow_pickle=False) as archive:
        if task["format"] == "combined":
            sections = np.asarray(archive["sections"]).astype(str)
            active = sections == str(task["slide_id"])
            if not active.any():
                raise ValueError(f"section absent from prediction: {task['slide_id']}")
            return (
                np.asarray(archive["prediction"])[active],
                decode(archive["genes"]),
                decode(np.asarray(archive["barcodes"])[active]),
            )
        return (
            np.asarray(archive["prediction"]),
            decode(archive["genes"]),
            decode(archive["barcodes"]),
        )


def worker(task: Mapping[str, Any]) -> pd.DataFrame:
    unified = metrics_module()
    with np.load(LABELS / f"{task['slide_id']}.npz", allow_pickle=False) as archive:
        target_genes = decode(archive["genes"])
        target_barcodes = decode(archive["barcodes"])
        patient_id = str(archive["patient_id"].item())
        target = np.asarray(archive["log1p_raw_counts"], dtype=np.float64)
    if patient_id != task["patient_id"]:
        raise ValueError("patient identity differs")
    if task["target_transform"] == "section_zscore_log1p_raw":
        target = section_zscore(target)
    prediction, prediction_genes, prediction_barcodes = load_prediction(task)
    prediction = align_prediction(
        prediction, prediction_genes, prediction_barcodes, target_genes, target_barcodes
    )
    if prediction.shape != target.shape or not np.isfinite(prediction).all():
        raise ValueError("invalid aligned prediction")
    # The shared helper also computes JSD, which is undefined for the negative
    # values in a method-native z-scored target.  JSD is not part of Section 9.2;
    # suppress that unused intermediate's warning and do not persist it.
    with np.errstate(invalid="ignore", divide="ignore"):
        values = unified.benchmark_style_metrics(target, prediction)
    return pd.DataFrame(
        {
            "method": task["method"],
            "fold_id": task["fold_id"],
            "slide_id": task["slide_id"],
            "patient_id": patient_id,
            "gene": target_genes,
            "pcc": values["pcc_recalculated"],
            "mi": values["mi"],
            "auc_0": values["auc_0"],
            "mse_minmax": unified.per_gene_minmax_mse(target, prediction),
            "ssim": values["ssim"],
            "kl_spatial_softmax": values["kl_spatial_softmax"],
            "mse_native": values["mse_native"],
            "nrmse_range": values["nrmse_range"],
            "nrmse_sd": values["nrmse_sd"],
        }
    )


def verified_tasks() -> list[dict[str, Any]]:
    tasks: list[dict[str, Any]] = []
    for method_id, method in (
        ("holotea", "HoloTea"), ("spahgc", "SpaHGC"), ("stag", "STAG")
    ):
        for fold_id in range(4):
            root = ADDITIONAL_RESULTS / method_id / f"fold{fold_id}"
            metadata_path = root / "run_metadata.json"
            complete = json.loads((root / "complete.json").read_text())
            if sha256(metadata_path) != complete["run_metadata_sha256"]:
                raise ValueError(f"metadata checksum failure: {method}/{fold_id}")
            metadata = json.loads(metadata_path.read_text())
            prediction = root / "predictions.npz"
            if sha256(prediction) != metadata["prediction_sha256"]:
                raise ValueError(f"prediction checksum failure: {method}/{fold_id}")
            if metadata["outer_test_labels_opened_before_prediction"]:
                raise ValueError(f"outer target was opened: {method}/{fold_id}")
            with np.load(prediction, allow_pickle=False) as archive:
                sections = sorted(set(np.asarray(archive["sections"]).astype(str)))
                patients = np.asarray(archive["patients"]).astype(str)
                section_array = np.asarray(archive["sections"]).astype(str)
                for slide_id in sections:
                    patient_values = set(patients[section_array == slide_id])
                    if len(patient_values) != 1:
                        raise ValueError("combined prediction patient identity failure")
                    tasks.append(
                        {
                            "method": method, "fold_id": fold_id,
                            "slide_id": slide_id, "patient_id": patient_values.pop(),
                            "prediction": str(prediction), "format": "combined",
                            "target_transform": "section_zscore_log1p_raw",
                        }
                    )

    for fold_id in range(4):
        root = REMAINING_RESULTS / "hespotex_fixed997" / f"fold{fold_id}"
        metadata_path = root / "run_metadata.json"
        complete = json.loads((root / "complete.json").read_text())
        if sha256(metadata_path) != complete["run_metadata_sha256"]:
            raise ValueError(f"HESpotEx metadata checksum failure: {fold_id}")
        metadata = json.loads(metadata_path.read_text())
        if metadata["outer_test_expression_opened"]:
            raise ValueError("HESpotEx outer target was opened")
        for slide_id, expected_hash in metadata["prediction_sha256"].items():
            prediction = root / "predictions" / f"{slide_id}.npz"
            if sha256(prediction) != expected_hash:
                raise ValueError(f"HESpotEx prediction checksum failure: {slide_id}")
            with np.load(prediction, allow_pickle=False) as archive:
                patient_id = str(archive["patient"].item())
            tasks.append(
                {
                    "method": "HESpotEx", "fold_id": fold_id,
                    "slide_id": slide_id, "patient_id": patient_id,
                    "prediction": str(prediction), "format": "single",
                    "target_transform": "section_zscore_log1p_raw",
                }
            )

    index = pd.read_csv(
        STFLOW / "canonical_ridge_common_contract_evaluation/detail/prediction_artifact_index.csv"
    )
    index = index.loc[index.method == "Ciga + STFlow"].copy()
    if len(index) != 12 or bool(index.contains_truth.any()):
        raise ValueError("STFlow prediction index is incomplete or contains truth")
    for row in index.itertuples(index=False):
        prediction = Path(row.path)
        if sha256(prediction) != row.sha256:
            raise ValueError(f"STFlow prediction checksum failure: {row.sample_id}")
        tasks.append(
            {
                "method": "STFlow", "fold_id": int(row.fold_id),
                "slide_id": str(row.sample_id), "patient_id": str(row.patient_id),
                "prediction": str(prediction), "format": "single",
                "target_transform": "log1p_raw",
            }
        )
    if len(tasks) != 60:
        raise ValueError(f"expected 60 method/slide tasks, found {len(tasks)}")
    return tasks


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    tasks = verified_tasks()  # No target shard has been opened before this point.
    frames = []
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(worker, task) for task in tasks]
        for index, future in enumerate(as_completed(futures), start=1):
            frames.append(future.result())
            if index % 10 == 0 or index == len(futures):
                print(f"[metrics] {index}/{len(futures)}", flush=True)
    detail = pd.concat(frames, ignore_index=True)
    patient_gene = detail.groupby(
        ["method", "patient_id", "gene"], as_index=False, sort=False
    )[list(METRICS)].mean()
    gene = patient_gene.groupby(
        ["method", "gene"], as_index=False, sort=False
    )[list(METRICS)].mean()
    summary = gene.groupby("method", as_index=False, sort=False)[list(METRICS)].mean()
    summary = summary.sort_values("pcc", ascending=False).reset_index(drop=True)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    detail.to_csv(output / "per_section_gene_metrics.csv.gz", index=False)
    patient_gene.to_csv(output / "per_patient_gene_metrics.csv.gz", index=False)
    gene.to_csv(output / "per_gene_metrics.csv.gz", index=False)
    summary.to_csv(output / "summary.csv", index=False)
    metadata = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "prediction_artifacts_verified_before_truth_access": True,
        "tasks": len(tasks),
        "methods": sorted(summary.method.tolist()),
        "genes": int(detail.gene.nunique()),
        "patients": int(detail.patient_id.nunique()),
        "aggregation": "section/gene -> patient/gene -> equal four-patient gene -> panel",
        "target_transforms": {
            "STFlow": "log1p(raw count)",
            "HoloTea/SpaHGC/STAG/HESpotEx": "within-section zscore(log1p(raw count))",
        },
        "metric_implementation": str(METRIC_CODE),
    }
    atomic_json(metadata, output / "metadata.json")
    manifests = {}
    for name in ("per_section_gene_metrics.csv.gz", "per_patient_gene_metrics.csv.gz", "per_gene_metrics.csv.gz", "summary.csv", "metadata.json"):
        manifests[name] = sha256(output / name)
    atomic_json({"files": manifests}, output / "result_manifest.json")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
