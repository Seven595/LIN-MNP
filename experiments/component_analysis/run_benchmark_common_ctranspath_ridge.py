#!/usr/bin/env python3
"""Run one canonical Benchmark CTransPath + affine Ridge baseline.

The experiment contract is deliberately invariant to comparator identity:

* cSCC fixed 997-gene panel;
* official ``log1p(raw count)`` targets;
* strict four-patient outer CV (2 train + 1 validation + 1 test patient);
* all three outer-test sections per patient;
* Ridge alpha selected only on the validation patient.

The held-out expression shards are not opened by this training script.  They
are opened only later by the independent Benchmark evaluator, after prediction
artifacts and their hashes have been materialised.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from lin_mnp.paths import CACHE_ROOT, DATA_ROOT, REPOSITORY_ROOT, RESULTS_ROOT

PROJECT = REPOSITORY_ROOT
COHORT = DATA_ROOT / "cscc/cohort"
DERIVED = DATA_ROOT / "cscc/derived"
FOLDS = COHORT / "splits/heto_official_cv4/folds.json"
GENES = (
    DERIVED
    / "heto_official_cv4/fold0/locked997/prepared/genes.txt"
)
RAW_LABELS = DATA_ROOT / "cscc/labels"
DESCRIPTORS = (
    CACHE_ROOT / "cscc/ctranspath_multiscale/slides"
)
DEFAULT_OUTPUT = RESULTS_ROOT / "component_analysis/common_contract"
ALPHAS = (
    0.01,
    0.1,
    1.0,
    10.0,
    100.0,
    1_000.0,
    10_000.0,
    100_000.0,
    1_000_000.0,
    10_000_000.0,
    100_000_000.0,
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def correlations(target: np.ndarray, prediction: np.ndarray) -> np.ndarray:
    target = target - target.mean(0, keepdims=True)
    prediction = prediction - prediction.mean(0, keepdims=True)
    denominator = np.sqrt(
        np.square(target).sum(0) * np.square(prediction).sum(0)
    )
    return np.divide(
        (target * prediction).sum(0),
        denominator,
        out=np.full(target.shape[1], np.nan),
        where=denominator > 1e-12,
    )


class SpectralRidge:
    """One affine map with training-patient feature standardisation."""

    def __init__(self, features: np.ndarray, target: np.ndarray):
        self.x_mean = features.mean(0)
        self.x_scale = features.std(0)
        self.x_scale[~np.isfinite(self.x_scale) | (self.x_scale < 1e-6)] = 1.0
        self.y_mean = target.mean(0)
        x = torch.from_numpy((features - self.x_mean) / self.x_scale)
        y = torch.from_numpy(target - self.y_mean)
        gram = x.T @ x
        rhs = x.T @ y
        self.values, self.vectors = torch.linalg.eigh(gram)
        self.projected_rhs = self.vectors.T @ rhs

    def predict(self, features: np.ndarray, alpha: float) -> np.ndarray:
        weights = self.vectors @ (
            self.projected_rhs / (self.values[:, None] + float(alpha))
        )
        x = torch.from_numpy((features - self.x_mean) / self.x_scale)
        return (x @ weights).numpy() + self.y_mean


def read_folds() -> dict[int, Mapping[str, Any]]:
    payload = json.loads(FOLDS.read_text())
    folds = {int(row["fold_id"]): row for row in payload["folds"]}
    if set(folds) != {0, 1, 2, 3}:
        raise ValueError("Benchmark contract requires exactly four outer folds")
    return folds


def read_gene_panel() -> tuple[str, ...]:
    genes = tuple(line.strip() for line in GENES.read_text().splitlines() if line.strip())
    if len(genes) != 997 or len(set(genes)) != 997:
        raise ValueError("invalid fixed-997 gene panel")
    return genes


def load_features(
    scale: int, descriptor_dir: Path = DESCRIPTORS
) -> tuple[dict[str, np.ndarray], dict[str, dict[str, Any]]]:
    features: dict[str, np.ndarray] = {}
    identities: dict[str, dict[str, Any]] = {}
    for path in sorted(descriptor_dir.glob("*.npz")):
        with np.load(path, allow_pickle=False) as archive:
            if bool(archive["target_data_accessed"].item()):
                raise ValueError(f"target-contaminated descriptor: {path}")
            sample = str(archive["sample_id"].item())
            indices = np.flatnonzero(archive["crop_sizes_px"].astype(int) == scale)
            if len(indices) != 1:
                raise ValueError(f"missing unique {scale}-px descriptor: {path}")
            matrix = np.asarray(
                archive["pooled"][:, int(indices[0]), :], dtype=np.float64
            )
            barcodes = tuple(archive["barcodes"].astype(str))
            patient = str(archive["patient_id"].item())
        if matrix.shape != (len(barcodes), 768) or not np.isfinite(matrix).all():
            raise ValueError(f"invalid descriptor matrix: {path}/{matrix.shape}")
        features[sample] = matrix
        identities[sample] = {
            "barcodes": barcodes,
            "patient_id": patient,
            "path": str(path),
            "sha256": sha256_file(path),
        }
    if len(features) != 12 or sum(map(len, features.values())) != 8671:
        raise ValueError("incomplete Benchmark descriptor coverage")
    return features, identities


def load_target(
    sample: str,
    expected_barcodes: Sequence[str],
    canonical_genes: Sequence[str],
) -> np.ndarray:
    """Open one explicitly supervised label shard and align its gene case."""
    path = RAW_LABELS / f"{sample}.npz"
    with np.load(path, allow_pickle=False) as archive:
        barcodes = tuple(archive["barcodes"].astype(str))
        source_genes = tuple(archive["genes"].astype(str))
        target = np.asarray(archive["log1p_raw_counts"], dtype=np.float64)
    if barcodes != tuple(expected_barcodes):
        raise ValueError(f"target/descriptor barcode mismatch: {sample}")
    lookup = {gene.casefold(): index for index, gene in enumerate(source_genes)}
    order = np.asarray([lookup[gene.casefold()] for gene in canonical_genes])
    target = target[:, order]
    if target.shape != (len(barcodes), 997) or not np.isfinite(target).all():
        raise ValueError(f"invalid target: {sample}/{target.shape}")
    return target


def validation_score(
    targets: Mapping[str, np.ndarray], predictions: Mapping[str, np.ndarray]
) -> float:
    return float(
        np.mean(
            [
                np.nanmean(correlations(targets[sample], prediction))
                for sample, prediction in predictions.items()
            ]
        )
    )


def run(scale: int, output_root: Path, descriptor_dir: Path = DESCRIPTORS) -> None:
    torch.set_default_dtype(torch.float64)
    torch.set_num_threads(32)
    torch.set_num_interop_threads(1)
    genes = read_gene_panel()
    features, identities = load_features(scale, descriptor_dir)
    method_id = f"ctranspath{scale}_affine_ridge_common_raw"
    for fold_id, fold in sorted(read_folds().items()):
        output = output_root / method_id / f"fold{fold_id}"
        if (output / "complete.json").is_file():
            continue
        if output.exists():
            raise FileExistsError(f"partial immutable output: {output}")
        output.mkdir(parents=True)
        started = time.perf_counter()
        train_ids = tuple(map(str, fold["train_sample_ids"]))
        val_ids = tuple(map(str, fold["val_sample_ids"]))
        test_ids = tuple(map(str, fold["test_sample_ids"]))
        # Test labels are intentionally absent from this dictionary.
        supervised_ids = train_ids + val_ids
        targets = {
            sample: load_target(
                sample, identities[sample]["barcodes"], genes
            )
            for sample in supervised_ids
        }
        ridge = SpectralRidge(
            np.concatenate([features[sample] for sample in train_ids]),
            np.concatenate([targets[sample] for sample in train_ids]),
        )
        candidates = []
        for alpha in ALPHAS:
            predictions = {
                sample: ridge.predict(features[sample], alpha) for sample in val_ids
            }
            candidates.append(
                {
                    "alpha": alpha,
                    "validation_mean_slide_gene_pcc": validation_score(
                        {sample: targets[sample] for sample in val_ids}, predictions
                    ),
                }
            )
        candidates.sort(
            key=lambda row: (-row["validation_mean_slide_gene_pcc"], row["alpha"])
        )
        selected = candidates[0]
        prediction_rows = []
        for sample in test_ids:
            prediction = ridge.predict(
                features[sample], float(selected["alpha"])
            ).astype(np.float32)
            path = output / "predictions" / f"{sample}.npz"
            path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                path,
                prediction=prediction,
                genes=np.asarray(genes),
                barcodes=np.asarray(identities[sample]["barcodes"]),
                sample_id=np.asarray(sample),
                patient_id=np.asarray(str(fold["test_patient_id"])),
                fold_id=np.asarray(fold_id, dtype=np.int32),
                target_data_accessed=np.asarray(False),
            )
            prediction_rows.append(
                {
                    "sample_id": sample,
                    "patient_id": str(fold["test_patient_id"]),
                    "path": str(path.relative_to(output)),
                    "sha256": sha256_file(path),
                    "n_spots": len(prediction),
                    "n_genes": prediction.shape[1],
                }
            )
        metadata = {
            "schema_version": 1,
            "source_revision": "benchmark-common-contract-v2",
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "experiment": "Original_encoder_direct_head_attribution_v1/benchmark_common_contract_v2",
            "method": {
                "method_id": method_id,
                "display_name": f"CTransPath-{scale} + affine Ridge",
                "encoder": f"frozen CTransPath pooled descriptor, {scale} px",
                "downstream": "single affine Ridge map",
                "target_transform": "official log1p(raw count)",
            },
            "fold": {
                "fold_id": fold_id,
                "train_ids": list(train_ids),
                "val_ids": list(val_ids),
                "test_ids": list(test_ids),
                "train_patients": list(map(str, fold["train_patient_ids"])),
                "val_patient": str(fold["val_patient_id"]),
                "test_patient": str(fold["test_patient_id"]),
            },
            "target_parameters": {"kind": "log1p_raw"},
            "feature_files": {
                sample: {
                    "path": identities[sample]["path"],
                    "sha256": identities[sample]["sha256"],
                }
                for sample in sorted(identities)
            },
            "selected_alpha": selected["alpha"],
            "validation_candidates": candidates,
            "best_epoch_selected_by_validation_only": 0,
            "best_validation_mean_gene_pcc": selected[
                "validation_mean_slide_gene_pcc"
            ],
            "checkpoint_selection_scope": "inner_validation_patient",
            "outer_test_labels_opened_during_fit": False,
            "predictions_contain_truth": False,
            "coverage_contract": "all_outer_test_slides",
            "predictions": prediction_rows,
            "wall_time_seconds": time.perf_counter() - started,
        }
        atomic_json(output / "run_metadata.json", metadata)
        atomic_json(
            output / "complete.json",
            {
                "run_metadata_sha256": sha256_file(output / "run_metadata.json"),
                "prediction_sha256": {
                    row["sample_id"]: row["sha256"] for row in prediction_rows
                },
            },
        )
        print(
            json.dumps(
                {
                    "method": method_id,
                    "fold": fold_id,
                    "alpha": selected["alpha"],
                    "validation_pcc": selected["validation_mean_slide_gene_pcc"],
                }
            ),
            flush=True,
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scale", type=int, choices=(96, 192, 224, 256, 384), required=True)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--descriptors", type=Path, default=DESCRIPTORS)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(args.scale, args.output, args.descriptors)
