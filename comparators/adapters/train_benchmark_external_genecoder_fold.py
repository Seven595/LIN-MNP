#!/usr/bin/env python3
"""Run a source-equation GeneCodeR adapter on one benchmark patient fold.

The released R package delegates fitting to ``gcode``.  Its Python port does
not parse (``!alpha_fix``), and no R runtime is available in the locked
benchmark environment.  This adapter therefore solves the released linear
transformation directly: two 500-dimensional IRLBA-style SVD bases followed
by an orthogonal Procrustes alignment of the paired spot scores.  Prediction
uses the exact ``genecoder.transform`` matrix equation.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Mapping, Sequence

import numpy as np
from sklearn.utils.extmath import randomized_svd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from lin_mnp.benchmark.benchmark_external import (  # noqa: E402
    PATCH_ROOT,
    apply_target_transform,
    fit_target_transform,
    load_folds,
    load_slides,
)
from lin_mnp.paths import RESULTS_ROOT


DEFAULT_OUTPUT = RESULTS_ROOT / "comparator_reruns"
GENECODER_SOURCE = PROJECT_ROOT / "comparators/methods/GeneCodeR/R/genecoder.R"
GCODE_SOURCE = PROJECT_ROOT / "comparators/methods/gcode_R/R/gcode.R"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def source_revision(path: Path) -> str:
    if (path.joinpath(".git").exists()):
        return subprocess.check_output(
            ["git", "-C", str(path), "rev-parse", "HEAD"], text=True
        ).strip()
    return "official-main-raw-files; identified by official_source_sha256"


def image_matrix(
    slides: Mapping[str, object], sample_ids: Sequence[str], window: int = 30
) -> np.ndarray:
    patches = np.load(PATCH_ROOT / "patches_224_uint8.npy", mmap_mode="r")
    offset = (patches.shape[1] - window) // 2
    rows = []
    for sample_id in sample_ids:
        slide = slides[sample_id]
        crop = np.asarray(
            patches[slide.patch_start : slide.patch_stop,
                    offset : offset + window, offset : offset + window],
            dtype=np.float32,
        )
        rows.append(crop.reshape(len(crop), -1) / 255.0)
    return np.concatenate(rows).astype(np.float32, copy=False)


def target_matrix(
    slides: Mapping[str, object],
    sample_ids: Sequence[str],
    parameters: Mapping[str, object],
) -> np.ndarray:
    return np.concatenate(
        [apply_target_transform(slides[sample_id], parameters) for sample_id in sample_ids]
    ).astype(np.float32, copy=False)


def fit_source_equation(
    image: np.ndarray, expression: np.ndarray, dimension: int, seed: int
) -> dict[str, np.ndarray]:
    image_mean = image.mean(axis=0, dtype=np.float64).astype(np.float32)
    expression_mean = expression.mean(axis=0, dtype=np.float64).astype(np.float32)
    image_centered = image - image_mean
    expression_centered = expression - expression_mean
    rank = min(dimension, image.shape[0] - 1, image.shape[1], expression.shape[1])
    # gcode uses one-iteration IRLBA for beta initialisation.  Randomized SVD
    # with one power iteration is the numerically corresponding Python solver.
    _, _, image_beta = randomized_svd(
        image_centered, n_components=rank, n_iter=1, random_state=seed
    )
    _, _, expression_beta = randomized_svd(
        expression_centered, n_components=rank, n_iter=1, random_state=seed + 1
    )
    image_beta = image_beta.T.astype(np.float32)
    expression_beta = expression_beta.T.astype(np.float32)
    image_scores = image_centered @ image_beta
    expression_scores = expression_centered @ expression_beta
    left, _, right = np.linalg.svd(
        image_scores.T.astype(np.float64) @ expression_scores.astype(np.float64),
        full_matrices=False,
    )
    rotation = (left @ right).astype(np.float32)
    # beta_image is orthonormal, so the released
    # beta_H %*% ginv(t(beta_H)%*%beta_H) %*% t(beta_E) equation reduces to:
    mapping = (image_beta @ rotation @ expression_beta.T).astype(np.float32)
    return {
        "image_mean": image_mean,
        "expression_mean": expression_mean,
        "image_beta": image_beta,
        "expression_beta": expression_beta,
        "rotation": rotation,
        "mapping": mapping,
    }


def predict(image: np.ndarray, model: Mapping[str, np.ndarray]) -> np.ndarray:
    return (
        (image - model["image_mean"]) @ model["mapping"]
        + model["expression_mean"]
    ).astype(np.float32)


def mean_slide_gene_pcc(
    predictions: Mapping[str, np.ndarray], targets: Mapping[str, np.ndarray]
) -> float:
    rows = []
    for sample_id in sorted(predictions):
        pred = predictions[sample_id].astype(np.float64)
        true = targets[sample_id].astype(np.float64)
        pred -= pred.mean(0, keepdims=True)
        true -= true.mean(0, keepdims=True)
        denominator = np.sqrt(np.sum(pred * pred, 0) * np.sum(true * true, 0))
        rows.append(np.divide(
            np.sum(pred * true, 0), denominator,
            out=np.full_like(denominator, np.nan), where=denominator > 0,
        ))
    return float(np.nanmean(np.nanmean(np.stack(rows), axis=0)))


def split_predictions(
    matrix: np.ndarray, slides: Mapping[str, object], sample_ids: Sequence[str]
) -> dict[str, np.ndarray]:
    result = {}
    start = 0
    for sample_id in sample_ids:
        stop = start + len(slides[sample_id].barcodes)
        result[sample_id] = matrix[start:stop]
        start = stop
    if start != len(matrix):
        raise ValueError("prediction split does not cover the matrix")
    return result


def train_fold(args: argparse.Namespace) -> Path:
    if not GENECODER_SOURCE.is_file() or not GCODE_SOURCE.is_file():
        raise FileNotFoundError("GeneCodeR/gcode official R sources are missing")
    folds = load_folds()
    fold = folds[args.fold]
    slides, genes = load_slides()
    parameters = fit_target_transform(slides, fold.train_ids, "raw_counts")
    output = (args.output / "genecoder" / f"fold{fold.fold_id}").resolve()
    if (output / "complete.json").is_file():
        return output
    if output.exists():
        raise FileExistsError(f"incomplete fold output exists: {output}")
    staging = output.with_name(f"{output.name}.tmp-{os.getpid()}")
    staging.mkdir(parents=True, exist_ok=False)
    started = time.time()
    seed = args.seed + fold.fold_id

    train_image = image_matrix(slides, fold.train_ids, args.window)
    train_expression = target_matrix(slides, fold.train_ids, parameters)
    model = fit_source_equation(train_image, train_expression, args.dimension, seed)
    del train_image, train_expression

    val_image = image_matrix(slides, fold.val_ids, args.window)
    val_prediction = split_predictions(predict(val_image, model), slides, fold.val_ids)
    val_targets = {
        sample_id: apply_target_transform(slides[sample_id], parameters)
        for sample_id in fold.val_ids
    }
    validation_pcc = mean_slide_gene_pcc(val_prediction, val_targets)
    test_image = image_matrix(slides, fold.test_ids, args.window)
    test_prediction = split_predictions(predict(test_image, model), slides, fold.test_ids)

    checkpoint = staging / "best_checkpoint.npz"
    np.savez_compressed(checkpoint, **model)
    history = staging / "training_history.csv"
    with history.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=[
            "iteration", "validation_mean_gene_pcc", "elapsed_seconds"
        ])
        writer.writeheader()
        writer.writerow({
            "iteration": 1,
            "validation_mean_gene_pcc": validation_pcc,
            "elapsed_seconds": time.time() - started,
        })

    prediction_dir = staging / "predictions"
    prediction_dir.mkdir()
    prediction_rows = []
    for sample_id in fold.test_ids:
        path = prediction_dir / f"{sample_id}.npz"
        np.savez_compressed(
            path,
            prediction=test_prediction[sample_id],
            genes=np.asarray(genes),
            barcodes=np.asarray(slides[sample_id].barcodes),
            sample_id=np.asarray(sample_id),
            patient_id=np.asarray(slides[sample_id].patient_id),
            fold_id=np.asarray(fold.fold_id, dtype=np.int32),
        )
        prediction_rows.append({
            "sample_id": sample_id,
            "patient_id": slides[sample_id].patient_id,
            "path": str(path.relative_to(staging)),
            "sha256": sha256_file(path),
            "n_spots": len(slides[sample_id].barcodes),
            "n_genes": len(genes),
        })

    method = {
        "method_id": "genecoder",
        "display_name": "GeneCodeR",
        "source_dir": "comparators/methods/GeneCodeR + comparators/methods/gcode_R",
        "source_kind": "official released transformation equation with Python numerical adapter",
        "target_transform": "raw_counts",
        "i_dim": args.dimension,
        "j_dim": args.dimension,
        "max_iterations_reported_by_benchmark": 20,
        "window_size": args.window,
        "runtime_compatibility_patches": [
            "official Python port is syntactically invalid at !alpha_fix",
            "R is unavailable in the locked environment",
            "released linear transform solved by one-step IRLBA-style SVD and orthogonal Procrustes",
        ],
    }
    metadata = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "method": method,
        "source_revision": {
            "GeneCodeR": source_revision(PROJECT_ROOT / "comparators/methods/GeneCodeR"),
            "gcode_R": source_revision(PROJECT_ROOT / "comparators/methods/gcode_R"),
        },
        "official_source_sha256": {
            "genecoder.R": sha256_file(GENECODER_SOURCE),
            "gcode.R": sha256_file(GCODE_SOURCE),
        },
        "fold": {
            "fold_id": fold.fold_id,
            "train_ids": list(fold.train_ids),
            "val_ids": list(fold.val_ids),
            "test_ids": list(fold.test_ids),
        },
        "seed": seed,
        "best_epoch_selected_by_validation_only": 1,
        "best_validation_mean_gene_pcc": validation_pcc,
        "target_parameters": parameters,
        "patch_manifest_sha256": sha256_file(PATCH_ROOT / "manifest.json"),
        "predictions_contain_truth": False,
        "predictions": prediction_rows,
        "wall_time_seconds": time.time() - started,
    }
    (staging / "run_metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    complete = {
        "run_metadata_sha256": sha256_file(staging / "run_metadata.json"),
        "training_history_sha256": sha256_file(history),
        "checkpoint_sha256": sha256_file(checkpoint),
        "prediction_sha256": {row["sample_id"]: row["sha256"] for row in prediction_rows},
    }
    (staging / "complete.json").write_text(
        json.dumps(complete, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    staging.rename(output)
    return output


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fold", type=int, required=True)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--dimension", type=int, default=500)
    parser.add_argument("--window", type=int, default=30)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    print(json.dumps({"output": str(train_fold(parse_args(argv)))}, sort_keys=True))


if __name__ == "__main__":
    main()
