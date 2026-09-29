#!/usr/bin/env python3
"""Evaluate Lin-MNP across ImageNet and pathology encoders.

Each encoder is evaluated as patient-equal ridge (α=1e6) and as Lin-MNP
(same ridge plus one Borda cross-section step).  Same cSCC RGB224 field.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from lin_mnp import model
from lin_mnp.paths import CACHE_ROOT, RESULTS_ROOT


RESULT = RESULTS_ROOT / "encoder_transfer"
EXTRA = CACHE_ROOT / "cscc/encoder_panel"

ENCODERS = {
    "ResNet18": {
        "family": "imagenet_cnn",
        "context": EXTRA / "ResNet18_context_float32.npy",
        "local": EXTRA / "ResNet18_local_float32.npy",
        "dim": 512,
    },
    "ResNet50": {
        "family": "imagenet_cnn",
        "context": EXTRA / "ResNet50_context_float32.npy",
        "local": EXTRA / "ResNet50_local_float32.npy",
        "dim": 2048,
    },
    "VGG16": {
        "family": "imagenet_cnn",
        "context": EXTRA / "VGG16_context_float32.npy",
        "local": EXTRA / "VGG16_local_float32.npy",
        "dim": 512,
    },
    "DenseNet121": {
        "family": "imagenet_cnn",
        "context": EXTRA / "DenseNet121_context_float32.npy",
        "local": EXTRA / "DenseNet121_local_float32.npy",
        "dim": 1024,
    },
    "QuiltNet-B-32": {
        "family": "pathology_clip",
        "context": EXTRA / "QuiltNet-B-32_context_float32.npy",
        "local": EXTRA / "QuiltNet-B-32_local_float32.npy",
        "dim": 512,
    },
    "CTransPath": {
        "family": "pathology_pfm",
        "context": CACHE_ROOT / "cscc/ctranspath_context/pooled_768_float16.npy",
        "local": None,
        "dim": 768,
    },
    "Phikon-v2": {
        "family": "pathology_pfm",
        "context": CACHE_ROOT / "cscc/phikon_v2/context.npy",
        "local": CACHE_ROOT / "cscc/phikon_v2/local.npy",
        "dim": 1024,
    },
    "UNI-v1": {
        "family": "pathology_pfm",
        "context": CACHE_ROOT / "cscc/uni_v1/context.npy",
        "local": CACHE_ROOT / "cscc/uni_v1/local.npy",
        "dim": 1024,
    },
}


def fit_ridge(feature, target, patients, patient_ids, device):
    feature = np.asarray(feature, dtype=np.float32)
    target = np.asarray(target, dtype=np.float32)
    weight = model.patient_equal_weights(patients, patient_ids)
    total = float(weight.sum())
    feature_mean = np.sum(weight[:, None] * feature, axis=0) / total
    feature_variance = np.sum(
        weight[:, None] * np.square(feature - feature_mean), axis=0
    ) / total
    feature_scale = np.maximum(np.sqrt(feature_variance), 1e-5)
    target_mean = np.sum(weight[:, None] * target, axis=0) / total
    root_weight = np.sqrt(weight)[:, None]
    x = torch.from_numpy(
        (root_weight * ((feature - feature_mean) / feature_scale)).astype(np.float64)
    ).to(device)
    y = torch.from_numpy(
        (root_weight * (target - target_mean)).astype(np.float64)
    ).to(device)
    gram = x.T @ x
    gram.diagonal().add_(model.RIDGE_ALPHA)
    coefficient = torch.linalg.solve(gram, x.T @ y).float().cpu().numpy()
    return {
        "feature_mean": feature_mean.astype(np.float32),
        "feature_scale": feature_scale.astype(np.float32),
        "target_mean": target_mean.astype(np.float32),
        "weight": coefficient.astype(np.float32),
    }


def aggregate(records, key):
    by_patient = {}
    for row in records:
        by_patient.setdefault(row["patient"], {"pcc": [], "mse": []})
        by_patient[row["patient"]]["pcc"].append(row[key]["pcc"])
        by_patient[row["patient"]]["mse"].append(row[key]["mse"])
    patients = {
        patient: {
            "pcc": float(np.mean(by_patient[patient]["pcc"])),
            "mse": float(np.mean(by_patient[patient]["mse"])),
        }
        for patient in ("P2", "P5", "P10")
    }
    return {
        "patient_equal_pcc": float(np.mean([patients[p]["pcc"] for p in ("P2", "P5", "P10")])),
        "patient_equal_mse": float(np.mean([patients[p]["mse"] for p in ("P2", "P5", "P10")])),
        "patients": patients,
    }


def evaluate_encoder(spec, device):
    context_cache = np.load(Path(spec["context"]), mmap_mode="r")
    local_cache = (
        model.load_local_cache(model.CONFIG)
        if spec["local"] is None
        else np.load(Path(spec["local"]), mmap_mode="r")
    )
    if context_cache.shape[1] != spec["dim"] or local_cache.shape[1] != spec["dim"]:
        raise ValueError(f"dim failure {spec}")
    config = dict(model.CONFIG)
    folds = json.loads(Path(config["folds"]).read_text())["folds"]
    records = []
    for fold in folds:
        train = model.LabeledSpotDataset(config, fold["train_sample_ids"])
        index = np.asarray(train.patch_indices, dtype=np.int64)
        artifact = fit_ridge(
            np.asarray(context_cache[index], dtype=np.float32),
            train.target_matrix, train.patients, fold["train_patient_ids"], device,
        )
        val = model.LabeledSpotDataset(config, fold["val_sample_ids"])
        index = np.asarray(val.patch_indices, dtype=np.int64)
        context = np.asarray(context_cache[index], dtype=np.float32)
        local = np.asarray(local_cache[index], dtype=np.float32)
        ridge = model.predict_ridge(context, artifact)
        graph = model.build_section_balanced_graph(val.sections, context, local)
        full = model.diffuse(ridge, graph, model.DIFFUSION_DEPTH)
        records.append({
            "fold": int(fold["fold_id"]),
            "patient": fold["val_patient_id"],
            "ridge": {
                "pcc": model.section_pcc(val.target_matrix, ridge, val.sections),
                "mse": model.section_mse(val.target_matrix, ridge, val.sections),
            },
            "lin_mnp": {
                "pcc": model.section_pcc(val.target_matrix, full, val.sections),
                "mse": model.section_mse(val.target_matrix, full, val.sections),
            },
        })
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit(f"requested CUDA device is unavailable: {args.device}")
    RESULT.mkdir(parents=True, exist_ok=True)
    encoders = {}
    for name, spec in ENCODERS.items():
        records = evaluate_encoder(spec, device)
        ridge = aggregate(records, "ridge")
        full = aggregate(records, "lin_mnp")
        encoders[name] = {
            "family": spec["family"],
            "dim": spec["dim"],
            "folds": records,
            "ridge": ridge,
            "lin_mnp": full,
            "delta_pcc": full["patient_equal_pcc"] - ridge["patient_equal_pcc"],
            "delta_mse": full["patient_equal_mse"] - ridge["patient_equal_mse"],
        }
        print(json.dumps({
            "encoder": name,
            "ridge": ridge["patient_equal_pcc"],
            "lin_mnp": full["patient_equal_pcc"],
            "delta": encoders[name]["delta_pcc"],
        }), flush=True)
    summary = {
        "schema_version": 1,
        "status": "COMPLETE",
        "experiment_id": "encoder_transfer_cscc",
        "protocol": (
            "cSCC development validation; P2 folds averaged first, then equal P2/P5/P10; "
            "frozen α=1e6 raw ridge; Lin-MNP adds one Borda cross-section step"
        ),
        "outer_test_labels_accessed": False,
        "encoders": encoders,
    }
    (RESULT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({
        "status": "COMPLETE",
        "rows": [
            {
                "encoder": name,
                "family": row["family"],
                "ridge_pcc": row["ridge"]["patient_equal_pcc"],
                "lin_mnp_pcc": row["lin_mnp"]["patient_equal_pcc"],
                "delta_pcc": row["delta_pcc"],
                "ridge_mse": row["ridge"]["patient_equal_mse"],
                "lin_mnp_mse": row["lin_mnp"]["patient_equal_mse"],
                "patients": {
                    patient: {
                        "ridge": row["ridge"]["patients"][patient]["pcc"],
                        "lin_mnp": row["lin_mnp"]["patients"][patient]["pcc"],
                    }
                    for patient in ("P2", "P5", "P10")
                },
            }
            for name, row in encoders.items()
        ],
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
