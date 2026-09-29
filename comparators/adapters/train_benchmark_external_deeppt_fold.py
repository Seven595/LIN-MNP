#!/usr/bin/env python3
"""Run the official two-stage DeepPT architecture on one benchmark patient fold."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import random
import sys
import time
from typing import Mapping, Sequence

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


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
from lin_mnp.paths import CACHE_ROOT, RESULTS_ROOT


DEFAULT_OUTPUT = RESULTS_ROOT / "comparator_reruns"
FEATURE_ROOT = Path(
    os.environ.get(
        "LIN_MNP_DEEPPT_FEATURE_ROOT",
        str(CACHE_ROOT / "cscc/deeppt_resnet50"),
    )
)
AUTHOR_ROOT = PROJECT_ROOT / "comparators/methods/DeepPT/zenodo_11125591"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_class(path: Path, module_name: str, class_name: str) -> type[nn.Module]:
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return getattr(module, class_name)


def _indices(slides: Mapping[str, object], sample_ids: Sequence[str]) -> np.ndarray:
    return np.concatenate(
        [
            np.arange(slides[name].patch_start, slides[name].patch_stop, dtype=np.int64)
            for name in sample_ids
        ]
    )


def _pcc(true: np.ndarray, prediction: np.ndarray) -> np.ndarray:
    true = true.astype(np.float64) - true.mean(0, keepdims=True)
    prediction = prediction.astype(np.float64) - prediction.mean(0, keepdims=True)
    denominator = np.sqrt(np.sum(true * true, 0) * np.sum(prediction * prediction, 0))
    return np.divide(
        np.sum(true * prediction, 0),
        denominator,
        out=np.full_like(denominator, np.nan),
        where=denominator > 0,
    )


def mean_slide_gene_pcc(
    predictions: Mapping[str, np.ndarray], targets: Mapping[str, np.ndarray]
) -> float:
    values = [_pcc(targets[name], predictions[name]) for name in sorted(predictions)]
    return float(np.nanmean(np.nanmean(np.stack(values), axis=0)))


def mlp_batch(model: nn.Module, feature: torch.Tensor) -> torch.Tensor:
    """Vectorized form of the released model's one-slide/one-tile forward pass."""
    return model.layer1(model.layer0(feature))


@torch.no_grad()
def encode(
    autoencoder: nn.Module,
    features: np.ndarray,
    indices: np.ndarray,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    autoencoder.eval()
    result = np.empty((len(indices), 512), dtype=np.float32)
    for start in range(0, len(indices), batch_size):
        source = torch.from_numpy(
            np.asarray(features[indices[start : start + batch_size]], dtype=np.float32).copy()
        ).to(device)
        result[start : start + len(source)] = (
            autoencoder.encoder(source).float().cpu().numpy()
        )
    return result


@torch.no_grad()
def predict(
    model: nn.Module,
    feature: np.ndarray,
    sample_ids: Sequence[str],
    slide_sizes: Mapping[str, int],
    n_genes: int,
    device: torch.device,
    batch_size: int,
) -> dict[str, np.ndarray]:
    model.eval()
    combined = np.empty((len(feature), n_genes), dtype=np.float32)
    for start in range(0, len(feature), batch_size):
        value = torch.from_numpy(feature[start : start + batch_size]).to(device)
        combined[start : start + len(value)] = mlp_batch(model, value).float().cpu().numpy()
    result = {}
    offset = 0
    for sample_id in sample_ids:
        stop = offset + slide_sizes[sample_id]
        result[sample_id] = combined[offset:stop]
        offset = stop
    return result


def train_fold(args: argparse.Namespace) -> Path:
    feature_path = FEATURE_ROOT / "resnet50_imagenet_v2_float32.npy"
    feature_metadata = FEATURE_ROOT / "metadata.json"
    if not feature_path.is_file() or not feature_metadata.is_file():
        raise FileNotFoundError("run cache_benchmark_deeppt_embeddings.py first")

    folds = load_folds()
    fold = folds[args.fold]
    fixed_final_epoch = (
        os.environ.get("LIN_MNP_CHECKPOINT_MODE") == "fixed_final_epoch"
    )
    slides, genes = load_slides()
    parameters = fit_target_transform(slides, fold.train_ids, "log1p_cp10k")
    total_spots = max(slide.patch_stop for slide in slides.values())
    all_targets = np.empty((total_spots, len(genes)), dtype=np.float32)
    for slide in slides.values():
        all_targets[slide.patch_start : slide.patch_stop] = apply_target_transform(
            slide, parameters
        )
    train_index = _indices(slides, fold.train_ids)
    val_index = _indices(slides, fold.val_ids)
    test_index = _indices(slides, fold.test_ids)

    output = (args.output / "deeppt" / f"fold{fold.fold_id}").resolve()
    if (output / "complete.json").is_file():
        return output
    if output.exists():
        raise FileExistsError(f"incomplete fold output exists: {output}")
    staging = output.with_name(f"{output.name}.tmp-{os.getpid()}")
    staging.mkdir(parents=True, exist_ok=False)

    seed = args.seed + fold.fold_id
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    device = torch.device(args.device)
    features = np.load(feature_path, mmap_mode="r")

    autoencoder_source = AUTHOR_ROOT / "12AE/model_AE.py"
    mlp_source = AUTHOR_ROOT / "13DeepPT_train/model_MLP.py"
    AutoEncoder = _load_class(autoencoder_source, "deeppt_author_autoencoder", "AutoEncoder")
    # model_MLP imports its sibling utils module.
    sys.path.insert(0, str(mlp_source.parent))
    MLPRegression = _load_class(mlp_source, "deeppt_author_mlp", "MLP_regression")

    autoencoder = AutoEncoder(2048, 512, 2048).to(device)
    ae_optimizer = torch.optim.Adam(autoencoder.parameters(), lr=1e-4)
    ae_train = TensorDataset(
        torch.from_numpy(np.asarray(features[train_index], dtype=np.float32).copy())
    )
    ae_loader = DataLoader(ae_train, batch_size=32, shuffle=True, drop_last=False)
    history = []
    started = time.time()
    for epoch in range(1, args.ae_epochs + 1):
        autoencoder.train()
        loss_sum = 0.0
        count = 0
        for (feature,) in ae_loader:
            feature = feature.to(device)
            ae_optimizer.zero_grad(set_to_none=True)
            reconstruction = autoencoder(feature)
            loss = nn.functional.mse_loss(reconstruction, feature)
            loss.backward()
            ae_optimizer.step()
            loss_sum += float(loss.detach()) * len(feature)
            count += len(feature)
        if epoch == 1 or epoch % 25 == 0 or epoch == args.ae_epochs:
            print(
                json.dumps(
                    {
                        "stage": "autoencoder",
                        "epoch": epoch,
                        "train_mse": loss_sum / count,
                        "elapsed_seconds": time.time() - started,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )

    train_feature = encode(autoencoder, features, train_index, device, 256)
    val_feature = encode(autoencoder, features, val_index, device, 256)
    test_feature = encode(autoencoder, features, test_index, device, 256)
    del features

    train_target = all_targets[train_index]
    val_target = all_targets[val_index]
    bias = nn.Parameter(torch.from_numpy(train_target.mean(0)).to(device))
    model = MLPRegression(512, 512, len(genes), 0.2, bias).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    train_loader = DataLoader(
        TensorDataset(torch.from_numpy(train_feature), torch.from_numpy(train_target)),
        batch_size=32,
        shuffle=True,
        drop_last=False,
    )
    slide_sizes = {name: len(slides[name].barcodes) for name in slides}
    val_targets = {}
    offset = 0
    for name in fold.val_ids:
        stop = offset + slide_sizes[name]
        val_targets[name] = val_target[offset:stop]
        offset = stop

    best_score = -np.inf
    best_epoch = -1
    no_improvement = 0
    checkpoint = staging / "best_checkpoint.pt"
    for epoch in range(1, args.mlp_epochs + 1):
        model.train()
        loss_sum = 0.0
        count = 0
        for feature, target in train_loader:
            feature = feature.to(device)
            target = target.to(device)
            optimizer.zero_grad(set_to_none=True)
            prediction = mlp_batch(model, feature)
            loss = nn.functional.mse_loss(prediction, target)
            loss.backward()
            optimizer.step()
            loss_sum += float(loss.detach()) * len(target)
            count += len(target)
        val_prediction = predict(
            model, val_feature, fold.val_ids, slide_sizes, len(genes), device, 256
        )
        score = mean_slide_gene_pcc(val_prediction, val_targets)
        if fixed_final_epoch or (np.isfinite(score) and score > best_score):
            best_score = score
            best_epoch = epoch
            no_improvement = 0
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "autoencoder_state": autoencoder.state_dict(),
                    "epoch": epoch,
                    "validation_mean_gene_pcc": score,
                },
                checkpoint,
            )
        else:
            no_improvement += 1
        record = {
            "stage": "expression_mlp",
            "epoch": epoch,
            "train_mse": loss_sum / count,
            "validation_mean_gene_pcc": score,
            "best_validation_mean_gene_pcc": best_score,
            "elapsed_seconds": time.time() - started,
        }
        history.append(record)
        print(json.dumps(record, sort_keys=True), flush=True)
        if not fixed_final_epoch and no_improvement >= args.patience:
            break

    state = torch.load(checkpoint, map_location=device)
    model.load_state_dict(state["model_state"])
    test_prediction = predict(
        model, test_feature, fold.test_ids, slide_sizes, len(genes), device, 256
    )
    with (staging / "training_history.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)

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
        prediction_rows.append(
            {
                "sample_id": sample_id,
                "patient_id": slides[sample_id].patient_id,
                "path": str(path.relative_to(staging)),
                "sha256": sha256_file(path),
                "n_spots": len(slides[sample_id].barcodes),
                "n_genes": len(genes),
            }
        )

    method = {
        "method_id": "deeppt",
        "display_name": "DeepPT",
        "source_dir": "comparators/methods/DeepPT/zenodo_11125591",
        "source_kind": "official Zenodo DeepPT modules with benchmark spot-level data adapter",
        "target_transform": "log1p_cp10k",
        "ae_epochs": args.ae_epochs,
        "mlp_max_epochs": args.mlp_epochs,
        "early_stop_patience": args.patience,
        "batch_size": 32,
        "ae_learning_rate": 1e-4,
        "mlp_learning_rate": 1e-3,
        "dropout": 0.2,
        "input_size": 224,
        "optimizer": "Adam",
        "runtime_compatibility_patches": [
            "official slide-level MLP vectorized as one tile per benchmark spot",
            "fixed benchmark patches used; official whole-slide Macenko preprocessing not repeated",
        ],
    }
    metadata = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "method": method,
        "source_revision": "Zenodo 11125591 v1.1",
        "official_source_sha256": {
            "model_AE.py": sha256_file(autoencoder_source),
            "model_MLP.py": sha256_file(mlp_source),
        },
        "fold": {
            "fold_id": fold.fold_id,
            "train_ids": list(fold.train_ids),
            "val_ids": list(fold.val_ids),
            "test_ids": list(fold.test_ids),
        },
        "seed": seed,
        "device": str(device),
        "best_epoch_selected_by_validation_only": (
            best_epoch
            if os.environ.get("LIN_MNP_CHECKPOINT_SELECTION_SCOPE")
            == "inner_validation"
            else None
        ),
        "best_epoch_selected_by_outer_test": (
            best_epoch
            if os.environ.get("LIN_MNP_CHECKPOINT_SELECTION_SCOPE")
            == "outer_test"
            else None
        ),
        "fixed_refit_epoch": best_epoch if fixed_final_epoch else None,
        "checkpoint_selection_scope": (
            "fixed_epoch_selected_on_inner_validation_then_refit_on_all_outer_train"
            if fixed_final_epoch
            else os.environ.get(
                "LIN_MNP_CHECKPOINT_SELECTION_SCOPE", "inner_validation"
            )
        ),
        "best_validation_mean_gene_pcc": best_score,
        "target_parameters": {
            key: value.tolist() if isinstance(value, np.ndarray) else value
            for key, value in parameters.items()
        },
        "feature_metadata_sha256": sha256_file(feature_metadata),
        "feature_sha256": sha256_file(feature_path),
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
        "training_history_sha256": sha256_file(staging / "training_history.csv"),
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
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--ae-epochs", type=int, default=500)
    parser.add_argument("--mlp-epochs", type=int, default=500)
    parser.add_argument("--patience", type=int, default=50)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    print(json.dumps({"output": str(train_fold(parse_args(argv)))}, sort_keys=True))


if __name__ == "__main__":
    main()
