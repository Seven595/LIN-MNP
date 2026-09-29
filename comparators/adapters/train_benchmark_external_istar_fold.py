#!/usr/bin/env python3
"""Train the released iStar ForwardSum head on one benchmark subset fold."""

from __future__ import annotations

import argparse
from contextlib import nullcontext
import csv
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import time
from typing import Mapping, Sequence

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset


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
CACHE_ROOT = Path(
    os.environ.get(
        "LIN_MNP_ISTAR_FEATURE_ROOT",
        str(CACHE_ROOT / "cscc/istar_hipt"),
    )
)
ISTAR_ROOT = PROJECT_ROOT / "comparators/methods/iStar"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class FeedForward(nn.Module):
    """Numerically identical to the released iStar ``FeedForward`` class."""

    def __init__(self, n_inp: int, n_out: int, activation: nn.Module | None = None):
        super().__init__()
        self.linear = nn.Linear(n_inp, n_out)
        self.activation = activation or nn.LeakyReLU(0.1, inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.activation(self.linear(x))


class ELU(nn.Module):
    def __init__(self, alpha: float = 0.01, beta: float = 0.01):
        super().__init__()
        self.activation = nn.ELU(alpha=alpha, inplace=True)
        self.beta = beta

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.activation(x) + self.beta


class ForwardSumModel(nn.Module):
    """Released iStar 579-256x4-997 nonlinear cell-to-spot network."""

    def __init__(self, n_inp: int, n_out: int):
        super().__init__()
        self.net_lat = nn.Sequential(
            FeedForward(n_inp, 256), FeedForward(256, 256),
            FeedForward(256, 256), FeedForward(256, 256),
        )
        self.net_out = FeedForward(256, n_out, activation=ELU())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net_out(self.net_lat(x)).mean(dim=-2)


class IStarDataset(Dataset):
    def __init__(
        self,
        feature_path: Path,
        cache_rows: Mapping[str, Mapping[str, object]],
        slides: Mapping[str, object],
        sample_ids: Sequence[str],
        target_parameters: Mapping[str, object],
        feature_mean: np.ndarray,
        feature_std: np.ndarray,
    ) -> None:
        self.feature_path = feature_path
        self._features: np.ndarray | None = None
        features = []
        targets = []
        slide_indices = []
        local_indices = []
        for slide_index, sample_id in enumerate(sample_ids):
            row = cache_rows[sample_id]
            features.extend(range(int(row["start"]), int(row["stop"])))
            targets.append(apply_target_transform(slides[sample_id], target_parameters))
            slide_indices.extend([slide_index] * len(slides[sample_id].barcodes))
            local_indices.extend(range(len(slides[sample_id].barcodes)))
        self.indices = np.asarray(features, dtype=np.int64)
        self.targets = np.concatenate(targets).astype(np.float32, copy=False)
        self.slide_indices = np.asarray(slide_indices, dtype=np.int16)
        self.local_indices = np.asarray(local_indices, dtype=np.int32)
        self.feature_mean = feature_mean.astype(np.float32)
        self.feature_std = feature_std.astype(np.float32)

    @property
    def features(self) -> np.ndarray:
        if self._features is None:
            self._features = np.load(self.feature_path, mmap_mode="r")
        return self._features

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int):
        feature = np.asarray(self.features[self.indices[index]], dtype=np.float32)
        feature = (feature - self.feature_mean) / self.feature_std
        return (
            torch.from_numpy(feature), torch.from_numpy(self.targets[index]),
            int(self.slide_indices[index]), int(self.local_indices[index]),
        )


def first_slide(slides: Mapping[str, object], patient_id: str) -> str:
    return min(sample_id for sample_id, slide in slides.items() if slide.patient_id == patient_id)


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


@torch.no_grad()
def predict(
    model: nn.Module,
    loader: DataLoader,
    sample_ids: Sequence[str],
    slides: Mapping[str, object],
    n_genes: int,
    device: torch.device,
    amp: bool,
) -> dict[str, np.ndarray]:
    model.eval()
    output = {
        sample_id: np.empty((len(slides[sample_id].barcodes), n_genes), dtype=np.float32)
        for sample_id in sample_ids
    }
    for feature, _, slide_index, local_index in loader:
        feature = feature.to(device, non_blocking=True)
        context = torch.autocast("cuda", dtype=torch.float16) if amp else nullcontext()
        with context:
            value = model(feature).float().cpu().numpy()
        for row, (slide, local) in enumerate(zip(slide_index.numpy(), local_index.numpy())):
            output[sample_ids[int(slide)]][int(local)] = value[row]
    return output


def train_fold(args: argparse.Namespace) -> Path:
    feature_path = CACHE_ROOT / "features_float16.npy"
    cache_metadata_path = CACHE_ROOT / "metadata.json"
    if not feature_path.is_file() or not cache_metadata_path.is_file():
        raise FileNotFoundError("run cache_benchmark_istar_features.py first")
    cache_metadata = json.loads(cache_metadata_path.read_text(encoding="utf-8"))
    if sha256_file(feature_path) != cache_metadata["feature_sha256"]:
        raise ValueError("iStar feature cache hash mismatch")
    cache_rows = {row["sample_id"]: row for row in cache_metadata["slides"]}
    slides, genes = load_slides()
    fold = load_folds()[args.fold]
    external_contract = bool(os.environ.get("LIN_MNP_EXTERNAL_CONTRACT"))
    fixed_final_epoch = (
        os.environ.get("LIN_MNP_CHECKPOINT_MODE") == "fixed_final_epoch"
    )
    if external_contract:
        train_ids, val_ids, test_ids = fold.train_ids, fold.val_ids, fold.test_ids
    else:
        train_ids = tuple(first_slide(slides, patient) for patient in fold.train_patients)
        val_ids = (first_slide(slides, fold.val_patient),)
        test_ids = (first_slide(slides, fold.test_patient),)
    parameters = fit_target_transform(slides, train_ids, "istar_train_minmax_raw_counts")
    output = (args.output / "istar" / f"fold{fold.fold_id}").resolve()
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
    raw_features = np.load(feature_path, mmap_mode="r")
    train_indices = np.concatenate([
        np.arange(int(cache_rows[sample]["start"]), int(cache_rows[sample]["stop"]))
        for sample in train_ids
    ])
    # Fit feature normalization using training patients only.
    train_feature = np.asarray(raw_features[train_indices], dtype=np.float32)
    feature_mean = train_feature.mean(axis=(0, 1), dtype=np.float64).astype(np.float32)
    feature_std = train_feature.std(axis=(0, 1), dtype=np.float64).astype(np.float32)
    feature_std[~np.isfinite(feature_std) | (feature_std < 1e-6)] = 1.0
    del train_feature, raw_features

    train_data = IStarDataset(
        feature_path, cache_rows, slides, train_ids, parameters, feature_mean, feature_std
    )
    val_data = IStarDataset(
        feature_path, cache_rows, slides, val_ids, parameters, feature_mean, feature_std
    )
    test_data = IStarDataset(
        feature_path, cache_rows, slides, test_ids, parameters, feature_mean, feature_std
    )
    batch_size = min(128, max(1, len(train_data) // 16))
    options = dict(batch_size=batch_size, num_workers=args.workers, pin_memory=True)
    train_loader = DataLoader(train_data, shuffle=True, **options)
    val_loader = DataLoader(val_data, shuffle=False, **options)
    test_loader = DataLoader(test_data, shuffle=False, **options)
    model = ForwardSumModel(579, len(genes)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp)
    val_targets = {
        sample_id: apply_target_transform(slides[sample_id], parameters)
        for sample_id in val_ids
    }
    best_score = -np.inf
    best_epoch = -1
    checkpoint = staging / "best_checkpoint.pt"
    history_rows = []
    started = time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        loss_sum = 0.0
        count = 0
        for feature, target, _, _ in train_loader:
            feature = feature.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            context = torch.autocast("cuda", dtype=torch.float16) if args.amp else nullcontext()
            with context:
                prediction = model(feature)
                loss = (prediction - target).square().mean()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            loss_sum += float(loss.detach()) * len(target)
            count += len(target)
        val_prediction = predict(
            model, val_loader, val_ids, slides, len(genes), device, args.amp
        )
        score = mean_slide_gene_pcc(val_prediction, val_targets)
        if fixed_final_epoch or (np.isfinite(score) and score > best_score):
            best_score = score
            best_epoch = epoch
            torch.save({"model": model.state_dict(), "epoch": epoch, "score": score}, checkpoint)
        row = {
            "epoch": epoch,
            "train_mse": loss_sum / max(count, 1),
            "validation_mean_gene_pcc": score,
            "best_validation_mean_gene_pcc": best_score,
            "elapsed_seconds": time.time() - started,
        }
        history_rows.append(row)
        if epoch == 1 or epoch % 20 == 0 or epoch == args.epochs:
            print(json.dumps(row, sort_keys=True), flush=True)
    state = torch.load(checkpoint, map_location=device)
    model.load_state_dict(state["model"])
    test_prediction = predict(
        model, test_loader, test_ids, slides, len(genes), device, args.amp
    )

    history = staging / "training_history.csv"
    with history.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(history_rows[0]))
        writer.writeheader()
        writer.writerows(history_rows)
    prediction_dir = staging / "predictions"
    prediction_dir.mkdir()
    prediction_rows = []
    for sample_id in test_ids:
        path = prediction_dir / f"{sample_id}.npz"
        np.savez_compressed(
            path,
            prediction=test_prediction[sample_id], genes=np.asarray(genes),
            barcodes=np.asarray(slides[sample_id].barcodes), sample_id=np.asarray(sample_id),
            patient_id=np.asarray(slides[sample_id].patient_id),
            fold_id=np.asarray(fold.fold_id, dtype=np.int32),
        )
        prediction_rows.append({
            "sample_id": sample_id, "patient_id": slides[sample_id].patient_id,
            "path": str(path.relative_to(staging)), "sha256": sha256_file(path),
            "n_spots": len(slides[sample_id].barcodes), "n_genes": len(genes),
        })
    method = {
        "method_id": "istar", "display_name": "iStar",
        "source_dir": "comparators/methods/iStar", "source_kind": "official HIPT encoders and ForwardSum architecture with benchmark subset adapter",
        "target_transform": "train-gene-minmax raw counts",
        "epochs": args.epochs, "learning_rate": 1e-4, "input_features": 579,
        "coverage_contract": (
            "HESpotEx_all_sections" if external_contract
            else "benchmark_first_slide_per_patient"
        ),
        "runtime_compatibility_patches": cache_metadata["runtime_compatibility_patches"] + [
            "PyTorch loop replaces unavailable pytorch-lightning without changing network/loss/optimizer",
            "validation-only checkpoint selection added for the common four-fold contract",
        ],
    }
    metadata = {
        "schema_version": 1, "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "method": method,
        "source_revision": subprocess.check_output(
            ["git", "-C", str(ISTAR_ROOT), "rev-parse", "HEAD"], text=True
        ).strip(),
        "official_source_sha256": {
            "impute.py": sha256_file(ISTAR_ROOT / "impute.py"),
            "hipt_4k.py": sha256_file(ISTAR_ROOT / "hipt_4k.py"),
        },
        "coverage_contract": (
            "HESpotEx_all_sections" if external_contract
            else "benchmark_first_slide_per_patient"
        ),
        "fold": {
            "fold_id": fold.fold_id, "train_ids": list(train_ids),
            "val_ids": list(val_ids), "test_ids": list(test_ids),
            "original_full_test_ids": list(fold.test_ids),
        },
        "seed": seed, "device": str(device),
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
        "feature_mean": feature_mean.tolist(), "feature_std": feature_std.tolist(),
        "feature_metadata_sha256": sha256_file(cache_metadata_path),
        "feature_sha256": sha256_file(feature_path),
        "patch_manifest_sha256": sha256_file(PATCH_ROOT / "manifest.json"),
        "predictions_contain_truth": False, "predictions": prediction_rows,
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
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=400)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    print(json.dumps({"output": str(train_fold(parse_args(argv)))}, sort_keys=True))


if __name__ == "__main__":
    main()
