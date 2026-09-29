#!/usr/bin/env python3
"""Train benchmark-configured EGNv1 on one official cSCC patient fold."""

from __future__ import annotations

import argparse
from contextlib import nullcontext
import csv
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import random
import subprocess
import sys
import time
from typing import Mapping, Sequence

import numpy as np
from sklearn.neighbors import NearestNeighbors
import torch
import torch.nn as nn
import torch.nn.functional as F
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
EMBEDDING_ROOT = Path(
    os.environ.get(
        "LIN_MNP_EGN_EMBEDDING_ROOT",
        str(CACHE_ROOT / "cscc/egn_embeddings"),
    )
)
MEAN = torch.tensor((0.5476, 0.5218, 0.6881)).view(1, 3, 1, 1)
STD = torch.tensor((0.2461, 0.2101, 0.1649)).view(1, 3, 1, 1)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build_model(n_genes: int) -> nn.Module:
    path = PROJECT_ROOT / "comparators/methods/EGN/v1/egn.py"
    spec = importlib.util.spec_from_file_location("benchmark_author_egnv1", path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    model = module.EGN(
        image_size=224,
        patch_size=32,
        num_classes=n_genes,
        dim=1024,
        depth=8,
        heads=16,
        mlp_dim=4096,
        bhead=16,
        bdim=64,
        bfre=2,
        mdim=2048,
        player=2,
        linear_projection=True,
    )
    # The release hard-codes its demonstration panel at 250 genes.  These two
    # projections are the only gene-dimension-dependent layers.
    model.transformer.encoder.proe = nn.Linear(2048 + n_genes, 1024)
    model.transformer.encoder.to_v = nn.Linear(n_genes, 1024)
    # The released EB block assumes the original 256/32 = 8x8 token grid.
    # HEtoSGEBench explicitly changes the input to 224 px, hence a 7x7 grid.
    for _, eb, _ in model.transformer.layers:
        if eb is None:
            continue
        eb.to_x = nn.Sequential(
            nn.LayerNorm(1024),
            nn.Linear(1024, 16 * 64),
            module.Rearrange("b (h w) (n c) -> b n c h w", n=16, h=7, w=7),
        )
        eb.to_k = nn.Sequential(
            nn.LayerNorm(1024),
            nn.Linear(1024, 7 * 7 * 16),
            module.Rearrange("b l (n h w) -> b n l h w", n=16, h=7, w=7),
        )
        eb.to_v = nn.Sequential(
            module.Rearrange("b n c h w -> b (h w) (n c)"),
            nn.Linear(16 * 64, 1024, bias=False),
        )
    return model


def _slide_global_indices(slide: object) -> np.ndarray:
    return np.arange(slide.patch_start, slide.patch_stop, dtype=np.int64)


def build_neighbors(
    slides: Mapping[str, object],
    query_ids: Sequence[str],
    train_ids: Sequence[str],
    embeddings: np.ndarray,
    neighbors: int = 9,
) -> dict[str, np.ndarray]:
    result = {}
    for sample_id in query_ids:
        candidate_ids = [value for value in train_ids if value != sample_id]
        query_index = _slide_global_indices(slides[sample_id])
        # Some section-level external folds contain exactly one training slide.
        # For that training slide there is consequently no distinct training
        # section from which to draw exemplars.  Use same-slide neighbours in
        # this degenerate case, while explicitly excluding the query spot.  Test
        # and validation slides continue to draw exclusively from training
        # slides, so this fallback cannot expose held-out targets.
        single_train_fallback = not candidate_ids
        candidate_index = (
            query_index
            if single_train_fallback
            else np.concatenate(
                [_slide_global_indices(slides[value]) for value in candidate_ids]
            )
        )
        requested = neighbors + 1 if single_train_fallback else neighbors
        search = NearestNeighbors(
            n_neighbors=min(requested, len(candidate_index)),
            metric="euclidean",
            algorithm="brute",
            n_jobs=-1,
        ).fit(np.asarray(embeddings[candidate_index], dtype=np.float32))
        local = search.kneighbors(
            np.asarray(embeddings[query_index], dtype=np.float32),
            return_distance=False,
        )
        selected = candidate_index[local]
        if single_train_fallback:
            selected = np.stack(
                [row[row != query][:neighbors] for row, query in zip(selected, query_index)]
            )
        if selected.shape[1] != neighbors:
            raise ValueError("training fold does not contain nine exemplar candidates")
        result[sample_id] = selected.astype(np.int64)
    return result


class EGNDataset(Dataset):
    def __init__(
        self,
        slides: Mapping[str, object],
        sample_ids: Sequence[str],
        all_targets: np.ndarray,
        neighbors: Mapping[str, np.ndarray],
        embedding_path: Path,
    ) -> None:
        self.slides = slides
        self.sample_ids = tuple(sample_ids)
        self.all_targets = all_targets
        self.neighbors = neighbors
        self.embedding_path = embedding_path
        self._patches: np.ndarray | None = None
        self._embeddings: np.ndarray | None = None
        rows = []
        for slide_index, sample_id in enumerate(self.sample_ids):
            slide = slides[sample_id]
            for local_index, global_index in enumerate(_slide_global_indices(slide)):
                rows.append((slide_index, local_index, global_index))
        self.rows = np.asarray(rows, dtype=np.int64)

    @property
    def patches(self) -> np.ndarray:
        if self._patches is None:
            self._patches = np.load(PATCH_ROOT / "patches_224_uint8.npy", mmap_mode="r")
        return self._patches

    @property
    def embeddings(self) -> np.ndarray:
        if self._embeddings is None:
            self._embeddings = np.load(self.embedding_path, mmap_mode="r")
        return self._embeddings

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int):
        slide_index, local_index, global_index = self.rows[index]
        sample_id = self.sample_ids[int(slide_index)]
        exemplar_index = self.neighbors[sample_id][int(local_index)]
        return (
            torch.from_numpy(np.asarray(self.patches[global_index]).copy()),
            torch.from_numpy(self.all_targets[global_index]),
            torch.from_numpy(np.asarray(self.embeddings[global_index]).copy()),
            torch.from_numpy(np.asarray(self.embeddings[exemplar_index]).copy()),
            torch.from_numpy(self.all_targets[exemplar_index]),
            int(slide_index),
            int(local_index),
        )


def batch_correlation_loss(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    pred = prediction - prediction.mean(dim=0, keepdim=True)
    true = target - target.mean(dim=0, keepdim=True)
    pred_ss = pred.square().sum(0)
    true_ss = true.square().sum(0)
    valid = (pred_ss > 1e-8) & (true_ss > 1e-8)
    denominator = torch.sqrt(torch.clamp(pred_ss * true_ss, min=1e-8))
    correlation = (pred * true).sum(0) / denominator
    if valid.any():
        return 1.0 - correlation[valid].mean()
    return prediction.sum() * 0.0


def model_input(batch: Sequence[torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    patch, _, own, exemplar, exemplar_target, _, _ = batch
    image = patch.to(device, non_blocking=True).permute(0, 3, 1, 2).float() / 255.0
    image = (image - MEAN.to(device)) / STD.to(device)
    return {
        "img": image,
        "p_feature": own.to(device, non_blocking=True).float().unsqueeze(1),
        "op_feature": exemplar.to(device, non_blocking=True).float(),
        "op_count": exemplar_target.to(device, non_blocking=True).float(),
    }


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
        rows.append(
            np.divide(
                np.sum(pred * true, 0),
                denominator,
                out=np.full_like(denominator, np.nan),
                where=denominator > 0,
            )
        )
    return float(np.nanmean(np.nanmean(np.stack(rows), axis=0)))


@torch.no_grad()
def predict(
    model: nn.Module,
    loader: DataLoader,
    sample_ids: Sequence[str],
    slide_sizes: Mapping[str, int],
    n_genes: int,
    device: torch.device,
    amp: bool,
) -> dict[str, np.ndarray]:
    model.eval()
    result = {
        sample_id: np.empty((slide_sizes[sample_id], n_genes), dtype=np.float32)
        for sample_id in sample_ids
    }
    for batch in loader:
        context = torch.autocast("cuda", dtype=torch.float16) if amp else nullcontext()
        with context:
            output = model(model_input(batch, device))
        for row, (slide_index, local_index) in enumerate(
            zip(batch[-2].numpy(), batch[-1].numpy())
        ):
            result[sample_ids[int(slide_index)]][int(local_index)] = (
                output[row].detach().float().cpu().numpy()
            )
    return result


def train_fold(args: argparse.Namespace) -> Path:
    embedding_path = EMBEDDING_ROOT / "resnet50_embeddings_float32.npy"
    embedding_metadata = EMBEDDING_ROOT / "resnet50_metadata.json"
    if not embedding_path.is_file() or not embedding_metadata.is_file():
        raise FileNotFoundError("run cache_benchmark_egn_embeddings.py --backbone resnet50")
    folds = load_folds()
    fold = folds[args.fold]
    fixed_final_epoch = (
        os.environ.get("LIN_MNP_CHECKPOINT_MODE") == "fixed_final_epoch"
    )
    slides, genes = load_slides()
    parameters = fit_target_transform(slides, fold.train_ids, "egn_minmax_log10_raw")
    total_spots = max(slide.patch_stop for slide in slides.values())
    all_targets = np.empty((total_spots, len(genes)), dtype=np.float32)
    for slide in slides.values():
        all_targets[slide.patch_start : slide.patch_stop] = apply_target_transform(
            slide, parameters
        )
    embeddings = np.load(embedding_path, mmap_mode="r")
    all_ids = (*fold.train_ids, *fold.val_ids, *fold.test_ids)
    neighbor_map = build_neighbors(slides, all_ids, fold.train_ids, embeddings, 9)
    del embeddings

    output = (args.output / "egnv1" / f"fold{fold.fold_id}").resolve()
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
    model = build_model(len(genes)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5, weight_decay=1e-4)
    train_data = EGNDataset(slides, fold.train_ids, all_targets, neighbor_map, embedding_path)
    val_data = EGNDataset(slides, fold.val_ids, all_targets, neighbor_map, embedding_path)
    test_data = EGNDataset(slides, fold.test_ids, all_targets, neighbor_map, embedding_path)
    options = dict(batch_size=32, num_workers=args.workers, pin_memory=True)
    train_loader = DataLoader(train_data, shuffle=True, drop_last=True, **options)
    val_loader = DataLoader(val_data, shuffle=False, drop_last=False, **options)
    test_loader = DataLoader(test_data, shuffle=False, drop_last=False, **options)
    total_steps = args.epochs * len(train_loader)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda step: 0.5 * (1.0 + math.cos(math.pi * min(step, total_steps) / total_steps)),
    )
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp)
    slide_sizes = {sample: len(slides[sample].barcodes) for sample in slides}
    val_targets = {
        sample: all_targets[slides[sample].patch_start : slides[sample].patch_stop]
        for sample in fold.val_ids
    }
    best_score = -np.inf
    best_epoch = -1
    history = []
    checkpoint = staging / "best_checkpoint.pt"
    started = time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        loss_sum = 0.0
        count = 0
        for batch in train_loader:
            target = batch[1].to(device, non_blocking=True).float()
            optimizer.zero_grad(set_to_none=True)
            context = torch.autocast("cuda", dtype=torch.float16) if args.amp else nullcontext()
            with context:
                prediction = model(model_input(batch, device))
                loss = F.mse_loss(prediction, target) + 0.5 * batch_correlation_loss(
                    prediction, target
                )
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            loss_sum += float(loss.detach()) * len(target)
            count += len(target)
        val_prediction = predict(
            model, val_loader, fold.val_ids, slide_sizes, len(genes), device, args.amp
        )
        score = mean_slide_gene_pcc(val_prediction, val_targets)
        if fixed_final_epoch or (np.isfinite(score) and score > best_score):
            best_score = score
            best_epoch = epoch
            torch.save(
                {"model_state": model.state_dict(), "epoch": epoch, "validation_mean_gene_pcc": score},
                checkpoint,
            )
        record = {
            "epoch": epoch,
            "train_objective": loss_sum / max(count, 1),
            "validation_mean_gene_pcc": score,
            "best_validation_mean_gene_pcc": best_score,
            "elapsed_seconds": time.time() - started,
        }
        history.append(record)
        print(json.dumps(record, sort_keys=True), flush=True)
    state = torch.load(checkpoint, map_location=device)
    model.load_state_dict(state["model_state"])
    test_prediction = predict(
        model, test_loader, fold.test_ids, slide_sizes, len(genes), device, args.amp
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
        "method_id": "egnv1",
        "display_name": "EGNv1",
        "source_dir": "comparators/methods/EGN/v1",
        "source_kind": "released EGN class with benchmark 997-gene dimensional adapter",
        "target_transform": "egn_minmax_log10_raw",
        "epochs": args.epochs,
        "batch_size": 32,
        "learning_rate": 5e-5,
        "weight_decay": 1e-4,
        "input_size": 224,
        "augmentation": "released normalization; no augmentation",
        "optimizer": "AdamW + cosine annealing",
        "nearest_exemplars": 9,
        "runtime_compatibility_patches": [
            f"250-to-{len(genes)} gene-dependent projections",
            "released 8x8 EB reshape changed to benchmark-required 7x7 token grid",
            "zero-variance genes masked before batch-PCC square-root differentiation",
        ],
    }
    metadata = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "method": method,
        "source_revision": subprocess.check_output(
            ["git", "-C", str(PROJECT_ROOT / "comparators/methods/EGN"), "rev-parse", "HEAD"], text=True
        ).strip(),
        "fold": {
            "fold_id": fold.fold_id,
            "train_ids": list(fold.train_ids),
            "val_ids": list(fold.val_ids),
            "test_ids": list(fold.test_ids),
        },
        "seed": seed,
        "device": str(device),
        "mixed_precision": bool(args.amp),
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
        "embedding_metadata_sha256": sha256_file(embedding_metadata),
        "embedding_sha256": sha256_file(embedding_path),
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
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    print(json.dumps({"output": str(train_fold(parse_args(argv)))}, sort_keys=True))


if __name__ == "__main__":
    main()
