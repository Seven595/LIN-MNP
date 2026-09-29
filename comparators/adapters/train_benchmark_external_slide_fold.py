#!/usr/bin/env python3
"""Train one released whole-slide comparator on one official cSCC fold.

The adapter calls the author network classes for HisToGene, Hist2ST and
THItoGene.  Only data loading and the optimizer loop are replaced, because the
released repositories hard-code their own file layouts and single-slide
holdouts.  Patient train/validation/test membership and validation-only epoch
selection come from the shared HEtoSGEBench contract.
"""

from __future__ import annotations

import argparse
from contextlib import nullcontext
import csv
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import time
import types
from typing import Any, Mapping, Sequence

import numpy as np
from scipy.spatial import cKDTree
import torch
import torch.nn as nn
import torch.nn.functional as F


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from lin_mnp.benchmark.benchmark_external import (  # noqa: E402
    PATCH_ROOT,
    SlideBundle,
    apply_target_transform,
    fit_target_transform,
    load_folds,
    load_slides,
)
from lin_mnp.paths import RESULTS_ROOT


DEFAULT_OUTPUT = RESULTS_ROOT / "comparator_reruns"


@dataclass(frozen=True)
class MethodConfig:
    method_id: str
    display_name: str
    source_dir: str
    source_kind: str
    target_transform: str
    epochs: int
    learning_rate: float
    input_size: int
    augmentation: str
    loss: str


METHODS = {
    "histogene": MethodConfig(
        method_id="histogene",
        display_name="HisToGene",
        source_dir="comparators/methods/mclSTExp/baselines/HisToGene",
        source_kind="released author HisToGene class with patient-CV data adapter",
        target_transform="log1p_cpm",
        epochs=100,
        learning_rate=1e-5,
        input_size=112,
        augmentation="none in released ViT whole-slide dataset path",
        loss="MSE",
    ),
    "hist2st": MethodConfig(
        method_id="hist2st",
        display_name="Hist2ST",
        source_dir="comparators/methods/Hist2ST",
        source_kind="released author Hist2ST class with benchmark channel=16/patch=224 settings",
        target_transform="log1p_cpm",
        epochs=350,
        learning_rate=1e-5,
        input_size=224,
        augmentation="five-view released self-distillation transform",
        loss="MSE + 0.25 ZINB + 0.5 self-distillation",
    ),
    "thitogene": MethodConfig(
        method_id="thitogene",
        display_name="THItoGene",
        source_dir="comparators/methods/THItoGene",
        source_kind="released author THItoGene class with patient-CV data adapter",
        target_transform="log1p_cpm",
        epochs=200,
        learning_rate=1e-5,
        input_size=112,
        augmentation="none in released whole-slide training path",
        loss="released-code MSE (paper text says NLL but no NLL head/loss is released)",
    ),
}

# The released implementations use quadratic whole-slide attention (and, for
# Hist2ST/THItoGene, dense graph operators).  HEST contains up to ~21k spots,
# well beyond the few-hundred-spot slides bundled by the authors.  These limits
# were established by forward/backward memory smoke tests on a 24 GiB GPU.
# Only target-free spatial partitioning is added; every spot remains in every
# epoch and predictions are restored to canonical barcode order.
MAX_SPOTS_PER_SPATIAL_BLOCK = {
    "histogene": 4096,
    "hist2st": 1024,
    "thitogene": 2048,
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _install_import_stubs() -> None:
    if "pytorch_lightning" not in sys.modules:
        lightning = types.ModuleType("pytorch_lightning")

        class LightningModule(nn.Module):
            def log(self, *args: Any, **kwargs: Any) -> None:
                return None

            def save_hyperparameters(self, *args: Any, **kwargs: Any) -> None:
                return None

        lightning.LightningModule = LightningModule
        sys.modules["pytorch_lightning"] = lightning
    if "torchmetrics.functional" not in sys.modules:
        torchmetrics = types.ModuleType("torchmetrics")
        functional = types.ModuleType("torchmetrics.functional")
        functional.accuracy = lambda prediction, target, *args, **kwargs: (
            prediction == target
        ).float().mean()
        torchmetrics.functional = functional
        sys.modules["torchmetrics"] = torchmetrics
        sys.modules["torchmetrics.functional"] = functional


def source_revision(config: MethodConfig) -> str:
    path = PROJECT_ROOT / config.source_dir
    repository = path
    while repository != PROJECT_ROOT and not (repository / ".git").exists():
        repository = repository.parent
    return subprocess.check_output(
        ["git", "-C", str(repository), "rev-parse", "HEAD"], text=True
    ).strip()


def build_model(method: str, n_genes: int) -> nn.Module:
    _install_import_stubs()
    config = METHODS[method]
    source = PROJECT_ROOT / config.source_dir
    sys.path.insert(0, str(source))
    if method == "histogene":
        from vis_model import HisToGene  # type: ignore

        return HisToGene(
            patch_size=112,
            n_layers=8,
            n_genes=n_genes,
            dim=1024,
            learning_rate=config.learning_rate,
            dropout=0.1,
            n_pos=64,
        )
    if method == "hist2st":
        from HIST2ST import Hist2ST  # type: ignore

        return Hist2ST(
            learning_rate=config.learning_rate,
            fig_size=224,
            dropout=0.2,
            n_pos=64,
            kernel_size=5,
            patch_size=7,
            n_genes=n_genes,
            depth1=2,
            depth2=8,
            depth3=4,
            heads=16,
            channel=16,
            zinb=0.25,
            nb=False,
            bake=5,
            lamb=0.5,
            policy="mean",
        )
    if method == "thitogene":
        # vis_model imports every name from utils, although the THItoGene
        # network class itself uses none of them.  Avoid pulling the unrelated
        # plotting/data dependency scprep into the architecture import.
        sys.modules.setdefault("utils", types.ModuleType("utils"))
        from vis_model import THItoGene  # type: ignore

        return THItoGene(
            n_layers=4,
            n_genes=n_genes,
            learning_rate=config.learning_rate,
            dropout=0.2,
            n_pos=64,
            heads=[16, 8],
            caps=20,
            route_dim=64,
        )
    raise ValueError(f"unknown method: {method}")


def adjacency(coordinates: np.ndarray, neighbors: int = 4) -> torch.Tensor:
    coordinates = np.asarray(coordinates, dtype=np.float64)
    tree = cKDTree(coordinates)
    _, indices = tree.query(coordinates, k=min(neighbors + 1, len(coordinates)))
    matrix = np.zeros((len(coordinates), len(coordinates)), dtype=np.float32)
    for row, values in enumerate(np.atleast_2d(indices)):
        matrix[row, np.asarray(values[1:], dtype=int)] = 1.0
    return torch.from_numpy(matrix)


def spatial_blocks(coordinates: np.ndarray, maximum: int) -> tuple[np.ndarray, ...]:
    """Recursively bisect the longest spatial axis into deterministic blocks."""

    coordinates = np.asarray(coordinates, dtype=np.float64)
    pending = [np.arange(len(coordinates), dtype=np.int64)]
    result: list[np.ndarray] = []
    while pending:
        indices = pending.pop()
        if len(indices) <= maximum:
            result.append(np.sort(indices))
            continue
        current = coordinates[indices]
        axis = int(np.argmax(np.ptp(current, axis=0)))
        # The canonical spot index is an explicit secondary key, making ties
        # deterministic without looking at expression or patient identity.
        order = np.lexsort((indices, current[:, axis]))
        middle = len(indices) // 2
        left, right = indices[order[:middle]], indices[order[middle:]]
        pending.extend((right, left))
    result.sort(key=lambda value: int(value.min()))
    if sum(map(len, result)) != len(coordinates):
        raise RuntimeError("spatial partition lost spots")
    joined = np.concatenate(result)
    if len(np.unique(joined)) != len(coordinates):
        raise RuntimeError("spatial partition duplicated spots")
    return tuple(result)


def load_patch_tensor(
    slide: SlideBundle, input_size: int, indices: np.ndarray | None = None
) -> torch.Tensor:
    cache = np.load(PATCH_ROOT / "patches_224_uint8.npy", mmap_mode="r")
    selection: slice | np.ndarray
    if indices is None:
        selection = slice(slide.patch_start, slide.patch_stop)
    else:
        selection = slide.patch_start + np.asarray(indices, dtype=np.int64)
    patches = torch.from_numpy(
        np.asarray(cache[selection]).copy()
    ).permute(0, 3, 1, 2)
    if input_size != 224:
        # Area interpolation is deterministic and retains the released 0..255
        # tensor scale used by these three repositories.
        patches = F.interpolate(
            patches.float(), size=(input_size, input_size), mode="area"
        ).round().clamp_(0, 255).to(torch.uint8)
    return patches


def embedding_positions(coordinates: np.ndarray, n_pos: int = 64) -> np.ndarray:
    """Map an arbitrary assay grid to the released finite position tables.

    The three released networks index two ``nn.Embedding(n_pos, ...)`` tables.
    Their bundled datasets already provide small, zero-based integer grids,
    whereas Visium/HEST coordinates can be offset and can exceed 63 (including
    array-column encodings with gaps).  A per-slide monotone affine map is the
    target-free extension of that released convention: it retains the two
    spatial axes and endpoints while guaranteeing valid embedding indices.
    """

    value = np.asarray(coordinates, dtype=np.float64)
    if value.ndim != 2 or value.shape[1] != 2:
        raise ValueError(f"expected N x 2 coordinates, got {value.shape}")
    result = np.zeros(value.shape, dtype=np.int64)
    for axis in range(2):
        current = value[:, axis]
        minimum = float(np.min(current))
        span = float(np.max(current) - minimum)
        if span > 0:
            result[:, axis] = np.rint((current - minimum) * (n_pos - 1) / span).astype(
                np.int64
            )
    return np.clip(result, 0, n_pos - 1)


def _batch(
    slide: SlideBundle,
    target_parameters: Mapping[str, np.ndarray | str],
    input_size: int,
    device: torch.device,
    indices: np.ndarray | None = None,
) -> dict[str, torch.Tensor]:
    selection: slice | np.ndarray = slice(None) if indices is None else indices
    patches = load_patch_tensor(slide, input_size, indices).to(device=device).float()
    positions = torch.from_numpy(
        embedding_positions(slide.array_coordinates)[selection]
    ).to(device)
    target = torch.from_numpy(
        apply_target_transform(slide, target_parameters)[selection]
    ).to(device)
    graph = adjacency(slide.array_coordinates[selection], neighbors=4).to(device)
    raw = torch.from_numpy(slide.raw_counts[selection]).to(device)
    totals = raw.sum(dim=1)
    positive = totals[totals > 0]
    median = torch.median(positive) if len(positive) else torch.tensor(1.0, device=device)
    size_factor = totals / torch.clamp(median, min=1.0)
    return {
        "patches": patches,
        "positions": positions,
        "target": target,
        "adjacency": graph,
        "raw": raw,
        "size_factor": size_factor,
    }


def forward_model(method: str, model: nn.Module, batch: Mapping[str, torch.Tensor]) -> torch.Tensor:
    patch = batch["patches"]
    position = batch["positions"]
    if method == "histogene":
        flattened = patch.permute(0, 2, 3, 1).reshape(len(patch), -1)
        return model(flattened.unsqueeze(0), position.unsqueeze(0)).squeeze(0)
    if method == "hist2st":
        prediction, _, _ = model(
            patch.unsqueeze(0), position.unsqueeze(0), batch["adjacency"]
        )
        return prediction
    if method == "thitogene":
        return model(patch.unsqueeze(0), position.unsqueeze(0), batch["adjacency"])
    raise ValueError(method)


def training_loss(method: str, model: nn.Module, batch: Mapping[str, torch.Tensor]) -> torch.Tensor:
    if method == "hist2st":
        packed = (
            batch["patches"].unsqueeze(0),
            batch["positions"].unsqueeze(0),
            batch["target"].unsqueeze(0),
            batch["adjacency"].unsqueeze(0),
            batch["raw"].unsqueeze(0),
            batch["size_factor"].unsqueeze(0),
        )
        return model.training_step(packed, 0)
    prediction = forward_model(method, model, batch)
    return F.mse_loss(prediction.view_as(batch["target"]), batch["target"])


def _pcc(true: np.ndarray, pred: np.ndarray) -> np.ndarray:
    true = true.astype(np.float64) - true.mean(axis=0, keepdims=True)
    pred = pred.astype(np.float64) - pred.mean(axis=0, keepdims=True)
    numerator = np.sum(true * pred, axis=0)
    denominator = np.sqrt(np.sum(true * true, axis=0) * np.sum(pred * pred, axis=0))
    return np.divide(
        numerator,
        denominator,
        out=np.full_like(numerator, np.nan),
        where=denominator > 0,
    )


@torch.no_grad()
def predict_slides(
    method: str,
    model: nn.Module,
    slides: Mapping[str, SlideBundle],
    sample_ids: Sequence[str],
    target_parameters: Mapping[str, np.ndarray | str],
    input_size: int,
    device: torch.device,
    amp: bool = False,
) -> dict[str, np.ndarray]:
    model.eval()
    predictions = {}
    maximum = MAX_SPOTS_PER_SPATIAL_BLOCK[method]
    for sample_id in sample_ids:
        slide = slides[sample_id]
        prediction = np.empty((len(slide.barcodes), len(slide.genes)), dtype=np.float32)
        for indices in spatial_blocks(slide.array_coordinates, maximum):
            batch = _batch(slide, target_parameters, input_size, device, indices)
            amp_context = (
                torch.autocast(device_type="cuda", dtype=torch.float16)
                if amp and device.type == "cuda"
                else nullcontext()
            )
            with amp_context:
                current = forward_model(method, model, batch)
            prediction[indices] = current.detach().float().cpu().numpy().astype(np.float32)
            del batch, current
        predictions[sample_id] = prediction
    return predictions


def validation_pcc(
    predictions: Mapping[str, np.ndarray],
    slides: Mapping[str, SlideBundle],
    target_parameters: Mapping[str, np.ndarray | str],
) -> float:
    values = []
    for sample_id in sorted(predictions):
        target = apply_target_transform(slides[sample_id], target_parameters)
        values.append(_pcc(target, predictions[sample_id]))
    return float(np.nanmean(np.nanmean(np.stack(values), axis=0)))


def train_fold(args: argparse.Namespace) -> Path:
    config = METHODS[args.method]
    folds = load_folds()
    fold = folds[args.fold]
    slides, genes = load_slides()
    target_parameters = fit_target_transform(
        slides, fold.train_ids, config.target_transform
    )
    epochs = config.epochs if args.max_epochs is None else args.max_epochs
    fixed_final_epoch = (
        os.environ.get("LIN_MNP_CHECKPOINT_MODE") == "fixed_final_epoch"
    )

    fold_output = (args.output / config.method_id / f"fold{fold.fold_id}").resolve()
    if (fold_output / "complete.json").is_file():
        return fold_output
    if fold_output.exists():
        raise FileExistsError(f"incomplete fold output already exists: {fold_output}")
    staging = fold_output.with_name(f"{fold_output.name}.tmp-{os.getpid()}")
    staging.mkdir(parents=True, exist_ok=False)

    seed = args.seed + fold.fold_id
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    device = torch.device(args.device)
    model = build_model(config.method_id, len(genes)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp)
    checkpoint_path = staging / "best_checkpoint.pt"
    history = []
    best_score = float("-inf")
    best_epoch = -1
    start_time = time.time()

    for epoch in range(1, epochs + 1):
        model.train()
        train_ids = list(fold.train_ids)
        random.shuffle(train_ids)
        loss_sum = 0.0
        spot_count = 0
        for sample_id in train_ids:
            slide = slides[sample_id]
            blocks = spatial_blocks(
                slide.array_coordinates,
                MAX_SPOTS_PER_SPATIAL_BLOCK[config.method_id],
            )
            optimizer.zero_grad(set_to_none=True)
            for indices in blocks:
                batch = _batch(
                    slide, target_parameters, config.input_size, device, indices
                )
                amp_context = (
                    torch.autocast(device_type="cuda", dtype=torch.float16)
                    if args.amp and device.type == "cuda"
                    else nullcontext()
                )
                with amp_context:
                    loss = training_loss(config.method_id, model, batch)
                    weighted_loss = loss * (len(indices) / len(slide.barcodes))
                scaler.scale(weighted_loss).backward()
                loss_sum += float(loss.detach().cpu()) * len(indices)
                spot_count += len(indices)
                del batch, loss, weighted_loss
            scaler.step(optimizer)
            scaler.update()
        val_predictions = predict_slides(
            config.method_id,
            model,
            slides,
            fold.val_ids,
            target_parameters,
            config.input_size,
            device,
            args.amp,
        )
        score = validation_pcc(val_predictions, slides, target_parameters)
        if fixed_final_epoch or (np.isfinite(score) and score > best_score):
            best_score = score
            best_epoch = epoch
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "epoch": epoch,
                    "validation_mean_gene_pcc": score,
                    "method_config": asdict(config),
                },
                checkpoint_path,
            )
        record = {
            "epoch": epoch,
            "train_objective": loss_sum / max(spot_count, 1),
            "validation_mean_gene_pcc": score,
            "best_validation_mean_gene_pcc": best_score,
            "elapsed_seconds": time.time() - start_time,
        }
        history.append(record)
        print(json.dumps(record, sort_keys=True), flush=True)

    if best_epoch < 0:
        raise RuntimeError("no finite validation PCC checkpoint was produced")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state"])
    test_predictions = predict_slides(
        config.method_id,
        model,
        slides,
        fold.test_ids,
        target_parameters,
        config.input_size,
        device,
        args.amp,
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
            prediction=test_predictions[sample_id],
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
    metadata = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "method": asdict(config),
        "source_revision": source_revision(config),
        "fold": asdict(fold),
        "seed": seed,
        "device": str(device),
        "mixed_precision": bool(args.amp),
        "formal_paper_epoch_count": args.max_epochs is None,
        "executed_epochs": epochs,
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
                "LIN_MNP_CHECKPOINT_SELECTION_SCOPE",
                (
                    "outer_test_HESpotEx_protocol"
                    if os.environ.get("LIN_MNP_EXTERNAL_CONTRACT")
                    else "inner_validation"
                ),
            )
        ),
        "best_validation_mean_gene_pcc": best_score,
        "target_parameters": {
            key: value.tolist() if isinstance(value, np.ndarray) else value
            for key, value in target_parameters.items()
        },
        "patch_manifest_sha256": sha256_file(PATCH_ROOT / "manifest.json"),
        "position_index_adapter": {
            "kind": "per_slide_axiswise_affine_round",
            "n_pos": 64,
            "input": "array_coordinates",
            "uses_expression_or_labels": False,
        },
        "large_slide_adapter": {
            "kind": "recursive_longest_axis_spatial_bisection",
            "maximum_spots_per_block": MAX_SPOTS_PER_SPATIAL_BLOCK[config.method_id],
            "all_training_spots_used_each_epoch": True,
            "training_gradient_policy": "spot_weighted_block_losses; one optimizer step per slide",
            "prediction_reassembly": "canonical_barcode_order",
            "uses_expression_or_labels_for_partition": False,
        },
        "predictions_contain_truth": False,
        "predictions": prediction_rows,
        "wall_time_seconds": time.time() - start_time,
    }
    (staging / "run_metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    complete = {
        "run_metadata_sha256": sha256_file(staging / "run_metadata.json"),
        "training_history_sha256": sha256_file(staging / "training_history.csv"),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "prediction_sha256": {row["sample_id"]: row["sha256"] for row in prediction_rows},
    }
    (staging / "complete.json").write_text(
        json.dumps(complete, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    fold_output.parent.mkdir(parents=True, exist_ok=True)
    staging.rename(fold_output)
    return fold_output


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=sorted(METHODS), required=True)
    parser.add_argument("--fold", type=int, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--max-epochs", type=int)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    output = train_fold(parse_args(argv))
    print(json.dumps({"output": str(output)}, sort_keys=True))


if __name__ == "__main__":
    main()
