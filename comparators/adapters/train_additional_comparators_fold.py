#!/usr/bin/env python3
"""Run GR2ST, HECLIP, HE2Gene or asGNN on one locked benchmark fold.

The script is an explicit data adapter around the released model code.  It
keeps the official patient split, the ordered 997-gene panel and the immutable
224-pixel patch cache.  Test targets are not used for fitting or checkpoint
selection; prediction archives deliberately contain no truth matrix.
"""

from __future__ import annotations

import argparse
from contextlib import nullcontext
import copy
import csv
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import time
from typing import Any, Mapping, Sequence

import numpy as np
from scipy.spatial.distance import cdist
from sklearn.cluster import AffinityPropagation
from sklearn.decomposition import PCA
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


DEFAULT_OUTPUT = RESULTS_ROOT / "additional_comparator_reruns"
PHIKON_ROOT = Path(os.environ.get(
    "LIN_MNP_PHIKON_ROOT",
    str(CACHE_ROOT / "cscc/phikon_v2"),
))
RESNET_ROOT = Path(os.environ.get(
    "LIN_MNP_RESNET_ROOT",
    str(CACHE_ROOT / "cscc/deeppt_resnet50"),
))


@dataclass(frozen=True)
class MethodConfig:
    method_id: str
    display_name: str
    source_dir: str
    source_kind: str
    target_transform: str
    epochs: int
    batch_size: int
    learning_rate: float
    weight_decay: float
    early_stop_patience: int | None
    input_size: int
    inference: str


METHODS = {
    "gr2st": MethodConfig(
        "gr2st", "GR2ST", "comparators/methods/GR2ST",
        "released GR2ST network with locked997 and label-free cell-type adapter",
        "log1p_cp10k", 100, 256, 1e-4, 1e-3, None, 224,
        "direct image-branch MoE regression",
    ),
    "heclip": MethodConfig(
        "heclip", "HECLIP", "comparators/methods/HECLIP",
        "released HECLIP network and image-centric contrastive objective",
        "log1p_cp10k", 15, 128, 1e-3, 1e-3, None, 224,
        "top-50 training-reference expression retrieval",
    ),
    "he2gene": MethodConfig(
        "he2gene", "HE2Gene", "comparators/methods/HE2Gene",
        "released HE2Gene-Spat ResNet50 with unavailable pathology loss disabled",
        "log1p_raw", 100, 32, 1e-5, 1e-4, 10, 224,
        "direct target-plus-auxiliary gene heads",
    ),
    "asgnn": MethodConfig(
        "asgnn", "asGNN", "comparators/methods/asGNN",
        "released GTN and affinity-clustering graph equations with locked997 adapter",
        "log1p_raw", 201, 1, 5e-3, 0.0, 60, 224,
        "section graph node regression",
    ),
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def source_revision(relative: str) -> str:
    return subprocess.check_output(
        ["git", "-C", str(PROJECT_ROOT / relative), "rev-parse", "HEAD"],
        text=True,
    ).strip()


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def finite_mean_gene_pcc(
    predictions: Mapping[str, np.ndarray],
    targets: Mapping[str, np.ndarray],
) -> float:
    values: list[float] = []
    for sample_id, prediction in predictions.items():
        target = np.asarray(targets[sample_id], dtype=np.float64)
        prediction = np.asarray(prediction, dtype=np.float64)
        target_centered = target - target.mean(axis=0, keepdims=True)
        pred_centered = prediction - prediction.mean(axis=0, keepdims=True)
        denominator = np.sqrt(
            np.square(target_centered).sum(axis=0)
            * np.square(pred_centered).sum(axis=0)
        )
        pcc = np.divide(
            (target_centered * pred_centered).sum(axis=0),
            denominator,
            out=np.full(target.shape[1], np.nan),
            where=denominator > 1e-12,
        )
        values.extend(pcc[np.isfinite(pcc)].tolist())
    return float(np.mean(values)) if values else float("nan")


class IndexedPatchDataset(Dataset):
    def __init__(self, slides, sample_ids, target_parameters, *, augment: bool = False):
        self.slides = slides
        self.sample_ids = tuple(sample_ids)
        self.augment = augment
        self.patch_path = PATCH_ROOT / "patches_224_uint8.npy"
        self._patches = None
        self.global_indices = []
        self.targets = []
        self.slide_indices = []
        self.local_indices = []
        for slide_index, sample_id in enumerate(self.sample_ids):
            slide = slides[sample_id]
            self.global_indices.extend(range(slide.patch_start, slide.patch_stop))
            self.targets.append(apply_target_transform(slide, target_parameters))
            self.slide_indices.extend([slide_index] * len(slide.barcodes))
            self.local_indices.extend(range(len(slide.barcodes)))
        self.global_indices = np.asarray(self.global_indices, dtype=np.int64)
        self.targets = np.concatenate(self.targets).astype(np.float32)
        self.slide_indices = np.asarray(self.slide_indices, dtype=np.int16)
        self.local_indices = np.asarray(self.local_indices, dtype=np.int32)

    @property
    def patches(self):
        if self._patches is None:
            self._patches = np.load(self.patch_path, mmap_mode="r")
        return self._patches

    def __len__(self):
        return len(self.global_indices)

    def __getitem__(self, index):
        patch = np.asarray(self.patches[self.global_indices[index]]).copy()
        return (
            torch.from_numpy(patch),
            torch.from_numpy(self.targets[index]),
            int(self.slide_indices[index]),
            int(self.local_indices[index]),
        )


class HE2GeneDataset(IndexedPatchDataset):
    def __init__(self, slides, sample_ids, target_parameters, neighbor_lookup, *, augment):
        super().__init__(slides, sample_ids, target_parameters, augment=augment)
        self.neighbor_lookup = neighbor_lookup

    def __getitem__(self, index):
        patch, target, slide_index, local_index = super().__getitem__(index)
        sample_id = self.sample_ids[slide_index]
        neighbors = self.neighbor_lookup[sample_id][local_index]
        slide = self.slides[sample_id]
        neighbor_global = slide.patch_start + neighbors
        neighbor_patches = np.asarray(self.patches[neighbor_global]).copy()
        return patch, torch.from_numpy(neighbor_patches), target, slide_index, local_index


def augment_batch(images: torch.Tensor) -> torch.Tensor:
    result = []
    for image in images:
        if torch.rand(()) > 0.5:
            image = torch.flip(image, dims=(-1,))
        if torch.rand(()) > 0.5:
            image = torch.flip(image, dims=(-2,))
        image = torch.rot90(image, int(torch.randint(0, 4, ()).item()), (-2, -1))
        result.append(image)
    return torch.stack(result)


def patch_tensor(
    patches: torch.Tensor,
    device: torch.device,
    *,
    augment: bool,
    normalize: tuple[torch.Tensor, torch.Tensor] | None,
    bgr_255: bool = False,
) -> torch.Tensor:
    image = patches.to(device=device, non_blocking=True).permute(0, 3, 1, 2).float()
    if not bgr_255:
        image = image / 255.0
    else:
        image = image[:, [2, 1, 0]]
    if augment:
        image = augment_batch(image)
    if normalize is not None:
        mean, std = normalize
        image = (image - mean) / std
    return image


def train_image_mean_std(dataset: IndexedPatchDataset) -> tuple[np.ndarray, np.ndarray]:
    patches = np.load(dataset.patch_path, mmap_mode="r")
    channel_sum = np.zeros(3, dtype=np.float64)
    channel_square_sum = np.zeros(3, dtype=np.float64)
    pixel_count = 0
    for start in range(0, len(dataset.global_indices), 128):
        values = np.asarray(
            patches[dataset.global_indices[start:start + 128]], dtype=np.float32
        ) / 255.0
        channel_sum += values.sum(axis=(0, 1, 2), dtype=np.float64)
        channel_square_sum += np.square(values).sum(axis=(0, 1, 2), dtype=np.float64)
        pixel_count += int(np.prod(values.shape[:3]))
    mean64 = channel_sum / pixel_count
    variance = np.maximum(channel_square_sum / pixel_count - np.square(mean64), 0)
    mean = mean64.astype(np.float32)
    std = np.sqrt(variance).astype(np.float32)
    std[std < 1e-6] = 1.0
    return mean, std


def split_predictions(flat: np.ndarray, sample_ids, slides) -> dict[str, np.ndarray]:
    output = {}
    start = 0
    for sample_id in sample_ids:
        stop = start + len(slides[sample_id].barcodes)
        output[sample_id] = flat[start:stop].astype(np.float32, copy=False)
        start = stop
    if start != len(flat):
        raise ValueError("prediction-to-slide mapping is incomplete")
    return output


def save_checkpoint(model: nn.Module, path: Path, **extra) -> dict[str, torch.Tensor]:
    state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    torch.save({"model_state": state, **extra}, path)
    return state


def load_official_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def predict_heclip_images(model, loader, device) -> np.ndarray:
    embeddings = []
    model.eval()
    with torch.no_grad():
        for patches, _, _, _ in loader:
            image = patch_tensor(
                patches, device, augment=False, normalize=None, bgr_255=True
            )
            feature = model.image_encoder(image)
            embeddings.append(model.image_projection(feature).cpu())
    return torch.cat(embeddings).numpy().astype(np.float32)


def heclip_retrieve(
    train_image_embeddings: np.ndarray,
    train_targets: np.ndarray,
    query_embeddings: np.ndarray,
    top_k: int = 50,
) -> np.ndarray:
    reference = train_image_embeddings.astype(np.float64)
    query = query_embeddings.astype(np.float64)
    reference /= np.maximum(np.linalg.norm(reference, axis=1, keepdims=True), 1e-12)
    query /= np.maximum(np.linalg.norm(query, axis=1, keepdims=True), 1e-12)
    result = np.empty((len(query), train_targets.shape[1]), dtype=np.float32)
    k = min(top_k, len(reference))
    for start in range(0, len(query), 512):
        similarity = query[start:start + 512] @ reference.T
        indices = np.argpartition(similarity, -k, axis=1)[:, -k:]
        result[start:start + len(indices)] = train_targets[indices].mean(axis=1)
    return result


def run_heclip(context, args):
    config, slides, genes, fold, target_parameters, device, staging = context
    code_root = PROJECT_ROOT / "comparators/methods/HECLIP/code"
    if str(code_root) not in sys.path:
        sys.path.insert(0, str(code_root))
    from models_hvg import HECLIPModel  # type: ignore

    model_config = {
        "projection_dim": len(genes), "temperature": 1.0,
        "embedding_dim": 2048, "model_name": "resnet50", "dropout": 0.1,
    }
    model = HECLIPModel(model_config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp)
    train_data = IndexedPatchDataset(slides, fold.train_ids, target_parameters, augment=True)
    val_data = IndexedPatchDataset(slides, fold.val_ids, target_parameters)
    test_data = IndexedPatchDataset(slides, fold.test_ids, target_parameters)
    options = dict(num_workers=args.workers, pin_memory=True, persistent_workers=args.workers > 0)
    train_loader = DataLoader(train_data, batch_size=config.batch_size, shuffle=True, drop_last=True, **options)
    train_eval_loader = DataLoader(train_data, batch_size=config.batch_size, shuffle=False, **options)
    val_loader = DataLoader(val_data, batch_size=config.batch_size, shuffle=False, **options)
    test_loader = DataLoader(test_data, batch_size=config.batch_size, shuffle=False, **options)
    val_targets = {sample: apply_target_transform(slides[sample], target_parameters) for sample in fold.val_ids}
    best_score, best_epoch, best_state = -np.inf, -1, None
    history = []
    epochs = args.max_epochs or config.epochs
    started = time.time()
    for epoch in range(1, epochs + 1):
        model.train()
        total, count = 0.0, 0
        for patches, targets, _, _ in train_loader:
            image = patch_tensor(patches, device, augment=True, normalize=None, bgr_255=True)
            target = targets.to(device=device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            amp = torch.autocast("cuda", dtype=torch.float16) if args.amp else nullcontext()
            with amp:
                loss = model({"image": image, "reduced_expression": target})
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total += float(loss.detach()) * len(target)
            count += len(target)
        reference_embeddings = predict_heclip_images(model, train_eval_loader, device)
        query_embeddings = predict_heclip_images(model, val_loader, device)
        flat = heclip_retrieve(reference_embeddings, train_data.targets, query_embeddings)
        val_predictions = split_predictions(flat, fold.val_ids, slides)
        score = finite_mean_gene_pcc(val_predictions, val_targets)
        if (
            os.environ.get("LIN_MNP_CHECKPOINT_MODE") == "fixed_final_epoch"
            or (np.isfinite(score) and score > best_score)
        ):
            best_score, best_epoch = score, epoch
            best_state = save_checkpoint(
                model, staging / "best_checkpoint.pt", epoch=epoch,
                validation_mean_gene_pcc=score,
            )
        row = {
            "epoch": epoch, "train_contrastive_loss": total / max(count, 1),
            "validation_mean_gene_pcc": score,
            "best_validation_mean_gene_pcc": best_score,
            "elapsed_seconds": time.time() - started,
        }
        history.append(row)
        print(json.dumps(row, sort_keys=True), flush=True)
    if best_state is None:
        raise RuntimeError("HECLIP produced no validation checkpoint")
    model.load_state_dict(best_state)
    reference_embeddings = predict_heclip_images(model, train_eval_loader, device)
    query_embeddings = predict_heclip_images(model, test_loader, device)
    flat = heclip_retrieve(reference_embeddings, train_data.targets, query_embeddings)
    return split_predictions(flat, fold.test_ids, slides), history, best_epoch, best_score, {
        "top_k": 50,
        "official_image_value_contract": "BGR float 0..255 without ImageNet normalization",
        "locked_patch_adaptation": "common benchmark 224px cache instead of native 256px crop",
    }


def build_neighbor_lookup(slides) -> dict[str, np.ndarray]:
    lookup = {}
    for sample_id, slide in slides.items():
        distance = cdist(slide.array_coordinates, slide.array_coordinates)
        order = np.argsort(distance, axis=1)[:, 1:9]
        if order.shape[1] < 8:
            order = np.pad(order, ((0, 0), (0, 8 - order.shape[1])), mode="edge")
        lookup[sample_id] = order.astype(np.int64)
    return lookup


def he2gene_gene_split(genes: Sequence[str]) -> tuple[np.ndarray, np.ndarray]:
    target_path = (
        PROJECT_ROOT / "comparators/methods/HE2Gene/data/breast/target_genes.txt"
    )
    official = (
        {
            value.strip().casefold()
            for value in target_path.read_text(encoding="utf-8").splitlines()
            if value.strip()
        }
        if target_path.is_file()
        else set()
    )
    target = [index for index, gene in enumerate(genes) if gene.casefold() in official]
    target.extend(index for index in range(len(genes)) if index not in set(target))
    target = np.asarray(target[:250], dtype=np.int64)
    target_set = set(target.tolist())
    auxiliary = np.asarray([i for i in range(len(genes)) if i not in target_set], dtype=np.int64)
    return target, auxiliary


def predict_he2gene(model, loader, device, normalization, target_idx, aux_idx, n_genes):
    values = []
    model.eval()
    with torch.no_grad():
        for patches, _, _, _, _ in loader:
            image = patch_tensor(patches, device, augment=False, normalize=normalization)
            symmetries = []
            for flip in (False, True):
                base = torch.flip(image, dims=(-1,)) if flip else image
                symmetries.extend(torch.rot90(base, k, (-2, -1)) for k in range(4))
            stacked = torch.cat(symmetries, dim=0)
            target, auxiliary, _ = model(stacked, torch.empty(0, device=device))
            target = target.reshape(8, len(image), -1).mean(dim=0)
            auxiliary = auxiliary.reshape(8, len(image), -1).mean(dim=0)
            prediction = torch.empty((len(image), n_genes), device=device)
            prediction[:, target_idx] = target
            prediction[:, aux_idx] = auxiliary
            values.append(prediction.cpu())
    return torch.cat(values).numpy().astype(np.float32)


def run_he2gene(context, args):
    config, slides, genes, fold, target_parameters, device, staging = context
    module = load_official_module(
        "he2gene_official_resnet",
        PROJECT_ROOT / "comparators/methods/HE2Gene/src/models/resnet.py",
    )
    target_idx, aux_idx = he2gene_gene_split(genes)
    model = module.AuxAnnotateSpatialResNet50(
        len(target_idx), len(aux_idx), pretrain=True
    ).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp)
    neighbors = build_neighbor_lookup(slides)
    train_data = HE2GeneDataset(slides, fold.train_ids, target_parameters, neighbors, augment=True)
    val_data = HE2GeneDataset(slides, fold.val_ids, target_parameters, neighbors, augment=False)
    test_data = HE2GeneDataset(slides, fold.test_ids, target_parameters, neighbors, augment=False)
    mean, std = train_image_mean_std(train_data)
    normalization = (
        torch.tensor(mean, device=device).view(1, 3, 1, 1),
        torch.tensor(std, device=device).view(1, 3, 1, 1),
    )
    options = dict(num_workers=args.workers, pin_memory=True, persistent_workers=args.workers > 0)
    train_loader = DataLoader(train_data, batch_size=config.batch_size, shuffle=True, **options)
    val_loader = DataLoader(val_data, batch_size=max(config.batch_size, 16), shuffle=False, **options)
    test_loader = DataLoader(test_data, batch_size=max(config.batch_size, 16), shuffle=False, **options)
    val_targets = {sample: apply_target_transform(slides[sample], target_parameters) for sample in fold.val_ids}
    best_score, best_epoch, best_state, no_improvement = -np.inf, -1, None, 0
    history = []
    epochs = args.max_epochs or config.epochs
    started = time.time()
    target_t = torch.as_tensor(target_idx, device=device)
    aux_t = torch.as_tensor(aux_idx, device=device)
    for epoch in range(1, epochs + 1):
        model.train()
        total, count = 0.0, 0
        for patches, neighbor_patches, targets, _, _ in train_loader:
            image = patch_tensor(patches, device, augment=True, normalize=normalization)
            shape = neighbor_patches.shape
            neighbor = patch_tensor(
                neighbor_patches.reshape(-1, *shape[2:]), device,
                augment=True, normalize=normalization,
            ).reshape(shape[0], shape[1], 3, 224, 224)
            truth = targets.to(device=device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            amp = torch.autocast("cuda", dtype=torch.float16) if args.amp else nullcontext()
            with amp:
                target_pred, aux_pred, _, neighbor_target, _, _ = model(image, neighbor)
                target_loss = F.mse_loss(target_pred, truth.index_select(1, target_t))
                # On a fixed panel with <=250 genes every gene belongs to the
                # released target head, leaving the auxiliary head empty.  An
                # MSE over two empty tensors is NaN and would poison all model
                # parameters, so the absent branch contributes exactly zero.
                aux_loss = (
                    F.mse_loss(aux_pred, truth.index_select(1, aux_t))
                    if aux_t.numel()
                    else target_pred.sum() * 0.0
                )
                spatial_loss = F.mse_loss(target_pred, neighbor_target.mean(dim=1))
                loss = target_loss + 20.0 * aux_loss + 0.01 * spatial_loss
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            total += float(loss.detach()) * len(truth)
            count += len(truth)
        model.eval()
        flat = predict_he2gene(
            model, val_loader, device, normalization, target_t, aux_t, len(genes)
        )
        score = finite_mean_gene_pcc(
            split_predictions(flat, fold.val_ids, slides), val_targets
        )
        fixed_final_epoch = (
            os.environ.get("LIN_MNP_CHECKPOINT_MODE") == "fixed_final_epoch"
        )
        if fixed_final_epoch or (np.isfinite(score) and score > best_score):
            best_score, best_epoch, no_improvement = score, epoch, 0
            best_state = save_checkpoint(
                model, staging / "best_checkpoint.pt", epoch=epoch,
                validation_mean_gene_pcc=score,
            )
        else:
            no_improvement += 1
        row = {
            "epoch": epoch, "train_total_loss": total / max(count, 1),
            "validation_mean_gene_pcc": score,
            "best_validation_mean_gene_pcc": best_score,
            "learning_rate": optimizer.param_groups[0]["lr"],
            "elapsed_seconds": time.time() - started,
        }
        history.append(row)
        print(json.dumps(row, sort_keys=True), flush=True)
        scheduler.step()
        if (
            not fixed_final_epoch
            and config.early_stop_patience
            and no_improvement >= config.early_stop_patience
        ):
            break
    if best_state is None:
        raise RuntimeError("HE2Gene produced no validation checkpoint")
    model.load_state_dict(best_state)
    flat = predict_he2gene(model, test_loader, device, normalization, target_t, aux_t, len(genes))
    return split_predictions(flat, fold.test_ids, slides), history, best_epoch, best_score, {
        "target_gene_count": len(target_idx), "auxiliary_gene_count": len(aux_idx),
        "target_gene_rule": "released breast target list overlap then locked panel order",
        "auxiliary_loss_weight": 20.0 if len(aux_idx) else 0.0,
        "auxiliary_empty_branch_policy": "exact zero loss when the fixed panel has no auxiliary genes",
        "spatial_loss_weight": 0.01,
        "pathology_loss_weight": 0.0,
        "pathology_loss_reason": "the data contract has no tumor/normal spot annotations",
        "neighbor_rule": "eight nearest spots within each section",
        "inference_augmentation": "mean of the released eight dihedral symmetries",
        "normalization": {"train_only_rgb_mean": mean.tolist(), "train_only_rgb_std": std.tolist()},
    }


def load_phikon(slides) -> dict[str, np.ndarray]:
    result = {}
    requested_crop = int(os.environ.get("LIN_MNP_PHIKON_CROP_SIZE", "192"))
    for sample_id, slide in slides.items():
        with np.load(PHIKON_ROOT / "slides" / f"{sample_id}.npz", allow_pickle=False) as archive:
            if tuple(archive["barcodes"].astype(str)) != slide.barcodes:
                raise ValueError(f"Phikon barcode mismatch: {sample_id}")
            crops = archive["crop_sizes_px"].astype(int).tolist()
            if requested_crop not in crops:
                raise ValueError(
                    f"Phikon crop {requested_crop} missing for {sample_id}: {crops}"
                )
            index = crops.index(requested_crop)
            result[sample_id] = archive["cls"][:, index].astype(np.float32)
    return result


def spatial_chunks(slide, cap: int, seed: int) -> list[np.ndarray]:
    coords = np.asarray(slide.array_coordinates)
    order = np.lexsort((coords[:, 1], coords[:, 0]))
    chunks = [order[start:start + cap] for start in range(0, len(order), cap)]
    random.Random(seed).shuffle(chunks)
    return chunks


def predict_gr2st(model, features, sample_ids, device):
    result = {}
    model.eval()
    with torch.no_grad():
        for sample_id in sample_ids:
            values = []
            for start in range(0, len(features[sample_id]), 512):
                image = torch.from_numpy(features[sample_id][start:start + 512]).to(device)
                values.append(model.predict_expression(image).cpu())
            result[sample_id] = torch.cat(values).numpy().astype(np.float32)
    return result


def run_gr2st(context, args):
    config, slides, genes, fold, target_parameters, device, staging = context
    module = load_official_module(
        "gr2st_official_model", PROJECT_ROOT / "comparators/methods/GR2ST/GR2ST/GR2ST/model.py"
    )
    model = module.GR2ST(
        temperature=1.0, image_dim=1024, spot_dim=len(genes), projection_dim=256,
        heads_num=8, dropout=0.1, fusion_type="sum", alpha_mse=50.0,
        alpha_gate=1.0, alpha_entropy=0.01, spatial_radius=3.0,
        conf_threshold=0.6,
    ).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp)
    features = load_phikon(slides)
    targets = {sample: apply_target_transform(slides[sample], target_parameters) for sample in slides}
    best_score, best_epoch, best_state = -np.inf, -1, None
    history = []
    epochs = args.max_epochs or config.epochs
    started = time.time()
    for epoch in range(1, epochs + 1):
        model.train()
        total, count = 0.0, 0
        train_ids = list(fold.train_ids)
        random.Random(args.seed + fold.fold_id + epoch).shuffle(train_ids)
        for sample_id in train_ids:
            for indices in spatial_chunks(slides[sample_id], config.batch_size, epoch):
                batch = {
                    "image_features": torch.from_numpy(features[sample_id][indices]).to(device),
                    "expression": torch.from_numpy(targets[sample_id][indices]).to(device),
                    "position": torch.from_numpy(slides[sample_id].array_coordinates[indices]).to(device),
                    "cell_type": torch.zeros(len(indices), dtype=torch.long, device=device),
                }
                optimizer.zero_grad(set_to_none=True)
                amp = torch.autocast("cuda", dtype=torch.float16) if args.amp else nullcontext()
                with amp:
                    loss = model(batch)
                if not torch.isfinite(loss):
                    raise FloatingPointError("GR2ST non-finite loss")
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                total += float(loss.detach()) * len(indices)
                count += len(indices)
        validation = predict_gr2st(model, features, fold.val_ids, device)
        score = finite_mean_gene_pcc(
            validation, {sample: targets[sample] for sample in fold.val_ids}
        )
        if (
            os.environ.get("LIN_MNP_CHECKPOINT_MODE") == "fixed_final_epoch"
            or (np.isfinite(score) and score > best_score)
        ):
            best_score, best_epoch = score, epoch
            best_state = save_checkpoint(
                model, staging / "best_checkpoint.pt", epoch=epoch,
                validation_mean_gene_pcc=score,
            )
        row = {
            "epoch": epoch, "train_gr2st_loss": total / max(count, 1),
            "validation_mean_gene_pcc": score,
            "best_validation_mean_gene_pcc": best_score,
            "elapsed_seconds": time.time() - started,
        }
        history.append(row)
        print(json.dumps(row, sort_keys=True), flush=True)
    if best_state is None:
        raise RuntimeError("GR2ST produced no validation checkpoint")
    model.load_state_dict(best_state)
    return predict_gr2st(model, features, fold.test_ids, device), history, best_epoch, best_score, {
        "image_feature": (
            "Phikon-v2 CLS1024 at fixed "
            f"{int(os.environ.get('LIN_MNP_PHIKON_CROP_SIZE', '192'))}px crop"
        ),
        "cell_type_adapter": "single unknown class because the data contract has no spot cell-type labels",
        "graph_batch_cap": config.batch_size,
        "graph_batch_rule": "spatial lexicographic chunks, shuffled by epoch",
    }


def load_resnet_features(slides) -> dict[str, np.ndarray]:
    metadata = json.loads((RESNET_ROOT / "metadata.json").read_text(encoding="utf-8"))
    array = np.load(RESNET_ROOT / "resnet50_imagenet_v2_float32.npy", mmap_mode="r")
    result = {}
    rows = {row["sample_id"]: row for row in metadata["spot_order"]}
    for sample_id, slide in slides.items():
        row = rows[sample_id]
        values = np.asarray(array[int(row["start"]):int(row["stop"])]).copy()
        if len(values) != len(slide.barcodes):
            raise ValueError(f"ResNet feature length mismatch: {sample_id}")
        result[sample_id] = values
    return result


def knn_edges(coords: np.ndarray, k: int = 8) -> np.ndarray:
    distance = cdist(coords, coords)
    neighbor = np.argsort(distance, axis=1)[:, 1:k + 1]
    source = np.repeat(np.arange(len(coords)), neighbor.shape[1])
    target = neighbor.reshape(-1)
    edges = np.stack([np.concatenate([source, target]), np.concatenate([target, source])])
    return np.unique(edges, axis=1).astype(np.int64)


def graph_filter_affinity(
    feature: np.ndarray, edge_index: np.ndarray, p_factor: float, c_sp: float
) -> np.ndarray:
    n = len(feature)
    adjacency = [[] for _ in range(n)]
    for left, right in edge_index.T:
        adjacency[int(left)].append(int(right))
    shortest = np.full((n, n), 5.0, dtype=np.float32)
    np.fill_diagonal(shortest, 0.0)
    for start in range(n):
        frontier = [start]
        visited = {start}
        for level in range(1, 6):
            next_frontier = []
            for node in frontier:
                for neighbor in adjacency[node]:
                    if neighbor not in visited:
                        visited.add(neighbor)
                        next_frontier.append(neighbor)
                        shortest[start, neighbor] = level
            frontier = next_frontier
            if not frontier:
                break
    feature_distance = cdist(feature, feature).astype(np.float32)
    similarity = -(c_sp * shortest + feature_distance)
    median = float(np.median(similarity))
    np.fill_diagonal(similarity, median)
    labels = AffinityPropagation(
        damping=0.88, max_iter=1600, preference=median * p_factor,
        affinity="precomputed", random_state=0,
    ).fit_predict(similarity)
    keep = labels[edge_index[0]] == labels[edge_index[1]]
    filtered = edge_index[:, keep]
    loops = np.stack([np.arange(n), np.arange(n)])
    return np.unique(np.concatenate([filtered, loops], axis=1), axis=1).astype(np.int64)


def graph_loss(model, records, device, *, train: bool, optimizer=None):
    total, count = 0.0, 0
    if train:
        model.train()
    else:
        model.eval()
    context = nullcontext() if train else torch.no_grad()
    with context:
        for record in records:
            x = torch.from_numpy(record["x"]).to(device)
            y = torch.from_numpy(record["y"]).to(device)
            edges = torch.from_numpy(record["edge_index"]).to(device)
            batch = torch.zeros(len(x), dtype=torch.long, device=device)
            output = model(x, edges, batch)
            mse = F.mse_loss(output, y)
            oc = output - output.mean(dim=0)
            yc = y - y.mean(dim=0)
            denom = torch.linalg.vector_norm(oc, dim=0) * torch.linalg.vector_norm(yc, dim=0)
            corr = torch.where(denom > 1e-8, (oc * yc).sum(dim=0) / denom, 0).mean()
            loss = mse + 1.0 - corr
            if train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
            total += float(loss.detach())
            count += 1
    return total / max(count, 1)


def predict_graphs(model, records, device):
    result = {}
    model.eval()
    with torch.no_grad():
        for record in records:
            x = torch.from_numpy(record["x"]).to(device)
            edges = torch.from_numpy(record["edge_index"]).to(device)
            batch = torch.zeros(len(x), dtype=torch.long, device=device)
            result[record["sample_id"]] = model(x, edges, batch).cpu().numpy().astype(np.float32)
    return result


def train_graph_candidate(module, records, val_records, device, genes, hidden, lr, epochs, patience):
    model = module.GTN(hidden, input_dim=records[0]["x"].shape[1], num_classes=len(genes), num_layers=3).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    targets = {record["sample_id"]: record["y"] for record in val_records}
    best_score, best_state, best_epoch, no_improvement = -np.inf, None, -1, 0
    for epoch in range(1, epochs + 1):
        graph_loss(model, records, device, train=True, optimizer=optimizer)
        score = finite_mean_gene_pcc(predict_graphs(model, val_records, device), targets)
        fixed_final_epoch = (
            os.environ.get("LIN_MNP_CHECKPOINT_MODE") == "fixed_final_epoch"
        )
        if fixed_final_epoch or (np.isfinite(score) and score > best_score):
            best_score, best_epoch, no_improvement = score, epoch, 0
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        else:
            no_improvement += 1
        if not fixed_final_epoch and no_improvement >= patience:
            break
    if best_state is None:
        raise RuntimeError("asGNN candidate produced no validation checkpoint")
    model.load_state_dict(best_state)
    return model, best_score, best_epoch


def run_asgnn(context, args):
    config, slides, genes, fold, target_parameters, device, staging = context
    source = PROJECT_ROOT / "comparators/methods/asGNN"
    if str(source) not in sys.path:
        sys.path.insert(0, str(source))
    import GNN_core as module  # type: ignore

    features = load_resnet_features(slides)
    train_matrix = np.concatenate([features[sample] for sample in fold.train_ids])
    pca = PCA(n_components=32, whiten=True, random_state=args.seed + fold.fold_id)
    pca.fit(train_matrix)
    records = {}
    for sample_id, slide in slides.items():
        records[sample_id] = {
            "sample_id": sample_id,
            "x": pca.transform(features[sample_id]).astype(np.float32),
            "y": apply_target_transform(slide, target_parameters).astype(np.float32),
            "edge_index": knn_edges(slide.array_coordinates),
        }
    train_records = [records[sample] for sample in fold.train_ids]
    val_records = [records[sample] for sample in fold.val_ids]
    test_records = [records[sample] for sample in fold.test_ids]
    epochs = args.max_epochs or config.epochs
    if args.locked_adaptation is not None:
        locked = json.loads(args.locked_adaptation.read_text(encoding="utf-8"))
        hidden = int(locked["selected_hidden_channels"])
        lr = float(locked["selected_learning_rate"])
        p_factor = float(locked["selected_p_factor"])
        c_sp = float(locked["selected_c_sp"])
        for record in records.values():
            record["edge_index"] = graph_filter_affinity(
                record["x"], record["edge_index"], p_factor, c_sp
            )
        train_records = [records[sample] for sample in fold.train_ids]
        val_records = [records[sample] for sample in fold.val_ids]
        test_records = [records[sample] for sample in fold.test_ids]
        model, best_score, best_epoch = train_graph_candidate(
            module, train_records, val_records, device, genes, hidden, lr,
            epochs, epochs,
        )
        save_checkpoint(
            model, staging / "best_checkpoint.pt", epoch=best_epoch,
            validation_mean_gene_pcc=best_score,
        )
        history = [{
            "stage": "locked_inner_validation_choice_refit",
            "hidden_channels": hidden,
            "learning_rate": lr,
            "p_factor": p_factor,
            "c_sp": c_sp,
            "best_epoch": best_epoch,
            "validation_mean_gene_pcc": best_score,
        }]
        return predict_graphs(model, test_records, device), history, best_epoch, best_score, {
            "node_feature": "ImageNet-v2 ResNet50 2048D common cache",
            "train_only_feature_reduction": "whitened PCA32",
            "initial_graph": "symmetric spatial 8-nearest-neighbor graph",
            "adaptive_graph": "released AffinityPropagation similarity and edge filtering equations",
            "selected_hidden_channels": hidden,
            "selected_learning_rate": lr,
            "selected_p_factor": p_factor,
            "selected_c_sp": c_sp,
            "search_budget": {
                "backbone_candidates": 0,
                "adaptive_graph_candidates": 0,
                "choice_source": str(args.locked_adaptation.resolve()),
            },
        }
    search = [(50, 1e-3), (100, 5e-3), (250, 1e-2)]
    history = []
    started = time.time()
    initial_candidates = []
    for hidden, lr in search:
        model, score, epoch = train_graph_candidate(
            module, train_records, val_records, device, genes, hidden, lr,
            epochs, min(config.early_stop_patience or 60, max(2, epochs)),
        )
        initial_candidates.append((score, hidden, lr, epoch, model))
        history.append({
            "stage": "backbone_validation_search", "hidden_channels": hidden,
            "learning_rate": lr, "best_epoch": epoch,
            "validation_mean_gene_pcc": score,
            "elapsed_seconds": time.time() - started,
        })
        print(json.dumps(history[-1], sort_keys=True), flush=True)
    _, hidden, lr, _, initial_model = max(initial_candidates, key=lambda row: row[0])
    cluster_grid = [(p, c) for p in (1.0, 3.0, 5.0) for c in (0.2, 0.5, 0.8)]
    cluster_candidates = []
    val_targets = {record["sample_id"]: record["y"] for record in val_records}
    for p_factor, c_sp in cluster_grid:
        adapted = []
        for record in val_records:
            item = dict(record)
            item["edge_index"] = graph_filter_affinity(
                item["x"], item["edge_index"], p_factor, c_sp
            )
            adapted.append(item)
        score = finite_mean_gene_pcc(predict_graphs(initial_model, adapted, device), val_targets)
        cluster_candidates.append((score, p_factor, c_sp))
        history.append({
            "stage": "adaptive_graph_validation_search", "p_factor": p_factor,
            "c_sp": c_sp, "validation_mean_gene_pcc": score,
            "elapsed_seconds": time.time() - started,
        })
        print(json.dumps(history[-1], sort_keys=True), flush=True)
    _, p_factor, c_sp = max(cluster_candidates, key=lambda row: row[0])
    for record in records.values():
        record["edge_index"] = graph_filter_affinity(
            record["x"], record["edge_index"], p_factor, c_sp
        )
    train_records = [records[sample] for sample in fold.train_ids]
    val_records = [records[sample] for sample in fold.val_ids]
    test_records = [records[sample] for sample in fold.test_ids]
    model, best_score, best_epoch = train_graph_candidate(
        module, train_records, val_records, device, genes, hidden, lr,
        epochs, min(config.early_stop_patience or 60, max(2, epochs)),
    )
    save_checkpoint(
        model, staging / "best_checkpoint.pt", epoch=best_epoch,
        validation_mean_gene_pcc=best_score,
    )
    history.append({
        "stage": "final_adaptive_graph_fit", "hidden_channels": hidden,
        "learning_rate": lr, "p_factor": p_factor, "c_sp": c_sp,
        "best_epoch": best_epoch, "validation_mean_gene_pcc": best_score,
        "elapsed_seconds": time.time() - started,
    })
    return predict_graphs(model, test_records, device), history, best_epoch, best_score, {
        "node_feature": "ImageNet-v2 ResNet50 2048D common cache",
        "train_only_feature_reduction": "whitened PCA32",
        "initial_graph": "symmetric spatial 8-nearest-neighbor graph",
        "adaptive_graph": "released AffinityPropagation similarity and edge filtering equations",
        "selected_hidden_channels": hidden, "selected_learning_rate": lr,
        "selected_p_factor": p_factor, "selected_c_sp": c_sp,
        "search_budget": {
            "backbone_candidates": len(search), "adaptive_graph_candidates": len(cluster_grid)
        },
    }


RUNNERS = {
    "gr2st": run_gr2st,
    "heclip": run_heclip,
    "he2gene": run_he2gene,
    "asgnn": run_asgnn,
}


def write_outputs(context, predictions, history, best_epoch, best_score, adaptation, args, started):
    config, slides, genes, fold, target_parameters, device, staging = context
    if set(predictions) != set(fold.test_ids):
        raise ValueError("outer-test prediction coverage differs from fold contract")
    with (staging / "training_history.csv").open("w", newline="", encoding="utf-8") as handle:
        fieldnames = sorted({key for row in history for key in row})
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(history)
    prediction_root = staging / "predictions"
    prediction_root.mkdir()
    rows = []
    for sample_id in fold.test_ids:
        prediction = np.asarray(predictions[sample_id], dtype=np.float32)
        expected = (len(slides[sample_id].barcodes), len(genes))
        if prediction.shape != expected or not np.all(np.isfinite(prediction)):
            raise ValueError(f"invalid prediction matrix for {sample_id}: {prediction.shape}")
        path = prediction_root / f"{sample_id}.npz"
        np.savez_compressed(
            path, prediction=prediction, genes=np.asarray(genes),
            barcodes=np.asarray(slides[sample_id].barcodes), sample_id=np.asarray(sample_id),
            patient_id=np.asarray(slides[sample_id].patient_id),
            fold_id=np.asarray(fold.fold_id, dtype=np.int32),
        )
        rows.append({
            "sample_id": sample_id, "patient_id": slides[sample_id].patient_id,
            "path": str(path.relative_to(staging)), "sha256": sha256_file(path),
            "n_spots": expected[0], "n_genes": expected[1],
        })
    external_contract_path = os.environ.get("LIN_MNP_EXTERNAL_CONTRACT")
    selection_scope = os.environ.get("LIN_MNP_CHECKPOINT_SELECTION_SCOPE")
    fixed_final_epoch = (
        os.environ.get("LIN_MNP_CHECKPOINT_MODE") == "fixed_final_epoch"
    )
    selection_metadata = (
        {
            "fixed_refit_epoch": best_epoch,
            "checkpoint_selection_scope": (
                "fixed_epoch_selected_on_inner_validation_then_refit_on_all_outer_train"
            ),
        }
        if fixed_final_epoch
        else
        {
            "best_epoch_selected_by_outer_test": best_epoch,
            "checkpoint_selection_scope": "outer_test_section_to_match_HESPotEx",
        }
        if selection_scope == "outer_test"
        else {
            "best_epoch_selected_by_validation_only": best_epoch,
            "checkpoint_selection_scope": selection_scope or "inner_validation_patient",
        }
    )
    metadata = {
        "schema_version": 1, "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "method": asdict(config), "source_revision": source_revision(config.source_dir),
        "fold": asdict(fold), "seed": args.seed + fold.fold_id,
        "device": str(device), "mixed_precision": bool(args.amp),
        "executed_epochs": args.max_epochs or config.epochs,
        "best_validation_mean_gene_pcc": best_score,
        **selection_metadata,
        "target_parameters": {
            key: value.tolist() if isinstance(value, np.ndarray) else value
            for key, value in target_parameters.items()
        },
        "patch_manifest_sha256": sha256_file(PATCH_ROOT / "manifest.json"),
        "predictions_contain_truth": False, "predictions": rows,
        "coverage_contract": "all_outer_test_slides",
        "runtime_compatibility_patches": [adaptation],
        "wall_time_seconds": time.time() - started,
    }
    if external_contract_path:
        contract_path = Path(external_contract_path).resolve()
        contract = json.loads(contract_path.read_text(encoding="utf-8"))
        metadata.update({
            "external_contract_path": str(contract_path),
            "external_contract_sha256": sha256_file(contract_path),
            "external_protocol": contract.get("protocol"),
            "external_cohort": contract.get("cohort"),
        })
    (staging / "run_metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    checkpoint = staging / "best_checkpoint.pt"
    complete = {
        "run_metadata_sha256": sha256_file(staging / "run_metadata.json"),
        "training_history_sha256": sha256_file(staging / "training_history.csv"),
        "checkpoint_sha256": sha256_file(checkpoint),
        "prediction_sha256": {row["sample_id"]: row["sha256"] for row in rows},
    }
    (staging / "complete.json").write_text(
        json.dumps(complete, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def train_fold(args):
    config = METHODS[args.method]
    folds = load_folds()
    if args.fold not in folds:
        raise ValueError(args.fold)
    fold = folds[args.fold]
    fold_output = (args.output / args.method / f"fold{args.fold}").resolve()
    if fold_output.exists():
        raise FileExistsError(fold_output)
    staging = fold_output.with_name(f"{fold_output.name}.tmp-{os.getpid()}")
    staging.mkdir(parents=True, exist_ok=False)
    started = time.time()
    seed_all(args.seed + fold.fold_id)
    torch.backends.cudnn.benchmark = True
    device = torch.device(args.device)
    slides, genes = load_slides()
    target_parameters = fit_target_transform(slides, fold.train_ids, config.target_transform)
    context = (config, slides, genes, fold, target_parameters, device, staging)
    try:
        result = RUNNERS[args.method](context, args)
        write_outputs(context, *result, args, started)
        staging.rename(fold_output)
    except BaseException:
        raise
    return fold_output


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=sorted(METHODS), required=True)
    parser.add_argument("--fold", type=int, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--max-epochs", type=int)
    parser.add_argument(
        "--locked-adaptation",
        type=Path,
        help="JSON choice selected only on the nested inner-validation section.",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args(argv)


def main(argv=None):
    output = train_fold(parse_args(argv))
    print(json.dumps({"output": str(output)}, sort_keys=True))


if __name__ == "__main__":
    main()
