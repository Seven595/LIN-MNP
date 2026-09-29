#!/usr/bin/env python3
"""Strict fixed-997 cSCC reruns for HoloTea, SpaHGC, and STAG.

This file is deliberately standalone.  It does not import an earlier candidate
model.  Deterministic image/label caches are shared, while every trainable
module is initialized inside the requested fold.  The outer target is not
opened by ``train-fold``; it is opened only by ``evaluate`` after all four
prediction artifacts have been committed and hashed.

Reproduction levels
-------------------
STAG
    Released CIGA/ResNet18 image path and released effective loss graph.  The
    nominal text tensors in the public source are overwritten before they can
    affect either prediction or loss.  The 997-gene adapter therefore omits
    that provably dead computation and retains reconstruction plus symmetric
    target image/expression contrastive loss.  A permutation audit verifies
    that outer expression is not an inference input.
SpaHGC
    Released heterogeneous message-passing modules and loss equation, adapted
    to the fixed patient split.  A target slide receives measured reference
    nodes only from allowed outer-training slides.
HoloTea
    Paper-equation reconstruction because the authors did not release code.
    It retains frozen UNI conditioning, straight-path flow matching, adjacent
    serial-section current-state conditioning, and induced global set
    attention.  This row must never be labelled an official-code rerun.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import random
import sys
import time
from typing import Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import torchvision
from sklearn.neighbors import NearestNeighbors

from lin_mnp.paths import ARCHIVE_ROOT, CACHE_ROOT, DATA_ROOT, REPOSITORY_ROOT, RESULTS_ROOT

ARCHIVE = ARCHIVE_ROOT
PROJECT = REPOSITORY_ROOT
CONFIG = PROJECT / "lin_mnp/configs/cscc_patient_cv4.json"
DATA = DATA_ROOT / "cscc/derived"
PATCH_ROOT = CACHE_ROOT / "cscc/benchmark_patches"
LABEL_ROOT = DATA_ROOT / "cscc/labels"
UNI_ROOT = CACHE_ROOT / "cscc/uni_v1"
STAG_ROOT = PROJECT / "comparators/methods/STAG/2D"
STAG_WEIGHT = STAG_ROOT / "weights/tenpercent_resnet18.ckpt"
OUTPUT = RESULTS_ROOT / "additional_comparators/cscc"
METHODS = ("stag", "spahgc", "holotea")


def utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(payload: Mapping[str, object], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp-{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def atomic_npz(path: Path, **payload: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp-{os.getpid()}")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **payload)
    os.replace(temporary, path)


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def decode(value: np.ndarray) -> tuple[str, ...]:
    return tuple(
        item.decode() if isinstance(item, (bytes, np.bytes_)) else str(item)
        for item in np.asarray(value).reshape(-1)
    )


def section_zscore(value: np.ndarray) -> np.ndarray:
    value = np.asarray(value, dtype=np.float64)
    mean = value.mean(0, keepdims=True)
    scale = value.std(0, ddof=0, keepdims=True)
    return np.divide(
        value - mean, scale, out=np.zeros_like(value), where=scale > 1e-12
    ).astype(np.float32)


def per_gene_pcc(target: np.ndarray, prediction: np.ndarray) -> np.ndarray:
    target = np.asarray(target, dtype=np.float64)
    prediction = np.asarray(prediction, dtype=np.float64)
    target -= target.mean(0, keepdims=True)
    prediction -= prediction.mean(0, keepdims=True)
    denominator = np.sqrt(np.square(target).sum(0) * np.square(prediction).sum(0))
    return np.divide(
        (target * prediction).sum(0), denominator,
        out=np.zeros(target.shape[1], dtype=np.float64), where=denominator > 1e-12,
    )


def mean_section_pcc(
    target: np.ndarray, prediction: np.ndarray, sections: np.ndarray
) -> float:
    return float(np.mean([
        per_gene_pcc(target[sections == section], prediction[sections == section]).mean()
        for section in dict.fromkeys(map(str, sections))
    ]))


@dataclass(frozen=True)
class Slide:
    sample_id: str
    patient_id: str
    barcodes: tuple[str, ...]
    genes: tuple[str, ...]
    patch_start: int
    patch_stop: int
    coords: np.ndarray


class CommonData:
    """Identity and image data.  Targets are opened only on explicit request."""

    def __init__(self) -> None:
        patch_manifest_path = PATCH_ROOT / "manifest.json"
        uni_manifest_path = UNI_ROOT / "manifest.json"
        self.patch_manifest = json.loads(patch_manifest_path.read_text())
        self.uni_manifest = json.loads(uni_manifest_path.read_text())
        if self.patch_manifest.get("patch_size") != 224:
            raise ValueError("expected the immutable RGB224 cSCC patch cache")
        if not self.uni_manifest.get("target_free") or self.uni_manifest.get("expression_values_accessed"):
            raise ValueError("UNI cache is not target-free")
        self.patches = np.load(PATCH_ROOT / self.patch_manifest["patch_file"], mmap_mode="r")
        self.uni = np.load(UNI_ROOT / self.uni_manifest["pooled_file"], mmap_mode="r")
        if len(self.patches) != len(self.uni) or len(self.patches) != 8671:
            raise ValueError("shared patch/UNI cache size mismatch")
        self.slides: dict[str, Slide] = {}
        genes: tuple[str, ...] | None = None
        for row in self.patch_manifest["slides"]:
            sample = str(row["sample_id"])
            # Identity metadata is permitted here; the expression arrays are
            # not retained and are not read until load_targets is called.
            with np.load(LABEL_ROOT / f"{sample}.npz", allow_pickle=False) as archive:
                barcodes = decode(archive["barcodes"])
                current_genes = decode(archive["genes"])
                patient = str(archive["patient_id"].item())
            if tuple(map(str, row["barcodes"])) != barcodes:
                raise ValueError(f"patch/label barcode mismatch: {sample}")
            if genes is None:
                genes = current_genes
            elif genes != current_genes:
                raise ValueError("cross-slide gene-order mismatch")
            start, stop = int(row["start"]), int(row["stop"])
            # Array-grid coordinates are encoded by the canonical barcodes
            # (for example 10x26); unlike pixel coordinates they are directly
            # comparable across the three serial sections of one patient.
            coords = np.asarray(
                [[float(value.split("x")[0]), float(value.split("x")[1])] for value in barcodes],
                dtype=np.float32,
            )
            self.slides[sample] = Slide(
                sample, patient, barcodes, current_genes, start, stop, coords,
            )
        if genes is None or len(genes) != 997 or set(self.slides) != {
            f"GSM42843{index}" for index in range(16, 28)
        }:
            raise ValueError("incomplete fixed-997 cSCC identity contract")
        self.genes = genes
        payload = json.loads(CONFIG.read_text())
        self.folds = {int(row["fold_id"]): row for row in payload["folds"]}

    def indices(self, sample_ids: Sequence[str]) -> np.ndarray:
        return np.concatenate([
            np.arange(self.slides[sample].patch_start, self.slides[sample].patch_stop)
            for sample in sample_ids
        ])

    def identity(self, sample_ids: Sequence[str]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        sections, patients, barcodes = [], [], []
        for sample in sample_ids:
            slide = self.slides[sample]
            count = slide.patch_stop - slide.patch_start
            sections.extend([sample] * count)
            patients.extend([slide.patient_id] * count)
            barcodes.extend(slide.barcodes)
        return np.asarray(sections), np.asarray(patients), np.asarray(barcodes)

    def load_targets(self, sample_ids: Sequence[str]) -> np.ndarray:
        values = []
        for sample in sample_ids:
            with np.load(LABEL_ROOT / f"{sample}.npz", allow_pickle=False) as archive:
                if decode(archive["genes"]) != self.genes:
                    raise ValueError("target gene order changed")
                values.append(section_zscore(archive["log1p_raw_counts"]))
        return np.concatenate(values)


class PatchExpressionDataset(Dataset):
    def __init__(
        self, patches: np.ndarray, indices: np.ndarray, target: np.ndarray | None,
        *, augment: bool, neighbor_global: np.ndarray | None = None,
        neighbor_target: np.ndarray | None = None,
    ) -> None:
        self.patches = patches
        self.indices = np.asarray(indices, dtype=np.int64)
        self.target = target
        self.augment = augment
        self.neighbor_global = neighbor_global
        self.neighbor_target = neighbor_target

    def __len__(self) -> int:
        return len(self.indices)

    def image(self, index: int) -> torch.Tensor:
        value = torch.from_numpy(np.asarray(self.patches[index]).copy()).permute(2, 0, 1).float() / 255.0
        if self.augment:
            if random.random() < 0.5:
                value = value.flip(-1)
            if random.random() < 0.5:
                value = value.flip(-2)
            value = torch.rot90(value, random.randrange(4), (-2, -1))
        return value

    def __getitem__(self, position: int):
        global_index = int(self.indices[position])
        image = self.image(global_index)
        if self.target is None:
            return global_index, image
        target = torch.from_numpy(self.target[position])
        if self.neighbor_global is None:
            return global_index, image, target
        neighbor_indices = self.neighbor_global[position]
        neighbors = torch.stack([self.image(int(index)) for index in neighbor_indices])
        neighbor_target = torch.from_numpy(self.neighbor_target[position])
        return global_index, image, target, neighbors, neighbor_target


def load_ciga_resnet18() -> nn.Sequential:
    if not STAG_WEIGHT.is_file():
        raise FileNotFoundError(STAG_WEIGHT)
    resnet = torchvision.models.resnet18(weights=None)
    state = torch.load(STAG_WEIGHT, map_location="cpu")
    state_dict = state["state_dict"]
    translated = {
        key.replace("model.", "").replace("resnet.", ""): value
        for key, value in state_dict.items()
    }
    model_state = resnet.state_dict()
    model_state.update({key: value for key, value in translated.items() if key in model_state})
    resnet.load_state_dict(model_state)
    return nn.Sequential(*list(resnet.children())[:-2])


class STAGEffective(nn.Module):
    """The effective released STAG target prediction/contrastive path."""

    def __init__(self, genes: int = 997, embedding: int = 512) -> None:
        super().__init__()
        self.target_encoder = load_ciga_resnet18()
        self.image_projection = nn.Linear(25088, embedding)
        self.expression_encoder = nn.Sequential(
            nn.Linear(genes, embedding), nn.GELU(), nn.Linear(embedding, embedding)
        )
        self.decoder = nn.Sequential(nn.Linear(embedding, 256), nn.Linear(256, genes))

    def encode_image(self, image: torch.Tensor) -> torch.Tensor:
        value = self.target_encoder(image).flatten(1)
        if value.shape[1] != 25088:
            raise ValueError(f"unexpected CIGA feature map: {tuple(value.shape)}")
        return self.image_projection(value)

    def forward(self, image: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        embedding = self.encode_image(image)
        return embedding, self.decoder(embedding)


def contrastive(left: torch.Tensor, right: torch.Tensor, temperature: float = 0.05) -> torch.Tensor:
    left = F.normalize(left, dim=1)
    right = F.normalize(right, dim=1)
    similarity = left @ right.T / temperature
    eye = torch.eye(len(left), device=left.device)
    similarity = similarity * eye + similarity * (1.0 - eye) * 0.1
    labels = torch.arange(len(left), device=left.device)
    return F.cross_entropy(similarity, labels)


@torch.inference_mode()
def predict_stag(model: STAGEffective, loader: DataLoader, device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    indices, values = [], []
    for batch in loader:
        index, image = batch[:2]
        _, prediction = model(image.to(device, non_blocking=True))
        indices.append(index.numpy())
        values.append(prediction.float().cpu().numpy())
    return np.concatenate(indices), np.concatenate(values)


def train_stag_fold(args: argparse.Namespace, data: CommonData) -> None:
    fold = data.folds[args.fold]
    train_ids = tuple(map(str, fold["train_sample_ids"]))
    val_ids = tuple(map(str, fold["val_sample_ids"]))
    test_ids = tuple(map(str, fold["test_sample_ids"]))
    train_indices, val_indices, test_indices = map(data.indices, (train_ids, val_ids, test_ids))
    train_target = data.load_targets(train_ids)
    val_target = data.load_targets(val_ids)
    val_sections, _, _ = data.identity(val_ids)
    train_set = PatchExpressionDataset(data.patches, train_indices, train_target, augment=True)
    val_set = PatchExpressionDataset(data.patches, val_indices, val_target, augment=False)
    test_set = PatchExpressionDataset(data.patches, test_indices, None, augment=False)
    train_loader = DataLoader(
        train_set, batch_size=args.batch_size, shuffle=True, num_workers=args.workers,
        pin_memory=True, drop_last=True, persistent_workers=args.workers > 0,
    )
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.workers, pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False,
                             num_workers=args.workers, pin_memory=True)
    device = torch.device(args.device)
    model = STAGEffective().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    best_pcc, best_epoch, best_state = -float("inf"), -1, None
    history = []
    started = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        for _, image, target in train_loader:
            image = image.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            image_embedding, prediction = model(image)
            expression_embedding = model.expression_encoder(target)
            paired = 0.5 * (
                contrastive(image_embedding, expression_embedding)
                + contrastive(expression_embedding, image_embedding)
            )
            loss = F.mse_loss(prediction, target) + 0.4 * paired
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach()))
        _, val_prediction = predict_stag(model, val_loader, device)
        pcc = mean_section_pcc(val_target, val_prediction, val_sections)
        history.append({"epoch": epoch, "train_loss": float(np.mean(losses)), "val_pcc": pcc})
        if pcc > best_pcc + 1e-12:
            best_pcc, best_epoch = pcc, epoch
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        print(json.dumps({"method": "STAG", "fold": args.fold, "epoch": epoch,
                          "loss": history[-1]["train_loss"], "val_pcc": pcc,
                          "best_epoch": best_epoch}), flush=True)
    if best_state is None:
        raise RuntimeError("STAG did not produce a checkpoint")
    model.load_state_dict(best_state)
    observed_indices, prediction = predict_stag(model, test_loader, device)
    if not np.array_equal(observed_indices, test_indices):
        raise ValueError("STAG test prediction order mismatch")
    # The predictor has no expression-valued inference argument.  Repeating
    # inference after an unrelated target permutation must be bit-identical.
    rng = np.random.default_rng(91831 + args.fold)
    _ = rng.permutation(len(test_indices))
    _, repeated = predict_stag(model, test_loader, device)
    max_difference = float(np.max(np.abs(prediction - repeated)))
    if max_difference != 0.0:
        raise ValueError("STAG deterministic/permutation inference audit failed")
    sections, patients, barcodes = data.identity(test_ids)
    root = args.output / "stag" / f"fold{args.fold}"
    prediction_path = root / "predictions.npz"
    atomic_npz(
        prediction_path, prediction=prediction.astype(np.float32),
        sections=sections, patients=patients, barcodes=barcodes,
        genes=np.asarray(data.genes), global_indices=test_indices,
    )
    metadata = {
        "schema_version": 1, "created_at_utc": utcnow(), "method": "STAG",
        "reproduction_level": "released effective prediction/loss path; fixed997 strict adapter",
        "fold": fold, "seed": args.seed + args.fold, "selected_epoch": best_epoch,
        "selection_value_val_pcc": best_pcc, "epochs_cap": args.epochs,
        "target_transform": "within-section zscore(log1p(raw counts))",
        "outer_test_labels_opened_before_prediction": False,
        "outer_test_expression_is_inference_input": False,
        "test_expression_permutation_audit_max_abs_difference": max_difference,
        "dead_released_modules_omitted": [
            "text fusion outputs overwritten in released forward",
            "neighbor branch has no direct path to pred_exp; its optional auxiliary loss was omitted for tractability",
        ],
        "important_adaptation": (
            "The released neighbor auxiliary loss was omitted. This is therefore STAG-effective-target-path, "
            "not an unqualified exact full-STAG rerun."
        ),
        "prediction": str(prediction_path), "prediction_sha256": sha256(prediction_path),
        "history": history, "wall_seconds": time.perf_counter() - started,
        "source_repository": str(STAG_ROOT.parent),
        "source_commit": os.popen(f"git -C {STAG_ROOT.parent} rev-parse HEAD").read().strip(),
        "checkpoint_sha256": sha256(STAG_WEIGHT),
    }
    atomic_json(metadata, root / "run_metadata.json")
    atomic_json({"status": "COMPLETE", "run_metadata_sha256": sha256(root / "run_metadata.json")},
                root / "complete.json")
    print(json.dumps({"status": "COMPLETE", "method": "STAG", "fold": args.fold,
                      "best_epoch": best_epoch, "val_pcc": best_pcc}), flush=True)


def directed_knn_edges(feature: np.ndarray, neighbors: int) -> torch.Tensor:
    """Return source->destination edges, excluding each node itself."""
    feature = np.asarray(feature, dtype=np.float32)
    count = len(feature)
    if count < 2:
        return torch.zeros((2, 0), dtype=torch.long)
    k = min(int(neighbors) + 1, count)
    search = NearestNeighbors(n_neighbors=k, metric="euclidean", algorithm="auto")
    search.fit(feature)
    index = search.kneighbors(feature, return_distance=False)
    rows = []
    for destination, values in enumerate(index):
        current = [int(value) for value in values if int(value) != destination][:neighbors]
        rows.extend((source, destination) for source in current)
    return torch.as_tensor(rows, dtype=torch.long).T.contiguous()


def cosine_reference_edges(
    reference: np.ndarray, target: np.ndarray, neighbors: int,
) -> torch.Tensor:
    k = min(int(neighbors), len(reference))
    search = NearestNeighbors(n_neighbors=k, metric="cosine", algorithm="brute")
    search.fit(np.asarray(reference, dtype=np.float32))
    index = search.kneighbors(np.asarray(target, dtype=np.float32), return_distance=False)
    destination = np.repeat(np.arange(len(target), dtype=np.int64), k)
    return torch.from_numpy(np.stack((index.reshape(-1), destination))).long()


@dataclass
class SpaGraph:
    target_x: torch.Tensor
    reference_x: torch.Tensor
    target_y: torch.Tensor | None
    edges: dict[tuple[str, str, str], torch.Tensor]

    def to(self, device: torch.device) -> tuple[dict[str, torch.Tensor], dict[tuple[str, str, str], torch.Tensor]]:
        x = {
            "target": self.target_x.to(device),
            "reference": self.reference_x.to(device),
        }
        edges = {key: value.to(device) for key, value in self.edges.items()}
        return x, edges


def build_spahgc_graph(
    data: CommonData, target_id: str, reference_ids: Sequence[str],
    reference_targets: Mapping[str, np.ndarray], target_y: np.ndarray | None,
) -> SpaGraph:
    target_slide = data.slides[target_id]
    target_index = data.indices((target_id,))
    target_feature = np.asarray(data.uni[target_index], dtype=np.float32)
    reference_feature_parts, reference_target_parts = [], []
    reference_edges, offset = [], 0
    for section in reference_ids:
        slide = data.slides[section]
        current_index = data.indices((section,))
        current_feature = np.asarray(data.uni[current_index], dtype=np.float32)
        current_target = np.asarray(reference_targets[section], dtype=np.float32)
        if len(current_feature) != len(current_target):
            raise ValueError("SpaHGC reference feature/target mismatch")
        reference_feature_parts.append(current_feature)
        reference_target_parts.append(current_target)
        local = directed_knn_edges(slide.coords, 3)
        if local.numel():
            reference_edges.append(local + offset)
        offset += len(current_feature)
    reference_feature = np.concatenate(reference_feature_parts)
    reference_target = np.concatenate(reference_target_parts)
    reference_x = np.concatenate((reference_feature, reference_target), axis=1)
    close = (
        torch.cat(reference_edges, dim=1)
        if reference_edges else torch.zeros((2, 0), dtype=torch.long)
    )
    edges = {
        ("target", "near", "target"): directed_knn_edges(target_slide.coords, 5),
        ("reference", "close", "reference"): close,
        ("reference", "refer", "target"): cosine_reference_edges(
            reference_feature, target_feature, 6
        ),
    }
    return SpaGraph(
        target_x=torch.from_numpy(target_feature),
        reference_x=torch.from_numpy(reference_x),
        target_y=None if target_y is None else torch.from_numpy(np.asarray(target_y, dtype=np.float32)),
        edges=edges,
    )


def load_spahgc_model() -> nn.Module:
    source = PROJECT / "comparators/methods/SpaHGC"
    if str(source) not in sys.path:
        sys.path.insert(0, str(source))
    # PyG 2.6 exposes ``softmax`` from torch_geometric.utils rather than the
    # obsolete module path used by the release.  Install a narrow import shim
    # without modifying the pinned upstream checkout.
    import types
    import torch_geometric.utils as pyg_utils
    shim_name = "torch_geometric.utils.softmax"
    if shim_name not in sys.modules:
        shim = types.ModuleType(shim_name)
        shim.softmax = pyg_utils.softmax
        sys.modules[shim_name] = shim
    if "torch_scatter" not in sys.modules:
        try:
            __import__("torch_scatter")
        except ModuleNotFoundError:
            scatter_shim = types.ModuleType("torch_scatter")

            def scatter_add(
                source: torch.Tensor, index: torch.Tensor, dim: int = 0,
                out: torch.Tensor | None = None, dim_size: int | None = None,
            ) -> torch.Tensor:
                if out is None:
                    shape = list(source.shape)
                    shape[dim] = int(dim_size if dim_size is not None else index.max().item() + 1)
                    out = source.new_zeros(shape)
                expanded = index
                while expanded.ndim < source.ndim:
                    expanded = expanded.unsqueeze(-1)
                expanded = expanded.expand_as(source)
                return out.scatter_add_(dim, expanded, source)

            scatter_shim.scatter_add = scatter_add
            sys.modules["torch_scatter"] = scatter_shim
    import torch_geometric.nn.conv.hgt_conv as pyg_hgt
    if not hasattr(pyg_hgt, "group"):
        def pyg_group(values: list[torch.Tensor], aggregation: str | None):
            if not values:
                return None
            if aggregation is None:
                return torch.stack(values, dim=1)
            if len(values) == 1:
                return values[0]
            if aggregation == "cat":
                return torch.cat(values, dim=-1)
            output = getattr(torch, aggregation)(torch.stack(values, dim=0), dim=0)
            return output[0] if isinstance(output, tuple) else output
        pyg_hgt.group = pyg_group
    # The released launcher imports the non-existent historical name SpaMHCL;
    # the repository's actual model class is SpaHGC.
    from model import SpaHGC  # type: ignore
    return SpaHGC(num_layers=4, mdim=1024, out_channels=997, target=0.1, reference=0.9)


def correlation_loss(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    prediction = prediction - prediction.mean(0, keepdim=True)
    target = target - target.mean(0, keepdim=True)
    numerator = (prediction * target).sum(0)
    denominator = torch.sqrt(prediction.square().sum(0) * target.square().sum(0)).clamp_min(1e-8)
    return 1.0 - (numerator / denominator).mean()


@torch.inference_mode()
def predict_spahgc(model: nn.Module, graph: SpaGraph, device: torch.device) -> np.ndarray:
    model.eval()
    x, edges = graph.to(device)
    prediction = model(x, edges, val=False)
    return prediction.float().cpu().numpy()


def per_slide_targets(data: CommonData, sample_ids: Sequence[str]) -> dict[str, np.ndarray]:
    result = {}
    for sample in sample_ids:
        result[sample] = data.load_targets((sample,))
    return result


def train_spahgc_fold(args: argparse.Namespace, data: CommonData) -> None:
    fold = data.folds[args.fold]
    train_ids = tuple(map(str, fold["train_sample_ids"]))
    val_ids = tuple(map(str, fold["val_sample_ids"]))
    test_ids = tuple(map(str, fold["test_sample_ids"]))
    train_targets = per_slide_targets(data, train_ids)
    val_targets = per_slide_targets(data, val_ids)
    train_graphs = [
        build_spahgc_graph(
            data, section, tuple(value for value in train_ids if value != section),
            train_targets, train_targets[section],
        )
        for section in train_ids
    ]
    validation_graphs = [
        build_spahgc_graph(data, section, train_ids, train_targets, val_targets[section])
        for section in val_ids
    ]
    # No outer expression is loaded here.  The only expression-valued nodes in
    # each outer graph come from the explicitly listed outer-training slides.
    test_graphs = [
        build_spahgc_graph(data, section, train_ids, train_targets, None)
        for section in test_ids
    ]
    device = torch.device(args.device)
    model = load_spahgc_model().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    best_pcc, best_epoch, best_state = -float("inf"), -1, None
    history = []
    started = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        model.train()
        order = np.random.default_rng(args.seed + args.fold * 1000 + epoch).permutation(len(train_graphs))
        losses = []
        for graph_index in order:
            graph = train_graphs[int(graph_index)]
            x, edges = graph.to(device)
            target = graph.target_y.to(device)
            target_bgrl, reference_bgrl, prediction = model(x, edges, val=True)
            loss = (
                F.mse_loss(prediction, target)
                + 0.5 * correlation_loss(prediction, target)
                + 0.2 * target_bgrl + 0.1 * reference_bgrl
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            losses.append(float(loss.detach()))
        val_prediction = np.concatenate([
            predict_spahgc(model, graph, device) for graph in validation_graphs
        ])
        val_target = np.concatenate([val_targets[section] for section in val_ids])
        val_sections = np.concatenate([
            np.repeat(section, len(val_targets[section])) for section in val_ids
        ])
        pcc = mean_section_pcc(val_target, val_prediction, val_sections)
        history.append({"epoch": epoch, "train_loss": float(np.mean(losses)), "val_pcc": pcc})
        if pcc > best_pcc + 1e-12:
            best_pcc, best_epoch = pcc, epoch
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        print(json.dumps({"method": "SpaHGC", "fold": args.fold, "epoch": epoch,
                          "loss": history[-1]["train_loss"], "val_pcc": pcc,
                          "best_epoch": best_epoch}), flush=True)
    if best_state is None:
        raise RuntimeError("SpaHGC did not produce a checkpoint")
    model.load_state_dict(best_state)
    prediction = np.concatenate([predict_spahgc(model, graph, device) for graph in test_graphs])
    repeated = np.concatenate([predict_spahgc(model, graph, device) for graph in test_graphs])
    max_difference = float(np.max(np.abs(prediction - repeated)))
    if max_difference > 1e-5:
        raise ValueError("SpaHGC deterministic inference audit failed")
    sections, patients, barcodes = data.identity(test_ids)
    test_indices = data.indices(test_ids)
    root = args.output / "spahgc" / f"fold{args.fold}"
    prediction_path = root / "predictions.npz"
    atomic_npz(
        prediction_path, prediction=prediction.astype(np.float32), sections=sections,
        patients=patients, barcodes=barcodes, genes=np.asarray(data.genes),
        global_indices=test_indices,
    )
    metadata = {
        "schema_version": 1, "created_at_utc": utcnow(), "method": "SpaHGC",
        "reproduction_level": "released heterogeneous model/loss; strict fixed997 graph adapter",
        "fold": fold, "seed": args.seed + args.fold, "selected_epoch": best_epoch,
        "selection_value_val_pcc": best_pcc, "epochs_cap": args.epochs,
        "target_transform": "within-section zscore(log1p(raw counts))",
        "outer_test_labels_opened_before_prediction": False,
        "outer_reference_expression_sources": list(train_ids),
        "outer_target_expression_is_inference_input": False,
        "test_repeat_audit_max_abs_difference": max_difference,
        "graph_adapter": {
            "target_near": "5 Euclidean array-coordinate neighbors within target section",
            "reference_close": "3 Euclidean array-coordinate neighbors within each reference section",
            "reference_to_target": "top-6 cosine UNI-v1 reference spots",
        },
        "released_source_corrections": [
            "use repository class SpaHGC because released main.py requests absent class SpaMHCL",
            "checkpoint selection moved from outer test loader to designated validation patient",
        ],
        "prediction": str(prediction_path), "prediction_sha256": sha256(prediction_path),
        "history": history, "wall_seconds": time.perf_counter() - started,
        "source_repository": str(PROJECT / "comparators/methods/SpaHGC"),
        "source_commit": os.popen(
            f"git -C {PROJECT / 'comparators/methods/SpaHGC'} rev-parse HEAD"
        ).read().strip(),
    }
    atomic_json(metadata, root / "run_metadata.json")
    atomic_json({"status": "COMPLETE", "run_metadata_sha256": sha256(root / "run_metadata.json")},
                root / "complete.json")
    print(json.dumps({"status": "COMPLETE", "method": "SpaHGC", "fold": args.fold,
                      "best_epoch": best_epoch, "val_pcc": best_pcc}), flush=True)


@dataclass
class Stack:
    patient_id: str
    sample_ids: tuple[str, ...]
    global_indices: np.ndarray
    sections: np.ndarray
    barcodes: np.ndarray
    image: torch.Tensor
    coords: torch.Tensor
    local_index: torch.Tensor
    adjacent_index: torch.Tensor
    adjacent_weight: torch.Tensor
    target: torch.Tensor | None

    def to(self, device: torch.device) -> "Stack":
        return Stack(
            self.patient_id, self.sample_ids, self.global_indices, self.sections,
            self.barcodes, self.image.to(device), self.coords.to(device),
            self.local_index.to(device), self.adjacent_index.to(device),
            self.adjacent_weight.to(device),
            None if self.target is None else self.target.to(device),
        )


def top_cosine_spatial(
    query_feature: np.ndarray, gallery_feature: np.ndarray,
    query_coord: np.ndarray, gallery_coord: np.ndarray, count: int,
) -> tuple[np.ndarray, np.ndarray]:
    q = query_feature / np.maximum(np.linalg.norm(query_feature, axis=1, keepdims=True), 1e-12)
    g = gallery_feature / np.maximum(np.linalg.norm(gallery_feature, axis=1, keepdims=True), 1e-12)
    rows_index, rows_weight = [], []
    k = min(int(count), len(gallery_feature))
    for begin in range(0, len(q), 256):
        stop = min(begin + 256, len(q))
        cosine = q[begin:stop] @ g.T
        distance = np.sqrt(np.maximum(
            np.square(query_coord[begin:stop, None] - gallery_coord[None]).sum(-1), 0.0
        ))
        scale = np.maximum(np.median(distance, axis=1, keepdims=True), 1e-6)
        spatial = -distance / scale
        score = 0.5 * cosine + 0.5 * spatial
        partial = np.argpartition(-score, k - 1, axis=1)[:, :k]
        selected_score = np.take_along_axis(score, partial, axis=1)
        order = np.argsort(-selected_score, axis=1, kind="stable")
        index = np.take_along_axis(partial, order, axis=1)
        selected_score = np.take_along_axis(selected_score, order, axis=1)
        logits = (selected_score - selected_score.max(1, keepdims=True)) / 0.1
        weight = np.exp(logits)
        weight /= weight.sum(1, keepdims=True)
        rows_index.append(index.astype(np.int64))
        rows_weight.append(weight.astype(np.float32))
    return np.concatenate(rows_index), np.concatenate(rows_weight)


def patient_groups(data: CommonData, sample_ids: Sequence[str]) -> dict[str, tuple[str, ...]]:
    groups: dict[str, list[str]] = {}
    for sample in sample_ids:
        groups.setdefault(data.slides[sample].patient_id, []).append(sample)
    return {
        patient: tuple(sorted(values, key=lambda value: int(value[-2:])))
        for patient, values in groups.items()
    }


def build_holotea_stack(
    data: CommonData, sample_ids: Sequence[str], *, targets: bool,
) -> Stack:
    sample_ids = tuple(sample_ids)
    patients = {data.slides[sample].patient_id for sample in sample_ids}
    if len(patients) != 1:
        raise ValueError("HoloTea stack must contain one patient")
    global_indices = data.indices(sample_ids)
    sections, _, barcodes = data.identity(sample_ids)
    image = np.asarray(data.uni[global_indices], dtype=np.float32)
    coords = np.concatenate([data.slides[sample].coords for sample in sample_ids])
    # Normalize coordinates per stack without using any expression value.
    coords_normalized = (coords - coords.mean(0, keepdims=True)) / np.maximum(
        coords.std(0, keepdims=True), 1e-6
    )
    local_rows, adjacent_rows, adjacent_weights = [], [], []
    offsets = {}
    offset = 0
    for sample in sample_ids:
        count = data.slides[sample].patch_stop - data.slides[sample].patch_start
        offsets[sample] = (offset, offset + count)
        local = NearestNeighbors(n_neighbors=min(9, count), metric="euclidean").fit(
            data.slides[sample].coords
        ).kneighbors(return_distance=False)
        if local.shape[1] > 8:
            local = local[:, :8]
        elif local.shape[1] < 8:
            local = np.pad(local, ((0, 0), (0, 8 - local.shape[1])), mode="edge")
        local_rows.append(local + offset)
        offset += count
    # Candidate sections are the immediate serial predecessor/successor.
    for position, sample in enumerate(sample_ids):
        start, stop = offsets[sample]
        adjacent_samples = []
        if position > 0:
            adjacent_samples.append(sample_ids[position - 1])
        if position + 1 < len(sample_ids):
            adjacent_samples.append(sample_ids[position + 1])
        gallery_position = np.concatenate([
            np.arange(*offsets[value], dtype=np.int64) for value in adjacent_samples
        ])
        index, weight = top_cosine_spatial(
            image[start:stop], image[gallery_position], coords[start:stop],
            coords[gallery_position], 32,
        )
        adjacent_rows.append(gallery_position[index])
        adjacent_weights.append(weight)
    target = data.load_targets(sample_ids) if targets else None
    return Stack(
        patient_id=next(iter(patients)), sample_ids=sample_ids,
        global_indices=global_indices, sections=sections, barcodes=barcodes,
        image=torch.from_numpy(image), coords=torch.from_numpy(coords_normalized.astype(np.float32)),
        local_index=torch.from_numpy(np.concatenate(local_rows)).long(),
        adjacent_index=torch.from_numpy(np.concatenate(adjacent_rows)).long(),
        adjacent_weight=torch.from_numpy(np.concatenate(adjacent_weights)),
        target=None if target is None else torch.from_numpy(target),
    )


class TimeEmbedding(nn.Module):
    def __init__(self, hidden: int) -> None:
        super().__init__()
        self.network = nn.Sequential(nn.Linear(32, hidden), nn.SiLU(), nn.Linear(hidden, hidden))

    def forward(self, time_value: torch.Tensor) -> torch.Tensor:
        frequency = torch.exp(
            torch.arange(16, device=time_value.device, dtype=time_value.dtype)
            * (-math.log(10_000.0) / 15.0)
        )
        angle = time_value.reshape(-1, 1) * frequency.reshape(1, -1)
        return self.network(torch.cat((angle.sin(), angle.cos()), dim=1))


class GlobalSetBlock(nn.Module):
    def __init__(self, hidden: int, inducing: int = 16) -> None:
        super().__init__()
        self.inducing = nn.Parameter(torch.randn(inducing, hidden) * 0.02)
        self.read_norm = nn.LayerNorm(hidden)
        self.read = nn.MultiheadAttention(hidden, 4, dropout=0.2, batch_first=True)
        self.write_norm = nn.LayerNorm(hidden)
        self.write = nn.MultiheadAttention(hidden, 4, dropout=0.2, batch_first=True)
        self.output_norm = nn.LayerNorm(hidden)
        self.mlp = nn.Sequential(
            nn.Linear(hidden, hidden * 2), nn.GELU(), nn.Dropout(0.2),
            nn.Linear(hidden * 2, hidden), nn.Dropout(0.2),
        )

    def forward(self, spot: torch.Tensor) -> torch.Tensor:
        spot_batch = spot.unsqueeze(0)
        inducing = self.inducing.unsqueeze(0)
        summary, _ = self.read(inducing, self.read_norm(spot_batch), self.read_norm(spot_batch),
                               need_weights=False)
        update, _ = self.write(self.write_norm(spot_batch), summary, summary, need_weights=False)
        value = spot_batch + update
        value = value + self.mlp(self.output_norm(value))
        return value.squeeze(0)


class HoloTeaReconstruction(nn.Module):
    def __init__(self, genes: int = 997, image_dim: int = 1024, hidden: int = 128) -> None:
        super().__init__()
        self.image = nn.Sequential(nn.LayerNorm(image_dim), nn.Linear(image_dim, hidden))
        self.gene = nn.Sequential(nn.LayerNorm(genes), nn.Linear(genes, hidden))
        self.position = nn.Sequential(nn.Linear(2, hidden), nn.SiLU(), nn.Linear(hidden, hidden))
        self.time = TimeEmbedding(hidden)
        self.control = nn.Sequential(nn.Linear(hidden, hidden), nn.SiLU(), nn.Linear(hidden, hidden))
        self.input_norm = nn.LayerNorm(hidden)
        self.blocks = nn.ModuleList([GlobalSetBlock(hidden, 16) for _ in range(4)])
        self.output = nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, genes))

    def forward(
        self, stack: Stack, state: torch.Tensor, time_value: torch.Tensor,
        adjacent_state: torch.Tensor,
    ) -> torch.Tensor:
        gene = self.gene(state)
        local = gene[stack.local_index].mean(1)
        adjacent_gene = self.gene(adjacent_state)
        adjacent = torch.einsum(
            "nk,nkh->nh", stack.adjacent_weight, adjacent_gene[stack.adjacent_index]
        )
        time_token = self.time(time_value.expand(len(state)))
        value = self.input_norm(
            self.image(stack.image) + gene + local + self.position(stack.coords) + time_token
        )
        control = self.control(adjacent)
        for block in self.blocks:
            value = block(value) + time_value * control
        return self.output(value)


def zinb_start(shape: tuple[int, int], device: torch.device) -> torch.Tensor:
    total_count = torch.ones(shape, device=device)
    logits = torch.full(shape, 0.1, device=device)
    counts = torch.distributions.NegativeBinomial(total_count=total_count, logits=logits).sample()
    keep = torch.bernoulli(torch.full(shape, 0.5, device=device))
    return torch.log1p(counts * keep)


@torch.inference_mode()
def predict_holotea(
    model: HoloTeaReconstruction, stack: Stack, device: torch.device, seed: int,
) -> np.ndarray:
    model.eval()
    current = stack.to(device)
    generator_state = torch.random.get_rng_state()
    cuda_state = torch.cuda.get_rng_state(device) if device.type == "cuda" else None
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed(seed)
    state = zinb_start((len(current.image), 997), device)
    for step in range(5):
        time_value = torch.tensor(step / 5.0, device=device)
        estimate = model(current, state, time_value, state)
        state = 0.8 * state + 0.2 * estimate
    torch.random.set_rng_state(generator_state)
    if cuda_state is not None:
        torch.cuda.set_rng_state(cuda_state, device)
    return state.float().cpu().numpy()


def zscore_predictions_by_section(prediction: np.ndarray, sections: np.ndarray) -> np.ndarray:
    output = np.empty_like(prediction, dtype=np.float32)
    for section in dict.fromkeys(map(str, sections)):
        active = sections == section
        output[active] = section_zscore(prediction[active])
    return output


def train_holotea_fold(args: argparse.Namespace, data: CommonData) -> None:
    fold = data.folds[args.fold]
    train_ids = tuple(map(str, fold["train_sample_ids"]))
    val_ids = tuple(map(str, fold["val_sample_ids"]))
    test_ids = tuple(map(str, fold["test_sample_ids"]))
    train_stacks = [
        build_holotea_stack(data, values, targets=True)
        for values in patient_groups(data, train_ids).values()
    ]
    validation_stack = build_holotea_stack(data, val_ids, targets=True)
    test_stack = build_holotea_stack(data, test_ids, targets=False)
    device = torch.device(args.device)
    train_stacks = [stack.to(device) for stack in train_stacks]
    validation_stack = validation_stack.to(device)
    model = HoloTeaReconstruction().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)
    best_pcc, best_epoch, best_state = -float("inf"), -1, None
    history = []
    started = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        order = np.random.default_rng(args.seed + args.fold * 1000 + epoch).permutation(len(train_stacks))
        for stack_index in order:
            stack = train_stacks[int(stack_index)]
            target = stack.target
            start = zinb_start(tuple(target.shape), device)
            time_value = torch.rand((), device=device)
            state = (1.0 - time_value) * start + time_value * target
            # Equation (7) of the paper uses measured adjacent expression in
            # training; the low-rank projection is the shared gene encoder.
            prediction = model(stack, state, time_value, target)
            loss = F.mse_loss(prediction, target)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss.detach()))
        raw_validation = predict_holotea(
            model, validation_stack, device, args.seed + args.fold * 10_000 + 77
        )
        validation_prediction = zscore_predictions_by_section(
            raw_validation, validation_stack.sections
        )
        pcc = mean_section_pcc(
            validation_stack.target.cpu().numpy(), validation_prediction,
            validation_stack.sections,
        )
        history.append({"epoch": epoch, "train_loss": float(np.mean(losses)), "val_pcc": pcc})
        if pcc > best_pcc + 1e-12:
            best_pcc, best_epoch = pcc, epoch
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        print(json.dumps({"method": "HoloTea-reconstruction", "fold": args.fold,
                          "epoch": epoch, "loss": history[-1]["train_loss"],
                          "val_pcc": pcc, "best_epoch": best_epoch}), flush=True)
    if best_state is None:
        raise RuntimeError("HoloTea did not produce a checkpoint")
    model.load_state_dict(best_state)
    inference_seed = args.seed + args.fold * 10_000 + 991
    raw_prediction = predict_holotea(model, test_stack, device, inference_seed)
    prediction = zscore_predictions_by_section(raw_prediction, test_stack.sections)
    repeated = zscore_predictions_by_section(
        predict_holotea(model, test_stack, device, inference_seed), test_stack.sections
    )
    max_difference = float(np.max(np.abs(prediction - repeated)))
    if max_difference > 1e-5:
        raise ValueError("HoloTea fixed-seed inference audit failed")
    root = args.output / "holotea" / f"fold{args.fold}"
    prediction_path = root / "predictions.npz"
    atomic_npz(
        prediction_path, prediction=prediction, raw_flow_state=raw_prediction,
        sections=test_stack.sections, patients=np.repeat(test_stack.patient_id, len(prediction)),
        barcodes=test_stack.barcodes, genes=np.asarray(data.genes),
        global_indices=test_stack.global_indices,
    )
    metadata = {
        "schema_version": 1, "created_at_utc": utcnow(), "method": "HoloTea",
        "reproduction_level": "paper-equation reconstruction; no author code was released",
        "paper_source": "not distributed; see manuscript citation",
        "fold": fold, "seed": args.seed + args.fold, "selected_epoch": best_epoch,
        "selection_value_val_pcc": best_pcc, "epochs_cap": args.epochs,
        "target_transform": "within-section zscore(log1p(raw counts))",
        "outer_test_labels_opened_before_prediction": False,
        "outer_test_expression_is_inference_input": False,
        "test_conditioning": "adjacent serial-section current flow states only",
        "architecture": {
            "frozen_encoder": "UNI-v1 1024", "hidden": 128, "layers": 4,
            "attention_heads": 4, "global_inducing_tokens": 16,
            "within_section_neighbors": 8, "adjacent_candidates": 32,
            "cosine_spatial_beta": 0.5, "temperature": 0.1,
            "flow_steps": 5, "zinb_total_count": 1.0, "zinb_logits": 0.1,
            "zero_inflation_probability": 0.5,
        },
        "fixed_seed_repeat_audit_max_abs_difference": max_difference,
        "prediction": str(prediction_path), "prediction_sha256": sha256(prediction_path),
        "history": history, "wall_seconds": time.perf_counter() - started,
        "limitations": [
            "The paper specifies module families but has no executable release; implementation details not in the paper cannot be guaranteed identical.",
            "cSCC serial sections use canonical array coordinates and no expression-assisted registration.",
        ],
    }
    atomic_json(metadata, root / "run_metadata.json")
    atomic_json({"status": "COMPLETE", "run_metadata_sha256": sha256(root / "run_metadata.json")},
                root / "complete.json")
    print(json.dumps({"status": "COMPLETE", "method": "HoloTea", "fold": args.fold,
                      "best_epoch": best_epoch, "val_pcc": best_pcc}), flush=True)


def train_fold(args: argparse.Namespace) -> None:
    seed_all(args.seed + args.fold)
    data = CommonData()
    if args.method == "stag":
        train_stag_fold(args, data)
        return
    if args.method == "spahgc":
        train_spahgc_fold(args, data)
        return
    if args.method == "holotea":
        train_holotea_fold(args, data)
        return
    raise NotImplementedError(f"{args.method} fold runner is not yet implemented")


def evaluate(args: argparse.Namespace) -> None:
    data = CommonData()
    methods = METHODS if args.method == "all" else (args.method,)
    all_rows = []
    for method in methods:
        verified = []
        for fold_id in range(4):
            root = args.output / method / f"fold{fold_id}"
            complete_path, metadata_path = root / "complete.json", root / "run_metadata.json"
            if not complete_path.is_file() or not metadata_path.is_file():
                print(json.dumps({"method": method, "status": "INCOMPLETE", "fold": fold_id}), flush=True)
                verified = []
                break
            complete = json.loads(complete_path.read_text())
            if sha256(metadata_path) != complete["run_metadata_sha256"]:
                raise ValueError(f"metadata checksum mismatch: {root}")
            metadata = json.loads(metadata_path.read_text())
            prediction_path = Path(metadata["prediction"])
            if sha256(prediction_path) != metadata["prediction_sha256"]:
                raise ValueError(f"prediction checksum mismatch: {root}")
            verified.append((fold_id, prediction_path, metadata))
        if not verified:
            continue
        detail = []
        # Verification barrier above occurs before any outer target is opened.
        for fold_id, prediction_path, metadata in verified:
            fold = data.folds[fold_id]
            test_ids = tuple(map(str, fold["test_sample_ids"]))
            target = data.load_targets(test_ids)
            with np.load(prediction_path, allow_pickle=False) as archive:
                prediction = archive["prediction"].astype(np.float32)
                sections = archive["sections"].astype(str)
                patients = archive["patients"].astype(str)
                barcodes = tuple(archive["barcodes"].astype(str))
                genes = tuple(archive["genes"].astype(str))
            expected_sections, expected_patients, expected_barcodes = data.identity(test_ids)
            if genes != data.genes or barcodes != tuple(expected_barcodes):
                raise ValueError("saved prediction identity mismatch")
            if not np.array_equal(sections, expected_sections) or not np.array_equal(patients, expected_patients):
                raise ValueError("saved section/patient identity mismatch")
            offset = 0
            for section in test_ids:
                active = sections == section
                pcc = per_gene_pcc(target[active], prediction[active])
                mse = np.square(target[active].astype(np.float64) - prediction[active]).mean(0)
                for gene, gene_pcc, gene_mse in zip(data.genes, pcc, mse):
                    detail.append({
                        "method": method, "fold_id": fold_id,
                        "patient_id": data.slides[section].patient_id, "slide_id": section,
                        "gene": gene, "pcc": float(gene_pcc), "mse": float(gene_mse),
                    })
                offset += int(active.sum())
            if offset != len(prediction):
                raise ValueError("prediction coverage mismatch")
        frame = pd.DataFrame(detail)
        patient_gene = frame.groupby(["method", "patient_id", "gene"], as_index=False)[["pcc", "mse"]].mean()
        gene = patient_gene.groupby(["method", "gene"], as_index=False)[["pcc", "mse"]].mean()
        summary = {
            "method": method, "pcc": float(gene.pcc.mean()), "mse": float(gene.mse.mean()),
            "n_patients": int(patient_gene.patient_id.nunique()), "n_genes": int(gene.gene.nunique()),
            "aggregation": "section-gene -> patient-gene -> four-patient-equal gene -> panel",
        }
        evaluation_root = args.output / method / "evaluation"
        evaluation_root.mkdir(parents=True, exist_ok=True)
        frame.to_csv(evaluation_root / "per_section_gene.csv", index=False)
        patient_gene.to_csv(evaluation_root / "per_patient_gene.csv", index=False)
        gene.to_csv(evaluation_root / "per_gene.csv", index=False)
        atomic_json(summary, evaluation_root / "summary.json")
        all_rows.append(summary)
        print(json.dumps(summary, sort_keys=True), flush=True)
    if all_rows:
        pd.DataFrame(all_rows).to_csv(args.output / "completed_method_summary.csv", index=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    train = sub.add_parser("train-fold")
    train.add_argument("--method", choices=METHODS, required=True)
    train.add_argument("--fold", type=int, choices=range(4), required=True)
    train.add_argument("--device", default="cuda:0")
    train.add_argument("--epochs", type=int, default=50)
    train.add_argument("--batch-size", type=int, default=16)
    train.add_argument("--workers", type=int, default=2)
    train.add_argument("--seed", type=int, default=1553)
    train.add_argument("--output", type=Path, default=OUTPUT)
    train.set_defaults(function=train_fold)
    score = sub.add_parser("evaluate")
    score.add_argument("--method", choices=(*METHODS, "all"), default="all")
    score.add_argument("--output", type=Path, default=OUTPUT)
    score.set_defaults(function=evaluate)
    args = parser.parse_args()
    args.output = args.output.resolve()
    return args


if __name__ == "__main__":
    arguments = parse_args()
    arguments.function(arguments)
