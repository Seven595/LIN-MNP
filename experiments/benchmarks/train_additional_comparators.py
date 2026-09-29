#!/usr/bin/env python3
"""Train one strict HEST/STImage fold for HoloTea, SpaHGC, or STAG.

All trainable modules below are defined in this file and initialized anew in
every invocation.  Only immutable RGB224 patches and frozen UNI-v1 embeddings
are shared.  The external contract supplies train/validation/test identities;
test expression is never opened by this process.
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
import subprocess
import sys
import time
import types
from typing import Mapping, Sequence

import numpy as np
from sklearn.neighbors import NearestNeighbors
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import torchvision

from lin_mnp.paths import ARCHIVE_ROOT, CACHE_ROOT, REPOSITORY_ROOT

PROJECT = REPOSITORY_ROOT
ARCHIVE = ARCHIVE_ROOT
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))
from lin_mnp.benchmark.benchmark_external import (  # noqa: E402
    PATCH_ROOT, apply_target_transform, fit_target_transform, load_folds, load_slides,
)

UNI_ROOT = Path(os.environ.get(
    "LIN_MNP_UNI_ROOT", str(CACHE_ROOT / "shared/uni_v1"),
))
STAG_WEIGHT = PROJECT / "comparators/methods/STAG/2D/weights/tenpercent_resnet18.ckpt"
METHODS = ("holotea", "spahgc", "stag")


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
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def atomic_npz(path: Path, **payload: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp-{os.getpid()}")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **payload)
    os.replace(temporary, path)


def seed_all(seed: int) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def per_gene_pcc(target: np.ndarray, prediction: np.ndarray) -> np.ndarray:
    left = target.astype(np.float64) - target.mean(0, keepdims=True)
    right = prediction.astype(np.float64) - prediction.mean(0, keepdims=True)
    denominator = np.sqrt(np.square(left).sum(0) * np.square(right).sum(0))
    return np.divide((left * right).sum(0), denominator,
                     out=np.full(target.shape[1], np.nan), where=denominator > 1e-12)


def mean_slide_pcc(targets: Mapping[str, np.ndarray], predictions: Mapping[str, np.ndarray]) -> float:
    return float(np.nanmean([
        np.nanmean(per_gene_pcc(targets[name], predictions[name])) for name in predictions
    ]))


def fold_payload(fold: object) -> dict[str, object]:
    return {
        "fold_id": int(fold.fold_id), "train_ids": list(fold.train_ids),
        "val_ids": list(fold.val_ids), "test_ids": list(fold.test_ids),
        "train_patients": list(fold.train_patients), "val_patient": fold.val_patient,
        "test_patient": fold.test_patient,
    }


class Data:
    def __init__(self, fold_id: int) -> None:
        self.slides, self.genes = load_slides()
        self.fold = load_folds()[fold_id]
        metadata = json.loads((UNI_ROOT / "metadata.json").read_text(encoding="utf-8"))
        if not metadata.get("target_free") or metadata.get("expression_values_accessed"):
            raise ValueError("UNI cache is not target-free")
        self.uni = np.load(UNI_ROOT / "embeddings_float32.npy", mmap_mode="r")
        self.patches = np.load(PATCH_ROOT / "patches_224_uint8.npy", mmap_mode="r")
        if len(self.uni) != len(self.patches):
            raise ValueError("patch/UNI size mismatch")
        self.parameters = fit_target_transform(self.slides, self.fold.train_ids, "log1p_raw")

    def indices(self, names: Sequence[str]) -> np.ndarray:
        return np.concatenate([
            np.arange(self.slides[name].patch_start, self.slides[name].patch_stop, dtype=np.int64)
            for name in names
        ])

    def target(self, name: str) -> np.ndarray:
        return apply_target_transform(self.slides[name], self.parameters).astype(np.float32)

    def targets(self, names: Sequence[str]) -> dict[str, np.ndarray]:
        return {name: self.target(name) for name in names}

    def coords(self, name: str) -> np.ndarray:
        value = np.asarray(self.slides[name].array_coordinates, dtype=np.float32)
        if value.ndim != 2 or value.shape[1] != 2 or not np.isfinite(value).all():
            value = np.asarray(self.slides[name].coordinates, dtype=np.float32)
        return value


class SpotDataset(Dataset):
    def __init__(self, data: Data, names: Sequence[str], targets: Mapping[str, np.ndarray] | None,
                 augment: bool) -> None:
        self.data, self.names, self.targets, self.augment = data, tuple(names), targets, augment
        rows = []
        for slide_index, name in enumerate(self.names):
            slide = data.slides[name]
            for local, global_index in enumerate(range(slide.patch_start, slide.patch_stop)):
                rows.append((slide_index, local, global_index))
        self.rows = np.asarray(rows, dtype=np.int64)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, position: int):
        slide_index, local, global_index = map(int, self.rows[position])
        image = torch.from_numpy(np.asarray(self.data.patches[global_index]).copy()).permute(2, 0, 1).float() / 255
        if self.augment:
            if random.random() < .5: image = image.flip(-1)
            if random.random() < .5: image = image.flip(-2)
            image = torch.rot90(image, random.randrange(4), (-2, -1))
        target = torch.empty(0) if self.targets is None else torch.from_numpy(self.targets[self.names[slide_index]][local])
        return image, target, slide_index, local


def load_ciga_resnet18() -> nn.Sequential:
    resnet = torchvision.models.resnet18(weights=None)
    state = torch.load(STAG_WEIGHT, map_location="cpu")["state_dict"]
    translated = {key.replace("model.", "").replace("resnet.", ""): value for key, value in state.items()}
    current = resnet.state_dict(); current.update({k: v for k, v in translated.items() if k in current})
    resnet.load_state_dict(current)
    return nn.Sequential(*list(resnet.children())[:-2])


class STAGEffective(nn.Module):
    def __init__(self, genes: int, embedding: int = 512) -> None:
        super().__init__()
        self.target_encoder = load_ciga_resnet18()
        self.image_projection = nn.Linear(25088, embedding)
        self.expression_encoder = nn.Sequential(nn.Linear(genes, embedding), nn.GELU(), nn.Linear(embedding, embedding))
        self.decoder = nn.Sequential(nn.Linear(embedding, 256), nn.Linear(256, genes))

    def forward(self, image: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        embedding = self.image_projection(self.target_encoder(image).flatten(1))
        return embedding, self.decoder(embedding)


def contrastive(left: torch.Tensor, right: torch.Tensor, temperature: float = .05) -> torch.Tensor:
    left, right = F.normalize(left, dim=1), F.normalize(right, dim=1)
    similarity = left @ right.T / temperature
    eye = torch.eye(len(left), device=left.device)
    similarity = similarity * eye + similarity * (1 - eye) * .1
    return F.cross_entropy(similarity, torch.arange(len(left), device=left.device))


@torch.inference_mode()
def predict_stag(model: STAGEffective, loader: DataLoader, names: Sequence[str], sizes: Mapping[str, int],
                 genes: int, device: torch.device) -> dict[str, np.ndarray]:
    model.eval(); result = {name: np.empty((sizes[name], genes), np.float32) for name in names}
    for image, _, slide_index, local in loader:
        prediction = model(image.to(device, non_blocking=True))[1].float().cpu().numpy()
        for row, (s, i) in enumerate(zip(slide_index.numpy(), local.numpy())):
            result[names[int(s)]][int(i)] = prediction[row]
    return result


def train_stag(args: argparse.Namespace, data: Data, device: torch.device):
    train_y, val_y = data.targets(data.fold.train_ids), data.targets(data.fold.val_ids)
    sizes = {name: len(data.slides[name].barcodes) for name in (*data.fold.val_ids, *data.fold.test_ids)}
    loaders = {
        "train": DataLoader(SpotDataset(data, data.fold.train_ids, train_y, True), batch_size=args.batch_size,
                            shuffle=True, num_workers=args.workers, pin_memory=True, drop_last=True),
        "val": DataLoader(SpotDataset(data, data.fold.val_ids, val_y, False), batch_size=args.batch_size,
                          shuffle=False, num_workers=args.workers, pin_memory=True),
        "test": DataLoader(SpotDataset(data, data.fold.test_ids, None, False), batch_size=args.batch_size,
                           shuffle=False, num_workers=args.workers, pin_memory=True),
    }
    model = STAGEffective(len(data.genes)).to(device); optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    best = (-float("inf"), -1, None); history = []
    for epoch in range(1, args.epochs + 1):
        model.train(); losses = []
        for image, target, _, _ in loaders["train"]:
            image, target = image.to(device), target.to(device)
            image_embedding, prediction = model(image); expression_embedding = model.expression_encoder(target)
            loss = F.mse_loss(prediction, target) + .4 * .5 * (
                contrastive(image_embedding, expression_embedding) + contrastive(expression_embedding, image_embedding))
            optimizer.zero_grad(set_to_none=True); loss.backward(); optimizer.step(); losses.append(float(loss.detach()))
        val_prediction = predict_stag(model, loaders["val"], data.fold.val_ids, sizes, len(data.genes), device)
        score = mean_slide_pcc(val_y, val_prediction); history.append({"epoch": epoch, "loss": float(np.mean(losses)), "val_pcc": score})
        if score > best[0] + 1e-12:
            best = (score, epoch, {k: v.detach().cpu().clone() for k, v in model.state_dict().items()})
        print(json.dumps({"method": "STAG", "epoch": epoch, "val_pcc": score, "best_epoch": best[1]}), flush=True)
        if epoch - best[1] >= 10:
            break
    model.load_state_dict(best[2]); prediction = predict_stag(model, loaders["test"], data.fold.test_ids, sizes, len(data.genes), device)
    return prediction, best[1], best[0], history, {
        "encoder": "released CIGA-pretrained ResNet18", "effective_target_path": True,
        "early_stopping_patience": 10,
    }


def directed_knn(feature: np.ndarray, neighbors: int) -> torch.Tensor:
    count = len(feature)
    if count < 2: return torch.zeros((2, 0), dtype=torch.long)
    index = NearestNeighbors(n_neighbors=min(neighbors + 1, count)).fit(feature).kneighbors(return_distance=False)
    rows = []
    for destination, values in enumerate(index):
        current = [int(x) for x in values if int(x) != destination][:neighbors]
        rows.extend((source, destination) for source in current)
    return torch.tensor(rows, dtype=torch.long).T.contiguous()


def cosine_edges(reference: np.ndarray, target: np.ndarray, neighbors: int) -> torch.Tensor:
    k = min(neighbors, len(reference))
    # Exact cosine top-k, evaluated in bounded query blocks on the fold's GPU.
    # This is algebraically the same ranking as sklearn's brute-force cosine
    # search but avoids materialising a multi-gigabyte all-pairs CPU matrix for
    # the largest HEST kidney sections.
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    gallery = F.normalize(torch.from_numpy(np.asarray(reference, np.float32)).to(device), dim=1)
    rows = []
    with torch.inference_mode():
        for begin in range(0, len(target), 512):
            query = F.normalize(torch.from_numpy(np.asarray(target[begin:begin + 512], np.float32)).to(device), dim=1)
            rows.append(torch.topk(query @ gallery.T, k=k, dim=1, largest=True, sorted=True).indices.cpu().numpy())
    index = np.concatenate(rows)
    return torch.from_numpy(np.stack((index.reshape(-1), np.repeat(np.arange(len(target)), k)))).long()


@dataclass
class SpaGraph:
    target_x: torch.Tensor
    reference_x: torch.Tensor
    target_y: torch.Tensor | None
    edges: dict[tuple[str, str, str], torch.Tensor]

    def to(self, device: torch.device):
        return ({"target": self.target_x.to(device), "reference": self.reference_x.to(device)},
                {k: v.to(device) for k, v in self.edges.items()})


def build_spa_graph(data: Data, target_name: str, reference_names: Sequence[str],
                    reference_targets: Mapping[str, np.ndarray], target_y: np.ndarray | None) -> SpaGraph:
    if not reference_names:
        reference_names = (target_name,)
    target_index = data.indices((target_name,)); target_feature = np.asarray(data.uni[target_index], np.float32)
    feature_parts, target_parts, local_edges, offset = [], [], [], 0
    for name in reference_names:
        feature = np.asarray(data.uni[data.indices((name,))], np.float32)
        feature_parts.append(feature); target_parts.append(reference_targets[name])
        edge = directed_knn(data.coords(name), 3)
        if edge.numel(): local_edges.append(edge + offset)
        offset += len(feature)
    reference_feature, reference_y = np.concatenate(feature_parts), np.concatenate(target_parts)
    edges = {
        ("target", "near", "target"): directed_knn(data.coords(target_name), 5),
        ("reference", "close", "reference"): torch.cat(local_edges, 1) if local_edges else torch.zeros((2, 0), dtype=torch.long),
        ("reference", "refer", "target"): cosine_edges(reference_feature, target_feature, 6),
    }
    return SpaGraph(torch.from_numpy(target_feature), torch.from_numpy(np.concatenate((reference_feature, reference_y), 1)),
                    None if target_y is None else torch.from_numpy(target_y), edges)


def build_spa_group_graph(data: Data, target_names: Sequence[str], reference_names: Sequence[str],
                          reference_targets: Mapping[str, np.ndarray],
                          target_targets: Mapping[str, np.ndarray] | None,
                          reference_cap: int = 8192) -> tuple[SpaGraph, dict[str, tuple[int, int]]]:
    """Build one scalable released-core graph without discarding target spots.

    Target sections are concatenated with within-section edges only. Reference
    spots use a deterministic, section-balanced reservoir; the released model
    consumes only the top-six reference links per target, so this bounds memory
    while preserving equal access to every allowed training section.
    """
    target_parts, target_edges, slices, offset = [], [], {}, 0
    for name in target_names:
        feature = np.asarray(data.uni[data.indices((name,))], np.float32)
        target_parts.append(feature); edge = directed_knn(data.coords(name), 5)
        if edge.numel(): target_edges.append(edge + offset)
        slices[name] = (offset, offset + len(feature)); offset += len(feature)
    target_feature = np.concatenate(target_parts)
    allowance = max(1, reference_cap // max(1, len(reference_names)))
    reference_features, reference_values, reference_edges, offset = [], [], [], 0
    for name in reference_names:
        feature = np.asarray(data.uni[data.indices((name,))], np.float32); count = min(len(feature), allowance)
        selected = np.linspace(0, len(feature) - 1, count, dtype=np.int64)
        reference_features.append(feature[selected]); reference_values.append(reference_targets[name][selected])
        edge = directed_knn(data.coords(name)[selected], 3)
        if edge.numel(): reference_edges.append(edge + offset)
        offset += count
    reference_feature, reference_y = np.concatenate(reference_features), np.concatenate(reference_values)
    target_y = None if target_targets is None else np.concatenate([target_targets[name] for name in target_names])
    graph = SpaGraph(
        torch.from_numpy(target_feature), torch.from_numpy(np.concatenate((reference_feature, reference_y), 1)),
        None if target_y is None else torch.from_numpy(target_y),
        {
            ("target", "near", "target"): torch.cat(target_edges, 1) if target_edges else torch.zeros((2, 0), dtype=torch.long),
            ("reference", "close", "reference"): torch.cat(reference_edges, 1) if reference_edges else torch.zeros((2, 0), dtype=torch.long),
            ("reference", "refer", "target"): cosine_edges(reference_feature, target_feature, 6),
        },
    )
    return graph, slices


def load_spahgc(genes: int) -> nn.Module:
    source = PROJECT / "comparators/methods/SpaHGC"; sys.path.insert(0, str(source))
    import torch_geometric.utils as pyg_utils
    if "torch_geometric.utils.softmax" not in sys.modules:
        shim = types.ModuleType("torch_geometric.utils.softmax"); shim.softmax = pyg_utils.softmax
        sys.modules["torch_geometric.utils.softmax"] = shim
    if "torch_scatter" not in sys.modules:
        try: __import__("torch_scatter")
        except ModuleNotFoundError:
            shim = types.ModuleType("torch_scatter")
            def scatter_add(src, index, dim=0, out=None, dim_size=None):
                if out is None:
                    shape = list(src.shape); shape[dim] = int(dim_size if dim_size is not None else index.max().item() + 1); out = src.new_zeros(shape)
                expanded = index
                while expanded.ndim < src.ndim: expanded = expanded.unsqueeze(-1)
                return out.scatter_add_(dim, expanded.expand_as(src), src)
            shim.scatter_add = scatter_add; sys.modules["torch_scatter"] = shim
    import torch_geometric.nn.conv.hgt_conv as pyg_hgt
    if not hasattr(pyg_hgt, "group"):
        def group(values, aggregation):
            if not values: return None
            if aggregation is None: return torch.stack(values, 1)
            if len(values) == 1: return values[0]
            if aggregation == "cat": return torch.cat(values, -1)
            value = getattr(torch, aggregation)(torch.stack(values), dim=0)
            return value[0] if isinstance(value, tuple) else value
        pyg_hgt.group = group
    from model import SpaHGC  # type: ignore
    return SpaHGC(num_layers=4, mdim=1024, out_channels=genes, target=.1, reference=.9)


def correlation_loss(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    prediction, target = prediction - prediction.mean(0), target - target.mean(0)
    denominator = torch.sqrt(prediction.square().sum(0) * target.square().sum(0)).clamp_min(1e-8)
    return 1 - ((prediction * target).sum(0) / denominator).mean()


@torch.inference_mode()
def predict_spa(model: nn.Module, graph: SpaGraph, device: torch.device) -> np.ndarray:
    model.eval(); x, edges = graph.to(device); return model(x, edges, val=False).float().cpu().numpy()


def train_spahgc(args: argparse.Namespace, data: Data, device: torch.device):
    train_y, val_y = data.targets(data.fold.train_ids), data.targets(data.fold.val_ids)
    scalable = len(data.fold.train_ids) > 8
    if scalable:
        halves = (tuple(data.fold.train_ids[::2]), tuple(data.fold.train_ids[1::2]))
        train_graphs = []
        for index, names in enumerate(halves):
            if not names: continue
            references = halves[1 - index] or names
            train_graphs.append(build_spa_group_graph(data, names, references, train_y, train_y)[0])
        validation_graph, validation_slices = build_spa_group_graph(
            data, data.fold.val_ids, data.fold.train_ids, train_y, val_y)
        test_graph, test_slices = build_spa_group_graph(
            data, data.fold.test_ids, data.fold.train_ids, train_y, None)
        val_graphs, test_graphs = None, None
    else:
        train_graphs = [build_spa_graph(data, name, tuple(x for x in data.fold.train_ids if x != name), train_y, train_y[name]) for name in data.fold.train_ids]
        val_graphs = {name: build_spa_graph(data, name, data.fold.train_ids, train_y, val_y[name]) for name in data.fold.val_ids}
        test_graphs = {name: build_spa_graph(data, name, data.fold.train_ids, train_y, None) for name in data.fold.test_ids}
    model = load_spahgc(len(data.genes)).to(device); optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    best = (-float("inf"), -1, None); history = []
    for epoch in range(1, args.epochs + 1):
        model.train(); losses = []
        order = np.random.default_rng(args.seed + epoch).permutation(len(train_graphs))
        for index in order:
            graph = train_graphs[int(index)]; x, edges = graph.to(device); target = graph.target_y.to(device)
            target_bgrl, reference_bgrl, prediction = model(x, edges, val=True)
            loss = F.mse_loss(prediction, target) + .5 * correlation_loss(prediction, target) + .2 * target_bgrl + .1 * reference_bgrl
            optimizer.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 5); optimizer.step(); losses.append(float(loss.detach()))
        if scalable:
            flat = predict_spa(model, validation_graph, device)
            val_prediction = {name: flat[slice(*validation_slices[name])] for name in data.fold.val_ids}
        else:
            val_prediction = {name: predict_spa(model, graph, device) for name, graph in val_graphs.items()}
        score = mean_slide_pcc(val_y, val_prediction); history.append({"epoch": epoch, "loss": float(np.mean(losses)), "val_pcc": score})
        if score > best[0] + 1e-12: best = (score, epoch, {k: v.detach().cpu().clone() for k, v in model.state_dict().items()})
        print(json.dumps({"method": "SpaHGC", "epoch": epoch, "val_pcc": score, "best_epoch": best[1]}), flush=True)
    model.load_state_dict(best[2])
    if scalable:
        flat = predict_spa(model, test_graph, device)
        prediction = {name: flat[slice(*test_slices[name])] for name in data.fold.test_ids}
    else:
        prediction = {name: predict_spa(model, graph, device) for name, graph in test_graphs.items()}
    return prediction, best[1], best[0], history, {
        "released_core": True,
        "single_training_section_reference": "same-section training-only fallback",
        "scalable_group_graph": scalable,
        "scalable_rule": "when >8 train sections, two disjoint target/reference half-graphs; 8192 section-balanced reference spots; all target spots retained",
    }


@dataclass
class Stack:
    name: str
    image: torch.Tensor
    coords: torch.Tensor
    local_index: torch.Tensor
    adjacent_index: torch.Tensor
    adjacent_weight: torch.Tensor
    target: torch.Tensor | None

    def to(self, device: torch.device):
        return Stack(self.name, self.image.to(device), self.coords.to(device), self.local_index.to(device),
                     self.adjacent_index.to(device), self.adjacent_weight.to(device),
                     None if self.target is None else self.target.to(device))


def build_stack(data: Data, name: str, target: np.ndarray | None) -> Stack:
    feature = np.asarray(data.uni[data.indices((name,))], np.float32); coords = data.coords(name)
    normalized = (coords - coords.mean(0, keepdims=True)) / np.maximum(coords.std(0, keepdims=True), 1e-6)
    count = len(feature); k = min(9, count)
    local = NearestNeighbors(n_neighbors=k).fit(coords).kneighbors(return_distance=False)
    if local.shape[1] < 8: local = np.pad(local, ((0, 0), (0, 8 - local.shape[1])), mode="edge")
    else: local = local[:, :8]
    weight = np.full(local.shape, 1 / local.shape[1], np.float32)
    return Stack(name, torch.from_numpy(feature), torch.from_numpy(normalized.astype(np.float32)),
                 torch.from_numpy(local).long(), torch.from_numpy(local).long(), torch.from_numpy(weight),
                 None if target is None else torch.from_numpy(target))


class TimeEmbedding(nn.Module):
    def __init__(self, hidden: int):
        super().__init__(); self.network = nn.Sequential(nn.Linear(32, hidden), nn.SiLU(), nn.Linear(hidden, hidden))
    def forward(self, time: torch.Tensor):
        frequency = torch.exp(torch.arange(16, device=time.device, dtype=time.dtype) * (-math.log(10000) / 15))
        angle = time.reshape(-1, 1) * frequency.reshape(1, -1)
        return self.network(torch.cat((angle.sin(), angle.cos()), 1))


class GlobalSetBlock(nn.Module):
    def __init__(self, hidden: int):
        super().__init__(); self.inducing = nn.Parameter(torch.randn(16, hidden) * .02)
        self.read_norm = nn.LayerNorm(hidden); self.read = nn.MultiheadAttention(hidden, 4, .2, batch_first=True)
        self.write_norm = nn.LayerNorm(hidden); self.write = nn.MultiheadAttention(hidden, 4, .2, batch_first=True)
        self.output_norm = nn.LayerNorm(hidden); self.mlp = nn.Sequential(nn.Linear(hidden, hidden * 2), nn.GELU(), nn.Dropout(.2), nn.Linear(hidden * 2, hidden), nn.Dropout(.2))
    def forward(self, spot: torch.Tensor):
        spots, inducing = spot.unsqueeze(0), self.inducing.unsqueeze(0)
        summary = self.read(inducing, self.read_norm(spots), self.read_norm(spots), need_weights=False)[0]
        update = self.write(self.write_norm(spots), summary, summary, need_weights=False)[0]
        value = spots + update; return (value + self.mlp(self.output_norm(value))).squeeze(0)


class HoloTea(nn.Module):
    def __init__(self, genes: int, hidden: int = 128):
        super().__init__(); self.image = nn.Sequential(nn.LayerNorm(1024), nn.Linear(1024, hidden))
        self.gene = nn.Sequential(nn.LayerNorm(genes), nn.Linear(genes, hidden)); self.position = nn.Sequential(nn.Linear(2, hidden), nn.SiLU(), nn.Linear(hidden, hidden))
        self.time = TimeEmbedding(hidden); self.control = nn.Sequential(nn.Linear(hidden, hidden), nn.SiLU(), nn.Linear(hidden, hidden))
        self.input_norm = nn.LayerNorm(hidden); self.blocks = nn.ModuleList([GlobalSetBlock(hidden) for _ in range(4)])
        self.output = nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, genes))
    def forward(self, stack: Stack, state: torch.Tensor, time: torch.Tensor, adjacent_state: torch.Tensor):
        gene = self.gene(state); local = gene[stack.local_index].mean(1); adjacent_gene = self.gene(adjacent_state)
        adjacent = torch.einsum("nk,nkh->nh", stack.adjacent_weight, adjacent_gene[stack.adjacent_index])
        value = self.input_norm(self.image(stack.image) + gene + local + self.position(stack.coords) + self.time(time.expand(len(state))))
        control = self.control(adjacent)
        for block in self.blocks: value = block(value) + time * control
        return self.output(value)


def zinb_start(shape: tuple[int, int], device: torch.device):
    counts = torch.distributions.NegativeBinomial(total_count=torch.ones(shape, device=device), logits=torch.full(shape, .1, device=device)).sample()
    return torch.log1p(counts * torch.bernoulli(torch.full(shape, .5, device=device)))


@torch.inference_mode()
def predict_holotea(model: HoloTea, stack: Stack, device: torch.device, seed: int, genes: int):
    model.eval(); stack = stack.to(device); cpu_state = torch.random.get_rng_state(); cuda_state = torch.cuda.get_rng_state(device)
    torch.manual_seed(seed); torch.cuda.manual_seed(seed); state = zinb_start((len(stack.image), genes), device)
    for step in range(5):
        time = torch.tensor(step / 5, device=device); state = .8 * state + .2 * model(stack, state, time, state)
    torch.random.set_rng_state(cpu_state); torch.cuda.set_rng_state(cuda_state, device)
    return state.float().cpu().numpy()


def train_holotea(args: argparse.Namespace, data: Data, device: torch.device):
    train_y, val_y = data.targets(data.fold.train_ids), data.targets(data.fold.val_ids)
    train = [build_stack(data, name, train_y[name]).to(device) for name in data.fold.train_ids]
    validation = {name: build_stack(data, name, val_y[name]) for name in data.fold.val_ids}
    test = {name: build_stack(data, name, None) for name in data.fold.test_ids}
    model = HoloTea(len(data.genes)).to(device); optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)
    best = (-float("inf"), -1, None); history = []
    for epoch in range(1, args.epochs + 1):
        model.train(); losses = []
        for index in np.random.default_rng(args.seed + epoch).permutation(len(train)):
            stack = train[int(index)]; target = stack.target; start = zinb_start(tuple(target.shape), device); time_value = torch.rand((), device=device)
            state = (1 - time_value) * start + time_value * target; prediction = model(stack, state, time_value, target)
            loss = F.mse_loss(prediction, target); optimizer.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1); optimizer.step(); losses.append(float(loss.detach()))
        val_prediction = {name: predict_holotea(model, stack, device, args.seed + epoch * 1009, len(data.genes)) for name, stack in validation.items()}
        score = mean_slide_pcc(val_y, val_prediction); history.append({"epoch": epoch, "loss": float(np.mean(losses)), "val_pcc": score})
        if score > best[0] + 1e-12: best = (score, epoch, {k: v.detach().cpu().clone() for k, v in model.state_dict().items()})
        print(json.dumps({"method": "HoloTea", "epoch": epoch, "val_pcc": score, "best_epoch": best[1]}), flush=True)
    model.load_state_dict(best[2]); prediction = {name: predict_holotea(model, stack, device, args.seed + 991, len(data.genes)) for name, stack in test.items()}
    return prediction, best[1], best[0], history, {"paper_equation_reconstruction": True, "serial_control_boundary": "disabled; no unrelated sections treated as serial neighbors"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=METHODS, required=True); parser.add_argument("--fold", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True); parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--epochs", type=int, default=30); parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--workers", type=int, default=2); parser.add_argument("--seed", type=int, default=1553)
    args = parser.parse_args(); seed_all(args.seed + args.fold); data = Data(args.fold); device = torch.device(args.device); started = time.perf_counter()
    function = {"stag": train_stag, "spahgc": train_spahgc, "holotea": train_holotea}[args.method]
    predictions, selected_epoch, val_pcc, history, adaptation = function(args, data, device)
    root = args.output / args.method / f"fold{args.fold}"; hashes = {}
    for name in data.fold.test_ids:
        path = root / "predictions" / f"{name}.npz"
        atomic_npz(path, prediction=predictions[name].astype(np.float32), genes=np.asarray(data.genes),
                   barcodes=np.asarray(data.slides[name].barcodes))
        hashes[name] = sha256(path)
    metadata = {
        "schema_version": 1, "created_at_utc": utcnow(), "method": args.method,
        "reproduction_level": {"holotea": "paper-equation reconstruction", "spahgc": "released heterogeneous core strict adapter", "stag": "released effective prediction path strict adapter"}[args.method],
        "fold": fold_payload(data.fold), "seed": args.seed + args.fold, "epochs_cap": args.epochs,
        "selected_epoch": selected_epoch, "selection_value_val_pcc": val_pcc,
        "target_transform": "common deposited 50-gene target", "checkpoint_selection": "inner validation only",
        "outer_test_labels_opened_before_prediction": False, "outer_test_expression_is_inference_input": False,
        "prediction_sha256": hashes, "history": history, "adaptation": adaptation,
        "wall_seconds": time.perf_counter() - started,
    }
    atomic_json(metadata, root / "run_metadata.json")
    atomic_json({"status": "COMPLETE", "run_metadata_sha256": sha256(root / "run_metadata.json")}, root / "complete.json")
    print(json.dumps({"status": "COMPLETE", "method": args.method, "fold": args.fold, "selected_epoch": selected_epoch, "val_pcc": val_pcc}), flush=True)


if __name__ == "__main__":
    main()
