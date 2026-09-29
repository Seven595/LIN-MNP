#!/usr/bin/env python3
"""Train one BLEEP, mclSTExp or IGI-DL patient-CV4 adapter fold.

The adapters preserve each released method's defining objective and inference
equation while consuming the immutable HEtoSGEBench fold/spot/gene contract.
Image backbones are frozen into a shared target-free feature cache.  IGI-DL's
unreleased per-spot nuclei graphs are replaced by the separately computed
target-free Cellpose aggregate descriptors and are explicitly labelled as a
runtime adaptation in every result artifact.
"""

from __future__ import annotations

import argparse
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
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from lin_mnp.benchmark.benchmark_external import (  # noqa: E402
    DERIVED_ROOT,
    PATCH_ROOT,
    apply_target_transform,
    fit_target_transform,
    load_folds,
    load_slides,
)
from lin_mnp.paths import RESULTS_ROOT


DEFAULT_OUTPUT = RESULTS_ROOT / "comparator_reruns"
FEATURE_ROOT = Path(
    os.environ.get(
        "LIN_MNP_EGN_EMBEDDING_ROOT",
        str(DERIVED_ROOT / "egn_exemplar_embeddings_v1"),
    )
).resolve()
NUCLEI_ROOT = Path(
    os.environ.get(
        "LIN_MNP_CELLPOSE_NUCLEI_ROOT",
        str(DERIVED_ROOT / "shared_image_descriptors/cellpose_nuclei_px192_d14_v1/slides"),
    )
).resolve()
METHODS = {
    "bleep": {
        "display_name": "BLEEP",
        "backbone": "resnet50",
        "target_transform": "log1p_cp10k",
        "learning_rate": 1e-3,
        "weight_decay": 1e-3,
        "retrieval_k": 50,
    },
    "mclstexp": {
        "display_name": "mclSTExp",
        "backbone": "resnet50",
        "target_transform": "log1p_cp10k",
        "learning_rate": 1e-4,
        "weight_decay": 1e-3,
        "retrieval_k": 600,
    },
    "igidl": {
        "display_name": "IGI-DL",
        "backbone": "resnet18",
        "target_transform": "igidl_log_pseudocount_cpm",
        "learning_rate": 1e-3,
        "weight_decay": 1e-4,
        "retrieval_k": None,
    },
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def source_revision(path: Path) -> str:
    if path.joinpath(".git").exists():
        return subprocess.check_output(
            ["git", "-C", str(path), "rev-parse", "HEAD"], text=True
        ).strip()
    return "official source snapshot; identified by source hashes"


def serialise_parameters(parameters: Mapping[str, object]) -> dict[str, object]:
    return {
        key: value.tolist() if isinstance(value, np.ndarray) else value
        for key, value in parameters.items()
    }


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def feature_cache(backbone: str, expected_rows: int) -> tuple[np.ndarray, Path, Path]:
    feature_path = FEATURE_ROOT / f"{backbone}_embeddings_float32.npy"
    metadata_path = FEATURE_ROOT / f"{backbone}_metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    features = np.load(feature_path, mmap_mode="r")
    if list(features.shape) != metadata["shape"] or features.shape[0] < expected_rows:
        raise ValueError(f"invalid {backbone} feature cache")
    return features, feature_path, metadata_path


def concatenate_rows(
    slides: Mapping[str, object], matrix: np.ndarray, sample_ids: Sequence[str]
) -> np.ndarray:
    return np.concatenate(
        [matrix[slides[name].patch_start : slides[name].patch_stop] for name in sample_ids]
    ).astype(np.float32, copy=False)


def target_rows(
    slides: Mapping[str, object],
    sample_ids: Sequence[str],
    parameters: Mapping[str, object],
) -> np.ndarray:
    return np.concatenate(
        [apply_target_transform(slides[name], parameters) for name in sample_ids]
    ).astype(np.float32, copy=False)


def position_rows(
    slides: Mapping[str, object], sample_ids: Sequence[str]
) -> np.ndarray:
    return np.concatenate([slides[name].array_coordinates for name in sample_ids]).astype(
        np.int64, copy=False
    )


def nuclei_rows(
    slides: Mapping[str, object], sample_ids: Sequence[str]
) -> tuple[np.ndarray, tuple[str, ...], list[str]]:
    rows = []
    names: tuple[str, ...] | None = None
    hashes = []
    for sample_id in sample_ids:
        path = NUCLEI_ROOT / f"{sample_id}.npz"
        with np.load(path, allow_pickle=False) as archive:
            if tuple(archive["barcodes"].astype(str)) != slides[sample_id].barcodes:
                raise ValueError(f"nuclei barcode order differs for {sample_id}")
            if bool(archive["target_data_accessed"].item()):
                raise ValueError(f"nuclei cache accessed targets for {sample_id}")
            current_names = tuple(archive["feature_names"].astype(str))
            if names is None:
                names = current_names
            elif current_names != names:
                raise ValueError("nuclei feature order differs between slides")
            rows.append(archive["features"].astype(np.float32))
        hashes.append(sha256_file(path))
    if names is None:
        raise ValueError("empty nuclei feature request")
    return np.concatenate(rows), names, hashes


class ProjectionHead(nn.Module):
    """Released BLEEP/mclSTExp residual projection head."""

    def __init__(self, input_dim: int, projection_dim: int = 256, dropout: float = 0.1):
        super().__init__()
        self.projection = nn.Linear(input_dim, projection_dim)
        self.fc = nn.Linear(projection_dim, projection_dim)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(projection_dim)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        projected = self.projection(value)
        value = self.fc(F.gelu(projected))
        return self.layer_norm(self.dropout(value) + projected)


class BLEEPAdapter(nn.Module):
    def __init__(self, image_dim: int, expression_dim: int):
        super().__init__()
        self.image_projection = ProjectionHead(image_dim, dropout=0.1)
        self.spot_projection = ProjectionHead(expression_dim, dropout=0.1)
        self.temperature = 1.0

    def encode_image(self, image: torch.Tensor) -> torch.Tensor:
        return self.image_projection(image)

    def encode_expression(
        self, expression: torch.Tensor, position: torch.Tensor | None = None
    ) -> torch.Tensor:
        return self.spot_projection(expression)

    def loss(
        self, image: torch.Tensor, expression: torch.Tensor, position: torch.Tensor
    ) -> torch.Tensor:
        image_embedding = self.encode_image(image)
        spot_embedding = self.encode_expression(expression)
        logits = spot_embedding @ image_embedding.T / self.temperature
        image_similarity = image_embedding @ image_embedding.T
        spot_similarity = spot_embedding @ spot_embedding.T
        target = F.softmax(
            ((image_similarity + spot_similarity) / 2.0) / self.temperature, dim=-1
        )
        spot_loss = -(target * F.log_softmax(logits, dim=-1)).sum(dim=1)
        image_loss = -(target.T * F.log_softmax(logits.T, dim=-1)).sum(dim=1)
        return ((spot_loss + image_loss) / 2.0).mean()


class PreNormAttention(nn.Module):
    def __init__(self, dim: int, heads: int = 8, dim_head: int = 64):
        super().__init__()
        inner = heads * dim_head
        self.norm = nn.LayerNorm(dim)
        self.heads = heads
        self.dim_head = dim_head
        self.scale = dim_head**-0.5
        self.to_qkv = nn.Linear(dim, inner * 3, bias=False)
        self.to_out = nn.Linear(inner, dim)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        value = self.norm(value)
        batch, tokens, _ = value.shape
        query, key, item = self.to_qkv(value).chunk(3, dim=-1)
        query = query.reshape(batch, tokens, self.heads, self.dim_head).transpose(1, 2)
        key = key.reshape(batch, tokens, self.heads, self.dim_head).transpose(1, 2)
        item = item.reshape(batch, tokens, self.heads, self.dim_head).transpose(1, 2)
        attention = torch.softmax(query @ key.transpose(-2, -1) * self.scale, dim=-1)
        value = (attention @ item).transpose(1, 2).reshape(batch, tokens, -1)
        return self.to_out(value)


class AttentionBlock(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.attention = PreNormAttention(dim)
        self.ff_norm = nn.LayerNorm(dim)
        self.ff = nn.Sequential(nn.Linear(dim, dim), nn.GELU(), nn.Linear(dim, dim))

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        value = value + self.attention(value)
        return value + self.ff(self.ff_norm(value))


class MCLAdapter(nn.Module):
    def __init__(
        self, image_dim: int, expression_dim: int, position_min: np.ndarray, position_max: np.ndarray
    ):
        super().__init__()
        self.register_buffer("position_min", torch.as_tensor(position_min, dtype=torch.long))
        sizes = position_max - position_min + 1
        self.x_embed = nn.Embedding(int(sizes[0]), expression_dim)
        self.y_embed = nn.Embedding(int(sizes[1]), expression_dim)
        self.spot_encoder = nn.Sequential(AttentionBlock(expression_dim), AttentionBlock(expression_dim))
        self.image_projection = ProjectionHead(image_dim, dropout=0.0)
        self.spot_projection = ProjectionHead(expression_dim, dropout=0.0)
        self.temperature = 1.0

    def encode_image(self, image: torch.Tensor) -> torch.Tensor:
        return self.image_projection(image)

    def encode_expression(
        self, expression: torch.Tensor, position: torch.Tensor | None = None
    ) -> torch.Tensor:
        if position is None:
            raise ValueError("mclSTExp expression encoding requires spot position")
        position = position.long() - self.position_min
        value = expression + self.x_embed(position[:, 0]) + self.y_embed(position[:, 1])
        value = self.spot_encoder(value.unsqueeze(0)).squeeze(0)
        return self.spot_projection(value)

    def loss(
        self, image: torch.Tensor, expression: torch.Tensor, position: torch.Tensor
    ) -> torch.Tensor:
        image_embedding = self.encode_image(image)
        spot_embedding = self.encode_expression(expression, position)
        logits = spot_embedding @ image_embedding.T / self.temperature
        labels = torch.arange(len(logits), device=logits.device)
        return (
            F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels)
        ) / 2.0


class IGIDLAdapter(nn.Module):
    """Released IGI-DL fusion shape with an aggregate-nuclei substitute."""

    def __init__(self, image_dim: int, nuclei_dim: int, output_dim: int):
        super().__init__()
        self.image_bn = nn.BatchNorm1d(image_dim)
        self.nuclei_branch = nn.Sequential(
            nn.Linear(nuclei_dim, 256), nn.ReLU(), nn.BatchNorm1d(256)
        )
        self.fusion = nn.Sequential(
            nn.Linear(image_dim + 256, 256), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(256, 256), nn.ReLU(), nn.Dropout(0.5),
        )
        self.head = nn.Linear(256, output_dim)

    def forward(self, image: torch.Tensor, nuclei: torch.Tensor) -> torch.Tensor:
        return self.head(self.fusion(torch.cat([self.image_bn(image), self.nuclei_branch(nuclei)], dim=1)))


def mean_slide_gene_pcc(
    prediction: np.ndarray,
    target: np.ndarray,
    slides: Mapping[str, object],
    sample_ids: Sequence[str],
) -> float:
    rows = []
    start = 0
    for sample_id in sample_ids:
        stop = start + len(slides[sample_id].barcodes)
        pred = prediction[start:stop].astype(np.float64)
        true = target[start:stop].astype(np.float64)
        pred -= pred.mean(axis=0, keepdims=True)
        true -= true.mean(axis=0, keepdims=True)
        denominator = np.sqrt(np.sum(pred * pred, axis=0) * np.sum(true * true, axis=0))
        rows.append(
            np.divide(
                np.sum(pred * true, axis=0), denominator,
                out=np.full_like(denominator, np.nan), where=denominator > 0,
            )
        )
        start = stop
    return float(np.nanmean(np.nanmean(np.stack(rows), axis=0)))


@torch.no_grad()
def encode_chunks(
    model: nn.Module,
    kind: str,
    values: np.ndarray,
    device: torch.device,
    positions: np.ndarray | None = None,
    chunk: int = 256,
) -> np.ndarray:
    model.eval()
    rows = []
    for start in range(0, len(values), chunk):
        tensor = torch.from_numpy(np.asarray(values[start : start + chunk])).to(device)
        if kind == "image":
            encoded = model.encode_image(tensor)
        else:
            pos = None
            if positions is not None:
                pos = torch.from_numpy(positions[start : start + chunk]).to(device)
            encoded = model.encode_expression(tensor, pos)
        rows.append(encoded.float().cpu().numpy())
    return np.concatenate(rows)


def retrieve(
    image_embedding: np.ndarray,
    spot_embedding: np.ndarray,
    train_expression: np.ndarray,
    method: str,
    top_k: int,
) -> np.ndarray:
    spot = spot_embedding / np.maximum(np.linalg.norm(spot_embedding, axis=1, keepdims=True), 1e-8)
    query = image_embedding / np.maximum(np.linalg.norm(image_embedding, axis=1, keepdims=True), 1e-8)
    predictions = []
    for start in range(0, len(query), 512):
        current = query[start : start + 512]
        similarity = current @ spot.T
        k = min(top_k, similarity.shape[1])
        indices = np.argpartition(similarity, -k, axis=1)[:, -k:]
        if method == "bleep":
            predictions.append(train_expression[indices].mean(axis=1))
            continue
        selected = spot_embedding[indices]
        raw_query = image_embedding[start : start + len(current), None, :]
        distance = np.linalg.norm(selected - raw_query, axis=2)
        weights = 1.0 / np.maximum(distance * distance, 1e-12)
        weights /= weights.sum(axis=1, keepdims=True)
        predictions.append(np.einsum("nk,nkg->ng", weights, train_expression[indices]))
    return np.concatenate(predictions).astype(np.float32)


def train_retrieval(
    method: str,
    model: nn.Module,
    train_feature: np.ndarray,
    train_target: np.ndarray,
    train_position: np.ndarray,
    val_feature: np.ndarray,
    val_target: np.ndarray,
    val_position: np.ndarray,
    slides: Mapping[str, object],
    val_ids: Sequence[str],
    device: torch.device,
    epochs: int,
    batch_size: int,
    patience: int,
    learning_rate: float,
    weight_decay: float,
    retrieval_k: int,
    seed: int,
) -> tuple[dict[str, torch.Tensor], list[dict[str, object]], float, int]:
    fixed_final_epoch = (
        os.environ.get("LIN_MNP_CHECKPOINT_MODE") == "fixed_final_epoch"
    )
    dataset = TensorDataset(
        torch.from_numpy(np.asarray(train_feature)),
        torch.from_numpy(train_target),
        torch.from_numpy(train_position),
    )
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, generator=generator)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    best_state: dict[str, torch.Tensor] | None = None
    best_pcc = -float("inf")
    best_epoch = 0
    stale = 0
    history = []
    for epoch in range(1, epochs + 1):
        model.train()
        losses = []
        for image, expression, position in loader:
            image = image.to(device)
            expression = expression.to(device)
            position = position.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = model.loss(image, expression, position)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        train_spot = encode_chunks(
            model, "expression", train_target, device,
            positions=train_position if method == "mclstexp" else None,
        )
        val_image = encode_chunks(model, "image", val_feature, device)
        prediction = retrieve(
            val_image, train_spot, train_target, method, retrieval_k
        )
        pcc = mean_slide_gene_pcc(prediction, val_target, slides, val_ids)
        history.append({"epoch": epoch, "train_loss": float(np.mean(losses)), "validation_mean_gene_pcc": pcc})
        if fixed_final_epoch or pcc > best_pcc:
            best_pcc = pcc
            best_epoch = epoch
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
        if not fixed_final_epoch and stale >= patience:
            break
    if best_state is None:
        raise RuntimeError("training did not produce a checkpoint")
    return best_state, history, best_pcc, best_epoch


def train_igidl(
    model: IGIDLAdapter,
    train_feature: np.ndarray,
    train_nuclei: np.ndarray,
    train_target: np.ndarray,
    val_feature: np.ndarray,
    val_nuclei: np.ndarray,
    val_target: np.ndarray,
    slides: Mapping[str, object],
    val_ids: Sequence[str],
    device: torch.device,
    epochs: int,
    batch_size: int,
    patience: int,
    learning_rate: float,
    weight_decay: float,
    seed: int,
) -> tuple[dict[str, torch.Tensor], list[dict[str, object]], float, int]:
    fixed_final_epoch = (
        os.environ.get("LIN_MNP_CHECKPOINT_MODE") == "fixed_final_epoch"
    )
    dataset = TensorDataset(
        torch.from_numpy(np.asarray(train_feature)),
        torch.from_numpy(train_nuclei),
        torch.from_numpy(train_target),
    )
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=True,
        generator=torch.Generator().manual_seed(seed),
        # IGI-DL uses BatchNorm1d in both streams.  A singleton remainder is
        # invalid in training mode, so omit only that final shuffled item.
        drop_last=(len(dataset) % batch_size == 1),
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    best_state: dict[str, torch.Tensor] | None = None
    best_pcc = -float("inf")
    best_epoch = 0
    stale = 0
    history = []
    for epoch in range(1, epochs + 1):
        model.train()
        losses = []
        for image, nuclei, target in loader:
            optimizer.zero_grad(set_to_none=True)
            prediction = model(image.to(device), nuclei.to(device))
            loss = F.mse_loss(prediction, target.to(device))
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        prediction = predict_igidl(model, val_feature, val_nuclei, device)
        pcc = mean_slide_gene_pcc(prediction, val_target, slides, val_ids)
        history.append({"epoch": epoch, "train_loss": float(np.mean(losses)), "validation_mean_gene_pcc": pcc})
        if fixed_final_epoch or pcc > best_pcc:
            best_pcc = pcc
            best_epoch = epoch
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
        if not fixed_final_epoch and stale >= patience:
            break
    if best_state is None:
        raise RuntimeError("training did not produce a checkpoint")
    return best_state, history, best_pcc, best_epoch


@torch.no_grad()
def predict_igidl(
    model: IGIDLAdapter,
    feature: np.ndarray,
    nuclei: np.ndarray,
    device: torch.device,
) -> np.ndarray:
    model.eval()
    rows = []
    for start in range(0, len(feature), 512):
        image = torch.from_numpy(np.asarray(feature[start : start + 512])).to(device)
        current_nuclei = torch.from_numpy(nuclei[start : start + 512]).to(device)
        rows.append(model(image, current_nuclei).float().cpu().numpy())
    return np.concatenate(rows).astype(np.float32)


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
        raise ValueError("prediction split does not cover matrix")
    return result


def train_fold(args: argparse.Namespace) -> Path:
    specification = METHODS[args.method]
    folds = load_folds()
    fold = folds[args.fold]
    fixed_final_epoch = (
        os.environ.get("LIN_MNP_CHECKPOINT_MODE") == "fixed_final_epoch"
    )
    slides, genes = load_slides()
    parameters = fit_target_transform(slides, fold.train_ids, specification["target_transform"])
    output = (args.output / args.method / f"fold{fold.fold_id}").resolve()
    if (output / "complete.json").is_file():
        return output
    if output.exists():
        raise FileExistsError(f"incomplete fold output exists: {output}")
    staging = output.with_name(f"{output.name}.tmp-{os.getpid()}")
    staging.mkdir(parents=True, exist_ok=False)
    started = time.time()
    seed = args.seed + fold.fold_id
    seed_everything(seed)
    device = torch.device(args.device)

    # A shared target-free cache may be a superset of the current HEST/STImage
    # contract.  Global patch offsets, rather than the sum of this contract's
    # rows, define the minimum compatible cache extent.
    expected_rows = max(slide.patch_stop for slide in slides.values())
    cache, cache_path, cache_metadata_path = feature_cache(
        str(specification["backbone"]), expected_rows
    )
    train_feature = concatenate_rows(slides, cache, fold.train_ids)
    val_feature = concatenate_rows(slides, cache, fold.val_ids)
    test_feature = concatenate_rows(slides, cache, fold.test_ids)
    train_target = target_rows(slides, fold.train_ids, parameters)
    val_target = target_rows(slides, fold.val_ids, parameters)
    train_position = position_rows(slides, fold.train_ids)
    val_position = position_rows(slides, fold.val_ids)

    compatibility = [
        "shared target-free 224px frozen ImageNet feature cache replaces end-to-end backbone updates",
        "fixed patient-CV4 split and locked 997-gene panel replace the source dataset split/panel",
    ]
    nuclei_metadata: dict[str, object] = {}
    if args.method == "bleep":
        model: nn.Module = BLEEPAdapter(train_feature.shape[1], train_target.shape[1]).to(device)
        best_state, history, best_pcc, best_epoch = train_retrieval(
            args.method, model, train_feature, train_target, train_position,
            val_feature, val_target, val_position, slides, fold.val_ids, device,
            args.epochs, args.batch_size, args.patience,
            float(specification["learning_rate"]), float(specification["weight_decay"]),
            int(specification["retrieval_k"]), seed,
        )
    elif args.method == "mclstexp":
        all_position = np.concatenate([train_position, val_position, position_rows(slides, fold.test_ids)])
        model = MCLAdapter(
            train_feature.shape[1], train_target.shape[1],
            all_position.min(axis=0), all_position.max(axis=0),
        ).to(device)
        best_state, history, best_pcc, best_epoch = train_retrieval(
            args.method, model, train_feature, train_target, train_position,
            val_feature, val_target, val_position, slides, fold.val_ids, device,
            args.epochs, args.batch_size, args.patience,
            float(specification["learning_rate"]), float(specification["weight_decay"]),
            int(specification["retrieval_k"]), seed,
        )
    else:
        train_nuclei, nuclei_names, train_hashes = nuclei_rows(slides, fold.train_ids)
        val_nuclei, _, val_hashes = nuclei_rows(slides, fold.val_ids)
        test_nuclei, _, test_hashes = nuclei_rows(slides, fold.test_ids)
        nuclei_mean = train_nuclei.mean(axis=0)
        nuclei_scale = train_nuclei.std(axis=0, ddof=1)
        nuclei_scale[~np.isfinite(nuclei_scale) | (nuclei_scale < 1e-6)] = 1.0
        train_nuclei = ((train_nuclei - nuclei_mean) / nuclei_scale).astype(np.float32)
        val_nuclei = ((val_nuclei - nuclei_mean) / nuclei_scale).astype(np.float32)
        test_nuclei = ((test_nuclei - nuclei_mean) / nuclei_scale).astype(np.float32)
        model = IGIDLAdapter(train_feature.shape[1], train_nuclei.shape[1], len(genes)).to(device)
        best_state, history, best_pcc, best_epoch = train_igidl(
            model, train_feature, train_nuclei, train_target,
            val_feature, val_nuclei, val_target, slides, fold.val_ids, device,
            args.epochs, args.batch_size, args.patience,
            float(specification["learning_rate"]), float(specification["weight_decay"]), seed,
        )
        compatibility.extend([
            "released per-nucleus 55-feature graphs are unavailable for the canonical spots",
            "target-free five-feature Cellpose aggregate descriptors replace the GIN graph branch",
            "official same-cohort cSCC checkpoint is not loaded because it violates patient-disjoint CV4",
        ])
        nuclei_metadata = {
            "feature_names": list(nuclei_names),
            "train_mean": nuclei_mean.tolist(),
            "train_scale": nuclei_scale.tolist(),
            "input_sha256": train_hashes + val_hashes + test_hashes,
        }

    model.load_state_dict(best_state)
    checkpoint = staging / "best_checkpoint.pt"
    torch.save(best_state, checkpoint)
    if args.method in {"bleep", "mclstexp"}:
        train_spot = encode_chunks(
            model, "expression", train_target, device,
            positions=train_position if args.method == "mclstexp" else None,
        )
        test_image = encode_chunks(model, "image", test_feature, device)
        test_matrix = retrieve(
            test_image, train_spot, train_target, args.method,
            int(specification["retrieval_k"]),
        )
    else:
        test_matrix = predict_igidl(model, test_feature, test_nuclei, device)
    test_prediction = split_predictions(test_matrix, slides, fold.test_ids)

    history_path = staging / "training_history.csv"
    with history_path.open("w", newline="", encoding="utf-8") as handle:
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
        prediction_rows.append({
            "sample_id": sample_id,
            "patient_id": slides[sample_id].patient_id,
            "path": str(path.relative_to(staging)),
            "sha256": sha256_file(path),
            "n_spots": len(slides[sample_id].barcodes),
            "n_genes": len(genes),
        })

    source_dir = {
        "bleep": PROJECT_ROOT / "comparators/methods/BLEEP",
        "mclstexp": PROJECT_ROOT / "comparators/methods/mclSTExp",
        "igidl": PROJECT_ROOT / "comparators/methods/IGI-DL",
    }[args.method]
    source_files = {
        "bleep": [source_dir / "models.py", source_dir / "modules.py"],
        "mclstexp": [source_dir / "model.py", source_dir / "evel_cscc.py"],
        "igidl": [
            source_dir / "model_training/integrated_model/GNN_CNN_fusion_model.py",
            source_dir / "preprocessing/gene_count_transform.py",
        ],
    }[args.method]
    method_metadata = {
        "method_id": args.method,
        "display_name": specification["display_name"],
        "source_dir": str(source_dir.relative_to(PROJECT_ROOT)),
        "source_kind": "official released defining equations with explicitly recorded patient-CV4 adapter",
        "target_transform": specification["target_transform"],
        "frozen_backbone": specification["backbone"],
        "retrieval_k": specification["retrieval_k"],
        "runtime_compatibility_patches": compatibility,
    }
    metadata = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "method": method_metadata,
        "source_revision": source_revision(source_dir),
        "official_source_sha256": {str(path.relative_to(source_dir)): sha256_file(path) for path in source_files},
        "fold": {
            "fold_id": fold.fold_id,
            "train_ids": list(fold.train_ids),
            "val_ids": list(fold.val_ids),
            "test_ids": list(fold.test_ids),
        },
        "seed": seed,
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
        "selection_policy": (
            "fixed epoch selected by inner validation, followed by all-outer-train refit"
            if fixed_final_epoch
            else os.environ.get(
                "LIN_MNP_CHECKPOINT_SELECTION_SCOPE",
                (
                    "outer-test PCC under HESPotEx-aligned section-LOSO"
                    if os.environ.get("LIN_MNP_EXTERNAL_CONTRACT")
                    else "validation-only PCC under Benchmark patient outer-CV"
                ),
            )
        ),
        "best_validation_mean_gene_pcc": best_pcc,
        "target_parameters": serialise_parameters(parameters),
        "coverage_contract": "all_outer_test_slides",
        "patch_manifest_sha256": sha256_file(PATCH_ROOT / "manifest.json"),
        "feature_cache": {
            "path": str(cache_path),
            "sha256": sha256_file(cache_path),
            "metadata_path": str(cache_metadata_path),
            "metadata_sha256": sha256_file(cache_metadata_path),
        },
        "nuclei_adapter": nuclei_metadata,
        "predictions_contain_truth": False,
        "predictions": prediction_rows,
        "wall_time_seconds": time.time() - started,
    }
    metadata_path = staging / "run_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    complete = {
        "run_metadata_sha256": sha256_file(metadata_path),
        "training_history_sha256": sha256_file(history_path),
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
    parser.add_argument("--method", required=True, choices=sorted(METHODS))
    parser.add_argument("--fold", type=int, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    print(json.dumps({"output": str(train_fold(parse_args(argv)))}, sort_keys=True))


if __name__ == "__main__":
    main()
