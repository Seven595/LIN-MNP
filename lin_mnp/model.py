#!/usr/bin/env python3
"""Reference implementation of Lin-MNP for the cSCC patient-CV4 benchmark.

Fixed, not selected:
  raw 768-d CTransPath context, ridge α=1e6, one cross-section diffusion step,
  top-64 neighbors per other section, inverse temperature=10, equal Borda of
  context and local ranks, equal mass over other sections.

Each outer fold fits one patient-balanced shared Ridge model on the two training
patients and predicts the held-out test patient without opening its expression
labels.  Evaluation is a separate command that verifies all four prediction
artifacts before loading test labels.

Training targets are log1p raw counts standardized within each section and
gene. Outputs remain in that standardized expression space. Within-section
gene-wise PCC is unchanged if its scoring reference is unstandardized log1p
raw counts; squared-error metrics do depend on this target treatment.

Usage:
  python -m lin_mnp.model train --fold 0 --device cpu
  python -m lin_mnp.model evaluate
  python -m lin_mnp.model infer --checkpoint checkpoint.pt --sections S1 S2 --output pred.npz
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from lin_mnp.paths import CACHE_ROOT, DATA_ROOT, RESULTS_ROOT


PACKAGE_ROOT = Path(__file__).resolve().parent
SOURCE_PATH = Path(__file__).resolve()
CONTEXT_DIM = 768
RIDGE_ALPHA = 1_000_000.0
DIFFUSION_DEPTH = 1
NEIGHBORS_PER_OTHER_SECTION = 64
INVERSE_TEMPERATURE = 10.0

CONFIG = {
    "schema_version": 1,
    "method_id": "lin_mnp",
    "scientific_model_identity": "Lin-MNP",
    "folds": PACKAGE_ROOT / "configs/cscc_patient_cv4.json",
    "patch_cache": CACHE_ROOT / "cscc/raw_patches",
    "foundation_cache": CACHE_ROOT / "cscc/ctranspath_context",
    "local_core_manifest": CACHE_ROOT / "cscc/ctranspath_local/manifest.json",
    "label_root": DATA_ROOT / "cscc/labels",
    "label_field": "log1p_raw_counts",
    "gene_count": 997,
    "ordered_gene_checksum": (
        "b0faa60e845b767ff7caf61bfe14c1133560ca8f84a6c2ec2009cda0f93a4888"
    ),
    "runtime_root": RESULTS_ROOT / "cscc/folds",
}

_VERIFIED_POOLED = set()


def sha256(path: Path | str) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def array_sha256(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    return hashlib.sha256(memoryview(array).cast("B")).hexdigest()


def decode(values: np.ndarray) -> tuple[str, ...]:
    return tuple(
        value.decode("utf-8") if isinstance(value, bytes) else str(value)
        for value in np.asarray(values).tolist()
    )


def section_zscore(values: np.ndarray) -> np.ndarray:
    values64 = np.asarray(values, dtype=np.float64)
    mean = values64.mean(axis=0, keepdims=True)
    scale = values64.std(axis=0, ddof=0, keepdims=True)
    return np.divide(
        values64 - mean, scale, out=np.zeros_like(values64), where=scale > 1e-12,
    ).astype(np.float32)


def section_pcc(
    target: np.ndarray, prediction: np.ndarray, sections: Sequence[str]
) -> float:
    sections = np.asarray(sections)
    values = []
    for section in dict.fromkeys(str(value) for value in sections):
        active = sections == section
        left = np.asarray(target[active], dtype=np.float64)
        right = np.asarray(prediction[active], dtype=np.float64)
        left -= left.mean(0, keepdims=True)
        right -= right.mean(0, keepdims=True)
        denominator = np.sqrt(np.square(left).sum(0) * np.square(right).sum(0))
        values.append(float(np.divide(
            (left * right).sum(0), denominator,
            out=np.zeros(left.shape[1]), where=denominator > 1e-12,
        ).mean()))
    return float(np.mean(values))


def section_mse(
    target: np.ndarray, prediction: np.ndarray, sections: Sequence[str]
) -> float:
    sections = np.asarray(sections)
    return float(np.mean([
        np.square(target[sections == section] - prediction[sections == section]).mean()
        for section in dict.fromkeys(str(value) for value in sections)
    ]))


def normalize_rows(value: np.ndarray) -> np.ndarray:
    value = np.asarray(value, dtype=np.float32)
    if value.ndim != 2 or not np.isfinite(value).all():
        raise ValueError("descriptor matrix must be finite and two-dimensional")
    return value / np.maximum(np.linalg.norm(value, axis=1, keepdims=True), 1e-12)


def patient_equal_weights(
    patients: Sequence[str], patient_ids: Sequence[str]
) -> np.ndarray:
    patients = np.asarray(patients)
    patient_ids = tuple(str(value) for value in patient_ids)
    if not patient_ids or set(map(str, patients)) != set(patient_ids):
        raise ValueError("Lin-MNP patient set mismatch")
    count = len(patients)
    weight = np.zeros(count, dtype=np.float64)
    for patient in patient_ids:
        active = patients == patient
        if not np.any(active):
            raise ValueError("Lin-MNP empty training patient")
        weight[active] = count / (len(patient_ids) * int(active.sum()))
    if not np.isclose(weight.sum(), count):
        raise ValueError("Lin-MNP patient weight normalization failure")
    return weight


def fit_patient_balanced_ridge(
    feature: np.ndarray,
    target: np.ndarray,
    patients: Sequence[str],
    patient_ids: Sequence[str],
    device: torch.device | str,
) -> dict[str, Any]:
    feature = np.asarray(feature, dtype=np.float32)
    target = np.asarray(target, dtype=np.float32)
    if feature.ndim != 2 or feature.shape[1] != CONTEXT_DIM:
        raise ValueError("Lin-MNP requires a raw 768-dimensional context feature")
    if target.ndim != 2 or target.shape[0] != feature.shape[0]:
        raise ValueError("Lin-MNP feature and target matrices must align by spot")
    if len(patients) != feature.shape[0]:
        raise ValueError("Lin-MNP patient identifiers must align with the matrices")
    if not np.isfinite(feature).all() or not np.isfinite(target).all():
        raise ValueError("Lin-MNP fitting inputs must be finite")
    weight = patient_equal_weights(patients, patient_ids)
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
    gram.diagonal().add_(RIDGE_ALPHA)
    coefficient = torch.linalg.solve(gram, x.T @ y).float().cpu().numpy()
    return {
        "feature_mean": feature_mean.astype(np.float32),
        "feature_scale": feature_scale.astype(np.float32),
        "target_mean": target_mean.astype(np.float32),
        "weight": coefficient.astype(np.float32),
        "alpha": RIDGE_ALPHA,
        "decoder_kind": "patient_balanced_shared_ridge",
    }


def predict_ridge(feature: np.ndarray, artifact: Mapping[str, Any]) -> np.ndarray:
    feature = np.asarray(feature, dtype=np.float32)
    if feature.ndim != 2 or feature.shape[1] != CONTEXT_DIM:
        raise ValueError("Lin-MNP requires a raw 768-dimensional context feature")
    if not np.isfinite(feature).all():
        raise ValueError("Lin-MNP prediction features must be finite")
    x = ((feature - artifact["feature_mean"]) / artifact["feature_scale"])
    return (x @ artifact["weight"] + artifact["target_mean"]).astype(np.float32)


def top(similarity: np.ndarray, count: int) -> np.ndarray:
    similarity = np.asarray(similarity)
    if similarity.ndim != 2 or count <= 0:
        raise ValueError("neighbour selection requires a matrix and positive count")
    if similarity.shape[1] < count:
        raise ValueError("Lin-MNP has insufficient cross-section neighbours")
    partial = np.argpartition(-similarity, count - 1, axis=1)[:, :count]
    score = np.take_along_axis(similarity, partial, axis=1)
    order = np.argsort(-score, axis=1, kind="stable")
    return np.take_along_axis(partial, order, axis=1)


def softmax(
    score: np.ndarray, inverse_temperature: float = INVERSE_TEMPERATURE
) -> np.ndarray:
    score = np.asarray(score, dtype=np.float32)
    if score.ndim != 2 or score.shape[1] == 0 or not np.isfinite(score).all():
        raise ValueError("propagation scores must be a finite nonempty matrix")
    if not np.isfinite(inverse_temperature) or inverse_temperature <= 0:
        raise ValueError("inverse temperature must be finite and positive")
    logits = float(inverse_temperature) * (score - score.max(1, keepdims=True))
    weight = np.exp(logits)
    return (weight / weight.sum(1, keepdims=True)).astype(np.float32)


def pair_similarity(
    context: np.ndarray,
    local: np.ndarray,
    query: np.ndarray,
    gallery: np.ndarray,
) -> np.ndarray:
    context_score = context[query] @ context[gallery].T
    local_score = local[query] @ local[gallery].T
    context_order = np.argsort(-context_score, axis=1, kind="stable")
    local_order = np.argsort(-local_score, axis=1, kind="stable")
    context_rank = np.empty(context_order.shape, dtype=np.int32)
    local_rank = np.empty(local_order.shape, dtype=np.int32)
    rank = np.arange(context_order.shape[1], dtype=np.int32)[None, :]
    np.put_along_axis(context_rank, context_order, rank, axis=1)
    np.put_along_axis(local_rank, local_order, rank, axis=1)
    denominator = float(max(context_order.shape[1] - 1, 1))
    return 1.0 - (context_rank + local_rank).astype(np.float32) / (2.0 * denominator)


def variant_similarity(
    context: np.ndarray,
    local: np.ndarray,
    query: np.ndarray,
    gallery: np.ndarray,
    fusion: str,
) -> np.ndarray:
    """Return the descriptor similarity used by propagation-control analyses.

    Inputs must already be row-normalized. ``borda`` is the released Lin-MNP
    rank fusion, ``score`` averages the two cosine similarities, and
    ``context`` uses the contextual view alone.
    """

    if fusion == "borda":
        return pair_similarity(context, local, query, gallery)
    context_score = context[query] @ context[gallery].T
    if fusion == "context":
        return context_score.astype(np.float32)
    if fusion == "score":
        local_score = local[query] @ local[gallery].T
        return (0.5 * (context_score + local_score)).astype(np.float32)
    raise ValueError(f"unknown descriptor fusion: {fusion}")


def build_section_balanced_graph(
    sections: Sequence[str], context: np.ndarray, local: np.ndarray
) -> dict[str, np.ndarray]:
    sections = np.asarray(sections)
    section_ids = tuple(dict.fromkeys(str(value) for value in sections))
    if len(section_ids) < 2:
        raise ValueError("Lin-MNP requires multiple sections")
    context = normalize_rows(context)
    local = normalize_rows(local)
    total = NEIGHBORS_PER_OTHER_SECTION * (len(section_ids) - 1)
    index = np.empty((len(sections), total), dtype=np.int64)
    weight = np.empty((len(sections), total), dtype=np.float32)
    for query_section in section_ids:
        query = np.flatnonzero(sections == query_section)
        offset = 0
        for gallery_section in section_ids:
            if gallery_section == query_section:
                continue
            gallery = np.flatnonzero(sections == gallery_section)
            similarity = pair_similarity(context, local, query, gallery)
            selected = top(similarity, NEIGHBORS_PER_OTHER_SECTION)
            block = slice(offset, offset + NEIGHBORS_PER_OTHER_SECTION)
            index[query, block] = gallery[selected]
            weight[query, block] = softmax(
                np.take_along_axis(similarity, selected, axis=1),
                INVERSE_TEMPERATURE,
            ) / float(len(section_ids) - 1)
            offset += NEIGHBORS_PER_OTHER_SECTION
    if not np.allclose(weight.sum(1), 1.0, atol=2e-6):
        raise ValueError("Lin-MNP graph is not row stochastic")
    return {"index": index, "weight": weight}


def cross_step(
    prediction: np.ndarray,
    graph: Mapping[str, np.ndarray],
    chunk_size: int = 256,
) -> np.ndarray:
    value = np.asarray(prediction, dtype=np.float32)
    output = np.empty_like(value)
    for start in range(0, len(value), int(chunk_size)):
        stop = min(start + int(chunk_size), len(value))
        output[start:stop] = np.einsum(
            "nk,nkg->ng", graph["weight"][start:stop],
            value[graph["index"][start:stop]], optimize=True,
        )
    return output


def diffuse(
    prediction: np.ndarray,
    graph: Mapping[str, np.ndarray],
    depth: int = DIFFUSION_DEPTH,
) -> np.ndarray:
    value = np.asarray(prediction, dtype=np.float32)
    for _ in range(int(depth)):
        value = cross_step(value, graph)
    return value


def forward(
    context: np.ndarray,
    local: np.ndarray,
    sections: Sequence[str],
    artifact: Mapping[str, Any],
) -> np.ndarray:
    """Label-free prediction: shared ridge on raw context, then one diffusion step."""
    graph = build_section_balanced_graph(sections, context, local)
    return diffuse(predict_ridge(context, artifact), graph, DIFFUSION_DEPTH)


class LabeledSpotDataset:
    """Load aligned descriptors and section-standardized log1p expression."""

    def __init__(self, config, section_ids):
        cache_root = Path(config["patch_cache"])
        raw_manifest_path = cache_root / "manifest.json"
        raw_manifest = json.loads(raw_manifest_path.read_text(encoding="utf-8"))
        if (
            not raw_manifest["target_free"]
            or raw_manifest["foundation_feature_values_accessed"]
            or raw_manifest["native_crop_px"] != 384
            or raw_manifest["network_input_px"] != 224
            or not raw_manifest["both_branches_receive_identical_tensor"]
        ):
            raise ValueError("context raw cache contract failure")
        index_path = cache_root / raw_manifest["index_file"]
        if (
            "index_file_sha256" in raw_manifest
            and sha256(index_path) != raw_manifest["index_file_sha256"]
        ):
            raise ValueError("context raw-cache index checksum failure")
        with index_path.open(
            encoding="utf-8", newline=""
        ) as handle:
            index_rows = list(csv.DictReader(handle))
        rows_by_section = {}
        for row in index_rows:
            rows_by_section.setdefault(row["section"], []).append(row)
        self.patch_indices = []
        targets = []
        self.sections = []
        self.patients = []
        self.barcodes = []
        self.genes = None
        label_root = Path(config["label_root"])
        excluded = tuple(config.get("excluded_genes", []))
        for section in section_ids:
            rows = rows_by_section[section]
            with np.load(label_root / f"{section}.npz", allow_pickle=False) as archive:
                barcodes = decode(archive["barcodes"])
                source_genes = decode(archive["genes"])
                patient = str(archive["patient_id"].item())
                raw_target = np.asarray(archive[config["label_field"]], dtype=np.float32)
            retained = np.asarray(
                [gene not in excluded for gene in source_genes], dtype=bool
            )
            genes = tuple(gene for gene, keep in zip(source_genes, retained) if keep)
            raw_target = raw_target[:, retained]
            if len(genes) != int(config["gene_count"]):
                raise ValueError("context gene count failure")
            if hashlib.sha256("\n".join(genes).encode()).hexdigest() != config[
                "ordered_gene_checksum"
            ]:
                raise ValueError("context gene order failure")
            if self.genes is None:
                self.genes = genes
            elif genes != self.genes:
                raise ValueError("context cross-section gene order failure")
            if tuple(row["barcode"] for row in rows) != barcodes:
                raise ValueError("context cache/label barcode failure")
            self.patch_indices.extend(int(row["global_index"]) for row in rows)
            targets.append(section_zscore(raw_target))
            self.sections.extend([section] * len(rows))
            self.patients.extend([patient] * len(rows))
            self.barcodes.extend(barcodes)
        self.target_matrix = np.concatenate(targets, axis=0)
        foundation_root = Path(config["foundation_cache"])
        foundation = json.loads((foundation_root / "manifest.json").read_text(encoding="utf-8"))
        if (
            not foundation["target_free"]
            or foundation["expression_values_accessed"]
            or foundation["foundation_model"] != "CTransPath"
            or foundation["native_context_pixels"] != 384
            or foundation["shared_branch_input_resolution"] != 224
        ):
            raise ValueError("context feature cache provenance failure")
        if foundation["raw_patch_cache_manifest_sha256"] != sha256(raw_manifest_path):
            raise ValueError("context feature/raw alignment failure")
        pooled_path = foundation_root / foundation["pooled_file"]
        resolved = foundation_root.resolve()
        if resolved not in _VERIFIED_POOLED:
            if sha256(pooled_path) != foundation["pooled_sha256"]:
                raise ValueError("context pooled checksum failure")
            _VERIFIED_POOLED.add(resolved)
        self.ctranspath = np.load(pooled_path, mmap_mode="r")
        if self.ctranspath.shape != (int(raw_manifest["patch_count"]), CONTEXT_DIM):
            raise ValueError("context CTransPath shape failure")


class InferenceSpotDataset:
    def __init__(self, config, section_ids):
        cache_root = Path(config["patch_cache"])
        raw_manifest_path = cache_root / "manifest.json"
        raw_manifest = json.loads(raw_manifest_path.read_text(encoding="utf-8"))
        if (
            not raw_manifest["target_free"]
            or raw_manifest["foundation_feature_values_accessed"]
            or raw_manifest["native_crop_px"] != 384
            or raw_manifest["network_input_px"] != 224
            or not raw_manifest["both_branches_receive_identical_tensor"]
        ):
            raise ValueError("label-free context raw cache contract failure")
        index_path = cache_root / raw_manifest["index_file"]
        if (
            "index_file_sha256" in raw_manifest
            and sha256(index_path) != raw_manifest["index_file_sha256"]
        ):
            raise ValueError("label-free raw-cache index checksum failure")
        with index_path.open(
            encoding="utf-8", newline=""
        ) as handle:
            rows = list(csv.DictReader(handle))
        rows_by_section = {}
        for row in rows:
            rows_by_section.setdefault(row["section"], []).append(row)
        self.patch_indices = []
        self.sections = []
        self.patients = []
        self.barcodes = []
        requested = tuple(map(str, section_ids))
        if not requested or len(set(requested)) != len(requested):
            raise ValueError("label-free section list must be nonempty and unique")
        for section in requested:
            local_rows = rows_by_section[section]
            self.patch_indices.extend(int(row["global_index"]) for row in local_rows)
            self.sections.extend([section] * len(local_rows))
            self.patients.extend(row["patient"] for row in local_rows)
            self.barcodes.extend(row["barcode"] for row in local_rows)
        if len(set(self.patients)) != 1:
            raise ValueError("Lin-MNP inference sections must belong to one patient")
        foundation_root = Path(config["foundation_cache"])
        foundation = json.loads((foundation_root / "manifest.json").read_text(encoding="utf-8"))
        if (
            not foundation["target_free"]
            or foundation["expression_values_accessed"]
            or foundation["foundation_model"] != "CTransPath"
            or foundation["native_context_pixels"] != 384
            or foundation["shared_branch_input_resolution"] != 224
        ):
            raise ValueError("label-free context feature provenance failure")
        if foundation["raw_patch_cache_manifest_sha256"] != sha256(raw_manifest_path):
            raise ValueError("label-free context feature/raw alignment failure")
        pooled_path = foundation_root / foundation["pooled_file"]
        resolved = foundation_root.resolve()
        if resolved not in _VERIFIED_POOLED:
            if sha256(pooled_path) != foundation["pooled_sha256"]:
                raise ValueError("label-free context pooled checksum failure")
            _VERIFIED_POOLED.add(resolved)
        self.ctranspath = np.load(pooled_path, mmap_mode="r")
        if self.ctranspath.shape != (int(raw_manifest["patch_count"]), CONTEXT_DIM):
            raise ValueError("label-free CTransPath cache shape failure")

    def context_array(self):
        indices = np.asarray(self.patch_indices, dtype=np.int64)
        return np.asarray(self.ctranspath[indices], dtype=np.float32)


def load_local_cache(config: Mapping[str, Any]) -> np.ndarray:
    manifest_path = Path(config["local_core_manifest"])
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (
        not manifest["target_free"]
        or manifest["expression_values_accessed"]
        or manifest["foundation_model"] != "CTransPath"
        or int(manifest["source_local_view_native_pixels"]) != 224
    ):
        raise ValueError("Lin-MNP local feature provenance failure")
    raw_manifest_path = Path(config["patch_cache"]) / "manifest.json"
    if (
        "raw_patch_cache_manifest_sha256" in manifest
        and manifest["raw_patch_cache_manifest_sha256"] != sha256(raw_manifest_path)
    ):
        raise ValueError("Lin-MNP local feature/raw alignment failure")
    path = manifest_path.parent / manifest["pooled_file"]
    if sha256(path) != manifest["pooled_sha256"]:
        raise ValueError("Lin-MNP local feature checksum failure")
    values = np.load(path, mmap_mode="r")
    if tuple(values.shape) != tuple(manifest["pooled_shape"]):
        raise ValueError("Lin-MNP local feature shape failure")
    if values.ndim != 2 or values.shape[1] != CONTEXT_DIM:
        raise ValueError("Lin-MNP local features must have shape [spots, 768]")
    return values


def _fold(config: Mapping[str, Any], fold_id: int) -> dict[str, Any]:
    folds = json.loads(Path(config["folds"]).read_text(encoding="utf-8"))["folds"]
    try:
        return next(row for row in folds if int(row["fold_id"]) == int(fold_id))
    except StopIteration as error:
        raise ValueError(f"unknown cSCC fold: {fold_id}") from error


def train_fold(
    fold_id: int,
    device_name: str,
    config: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Fit one outer fold and create target-free test predictions."""

    config = dict(CONFIG if config is None else config)
    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit(f"requested CUDA device is unavailable: {device_name}")
    fold = _fold(config, fold_id)
    output = Path(config["runtime_root"]) / f"fold{fold_id}"
    if output.exists():
        raise SystemExit(f"refusing to overwrite Lin-MNP fold: {output}")
    output.mkdir(parents=True)
    started = time.perf_counter()
    local_cache = load_local_cache(config)
    train_dataset = LabeledSpotDataset(config, fold["train_sample_ids"])
    train_index = np.asarray(train_dataset.patch_indices, dtype=np.int64)
    context = np.asarray(train_dataset.ctranspath[train_index], dtype=np.float32)
    patient_ids = tuple(fold["train_patient_ids"])
    shared = fit_patient_balanced_ridge(
        context, train_dataset.target_matrix, train_dataset.patients, patient_ids, device,
    )
    contract = {
        "geometry": "raw",
        "target_reference": config["label_field"],
        "training_target_transform": "within_section_gene_zscore",
        "prediction_space": "within_section_gene_zscore_log1p_raw_counts",
        "pcc_scoring_reference": "log1p_raw_counts",
        "alpha": RIDGE_ALPHA,
        "depth": DIFFUSION_DEPTH,
        "neighbors_per_other_section": NEIGHBORS_PER_OTHER_SECTION,
        "cross_inverse_temperature": INVERSE_TEMPERATURE,
        "selection": "none_frozen_contract",
    }
    checkpoint = {
        "schema_version": 1,
        "method_id": config["method_id"],
        "scientific_model_identity": config["scientific_model_identity"],
        "fold": int(fold_id),
        "genes": tuple(train_dataset.genes),
        "contract": contract,
        "shared_artifact": shared,
        "training_patient_count": len(patient_ids),
        "patient_specific_parameters": False,
        "validation_samples_not_opened_during_fit": fold["val_sample_ids"],
        "outer_test_samples_not_opened": fold["test_sample_ids"],
        "source_sha256": sha256(SOURCE_PATH),
    }
    checkpoint_path = output / "checkpoint.pt"
    torch.save(checkpoint, checkpoint_path)

    inference_dataset = InferenceSpotDataset(config, fold["test_sample_ids"])
    infer_index = np.asarray(inference_dataset.patch_indices, dtype=np.int64)
    context = inference_dataset.context_array()
    pointwise = predict_ridge(context, shared)
    prediction = forward(
        context,
        np.asarray(local_cache[infer_index], dtype=np.float32),
        np.asarray(inference_dataset.sections),
        shared,
    )
    prediction_path = output / "outer_test_predictions.npz"
    np.savez_compressed(
        prediction_path,
        pointwise=pointwise.astype(np.float32),
        lin_mnp=prediction.astype(np.float32),
        sections=np.asarray(inference_dataset.sections),
        barcodes=np.asarray(inference_dataset.barcodes),
        genes=np.asarray(train_dataset.genes),
    )

    record = {
        "schema_version": 1,
        "status": "COMPLETE",
        "method_id": config["method_id"],
        "scientific_model_identity": config["scientific_model_identity"],
        "fold": int(fold_id),
        "patient": fold["test_patient_id"],
        "contract": contract,
        "fit_samples": list(fold["train_sample_ids"]),
        "validation_samples_not_opened": list(fold["val_sample_ids"]),
        "test_samples": list(fold["test_sample_ids"]),
        "prediction_created_without_test_labels": True,
        "validation_labels_used_for_training_or_selection": False,
        "outer_test_labels_accessed": False,
        "checkpoint_sha256": sha256(checkpoint_path),
        "prediction_file_sha256": sha256(prediction_path),
        "pointwise_array_sha256": array_sha256(pointwise),
        "lin_mnp_array_sha256": array_sha256(prediction),
        "input_manifest_sha256": {
            "folds": sha256(config["folds"]),
            "raw_patches": sha256(Path(config["patch_cache"]) / "manifest.json"),
            "context": sha256(Path(config["foundation_cache"]) / "manifest.json"),
            "local": sha256(config["local_core_manifest"]),
        },
        "wall_seconds": time.perf_counter() - started,
    }
    metadata_path = output / "run_metadata.json"
    metadata_path.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    (output / "complete.json").write_text(
        json.dumps(
            {
                "status": "COMPLETE",
                "run_metadata_sha256": sha256(metadata_path),
                "prediction_file_sha256": sha256(prediction_path),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps(record, indent=2), flush=True)
    return record


def evaluate_folds(
    input_root: Path,
    config: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Verify and score all four outer-test folds with equal patient weight."""

    config = dict(CONFIG if config is None else config)
    output_path = Path(input_root) / "evaluation.json"
    if output_path.exists():
        raise SystemExit(f"refusing to overwrite evaluation: {output_path}")
    folds = json.loads(Path(config["folds"]).read_text(encoding="utf-8"))["folds"]
    expected_manifests = {
        "folds": sha256(config["folds"]),
        "raw_patches": sha256(Path(config["patch_cache"]) / "manifest.json"),
        "context": sha256(Path(config["foundation_cache"]) / "manifest.json"),
        "local": sha256(config["local_core_manifest"]),
    }
    verified: list[tuple[dict[str, Any], Path, dict[str, Any]]] = []
    for fold in folds:
        root = Path(input_root) / f"fold{int(fold['fold_id'])}"
        metadata_path = root / "run_metadata.json"
        prediction_path = root / "outer_test_predictions.npz"
        checkpoint_path = root / "checkpoint.pt"
        complete = json.loads((root / "complete.json").read_text(encoding="utf-8"))
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if complete["run_metadata_sha256"] != sha256(metadata_path):
            raise ValueError(f"metadata checksum failure: {root}")
        if complete["prediction_file_sha256"] != sha256(prediction_path):
            raise ValueError(f"prediction checksum failure: {root}")
        if metadata["checkpoint_sha256"] != sha256(checkpoint_path):
            raise ValueError(f"checkpoint checksum failure: {root}")
        if (
            int(metadata["fold"]) != int(fold["fold_id"])
            or str(metadata["patient"]) != str(fold["test_patient_id"])
        ):
            raise ValueError(f"fold identity mismatch: {root}")
        if metadata["outer_test_labels_accessed"]:
            raise ValueError(f"test labels were accessed during training: {root}")
        if metadata["input_manifest_sha256"] != expected_manifests:
            raise ValueError(f"input manifest mismatch: {root}")
        verified.append((fold, prediction_path, metadata))

    rows: list[dict[str, Any]] = []
    for fold, prediction_path, metadata in verified:
        target = LabeledSpotDataset(config, fold["test_sample_ids"])
        with np.load(prediction_path, allow_pickle=False) as archive:
            if {"target", "expression", "truth"} & set(archive.files):
                raise ValueError(f"prediction artifact contains labels: {prediction_path}")
            if decode(archive["sections"]) != tuple(target.sections):
                raise ValueError(f"section mismatch: {prediction_path}")
            if decode(archive["barcodes"]) != tuple(target.barcodes):
                raise ValueError(f"barcode mismatch: {prediction_path}")
            if decode(archive["genes"]) != tuple(target.genes):
                raise ValueError(f"gene mismatch: {prediction_path}")
            predictions = {
                "linear_readout": np.asarray(archive["pointwise"], dtype=np.float32),
                "lin_mnp": np.asarray(archive["lin_mnp"], dtype=np.float32),
            }
        for method, prediction in predictions.items():
            if prediction.shape != target.target_matrix.shape:
                raise ValueError(
                    f"{method} prediction/target shape mismatch: {prediction_path}"
                )
            if not np.isfinite(prediction).all():
                raise ValueError(f"{method} prediction contains non-finite values")
        if array_sha256(predictions["linear_readout"]) != metadata[
            "pointwise_array_sha256"
        ]:
            raise ValueError(f"pointwise array checksum failure: {prediction_path}")
        if array_sha256(predictions["lin_mnp"]) != metadata[
            "lin_mnp_array_sha256"
        ]:
            raise ValueError(f"Lin-MNP array checksum failure: {prediction_path}")
        for method, prediction in predictions.items():
            rows.append(
                {
                    "fold": int(fold["fold_id"]),
                    "patient": str(fold["test_patient_id"]),
                    "method": method,
                    "pcc": section_pcc(
                        target.target_matrix, prediction, target.sections
                    ),
                    "mse": section_mse(
                        target.target_matrix, prediction, target.sections
                    ),
                }
            )

    summary = {}
    for method in ("linear_readout", "lin_mnp"):
        selected = [row for row in rows if row["method"] == method]
        summary[method] = {
            "mean_pcc": float(np.mean([row["pcc"] for row in selected])),
            "mean_mse": float(np.mean([row["mse"] for row in selected])),
        }
    summary["lin_mnp"]["mean_pcc_change"] = (
        summary["lin_mnp"]["mean_pcc"] - summary["linear_readout"]["mean_pcc"]
    )
    result = {
        "schema_version": 1,
        "protocol": "cSCC patient-CV4 outer-test evaluation",
        "aggregation": "mean within section, then equal weight across test patients",
        "target_reference": config["label_field"],
        "evaluation_target_transform": "within_section_gene_zscore",
        "pcc_equivalent_reference": "unstandardized log1p raw counts within section",
        "mse_space": "within_section_gene_zscore_log1p_raw_counts",
        "folds": rows,
        "summary": summary,
    }
    output_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)
    return result


def infer(checkpoint_path, sections, output_path, config=None):
    config = dict(CONFIG if config is None else config)
    output_path = Path(output_path)
    if output_path.exists():
        raise SystemExit(f"refusing to overwrite inference output: {output_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if checkpoint["scientific_model_identity"] != config["scientific_model_identity"]:
        raise ValueError("Lin-MNP inference checkpoint identity failure")
    local_cache = load_local_cache(config)
    dataset = InferenceSpotDataset(config, list(sections))
    index = np.asarray(dataset.patch_indices, dtype=np.int64)
    prediction = forward(
        dataset.context_array(),
        np.asarray(local_cache[index], dtype=np.float32),
        np.asarray(dataset.sections),
        checkpoint["shared_artifact"],
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path, lin_mnp=prediction.astype(np.float32),
        sections=np.asarray(dataset.sections), barcodes=np.asarray(dataset.barcodes),
        genes=np.asarray(checkpoint["genes"]),
    )
    record = {
        "scientific_model_identity": config["scientific_model_identity"],
        "checkpoint": str(Path(checkpoint_path).resolve()),
        "sections": list(sections),
        "spot_count": len(prediction),
        "gene_count": int(prediction.shape[1]),
        "prediction_space": "within_section_gene_zscore_log1p_raw_counts",
        "lin_mnp_array_sha256": array_sha256(prediction),
        "expression_label_files_opened": False,
    }
    output_path.with_suffix(output_path.suffix + ".json").write_text(
        json.dumps(record, indent=2) + "\n"
    )
    print(json.dumps(record, indent=2))
    return record


def main():
    parser = argparse.ArgumentParser(description="Train or run Lin-MNP")
    sub = parser.add_subparsers(dest="command", required=True)
    train_parser = sub.add_parser("train")
    train_parser.add_argument("--fold", required=True, type=int, choices=range(4))
    train_parser.add_argument(
        "--device", default="cpu", help="Torch device for the float64 Ridge solve."
    )
    train_parser.add_argument(
        "--output-root",
        type=Path,
        help="Override LIN_MNP_RESULTS_ROOT/cscc/folds for this run.",
    )
    evaluate_parser = sub.add_parser(
        "evaluate", help="Verify and evaluate all four outer-test folds."
    )
    evaluate_parser.add_argument(
        "--input-root",
        type=Path,
        default=Path(CONFIG["runtime_root"]),
        help="Directory containing fold0 through fold3.",
    )
    infer_parser = sub.add_parser("infer")
    infer_parser.add_argument("--checkpoint", required=True, type=Path)
    infer_parser.add_argument("--sections", required=True, nargs="+")
    infer_parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.command == "train":
        config = dict(CONFIG)
        if args.output_root is not None:
            config["runtime_root"] = args.output_root
        train_fold(args.fold, args.device, config)
    elif args.command == "evaluate":
        evaluate_folds(args.input_root)
    else:
        infer(args.checkpoint, args.sections, args.output)


if __name__ == "__main__":
    main()
