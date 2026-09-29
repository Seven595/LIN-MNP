#!/usr/bin/env python3
"""Train benchmark-configured EGNv2 on one official cSCC patient fold."""

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
import types
from typing import Mapping, NamedTuple, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn as pyg


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


class SlideGraph(NamedTuple):
    sample_id: str
    window: torch.Tensor
    target: torch.Tensor
    example: torch.Tensor
    window_edge: torch.Tensor
    example_edge: torch.Tensor
    cross_edge: torch.Tensor


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build_model(n_genes: int) -> nn.Module:
    source = PROJECT_ROOT / "comparators/methods/EGN/v2"
    sys.path.insert(0, str(source))
    import model as author_model  # type: ignore

    model = author_model.HeteroGNN(num_layers=4, mdim=512)
    model.pretransform_exp = pyg.Linear(512 + n_genes, 512, bias=False)
    model.pretransform_ey = pyg.Linear(n_genes, 512, bias=False)
    model.lin = pyg.Linear(512, n_genes)

    def benchmark_forward(
        self: nn.Module,
        x_dict: Mapping[str, torch.Tensor],
        edge_index_dict: Mapping[tuple[str, str, str], torch.Tensor],
    ) -> torch.Tensor:
        raw_example = x_dict["example"]
        hidden = {
            "window": self.post_transform(self.pretransform_win(x_dict["window"])),
            "example": self.post_transform(self.pretransform_exp(raw_example)),
            "example_y": self.pretransform_ey(raw_example[:, -n_genes:]),
        }
        for convolution in self.convs:
            hidden = convolution(hidden, edge_index_dict)
            hidden = {key: self.leaklyrelu(value) for key, value in hidden.items()}
        return self.lin(self.pool(hidden, edge_index_dict))

    model.forward = types.MethodType(benchmark_forward, model)
    return model


def _indices(slide: object) -> np.ndarray:
    return np.arange(slide.patch_start, slide.patch_stop, dtype=np.int64)


@torch.no_grad()
def nearest_l1(
    query: np.ndarray,
    candidate: np.ndarray,
    candidate_indices: np.ndarray,
    device: torch.device,
    neighbors: int,
    query_indices: np.ndarray | None = None,
    chunk: int = 256,
) -> np.ndarray:
    reference = torch.from_numpy(np.asarray(candidate, dtype=np.float32)).to(device)
    reference_indices = torch.from_numpy(candidate_indices).to(device)
    rows = []
    for start in range(0, len(query), chunk):
        value = torch.from_numpy(
            np.asarray(query[start : start + chunk], dtype=np.float32)
        ).to(device)
        distance = torch.cdist(value, reference, p=1)
        if query_indices is not None:
            # A few official HEST folds contain only one training section.  EGN's
            # usual cross-section exemplar pool is then empty.  Use other spots
            # from that training section, while explicitly masking each query's
            # own row so its expression can never be copied into itself.
            query_global = torch.from_numpy(
                query_indices[start : start + len(value)]
            ).to(device)
            distance.masked_fill_(
                query_global[:, None] == reference_indices[None, :], float("inf")
            )
        rows.append(torch.topk(distance, neighbors, largest=False).indices.cpu().numpy())
    return candidate_indices[np.concatenate(rows)]


def build_graphs(
    slides: Mapping[str, object],
    train_ids: Sequence[str],
    all_targets: np.ndarray,
    embeddings: np.ndarray,
    device: torch.device,
) -> dict[str, SlideGraph]:
    graphs = {}
    for sample_id, slide in slides.items():
        candidate_ids = [value for value in train_ids if value != sample_id]
        query_index = _indices(slide)
        single_train_fallback = not candidate_ids
        candidate_index = (
            query_index
            if single_train_fallback
            else np.concatenate([_indices(slides[value]) for value in candidate_ids])
        )
        exemplar_index = nearest_l1(
            embeddings[query_index],
            embeddings[candidate_index],
            candidate_index,
            device,
            neighbors=6,
            query_indices=query_index if single_train_fallback else None,
        )
        own_embedding = torch.from_numpy(
            np.asarray(embeddings[query_index], dtype=np.float32).copy()
        )
        exemplar_embedding = torch.from_numpy(
            np.asarray(embeddings[exemplar_index], dtype=np.float32).copy()
        )
        exemplar_target = torch.from_numpy(all_targets[exemplar_index].copy())
        combined = torch.cat((exemplar_embedding, exemplar_target), dim=-1).reshape(
            -1, 512 + all_targets.shape[1]
        )
        unique, inverse = torch.unique(combined, dim=0, return_inverse=True)
        spot_index = torch.arange(len(query_index)).repeat_interleave(6)
        cross_edge = torch.stack((inverse, spot_index), dim=0)
        coordinates = torch.from_numpy(slide.array_coordinates.astype(np.float32))
        window_edge = pyg.radius_graph(
            coordinates,
            r=np.sqrt(2),
            loop=False,
            max_num_neighbors=5,
            flow="source_to_target",
            num_workers=1,
        )
        example_edge = pyg.knn_graph(unique, k=3, loop=False)
        graphs[sample_id] = SlideGraph(
            sample_id=sample_id,
            window=own_embedding,
            target=torch.from_numpy(all_targets[query_index].copy()),
            example=unique,
            window_edge=window_edge,
            example_edge=example_edge,
            cross_edge=cross_edge,
        )
    return graphs


def to_device(
    graph: SlideGraph, device: torch.device
) -> tuple[dict[str, torch.Tensor], dict[tuple[str, str, str], torch.Tensor], torch.Tensor]:
    nodes = {
        "window": graph.window.to(device),
        "example": graph.example.to(device),
    }
    edges = {
        ("window", "near", "window"): graph.window_edge.to(device),
        ("example", "close", "example"): graph.example_edge.to(device),
        ("example", "refer", "window"): graph.cross_edge.to(device),
    }
    return nodes, edges, graph.target.to(device)


def correlation_loss(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    pred = prediction - prediction.mean(0, keepdim=True)
    true = target - target.mean(0, keepdim=True)
    pred_ss = pred.square().sum(0)
    true_ss = true.square().sum(0)
    valid = (pred_ss > 1e-8) & (true_ss > 1e-8)
    denominator = torch.sqrt(torch.clamp(pred_ss * true_ss, min=1e-8))
    correlation = (pred * true).sum(0) / denominator
    if valid.any():
        return 1.0 - correlation[valid].mean()
    return prediction.sum() * 0.0


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


@torch.no_grad()
def predict(
    model: nn.Module,
    graphs: Mapping[str, SlideGraph],
    sample_ids: Sequence[str],
    device: torch.device,
) -> dict[str, np.ndarray]:
    model.eval()
    result = {}
    for sample_id in sample_ids:
        nodes, edges, _ = to_device(graphs[sample_id], device)
        result[sample_id] = model(nodes, edges).float().cpu().numpy()
    return result


def validation_score(
    predictions: Mapping[str, np.ndarray], graphs: Mapping[str, SlideGraph]
) -> float:
    values = [
        _pcc(graphs[sample_id].target.numpy(), predictions[sample_id])
        for sample_id in sorted(predictions)
    ]
    return float(np.nanmean(np.nanmean(np.stack(values), axis=0)))


def train_fold(args: argparse.Namespace) -> Path:
    embedding_path = EMBEDDING_ROOT / "resnet18_embeddings_float32.npy"
    embedding_metadata = EMBEDDING_ROOT / "resnet18_metadata.json"
    if not embedding_path.is_file() or not embedding_metadata.is_file():
        raise FileNotFoundError("run cache_benchmark_egn_embeddings.py --backbone resnet18")
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
    device = torch.device(args.device)
    embeddings = np.load(embedding_path, mmap_mode="r")
    graphs = build_graphs(slides, fold.train_ids, all_targets, embeddings, device)
    del embeddings

    output = (args.output / "egnv2" / f"fold{fold.fold_id}").resolve()
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
    model = build_model(len(genes)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-4)
    checkpoint = staging / "best_checkpoint.pt"
    best_score = -np.inf
    best_epoch = -1
    history = []
    started = time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        train_ids = list(fold.train_ids)
        random.shuffle(train_ids)
        loss_sum = 0.0
        spot_count = 0
        for sample_id in train_ids:
            nodes, edges, target = to_device(graphs[sample_id], device)
            optimizer.zero_grad(set_to_none=True)
            prediction = model(nodes, edges)
            loss = F.mse_loss(prediction, target) + 0.5 * correlation_loss(
                prediction, target
            )
            loss.backward()
            optimizer.step()
            loss_sum += float(loss.detach()) * len(target)
            spot_count += len(target)
        val_prediction = predict(model, graphs, fold.val_ids, device)
        score = validation_score(val_prediction, graphs)
        if fixed_final_epoch or (np.isfinite(score) and score > best_score):
            best_score = score
            best_epoch = epoch
            torch.save(
                {"model_state": model.state_dict(), "epoch": epoch, "validation_mean_gene_pcc": score},
                checkpoint,
            )
        record = {
            "epoch": epoch,
            "train_objective": loss_sum / max(spot_count, 1),
            "validation_mean_gene_pcc": score,
            "best_validation_mean_gene_pcc": best_score,
            "elapsed_seconds": time.time() - started,
        }
        history.append(record)
        print(json.dumps(record, sort_keys=True), flush=True)
    state = torch.load(checkpoint, map_location=device)
    model.load_state_dict(state["model_state"])
    test_prediction = predict(model, graphs, fold.test_ids, device)
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
        "method_id": "egnv2",
        "display_name": "EGNv2",
        "source_dir": "comparators/methods/EGN/v2",
        "source_kind": f"released EGNv2 modules with {len(genes)}-feature dimensional adapter",
        "target_transform": "egn_minmax_log10_raw",
        "epochs": args.epochs,
        "batch_size": 1,
        "learning_rate": 5e-4,
        "weight_decay": 1e-4,
        "input_size": 224,
        "optimizer": "AdamW",
        "nearest_exemplars": 6,
        "graph_layers": 4,
        "runtime_compatibility_patches": [
            f"250-to-{len(genes)} gene-dependent projections",
            "preserve raw exemplar expression before released pretransform",
            "zero-variance genes masked before batch-PCC square-root differentiation",
            "single-training-section folds use leave-self within-section exemplars",
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
        "mixed_precision": False,
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
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    print(json.dumps({"output": str(train_fold(parse_args(argv)))}, sort_keys=True))


if __name__ == "__main__":
    main()
