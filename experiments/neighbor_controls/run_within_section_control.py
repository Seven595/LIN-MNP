#!/usr/bin/env python3
"""Neighbour-selection controls for the cSCC Lin-MNP operator.

The pointwise readout is loaded from the reference Lin-MNP fold outputs and
held fixed. Only the graph changes. Every graph is built from target-free
descriptors or coordinates; outer-test labels are opened only by ``evaluate``
after all prediction artifacts have been verified.

Arms:

* ``identity`` -- the Ridge prediction with no spatial operator.
* ``within_section_borda_k64`` -- same Borda context/local similarity, k=64,
  gallery restricted to the query section excluding self.
* ``within_section_borda_k128`` -- same, k=128, matching Lin-MNP total
  neighbour count (64 per other section × 2 other sections).
* ``within_section_coord_k128`` -- Euclidean pixel-coordinate kNN, k=128,
  softmax on the same rank-to-[0,1] map used by Borda.
* ``within_section_random_k128`` -- uniform average of 128 random other spots
  in the same section (seeded).
* ``cross_section_random_k128`` -- 64 random spots from each other section,
  uniform within section, equal section mass.
* ``cross_section_lin_mnp`` -- the released operator (must recover 0.456993).

"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import csv
import json
from pathlib import Path

import numpy as np
import pandas as pd

from lin_mnp import model
from lin_mnp.paths import CACHE_ROOT, RESULTS_ROOT

FOLDS = Path(model.CONFIG["folds"])
BASELINE_RESULTS = RESULTS_ROOT / "cscc/folds"
OUTPUT = RESULTS_ROOT / "neighbor_controls/cscc"
K_PER_OTHER = model.NEIGHBORS_PER_OTHER_SECTION
K_MATCHED = K_PER_OTHER * 2  # Every cSCC patient has three sections.
ARMS = (
    "identity",
    "within_section_borda_k64",
    "within_section_borda_k128",
    "within_section_coord_k128",
    "within_section_random_k128",
    "cross_section_random_k128",
    "cross_section_lin_mnp",
)


def write_json(payload: dict, path: Path) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def per_gene_pcc(target: np.ndarray, prediction: np.ndarray) -> np.ndarray:
    left = np.asarray(target, dtype=np.float64)
    right = np.asarray(prediction, dtype=np.float64)
    left -= left.mean(axis=0, keepdims=True)
    right -= right.mean(axis=0, keepdims=True)
    denominator = np.sqrt(np.square(left).sum(0) * np.square(right).sum(0))
    return np.divide(
        (left * right).sum(0),
        denominator,
        out=np.zeros(left.shape[1], dtype=np.float64),
        where=denominator > 1e-12,
    )


def load_coordinates(indices: np.ndarray) -> np.ndarray:
    raw_index = CACHE_ROOT / "cscc/raw_patches/index.csv"
    by_global = {}
    with raw_index.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            by_global[int(row["global_index"])] = (
                float(row["center_x"]), float(row["center_y"])
            )
    return np.asarray([by_global[int(i)] for i in indices], dtype=np.float64)


def graph_from_blocks(n: int, blocks: list[tuple[np.ndarray, np.ndarray, np.ndarray]]):
    """Assemble a row-stochastic graph from (query_idx, neighbor_idx, weight) blocks."""
    width = max(item[1].shape[1] for item in blocks)
    index = np.zeros((n, width), dtype=np.int64)
    weight = np.zeros((n, width), dtype=np.float32)
    for query, neigh, w in blocks:
        index[query, :neigh.shape[1]] = neigh
        weight[query, :neigh.shape[1]] = w
    if not np.allclose(weight.sum(1), 1.0, atol=2e-6):
        raise ValueError("graph is not row stochastic")
    return {"index": index, "weight": weight}


def within_section_graph(sections, similarity, k, rng=None, uniform=False):
    sections = np.asarray(sections).astype(str)
    blocks = []
    for section in dict.fromkeys(sections):
        query = np.flatnonzero(sections == section)
        count = min(int(k), len(query) - 1)
        if count < 1:
            raise ValueError(f"section {section} has fewer than 2 spots")
        if rng is not None:
            neigh = np.empty((len(query), count), dtype=np.int64)
            for local, global_i in enumerate(query):
                others = np.delete(query, local)
                neigh[local] = rng.choice(others, size=count, replace=False)
            score = np.zeros((len(query), count), dtype=np.float32)
        else:
            block = similarity[np.ix_(query, query)].copy()
            block[np.arange(len(query)), np.arange(len(query))] = -np.inf
            selected = model.top(block, count)
            neigh = query[selected]
            score = np.take_along_axis(block, selected, axis=1)
        if uniform:
            w = np.full((len(query), count), 1.0 / count, dtype=np.float32)
        else:
            w = model.softmax(score)
        blocks.append((query, neigh, w))
    return graph_from_blocks(len(sections), blocks)


def borda_matrix(context, local):
    context = model.normalize_rows(context)
    local = model.normalize_rows(local)
    n = len(context)
    # pair_similarity expects query/gallery index arrays.
    all_idx = np.arange(n)
    return model.pair_similarity(context, local, all_idx, all_idx)


def coord_similarity(coordinates):
    delta = coordinates[:, None, :] - coordinates[None, :, :]
    distance = np.sqrt(np.square(delta).sum(-1))
    # Rank-to-[0,1] map, identical to Lin-MNP Borda after the two ranks are averaged.
    order = np.argsort(distance, axis=1, kind="stable")
    rank = np.empty_like(order, dtype=np.int32)
    np.put_along_axis(rank, order, np.arange(len(coordinates), dtype=np.int32)[None, :], axis=1)
    denominator = float(max(len(coordinates) - 1, 1))
    return 1.0 - rank.astype(np.float32) / denominator


def cross_section_random_graph(sections, k_per_other, rng):
    sections = np.asarray(sections).astype(str)
    section_ids = tuple(dict.fromkeys(sections))
    if len(section_ids) < 2:
        raise ValueError("cross-section random requires multiple sections")
    width = int(k_per_other) * (len(section_ids) - 1)
    index = np.empty((len(sections), width), dtype=np.int64)
    weight = np.empty((len(sections), width), dtype=np.float32)
    for query_section in section_ids:
        query = np.flatnonzero(sections == query_section)
        offset = 0
        for gallery_section in section_ids:
            if gallery_section == query_section:
                continue
            gallery = np.flatnonzero(sections == gallery_section)
            k = min(int(k_per_other), len(gallery))
            chosen = np.empty((len(query), k), dtype=np.int64)
            for i in range(len(query)):
                chosen[i] = rng.choice(gallery, size=k, replace=False)
            block = slice(offset, offset + k)
            index[query, block] = chosen
            weight[query, block] = 1.0 / (k * (len(section_ids) - 1))
            offset += k
    if not np.allclose(weight.sum(1), 1.0, atol=2e-6):
        raise ValueError("cross-section random graph is not row stochastic")
    return {"index": index, "weight": weight}


def apply_identity(prediction, *_):
    return np.asarray(prediction, dtype=np.float32)


def train_fold(args):
    output = args.output / f"fold{args.fold}"
    if output.exists():
        raise SystemExit(f"refusing to overwrite: {output}")
    source = BASELINE_RESULTS / f"fold{args.fold}/outer_test_predictions.npz"
    if not source.is_file():
        raise SystemExit(f"missing BASELINE_RESULTS prediction: {source}")

    fold = next(row for row in json.loads(FOLDS.read_text())["folds"]
                if int(row["fold_id"]) == args.fold)
    dataset = model.InferenceSpotDataset(model.CONFIG, fold["test_sample_ids"])
    indices = np.asarray(dataset.patch_indices, dtype=np.int64)
    local_cache = model.load_local_cache(model.CONFIG)
    context = dataset.context_array()
    local = np.asarray(local_cache[indices], dtype=np.float32)
    sections = np.asarray(dataset.sections)
    barcodes = tuple(dataset.barcodes)
    with np.load(source, allow_pickle=False) as archive:
        if model.decode(archive["sections"]) != tuple(sections):
            raise ValueError("BASELINE_RESULTS section mismatch")
        if model.decode(archive["barcodes"]) != barcodes:
            raise ValueError("BASELINE_RESULTS barcode mismatch")
        spot = archive["pointwise"].astype(np.float32)
        genes = model.decode(archive["genes"])

    coordinates = load_coordinates(indices)
    rng = np.random.default_rng(1741 + args.fold)
    borda = borda_matrix(context, local)
    coord = coord_similarity(coordinates)

    graphs = {
        "identity": None,
        "within_section_borda_k64": within_section_graph(sections, borda, K_PER_OTHER),
        "within_section_borda_k128": within_section_graph(sections, borda, K_MATCHED),
        "within_section_coord_k128": within_section_graph(sections, coord, K_MATCHED),
        "within_section_random_k128": within_section_graph(
            sections, borda, K_MATCHED, rng=rng, uniform=True
        ),
        "cross_section_random_k128": cross_section_random_graph(
            sections, K_PER_OTHER, rng
        ),
        "cross_section_lin_mnp": model.build_section_balanced_graph(
            sections, context, local
        ),
    }

    predictions = {}
    for name, graph in graphs.items():
        predictions[name] = spot if graph is None else model.diffuse(spot, graph)

    # The released operator must reproduce the archived BASELINE_RESULTS final prediction.
    with np.load(source, allow_pickle=False) as archive:
        reference = archive["lin_mnp"].astype(np.float64)
    recovered = float(np.max(np.abs(
        predictions["cross_section_lin_mnp"].astype(np.float64) - reference
    )))

    output.mkdir(parents=True)
    prediction_path = output / "outer_test_predictions.npz"
    np.savez_compressed(
        prediction_path,
        sections=sections, barcodes=np.asarray(barcodes), genes=np.asarray(genes),
        **{name: value.astype(np.float32) for name, value in predictions.items()},
    )
    metadata = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "experiment": "within-section smoothing control for Lin-MNP propagation",
        "fold": fold,
        "readout": "reference Lin-MNP pointwise patient-balanced Ridge",
        "pointwise_source": str(source),
        "pointwise_source_sha256": model.sha256(source),
        "neighbor_protocol": {
            "k_per_other_section": K_PER_OTHER,
            "k_matched_total": K_MATCHED,
            "inverse_temperature": model.INVERSE_TEMPERATURE,
            "every_patient_has_three_sections": True,
        },
        "cross_section_lin_mnp_vs_baseline_max_abs": recovered,
        "prediction_sha256": model.sha256(prediction_path),
        "prediction_array_sha256": {
            name: model.array_sha256(value) for name, value in predictions.items()
        },
        "outer_test_expression_opened": False,
        "labels_used_only_to_load_baseline_spot_predictions": False,
    }
    metadata_path = output / "run_metadata.json"
    write_json(metadata, metadata_path)
    write_json(
        {
            "status": "COMPLETE",
            "prediction_sha256": model.sha256(prediction_path),
            "run_metadata_sha256": model.sha256(metadata_path),
        },
        output / "complete.json",
    )
    print(json.dumps({
        "fold": args.fold,
        "cross_section_vs_baseline_max_abs": recovered,
        "prediction_sha256": metadata["prediction_sha256"][:16],
    }), flush=True)


def evaluate(args):
    records = []
    for fold_id in range(4):
        root = args.output / f"fold{fold_id}"
        complete = json.loads((root / "complete.json").read_text())
        metadata_path = root / "run_metadata.json"
        prediction_path = root / "outer_test_predictions.npz"
        if model.sha256(metadata_path) != complete["run_metadata_sha256"]:
            raise ValueError("metadata checksum failure")
        if model.sha256(prediction_path) != complete["prediction_sha256"]:
            raise ValueError("prediction checksum failure")
        metadata = json.loads(metadata_path.read_text())
        if metadata["outer_test_expression_opened"]:
            raise ValueError("outer test opened during graph construction")
        records.append((fold_id, root, metadata))

    rows = []
    recoveries = []
    for fold_id, root, metadata in records:
        recoveries.append(metadata["cross_section_lin_mnp_vs_baseline_max_abs"])
        fold = metadata["fold"]
        target = model.LabeledSpotDataset(model.CONFIG, fold["test_sample_ids"])
        with np.load(root / "outer_test_predictions.npz", allow_pickle=False) as archive:
            forbidden = {"target", "expression", "truth"} & set(archive.files)
            if forbidden:
                raise ValueError(f"prediction contains labels: {forbidden}")
            if model.decode(archive["sections"]) != tuple(target.sections):
                raise ValueError("evaluation section mismatch")
            if model.decode(archive["barcodes"]) != tuple(target.barcodes):
                raise ValueError("evaluation barcode mismatch")
            for arm in ARMS:
                prediction = archive[arm].astype(np.float32)
                section_array = np.asarray(target.sections)
                for section in dict.fromkeys(target.sections):
                    active = section_array == section
                    pcc = per_gene_pcc(target.target_matrix[active], prediction[active])
                    for gene, value in zip(target.genes, pcc):
                        rows.append({
                            "fold_id": fold_id, "patient_id": fold["test_patient_id"],
                            "arm": arm, "gene": gene, "pcc": float(value),
                        })
    detail = pd.DataFrame(rows)
    patient_gene = detail.groupby(["arm", "patient_id", "gene"], as_index=False).pcc.mean()
    gene = patient_gene.groupby(["arm", "gene"], as_index=False).pcc.mean()
    summary = gene.groupby("arm", as_index=False).pcc.mean()
    identity = float(summary.loc[summary.arm == "identity", "pcc"].iloc[0])
    summary["delta_vs_identity"] = summary.pcc - identity
    patient = patient_gene.groupby(["arm", "patient_id"], as_index=False).pcc.mean()

    output = args.output / "evaluation"
    output.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output / "summary.csv", index=False)
    patient.to_csv(output / "per_patient.csv", index=False)
    gene.to_csv(output / "per_gene.csv.gz", index=False)
    write_json({
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "aggregation": "section-gene -> patient-gene -> equal four-patient gene -> panel",
        "lin_mnp_recovery_max_abs_per_fold": recoveries,
        "outer_test_used_for_evaluation_only": True,
    }, output / "metadata.json")

    order = list(ARMS)
    print(summary.set_index("arm").loc[order].to_string())
    print()
    print(patient.pivot(index="arm", columns="patient_id", values="pcc").loc[order].to_string())
    print()
    print("Lin-MNP recovery max-abs per fold:", recoveries)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    train = sub.add_parser("train-fold")
    train.add_argument("--fold", type=int, choices=range(4), required=True)
    train.add_argument("--output", type=Path, default=OUTPUT)
    evaluate_parser = sub.add_parser("evaluate")
    evaluate_parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    args.output = args.output.resolve()
    if args.action == "train-fold":
        train_fold(args)
    else:
        evaluate(args)


if __name__ == "__main__":
    main()
