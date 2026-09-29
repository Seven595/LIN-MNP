#!/usr/bin/env python3
"""Apply the frozen Lin-MNP propagation step to comparator predictions.

The graph, neighbour count, inverse temperature and rank fusion are imported
from Lin-MNP without modification.  For each cSCC patient-CV4 outer-test patient
the graph is built once from the held-out patient's own frozen CTransPath
context and local descriptors, then applied to (a) the pointwise ridge readout,
reproducing the reported 0.421 -> 0.457, and (b) every comparator method with
saved spot-level predictions on the same sections.

Every method is scored in its own native target space using the benchmark's own
target constructor, so the unadapted column reproduces the published strict
patient-CV4 leaderboard gene by gene.
The scoring target is identical before and after propagation. Only comparator
predictions are standardized within section before the MNP step; this does not
change the measured-expression target or refit the predictor.

Usage:
  python -m lin_mnp.analysis.evaluate_comparator_transfer
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import wilcoxon

from lin_mnp import model
from lin_mnp.benchmark import benchmark_external as benchmark
from lin_mnp.paths import ARCHIVE_ROOT, RESULTS_ROOT

FROZEN_RESULT_ROOT = RESULTS_ROOT / "cscc/folds"
COMPARATOR_INDEX = (
    ARCHIVE_ROOT / "detail/01_benchmark/comparators_historical"
    / "benchmark_prediction_artifact_index.csv"
)
OFFICIAL_PER_PATIENT = (
    ARCHIVE_ROOT / "benchmark_strict"
    / "per_patient_gene_metrics.csv.gz"
)
READOUT_LABEL = "Ridge readout"


def gene_pcc(target, prediction):
    left = np.asarray(target, dtype=np.float64)
    right = np.asarray(prediction, dtype=np.float64)
    left = left - left.mean(0, keepdims=True)
    right = right - right.mean(0, keepdims=True)
    denominator = np.sqrt(np.square(left).sum(0) * np.square(right).sum(0))
    return np.divide(
        (left * right).sum(0), denominator,
        out=np.zeros(left.shape[1]), where=denominator > 1e-12,
    )


def patient_gene_pcc(targets, predictions):
    """Equal-weight mean of within-section gene-wise PCC over one patient."""
    return np.mean(
        np.stack([gene_pcc(t, p) for t, p in zip(targets, predictions)]), axis=0
    )


def standardise(block):
    centre = block.mean(0, keepdims=True)
    scale = block.std(0, keepdims=True)
    return (block - centre) / np.where(scale > 1e-8, scale, 1.0)


def split_by_section(matrix, section_labels, sections):
    return [np.asarray(matrix[section_labels == section]) for section in sections]


def load_comparator_index():
    with COMPARATOR_INDEX.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    by_method = {}
    for row in rows:
        entry = by_method.setdefault(row["method"], {})
        entry[row["slide_id"]] = Path(row["prediction_path"])
    return by_method


def read_prediction(path, barcodes, genes):
    with np.load(path, allow_pickle=False) as payload:
        prediction = np.asarray(payload["prediction"], dtype=np.float32)
        file_barcodes = [str(item) for item in payload["barcodes"]]
        file_genes = [str(item).casefold() for item in payload["genes"]]
    wanted_genes = [gene.casefold() for gene in genes]
    if sorted(file_genes) != sorted(wanted_genes) or sorted(file_barcodes) != sorted(barcodes):
        raise ValueError(f"identity mismatch for {path}")
    barcode_position = {code: i for i, code in enumerate(file_barcodes)}
    gene_position = {gene: i for i, gene in enumerate(file_genes)}
    aligned = prediction[np.ix_(
        np.asarray([barcode_position[code] for code in barcodes]),
        np.asarray([gene_position[gene] for gene in wanted_genes]),
    )]
    if not np.isfinite(aligned).all():
        raise ValueError(f"non-finite prediction in {path}")
    return aligned


def native_target_parameters(prediction_path, slides, train_ids):
    """Rebuild the fold's native target parameters from the rerun metadata."""
    metadata = json.loads(
        (prediction_path.parent.parent / "run_metadata.json").read_text(encoding="utf-8")
    )
    stored = dict(metadata["target_parameters"])
    kind = str(stored["kind"])
    parameters = {"kind": kind}
    for key in ("mean", "scale", "minimum", "maximum"):
        if key in stored:
            parameters[key] = np.asarray(stored[key], dtype=np.float32)
    if len(parameters) == 1 and kind not in {
        "raw_counts", "log1p_raw", "log1p_cp10k", "log1p_cpm",
        "igidl_log_pseudocount_cpm",
    }:
        parameters = benchmark.fit_target_transform(slides, train_ids, kind)
    return parameters


def run_fold(fold, comparators, local_cache, slides):
    fold_id = int(fold["fold_id"])
    sections = list(fold["test_sample_ids"])
    patient = str(fold["test_patient_id"])
    train_ids = list(fold["train_sample_ids"])

    dataset = model.LabeledSpotDataset(model.CONFIG, sections)
    patch_index = np.asarray(dataset.patch_indices, dtype=np.int64)
    context = np.asarray(dataset.ctranspath[patch_index], dtype=np.float32)
    local = np.asarray(local_cache[patch_index], dtype=np.float32)
    section_labels = np.asarray([str(item) for item in dataset.sections])
    genes = list(dataset.genes)
    barcodes = [str(item) for item in dataset.barcodes]

    for section in sections:
        slide_barcodes = [str(item) for item in slides[section].barcodes]
        mine = [code for code, label in zip(barcodes, section_labels) if label == section]
        if slide_barcodes != mine:
            raise ValueError(f"barcode order mismatch for {section}")
        if [g.casefold() for g in slides[section].genes] != [g.casefold() for g in genes]:
            raise ValueError(f"gene order mismatch for {section}")

    graph = model.build_section_balanced_graph(section_labels, context, local)

    checkpoint = torch.load(
        FROZEN_RESULT_ROOT / f"fold{fold_id}/checkpoint.pt",
        map_location="cpu",
        weights_only=False,
    )
    readout = model.predict_ridge(context, checkpoint["shared_artifact"])
    adapted = model.diffuse(readout, graph)

    with np.load(
        FROZEN_RESULT_ROOT / f"fold{fold_id}/outer_test_predictions.npz",
        allow_pickle=False,
    ) as payload:
        key = next(
            (name for name in ("lin_mnp", "prediction", "candidate") if name in payload.files),
            None,
        )
        if key is None:
            raise ValueError(f"no prediction matrix in {path}")
        saved = np.asarray(payload[key], dtype=np.float32)
        saved_sections = [str(item) for item in payload["sections"]]
        saved_barcodes = [str(item) for item in payload["barcodes"]]
    if saved_sections != list(section_labels) or saved_barcodes != barcodes:
        raise ValueError("reproduction alignment failure")
    reproduction_error = float(np.abs(saved - adapted).max())

    # Lin-MNP is fitted on section z-scored log1p raw counts, a per-section,
    # per-gene affine map of the official target, so within-section gene-wise
    # PCC against log1p raw counts is exact.
    official_targets = [
        np.asarray(slides[section].target("log1p_raw"), dtype=np.float32)
        for section in sections
    ]
    records = [{
        "method": READOUT_LABEL,
        "native_space": "section z-score of log1p raw counts",
        "unadapted": patient_gene_pcc(
            official_targets, split_by_section(readout, section_labels, sections)
        ),
        "adapted": patient_gene_pcc(
            official_targets, split_by_section(adapted, section_labels, sections)
        ),
        "adapted_unscaled": patient_gene_pcc(
            official_targets, split_by_section(adapted, section_labels, sections)
        ),
    }]

    for method, paths in sorted(comparators.items()):
        if not all(section in paths for section in sections):
            continue
        parameters = native_target_parameters(paths[sections[0]], slides, train_ids)
        native_targets = [
            np.asarray(benchmark.apply_target_transform(slides[section], parameters), dtype=np.float32)
            for section in sections
        ]
        blocks = [
            read_prediction(
                paths[section],
                [code for code, label in zip(barcodes, section_labels) if label == section],
                genes,
            )
            for section in sections
        ]
        prediction = np.concatenate(blocks, axis=0)
        scaled = np.concatenate([standardise(block) for block in blocks], axis=0)
        smoothed = model.diffuse(scaled, graph)
        smoothed_unscaled = model.diffuse(prediction, graph)
        records.append({
            "method": method,
            "native_space": str(parameters["kind"]),
            "unadapted": patient_gene_pcc(native_targets, blocks),
            "adapted": patient_gene_pcc(
                native_targets, split_by_section(smoothed, section_labels, sections)
            ),
            "adapted_unscaled": patient_gene_pcc(
                native_targets, split_by_section(smoothed_unscaled, section_labels, sections)
            ),
        })

    return {
        "fold_id": fold_id,
        "patient": patient,
        "sections": sections,
        "reproduction_max_abs_error": reproduction_error,
        "records": records,
    }


def audit_against_official(fold_results):
    """Compare the unadapted column with the published strict-CV4 gene table."""
    official = pd.read_csv(OFFICIAL_PER_PATIENT)
    report = {}
    for fold in fold_results:
        for record in fold["records"]:
            if record["method"] == READOUT_LABEL:
                continue
            subset = official[
                (official["method"] == record["method"])
                & (official["patient_id"] == fold["patient"])
            ]
            if subset.empty:
                report[record["method"]] = None
                continue
            published = subset.sort_values("gene")["pcc"].to_numpy(dtype=np.float64)
            if published.shape != record["unadapted"].shape:
                report[record["method"]] = None
                continue
            error = float(np.abs(published.mean() - record["unadapted"].mean()))
            report[record["method"]] = max(report.get(record["method"]) or 0.0, error)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=RESULTS_ROOT / "comparator_transfer")
    arguments = parser.parse_args()
    output = arguments.output
    output.mkdir(parents=True, exist_ok=True)

    folds = json.loads(Path(model.CONFIG["folds"]).read_text())["folds"]
    comparators = load_comparator_index()
    local_cache = model.load_local_cache(model.CONFIG)
    slides, _ = benchmark.load_slides()

    fold_results = [run_fold(fold, comparators, local_cache, slides) for fold in folds]

    covered = [
        record["method"] for record in fold_results[0]["records"]
        if all(
            any(item["method"] == record["method"] for item in fold["records"])
            for fold in fold_results
        )
    ]
    dropped = sorted(set(comparators) - set(covered))
    baseline_audit = audit_against_official(fold_results)

    summary = []
    for method in covered:
        columns = {"unadapted": [], "adapted": [], "adapted_unscaled": []}
        per_patient = {}
        native = ""
        for fold in fold_results:
            record = next(r for r in fold["records"] if r["method"] == method)
            native = record["native_space"]
            for key in columns:
                columns[key].append(record[key])
            per_patient[fold["patient"]] = {
                "unadapted": float(record["unadapted"].mean()),
                "adapted": float(record["adapted"].mean()),
                "delta": float((record["adapted"] - record["unadapted"]).mean()),
            }
        gene = {
            key: np.mean(np.stack(value), axis=0) for key, value in columns.items()
        }
        panel = {key: float(value.mean()) for key, value in gene.items()}
        gene_delta = gene["adapted"] - gene["unadapted"]
        summary.append({
            "method": method,
            "native_target_space": native,
            "unadapted_pcc": panel["unadapted"],
            "adapted_pcc": panel["adapted"],
            "delta": panel["adapted"] - panel["unadapted"],
            "genes_improved_of_997": int(np.count_nonzero(gene_delta > 0)),
            "gene_wilcoxon_p": float(
                wilcoxon(gene["adapted"], gene["unadapted"]).pvalue
            ),
            "adapted_pcc_without_section_scaling": panel["adapted_unscaled"],
            "patients_improved": sum(
                1 for value in per_patient.values() if value["delta"] > 0
            ),
            "published_baseline_max_abs_error": baseline_audit.get(method),
            "per_patient": per_patient,
        })

    payload = {
        "schema_version": 1,
        "protocol": "cSCC fixed-997 strict patient CV4, four held-out patients",
        "scoring": "each method in its own native target space, section-gene -> "
                   "patient-gene -> equal four-patient gene -> panel",
        "operator": {
            "neighbours_per_other_section": model.NEIGHBORS_PER_OTHER_SECTION,
            "inverse_temperature": model.INVERSE_TEMPERATURE,
            "depth": model.DIFFUSION_DEPTH,
            "rank_fusion": "equal Borda of context and local ranks",
            "graph_source": "held-out patient's own frozen CTransPath descriptors",
            "training_free": True,
        },
        "reproduction_max_abs_error": max(
            fold["reproduction_max_abs_error"] for fold in fold_results
        ),
        "methods_without_full_section_coverage": dropped,
        "summary": sorted(summary, key=lambda item: -item["adapted_pcc"]),
    }
    comparator_deltas = [
        item["delta"] for item in summary if item["method"] != READOUT_LABEL
    ]
    payload["comparator_overview"] = {
        "methods": len(comparator_deltas),
        "improved": int(sum(1 for value in comparator_deltas if value > 0)),
        "median_delta": float(np.median(comparator_deltas)),
        "min_delta": float(np.min(comparator_deltas)),
        "max_delta": float(np.max(comparator_deltas)),
        "readout_delta": next(
            item["delta"] for item in summary if item["method"] == READOUT_LABEL
        ),
    }
    (output / "summary.json").write_text(json.dumps(payload, indent=1), encoding="utf-8")

    with (output / "summary.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow([
            "method", "native_target_space", "unadapted_pcc", "adapted_pcc", "delta",
            "patients_improved_of_4", "genes_improved_of_997", "gene_wilcoxon_p",
            "adapted_pcc_without_section_scaling",
            "published_baseline_max_abs_error",
        ])
        for item in payload["summary"]:
            writer.writerow([
                item["method"], item["native_target_space"],
                f"{item['unadapted_pcc']:.6f}", f"{item['adapted_pcc']:.6f}",
                f"{item['delta']:+.6f}", item["patients_improved"],
                item["genes_improved_of_997"], f"{item['gene_wilcoxon_p']:.3e}",
                f"{item['adapted_pcc_without_section_scaling']:.6f}",
                "" if item["published_baseline_max_abs_error"] is None
                else f"{item['published_baseline_max_abs_error']:.2e}",
            ])

    print(f"Lin-MNP prediction reproduction max abs error: "
          f"{payload['reproduction_max_abs_error']:.3e}")
    print(f"methods dropped for section coverage: {dropped}")
    print(f"{'method':14s} {'base':>7s} {'+adapt':>7s} {'delta':>8s} "
          f"{'up/4':>5s} {'genes':>6s} {'wilcoxon':>10s} {'vs pub':>8s}")
    for item in payload["summary"]:
        error = item["published_baseline_max_abs_error"]
        print(
            f"{item['method']:14s} {item['unadapted_pcc']:7.4f} "
            f"{item['adapted_pcc']:7.4f} {item['delta']:+8.4f} "
            f"{item['patients_improved']:5d} {item['genes_improved_of_997']:6d} "
            f"{item['gene_wilcoxon_p']:10.2e} "
            f"{'n/a' if error is None else f'{error:.0e}':>8s}"
        )
    print(json.dumps(payload["comparator_overview"], indent=1))


if __name__ == "__main__":
    main()
