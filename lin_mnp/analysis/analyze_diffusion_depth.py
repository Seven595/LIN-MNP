#!/usr/bin/env python3
"""Analyze diffusion depth and residual structure on cSCC.

(a) Iterated propagation.  The one-step rule of Lin-MNP is applied repeatedly to
    the same graph, for the ridge readout and for every comparator with saved
    spot-level predictions, so the reported single step can be compared with two
    steps and with the fixed point.

(b) Residual structure.  For each predictor the residual against its own native
    target is measured for how far it is already smooth over the same graph,
    rho_res = gene-wise PCC(R, A R).  The question is whether rho_res predicts
    the propagation gain where pointwise accuracy does not.

Graph, neighbour count, inverse temperature and rank fusion are imported from
Lin-MNP without modification, and every method is scored in its own native target
space, exactly as in the comparator-transfer analysis.

Usage:
  python -m lin_mnp.analysis.analyze_diffusion_depth
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
from scipy.stats import spearmanr

from lin_mnp import model
from lin_mnp.analysis import evaluate_comparator_transfer as comparator_transfer
from lin_mnp.benchmark import benchmark_external as benchmark
from lin_mnp.paths import RESULTS_ROOT

FROZEN_RESULT_ROOT = comparator_transfer.FROZEN_RESULT_ROOT
READOUT_LABEL = comparator_transfer.READOUT_LABEL
STEPS = (1, 2, 3, 5, 10)


def gene_mean_pcc(targets, blocks):
    return comparator_transfer.patient_gene_pcc(targets, blocks).mean()


def residual_smoothness(residual, graph, section_labels, sections):
    """Gene-wise PCC between the residual and the residual averaged over the graph."""
    smoothed = model.diffuse(residual, graph)
    values = []
    for section in sections:
        mask = section_labels == section
        values.append(comparator_transfer.gene_pcc(residual[mask], smoothed[mask]))
    return float(np.nanmean(np.stack(values)))


def run_fold(fold, comparators, local_cache, slides):
    fold_id = int(fold["fold_id"])
    sections = list(fold["test_sample_ids"])
    train_ids = list(fold["train_sample_ids"])

    dataset = model.LabeledSpotDataset(model.CONFIG, sections)
    patch_index = np.asarray(dataset.patch_indices, dtype=np.int64)
    context = np.asarray(dataset.ctranspath[patch_index], dtype=np.float32)
    local = np.asarray(local_cache[patch_index], dtype=np.float32)
    section_labels = np.asarray([str(item) for item in dataset.sections])
    genes = list(dataset.genes)
    barcodes = [str(item) for item in dataset.barcodes]

    graph = model.build_section_balanced_graph(section_labels, context, local)

    def split(matrix):
        return comparator_transfer.split_by_section(matrix, section_labels, sections)

    def iterate(matrix, targets):
        out, current = {}, matrix
        for step in range(1, max(STEPS) + 1):
            current = model.diffuse(current, graph)
            if step in STEPS:
                out[step] = float(gene_mean_pcc(targets, split(current)))
        return out

    records = []

    checkpoint = torch.load(
        FROZEN_RESULT_ROOT / f"fold{fold_id}/checkpoint.pt",
        map_location="cpu",
        weights_only=False,
    )
    readout = model.predict_ridge(context, checkpoint["shared_artifact"])
    readout_targets = [
        np.asarray(slides[section].target("log1p_raw"), dtype=np.float32)
        for section in sections
    ]
    # Residual structure is measured on a common scale: both the prediction and
    # the target are standardised within each section before subtraction.
    standard_truth = np.concatenate(
        [comparator_transfer.standardise(block) for block in readout_targets], axis=0
    )
    standard_readout = np.concatenate(
        [comparator_transfer.standardise(block) for block in split(readout)], axis=0
    )
    records.append({
        "method": READOUT_LABEL,
        "base": float(gene_mean_pcc(readout_targets, split(readout))),
        "steps": iterate(readout, readout_targets),
        "rho_res": residual_smoothness(
            standard_readout - standard_truth, graph, section_labels, sections
        ),
    })

    for method, paths in sorted(comparators.items()):
        if not all(section in paths for section in sections):
            continue
        parameters = comparator_transfer.native_target_parameters(
            paths[sections[0]], slides, train_ids
        )
        native_targets = [
            np.asarray(
                benchmark.apply_target_transform(slides[section], parameters),
                dtype=np.float32,
            )
            for section in sections
        ]
        blocks = [
            comparator_transfer.read_prediction(
                paths[section],
                [c for c, label in zip(barcodes, section_labels) if label == section],
                genes,
            )
            for section in sections
        ]
        scaled = np.concatenate(
            [comparator_transfer.standardise(block) for block in blocks], axis=0
        )
        standard_native = np.concatenate(
            [comparator_transfer.standardise(block) for block in native_targets], axis=0
        )
        records.append({
            "method": method,
            "base": float(gene_mean_pcc(native_targets, blocks)),
            "steps": iterate(scaled, native_targets),
            "rho_res": residual_smoothness(
                scaled - standard_native, graph, section_labels, sections
            ),
        })

    return {"patient": str(fold["test_patient_id"]), "records": records}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, default=RESULTS_ROOT / "diffusion_depth"
    )
    arguments = parser.parse_args()
    output = arguments.output
    output.mkdir(parents=True, exist_ok=True)

    folds = json.loads(Path(model.CONFIG["folds"]).read_text())["folds"]
    comparators = comparator_transfer.load_comparator_index()
    local_cache = model.load_local_cache(model.CONFIG)
    slides, _ = benchmark.load_slides()

    fold_results = []
    for fold in folds:
        fold_results.append(run_fold(fold, comparators, local_cache, slides))
        print("fold %s done" % fold_results[-1]["patient"], flush=True)

    covered = [
        r["method"] for r in fold_results[0]["records"]
        if all(any(i["method"] == r["method"] for i in f["records"]) for f in fold_results)
    ]

    rows = []
    for method in covered:
        picked = [
            next(r for r in f["records"] if r["method"] == method) for f in fold_results
        ]
        row = {
            "method": method,
            "base": float(np.mean([p["base"] for p in picked])),
            "rho_res": float(np.mean([p["rho_res"] for p in picked])),
        }
        for step in STEPS:
            row["step%d" % step] = float(np.mean([p["steps"][step] for p in picked]))
        row["delta"] = row["step1"] - row["base"]
        row["delta_step2_minus_step1"] = row["step2"] - row["step1"]
        rows.append(row)

    rows.sort(key=lambda r: -r["step1"])
    comp = [r for r in rows if r["method"] != READOUT_LABEL]

    def corr(key):
        stat = spearmanr([r[key] for r in comp], [r["delta"] for r in comp])
        return {"spearman_rho": float(stat.statistic), "p_value": float(stat.pvalue)}

    payload = {
        "protocol": "cSCC strict patient CV4, outer test",
        "operator": "Lin-MNP graph, unchanged; propagation applied repeatedly",
        "steps_reported": list(STEPS),
        "n_predictors": len(rows),
        "delta_vs_base_accuracy": corr("base"),
        "delta_vs_residual_smoothness": corr("rho_res"),
        "table": rows,
    }
    (output / "summary.json").write_text(json.dumps(payload, indent=1), encoding="utf-8")
    with (output / "summary.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print("\nmethod            base   1step  2step  3step  5step 10step  rho_res")
    for r in rows:
        print("%-16s %6.3f %6.3f %6.3f %6.3f %6.3f %6.3f  %7.3f" % (
            r["method"][:16], r["base"], r["step1"], r["step2"], r["step3"],
            r["step5"], r["step10"], r["rho_res"]))
    print("\ndelta vs base accuracy   :", payload["delta_vs_base_accuracy"])
    print("delta vs residual smooth :", payload["delta_vs_residual_smoothness"])


if __name__ == "__main__":
    main()
