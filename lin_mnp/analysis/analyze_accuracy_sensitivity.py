#!/usr/bin/env python3
"""Analyze how propagation gain depends on pointwise accuracy.

Real predictors only reach PCC 0.42 on cSCC, so the archived comparisons cannot
say whether the operator would still help an accurate pointwise predictor.  This
script sweeps synthetic pointwise predictors of controlled accuracy, built from
the held-out target itself, and applies the unchanged Lin-MNP operator to each.

Two noise models bracket the real case:

* independent noise per spot, which averaging removes most efficiently and which
  therefore overstates the benefit of propagation; and
* noise pre-smoothed over the same morphology graph, standing for the
  morphology-correlated error an image-derived predictor actually makes, which
  understates it.

The synthetic predictors use held-out expression by construction.  They are a
diagnostic of the operator, not a prediction method.

Usage:
  python -m lin_mnp.analysis.analyze_accuracy_sensitivity
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from lin_mnp import model
from lin_mnp.benchmark import benchmark_external as benchmark
from lin_mnp.paths import RESULTS_ROOT

TARGETS = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99)
REPEATS = 5
SEED = 20260914


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


def standardise_sections(matrix, section_labels, sections):
    value = np.array(matrix, dtype=np.float32)
    for section in sections:
        active = section_labels == section
        block = value[active]
        scale = block.std(0, keepdims=True)
        value[active] = (block - block.mean(0, keepdims=True)) / np.where(
            scale > 1e-8, scale, 1.0
        )
    return value


def build_fold(fold, local_cache, slides):
    sections = list(fold["test_sample_ids"])
    dataset = model.LabeledSpotDataset(model.CONFIG, sections)
    patch_index = np.asarray(dataset.patch_indices, dtype=np.int64)
    context = np.asarray(dataset.ctranspath[patch_index], dtype=np.float32)
    local = np.asarray(local_cache[patch_index], dtype=np.float32)
    section_labels = np.asarray([str(item) for item in dataset.sections])
    graph = model.build_section_balanced_graph(section_labels, context, local)
    truth = np.concatenate(
        [np.asarray(slides[s].target("log1p_raw"), dtype=np.float32) for s in sections],
        axis=0,
    )
    return {
        "patient": str(fold["test_patient_id"]),
        "sections": sections,
        "section_labels": section_labels,
        "graph": graph,
        "truth": truth,
        "standard_truth": standardise_sections(truth, section_labels, sections),
    }


def score(fold, prediction):
    """Equal-weight mean over sections of within-section gene-wise PCC."""
    labels = fold["section_labels"]
    return np.mean(
        np.stack([
            gene_pcc(fold["truth"][labels == s], prediction[labels == s])
            for s in fold["sections"]
        ]),
        axis=0,
    )


def synthesise(fold, level, noise, correlated):
    """Mix standardised truth with noise so that gene-wise PCC is about `level`."""
    if correlated:
        noise = model.diffuse(noise.astype(np.float32), fold["graph"])
    noise = standardise_sections(noise, fold["section_labels"], fold["sections"])
    mixed = level * fold["standard_truth"] + np.sqrt(1.0 - level ** 2) * noise
    return standardise_sections(mixed, fold["section_labels"], fold["sections"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=RESULTS_ROOT / "accuracy_sensitivity")
    arguments = parser.parse_args()
    output = arguments.output
    output.mkdir(parents=True, exist_ok=True)

    folds = json.loads(Path(model.CONFIG["folds"]).read_text())["folds"]
    local_cache = model.load_local_cache(model.CONFIG)
    slides, _ = benchmark.load_slides()
    prepared = [build_fold(fold, local_cache, slides) for fold in folds]

    generator = np.random.default_rng(SEED)
    rows = []
    for correlated in (False, True):
        for level in TARGETS:
            unadapted, adapted = [], []
            for repeat in range(REPEATS):
                gene_before, gene_after = [], []
                for fold in prepared:
                    noise = generator.standard_normal(fold["truth"].shape).astype(np.float32)
                    pointwise = synthesise(fold, level, noise, correlated)
                    gene_before.append(score(fold, pointwise))
                    gene_after.append(
                        score(fold, model.diffuse(pointwise, fold["graph"]))
                    )
                unadapted.append(np.mean(np.stack(gene_before), axis=0).mean())
                adapted.append(np.mean(np.stack(gene_after), axis=0).mean())
            rows.append({
                "noise": "morphology-correlated" if correlated else "independent",
                "requested_level": level,
                "unadapted_pcc": float(np.mean(unadapted)),
                "adapted_pcc": float(np.mean(adapted)),
                "delta": float(np.mean(adapted) - np.mean(unadapted)),
                "delta_sd_over_repeats": float(
                    np.std(np.asarray(adapted) - np.asarray(unadapted), ddof=1)
                ),
            })
            print(
                f"{rows[-1]['noise']:22s} level {level:4.2f} "
                f"base {rows[-1]['unadapted_pcc']:6.3f} "
                f"adapt {rows[-1]['adapted_pcc']:6.3f} "
                f"delta {rows[-1]['delta']:+7.4f}"
            )

    crossings = {}
    for noise in ("independent", "morphology-correlated"):
        series = [r for r in rows if r["noise"] == noise]
        crossing = None
        for previous, current in zip(series, series[1:]):
            if previous["delta"] > 0 >= current["delta"]:
                span = previous["delta"] - current["delta"]
                weight = previous["delta"] / span if span > 0 else 0.0
                crossing = (
                    previous["unadapted_pcc"]
                    + weight * (current["unadapted_pcc"] - previous["unadapted_pcc"])
                )
                break
        crossings[noise] = crossing

    payload = {
        "schema_version": 1,
        "protocol": "cSCC fixed-997 strict patient CV4, four held-out patients",
        "operator": "unchanged Lin-MNP: 64 neighbours per other section, inverse "
                    "temperature 10, equal Borda fusion, one step",
        "synthetic_predictors_use_heldout_expression": True,
        "repeats_per_point": REPEATS,
        "break_even_pointwise_pcc": crossings,
        "highest_real_pointwise_pcc": 0.420963,
        "sweep": rows,
    }
    (output / "summary.json").write_text(json.dumps(payload, indent=1), encoding="utf-8")
    with (output / "sweep.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    print()
    print(json.dumps(crossings, indent=1))


if __name__ == "__main__":
    main()
