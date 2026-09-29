#!/usr/bin/env python3
"""Analyze which descriptor view carries the propagation gain on cSCC.

The readout uses the contextual descriptor alone, whereas the Lin-MNP graph fuses
contextual and local ranks.  Part of the reported increment could therefore come
from the local view entering the model for the first time rather than from
relations between spots.  This script rebuilds the same graph from one view at a
time, holding the neighbour count, inverse temperature, gallery and single step
fixed, and rescores the identical pointwise readout.

Usage:
  python -m lin_mnp.analysis.analyze_graph_views
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from lin_mnp import model
from lin_mnp.analysis import evaluate_comparator_transfer as comparator_transfer
from lin_mnp.benchmark import benchmark_external as benchmark
from lin_mnp.paths import RESULTS_ROOT

VIEWS = ("dual (Lin-MNP)", "contextual only", "local only")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=RESULTS_ROOT / "graph_views")
    arguments = parser.parse_args()
    output = arguments.output
    output.mkdir(parents=True, exist_ok=True)

    folds = json.loads(Path(model.CONFIG["folds"]).read_text())["folds"]
    local_cache = model.load_local_cache(model.CONFIG)
    slides, _ = benchmark.load_slides()

    per_patient = {view: {} for view in VIEWS}
    base_per_patient = {}

    for fold in folds:
        fold_id = int(fold["fold_id"])
        sections = list(fold["test_sample_ids"])
        patient = str(fold["test_patient_id"])

        dataset = model.LabeledSpotDataset(model.CONFIG, sections)
        patch_index = np.asarray(dataset.patch_indices, dtype=np.int64)
        context = np.asarray(dataset.ctranspath[patch_index], dtype=np.float32)
        local = np.asarray(local_cache[patch_index], dtype=np.float32)
        section_labels = np.asarray([str(item) for item in dataset.sections])

        checkpoint = torch.load(
            comparator_transfer.FROZEN_RESULT_ROOT / f"fold{fold_id}/checkpoint.pt",
            map_location="cpu",
            weights_only=False,
        )
        readout = model.predict_ridge(context, checkpoint["shared_artifact"])
        targets = [
            np.asarray(slides[section].target("log1p_raw"), dtype=np.float32)
            for section in sections
        ]

        def score(matrix):
            return float(comparator_transfer.patient_gene_pcc(
                targets,
                comparator_transfer.split_by_section(matrix, section_labels, sections),
            ).mean())

        base_per_patient[patient] = score(readout)
        # Passing the same array as both views makes the Borda fusion collapse to
        # that view's ranking, so the graph is built from one descriptor only.
        for view, pair in zip(VIEWS, ((context, local), (context, context), (local, local))):
            graph = model.build_section_balanced_graph(
                section_labels, pair[0], pair[1]
            )
            per_patient[view][patient] = score(model.diffuse(readout, graph))
        print("fold %s done" % patient, flush=True)

    patients = list(base_per_patient)
    base = float(np.mean([base_per_patient[p] for p in patients]))
    rng = np.random.default_rng(17)
    rows = []
    for view in VIEWS:
        deltas = np.array([per_patient[view][p] - base_per_patient[p] for p in patients])
        boot = rng.choice(deltas, size=(2000, len(patients)), replace=True).mean(axis=1)
        low, high = np.percentile(boot, [2.5, 97.5])
        rows.append({
            "graph_view": view,
            "pcc": float(np.mean([per_patient[view][p] for p in patients])),
            "delta": float(deltas.mean()),
            "patients_improved": int((deltas > 0).sum()),
            "interval_low": float(low),
            "interval_high": float(high),
            "per_patient_delta": {p: float(d) for p, d in zip(patients, deltas)},
        })

    payload = {
        "protocol": "cSCC strict patient CV4, outer test",
        "pointwise_readout_pcc": base,
        "note": "graph parameters, gallery and single step identical across rows",
        "table": rows,
    }
    (output / "summary.json").write_text(json.dumps(payload, indent=1), encoding="utf-8")

    print("\npointwise readout PCC = %.4f" % base)
    print("%-18s %7s %8s %5s  %s" % ("graph view", "PCC", "delta", "n+", "95% interval"))
    for r in rows:
        print("%-18s %7.4f %+8.4f %5d  [%+.3f, %+.3f]" % (
            r["graph_view"], r["pcc"], r["delta"], r["patients_improved"],
            r["interval_low"], r["interval_high"]))


if __name__ == "__main__":
    main()
