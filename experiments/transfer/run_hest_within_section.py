#!/usr/bin/env python3
"""Within-section adapter applied to the HEST Lin-MNP runs.

On HEST the released spatial operator links only sections belonging to the same
official patient fold, so it degenerates to the identity whenever the test fold
holds a single section.  That is the common case: the selected `spatial_depth`
is 0 on 25 of 29 folds with CTransPath and 26 of 29 with UNI-v1.  The published
`Lin-MNP_full` aggregates are therefore almost entirely PFM + Ridge.

The cSCC neighbour control showed that a within-section morphology graph retains
the propagation gain while requiring only one section. This runner starts from
the archived pointwise Ridge predictions and changes only the propagation
operator.

Reported arms per encoder:

* ``published_lin_mnp`` -- the reported configuration, reproduced as an
  integrity check.
* ``ridge_only``    -- `Lin-MNP_no_spatial_transport`, i.e. the adapter removed.
* ``borda_k*`` / ``minimal_k*`` -- within-section adapter at several k.

The declared transfer arm is ``borda_k128``, the cSCC setting carried over
untuned.  Targets ship inside the archived prediction files, so no separate
label load is needed; the graphs use only cached features.
"""

from __future__ import annotations

import argparse
import collections
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np
import pandas as pd

from lin_mnp import model
from lin_mnp.within_section import (
    apply_grid,
    mean_gene_pcc_and_mse,
    radial_pool_weights,
)
from lin_mnp.paths import ARCHIVE_ROOT, CACHE_ROOT, RESULTS_ROOT

ARCHIVE = ARCHIVE_ROOT
ARCHIVED_RESULTS = ARCHIVE / "legacy/development"

OUTPUT = RESULTS_ROOT / "transfer/hest"

ENCODERS = {
    "ctranspath": {
        "results": ARCHIVED_RESULTS / "transfer/hest/ctranspath",
        "cache": CACHE_ROOT / "hest/ctranspath",
        "pooled": "pooled_768_float16.npy",
        "local": None,  # radial soft pool of the 7x7 token map
        "published": 0.357228,
    },
    "uni_v1": {
        "results": ARCHIVED_RESULTS / "transfer/hest/uni_v1",
        "cache": CACHE_ROOT / "hest/uni_v1",
        "pooled": "pooled_1024_float32.npy",
        "local": "local_1024_float32.npy",
        "published": 0.381026,
    },
}
CONFIGS = {
    "borda": {"fusion": "borda", "weighting": "softmax"},
    "minimal": {"fusion": "score", "weighting": "uniform"},
}
K_GRID = (8, 16, 32, 64, 128)
DECLARED_ARM = "borda_k128"


def slide_views(sample_id: str, spec: dict, weights: np.ndarray):
    root = spec["cache"] / sample_id
    with np.load(root / "metadata.npz", allow_pickle=False) as archive:
        barcodes = archive["barcodes"].astype(str)
    context = np.asarray(np.load(root / spec["pooled"], mmap_mode="r"), dtype=np.float32)
    if spec["local"] is not None:
        local = np.asarray(np.load(root / spec["local"], mmap_mode="r"), dtype=np.float32)
    else:
        spatial = np.load(root / "spatial_49x768_float16.npy", mmap_mode="r")
        local = np.empty_like(context)
        for start in range(0, len(spatial), 512):
            stop = min(start + 512, len(spatial))
            local[start:stop] = np.einsum(
                "t,ntd->nd", weights,
                np.asarray(spatial[start:stop], dtype=np.float32), optimize=True)
    return context, local, barcodes


def run(args):
    spec = ENCODERS[args.encoder]
    weights = radial_pool_weights()
    arms = ["published_lin_mnp", "ridge_only"] + [
        f"{n}_k{k}" for n in CONFIGS for k in K_GRID
    ]
    rows = []
    folds = sorted(spec["results"].glob("*/fold*/predictions.npz"))
    if len(folds) != 29:
        raise SystemExit(f"expected 29 folds, found {len(folds)}")

    for path in folds:
        dataset = path.parts[-3]
        fold = path.parts[-2]
        with np.load(path, allow_pickle=False) as archive:
            target = archive["target"].astype(np.float32)
            full = archive["lin_mnp"].astype(np.float32)
            ridge = archive["ridge"].astype(np.float32)
            sections = archive["sections"].astype(str)
            barcodes = archive["barcodes"].astype(str)

        adapted = {name: ridge.copy() for name in arms[2:]}
        for section in dict.fromkeys(sections):
            mask = sections == section
            context, local, cache_barcodes = slide_views(section, spec, weights)
            position = {code: i for i, code in enumerate(cache_barcodes)}
            order = np.asarray([position[code] for code in barcodes[mask]])
            for name, config in CONFIGS.items():
                produced = apply_grid(
                    ridge[mask], context[order], local[order], K_GRID, **config)
                for k, value in produced.items():
                    adapted[f"{name}_k{k}"][mask] = value

        scored = {"published_lin_mnp": full, "ridge_only": ridge} | adapted
        for arm, prediction in scored.items():
            rows.append({
                "encoder": args.encoder, "dataset": dataset, "fold": fold,
                    "arm": arm, **mean_gene_pcc_and_mse(target, prediction),
            })
        print(json.dumps({
            "dataset": dataset, "fold": fold, "sections": len(set(sections)),
            "spots": int(len(target)),
        }), flush=True)

    detail = pd.DataFrame(rows)
    per_dataset = detail.groupby(["arm", "dataset"], as_index=False)[["pcc", "mse"]].mean()
    summary = per_dataset.groupby("arm", as_index=False)[["pcc", "mse"]].mean()
    baseline = float(summary.loc[summary.arm == "published_lin_mnp", "pcc"].iloc[0])
    summary["delta_vs_published"] = summary.pcc - baseline

    out = args.output / args.encoder
    out.mkdir(parents=True, exist_ok=True)
    detail.to_csv(out / "per_fold.csv", index=False)
    per_dataset.to_csv(out / "per_dataset.csv", index=False)
    summary.to_csv(out / "summary.csv", index=False)
    (out / "metadata.json").write_text(json.dumps({
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "encoder": args.encoder,
        "aggregation": "fold mean within dataset, then equal 10-dataset mean",
        "configs": CONFIGS, "k_grid": list(K_GRID),
        "declared_transfer_arm": DECLARED_ARM,
        "inverse_temperature": model.INVERSE_TEMPERATURE,
        "published_lin_mnp_pcc": baseline,
        "published_pcc": spec["published"],
        "reproduction_abs_difference": abs(baseline - spec["published"]),
    }, indent=2) + "\n")

    print()
    print(summary.set_index("arm").loc[arms].to_string())
    print()
    print(f"published_lin_mnp {baseline:.6f} vs reported {spec['published']:.6f} "
          f"|diff| {abs(baseline - spec['published']):.6f}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--encoder", choices=tuple(ENCODERS), required=True)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    args.output = args.output.resolve()
    run(args)


if __name__ == "__main__":
    main()
