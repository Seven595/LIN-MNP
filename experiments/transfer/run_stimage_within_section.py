#!/usr/bin/env python3
"""Within-section adapter applied to the STImage Lin-MNP runs.

STImage supplies independent cancer slides with no serial-section patient
groups, so the released configuration switches the spatial operator off
outright (`"spatial_transport": {"enabled": false, "depth": 0}`) and the
published `Lin-MNP_full` is exactly the Ridge spot prediction.

The cSCC neighbour control showed that a within-section morphology graph retains
the propagation gain while staying inside one slide. That operator is applied
here to archived pointwise Ridge predictions without refitting them.

Encoder note.  UNI-v1 and CTransPath both supply a contextual descriptor and a
radial soft-pooled local view for all 146 slides, so both rows use dual-view
Borda ranks.
"""

from __future__ import annotations

import argparse
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

OUTPUT = RESULTS_ROOT / "transfer/stimage"
TISSUES = ("brain", "breast", "colon", "mouth", "prostate", "skin", "stomach")

ENCODERS = {
    "ctranspath": {
        "results": ARCHIVED_RESULTS / "transfer/stimage/ctranspath",
        "manifest": CACHE_ROOT / "stimage/ctranspath/manifest.json",
        "published": 0.285210,
        "views": "dual",
    },
    "uni_v1": {
        "results": ARCHIVED_RESULTS / "transfer/stimage/uni_v1",
        "manifest": CACHE_ROOT / "stimage/uni_v1/manifest.json",
        "published": 0.299513,
        "views": "dual",
    },
}
CONFIGS = {
    "borda": {"fusion": "borda", "weighting": "softmax"},
    "minimal": {"fusion": "score", "weighting": "uniform"},
}
CONTEXT_CONFIGS = {
    "context": {"fusion": "context", "weighting": "softmax"},
    "context_uniform": {"fusion": "context", "weighting": "uniform"},
}
DECLARED_K = 128
K_GRID = (8, 16, 32, 64, 128)


def slide_index(spec: dict) -> dict:
    payload = json.loads(Path(spec["manifest"]).read_text())
    return {str(item["sample_id"]): item for item in payload["slides"]}


def slide_views(entry: dict, views: str, weights: np.ndarray):
    feature_path = Path(entry["feature_path"])
    root = feature_path.parent
    # The STImage metadata stores barcodes as an object array, matching the
    # released runner's own `allow_pickle=True` load.
    with np.load(root / "metadata.npz", allow_pickle=True) as archive:
        barcodes = archive["barcodes"].astype(str)
    context = np.asarray(np.load(feature_path, mmap_mode="r"), dtype=np.float32)
    if views == "context_only":
        return context, context, barcodes
    local_path = entry.get("local_path")
    if local_path is not None:
        local = np.asarray(np.load(local_path, mmap_mode="r"), dtype=np.float32)
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
    configs = CONFIGS if spec["views"] == "dual" else CONTEXT_CONFIGS
    weights = radial_pool_weights()
    index = slide_index(spec)
    arms = ["published_lin_mnp"] + [f"{n}_k{k}" for n in configs for k in K_GRID]

    rows = []
    for tissue in TISSUES:
        for path in sorted((spec["results"] / tissue).glob("seed*/predictions.npz")):
            seed = path.parent.name
            with np.load(path, allow_pickle=False) as archive:
                target = archive["target"].astype(np.float32)
                ridge = archive["lin_mnp"].astype(np.float32)
                sections = archive["sections"].astype(str)
                barcodes = archive["barcodes"].astype(str)

            adapted = {name: ridge.copy() for name in arms[1:]}
            for section in dict.fromkeys(sections):
                mask = sections == section
                context, local, cache_barcodes = slide_views(
                    index[section], spec["views"], weights)
                position = {code: i for i, code in enumerate(cache_barcodes)}
                order = np.asarray([position[code] for code in barcodes[mask]])
                for name, config in configs.items():
                    produced = apply_grid(
                        ridge[mask], context[order], local[order], K_GRID, **config)
                    for k, value in produced.items():
                        adapted[f"{name}_k{k}"][mask] = value

            scored = {"published_lin_mnp": ridge} | adapted
            for arm, prediction in scored.items():
                rows.append({
                    "encoder": args.encoder, "tissue": tissue, "seed": seed,
                    "arm": arm, **mean_gene_pcc_and_mse(target, prediction),
                })
            print(json.dumps({
                "tissue": tissue, "seed": seed, "slides": len(set(sections)),
                "spots": int(len(target)),
            }), flush=True)

    detail = pd.DataFrame(rows)
    per_tissue = detail.groupby(["arm", "tissue"], as_index=False)[["pcc", "mse"]].mean()
    summary = per_tissue.groupby("arm", as_index=False)[["pcc", "mse"]].mean()
    baseline = float(summary.loc[summary.arm == "published_lin_mnp", "pcc"].iloc[0])
    summary["delta_vs_published"] = summary.pcc - baseline

    out = args.output / args.encoder
    out.mkdir(parents=True, exist_ok=True)
    detail.to_csv(out / "per_seed.csv", index=False)
    per_tissue.to_csv(out / "per_tissue.csv", index=False)
    summary.to_csv(out / "summary.csv", index=False)
    (out / "metadata.json").write_text(json.dumps({
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "encoder": args.encoder,
        "views": spec["views"],
        "aggregation": "seed mean within tissue, then equal 7-tissue mean",
        "configs": configs, "k_grid": list(K_GRID), "declared_k": DECLARED_K,
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
