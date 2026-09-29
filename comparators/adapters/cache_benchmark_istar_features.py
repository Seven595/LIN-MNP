#!/usr/bin/env python3
"""Cache official iStar/HIPT features for the benchmark first-slide subset."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
from typing import Sequence

import numpy as np
import torch
import torch.nn.functional as F


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
ISTAR_ROOT = PROJECT_ROOT / "comparators/methods/iStar"
if str(ISTAR_ROOT) not in sys.path:
    sys.path.insert(0, str(ISTAR_ROOT))

from lin_mnp.benchmark.benchmark_external import PATCH_ROOT  # noqa: E402
from lin_mnp.paths import CACHE_ROOT
from hipt_model_utils import get_vit256, get_vit4k  # noqa: E402


DEFAULT_OUTPUT = CACHE_ROOT / "cscc/istar_hipt"
WEIGHT_256 = ISTAR_ROOT / "checkpoints/vit256_small_dino.pth"
WEIGHT_4K = ISTAR_ROOT / "checkpoints/vit4k_xs_dino.pth"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def disk_mask(radius: float = 6.875) -> np.ndarray:
    axis = np.arange(16, dtype=np.float32) - 7.5
    yy, xx = np.meshgrid(axis, axis, indexing="ij")
    return (yy * yy + xx * xx <= radius * radius).reshape(-1)


@torch.inference_mode()
def build_cache(args: argparse.Namespace) -> Path:
    metadata_path = args.output / "metadata.json"
    feature_path = args.output / "features_float16.npy"
    if metadata_path.is_file() and feature_path.is_file():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if sha256_file(feature_path) != metadata["feature_sha256"]:
            raise ValueError("cached iStar feature hash mismatch")
        return feature_path
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError(f"incomplete iStar cache exists: {args.output}")
    for path in (WEIGHT_256, WEIGHT_4K):
        if not path.is_file() or path.stat().st_size < 100_000_000:
            raise FileNotFoundError(f"iStar HIPT checkpoint is missing or truncated: {path}")

    external_contract = bool(os.environ.get("LIN_MNP_EXTERNAL_CONTRACT"))
    if external_contract:
        patch_manifest = json.loads(
            (PATCH_ROOT / "manifest.json").read_text(encoding="utf-8")
        )
        manifest_rows = sorted(
            patch_manifest["slides"], key=lambda value: int(value["start"])
        )
        sample_ids = tuple(str(row["sample_id"]) for row in manifest_rows)
    else:
        from lin_mnp.benchmark.benchmark_external import load_slides
        slides, _ = load_slides()
        sample_ids = tuple(sorted(
            min(sample for sample, slide in slides.items() if slide.patient_id == patient)
            for patient in sorted({slide.patient_id for slide in slides.values()})
        ))
    rows = []
    start = 0
    for sample_id in sample_ids:
        if external_contract:
            source = next(row for row in manifest_rows if str(row["sample_id"]) == sample_id)
            row_start, stop = int(source["start"]), int(source["stop"])
            if row_start != start:
                raise ValueError("external patch manifest must be contiguous")
            rows.append({
                "sample_id": sample_id,
                "patient_id": str(source.get("patient_id", sample_id)),
                "start": row_start, "stop": stop,
                "barcodes": list(map(str, source["barcodes"])),
            })
        else:
            stop = start + len(slides[sample_id].barcodes)
            rows.append({
                "sample_id": sample_id,
                "patient_id": slides[sample_id].patient_id,
                "start": start, "stop": stop,
                "barcodes": list(slides[sample_id].barcodes),
            })
        start = stop
    mask = disk_mask(args.radius_cells)
    n_cells = int(mask.sum())
    args.output.mkdir(parents=True, exist_ok=True)
    staging_feature = args.output / "features_float16.tmp.npy"
    output = np.lib.format.open_memmap(
        staging_feature, mode="w+", dtype=np.float16, shape=(start, n_cells, 579)
    )
    patches = np.load(PATCH_ROOT / "patches_224_uint8.npy", mmap_mode="r")
    device = torch.device(args.device)
    model256 = get_vit256(pretrained_weights=str(WEIGHT_256)).to(device).eval()
    model4k = get_vit4k(pretrained_weights=str(WEIGHT_4K)).to(device).eval()
    write_start = 0
    patch_canvas_size = int(patches.shape[1])
    if patches.shape[2] != patch_canvas_size or args.raw_field_size > patch_canvas_size:
        raise ValueError(
            f"raw field {args.raw_field_size}px is incompatible with patch canvas "
            f"{tuple(patches.shape[1:3])}"
        )
    crop_start = (patch_canvas_size - args.raw_field_size) // 2
    for row in rows:
        indices = np.arange(int(row["start"]), int(row["stop"]))
        for batch_start in range(0, len(indices), args.batch_size):
            selected = indices[batch_start : batch_start + args.batch_size]
            array = np.asarray(
                patches[selected, crop_start : crop_start + args.raw_field_size,
                        crop_start : crop_start + args.raw_field_size],
                dtype=np.float32,
            )
            image = torch.from_numpy(array).to(device).permute(0, 3, 1, 2) / 255.0
            image = F.interpolate(image, size=(256, 256), mode="bilinear", align_corners=False)
            normalized = (image - 0.5) / 0.5
            features = model256.forward_all(normalized)
            cls256 = features[:, 0]
            sub256 = features[:, 1:]
            global_features = model4k.forward_all(cls256[:, :, None, None])[:, 0]
            rgb = F.adaptive_avg_pool2d(image, (16, 16)).flatten(2).transpose(1, 2)
            combined = torch.cat(
                (global_features[:, None].expand(-1, 256, -1), sub256, rgb), dim=-1
            )
            combined = combined[:, torch.from_numpy(mask).to(device)]
            batch_stop = write_start + len(selected)
            output[write_start:batch_stop] = combined.cpu().to(torch.float16).numpy()
            write_start = batch_stop
    output.flush()
    del output, model256, model4k
    if write_start != start:
        raise ValueError("iStar cache did not cover all first-slide spots")
    staging_feature.rename(feature_path)
    metadata = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_revision": os.popen(f"git -C {ISTAR_ROOT} rev-parse HEAD").read().strip(),
        "coverage_contract": (
            "all HESpotEx sections" if external_contract
            else "first slide per patient, as HEtoSGEBench iStar subset"
        ),
        "sample_ids": list(sample_ids),
        "slides": rows,
        "feature_shape": [start, n_cells, 579],
        "feature_dtype": "float16",
        "feature_sha256": sha256_file(feature_path),
        "patch_manifest_sha256": sha256_file(PATCH_ROOT / "manifest.json"),
        "weight_sha256": {
            WEIGHT_256.name: sha256_file(WEIGHT_256),
            WEIGHT_4K.name: sha256_file(WEIGHT_4K),
        },
        "physical_adapter": {
            "raw_pixel_size_um": 0.8,
            "raw_field_size_pixels": args.raw_field_size,
            "resized_field_size_pixels": 256,
            "cell_grid": [16, 16],
            "cell_width_um": 8.0,
            "spot_radius_um": 55.0,
            "radius_cells": args.radius_cells,
            "cells_per_spot": n_cells,
        },
        "runtime_compatibility_patches": [
            "spot-aligned 160px fields replace unreleased multi-slide dynamic whole-slide loader",
            "one HIPT-4K context token per spot field; released nonlinear ForwardSum head retained",
            "released whole-slide shifted/smoothed feature pass omitted",
        ],
    }
    metadata_path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return feature_path


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--raw-field-size", type=int, default=160)
    parser.add_argument("--radius-cells", type=float, default=6.875)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    print(json.dumps({"output": str(build_cache(parse_args(argv)))}, sort_keys=True))


if __name__ == "__main__":
    main()
