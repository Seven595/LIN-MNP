#!/usr/bin/env python3
"""Create a target-blind canonical 384-pixel cSCC patch cache.

The row identities are copied from the audited Benchmark 224-pixel manifest,
while every RGB crop is re-read from the source H&E image at a true 384x384
spot-centred field of view.  Expression matrices are never opened.  Original
encoders may subsequently resize these source fields to their native network
input size; this is the same field-of-view convention used by CTransPath-384.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import gzip
import hashlib
import io
import json
import os
from pathlib import Path

import numpy as np
from PIL import Image

from lin_mnp.paths import CACHE_ROOT, DATA_ROOT

DERIVED = DATA_ROOT / "cscc/derived"
SOURCE_MANIFEST = DERIVED / "shared_raw_patches/benchmark_224_v1/manifest.json"
PREPARED = DERIVED / "heto_official_cv4/fold0/locked997/prepared/slides"
DEFAULT_OUTPUT = CACHE_ROOT / "cscc/context384_patches"
PATCH_SIZE = 384


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build(output: Path) -> Path:
    output = output.resolve()
    complete = output / "complete.json"
    if complete.is_file():
        payload = json.loads(complete.read_text(encoding="utf-8"))
        patch_path = output / "patches_384_uint8.npy"
        if sha256_file(patch_path) != payload["patch_sha256"]:
            raise ValueError("completed 384-pixel cache hash mismatch")
        return output
    if output.exists():
        raise FileExistsError(f"partial immutable output exists: {output}")

    source = json.loads(SOURCE_MANIFEST.read_text(encoding="utf-8"))
    rows = list(source["slides"])
    if len(rows) != 12 or sum(int(r["stop"]) - int(r["start"]) for r in rows) != 8671:
        raise ValueError("invalid canonical Benchmark manifest")
    staging = output.with_name(f"{output.name}.tmp-{os.getpid()}")
    staging.mkdir(parents=True, exist_ok=False)
    patch_path = staging / "patches_384_uint8.npy"
    matrix = np.lib.format.open_memmap(
        patch_path, mode="w+", dtype=np.uint8,
        shape=(8671, PATCH_SIZE, PATCH_SIZE, 3),
    )

    Image.MAX_IMAGE_PIXELS = None
    half = PATCH_SIZE // 2
    audited_rows = []
    for row in rows:
        sample = str(row["sample_id"])
        prepared_path = PREPARED / f"{sample}.npz"
        with np.load(prepared_path, allow_pickle=False) as prepared:
            barcodes = prepared["barcodes"].astype(str)
            coordinates = np.rint(prepared["image_coords"]).astype(np.int64)
        if tuple(barcodes) != tuple(map(str, row["barcodes"])):
            raise ValueError(f"barcode mismatch: {sample}")
        image_path = Path(row["image_path"])
        if sha256_file(image_path) != str(row["image_sha256"]):
            raise ValueError(f"source image changed: {sample}")
        with gzip.open(image_path, "rb") as handle:
            image = Image.open(io.BytesIO(handle.read())).convert("RGB")
            image.load()
        start, stop = int(row["start"]), int(row["stop"])
        for local, (x_coord, y_coord) in enumerate(coordinates):
            crop = image.crop((
                int(x_coord) - half, int(y_coord) - half,
                int(x_coord) + half, int(y_coord) + half,
            ))
            matrix[start + local] = np.asarray(crop, dtype=np.uint8)
        matrix.flush()
        audited_rows.append({
            **row,
            "prepared_path": str(prepared_path.resolve()),
            "prepared_sha256": sha256_file(prepared_path),
        })
        print(json.dumps({"sample_id": sample, "spots": stop - start}), flush=True)
    del matrix

    manifest = {
        "schema_version": 1,
        "artifact_id": "cscc12_heto_benchmark_384_patch_cache_v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_identity_manifest": str(SOURCE_MANIFEST.resolve()),
        "source_identity_manifest_sha256": sha256_file(SOURCE_MANIFEST),
        "patch_size": PATCH_SIZE,
        "patch_layout": "NHWC RGB uint8; true 384x384 source field centred on image_coords",
        "patch_file": patch_path.name,
        "n_spots": 8671,
        "slides": audited_rows,
        "target_data_accessed": False,
    }
    manifest_path = staging / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    complete_payload = {
        "patch_sha256": sha256_file(patch_path),
        "manifest_sha256": sha256_file(manifest_path),
        "n_spots": 8671,
        "target_data_accessed": False,
    }
    (staging / "complete.json").write_text(
        json.dumps(complete_payload, indent=2, sort_keys=True) + "\n"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    staging.rename(output)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    print(json.dumps({"output": str(build(args.output))}, sort_keys=True))


if __name__ == "__main__":
    main()
