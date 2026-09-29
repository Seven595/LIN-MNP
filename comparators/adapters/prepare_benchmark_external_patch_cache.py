#!/usr/bin/env python3
"""Create the shared 224 px cSCC patch cache for external-method reruns.

The cache contains images only.  Expression labels remain in the immutable
cohort shards and are opened independently by each patient-CV fold.  A single
global patch order is used for every method so prediction rows can be audited
against slide/barcode identities without relying on a method's private data
loader.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import gzip
import hashlib
import io
import json
from pathlib import Path
from typing import Sequence

import numpy as np
from PIL import Image

from lin_mnp.paths import CACHE_ROOT, DATA_ROOT


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DERIVED_ROOT = DATA_ROOT / "cscc/derived"
PREPARED = DERIVED_ROOT / "heto_official_cv4/fold0/locked997/prepared/slides"
IMAGE_ROOT = DATA_ROOT / "cscc/cohort/raw/he_spot_only"
DEFAULT_OUTPUT = CACHE_ROOT / "cscc/benchmark_patches"
PATCH_SIZE = 224


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _image_path(sample_id: str) -> Path:
    matches = sorted(IMAGE_ROOT.glob(f"{sample_id}_*.jpg.gz"))
    if len(matches) != 1:
        raise ValueError(f"expected one H&E image for {sample_id}, found {matches}")
    return matches[0]


def build_cache(output: Path) -> Path:
    output = output.resolve()
    if output.exists():
        manifest = output / "manifest.json"
        if not manifest.is_file():
            raise FileExistsError(f"non-cache path already exists: {output}")
        return output

    slide_paths = sorted(PREPARED.glob("*.npz"))
    if len(slide_paths) != 12:
        raise ValueError(f"expected 12 prepared slides, found {len(slide_paths)}")

    slide_rows = []
    total = 0
    genes = None
    for path in slide_paths:
        with np.load(path, allow_pickle=False) as data:
            sample_id = str(data["sample_id"].item())
            barcodes = data["barcodes"].astype(str)
            current_genes = tuple(data["genes"].astype(str))
            if genes is None:
                genes = current_genes
            elif current_genes != genes:
                raise ValueError(f"gene order differs in {path}")
            start = total
            total += len(barcodes)
            slide_rows.append(
                {
                    "sample_id": sample_id,
                    "prepared_path": str(path.resolve()),
                    "prepared_sha256": sha256_file(path),
                    "image_path": str(_image_path(sample_id).resolve()),
                    "image_sha256": sha256_file(_image_path(sample_id)),
                    "start": start,
                    "stop": total,
                    "n_spots": len(barcodes),
                    "barcodes": barcodes.tolist(),
                }
            )
    if total != 8671 or genes is None or len(genes) != 997:
        raise ValueError(f"unexpected cache dimensions: spots={total}, genes={len(genes or ())}")

    staging = output.with_name(f"{output.name}.tmp")
    staging.mkdir(parents=True, exist_ok=False)
    patch_path = staging / "patches_224_uint8.npy"
    patches = np.lib.format.open_memmap(
        patch_path,
        mode="w+",
        dtype=np.uint8,
        shape=(total, PATCH_SIZE, PATCH_SIZE, 3),
    )

    Image.MAX_IMAGE_PIXELS = None
    half = PATCH_SIZE // 2
    for row in slide_rows:
        prepared_path = Path(row["prepared_path"])
        with np.load(prepared_path, allow_pickle=False) as data:
            coordinates = np.rint(data["image_coords"]).astype(np.int64)
        with gzip.open(row["image_path"], "rb") as handle:
            image = Image.open(io.BytesIO(handle.read())).convert("RGB")
            image.load()
        for local_index, (x_coord, y_coord) in enumerate(coordinates):
            crop = image.crop(
                (
                    int(x_coord) - half,
                    int(y_coord) - half,
                    int(x_coord) + half,
                    int(y_coord) + half,
                )
            )
            patches[int(row["start"]) + local_index] = np.asarray(crop, dtype=np.uint8)
        patches.flush()

    del patches
    manifest = {
        "schema_version": 1,
        "artifact_id": "cscc12_heto_benchmark_224_patch_cache_v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "patch_size": PATCH_SIZE,
        "patch_layout": "NHWC RGB uint8; centered on image_coords (x,y)",
        "n_spots": total,
        "n_genes_in_identity_contract": len(genes),
        "gene_order_sha256": hashlib.sha256("\n".join(genes).encode()).hexdigest(),
        "patch_file": patch_path.name,
        "patch_file_sha256": sha256_file(patch_path),
        "slides": slide_rows,
    }
    (staging / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    staging.rename(output)
    return output


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    output = build_cache(args.output)
    print(json.dumps({"output": str(output)}, sort_keys=True))


if __name__ == "__main__":
    main()
