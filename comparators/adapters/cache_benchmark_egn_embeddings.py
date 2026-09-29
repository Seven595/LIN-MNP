#!/usr/bin/env python3
"""Cache author-style ResNet exemplar embeddings for all 8,671 cSCC spots."""

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
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import torchvision


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from lin_mnp.benchmark.benchmark_external import PATCH_ROOT  # noqa: E402
from lin_mnp.paths import CACHE_ROOT


DEFAULT_OUTPUT = CACHE_ROOT / "cscc/egn_embeddings"


class PatchIndex(Dataset):
    def __init__(self) -> None:
        self.path = PATCH_ROOT / "patches_224_uint8.npy"
        self._array: np.ndarray | None = None
        self.length = int(np.load(self.path, mmap_mode="r").shape[0])

    @property
    def array(self) -> np.ndarray:
        if self._array is None:
            self._array = np.load(self.path, mmap_mode="r")
        return self._array

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        return torch.from_numpy(np.asarray(self.array[index]).copy()), index


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build_encoder(backbone: str) -> tuple[nn.Module, int, str]:
    if backbone == "resnet50":
        model = torchvision.models.resnet50(
            weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V2
        )
        width = model.fc.in_features
        revision = "torchvision ResNet50 IMAGENET1K_V2"
    elif backbone == "resnet18":
        model = torchvision.models.resnet18(
            weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1
        )
        width = model.fc.in_features
        revision = "torchvision ResNet18 IMAGENET1K_V1"
    else:
        raise ValueError(backbone)
    return nn.Sequential(*list(model.children())[:-1]), width, revision


def run(args: argparse.Namespace) -> Path:
    args.output.mkdir(parents=True, exist_ok=True)
    final = args.output / f"{args.backbone}_embeddings_float32.npy"
    metadata_path = args.output / f"{args.backbone}_metadata.json"
    if final.is_file() and metadata_path.is_file():
        return final
    if final.exists() or metadata_path.exists():
        raise FileExistsError("incomplete immutable EGN embedding cache")
    staging = final.with_name(f"{final.name}.tmp-{os.getpid()}")
    encoder, width, revision = build_encoder(args.backbone)
    device = torch.device(args.device)
    encoder = encoder.to(device).eval()
    dataset = PatchIndex()
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
    )
    output = np.lib.format.open_memmap(
        staging, mode="w+", dtype=np.float32, shape=(len(dataset), width)
    )
    with torch.no_grad():
        for patch, index in loader:
            # This is the normalization used by the released exemplar builder.
            image = patch.to(device, non_blocking=True).permute(0, 3, 1, 2).float()
            image = (image / 255.0 - 0.5) / 0.5
            embedding = encoder(image).flatten(1).float().cpu().numpy()
            output[index.numpy()] = embedding
    output.flush()
    del output
    staging.rename(final)
    patch_manifest = json.loads((PATCH_ROOT / "manifest.json").read_text(encoding="utf-8"))
    spot_order = [
        {
            "sample_id": str(row["sample_id"]), "start": int(row["start"]),
            "stop": int(row["stop"]), "n_spots": int(row["n_spots"]),
        }
        for row in sorted(patch_manifest["slides"], key=lambda value: int(value["start"]))
    ]
    metadata = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "backbone": args.backbone,
        "encoder": revision,
        "normalization": "(RGB/255 - 0.5) / 0.5 as released EGN build_exemplar.py",
        "shape": [len(dataset), width],
        "dtype": "float32",
        "spot_order": spot_order,
        "patch_manifest_sha256": sha256_file(PATCH_ROOT / "manifest.json"),
        "embedding_sha256": sha256_file(final),
    }
    metadata_path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return final


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backbone", choices=("resnet50", "resnet18"), required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    print(json.dumps({"output": str(run(parse_args(argv)))}, sort_keys=True))


if __name__ == "__main__":
    main()
