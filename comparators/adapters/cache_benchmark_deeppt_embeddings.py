#!/usr/bin/env python3
"""Cache frozen official DeepPT ResNet50 features for all benchmark spots."""

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
from torch.utils.data import DataLoader, Dataset


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from lin_mnp.benchmark.benchmark_external import PATCH_ROOT  # noqa: E402
from lin_mnp.paths import CACHE_ROOT


DEFAULT_OUTPUT = CACHE_ROOT / "cscc/deeppt_resnet50"
AUTHOR_SOURCE = PROJECT_ROOT / "comparators/methods/DeepPT/zenodo_11125591"
IMAGENET_MEAN = torch.tensor((0.485, 0.456, 0.406)).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor((0.229, 0.224, 0.225)).view(1, 3, 1, 1)


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


def run(args: argparse.Namespace) -> Path:
    source = AUTHOR_SOURCE / "11slide_processing/utils_preprocessing.py"
    weights = Path.home() / ".cache/torch/hub/checkpoints/resnet50-11ad3fa6.pth"
    if not source.is_file():
        raise FileNotFoundError(source)
    if not weights.is_file():
        raise FileNotFoundError(weights)
    sys.path.insert(0, str(source.parent))
    from utils_preprocessing import Feature_Extraction  # type: ignore

    args.output.mkdir(parents=True, exist_ok=True)
    final = args.output / "resnet50_imagenet_v2_float32.npy"
    metadata_path = args.output / "metadata.json"
    if final.is_file() and metadata_path.is_file():
        return final
    if final.exists() or metadata_path.exists():
        raise FileExistsError("incomplete immutable DeepPT feature cache")
    staging = final.with_name(f"{final.name}.tmp-{os.getpid()}")

    model = Feature_Extraction(model_type="load_from_saved_file")
    # The official pipeline loads the raw torchvision ResNet state into its
    # ``Feature_Extraction.resnet`` child rather than into the wrapper.
    model.resnet.load_state_dict(torch.load(weights, map_location="cpu"))
    device = torch.device(args.device)
    model = model.to(device).eval()
    dataset = PatchIndex()
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
    )
    output = np.lib.format.open_memmap(
        staging, mode="w+", dtype=np.float32, shape=(len(dataset), 2048)
    )
    mean = IMAGENET_MEAN.to(device)
    std = IMAGENET_STD.to(device)
    with torch.no_grad():
        for patch, index in loader:
            image = patch.to(device, non_blocking=True).permute(0, 3, 1, 2).float()
            image = (image / 255.0 - mean) / std
            output[index.numpy()] = model(image).float().cpu().numpy()
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
        "source": "official DeepPT Feature_Extraction from Zenodo 11125591",
        "official_source_sha256": sha256_file(source),
        "official_archive_md5": "5289156ff7168d310dca292a1c3d0089",
        "weights": "torchvision ResNet50 IMAGENET1K_V2",
        "weights_sha256": sha256_file(weights),
        "normalization": "ImageNet mean/std as official 1main_processing.py",
        "stain_normalization": "not repeated: benchmark-provided fixed 224-px patches used",
        "shape": [len(dataset), 2048],
        "dtype": "float32",
        "spot_order": spot_order,
        "patch_manifest_sha256": sha256_file(PATCH_ROOT / "manifest.json"),
        "feature_sha256": sha256_file(final),
    }
    metadata_path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return final


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    print(json.dumps({"output": str(run(parse_args(argv)))}, sort_keys=True))


if __name__ == "__main__":
    main()
