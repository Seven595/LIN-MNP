#!/usr/bin/env python3
"""Cache target-free OmiCLIP image embeddings for all canonical cSCC spots."""

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
from PIL import Image
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from open_clip import create_model_from_pretrained


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from lin_mnp.benchmark.benchmark_external import PATCH_ROOT  # noqa: E402
from lin_mnp.paths import CACHE_ROOT, MODEL_ROOT


DEFAULT_CHECKPOINT = MODEL_ROOT / "omiclip/checkpoint.pt"
DEFAULT_OUTPUT = CACHE_ROOT / "cscc/omiclip_embeddings"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class PatchDataset(Dataset):
    def __init__(self, path: Path, transform) -> None:
        self.path = path
        self.transform = transform
        self._patches: np.ndarray | None = None

    @property
    def patches(self) -> np.ndarray:
        if self._patches is None:
            self._patches = np.load(self.path, mmap_mode="r")
        return self._patches

    def __len__(self) -> int:
        return int(np.load(self.path, mmap_mode="r").shape[0])

    def __getitem__(self, index: int) -> torch.Tensor:
        return self.transform(Image.fromarray(np.asarray(self.patches[index])))


def cache(args: argparse.Namespace) -> Path:
    output = args.output.resolve()
    complete_path = output / "complete.json"
    if complete_path.is_file():
        complete = json.loads(complete_path.read_text(encoding="utf-8"))
        feature_path = output / "embeddings_float32.npy"
        if sha256_file(feature_path) != complete["embedding_sha256"]:
            raise ValueError("completed OmiCLIP embedding cache hash mismatch")
        return output
    if output.exists():
        raise FileExistsError(f"incomplete cache exists: {output}")
    if not args.checkpoint.is_file():
        raise FileNotFoundError(f"OmiCLIP checkpoint is missing: {args.checkpoint}")
    staging = output.with_name(f"{output.name}.tmp-{os.getpid()}")
    staging.mkdir(parents=True, exist_ok=False)
    model, preprocess = create_model_from_pretrained(
        "coca_ViT-L-14",
        pretrained=str(args.checkpoint),
        device=args.device,
        precision="fp16" if args.amp else "fp32",
        weights_only=False,
    )
    model.eval()
    dataset = PatchDataset(PATCH_ROOT / "patches_224_uint8.npy", preprocess)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
    )
    rows = []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(args.device, non_blocking=True)
            with torch.autocast(
                device_type="cuda", dtype=torch.float16,
                enabled=args.amp and str(args.device).startswith("cuda"),
            ):
                embedding = model.encode_image(batch)
            rows.append(F.normalize(embedding.float(), p=2, dim=-1).cpu().numpy())
    matrix = np.concatenate(rows).astype(np.float32)
    expected_rows = len(dataset)
    if matrix.shape[0] != expected_rows or not np.isfinite(matrix).all():
        raise ValueError(f"invalid OmiCLIP embedding matrix: {matrix.shape}")
    feature_path = staging / "embeddings_float32.npy"
    np.save(feature_path, matrix)
    metadata = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "model": "released OmiCLIP coca_ViT-L-14",
        "checkpoint_path": str(args.checkpoint.resolve()),
        "checkpoint_sha256": sha256_file(args.checkpoint),
        "patch_manifest_sha256": sha256_file(PATCH_ROOT / "manifest.json"),
        "shape": list(matrix.shape),
        "dtype": "float32",
        "l2_normalized": True,
        "target_data_accessed": False,
        "runtime_compatibility_patch": (
            "weights_only=False for the hash-verified official checkpoint because "
            "it contains a NumPy scalar rejected by PyTorch's restricted loader"
        ),
    }
    metadata_path = staging / "metadata.json"
    metadata_path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    complete = {
        "embedding_sha256": sha256_file(feature_path),
        "metadata_sha256": sha256_file(metadata_path),
    }
    (staging / "complete.json").write_text(
        json.dumps(complete, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    staging.rename(output)
    return output


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--amp", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    print(json.dumps({"output": str(cache(parse_args(argv)))}, sort_keys=True))


if __name__ == "__main__":
    main()
