#!/usr/bin/env python3
"""Fit common-contract Ridge heads to every original encoder at 384-pixel FOV."""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import sys

from lin_mnp.paths import ARCHIVE_ROOT, CACHE_ROOT, DATA_ROOT, REPOSITORY_ROOT, RESULTS_ROOT

PROJECT = REPOSITORY_ROOT
CODE = ARCHIVE_ROOT / "component_analysis/code"
FEATURES = CACHE_ROOT / "cscc/original_encoders_384"
OUTPUT = RESULTS_ROOT / "component_analysis/original_encoders_384/ridge"
SHARED = {
    "gr2st": {
        "variant": "phikon_v2_cls384",
        "encoder": "Phikon-v2 CLS from 384 px source field",
        "methods": ["GR2ST"], "kind": "phikon", "scale": 384, "dim": 1024,
    },
    "resnet50_imagenet_v2": {
        "variant": "resnet50_imagenet_v2_source384",
        "encoder": "ImageNet-v2 ResNet50 from 384 px source field",
        "methods": ["asGNN", "DeepPT"], "kind": "flat",
        "matrix": FEATURES / "deeppt/resnet50_imagenet_v2_float32.npy",
        "metadata": FEATURES / "deeppt/metadata.json", "dim": 2048,
    },
    "resnet50_egn": {
        "variant": "resnet50_egn_source384",
        "encoder": "ImageNet-v2 ResNet50 with EGN preprocessing from 384 px source field",
        "methods": ["BLEEP", "mclSTExp", "EGNv1"], "kind": "flat",
        "matrix": FEATURES / "egn/resnet50_embeddings_float32.npy",
        "metadata": FEATURES / "egn/resnet50_metadata.json", "dim": 2048,
    },
    "resnet18_egn": {
        "variant": "resnet18_egn_source384",
        "encoder": "ImageNet ResNet18 with EGN preprocessing from 384 px source field",
        "methods": ["IGI-DL", "EGNv2"], "kind": "flat",
        "matrix": FEATURES / "egn/resnet18_embeddings_float32.npy",
        "metadata": FEATURES / "egn/resnet18_metadata.json", "dim": 512,
    },
    "omiclip": {
        "variant": "omiclip_coca_source384",
        "encoder": "released OmiCLIP CoCa ViT-L/14 from 384 px source field",
        "methods": ["OmiCLIP"], "kind": "flat_manifest_order",
        "matrix": FEATURES / "omiclip/embeddings_float32.npy",
        "metadata": FEATURES / "omiclip/metadata.json", "dim": 768,
    },
    "ciga": {
        "variant": "stflow_ciga_resnet18_source384",
        "encoder": "Ciga tenpercent ResNet18 from 384 px source field",
        "methods": ["STFlow"], "kind": "flat",
        "matrix": FEATURES / "ciga/features_float32.npy",
        "metadata": FEATURES / "ciga/metadata.json", "dim": 512,
    },
}
MISSING = ("he2gene", "heclip", "stnet", "deepspace", "tcgn")
METHODS = (*SHARED, *MISSING)


def load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=METHODS, required=True)
    parser.add_argument("--fold", type=int, choices=range(4))
    args = parser.parse_args()
    if args.method in SHARED:
        module = load(
            f"encoder384_ridge_{args.method}",
            CODE / "run_benchmark_common_original_encoder_ridge.py",
        )
        entry = SHARED[args.method]
        module.VARIANTS.clear()
        module.VARIANTS[entry["variant"]] = {
            key: value for key, value in entry.items() if key != "variant"
        }
        module.PHIKON = (
            DATA_ROOT / "cscc/derived/shared_image_descriptors/phikon_v2/slides"
        )
        module.run_variant(entry["variant"], OUTPUT)
        result = OUTPUT / f"{entry['variant']}_affine_ridge_common_raw"
    else:
        if args.fold is None:
            raise ValueError("missing-encoder Ridge requires --fold")
        module = load(
            f"encoder384_missing_ridge_{args.method}_{args.fold}",
            CODE / "run_missing_original_encoder_ridge.py",
        )
        result = module.run(
            args.method, args.fold, FEATURES / "missing", OUTPUT
        )
    print(json.dumps({"method": args.method, "fold": args.fold, "output": str(result)}))


if __name__ == "__main__":
    main()
