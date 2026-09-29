#!/usr/bin/env python3
"""Retain each released downstream after changing only original-encoder FOV to 384 px."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys

import numpy as np

from lin_mnp.paths import ARCHIVE_ROOT, CACHE_ROOT, REPOSITORY_ROOT, RESULTS_ROOT

PROJECT = REPOSITORY_ROOT
SCRIPTS = PROJECT / "comparators/adapters"
CODE = ARCHIVE_ROOT / "component_analysis/code"
FEATURES = CACHE_ROOT / "cscc/original_encoders_384"
OUTPUT = RESULTS_ROOT / "component_analysis/original_encoders_384/downstream"
METHODS = (
    "gr2st", "asgnn", "bleep", "deeppt", "mclstexp", "igidl",
    "egnv1", "egnv2", "omiclip", "istar", "he2gene", "heclip",
    "tcgn", "stnet", "deepspace",
)
MISSING = {"he2gene", "heclip", "tcgn", "stnet", "deepspace"}
DISPLAY = {
    "gr2st": "Phikon-v2", "asgnn": "ImageNet-v2 ResNet50",
    "bleep": "ImageNet-v2 ResNet50 with EGN preprocessing",
    "deeppt": "ImageNet-v2 ResNet50", "mclstexp": "ImageNet-v2 ResNet50 with EGN preprocessing",
    "igidl": "ImageNet ResNet18 with EGN preprocessing",
    "egnv1": "ImageNet-v2 ResNet50 with EGN preprocessing",
    "egnv2": "ImageNet ResNet18 with EGN preprocessing",
    "omiclip": "released OmiCLIP CoCa ViT-L/14", "istar": "released HIPT ViT-256 + ViT-4K",
    "he2gene": "HE2Gene ImageNet ResNet50", "heclip": "HECLIP ImageNet ResNet50",
    "tcgn": "TCGN CMT/graph feature extractor", "stnet": "ST-Net ImageNet DenseNet121",
    "deepspace": "DeepSpaCE ImageNet VGG16",
}
DIM = {"he2gene": 2048, "heclip": 2048, "tcgn": 1280, "stnet": 1024, "deepspace": 4096}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def force_common_target(module) -> None:
    original_apply = module.apply_target_transform
    module.fit_target_transform = lambda _slides, _ids, _kind: {"kind": "log1p_raw"}
    module.apply_target_transform = original_apply


def repair(result: Path, method: str) -> None:
    metadata_path = result / "run_metadata.json"
    complete_path = result / "complete.json"
    metadata = json.loads(metadata_path.read_text())
    metadata["experiment"] = "Original_encoder_direct_head_attribution_v1/original_encoder384_v1"
    metadata["target_parameters"] = {"kind": "log1p_raw"}
    if isinstance(metadata.get("method"), dict):
        metadata["method"]["target_transform"] = "official log1p(raw count)"
        metadata["method"]["encoder"] = f"{DISPLAY[method]} from true 384 px source field"
    metadata["original_encoder384_control"] = {
        "source_field_size_px": 384,
        "encoder_input_size": "released native input size",
        "downstream_retained": True,
        "target_transform": "official log1p(raw count)",
        "outer_split": "official patient CV4",
        "selection": "inner validation patient only",
    }
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    complete = json.loads(complete_path.read_text())
    complete["run_metadata_sha256"] = sha256_file(metadata_path)
    complete_path.write_text(json.dumps(complete, indent=2, sort_keys=True) + "\n")


def run_standard(args) -> Path:
    method, fold = args.method, args.fold
    os.environ["LIN_MNP_CHECKPOINT_SELECTION_SCOPE"] = "inner_validation"
    if method in {"gr2st", "asgnn"}:
        module = load(f"orig384_additional_{method}_{fold}", SCRIPTS / "train_additional_comparators_fold.py")
        force_common_target(module)
        if method == "gr2st":
            os.environ["LIN_MNP_PHIKON_CROP_SIZE"] = "384"
        else:
            module.RESNET_ROOT = FEATURES / "deeppt"
        namespace = argparse.Namespace(
            method=method, fold=fold, device=args.device, workers=2, seed=17,
            amp=True, max_epochs=None,
            locked_adaptation=None, output=OUTPUT,
        )
        result = module.train_fold(namespace)
    elif method in {"bleep", "mclstexp", "igidl"}:
        module = load(f"orig384_retrieval_{method}_{fold}", SCRIPTS / "train_benchmark_external_retrieval_fold.py")
        force_common_target(module)
        module.FEATURE_ROOT = FEATURES / "egn"
        namespace = argparse.Namespace(
            method=method, fold=fold, device=args.device, seed=17,
            epochs=30, patience=8, batch_size=128, output=OUTPUT,
        )
        result = module.train_fold(namespace)
    elif method == "deeppt":
        module = load(f"orig384_deeppt_{fold}", SCRIPTS / "train_benchmark_external_deeppt_fold.py")
        force_common_target(module)
        module.FEATURE_ROOT = FEATURES / "deeppt"
        result = module.train_fold(argparse.Namespace(
            fold=fold, device=args.device, seed=17, ae_epochs=500,
            mlp_epochs=500, patience=50, output=OUTPUT,
        ))
    elif method == "egnv1":
        module = load(f"orig384_egnv1_{fold}", SCRIPTS / "train_benchmark_external_egnv1_fold.py")
        force_common_target(module)
        module.EMBEDDING_ROOT = FEATURES / "egn"
        result = module.train_fold(argparse.Namespace(
            fold=fold, device=args.device, workers=2, seed=17, amp=True,
            epochs=50, output=OUTPUT,
        ))
    elif method == "egnv2":
        module = load(f"orig384_egnv2_{fold}", SCRIPTS / "train_benchmark_external_egnv2_fold.py")
        force_common_target(module)
        module.EMBEDDING_ROOT = FEATURES / "egn"
        result = module.train_fold(argparse.Namespace(
            fold=fold, device=args.device, seed=17, epochs=300, output=OUTPUT,
        ))
    elif method == "omiclip":
        module = load(f"orig384_omiclip_{fold}", SCRIPTS / "train_benchmark_external_omiclip_fold.py")
        force_common_target(module)
        result = module.run_fold(argparse.Namespace(
            fold=fold, device=args.device, checkpoint=module.DEFAULT_CHECKPOINT,
            cache=FEATURES / "omiclip", batch_size=128, output=OUTPUT,
        ))
    else:
        module = load(f"orig384_istar_{fold}", SCRIPTS / "train_benchmark_external_istar_fold.py")
        force_common_target(module)
        module.CACHE_ROOT = FEATURES / "istar"
        result = module.train_fold(argparse.Namespace(
            fold=fold, device=args.device, workers=2, epochs=400,
            seed=17, amp=True, output=OUTPUT,
        ))
    repair(result, method)
    return result


def missing_features(module, method: str, fold: int):
    _, identities = module.load_features(384)
    patch_manifest = json.loads(
        (ROOT / "inputs/benchmark_384_v1/manifest.json").read_text()
    )
    offsets = {str(row["sample_id"]): row for row in patch_manifest["slides"]}
    suffix = f"fold{fold}" if method in {"he2gene", "tcgn"} else "common"
    cache = FEATURES / "missing" / method / suffix
    metadata = json.loads((cache / "metadata.json").read_text())
    complete = json.loads((cache / "complete.json").read_text())
    path = cache / "features_float32.npy"
    if sha256_file(path) != complete["features_sha256"]:
        raise ValueError(f"feature hash mismatch: {path}")
    flat = np.load(path, mmap_mode="r")
    features = {
        sample: np.asarray(
            flat[int(offsets[sample]["start"]):int(offsets[sample]["stop"])],
            dtype=np.float32,
        )
        for sample in identities
    }
    if metadata.get("source_field_size_px") != 384:
        raise ValueError(f"not a 384 px feature cache: {cache}")
    return features, identities


def run_missing(args) -> Path:
    module = load(
        f"orig384_missing_{args.method}_{args.fold}",
        CODE / "run_missing_ctranspath_original_downstream.py",
    )
    original_loader = module.load_features
    def loader(_scale):
        module.load_features = original_loader
        try:
            return missing_features(module, args.method, args.fold)
        finally:
            module.load_features = loader
    module.load_features = loader
    module.FEATURE_DIM = DIM[args.method]
    result = module.train(args.method, args.fold, module.torch.device(args.device), OUTPUT)
    repair(result, args.method)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=METHODS, required=True)
    parser.add_argument("--fold", type=int, choices=range(4), required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    result = run_missing(args) if args.method in MISSING else run_standard(args)
    print(json.dumps({"method": args.method, "fold": args.fold, "output": str(result)}))


if __name__ == "__main__":
    main()
