#!/usr/bin/env python3
"""Retrain original downstream pipelines on the shared Benchmark raw-log1p target."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys

from lin_mnp.paths import REPOSITORY_ROOT, RESULTS_ROOT

PROJECT = REPOSITORY_ROOT
SCRIPTS = PROJECT / "comparators/adapters"
OUTPUT = RESULTS_ROOT / "component_analysis/common_contract/original_encoder_original_downstream_common_raw"
METHODS = (
    "gr2st",
    "bleep",
    "deeppt",
    "mclstexp",
    "igidl",
    "egnv1",
    "egnv2",
    "omiclip",
    "istar",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_script(name: str, filename: str):
    path = SCRIPTS / filename
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def force_common_target(module) -> None:
    original_apply = module.apply_target_transform

    def fit_target_transform(_slides, _train_ids, _requested):
        return {"kind": "log1p_raw"}

    module.fit_target_transform = fit_target_transform
    module.apply_target_transform = original_apply


def run(method: str, fold: int, device: str, output: Path) -> Path:
    os.environ["LIN_MNP_CHECKPOINT_SELECTION_SCOPE"] = "inner_validation"
    if method == "gr2st":
        module = load_script("common_raw_additional", "train_additional_comparators_fold.py")
        force_common_target(module)
        args = argparse.Namespace(
            method="gr2st", fold=fold, device=device, workers=2, seed=17,
            amp=True, max_epochs=None,
            locked_adaptation=None, output=output,
        )
        result = module.train_fold(args)
    elif method in {"bleep", "mclstexp", "igidl"}:
        module = load_script("common_raw_retrieval", "train_benchmark_external_retrieval_fold.py")
        force_common_target(module)
        args = argparse.Namespace(
            method=method, fold=fold, device=device, seed=17,
            epochs=30, patience=8, batch_size=128, output=output,
        )
        result = module.train_fold(args)
    elif method == "deeppt":
        module = load_script("common_raw_deeppt", "train_benchmark_external_deeppt_fold.py")
        force_common_target(module)
        args = argparse.Namespace(
            fold=fold, device=device, seed=17, ae_epochs=500, mlp_epochs=500,
            patience=50, output=output,
        )
        result = module.train_fold(args)
    elif method == "omiclip":
        module = load_script("common_raw_omiclip", "train_benchmark_external_omiclip_fold.py")
        force_common_target(module)
        args = argparse.Namespace(
            fold=fold, device=device, checkpoint=module.DEFAULT_CHECKPOINT,
            cache=module.DEFAULT_CACHE, batch_size=128, output=output,
        )
        result = module.run_fold(args)
    elif method == "egnv1":
        module = load_script("common_raw_egnv1", "train_benchmark_external_egnv1_fold.py")
        force_common_target(module)
        args = argparse.Namespace(
            fold=fold, device=device, workers=2, seed=17, amp=True,
            epochs=50, output=output,
        )
        result = module.train_fold(args)
    elif method == "egnv2":
        module = load_script("common_raw_egnv2", "train_benchmark_external_egnv2_fold.py")
        force_common_target(module)
        args = argparse.Namespace(
            fold=fold, device=device, seed=17, epochs=300, output=output,
        )
        result = module.train_fold(args)
    else:
        module = load_script("common_raw_istar", "train_benchmark_external_istar_fold.py")
        force_common_target(module)
        args = argparse.Namespace(
            fold=fold, device=device, workers=2, epochs=400, seed=17,
            amp=True, output=output,
        )
        result = module.train_fold(args)

    metadata_path = result / "run_metadata.json"
    metadata = json.loads(metadata_path.read_text())
    if metadata["target_parameters"] != {"kind": "log1p_raw"}:
        raise ValueError(f"common target patch failed: {result}")
    native = metadata["method"].get("target_transform")
    if native != "official log1p(raw count)":
        metadata["method"]["native_target_transform_before_common_contract"] = native
    metadata["method"]["target_transform"] = "official log1p(raw count)"
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    complete_path = result / "complete.json"
    complete = json.loads(complete_path.read_text())
    complete["run_metadata_sha256"] = sha256_file(metadata_path)
    complete_path.write_text(json.dumps(complete, indent=2, sort_keys=True) + "\n")
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", required=True, choices=METHODS)
    parser.add_argument("--fold", required=True, type=int, choices=range(4))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", type=Path, default=OUTPUT)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    print(json.dumps({"output": str(run(args.method, args.fold, args.device, args.output))}))
