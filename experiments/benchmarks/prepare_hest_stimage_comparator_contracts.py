#!/usr/bin/env python3
"""Build the common HEST/STImage 50-gene comparator contracts.

The 224-pixel RGB cache is approximately 84 GB. It is deterministic and
contains no expression value. Compact
target and coordinate bundles plus all split contracts are archived on disk.
The builder is resumable at slide boundaries.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
from typing import Sequence

import anndata as ad
import h5py
import numpy as np
import pandas as pd
from PIL import Image

from lin_mnp.paths import ARCHIVE_ROOT, DATA_ROOT, RESULTS_ROOT

ARCHIVE = ARCHIVE_ROOT
HEST = DATA_ROOT / "hest"
STIMAGE = DATA_ROOT / "stimage"
STIMAGE_PROTOCOL = ARCHIVE / "stimage/protocol"
STIMAGE_TARGET = DATA_ROOT / "stimage/targets"
RUNTIME = RESULTS_ROOT / "runtime/hest_stimage"
OUTPUT = ARCHIVE / "benchmarks/hest_stimage/contracts"
HEST_DATASETS = ("IDC", "PRAD", "PAAD", "SKCM", "COAD", "READ", "CCRCC", "HCC", "LUNG", "LYMPH_IDC")
STIMAGE_TISSUES = ("breast", "brain", "skin", "mouth", "prostate", "stomach", "colon")
Image.MAX_IMAGE_PIXELS = None


def atomic_json(payload: object, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp-{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def decode(value: np.ndarray) -> np.ndarray:
    return np.asarray([
        item.decode() if isinstance(item, (bytes, np.bytes_)) else str(item)
        for item in np.asarray(value).reshape(-1)
    ])


def crop_context(image: Image.Image, x: float, y: float, radius: float) -> np.ndarray:
    side = max(16, int(round(8.0 * float(radius))))
    half = side // 2
    left, top = int(round(float(x))) - half, int(round(float(y))) - half
    right, bottom = left + side, top + side
    canvas = Image.new("RGB", (side, side), (255, 255, 255))
    source = (max(left, 0), max(top, 0), min(right, image.width), min(bottom, image.height))
    if source[0] < source[2] and source[1] < source[3]:
        region = image.crop(source).convert("RGB")
        canvas.paste(region, (source[0] - left, source[1] - top))
        region.close()
    resized = canvas.resize((224, 224), Image.Resampling.BICUBIC)
    value = np.asarray(resized, dtype=np.uint8).copy()
    resized.close(); canvas.close()
    return value


def hest_samples() -> list[dict[str, object]]:
    rows = []
    for dataset in HEST_DATASETS:
        samples = set()
        for path in (HEST / dataset / "splits").glob("*.csv"):
            samples.update(pd.read_csv(path)["sample_id"].astype(str))
        for sample in sorted(samples):
            path = HEST / dataset / "patches" / f"{sample}.h5"
            with h5py.File(path, "r") as archive:
                count = len(archive["barcode"])
            rows.append({"source": "HEST", "group": dataset, "sample": sample,
                         "count": count, "patch_path": str(path)})
    return rows


def hest_spatial_validation_tasks() -> list[dict[str, object]]:
    """Materialize the frozen one-slide spatial-quadrant inner splits.

    A virtual training slide contains the three non-held quadrants contiguously
    and a virtual validation slide contains the held quadrant.  Keeping these
    as two ordinary SlideBundle objects lets every external method use the same
    train/validation observations without adding model-specific mask logic.
    """

    rows: list[dict[str, object]] = []
    for dataset in HEST_DATASETS:
        split_root = HEST / dataset / "splits"
        for train_path in sorted(
            split_root.glob("train_*.csv"),
            key=lambda path: int(path.stem.split("_")[-1]),
        ):
            official_train = pd.read_csv(train_path)["sample_id"].astype(str).tolist()
            if len(official_train) != 1:
                continue
            fold = int(train_path.stem.split("_")[-1])
            original = official_train[0]
            patch_path = HEST / dataset / "patches" / f"{original}.h5"
            with h5py.File(patch_path, "r") as archive:
                coords = np.asarray(archive["coords"][:], dtype=np.float32)
            x_high = coords[:, 0] > np.median(coords[:, 0])
            y_high = coords[:, 1] > np.median(coords[:, 1])
            quadrant = x_high.astype(np.int8) + 2 * y_high.astype(np.int8)
            viable = [
                value for value in range(4)
                if int((quadrant == value).sum()) >= 9
                and int((quadrant != value).sum()) >= 9
            ]
            if not viable:
                raise ValueError(f"{dataset}/{original}: cannot form a spatial validation quadrant")
            for seed in (1, 2, 3):
                preferred = (seed - 1) % 4
                chosen = preferred if preferred in viable else viable[0]
                prefix = f"__benchmark_{dataset}_fold{fold}_seed{seed}_{original}"
                for role, indices in (
                    ("train", np.flatnonzero(quadrant != chosen)),
                    ("validation", np.flatnonzero(quadrant == chosen)),
                ):
                    rows.append({
                        "source": "HEST", "group": dataset,
                        "sample": f"{prefix}_{role}", "original_sample": original,
                        "count": int(len(indices)), "patch_path": str(patch_path),
                        "indices": indices.astype(np.int64), "fold": fold,
                        "seed": seed, "quadrant": int(chosen), "role": role,
                    })
    return rows


def stimage_samples() -> list[dict[str, object]]:
    rows = []
    for tissue in STIMAGE_TISSUES:
        protocol = json.loads((STIMAGE_PROTOCOL / f"{tissue}.json").read_text())
        for sample in protocol["gene_panel"]["slides"]:
            target_path = STIMAGE_TARGET / tissue / f"{sample}.npz"
            # The frozen source bundle stores barcode strings in an object array.
            # This local trusted artifact requires pickle for string decoding;
            # no expression value is used at this discovery stage.
            with np.load(target_path, allow_pickle=True) as target:
                count = len(target["barcodes"])
            rows.append({"source": "STImage", "group": tissue, "sample": sample,
                         "count": count, "target_path": str(target_path)})
    return rows


def build_patch_cache() -> Path:
    patch_root = RUNTIME / "patches_224_v1"
    manifest_path = patch_root / "manifest.json"
    if manifest_path.is_file():
        return patch_root
    patch_root.mkdir(parents=True, exist_ok=True)
    tasks = hest_samples() + hest_spatial_validation_tasks() + stimage_samples()
    seen = set()
    for row in tasks:
        sample = str(row["sample"])
        if sample in seen:
            raise ValueError(f"duplicate cross-benchmark sample id: {sample}")
        seen.add(sample)
    total = int(sum(int(row["count"]) for row in tasks))
    patch_path = patch_root / "patches_224_uint8.partial.npy"
    final_path = patch_root / "patches_224_uint8.npy"
    progress_path = patch_root / "progress.json"
    progress = json.loads(progress_path.read_text()) if progress_path.is_file() else {"completed": [], "rows": []}
    completed = set(progress["completed"])
    if final_path.is_file() and not manifest_path.is_file():
        raise FileExistsError("final patch array exists without manifest")
    if patch_path.is_file():
        patches = np.lib.format.open_memmap(patch_path, mode="r+")
        if patches.shape != (total, 224, 224, 3):
            raise ValueError("partial patch cache shape mismatch")
    else:
        patches = np.lib.format.open_memmap(
            patch_path, mode="w+", dtype=np.uint8, shape=(total, 224, 224, 3)
        )
    rows, offset = [], 0
    for task in tasks:
        sample, count = str(task["sample"]), int(task["count"])
        start, stop = offset, offset + count
        offset = stop
        if sample in completed:
            prior = next(row for row in progress["rows"] if row["sample_id"] == sample)
            if int(prior["start"]) != start or int(prior["stop"]) != stop:
                raise ValueError("resumed patch offset mismatch")
            rows.append(prior)
            continue
        if task["source"] == "HEST":
            with h5py.File(str(task["patch_path"]), "r") as archive:
                indices = task.get("indices")
                if indices is None:
                    barcodes = decode(archive["barcode"][:])
                    coords = np.asarray(archive["coords"][:], dtype=np.float32)
                    for begin in range(0, count, 256):
                        end = min(begin + 256, count)
                        patches[start + begin:start + end] = np.asarray(
                            archive["img"][begin:end], dtype=np.uint8
                        )
                else:
                    indices = np.asarray(indices, dtype=np.int64)
                    barcodes = decode(archive["barcode"][indices])
                    coords = np.asarray(archive["coords"][indices], dtype=np.float32)
                    for begin in range(0, count, 256):
                        end = min(begin + 256, count)
                        patches[start + begin:start + end] = np.asarray(
                            archive["img"][indices[begin:end]], dtype=np.uint8
                        )
        else:
            coordinate = pd.read_csv(
                STIMAGE / "Visium/coord" / f"{sample}_coord.csv", index_col=0
            )
            with np.load(str(task["target_path"]), allow_pickle=True) as target:
                barcodes = target["barcodes"].astype(str)
            missing = set(barcodes) - set(coordinate.index.astype(str))
            if missing:
                raise ValueError(f"{sample}: missing {len(missing)} target barcodes in coordinates")
            coordinate.index = coordinate.index.astype(str)
            coordinate = coordinate.loc[list(barcodes)]
            coords = coordinate[["xaxis", "yaxis"]].to_numpy(dtype=np.float32)
            radii = coordinate["r"].to_numpy(dtype=np.float32)
            with Image.open(STIMAGE / "Visium/image" / f"{sample}.png") as opened:
                image = opened.convert("RGB")
                for index, ((x, y), radius) in enumerate(zip(coords, radii)):
                    patches[start + index] = crop_context(image, x, y, radius)
                image.close()
        row = {
            "sample_id": sample, "patient_id": sample, "source": task["source"],
            "group": task["group"], "start": start, "stop": stop, "n_spots": count,
            "barcodes": barcodes.astype(str).tolist(),
        }
        if "original_sample" in task:
            row.update({
                "original_sample_id": str(task["original_sample"]),
                "spatial_quadrant": int(task["quadrant"]),
                "virtual_role": str(task["role"]),
                "seed": int(task["seed"]), "fold": int(task["fold"]),
            })
        rows.append(row)
        completed.add(sample)
        patches.flush()
        atomic_json({"completed": sorted(completed), "rows": rows}, progress_path)
        print(json.dumps({"status": "PATCH_COMPLETE", "sample": sample,
                          "source": task["source"], "spots": count,
                          "completed": len(completed), "total_slides": len(tasks)}), flush=True)
    if offset != total or len(rows) != len(tasks):
        raise ValueError("patch cache completion mismatch")
    patches.flush(); del patches
    os.replace(patch_path, final_path)
    manifest = {
        "schema_version": 1, "artifact_id": "hest_stimage_rgb224_v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "target_free": True, "expression_values_accessed": False,
        "patch_size": 224, "patch_layout": "NHWC RGB uint8",
        "patch_file": final_path.name, "n_spots": total, "slides": rows,
        "crop_contract": {
            "HEST": "publisher-deposited RGB224 patch",
            "STImage": "eight-radius square, white padding, bicubic resize to RGB224",
        },
    }
    atomic_json(manifest, manifest_path)
    return patch_root


def hest_patient_map(dataset: str) -> dict[str, str]:
    mapping = {}
    for path in sorted((HEST / dataset / "splits").glob("test_*.csv")):
        patient = f"official_patient_fold_{path.stem.split('_')[-1]}"
        for sample in pd.read_csv(path)["sample_id"].astype(str):
            if sample in mapping:
                raise ValueError(f"{dataset}: duplicate official test sample {sample}")
            mapping[sample] = patient
    return mapping


def prepare_hest(patch_root: Path) -> None:
    for dataset in HEST_DATASETS:
        root = OUTPUT / "hest" / dataset
        target_root, coordinate_root = root / "target", root / "coordinates"
        target_root.mkdir(parents=True, exist_ok=True)
        coordinate_root.mkdir(parents=True, exist_ok=True)
        genes = tuple(json.loads((HEST / dataset / "var_50genes.json").read_text())["genes"])
        (target_root / "gene_panel.txt").write_text("\n".join(genes) + "\n")
        samples = sorted(hest_patient_map(dataset))
        patient_map = hest_patient_map(dataset)
        for sample in samples:
            target_path, coordinate_path = target_root / "slides" / f"{sample}.npz", coordinate_root / f"{sample}.npz"
            if target_path.is_file() and coordinate_path.is_file():
                continue
            target_path.parent.mkdir(parents=True, exist_ok=True)
            with h5py.File(HEST / dataset / "patches" / f"{sample}.h5", "r") as patch:
                barcodes = decode(patch["barcode"][:])
                coords = np.asarray(patch["coords"][:], dtype=np.float32)
            value = ad.read_h5ad(HEST / dataset / "adata" / f"{sample}.h5ad")
            missing = set(barcodes) - set(value.obs_names.astype(str))
            if missing or any(gene not in value.var_names for gene in genes):
                raise ValueError(f"{dataset}/{sample}: target identity mismatch")
            matrix = value[list(barcodes), list(genes)].X
            matrix = matrix.toarray() if hasattr(matrix, "toarray") else np.asarray(matrix)
            target = np.log1p(np.asarray(matrix, dtype=np.float32))
            obs = value.obs.loc[list(barcodes)]
            if {"array_row", "array_col"}.issubset(obs):
                array_coords = obs[["array_row", "array_col"]].to_numpy(dtype=np.int64)
            else:
                array_coords = np.rint(coords).astype(np.int64)
            with target_path.open("wb") as handle:
                np.savez_compressed(
                    handle, target=target, barcodes=barcodes, genes=np.asarray(genes),
                    patient_id=np.asarray(patient_map[sample]), sample_id=np.asarray(sample),
                )
            with coordinate_path.open("wb") as handle:
                np.savez_compressed(handle, barcodes=barcodes, coords=coords, array_coords=array_coords)
        virtual_by_key: dict[tuple[int, int], dict[str, str]] = {}
        for task in hest_spatial_validation_tasks():
            if str(task["group"]) != dataset:
                continue
            virtual = str(task["sample"])
            original = str(task["original_sample"])
            indices = np.asarray(task["indices"], dtype=np.int64)
            target_path = target_root / "slides" / f"{virtual}.npz"
            coordinate_path = coordinate_root / f"{virtual}.npz"
            with np.load(target_root / "slides" / f"{original}.npz", allow_pickle=False) as source:
                target = source["target"][indices].astype(np.float32)
                barcodes = source["barcodes"][indices].astype(str)
            with np.load(coordinate_root / f"{original}.npz", allow_pickle=False) as source:
                coords = source["coords"][indices].astype(np.float32)
                array_coords = source["array_coords"][indices].astype(np.int64)
            if not target_path.is_file():
                with target_path.open("wb") as handle:
                    np.savez_compressed(
                        handle, target=target, barcodes=barcodes,
                        genes=np.asarray(genes), patient_id=np.asarray(patient_map[original]),
                        sample_id=np.asarray(virtual), original_sample_id=np.asarray(original),
                    )
            if not coordinate_path.is_file():
                with coordinate_path.open("wb") as handle:
                    np.savez_compressed(
                        handle, barcodes=barcodes, coords=coords,
                        array_coords=array_coords,
                    )
            key = (int(task["fold"]), int(task["seed"]))
            virtual_by_key.setdefault(key, {})[str(task["role"])] = virtual
        fold_count = len(list((HEST / dataset / "splits").glob("test_*.csv")))
        for seed in (1, 2, 3):
            folds = []
            for fold in range(fold_count):
                official_train = pd.read_csv(HEST / dataset / "splits" / f"train_{fold}.csv")["sample_id"].astype(str).tolist()
                test = pd.read_csv(HEST / dataset / "splits" / f"test_{fold}.csv")["sample_id"].astype(str).tolist()
                patients = list(dict.fromkeys(patient_map[sample] for sample in official_train))
                if len(patients) >= 2:
                    held = patients[(seed - 1) % len(patients)]
                    validation = [sample for sample in official_train if patient_map[sample] == held]
                    train = [sample for sample in official_train if patient_map[sample] != held]
                    selection = "hold-out-one-official-training-patient"
                elif len(official_train) >= 2:
                    validation = [sorted(official_train)[(seed - 1) % len(official_train)]]
                    train = [sample for sample in official_train if sample not in validation]
                    selection = "held training slide"
                else:
                    split = virtual_by_key[(fold, seed)]
                    train = [split["train"]]
                    validation = [split["validation"]]
                    selection = "spatial quadrant of the single official training slide"
                expanded_patient_map = dict(patient_map)
                for sample in train + validation:
                    if sample.startswith("__benchmark_"):
                        expanded_patient_map[sample] = patient_map[official_train[0]]
                folds.append({
                    "fold_id": fold, "train_sample_ids": train,
                    "val_sample_ids": validation, "test_sample_ids": test,
                    "train_patient_ids": sorted({expanded_patient_map[sample] for sample in train}),
                    "val_patient_id": expanded_patient_map[validation[0]],
                    "test_patient_id": patient_map[test[0]],
                    "selection_contract": selection,
                })
            contract = {
                "schema_version": 1, "cohort": f"HEST/{dataset}/seed{seed}",
                "protocol": "HEST 50-gene training-only selection",
                "target_bundle": str(target_root), "patch_root": str(patch_root),
                "coordinate_root": str(coordinate_root),
                "section_ids": sorted(set(samples).union(
                    sample for fold_row in folds
                    for sample in fold_row["train_sample_ids"] + fold_row["val_sample_ids"]
                )),
                "patient_by_section": {
                    **patient_map,
                    **{
                        sample: patient_map[str(sample).split("_seed", 1)[1].split("_", 1)[1].rsplit("_", 1)[0]]
                        for fold_row in folds
                        for sample in fold_row["train_sample_ids"] + fold_row["val_sample_ids"]
                        if str(sample).startswith("__benchmark_")
                    },
                }, "fold_count": fold_count,
                "folds": folds, "seed": seed,
            }
            atomic_json(contract, root / f"seed{seed}.json")
        print(json.dumps({"status": "HEST_CONTRACT_COMPLETE", "dataset": dataset}), flush=True)


def prepare_stimage(patch_root: Path) -> None:
    for tissue in STIMAGE_TISSUES:
        protocol = json.loads((STIMAGE_PROTOCOL / f"{tissue}.json").read_text())
        root = OUTPUT / "stimage" / tissue
        target_root, coordinate_root = root / "target", root / "coordinates"
        target_root.mkdir(parents=True, exist_ok=True)
        coordinate_root.mkdir(parents=True, exist_ok=True)
        genes = tuple(protocol["gene_panel"]["genes"])
        samples = tuple(protocol["gene_panel"]["slides"])
        (target_root / "gene_panel.txt").write_text("\n".join(genes) + "\n")
        for sample in samples:
            target_path, coordinate_path = target_root / "slides" / f"{sample}.npz", coordinate_root / f"{sample}.npz"
            if target_path.is_file() and coordinate_path.is_file():
                continue
            target_path.parent.mkdir(parents=True, exist_ok=True)
            with np.load(STIMAGE_TARGET / tissue / f"{sample}.npz", allow_pickle=True) as source:
                target = source["target"].astype(np.float32)
                barcodes = source["barcodes"].astype(str)
                source_genes = tuple(source["genes"].astype(str))
            if source_genes != genes:
                raise ValueError(f"{tissue}/{sample}: frozen target gene mismatch")
            coordinate = pd.read_csv(STIMAGE / "Visium/coord" / f"{sample}_coord.csv", index_col=0)
            coordinate.index = coordinate.index.astype(str)
            if set(barcodes) - set(coordinate.index):
                raise ValueError(f"{tissue}/{sample}: coordinate barcode mismatch")
            coordinate = coordinate.loc[list(barcodes)]
            coords = coordinate[["xaxis", "yaxis"]].to_numpy(dtype=np.float32)
            array_columns = [value for value in ("x", "y", "array_row", "array_col") if value in coordinate]
            array_coords = (
                coordinate[array_columns[:2]].to_numpy(dtype=np.int64)
                if len(array_columns) >= 2 else np.rint(coords).astype(np.int64)
            )
            with target_path.open("wb") as handle:
                np.savez_compressed(
                    handle, target=target, barcodes=barcodes, genes=np.asarray(genes),
                    patient_id=np.asarray(sample), sample_id=np.asarray(sample),
                )
            with coordinate_path.open("wb") as handle:
                np.savez_compressed(handle, barcodes=barcodes, coords=coords, array_coords=array_coords)
        for seed_text, split in protocol["splits"].items():
            seed = int(seed_text)
            contract = {
                "schema_version": 1, "cohort": f"STImage/{tissue}/seed{seed}",
                "protocol": "STImage fixed 8:1:1 50-gene split",
                "target_bundle": str(target_root), "patch_root": str(patch_root),
                "coordinate_root": str(coordinate_root), "section_ids": list(samples),
                "patient_by_section": {sample: sample for sample in samples},
                "fold_count": 1, "seed": seed,
                "folds": [{
                    "fold_id": 0, "train_sample_ids": split["train"],
                    "val_sample_ids": split["validation"], "test_sample_ids": split["test"],
                    "train_patient_ids": split["train"],
                    "val_patient_id": "validation_slides", "test_patient_id": "test_slides",
                    "selection_contract": "validation selects checkpoint; test opened once",
                }],
            }
            atomic_json(contract, root / f"seed{seed}.json")
        print(json.dumps({"status": "STIMAGE_CONTRACT_COMPLETE", "tissue": tissue}), flush=True)


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("patches", "contracts", "all"), default="all")
    args = parser.parse_args(argv)
    patch_root = build_patch_cache() if args.stage in {"patches", "all"} else RUNTIME / "patches_224_v1"
    if args.stage in {"contracts", "all"}:
        if not (patch_root / "manifest.json").is_file():
            raise FileNotFoundError("complete patch cache required before contracts")
        prepare_hest(patch_root)
        prepare_stimage(patch_root)
        atomic_json({
            "schema_version": 1, "status": "COMPLETE",
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "patch_root": str(patch_root), "hest_datasets": HEST_DATASETS,
            "stimage_tissues": STIMAGE_TISSUES,
        }, OUTPUT / "manifest.json")


if __name__ == "__main__":
    main()
