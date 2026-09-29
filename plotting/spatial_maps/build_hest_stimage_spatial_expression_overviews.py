#!/usr/bin/env python3
"""Build selected-gene spatial maps for HEST and STImage.

The renderer extends the established Lin-MNP spatial-expression overview to
the two external benchmarks.  It uses one deterministic held-out seed-1/fold-0
sample per organ/group, the latest Lin-MNP prediction, and every available
comparison-method prediction. Selection is descriptive and test based;
it is never used as a cohort-level performance estimate.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import gzip
import hashlib
import json
from pathlib import Path
import shutil
from typing import Any, Iterable

import anndata as ad
import cv2
import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import numpy as np
import pandas as pd
from PIL import Image
from scipy.spatial import cKDTree

from lin_mnp.paths import ARCHIVE_ROOT, DATA_ROOT, RESULTS_ROOT

# Several archived STImage slides are genuine whole-slide PNGs larger than
# Pillow's generic anti-decompression-bomb threshold.  Paths are fixed local
# benchmark assets, not untrusted uploads.
Image.MAX_IMAGE_PIXELS = None


OUTPUT_ROOT = RESULTS_ROOT / "figures/spatial_maps"
CAMPAIGN_ROOT = (
    ARCHIVE_ROOT / "benchmarks/hest_stimage"
)
LATEST_ROOTS = {
    "hest": ARCHIVE_ROOT / "transfer/hest/uni_v1",
    "stimage": ARCHIVE_ROOT / "transfer/stimage/uni_v1",
}
HEST_DATA_ROOT = DATA_ROOT / "hest"
STIMAGE_DATA_ROOT = DATA_ROOT / "stimage/Visium"

METHOD_IDS = (
    "hespotex",
    "he2gene",
    "asgnn",
    "gr2st",
    "heclip",
    "tcgn",
    "omiclip",
    "igidl",
    "bleep",
    "hist2st",
    "thitogene",
    "mclstexp",
    "stnet",
    "deepspace",
    "deeppt",
    "histogene",
    "egnv1",
    "egnv2",
    "genecoder",
    "istar",
    "holotea",
    "spahgc",
    "stag",
)
DISPLAY_NAMES = {
    "Lin-MNP": "Lin-MNP",
    "hespotex": "HESpotEx",
    "he2gene": "HE2Gene",
    "asgnn": "asGNN",
    "gr2st": "GR2ST",
    "heclip": "HECLIP",
    "tcgn": "TCGN",
    "omiclip": "OmiCLIP",
    "igidl": "IGI-DL",
    "bleep": "BLEEP",
    "hist2st": "Hist2ST",
    "thitogene": "THItoGene",
    "mclstexp": "mclSTExp",
    "stnet": "ST-Net",
    "deepspace": "DeepSpaCE",
    "deeppt": "DeepPT",
    "histogene": "HisToGene",
    "egnv1": "EGNv1",
    "egnv2": "EGNv2",
    "genecoder": "GeneCodeR",
    "istar": "iStar",
    "holotea": "HoloTea",
    "spahgc": "SpaHGC",
    "stag": "STAG",
}
BENCHMARK_LABELS = {"hest": "HEST", "stimage": "STImage"}
OUTPUT_DIRS = {
    "hest": "hest_external_50gene",
    "stimage": "stimage_external_50gene",
}


def decode(values: np.ndarray) -> tuple[str, ...]:
    return tuple(
        value.decode("utf-8") if isinstance(value, bytes) else str(value)
        for value in np.asarray(values).reshape(-1)
    )


def safe_name(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in value)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def pearson(left: np.ndarray, right: np.ndarray) -> float:
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    valid = np.isfinite(left) & np.isfinite(right)
    left = left[valid]
    right = right[valid]
    if len(left) < 2 or np.std(left) <= 1e-12 or np.std(right) <= 1e-12:
        return float("nan")
    return float(np.corrcoef(left, right)[0, 1])


def minmax(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    finite = np.isfinite(values)
    output = np.zeros(values.shape, dtype=np.float64)
    if not finite.any():
        return output
    low = float(np.nanmin(values[finite]))
    high = float(np.nanmax(values[finite]))
    if high > low:
        output[finite] = (values[finite] - low) / (high - low)
    return output


def enhance_hest_mosaic_for_display(image: Image.Image) -> Image.Image:
    """Increase H&E visibility without changing spot geometry or predictions.

    The publisher-deposited downscaled HEST image is lighter than the cSCC H&E
    crop at panel scale. White-balance each channel, amplify optical density,
    then apply modest colour and unsharp enhancement. This transform is used
    only by the H&E display panel; expression maps, coordinates and all PCC
    values remain untouched.
    """

    array = np.asarray(image.convert("RGB"), dtype=np.float32)
    white = np.percentile(array.reshape(-1, 3), 99.0, axis=0)
    white = np.maximum(white, 1.0)
    relative = np.clip((array + 1.0) / (white[None, None, :] + 1.0), 1e-4, 1.0)
    optical_density = -np.log(relative) * 1.25
    enhanced = white[None, None, :] * np.exp(-optical_density)
    enhanced = np.clip(enhanced, 0.0, 255.0).astype(np.uint8)

    hsv = cv2.cvtColor(enhanced, cv2.COLOR_RGB2HSV).astype(np.float32)
    hsv[..., 1] = np.clip(hsv[..., 1] * 1.12, 0.0, 255.0)
    coloured = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)
    blurred = cv2.GaussianBlur(coloured, (0, 0), sigmaX=0.9, sigmaY=0.9)
    sharpened = cv2.addWeighted(coloured, 1.18, blurred, -0.18, 0.0)
    return Image.fromarray(np.clip(sharpened, 0, 255).astype(np.uint8), mode="RGB")


def aligned_matrix(
    path: Path,
    target_barcodes: tuple[str, ...],
    target_genes: tuple[str, ...],
) -> np.ndarray:
    with np.load(path, allow_pickle=False) as archive:
        matrix = np.asarray(archive["prediction"], dtype=np.float64)
        barcodes = decode(archive["barcodes"])
        genes = decode(archive["genes"])
    if set(barcodes) != set(target_barcodes) or set(genes) != set(target_genes):
        raise ValueError(f"prediction identity mismatch: {path}")
    row = {value: index for index, value in enumerate(barcodes)}
    col = {value: index for index, value in enumerate(genes)}
    result = matrix[
        np.asarray([row[value] for value in target_barcodes]),
        :,
    ][:, np.asarray([col[value] for value in target_genes])]
    if not np.isfinite(result).all():
        raise ValueError(f"non-finite prediction: {path}")
    return result


def representative_records(benchmark: str) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    contract_root = CAMPAIGN_ROOT / "contracts" / benchmark
    for contract_path in sorted(contract_root.glob("*/seed1.json")):
        contract = json.loads(contract_path.read_text(encoding="utf-8"))
        group = contract_path.parent.name
        fold = contract["folds"][0]
        sample = str(fold["test_sample_ids"][0])
        records.append(
            {
                "benchmark": benchmark,
                "group": group,
                "sample": sample,
                "fold_id": int(fold["fold_id"]),
                "contract_path": contract_path,
                "target_path": contract_path.parent
                / "target/slides"
                / f"{sample}.npz",
                "coordinate_path": contract_path.parent
                / "coordinates"
                / f"{sample}.npz",
            }
        )
    return records


def latest_prediction(record: dict[str, Any]) -> tuple[np.ndarray, Path]:
    benchmark = record["benchmark"]
    group = record["group"]
    sample = record["sample"]
    if benchmark == "hest":
        paths = sorted((LATEST_ROOTS[benchmark] / group).glob("fold*/predictions.npz"))
    else:
        paths = [LATEST_ROOTS[benchmark] / group / "seed1/predictions.npz"]
    matches: list[tuple[np.ndarray, Path]] = []
    for path in paths:
        with np.load(path, allow_pickle=False) as archive:
            sections = np.asarray(decode(archive["sections"]))
            active = sections == sample
            if not active.any():
                continue
            prediction = np.asarray(archive["lin_mnp"], dtype=np.float64)[active]
            barcodes = np.asarray(decode(archive["barcodes"]))[active]
            genes = decode(archive["genes"])
        target = load_target(record)
        row = {value: index for index, value in enumerate(barcodes.tolist())}
        col = {value: index for index, value in enumerate(genes)}
        if set(row) != set(target["barcodes"]) or set(col) != set(target["genes"]):
            raise ValueError(f"latest prediction identity mismatch: {path}/{sample}")
        aligned = prediction[
            np.asarray([row[value] for value in target["barcodes"]]), :
        ][:, np.asarray([col[value] for value in target["genes"]])]
        matches.append((aligned, path))
    if len(matches) != 1:
        raise ValueError(f"expected one latest prediction for {benchmark}/{group}/{sample}")
    return matches[0]


def load_target(record: dict[str, Any]) -> dict[str, Any]:
    with np.load(record["target_path"], allow_pickle=False) as archive:
        target = np.asarray(archive["target"], dtype=np.float64)
        barcodes = decode(archive["barcodes"])
        genes = decode(archive["genes"])
    if target.shape != (len(barcodes), len(genes)):
        raise ValueError(f"target shape mismatch: {record['target_path']}")
    return {"target": target, "barcodes": barcodes, "genes": genes}


def load_record(record: dict[str, Any]) -> dict[str, Any]:
    target = load_target(record)
    latest, latest_path = latest_prediction(record)
    predictions: dict[str, np.ndarray] = {"Lin-MNP": latest}
    paths: dict[str, Path] = {"Lin-MNP": latest_path}
    for method in METHOD_IDS:
        path = (
            CAMPAIGN_ROOT
            / "runs"
            / record["benchmark"]
            / record["group"]
            / "seed1"
            / method
            / f"fold{record['fold_id']}"
            / "predictions"
            / f"{record['sample']}.npz"
        )
        if not path.is_file():
            raise FileNotFoundError(path)
        predictions[method] = aligned_matrix(path, target["barcodes"], target["genes"])
        paths[method] = path
    return {**record, **target, "predictions": predictions, "paths": paths}


def candidate_table(records: list[dict[str, Any]]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for record in records:
        loaded = load_record(record)
        for gene_index, gene in enumerate(loaded["genes"]):
            truth = loaded["target"][:, gene_index]
            pccs = {
                method: pearson(truth, matrix[:, gene_index])
                for method, matrix in loaded["predictions"].items()
            }
            finite_baselines = {
                method: value
                for method, value in pccs.items()
                if method != "Lin-MNP" and np.isfinite(value)
            }
            best_method = max(finite_baselines, key=finite_baselines.get)
            rows.append(
                {
                    "benchmark": loaded["benchmark"],
                    "group": loaded["group"],
                    "sample": loaded["sample"],
                    "fold_id": loaded["fold_id"],
                    "gene": gene,
                    "lin_mnp_pcc": pccs["Lin-MNP"],
                    "best_baseline_method": DISPLAY_NAMES[best_method],
                    "best_baseline_pcc": finite_baselines[best_method],
                    "pcc_margin": pccs["Lin-MNP"] - finite_baselines[best_method],
                }
            )
    result = pd.DataFrame(rows)
    result["lin_mnp_rank_within_sample"] = result.groupby(
        ["benchmark", "group", "sample"]
    )["lin_mnp_pcc"].rank(ascending=False, method="first")
    return result.sort_values(
        ["benchmark", "group", "lin_mnp_rank_within_sample"], kind="stable"
    ).reset_index(drop=True)


def select_genes(candidates: pd.DataFrame) -> pd.DataFrame:
    selected: dict[tuple[str, str, str, str], dict[str, Any]] = {}

    def add(row: pd.Series, category: str, rationale: str) -> None:
        key = (
            str(row["benchmark"]),
            str(row["group"]),
            str(row["sample"]),
            str(row["gene"]),
        )
        if key not in selected:
            selected[key] = {**row.to_dict(), "categories": [], "rationales": []}
        selected[key]["categories"].append(category)
        selected[key]["rationales"].append(rationale)

    for benchmark, current in candidates.groupby("benchmark", sort=True):
        qualifying = current.loc[
            current["pcc_margin"].gt(0)
            & current["lin_mnp_rank_within_sample"].le(10)
        ].sort_values(["pcc_margin", "lin_mnp_pcc"], ascending=False)
        for _, row in qualifying.head(10).iterrows():
            add(
                row,
                "benchmark_top10_advantage",
                "Top-10 Lin-MNP gene within its representative sample and positive margin over every baseline; among these, one of the ten largest benchmark-wide margins.",
            )

        outside_top = current.loc[
            current["pcc_margin"].gt(0)
            & current["lin_mnp_rank_within_sample"].gt(10)
        ].sort_values(["pcc_margin", "lin_mnp_pcc"], ascending=False)
        for _, row in outside_top.head(5).iterrows():
            add(
                row,
                "benchmark_large_margin",
                "Outside the within-sample Lin-MNP top 10 but among the five largest positive benchmark-wide margins over every baseline.",
            )

        for group, group_rows in current.groupby("group", sort=True):
            winners = group_rows.loc[group_rows["pcc_margin"].gt(0)].sort_values(
                ["pcc_margin", "lin_mnp_pcc"], ascending=False
            )
            if len(winners):
                add(
                    winners.iloc[0],
                    "group_best_advantage",
                    f"Largest Lin-MNP margin over every baseline in the {group} representative held-out sample.",
                )
            else:
                control = group_rows.sort_values("lin_mnp_pcc", ascending=False).iloc[0]
                add(
                    control,
                    "group_best_control_no_win",
                    f"Highest-PCC Lin-MNP gene in {group}; retained as an explicit control because Lin-MNP does not beat the strongest baseline on any of this sample's 50 genes.",
                )

    rows = []
    for value in selected.values():
        value["selection_category"] = ";".join(value.pop("categories"))
        value["selection_rationale"] = " ".join(value.pop("rationales"))
        rows.append(value)
    return pd.DataFrame(rows).sort_values(
        ["benchmark", "group", "lin_mnp_rank_within_sample", "gene"], kind="stable"
    ).reset_index(drop=True)


def median_spacing(coords: np.ndarray) -> float:
    if len(coords) < 2:
        return 1.0
    distances, _ = cKDTree(coords).query(coords, k=2)
    return float(np.median(distances[:, 1]))


def decode_h5_barcodes(values: np.ndarray) -> tuple[str, ...]:
    output = []
    for value in np.asarray(values).reshape(-1):
        if isinstance(value, bytes):
            output.append(value.decode("utf-8"))
        else:
            output.append(str(value))
    return tuple(output)


def build_hest_mosaic(record: dict[str, Any], cache_root: Path) -> tuple[Image.Image, np.ndarray, float, Path]:
    group = record["group"]
    sample = record["sample"]
    adata_path = HEST_DATA_ROOT / group / "adata" / f"{sample}.h5ad"
    h5_path = HEST_DATA_ROOT / group / "patches" / f"{sample}.h5"
    cache_root.mkdir(parents=True, exist_ok=True)
    image_path = cache_root / f"{safe_name(group)}__{safe_name(sample)}.png"
    geometry_path = cache_root / f"{safe_name(group)}__{safe_name(sample)}.npz"
    with np.load(record["coordinate_path"], allow_pickle=False) as archive:
        coord_barcodes = decode(archive["barcodes"])
        coords = np.asarray(archive["coords"], dtype=np.float64)
    target = load_target(record)
    coord_index = {value: index for index, value in enumerate(coord_barcodes)}
    coords = coords[np.asarray([coord_index[value] for value in target["barcodes"]])]

    if image_path.is_file() and geometry_path.is_file():
        with np.load(geometry_path, allow_pickle=False) as archive:
            background_kind = (
                str(archive["background_kind"].item())
                if "background_kind" in archive.files
                else "legacy_patch_mosaic"
            )
            if background_kind == "h5ad_downscaled_fullres":
                local = np.asarray(archive["local_coordinates"], dtype=np.float64)
                spot_diameter = float(archive["spot_diameter"].item())
                return Image.open(image_path).convert("RGB"), local, spot_diameter, image_path

    # HEST h5ad files include a clean publisher-deposited low-resolution H&E
    # image and the matching full-resolution spot coordinates. Prefer this
    # source over reconstructing a tissue image from overlapping spot patches.
    if adata_path.is_file():
        adata = ad.read_h5ad(adata_path, backed="r")
        try:
            spatial_payload = adata.uns.get("spatial", {})
            library_id = next(iter(spatial_payload), None)
            library = spatial_payload.get(library_id, {}) if library_id else {}
            images = library.get("images", {})
            scalefactors = library.get("scalefactors", {})
            if "downscaled_fullres" in images:
                full_image = Image.fromarray(
                    np.asarray(images["downscaled_fullres"], dtype=np.uint8),
                    mode="RGB",
                )
                coordinate_scale = float(
                    scalefactors["tissue_downscaled_fullres_scalef"]
                )
                full_spot_diameter = float(scalefactors["spot_diameter_fullres"])
                adata_barcodes = tuple(str(value) for value in adata.obs_names)
                adata_index = {value: index for index, value in enumerate(adata_barcodes)}
                missing = [
                    value for value in target["barcodes"] if value not in adata_index
                ]
                if missing:
                    raise ValueError(
                        f"{sample}: {len(missing)} target barcodes are absent from {adata_path}"
                    )
                spatial_coordinates = np.asarray(adata.obsm["spatial"], dtype=np.float64)
                selected = np.asarray(
                    [adata_index[value] for value in target["barcodes"]], dtype=np.int64
                )
                image_coordinates = spatial_coordinates[selected] * coordinate_scale

                scaled_diameter = full_spot_diameter * coordinate_scale
                padding = max(1.8 * scaled_diameter, 8.0)
                left = max(0, int(np.floor(image_coordinates[:, 0].min() - padding)))
                top = max(0, int(np.floor(image_coordinates[:, 1].min() - padding)))
                right = min(
                    full_image.width,
                    int(np.ceil(image_coordinates[:, 0].max() + padding)),
                )
                bottom = min(
                    full_image.height,
                    int(np.ceil(image_coordinates[:, 1].max() + padding)),
                )
                image = full_image.crop((left, top, right, bottom))
                local = image_coordinates - np.asarray([left, top], dtype=np.float64)
                spot_diameter = scaled_diameter
                image.save(image_path, optimize=True)
                np.savez_compressed(
                    geometry_path,
                    local_coordinates=local,
                    spot_diameter=np.asarray(spot_diameter),
                    source_h5ad=np.asarray(str(adata_path.resolve())),
                    embedded_image_key=np.asarray(
                        f"spatial/{library_id}/images/downscaled_fullres"
                    ),
                    coordinate_scale=np.asarray(coordinate_scale),
                    crop_origin=np.asarray([left, top], dtype=np.float64),
                    background_kind=np.asarray("h5ad_downscaled_fullres"),
                )
                return image, local, spot_diameter, image_path
        finally:
            adata.file.close()

    padding = 224.0
    low = coords.min(axis=0) - padding
    high = coords.max(axis=0) + padding
    span = high - low
    scale = min(1600.0 / max(span), 1.0)
    width = max(32, int(np.ceil(span[0] * scale)))
    height = max(32, int(np.ceil(span[1] * scale)))
    thumb = max(3, int(round(224.0 * scale)))
    accumulator = np.zeros((height, width, 3), dtype=np.float32)
    weight = np.zeros((height, width, 1), dtype=np.float32)

    with h5py.File(h5_path, "r") as handle:
        h5_barcodes = decode_h5_barcodes(handle["barcode"][:])
        h5_index = {value: index for index, value in enumerate(h5_barcodes)}
        for spot_index, barcode in enumerate(target["barcodes"]):
            patch = np.asarray(handle["img"][h5_index[barcode]], dtype=np.uint8)
            small = cv2.resize(patch, (thumb, thumb), interpolation=cv2.INTER_AREA)
            center = (coords[spot_index] - low) * scale
            x0 = int(round(center[0] - thumb / 2))
            y0 = int(round(center[1] - thumb / 2))
            x1 = min(width, x0 + thumb)
            y1 = min(height, y0 + thumb)
            sx0 = max(0, -x0)
            sy0 = max(0, -y0)
            x0 = max(0, x0)
            y0 = max(0, y0)
            if x1 <= x0 or y1 <= y0:
                continue
            block = small[sy0 : sy0 + (y1 - y0), sx0 : sx0 + (x1 - x0)]
            accumulator[y0:y1, x0:x1] += block
            weight[y0:y1, x0:x1] += 1.0
    canvas = np.full((height, width, 3), 246.0, dtype=np.float32)
    active = weight[..., 0] > 0
    canvas[active] = accumulator[active] / weight[active]
    image = Image.fromarray(np.clip(canvas, 0, 255).astype(np.uint8), mode="RGB")
    image.save(image_path, optimize=True)
    local = (coords - low) * scale
    spot_diameter = median_spacing(local) * 0.72
    np.savez_compressed(
        geometry_path,
        local_coordinates=local,
        spot_diameter=np.asarray(spot_diameter),
        source_h5=np.asarray(str(h5_path.resolve())),
        scale=np.asarray(scale),
        origin=low,
        background_kind=np.asarray("spot_patch_mosaic"),
    )
    return image, local, spot_diameter, image_path


def load_stimage_geometry(record: dict[str, Any]) -> tuple[Image.Image, np.ndarray, float, Path]:
    sample = record["sample"]
    image_path = STIMAGE_DATA_ROOT / "image" / f"{sample}.png"
    image = Image.open(image_path).convert("RGB")
    with np.load(record["coordinate_path"], allow_pickle=False) as archive:
        coord_barcodes = decode(archive["barcodes"])
        coords = np.asarray(archive["coords"], dtype=np.float64)
    target = load_target(record)
    index = {value: idx for idx, value in enumerate(coord_barcodes)}
    coords = coords[np.asarray([index[value] for value in target["barcodes"]])]
    spacing = median_spacing(coords)
    padding = max(1.8 * spacing, 20.0)
    left = max(0, int(np.floor(coords[:, 0].min() - padding)))
    top = max(0, int(np.floor(coords[:, 1].min() - padding)))
    right = min(image.width, int(np.ceil(coords[:, 0].max() + padding)))
    bottom = min(image.height, int(np.ceil(coords[:, 1].max() + padding)))
    crop = image.crop((left, top, right, bottom))
    image.close()
    local = coords - np.asarray([left, top])
    scale = min(1600.0 / max(crop.size), 1.0)
    if scale < 1.0:
        resized = (
            max(1, int(round(crop.width * scale))),
            max(1, int(round(crop.height * scale))),
        )
        crop = crop.resize(resized, resample=Image.Resampling.LANCZOS)
        local = local * scale
    return crop, local, spacing * scale * 0.72, image_path


def draw_expression(
    axis: plt.Axes,
    background: Image.Image,
    coordinates: np.ndarray,
    values: np.ndarray,
    spot_diameter: float,
    title: str,
    pcc: float | None,
) -> None:
    axis.imshow(background, alpha=0.60)
    diameter = float(np.clip(210.0 * spot_diameter / max(background.size), 4.2, 11.0))
    axis.scatter(
        coordinates[:, 0],
        coordinates[:, 1],
        c=minmax(values),
        cmap="viridis",
        norm=Normalize(0.0, 1.0),
        s=diameter**2,
        edgecolors=(1.0, 1.0, 1.0, 0.52),
        linewidths=0.18,
        alpha=0.99,
        rasterized=True,
    )
    suffix = "" if pcc is None else (f"\nPCC = {pcc:.3f}" if np.isfinite(pcc) else "\nPCC = N/A")
    axis.set_title(f"{title}{suffix}", fontsize=17, pad=8)
    axis.set_xlim(0, background.width)
    axis.set_ylim(background.height, 0)
    axis.set_aspect("equal")
    axis.axis("off")


def plot_one(
    row: pd.Series,
    loaded: dict[str, Any],
    output: Path,
    hest_cache: Path,
    dpi: int,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    gene = str(row.gene)
    gene_index = loaded["genes"].index(gene)
    truth = loaded["target"][:, gene_index]
    if loaded["benchmark"] == "hest":
        background, coordinates, diameter, image_path = build_hest_mosaic(loaded, hest_cache)
        he_background = enhance_hest_mosaic_for_display(background)
        image_title = "H&E image"
    else:
        background, coordinates, diameter, image_path = load_stimage_geometry(loaded)
        he_background = background
        image_title = "H&E image"
    if len(coordinates) != len(truth):
        raise ValueError("geometry/target length mismatch")

    methods = ("Lin-MNP",) + METHOD_IDS
    pccs = {
        method: pearson(truth, loaded["predictions"][method][:, gene_index])
        for method in methods
    }
    if not np.isclose(pccs["Lin-MNP"], float(row.lin_mnp_pcc), atol=2e-6):
        raise ValueError(f"selection/render PCC mismatch: {loaded['benchmark']}/{loaded['group']}/{gene}")

    figure = plt.figure(figsize=(22.4, 14.4))
    grid = figure.add_gridspec(4, 7)
    if loaded["benchmark"] == "hest":
        # HEST sections are markedly taller than they are wide. Let the H&E
        # overview occupy two rows so it has comparable visual area to the
        # near-square cSCC H&E panel, without stretching or cropping tissue.
        image_axis = figure.add_subplot(grid[0:2, 0])
        panel_axes = [
            figure.add_subplot(grid[row, column])
            for row in range(4)
            for column in range(7)
            if not (row == 1 and column == 0) and not (row == 0 and column == 0)
        ]
    else:
        all_axes = [
            figure.add_subplot(grid[row, column])
            for row in range(4)
            for column in range(7)
        ]
        image_axis, panel_axes = all_axes[0], all_axes[1:]
    image_axis.imshow(he_background)
    image_axis.set_title(image_title, fontsize=17, pad=8)
    image_axis.axis("off")
    image_axis.set_aspect("equal")
    draw_expression(panel_axes[0], background, coordinates, truth, diameter, "Ground truth", None)
    for axis, method in zip(panel_axes[1:], methods):
        draw_expression(
            axis,
            background,
            coordinates,
            loaded["predictions"][method][:, gene_index],
            diameter,
            DISPLAY_NAMES[method],
            pccs[method],
        )
    for axis in panel_axes[1 + len(methods) :]:
        axis.axis("off")
    figure.subplots_adjust(
        left=0.016,
        right=0.925,
        top=0.855,
        bottom=0.030,
        wspace=0.10,
        hspace=0.42,
    )
    color_axis = figure.add_axes([0.958, 0.19, 0.010, 0.59])
    colorbar = figure.colorbar(
        plt.cm.ScalarMappable(norm=Normalize(0.0, 1.0), cmap="viridis"),
        cax=color_axis,
    )
    colorbar.ax.set_title("Scaled\nexpression", fontsize=16, pad=10)
    colorbar.ax.tick_params(labelsize=15)
    figure.suptitle(
        f"{BENCHMARK_LABELS[loaded['benchmark']]} | {loaded['group']} | {gene} | {loaded['sample']}\n{row.selection_category}",
        fontsize=21,
        y=0.988,
    )
    gene_dir = output / OUTPUT_DIRS[loaded["benchmark"]] / safe_name(gene)
    gene_dir.mkdir(parents=True, exist_ok=True)
    filename = (
        f"all_{safe_name(str(row.selection_category))}_{safe_name(loaded['group'])}_"
        f"{safe_name(loaded['sample'])}.png"
    )
    figure_path = gene_dir / filename
    figure.savefig(figure_path, dpi=dpi, facecolor="white")
    plt.close(figure)

    panel_rows = []
    for method in methods:
        panel_rows.append(
            {
                "dataset": BENCHMARK_LABELS[loaded["benchmark"]],
                "group": loaded["group"],
                "gene": gene,
                "section": loaded["sample"],
                "fold_id": loaded["fold_id"],
                "selection_category": row.selection_category,
                "method": DISPLAY_NAMES[method],
                "pcc": pccs[method],
                "prediction_path": str(loaded["paths"][method].resolve()),
                "figure_path": str(figure_path.resolve()),
                "pcc_recomputed_from_plotted_arrays": True,
            }
        )
    spot_rows = []
    vectors = {"Ground truth": truth}
    vectors.update(
        {
            DISPLAY_NAMES[method]: loaded["predictions"][method][:, gene_index]
            for method in methods
        }
    )
    for method, values in vectors.items():
        scaled = minmax(values)
        pcc = None if method == "Ground truth" else pccs[
            next(key for key, label in DISPLAY_NAMES.items() if label == method)
        ]
        for barcode, coord, raw, display in zip(
            loaded["barcodes"], coordinates, values, scaled
        ):
            spot_rows.append(
                {
                    "dataset": BENCHMARK_LABELS[loaded["benchmark"]],
                    "group": loaded["group"],
                    "gene": gene,
                    "section": loaded["sample"],
                    "barcode": barcode,
                    "x": float(coord[0]),
                    "y": float(coord[1]),
                    "method": method,
                    "expression_value": float(raw),
                    "panel_minmax_value": float(display),
                    "pcc": pcc,
                }
            )
    inventory = {
        "dataset": BENCHMARK_LABELS[loaded["benchmark"]],
        "group": loaded["group"],
        "gene": gene,
        "section": loaded["sample"],
        "selection_category": row.selection_category,
        "lin_mnp_pcc": pccs["Lin-MNP"],
        "best_baseline_pcc": float(row.best_baseline_pcc),
        "pcc_margin": float(row.pcc_margin),
        "n_spots": len(truth),
        "n_methods": len(methods),
        "image_path": str(image_path.resolve()),
        "figure_path": str(figure_path.resolve()),
    }
    return inventory, panel_rows, spot_rows


def write_gzip_csv(rows: Iterable[dict[str, Any]], path: Path) -> None:
    rows = iter(rows)
    first = next(rows, None)
    if first is None:
        raise ValueError(f"no rows for {path}")
    columns = list(first)
    with gzip.open(path, "wt", encoding="utf-8", newline="") as handle:
        pd.DataFrame([first], columns=columns).to_csv(handle, index=False)
        batch = []
        for row in rows:
            batch.append(row)
            if len(batch) >= 100_000:
                pd.DataFrame(batch, columns=columns).to_csv(handle, index=False, header=False)
                batch.clear()
        if batch:
            pd.DataFrame(batch, columns=columns).to_csv(handle, index=False, header=False)


def update_readmes(output: Path, selected: pd.DataFrame) -> None:
    root_path = output / "README.md"
    root = root_path.read_text(encoding="utf-8") if root_path.is_file() else "# Lin-MNP spatial-expression overviews\n"
    marker = "\n## HEST and STImage external selected-gene overviews\n"
    if marker in root:
        root = root.split(marker)[0].rstrip() + "\n"
    root += f"""
{marker}
The package now covers all four result families: cSCC, HER2+, HEST and
STImage. HEST and STImage figures are stored under `hest_external_50gene/`
and `stimage_external_50gene/`. They use the latest Lin-MNP prediction and all
23 available comparison methods in the established spatial-map visual
language. Because this is H&E + ground truth + 24 methods, the external panel
is expanded from 3 × 6 to 4 × 7 without dropping methods.

One deterministic seed-1/fold-0 held-out section is used per HEST cancer group
and STImage organ. Selection is descriptive: benchmark-wide top-10 advantage,
large-margin genes, and one best-advantage gene per group. If a group has no
Lin-MNP win among its 50 genes, its highest-PCC gene is retained and explicitly
marked as a no-win control. These test-selected maps do not define aggregate
performance claims.

HEST uses the clean publisher-deposited `downscaled_fullres` H&E image embedded
in each h5ad file, aligned through `obsm['spatial']`; STImage uses the archived
raw H&E image.

External output tables:

- `external_selected_genes.csv`
- `external_selection_candidates_all.csv.gz`
- `external_figure_inventory.csv`
- `external_panel_pcc.csv`
- `external_spot_expression_detail.csv.gz`
- `external_provenance.json`

Rebuild the external figures with:

```bash
python -B \\
  {Path(__file__).resolve()} --output {output.resolve()} --dpi 220
```
"""
    root_path.write_text(root, encoding="utf-8")

    for benchmark in ("hest", "stimage"):
        subset = selected.loc[selected["benchmark"].eq(benchmark)]
        readme = f"""# {BENCHMARK_LABELS[benchmark]} external spatial-expression overviews

This directory contains {len(subset)} selected gene/sample figures from
{subset['group'].nunique()} cancer groups or organs. Each figure shows H&E,
ground truth, Lin-MNP and all 23 available comparison methods.
Every map is independently min-max scaled for display only; every PCC is
recomputed from the unscaled arrays shown. The displayed sample is the first
held-out sample in the archived seed-1/fold-0 contract for its group.

Gene choice is test-selected and intended only for qualitative visualization.
See `../external_selected_genes.csv` for the exact rule, margin and explicit
no-win controls.
"""
        path = output / OUTPUT_DIRS[benchmark] / "README.md"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(readme, encoding="utf-8")


def update_manifest(output: Path) -> None:
    manifest_path = output / "result_manifest.json"
    files = []
    for path in sorted(output.rglob("*")):
        if (
            path.is_file()
            and path != manifest_path
            and "__pycache__" not in path.parts
            and path.suffix != ".pyc"
        ):
            files.append(
                {
                    "path": str(path.relative_to(output)),
                    "bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
            )
    payload = {
        "schema_version": 2,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "model": "Lin-MNP",
        "datasets": ["cSCC", "HER2+", "HEST", "STImage"],
        "files": files,
    }
    manifest_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--dpi", type=int, default=220)
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)

    records = representative_records("hest") + representative_records("stimage")
    candidates = candidate_table(records)
    selected = select_genes(candidates)
    candidates.to_csv(
        output / "external_selection_candidates_all.csv.gz",
        index=False,
        compression="gzip",
    )
    selected.to_csv(output / "external_selected_genes.csv", index=False)
    if args.prepare_only:
        print(
            json.dumps(
                {
                    "status": "prepared",
                    "records": len(records),
                    "candidates": len(candidates),
                    "selected": len(selected),
                },
                indent=2,
            )
        )
        return

    record_index = {
        (record["benchmark"], record["group"], record["sample"]): record
        for record in records
    }
    loaded_cache: dict[tuple[str, str, str], dict[str, Any]] = {}
    inventories: list[dict[str, Any]] = []
    panel_rows: list[dict[str, Any]] = []
    hest_cache = output / "source_data/hest_mosaic_backgrounds"
    spot_path = output / "external_spot_expression_detail.csv.gz"
    spot_count = 0
    wrote_header = False
    with gzip.open(spot_path, "wt", encoding="utf-8", newline="") as spot_handle:
        for index, row in selected.iterrows():
            key = (
                str(row["benchmark"]),
                str(row["group"]),
                str(row["sample"]),
            )
            if key not in loaded_cache:
                loaded_cache[key] = load_record(record_index[key])
            inventory, panels, spots = plot_one(
                row, loaded_cache[key], output, hest_cache, args.dpi
            )
            inventories.append(inventory)
            panel_rows.extend(panels)
            pd.DataFrame(spots).to_csv(
                spot_handle, index=False, header=not wrote_header
            )
            wrote_header = True
            spot_count += len(spots)
            del spots
            print(
                f"[{index + 1:02d}/{len(selected):02d}] "
                f"{row.benchmark}/{row.group}/{row.gene}"
            )

    inventory = pd.DataFrame(inventories)
    panels = pd.DataFrame(panel_rows)
    inventory.to_csv(output / "external_figure_inventory.csv", index=False)
    panels.to_csv(output / "external_panel_pcc.csv", index=False)

    # Exact numerical audit: all stored Lin-MNP and strongest-baseline values
    # must equal the recomputed values used by the renderer.
    audit = selected.merge(
        panels.loc[:, ["dataset", "group", "gene", "section", "method", "pcc"]],
        left_on=["group", "gene", "sample"],
        right_on=["group", "gene", "section"],
        how="left",
    )
    current_model = audit.loc[audit["method"].eq("Lin-MNP")]
    max_delta = float(
        np.nanmax(
            np.abs(current_model["lin_mnp_pcc"] - current_model["pcc"])
        )
    )
    if max_delta > 2e-6:
        raise ValueError(f"render audit failed: max Lin-MNP delta={max_delta}")

    provenance = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "created_by": str(Path(__file__).resolve()),
        "created_by_sha256": sha256_file(Path(__file__).resolve()),
        "latest_model": "Lin-MNP",
        "benchmarks": ["HEST", "STImage"],
        "comparison_campaign": str(CAMPAIGN_ROOT.resolve()),
        "latest_prediction_roots": {
            key: str(value.resolve()) for key, value in LATEST_ROOTS.items()
        },
        "selection_is_test_descriptive": True,
        "selection_scope": "one deterministic seed-1/fold-0 held-out sample per group",
        "n_groups": int(selected[["benchmark", "group"]].drop_duplicates().shape[0]),
        "n_selected_figures": int(len(selected)),
        "n_comparison_methods": len(METHOD_IDS),
        "render_grid": "4x7",
        "panel_scaling": "independent min-max for display only",
        "pcc_space": "unscaled archived 50-gene target/prediction arrays",
        "lin_mnp_render_audit_max_abs_delta": max_delta,
        "hest_background": "publisher-deposited downscaled_fullres H&E embedded in h5ad",
        "stimage_background": "archived raw H&E image",
    }
    (output / "external_provenance.json").write_text(
        json.dumps(provenance, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    write_readme = update_readmes
    write_readme(output, selected)
    update_manifest(output)
    print(
        json.dumps(
            {
                "status": "complete",
                "figures": len(inventory),
                "panel_rows": len(panels),
                "spot_rows": spot_count,
                "output": str(output),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
