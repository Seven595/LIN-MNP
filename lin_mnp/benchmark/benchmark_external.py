"""Shared data contract for local HEtoSGEBench comparator reruns.

This module deliberately owns no model code.  External networks consume the
same image cache, patient-level folds, ordered 997-gene panel and barcode
identities through this interface, while retaining their method-specific
target transformation and architecture.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import anndata as ad
import numpy as np
import torch
from torch.utils.data import Dataset

from lin_mnp.paths import CACHE_ROOT, DATA_ROOT

COHORT_ROOT = DATA_ROOT / "cscc/cohort"
DERIVED_ROOT = DATA_ROOT / "cscc/derived"
FOLDS_PATH = COHORT_ROOT / "splits/heto_official_cv4/folds.json"
PREPARED_ROOT = DERIVED_ROOT / "heto_official_cv4/fold0/locked997/prepared/slides"
LABEL_ROOT = DATA_ROOT / "cscc/labels"
EXTERNAL_CONTRACT_ENV = "LIN_MNP_EXTERNAL_CONTRACT"


def _external_contract() -> dict[str, object] | None:
    value = os.environ.get(EXTERNAL_CONTRACT_ENV)
    if not value:
        return None
    path = Path(value).resolve()
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["_contract_path"] = str(path)
    return payload


_ACTIVE_CONTRACT = _external_contract()
PATCH_ROOT = (
    Path(str(_ACTIVE_CONTRACT["patch_root"])).resolve()
    if _ACTIVE_CONTRACT is not None
    else CACHE_ROOT / "cscc/benchmark_patches"
)
H5AD_ROOT = COHORT_ROOT / "standardized_h5ad"


@dataclass(frozen=True)
class FoldSpec:
    fold_id: int
    train_ids: tuple[str, ...]
    val_ids: tuple[str, ...]
    test_ids: tuple[str, ...]
    train_patients: tuple[str, ...]
    val_patient: str
    test_patient: str


@dataclass(frozen=True)
class SlideBundle:
    sample_id: str
    patient_id: str
    barcodes: tuple[str, ...]
    genes: tuple[str, ...]
    coordinates: np.ndarray
    array_coordinates: np.ndarray
    raw_counts: np.ndarray
    log1p_raw: np.ndarray
    log1p_cp10k: np.ndarray
    full_library_size: np.ndarray
    patch_start: int
    patch_stop: int
    full_gene_count: int | None = None
    contract_target: np.ndarray | None = None

    def target(self, transform: str) -> np.ndarray:
        key = transform.casefold()
        if key == "hespotex_target":
            if self.contract_target is None:
                raise ValueError("hespotex_target requested outside an external contract")
            return self.contract_target
        if key == "raw_counts":
            return self.raw_counts
        if key == "log1p_raw":
            return self.log1p_raw
        if key == "log1p_cp10k":
            return self.log1p_cp10k
        if key == "log1p_cpm":
            denominator = np.maximum(self.full_library_size[:, None], 1.0)
            return np.log1p(self.raw_counts / denominator * 1_000_000.0).astype(
                np.float32
            )
        if key == "igidl_log_pseudocount_cpm":
            if self.full_gene_count is None:
                raise ValueError("IGI-DL target requires the full matrix gene count")
            # Released IGI-DL preprocessing adds one pseudocount to every gene,
            # divides by the pseudocount-adjusted full-library total, scales to
            # one million and then takes the natural logarithm.
            denominator = self.full_library_size[:, None] + float(self.full_gene_count)
            return np.log((self.raw_counts + 1.0) / denominator * 1_000_000.0).astype(
                np.float32
            )
        if key == "log10_raw":
            return np.log10(1.0 + self.raw_counts).astype(np.float32)
        raise ValueError(f"unknown target transform: {transform}")


def load_folds(path: Path | None = None) -> dict[int, FoldSpec]:
    contract = _ACTIVE_CONTRACT
    if path is None and contract is not None:
        payload = contract
    else:
        path = FOLDS_PATH if path is None else path
        payload = json.loads(path.read_text(encoding="utf-8"))
    result: dict[int, FoldSpec] = {}
    test_patients: set[str] = set()
    for row in payload["folds"]:
        spec = FoldSpec(
            fold_id=int(row["fold_id"]),
            train_ids=tuple(map(str, row["train_sample_ids"])),
            val_ids=tuple(map(str, row["val_sample_ids"])),
            test_ids=tuple(map(str, row["test_sample_ids"])),
            train_patients=tuple(map(str, row["train_patient_ids"])),
            val_patient=str(row["val_patient_id"]),
            test_patient=str(row["test_patient_id"]),
        )
        if spec.fold_id in result:
            raise ValueError("duplicate fold")
        if (contract is None or path is not None) and spec.test_patient in test_patients:
            raise ValueError("duplicate fold or outer-test patient")
        result[spec.fold_id] = spec
        test_patients.add(spec.test_patient)
    if contract is None or path is not None:
        if set(result) != {0, 1, 2, 3} or len(test_patients) != 4:
            raise ValueError("benchmark contract requires four unique outer-test patients")
    else:
        expected = int(contract["fold_count"])
        if set(result) != set(range(expected)):
            raise ValueError("external contract requires consecutive section folds")
        test_ids = [value for spec in result.values() for value in spec.test_ids]
        if len(test_ids) != expected or len(set(test_ids)) != expected:
            raise ValueError("external contract must test every section exactly once")
    return result


def _full_library_sizes(sample_id: str, barcodes: Sequence[str]) -> np.ndarray:
    paths = sorted(H5AD_ROOT.glob(f"{sample_id}_*.h5ad"))
    if len(paths) != 1:
        raise ValueError(f"expected one h5ad for {sample_id}, found {paths}")
    data = ad.read_h5ad(paths[0], backed="r")
    try:
        if "total_counts" not in data.obs:
            raise ValueError(f"total_counts missing in {paths[0]}")
        totals = data.obs["total_counts"].astype(float)
        missing = set(barcodes) - set(map(str, totals.index))
        if missing:
            raise ValueError(f"h5ad is missing {len(missing)} canonical barcodes")
        return totals.loc[list(barcodes)].to_numpy(dtype=np.float32)
    finally:
        data.file.close()


def _full_gene_count(sample_id: str) -> int:
    paths = sorted(H5AD_ROOT.glob(f"{sample_id}_*.h5ad"))
    if len(paths) != 1:
        raise ValueError(f"expected one h5ad for {sample_id}, found {paths}")
    data = ad.read_h5ad(paths[0], backed="r")
    try:
        return int(data.n_vars)
    finally:
        data.file.close()


def load_slides(
    patch_root: Path | None = None,
) -> tuple[dict[str, SlideBundle], tuple[str, ...]]:
    patch_root = PATCH_ROOT if patch_root is None else patch_root
    if _ACTIVE_CONTRACT is not None:
        return _load_external_contract_slides(_ACTIVE_CONTRACT, patch_root)
    manifest_path = patch_root / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"shared patch cache is missing; run prepare_benchmark_external_patch_cache.py: {patch_root}"
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    rows = {str(row["sample_id"]): row for row in manifest["slides"]}
    result: dict[str, SlideBundle] = {}
    gene_order: tuple[str, ...] | None = None
    for sample_id in sorted(rows):
        row = rows[sample_id]
        with np.load(PREPARED_ROOT / f"{sample_id}.npz", allow_pickle=False) as prepared:
            barcodes = tuple(prepared["barcodes"].astype(str))
            genes = tuple(prepared["genes"].astype(str))
            coordinates = prepared["coords"].astype(np.float32)
            array_coordinates = prepared["array_coords"].astype(np.int64)
            cp10k = prepared["expr"].astype(np.float32)
        with np.load(LABEL_ROOT / f"{sample_id}.npz", allow_pickle=False) as labels:
            if tuple(labels["barcodes"].astype(str)) != barcodes:
                raise ValueError(f"label barcode order differs for {sample_id}")
            label_genes = tuple(labels["genes"].astype(str))
            label_index = {gene.casefold(): index for index, gene in enumerate(label_genes)}
            if len(label_index) != len(label_genes) or {
                gene.casefold() for gene in genes
            } != set(label_index):
                raise ValueError(f"label gene identities differ for {sample_id}")
            order = np.asarray([label_index[gene.casefold()] for gene in genes])
            raw = labels["raw_counts"][:, order].astype(np.float32)
            log1p_raw = labels["log1p_raw_counts"][:, order].astype(np.float32)
            patient_id = str(labels["patient_id"].item())
        if tuple(map(str, row["barcodes"])) != barcodes:
            raise ValueError(f"patch cache barcode order differs for {sample_id}")
        if gene_order is None:
            gene_order = genes
        elif genes != gene_order:
            raise ValueError(f"gene order differs for {sample_id}")
        result[sample_id] = SlideBundle(
            sample_id=sample_id,
            patient_id=patient_id,
            barcodes=barcodes,
            genes=genes,
            coordinates=coordinates,
            array_coordinates=array_coordinates,
            raw_counts=raw,
            log1p_raw=log1p_raw,
            log1p_cp10k=cp10k,
            full_library_size=_full_library_sizes(sample_id, barcodes),
            patch_start=int(row["start"]),
            patch_stop=int(row["stop"]),
            full_gene_count=_full_gene_count(sample_id),
        )
    if gene_order is None or len(result) != 12 or len(gene_order) != 997:
        raise ValueError("incomplete benchmark slide contract")
    return result, gene_order


def _load_external_contract_slides(
    contract: Mapping[str, object], patch_root: Path
) -> tuple[dict[str, SlideBundle], tuple[str, ...]]:
    """Load a HESpotEx target/patch identity contract.

    Every native target-transform request is routed to the same released
    HESpotEx target matrix later in :func:`fit_target_transform`.  The duplicate
    SlideBundle fields exist only because the benchmark model adapters consume
    that historical interface.
    """

    manifest = json.loads((patch_root / "manifest.json").read_text(encoding="utf-8"))
    target_root = Path(str(contract["target_bundle"])).resolve()
    coordinate_root = Path(str(contract["coordinate_root"])).resolve()
    expected_sections = tuple(map(str, contract["section_ids"]))
    rows = {str(row["sample_id"]): row for row in manifest["slides"]}
    if set(rows) != set(expected_sections):
        raise ValueError("patch manifest and external section contract differ")
    genes = tuple(
        value.strip()
        for value in (target_root / "gene_panel.txt").read_text(encoding="utf-8").splitlines()
        if value.strip()
    )
    result: dict[str, SlideBundle] = {}
    for sample_id in expected_sections:
        row = rows[sample_id]
        with np.load(target_root / "slides" / f"{sample_id}.npz", allow_pickle=False) as archive:
            barcodes = tuple(archive["barcodes"].astype(str))
            current_genes = tuple(archive["genes"].astype(str))
            target = archive["target"].astype(np.float32)
            patient_id = (
                str(archive["patient_id"].item())
                if "patient_id" in archive.files
                else str(
                    row.get(
                        "patient_id",
                        dict(contract["patient_by_section"])[sample_id],
                    )
                )
            )
        if current_genes != genes or tuple(map(str, row["barcodes"])) != barcodes:
            raise ValueError(f"external target identity mismatch for {sample_id}")
        with np.load(coordinate_root / f"{sample_id}.npz", allow_pickle=False) as archive:
            coordinate_barcodes = tuple(archive["barcodes"].astype(str))
            if coordinate_barcodes != barcodes:
                raise ValueError(f"external coordinate identity mismatch for {sample_id}")
            coordinates = archive["coords"].astype(np.float32)
            array_coordinates = archive["array_coords"].astype(np.int64)
        result[sample_id] = SlideBundle(
            sample_id=sample_id,
            patient_id=patient_id,
            barcodes=barcodes,
            genes=genes,
            coordinates=coordinates,
            array_coordinates=array_coordinates,
            raw_counts=np.expm1(np.clip(target, 0, 20)).astype(np.float32),
            log1p_raw=target.copy(),
            log1p_cp10k=target.copy(),
            full_library_size=np.ones(len(barcodes), dtype=np.float32),
            patch_start=int(row["start"]),
            patch_stop=int(row["stop"]),
            contract_target=target,
        )
    if len(result) != len(expected_sections) or not genes:
        raise ValueError("incomplete external slide contract")
    return result, genes


def fit_target_transform(
    slides: Mapping[str, SlideBundle],
    train_ids: Sequence[str],
    transform: str,
) -> dict[str, np.ndarray | str]:
    """Fit fold-training-only target normalisation parameters."""

    if any(slide.contract_target is not None for slide in slides.values()):
        if not all(slide.contract_target is not None for slide in slides.values()):
            raise ValueError("mixed benchmark and HESpotEx slide contracts")
        return {"kind": "hespotex_target", "requested_native_transform": transform}
    key = transform.casefold()
    if key in {
        "raw_counts",
        "log1p_raw",
        "log1p_cp10k",
        "log1p_cpm",
        "igidl_log_pseudocount_cpm",
    }:
        return {"kind": key}
    if key == "egn_minmax_log10_raw":
        matrix = np.concatenate([slides[name].target("log10_raw") for name in train_ids])
        minimum = matrix.min(axis=0).astype(np.float32)
        maximum = matrix.max(axis=0).astype(np.float32)
        return {"kind": key, "minimum": minimum, "maximum": maximum}
    if key == "istar_train_minmax_raw_counts":
        matrix = np.concatenate([slides[name].target("raw_counts") for name in train_ids])
        minimum = matrix.min(axis=0).astype(np.float32)
        maximum = matrix.max(axis=0).astype(np.float32)
        return {"kind": key, "minimum": minimum, "maximum": maximum}
    if key == "train_zscore_log1p_cp10k":
        matrix = np.concatenate([slides[name].target("log1p_cp10k") for name in train_ids])
        mean = matrix.mean(axis=0).astype(np.float32)
        scale = matrix.std(axis=0, ddof=1).astype(np.float32)
        scale[~np.isfinite(scale) | (scale < 1e-6)] = 1.0
        return {"kind": key, "mean": mean, "scale": scale}
    raise ValueError(f"unknown fitted target transform: {transform}")


def apply_target_transform(
    slide: SlideBundle, parameters: Mapping[str, np.ndarray | str]
) -> np.ndarray:
    key = str(parameters["kind"])
    if key in {
        "raw_counts",
        "log1p_raw",
        "log1p_cp10k",
        "log1p_cpm",
        "igidl_log_pseudocount_cpm",
        "hespotex_target",
    }:
        return slide.target(key)
    if key == "egn_minmax_log10_raw":
        minimum = np.asarray(parameters["minimum"], dtype=np.float32)
        maximum = np.asarray(parameters["maximum"], dtype=np.float32)
        return ((slide.target("log10_raw") - minimum) / (maximum - minimum + 1e-8)).astype(
            np.float32
        )
    if key == "istar_train_minmax_raw_counts":
        minimum = np.asarray(parameters["minimum"], dtype=np.float32)
        maximum = np.asarray(parameters["maximum"], dtype=np.float32)
        return ((slide.target("raw_counts") - minimum) / (maximum - minimum + 1e-8)).astype(
            np.float32
        )
    if key == "train_zscore_log1p_cp10k":
        mean = np.asarray(parameters["mean"], dtype=np.float32)
        scale = np.asarray(parameters["scale"], dtype=np.float32)
        return ((slide.target("log1p_cp10k") - mean) / scale).astype(np.float32)
    raise ValueError(f"unknown fitted target transform: {key}")


class SpotPatchDataset(Dataset):
    """Spot-level view over the immutable shared patch memmap."""

    def __init__(
        self,
        slides: Mapping[str, SlideBundle],
        sample_ids: Sequence[str],
        target_parameters: Mapping[str, np.ndarray | str],
        patch_root: Path = PATCH_ROOT,
    ) -> None:
        self.slides = slides
        self.sample_ids = tuple(map(str, sample_ids))
        self.target_parameters = target_parameters
        self.patch_path = patch_root / "patches_224_uint8.npy"
        self._patches: np.ndarray | None = None
        targets = []
        global_indices = []
        slide_indices = []
        local_indices = []
        for slide_index, sample_id in enumerate(self.sample_ids):
            slide = slides[sample_id]
            targets.append(apply_target_transform(slide, target_parameters))
            global_indices.extend(range(slide.patch_start, slide.patch_stop))
            slide_indices.extend([slide_index] * len(slide.barcodes))
            local_indices.extend(range(len(slide.barcodes)))
        self.targets = np.concatenate(targets).astype(np.float32, copy=False)
        self.global_indices = np.asarray(global_indices, dtype=np.int64)
        self.slide_indices = np.asarray(slide_indices, dtype=np.int16)
        self.local_indices = np.asarray(local_indices, dtype=np.int32)
        if len(self.targets) != len(self.global_indices):
            raise ValueError("spot mapping and target dimensions differ")

    @property
    def patches(self) -> np.ndarray:
        if self._patches is None:
            self._patches = np.load(self.patch_path, mmap_mode="r")
        return self._patches

    def __len__(self) -> int:
        return len(self.global_indices)

    def __getitem__(self, index: int):
        patch = np.asarray(self.patches[self.global_indices[index]])
        return (
            torch.from_numpy(patch.copy()),
            torch.from_numpy(self.targets[index]),
            int(self.slide_indices[index]),
            int(self.local_indices[index]),
        )


def concatenate_targets(
    slides: Mapping[str, SlideBundle],
    sample_ids: Iterable[str],
    parameters: Mapping[str, np.ndarray | str],
) -> np.ndarray:
    return np.concatenate(
        [apply_target_transform(slides[name], parameters) for name in sample_ids]
    ).astype(np.float32, copy=False)


__all__ = [
    "COHORT_ROOT",
    "DERIVED_ROOT",
    "EXTERNAL_CONTRACT_ENV",
    "FOLDS_PATH",
    "PATCH_ROOT",
    "FoldSpec",
    "SlideBundle",
    "SpotPatchDataset",
    "apply_target_transform",
    "concatenate_targets",
    "fit_target_transform",
    "load_folds",
    "load_slides",
]
