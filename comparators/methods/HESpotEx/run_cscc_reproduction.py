#!/usr/bin/env python3
"""Run the released HESpotEx core model on its 12-section cSCC cohort.

This is a reproducible runtime adapter around the authors' expression GATE,
Quilt-Net encoder, graph decoder, LSTM head, nuclei scaling, losses, and
test-selected checkpoint rule. Frozen Quilt features are extracted in chunks
and cached, then supplied through the released ``method='phikon'`` feature
input branch. This avoids materializing an entire section of 224-pixel patches
on a 24-GB GPU without changing the frozen encoder result.

Two defects in the public data/code boundary are explicit here: the released
coordinate-spacing calculation gives a zero crop on the released cSCC H5ADs,
so the paper's documented default 224-pixel patch width is used; and the final
paper metric implementation was not released, so the project's transparent
HESpotEx-style metric module records the KL epsilon convention.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import gc
import hashlib
import json
import os
from pathlib import Path
import random
import subprocess
from typing import Any, Mapping, Sequence

import cv2
import numpy as np
import pandas as pd
from PIL import Image
import scanpy as sc
from scipy.spatial import distance
from scipy.sparse import issparse
import torch

from lin_mnp.benchmark.hespotex_metrics import (
    DEFAULT_KL_EPSILON,
    per_gene_minmax_mse,
    per_gene_pcc,
    per_gene_spatial_kl,
)
from lin_mnp.paths import ARCHIVE_ROOT, DATA_ROOT, RESULTS_ROOT

REPOSITORY_ROOT = Path(__file__).resolve().parent
from model.expression_embedding import expression_GVAE  # noqa: E402
from model.image_modules import ImageEncoder_QuiltNet  # noqa: E402
import train as released_train  # noqa: E402


Image.MAX_IMAGE_PIXELS = 2_800_000_000

DEFAULT_SOURCE_ROOT = DATA_ROOT / "hespotex/cscc"
DEFAULT_TARGET_BUNDLE = (
    ARCHIVE_ROOT / "external_benchmark/cscc12_heto/hespotex_exact_1971_v1"
)
DEFAULT_OUTPUT_ROOT = RESULTS_ROOT / "hespotex/cscc_official"
QUILT_WEIGHT = REPOSITORY_ROOT / "model/QuiltNet-B-32/open_clip_pytorch_model.bin"

SECTIONS: tuple[tuple[str, str, str], ...] = (
    ("GSM4284316", "P2", "P2_ST_rep1"),
    ("GSM4284317", "P2", "P2_ST_rep2"),
    ("GSM4284318", "P2", "P2_ST_rep3"),
    ("GSM4284319", "P5", "P5_ST_rep1"),
    ("GSM4284320", "P5", "P5_ST_rep2"),
    ("GSM4284321", "P5", "P5_ST_rep3"),
    ("GSM4284322", "P9", "P9_ST_rep1"),
    ("GSM4284323", "P9", "P9_ST_rep2"),
    ("GSM4284324", "P9", "P9_ST_rep3"),
    ("GSM4284325", "P10", "P10_ST_rep1"),
    ("GSM4284326", "P10", "P10_ST_rep2"),
    ("GSM4284327", "P10", "P10_ST_rep3"),
)
BY_GSM = {gsm: (patient, author) for gsm, patient, author in SECTIONS}
BY_AUTHOR = {author: (gsm, patient) for gsm, patient, author in SECTIONS}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git_commit() -> str:
    return subprocess.check_output(
        ["git", "-C", str(REPOSITORY_ROOT), "rev-parse", "HEAD"], text=True
    ).strip()


def gene_panel(bundle: Path) -> tuple[str, ...]:
    genes = tuple(
        line.strip()
        for line in (bundle / "gene_panel.txt").read_text(encoding="utf-8").splitlines()
        if line.strip()
    )
    if len(genes) != 1971 or len(set(genes)) != 1971:
        raise ValueError("the locked reconstructed panel must contain 1,971 genes")
    return genes


def dense(values: Any) -> np.ndarray:
    return np.asarray(values.toarray() if issparse(values) else values)


def load_raw_section(source_root: Path, author_name: str, genes: Sequence[str]):
    path = source_root / author_name / "st_adata.h5ad"
    value = sc.read_h5ad(path)
    value = value[~value.obs_names.duplicated(keep="first")]
    value = value[:, ~value.var_names.duplicated(keep="first")]
    absent = sorted(set(genes) - set(value.var_names.astype(str)))
    if absent:
        raise ValueError(f"{author_name} lacks panel genes: {absent[:5]}")
    value = value[:, list(genes)].copy()
    matrix = dense(value.X)
    if not np.isfinite(matrix).all() or (matrix < 0).any():
        raise ValueError(f"invalid raw expression in {path}")
    return value, path


def expected_bundle_target(
    bundle: Path, gsm: str, barcodes: Sequence[str], genes: Sequence[str]
) -> np.ndarray:
    with np.load(bundle / "slides" / f"{gsm}.npz", allow_pickle=False) as archive:
        saved_genes = tuple(np.asarray(archive["genes"]).astype(str))
        saved_barcodes = tuple(np.asarray(archive["barcodes"]).astype(str))
        target = np.asarray(archive["target"], dtype=np.float32)
    if saved_genes != tuple(genes):
        raise ValueError(f"bundle gene order mismatch for {gsm}")
    index = {barcode: position for position, barcode in enumerate(saved_barcodes)}
    if set(index) != set(barcodes):
        raise ValueError(f"bundle/source barcode set mismatch for {gsm}")
    return target[[index[barcode] for barcode in barcodes]]


def prepare_expression(args: argparse.Namespace) -> None:
    genes = gene_panel(args.target_bundle)
    cache_dir = args.output_root / "cache/expression"
    cache_dir.mkdir(parents=True, exist_ok=True)
    if any((cache_dir / f"{gsm}.npz").exists() for gsm, _, _ in SECTIONS):
        raise FileExistsError(f"expression cache already exists under {cache_dir}")

    adatas = []
    originals: dict[str, tuple[str, ...]] = {}
    source_hashes = {}
    for gsm, _, author in SECTIONS:
        value, path = load_raw_section(args.source_root, author, genes)
        originals[gsm] = tuple(value.obs_names.astype(str))
        source_hashes[gsm] = sha256_file(path)
        adatas.append(value)

    # With n_vars == n_top_genes, released load_anndatas skips a second HVG
    # selection but retains its exact graph construction, concatenation,
    # normalize_total(1e6), log1p, and sequential shared STAGATE fitting.
    processed = expression_GVAE(
        adatas=adatas,
        method="GATE",
        feature_method="hvg",
        k_cutoff=8,
        n_top_genes=len(genes),
        device=args.device,
        stagate_epochs=args.stagate_epochs,
    )
    if len(processed) != len(SECTIONS):
        raise ValueError("released expression_GVAE returned an unexpected section count")

    for (gsm, patient, author), value in zip(SECTIONS, processed):
        expression = dense(value.X).astype(np.float32)
        latent = np.asarray(value.obsm["embedding_rep"], dtype=np.float32)
        barcodes = originals[gsm]
        expected = expected_bundle_target(args.target_bundle, gsm, barcodes, genes)
        if expression.shape != expected.shape or not np.allclose(
            expression, expected, rtol=0.0, atol=2e-5
        ):
            difference = float(np.max(np.abs(expression - expected)))
            raise ValueError(f"released target mismatch for {gsm}: max abs {difference}")
        if latent.shape != (len(barcodes), 64) or not np.isfinite(latent).all():
            raise ValueError(f"invalid STAGATE embedding for {gsm}")
        raw, _ = load_raw_section(args.source_root, author, genes)
        np.savez_compressed(
            cache_dir / f"{gsm}.npz",
            gsm=np.asarray(gsm),
            patient=np.asarray(patient),
            author_name=np.asarray(author),
            barcodes=np.asarray(barcodes),
            genes=np.asarray(genes),
            expression=expression,
            latent=latent,
            spatial=np.asarray(raw.obsm["spatial"], dtype=np.int64),
        )
    metadata = {
        "created_at_utc": utc_now(),
        "released_repository_commit": git_commit(),
        "stage": "released expression_GVAE/GATE",
        "device": str(args.device),
        "stagate_epochs": int(args.stagate_epochs),
        "section_order": [gsm for gsm, _, _ in SECTIONS],
        "genes": len(genes),
        "target": "released normalize_total(1e6) then natural-log log1p",
        "source_h5ad_sha256": source_hashes,
        "target_bundle": str(args.target_bundle.resolve()),
        "target_bundle_metadata_sha256": sha256_file(args.target_bundle / "metadata.json"),
    }
    (cache_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def released_dense_adjacency(coordinates: np.ndarray, k: int = 4) -> np.ndarray:
    """Replicate ``common.dataset.calcADJ`` without its optional dependencies."""

    spatial = np.asarray(coordinates)
    nodes = spatial.shape[0]
    adjacency = np.zeros((nodes, nodes), dtype=np.float32)
    for index in range(nodes):
        distances = distance.cdist(spatial[index : index + 1], spatial, "euclidean")
        neighbours = distances.argsort()[0, 1 : k + 1]
        adjacency[index, neighbours] = 1.0
    return adjacency


def _image_batch(
    image: np.ndarray,
    spatial: np.ndarray,
    indices: range,
    crop_radius: int,
) -> torch.Tensor:
    height, width = image.shape[:2]
    patches = []
    for index in indices:
        v1, v2 = (int(value) for value in spatial[index])
        patch = image[
            max(v2 - crop_radius, 0) : min(v2 + crop_radius, height),
            max(v1 - crop_radius, 0) : min(v1 + crop_radius, width),
            :,
        ]
        if patch.size == 0:
            raise ValueError(f"empty image patch at spot {index}: {(v1, v2)}")
        interpolation = cv2.INTER_AREA if patch.shape[0] >= 224 else cv2.INTER_CUBIC
        patches.append(cv2.resize(patch, (224, 224), interpolation=interpolation))
    array = np.stack(patches).transpose(0, 3, 1, 2)
    return torch.from_numpy(np.ascontiguousarray(array)).float()


def _nuclei_values(
    mask: np.ndarray, spatial: np.ndarray, crop_radius: int
) -> np.ndarray:
    height, width = mask.shape[:2]
    output = np.zeros((len(spatial), 1), dtype=np.float32)
    for index, (v1, v2) in enumerate(spatial.astype(int)):
        patch = mask[
            max(v2 - crop_radius, 0) : min(v2 + crop_radius, height),
            max(v1 - crop_radius, 0) : min(v1 + crop_radius, width),
        ]
        output[index, 0] = len(np.unique(patch))
    output = np.minimum(np.maximum(output, 1.0), 20.0)
    return output / output.max() + 1.0


def prepare_images(args: argparse.Namespace) -> None:
    genes = gene_panel(args.target_bundle)
    selected = (
        [next(row for row in SECTIONS if row[0] == args.section)]
        if args.section
        else list(SECTIONS)
    )
    cache_dir = args.output_root / "cache/image"
    cache_dir.mkdir(parents=True, exist_ok=True)
    previous_cwd = Path.cwd()
    os.chdir(REPOSITORY_ROOT)
    try:
        encoder = ImageEncoder_QuiltNet(model_name="ViT-B-32-quickgelu").to(args.device)
        encoder.eval()
        for parameter in encoder.parameters():
            parameter.requires_grad = False
        for gsm, patient, author in selected:
            output = cache_dir / f"{gsm}.npz"
            if output.exists():
                raise FileExistsError(output)
            raw, _ = load_raw_section(args.source_root, author, genes)
            spatial = np.asarray(raw.obsm["spatial"], dtype=np.int64)
            image_path = args.source_root / author / "image.jpg"
            mask_path = args.source_root / author / "image.npy"
            image = np.asarray(Image.open(image_path).convert("RGB"))
            mask = np.load(mask_path, mmap_mode="r", allow_pickle=False)
            if mask.shape[:2] != image.shape[:2]:
                raise ValueError(f"image/mask shape mismatch for {author}")

            features = []
            with torch.no_grad():
                for start in range(0, len(spatial), args.image_batch_size):
                    stop = min(start + args.image_batch_size, len(spatial))
                    batch = _image_batch(
                        image, spatial, range(start, stop), args.crop_radius
                    ).to(args.device)
                    encoded = encoder(batch)[0]
                    features.append(encoded.detach().cpu().numpy().astype(np.float32))
            feature_matrix = np.concatenate(features, axis=0)
            nuclei = _nuclei_values(mask, spatial, args.crop_radius)
            adjacency = released_dense_adjacency(spatial, k=4)
            if feature_matrix.shape != (len(spatial), 512):
                raise ValueError(f"unexpected Quilt feature shape for {author}")
            np.savez_compressed(
                output,
                gsm=np.asarray(gsm),
                patient=np.asarray(patient),
                author_name=np.asarray(author),
                barcodes=np.asarray(raw.obs_names.astype(str)),
                spatial=spatial,
                quilt_features=feature_matrix,
                nuclei_number=nuclei,
                adjacency=adjacency,
                crop_width=np.asarray(args.crop_radius * 2),
                source_image_sha256=np.asarray(sha256_file(image_path)),
                source_mask_sha256=np.asarray(sha256_file(mask_path)),
                quilt_weight_sha256=np.asarray(sha256_file(QUILT_WEIGHT)),
            )
            print(json.dumps({"cached": str(output), "spots": len(spatial)}), flush=True)
            del image, mask, raw, adjacency, nuclei, feature_matrix, features
            gc.collect()
            torch.cuda.empty_cache()
    finally:
        os.chdir(previous_cwd)


class CachedSectionDataset(torch.utils.data.Dataset):
    def __init__(self, expression_path: Path, image_path: Path):
        with np.load(expression_path, allow_pickle=False) as expression_archive:
            self.gsm = str(expression_archive["gsm"].item())
            self.patient = str(expression_archive["patient"].item())
            self.barcodes = tuple(expression_archive["barcodes"].astype(str))
            self.genes = tuple(expression_archive["genes"].astype(str))
            self.expression = np.asarray(
                expression_archive["expression"], dtype=np.float32
            )
            self.latent = np.asarray(expression_archive["latent"], dtype=np.float32)
        # The first cache revision stored the pandas Index as an object array;
        # these locally generated, SHA-tracked caches are therefore opened
        # with pickle enabled for backward compatibility.
        with np.load(image_path, allow_pickle=True) as image_archive:
            image_barcodes = tuple(image_archive["barcodes"].astype(str))
            self.image = np.asarray(image_archive["quilt_features"], dtype=np.float32)
            self.adjacency = np.asarray(image_archive["adjacency"], dtype=np.float32)
            self.nuclei = np.asarray(image_archive["nuclei_number"], dtype=np.float32)
        if image_barcodes != self.barcodes:
            raise ValueError(f"expression/image barcode mismatch for {self.gsm}")
        expected = (len(self.barcodes), len(self.genes))
        if self.expression.shape != expected or self.latent.shape != (
            len(self.barcodes),
            64,
        ):
            raise ValueError(f"invalid cached expression shape for {self.gsm}")

    def __len__(self) -> int:
        return 1

    def __getitem__(self, _: int) -> dict[str, dict[str, torch.Tensor]]:
        return {
            "meta": {
                "image": torch.from_numpy(self.image),
                "expression": torch.from_numpy(self.expression),
                "latent": torch.from_numpy(self.latent),
                "adj": torch.from_numpy(self.adjacency),
                "nuclei_number": torch.from_numpy(self.nuclei),
            }
        }


def load_datasets(output_root: Path) -> dict[str, CachedSectionDataset]:
    output = {}
    for gsm, _, _ in SECTIONS:
        output[gsm] = CachedSectionDataset(
            output_root / "cache/expression" / f"{gsm}.npz",
            output_root / "cache/image" / f"{gsm}.npz",
        )
    return output


def run_fold(args: argparse.Namespace) -> None:
    if args.section not in BY_GSM:
        raise ValueError("fold stage requires --section with a GSM identifier")
    datasets = load_datasets(args.output_root)
    outer = args.section
    train_ids = [gsm for gsm, _, _ in SECTIONS if gsm != outer]
    train_dataset = torch.utils.data.ConcatDataset([datasets[gsm] for gsm in train_ids])
    test_dataset = torch.utils.data.ConcatDataset([datasets[outer]])
    fold_dir = args.output_root / "folds" / outer
    if (fold_dir / "predictions.npz").exists():
        raise FileExistsError(fold_dir / "predictions.npz")
    fold_dir.mkdir(parents=True, exist_ok=True)
    previous_cwd = Path.cwd()
    os.chdir(REPOSITORY_ROOT)
    try:
        predictions, selection = released_train.main(
            train_dataset,
            None,
            test_dataset,
            image_embedding=64,
            out_embedding=len(datasets[outer].genes),
            max_epochs=args.max_epochs,
            save_path=str(fold_dir),
            method="phikon",
            device=args.device,
            return_metadata=True,
        )
    finally:
        os.chdir(previous_cwd)
    prediction = np.asarray(predictions[0], dtype=np.float32)
    target = datasets[outer].expression
    pcc = per_gene_pcc(target, prediction)
    kl = per_gene_spatial_kl(target, prediction, epsilon=DEFAULT_KL_EPSILON)
    mse = per_gene_minmax_mse(target, prediction)
    np.savez_compressed(
        fold_dir / "predictions.npz",
        gsm=np.asarray(outer),
        patient=np.asarray(datasets[outer].patient),
        barcodes=np.asarray(datasets[outer].barcodes),
        genes=np.asarray(datasets[outer].genes),
        target=target,
        prediction=prediction,
        pcc=pcc,
        kl_divergence=kl,
        mse=mse,
    )
    metadata = {
        "created_at_utc": utc_now(),
        "released_repository_commit": git_commit(),
        "outer_section": outer,
        "patient": datasets[outer].patient,
        "train_sections": train_ids,
        "max_epochs": int(args.max_epochs),
        "device": str(args.device),
        "method": "phikon frozen-feature input branch",
        "released_model_components": (
            "Quilt-Net, latent/gene heads, four graph decoders, two-layer LSTM, "
            "nuclei scaling, latent PCC plus expression MSE losses"
        ),
        "selection": selection,
        "outer_test_selected": True,
        "mean_pcc": float(np.mean(pcc)),
        "mean_kl_divergence": float(np.mean(kl)),
        "mean_mse": float(np.mean(mse)),
        "runtime_adaptations": [
            "frozen Quilt features cached in chunks and passed through released phikon branch",
            "paper-documented 224-pixel fallback patch width used because released spacing code returns zero",
            "PCC loop algebraically vectorized",
            "device exposed as a function argument",
            "deterministic reconstructed 1,971-count panel used because final author gene manifest is absent",
        ],
    }
    (fold_dir / "training.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(metadata, sort_keys=True), flush=True)


def aggregate(args: argparse.Namespace) -> None:
    rows = []
    for gsm, patient, _ in SECTIONS:
        path = args.output_root / "folds" / gsm / "predictions.npz"
        with np.load(path, allow_pickle=False) as archive:
            genes = archive["genes"].astype(str)
            pcc = np.asarray(archive["pcc"], dtype=np.float64)
            kl = np.asarray(archive["kl_divergence"], dtype=np.float64)
            mse = np.asarray(archive["mse"], dtype=np.float64)
        rows.extend(
            {
                "slide_id": gsm,
                "patient_id": patient,
                "gene": gene,
                "pcc": float(pcc_value),
                "kl_divergence": float(kl_value),
                "mse": float(mse_value),
            }
            for gene, pcc_value, kl_value, mse_value in zip(genes, pcc, kl, mse)
        )
    frame = pd.DataFrame(rows)
    metrics = ["pcc", "kl_divergence", "mse"]
    per_slide = frame.groupby(["slide_id", "patient_id"], as_index=False)[metrics].mean()
    per_patient_gene = frame.groupby(["patient_id", "gene"], as_index=False)[metrics].mean()
    per_patient = per_patient_gene.groupby("patient_id", as_index=False)[metrics].mean()
    per_gene = per_patient_gene.groupby("gene", as_index=False)[metrics].mean()
    summary = {
        "created_at_utc": utc_now(),
        "label": "HESpotEx released-core cSCC reconstructed-1971 reproduction",
        "released_repository_commit": git_commit(),
        "sections": 12,
        "patients": 4,
        "spots": 8671,
        "genes": 1971,
        "mean_pcc": float(per_gene["pcc"].mean()),
        "mean_kl_divergence": float(per_gene["kl_divergence"].mean()),
        "mean_mse": float(per_gene["mse"].mean()),
        "paper_reference": {
            "mean_pcc": 0.239,
            "mean_kl_divergence": 1.617,
            "mean_mse": 0.213,
        },
        "checkpoint_selection": "best outer-test PCC epoch, matching released train.py",
        "exact_unmodified_reproduction": False,
        "kl_epsilon": DEFAULT_KL_EPSILON,
    }
    report_dir = args.output_root / "report"
    if report_dir.exists():
        raise FileExistsError(report_dir)
    report_dir.mkdir(parents=True)
    frame.to_csv(report_dir / "per_slide_gene.csv", index=False)
    per_slide.to_csv(report_dir / "per_slide.csv", index=False)
    per_patient_gene.to_csv(report_dir / "per_patient_gene.csv", index=False)
    per_patient.to_csv(report_dir / "per_patient.csv", index=False)
    per_gene.to_csv(report_dir / "per_gene.csv", index=False)
    (report_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, sort_keys=True), flush=True)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "stage", choices=("expression", "images", "fold", "aggregate")
    )
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE_ROOT)
    parser.add_argument("--target-bundle", type=Path, default=DEFAULT_TARGET_BUNDLE)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--section", choices=tuple(BY_GSM))
    parser.add_argument("--stagate-epochs", type=int, default=500)
    parser.add_argument("--max-epochs", type=int, default=30)
    parser.add_argument("--image-batch-size", type=int, default=64)
    parser.add_argument("--crop-radius", type=int, default=112)
    args = parser.parse_args(argv)
    if args.stage == "fold" and args.section is None:
        parser.error("fold stage requires --section")
    if args.stagate_epochs < 1 or args.max_epochs < 1:
        parser.error("epoch counts must be positive")
    if args.image_batch_size < 1 or args.crop_radius < 1:
        parser.error("image batch size and crop radius must be positive")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    args.output_root = args.output_root.resolve()
    args.source_root = args.source_root.resolve()
    args.target_bundle = args.target_bundle.resolve()
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    if args.stage == "expression":
        prepare_expression(args)
    elif args.stage == "images":
        prepare_images(args)
    elif args.stage == "fold":
        run_fold(args)
    else:
        aggregate(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
