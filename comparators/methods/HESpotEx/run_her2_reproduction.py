#!/usr/bin/env python3
"""Run the released HESpotEx core on the exact 24-section HER2+ benchmark.

This runtime adapter uses the authors' Zenodo H5AD, H&E and nuclei-mask files,
the exact ordered 2,759-row Fig. 2a panel, released expression GATE, Quilt-Net,
graph/LSTM predictor, two released losses, and released outer-test-selected
checkpoint rule.  Frozen Quilt features are cached in chunks and passed via
the released ``method='phikon'`` branch, as in the cSCC reproduction adapter.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
from pathlib import Path
import random
from typing import Sequence

import numpy as np
import pandas as pd
from PIL import Image
import scanpy as sc
import torch

import run_cscc_reproduction as base
from lin_mnp.paths import ARCHIVE_ROOT, DATA_ROOT, RESULTS_ROOT


REPOSITORY_ROOT = Path(__file__).resolve().parent
DEFAULT_H5AD_ROOT = DATA_ROOT / "hespotex/her2/h5ad"
DEFAULT_IMAGE_ROOT = DATA_ROOT / "hespotex/her2/images"
DEFAULT_TARGET = (
    ARCHIVE_ROOT / "external_benchmark/her2st24_hespotex/hespotex_exact_2759_v1"
)
DEFAULT_OUTPUT = RESULTS_ROOT / "hespotex/her2_official"
SECTIONS = tuple(
    [f"B{i}" for i in range(1, 7)]
    + [f"C{i}" for i in range(1, 7)]
    + [f"D{i}" for i in range(1, 7)]
    + [f"G{i}" for i in range(1, 4)]
    + [f"H{i}" for i in range(1, 4)]
)

Image.MAX_IMAGE_PIXELS = 2_800_000_000


def gene_panel(bundle: Path) -> tuple[str, ...]:
    genes = tuple(
        line.strip()
        for line in (bundle / "gene_panel.txt").read_text(encoding="utf-8").splitlines()
        if line.strip()
    )
    if len(genes) != 2_759 or len(set(genes)) != 2_759:
        raise ValueError("the exact Fig. 2a panel must contain 2,759 unique rows")
    return genes


def load_raw_section(h5ad_root: Path, section: str, genes: Sequence[str]):
    path = h5ad_root / section / "st_adata_allgene.h5ad"
    value = sc.read_h5ad(path)
    value = value[~value.obs_names.duplicated(keep="first")]
    value = value[:, ~value.var_names.duplicated(keep="first")]
    absent = sorted(set(genes) - set(value.var_names.astype(str)))
    if absent:
        raise ValueError(f"{section} lacks exact panel rows: {absent[:5]}")
    value = value[:, list(genes)].copy()
    matrix = base.dense(value.X)
    if not np.isfinite(matrix).all() or (matrix < 0).any():
        raise ValueError(f"invalid source matrix in {path}")
    return value, path


def bundle_target(
    bundle: Path, section: str, barcodes: Sequence[str], genes: Sequence[str]
) -> np.ndarray:
    with np.load(bundle / "slides" / f"{section}.npz", allow_pickle=False) as archive:
        observed_genes = tuple(archive["genes"].astype(str))
        observed_barcodes = tuple(archive["barcodes"].astype(str))
        target = np.asarray(archive["target"], dtype=np.float32)
    if observed_genes != tuple(genes):
        raise ValueError(f"bundle gene order mismatch for {section}")
    index = {barcode: position for position, barcode in enumerate(observed_barcodes)}
    if set(index) != set(barcodes):
        raise ValueError(f"bundle barcode mismatch for {section}")
    return target[[index[barcode] for barcode in barcodes]]


def prepare_expression(args: argparse.Namespace) -> None:
    genes = gene_panel(args.target_bundle)
    cache = args.output_root / "cache/expression"
    cache.mkdir(parents=True, exist_ok=True)
    if any((cache / f"{section}.npz").exists() for section in SECTIONS):
        raise FileExistsError(f"expression cache already exists under {cache}")
    adatas = []
    source_hashes = {}
    originals = {}
    for section in SECTIONS:
        value, path = load_raw_section(args.h5ad_root, section, genes)
        adatas.append(value)
        originals[section] = tuple(value.obs_names.astype(str))
        source_hashes[section] = base.sha256_file(path)
    processed = base.expression_GVAE(
        adatas=adatas,
        method="GATE",
        feature_method="hvg",
        k_cutoff=8,
        n_top_genes=len(genes),
        device=args.device,
        stagate_epochs=args.stagate_epochs,
    )
    if len(processed) != len(SECTIONS):
        raise ValueError("released expression_GVAE returned wrong section count")
    for section, value in zip(SECTIONS, processed):
        expression = base.dense(value.X).astype(np.float32)
        latent = np.asarray(value.obsm["embedding_rep"], dtype=np.float32)
        barcodes = originals[section]
        expected = bundle_target(args.target_bundle, section, barcodes, genes)
        if expression.shape != expected.shape or not np.allclose(
            expression, expected, rtol=0, atol=2e-5
        ):
            difference = float(np.max(np.abs(expression - expected)))
            raise ValueError(f"released target mismatch for {section}: {difference}")
        if latent.shape != (len(barcodes), 64) or not np.isfinite(latent).all():
            raise ValueError(f"invalid released GATE embedding for {section}")
        raw, _ = load_raw_section(args.h5ad_root, section, genes)
        np.savez_compressed(
            cache / f"{section}.npz",
            gsm=np.asarray(section),
            patient=np.asarray(section[0]),
            author_name=np.asarray(section),
            barcodes=np.asarray(barcodes, dtype=str),
            genes=np.asarray(genes, dtype=str),
            expression=expression,
            latent=latent,
            spatial=np.asarray(raw.obsm["spatial"], dtype=np.int64),
        )
    metadata = {
        "created_at_utc": base.utc_now(),
        "released_repository_commit": base.git_commit(),
        "stage": "released expression_GVAE/GATE",
        "sections": list(SECTIONS),
        "spots": 7_728,
        "genes": 2_759,
        "stagate_epochs": args.stagate_epochs,
        "device": args.device,
        "source_h5ad_sha256": source_hashes,
        "target_bundle": str(args.target_bundle),
        "target_bundle_metadata_sha256": base.sha256_file(
            args.target_bundle / "metadata.json"
        ),
    }
    (cache / "metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def prepare_images(args: argparse.Namespace) -> None:
    genes = gene_panel(args.target_bundle)
    selected = (
        (args.section,)
        if args.section
        else SECTIONS[args.shard_index :: args.num_shards]
    )
    cache = args.output_root / "cache/image"
    cache.mkdir(parents=True, exist_ok=True)
    previous = Path.cwd()
    os.chdir(REPOSITORY_ROOT)
    try:
        encoder = base.ImageEncoder_QuiltNet(model_name="ViT-B-32-quickgelu").to(
            args.device
        )
        encoder.eval()
        for parameter in encoder.parameters():
            parameter.requires_grad = False
        for section in selected:
            output = cache / f"{section}.npz"
            if output.exists():
                raise FileExistsError(output)
            raw, _ = load_raw_section(args.h5ad_root, section, genes)
            spatial = np.asarray(raw.obsm["spatial"], dtype=np.int64)
            image_path = args.image_root / section / "image.jpg"
            mask_path = args.image_root / section / "image.npy"
            image = np.asarray(Image.open(image_path).convert("RGB"))
            mask = np.load(mask_path, mmap_mode="r", allow_pickle=False)
            if mask.shape[:2] != image.shape[:2]:
                raise ValueError(f"image/mask mismatch for {section}")
            features = []
            with torch.no_grad():
                for start in range(0, len(spatial), args.image_batch_size):
                    stop = min(start + args.image_batch_size, len(spatial))
                    batch = base._image_batch(
                        image, spatial, range(start, stop), args.crop_radius
                    ).to(args.device)
                    encoded = encoder(batch)[0]
                    features.append(
                        encoded.detach().cpu().numpy().astype(np.float32)
                    )
            quilt = np.concatenate(features, axis=0)
            nuclei = base._nuclei_values(mask, spatial, args.crop_radius)
            adjacency = base.released_dense_adjacency(spatial, k=4)
            if quilt.shape != (len(spatial), 512):
                raise ValueError(f"unexpected Quilt shape for {section}")
            np.savez_compressed(
                output,
                gsm=np.asarray(section),
                patient=np.asarray(section[0]),
                author_name=np.asarray(section),
                barcodes=np.asarray(raw.obs_names.astype(str), dtype=str),
                spatial=spatial,
                quilt_features=quilt,
                nuclei_number=nuclei,
                adjacency=adjacency,
                crop_width=np.asarray(args.crop_radius * 2),
                source_image_sha256=np.asarray(base.sha256_file(image_path)),
                source_mask_sha256=np.asarray(base.sha256_file(mask_path)),
                quilt_weight_sha256=np.asarray(base.sha256_file(base.QUILT_WEIGHT)),
            )
            print(json.dumps({"cached": str(output), "spots": len(spatial)}), flush=True)
            del raw, image, mask, features, quilt, nuclei, adjacency
            gc.collect()
            torch.cuda.empty_cache()
    finally:
        os.chdir(previous)


def load_datasets(output_root: Path) -> dict[str, base.CachedSectionDataset]:
    return {
        section: base.CachedSectionDataset(
            output_root / "cache/expression" / f"{section}.npz",
            output_root / "cache/image" / f"{section}.npz",
        )
        for section in SECTIONS
    }


def run_fold(args: argparse.Namespace) -> None:
    if args.section not in SECTIONS:
        raise ValueError("fold stage requires a section")
    datasets = load_datasets(args.output_root)
    outer = args.section
    train_ids = [section for section in SECTIONS if section != outer]
    train_dataset = torch.utils.data.ConcatDataset([datasets[value] for value in train_ids])
    test_dataset = torch.utils.data.ConcatDataset([datasets[outer]])
    fold = args.output_root / "folds" / outer
    if (fold / "predictions.npz").exists():
        raise FileExistsError(fold / "predictions.npz")
    fold.mkdir(parents=True, exist_ok=True)
    previous = Path.cwd()
    os.chdir(REPOSITORY_ROOT)
    try:
        predictions, selection = base.released_train.main(
            train_dataset,
            None,
            test_dataset,
            image_embedding=64,
            out_embedding=2_759,
            max_epochs=args.max_epochs,
            save_path=str(fold),
            method="phikon",
            device=args.device,
            return_metadata=True,
        )
    finally:
        os.chdir(previous)
    prediction = np.asarray(predictions[0], dtype=np.float32)
    target = datasets[outer].expression
    values = {
        "pcc": base.per_gene_pcc(target, prediction),
        "kl_divergence": base.per_gene_spatial_kl(
            target, prediction, epsilon=base.DEFAULT_KL_EPSILON
        ),
        "mse": base.per_gene_minmax_mse(target, prediction),
    }
    np.savez_compressed(
        fold / "predictions.npz",
        section_id=np.asarray(outer),
        patient_id=np.asarray(outer[0]),
        barcodes=np.asarray(datasets[outer].barcodes, dtype=str),
        genes=np.asarray(datasets[outer].genes, dtype=str),
        target=target,
        prediction=prediction,
        **values,
    )
    metadata = {
        "created_at_utc": base.utc_now(),
        "released_repository_commit": base.git_commit(),
        "outer_section": outer,
        "patient_id": outer[0],
        "train_sections": train_ids,
        "max_epochs": args.max_epochs,
        "method": "phikon frozen-feature input branch",
        "selection": selection,
        "outer_test_selected": True,
        "mean_pcc": float(values["pcc"].mean()),
        "mean_kl_divergence": float(values["kl_divergence"].mean()),
        "mean_mse": float(values["mse"].mean()),
        "runtime_adaptations": [
            "frozen Quilt features cached in chunks and passed through released phikon branch",
            "exact author Fig. 2a 2,759-row panel and Zenodo nuclei masks",
            "released test-selected checkpoint rule retained",
            "PCC loop algebraically vectorized and device exposed as an argument",
        ],
    }
    (fold / "training.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(metadata, sort_keys=True), flush=True)


def aggregate(args: argparse.Namespace) -> None:
    rows = []
    for section in SECTIONS:
        with np.load(
            args.output_root / "folds" / section / "predictions.npz",
            allow_pickle=False,
        ) as archive:
            genes = archive["genes"].astype(str)
            metrics = {
                key: np.asarray(archive[key], dtype=np.float64)
                for key in ("pcc", "kl_divergence", "mse")
            }
        rows.extend(
            {
                "slide_id": section,
                "patient_id": section[0],
                "gene": gene,
                **{key: float(value[index]) for key, value in metrics.items()},
            }
            for index, gene in enumerate(genes)
        )
    frame = pd.DataFrame(rows)
    metric_names = ["pcc", "kl_divergence", "mse"]
    per_slide = frame.groupby(["slide_id", "patient_id"], as_index=False)[metric_names].mean()
    per_patient = frame.groupby("patient_id", as_index=False)[metric_names].mean()
    per_gene = frame.groupby("gene", sort=False, as_index=False)[metric_names].mean()
    per_gene.insert(1, "is_biological_gene", per_gene["gene"] != "pixel_x")
    boxplot_wide = per_gene.assign(
        point_id=lambda value: "HESpotEx-official-code::" + value["gene"],
        method="HESpotEx official-code rerun",
        feature_order=np.arange(len(per_gene)),
    )[
        [
            "point_id", "method", "feature_order", "gene",
            "is_biological_gene", *metric_names,
        ]
    ]
    boxplot_long = boxplot_wide.melt(
        id_vars=[
            "point_id", "method", "feature_order", "gene",
            "is_biological_gene",
        ],
        value_vars=metric_names,
        var_name="metric",
        value_name="value",
    )
    biological = frame.loc[frame["gene"] != "pixel_x"]
    summary = {
        "created_at_utc": base.utc_now(),
        "label": "HESpotEx released-core HER2+ exact-2759 reproduction",
        "released_repository_commit": base.git_commit(),
        "sections": 24,
        "patients": 5,
        "spots": 7_728,
        "features": 2_759,
        "mean_pcc": float(frame["pcc"].mean()),
        "mean_kl_divergence": float(frame["kl_divergence"].mean()),
        "mean_mse": float(frame["mse"].mean()),
        "biological_2758": {
            key: float(biological[key].mean()) for key in metric_names
        },
        "paper_source_reference": {
            "mean_pcc": 0.32168899062017514,
            "mean_kl_divergence": 1.3205841263209845,
            "mean_mse": 0.19418536815014759,
        },
        "checkpoint_selection": "best outer-test PCC epoch, matching released train.py",
        "exact_unmodified_reproduction": False,
        "kl_epsilon": base.DEFAULT_KL_EPSILON,
    }
    report = args.output_root / "report"
    if report.exists():
        raise FileExistsError(report)
    report.mkdir(parents=True)
    frame.to_csv(report / "per_slide_gene.csv", index=False)
    per_slide.to_csv(report / "per_slide.csv", index=False)
    per_patient.to_csv(report / "per_patient.csv", index=False)
    per_gene.to_csv(report / "per_gene.csv", index=False)
    boxplot_wide.to_csv(report / "fig2a_boxplot_points_wide.csv", index=False)
    boxplot_long.to_csv(report / "fig2a_boxplot_points_long.csv", index=False)
    (report / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, sort_keys=True), flush=True)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("expression", "images", "fold", "aggregate"))
    parser.add_argument("--h5ad-root", type=Path, default=DEFAULT_H5AD_ROOT)
    parser.add_argument("--image-root", type=Path, default=DEFAULT_IMAGE_ROOT)
    parser.add_argument("--target-bundle", type=Path, default=DEFAULT_TARGET)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--section", choices=SECTIONS)
    parser.add_argument("--stagate-epochs", type=int, default=500)
    parser.add_argument("--max-epochs", type=int, default=30)
    parser.add_argument("--image-batch-size", type=int, default=64)
    parser.add_argument("--crop-radius", type=int, default=112)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    args = parser.parse_args(argv)
    if args.stage == "fold" and args.section is None:
        parser.error("fold stage requires --section")
    if args.stagate_epochs < 1 or args.max_epochs < 1:
        parser.error("epoch counts must be positive")
    if args.num_shards < 1 or not 0 <= args.shard_index < args.num_shards:
        parser.error("require num-shards >= 1 and 0 <= shard-index < num-shards")
    if args.section is not None and (args.num_shards != 1 or args.shard_index != 0):
        parser.error("--section cannot be combined with image sharding")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    for key in ("h5ad_root", "image_root", "target_bundle", "output_root"):
        setattr(args, key, getattr(args, key).resolve())
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
