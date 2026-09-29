#!/usr/bin/env python3
"""Within-section adapter applied to the HESPotEx-aligned Lin-MNP runs.

The released HESPotEx runs report Lin-MNP with the spatial adapter
switched off: section-LOSO holds out exactly one section, the released operator
needs at least two sections per patient, so its metadata records
``test_graph = "one section per fold, therefore identity A=I"``.  Those numbers
are therefore pure PFM + Ridge.

The cSCC neighbour control showed that a within-section morphology graph retains
the propagation gain without access to serial sections. This runner applies the
same form of graph to archived pointwise Ridge predictions without refitting
the predictor.

Views.  The HESPotEx descriptors are multiscale CTransPath at 96/192/384 px,
already L2-normalized per scale.  The Ridge input is scale index 2 (384 px), so
that is the ``context`` view; scale index 1 (192 px) is the ``local`` view, the
closest analogue of the 224 px local core used on cSCC.

Arms.  ``identity`` must reproduce the published aggregate exactly, which is the
correctness check for this harness.  The declared transfer arm is
``borda_k128``: the same form and neighbour count as the matched within-section
arm on cSCC, carried over without retuning.  The remaining arms are reported as
a sensitivity grid, not as candidates for selection.

Targets are opened only after every adapted prediction has been produced.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np
import pandas as pd

from lin_mnp import model
from lin_mnp.within_section import apply as apply_within_section
from lin_mnp.benchmark import hespotex_metrics
from lin_mnp.benchmark import metrics as benchmark_metrics
from lin_mnp.paths import ARCHIVE_ROOT, DATA_ROOT, RESULTS_ROOT

ARCHIVE = ARCHIVE_ROOT
DATA = DATA_ROOT
HESPOTEX_ROOT = ARCHIVE / "hespotex/lin_mnp"
CONTRACT_ROOT = ARCHIVE / "hespotex/contracts"

OUTPUT = RESULTS_ROOT / "transfer/hespotex"

CONFIGS = {
    "borda": {"fusion": "borda", "weighting": "softmax"},
    "minimal": {"fusion": "score", "weighting": "uniform"},
}
K_GRID = (8, 16, 32, 64, 128)
DECLARED_ARM = "borda_k128"
PUBLISHED = {"cscc": 0.283211, "her2": 0.340504}


def decode(value: np.ndarray) -> tuple[str, ...]:
    return tuple(str(item) for item in np.asarray(value).tolist())


def contract(cohort: str) -> dict:
    return json.loads((CONTRACT_ROOT / f"{cohort}.json").read_text(encoding="utf-8"))


def cohort_paths(cohort: str) -> tuple[Path, Path]:
    root = DATA / ("her2" if cohort == "her2" else "cscc") / "derived"
    panel = "hespotex_exact_2759_v1" if cohort == "her2" else "hespotex_exact_1971_v1"
    return (
        root / panel,
        root / "shared_image_descriptors/ctranspath_multiscale_px96-192-384/slides",
    )


def load_views(section: str, descriptor_root: Path):
    with np.load(descriptor_root / f"{section}.npz", allow_pickle=False) as payload:
        if bool(payload["target_data_accessed"].item()):
            raise ValueError("descriptor declares target access")
        if str(payload["sample_id"].item()) != section:
            raise ValueError("descriptor section mismatch")
        pooled = np.asarray(payload["pooled"], dtype=np.float32)
        barcodes = decode(payload["barcodes"])
    if pooled.ndim != 3 or pooled.shape[1:] != (3, 768):
        raise ValueError("expected multiscale CTransPath [spots,3,768]")
    return pooled[:, 2, :], pooled[:, 1, :], barcodes


def within_section_apply(prediction, context, local, k, fusion, weighting):
    """Average each spot's prediction over its k nearest neighbours in the section."""
    return apply_within_section(
        prediction,
        context,
        local,
        k,
        fusion=fusion,
        weighting=weighting,
    )


def build(args):
    output = args.output / args.cohort
    if output.exists():
        raise SystemExit(f"refusing to overwrite: {output}")
    specification = contract(args.cohort)
    _, descriptor_root = cohort_paths(args.cohort)

    output.mkdir(parents=True)
    records = []
    for fold in specification["folds"]:
        fold_id = int(fold["fold_id"])
        section = str(fold["test_sample_ids"][0])
        source = HESPOTEX_ROOT / args.cohort / "fixed" / f"fold{fold_id}/predictions.npz"
        with np.load(source, allow_pickle=False) as archive:
            ridge = archive["prediction"].astype(np.float32)
            prediction_barcodes = decode(archive["barcodes"])
            genes = decode(archive["genes"])
        context, local, barcodes = load_views(section, descriptor_root)
        if barcodes != prediction_barcodes:
            raise ValueError(f"barcode mismatch on {section}")
        if len(ridge) != len(context):
            raise ValueError(f"spot count mismatch on {section}")

        arms = {"identity": ridge}
        effective = {}
        for name, spec in CONFIGS.items():
            for k in K_GRID:
                arm = f"{name}_k{k}"
                arms[arm], effective[arm] = within_section_apply(
                    ridge, context, local, k, **spec
                )
        np.savez_compressed(
            output / f"fold{fold_id}.npz",
            barcodes=np.asarray(barcodes), genes=np.asarray(genes),
            **{name: value.astype(np.float32) for name, value in arms.items()},
        )
        records.append({
            "fold_id": fold_id, "section": section, "spots": int(len(ridge)),
            "effective_k": effective, "ridge_source": str(source),
        })
        print(json.dumps({
            "cohort": args.cohort, "fold": fold_id, "section": section,
            "spots": int(len(ridge)),
        }), flush=True)

    (output / "build_metadata.json").write_text(json.dumps({
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "cohort": args.cohort,
        "views": "context = 384 px scale 2 (Ridge input); local = 192 px scale 1",
        "configs": CONFIGS,
        "k_grid": list(K_GRID),
        "declared_transfer_arm": DECLARED_ARM,
        "inverse_temperature": model.INVERSE_TEMPERATURE,
        "ridge_predictions_reused_verbatim": True,
        "target_data_opened": False,
        "folds": records,
    }, indent=2) + "\n")


def evaluate(args):
    specification = contract(args.cohort)
    target_root, _ = cohort_paths(args.cohort)
    root = args.output / args.cohort
    arms = ["identity"] + [f"{n}_k{k}" for n in CONFIGS for k in K_GRID]

    rows = []
    for fold in specification["folds"]:
        fold_id = int(fold["fold_id"])
        section = str(fold["test_sample_ids"][0])
        patient = specification["patient_by_section"][section]
        with np.load(target_root / "slides" / f"{section}.npz", allow_pickle=False) as p:
            target = np.asarray(p["target"], dtype=np.float32)
            target_barcodes = decode(p["barcodes"])
            target_genes = decode(p["genes"])
        with np.load(root / f"fold{fold_id}.npz", allow_pickle=False) as archive:
            barcodes = decode(archive["barcodes"])
            genes = decode(archive["genes"])
            gene_position = {value.casefold(): index for index, value in enumerate(genes)}
            barcode_position = {value: index for index, value in enumerate(barcodes)}
            try:
                gene_order = np.asarray(
                    [gene_position[value.casefold()] for value in target_genes]
                )
                barcode_order = np.asarray(
                    [barcode_position[value] for value in target_barcodes]
                )
            except KeyError as error:
                raise ValueError(f"prediction identity mismatch on {section}") from error
            for arm in arms:
                prediction = archive[arm].astype(np.float32)[barcode_order][:, gene_order]
                pcc = hespotex_metrics.per_gene_pcc(target, prediction)
                for gene, value in zip(target_genes, pcc):
                    rows.append({
                        "arm": arm, "fold_id": fold_id, "section": section,
                        "patient_id": patient, "gene": gene, "pcc": float(value),
                    })
    detail = pd.DataFrame(rows)
    section_gene = detail.groupby(
        ["arm", "section", "patient_id", "gene"], as_index=False
    ).pcc.mean()
    if args.cohort == "her2":
        gene = section_gene.groupby(["arm", "gene"], as_index=False).pcc.mean()
    else:
        patient_gene = section_gene.groupby(
            ["arm", "patient_id", "gene"], as_index=False
        ).pcc.mean()
        gene = patient_gene.groupby(["arm", "gene"], as_index=False).pcc.mean()
    summary = gene.groupby("arm", as_index=False).pcc.mean()
    identity = float(summary.loc[summary.arm == "identity", "pcc"].iloc[0])
    summary["delta_vs_identity"] = summary.pcc - identity

    published = PUBLISHED[args.cohort]
    drift = abs(identity - published)

    out = root / "evaluation"
    out.mkdir(parents=True, exist_ok=True)
    summary.to_csv(out / "summary.csv", index=False)
    gene.to_csv(out / "per_gene.csv.gz", index=False)
    (out / "metadata.json").write_text(json.dumps({
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "cohort": args.cohort,
        "aggregation": (
            "section-gene -> equal 24-section gene -> panel" if args.cohort == "her2"
            else "section-gene -> patient-gene -> equal four-patient gene -> panel"
        ),
        "identity_pcc": identity,
        "reported_identity_pcc": published,
        "identity_vs_published_abs_difference": drift,
        "declared_transfer_arm": DECLARED_ARM,
        "declared_transfer_arm_pcc": float(
            summary.loc[summary.arm == DECLARED_ARM, "pcc"].iloc[0]
        ),
    }, indent=2) + "\n")

    print(summary.set_index("arm").loc[arms].to_string())
    print()
    print(f"identity {identity:.6f} vs published {published:.6f}  |diff| {drift:.6f}")


DISPLAY = ("pcc", "mi", "auc_0", "mse_minmax", "ssim", "kl_divergence")


def per_gene_metrics(target: np.ndarray, prediction: np.ndarray) -> pd.DataFrame:
    """Compute the six HESpotEx-aligned metrics without archived code imports."""

    rows = []
    pcc = hespotex_metrics.per_gene_pcc(target, prediction)
    mse = hespotex_metrics.per_gene_minmax_mse(target, prediction)
    kl = hespotex_metrics.per_gene_spatial_kl(target, prediction)
    for index in range(target.shape[1]):
        truth = target[:, index]
        estimate = prediction[:, index]
        rows.append(
            {
                "pcc": float(pcc[index]),
                "mi": benchmark_metrics.normalized_mutual_information(truth, estimate),
                "auc_0": benchmark_metrics.threshold_auc(truth, estimate, 0.0),
                "mse_minmax": float(mse[index]),
                "ssim": benchmark_metrics.spatial_ssim(truth, estimate),
                "kl_divergence": float(kl[index]),
            }
        )
    return pd.DataFrame(rows)


def metrics(args):
    """Compute the six reported metrics for selected prediction arms."""
    specification = contract(args.cohort)
    target_root, _ = cohort_paths(args.cohort)
    root = args.output / args.cohort
    frames = []
    for fold in specification["folds"]:
        fold_id = int(fold["fold_id"])
        section = str(fold["test_sample_ids"][0])
        with np.load(root / f"fold{fold_id}.npz", allow_pickle=False) as archive:
            genes = np.asarray(archive["genes"])
            barcodes = np.asarray(archive["barcodes"])
            with np.load(
                target_root / "slides" / f"{section}.npz", allow_pickle=False
            ) as target_archive:
                target = np.asarray(target_archive["target"], dtype=np.float32)
                target_genes = decode(target_archive["genes"])
                target_barcodes = decode(target_archive["barcodes"])
            gene_position = {
                str(value).casefold(): index for index, value in enumerate(genes)
            }
            barcode_position = {str(value): index for index, value in enumerate(barcodes)}
            gene_order = np.asarray(
                [gene_position[value.casefold()] for value in target_genes]
            )
            barcode_order = np.asarray(
                [barcode_position[value] for value in target_barcodes]
            )
            for arm in args.arms:
                prediction = np.asarray(archive[arm], dtype=np.float32)[barcode_order][
                    :, gene_order
                ]
                frame = per_gene_metrics(target, prediction)
                frame.insert(0, "gene", target_genes)
                frame.insert(0, "patient_id", specification["patient_by_section"][section])
                frame.insert(0, "slide_id", section)
                frame.insert(0, "fold_id", fold_id)
                frame.insert(0, "method", arm)
                frame.insert(0, "variant", arm)
                frame.insert(0, "cohort", args.cohort)
                frames.append(frame)
        print(f"[metrics] fold {fold_id} done", flush=True)

    detail = pd.concat(frames, ignore_index=True)
    columns = list(DISPLAY)
    section_gene = detail.groupby(
        ["variant", "slide_id", "patient_id", "gene"], as_index=False
    )[columns].mean()
    if args.cohort == "her2":
        gene = section_gene.groupby(["variant", "gene"], as_index=False)[columns].mean()
    else:
        patient_gene = section_gene.groupby(
            ["variant", "patient_id", "gene"], as_index=False
        )[columns].mean()
        gene = patient_gene.groupby(["variant", "gene"], as_index=False)[columns].mean()
    summary = gene.groupby("variant", as_index=False)[columns].mean()

    out = root / "evaluation"
    out.mkdir(parents=True, exist_ok=True)
    summary.to_csv(out / "six_metric_summary.csv", index=False)
    print(summary.set_index("variant").loc[list(args.arms)].to_string())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    for name in ("build", "evaluate", "metrics"):
        p = sub.add_parser(name)
        p.add_argument("--cohort", choices=("cscc", "her2"), required=True)
        p.add_argument("--output", type=Path, default=OUTPUT)
        if name == "metrics":
            p.add_argument("--arms", nargs="+", default=["identity", DECLARED_ARM])
    args = parser.parse_args()
    args.output = args.output.resolve()
    {"build": build, "evaluate": evaluate, "metrics": metrics}[args.action](args)


if __name__ == "__main__":
    main()
