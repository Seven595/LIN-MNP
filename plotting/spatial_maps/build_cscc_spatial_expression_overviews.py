#!/usr/bin/env python3
"""Rebuild the established spatial-expression overview with Lin-MNP.

The visual design, comparison-method roster, gene-selection categories, tissue
geometry and per-panel scaling are imported from the audited v4 overview code.
Only the Lin-MNP prediction source and the derived selection evidence are
replaced.  This keeps the new figures visually identical to the established
panels while preventing the promoted model from inheriting an older gene list.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib
import json
import multiprocessing as mp
from pathlib import Path
import sys
from typing import Any, Sequence

import numpy as np
import pandas as pd

from lin_mnp.paths import ARCHIVE_ROOT, RESULTS_ROOT

OUTPUT_ROOT = RESULTS_ROOT / "figures/spatial_maps"
LEGACY_ROOT = (
    ARCHIVE_ROOT / "spatial_maps/legacy"
)
LEGACY_CODE = LEGACY_ROOT / "code"
LIN_MNP_BENCHMARK_ROOT = RESULTS_ROOT / "cscc/folds"
LIN_MNP_HESPOTEX_ROOT = ARCHIVE_ROOT / "hespotex/lin_mnp"
LIN_MNP_BENCHMARK_METRICS = (
    ARCHIVE_ROOT / "benchmark/lin_mnp/per_section_gene_metrics.csv.gz"
)
LIN_MNP_HESPOTEX_METRICS = (
    ARCHIVE_ROOT / "hespotex/lin_mnp/per_section_gene_metrics.csv.gz"
)
LEGACY_HESPOTEX_INDEX = (
    ARCHIVE_ROOT / "hespotex/prediction_artifact_index.csv"
)
LEGACY_HESPOTEX_METRICS = (
    ARCHIVE_ROOT / "hespotex/per_section_gene_metrics.csv.gz"
)
LEGACY_SPOT_DETAIL = LEGACY_ROOT / "spot_expression_detail.csv.gz"

# The original full-matrix artifacts for these six HER2+ methods were removed
# during an earlier result cleanup.  The audited v4 package still preserves the
# exact per-spot values for every gene that it plotted.  Reconstruct only the
# representative-section matrices needed here from that lossless table, while
# retaining the complete 2,759-gene metric table for rank calculation.
RECONSTRUCTED_HER2_METHODS = (
    "TCGN",
    "THItoGene",
    "Hist2ST",
    "mclSTExp",
    "BLEEP",
    "IGI-DL",
)
LEGACY_MODEL_NAME = "AnchorFlow-ST"
MODEL_NAME = "Lin-MNP"

legacy_build: Any = None
legacy_plot: Any = None
ORIGINAL_BENCHMARK_INDEX: Any = None
ORIGINAL_BUILD_CSCC_METRICS: Any = None
ORIGINAL_WRITE_READMES: Any = None

_BENCHMARK_INDEX: pd.DataFrame | None = None
_HER2_METRICS: pd.DataFrame | None = None
_HER2_RENDERABLE_GENES: frozenset[str] | None = None


def load_legacy_builders() -> None:
    """Load the companion plotting helpers and report a portable setup error."""

    global legacy_build, legacy_plot
    global ORIGINAL_BENCHMARK_INDEX, ORIGINAL_BUILD_CSCC_METRICS
    global ORIGINAL_WRITE_READMES
    if legacy_build is not None:
        return
    if not LEGACY_CODE.is_dir():
        raise FileNotFoundError(
            "cSCC spatial maps require the companion bundle at "
            f"{LEGACY_CODE}; set LIN_MNP_ARCHIVE_ROOT"
        )
    if str(LEGACY_CODE) not in sys.path:
        sys.path.insert(0, str(LEGACY_CODE))
    legacy_build = importlib.import_module("build_spatial_expression_overviews")
    legacy_plot = importlib.import_module("plot_spatial_expression_overviews")
    ORIGINAL_BENCHMARK_INDEX = legacy_plot.benchmark_prediction_index
    ORIGINAL_BUILD_CSCC_METRICS = legacy_plot.build_cscc_common_metrics
    ORIGINAL_WRITE_READMES = legacy_build.write_readmes


def argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--dpi", type=int, default=220)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--prepare-only", action="store_true")
    return parser


# Help should remain available before optional manuscript artifacts are
# installed.  argparse exits here after printing it; real runs continue to the
# dependency loader below.
if __name__ == "__main__" and any(arg in {"-h", "--help"} for arg in sys.argv[1:]):
    argument_parser().parse_args()


# The renderer subclasses a class from the companion plotting bundle, so that
# dependency must be available before the subclass is defined.  Keeping this in
# one loader gives users a clear configuration error instead of an obscure
# import or attribute error.
load_legacy_builders()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def decode(values: np.ndarray) -> tuple[str, ...]:
    return tuple(
        value.decode("utf-8") if isinstance(value, bytes) else str(value)
        for value in np.asarray(values).reshape(-1)
    )


def prepare_cscc_prediction_index(output: Path) -> pd.DataFrame:
    """Replace legacy model rows with section-level Lin-MNP artifacts."""

    old = ORIGINAL_BENCHMARK_INDEX()
    old = old.loc[~old["model"].eq(LEGACY_MODEL_NAME)].copy()
    derived_root = output / "source_data/lin_mnp_predictions/cscc_benchmark_fixed997"
    derived_root.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    source_rows: list[dict[str, Any]] = []

    for fold_id in range(4):
        source_path = LIN_MNP_BENCHMARK_ROOT / f"fold{fold_id}/outer_test_predictions.npz"
        if not source_path.is_file():
            raise FileNotFoundError(source_path)
        with np.load(source_path, allow_pickle=False) as archive:
            predictions = np.asarray(archive["lin_mnp"], dtype=np.float32)
            sections = np.asarray(decode(archive["sections"]))
            barcodes = np.asarray(decode(archive["barcodes"]))
            genes = np.asarray(decode(archive["genes"]))
        if predictions.shape != (len(sections), len(genes)):
            raise ValueError(f"invalid LIN_MNP Benchmark artifact: {source_path}")
        for section in sorted(set(sections.tolist())):
            active = sections == section
            slide = legacy_plot.load_slides()[0][section]
            section_barcodes = tuple(barcodes[active].tolist())
            if set(section_barcodes) != set(slide.barcodes):
                raise ValueError(f"LIN_MNP barcode mismatch: {section}")
            target_path = derived_root / f"{section}.npz"
            np.savez_compressed(
                target_path,
                prediction=predictions[active],
                barcodes=barcodes[active],
                genes=genes,
                fold_id=np.asarray(fold_id, dtype=np.int32),
                section_id=np.asarray(section),
                patient_id=np.asarray(slide.patient_id),
                source_prediction_space=np.asarray(
                    "within-section gene-z-scored log1p(raw counts)"
                ),
                source_path=np.asarray(str(source_path.resolve())),
            )
            checksum = sha256_file(target_path)
            rows.append(
                {
                    "method_id": "lin_mnp",
                    "model": MODEL_NAME,
                    "slide_id": section,
                    "patient_id": slide.patient_id,
                    "fold_id": fold_id,
                    "path": str(target_path.resolve()),
                    "sha256": checksum,
                    "contains_spot_prediction": True,
                    "contains_truth": False,
                    "sample_id": section,
                    "section": section,
                }
            )
            source_rows.append(
                {
                    "dataset": "cSCC Benchmark fixed-997",
                    "section": section,
                    "patient_id": slide.patient_id,
                    "fold_id": fold_id,
                    "source_path": str(source_path.resolve()),
                    "source_sha256": sha256_file(source_path),
                    "derived_path": str(target_path.resolve()),
                    "derived_sha256": checksum,
                    "model": "Lin-MNP",
                }
            )

    latest = pd.DataFrame(rows)
    if len(latest) != 12 or latest["slide_id"].nunique() != 12:
        raise ValueError(f"expected 12 LIN_MNP Benchmark sections, found {len(latest)}")
    index = pd.concat([old, latest], ignore_index=True, sort=False)
    index.to_csv(output / "source_data/benchmark_prediction_index.csv", index=False)
    pd.DataFrame(source_rows).to_csv(
        output / "source_data/lin_mnp_prediction_provenance.csv", index=False
    )
    return index


def latest_benchmark_prediction_index() -> pd.DataFrame:
    if _BENCHMARK_INDEX is None:
        raise RuntimeError("LIN_MNP Benchmark prediction index has not been prepared")
    return _BENCHMARK_INDEX.copy()


class LIN_MNPCsccCommonSpace(legacy_plot.CsccCommonSpace):
    """Use native LIN_MNP scores where PCC and panel min-max are affine invariant."""

    def prediction(
        self,
        method: str,
        path: Path,
        section: str,
        target_barcodes: tuple[str, ...],
        target_genes: tuple[str, ...],
    ) -> np.ndarray:
        if method != MODEL_NAME:
            return super().prediction(
                method, path, section, target_barcodes, target_genes
            )
        key = (method, str(path.resolve()))
        if key in self._prediction_cache:
            return self._prediction_cache[key]
        slide = self.slides[section]
        if slide.barcodes != target_barcodes or slide.genes != target_genes:
            raise ValueError(f"LIN_MNP common-space identity mismatch: {section}")
        native, _, _ = legacy_plot._aligned_prediction_archive(
            path, target_barcodes, target_genes
        )
        if not np.isfinite(native).all():
            raise ValueError(f"invalid LIN_MNP prediction: {section}")
        self._prediction_cache[key] = native
        self._audit[key] = {
            "method": method,
            "section": section,
            "prediction_path": str(path.resolve()),
            "metadata_path": str(path.resolve()),
            "source_prediction_space": (
                "within-section gene-z-scored log1p(raw counts)"
            ),
            "common_target_space": legacy_plot.CSCC_COMMON_TARGET_SPACE,
            "conversion": (
                "native affine coordinates retained: PCC and independent panel "
                "min-max rendering are invariant to the omitted positive affine map"
            ),
            "n_values": int(native.size),
            "n_clipped_before_log1p": 0,
        }
        return native


def build_cscc_metrics(prediction_index: pd.DataFrame) -> pd.DataFrame:
    result = ORIGINAL_BUILD_CSCC_METRICS(prediction_index)
    observed = result.loc[result["model"].eq(MODEL_NAME)].copy()
    expected = pd.read_csv(
        LIN_MNP_BENCHMARK_METRICS,
        usecols=["slide_id", "patient_id", "gene", "pcc"],
    )
    expected = expected.loc[
        expected["slide_id"].isin(legacy_plot.COMMON_ISTAR_SECTIONS)
    ]
    check = observed.merge(
        expected,
        on=["slide_id", "patient_id", "gene"],
        how="inner",
        suffixes=("_render", "_archive"),
        validate="one_to_one",
    )
    if len(check) != 4 * 997:
        raise ValueError(f"incomplete LIN_MNP cSCC metric audit: {len(check)}")
    delta = np.nanmax(np.abs(check["pcc_render"] - check["pcc_archive"]))
    if not np.isfinite(delta) or delta > 2e-6:
        raise ValueError(f"LIN_MNP cSCC metric mismatch: max |delta|={delta}")
    return result


def prepare_her2_index(output: Path) -> Path:
    """Create the legacy-format index with fixed-contract LIN_MNP predictions."""

    old = pd.read_csv(LEGACY_HESPOTEX_INDEX, dtype={"slide_id": str})
    anchor_targets = old.loc[
        old["dataset"].eq("HER2+") & old["method"].eq(LEGACY_MODEL_NAME),
        ["slide_id", "target_path"],
    ].set_index("slide_id")["target_path"]
    retained = old.loc[
        ~(old["dataset"].eq("HER2+") & old["method"].eq(LEGACY_MODEL_NAME))
    ].copy()
    rows: list[dict[str, str]] = []
    for path in sorted((LIN_MNP_HESPOTEX_ROOT / "her2/fixed").glob("fold*/predictions.npz")):
        with np.load(path, allow_pickle=False) as archive:
            section = str(archive["sample_id"].item())
            patient = str(archive["patient_id"].item())
        rows.append(
            {
                "dataset": "HER2+",
                "method_id": "lin_mnp",
                "method": MODEL_NAME,
                "slide_id": section,
                "patient_id": patient,
                "prediction_path": str(path.resolve()),
                "target_path": str(anchor_targets.loc[section]),
            }
        )
    latest = pd.DataFrame(rows)
    if len(latest) != 24 or latest["slide_id"].nunique() != 24:
        raise ValueError(f"expected 24 LIN_MNP HER2+ sections, found {len(latest)}")
    index = pd.concat([retained, latest], ignore_index=True, sort=False)
    index = reconstruct_missing_her2_render_artifacts(index, output)
    target = output / "source_data/hespotex_prediction_artifact_index.csv"
    index.to_csv(target, index=False)
    return target


def load_legacy_her2_render_values() -> pd.DataFrame:
    """Load exact legacy values needed to replace deleted prediction matrices."""

    columns = [
        "dataset",
        "gene",
        "slide_id",
        "barcode",
        "method",
        "expression_value",
    ]
    frames: list[pd.DataFrame] = []
    for chunk in pd.read_csv(
        LEGACY_SPOT_DETAIL,
        usecols=columns,
        dtype={"slide_id": str, "barcode": str},
        chunksize=250_000,
    ):
        active = (
            chunk["dataset"].eq("her2")
            & chunk["slide_id"].isin(legacy_build.HER2_REPRESENTATIVE_SECTIONS)
            & chunk["method"].isin(RECONSTRUCTED_HER2_METHODS)
        )
        if active.any():
            frames.append(chunk.loc[active].copy())
    if not frames:
        raise ValueError(f"no recoverable HER2+ spot values in {LEGACY_SPOT_DETAIL}")
    result = pd.concat(frames, ignore_index=True)
    if result.duplicated(["method", "slide_id", "gene", "barcode"]).any():
        raise ValueError("duplicate rows in recoverable HER2+ spot-value table")
    return result


def reconstruct_missing_her2_render_artifacts(
    index: pd.DataFrame, output: Path
) -> pd.DataFrame:
    """Materialize lossless stand-ins for six deleted HER2+ panel artifacts."""

    global _HER2_RENDERABLE_GENES
    detail = load_legacy_her2_render_values()
    _HER2_RENDERABLE_GENES = frozenset(detail["gene"].astype(str).unique())
    if len(_HER2_RENDERABLE_GENES) != 45:
        raise ValueError(
            "legacy HER2+ render catalog changed: "
            f"{len(_HER2_RENDERABLE_GENES)} genes, expected 45"
        )
    derived_root = output / "source_data/reconstructed_legacy_her2_predictions"
    derived_root.mkdir(parents=True, exist_ok=True)
    reconstruction_rows: list[dict[str, Any]] = []
    updated = index.copy()
    source_checksum = sha256_file(LEGACY_SPOT_DETAIL)

    for section in legacy_build.HER2_REPRESENTATIVE_SECTIONS:
        target = legacy_plot.load_target(legacy_plot.COHORTS["her2"], section)
        barcodes = tuple(map(str, target["barcodes"]))
        genes = tuple(map(str, target["genes"]))
        barcode_index = {barcode: idx for idx, barcode in enumerate(barcodes)}
        gene_index = {gene: idx for idx, gene in enumerate(genes)}
        section_detail = detail.loc[detail["slide_id"].eq(section)]
        for method in RECONSTRUCTED_HER2_METHODS:
            current = section_detail.loc[section_detail["method"].eq(method)]
            observed_genes = set(current["gene"].astype(str))
            if observed_genes != set(_HER2_RENDERABLE_GENES):
                raise ValueError(f"incomplete reconstructed genes: {method}/{section}")
            prediction = np.zeros((len(barcodes), len(genes)), dtype=np.float32)
            for row in current.itertuples(index=False):
                try:
                    i = barcode_index[str(row.barcode)]
                    j = gene_index[str(row.gene)]
                except KeyError as exc:
                    raise ValueError(
                        f"reconstruction identity mismatch: {method}/{section}/{exc}"
                    ) from exc
                prediction[i, j] = float(row.expression_value)
            path = derived_root / method.lower().replace("-", "_") / f"{section}.npz"
            path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                path,
                prediction=prediction,
                barcodes=np.asarray(barcodes),
                genes=np.asarray(genes),
                reconstructed_gene_mask=np.asarray(
                    [gene in _HER2_RENDERABLE_GENES for gene in genes], dtype=np.bool_
                ),
                source_spot_detail=np.asarray(str(LEGACY_SPOT_DETAIL.resolve())),
                source_spot_detail_sha256=np.asarray(source_checksum),
            )
            mask = (
                updated["dataset"].eq("HER2+")
                & updated["method"].eq(method)
                & updated["slide_id"].astype(str).eq(section)
            )
            if int(mask.sum()) != 1:
                raise ValueError(f"HER2+ index mismatch: {method}/{section}")
            updated.loc[mask, "prediction_path"] = str(path.resolve())
            reconstruction_rows.append(
                {
                    "dataset": "HER2+",
                    "method": method,
                    "slide_id": section,
                    "derived_path": str(path.resolve()),
                    "derived_sha256": sha256_file(path),
                    "source_path": str(LEGACY_SPOT_DETAIL.resolve()),
                    "source_sha256": source_checksum,
                    "n_reconstructed_genes": len(_HER2_RENDERABLE_GENES),
                    "n_total_gene_columns": len(genes),
                    "noncatalog_columns_are_placeholders": True,
                }
            )
    pd.DataFrame(reconstruction_rows).to_csv(
        output / "source_data/reconstructed_legacy_her2_provenance.csv", index=False
    )
    return updated


def prepare_her2_metrics(output: Path) -> pd.DataFrame:
    methods = set(legacy_plot.HER2_METHODS)
    methods.discard(LEGACY_MODEL_NAME)
    methods.add(MODEL_NAME)
    baseline = pd.read_csv(
        LEGACY_HESPOTEX_METRICS,
        low_memory=False,
        usecols=["dataset", "method", "slide_id", "patient_id", "gene", "pcc"],
    )
    baseline = baseline.loc[
        baseline["dataset"].eq("HER2+")
        & baseline["method"].isin(methods - {MODEL_NAME})
    ].copy()
    latest = pd.read_csv(
        LIN_MNP_HESPOTEX_METRICS,
        usecols=["cohort", "slide_id", "patient_id", "gene", "pcc"],
    )
    latest = latest.loc[latest["cohort"].eq("her2")].drop(columns="cohort")
    latest.insert(0, "dataset", "HER2+")
    latest.insert(1, "method", MODEL_NAME)
    result = pd.concat([baseline, latest], ignore_index=True, sort=False)
    expected = 24 * 2759 * len(methods)
    if len(result) != expected:
        raise ValueError(f"HER2+ metric coverage mismatch: {len(result)}/{expected}")
    if result.duplicated(["method", "slide_id", "gene"]).any():
        raise ValueError("duplicate HER2+ method/section/gene metrics")
    result.to_csv(
        output / "source_data/her2_per_section_gene_pcc.csv.gz",
        index=False,
        compression="gzip",
    )
    return result[["slide_id", "patient_id", "gene", "method", "pcc"]]


def latest_her2_section_metrics() -> pd.DataFrame:
    if _HER2_METRICS is None:
        raise RuntimeError("LIN_MNP HER2+ metrics have not been prepared")
    return _HER2_METRICS.copy()


def select_top_genes_with_renderable_her2(
    metrics: pd.DataFrame, dataset: str
) -> pd.DataFrame:
    """Apply the original rule after an artifact-availability gate for HER2+."""

    if dataset == "her2":
        if _HER2_RENDERABLE_GENES is None:
            raise RuntimeError("HER2+ renderable-gene catalog has not been prepared")
        metrics = metrics.loc[metrics["gene"].isin(_HER2_RENDERABLE_GENES)].copy()
    return legacy_build._original_select_top_genes(metrics, dataset)


def select_large_margin_with_renderable_her2(
    metrics: pd.DataFrame, dataset: str
) -> pd.DataFrame:
    """Apply the original rule after an artifact-availability gate for HER2+."""

    if dataset == "her2":
        if _HER2_RENDERABLE_GENES is None:
            raise RuntimeError("HER2+ renderable-gene catalog has not been prepared")
        metrics = metrics.loc[metrics["gene"].isin(_HER2_RENDERABLE_GENES)].copy()
    return legacy_build._original_select_large_margin(metrics, dataset)


def write_readmes(output: Path, selected: pd.DataFrame) -> None:
    ORIGINAL_WRITE_READMES(output, selected)
    for name in ("cscc_benchmark_fixed997", "her2_hespotex_exact2759"):
        path = output / name / "README.md"
        text = path.read_text(encoding="utf-8")
        text = text.replace("# cscc spatial-expression overviews", "# cSCC Lin-MNP spatial-expression overviews")
        text = text.replace("# her2 spatial-expression overviews", "# HER2+ Lin-MNP spatial-expression overviews")
        text = text.replace(
            "cSCC retains\n  the exhaustive ten qualifying genes",
            "cSCC retains\n  the ten highest-margin qualifying genes",
        )
        text += (
            "\nThe Lin-MNP panels and all gene-selection statistics in this directory "
            "were recomputed from the fixed-contract Lin-MNP predictions. "
            "The comparison-method predictions and the established v4 visual design "
            "are unchanged.\n"
        )
        path.write_text(text, encoding="utf-8")


def configure(output: Path) -> None:
    global _BENCHMARK_INDEX, _HER2_METRICS
    output.mkdir(parents=True, exist_ok=True)
    (output / "source_data").mkdir(exist_ok=True)
    _BENCHMARK_INDEX = prepare_cscc_prediction_index(output)
    local_index = prepare_her2_index(output)
    _HER2_METRICS = prepare_her2_metrics(output)

    common_space = LIN_MNPCsccCommonSpace()
    legacy_plot.HER2_METHODS = tuple(
        MODEL_NAME if method == LEGACY_MODEL_NAME else method
        for method in legacy_plot.HER2_METHODS
    )
    legacy_plot._CSCC_COMMON_SPACE = common_space
    legacy_plot.benchmark_prediction_index = latest_benchmark_prediction_index
    legacy_plot.build_cscc_common_metrics = build_cscc_metrics
    legacy_plot.LOCAL_INDEX = local_index

    legacy_build.benchmark_prediction_index = latest_benchmark_prediction_index
    legacy_build.current_benchmark_prediction_index = latest_benchmark_prediction_index
    legacy_build.build_cscc_common_metrics = build_cscc_metrics
    legacy_build.load_her2_section_metrics = latest_her2_section_metrics
    legacy_build.LOCAL_INDEX = local_index
    legacy_build.HER2_LARGE_MARGIN = 0.05
    if not hasattr(legacy_build, "_original_select_top_genes"):
        legacy_build._original_select_top_genes = legacy_build.select_top_genes
    if not hasattr(legacy_build, "_original_select_large_margin"):
        legacy_build._original_select_large_margin = legacy_build.select_large_margin
    legacy_build.select_top_genes = select_top_genes_with_renderable_her2
    legacy_build.select_large_margin = select_large_margin_with_renderable_her2
    legacy_build.write_readmes = write_readmes
    legacy_build.PRESERVED_SUPPLEMENTAL_ROOT_FIGURES = {
        "cscc_benchmark_fixed997": (),
        "her2_hespotex_exact2759": (),
    }


def update_metadata(output: Path) -> None:
    provenance_path = output / "provenance.json"
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    provenance.update(
        {
            "created_by": str(Path(__file__).resolve()),
            "created_by_sha256": sha256_file(Path(__file__).resolve()),
            "style_and_selection_logic_reference": str(LEGACY_ROOT.resolve()),
            "lin_mnp_model": "Lin-MNP",
            "lin_mnp_benchmark_source": str(LIN_MNP_BENCHMARK_ROOT.resolve()),
            "lin_mnp_hespotex_source": str(LIN_MNP_HESPOTEX_ROOT.resolve()),
            "her2_large_margin_threshold": 0.05,
            "her2_large_margin_threshold_note": (
                "Updated from 0.25 because the promoted fixed-contract model has "
                "a smaller empirical margin range; the category still requires "
                "PCC >= 0.40, rank outside the top 5%, and superiority to every "
                "plotted baseline."
            ),
            "her2_selection_artifact_availability_gate": (
                "Latest-model top and large-margin candidates were restricted to "
                "the 45 genes with losslessly retained per-spot predictions for "
                "all 15 comparison methods. Section ranks and rank fractions were "
                "still computed over the complete 2,759-gene contract."
            ),
        }
    )
    provenance_path.write_text(
        json.dumps(provenance, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    readme = f"""# Lin-MNP spatial-expression overviews

This package applies the audited selection logic and exact 3 × 6 visual design
from `{LEGACY_ROOT}` to the latest fixed-contract Lin-MNP model.

- `cscc_benchmark_fixed997/`: cSCC Benchmark fixed-997 maps on the four
  non-redundant sections shared by all 16 plotted methods.
- `her2_hespotex_exact2759/`: HER2+ HESPotEx exact-2,759 maps on one
  deterministic representative section per patient/H&E group.
- `selected_genes.csv` and `selected_gene_sections.csv`: current LIN_MNP-based
  gene and section choices.
- `panel_pcc.csv`: PCC recomputed from the exact arrays shown in every panel.
- `spot_expression_detail.csv.gz`: plotted spot values and display-scaled values.
- `source_data/`: the LIN_MNP prediction indices and source provenance used by
  this rebuild.

Each panel contains the H&E image, ground truth, Lin-MNP and the same 15
comparison methods in the established order. Each map is independently
min-max scaled for display only; PCC is computed before display scaling.

The six comparison-method full matrices deleted during earlier cleanup were
losslessly reconstructed for the five representative HER2+ sections from the
audited v4 `spot_expression_detail.csv.gz`. Consequently, latest-model top and
large-margin selection is restricted to the 45 genes for which every plotted
method retains exact spot values. The section ranks themselves still use all
2,759 genes; literature and explicitly requested genes follow the established
v4 rules.

Rebuild with:

```bash
python -B \\
  {Path(__file__).resolve()} \\
  --output {output.resolve()} --dpi 220 --workers 4
```
"""
    (output / "README.md").write_text(readme, encoding="utf-8")


def write_manifest(output: Path) -> None:
    files = []
    for path in sorted(output.rglob("*")):
        if (
            path.is_file()
            and path.name != "result_manifest.json"
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
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "model": "Lin-MNP",
        "files": files,
    }
    (output / "result_manifest.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def main(argv: Sequence[str] | None = None) -> None:
    args = argument_parser().parse_args(argv)
    load_legacy_builders()
    if mp.get_start_method() != "fork":
        raise RuntimeError("the inherited audited renderer requires fork workers")
    output = args.output.resolve()
    configure(output)
    if args.prepare_only:
        cscc_long = build_cscc_metrics(latest_benchmark_prediction_index())
        cscc = legacy_build.complete_metric_table(
            cscc_long,
            dataset="cscc",
            method_column="model",
            methods=legacy_plot.CSCC_METHODS,
        )
        her2 = legacy_build.complete_metric_table(
            latest_her2_section_metrics(),
            dataset="her2",
            method_column="method",
            methods=legacy_plot.HER2_METHODS,
        )
        print(
            json.dumps(
                {
                    "status": "prepared",
                    "cscc_rows": len(cscc),
                    "her2_rows": len(her2),
                },
                indent=2,
            )
        )
        return
    legacy_build.run(output, dpi=args.dpi, workers=args.workers)
    update_metadata(output)
    write_manifest(output)
    print(json.dumps({"status": "complete", "output": str(output)}, indent=2))


if __name__ == "__main__":
    main()
