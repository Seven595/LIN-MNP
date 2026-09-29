#!/usr/bin/env python3
"""Summarize encoder-swap and downstream-swap interventions without mixing protocols."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from lin_mnp.paths import ARCHIVE_ROOT, RESULTS_ROOT

RESULT = RESULTS_ROOT / "component_analysis"
STORAGE = ARCHIVE_ROOT / "component_analysis"
ORIGINAL = RESULTS_ROOT / "comparator_reruns"
NEW_BENCH = STORAGE / "benchmark_ctranspath192_original_downstream"
HEAD_SUMMARY = ARCHIVE_ROOT / "component_analysis/matched_downstream_swap_summary.csv"
HES_NEW = STORAGE / "hespotex_ctranspath192_original_downstream"
HES_ORIGINAL = {
    "cscc": RESULTS_ROOT / "hespotex/cscc_official/report",
    "her2": RESULTS_ROOT / "hespotex/her2_official/report",
}
REG = STORAGE / "reg2st_encoder_pair"
CTP_RIDGE = STORAGE / "ctranspath192_ridge_baseline"
REG_RIDGE = ARCHIVE_ROOT / "component_analysis/reg2st_phikon_direct_ridge"


METHODS = ("GR2ST", "asGNN", "BLEEP", "DeepPT", "mclSTExp", "IGI-DL", "EGNv1", "EGNv2", "OmiCLIP", "iStar")
ORIGINAL_ID = {
    "GR2ST": "gr2st", "asGNN": "asgnn", "BLEEP": "bleep",
    "DeepPT": "deeppt", "mclSTExp": "mclstexp", "IGI-DL": "igidl",
    "EGNv1": "egnv1", "EGNv2": "egnv2", "OmiCLIP": "omiclip", "iStar": "istar",
}
OLD_SWAP = {
    "GR2ST": ARCHIVE_ROOT / "legacy/component_analysis/gr2st/evaluation",
    "asGNN": ARCHIVE_ROOT / "legacy/component_analysis/asgnn/evaluation",
    "BLEEP": ARCHIVE_ROOT / "legacy/component_analysis/bleep/evaluation",
}
BENCHMARK_RIDGE_VARIANT = {
    "GR2ST": "log1p_cp10k",
    "asGNN": "log1p_raw",
    "BLEEP": "log1p_cp10k",
    "DeepPT": "log1p_cp10k",
    "mclSTExp": "log1p_cp10k",
    "IGI-DL": "igidl_log_pseudocount_cpm",
    "EGNv1": "egn_minmax_log10_raw",
    "EGNv2": "egn_minmax_log10_raw",
    "OmiCLIP": "log1p_cp10k",
    "iStar": "istar_train_minmax_raw_counts",
}
ORIGINAL_EVAL = {
    "GR2ST": RESULTS_ROOT / "additional_comparator_reruns/gr2st/evaluation",
    "asGNN": RESULTS_ROOT / "additional_comparator_reruns/asgnn/evaluation",
}


def evaluation_pcc(path: Path) -> float:
    value = json.loads((path / "summary.json").read_text())
    return float(value["strict_all_four_outer_folds"]["pcc"]["mean"])


def pcc_detail(path: Path) -> pd.DataFrame:
    table = pd.read_csv(path / "detail/per_patient_gene_metrics.csv")
    return table.loc[:, ["patient_id", "gene", "pcc"]]


def patient_bootstrap_delta(
    original: pd.DataFrame, replacement: pd.DataFrame, samples: int, seed: int
) -> tuple[float, float, float]:
    merged = original.merge(replacement, on=["patient_id", "gene"], suffixes=("_original", "_replacement"), validate="one_to_one")
    merged["delta"] = merged.pcc_replacement - merged.pcc_original
    per_patient = merged.groupby("patient_id", sort=True).delta.mean().to_numpy()
    point = float(np.nanmean(per_patient))
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(per_patient), size=(samples, len(per_patient)))
    values = np.nanmean(per_patient[indices], axis=1)
    low, high = np.nanpercentile(values, [2.5, 97.5])
    return point, float(low), float(high)


def benchmark(samples: int, seed: int) -> pd.DataFrame:
    head = pd.read_csv(HEAD_SUMMARY)
    head = head.loc[head.protocol == "Benchmark strict patient CV4"].set_index("method")
    rows = []
    for method in METHODS:
        original_eval = ORIGINAL_EVAL.get(method, ORIGINAL / ORIGINAL_ID[method] / "evaluation")
        swap_eval = OLD_SWAP.get(method, NEW_BENCH / ORIGINAL_ID[method] / "evaluation")
        if not (swap_eval / "summary.json").is_file():
            raise FileNotFoundError(f"missing completed encoder swap: {swap_eval}")
        original_pcc = evaluation_pcc(original_eval)
        encoder_pcc = evaluation_pcc(swap_eval)
        ctp_ridge_eval = (
            CTP_RIDGE / "benchmark" / BENCHMARK_RIDGE_VARIANT[method] / "evaluation"
        )
        ctp_ridge_pcc = evaluation_pcc(ctp_ridge_eval)
        delta, low, high = patient_bootstrap_delta(
            pcc_detail(original_eval), pcc_detail(swap_eval), samples, seed
        )
        ctp_downstream_delta, ctp_downstream_low, ctp_downstream_high = (
            patient_bootstrap_delta(
                pcc_detail(swap_eval), pcc_detail(ctp_ridge_eval), samples, seed
            )
        )
        row = head.loc[method]
        rows.append({
            "method": method, "protocol": "Benchmark strict patient CV4",
            "original_encoder_original_downstream_pcc": original_pcc,
            "ctranspath192_original_downstream_pcc": encoder_pcc,
            "encoder_swap_delta": delta, "encoder_swap_delta_ci_low": low,
            "encoder_swap_delta_ci_high": high,
            "original_encoder_ridge_pcc": float(row.original_encoder_ridge_pcc),
            "downstream_swap_delta": float(row.ridge_minus_original_pcc),
            "downstream_swap_delta_ci_low": float(row.delta_ci_low),
            "downstream_swap_delta_ci_high": float(row.delta_ci_high),
            "ctranspath192_ridge_pcc": ctp_ridge_pcc,
            "ctranspath_downstream_swap_delta": ctp_downstream_delta,
            "ctranspath_downstream_swap_delta_ci_low": ctp_downstream_low,
            "ctranspath_downstream_swap_delta_ci_high": ctp_downstream_high,
            "factorial_interaction_point": (
                ctp_downstream_delta - float(row.ridge_minus_original_pcc)
            ),
            "encoder_swap_complete": True,
            "encoder_swap_note": "CTransPath-192; original downstream and target transform retained",
        })
    return pd.DataFrame(rows)


def section_bootstrap_delta(original: pd.DataFrame, replacement: pd.DataFrame, samples: int, seed: int):
    keys = ["slide_id", "patient_id", "gene"]
    merged = original[keys + ["pcc"]].merge(replacement[keys + ["pcc"]], on=keys, suffixes=("_original", "_replacement"), validate="one_to_one")
    merged["delta"] = merged.pcc_replacement - merged.pcc_original
    per_section = merged.groupby("slide_id", sort=True).delta.mean().to_numpy()
    point = float(per_section.mean())
    rng = np.random.default_rng(seed)
    index = rng.integers(0, len(per_section), size=(samples, len(per_section)))
    boot = per_section[index].mean(1)
    low, high = np.percentile(boot, [2.5, 97.5])
    return point, float(low), float(high)


def hespotex(samples: int, seed: int) -> pd.DataFrame:
    head = pd.read_csv(HEAD_SUMMARY)
    rows = []
    for cohort, protocol in (("cscc", "HESpotEx cSCC section-LOSO"), ("her2", "HESpotEx HER2+ section-LOSO")):
        matched = head.loc[head.protocol == protocol].iloc[0]
        original = pd.read_csv(HES_ORIGINAL[cohort] / "per_slide_gene.csv")
        replacement = pd.read_csv(HES_NEW / cohort / "report/per_slide_gene.csv")
        ctp_ridge = pd.read_csv(
            CTP_RIDGE / "hespotex" / cohort / "report/per_slide_gene.csv"
        )
        delta, low, high = section_bootstrap_delta(original, replacement, samples, seed)
        ctp_downstream_delta, ctp_downstream_low, ctp_downstream_high = (
            section_bootstrap_delta(replacement, ctp_ridge, samples, seed)
        )
        summary = json.loads((HES_NEW / cohort / "report/summary.json").read_text())
        ctp_ridge_summary = json.loads(
            (CTP_RIDGE / "hespotex" / cohort / "report/summary.json").read_text()
        )
        rows.append({
            "cohort": cohort, "protocol": protocol,
            "quiltnet_original_downstream_pcc": float(matched.original_model_pcc),
            "ctranspath192_original_downstream_pcc": float(summary["mean_pcc"]),
            "encoder_swap_delta": float(summary["mean_pcc"]) - float(matched.original_model_pcc),
            "encoder_swap_section_bootstrap_delta": delta,
            "encoder_swap_delta_ci_low": low, "encoder_swap_delta_ci_high": high,
            "quiltnet_ridge_pcc": float(matched.original_encoder_ridge_pcc),
            "downstream_swap_delta": float(matched.ridge_minus_original_pcc),
            "downstream_swap_delta_ci_low": float(matched.delta_ci_low),
            "downstream_swap_delta_ci_high": float(matched.delta_ci_high),
            "ctranspath192_ridge_pcc": float(ctp_ridge_summary["mean_pcc"]),
            "ctranspath_downstream_swap_delta": ctp_downstream_delta,
            "ctranspath_downstream_swap_delta_ci_low": ctp_downstream_low,
            "ctranspath_downstream_swap_delta_ci_high": ctp_downstream_high,
            "factorial_interaction_point": (
                ctp_downstream_delta - float(matched.ridge_minus_original_pcc)
            ),
            "selection": "outer-test-selected to match released HESpotEx",
        })
    return pd.DataFrame(rows)


def reg2st(samples: int, seed: int) -> pd.DataFrame:
    summaries = {}
    details = {}
    for encoder in ("phikon", "ctranspath"):
        path = REG / encoder / "report"
        summaries[encoder] = json.loads((path / "summary.json").read_text())
        details[encoder] = pd.read_csv(path / "per_patient_gene_pcc.csv")
        details[encoder] = details[encoder].rename(columns={"patient": "patient_id"})
    delta, low, high = patient_bootstrap_delta(
        details["phikon"], details["ctranspath"], samples, seed
    )
    ridge_section = pd.read_csv(REG_RIDGE / "per_section_gene_pcc.csv.gz")
    ridge_patient = (
        ridge_section.groupby(["patient", "gene"], as_index=False).pcc.mean()
        .rename(columns={"patient": "patient_id"})
    )
    downstream_delta, downstream_low, downstream_high = patient_bootstrap_delta(
        details["phikon"], ridge_patient, samples, seed
    )
    ctp_ridge_summary = json.loads(
        (CTP_RIDGE / "reg2st/report/summary.json").read_text()
    )
    ctp_ridge_patient = pd.read_csv(
        CTP_RIDGE / "reg2st/report/per_patient_gene_pcc.csv"
    ).rename(columns={"patient": "patient_id"})
    ctp_downstream_delta, ctp_downstream_low, ctp_downstream_high = (
        patient_bootstrap_delta(
            details["ctranspath"], ctp_ridge_patient, samples, seed
        )
    )
    published = pd.read_csv(HEAD_SUMMARY)
    published = published.loc[published.method == "Reg2ST"].iloc[0]
    return pd.DataFrame([{
        "protocol": "local matched Reg2ST cSCC section-LOSO",
        "phikon_original_downstream_pcc": summaries["phikon"]["strict_equal_patient_mean_gene_pcc"],
        "ctranspath192_original_downstream_pcc": summaries["ctranspath"]["strict_equal_patient_mean_gene_pcc"],
        "encoder_swap_delta": delta, "encoder_swap_delta_ci_low": low,
        "encoder_swap_delta_ci_high": high,
        "published_reg2st_aggregate_pcc": float(published.original_model_pcc),
        "phikon_ridge_pcc": float(published.original_encoder_ridge_pcc),
        "local_downstream_swap_delta": downstream_delta,
        "local_downstream_swap_delta_ci_low": downstream_low,
        "local_downstream_swap_delta_ci_high": downstream_high,
        "ctranspath192_ridge_pcc": ctp_ridge_summary[
            "strict_equal_patient_mean_gene_pcc"
        ],
        "ctranspath_downstream_swap_delta": ctp_downstream_delta,
        "ctranspath_downstream_swap_delta_ci_low": ctp_downstream_low,
        "ctranspath_downstream_swap_delta_ci_high": ctp_downstream_high,
        "factorial_interaction_point": ctp_downstream_delta - downstream_delta,
        "note": "local Phikon/CTransPath crossed with original Reg2ST/Ridge comparisons are paired on 12 sections; publisher aggregate remains descriptive because its array has 14 rather than 12 rows",
    }])


def all_sixteen(benchmark_table: pd.DataFrame, hespotex_table: pd.DataFrame, reg2st_table: pd.DataFrame) -> pd.DataFrame:
    """Put all 16 matched conditions in one schema without hiding protocol differences."""
    head = pd.read_csv(HEAD_SUMMARY)
    rows = []
    for record in benchmark_table.to_dict("records"):
        matched = head.loc[
            (head.protocol == "Benchmark strict patient CV4") & (head.method == record["method"])
        ].iloc[0]
        rows.append({
            "condition": record["method"], "protocol": record["protocol"],
            "original_encoder": matched.encoder,
            "original_downstream": matched.original_downstream,
            "e0_d0_pcc": record["original_encoder_original_downstream_pcc"],
            "replacement_encoder": "CTransPath, 192 px",
            "e1_d0_pcc": record["ctranspath192_original_downstream_pcc"],
            "encoder_swap_delta": record["encoder_swap_delta"],
            "encoder_swap_ci_low": record["encoder_swap_delta_ci_low"],
            "encoder_swap_ci_high": record["encoder_swap_delta_ci_high"],
            "e0_ridge_pcc": record["original_encoder_ridge_pcc"],
            "downstream_swap_delta": record["downstream_swap_delta"],
            "downstream_swap_ci_low": record["downstream_swap_delta_ci_low"],
            "downstream_swap_ci_high": record["downstream_swap_delta_ci_high"],
            "e1_ridge_pcc": record["ctranspath192_ridge_pcc"],
            "ctranspath_downstream_swap_delta": record[
                "ctranspath_downstream_swap_delta"
            ],
            "ctranspath_downstream_swap_ci_low": record[
                "ctranspath_downstream_swap_delta_ci_low"
            ],
            "ctranspath_downstream_swap_ci_high": record[
                "ctranspath_downstream_swap_delta_ci_high"
            ],
            "factorial_interaction_point": record["factorial_interaction_point"],
            "status": "complete paired 2x2 factorial intervention",
        })
    for record in hespotex_table.to_dict("records"):
        matched = head.loc[head.protocol == record["protocol"]].iloc[0]
        rows.append({
            "condition": f"HESpotEx {record['cohort']}", "protocol": record["protocol"],
            "original_encoder": matched.encoder,
            "original_downstream": matched.original_downstream,
            "e0_d0_pcc": record["quiltnet_original_downstream_pcc"],
            "replacement_encoder": "CTransPath, 192 px",
            "e1_d0_pcc": record["ctranspath192_original_downstream_pcc"],
            "encoder_swap_delta": record["encoder_swap_section_bootstrap_delta"],
            "encoder_swap_ci_low": record["encoder_swap_delta_ci_low"],
            "encoder_swap_ci_high": record["encoder_swap_delta_ci_high"],
            "e0_ridge_pcc": record["quiltnet_ridge_pcc"],
            "downstream_swap_delta": record["downstream_swap_delta"],
            "downstream_swap_ci_low": record["downstream_swap_delta_ci_low"],
            "downstream_swap_ci_high": record["downstream_swap_delta_ci_high"],
            "e1_ridge_pcc": record["ctranspath192_ridge_pcc"],
            "ctranspath_downstream_swap_delta": record[
                "ctranspath_downstream_swap_delta"
            ],
            "ctranspath_downstream_swap_ci_low": record[
                "ctranspath_downstream_swap_delta_ci_low"
            ],
            "ctranspath_downstream_swap_ci_high": record[
                "ctranspath_downstream_swap_delta_ci_high"
            ],
            "factorial_interaction_point": record["factorial_interaction_point"],
            "status": "complete paired 2x2 factorial intervention; outer-test-selected HESpotEx alignment",
        })
    stflow = head.loc[head.protocol == "STFlow-native 50-gene strict patient CV4"].copy()
    ctranspath_pcc = float(stflow.loc[stflow.method == "STFlow + CTransPath", "original_model_pcc"].iloc[0])
    ctranspath_ridge = float(stflow.loc[stflow.method == "STFlow + CTransPath", "original_encoder_ridge_pcc"].iloc[0])
    ctranspath_downstream_delta = ctranspath_ridge - ctranspath_pcc
    ctp_row = stflow.loc[stflow.method == "STFlow + CTransPath"].iloc[0]
    for record in stflow.to_dict("records"):
        is_reference = record["method"] == "STFlow + CTransPath"
        rows.append({
            "condition": record["method"], "protocol": record["protocol"],
            "original_encoder": record["encoder"],
            "original_downstream": record["original_downstream"],
            "e0_d0_pcc": record["original_model_pcc"],
            "replacement_encoder": np.nan if is_reference else "CTransPath, 192 px",
            "e1_d0_pcc": np.nan if is_reference else ctranspath_pcc,
            "encoder_swap_delta": np.nan if is_reference else ctranspath_pcc - float(record["original_model_pcc"]),
            "encoder_swap_ci_low": np.nan, "encoder_swap_ci_high": np.nan,
            "e0_ridge_pcc": record["original_encoder_ridge_pcc"],
            "downstream_swap_delta": record["ridge_minus_original_pcc"],
            "downstream_swap_ci_low": record["delta_ci_low"],
            "downstream_swap_ci_high": record["delta_ci_high"],
            "e1_ridge_pcc": ctranspath_ridge,
            "ctranspath_downstream_swap_delta": ctranspath_downstream_delta,
            "ctranspath_downstream_swap_ci_low": ctp_row["delta_ci_low"],
            "ctranspath_downstream_swap_ci_high": ctp_row["delta_ci_high"],
            "factorial_interaction_point": (
                ctranspath_downstream_delta - float(record["ridge_minus_original_pcc"])
            ),
            "status": "local multi-PFM control; official UNI/GigaPath unavailable",
        })
    record = reg2st_table.iloc[0]
    rows.append({
        "condition": "Reg2ST cSCC", "protocol": record.protocol,
        "original_encoder": "Phikon-v2 released 224-px features",
        "original_downstream": "projection + expression alignment + spatial decoder + WiKG + ZINB",
        "e0_d0_pcc": record.phikon_original_downstream_pcc,
        "replacement_encoder": "CTransPath, 192 px",
        "e1_d0_pcc": record.ctranspath192_original_downstream_pcc,
        "encoder_swap_delta": record.encoder_swap_delta,
        "encoder_swap_ci_low": record.encoder_swap_delta_ci_low,
        "encoder_swap_ci_high": record.encoder_swap_delta_ci_high,
        "e0_ridge_pcc": record.phikon_ridge_pcc,
        "downstream_swap_delta": record.local_downstream_swap_delta,
        "downstream_swap_ci_low": record.local_downstream_swap_delta_ci_low,
        "downstream_swap_ci_high": record.local_downstream_swap_delta_ci_high,
        "e1_ridge_pcc": record.ctranspath192_ridge_pcc,
        "ctranspath_downstream_swap_delta": record.ctranspath_downstream_swap_delta,
        "ctranspath_downstream_swap_ci_low": record.ctranspath_downstream_swap_delta_ci_low,
        "ctranspath_downstream_swap_ci_high": record.ctranspath_downstream_swap_delta_ci_high,
        "factorial_interaction_point": record.factorial_interaction_point,
        "status": "complete local 12-section paired 2x2 factorial intervention; released 14-row aggregate retained separately",
    })
    table = pd.DataFrame(rows)
    if len(table) != 16:
        raise ValueError(f"expected 16 unified conditions, found {len(table)}")
    return table


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bootstrap-samples", type=int, default=20_000)
    parser.add_argument("--bootstrap-seed", type=int, default=17)
    args = parser.parse_args()
    RESULT.mkdir(parents=True, exist_ok=True)
    benchmark_table = benchmark(args.bootstrap_samples, args.bootstrap_seed)
    hespotex_table = hespotex(args.bootstrap_samples, args.bootstrap_seed)
    reg2st_table = reg2st(args.bootstrap_samples, args.bootstrap_seed)
    tables = {
        "complete_two_way_benchmark.csv": benchmark_table,
        "complete_two_way_hespotex.csv": hespotex_table,
        "complete_two_way_reg2st.csv": reg2st_table,
        "complete_two_way_all_16.csv": all_sixteen(benchmark_table, hespotex_table, reg2st_table),
    }
    tables["complete_factorial_all_16.csv"] = tables[
        "complete_two_way_all_16.csv"
    ]
    for name, table in tables.items():
        table.to_csv(RESULT / name, index=False)
        print(RESULT / name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
