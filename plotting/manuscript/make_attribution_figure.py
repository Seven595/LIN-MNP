#!/usr/bin/env python3
"""Build the matched-descriptor attribution figure for the manuscript.

The source table's historical ``common_raw`` column names denote the common
log1p raw-count PCC reference, not identical training-target normalization.
The primary CTransPath-384 readout fits section-standardized targets, while
native modules and readouts paired with method-specific representations fit
unstandardized log1p raw counts.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from lin_mnp.paths import ARCHIVE_ROOT, RESULTS_ROOT

SOURCE = (
    ARCHIVE_ROOT
    / "reports"
    / "comparator_results"
    / "benchmark_common_contract_attribution.csv"
)
OUTPUT = RESULTS_ROOT / "figures/figure2.png"

SENSITIVITY = {"OmiCLIP", "HECLIP", "iStar"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    frame = pd.read_csv(SOURCE)
    # The manuscript panel presents only the prespecified primary
    # feature-substitution analyses. OmiCLIP/HECLIP require a joint
    # image--text interface and iStar has reduced section coverage; those
    # sensitivity analyses remain documented in the Supplementary tables.
    frame = frame.loc[~frame["method"].isin(SENSITIVITY)].copy()
    frame["analysis"] = "Primary"

    colours = {"Primary": "#2468B4", "Interface/coverage sensitivity": "#8A8F98"}
    markers = {"Primary": "o", "Interface/coverage sensitivity": "D"}

    # The main text uses a single-column figure with two compact, approximately
    # square plotting areas arranged side by side.
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(8.8, 5.2),
        gridspec_kw={"width_ratios": [1.0, 1.0]},
    )

    left = axes[0]
    for analysis, group in frame.groupby("analysis", sort=False):
        left.scatter(
            group["original_encoder_ridge_common_raw_pcc"],
            group["original_encoder_original_downstream_common_raw_pcc"],
            s=95,
            c=colours[analysis],
            marker=markers[analysis],
            edgecolor="white",
            linewidth=1.1,
            label=(
                "Sensitivity analyses"
                if analysis == "Interface/coverage sensitivity"
                else analysis
            ),
            zorder=3,
        )
    lower = min(
        frame["original_encoder_ridge_common_raw_pcc"].min(),
        frame["original_encoder_original_downstream_common_raw_pcc"].min(),
    ) - 0.02
    upper = max(
        frame["original_encoder_ridge_common_raw_pcc"].max(),
        frame["original_encoder_original_downstream_common_raw_pcc"].max(),
    ) + 0.02
    left.plot(
        [lower, upper],
        [lower, upper],
        color="#4B5563",
        linewidth=2.0,
        linestyle="--",
    )
    left.set_xlim(lower, upper)
    left.set_ylim(lower, upper)
    left.set_xlabel("Ridge PCC on native\nrepresentation", fontsize=12.5, labelpad=7)
    left.set_ylabel("Downstream-head PCC", fontsize=13, labelpad=8)

    right = axes[1]
    ordered = frame.sort_values("ctranspath384_original_downstream_common_raw_pcc")
    positions = range(len(ordered))
    for position, (_, row) in zip(positions, ordered.iterrows()):
        analysis = row["analysis"]
        right.scatter(
            row["ctranspath384_original_downstream_common_raw_pcc"],
            position,
            s=95,
            c=colours[analysis],
            marker=markers[analysis],
            edgecolor="white",
            linewidth=1.1,
            zorder=3,
        )
    ridge = float(frame["ctranspath384_ridge_common_raw_pcc"].iloc[0])
    right.axvline(ridge, color="#B42318", linewidth=2.6, linestyle="--", label=f"Ridge ({ridge:.3f})")
    right.set_yticks(list(positions), ordered["method"])
    # Only the 13 primary methods are plotted; keep the y-axis tight to their
    # ranked rows instead of retaining the former 16-method padding.
    right.set_ylim(-0.6, len(ordered) - 0.4)
    right.set_xlabel("PCC on common\nCTransPath-384 features", fontsize=12.5, labelpad=7)
    right.legend(
        frameon=False,
        fontsize=11.5,
        loc="lower left",
        bbox_to_anchor=(0.0, 1.0),
        borderaxespad=0,
        handletextpad=0.5,
    )

    for axis in axes:
        axis.grid(axis="both", color="#E5E7EB", linewidth=1.2, zorder=0)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
        axis.tick_params(labelsize=10.5)
        axis.set_box_aspect(1)

    fig.tight_layout(w_pad=1.8)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=300, bbox_inches="tight")
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
