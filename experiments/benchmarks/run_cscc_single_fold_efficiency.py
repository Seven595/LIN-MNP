#!/usr/bin/env python3
"""Serial single-fold runtime benchmark for the 25 cSCC Benchmark methods."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time
from typing import Iterable


from lin_mnp.paths import ARCHIVE_ROOT, REPOSITORY_ROOT, RESULTS_ROOT

PROJECT = REPOSITORY_ROOT
ARCHIVE = ARCHIVE_ROOT
HERE = ARCHIVE / "tools"
ATTRIBUTION = ARCHIVE / "legacy/component_analysis/code"
PYTHON = Path(sys.executable)
DEFAULT_OUTPUT = RESULTS_ROOT / "runtime_benchmark"
QUALITY = ARCHIVE / "detail/00_plot_ready/main_protocol_summary_all_methods.csv"
COMMON_ORIGINAL = ATTRIBUTION / "run_benchmark_common_original_downstream.py"
COMMON_MISSING = ATTRIBUTION / "run_benchmark_common_missing_original_downstream.py"


DISPLAY = {
    "lin_mnp": "Lin-MNP",
    "spahgc": "SpaHGC", "stflow": "STFlow", "hespotex": "HESpotEx",
    "he2gene": "HE2Gene", "stag": "STAG", "egnv1": "EGNv1",
    "egnv2": "EGNv2", "asgnn": "asGNN", "holotea": "HoloTea",
    "istar": "iStar", "tcgn": "TCGN", "stnet": "ST-Net",
    "thitogene": "THItoGene", "histogene": "HisToGene",
    "gr2st": "GR2ST", "hist2st": "Hist2ST", "deepspace": "DeepSpaCE",
    "mclstexp": "mclSTExp", "bleep": "BLEEP", "igidl": "IGI-DL",
    "deeppt": "DeepPT", "heclip": "HECLIP", "genecoder": "GeneCodeR",
    "omiclip": "OmiCLIP",
}

# Light jobs first so the append-only table remains useful during a long campaign.
ORDER = (
    "lin_mnp", "genecoder", "igidl", "hespotex", "holotea",
    "omiclip", "bleep", "mclstexp", "asgnn", "egnv2", "spahgc",
    "deeppt", "gr2st", "deepspace", "heclip", "stag", "stflow",
    "histogene", "istar", "thitogene", "egnv1", "stnet", "he2gene",
    "hist2st", "tcgn",
)

CACHE_SCOPE = {
    "lin_mnp", "spahgc", "stflow", "hespotex", "stag", "egnv1",
    "egnv2", "asgnn", "holotea", "istar", "gr2st", "mclstexp", "bleep",
    "igidl", "deeppt", "omiclip",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def command(method: str, output: Path) -> list[str]:
    runs = output / "runs"
    common_original = {
        "gr2st", "bleep", "deeppt", "mclstexp", "igidl", "egnv1",
        "egnv2", "omiclip", "istar",
    }
    common_missing = {
        "he2gene", "heclip", "tcgn", "stnet", "thitogene", "histogene",
        "hist2st", "deepspace", "genecoder",
    }
    if method == "lin_mnp":
        return [
            str(PYTHON), "-m", "lin_mnp.model", "train",
            "--fold", "0", "--device", "cuda:0", "--output-root",
            str(runs / method),
        ]
    if method in common_original:
        return [
            str(PYTHON), str(COMMON_ORIGINAL), "--method", method,
            "--fold", "0", "--device", "cuda:0", "--output", str(runs),
        ]
    if method in common_missing:
        return [
            str(PYTHON), str(COMMON_MISSING), "--method", method,
            "--fold", "0", "--device", "cuda:0", "--output", str(runs),
        ]
    if method == "asgnn":
        return [
            str(PYTHON), str(PROJECT / "comparators/adapters/train_additional_comparators_fold.py"),
            "--method", "asgnn", "--fold", "0", "--device", "cuda:0",
            "--workers", "2", "--amp", "--output", str(runs),
        ]
    if method in {"holotea", "spahgc", "stag", "hespotex"}:
        return [
            str(PYTHON), str(HERE / "run_fixed997_raw_adapter.py"),
            "--method", method, "--fold", "0", "--device", "cuda:0",
            "--output", str(runs),
        ]
    if method == "stflow":
        return [
            str(PYTHON), str(ATTRIBUTION / "run_benchmark_common_stflow997.py"),
            "fold", "--encoder", "ciga", "--scale", "224", "--fold", "0",
            "--device", "cuda:0", "--epochs", "100", "--sample-times", "10",
            "--eval-step", "1", "--patience", "20", "--output",
            str(runs / method),
        ]
    raise KeyError(method)


def output_path(method: str, output: Path) -> Path:
    runs = output / "runs"
    if method == "lin_mnp":
        return runs / method / "fold0"
    if method == "hespotex":
        return runs / "hespotex_fixed997" / "fold0"
    if method == "stflow":
        return runs / method / "ciga_224px" / "fold0"
    return runs / method / "fold0"


def gpu_memory(gpu: int) -> float:
    result = subprocess.run(
        [
            "nvidia-smi", f"--id={gpu}", "--query-gpu=memory.used",
            "--format=csv,noheader,nounits",
        ],
        check=True, capture_output=True, text=True,
    )
    return float(result.stdout.strip().splitlines()[0])


def parse_peak_rss(path: Path) -> int | None:
    if not path.is_file():
        return None
    match = re.search(
        r"Maximum resident set size \(kbytes\):\s*(\d+)",
        path.read_text(encoding="utf-8", errors="replace"),
    )
    return None if match is None else int(match.group(1))


def metadata_wall(root: Path) -> float | None:
    candidates = (root / "run_metadata.json", root / "evaluation.json")
    for path in candidates:
        if not path.is_file():
            continue
        record = json.loads(path.read_text(encoding="utf-8"))
        for key in ("wall_seconds", "wall_time_seconds", "total_wall_seconds"):
            if key in record:
                return float(record[key])
    return None


def completed(root: Path) -> bool:
    return (root / "complete.json").is_file() or (root / "evaluation.json").is_file()


def load_quality() -> dict[str, float]:
    result = {}
    with QUALITY.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row["protocol"] == "benchmark_strict_patient_outer_cv4":
                result[row["method_id"]] = float(row["pcc"])
    return result


def write_csv(rows: Iterable[dict[str, object]], path: Path) -> None:
    rows = list(rows)
    if not rows:
        return
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def write_summary(rows: list[dict[str, object]], output: Path) -> None:
    completed_rows = [row for row in rows if row["status"] == "complete"]
    lin_mnp_wall = next(
        (float(row["task_wall_seconds"]) for row in completed_rows
         if row["method_id"] == "lin_mnp"),
        None,
    )
    lines = [
        "# cSCC fixed-997 single-fold computational efficiency",
        "",
        "## Contract",
        "",
        "- Dataset: GSE144240 cSCC, ordered 997-gene panel.",
        "- Split: strict patient-CV4 outer fold 0; the same train, validation and test patients for all methods.",
        "- Target: `log1p(raw count)`; validation-only selection where the method has a selection stage.",
        "- Hardware: one isolated physical GPU, methods executed serially.",
        "- Runtime: process launch through fitted-model test prediction. Frozen target-free feature caches are treated as reusable inputs; methods that train an image encoder end-to-end retain that cost.",
        "- Replication: one fold and one run only. Times are point measurements, not uncertainty estimates.",
        "",
        "## Results",
        "",
        "| Method | Strict CV4 PCC¹ | Fold-0 wall time (s) | Relative to Lin-MNP | Peak GPU increment (MiB) | Peak CPU RSS (MiB) | Timing input |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for row in sorted(completed_rows, key=lambda value: float(value["task_wall_seconds"])):
        wall = float(row["task_wall_seconds"])
        ratio = "—" if lin_mnp_wall is None else f"{wall / lin_mnp_wall:.1f}×"
        gpu = row["gpu_peak_increment_mib"]
        rss = row["cpu_peak_rss_kib"]
        lines.append(
            f"| {row['method']} | {float(row['strict_cv4_pcc']):.6f} | {wall:.3f} | "
            f"{ratio} | {float(gpu):.0f} | {float(rss) / 1024:.0f} | "
            f"{row['timing_input_scope']} |"
        )
    lines += [
        "",
        "¹ Strict CV4 PCC is the already locked four-fold quality result and is shown only to form the accuracy-cost view; runtime itself is measured on fold 0.",
        "",
        "## Interpretation boundary",
        "",
        "This experiment supports comparison of fold-specific fitting/selection/inference cost under the local reproduction. It does not yet compare cold image-feature extraction for methods that use a frozen cache, and therefore must not be described as a full raw-WSI-to-expression wall-clock benchmark.",
        "",
    ]
    failed = [row for row in rows if row["status"] != "complete"]
    if failed:
        lines += ["## Incomplete runs", ""]
        for row in failed:
            lines.append(f"- {row['method']}: {row['status']} (exit {row['return_code']})")
        lines.append("")
    (output / "RESULTS.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu", type=int, default=4)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--methods", default=",".join(ORDER))
    parser.add_argument("--retry-failed", action="store_true")
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    (output / "logs").mkdir(exist_ok=True)
    methods = tuple(value.strip() for value in args.methods.split(",") if value.strip())
    unknown = sorted(set(methods) - set(DISPLAY))
    if unknown:
        raise ValueError(f"unknown methods: {unknown}")

    quality = load_quality()
    environment = os.environ.copy()
    environment.update({
        "CUDA_VISIBLE_DEVICES": str(args.gpu),
        "OMP_NUM_THREADS": "8", "MKL_NUM_THREADS": "8",
        "OPENBLAS_NUM_THREADS": "8", "NUMEXPR_NUM_THREADS": "8",
        "TOKENIZERS_PARALLELISM": "false", "HF_HUB_OFFLINE": "1",
        "LIN_MNP_CHECKPOINT_SELECTION_SCOPE": "inner_validation",
    })
    gpu_info = subprocess.run(
        [
            "nvidia-smi", f"--id={args.gpu}",
            "--query-gpu=index,name,driver_version,memory.total",
            "--format=csv,noheader",
        ], check=True, capture_output=True, text=True,
    ).stdout.strip()
    manifest = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": "GSE144240 cSCC",
        "gene_panel": "ordered fixed-997",
        "target": "log1p(raw count)",
        "fold": 0,
        "execution": "serial single physical GPU",
        "gpu": gpu_info,
        "feature_cache_policy": (
            "reusable target-free frozen image descriptors excluded from primary runtime; "
            "end-to-end trainable encoders remain inside their method runtime"
        ),
        "methods": list(methods),
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )

    records_path = output / "runtime_records.json"
    if records_path.is_file():
        records = json.loads(records_path.read_text(encoding="utf-8"))
    else:
        records = []
    existing = {row["method_id"]: row for row in records}

    for index, method in enumerate(methods, 1):
        destination = output_path(method, output)
        if method in existing and existing[method]["status"] == "complete":
            print(json.dumps({"event": "skip_complete", "method": method}), flush=True)
            continue
        if method in existing and not args.retry_failed:
            print(json.dumps({"event": "skip_failed", "method": method}), flush=True)
            continue
        if destination.exists() and not completed(destination):
            raise RuntimeError(f"incomplete output must be audited before retry: {destination}")

        cmd = command(method, output)
        log_path = output / "logs" / f"{index:02d}_{method}.log"
        time_path = output / "logs" / f"{index:02d}_{method}.time.txt"
        baseline = gpu_memory(args.gpu)
        peak = baseline
        started = time.perf_counter()
        print(json.dumps({
            "event": "start", "index": index, "total": len(methods),
            "method": method, "gpu": args.gpu,
        }), flush=True)
        with log_path.open("w", encoding="utf-8") as log:
            time_binary = shutil.which("time")
            if time_binary is None:
                raise RuntimeError("GNU time is required for efficiency measurements")
            process = subprocess.Popen(
                [time_binary, "-v", "-o", str(time_path), *cmd],
                cwd=PROJECT, env=environment, stdout=log,
                stderr=subprocess.STDOUT, text=True,
            )
            while process.poll() is None:
                try:
                    peak = max(peak, gpu_memory(args.gpu))
                except Exception:
                    pass
                time.sleep(0.2)
        elapsed = time.perf_counter() - started
        status = "complete" if process.returncode == 0 and completed(destination) else "failed"
        row = {
            "method_id": method,
            "method": DISPLAY[method],
            "status": status,
            "return_code": int(process.returncode),
            "strict_cv4_pcc": quality[method],
            "task_wall_seconds": elapsed,
            "internal_wall_seconds": metadata_wall(destination),
            "gpu_baseline_mib": baseline,
            "gpu_peak_total_mib": peak,
            "gpu_peak_increment_mib": max(0.0, peak - baseline),
            "cpu_peak_rss_kib": parse_peak_rss(time_path),
            "timing_input_scope": (
                "cached frozen features" if method in CACHE_SCOPE
                else "patch/end-to-end or deterministic input"
            ),
            "output_path": str(destination),
            "log_path": str(log_path),
            "command": json.dumps(cmd),
        }
        existing[method] = row
        records = [existing[key] for key in methods if key in existing]
        records_path.write_text(
            json.dumps(records, indent=2) + "\n", encoding="utf-8"
        )
        write_csv(records, output / "runtime_summary.csv")
        write_summary(records, output)
        print(json.dumps({
            "event": status, "method": method, "wall_seconds": elapsed,
            "gpu_peak_increment_mib": row["gpu_peak_increment_mib"],
            "completed": sum(value["status"] == "complete" for value in records),
        }), flush=True)
        if status != "complete":
            print(json.dumps({"event": "failure_log", "path": str(log_path)}), flush=True)

    final_records = [existing[key] for key in methods if key in existing]
    write_csv(final_records, output / "runtime_summary.csv")
    write_summary(final_records, output)
    manifest["completed_at_utc"] = datetime.now(timezone.utc).isoformat()
    manifest["complete_methods"] = [
        row["method_id"] for row in final_records if row["status"] == "complete"
    ]
    manifest["failed_methods"] = [
        row["method_id"] for row in final_records if row["status"] != "complete"
    ]
    manifest["runtime_summary_sha256"] = sha256(output / "runtime_summary.csv")
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({
        "event": "campaign_complete", "completed": len(manifest["complete_methods"]),
        "failed": manifest["failed_methods"], "output": str(output),
    }), flush=True)


if __name__ == "__main__":
    main()
