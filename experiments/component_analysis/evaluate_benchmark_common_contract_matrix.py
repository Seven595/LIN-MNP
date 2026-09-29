#!/usr/bin/env python3
"""Evaluate every completed common-target downstream run."""

from __future__ import annotations

from collections import deque
import json
from pathlib import Path
import subprocess
import sys
import time

from lin_mnp.paths import ARCHIVE_ROOT, REPOSITORY_ROOT, RESULTS_ROOT

PROJECT = REPOSITORY_ROOT
PYTHON = Path(sys.executable)
EVALUATOR = ARCHIVE_ROOT / "tools/evaluate_benchmark_external_rerun.py"
ROOT = RESULTS_ROOT / "component_analysis/common_contract"
SETS = {
    "original_encoder_original_downstream_common_raw": (
        "gr2st", "bleep", "deeppt", "mclstexp", "igidl",
        "egnv1", "egnv2", "omiclip", "istar", "he2gene", "heclip",
        "tcgn", "stnet", "thitogene", "histogene", "hist2st",
        "deepspace", "genecoder",
    ),
    "ctranspath384_original_downstream_common_raw": (
        "gr2st", "asgnn", "bleep", "deeppt", "mclstexp", "igidl",
        "egnv1", "egnv2", "omiclip", "istar", "he2gene", "heclip",
        "tcgn", "stnet", "deepspace",
    ),
    "missing_original_encoder_affine_ridge_common_raw": (
        "he2gene", "heclip", "tcgn", "stnet", "deepspace",
    ),
}


def main() -> None:
    pending = deque()
    for group, methods in SETS.items():
        group_root = ROOT / group
        for method in methods:
            if (group_root / method / "evaluation/summary.json").is_file():
                continue
            pending.append((group, method))
    print(json.dumps({"event": "evaluation_start", "pending": len(pending)}), flush=True)
    active = {}
    failures = []
    slots = range(4)
    while pending or active:
        for slot in slots:
            if slot in active or not pending:
                continue
            group, method = pending.popleft()
            command = [
                str(PYTHON), str(EVALUATOR), "--root", str(ROOT / group),
                "--method", method, "--bootstrap-samples", "2000",
                "--bootstrap-seed", "17",
            ]
            process = subprocess.Popen(command, cwd=str(PROJECT))
            active[slot] = (process, group, method)
            print(json.dumps({"event": "evaluation_started", "group": group, "method": method, "pid": process.pid}), flush=True)
        time.sleep(3)
        for slot, (process, group, method) in list(active.items()):
            code = process.poll()
            if code is None:
                continue
            if code:
                failures.append({"group": group, "method": method, "returncode": code})
                print(json.dumps({"event": "evaluation_failed", "group": group, "method": method, "returncode": code}), flush=True)
            else:
                print(json.dumps({"event": "evaluation_completed", "group": group, "method": method}), flush=True)
            del active[slot]
    report = {"event": "evaluation_complete", "failures": failures}
    (ROOT / "evaluation_run_report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report), flush=True)
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
