#!/usr/bin/env python3
"""Run and independently score the HEST/STImage 50-gene comparator campaign.

Each invocation handles one method and seals its predictions and metadata
before the next method starts. All neural comparators use the declared
30-epoch cap and validation-only checkpoint selection.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Sequence

import numpy as np

from lin_mnp.paths import ARCHIVE_ROOT, REPOSITORY_ROOT, RESULTS_ROOT

ARCHIVE = ARCHIVE_ROOT
PROJECT = REPOSITORY_ROOT
CONTRACTS = ARCHIVE / "benchmarks/hest_stimage/contracts"
OUTPUT = RESULTS_ROOT / "benchmarks/hest_stimage"
RUNTIME = RESULTS_ROOT / "runtime/hest_stimage"
FEATURES = RUNTIME / "features"
ADAPTERS = PROJECT / "comparators/adapters"
BENCHMARK_CODE = PROJECT / "experiments/benchmarks"
METHODS = (
    "holotea", "spahgc", "stag",
    "pacest", "hespotex",
    "he2gene", "asgnn", "gr2st", "heclip", "tcgn", "omiclip", "igidl",
    "bleep", "hist2st", "thitogene", "mclstexp", "stnet", "deepspace",
    "deeppt", "histogene", "egnv1", "egnv2", "genecoder", "istar",
)
DISPLAY = {
    "holotea": "HoloTea", "spahgc": "SpaHGC", "stag": "STAG",
    "pacest": "Legacy PACE-ST",
    "hespotex": "HESpotEx released core",
    "he2gene": "HE2Gene", "asgnn": "asGNN", "gr2st": "GR2ST",
    "heclip": "HECLIP", "tcgn": "TCGN", "omiclip": "OmiCLIP",
    "igidl": "IGI-DL", "bleep": "BLEEP", "hist2st": "Hist2ST",
    "thitogene": "THItoGene", "mclstexp": "mclSTExp", "stnet": "ST-Net",
    "deepspace": "DeepSpaCE", "deeppt": "DeepPT", "histogene": "HisToGene",
    "egnv1": "EGNv1", "egnv2": "EGNv2", "genecoder": "GeneCodeR",
    "istar": "iStar",
}
HEST_DATASETS = ("IDC", "PRAD", "PAAD", "SKCM", "COAD", "READ", "CCRCC", "HCC", "LUNG", "LYMPH_IDC")
STIMAGE_TISSUES = ("breast", "brain", "skin", "mouth", "prostate", "stomach", "colon")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def contracts() -> list[dict[str, object]]:
    rows = []
    for benchmark, groups in (("hest", HEST_DATASETS), ("stimage", STIMAGE_TISSUES)):
        for group in groups:
            for seed in (1, 2, 3):
                path = CONTRACTS / benchmark / group / f"seed{seed}.json"
                payload = json.loads(path.read_text(encoding="utf-8"))
                rows.append({
                    "benchmark": benchmark, "group": group, "seed": seed,
                    "path": path, "fold_count": int(payload["fold_count"]),
                })
    return rows


def environment(contract: Path, gpu: str, threads: int) -> dict[str, str]:
    value = os.environ.copy()
    value.update({
        "LIN_MNP_EXTERNAL_CONTRACT": str(contract.resolve()),
        "LIN_MNP_EGN_EMBEDDING_ROOT": str((FEATURES / "egn").resolve()),
        "LIN_MNP_DEEPPT_FEATURE_ROOT": str((FEATURES / "deeppt").resolve()),
        "LIN_MNP_ISTAR_FEATURE_ROOT": str((FEATURES / "istar").resolve()),
        "LIN_MNP_CELLPOSE_NUCLEI_ROOT": str((FEATURES / "cellpose/slides").resolve()),
        "LIN_MNP_UNI_ROOT": str((FEATURES / "uni_v1").resolve()),
        # GR2ST's released adapter appends ``slides/<sample>.npz`` itself.
        "LIN_MNP_PHIKON_ROOT": str((FEATURES / "phikon").resolve()),
        "LIN_MNP_PHIKON_CROP_SIZE": "224",
        "LIN_MNP_RESNET_ROOT": str((FEATURES / "deeppt").resolve()),
        "LIN_MNP_CHECKPOINT_SELECTION_SCOPE": "inner_validation",
        "CUDA_VISIBLE_DEVICES": gpu, "OMP_NUM_THREADS": str(threads),
        "MKL_NUM_THREADS": str(threads), "HF_HUB_OFFLINE": "1",
    })
    return value


def fold_command(method: str, fold: int, seed: int, output: Path, epochs: int) -> list[str]:
    common = ["--fold", str(fold), "--seed", str(seed), "--output", str(output)]
    if method in {"holotea", "spahgc", "stag"}:
        batch_size = "128" if method == "stag" else "16"
        return [sys.executable, str(BENCHMARK_CODE / "train_additional_comparators.py"),
                "--method", method, "--device", "cuda:0", "--epochs", str(epochs),
                "--workers", "4" if method == "stag" else "2", "--batch-size", batch_size, *common]
    if method == "pacest":
        return [sys.executable, str(ARCHIVE / "tools/train_pacest_external_fold.py"),
                "--device", "cuda:0", "--epochs", str(epochs),
                "--cache", str(FEATURES / "ctranspath"), *common]
    if method == "hespotex":
        return [sys.executable, str(ARCHIVE / "tools/train_hespotex_strict_external_fold.py"),
                "--device", "cuda:0", "--epochs", str(epochs),
                "--quilt", str(FEATURES / "quilt"),
                "--nuclei", str(FEATURES / "cellpose/slides"), *common]
    if method in {"stnet", "deepspace", "tcgn"}:
        command = [sys.executable, str(ADAPTERS / "train_benchmark_external_spot_fold.py"),
                "--method", method, "--device", "cuda:0", "--workers", "2",
                "--max-epochs", str(epochs), "--compact"]
        # The released TCGN graph/CMT path is seed-dependently non-finite in
        # half precision even when graph statistics alone are promoted.  Use
        # one fixed FP32 policy for every fold; the other two CNNs retain AMP.
        if method != "tcgn":
            command.append("--amp")
        return [*command, *common]
    if method in {"histogene", "hist2st", "thitogene"}:
        return [sys.executable, str(ADAPTERS / "train_benchmark_external_slide_fold.py"),
                "--method", method, "--device", "cuda:0", "--max-epochs", str(epochs),
                "--amp", *common]
    if method in {"bleep", "mclstexp", "igidl"}:
        return [sys.executable, str(ADAPTERS / "train_benchmark_external_retrieval_fold.py"),
                "--method", method, "--device", "cuda:0", "--epochs", str(epochs),
                "--patience", str(epochs), "--batch-size", "128", *common]
    if method == "omiclip":
        return [sys.executable, str(ADAPTERS / "train_benchmark_external_omiclip_fold.py"),
                "--device", "cuda:0", "--cache", str(FEATURES / "omiclip"),
                "--batch-size", "128", "--fold", str(fold), "--output", str(output)]
    script = {
        "deeppt": "train_benchmark_external_deeppt_fold.py",
        "egnv1": "train_benchmark_external_egnv1_fold.py",
        "egnv2": "train_benchmark_external_egnv2_fold.py",
        "genecoder": "train_benchmark_external_genecoder_fold.py",
        "istar": "train_benchmark_external_istar_fold.py",
    }.get(method)
    if script is not None:
        command = [sys.executable, str(ADAPTERS / script)]
        if method == "deeppt":
            command += ["--device", "cuda:0", "--ae-epochs", str(epochs),
                        "--mlp-epochs", str(epochs), "--patience", str(epochs)]
        elif method == "egnv1":
            command += ["--device", "cuda:0", "--workers", "2", "--epochs", str(epochs), "--amp"]
        elif method == "egnv2":
            command += ["--device", "cuda:0", "--epochs", str(epochs)]
        elif method == "istar":
            command += ["--device", "cuda:0", "--workers", "2", "--epochs", str(epochs), "--amp"]
        command += common
        return command
    if method in {"gr2st", "heclip", "he2gene", "asgnn"}:
        return [sys.executable, str(ADAPTERS / "train_additional_comparators_fold.py"),
                "--method", method, "--device", "cuda:0", "--workers", "2",
                "--max-epochs", str(epochs), "--amp", *common]
    raise ValueError(method)


def compact_fold(fold_root: Path) -> None:
    removed = []
    for name in ("best_checkpoint.pt", "best_checkpoint.npz", "training_text_embeddings.npz"):
        path = fold_root / name
        if path.is_file():
            removed.append({"name": name, "sha256": sha256(path), "bytes": path.stat().st_size})
            path.unlink()
    if removed:
        (fold_root / "checkpoint_retention.json").write_text(json.dumps({
            "schema_version": 1,
            "policy": "checkpoint removed after prediction export and hash verification",
            "predictions_and_training_history_retained": True, "removed": removed,
        }, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def release_launch_lock(lock_path: Path) -> None:
    """Release one task lock without touching another launcher's lock."""
    owner_path = lock_path / "owner.json"
    try:
        owner = json.loads(owner_path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return
    if int(owner.get("pid", -1)) != os.getpid():
        return
    owner_path.unlink(missing_ok=True)
    try:
        lock_path.rmdir()
    except FileNotFoundError:
        pass


def acquire_launch_lock(lock_path: Path, task_id: str) -> bool:
    """Atomically claim a fold, reclaiming only locks owned by dead runners."""
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        lock_path.mkdir()
    except FileExistsError:
        owner_path = lock_path / "owner.json"
        try:
            owner = json.loads(owner_path.read_text(encoding="utf-8"))
            owner_pid = int(owner["pid"])
            os.kill(owner_pid, 0)
            return False
        except ProcessLookupError:
            pass
        except (FileNotFoundError, KeyError, ValueError, json.JSONDecodeError):
            # A competing launcher may be between mkdir and owner.json. Wait
            # rather than guessing that a just-created lock is stale.
            return False
        owner_path.unlink(missing_ok=True)
        try:
            lock_path.rmdir()
        except (FileNotFoundError, OSError):
            return False
        try:
            lock_path.mkdir()
        except FileExistsError:
            return False
    (lock_path / "owner.json").write_text(json.dumps({
        "pid": os.getpid(), "task_id": task_id,
        "acquired_at_utc": datetime.now(timezone.utc).isoformat(),
    }, sort_keys=True) + "\n", encoding="utf-8")
    return True


def run_method(args: argparse.Namespace) -> None:
    tasks = []
    task_ordinal = 0
    for row in contracts():
        run_root = OUTPUT / "runs" / str(row["benchmark"]) / str(row["group"]) / f"seed{row['seed']}"
        for fold in range(int(row["fold_count"])):
            assigned = task_ordinal % args.task_shard_count == args.task_shard_index
            task_ordinal += 1
            if not assigned:
                continue
            fold_root = run_root / args.method / f"fold{fold}"
            if (fold_root / "complete.json").is_file():
                compact_fold(fold_root)
                continue
            tasks.append({**row, "fold": fold, "run_root": run_root,
                          "id": f"{row['benchmark']}:{row['group']}:s{row['seed']}:f{fold}"})
    logs = OUTPUT / "launcher_logs" / args.method
    logs.mkdir(parents=True, exist_ok=True)
    pending = list(reversed(tasks)) if args.task_order == "reverse" else list(tasks)
    running, records = [], {}
    while pending or running:
        while pending and len(running) < len(args.gpus):
            task = pending.pop(0)
            # Long-lived forward/reverse launchers can converge after another
            # shard has completed a task that was pending when this process
            # started. Re-check at dispatch time so stale queues never retrain
            # or compete for the same fold output directory.
            fold_root = Path(task["run_root"]) / args.method / f"fold{task['fold']}"
            if (fold_root / "complete.json").is_file():
                compact_fold(fold_root)
                continue
            lock_path = (
                Path(task["run_root"]) / ".launch_locks" /
                f"{args.method}.fold{task['fold']}.lock"
            )
            if not acquire_launch_lock(lock_path, str(task["id"])):
                # Keep the task eligible in case the other launcher fails;
                # dispatch another task only after this lock is resolved.
                pending.append(task)
                break
            occupied = {str(item["gpu"]) for item in running}
            gpu = next(value for value in args.gpus if value not in occupied)
            log = logs / f"{str(task['id']).replace(':', '__')}.log"
            handle = log.open("ab")
            command = fold_command(args.method, int(task["fold"]), int(task["seed"]),
                                   Path(task["run_root"]), args.epochs)
            handle.write((json.dumps({"command": command, "gpu": gpu}) + "\n").encode()); handle.flush()
            try:
                process = subprocess.Popen(command, cwd=PROJECT,
                    env=environment(Path(task["path"]), gpu, args.threads),
                    stdout=handle, stderr=subprocess.STDOUT)
            except BaseException:
                handle.close()
                release_launch_lock(lock_path)
                raise
            running.append({**task, "gpu": gpu, "process": process, "handle": handle,
                            "log": str(log), "started": time.time(), "lock": lock_path})
        time.sleep(2)
        active = []
        for task in running:
            code = task["process"].poll()
            if code is None:
                active.append(task); continue
            task["handle"].close()
            release_launch_lock(Path(task["lock"]))
            record = {"return_code": code, "gpu": task["gpu"], "log": task["log"],
                      "seconds": time.time() - float(task["started"])}
            records[str(task["id"])] = record
            print(json.dumps({task["id"]: record}, sort_keys=True), flush=True)
            if code == 0:
                compact_fold(Path(task["run_root"]) / args.method / f"fold{task['fold']}")
            else:
                for other in active:
                    other["process"].terminate(); other["handle"].close()
                raise RuntimeError(f"failed task {task['id']}; see {task['log']}")
        running = active
    status = {"schema_version": 1, "method": args.method, "display": DISPLAY[args.method],
              "epochs": args.epochs, "checkpoint_selection": "inner validation only",
              "task_count": len(tasks), "records": records,
              "task_shard_index": args.task_shard_index,
              "task_shard_count": args.task_shard_count,
              "task_order": args.task_order,
              "finished_at_utc": datetime.now(timezone.utc).isoformat()}
    (OUTPUT / "launcher_status").mkdir(parents=True, exist_ok=True)
    status_name = (
        f"{args.method}.{args.task_order}.json"
        if args.task_shard_count == 1
        else f"{args.method}.{args.task_order}.shard{args.task_shard_index}-of-{args.task_shard_count}.json"
    )
    (OUTPUT / "launcher_status" / status_name).write_text(
        json.dumps(status, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def decode(values: np.ndarray) -> tuple[str, ...]:
    return tuple(map(str, np.asarray(values).reshape(-1)))


def pcc(target: np.ndarray, prediction: np.ndarray) -> np.ndarray:
    left = target.astype(np.float64) - target.mean(0, keepdims=True)
    right = prediction.astype(np.float64) - prediction.mean(0, keepdims=True)
    denominator = np.sqrt(np.square(left).sum(0) * np.square(right).sum(0))
    return np.divide((left * right).sum(0), denominator,
                     out=np.full(target.shape[1], np.nan), where=denominator > 1e-12)


def prediction(method: str, run_root: Path, fold: int, sample: str,
               target_root: Path, genes: tuple[str, ...]) -> tuple[np.ndarray, np.ndarray]:
    fold_root = run_root / method / f"fold{fold}"
    complete = json.loads((fold_root / "complete.json").read_text(encoding="utf-8"))
    metadata_path = fold_root / "run_metadata.json"
    if sha256(metadata_path) != complete["run_metadata_sha256"]:
        raise ValueError(f"metadata hash failure: {fold_root}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if sample not in tuple(map(str, metadata["fold"].get("test_ids", []))):
        raise ValueError(f"test identity failure: {fold_root}/{sample}")
    pred_path = fold_root / "predictions" / f"{sample}.npz"
    prediction_hashes = complete.get("prediction_sha256", metadata.get("prediction_sha256"))
    if not isinstance(prediction_hashes, dict) or sha256(pred_path) != prediction_hashes[sample]:
        raise ValueError(f"prediction hash failure: {pred_path}")
    with np.load(pred_path, allow_pickle=False) as archive:
        pred = archive["prediction"].astype(np.float32)
        pred_genes, pred_barcodes = decode(archive["genes"]), decode(archive["barcodes"])
    with np.load(target_root / "slides" / f"{sample}.npz", allow_pickle=False) as archive:
        truth = archive["target"].astype(np.float32)
        target_genes, target_barcodes = decode(archive["genes"]), decode(archive["barcodes"])
    if pred.shape != truth.shape or pred_genes != genes or target_genes != genes or pred_barcodes != target_barcodes:
        raise ValueError(f"prediction/target alignment failure: {pred_path}")
    if not np.isfinite(pred).all():
        raise ValueError(f"nonfinite prediction: {pred_path}")
    if method == "pacest":
        # The archived PACE-ST head is trained in within-section standardized
        # expression space.  Standardize both arrays section-wise before the
        # common pooled STImage metric; within-section PCC is unchanged.
        truth = (truth - truth.mean(0, keepdims=True)) / np.maximum(
            truth.std(0, keepdims=True), 1e-4)
        pred = (pred - pred.mean(0, keepdims=True)) / np.maximum(
            pred.std(0, keepdims=True), 1e-4)
    return truth, pred


def evaluate(method: str) -> dict[str, object]:
    detail = []
    hest_group = []
    for dataset in HEST_DATASETS:
        seed_values = []
        for seed in (1, 2, 3):
            contract_path = CONTRACTS / "hest" / dataset / f"seed{seed}.json"
            contract = json.loads(contract_path.read_text(encoding="utf-8"))
            target_root = Path(contract["target_bundle"])
            genes = tuple(x.strip() for x in (target_root / "gene_panel.txt").read_text().splitlines() if x.strip())
            fold_values = []
            for fold in contract["folds"]:
                fold_pcc, fold_mse = [], []
                for sample in map(str, fold["test_sample_ids"]):
                    truth, pred = prediction(method, OUTPUT / "runs/hest" / dataset / f"seed{seed}",
                                             int(fold["fold_id"]), sample, target_root, genes)
                    fold_pcc.extend(pcc(truth, pred).tolist())
                    fold_mse.append(float(np.mean(np.square(truth - pred))))
                row = {"benchmark": "HEST", "group": dataset, "seed": seed,
                       "fold": int(fold["fold_id"]), "pcc": float(np.nanmean(fold_pcc)),
                       "mse": float(np.mean(fold_mse))}
                detail.append(row); fold_values.append((row["pcc"], row["mse"]))
            seed_values.append(tuple(np.mean(fold_values, axis=0)))
        group_value = tuple(np.mean(seed_values, axis=0)); hest_group.append(group_value)
    stimage_group = []
    for tissue in STIMAGE_TISSUES:
        seed_values = []
        for seed in (1, 2, 3):
            contract_path = CONTRACTS / "stimage" / tissue / f"seed{seed}.json"
            contract = json.loads(contract_path.read_text(encoding="utf-8")); fold = contract["folds"][0]
            target_root = Path(contract["target_bundle"])
            genes = tuple(x.strip() for x in (target_root / "gene_panel.txt").read_text().splitlines() if x.strip())
            truths, predictions = [], []
            for sample in map(str, fold["test_sample_ids"]):
                truth, pred = prediction(method, OUTPUT / "runs/stimage" / tissue / f"seed{seed}",
                                         0, sample, target_root, genes)
                truths.append(truth); predictions.append(pred)
            truth, pred = np.concatenate(truths), np.concatenate(predictions)
            row = {"benchmark": "STImage", "group": tissue, "seed": seed, "fold": 0,
                   "pcc": float(np.nanmean(pcc(truth, pred))),
                   "mse": float(np.mean(np.square(truth - pred)))}
            detail.append(row); seed_values.append((row["pcc"], row["mse"]))
        stimage_group.append(tuple(np.mean(seed_values, axis=0)))
    summary = {"schema_version": 1, "method_id": method, "method": DISPLAY[method],
               "target": (
                   "within-section standardized common log1p raw; 50 genes"
                   if method == "pacest" else "common deposited log1p raw counts; 50 genes"
               ),
               "selection": "training-only validation; 30-epoch cap where applicable",
               "HEST": {"pcc": float(np.mean(hest_group, axis=0)[0]),
                        "mse": float(np.mean(hest_group, axis=0)[1]), "datasets": 10, "seeds": 3},
               "STImage": {"pcc": float(np.mean(stimage_group, axis=0)[0]),
                           "mse": float(np.mean(stimage_group, axis=0)[1]), "tissues": 7, "seeds": 3},
               "evaluated_at_utc": datetime.now(timezone.utc).isoformat()}
    out = OUTPUT / "evaluation" / method; out.mkdir(parents=True, exist_ok=True)
    with (out / "detail.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(detail[0])); writer.writeheader(); writer.writerows(detail)
    (out / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, sort_keys=True), flush=True)
    return summary


def cache_command(kind: str) -> list[str]:
    mapping = {
        "egn50": [sys.executable, str(ADAPTERS / "cache_benchmark_egn_embeddings.py"), "--backbone", "resnet50", "--device", "cuda:0", "--workers", "2", "--output", str(FEATURES / "egn")],
        "egn18": [sys.executable, str(ADAPTERS / "cache_benchmark_egn_embeddings.py"), "--backbone", "resnet18", "--device", "cuda:0", "--workers", "2", "--output", str(FEATURES / "egn")],
        "deeppt": [sys.executable, str(ADAPTERS / "cache_benchmark_deeppt_embeddings.py"), "--device", "cuda:0", "--workers", "2", "--output", str(FEATURES / "deeppt")],
        "istar": [sys.executable, str(ADAPTERS / "cache_benchmark_istar_features.py"), "--device", "cuda:0", "--batch-size", "16", "--output", str(FEATURES / "istar")],
        "omiclip": [sys.executable, str(ADAPTERS / "cache_benchmark_omiclip_embeddings.py"), "--device", "cuda:0", "--batch-size", "32", "--workers", "2", "--amp", "--output", str(FEATURES / "omiclip")],
        "phikon": [sys.executable, str(ARCHIVE / "tools/cache_hespotex_phikon.py"), "--device", "cuda:0", "--batch-size", "32", "--output", str(FEATURES / "phikon")],
    }
    return mapping[kind]


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__); sub = parser.add_subparsers(dest="action", required=True)
    run = sub.add_parser("run"); run.add_argument("--method", choices=METHODS, required=True); run.add_argument("--gpus", default="0,1,2,3,4,5,6,7"); run.add_argument("--epochs", type=int, default=30); run.add_argument("--threads", type=int, default=4); run.add_argument("--task-shard-index", type=int, default=0); run.add_argument("--task-shard-count", type=int, default=1); run.add_argument("--task-order", choices=("forward", "reverse"), default="forward")
    score = sub.add_parser("evaluate"); score.add_argument("--method", choices=METHODS, required=True)
    cache = sub.add_parser("cache"); cache.add_argument("--kind", choices=("egn50", "egn18", "deeppt", "istar", "omiclip", "phikon", "ctranspath", "quilt"), required=True); cache.add_argument("--gpu", default="0")
    args = parser.parse_args(argv)
    if args.action == "run":
        if args.task_shard_count < 1 or not 0 <= args.task_shard_index < args.task_shard_count:
            parser.error("task shard requires 0 <= index < count and count >= 1")
        args.gpus = tuple(x.strip() for x in args.gpus.split(",") if x.strip()); run_method(args)
    elif args.action == "evaluate": evaluate(args.method)
    else:
        representative = CONTRACTS / "hest/IDC/seed1.json"
        command = (cache_command(args.kind) if args.kind not in {"ctranspath", "quilt"} else [
            sys.executable, str(ARCHIVE / "tools/cache_global_ctranspath224.py"),
            "--patch-root", str(RUNTIME / "patches_224_v1"),
            "--output", str(FEATURES / "ctranspath"), "--device", "cuda:0",
        ])
        if args.kind == "quilt":
            command = [sys.executable, str(ARCHIVE / "tools/cache_global_hespotex_quilt.py"),
                       "--patch-root", str(RUNTIME / "patches_224_v1"),
                       "--output", str(FEATURES / "quilt"), "--device", "cuda:0"]
        subprocess.run(command, cwd=PROJECT,
                       env=environment(representative, args.gpu, 4), check=True)


if __name__ == "__main__":
    main()
