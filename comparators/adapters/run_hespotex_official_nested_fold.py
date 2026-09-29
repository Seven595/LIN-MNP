#!/usr/bin/env python3
"""Run one nested-validation/refit fold of the released HESpotEx core."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
import time

import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[2]
HESPOTEX_ROOT = PROJECT_ROOT / "comparators/methods/HESpotEx"
if str(HESPOTEX_ROOT) not in sys.path:
    sys.path.insert(0, str(HESPOTEX_ROOT))

from lin_mnp.paths import RESULTS_ROOT
import run_cscc_reproduction as cscc
import run_her2_reproduction as her2


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cohort", choices=("her2", "cscc"), required=True)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--fold", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-selection-epochs", type=int, default=30)
    args = parser.parse_args()

    contract_path = args.contract.resolve()
    output_root = args.output.resolve()
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    row = contract["folds"][args.fold]
    outer_id = str(row["outer_test_sample_id"])
    validation_id = str(row["inner_validation_sample_id"])
    inner_train = tuple(map(str, row["inner_train_sample_ids"]))
    outer_train = tuple(map(str, row["outer_train_sample_ids"]))
    output = (
        output_root
        / contract["cohort_id"]
        / "hespotex_official"
        / f"fold{args.fold}"
    )
    if (output / "complete.json").is_file():
        print(json.dumps({"output": str(output), "status": "existing"}))
        return
    if output.exists():
        raise FileExistsError(output)
    staging = output.with_name(f"{output.name}.tmp")
    staging.mkdir(parents=True)
    selection_dir = staging / "selection"
    refit_dir = staging / "refit"
    selection_dir.mkdir()
    refit_dir.mkdir()
    started = time.time()

    if args.cohort == "her2":
        cache_root = RESULTS_ROOT / "hespotex/her2_official"
        datasets = her2.load_datasets(cache_root)
        out_embedding = 2_759
        released_train = her2.base.released_train
    else:
        cache_root = RESULTS_ROOT / "hespotex/cscc_official"
        datasets = cscc.load_datasets(cache_root)
        out_embedding = len(datasets[outer_id].genes)
        released_train = cscc.released_train

    inner_train_dataset = torch.utils.data.ConcatDataset(
        [datasets[value] for value in inner_train]
    )
    validation_dataset = torch.utils.data.ConcatDataset([datasets[validation_id]])
    previous = Path.cwd()
    os.chdir(HESPOTEX_ROOT)
    try:
        _, selection = released_train.main(
            inner_train_dataset,
            validation_dataset,
            validation_dataset,
            image_embedding=64,
            out_embedding=out_embedding,
            max_epochs=args.max_selection_epochs,
            save_path=str(selection_dir),
            method="phikon",
            device=args.device,
            return_metadata=True,
            checkpoint_selection="validation",
        )
        selected_epoch = int(selection["best_epoch_one_based"])
        outer_train_dataset = torch.utils.data.ConcatDataset(
            [datasets[value] for value in outer_train]
        )
        outer_test_dataset = torch.utils.data.ConcatDataset([datasets[outer_id]])
        predictions, refit = released_train.main(
            outer_train_dataset,
            None,
            outer_test_dataset,
            image_embedding=64,
            out_embedding=out_embedding,
            max_epochs=selected_epoch,
            save_path=str(refit_dir),
            method="phikon",
            device=args.device,
            return_metadata=True,
            checkpoint_selection="fixed_final_epoch",
        )
    finally:
        os.chdir(previous)

    prediction = np.asarray(predictions[0], dtype=np.float32)
    prediction_root = staging / "predictions"
    prediction_root.mkdir()
    prediction_path = prediction_root / f"{outer_id}.npz"
    np.savez_compressed(
        prediction_path,
        prediction=prediction,
        genes=np.asarray(datasets[outer_id].genes, dtype=str),
        barcodes=np.asarray(datasets[outer_id].barcodes, dtype=str),
        sample_id=np.asarray(outer_id),
        patient_id=np.asarray(datasets[outer_id].patient),
        fold_id=np.asarray(args.fold, dtype=np.int32),
    )
    metadata = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "method_id": "hespotex_official",
        "method": "HESpotEx official rerun",
        "fold_id": args.fold,
        "outer_test_sample_id": outer_id,
        "outer_train_sample_ids": list(outer_train),
        "inner_validation_sample_id": validation_id,
        "inner_train_sample_ids": list(inner_train),
        "max_selection_epochs": args.max_selection_epochs,
        "selection": selection,
        "selected_epoch": selected_epoch,
        "refit": refit,
        "refit_epochs": selected_epoch,
        "checkpoint_selection_scope": "inner_validation_only",
        "refit_scope": "all_outer_train_sections",
        "outer_target_used_for_training_or_selection": False,
        "target_opened_for_final_scoring": False,
        "cache_root": str(cache_root.resolve()),
        "prediction_sha256": sha256_file(prediction_path),
        "wall_time_seconds": time.time() - started,
    }
    (staging / "run_metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    complete = {
        "run_metadata_sha256": sha256_file(staging / "run_metadata.json"),
        "selection_checkpoint_sha256": sha256_file(
            selection_dir / "best_unLSTM.pt"
        ),
        "refit_checkpoint_sha256": sha256_file(refit_dir / "best_unLSTM.pt"),
        "prediction_sha256": sha256_file(prediction_path),
    }
    (staging / "complete.json").write_text(
        json.dumps(complete, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    staging.rename(output)
    print(json.dumps({"output": str(output), "selected_epoch": selected_epoch}))


if __name__ == "__main__":
    main()
