#!/usr/bin/env python3
"""Run released OmiCLIP/Loki PredEx on one fixed benchmark patient fold."""

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
from typing import Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from open_clip import create_model_from_pretrained, get_tokenizer


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from lin_mnp.benchmark.benchmark_external import (  # noqa: E402
    PATCH_ROOT,
    apply_target_transform,
    fit_target_transform,
    load_folds,
    load_slides,
)
from lin_mnp.paths import CACHE_ROOT, MODEL_ROOT, RESULTS_ROOT


DEFAULT_OUTPUT = RESULTS_ROOT / "comparator_reruns"
DEFAULT_CHECKPOINT = MODEL_ROOT / "omiclip/checkpoint.pt"
DEFAULT_CACHE = CACHE_ROOT / "cscc/omiclip_embeddings"
LOKI_SOURCE = PROJECT_ROOT / "comparators/methods/OmiCLIP_Loki/src/loki"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def source_revision(path: Path) -> str:
    if path.joinpath(".git").exists():
        return subprocess.check_output(
            ["git", "-C", str(path), "rev-parse", "HEAD"], text=True
        ).strip()
    return "official source snapshot; identified by source hashes"


def target_rows(
    slides: Mapping[str, object], sample_ids: Sequence[str], parameters: Mapping[str, object]
) -> np.ndarray:
    return np.concatenate(
        [apply_target_transform(slides[name], parameters) for name in sample_ids]
    ).astype(np.float32, copy=False)


def cache_rows(
    slides: Mapping[str, object], cache: np.ndarray, sample_ids: Sequence[str]
) -> np.ndarray:
    return np.concatenate(
        [cache[slides[name].patch_start : slides[name].patch_stop] for name in sample_ids]
    ).astype(np.float32, copy=False)


def top_gene_sentences(expression: np.ndarray, genes: Sequence[str], count: int = 50) -> list[str]:
    # The release accepts a caller-provided housekeeping list.  No such list is
    # distributed with Loki, so only its unconditional punctuation exclusions
    # are applied here and the empty housekeeping set is recorded in metadata.
    eligible = np.asarray(
        [index for index, gene in enumerate(genes) if "." not in gene and "-" not in gene],
        dtype=np.int64,
    )
    # External 50-gene panels can contain punctuation-filtered symbols.  Loki's
    # released preprocessing drops those genes, so use every remaining eligible
    # gene when fewer than 50 survive instead of making the method undefined.
    count = min(count, len(eligible))
    if count == 0:
        raise ValueError("no eligible genes for OmiCLIP sentences")
    values = expression[:, eligible]
    local = np.argpartition(values, -count, axis=1)[:, -count:]
    order = np.take_along_axis(values, local, axis=1).argsort(axis=1)[:, ::-1]
    selected = eligible[np.take_along_axis(local, order, axis=1)]
    gene_array = np.asarray(genes)
    return [" ".join(gene_array[row]) for row in selected]


@torch.no_grad()
def encode_texts(
    model: torch.nn.Module,
    tokenizer,
    texts: Sequence[str],
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    rows = []
    for start in range(0, len(texts), batch_size):
        tokens = tokenizer(list(texts[start : start + batch_size])).to(device)
        # open_clip 3.x leaves the CoCa token-embedding activation in fp32
        # while ``precision=fp16`` converts attention weights.  The released
        # inference path is restored by the same autocast used for images.
        with torch.autocast(
            device_type="cuda", dtype=torch.float16,
            enabled=device.type == "cuda",
        ):
            embedding = model.encode_text(tokens)
        rows.append(F.normalize(embedding.float(), p=2, dim=-1).cpu().numpy())
    return np.concatenate(rows).astype(np.float32)


def loki_predict(
    image_embedding: np.ndarray,
    text_embedding: np.ndarray,
    train_expression: np.ndarray,
) -> tuple[np.ndarray, dict[str, float]]:
    similarity = image_embedding @ text_embedding.T
    denominator = similarity.sum(axis=1, keepdims=True)
    epsilon = np.finfo(np.float32).eps
    near_zero = np.abs(denominator) < epsilon
    safe = np.where(near_zero, np.where(denominator < 0, -epsilon, epsilon), denominator)
    prediction = similarity @ train_expression / safe
    diagnostics = {
        "similarity_min": float(similarity.min()),
        "similarity_max": float(similarity.max()),
        "similarity_mean": float(similarity.mean()),
        "near_zero_row_sums": int(near_zero.sum()),
    }
    return prediction.astype(np.float32), diagnostics


def mean_slide_gene_pcc(
    prediction: np.ndarray,
    target: np.ndarray,
    slides: Mapping[str, object],
    sample_ids: Sequence[str],
) -> float:
    rows = []
    start = 0
    for sample_id in sample_ids:
        stop = start + len(slides[sample_id].barcodes)
        pred = prediction[start:stop].astype(np.float64)
        true = target[start:stop].astype(np.float64)
        pred -= pred.mean(0, keepdims=True)
        true -= true.mean(0, keepdims=True)
        denominator = np.sqrt(np.sum(pred * pred, 0) * np.sum(true * true, 0))
        rows.append(np.divide(
            np.sum(pred * true, 0), denominator,
            out=np.full_like(denominator, np.nan), where=denominator > 0,
        ))
        start = stop
    return float(np.nanmean(np.nanmean(np.stack(rows), axis=0)))


def split_predictions(
    matrix: np.ndarray, slides: Mapping[str, object], sample_ids: Sequence[str]
) -> dict[str, np.ndarray]:
    rows = {}
    start = 0
    for sample_id in sample_ids:
        stop = start + len(slides[sample_id].barcodes)
        rows[sample_id] = matrix[start:stop]
        start = stop
    return rows


def run_fold(args: argparse.Namespace) -> Path:
    fold = load_folds()[args.fold]
    slides, genes = load_slides()
    parameters = fit_target_transform(slides, fold.train_ids, "log1p_cp10k")
    output = (args.output / "omiclip" / f"fold{fold.fold_id}").resolve()
    if (output / "complete.json").is_file():
        return output
    if output.exists():
        raise FileExistsError(f"incomplete fold output exists: {output}")
    staging = output.with_name(f"{output.name}.tmp-{os.getpid()}")
    staging.mkdir(parents=True, exist_ok=False)
    started = time.time()
    complete_cache = json.loads((args.cache / "complete.json").read_text(encoding="utf-8"))
    cache_metadata = json.loads((args.cache / "metadata.json").read_text(encoding="utf-8"))
    cache_path = args.cache / "embeddings_float32.npy"
    if sha256_file(cache_path) != complete_cache["embedding_sha256"]:
        raise ValueError("OmiCLIP image cache hash mismatch")
    image_cache = np.load(cache_path, mmap_mode="r")
    train_expression = target_rows(slides, fold.train_ids, parameters)
    sentences = top_gene_sentences(train_expression, genes)

    device = torch.device(args.device)
    model, _ = create_model_from_pretrained(
        "coca_ViT-L-14", pretrained=str(args.checkpoint), device=device,
        precision="fp16", weights_only=False,
    )
    model.eval()
    tokenizer = get_tokenizer("coca_ViT-L-14")
    text_embedding = encode_texts(model, tokenizer, sentences, device, args.batch_size)
    del model
    torch.cuda.empty_cache()

    val_image = cache_rows(slides, image_cache, fold.val_ids)
    val_target = target_rows(slides, fold.val_ids, parameters)
    val_prediction, val_diagnostics = loki_predict(val_image, text_embedding, train_expression)
    validation_pcc = mean_slide_gene_pcc(val_prediction, val_target, slides, fold.val_ids)
    test_image = cache_rows(slides, image_cache, fold.test_ids)
    test_matrix, test_diagnostics = loki_predict(test_image, text_embedding, train_expression)
    predictions = split_predictions(test_matrix, slides, fold.test_ids)

    checkpoint = staging / "training_text_embeddings.npz"
    np.savez_compressed(
        checkpoint,
        embedding=text_embedding,
        train_sample_ids=np.asarray(fold.train_ids),
        sentence_sha256=np.asarray(
            hashlib.sha256("\n".join(sentences).encode("utf-8")).hexdigest()
        ),
    )
    history = staging / "training_history.csv"
    with history.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["epoch", "validation_mean_gene_pcc", "elapsed_seconds"],
        )
        writer.writeheader()
        writer.writerow({
            "epoch": 0,
            "validation_mean_gene_pcc": validation_pcc,
            "elapsed_seconds": time.time() - started,
        })
    prediction_dir = staging / "predictions"
    prediction_dir.mkdir()
    prediction_rows = []
    for sample_id in fold.test_ids:
        path = prediction_dir / f"{sample_id}.npz"
        np.savez_compressed(
            path,
            prediction=predictions[sample_id],
            genes=np.asarray(genes),
            barcodes=np.asarray(slides[sample_id].barcodes),
            sample_id=np.asarray(sample_id),
            patient_id=np.asarray(slides[sample_id].patient_id),
            fold_id=np.asarray(fold.fold_id, dtype=np.int32),
        )
        prediction_rows.append({
            "sample_id": sample_id,
            "patient_id": slides[sample_id].patient_id,
            "path": str(path.relative_to(staging)),
            "sha256": sha256_file(path),
            "n_spots": len(slides[sample_id].barcodes),
            "n_genes": len(genes),
        })
    method = {
        "method_id": "omiclip",
        "display_name": "OmiCLIP",
        "source_dir": "comparators/methods/OmiCLIP_Loki/src/loki",
        "source_kind": "released OmiCLIP checkpoint and Loki PredEx source equation",
        "target_transform": "log1p_cp10k",
        "model": "coca_ViT-L-14",
        "gene_sentence_size": min(
            50, sum("." not in gene and "-" not in gene for gene in genes)
        ),
        "housekeeping_exclusion": "empty because caller-supplied list is not distributed with Loki",
        "inference_equation": "raw cosine similarity @ training expression / row similarity sum",
        "runtime_compatibility_patches": [
            "locked997 training expression supplies each training spot's top-50 gene sentence",
            "only training-fold expression is encoded as text; validation/test expression is never encoded",
            "caller-supplied housekeeping list is unavailable, so only released punctuation filters are applied",
            "50-gene panels use all punctuation-eligible genes when fewer than 50 survive the released filter",
            "weights_only=False is required for the hash-verified official checkpoint's NumPy scalar metadata",
            "CUDA autocast reconciles open_clip 3.x CoCa fp32 token activations with fp16 attention weights",
        ],
    }
    metadata = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "method": method,
        "source_revision": source_revision(LOKI_SOURCE),
        "official_source_sha256": {
            "utils.py": sha256_file(LOKI_SOURCE / "src/loki/utils.py"),
            "preprocess.py": sha256_file(LOKI_SOURCE / "src/loki/preprocess.py"),
            "predex.py": sha256_file(LOKI_SOURCE / "src/loki/predex.py"),
        },
        "official_checkpoint": {
            "path": str(args.checkpoint.resolve()),
            "sha256": cache_metadata["checkpoint_sha256"],
        },
        "fold": {
            "fold_id": fold.fold_id,
            "train_ids": list(fold.train_ids),
            "val_ids": list(fold.val_ids),
            "test_ids": list(fold.test_ids),
        },
        "best_epoch_selected_by_validation_only": 0,
        "best_validation_mean_gene_pcc": validation_pcc,
        "validation_similarity_diagnostics": val_diagnostics,
        "test_similarity_diagnostics": test_diagnostics,
        "target_parameters": {
            key: value.tolist() if isinstance(value, np.ndarray) else value
            for key, value in parameters.items()
        },
        "coverage_contract": "all_outer_test_slides",
        "patch_manifest_sha256": sha256_file(PATCH_ROOT / "manifest.json"),
        "image_embedding_cache": {
            "path": str(args.cache.resolve()),
            "embedding_sha256": complete_cache["embedding_sha256"],
        },
        "predictions_contain_truth": False,
        "predictions": prediction_rows,
        "wall_time_seconds": time.time() - started,
    }
    metadata_path = staging / "run_metadata.json"
    metadata_path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    complete = {
        "run_metadata_sha256": sha256_file(metadata_path),
        "training_history_sha256": sha256_file(history),
        "checkpoint_sha256": sha256_file(checkpoint),
        "prediction_sha256": {row["sample_id"]: row["sha256"] for row in prediction_rows},
    }
    (staging / "complete.json").write_text(
        json.dumps(complete, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    staging.rename(output)
    return output


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fold", type=int, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    print(json.dumps({"output": str(run_fold(parse_args(argv)))}, sort_keys=True))


if __name__ == "__main__":
    main()
