#!/usr/bin/env python3
"""Train one spot-level comparator on one official cSCC patient-CV fold.

Supported methods are those whose released architecture consumes independent
spot patches: ST-Net, DeepSpaCE, the benchmark's DeepPT reimplementation and
TCGN.  The script uses the authors' network definition (or the architecture
stated in the benchmark for DeepPT), while the common data adapter enforces the
same fold, spot, barcode and 997-gene identities for every method.
"""

from __future__ import annotations

import argparse
from contextlib import nullcontext
import csv
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import time
from typing import Mapping, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision
import timm


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from lin_mnp.benchmark.benchmark_external import (  # noqa: E402
    PATCH_ROOT,
    SpotPatchDataset,
    apply_target_transform,
    fit_target_transform,
    load_folds,
    load_slides,
)
from lin_mnp.paths import RESULTS_ROOT


DEFAULT_OUTPUT = RESULTS_ROOT / "comparator_reruns"
IMAGENET_MEAN = torch.tensor((0.485, 0.456, 0.406)).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor((0.229, 0.224, 0.225)).view(1, 3, 1, 1)


@dataclass(frozen=True)
class MethodConfig:
    method_id: str
    display_name: str
    source_dir: str | None
    source_kind: str
    target_transform: str
    epochs: int
    batch_size: int
    learning_rate: float
    weight_decay: float
    input_size: int
    early_stop_patience: int | None
    augmentation: str
    optimizer: str


METHODS = {
    "stnet": MethodConfig(
        method_id="stnet",
        display_name="ST-Net",
        source_dir="comparators/methods/ST-Net",
        source_kind="architecture-faithful released-source rerun",
        target_transform="log1p_cpm",
        epochs=50,
        batch_size=32,
        learning_rate=1e-5,
        weight_decay=0.0,
        input_size=224,
        early_stop_patience=None,
        augmentation="random rotations and flips",
        optimizer="Adam",
    ),
    "deepspace": MethodConfig(
        method_id="deepspace",
        display_name="DeepSpaCE",
        source_dir="comparators/methods/DeepSpaCE",
        source_kind="released VGG16 architecture; train-only z-score substitute for unavailable SCTransform",
        target_transform="train_zscore_log1p_cp10k",
        epochs=50,
        batch_size=32,
        learning_rate=1e-4,
        weight_decay=1e-4,
        input_size=224,
        early_stop_patience=5,
        augmentation="flip, rotation and mild colour jitter",
        optimizer="Adam",
    ),
    "tcgn": MethodConfig(
        method_id="tcgn",
        display_name="TCGN",
        source_dir="comparators/methods/TCGN",
        source_kind="released author TCGN class; author-code 224 px executable path",
        target_transform="log1p_cpm",
        epochs=80,
        batch_size=32,
        learning_rate=1e-5,
        weight_decay=0.0,
        input_size=224,
        early_stop_patience=None,
        augmentation="random rotations and flips",
        optimizer="Adam(beta1=0.9,beta2=0.999)",
    ),
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_torch_save(payload: Mapping[str, object], path: Path) -> None:
    """Commit a resumable state without exposing a partial checkpoint."""

    temporary = path.with_suffix(path.suffix + f".tmp-{os.getpid()}")
    torch.save(dict(payload), temporary)
    os.replace(temporary, path)


def source_revision(source_dir: str | None) -> str | None:
    if source_dir is None:
        return None
    return subprocess.check_output(
        ["git", "-C", str(PROJECT_ROOT / source_dir), "rev-parse", "HEAD"], text=True
    ).strip()


def _load_author_tcgn(n_genes: int) -> nn.Module:
    source = PROJECT_ROOT / "comparators/methods/TCGN"
    sys.path.insert(0, str(source))
    # The released model imports memory_profiler only for an unused decorator.
    if importlib.util.find_spec("memory_profiler") is None:
        import types

        stub = types.ModuleType("memory_profiler")
        stub.profile = lambda function: function
        sys.modules["memory_profiler"] = stub
    from model import TCGN  # type: ignore

    # The release computes D^(-1/2) A D^(-1/2) as two dense matrix
    # multiplications, even though D is diagonal.  That is algebraically the
    # elementwise operation below, but the released implementation takes many
    # minutes per batch on CPU.  Patch both module globals used by the author
    # code; this changes only the implementation of the same normalization.
    import build_graph_tensor as author_graph  # type: ignore
    import censnet_block as author_censnet  # type: ignore

    def diagonal_normalize_tensor(
        value: torch.Tensor, plus_one: bool = False
    ) -> torch.Tensor:
        with torch.no_grad():
            if plus_one:
                row_sum = torch.sum(
                    value
                    + torch.diag_embed(
                        torch.ones(
                            value.shape[0],
                            value.shape[1],
                            dtype=value.dtype,
                            device=value.device,
                        )
                    ),
                    dim=-1,
                )
            else:
                row_sum = torch.sum(value, dim=-1)
            inverse_sqrt = row_sum ** (-0.5)
            inverse_sqrt = torch.where(
                torch.isfinite(inverse_sqrt),
                inverse_sqrt,
                torch.zeros_like(inverse_sqrt),
            )
        return (
            inverse_sqrt.unsqueeze(-1)
            * value
            * inverse_sqrt.unsqueeze(-2)
        )

    author_graph.normalize_tensor = diagonal_normalize_tensor
    author_censnet.normalize_tensor = diagonal_normalize_tensor

    # Keep the author's graph construction equations, but execute them on the
    # tensor's device.  The release copies differentiable attention/features
    # to CPU, constructs scipy matrices sample by sample, then copies every
    # object back to CUDA.  A dense incidence tensor is accepted by every
    # downstream `.to_dense()`/matrix operation and is numerically equivalent.
    spatial_cache: dict[tuple[int, str], torch.Tensor] = {}

    def spatial_prior(nodes: int, device: torch.device) -> torch.Tensor:
        key = (nodes, str(device))
        if key not in spatial_cache:
            height = int(nodes ** 0.5)
            indices = torch.arange(nodes, dtype=torch.float32, device=device)
            coordinates = torch.stack((indices // height, indices % height), dim=1)
            distances = torch.cdist(coordinates, coordinates)
            spatial_cache[key] = 1.0 / (distances / 3.0 + 1.0)
        return spatial_cache[key]

    def device_process_feature_float(
        node_feature: torch.Tensor, attention_graph: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, float]:
        topk: float = 4
        batch_size, nodes, width = node_feature.shape
        heads = attention_graph.shape[1]
        graph_mean = attention_graph.mean(dim=1).reshape(batch_size, -1)
        selection_count = int(topk * nodes)
        with torch.no_grad():
            while True:
                threshold = torch.topk(
                    graph_mean, k=selection_count, dim=-1
                ).values[:, selection_count - 1].unsqueeze(-1)
                ties = torch.sum(graph_mean == threshold)
                if int(ties) == batch_size or selection_count >= graph_mean.shape[1]:
                    break
                selection_count += 1
                topk = selection_count / nodes
        selected = graph_mean >= threshold
        sparse_graph = graph_mean.masked_fill(~selected, 0).reshape(
            batch_size, nodes, nodes
        )
        selected = selected.reshape(batch_size, nodes, nodes)

        scale = width // 2
        inner = -2 * torch.matmul(node_feature, node_feature.transpose(2, 1))
        square = torch.sum(node_feature * node_feature, dim=-1, keepdim=True)
        l2 = torch.sqrt((square + inner + square.transpose(2, 1) + scale) / scale)
        l2 = 1 / l2
        l2 = l2.masked_fill(~selected, 0)
        length = torch.sqrt(torch.sum(node_feature * node_feature, dim=-1, keepdim=True))
        normalized = node_feature / length
        cosine = (normalized @ normalized.transpose(-1, -2) + 1) / 2
        cosine = cosine.masked_fill(~selected, 0)
        prior = spatial_prior(nodes, node_feature.device).expand(batch_size, -1, -1)
        prior = prior.masked_fill(~selected, 0)

        mask = l2 != 0
        edge_parts = []
        for head in range(heads):
            value = attention_graph[:, head].masked_fill(~selected, 0)
            edge_parts.append(value[mask].view(batch_size, -1, 1))
        edge_parts.extend(
            [
                sparse_graph[mask].view(batch_size, -1, 1),
                prior[mask].view(batch_size, -1, 1),
                l2[mask].view(batch_size, -1, 1),
                cosine[mask].view(batch_size, -1, 1),
            ]
        )
        edge_features = torch.cat(edge_parts, dim=-1)

        with torch.no_grad():
            edge_index = torch.nonzero(selected).view(batch_size, -1, 3)
            source = edge_index[:, :, 1].unsqueeze(-1) + 1
            target = edge_index[:, :, 2].unsqueeze(-1) + 1
            source_index = source.squeeze(-1)
            target_index = target.squeeze(-1)
            edge_count = edge_index.shape[1]
            # The four released matrix-product comparisons are exactly the
            # statement that two directed edges share at least one endpoint.
            # Direct ID comparisons preserve the same binary adjacency while
            # avoiding four cubic E x E matrix products.
            edge_adjacency = (
                (source_index.unsqueeze(2) == source_index.unsqueeze(1))
                | (target_index.unsqueeze(2) == source_index.unsqueeze(1))
                | (source_index.unsqueeze(2) == target_index.unsqueeze(1))
                | (target_index.unsqueeze(2) == target_index.unsqueeze(1))
            ).to(node_feature.dtype)
            edge_adjacency = diagonal_normalize_tensor(edge_adjacency, plus_one=True)

            incidence = torch.zeros(
                batch_size,
                nodes,
                edge_count,
                dtype=node_feature.dtype,
                device=node_feature.device,
            )
            column = torch.arange(edge_count, device=node_feature.device).expand(batch_size, -1)
            batch = torch.arange(batch_size, device=node_feature.device).unsqueeze(1).expand(-1, edge_count)
            incidence.index_put_((batch, source.squeeze(-1) - 1, column), torch.ones_like(column, dtype=node_feature.dtype), accumulate=True)
            incidence.index_put_((batch, target.squeeze(-1) - 1, column), torch.ones_like(column, dtype=node_feature.dtype), accumulate=True)

        return (
            edge_features,
            edge_adjacency,
            diagonal_normalize_tensor(selected.to(node_feature.dtype)),
            incidence,
            topk,
        )

    def device_process_feature(
        node_feature: torch.Tensor, attention_graph: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, float]:
        # Under AMP the released squared-distance identity can evaluate as
        # ``inf - inf`` for otherwise finite CMT features.  Construct only the
        # graph statistics in FP32, then return their bounded values in the
        # model dtype.  The image encoder and CensNet remain under AMP.
        output_dtype = node_feature.dtype
        device_type = node_feature.device.type
        with torch.autocast(device_type=device_type, enabled=False):
            values = device_process_feature_float(
                node_feature.float(), attention_graph.float()
            )
        tensors = tuple(value.to(output_dtype) for value in values[:4])
        return (*tensors, values[4])

    author_censnet.process_feature = device_process_feature

    # The released edge-layer normalization divides every column by its
    # maximum without guarding all-zero columns.  A valid sparse graph can
    # therefore produce 0/0, poison the whole prediction with NaNs, and make
    # the following batch fail with an empty graph.  Preserve the released
    # equations wherever the denominator is defined and map only undefined
    # all-zero columns to zero.
    def stable_forward_single_batch_float(
        self: nn.Module,
        H_v: torch.Tensor,
        H_e: torch.Tensor,
        adj_e: torch.Tensor,
        adj_v: torch.Tensor,
        T: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.node_layer:
            multiplier1 = (
                torch.mm(T, torch.diag((H_e @ self.p.t()).t()[0]))
                @ T.to_dense().t()
            )
            mask1 = torch.eye(multiplier1.shape[0], device=multiplier1.device)
            M1 = mask1 + (1.0 - mask1) * multiplier1
            adjusted_A = torch.mul(M1, adj_v)
            output = torch.mm(adjusted_A, torch.mm(H_v, self.weight))
            ret = self.activation(output)
            if self.bias is not None:
                ret = output + self.bias
            return ret, H_e

        multiplier2 = (
            torch.spmm(T.t(), torch.diag((H_v @ self.p.t()).t()[0]))
            @ T.to_dense()
        )
        mask2 = torch.eye(multiplier2.shape[0], device=multiplier2.device)
        M3 = mask2 + (1.0 - mask2) * multiplier2
        adjusted_A = torch.mul(M3, adj_e)
        maximum = adjusted_A.max(0, keepdim=True)[0]
        defined = maximum != 0
        safe_maximum = torch.where(defined, maximum, torch.ones_like(maximum))
        normalized_adjusted_A = adjusted_A / safe_maximum
        normalized_adjusted_A = normalized_adjusted_A.masked_fill(~defined, 0)
        output = torch.mm(normalized_adjusted_A, torch.mm(H_e, self.weight))
        ret = self.activation(output)
        if self.bias is not None:
            ret = output + self.bias
        return H_v, ret

    def stable_forward_single_batch(
        self: nn.Module,
        H_v: torch.Tensor,
        H_e: torch.Tensor,
        adj_e: torch.Tensor,
        adj_v: torch.Tensor,
        T: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # CensNet repeatedly multiplies learned (unbounded) node/edge gates by
        # dense adjacency matrices.  Keeping these small graph layers in FP32
        # prevents seed-dependent half overflow after an optimizer update.
        output_dtype = H_v.dtype
        with torch.autocast(device_type=H_v.device.type, enabled=False):
            left, right = stable_forward_single_batch_float(
                self, H_v.float(), H_e.float(), adj_e.float(), adj_v.float(), T.float()
            )
        return left.to(output_dtype), right.to(output_dtype)

    author_censnet.GraphConvolution.forward_single_batch = stable_forward_single_batch

    model = TCGN(num_classes=n_genes, img_size=224)
    pretrained = source / "pretrained/cmt_tiny.pth"
    if pretrained.is_file():
        state = torch.load(pretrained, map_location="cpu")
        model.load_state_dict(state, strict=False)
    return model


def build_model(method: str, n_genes: int) -> nn.Module:
    if method == "stnet":
        model = timm.create_model(
            "densenet121.tv_in1k", pretrained=True, num_classes=n_genes
        )
        # This is an explicit detail of the released ST-Net implementation.
        nn.init.zeros_(model.get_classifier().weight)
        nn.init.zeros_(model.get_classifier().bias)
        return model
    if method == "deepspace":
        model = torchvision.models.vgg16(
            weights=torchvision.models.VGG16_Weights.IMAGENET1K_V1,
        )
        model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, n_genes)
        return model
    if method == "tcgn":
        return _load_author_tcgn(n_genes)
    raise ValueError(f"unknown method: {method}")


def preprocess(
    patches: torch.Tensor,
    *,
    device: torch.device,
    input_size: int,
    augmentation: str | None,
) -> torch.Tensor:
    image = patches.to(device=device, non_blocking=True).permute(0, 3, 1, 2)
    image = image.to(dtype=torch.float32).div_(255.0)
    if input_size != image.shape[-1]:
        image = F.interpolate(
            image, size=(input_size, input_size), mode="bilinear", align_corners=False,
            antialias=True,
        )
    if augmentation:
        if "rotation" in augmentation:
            image = torch.rot90(image, random.randrange(4), dims=(-2, -1))
        if "flip" in augmentation and random.random() < 0.5:
            image = torch.flip(image, dims=(-1,))
        if "flip" in augmentation and random.random() < 0.5:
            image = torch.flip(image, dims=(-2,))
        if "colour" in augmentation:
            gain = 0.9 + 0.2 * random.random()
            bias = -0.05 + 0.1 * random.random()
            image = (image * gain + bias).clamp_(0.0, 1.0)
    mean = IMAGENET_MEAN.to(device)
    std = IMAGENET_STD.to(device)
    return (image - mean) / std


def forward_prediction(
    method: str, model: nn.Module, image: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor | None]:
    output = model(image)
    return output, None


def mean_slide_gene_pcc(
    predictions: Mapping[str, np.ndarray],
    targets: Mapping[str, np.ndarray],
) -> float:
    slide_gene = []
    for sample_id in sorted(predictions):
        pred = np.asarray(predictions[sample_id], dtype=np.float64)
        true = np.asarray(targets[sample_id], dtype=np.float64)
        pred = pred - pred.mean(axis=0, keepdims=True)
        true = true - true.mean(axis=0, keepdims=True)
        numerator = np.sum(pred * true, axis=0)
        denominator = np.sqrt(np.sum(pred * pred, axis=0) * np.sum(true * true, axis=0))
        corr = np.divide(
            numerator,
            denominator,
            out=np.full_like(numerator, np.nan),
            where=denominator > 0,
        )
        slide_gene.append(corr)
    return float(np.nanmean(np.nanmean(np.stack(slide_gene), axis=0)))


@torch.no_grad()
def predict_dataset(
    method: str,
    model: nn.Module,
    loader: DataLoader,
    sample_ids: Sequence[str],
    slide_sizes: Mapping[str, int],
    n_genes: int,
    device: torch.device,
    input_size: int,
) -> dict[str, np.ndarray]:
    model.eval()
    result = {
        sample_id: np.empty((slide_sizes[sample_id], n_genes), dtype=np.float32)
        for sample_id in sample_ids
    }
    for patches, _, slide_indices, local_indices in loader:
        image = preprocess(
            patches, device=device, input_size=input_size, augmentation=None
        )
        prediction, _ = forward_prediction(method, model, image)
        prediction = prediction.detach().cpu().numpy().astype(np.float32)
        for row, (slide_index, local_index) in enumerate(
            zip(slide_indices.numpy(), local_indices.numpy())
        ):
            result[sample_ids[int(slide_index)]][int(local_index)] = prediction[row]
    return result


def train_fold(args: argparse.Namespace) -> Path:
    config = METHODS[args.method]
    folds = load_folds()
    fold = folds[args.fold]
    slides, genes = load_slides()
    target_parameters = fit_target_transform(
        slides, fold.train_ids, config.target_transform
    )
    train_data = SpotPatchDataset(slides, fold.train_ids, target_parameters)
    val_data = SpotPatchDataset(slides, fold.val_ids, target_parameters)
    test_data = SpotPatchDataset(slides, fold.test_ids, target_parameters)
    epochs = config.epochs if args.max_epochs is None else args.max_epochs
    fixed_final_epoch = (
        os.environ.get("LIN_MNP_CHECKPOINT_MODE") == "fixed_final_epoch"
    )

    fold_output = (args.output / config.method_id / f"fold{fold.fold_id}").resolve()
    if (fold_output / "complete.json").is_file():
        return fold_output
    if fold_output.exists():
        raise FileExistsError(f"incomplete fold output already exists: {fold_output}")
    staging = fold_output.with_name(f"{fold_output.name}.tmp-{os.getpid()}")
    staging.mkdir(parents=True, exist_ok=False)

    resume = None
    if args.resume_state is not None:
        if args.workers != 0:
            raise ValueError("exact resume requires --workers 0")
        resume = torch.load(args.resume_state, map_location="cpu", weights_only=False)
        if (
            resume.get("method_id") != config.method_id
            or int(resume.get("fold_id", -1)) != fold.fold_id
            or int(resume.get("seed", -1)) != args.seed + fold.fold_id
        ):
            raise ValueError("resume state differs from method/fold/seed contract")

    random.seed(args.seed + fold.fold_id)
    np.random.seed(args.seed + fold.fold_id)
    torch.manual_seed(args.seed + fold.fold_id)
    torch.cuda.manual_seed_all(args.seed + fold.fold_id)
    torch.backends.cudnn.benchmark = True
    device = torch.device(args.device)
    model = build_model(config.method_id, len(genes)).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
        betas=(0.9, 0.999),
    )
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp)

    loader_options = {
        "batch_size": config.batch_size,
        "num_workers": args.workers,
        "pin_memory": True,
        "persistent_workers": args.workers > 0,
    }
    train_loader = DataLoader(train_data, shuffle=True, drop_last=False, **loader_options)
    val_loader = DataLoader(val_data, shuffle=False, drop_last=False, **loader_options)
    test_loader = DataLoader(test_data, shuffle=False, drop_last=False, **loader_options)
    slide_sizes = {name: len(slides[name].barcodes) for name in slides}
    val_targets = {
        name: apply_target_transform(slides[name], target_parameters) for name in fold.val_ids
    }

    best_score = float("-inf")
    best_epoch = -1
    best_model_state = None
    no_improvement = 0
    history = []
    start_epoch = 1
    elapsed_offset = 0.0
    checkpoint_path = staging / "best_checkpoint.pt"
    latest_state_path = staging / "latest_training_state.pt"
    if resume is not None:
        model.load_state_dict(resume["model_state"])
        optimizer.load_state_dict(resume["optimizer_state"])
        scaler.load_state_dict(resume["scaler_state"])
        best_score = float(resume["best_score"])
        best_epoch = int(resume["best_epoch"])
        best_model_state = resume["best_model_state"]
        no_improvement = int(resume["no_improvement"])
        history = list(resume["history"])
        start_epoch = int(resume["completed_epoch"]) + 1
        elapsed_offset = float(resume["elapsed_seconds"])
        random.setstate(resume["python_random_state"])
        np.random.set_state(resume["numpy_random_state"])
        torch.set_rng_state(resume["torch_random_state"])
        if device.type == "cuda":
            torch.cuda.set_rng_state_all(resume["cuda_random_states"])
        torch.save(
            {
                "model_state": best_model_state,
                "epoch": best_epoch,
                "validation_mean_gene_pcc": best_score,
                "method_config": asdict(config),
            },
            checkpoint_path,
        )
    start_time = time.time()
    for epoch in range(start_epoch, epochs + 1):
        model.train()
        loss_sum = torch.zeros((), device=device)
        sample_count = 0
        for patches, target, _, _ in train_loader:
            image = preprocess(
                patches,
                device=device,
                input_size=config.input_size,
                augmentation=config.augmentation,
            )
            target = target.to(device=device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            amp_context = (
                torch.autocast(device_type="cuda", dtype=torch.float16)
                if args.amp
                else nullcontext()
            )
            with amp_context:
                prediction, reconstruction_loss = forward_prediction(
                    config.method_id, model, image
                )
                loss = F.mse_loss(prediction, target)
                if reconstruction_loss is not None:
                    loss = loss + reconstruction_loss
            if not torch.isfinite(loss):
                raise FloatingPointError(
                    f"{config.method_id} produced a non-finite training loss "
                    f"at fold={fold.fold_id}, epoch={epoch}"
                )
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            loss_sum += loss.detach() * len(target)
            sample_count += len(target)

        val_predictions = predict_dataset(
            config.method_id,
            model,
            val_loader,
            fold.val_ids,
            slide_sizes,
            len(genes),
            device,
            config.input_size,
        )
        val_score = mean_slide_gene_pcc(val_predictions, val_targets)
        improved = fixed_final_epoch or (
            np.isfinite(val_score) and val_score > best_score
        )
        if improved:
            best_score = val_score
            best_epoch = epoch
            best_model_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
            no_improvement = 0
            torch.save(
                {
                    "model_state": best_model_state,
                    "epoch": epoch,
                    "validation_mean_gene_pcc": val_score,
                    "method_config": asdict(config),
                },
                checkpoint_path,
            )
        else:
            no_improvement += 1
        record = {
            "epoch": epoch,
            "train_mse_objective": float(loss_sum.cpu()) / max(sample_count, 1),
            "validation_mean_gene_pcc": val_score,
            "best_validation_mean_gene_pcc": best_score,
            "elapsed_seconds": elapsed_offset + time.time() - start_time,
        }
        history.append(record)
        if best_model_state is None:
            raise RuntimeError("no finite validation checkpoint was produced")
        if not args.compact:
            atomic_torch_save(
                {
                "schema_version": 1,
                "method_id": config.method_id,
                "fold_id": fold.fold_id,
                "seed": args.seed + fold.fold_id,
                "completed_epoch": epoch,
                "elapsed_seconds": record["elapsed_seconds"],
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "scaler_state": scaler.state_dict(),
                "best_score": best_score,
                "best_epoch": best_epoch,
                "best_model_state": best_model_state,
                "no_improvement": no_improvement,
                "history": history,
                "python_random_state": random.getstate(),
                "numpy_random_state": np.random.get_state(),
                "torch_random_state": torch.get_rng_state(),
                "cuda_random_states": (
                    torch.cuda.get_rng_state_all() if device.type == "cuda" else []
                ),
                },
                latest_state_path,
            )
        print(json.dumps(record, sort_keys=True), flush=True)
        if (
            not fixed_final_epoch
            and
            config.early_stop_patience is not None
            and no_improvement >= config.early_stop_patience
        ):
            break

    if best_epoch < 0:
        raise RuntimeError("no finite validation PCC checkpoint was produced")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state"])
    test_predictions = predict_dataset(
        config.method_id,
        model,
        test_loader,
        fold.test_ids,
        slide_sizes,
        len(genes),
        device,
        config.input_size,
    )

    with (staging / "training_history.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)
    prediction_dir = staging / "predictions"
    prediction_dir.mkdir()
    prediction_rows = []
    for sample_id in fold.test_ids:
        path = prediction_dir / f"{sample_id}.npz"
        np.savez_compressed(
            path,
            prediction=test_predictions[sample_id],
            genes=np.asarray(genes),
            barcodes=np.asarray(slides[sample_id].barcodes),
            sample_id=np.asarray(sample_id),
            patient_id=np.asarray(slides[sample_id].patient_id),
            fold_id=np.asarray(fold.fold_id, dtype=np.int32),
        )
        prediction_rows.append(
            {
                "sample_id": sample_id,
                "patient_id": slides[sample_id].patient_id,
                "path": str(path.relative_to(staging)),
                "sha256": sha256_file(path),
                "n_spots": len(slides[sample_id].barcodes),
                "n_genes": len(genes),
            }
        )

    metadata = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "method": asdict(config),
        "source_revision": source_revision(config.source_dir),
        "fold": asdict(fold),
        "seed": args.seed + fold.fold_id,
        "device": str(device),
        "mixed_precision": bool(args.amp),
        "executed_epochs": epochs,
        "formal_paper_epoch_count": args.max_epochs is None,
        "runtime_compatibility_patches": (
            [
                {
                    "name": "tcgn_diagonal_normalization",
                    "scope": "performance plus disclosed zero-degree stability correction",
                    "equivalence": (
                        "D^-1/2 A D^-1/2 evaluated by elementwise diagonal scaling; "
                        "defined positive degrees are unchanged and non-finite inverse "
                        "degrees for isolated nodes map to zero"
                    ),
                    "bit_exact_on_positive_degree_nodes": True,
                },
                {
                    "name": "tcgn_device_graph_construction_and_aligned_incidence",
                    "scope": "performance plus disclosed sample-alignment correction",
                    "equivalence": (
                        "released graph equations evaluated as batched device tensors; "
                        "dense incidence preserves input sample order instead of the "
                        "released helper's batch-order reversal"
                    ),
                    "bit_exact_to_released_batch_order_bug": False,
                },
                {
                    "name": "tcgn_shared_endpoint_adjacency",
                    "scope": "performance-only",
                    "equivalence": "four released edge-endpoint matrix-product equalities replaced by their exact endpoint-ID equality union",
                },
                {
                    "name": "tcgn_zero_column_normalization_guard",
                    "scope": "disclosed numerical-stability correction",
                    "equivalence": (
                        "identical adjusted_A/column_max equation for nonzero "
                        "columns; released undefined 0/0 columns map to zero"
                    ),
                    "bit_exact_on_defined_columns": True,
                },
                {
                    "name": "tcgn_fp32_graph_path_under_amp",
                    "scope": "disclosed numerical-stability correction",
                    "equivalence": (
                        "released distance, cosine, normalization and CensNet "
                        "dense graph equations evaluated in FP32 to prevent "
                        "half-precision inf-inf/overflow; graph outputs are "
                        "cast back to the model autocast dtype"
                    ),
                    "image_encoder_and_censnet_amp_unchanged": True,
                },
            ]
            if config.method_id == "tcgn"
            else []
        ),
        "best_epoch_selected_by_validation_only": (
            best_epoch
            if os.environ.get("LIN_MNP_CHECKPOINT_SELECTION_SCOPE")
            == "inner_validation"
            else None
        ),
        "best_epoch_selected_by_outer_test": (
            best_epoch
            if os.environ.get("LIN_MNP_CHECKPOINT_SELECTION_SCOPE")
            == "outer_test"
            else None
        ),
        "fixed_refit_epoch": best_epoch if fixed_final_epoch else None,
        "checkpoint_selection_scope": (
            "fixed_epoch_selected_on_inner_validation_then_refit_on_all_outer_train"
            if fixed_final_epoch
            else os.environ.get(
                "LIN_MNP_CHECKPOINT_SELECTION_SCOPE",
                (
                    "outer_test_HESpotEx_protocol"
                    if os.environ.get("LIN_MNP_EXTERNAL_CONTRACT")
                    else "inner_validation"
                ),
            )
        ),
        "best_validation_mean_gene_pcc": best_score,
        "target_parameters": {
            key: value.tolist() if isinstance(value, np.ndarray) else value
            for key, value in target_parameters.items()
        },
        "patch_manifest_sha256": sha256_file(PATCH_ROOT / "manifest.json"),
        "predictions_contain_truth": False,
        "predictions": prediction_rows,
        "wall_time_seconds": elapsed_offset + time.time() - start_time,
        "resumed_from": (
            str(args.resume_state.resolve()) if args.resume_state is not None else None
        ),
    }
    (staging / "run_metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    complete = {
        "run_metadata_sha256": sha256_file(staging / "run_metadata.json"),
        "training_history_sha256": sha256_file(staging / "training_history.csv"),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "prediction_sha256": {row["sample_id"]: row["sha256"] for row in prediction_rows},
    }
    if latest_state_path.is_file():
        complete["latest_training_state_sha256"] = sha256_file(latest_state_path)
    (staging / "complete.json").write_text(
        json.dumps(complete, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    fold_output.parent.mkdir(parents=True, exist_ok=True)
    staging.rename(fold_output)
    return fold_output


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=sorted(METHODS), required=True)
    parser.add_argument("--fold", type=int, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--max-epochs", type=int)
    parser.add_argument("--compact", action="store_true")
    parser.add_argument("--resume-state", type=Path)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    output = train_fold(parse_args(argv))
    print(json.dumps({"output": str(output)}, sort_keys=True))


if __name__ == "__main__":
    main()
