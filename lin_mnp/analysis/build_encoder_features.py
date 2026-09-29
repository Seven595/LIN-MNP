#!/usr/bin/env python3
"""Extract extra frozen encoders from the shared cSCC RGB224 cache.

ImageNet: ResNet18, ResNet50, DenseNet121, VGG16, EfficientNet-B0.
Pathology: QuiltNet-B-32.
Context is global pooled; local is a 7x7 radial soft pool (radius 72 px).
Existing encoder files are reused.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch
import torch.nn.functional as F
from torchvision.models import (
    DenseNet121_Weights, EfficientNet_B0_Weights, ResNet18_Weights,
    ResNet50_Weights, VGG16_Weights, densenet121, efficientnet_b0,
    resnet18, resnet50, vgg16,
)

from lin_mnp.paths import CACHE_ROOT


RAW = CACHE_ROOT / "cscc/raw_patches"
OUTPUT = CACHE_ROOT / "cscc/encoder_panel"
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def radial_weights(grid: int, stride: float, device: torch.device) -> torch.Tensor:
    center = (grid - 1) / 2.0
    coordinate = (np.arange(grid, dtype=np.float64) - center) * stride
    vertical, horizontal = np.meshgrid(coordinate, coordinate, indexing="ij")
    radius = np.sqrt(horizontal**2 + vertical**2)
    weight = 1.0 / (1.0 + np.exp((radius - 72.0) / 16.0))
    weight /= weight.sum()
    return torch.from_numpy(weight.reshape(-1).astype(np.float32)).to(device)


def normalize(images: np.ndarray, device: torch.device) -> torch.Tensor:
    value = torch.from_numpy(np.ascontiguousarray(images)).to(
        device=device, dtype=torch.float32, non_blocking=True,
    ).permute(0, 3, 1, 2).div_(255.0)
    mean = value.new_tensor(MEAN)[None, :, None, None]
    std = value.new_tensor(STD)[None, :, None, None]
    return value.sub_(mean).div_(std)


class CNNEncoder(torch.nn.Module):
    def __init__(self, kind: str):
        super().__init__()
        self.kind = kind
        if kind == "ResNet18":
            model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
            self.stem = torch.nn.Sequential(
                model.conv1, model.bn1, model.relu, model.maxpool,
                model.layer1, model.layer2, model.layer3, model.layer4,
            )
            self.dim = 512
        elif kind == "ResNet50":
            model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
            self.stem = torch.nn.Sequential(
                model.conv1, model.bn1, model.relu, model.maxpool,
                model.layer1, model.layer2, model.layer3, model.layer4,
            )
            self.dim = 2048
        elif kind == "DenseNet121":
            model = densenet121(weights=DenseNet121_Weights.IMAGENET1K_V1)
            self.stem = model.features
            self.dim = 1024
        elif kind == "VGG16":
            model = vgg16(weights=VGG16_Weights.IMAGENET1K_V1)
            self.stem = model.features
            self.dim = 512
        elif kind == "EfficientNet-B0":
            model = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
            self.stem = model.features
            self.dim = 1280
        else:
            raise ValueError(kind)

    def forward(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        spatial = self.stem(images)
        if self.kind == "DenseNet121":
            spatial = F.relu(spatial, inplace=False)
        context = F.adaptive_avg_pool2d(spatial, 1).flatten(1)
        tokens = spatial.flatten(2).transpose(1, 2)
        return context, tokens


CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)


def normalize_clip(images: np.ndarray, device: torch.device) -> torch.Tensor:
    value = torch.from_numpy(np.ascontiguousarray(images)).to(
        device=device, dtype=torch.float32, non_blocking=True,
    ).permute(0, 3, 1, 2).div_(255.0)
    mean = value.new_tensor(CLIP_MEAN)[None, :, None, None]
    std = value.new_tensor(CLIP_STD)[None, :, None, None]
    return value.sub_(mean).div_(std)


def load_quiltnet(device: torch.device):
    import os
    import open_clip
    os.environ["HF_HUB_OFFLINE"] = "1"
    model, _ = open_clip.create_model_from_pretrained(
        "hf-hub:wisdomik/QuiltNet-B-32", device=device,
    )
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


def quiltnet_forward(model, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    visual = model.visual
    x = visual.conv1(images)
    x = x.reshape(x.shape[0], x.shape[1], -1).permute(0, 2, 1)
    cls = visual.class_embedding.to(x.dtype) + torch.zeros(
        x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device
    )
    x = torch.cat([cls, x], dim=1)
    x = x + visual.positional_embedding.to(x.dtype)
    x = visual.ln_pre(x)
    x = visual.transformer(x.permute(1, 0, 2)).permute(1, 0, 2)
    x = visual.ln_post(x)
    tokens = x[:, 1:]
    if visual.proj is not None:
        tokens = tokens @ visual.proj
    context = model.encode_image(images, normalize=False)
    return context, tokens


def extract_cnn(name, encoder, patches, device, batch_size, weights):
    count, dim = len(patches), encoder.dim
    context = np.empty((count, dim), dtype=np.float32)
    local = np.empty((count, dim), dtype=np.float32)
    encoder.eval().to(device)
    for parameter in encoder.parameters():
        parameter.requires_grad_(False)
    with torch.inference_mode():
        for start in range(0, count, batch_size):
            stop = min(start + batch_size, count)
            images = normalize(patches[start:stop], device)
            pooled, tokens = encoder(images)
            if tokens.shape[1] != 49:
                raise ValueError(f"{name} token grid is {tokens.shape[1]}, expected 49")
            derived = torch.einsum("t,ntd->nd", weights, tokens.float())
            context[start:stop] = pooled.float().cpu().numpy()
            local[start:stop] = derived.cpu().numpy()
    return context, local


def extract_quiltnet(patches, device, batch_size):
    model = load_quiltnet(device)
    count = len(patches)
    visual = model.visual
    dim = int(visual.output_dim) if hasattr(visual, "output_dim") else 512
    context = np.empty((count, dim), dtype=np.float32)
    local = np.empty((count, dim), dtype=np.float32)
    weights = radial_weights(7, 32.0, device)
    with torch.inference_mode():
        for start in range(0, count, batch_size):
            stop = min(start + batch_size, count)
            images = normalize_clip(patches[start:stop], device)
            pooled, tokens = quiltnet_forward(model, images)
            if tokens.shape[1] != 49:
                raise ValueError(f"QuiltNet token grid is {tokens.shape[1]}")
            derived = torch.einsum("t,ntd->nd", weights, tokens.float())
            context[start:stop] = pooled.float().cpu().numpy()
            local[start:stop] = derived.cpu().numpy()
            dim = pooled.shape[1]
    return context, local, dim


def save_pair(output: Path, name: str, context: np.ndarray, local: np.ndarray, meta: dict):
    context_path = output / f"{name}_context_float32.npy"
    local_path = output / f"{name}_local_float32.npy"
    np.save(context_path, context)
    np.save(local_path, local)
    return {
        **meta,
        "context_file": context_path.name,
        "local_file": local_path.name,
        "context_sha256": sha256(context_path),
        "local_sha256": sha256(local_path),
        "shape": list(context.shape),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:5")
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise SystemExit("extraction requires CUDA")
    raw = json.loads((RAW / "manifest.json").read_text())
    if not raw["target_free"] or raw["network_input_px"] != 224:
        raise ValueError("raw cache contract failure")
    patches = np.load(RAW / raw["patch_file"], mmap_mode="r")
    if patches.shape != (int(raw["patch_count"]), 224, 224, 3):
        raise ValueError("raw patch shape failure")
    OUTPUT.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    weights7 = radial_weights(7, 32.0, device)
    records = {}
    for name in ("ResNet18", "ResNet50", "DenseNet121", "VGG16"):
        context_path = OUTPUT / f"{name}_context_float32.npy"
        local_path = OUTPUT / f"{name}_local_float32.npy"
        if context_path.exists() and local_path.exists():
            context = np.load(context_path, mmap_mode="r")
            print(json.dumps({"reusing": name, "shape": list(context.shape)}), flush=True)
            records[name] = {
                "family": "imagenet_cnn",
                "dim": int(context.shape[1]),
                "local_kind": "conv_7x7_radial_soft_pool",
                "context_file": context_path.name,
                "local_file": local_path.name,
                "context_sha256": sha256(context_path),
                "local_sha256": sha256(local_path),
                "shape": list(context.shape),
            }
            continue
        print(json.dumps({"extracting": name}), flush=True)
        encoder = CNNEncoder(name)
        context, local = extract_cnn(
            name, encoder, patches, device, args.batch_size, weights7
        )
        records[name] = save_pair(OUTPUT, name, context, local, {
            "family": "imagenet_cnn",
            "dim": int(context.shape[1]),
            "local_kind": "conv_7x7_radial_soft_pool",
        })
        del encoder
        torch.cuda.empty_cache()
    quilt_context = OUTPUT / "QuiltNet-B-32_context_float32.npy"
    quilt_local = OUTPUT / "QuiltNet-B-32_local_float32.npy"
    if quilt_context.exists() and quilt_local.exists():
        context = np.load(quilt_context, mmap_mode="r")
        print(json.dumps({"reusing": "QuiltNet-B-32", "shape": list(context.shape)}), flush=True)
        records["QuiltNet-B-32"] = {
            "family": "pathology_clip",
            "dim": int(context.shape[1]),
            "local_kind": "vit_b32_token7x7_radial_soft_pool",
            "context_file": quilt_context.name,
            "local_file": quilt_local.name,
            "context_sha256": sha256(quilt_context),
            "local_sha256": sha256(quilt_local),
            "shape": list(context.shape),
        }
    else:
        print(json.dumps({"extracting": "QuiltNet-B-32"}), flush=True)
        context, local, dim = extract_quiltnet(patches, device, args.batch_size)
        records["QuiltNet-B-32"] = save_pair(OUTPUT, "QuiltNet-B-32", context, local, {
            "family": "pathology_clip",
            "dim": int(dim),
            "local_kind": "vit_b32_token7x7_radial_soft_pool",
        })
    manifest = {
        "schema_version": 1,
        "artifact_id": "cscc_encoder_panel",
        "target_free": True,
        "expression_values_accessed": False,
        "raw_patch_cache_manifest_sha256": sha256(RAW / "manifest.json"),
        "patch_count": int(raw["patch_count"]),
        "encoders": records,
        "builder_sha256": sha256(Path(__file__)),
        "wall_seconds": time.perf_counter() - started,
    }
    (OUTPUT / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()
