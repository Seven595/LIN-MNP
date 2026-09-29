#!/usr/bin/env python3
"""Assemble the source-data-backed cSCC panels used in Supplementary Figure S1."""

from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image, ImageChops

from lin_mnp.paths import ARCHIVE_ROOT, RESULTS_ROOT

PANEL_DIR = ARCHIVE_ROOT / "figures/benchmark/panels"
OUTPUT = RESULTS_ROOT / "figures/supplementary_benchmark_endpoints.png"

PANELS = {
    "left": PANEL_DIR / "benchmark_fig2d_cscc_strict.png",
    "upper_right": PANEL_DIR / "benchmark_fig3a_cscc_strict.png",
    "lower_right": PANEL_DIR / "benchmark_fig3d_cscc_strict.png",
}


def trim_white(image: Image.Image) -> Image.Image:
    """Remove uniform white margins while retaining a small safety border."""

    rgb = image.convert("RGB")
    background = Image.new("RGB", rgb.size, "white")
    difference = ImageChops.difference(rgb, background)
    box = difference.getbbox()
    if box is None:
        return rgb
    left, upper, right, lower = box
    pad = 12
    return rgb.crop(
        (
            max(0, left - pad),
            max(0, upper - pad),
            min(rgb.width, right + pad),
            min(rgb.height, lower + pad),
        )
    )


def fit(image: Image.Image, width: int, height: int) -> Image.Image:
    """Resize a panel to fit inside a target box without changing its aspect ratio."""

    scale = min(width / image.width, height / image.height)
    size = (round(image.width * scale), round(image.height * scale))
    return image.resize(size, Image.Resampling.LANCZOS)


def paste_centred(canvas: Image.Image, image: Image.Image, box: tuple[int, int, int, int]) -> None:
    left, upper, right, lower = box
    x = left + (right - left - image.width) // 2
    y = upper + (lower - upper - image.height) // 2
    canvas.paste(image, (x, y))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    missing = [str(path) for path in PANELS.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing benchmark source panels: {missing}")

    canvas = Image.new("RGB", (4200, 2100), "white")
    margin = 25
    gap = 25
    split = 2390
    half = (canvas.height - 2 * margin - gap) // 2

    boxes = {
        "left": (margin, margin, split, canvas.height - margin),
        "upper_right": (split + gap, margin, canvas.width - margin, margin + half),
        "lower_right": (
            split + gap,
            margin + half + gap,
            canvas.width - margin,
            canvas.height - margin,
        ),
    }

    for name, path in PANELS.items():
        panel = trim_white(Image.open(path))
        left, upper, right, lower = boxes[name]
        panel = fit(panel, right - left, lower - upper)
        paste_centred(canvas, panel, boxes[name])

    args.output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(args.output, dpi=(300, 300), optimize=True)
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
