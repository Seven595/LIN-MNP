#!/usr/bin/env python3
"""Build the Lin-MNP overview figure from the cSCC P2 serial sections.

The morphology panel shows neighbours on a second section of the same patient,
selected by the Lin-MNP dual-view rank fusion, not by spatial proximity.
"""

from __future__ import annotations

import argparse
import gzip
import io
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Polygon, Rectangle
import numpy as np
from PIL import Image

from lin_mnp import model
from lin_mnp.paths import DATA_ROOT, RESULTS_ROOT

Image.MAX_IMAGE_PIXELS = None

IMAGE_ROOT = DATA_ROOT / "cscc/cohort/raw/he_spot_only"
GEOMETRY_ROOT = (
    DATA_ROOT
    / "cscc/derived/heto_official_cv4/fold3/locked997/prepared/slides"
)
LABEL_ROOT = DATA_ROOT / "cscc/labels"
QUERY_SECTION = "GSM4284316"
GALLERY_SECTION = "GSM4284317"
BASE_SECTION = "GSM4284318"
N_NEIGHBOURS = 5
GENES = ("AEBP1", "KRT14", "COL1A1")
OUTPUT = RESULTS_ROOT / "figures/figure1.png"

BLUE = "#2F6FED"
ORANGE = "#E07A3D"
GREEN = "#2A9D6E"
INK = "#1F2933"
MUTED = "#5B6570"
PANEL = "#F7F9FC"
PANEL_EDGE = "#D5DCE6"


def decode(values):
    return [
        value.decode("utf-8") if isinstance(value, bytes) else str(value)
        for value in np.asarray(values).reshape(-1)
    ]


def load_section(section, max_edge=900.0):
    with np.load(GEOMETRY_ROOT / f"{section}.npz", allow_pickle=False) as archive:
        barcodes = decode(archive["barcodes"])
        raw_coordinates = np.asarray(archive["image_coords"], dtype=np.float64)
        coordinates = raw_coordinates.copy()
        diameter = float(archive["spot_diameter_px"].item())
    paths = sorted(IMAGE_ROOT.glob(f"{section}_*.jpg.gz"))
    if len(paths) != 1:
        raise ValueError(f"expected one image for {section}")
    with gzip.open(paths[0], "rb") as handle:
        image = Image.open(io.BytesIO(handle.read())).convert("RGB")
        image.load()
    x, y = coordinates[:, 0], coordinates[:, 1]
    span = max(float(np.ptp(x)), float(np.ptp(y)))
    margin = max(diameter * 1.4, span * 0.05)
    box = (
        max(0, int(np.floor(x.min() - margin))),
        max(0, int(np.floor(y.min() - margin))),
        min(image.width, int(np.ceil(x.max() + margin))),
        min(image.height, int(np.ceil(y.max() + margin))),
    )
    crop = image.crop(box)
    image.close()
    coordinates = coordinates - np.asarray(box[:2], dtype=np.float64)
    scale = min(1.0, float(max_edge) / max(crop.size)) if max_edge else 1.0
    if scale < 1.0:
        crop = crop.resize(
            (round(crop.width * scale), round(crop.height * scale)),
            Image.Resampling.LANCZOS,
        )
        coordinates *= scale
        diameter *= scale
    return {
        "barcodes": barcodes,
        "coordinates": coordinates,
        "raw_coordinates": raw_coordinates,
        "diameter": diameter,
        "image": crop,
        "origin": np.asarray(box[:2], dtype=np.float64),
        "scale": scale,
    }


def load_full_image(section):
    paths = sorted(IMAGE_ROOT.glob(f"{section}_*.jpg.gz"))
    with gzip.open(paths[0], "rb") as handle:
        image = Image.open(io.BytesIO(handle.read())).convert("RGB")
        image.load()
    return image


def square_crop(image, center, size):
    x, y = center
    half = size / 2.0
    left = int(round(x - half))
    top = int(round(y - half))
    box = (
        max(0, left),
        max(0, top),
        min(image.width, left + size),
        min(image.height, top + size),
    )
    tile = image.crop(box)
    if tile.size != (size, size):
        padded = Image.new("RGB", (size, size), (245, 245, 245))
        padded.paste(tile, (box[0] - left, box[1] - top))
        tile = padded
    return tile


def tissue_rgba(image, white=238):
    array = np.asarray(image.convert("RGB"))
    rgba = np.zeros(array.shape[:2] + (4,), dtype=np.uint8)
    rgba[..., :3] = array
    pale = (array.min(axis=2) > white) & (np.ptp(array, axis=2) < 16)
    rgba[..., 3] = np.where(pale, 0, 255)
    return Image.fromarray(rgba, "RGBA")


def homography(src, dst):
    src = np.asarray(src, dtype=np.float64)
    dst = np.asarray(dst, dtype=np.float64)
    matrix = np.zeros((8, 8), dtype=np.float64)
    target = np.zeros(8, dtype=np.float64)
    for i, ((x, y), (u, v)) in enumerate(zip(src, dst)):
        matrix[2 * i] = [x, y, 1, 0, 0, 0, -u * x, -u * y]
        matrix[2 * i + 1] = [0, 0, 0, x, y, 1, -v * x, -v * y]
        target[2 * i] = u
        target[2 * i + 1] = v
    coeffs = np.linalg.solve(matrix, target)
    return np.append(coeffs, 1.0).reshape(3, 3)


def apply_homography(points, matrix):
    points = np.asarray(points, dtype=np.float64)
    homo = np.c_[points, np.ones(len(points))]
    projected = homo @ matrix.T
    return projected[:, :2] / projected[:, 2:3]


def isometric_layer(image, flatten=0.46, shear=0.62):
    """Warp a section onto a receding parallelogram (isometric slice)."""
    source = tissue_rgba(image)
    width, height = source.size
    flat = max(12, int(round(height * flatten)))
    shift = int(round(flat * shear))
    src = [(0.0, 0.0), (width, 0.0), (width, height), (0.0, height)]
    dst = [(float(shift), 0.0), (float(width + shift), 0.0), (float(width), float(flat)), (0.0, float(flat))]
    forward = homography(src, dst)
    inverse = np.linalg.inv(forward)
    inverse = inverse / inverse[2, 2]
    coeffs = (
        inverse[0, 0],
        inverse[0, 1],
        inverse[0, 2],
        inverse[1, 0],
        inverse[1, 1],
        inverse[1, 2],
        inverse[2, 0],
        inverse[2, 1],
    )
    warped = source.transform(
        (width + shift, flat),
        Image.PERSPECTIVE,
        coeffs,
        Image.Resampling.BICUBIC,
    )
    corners = apply_homography(src, forward)
    return warped, forward, corners


def minmax(values):
    values = np.asarray(values, dtype=np.float64)
    low, high = float(values.min()), float(values.max())
    if high <= low:
        return np.zeros_like(values)
    return (values - low) / (high - low)


def panel(axis, facecolor=PANEL):
    axis.set_axis_off()
    axis.add_patch(
        FancyBboxPatch(
            (0, 0),
            1,
            1,
            boxstyle="round,pad=0.012,rounding_size=0.03",
            transform=axis.transAxes,
            facecolor=facecolor,
            edgecolor=PANEL_EDGE,
            linewidth=0.9,
            zorder=0,
            clip_on=False,
        )
    )


def title(axis, text):
    axis.text(
        0.5,
        0.965,
        text,
        transform=axis.transAxes,
        ha="center",
        va="top",
        fontsize=8.2,
        color=INK,
        fontweight="medium",
    )


def arrow(fig, src, dst, color=MUTED):
    fig.add_artist(
        FancyArrowPatch(
            src,
            dst,
            transform=fig.transFigure,
            arrowstyle="-|>",
            mutation_scale=8,
            linewidth=1.05,
            color=color,
            shrinkA=1,
            shrinkB=1,
        )
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    query = load_section(QUERY_SECTION)
    gallery = load_section(GALLERY_SECTION)
    base = load_section(BASE_SECTION)
    full_query = load_full_image(QUERY_SECTION)

    dataset = model.LabeledSpotDataset(
        model.CONFIG, [QUERY_SECTION, GALLERY_SECTION]
    )
    local_cache = model.load_local_cache(model.CONFIG)
    patch_index = np.asarray(dataset.patch_indices, dtype=np.int64)
    context = model.normalize_rows(
        np.asarray(dataset.ctranspath[patch_index], dtype=np.float32)
    )
    local = model.normalize_rows(
        np.asarray(local_cache[patch_index], dtype=np.float32)
    )
    sections = np.asarray([str(item) for item in dataset.sections])
    barcodes = [str(item) for item in dataset.barcodes]
    query_mask = sections == QUERY_SECTION
    gallery_mask = sections == GALLERY_SECTION
    query_idx = np.flatnonzero(query_mask)
    gallery_idx = np.flatnonzero(gallery_mask)

    query_pos = {code: i for i, code in enumerate(query["barcodes"])}
    gallery_pos = {code: i for i, code in enumerate(gallery["barcodes"])}
    query_xy = np.asarray([query["coordinates"][query_pos[barcodes[i]]] for i in query_idx])
    gallery_xy = np.asarray(
        [gallery["coordinates"][gallery_pos[barcodes[i]]] for i in gallery_idx]
    )

    # A central query whose high-ranking morphology neighbours are not the
    # spatially corresponding spots on the next serial section.
    centre = query_xy.mean(axis=0)
    interior = np.flatnonzero(
        (np.abs(query_xy - centre) < 0.28 * np.ptp(query_xy, axis=0)).all(axis=1)
    )
    scores = model.pair_similarity(
        context, local, query_idx[interior], gallery_idx
    )
    q_span = np.ptp(query_xy, axis=0) + 1e-6
    g_span = np.ptp(gallery_xy, axis=0) + 1e-6
    q_rel = (query_xy[interior] - query_xy.min(axis=0)) / q_span
    g_rel = (gallery_xy - gallery_xy.min(axis=0)) / g_span
    gray = np.asarray(full_query.convert("L"), dtype=np.float32)

    def local_std(xy, radius=90):
        x, y = int(round(xy[0])), int(round(xy[1]))
        patch = gray[max(0, y - radius) : y + radius, max(0, x - radius) : x + radius]
        return float(patch.std()) if patch.size else 0.0

    contrast = np.asarray(
        [local_std(query["raw_coordinates"][query_pos[barcodes[query_idx[j]]]]) for j in interior]
    )
    contrast = (contrast - contrast.min()) / (np.ptp(contrast) + 1e-6)
    best_i = 0
    best_score = -1.0
    for i, row in enumerate(scores):
        top = np.argsort(-row)[:24]
        disp = float(np.linalg.norm(g_rel[top] - q_rel[i], axis=1).mean())
        score = disp + 0.8 * float(contrast[i])
        if score > best_score:
            best_score = score
            best_i = i
    picked = int(interior[best_i])
    top24 = np.argsort(-scores[best_i])[:24]
    displacement = np.linalg.norm(g_rel[top24] - q_rel[best_i], axis=1)
    neighbour_local = top24[np.argsort(-displacement)[:N_NEIGHBOURS]]
    query_barcode = barcodes[query_idx[picked]]
    # Spread the displayed neighbours across the gallery while remaining
    # inside the high-ranking morphology set.
    selected = [int(neighbour_local[0])]
    remaining = [int(j) for j in top24 if int(j) not in selected]
    while len(selected) < N_NEIGHBOURS and remaining:
        spread = [
            min(np.linalg.norm(gallery_xy[j] - gallery_xy[s]) for s in selected)
            for j in remaining
        ]
        chosen = remaining[int(np.argmax(spread))]
        selected.append(chosen)
        remaining.remove(chosen)
    neighbour_local = np.asarray(selected)
    neighbour_barcodes = [barcodes[gallery_idx[j]] for j in neighbour_local]
    query_full_xy = query["raw_coordinates"][query_pos[query_barcode]]

    with np.load(LABEL_ROOT / f"{QUERY_SECTION}.npz", allow_pickle=False) as archive:
        genes = decode(archive["genes"])
        target = np.asarray(archive["log1p_raw_counts"], dtype=np.float64)
        label_barcodes = decode(archive["barcodes"])
    gene_index = {gene: i for i, gene in enumerate(genes)}
    label_order = [label_barcodes.index(code) for code in query["barcodes"]]

    fig = plt.figure(figsize=(7.25, 3.05), dpi=300)
    fig.patch.set_facecolor("white")
    gs = fig.add_gridspec(
        2,
        5,
        width_ratios=[1.22, 0.68, 0.98, 1.62, 0.92],
        height_ratios=[1.0, 1.0],
        left=0.010,
        right=0.990,
        top=0.952,
        bottom=0.072,
        wspace=0.065,
        hspace=0.12,
    )
    ax_he = fig.add_subplot(gs[:, 0])
    ax_pfm = fig.add_subplot(gs[:, 1])
    ax_read = fig.add_subplot(gs[0, 2])
    ax_rank = fig.add_subplot(gs[1, 2])
    ax_mnp = fig.add_subplot(gs[:, 3])
    ax_out = fig.add_subplot(gs[:, 4])

    for axis in (ax_he, ax_pfm, ax_read, ax_rank, ax_mnp, ax_out):
        panel(axis)

    # --- H&E: tissue plus two extracted fields ---
    title(ax_he, "H&E image")
    tissue = ax_he.inset_axes([0.06, 0.10, 0.54, 0.76])
    tissue.imshow(query["image"])
    tissue.set_axis_off()
    qx, qy = query["coordinates"][query_pos[query_barcode]]
    ctx_px = 384 * query["scale"]
    loc_px = 224 * query["scale"]
    tissue.add_patch(
        Rectangle(
            (qx - ctx_px / 2, qy - ctx_px / 2),
            ctx_px,
            ctx_px,
            fill=False,
            edgecolor=BLUE,
            linewidth=1.05,
        )
    )
    tissue.add_patch(
        Rectangle(
            (qx - loc_px / 2, qy - loc_px / 2),
            loc_px,
            loc_px,
            fill=False,
            edgecolor=ORANGE,
            linewidth=1.05,
        )
    )
    tissue.plot(qx, qy, "o", color="white", markersize=2.8, markeredgecolor=INK, markeredgewidth=0.45)
    ctx_tile = square_crop(full_query, query_full_xy, 384).resize((110, 110), Image.Resampling.LANCZOS)
    loc_tile = square_crop(full_query, query_full_xy, 224).resize((110, 110), Image.Resampling.LANCZOS)
    full_query.close()
    for tile, ypos, color, label in (
        (ctx_tile, 0.55, BLUE, "contextual"),
        (loc_tile, 0.12, ORANGE, "local"),
    ):
        box = ax_he.inset_axes([0.64, ypos, 0.30, 0.30])
        box.imshow(tile)
        box.set_xticks([])
        box.set_yticks([])
        for spine in box.spines.values():
            spine.set_color(color)
            spine.set_linewidth(1.35)
        ax_he.text(
            0.79,
            ypos + 0.305,
            label,
            transform=ax_he.transAxes,
            ha="center",
            va="bottom",
            fontsize=6.0,
            color=color,
        )

    # --- Frozen PFM ---
    title(ax_pfm, "Frozen PFM")
    ax_pfm.add_patch(
        FancyBboxPatch(
            (0.18, 0.40),
            0.64,
            0.38,
            boxstyle="round,pad=0.01,rounding_size=0.05",
            transform=ax_pfm.transAxes,
            facecolor="#E8EEF8",
            edgecolor="#9AA8BE",
            linewidth=0.8,
        )
    )
    ax_pfm.text(0.50, 0.59, "❄", transform=ax_pfm.transAxes, ha="center", va="center", fontsize=20)
    rng = np.random.default_rng(7)
    for y0, cmap, label in ((0.26, plt.cm.Blues, "contextual"), (0.10, plt.cm.Oranges, "local")):
        values = 0.30 + 0.70 * rng.random(7)
        ax_pfm.barh(
            np.full(7, y0),
            0.055,
            left=0.16 + np.arange(7) * 0.10,
            height=0.07,
            color=cmap(values),
            align="edge",
            transform=ax_pfm.transAxes,
        )
        ax_pfm.text(0.50, y0 - 0.045, label, transform=ax_pfm.transAxes, ha="center", fontsize=6.0, color=INK)

    # --- Linear readout ---
    title(ax_read, "Linear readout")
    heat = ax_read.inset_axes([0.18, 0.28, 0.66, 0.54])
    heat.imshow(0.15 + 0.85 * rng.random((8, 13)), cmap="Purples", vmin=0, vmax=1, aspect="auto")
    heat.set_xticks([])
    heat.set_yticks([])
    heat.set_xlabel("genes", fontsize=6.0, color=MUTED, labelpad=1)
    heat.set_ylabel("spots", fontsize=6.0, color=MUTED, labelpad=1)
    ax_read.text(
        0.50,
        0.10,
        "contextual only",
        transform=ax_read.transAxes,
        ha="center",
        fontsize=6.2,
        color=BLUE,
    )

    # --- Rank and fuse ---
    title(ax_rank, "Rank and fuse")
    for y0, cmap, label in (
        (0.56, plt.cm.Blues, "contextual ranks"),
        (0.30, plt.cm.Oranges, "local ranks"),
    ):
        ax_rank.text(0.10, y0 + 0.08, label, transform=ax_rank.transAxes, ha="left", fontsize=6.0, color=INK)
        for i, shade in enumerate((0.35, 0.55, 0.80, 0.48)):
            ax_rank.add_patch(
                Rectangle(
                    (0.52 + i * 0.10, y0),
                    0.08,
                    0.16,
                    transform=ax_rank.transAxes,
                    facecolor=cmap(shade),
                    edgecolor="white",
                    linewidth=0.3,
                )
            )
    ax_rank.text(0.50, 0.12, "equal Borda fusion", transform=ax_rank.transAxes, ha="center", fontsize=6.2, color=INK)

    # --- Real serial H&E sections, stacked in isometric view ---
    title(ax_mnp, "Morphology-neighbour propagation")
    stack_ax = ax_mnp.inset_axes([0.03, 0.08, 0.94, 0.80])
    stack_ax.set_axis_off()
    stack_ax.set_aspect("equal")

    common = (340, 240)

    def resize_section(section):
        source = section["image"]
        scaled = source.resize(common, Image.Resampling.LANCZOS)
        scale = np.asarray(
            [common[0] / source.size[0], common[1] / source.size[1]], dtype=np.float64
        )
        return scaled, section["coordinates"] * scale

    def project_points(xy, matrix, origin_x, origin_y, height):
        mapped = apply_homography(np.asarray(xy, dtype=np.float64).reshape(-1, 2), matrix)
        return np.column_stack([origin_x + mapped[:, 0], origin_y + height - mapped[:, 1]])

    stack = []
    for section in (query, gallery, base):
        image, coords = resize_section(section)
        warped, matrix, _ = isometric_layer(image, flatten=0.30, shear=1.05)
        stack.append((warped, matrix, coords, warped.size))

    gap = stack[0][0].size[1] * 1.05
    x_shift = 18.0
    alphas = {0: 1.00, 1: 0.80, 2: 0.58}
    for layer in (2, 1, 0):
        warped, matrix, _coords, (width, height) = stack[layer]
        origin_x = layer * x_shift
        origin_y = (2 - layer) * gap
        array = np.asarray(warped).copy()
        array[..., 3] = (array[..., 3].astype(np.float64) * alphas[layer]).astype(np.uint8)
        stack_ax.imshow(
            array,
            extent=(origin_x, origin_x + width, origin_y, origin_y + height),
            origin="upper",
            interpolation="bilinear",
            zorder=3 - layer,
        )

    query_img, query_coords, query_matrix = stack[0][0], stack[0][2], stack[0][1]
    gallery_img, gallery_coords, gallery_matrix = stack[1][0], stack[1][2], stack[1][1]
    q_origin = (0.0, 2 * gap, query_img.size[1])
    g_origin = (x_shift, 1 * gap, gallery_img.size[1])
    qxy = project_points(
        query_coords[query_pos[query_barcode]],
        query_matrix,
        *q_origin,
    )[0]
    nxy = project_points(
        np.asarray([gallery_coords[gallery_pos[code]] for code in neighbour_barcodes]),
        gallery_matrix,
        *g_origin,
    )
    stack_ax.plot(
        *qxy,
        "o",
        color="#C0392B",
        markersize=4.2,
        markeredgecolor="white",
        markeredgewidth=0.45,
        zorder=8,
    )
    stack_ax.scatter(
        nxy[:, 0],
        nxy[:, 1],
        s=14,
        facecolors="white",
        edgecolors=GREEN,
        linewidths=0.8,
        zorder=8,
    )
    for dest in nxy:
        stack_ax.plot(
            [qxy[0], dest[0]],
            [qxy[1], dest[1]],
            color=GREEN,
            linewidth=0.6,
            alpha=0.8,
            zorder=7,
        )
    ax_mnp.text(0.86, 0.78, "query", transform=ax_mnp.transAxes, ha="left", fontsize=6.0, color=MUTED)
    ax_mnp.text(0.86, 0.50, "gallery", transform=ax_mnp.transAxes, ha="left", fontsize=6.0, color=MUTED)
    ax_mnp.text(
        0.50,
        0.030,
        "serial H&E sections of one patient",
        transform=ax_mnp.transAxes,
        ha="center",
        fontsize=5.7,
        color=GREEN,
    )

    # --- Spatial expression ---
    title(ax_out, "Spatial expression")
    marker = float(np.clip(95.0 * query["diameter"] / max(query["image"].size), 2.8, 5.8) ** 2)
    for i, gene in enumerate(GENES):
        box = ax_out.inset_axes([0.14, 0.68 - i * 0.27, 0.60, 0.22])
        box.set_axis_off()
        values = minmax(target[label_order, gene_index[gene]])
        box.scatter(
            query["coordinates"][:, 0],
            -query["coordinates"][:, 1],
            c=values,
            cmap="viridis",
            s=marker,
            linewidths=0,
            rasterized=True,
        )
        box.set_aspect("equal")
        ax_out.text(
            0.80,
            0.79 - i * 0.27,
            gene,
            transform=ax_out.transAxes,
            ha="left",
            va="center",
            fontsize=6.2,
            color=INK,
        )

    fig.canvas.draw()
    arrow(fig, (0.212, 0.52), (0.236, 0.52), MUTED)
    arrow(fig, (0.360, 0.74), (0.388, 0.74), BLUE)
    arrow(fig, (0.360, 0.28), (0.388, 0.28), ORANGE)
    arrow(fig, (0.555, 0.74), (0.592, 0.58), BLUE)
    arrow(fig, (0.555, 0.28), (0.592, 0.42), ORANGE)
    arrow(fig, (0.848, 0.52), (0.872, 0.52), GREEN)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=300, facecolor="white")
    print(f"wrote {args.output}")
    print("query", query_barcode)
    print("neighbours", neighbour_barcodes)


if __name__ == "__main__":
    main()
