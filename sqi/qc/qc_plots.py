# sqi/qc/qc_plots.py
from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Sequence, Tuple

from matplotlib.patches import Rectangle

# ---------------------------
# SQI DISTRIBUTIONS
# ---------------------------

def plot_sqi_distribution(
    sqi: Dict[int, float],
    *,
    title: Optional[str] = None,
    ax: Optional[plt.Axes] = None,
    color: str = "tab:blue",
    label: Optional[str] = None,
):
    """
    Plot log10(SQI) distribution across cells.

    sqi: dict {cell_id: sqi_value}
    """
    values = np.array(list(sqi.values()), dtype=float)
    values = values[np.isfinite(values) & (values > 0)]

    if ax is None:
        fig, ax = plt.subplots(figsize=(4, 3))

    ax.hist(np.log10(values), bins=40, alpha=0.7, color=color, label=label)
    ax.axvline(0, linestyle="--", color="k", linewidth=1)

    ax.set_xlabel("log10(SQI)")
    ax.set_ylabel("Cell count")

    if title is not None:
        ax.set_title(title)

    if label is not None:
        ax.legend()

    return ax


def plot_sqi_real_vs_null(
    real_sqi: Dict[int, float],
    null_sqi: Dict[int, float],
    *,
    ax: Optional[plt.Axes] = None,
    title: Optional[str] = None,
):
    """Overlay real vs null log10(SQI) histograms for sanity check."""
    def _to_log(d):
        v = np.array(list(d.values()), dtype=float)
        v = v[np.isfinite(v) & (v > 0)]
        return np.log10(v)

    real_log = _to_log(real_sqi)
    null_log = _to_log(null_sqi)

    if ax is None:
        _, ax = plt.subplots(figsize=(5, 3.5))

    all_vals = np.concatenate([real_log, null_log]) if (len(real_log) + len(null_log)) else np.array([0.0])
    bins = np.linspace(all_vals.min() - 0.3, all_vals.max() + 0.3, 50)

    ax.hist(null_log, bins=bins, alpha=0.55, color="tab:gray", label="null (uniform)")
    ax.hist(real_log, bins=bins, alpha=0.55, color="tab:blue", label="real")
    ax.axvline(0, ls="--", color="k", lw=0.8)
    ax.set_xlabel("log10(SQI)")
    ax.set_ylabel("Cell count")
    ax.legend()
    if title:
        ax.set_title(title)
    return ax


# ---------------------------
# TISSUE OVERVIEW (FIG 1)
# ---------------------------

def plot_tissue_overview(
    mosaic_img: np.ndarray,
    fov_bboxes: Sequence[Tuple[int, int, int, int]],
    fov_labels: Sequence[str],
    *,
    title: Optional[str] = None,
    out_path: Optional[str] = None,
):
    """
    Display mosaic image with FOV bounding boxes highlighted.

    Parameters
    ----------
    mosaic_img  : (H, W) float32 mosaic image at mosaic resolution.
    fov_bboxes  : list of (r0, c0, r1, c1) in mosaic pixel coords.
    fov_labels  : list of FOV id strings (same length as fov_bboxes).
    """
    fig, ax = plt.subplots(figsize=(8, 8))
    nonzero = mosaic_img[mosaic_img > 0]
    if len(nonzero) > 0:
        vmin = np.percentile(nonzero, 1)
        vmax = np.percentile(nonzero, 99)
    else:
        vmin, vmax = 0, 1
    ax.imshow(mosaic_img, cmap="gray", vmin=vmin, vmax=vmax)

    for (r0, c0, r1, c1), label in zip(fov_bboxes, fov_labels):
        rect = Rectangle((c0, r0), c1 - c0, r1 - r0,
                          linewidth=1.5, edgecolor="orange", facecolor="none")
        ax.add_patch(rect)
        ax.text(c0, r0 - 2, label, fontsize=5, color="orange",
                va="bottom", ha="left")

    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title or "Tissue overview")
    fig.tight_layout()

    if out_path:
        fig.savefig(out_path, dpi=200)
        plt.close(fig)
    return fig, ax


# ---------------------------
# CHANNEL PROJECTIONS (FIG 2)
# ---------------------------

def plot_channel_projections_with_spots(
    channel_images: Dict[int, np.ndarray],
    spots_per_channel: Dict[int, np.ndarray],
    fov_id: str,
    out_path: Optional[str] = None,
):
    """
    Per-channel max projection with detected spots overlaid.

    Parameters
    ----------
    channel_images     : {ch_idx: 2D float32 array}
    spots_per_channel  : {ch_idx: (N, 2) array of (row, col)}
    """
    channels = sorted(channel_images.keys())
    n_ch = len(channels)
    if n_ch == 0:
        return None, None

    fig, axes = plt.subplots(n_ch, 2, figsize=(8, 3.5 * n_ch),
                              squeeze=False)

    for i, ch in enumerate(channels):
        img = channel_images[ch]
        img_h, img_w = img.shape[:2]
        vmin = np.percentile(img, 1)
        vmax = np.percentile(img, 99.5)

        # Left: raw
        axes[i, 0].imshow(img, cmap="gray", vmin=vmin, vmax=vmax)
        axes[i, 0].set_title(f"ch{ch} raw", fontsize=9)
        axes[i, 0].set_xticks([])
        axes[i, 0].set_yticks([])

        # Right: raw + spots
        axes[i, 1].imshow(img, cmap="gray", vmin=vmin, vmax=vmax)
        pts = spots_per_channel.get(ch)
        if pts is not None and len(pts) > 0:
            rows, cols = pts[:, 0], pts[:, 1]
            # Scale spots if they were detected at a different resolution
            if len(rows) > 0:
                max_spot_row = np.max(rows)
                max_spot_col = np.max(cols)
                if max_spot_row > img_h * 1.05 or max_spot_col > img_w * 1.05:
                    scale_r = img_h / (max_spot_row + 1)
                    scale_c = img_w / (max_spot_col + 1)
                    rows = rows * scale_r
                    cols = cols * scale_c
            axes[i, 1].scatter(cols, rows,
                               s=1, c="red", alpha=0.5, linewidths=0)
        n_pts = len(pts) if pts is not None else 0
        axes[i, 1].set_title(f"ch{ch} + spots ({n_pts})", fontsize=9)
        axes[i, 1].set_xlim(0, img_w)
        axes[i, 1].set_ylim(img_h, 0)
        axes[i, 1].set_xticks([])
        axes[i, 1].set_yticks([])

    fig.suptitle(f"Channel projections - FOV {fov_id}", fontsize=11)
    fig.tight_layout()

    if out_path:
        fig.savefig(out_path, dpi=200)
        plt.close(fig)
    return fig, axes


# ---------------------------
# MASKS OVERLAY (FIG 3)
# ---------------------------

def plot_masks_overlay(
    dapi: np.ndarray,
    nuclei_labels: np.ndarray,
    fg_mask: np.ndarray,
    bg_mask: np.ndarray,
    fov_id: str,
    out_path: Optional[str] = None,
):
    """
    DAPI image with nuclei boundaries, FG ring, and BG region overlaid.
    """
    from scipy.ndimage import binary_erosion

    # Reference shape from labels/masks (they always match)
    H, W = nuclei_labels.shape[:2]

    fig, ax = plt.subplots(figsize=(7, 7))

    # Base: DAPI (may differ in size from labels — use extent to align)
    vmax = np.percentile(dapi, 99.5)
    ax.imshow(dapi, cmap="gray", vmin=0, vmax=vmax,
              extent=[0, W, H, 0])  # align to labels pixel grid

    # Nuclei boundaries (blue) — outer edge via erosion (no skimage needed)
    nuc_mask = nuclei_labels > 0
    boundaries = nuc_mask & ~binary_erosion(nuc_mask)
    blue_overlay = np.zeros((H, W, 4), dtype=np.float32)
    blue_overlay[boundaries] = [0.2, 0.4, 1.0, 0.7]
    ax.imshow(blue_overlay, extent=[0, W, H, 0])

    # FG ring (green) — fg_mask excluding nucleus interior
    fg_ring = fg_mask & (nuclei_labels == 0)
    green_overlay = np.zeros((H, W, 4), dtype=np.float32)
    green_overlay[fg_ring] = [0.2, 0.8, 0.2, 0.25]
    ax.imshow(green_overlay, extent=[0, W, H, 0])

    # BG (red)
    red_overlay = np.zeros((H, W, 4), dtype=np.float32)
    red_overlay[bg_mask] = [0.9, 0.2, 0.2, 0.25]
    ax.imshow(red_overlay, extent=[0, W, H, 0])

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=(0.2, 0.4, 1.0, 0.7), label="Nuclei"),
        Patch(facecolor=(0.2, 0.8, 0.2, 0.25), label="FG / cell-proximal"),
        Patch(facecolor=(0.9, 0.2, 0.2, 0.25), label="BG / cell-distal"),
    ]
    ax.legend(handles=legend_elements, loc="upper right", fontsize=8,
              framealpha=0.8)

    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(f"FG / BG masks - FOV {fov_id}", fontsize=11)
    fig.tight_layout()

    if out_path:
        fig.savefig(out_path, dpi=200)
        plt.close(fig)
    return fig, ax


def visualize_nuclei_rings_and_spots_napari(
    dapi: np.ndarray,
    nuclei_labels: np.ndarray,
    cell_proximal: np.ndarray,
    cell_distal: np.ndarray,
    spots_rc: np.ndarray,
    *,
    valid_mask: np.ndarray | None = None,
    spot_size: float = 4.0,
):
    """
    Visualize DAPI + nuclei + FG/BG rings + RNA spots in Napari.

    spots_rc: (N, 2) array in (row, col) order
    """
    import napari
    viewer = napari.Viewer()

    viewer.add_image(
        dapi,
        name="DAPI",
        contrast_limits=[0, np.percentile(dapi, 99.8)],
    )

    viewer.add_labels(
        nuclei_labels.astype(np.int32),
        name="nuclei",
    )

    viewer.add_labels(
        cell_proximal.astype(np.int32),
        name="cell_proximal (FG)",
        opacity=0.35,
    )

    viewer.add_labels(
        cell_distal.astype(np.int32),
        name="cell_distal (BG)",
        opacity=0.35,
    )
    if valid_mask is not None:
        viewer.add_labels(
            valid_mask.astype(np.uint8),
            name="valid_mask (mosaic)",
            opacity=0.25,
        )

    viewer.add_points(
        spots_rc,
        name="RNA spots",
        size=spot_size,
        face_color="yellow",
        opacity=0.9,
    )

    napari.run()
