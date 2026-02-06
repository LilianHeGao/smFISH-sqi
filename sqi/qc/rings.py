from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple, Dict

import numpy as np
from scipy import ndimage as ndi
from skimage.segmentation import find_boundaries


@dataclass
class RingConfig:
    """
    Defines foreground and background regions around each nucleus.

    Design principles:
    - Foreground (FG): nucleus dilated by fg_dilate_px (captures perinuclear signal)
    - Background (BG): a ring outside nucleus:
        between bg_inner_px and bg_outer_px from the nucleus boundary
    - Exclude overlaps with other nuclei to prevent contamination.
    """
    fg_dilate_px: int = 3
    bg_inner_px: int = 6
    bg_outer_px: int = 20


def _disk_radius_mask(radius: int) -> np.ndarray:
    r = int(radius)
    if r <= 0:
        return np.ones((1, 1), dtype=bool)
    yy, xx = np.ogrid[-r:r+1, -r:r+1]
    return (xx * xx + yy * yy) <= r * r


def build_fg_bg_masks(
    nuclei_labels: np.ndarray,
    cfg: RingConfig,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, int]]:
    """
    Build per-pixel FG mask (union across nuclei) and BG mask (union across nuclei).
    Also returns summary stats for bookkeeping.

    Outputs are boolean masks (H,W).

    Note: For per-cell QC, we’ll compute per-label FG/BG later by masking with label id.
    """
    if nuclei_labels.ndim != 2:
        raise ValueError("nuclei_labels must be 2D")

    labels = nuclei_labels.astype(np.int32, copy=False)
    H, W = labels.shape
    n = int(labels.max())

    nuclei_bin = labels > 0

    # FG: dilate nuclei
    fg = ndi.binary_dilation(nuclei_bin, structure=_disk_radius_mask(cfg.fg_dilate_px))

    # BG ring: distance from nuclei boundary
    # Use distance transform on background to compute distance to nearest nucleus pixel.
    dist_to_nuclei = ndi.distance_transform_edt(~nuclei_bin)

    bg = (dist_to_nuclei >= cfg.bg_inner_px) & (dist_to_nuclei <= cfg.bg_outer_px)

    # Exclude any pixels that belong to nuclei or FG (keep BG clean)
    bg &= ~fg

    # Optional: remove boundary pixels to reduce edge effects (sometimes helps)
    boundaries = find_boundaries(labels, mode="thick")
    bg &= ~boundaries

    stats = {
        "n_nuclei": n,
        "fg_px": int(fg.sum()),
        "bg_px": int(bg.sum()),
        "img_h": H,
        "img_w": W,
    }
    return fg, bg, stats


def per_cell_fg_bg(
    nuclei_labels: np.ndarray,
    fg_union: np.ndarray,
    bg_union: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert union masks into per-cell masks by intersecting with each label's territory.

    For FG per cell: pixels in fg_union that are closest to that nucleus label
    For BG per cell: pixels in bg_union that are closest to that nucleus label

    Implementation:
    - Compute nearest nucleus label for each pixel using a distance transform with indices.
    - Assign FG/BG pixels to the nearest label id, then you can aggregate per label.
    """
    labels = nuclei_labels.astype(np.int32, copy=False)
    nuclei_bin = labels > 0

    # Distance transform returns nearest nucleus pixel indices for every background pixel
    # We use those indices to map each pixel to a nucleus label.
    _, (iy, ix) = ndi.distance_transform_edt(~nuclei_bin, return_indices=True)
    nearest_label = labels[iy, ix]  # 0 where no nuclei exist (should be rare)

    fg_label_map = np.where(fg_union, nearest_label, 0).astype(np.int32, copy=False)
    bg_label_map = np.where(bg_union, nearest_label, 0).astype(np.int32, copy=False)

    return fg_label_map, bg_label_map
