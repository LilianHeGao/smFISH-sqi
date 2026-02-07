from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple
import os
import hashlib
import json

import numpy as np
from scipy import ndimage as ndi
from skimage.morphology import disk, remove_small_objects

try:
    from skimage.morphology import closing
except ImportError:
    from skimage.morphology import binary_closing as closing

try:
    import tifffile
except Exception:  # pragma: no cover
    tifffile = None


# ============================================================
# Config
# ============================================================

@dataclass
class MosaicValidMaskConfig:
    """
    Support-only tissue mask.

    This mask answers ONLY:
    'Is there tissue here at all?'

    It must NOT:
    - judge signal quality
    - exclude dim tissue
    - react to illumination variation
    """
    downsample: int = 4
    closing_radius: int = 20          # full-res pixels
    min_object_size: int = 20_000     # full-res pixels
    fill_holes: bool = True


# ============================================================
# Core logic
# ============================================================
def compute_global_valid_mask_from_mosaic(
    mosaic_img: np.ndarray,
    cfg: MosaicValidMaskConfig,
) -> np.ndarray:
    """
    Compute a tissue-support mask that captures the true tissue footprint
    (borders + holes) without judging signal quality.
    """

    if mosaic_img.ndim != 2:
        raise ValueError("mosaic_img must be 2D")

    m = mosaic_img.astype(np.float32, copy=False)
    full_shape = m.shape

    # -------------------------------------------------
    # 1. Downsample (for speed only)
    # -------------------------------------------------
    ds = max(1, int(cfg.downsample))
    if ds > 1:
        m_small = m[::ds, ::ds]
    else:
        m_small = m

    # -------------------------------------------------
    # 2. Remove nuclei-scale structure
    #    (estimate tissue envelope)
    # -------------------------------------------------
    # This blur radius should be MUCH larger than nuclei
    blur_radius = max(10, cfg.closing_radius // ds)
    m_blur = ndi.gaussian_filter(m_small, sigma=blur_radius)

    # -------------------------------------------------
    # 3. Very permissive threshold on envelope
    # -------------------------------------------------
    nz = m_blur[m_blur > 0]
    if nz.size == 0:
        return np.zeros(full_shape, dtype=bool)

    t = np.percentile(nz, 5)   # tissue-support threshold
    valid = m_blur > t

    # -------------------------------------------------
    # 4. Morphology (still in downsampled space)
    # -------------------------------------------------
    closing_r = max(1, cfg.closing_radius // ds)
    min_obj = max(1, cfg.min_object_size // (ds * ds))

    valid = closing(valid, footprint=disk(closing_r))

    if cfg.fill_holes:
        valid = ndi.binary_fill_holes(valid)

    if min_obj > 0:
        valid = remove_small_objects(valid, min_size=min_obj)

    # -------------------------------------------------
    # 5. Upsample back to full resolution
    # -------------------------------------------------
    if ds > 1:
        valid = ndi.zoom(valid.astype(np.uint8), ds, order=0).astype(bool)
        valid = valid[: full_shape[0], : full_shape[1]]

    return valid


# ============================================================
# Cache helpers
# ============================================================

def _cfg_hash(cfg: MosaicValidMaskConfig) -> str:
    d = {
        "downsample": cfg.downsample,
        "closing_radius": cfg.closing_radius,
        "min_object_size": cfg.min_object_size,
        "fill_holes": cfg.fill_holes,
    }
    return hashlib.md5(json.dumps(d, sort_keys=True).encode()).hexdigest()[:8]


def _mask_cache_path(cache_root: str, mosaic_tif_path: str, cfg: MosaicValidMaskConfig) -> str:
    os.makedirs(cache_root, exist_ok=True)
    base = os.path.basename(mosaic_tif_path).replace(".tif", "").replace(".tiff", "")
    return os.path.join(cache_root, f"{base}_tissue_mask_{_cfg_hash(cfg)}.tiff")


def load_or_compute_global_valid_mask(
    mosaic_tif_path: str,
    cache_root: str,
    cfg: MosaicValidMaskConfig,
    *,
    force: bool = False,
) -> np.ndarray:
    """
    Load cached tissue mask or compute it from mosaic TIFF.
    """
    if tifffile is None:
        raise ImportError("tifffile is required (pip install tifffile).")

    cache_path = _mask_cache_path(cache_root, mosaic_tif_path, cfg)

    if (not force) and os.path.exists(cache_path):
        return tifffile.imread(cache_path).astype(bool)

    mosaic = tifffile.imread(mosaic_tif_path).astype(np.float32, copy=False)
    valid = compute_global_valid_mask_from_mosaic(mosaic, cfg)

    tifffile.imwrite(cache_path, valid.astype(np.uint8))
    return valid


# ============================================================
# FOV cropping
# ============================================================

def crop_valid_mask_for_fov(
    global_valid_mask: np.ndarray,
    fov_anchor_xy: Tuple[float, float],
    fov_shape_hw: Tuple[int, int],
    *,
    mosaic_resc: int = 1,
    anchor_is_upper_left: bool = True,
    round_anchor: bool = True,
) -> np.ndarray:
    """
    Crop mosaic-level valid mask to a single FOV.
    """

    Hm, Wm = global_valid_mask.shape
    hf_full, wf_full = map(int, fov_shape_hw)

    hf = hf_full // mosaic_resc
    wf = wf_full // mosaic_resc

    a0, a1 = fov_anchor_xy
    if round_anchor:
        a0 = int(round(a0))
        a1 = int(round(a1))

    if anchor_is_upper_left:
        r0, c0 = a0, a1
    else:
        r0 = int(round(a0 - hf / 2))
        c0 = int(round(a1 - wf / 2))

    r1, c1 = r0 + hf, c0 + wf

    crop = np.zeros((hf, wf), dtype=bool)

    mr0 = max(0, r0);  mc0 = max(0, c0)
    mr1 = min(Hm, r1); mc1 = min(Wm, c1)

    if mr1 <= mr0 or mc1 <= mc0:
        return np.zeros((hf_full, wf_full), dtype=bool)

    or0 = mr0 - r0
    oc0 = mc0 - c0

    crop[or0 : or0 + (mr1 - mr0), oc0 : oc0 + (mc1 - mc0)] = \
        global_valid_mask[mr0:mr1, mc0:mc1]

    if mosaic_resc > 1:
        out = ndi.zoom(crop.astype(np.uint8), mosaic_resc, order=0).astype(bool)
        out = out[:hf_full, :wf_full]
    else:
        out = crop

    return out


# ============================================================
# Debug helper
# ============================================================

def overlay_bbox_on_mosaic(
    mosaic_img: np.ndarray,
    fov_anchor_xy: Tuple[float, float],
    fov_shape_hw: Tuple[int, int],
    *,
    mosaic_resc: int = 1,
    anchor_is_upper_left: bool = True,
):
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    hf = int(fov_shape_hw[0]) // mosaic_resc
    wf = int(fov_shape_hw[1]) // mosaic_resc
    a0, a1 = fov_anchor_xy

    if anchor_is_upper_left:
        r0, c0 = a0, a1
    else:
        r0 = a0 - hf / 2
        c0 = a1 - wf / 2

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(mosaic_img, cmap="gray")
    ax.add_patch(Rectangle((c0, r0), wf, hf, fill=False, linewidth=2, edgecolor="red"))
    ax.set_title("FOV bbox on mosaic")
    ax.set_axis_off()
    return fig, ax
