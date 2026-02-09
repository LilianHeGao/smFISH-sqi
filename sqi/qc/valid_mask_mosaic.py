# sqi/qc/valid_mask_mosaic.py
#
# FOV-level cropping of a pre-computed mosaic tissue mask.
# Mask generation lives in scripts/build_tissue_mask_qupath_style.py.
from __future__ import annotations

from typing import Tuple

import numpy as np
from scipy import ndimage as ndi


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
    Crop mosaic-level tissue mask to a single FOV.

    Parameters
    ----------
    global_valid_mask:
        (H_mosaic, W_mosaic) boolean mask at mosaic resolution.
    fov_anchor_xy:
        (dim0, dim1) anchor in mosaic pixel coordinates from compose_mosaic.
    fov_shape_hw:
        (H_fov, W_fov) in **full-resolution** pixels.
    mosaic_resc:
        The rescale factor used to build the mosaic (MosaicBuildConfig.resc).
        Anchor and mask live at this downsampled scale; fov_shape_hw is full-res.
    anchor_is_upper_left:
        True  -> anchor is upper-left corner of FOV in mosaic.
        False -> anchor is center of FOV (compose_mosaic default).

    Returns
    -------
    valid_mask_fov:
        (H_fov, W_fov) boolean at full resolution.  Out-of-bounds = False.
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
