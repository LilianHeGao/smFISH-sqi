from copy import deepcopy
import numpy as np
from .rings import build_fg_bg_masks

def ring_sensitivity_sweep(
    nuclei_labels,
    base_cfg,
    scales=(0.8, 0.9, 1.0, 1.1, 1.2),
):
    """
    Scale all ring parameters together and rebuild masks.
    """
    results = {}

    for s in scales:
        cfg = deepcopy(base_cfg)
        cfg.fg_dilate_px = int(round(cfg.fg_dilate_px * s))
        cfg.bg_inner_px = int(round(cfg.bg_inner_px * s))
        cfg.bg_outer_px = int(round(cfg.bg_outer_px * s))

        fg, bg, stats = build_fg_bg_masks(nuclei_labels, cfg)
        results[s] = (cfg, fg, bg, stats)

    return results
