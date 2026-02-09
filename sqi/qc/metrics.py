import numpy as np


def compute_sqi_from_label_maps(
    fg_label_map: np.ndarray,
    bg_label_map: np.ndarray,
    spots_rc: np.ndarray,
    spot_weights=None,
):
    """
    Compute SQI per cell using per-cell FG/BG label maps.

    Parameters
    ----------
    spots_rc : (N, 2) array of (row, col)
    spot_weights : (N,) array of per-spot quality weights, or None.
        None → unit weights (count-based, backward-compatible).
        Provided → quality-weighted accumulation.
    """
    fg_area = {}
    bg_area = {}
    sqi = {}

    labels = np.unique(np.concatenate([fg_label_map, bg_label_map]))
    labels = labels[labels > 0]

    # precompute pixel areas
    for lab in labels:
        fg_area[lab] = (fg_label_map == lab).sum()
        bg_area[lab] = (bg_label_map == lab).sum()

    # accumulate (weighted) spot counts
    fg_counts = {lab: 0.0 for lab in labels}
    bg_counts = {lab: 0.0 for lab in labels}

    for i, (r, c) in enumerate(spots_rc):
        w = float(spot_weights[i]) if spot_weights is not None else 1.0
        lab_fg = fg_label_map[r, c]
        lab_bg = bg_label_map[r, c]

        if lab_fg > 0:
            fg_counts[lab_fg] += w
        elif lab_bg > 0:
            bg_counts[lab_bg] += w

    # compute SQI
    for lab in labels:
        if bg_counts[lab] == 0 or bg_area[lab] == 0:
            sqi[lab] = np.nan
        else:
            sqi[lab] = (fg_counts[lab] / fg_area[lab]) / (
                bg_counts[lab] / bg_area[lab]
            )

    return sqi, fg_counts, bg_counts
