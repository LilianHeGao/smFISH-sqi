import numpy as np

def compute_sqi_from_label_maps(
    fg_label_map: np.ndarray,
    bg_label_map: np.ndarray,
    spots_rc: np.ndarray,
):
    """
    Compute SQI₁ per cell using per-cell FG/BG label maps.

    spots_rc: (N,2) array of (row, col)
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

    # count spots
    fg_counts = {lab: 0 for lab in labels}
    bg_counts = {lab: 0 for lab in labels}

    for r, c in spots_rc:
        lab_fg = fg_label_map[r, c]
        lab_bg = bg_label_map[r, c]

        if lab_fg > 0:
            fg_counts[lab_fg] += 1
        elif lab_bg > 0:
            bg_counts[lab_bg] += 1

    # compute SQI
    for lab in labels:
        if bg_counts[lab] == 0 or bg_area[lab] == 0:
            sqi[lab] = np.nan
        else:
            sqi[lab] = (fg_counts[lab] / fg_area[lab]) / (
                bg_counts[lab] / bg_area[lab]
            )

    return sqi, fg_counts, bg_counts
