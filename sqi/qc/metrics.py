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


def sqi_sanity_check(
    fg_label_map: np.ndarray,
    bg_label_map: np.ndarray,
    spots_rc: np.ndarray,
    spot_weights=None,
    *,
    seed: int = 42,
):
    """
    Null-model sanity check for the SQI metric.

    For each cell, count its real spots, then uniformly sample the same
    number of pseudo-spots from that cell's valid-tissue territory
    (FG ∪ BG Voronoi region).  Re-run SQI on the pseudo-spots.

    If the metric works, real SQI >> 1 while null SQI ≈ 1.

    Returns
    -------
    real_sqi : dict {cell_id: float}
    null_sqi : dict {cell_id: float}
    auc : float
        ROC AUC (Mann-Whitney U) of log10(real) vs log10(null).
        NaN if either group has fewer than 5 finite positive values.
    """
    # --- real SQI ---
    real_sqi, _, _ = compute_sqi_from_label_maps(
        fg_label_map, bg_label_map, spots_rc, spot_weights,
    )

    # --- count actual (unweighted) spots per cell ---
    cell_counts = {}
    for r, c in spots_rc:
        lab = fg_label_map[r, c]
        if lab == 0:
            lab = bg_label_map[r, c]
        if lab > 0:
            cell_counts[lab] = cell_counts.get(lab, 0) + 1

    # --- build per-cell pixel pool (valid tissue = FG ∪ BG) ---
    combined = np.where(fg_label_map > 0, fg_label_map, bg_label_map)
    rows, cols = np.where(combined > 0)
    labs = combined[rows, cols]

    order = np.argsort(labs)
    rows, cols, labs = rows[order], cols[order], labs[order]
    unique_labs, starts, counts = np.unique(
        labs, return_index=True, return_counts=True,
    )
    pool = {}
    for i, lab in enumerate(unique_labs):
        s = starts[i]
        pool[lab] = np.column_stack([rows[s : s + counts[i]],
                                     cols[s : s + counts[i]]])

    # --- sample pseudo-spots, matching per-cell real count ---
    rng = np.random.default_rng(seed)
    pseudo_list = []
    for lab, n in cell_counts.items():
        px = pool.get(lab)
        if px is None or len(px) == 0:
            continue
        idx = rng.choice(len(px), size=n, replace=True)
        pseudo_list.append(px[idx])

    if not pseudo_list:
        return real_sqi, {}, np.nan

    pseudo_spots = np.concatenate(pseudo_list, axis=0)

    # --- null SQI (unit weights — pure spatial test) ---
    null_sqi, _, _ = compute_sqi_from_label_maps(
        fg_label_map, bg_label_map, pseudo_spots, spot_weights=None,
    )

    # --- AUC: Mann-Whitney U between log10 real vs null ---
    real_log = np.array([
        np.log10(v) for v in real_sqi.values()
        if np.isfinite(v) and v > 0
    ])
    null_log = np.array([
        np.log10(v) for v in null_sqi.values()
        if np.isfinite(v) and v > 0
    ])

    if len(real_log) < 5 or len(null_log) < 5:
        auc = np.nan
    else:
        # Mann-Whitney U (no sklearn)
        u = 0.0
        for r in real_log:
            u += np.sum(r > null_log) + 0.5 * np.sum(r == null_log)
        auc = u / (len(real_log) * len(null_log))

    return real_sqi, null_sqi, auc
