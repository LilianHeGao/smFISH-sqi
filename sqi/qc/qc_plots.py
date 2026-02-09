# sqi/qc/qc_plots.py
from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Optional

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
