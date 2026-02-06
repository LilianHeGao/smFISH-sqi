# sqi/qc/qc_plots.py
from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Optional
import napari

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
    values = values[np.isfinite(values)]

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


def plot_real_vs_null_sqi(
    sqi_real: Dict[int, float],
    sqi_null: Dict[int, float],
    *,
    title: Optional[str] = None,
):
    """
    Overlay real vs null SQI distributions.
    """
    fig, ax = plt.subplots(figsize=(4.5, 3.2))

    plot_sqi_distribution(
        sqi_real, ax=ax, color="tab:blue", label="Real"
    )
    plot_sqi_distribution(
        sqi_null, ax=ax, color="tab:orange", label="Null"
    )

    if title is not None:
        ax.set_title(title)

    plt.tight_layout()
    return fig, ax


# ---------------------------
# SENSITIVITY PLOTS
# ---------------------------

def plot_sqi_sensitivity(
    sqi_by_scale: Dict[float, Dict[int, float]],
    *,
    stat: str = "median",
):
    """
    Plot SQI summary statistic vs ring scale.

    sqi_by_scale: {scale: {cell_id: sqi}}
    """
    scales = sorted(sqi_by_scale.keys())
    stats = []

    for s in scales:
        vals = np.array(list(sqi_by_scale[s].values()), dtype=float)
        vals = vals[np.isfinite(vals)]
        if stat == "median":
            stats.append(np.median(vals))
        elif stat == "mean":
            stats.append(np.mean(vals))
        else:
            raise ValueError(f"Unknown stat: {stat}")

    fig, ax = plt.subplots(figsize=(4, 3))
    ax.plot(scales, np.log10(stats), marker="o")

    ax.set_xlabel("Ring scale factor")
    ax.set_ylabel(f"log10({stat} SQI)")
    ax.set_title("SQI sensitivity to ring size")

    plt.tight_layout()
    return fig, ax

def visualize_rings_matplotlib(
    nuclei_labels,
    fg_union,
    bg_union,
    *,
    cell_id: Optional[int] = None,
    ax: Optional[plt.Axes] = None,
):
    """
    Visualize nuclei + FG/BG rings.
    If cell_id is provided, highlight only that cell.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(5, 5))

    img = np.zeros((*nuclei_labels.shape, 3), dtype=float)

    # nuclei in gray
    img[..., :] = (nuclei_labels > 0)[..., None] * 0.3

    # FG in green
    img[fg_union] = [0.1, 0.8, 0.1]

    # BG in red
    img[bg_union] = [0.9, 0.2, 0.2]

    if cell_id is not None:
        mask = nuclei_labels == cell_id
        img[mask] = [0.2, 0.2, 1.0]  # highlight nucleus

    ax.imshow(img)
    ax.set_axis_off()
    ax.set_title("FG (green) / BG (red)")

    return ax

def visualize_rings_napari(
    viewer,
    nuclei_labels,
    fg_union,
    bg_union,
):
    """
    Add FG/BG rings to an existing Napari viewer.
    """
    viewer.add_labels(nuclei_labels, name="nuclei")

    viewer.add_labels(
        fg_union.astype(int),
        name="FG ring",
        opacity=0.4,
    )

    viewer.add_labels(
        bg_union.astype(int),
        name="BG ring",
        opacity=0.4,
    )

def visualize_nuclei_with_rings_napari(
    dapi: np.ndarray,
    nuclei_labels: np.ndarray,
    fg_union: np.ndarray,
    bg_union: np.ndarray,
):
    """
    Visualize nuclei + FG/BG rings in Napari, matching existing nuclei checks.
    """
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
        fg_union.astype(np.int32),
        name="FG ring",
        opacity=0.4,
    )

    viewer.add_labels(
        bg_union.astype(np.int32),
        name="BG ring",
        opacity=0.4,
    )

    napari.run()

def visualize_nuclei_rings_and_spots_napari(
    dapi: np.ndarray,
    nuclei_labels: np.ndarray,
    fg_union: np.ndarray,
    bg_union: np.ndarray,
    spots_rc: np.ndarray,
    *,
    spot_size: float = 4.0,
):
    """
    Visualize DAPI + nuclei + FG/BG rings + RNA spots in Napari.

    spots_rc: (N, 2) array in (row, col) order
    """
    viewer = napari.Viewer()

    # --- DAPI ---
    viewer.add_image(
        dapi,
        name="DAPI",
        contrast_limits=[0, np.percentile(dapi, 99.8)],
    )

    # --- nuclei ---
    viewer.add_labels(
        nuclei_labels.astype(np.int32),
        name="nuclei",
    )

    # --- FG / BG ---
    viewer.add_labels(
        fg_union.astype(np.int32),
        name="FG ring",
        opacity=0.35,
    )

    viewer.add_labels(
        bg_union.astype(np.int32),
        name="BG ring",
        opacity=0.35,
    )

    # --- spots ---
    viewer.add_points(
        spots_rc,
        name="RNA spots",
        size=spot_size,
        face_color="yellow",
        opacity=0.9,
    )

    napari.run()
