from __future__ import annotations

from typing import Tuple
import numpy as np
import tifffile as tiff

import zarr


def read_tif_2d(path: str) -> np.ndarray:
    img = tiff.imread(path)
    if img.ndim == 3:
        # If (C,H,W) or (Z,H,W), take first plane by default.
        # You can upgrade later with explicit config.
        img = img[0]
    if img.ndim != 2:
        raise ValueError(f"Expected 2D after squeeze, got shape={img.shape}")
    return img


def write_labels_tif(path: str, labels: np.ndarray) -> None:
    tiff.imwrite(path, labels.astype(np.int32), compression="zlib")

def read_dapi_from_zarr(
    zarr_path: str,
    channel: int = -1,
    scale: int = 0,
    z_project: str = "max",
) -> np.ndarray:
    """
    Load DAPI channel from an OME-Zarr file.

    Parameters
    ----------
    zarr_path : str
        Path to .zarr root
    channel : int
        Channel index for DAPI (default: -1)
    scale : int
        Multiscale level (0 = highest resolution)
    z_project : {"max", "mean"}
        Z projection method

    Returns
    -------
    dapi_2d : (Y, X) float32
    """
    root = zarr.open(zarr_path, mode="r")

    # Handle multiscale vs single-scale
    if "multiscales" in root.attrs:
        arr = root[str(scale)]
    else:
        arr = root

    data = arr[:]  # load lazily-backed array

    # Infer dimension order
    if data.ndim == 5:
        # (C, Z, Y, X) or (S, C, Z, Y, X)
        if data.shape[0] < 10:  # heuristic: channel-first
            dapi = data[channel]
        else:
            dapi = data[scale, channel]
    elif data.ndim == 4:
        # (C, Z, Y, X)
        dapi = data[channel]
    else:
        raise ValueError(f"Unsupported zarr shape: {data.shape}")

    # Z projection
    if dapi.ndim == 3:
        if z_project == "max":
            dapi_2d = dapi.max(axis=0)
        elif z_project == "mean":
            dapi_2d = dapi.mean(axis=0)
        else:
            raise ValueError(f"Unknown z_project: {z_project}")
    elif dapi.ndim == 2:
        dapi_2d = dapi
    else:
        raise ValueError(f"Unexpected DAPI shape: {dapi.shape}")

    return dapi_2d.astype(np.float32, copy=False)
