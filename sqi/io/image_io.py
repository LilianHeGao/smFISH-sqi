from __future__ import annotations

from typing import Tuple
import numpy as np
import tifffile as tiff

import zarr

import os
from dask import array as da


def _fov_variants(fov: str) -> list:
    """Return FOV id variants: original, stripped, zero-padded to 3/4 digits."""
    seen = []
    for v in [fov, fov.lstrip("0") or "0", fov.zfill(3), fov.zfill(4)]:
        if v not in seen:
            seen.append(v)
    return seen


def _resolve_fov_data_path(dirname: str, fov: str) -> str:
    """Try FOV variants (e.g. '040', '40', '0040') for data directory."""
    candidates = _fov_variants(fov)
    for f in candidates:
        p = os.path.join(dirname, f, "data")
        if os.path.exists(p):
            return p
    raise FileNotFoundError(
        f"Expected data directory not found. Tried: "
        + ", ".join(os.path.join(dirname, f, "data") for f in candidates)
    )


def _resolve_xml_path(dirname: str, zarr_basename: str, fov: str) -> str:
    """Try FOV variants for the XML metadata file."""
    prefix = zarr_basename.rsplit("_", 1)[0]  # e.g. 'Conv_zscan11'
    candidates = _fov_variants(fov)
    for f in candidates:
        p = os.path.join(dirname, f"{prefix}_{f}.xml")
        if os.path.exists(p):
            return p
    raise FileNotFoundError(
        f"XML metadata not found. Tried: "
        + ", ".join(f"{prefix}_{f}.xml" for f in candidates)
    )


def read_dapi_from_conv_zarr(
    zarr_path: str,
    channel: int = -1,
    z_project: str = "max",
) -> np.ndarray:
    """
    Read DAPI from MERFISH Conv Zarr layout (Bintu/Bogdan style).

    Expected layout:
        Conv_zscanXXX.zarr   (metadata only)
        Conv_zscanXXX/       (directory)
            └── data/        (Zarr array)

    Parameters
    ----------
    zarr_path : str
        Path to Conv_zscanXXX.zarr
    channel : int
        DAPI channel index (default: -1)
    z_project : {"max", "mean"}

    Returns
    -------
    dapi_2d : (Y, X) float32
    """
    dirname = os.path.dirname(zarr_path)
    fov = os.path.basename(zarr_path).split("_")[-1].split(".")[0]

    data_path = _resolve_fov_data_path(dirname, fov)

    # Load lazily (dask), skip first frame as in original pipeline
    image = da.from_zarr(data_path)[1:]

    shape = image.shape
    zarr_base = os.path.basename(zarr_path).split(".")[0]
    xml_file = _resolve_xml_path(dirname, zarr_base, fov)

    txt = open(xml_file, "r").read()

    tag = "<z_offsets type=\"string\">"
    zstack = txt.split(tag)[-1].split("</")[0]

    nchannels = int(zstack.split(":")[-1])

    nzs = (shape[0] // nchannels) * nchannels
    image = image[:nzs]

    # reshape → (Z, C, Y, X) → (C, Z, Y, X)
    image = image.reshape(
        shape[0] // nchannels,
        nchannels,
        shape[-2],
        shape[-1],
    ).swapaxes(0, 1)

    # Select DAPI channel
    dapi = image[channel]

    # Z projection
    if z_project == "max":
        dapi_2d = dapi.max(axis=0)
    elif z_project == "mean":
        dapi_2d = dapi.mean(axis=0)
    else:
        raise ValueError(f"Unknown z_project: {z_project}")

    return dapi_2d.compute().astype(np.float32, copy=False)

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

def read_multichannel_from_conv_zarr(
    zarr_path: str,
    return_pos: bool = False,
):
    """
    Read full multi-channel image from Bintu/Bogdan Conv Zarr layout.

    Returns
    -------
    image : dask array, shape (C, Z, Y, X)
    x, y  : float (only if return_pos=True)
    """
    dirname = os.path.dirname(zarr_path)
    fov = os.path.basename(zarr_path).split("_")[-1].split(".")[0]
    data_path = _resolve_fov_data_path(dirname, fov)
    image = da.from_zarr(data_path)[1:]

    shape = image.shape
    zarr_base = os.path.basename(zarr_path).split(".")[0]
    xml_file = _resolve_xml_path(dirname, zarr_base, fov)

    txt = open(xml_file, "r").read()
    tag = '<z_offsets type="string">'
    zstack = txt.split(tag)[-1].split("</")[0]

    tag = '<stage_position type="custom">'
    x, y = eval(txt.split(tag)[-1].split("</")[0])

    nchannels = int(zstack.split(":")[-1])
    nzs = (shape[0] // nchannels) * nchannels
    image = image[:nzs].reshape(
        shape[0] // nchannels, nchannels, shape[-2], shape[-1]
    )
    image = image.swapaxes(0, 1)

    if return_pos:
        return image, x, y
    return image


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
