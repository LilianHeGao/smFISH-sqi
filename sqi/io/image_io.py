from __future__ import annotations

from typing import Tuple
import numpy as np
import tifffile as tiff


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
