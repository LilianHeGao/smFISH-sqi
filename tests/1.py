"""
QuPath-style tissue detection for large mosaics (Python implementation).

This builds a *geometric tissue support mask*:
- ignores nuclei-scale texture
- captures true tissue borders + holes
- invariant to DAPI density

Workflow:
1. Large Gaussian blur (remove cellular detail)
2. Robust background thresholding
3. Morphological cleanup
4. Save binary mask
5. Optional Napari visualization (mosaic + mask)

This mask is intended for:
- defining tissue-supported regions
- cropping to FOVs
NOT for quality judgment.
"""

from __future__ import annotations

import os
import argparse
import numpy as np
import tifffile as tiff
from scipy import ndimage as ndi
from skimage.morphology import (
    remove_small_objects,
    remove_small_holes,
    disk,
    closing,
)


# ------------------------------------------------------------
# Core tissue detection (QuPath-style)
# ------------------------------------------------------------

def build_tissue_mask_qupath_style(
    mosaic: np.ndarray,
    *,
    downsample: int = 4,
    blur_sigma_fullres: float = 30.0,
    threshold_percentile: float = 5.0,
    min_object_size_fullres: int = 50_000,
    fill_holes: bool = True,
) -> np.ndarray:
    """
    Parameters
    ----------
    mosaic : 2D ndarray
        Full-resolution mosaic image (single channel, e.g. DAPI).
    downsample : int
        Downsample factor for processing (>=2 strongly recommended).
    blur_sigma_fullres : float
        Gaussian sigma in FULL-RES pixels (controls tissue envelope scale).
        80–150 works well for cortex slabs.
    threshold_percentile : float
        Percentile (on blurred nonzero pixels) separating tissue vs background.
        Lower = more inclusive.
    min_object_size_fullres : int
        Minimum tissue area to keep (FULL-RES pixels).
    """

    if mosaic.ndim != 2:
        raise ValueError("Mosaic must be 2D")

    H, W = mosaic.shape
    ds = max(1, int(downsample))

    # --------------------------------------------------
    # 1. Downsample (purely for speed)
    # --------------------------------------------------
    if ds > 1:
        img = mosaic[::ds, ::ds].astype(np.float32)
    else:
        img = mosaic.astype(np.float32)

    # --------------------------------------------------
    # 2. Remove cellular-scale texture (key step)
    # --------------------------------------------------
    sigma_ds = blur_sigma_fullres / ds
    sigma_ds = max(5.0, sigma_ds)

    img_blur = ndi.gaussian_filter(img, sigma=sigma_ds)

    # --------------------------------------------------
    # 3. Robust threshold (background vs tissue)
    # --------------------------------------------------
    nz = img_blur[img_blur > 0]
    if nz.size == 0:
        return np.zeros((H, W), dtype=bool)

    t = np.percentile(nz, threshold_percentile)
    tissue_ds = img_blur > t

    # --------------------------------------------------
    # 4. Morphological cleanup (downsampled space)
    # --------------------------------------------------
    tissue_ds = closing(tissue_ds, footprint=disk(3))

    if fill_holes:
        tissue_ds = remove_small_holes(tissue_ds, area_threshold=500)

    min_obj_ds = int(min_object_size_fullres / (ds * ds))
    if min_obj_ds > 0:
        tissue_ds = remove_small_objects(tissue_ds, min_size=min_obj_ds)

    # --------------------------------------------------
    # 5. Upsample back to full resolution
    # --------------------------------------------------
    if ds > 1:
        tissue = ndi.zoom(
            tissue_ds.astype(np.uint8),
            zoom=ds,
            order=0,
        ).astype(bool)
        tissue = tissue[:H, :W]
    else:
        tissue = tissue_ds

    # --------------------------------------------------
    # FULL-RES boundary refinement (QuPath-style)
    # --------------------------------------------------
    # Smooth the binary mask slightly at full resolution
    # to recover curved boundaries lost during downsampling
    tissue = ndi.binary_opening(tissue, iterations=1)
    tissue = ndi.binary_closing(tissue, iterations=2)

    return tissue



# ------------------------------------------------------------
# CLI + visualization
# ------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mosaic", required=True, help="Mosaic TIFF (2D, single channel).")
    parser.add_argument("--out", required=True, help="Output tissue mask TIFF.")
    parser.add_argument("--downsample", type=int, default=4)
    parser.add_argument("--blur-sigma", type=float, default=30.0,
                        help="Gaussian sigma in FULL-RES pixels (QuPath-like).")
    parser.add_argument("--threshold-percentile", type=float, default=5.0)
    parser.add_argument("--min-object-size", type=int, default=50_000)
    parser.add_argument("--no-fill-holes", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--napari", action="store_true",
                        help="Open Napari to visualize mosaic + tissue mask.")
    args = parser.parse_args()

    if (not args.force) and os.path.exists(args.out):
        print(f"[CACHE HIT] {args.out}")
        mask = tiff.imread(args.out).astype(bool)
    else:
        print("[INFO] Loading mosaic:", args.mosaic)
        mosaic = tiff.imread(args.mosaic)

        print("[INFO] Building tissue mask (QuPath-style)...")
        mask = build_tissue_mask_qupath_style(
            mosaic,
            downsample=args.downsample,
            blur_sigma_fullres=args.blur_sigma,
            threshold_percentile=args.threshold_percentile,
            min_object_size_fullres=args.min_object_size,
            fill_holes=(not args.no_fill_holes),
        )

        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        tiff.imwrite(args.out, mask.astype(np.uint8))

        print(f"[OK] Tissue mask saved: {args.out}")
        print(f"[OK] Coverage: {mask.mean():.4f}")

    # --------------------------------------------------
    # Napari visualization (automatic)
    # --------------------------------------------------
    if args.napari:
        import napari

        img = tiff.imread(args.mosaic)
        viewer = napari.Viewer()
        viewer.add_image(img, name="mosaic", contrast_limits=[img.min(), img.max()])
        viewer.add_labels(mask, name="tissue_mask")
        napari.run()


if __name__ == "__main__":
    main()
