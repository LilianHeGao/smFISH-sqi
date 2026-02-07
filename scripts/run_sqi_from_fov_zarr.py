"""
End-to-end SQI pipeline for a single FOV.

Example
-------
python scripts/run_sqi_from_fov_zarr.py \
  --fov_zarr  //server/data/Conv_zscan1_074.zarr
  --data_fld  /data/H1_PTBP1_TH_GFAP_set1
  --cache_root /cache
  --out_root   /output
"""
import argparse
import json
import os
from pathlib import Path

import numpy as np
import tifffile as tiff
from skimage.feature import blob_log

from sqi.io.image_io import (
    read_dapi_from_conv_zarr,
    read_multichannel_from_conv_zarr,
    write_labels_tif,
)
from segmentation.cellpose_backend import CellposeBackend, CellposeNucleiConfig
from sqi.qc.mosaic_coords import (
    build_mosaic_and_coords, build_fov_anchor_index,
    fov_id_from_zarr_path, MosaicBuildConfig, mosaic_cache_paths,
)
from sqi.qc.valid_mask_mosaic import crop_valid_mask_for_fov
from sqi.qc.rings import (
    CellProximalConfig, build_cell_proximal_and_distal_masks, per_cell_fg_bg,
)
from sqi.qc.metrics import compute_sqi_from_label_maps
from sqi.qc.qc_plots import plot_sqi_distribution


# ----------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------

def detect_spots(fov_zarr: str) -> np.ndarray:
    """Detect RNA spots (all non-DAPI channels) via LoG."""
    im = read_multichannel_from_conv_zarr(fov_zarr)
    spot_channels = im[:-1]  # DAPI is last

    spots = []
    for c in range(spot_channels.shape[0]):
        img = spot_channels[c]
        if img.ndim == 3:
            img = np.max(img, axis=0)
        img = np.array(img, dtype=np.float32)

        blobs = blob_log(img, min_sigma=1.0, max_sigma=2.5,
                         num_sigma=5, threshold=0.02)
        if blobs.size > 0:
            spots.append(blobs[:, :2])

    if spots:
        return np.vstack(spots).astype(np.float32)
    return np.zeros((0, 2), dtype=np.float32)


def get_or_build_tissue_mask(mosaic_tif: str, cache_root: str) -> np.ndarray:
    """Load or build tissue mask from the mosaic TIFF."""
    mask_tif = mosaic_tif.replace(".tiff", "_tissue_mask.tiff")

    if os.path.exists(mask_tif):
        return tiff.imread(mask_tif).astype(bool)

    from build_tissue_mask_qupath_style import build_tissue_mask_qupath_style
    mosaic = tiff.imread(mosaic_tif).astype(np.float32)
    mask = build_tissue_mask_qupath_style(mosaic, downsample=4)
    tiff.imwrite(mask_tif, mask.astype(np.uint8))
    print("[INFO] Saved tissue mask:", mask_tif)
    return mask


# ----------------------------------------------------------------
# Main pipeline
# ----------------------------------------------------------------

def main(args):
    fov_zarr = args.fov_zarr
    fov_id = fov_id_from_zarr_path(fov_zarr)
    resc = args.resc

    cache_dir = Path(args.cache_root) / fov_id
    out_dir = Path(args.out_root) / fov_id
    cache_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    # =====================================================
    # 1. DAPI
    # =====================================================
    print(f"[1/7] Loading DAPI from {fov_zarr} ...")
    dapi = read_dapi_from_conv_zarr(fov_zarr, channel=-1)

    # =====================================================
    # 2. Nuclei segmentation (cached)
    # =====================================================
    labels_path = cache_dir / "nuclei_labels.tif"
    if labels_path.exists() and not args.force:
        print("[2/7] Loading cached nuclei labels")
        labels = tiff.imread(str(labels_path)).astype(np.int32)
    else:
        print("[2/7] Running Cellpose nuclei segmentation ...")
        cfg = CellposeNucleiConfig(model_type="nuclei", channels=(0, 0))
        backend = CellposeBackend(cfg)
        labels, meta = backend.segment_nuclei(dapi)
        write_labels_tif(str(labels_path), labels)
        (cache_dir / "nuclei_meta.json").write_text(json.dumps(meta, indent=2))
    print(f"       n_nuclei = {labels.max()}")

    # =====================================================
    # 3. Spot detection (cached)
    # =====================================================
    spots_path = cache_dir / "spots_rc.npy"
    if spots_path.exists() and not args.force:
        print("[3/7] Loading cached spots")
        spots_rc = np.load(str(spots_path)).astype(np.int32)
    else:
        print("[3/7] Detecting spots (LoG) ...")
        spots_rc = detect_spots(fov_zarr)
        np.save(str(spots_path), spots_rc)
    print(f"       n_spots = {len(spots_rc)}")

    # =====================================================
    # 4. Mosaic coordinates + tissue mask
    # =====================================================
    print("[4/7] Loading mosaic coordinates ...")
    mosaic_cfg = MosaicBuildConfig(resc=resc, icol=1, frame=20, rescz=2)
    _, fls_, xs, ys = build_mosaic_and_coords(
        args.data_fld, mosaic_cfg, cache_root=args.cache_root, cache=True,
    )

    mosaic_tif, _ = mosaic_cache_paths(args.data_fld, mosaic_cfg, args.cache_root)
    global_valid = get_or_build_tissue_mask(mosaic_tif, args.cache_root)

    # =====================================================
    # 5. Crop valid mask for this FOV
    # =====================================================
    print("[5/7] Cropping tissue mask for FOV ...")
    fov_index = build_fov_anchor_index(fls_, xs, ys)
    anchor_xy = fov_index[fov_id]

    valid_mask = crop_valid_mask_for_fov(
        global_valid_mask=global_valid,
        fov_anchor_xy=anchor_xy,
        fov_shape_hw=labels.shape,
        mosaic_resc=resc,
        anchor_is_upper_left=False,
    )
    print(f"       valid coverage = {valid_mask.mean():.3f}")
    tiff.imwrite(str(cache_dir / "valid_mask.tif"), valid_mask.astype(np.uint8))

    # =====================================================
    # 6. FG / BG masks
    # =====================================================
    print("[6/7] Building cell-proximal / cell-distal masks ...")
    cp_cfg = CellProximalConfig(cell_proximal_px=args.cell_proximal_px)
    fg_mask, bg_mask, region_stats = build_cell_proximal_and_distal_masks(
        labels, valid_mask, cp_cfg,
    )
    fg_label_map, bg_label_map = per_cell_fg_bg(labels, fg_mask, bg_mask)
    print(f"       {region_stats}")

    # =====================================================
    # 7. Compute SQI
    # =====================================================
    print("[7/7] Computing SQI ...")
    spots_int = spots_rc.astype(np.int32)
    # clip spots to image bounds
    spots_int[:, 0] = np.clip(spots_int[:, 0], 0, labels.shape[0] - 1)
    spots_int[:, 1] = np.clip(spots_int[:, 1], 0, labels.shape[1] - 1)

    sqi, fg_counts, bg_counts = compute_sqi_from_label_maps(
        fg_label_map, bg_label_map, spots_int,
    )

    # --- Save results ---
    vals = np.array([v for v in sqi.values() if np.isfinite(v)])
    summary = {
        "fov_id": fov_id,
        **region_stats,
        "n_spots": len(spots_rc),
        "n_cells_with_sqi": len(vals),
        "median_sqi": float(np.median(vals)) if len(vals) else None,
        "mean_log10_sqi": float(np.mean(np.log10(vals[vals > 0]))) if np.any(vals > 0) else None,
    }

    (out_dir / "sqi_summary.json").write_text(json.dumps(summary, indent=2))

    # Per-cell CSV
    import csv
    with open(out_dir / "sqi_per_cell.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["cell_id", "sqi", "fg_spots", "bg_spots"])
        for cid in sorted(sqi.keys()):
            w.writerow([cid, sqi[cid], fg_counts.get(cid, 0), bg_counts.get(cid, 0)])

    # Plot
    import matplotlib
    matplotlib.use("Agg")
    ax = plot_sqi_distribution(sqi, title=f"SQI — FOV {fov_id}")
    fig = ax.get_figure()
    fig.tight_layout()
    fig.savefig(str(out_dir / "sqi_distribution.png"), dpi=200)

    print("=" * 50)
    print(f"[DONE] FOV {fov_id}")
    print(f"  median SQI     = {summary['median_sqi']}")
    print(f"  mean log10 SQI = {summary['mean_log10_sqi']}")
    print(f"  output dir     = {out_dir}")
    print("=" * 50)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="End-to-end SQI for a single FOV")
    parser.add_argument("--fov_zarr", required=True,
                        help="Path to Conv_zscanX_NNN.zarr")
    parser.add_argument("--data_fld", required=True,
                        help="Parent folder containing all FOV .zarr files (for mosaic coords)")
    parser.add_argument("--cache_root", required=True,
                        help="Directory for cached intermediates (mosaic, masks, labels)")
    parser.add_argument("--out_root", required=True,
                        help="Directory for SQI results")
    parser.add_argument("--resc", type=int, default=4,
                        help="Mosaic rescale factor (default: 4)")
    parser.add_argument("--cell_proximal_px", type=int, default=24,
                        help="Dilation radius for cell-proximal region (default: 24)")
    parser.add_argument("--force", action="store_true",
                        help="Recompute all cached intermediates")
    main(parser.parse_args())
