"""
End-to-end SQI pipeline for a single FOV.

Example
-------
python scripts/run_sqi_from_fov_zarr.py \
  --fov_zarr  //server/data/Conv_zscan1_074.zarr \
  --data_fld  /data/H1_PTBP1_TH_GFAP_set1 \
  --cache_root /cache \
  --out_root   /output
"""
import argparse
import json
import os
from pathlib import Path

import numpy as np
import tifffile as tiff

from sqi.io.image_io import (
    read_dapi_from_conv_zarr,
    read_multichannel_from_conv_zarr,
    write_labels_tif,
)
from sqi.io.spots_io import read_spots_parquet, write_spots_parquet, write_spots_meta
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
from sqi.spot_calling.spotiflow_backend import SpotiflowBackend, SpotiflowConfig
from sqi.spot_features.features import compute_spot_features, SpotFeatureConfig
from sqi.spot_features.quality import compute_quality_scores, QualityGateConfig


# ----------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------

def detect_spots_spotiflow(
    fov_zarr: str,
    labels: np.ndarray,
    fg_mask: np.ndarray,
    bg_mask: np.ndarray,
    cache_dir: Path,
    *,
    model: str = "general",
    prob_thresh: float = 0.5,
    force: bool = False,
):
    """Detect spots via Spotiflow, compute features + quality, cache as parquet."""
    parquet_path = cache_dir / "spots.parquet"
    meta_path = cache_dir / "spots_meta.json"

    if parquet_path.exists() and meta_path.exists() and not force:
        print("       Loading cached spots parquet")
        return read_spots_parquet(str(parquet_path))

    im = read_multichannel_from_conv_zarr(fov_zarr)
    n_channels = im.shape[0]
    spot_channel_indices = list(range(n_channels - 1))

    sf_cfg = SpotiflowConfig(pretrained_model=model, prob_thresh=prob_thresh)
    backend = SpotiflowBackend(sf_cfg)

    all_spots, all_scores, all_channels, all_metas = [], [], [], []

    for ch_idx in spot_channel_indices:
        ch_img = im[ch_idx]
        if ch_img.ndim == 3:
            ch_2d = np.array(ch_img.max(axis=0), dtype=np.float32)
        else:
            ch_2d = np.array(ch_img, dtype=np.float32)

        spots_rc, scores, meta = backend.detect(ch_2d)
        print(f"       channel {ch_idx}: {len(spots_rc)} spots")
        all_spots.append(spots_rc)
        all_scores.append(scores)
        all_channels.append(np.full(len(spots_rc), ch_idx, dtype=np.int32))
        all_metas.append(meta)

    if all_spots:
        spots_rc = np.vstack(all_spots).astype(np.float32)
        scores = np.concatenate(all_scores).astype(np.float32)
        channels = np.concatenate(all_channels)
    else:
        spots_rc = np.zeros((0, 2), dtype=np.float32)
        scores = np.zeros(0, dtype=np.float32)
        channels = np.zeros(0, dtype=np.int32)

    # Composite image for intensity features
    composite = np.zeros(labels.shape, dtype=np.float32)
    for ch_idx in spot_channel_indices:
        ch_img = im[ch_idx]
        if ch_img.ndim == 3:
            ch_2d = np.array(ch_img.max(axis=0), dtype=np.float32)
        else:
            ch_2d = np.array(ch_img, dtype=np.float32)
        composite = np.maximum(composite, ch_2d)

    df = compute_spot_features(
        composite, spots_rc, scores, labels, fg_mask, bg_mask,
    )
    df.insert(2, "channel", channels)
    df = compute_quality_scores(df)

    write_spots_parquet(df, str(parquet_path))
    write_spots_meta({
        "n_spots_total": len(df),
        "n_pass_permissive": int(df["pass_permissive"].sum()),
        "n_pass_conservative": int(df["pass_conservative"].sum()),
        "per_channel": all_metas,
    }, str(meta_path))

    return df


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
    # 3. Mosaic coordinates + tissue mask
    # =====================================================
    print("[3/7] Loading mosaic coordinates ...")
    mosaic_cfg = MosaicBuildConfig(resc=resc, icol=1, frame=20, rescz=2)
    _, fls_, xs, ys = build_mosaic_and_coords(
        args.data_fld, mosaic_cfg, cache_root=args.cache_root, cache=True,
    )

    mosaic_tif, _ = mosaic_cache_paths(args.data_fld, mosaic_cfg, args.cache_root)
    global_valid = get_or_build_tissue_mask(mosaic_tif, args.cache_root)

    # =====================================================
    # 4. Crop valid mask for this FOV
    # =====================================================
    print("[4/7] Cropping tissue mask for FOV ...")
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
    # 5. FG / BG masks
    # =====================================================
    print("[5/7] Building cell-proximal / cell-distal masks ...")
    cp_cfg = CellProximalConfig(cell_proximal_px=args.cell_proximal_px)
    fg_mask, bg_mask, region_stats = build_cell_proximal_and_distal_masks(
        labels, valid_mask, cp_cfg,
    )
    fg_label_map, bg_label_map = per_cell_fg_bg(labels, fg_mask, bg_mask)
    print(f"       {region_stats}")

    # Save masks for standalone run_spots.py usage
    tiff.imwrite(str(cache_dir / "fg_mask.tif"), fg_mask.astype(np.uint8))
    tiff.imwrite(str(cache_dir / "bg_mask.tif"), bg_mask.astype(np.uint8))

    # =====================================================
    # 6. Spot detection (Spotiflow + features + quality)
    # =====================================================
    print("[6/7] Detecting spots (Spotiflow) ...")
    spots_df = detect_spots_spotiflow(
        fov_zarr, labels, fg_mask, bg_mask, cache_dir,
        model=args.spot_model,
        prob_thresh=args.prob_thresh,
        force=args.force,
    )

    # Filter to conservative-pass spots, use q_score as weight
    spots_pass = spots_df[spots_df["pass_conservative"]].copy()
    spots_int = np.column_stack([
        np.clip(spots_pass["row"].values.astype(np.int32), 0, labels.shape[0] - 1),
        np.clip(spots_pass["col"].values.astype(np.int32), 0, labels.shape[1] - 1),
    ])
    weights = spots_pass["q_score"].values.astype(np.float32)
    print(f"       n_spots_total = {len(spots_df)}, n_pass = {len(spots_int)}")

    # =====================================================
    # 7. Compute SQI (quality-weighted)
    # =====================================================
    print("[7/7] Computing SQI ...")
    sqi, fg_counts, bg_counts = compute_sqi_from_label_maps(
        fg_label_map, bg_label_map, spots_int,
        spot_weights=weights,
    )

    # --- Save results ---
    vals = np.array([v for v in sqi.values() if np.isfinite(v)])
    summary = {
        "fov_id": fov_id,
        **region_stats,
        "n_spots_total": len(spots_df),
        "n_spots_pass": len(spots_int),
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
            w.writerow([cid,
                        f"{sqi[cid]:.4f}" if np.isfinite(sqi[cid]) else "nan",
                        f"{fg_counts.get(cid, 0):.4f}",
                        f"{bg_counts.get(cid, 0):.4f}"])

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
    parser.add_argument("--spot_model", default="general",
                        help="Spotiflow pretrained model (default: general)")
    parser.add_argument("--prob_thresh", type=float, default=0.5,
                        help="Spotiflow probability threshold (default: 0.5)")
    parser.add_argument("--force", action="store_true",
                        help="Recompute all cached intermediates")
    main(parser.parse_args())
