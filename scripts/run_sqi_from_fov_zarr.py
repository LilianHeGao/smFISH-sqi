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
import csv
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import tifffile as tiff

from sqi.io.image_io import (
    read_dapi_from_conv_zarr,
    read_multichannel_from_conv_zarr,
    write_labels_tif,
)
from sqi.io.spots_io import read_spots_parquet, write_spots_parquet, write_spots_meta
from segmentation.cellpose_backend import CellposeBackend, CellposeNucleiConfig
from sqi.qc.mosaic_coords import (
    build_mosaic_and_coords, build_fov_anchor_index, lookup_fov_anchor,
    fov_id_from_zarr_path, MosaicBuildConfig, mosaic_cache_paths,
)
from sqi.qc.valid_mask_mosaic import crop_valid_mask_for_fov
from sqi.qc.rings import (
    CellProximalConfig, build_cell_proximal_and_distal_masks, per_cell_fg_bg,
)
from sqi.qc.metrics import compute_sqi_from_label_maps, sqi_sanity_check
from sqi.qc.qc_plots import plot_sqi_distribution, plot_sqi_real_vs_null
from sqi.spot_calling.spotiflow_backend import SpotiflowBackend, SpotiflowConfig
from sqi.spot_features.features import compute_spot_features, SpotFeatureConfig
from sqi.spot_features.quality import compute_quality_scores


# ----------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------

def _zproject(ch_img):
    """Max z-project a (Z,Y,X) or (Y,X) dask/numpy array to 2D float32."""
    if ch_img.ndim == 3:
        return np.array(ch_img.max(axis=0), dtype=np.float32)
    return np.array(ch_img, dtype=np.float32)


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
    """Detect spots per channel, compute per-channel features + quality, cache as parquet."""
    parquet_path = cache_dir / "spots.parquet"
    meta_path = cache_dir / "spots_meta.json"

    if parquet_path.exists() and meta_path.exists() and not force:
        print("       Loading cached spots parquet")
        return read_spots_parquet(str(parquet_path))

    im = read_multichannel_from_conv_zarr(fov_zarr)
    n_channels = im.shape[0]
    spot_channel_indices = list(range(n_channels - 1))

    # Pre-compute z-projected images per channel
    ch_images = {ch: _zproject(im[ch]) for ch in spot_channel_indices}

    # Detect per channel
    sf_cfg = SpotiflowConfig(pretrained_model=model, prob_thresh=prob_thresh)
    backend = SpotiflowBackend(sf_cfg)

    all_spots, all_scores, all_metas = {}, {}, {}
    for ch_idx in spot_channel_indices:
        spots_rc, scores, meta = backend.detect(ch_images[ch_idx])
        print(f"       channel {ch_idx}: {len(spots_rc)} spots")
        all_spots[ch_idx] = spots_rc
        all_scores[ch_idx] = scores
        all_metas[ch_idx] = meta

    # Per-channel features
    feat_cfg = SpotFeatureConfig()
    dfs = []
    for ch_idx in spot_channel_indices:
        if len(all_spots[ch_idx]) == 0:
            continue
        ch_df = compute_spot_features(
            ch_images[ch_idx],
            all_spots[ch_idx],
            all_scores[ch_idx],
            labels, fg_mask, bg_mask, feat_cfg,
        )
        ch_df.insert(2, "channel", ch_idx)
        dfs.append(ch_df)

    df = pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()

    # Per-channel quality scoring
    if len(df) > 0:
        df = compute_quality_scores(df)

    write_spots_parquet(df, str(parquet_path))
    write_spots_meta({
        "n_spots_total": len(df),
        "n_pass_conservative": int(df["pass_conservative"].sum()) if len(df) > 0 else 0,
        "per_channel": {str(k): v for k, v in all_metas.items()},
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


def _compute_sqi_for_spots(fg_label_map, bg_label_map, spots_df):
    """Filter to pass_conservative, compute quality-weighted SQI."""
    spots_pass = spots_df[spots_df["pass_conservative"]].copy()
    if len(spots_pass) == 0:
        return {}, {}, {}, 0

    spots_int = np.column_stack([
        np.clip(spots_pass["row"].values.astype(np.int32), 0, fg_label_map.shape[0] - 1),
        np.clip(spots_pass["col"].values.astype(np.int32), 0, fg_label_map.shape[1] - 1),
    ])
    weights = spots_pass["q_score"].values.astype(np.float32)

    sqi, fg_counts, bg_counts = compute_sqi_from_label_maps(
        fg_label_map, bg_label_map, spots_int,
        spot_weights=weights,
    )
    return sqi, fg_counts, bg_counts, len(spots_int)


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
    anchor_xy = lookup_fov_anchor(fov_index, fov_id)

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

    tiff.imwrite(str(cache_dir / "fg_mask.tif"), fg_mask.astype(np.uint8))
    tiff.imwrite(str(cache_dir / "bg_mask.tif"), bg_mask.astype(np.uint8))

    # =====================================================
    # 6. Spot detection (Spotiflow, per-channel features + quality)
    # =====================================================
    print("[6/7] Detecting spots (Spotiflow) ...")
    spots_df = detect_spots_spotiflow(
        fov_zarr, labels, fg_mask, bg_mask, cache_dir,
        model=args.spot_model,
        prob_thresh=args.prob_thresh,
        force=args.force,
    )

    if len(spots_df) == 0:
        print("[WARN] No spots detected. Skipping SQI.")
        return

    channels = sorted(spots_df["channel"].unique())
    print(f"       n_spots_total = {len(spots_df)}, "
          f"n_pass = {spots_df['pass_conservative'].sum()}")

    # =====================================================
    # 7. Compute SQI (per-channel + total, quality-weighted)
    # =====================================================
    print("[7/7] Computing SQI ...")

    # --- Per-channel SQI ---
    per_ch_sqi = {}
    for ch in channels:
        ch_df = spots_df[spots_df["channel"] == ch]
        sqi_ch, fg_ch, bg_ch, n_pass_ch = _compute_sqi_for_spots(
            fg_label_map, bg_label_map, ch_df,
        )
        vals_ch = np.array([v for v in sqi_ch.values() if np.isfinite(v) and v > 0])
        per_ch_sqi[int(ch)] = {
            "n_spots": len(ch_df),
            "n_pass": n_pass_ch,
            "n_cells_with_sqi": len(vals_ch),
            "median_sqi": float(np.median(vals_ch)) if len(vals_ch) else None,
            "mean_log10_sqi": float(np.mean(np.log10(vals_ch))) if len(vals_ch) else None,
        }
        print(f"       ch{ch}: {n_pass_ch} pass, "
              f"median_sqi={per_ch_sqi[int(ch)]['median_sqi']}")

    # --- Total SQI (all channels combined) ---
    sqi_total, fg_total, bg_total, n_pass_total = _compute_sqi_for_spots(
        fg_label_map, bg_label_map, spots_df,
    )

    # --- Sanity check: real vs null SQI ---
    spots_pass = spots_df[spots_df["pass_conservative"]].copy()
    spots_int = np.column_stack([
        np.clip(spots_pass["row"].values.astype(np.int32), 0, fg_label_map.shape[0] - 1),
        np.clip(spots_pass["col"].values.astype(np.int32), 0, fg_label_map.shape[1] - 1),
    ])
    weights = spots_pass["q_score"].values.astype(np.float32)
    real_sqi_sc, null_sqi_sc, sanity_auc = sqi_sanity_check(
        fg_label_map, bg_label_map, spots_int, weights,
    )
    real_vals_sc = np.array([v for v in real_sqi_sc.values() if np.isfinite(v) and v > 0])
    null_vals_sc = np.array([v for v in null_sqi_sc.values() if np.isfinite(v) and v > 0])
    if len(real_vals_sc) and len(null_vals_sc):
        print(f"       sanity check: real median={np.median(real_vals_sc):.2f}, "
              f"null median={np.median(null_vals_sc):.2f}, AUC={sanity_auc:.3f}")
    else:
        print("       sanity check: insufficient data")

    sqi_reliable = bool(np.isfinite(sanity_auc) and sanity_auc >= 0.6)
    if not sqi_reliable:
        print(f"WARNING: FG/BG separation insufficient for this FOV "
              f"(AUC={sanity_auc:.2f}), SQI may not be informative.")

    # --- Save results ---
    vals = np.array([v for v in sqi_total.values() if np.isfinite(v) and v > 0])
    summary = {
        "fov_id": fov_id,
        **region_stats,
        "n_spots_total": len(spots_df),
        "n_spots_pass": n_pass_total,
        "n_cells_with_sqi": len(vals),
        "median_sqi": float(np.median(vals)) if len(vals) else None,
        "mean_log10_sqi": float(np.mean(np.log10(vals))) if len(vals) else None,
        "sanity_auc": float(sanity_auc) if np.isfinite(sanity_auc) else None,
        "sqi_reliable": sqi_reliable,
        "per_channel": per_ch_sqi,
    }

    (out_dir / "sqi_summary.json").write_text(json.dumps(summary, indent=2))

    # Per-cell CSV (total SQI)
    with open(out_dir / "sqi_per_cell.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["cell_id", "sqi", "fg_spots", "bg_spots"])
        for cid in sorted(sqi_total.keys()):
            w.writerow([cid,
                        f"{sqi_total[cid]:.4f}" if np.isfinite(sqi_total[cid]) else "nan",
                        f"{fg_total.get(cid, 0):.4f}",
                        f"{bg_total.get(cid, 0):.4f}"])

    # Per-cell per-channel CSV
    with open(out_dir / "sqi_per_cell_per_channel.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["cell_id", "channel", "sqi", "fg_spots", "bg_spots"])
        for ch in channels:
            ch_df = spots_df[spots_df["channel"] == ch]
            sqi_ch, fg_ch, bg_ch, _ = _compute_sqi_for_spots(
                fg_label_map, bg_label_map, ch_df,
            )
            for cid in sorted(sqi_ch.keys()):
                w.writerow([cid, int(ch),
                            f"{sqi_ch[cid]:.4f}" if np.isfinite(sqi_ch[cid]) else "nan",
                            f"{fg_ch.get(cid, 0):.4f}",
                            f"{bg_ch.get(cid, 0):.4f}"])

    # Plot
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, len(channels) + 1,
                             figsize=(4 * (len(channels) + 1), 3))
    if len(channels) + 1 == 1:
        axes = [axes]

    # Per-channel histograms
    colors = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple"]
    for i, ch in enumerate(channels):
        ch_df = spots_df[spots_df["channel"] == ch]
        sqi_ch, _, _, _ = _compute_sqi_for_spots(fg_label_map, bg_label_map, ch_df)
        plot_sqi_distribution(sqi_ch, ax=axes[i],
                              color=colors[i % len(colors)],
                              title=f"ch{ch}")

    # Total histogram
    plot_sqi_distribution(sqi_total, ax=axes[-1], title="Total")

    fig.suptitle(f"SQI — FOV {fov_id}", fontsize=12)
    fig.tight_layout()
    fig.savefig(str(out_dir / "sqi_distribution.png"), dpi=200)
    plt.close(fig)

    # Sanity check plot: real vs null
    fig_sc, ax_sc = plt.subplots(figsize=(5, 3.5))
    plot_sqi_real_vs_null(real_sqi_sc, null_sqi_sc, ax=ax_sc,
                          title=f"SQI sanity check — FOV {fov_id}")
    fig_sc.tight_layout()
    fig_sc.savefig(str(out_dir / "sqi_sanity_check.png"), dpi=200)
    plt.close(fig_sc)

    print("=" * 50)
    print(f"[DONE] FOV {fov_id}")
    print(f"  median SQI (total) = {summary['median_sqi']}")
    print(f"  mean log10 SQI     = {summary['mean_log10_sqi']}")
    print(f"  output dir         = {out_dir}")
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
