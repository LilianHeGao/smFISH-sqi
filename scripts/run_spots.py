"""
Spot detection + per-channel feature extraction + quality scoring for a single FOV.

Example
-------
python scripts/run_spots.py \
  --fov_zarr  //server/data/Conv_zscan1_074.zarr \
  --labels    /cache/074/nuclei_labels.tif \
  --fg_mask   /cache/074/fg_mask.tif \
  --bg_mask   /cache/074/bg_mask.tif \
  --out_dir   /output/074
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import tifffile as tiff

from sqi.io.image_io import read_multichannel_from_conv_zarr
from sqi.io.spots_io import write_spots_parquet, write_spots_meta
from sqi.spot_calling.spotiflow_backend import SpotiflowBackend, SpotiflowConfig
from sqi.spot_features.features import compute_spot_features, SpotFeatureConfig
from sqi.spot_features.quality import compute_quality_scores, QualityGateConfig


def _zproject(ch_img):
    """Max z-project a (Z,Y,X) or (Y,X) dask/numpy array to 2D float32."""
    if ch_img.ndim == 3:
        return np.array(ch_img.max(axis=0), dtype=np.float32)
    return np.array(ch_img, dtype=np.float32)


def main(args):
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    parquet_path = out_dir / "spots.parquet"
    meta_path = out_dir / "spots_meta.json"

    if parquet_path.exists() and meta_path.exists() and not args.force:
        print(f"[CACHE HIT] {parquet_path}")
        return

    # --------------------------------------------------
    # 1. Load image channels (non-DAPI), z-project
    # --------------------------------------------------
    print("[1/4] Loading image channels ...")
    im = read_multichannel_from_conv_zarr(args.fov_zarr)
    n_channels = im.shape[0]
    spot_channel_indices = list(range(n_channels - 1))
    print(f"       {len(spot_channel_indices)} spot channels, DAPI at index {n_channels - 1}")

    # Pre-compute z-projected images per channel
    ch_images = {ch: _zproject(im[ch]) for ch in spot_channel_indices}

    # --------------------------------------------------
    # 2. Detect spots per channel
    # --------------------------------------------------
    print("[2/4] Running Spotiflow per channel ...")
    sf_cfg = SpotiflowConfig(
        pretrained_model=args.model,
        prob_thresh=args.prob_thresh,
    )
    backend = SpotiflowBackend(sf_cfg)

    all_spots, all_scores, all_channels, all_metas = {}, {}, {}, {}

    for ch_idx in spot_channel_indices:
        spots_rc, scores, meta = backend.detect(ch_images[ch_idx])
        print(f"       channel {ch_idx}: {len(spots_rc)} spots")
        all_spots[ch_idx] = spots_rc
        all_scores[ch_idx] = scores
        all_metas[ch_idx] = meta

    # --------------------------------------------------
    # 3. Compute per-spot features (per channel)
    # --------------------------------------------------
    print("[3/4] Computing spot features per channel ...")
    labels = tiff.imread(args.labels).astype(np.int32)
    fg_mask = tiff.imread(args.fg_mask).astype(bool)
    bg_mask = tiff.imread(args.bg_mask).astype(bool)
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

    if dfs:
        df = pd.concat(dfs, ignore_index=True)
    else:
        df = pd.DataFrame()

    print(f"       total spots: {len(df)}")

    # --------------------------------------------------
    # 4. Quality scoring (per channel)
    # --------------------------------------------------
    print("[4/4] Computing quality scores (per channel) ...")
    q_cfg = QualityGateConfig()
    if len(df) > 0:
        df = compute_quality_scores(df, q_cfg)

    # --------------------------------------------------
    # Save
    # --------------------------------------------------
    write_spots_parquet(df, str(parquet_path))

    per_ch_summary = {}
    for ch_idx in spot_channel_indices:
        ch_sub = df[df["channel"] == ch_idx] if len(df) > 0 else df
        per_ch_summary[str(ch_idx)] = {
            **all_metas.get(ch_idx, {}),
            "n_spots": len(ch_sub),
            "n_pass_permissive": int(ch_sub["pass_permissive"].sum()) if len(ch_sub) > 0 else 0,
            "n_pass_conservative": int(ch_sub["pass_conservative"].sum()) if len(ch_sub) > 0 else 0,
        }

    combined_meta = {
        "fov_zarr": args.fov_zarr,
        "n_channels": len(spot_channel_indices),
        "spotiflow_config": sf_cfg.__dict__,
        "feature_config": feat_cfg.__dict__,
        "quality_config": q_cfg.__dict__,
        "n_spots_total": len(df),
        "n_pass_permissive": int(df["pass_permissive"].sum()) if len(df) > 0 else 0,
        "n_pass_conservative": int(df["pass_conservative"].sum()) if len(df) > 0 else 0,
        "per_channel": per_ch_summary,
    }
    write_spots_meta(combined_meta, str(meta_path))

    print("=" * 50)
    print(f"[DONE] {len(df)} spots")
    for ch_idx in spot_channel_indices:
        s = per_ch_summary[str(ch_idx)]
        print(f"  ch{ch_idx}: {s['n_spots']} total, {s['n_pass_conservative']} pass_conservative")
    print(f"  output: {out_dir}")
    print("=" * 50)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Spot detection + features + quality for a single FOV",
    )
    parser.add_argument("--fov_zarr", required=True,
                        help="Path to Conv_zscanX_NNN.zarr")
    parser.add_argument("--labels", required=True,
                        help="Nuclei labels TIFF (int32)")
    parser.add_argument("--fg_mask", required=True,
                        help="FG (cell-proximal) mask TIFF")
    parser.add_argument("--bg_mask", required=True,
                        help="BG (cell-distal) mask TIFF")
    parser.add_argument("--out_dir", required=True,
                        help="Output directory for spots.parquet + spots_meta.json")
    parser.add_argument("--model", default="general",
                        help="Spotiflow pretrained model (default: general)")
    parser.add_argument("--prob_thresh", type=float, default=0.5,
                        help="Spotiflow probability threshold (default: 0.5)")
    parser.add_argument("--force", action="store_true",
                        help="Recompute even if cached")
    main(parser.parse_args())
