import argparse
import json
from pathlib import Path

import numpy as np
import tifffile as tiff

from sqi.io.image_io import read_tif_2d
from sqi.qc.rings import RingConfig, build_fg_bg_masks, per_cell_fg_bg
from sqi.qc.qc_plots import QC1Config, compute_per_cell_stats, plot_qc1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--intensity", required=True, help="RNA-like channel 2D tif (for QC1)")
    ap.add_argument("--labels", required=True, help="nuclei_labels.tif")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--fg_dilate", type=int, default=3)
    ap.add_argument("--bg_inner", type=int, default=6)
    ap.add_argument("--bg_outer", type=int, default=20)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    intensity = read_tif_2d(args.intensity).astype(np.float32, copy=False)
    labels = tiff.imread(args.labels).astype(np.int32, copy=False)

    ring_cfg = RingConfig(
        fg_dilate_px=args.fg_dilate,
        bg_inner_px=args.bg_inner,
        bg_outer_px=args.bg_outer,
    )

    fg_union, bg_union, ring_stats = build_fg_bg_masks(labels, ring_cfg)
    fg_label_map, bg_label_map = per_cell_fg_bg(labels, fg_union, bg_union)

    stats = compute_per_cell_stats(intensity, labels, fg_label_map, bg_label_map)
    fig, summary = plot_qc1(stats, QC1Config(out_png=str(out_dir / "sqi_qc1_fg_bg_ratio.png")))

    fig.savefig(out_dir / "sqi_qc1_fg_bg_ratio.png")
    (out_dir / "sqi_qc1_summary.json").write_text(json.dumps({**ring_stats, **summary}, indent=2))

    # Optional debug outputs (handy in napari)
    tiff.imwrite(out_dir / "fg_union.tif", fg_union.astype(np.uint8) * 255, compression="zlib")
    tiff.imwrite(out_dir / "bg_union.tif", bg_union.astype(np.uint8) * 255, compression="zlib")

    print("Saved QC plot:", out_dir / "sqi_qc1_fg_bg_ratio.png")
    print("Saved summary:", out_dir / "sqi_qc1_summary.json")


if __name__ == "__main__":
    main()
