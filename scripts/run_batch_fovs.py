"""
Batch runner: auto-discover FOV zarrs, randomly select N,
run the SQI pipeline on each, and generate a tissue overview
with all selected FOVs highlighted.

Called by run_batch_fovs.bat.
"""
import argparse
import glob
import json
import os
import random
import subprocess
import sys
from pathlib import Path

import numpy as np
import tifffile as tiff

from sqi.io.image_io import read_multichannel_from_conv_zarr
from sqi.qc.mosaic_coords import (
    build_mosaic_and_coords, build_fov_anchor_index, lookup_fov_anchor,
    fov_id_from_zarr_path, MosaicBuildConfig, mosaic_cache_paths,
)
from sqi.qc.qc_plots import plot_tissue_overview


def discover_fov_zarrs(data_fld: str) -> list[str]:
    pattern = os.path.join(data_fld, "Conv_zscan*.zarr")
    return sorted(glob.glob(pattern))


def get_or_build_tissue_mask(mosaic_tif: str, cache_root: str) -> np.ndarray:
    mask_tif = mosaic_tif.replace(".tiff", "_tissue_mask.tiff")
    if os.path.exists(mask_tif):
        return tiff.imread(mask_tif).astype(bool)
    from build_tissue_mask_qupath_style import build_tissue_mask_qupath_style
    mosaic = tiff.imread(mosaic_tif).astype(np.float32)
    mask = build_tissue_mask_qupath_style(mosaic, downsample=4)
    tiff.imwrite(mask_tif, mask.astype(np.uint8))
    print("[INFO] Saved tissue mask:", mask_tif)
    return mask


def get_fov_shape_from_zarr(zarr_path: str, resc: int) -> tuple[int, int]:
    """Read one zarr to get the FOV shape at full resolution."""
    im = read_multichannel_from_conv_zarr(zarr_path)
    # im shape: (C, Z, Y, X) or (C, Y, X)
    h, w = im.shape[-2], im.shape[-1]
    return (h, w)


def compute_fov_bbox_mosaic(
    anchor_xy: tuple[float, float],
    fov_shape_hw: tuple[int, int],
    resc: int,
) -> tuple[int, int, int, int]:
    """Compute (r0, c0, r1, c1) in mosaic coords from center anchor."""
    a0, a1 = anchor_xy
    hf = fov_shape_hw[0] // (2 * resc)
    wf = fov_shape_hw[1] // (2 * resc)
    r0 = int(round(a0 - hf))
    c0 = int(round(a1 - wf))
    r1 = int(round(a0 + hf))
    c1 = int(round(a1 + wf))
    return (r0, c0, r1, c1)


def main():
    parser = argparse.ArgumentParser(
        description="Batch SQI: auto-discover FOVs, run pipeline, generate tissue overview",
    )
    parser.add_argument("--data_fld", required=True,
                        help="Folder containing Conv_zscan*.zarr files")
    parser.add_argument("--cache_root", required=True)
    parser.add_argument("--out_root", required=True)
    parser.add_argument("--n_fovs", type=int, default=10,
                        help="Number of FOVs to randomly select (default: 10)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resc", type=int, default=4,
                        help="Mosaic rescale factor (default: 4)")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    random.seed(args.seed)
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    # --------------------------------------------------
    # 1. Discover and select FOVs
    # --------------------------------------------------
    all_zarrs = discover_fov_zarrs(args.data_fld)
    if not all_zarrs:
        print(f"[ERROR] No Conv_zscan*.zarr found in {args.data_fld}")
        sys.exit(1)

    n_pick = min(args.n_fovs, len(all_zarrs))
    selected = random.sample(all_zarrs, n_pick)
    selected_ids = [fov_id_from_zarr_path(z) for z in selected]

    print("=" * 60)
    print(f"Selected {n_pick} / {len(all_zarrs)} FOVs:")
    for fov_zarr, fid in zip(selected, selected_ids):
        print(f"  {fid}  ({os.path.basename(fov_zarr)})")
    print("=" * 60)

    # --------------------------------------------------
    # 2. Run pipeline on each FOV
    # --------------------------------------------------
    failed = []
    for i, (fov_zarr, fid) in enumerate(zip(selected, selected_ids)):
        print(f"\n[{i+1}/{n_pick}] Processing FOV {fid} ...")
        cmd = [
            sys.executable, "scripts/run_sqi_from_fov_zarr.py",
            "--fov_zarr", fov_zarr,
            "--data_fld", args.data_fld,
            "--cache_root", args.cache_root,
            "--out_root", args.out_root,
            "--resc", str(args.resc),
        ]
        if args.force:
            cmd.append("--force")

        result = subprocess.run(cmd)
        if result.returncode != 0:
            print(f"[ERROR] FOV {fid} failed!")
            failed.append(fid)
        else:
            print(f"[OK] FOV {fid} done.")

    # --------------------------------------------------
    # 3. Generate tissue overview with all selected FOVs
    # --------------------------------------------------
    print("\n[POST] Generating tissue overview ...")

    import matplotlib
    matplotlib.use("Agg")

    mosaic_cfg = MosaicBuildConfig(resc=args.resc)
    _, fls_, xs, ys = build_mosaic_and_coords(
        args.data_fld, mosaic_cfg, cache_root=args.cache_root, cache=True,
    )
    fov_index = build_fov_anchor_index(fls_, xs, ys)

    mosaic_tif, _ = mosaic_cache_paths(args.data_fld, mosaic_cfg, args.cache_root)
    tissue_mask = get_or_build_tissue_mask(mosaic_tif, args.cache_root)

    # Get FOV shape from the first selected zarr
    fov_shape = get_fov_shape_from_zarr(selected[0], args.resc)

    bboxes = []
    labels = []
    for fov_zarr, fid in zip(selected, selected_ids):
        try:
            anchor = lookup_fov_anchor(fov_index, fid)
            bbox = compute_fov_bbox_mosaic(anchor, fov_shape, args.resc)
            bboxes.append(bbox)
            labels.append(fid)
        except KeyError as e:
            print(f"  [WARN] Skipping FOV {fid} in overview: {e}")

    dataset_name = os.path.basename(args.data_fld)
    plot_tissue_overview(
        tissue_mask, bboxes, labels,
        title=f"Tissue overview — {dataset_name} ({n_pick} FOVs)",
        out_path=str(out_root / "tissue_overview.png"),
    )

    # --------------------------------------------------
    # 4. Summary
    # --------------------------------------------------
    print("\n" + "=" * 60)
    print(f"Batch complete: {n_pick - len(failed)}/{n_pick} succeeded")
    if failed:
        print(f"  Failed: {failed}")
    print(f"  Output: {out_root}")
    print(f"  Tissue overview: {out_root / 'tissue_overview.png'}")
    print("=" * 60)

    # Collect AUCs from summaries
    aucs = {}
    for fid in selected_ids:
        summary_path = out_root / fid / "sqi_summary.json"
        if summary_path.exists():
            with open(summary_path) as f:
                data = json.load(f)
            aucs[fid] = data.get("sanity_auc")

    if aucs:
        print("\nSanity-check AUCs:")
        for fid, auc in aucs.items():
            status = "OK" if auc is not None and auc >= 0.6 else "WARN"
            auc_str = f"{auc:.3f}" if auc is not None else "N/A"
            print(f"  FOV {fid}: AUC={auc_str}  [{status}]")


if __name__ == "__main__":
    main()
