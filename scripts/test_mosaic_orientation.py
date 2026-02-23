"""
Test mosaic tile orientation before running the full batch.

Builds 4 small mosaics (rot_k=0,1,2,3) from a subset of FOVs
and saves a 2x2 comparison figure.

Usage
-----
python scripts/test_mosaic_orientation.py \
    --data_fld "\\\\192.168.0.116\\durian3\\Lilian\\..." \
    --cache_root "\\\\server\\cache" \
    --out orientation_test.png

Optional:
    --n_tiles 80       (default: use up to 30 tiles for speed)
    --resc 4           (mosaic downsample factor)
    --icol 1           (channel index to visualise)
    --frame 20         (z-frame index, or 'all' for max projection)
"""
import argparse
import glob
import os

import numpy as np
from tqdm import tqdm

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sqi.io.image_io import read_multichannel_from_conv_zarr
from sqi.qc.mosaic_coords import compose_mosaic


def build_quick_mosaic(ims_raw, xs_um, ys_um, resc, rot_k):
    """Apply a specific rot_k, then T[::-1,::-1], and compose."""
    rotated = [np.rot90(im, k=rot_k) for im in ims_raw]
    tiles = [im.T[::-1, ::-1] for im in rotated]
    um_per_pix = 0.1083333 * resc
    mosaic, xs, ys = compose_mosaic(
        tiles, xs_um, ys_um,
        ims_c=None, um_per_pix=um_per_pix, rot=0, return_coords=True,
    )
    return mosaic, xs, ys


def main():
    parser = argparse.ArgumentParser(
        description="Test mosaic orientation (rot_k=0..3)",
    )
    parser.add_argument("--data_fld", required=True)
    parser.add_argument("--cache_root", default=None,
                        help="Not used for computation, only for consistency")
    parser.add_argument("--out", default="orientation_test.png",
                        help="Output figure path (default: orientation_test.png)")
    parser.add_argument("--n_tiles", type=int, default=80,
                        help="Max tiles to use (default: 30, for speed)")
    parser.add_argument("--resc", type=int, default=4)
    parser.add_argument("--icol", type=int, default=1)
    parser.add_argument("--frame", default="20")
    parser.add_argument("--rescz", type=int, default=2)
    args = parser.parse_args()

    icol = args.icol
    resc = args.resc
    rescz = args.rescz
    frame = args.frame

    # --------------------------------------------------
    # 1. Read tiles
    # --------------------------------------------------
    fls_ = sorted(glob.glob(os.path.join(args.data_fld, "Conv_zscan*.zarr")))
    if not fls_:
        print(f"[ERROR] No Conv_zscan*.zarr found in {args.data_fld}")
        return

    fls_ = fls_[:args.n_tiles]
    print(f"Reading {len(fls_)} tiles ...")

    ims_raw = []
    xs_um, ys_um = [], []
    for fl in tqdm(fls_):
        try:
            im, x, y = read_multichannel_from_conv_zarr(fl, return_pos=True)
            if str(frame).lower() == "all":
                tile = np.array(
                    np.max(im[icol][::rescz, ::resc, ::resc][:, ::-1], axis=0),
                    dtype=np.float32,
                )
            else:
                tile = np.array(
                    im[icol][int(frame), ::resc, ::resc][:, ::-1],
                    dtype=np.float32,
                )
            ims_raw.append(tile)
            xs_um.append(x)
            ys_um.append(y)
        except Exception as e:
            print(f"  skip {os.path.basename(fl)}: {e}")

    if not ims_raw:
        print("[ERROR] No tiles loaded.")
        return

    print(f"Loaded {len(ims_raw)} tiles, tile shape={ims_raw[0].shape}")

    # --------------------------------------------------
    # 2. Build mosaics for k=0,1,2,3
    # --------------------------------------------------
    fig, axes = plt.subplots(2, 2, figsize=(16, 16))

    for k, ax in zip(range(4), axes.flat):
        print(f"  Building mosaic rot_k={k} ...")
        mosaic, xs, ys = build_quick_mosaic(ims_raw, xs_um, ys_um, resc, rot_k=k)

        # Display with reference-style contrast (exclude zeros)
        ds = 3
        img = mosaic[::ds, ::ds]
        nonzero = img[img > 0]
        if len(nonzero) > 0:
            vmin = np.percentile(nonzero, 1)
            vmax = np.percentile(nonzero, 99)
        else:
            vmin, vmax = 0, 1

        ax.imshow(img, cmap="gray", vmin=vmin, vmax=vmax)

        # Annotate FOV ids
        # xs = column centers, ys = row centers from compose_mosaic
        # ax.text(x, y) expects (col, row) for imshow
        for x_, y_, fl_ in zip(xs, ys, fls_):
            fov_id = os.path.basename(fl_).split("_")[-1].split(".")[0]
            ax.text(x_ / ds, y_ / ds, fov_id, color="red", fontsize=4,
                    ha="center", va="center")

        ax.set_title(f"rot_k={k}", fontsize=14)
        ax.set_xticks([])
        ax.set_yticks([])

    fig.suptitle(f"Mosaic orientation test — {os.path.basename(args.data_fld)}\n"
                 f"({len(ims_raw)} tiles, resc={resc}, icol={icol}, frame={frame})",
                 fontsize=13)
    fig.tight_layout()
    out_path = os.path.abspath(args.out)
    out_dir = os.path.dirname(out_path)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"\nSaved: {out_path}")
    print("Pick the rot_k value where FOV labels match their spatial positions,")
    print("then pass --rot_k <value> to run_batch_fovs.py.")


if __name__ == "__main__":
    main()
