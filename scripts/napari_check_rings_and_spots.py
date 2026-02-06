import numpy as np
import tifffile as tiff

from sqi.io.image_io import read_dapi_from_conv_zarr
from sqi.qc.rings import RingConfig, build_fg_bg_masks
from sqi.qc.qc_plots import visualize_nuclei_rings_and_spots_napari

# --------------------------------------------------
# PATHS (your dataset)
# --------------------------------------------------
ZARR = r"\\192.168.0.73\Papaya13\Sasha\20251105_6OHDA\H1\H1_PTBP1_TH_GFAP_set1\Conv_zscan1_074.zarr"
LABELS = r"output\H1_seg\nuclei_labels.tif"

# TODO: update this to your actual spot file
SPOTS_NPY = r"output\H1_spots\spots_rc.npy"   # shape (N,2), row/col

# --------------------------------------------------
# LOAD DATA
# --------------------------------------------------
print("Loading DAPI...")
dapi = read_dapi_from_conv_zarr(ZARR, channel=-1)

print("Loading nuclei labels...")
labels = tiff.imread(LABELS).astype(np.int32)

print("Loading spots...")
spots_rc = np.load(SPOTS_NPY).astype(np.float32)

# --------------------------------------------------
# BUILD RINGS
# --------------------------------------------------
cfg = RingConfig(
    fg_dilate_px=3,
    bg_inner_px=6,
    bg_outer_px=20,
)

fg_union, bg_union, stats = build_fg_bg_masks(labels, cfg)
print("Ring stats:", stats)

# --------------------------------------------------
# VISUALIZE
# --------------------------------------------------
visualize_nuclei_rings_and_spots_napari(
    dapi=dapi,
    nuclei_labels=labels,
    fg_union=fg_union,
    bg_union=bg_union,
    spots_rc=spots_rc,
    spot_size=4.0,
)
