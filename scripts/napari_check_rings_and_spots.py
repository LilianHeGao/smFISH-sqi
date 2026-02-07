import numpy as np
import tifffile as tiff

from sqi.io.image_io import read_dapi_from_conv_zarr
from sqi.qc.rings import CellProximalConfig, build_cell_proximal_and_distal_masks
from sqi.qc.qc_plots import visualize_nuclei_rings_and_spots_napari

from sqi.qc.mosaic_coords import (
    build_mosaic_and_coords, build_fov_anchor_index,
    fov_id_from_zarr_path, MosaicBuildConfig, mosaic_cache_paths,
)
from sqi.qc.valid_mask_mosaic import crop_valid_mask_for_fov, overlay_bbox_on_mosaic

# --------------------------------------------------
# PATHS (your dataset)
# --------------------------------------------------
ZARR = r"\\192.168.0.73\Papaya13\Sasha\20251105_6OHDA\H1\H1_PTBP1_TH_GFAP_set1\Conv_zscan1_074.zarr"
LABELS = r"output\H1_seg\nuclei_labels.tif"
SPOTS_NPY = r"output\H1_spots\spots_rc.npy"   # shape (N,2), row/col

DATA_FLD = r"M:\Sasha\20251105_6OHDA\H1\H1_PTBP1_TH_GFAP_set1"
CACHE_ROOT = r"\\192.168.0.73\Papaya13\Lilian\merfish_sqi_cache"

# --------------------------------------------------
# LOAD DATA
# --------------------------------------------------
print("Loading DAPI...")
dapi = read_dapi_from_conv_zarr(ZARR, channel=-1)

print("Loading nuclei labels...")
labels = tiff.imread(LABELS).astype(np.int32)

print("Loading spots...")
spots_rc = np.load(SPOTS_NPY).astype(np.float32)

print("Subsampling spots for Napari...")
MAX_SPOTS = 5000000
if spots_rc.shape[0] > MAX_SPOTS:
    idx = np.random.choice(spots_rc.shape[0], MAX_SPOTS, replace=False)
    spots_vis = spots_rc[idx]
else:
    spots_vis = spots_rc

# --------------------------------------------------
# MOSAIC + COORDS  (both from the SAME build)
# --------------------------------------------------
mosaic_cfg = MosaicBuildConfig(resc=4, icol=1, frame=20, rescz=2, force=False)
_, fls_, xs, ys = build_mosaic_and_coords(
    DATA_FLD, mosaic_cfg, cache_root=CACHE_ROOT, cache=True,
)

# Get the mosaic TIFF that build_mosaic_and_coords cached
# (this is the mosaic the coordinates refer to)
mosaic_tif, _ = mosaic_cache_paths(DATA_FLD, mosaic_cfg, CACHE_ROOT)

# --------------------------------------------------
# TISSUE MASK  (from the SAME mosaic, not a different one)
# --------------------------------------------------
import os
tissue_mask_tif = mosaic_tif.replace(".tiff", "_tissue_mask.tiff")

if os.path.exists(tissue_mask_tif):
    print("Loading tissue mask from:", tissue_mask_tif)
    global_valid = tiff.imread(tissue_mask_tif).astype(bool)
else:
    print("Building tissue mask from:", mosaic_tif)
    # import the QuPath-style builder
    import sys; sys.path.insert(0, os.path.dirname(__file__))
    from build_tissue_mask_qupath_style import build_tissue_mask_qupath_style

    mosaic_img = tiff.imread(mosaic_tif).astype(np.float32)
    global_valid = build_tissue_mask_qupath_style(mosaic_img, downsample=4)
    tiff.imwrite(tissue_mask_tif, global_valid.astype(np.uint8))
    print("Saved tissue mask:", tissue_mask_tif)

# --------------------------------------------------
# CROP VALID MASK FOR THIS FOV
# --------------------------------------------------
fov_index = build_fov_anchor_index(fls_, xs, ys)
fov_id = fov_id_from_zarr_path(ZARR)
anchor_xy = fov_index[fov_id]

# --- DEBUG ---
print("=" * 60)
print(f"[DEBUG] tissue mask shape  : {global_valid.shape}")
print(f"[DEBUG] FOV labels shape   : {labels.shape}")
print(f"[DEBUG] anchor (dim0,dim1) : {anchor_xy}")
print(f"[DEBUG] mosaic_resc        : {mosaic_cfg.resc}")
tile_h = labels.shape[0] // mosaic_cfg.resc
tile_w = labels.shape[1] // mosaic_cfg.resc
print(f"[DEBUG] expected tile size : ({tile_h}, {tile_w})")
r0 = round(anchor_xy[0] - tile_h / 2)
c0 = round(anchor_xy[1] - tile_w / 2)
print(f"[DEBUG] crop r0,c0        : ({r0}, {c0})")
print(f"[DEBUG] crop r1,c1        : ({r0 + tile_h}, {c0 + tile_w})")
print(f"[DEBUG] tissue mask True%  : {global_valid.mean():.4f}")
print("=" * 60)

valid_mask = crop_valid_mask_for_fov(
    global_valid_mask=global_valid,
    fov_anchor_xy=anchor_xy,
    fov_shape_hw=labels.shape,
    mosaic_resc=mosaic_cfg.resc,
    anchor_is_upper_left=False,
)
print(f"[DEBUG] cropped mask True% : {valid_mask.mean():.4f}")

# --------------------------------------------------
# BUILD RINGS
# --------------------------------------------------
cp_cfg = CellProximalConfig(cell_proximal_px=24)

cell_proximal, cell_distal, stats = build_cell_proximal_and_distal_masks(
    nuclei_labels=labels,
    valid_mask=valid_mask,
    cfg=cp_cfg,
)

print("Region stats:", stats)

# --------------------------------------------------
# VISUALIZE
# --------------------------------------------------
visualize_nuclei_rings_and_spots_napari(
    dapi=dapi,
    nuclei_labels=labels,
    cell_proximal=cell_proximal,
    cell_distal=cell_distal,
    spots_rc=spots_vis,
    valid_mask=valid_mask,
)
