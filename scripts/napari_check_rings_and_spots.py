import numpy as np
import tifffile as tiff

from sqi.io.image_io import read_dapi_from_conv_zarr
from sqi.qc.rings import CellProximalConfig, build_cell_proximal_and_distal_masks
from sqi.qc.qc_plots import visualize_nuclei_rings_and_spots_napari

from sqi.qc.mosaic_coords import build_mosaic_and_coords, build_fov_anchor_index, fov_id_from_zarr_path, MosaicBuildConfig
from sqi.qc.valid_mask_mosaic import crop_valid_mask_for_fov, overlay_bbox_on_mosaic

# --------------------------------------------------
# PATHS (your dataset)
# --------------------------------------------------
ZARR = r"\\192.168.0.73\Papaya13\Sasha\20251105_6OHDA\H1\H1_PTBP1_TH_GFAP_set1\Conv_zscan1_074.zarr"
LABELS = r"output\H1_seg\nuclei_labels.tif"
MOSAIC_TIF = r"\\192.168.0.73\Papaya13\Sasha\20251105_6OHDA\H1\H1mosaics\H1_PTBP1_TH_GFAP_set1_middle15_col-1.tiff"
MOSAIC_VALID_MASK_TIF = r"\\192.168.0.73\Papaya13\Lilian\merfish_sqi_cache\H1_PTBP1_TH_GFAP_set1_middle15_col-1_tissue_mask.tiff"

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

print("Subsampling spots for Napari...")
MAX_SPOTS = 5000000 #safe for Napari

if spots_rc.shape[0] > MAX_SPOTS:
    idx = np.random.choice(spots_rc.shape[0], MAX_SPOTS, replace=False)
    spots_vis = spots_rc[idx]
else:
    spots_vis = spots_rc

# --------------------------------------------------
# BUILD RINGS
# --------------------------------------------------
CACHE_ROOT = r"\\192.168.0.73\Papaya13\Lilian\merfish_sqi_cache"

print("Loading tissue mask from:", MOSAIC_VALID_MASK_TIF)
global_valid = tiff.imread(MOSAIC_VALID_MASK_TIF).astype(bool)

# TODO: crop global_valid to this FOV using anchor coords (next step)

DATA_FLD = r"M:\Sasha\20251105_6OHDA\H1\H1_PTBP1_TH_GFAP_set1"

mosaic_cfg = MosaicBuildConfig(resc=4, icol=1, frame=20, rescz=2, force=False)
_, fls_, xs, ys = build_mosaic_and_coords(
    DATA_FLD,
    mosaic_cfg,
    cache_root=CACHE_ROOT,
    cache=True,
)

# --- Crop valid mask for this FOV ---
fov_index = build_fov_anchor_index(fls_, xs, ys)
fov_id = fov_id_from_zarr_path(ZARR)
anchor_xy = fov_index[fov_id]

valid_mask = crop_valid_mask_for_fov(
    global_valid_mask=global_valid,
    fov_anchor_xy=anchor_xy,
    fov_shape_hw=labels.shape,
    mosaic_resc=1,              # IMPORTANT: mask is already full-res
    anchor_is_upper_left=False,
)


cp_cfg = CellProximalConfig(cell_proximal_px=24)

cell_proximal, cell_distal, stats = build_cell_proximal_and_distal_masks(
    nuclei_labels=labels,
    valid_mask=valid_mask,
    cfg=cp_cfg,
)

print("Region stats:", stats)

# ---- DEBUG: visualize FOV bbox on mosaic (run once) ----
overlay_bbox_on_mosaic(
    mosaic_img=global_valid.astype(float),
    fov_anchor_xy=anchor_xy,
    fov_shape_hw=labels.shape,
    mosaic_resc=mosaic_cfg.resc,
    anchor_is_upper_left=False,
)

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

