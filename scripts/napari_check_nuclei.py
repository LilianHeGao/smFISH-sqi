import napari
import numpy as np
import tifffile as tiff

from sqi.io.image_io import read_dapi_from_conv_zarr

# --- paths ---
ZARR = r"\\192.168.0.73\Papaya13\Sasha\20251105_6OHDA\H1\H1_PTBP1_TH_GFAP_set1\Conv_zscan1_074.zarr"
LABELS = r"output\H1_seg\nuclei_labels.tif"

# --- load data ---
dapi = read_dapi_from_conv_zarr(ZARR, channel=-1)
labels = tiff.imread(LABELS).astype(np.int32)

# --- visualize ---
viewer = napari.Viewer()
viewer.add_image(dapi, name="DAPI", contrast_limits=[0, np.percentile(dapi, 99.8)])
viewer.add_labels(labels, name="nuclei")
napari.run()
