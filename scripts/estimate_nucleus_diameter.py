import numpy as np
import tifffile as tiff
from skimage.measure import regionprops

labels = tiff.imread(r"output\H1_seg\nuclei_labels.tif").astype(np.int32)

# compute equivalent diameter per nucleus
diams = []
for r in regionprops(labels):
    if r.area > 50:  # filter tiny debris
        diams.append(r.equivalent_diameter)

diams = np.array(diams)

print("N =", len(diams))
print("median diameter (px):", np.median(diams))
print("mean diameter (px):", diams.mean())
print("10–90% range (px):", np.percentile(diams, [10, 90]))
