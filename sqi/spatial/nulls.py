import numpy as np

def randomize_spots_uniform(spots_rc, valid_mask):
    """
    Randomize spot locations uniformly within valid_mask.
    Preserves spot count.
    """
    coords = np.column_stack(np.where(valid_mask))
    idx = np.random.choice(len(coords), size=len(spots_rc), replace=True)
    return coords[idx]
