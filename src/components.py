import numpy as np
from scipy.ndimage import label

def largest_component(mask):
    lbl, n = label(mask)
    if n == 0:
        return mask, 0
    counts = np.bincount(lbl.ravel())
    counts[0] = 0  # 背景忽略
    k = counts.argmax()
    return (lbl == k), k
