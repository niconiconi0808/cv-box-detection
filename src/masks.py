import numpy as np
from scipy.ndimage import binary_opening, binary_closing, generate_binary_structure

def clean_mask(mask, open_size=3, close_size=5):
    # 3x3/5x5 结构元素
    selem_o = generate_binary_structure(2, 1)
    selem_c = generate_binary_structure(2, 1)

    m = binary_opening(mask, iterations=max(1, open_size//2), structure=selem_o)
    m = binary_closing(m, iterations=max(1, close_size//2), structure=selem_c)
    return m
