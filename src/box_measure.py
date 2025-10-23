import numpy as np
from .geometry import plane_plane_distance

def box_height(n_floor, d_floor, n_top, d_top):
    return plane_plane_distance(n_floor, d_floor, n_top, d_top)

def box_length_width(PC, box_mask):
    pts = PC[box_mask]  # (N,3)
    from .corners import length_width_from_top_points
    L, W = length_width_from_top_points(pts, axis="auto")
    return L, W
