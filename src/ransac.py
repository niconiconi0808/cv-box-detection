import numpy as np
from .geometry import plane_from_3pts, point_plane_signed_distance

def ransac_plane(PC, valid_mask, thresh=0.01, max_iters=1000, rng=None):
    """
    在点云上拟合主平面
    返回: n, d, inlier_mask
    """
    if rng is None:
        rng = np.random.default_rng(42)

    H, W, _ = PC.shape
    pts = PC.reshape(-1, 3)
    vm = valid_mask.reshape(-1)
    idxs = np.flatnonzero(vm)
    if len(idxs) < 3:
        raise ValueError("有效点不足以拟合平面")

    best_inliers = None
    best_count = -1
    best_model = (None, None)

    for _ in range(max_iters):
        # 随机抽 3 点
        sample = rng.choice(idxs, size=3, replace=False)
        p1, p2, p3 = pts[sample]
        n, d = plane_from_3pts(p1, p2, p3)
        if n is None:
            continue

        # 距离阈值内的内点
        dist = np.abs(point_plane_signed_distance(pts, n, d))
        inliers = (dist < thresh) & vm
        count = inliers.sum()
        if count > best_count:
            best_count = count
            best_inliers = inliers
            best_model = (n, d)

        # 如果几乎所有有效点都是内点可提前结束
        if count > 0.9 * vm.sum(): break

    n, d = best_model
    inlier_mask = best_inliers.reshape(H, W)
    return n, d, inlier_mask
