import numpy as np

# 平面用 (n, d) 表示，其中 n 为单位法向量，满足 n·x + d = 0
def plane_from_3pts(p1, p2, p3):
    v1 = p2 - p1
    v2 = p3 - p1
    n = np.cross(v1, v2)
    norm = np.linalg.norm(n)
    if norm < 1e-9:
        return None, None
    n = n / norm
    d = -np.dot(n, p1)
    return n, d

def point_plane_signed_distance(pts, n, d):
    # pts: (..., 3)
    return (pts @ n + d)

def plane_plane_distance(n1, d1, n2, d2):
    # 两平面法向量方向一致时，平行距离 = |d2 - d1|
    # 若方向相反，取反一个法向量保证一致
    if np.dot(n1, n2) < 0:
        n2, d2 = -n2, -d2
    # 夹角不近似 0 时要先校准，这里假设已找到同向主平面
    return abs(d2 - d1)
