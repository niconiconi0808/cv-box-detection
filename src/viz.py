import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa
from scipy import ndimage as ndi
def show_image(img, title="", cmap="gray"):
    plt.figure()
    plt.imshow(img, cmap=cmap)
    plt.title(title)
    plt.axis("off")

def show_mask(mask, title="mask"):
    show_image(mask.astype(np.uint8)*255, title)

def scatter_pc(PC, valid_mask=None, step=5, title="point cloud"):
    H, W, _ = PC.shape
    ii, jj = np.mgrid[0:H:step, 0:W:step]
    pts = PC[ii, jj].reshape(-1, 3)
    if valid_mask is not None:
        vm = valid_mask[ii, jj].reshape(-1)
        pts = pts[vm]

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(pts[:,0], pts[:,1], pts[:,2], s=1)
    ax.set_title(title)
    ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("z")

def visualize_box_scene(base_img, floor_mask, box_top_mask, valid_mask=None):
    H, W = floor_mask.shape
    if valid_mask is None:
        valid_mask = np.ones((H, W), dtype=bool)

    # 计算三个分区
    floor = floor_mask & valid_mask
    top   = box_top_mask & valid_mask
    other = (~floor_mask) & (~box_top_mask) & valid_mask

    # 构造 RGB 画布
    rgb = np.zeros((H, W, 3), dtype=np.float32)
    rgb[..., :] = 0.2  # invalid / 背景暗灰
    rgb[floor] = [0.55, 0.95, 0.55]  # 绿色
    rgb[other] = [0.15, 0.25, 0.85]  # 蓝色
    rgb[top]   = [0.85, 0.15, 0.15]  # 红色

    # 叠加盒顶边框（细红线）
    edge = ndi.binary_dilation(top, iterations=1) ^ top
    rgb[edge] = [1.0, 0.1, 0.1]

    plt.figure(figsize=(7,5))
    if base_img is not None:
        # 背景做一点对比度（透明度）叠加，便于看细节
        base = np.asarray(base_img, dtype=np.float32)
        base = np.nan_to_num(base)
        if base.ndim == 3 and base.shape[-1] == 1:
            base = base[...,0]
        base = (base - np.percentile(base, 2)) / (np.percentile(base, 98) - np.percentile(base, 2) + 1e-6)
        base = np.clip(base, 0, 1)
        base_rgb = np.stack([base, base, base], axis=-1)
        show = 0.35 * base_rgb + 0.75 * rgb  # 混合
    else:
        show = rgb

    plt.imshow(show, interpolation="nearest")
    plt.axis("off")

    # 根据盒顶 mask 放置四个方向标签
    if top.any():
        ys, xs = np.where(top)
        y0, y1 = int(ys.min()), int(ys.max())
        x0, x1 = int(xs.min()), int(xs.max())
        pad = max(5, (x1 - x0) // 20)  # 边距
        # top/left/right/bottom 文字
        plt.text((x0+x1)//2, max(y0 - 2*pad, 0), "top", color="k", ha="center", va="bottom", fontsize=10, weight="bold")
        plt.text((x0+x1)//2, min(y1 + 2*pad, H-1), "bottom", color="k", ha="center", va="top", fontsize=10, weight="bold")
        plt.text(max(x0 - 2*pad, 0), (y0+y1)//2, "left", color="k", ha="right", va="center", fontsize=10, weight="bold")
        plt.text(min(x1 + 2*pad, W-1), (y0+y1)//2, "right", color="k", ha="left", va="center", fontsize=10, weight="bold")

