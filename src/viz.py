import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa

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
