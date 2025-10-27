from pathlib import Path
import numpy as np
from scipy.io import loadmat

def load_example(mat_path: str):
    data = loadmat(mat_path)
    keys = list(data.keys())

    # 去掉元数据键
    keys = [k for k in keys if not k.startswith("__")]
    # 找到带数字后缀的三类键
    amplitudes = next((k for k in keys if k.startswith("amplitudes")), None)
    distances = next((k for k in keys if k.startswith("distances")), None)
    cloud = next((k for k in keys if k.startswith("cloud")), None)

    # 按文件里的实际键名取
    A  = data[amplitudes]   # (424, 512) uint16
    D  = data[distances]    # (424, 512) float64
    PC = data[cloud]        # (424, 512, 3) float64

    # 有效点掩码（z 有效）
    valid_mask = np.isfinite(PC[..., 2]) & (PC[..., 2] != 0)

    return A, D, PC, valid_mask

def ensure_dir(path: str):
    Path(path).mkdir(parents=True, exist_ok=True)
