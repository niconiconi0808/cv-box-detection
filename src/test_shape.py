from pathlib import Path
import numpy as np

def is_v73_mat(path: str) -> bool:
    with open(path, "rb") as f:
        head = f.read(128)
    return b"MATLAB 7.3" in head

def inspect_mat(path: str):
    print("File:", path)
    if is_v73_mat(path):
        print("-> Detected MATLAB v7.3 (HDF5). Use h5py to read.")
        import h5py
        with h5py.File(path, "r") as f:
            def walk(g, prefix=""):
                for k, v in g.items():
                    p = f"{prefix}/{k}"
                    if isinstance(v, h5py.Group):
                        walk(v, p)
                    else:
                        print(f"[DATASET] {p} shape={v.shape} dtype={v.dtype}")
            walk(f, "")
    else:
        print("-> Detected legacy MAT (v7.2 or lower). Use scipy.io.loadmat.")
        from scipy.io import loadmat
        m = loadmat(path, squeeze_me=True, simplify_cells=True)
        print("Top-level keys:", [k for k in m.keys() if not k.startswith("__")])
        for k, v in m.items():
            if k.startswith("__"):
                continue
            if isinstance(v, np.ndarray):
                print(f"[NDARRAY] {k:20s} shape={v.shape} dtype={v.dtype}")
            elif isinstance(v, dict):
                print(f"[DICT]    {k:20s} (nested struct) has keys: {list(v.keys())[:8]} ...")
            else:
                # 其他类型就简略打印
                t = type(v).__name__
                print(f"[{t.upper():7s}] {k}")

inspect_mat("D:\PycharmProjects\cv-box-detection\data\example1kinect.mat")
inspect_mat("D:\PycharmProjects\cv-box-detection\data\example2kinect.mat")
inspect_mat("D:\PycharmProjects\cv-box-detection\data\example3kinect.mat")