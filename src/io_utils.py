from pathlib import Path
import numpy as np
from scipy.io import loadmat

# ---- 工具函数：把 mat 里可能的 MATLAB struct 展开成纯 dict ----
def _unwrap_mat_struct(obj):
    # scipy.loadmat(squeeze_me=True, simplify_cells=True) 已经很友好，
    # 但有时变量会包在一层 dict/struct 里，这里递归展开。
    if isinstance(obj, dict):
        out = {}
        for k, v in obj.items():
            out[k] = _unwrap_mat_struct(v)
        return out
    else:
        return obj

def _collect_arrays(d):
    """扁平收集所有 ndarray；如果有嵌套 dict/struct，递归进入"""
    arrays = {}
    def rec(prefix, val):
        if isinstance(val, dict):
            for k, v in val.items():
                rec(f"{prefix}.{k}" if prefix else k, v)
        elif isinstance(val, np.ndarray):
            arrays[prefix] = val
    rec("", d)
    return arrays

def _looks_like_A(arr):
    return arr.ndim == 2 or (arr.ndim == 3 and arr.shape[-1] == 1)

def _looks_like_D(arr):
    return arr.ndim == 2 or (arr.ndim == 3 and arr.shape[-1] == 1)

def _looks_like_PC(arr):
    return arr.ndim == 3 and arr.shape[-1] == 3

# 常见别名（全部小写比较）
_A_ALIASES = {"a","amp","amplitude","toa","tofamp"}
_D_ALIASES = {"d","depth","dist","distance","range","z","kinectdepth"}
_PC_ALIASES = {"pc","pointcloud","points","xyz","cloud","xyzmap"}

def load_example(mat_path: str):
    """
    智能加载 .mat：自动识别 A/D/PC 变量并返回 (A, D, PC, valid_mask)
    - A, D: (H,W)
    - PC:   (H,W,3)
    - valid: PC[...,2] != 0
    """
    m = loadmat(mat_path, squeeze_me=True, simplify_cells=True)
    m = {k:v for k,v in m.items() if not k.startswith("__")}  # 去掉 meta 键
    m = _unwrap_mat_struct(m)
    arrays = _collect_arrays(m)

    # 先尝试按名字匹配
    A = D = PC = None
    for k, arr in arrays.items():
        lk = k.split(".")[-1].lower()
        if A is None and lk in _A_ALIASES and _looks_like_A(arr): A = arr
        if D is None and lk in _D_ALIASES and _looks_like_D(arr): D = arr
        if PC is None and lk in _PC_ALIASES and _looks_like_PC(arr): PC = arr

    # 再按形状猜（防止名字完全对不上）
    if PC is None:
        cands = [(k,a) for k,a in arrays.items() if _looks_like_PC(a)]
        if len(cands) == 1:
            PC = cands[0][1]
        elif len(cands) > 1:
            # 取像素最多的那个
            PC = max(cands, key=lambda kv: kv[1].shape[0]*kv[1].shape[1])[1]

    if A is None:
        cands = [(k,a) for k,a in arrays.items() if _looks_like_A(a)]
        if len(cands) >= 1:
            # 优先与 PC 形状匹配的
            if PC is not None:
                H,W,_ = PC.shape
                fits = [a for _,a in cands if (a.shape[:2] == (H,W)) or
                        (a.ndim==3 and a.shape[:2]==(H,W) and a.shape[-1]==1)]
                A = (fits[0] if fits else cands[0][1])
            else:
                A = cands[0][1]

    if D is None:
        cands = [(k,a) for k,a in arrays.items() if _looks_like_D(a)]
        if len(cands) >= 1:
            if PC is not None:
                H,W,_ = PC.shape
                fits = [a for _,a in cands if (a.shape[:2] == (H,W)) or
                        (a.ndim==3 and a.shape[:2]==(H,W) and a.shape[-1]==1)]
                D = (fits[0] if fits else cands[0][1])
            else:
                D = cands[0][1]

    # 规范形状
    def _squeeze2(a):
        if a is None: return None
        if a.ndim==3 and a.shape[-1]==1: a = a[...,0]
        return a

    A = _squeeze2(A)
    D = _squeeze2(D)

    # 最终检查
    if PC is None or A is None or D is None:
        # 打印可用键与形状，方便你排查
        summary = "\n".join([f"- {k}: shape={v.shape}, dtype={v.dtype}" for k,v in arrays.items()])
        raise ValueError(
            "未能在 MAT 中自动识别出 A/D/PC。\n"
            "请看下面的变量清单，并告诉我对应的是哪些：\n" + summary
        )

    # 有些数据集的 invalid 会是 0 或 NaN，这里都处理
    valid_mask = np.isfinite(PC[...,2]) & (PC[...,2] != 0)
    return A, D, PC, valid_mask

def ensure_dir(path: str):
    Path(path).mkdir(parents=True, exist_ok=True)
