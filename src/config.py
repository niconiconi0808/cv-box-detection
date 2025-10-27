DATA_DIR = "data"
OUT_DIR = "outputs"

# RANSAC 参数
RANSAC_THRESH = 0.01      # 点到平面的距离阈值
RANSAC_MAX_ITERS = 1000

# 形态学参数
MORPH_OPEN_SIZE = 3
MORPH_CLOSE_SIZE = 5

# 可视化采样（3D散点太多会卡）
PC_SCATTER_SUBSAMPLE = 5  # 每隔多少像素取一个点
