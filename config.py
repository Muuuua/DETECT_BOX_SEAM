"""集中可调参数。所有阈值/分辨率/角度在此一处修改。"""

# ---- 相机 ----
COLOR_WIDTH = 1920
COLOR_HEIGHT = 1080
COLOR_FPS = 30

DEPTH_WIDTH = 1280
DEPTH_HEIGHT = 720
DEPTH_FPS = 30
# D415 高精度档，降低深度噪声
DEPTH_VISUAL_PRESET = "high_accuracy"

# ---- 工作距离（米），用于 ROI 与有效深度过滤 ----
WORK_DISTANCE = (0.35, 0.45)
# 深度无效值上限（米），超过视为无回波
DEPTH_MAX_M = 1.5

# ---- 缝隙 2D 检测 ----
# 检测在降采样图上进行以提速，结果坐标再映射回全分辨率用于 deproject
DETECT_SCALE = 0.5

# CLAHE
CLAHE_CLIP = 2.0
CLAHE_TILE = (8, 8)

# Canny
CANNY_LOW = 30
CANNY_HIGH = 90

# 形态学闭运算核大小
MORPH_KERNEL = 5

# HoughLinesP
HOUGH_RHO = 1
HOUGH_THETA_DEG = 1.0  # 角度分辨率，代码中以 np.pi/180 构造
HOUGH_THRESHOLD = 50
HOUGH_MIN_LINE_LEN = 300   # 像素，缝隙应较长
HOUGH_MAX_LINE_GAP = 30

# 缝隙方向：固定为竖直方向。SEAM_VERTICAL_TOL_DEG 为相对竖直的最大夹角（度）
SEAM_VERTICAL_TOL_DEG = 10.0
# 缝隙方向约束：相对箱子长轴最大夹角（度）
MAX_LINE_ANGLE_DEG = 15.0
SEAM_ANGLE_TOL_DEG = 10.0
SEAM_CENTER_TOL_FRAC = 0.5
SEAM_LOCK_DIST_FRAC = 0.06
SEAM_LOCK_ANGLE_DEG = 8.0
# 时序 EMA 平滑系数（新检测权重）；越小越稳但越滞后
SEAM_SMOOTH_ALPHA = 0.3
BOX_MASK_ERODE = 15

# 边缘定位：沿骨架法向剖面采样数与搜索半宽（像素）
PROFILE_SAMPLES = 41
PROFILE_HALF_WIDTH = 25

# ---- 平面拟合 ----
RANSAC_RESIDUAL_MM = 2.0   # 内点阈值
RANSAC_MAX_TRIALS = 100
# 平面拟合 ROI 抽样步长（像素），越大越快越粗
PLANE_STRIDE = 4
# 平面拟合要求最少有效 3D 点数
MIN_PLANE_POINTS = 500

# ---- 几何量 ----
# 深度噪声阈值（米），|高度差| 低于此值标记 unreliable
DEPTH_NOISE_MM = 1.0
DEPTH_NOISE_M = DEPTH_NOISE_MM / 1000.0

# ---- 可视化 ----
ROI_MARGIN = 60  # 箱顶 ROI 相对画面边缘的内缩（像素）
# 显示窗口最大宽度（像素），超过则等比缩放以适应屏幕
DISPLAY_MAX_WIDTH = 1280

# ---- 箱子检测（基于深度）----
# 取有效深度的低端百分位作为“最近表面”参考深度
BOX_DEPTH_REF_PERCENTILE = 5
# 参考深度以下容差（米），略微包到最近边缘噪声
BOX_DEPTH_BEHIND_M = 0.02
# 参考深度以上容差（米），覆盖倾斜箱顶的厚度（35cm 箱子倾斜约 20° 时约 ±7cm）
BOX_DEPTH_TOL_M = 0.08
# 最近有效深度下限（米），滤掉无效 0
BOX_DEPTH_MIN_M = 0.1
BOX_OPEN_KERNEL = 5    # 开运算去噪
BOX_CLOSE_KERNEL = 15  # 闭运算填洞
# 箱子最小面积占画面比例，小于此视为未检出
MIN_BOX_AREA_FRAC = 0.05
# 深度标定文件（tune_depth.py 保存），存在则优先覆盖上面的默认值
DEPTH_CALIB_FILE = "depth_calib.json"

# ---- 旧 HSV 颜色检测参数（已弃用，保留兼容）----
HSV_LOWER = (20, 50, 80)
HSV_UPPER = (35, 200, 255)
HSV_CALIB_FILE = "hsv_calib.json"
    