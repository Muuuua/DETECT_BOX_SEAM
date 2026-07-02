# D415 纸箱封箱缝检测

使用 Intel RealSense D415 相机在 0.35–0.45 m 工作距离下，实时检测 35 cm 纸箱顶部沿长边的封箱胶带缝隙，输出：

- 缝隙两端点 3D 坐标（相机坐标系，米）
- 缝隙长度（mm）
- 缝隙宽度（mm，及像素宽度）
- 胶带面相对箱顶平面的高度差（mm，噪声量级时标注 `unreliable`）

并在视频流上叠加标注。

## 依赖

### 硬件

| 项目 | 要求 |
|------|------|
| 相机 | Intel RealSense **D415** |
| 工作距离 | 0.35–0.45 m（见 `config.py` 中 `WORK_DISTANCE`） |
| 连接 | USB 3.0 |

### 系统（需单独安装，不在本仓库内）

| 依赖 | 说明 |
|------|------|
| [Intel RealSense SDK 2.0](https://github.com/IntelRealSense/librealsense/releases) | 相机驱动与 `pyrealsense2` 运行所需底层库（Windows 安装 `.exe` 即可） |
| Python | **3.10** 推荐（3.9–3.11 一般可用） |

> 本仓库**仅含应用源码**，不包含 `librealsense` SDK 源码。克隆后请在本机单独安装 RealSense SDK。

### Python 包

| 包 | 用途 |
|----|------|
| `pyrealsense2` | D415 采集、深度对齐、deproject |
| `opencv-python` | 图像处理、Canny/Hough、窗口显示 |
| `numpy` | 数组与几何计算 |
| `scikit-learn` | RANSAC 平面拟合 |

清单文件：

- **Conda（推荐）**：`environment.yml`
- **pip / venv**：`requirements.txt`

## 环境安装

### 方式 A：Conda（推荐）

```bash
git clone https://github.com/Muuuua/DETECT_BOX_SEAM.git
cd DETECT_BOX_SEAM
conda env create -f environment.yml
conda activate box-seam
```

### 方式 B：pip + 虚拟环境

```bash
git clone https://github.com/Muuuua/DETECT_BOX_SEAM.git
cd DETECT_BOX_SEAM
python -m venv .venv
# Windows:
.venv\Scripts\activate
# Linux/macOS:
# source .venv/bin/activate
pip install -r requirements.txt
```

### 验证依赖

```bash
python -c "import pyrealsense2, cv2, numpy, sklearn; print('ok')"
```

## 运行

```bash
# 1. 标定箱子深度检测参数（对着箱子调滑块，让绿框稳定包住箱子顶面）
python tune_depth.py

# 2. 正式检测
python main.py

# 3. 仅箱子检测（不检测缝隙）
python main_boxonly.py

# 4. 透明胶带颜色加权测试（独立脚本，含热力图）
python test_seam_color.py
```

ESC 退出。`tune_depth`：`s` 保存、`q`/ESC 退出。

## 文件结构

| 文件 | 作用 |
|------|------|
| `config.py` | 所有可调参数集中处 |
| `camera.py` | D415 采集 + align_depth_to_color + high accuracy 档 |
| `box_detect.py` | 箱子**深度**检测：最近深度层 + 最大轮廓，输出掩膜/外接矩形 |
| `seam_detect.py` | CLAHE + Canny + 形态学 + HoughLinesP + 鲁棒中线聚合（限箱子内） |
| `geometry.py` | RANSAC 平面拟合（限箱子内） + deproject + 3D 几何量 |
| `visualize.py` | 箱子框 + 缝隙标注 + 深度伪彩 |
| `main.py` | 实时主循环 |
| `main_boxonly.py` | 仅箱子检测 |
| `main_raw.py` | 原始/调试入口 |
| `tune_depth.py` | 深度阈值实时滑块调参工具 |
| `test_seam_color.py` | 透明胶带颜色加权 + 缝隙热力图（独立测试） |
| `tune_hsv.py` | HSV 调参（已弃用，保留兼容） |
| `environment.yml` | Conda 环境定义 |
| `requirements.txt` | pip 依赖清单 |
| `depth_calib.json` | 深度标定结果（可选，现场可用 `tune_depth.py` 重新生成） |

## 工作流程

1. `box_detect` 用深度阈值圈出箱子顶面（最近表面层）→ 掩膜与最小外接矩形。
2. `seam_detect` 仅在箱子掩膜内检测竖直封箱缝隙。
3. `geometry.fit_plane` 仅在箱子掩膜内（剔除缝隙带）拟合箱顶平面。
4. 融合输出 3D 坐标 / 长度 / 宽度 / 胶带高度差。

## 调参提示

- 检测不到缝隙：降低 `HOUGH_MIN_LINE_LEN` / `HOUGH_THRESHOLD`，或放宽 `MAX_LINE_ANGLE_DEG`。
- 误检多：提高 `CANNY_HIGH`，缩小 `MAX_LINE_ANGLE_DEG`。
- 平面拟合不稳：调大 `RANSAC_RESIDUAL_MM`，或缩小 `ROI_MARGIN` 聚焦箱顶。
- 高度差始终 `unreliable`：透明胶带平贴箱顶，深度差在噪声内，属预期行为。
- 帧率低：在 `config.py` 把 `COLOR_WIDTH/HEIGHT` 改为 1280×720。

## 验证步骤

1. `python -c "import pyrealsense2, cv2, numpy, sklearn; print('ok')"`
2. 连上 D415，运行 `python main.py`，确认两窗口出图。
3. 在 0.4 m 放一张已知宽度（如 5 mm）胶带于箱顶，看像素宽度是否合理。
4. 倾斜箱顶下观察 `plane inlier` 比例应 > 70%。
5. 平贴胶带箱顶，确认绿色骨架线稳定检出。
6. 用已知宽度封箱胶（如 50 mm）核对宽度输出误差 < 2 mm。
7. 观察 fps，掉帧则降 color 分辨率。
8. 平贴胶带场景下 `height diff` 应显示 `(unreliable)`。
