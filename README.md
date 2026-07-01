# D415 纸箱封箱缝检测

使用 Intel RealSense D415 相机在 0.35–0.45m 工作距离下，实时检测 35cm 纸箱顶部沿长边的封箱胶带缝隙，输出：

- 缝隙两端点 3D 坐标（相机坐标系，米）
- 缝隙长度（mm）
- 缝隙宽度（mm，及像素宽度）
- 胶带面相对箱顶平面的高度差（mm，噪声量级时标注 `unreliable`）

并在视频流上叠加标注。

## 环境（Anaconda）

```bash
conda env create -f environment.yml
conda activate box-seam
```

## 运行

```bash
# 1. 标定箱子深度检测参数（对着箱子调滑块，让绿框稳定包住箱子顶面）
python tune_depth.py

# 2. 正式检测
python main.py

# 3. 仅箱子检测（不检测缝隙）
python main_boxonly.py
```

ESC 退出。tune_depth：`s` 保存、`q`/ESC 退出。

## 文件结构

| 文件 | 作用 |
|---|---|
| `config.py` | 所有可调参数集中处 |
| `camera.py` | D415 采集 + align_depth_to_color + high accuracy 档 |
| `box_detect.py` | 箱子**深度**检测：最近深度层 + 最大轮廓，输出掩膜/外接矩形 |
| `tune_depth.py` | 深度阈值实时滑块调参工具 |
| `seam_detect.py` | CLAHE + Canny + 形态学 + HoughLinesP + 鲁棒中线聚合（限箱子内） |
| `geometry.py` | RANSAC 平面拟合（限箱子内） + deproject + 3D 几何量 |
| `visualize.py` | 箱子框 + 缝隙标注 + 深度伪彩 |
| `main.py` | 实时主循环 |

## 工作流程

1. `box_detect` 用深度阈值圈出箱子顶面（最近表面层）→ 掩膜与最小外接矩形。
2. `seam_detect` 仅在箱子掩膜内检测竖直封箱缝隙。
3. `geometry.fit_plane` 仅在箱子掩膜内（剔除缝隙带）拟合箱顶平面。
4. 融合输出 3D 坐标 / 长度 / 宽度 / 胶带高度差。

## 调参提示

- 检测不到缝隙：降低 `HOUGH_MIN_LINE_LEN` / `HOUGH_THRESHOLD`，或放宽 `MAX_LINE_ANGLE_DEG`。
- 误检多：提高 `CANNY_HIGH`，缩小 `MAX_LINE_ANGLE_DEG`。
- 平面拟合不稳：调大 `RANSAC_RESIDUAL_MM`，或缩小 `ROI_MARGIN` 聚焦箱顶。
- 高度差始终 `unreliable`：胶带本就平贴，属预期行为。
- 帧率低：在 `config.py` 把 `COLOR_WIDTH/HEIGHT` 改为 1280×720。

## 验证步骤

1. `python -c "import pyrealsense2, cv2, numpy, sklearn; print('ok')"`
2. 连上 D415，运行 `python main.py`，确认两窗口出图。
3. 在 0.4m 放一张已知宽度（如 5mm）黑胶带于白纸上，看像素宽度是否符合 ~12 像素。
4. 倾斜箱顶下观察 `plane inlier` 比例应 > 70%。
5. 平贴胶带箱顶，确认绿色骨架线稳定检出。
6. 用已知宽度封箱胶（如 50mm）核对宽度输出误差 < 2mm。
7. 观察 fps，掉帧则降 color 分辨率。
8. 平贴胶带场景下"height diff"应显示 `(unreliable)`。
