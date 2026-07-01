"""箱子表面检测（基于深度）：

箱子顶面是画面中最近的近水平面，背景更远。取最近的有效深度作为参考，
保留参考深度附近的一个厚度层即为箱顶掩膜，再取最大轮廓。

导入时若存在 depth_calib.json（tune_depth.py 保存），用其中的值覆盖 config 默认。
"""
import os
import json

import cv2
import numpy as np

import config


def _load_calib():
    """读 depth_calib.json 覆盖 config 默认值。"""
    path = config.DEPTH_CALIB_FILE
    if not os.path.isfile(path):
        return
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if "BOX_DEPTH_TOL_M" in data:
            config.BOX_DEPTH_TOL_M = float(data["BOX_DEPTH_TOL_M"])
        if "BOX_DEPTH_REF_PERCENTILE" in data:
            config.BOX_DEPTH_REF_PERCENTILE = int(data["BOX_DEPTH_REF_PERCENTILE"])
        print(f"[box_detect] 使用标定深度参数: "
              f"tol={config.BOX_DEPTH_TOL_M*1000:.0f}mm "
              f"ref_pct={config.BOX_DEPTH_REF_PERCENTILE}")
    except Exception as e:
        print(f"[box_detect] 读取深度标定失败 {e}，回退默认值")


_load_calib()


def detect_box(depth_m):
    """在深度图上检测箱子顶面。

    depth_m: 全分辨率对齐到 color 的深度（米），无效为 0。
    返回 dict 或 None：
      mask: 全分辨率二值掩膜（箱子区域=255）
      contour: 最大轮廓
      rect: cv2.minAreaRect
      bbox: 轴对齐外接矩形 (x,y,w,h)
      d_ref: 参考深度（米）
    """
    if depth_m is None:
        return None
    valid = depth_m > config.BOX_DEPTH_MIN_M
    if not np.any(valid):
        return None

    # 最近表面参考深度：取有效深度的低端百分位，避开噪声尖点
    d_near = float(np.percentile(depth_m[valid], config.BOX_DEPTH_REF_PERCENTILE))
    lo = max(config.BOX_DEPTH_MIN_M, d_near - config.BOX_DEPTH_BEHIND_M)
    hi = d_near + config.BOX_DEPTH_TOL_M
    mask = ((depth_m > lo) & (depth_m < hi)).astype(np.uint8) * 255

    kopen = cv2.getStructuringElement(cv2.MORPH_RECT,
                                      (config.BOX_OPEN_KERNEL, config.BOX_OPEN_KERNEL))
    kclose = cv2.getStructuringElement(cv2.MORPH_RECT,
                                       (config.BOX_CLOSE_KERNEL, config.BOX_CLOSE_KERNEL))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kopen)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kclose)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    cnt = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(cnt)
    frame_area = depth_m.shape[0] * depth_m.shape[1]
    if area < config.MIN_BOX_AREA_FRAC * frame_area:
        return None

    rect = cv2.minAreaRect(cnt)
    x, y, w, h = cv2.boundingRect(cnt)
    clean = np.zeros_like(mask)
    cv2.drawContours(clean, [cnt], -1, 255, thickness=cv2.FILLED)
    return {
        "mask": clean,
        "contour": cnt,
        "rect": rect,
        "bbox": (int(x), int(y), int(w), int(h)),
        "d_ref": d_near,
    }
