"""RGB 封箱缝隙 2D 检测：CLAHE + Canny + 形态学 + HoughLinesP + 法向剖面边缘定位。

检测在降采样图上进行以提速，返回的坐标已映射回全分辨率（用于 deproject）。
"""
import math
import numpy as np
import cv2

import config


def _near_horizontal_angle(dx, dy):
    """直线相对水平方向夹角（度），0 = 完全水平。"""
    return abs(math.degrees(math.atan2(abs(dy), abs(dx))))


def _near_vertical_angle(dx, dy):
    """直线相对竖直方向夹角（度），0 = 完全竖直。"""
    return abs(math.degrees(math.atan2(abs(dx), abs(dy))))


def _line_angle_diff_deg(dx1, dy1, dx2, dy2):
    """两条直线方向夹角（度，0-90），忽略方向正负。"""
    a1 = math.atan2(dy1, dx1)
    a2 = math.atan2(dy2, dx2)
    d = abs(math.degrees(a1 - a2)) % 180.0
    return min(d, 180.0 - d)


def _point_line_dist(px, py, x1, y1, x2, y2):
    """点 (px,py) 到直线 (x1,y1)-(x2,y2) 的垂直距离。"""
    dx, dy = x2 - x1, y2 - y1
    L = math.hypot(dx, dy)
    if L < 1e-6:
        return math.hypot(px - x1, py - y1)
    return abs(dx * (y1 - py) - dy * (x1 - px)) / L


def _box_long_axis(rect):
    """从 minAreaRect (center,(w,h),angle) 得到长轴单位向量与边长。

    返回 (long_dx, long_dy, short_side_len, long_side_len)。
    """
    pts = cv2.boxPoints(rect)
    edges = [(pts[(i + 1) % 4] - pts[i]) for i in range(4)]
    lens = [math.hypot(e[0], e[1]) for e in edges]
    i_long = int(np.argmax(lens))
    e = edges[i_long]
    L = math.hypot(e[0], e[1])
    long_dx, long_dy = e[0] / L, e[1] / L
    short_side = min(lens)
    long_side = max(lens)
    return long_dx, long_dy, short_side, long_side


def detect_seam(color_bgr, box=None, prev_skel=None):
    """在 color 帧上检测封箱缝隙。

    box: box_detect.detect_box 返回的 dict（含全分辨率 mask），仅在箱子内检测。
    prev_skel: 上一帧缝隙骨架 (p1,p2)（全分辨率），用于时序锁定，避免跳到其他凹痕。
    返回 dict 或 None（坐标均为全分辨率像素）：
      skeleton: (p1, p2)
      edge1, edge2: 两条边缘点像素
      width_px: 全分辨率下边缘像素宽度
      line_angle_deg: 骨架相对水平夹角
    """
    s = config.DETECT_SCALE
    if s < 1.0:
        small = cv2.resize(color_bgr, None, fx=s, fy=s,
                           interpolation=cv2.INTER_AREA)
    else:
        small = color_bgr
    inv = 1.0 / s

    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=config.CLAHE_CLIP, tileGridSize=config.CLAHE_TILE)
    enhanced = clahe.apply(gray)
    edges = cv2.Canny(enhanced, config.CANNY_LOW, config.CANNY_HIGH)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT,
                                       (config.MORPH_KERNEL, config.MORPH_KERNEL))
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

    center_s = None
    short_side_s = None
    long_side_s = None
    small_mask = None
    if box is not None:
        full_mask = box["mask"]
        erode_px = max(1, int(round(config.BOX_MASK_ERODE * s)))
        kern = cv2.getStructuringElement(cv2.MORPH_RECT, (erode_px, erode_px))
        full_mask_erode = cv2.erode(full_mask, kern)
        small_mask = cv2.resize(full_mask_erode,
                                (small.shape[1], small.shape[0]),
                                interpolation=cv2.INTER_NEAREST)
        edges = cv2.bitwise_and(edges, small_mask)
        _, _, short_side, long_side = _box_long_axis(box["rect"])
        cx, cy = box["rect"][0]
        center_s = (cx * s, cy * s)
        short_side_s = short_side * s
        long_side_s = long_side * s

    lines = cv2.HoughLinesP(
        edges,
        rho=config.HOUGH_RHO,
        theta=np.pi / 180 * config.HOUGH_THETA_DEG,
        threshold=config.HOUGH_THRESHOLD,
        minLineLength=config.HOUGH_MIN_LINE_LEN * s,
        maxLineGap=config.HOUGH_MAX_LINE_GAP * s,
    )
    if lines is None:
        return None

    candidates = []
    for ln in lines[:, 0, :]:
        x1, y1, x2, y2 = ln
        dx, dy = x2 - x1, y2 - y1
        L = math.hypot(dx, dy)
        if L < 1:
            continue
        v_ang = _near_vertical_angle(dx, dy)
        if v_ang > config.SEAM_VERTICAL_TOL_DEG:
            continue
        if center_s is not None and short_side_s is not None:
            d = _point_line_dist(center_s[0], center_s[1],
                                 x1, y1, x2, y2)
            tol = config.SEAM_CENTER_TOL_FRAC * (short_side_s / 2.0)
            if d > tol:
                continue
        candidates.append((float(x1), float(y1), float(x2), float(y2), L))
    if not candidates:
        return None

    used = candidates
    if prev_skel is not None and long_side_s is not None:
        (px1, py1), (px2, py2) = prev_skel
        pmx = (px1 + px2) / 2.0 * s
        pmy = (py1 + py2) / 2.0 * s
        pdx, pdy = (px2 - px1) * s, (py2 - py1) * s
        lock_dist = config.SEAM_LOCK_DIST_FRAC * long_side_s
        locked = []
        for (x1, y1, x2, y2, L) in candidates:
            mx = (x1 + x2) / 2.0
            my = (y1 + y2) / 2.0
            if math.hypot(mx - pmx, my - pmy) > lock_dist:
                continue
            if _line_angle_diff_deg(x2 - x1, y2 - y1, pdx, pdy) > config.SEAM_LOCK_ANGLE_DEG:
                continue
            locked.append((x1, y1, x2, y2, L))
        if len(locked) >= 1:
            used = locked

    skel_small = _robust_line(used, small_mask)
    if skel_small is None:
        return None
    (sx1, sy1), (sx2, sy2) = skel_small
    (sx1, sy1), (sx2, sy2) = _refine_skeleton_to_crease(enhanced, (sx1, sy1), (sx2, sy2))
    ang = _near_vertical_angle(sx2 - sx1, sy2 - sy1)
    edge1_s, edge2_s, width_s = _locate_edges(enhanced, (sx1, sy1), (sx2, sy2))
    if edge1_s is None:
        return None
    p1 = (sx1 * inv, sy1 * inv)
    p2 = (sx2 * inv, sy2 * inv)
    edge1 = (edge1_s[0] * inv, edge1_s[1] * inv)
    edge2 = (edge2_s[0] * inv, edge2_s[1] * inv)
    return {
        "skeleton": (p1, p2),
        "edge1": edge1,
        "edge2": edge2,
        "width_px": width_s * inv,
        "line_angle_deg": ang,
    }


def _robust_line(cands, mask):
    """把多条共线候选段聚合成一条鲁棒中线，并沿方向延伸到掩膜边界。"""
    if not cands:
        return None
    total_w = sum(c[4] for c in cands)
    dx = sum((c[2] - c[0]) * c[4] for c in cands) / total_w
    dy = sum((c[3] - c[1]) * c[4] for c in cands) / total_w
    Ld = math.hypot(dx, dy)
    if Ld < 1e-6:
        return None
    dx, dy = dx / Ld, dy / Ld
    if dy < 0:
        dx, dy = -dx, -dy
    mx = float(np.median([(c[0] + c[2]) / 2.0 for c in cands]))
    my = float(np.median([(c[1] + c[3]) / 2.0 for c in cands]))

    if mask is not None:
        p1, p2 = _extend_to_mask(mask, mx, my, dx, dy)
    else:
        ts = []
        for (x1, y1, x2, y2, L) in cands:
            ts.append((x1 - mx) * dx + (y1 - my) * dy)
            ts.append((x2 - mx) * dx + (y2 - my) * dy)
        tmin, tmax = min(ts), max(ts)
        p1 = (mx + tmin * dx, my + tmin * dy)
        p2 = (mx + tmax * dx, my + tmax * dy)
    return p1, p2


def _extend_to_mask(mask, mx, my, dx, dy, max_steps=4000):
    """从 (mx,my) 沿 (dx,dy) 双向延伸，直到离开掩膜，返回两端点。"""
    h, w = mask.shape

    def step(sx, sy, sdx, sdy):
        x, y = sx, sy
        last = (x, y)
        for _ in range(max_steps):
            x += sdx
            y += sdy
            xi, yi = int(round(x)), int(round(y))
            if not (0 <= xi < w and 0 <= yi < h) or mask[yi, xi] == 0:
                return last
            last = (x, y)
        return last

    p1 = step(mx, my, -dx, -dy)
    p2 = step(mx, my, dx, dy)
    return p1, p2


def _profile_along_normal(enhanced, cx, cy, nx, ny, half_i):
    """在 (cx,cy) 沿法向采样灰度剖面，返回 (offsets, values) 或 (None, None)。"""
    h, w = enhanced.shape
    offsets = np.arange(-half_i, half_i + 1)
    xs = cx + nx * offsets
    ys = cy + ny * offsets
    xi = np.round(xs).astype(int)
    yi = np.round(ys).astype(int)
    ok = (xi >= 0) & (xi < w) & (yi >= 0) & (yi < h)
    if ok.sum() < 5:
        return None, None
    return offsets[ok], enhanced[yi[ok], xi[ok]].astype(np.float32)


def _crease_offset_in_profile(prof, offsets):
    """在法向剖面内定位折线：优先取胶带两边缘之间的最暗点。"""
    if prof is None or len(prof) < 5:
        return None
    grad = np.abs(np.gradient(prof))
    mid = len(grad) // 2
    if mid < 1 or mid >= len(grad) - 1:
        return float(offsets[int(np.argmin(prof))])
    a = int(np.argmax(grad[:mid]))
    b = int(mid + np.argmax(grad[mid:]))
    if grad[a] <= 0 or grad[b] <= 0 or b <= a + 1:
        return float(offsets[int(np.argmin(prof))])
    inner = prof[a:b + 1]
    crease_i = a + int(np.argmin(inner))
    return float(offsets[crease_i])


def _refine_skeleton_to_crease(enhanced, p1, p2):
    """沿法向把骨架从胶带边缘拉回到折线（压痕）最暗处。"""
    if not config.CREASE_REFINE_ENABLED:
        return p1, p2
    x1, y1 = p1
    x2, y2 = p2
    dx, dy = x2 - x1, y2 - y1
    L = math.hypot(dx, dy)
    if L < 1:
        return p1, p2
    tx, ty = dx / L, dy / L
    nx, ny = -ty, tx
    half_i = int(round(config.PROFILE_HALF_WIDTH * config.DETECT_SCALE))
    max_shift = config.CREASE_REFINE_MAX_SHIFT
    shifts = []
    for s in np.linspace(0.15, 0.85, config.PROFILE_SAMPLES):
        cx = x1 + s * dx
        cy = y1 + s * dy
        offs, prof = _profile_along_normal(enhanced, cx, cy, nx, ny, half_i)
        if prof is None:
            continue
        off = _crease_offset_in_profile(prof, offs)
        if off is not None:
            shifts.append(off)
    if len(shifts) < config.CREASE_REFINE_MIN_SAMPLES:
        return p1, p2
    shift = float(np.median(shifts))
    shift = float(np.clip(shift, -max_shift, max_shift))
    if abs(shift) < 0.3:
        return p1, p2
    return (x1 + nx * shift, y1 + ny * shift), (x2 + nx * shift, y2 + ny * shift)


def _locate_edges(enhanced, p1, p2):
    """沿骨架法向采样若干剖面，取两侧最强梯度峰定位胶带两条边缘。"""
    x1, y1 = p1
    x2, y2 = p2
    dx, dy = x2 - x1, y2 - y1
    L = math.hypot(dx, dy)
    if L < 1:
        return None, None, 0.0
    tx, ty = dx / L, dy / L
    nx, ny = -ty, tx
    half_i = int(round(config.PROFILE_HALF_WIDTH * config.DETECT_SCALE))
    samples = config.PROFILE_SAMPLES

    e1_pts = []
    e2_pts = []
    widths = []
    for s in np.linspace(0.15, 0.85, samples):
        cx = x1 + s * dx
        cy = y1 + s * dy
        offs, prof = _profile_along_normal(enhanced, cx, cy, nx, ny, half_i)
        if prof is None:
            continue
        grad = np.abs(np.gradient(prof))
        mid = len(grad) // 2
        if mid < 1 or mid >= len(grad) - 1:
            continue
        a = int(np.argmax(grad[:mid]))
        b = int(mid + np.argmax(grad[mid:]))
        if grad[a] <= 0 or grad[b] <= 0:
            continue
        width_px = abs(b - a)
        if width_px < 2:
            continue
        oa = float(offs[a])
        ob = float(offs[b])
        e1_pts.append((cx + nx * oa, cy + ny * oa))
        e2_pts.append((cx + nx * ob, cy + ny * ob))
        widths.append(width_px)

    if len(widths) < 3:
        return None, None, 0.0
    e1 = np.mean(e1_pts, axis=0)
    e2 = np.mean(e2_pts, axis=0)
    return (float(e1[0]), float(e1[1])), (float(e2[0]), float(e2[1])), float(np.mean(widths))
