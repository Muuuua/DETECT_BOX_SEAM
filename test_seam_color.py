"""独立测试：箱子 ROI 内缝隙检测 + 颜色加权。

透明胶带场景：缝隙压痕偏暗、胶带面可能反光；高光掩膜 + 列投影主检测 + Hough 回退。
分列显示 dark / tape_bright 热力图便于调参。运行: python test_seam_color.py  ESC 退出。
"""
import math
import time

import cv2
import numpy as np

import box_detect
from camera import D415Camera
from seam_detect import _refine_skeleton_to_crease

# ---- 基础检测参数 ----
DETECT_SCALE = 0.5
CLAHE_CLIP = 2.0
CLAHE_TILE = (8, 8)
CANNY_LOW = 30
CANNY_HIGH = 90
MORPH_KERNEL = 5
HOUGH_RHO = 1
HOUGH_THETA_DEG = 1.0
HOUGH_THRESHOLD = 50
HOUGH_MIN_LINE_LEN = 300
HOUGH_MAX_LINE_GAP = 30
SEAM_CENTER_TOL_FRAC = 0.5
SEAM_LOCK_DIST_FRAC = 0.06
SEAM_LOCK_ANGLE_DEG = 8.0
BOX_MASK_ERODE = 15
PROFILE_SAMPLES = 41
PROFILE_HALF_WIDTH = 25
DISPLAY_MAX_WIDTH = 1280

# ---- 缝隙颜色特征（相对哑光箱顶）----
# 压痕/折线：比箱顶更暗（透明胶带下常见阴影缝）
CREASE_DARK_SPAN = 25.0
CREASE_WEIGHT = 0.75
# 透明膜反光：更亮 + 饱和度更低（权重宜低，避免整面误检）
TAPE_BRIGHT_SPAN = 25.0
TAPE_BRIGHT_MAX = 30.0
TAPE_SAT_SPAN = 35.0
BRIGHT_TAPE_WEIGHT = 0.2
MIN_COLOR_WEIGHT = 0.25
COLOR_WEIGHT_GAIN = 3.0
SEAM_SCORE_FLOOR = 0.05
BROWN_L_MIN = 40
BROWN_L_MAX = 155
LINE_COLOR_SAMPLES = 24
LINE_PATCH_RADIUS = 2

# ---- 高光掩膜（反光像素不参与评分与哑光参考）----
SPECULAR_L_MIN = 200
SPECULAR_S_MAX = 15

# ---- 列投影检测（主路径：沿箱短轴剖面找亮度谷底）----
VALLEY_PROFILE_SAMPLES = 36     # 沿长轴采样剖面数
VALLEY_PROFILE_PTS = 48         # 每条剖面沿短轴采样点数
VALLEY_SEARCH_FRAC = 0.42       # 短轴半宽内搜索谷底
VALLEY_MIN_SAMPLES = 8          # 有效剖面最少条数
VALLEY_PREV_LOCK_FRAC = 0.08    # 有上一帧时缩窄横向搜索
MATTE_CENTER_EXCLUDE_FRAC = 0.22  # 估哑光参考时剔除箱心胶带带

# 局部对比度热力图（不依赖全局参考，压痕相对邻域变暗）
LOCAL_DARK_SPAN = 18.0
LOCAL_BLUR_FRAC = 0.09          # 高斯核 ≈ 短边×scale×此比例

# Hough 回退：与长轴夹角容差（度），替代固定竖直假设
SEAM_LONG_AXIS_TOL_DEG = 12.0

# ---- 时序平滑（减轻跳动）----
SEAM_SMOOTH_ALPHA = 0.25
HEAT_SMOOTH_ALPHA = 0.5
LOST_RESET = 15
SHOW_ALL_CANDIDATES = False


def _line_angle_diff_deg(dx1, dy1, dx2, dy2):
    a1 = math.atan2(dy1, dx1)
    a2 = math.atan2(dy2, dx2)
    d = abs(math.degrees(a1 - a2)) % 180.0
    return min(d, 180.0 - d)


def _point_line_dist(px, py, x1, y1, x2, y2):
    dx, dy = x2 - x1, y2 - y1
    L = math.hypot(dx, dy)
    if L < 1e-6:
        return math.hypot(px - x1, py - y1)
    return abs(dx * (y1 - py) - dy * (x1 - px)) / L


def _box_long_axis(rect):
    pts = cv2.boxPoints(rect)
    edges = [(pts[(i + 1) % 4] - pts[i]) for i in range(4)]
    lens = [math.hypot(e[0], e[1]) for e in edges]
    i_long = int(np.argmax(lens))
    e = edges[i_long]
    L = math.hypot(e[0], e[1])
    return e[0] / L, e[1] / L, min(lens), max(lens)


def _ema_point(new_pt, old_pt, alpha):
    return (new_pt[0] * alpha + old_pt[0] * (1 - alpha),
            new_pt[1] * alpha + old_pt[1] * (1 - alpha))


def _specular_mask(lab, hsv, valid=None):
    """强高光像素：L* 过高或饱和度极低（透明胶带镜面反射）。"""
    L = lab[:, :, 0]
    spec = (L > SPECULAR_L_MIN) | (hsv[:, :, 1] < SPECULAR_S_MAX)
    if valid is not None:
        spec = spec & valid
    return spec


def _box_axes(rect):
    """返回两对正交轴（边1、边2）及对应边长。不假定哪条是缝。"""
    pts = cv2.boxPoints(rect)
    e0 = pts[1] - pts[0]
    e1 = pts[2] - pts[1]
    L0 = float(math.hypot(e0[0], e0[1]))
    L1 = float(math.hypot(e1[0], e1[1]))
    if L0 < 1e-6 or L1 < 1e-6:
        return (1.0, 0.0, 0.0, 1.0, 100.0, 100.0)
    d0 = (e0[0] / L0, e0[1] / L0)
    d1 = (e1[0] / L1, e1[1] / L1)
    return d0[0], d0[1], d1[0], d1[1], L0, L1


def _box_axes_legacy(rect):
    """兼容旧代码：长边为轴0，短边为轴1。"""
    ax0x, ax0y, ax1x, ax1y, len0, len1 = _box_axes(rect)
    if len0 >= len1:
        return ax0x, ax0y, ax1x, ax1y, len1, len0
    return ax1x, ax1y, ax0x, ax0y, len0, len1


def _center_exclusion_mask(h, w, cx, cy, lat_dx, lat_dy, lat_len, scale, frac):
    """剔除箱心沿横向（剖面方向）的胶带带，用于哑光参考。"""
    ys, xs = np.mgrid[0:h, 0:w].astype(np.float32)
    u = (xs - cx * scale) * lat_dx + (ys - cy * scale) * lat_dy
    half = frac * lat_len * scale
    return np.abs(u) > half


def _estimate_box_matte(bgr, mask, exclude_center=None):
    """ROI 内哑光箱顶 Lab + HSV 参考（排除高光与箱心胶带带）。"""
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    L = lab[:, :, 0]
    valid = mask > 0
    spec = _specular_mask(lab, hsv, valid)
    matte = valid & ~spec & (L >= BROWN_L_MIN) & (L <= BROWN_L_MAX)
    if exclude_center is not None:
        matte = matte & exclude_center
    if int(matte.sum()) < 80:
        matte = valid & ~spec
        if exclude_center is not None:
            matte = matte & exclude_center
    if int(matte.sum()) < 40:
        matte = valid & ~spec
    if int(matte.sum()) < 20:
        matte = valid
    if not np.any(matte):
        return (np.array([128.0, 128.0, 128.0], dtype=np.float32),
                np.array([15.0, 80.0, 120.0], dtype=np.float32))
    lab_ref = np.median(lab[matte].reshape(-1, 3), axis=0).astype(np.float32)
    hsv_ref = np.median(hsv[matte].reshape(-1, 3), axis=0).astype(np.float32)
    return lab_ref, hsv_ref


def _patch_mean_lab_hsv(lab, hsv, cx, cy, radius):
    h, w = lab.shape[:2]
    xi, yi = int(round(cx)), int(round(cy))
    r = max(1, radius)
    x0, x1 = max(0, xi - r), min(w, xi + r + 1)
    y0, y1 = max(0, yi - r), min(h, yi + r + 1)
    lp = lab[y0:y1, x0:x1]
    hp = hsv[y0:y1, x0:x1]
    if lp.size == 0:
        return None, None
    return lp.reshape(-1, 3).mean(axis=0), hp.reshape(-1, 3).mean(axis=0)


def _pixel_seam_score(lab, hsv, box_lab, box_hsv, mask=None):
    """逐像素缝隙疑似度 0~1：压痕变暗 或 透明膜反光，取较强者；高光像素置零。"""
    ref_l = float(box_lab[0])
    ref_s = float(box_hsv[1])
    L = lab[:, :, 0].astype(np.float32)
    dark = np.clip((ref_l - L) / CREASE_DARK_SPAN, 0.0, 1.0)
    d_l = L - ref_l
    bright = np.clip(d_l / TAPE_BRIGHT_SPAN, 0.0, 1.0)
    bright[d_l <= 0] = 0.0
    bright[d_l > TAPE_BRIGHT_MAX] *= 0.35
    d_s = ref_s - hsv[:, :, 1].astype(np.float32)
    low_sat = np.clip(d_s / TAPE_SAT_SPAN, 0.0, 1.0)
    low_sat[d_s <= 0] = 0.0
    tape_bright = 0.6 * bright + 0.4 * low_sat
    seam = np.maximum(CREASE_WEIGHT * dark, BRIGHT_TAPE_WEIGHT * tape_bright)

    valid = (mask > 0) if mask is not None else np.ones(L.shape, dtype=bool)
    spec = _specular_mask(lab, hsv, valid)
    dark = dark.astype(np.float32)
    tape_bright = tape_bright.astype(np.float32)
    dark[spec] = 0.0
    tape_bright[spec] = 0.0
    seam = seam.astype(np.float32)
    seam[spec] = 0.0
    if mask is not None:
        dark[mask == 0] = 0.0
        tape_bright[mask == 0] = 0.0
        seam[mask == 0] = 0.0
    return seam, dark, tape_bright


def _local_dark_map(L, mask, spec, short_len, scale):
    """局部邻域对比：压痕比周围更暗，不依赖易被胶带污染的 global ref。"""
    k = int(max(11, short_len * scale * LOCAL_BLUR_FRAC)) | 1
    Lf = L.astype(np.float32)
    blur = cv2.GaussianBlur(Lf, (k, k), 0)
    local = np.clip((blur - Lf) / LOCAL_DARK_SPAN, 0.0, 1.0)
    local[spec] = 0.0
    if mask is not None:
        local[mask == 0] = 0.0
    return local.astype(np.float32)


def _line_seam_score(lab, hsv, x1, y1, x2, y2, box_lab, box_hsv):
    """沿线采样平均缝隙疑似度。"""
    ref_l = float(box_lab[0])
    ref_s = float(box_hsv[1])
    scores = []
    for t in np.linspace(0.05, 0.95, LINE_COLOR_SAMPLES):
        cx = x1 + t * (x2 - x1)
        cy = y1 + t * (y2 - y1)
        mean_lab, mean_hsv = _patch_mean_lab_hsv(lab, hsv, cx, cy, LINE_PATCH_RADIUS)
        if mean_lab is None:
            continue
        d_l = float(mean_lab[0]) - ref_l
        dark = float(np.clip((ref_l - float(mean_lab[0])) / CREASE_DARK_SPAN, 0.0, 1.0))
        bright = float(np.clip(d_l / TAPE_BRIGHT_SPAN, 0.0, 1.0)) if d_l > 0 else 0.0
        if d_l > TAPE_BRIGHT_MAX:
            bright *= 0.35
        d_s = ref_s - float(mean_hsv[1])
        low_sat = float(np.clip(d_s / TAPE_SAT_SPAN, 0.0, 1.0)) if d_s > 0 else 0.0
        tape_bright = 0.6 * bright + 0.4 * low_sat
        scores.append(max(CREASE_WEIGHT * dark, BRIGHT_TAPE_WEIGHT * tape_bright))
    if not scores:
        return 0.0
    return float(np.mean(scores))


def _extend_to_mask(mask, mx, my, dx, dy, max_steps=4000):
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

    return step(mx, my, -dx, -dy), step(mx, my, dx, dy)


def _robust_line(cands, mask):
    if not cands:
        return None
    total_w = sum(c[4] for c in cands)
    if total_w < 1e-6:
        return None
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
        return _extend_to_mask(mask, mx, my, dx, dy)
    ts = []
    for (x1, y1, x2, y2, _w) in cands:
        ts.append((x1 - mx) * dx + (y1 - my) * dy)
        ts.append((x2 - mx) * dx + (y2 - my) * dy)
    tmin, tmax = min(ts), max(ts)
    return (mx + tmin * dx, my + tmin * dy), (mx + tmax * dx, my + tmax * dy)


def _locate_edges(enhanced, p1, p2):
    x1, y1 = p1
    x2, y2 = p2
    dx, dy = x2 - x1, y2 - y1
    L = math.hypot(dx, dy)
    if L < 1:
        return None, None, 0.0
    tx, ty = dx / L, dy / L
    nx, ny = -ty, tx
    h, w = enhanced.shape
    half_i = int(round(PROFILE_HALF_WIDTH * DETECT_SCALE))
    e1_pts, e2_pts, widths = [], [], []
    for s in np.linspace(0.15, 0.85, PROFILE_SAMPLES):
        cx = x1 + s * dx
        cy = y1 + s * dy
        offsets = np.arange(-half_i, half_i + 1)
        xs = cx + nx * offsets
        ys = cy + ny * offsets
        xi = np.round(xs).astype(int)
        yi = np.round(ys).astype(int)
        ok = (xi >= 0) & (xi < w) & (yi >= 0) & (yi < h)
        if ok.sum() < 5:
            continue
        prof = enhanced[yi[ok], xi[ok]].astype(np.float32)
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
        e1_pts.append((cx + nx * offsets[a], cy + ny * offsets[a]))
        e2_pts.append((cx + nx * offsets[b], cy + ny * offsets[b]))
        widths.append(width_px)
    if len(widths) < 3:
        return None, None, 0.0
    e1 = np.mean(e1_pts, axis=0)
    e2 = np.mean(e2_pts, axis=0)
    return (float(e1[0]), float(e1[1])), (float(e2[0]), float(e2[1])), float(np.mean(widths))


def _score_maps_at_scale(bgr, mask_full, scale, box=None):
    """在检测尺度上计算分数图与哑光参考。"""
    if scale < 1.0:
        small = cv2.resize(bgr, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        small_mask = cv2.resize(mask_full, (small.shape[1], small.shape[0]),
                                interpolation=cv2.INTER_NEAREST)
    else:
        small = bgr
        small_mask = mask_full
    lab = cv2.cvtColor(small, cv2.COLOR_BGR2LAB)
    hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
    exclude = None
    short_len = 200.0
    if box is not None:
        cx, cy = box["rect"][0]
        _, _, lat_dx, lat_dy, lat_len, _ = _box_axes_legacy(box["rect"])
        h, w = small.shape[:2]
        exclude = _center_exclusion_mask(
            h, w, cx, cy, lat_dx, lat_dy, lat_len, scale, MATTE_CENTER_EXCLUDE_FRAC)
        _, _, _, _, len0, len1 = _box_axes(box["rect"])
        short_len = min(len0, len1)
    box_lab, box_hsv = _estimate_box_matte(small, small_mask, exclude_center=exclude)
    seam, dark, tape_bright = _pixel_seam_score(lab, hsv, box_lab, box_hsv, small_mask)
    L = lab[:, :, 0]
    valid = small_mask > 0
    spec = _specular_mask(lab, hsv, valid)
    local_dark = _local_dark_map(L, small_mask, spec, short_len, scale)
    return small, small_mask, box_lab, box_hsv, seam, local_dark, tape_bright


def _smooth_1d(arr, ksize):
    k = max(3, int(ksize) | 1)
    kernel = cv2.getGaussianKernel(k, 0).flatten().astype(np.float32)
    kernel /= kernel.sum()
    return np.convolve(arr, kernel, mode="same")


def _sample_line_profile(map2d, mask, x0, y0, dir_x, dir_y, half_len, n_pts):
    """沿 (dir_x,dir_y) 采样剖面，跳过掩膜外点。"""
    ts = np.linspace(-half_len, half_len, n_pts)
    h, w = map2d.shape
    offs, vals = [], []
    for t in ts:
        x, y = x0 + dir_x * t, y0 + dir_y * t
        xi, yi = int(round(x)), int(round(y))
        if 0 <= xi < w and 0 <= yi < h and mask[yi, xi] > 0:
            offs.append(float(t))
            vals.append(float(map2d[yi, xi]))
    if len(vals) < 5:
        return None, None
    return np.array(offs, dtype=np.float32), np.array(vals, dtype=np.float32)


def _peak_offset_in_profile(offs, prof):
    """剖面内局部暗度峰值（折线相对邻域最暗）。"""
    if prof is None or len(prof) < 5:
        return None
    sm = _smooth_1d(prof, 5)
    return float(offs[int(np.argmax(sm))])


def _axis_stripe_score(local_dark, mask, cx, cy, seam_dx, seam_dy, seam_half_len):
    """沿候选缝方向过箱心的线积分：方向正确时连续命中胶带/折线。"""
    ts = np.linspace(-seam_half_len, seam_half_len, 40)
    h, w = local_dark.shape
    vals = []
    for t in ts:
        x, y = cx + seam_dx * t, cy + seam_dy * t
        xi, yi = int(round(x)), int(round(y))
        if 0 <= xi < w and 0 <= yi < h and mask[yi, xi] > 0:
            vals.append(float(local_dark[yi, xi]))
    if len(vals) < 8:
        return 0.0
    return float(np.mean(vals))


def _skeleton_axis_dot(prev_skel, seam_dx, seam_dy):
    """上一帧骨架与候选缝方向的吻合度 0~1。"""
    (px1, py1), (px2, py2) = prev_skel
    pdx, pdy = px2 - px1, py2 - py1
    L = math.hypot(pdx, pdy)
    if L < 1e-6:
        return 0.0
    return abs(pdx / L * seam_dx + pdy / L * seam_dy)


def _detect_seam_valley_one_axis(local_dark, mask, cx_s, cy_s, seam_dx, seam_dy,
                                   lat_dx, lat_dy, lat_len, seam_len, scale,
                                   prev_skel=None, prev_axis_dot=None):
    """沿 seam 方向布置剖面，横向(lat)搜索 local-dark 峰。"""
    half_lat = VALLEY_SEARCH_FRAC * lat_len * scale / 2.0
    half_seam = seam_len * scale * 0.42

    u_lo, u_hi = -half_lat, half_lat
    if prev_skel is not None and prev_axis_dot is not None and prev_axis_dot > 0.85:
        (px1, py1), (px2, py2) = prev_skel
        pmx = (px1 + px2) / 2.0 * scale
        pmy = (py1 + py2) / 2.0 * scale
        prev_u = (pmx - cx_s) * lat_dx + (pmy - cy_s) * lat_dy
        lock = VALLEY_PREV_LOCK_FRAC * seam_len * scale
        u_lo = max(u_lo, prev_u - lock)
        u_hi = min(u_hi, prev_u + lock)
        if u_hi - u_lo < 4:
            return None

    lateral_offs = []
    contrasts = []
    for frac in np.linspace(0.12, 0.88, VALLEY_PROFILE_SAMPLES):
        t_along = (frac - 0.5) * 2.0 * half_seam
        bx = cx_s + t_along * seam_dx
        by = cy_s + t_along * seam_dy
        offs, prof = _sample_line_profile(
            local_dark, mask, bx, by, lat_dx, lat_dy, half_lat, VALLEY_PROFILE_PTS)
        if prof is None:
            continue
        u = _peak_offset_in_profile(offs, prof)
        if u is None or u < u_lo or u > u_hi:
            continue
        sm = _smooth_1d(prof, 5)
        contrast = float(sm.max() - np.percentile(sm, 25))
        if contrast < 0.04:
            continue
        lateral_offs.append(u)
        contrasts.append(contrast)

    if len(lateral_offs) < VALLEY_MIN_SAMPLES:
        return None

    u_med = float(np.median(lateral_offs))
    mx = cx_s + u_med * lat_dx
    my = cy_s + u_med * lat_dy
    p1, p2 = _extend_to_mask(mask, mx, my, seam_dx, seam_dy)
    score = float(np.median(contrasts))
    return {
        "skeleton_small": (p1, p2),
        "score": score,
        "lateral_u": u_med,
        "seam_dx": seam_dx,
        "seam_dy": seam_dy,
        "n_profiles": len(lateral_offs),
    }


def _detect_seam_valley(enhanced, local_dark, mask, box, scale, prev_skel=None):
    """双轴竞争：方顶/近方顶时几何长边≠缝方向，用条纹积分+剖面得分选轴。"""
    if box is None:
        return None
    cx, cy = box["rect"][0]
    cx_s, cy_s = cx * scale, cy * scale
    ax0x, ax0y, ax1x, ax1y, len0, len1 = _box_axes(box["rect"])

    axis_candidates = [
        ("a0", ax0x, ax0y, ax1x, ax1y, len1, len0),
        ("a1", ax1x, ax1y, ax0x, ax0y, len0, len1),
        ("a0-", -ax0x, -ax0y, ax1x, ax1y, len1, len0),
        ("a1-", -ax1x, -ax1y, ax0x, ax0y, len0, len1),
    ]
    # 去重（反向与正向条纹分相同）
    seen = set()
    unique = []
    for item in axis_candidates:
        key = (round(item[1], 3), round(item[2], 3))
        if key in seen:
            continue
        seen.add(key)
        unique.append(item)

    best = None
    for name, sdx, sdy, ldx, ldy, lat_len, seam_len in unique:
        stripe = _axis_stripe_score(
            local_dark, mask, cx_s, cy_s, sdx, sdy, seam_len * scale * 0.42)
        axis_dot = _skeleton_axis_dot(prev_skel, sdx, sdy) if prev_skel else 0.0
        result = _detect_seam_valley_one_axis(
            local_dark, mask, cx_s, cy_s, sdx, sdy, ldx, ldy, lat_len, seam_len, scale,
            prev_skel=prev_skel, prev_axis_dot=axis_dot)
        if result is None:
            continue
        combined = (result["score"] * (0.25 + stripe)
                    * (0.55 + 0.45 * axis_dot)
                    * (0.7 + 0.03 * result["n_profiles"]))
        result["score"] = combined
        result["stripe"] = stripe
        result["axis"] = name
        if best is None or combined > best["score"]:
            best = result

    return best


def _detect_seam_hough(color_bgr, box, prev_skel, small, small_mask, enhanced,
                       box_lab, box_hsv, inv):
    """Hough 回退：线段须沿箱两条边方向之一且过箱心附近。"""
    s = DETECT_SCALE
    center_s = None
    axes = []
    lat_lens = []
    if box is not None:
        ax0x, ax0y, ax1x, ax1y, len0, len1 = _box_axes(box["rect"])
        axes = [(ax0x, ax0y), (ax1x, ax1y), (-ax0x, -ax0y), (-ax1x, -ax1y)]
        lat_lens = [len1, len0, len1, len0]
        cx, cy = box["rect"][0]
        center_s = (cx * s, cy * s)

    edges = cv2.Canny(enhanced, CANNY_LOW, CANNY_HIGH)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (MORPH_KERNEL, MORPH_KERNEL))
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
    if small_mask is not None:
        edges = cv2.bitwise_and(edges, small_mask)

    lab_small = cv2.cvtColor(small, cv2.COLOR_BGR2LAB)
    hsv_small = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)

    lines = cv2.HoughLinesP(
        edges, HOUGH_RHO, np.pi / 180 * HOUGH_THETA_DEG, HOUGH_THRESHOLD,
        minLineLength=HOUGH_MIN_LINE_LEN * s, maxLineGap=HOUGH_MAX_LINE_GAP * s,
    )
    if lines is None:
        return None

    raw_candidates = []
    weighted_candidates = []
    for ln in lines[:, 0, :]:
        x1, y1, x2, y2 = map(float, ln)
        dx, dy = x2 - x1, y2 - y1
        L = math.hypot(dx, dy)
        if L < 1:
            continue
        ang_ok = False
        for adx, ady in axes:
            if _line_angle_diff_deg(dx, dy, adx, ady) <= SEAM_LONG_AXIS_TOL_DEG:
                ang_ok = True
                break
        if not ang_ok:
            continue
        if center_s is not None and lat_lens:
            lat_half = min(lat_lens) * s / 2.0
            d = _point_line_dist(center_s[0], center_s[1], x1, y1, x2, y2)
            if d > SEAM_CENTER_TOL_FRAC * lat_half:
                continue
        seam_score = _line_seam_score(lab_small, hsv_small, x1, y1, x2, y2, box_lab, box_hsv)
        color_factor = MIN_COLOR_WEIGHT + COLOR_WEIGHT_GAIN * max(seam_score, SEAM_SCORE_FLOOR)
        combined_w = L * color_factor
        raw_candidates.append((x1, y1, x2, y2, L, seam_score, combined_w))
        weighted_candidates.append((x1, y1, x2, y2, combined_w, seam_score))

    if not weighted_candidates:
        return None

    used = weighted_candidates
    if prev_skel is not None and box is not None:
        (px1, py1), (px2, py2) = prev_skel
        pmx = (px1 + px2) / 2.0 * s
        pmy = (py1 + py2) / 2.0 * s
        pdx, pdy = (px2 - px1) * s, (py2 - py1) * s
        _, _, _, _, len0, len1 = _box_axes(box["rect"])
        lock_dist = SEAM_LOCK_DIST_FRAC * max(len0, len1) * s
        locked = []
        for cand in weighted_candidates:
            x1, y1, x2, y2, cw, ts = cand
            mx = (x1 + x2) / 2.0
            my = (y1 + y2) / 2.0
            if math.hypot(mx - pmx, my - pmy) > lock_dist:
                continue
            if _line_angle_diff_deg(x2 - x1, y2 - y1, pdx, pdy) > SEAM_LOCK_ANGLE_DEG:
                continue
            locked.append(cand)
        if locked:
            used = locked

    skel_base = _robust_line([(c[0], c[1], c[2], c[3], c[4]) for c in raw_candidates], small_mask)
    skel_color = _robust_line([(c[0], c[1], c[2], c[3], c[4]) for c in used], small_mask)
    if skel_color is None:
        return None

    best = max(used, key=lambda c: c[4])
    return {
        "skeleton_small": skel_color,
        "skeleton_baseline_small": skel_base,
        "seam_score": best[5],
        "candidates": raw_candidates,
        "method": "hough",
    }


def detect_seam_color_weighted(color_bgr, box, prev_skel=None):
    s = DETECT_SCALE
    inv = 1.0 / s
    if box is None:
        return None

    mask_full = _box_roi_mask(box)
    small, small_mask, box_lab, box_hsv, seam_map, local_dark, tape_map = _score_maps_at_scale(
        color_bgr, mask_full, s, box=box)

    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=CLAHE_CLIP, tileGridSize=CLAHE_TILE)
    enhanced = clahe.apply(gray)

    proj = _detect_seam_valley(enhanced, local_dark, small_mask, box, s, prev_skel=prev_skel)
    hough_data = None
    method = "valley"
    seam_score = 0.0
    skel_small = None
    skel_base_small = None
    raw_candidates = []

    if proj is not None:
        skel_small = proj["skeleton_small"]
        seam_score = min(1.0, proj["score"])
        method = f"valley-{proj.get('axis', '?')}"
    else:
        hough_data = _detect_seam_hough(
            color_bgr, box, prev_skel, small, small_mask, enhanced,
            box_lab, box_hsv, inv)
        if hough_data is None:
            return None
        method = "hough"
        skel_small = hough_data["skeleton_small"]
        skel_base_small = hough_data.get("skeleton_baseline_small")
        seam_score = hough_data["seam_score"]
        raw_candidates = hough_data.get("candidates", [])

    (sx1, sy1), (sx2, sy2) = skel_small
    (sx1, sy1), (sx2, sy2) = _refine_skeleton_to_crease(enhanced, (sx1, sy1), (sx2, sy2))
    edge1_s, edge2_s, width_s = _locate_edges(enhanced, (sx1, sy1), (sx2, sy2))
    p1 = (sx1 * inv, sy1 * inv)
    p2 = (sx2 * inv, sy2 * inv)
    if edge1_s is None:
        edge1, edge2, width_px = p1, p2, 0.0
    else:
        edge1 = (edge1_s[0] * inv, edge1_s[1] * inv)
        edge2 = (edge2_s[0] * inv, edge2_s[1] * inv)
        width_px = width_s * inv

    skel_base = None
    if hough_data is not None and skel_base_small is not None:
        skel_base = (
            (skel_base_small[0][0] * inv, skel_base_small[0][1] * inv),
            (skel_base_small[1][0] * inv, skel_base_small[1][1] * inv),
        )

    return {
        "skeleton": (p1, p2),
        "skeleton_baseline": skel_base,
        "edge1": edge1,
        "edge2": edge2,
        "width_px": width_px,
        "tape_score": seam_score,
        "seam_score": seam_score,
        "detect_method": method,
        "box_lab": box_lab,
        "box_hsv": box_hsv,
        "candidates": [
            {"seg": ((c[0] * inv, c[1] * inv), (c[2] * inv, c[3] * inv)),
             "geom_len": c[4], "tape_score": c[5], "seam_score": c[5], "weight": c[6]}
            for c in raw_candidates
        ],
    }


def _smooth_seam(new_seam, smoothed, alpha):
    if smoothed is None:
        return dict(new_seam)
    out = dict(new_seam)
    p1 = _ema_point(new_seam["skeleton"][0], smoothed["skeleton"][0], alpha)
    p2 = _ema_point(new_seam["skeleton"][1], smoothed["skeleton"][1], alpha)
    out["skeleton"] = (p1, p2)
    out["edge1"] = _ema_point(new_seam["edge1"], smoothed["edge1"], alpha)
    out["edge2"] = _ema_point(new_seam["edge2"], smoothed["edge2"], alpha)
    out["width_px"] = new_seam["width_px"] * alpha + smoothed["width_px"] * (1 - alpha)
    out["tape_score"] = new_seam["tape_score"] * alpha + smoothed["tape_score"] * (1 - alpha)
    out["seam_score"] = out["tape_score"]
    return out


def _box_roi_mask(box):
    erode_px = max(1, BOX_MASK_ERODE)
    kern = cv2.getStructuringElement(cv2.MORPH_RECT, (erode_px, erode_px))
    return cv2.erode(box["mask"], kern)


def _seam_score_maps(bgr, box_lab, box_hsv, mask, box=None, scale=1.0):
    """ROI 内 combined / local_dark / tape_bright 分数图（全分辨率）。"""
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    seam, dark, tape = _pixel_seam_score(lab, hsv, box_lab, box_hsv, mask)
    L = lab[:, :, 0]
    valid = mask > 0
    spec = _specular_mask(lab, hsv, valid)
    short_len = 200.0
    if box is not None:
        _, _, _, _, len0, len1 = _box_axes(box["rect"])
        short_len = min(len0, len1)
    local_dark = _local_dark_map(L, mask, spec, short_len, scale)
    return seam, local_dark, tape


def _render_seam_heatmap(bgr, score, mask, skeleton=None, title="seam"):
    """渲染热力图：黄/白=缝隙疑似高，蓝/紫=接近箱顶本色。"""
    roi = score[mask > 0]
    if roi.size > 0:
        hi = float(np.percentile(roi, 97))
        disp = np.clip(score / max(hi, 0.08), 0.0, 1.0)
    else:
        disp = score.copy()
    heat = cv2.applyColorMap((disp * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
    gray3 = cv2.cvtColor(cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)
    vis = cv2.addWeighted(heat, 0.7, gray3, 0.3, 0)
    vis[mask == 0] = 0

    if skeleton is not None:
        p1, p2 = skeleton
        cv2.line(vis, (int(p1[0]), int(p1[1])), (int(p2[0]), int(p2[1])), (0, 255, 0), 2, cv2.LINE_AA)

    bar_h, bar_w = 14, 160
    bar = np.linspace(255, 0, bar_w, dtype=np.uint8).reshape(1, -1)
    bar = cv2.applyColorMap(bar, cv2.COLORMAP_TURBO)
    bar = cv2.resize(bar, (bar_w, bar_h), interpolation=cv2.INTER_NEAREST)
    x0 = max(10, vis.shape[1] - bar_w - 20)
    y0 = vis.shape[0] - bar_h - 36
    vis[y0:y0 + bar_h, x0:x0 + bar_w] = bar
    cv2.putText(vis, "high", (x0 + bar_w - 38, y0 - 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(vis, "low", (x0, y0 - 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(vis, title, (10, vis.shape[0] - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    return vis


def _seam_maps_for_box(color_bgr, box, skeleton=None):
    """返回 (seam, local_dark, tape) 全分辨率分数图。"""
    mask = _box_roi_mask(box)
    s = DETECT_SCALE
    if s < 1.0:
        small = cv2.resize(color_bgr, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
        small_mask = cv2.resize(mask, (small.shape[1], small.shape[0]),
                                interpolation=cv2.INTER_NEAREST)
    else:
        small = color_bgr
        small_mask = mask
    cx, cy = box["rect"][0]
    _, _, lat_dx, lat_dy, lat_len, _ = _box_axes_legacy(box["rect"])
    h, w = small.shape[:2]
    exclude = _center_exclusion_mask(
        h, w, cx, cy, lat_dx, lat_dy, lat_len, s, MATTE_CENTER_EXCLUDE_FRAC)
    box_lab, box_hsv = _estimate_box_matte(small, small_mask, exclude_center=exclude)
    seam, local_dark, tape = _seam_score_maps(
        color_bgr, box_lab, box_hsv, mask, box=box, scale=1.0)
    sk = skeleton
    return (seam, local_dark, tape,
            _render_seam_heatmap(color_bgr, seam, mask, sk, title="combined"),
            _render_seam_heatmap(color_bgr, local_dark, mask, sk, title="local-dark"),
            _render_seam_heatmap(color_bgr, tape, mask, sk, title="tape=reflect"))


def _ema_score(new_s, old_s, alpha):
    if old_s is None or old_s.shape != new_s.shape:
        return new_s
    return new_s.astype(np.float32) * alpha + old_s * (1.0 - alpha)


def _fit_display(img):
    h, w = img.shape[:2]
    if w <= DISPLAY_MAX_WIDTH:
        return img
    scale = DISPLAY_MAX_WIDTH / w
    return cv2.resize(img, (DISPLAY_MAX_WIDTH, int(h * scale)), interpolation=cv2.INTER_AREA)


def _draw_overlay(color, box, result, held=False):
    out = color.copy()
    if box is not None:
        cv2.polylines(out, [cv2.boxPoints(box["rect"]).astype(int)], True, (255, 128, 0), 2)

    if result is None:
        cv2.putText(out, "no seam", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        return out

    if SHOW_ALL_CANDIDATES:
        for cand in result["candidates"]:
            (a, b) = cand["seg"]
            ts = cand["tape_score"]
            thick = 1 + int(ts * 2)
            col = (int(80 * (1 - ts)), int(80 + 175 * ts), int(180 + 75 * ts))
            cv2.line(out, (int(a[0]), int(a[1])), (int(b[0]), int(b[1])), col, thick, cv2.LINE_AA)

    if result.get("skeleton_baseline"):
        p1, p2 = result["skeleton_baseline"]
        cv2.line(out, (int(p1[0]), int(p1[1])), (int(p2[0]), int(p2[1])), (140, 140, 140), 1, cv2.LINE_AA)

    p1, p2 = result["skeleton"]
    e1, e2 = result["edge1"], result["edge2"]
    cv2.line(out, (int(p1[0]), int(p1[1])), (int(p2[0]), int(p2[1])), (0, 255, 0), 3, cv2.LINE_AA)
    if result["width_px"] > 0:
        cv2.line(out, (int(e1[0]), int(e1[1])), (int(e2[0]), int(e2[1])), (0, 255, 255), 2, cv2.LINE_AA)

    bl = result["box_lab"]
    tag = " (hold)" if held else ""
    method = result.get("detect_method", "?")
    lines = [
        f"seam score: {result.get('seam_score', result['tape_score']):.2f}  [{method}]{tag}",
        f"width: {result['width_px']:.1f} px",
        f"box L={float(np.clip(bl[0], 0, 255)):.0f}  S={result['box_hsv'][1]:.0f}",
        "green=seam (auto axis)",
    ]
    for i, txt in enumerate(lines):
        y = 28 + i * 26
        cv2.putText(out, txt, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(out, txt, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 1, cv2.LINE_AA)
    return out


def main():
    cam = D415Camera()
    cam.start()
    print("缝隙热力图测试 — ESC 退出")
    print("窗口: overlay | heatmap | heat-local-dark | heat-tape")
    print("检测: 双轴竞争谷底 + 条纹积分选方向")
    prev_skel = None
    smoothed = None
    smooth_seam = None
    smooth_dark = None
    smooth_tape = None
    lost_count = 0
    fps = 0.0
    last_t = time.time()
    panel_h, panel_w = 480, 640
    try:
        while True:
            res = cam.get_frames()
            if res is None:
                continue
            color, depth_m, _, _ = res
            box = box_detect.detect_box(depth_m)
            raw = None
            heat_combined = heat_dark = heat_tape = None
            held = False

            if box is not None:
                mask = _box_roi_mask(box)
                seam_raw, dark_raw, tape_raw, _, _, _ = _seam_maps_for_box(color, box)
                smooth_seam = _ema_score(seam_raw, smooth_seam, HEAT_SMOOTH_ALPHA)
                smooth_dark = _ema_score(dark_raw, smooth_dark, HEAT_SMOOTH_ALPHA)
                smooth_tape = _ema_score(tape_raw, smooth_tape, HEAT_SMOOTH_ALPHA)
                skel_draw = smoothed["skeleton"] if smoothed else None
                heat_combined = _render_seam_heatmap(
                    color, smooth_seam, mask, skel_draw, title="combined")
                heat_dark = _render_seam_heatmap(
                    color, smooth_dark, mask, skel_draw, title="local-dark")
                heat_tape = _render_seam_heatmap(
                    color, smooth_tape, mask, skel_draw, title="tape=reflect")

                raw = detect_seam_color_weighted(color, box, prev_skel=prev_skel)
                if raw is not None:
                    smoothed = _smooth_seam(raw, smoothed, SEAM_SMOOTH_ALPHA)
                    prev_skel = smoothed["skeleton"]
                    lost_count = 0
                    sk = smoothed["skeleton"]
                    heat_combined = _render_seam_heatmap(color, smooth_seam, mask, sk, title="combined")
                    heat_dark = _render_seam_heatmap(color, smooth_dark, mask, sk, title="local-dark")
                    heat_tape = _render_seam_heatmap(color, smooth_tape, mask, sk, title="tape=reflect")
                else:
                    lost_count += 1
                    if lost_count > LOST_RESET:
                        prev_skel = None
                        smoothed = None
            else:
                prev_skel = None
                smoothed = None
                smooth_seam = smooth_dark = smooth_tape = None
                lost_count = 0

            display = smoothed
            held = raw is None and display is not None and lost_count > 0

            overlay = _draw_overlay(color, box, display, held=held)
            now = time.time()
            fps = 0.8 * fps + 0.2 / max(now - last_t, 1e-6)
            last_t = now
            cv2.putText(overlay, f"fps: {fps:.1f}", (overlay.shape[1] - 130, 28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            cv2.imshow("seam color test - overlay", _fit_display(overlay))
            for win, img in (
                ("seam color test - heatmap", heat_combined),
                ("seam color test - heat-dark", heat_dark),
                ("seam color test - heat-tape", heat_tape),
            ):
                if img is not None:
                    cv2.imshow(win, _fit_display(img))
                else:
                    blank = np.zeros((panel_h, panel_w, 3), dtype=np.uint8)
                    cv2.putText(blank, "no box ROI", (20, panel_h // 2),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (180, 180, 180), 2)
                    cv2.imshow(win, blank)

            if (cv2.waitKey(1) & 0xFF) == 27:
                break
    finally:
        cam.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
