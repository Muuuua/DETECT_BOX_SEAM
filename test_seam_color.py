"""独立测试：箱子 ROI 内缝隙检测 + 颜色加权。

透明胶带场景：缝隙压痕偏暗、胶带面可能反光，热力图与加权同时考虑两种偏离。
含 EMA 时序平滑。运行: python test_seam_color.py  ESC 退出。
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
SEAM_VERTICAL_TOL_DEG = 10.0
SEAM_CENTER_TOL_FRAC = 0.5
SEAM_LOCK_DIST_FRAC = 0.06
SEAM_LOCK_ANGLE_DEG = 8.0
BOX_MASK_ERODE = 15
PROFILE_SAMPLES = 41
PROFILE_HALF_WIDTH = 25
DISPLAY_MAX_WIDTH = 1280

# ---- 缝隙颜色特征（相对哑光箱顶）----
# 压痕/折线：比箱顶更暗（透明胶带下常见阴影缝）
CREASE_DARK_SPAN = 40.0
CREASE_WEIGHT = 0.55
# 透明膜反光：更亮 + 饱和度更低
TAPE_BRIGHT_SPAN = 25.0
TAPE_BRIGHT_MAX = 45.0
TAPE_SAT_SPAN = 35.0
BRIGHT_TAPE_WEIGHT = 0.45
MIN_COLOR_WEIGHT = 0.25
COLOR_WEIGHT_GAIN = 3.0
SEAM_SCORE_FLOOR = 0.05
BROWN_L_MIN = 40
BROWN_L_MAX = 175
LINE_COLOR_SAMPLES = 24
LINE_PATCH_RADIUS = 2

# ---- 时序平滑（减轻跳动）----
SEAM_SMOOTH_ALPHA = 0.25
HEAT_SMOOTH_ALPHA = 0.35
LOST_RESET = 15
SHOW_ALL_CANDIDATES = False


def _near_vertical_angle(dx, dy):
    return abs(math.degrees(math.atan2(abs(dx), abs(dy))))


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


def _estimate_box_matte(bgr, mask):
    """ROI 内哑光箱顶 Lab + HSV 参考（排除强高光）。"""
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    L = lab[:, :, 0]
    valid = mask > 0
    matte = valid & (L >= BROWN_L_MIN) & (L <= BROWN_L_MAX)
    if int(matte.sum()) < 80:
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


def _pixel_seam_score(lab, hsv, box_lab, box_hsv):
    """逐像素缝隙疑似度 0~1：压痕变暗 或 透明膜反光，取较强者。"""
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
    return seam.astype(np.float32), dark, tape_bright


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


def detect_seam_color_weighted(color_bgr, box, prev_skel=None):
    s = DETECT_SCALE
    small = cv2.resize(color_bgr, None, fx=s, fy=s, interpolation=cv2.INTER_AREA) if s < 1.0 else color_bgr
    inv = 1.0 / s

    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=CLAHE_CLIP, tileGridSize=CLAHE_TILE)
    enhanced = clahe.apply(gray)
    edges = cv2.Canny(enhanced, CANNY_LOW, CANNY_HIGH)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (MORPH_KERNEL, MORPH_KERNEL))
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

    center_s = short_side_s = long_side_s = None
    small_mask = None
    box_lab = box_hsv = None
    if box is not None:
        erode_px = max(1, int(round(BOX_MASK_ERODE * s)))
        kern = cv2.getStructuringElement(cv2.MORPH_RECT, (erode_px, erode_px))
        full_mask_erode = cv2.erode(box["mask"], kern)
        small_mask = cv2.resize(full_mask_erode, (small.shape[1], small.shape[0]),
                                interpolation=cv2.INTER_NEAREST)
        edges = cv2.bitwise_and(edges, small_mask)
        box_lab, box_hsv = _estimate_box_matte(small, small_mask)
        _, _, short_side, long_side = _box_long_axis(box["rect"])
        cx, cy = box["rect"][0]
        center_s = (cx * s, cy * s)
        short_side_s = short_side * s
        long_side_s = long_side * s

    lab_small = cv2.cvtColor(small, cv2.COLOR_BGR2LAB)
    hsv_small = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)

    lines = cv2.HoughLinesP(
        edges, HOUGH_RHO, np.pi / 180 * HOUGH_THETA_DEG, HOUGH_THRESHOLD,
        minLineLength=HOUGH_MIN_LINE_LEN * s, maxLineGap=HOUGH_MAX_LINE_GAP * s,
    )
    if lines is None or box_lab is None:
        return None

    raw_candidates = []
    weighted_candidates = []
    for ln in lines[:, 0, :]:
        x1, y1, x2, y2 = map(float, ln)
        dx, dy = x2 - x1, y2 - y1
        L = math.hypot(dx, dy)
        if L < 1:
            continue
        if _near_vertical_angle(dx, dy) > SEAM_VERTICAL_TOL_DEG:
            continue
        if center_s is not None and short_side_s is not None:
            d = _point_line_dist(center_s[0], center_s[1], x1, y1, x2, y2)
            if d > SEAM_CENTER_TOL_FRAC * (short_side_s / 2.0):
                continue
        seam_score = _line_seam_score(lab_small, hsv_small, x1, y1, x2, y2, box_lab, box_hsv)
        color_factor = MIN_COLOR_WEIGHT + COLOR_WEIGHT_GAIN * max(seam_score, SEAM_SCORE_FLOOR)
        combined_w = L * color_factor
        raw_candidates.append((x1, y1, x2, y2, L, seam_score, combined_w))
        weighted_candidates.append((x1, y1, x2, y2, combined_w, seam_score))

    if not weighted_candidates:
        return None

    used = weighted_candidates
    if prev_skel is not None and long_side_s is not None:
        (px1, py1), (px2, py2) = prev_skel
        pmx = (px1 + px2) / 2.0 * s
        pmy = (py1 + py2) / 2.0 * s
        pdx, pdy = (px2 - px1) * s, (py2 - py1) * s
        lock_dist = SEAM_LOCK_DIST_FRAC * long_side_s
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

    (sx1, sy1), (sx2, sy2) = skel_color
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

    best = max(used, key=lambda c: c[4])
    return {
        "skeleton": (p1, p2),
        "skeleton_baseline": (
            (skel_base[0][0] * inv, skel_base[0][1] * inv),
            (skel_base[1][0] * inv, skel_base[1][1] * inv),
        ) if skel_base else None,
        "edge1": edge1,
        "edge2": edge2,
        "width_px": width_px,
        "tape_score": best[5],
        "seam_score": best[5],
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


def _seam_score_map(bgr, box_lab, box_hsv, mask):
    """ROI 内缝隙疑似度 float 图 0~1。"""
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    seam, _, _ = _pixel_seam_score(lab, hsv, box_lab, box_hsv)
    seam[mask == 0] = 0.0
    return seam


def _render_seam_heatmap(bgr, score, mask, skeleton=None):
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
    cv2.putText(vis, "yellow=seam likely", (10, vis.shape[0] - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    return vis


def _seam_map_for_box(color_bgr, box, skeleton=None):
    mask = _box_roi_mask(box)
    s = DETECT_SCALE
    if s < 1.0:
        small = cv2.resize(color_bgr, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
        small_mask = cv2.resize(mask, (small.shape[1], small.shape[0]),
                                interpolation=cv2.INTER_NEAREST)
    else:
        small = color_bgr
        small_mask = mask
    box_lab, box_hsv = _estimate_box_matte(small, small_mask)
    score = _seam_score_map(color_bgr, box_lab, box_hsv, mask)
    return score, _render_seam_heatmap(color_bgr, score, mask, skeleton)


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
    lines = [
        f"seam score: {result.get('seam_score', result['tape_score']):.2f}{tag}",
        f"width: {result['width_px']:.1f} px",
        f"box L*={bl[0]:.0f}  S={result['box_hsv'][1]:.0f}",
        "green=detected seam  heat=yellow high",
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
    print("热力图: 黄/白=缝隙疑似高  蓝/紫=接近箱顶")
    prev_skel = None
    smoothed = None
    smooth_score = None
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
            heat_vis = None
            held = False

            if box is not None:
                score_raw, _ = _seam_map_for_box(color, box)
                smooth_score = _ema_score(score_raw, smooth_score, HEAT_SMOOTH_ALPHA)
                mask = _box_roi_mask(box)
                skel_draw = smoothed["skeleton"] if smoothed else None
                heat_vis = _render_seam_heatmap(color, smooth_score, mask, skeleton=skel_draw)
                raw = detect_seam_color_weighted(color, box, prev_skel=prev_skel)
                if raw is not None:
                    smoothed = _smooth_seam(raw, smoothed, SEAM_SMOOTH_ALPHA)
                    prev_skel = smoothed["skeleton"]
                    lost_count = 0
                    heat_vis = _render_seam_heatmap(
                        color, smooth_score, mask, skeleton=smoothed["skeleton"])
                else:
                    lost_count += 1
                    if lost_count > LOST_RESET:
                        prev_skel = None
                        smoothed = None
            else:
                prev_skel = None
                smoothed = None
                smooth_score = None
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
            if heat_vis is not None:
                cv2.imshow("seam color test - heatmap", _fit_display(heat_vis))
            else:
                blank = np.zeros((panel_h, panel_w, 3), dtype=np.uint8)
                cv2.putText(blank, "no box ROI", (20, panel_h // 2),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (180, 180, 180), 2)
                cv2.imshow("seam color test - heatmap", blank)

            if (cv2.waitKey(1) & 0xFF) == 27:
                break
    finally:
        cam.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
