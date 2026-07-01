"""可视化：RGB 叠加标注 + 深度伪彩窗口。"""
import numpy as np
import cv2

import config


def draw_overlay(color_bgr, seam, metrics, plane_info, box=None):
    """在 color 上绘制箱子框 + 缝隙标注 + 几何量文本，返回新图。"""
    out = color_bgr.copy()

    if box is not None:
        rect_pts = cv2.boxPoints(box["rect"]).astype(int)
        cv2.polylines(out, [rect_pts], True, (255, 0, 0), 2)

    if seam is not None:
        (x1, y1), (x2, y2) = seam["skeleton"]
        e1, e2 = seam["edge1"], seam["edge2"]
        cv2.line(out, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        cv2.line(out, (int(e1[0]), int(e1[1])), (int(e2[0]), int(e2[1])),
                 (0, 255, 255), 1)
        cv2.circle(out, (int(x1), int(y1)), 6, (0, 0, 255), -1)
        cv2.circle(out, (int(x2), int(y2)), 6, (0, 0, 255), -1)

    lines = []
    if box is None:
        lines.append("no box found")
    elif metrics is not None:
        p1 = metrics["p1_3d"]
        p2 = metrics["p2_3d"]
        lines.append(f"p1 (m): ({p1[0]:.3f}, {p1[1]:.3f}, {p1[2]:.3f})")
        lines.append(f"p2 (m): ({p2[0]:.3f}, {p2[1]:.3f}, {p2[2]:.3f})")
        lines.append(f"length: {metrics['length_m']*1000:.1f} mm")
        lines.append(f"width:  {metrics['width_m']*1000:.1f} mm  ({metrics['width_px']:.1f} px)")
        hd = metrics["height_diff_m"] * 1000.0
        tag = "" if metrics["height_reliable"] else " (unreliable)"
        lines.append(f"height diff: {hd:+.2f} mm{tag}")
        if plane_info is not None:
            lines.append(f"plane inlier: {plane_info['inlier_ratio']*100:.0f}%")
    elif seam is None:
        lines.append("box ok, no seam found")
    else:
        lines.append("box ok, no metrics (depth/plane missing)")

    y0 = 30
    for i, txt in enumerate(lines):
        cv2.putText(out, txt, (10, y0 + i * 26),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(out, txt, (10, y0 + i * 26),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 1, cv2.LINE_AA)
    return out


def draw_depth(depth_m, plane_info=None, box=None):
    """深度伪彩图，可选叠加箱子轮廓。返回 BGR 图。"""
    d = depth_m.copy()
    d[d <= 0] = 0
    d[d > config.DEPTH_MAX_M] = 0
    if d.max() > 0:
        vis = (d / config.DEPTH_MAX_M * 255).clip(0, 255).astype(np.uint8)
    else:
        vis = np.zeros_like(d, dtype=np.uint8)
    color = cv2.applyColorMap(vis, cv2.COLORMAP_JET)
    color[d <= 0] = (0, 0, 0)
    if box is not None:
        rect_pts = cv2.boxPoints(box["rect"]).astype(int)
        cv2.polylines(color, [rect_pts], True, (255, 0, 0), 2)
    return color
