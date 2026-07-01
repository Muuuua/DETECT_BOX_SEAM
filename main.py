"""实时主循环：采集 -> 缝隙检测 -> 平面拟合 -> 几何量 -> 叠加显示。ESC 退出。"""
import time

import cv2

import config
from camera import D415Camera
import box_detect
import seam_detect
import geometry
import visualize


def _fit_display(img):
    """等比缩放到 DISPLAY_MAX_WIDTH 以内以适应屏幕。"""
    h, w = img.shape[:2]
    if w <= config.DISPLAY_MAX_WIDTH:
        return img
    scale = config.DISPLAY_MAX_WIDTH / w
    return cv2.resize(img, (config.DISPLAY_MAX_WIDTH, int(h * scale)),
                      interpolation=cv2.INTER_AREA)


def _ema_point(new_pt, old_pt, alpha):
    return (new_pt[0] * alpha + old_pt[0] * (1 - alpha),
            new_pt[1] * alpha + old_pt[1] * (1 - alpha))


def _smooth_seam(new_seam, smoothed, alpha):
    """对新检测缝隙做 EMA 平滑。smoothed 为 None 时直接用新值。"""
    if smoothed is None:
        return {
            "skeleton": new_seam["skeleton"],
            "edge1": new_seam["edge1"],
            "edge2": new_seam["edge2"],
            "width_px": new_seam["width_px"],
            "line_angle_deg": new_seam["line_angle_deg"],
        }
    p1 = _ema_point(new_seam["skeleton"][0], smoothed["skeleton"][0], alpha)
    p2 = _ema_point(new_seam["skeleton"][1], smoothed["skeleton"][1], alpha)
    e1 = _ema_point(new_seam["edge1"], smoothed["edge1"], alpha)
    e2 = _ema_point(new_seam["edge2"], smoothed["edge2"], alpha)
    w = new_seam["width_px"] * alpha + smoothed["width_px"] * (1 - alpha)
    return {
        "skeleton": (p1, p2),
        "edge1": e1,
        "edge2": e2,
        "width_px": w,
        "line_angle_deg": new_seam["line_angle_deg"],
    }


def main():
    cam = D415Camera()
    cam.start()
    print("D415 已启动，按 ESC 退出。")
    last_t = time.time()
    fps = 0.0
    prev_skel = None        # 用于 seam_detect 的时序锁定（平滑后位置）
    smoothed = None         # EMA 平滑后的缝隙
    lost_count = 0
    LOST_RESET = 15
    try:
        while True:
            res = cam.get_frames()
            if res is None:
                continue
            color, depth_m, intrinsics, _ = res

            box = box_detect.detect_box(depth_m)
            if box is not None:
                raw_seam = seam_detect.detect_seam(color, box=box, prev_skel=prev_skel)
                if raw_seam is not None:
                    smoothed = _smooth_seam(raw_seam, smoothed, config.SEAM_SMOOTH_ALPHA)
                    seam = smoothed
                    prev_skel = smoothed["skeleton"]
                    lost_count = 0
                else:
                    lost_count += 1
                    if lost_count > LOST_RESET:
                        prev_skel = None
                        smoothed = None
                    seam = smoothed  # 丢失帧沿用上次平滑结果（可为 None）
                plane_info = geometry.fit_plane(
                    depth_m, intrinsics,
                    seam["skeleton"] if seam else None, box=box)
                metrics = geometry.compute_metrics(seam, depth_m, intrinsics,
                                                   plane_info)
            else:
                seam = None
                plane_info = None
                metrics = None
                prev_skel = None
                smoothed = None
                lost_count = 0

            overlay = visualize.draw_overlay(color, seam, metrics, plane_info, box)
            depth_vis = visualize.draw_depth(depth_m, plane_info, box)

            now = time.time()
            fps = 0.8 * fps + 0.2 * (1.0 / max(now - last_t, 1e-6))
            last_t = now
            cv2.putText(overlay, f"fps: {fps:.1f}", (overlay.shape[1] - 140, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

            cv2.imshow("box seam detect - color", _fit_display(overlay))
            cv2.imshow("box seam detect - depth", _fit_display(depth_vis))
            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # ESC
                break
    except KeyboardInterrupt:
        pass
    finally:
        cam.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
