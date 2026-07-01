"""无时序平滑的原始检测版本：每帧独立检测，不做 EMA、不做时序锁定。

用于对照观察原始检测的跳变行为，或在调参时看单帧响应。
ESC 退出。
"""
import time

import cv2

import config
from camera import D415Camera
import box_detect
import seam_detect
import geometry
import visualize


def _fit_display(img):
    h, w = img.shape[:2]
    if w <= config.DISPLAY_MAX_WIDTH:
        return img
    scale = config.DISPLAY_MAX_WIDTH / w
    return cv2.resize(img, (config.DISPLAY_MAX_WIDTH, int(h * scale)),
                      interpolation=cv2.INTER_AREA)


def main():
    cam = D415Camera()
    cam.start()
    print("D415 已启动（原始检测，无平滑），按 ESC 退出。")
    last_t = time.time()
    fps = 0.0
    try:
        while True:
            res = cam.get_frames()
            if res is None:
                continue
            color, depth_m, intrinsics, _ = res

            box = box_detect.detect_box(depth_m)
            if box is not None:
                # prev_skel=None：禁用时序锁定，纯单帧检测
                seam = seam_detect.detect_seam(color, box=box, prev_skel=None)
                plane_info = geometry.fit_plane(
                    depth_m, intrinsics,
                    seam["skeleton"] if seam else None, box=box)
                metrics = geometry.compute_metrics(seam, depth_m, intrinsics,
                                                   plane_info)
            else:
                seam = None
                plane_info = None
                metrics = None

            overlay = visualize.draw_overlay(color, seam, metrics, plane_info, box)
            depth_vis = visualize.draw_depth(depth_m, plane_info, box)

            now = time.time()
            fps = 0.8 * fps + 0.2 * (1.0 / max(now - last_t, 1e-6))
            last_t = now
            cv2.putText(overlay, f"fps: {fps:.1f} (raw)", (overlay.shape[1] - 200, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

            cv2.imshow("box seam detect RAW - color", _fit_display(overlay))
            cv2.imshow("box seam detect RAW - depth", _fit_display(depth_vis))
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
