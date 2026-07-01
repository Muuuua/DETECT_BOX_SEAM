"""仅箱子检测版本：不做缝隙检测，只圈出箱子 + 拟合箱顶平面 + 显示。

用于先把箱子颜色/HSV 标定和箱子检测调稳，再回到 main.py 调缝隙。
ESC 退出。
"""
import time

import cv2

import config
from camera import D415Camera
import box_detect
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
    print("D415 已启动（仅箱子检测），按 ESC 退出。")
    last_t = time.time()
    fps = 0.0
    try:
        while True:
            res = cam.get_frames()
            if res is None:
                continue
            color, depth_m, intrinsics, _ = res

            box = box_detect.detect_box(depth_m)
            plane_info = None
            if box is not None:
                plane_info = geometry.fit_plane(depth_m, intrinsics,
                                                seam_skel=None, box=box)

            overlay = visualize.draw_overlay(color, seam=None, metrics=None,
                                             plane_info=plane_info, box=box)
            depth_vis = visualize.draw_depth(depth_m, plane_info, box)

            now = time.time()
            fps = 0.8 * fps + 0.2 * (1.0 / max(now - last_t, 1e-6))
            last_t = now
            cv2.putText(overlay, f"fps: {fps:.1f}", (overlay.shape[1] - 140, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
            if plane_info is not None:
                cv2.putText(overlay,
                            f"plane inlier: {plane_info['inlier_ratio']*100:.0f}%",
                            (overlay.shape[1] - 300, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2,
                            cv2.LINE_AA)

            cv2.imshow("box only - color", _fit_display(overlay))
            cv2.imshow("box only - depth", _fit_display(depth_vis))
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
