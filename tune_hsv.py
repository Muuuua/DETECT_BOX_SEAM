"""HSV 实时调参工具。

运行后出现滑块窗口，实时调整 H/S/V 上下限，并把 mask 与检测结果叠加显示。
按 s 保存当前阈值到 hsv_calib.json（box_detect 会自动优先读取）；
按 r 重新读取已保存的标定；按 q/ESC 退出。
"""
import json
import os

import cv2
import numpy as np

import config
from camera import D415Camera
import box_detect


WIN = "tune_hsv"
TRACKBARS = [
    ("H_low", 0, 179, config.HSV_LOWER[0]),
    ("H_high", 0, 179, config.HSV_UPPER[0]),
    ("S_low", 0, 255, config.HSV_LOWER[1]),
    ("S_high", 0, 255, config.HSV_UPPER[1]),
    ("V_low", 0, 255, config.HSV_LOWER[2]),
    ("V_high", 0, 255, config.HSV_UPPER[2]),
]


def _fit_display(img):
    """等比缩放到 DISPLAY_MAX_WIDTH 以内以适应屏幕。"""
    h, w = img.shape[:2]
    if w <= config.DISPLAY_MAX_WIDTH:
        return img
    scale = config.DISPLAY_MAX_WIDTH / w
    return cv2.resize(img, (config.DISPLAY_MAX_WIDTH, int(h * scale)),
                      interpolation=cv2.INTER_AREA)


def _read_trackbar():
    vals = {name: cv2.getTrackbarPos(name, WIN) for name, _, _, _ in TRACKBARS}
    lower = (vals["H_low"], vals["S_low"], vals["V_low"])
    upper = (vals["H_high"], vals["S_high"], vals["V_high"])
    return lower, upper


def _save(lower, upper):
    data = {"lower": list(lower), "upper": list(upper)}
    with open(config.HSV_CALIB_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f)
    print(f"[tune_hsv] 已保存到 {config.HSV_CALIB_FILE}: lower={lower} upper={upper}")


def main():
    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
    for name, lo, hi, default in TRACKBARS:
        cv2.createTrackbar(name, WIN, int(default), hi, lambda x: None)

    cam = D415Camera()
    cam.start()
    print("tune_hsv 已启动。s=保存 r=重载 q/ESC=退出")
    try:
        while True:
            res = cam.get_frames()
            if res is None:
                continue
            color, depth_m, intrinsics, _ = res

            lower, upper = _read_trackbar()
            # 临时覆盖 box_detect 的标定缓存
            box_detect._CALIB = (lower, upper)

            hsv = cv2.cvtColor(color, cv2.COLOR_BGR2HSV)
            mask = cv2.inRange(hsv, np.array(lower, dtype=np.uint8),
                               np.array(upper, dtype=np.uint8))
            kopen = cv2.getStructuringElement(
                cv2.MORPH_RECT, (config.BOX_OPEN_KERNEL, config.BOX_OPEN_KERNEL))
            kclose = cv2.getStructuringElement(
                cv2.MORPH_RECT, (config.BOX_CLOSE_KERNEL, config.BOX_CLOSE_KERNEL))
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kopen)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kclose)

            mask_bgr = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
            overlay = cv2.addWeighted(color, 0.6, mask_bgr, 0.4, 0)

            box = box_detect.detect_box(color)
            if box is not None:
                rect_pts = cv2.boxPoints(box["rect"]).astype(int)
                cv2.polylines(overlay, [rect_pts], True, (0, 255, 0), 2)
                cv2.putText(overlay, "box OK", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
            else:
                cv2.putText(overlay, "no box", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, cv2.LINE_AA)

            info = f"L={lower} U={upper}"
            cv2.putText(overlay, info, (10, overlay.shape[0] - 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)

            cv2.imshow(WIN, _fit_display(overlay))
            cv2.imshow("tune_hsv - mask", _fit_display(mask))

            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord('q')):
                break
            elif key == ord('s'):
                _save(lower, upper)
            elif key == ord('r'):
                box_detect.reload_calib()
                print("[tune_hsv] 已重载标定文件")
    except KeyboardInterrupt:
        pass
    finally:
        cam.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
