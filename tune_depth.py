"""深度箱子检测调参工具。

实时显示深度伪彩 + 检测到的箱子掩膜/框，滑块调节深度容差。
按 s 保存当前参数到 depth_calib.json（box_detect 不会自动读，仅作记录）；
按 q/ESC 退出。
"""
import json

import cv2
import numpy as np

import config
from camera import D415Camera
import box_detect


WIN = "tune_depth"


def _fit_display(img):
    h, w = img.shape[:2]
    if w <= config.DISPLAY_MAX_WIDTH:
        return img
    scale = config.DISPLAY_MAX_WIDTH / w
    return cv2.resize(img, (config.DISPLAY_MAX_WIDTH, int(h * scale)),
                      interpolation=cv2.INTER_AREA)


def _depth_pseudo(depth_m):
    d = depth_m.copy()
    d[d <= 0] = 0
    dmax = config.DEPTH_MAX_M
    vis = (d / dmax * 255).clip(0, 255).astype(np.uint8)
    color = cv2.applyColorMap(vis, cv2.COLORMAP_JET)
    color[d <= 0] = (0, 0, 0)
    return color


def main():
    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
    cv2.createTrackbar("tol_mm", WIN, int(config.BOX_DEPTH_TOL_M * 1000), 200, lambda x: None)
    cv2.createTrackbar("ref_pct", WIN, config.BOX_DEPTH_REF_PERCENTILE, 30, lambda x: None)

    cam = D415Camera()
    cam.start()
    print("tune_depth 已启动。s=保存 q/ESC=退出")
    try:
        while True:
            res = cam.get_frames()
            if res is None:
                continue
            color, depth_m, intrinsics, _ = res

            # 临时覆盖配置
            config.BOX_DEPTH_TOL_M = cv2.getTrackbarPos("tol_mm", WIN) / 1000.0
            config.BOX_DEPTH_REF_PERCENTILE = cv2.getTrackbarPos("ref_pct", WIN)

            box = box_detect.detect_box(depth_m)
            pseudo = _depth_pseudo(depth_m)
            if box is not None:
                rect_pts = cv2.boxPoints(box["rect"]).astype(int)
                cv2.polylines(pseudo, [rect_pts], True, (0, 255, 0), 2)
                cv2.polylines(color, [rect_pts], True, (0, 255, 0), 2)
                cv2.putText(pseudo, f"box OK  d_ref={box['d_ref']:.3f}m",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                            (0, 255, 0), 2, cv2.LINE_AA)
            else:
                cv2.putText(pseudo, "no box", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, cv2.LINE_AA)

            info = f"tol={config.BOX_DEPTH_TOL_M*1000:.0f}mm ref_pct={config.BOX_DEPTH_REF_PERCENTILE}"
            cv2.putText(pseudo, info, (10, pseudo.shape[0] - 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)

            cv2.imshow(WIN, _fit_display(pseudo))
            cv2.imshow("tune_depth - color", _fit_display(color))
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord('q')):
                break
            elif key == ord('s'):
                with open("depth_calib.json", "w", encoding="utf-8") as f:
                    json.dump({
                        "BOX_DEPTH_TOL_M": config.BOX_DEPTH_TOL_M,
                        "BOX_DEPTH_REF_PERCENTILE": config.BOX_DEPTH_REF_PERCENTILE,
                    }, f)
                print(f"[tune_depth] 已保存 depth_calib.json: {info}")
    except KeyboardInterrupt:
        pass
    finally:
        cam.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
