"""箱顶平面拟合 + deproject + 3D 几何量计算。"""
import numpy as np
import pyrealsense2 as rs
from sklearn.linear_model import RANSACRegressor

import config


def _pixel_to_3d(u, v, depth_m, intrinsics):
    """单像素 -> 相机坐标系 3D 点 (x,y,z) 米；depth<=0 返回 None。"""
    d = float(depth_m)
    if d <= 0:
        return None
    pt = rs.rs2_deproject_pixel_to_point(intrinsics, [float(u), float(v)], d)
    return np.array(pt, dtype=np.float32)


def _pixels_to_points(uv, depth_m, intrinsics):
    """批量像素 -> 3D 点，返回 (N,3) 数组，剔除无效深度。

    向量化实现（忽略畸变，箱顶通常在中心区域，畸变影响可忽略），用于平面拟合提速。
    """
    uv = np.asarray(uv)
    if uv.shape[0] == 0:
        return np.zeros((0, 3), dtype=np.float32)
    u = uv[:, 0].astype(np.float32)
    v = uv[:, 1].astype(np.float32)
    iu = np.round(u).astype(int)
    iv = np.round(v).astype(int)
    iu = np.clip(iu, 0, depth_m.shape[1] - 1)
    iv = np.clip(iv, 0, depth_m.shape[0] - 1)
    d = depth_m[iv, iu]
    valid = d > 0
    if not np.any(valid):
        return np.zeros((0, 3), dtype=np.float32)
    fx = float(intrinsics.fx)
    fy = float(intrinsics.fy)
    ppx = float(intrinsics.ppx)
    ppy = float(intrinsics.ppy)
    u_v = u[valid]
    v_v = v[valid]
    d_v = d[valid]
    x = (u_v - ppx) * d_v / fx
    y = (v_v - ppy) * d_v / fy
    return np.stack([x, y, d_v], axis=0).T.astype(np.float32)


def fit_plane(depth_m, intrinsics, seam_skel=None, box=None):
    """在箱顶 ROI 内拟合平面。

    box: box_detect.detect_box 返回的 dict（含全分辨率 mask），平面拟合仅在箱子内。
    seam_skel: (p1,p2) 用于剔除缝隙附近像素，避免胶带干扰平面拟合。
    返回 dict: {plane: (a,b,c,d) 平面方程 ax+by+cz+d=0, normal, inlier_ratio} 或 None。
    """
    h, w = depth_m.shape
    stride = config.PLANE_STRIDE
    ys, xs = np.mgrid[0:h:stride, 0:w:stride]
    sub = depth_m[ys, xs]
    valid = (sub > config.WORK_DISTANCE[0]) & (sub < config.WORK_DISTANCE[1])

    # 限定到箱子掩膜内
    if box is not None:
        small_mask = box["mask"][ys, xs] > 0
        valid = valid & small_mask
    else:
        m = config.ROI_MARGIN
        in_margin = (xs >= m) & (xs < w - m) & (ys >= m) & (ys < h - m)
        valid = valid & in_margin

    xs_v = xs[valid]
    ys_v = ys[valid]
    ds_v = sub[valid]

    if len(xs_v) < config.MIN_PLANE_POINTS:
        return None

    # 剔除缝隙附近像素（沿骨架一定带状区域）
    if seam_skel is not None:
        (x1, y1), (x2, y2) = seam_skel
        dx, dy = x2 - x1, y2 - y1
        L = np.hypot(dx, dy)
        if L > 1:
            tx, ty = dx / L, dy / L
            nx, ny = -ty, tx
            px = xs_v - x1
            py = ys_v - y1
            along = px * tx + py * ty
            perp = px * nx + py * ny
            on_seg = (along >= -10) & (along <= L + 10)
            near = on_seg & (np.abs(perp) < config.PROFILE_HALF_WIDTH)
            keep = ~near
            xs_v = xs_v[keep]
            ys_v = ys_v[keep]
            ds_v = ds_v[keep]
            if len(xs_v) < config.MIN_PLANE_POINTS:
                return None

    uv = np.stack([xs_v, ys_v], axis=1)
    pts = _pixels_to_points(uv, depth_m, intrinsics)
    if pts.shape[0] < config.MIN_PLANE_POINTS:
        return None

    X = pts[:, :2]
    z = pts[:, 2]
    ransac = RANSACRegressor(
        residual_threshold=config.RANSAC_RESIDUAL_MM / 1000.0,
        max_trials=config.RANSAC_MAX_TRIALS,
        random_state=0,
    )
    try:
        ransac.fit(X, z)
    except Exception:
        return None
    a, b = ransac.estimator_.coef_
    c_const = ransac.estimator_.intercept_
    # 平面: z = a*x + b*y + c_const -> -a*x - b*y + 1*z - c_const = 0
    plane = np.array([-a, -b, 1.0, -c_const], dtype=np.float32)
    inlier_ratio = float(ransac.inlier_mask_.mean())
    return {"plane": plane, "normal": plane[:3] / np.linalg.norm(plane[:3]),
            "inlier_ratio": inlier_ratio}


def point_to_plane_dist(p, plane):
    """点 p=(x,y,z) 到平面 ax+by+cz+d=0 的有向距离。"""
    n = plane[:3]
    return float(np.dot(n, p) + plane[3]) / float(np.linalg.norm(n))


def compute_metrics(seam, depth_m, intrinsics, plane_info):
    """计算缝隙 3D 几何量。

    返回 dict:
      p1_3d, p2_3d: 骨架端点 3D 坐标 (米)
      length_m: 缝隙 3D 长度
      width_m: 缝隙宽度（3D）
      width_px: 像素宽度
      height_diff_m: 胶带面相对平面高度差（有向）
      height_reliable: |height_diff| 是否高于噪声阈值
      edge1_3d, edge2_3d: 边缘 3D 点
    失败返回 None。
    """
    if seam is None or plane_info is None:
        return None
    (x1, y1), (x2, y2) = seam["skeleton"]
    e1, e2 = seam["edge1"], seam["edge2"]

    def safe_3d(u, v):
        iu, iv = int(round(u)), int(round(v))
        if not (0 <= iu < depth_m.shape[1] and 0 <= iv < depth_m.shape[0]):
            return None
        d = depth_m[iv, iu]
        if d <= 0:
            return None
        return _pixel_to_3d(u, v, d, intrinsics)

    p1 = safe_3d(x1, y1)
    p2 = safe_3d(x2, y2)
    e1p = safe_3d(*e1)
    e2p = safe_3d(*e2)
    if p1 is None or p2 is None or e1p is None or e2p is None:
        return None

    length_m = float(np.linalg.norm(p1 - p2))
    width_m = float(np.linalg.norm(e1p - e2p))

    # 胶带面高度差：取骨架中点 3D 点到平面距离
    mid_u = (x1 + x2) / 2.0
    mid_v = (y1 + y2) / 2.0
    mid = safe_3d(mid_u, mid_v)
    if mid is None:
        mid = 0.5 * (p1 + p2)
    h = point_to_plane_dist(mid, plane_info["plane"])
    reliable = abs(h) >= config.DEPTH_NOISE_M

    return {
        "p1_3d": p1, "p2_3d": p2,
        "edge1_3d": e1p, "edge2_3d": e2p,
        "length_m": length_m,
        "width_m": width_m,
        "width_px": seam["width_px"],
        "height_diff_m": h,
        "height_reliable": bool(reliable),
    }
