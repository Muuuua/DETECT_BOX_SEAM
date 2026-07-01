"""D415 采集与对齐封装。"""
import pyrealsense2 as rs
import numpy as np

import config


class D415Camera:
    def __init__(self):
        self.pipeline = None
        self.config = None
        self.align = None
        self.depth_scale = 0.0
        self.intrinsics = None

    def start(self):
        self.pipeline = rs.pipeline()
        cfg = rs.config()
        cfg.enable_stream(rs.stream.color, config.COLOR_WIDTH, config.COLOR_HEIGHT,
                          rs.format.bgr8, config.COLOR_FPS)
        cfg.enable_stream(rs.stream.depth, config.DEPTH_WIDTH, config.DEPTH_HEIGHT,
                          rs.format.z16, config.DEPTH_FPS)
        self.config = cfg

        profile = self.pipeline.start(cfg)
        dev = profile.get_device()
        depth_sensor = dev.first_depth_sensor()
        if depth_sensor.supports(rs.option.visual_preset):
            depth_sensor.set_option(rs.option.visual_preset,
                                    rs.rs400_visual_preset.high_accuracy)
        self.depth_scale = depth_sensor.get_depth_scale()

        # color 流内参（用于 deproject，因 depth 已对齐到 color）
        color_profile = profile.get_stream(rs.stream.color)
        self.intrinsics = color_profile.as_video_stream_profile().get_intrinsics()

        self.align = rs.align(rs.stream.color)

    def get_frames(self):
        """返回 (color_bgr HxWx3, depth_m HxW float32, intrinsics, depth_scale)。
        depth_m 已对齐到 color 像素，单位米，无效为 0。"""
        frames = self.pipeline.wait_for_frames()
        aligned = self.align.process(frames)
        color_frame = aligned.get_color_frame()
        depth_frame = aligned.get_depth_frame()
        if not color_frame or not depth_frame:
            return None
        color = np.asanyarray(color_frame.get_data())
        depth_m = np.asanyarray(depth_frame.get_data()).astype(np.float32) * self.depth_scale
        return color, depth_m, self.intrinsics, self.depth_scale

    def stop(self):
        if self.pipeline is not None:
            self.pipeline.stop()
            self.pipeline = None
