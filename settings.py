# settings.py
import math
import cv2
import numpy as np
import pyrealsense2 as rs
import multiprocessing as mp

def init():
    global zoom
    #[zoom factor, zoom in (T) or out (F), show them window (T or F), [y,x] of mouse]
    zoom = [1, True, False, [0,0]]
    global window_name
    window_name = "I dont know what are you doing in it?"
    
    global cores
    cores = mp.cpu_count()

    global color_compare
    color_compare = [-np.inf,-np.inf,-np.inf]

    global tire_pos
    tire_pos = []

    global mode
    mode = "None"

    global pipeline, pc, decimate, colorizer, w, h, out, state, depth_intrinsics
    class AppState:

        def __init__(self, *args, **kwargs):
            self.WIN_NAME = 'RealSense'
            self.pitch, self.yaw = math.radians(-10), math.radians(-15)
            self.translation = np.array([0, 0, -1], dtype=np.float32)
            self.distance = 2
            self.prev_mouse = 0, 0
            self.mouse_btns = [False, False, False]
            self.paused = False
            self.decimate = 0
            self.scale = True
            self.color = True

        def reset(self):
            self.pitch, self.yaw, self.distance = 0, 0, 2
            self.translation[:] = 0, 0, -1

        @property
        def rotation(self):
            Rx, _ = cv2.Rodrigues((self.pitch, 0, 0))
            Ry, _ = cv2.Rodrigues((0, self.yaw, 0))
            return np.dot(Ry, Rx).astype(np.float32)

        @property
        def pivot(self):
            return self.translation + np.array((0, 0, self.distance), dtype=np.float32)
    state = AppState()

    # Configure depth and color streams
    
    pipeline = rs.pipeline()
    config = rs.config()

    pipeline_wrapper = rs.pipeline_wrapper(pipeline)

    try:
        pipeline_profile = config.resolve(pipeline_wrapper)
        device = pipeline_profile.get_device()
        ctx = rs.context()


        config.enable_stream(rs.stream.depth, rs.format.z16, 30)
        config.enable_stream(rs.stream.color, rs.format.bgr8, 30)
    except:
        rs.config.enable_device_from_file(config, "Inputs/tire5.bag")
        config.enable_stream(rs.stream.depth, rs.format.z16, 30)
        config.enable_stream(rs.stream.color)

    # Start streaming
    pipeline.start(config)

    # Get stream profile and camera intrinsics
    profile = pipeline.get_active_profile()
    depth_profile = rs.video_stream_profile(profile.get_stream(rs.stream.depth))
    depth_intrinsics = depth_profile.get_intrinsics()
    w, h = depth_intrinsics.width, depth_intrinsics.height

    # Processing blocks
    pc = rs.pointcloud()
    decimate = rs.decimation_filter()
    decimate.set_option(rs.option.filter_magnitude, 2 ** state.decimate)
    colorizer = rs.colorizer()

    out = np.empty((h, w, 3), dtype=np.uint8)