# License: Apache 2.0. See LICENSE file in root directory.
# Copyright(c) 2015-2017 Intel Corporation. All Rights Reserved.

"""
OpenCV and Numpy Point cloud Software Renderer

This sample is mostly for demonstration and educational purposes.
It really doesn't offer the quality or performance that can be
achieved with hardware acceleration.

Usage:
------
Mouse: 
    Drag with left button to rotate around pivot (thick small axes), 
    with right button to translate and the wheel to zoom.

Keyboard: 
    [p]     Pause
    [r]     Reset View
    [d]     Cycle through decimation values
    [z]     Toggle point scaling
    [c]     Toggle color sourcepipeline
    [s]     Save PNG (./out.png)
    [e]     Export points to ply (./out.ply)
    [q\ESC] Quit
"""

import cv2
import numpy as np
import pyrealsense2 as rs
import settings
from d3_viewer_functions import *
from additional_functions import GetMouseInfo
from additional_functions import FindGaps

def init():
    settings.init()

    cv2.namedWindow(settings.state.WIN_NAME, cv2.WINDOW_AUTOSIZE)
    cv2.resizeWindow(settings.state.WIN_NAME, settings.w, settings.h)
    cv2.setMouseCallback(settings.state.WIN_NAME, mouse_cb)


    while True:
        # Grab camera data
        verts = []
        if not settings.state.paused:
            # Wait for a coherent pair of frames: depth and color
            frames = settings.pipeline.wait_for_frames()

            depth_frame = frames.get_depth_frame()
            color_frame = frames.get_color_frame()


            depth_frame = settings.decimate.process(depth_frame)
            depth_image = np.asanyarray(depth_frame.get_data())
            color_image = np.asanyarray(color_frame.get_data())

            depth_colormap = np.asanyarray(
                settings.colorizer.colorize(depth_frame).get_data())

            if settings.state.color:
                mapped_frame, color_source = color_frame, color_image
            else:
                mapped_frame, color_source = depth_frame, depth_colormap

            points = settings.pc.calculate(depth_frame)
            settings.pc.map_to(mapped_frame)

            # Pointcloud data to arrays
            v, t = points.get_vertices(), points.get_texture_coordinates()
            verts = np.asanyarray(v).view(np.float32).reshape(-1, 3)  # xyz
            #Scale it up
            verts = verts*10
    
            texcoords = np.asanyarray(t).view(np.float32).reshape(-1, 2)  # uv

            cv2.imshow("Color", color_image)

        # Render
        Render(settings.depth_intrinsics, verts, texcoords, color_source)
        key = cv2.waitKey(1)

        if key == ord("r"):
            settings.state.reset()

        if key == ord("p"):
            if settings.state.paused:
                cv2.setMouseCallback('Color', lambda *args : None)
                cv2.destroyWindow('Data')
            else:
                cv2.setMouseCallback('Color', GetMouseInfo, [color_source, verts,0,0])   
                GetMouseInfo(0,0,0,0,[color_source, verts])
            settings.state.paused ^= True

        if key == ord("d"):
            settings.state.decimate = (settings.state.decimate + 1) % 3
            settings.decimate.set_option(rs.option.filter_magnitude, 2 ** settings.state.decimate)
            print(settings.state.decimate)

        if key == ord("z"):
            settings.state.scale ^= True

        if key == ord("c"):
            settings.state.color ^= True

        if key == ord("s"):
            cv2.imwrite('./out.png', settings.out)

        if key == ord("e"):
            points.export_to_ply('./out.ply', mapped_frame)

        if key in (27, ord("q"), 8) or cv2.getWindowProperty(settings.state.WIN_NAME, cv2.WND_PROP_AUTOSIZE) < 0:
            #Stop streaming
            cv2.destroyAllWindows()
            settings.pipeline.stop()
            return