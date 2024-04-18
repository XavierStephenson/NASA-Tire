import numpy as np
import cv2
import time
import cv2
import numpy as np
import pyrealsense2 as rs
import os
import settings
from d3_viewer_functions import *
from additional_functions import *
from menu_functions import *



def init():
    settings.init()
    while True:
        
        outfile = GetFolder()
        if not outfile:
            #cv2.destroyAllWindows()
            return
        outfile += 'Raw.npz'
        #selected temp files
       
        #If not a temp file, load all the frames
        npzfile = np.load(outfile)
        all_verts = npzfile['all_verts']
        all_color_images = npzfile['all_color_images']
        all_depth_colormaps = npzfile['all_depth_colormaps']
        all_footprints = npzfile['all_footprints']

        cv2.namedWindow("Color", cv2.WINDOW_AUTOSIZE)


        frame_numb = 0
        frameR = 5
        verts = []
        power = 1
        h,w = all_color_images[0].shape[:2]
        last_slash = len(outfile) - 1 - outfile[::-1].index('/')
        path = outfile[0:last_slash]

        try:
            os.makedirs(path+'/3D Objects')
        except:
            pass

        #display the stuff
        while True:

            if not settings.state.paused:
                time.sleep(1/frameR)
                frame_numb +=1
                if frame_numb >= len(all_verts):
                    frame_numb = 0
            

                depth_colormap = all_depth_colormaps[frame_numb]
                color_image = all_color_images[frame_numb]
                footprint = all_footprints[frame_numb].copy()


                verts = all_verts[frame_numb]  # xyz
                
                if settings.state.color:
                    color_source = color_image
                    #truth_tire, truth_floor = Arrays(verts.copy(), color_image.copy())
                    #color_source *= truth_tire
                    cv2.imshow("Color", color_image)
                else:
                    color_source = depth_colormap
                    image = np.ones((h,w,3), dtype=int)*255
                    footprint *= (1/power)*image
                    cv2.imshow("Color", footprint)

            key = cv2.waitKey(1)
            if key == 8:
                cv2.destroyAllWindows()
                break
            if key != -1:
                settings.mode = 'None'
            if key == ord("f"):
                settings.mode = "Floor"
            if key == ord("t"):
                settings.mode = 'Tire'


            if key == ord('r'):
                SeperatePixs(color_image.copy())
                print(settings.tire_pos)

            if key == ord('e'):
                FindColorMath(color_image.copy())
            
            if key == ord("p"):
                if settings.state.paused:
                    cv2.setMouseCallback('Color', lambda *args : None)
                    cv2.destroyWindow('Data')
                else:
                    cv2.setMouseCallback('Color', GetColorInfo, color_source)   
                settings.state.paused ^= True

    
            if key == ord('w'):
                frameR += 1
                power += 1
            if key == ord('s'):
                if frameR> 1:
                    frameR += -1
                power += -1

            if key == ord('z'):
                if settings.zoom[2]:
                    cv2.setMouseCallback('Color', lambda *args : None)
                    cv2.destroyWindow('Zoom')
                    settings.zoom[2] = False
                else:
                    settings.zoom[2] = True
            if key == ord('x'):
                settings.zoom[1] = not settings.zoom[1]

            if settings.state.color:
                Zoom([color_image, verts])
            else:
                Zoom([footprint, verts])  

            if key in (27, ord("q")):
                return