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



def init():
    while True:
        
        settings.init()
        outfile = GetFile()

        print(outfile)
        if not outfile:
            cv2.destroyAllWindows()
            return
        
        #selected temp files
        if outfile[-16:] == '/Temporary Files':
            outfile+='/'
            i = 1
            all_verts = []
            all_texcoords = []
            all_color_images = []
            all_depth_colormaps = []
            
            #Go through each temp file, combining them into one
            while os.path.isfile(outfile+str(i)+'.npz'):
                npzfile = np.load(outfile+str(i)+'.npz')
                all_verts.append(npzfile['now_verts'])
                all_texcoords.append(npzfile['now_texcoords'])
                all_color_images.append(npzfile['now_color_images'])
                all_depth_colormaps.append(npzfile['now_depth_colormaps'])
                i+=1

            path = outfile[:-16]

            Save(path, all_verts, all_texcoords, all_color_images, all_depth_colormaps)
            return

        #If not a temp file, load all the frames
        npzfile = np.load(outfile)
        all_verts = npzfile['all_verts']
        all_texcoords = npzfile['all_texcoords']
        all_color_images = npzfile['all_color_images']
        all_depth_colormaps = npzfile['all_depth_colormaps']
        all_footprints = npzfile['all_footprints']

        cv2.namedWindow(settings.state.WIN_NAME, cv2.WINDOW_AUTOSIZE)
        cv2.resizeWindow(settings.state.WIN_NAME, settings.w, settings.h)
        cv2.setMouseCallback(settings.state.WIN_NAME, mouse_cb)

        cv2.namedWindow("Color", cv2.WINDOW_AUTOSIZE)


        frame_numb = 0
        frameR = 5
        verts = []
        power = 1
        h,w = all_color_images[0].shape[:2]
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
            

                if settings.state.color:
                    color_source = color_image
                    cv2.imshow("Color", color_image)
                else:
                    color_source = depth_colormap
                    image = np.ones((h,w,3), dtype=int)*255
                    footprint *= (1/power)*image
                    cv2.imshow("Color", footprint)

                verts = all_verts[frame_numb]  # xyz
                
                texcoords = all_texcoords[frame_numb]
                
                
            # Render
            Render(settings.depth_intrinsics, verts, texcoords, color_source)
            key = cv2.waitKey(1)
            if key == 8:
                cv2.destroyAllWindows()
                npzfile.close()
                break
            
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


            if key == ord("c"):
                settings.state.color ^= True


            if key == ord("e"):
                MakeObj(verts)

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

            if key in (27, ord("q")) or cv2.getWindowProperty(settings.state.WIN_NAME, cv2.WND_PROP_AUTOSIZE) < 0:
                return
