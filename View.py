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



def onclick(event, x,y,flags, param):
    cv2.namedWindow("Contact Data", cv2.WINDOW_AUTOSIZE)
    img = cv2.imread('Images/blank.png')


    i = x//param[2]
    j = y//param[2]

    avg_depth = str(param[3][j][i])
    i, j, dist = FindClosestGap(x,y,param[2],param[3])

    depth = str(param[1][y*param[0].shape[1]+x][2])
    avg_depth_closest = str(param[3][j][i])
    cv2.putText(img, "Actually Depth, Average depth, and Closest Average Floor Depth", (10, 20), cv2.FONT_HERSHEY_SIMPLEX , .25, (0, 0, 0), 1, cv2.LINE_AA) 
    cv2.putText(img, depth, (10, 50), cv2.FONT_HERSHEY_SIMPLEX , .5, (0, 0, 0), 1, cv2.LINE_AA) 
    cv2.putText(img, avg_depth, (10, 80), cv2.FONT_HERSHEY_SIMPLEX , .5, (0, 0, 0), 1, cv2.LINE_AA) 
    cv2.putText(img, avg_depth_closest, (10, 110), cv2.FONT_HERSHEY_SIMPLEX , .5, (0, 0, 0), 1, cv2.LINE_AA) 
    
    cv2.putText(img, "Dist Of Closest", (10, 140), cv2.FONT_HERSHEY_SIMPLEX , .5, (0, 0, 0), 1, cv2.LINE_AA)
    cv2.putText(img, str(dist), (10, 170), cv2.FONT_HERSHEY_SIMPLEX , .5, (0, 0, 0), 1, cv2.LINE_AA)


    cv2.imshow('Contact Data', img) 


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

        """step = 5
        tire_gaps = FindGaps(all_verts[0], all_color_images[0], step)
        overlay = all_color_images[0].copy()
        for j in range(len(tire_gaps)):
            for i in range(len(tire_gaps[j])):
                if tire_gaps[j][i]:
                    grey = 255*(tire_gaps[j][i]*.5)
                    square_color = (grey,grey,grey)
                else:
                    square_color = (0,0,0)
                cv2.rectangle(overlay, (i*step, j*step), ((i+1)*step, (j+1)*step), square_color, -1)
        alpha = .8
        img = cv2.addWeighted(overlay, alpha, all_color_images[0], 1 - alpha, 0)
        cv2.imshow('Color',img)
        cv2.setMouseCallback('Color', onclick, [all_color_images[0], all_verts[0],step,tire_gaps])   
        """
        frame_numb = 0
        frameR = 5
        verts = []
        #display the stuff
        while True:

            if not settings.state.paused:
                time.sleep(1/frameR)
                frame_numb +=1
                if frame_numb >= len(all_verts):
                    frame_numb = 0
            

                depth_colormap = all_depth_colormaps[frame_numb]
                color_image = all_color_images[frame_numb]
                footprint = all_footprints[frame_numb]
            

                if settings.state.color:
                    color_source = color_image
                    cv2.imshow("Color", color_image)
                else:
                    color_source = depth_colormap
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
            if key == ord('s'):
                if frameR> 1:
                    frameR += -1
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