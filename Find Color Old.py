#####################################################
##               Read bag from file                ##
#####################################################


# First import library
import pyrealsense2 as rs
# Import Numpy for easy array manipulation
import numpy as np
# Import OpenCV for easy image rendering
import cv2
import math
import time
import os


degree_Fovx = 70
degree_Fovy = 80
input = 'Inputs/cap1.bag'
settings = {'Min Hue': 0,'Max Hue': 0, 'Min Saturation':0,'Max Saturation': 0, 'Mask Type':'none', 'Grid Size':0 ,'# of Skipped Frames':0,"Frame Rate":10, 'FileName':input[:-4]}



# Create pipeline
pipeline = rs.pipeline()

# Create a config object
config = rs.config()

# Tell config that we will use a recorded device from file to be used by the pipeline through playback.
rs.config.enable_device_from_file(config, input)

# Configure the pipeline to stream the depth stream
# Change this parameters according to the recorded bag file resolution
config.enable_stream(rs.stream.depth, rs.format.z16, 30)
config.enable_stream(rs.stream.color)

# Start streaming from file
pipeline.start(config)

# Create colorizer object
colorizer = rs.colorizer()

FOVx = degree_Fovx*math.pi/360
FOVy = degree_Fovy*math.pi/360
frame_numbs = [0]

def Export():
    frame_rate = settings['Frame Rate']
    file_name = settings['FileName']

    file_name += '.avi'
    #makes sure it doesnt name it same thing
    while os.path.isfile(file_name):
        index_open = file_name.find('(')
        index_close = file_name.find(')')
        
        if index_open != -1:
           new_int = str( int(file_name[index_open+1:index_close])+1 )
           file_name = file_name[:index_open+1] + new_int + file_name[index_close:]
        else:
            file_name = file_name[:file_name.find('.')] + ' (1)' + '.avi'
        

    # Define the codec and create VideoWriter object.The output is stored in 'outpy.avi' file.
    out = cv2.VideoWriter(file_name,cv2.VideoWriter_fourcc('M','J','P','G'), frame_rate, (len(color_images[0][0]),len(color_images[0])))
    
    for i in range(len(color_images_aft[0])):
         # Write the frame into the file 'output.avi'
        out.write(color_images_aft[0][i])
    
    out.release()
def NewFile(file_name, points, faces):
    file_name += '.obj'
    txt = '#Blank Comment\n\n'
    for point in points:
        txt += 'v'
        for dim in point:
            txt += ' '+str(dim)
        txt += '\n'

    for face in faces:
        txt += 'f'
        for point_index in face:
            txt += ' '+str(point_index+1)
        txt += '\n'
    while True:
        try:
            with open(file_name, 'x') as f:
                f.write(txt)
                break
        except:
                print('stuck in here')
                index_open = file_name.find('(')
                index_close = file_name.find(')')
                
                if index_open != -1:
                    new_int = str( int(file_name[index_open+1:index_close])+1 )
                    file_name = file_name[:index_open+1] + new_int + file_name[index_close:]
                else:
                    file_name = file_name[:file_name.find('.')] + ' (1)' + '.obj'

def ObjExport(file_name, frame_number):
    points = []
    faces = []
    depth_image = depth_images_skip[0][frame_number]
    
    index = -1
    width = len(depth_image[0])
    for row in range(len(depth_image)):
            for col in range(len(depth_image[row])):
                depth = depth_image[row][col]
                points.append( [item * depth for item in consts[row][col]] )
                index += 1

                if depth == 0:
                    continue
                
                try:
                    if depth_image[row+1][col] != 0 and depth_image[row][col+1] != 0 and depth_image[row+1][col+1] != 0:
                        face1 = [ index ]
                        face1.append( index + width  + 1 )
                        
                        face2 = face1.copy()
                        face1.append( index + width )

                        face2.append( index + 1 )
                    
                        faces.append(face1)
                        faces.append(face2)
                except:
                    pass
                
    NewFile(file_name,points,faces)
def ObjVid():
    path = settings['FileName'] + '/' + settings['FileName']
    os.makedirs(settings['FileName'])
    for frame_number in range(len(depth_images_skip[0])):
        print('exporting frame'+str(frame_number)+'/'+str(len(depth_images_skip[0])))
        ObjExport(path+str(frame_number), frame_number)


def ChangeSettings(numb, key):
    img = cv2.imread('Images/settings_blank.png')
    #space to move forwards
    if key == 32:
        numb = (numb + 1) % len(list(settings))
    else:
        setting_name = list(settings)[numb]

        if type(settings[setting_name]) is int:
            try:
                if key == 8:
                    settings[setting_name] = int ( str(settings[setting_name])[:-1] )
                else:
                    settings[setting_name] = int ( str(settings[setting_name]) + chr(key) )
            except:
                if key == 8:
                    settings[setting_name] = 0

        else:
            if key == 8:
                settings[setting_name] = settings[setting_name][:-1]
            else:
                settings[setting_name] += chr(key)
    cv2.rectangle(img, (10, (50*numb)+20), (400, (50*numb)+40), (225,225,225), -1)
    for i in range(len(list(settings))):
        setting_name = list(settings)[i]
        setting_val = settings[setting_name]
        cv2.putText(img, setting_name, (10, (50*i)+20), cv2.FONT_HERSHEY_SIMPLEX , .5, (0, 0, 0), 1, cv2.LINE_AA) 
        cv2.putText(img, str(setting_val), (10, (50*i)+40), cv2.FONT_HERSHEY_SIMPLEX , .5, (0, 0, 0), 1, cv2.LINE_AA) 
    cv2.imshow('Settings', img)
    return numb

data_display = [True]

def Pause():
    setting_numb = 0
    data_display[0] = True
    cv2.setMouseCallback('Original', on_click)
    cv2.setMouseCallback('Depth', on_click)
    cv2.setMouseCallback('Mask', on_click)
    on_click(0,0,0,0,0)
    
    while True:

        key = cv2.waitKey(1)
        if key == -1:
            continue
        #if pressed escape exit program
        if key == 27:
            return True
        if key == 13:
            data_display[0] = not data_display[0]
            if data_display[0]:
                cv2.destroyWindow('Settings')
        if data_display[0]:
            match key:
                #if pressed space unpause
                case 32:
                    cv2.destroyWindow('Data')
                    cv2.setMouseCallback('Original', lambda *args : None)
                    cv2.setMouseCallback('Depth', lambda *args : None)
                    cv2.setMouseCallback('Mask', lambda *args : None)
                    return False
                #a or d to move frame by fram
                case 97:
                    frame_numbs[0] = (frame_numbs[0]-1) % len(depth_images_skip[0])
                    cv2.imshow("Original", color_images_skip[0][frame_numbs[0]])
                    cv2.imshow("Mask", color_images_aft[0][frame_numbs[0]])
                    cv2.imshow("Depth", depth_color_images_aft[0][frame_numbs[0]])
                case 100:
                    frame_numbs[0] = (frame_numbs[0]+1) % len(depth_images_skip[0])
                    cv2.imshow("Original", color_images_skip[0][frame_numbs[0]])
                    cv2.imshow("Mask", color_images_aft[0][frame_numbs[0]])
                    cv2.imshow("Depth", depth_color_images_aft[0][frame_numbs[0]])
                #e to export
                case 101:
                    Export()
                #r to redo math
                case 114:
                    cv2.setMouseCallback('Original', lambda *args : None)
                    cv2.setMouseCallback('Depth', lambda *args : None)
                    cv2.setMouseCallback('Mask', lambda *args : None)
                    cv2.destroyAllWindows()
                    depth_color_images_aft[0] = []
                    color_images_aft[0] = []
                    color_images_skip[0] = []
                    depth_images_skip[0] = []
                    print('see',depth_color_images_aft[0])
                    Do_All_Math()
                    return False
        else:
            setting_numb = ChangeSettings(setting_numb, key)

def on_click(event, x, y, p1, p2):
    #make window to display color and depth info for selected pixel
    cv2.namedWindow("Data", cv2.WINDOW_AUTOSIZE)
    img = cv2.imread('Images/blank.png')
    hsv_image = cv2.cvtColor(color_images_skip[0][frame_numbs[0]], cv2.COLOR_BGR2HSV)
 
    #display color info
    color = color_images_skip[0][frame_numbs[0]][y][x]
    color_text = str(hsv_image[y][x])+str(color)

    [c1, c2, c3] = color_images_skip[0][frame_numbs[0]][y][x]
    cv2.putText(img, "HSV            BRG", (10, 20), cv2.FONT_HERSHEY_SIMPLEX , .5, (0, 0, 0), 1, cv2.LINE_AA) 
    cv2.putText(img, color_text, (10, 50), cv2.FONT_HERSHEY_SIMPLEX , .5, (0, 0, 0), 1, cv2.LINE_AA) 
    cv2.rectangle(img, (200, 5), (250, 25), (int(c1), int(c2), int(c3)), -1)

    #display depth info
    depth = depth_images_skip[0][frame_numbs[0]][y][x]
    if(depth==0):
        depth_text='null'
    else:
        depth_text = str(depth)+"mm"
    
    cv2.putText(img, "Depth", (10, 75), cv2.FONT_HERSHEY_SIMPLEX , .5, (0, 0, 0), 1, cv2.LINE_AA) 
    cv2.putText(img, depth_text, (10, 100), cv2.FONT_HERSHEY_SIMPLEX , .5, (0, 0, 0), 1, cv2.LINE_AA) 

    #display coordinate
    coord_text = str([int(item * depth) for item in consts[y][x]])
    cv2.putText(img, "Coords", (10, 125), cv2.FONT_HERSHEY_SIMPLEX , .5, (0, 0, 0), 1, cv2.LINE_AA) 
    cv2.putText(img, coord_text, (10, 150), cv2.FONT_HERSHEY_SIMPLEX , .5, (0, 0, 0), 1, cv2.LINE_AA) 
    
    # Displaying the image 
    cv2.imshow('Data', img) 

def Neigh(arr, row, col, size):
    
    s_row = row - math.floor(size/2)
    s_col = col - math.floor(size/2)
    for i in range(size):
        for j in range(size):
            try:
                arr[i+s_row][j+s_col] += 1
            except:
                pass

def Mask(depth_color_image, color_image, depth_image):
    hsv_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2HSV)
    color_image_aft = color_image.copy()
    minH = settings['Min Hue']
    maxH = settings['Max Hue']
    minS = settings['Min Saturation']
    maxS = settings['Max Saturation']
    type = settings['Mask Type']
    grid_size = settings['Grid Size']

    #Masks pix if it is in range
    if type == 'single':
        for row in range(len(color_image)):
            for col in range(len(color_image[row])):
                pix = hsv_image[row][col]
                if pix[0] > minH and pix[0] < maxH:
                    if pix[1] > minS and pix[1] < maxS:
                            depth_color_image[row][col] = [255,255,255]
                            color_image_aft[row][col] = [0,0,0]

    #masks pix if every pix in a certain distance is in range
    if type == "grid":
        delete_arr = []
        for row in range(len(color_image)):
            delete_arr.append([])
            for col in range(len(color_image[row])):
                delete_arr[row].append(0)

        for row in range(len(color_image)):
            for col in range(len(color_image[row])):
                pix = hsv_image[row][col]
                if pix[0] > minH and pix[0] < maxH:
                    if pix[1] > minS and pix[1] < maxS:
                        Neigh(delete_arr, row, col, grid_size[0])

        for row in range(len(color_image)):
            for col in range(len(color_image[row])):
                if delete_arr[row][col] >= grid_size**2:
                    depth_color_image[row][col] = [0,0,0]
                    color_image_aft[row][col] = [0,0,0]
        
    if type == "overlay":
        #color_image_aft = cv2.addWeighted(color_image,0.7,depth_color_image,0.3,0)
        for row in range(len(color_image)):
           for col in range(len(color_image[row])):
               for i in range(3):
                   color_image_aft[row][col][i] = (int(color_image[row][col][i])*2+int(depth_color_image[row][col][i]))/3
    
    return [depth_color_image, color_image_aft]

consts = []
def Coords(shape):
    center = [shape[1]/2, shape[0]/2]
    for piy in range(shape[0]):
        consts.append([])
        for pix in range(shape[1]):
            tan_theta = math.tan( ( ((pix-center[0])/(center[0]))*FOVx ) + math.pi/2 )
            tan_beta  = math.tan( ( ((piy-center[1])/(center[1]))*FOVy ) + math.pi/2 )
            
            tan_theta_sqr = tan_theta ** 2
            tan_beta_sqr  = tan_beta ** 2
            
            x = tan_beta/math.sqrt((tan_beta_sqr+tan_theta_sqr+(tan_theta_sqr*tan_beta_sqr))) * math.copysign(1,tan_beta*tan_theta)
            y = x*tan_theta/tan_beta
            z = x*tan_theta

            '''         
            theta = ((pix-center[0])/(center[0]))*FOVx + math.pi/2
            beta  = ((pix-center[1])/(center[1]))*FOVx + math.pi/2
            x = math.sin(theta) * math.cos(beta)
            y = math.cos(theta)
            z = math.sin(theta) * math.sin(beta)
            '''
            consts[piy].append([-x, y, z])
            



depth_color_images = []
color_images = []
depth_images = []
def Get_All_Frames():
    first = True
    last_time = 0
    while True:
        # Get frameset
        frames = pipeline.wait_for_frames()
        

        # Get frames
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()
        time_color = rs.frame.get_timestamp(color_frame)

        # Colorize depth frame to jet colormap
        depth_color_frame = colorizer.colorize(depth_frame)

        # Convert depth_frame to numpy array to render image in opencv
        depth_color_image = np.asanyarray(depth_color_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())
        depth_image = np.asanyarray(depth_frame.get_data())

        if first:
            #gets all constants for coordinate maping
            Coords(depth_image.shape)
            first = False
        
        if last_time >= time_color:
            break
        last_time = time_color
        #last_time += 1
        #if last_time >100:
        #   break

        depth_color_image = depth_color_image.copy()
        color_image = color_image.copy()
        depth_image = depth_image.copy()

        depth_color_images.append(depth_color_image)
        color_images.append(color_image)
        depth_images.append(depth_image)
Get_All_Frames()
print('got all frames', len(color_images))


depth_color_images_aft = [[]]
color_images_aft = [[]]
color_images_skip = [[]]
depth_images_skip = [[]]
def Do_All_Math():
    time.sleep(2)
    print('see',depth_color_images_aft[0])
    skip = settings['# of Skipped Frames']
    #cv2.namedWindow("Why", cv2.WINDOW_AUTOSIZE)
   
  
    for i in range(0,len(color_images),skip+1):

        # Convert depth_frame to numpy array to render image in opencv
        depth_color_image = depth_color_images[i].copy()
        color_image = color_images[i].copy()
        depth_image = depth_images[i].copy()
        
        [depth_color_image_aft, color_image_aft] = Mask(depth_color_image, color_image, depth_image)

        imgage = cv2.imread('Images/blank.png')
        text = str(i)+'/'+str(len(color_images))
        cv2.putText(imgage, text, (10, 20), cv2.FONT_HERSHEY_SIMPLEX , 1, (0, 0, 0), 1, cv2.LINE_AA) 
        cv2.imshow("Loading", imgage)
        
        print(i)
        


        depth_color_images_aft[0].append(depth_color_image_aft)
        color_images_aft[0].append(color_image_aft)
        color_images_skip[0].append(color_image)
        depth_images_skip[0].append(depth_image)
    cv2.destroyAllWindows()
    frame_numbs[0] = 0
Do_All_Math()
print('Did all math')


def Show_Results():
    while True:
        frame_numbs[0] += 1
        if frame_numbs[0] >= len(depth_color_images_aft[0]):
            frame_numbs[0] = 0
        # Render image in opencv window
        # print (len(color_images_skip[0]))
        cv2.imshow("Original", color_images_skip[0][frame_numbs[0]])
        cv2.imshow("Mask", color_images_aft[0][frame_numbs[0]])
        cv2.imshow("Depth", depth_color_images_aft[0][frame_numbs[0]])

        time.sleep(1/settings['Frame Rate'])
        
        key = cv2.waitKey(1)
        if key != -1:
            print(len(depth_images_skip),len(color_images_skip))
        match key:
            case -1:
                continue
            # if pressed space pause
            case 32:
                if Pause():
                    cv2.destroyAllWindows()
                    break
            # if pressed escape exit program
            case 27:
                cv2.destroyAllWindows()
                break
            case 119:
                settings['Frame Rate'] += 1
            case 115:
                if settings['Frame Rate'] > 1:
                    settings['Frame Rate'] += -1
            case 101:
                #ObjVid()
                ObjExport(settings['FileName'], frame_numbs[0])

Show_Results()