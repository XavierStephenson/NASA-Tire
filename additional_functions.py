import cv2
import settings
import os
import numpy as np
import shutil
import math
import multiprocessing as mp
#from menu_functions import *
import menu_functions

#by Tire i mean not green
def Tire(pix):
    minH = 120
    maxH = 256
    minS = 220
    maxS = 300
    maxR = 150
    #if pix[0] < minH or pix[1] < minS: return True
    if pix[2] > maxR: return True
    return False

def DoMath(vertices, color, colormap,f):
    w = color.shape[1]
    step = 6
    tire_gaps = FindGaps(vertices, color, step)
    hsv_image = cv2.cvtColor(color, cv2.COLOR_BGR2HSV).copy()

    footprint = color.copy()
    tirecolor = color.copy()
    tiredepth = colormap.copy()
    floorcolor = color.copy()
    floordepth = colormap.copy()
    contact = color.copy()

    if f%(mp.cpu_count()) == 0: print('Group',f//mp.cpu_count(),":")
    for i in range(len(vertices)):
        x = i%w
        y = i//w
        footprint[y][x] = [255,255,255]
        if f%(mp.cpu_count()) == 0 and x==w-1 and y%60 == 0: print(f//mp.cpu_count(),int((i+(w*60))/len(vertices)*1000)/10,'%')
        
        if vertices[i][2] > 2:
            vertices[i] = False
            floorcolor[y][x] = [70,70,70]
            tirecolor[y][x] = [70,70,70]
            contact[y][x] = [70,70,70]

        elif vertices[i][2] == 0:
            floorcolor[y][x] = [0,0,0]
            tirecolor[y][x] = [0,0,0]
            contact[y][x] = [0,0,0]

        elif sum(color[y][x]) < 330:
            tirecolor[y][x] = [0,0,0]
            tiredepth[y][x] = [0,0,0]

        else:
            floorcolor[y][x] = [0,0,0]
            floordepth[y][x] = [0,0,0]

            #index = FindClosestGap(x,y,step,tire_gaps)
            index = False
            if index:
                tire_i, tire_j = index
                dist = abs(tire_gaps[tire_j][tire_i] - vertices[i][2])
                footprint[y][x] = [255*dist,255*dist,255*dist]
                
                if dist < .02:
                    contact[y][x] = [0,255,0]


            
        
    return [[contact, colormap],[tirecolor,tiredepth],[floorcolor,floordepth], footprint]

def FindGaps(vertices, color,step):
    h, w = color.shape[:2]
    tire_gaps = []

    for j in range(0,h,step):
        tire_gaps.append([])
        for i in range(0,w,step):
            tire_gaps[-1].append(True)
            #Square
            this_square_sum_z = 0
            for jj in range(j,j+step,1):
                for ii in range(i,i+step,1):
                    try:
                        if Tire(color[jj][ii]) or vertices[jj*w+ii][2] == 0 or vertices[jj*w+ii][2] > 2:
                            tire_gaps[-1][-1] = False
                        else:
                            this_square_sum_z += vertices[jj*w+ii][2]
                    except:
                        pass
            if tire_gaps[-1][-1]:
                tire_gaps[-1][-1] = this_square_sum_z/(step*step)

    return tire_gaps

def FindClosestGap(x,y, step, tire_gaps):
    tire_gaps_dict = {}
    for j in range(len(tire_gaps)):
        for i in range(len(tire_gaps[j])):
            if tire_gaps[j][i]:
                dist_from_point = math.sqrt(((i+.5)*step-x)**2 + ((j+.5)*step-y)**2)
                tire_gaps_dict[dist_from_point] = [i,j]
    sorted_dict = dict(sorted(tire_gaps_dict.items()))
    closest_dist = list(sorted_dict.items())[0][0]
    if closest_dist > 30:
        return False
    """i,j = list(sorted_dict.items())[0][1]
    dist = list(sorted_dict.items())[0][0]
    return i,j,dist"""
    return list(sorted_dict.items())[0][1]
   



def Save(path, all_verts, all_texcoords, all_color_images, all_depth_colormaps):
    
    def MakeFile(name):
        try:
            with open(path+name+'.npz', 'x') as f: pass
        except:
            os.remove(path+name+'.npz')
            with open(path+name+'.npz', 'x') as f: pass
    
    MakeFile('Raw')
    all_footprints = all_color_images.copy()
    np.savez(path+'Raw', all_verts=all_verts, all_texcoords=all_texcoords, all_color_images=all_color_images, all_depth_colormaps=all_depth_colormaps, all_footprints=all_footprints)

    """
    MakeFile('Clean')
    all_footprints = all_color_images.copy()
    np.savez(path+'Clean', all_verts=all_verts, all_texcoords=all_texcoords, all_color_images=all_color_images, all_depth_colormaps=all_depth_colormaps, all_footprints=all_footprints)
    """
    contact_color, contact_map= [[],[]]
    tire_color, tire_map = [[],[]]
    floor_color, floor_map = [[],[]]
    all_footprints = []
    all_verts = all_verts[:1]
    #Bundling input for multiproccessing
    inputs = []
    for f in range(len(all_verts)):
        inputs.append([all_verts[f].copy(),all_color_images[f].copy(),all_depth_colormaps[f].copy(),f])
    
    #Uses all cores on the computer
    print("Using All",mp.cpu_count(),"cores")
    print(len(all_verts),'frames',len(all_verts)//mp.cpu_count()+1,'Groups')
    pool = mp.Pool(mp.cpu_count())
    results = pool.starmap(DoMath, inputs)
    pool.close() 
    print("Finished!")

    #Unbundling result for use
    for f in range(len(all_verts)):
        contact,tire,floor, footprint = results[f]

        contact_color.append(contact[0]) 
        contact_map.append(contact[1]) 
        tire_color.append(tire[0]) 
        tire_map.append(tire[1]) 
        floor_color.append(floor[0]) 
        floor_map.append(floor[1])
        all_footprints.append(footprint) 
    

    MakeFile('Contact')
    all_color_images, all_depth_colormaps = contact_color, contact_map
    np.savez(path+'Contact', all_verts=all_verts, all_texcoords=all_texcoords, all_color_images=all_color_images, all_depth_colormaps=all_depth_colormaps, all_footprints=all_footprints )
    
    MakeFile('Tire')
    all_color_images, all_depth_colormaps = tire_color, tire_map
    np.savez(path+'Tire', all_verts=all_verts, all_texcoords=all_texcoords, all_color_images=all_color_images, all_depth_colormaps=all_depth_colormaps, all_footprints=all_footprints )
    
    MakeFile('Floor')
    all_color_images, all_depth_colormaps = floor_color, floor_map
    np.savez(path+'Floor', all_verts=all_verts, all_texcoords=all_texcoords, all_color_images=all_color_images, all_depth_colormaps=all_depth_colormaps, all_footprints=all_footprints )
    try:
        shutil.rmtree(path+'Temporary Files')
    except:
        pass

def Zoom(images):
    if settings.zoom[2]:
        height, width, c = images[0].shape

        def PosToCrop():
            yshift, xshift = [height/2, width/2]
            yclick = settings.zoom[3][0]*settings.zoom[0]
            xclick = settings.zoom[3][1]*settings.zoom[0]
            y1,y2,x1,x2  = [yclick-yshift, yclick+yshift,xclick-xshift, xclick+xshift]

            if y1 < 0:
                y1,y2 = [0,height]
            if y2 > height*settings.zoom[0]:
                y1, y2 = [height*settings.zoom[0]-height, height*settings.zoom[0]]
            
            if x1 < 0:
                x1,x2 = [0,width]
            if x2 > width*settings.zoom[0]:
                x1, x2 = [width*settings.zoom[0]-width, width*settings.zoom[0]]
            
            return [y1,y2,x1,x2]
    
        def ZoomView(event, x, y, p1, p2):
            if event == 10:
                if not settings.zoom[1]:
                    settings.zoom[0] *= .9
                else:
                    settings.zoom[0] *= 1.1
            if settings.zoom[0] < 1: settings.zoom[0] = 1

            settings.zoom[3] = [y,x]

            img = CropAndCursor(images[0].copy(), x, y)

            cv2.imshow('Zoom', img)
            GetMouseInfo(0,x,y,0,images)

        def CropAndCursor(image, x, y):
            
            overlay = image.copy()
            cv2.rectangle(overlay, (x, y-5), (x+1, y+5), (0,0,255), -1)
            cv2.rectangle(overlay, (x-5, y), (x+5, y+1), (0,0,255), -1)
            alpha = 1*(1/settings.zoom[0])  
            img = cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0)


            img = cv2.resize(img, None, fx=settings.zoom[0], fy=settings.zoom[0])
            y1, y2, x1, x2 = PosToCrop()
            img = img[int(y1):int(y2), int(x1):int(x2)]
            return img

        cv2.setMouseCallback('Color', ZoomView)
        img = CropAndCursor(images[0].copy(), settings.zoom[3][1], settings.zoom[3][0])
        cv2.imshow('Zoom', img)

def GetMouseInfo(event, x, y, flag, param):
   
    #images 0 is color source
    #images 1 is vert
    images = param
    
    cv2.namedWindow("Data", cv2.WINDOW_AUTOSIZE)
    img = cv2.imread('Images/blank.png')
    hsv_image = cv2.cvtColor(images[0], cv2.COLOR_BGR2HSV)

    #display color info
    color = images[0][y][x]
    color_text = str(hsv_image[y][x])+str(color)

    [c1, c2, c3] = images[0][y][x]
    cv2.putText(img, "HSV            BRG", (10, 20), cv2.FONT_HERSHEY_SIMPLEX , .5, (0, 0, 0), 1, cv2.LINE_AA) 
    cv2.putText(img, color_text, (10, 50), cv2.FONT_HERSHEY_SIMPLEX , .5, (0, 0, 0), 1, cv2.LINE_AA) 
    cv2.rectangle(img, (200, 5), (250, 25), (int(c1), int(c2), int(c3)), -1)


    coords_text = str([int(item*10000)/10000 for item in images[1][y*images[0].shape[1] + x]])
    cv2.putText(img, "Coords", (10, 75), cv2.FONT_HERSHEY_SIMPLEX , .5, (0, 0, 0), 1, cv2.LINE_AA) 
    cv2.putText(img, coords_text, (10, 100), cv2.FONT_HERSHEY_SIMPLEX , .5, (0, 0, 0), 1, cv2.LINE_AA) 

    cv2.putText(img, "It Is Tire:", (10, 125), cv2.FONT_HERSHEY_SIMPLEX , .5, (0, 0, 0), 1, cv2.LINE_AA) 
    cv2.putText(img, str(Tire(color)), (10, 150), cv2.FONT_HERSHEY_SIMPLEX , .5, (0, 0, 0), 1, cv2.LINE_AA) 

    cv2.putText(img, str(x)+','+str(y), (10, 175), cv2.FONT_HERSHEY_SIMPLEX , .5, (0, 0, 0), 1, cv2.LINE_AA) 
    cv2.imshow('Data', img) 

def MakeFolder():
    first = True
    while True:

        #if folder doesnt exist make it and return the path
        window_name = 'Folder'
        if first:
            msg1 = "Please Type Destination Folder"
            msg2 = "Hit Enter When Finished"
        else:
            msg1 = "That Folder Already Exist"
            msg2 = "Press Any Key To Try Again"
        folder_name = menu_functions.TypeMenu(window_name,msg1,msg2)
        first = False
        print(folder_name)
        try:
            os.makedirs('Outputs/'+folder_name)
            path = 'Outputs/'+folder_name+'/'
            cv2.destroyWindow(window_name)
            return path
        #if folder exists inform user and try again
        except:
            pass
            
def AskNew():
    window_name = "What Type of Data to Anylalize"
    options = ['New Data','Old Data']
    return menu_functions.ClickMenu(window_name, window_name, options, True)

def GetFile():
    #get list of folders in outputs, put most recent at top
    allfolders = [f for f in os.listdir('Outputs') if not os.path.isfile(os.path.join('Outputs', f))]
    folder_dict = {}
    for i in allfolders:
        folder_dict[i] = os.path.getmtime(os.path.join('Outputs', i))
    res = {key: val for key, val in sorted(folder_dict.items(), reverse=True, key = lambda ele: ele[1])}
    allfolders = list(res.keys())

    while True:
        #Get what folder user clicked on, if they backspaced at folder level return false
        i = menu_functions.ClickMenu('Select Folder', 'Select Folder', allfolders, False)
        if i == 0:
            return False
        else:
            folder = allfolders[i-1]  

        
        #Get what file they clicked on, if back space cont in loop, else get file and return path
        allfiles = os.listdir('Outputs/'+folder)
        i = menu_functions.ClickMenu(folder, folder+': Pleae Select File', allfiles, False)
        if i !=0 :
            file = allfiles[i-1]
            path = 'Outputs/'+folder+'/'+file
            return path

def GetFolder():
    #get list of folders in outputs, put most recent at top
    allfolders = [f for f in os.listdir('Outputs') if not os.path.isfile(os.path.join('Outputs', f))]
    folder_dict = {}
    for i in allfolders:
        folder_dict[i] = os.path.getmtime(os.path.join('Outputs', i))
    res = {key: val for key, val in sorted(folder_dict.items(), reverse=True, key = lambda ele: ele[1])}
    allfolders = list(res.keys())


    #Get what folder user clicked on, if they backspaced at folder level return false
    i = menu_functions.ClickMenu('Select Folder', 'Select Folder', allfolders, False)
    if i == 0:
        return False
    else:
        folder = allfolders[i-1]  
        path = 'Outputs/'+folder+'/'
        return path
                        
def OldSave(path):
    outfile = path+'Raw.npz'
    npzfile = np.load(outfile)
    all_verts = npzfile['all_verts']
    all_texcoords = npzfile['all_texcoords']
    all_color_images = npzfile['all_color_images']
    all_depth_colormaps = npzfile['all_depth_colormaps']
    npzfile.close()

    shutil.rmtree(path)
    os.makedirs(path)
    Save(path, all_verts, all_texcoords, all_color_images, all_depth_colormaps)

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

#Make a 3d moldel of thing with each pix being its approx size
def MakeObj(verts):
    while True:
     
        window_name = 'File Name'
        file_name = menu_functions.TypeMenu(window_name,"Please Type Destination File","Press Enter When Done")
        onlyfiles = [f for f in os.listdir('Outputs') if os.path.isfile(os.path.join('Outputs'+file_name, f))]
        print(onlyfiles)
        if file_name not in onlyfiles:
            cv2.destroyWindow(window_name)
            break
        else:
            img = cv2.imread('Images/blank.png')
            cv2.putText(img, "Filde Already Exists", (10, 20), cv2.FONT_HERSHEY_SIMPLEX , .5, (0, 0, 0), 1, cv2.LINE_AA) 
            cv2.putText(img, "Press Any Key To Try Again", (10, 40), cv2.FONT_HERSHEY_SIMPLEX , .5, (0, 0, 0), 1, cv2.LINE_AA) 
            cv2.imshow(window_name, img)
    points = []
    faces = []
    
    for i in range(len(verts)):
        points.append( verts[i] )
        if verts[i][2] == 0:
            continue
        try:
            if verts[i+1][2] != 0 and verts[i+settings.w][2] != 0 and verts[i+settings.w+1][2] != 0:
                face1 = [ i ]
                face1.append( i + settings.w  + 1 )
                
                face2 = face1.copy()
                face1.append( i + settings.w )

                face2.append( i + 1 )
            
                faces.append(face1)
                faces.append(face2)
        except:
            pass
                
    NewFile(file_name,points,faces)
