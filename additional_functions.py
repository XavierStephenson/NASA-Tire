import cv2
import settings
import os
import numpy as np
import shutil
import math
import multiprocessing as mp
import timeit
from stl import mesh
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



def ArraysOld(verts, color):
    h, w = color.shape[:2]

    tire_arr = np.zeros((h, w, 3), dtype=bool)
    floor_arr = tire_arr.copy()
    hsv_image = cv2.cvtColor(color, cv2.COLOR_BGR2HSV)
    for i in range(len(verts)):
        x = i%w
        y = i//w
        if verts[i][2] < 2 and verts[i][2] != 0:
            if hsv_image[y][x][0] > 100:
                tire_arr[y][x] = [True,True,True]
            else:
                floor_arr[y][x] = [True,True,True]
    return [tire_arr, floor_arr]

def Arrays(verts, color):
    
    h,w = color.shape[:2]
    #color = cv2.cvtColor(color, cv2.COLOR_BGR2HSV)
    color = color.reshape((len(verts),3))
    truth_tire = np.ones_like(verts)
    truth_floor = np.ones_like(verts)
    #Falsing the 0's
    truth = np.sign(np.sign(sum(np.transpose(verts != [0,0,0]))-1)+1).reshape(len(verts),1)
    truth_tire  *= truth
    truth_floor *= truth

    #Falsing the z>2's
    truth = np.sign(np.sign(sum(np.transpose(verts <= [-np.inf,-np.inf,2]))-1)+1).reshape(len(verts),1)
    truth_tire *= truth
    truth_floor *= truth

    #Falsing not the not Tire color
    truth1 = color > [0,0,0]
    truth2 = color < [100,100,100]
    truth = sum(np.transpose(truth1*truth2)) == 3
    truth = np.array(truth, dtype=int).reshape(len(truth),1)
    truth_tire *= truth

    #inverting that for floor
    truth_floor *= np.logical_not(truth)

    #putting back to correct shape
    truth_tire = np.array(truth_tire.reshape(h,w,3), dtype=bool)
    truth_floor = np.array(truth_floor.reshape(h,w,3), dtype=bool)

    return [truth_tire, truth_floor]


#if every el in compare_arr is greater than its equivalent in an in_arr then the output is [0 0 0] in the in_arr spot
#array with shape (huge, 3)
def CompareArr(in_arr,compare_arr):
    truth = np.sign(np.sign(sum(np.transpose(in_arr >= compare_arr))-1)+1).reshape((len(in_arr)),1)
    return np.ones_like(in_arr,dtype=bool)*truth

def FindGaps(verts, floor_arr, shape, step, color):
    h, w = shape
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
                        if not floor_arr[jj][ii][0]:
                            tire_gaps[-1][-1] = False
                            break
                        else:
                            this_square_sum_z += verts[jj*w+ii][2]
                    except:
                        pass
            if tire_gaps[-1][-1]:
                tire_gaps[-1][-1] = this_square_sum_z/((step)**2)
              
    return tire_gaps

def DoMath(verts, color, colormap, f):
    step = 5
    h, w = color.shape[:2]

    truth_tire, truth_floor = Arrays(verts, color)
    tire_gaps = FindGaps(verts, truth_floor, color.shape[:2], step, color.copy())

    gap_centers = []
    for j in range(len(tire_gaps)):
        for i in range(len(tire_gaps[j])):
            if tire_gaps[j][i]:
                gap_centers.append([(i+.5)*step,(j+.5)*step])
    gap_centers = np.array(gap_centers)
    if f%(settings.cores) == 0: print('start reshaping')
    vert_shaped = verts.reshape(h,w,3)

    tire_color = truth_tire*color
    tire_map = truth_tire*colormap
    tire_verts = (truth_tire*vert_shaped).reshape(len(verts),3)

    floor_color = truth_floor*color
    floor_map = truth_floor*colormap
    floor_verts = (truth_floor*vert_shaped).reshape(len(verts),3)

    contact_color = color.copy()
    if f%(settings.cores) == 0: print('done reshaping')
    diff_arr = np.ones((h, w, 3), dtype=float)
    
    for index in range(len(verts)):
       
        x = index%w
        y = index//w

        if f%(settings.cores) == 0 and x==w-1 and y%10 == 0: 
            text = str(f//settings.cores)+': '+str(int((index+(w*10))/len(verts)*1000)/10)+'%'
            print(text)
        if truth_tire[y][x][0]:
            point = np.array((x,y))
            distances = np.linalg.norm(gap_centers-point, axis=1)
            min_index = np.argmin(distances)
            i,j = gap_centers[min_index]/step
            i = int(i)
            j = int(j)
            dist = distances[min_index]

            if dist < 30 and tire_gaps[j][i]:
                diff = abs(verts[index][2]-tire_gaps[j][i])
                diff_arr[y][x] *= diff
                if diff < .01:
                    contact_color[y][x] = [0,255,0]
    

    return [[verts, contact_color, colormap],[tire_verts, tire_color,tire_map],[floor_verts, floor_color,floor_map], diff_arr]  


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
    contact, tire, floor, all_footprints = [[],[],[],[]]


    #Bundling input for multiproccessing
    inputs = []
    for f in range(len(all_verts)):
        inputs.append([all_verts[f].copy(),all_color_images[f].copy(),all_depth_colormaps[f].copy(),f])
    
    #Uses all cores on the computer
    text = "Using All "+str(settings.cores)+" cores"

    print(text)
    text = str(len(all_verts))+' frames '+str(len(all_verts)//settings.cores+1)+' Groups'
    print(text)
  

    pool = mp.Pool(settings.cores)
    results = pool.starmap(DoMath, inputs)
    pool.close() 
    print("Finished!")

    #Unbundling result for use
    for i in range(3):
        contact.append([])
        tire.append([])
        floor.append([])

    for f in range(len(all_verts)):
        contact_f, tire_f, floor_f, footprint_f = results[f]
        all_footprints.append(footprint_f)

        for i in range(len(contact_f)):
            contact[i].append(contact_f[i])
            tire[i].append(tire_f[i])
            floor[i].append(floor_f[i])
            
    

    MakeFile('Contact')
    all_verts, all_color_images, all_depth_colormaps = contact
    np.savez(path+'Contact', all_verts=all_verts, all_texcoords=all_texcoords, all_color_images=all_color_images, all_depth_colormaps=all_depth_colormaps, all_footprints=all_footprints )
    
    MakeFile('Tire')
    all_verts, all_color_images, all_depth_colormaps = tire
    np.savez(path+'Tire', all_verts=all_verts, all_texcoords=all_texcoords, all_color_images=all_color_images, all_depth_colormaps=all_depth_colormaps, all_footprints=all_footprints )
    
    MakeFile('Floor')
    all_verts, all_color_images, all_depth_colormaps = floor
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

def SeperatePixs(color_image):
    color_image = color_image.reshape(color_image.size//3, 3)
    truth1 = color_image > [0,0,0]
    truth2 = color_image < [110,110,110]
    truth = sum(np.transpose(truth1*truth2)) == 3
    settings.tire_pos = truth

    print(settings.tire_pos)
    print(sum(settings.tire_pos), len(settings.tire_pos))

def GetColorInfo(event, x, y, flag, color_image):
    #key = cv2.waitKey(1)
    
    cv2.namedWindow("Data", cv2.WINDOW_AUTOSIZE)
    img = cv2.imread('Images/blank.png')
    hsv_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2HSV)

    #display color info
    color = np.array(color_image[y][x],dtype=int)
    if settings.mode == 'Floor':
        print(settings.mode,color)
        settings.floor_colors.append(color)
    if settings.mode == 'Tire':
        print(settings.mode,color)
        settings.tire_colors.append(color)
    cv2.putText(img, settings.mode+str(color), (10, 75), cv2.FONT_HERSHEY_SIMPLEX , .5, (0, 0, 0), 1, cv2.LINE_AA) 
    color_text = str(hsv_image[y][x])+str(color)

    cv2.imshow('Data', img) 
                      
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

def NewFile(path, points, faces):

    text1 = '\nv '.join(' '.join('%0.3f' %x for x in y) for y in points)
    text1 = 'v '+ text1

    faces += 1
    text2 = '\nf '.join(' '.join('%i' %x for x in y) for y in faces)
    text2 = 'f ' +text2
    txt = text1 +'\n'+ text2

    with open(path, 'x') as f:
        f.write(txt)

def FindColorMath(color_image):
    color_image  = color_image.reshape(color_image.size//3, 3)
    tire_colors  = color_image[settings.tire_pos]
    floor_colors = color_image[np.logical_not(settings.tire_pos)]

    tire_colors = np.transpose(tire_colors)
    lil = np.amin(tire_colors,1)+1
    big = np.amax(tire_colors,1)-1
    print('Above',lil,'Below',big)
    
    truth1 = floor_colors > lil
    truth2 = floor_colors < big
    truth = sum(np.transpose(truth1*truth2)) == 3
    if np.any(truth):
        floor_as_tires = np.count_nonzero(truth)
        total_floor = len(truth)
        print('Oh NO',floor_as_tires,'/',total_floor,floor_as_tires/total_floor,'floor pixels count as tires')
    else:
        settings.color_compare = [lil, big]

def MakeObjOld(all_verts, shape, path, file_name):
    #for i in range(len(all_verts)):
        i=0
        startTime = timeit.default_timer()
        print('Exporting Frame',str(i))
        verts = all_verts[i]

        h, w = shape
        #Assuming the pix are square
        #when allowing to be rectangel the width was too large
        fovy = settings.depth_intrinsics.ppy*90/math.pi

        consty = 2*math.tan(fovy)/h
        points = []
        faces = []

        j = 0
        for vert in verts:
            if vert[2] == 0 or vert[2] > 2:
                continue
            py = consty*vert[2]
             
            p2 = vert.copy()
            p3 = vert.copy()
            p4 = vert.copy()

            p2[0] += py
            p3[1] += py
            p4[0] += py
            p4[1] += py

    
            points.extend([vert.copy(),p2,p3,p4])

            faces.append([j, j+1, j+2])
            faces.append([j+1, j+2, j+3])
            j+=4
        MathTime = timeit.default_timer()
        obj_path = path+file_name+str(i+1)+'.obj'
        NewFile(obj_path,points,np.array(faces))
        FileTime = timeit.default_timer()
        print('Math', MathTime-startTime, FileTime-startTime)


def MakeObj(verts, shape, path, file_name, frame):
   
    startTime = timeit.default_timer()
    if frame%settings.cores == 0: print('Exporting Frames', frame,':',str(frame+settings.cores))
    

    #deleting all the [0,0,0]
    truth = np.array(np.sign(sum(np.transpose(verts != [0,0,0]))-1)+1, dtype=bool)
    verts = verts[truth]

    #deleting anything with a z>2
    truth = np.array(sum(np.transpose(verts <= [-np.inf,-np.inf,2])), dtype=bool)
    verts = verts[truth]


    h, w = shape
    #consty multiplied by depth gives he height and since square width of the pix
    fovy = settings.depth_intrinsics.ppy*90/math.pi
    consty = 2*math.tan(fovy)/h
    
    arr_1d_0_0_py = (verts*np.full_like(verts, [0,0,consty])).reshape(len(verts)*3)
    arr_1d_0_py_0 = np.append(np.delete(arr_1d_0_0_py, 0), 0)
    add_py_to_y = arr_1d_0_py_0.reshape(len(verts),3)

    arr_1d_py_0_0 = np.append(np.delete(arr_1d_0_py_0, 0), 0)
    add_py_to_x = arr_1d_py_0_0.reshape(len(verts),3)

    points1 = verts.copy()
    points2 = verts + add_py_to_x
    points3 = verts + add_py_to_y
    points4 = points3 + add_py_to_x

    points = np.concatenate((points1,points2,points3,points4))

    #the way points are set up face must 0,v,2v,3v and then 1,v+1,2v+1,3v+1 and so on till v-1, 2v-1, 3v-1, 4v-1
    #for triangles it should be 0,v,2v and v,2v,3v
    face1 = np.arange(3)*len(verts)
    faces1 = np.full((len(verts),3), face1) + np.arange(len(verts)).reshape(len(verts),1)

    face2 =  np.arange(1,4)*len(verts)
    faces2 = np.full((len(verts),3), face2) + np.arange(len(verts)).reshape(len(verts),1)

    faces = np.concatenate((faces1,faces2))

    MathTime = timeit.default_timer()
    obj_path = path+file_name+str(frame+1)+'.stl'
    #NewFile(obj_path,points,faces)
    # Create the mesh
    cube = mesh.Mesh(np.zeros(faces.shape[0], dtype=mesh.Mesh.dtype))
    for i, f in enumerate(faces):
        for j in range(3):
            cube.vectors[i][j] = points[f[j],:]

    # Write the mesh to file
    cube.save(obj_path)
    FileTime = timeit.default_timer()
    if frame%settings.cores == 0: print("Done", MathTime-startTime, FileTime-MathTime)
