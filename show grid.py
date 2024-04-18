import numpy as np
import cv2
from additional_functions import *
from scipy.spatial import Voronoi, voronoi_plot_2d
import matplotlib.pyplot as plt

def onclick(event, x,y,flags, param):
    cv2.namedWindow("Contact Data", cv2.WINDOW_AUTOSIZE)
    img = cv2.imread('Images/blank.png')


    i1 = x//step
    j1 = y//step

    avg_depth = str(tire_gaps[j1][i1])

    point = np.array((x,y))
    distances = np.linalg.norm(gap_centers-point, axis=1)
    min_index = np.argmin(distances)
    i,j = gap_centers[min_index]/step
    i = int(i)
    j = int(j)
    dist = distances[min_index]

    depth = verts[y*w+x][2]

    this_avg_depth_closest = avg_depth_closest[y][x]
    if depth== 0 :
        diff = "False"
    else:
        diff = this_avg_depth_closest-depth
    cv2.putText(img, "Actually Depth and Closest Average Floor Depth", (10, 20), cv2.FONT_HERSHEY_SIMPLEX , .25, (0, 0, 0), 1, cv2.LINE_AA) 
    cv2.putText(img, str(depth), (10, 50), cv2.FONT_HERSHEY_SIMPLEX , .5, (0, 0, 0), 1, cv2.LINE_AA) 
    #cv2.putText(img, str(avg_depth), (10, 80), cv2.FONT_HERSHEY_SIMPLEX , .5, (0, 0, 0), 1, cv2.LINE_AA) 
    cv2.putText(img, str(this_avg_depth_closest), (10, 80), cv2.FONT_HERSHEY_SIMPLEX , .5, (0, 0, 0), 1, cv2.LINE_AA) 
    cv2.putText(img, str(diff), (10, 110), cv2.FONT_HERSHEY_SIMPLEX , .5, (0, 0, 0), 1, cv2.LINE_AA) 

    cv2.putText(img, "Dist Of Closest", (10, 140), cv2.FONT_HERSHEY_SIMPLEX , .5, (0, 0, 0), 1, cv2.LINE_AA)
    cv2.putText(img, str(i)+','+str(j)+'----'+str(i1)+','+str(j1), (10, 170), cv2.FONT_HERSHEY_SIMPLEX , .5, (0, 0, 0), 1, cv2.LINE_AA)
    cv2.putText(img, str(dist), (10, 200), cv2.FONT_HERSHEY_SIMPLEX , .5, (0, 0, 0), 1, cv2.LINE_AA)


    cv2.imshow('Contact Data', img) 


outfile = 'Outputs/New/Raw.npz'
npzfile = np.load(outfile)
verts = npzfile['all_verts'][0]
color = npzfile['all_color_images'][0]
print('Loaded Data')

step = 5
img = cv2.imread('Images/blank.png')
truth_tire, truth_floor = Arrays(verts, color)
#truth_tire, truth_floor = ArraysOld(verts, color)
print('Seperated Tire')

tire_gaps = FindGaps(verts, truth_floor, color.shape[:2], step, color.copy())
print('Found Gaps')

overlay = color.copy()*0
gap_centers = []
for j in range(len(tire_gaps)):
    for i in range(len(tire_gaps[j])):
        if tire_gaps[j][i]:
            cv2.rectangle(overlay, (i*step, j*step), ((i+1)*step, (j+1)*step), (255,255,255), -1)
            gap_centers.append([(i+.5)*step,(j+.5)*step])
gap_centers = np.array(gap_centers)
print('Found Gap Centers')
h, w = color.shape[:2]
"""
avg_depth_closest = np.zeros((h,w), dtype=float)
for index in range(len(verts)):
    x = index%w
    y = index//w
    if x==w-1 and y%50 == 0: 
        text = str(int((index+(w*50))/len(verts)*1000)/10)+'%'
        print(text)
    if truth_tire[y][x][0]:

        point = np.array((x,y))
        distances = np.linalg.norm(gap_centers-point, axis=1)
        min_index = np.argmin(distances)
        i,j = gap_centers[min_index]/step
        i = int(i)
        j = int(j)
        dist = distances[min_index]
        if dist < 30:
            avg_depth_closest[y][x] = tire_gaps[j][i]"""

alpha = .6
img = cv2.addWeighted(overlay, alpha, color, 1 - alpha, 0)
cv2.imshow('Color', img)

#cv2.setMouseCallback('Color', onclick)  

vor = Voronoi(gap_centers*np.full_like(gap_centers,[1,-1]))
#fig = voronoi_plot_2d(vor)
#print(vor.regions)
this_color = 0
for region in vor.regions:
    this_color += 36
    pts = []
    for index in region:
        x, y = np.array(vor.vertices[index], dtype=int)
        y *= -1
        if x>0 and y>0:
            if x<w and y<h:
                pts.append([x,y])
        cv2.rectangle(img, (x, y), (x+5, y+5), (0,0,this_color), -1)
    if len(pts)>2:
        #print(pts)
        pts = np.array(pts)
        #pts *= np.full_like(pts,[1,-1])
        cv2.fillPoly(img, np.int32([pts]), (this_color,0,this_color))



#cv2.imshow('Color', img)
plt.show()


while True:
    cv2.waitKey(2)

