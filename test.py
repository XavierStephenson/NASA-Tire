import numpy as np
from additional_functions import *
outfile = 'Outputs/Color Ref/Raw.npz'
npzfile = np.load(outfile)
verts = npzfile['all_verts'][0]
color = npzfile['all_color_images'][0]
from scipy.spatial import Voronoi, voronoi_plot_2d
import matplotlib.pyplot as plt


step = 6
truth_tire, truth_floor = Arrays(verts, color)
h, w = color.shape[:2]

print('Seperated Tire')
#tire_gaps = FindGaps(verts, truth_floor, color.shape[:2], step, color.copy())
print('Found Gaps')

cv2.imshow('Orig',color)
new_color = np.zeros_like(color) + 255
new_color *= truth_tire
#color += np.array([0,0,255], dtype='uint8')


cv2.imshow('Seperate',new_color)
#plt.show()
def mouse(event, x, y, flag, param):
   
    #images 0 is color source
    #images 1 is vert
    images = param
    
    cv2.namedWindow("Data", cv2.WINDOW_AUTOSIZE)
    img = cv2.imread('Images/blank.png')
    hsv_image = cv2.cvtColor(color, cv2.COLOR_BGR2HSV)

    #display color info
    this_color = color[y][x]
    color_text = str(this_color)

    [c1, c2, c3] = this_color
    cv2.putText(img, "HSV            BRG", (10, 20), cv2.FONT_HERSHEY_SIMPLEX , .5, (0, 0, 0), 1, cv2.LINE_AA) 
    cv2.putText(img, color_text, (10, 50), cv2.FONT_HERSHEY_SIMPLEX , .5, (0, 0, 0), 1, cv2.LINE_AA) 
    cv2.rectangle(img, (200, 5), (250, 25), (int(c1), int(c2), int(c3)), -1)

    this_color = new_color[y][x]
    color_text = str(this_color)
    cv2.putText(img, color_text, (10, 80), cv2.FONT_HERSHEY_SIMPLEX , .5, (0, 0, 0), 1, cv2.LINE_AA) 

    
    cv2.imshow('Data',img)
cv2.setMouseCallback('Orig', mouse)   
cv2.setMouseCallback('Seperate', mouse)   
while True:
    cv2.waitKey(1)