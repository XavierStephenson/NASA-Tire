import numpy as np
import cv2
color = np.array([[48,23,20],[101,5,6],[3,101,5],[50,30,40],[19,15,26]])
truth_tire = np.ones_like(color)


outfile = 'Outputs/Color Ref/Raw.npz'
npzfile = np.load(outfile)
ref_color_image_img = npzfile['all_color_images'][0]

ref_color_image = ref_color_image_img.reshape(ref_color_image_img.size//3, 3)
truth1 = ref_color_image > [0,0,0]
truth2 = ref_color_image < [110,110,110]
tire_pos = sum(np.transpose(truth1*truth2)) == 3

outfile = 'Outputs/Color Ref2/Raw.npz'
npzfile = np.load(outfile)
color_image_img = npzfile['all_color_images'][0]


color_image  = color_image_img.reshape(color_image_img.size//3, 3)
tire_colors  = color_image[tire_pos]
floor_colors = color_image[np.logical_not(tire_pos)]






alpha = .5
img = cv2.addWeighted(color_image_img, alpha, ref_color_image_img, 1 - alpha, 0)
cv2.imshow('Ref2', color_image_img)
cv2.imshow('combine', img)

print(tire_colors)

tire_colors = np.transpose(tire_colors)
lil = np.amin(tire_colors,1)+1
big = np.amax(tire_colors,1)-1
print('Above',lil,'Below',big)

while True:
    key = cv2.waitKey(1)
    if key == 8:
        cv2.destroyAllWindows()
        npzfile.close()
        break
