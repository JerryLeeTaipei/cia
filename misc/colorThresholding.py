import cv2
import numpy as np
from matplotlib import pyplot as plt

src = cv2.imread('sign-limit.jpg')
img = cv2.cvtColor(src, cv2.COLOR_BGR2RGB )
r,g,b = cv2.split (img)
r_mean = np.mean(r)
g_mean = np.mean(g)
b_mean = np.mean(b)
# filter out the red from the light: (g,b) val > mean 
red_mask = np.logical_and( (g > g_mean), ( b > b_mean ) )


img_r = r.copy()
img_r[red_mask] = 0
img_r = cv2.GaussianBlur(img_r,(5,5),0)

# global thresholding
ret1,th1 = cv2.threshold(img_r, r_mean, 255,cv2.THRESH_BINARY)

# plot all the images
images = [img, r, img_r, th1]
titles = ['img','red',
          'red_mask', 'red_mask-th1']

for i in range(4):
    plt.subplot(2,2,i+1)
    plt.imshow(images[i],cmap = 'gray')
    plt.title(titles[i])
    plt.xticks([])
    plt.yticks([])
    
plt.show()
