import cv2
import numpy as np
from matplotlib import pyplot as plt

def shape_detect(c):
    # initialize the shape name and approximate the contour
    shape = "unidentified"
    peri = cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, 0.04 * peri, True)
    # if the shape is a triangle, it will have 3 vertices
    if len(approx) == 3:
        shape = "triangle"
    # if the shape has 4 vertices, it is either a square or
    # a rectangle
    elif len(approx) == 4:
        # compute the bounding box of the contour and use the
        # bounding box to compute the aspect ratio
        (x, y, w, h) = cv2.boundingRect(approx)
        ar = w / float(h)
        # a square will have an aspect ratio that is approximately
        # equal to one, otherwise, the shape is a rectangle
        shape = "square" if ar >= 0.95 and ar <= 1.05 else "rectangle"    
    # return the name of the shape
    return shape

# load the image and resize it to a smaller factor so that
# the shapes can be approximated better
src = cv2.imread('sign-warn.jpg')
resized = cv2.resize(src ,None,fx=0.5,fy=0.5, interpolation=cv2.INTER_AREA)
ratio = 2
 
# convert the resized image to grayscale, blur it slightly,
# and threshold it
#gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
#blurred = cv2.GaussianBlur(gray, (5, 5), 0)
#thresh = cv2.threshold(blurred, 60, 255, cv2.THRESH_BINARY)[1]

img = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB )
height, width = img.shape[:2]
r,g,b = cv2.split (img)
r_mean = np.mean(r)
g_mean = np.mean(g)
b_mean = np.mean(b)
# filter out the red from the light: (g,b) val > mean
red_mask = np.logical_and( (r > r_mean), (g > g_mean), ( b > b_mean ) )
img_r = r.copy()
img_r[red_mask] = 0
img_r = cv2.GaussianBlur(img_r,(5,5),0)

# find contours in the thresholded image and initialize the
# shape detector
im2, cnts, hierarchy = cv2.findContours(img_r, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

shape = "undefined"
# loop over the contours
for c in cnts:
    # detect the name of the shape using only the contour
    shape = shape_detect(c)
    # multiply the contour (x, y)-coordinates by the resize ratio,
    # then draw the contours and the name of the shape on the image
    c = c.astype("float")
    c *= ratio
    c = c.astype("int")
    cv2.drawContours(src, [c], 0, (0, 255, 0), 2)
    # compute the center of the contour, then draw the text
    M = cv2.moments(c)
    if (M["m00"] !=0) and (M["m00"] !=0) :
        cX = int((M["m10"] / M["m00"]) )
        cY = int((M["m01"] / M["m00"]) )	
        cv2.putText(src, shape, (cX, cY), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        print("draw %s(%d,%d)\n", shape, cX, cY)
    else:
        print("%s\n", shape)

# show the output image
cv2.imshow("Image", src)
cv2.waitKey(0)

