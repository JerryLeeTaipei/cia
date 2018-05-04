import cv2
import numpy as np

camera = cv2.VideoCapture(1)

# create a kernel for the dilation operation,
k=np.ones((3,3),np.uint8)

# initialize the first frame
f1_gray = None

while(True):
    # grab the current frame
    (grabbed, f2) = camera.read()
    # if the frame could not be grabbed, end it.
    if not grabbed:
        break
    
    # convert it to grayscale, and blur it
    f2_gray = cv2.cvtColor(f2, cv2.COLOR_BGR2GRAY)
    f2_gray = cv2.GaussianBlur(f2_gray, (21, 21), 0)

    # if the first frame has not set, initialize it
    if f1_gray is None:
        f1_gray = f2_gray
        continue

    # compute the absolute difference between the current frame and the last frame
    frameDelta = cv2.absdiff(f1_gray, f2_gray) 
    # convert this noise-removed output into a binary image
    ret, th = cv2.threshold(frameDelta, 25, 255, cv2.THRESH_BINARY)
    # dilate the image so that it is easier for us to find the boundary clearly
    dilated=cv2.dilate(th, k, iterations=2)
    # find the contour
    im2, contours, hierarchy= cv2.findContours(dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # draw the contour
    o_frame = f2
    # 1 draw all countours
    cv2.drawContours(o_frame, contours, -1, (0,255,0), 2 )
    # 2 draw the bounding box for the biggest objects
    max_area = 0
    x = 0
    y = 0
    w = 0
    h = 0
    for c in contours:
        # if the contour is small, ignore it
        area = cv2.contourArea(c)
        if area < max_area:
            continue
        max_area = area
        # compute the bounding box for the contour, draw it on the frame,
        (x, y, w, h) = cv2.boundingRect(c)
        
    cv2.rectangle(o_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)	
    cv2.imshow('Output', o_frame )
    # assign the latest frame to the older frame
    f1_gray = f2_gray
    # terminate the loop once we detect the Esc keypress
    if cv2.waitKey(5) == 27 :
        break

# release the camera and destroy the display window
camera.release()
cv2.destroyAllWindows()
