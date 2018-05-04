import numpy as np
import cv2

image=cv2.imread('barcode.jpg',1)
input = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

hor_der = cv2.Sobel(input, ddepth = -1 , dx = 1, dy = 0, ksize=5)
ver_der = cv2.Sobel(input, ddepth = -1 , dx = 0, dy = 1, ksize=5)
diff = cv2.subtract(hor_der, ver_der)
diff = cv2.convertScaleAbs(diff)

blur = cv2.GaussianBlur(diff, (3, 3),0)
ret, th = cv2.threshold(blur, 225, 255, cv2.THRESH_BINARY)
morph=4
dilated = cv2.dilate(th, None, iterations = morph)
eroded = cv2.erode(dilated, None, iterations = morph)

_, contours, _= cv2.findContours(eroded, cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
areas = [cv2.contourArea(temp) for temp in contours]
max_index = np.argmax(areas)
largest_contour=contours[max_index]


x,y,width,height = cv2.boundingRect(largest_contour)
cv2.rectangle(image,(x,y),(x+width,y+height),(0,255,0),2)
cv2.imshow('image',image)
cv2.waitKey(0)
cv2.destroyAllWindows()
