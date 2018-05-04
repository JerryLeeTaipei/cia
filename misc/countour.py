import numpy as np
import cv2
import matplotlib.pyplot as plt

im = cv2.imread('licensePlate.jpg')
imgray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
ret, thresh = cv2.threshold(imgray, 127, 255, 0)
im2, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)


cv2.drawContours(im, contours, 100, (0,255,0), 3)
plt.imshow(im)
plt.title('Contours')
plt.xticks([])
plt.yticks([])
plt.show()
