import cv2
from matplotlib import pyplot as plt

img = cv2.imread('building.jpg',1)
input=cv2.cvtColor(img,cv2.COLOR_RGB2BGR)
histr_RED = cv2.calcHist([input],[0],None,[256],[0,256])
histr_GREEN = cv2.calcHist([input],[1],None,[256],[0,256])
histr_BLUE = cv2.calcHist([input],[2],None,[256],[0,256])

plt.subplot(221),plt.imshow(input),plt.title('Original Image'),plt.xticks([]),plt.yticks([])
plt.subplot(222),plt.plot(histr_RED,color='r'),
plt.title('Red'), plt.xlim([0,256]), plt.yticks([])
plt.subplot(223),plt.plot(histr_GREEN,color='g'), plt.title('Green'),
plt.xlim([0,256]), plt.yticks([])
plt.subplot(224),plt.plot(histr_BLUE,color='b'), plt.title('Blue'),
plt.xlim([0,256]), plt.yticks([])
plt.show()
