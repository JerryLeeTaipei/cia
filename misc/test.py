import cv2
import matplotlib.pyplot as plt


img=cv2.imread('grid.jpg',1)

laplacian = cv2.Laplacian(img,ddepth=cv2.CV_32F, ksize=17,scale=1,delta=0,borderType=cv2.BORDER_DEFAULT)
sobel = cv2.Sobel(img,ddepth=cv2.CV_32F,dx=1,dy=0, ksize=11,scale=1,delta=0,borderType=cv2.BORDER_DEFAULT)
scharr = cv2.Scharr(img,ddepth=cv2.CV_32F,dx=1,dy=0,scale=1,delta=0,borderType=cv2.BORDER_DEFAULT)
sobelx = cv2.Sobel(img,cv2.CV_64F,1,0,ksize=7)
sobely = cv2.Sobel(img,cv2.CV_64F,0,1,ksize=7)
    
images=[img,laplacian,sobel,scharr,sobelx,sobely]
titles=['Original','Laplacian','Sobel','Scharr', 'Sobel-x','Sobel-y']

for i in range(6):
    plt.subplot(3,2,i+1)
    plt.imshow(images[i],cmap = 'gray')
    plt.title(titles[i]), plt.xticks([]), plt.yticks([])

plt.show()
