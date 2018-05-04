import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread('/home/jerry/licensePlate.jpg',0) # Load an image
gray = cv2.cvtColor(img,cv2.COLOR_GRAY2BGR) # Convert it to grayscale
edges1 = cv2.Canny(gray,50,150,apertureSize = 3)
lines = cv2.HoughLines(edges1,1,np.pi/180,150) # 200,

if lines is not None:
    #for x in range(0, len(lines)):
    for line in lines:
        #for d,theta in lines[x]:
        for d,theta in line:
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a*d
            y0 = b*d
            x1 = int(x0 + 1000*(-b))
            y1 = int(y0 + 1000*(a))
            x2 = int(x0 - 1000*(-b))
            y2 = int(y0 - 1000*(a))
            cv2.line(img,(x1,y1),(x2,y2),(0,0,255),2)
            print("(%d,%d):(%d,%d)-(%d,%d)\n",d,theta , x1,y1, x2, y2)
else:
    print('no line is found\n')

images = [img,gray,edges1]
titles = ['Original','Gray','Edges']
for i in range(3):
    plt.subplot(1,3,i+1)
    plt.imshow(images[i],cmap = 'gray')
    plt.title(titles[i]),
    plt.xticks([]), plt.yticks([])

plt.show()

