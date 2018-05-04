import cv2
import matplotlib.pyplot as plt
img = cv2.imread('/home/jerry/albert.jpg',0)
edges1 = cv2.Canny(img,100,200,L2gradient=False)
edges2 = cv2.Canny(img,100,200,L2gradient=True)
images = [img,edges1,edges2]
titles = ['Original','L1 Gradient','L2 Gradient']
for i in range(3):
    plt.subplot(1,3,i+1)
    plt.imshow(images[i],cmap = 'gray')
    plt.title(titles[i]),
    plt.xticks([]), plt.yticks([])

plt.show()
