import cv2
import numpy as np
import matplotlib.pyplot as plt

image=cv2.imread('licensePlate.jpg')
input = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
Z=input.reshape((-1,3))
Z=np.float32(Z)
criteria=(cv2.TERM_CRITERIA_EPS+ cv2.TERM_CRITERIA_MAX_ITER,10,1.0)
K=2
ret,label1,center1=cv2.kmeans(Z,K, None,criteria,10,cv2.KMEANS_RANDOM_CENTERS)
center1=np.uint8(center1)
res1=center1[label1.flatten()]
output1=res1.reshape((image.shape))
K=4
ret,label2,center2=cv2.kmeans(Z,K,None,criteria,10,cv2.KMEANS_RANDOM_CENTERS)
center2=np.uint8(center2)
res2=center2[label2.flatten()]
output2=res2.reshape((image.shape))
K=8
ret,label3,center3=cv2.kmeans(Z,K,None,criteria,10,cv2.KMEANS_RANDOM_CENTERS)

center3=np.uint8(center3)
res3=center3[label3.flatten()]
output3=res3.reshape((image.shape))
titles=['Original','K=2','K=4','K=8']
output=[input,output1,output2,output3]
for i in range(4):
    plt.subplot(2,2,i+1),plt.imshow(output[i]),plt.title(titles[i])
    plt.xticks([]),plt.yticks([])
    
plt.show()
