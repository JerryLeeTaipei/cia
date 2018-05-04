import cv2
import pymeanshift as pms
from matplotlib import pyplot as plt

original_image = cv2.imread("licensePlate.jpg")
#changing the colorspace from BGR->RGB
input_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB )

(segmented_image, labels_image, number_regions) = pms.segment(input_image, spatial_radius=6, range_radius=4.5, min_density=50)


plt.subplot(131),plt.imshow(input_image),plt.title('input_image')
plt.xticks([]),plt.yticks([])
plt.subplot(132),plt.imshow(segmented_image),plt.title(
'Segmented Output')
plt.xticks([]),plt.yticks([])
plt.subplot(133),plt.imshow(labels_image),plt.title(
'Labeled Output')
plt.xticks([]),plt.yticks([])
plt.show()
