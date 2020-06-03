import cv2
import matplotlib.pyplot as plt

img = cv2.imread('./data/val/0016E5_07959.png')
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
mask = cv2.imread('./data/valannot/0016E5_07959.png', 0)

plt.subplot(121)
plt.imshow(img)
plt.grid(False)
plt.xticks([])
plt.yticks([])
plt.title('Image')
plt.subplot(122)
plt.imshow(mask)
plt.grid(False)
plt.xticks([])
plt.yticks([])
plt.title('Mask')
plt.savefig('./vis_images/example_data.png')
