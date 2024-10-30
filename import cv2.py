import cv2
import matplotlib.pyplot as plt

# Open the image
img = cv2.imread('dancing-spider.jpg')

# Apply Canny edge detection
edges = cv2.Canny(img, 100, 200, 3, L2gradient=True)

# Save and display the result
plt.figure()
plt.title('Spider')
plt.imsave('dancing-spider-canny.png', edges, cmap='gray', format='png')
plt.imshow(edges, cmap='gray')
plt.axis('off')
plt.show()
