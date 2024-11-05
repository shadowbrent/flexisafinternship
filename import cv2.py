import cv2
import numpy as np
import matplotlib.pyplot as plt

# Open the image
img = cv2.imread('dancing-spider.jpg')
img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)  # Convert to HSV color space

# Define the color range for detection (e.g., for red color)
lower_color = np.array([0, 120, 70])     # Lower bound for red hue
upper_color = np.array([10, 255, 255])   # Upper bound for red hue

# Create a mask for the color
mask = cv2.inRange(img_hsv, lower_color, upper_color)
color_detected = cv2.bitwise_and(img, img, mask=mask)

# Find contours on the color-detected mask
contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Draw contours on a copy of the original image
contour_image = img.copy()
cv2.drawContours(contour_image, contours, -1, (0, 255, 0), 2)  # Draw green contours

# Apply Canny edge detection
edges = cv2.Canny(img, 100, 200, 3, L2gradient=True)

# Display results
# Show original image with color detection
plt.figure()
plt.title('Detected Color with Contours')
plt.imshow(cv2.cvtColor(contour_image, cv2.COLOR_BGR2RGB))
plt.axis('off')
plt.show()

# Show edge detection
plt.figure()
plt.title('Canny Edge Detection')
plt.imshow(edges, cmap='gray')
plt.axis('off')
plt.show()

# Save the images
plt.imsave('dancing-spider-color-detected.png', cv2.cvtColor(contour_image, cv2.COLOR_BGR2RGB))
plt.imsave('dancing-spider-canny.png', edges, cmap='gray', format='png')
