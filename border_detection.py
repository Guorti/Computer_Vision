import cv2
import numpy as np
from matplotlib import pyplot as plt

FILENAME = "9.jpg"

# Load image in grayscale
img_original = cv2.imread(f"piano_images/{FILENAME}")
img_original2 = img_original.copy()
img = cv2.imread(f"piano_images/{FILENAME}", cv2.IMREAD_GRAYSCALE)
 
# Apply Gaussian Blur to reduce noise
#1.4
#blur = cv2.GaussianBlur(img, (5, 5), 2)
blur = cv2.bilateralFilter(img,9,50,100)

# Make whites whiter
alpha = 1.3  # Increase contrast (make whites whiter, blacks blacker)
beta = 1.2    # Increase brightness (shift all pixels up)
bright_contrast_image = cv2.convertScaleAbs(blur, alpha=alpha, beta=beta)

# Define mejor el contraste, permite identificar mejor los bordes
equalized = cv2.equalizeHist(bright_contrast_image)
#equalized = cv2.equalizeHist(equalized)

"""
Thresholding:
“Any pixel brighter than 185, make it completely white (255)
Any pixel 200 or darker, make it completely black (0).”

_, thresh = cv2.threshold(equalized, 185, 200, cv2.THRESH_TOZERO)
"""
_, thresh = cv2.threshold(equalized, 175, 255, cv2.THRESH_TOZERO)
#thresh = cv2.adaptiveThreshold(blur,255,cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY,11,2)

# Apply Canny Edge Detector
edges6 = cv2.Canny(thresh, threshold1=140, threshold2=350)

# Dilatacion para hacer mas gruesos los bordes de Canny porque son muy delgados
kernel = np.ones((3,3), np.uint8)
dilated_edges = cv2.dilate(edges6, kernel, iterations=1)

# Contorno
contours, hierarchy = cv2.findContours(dilated_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnt = contours[0]
c = max(contours, key=cv2.contourArea)

# Intento de obtener el rectangulo irregular del piano
# Compute the convex hull
hull = cv2.convexHull(c)

# Create an empty black image
drawing = np.zeros((thresh.shape[0], thresh.shape[1]), np.uint8)

# Define colors
color_hull = (255, 255, 255) # white

# Draw convex hull (must be in a list)
cv2.drawContours(drawing, [hull], -1, color_hull, 1, 8)
#drawing = cv2.dilate(drawing, kernel, iterations=1)
epsilon = 0.02 * cv2.arcLength(hull, True)
approx = cv2.approxPolyDP(hull, epsilon, True)
# Check if it forms a quadrilateral
if len(approx) == 4:
    corners = approx.reshape(-1, 2)  # shape: (4, 2)
    # Convert to color image if needed
    if len(drawing.shape) == 2 or drawing.shape[2] == 1:
        drawing = cv2.cvtColor(drawing, cv2.COLOR_GRAY2BGR)
    # Draw corners
    for (x, y) in corners:
        cv2.circle(drawing, (x, y), radius=6, color=(0, 0, 255), thickness=-1)  # Red dots
        print(f"x: {x} | y: {y}")



# Display result
cv2.imshow("Threshold", thresh)

cv2.imshow("Hull", drawing)

# BLUR
#cv2.imshow("BLUR", blur)

cv2.waitKey(0)
cv2.destroyAllWindows()