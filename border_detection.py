import cv2
import numpy as np
from matplotlib import pyplot as plt

FILENAME = "11.jpg"

# Resize the image

image_raw = cv2.imread(f"piano_images/{FILENAME}", cv2.IMREAD_GRAYSCALE)

original_height, original_width = image_raw.shape[:2]
new_width = 450
aspect_ratio = new_width / original_width
new_height = int(original_height * aspect_ratio) 

img = cv2.resize(image_raw, (new_width, new_height))

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

convex_copy = drawing.copy()
convex_copy_cd = drawing.copy()

#drawing = cv2.dilate(drawing, kernel, iterations=1)     .02
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

#convex_copy = cv2.dilate(convex_copy, kernel, iterations=1)

# INTENTO SUBPIX
# Prepare corner coordinates as float32 (required)
corners_float = np.float32(corners).reshape(-1, 1, 2)
# Define criteria for the refinement (stop after 30 iterations or epsilon < 0.01)
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.01)
# Refine corner positions
cv2.cornerSubPix(convex_copy, corners_float, winSize=(5, 5), zeroZone=(-1, -1), criteria=criteria)
# Draw refined corners
for point in corners_float:
    x, y = point[0]
    cv2.circle(convex_copy, (int(x), int(y)), 6, (255, 0, 0), -1)  # Blue = refined
    print(f"Refined x: {x:.2f} | y: {y:.2f}")

# Display result
#cv2.imshow("Threshold", thresh)

cv2.imshow("CORN", convex_copy)

#cv2.imshow("Edges COrners", convex_copy_cd)

cv2.imshow("Hull", drawing)
# BLUR
#cv2.imshow("BLUR", blur)

cv2.waitKey(0)
cv2.destroyAllWindows()