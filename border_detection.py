import cv2
import numpy as np

FILENAME = "5.JPG"

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

clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
clahe_image = clahe.apply(bright_contrast_image)

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

# Posterior a la dilatacion se puede aplicar erosion, sin embargo no parece necesario
closed_edges = cv2.erode(dilated_edges, kernel)

# Contorno
contours, hierarchy = cv2.findContours(dilated_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnt = contours[0]
c = max(contours, key=cv2.contourArea)

epsilon = 0.99*cv2.arcLength(c,True)
approx = cv2.approxPolyDP(c,epsilon,True)

# Contornos SIN aproximacion
cv2.drawContours(img_original, [c], -1, (0,255,0), 3)

# Contornos CON aproximacion
cv2.drawContours(img_original2, [approx], -1, (0,255,0), 3)

# Rectangulo
x,y,w,h = cv2.boundingRect(cnt)
cv2.rectangle(img_original,(x,y),(x+w,y+h),(0,255,0),2)

rect = cv2.minAreaRect(c)
box = cv2.boxPoints(rect)
box = np.int32(box)
cv2.drawContours(img_original,[box],0,(0,0,255),2)





# Display result
# Canny engrosado
cv2.imshow("Canny Dilated", dilated_edges)

# Canny erosion
#cv2.imshow("Canny Dilated", closed_edges)

cv2.imshow("Canny Edge Detection", edges6)

cv2.imshow("Closed", closed_edges)

cv2.imshow("Contour", img_original)

cv2.imshow("Approx Contour", img_original2)


cv2.imshow("WHITER", bright_contrast_image)
cv2.imshow("Equalized", equalized)
cv2.imshow("Threshold", thresh)

# BLUR
#cv2.imshow("BLUR", blur)

cv2.waitKey(0)
cv2.destroyAllWindows()
