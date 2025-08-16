import cv2
import numpy as np
from matplotlib import pyplot as plt

FILENAME = "29.png"
RESIZE_WIDTH = 450
PIANO_AREA_XSECTION_OFFSET = 10
PIANO_AREA_YSECTION_PERCENTAGE = 0.4


def is_piano_inside_area(corners, img_height):
    piano_area_ysection = img_height * PIANO_AREA_YSECTION_PERCENTAGE
    for corner in corners:
        x, y = corner.ravel()
        if not ((PIANO_AREA_XSECTION_OFFSET <= x <= (RESIZE_WIDTH - PIANO_AREA_XSECTION_OFFSET))
            and (0 <= y <= piano_area_ysection)):
            return False
    return True
        


# Resize the image

image_raw = cv2.imread(f"piano_images/{FILENAME}", cv2.IMREAD_GRAYSCALE)

original_height, original_width = image_raw.shape[:2]
aspect_ratio = RESIZE_WIDTH / original_width
new_height = int(original_height * aspect_ratio) 

img = cv2.resize(image_raw, (RESIZE_WIDTH, new_height))

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

_, thresh = cv2.threshold(equalized, 185, 200, cv2.THRESH_TOZERO)
"""
_, thresh = cv2.threshold(equalized, 176, 255, cv2.THRESH_TOZERO)
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



# Shi-Tomasi corner detection
corners_st = cv2.goodFeaturesToTrack(
    drawing,
    maxCorners=4,
    qualityLevel=0.01,
    minDistance=30,
    useHarrisDetector=False
)

if corners_st is not None:
    corners_st = np.int32(corners_st)
    for corner in corners_st:
        x, y = corner.ravel()
        cv2.circle(drawing, (x, y), 3, (255, 0, 0), -1)
    if is_piano_inside_area(corners_st, new_height):
        print("INSIDE")
    else:
        print("OUTSIDE")
    





# Recordar que new height no es el verdadero tamanho de la imagen
# Sin embargo es mejor reducir el tamanho de la imagen asi que no importa
piano_delimiter_section_height = new_height * PIANO_AREA_YSECTION_PERCENTAGE
drawing = cv2.line(drawing, (0, int(piano_delimiter_section_height)), (RESIZE_WIDTH, int(piano_delimiter_section_height)), (255, 255, 255), 1)

drawing = cv2.line(drawing,
                   (PIANO_AREA_XSECTION_OFFSET, 0),
                   (PIANO_AREA_XSECTION_OFFSET, int(piano_delimiter_section_height)),
                   (255, 255, 255), 1)
drawing = cv2.line(drawing,
                   (RESIZE_WIDTH - PIANO_AREA_XSECTION_OFFSET, 0),
                   (RESIZE_WIDTH - PIANO_AREA_XSECTION_OFFSET, int(piano_delimiter_section_height)),
                   (255, 255, 255), 1)







# Display result
cv2.imshow("Original", img)

cv2.imshow("BRIGHTER", bright_contrast_image)

cv2.imshow("Threshold", thresh)

cv2.imshow("GoodFeatures", drawing)

# BLUR
#cv2.imshow("BLUR", blur)

cv2.waitKey(0)
cv2.destroyAllWindows()
