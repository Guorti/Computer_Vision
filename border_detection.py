import cv2
 
# Load image in grayscale
img = cv2.imread("piano_images/4.jpg", cv2.IMREAD_GRAYSCALE)
 
# Apply Gaussian Blur to reduce noise
blur = cv2.GaussianBlur(img, (5, 5), 1.4)
 
# Apply Canny Edge Detector
edges1 = cv2.Canny(blur, threshold1=100, threshold2=200)
#edges2 = cv2.Canny(blur, threshold1=100, threshold2=200, apertureSize=3, L2gradient=True)
#edges3 = cv2.Canny(blur, threshold1=100, threshold2=200, apertureSize=5)
edges3 = cv2.Canny(blur, threshold1=180, threshold2=200)

edges4 = cv2.Canny(blur, threshold1=160, threshold2=350)
#Buena opcion bajo contraste
edges5 = cv2.Canny(blur, threshold1=200, threshold2=440)
#Mejor opcion por el momento
edges6 = cv2.Canny(blur, threshold1=200, threshold2=395)

 
# Display result
cv2.imshow("Canny Edge Detection -1", edges1)

cv2.imshow("Canny Edge Detection -3", edges3)
cv2.imshow("Canny Edge Detection -4", edges4)
cv2.imshow("Canny Edge Detection -5", edges5)
cv2.imshow("Canny Edge Detection -6", edges6)

cv2.imshow("BLUR", blur)


cv2.waitKey(0)
cv2.destroyAllWindows()