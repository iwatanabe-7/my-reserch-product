import cv2
import numpy as np

image = cv2.imread('data/image/my-run-dark-10-3/run_dark_10-3_000.jpg')

alpha = 5
beta = 100

adjusted_image = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)

cv2.imshow("Original Image", adjusted_image)

cv2.waitKey(0)
cv2.destroyAllWindows()
