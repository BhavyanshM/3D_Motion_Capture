import cv2
import numpy as np
from scipy import signal


img = cv2.imread("frame_2.jpg")

img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

img = np.array(img)

gaussian = np.array([[1,5,1],[5,10,5],[1,5,1]])/34
gauss = cv2.filter2D(img, -1, gaussian)

grad_x = np.array([[1,0,-1],[1,0,-1],[1,0,-1]])
grad_y = np.array([[1,0,1],[0,0,0],[-1,0,-1]])
gx = cv2.filter2D(gauss, -1, grad_y)
gy = cv2.filter2D(gauss, -1, grad_x)

grad = ((gx) + (gy))


print(grad)

cv2.namedWindow("Original", cv2.WINDOW_NORMAL)
cv2.resizeWindow('Original', 1920,1280)
cv2.imshow("Original", gauss)

cv2.namedWindow("Tracer", cv2.WINDOW_NORMAL)
cv2.resizeWindow('Tracer', 1920,1280)
cv2.imshow("Tracer", grad)

cv2.waitKey(0)

