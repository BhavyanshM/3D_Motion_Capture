import cv2
import numpy as np
import time

cap1 = cv2.VideoCapture(1)
cap2 = cv2.VideoCapture(2)
cap1.set(cv2.CAP_PROP_FRAME_WIDTH, 512)
cap1.set(cv2.CAP_PROP_FRAME_HEIGHT, 384)
cap2.set(cv2.CAP_PROP_FRAME_WIDTH, 512)
cap2.set(cv2.CAP_PROP_FRAME_HEIGHT, 384)


	# img = cv2.imread("Chess.JPG")

px1, px2, py1, py2 = 0, 0, 0, 0

while True:
	ret1, img1 = cap1.read()
	assert ret1
	ret2, img2 = cap2.read()
	assert ret2


	# res = cv2.add(res, resG)
	# res = cv2.add(res, resB)
	cv2.namedWindow("Image1", cv2.WINDOW_NORMAL)
	cv2.namedWindow("Image2", cv2.WINDOW_NORMAL)
	cv2.resizeWindow("Image1", 800, 600)
	cv2.resizeWindow("Image2", 800, 600)

	lower_red = np.array([47, 78, 93])
	upper_red = np.array([72, 162, 245])
	# print(lh, uh, ls, us, lv, uv)
	# (47 72 78 162 93 245)

	hsv1 = cv2.cvtColor(img1,cv2.COLOR_BGR2HSV)
	hsv2 = cv2.cvtColor(img2,cv2.COLOR_BGR2HSV)

	
	mask1 = cv2.inRange(hsv1, lower_red, upper_red)
	M = cv2.moments(mask1)
	cX1 = int(M["m10"] / M["m00"]) if M["m00"] != 0 else px1
	cY1 = int(M["m01"] / M["m00"]) if M["m00"] != 0 else py1
	cv2.line(mask1, (cX1 + 20, cY1), (cX1 - 20, cY1), 255, 1)
	cv2.line(mask1, (cX1, cY1 + 20), (cX1, cY1 - 20), 255, 1)
	cv2.imshow('Image1', mask1)

	mask2 = cv2.inRange(hsv2, lower_red, upper_red)
	M = cv2.moments(mask2)
	cX2 = int(M["m10"] / M["m00"]) if M["m00"] != 0 else px2
	cY2 = int(M["m01"] / M["m00"]) if M["m00"] != 0 else py2
	cv2.line(mask2, (cX2, cY2 + 20), (cX2, cY2 - 20), 255, 1)
	cv2.line(mask2, (cX2 + 20, cY2), (cX2 - 20, cY2), 255, 1)
	cv2.imshow('Image2', mask2)

	px1, py1, px2, py2 = cX1, cY1, cX2, cY2
	

	# cv2.imshow("Image1", img1)
	# cv2.imshow("Image2", img2)

	code = cv2.waitKeyEx(1)
	if code == ord('q'):
		break

cv2.destroyAllWindows()