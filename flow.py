import cv2
import numpy as np
import time

cap1 = cv2.VideoCapture(0)
cap2 = cv2.VideoCapture(1)

# cap.set(cv2.CAP_PROP_FRAME_WIDTH, 400)
# cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 300)
	# img = cv2.imread("Chess.JPG")

ret1, temp1 = cap1.read()
prev1 = temp1

ret2, temp2 = cap2.read()
prev2 = temp2


prevTime = time.time()
num_frames = 0
total_time = 0

while True:
	num_frames += 1
	ret1, temp1 = cap1.read()
	img1 = temp1

	ret2, temp2 = cap2.read()
	img2 = temp2


	diff1 = cv2.subtract(img1, prev1)
	diff2 = cv2.subtract(img2, prev2)

	cv2.namedWindow('Result1', cv2.WINDOW_NORMAL)
	cv2.resizeWindow('Result1', 800, 600)
	cv2.imshow("Result1", diff1)
	prev1 = img1

	cv2.namedWindow('Result2', cv2.WINDOW_NORMAL)
	cv2.resizeWindow('Result2', 800, 600)
	cv2.imshow("Result2", diff2)
	prev2 = img2


	curTime = time.time()
	diffTime = curTime - prevTime
	prevTime = curTime
	total_time += diffTime
	if total_time > 1:
		print('FPS:',num_frames, total_time)
		total_time = 0
		num_frames = 0

	code = cv2.waitKeyEx(1)
	if code == ord('q'):
		break

cv2.destroyAllWindows()