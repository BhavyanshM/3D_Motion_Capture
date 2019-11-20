import cv2
import numpy as np
import time

cap = cv2.VideoCapture(0)
# cap.set(cv2.CAP_PROP_FRAME_WIDTH, 400)
# cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 300)
	# img = cv2.imread("Chess.JPG")

ret, temp = cap.read()
# ret, prev = cv2.threshold(temp, 127, 255, cv2.THRESH_BINARY)
prev = temp
prevTime = time.time()
num_frames = 0
total_time = 0

while True:
	num_frames += 1
	ret, temp = cap.read()
	# ret, img = cv2.threshold(temp, 127, 255, cv2.THRESH_BINARY)
	img = temp
	# img = img[200:650,200:650,:]


	diff = cv2.subtract(img, prev)

	cv2.namedWindow('Result', cv2.WINDOW_NORMAL)
	cv2.resizeWindow('Result', 1200,900)
	cv2.imshow("Result", diff)
	prev = img



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