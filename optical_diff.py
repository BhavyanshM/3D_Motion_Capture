import cv2
import numpy as np
import time

cap = cv2.VideoCapture(0)


cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)
cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)
cap.set(cv2.CAP_PROP_EXPOSURE, -4)
time.sleep(2)
# cap.set(cv2.CAP_PROP_FRAME_WIDTH, 400)
# cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 300)
	# img = cv2.imread("Chess.JPG")

# ret, prev = cv2.threshold(temp, 127, 255, cv2.THRESH_BINARY)
prevTime = time.time()
num_frames = 0
total_time = 0

print("Sleeping")
# time.sleep(5)
ret, first = cap.read()
print("Captured!")

while True:
	num_frames += 1
	ret, img = cap.read()
	# ret, img = cv2.threshold(temp, 127, 255, cv2.THRESH_BINARY)
	# img = img[200:650,200:650,:]


	diff = cv2.absdiff(img, first)
	mask = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
	thresh = 15
	imask = mask>thresh
	canvas = np.zeros_like(img, np.uint8)
	canvas[imask] = img[imask]



	cv2.namedWindow('Result', cv2.WINDOW_NORMAL)
	cv2.resizeWindow('Result', 1800,1350)
	cv2.imshow("Result", canvas)

	# cv2.namedWindow('First', cv2.WINDOW_NORMAL)
	# cv2.resizeWindow('First', 1800,1350)
	# cv2.imshow("First", first)


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