import cv2
import numpy as np
import time

cap1 = cv2.VideoCapture(0)
cap2 = cv2.VideoCapture(1)

cap1.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)
cap1.set(cv2.CAP_PROP_AUTOFOCUS, 0)
cap1.set(cv2.CAP_PROP_EXPOSURE, -4)

cap2.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)
cap2.set(cv2.CAP_PROP_AUTOFOCUS, 0)
cap2.set(cv2.CAP_PROP_EXPOSURE, -4)

# cap.set(cv2.CAP_PROP_FRAME_WIDTH, 400)
# cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 300)
	# img = cv2.imread("Chess.JPG")

# ret, prev = cv2.threshold(temp, 127, 255, cv2.THRESH_BINARY)
prevTime = time.time()
num_frames = 0
total_time = 0

# time.sleep(5)
ret1, first1 = cap1.read()
ret2, first2 = cap2.read()
set1, set2 = 0, 0
print("Captured!")

while True:
	num_frames += 1
	ret1, img1 = cap1.read()
	ret2, img2 = cap2.read()
	# ret, img = cv2.threshold(temp, 127, 255, cv2.THRESH_BINARY)
	# img = img[200:650,200:650,:]


	diff1 = cv2.absdiff(img1, first1)
	mask1 = cv2.cvtColor(diff1, cv2.COLOR_BGR2GRAY)
	thresh1 = 15
	imask1 = mask1>thresh1
	canvas1 = np.zeros_like(img1, np.uint8)
	canvas1[imask1] = img1[imask1]

	diff2 = cv2.absdiff(img2, first2)
	mask2 = cv2.cvtColor(diff2, cv2.COLOR_BGR2GRAY)
	thresh2 = 15
	imask2 = mask2>thresh2
	canvas2 = np.zeros_like(img2, np.uint8)
	canvas2[imask2] = img2[imask2]

	tot1, tot2 = np.sum(imask1), np.sum(imask2)

	# print("Img1:", tot1, "\tImg2:", tot2)

	# if tot1 > 1000 and set1 == 0:
	# 	first1 = img1
	# 	set1 = 1
	# 	print("Changed Image 1")
	# if tot2 > 1000 and set2 == 0:
	# 	first2 = img2
	# 	set2 = 1
	# 	print("Changed Image 2")


	cv2.namedWindow('Result1', cv2.WINDOW_NORMAL)
	cv2.resizeWindow('Result1', 800,600)
	cv2.imshow("Result1", canvas1)

	cv2.namedWindow('Result2', cv2.WINDOW_NORMAL)
	cv2.resizeWindow('Result2', 800,600)
	cv2.imshow("Result2", canvas2)

	# cv2.namedWindow('First', cv2.WINDOW_NORMAL)
	# cv2.resizeWindow('First', 1800,1350)
	# cv2.imshow("First", first)


	curTime = time.time()
	diffTime = curTime - prevTime
	prevTime = curTime
	total_time += diffTime
	if total_time > 1:
		# print('FPS:',num_frames, total_time)
		total_time = 0
		num_frames = 0

	code = cv2.waitKeyEx(1)
	if code == ord('q'):
		break

cv2.destroyAllWindows()