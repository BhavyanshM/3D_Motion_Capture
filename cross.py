import cv2
import numpy as np


cap = cv2.VideoCapture(1)

	# img = cv2.imread("Chess.JPG")

while True:
	ret, img = cap.read()

	# img = img[200:650,200:650,:]

	rows, cols, channels = img.shape
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	# ret, gray = cv2.threshold(grayRange, 127, 255, cv2.THRESH_BINARY)



	u, v = 20, 20
	radius = 10
	red = (255, 10, 10)
	blue = (10, 255, 10)
	thickness = 1

	fr = 10


	for i in range(u, rows-u, 10):
		for j in range(v, cols-v,10):
			white, prev, cur, grad = 0, 0, 0, 0
			for t in range(0, 360, 72):
				tdeg = t/180*np.pi
				nx, ny = int(i+fr*np.sin(tdeg)),int(j+fr*np.cos(tdeg))
				
				if gray[nx][ny] < 128:
					cur = 0
				if gray[nx][ny] > 128:
					cur = 1
				if cur != prev:
					grad += 1
				prev = cur
			if grad == 4:
				img = cv2.circle(img, (int(j),int(i)), fr, (90, 180, 255 ), 2)


	cv2.imshow("Result", img)

	code = cv2.waitKeyEx(1)
	if code == ord('q'):
		break

cv2.destroyAllWindows()