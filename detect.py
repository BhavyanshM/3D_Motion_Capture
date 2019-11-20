import cv2
import numpy as np


cap = cv2.VideoCapture(1)

	# img = cv2.imread("Chess.JPG")

while True:
	ret, img = cap.read()

	# img = img[200:650,200:650,:]

	rows, cols, channels = img.shape

	# print("Image Shape:", img.shape)

	H = np.array(	[	[1, -0.5, 0],
						[0.5, 0.4, 0],
						[0, 0, 1]])

	grayRange = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	ret, gray = cv2.threshold(grayRange, 127, 255, cv2.THRESH_BINARY)

	# print("Before Warp:", gray.shape)

	# gray = cv2.warpPerspective(gray, H, (rows, cols)).T

	# print("After Warp:", gray.shape)



	u, v = 20, 20
	radius = 10
	red = (255, 10, 10)
	blue = (10, 255, 10)
	thickness = 1

	fr = 6

	# gaussian = np.array([[1,5,1],[5,10,5],[1,5,1]])/34
	# gauss = cv2.filter2D(gray, -1, gaussian)

	grad_x = np.array([[1,0,-1],[1,0,-1],[1,0,-1]])
	grad_y = np.array([[1,0,1],[0,0,0],[-1,0,-1]])
	grad_xu = np.array([[-1,0,1],[-1,0,1],[-1,0,1]])
	grad_yu = np.array([[-1,0,-1],[0,0,0],[1,0,1]])
	gx = cv2.filter2D(gray, -1, grad_y)
	gy = cv2.filter2D(gray, -1, grad_x)
	gxu = cv2.filter2D(gray, -1, grad_xu)
	gyu = cv2.filter2D(gray, -1, grad_yu)


	gradColor = np.ones([gray.shape[0],gray.shape[1],3])
	gradGreen = gradColor.copy()
	gradGreen[:,:,0] *= 64/255.0
	gradRed = gradGreen.copy()
	gradRed[:,:,1] *= 128/255.0
	gradBlue = gradRed.copy()
	gradBlue[:,:,2] *= 100/255.0
	gradWhite = gradBlue.copy()
	gradWhite[:,:,0] *= 0


	grad = gx + gy + gxu + gyu
	resW = cv2.bitwise_and(gradWhite,gradWhite,mask = gx)
	resB = cv2.bitwise_and(gradBlue,gradBlue,mask = gy)
	resG = cv2.bitwise_and(gradGreen,gradGreen,mask = gxu)
	resR = cv2.bitwise_and(gradRed,gradRed,mask = gyu)
	res = resW + resB + resG + resR
	print("grad")

	# for i in range(u, rows-u, 8):
	# 	for j in range(v, cols-v,8):
	# 		white, prev, cur, grad = 0, 0, 0, 0
	# 		for t in range(0, 360, 45):
	# 			tdeg = t/180*np.pi
	# 			nx, ny = int(i+fr*np.sin(tdeg)),int(j+fr*np.cos(tdeg))
				
	# 			if gray[nx][ny] < 128:
	# 				cur = 0
	# 			if gray[nx][ny] > 128:
	# 				cur = 1
	# 			if cur != prev:
	# 				grad += 1
	# 			prev = cur
	# 		if grad == 4:
	# 			newImg = cv2.circle(newImg, (int(j),int(i)), fr, (90, 180, 255 ), 2)


	cv2.imshow("Filter2D", grad)

	res = cv2.add(resW, resR)
	# res = cv2.add(res, resG)
	# res = cv2.add(res, resB)
	cv2.imshow("Result", res)

	code = cv2.waitKeyEx(100)
	if code == ord('q'):
		break

cv2.destroyAllWindows()