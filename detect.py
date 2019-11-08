import cv2
import numpy as np


img = cv2.imread("Chess.JPG")


rows, cols, channels = img.shape

print("Image Shape:", img.shape)

H = np.array(	[	[1, -0.8, 0],
					[0.8, 0.1, 0],
					[0, 0, 1]])

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

print("Before Warp:", gray.shape)

gray = cv2.warpPerspective(gray, H, (rows, cols)).T

print("After Warp:", gray.shape)

img = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)


u, v = 20, 20
radius = 10
red = (255, 10, 10)
blue = (10, 255, 10)
thickness = 1

fr = 10

for i in range(u, rows-u, 3):
	for j in range(v, cols-v,3):
		b, w, prev, cur, grad = 0, 0, 0, 0, 0
		for t in range(0, 360, 45):
			tdeg = t/180*np.pi
			nx, ny = int(i+fr*np.sin(tdeg)),int(j+fr*np.cos(tdeg))
			
			# print(i,int(fr*np.sin(tdeg)), "\t\t\t", j, int(fr*np.cos(tdeg)), "\t\t\t",t)
			
			
			if gray[nx][ny] < 128:
				b += 1
				cur = 0
			if gray[nx][ny] > 128:
				w += 1
				cur = 1
			if cur != prev:
				grad += 1
			prev = cur
			# print(int(i+fr*np.sin(tdeg)), "\t\t\t", int(j+fr*np.cos(tdeg)), "\t\t\t",t)
			# img = cv2.circle(img, (int(j+fr*np.cos(tdeg)),int(i+fr*np.sin(tdeg))), 1, (255, 255, 255 ), 1)
		if grad == 4:
			img = cv2.circle(img, (int(j),int(i)), 10, (90, 180, 255 ), 2)

# if gray[i-u][j-v] > 128 and gray[i-u][j+v] < 128 and gray[i+u][j+v] > 128 and gray[i+u][j-v] < 128:
# 	# print("BLUE:", i,j)
# 	img = cv2.circle(img, (j,i), radius, blue, thickness)


cv2.imshow("Gray", img)

cv2.waitKey(0)
cv2.destroyAllWindows()