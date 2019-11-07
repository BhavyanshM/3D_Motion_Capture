import cv2
import numpy as np


img = cv2.imread("Chess.JPG")

rows, cols, channels = img.shape

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)



u, v = 5, 5
radius = 10
red = (255, 10, 10)
blue = (10, 255, 10)
thickness = 1

for i in range(rows-u):
	for j in range(cols-v):
		if gray[i-u][j-v] < 128 and gray[i-u][j+v] > 128 and gray[i+u][j+v] < 128 and gray[i+u][j-v] > 128:
			print("RED:", i,j)
			img = cv2.circle(img, (i,j), radius, red, thickness)

		if gray[i-u][j-v] > 128 and gray[i-u][j+v] < 128 and gray[i+u][j+v] > 128 and gray[i+u][j-v] < 128:
			print("BLUE:", i,j)
			img = cv2.circle(img, (i,j), radius, blue, thickness)

	if cv2.waitKey(1) & 0xFF == ord('q'):
		break

cv2.imshow("Image", img)

cv2.waitKey(0)
cv2.destroyAllWindows()