import cv2
import numpy as np


img = 255*np.ones((1080, 1920))


u, v = 20, 20
radius = 10
red = (255, 10, 10)
blue = (10, 255, 10)
thickness = 1

pos_scale = 200
size_scale = 10

i, j = 500, 500

for r in range(5):
	for t in range(0, 360, 45):
		tdeg = ((r%2 + 1)*45+t)/180*np.pi
		nx, ny = i+int(  pos_scale*(r)*np.cos(tdeg)  ) ,j+int(  pos_scale*(r)*np.sin(tdeg)  )
		print(nx, ny)
		img = cv2.circle(img, ( nx , ny ), (5-r)*size_scale, (0, 0, 0), -1)

# if gray[i-u][j-v] > 128 and gray[i-u][j+v] < 128 and gray[i+u][j+v] > 128 and gray[i+u][j-v] < 128:
# 	# print("BLUE:", i,j)
# 	img = cv2.circle(img, (j,i), radius, blue, thickness)


cv2.imshow("Gray", img)

cv2.waitKey(0)
cv2.destroyAllWindows()