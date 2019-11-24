import cv2
import numpy as np
import time
import math
import matplotlib.pyplot as plt
from scipy.optimize import least_squares

cap1 = cv2.VideoCapture(0)
cap2 = cv2.VideoCapture(1)
cap1.set(cv2.CAP_PROP_FRAME_WIDTH, 512)
cap1.set(cv2.CAP_PROP_FRAME_HEIGHT, 384)
cap2.set(cv2.CAP_PROP_FRAME_WIDTH, 512)
cap2.set(cv2.CAP_PROP_FRAME_HEIGHT, 384)

n_cams, n_points, n_obs = 2, 200, 400

init_params = np.random.random(n_cams*9 + n_points*3)
points_2d = np.ones((n_obs, 2), dtype=int)
cam_indices = np.zeros(n_obs, dtype=int)
point_indices = np.zeros(n_obs, dtype=int)

def cameraMatrix(f, cx, cy):
	return np.array([[f, 0, cx],[0, f, cy],[0, 0, 1]])

def cartesian2d(points):
	points[:,0] = np.divide(points[:,0], points[:,2])
	points[:,1] = np.divide(points[:,1], points[:,2])
	points[:,2] = np.divide(points[:,2], points[:,2])
	return points

def degToRad(dx,dy,dz):
	return (rdx)*math.pi/180, (rdy)*math.pi/180, (rdz)*math.pi/180


def rotX(ax):
	return np.array([	[1,			0,			0		],
					[0,			math.cos(ax), 	-math.sin(ax)],
					[0, 		math.sin(ax), 	math.cos(ax)	]	])

def rotY(ay):
	return np.array([	[math.cos(ay), 	0, 		math.sin(ay)],
					[0, 			1, 		0],
					[-math.sin(ay), 0, 	math.cos(ay)]	])

def rotZ(az):
	return np.array([	[math.cos(az), 	-math.sin(az), 	0],
					[math.sin(az), 	math.cos(az), 	0],
					[0, 			0, 				1]			])

def rotXYZ(ax, ay, az):
	return rotZ(az) @ rotY(ay) @ rotX(ax)

def rotate(points, rots):
	rot_points = np.zeros_like(points)
	for i in range(len(points)):
		rot_points[i] = points[i] @ rotXYZ(rots[i,0], rots[i,1], rots[i,2])
	return rot_points


def project2d(points, camera_params):
	"""Convert 3-D points to 2-D by projecting onto images."""
	points_proj = rotate(points, camera_params[:, :3])
	points_proj += camera_params[:, 3:6]
	points_proj = points_proj[:, :2] / points_proj[:, 2, np.newaxis]
	f = camera_params[:, 6]
	k1 = camera_params[:, 7]
	k2 = camera_params[:, 8]
	n = np.sum(points_proj**2, axis=1)
	r = 1 + k1 * n + k2 * n**2
	points_proj *= (r * f)[:, np.newaxis]
	return points_proj

def residuals(params, n_cams, n_points, cam_indices, point_indices, points_2d):
	"""Generate residuals between observed and reprojected points"""
	camera_params = params[:n_cams * 9].reshape((n_cams, 9))
	points_3d = params[n_cams * 9:].reshape((n_points, 3))
	points_proj = project2d(points_3d[point_indices], camera_params[cam_indices])
	return (points_proj - points_2d).ravel()


correspondences = []
px1, px2, py1, py2 = 0, 0, 0, 0

while True:
	ret1, img1 = cap1.read()
	assert ret1
	ret2, img2 = cap2.read()
	assert ret2

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

	mask2 = cv2.inRange(hsv2, lower_red, upper_red)
	M = cv2.moments(mask2)
	cX2 = int(M["m10"] / M["m00"]) if M["m00"] != 0 else px2
	cY2 = int(M["m01"] / M["m00"]) if M["m00"] != 0 else py2
	cv2.line(mask2, (cX2, cY2 + 20), (cX2, cY2 - 20), 255, 1)
	cv2.line(mask2, (cX2 + 20, cY2), (cX2 - 20, cY2), 255, 1)

	corresp = px1, py1, px2, py2 = cX1, cY1, cX2, cY2
	canvas1 = np.zeros_like(img1, np.uint8)
	canvas2 = np.zeros_like(img2, np.uint8)

	if np.sum(mask1) > 25000 and np.sum(mask2) > 25000 and len(correspondences) < n_points:
		pn = len(correspondences)
		points_2d[pn*2, :] = np.array([px1, py1])
		points_2d[pn*2 + 1, :] = np.array([px2, py2])
		point_indices[pn*2], point_indices[pn*2 + 1] = pn, pn
		cam_indices[pn*2], cam_indices[pn*2 + 1] = 0, 1
		correspondences.append(corresp)


		for c in correspondences:
			x1, y1, x2, y2 = c
			c1 = (x1/(x1+y1)*255, y1/(x1+y1)*255, 255)
			c2 = (x2/(x2+y2)*255, y2/(x2+y2)*255, 255)
			cv2.circle(canvas1, (x1,y1), 5, c1, -1)
			cv2.circle(canvas2, (x2,y2), 5, c1, -1)		
		

		cv2.namedWindow("Image1", cv2.WINDOW_NORMAL)
		cv2.namedWindow("Image2", cv2.WINDOW_NORMAL)
		cv2.resizeWindow("Image1", 1600, 900)
		cv2.resizeWindow("Image2", 1600, 900)
		cv2.imshow('Image1', canvas1)
		cv2.imshow('Image2', canvas2)

		print(pn)

	if(len(correspondences) == n_points):
		break
	code = cv2.waitKeyEx(1)
	if code == ord('q'):
		break

# res = residuals(init_params, n_cams, n_points, cam_indices, point_indices, points_2d)

cv2.destroyAllWindows()
print("Optimization started.")

t0 = time.time()
res = least_squares(residuals, init_params, verbose=2, ftol=1e-2, method='trf',args=(n_cams, n_points, cam_indices, point_indices, points_2d))
t1 = time.time()

print(res.x)
print("Optimization took {0:.0f} seconds".format(t1 - t0))

plt.rcParams.update({'font.size': 14})
fig = plt.figure(figsize=(35,15))
plt.plot(res.fun)
plt.show()


