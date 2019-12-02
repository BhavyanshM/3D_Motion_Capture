import sys
import numpy as np
import time
import math
import matplotlib.pyplot as plt

from mpl_toolkits import mplot3d

import keras.backend as K
from keras.models import Sequential
from keras.layers import Dense

import tensorflow as tf

np.set_printoptions(threshold=sys.maxsize)

cx, cy = 192, 256

def cameraMatrix(f, cx, cy):
	return np.array([[f, 0, cx],[0, f, cy],[0, 0, 1]])


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

def transXYZ(bx, by, bz):
	return np.array([bx, by, bz])

def project(K, M, Points):
	return K @ M @ Points

def unscaleH2D(points):
	# points[0,:] = np.divide(points[0,:], points[2,:])
	# points[1,:] = np.divide(points[1,:], points[2,:])
	# points[2,:] = np.divide(points[2,:], points[2,:])
	points[0] = np.divide(points[0], points[2])
	points[1] = np.divide(points[1], points[2])
	points[2] = np.divide(points[2], points[2])
	return points

def project2d(corresp, out):
	"""Convert 3-D points to 2-D by projecting onto images."""
	print(out[0])
	# X, Y, Z, rx, ry, rz, tx, ty, tz, f = (out[0], out[1], 
	# 									 out[2], out[3], 
	# 									 out[4], out[5], 
	# 									 out[6], out[7], 
	# 									 out[8], out[9])
	X, Y, Z, rx, ry, rz, tx, ty, tz, f = 0,2.0,3.0,4,5,6,7,8,9,10
	p3d = np.array([X, Y, Z, 1]).T
	K = cameraMatrix(f, cx, cy)
	R_tot = rotXYZ(rx, ry, rz)
	T = transXYZ(tx, ty, tz)
	M_ext = np.column_stack((R_tot, T))
	p = project(K, M_ext, p3d)
	p = unscaleH2D(p)
	z = unscaleH2D(np.array([X,Y,Z]))
	proj_corresp = np.hstack((z[:2], p[:2]))
	return proj_corresp

def custom_loss(corresp):
	def net_loss(y_true, y_pred):
		# temp = np.hstack((corresp, corresp))
		# temp = np.hstack((temp, corresp))
		# proj_point = project2d(corresp, y_pred)
		# a = tf.constant([[1],[0],[2],[-3]], dtype=tf.float32)
		# x = np.zeros()
		ind1 = [[0,1],]*10000
		ind2 = [[0,2],]*10000
		a = tf.gather_nd(corresp, ind1)
		b = tf.gather_nd(y_pred, ind2)
		print(a,b)
		proj_point = tf.subtract(a, b)
		return K.sum(K.abs(proj_point))

	return net_loss


def build_model():
	model = Sequential()
	model.add(Dense(16, input_dim=4, activation='relu'))
	model.add(Dense(64, activation='relu'))
	model.add(Dense(64, activation='sigmoid'))
	model.add(Dense(32, activation='relu'))
	model.add(Dense(8, activation='linear'))
	return model


if __name__ == "__main__":
	ds = np.loadtxt('corresp_10000.csv', delimiter=',')


	# print(ds)

	model = build_model()
	cor = tf.constant(ds, dtype=tf.float32)
	model.compile(loss=custom_loss(cor), optimizer='adam')
	print(ds.dtype)
	model.fit(ds, ds, batch_size=1, epochs=3)

	x = np.array([[2,3,4,5]])
	print("Predict({})".format(x), model.predict(x))
