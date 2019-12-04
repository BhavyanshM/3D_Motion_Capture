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

def project2d(pred):
	X, Y, Z, rx, ry, rz, tx, ty, tz, f = 	tf.gather_nd(pred, [[0,0]]),\
											tf.gather_nd(pred, [[0,1]]),\
											tf.gather_nd(pred, [[0,2]]),\
											tf.gather_nd(pred, [[0,3]]),\
											tf.gather_nd(pred, [[0,4]]),\
											tf.gather_nd(pred, [[0,5]]),\
											tf.gather_nd(pred, [[0,6]]),\
											tf.gather_nd(pred, [[0,7]]),\
											tf.gather_nd(pred, [[0,8]]),\
											tf.gather_nd(pred, [[0,9]])	


	print("X:",X)

	S = tf.stack([X,Y,Z,tf.constant([1], dtype=tf.float32)])
	print("S:",S)

	proj_corresp = pred

	xc = tf.constant([cx], dtype=tf.float32)
	yc = tf.constant([cy], dtype=tf.float32)

	zr = tf.constant([0], dtype=tf.float32)
	on = tf.constant([1], dtype=tf.float32)

	K = tf.reshape(tf.stack([[f, zr, xc],[zr, f, yc],[zr, zr, on]]), [3,3])

	print("K:",K)


	Rx = tf.reshape(tf.stack([[on, zr, zr],[zr, tf.math.cos(rx), -tf.math.sin(rx)],[zr, tf.math.sin(rx), tf.math.cos(rx)]]), [3,3])
	print("Rx:",Rx)
	Ry = tf.reshape(tf.stack([[tf.math.cos(ry), zr, tf.math.sin(ry)],[zr, on, zr],[-tf.math.sin(ry), zr, tf.math.cos(ry)]]), [3,3])
	print("Ry:",Ry)
	Rz = tf.reshape(tf.stack([[tf.math.cos(rz), -tf.math.sin(rz), zr],[tf.math.sin(rz), tf.math.cos(rz), zr],[zr, zr, on]]), [3,3])
	print("Rz:",Rz)

	R = tf.matmul(Rz, tf.matmul(Ry,Rx))
	print("R:",R)

	T = tf.stack([tx,ty,tz])
	print("T:",T)

	M = tf.concat([R,T], axis=1)
	print("M:",M)

	P = tf.matmul(K,tf.matmul(M,S))
	print("P:",P)

	# K = cameraMatrix(f, cx, cy)
	# R_tot = rotXYZ(rx, ry, rz)
	# T = transXYZ(tx, ty, tz)
	# M_ext = np.column_stack((R_tot, T))
	# p = project(K, M_ext, p3d)
	# p = unscaleH2D(p)
	# z = unscaleH2D(np.array([X,Y,Z]))
	
	# p3d = np.array([X, Y, Z, 1]).T
	# proj_corresp = np.hstack((z[:2], p[:2]))
	

	return proj_corresp

def custom_loss(corresp):
	def net_loss(y_true, y_pred):
		proj_point = project2d(y_pred)
		return K.sum(K.abs(y_pred))

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
