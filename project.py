import numpy as np
import math
import matplotlib.pyplot as plt
# %matplotlib inline

np.set_printoptions(formatter={'float': lambda x: "{0:0.2f}".format(x)})

f, cx, cy = 512, 256, 256
K = np.array([[f, 0, cx],[0, f, cy],[0, 0, 1]])

N = 8
P_M = np.array([[1, -1, 1, -1, 1, -1, 1, -1], 
				[1, 1, -1, -1, 1, 1, -1, -1], 
				[1, 1, 1, 1, -1, -1, -1, -1], 
				[1, 1, 1, 1, 1, 1, 1, 1	]	])

rdx, rdy, rdz = 0, 0, 0
tx, ty, tz = 0, 6, 6

def degToRad(dx,dy,dz)
	return ax, ay, az = (rdx)*math.pi/180, (rdy)*math.pi/180, (rdz)*math.pi/180


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
	points[0,:] = np.divide(points[0,:], points[2,:])
	points[1,:] = np.divide(points[1,:], points[2,:])
	points[2,:] = np.divide(points[2,:], points[2,:])
	return points



ax, ay, az = degToRad(rdx, rdy, rdz)
R_tot = rotXYZ(ax, ay, az)
T = transXYZ(tx, ty, tz)
M_ext = np.column_stack((R_tot, T))
p = project(K, M_ext, P_M)
p = unscaleH2D(p)


A = np.zeros((N*2, 12))

for i in range(N):
	X, Y, Z = P_M[0,i], P_M[1,i], P_M[2,i]
	x, y = p[0, i], p[1,i]
	# print(X,Y,Z,x,y)
	A[i*2, :] = np.array([X, Y, Z, 0, 0, 0, -x*X, -x*Y, -x*Z, 1, 0, -x])
	A[i*2+1, :] = np.array([0, 0, 0, X, Y, Z, -y*X, -y*Y, -y*Z, 0, 1, -y])

# print(A)

U, D, V = np.linalg.svd(A, full_matrices=False)

x = V[-1,:]

M = np.array([	[x[0],x[1],x[2],x[9] ],
				[x[3],x[4],x[5],x[10]],
				[x[6],x[7],x[8],x[11]]	])

print(M)
R_est = M[0:3,0:3]
T_est = M[:,3]

print(R_est)
print(T_est)
# print(U@np.diag(D)@V)
# print(np.diag(D))
# print(V)

# print(p)
# print(M_ext)
# print(np.eye(3) @ K)

plt.plot(p[0,:], p[1,:], 'ro')
plt.title('Projection')
plt.xlabel('X')
plt.ylabel('Y')
plt.show()



