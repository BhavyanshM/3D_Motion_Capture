import numpy as np
import math
import matplotlib.pyplot as plt
# %matplotlib inline

np.set_printoptions(formatter={'float': lambda x: "{0:0.2f}".format(x)})

f, cx, cy = 512, 256, 256
K = np.array([[f, 0, cx],[0, f, cy],[0, 0, 1]])

N = 8
P_M = np.array([	[1, -1, 1, -1, 1, -1, 1, -1], 
		[1, 1, -1, -1, 1, 1, -1, -1], 
		[1, 1, 1, 1, -1, -1, -1, -1], 
		[1, 1, 1, 1, 1, 1, 1, 1	]		])

ax, ay, az = 0, 20*math.pi/180, -30*math.pi/180
tx, ty, tz = 0, 0, 6
Rx = np.array([	[1,			0,			0		],
				[0,			math.cos(ax), 	-math.sin(ax)],
				[0, 		math.sin(ax), 	math.cos(ax)	]	])

Ry = np.array([	[math.cos(ay), 	0, 		math.sin(ay)],
				[0, 			1, 		0],
				[-math.sin(ay), 0, 	math.cos(ay)]	])

Rz = np.array([	[math.cos(az), 	-math.sin(az), 	0],
				[math.sin(az), 	math.cos(az), 	0],
				[0, 			0, 				1]			])


Rtrue = Rz @ Ry @ Rx
T = np.array([tx, ty, tz])
M_ext = np.column_stack((Rtrue, T))

p = K @ M_ext @ P_M

p[0,:] = np.divide(p[0,:], p[2,:])
p[1,:] = np.divide(p[1,:], p[2,:])
p[2,:] = np.divide(p[2,:], p[2,:])

print(p)
# print(M_ext)
# print(np.eye(3) @ K)

plt.figure(figsize=(512,512))
plt.plot(p[0,:], p[1,:])
plt.title('Projection')
plt.xlabel('X')
plt.ylabel('Y')
plt.show()

